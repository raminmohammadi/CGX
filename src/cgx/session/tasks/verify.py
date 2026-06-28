

"""VERIFY executor: run impacted tests after an APPLY (or BOOTSTRAP_ENV).

Reads the upstream ``APPLIED_CHANGES`` artifact to learn which files
the apply step wrote, then delegates to
:func:`cgx.codegen.test_runner.run_tests_on_disk` which selects impacted
tests via import-graph heuristics and runs pytest against the live
working tree. The result is persisted as a ``VERIFY_REPORT`` artifact.

When a ``build_artifact_id`` is present (greenfield path), VERIFY picks
the project venv's python from the BUILD_REPORT and feeds it to the
pytest subprocess so the tests run against the bootstrapped environment
rather than the host interpreter.

The outcome of every pytest run is classified into a stable enum so the
UI -- and any downstream retry logic -- can distinguish a real test
failure (``assertions_failed``) from a setup problem
(``collection_error`` / ``pytest_missing``) without re-parsing stderr.

VERIFY is intentionally a terminal kind in the session router -- it
does not spawn a successor. The caller decides what to do next (post a
new objective, roll back via backup_dir, etc.) once the report is
visible.
"""

from __future__ import annotations

import logging
import os
import shlex
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from cgx.session.models import (
    Artifact,
    ArtifactKind,
    TaskKind,
    TaskNode,
)
from cgx.session.tasks.base import (
    ExecutorDeps,
    ExecutorResult,
    register_executor,
)

logger = logging.getLogger(__name__)


# Outcome enum surfaced in VERIFY_REPORT.outcome. ``passed`` is the
# happy path; ``assertions_failed`` is a real logic failure; everything
# else is a setup / environment problem that the user (or a future
# retry loop) should resolve before the tests can be trusted.
VERIFY_OUTCOMES: Tuple[str, ...] = (
    "passed",
    "assertions_failed",
    "collection_error",
    "no_tests_collected",
    "timeout",
    "pytest_missing",
    "skipped",
)


@register_executor(TaskKind.VERIFY)
def run_verify(task: TaskNode, deps: ExecutorDeps) -> ExecutorResult:
    """Run impacted tests for the just-applied changes."""
    if not deps.project_root:
        return ExecutorResult(failure="VERIFY requires project_root in deps")
    if deps.store is None:
        return ExecutorResult(
            failure="VERIFY requires a session store in deps")

    changed_files = _resolve_changed_files(task, deps)
    python_exe = _resolve_python_exe(task, deps)
    build_artifact_id = str(
        task.inputs.get("build_artifact_id") or "").strip() or None

    # Lazy import: test_runner pulls subprocess + pytest discovery.
    from cgx.codegen.test_runner import run_tests_on_disk

    timeout = float(task.inputs.get("timeout_seconds") or 180.0)
    # Allocate a junitxml sink so the classifier can consume structured
    # failure records instead of re-parsing pytest's stdout. ``-rN`` keeps
    # the short summary off (we already have the XML); ``--tb=long`` makes
    # the rendered traceback usable when a human opens the artifact.
    junit_fd, junit_path = tempfile.mkstemp(prefix="cgx_junit_", suffix=".xml")
    os.close(junit_fd)
    extra_pytest_args = (
        "-q", "--no-header", "-rN", "--tb=long",
        f"--junitxml={junit_path}",
    )
    try:
        outcome = run_tests_on_disk(
            deps.project_root, changed_files,
            timeout_seconds=timeout,
            python_exe=python_exe,
            extra_pytest_args=extra_pytest_args,
        )
    except Exception as exc:
        logger.exception("VERIFY: run_tests_on_disk crashed")
        _unlink_quiet(junit_path)
        return ExecutorResult(
            failure=f"verify failed: {type(exc).__name__}: {exc}")

    failures = _parse_junit_failures(junit_path)
    _unlink_quiet(junit_path)

    tests_passed = bool(outcome.ran and outcome.returncode == 0)
    verify_outcome = _classify_outcome(outcome)
    mode = str(task.inputs.get("mode") or "explore").strip() or "explore"
    reproduce_cmd = _build_reproduce_cmd(
        deps.project_root, python_exe, list(outcome.tests_selected))
    content = {
        "apply_artifact_id": task.inputs.get("apply_artifact_id"),
        "plan_artifact_id": task.inputs.get("plan_artifact_id"),
        "scaffold_artifact_id": task.inputs.get("scaffold_artifact_id"),
        "build_artifact_id": build_artifact_id,
        "python_exe": python_exe,
        "mode": mode,
        "changed_files": list(changed_files),
        "ran": bool(outcome.ran),
        "tests_passed": tests_passed,
        "outcome": verify_outcome,
        "returncode": int(outcome.returncode),
        "tests_selected": list(outcome.tests_selected),
        "stdout": outcome.stdout or "",
        "stderr": outcome.stderr or "",
        "skipped_reason": outcome.skipped_reason,
        "reproduce_cmd": reproduce_cmd,
        "failures": failures,
    }
    # Pre-compute the repair-loop signature here (rather than in the
    # router) so the deterministic Router stays IO-free; the value is
    # stable across re-runs of the same failure.
    from cgx.session.repair.classify import failure_signature
    sig = failure_signature(content)
    content["failure_signature"] = sig
    artifact = Artifact.new(
        session_id=task.session_id,
        produced_by_task_id=task.task_id,
        kind=ArtifactKind.VERIFY_REPORT,
        content=content,
    )
    return ExecutorResult(
        outputs={
            "verify_artifact_id": artifact.artifact_id,
            "ran": bool(outcome.ran),
            "tests_passed": tests_passed,
            "outcome": verify_outcome,
            "tests_selected_count": len(outcome.tests_selected),
            "failure_signature": sig,
            "returncode": int(outcome.returncode),
        },
        artifact=artifact,
    )


# --------------------- helpers ---------------------

def _resolve_changed_files(task: TaskNode, deps: ExecutorDeps) -> List[str]:
    """Return the list of changed files for impact-based test selection.

    Prefers the explicit ``changed_files`` input (if a caller set one);
    otherwise reads ``APPLIED_CHANGES.applied_files`` via the upstream
    apply artifact. Returns an empty list when neither is available --
    ``run_tests_on_disk`` then falls back to discovering all tests.
    """
    explicit = task.inputs.get("changed_files")
    if isinstance(explicit, list) and explicit:
        return [str(p) for p in explicit if str(p).strip()]

    apply_artifact_id = str(task.inputs.get("apply_artifact_id") or "").strip()
    if not apply_artifact_id:
        return []
    artifact = deps.store.get_artifact(apply_artifact_id)
    if artifact is None or artifact.kind is not ArtifactKind.APPLIED_CHANGES:
        return []
    applied: Any = (artifact.content or {}).get("applied_files") or []
    if not isinstance(applied, list):
        return []
    return [str(p) for p in applied if str(p).strip()]


def _resolve_python_exe(task: TaskNode,
                        deps: ExecutorDeps) -> Optional[str]:
    """Return the venv python from the upstream BUILD_REPORT, if any.

    Only greenfield sessions carry a ``build_artifact_id`` (the router
    inserts BOOTSTRAP_ENV between APPLY and VERIFY for that mode);
    explore-mode VERIFY falls back to ``test_runner._project_python_exe``
    which auto-detects an existing ``.venv``.
    """
    build_artifact_id = str(task.inputs.get("build_artifact_id") or "").strip()
    if not build_artifact_id:
        return None
    artifact = deps.store.get_artifact(build_artifact_id)
    if artifact is None or artifact.kind is not ArtifactKind.BUILD_REPORT:
        return None
    py = (artifact.content or {}).get("python_exe")
    return str(py) if isinstance(py, str) and py else None


def _classify_outcome(outcome: Any) -> str:
    """Map a :class:`TestRunOutcome` to a stable VERIFY_OUTCOMES token.

    Pytest's documented exit codes drive the classification:

    * 0 -> passed
    * 1 -> assertions_failed (one or more tests failed)
    * 2 -> collection_error (usage / import error during collection)
    * 3 -> collection_error (internal pytest error)
    * 4 -> collection_error (pytest CLI usage error)
    * 5 -> no_tests_collected
    * 124 -> timeout (set by ``run_pytest_paths`` on ``TimeoutExpired``)

    A ``skipped_reason`` of ``"pytest not installed"`` is mapped to
    ``pytest_missing`` so the UI can offer a targeted remediation
    instead of pointing the user at the test code.
    """
    if not outcome.ran:
        if (outcome.skipped_reason or "").lower().startswith(
                "pytest not installed"):
            return "pytest_missing"
        return "skipped"
    rc = int(outcome.returncode)
    if rc == 0:
        return "passed"
    if rc == 1:
        return "assertions_failed"
    if rc == 5:
        return "no_tests_collected"
    if rc == 124:
        return "timeout"
    return "collection_error"


def _build_reproduce_cmd(project_root: str,
                         python_exe: Optional[str],
                         tests_selected: Sequence[str]) -> Optional[str]:
    """Return a single shell line that re-runs the exact pytest invocation.

    Mirrors the cwd, interpreter and extra args used by
    :func:`cgx.codegen.test_runner.run_pytest_paths` -- including the
    auto-detected venv python when the caller didn't pin one -- so a
    developer can paste the line into a terminal and reproduce the
    failure verbatim. Returns ``None`` when no tests were selected
    (nothing meaningful to reproduce).
    """
    if not tests_selected:
        return None
    # Late import to avoid a top-level dep cycle through test_runner's
    # subprocess + pytest discovery.
    from cgx.codegen.test_runner import _project_python_exe
    root = Path(project_root).resolve()
    py = python_exe or _project_python_exe(root)
    rel_tests: List[str] = []
    for t in tests_selected:
        p = Path(t)
        try:
            rel_tests.append(str(p.resolve().relative_to(root)))
        except (ValueError, OSError):
            rel_tests.append(str(p))
    parts = [shlex.quote(py), "-m", "pytest", "-q", "--no-header",
             *(shlex.quote(t) for t in rel_tests)]
    return f"cd {shlex.quote(str(root))} && " + " ".join(parts)



def _parse_junit_failures(junit_path: str) -> List[Dict[str, str]]:
    """Return structured failure / error records from a JUnit XML file.

    Pytest writes one ``<testcase>`` element per collected test under
    ``--junitxml``; failures carry a nested ``<failure>`` (assertion
    failed) and errors a ``<error>`` (collection / setup / teardown
    crash). For each, we extract the ``classname.name`` nodeid, the
    type / message attributes, and the captured traceback text so the
    classifier can pattern-match on import errors, name errors etc.
    without re-parsing the human-oriented stdout.

    Returns an empty list when the file is missing, empty, or unparsable
    -- VERIFY still records ``stdout`` / ``stderr`` for the human view.
    """
    try:
        if not junit_path or not os.path.isfile(junit_path):
            return []
        if os.path.getsize(junit_path) == 0:
            return []
        tree = ET.parse(junit_path)
    except (ET.ParseError, OSError) as exc:
        logger.debug("VERIFY: junitxml parse failed: %s", exc)
        return []
    failures: List[Dict[str, str]] = []
    for case in tree.iter("testcase"):
        for tag in ("failure", "error"):
            node = case.find(tag)
            if node is None:
                continue
            classname = case.get("classname") or ""
            name = case.get("name") or ""
            if classname and name:
                nodeid = f"{classname}::{name}"
            else:
                nodeid = classname or name or ""
            failures.append({
                "nodeid": nodeid,
                "kind": tag,
                "type": node.get("type") or "",
                "message": node.get("message") or "",
                "traceback": (node.text or "").strip(),
            })
    return failures


def _unlink_quiet(path: str) -> None:
    """Best-effort unlink; swallow ENOENT and permission errors."""
    try:
        os.unlink(path)
    except OSError:
        pass
