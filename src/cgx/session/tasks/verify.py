

"""VERIFY executor: run impacted tests after an APPLY (or BOOTSTRAP_ENV).

Reads the upstream ``APPLIED_CHANGES`` artifact to learn which files
the apply step wrote, then runs every language runner the project
matches via :mod:`cgx.codegen.test_runners`. The Python stack keeps the
junit-backed ``run_tests_on_disk`` path (so the classifier still sees
structured failures); any other detected stack (npm, ...) runs through
the registry so a JS/TS project gets a real build/test signal instead
of a silent "no tests -> skipped -> success". The merged result is
persisted as a ``VERIFY_REPORT`` artifact.

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
from dataclasses import dataclass, field
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
# happy path; ``assertions_failed`` is a real logic failure; ``failed``
# is a non-pytest runner (e.g. an ``npm`` build/test) that exited
# non-zero; ``no_tests`` is a non-pytest runner that only ran a build
# smoke because the project wired up no tests (a passing *build* is not
# a passing *suite*); everything else is a setup / environment problem
# that the user (or a future retry loop) should resolve before the tests
# can be trusted.
VERIFY_OUTCOMES: Tuple[str, ...] = (
    "passed",
    "assertions_failed",
    "failed",
    "no_tests",
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

    timeout = float(task.inputs.get("timeout_seconds") or 180.0)

    # Detect which language runners apply. Python keeps the junit-backed
    # pytest path (so the classifier still sees structured failures);
    # every other detected stack (npm, ...) runs via the registry so a
    # JS/TS project gets a real build/test signal instead of being
    # silently skipped.
    from cgx.codegen.test_runners import detect_test_runners
    try:
        runners = detect_test_runners(deps.project_root)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("VERIFY: runner detection failed: %s", exc)
        runners = []
    run_pytest = (not runners) or any(r.name == "pytest" for r in runners)
    other_runners = [r for r in runners if r.name != "pytest"]

    pytest_outcome = None
    failures: List[Dict[str, str]] = []
    if run_pytest:
        # Lazy import: test_runner pulls subprocess + pytest discovery.
        from cgx.codegen.test_runner import run_tests_on_disk

        # Allocate a junitxml sink so the classifier can consume
        # structured failure records instead of re-parsing pytest's
        # stdout. ``-rN`` keeps the short summary off (we already have
        # the XML); ``--tb=long`` makes the rendered traceback usable
        # when a human opens the artifact.
        junit_fd, junit_path = tempfile.mkstemp(
            prefix="cgx_junit_", suffix=".xml")
        os.close(junit_fd)
        extra_pytest_args = (
            "-q", "--no-header", "-rN", "--tb=long",
            "--ignore=.cgx-backups",
            f"--junitxml={junit_path}",
        )
        try:
            pytest_outcome = run_tests_on_disk(
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

    other_pairs = _run_other_runners(
        other_runners, deps.project_root, changed_files,
        timeout=timeout, python_exe=python_exe)

    combined = _combine_verify_outcomes(pytest_outcome, other_pairs)
    verify_outcome = combined.outcome
    tests_passed = verify_outcome == "passed"
    mode = str(task.inputs.get("mode") or "explore").strip() or "explore"
    reproduce_cmd = _build_reproduce_cmd(
        deps.project_root, python_exe, combined.pytest_tests_selected)
    content = {
        "apply_artifact_id": task.inputs.get("apply_artifact_id"),
        "plan_artifact_id": task.inputs.get("plan_artifact_id"),
        "scaffold_artifact_id": task.inputs.get("scaffold_artifact_id"),
        "build_artifact_id": build_artifact_id,
        "python_exe": python_exe,
        "mode": mode,
        "changed_files": list(changed_files),
        "ran": combined.ran,
        "tests_passed": tests_passed,
        "outcome": verify_outcome,
        "returncode": combined.returncode,
        "tests_selected": list(combined.tests_selected),
        "stdout": combined.stdout,
        "stderr": combined.stderr,
        "skipped_reason": combined.skipped_reason,
        "reproduce_cmd": reproduce_cmd,
        "failures": failures,
        "js_tests_present": combined.js_tests_present,
        "js_tests_ran": combined.js_tests_ran,
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
            "ran": combined.ran,
            "tests_passed": tests_passed,
            "outcome": verify_outcome,
            "tests_selected_count": len(combined.pytest_tests_selected),
            # Progress-ledger counts read by the router's coverage-aware
            # repair budget (#5, see cgx.session.router.
            # _repair_progress_stalled). ``failing_count`` is the primary
            # trend; ``passing_count`` / ``collected_count`` let a round
            # that fixed nothing net-new still count as forward progress
            # when it made MORE tests pass or collected more tests than the
            # previous round. They are only a trustworthy *execution*
            # signal when the suite actually ran to completion; on a
            # non-executing outcome (collection_error / timeout / ...) an
            # empty junit means "nothing ran", not "nothing failed", so
            # :func:`_progress_counts` forces ``passing_count`` to 0 and
            # leaves ``failing_count`` unknown rather than emitting a false
            # "0 failing / N passing" that the router reads as progress.
            **_progress_counts(
                verify_outcome, failures, combined.pytest_tests_selected),
            "failure_signature": sig,
            "returncode": combined.returncode,
            "js_tests_present": combined.js_tests_present,
            "js_tests_ran": combined.js_tests_ran,
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


# Outcomes where pytest actually executed the suite to completion, so a
# junit-derived pass/fail count is a trustworthy, round-over-round
# comparable progress signal. Every other outcome means the suite never
# ran (collection error, timeout, missing pytest, empty collection): a
# junit that enumerates zero failures there is "nothing executed", NOT
# "nothing failed".
_EXECUTED_VERIFY_OUTCOMES: frozenset = frozenset(
    {"passed", "assertions_failed"})


def _progress_counts(
        outcome: str,
        failures: Sequence[Dict[str, str]],
        pytest_selected: Sequence[str]) -> Dict[str, Optional[int]]:
    """Return the junit-derived {failing,collected,passing}_count ledger.

    See :func:`cgx.session.router._repair_progress_stalled`. When the
    suite executed (:data:`_EXECUTED_VERIFY_OUTCOMES`) the counts are the
    plain junit arithmetic. When it did not, ``passing_count`` is 0 (no
    test passed) and ``failing_count`` is the real erroring-module count
    only when junit actually enumerated one (a comparable per-collection
    trend the router trusts for ``collection_error``); an empty junit
    yields ``None`` so the router's progress gate stays inconclusive and
    the signature-flap + REPAIR_BUDGET backstops bound the loop instead of
    a bogus "dropped to 0 failing" that reads as forward progress.
    """
    n_fail = len(failures)
    n_sel = len(pytest_selected)
    if outcome in _EXECUTED_VERIFY_OUTCOMES:
        return {
            "failing_count": n_fail,
            "collected_count": n_sel,
            "passing_count": max(0, n_sel - n_fail),
        }
    return {
        "failing_count": n_fail if n_fail > 0 else None,
        "collected_count": n_sel,
        "passing_count": 0,
    }


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


@dataclass
class _Combined:
    """The single pass/fail signal merged across every language runner."""
    ran: bool
    returncode: int
    outcome: str
    tests_selected: List[str] = field(default_factory=list)
    pytest_tests_selected: List[str] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    skipped_reason: Optional[str] = None
    # Coverage signal for the non-pytest (JS/TS) half of a polyglot repo:
    # ``js_tests_present`` is True when the tree carries scaffolded JS test
    # files, ``js_tests_ran`` when a JS runner actually executed a real
    # suite. Present-but-not-ran is the ses_4cbf963cdc67435a blind spot --
    # exposed here so P2 can fail closed rather than let a passing Python
    # half mask an unrun React suite.
    js_tests_present: bool = False
    js_tests_ran: bool = False


# Severity ordering among the "hard" (definitively-failed) tokens: when
# several runners fail, the highest-severity token is reported; ties are
# broken in favour of the pytest token so the Python repair classifiers
# keep their structured input. ``failed`` is the generic non-pytest
# runner failure (e.g. an npm build/test that exited non-zero).
_HARD_FAILURE_SEVERITY: Dict[str, int] = {
    "collection_error": 4,
    "timeout": 4,
    "assertions_failed": 3,
    "failed": 3,
}


def _classify_other_outcome(outcome: Any) -> str:
    """Map a non-pytest :class:`TestRunOutcome` to a VERIFY token.

    Non-pytest runners (npm, ...) do not follow pytest's exit-code
    contract, so the mapping is coarse: the timeout sentinel (124) is
    ``timeout`` and any other non-zero exit is the generic ``failed``. A
    zero exit is ``passed`` only when the runner executed a real test
    suite; a zero exit from a build-only smoke (``ran_tests`` False, i.e.
    the project wired up no tests) is the honest ``no_tests`` -- a passing
    build is not a passing suite. A runner that never ran degrades to
    ``skipped`` (e.g. npm not installed, or no test/build script).
    """
    if not outcome.ran:
        return "skipped"
    rc = int(outcome.returncode)
    if rc == 124:
        return "timeout"
    if rc != 0:
        return "failed"
    if not getattr(outcome, "ran_tests", True):
        return "no_tests"
    return "passed"


def _run_other_runners(
        runners: Sequence[Any], project_root: str,
        changed_files: Sequence[str], *, timeout: float,
        python_exe: Optional[str]) -> List[Tuple[str, Any]]:
    """Run each non-pytest runner, returning ``(name, outcome)`` pairs.

    A runner that raises is converted into a non-fatal skipped outcome
    so one broken stack never aborts the whole VERIFY.
    """
    pairs: List[Tuple[str, Any]] = []
    for r in runners:
        try:
            oc = r.run(project_root, list(changed_files),
                       timeout_seconds=timeout, python_exe=python_exe)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("VERIFY: test runner %s crashed: %s", r.name, exc)
            from cgx.codegen.test_runner import TestRunOutcome
            oc = TestRunOutcome(
                ran=False,
                skipped_reason=f"{r.name}: {type(exc).__name__}: {exc}")
        pairs.append((r.name, oc))
    return pairs


def _pick_combined_token(
        components: Sequence[Tuple[str, Any, str, bool]]) -> str:
    """Return the single VERIFY outcome token for the merged run.

    A definitive failure anywhere dominates (highest severity wins, with
    pytest preferred on a tie). Otherwise a genuine ``passed`` anywhere
    wins, so a Python-test-free half of a polyglot repo does not mask a
    passing JS build. ``no_tests_collected`` / ``pytest_missing`` are
    reported only when nothing passed and nothing hard-failed.
    """
    tokens = [(is_pytest, token) for _, _, token, is_pytest in components]
    hard = [(is_pytest, token) for is_pytest, token in tokens
            if token in _HARD_FAILURE_SEVERITY]
    if hard:
        hard.sort(key=lambda t: (_HARD_FAILURE_SEVERITY[t[1]], t[0]),
                  reverse=True)
        return hard[0][1]
    for want in ("passed", "no_tests", "no_tests_collected", "pytest_missing"):
        if any(token == want for _, token in tokens):
            return want
    return "skipped"


def _combine_verify_outcomes(
        pytest_outcome: Any,
        other_pairs: Sequence[Tuple[str, Any]]) -> _Combined:
    """Merge the pytest outcome and every other runner into one signal."""
    components: List[Tuple[str, Any, str, bool]] = []
    if pytest_outcome is not None:
        components.append(
            ("pytest", pytest_outcome, _classify_outcome(pytest_outcome), True))
    for name, oc in other_pairs:
        components.append((name, oc, _classify_other_outcome(oc), False))

    ran = any(oc.ran for _, oc, _, _ in components)
    all_selected: List[str] = []
    pytest_selected: List[str] = []
    out_chunks: List[str] = []
    err_chunks: List[str] = []
    skips: List[str] = []
    for name, oc, _token, is_pytest in components:
        all_selected.extend(oc.tests_selected)
        if is_pytest:
            pytest_selected = list(oc.tests_selected)
        if oc.stdout:
            out_chunks.append(f"== {name} ==\n{oc.stdout}")
        if oc.stderr:
            err_chunks.append(f"== {name} ==\n{oc.stderr}")
        if not oc.ran and oc.skipped_reason:
            skips.append(f"{name}: {oc.skipped_reason}")

    outcome = _pick_combined_token(components)
    returncode = 0
    for _name, oc, token, _is_pytest in components:
        if token == outcome and oc.ran:
            returncode = int(oc.returncode)
            break
    # Non-pytest (JS/TS) coverage signal for P2's fail-closed policy: did
    # the tree carry scaffolded JS test files, and did any JS runner
    # actually execute a real suite (a build-only smoke has ran_tests False)?
    js_present = any(
        bool(getattr(oc, "tests_present", None))
        for _name, oc, _token, is_pytest in components if not is_pytest)
    js_ran = any(
        oc.ran and bool(getattr(oc, "ran_tests", False))
        for _name, oc, _token, is_pytest in components if not is_pytest)
    return _Combined(
        ran=ran,
        returncode=returncode,
        outcome=outcome,
        tests_selected=all_selected,
        pytest_tests_selected=pytest_selected,
        stdout="\n\n".join(out_chunks),
        stderr="\n\n".join(err_chunks),
        skipped_reason=(
            None if ran else ("; ".join(skips) or "no test runner detected")),
        js_tests_present=js_present,
        js_tests_ran=js_ran,
    )


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
