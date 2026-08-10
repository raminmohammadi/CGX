"""RE_VERIFY executor: incremental re-verification of a scoped fix (C2).

Spawned only after a DIAGNOSE-driven scoped fix whose origin gate was
VERIFY (the runtime-origin path re-runs RUNTIME_VERIFY directly). Instead
of re-running the full ``BOOTSTRAP_ENV -> API_CHECK -> SMOKE -> VERIFY``
chain, RE_VERIFY re-runs pytest against *only* the test file(s) the origin
VERIFY_REPORT recorded as failing -- the venv is already provisioned and
every other gate already passed, so re-running them is wasted time.

The report is a :data:`ArtifactKind.VERIFY_REPORT` identical in shape to
VERIFY (same ``classification`` + ``failure_signature`` + progress
counts), so the pure router's :func:`_verify_successors` edge dispatches
it unchanged: green hands off to RUNTIME_VERIFY, a still-failing
reasoning-class outcome routes back to DIAGNOSE under the shared budget.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

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
from cgx.session.tasks.verify import (
    _build_reproduce_cmd,
    _combine_verify_outcomes,
    _parse_junit_failures,
    _progress_counts,
    _resolve_python_exe,
    _unlink_quiet,
)
from cgx.trace import emit_trace

logger = logging.getLogger(__name__)


@register_executor(TaskKind.RE_VERIFY)
def run_re_verify(task: TaskNode, deps: ExecutorDeps) -> ExecutorResult:
    """Re-run pytest against only the origin report's failing test file(s)."""
    if not deps.project_root:
        return ExecutorResult(failure="RE_VERIFY requires project_root in deps")
    if deps.store is None:
        return ExecutorResult(
            failure="RE_VERIFY requires a session store in deps")

    report_id = str(task.inputs.get("reverify_report_id") or "").strip()
    if not report_id:
        return ExecutorResult(
            failure="RE_VERIFY missing reverify_report_id in inputs")
    origin = deps.store.get_artifact(report_id)
    if origin is None or origin.kind is not ArtifactKind.VERIFY_REPORT:
        return ExecutorResult(
            failure=f"RE_VERIFY: artifact {report_id!r} missing or wrong kind "
                    f"(need {ArtifactKind.VERIFY_REPORT.value})")
    origin_content = dict(origin.content or {})

    target_files = _failing_test_files(origin_content)
    changed_files = [str(f) for f in (origin_content.get("changed_files") or [])
                     if str(f).strip()]
    python_exe = _resolve_python_exe(task, deps)
    timeout = float(task.inputs.get("timeout_seconds") or 180.0)

    junit_fd, junit_path = tempfile.mkstemp(prefix="cgx_reverify_", suffix=".xml")
    os.close(junit_fd)
    extra_pytest_args = (
        "-q", "--no-header", "-rN", "--tb=long",
        "--ignore=.cgx-backups", f"--junitxml={junit_path}",
    )
    try:
        pytest_outcome = _run_scoped(
            deps.project_root, target_files, changed_files,
            timeout=timeout, python_exe=python_exe,
            extra_pytest_args=extra_pytest_args)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("RE_VERIFY: scoped pytest run crashed")
        _unlink_quiet(junit_path)
        return ExecutorResult(
            failure=f"re_verify failed: {type(exc).__name__}: {exc}")
    failures = _parse_junit_failures(junit_path)
    _unlink_quiet(junit_path)

    combined = _combine_verify_outcomes(pytest_outcome, [])
    verify_outcome = combined.outcome
    tests_passed = verify_outcome == "passed"
    mode = str(task.inputs.get("mode") or "explore").strip() or "explore"
    reproduce_cmd = _build_reproduce_cmd(
        deps.project_root, python_exe, combined.pytest_tests_selected)
    # Carry the origin's JS coverage signal forward so the terminal
    # fail-closed policy still sees a present-but-unrun JS suite (RE_VERIFY
    # only re-runs the impacted Python file(s), never the JS half).
    js_present = bool(origin_content.get("js_tests_present"))
    js_ran = bool(origin_content.get("js_tests_ran"))
    content: Dict[str, Any] = {
        "apply_artifact_id": task.inputs.get("apply_artifact_id"),
        "plan_artifact_id": task.inputs.get("plan_artifact_id"),
        "scaffold_artifact_id": task.inputs.get("scaffold_artifact_id"),
        "build_artifact_id": task.inputs.get("build_artifact_id"),
        "python_exe": python_exe,
        "mode": mode,
        "changed_files": changed_files,
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
        "js_tests_present": js_present,
        "js_tests_ran": js_ran,
        # Provenance marker so the artifact reads as an incremental re-run
        # rather than a from-scratch VERIFY (design §C2).
        "reverify": True,
        "reverify_report_id": report_id,
    }
    from cgx.session.repair.classify import (
        classify_verify_report, failure_signature)
    sig = failure_signature(content)
    content["failure_signature"] = sig
    classification = classify_verify_report(content)
    content["classification"] = classification
    emit_trace(
        "re_verify", outcome=verify_outcome, classification=classification,
        signature=sig, scoped_files=len(target_files),
        origin_report_id=report_id)
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
            **_progress_counts(
                verify_outcome, failures, combined.pytest_tests_selected),
            "failure_signature": sig,
            "classification": classification,
            "returncode": combined.returncode,
            "js_tests_present": js_present,
            "js_tests_ran": js_ran,
            "reverify": True,
        },
        artifact=artifact,
    )


# --------------------- helpers ---------------------

def _failing_test_files(report_content: Dict[str, Any]) -> List[str]:
    """Return the test file(s) the origin VERIFY_REPORT recorded as failing.

    Each junit failure ``nodeid`` is ``classname::name`` where classname
    is the dotted module path (e.g. ``tests.test_foo.TestBar``); the file
    that failed is the one in the report's ``tests_selected`` whose stem
    (``test_foo``) appears as a component of that path. Falls back to the
    full selected set when nothing resolves (e.g. a collection error that
    named no case) so the re-run never silently exercises zero tests.
    """
    selected = [str(t) for t in (report_content.get("tests_selected") or [])
                if str(t).strip()]
    failures = report_content.get("failures") or []
    if not selected or not failures:
        return selected
    wanted_stems = set()
    for f in failures:
        classname = str((f or {}).get("nodeid") or "").split("::", 1)[0]
        for part in classname.split("."):
            part = part.strip()
            if part:
                wanted_stems.add(part)
    matched = [p for p in selected if Path(p).stem in wanted_stems]
    return matched or selected


def _run_scoped(project_root: str, target_files: List[str],
                changed_files: List[str], *, timeout: float,
                python_exe, extra_pytest_args):
    """Run pytest on the failing file(s), or the impacted set when none named.

    Lazy import: :mod:`cgx.codegen.test_runner` pulls subprocess + pytest
    discovery, kept off the module import path.
    """
    from cgx.codegen.test_runner import run_pytest_paths, run_tests_on_disk
    if target_files:
        return run_pytest_paths(
            project_root, target_files, timeout_seconds=timeout,
            python_exe=python_exe, extra_pytest_args=extra_pytest_args)
    return run_tests_on_disk(
        project_root, changed_files, timeout_seconds=timeout,
        python_exe=python_exe, extra_pytest_args=extra_pytest_args)
