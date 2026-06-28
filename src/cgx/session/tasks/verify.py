

"""VERIFY executor: run impacted tests after an APPLY.

Reads the upstream ``APPLIED_CHANGES`` artifact to learn which files
the apply step wrote, then delegates to
:func:`cgx.codegen.test_runner.run_tests_on_disk` which selects impacted
tests via import-graph heuristics and runs pytest against the live
working tree. The result is persisted as a ``VERIFY_REPORT`` artifact.

VERIFY is intentionally a terminal kind in the session router -- it
does not spawn a successor. The caller decides what to do next (post a
new objective, roll back via backup_dir, etc.) once the report is
visible.
"""

from __future__ import annotations

import logging
from typing import Any, List

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


@register_executor(TaskKind.VERIFY)
def run_verify(task: TaskNode, deps: ExecutorDeps) -> ExecutorResult:
    """Run impacted tests for the just-applied changes."""
    if not deps.project_root:
        return ExecutorResult(failure="VERIFY requires project_root in deps")
    if deps.store is None:
        return ExecutorResult(
            failure="VERIFY requires a session store in deps")

    changed_files = _resolve_changed_files(task, deps)

    # Lazy import: test_runner pulls subprocess + pytest discovery.
    from cgx.codegen.test_runner import run_tests_on_disk

    timeout = float(task.inputs.get("timeout_seconds") or 180.0)
    try:
        outcome = run_tests_on_disk(
            deps.project_root, changed_files,
            timeout_seconds=timeout,
        )
    except Exception as exc:
        logger.exception("VERIFY: run_tests_on_disk crashed")
        return ExecutorResult(
            failure=f"verify failed: {type(exc).__name__}: {exc}")

    tests_passed = bool(outcome.ran and outcome.returncode == 0)
    mode = str(task.inputs.get("mode") or "explore").strip() or "explore"
    artifact = Artifact.new(
        session_id=task.session_id,
        produced_by_task_id=task.task_id,
        kind=ArtifactKind.VERIFY_REPORT,
        content={
            "apply_artifact_id": task.inputs.get("apply_artifact_id"),
            "plan_artifact_id": task.inputs.get("plan_artifact_id"),
            "scaffold_artifact_id": task.inputs.get("scaffold_artifact_id"),
            "mode": mode,
            "changed_files": list(changed_files),
            "ran": bool(outcome.ran),
            "tests_passed": tests_passed,
            "returncode": int(outcome.returncode),
            "tests_selected": list(outcome.tests_selected),
            "stdout": outcome.stdout or "",
            "stderr": outcome.stderr or "",
            "skipped_reason": outcome.skipped_reason,
        },
    )
    return ExecutorResult(
        outputs={
            "verify_artifact_id": artifact.artifact_id,
            "ran": bool(outcome.ran),
            "tests_passed": tests_passed,
            "tests_selected_count": len(outcome.tests_selected),
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
