

"""APPLY executor: write an approved plan's diffs to the working tree.

Reads the upstream plan artifact -- a ``CODE_CHANGE_PLAN`` (produced by
``PLAN_CHANGE`` and gated by an ``APPROVE`` ASK), a ``SCAFFOLD_PATCHES``
(greenfield loop), or a ``REPAIR_PLAN`` (auto-repair loop) -- then hands
the diffs to :func:`cgx.codegen.disk_apply.apply_diffs_to_disk`. The
disk-apply helper backs every modified file up under
``.cgx-backups/<run_id>/`` before overwriting, so this executor stays
recoverable.

The result lands in an ``APPLIED_CHANGES`` artifact carrying the lists
of applied/failed files plus the backup path, which the downstream
``VERIFY`` task uses to scope its test selection.
"""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)


@register_executor(TaskKind.APPLY)
def run_apply(task: TaskNode, deps: ExecutorDeps) -> ExecutorResult:
    """Apply the plan's diffs to ``deps.project_root``."""
    if not deps.project_root:
        return ExecutorResult(failure="APPLY requires project_root in deps")
    if deps.store is None:
        return ExecutorResult(
            failure="APPLY requires a session store in deps")
    # The upstream artifact id can arrive under any of three names
    # depending on which loop produced it: ``CODE_CHANGE_PLAN``
    # (existing-codebase write loop), ``SCAFFOLD_PATCHES`` (greenfield
    # loop), or ``REPAIR_PLAN`` (auto-repair loop, wired in via
    # ``_repair_to_apply_or_ask``). All three carry a ``diffs`` list
    # shaped for ``apply_diffs_to_disk``.
    plan_artifact_id = str(
        task.inputs.get("scaffold_artifact_id")
        or task.inputs.get("plan_artifact_id")
        or "").strip()
    if not plan_artifact_id:
        return ExecutorResult(
            failure="APPLY missing plan_artifact_id/scaffold_artifact_id")

    plan = deps.store.get_artifact(plan_artifact_id)
    if plan is None or plan.kind not in (
            ArtifactKind.CODE_CHANGE_PLAN, ArtifactKind.SCAFFOLD_PATCHES,
            ArtifactKind.REPAIR_PLAN):
        return ExecutorResult(
            failure=f"APPLY: artifact {plan_artifact_id!r} missing or wrong "
                    "kind (need CODE_CHANGE_PLAN, SCAFFOLD_PATCHES, or "
                    "REPAIR_PLAN)")

    diffs = _coerce_diffs((plan.content or {}).get("diffs"))
    if not diffs:
        return ExecutorResult(
            failure="APPLY: upstream artifact carries no diffs")

    # Lazy import: disk_apply pulls in the validator + diff parser.
    from cgx.codegen.disk_apply import apply_diffs_to_disk

    try:
        report = apply_diffs_to_disk(deps.project_root, diffs)
    except Exception as exc:
        logger.exception("APPLY: apply_diffs_to_disk crashed")
        return ExecutorResult(
            failure=f"apply failed: {type(exc).__name__}: {exc}")

    applied_files = list(report.get("applied_files") or [])
    failed_files = list(report.get("failed_files") or [])
    backup_dir = report.get("backup_dir")
    smoke_ok = bool(report.get("smoke_ok"))

    artifact = Artifact.new(
        session_id=task.session_id,
        produced_by_task_id=task.task_id,
        kind=ArtifactKind.APPLIED_CHANGES,
        content={
            "plan_artifact_id": plan_artifact_id,
            "source_artifact_kind": plan.kind.value,
            "applied_files": applied_files,
            "failed_files": failed_files,
            "backup_dir": backup_dir,
            "smoke_ok": smoke_ok,
            "diffs": diffs,
        },
    )

    # If nothing was applied, the loop should not progress to VERIFY
    # with empty results -- surface the failure explicitly so the user
    # sees a clear stop point.
    failure: str | None = None
    if not applied_files:
        failure = ("apply produced no applied_files; "
                   f"{len(failed_files)} failed file(s)")

    return ExecutorResult(
        outputs={
            "apply_artifact_id": artifact.artifact_id,
            "applied_count": len(applied_files),
            "failed_count": len(failed_files),
            "failed_files": failed_files,
            "backup_dir": backup_dir,
        },
        artifact=artifact,
        failure=failure,
    )


# --------------------- helpers ---------------------

def _coerce_diffs(raw: Any) -> List[Dict[str, str]]:
    """Re-normalise diffs in case the planner persisted mixed-key entries."""
    out: List[Dict[str, str]] = []
    if not isinstance(raw, list):
        return out
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        file = str(entry.get("file") or entry.get("path") or "").strip()
        patch = str(entry.get("patch") or entry.get("diff") or "")
        if not file or not patch:
            continue
        out.append({"file": file, "patch": patch})
    return out
