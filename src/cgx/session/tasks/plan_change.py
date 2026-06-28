

"""PLAN_CHANGE executor: synthesise a concrete code-change plan + diffs.

Wraps the legacy :func:`cgx.answer.engine.generate_code_plan` so the
session loop can produce a typed ``CODE_CHANGE_PLAN`` artifact that
downstream ``APPLY`` reads. The task text fed to the planner is built
from the user's chosen recommendation (title + rationale) plus the
prior goal so retrieval still has the high-level objective in scope.
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


@register_executor(TaskKind.PLAN_CHANGE)
def run_plan_change(task: TaskNode, deps: ExecutorDeps) -> ExecutorResult:
    """Generate a code-change plan and persist it as ``CODE_CHANGE_PLAN``."""
    if not deps.index_dir or not deps.records_path:
        return ExecutorResult(
            failure="PLAN_CHANGE requires index_dir + records_path in deps")
    if deps.provider is None:
        return ExecutorResult(failure="PLAN_CHANGE requires an LLM provider")

    task_text = _compose_task_text(task)
    if not task_text:
        return ExecutorResult(failure="PLAN_CHANGE: empty task description")

    # Imported lazily; the answer engine drags retrieval + FAISS in.
    from cgx.answer.engine import generate_code_plan

    try:
        result = generate_code_plan(
            index_dir=deps.index_dir,
            records_path=deps.records_path,
            task=task_text,
            provider=deps.provider,
            project_root=deps.project_root,
            embedder=deps.extra.get("embedder") if deps.extra else None,
        )
    except Exception as exc:
        logger.exception("PLAN_CHANGE: generate_code_plan crashed")
        return ExecutorResult(
            failure=f"plan_change failed: {type(exc).__name__}: {exc}")

    plan_md = str((result or {}).get("plan_md") or "")
    diffs = _normalize_diffs((result or {}).get("diffs"))
    citations = (result or {}).get("citations") or []
    confidence = (result or {}).get("confidence")

    artifact = Artifact.new(
        session_id=task.session_id,
        produced_by_task_id=task.task_id,
        kind=ArtifactKind.CODE_CHANGE_PLAN,
        content={
            "task_text": task_text,
            "prior_goal": task.inputs.get("prior_goal") or "",
            "recommendation": task.inputs.get("recommendation") or {},
            "plan_md": plan_md,
            "diffs": diffs,
            "citations": citations,
            "confidence": confidence,
        },
    )
    return ExecutorResult(
        outputs={
            "plan_artifact_id": artifact.artifact_id,
            "diffs_count": len(diffs),
            "confidence": confidence,
        },
        artifact=artifact,
    )


# --------------------- helpers ---------------------

def _compose_task_text(task: TaskNode) -> str:
    """Build the planner's task description from session context."""
    parts: List[str] = []
    prior_goal = str(task.inputs.get("prior_goal") or "").strip()
    rec = task.inputs.get("recommendation") or {}
    title = str(rec.get("title") or "").strip()
    rationale = str(rec.get("rationale") or "").strip()
    anchor = str(task.inputs.get("anchor_chunk_id") or "").strip()
    if prior_goal:
        parts.append(f"Original goal: {prior_goal}")
    if title:
        parts.append(f"Change: {title}")
    if rationale:
        parts.append(f"Why: {rationale}")
    if anchor:
        parts.append(f"Anchor chunk: {anchor}")
    return "\n".join(parts).strip()


def _normalize_diffs(raw: Any) -> List[Dict[str, str]]:
    """Coerce the planner's diff list into ``[{"file","patch"}, ...]``.

    The legacy planner emits either ``{"file","patch"}`` or
    ``{"path","diff"}`` keys depending on which prompt branch fired;
    normalise to the shape ``apply_diffs_to_disk`` expects.
    """
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
