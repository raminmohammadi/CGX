

"""SCAFFOLD executor: generate file contents for an approved WORK_PLAN.

Iterates through ``WORK_PLAN.layers`` in declaration order, calling
:func:`cgx.answer.engine.generate_single_scaffold_file` once per file.
Each generated file's content is fed back into the context for the
next file so cross-file imports resolve correctly (the legacy
scaffold pipeline does the same thing).

Emits an :class:`Artifact` of kind ``SCAFFOLD_PATCHES`` whose
``diffs`` list is shaped for :func:`cgx.codegen.disk_apply.apply_diffs_to_disk`
-- the downstream APPLY task can therefore reuse the existing
disk-apply path without special casing greenfield.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

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


@register_executor(TaskKind.SCAFFOLD)
def run_scaffold(task: TaskNode, deps: ExecutorDeps) -> ExecutorResult:
    """Generate file contents for every entry in the work plan."""
    if deps.provider is None:
        return ExecutorResult(failure="SCAFFOLD requires an LLM provider")
    if deps.store is None:
        return ExecutorResult(
            failure="SCAFFOLD requires a session store in deps")

    work_plan_id = str(task.inputs.get("work_plan_artifact_id") or "").strip()
    if not work_plan_id:
        return ExecutorResult(
            failure="SCAFFOLD missing work_plan_artifact_id")
    work_plan = deps.store.get_artifact(work_plan_id)
    if work_plan is None or work_plan.kind is not ArtifactKind.WORK_PLAN:
        return ExecutorResult(
            failure=f"SCAFFOLD: work plan {work_plan_id!r} missing")

    content = work_plan.content or {}
    layers = content.get("layers") or []
    goal = str(content.get("composed_goal")
               or content.get("prior_goal") or "").strip()
    if not layers:
        return ExecutorResult(
            failure="SCAFFOLD: work plan carries no layers")

    # Lazy import: drags the scaffold prompt templates.
    from cgx.answer.engine import generate_single_scaffold_file

    diffs: List[Dict[str, str]] = []
    existing_with_content: List[Dict[str, str]] = []
    generated: List[Dict[str, Any]] = []
    failed: List[Dict[str, str]] = []

    for layer in layers:
        if not isinstance(layer, dict):
            continue
        layer_name = str(layer.get("name") or "project").strip()
        for entry in (layer.get("files") or []):
            if not isinstance(entry, dict):
                continue
            path = str(entry.get("path") or "").strip()
            description = str(entry.get("description") or path).strip()
            if not path:
                continue
            try:
                result = generate_single_scaffold_file(
                    path, description, deps.provider,
                    layer=layer_name,
                    existing_files_with_content=list(existing_with_content),
                    goal=goal,
                )
            except Exception as exc:
                logger.exception(
                    "SCAFFOLD: generate_single_scaffold_file crashed for %s",
                    path)
                failed.append({
                    "file": path,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                continue

            file_path = str(result.get("file") or path).strip()
            patch = str(result.get("patch") or "")
            file_content = str(result.get("content") or "")
            if not file_path or not patch:
                failed.append({
                    "file": path,
                    "error": "generator returned empty patch",
                })
                continue

            diffs.append({"file": file_path, "patch": patch})
            existing_with_content.append({
                "path": file_path, "content": file_content,
            })
            generated.append({
                "file": file_path,
                "layer": layer_name,
                "syntax_ok": bool(result.get("syntax_ok")),
                "confidence": result.get("confidence"),
                "bytes": len(file_content),
            })

    if not diffs:
        return ExecutorResult(
            failure="SCAFFOLD: every file generation failed")

    artifact = Artifact.new(
        session_id=task.session_id,
        produced_by_task_id=task.task_id,
        kind=ArtifactKind.SCAFFOLD_PATCHES,
        content={
            "work_plan_artifact_id": work_plan_id,
            "prior_goal": content.get("prior_goal"),
            "composed_goal": goal,
            "diffs": diffs,
            "generated": generated,
            "failed": failed,
        },
    )
    return ExecutorResult(
        outputs={
            "scaffold_artifact_id": artifact.artifact_id,
            "generated_count": len(generated),
            "failed_count": len(failed),
        },
        artifact=artifact,
    )
