

"""DECOMPOSE executor: turn clarified requirements into a work plan.

Wraps :func:`cgx.answer.engine.plan_scaffold_manifest` so the
greenfield loop produces a typed :class:`Artifact` of kind
``WORK_PLAN`` carrying the file manifest (``plan_md`` + ``layers``) the
downstream ``SCAFFOLD`` executor iterates.

The clarify answers (collected via ASK_USER(CLARIFY_ANSWERS)) are
folded into the goal string so the manifest planner sees the user's
tech-stack / scope decisions in its prompt.
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


@register_executor(TaskKind.DECOMPOSE)
def run_decompose(task: TaskNode, deps: ExecutorDeps) -> ExecutorResult:
    """Produce a ``WORK_PLAN`` artifact from a clarified objective."""
    if deps.provider is None:
        return ExecutorResult(failure="DECOMPOSE requires an LLM provider")
    if deps.store is None:
        return ExecutorResult(
            failure="DECOMPOSE requires a session store in deps")

    prior_goal = str(task.inputs.get("prior_goal") or "").strip()
    answers = task.inputs.get("answers") or {}
    if not isinstance(answers, dict):
        answers = {}
    questions = _load_questions(task, deps)

    composed_goal = _compose_goal(prior_goal, questions, answers)
    if not composed_goal:
        return ExecutorResult(failure="DECOMPOSE: empty composed goal")

    # Lazy import: the answer engine drags retrieval + prompt builders.
    from cgx.answer.engine import plan_scaffold_manifest

    try:
        result = plan_scaffold_manifest(
            composed_goal, deps.provider, goal=composed_goal)
    except Exception as exc:
        logger.exception("DECOMPOSE: plan_scaffold_manifest crashed")
        return ExecutorResult(
            failure=f"decompose failed: {type(exc).__name__}: {exc}")

    plan_md = str((result or {}).get("plan_md") or "")
    layers = _coerce_layers((result or {}).get("layers"))
    if not _layer_file_count(layers):
        return ExecutorResult(
            failure="DECOMPOSE: planner returned an empty manifest")

    artifact = Artifact.new(
        session_id=task.session_id,
        produced_by_task_id=task.task_id,
        kind=ArtifactKind.WORK_PLAN,
        content={
            "prior_goal": prior_goal,
            "composed_goal": composed_goal,
            "answers": dict(answers),
            "plan_md": plan_md,
            "layers": layers,
        },
    )
    return ExecutorResult(
        outputs={
            "work_plan_artifact_id": artifact.artifact_id,
            "file_count": _layer_file_count(layers),
            "layer_count": len(layers),
        },
        artifact=artifact,
    )


# --------------------- helpers ---------------------

def _load_questions(task: TaskNode,
                    deps: ExecutorDeps) -> List[Dict[str, Any]]:
    """Pull the question list off the upstream REQUIREMENTS_SHEET."""
    artifact_id = str(
        task.inputs.get("requirements_artifact_id") or "").strip()
    if not artifact_id:
        return []
    artifact = deps.store.get_artifact(artifact_id)
    if artifact is None or artifact.kind is not ArtifactKind.REQUIREMENTS_SHEET:
        return []
    qs = (artifact.content or {}).get("questions") or []
    if not isinstance(qs, list):
        return []
    return [q for q in qs if isinstance(q, dict)]


def _compose_goal(prior_goal: str,
                  questions: List[Dict[str, Any]],
                  answers: Dict[str, Any]) -> str:
    """Render a single goal string that bakes the clarify answers in."""
    parts: List[str] = []
    if prior_goal:
        parts.append(prior_goal)
    qa_lines: List[str] = []
    for q in questions:
        qid = str(q.get("id") or "").strip()
        prompt = str(q.get("prompt") or "").strip()
        answer = str(answers.get(qid) or "").strip()
        if not (qid and prompt and answer):
            continue
        qa_lines.append(f"- {prompt} -> {answer}")
    # Surface any free-form answers the user supplied for question ids
    # not seen in the requirements sheet (defensive: tests / older UIs).
    seen_ids = {str(q.get("id") or "") for q in questions}
    for qid, answer in answers.items():
        if str(qid) in seen_ids:
            continue
        ans = str(answer or "").strip()
        if ans:
            qa_lines.append(f"- {qid}: {ans}")
    if qa_lines:
        parts.append("User clarifications:\n" + "\n".join(qa_lines))
    return "\n\n".join(parts).strip()


def _coerce_layers(raw: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(raw, list):
        return out
    for layer in raw:
        if not isinstance(layer, dict):
            continue
        name = str(layer.get("name") or "project").strip()
        files: List[Dict[str, str]] = []
        for f in (layer.get("files") or []):
            if not isinstance(f, dict):
                continue
            path = str(f.get("path") or "").strip()
            desc = str(f.get("description") or path).strip()
            if not path:
                continue
            files.append({"path": path, "description": desc})
        out.append({"name": name, "files": files})
    return out


def _layer_file_count(layers: List[Dict[str, Any]]) -> int:
    return sum(len(layer.get("files") or []) for layer in layers)
