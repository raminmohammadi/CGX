

"""EXPLORE executor: turn a goal into a typed list of directions.

The legacy agent loop did this through ``answer_with_llm`` with
``mode_override="clarify_paths"`` and then re-parsed the rendered
Markdown to find the choices. We bypass that round-trip: the same
function already returns the structured option list in
``debug["options"]``, so the executor lifts it directly into:

* one :class:`Artifact` of kind ``DIRECTIONS_LIST`` -- the canonical
  output every downstream task references.
* one :class:`Fact` of kind ``ANCHOR`` per option, so the KB can be
  queried by anchor ``chunk_id`` without re-reading the artifact.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from cgx.session.models import (
    Artifact,
    ArtifactKind,
    Fact,
    FactKind,
    TaskKind,
    TaskNode,
)
from cgx.session.tasks.base import (
    ExecutorDeps,
    ExecutorResult,
    register_executor,
)

logger = logging.getLogger(__name__)


@register_executor(TaskKind.EXPLORE)
def run_explore(task: TaskNode, deps: ExecutorDeps) -> ExecutorResult:
    """Run a clarify_paths retrieval against the indexed project."""
    goal = str(task.inputs.get("goal") or "").strip()
    if not goal:
        return ExecutorResult(failure="EXPLORE task has empty goal")
    if not deps.index_dir or not deps.records_path:
        return ExecutorResult(
            failure="EXPLORE requires index_dir + records_path in deps")
    if deps.provider is None:
        return ExecutorResult(failure="EXPLORE requires an LLM provider")
    # Pre-flight: refuse early when the index files are missing so the
    # user gets a clear hint instead of a FileNotFoundError deep in the
    # retrieval stack. This is also the signal greenfield sessions use
    # to bypass EXPLORE entirely (the router spawns CLARIFY_REQUIREMENTS
    # instead), but we keep the guard here for direct callers.
    from cgx.session.mode import _has_usable_index
    if not _has_usable_index(deps.index_dir, deps.records_path):
        return ExecutorResult(
            failure=("EXPLORE: no usable index at "
                     f"{deps.index_dir!r}; build one first or start a "
                     "greenfield session for new projects"))

    # Imported lazily so the session package doesn't drag the answer
    # engine into every import path (it pulls FAISS / sentence-tx).
    from cgx.answer.engine import answer_with_llm

    try:
        result = answer_with_llm(
            index_dir=deps.index_dir,
            records_path=deps.records_path,
            question=goal,
            provider=deps.provider,
            mode_override="clarify_paths",
        )
    except Exception as exc:
        logger.exception("EXPLORE: answer_with_llm crashed")
        return ExecutorResult(
            failure=f"clarify_paths failed: {type(exc).__name__}: {exc}")

    debug = result.get("debug") if isinstance(result, dict) else None
    options = _coerce_options((debug or {}).get("options"))
    sources = (debug or {}).get("sources") or []

    artifact = Artifact.new(
        session_id=task.session_id,
        produced_by_task_id=task.task_id,
        kind=ArtifactKind.DIRECTIONS_LIST,
        content={
            "goal": goal,
            "restatement": (debug or {}).get("restatement", ""),
            "follow_up_question": (debug or {}).get("follow_up_question", ""),
            "options": options,
            "answer_md": result.get("answer_md", "") if isinstance(result, dict) else "",
        },
    )

    facts: List[Fact] = []
    for opt in options:
        chunk_id = str(opt.get("chunk_id") or "").strip()
        if not chunk_id:
            continue
        source = _find_source(sources, chunk_id)
        facts.append(Fact.new(
            session_id=task.session_id,
            kind=FactKind.ANCHOR,
            content={
                "chunk_id": chunk_id,
                "title": opt.get("title", ""),
                "rationale": opt.get("rationale", ""),
                "path": (source or {}).get("path"),
                "symbol": (source or {}).get("symbol"),
            },
            surfaced_in_task_id=task.task_id,
        ))

    return ExecutorResult(
        outputs={
            "options_count": len(options),
            "directions_artifact_id": artifact.artifact_id,
            "confidence": result.get("confidence") if isinstance(result, dict) else None,
        },
        facts=facts,
        artifact=artifact,
    )


# --------------------- helpers ---------------------

def _coerce_options(raw: Any) -> List[Dict[str, Any]]:
    """Defensive coercion -- the engine usually returns clean dicts but
    older models occasionally produce ``None`` or stringified slots.
    """
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id") or "").strip()
        title = str(item.get("title") or "").strip()
        rationale = str(item.get("rationale") or "").strip()
        if not chunk_id or not title:
            continue
        out.append({
            "chunk_id": chunk_id,
            "title": title,
            "rationale": rationale,
        })
    return out


def _find_source(sources: List[Dict[str, Any]], chunk_id: str
                 ) -> Dict[str, Any] | None:
    for s in sources or []:
        if str(s.get("chunk_id") or "") == chunk_id:
            return s
    return None
