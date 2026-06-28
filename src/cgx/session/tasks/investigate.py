

"""INVESTIGATE executor: deep retrieval anchored on a chosen direction.

Runs once the user has resolved an ``ASK_USER`` of kind ``CHOOSE_PATH``
with a ``Decision`` carrying the picked option's ``anchor_chunk_id``.
The executor synthesises a focused question from the prior goal + the
chosen option's title and calls ``answer_with_llm`` in its default mode
(grounded answer with citations). The result lands in a
``FINDINGS_BUNDLE`` artifact that the downstream ``RECOMMEND`` task
reads.

Sources surfaced by retrieval are projected into typed ``FILE`` /
``SYMBOL`` facts so the KB accumulates structured knowledge instead of
only opaque artifacts. The executor does *not* mutate prior anchor
facts -- staleness is a later-phase concern.
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


@register_executor(TaskKind.INVESTIGATE)
def run_investigate(task: TaskNode, deps: ExecutorDeps) -> ExecutorResult:
    """Run anchored retrieval and persist a ``FINDINGS_BUNDLE``."""
    anchor = str(task.inputs.get("anchor_chunk_id") or "").strip()
    if not anchor:
        return ExecutorResult(failure="INVESTIGATE missing anchor_chunk_id")
    if not deps.index_dir or not deps.records_path:
        return ExecutorResult(
            failure="INVESTIGATE requires index_dir + records_path in deps")
    if deps.provider is None:
        return ExecutorResult(failure="INVESTIGATE requires an LLM provider")

    question = _compose_question(task)

    # Imported lazily so the session package doesn't drag the answer
    # engine into every import path (it pulls FAISS / sentence-tx).
    from cgx.answer.engine import answer_with_llm

    try:
        result = answer_with_llm(
            index_dir=deps.index_dir,
            records_path=deps.records_path,
            question=question,
            provider=deps.provider,
        )
    except Exception as exc:
        logger.exception("INVESTIGATE: answer_with_llm crashed")
        return ExecutorResult(
            failure=f"investigate failed: {type(exc).__name__}: {exc}")

    debug = result.get("debug") if isinstance(result, dict) else {}
    sources: List[Dict[str, Any]] = (debug or {}).get("sources") or []
    answer_md = (result.get("answer_md")
                 if isinstance(result, dict) else "") or ""
    citations = (result.get("citations")
                 if isinstance(result, dict) else []) or []
    confidence = (result.get("confidence")
                  if isinstance(result, dict) else None)

    artifact = Artifact.new(
        session_id=task.session_id,
        produced_by_task_id=task.task_id,
        kind=ArtifactKind.FINDINGS_BUNDLE,
        content={
            "anchor_chunk_id": anchor,
            "title": task.inputs.get("title") or "",
            "prior_goal": task.inputs.get("prior_goal") or "",
            "question": question,
            "answer_md": answer_md,
            "citations": citations,
            "sources": sources,
            "confidence": confidence,
        },
    )

    facts = _facts_from_sources(task, sources)

    return ExecutorResult(
        outputs={
            "findings_artifact_id": artifact.artifact_id,
            "anchor_chunk_id": anchor,
            "confidence": confidence,
            "sources_count": len(sources),
        },
        facts=facts,
        artifact=artifact,
    )


# --------------------- helpers ---------------------

def _compose_question(task: TaskNode) -> str:
    """Build the focused query the LLM sees.

    The chosen option's title + rationale are the strongest signal for
    what the user wants explained; ``prior_goal`` keeps the overall
    objective visible so retrieval doesn't drift off-topic.
    """
    parts: List[str] = []
    prior_goal = str(task.inputs.get("prior_goal") or "").strip()
    title = str(task.inputs.get("title") or "").strip()
    rationale = str(task.inputs.get("rationale") or "").strip()
    anchor = str(task.inputs.get("anchor_chunk_id") or "").strip()
    if prior_goal:
        parts.append(f"Original goal: {prior_goal}")
    if title:
        parts.append(f"Focused on: {title}")
    if rationale:
        parts.append(f"Why: {rationale}")
    if anchor:
        parts.append(f"Anchor chunk: {anchor}")
    parts.append("Explain how this code relates to the goal, what it "
                 "currently does, and what concretely would need to "
                 "change to achieve the goal.")
    return "\n".join(parts)


def _facts_from_sources(task: TaskNode,
                        sources: List[Dict[str, Any]]) -> List[Fact]:
    facts: List[Fact] = []
    seen_paths: set[str] = set()
    seen_symbols: set[str] = set()
    for s in sources:
        path = str((s or {}).get("path") or "").strip()
        symbol = str((s or {}).get("symbol") or "").strip()
        chunk_id = str((s or {}).get("chunk_id") or "").strip()
        if path and path not in seen_paths:
            seen_paths.add(path)
            facts.append(Fact.new(
                task.session_id, FactKind.FILE,
                {"path": path, "chunk_id": chunk_id},
                surfaced_in_task_id=task.task_id))
        if symbol and symbol not in seen_symbols:
            seen_symbols.add(symbol)
            facts.append(Fact.new(
                task.session_id, FactKind.SYMBOL,
                {"symbol": symbol, "path": path, "chunk_id": chunk_id},
                surfaced_in_task_id=task.task_id))
    return facts
