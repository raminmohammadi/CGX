

"""RECOMMEND executor: synthesise typed recommendations from findings.

Reads the upstream ``FINDINGS_BUNDLE`` artifact (produced by
``INVESTIGATE``) and asks the LLM in JSON mode for 2-4 concrete
next-step recommendations. Each recommendation is a typed dict carrying
a ``kind`` token that the router will use in Phase 3+ to decide what to
spawn when the user picks one (``investigate_more`` -> a second
INVESTIGATE; ``plan_change`` -> PLAN_CHANGE; etc.).

The structured contract keeps the downstream router deterministic --
the LLM never controls control flow directly, it only proposes typed
options the router can dispatch on.
"""

from __future__ import annotations

import json
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


_ALLOWED_KINDS = {"investigate_more", "plan_change", "ask_followup", "done"}


_SYSTEM_PROMPT = (
    "You synthesize next-step recommendations from a code investigation "
    "report. Output STRICT JSON only, no prose. Schema:\n"
    '{"recommendations": [{"id": "r1", "title": "...", '
    '"rationale": "...", "kind": "investigate_more"|"plan_change"|'
    '"ask_followup"|"done", "anchor_chunk_id": "<optional>"}]}\n'
    "Rules:\n"
    "- Produce 2 to 4 recommendations. Each must be concrete and "
    "actionable.\n"
    "- Ground every recommendation in the FINDINGS; do not invent "
    "symbols or paths that are not referenced there.\n"
    "- ``kind`` must be one of the four allowed tokens.\n"
    "- ``anchor_chunk_id`` is required when ``kind`` is "
    "``investigate_more`` and should match a chunk_id present in the "
    "findings sources.\n"
)


@register_executor(TaskKind.RECOMMEND)
def run_recommend(task: TaskNode, deps: ExecutorDeps) -> ExecutorResult:
    """Read the findings bundle and emit a ``RECOMMENDATION_LIST``."""
    artifact_id = str(task.inputs.get("findings_artifact_id") or "").strip()
    if not artifact_id:
        return ExecutorResult(failure="RECOMMEND missing findings_artifact_id")
    if deps.provider is None:
        return ExecutorResult(failure="RECOMMEND requires an LLM provider")
    if deps.store is None:
        return ExecutorResult(
            failure="RECOMMEND requires a session store in deps")

    findings = deps.store.get_artifact(artifact_id)
    if findings is None or findings.kind is not ArtifactKind.FINDINGS_BUNDLE:
        return ExecutorResult(
            failure=f"RECOMMEND: findings artifact {artifact_id!r} "
                    "missing or wrong kind")

    user_context = _build_user_context(findings.content or {})
    try:
        resp = deps.provider.chat(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_context},
            ],
            temperature=0.2,
            force_json=True,
        )
    except Exception as exc:
        logger.exception("RECOMMEND: provider.chat crashed")
        return ExecutorResult(
            failure=f"recommend failed: {type(exc).__name__}: {exc}")

    raw = (resp or {}).get("content") or ""
    parsed = _parse_recommendations(raw)
    allowed_chunks = _allowed_chunk_ids(findings.content or {})
    recs = _validate_recommendations(parsed, allowed_chunks)

    if not recs:
        # Defensive fallback: synthesise a single ``done`` recommendation
        # so the loop can still progress to ASK_USER instead of failing
        # the whole task on a transient LLM JSON miss.
        recs = [{
            "id": "r1",
            "title": "Wrap up this investigation",
            "rationale": ("The model did not return usable structured "
                          "recommendations; close the loop here."),
            "kind": "done",
        }]

    artifact = Artifact.new(
        session_id=task.session_id,
        produced_by_task_id=task.task_id,
        kind=ArtifactKind.RECOMMENDATION_LIST,
        content={
            "findings_artifact_id": artifact_id,
            "anchor_chunk_id": (findings.content or {}).get("anchor_chunk_id"),
            "prior_goal": (findings.content or {}).get("prior_goal"),
            "recommendations": recs,
        },
    )
    return ExecutorResult(
        outputs={
            "recommendations_artifact_id": artifact.artifact_id,
            "recommendations_count": len(recs),
        },
        artifact=artifact,
    )


# --------------------- helpers ---------------------

def _build_user_context(findings: Dict[str, Any]) -> str:
    sources = findings.get("sources") or []
    src_lines = []
    for s in sources[:12]:
        if not isinstance(s, dict):
            continue
        src_lines.append(
            f"- chunk_id={s.get('chunk_id')} path={s.get('path')} "
            f"symbol={s.get('symbol') or ''}")
    return (
        f"PRIOR GOAL: {findings.get('prior_goal') or ''}\n"
        f"FOCUS TITLE: {findings.get('title') or ''}\n"
        f"ANCHOR CHUNK: {findings.get('anchor_chunk_id') or ''}\n\n"
        f"FINDINGS (answer_md):\n{findings.get('answer_md') or ''}\n\n"
        f"SOURCES:\n" + ("\n".join(src_lines) or "(none)")
    )


def _allowed_chunk_ids(findings: Dict[str, Any]) -> set:
    out: set = set()
    anchor = str(findings.get("anchor_chunk_id") or "").strip()
    if anchor:
        out.add(anchor)
    for s in findings.get("sources") or []:
        if isinstance(s, dict):
            cid = str(s.get("chunk_id") or "").strip()
            if cid:
                out.add(cid)
    return out


def _parse_recommendations(raw: str) -> List[Dict[str, Any]]:
    raw = (raw or "").strip()
    if not raw:
        return []
    try:
        obj = json.loads(raw)
    except Exception:
        return []
    if not isinstance(obj, dict):
        return []
    items = obj.get("recommendations")
    if not isinstance(items, list):
        return []
    return [i for i in items if isinstance(i, dict)]


def _validate_recommendations(items: List[Dict[str, Any]],
                              allowed_chunks: set) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, item in enumerate(items[:4]):
        title = str(item.get("title") or "").strip()
        rationale = str(item.get("rationale") or "").strip()
        kind = str(item.get("kind") or "").strip()
        if not title or kind not in _ALLOWED_KINDS:
            continue
        anchor: Optional[str] = None
        raw_anchor = str(item.get("anchor_chunk_id") or "").strip()
        if raw_anchor and raw_anchor in allowed_chunks:
            anchor = raw_anchor
        if kind == "investigate_more" and anchor is None:
            # Drop investigate_more recs that don't resolve to a known
            # chunk -- the router can't act on them deterministically.
            continue
        out.append({
            "id": str(item.get("id") or f"r{idx + 1}"),
            "title": title,
            "rationale": rationale,
            "kind": kind,
            "anchor_chunk_id": anchor,
        })
    return out
