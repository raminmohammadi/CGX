"""User Activity: per-run observation store + recorder (Subsystem C).

The :class:`RunStore` persists one row per ask/plan run; :func:`record_run`
is the best-effort recorder the webui handlers call once a run's ``meta`` is
assembled. Recording never breaks a request -- every failure is swallowed and
logged -- so the observability layer stays strictly non-critical.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from cgx.activity.store import (
    KINDS,
    RunRecord,
    RunStore,
    default_run_db_path,
    get_default_run_store,
)

if TYPE_CHECKING:
    from cgx.govdata import GovernanceConfig

logger = logging.getLogger(__name__)


def _as_int(v: Any) -> Optional[int]:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _as_float(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def record_run(*, kind: str, run_id: str,
               meta: Optional[Dict[str, Any]] = None,
               sources: Optional[List[Dict[str, Any]]] = None,
               question: str = "", model: Optional[str] = None,
               owner: Optional[str] = None,
               project_root: Optional[str] = None,
               latency_ms: Optional[float] = None,
               status: str = "ok",
               store: Optional[RunStore] = None,
               policy: Optional["GovernanceConfig"] = None) -> None:
    """Persist one run observation, extracting signals from ``meta`` defensively.

    The stored ``question`` (and any string labels) pass through the
    data-governance text policy (Subsystem M): PII is scrubbed when
    ``scrub_pii`` is on and the text is capped to a preview when
    ``store_full_text`` is off. Best-effort: any failure is logged and
    swallowed so a recording error can never break the ask/plan response.
    """
    try:
        meta = meta or {}
        if policy is None:
            from cgx.govdata import GovernanceConfig
            policy = GovernanceConfig.from_env()
        citations = meta.get("citations") or []
        n_citations = len(citations) if isinstance(citations, list) else 0
        n_sources = len(sources or [])
        confidence = _as_float(meta.get("confidence"))
        grounded: Optional[bool] = None
        if n_sources or n_citations or confidence is not None:
            grounded = n_citations > 0
        labels = {k: meta[k] for k in ("intent", "mode")
                  if k in meta and isinstance(meta[k], (str, int, float))}
        labels = {k: (policy.apply_text_policy(v) if isinstance(v, str) else v)
                  for k, v in labels.items()}
        rec = RunRecord(
            kind=kind, run_id=run_id,
            model=model or meta.get("model"),
            prompt_version=meta.get("prompt_version"),
            owner=owner, project_root=project_root,
            tokens_in=_as_int(meta.get("tokens_in")),
            tokens_out=_as_int(meta.get("tokens_out")),
            tokens_total=_as_int(meta.get("tokens_total")),
            cost_usd=_as_float(meta.get("cost_usd")),
            latency_ms=_as_float(latency_ms),
            n_sources=n_sources, n_citations=n_citations,
            confidence=confidence, grounded=grounded,
            status=status,
            question=policy.apply_text_policy(question or ""),
            labels=labels,
        )
        (store or get_default_run_store()).record(rec)
    except Exception as e:  # pragma: no cover - activity is non-critical
        logger.warning("record_run failed: %s", e)


def record_agent_turn(*, session: Any, run_id: str,
                      facts: Optional[List[Any]] = None,
                      latency_ms: Optional[float] = None,
                      status: str = "ok",
                      store: Optional[RunStore] = None) -> None:
    """Persist one agent *turn* as a ``kind="agent"`` run observation.

    The session agent loop has no per-request ``meta`` like ask/plan; its
    token/cost accounting lives on the ``LLM_CALL`` facts the runner drained
    during the turn. ``facts`` is the list of :class:`~cgx.session.models.Fact`
    produced *this turn* (the caller diffs the KB before/after the drain so
    prior turns are not double-counted). Their per-call ``content`` totals are
    summed into the ``meta`` shape :func:`record_run` already understands, so
    agent turns surface in the Activity dashboard and register their project
    root on the trace-explorer allow-list. Best-effort: any failure is
    swallowed so recording can never break the agent loop.
    """
    try:
        tokens_in = tokens_out = tokens_total = 0
        cost_usd = 0.0
        model: Optional[str] = None
        for fact in facts or []:
            content = getattr(fact, "content", None) or {}
            tokens_in += _as_int(content.get("tokens_in")) or 0
            tokens_out += _as_int(content.get("tokens_out")) or 0
            tokens_total += _as_int(content.get("tokens_total")) or 0
            cost_usd += _as_float(content.get("cost_usd")) or 0.0
            if model is None and content.get("model"):
                model = str(content.get("model"))
        meta = {"tokens_in": tokens_in, "tokens_out": tokens_out,
                "tokens_total": tokens_total, "cost_usd": cost_usd,
                "mode": getattr(getattr(session, "mode", None), "value", None)}
        record_run(
            kind="agent", run_id=run_id, meta=meta, model=model,
            project_root=getattr(session, "project_root", None),
            question=getattr(session, "original_objective", "") or "",
            latency_ms=latency_ms, status=status, store=store)
    except Exception as e:  # pragma: no cover - activity is non-critical
        logger.warning("record_agent_turn failed: %s", e)


__all__ = [
    "KINDS",
    "RunRecord",
    "RunStore",
    "default_run_db_path",
    "get_default_run_store",
    "record_run",
    "record_agent_turn",
]
