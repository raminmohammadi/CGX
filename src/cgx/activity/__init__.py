"""User Activity: per-run observation store + recorder (Subsystem C).

The :class:`RunStore` persists one row per ask/plan run; :func:`record_run`
is the best-effort recorder the webui handlers call once a run's ``meta`` is
assembled. Recording never breaks a request -- every failure is swallowed and
logged -- so the observability layer stays strictly non-critical.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from cgx.activity.store import (
    KINDS,
    RunRecord,
    RunStore,
    default_run_db_path,
    get_default_run_store,
)

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
               store: Optional[RunStore] = None) -> None:
    """Persist one run observation, extracting signals from ``meta`` defensively.

    Best-effort: any failure is logged and swallowed so a recording error can
    never break the ask/plan response.
    """
    try:
        meta = meta or {}
        citations = meta.get("citations") or []
        n_citations = len(citations) if isinstance(citations, list) else 0
        n_sources = len(sources or [])
        confidence = _as_float(meta.get("confidence"))
        grounded: Optional[bool] = None
        if n_sources or n_citations or confidence is not None:
            grounded = n_citations > 0
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
            status=status, question=(question or "")[:500],
            labels={k: meta[k] for k in ("intent", "mode")
                    if k in meta and isinstance(meta[k], (str, int, float))},
        )
        (store or get_default_run_store()).record(rec)
    except Exception as e:  # pragma: no cover - activity is non-critical
        logger.warning("record_run failed: %s", e)


__all__ = [
    "KINDS",
    "RunRecord",
    "RunStore",
    "default_run_db_path",
    "get_default_run_store",
    "record_run",
]
