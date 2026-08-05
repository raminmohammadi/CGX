"""Always-on metrics for the index-build and retrieval pipelines (Subsystem B).

Mirrors the RED-style LLM metrics in :mod:`cgx.session.llm_trace`: emission is
best-effort (wrapped so metrics never break a build or query) and independent
of the ``CGX_TRACE`` toggle, so indexing and retrieval stay observable in
production even with tracing off. The :func:`metered` decorator times a call
and hands ``(elapsed_ms, status, result)`` to a recorder that pulls counts out
of the pipeline's return value.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, Dict, Optional

from cgx import metrics as _metrics

# Candidate counts are small integers, so give the histogram count-scale
# buckets rather than the default millisecond ladder.
_RETRIEVAL_CANDIDATE_BUCKETS = (1.0, 5.0, 10.0, 20.0, 50.0, 100.0)


def _status_for(exc: BaseException) -> str:
    """Map an exception to a low-cardinality outcome label."""
    return "cancelled" if type(exc).__name__ == "IndexBuildCancelled" else "error"


def record_index_build(elapsed_ms: float, status: str,
                       result: Optional[Dict[str, Any]]) -> None:
    """Emit build counter + duration histogram + current index-size gauges."""
    try:
        mode = "full"
        if isinstance(result, dict):
            mode = "incremental" if result.get("incremental") else "full"
        _metrics.inc("cgx_index_builds_total",
                     help="Index builds by mode/status.",
                     mode=mode, status=status)
        _metrics.observe("cgx_index_build_duration_ms", elapsed_ms,
                         help="Index build wall-clock duration in ms.")
        if isinstance(result, dict):
            counts = result.get("counts") or {}
            if counts:
                _metrics.set_gauge("cgx_index_records", float(max(counts.values())),
                                   help="Records in the most recent index build.")
            parse = result.get("parse") or {}
            if "total_files" in parse:
                _metrics.set_gauge("cgx_index_files", float(parse["total_files"]),
                                   help="Source files in the most recent index build.")
    except Exception:  # pragma: no cover - metrics must never break a build
        pass


def record_retrieval(elapsed_ms: float, status: str,
                     result: Optional[Dict[str, Any]]) -> None:
    """Emit query counter + latency histogram + candidate-count distribution."""
    try:
        _metrics.inc("cgx_retrieval_queries_total",
                     help="Retrieval queries by status.", status=status)
        _metrics.observe("cgx_retrieval_latency_ms", elapsed_ms,
                         help="Retrieval pipeline latency in ms.")
        if isinstance(result, dict):
            hits = result.get("hits")
            if isinstance(hits, list):
                _metrics.observe("cgx_retrieval_candidates", float(len(hits)),
                                 help="Fused retrieval candidates per query.",
                                 buckets=_RETRIEVAL_CANDIDATE_BUCKETS)
    except Exception:  # pragma: no cover - metrics must never break a query
        pass


def metered(recorder: Callable[[float, str, Any], None]) -> Callable:
    """Decorator: time a call, then hand ``(elapsed_ms, status, result)`` to
    ``recorder``. On success ``status='ok'`` and ``result`` is the return value;
    on failure the outcome is recorded (``error``/``cancelled``) and the
    exception re-raised, so instrumentation never changes behaviour."""
    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrap(*a: Any, **kw: Any) -> Any:
            t0 = time.perf_counter()
            try:
                result = fn(*a, **kw)
            except BaseException as exc:
                recorder((time.perf_counter() - t0) * 1000.0, _status_for(exc), None)
                raise
            recorder((time.perf_counter() - t0) * 1000.0, "ok", result)
            return result
        return wrap
    return deco
