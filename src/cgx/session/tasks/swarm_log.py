"""Structured per-turn logging + progress for the swarm executors.

The first swarm cut appended freeform blobs to ``debug_tech_lead.log`` /
``debug_developer.log`` in the process CWD -- unusable for tracing a run and
polluting whatever directory the agent happened to run from. These helpers
route every swarm beat through the same two sinks the greenfield loop uses:
:func:`cgx.session.agent_log.log_event` (append-only JSONL in the project's
``.cgx`` dir) and :func:`cgx.trace.emit_trace` (the in-memory trace ring). A
single ``swarm_beat`` call records both, so a swarm run is as inspectable as a
greenfield one without any bespoke log files.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from cgx.session.agent_log import log_event
from cgx.trace import emit_trace

# Cache one SessionStore per project root. ``swarm_beat`` fires on every step of
# a run (plan, per-file generate/write, tool calls, verify rounds), and the
# first cut opened a fresh SQLite connection on each beat -- a hot, chatty path.
# The store is a thin, thread-safe handle over the project's ``.cgx`` DB, so a
# process-lifetime cache keyed by root removes the per-beat open.
_STORE_CACHE: Dict[str, Any] = {}


def _get_store(project_root: str) -> Any:
    """Return a cached SessionStore for ``project_root`` (opened on first use)."""
    store = _STORE_CACHE.get(project_root)
    if store is None:
        from cgx.session.store import SessionStore
        store = SessionStore(project_root)
        _STORE_CACHE[project_root] = store
    return store


def swarm_beat(project_root: Optional[str], role: str, phase: str,
               **fields: Any) -> None:
    """Record one swarm step to the agent log and the trace ring.

    ``role`` is ``tech_lead`` / ``developer``; ``phase`` names the beat
    (``plan``/``normalize``/``generate``/``gate``/``ast_fallback``/``write``/
    ``report``). Extra fields (file, index, total, ok, method, bytes, error)
    are folded into both sinks. Never raises: logging must not be able to fail
    a generation run.
    """
    payload = {"role": role, "phase": phase, **fields}
    try:
        log_event(project_root, "swarm_beat", **payload)
    except Exception:  # pragma: no cover - logging is best-effort
        pass
    try:
        emit_trace("swarm_beat", **payload)
    except Exception:  # pragma: no cover - tracing is best-effort
        pass
    
    # Also write to the SessionStore as a Fact so the UI dashboard can read it!
    try:
        if project_root:
            from cgx.trace import trace_context
            from cgx.session.models import Fact, FactKind
            import time
            import uuid
            ctx = trace_context.get()
            session_id = ctx.get("session_id")
            task_id = ctx.get("task_id")
            if session_id and task_id:
                store = _get_store(project_root)
                fact = Fact(
                    fact_id="fact_" + uuid.uuid4().hex[:16],
                    session_id=session_id,
                    kind=FactKind.SWARM_BEAT,
                    content=payload,
                    surfaced_in_task_id=task_id,
                    created_at=int(time.time()),
                    updated_at=int(time.time())
                )
                store.add_fact(fact)
    except Exception:  # pragma: no cover - DB write is best-effort here
        pass
