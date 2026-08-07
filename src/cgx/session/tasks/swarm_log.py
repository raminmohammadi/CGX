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

from typing import Any, Optional

from cgx.session.agent_log import log_event
from cgx.trace import emit_trace


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
