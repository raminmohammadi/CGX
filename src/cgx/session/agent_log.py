

"""Project-local JSONL trace for the agent session loop.

Writes one JSON object per line to ``<project_root>/.cgx/agent.log`` so
operators can reconstruct what the agent did -- task transitions,
executor crashes, repair branches -- without spelunking through the
SQLite store. The file uses a rotating handler (1 MiB x 4 files) so a
long-running session can't fill the disk.

This module is intentionally orthogonal to :mod:`cgx.logging_setup`
(stdout text logger). The records here are structured JSONL meant for
machine consumption -- ``jq`` first, human eyes second.

Usage::

    from cgx.session.agent_log import log_event

    log_event(project_root, "task_started",
              session_id=s.session_id, task_id=t.task_id,
              kind=t.kind.value)

A ``None`` ``project_root`` makes :func:`log_event` a no-op so callers
don't need to special-case sessions without a working tree (REPL,
tests, throwaway explore sessions).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_LOG_DIR_NAME = ".cgx"
_LOG_FILE_NAME = "agent.log"
_MAX_BYTES = 1 * 1024 * 1024  # 1 MiB
_BACKUP_COUNT = 3

# Handler cache keyed by resolved project_root so re-opening the same
# file across many log_event calls doesn't churn file descriptors.
_handlers: Dict[str, RotatingFileHandler] = {}
_handlers_lock = threading.Lock()


class _JsonlFormatter(logging.Formatter):
    """One JSON object per record, no enclosing array.

    Pulls structured fields from ``record.agent_event`` (a dict set by
    :func:`log_event`) and merges them with ``ts`` / ``event`` so the
    line is self-describing.
    """

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        payload = getattr(record, "agent_event", None) or {}
        out: Dict[str, Any] = {
            "ts": payload.get("ts") or time.time(),
            "event": payload.get("event") or record.msg,
        }
        for k, v in payload.items():
            if k in ("ts", "event"):
                continue
            out[k] = v
        try:
            return json.dumps(out, default=str, ensure_ascii=False)
        except Exception:  # pragma: no cover - defensive
            return json.dumps({"ts": out["ts"], "event": "log_format_error"})


def _resolve_path(project_root: str) -> Path:
    return Path(project_root).resolve() / _LOG_DIR_NAME / _LOG_FILE_NAME


def _handler_for(project_root: str) -> Optional[RotatingFileHandler]:
    key = str(Path(project_root).resolve())
    with _handlers_lock:
        h = _handlers.get(key)
        if h is not None:
            return h
        path = _resolve_path(project_root)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            h = RotatingFileHandler(
                str(path), maxBytes=_MAX_BYTES,
                backupCount=_BACKUP_COUNT, encoding="utf-8")
            h.setFormatter(_JsonlFormatter())
            h.setLevel(logging.INFO)
            _handlers[key] = h
            return h
        except Exception as exc:
            logger.warning(
                "agent_log: failed to open %s (%s); disabling for this root",
                path, exc)
            return None


def log_event(project_root: Optional[str], event: str, **fields: Any) -> None:
    """Append one JSON line to the project-local agent log.

    No-op when ``project_root`` is falsy. All exceptions are swallowed
    so a busted log file never fails an executor -- a WARNING is
    emitted to the stdout logger instead.
    """
    if not project_root:
        return
    handler = _handler_for(project_root)
    if handler is None:
        return
    record = logging.LogRecord(
        name="cgx.session.agent_log", level=logging.INFO, pathname=__file__,
        lineno=0, msg=event, args=None, exc_info=None)
    record.agent_event = {"ts": time.time(), "event": event, **fields}
    try:
        handler.emit(record)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("agent_log: emit failed for %s (%s)", event, exc)


def reset_for_tests() -> None:
    """Close + drop every cached handler. Test-only."""
    with _handlers_lock:
        for h in _handlers.values():
            try:
                h.close()
            except Exception:
                pass
        _handlers.clear()
