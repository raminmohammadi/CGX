

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
import os
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

# Session-stable mirror lives under the config dir (``$CGX_CONFIG_DIR`` or
# ``~/.cgx``), keyed by session id, so it survives the project directory
# being regenerated / sent to Trash on a re-scaffold -- the project-local
# ``<project_root>/.cgx/agent.log`` goes with the trashed tree, but the
# mirror does not. A distinct subdir keeps it clear of the chat-session
# JSONL store under ``<config>/sessions``.
_STABLE_SUBDIR = "agent-sessions"

# Handler cache keyed by resolved log-file path so re-opening the same
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


class _QuietRotatingFileHandler(RotatingFileHandler):
    """RotatingFileHandler that never floods stderr on emit failures.

    The stock handler routes emit exceptions to ``logging.Handler.handleError``,
    which (with ``logging.raiseExceptions`` on, the default) prints the full
    traceback to stderr. When a re-scaffold trashes the log directory out from
    under an open handler, every subsequent emit -- one per traced call --
    prints a fresh traceback, drowning the console. Routing the failure to our
    own WARNING logger keeps the diagnostic without the flood; the stale handler
    is rebuilt on the next :func:`_emit_to` via the directory-liveness check.
    """

    def handleError(self, record: logging.LogRecord) -> None:  # noqa: D401
        logger.warning("agent_log: emit failed for %s", getattr(record, "msg", ""))


def _resolve_path(project_root: str) -> Path:
    return Path(project_root).resolve() / _LOG_DIR_NAME / _LOG_FILE_NAME


def _config_dir() -> Path:
    return Path(os.environ.get("CGX_CONFIG_DIR", str(Path.home() / _LOG_DIR_NAME)))


def _stable_path(session_id: str) -> Path:
    return _config_dir() / _STABLE_SUBDIR / session_id / _LOG_FILE_NAME


def _handler_for_path(path: Path) -> Optional[RotatingFileHandler]:
    key = str(path)
    with _handlers_lock:
        h = _handlers.get(key)
        if h is not None:
            return h
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            h = _QuietRotatingFileHandler(
                str(path), maxBytes=_MAX_BYTES,
                backupCount=_BACKUP_COUNT, encoding="utf-8")
            h.setFormatter(_JsonlFormatter())
            h.setLevel(logging.INFO)
            _handlers[key] = h
            return h
        except Exception as exc:
            logger.warning(
                "agent_log: failed to open %s (%s); disabling for this path",
                path, exc)
            return None


def _handler_dir_alive(handler: RotatingFileHandler) -> bool:
    """True when the cached handler's log directory still exists.

    A re-scaffold trashes / regenerates the project tree, taking the
    project-local ``.cgx`` directory (and the open ``agent.log``) with it.
    The cached ``RotatingFileHandler`` still points at that dead path, so
    its next ``emit`` -> ``shouldRollover`` -> ``_open`` raises
    ``FileNotFoundError`` deep inside stdlib logging, where ``handleError``
    swallows it and floods stderr instead of letting our ``try/except``
    react. Checking the parent directory up front lets us rebuild the
    handler (which re-creates the directory) before that happens.
    """
    try:
        return Path(handler.baseFilename).parent.is_dir()
    except Exception:  # pragma: no cover - defensive
        return False


def _emit_to(path: Path, event: str, ts: float, fields: Dict[str, Any]) -> None:
    handler = _handler_for_path(path)
    if handler is None:
        return
    # If the log directory vanished under the cached handler (project tree
    # re-scaffolded / trashed), evict the stale handler so _handler_for_path
    # re-creates the directory and re-opens a fresh file below.
    if not _handler_dir_alive(handler):
        _drop_handler_for(path)
        handler = _handler_for_path(path)
        if handler is None:
            return
    record = logging.LogRecord(
        name="cgx.session.agent_log", level=logging.INFO, pathname=__file__,
        lineno=0, msg=event, args=None, exc_info=None)
    record.agent_event = {"ts": ts, "event": event, **fields}
    try:
        handler.emit(record)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("agent_log: emit failed for %s (%s)", event, exc)


def log_event(project_root: Optional[str], event: str, **fields: Any) -> None:
    """Append one JSON line to the project-local agent log.

    No-op when ``project_root`` is falsy. When a ``session_id`` field is
    present the same line is also mirrored to a session-stable log under
    the config dir (``<config>/agent-sessions/<session_id>/agent.log``) so
    the trail survives the project directory being regenerated / trashed on
    a re-scaffold. Both writes share one timestamp so the lines match
    exactly. All exceptions are swallowed so a busted log file never fails
    an executor -- a WARNING is emitted to the stdout logger instead.
    """
    if not project_root:
        return
    ts = time.time()
    _emit_to(_resolve_path(project_root), event, ts, fields)
    session_id = fields.get("session_id")
    if session_id:
        _emit_to(_stable_path(str(session_id)), event, ts, fields)


def _drop_handler_for(path: Path) -> None:
    """Close + evict the cached rotating handler for ``path`` (if any).

    Releases the OS file descriptor so the file can be unlinked; the next
    :func:`log_event` for that path lazily re-opens a fresh handler.
    """
    key = str(path)
    with _handlers_lock:
        h = _handlers.pop(key, None)
    if h is not None:
        try:
            h.close()
        except Exception:  # pragma: no cover - defensive
            pass


def _unlink_log_file(path: Path) -> bool:
    """Unlink one trace-log file with a hard safety gate.

    Deletion is deliberately narrow so a caller-supplied ``project_root``
    can never be leveraged to remove anything other than a CGX trace log:

    * the basename must be the known log file or one of its rotation
      backups (``agent.log`` / ``agent.log.1`` ...), never anything else;
    * the target must be an existing *regular file*, never a directory;
    * symlinks are refused outright (``lstat``-based check) so a planted
      ``agent.log -> /etc/shadow`` symlink can't redirect the unlink.

    Returns True when a file was removed.
    """
    name = path.name
    allowed = {_LOG_FILE_NAME} | {
        f"{_LOG_FILE_NAME}.{i}" for i in range(1, _BACKUP_COUNT + 1)
    }
    if name not in allowed:
        return False
    try:
        st = path.lstat()
    except (FileNotFoundError, OSError):
        return False
    import stat as _stat
    # Regular files only: reject symlinks (S_ISLNK) and directories.
    if not _stat.S_ISREG(st.st_mode):
        return False
    _drop_handler_for(path)
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("agent_log: unlink failed for %s (%s)", path, e)
        return False


def delete_project_trace_log(project_root: str) -> int:
    """Delete a project's ``<root>/.cgx/agent.log`` (+ rotation backups).

    Only ever touches files literally named ``agent.log`` /
    ``agent.log.<n>`` inside the resolved ``<project_root>/.cgx``
    directory. The filename is a compile-time constant, so even a hostile
    ``project_root`` cannot escape to an arbitrary path via traversal, and
    :func:`_unlink_log_file` refuses symlinks and non-regular files.
    Returns the number of files removed; a no-op (returns 0) when the
    project has no agent log.
    """
    if not project_root:
        return 0
    base = _resolve_path(project_root)
    candidates = [base] + [
        base.with_name(f"{_LOG_FILE_NAME}.{i}")
        for i in range(1, _BACKUP_COUNT + 1)
    ]
    return sum(1 for p in candidates if _unlink_log_file(p))


def stable_trace_log_path(session_id: str) -> Path:
    """Return the session-stable trace mirror for ``session_id``.

    The mirror at ``<config>/agent-sessions/<session_id>/agent.log`` is
    written alongside the project-local log whenever a ``session_id`` is in
    scope (see :func:`log_event`) and survives the project directory being
    regenerated / trashed on a re-scaffold. The trace explorer falls back to
    it when the project-local ``agent.log`` is gone. ``session_id`` is only
    ever an id CGX itself minted (never raw request input); the caller is
    responsible for validating it before using the returned path.
    """
    return _stable_path(str(session_id))


def reset_for_tests() -> None:
    """Close + drop every cached handler. Test-only."""
    with _handlers_lock:
        for h in _handlers.values():
            try:
                h.close()
            except Exception:
                pass
        _handlers.clear()
