"""Lightweight SQLite-backed task registry.

Each SSE request (ask, plan, agent, index) registers a task here before
streaming begins.  The registry:

* Assigns a stable task_id and records the request type + status.
* Accumulates every emitted event so the frontend can replay them when
  the user switches tabs and comes back.
* Exposes a per-task ``threading.Event`` that routes can set to signal
  cancellation to the streaming generator.

A single WAL-mode SQLite file lives at ``~/.cgx/tasks.db``.  Reads and
writes are serialised through a module-level ``threading.Lock`` so multiple
uvicorn worker threads never contend on the same connection.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DB_PATH = Path.home() / ".cgx" / "tasks.db"
_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None

# In-memory cancel events — not persisted, intentionally reset on server
# restart (outstanding tasks from a previous run can't be cancelled anyway).
_cancel_events: Dict[str, threading.Event] = {}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA synchronous=NORMAL")
        _init_schema(_conn)
        logger.info("task_store: opened %s", _DB_PATH)
    return _conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS tasks (
            id          TEXT PRIMARY KEY,
            type        TEXT NOT NULL,
            status      TEXT NOT NULL DEFAULT 'pending',
            created_at  REAL NOT NULL,
            started_at  REAL,
            ended_at    REAL,
            request_json TEXT,
            error       TEXT
        );

        CREATE TABLE IF NOT EXISTS task_events (
            rowid       INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id     TEXT NOT NULL,
            event_type  TEXT NOT NULL,
            payload_json TEXT,
            at          REAL NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_task_events_task_id
            ON task_events (task_id);
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_task(task_type: str, request_data: Optional[Dict[str, Any]] = None) -> str:
    """Register a new task and return its generated id."""
    import uuid
    task_id = uuid.uuid4().hex
    now = time.time()
    _cancel_events[task_id] = threading.Event()
    with _lock:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO tasks (id, type, status, created_at, request_json) "
            "VALUES (?, ?, 'running', ?, ?)",
            (task_id, task_type, now,
             json.dumps(request_data, default=str) if request_data else None),
        )
        conn.commit()
    logger.info("task_store: created task id=%s type=%s", task_id, task_type)
    return task_id


def finish_task(task_id: str, *, error: Optional[str] = None) -> None:
    """Mark task as done or failed."""
    status = "failed" if error else "done"
    now = time.time()
    with _lock:
        conn = _get_conn()
        conn.execute(
            "UPDATE tasks SET status=?, ended_at=?, error=? WHERE id=?",
            (status, now, error, task_id),
        )
        conn.commit()
    _cancel_events.pop(task_id, None)
    logger.info("task_store: finished task id=%s status=%s", task_id, status)


def cancel_task(task_id: str) -> bool:
    """Signal cancellation; returns True if the task existed."""
    ev = _cancel_events.get(task_id)
    if ev is None:
        return False
    ev.set()
    now = time.time()
    with _lock:
        conn = _get_conn()
        conn.execute(
            "UPDATE tasks SET status='cancelled', ended_at=? WHERE id=? AND status='running'",
            (now, task_id),
        )
        conn.commit()
    logger.info("task_store: cancelled task id=%s", task_id)
    return True


def get_cancel_event(task_id: str) -> Optional[threading.Event]:
    return _cancel_events.get(task_id)


def append_event(task_id: str, event_type: str, payload: Any) -> None:
    """Persist an SSE event for later replay."""
    now = time.time()
    with _lock:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO task_events (task_id, event_type, payload_json, at) "
            "VALUES (?, ?, ?, ?)",
            (task_id, event_type,
             json.dumps(payload, default=str) if payload is not None else None,
             now),
        )
        conn.commit()


def get_task(task_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        conn = _get_conn()
        row = conn.execute(
            "SELECT id, type, status, created_at, started_at, ended_at, error "
            "FROM tasks WHERE id=?",
            (task_id,),
        ).fetchone()
    if row is None:
        return None
    cols = ("id", "type", "status", "created_at", "started_at", "ended_at", "error")
    return dict(zip(cols, row))


def get_task_events(task_id: str) -> List[Dict[str, Any]]:
    """Return all recorded events for a task (for frontend replay)."""
    with _lock:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT event_type, payload_json, at FROM task_events "
            "WHERE task_id=? ORDER BY rowid",
            (task_id,),
        ).fetchall()
    out = []
    for ev_type, payload_json, at in rows:
        try:
            payload = json.loads(payload_json) if payload_json else {}
        except Exception:
            payload = {}
        out.append({"type": ev_type, "payload": payload, "at": at})
    return out


def list_tasks(limit: int = 50) -> List[Dict[str, Any]]:
    """Most-recent tasks first."""
    with _lock:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT id, type, status, created_at, ended_at, error "
            "FROM tasks ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    cols = ("id", "type", "status", "created_at", "ended_at", "error")
    return [dict(zip(cols, r)) for r in rows]


def prune_old_tasks(max_age_seconds: float = 3600 * 24) -> int:
    """Delete task records older than ``max_age_seconds``. Returns rows removed."""
    cutoff = time.time() - max_age_seconds
    with _lock:
        conn = _get_conn()
        ids = [r[0] for r in conn.execute(
            "SELECT id FROM tasks WHERE created_at < ?", (cutoff,)
        ).fetchall()]
        if ids:
            placeholders = ",".join("?" * len(ids))
            conn.execute(f"DELETE FROM task_events WHERE task_id IN ({placeholders})", ids)
            conn.execute(f"DELETE FROM tasks WHERE id IN ({placeholders})", ids)
            conn.commit()
    return len(ids)
