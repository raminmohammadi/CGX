

"""SQLite-backed persistence for session state.

One database file holds every session for a given project root. Each
row stores the dataclass as a JSON blob plus a few indexed columns
(session_id, status, timestamps) so common queries don't have to
parse JSON.

Writes go through :meth:`SessionStore.publish` so the in-process
event bus stays in sync with the on-disk state. The store does not
own conflict-resolution between multiple writers; the router (Phase
1+) acquires a per-session asyncio lock before issuing writes.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from cgx.session.events import Event, EventBus, EventType, get_default_bus
from cgx.session.models import (
    Artifact,
    ArtifactKind,
    Decision,
    DecisionKind,
    DecisionLog,
    Fact,
    FactKind,
    KnowledgeBase,
    Session,
    SessionMode,
    SessionStatus,
    TaskKind,
    TaskNode,
    TaskNodeStatus,
)

logger = logging.getLogger(__name__)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    status     TEXT NOT NULL,
    title      TEXT NOT NULL,
    project_root TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    data_json  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    task_id     TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    parent_id   TEXT,
    kind        TEXT NOT NULL,
    status      TEXT NOT NULL,
    created_at  REAL NOT NULL,
    data_json   TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_tasks_session ON tasks(session_id);
CREATE INDEX IF NOT EXISTS idx_tasks_parent  ON tasks(parent_id);

CREATE TABLE IF NOT EXISTS facts (
    fact_id    TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    kind       TEXT NOT NULL,
    stale      INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    data_json  TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_facts_session ON facts(session_id);

CREATE TABLE IF NOT EXISTS decisions (
    decision_id     TEXT PRIMARY KEY,
    session_id      TEXT NOT NULL,
    resolved_task_id TEXT NOT NULL,
    kind            TEXT NOT NULL,
    made_at         REAL NOT NULL,
    data_json       TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_decisions_session ON decisions(session_id);
CREATE INDEX IF NOT EXISTS idx_decisions_task    ON decisions(resolved_task_id);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id        TEXT PRIMARY KEY,
    session_id         TEXT NOT NULL,
    produced_by_task_id TEXT NOT NULL,
    kind               TEXT NOT NULL,
    created_at         REAL NOT NULL,
    data_json          TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_artifacts_session ON artifacts(session_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_task    ON artifacts(produced_by_task_id);
"""


def default_db_path(project_root: Optional[str | Path] = None) -> Path:
    """Where the session DB lives for a given project.

    Project-local at ``<project_root>/.cgx/sessions.db`` when
    ``project_root`` is provided, falling back to user-global at
    ``~/.cgx/sessions.db`` for the bare ``SessionStore()`` case
    (e.g. interactive scripts, tests with a tmp HOME).
    """
    if project_root:
        return Path(project_root) / ".cgx" / "sessions.db"
    return Path.home() / ".cgx" / "sessions.db"


class SessionStore:
    """Thin SQLite wrapper for session aggregates.

    One instance per database path. Connections are reused; reads
    return fresh dataclasses parsed from the stored JSON blob, so
    callers may mutate them without affecting the store.
    """

    def __init__(self, db_path: Optional[str | Path] = None, *,
                 project_root: Optional[str | Path] = None,
                 bus: Optional[EventBus] = None) -> None:
        self._path = Path(db_path) if db_path else default_db_path(project_root)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._bus = bus if bus is not None else get_default_bus()

    @property
    def path(self) -> Path:
        return self._path

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ----------------------- internal helpers -----------------------

    @contextmanager
    def _txn(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def _emit(self, type_: EventType, session_id: str,
              payload: Dict[str, Any]) -> None:
        try:
            self._bus.publish(Event(type=type_, session_id=session_id,
                                    payload=payload))
        except Exception as e:  # pragma: no cover - bus is defensive
            logger.warning("store: publish failed for %s: %s: %s",
                           type_.value, type(e).__name__, e)

    # ----------------------- sessions -----------------------

    def save_session(self, session: Session) -> None:
        session.updated_at = time.time()
        existed = self.get_session(session.session_id) is not None
        # UPSERT instead of ``INSERT OR REPLACE`` because the latter
        # deletes the prior row before re-inserting, which triggers
        # ``ON DELETE CASCADE`` on tasks/facts/decisions/artifacts and
        # wipes the entire session every time we save it.
        with self._txn() as conn:
            conn.execute(
                "INSERT INTO sessions "
                "(session_id, status, title, project_root, created_at, "
                "updated_at, data_json) VALUES (?,?,?,?,?,?,?) "
                "ON CONFLICT(session_id) DO UPDATE SET "
                "status=excluded.status, title=excluded.title, "
                "project_root=excluded.project_root, "
                "updated_at=excluded.updated_at, "
                "data_json=excluded.data_json",
                (session.session_id, session.status.value, session.title,
                 session.project_root, session.created_at,
                 session.updated_at,
                 json.dumps(session.to_dict(), default=str)),
            )
        evt = (EventType.SESSION_UPDATED if existed
               else EventType.SESSION_CREATED)
        self._emit(evt, session.session_id, session.to_dict())

    def get_session(self, session_id: str) -> Optional[Session]:
        with self._lock:
            row = self._conn.execute(
                "SELECT data_json FROM sessions WHERE session_id=?",
                (session_id,)).fetchone()
        if row is None:
            return None
        return _session_from_json(row[0])

    def list_sessions(self, *, project_root: Optional[str] = None,
                      limit: int = 100) -> List[Session]:
        with self._lock:
            if project_root is not None:
                rows = self._conn.execute(
                    "SELECT data_json FROM sessions "
                    "WHERE project_root=? "
                    "ORDER BY updated_at DESC LIMIT ?",
                    (project_root, limit)).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT data_json FROM sessions "
                    "ORDER BY updated_at DESC LIMIT ?",
                    (limit,)).fetchall()
        return [_session_from_json(r[0]) for r in rows]

    def delete_session(self, session_id: str) -> bool:
        with self._txn() as conn:
            cur = conn.execute("DELETE FROM sessions WHERE session_id=?",
                               (session_id,))
            return cur.rowcount > 0

    # ----------------------- tasks -----------------------

    def save_task(self, task: TaskNode) -> None:
        prior = self.get_task(task.task_id)
        with self._txn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO tasks "
                "(task_id, session_id, parent_id, kind, status, "
                "created_at, data_json) VALUES (?,?,?,?,?,?,?)",
                (task.task_id, task.session_id, task.parent_task_id,
                 task.kind.value, task.status.value, task.created_at,
                 json.dumps(task.to_dict(), default=str)),
            )
        if prior is None:
            self._emit(EventType.TASK_CREATED, task.session_id,
                       task.to_dict())
        elif prior.status is not task.status:
            self._emit(EventType.TASK_STATUS_CHANGED, task.session_id,
                       {"task_id": task.task_id,
                        "from": prior.status.value,
                        "to": task.status.value,
                        "task": task.to_dict()})
        if task.status is TaskNodeStatus.DONE:
            self._emit(EventType.TASK_COMPLETED, task.session_id,
                       task.to_dict())
        elif task.status is TaskNodeStatus.FAILED:
            self._emit(EventType.TASK_FAILED, task.session_id,
                       task.to_dict())

    def get_task(self, task_id: str) -> Optional[TaskNode]:
        with self._lock:
            row = self._conn.execute(
                "SELECT data_json FROM tasks WHERE task_id=?",
                (task_id,)).fetchone()
        if row is None:
            return None
        return _task_from_json(row[0])

    def list_tasks(self, session_id: str) -> List[TaskNode]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT data_json FROM tasks WHERE session_id=? "
                "ORDER BY created_at ASC",
                (session_id,)).fetchall()
        return [_task_from_json(r[0]) for r in rows]

    def tasks_by_status(self, session_id: str,
                        status: TaskNodeStatus) -> List[TaskNode]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT data_json FROM tasks "
                "WHERE session_id=? AND status=? "
                "ORDER BY created_at ASC",
                (session_id, status.value)).fetchall()
        return [_task_from_json(r[0]) for r in rows]

    # ----------------------- facts (KB) -----------------------

    def add_fact(self, fact: Fact) -> None:
        with self._txn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO facts "
                "(fact_id, session_id, kind, stale, created_at, "
                "data_json) VALUES (?,?,?,?,?,?)",
                (fact.fact_id, fact.session_id, fact.kind.value,
                 1 if fact.stale else 0, fact.created_at,
                 json.dumps(fact.to_dict(), default=str)),
            )
        self._emit(EventType.FACT_ADDED, fact.session_id, fact.to_dict())

    def mark_facts_stale(self, session_id: str,
                         fact_ids: Iterable[str]) -> int:
        ids = list(fact_ids)
        if not ids:
            return 0
        # Re-serialise the stored blob so ``load_kb`` -- which rebuilds
        # facts from ``data_json`` -- reflects the staleness flip. The
        # indexed ``stale`` column is updated in lockstep.
        n = 0
        now = time.time()
        with self._txn() as conn:
            placeholders = ",".join("?" * len(ids))
            rows = conn.execute(
                f"SELECT fact_id, data_json FROM facts "
                f"WHERE session_id=? AND fact_id IN ({placeholders})",
                (session_id, *ids)).fetchall()
            for fid, blob in rows:
                d = json.loads(blob)
                d["stale"] = True
                d["updated_at"] = now
                conn.execute(
                    "UPDATE facts SET stale=1, data_json=? "
                    "WHERE fact_id=?",
                    (json.dumps(d, default=str), fid))
                n += 1
        if n:
            self._emit(EventType.FACT_STALE, session_id,
                       {"fact_ids": ids, "count": n})
        return n

    def load_kb(self, session_id: str) -> KnowledgeBase:
        with self._lock:
            rows = self._conn.execute(
                "SELECT data_json FROM facts WHERE session_id=? "
                "ORDER BY created_at ASC",
                (session_id,)).fetchall()
        kb = KnowledgeBase(session_id=session_id)
        for r in rows:
            kb.add(_fact_from_json(r[0]))
        return kb

    # ----------------------- decisions -----------------------

    def record_decision(self, decision: Decision) -> None:
        with self._txn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO decisions "
                "(decision_id, session_id, resolved_task_id, kind, "
                "made_at, data_json) VALUES (?,?,?,?,?,?)",
                (decision.decision_id, decision.session_id,
                 decision.resolved_task_id, decision.kind.value,
                 decision.made_at,
                 json.dumps(decision.to_dict(), default=str)),
            )
        self._emit(EventType.DECISION_RECORDED, decision.session_id,
                   decision.to_dict())

    def load_decisions(self, session_id: str) -> DecisionLog:
        with self._lock:
            rows = self._conn.execute(
                "SELECT data_json FROM decisions WHERE session_id=? "
                "ORDER BY made_at ASC",
                (session_id,)).fetchall()
        log = DecisionLog(session_id=session_id)
        for r in rows:
            log.add(_decision_from_json(r[0]))
        return log

    # ----------------------- artifacts -----------------------

    def save_artifact(self, artifact: Artifact) -> None:
        with self._txn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO artifacts "
                "(artifact_id, session_id, produced_by_task_id, kind, "
                "created_at, data_json) VALUES (?,?,?,?,?,?)",
                (artifact.artifact_id, artifact.session_id,
                 artifact.produced_by_task_id, artifact.kind.value,
                 artifact.created_at,
                 json.dumps(artifact.to_dict(), default=str)),
            )
        self._emit(EventType.ARTIFACT_CREATED, artifact.session_id,
                   artifact.to_dict())

    def get_artifact(self, artifact_id: str) -> Optional[Artifact]:
        with self._lock:
            row = self._conn.execute(
                "SELECT data_json FROM artifacts WHERE artifact_id=?",
                (artifact_id,)).fetchone()
        if row is None:
            return None
        return _artifact_from_json(row[0])

    def list_artifacts(self, session_id: str) -> List[Artifact]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT data_json FROM artifacts WHERE session_id=? "
                "ORDER BY created_at ASC",
                (session_id,)).fetchall()
        return [_artifact_from_json(r[0]) for r in rows]


# ----------------------- json -> dataclass parsers -----------------------

def _session_from_json(blob: str) -> Session:
    d = json.loads(blob)
    return Session(
        session_id=d["session_id"],
        title=d["title"],
        original_objective=d["original_objective"],
        status=SessionStatus(d.get("status", "active")),
        mode=SessionMode(d.get("mode", "explore")),
        current_focus=d.get("current_focus"),
        root_task_id=d.get("root_task_id"),
        project_root=d.get("project_root"),
        created_at=float(d.get("created_at") or 0.0),
        updated_at=float(d.get("updated_at") or 0.0),
        max_task_runs=(int(d["max_task_runs"])
                       if d.get("max_task_runs") is not None else None),
        max_wall_seconds=(float(d["max_wall_seconds"])
                          if d.get("max_wall_seconds") is not None else None),
        headless=bool(d.get("headless", False)),
        task_runs=int(d.get("task_runs") or 0),
        first_task_started_at=(float(d["first_task_started_at"])
                               if d.get("first_task_started_at") is not None
                               else None),
    )


def _task_from_json(blob: str) -> TaskNode:
    d = json.loads(blob)
    return TaskNode(
        task_id=d["task_id"],
        session_id=d["session_id"],
        kind=TaskKind(d["kind"]),
        name=d.get("name", ""),
        description=d.get("description", ""),
        parent_task_id=d.get("parent_task_id"),
        status=TaskNodeStatus(d.get("status", "pending")),
        inputs=dict(d.get("inputs") or {}),
        outputs=d.get("outputs"),
        error=d.get("error"),
        blockers=list(d.get("blockers") or []),
        children=list(d.get("children") or []),
        consumed_decision_ids=list(d.get("consumed_decision_ids") or []),
        produced_artifact_id=d.get("produced_artifact_id"),
        created_at=float(d.get("created_at") or 0.0),
        started_at=d.get("started_at"),
        completed_at=d.get("completed_at"),
    )


def _fact_from_json(blob: str) -> Fact:
    d = json.loads(blob)
    return Fact(
        fact_id=d["fact_id"],
        session_id=d["session_id"],
        kind=FactKind(d["kind"]),
        content=dict(d.get("content") or {}),
        surfaced_in_task_id=d.get("surfaced_in_task_id"),
        stale=bool(d.get("stale", False)),
        created_at=float(d.get("created_at") or 0.0),
        updated_at=float(d.get("updated_at") or 0.0),
    )


def _decision_from_json(blob: str) -> Decision:
    d = json.loads(blob)
    return Decision(
        decision_id=d["decision_id"],
        session_id=d["session_id"],
        resolved_task_id=d["resolved_task_id"],
        kind=DecisionKind(d["kind"]),
        question=d.get("question", ""),
        chosen=dict(d.get("chosen") or {}),
        rationale=d.get("rationale"),
        made_at=float(d.get("made_at") or 0.0),
    )


def _artifact_from_json(blob: str) -> Artifact:
    d = json.loads(blob)
    return Artifact(
        artifact_id=d["artifact_id"],
        session_id=d["session_id"],
        produced_by_task_id=d["produced_by_task_id"],
        kind=ArtifactKind(d["kind"]),
        content=dict(d.get("content") or {}),
        created_at=float(d.get("created_at") or 0.0),
    )
