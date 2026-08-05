

"""Feedback model + a small SQLite-backed feedback store.

User signals (thumbs up/down + optional comment) on an ``ask`` answer or a
``plan`` result, joined to the provenance keys minted by Subsystem F
(``run_id`` + ``prompt_version`` + ``model``) so a rating can be tied back to
the exact execution that produced it. The store mirrors
:class:`cgx.monitor.alerts.AlertStore`'s conventions (one sqlite file, WAL,
path-injection barrier, ``$CGX_CONFIG_DIR``-aware default path).
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

RATINGS = ("up", "down")


@dataclass
class Feedback:
    """One user rating. ``rating`` is ``"up"``/``"down"``; the rest is context."""

    rating: str
    run_id: Optional[str] = None
    session_id: Optional[str] = None
    kind: str = "ask"  # "ask" | "plan"
    comment: str = ""
    question: str = ""
    answer_preview: str = ""
    model: Optional[str] = None
    prompt_version: Optional[str] = None
    labels: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    feedback_id: str = field(
        default_factory=lambda: "fb_" + uuid.uuid4().hex[:16])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feedback_id": self.feedback_id,
            "created_at": self.created_at,
            "rating": self.rating,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "kind": self.kind,
            "comment": self.comment,
            "question": self.question,
            "answer_preview": self.answer_preview,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "labels": dict(self.labels),
        }


_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id    TEXT PRIMARY KEY,
    created_at     REAL NOT NULL,
    rating         TEXT NOT NULL,
    run_id         TEXT,
    session_id     TEXT,
    kind           TEXT NOT NULL,
    comment        TEXT NOT NULL,
    question       TEXT NOT NULL,
    answer_preview TEXT NOT NULL,
    model          TEXT,
    prompt_version TEXT,
    labels_json    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feedback_created ON feedback(created_at);
CREATE INDEX IF NOT EXISTS idx_feedback_run     ON feedback(run_id);
CREATE INDEX IF NOT EXISTS idx_feedback_rating  ON feedback(rating);
"""

_COLS = ("feedback_id", "created_at", "rating", "run_id", "session_id",
         "kind", "comment", "question", "answer_preview", "model",
         "prompt_version", "labels_json")


def default_feedback_db_path(project_root: Optional[str | Path] = None) -> Path:
    """Project-local ``<project_root>/.cgx/feedback.db`` when a root is given,
    else the user-global config dir (``$CGX_CONFIG_DIR`` or ``~/.cgx``)."""
    if project_root:
        root = os.path.realpath(os.fspath(project_root))
        return Path(root) / ".cgx" / "feedback.db"
    base = os.environ.get("CGX_CONFIG_DIR") or str(Path.home() / ".cgx")
    return Path(base) / "feedback.db"


class FeedbackStore:
    """Thin SQLite wrapper for persisted user feedback."""

    def __init__(self, db_path: Optional[str | Path] = None, *,
                 project_root: Optional[str | Path] = None) -> None:
        if db_path is not None and os.fspath(db_path) == ":memory:":
            connect_target = ":memory:"
            self._path = Path(":memory:")
        else:
            raw = Path(db_path) if db_path else default_feedback_db_path(project_root)
            resolved = os.path.realpath(os.fspath(raw))
            if not resolved.startswith(os.sep):
                raise ValueError(f"feedback DB path is not absolute: {raw!r}")
            self._path = Path(resolved)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            connect_target = str(self._path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(connect_target, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    @property
    def path(self) -> Path:
        return self._path

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def record(self, fb: Feedback) -> str:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO feedback (feedback_id, created_at, "
                "rating, run_id, session_id, kind, comment, question, "
                "answer_preview, model, prompt_version, labels_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (fb.feedback_id, fb.created_at, fb.rating, fb.run_id,
                 fb.session_id, fb.kind, fb.comment, fb.question,
                 fb.answer_preview, fb.model, fb.prompt_version,
                 json.dumps(fb.labels)),
            )
            self._conn.commit()
        return fb.feedback_id

    def recent(self, *, limit: int = 100, rating: Optional[str] = None,
               kind: Optional[str] = None, run_id: Optional[str] = None,
               since: Optional[float] = None) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        if rating:
            clauses.append("rating = ?")
            params.append(rating)
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        if run_id:
            clauses.append("run_id = ?")
            params.append(run_id)
        if since is not None:
            clauses.append("created_at >= ?")
            params.append(float(since))
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = ("SELECT " + ", ".join(_COLS) + " FROM feedback" + where +
               " ORDER BY created_at DESC LIMIT ?")
        params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            d = dict(zip(_COLS, r, strict=False))
            d["labels"] = json.loads(d.pop("labels_json") or "{}")
            out.append(d)
        return out

    # --- data lifecycle (Subsystem M): retention + right-to-erasure --------
    def purge(self, *, before: float) -> int:
        """Delete feedback recorded before ``before`` (epoch); row count."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM feedback WHERE created_at < ?", (float(before),))
            self._conn.commit()
            return int(cur.rowcount)

    def delete_run(self, run_id: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM feedback WHERE run_id = ?", (run_id,))
            self._conn.commit()
            return int(cur.rowcount)

    def stats(self, *, since: Optional[float] = None) -> Dict[str, Any]:
        """Aggregate up/down counts (overall and per kind) for dashboards."""
        where, params = "", []
        if since is not None:
            where, params = " WHERE created_at >= ?", [float(since)]
        with self._lock:
            rows = self._conn.execute(
                "SELECT kind, rating, COUNT(*) FROM feedback" + where +
                " GROUP BY kind, rating", params).fetchall()
        by_kind: Dict[str, Dict[str, int]] = {}
        up = down = 0
        for kind, rating, n in rows:
            by_kind.setdefault(kind, {"up": 0, "down": 0})[rating] = int(n)
            if rating == "up":
                up += int(n)
            elif rating == "down":
                down += int(n)
        total = up + down
        return {"total": total, "up": up, "down": down,
                "satisfaction": (up / total) if total else None,
                "by_kind": by_kind}


# --- process-wide default store -------------------------------------------
# The webui feedback route writes through this singleton so it shares one
# feedback DB (``~/.cgx/feedback.db``), mirroring cgx.monitor's default.
_DEFAULT: Optional[FeedbackStore] = None
_DEFAULT_LOCK = threading.Lock()


def get_default_store() -> FeedbackStore:
    """Return the lazily-constructed process-wide :class:`FeedbackStore`."""
    global _DEFAULT
    if _DEFAULT is None:
        with _DEFAULT_LOCK:
            if _DEFAULT is None:
                _DEFAULT = FeedbackStore()
    return _DEFAULT
