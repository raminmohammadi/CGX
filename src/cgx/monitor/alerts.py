

"""Alert model + a small SQLite-backed alert store.

Alerts are the output of the AIOps monitors in :mod:`cgx.monitor.checks`.
They are persisted so the (later) admin page can list recent quality/cost
incidents, and mirrored to the metrics registry so they also show up on the
Prometheus scrape. The store mirrors :class:`cgx.session.store.SessionStore`'s
conventions (one sqlite file, WAL, path-injection barrier) but is deliberately
standalone -- monitors run outside any session too.
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

# Ordered by increasing urgency so callers can compare / filter.
SEVERITIES = ("info", "warning", "critical")


@dataclass
class Alert:
    """One monitor finding. ``value``/``threshold`` are optional numerics."""

    code: str
    severity: str
    message: str
    value: Optional[float] = None
    threshold: Optional[float] = None
    run_id: Optional[str] = None
    labels: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    alert_id: str = field(default_factory=lambda: "alert_" + uuid.uuid4().hex[:16])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
            "run_id": self.run_id,
            "labels": dict(self.labels),
            "created_at": self.created_at,
        }


_SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    alert_id   TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    code       TEXT NOT NULL,
    severity   TEXT NOT NULL,
    run_id     TEXT,
    value      REAL,
    threshold  REAL,
    message    TEXT NOT NULL,
    labels_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at);
CREATE INDEX IF NOT EXISTS idx_alerts_code    ON alerts(code);
"""


def default_alert_db_path(project_root: Optional[str | Path] = None) -> Path:
    """Project-local ``<project_root>/.cgx/monitor.db`` when a root is given,
    else the user-global config dir (``$CGX_CONFIG_DIR`` or ``~/.cgx``)."""
    if project_root:
        root = os.path.realpath(os.fspath(project_root))
        return Path(root) / ".cgx" / "monitor.db"
    base = os.environ.get("CGX_CONFIG_DIR") or str(Path.home() / ".cgx")
    return Path(base) / "monitor.db"


class AlertStore:
    """Thin SQLite wrapper for persisted alerts."""

    def __init__(self, db_path: Optional[str | Path] = None, *,
                 project_root: Optional[str | Path] = None) -> None:
        if db_path is not None and os.fspath(db_path) == ":memory:":
            connect_target = ":memory:"
            self._path = Path(":memory:")
        else:
            raw = Path(db_path) if db_path else default_alert_db_path(project_root)
            resolved = os.path.realpath(os.fspath(raw))
            if not resolved.startswith(os.sep):
                raise ValueError(f"alert DB path is not absolute: {raw!r}")
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

    def record(self, alert: Alert) -> str:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO alerts (alert_id, created_at, code, "
                "severity, run_id, value, threshold, message, labels_json) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (alert.alert_id, alert.created_at, alert.code, alert.severity,
                 alert.run_id, alert.value, alert.threshold, alert.message,
                 json.dumps(alert.labels)),
            )
            self._conn.commit()
        return alert.alert_id

    def recent(self, *, limit: int = 100, severity: Optional[str] = None,
               code: Optional[str] = None, since: Optional[float] = None,
               ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        if severity:
            clauses.append("severity = ?")
            params.append(severity)
        if code:
            clauses.append("code = ?")
            params.append(code)
        if since is not None:
            clauses.append("created_at >= ?")
            params.append(float(since))
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = ("SELECT alert_id, created_at, code, severity, run_id, value, "
               "threshold, message, labels_json FROM alerts" + where +
               " ORDER BY created_at DESC LIMIT ?")
        params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        cols = ("alert_id", "created_at", "code", "severity", "run_id",
                "value", "threshold", "message", "labels_json")
        out: List[Dict[str, Any]] = []
        for r in rows:
            d = dict(zip(cols, r, strict=False))
            d["labels"] = json.loads(d.pop("labels_json") or "{}")
            out.append(d)
        return out

    # --- data lifecycle (Subsystem M): retention + right-to-erasure --------
    def purge(self, *, before: float) -> int:
        """Delete alerts recorded before ``before`` (epoch); row count."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM alerts WHERE created_at < ?", (float(before),))
            self._conn.commit()
            return int(cur.rowcount)

    def delete_run(self, run_id: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM alerts WHERE run_id = ?", (run_id,))
            self._conn.commit()
            return int(cur.rowcount)
