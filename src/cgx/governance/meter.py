

"""SQLite-backed per-owner usage meter.

Every governed LLM call appends one row (owner, UTC day, tokens, cost) so the
:class:`~cgx.governance.manager.QuotaManager` can aggregate a day's spend for
budget checks and the usage-meter API can report per-owner totals. Mirrors the
store conventions used by :class:`cgx.monitor.alerts.AlertStore` (one sqlite
file, WAL, path-injection barrier, ``$CGX_CONFIG_DIR``-aware default path).
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    day        TEXT NOT NULL,
    owner      TEXT NOT NULL,
    model      TEXT,
    provider   TEXT,
    tokens_in  INTEGER NOT NULL,
    tokens_out INTEGER NOT NULL,
    cost_usd   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_usage_owner_day ON usage(owner, day);
"""


def today() -> str:
    """UTC day bucket (``YYYY-MM-DD``) used as the rolling budget window."""
    return time.strftime("%Y-%m-%d", time.gmtime())


def default_usage_db_path(project_root: Optional[str | Path] = None) -> Path:
    """Project-local ``<root>/.cgx/usage.db`` or the user-global config dir."""
    if project_root:
        root = os.path.realpath(os.fspath(project_root))
        return Path(root) / ".cgx" / "usage.db"
    base = os.environ.get("CGX_CONFIG_DIR") or str(Path.home() / ".cgx")
    return Path(base) / "usage.db"


class UsageMeter:
    """Thin SQLite wrapper recording and aggregating per-owner LLM usage."""

    def __init__(self, db_path: Optional[str | Path] = None, *,
                 project_root: Optional[str | Path] = None) -> None:
        if db_path is not None and os.fspath(db_path) == ":memory:":
            connect_target = ":memory:"
            self._path = Path(":memory:")
        else:
            raw = Path(db_path) if db_path else default_usage_db_path(project_root)
            resolved = os.path.realpath(os.fspath(raw))
            if not resolved.startswith(os.sep):
                raise ValueError(f"usage DB path is not absolute: {raw!r}")
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

    def record(self, owner: str, *, tokens_in: int, tokens_out: int,
               cost_usd: float, model: Optional[str] = None,
               provider: Optional[str] = None,
               day: Optional[str] = None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO usage (created_at, day, owner, model, provider, "
                "tokens_in, tokens_out, cost_usd) VALUES (?,?,?,?,?,?,?,?)",
                (time.time(), day or today(), owner, model, provider,
                 int(tokens_in), int(tokens_out), float(cost_usd)),
            )
            self._conn.commit()

    def totals(self, owner: str, *, day: Optional[str] = None) -> Dict[str, Any]:
        """Aggregate one owner's usage for ``day`` (defaults to today)."""
        d = day or today()
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(tokens_in),0), COALESCE(SUM(tokens_out),0), "
                "COALESCE(SUM(cost_usd),0.0), COUNT(*) FROM usage "
                "WHERE owner = ? AND day = ?", (owner, d)).fetchone()
        tin, tout, cost, calls = row
        return {"owner": owner, "day": d, "tokens_in": int(tin),
                "tokens_out": int(tout), "tokens_total": int(tin) + int(tout),
                "cost_usd": round(float(cost), 6), "calls": int(calls)}

    def owners(self, *, day: Optional[str] = None) -> List[str]:
        d = day or today()
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT owner FROM usage WHERE day = ? ORDER BY owner",
                (d,)).fetchall()
        return [r[0] for r in rows]

    def summary(self, *, day: Optional[str] = None) -> List[Dict[str, Any]]:
        """Per-owner totals for ``day`` (for the usage dashboard)."""
        return [self.totals(o, day=day) for o in self.owners(day=day)]
