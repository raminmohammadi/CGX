"""Per-run observation store powering the User Activity page (Subsystem C).

Every ask/plan run appends one :class:`RunRecord` -- the provenance keys
(``run_id``/``model``/``prompt_version``), the grounding signals already
computed by the answer pipeline (sources/citations/confidence), and the
token/cost/latency accounting -- so the activity page can list a user's runs
and join each back to its feedback (Subsystem H) and monitor alerts
(Subsystem G). Mirrors :class:`cgx.monitor.alerts.AlertStore`'s conventions:
one sqlite file, WAL, path-injection barrier, ``$CGX_CONFIG_DIR``-aware path.
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

KINDS = ("ask", "plan", "agent")

_COLS = ("run_id", "created_at", "kind", "model", "prompt_version", "owner",
         "project_root", "tokens_in", "tokens_out", "tokens_total", "cost_usd",
         "latency_ms", "n_sources", "n_citations", "confidence", "grounded",
         "status", "question", "labels_json")


@dataclass
class RunRecord:
    """One observed ask/plan run. Numeric fields are best-effort / nullable."""

    kind: str
    run_id: str = ""
    model: Optional[str] = None
    prompt_version: Optional[str] = None
    owner: Optional[str] = None
    project_root: Optional[str] = None
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    tokens_total: Optional[int] = None
    cost_usd: Optional[float] = None
    latency_ms: Optional[float] = None
    n_sources: int = 0
    n_citations: int = 0
    confidence: Optional[float] = None
    grounded: Optional[bool] = None
    status: str = "ok"
    question: str = ""
    labels: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.run_id:
            self.run_id = "run_" + uuid.uuid4().hex[:16]

    def to_dict(self) -> Dict[str, Any]:
        d = {c: getattr(self, c) for c in _COLS if c != "labels_json"}
        d["labels"] = dict(self.labels)
        return d


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    created_at  REAL NOT NULL,
    kind        TEXT NOT NULL,
    model       TEXT,
    prompt_version TEXT,
    owner       TEXT,
    project_root TEXT,
    tokens_in   INTEGER,
    tokens_out  INTEGER,
    tokens_total INTEGER,
    cost_usd    REAL,
    latency_ms  REAL,
    n_sources   INTEGER,
    n_citations INTEGER,
    confidence  REAL,
    grounded    INTEGER,
    status      TEXT NOT NULL,
    question    TEXT,
    labels_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at);
CREATE INDEX IF NOT EXISTS idx_runs_owner   ON runs(owner);
CREATE INDEX IF NOT EXISTS idx_runs_kind    ON runs(kind);
"""


def default_run_db_path(project_root: Optional[str | Path] = None) -> Path:
    """Project-local ``<root>/.cgx/activity.db`` or the user-global config dir."""
    if project_root:
        root = os.path.realpath(os.fspath(project_root))
        return Path(root) / ".cgx" / "activity.db"
    base = os.environ.get("CGX_CONFIG_DIR") or str(Path.home() / ".cgx")
    return Path(base) / "activity.db"


class RunStore:
    """Thin SQLite wrapper for persisted per-run observation records."""

    def __init__(self, db_path: Optional[str | Path] = None, *,
                 project_root: Optional[str | Path] = None) -> None:
        if db_path is not None and os.fspath(db_path) == ":memory:":
            connect_target = ":memory:"
            self._path = Path(":memory:")
        else:
            raw = Path(db_path) if db_path else default_run_db_path(project_root)
            resolved = os.path.realpath(os.fspath(raw))
            if not resolved.startswith(os.sep):
                raise ValueError(f"activity DB path is not absolute: {raw!r}")
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

    def record(self, rec: RunRecord) -> str:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO runs (" + ", ".join(_COLS) + ") "
                "VALUES (" + ",".join("?" * len(_COLS)) + ")",
                (rec.run_id, rec.created_at, rec.kind, rec.model,
                 rec.prompt_version, rec.owner, rec.project_root, rec.tokens_in,
                 rec.tokens_out, rec.tokens_total, rec.cost_usd, rec.latency_ms,
                 rec.n_sources, rec.n_citations, rec.confidence,
                 (None if rec.grounded is None else int(rec.grounded)),
                 rec.status, rec.question, json.dumps(rec.labels)),
            )
            self._conn.commit()
        return rec.run_id

    def recent(self, *, limit: int = 100, kind: Optional[str] = None,
               owner: Optional[str] = None, status: Optional[str] = None,
               since: Optional[float] = None) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        for col, val in (("kind", kind), ("owner", owner), ("status", status)):
            if val:
                clauses.append(f"{col} = ?")
                params.append(val)
        if since is not None:
            clauses.append("created_at >= ?")
            params.append(float(since))
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = ("SELECT " + ", ".join(_COLS) + " FROM runs" + where +
               " ORDER BY created_at DESC LIMIT ?")
        params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_dict(r) for r in rows]

    def get(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT " + ", ".join(_COLS) + " FROM runs WHERE run_id = ?",
                (run_id,)).fetchone()
        return _row_to_dict(row) if row else None

    def summary(self, *, since: Optional[float] = None) -> Dict[str, Any]:
        """Aggregate run counts + cost/token totals (overall and per kind)."""
        where, params = "", []
        if since is not None:
            where, params = " WHERE created_at >= ?", [float(since)]
        with self._lock:
            rows = self._conn.execute(
                "SELECT kind, COUNT(*), COALESCE(SUM(cost_usd),0), "
                "COALESCE(SUM(tokens_total),0), "
                "SUM(CASE WHEN status='ok' THEN 0 ELSE 1 END), "
                "COALESCE(SUM(latency_ms),0) "
                "FROM runs" + where + " GROUP BY kind", params).fetchall()
                
            model_rows = self._conn.execute(
                "SELECT model, COUNT(*), COALESCE(SUM(tokens_total),0), COALESCE(SUM(latency_ms),0) "
                "FROM runs" + where + " WHERE model IS NOT NULL GROUP BY model", params).fetchall()
                
        by_kind: Dict[str, Dict[str, Any]] = {}
        total = errors = tokens = 0
        cost = 0.0
        total_latency = 0.0
        for kind, n, c, tk, err, lat in rows:
            by_kind[kind] = {"runs": int(n), "cost_usd": round(float(c), 6),
                             "tokens_total": int(tk), "errors": int(err or 0)}
            total += int(n)
            cost += float(c)
            tokens += int(tk)
            errors += int(err or 0)
            total_latency += float(lat)
            
        by_model: Dict[str, Dict[str, Any]] = {}
        for m_model, m_n, m_tk, m_lat in model_rows:
            tps = (int(m_tk) / (float(m_lat) / 1000.0)) if float(m_lat) > 0 else 0.0
            by_model[m_model] = {
                "runs": int(m_n),
                "tokens_total": int(m_tk),
                "tps": round(tps, 2)
            }
            
        overall_tps = (tokens / (total_latency / 1000.0)) if total_latency > 0 else 0.0
        
        return {"total": total, "cost_usd": round(cost, 6),
                "tokens_total": tokens, "errors": errors, "by_kind": by_kind,
                "by_model": by_model, "overall_tps": round(overall_tps, 2)}

    # --- data lifecycle (Subsystem M): retention + right-to-erasure --------
    def purge(self, *, before: float) -> int:
        """Delete runs recorded before ``before`` (epoch seconds); row count."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM runs WHERE created_at < ?", (float(before),))
            self._conn.commit()
            return int(cur.rowcount)

    def delete_run(self, run_id: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM runs WHERE run_id = ?", (run_id,))
            self._conn.commit()
            return int(cur.rowcount)

    def delete_owner(self, owner: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM runs WHERE owner = ?", (owner,))
            self._conn.commit()
            return int(cur.rowcount)


def _row_to_dict(row: Any) -> Dict[str, Any]:
    d = dict(zip(_COLS, row, strict=False))
    d["labels"] = json.loads(d.pop("labels_json") or "{}")
    if d.get("grounded") is not None:
        d["grounded"] = bool(d["grounded"])
    return d


# --- process-wide default store (mirrors the monitor/feedback singletons) ---
_DEFAULT: Optional[RunStore] = None
_DEFAULT_LOCK = threading.Lock()


def get_default_run_store() -> RunStore:
    """Return the lazily-constructed process-wide :class:`RunStore`."""
    global _DEFAULT
    if _DEFAULT is None:
        with _DEFAULT_LOCK:
            if _DEFAULT is None:
                _DEFAULT = RunStore()
    return _DEFAULT
