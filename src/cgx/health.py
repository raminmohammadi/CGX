

"""Liveness and readiness checks backing ``/healthz`` and ``/readyz``.

Two Kubernetes-style probes, deliberately split by cost and meaning:

- **liveness** (:func:`liveness`): the process is up and the event loop is
  responsive. It never touches an external system or the disk, so a
  transient provider outage or a slow volume can't trigger a restart loop.
- **readiness** (:func:`readiness`): the subsystems required to *serve*
  traffic are usable -- the config dir is writable and the SQLite driver +
  session-DB parent dir accept a connection. Provider reachability and
  index presence are *reported* but do **not** gate readiness: the UI,
  ``/api/status`` and other read-only surfaces stay serviceable when a
  model backend is down or no index has been built yet.

Every check returns a small JSON-safe dict and never echoes secrets or raw
exception text (only the exception *type*), matching the redaction rules the
rest of the web layer already follows.
"""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from cgx import metrics as _metrics


def _config_dir() -> Path:
    return Path(os.environ.get("CGX_CONFIG_DIR", str(Path.home() / ".cgx")))


def _result(name: str, ok: bool, *, critical: bool,
            detail: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {"name": name, "ok": bool(ok), "critical": bool(critical),
            "detail": detail or {}}


def check_config_dir() -> Dict[str, Any]:
    """Critical: the CGX config dir exists (or can be created) and is writable."""
    path = _config_dir()
    try:
        path.mkdir(parents=True, exist_ok=True)
        writable = os.access(path, os.W_OK)
        return _result("config_dir", writable, critical=True,
                       detail={"path": str(path), "writable": writable})
    except Exception as e:  # pragma: no cover - defensive, hard to trigger
        return _result("config_dir", False, critical=True,
                       detail={"path": str(path), "error": type(e).__name__})


def check_session_db(project_root: Optional[str] = None) -> Dict[str, Any]:
    """Critical: SQLite works and the session-DB path is reachable.

    Opens the DB read-only when it already exists (no side effects on a
    probe); otherwise validates the driver against an in-memory DB and
    confirms the parent dir is writable so the store can be created lazily.
    """
    try:
        from cgx.session.store import default_db_path
        db_path = Path(default_db_path(project_root))
    except Exception as e:
        return _result("session_db", False, critical=True,
                       detail={"error": type(e).__name__})
    try:
        exists = db_path.exists()
        if exists:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        else:
            conn = sqlite3.connect(":memory:")
        try:
            conn.execute("SELECT 1")
        finally:
            conn.close()
        parent = db_path.parent
        writable = os.access(parent, os.W_OK) if parent.exists() else True
        ok = exists or writable
        return _result("session_db", ok, critical=True,
                       detail={"exists": exists, "writable": writable})
    except Exception as e:
        return _result("session_db", False, critical=True,
                       detail={"error": type(e).__name__})


def check_provider(base_url: Optional[str] = None) -> Dict[str, Any]:
    """Informational: best-effort Ollama reachability (never gates readiness)."""
    try:
        from cgx.answer import ollama_discovery
        health = (ollama_discovery.health_check(base_url) if base_url
                  else ollama_discovery.health_check())
        return _result("provider", bool(health.get("ok")), critical=False,
                       detail={"base_url": health.get("base_url"),
                               "models_count": health.get("models_count")})
    except Exception as e:
        return _result("provider", False, critical=False,
                       detail={"error": type(e).__name__})


def check_index(index_dir: Optional[str] = None) -> Dict[str, Any]:
    """Informational: whether a *completed* index (meta.json) is present."""
    idir = index_dir or os.environ.get("CGX_INDEX_DIR") or "/tmp/cgx_index/indices"
    meta = os.path.join(idir, "meta.json")
    present = os.path.isfile(meta)
    return _result("index", present, critical=False,
                   detail={"index_dir": idir, "present": present})


def liveness() -> Dict[str, Any]:
    """Cheap, dependency-free liveness signal."""
    return {"status": "ok", "checks": []}


def readiness(*, project_root: Optional[str] = None,
              base_url: Optional[str] = None,
              index_dir: Optional[str] = None) -> Dict[str, Any]:
    """Aggregate readiness; ``ready`` is True only if every *critical* check passes."""
    checks: List[Dict[str, Any]] = [
        check_config_dir(),
        check_session_db(project_root),
        check_provider(base_url),
        check_index(index_dir),
    ]
    ready = all(c["ok"] for c in checks if c["critical"])
    try:  # surface the outcome as a gauge for the /api/metrics scrape
        _metrics.set_gauge("cgx_ready", 1.0 if ready else 0.0,
                           help="1 when all critical readiness checks pass.")
    except Exception:  # pragma: no cover - metrics must never break a probe
        pass
    return {"status": "ready" if ready else "not_ready", "ready": ready,
            "ts": time.time(), "checks": checks}
