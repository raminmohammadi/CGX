

"""Persistent chat sessions for CGX.

Sessions are stored as append-only JSONL files under
``~/.cgx/sessions/<session_id>.jsonl``. Each line is a single message
object ``{"role": "user|assistant|system", "content": str, "at": float,
"meta": {...}}``. The session header (title, created_at) lives in
``index.json`` alongside the message files.

The module is dependency-free (stdlib only) and safe to call from the
web UI on every interaction; writes are atomic via ``os.replace``.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


logger = logging.getLogger(__name__)

CONFIG_DIR = Path(os.environ.get("CGX_CONFIG_DIR", str(Path.home() / ".cgx")))
SESSIONS_DIR = CONFIG_DIR / "sessions"
INDEX_PATH = SESSIONS_DIR / "index.json"


@dataclass
class SessionMeta:
    """Header record for a single session, persisted in ``index.json``."""

    id: str
    title: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    message_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _ensure_dir() -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def _load_index() -> Dict[str, Any]:
    if not INDEX_PATH.exists():
        return {"sessions": []}
    try:
        with INDEX_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("sessions"), list):
            return data
    except Exception as e:
        logger.warning("sessions: failed to load index %s: %s: %s",
                       INDEX_PATH, type(e).__name__, e)
    return {"sessions": []}


def _save_index(data: Dict[str, Any]) -> None:
    _ensure_dir()
    tmp = INDEX_PATH.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, INDEX_PATH)


def _path_for(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.jsonl"


def list_sessions() -> List[SessionMeta]:
    """Return all known sessions, most-recently-updated first."""
    data = _load_index()
    out: List[SessionMeta] = []
    for s in data.get("sessions", []):
        try:
            out.append(SessionMeta(
                id=str(s["id"]),
                title=str(s.get("title") or ""),
                created_at=float(s.get("created_at") or 0.0),
                updated_at=float(s.get("updated_at") or 0.0),
                message_count=int(s.get("message_count") or 0),
            ))
        except Exception as e:
            logger.warning("sessions: skipping malformed index entry %r: %s: %s",
                           s.get("id"), type(e).__name__, e)
            continue
    out.sort(key=lambda m: m.updated_at, reverse=True)
    return out


def create_session(title: str = "") -> SessionMeta:
    """Create a new empty session and return its metadata."""
    _ensure_dir()
    sid = uuid.uuid4().hex[:12]
    meta = SessionMeta(id=sid, title=(title or "Untitled"))
    _path_for(sid).touch()
    data = _load_index()
    data["sessions"].append(meta.to_dict())
    _save_index(data)
    return meta


def delete_session(session_id: str) -> bool:
    """Remove a session and its message log; returns True on success."""
    data = _load_index()
    before = len(data["sessions"])
    data["sessions"] = [s for s in data["sessions"] if s.get("id") != session_id]
    if len(data["sessions"]) == before:
        return False
    _save_index(data)
    try:
        _path_for(session_id).unlink(missing_ok=True)
    except Exception as e:
        logger.warning("sessions: failed to delete log for %s: %s: %s",
                       session_id, type(e).__name__, e)
    return True


def append_message(session_id: str, role: str, content: str,
                   meta: Optional[Dict[str, Any]] = None) -> None:
    """Append a single message to a session log."""
    _ensure_dir()
    if not _path_for(session_id).exists():
        raise FileNotFoundError(f"session {session_id} does not exist")
    record = {"role": str(role), "content": str(content),
              "at": time.time(), "meta": (meta or {})}
    with _path_for(session_id).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    # Update header.
    data = _load_index()
    for s in data["sessions"]:
        if s.get("id") == session_id:
            s["updated_at"] = record["at"]
            s["message_count"] = int(s.get("message_count") or 0) + 1
            if not (s.get("title") or "").strip() or s.get("title") == "Untitled":
                if role == "user":
                    s["title"] = content[:60]
            break
    _save_index(data)


def load_messages(session_id: str) -> List[Dict[str, Any]]:
    """Load every message in a session, in order."""
    p = _path_for(session_id)
    if not p.exists():
        return []
    out: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception as e:
                logger.warning("sessions: skipping malformed message in %s: %s: %s",
                               session_id, type(e).__name__, e)
                continue
    return out


def rename_session(session_id: str, title: str) -> bool:
    data = _load_index()
    for s in data["sessions"]:
        if s.get("id") == session_id:
            s["title"] = (title or "").strip() or s.get("title") or "Untitled"
            _save_index(data)
            return True
    return False
