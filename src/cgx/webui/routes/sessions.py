

"""Session list / create / fetch / delete.

Backed by :mod:`cgx.sessions` which persists conversation logs under
``$CGX_CONFIG_DIR/sessions``. The Ask SSE route is the only producer
that appends messages; this router just exposes the read/manage
surface so the UI can render history and a session picker.
"""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException

from cgx import sessions as _sessions
from cgx.webui.models import (
    SessionCreateRequest,
    SessionMessage,
    SessionSummary,
)


router = APIRouter(tags=["sessions"])


@router.get("/sessions", response_model=List[SessionSummary])
def list_sessions() -> List[SessionSummary]:
    return [
        SessionSummary(
            id=m.id, title=m.title,
            created_at=m.created_at, updated_at=m.updated_at,
            message_count=m.message_count,
        )
        for m in _sessions.list_sessions()
    ]


@router.post("/sessions", response_model=SessionSummary)
def create_session(req: SessionCreateRequest) -> SessionSummary:
    m = _sessions.create_session(title=(req.title or "").strip())
    return SessionSummary(
        id=m.id, title=m.title,
        created_at=m.created_at, updated_at=m.updated_at,
        message_count=m.message_count,
    )


@router.get("/sessions/{sid}/messages", response_model=List[SessionMessage])
def get_messages(sid: str) -> List[SessionMessage]:
    msgs = _sessions.load_messages(sid)
    out: List[SessionMessage] = []
    for m in msgs:
        out.append(SessionMessage(
            role=str(m.get("role") or "?"),
            content=str(m.get("content") or ""),
            at=m.get("at"),
            meta=(m.get("meta") if isinstance(m.get("meta"), dict) else None),
        ))
    return out


@router.delete("/sessions/{sid}")
def delete_session(sid: str) -> dict:
    ok = _sessions.delete_session(sid)
    if not ok:
        raise HTTPException(status_code=404, detail=f"session {sid!r} not found")
    return {"deleted": sid}
