

"""User feedback API (Subsystem H).

``POST /api/feedback`` records a thumbs up/down (+ optional comment) joined to
the ``run_id``/``prompt_version``/``model`` the ask/plan handler returned in
its ``meta``. ``GET /api/feedback`` lists recent ratings and ``GET
/api/feedback/stats`` returns aggregate satisfaction for the activity/admin
pages. Writes go through the process-wide :class:`cgx.feedback.FeedbackStore`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from cgx.webui.models import FeedbackRequest

logger = logging.getLogger(__name__)
router = APIRouter(tags=["feedback"])


@router.post("/feedback")
def submit_feedback(req: FeedbackRequest) -> JSONResponse:
    """Persist one rating; returns the stored ``feedback_id``."""
    rating = (req.rating or "").strip().lower()
    if rating not in ("up", "down"):
        return JSONResponse({"detail": "rating must be 'up' or 'down'"},
                            status_code=422)
    try:
        from cgx.feedback import Feedback, get_default_store
        fb = Feedback(
            rating=rating, run_id=req.run_id, session_id=req.session_id,
            kind=(req.kind or "ask"), comment=(req.comment or ""),
            question=(req.question or ""),
            answer_preview=(req.answer_preview or "")[:2000],
            model=req.model, prompt_version=req.prompt_version,
            labels=dict(req.labels or {}),
        )
        fid = get_default_store().record(fb)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("submit_feedback failed: %s", e)
        return JSONResponse({"detail": "could not record feedback"},
                            status_code=500)
    return JSONResponse({"ok": True, "feedback_id": fid})


@router.get("/feedback")
def list_feedback(
    limit: int = Query(100, ge=1, le=1000),
    rating: Optional[str] = Query(None),
    kind: Optional[str] = Query(None),
    run_id: Optional[str] = Query(None),
    since: Optional[float] = Query(None),
) -> JSONResponse:
    """Return recent feedback (most recent first), optionally filtered."""
    try:
        from cgx.feedback import get_default_store
        rows: List[Dict[str, Any]] = get_default_store().recent(
            limit=limit, rating=rating, kind=kind, run_id=run_id, since=since)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("list_feedback failed: %s", e)
        rows = []
    return JSONResponse({"feedback": rows, "count": len(rows)})


@router.get("/feedback/stats")
def feedback_stats(since: Optional[float] = Query(None)) -> JSONResponse:
    """Return aggregate up/down counts + satisfaction for dashboards."""
    try:
        from cgx.feedback import get_default_store
        stats = get_default_store().stats(since=since)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("feedback_stats failed: %s", e)
        stats = {"total": 0, "up": 0, "down": 0, "satisfaction": None,
                 "by_kind": {}}
    return JSONResponse(stats)
