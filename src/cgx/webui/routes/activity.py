

"""User Activity API (Subsystem C).

Read-only views over the per-run observation store
(:class:`cgx.activity.RunStore`): a filtered runs list, a per-run detail that
joins the run to its feedback (Subsystem H) and monitor/guardrail alerts
(Subsystems G/K), and an aggregate summary for the activity dashboard. Runs
are recorded by the ask/plan handlers, not by this route.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["activity"])


@router.get("/activity/runs")
def list_runs(
    kind: Optional[str] = Query(None),
    owner: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
) -> JSONResponse:
    """Return recent runs, most-recent first, with optional filters."""
    try:
        from cgx.activity import get_default_run_store
        rows = get_default_run_store().recent(
            limit=limit, kind=kind, owner=owner, status=status)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("list_runs failed: %s", e)
        rows = []
    return JSONResponse({"runs": rows, "count": len(rows)})


@router.get("/activity/summary")
def activity_summary() -> JSONResponse:
    """Return aggregate run counts + cost/token totals for the dashboard."""
    try:
        from cgx.activity import get_default_run_store
        summary = get_default_run_store().summary()
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("activity_summary failed: %s", e)
        summary = {"total": 0, "cost_usd": 0.0, "tokens_total": 0,
                   "errors": 0, "by_kind": {}}
    return JSONResponse(summary)


@router.get("/activity/runs/{run_id}")
def run_detail(run_id: str) -> JSONResponse:
    """Return one run joined to its feedback + monitor/guardrail alerts."""
    try:
        from cgx.activity import get_default_run_store
        run = get_default_run_store().get(run_id)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("run_detail failed: %s", e)
        run = None
    if run is None:
        return JSONResponse({"detail": "run not found"}, status_code=404)

    feedback: List[Dict[str, Any]] = []
    try:
        from cgx.feedback import get_default_store
        feedback = get_default_store().recent(run_id=run_id, limit=50)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("run_detail feedback lookup failed: %s", e)

    alerts: List[Dict[str, Any]] = []
    try:
        from cgx.monitor import get_default_monitor
        alerts = [a for a in get_default_monitor().recent(limit=500)
                  if a.get("run_id") == run_id]
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("run_detail alert lookup failed: %s", e)

    return JSONResponse({"run": run, "feedback": feedback, "alerts": alerts})
