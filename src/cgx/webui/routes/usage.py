

"""Cost & quota usage read API (Subsystem I).

Exposes the process-wide :class:`cgx.governance.QuotaManager`'s per-owner
usage + budget state so the activity page (own spend) and the admin page
(all owners) can render cost dashboards. Read-only: usage is recorded by the
:class:`~cgx.governance.provider.GovernedProvider` wired into the ask/plan and
agent-session handlers, not by this route.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["usage"])


@router.get("/usage")
def owner_usage(
    owner: Optional[str] = Query(None),
    day: Optional[str] = Query(None),
) -> JSONResponse:
    """Return one owner's current-day totals + budget limits/state."""
    try:
        from cgx.governance import get_default_quota_manager, resolve_owner
        mgr = get_default_quota_manager()
        who = owner or resolve_owner()
        status = mgr.check(who, enforce=False)
        status.update(mgr.meter.totals(who, day=day))
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("owner_usage failed: %s", e)
        return JSONResponse({"detail": "could not read usage"},
                            status_code=500)
    return JSONResponse(status)


@router.get("/usage/summary")
def usage_summary(day: Optional[str] = Query(None)) -> JSONResponse:
    """Return per-owner totals for the day (admin cost dashboard)."""
    try:
        from cgx.governance import get_default_quota_manager
        rows: List[Dict[str, Any]] = get_default_quota_manager().meter.summary(
            day=day)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("usage_summary failed: %s", e)
        rows = []
    return JSONResponse({"usage": rows, "count": len(rows)})
