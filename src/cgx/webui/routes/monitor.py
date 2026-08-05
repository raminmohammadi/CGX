

"""AIOps alerts read API.

Exposes the process-wide :class:`cgx.monitor.Monitor`'s persisted alerts so
the admin page (and operators via curl) can list recent quality/drift/cost
incidents. Read-only: alerts are produced by the monitors wired into the
ask/plan handlers, not by this route.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["monitor"])


@router.get("/monitor/alerts")
def list_alerts(
    limit: int = Query(100, ge=1, le=1000),
    severity: Optional[str] = Query(None),
    code: Optional[str] = Query(None),
    since: Optional[float] = Query(None),
) -> JSONResponse:
    """Return recent alerts (most recent first), optionally filtered."""
    try:
        from cgx.monitor import get_default_monitor
        alerts: List[Dict[str, Any]] = get_default_monitor().recent(
            limit=limit, severity=severity, code=code, since=since)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("list_alerts failed: %s", e)
        alerts = []
    return JSONResponse({"alerts": alerts, "count": len(alerts)})
