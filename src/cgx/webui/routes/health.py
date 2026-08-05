

"""Kubernetes-style probe endpoints.

``/healthz`` (liveness) and ``/readyz`` (readiness) live at the *root*
(not under ``/api``) so orchestrators and load balancers can hit the
conventional paths. They are mounted before the SPA catch-all so the
React fallback never shadows them.

Readiness returns HTTP 503 when a critical subsystem is unusable so a
load balancer stops routing traffic; provider/index checks are reported
but non-critical (see :mod:`cgx.health`).
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from cgx import health as _health

router = APIRouter(tags=["health"])


@router.get("/healthz", include_in_schema=False)
def healthz() -> dict:
    return _health.liveness()


@router.get("/readyz", include_in_schema=False, response_model=None)
def readyz() -> JSONResponse:
    report = _health.readiness()
    return JSONResponse(report, status_code=200 if report["ready"] else 503)
