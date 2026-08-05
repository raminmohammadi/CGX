

"""Prometheus scrape endpoint.

Exposes the in-process :mod:`cgx.metrics` registry (RED per-route, LLM
tokens/latency/cost, ...) in the Prometheus text exposition format so an
external Prometheus/Grafana stack -- or a curl -- can scrape fleet-level
telemetry. No external client library is required.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from cgx import metrics as _metrics

router = APIRouter(tags=["metrics"])

# Prometheus text exposition content type (OpenMetrics-compatible).
_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


@router.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
def get_metrics() -> PlainTextResponse:
    return PlainTextResponse(_metrics.render_prometheus(), media_type=_CONTENT_TYPE)
