"""Runtime-settings endpoints (Phase TR.4).

Currently exposes the curated function-call trace toggle backed by
:mod:`cgx.trace`. The endpoint is intentionally generic-shaped so
future runtime knobs (verbosity, structured logging on/off, etc.) can
slot in without a route bump.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from cgx.trace import is_trace_enabled, set_trace_enabled, trace_source


router = APIRouter(tags=["settings"])


class TraceSettings(BaseModel):
    enabled: bool
    source: str  # "env" or "runtime"


class TraceSettingsPatch(BaseModel):
    enabled: bool


@router.get("/settings/trace", response_model=TraceSettings)
def get_trace_settings() -> TraceSettings:
    """Return the current trace toggle state and how it's pinned."""
    return TraceSettings(enabled=is_trace_enabled(), source=trace_source())


@router.post("/settings/trace", response_model=TraceSettings)
def set_trace_settings(payload: TraceSettingsPatch) -> TraceSettings:
    """Flip the runtime trace flag.

    Returns ``409`` when the ``CGX_TRACE`` env var pins the flag so the
    operator can see the override is coming from the environment, not a
    stuck UI control.
    """
    if trace_source() == "env":
        raise HTTPException(
            status_code=409,
            detail=("trace toggle is pinned by the CGX_TRACE environment "
                    "variable; unset it to control the flag via the UI"),
        )
    set_trace_enabled(payload.enabled)
    return TraceSettings(enabled=is_trace_enabled(), source=trace_source())
