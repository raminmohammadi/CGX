

"""High-level workspace status used by the React TopBar.

A single GET so the SPA can render its "is everything wired up?"
header in one call: Ollama health, hardware probe, saved profile
count, session count, telemetry opt-in state.
"""

from __future__ import annotations

from fastapi import APIRouter

from cgx.answer import ollama_discovery
from cgx.answer.profiles import list_profiles
from cgx.webui.models import HardwareInfo, StatusResponse


router = APIRouter(tags=["status"])


@router.get("/status", response_model=StatusResponse)
def get_status() -> StatusResponse:
    try:
        from cgx import sessions as _sessions
        session_count = len(_sessions.list_sessions())
    except Exception:
        session_count = 0
    try:
        from cgx import telemetry
        tele_on = bool(telemetry.is_enabled())
    except Exception:
        tele_on = False
    try:
        health = ollama_discovery.health_check()
    except Exception as e:
        health = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    try:
        hw = ollama_discovery.detect_hardware()
    except Exception:
        hw = {}
    try:
        default_model = ollama_discovery.recommend_default_model()
    except Exception:
        default_model = "qwen2.5-coder:3b"
    return StatusResponse(
        ollama=health,
        hardware=HardwareInfo(**hw),
        telemetry_enabled=tele_on,
        profile_count=len(list_profiles()),
        session_count=session_count,
        default_model=default_model,
    )


@router.get("/health/ollama")
def ollama_health(base_url: str = "http://localhost:11434") -> dict:
    return ollama_discovery.health_check(base_url)
