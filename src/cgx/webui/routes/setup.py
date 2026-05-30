"""Discovery endpoints used by the Settings page.

Wraps :mod:`cgx.answer.ollama_discovery` so the React app can populate
the model dropdown and re-detect hardware without re-implementing the
heuristics client-side.
"""

from __future__ import annotations

from fastapi import APIRouter

from cgx.answer import ollama_discovery
from cgx.webui.models import HardwareInfo, ModelChoicesResponse


router = APIRouter(tags=["setup"])


@router.get("/setup/models", response_model=ModelChoicesResponse)
def models(base_url: str = "http://localhost:11434") -> ModelChoicesResponse:
    try:
        choices = ollama_discovery.model_choices(base_url)
    except Exception:
        choices = [tag for tag, *_ in ollama_discovery.RECOMMENDED_LADDER]
    try:
        default = ollama_discovery.recommend_default_model(base_url=base_url)
    except Exception:
        default = choices[0] if choices else "qwen2.5-coder:3b"
    return ModelChoicesResponse(choices=choices, recommended_default=default)


@router.get("/setup/hardware", response_model=HardwareInfo)
def hardware_probe() -> HardwareInfo:
    try:
        hw = ollama_discovery.detect_hardware()
    except Exception:
        hw = {}
    return HardwareInfo(**hw)
