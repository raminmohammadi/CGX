"""Hardware page payload — model fit matrix + local-vs-cloud tradeoffs.

Stateless: every request re-detects RAM/VRAM and re-annotates the
catalogue. Cheap because :mod:`cgx.answer.hardware_matrix` does no I/O.
"""

from __future__ import annotations

from fastapi import APIRouter

from cgx.answer import ollama_discovery
from cgx.answer.hardware_matrix import compute_local_fit, tradeoffs_rows
from cgx.webui.models import (
    HardwareInfo,
    HardwareMatrixResponse,
    HardwareMatrixRow,
    TradeoffRow,
)


router = APIRouter(tags=["hardware"])


@router.get("/hardware/matrix", response_model=HardwareMatrixResponse)
def matrix() -> HardwareMatrixResponse:
    try:
        hw = ollama_discovery.detect_hardware()
    except Exception:
        hw = {}
    rows = compute_local_fit(hw)
    return HardwareMatrixResponse(
        hardware=HardwareInfo(**hw),
        rows=[HardwareMatrixRow(**r) for r in rows],
        tradeoffs=[TradeoffRow(**t) for t in tradeoffs_rows()],
    )
