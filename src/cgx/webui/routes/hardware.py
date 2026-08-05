

"""Hardware page payload -- model fit matrix + local-vs-cloud tradeoffs.

Stateless: every request re-detects RAM/VRAM and re-annotates the
catalogue. The matrix merges the models the user has actually pulled into
Ollama so the table stays in sync with what's on disk, and ``/hardware/
hf_fit`` scores an arbitrary Hugging Face repo against the same budget.
"""

from __future__ import annotations

import logging
import re

from fastapi import APIRouter

from cgx.answer import ollama_discovery
from cgx.answer.hardware_matrix import (
    compute_local_fit,
    make_fit_row,
    params_from_name,
    parse_parameter_size,
    tradeoffs_rows,
)
from cgx.webui.models import (
    HardwareInfo,
    HardwareMatrixResponse,
    HardwareMatrixRow,
    HfModelFitResponse,
    TradeoffRow,
)


logger = logging.getLogger(__name__)
router = APIRouter(tags=["hardware"])

# Hugging Face Hub host is a compile-time constant; the user-supplied repo id
# is validated against this allowlist regex and only ever used as a path
# segment, so nothing attacker-controlled can redirect the request off-host
# (SSRF barrier). ``owner/name`` with a conservative character set.
_HF_HUB_BASE = "https://huggingface.co"
_HF_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


def _sanitize_for_log(value: object) -> str:
    """Return a single-line representation safe for plain-text logs."""
    return str(value).replace("\r", "").replace("\n", "")


@router.get("/hardware/matrix", response_model=HardwareMatrixResponse)
def matrix(base_url: str = ollama_discovery.DEFAULT_BASE_URL) -> HardwareMatrixResponse:
    try:
        hw = ollama_discovery.detect_hardware()
    except Exception:
        hw = {}
    rows = compute_local_fit(hw)

    # Flag catalogue rows the user has pulled, and append any installed-only
    # tags that aren't in the static catalogue so the table mirrors disk.
    try:
        installed = ollama_discovery.list_installed_models(base_url)
    except Exception:
        installed = []
    installed_by_name = {m["name"]: m for m in installed if m.get("name")}
    catalog_names = {r["model"] for r in rows}
    for r in rows:
        r["installed"] = r["model"] in installed_by_name
    for name, m in installed_by_name.items():
        if name in catalog_names:
            continue
        params = parse_parameter_size(m.get("parameter_size")) or params_from_name(name)
        rows.append(make_fit_row(
            name, params, hw,
            family=(m.get("family") or "installed"),
            notes="installed locally",
            installed=True,
        ))

    return HardwareMatrixResponse(
        hardware=HardwareInfo(**hw),
        rows=[HardwareMatrixRow(**r) for r in rows],
        tradeoffs=[TradeoffRow(**t) for t in tradeoffs_rows()],
    )


def _hf_model_spec(repo: str) -> dict:
    """Fetch a repo's public metadata from the Hub (params, gating, pipeline).

    Returns the exact param count from ``safetensors.total`` when the Hub
    reports it. The host is a fixed constant and ``repo`` is pre-validated
    against :data:`_HF_REPO_RE`, so it can only ever be a single ``owner/name``
    path segment (SSRF barrier). Raises on any upstream failure.
    """
    import requests as _req

    r = _req.get(f"{_HF_HUB_BASE}/api/models/{repo}", timeout=15)
    r.raise_for_status()
    data = r.json() if r.content else {}
    return data if isinstance(data, dict) else {}


@router.get("/hardware/hf_fit", response_model=HfModelFitResponse)
def hf_fit(repo: str) -> HfModelFitResponse:
    """Score a Hugging Face repo against the detected hardware budget.

    Powers the "Check fit" action in the Browse Hugging Face panel: resolves
    the model's parameter count (preferring the Hub's exact ``safetensors``
    total, falling back to the size hint in the repo id) and returns the same
    fit verdict the local catalogue uses. Degrades to an ``unknown`` verdict
    on any upstream failure instead of raising.
    """
    try:
        hw = ollama_discovery.detect_hardware()
    except Exception:
        hw = {}

    repo = (repo or "").strip()
    if not _HF_REPO_RE.match(repo):
        return HfModelFitResponse(
            repo=repo, reason="invalid repo id (expected owner/name)",
            hardware=HardwareInfo(**hw),
        )

    params_b = 0.0
    params_source = "unknown"
    pipeline_tag = None
    gated = False
    try:
        spec = _hf_model_spec(repo)
        st = spec.get("safetensors")
        total = st.get("total") if isinstance(st, dict) else None
        if isinstance(total, (int, float)) and total > 0:
            params_b = round(float(total) / 1e9, 2)
            params_source = "safetensors"
        pipeline_tag = spec.get("pipeline_tag") or None
        gated = bool(spec.get("gated"))
    except Exception as e:
        logger.info(
            "hardware.hf_fit: spec fetch failed for %r: %s",
            _sanitize_for_log(repo),
            type(e).__name__,
        )

    if params_b <= 0:
        params_b = params_from_name(repo)
        if params_b > 0:
            params_source = "name"

    row = make_fit_row(repo, params_b, hw, family="huggingface")
    return HfModelFitResponse(
        repo=repo,
        params_b=params_b,
        params_source=params_source,
        min_ram_gb=row["min_ram_gb"],
        rec_vram_gb=row["rec_vram_gb"],
        ctx_window=row["ctx_window"],
        fit=row["fit"],
        reason=row["reason"],
        pipeline_tag=pipeline_tag,
        gated=gated,
        hardware=HardwareInfo(**hw),
    )
