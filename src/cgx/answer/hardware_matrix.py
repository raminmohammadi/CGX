

"""Static hardware-vs-model matrix and local-vs-cloud trade-offs.

The data here is intentionally offline -- no network calls, no model
downloads. It augments :mod:`cgx.answer.ollama_discovery` (which holds a
small "recommended ladder") with a wider catalogue of locally-runnable
models so the UI can show users at a glance which models will fit on
their machine, and how the local path compares to a cloud endpoint on
the dimensions that actually matter (privacy, latency, cost, ceiling).

The thresholds below are deliberate approximations meant for UI sorting,
not capacity planning; they assume 4-bit quantised GGUF/AWQ-style
inference which is what Ollama serves by default.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


# (name, params_b, min_ram_gb, recommended_vram_gb, ctx_window, family, notes)
LOCAL_MODEL_CATALOG: List[Dict[str, Any]] = [
    # ── Qwen Coder ──────────────────────────────────────────────────────
    {"name": "qwen2.5-coder:1.5b", "params_b": 1.5, "min_ram_gb": 4.0,
     "recommended_vram_gb": 2.0, "ctx_window": 32768, "family": "coder",
     "notes": "smallest viable coder; CPU-friendly"},
    {"name": "qwen2.5-coder:3b", "params_b": 3.0, "min_ram_gb": 6.0,
     "recommended_vram_gb": 4.0, "ctx_window": 32768, "family": "coder",
     "notes": "balanced default for code Q&A"},
    {"name": "qwen2.5-coder:7b-instruct", "params_b": 7.0, "min_ram_gb": 10.0,
     "recommended_vram_gb": 8.0, "ctx_window": 32768, "family": "coder",
     "notes": "higher-quality coder; sweet spot on 16 GB GPUs"},
    {"name": "qwen2.5-coder:14b-instruct", "params_b": 14.0, "min_ram_gb": 20.0,
     "recommended_vram_gb": 16.0, "ctx_window": 32768, "family": "coder",
     "notes": "near-cloud coder quality; needs >=16 GB VRAM"},
    # ── DeepSeek Coder ──────────────────────────────────────────────────
    {"name": "deepseek-coder:6.7b", "params_b": 6.7, "min_ram_gb": 10.0,
     "recommended_vram_gb": 8.0, "ctx_window": 16384, "family": "coder",
     "notes": "strong FIM; good code completion on 8 GB GPU"},
    {"name": "deepseek-coder-v2:16b", "params_b": 16.0, "min_ram_gb": 20.0,
     "recommended_vram_gb": 16.0, "ctx_window": 163840, "family": "coder",
     "notes": "MoE architecture; very long context; needs >=16 GB VRAM"},
    # ── DeepSeek R1 (reasoning) ─────────────────────────────────────────
    {"name": "deepseek-r1:1.5b", "params_b": 1.5, "min_ram_gb": 4.0,
     "recommended_vram_gb": 2.0, "ctx_window": 65536, "family": "reasoning",
     "notes": "tiny reasoning model; runs on most laptops"},
    {"name": "deepseek-r1:7b", "params_b": 7.0, "min_ram_gb": 10.0,
     "recommended_vram_gb": 8.0, "ctx_window": 65536, "family": "reasoning",
     "notes": "chain-of-thought reasoning; solid on 8 GB GPU"},
    # ── Gemma 4 (Google) ────────────────────────────────────────────────
    # Sizes / contexts match the Ollama library page for ollama.com/library
    # /gemma4 (E2B/E4B = 128K, 12B/26B-A4B/31B = 256K). E variants are
    # "effective" parameter models for edge deployment.
    {"name": "gemma4:e2b", "params_b": 2.0, "min_ram_gb": 8.0,
     "recommended_vram_gb": 8.0, "ctx_window": 131072, "family": "general",
     "notes": "Effective 2B; ~7.2 GB on disk; mobile/edge tier"},
    {"name": "gemma4:e4b", "params_b": 4.0, "min_ram_gb": 12.0,
     "recommended_vram_gb": 10.0, "ctx_window": 131072, "family": "general",
     "notes": "Effective 4B (gemma4:latest alias); ~9.6 GB on disk"},
    {"name": "gemma4:12b", "params_b": 12.0, "min_ram_gb": 10.0,
     "recommended_vram_gb": 8.0, "ctx_window": 262144, "family": "general",
     "notes": "Workstation dense; ~7.6 GB on disk at default quant"},
    {"name": "gemma4:26b", "params_b": 26.0, "min_ram_gb": 22.0,
     "recommended_vram_gb": 18.0, "ctx_window": 262144, "family": "reasoning",
     "notes": "MoE (4B active/token); ~18 GB on disk"},
    {"name": "gemma4:31b", "params_b": 31.0, "min_ram_gb": 24.0,
     "recommended_vram_gb": 24.0, "ctx_window": 262144, "family": "reasoning",
     "notes": "Dense; ~20 GB on disk; near-cloud quality"},
    # ── Gemma 3 (Google) ────────────────────────────────────────────────
    {"name": "gemma3:1b", "params_b": 1.0, "min_ram_gb": 3.0,
     "recommended_vram_gb": 2.0, "ctx_window": 32768, "family": "general",
     "notes": "ultra-light; runs CPU-only on any modern laptop"},
    {"name": "gemma3:4b", "params_b": 4.0, "min_ram_gb": 6.0,
     "recommended_vram_gb": 4.0, "ctx_window": 131072, "family": "general",
     "notes": "capable laptop model with very long context"},
    {"name": "gemma2:2b", "params_b": 2.0, "min_ram_gb": 4.0,
     "recommended_vram_gb": 2.0, "ctx_window": 8192, "family": "general",
     "notes": "efficient small model; good quality-per-GB"},
    {"name": "gemma2:9b", "params_b": 9.0, "min_ram_gb": 12.0,
     "recommended_vram_gb": 8.0, "ctx_window": 8192, "family": "general",
     "notes": "high-quality general; sweet spot on 12 GB RAM"},
    # ── General purpose ─────────────────────────────────────────────────
    {"name": "llama3.2:3b-instruct", "params_b": 3.0, "min_ram_gb": 6.0,
     "recommended_vram_gb": 4.0, "ctx_window": 131072, "family": "general",
     "notes": "long context, light general-purpose"},
    {"name": "llama3.1:8b-instruct", "params_b": 8.0, "min_ram_gb": 12.0,
     "recommended_vram_gb": 8.0, "ctx_window": 131072, "family": "general",
     "notes": "general-purpose with strong reasoning"},
    {"name": "qwen2.5:7b-instruct", "params_b": 7.0, "min_ram_gb": 10.0,
     "recommended_vram_gb": 8.0, "ctx_window": 32768, "family": "general",
     "notes": "general-purpose alternative to llama 8b"},
    {"name": "phi3.5:3.8b-mini-instruct", "params_b": 3.8, "min_ram_gb": 6.0,
     "recommended_vram_gb": 4.0, "ctx_window": 131072, "family": "general",
     "notes": "small, long-context, low-RAM"},
]


def _effective_budget_gb(hw: Dict[str, Any]) -> float:
    """Return a single GB number representing how much model we can afford.

    Mirrors :func:`cgx.answer.ollama_discovery.recommend_default_model`:
    VRAM (when present) dominates because the model lives in GPU memory;
    otherwise fall back to system RAM.
    """
    ram = float(hw.get("ram_gb") or 0.0)
    vram = float(hw.get("gpu_vram_gb") or 0.0)
    if vram > 0:
        return max(ram, vram * 2.0)
    return ram


def _verdict(entry: Dict[str, Any], hw: Dict[str, Any]) -> Dict[str, str]:
    budget = _effective_budget_gb(hw)
    min_ram = float(entry["min_ram_gb"])
    vram = float(hw.get("gpu_vram_gb") or 0.0)
    rec_vram = float(entry["recommended_vram_gb"])
    if budget == 0:
        return {"fit": "unknown", "reason": "hardware probe returned no values"}
    if budget < min_ram * 0.9:
        return {"fit": "won't fit",
                "reason": f"need >={min_ram:g} GB, have {budget:.1f} GB"}
    if vram and vram < rec_vram * 0.75:
        return {"fit": "tight",
                "reason": f">={rec_vram:g} GB VRAM recommended, GPU has {vram:.1f} GB"}
    if budget < min_ram * 1.2:
        return {"fit": "tight",
                "reason": f"within ~20% of the {min_ram:g} GB minimum"}
    return {"fit": "fits", "reason": f"budget {budget:.1f} GB >= {min_ram:g} GB"}


def compute_local_fit(hw: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Annotate :data:`LOCAL_MODEL_CATALOG` with a fit verdict for ``hw``.

    Returns a list of dicts ready for tabular display. Rows are grouped
    by ``family`` (coder → general → reasoning) and then sorted ascending
    by ``params_b`` within each family.
    """
    hw = hw or {}
    rows: List[Dict[str, Any]] = []
    for entry in LOCAL_MODEL_CATALOG:
        v = _verdict(entry, hw)
        rows.append({
            "model": entry["name"],
            "params_b": entry["params_b"],
            "min_ram_gb": entry["min_ram_gb"],
            "rec_vram_gb": entry["recommended_vram_gb"],
            "ctx_window": entry["ctx_window"],
            "family": entry["family"],
            "fit": v["fit"],
            "reason": v["reason"],
            "notes": entry["notes"],
        })
    # Group by family, then ascending params within each family so related
    # models cluster in the UI table.
    _family_order = {"coder": 0, "general": 1, "reasoning": 2}
    rows.sort(key=lambda r: (_family_order.get(r["family"], 99),
                             r["params_b"], r["model"]))
    return rows


# Size hint embedded in a model tag / repo id, e.g. ``7b``, ``1.5B``,
# ``e4b`` (→ 4), ``qwen2.5-coder:7b-instruct`` (→ 7). A trailing ``b`` word
# boundary keeps it from matching version numbers like ``qwen2.5``. Digit
# runs are bounded (no real param count exceeds four integer digits) so the
# pattern stays linear on hostile input like ``"0" * 100000`` instead of
# backtracking quadratically.
_SIZE_HINT_RE = re.compile(r"(\d{1,4}(?:\.\d{1,3})?)\s*b\b", re.IGNORECASE)


def params_from_name(name: str) -> float:
    """Best-effort parameter count (in billions) parsed from a model name.

    Used for models that aren't in :data:`LOCAL_MODEL_CATALOG` -- an
    installed Ollama tag or a Hugging Face repo id -- so the fit table can
    still size them. Returns ``0.0`` when no ``<N>b`` hint is present.
    """
    m = _SIZE_HINT_RE.search(name or "")
    return float(m.group(1)) if m else 0.0


def parse_parameter_size(value: Optional[str]) -> float:
    """Parse Ollama's ``details.parameter_size`` (e.g. ``"7.6B"``) to billions.

    Returns ``0.0`` when the value is missing or unparseable.
    """
    if not value:
        return 0.0
    # Bounded digit runs (mirrors ``_SIZE_HINT_RE``) keep the match linear on
    # a hostile ``parameter_size`` value instead of backtracking quadratically.
    m = re.search(r"(\d{1,4}(?:\.\d{1,3})?)\s*([bmk])?", str(value).strip(), re.IGNORECASE)
    if not m:
        return 0.0
    n = float(m.group(1))
    unit = (m.group(2) or "b").lower()
    if unit == "m":
        return round(n / 1000.0, 3)
    if unit == "k":
        return round(n / 1_000_000.0, 6)
    return n


def estimate_requirements(params_b: float) -> Tuple[float, float]:
    """Approximate (min_ram_gb, recommended_vram_gb) for a 4-bit quant model.

    Calibrated against :data:`LOCAL_MODEL_CATALOG` so estimated rows line up
    with the hand-curated ones: ~0.55 GB of weights per billion params plus
    KV-cache / runtime overhead. Meant for UI sizing, not capacity planning.
    """
    if params_b <= 0:
        return (0.0, 0.0)
    rec_vram = round(params_b * 1.1 + 0.7, 1)
    min_ram = round(params_b * 1.3 + 2.5, 1)
    return (min_ram, rec_vram)


def make_fit_row(
    name: str,
    params_b: float,
    hw: Optional[Dict[str, Any]] = None,
    *,
    family: str = "general",
    ctx_window: int = 0,
    notes: str = "",
    min_ram_gb: Optional[float] = None,
    rec_vram_gb: Optional[float] = None,
    installed: bool = False,
) -> Dict[str, Any]:
    """Build a single fit-matrix row for an arbitrary model.

    Reuses :func:`_verdict` so installed Ollama tags and Hugging Face repos
    are scored the same way as the static catalogue. When ``min_ram_gb`` /
    ``rec_vram_gb`` aren't supplied they're derived from ``params_b`` via
    :func:`estimate_requirements`. A ``params_b`` of ``0`` yields an
    ``unknown`` verdict rather than a misleading "fits".
    """
    hw = hw or {}
    est_ram, est_vram = estimate_requirements(params_b)
    min_ram_gb = est_ram if min_ram_gb is None else min_ram_gb
    rec_vram_gb = est_vram if rec_vram_gb is None else rec_vram_gb
    if params_b <= 0:
        v = {"fit": "unknown", "reason": "parameter count could not be determined"}
    else:
        v = _verdict({"min_ram_gb": min_ram_gb, "recommended_vram_gb": rec_vram_gb}, hw)
    return {
        "model": name,
        "params_b": params_b,
        "min_ram_gb": min_ram_gb,
        "rec_vram_gb": rec_vram_gb,
        "ctx_window": ctx_window,
        "family": family,
        "fit": v["fit"],
        "reason": v["reason"],
        "notes": notes,
        "installed": installed,
    }


# Local-vs-cloud trade-offs. Pure editorial summary; no live numbers, no
# vendor-specific quotes. Each row is one decision dimension.
TRADEOFFS: List[Dict[str, str]] = [
    {"dimension": "Privacy / data egress",
     "local": "Prompts + code never leave the machine.",
     "cloud": "Prompts + retrieved snippets go to the provider; subject to their data policy.",
     "winner": "local"},
    {"dimension": "Marginal cost / token",
     "local": "Electricity only; zero per-call cost once the model is downloaded.",
     "cloud": "Pay-per-token; cost scales linearly with usage and context length.",
     "winner": "local"},
    {"dimension": "Quality ceiling",
     "local": "Capped by what fits on your hardware (approx 14B params on a 16 GB GPU).",
     "cloud": "Access to frontier models (100B+ params, long context, tool-use).",
     "winner": "cloud"},
    {"dimension": "Latency (cold)",
     "local": "First token after model load (seconds on small models, minutes on large).",
     "cloud": "Sub-second TTFT in steady state; spikes during provider load.",
     "winner": "tie"},
    {"dimension": "Latency (warm)",
     "local": "Predictable; bound by local GPU/CPU.",
     "cloud": "Variable; subject to rate limits + network round-trip.",
     "winner": "local"},
    {"dimension": "Offline use",
     "local": "Works on a plane / air-gapped network.",
     "cloud": "Requires connectivity.",
     "winner": "local"},
    {"dimension": "Setup effort",
     "local": "Install Ollama, pull a model (~GB-scale download).",
     "cloud": "Sign up, mint an API key, paste into a profile.",
     "winner": "cloud"},
    {"dimension": "Operational risk",
     "local": "Your machine = your SLO.",
     "cloud": "Vendor outages / price changes / model deprecations.",
     "winner": "local"},
]


def tradeoffs_rows() -> List[Dict[str, str]]:
    """Return the editorial local-vs-cloud comparison table."""
    return list(TRADEOFFS)
