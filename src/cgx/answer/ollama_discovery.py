"""Ollama installation discovery + hardware-aware model recommendations.

This module is intentionally dependency-light: a single ``requests`` call
against ``GET /api/tags`` to list installed models, and a small static
catalogue mapping VRAM/RAM hints to a recommended ladder. It returns plain
dicts so the Gradio UI can render them without extra coupling.
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

import requests

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT = float(os.environ.get("CGX_OLLAMA_DISCOVERY_TIMEOUT", "3.0"))


# Recommended ladder. Each entry: (tag, approx_params_b, min_ram_gb, role).
RECOMMENDED_LADDER: List[Tuple[str, float, float, str]] = [
    ("qwen2.5-coder:1.5b", 1.5, 4.0, "fast / low-RAM"),
    ("qwen2.5-coder:3b", 3.0, 6.0, "balanced default"),
    ("qwen2.5-coder:7b-instruct", 7.0, 10.0, "higher quality"),
    ("llama3.2:3b-instruct", 3.0, 6.0, "general"),
    ("llama3.1:8b-instruct", 8.0, 12.0, "general, higher quality"),
    ("qwen2.5:7b-instruct", 7.0, 10.0, "general"),
]


def list_installed_models(base_url: str = DEFAULT_BASE_URL) -> List[Dict[str, Any]]:
    """Return installed Ollama models, or [] if the server is unreachable."""
    url = base_url.rstrip("/") + "/api/tags"
    try:
        r = requests.get(url, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []
    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, list):
        return []
    out: List[Dict[str, Any]] = []
    for m in models:
        if not isinstance(m, dict):
            continue
        out.append({
            "name": m.get("name") or m.get("model") or "",
            "size": m.get("size"),
            "modified_at": m.get("modified_at"),
            "family": (m.get("details") or {}).get("family"),
            "parameter_size": (m.get("details") or {}).get("parameter_size"),
        })
    return [m for m in out if m["name"]]


def health_check(base_url: str = DEFAULT_BASE_URL) -> Dict[str, Any]:
    """Return a small status dict suitable for surfacing in the UI."""
    url = base_url.rstrip("/")
    try:
        r = requests.get(url + "/api/tags", timeout=DEFAULT_TIMEOUT)
        ok = r.ok
        return {
            "ok": ok,
            "base_url": url,
            "status_code": r.status_code,
            "models_count": len((r.json() or {}).get("models", [])) if ok else 0,
        }
    except Exception as e:
        return {"ok": False, "base_url": url, "error": f"{type(e).__name__}: {e}"}


def _detect_total_ram_gb() -> Optional[float]:
    try:
        meminfo = "/proc/meminfo"
        if os.path.exists(meminfo):
            with open(meminfo, "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return round(kb / (1024.0 * 1024.0), 1)
    except Exception:
        pass
    return None


def _detect_gpu_vram_gb() -> Optional[float]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
    try:
        import subprocess
        out = subprocess.run(
            [nvidia_smi, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2.0,
        )
        if out.returncode != 0:
            return None
        vals = [int(x.strip()) for x in out.stdout.splitlines() if x.strip().isdigit()]
        if not vals:
            return None
        return round(max(vals) / 1024.0, 1)
    except Exception:
        return None


def detect_hardware() -> Dict[str, Any]:
    """Best-effort hardware probe used to pick a sensible default model."""
    return {
        "ram_gb": _detect_total_ram_gb(),
        "gpu_vram_gb": _detect_gpu_vram_gb(),
    }


def recommend_default_model(installed: Optional[List[Dict[str, Any]]] = None,
                            base_url: str = DEFAULT_BASE_URL) -> str:
    """Pick the best recommended model that is installed, otherwise the most
    capable from the static ladder that fits in available RAM."""
    if installed is None:
        installed = list_installed_models(base_url)
    installed_names = {m["name"] for m in installed}
    hw = detect_hardware()
    ram = hw.get("ram_gb") or 0.0
    vram = hw.get("gpu_vram_gb") or 0.0
    budget = max(ram, vram * 2.0) if vram else ram
    affordable = [tag for tag, _params, min_ram, _role in RECOMMENDED_LADDER if min_ram <= budget or budget == 0]
    for tag in reversed(affordable):
        if tag in installed_names:
            return tag
    for tag in reversed(affordable):
        return tag
    return "qwen2.5-coder:3b"


def model_choices(base_url: str = DEFAULT_BASE_URL) -> List[str]:
    """Union of installed Ollama models + recommended ladder (installed first)."""
    installed = [m["name"] for m in list_installed_models(base_url)]
    seen = set(installed)
    out: List[str] = list(installed)
    for tag, _p, _r, _role in RECOMMENDED_LADDER:
        if tag not in seen:
            out.append(tag)
            seen.add(tag)
    return out
