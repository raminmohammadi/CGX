

"""Ollama installation discovery + hardware-aware model recommendations.

This module is intentionally dependency-light: a single ``requests`` call
against ``GET /api/tags`` to list installed models, and a small static
catalogue mapping VRAM/RAM hints to a recommended ladder. It returns plain
dicts so the web UI can render them without extra coupling.
"""

from __future__ import annotations

import logging
import os
import re
import platform
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

import requests

from cgx.logging_setup import sanitize_for_log

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT = float(os.environ.get("CGX_OLLAMA_DISCOVERY_TIMEOUT", "3.0"))

# Only these schemes may ever reach ``requests`` from a user-supplied base
# URL -- blocks ``file:``/``gopher:``/``dict:`` and other SSRF vectors.
_ALLOWED_URL_SCHEMES = frozenset({"http", "https"})


def validate_base_url(base_url: str) -> str:
    """Validate and normalize a user-provided provider base URL.

    The Settings page lets the user type an arbitrary Ollama / OpenAI-compat
    base URL that this process then fetches server-side, so an unvalidated
    value is a server-side request forgery (SSRF) vector. Only ``http`` /
    ``https`` URLs that carry a real host are accepted; embedded credentials
    are rejected, and the result is rebuilt from the validated components so
    no unexpected scheme or query/fragment can ride along into ``requests``.

    Returns the normalized ``scheme://authority[/path]`` string (no trailing
    slash) and raises :class:`ValueError` on anything else.
    """
    raw = (base_url or "").strip()
    if not raw:
        raise ValueError("base_url is required")
    parts = urlsplit(raw)
    scheme = parts.scheme.lower()
    if scheme not in _ALLOWED_URL_SCHEMES:
        raise ValueError(f"unsupported URL scheme: {parts.scheme or '(none)'!r}")
    if parts.username or parts.password:
        raise ValueError("base_url must not embed credentials")
    if not parts.hostname:
        raise ValueError("base_url is missing a host")
    return urlunsplit((scheme, parts.netloc, parts.path.rstrip("/"), "", ""))


# Recommended ladder. Each entry: (tag, approx_params_b, min_ram_gb, role).
RECOMMENDED_LADDER: List[Tuple[str, float, float, str]] = [
    ("qwen2.5-coder:1.5b", 1.5, 4.0, "fast / low-RAM"),
    ("qwen2.5-coder:3b", 3.0, 6.0, "balanced default"),
    ("qwen2.5-coder:7b-instruct", 7.0, 10.0, "higher quality"),
    ("gemma4:e2b", 2.0, 4.0, "general, QAT 4-bit, mobile/edge"),
    ("gemma4:e4b", 4.0, 6.0, "general, QAT 4-bit, laptop sweet spot"),
    ("llama3.2:3b-instruct", 3.0, 6.0, "general"),
    ("llama3.1:8b-instruct", 8.0, 12.0, "general, higher quality"),
    ("qwen2.5:7b-instruct", 7.0, 10.0, "general"),
]


def list_installed_models(base_url: str = DEFAULT_BASE_URL) -> List[Dict[str, Any]]:
    """Return installed Ollama models, or [] if the server is unreachable."""
    try:
        url = validate_base_url(base_url) + "/api/tags"
    except ValueError as e:
        logger.info("ollama_discovery: rejected base_url %r: %s",
                    sanitize_for_log(base_url), type(e).__name__)
        return []
    try:
        r = requests.get(url, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.info("ollama_discovery: list_installed_models %s unreachable: %s: %s",
                    url, type(e).__name__, e)
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
    # The error strings below deliberately omit the raw exception text: this
    # dict is returned straight to the Web UI (``/health/ollama``), so echoing
    # ``str(e)`` would surface internal detail (py/stack-trace-exposure). The
    # full exception is logged server-side for diagnostics instead.
    try:
        url = validate_base_url(base_url)
    except ValueError as e:
        # ``base_url`` is untrusted (typed on the Settings page), so strip any
        # CR/LF before logging: unescaped line breaks would let a caller forge
        # extra log records (py/log-injection, CWE-117).
        safe_base_url = base_url.replace("\n", " ").replace("\r", " ")
        logger.warning("ollama health_check: invalid base_url %r", safe_base_url)
        return {"ok": False, "base_url": base_url, "error": "invalid base_url"}
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
        logger.warning("ollama health_check: request to %r failed: %s",
                       url, type(e).__name__)
        return {"ok": False, "base_url": url, "error": type(e).__name__}


def list_running_models(base_url: str = DEFAULT_BASE_URL) -> List[Dict[str, Any]]:
    """Return models currently resident in Ollama via ``GET /api/ps``.

    Each entry exposes the bits the UI needs to render a "loaded model" pill:
    name, effective ``context_length`` (Ollama's KV-cache for this load), the
    ``size`` / ``size_vram`` byte counts (so the UI can compute the GPU/CPU
    split), and the keep-alive ``expires_at`` timestamp. Returns ``[]`` if
    the server is unreachable -- the SPA treats absence as "nothing loaded".
    """
    try:
        url = validate_base_url(base_url) + "/api/ps"
    except ValueError as e:
        logger.info("ollama_discovery: rejected base_url %r: %s", base_url, e)
        return []
    try:
        r = requests.get(url, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.info("ollama_discovery: list_running_models %s unreachable: %s: %s",
                    url, type(e).__name__, e)
        return []
    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, list):
        return []
    out: List[Dict[str, Any]] = []
    for m in models:
        if not isinstance(m, dict):
            continue
        size = m.get("size")
        size_vram = m.get("size_vram")
        # Ollama reports total resident bytes and the slice resident on the
        # GPU; when they're equal the model is fully on GPU, when size_vram
        # is 0 it's CPU-only, otherwise it's a split. Surface the raw bytes
        # and let the UI render the placement label.
        out.append({
            "name": m.get("name") or m.get("model") or "",
            "model": m.get("model") or m.get("name") or "",
            "size": size,
            "size_vram": size_vram,
            "context_length": m.get("context_length"),
            "expires_at": m.get("expires_at"),
            "digest": m.get("digest"),
        })
    return [m for m in out if m["name"]]


def _detect_total_ram_gb() -> Optional[float]:
    # 1. POSIX standard sysconf (works on macOS, Linux, FreeBSD, OpenBSD)
    try:
        if hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if pages > 0 and page_size > 0:
                return round((pages * page_size) / (1024.0 ** 3), 1)
    except Exception:
        pass

    # 2. macOS sysctl fallback
    if platform.system() == "Darwin":
        try:
            import subprocess
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=1.0,
            )
            if out.returncode == 0 and out.stdout.strip().isdigit():
                return round(int(out.stdout.strip()) / (1024.0 ** 3), 1)
        except Exception:
            pass

    # 3. Linux /proc/meminfo fallback
    try:
        meminfo = "/proc/meminfo"
        if os.path.exists(meminfo):
            with open(meminfo, "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return round(kb / (1024.0 * 1024.0), 1)
    except Exception as e:
        logger.info("ollama_discovery: RAM probe failed: %s: %s", type(e).__name__, e)

    # 4. psutil fallback
    try:
        import psutil  # type: ignore
        return round(psutil.virtual_memory().total / (1024.0 ** 3), 1)
    except Exception:
        pass

    return None


def _detect_mac_gpu(total_ram_gb: Optional[float] = None) -> Dict[str, Any]:
    """Probe macOS GPU (Apple Silicon or discrete/integrated Intel GPU)."""
    info: Dict[str, Any] = {
        "gpu_name": None,
        "gpu_vram_gb": None,
        "gpu_type": None,
        "is_unified_memory": False,
    }
    if platform.system() != "Darwin":
        return info

    chipset: Optional[str] = None
    cores: Optional[str] = None
    metal: Optional[str] = None
    vram_val: Optional[float] = None

    try:
        import subprocess
        out = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=3.0,
        )
        if out.returncode == 0:
            text = out.stdout
            m_chip = re.search(r"Chipset Model:\s*(.+)", text)
            if m_chip:
                chipset = m_chip.group(1).strip()
            m_cores = re.search(r"Total Number of Cores:\s*(\d+)", text)
            if m_cores:
                cores = m_cores.group(1).strip()
            m_metal = re.search(r"Metal Support:\s*(.+)", text)
            if m_metal:
                metal = m_metal.group(1).strip()
            m_vram = re.search(r"VRAM \(Total\):\s*([0-9.]+)\s*([GMK]B)", text, re.IGNORECASE)
            if m_vram:
                num = float(m_vram.group(1))
                unit = m_vram.group(2).upper()
                if unit == "GB":
                    vram_val = round(num, 1)
                elif unit == "MB":
                    vram_val = round(num / 1024.0, 1)
    except Exception as e:
        logger.info("ollama_discovery: Mac system_profiler probe failed: %s: %s", type(e).__name__, e)

    # Fallback to sysctl brand string if chipset wasn't extracted
    if not chipset:
        try:
            import subprocess
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=1.0,
            )
            if out.returncode == 0 and out.stdout.strip():
                chipset = out.stdout.strip()
        except Exception:
            pass

    is_apple_silicon = (
        platform.machine() in ("arm64", "aarch64")
        or (chipset is not None and "Apple" in chipset)
    )

    if is_apple_silicon:
        base_name = chipset or "Apple Silicon GPU"
        details = []
        if cores:
            details.append(f"{cores} cores")
        if metal:
            details.append(metal)
        gpu_name = f"{base_name} ({', '.join(details)})" if details else base_name
        info["gpu_name"] = gpu_name
        info["gpu_type"] = "apple_silicon"
        info["is_unified_memory"] = True
        # On Apple Silicon, unified memory is shared between CPU and GPU for Metal/Ollama inference.
        info["gpu_vram_gb"] = total_ram_gb
    elif chipset:
        info["gpu_name"] = chipset
        info["gpu_type"] = "discrete" if (vram_val and vram_val >= 2.0) else "integrated"
        info["is_unified_memory"] = False
        info["gpu_vram_gb"] = vram_val

    return info


def _detect_nvidia_gpu() -> Dict[str, Any]:
    """Probe NVIDIA GPU using nvidia-smi."""
    info: Dict[str, Any] = {
        "gpu_name": None,
        "gpu_vram_gb": None,
        "gpu_type": None,
        "is_unified_memory": False,
    }
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return info
    try:
        import subprocess
        out_vram = subprocess.run(
            [nvidia_smi, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2.0,
        )
        if out_vram.returncode == 0:
            vals = [int(x.strip()) for x in out_vram.stdout.splitlines() if x.strip().isdigit()]
            if vals:
                info["gpu_vram_gb"] = round(max(vals) / 1024.0, 1)
                info["gpu_type"] = "nvidia"
                info["is_unified_memory"] = False
        out_name = subprocess.run(
            [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=2.0,
        )
        if out_name.returncode == 0:
            names = [x.strip() for x in out_name.stdout.splitlines() if x.strip()]
            if names:
                info["gpu_name"] = names[0]
    except Exception as e:
        logger.info("ollama_discovery: NVIDIA probe failed: %s: %s", type(e).__name__, e)
    return info


def _detect_gpu_info(total_ram_gb: Optional[float] = None) -> Dict[str, Any]:
    # 1. Try NVIDIA first (if nvidia-smi is available)
    nvidia_info = _detect_nvidia_gpu()
    if nvidia_info["gpu_vram_gb"] is not None:
        return nvidia_info

    # 2. Try macOS (Apple Silicon or discrete/integrated Mac GPU)
    if platform.system() == "Darwin":
        mac_info = _detect_mac_gpu(total_ram_gb)
        if mac_info["gpu_name"] is not None or mac_info["gpu_vram_gb"] is not None:
            return mac_info

    return {
        "gpu_name": None,
        "gpu_vram_gb": None,
        "gpu_type": None,
        "is_unified_memory": False,
    }


def _detect_gpu_vram_gb() -> Optional[float]:
    ram = _detect_total_ram_gb()
    info = _detect_gpu_info(ram)
    return info.get("gpu_vram_gb")


_TORCH_PROBE_CACHE: Optional[Dict[str, Any]] = None
_TORCH_WARNING_LOGGED = False


def _detect_torch() -> Dict[str, Any]:
    """Probe torch's CUDA and Apple Silicon MPS availability and cache the result."""
    global _TORCH_PROBE_CACHE
    if _TORCH_PROBE_CACHE is not None:
        return _TORCH_PROBE_CACHE
    info: Dict[str, Any] = {
        "installed": False,
        "cuda_available": False,
        "mps_available": False,
        "torch_version": None,
        "cuda_build": None,
        "error": None,
    }
    try:
        import torch  # type: ignore
        info["installed"] = True
        info["torch_version"] = getattr(torch, "__version__", None)
        info["cuda_build"] = getattr(getattr(torch, "version", None), "cuda", None)
        try:
            info["cuda_available"] = bool(torch.cuda.is_available())
        except Exception as e:
            info["error"] = f"{type(e).__name__}: {e}"
        try:
            info["mps_available"] = bool(
                getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            )
        except Exception as e:
            if not info["error"]:
                info["error"] = f"{type(e).__name__}: {e}"
    except ImportError:
        pass
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
    _TORCH_PROBE_CACHE = info
    return info


def _detect_torch_cuda() -> Dict[str, Any]:
    """Backwards-compatible alias for :func:`_detect_torch`."""
    return _detect_torch()


def detect_hardware() -> Dict[str, Any]:
    """Best-effort hardware probe used to pick a sensible default model.

    Reports system RAM, GPU VRAM, GPU model name, unified memory status,
    and whether torch can see CUDA or Apple Silicon Metal (MPS).
    """
    global _TORCH_WARNING_LOGGED
    ram = _detect_total_ram_gb()
    gpu_info = _detect_gpu_info(ram)
    vram = gpu_info["gpu_vram_gb"]
    torch_info = _detect_torch()
    out: Dict[str, Any] = {
        "ram_gb": ram,
        "gpu_vram_gb": vram,
        "gpu_name": gpu_info["gpu_name"],
        "gpu_type": gpu_info["gpu_type"],
        "is_unified_memory": gpu_info["is_unified_memory"],
        "torch_installed": torch_info["installed"],
        "torch_cuda_available": torch_info["cuda_available"],
        "torch_mps_available": torch_info["mps_available"],
        "torch_version": torch_info["torch_version"],
        "torch_cuda_build": torch_info["cuda_build"],
        "torch_cuda_warning": None,
    }
    # Check for hardware / torch acceleration mismatches
    if gpu_info["gpu_type"] == "nvidia" and vram and torch_info["installed"] and not torch_info["cuda_available"]:
        msg = (
            "NVIDIA GPU detected but torch.cuda.is_available() is False; "
            "embeddings will fall back to CPU (~10x slower). Reinstall a "
            "torch wheel that matches your driver's CUDA series, e.g. "
            "`pip install --index-url https://download.pytorch.org/whl/cu128 "
            "torch` (adjust cu1XX to match nvidia-smi's CUDA Version column)."
        )
        if torch_info.get("error"):
            msg += f" Torch reported: {torch_info['error']}"
        out["torch_cuda_warning"] = msg
        if not _TORCH_WARNING_LOGGED:
            logger.warning("ollama_discovery: %s", msg)
            _TORCH_WARNING_LOGGED = True
    elif gpu_info["gpu_type"] == "apple_silicon" and torch_info["installed"] and not torch_info["mps_available"]:
        msg = (
            "Apple Silicon GPU detected but torch.backends.mps.is_available() is False; "
            "embeddings will fall back to CPU. Install a PyTorch build with MPS support (PyTorch 2.0+)."
        )
        if torch_info.get("error"):
            msg += f" Torch reported: {torch_info['error']}"
        out["torch_cuda_warning"] = msg
        if not _TORCH_WARNING_LOGGED:
            logger.warning("ollama_discovery: %s", msg)
            _TORCH_WARNING_LOGGED = True

    return out


def recommend_default_model(installed: Optional[List[Dict[str, Any]]] = None,
                            base_url: str = DEFAULT_BASE_URL) -> str:
    """Pick the best recommended model that is installed, otherwise the most
    capable from the static ladder that fits in available RAM/VRAM."""
    if installed is None:
        installed = list_installed_models(base_url)
    installed_names = {m["name"] for m in installed}
    hw = detect_hardware()
    ram = hw.get("ram_gb") or 0.0
    vram = hw.get("gpu_vram_gb") or 0.0
    is_unified = bool(hw.get("is_unified_memory"))
    if is_unified:
        budget = max(ram, vram)
    elif vram:
        budget = max(ram, vram * 2.0)
    else:
        budget = ram
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


# Regex: leading lowercase-letter run + optional dotted version digits.
# Examples: "qwen2.5-coder" → ("qwen", "2.5"); "gemma4" → ("gemma", "4");
# "deepseek-coder-v2" → ("deepseek", "") with sub="coder-v2".
_PREFIX_VERSION_RE = re.compile(r"^([a-z]+)([\d.]*)", re.IGNORECASE)
# Match a "<N>b" or "<N.M>b" size hint anywhere in the tag suffix.
# Captures e.g. "7b", "1.5b", "e2b" (→ 2), "3.8b-mini-instruct" (→ 3.8).
_SIZE_HINT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*b\b", re.IGNORECASE)


def _family_sort_key(
    name: str,
    params_lookup: Optional[Dict[str, float]] = None,
) -> Tuple[str, str, float, float, str]:
    """Sort key that clusters models by family / version / size.

    Returns ``(family_root, sub_family, version, params_b, name)`` so that
    ``sorted(names, key=_family_sort_key)`` groups e.g. all ``gemma*``
    together, then orders within the group by Gemma version (2 → 3 → 4),
    then by parameter size, then alphabetically.

    ``params_lookup`` lets callers supply an exact ``params_b`` per tag
    (e.g. from :data:`cgx.answer.hardware_matrix.LOCAL_MODEL_CATALOG`);
    when missing, the param count is parsed from the tag suffix
    (``qwen2.5-coder:7b-instruct`` → 7.0) so installed-only models that
    aren't in the catalogue still sort sensibly.
    """
    base = name.split(":", 1)[0]
    suffix = name[len(base) + 1:] if ":" in name else ""

    m = _PREFIX_VERSION_RE.match(base)
    if m:
        family_root = m.group(1).lower()
        ver_str = m.group(2)
        try:
            version = float(ver_str) if ver_str else 0.0
        except ValueError:
            version = 0.0
        sub_family = base[m.end():].lstrip("-").lower()
    else:
        family_root = base.lower()
        version = 0.0
        sub_family = ""

    params: Optional[float] = None
    if params_lookup is not None:
        params = params_lookup.get(name)
    if params is None:
        sm = _SIZE_HINT_RE.search(suffix)
        params = float(sm.group(1)) if sm else 0.0

    return (family_root, sub_family, version, params, name)


def sort_model_choices_by_family(
    names: List[str],
    params_lookup: Optional[Dict[str, float]] = None,
) -> List[str]:
    """Return ``names`` sorted to cluster related models together.

    Stable across runs (pure function of the inputs). Duplicates are
    preserved -- callers should de-dup upstream.
    """
    return sorted(names, key=lambda n: _family_sort_key(n, params_lookup))
