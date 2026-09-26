

from __future__ import annotations

"""Hardware-aware embedding budget: pick a safe (batch_size, max_length).

Indexing a large repo with the local torch embedder OOM-crashed laptops because
the pipeline embedded ``batch_size=64`` sequences padded to ``max_length=8192``.
The default code embedder (``jinaai/jina-embeddings-v2-base-code``) uses ALiBi
attention with no flash kernel on the CPU/MPS path, so the attention scores it
materializes are **O(L^2)**:

    attention_peak_bytes ~= batch_size * n_heads * max_length**2 * dtype_bytes

At (batch=64, L=8192, fp32) that is ~206 GB -- an instant out-of-memory freeze
on any laptop. The fix is to derive both knobs from the machine's actual memory
budget so the attention peak stays a small fraction of what's available, then
let the caller override explicitly.

Resolution order for each knob:
  1. explicit env override (``CGX_EMBED_MAXLEN`` / ``CGX_EMBED_BATCH``) -- the
     user asked for a value, honour it verbatim;
  2. otherwise auto-tune from detected RAM / VRAM (unified vs dedicated).

This module imports nothing heavy (no torch); it consumes the plain dict from
:func:`cgx.answer.ollama_discovery.detect_hardware`.
"""

import os
from typing import Any, Dict, Optional

from cgx.logging_setup import get_logger

logger = get_logger(__name__)

# jina-v2-base-code shape (representative for the default embedder). Used only
# to size the O(L^2) attention peak conservatively; a different model with
# fewer heads simply gets extra headroom.
_N_HEADS = 12
_DTYPE_BYTES = 4  # assume fp32 (the worst case on CPU/MPS)

# Fraction of the effective memory pool we allow the attention scores to use.
# Deliberately small: the same process also holds the model weights, the full
# chunk/record corpus, every embedding vector, the FAISS flat index, and the
# in-RAM BM25 postings -- embedding is not the only tenant.
_ATTENTION_BUDGET_FRACTION = 0.15
_ATTENTION_BUDGET_CAP_GB = 3.0

# Auto max_length by effective memory (GB). Capped at 2048 even on big
# machines: the vast majority of code chunks fit well under it, and the
# quadratic cost of going higher is rarely worth it. Users who truly need the
# model's full 8192 window set CGX_EMBED_MAXLEN explicitly.
_MAXLEN_TIERS = (
    (4.0, 512),
    (8.0, 1024),
    (16.0, 1536),
    (float("inf"), 2048),
)

_BATCH_MIN, _BATCH_MAX = 1, 32
_FALLBACK = {"batch_size": 8, "max_length": 1024}


def _effective_memory_gb(hw: Dict[str, Any]) -> float:
    """The memory pool that actually bounds a torch embedding pass, in GB.

    * Dedicated CUDA GPU -> VRAM is the hard ceiling (torch allocates there);
      keep 70% headroom for weights + workspace.
    * Apple / unified memory -> the GPU shares system RAM with the OS and the
      rest of the pipeline, so budget a conservative ~55% of total RAM.
    * CPU-only -> activations live in system RAM alongside the whole corpus +
      FAISS; budget ~40%.
    """
    ram = float(hw.get("ram_gb") or 0.0)
    vram = float(hw.get("gpu_vram_gb") or 0.0)
    if hw.get("torch_cuda_available") and not hw.get("is_unified_memory") and vram > 0:
        return vram * 0.70
    if hw.get("is_unified_memory") or hw.get("torch_mps_available"):
        return (ram or vram) * 0.55
    return ram * 0.40 if ram > 0 else 4.0  # unknown -> assume a small laptop


def _auto_max_length(effective_gb: float) -> int:
    for ceiling, value in _MAXLEN_TIERS:
        if effective_gb < ceiling:
            return value
    return _MAXLEN_TIERS[-1][1]


def _auto_batch_size(effective_gb: float, max_length: int) -> int:
    """Largest batch whose O(L^2) attention peak stays within the budget."""
    budget_gb = min(_ATTENTION_BUDGET_CAP_GB, _ATTENTION_BUDGET_FRACTION * effective_gb)
    budget_bytes = max(0.25, budget_gb) * (1024 ** 3)
    per_seq = _N_HEADS * (max_length ** 2) * _DTYPE_BYTES
    batch = int(budget_bytes // per_seq) if per_seq else _BATCH_MAX
    return max(_BATCH_MIN, min(_BATCH_MAX, batch))


def _env_int(name: str) -> Optional[int]:
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        v = int(raw)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def resolve_embed_budget(hardware: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return ``{batch_size, max_length, effective_gb, basis}``.

    Explicit ``CGX_EMBED_MAXLEN`` / ``CGX_EMBED_BATCH`` env values win per-knob;
    anything not pinned is auto-tuned from detected memory so a large-repo index
    can't OOM the machine.
    """
    if hardware is None:
        try:
            from cgx.answer.ollama_discovery import detect_hardware
            hardware = detect_hardware()
        except Exception as e:  # pragma: no cover - detection must never fail the build
            logger.warning("autotune: hardware detection failed (%s); using conservative defaults", e)
            hardware = {}

    effective_gb = _effective_memory_gb(hardware or {})
    if effective_gb <= 0:
        max_length, batch_size = _FALLBACK["max_length"], _FALLBACK["batch_size"]
    else:
        max_length = _auto_max_length(effective_gb)
        batch_size = _auto_batch_size(effective_gb, max_length)

    env_max, env_batch = _env_int("CGX_EMBED_MAXLEN"), _env_int("CGX_EMBED_BATCH")
    basis_bits = []
    if env_max is not None:
        max_length = env_max
        basis_bits.append("maxlen=env")
    else:
        basis_bits.append("maxlen=auto")
    if env_batch is not None:
        # A user-pinned max_length may still overflow, but an explicit batch is
        # an explicit choice -- honour it.
        batch_size = env_batch
        basis_bits.append("batch=env")
    else:
        # If max_length was pinned via env but batch was not, re-derive batch so
        # the pair stays memory-safe for that (possibly large) length.
        if env_max is not None and effective_gb > 0:
            batch_size = _auto_batch_size(effective_gb, max_length)
        basis_bits.append("batch=auto")

    result = {
        "batch_size": int(batch_size),
        "max_length": int(max_length),
        "effective_gb": round(float(effective_gb), 1),
        "basis": ",".join(basis_bits),
    }
    logger.info(
        "autotune: embed budget -> batch_size=%d max_length=%d (effective_mem=%.1fGB, %s)",
        result["batch_size"], result["max_length"], result["effective_gb"], result["basis"],
    )
    return result


__all__ = ["resolve_embed_budget"]
