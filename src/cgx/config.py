

# src/cgx/config.py
from __future__ import annotations

"""
Typed configuration objects for cgx.

- EmbeddingConfig: model + batching knobs.
- FaissConfig: index type/metric and ANN tuning.
- HybridSearchConfig: RRF + graph expansion knobs and final top-k.

These configs are **add-only** and do not assume any specific models. They can be
instantiated from environment variables and/or dict overrides without side-effects.
"""

import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if (v is not None and v != "") else default


def _as_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _as_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _as_bool(v: Any, default: bool) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


@dataclass(slots=True)
class EmbeddingConfig:
    """
    Configuration for embedding text (no model assumption).
    """
    model_name: str = field(default_factory=lambda: _env("CGX_EMBED_MODEL", "jinaai/jina-embeddings-v2-base-code"))
    batch_size: int = field(default_factory=lambda: _as_int(_env("CGX_EMBED_BATCH", "64"), 64))
    max_length: int = field(default_factory=lambda: _as_int(_env("CGX_EMBED_MAXLEN", "8192"), 8192))
    device: Optional[str] = field(default_factory=lambda: _env("CGX_EMBED_DEVICE", None))  # None -> auto

    @classmethod
    def from_overrides(cls, **overrides: Any) -> "EmbeddingConfig":
        obj = cls()
        for k, v in overrides.items():
            if hasattr(obj, k):
                setattr(obj, k, v)
        return obj

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FaissConfig:
    """
    Configuration for FAISS vector index.
    """
    metric: str = field(default_factory=lambda: _env("CGX_FAISS_METRIC", "cosine"))  # cosine | l2 | ip
    index: str = field(default_factory=lambda: _env("CGX_FAISS_INDEX", "flat"))       # flat | ivf | hnsw
    nlist: int = field(default_factory=lambda: _as_int(_env("CGX_FAISS_NLIST", "1024"), 1024))
    nprobe: int = field(default_factory=lambda: _as_int(_env("CGX_FAISS_NPROBE", "16"), 16))
    M: int = field(default_factory=lambda: _as_int(_env("CGX_FAISS_HNSW_M", "32"), 32))
    efConstruction: int = field(default_factory=lambda: _as_int(_env("CGX_FAISS_HNSW_EFC", "200"), 200))
    efSearch: int = field(default_factory=lambda: _as_int(_env("CGX_FAISS_HNSW_EFS", "64"), 64))
    use_gpu: bool = field(default_factory=lambda: _as_bool(_env("CGX_FAISS_GPU", "0"), False))

    @classmethod
    def from_overrides(cls, **overrides: Any) -> "FaissConfig":
        obj = cls()
        for k, v in overrides.items():
            if hasattr(obj, k):
                setattr(obj, k, v)
        return obj

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class HybridSearchConfig:
    """
    Configuration for hybrid search (semantic + lexical + graph + RRF).
    """
    top_k: int = field(default_factory=lambda: _as_int(_env("CGX_TOP_K", "30"), 30))
    k_intent: int = field(default_factory=lambda: _as_int(_env("CGX_K_INTENT", "50"), 50))
    k_impl: int = field(default_factory=lambda: _as_int(_env("CGX_K_IMPL", "50"), 50))
    k_lex: int = field(default_factory=lambda: _as_int(_env("CGX_K_LEX", "50"), 50))
    expand_top_n: int = field(default_factory=lambda: _as_int(_env("CGX_EXPAND_TOP_N", "10"), 10))
    expand_per_seed: int = field(default_factory=lambda: _as_int(_env("CGX_EXPAND_PER_SEED", "12"), 12))
    rrf_k: float = field(default_factory=lambda: _as_float(_env("CGX_RRF_K", "60.0"), 60.0))
    build_graph: bool = field(default_factory=lambda: _as_bool(_env("CGX_BUILD_GRAPH", "1"), True))

    @classmethod
    def from_overrides(cls, **overrides: Any) -> "HybridSearchConfig":
        obj = cls()
        for k, v in overrides.items():
            if hasattr(obj, k):
                setattr(obj, k, v)
        return obj

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
