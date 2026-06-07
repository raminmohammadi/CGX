

"""Curated catalog of embedding models offered in the web UI.

Each entry maps to a Hugging Face Hub repo that the existing loader in
:mod:`cgx.embeddings.build` already knows how to run (Sentence-Transformers
preferred, plain transformers fallback). :func:`is_cached` mirrors the cache
probe in :mod:`cgx.webui.launch` so the UI can decide whether to show a
"Pull" button or a "ready" badge for each model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class EmbedModel:
    """Static metadata for a UI-selectable embedding model."""

    name: str
    label: str
    dim: int
    max_tokens: int
    size_gb: float
    description: str
    kind: str = "huggingface"


EMBED_MODEL_CATALOG: List[EmbedModel] = [
    EmbedModel(
        name="jinaai/jina-embeddings-v2-base-code",
        label="Jina v2 Base Code",
        dim=768,
        max_tokens=8192,
        size_gb=0.32,
        description="Long-context code embedder. Default -- small and fast.",
    ),
    EmbedModel(
        name="BAAI/bge-m3",
        label="BGE-M3 (multilingual)",
        dim=1024,
        max_tokens=8192,
        size_gb=2.27,
        description="Multilingual dense retrieval. Strong general-purpose.",
    ),
    EmbedModel(
        name="Qwen/Qwen3-Embedding-8B",
        label="Qwen3 Embedding 8B",
        dim=4096,
        max_tokens=32768,
        size_gb=15.3,
        description="Highest quality. Needs ~16GB free disk and a GPU.",
    ),
]


def _hf_cache_root() -> Path:
    """Resolve the Hugging Face Hub cache directory using the standard env vars."""
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hub_cache:
        return Path(hub_cache)
    return Path.home() / ".cache" / "huggingface" / "hub"


def is_cached(model_name: str) -> bool:
    """Return True when a snapshot directory exists for ``model_name``."""
    safe = "models--" + model_name.replace("/", "--")
    snapshots = _hf_cache_root() / safe / "snapshots"
    try:
        return snapshots.is_dir() and any(snapshots.iterdir())
    except OSError:
        return False


def find_by_name(model_name: str) -> Optional[EmbedModel]:
    for m in EMBED_MODEL_CATALOG:
        if m.name == model_name:
            return m
    return None
