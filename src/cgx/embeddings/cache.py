# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ramin Mohammadi

"""Content-addressed embedding cache for incremental indexing.

The cache stores ``{sha256(corpus_text): np.ndarray}`` pairs on disk so
unchanged corpus rows skip the model entirely on re-index. Because the
key is the *text submitted to the embedder* (not the file path), the
cache is robust to file moves and partial-file edits: only modified
chunks miss the cache.

On-disk layout (inside ``<out_dir>/<cache_filename>``)::

    embedding_cache.npz
      ├── keys   : (N,) np.str_     — hex sha256 strings
      ├── values : (N, D) np.float32 embeddings
      └── meta   : () np.str_       — JSON-encoded {model_name, dim, normalize, version}

If the model name / dim / normalisation flag stored in ``meta`` does not
match the current invocation, the cache is treated as empty (the model
or its config changed and reusing stale vectors would be wrong).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np


# Bump this whenever a change in the embedding pipeline could invalidate
# previously cached vectors (e.g. fixing model loading, changing tokenisation,
# switching normalisation). Old caches with a different ``version`` field in
# their meta blob are treated as empty by ``load_cache``.
_CACHE_VERSION = 2


def hash_text(text: str) -> str:
    """Return the canonical sha256 hex digest used as the cache key."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _meta_dict(*, model_name: str, dim: int, normalize: bool) -> Dict[str, Any]:
    return {"version": _CACHE_VERSION, "model_name": str(model_name),
            "dim": int(dim), "normalize": bool(normalize)}


def load_cache(path: str, *, expected_meta: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Load ``{key: vector}`` from ``path``.

    Returns an empty dict if the file is missing, unreadable, or its
    ``meta`` blob disagrees with ``expected_meta`` on the non-version
    fields (``model_name``, ``dim``, ``normalize``).
    """
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with np.load(p, allow_pickle=False) as npz:
            if "meta" not in npz.files or "keys" not in npz.files or "values" not in npz.files:
                return {}
            meta = json.loads(str(npz["meta"].item()))
            # Reject caches written by older pipeline versions: their vectors
            # may have been produced by a broken model load (e.g. randomly
            # initialised weights) and reusing them silently is worse than
            # re-encoding.
            try:
                cached_version = int(meta.get("version", 0))
            except Exception:
                cached_version = 0
            if cached_version != _CACHE_VERSION:
                return {}
            for k in ("model_name", "dim", "normalize"):
                # ``dim == 0`` in the expected meta acts as a wildcard so
                # callers can probe without yet knowing the model dim.
                if k == "dim" and int(expected_meta.get(k) or 0) == 0:
                    continue
                if str(meta.get(k)) != str(expected_meta.get(k)):
                    return {}
            keys = [str(k) for k in npz["keys"].tolist()]
            values = np.asarray(npz["values"], dtype=np.float32)
        if len(keys) != values.shape[0]:
            return {}
        return {k: values[i] for i, k in enumerate(keys)}
    except Exception:
        return {}


def save_cache(path: str, store: Dict[str, np.ndarray], *,
               model_name: str, dim: int, normalize: bool) -> None:
    """Persist ``store`` to ``path`` atomically."""
    if not store:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    keys = list(store.keys())
    values = np.stack([np.asarray(store[k], dtype=np.float32) for k in keys], axis=0)
    meta = _meta_dict(model_name=model_name, dim=dim, normalize=normalize)
    # ``np.savez`` appends ``.npz`` when the path lacks that suffix, so we
    # name the temp file with a ``.npz`` extension already (just disambiguated)
    # to ensure the post-write rename targets a real file.
    # Use fixed-width unicode dtypes so the archive is loadable with
    # ``allow_pickle=False`` (object arrays require pickling).
    tmp = p.parent / (p.name + ".tmp.npz")
    np.savez(
        str(tmp),
        keys=np.asarray(keys, dtype=np.str_),
        values=values,
        meta=np.asarray(json.dumps(meta), dtype=np.str_),
    )
    os.replace(tmp, p)


def embed_with_cache(
    texts: List[str],
    *,
    encode_fn: Callable[[List[str]], np.ndarray],
    cache_path: str,
    model_name: str,
    normalize: bool,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Return embeddings for ``texts``, consulting + updating the cache.

    Parameters
    ----------
    texts
        Corpus texts to embed. Empty / non-string entries are still
        encoded; the caller is responsible for input sanitisation.
    encode_fn
        Function that takes a list of strings and returns an
        ``np.ndarray`` of shape ``(N, D)``. Called only on cache misses.
    cache_path
        Where to load / persist the cache. If the file is missing the
        cache starts empty.
    model_name, normalize
        Identifying the model + post-processing so a stale cache from a
        different model is ignored.

    Returns
    -------
    (embs, stats)
        ``embs`` is shape ``(len(texts), D)``. ``stats`` reports
        ``hits``, ``misses``, and the inferred ``dim``.
    """
    n = len(texts)
    keys = [hash_text(t) for t in texts]

    # Load the cache. ``dim=0`` is treated as a wildcard by ``load_cache``
    # so we don't need to know the model dim ahead of time.
    cache = load_cache(cache_path, expected_meta=_meta_dict(
        model_name=model_name, dim=0, normalize=normalize))

    missing_idx: List[int] = [i for i, k in enumerate(keys) if k not in cache]
    if missing_idx:
        missing_texts = [texts[i] for i in missing_idx]
        new_embs = np.asarray(encode_fn(missing_texts), dtype=np.float32)
        if new_embs.ndim != 2 or new_embs.shape[0] != len(missing_texts):
            raise RuntimeError(
                f"encode_fn returned shape {new_embs.shape}, expected "
                f"({len(missing_texts)}, D)")
        for j, i in enumerate(missing_idx):
            cache[keys[i]] = new_embs[j]
        dim = int(new_embs.shape[1])
    elif cache:
        dim = int(next(iter(cache.values())).shape[0])
    else:
        # Pathological: no texts at all.
        return np.zeros((0, 0), dtype=np.float32), {"hits": 0, "misses": 0, "dim": 0}

    embs = np.stack([cache[k] for k in keys], axis=0) if n else np.zeros((0, dim), dtype=np.float32)
    save_cache(cache_path, cache, model_name=model_name, dim=dim, normalize=normalize)
    return embs, {"hits": n - len(missing_idx), "misses": len(missing_idx), "dim": dim}
