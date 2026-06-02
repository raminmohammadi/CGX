# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ramin Mohammadi

# src/cgx/retrieval/ann_numpy.py
from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None


def _l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


class _NumPyIndex:
    """
    Minimal FAISS-like index:
      .search(Q, k) -> (D, I)
    For metric in {"cosine","ip"}: D are similarity scores (higher is better).
    For metric == "l2": D are L2 distances (lower is better).
    """
    def __init__(self, X: np.ndarray, metric: str):
        self.X = X.astype("float32", copy=False)
        self.metric = metric.lower()
        # meta-like hints for your TwoViewIndex.search_view
        self.faiss_metric = "IP" if self.metric in {"cosine", "ip"} else "L2"

    def search(self, Q: np.ndarray, k: int):
        Q = Q.astype("float32", copy=False)
        if Q.ndim == 1:
            Q = Q[None, :]

        if self.metric in {"cosine", "ip"}:
            # assume caller normalized Q if they want cosine behavior
            scores = self.X @ Q.T  # (N, 1)
            s = scores.ravel()
            k = min(k, s.shape[0])
            top = np.argpartition(-s, kth=k-1)[:k]
            top = top[np.argsort(-s[top], kind="mergesort")]
            return s[top][None, :].astype("float32"), top[None, :].astype("int64")
        else:
            # L2 distance
            # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
            aa = (self.X ** 2).sum(axis=1)
            bb = (Q ** 2).sum(axis=1)
            ab = self.X @ Q.T
            d2 = aa[:, None] + bb[None, :] - 2 * ab
            d = d2.ravel()
            k = min(k, d.shape[0])
            top = np.argpartition(d, kth=k-1)[:k]
            top = top[np.argsort(d[top], kind="mergesort")]
            return d[top][None, :].astype("float32"), top[None, :].astype("int64")


def _make_faiss_index(dim: int, metric: str, index: str, **kw):
    use_ip = (metric.lower() in {"cosine", "ip"})
    if index == "flat":
        return faiss.IndexFlatIP(dim) if use_ip else faiss.IndexFlatL2(dim)
    if index == "hnsw":
        M = int(kw.get("M", 32))
        efC = int(kw.get("efConstruction", 200))
        efS = int(kw.get("efSearch", 64))
        idx = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT if use_ip else faiss.METRIC_L2)
        idx.hnsw.efConstruction = efC
        idx.hnsw.efSearch = efS
        return idx
    if index == "ivf":
        nlist = int(kw.get("nlist", 1024))
        quant = faiss.IndexFlatIP(dim) if use_ip else faiss.IndexFlatL2(dim)
        return faiss.IndexIVFFlat(
            quant, dim, nlist,
            faiss.METRIC_INNER_PRODUCT if use_ip else faiss.METRIC_L2
        )
    # fallback
    return faiss.IndexFlatIP(dim) if use_ip else faiss.IndexFlatL2(dim)


def build_ann_index(
    embeddings: np.ndarray,
    *,
    ids: Optional[np.ndarray] = None,
    metric: str = "cosine",
    index: str = "flat",
    normalize: Optional[bool] = None,
    use_gpu: bool = False,
    **kwargs: Any,
):
    """
    Adapter that matches your `index_builder` callable signature.

    Returns:
        (index_like, meta_dict)

    - If FAISS is available, builds the requested index (flat/hnsw/ivf).
    - Otherwise, returns a NumPy-based index with a FAISS-like `.search`.
    - For metric in {"cosine","ip"}, vectors are L2-normalized so IP==cosine.
    """
    X = np.asarray(embeddings, dtype="float32")
    if X.ndim != 2:
        raise ValueError(f"build_ann_index: embeddings must be (N,D), got {X.shape}")

    metric = (metric or "cosine").lower()
    want_cos = metric in {"cosine", "ip"}

    if normalize is True or (normalize is None and want_cos):
        X = _l2_normalize_rows(X)

    meta: Dict[str, Any] = {
        "metric": metric,
        "index_type": index,
        "normalized_for_cosine": bool(normalize is True or (normalize is None and want_cos)),
        "faiss_metric": "IP" if want_cos else "L2",
        "use_gpu": bool(use_gpu),
        "dim": int(X.shape[1]),
    }

    if faiss is None:
        logger.info("build_ann_index: FAISS not found, using NumPy fallback.")
        return _NumPyIndex(X, metric=metric), meta

    # Build FAISS index
    idx = _make_faiss_index(X.shape[1], metric, index, **kwargs)

    # IVF needs training
    try:
        if isinstance(idx, getattr(faiss, "IndexIVFFlat", ())):
            if not idx.is_trained:
                idx.train(X)
            nprobe = int(kwargs.get("nprobe", 16))
            idx.nprobe = nprobe
            meta["nprobe"] = nprobe
    except Exception:
        pass

    # GPU?
    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            idx = faiss.index_cpu_to_gpu(res, 0, idx)
            meta["gpu"] = True
        except Exception as e:
            logger.warning("build_ann_index: GPU init failed; falling back to CPU (%s)", e)
            meta["gpu"] = False

    # Add vectors
    idx.add(X)

    return idx, meta
