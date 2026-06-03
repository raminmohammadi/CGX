

import numpy as np
import re
from typing import Any, Callable, Dict, List, Optional, Sequence



def semantic_search(
    query: str,
    embed_model: Any,                 # SentenceTransformer OR callable(texts)->np.ndarray
    index,                            # faiss.Index(…)
    chunks: List[Dict],
    *,
    top_k: int = 5,
    metric: Optional[str] = None,     # "cosine" | "l2" | "ip" ; if None, try to infer/assume cosine
    normalize_query: Optional[bool] = None,  # for cosine; None=auto
    index_meta: Optional[Dict] = None # optional meta returned by build_faiss_index(return_meta=True)
) -> List[Dict]:
    """
    Perform semantic search over a codebase using embeddings + FAISS.

    Parameters
    ----------
    query : str
        Natural-language text or code snippet.
    embed_model : object or callable
        - SentenceTransformers model with `.encode([text])`, OR
        - Callable that takes List[str] and returns np.ndarray (Q, D).
    index : faiss.Index
        FAISS index built from code embeddings.
    chunks : list[dict]
        Parsed chunks; must align with the ids used when building the index
        (typically ids = 0..N-1 in same order as `chunks`).
    top_k : int
        Number of nearest neighbors to return.
    metric : str or None
        If provided, one of {"cosine","l2","ip"}. If None, inferred from index_meta
        or defaults to "cosine".
    normalize_query : bool or None
        For cosine/IP: whether to L2-normalize query vectors. If None, auto-infer.
    index_meta : dict or None
        Optional metadata returned by build_faiss_index(..., return_meta=True).
        Used to infer metric and normalization.

    Returns
    -------
    list[dict]
        List of {"chunk": <chunk dict>, "distance": float, "score": float} ordered
        by descending score (higher is better). Distance semantics depend on metric.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("semantic_search: 'query' must be a non-empty string.")
    if not isinstance(chunks, list) or len(chunks) == 0:
        raise ValueError("semantic_search: 'chunks' must be a non-empty list.")

    # ---- infer metric & normalization intent ----
    m = (metric or (index_meta or {}).get("metric") or "cosine").lower()
    if m not in {"cosine", "l2", "ip", "inner_product", "dot"}:
        raise ValueError(f"semantic_search: unsupported metric '{m}'.")
    if m == "inner_product" or m == "dot":
        m = "ip"

    # ---- encode query vector ----
    if hasattr(embed_model, "encode"):
        q_vec = embed_model.encode([query], show_progress_bar=False)  # ST path
    elif callable(embed_model):
        q_vec = embed_model([query])  # custom callable
    else:
        raise TypeError("semantic_search: 'embed_model' must provide .encode([texts]) or be a callable(texts)->ndarray.")
    q_vec = np.asarray(q_vec, dtype=np.float32)
    if q_vec.ndim != 2 or q_vec.shape[0] != 1:
        raise ValueError(f"semantic_search: encoder returned shape {q_vec.shape}, expected (1, D).")

    # ---- normalize for cosine/IP if requested/inferred ----
    if m == "cosine" or m == "ip":
        # auto if not specified
        if normalize_query is None:
            # If the index_meta says it normalized database vectors, normalize query too
            normalized_db = (index_meta or {}).get("normalized_for_cosine", None)
            normalize_query = True if (m == "cosine" and normalized_db is not False) else False
        if normalize_query:
            denom = np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-12
            q_vec = q_vec / denom

    # ---- run search ----
    D, I = index.search(q_vec, int(top_k))

    # ---- unify scoring: higher 'score' is better; keep raw 'distance' too ----
    out: List[Dict] = []
    distances = D[0]
    indices = I[0]
    for d, idx in zip(distances, indices):
        if idx == -1:
            continue
        # Scoring:
        # - cosine (IndexIP): FAISS returns IP where higher is better -> score=d; distance ~ 1 - d
        # - ip              : same as above
        # - l2              : FAISS returns L2 distance; convert to a score using -distance
        if m in {"cosine", "ip"}:
            score = float(d)
            distance = float(1.0 - d)  # optional intuitive view
        else:  # l2
            score = float(-d)
            distance = float(d)
        # Map back to chunk
        try:
            chunk = chunks[int(idx)]
        except Exception:
            # If you used custom IDs, pass index_meta and a mapping instead.
            raise RuntimeError("semantic_search: chunk lookup failed. Ensure FAISS ids align with `chunks` or provide a mapping.")
        out.append({"chunk": chunk, "distance": distance, "score": score})

    # sort by score desc (FAISS usually returns sorted, but enforce)
    out.sort(key=lambda r: r["score"], reverse=True)
    return out
