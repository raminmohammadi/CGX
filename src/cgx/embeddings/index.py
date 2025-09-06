import numpy as np

def build_faiss_index(
    embeddings: np.ndarray,
    *,
    metric: str = "cosine",          # "cosine" | "l2" | "ip" (inner product / dot)
    index: str = "flat",             # "flat" | "ivf" | "hnsw"
    ids: np.ndarray | None = None,   # optional int64 ids to attach; default: 0..N-1
    normalize: bool | None = None,   # None -> infer; True -> L2-normalize; False -> assume already normalized
    # IVF params
    nlist: int = 1024,
    nprobe: int = 16,
    # HNSW params
    M: int = 32,
    efConstruction: int = 200,
    efSearch: int = 64,
    # GPU
    use_gpu: bool = False,
    gpu_device: int = 0,
    # Return extra info
    return_meta: bool = False,
):
    """
    Build a FAISS index for fast nearest-neighbor search over embeddings.

    Parameters
    ----------
    embeddings : np.ndarray, shape (N, D), dtype float32/float64
        Dense vectors (e.g., from `build_embeddings`). If metric="cosine",
        vectors MUST be L2-normalized. If not, set normalize=True or leave
        normalize=None (auto-infer & fix).
    metric : {"cosine","l2","ip"}, default "cosine"
        - "cosine" -> uses FAISS inner product on L2-normalized vectors
        - "l2"     -> Euclidean distance
        - "ip"     -> raw inner product (no normalization)
    index : {"flat","ivf","hnsw"}, default "flat"
        - "flat": exact search (IndexFlat*)
        - "ivf" : inverted lists (IndexIVFFlat) — must be trained
        - "hnsw": graph-based (IndexHNSWFlat)
    ids : np.ndarray[int64] | None
        Custom IDs to attach (length N). If None, uses [0..N-1].
    normalize : bool | None
        For metric="cosine":
          - True  -> L2-normalize a copy of embeddings before indexing
          - False -> assume embeddings already normalized (warn if not)
          - None  -> infer by checking norms; normalize if needed
        Ignored for "l2" and "ip".
    nlist, nprobe : IVF params
    M, efConstruction, efSearch : HNSW params
    use_gpu : bool
        Try to move index to GPU (falls back to CPU if unavailable).
    return_meta : bool
        If True, return (index, meta_dict).

    Returns
    -------
    faiss.Index or (faiss.Index, dict)
        Ready for `.search(Q, k)`. If return_meta=True, also returns:
        {
          "dim", "metric", "faiss_metric", "index_type",
          "nlist","nprobe","M","efConstruction","efSearch",
          "used_gpu","normalized_for_cosine","num_vectors"
        }

    Notes
    -----
    - For COSINE search, normalize query vectors the same way before searching:
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    """
    try:
        import faiss
    except Exception as e:
        raise RuntimeError("FAISS is not installed. Install `faiss-cpu` or `faiss-gpu`.") from e

    # ------------------------- validation -------------------------
    if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D numpy array of shape (N, D).")
    N, D = embeddings.shape
    if N == 0 or D == 0:
        raise ValueError(f"embeddings must have non-zero shape; got ({N}, {D}).")

    # ids
    if ids is None:
        ids = np.arange(N, dtype=np.int64)
    else:
        ids = np.asarray(ids)
        if ids.dtype != np.int64:
            ids = ids.astype(np.int64, copy=False)
        if ids.shape != (N,):
            raise ValueError(f"ids must have shape (N,), got {ids.shape} for N={N}.")
        if len(np.unique(ids)) != N:
            raise ValueError("ids must be unique.")

    # dtype & contiguity
    X = embeddings.astype(np.float32, copy=False)
    if not X.flags["C_CONTIGUOUS"]:
        X = np.ascontiguousarray(X, dtype=np.float32)

    # ------------------------- metric handling -------------------------
    metric_l = metric.lower()
    if metric_l not in {"cosine", "l2", "ip", "inner_product", "dot"}:
        raise ValueError(f"Unsupported metric '{metric}'. Use 'cosine', 'l2', or 'ip'.")
    if metric_l in {"inner_product", "dot"}:
        metric_l = "ip"

    use_ip = (metric_l in {"cosine", "ip"})
    faiss_metric = faiss.METRIC_INNER_PRODUCT if use_ip else faiss.METRIC_L2

    # For cosine: ensure L2 normalization (either provided or inferred)
    normalized_used = False
    if metric_l == "cosine":
        if normalize is None:
            norms = np.linalg.norm(X, axis=1)
            if not np.all(np.isfinite(norms)) or np.max(np.abs(norms - 1.0)) > 1e-3:
                X = X / (norms[:, None] + 1e-12)
                normalized_used = True
            else:
                normalized_used = True  # already normalized
        elif normalize:
            norms = np.linalg.norm(X, axis=1)
            X = X / (norms[:, None] + 1e-12)
            normalized_used = True
        else:
            norms = np.linalg.norm(X, axis=1)
            if np.max(np.abs(norms - 1.0)) > 5e-2:
                print("build_faiss_index: warning: metric='cosine' but vectors do not appear normalized.")
            normalized_used = False

    # ------------------------- index construction -------------------------
    def _make_flat():
        return faiss.IndexFlatIP(D) if use_ip else faiss.IndexFlatL2(D)

    def _make_ivf(nlist_val: int):
        quantizer = _make_flat()
        return faiss.IndexIVFFlat(quantizer, D, int(nlist_val), faiss_metric)

    def _make_hnsw():
        try:
            h = faiss.IndexHNSWFlat(D, int(M), faiss_metric)
        except TypeError:
            h = faiss.IndexHNSWFlat(D, int(M))
        h.hnsw.efConstruction = int(efConstruction)
        h.hnsw.efSearch = int(efSearch)
        return h

    idx_type = index.lower()
    if idx_type == "flat":
        base = _make_flat()
    elif idx_type == "ivf":
        # ensure nlist reasonable for the dataset size
        nlist_eff = int(nlist)
        if N < nlist_eff:
            nlist_eff = max(1, int(max(1, round(np.sqrt(N)))))
        base = _make_ivf(nlist_eff)
    elif idx_type == "hnsw":
        base = _make_hnsw()
    else:
        raise ValueError(f"Unsupported index '{index}'. Use 'flat', 'ivf', or 'hnsw'.")

    # Train IVF on CPU (simplest); set nprobe
    if hasattr(base, "is_trained") and not base.is_trained:
        base.train(X)
    if hasattr(base, "nprobe"):
        base.nprobe = int(nprobe)

    # ------------------------- optional GPU -------------------------
    used_gpu = False
    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            base = faiss.index_cpu_to_gpu(res, int(gpu_device), base)
            used_gpu = True
            # re-apply nprobe if IVF on GPU
            if hasattr(base, "nprobe"):
                base.nprobe = int(nprobe)
        except Exception as e:
            print(f"build_faiss_index: GPU requested but unavailable; using CPU. ({e})")
            used_gpu = False

    # Wrap with ID map (IndexIDMap2 preferred)
    try:
        idmap = faiss.IndexIDMap2(base)
    except Exception:
        idmap = faiss.IndexIDMap(base)

    # Add vectors
    idmap.add_with_ids(X, ids)

    if return_meta:
        meta = {
            "dim": D,
            "metric": metric_l,
            "faiss_metric": "IP" if use_ip else "L2",
            "index_type": idx_type,
            "nlist": int(nlist if idx_type == "ivf" else 0) or (base.nlist if hasattr(base, "nlist") else None),
            "nprobe": int(nprobe) if idx_type == "ivf" else None,
            "M": int(M) if idx_type == "hnsw" else None,
            "efConstruction": int(efConstruction) if idx_type == "hnsw" else None,
            "efSearch": int(efSearch) if idx_type == "hnsw" else None,
            "used_gpu": used_gpu,
            "normalized_for_cosine": normalized_used if metric_l == "cosine" else None,
            "num_vectors": int(N),
        }
        return idmap, meta

    return idmap
