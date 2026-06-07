

"""Embedding builder.

Heavy ML dependencies (``torch``, ``transformers``, ``sentence_transformers``)
are imported lazily inside :func:`build_embeddings` so that simply importing
this module -- or any module that re-exports it (e.g. ``cgx.pipeline.auto``) --
does not require the full ML stack. This keeps the UI and API-only providers
usable on machines without a local embedder installed.
"""

from typing import Any, List, Dict, Optional, Callable, Tuple
import threading
import numpy as np


# Module-level caches so the encoder model is loaded once per process and
# reused across calls. Without this the query path (which calls
# build_embeddings on every .encode()) reloads the model 2-3× per question,
# which is slow and floods the logs with HF init warnings.
_ST_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}
_HF_MODEL_CACHE: Dict[Tuple[str, str], Tuple[Any, Any]] = {}

# Locks prevent concurrent model loads (e.g. hybrid_retrieve_two_view runs
# intent and impl views in a ThreadPoolExecutor). Without them two threads
# both miss the cache, both call SentenceTransformer(device="cuda") in
# parallel, and the concurrent CUDA allocations trigger the meta-tensor error.
_ST_MODEL_LOCK = threading.Lock()
_HF_MODEL_LOCK = threading.Lock()


def _get_st_model(model_name: str, device: str, max_length: int) -> Any:
    """Return a (cached) SentenceTransformer for (model_name, device).

    trust_remote_code is required for models like jina-embeddings-v2-* which
    ship a custom BERT (GLU MLP + ALiBi). Without it, ST falls back to vanilla
    BertModel, leaves the encoder layers randomly initialized, and every
    input collapses to the same vector.
    """
    key = (model_name, device)
    # Fast path: model already cached.
    st_model = _ST_MODEL_CACHE.get(key)
    if st_model is None:
        # Slow path: acquire lock so concurrent callers (e.g. two ThreadPoolExecutor
        # workers encoding intent/impl views simultaneously) don't both try to load
        # the model on CUDA at the same time, which triggers the meta-tensor error.
        with _ST_MODEL_LOCK:
            st_model = _ST_MODEL_CACHE.get(key)  # re-check after acquiring
            if st_model is None:
                from sentence_transformers import SentenceTransformer
                # low_cpu_mem_usage=False prevents transformers>=4.35 from placing
                # weights on the "meta" device during from_pretrained, which would
                # cause a NotImplementedError when ST then calls .to(device).
                _no_meta = {"low_cpu_mem_usage": False}
                try:
                    st_model = SentenceTransformer(
                        model_name, device=device, trust_remote_code=True,
                        model_kwargs=_no_meta,
                    )
                except TypeError:
                    # Older sentence-transformers without model_kwargs / trust_remote_code.
                    try:
                        st_model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
                    except TypeError:
                        st_model = SentenceTransformer(model_name, device=device)
                _ST_MODEL_CACHE[key] = st_model
    try:
        cur = getattr(st_model, "max_seq_length", max_length) or max_length
        st_model.max_seq_length = min(int(cur), int(max_length))
    except Exception:
        pass
    return st_model


def _get_hf_model(model_name: str, device: str) -> Tuple[Any, Any]:
    """Return (cached) (tokenizer, model) for the HF transformers fallback."""
    key = (model_name, device)
    cached = _HF_MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    with _HF_MODEL_LOCK:
        cached = _HF_MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
        # low_cpu_mem_usage=False: transformers>=4.35 defaults to lazy meta-tensor
        # loading when this flag is True, causing .to(device) to raise
        # NotImplementedError ("Cannot copy out of meta tensor").
        model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, low_cpu_mem_usage=False
        )
        model.to(device)
        model.eval()
        _HF_MODEL_CACHE[key] = (tokenizer, model)
    return _HF_MODEL_CACHE[key]


def build_embeddings(
    chunks: List[Dict],
    model_name: str = "jinaai/jina-embeddings-v2-base-code",
    *,
    backend: str = "auto",                 # "auto" | "sentence-transformers" | "transformers"
    batch_size: int = 64,
    device: Optional[str] = None,          # None -> auto: cuda if available else cpu
    normalize: bool = True,                # L2 normalize output for cosine similarity
    max_length: Optional[int] = None,      # None -> sensible default per model family
    field_strategy: str = "auto",          # "auto" | "code_only" | "code_signature_doc" | "custom"
    text_builder: Optional[Callable[[Dict], str]] = None,  # used when field_strategy="custom"
) -> np.ndarray:
    """
    Build dense vector embeddings for code chunks or corpus rows using either a 
    Sentence-Transformers model (preferred) or a plain Hugging Face Transformers model.

    Parameters
    ----------
    chunks : list of dict
        Input data; supports two formats:
          - **Corpus rows** (from prepare_embedding_corpus):
              {
                "chunk_id": str,
                "view": "intent" | "impl",
                "text": str,
                ...
              }
          - **Raw parsed chunks** (from parse_codebase):
              {
                "code": str,
                "name": str,
                "signature": str,
                "docstring": str,
                "file": str,
                ...
              }

    model_name : str, default "jinaai/jina-embeddings-v2-base-code"
        Hugging Face Hub model id or local path. 
        - Defaults to Jina’s long-context code embeddings (~8k tokens).
        - Any Sentence-Transformers or Transformers encoder model is supported.

    backend : {"auto","sentence-transformers","transformers"}, default "auto"
        How to load/encode:
          - "sentence-transformers": force SentenceTransformer.encode
          - "transformers": force AutoModel + pooling
          - "auto": try Sentence-Transformers, fall back to Transformers

    batch_size : int, default 64
        Batch size for encoding.

    device : str or None, default None
        "cuda", "cpu", or "mps". 
        None auto-selects "cuda" if available, then "mps", else "cpu".

    normalize : bool, default True
        L2-normalize embeddings row-wise for cosine similarity search.

    max_length : int or None
        Token length for truncation.
        - Defaults to 8192 for Jina v2 code models.
        - Defaults to 512 otherwise.

    field_strategy : {"auto","code_only","code_signature_doc","custom"}, default "auto"
        How to compose text per chunk:
          - "auto": if "text" is present (corpus rows), use it; else fallback to "code_only".
          - "code_only": only the 'code' field
          - "code_signature_doc": combine signature + docstring + code
          - "custom": call `text_builder(chunk)`

    text_builder : Callable or None
        Required if field_strategy="custom". Receives a chunk/corpus row, returns a string.

    Returns
    -------
    np.ndarray
        Array of shape (N, D), float32, one embedding per input row (order preserved).

    Notes
    -----
    - Pooling strategy:
        * For BGE-* models: CLS pooling (per BAAI usage guide).
        * Otherwise: mean pooling with attention mask.
        * Sentence-Transformers models: use built-in .encode.
    - Normalization is always applied consistently for cosine similarity search.

    Examples
    --------
    # 1) Default: Jina v2 code embeddings
        emb = build_embeddings(chunks)

    # 2) Corpus rows (from prepare_embedding_corpus)
        corpus = prepare_embedding_corpus(records, which=("intent",))
        emb = build_embeddings(corpus)

    # 3) BGE with CLS pooling
        emb = build_embeddings(chunks, model_name="BAAI/bge-code-v1")

    # 4) Signature + docstring + code
        emb = build_embeddings(chunks, field_strategy="code_signature_doc")

    # 5) Custom builder
        def my_builder(ch):
            return f"{ch.get('file','')}\n{ch.get('signature','')}\n{ch.get('docstring','')}\n{ch.get('code','')}"
        emb = build_embeddings(chunks, field_strategy="custom", text_builder=my_builder)
    """
    if not isinstance(chunks, list) or len(chunks) == 0:
        raise ValueError("build_embeddings: 'chunks' must be a non-empty list.")
    if field_strategy == "custom" and not callable(text_builder):
        raise ValueError("build_embeddings: text_builder must be provided when field_strategy='custom'.")

    # ----------------------------
    # 1) Compose texts per chunk
    # ----------------------------
    def compose_text(ch: Dict) -> str:
        if field_strategy == "auto":
            if "text" in ch:   # corpus row
                return str(ch.get("text", ""))
            return str(ch.get("code", "") or "")
        elif field_strategy == "code_only":
            return str(ch.get("code", "") or "")
        elif field_strategy == "code_signature_doc":
            parts: List[str] = []
            if ch.get("name"):       parts.append(f"name: {ch['name']}")
            if ch.get("file"):       parts.append(f"file: {ch['file']}")
            if ch.get("signature"):  parts.append(f"signature: {ch['signature']}")
            if ch.get("docstring"):  parts.append(f"doc: {ch['docstring']}")
            code = ch.get("code", "") or ""
            parts.append("code:\n" + code)
            return "\n".join(parts)
        else:  # custom
            return str(text_builder(ch))

    texts: List[str] = [compose_text(ch) for ch in chunks]
    mask = [bool(t.strip()) for t in texts]
    if not any(mask):
        raise ValueError("build_embeddings: none of the inputs produced non-empty text.")
    texts = [t if m else " " for t, m in zip(texts, mask)]

    # ----------------------------
    # 2) Device, max_length, helpers
    # ----------------------------
    import torch  # lazy: only needed when actually encoding
    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    def _norm(x: np.ndarray) -> np.ndarray:
        if not normalize:
            return x.astype("float32", copy=False)
        denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return (x / denom).astype("float32", copy=False)

    name_lower = model_name.lower()
    if max_length is None:
        if "jina-embeddings-v2-base-code" in name_lower:
            max_length = 8192
        else:
            max_length = 512

    def _use_cls_pooling() -> bool:
        return "baai/bge" in name_lower or "bge-" in name_lower

    # ----------------------------
    # 3) Try Sentence-Transformers
    # ----------------------------
    if backend in ("auto", "sentence-transformers"):
        try:
            st_model = _get_st_model(model_name, device, max_length)
            embs = st_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
            return _norm(embs)
        except Exception:
            if backend == "sentence-transformers":
                raise
            # else fall through

    # ----------------------------
    # 4) Transformers fallback
    # ----------------------------
    tokenizer, model = _get_hf_model(model_name, device)

    def _mean_pool(last_hidden_state: "torch.Tensor", attention_mask: "torch.Tensor") -> "torch.Tensor":
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    pooled: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            toks = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            out = model(**toks)
            if _use_cls_pooling():
                vec = out.last_hidden_state[:, 0]
            else:
                vec = _mean_pool(out.last_hidden_state, toks["attention_mask"])
            pooled.append(vec.detach().cpu().numpy())

    embs = np.vstack(pooled)
    return _norm(embs)
