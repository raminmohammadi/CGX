import math
from typing import List, Dict, Optional, Callable, Iterable
import numpy as np
import torch

def build_embeddings(
    chunks: List[Dict],
    model_name: str = "jinaai/jina-embeddings-v2-base-code",
    *,
    backend: str = "auto",                 # "auto" | "sentence-transformers" | "transformers"
    batch_size: int = 64,
    device: Optional[str] = None,          # None -> auto: cuda if available else cpu
    normalize: bool = True,                # L2 normalize output for cosine similarity
    max_length: Optional[int] = None,      # None -> sensible default per model family
    field_strategy: str = "code_only",     # "code_only" | "code_signature_doc" | "custom"
    text_builder: Optional[Callable[[Dict], str]] = None,  # used when field_strategy="custom"
) -> np.ndarray:
    """
    Build dense vector embeddings for code chunks using either a Sentence-Transformers
    model (preferred) or a plain HF Transformers model with robust pooling.

    Parameters
    ----------
    chunks : list of dict
        Each dict SHOULD include at least:
          - "code": str          (source of function/class/method)
        Optionally:
          - "name": str          (symbol name)
          - "signature": str     (function signature)
          - "docstring": str     (docstring)
          - "file": str          (path)
    model_name : str, default "jinaai/jina-embeddings-v2-base-code"
        HF Hub model id or local path.
        Good defaults for code listed below.
    backend : {"auto","sentence-transformers","transformers"}, default "auto"
        How to load/encode. "auto" tries Sentence-Transformers then falls back to Transformers.
    batch_size : int, default 64
        Encoding batch size.
    device : str or None, default None
        "cuda", "cpu" or "mps". None auto-selects "cuda" if available else "cpu".
    normalize : bool, default True
        L2-normalize embeddings row-wise (recommended for cosine similarity search).
    max_length : int or None
        Token length for truncation. If None, uses sensible defaults:
          - 8192 for Jina v2 code models
          - 512 for most others
    field_strategy : {"code_only","code_signature_doc","custom"}, default "code_only"
        How to compose the text that gets embedded:
          - "code_only": only the 'code' field
          - "code_signature_doc": signature + docstring + code (when present)
          - "custom": call `text_builder(chunk)` and embed that string
    text_builder : Callable or None
        Required when field_strategy="custom". Receives chunk dict, returns str.

    Returns
    -------
    np.ndarray
        Array of shape (N, D) float32, one embedding per chunk (in the same order).

    Raises
    ------
    ValueError
        - If `chunks` is empty or no chunk contains usable text.
        - If `field_strategy="custom"` but `text_builder` is None.

    Notes
    -----
    - Pooling strategy:
        * For BGE-* models: CLS pooling (per BAAI usage guide). :contentReference[oaicite:0]{index=0}
        * For GraphCodeBERT (base): mean pooling works; community ST variants also exist. :contentReference[oaicite:1]{index=1}
        * For Sentence-Transformers models: use `SentenceTransformer.encode`.
    - Long context:
        * Jina v2 base code supports ~8k tokens (we default to 8192 if not overridden). :contentReference[oaicite:2]{index=2}
        
    Examples:
    # 1) Code-specialized, long context (default in this function)
        emb = build_embeddings(chunks)  # uses jinaai/jina-embeddings-v2-base-code

    # 2) BGE-Code with CLS pooling (auto-detected)
        emb = build_embeddings(chunks, model_name="BAAI/bge-code-v1")

    # 3) GraphCodeBERT ST variant (pure Sentence-Transformers path)
        emb = build_embeddings(chunks, model_name="buelfhood/SOCO-C-GraphCodeBERT-ST")

    # 4) Compose richer text for retrieval: signature + docstring + code
        emb = build_embeddings(chunks, field_strategy="code_signature_doc")

    # 5) Custom text builder
        def my_builder(ch):
            return f"{ch.get('file','')}\n{ch.get('name','')}\n{ch.get('signature','')}\n{ch.get('docstring','')}\n{ch.get('code','')}"
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
        if field_strategy == "code_only":
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
    # Filter out completely empty texts but preserve alignment
    mask = [bool(t.strip()) for t in texts]
    if not any(mask):
        raise ValueError("build_embeddings: none of the chunks produced non-empty text to embed.")
    # Replace empty texts with a safe placeholder to keep indices aligned
    texts = [t if m else " " for t, m in zip(texts, mask)]

    # ----------------------------
    # 2) Device, max_length, helpers
    # ----------------------------
    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    def _norm(x: np.ndarray) -> np.ndarray:
        if not normalize:
            return x.astype("float32", copy=False)
        denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return (x / denom).astype("float32", copy=False)

    # Heuristic default max_length
    name_lower = model_name.lower()
    if max_length is None:
        if "jina-embeddings-v2-base-code" in name_lower:
            max_length = 8192  # Jina v2 code supports 8k context. :contentReference[oaicite:3]{index=3}
        else:
            max_length = 512   # safe default for most HF encoder models

    # Detect families for pooling choice
    def _use_cls_pooling() -> bool:
        # BGE family recommends CLS pooling. :contentReference[oaicite:4]{index=4}
        return "baai/bge" in name_lower or "bge-" in name_lower

    # ----------------------------
    # 3) Try Sentence-Transformers
    # ----------------------------
    if backend in ("auto", "sentence-transformers"):
        try:
            from sentence_transformers import SentenceTransformer
            st_model = SentenceTransformer(model_name, device=device)
            try:
                # Respect model’s tokenizer max if smaller than requested
                st_model.max_seq_length = min(getattr(st_model, "max_seq_length", max_length), max_length)
            except Exception:
                pass

            embs = st_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,  # we normalize ourselves for consistency
            )
            return _norm(embs)
        except Exception as _st_err:
            if backend == "sentence-transformers":
                raise
            # else fall through to plain transformers backend

    # ----------------------------
    # 4) Plain Transformers fallback
    # ----------------------------
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.to(device)
    model.eval()

    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
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
            # Choose pooling
            if _use_cls_pooling():
                # Use first token (CLS) embedding
                vec = out.last_hidden_state[:, 0]
            else:
                # Default to mean pooling with attention mask
                vec = _mean_pool(out.last_hidden_state, toks["attention_mask"])
            pooled.append(vec.detach().cpu().numpy())

    embs = np.vstack(pooled)
    return _norm(embs)
