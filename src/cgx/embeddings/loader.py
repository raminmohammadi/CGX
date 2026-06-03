

"""Shared embedder loaders.

Two entry points used by the CLI surfaces:

* :func:`load_embedder_from_spec` — resolve a BYO embedder from a
  ``"module:attr"`` import spec (advanced users wiring in their own
  encoder). Honours the ``CGX_EMBEDDER_ALLOWLIST`` env var.
* :func:`load_embedder_from_model` — build a thin ``.encode``-style
  wrapper from a Hugging Face / Sentence-Transformers model name,
  reusing the cached model loaders in :mod:`cgx.embeddings.build` so
  repeated calls in the same process do not re-download or re-load
  the weights.

Heavy ML dependencies (``torch``, ``transformers``,
``sentence_transformers``) are imported lazily inside the functions so
that simply importing this module does not require the full ML stack.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import os
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


def load_embedder_from_spec(spec: str) -> Any:
    """Load an embedder object/factory from a ``"module:attr"`` spec.

    If the resolved attribute is a class it is instantiated with no
    args; if it is callable but does not expose ``.encode`` it is
    treated as a factory and called; otherwise the attribute itself is
    returned. The returned object must expose
    ``.encode(list[str]) -> ndarray``.

    Security note
    -------------
    This performs ``importlib.import_module(<user-supplied>)``, which
    runs that module's top-level code. Only pass module names you
    trust. To restrict to a short list, set
    ``CGX_EMBEDDER_ALLOWLIST=mod1,mod2`` in the environment.
    """
    if not spec or ":" not in spec:
        raise ValueError('Embedder spec must be "module:attr" (got %r)' % spec)
    mod_name, attr = spec.split(":", 1)
    allow = [
        s.strip()
        for s in (os.environ.get("CGX_EMBEDDER_ALLOWLIST", "") or "").split(",")
        if s.strip()
    ]
    if allow and mod_name not in allow:
        raise PermissionError(
            f"Embedder module {mod_name!r} not in CGX_EMBEDDER_ALLOWLIST={allow!r}"
        )
    mod = importlib.import_module(mod_name)
    obj = getattr(mod, attr)
    if inspect.isclass(obj):
        return obj()
    if callable(obj) and not hasattr(obj, "encode"):
        # factory
        return obj()
    return obj


def load_embedder_from_model(
    model_name: str,
    *,
    device: Optional[str] = None,
) -> Any:
    """Return a thin encoder wrapper for ``model_name``.

    Tries Sentence-Transformers first (preferred for code embeddings),
    falling back to a plain Hugging Face ``AutoModel`` with mean
    pooling. Both branches go through the cached loaders in
    :mod:`cgx.embeddings.build`, so concurrent callers share weights
    instead of re-loading the model.

    The returned object exposes ``.encode(list[str]) -> np.ndarray``
    of dtype ``float32``. Vectors are **not** normalised here — that
    is the caller's responsibility (the index pipelines apply L2
    normalisation downstream).
    """
    # Resolve device once so both branches see the same choice.
    if device is None:
        try:
            import torch  # lazy
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        except Exception:
            device = "cpu"

    # Default max sequence length mirrors build_embeddings: 8k for the
    # Jina v2 code family, 512 otherwise.
    max_length = 8192 if "jina-embeddings-v2-base-code" in model_name.lower() else 512

    # ---- Sentence-Transformers branch ------------------------------------
    try:
        from cgx.embeddings.build import _get_st_model

        st_model = _get_st_model(model_name, device, max_length)

        class _ST:
            def encode(self, texts: List[str]):
                import numpy as np
                vecs = st_model.encode(
                    texts,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                )
                return vecs.astype("float32", copy=False)

        logger.info("Embedder: sentence-transformers (%s)", model_name)
        return _ST()
    except Exception:
        pass

    # ---- Transformers fallback -------------------------------------------
    from cgx.embeddings.build import _get_hf_model

    tokenizer, model = _get_hf_model(model_name, device)

    def _encode(texts: List[str]):
        import numpy as np
        import torch

        with torch.no_grad():
            toks = tokenizer(
                texts, padding=True, truncation=True, max_length=max_length,
                return_tensors="pt",
            ).to(device)
            out = model(**toks)
            hidden = out.last_hidden_state
            mask = toks["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            return (summed / counts).cpu().numpy().astype("float32", copy=False)

    class _HF:
        def encode(self, texts: List[str]):
            return _encode(texts)

    logger.info("Embedder: transformers (%s)", model_name)
    return _HF()
