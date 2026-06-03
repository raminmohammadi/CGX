

"""Optional cross-encoder reranker over the top-N RRF-fused chunks.

The reranker is intentionally lazy-loaded:

* ``sentence_transformers`` (and therefore ``torch``) are only imported when
  :func:`get_default_reranker` is actually called.
* A no-op (``None`` model) is returned if the import fails, so the calling
  code can transparently fall back to the RRF ordering.

This keeps the no-torch / API-only install path clean while letting users
opt in to a higher-quality rerank stage by setting
``HybridConfig.enable_reranker = True``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple


def _text_for_chunk(
    cid: str,
    rec_by_id: Dict[str, Dict[str, Any]],
    chunks: Sequence[Dict[str, Any]],
) -> str:
    """Best-effort textual representation of a chunk for the cross-encoder.

    Prefers the record's `name + docstring + code` triple; falls back to the
    raw chunk text if records are unavailable. Returns at most ~2000 chars so
    the reranker sees a compact, signature-heavy view.
    """
    rec = rec_by_id.get(cid) or {}
    name = str(rec.get("name") or "")
    sig = str(rec.get("signature") or "")
    doc = str(rec.get("docstring") or rec.get("meta", {}).get("docstring") or "")
    code = str(rec.get("code") or "")
    if not (name or sig or doc or code):
        for ch in chunks:
            if str(ch.get("id") or "") == cid:
                name = name or str(ch.get("name") or "")
                code = code or str(ch.get("code") or "")
                doc = doc or str((ch.get("meta") or {}).get("docstring") or "")
                break
    parts = [p for p in (name, sig, doc, code) if p]
    text = "\n".join(parts).strip()
    return text[:2000] if len(text) > 2000 else text


_MODEL_CACHE: Dict[str, Any] = {}


def get_default_reranker(model_name: str):
    """Return a cached cross-encoder, or ``None`` if dependencies are missing."""
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
    except Exception:
        _MODEL_CACHE[model_name] = None
        return None
    try:
        model = CrossEncoder(model_name)
    except Exception:
        _MODEL_CACHE[model_name] = None
        return None
    _MODEL_CACHE[model_name] = model
    return model


def _minmax(scores: Sequence[float]) -> List[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return [0.5 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def rerank_chunks(
    *,
    query: str,
    fused: List[Tuple[str, float]],
    rec_by_id: Dict[str, Dict[str, Any]],
    chunks: Sequence[Dict[str, Any]] = (),
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n: int = 30,
    weight: float = 1.0,
    provenance: Optional[Dict[str, Dict[str, Any]]] = None,
    model: Any = None,
) -> List[Tuple[str, float]]:
    """Rerank the top-``top_n`` fused chunks with a cross-encoder.

    The returned list keeps the original ordering for chunks past ``top_n`` so
    the rerank stage only refines the head of the candidate pool. Scores are
    a convex combination of the (min-max normalised) RRF score and the
    (min-max normalised) cross-encoder relevance score, weighted by
    ``weight``.

    Parameters
    ----------
    fused
        List of ``(chunk_id, rrf_score)`` tuples sorted by RRF score desc.
    model
        Optional pre-loaded cross-encoder. When omitted the default loader is
        consulted; if it returns ``None`` (e.g. ``sentence_transformers`` is
        not installed) the original ``fused`` order is returned unchanged.
    """
    if not fused or top_n <= 0:
        return list(fused)
    weight = max(0.0, min(1.0, float(weight)))
    head = fused[:top_n]
    tail = fused[top_n:]

    enc = model if model is not None else get_default_reranker(model_name)
    if enc is None:
        return list(fused)

    pairs = [[query, _text_for_chunk(cid, rec_by_id, chunks)] for cid, _ in head]
    try:
        raw = enc.predict(pairs)
    except Exception:
        return list(fused)

    ce_scores = [float(x) for x in list(raw)]
    rrf_scores = [float(s) for _, s in head]
    ce_n = _minmax(ce_scores)
    rrf_n = _minmax(rrf_scores)
    rescored: List[Tuple[str, float]] = []
    for (cid, _), ce, rr in zip(head, ce_n, rrf_n):
        new_score = weight * ce + (1.0 - weight) * rr
        rescored.append((cid, float(new_score)))
        if provenance is not None:
            provenance.setdefault(cid, {})["reranker_score"] = float(ce)
    rescored.sort(key=lambda kv: -kv[1])
    return rescored + tail
