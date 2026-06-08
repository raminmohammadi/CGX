# src/cgx/retrieval/index.py
from __future__ import annotations

"""
Two-view retrieval index (intent + implementation) for code chunks.

This module builds and hosts *separate* ANN indices for the two deterministic
text views you already produce in S4:
  - "intent" (NL-friendly card)
  - "impl"   (code-ish body/signatures)

Design goals:
- Add-only and non-invasive: no changes to parsing/graph/records code.
- No model assumptions: you pass an embedder (with .encode) and an index builder.
- Deterministic and auditable: explicit mappings from index rows -> chunk ids.
- Ready for later hybrid fusion: exposes per-view searches, then you can RRF-fuse.

Primary entrypoint:
- TwoViewIndex.from_records(records, embedder, index_builder, ...)

Where:
- `records`       = output of S4 `make_index_records(...)`
- `embedder`      = object with `.encode(list[str]) -> np.ndarray`
- `index_builder` = callable(embeddings: np.ndarray, **kw) -> (faiss_index, meta_dict)
                    You can pass your `build_faiss_index` from earlier.

This file does NOT import faiss directly; it relies on the provided builder.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence
import logging
import numpy as np

# ✅ fixed: no silent fallback, always import directly
from cgx.embeddings.records import prepare_embedding_corpus

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class ViewSlice:
    """
    Immutable view of a single index:
      - rows:    corpus rows (dicts) with {"chunk_id","view","text","tokens_estimate",...}
      - ids:     np.ndarray[int64], FAISS ids aligned to rows (0..len(rows)-1)
      - index:   the ANN index returned by your builder
      - meta:    meta dict returned by your builder (metric, normalized flags, etc.)
      - dim:     embedding dimension (if provided by meta) -- optional
    """
    rows: List[Dict[str, Any]]
    ids: np.ndarray
    index: Any
    meta: Dict[str, Any]
    dim: Optional[int] = None



class TwoViewIndex:
    """
    Host for two independent indices:
      - intent (NL-friendly view)
      - impl   (code-ish view)

    Usage:
        tvi = TwoViewIndex.from_records(
            records,
            embedder=my_embedder,                         # must provide .encode(list[str]) -> ndarray
            index_builder=lambda X, **kw: build_faiss_index(X, **kw),
            metric="cosine",
            index_type="flat",
            batch=64,
        )

        intent_hits = tvi.search_view("intent", "improve database connectivity", top_k=10)
        impl_hits   = tvi.search_view("impl", "requests.get('https://')", top_k=10)

    Each search returns a list of:
        {"chunk_id": str, "row": dict, "score": float, "distance": float, "rank": int}
    """

    def __init__(self) -> None:
        self._intent: Optional[ViewSlice] = None
        self._impl: Optional[ViewSlice] = None
        self._chunk_id_to_rows: Dict[str, Dict[str, List[int]]] = {}  # chunk_id -> {"intent":[row_idx...], "impl":[row_idx...]}

    # ---------- construction ----------

    @classmethod
    def from_records(
        cls,
        records: List[Dict[str, Any]],
        *,
        embedder: Any,
        index_builder: Any,
        which: Sequence[str] = ("intent", "impl"),
        metric: str = "cosine",
        index_type: str = "flat",
        normalize: Optional[bool] = None,
        batch: int = 64,
        use_gpu: bool = False,
        builder_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "TwoViewIndex":
        """
        Build indices from S4 records.
        """
        if prepare_embedding_corpus is None or not callable(prepare_embedding_corpus):
            raise RuntimeError("TwoViewIndex: prepare_embedding_corpus is not available. Check imports.")
        if not hasattr(embedder, "encode"):
            raise TypeError("TwoViewIndex: 'embedder' must implement .encode(list[str])->ndarray.")
        if not callable(index_builder):
            raise TypeError("TwoViewIndex: 'index_builder' must be callable(X, **kwargs)->(index, meta).")

        builder_kwargs = dict(builder_kwargs or {})

        # Flatten records into corpus rows
        corpus = prepare_embedding_corpus(records, which=which)
        if corpus is None or not isinstance(corpus, list):
            raise RuntimeError(f"TwoViewIndex: prepare_embedding_corpus returned invalid value: {type(corpus)}")

        # Split per view
        intent_rows = [r for r in corpus if r.get("view") == "intent"]
        impl_rows   = [r for r in corpus if r.get("view") == "impl"]

        # Encode helper
        def _encode(rows: List[Dict[str, Any]]) -> np.ndarray:
            texts = [str(r.get("text") or "") for r in rows]
            embs = embedder.encode(texts)
            X = np.asarray(embs, dtype=np.float32)
            if X.ndim != 2 or X.shape[0] != len(rows):
                raise ValueError(f"TwoViewIndex: embedder returned shape {X.shape}, expected (N,D) with N={len(rows)}.")
            return X

        tvi = cls()

        if intent_rows:
            X_intent = _encode(intent_rows)
            ids_intent = np.arange(len(intent_rows), dtype=np.int64)
            idx_i, meta_i = index_builder(
                X_intent,
                metric=metric,
                index=index_type,
                ids=ids_intent,
                normalize=normalize,
                use_gpu=use_gpu,
                **builder_kwargs
            )
            tvi._intent = ViewSlice(rows=intent_rows, ids=ids_intent, index=idx_i, meta=meta_i or {}, dim=int(X_intent.shape[1]))

        if impl_rows:
            X_impl = _encode(impl_rows)
            ids_impl = np.arange(len(impl_rows), dtype=np.int64)
            idx_c, meta_c = index_builder(
                X_impl,
                metric=metric,
                index=index_type,
                ids=ids_impl,
                normalize=normalize,
                use_gpu=use_gpu,
                **builder_kwargs
            )
            tvi._impl = ViewSlice(rows=impl_rows, ids=ids_impl, index=idx_c, meta=meta_c or {}, dim=int(X_impl.shape[1]))

        # Build chunk->rows map
        tvi._chunk_id_to_rows = {}
        for view_name, vs in (("intent", tvi._intent), ("impl", tvi._impl)):
            if vs is None:
                continue
            for ridx, row in enumerate(vs.rows):
                cid = row.get("chunk_id")
                if not isinstance(cid, str):
                    continue
                tvi._chunk_id_to_rows.setdefault(cid, {}).setdefault(view_name, []).append(ridx)

        logger.info(
            "TwoViewIndex built: intent_rows=%d, impl_rows=%d",
            len(intent_rows), len(impl_rows)
        )
        return tvi

    # ---------- querying ----------

    def _ensure_view(self, view: str) -> ViewSlice:
        if view not in {"intent", "impl"}:
            raise ValueError("view must be one of {'intent','impl'}.")
        vs = self._intent if view == "intent" else self._impl
        if vs is None:
            raise RuntimeError(f"TwoViewIndex: '{view}' view is not available.")
        return vs

    def encode_query(self, embedder: Any, text: str, *, l2_normalize: Optional[bool] = None, metric: Optional[str] = None) -> np.ndarray:
        if not hasattr(embedder, "encode"):
            raise TypeError("encode_query: 'embedder' must implement .encode([text])->ndarray.")
        q = embedder.encode([str(text or "")])
        Q = np.asarray(q, dtype=np.float32)
        if Q.ndim != 2 or Q.shape[0] != 1:
            raise ValueError(f"encode_query: embedder returned shape {Q.shape}, expected (1,D).")
        m = (metric or "").lower()
        if l2_normalize or m in {"cosine", "ip", "inner_product", "dot"}:
            denom = np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12
            Q = Q / denom
        return Q

    def search_view(
        self,
        view: str,
        query: str,
        *,
        embedder: Any,
        top_k: int = 10,
        metric: Optional[str] = None,
        normalize_query: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        vs = self._ensure_view(view)
        m = (metric or vs.meta.get("metric") or "cosine").lower()
        if m in {"inner_product", "dot"}:
            m = "ip"

        Q = self.encode_query(embedder, query, l2_normalize=(normalize_query if normalize_query is not None else (m in {"cosine", "ip"})), metric=m)

        try:
            D, I = vs.index.search(Q, int(top_k))
        except Exception as e:
            raise RuntimeError(f"search_view: index.search failed for view='{view}': {e}") from e

        distances = D[0]
        indices   = I[0]

        out: List[Dict[str, Any]] = []
        rank = 1
        for dist, idx in zip(distances, indices):
            if int(idx) < 0:
                continue
            try:
                row = vs.rows[int(idx)]
            except Exception:
                rank += 1
                continue
            if m in {"cosine", "ip"}:
                score = float(dist)
                distance = float(1.0 - dist)
            else:
                distance = float(dist)
                score = float(-distance)
            out.append({
                "chunk_id": row.get("chunk_id"),
                "row": row,
                "distance": distance,
                "score": score,
                "rank": rank,
            })
            rank += 1

        return out

    # ---------- utilities ----------

    def rows_for_chunk(self, chunk_id: str) -> Dict[str, List[int]]:
        return self._chunk_id_to_rows.get(chunk_id, {"intent": [], "impl": []})

    def available_views(self) -> List[str]:
        return [v for v, vs in (("intent", self._intent), ("impl", self._impl)) if vs is not None]