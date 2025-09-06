# # src/cgx/retrieval/orchestrator.py
# from __future__ import annotations

# """
# Two-view retrieval orchestrator (ADD-ONLY).

# Purpose
# -------
# Provide deterministic, auditable retrieval that:
#   - Builds one FAISS index per VIEW ("intent", "impl") from S4 records/corpus.
#   - Runs semantic search on both views, optional lexical search on chunks,
#     and graph expansion, then fuses with RRF.
#   - Aggregates to files/classes and suggests insertion points for new code
#     using deterministic overlap signals (imports/attributes/signatures).

# This module DOES NOT:
#   - Assume any specific embedding model.
#   - Modify existing parse/graph/embedding code.
#   - Change return shapes elsewhere.

# External contracts
# ------------------
# Inputs are the S4 outputs and your graph:
#   - `records`: from `cgx.embeddings.records.make_index_records(...)`
#   - `corpus` : from `cgx.embeddings.records.prepare_embedding_corpus(...)`
#   - `chunks` : original S1/S2 parsed chunks (if you want lexical over chunks)
#   - `G`      : S1 graph

# Exports (public API)
# --------------------
# __all__ = [
#   "build_two_view_indices",           # create FAISS indices per view
#   "semantic_retrieve_two_view",       # run semantic on both views
#   "hybrid_retrieve_two_view",         # semantic (both) + lexical + graph + RRF
#   "aggregate_by_file",                # aggregate fused hits to files
#   "aggregate_by_class",               # aggregate fused hits to classes
#   "suggest_insertion_points",         # where new code should go (deterministic heuristics)
# ]

# Each function has strong docstrings, logging, and defensive error handling.
# """

# import math
# import re
# from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# import numpy as np

# # ---------------------------
# # Optional helper imports (prefer project helpers, fall back to internal impls)
# # ---------------------------
# try:
#     from src.cgx.embeddings.build import build_embeddings as _build_embeddings  # type: ignore
# except Exception:  # pragma: no cover
#     _build_embeddings = None  # type: ignore

# try:
#     from src.cgx.embeddings.index import build_faiss_index as _build_faiss_index  # type: ignore
# except Exception:  # pragma: no cover
#     _build_faiss_index = None  # type: ignore

# try:
#     from src.cgx.embeddings.search import semantic_search as _semantic_search  # type: ignore
# except Exception:  # pragma: no cover
#     _semantic_search = None  # type: ignore

# try:
#     from src.cgx.embeddings.views import attach_views_to_chunks as _attach_views_to_chunks  # type: ignore
# except Exception:  # pragma: no cover
#     _attach_views_to_chunks = None  # type: ignore


# try:
#     import faiss  # type: ignore
#     _FAISS_AVAILABLE = True
# except Exception:  # pragma: no cover
#     faiss = None  # type: ignore
#     _FAISS_AVAILABLE = False

# # local imports are optional; this module can function without them
# try:
#     from src.cgx.retrieval.hybrid import lexical_search as _lexical_search_default  # optional
# except Exception:  # pragma: no cover
#     _lexical_search_default = None  # type: ignore

# from cgx.logging_setup import get_logger
# logger = get_logger("orchestration")


# # ---------------------------
# # Utilities
# # ---------------------------

# def _norm_rows(x: np.ndarray) -> np.ndarray:
#     """
#     L2-normalize rows. Safe for zero vectors.
#     """
#     x = np.asarray(x, dtype=np.float32)
#     denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
#     return (x / denom).astype("float32", copy=False)


# def encode_texts(
#     texts: Sequence[str],
#     embedder: Any,
#     *,
#     normalize: bool = True,
# ) -> np.ndarray:
#     """
#     Encode a list of strings with a user-provided embedder.

#     The `embedder` must either:
#       - expose `.encode(list[str]) -> np.ndarray`, OR
#       - be a callable(list[str]) -> np.ndarray

#     No model is assumed here.

#     Returns
#     -------
#     np.ndarray of shape (N, D), dtype float32
#     """
#     if not isinstance(texts, (list, tuple)):
#         raise TypeError(f"encode_texts: 'texts' must be a list/tuple, got {type(texts)}")
#     if len(texts) == 0:
#         raise ValueError("encode_texts: 'texts' must be non-empty.")

#     if hasattr(embedder, "encode"):
#         vecs = embedder.encode(list(texts))  # type: ignore[attr-defined]
#     elif callable(embedder):
#         vecs = embedder(list(texts))
#     else:
#         raise TypeError("encode_texts: 'embedder' must provide .encode(...) or be callable(list[str])->ndarray")

#     arr = np.asarray(vecs, dtype=np.float32)
#     if arr.ndim != 2 or arr.shape[0] != len(texts):
#         raise ValueError(f"encode_texts: encoder returned shape {arr.shape}, expected (N,D) with N={len(texts)}")

#     return _norm_rows(arr) if normalize else arr


# def _faiss_build_index(
#     embeddings: np.ndarray,
#     *,
#     metric: str = "cosine",
#     ids: Optional[np.ndarray] = None,
#     index_type: str = "flat",
#     nlist: int = 1024,
#     nprobe: int = 16,
#     M: int = 32,
#     efConstruction: int = 200,
#     efSearch: int = 64,
# ) -> Tuple[Any, Dict[str, Any]]:
#     """
#     Minimal FAISS index builder (internal), to avoid cross-module assumptions.
#     If FAISS is not available, raises a RuntimeError with a clear message.
#     """
#     if not _FAISS_AVAILABLE:
#         raise RuntimeError("FAISS is not installed. Install 'faiss-cpu' or 'faiss-gpu' to use the orchestrator indices.")

#     X = np.asarray(embeddings, dtype=np.float32)
#     if X.ndim != 2:
#         raise ValueError("embeddings must be 2D (N,D).")
#     N, D = X.shape
#     if N == 0 or D == 0:
#         raise ValueError("embeddings must be non-empty.")

#     if ids is None:
#         ids = np.arange(N, dtype=np.int64)
#     else:
#         ids = np.asarray(ids, dtype=np.int64)
#         if ids.shape != (N,):
#             raise ValueError(f"'ids' must have shape (N,), got {ids.shape} for N={N}")

#     m = metric.lower()
#     if m not in {"cosine", "l2", "ip"}:
#         raise ValueError("metric must be one of {'cosine','l2','ip'}")

#     use_ip = (m in {"cosine", "ip"})
#     index_flat = faiss.IndexFlatIP(D) if use_ip else faiss.IndexFlatL2(D)

#     if index_type == "flat":
#         base = index_flat
#     elif index_type == "ivf":
#         nlist_eff = int(nlist)
#         if N < nlist_eff:
#             nlist_eff = max(1, int(max(1, round(math.sqrt(N)))))
#         base = faiss.IndexIVFFlat(index_flat, D, nlist_eff, faiss.METRIC_INNER_PRODUCT if use_ip else faiss.METRIC_L2)
#         if not base.is_trained:
#             base.train(X)
#         base.nprobe = int(nprobe)
#     elif index_type == "hnsw":
#         try:
#             base = faiss.IndexHNSWFlat(D, int(M), faiss.METRIC_INNER_PRODUCT if use_ip else faiss.METRIC_L2)
#         except TypeError:
#             base = faiss.IndexHNSWFlat(D, int(M))
#         base.hnsw.efConstruction = int(efConstruction)
#         base.hnsw.efSearch = int(efSearch)
#     else:
#         raise ValueError("index_type must be 'flat', 'ivf' or 'hnsw'.")

#     try:
#         idmap = faiss.IndexIDMap2(base)
#     except Exception:
#         idmap = faiss.IndexIDMap(base)
#     idmap.add_with_ids(X, ids)

#     meta = {
#         "metric": m,
#         "index_type": index_type,
#         "dim": int(D),
#         "num_vectors": int(N),
#         "nlist": int(getattr(base, "nlist", 0)) if hasattr(base, "nlist") else None,
#         "nprobe": int(getattr(base, "nprobe", 0)) if hasattr(base, "nprobe") else None,
#         "M": int(M) if index_type == "hnsw" else None,
#         "efConstruction": int(efConstruction) if index_type == "hnsw" else None,
#         "efSearch": int(efSearch) if index_type == "hnsw" else None,
#         "normalized_for_cosine": (m == "cosine"),
#     }
#     return idmap, meta


# def _semantic_search_faiss(
#     query: str,
#     embedder: Any,
#     index: Any,
#     *,
#     metric: str = "cosine",
#     top_k: int = 10,
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Encode a single query and run FAISS search. Returns (distances, ids).
#     Handles cosine/IP vs L2 distance semantics consistently.
#     """
#     q = encode_texts([query], embedder, normalize=(metric in {"cosine", "ip"}))
#     D, I = index.search(q, int(top_k))
#     return D[0], I[0]


# # ---------------------------
# # Helper adapters to prefer project helpers with graceful fallback
# # ---------------------------

# def _use_helper_build_embeddings(
#     texts: Sequence[str],
#     embedder: Any,
#     *,
#     normalize: bool,
# ) -> np.ndarray:
#     """
#     Prefer cgx.embeddings.build.build_embeddings; fall back to encode_texts.
#     """
#     if _build_embeddings is not None:
#         try:
#             return np.asarray(
#                 _build_embeddings(texts, embedder=embedder, normalize=normalize),
#                 dtype=np.float32,
#             )
#         except TypeError:
#             # try (texts, embedder)
#             try:
#                 arr = _build_embeddings(texts, embedder)  # type: ignore[misc]
#                 arr = np.asarray(arr, dtype=np.float32)
#                 return _norm_rows(arr) if normalize else arr
#             except Exception:
#                 logger.debug("build_embeddings signature mismatch; falling back to encode_texts.")
#         except Exception as e:
#             logger.debug("build_embeddings failed (%s); falling back to encode_texts.", e)
#     # fallback
#     return encode_texts(texts, embedder, normalize=normalize)


# def _use_helper_build_faiss_index(
#     embs: np.ndarray,
#     *,
#     metric: str,
#     ids: np.ndarray,
#     index_type: str,
#     nlist: int,
#     nprobe: int,
#     M: int,
#     efConstruction: int,
#     efSearch: int,
# ) -> Tuple[Any, Dict[str, Any]]:
#     """
#     Prefer cgx.embeddings.index.build_faiss_index; fall back to internal _faiss_build_index.
#     """
#     if _build_faiss_index is not None:
#         try:
#             return _build_faiss_index(
#                 embs,
#                 metric=metric,
#                 ids=ids,
#                 index_type=index_type,
#                 nlist=nlist,
#                 nprobe=nprobe,
#                 M=M,
#                 efConstruction=efConstruction,
#                 efSearch=efSearch,
#             )
#         except TypeError:
#             # Try minimal positional: (embs, metric, ids, index_type)
#             try:
#                 return _build_faiss_index(embs, metric, ids, index_type)  # type: ignore[misc]
#             except Exception as e:
#                 logger.debug("build_faiss_index signature mismatch; fallback to internal. (%s)", e)
#         except Exception as e:
#             logger.debug("build_faiss_index failed; fallback to internal. (%s)", e)
#     return _faiss_build_index(
#         embs,
#         metric=metric,
#         ids=ids,
#         index_type=index_type,
#         nlist=nlist,
#         nprobe=nprobe,
#         M=M,
#         efConstruction=efConstruction,
#         efSearch=efSearch,
#     )


# def _use_helper_semantic_search(
#     query: str,
#     *,
#     embedder: Any,
#     index: Any,
#     rows: List[Dict[str, Any]],
#     metric: str,
#     top_k: int,
# ) -> List[Tuple[str, float]]:
#     """
#     Prefer cgx.embeddings.search.semantic_search if available; otherwise run FAISS search here.
#     Expected return: list[(chunk_id, score)] ranked high→low.
#     """
#     if _semantic_search is not None:
#         try:
#             # Try most explicit signature
#             lst = _semantic_search(
#                 query=query,
#                 embedder=embedder,
#                 index=index,
#                 rows=rows,
#                 metric=metric,
#                 top_k=top_k,
#             )
#             if isinstance(lst, list):
#                 return [(str(cid), float(sc)) for cid, sc in lst]
#         except TypeError:
#             # Try looser signature
#             try:
#                 lst = _semantic_search(query, embedder, index, rows, top_k)  # type: ignore[misc]
#                 if isinstance(lst, list):
#                     return [(str(cid), float(sc)) for cid, sc in lst]
#             except Exception as e2:
#                 logger.debug("semantic_search signature mismatch; fallback to internal. (%s)", e2)
#         except Exception as e:
#             logger.debug("semantic_search failed; fallback to internal. (%s)", e)
#     # Fallback: local FAISS search
#     D, I = _semantic_search_faiss(query, embedder, index, metric=metric, top_k=top_k)
#     return _ranklist_from_dist_ids(D, I, rows, metric=metric)


# def _maybe_attach_views(intent_list: List[Tuple[str, float]], impl_list: List[Tuple[str, float]]) -> None:
#     """
#     Best-effort call to cgx.embeddings.views.attach_views_to_chunks for enrichment.
#     Safe no-op if helper is missing or signature doesn't match.
#     """
#     if _attach_views_to_chunks is None:
#         return
#     try:
#         # Most likely signature: mapping by view
#         _attach_views_to_chunks({"intent": intent_list, "impl": impl_list})  # type: ignore[misc]
#     except TypeError:
#         try:
#             # Alternate: pass two lists
#             _attach_views_to_chunks(intent_list, impl_list)  # type: ignore[misc]
#         except Exception:
#             pass
#     except Exception:
#         # Never break retrieval on enrichment failure
#         pass


# # ---------------------------
# # Public API: indices per view
# # ---------------------------

# __all__ = [
#     "build_two_view_indices",
#     "semantic_retrieve_two_view",
#     "hybrid_retrieve_two_view",
#     "aggregate_by_file",
#     "aggregate_by_class",
#     "suggest_insertion_points",
# ]

# def build_two_view_indices(
#     corpus: List[Dict[str, Any]],
#     *,
#     embedder: Any,
#     metric: str = "cosine",
#     index_type: str = "flat",
#     # advanced FAISS knobs (optional)
#     nlist: int = 1024,
#     nprobe: int = 16,
#     M: int = 32,
#     efConstruction: int = 200,
#     efSearch: int = 64,
# ) -> Dict[str, Any]:
#     """
#     Build one FAISS index per view ("intent", "impl") from the flattened S4 corpus.

#     Parameters
#     ----------
#     corpus : list of rows from prepare_embedding_corpus
#         Each row has {chunk_id, view, text, tokens_estimate, type, name, file}.
#         There must be exactly one row per (chunk_id, view).
#     embedder : object or callable
#         Encoder with .encode(list[str]) or callable(list[str])->ndarray.
#         No model assumption is made here.
#     metric : "cosine" | "l2" | "ip"
#         Retrieval metric for FAISS. For "cosine" we normalize encodings.
#     index_type : "flat" | "ivf" | "hnsw"
#         FAISS index type.

#     Returns
#     -------
#     dict
#         {
#           "views": {
#             "intent": {"index": faiss.Index, "meta": {...}, "rows": [corpus rows for intent], "ids": np.ndarray[int64]},
#             "impl":   {"index": faiss.Index, "meta": {...}, "rows": [corpus rows for impl], "ids": np.ndarray[int64]},
#           },
#           "metric": metric,
#         }
#     """
#     if not isinstance(corpus, list) or not corpus:
#         raise ValueError("build_two_view_indices: 'corpus' must be a non-empty list.")

#     per_view: Dict[str, List[Dict[str, Any]]] = {"intent": [], "impl": []}
#     for row in corpus:
#         vw = row.get("view")
#         if vw in per_view:
#             per_view[vw].append(row)

#     result: Dict[str, Any] = {"views": {}, "metric": metric}

#     for view_name, rows in per_view.items():
#         if not rows:
#             logger.warning("No rows for view '%s'; skipping index build.", view_name)
#             result["views"][view_name] = {"index": None, "meta": None, "rows": [], "ids": np.array([], dtype=np.int64)}
#             continue

#         texts = [str(r.get("text", "")) for r in rows]
#         normalize = (metric.lower() in {"cosine", "ip"})
#         # Prefer project helper for embeddings
#         embs = _use_helper_build_embeddings(texts, embedder, normalize=normalize)

#         ids = np.arange(len(rows), dtype=np.int64)
#         # Prefer project helper for FAISS index
#         index, meta = _use_helper_build_faiss_index(
#             embs,
#             metric=metric,
#             ids=ids,
#             index_type=index_type,
#             nlist=nlist,
#             nprobe=nprobe,
#             M=M,
#             efConstruction=efConstruction,
#             efSearch=efSearch,
#         )

#         result["views"][view_name] = {"index": index, "meta": meta, "rows": rows, "ids": ids}

#     return result


# # ---------------------------
# # Semantic on both views + RRF
# # ---------------------------

# def _ranklist_from_dist_ids(
#     distances: np.ndarray,
#     ids: np.ndarray,
#     rows: List[Dict[str, Any]],
#     *,
#     metric: str,
# ) -> List[Tuple[str, float]]:
#     """
#     Convert FAISS outputs into [(chunk_id, score)] with score high=good.
#     For cosine/IP: score=d (inner product). For L2: score=-distance.
#     """
#     out: List[Tuple[str, float]] = []
#     for d, i in zip(distances, ids):
#         if int(i) < 0:
#             continue
#         row = rows[int(i)]
#         cid = row.get("chunk_id")
#         if cid is None:
#             continue
#         if metric in {"cosine", "ip"}:
#             score = float(d)
#         else:
#             score = float(-d)
#         out.append((str(cid), score))
#     return out


# def _rrf_fuse(
#     lists: Sequence[List[Tuple[str, float]]],
#     *,
#     k: float = 60.0,
# ) -> Dict[str, float]:
#     """
#     Reciprocal Rank Fusion (deterministic). Input: list of ranked [(id, score)].
#     Returns: dict id -> fused_score (higher is better).
#     """
#     fused: Dict[str, float] = {}
#     for lst in lists:
#         for rank, (cid, _s) in enumerate(lst, start=1):
#             fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
#     return fused


# def semantic_retrieve_two_view(
#     query: str,
#     indices: Dict[str, Any],
#     *,
#     top_k_per_view: int = 10,
# ) -> Dict[str, List[Tuple[str, float]]]:
#     """
#     Run semantic search on both views independently.

#     NOTE: This lightweight wrapper is kept for API completeness but
#     production paths go through `hybrid_retrieve_two_view` which has
#     embedder access and supports lexical/graph fusion.
#     """
#     metric = indices.get("metric", "cosine")
#     out: Dict[str, List[Tuple[str, float]]] = {"intent": [], "impl": []}
#     for view_name in ("intent", "impl"):
#         view = indices.get("views", {}).get(view_name) or {}
#         index = view.get("index")
#         rows = view.get("rows") or []
#         if index is None or not rows:
#             continue
#         # Can't search without an embedder here; raise a helpful error.
#         raise RuntimeError(
#             "semantic_retrieve_two_view requires an embedder; "
#             "use hybrid_retrieve_two_view(query, indices=..., embedder=...)."
#         )
#     return out


# def hybrid_retrieve_two_view(
#     query: str,
#     *,
#     indices: Dict[str, Any],
#     embedder: Any,
#     chunks: Optional[List[Dict[str, Any]]] = None,
#     G: Any = None,
#     # semantic
#     top_k_per_view: int = 10,
#     # lexical (over chunks)
#     use_lexical: bool = True,
#     lexical_search_fn: Optional[Callable[..., List[Dict[str, Any]]]] = None,
#     lex_fields: Iterable[str] = ("code", "name", "id", "file", "meta.docstring"),
#     lex_regex: bool = True,
#     lex_case_sensitive: bool = False,
#     lex_whole_word: bool = False,
#     # graph expansion
#     neighbor_depth: int = 1,
#     relation_types: Optional[Iterable[str]] = ("calls", "defines"),
#     internal_only: Optional[bool] = None,  # True=only internal edges; False=only external; None=both
#     # fusion
#     rrf_k: float = 60.0,
# ) -> List[Dict[str, Any]]:
#     """
#     Full hybrid retrieval:
#       - semantic on "intent" and "impl" views (two independent indices)
#       - optional lexical over chunks
#       - optional 1-hop graph expansion seeded from semantic+lexical hits
#       - RRF fusion over (intent, impl, lexical, graph-proximity)

#     Returns
#     -------
#     list[dict]
#         Sorted by fused score desc. Each item:
#           {
#             "chunk_id": str,
#             "score": float,          # fused
#             "provenance": {
#                 "intent_rank": int|None,
#                 "intent_score": float|None,
#                 "impl_rank": int|None,
#                 "impl_score": float|None,
#                 "lexical_count": int|None,
#                 "graph_depth": int|None,
#             }
#           }
#     """
#     if not isinstance(query, str) or not query.strip():
#         raise ValueError("hybrid_retrieve_two_view: 'query' must be a non-empty string.")

#     metric = indices.get("metric", "cosine")
#     views = indices.get("views", {})
#     intent_v = views.get("intent") or {}
#     impl_v   = views.get("impl") or {}

#     # ---------- 1) Semantic per view ----------
#     sem_lists: Dict[str, List[Tuple[str, float]]] = {"intent": [], "impl": []}
#     ranks: Dict[str, Dict[str, int]] = {"intent": {}, "impl": {}}
#     scores: Dict[str, Dict[str, float]] = {"intent": {}, "impl": {}}

#     for view_name, view in (("intent", intent_v), ("impl", impl_v)):
#         index = view.get("index")
#         rows  = view.get("rows") or []
#         if index is None or not rows:
#             continue

#         lst = _use_helper_semantic_search(
#             query,
#             embedder=embedder,
#             index=index,
#             rows=rows,
#             metric=metric,
#             top_k=top_k_per_view,
#         )
#         sem_lists[view_name] = lst
#         for r, (cid, sc) in enumerate(lst, start=1):
#             ranks[view_name][cid] = r
#             scores[view_name][cid] = sc

#     # Optional enrichment hook; does not affect ranking.
#     _maybe_attach_views(sem_lists.get("intent", []), sem_lists.get("impl", []))

#     # ---------- 2) Lexical over chunks (optional) ----------
#     lex_counts: Dict[str, int] = {}
#     if use_lexical and chunks:
#         search_fn = lexical_search_fn or _lexical_search_default
#         if search_fn is None:
#             # fallback minimal literal/regex search if cgx.retrieval.hybrid.lexical_search is unavailable
#             def _fallback_lex(query_text, chunks, **kwargs):
#                 rx = re.compile(query_text if kwargs.get("regex", True) else re.escape(query_text),
#                                 0 if kwargs.get("case_sensitive", False) else re.IGNORECASE)
#                 out = []
#                 for ch in chunks:
#                     total = 0
#                     for f in kwargs.get("fields", ("code","name","id","file")):
#                         cur = ch
#                         for p in f.split("."):
#                             cur = cur.get(p, "") if isinstance(cur, dict) else ""
#                         s = str(cur or "")
#                         total += len(list(rx.finditer(s)))
#                     if total:
#                         out.append({"chunk": ch, "total_matches": total})
#                 out.sort(key=lambda r: -r["total_matches"])
#                 return out
#             search_fn = _fallback_lex

#         try:
#             lex = search_fn(
#                 query, chunks,
#                 fields=tuple(lex_fields),
#                 regex=bool(lex_regex),
#                 case_sensitive=bool(lex_case_sensitive),
#                 whole_word=bool(lex_whole_word),
#             )
#             for r in lex:
#                 cid = r.get("chunk", {}).get("id")
#                 if cid:
#                     lex_counts[str(cid)] = int(r.get("total_matches", 0))
#         except Exception as e:
#             logger.warning("Lexical search failed; continuing without lexical. (%s)", e)

#     # ---------- 3) Graph expansion (optional) ----------
#     # BFS around seeds; record minimal depth for each discovered chunk id.
#     graph_depths: Dict[str, int] = {}
#     if G is not None:
#         from collections import deque
#         seeds = set()
#         seeds.update([cid for cid, _ in sem_lists.get("intent", [])])
#         seeds.update([cid for cid, _ in sem_lists.get("impl", [])])
#         seeds.update(lex_counts.keys())

#         # Work on a DiGraph view if a MultiDiGraph was used elsewhere
#         try:
#             import networkx as nx  # type: ignore
#             _is_multi = isinstance(G, nx.MultiDiGraph)
#         except Exception:
#             _is_multi = False

#         Gq = G
#         if _is_multi:
#             # Project via your S1 helper if available; otherwise keep as-is but tolerate parallel edges.
#             try:
#                 from cgx.graph.build_graph import project_graph_for_visualization  # type: ignore
#                 Gq = project_graph_for_visualization(G)  # DiGraph
#             except Exception:
#                 Gq = G  # fall back; we will read one edge's attrs when parallel edges exist

#         def _edge_ok(ed: Dict[str, Any]) -> bool:
#             try:
#                 et = ed.get("type")
#                 if relation_types and et not in set(relation_types):
#                     return False
#                 if internal_only is True and ed.get("internal") is not True:
#                     return False
#                 if internal_only is False and ed.get("internal") is not False:
#                     return False
#                 return True
#             except Exception:
#                 return False

#         for start in list(seeds):
#             if start not in Gq:
#                 continue
#             q = deque([(start, 0)])
#             visited = {start}
#             while q:
#                 nid, d = q.popleft()
#                 if d >= int(neighbor_depth):
#                     continue
#                 # explore both directions
#                 for nbr in list(Gq.successors(nid)) + list(Gq.predecessors(nid)):
#                     if nbr in visited:
#                         continue
#                     try:
#                         ed = Gq[nid][nbr] if Gq.has_edge(nid, nbr) else Gq[nbr][nid]
#                         # DiGraph: dict; MultiDiGraph: dict-of-dicts -> take any
#                         attrs = ed if isinstance(ed, dict) and not any(isinstance(v, dict) for v in ed.values()) \
#                                 else (list(ed.values())[0] if isinstance(ed, dict) and ed else {})
#                     except Exception:
#                         attrs = {}
#                     if _edge_ok(attrs):
#                         visited.add(nbr)
#                         q.append((nbr, d + 1))
#                         # only record nodes that look like chunk ids (fast check: contains '::' or endswith .py)
#                         if ("::" in str(nbr)) or str(nbr).endswith(".py"):
#                             if nbr not in graph_depths:
#                                 graph_depths[str(nbr)] = d + 1

#     # ---------- 4) RRF fusion ----------
#     intent_list = sem_lists.get("intent", [])
#     impl_list   = sem_lists.get("impl", [])
#     fused = _rrf_fuse([intent_list, impl_list], k=float(rrf_k))

#     # add lexical and graph signals as extra RRF terms
#     if lex_counts:
#         # convert counts -> pseudo ranked list by descending counts
#         lex_ids_sorted = sorted(lex_counts.items(), key=lambda kv: (-kv[1], kv[0]))
#         lex_list = [(cid, float(cnt)) for cid, cnt in lex_ids_sorted]
#         fused_lex = _rrf_fuse([lex_list], k=float(rrf_k))
#         for cid, fs in fused_lex.items():
#             fused[cid] = fused.get(cid, 0.0) + fs

#     if graph_depths and neighbor_depth > 0:
#         # smaller depth = better; map depth (1..k) to ascending rank
#         graph_ranked = sorted(graph_depths.items(), key=lambda kv: (kv[1], kv[0]))
#         graph_list = [(cid, float(-depth)) for cid, depth in graph_ranked]  # score not used, we use only rank
#         fused_graph = _rrf_fuse([graph_list], k=float(rrf_k))
#         for cid, fs in fused_graph.items():
#             fused[cid] = fused.get(cid, 0.0) + fs

#     # ---------- 5) Output with provenance ----------
#     # create compact provenance
#     results: List[Dict[str, Any]] = []
#     all_ids = sorted(fused.keys(), key=lambda cid: fused[cid], reverse=True)
#     for cid in all_ids:
#         prov = {
#             "intent_rank": ranks["intent"].get(cid),
#             "intent_score": scores["intent"].get(cid),
#             "impl_rank": ranks["impl"].get(cid),
#             "impl_score": scores["impl"].get(cid),
#             "lexical_count": lex_counts.get(cid),
#             "graph_depth": graph_depths.get(cid),
#         }
#         results.append({"chunk_id": cid, "score": fused[cid], "provenance": prov})

#     return results


# # ---------------------------
# # Aggregation to files/classes
# # ---------------------------

# def _record_map(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
#     return {str(r.get("id")): r for r in records if isinstance(r, dict) and r.get("id")}

# def aggregate_by_file(
#     fused_hits: List[Dict[str, Any]],
#     records: List[Dict[str, Any]],
#     *,
#     top_symbols_per_file: int = 2,
#     centrality_bonus: float = 0.15,   # small, deterministic bump from calls degree
# ) -> List[Dict[str, Any]]:
#     """
#     Aggregate fused chunk scores to file level and return top files with
#     their top entry symbols.

#     Heuristic (deterministic):
#       - Sum fused scores for chunks in the same file.
#       - Add a small centrality bonus proportional to (calls_in + calls_out) from records.

#     Returns
#     -------
#     list[dict]
#       [
#         {
#           "file": "/abs/path/x.py",
#           "score": float,
#           "top_symbols": [chunk_id, ... up to top_symbols_per_file]
#         },
#         ...
#       ]
#     """
#     rec_map = _record_map(records)
#     buckets: Dict[str, List[Tuple[str, float]]] = {}

#     for hit in fused_hits:
#         cid = hit.get("chunk_id")
#         sc = float(hit.get("score", 0.0))
#         r = rec_map.get(str(cid))
#         if not r:
#             continue
#         f = r.get("file")
#         if not f:
#             continue
#         buckets.setdefault(f, []).append((str(cid), sc))

#     out: List[Dict[str, Any]] = []
#     for f, pairs in buckets.items():
#         # base sum
#         total = sum(sc for _, sc in pairs)
#         # centrality from records (sum of in+out across chunks in this file)
#         deg = 0
#         for cid, _ in pairs:
#             rr = rec_map.get(cid) or {}
#             deg += int(rr.get("calls_in_count", 0)) + int(rr.get("calls_out_count", 0))
#         bonus = centrality_bonus * math.log1p(deg)
#         # top symbols inside file
#         pairs_sorted = sorted(pairs, key=lambda kv: kv[1], reverse=True)
#         top_syms = [cid for cid, _ in pairs_sorted[: int(top_symbols_per_file)]]
#         out.append({"file": f, "score": float(total + bonus), "top_symbols": top_syms})

#     out.sort(key=lambda d: d["score"], reverse=True)
#     return out


# def aggregate_by_class(
#     fused_hits: List[Dict[str, Any]],
#     records: List[Dict[str, Any]],
#     *,
#     top_methods_per_class: int = 2,
# ) -> List[Dict[str, Any]]:
#     """
#     Aggregate fused hits to classes (by parent_class_id). For each class, sum the
#     fused scores of its methods and return top classes with top methods.
#     """
#     rec_map = _record_map(records)
#     buckets: Dict[str, List[Tuple[str, float]]] = {}

#     for hit in fused_hits:
#         cid = str(hit.get("chunk_id"))
#         sc = float(hit.get("score", 0.0))
#         r = rec_map.get(cid)
#         if not r:
#             continue
#         parent_cls = r.get("parent_class_id")
#         if parent_cls:
#             buckets.setdefault(parent_cls, []).append((cid, sc))

#     out: List[Dict[str, Any]] = []
#     for cls_id, pairs in buckets.items():
#         total = sum(sc for _, sc in pairs)
#         pairs_sorted = sorted(pairs, key=lambda kv: kv[1], reverse=True)
#         top_methods = [cid for cid, _ in pairs_sorted[: int(top_methods_per_class)]]
#         out.append({"class_id": cls_id, "score": float(total), "top_methods": top_methods})

#     out.sort(key=lambda d: d["score"], reverse=True)
#     return out


# # ---------------------------
# # Where should new code go?
# # ---------------------------

# def _collect_exemplar_signals(
#     exemplar_ids: Sequence[str],
#     records: List[Dict[str, Any]],
# ) -> Dict[str, Any]:
#     """
#     Deterministically gather overlap signals from exemplars:
#       - imports_used union
#       - attributes_used_root_reads union
#       - signature parameter names multiset
#     """
#     rec_map = _record_map(records)
#     imports, attrs, params = set(), set(), []

#     for cid in exemplar_ids:
#         r = rec_map.get(str(cid))
#         if not r:
#             continue
#         for imp in r.get("imports_used") or []:
#             if isinstance(imp, str) and imp:
#                 imports.add(imp)
#         for ar in r.get("attributes_used_root_reads") or []:
#             if isinstance(ar, str) and ar:
#                 attrs.add(ar)
#         sig = r.get("signature") or ""
#         # extract parameter names inside parentheses (rough, deterministic)
#         m = re.search(r"\((.*)\)", sig)
#         if m:
#             inner = m.group(1)
#             names = [p.strip().split(":")[0].split("=")[0] for p in inner.split(",") if p.strip()]
#             params.extend([n for n in names if n and n != "self"])

#     return {
#         "imports": sorted(imports),
#         "attributes": sorted(attrs),
#         "param_names": sorted(params),
#     }


# def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
#     A, B = set(a), set(b)
#     if not A and not B:
#         return 0.0
#     inter = len(A & B)
#     union = max(1, len(A | B))
#     return float(inter) / float(union)


# def suggest_insertion_points(
#     query: str,
#     fused_hits: List[Dict[str, Any]],
#     records: List[Dict[str, Any]],
#     *,
#     k_candidates: int = 5,
#     k_exemplars: int = 5,
# ) -> List[Dict[str, Any]]:
#     """
#     Suggest where a **new function** should live (file/class), using only
#     deterministic signals:

#       1) Take top-k exemplars from fused hits (existing symbols).
#       2) Compute overlap signatures from exemplars: imports, attributes, param names.
#       3) Consider candidate anchors among class/file records and rank by:
#          - imports overlap (Jaccard)
#          - attribute-domain overlap (Jaccard)
#          - signature similarity to **neighboring** methods/functions inside the candidate
#            (best-of Jaccard over param names).
#       4) Return top ranked candidates with two anchors:
#          - a likely caller (highest calls_in_count in that container)
#          - a neighbor with most similar signature (best param-overlap)

#     Returns a deterministic, auditable list:
#       [{"container_type": "class"|"file",
#         "container_id": <id or file path>,
#         "score": <float>,
#         "anchors": {"likely_caller": <chunk_id|None>, "similar_signature_neighbor": <chunk_id|None>}}]
#     """
#     rec_map = _record_map(records)

#     # 1) exemplars
#     exemplar_ids = [str(h.get("chunk_id")) for h in fused_hits[: int(k_exemplars)]]
#     sigs = _collect_exemplar_signals(exemplar_ids, records)

#     # 2) collect candidate containers (files + classes) from records
#     file_scores: Dict[str, float] = {}
#     class_scores: Dict[str, float] = {}

#     # pre-index children for anchors
#     children_by_container: Dict[str, List[str]] = {}

#     for r in records:
#         rtype = r.get("type")
#         rid = r.get("id")
#         if not rid:
#             continue

#         if rtype == "file":
#             # overlap on imports/attrs against union signals
#             s_imp = _jaccard(r.get("imports_used", []), sigs["imports"])
#             s_att = _jaccard(r.get("attributes_used_root_reads", []), sigs["attributes"])
#             score = 0.55 * s_imp + 0.45 * s_att
#             file_scores[rid] = score
#             # children (methods/functions defined in this file)
#             children_by_container[rid] = list(r.get("defines_children_ids") or [])

#         elif rtype == "class":
#             s_imp = _jaccard(r.get("imports_used", []), sigs["imports"])
#             s_att = _jaccard(r.get("attributes_used_root_reads", []), sigs["attributes"])
#             score = 0.55 * s_imp + 0.45 * s_att
#             class_scores[rid] = score
#             children_by_container[rid] = list(r.get("defines_children_ids") or [])

#     # 3) signature similarity term: best param-name overlap within container’s children
#     def _best_signature_overlap(child_ids: List[str]) -> float:
#         best = 0.0
#         for cid in child_ids:
#             rr = rec_map.get(cid) or {}
#             sig = rr.get("signature") or ""
#             m = re.search(r"\((.*)\)", sig)
#             if not m:
#                 continue
#             names = [p.strip().split(":")[0].split("=")[0] for p in m.group(1).split(",") if p.strip()]
#             names = [n for n in names if n and n != "self"]
#             best = max(best, _jaccard(names, sigs["param_names"]))
#         return best

#     for rid in list(file_scores.keys()):
#         file_scores[rid] += 0.3 * _best_signature_overlap(children_by_container.get(rid, []))

#     for rid in list(class_scores.keys()):
#         class_scores[rid] += 0.3 * _best_signature_overlap(children_by_container.get(rid, []))

#     # 4) anchors: pick likely caller (highest calls_in_count) and most similar signature neighbor
#     def _likely_caller(child_ids: List[str]) -> Optional[str]:
#         best_id, best_deg = None, -1
#         for cid in child_ids:
#             rr = rec_map.get(cid) or {}
#             deg = int(rr.get("calls_in_count", 0))
#             if deg > best_deg:
#                 best_deg = deg
#                 best_id = cid
#         return best_id

#     def _similar_signature_neighbor(child_ids: List[str]) -> Optional[str]:
#         best_id, best_sim = None, -1.0
#         for cid in child_ids:
#             rr = rec_map.get(cid) or {}
#             sig = rr.get("signature") or ""
#             m = re.search(r"\((.*)\)", sig)
#             if not m:
#                 continue
#             names = [p.strip().split(":")[0].split("=")[0] for p in m.group(1).split(",") if p.strip()]
#             names = [n for n in names if n and n != "self"]
#             sim = _jaccard(names, sigs["param_names"])
#             if sim > best_sim:
#                 best_sim = sim
#                 best_id = cid
#         return best_id

#     # 5) finalize candidates
#     file_ranked = sorted(file_scores.items(), key=lambda kv: (-kv[1], kv[0]))[: int(k_candidates)]
#     class_ranked = sorted(class_scores.items(), key=lambda kv: (-kv[1], kv[0]))[: int(k_candidates)]

#     out: List[Dict[str, Any]] = []
#     for rid, sc in class_ranked:
#         kids = children_by_container.get(rid, [])
#         out.append({
#             "container_type": "class",
#             "container_id": rid,
#             "score": float(sc),
#             "anchors": {
#                 "likely_caller": _likely_caller(kids),
#                 "similar_signature_neighbor": _similar_signature_neighbor(kids),
#             }
#         })
#     for rid, sc in file_ranked:
#         kids = children_by_container.get(rid, [])
#         out.append({
#             "container_type": "file",
#             "container_id": rid,
#             "score": float(sc),
#             "anchors": {
#                 "likely_caller": _likely_caller(kids),
#                 "similar_signature_neighbor": _similar_signature_neighbor(kids),
#             }
#         })

#     # sort combined deterministically
#     out.sort(key=lambda d: (-d["score"], d["container_type"], d["container_id"]))
#     return out


# src/cgx/retrieval/orchestrator.py
from __future__ import annotations

"""
Two-view retrieval orchestrator (ADD-ONLY).

Purpose
-------
Provide deterministic, auditable retrieval that:
  - Builds one FAISS index per VIEW ("intent", "impl") from S4 records/corpus.
  - Runs semantic search on both views, optional lexical search on chunks,
    and graph expansion, then fuses with RRF.
  - Aggregates to files/classes and suggests insertion points for new code
    using deterministic overlap signals (imports/attributes/signatures).

This module DOES NOT:
  - Assume any specific embedding model.
  - Modify existing parse/graph/embedding code.
  - Change return shapes elsewhere.
"""

import math
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------
# Optional helper imports (prefer project helpers, fall back to internal impls)
# ---------------------------
try:
    from src.cgx.embeddings.build import build_embeddings as _build_embeddings  # type: ignore
except Exception:  # pragma: no cover
    _build_embeddings = None  # type: ignore

try:
    from src.cgx.embeddings.index import build_faiss_index as _build_faiss_index  # type: ignore
except Exception:  # pragma: no cover
    _build_faiss_index = None  # type: ignore

try:
    from src.cgx.embeddings.search import semantic_search as _semantic_search  # type: ignore
except Exception:  # pragma: no cover
    _semantic_search = None  # type: ignore

try:
    from src.cgx.embeddings.views import attach_views_to_chunks as _attach_views_to_chunks  # type: ignore
except Exception:  # pragma: no cover
    _attach_views_to_chunks = None  # type: ignore

try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except Exception:  # pragma: no cover
    faiss = None  # type: ignore
    _FAISS_AVAILABLE = False

# local imports are optional; this module can function without them
try:
    from src.cgx.retrieval.hybrid import lexical_search as _lexical_search_default  # optional
except Exception:  # pragma: no cover
    _lexical_search_default = None  # type: ignore

from cgx.logging_setup import get_logger
logger = get_logger("orchestration")


# ---------------------------
# Utilities
# ---------------------------

def _norm_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return (x / denom).astype("float32", copy=False)


def encode_texts(
    texts: Sequence[str],
    embedder: Any,
    *,
    normalize: bool = True,
) -> np.ndarray:
    if not isinstance(texts, (list, tuple)):
        raise TypeError(f"encode_texts: 'texts' must be a list/tuple, got {type(texts)}")
    if len(texts) == 0:
        raise ValueError("encode_texts: 'texts' must be non-empty.")

    if hasattr(embedder, "encode"):
        vecs = embedder.encode(list(texts))  # type: ignore[attr-defined]
    elif callable(embedder):
        vecs = embedder(list(texts))
    else:
        raise TypeError("encode_texts: 'embedder' must provide .encode(...) or be callable(list[str])->ndarray")

    arr = np.asarray(vecs, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != len(texts):
        raise ValueError(f"encode_texts: encoder returned shape {arr.shape}, expected (N,D) with N={len(texts)}")

    return _norm_rows(arr) if normalize else arr


def _faiss_build_index(
    embeddings: np.ndarray,
    *,
    metric: str = "cosine",
    ids: Optional[np.ndarray] = None,
    index_type: str = "flat",
    nlist: int = 1024,
    nprobe: int = 16,
    M: int = 32,
    efConstruction: int = 200,
    efSearch: int = 64,
) -> Tuple[Any, Dict[str, Any]]:
    if not _FAISS_AVAILABLE:
        raise RuntimeError("FAISS is not installed. Install 'faiss-cpu' or 'faiss-gpu' to use the orchestrator indices.")

    X = np.asarray(embeddings, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("embeddings must be 2D (N,D).")
    N, D = X.shape
    if N == 0 or D == 0:
        raise ValueError("embeddings must be non-empty.")

    if ids is None:
        ids = np.arange(N, dtype=np.int64)
    else:
        ids = np.asarray(ids, dtype=np.int64)
        if ids.shape != (N,):
            raise ValueError(f"'ids' must have shape (N,), got {ids.shape} for N={N}")

    m = metric.lower()
    if m not in {"cosine", "l2", "ip"}:
        raise ValueError("metric must be one of {'cosine','l2','ip'}")

    use_ip = (m in {"cosine", "ip"})
    index_flat = faiss.IndexFlatIP(D) if use_ip else faiss.IndexFlatL2(D)

    if index_type == "flat":
        base = index_flat
    elif index_type == "ivf":
        nlist_eff = int(nlist)
        if N < nlist_eff:
            nlist_eff = max(1, int(max(1, round(math.sqrt(N)))))
        base = faiss.IndexIVFFlat(index_flat, D, nlist_eff, faiss.METRIC_INNER_PRODUCT if use_ip else faiss.METRIC_L2)
        if not base.is_trained:
            base.train(X)
        base.nprobe = int(nprobe)
    elif index_type == "hnsw":
        try:
            base = faiss.IndexHNSWFlat(D, int(M), faiss.METRIC_INNER_PRODUCT if use_ip else faiss.METRIC_L2)
        except TypeError:
            base = faiss.IndexHNSWFlat(D, int(M))
        base.hnsw.efConstruction = int(efConstruction)
        base.hnsw.efSearch = int(efSearch)
    else:
        raise ValueError("index_type must be 'flat', 'ivf' or 'hnsw'.")

    try:
        idmap = faiss.IndexIDMap2(base)
    except Exception:
        idmap = faiss.IndexIDMap(base)
    idmap.add_with_ids(X, ids)

    meta = {
        "metric": m,
        "index_type": index_type,
        "dim": int(D),
        "num_vectors": int(N),
        "nlist": int(getattr(base, "nlist", 0)) if hasattr(base, "nlist") else None,
        "nprobe": int(getattr(base, "nprobe", 0)) if hasattr(base, "nprobe") else None,
        "M": int(M) if index_type == "hnsw" else None,
        "efConstruction": int(efConstruction) if index_type == "hnsw" else None,
        "efSearch": int(efSearch) if index_type == "hnsw" else None,
        "normalized_for_cosine": (m == "cosine"),
    }
    return idmap, meta


def _semantic_search_faiss(
    query: str,
    embedder: Any,
    index: Any,
    *,
    metric: str = "cosine",
    top_k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    q = encode_texts([query], embedder, normalize=(metric in {"cosine", "ip"}))
    D, I = index.search(q, int(top_k))
    return D[0], I[0]


# ---------------------------
# Helper adapters to prefer project helpers with graceful fallback
# ---------------------------

def _use_helper_build_embeddings(
    texts: Sequence[str],
    embedder: Any,
    *,
    normalize: bool,
) -> np.ndarray:
    if _build_embeddings is not None:
        try:
            return np.asarray(
                _build_embeddings(texts, embedder=embedder, normalize=normalize),
                dtype=np.float32,
            )
        except TypeError:
            try:
                arr = _build_embeddings(texts, embedder)  # type: ignore[misc]
                arr = np.asarray(arr, dtype=np.float32)
                return _norm_rows(arr) if normalize else arr
            except Exception:
                logger.debug("build_embeddings signature mismatch; falling back to encode_texts.")
        except Exception as e:
            logger.debug("build_embeddings failed (%s); falling back to encode_texts.", e)
    return encode_texts(texts, embedder, normalize=normalize)


def _use_helper_build_faiss_index(
    embs: np.ndarray,
    *,
    metric: str,
    ids: np.ndarray,
    index_type: str,
    nlist: int,
    nprobe: int,
    M: int,
    efConstruction: int,
    efSearch: int,
) -> Tuple[Any, Dict[str, Any]]:
    if _build_faiss_index is not None:
        try:
            return _build_faiss_index(
                embs,
                metric=metric,
                ids=ids,
                index_type=index_type,
                nlist=nlist,
                nprobe=nprobe,
                M=M,
                efConstruction=efConstruction,
                efSearch=efSearch,
            )
        except TypeError:
            try:
                return _build_faiss_index(embs, metric, ids, index_type)  # type: ignore[misc]
            except Exception as e:
                logger.debug("build_faiss_index signature mismatch; fallback to internal. (%s)", e)
        except Exception as e:
            logger.debug("build_faiss_index failed; fallback to internal. (%s)", e)
    return _faiss_build_index(
        embs,
        metric=metric,
        ids=ids,
        index_type=index_type,
        nlist=nlist,
        nprobe=nprobe,
        M=M,
        efConstruction=efConstruction,
        efSearch=efSearch,
    )


def _use_helper_semantic_search(
    query: str,
    *,
    embedder: Any,
    index: Any,
    rows: List[Dict[str, Any]],
    metric: str,
    top_k: int,
) -> List[Tuple[str, float]]:
    if _semantic_search is not None:
        try:
            lst = _semantic_search(
                query=query,
                embedder=embedder,
                index=index,
                rows=rows,
                metric=metric,
                top_k=top_k,
            )
            if isinstance(lst, list):
                return [(str(cid), float(sc)) for cid, sc in lst]
        except TypeError:
            try:
                lst = _semantic_search(query, embedder, index, rows, top_k)  # type: ignore[misc]
                if isinstance(lst, list):
                    return [(str(cid), float(sc)) for cid, sc in lst]
            except Exception as e2:
                logger.debug("semantic_search signature mismatch; fallback to internal. (%s)", e2)
        except Exception as e:
            logger.debug("semantic_search failed; fallback to internal. (%s)", e)
    D, I = _semantic_search_faiss(query, embedder, index, metric=metric, top_k=top_k)
    return _ranklist_from_dist_ids(D, I, rows, metric=metric)


def _maybe_attach_views(intent_list: List[Tuple[str, float]], impl_list: List[Tuple[str, float]]) -> None:
    if _attach_views_to_chunks is None:
        return
    try:
        _attach_views_to_chunks({"intent": intent_list, "impl": impl_list})  # type: ignore[misc]
    except TypeError:
        try:
            _attach_views_to_chunks(intent_list, impl_list)  # type: ignore[misc]
        except Exception:
            pass
    except Exception:
        pass


# ---------------------------
# NEW: query tokenization & symbol extraction for lexical/boosting
# ---------------------------

_COMMON_STOP = {
    "what","does","this","function","class","explain","describe","how","to","where",
    "the","a","an","is","do","of","in","on","for","with","and","or","by","it","that"
}

def _extract_symbol_tokens(q: str) -> List[str]:
    q = q or ""
    quoted = re.findall(r"[`\"]([A-Za-z_][A-Za-z0-9_]*)[`\"]", q)
    bare = re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", q)
    out, seen = [], set()
    for t in quoted + bare:
        tl = t.lower()
        if tl in _COMMON_STOP:
            continue
        if tl not in seen:
            seen.add(tl); out.append(t)
    return out


# ---------------------------
# Public API: indices per view
# ---------------------------

__all__ = [
    "build_two_view_indices",
    "semantic_retrieve_two_view",
    "hybrid_retrieve_two_view",
    "aggregate_by_file",
    "aggregate_by_class",
    "suggest_insertion_points",
]

def build_two_view_indices(
    corpus: List[Dict[str, Any]],
    *,
    embedder: Any,
    metric: str = "cosine",
    index_type: str = "flat",
    # advanced FAISS knobs (optional)
    nlist: int = 1024,
    nprobe: int = 16,
    M: int = 32,
    efConstruction: int = 200,
    efSearch: int = 64,
) -> Dict[str, Any]:
    if not isinstance(corpus, list) or not corpus:
        raise ValueError("build_two_view_indices: 'corpus' must be a non-empty list.")

    per_view: Dict[str, List[Dict[str, Any]]] = {"intent": [], "impl": []}
    for row in corpus:
        vw = row.get("view")
        if vw in per_view:
            per_view[vw].append(row)

    result: Dict[str, Any] = {"views": {}, "metric": metric}

    for view_name, rows in per_view.items():
        if not rows:
            logger.warning("No rows for view '%s'; skipping index build.", view_name)
            result["views"][view_name] = {"index": None, "meta": None, "rows": [], "ids": np.array([], dtype=np.int64)}
            continue

        texts = [str(r.get("text", "")) for r in rows]
        normalize = (metric.lower() in {"cosine", "ip"})
        embs = _use_helper_build_embeddings(texts, embedder, normalize=normalize)

        ids = np.arange(len(rows), dtype=np.int64)
        index, meta = _use_helper_build_faiss_index(
            embs,
            metric=metric,
            ids=ids,
            index_type=index_type,
            nlist=nlist,
            nprobe=nprobe,
            M=M,
            efConstruction=efConstruction,
            efSearch=efSearch,
        )

        result["views"][view_name] = {"index": index, "meta": meta, "rows": rows, "ids": ids}

    return result


# ---------------------------
# Semantic on both views + RRF
# ---------------------------

def _ranklist_from_dist_ids(
    distances: np.ndarray,
    ids: np.ndarray,
    rows: List[Dict[str, Any]],
    *,
    metric: str,
) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    for d, i in zip(distances, ids):
        if int(i) < 0:
            continue
        row = rows[int(i)]
        cid = row.get("chunk_id")
        if cid is None:
            continue
        if metric in {"cosine", "ip"}:
            score = float(d)
        else:
            score = float(-d)
        out.append((str(cid), score))
    return out


def _rrf_fuse(
    lists: Sequence[List[Tuple[str, float]]],
    *,
    k: float = 60.0,
) -> Dict[str, float]:
    fused: Dict[str, float] = {}
    for lst in lists:
        for rank, (cid, _s) in enumerate(lst, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
    return fused


def semantic_retrieve_two_view(
    query: str,
    indices: Dict[str, Any],
    *,
    top_k_per_view: int = 10,
) -> Dict[str, List[Tuple[str, float]]]:
    metric = indices.get("metric", "cosine")
    out: Dict[str, List[Tuple[str, float]]] = {"intent": [], "impl": []}
    for view_name in ("intent", "impl"):
        view = indices.get("views", {}).get(view_name) or {}
        index = view.get("index")
        rows = view.get("rows") or []
        if index is None or not rows:
            continue
        raise RuntimeError(
            "semantic_retrieve_two_view requires an embedder; "
            "use hybrid_retrieve_two_view(query, indices=..., embedder=...)."
        )
    return out


def hybrid_retrieve_two_view(
    query: str,
    *,
    indices: Dict[str, Any],
    embedder: Any,
    chunks: Optional[List[Dict[str, Any]]] = None,
    G: Any = None,
    # semantic
    top_k_per_view: int = 10,
    # lexical (over chunks)
    use_lexical: bool = True,
    lexical_search_fn: Optional[Callable[..., List[Dict[str, Any]]]] = None,
    lex_fields: Iterable[str] = ("code", "name", "id", "file", "meta.docstring"),
    lex_regex: bool = True,
    lex_case_sensitive: bool = False,
    lex_whole_word: bool = False,
    # graph expansion
    neighbor_depth: int = 1,
    relation_types: Optional[Iterable[str]] = ("calls", "defines"),
    internal_only: Optional[bool] = None,
    # fusion
    rrf_k: float = 60.0,
) -> List[Dict[str, Any]]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("hybrid_retrieve_two_view: 'query' must be a non-empty string.")

    metric = indices.get("metric", "cosine")
    views = indices.get("views", {})
    intent_v = views.get("intent") or {}
    impl_v   = views.get("impl") or {}

    # -------- 0) symbol tokens (used for lexical + boosting)
    sym_tokens = _extract_symbol_tokens(query)

    # ---------- 1) Semantic per view ----------
    sem_lists: Dict[str, List[Tuple[str, float]]] = {"intent": [], "impl": []}
    ranks: Dict[str, Dict[str, int]] = {"intent": {}, "impl": {}}
    scores: Dict[str, Dict[str, float]] = {"intent": {}, "impl": {}}

    for view_name, view in (("intent", intent_v), ("impl", impl_v)):
        index = view.get("index")
        rows  = view.get("rows") or []
        if index is None or not rows:
            continue

        lst = _use_helper_semantic_search(
            query,
            embedder=embedder,
            index=index,
            rows=rows,
            metric=metric,
            top_k=top_k_per_view,
        )
        sem_lists[view_name] = lst
        for r, (cid, sc) in enumerate(lst, start=1):
            ranks[view_name][cid] = r
            scores[view_name][cid] = sc

    _maybe_attach_views(sem_lists.get("intent", []), sem_lists.get("impl", []))

    # ---------- 2) Lexical over chunks (tokenized) ----------
    lex_counts: Dict[str, int] = {}
    if use_lexical and chunks:
        # prefer project lexical fn; otherwise token-aware fallback
        search_fn = lexical_search_fn or _lexical_search_default
        if search_fn is None:
            # Build a token-aware regex (prefer symbol tokens; else content tokens)
            tokens = sym_tokens[:]
            if not tokens:
                tokens = [t for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", query) if t.lower() not in _COMMON_STOP]
            # no tokens -> skip lexical
            if tokens:
                # \btoken\b OR join, unless whole_word False
                def _mk_rx(tok: str, whole: bool) -> str:
                    return (r"\b" + re.escape(tok) + r"\b") if whole else re.escape(tok)
                pat = r"(" + "|".join(_mk_rx(t, bool(lex_whole_word)) for t in tokens) + r")"
                rx = re.compile(pat, 0 if bool(lex_case_sensitive) else re.IGNORECASE)

                def _field_get(d: Dict[str, Any], dotted: str) -> str:
                    cur: Any = d
                    for p in dotted.split("."):
                        cur = cur.get(p, "") if isinstance(cur, dict) else ""
                    return str(cur or "")

                for ch in chunks:
                    cid = str(ch.get("id") or "")
                    if not cid:
                        continue
                    total = 0
                    for f in tuple(lex_fields):
                        s = _field_get(ch, f)
                        if not s:
                            continue
                        total += len(list(rx.finditer(s)))
                    if total:
                        lex_counts[cid] = total
        else:
            try:
                lex = search_fn(
                    query, chunks,
                    fields=tuple(lex_fields),
                    regex=bool(lex_regex),
                    case_sensitive=bool(lex_case_sensitive),
                    whole_word=bool(lex_whole_word),
                )
                for r in lex:
                    cid = r.get("chunk", {}).get("id")
                    if cid:
                        lex_counts[str(cid)] = int(r.get("total_matches", 0))
            except Exception as e:
                logger.warning("Lexical search failed; continuing without lexical. (%s)", e)

    # ---------- 3) Graph expansion (seed with ids AND files) ----------
    graph_depths: Dict[str, int] = {}
    if G is not None:
        from collections import deque
        seeds = set()
        seeds.update([cid for cid, _ in sem_lists.get("intent", [])])
        seeds.update([cid for cid, _ in sem_lists.get("impl", [])])
        seeds.update(lex_counts.keys())

        # also seed with file paths for each known seed (helps graphs keyed by file nodes)
        id_to_file: Dict[str, str] = {}
        if chunks:
            for ch in chunks:
                cid = str(ch.get("id") or "")
                fp = str(ch.get("file") or "")
                if cid and fp:
                    id_to_file[cid] = fp
        for s in list(seeds):
            fp = id_to_file.get(str(s))
            if fp:
                seeds.add(fp)

        try:
            import networkx as nx  # type: ignore
            _is_multi = isinstance(G, nx.MultiDiGraph)
        except Exception:
            _is_multi = False

        Gq = G
        if _is_multi:
            try:
                from cgx.graph.build_graph import project_graph_for_visualization  # type: ignore
                Gq = project_graph_for_visualization(G)  # DiGraph
            except Exception:
                Gq = G

        def _edge_ok(ed: Dict[str, Any]) -> bool:
            try:
                et = ed.get("type")
                if relation_types and et not in set(relation_types):
                    return False
                if internal_only is True and ed.get("internal") is not True:
                    return False
                if internal_only is False and ed.get("internal") is not False:
                    return False
                return True
            except Exception:
                return False

        for start in list(seeds):
            if start not in Gq:
                continue
            qd = deque([(start, 0)])
            visited = {start}
            while qd:
                nid, d = qd.popleft()
                if d >= int(neighbor_depth):
                    continue
                for nbr in list(Gq.successors(nid)) + list(Gq.predecessors(nid)):
                    if nbr in visited:
                        continue
                    try:
                        ed = Gq[nid][nbr] if Gq.has_edge(nid, nbr) else Gq[nbr][nid]
                        attrs = ed if isinstance(ed, dict) and not any(isinstance(v, dict) for v in ed.values()) \
                                else (list(ed.values())[0] if isinstance(ed, dict) and ed else {})
                    except Exception:
                        attrs = {}
                    if _edge_ok(attrs):
                        visited.add(nbr)
                        qd.append((nbr, d + 1))
                        if ("::" in str(nbr)) or str(nbr).endswith(".py"):
                            if str(nbr) not in graph_depths:
                                graph_depths[str(nbr)] = d + 1

    # ---------- 4) Symbol-first boosting ----------
    symbol_boost: Dict[str, int] = {}
    if sym_tokens:
        # scan both views for exact symbol name or cid suffix match
        for view in ("intent", "impl"):
            vw = (indices.get("views") or {}).get(view) or {}
            for r in (vw.get("rows") or []):
                cid = str(r.get("chunk_id") or "")
                name = str(r.get("name") or "")
                nm_l = name.lower()
                for t in sym_tokens:
                    tl = t.lower()
                    if (f"::{t}" in cid) or (f"::{tl}" in cid.lower()) or (nm_l == tl):
                        symbol_boost[cid] = max(symbol_boost.get(cid, 0), 2)  # strong discrete boost

    # ---------- 5) RRF fusion + additive boosts ----------
    intent_list = sem_lists.get("intent", [])
    impl_list   = sem_lists.get("impl", [])
    fused = _rrf_fuse([intent_list, impl_list], k=float(rrf_k))

    # lexical term (rank-only via RRF on counts)
    if lex_counts:
        lex_ids_sorted = sorted(lex_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        lex_list = [(cid, float(cnt)) for cid, cnt in lex_ids_sorted]
        fused_lex = _rrf_fuse([lex_list], k=float(rrf_k))
        for cid, fs in fused_lex.items():
            fused[cid] = fused.get(cid, 0.0) + fs

    # graph proximity (rank-only)
    if graph_depths and neighbor_depth > 0:
        graph_ranked = sorted(graph_depths.items(), key=lambda kv: (kv[1], kv[0]))
        graph_list = [(cid, float(-depth)) for cid, depth in graph_ranked]
        fused_graph = _rrf_fuse([graph_list], k=float(rrf_k))
        for cid, fs in fused_graph.items():
            fused[cid] = fused.get(cid, 0.0) + fs

    # additive symbol boost (explicit, so it really bubbles up)
    if symbol_boost:
        for cid, times in symbol_boost.items():
            fused[cid] = fused.get(cid, 0.0) + 1.0 * float(times)

    # ---------- 6) Output with provenance ----------
    results: List[Dict[str, Any]] = []
    all_ids = sorted(fused.keys(), key=lambda cid: fused[cid], reverse=True)
    for cid in all_ids:
        prov = {
            "intent_rank": ranks["intent"].get(cid),
            "intent_score": scores["intent"].get(cid),
            "impl_rank": ranks["impl"].get(cid),
            "impl_score": scores["impl"].get(cid),
            "lexical_count": lex_counts.get(cid),
            "graph_depth": graph_depths.get(cid),
            "symbol_match": bool(cid in symbol_boost),
        }
        results.append({"chunk_id": cid, "score": fused[cid], "provenance": prov})

    return results


# ---------------------------
# Aggregation to files/classes
# ---------------------------

def _record_map(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(r.get("id")): r for r in records if isinstance(r, dict) and r.get("id")}

def aggregate_by_file(
    fused_hits: List[Dict[str, Any]],
    records: List[Dict[str, Any]],
    *,
    top_symbols_per_file: int = 2,
    centrality_bonus: float = 0.15,
) -> List[Dict[str, Any]]:
    rec_map = _record_map(records)
    buckets: Dict[str, List[Tuple[str, float]]] = {}

    for hit in fused_hits:
        cid = hit.get("chunk_id")
        sc = float(hit.get("score", 0.0))
        r = rec_map.get(str(cid))
        if not r:
            continue
        f = r.get("file")
        if not f:
            continue
        buckets.setdefault(f, []).append((str(cid), sc))

    out: List[Dict[str, Any]] = []
    for f, pairs in buckets.items():
        total = sum(sc for _, sc in pairs)
        deg = 0
        for cid, _ in pairs:
            rr = rec_map.get(cid) or {}
            deg += int(rr.get("calls_in_count", 0)) + int(rr.get("calls_out_count", 0))
        bonus = centrality_bonus * math.log1p(deg)
        pairs_sorted = sorted(pairs, key=lambda kv: kv[1], reverse=True)
        top_syms = [cid for cid, _ in pairs_sorted[: int(top_symbols_per_file)]]
        out.append({"file": f, "score": float(total + bonus), "top_symbols": top_syms})

    out.sort(key=lambda d: d["score"], reverse=True)
    return out


def aggregate_by_class(
    fused_hits: List[Dict[str, Any]],
    records: List[Dict[str, Any]],
    *,
    top_methods_per_class: int = 2,
) -> List[Dict[str, Any]]:
    rec_map = _record_map(records)
    buckets: Dict[str, List[Tuple[str, float]]] = {}

    for hit in fused_hits:
        cid = str(hit.get("chunk_id"))
        sc = float(hit.get("score", 0.0))
        r = rec_map.get(cid)
        if not r:
            continue
        parent_cls = r.get("parent_class_id")
        if parent_cls:
            buckets.setdefault(parent_cls, []).append((cid, sc))

    out: List[Dict[str, Any]] = []
    for cls_id, pairs in buckets.items():
        total = sum(sc for _, sc in pairs)
        pairs_sorted = sorted(pairs, key=lambda kv: kv[1], reverse=True)
        top_methods = [cid for cid, _ in pairs_sorted[: int(top_methods_per_class)]]
        out.append({"class_id": cls_id, "score": float(total), "top_methods": top_methods})

    out.sort(key=lambda d: d["score"], reverse=True)
    return out


# ---------------------------
# Where should new code go?
# ---------------------------

def _collect_exemplar_signals(
    exemplar_ids: Sequence[str],
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    rec_map = _record_map(records)
    imports, attrs, params = set(), set(), []

    for cid in exemplar_ids:
        r = rec_map.get(str(cid))
        if not r:
            continue
        for imp in r.get("imports_used") or []:
            if isinstance(imp, str) and imp:
                imports.add(imp)
        for ar in r.get("attributes_used_root_reads") or []:
            if isinstance(ar, str) and ar:
                attrs.add(ar)
        sig = r.get("signature") or ""
        m = re.search(r"\((.*)\)", sig)
        if m:
            inner = m.group(1)
            names = [p.strip().split(":")[0].split("=")[0] for p in inner.split(",") if p.strip()]
            params.extend([n for n in names if n and n != "self"])

    return {
        "imports": sorted(imports),
        "attributes": sorted(attrs),
        "param_names": sorted(params),
    }


def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 0.0
    inter = len(A & B)
    union = max(1, len(A | B))
    return float(inter) / float(union)


def suggest_insertion_points(
    query: str,
    fused_hits: List[Dict[str, Any]],
    records: List[Dict[str, Any]],
    *,
    k_candidates: int = 5,
    k_exemplars: int = 5,
) -> List[Dict[str, Any]]:
    rec_map = _record_map(records)

    exemplar_ids = [str(h.get("chunk_id")) for h in fused_hits[: int(k_exemplars)]]
    sigs = _collect_exemplar_signals(exemplar_ids, records)

    file_scores: Dict[str, float] = {}
    class_scores: Dict[str, float] = {}

    children_by_container: Dict[str, List[str]] = {}

    for r in records:
        rtype = r.get("type")
        rid = r.get("id")
        if not rid:
            continue

        if rtype == "file":
            s_imp = _jaccard(r.get("imports_used", []), sigs["imports"])
            s_att = _jaccard(r.get("attributes_used_root_reads", []), sigs["attributes"])
            score = 0.55 * s_imp + 0.45 * s_att
            file_scores[rid] = score
            children_by_container[rid] = list(r.get("defines_children_ids") or [])

        elif rtype == "class":
            s_imp = _jaccard(r.get("imports_used", []), sigs["imports"])
            s_att = _jaccard(r.get("attributes_used_root_reads", []), sigs["attributes"])
            score = 0.55 * s_imp + 0.45 * s_att
            class_scores[rid] = score
            children_by_container[rid] = list(r.get("defines_children_ids") or [])

    def _best_signature_overlap(child_ids: List[str]) -> float:
        best = 0.0
        for cid in child_ids:
            rr = rec_map.get(cid) or {}
            sig = rr.get("signature") or ""
            m = re.search(r"\((.*)\)", sig)
            if not m:
                continue
            names = [p.strip().split(":")[0].split("=")[0] for p in m.group(1).split(",") if p.strip()]
            names = [n for n in names if n and n != "self"]
            best = max(best, _jaccard(names, sigs["param_names"]))
        return best

    for rid in list(file_scores.keys()):
        file_scores[rid] += 0.3 * _best_signature_overlap(children_by_container.get(rid, []))

    for rid in list(class_scores.keys()):
        class_scores[rid] += 0.3 * _best_signature_overlap(children_by_container.get(rid, []))

    def _likely_caller(child_ids: List[str]) -> Optional[str]:
        best_id, best_deg = None, -1
        for cid in child_ids:
            rr = rec_map.get(cid) or {}
            deg = int(rr.get("calls_in_count", 0))
            if deg > best_deg:
                best_deg = deg
                best_id = cid
        return best_id

    def _similar_signature_neighbor(child_ids: List[str]) -> Optional[str]:
        best_id, best_sim = None, -1.0
        for cid in child_ids:
            rr = rec_map.get(cid) or {}
            sig = rr.get("signature") or ""
            m = re.search(r"\((.*)\)", sig)
            if not m:
                continue
            names = [p.strip().split(":")[0].split("=")[0] for p in m.group(1).split(",") if p.strip()]
            names = [n for n in names if n and n != "self"]
            sim = _jaccard(names, sigs["param_names"])
            if sim > best_sim:
                best_sim = sim
                best_id = cid
        return best_id

    file_ranked = sorted(file_scores.items(), key=lambda kv: (-kv[1], kv[0]))[: int(k_candidates)]
    class_ranked = sorted(class_scores.items(), key=lambda kv: (-kv[1], kv[0]))[: int(k_candidates)]

    out: List[Dict[str, Any]] = []
    for rid, sc in class_ranked:
        kids = children_by_container.get(rid, [])
        out.append({
            "container_type": "class",
            "container_id": rid,
            "score": float(sc),
            "anchors": {
                "likely_caller": _likely_caller(kids),
                "similar_signature_neighbor": _similar_signature_neighbor(kids),
            }
        })
    for rid, sc in file_ranked:
        kids = children_by_container.get(rid, [])
        out.append({
            "container_type": "file",
            "container_id": rid,
            "score": float(sc),
            "anchors": {
                "likely_caller": _likely_caller(kids),
                "similar_signature_neighbor": _similar_signature_neighbor(kids),
            }
        })

    out.sort(key=lambda d: (-d["score"], d["container_type"], d["container_id"]))
    return out
