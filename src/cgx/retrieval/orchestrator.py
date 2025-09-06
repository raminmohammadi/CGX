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
#     x = np.asarray(x, dtype=np.float32)
#     denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
#     return (x / denom).astype("float32", copy=False)


# def encode_texts(
#     texts: Sequence[str],
#     embedder: Any,
#     *,
#     normalize: bool = True,
# ) -> np.ndarray:
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
#     if _build_embeddings is not None:
#         try:
#             return np.asarray(
#                 _build_embeddings(texts, embedder=embedder, normalize=normalize),
#                 dtype=np.float32,
#             )
#         except TypeError:
#             try:
#                 arr = _build_embeddings(texts, embedder)  # type: ignore[misc]
#                 arr = np.asarray(arr, dtype=np.float32)
#                 return _norm_rows(arr) if normalize else arr
#             except Exception:
#                 logger.debug("build_embeddings signature mismatch; falling back to encode_texts.")
#         except Exception as e:
#             logger.debug("build_embeddings failed (%s); falling back to encode_texts.", e)
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
#     if _semantic_search is not None:
#         try:
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
#             try:
#                 lst = _semantic_search(query, embedder, index, rows, top_k)  # type: ignore[misc]
#                 if isinstance(lst, list):
#                     return [(str(cid), float(sc)) for cid, sc in lst]
#             except Exception as e2:
#                 logger.debug("semantic_search signature mismatch; fallback to internal. (%s)", e2)
#         except Exception as e:
#             logger.debug("semantic_search failed; fallback to internal. (%s)", e)
#     D, I = _semantic_search_faiss(query, embedder, index, metric=metric, top_k=top_k)
#     return _ranklist_from_dist_ids(D, I, rows, metric=metric)


# def _maybe_attach_views(intent_list: List[Tuple[str, float]], impl_list: List[Tuple[str, float]]) -> None:
#     if _attach_views_to_chunks is None:
#         return
#     try:
#         _attach_views_to_chunks({"intent": intent_list, "impl": impl_list})  # type: ignore[misc]
#     except TypeError:
#         try:
#             _attach_views_to_chunks(intent_list, impl_list)  # type: ignore[misc]
#         except Exception:
#             pass
#     except Exception:
#         pass


# # ---------------------------
# # NEW: query tokenization & symbol extraction for lexical/boosting
# # ---------------------------

# _COMMON_STOP = {
#     "what","does","this","function","class","explain","describe","how","to","where",
#     "the","a","an","is","do","of","in","on","for","with","and","or","by","it","that"
# }

# def _extract_symbol_tokens(q: str) -> List[str]:
#     q = q or ""
#     quoted = re.findall(r"[`\"]([A-Za-z_][A-Za-z0-9_]*)[`\"]", q)
#     bare = re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", q)
#     out, seen = [], set()
#     for t in quoted + bare:
#         tl = t.lower()
#         if tl in _COMMON_STOP:
#             continue
#         if tl not in seen:
#             seen.add(tl); out.append(t)
#     return out


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
#         embs = _use_helper_build_embeddings(texts, embedder, normalize=normalize)

#         ids = np.arange(len(rows), dtype=np.int64)
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
#     metric = indices.get("metric", "cosine")
#     out: Dict[str, List[Tuple[str, float]]] = {"intent": [], "impl": []}
#     for view_name in ("intent", "impl"):
#         view = indices.get("views", {}).get(view_name) or {}
#         index = view.get("index")
#         rows = view.get("rows") or []
#         if index is None or not rows:
#             continue
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
#     internal_only: Optional[bool] = None,
#     # fusion
#     rrf_k: float = 60.0,
# ) -> List[Dict[str, Any]]:
#     if not isinstance(query, str) or not query.strip():
#         raise ValueError("hybrid_retrieve_two_view: 'query' must be a non-empty string.")

#     metric = indices.get("metric", "cosine")
#     views = indices.get("views", {})
#     intent_v = views.get("intent") or {}
#     impl_v   = views.get("impl") or {}

#     # -------- 0) symbol tokens (used for lexical + boosting)
#     sym_tokens = _extract_symbol_tokens(query)

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

#     _maybe_attach_views(sem_lists.get("intent", []), sem_lists.get("impl", []))

#     # ---------- 2) Lexical over chunks (tokenized) ----------
#     lex_counts: Dict[str, int] = {}
#     if use_lexical and chunks:
#         # prefer project lexical fn; otherwise token-aware fallback
#         search_fn = lexical_search_fn or _lexical_search_default
#         if search_fn is None:
#             # Build a token-aware regex (prefer symbol tokens; else content tokens)
#             tokens = sym_tokens[:]
#             if not tokens:
#                 tokens = [t for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", query) if t.lower() not in _COMMON_STOP]
#             # no tokens -> skip lexical
#             if tokens:
#                 # \btoken\b OR join, unless whole_word False
#                 def _mk_rx(tok: str, whole: bool) -> str:
#                     return (r"\b" + re.escape(tok) + r"\b") if whole else re.escape(tok)
#                 pat = r"(" + "|".join(_mk_rx(t, bool(lex_whole_word)) for t in tokens) + r")"
#                 rx = re.compile(pat, 0 if bool(lex_case_sensitive) else re.IGNORECASE)

#                 def _field_get(d: Dict[str, Any], dotted: str) -> str:
#                     cur: Any = d
#                     for p in dotted.split("."):
#                         cur = cur.get(p, "") if isinstance(cur, dict) else ""
#                     return str(cur or "")

#                 for ch in chunks:
#                     cid = str(ch.get("id") or "")
#                     if not cid:
#                         continue
#                     total = 0
#                     for f in tuple(lex_fields):
#                         s = _field_get(ch, f)
#                         if not s:
#                             continue
#                         total += len(list(rx.finditer(s)))
#                     if total:
#                         lex_counts[cid] = total
#         else:
#             try:
#                 lex = search_fn(
#                     query, chunks,
#                     fields=tuple(lex_fields),
#                     regex=bool(lex_regex),
#                     case_sensitive=bool(lex_case_sensitive),
#                     whole_word=bool(lex_whole_word),
#                 )
#                 for r in lex:
#                     cid = r.get("chunk", {}).get("id")
#                     if cid:
#                         lex_counts[str(cid)] = int(r.get("total_matches", 0))
#             except Exception as e:
#                 logger.warning("Lexical search failed; continuing without lexical. (%s)", e)

#     # ---------- 3) Graph expansion (seed with ids AND files) ----------
#     graph_depths: Dict[str, int] = {}
#     if G is not None:
#         from collections import deque
#         seeds = set()
#         seeds.update([cid for cid, _ in sem_lists.get("intent", [])])
#         seeds.update([cid for cid, _ in sem_lists.get("impl", [])])
#         seeds.update(lex_counts.keys())

#         # also seed with file paths for each known seed (helps graphs keyed by file nodes)
#         id_to_file: Dict[str, str] = {}
#         if chunks:
#             for ch in chunks:
#                 cid = str(ch.get("id") or "")
#                 fp = str(ch.get("file") or "")
#                 if cid and fp:
#                     id_to_file[cid] = fp
#         for s in list(seeds):
#             fp = id_to_file.get(str(s))
#             if fp:
#                 seeds.add(fp)

#         try:
#             import networkx as nx  # type: ignore
#             _is_multi = isinstance(G, nx.MultiDiGraph)
#         except Exception:
#             _is_multi = False

#         Gq = G
#         if _is_multi:
#             try:
#                 from cgx.graph.build_graph import project_graph_for_visualization  # type: ignore
#                 Gq = project_graph_for_visualization(G)  # DiGraph
#             except Exception:
#                 Gq = G

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
#             qd = deque([(start, 0)])
#             visited = {start}
#             while qd:
#                 nid, d = qd.popleft()
#                 if d >= int(neighbor_depth):
#                     continue
#                 for nbr in list(Gq.successors(nid)) + list(Gq.predecessors(nid)):
#                     if nbr in visited:
#                         continue
#                     try:
#                         ed = Gq[nid][nbr] if Gq.has_edge(nid, nbr) else Gq[nbr][nid]
#                         attrs = ed if isinstance(ed, dict) and not any(isinstance(v, dict) for v in ed.values()) \
#                                 else (list(ed.values())[0] if isinstance(ed, dict) and ed else {})
#                     except Exception:
#                         attrs = {}
#                     if _edge_ok(attrs):
#                         visited.add(nbr)
#                         qd.append((nbr, d + 1))
#                         if ("::" in str(nbr)) or str(nbr).endswith(".py"):
#                             if str(nbr) not in graph_depths:
#                                 graph_depths[str(nbr)] = d + 1

#     # ---------- 4) Symbol-first boosting ----------
#     symbol_boost: Dict[str, int] = {}
#     if sym_tokens:
#         # scan both views for exact symbol name or cid suffix match
#         for view in ("intent", "impl"):
#             vw = (indices.get("views") or {}).get(view) or {}
#             for r in (vw.get("rows") or []):
#                 cid = str(r.get("chunk_id") or "")
#                 name = str(r.get("name") or "")
#                 nm_l = name.lower()
#                 for t in sym_tokens:
#                     tl = t.lower()
#                     if (f"::{t}" in cid) or (f"::{tl}" in cid.lower()) or (nm_l == tl):
#                         symbol_boost[cid] = max(symbol_boost.get(cid, 0), 2)  # strong discrete boost

#     # ---------- 5) RRF fusion + additive boosts ----------
#     intent_list = sem_lists.get("intent", [])
#     impl_list   = sem_lists.get("impl", [])
#     fused = _rrf_fuse([intent_list, impl_list], k=float(rrf_k))

#     # lexical term (rank-only via RRF on counts)
#     if lex_counts:
#         lex_ids_sorted = sorted(lex_counts.items(), key=lambda kv: (-kv[1], kv[0]))
#         lex_list = [(cid, float(cnt)) for cid, cnt in lex_ids_sorted]
#         fused_lex = _rrf_fuse([lex_list], k=float(rrf_k))
#         for cid, fs in fused_lex.items():
#             fused[cid] = fused.get(cid, 0.0) + fs

#     # graph proximity (rank-only)
#     if graph_depths and neighbor_depth > 0:
#         graph_ranked = sorted(graph_depths.items(), key=lambda kv: (kv[1], kv[0]))
#         graph_list = [(cid, float(-depth)) for cid, depth in graph_ranked]
#         fused_graph = _rrf_fuse([graph_list], k=float(rrf_k))
#         for cid, fs in fused_graph.items():
#             fused[cid] = fused.get(cid, 0.0) + fs

#     # additive symbol boost (explicit, so it really bubbles up)
#     if symbol_boost:
#         for cid, times in symbol_boost.items():
#             fused[cid] = fused.get(cid, 0.0) + 1.0 * float(times)

#     # ---------- 6) Output with provenance ----------
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
#             "symbol_match": bool(cid in symbol_boost),
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
#     centrality_bonus: float = 0.15,
# ) -> List[Dict[str, Any]]:
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
#         total = sum(sc for _, sc in pairs)
#         deg = 0
#         for cid, _ in pairs:
#             rr = rec_map.get(cid) or {}
#             deg += int(rr.get("calls_in_count", 0)) + int(rr.get("calls_out_count", 0))
#         bonus = centrality_bonus * math.log1p(deg)
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
#     rec_map = _record_map(records)

#     exemplar_ids = [str(h.get("chunk_id")) for h in fused_hits[: int(k_exemplars)]]
#     sigs = _collect_exemplar_signals(exemplar_ids, records)

#     file_scores: Dict[str, float] = {}
#     class_scores: Dict[str, float] = {}

#     children_by_container: Dict[str, List[str]] = {}

#     for r in records:
#         rtype = r.get("type")
#         rid = r.get("id")
#         if not rid:
#             continue

#         if rtype == "file":
#             s_imp = _jaccard(r.get("imports_used", []), sigs["imports"])
#             s_att = _jaccard(r.get("attributes_used_root_reads", []), sigs["attributes"])
#             score = 0.55 * s_imp + 0.45 * s_att
#             file_scores[rid] = score
#             children_by_container[rid] = list(r.get("defines_children_ids") or [])

#         elif rtype == "class":
#             s_imp = _jaccard(r.get("imports_used", []), sigs["imports"])
#             s_att = _jaccard(r.get("attributes_used_root_reads", []), sigs["attributes"])
#             score = 0.55 * s_imp + 0.45 * s_att
#             class_scores[rid] = score
#             children_by_container[rid] = list(r.get("defines_children_ids") or [])

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

#     out.sort(key=lambda d: (-d["score"], d["container_type"], d["container_id"]))
#     return out
from __future__ import annotations

"""
Two-view retrieval orchestrator (CLEANED & FIXED).

Purpose
-------
Provide deterministic, auditable retrieval by orchestrating:
  - ANN indices for both views ("intent", "impl") from S4 records/corpus.
  - Semantic retrieval via TwoViewIndex.
  - Lexical retrieval via LexicalIndex (BM25-lite).
  - Regex lexical fallback over chunks if no LexicalIndex available.
  - Hybrid fusion via HybridRetriever (ANN + lexical + graph).
  - Aggregation of results to files/classes.
  - Suggestions for insertion points using overlap signals.

This module DOES NOT:
  - Reimplement FAISS building or semantic search (delegates to ann_numpy/index/hybrid).
  - Reimplement lexical search (delegates to lexical).
  - Reimplement RRF (delegates to rrf).
"""

import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from cgx.logging_setup import get_logger
logger = get_logger("orchestration")

# canonical imports
from cgx.retrieval.ann_numpy import build_ann_index
from cgx.retrieval.index import TwoViewIndex, ViewSlice
from cgx.retrieval.lexical import LexicalIndex
from cgx.retrieval.hybrid import HybridRetriever, HybridConfig


# ---------------------------
# Public API
# ---------------------------

__all__ = [
    "build_two_view_indices",
    "hybrid_retrieve_two_view",
    "aggregate_by_file",
    "aggregate_by_class",
    "suggest_insertion_points",
]


# ---------------------------
# Index building
# ---------------------------

def build_two_view_indices(
    records: List[Dict[str, Any]],
    *,
    embedder: Any,
    metric: str = "cosine",
    index_type: str = "flat",
) -> TwoViewIndex:
    """
    Build a TwoViewIndex from parsed records (intent + impl views).
    """
    return TwoViewIndex.from_records(
        records,
        embedder=embedder,
        index_builder=build_ann_index,
        metric=metric,
        index_type=index_type,
    )


# ---------------------------
# Hybrid retrieval (delegate)
# ---------------------------

def _two_view_index_from_dict(
    indices: Any,
    records: Optional[List[Dict[str, Any]]] = None,
    *,
    embedder: Any = None,
) -> TwoViewIndex:
    """
    Shim: backward compat.

    If given a TwoViewIndex, return it.
    If given a dict (loaded from disk), wrap its FAISS indices + rows
    into a TwoViewIndex without re-embedding.
    """
    if isinstance(indices, TwoViewIndex):
        return indices
    if not isinstance(indices, dict):
        raise TypeError(f"indices must be dict or TwoViewIndex, got {type(indices)}")

    views = indices.get("views", {})
    intent = views.get("intent") or {}
    impl   = views.get("impl") or {}

    if not (intent.get("rows") or impl.get("rows")):
        raise ValueError("_two_view_index_from_dict: no rows available in indices dict.")

    tv = TwoViewIndex()

    if intent.get("rows") and intent.get("index") is not None:
        tv._intent = ViewSlice(
            rows=intent["rows"],
            ids=np.arange(len(intent["rows"]), dtype=np.int64),
            index=intent["index"],
            meta=intent.get("meta") or {},
            dim=None,
        )

    if impl.get("rows") and impl.get("index") is not None:
        tv._impl = ViewSlice(
            rows=impl["rows"],
            ids=np.arange(len(impl["rows"]), dtype=np.int64),
            index=impl["index"],
            meta=impl.get("meta") or {},
            dim=None,
        )

    # Build chunk->rows mapping
    tv._chunk_id_to_rows = {}
    for view_name, vs in (("intent", tv._intent), ("impl", tv._impl)):
        if vs is None:
            continue
        for ridx, row in enumerate(vs.rows):
            cid = row.get("chunk_id")
            if isinstance(cid, str):
                tv._chunk_id_to_rows.setdefault(cid, {}).setdefault(view_name, []).append(ridx)

    logger.info(
        "_two_view_index_from_dict: wrapped FAISS indices into TwoViewIndex "
        "(intent_rows=%d, impl_rows=%d)",
        len(intent.get("rows") or []),
        len(impl.get("rows") or []),
    )

    return tv


def hybrid_retrieve_two_view(
    query: str,
    *,
    indices: Optional[Any] = None,
    tv_index: Optional[TwoViewIndex] = None,
    records: Optional[List[Dict[str, Any]]] = None,
    lexical_index: Optional[LexicalIndex] = None,
    embedder: Any = None,
    chunks: Optional[List[Dict[str, Any]]] = None,
    G: Any = None,
    top_k_per_view: int = 10,
    use_lexical: bool = True,
    rrf_k: float = 60.0,
    top_k_total: int = 30,
    expand_top_n: int = 1,
    graph_depth: int = 1,
    neighbor_depth: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run hybrid retrieval (semantic + lexical + optional graph expansion).
    """
    if neighbor_depth is not None:
        graph_depth = neighbor_depth

    if indices is not None and tv_index is None:
        tv_index = _two_view_index_from_dict(indices, records=records, embedder=embedder)
        logger.info(
            "hybrid_retrieve_two_view: tv_index built=%s (records=%s)",
            type(tv_index),
            f"list[{len(records)}]" if isinstance(records, list) else type(records),
        )

    if tv_index is None:
        raise RuntimeError("hybrid_retrieve_two_view: tv_index is None. Ensure indices/records are valid.")

    if use_lexical and lexical_index is None and records:
        lexical_index = LexicalIndex.from_records(records)
        logger.info(
            "hybrid_retrieve_two_view: lexical_index built=%s (docs=%d)",
            type(lexical_index),
            getattr(lexical_index, "N", -1),
        )

    cfg = HybridConfig(
        k_intent=top_k_per_view,
        k_impl=top_k_per_view,
        k_lex=top_k_per_view if use_lexical else 0,
        expand_top_n=expand_top_n,
        graph_depth=graph_depth,
        rrf_k=rrf_k,
        top_k_chunks=top_k_total,
        top_k_files=10,
        top_k_classes=10,
    )

    retriever = HybridRetriever(
        tv_index=tv_index,
        records=records or [],
        lexical_index=lexical_index,
        chunks=chunks,
        G=G,
    )
    return retriever.search(query, embedder=embedder, cfg=cfg)


# ---------------------------
# Aggregation helpers
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
        deg = sum(
            int(rec_map.get(cid, {}).get("calls_in_count", 0))
            + int(rec_map.get(cid, {}).get("calls_out_count", 0))
            for cid, _ in pairs
        )
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
# Suggestion logic
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
        imports.update([imp for imp in r.get("imports_used") or [] if isinstance(imp, str)])
        attrs.update([ar for ar in r.get("attributes_used_root_reads") or [] if isinstance(ar, str)])
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
    return float(len(A & B)) / max(1, len(A | B))


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
        rid = r.get("id")
        if not rid:
            continue
        rtype = r.get("type")
        if rtype == "file":
            s_imp = _jaccard(r.get("imports_used", []), sigs["imports"])
            s_att = _jaccard(r.get("attributes_used_root_reads", []), sigs["attributes"])
            file_scores[rid] = 0.55 * s_imp + 0.45 * s_att
            children_by_container[rid] = list(r.get("defines_children_ids") or [])
        elif rtype == "class":
            s_imp = _jaccard(r.get("imports_used", []), sigs["imports"])
            s_att = _jaccard(r.get("attributes_used_root_reads", []), sigs["attributes"])
            class_scores[rid] = 0.55 * s_imp + 0.45 * s_att
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
                best_id = cid
                best_deg = deg
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
