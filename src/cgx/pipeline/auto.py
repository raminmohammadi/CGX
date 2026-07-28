

from __future__ import annotations
from cgx.logging_setup import get_logger
logger = get_logger(__name__)

"""
Auto-wired pipeline that exercises the canonical components:
- parse -> graph -> records -> two-view corpus
- embeddings.build.build_embeddings
- embeddings.index.build_faiss_index
- retrieval.orchestrator.hybrid_retrieve_two_view
- retrieval.orchestrator.aggregate_by_file/aggregate_by_class
- retrieval.orchestrator.suggest_insertion_points
- retrieval.orchestrator.analyze_change_impact  (NEW)

Adds graph + chunks persistence and loading so hybrid retrieval can
actually use lexical and graph expansion at query time.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import os
import json

from cgx.parser.parse_codebase import parse_codebase
from cgx.graph.build_graph import build_knowledge_graph
from cgx.embeddings.records import make_index_records, prepare_embedding_corpus
from cgx.embeddings.build import build_embeddings
from cgx.embeddings.index import build_faiss_index
from cgx.retrieval.orchestrator import (
    hybrid_retrieve_two_view,
    aggregate_by_file,
    aggregate_by_class,
    suggest_insertion_points,
    analyze_change_impact,  # NEW
)
from cgx.io.persist import save_indices, load_indices, save_jsonl, load_jsonl
from cgx.retrieval.lexical import get_cached_lexical_index
from cgx.answer.scope import apply_scope_penalty
from cgx.trace import traced

# Graph persistence
from networkx.readwrite import json_graph


def _now_iso() -> str:
    """Local, timezone-aware completion timestamp for the index manifest."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _ensure_tuple_parse(res):
    """Support both parse_codebase(project_root)->chunks and ->(chunks, calls)."""
    if isinstance(res, tuple) and len(res) >= 1:
        chunks = res[0]
        calls = res[1] if len(res) >= 2 else None
        return chunks, calls
    return res, None


def _to_faiss_ids(rows: List[Dict[str, Any]]) -> np.ndarray:
    """Return a dense 0..N-1 int64 id array for FAISS; keep original ids in meta/rows."""
    return np.arange(len(rows), dtype=np.int64)


def _encode_with_embedder(embedder: Any, rows: List[Dict[str, Any]], *, normalize: bool) -> np.ndarray:
    """Encode rows with a user-provided embedder exposing .encode(list[str])."""
    texts = [str(r.get("text") or r.get("code") or "") for r in rows]
    embs = np.asarray(embedder.encode(texts), dtype=np.float32)
    if normalize:
        denom = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
        embs = (embs / denom).astype(np.float32)
    return embs


class IndexBuildCancelled(Exception):
    """Raised inside :func:`run_index_auto` when a caller flips ``cancel_event``.

    The build polls the event at stage boundaries (parse / graph / embed /
    persist) so a Ctrl-C in the dashboard stops promptly and -- crucially --
    *before* any index files are written, so a cancelled build never leaves a
    half-written ``.cgx/index`` that later looks "ready".
    """


def run_index_auto(
    project_root: str,
    out_dir: str,
    metric: str = "cosine",
    index_type: str = "flat",
    which: Tuple[str, str] = ("intent", "impl"),
    model_name: str = "jinaai/jina-embeddings-v2-base-code",
    batch_size: int = 64,
    embedder: Optional[Any] = None,
    incremental: bool = True,
    cancel_event: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Build two-view FAISS indices and persist alongside records and graph/chunks.

    Parameters
    ----------
    embedder
        Optional object exposing ``.encode(list[str]) -> np.ndarray``. When
        provided it takes precedence over ``model_name`` and bypasses
        ``build_embeddings`` (useful for BYO encoders / tests / mocks).
    cancel_event
        Optional ``threading.Event``; when set, the build raises
        :class:`IndexBuildCancelled` at the next stage boundary.
    """
    os.makedirs(out_dir, exist_ok=True)

    def _ck() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise IndexBuildCancelled("index build cancelled")

    logger.info("=== run_index_auto starting ===")
    logger.info("project_root=%s out_dir=%s metric=%s index_type=%s model=%s",
                project_root, out_dir, metric, index_type, model_name)

    # ---------------- Parse & Graph ----------------
    # When incremental, reuse an on-disk parse cache so unchanged files are not
    # re-parsed; the graph is then rebuilt from the merged (mostly cached)
    # chunk set. A full parse is byte-for-byte equivalent to the incremental
    # result for the same tree, so disabling the flag stays a pure fallback.
    parse_stats: Dict[str, int] = {}
    if incremental:
        from cgx.parser.incremental import incremental_parse_codebase
        cache_path = os.path.join(out_dir, "parse_cache.json")
        chunks, calls, parse_stats = incremental_parse_codebase(
            project_root, cache_path=cache_path,
        )
    else:
        chunks, calls = _ensure_tuple_parse(parse_codebase(project_root))
    logger.info("Parsed codebase: %d chunks, %d calls", len(chunks), len(calls or []))
    _ck()

    G = build_knowledge_graph(chunks, calls)
    logger.info("Knowledge graph built: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    _ck()

    # ---------------- Records + Corpus ----------------
    records = make_index_records(chunks, G)
    logger.info("make_index_records produced %d records", len(records))
    _ck()

    try:
        corpus = prepare_embedding_corpus(records, which=which)
        if corpus is None:
            logger.error("prepare_embedding_corpus returned None (records=%d)", len(records))
            raise RuntimeError("prepare_embedding_corpus returned None")
        logger.info("prepare_embedding_corpus produced %d rows across views=%s", len(corpus), which)
    except Exception as e:
        logger.error("prepare_embedding_corpus failed: %s", e, exc_info=True)
        raise

    # ---------------- Split corpus per view ----------------
    per_view: Dict[str, List[Dict[str, Any]]] = {"intent": [], "impl": []}
    for row in corpus:
        vw = row.get("view")
        if vw in per_view:
            per_view[vw].append(row)

    for v in per_view:
        logger.info("View %s has %d rows", v, len(per_view[v]))

    # ---------------- Embeddings + FAISS (views built in parallel) ----------------
    from cgx.embeddings.cache import embed_with_cache
    indices: Dict[str, Any] = {"views": {}, "metric": metric}
    cache_stats_per_view: Dict[str, Dict[str, int]] = {}
    normalize = (metric in {"cosine", "ip"})

    def _make_progress_cb(view_name: str):
        """A throttled (done, total) callback that logs embedding progress.

        Emits a line every ~5% (and always at 100%) so a long-running embed
        of a large repo shows steady progress instead of going silent.
        """
        state = {"last_pct": -1}

        def _cb(done: int, total: int) -> None:
            if total <= 0:
                return
            pct = int(done * 100 / total)
            if done >= total or pct >= state["last_pct"] + 5:
                state["last_pct"] = pct
                logger.info("Embedding view=%s progress %d/%d (%d%%)",
                            view_name, done, total, pct)

        return _cb

    def _build_view(view_name: str, rows: List[Dict[str, Any]]):
        """Embed + index one corpus view; returns (view_name, index_dict, cache_stats)."""
        logger.info("Building index for view=%s (rows=%d)", view_name, len(rows))
        _ck()
        if not rows:
            logger.warning("No rows for view=%s, skipping", view_name)
            return view_name, {
                "index": None, "meta": None, "rows": [],
                "ids": np.array([], dtype=np.int64),
            }, {}

        try:
            if incremental:
                cache_path = os.path.join(out_dir, f"emb_cache_{view_name}.npz")
                texts = [str(r.get("text") or r.get("code") or "") for r in rows]

                progress_cb = _make_progress_cb(view_name)

                def _encode(missing_texts: List[str]) -> np.ndarray:
                    if embedder is not None and hasattr(embedder, "encode"):
                        arr = np.asarray(embedder.encode(missing_texts), dtype=np.float32)
                        if normalize and arr.size:
                            denom = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
                            arr = (arr / denom).astype(np.float32)
                        return arr
                    return build_embeddings(
                        [{"text": t} for t in missing_texts],
                        model_name=model_name,
                        backend="auto",
                        normalize=normalize,
                        batch_size=batch_size,
                        field_strategy="auto",
                        max_length=256,
                        progress_cb=progress_cb,
                    )

                embs, stats = embed_with_cache(
                    texts, encode_fn=_encode, cache_path=cache_path,
                    model_name=model_name, normalize=normalize,
                )
                logger.info("Embedding cache view=%s hits=%d misses=%d",
                            view_name, stats["hits"], stats["misses"])
            else:
                stats = {}
                if embedder is not None and hasattr(embedder, "encode"):
                    embs = _encode_with_embedder(embedder, rows, normalize=normalize)
                else:
                    embs = build_embeddings(
                        rows, model_name=model_name, backend="auto",
                        normalize=normalize, batch_size=batch_size,
                        field_strategy="auto", max_length=256,
                        progress_cb=_make_progress_cb(view_name),
                    )
            logger.info("Embedding done for view=%s shape=%s", view_name, np.asarray(embs).shape)
        except Exception as e:
            logger.error("Embedding failed for view=%s: %s", view_name, e, exc_info=True)
            raise

        embs = np.asarray(embs, dtype=np.float32)
        try:
            index, meta = build_faiss_index(embs, metric=metric, index=index_type, return_meta=True)
        except Exception as e:
            logger.error("build_faiss_index failed for view=%s: %s", view_name, e, exc_info=True)
            raise

        orig_ids = [r.get("chunk_id") for r in rows]
        meta = {**(meta or {}), "orig_chunk_ids": orig_ids}
        logger.info("Finished view=%s: index type=%s rows=%d", view_name, type(index).__name__, len(rows))
        return view_name, {
            "index": index, "meta": meta,
            "rows": rows, "ids": _to_faiss_ids(rows),
        }, stats

    # Run both views concurrently (embedding is the bottleneck; FAISS build is CPU-bound).
    # ThreadPoolExecutor is appropriate here: torch releases the GIL during GPU ops, and
    # FAISS C++ calls also release the GIL, so true parallelism occurs for both.
    logger.info("Building embeddings + FAISS for %d views in parallel", len(per_view))
    with ThreadPoolExecutor(max_workers=len(per_view)) as pool:
        futures = {pool.submit(_build_view, vn, rows): vn for vn, rows in per_view.items()}
        for fut in as_completed(futures):
            view_name = futures[fut]
            try:
                vn, view_dict, vstats = fut.result()
                indices["views"][vn] = view_dict
                if vstats:
                    cache_stats_per_view[vn] = vstats
            except IndexBuildCancelled:
                # Cancellation is expected control flow, not a build failure:
                # propagate quietly so the caller can stop without a scary trace.
                raise
            except Exception as e:
                logger.error("View %s failed: %s", view_name, e, exc_info=True)
                raise

    # A Ctrl-C between the last view finishing and the writes below must still
    # abort before any index file is persisted, so a cancelled build never
    # leaves a half-written index dir that later looks "ready".
    _ck()

    # Stamp provenance onto the index dict so save_indices records which model
    # built it (and when / for which project). A later query reads this back to
    # pick the matching embedder and to catch a dim mismatch up front.
    indexed_at = _now_iso()
    embed_model = getattr(embedder, "model_name", None) or model_name
    indices["embed_model"] = embed_model
    indices["index_type"] = index_type
    indices["project_root"] = os.path.abspath(project_root)
    indices["indexed_at"] = indexed_at
    indices["counts"] = {k: len(v) for k, v in per_view.items()}

    # ---------------- Persist ----------------
    try:
        save_indices(indices, os.path.join(out_dir, "indices"))
        logger.info("Saved indices -> %s/indices", out_dir)
    except Exception as e:
        logger.error("save_indices failed: %s", e, exc_info=True)
        raise

    try:
        save_jsonl(records, os.path.join(out_dir, "records.jsonl"))
        logger.info("Saved records.jsonl (%d records)", len(records))
    except Exception as e:
        logger.error("save_jsonl(records) failed: %s", e, exc_info=True)
        raise

    try:
        save_jsonl(chunks, os.path.join(out_dir, "chunks.jsonl"))
        logger.info("Saved chunks.jsonl (%d chunks)", len(chunks))
    except Exception as e:
        logger.error("save_jsonl(chunks) failed: %s", e, exc_info=True)
        raise

    graph_path = os.path.join(out_dir, "graph.json")
    try:
        with open(graph_path, "w", encoding="utf-8") as f:
            json.dump(json_graph.node_link_data(G), f)
        logger.info("Saved graph.json (nodes=%d edges=%d)", G.number_of_nodes(), G.number_of_edges())
    except Exception as e:
        logger.warning("Graph serialization failed, continuing: %s", e, exc_info=True)
        graph_path = None

    # Hierarchical repo map: cheap whole-repo context for the Planner, derived
    # from the intent-side record summaries and cached (fingerprint-keyed) so a
    # later load skips the rebuild when records are unchanged.
    repo_map_path = os.path.join(out_dir, "repo_map.json")
    try:
        from cgx.answer.repo_map import build_or_load_repo_map
        rmap = build_or_load_repo_map(records, cache_path=repo_map_path)
        logger.info("Saved repo_map.json (%s)", rmap.get("stats"))
    except Exception as e:
        logger.warning("Repo map build failed, continuing: %s", e, exc_info=True)
        repo_map_path = None

    result = {
        "counts": {k: len(v) for k, v in per_view.items()},
        "embed_model": embed_model,
        "indexed_at": indexed_at,
        "out": {
            "indices": os.path.join(out_dir, "indices"),
            "records": os.path.join(out_dir, "records.jsonl"),
            "chunks": os.path.join(out_dir, "chunks.jsonl"),
            "graph": graph_path,
            "repo_map": repo_map_path,
        },
        "incremental": bool(incremental),
        "embedding_cache": cache_stats_per_view,
        "parse": parse_stats,
    }
    logger.info("=== run_index_auto completed === %s", result["counts"])
    return result


# ---------------------------
# Query wrapper (ALL SIGNALS + IMPACT)
# ---------------------------

@traced("pipeline")
def run_query_auto(
    index_dir: str,
    records_path: str,
    query: str,
    *,
    model_name: str = "jinaai/jina-embeddings-v2-base-code",
    chunks_path: Optional[str] = None,
    graph_path: Optional[str] = None,
    top_k_per_view: int = 10,
    neighbor_depth: int = 1,
    use_lexical: bool = True,   # retained for API compatibility; hybrid ignores and uses lexical anyway
    single_view: Optional[str] = None,
    embedder: Optional[Any] = None,
    enable_reranker: Optional[bool] = None,
    scope: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load indices, records, chunks, and graph, then execute hybrid two-view retrieval
    using the same embedding pipeline as indexing. ALWAYS uses semantic+lexical+graph
    and fuses with RRF. Also returns impact analysis for change-style queries.
    """

    class BuildEmbedder:
        """Wrapper so build_embeddings can be used consistently at query time."""
        def __init__(self, model_name: str, batch_size: int = 64, normalize: bool = True, max_length: int = 256):
            self.model_name = model_name
            self.batch_size = batch_size
            self.normalize = normalize
            self.max_length = max_length
        def encode(self, texts: List[str]) -> np.ndarray:
            rows = [{"text": t} for t in texts]
            return build_embeddings(
                rows,
                model_name=self.model_name,
                backend="auto",
                normalize=self.normalize,
                batch_size=self.batch_size,
                field_strategy="auto",
                max_length=self.max_length,
            )

    indices = load_indices(index_dir)
    records = load_jsonl(records_path)
    chunks = load_jsonl(chunks_path) if chunks_path else None

    # Load graph if available. networkx>=3.4 changed the default edges key
    # in node_link_data from "links" to "edges"; detect whichever the saved
    # file actually uses so the loader is forward- and backward-compatible.
    G = None
    if graph_path:
        try:
            with open(graph_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            edges_key = "edges" if isinstance(data, dict) and "edges" in data else "links"
            G = json_graph.node_link_graph(data, edges=edges_key)
        except Exception as e:
            logger.warning("run_query_auto: failed to load graph (%s)", e)
            G = None

    if embedder is None or not hasattr(embedder, "encode"):
        # The index manifest records which model built it. When no explicit
        # embedder is supplied, trust that model over the caller's default so a
        # query never embeds with the wrong dim (the 768-vs-1024 crash). A BYO
        # embedder is respected as-is; the caller owns that choice.
        stored_model = indices.get("embed_model")
        if stored_model and stored_model != model_name:
            logger.info(
                "run_query_auto: overriding embed model %r -> %r (from index meta)",
                model_name, stored_model,
            )
            model_name = stored_model
        embedder = BuildEmbedder(model_name=model_name, batch_size=64, normalize=True)

    # Reuse a path-keyed LexicalIndex across queries to avoid rebuilding BM25
    # on every call (build is O(N) over all records).
    lex_idx = get_cached_lexical_index(records_path, records) if records else None

    # Hybrid retrieval (semantic+lexical+graph → RRF)
    retrieval_out = hybrid_retrieve_two_view(
        query,
        indices=indices,
        records=records,
        embedder=embedder,
        chunks=chunks,
        G=G,
        top_k_per_view=top_k_per_view,
        neighbor_depth=neighbor_depth,
        use_lexical=True,  # forced on
        lexical_index=lex_idx,
        enable_reranker=enable_reranker,
    )

    hits = retrieval_out.get("hits", [])
    if scope in {"src", "tests"}:
        before = len(hits)
        hits = apply_scope_penalty(hits, scope)
        logger.info("run_query_auto: scope=%s applied soft penalty (n=%d)", scope, before)
    top_files = retrieval_out.get("top_files", [])
    top_classes = retrieval_out.get("top_classes", [])

    # Impact analysis (NEW): even if the question isn't explicitly "which files...",
    # we compute it and let the UI/agent decide how to present it.
    impact = analyze_change_impact(query, hits, records, G)

    # Optional: insertion anchors (non-critical)
    try:
        anchors = suggest_insertion_points(query, hits, records, G=G, embedder=embedder)
    except Exception as e:
        logger.warning("suggest_insertion_points failed: %s", e)
        anchors = []

    out: Dict[str, Any] = {
        "hits": hits,
        "top_files": top_files,
        "top_classes": top_classes,
        "impact": impact,       # <= use this for "which files will be impacted?"
        "anchors": anchors,
        "debug": {
            "graph_used": bool(G is not None),
            "chunks_available": bool(chunks),
            "top_k_per_view": top_k_per_view,
            "neighbor_depth": neighbor_depth,
            "lexical_forced": True,
            "semantic_views": ["intent", "impl"],
            "fusion": "RRF",
        },
    }

    if single_view in {"intent", "impl"}:
        try:
            from cgx.embeddings.search import semantic_search
            view = indices.get("views", {}).get(single_view, {})
            index = view.get("index")
            rows = view.get("rows", [])
            out["single_view"] = {
                "view": single_view,
                "results": semantic_search(query, embedder, index, rows, top_k=top_k_per_view),
            }
        except Exception as e:
            out["single_view_error"] = f"{type(e).__name__}: {e}"

    return out
