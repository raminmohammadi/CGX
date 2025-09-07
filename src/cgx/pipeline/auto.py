from __future__ import annotations
from src.cgx.logging_setup import get_logger
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

# Graph persistence
from networkx.readwrite import json_graph


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


def run_index_auto(
    project_root: str,
    out_dir: str,
    metric: str = "cosine",
    index_type: str = "flat",
    which: Tuple[str, str] = ("intent", "impl"),
    model_name: str = "jinaai/jina-embeddings-v2-base-code",
    batch_size: int = 64,
) -> Dict[str, Any]:
    """
    Build two-view FAISS indices and persist alongside records and graph/chunks.
    """
    os.makedirs(out_dir, exist_ok=True)
    logger.info("=== run_index_auto starting ===")
    logger.info("project_root=%s out_dir=%s metric=%s index_type=%s model=%s",
                project_root, out_dir, metric, index_type, model_name)

    # ---------------- Parse & Graph ----------------
    chunks, calls = _ensure_tuple_parse(parse_codebase(project_root))
    logger.info("Parsed codebase: %d chunks, %d calls", len(chunks), len(calls or []))

    G = build_knowledge_graph(chunks, calls)
    logger.info("Knowledge graph built: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    # ---------------- Records + Corpus ----------------
    records = make_index_records(chunks, G)
    logger.info("make_index_records produced %d records", len(records))

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

    # ---------------- Embeddings + FAISS ----------------
    indices: Dict[str, Any] = {"views": {}, "metric": metric}
    for view_name, rows in per_view.items():
        logger.info("Building index for view=%s (rows=%d)", view_name, len(rows))
        if not rows:
            logger.warning("No rows for view=%s, skipping", view_name)
            indices["views"][view_name] = {
                "index": None,
                "meta": None,
                "rows": [],
                "ids": np.array([], dtype=np.int64),
            }
            continue

        # Embed
        try:
            embs = build_embeddings(
                rows,
                model_name=model_name,
                backend="auto",
                normalize=(metric in {"cosine", "ip"}),
                batch_size=batch_size,
                field_strategy="auto",
                max_length=256,
            )
        except Exception as e:
            logger.error("Embedding failed for view=%s: %s", view_name, e, exc_info=True)
            raise

        embs = np.asarray(embs, dtype=np.float32)

        # Build FAISS index
        try:
            index, meta = build_faiss_index(
                embs,
                metric=metric,
                index=index_type,
                return_meta=True,
            )
        except Exception as e:
            logger.error("build_faiss_index failed for view=%s: %s", view_name, e, exc_info=True)
            raise

        # Preserve chunk IDs
        orig_ids = [r.get("chunk_id") for r in rows]
        meta = {**(meta or {}), "orig_chunk_ids": orig_ids}

        indices["views"][view_name] = {
            "index": index,
            "meta": meta,
            "rows": rows,
            "ids": _to_faiss_ids(rows),
        }
        logger.info("Finished view=%s: index type=%s rows=%d", view_name, type(index).__name__, len(rows))

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

    result = {
        "counts": {k: len(v) for k, v in per_view.items()},
        "out": {
            "indices": os.path.join(out_dir, "indices"),
            "records": os.path.join(out_dir, "records.jsonl"),
            "chunks": os.path.join(out_dir, "chunks.jsonl"),
            "graph": graph_path,
        },
    }
    logger.info("=== run_index_auto completed === %s", result["counts"])
    return result


# ---------------------------
# Query wrapper (ALL SIGNALS + IMPACT)
# ---------------------------

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

    # Load graph if available
    G = None
    if graph_path:
        try:
            with open(graph_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            G = json_graph.node_link_graph(data, edges="links")
        except Exception as e:
            logger.warning("run_query_auto: failed to load graph (%s)", e)
            G = None

    embedder = BuildEmbedder(model_name=model_name, batch_size=64, normalize=True)

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
    )

    hits = retrieval_out.get("hits", [])
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
