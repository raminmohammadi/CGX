from __future__ import annotations

"""
Auto-wired pipeline that exercises the canonical components:
- parse -> graph -> records -> two-view corpus
- embeddings.build.build_embeddings
- embeddings.index.build_faiss_index
- retrieval.orchestrator.hybrid_retrieve_two_view
- retrieval.orchestrator.aggregate_by_file/aggregate_by_class
- retrieval.orchestrator.suggest_insertion_points

Adds **graph + chunks persistence and loading** so hybrid retrieval can
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
    embedder: Any,
    out_dir: str,
    metric: str = "cosine",
    index_type: str = "flat",
    which: Tuple[str, str] = ("intent", "impl"),
) -> Dict[str, Any]:
    """Build two-view FAISS indices and persist alongside records **and** graph/chunks.

    Returns a summary dict with counts and output paths.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Parse & Graph
    chunks, calls = _ensure_tuple_parse(parse_codebase(project_root))
    G = build_knowledge_graph(chunks, calls)

    # Records + corpus
    records = make_index_records(chunks, G)
    corpus = prepare_embedding_corpus(records, which=which)

    # Split corpus per-view
    per_view: Dict[str, List[Dict[str, Any]]] = {"intent": [], "impl": []}
    for row in corpus:
        vw = row.get("view")
        if vw in per_view:
            per_view[vw].append(row)

    # Build embeddings & FAISS per-view via the reusable helpers
    indices: Dict[str, Any] = {"views": {}, "metric": metric}
    for view_name, rows in per_view.items():
        texts = [str(r.get("text", "")) for r in rows]
        ids = _to_faiss_ids(rows)  # robust regardless of chunk_id type
        if len(texts) == 0:
            indices["views"][view_name] = {
                "index": None,
                "meta": None,
                "rows": [],
                "ids": np.array([], dtype=np.int64),
            }
            continue
        # Prefer the injected embedder if it exposes .encode
        if hasattr(embedder, "encode"):
            embs = embedder.encode(texts)
        else:
            # Fallback to project helper (keeps the function exercised)
            embs = build_embeddings(
                texts, backend="auto", normalize=(metric in {"cosine", "ip"}), batch_size=64
            )
        embs = np.asarray(embs, dtype=np.float32)
        if metric in {"cosine", "ip"}:
            # L2-normalize rows for cosine/IP
            norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
            embs = embs / norms
        index, meta = build_faiss_index(embs, metric=metric, index=index_type, return_meta=True)

        # Preserve original chunk ids alongside index metadata (they may be strings/paths)
        orig_ids = [r.get("chunk_id") for r in rows]
        meta = {**(meta or {}), "orig_chunk_ids": orig_ids}

        indices["views"][view_name] = {"index": index, "meta": meta, "rows": rows, "ids": ids}

    # Persist indices/records
    save_indices(indices, os.path.join(out_dir, "indices"))
    save_jsonl(records, os.path.join(out_dir, "records.jsonl"))

    # Persist raw chunks and graph for lexical + graph expansion at query time
    save_jsonl(chunks, os.path.join(out_dir, "chunks.jsonl"))
    graph_path = os.path.join(out_dir, "graph.json")
    try:
        with open(graph_path, "w", encoding="utf-8") as f:
            json.dump(json_graph.node_link_data(G), f)
    except Exception:
        # Fail-soft: if graph cannot be serialized, continue; query path will just skip it.
        graph_path = None

    return {
        "counts": {k: len(v) for k, v in per_view.items()},
        "out": {
            "indices": os.path.join(out_dir, "indices"),
            "records": os.path.join(out_dir, "records.jsonl"),
            "chunks": os.path.join(out_dir, "chunks.jsonl"),
            "graph": graph_path,
        },
    }


def run_query_auto(
    index_dir: str,
    records_path: str,
    embedder: Any,
    query: str,
    *,
    chunks_path: Optional[str] = None,
    graph_path: Optional[str] = None,
    top_k_per_view: int = 10,
    neighbor_depth: int = 1,
    use_lexical: bool = True,
    single_view: Optional[str] = None,
) -> Dict[str, Any]:
    """Load indices/records and execute hybrid two-view retrieval; also compute anchors.

    If `single_view` is set to "intent" or "impl", also perform a direct semantic search
    on that view (uses the same indices) and include its top-k in the payload.
    """
    indices = load_indices(index_dir)
    records = load_jsonl(records_path)
    chunks = load_jsonl(chunks_path) if chunks_path else None

    # Load graph if provided
    G = None
    if graph_path:
        try:
            with open(graph_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            G = json_graph.node_link_graph(data)
        except Exception:
            G = None

    # Hybrid (semantic both views + optional lexical + graph RRF)
    fused = hybrid_retrieve_two_view(
        query,
        indices=indices,
        embedder=embedder,
        chunks=chunks,
        G=G,
        top_k_per_view=top_k_per_view,
        neighbor_depth=neighbor_depth,
        use_lexical=use_lexical,
    )

    files = aggregate_by_file(fused, records)
    classes = aggregate_by_class(fused, records)
    anchors = suggest_insertion_points(query, fused, records)

    out: Dict[str, Any] = {
        "hits": fused,
        "top_files": files,
        "top_classes": classes,
        "anchors": anchors,
    }

    # Optional: single-view semantic
    if single_view in {"intent", "impl"}:
        try:
            from cgx.embeddings.search import semantic_search
            view = indices.get("views", {}).get(single_view, {})
            index = view.get("index")
            rows = view.get("rows", [])
            helper_chunks = [{"id": r.get("chunk_id"), "text": r.get("text", "")} for r in rows]
            out["single_view"] = {
                "view": single_view,
                "results": semantic_search(query, embedder, index, helper_chunks)[:top_k_per_view],
            }
        except Exception as e:
            out["single_view_error"] = f"{type(e).__name__}: {e}"

    # Extra debug flags for callers
    out["debug"] = {
        "graph_used": G is not None,
        "chunks_available": bool(chunks),
        "top_k_per_view": top_k_per_view,
        "neighbor_depth": neighbor_depth,
        "use_lexical": use_lexical,
    }

    return out