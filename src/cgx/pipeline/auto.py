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


import os
import numpy as np
from typing import Any, Dict, Tuple, List

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

    This pipeline is deterministic and consistent:
      - Parses a codebase into chunks + call graph.
      - Builds index records (with view_intent and view_impl).
      - Flattens into corpus rows.
      - Embeds each view via build_embeddings (always).
      - Builds FAISS indices for each view.
      - Persists metadata including original chunk IDs.

    Parameters
    ----------
    project_root : str
        Root path of the project to parse.
    out_dir : str
        Output directory to save indices and metadata.
    metric : {"cosine","ip","l2"}, default "cosine"
        Distance metric for FAISS.
    index_type : str, default "flat"
        FAISS index type ("flat","ivf","hnsw",...).
    which : tuple[str], default ("intent","impl")
        Which views to index.
    model_name : str, default "jinaai/jina-embeddings-v2-base-code"
        Hugging Face model ID for embeddings.
    batch_size : int, default 64
        Batch size for embedding.

    Returns
    -------
    dict
        {
          "views": {
            "intent": {"index": faiss.Index, "meta": {...}, "rows": [...], "ids": np.ndarray},
            "impl": {...}
          },
          "metric": str
        }
    """
    os.makedirs(out_dir, exist_ok=True)

    # Parse & build graph
    chunks, calls = _ensure_tuple_parse(parse_codebase(project_root))
    G = build_knowledge_graph(chunks, calls)

    # Records + corpus
    records = make_index_records(chunks, G)
    corpus = prepare_embedding_corpus(records, which=which)

    # Split corpus per view
    per_view: Dict[str, List[Dict[str, Any]]] = {"intent": [], "impl": []}
    for row in corpus:
        vw = row.get("view")
        if vw in per_view:
            per_view[vw].append(row)

    # Build embeddings & FAISS per view
    indices: Dict[str, Any] = {"views": {}, "metric": metric}
    for view_name, rows in per_view.items():
        if not rows:
            indices["views"][view_name] = {
                "index": None,
                "meta": None,
                "rows": [],
                "ids": np.array([], dtype=np.int64),
            }
            continue

        # Always use build_embeddings with corpus rows
        embs = build_embeddings(
            rows,
            model_name=model_name,
            backend="auto",
            normalize=(metric in {"cosine", "ip"}),
            batch_size=batch_size,
            field_strategy="auto",
            max_length=256,   # << safer default for GPU
        )


        embs = np.asarray(embs, dtype=np.float32)

        # Build FAISS index
        index, meta = build_faiss_index(
            embs,
            metric=metric,
            index=index_type,
            return_meta=True,
        )

        # Preserve original chunk IDs
        orig_ids = [r.get("chunk_id") for r in rows]
        meta = {**(meta or {}), "orig_chunk_ids": orig_ids}

        indices["views"][view_name] = {
            "index": index,
            "meta": meta,
            "rows": rows,
            "ids": _to_faiss_ids(rows),
        }

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
    query: str,
    *,
    model_name: str = "jinaai/jina-embeddings-v2-base-code",
    chunks_path: Optional[str] = None,
    graph_path: Optional[str] = None,
    top_k_per_view: int = 10,
    neighbor_depth: int = 1,
    use_lexical: bool = True,
    single_view: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load indices, records, chunks, and graph, then execute hybrid two-view retrieval
    using the same embedding pipeline as indexing.

    Ensures consistency between index-time and query-time embeddings by always
    wrapping `build_embeddings` in a standard interface.

    Retrieval flow
    --------------
    1. Load FAISS indices (intent and impl views).
    2. Embed the query using the same pipeline as indexing.
    3. Perform hybrid retrieval:
         - Semantic search on both views (intent + impl).
         - Optional lexical search (BM25/keyword).
         - Optional graph expansion using neighbors in the code graph.
         - Fuse results with RRF.
    4. Aggregate results by file, class, and insertion anchors.
    5. Optionally run a single-view semantic search ("intent" or "impl") for debug.

    Parameters
    ----------
    index_dir : str
        Path to directory containing persisted FAISS indices (from run_index_auto).
    records_path : str
        Path to JSONL file with index records (from run_index_auto).
    query : str
        Natural language or code query string.
    model_name : str, default "jinaai/jina-embeddings-v2-base-code"
        Hugging Face model ID used for query embedding. Must match the model used at index time.
    chunks_path : str or None
        Optional path to raw chunks JSONL file. Enables lexical search and code context expansion.
    graph_path : str or None
        Optional path to serialized graph JSON. Enables graph-based neighbor expansion.
    top_k_per_view : int, default 10
        Number of top results to retrieve per view.
    neighbor_depth : int, default 1
        Depth of graph neighbors to include during expansion.
    use_lexical : bool, default True
        Whether to include lexical retrieval (e.g., BM25).
    single_view : {"intent","impl"} or None
        If set, perform an additional semantic search only on that view.

    Returns
    -------
    dict
        {
          "hits": list,             # fused retrieval results
          "top_files": list,        # aggregated results by file
          "top_classes": list,      # aggregated results by class
          "anchors": list,          # suggested insertion points
          "single_view": dict,      # (optional) single-view semantic results
          "debug": dict             # debug info: graph_used, chunks_available, etc.
        }
    """

    # ---------------- Wrapper around build_embeddings ----------------
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
    use_lexical: bool = True,
    single_view: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load indices, records, chunks, and graph, then execute hybrid two-view retrieval
    using the same embedding pipeline as indexing.
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

    # Debug
    logger.info("run_query_auto: loaded %d records", len(records))
    for view_name, view in (indices.get("views") or {}).items():
        idx = view.get("index")
        rows = view.get("rows")
        logger.info(
            "run_query_auto: view=%s, index_type=%s, rows=%s",
            view_name,
            type(idx).__name__ if idx is not None else None,
            len(rows) if isinstance(rows, list) else None,
        )

    # Hybrid retrieval
    retrieval_out = hybrid_retrieve_two_view(
        query,
        indices=indices,
        records=records,
        embedder=embedder,
        chunks=chunks,
        G=G,
        top_k_per_view=top_k_per_view,
        neighbor_depth=neighbor_depth,
        use_lexical=use_lexical,
    )

    # FIX: extract proper hits list
    hits = retrieval_out.get("chunks", [])

    files = retrieval_out.get("files", [])
    classes = retrieval_out.get("classes", [])
    anchors = suggest_insertion_points(query, hits, records)

    out: Dict[str, Any] = {
        "hits": hits,
        "top_files": files,
        "top_classes": classes,
        "anchors": anchors,
    }

    if single_view in {"intent", "impl"}:
        try:
            from cgx.embeddings.search import semantic_search
            view = indices.get("views", {}).get(single_view, {})
            index = view.get("index")
            rows = view.get("rows", [])

            logger.info(
                "run_query_auto: single_view=%s, index_type=%s, rows=%d",
                single_view,
                type(index).__name__ if index is not None else None,
                len(rows) if isinstance(rows, list) else -1,
            )

            out["single_view"] = {
                "view": single_view,
                "results": semantic_search(query, embedder, index, rows, top_k=top_k_per_view),
            }
        except Exception as e:
            out["single_view_error"] = f"{type(e).__name__}: {e}"

    out["debug"] = {
        "graph_used": G is not None,
        "chunks_available": bool(chunks),
        "top_k_per_view": top_k_per_view,
        "neighbor_depth": neighbor_depth,
        "use_lexical": use_lexical,
    }

    return out