# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ramin Mohammadi

# src/cgx/pipeline/run.py
from __future__ import annotations

"""
End-to-end deterministic pipeline runners (ADD-ONLY).

This version aligns with your existing S4 `records.py` API:
  - make_index_records(chunks, G=..., normalize_impl=..., strip_literals=...)
  - prepare_embedding_corpus(records, which=('intent','impl'))
"""

import logging
import os
from typing import Any, Dict, Optional

from cgx.logging_setup import get_logger
logger = get_logger("run")


# core
from cgx.parser.parse_codebase import parse_codebase
from cgx.graph.build_graph import build_knowledge_graph
from cgx.embeddings.records import make_index_records, prepare_embedding_corpus

# retrieval orchestrator (added elsewhere in S5)
from cgx.retrieval.orchestrator import (
    build_two_view_indices,
    hybrid_retrieve_two_view,
    aggregate_by_file,
    aggregate_by_class,
)

# persistence
from cgx.io.persist import (
    save_indices, load_indices,
    save_jsonl, load_jsonl,
    save_graph_json, load_graph_json,
)


def run_index(
    project_root: str,
    *,
    embedder: Any,
    out_dir: str,
    metric: str = "cosine",
    index_type: str = "flat",
    normalize_impl: bool = True,
    strip_literals_impl: bool = False,
) -> Dict[str, Any]:
    """
    Parse → graph → records → views → embed two-view corpus → FAISS indices → persist.

    All operations are deterministic and auditable. No model assumptions beyond the
    injected `embedder` needing an `.encode(list[str]) -> np.ndarray` method.
    """
    if not os.path.isdir(project_root):
        raise NotADirectoryError(f"run_index: project_root not found: {project_root}")
    os.makedirs(out_dir, exist_ok=True)

    logger.info("Parsing project...")
    chunks, calls = parse_codebase(project_root)
    logger.info("Parsed: %d chunks, %d call sites", len(chunks), len(calls))

    logger.info("Building knowledge graph...")
    G = build_knowledge_graph(chunks, calls)
    logger.info("Graph nodes=%d, edges=%d", G.number_of_nodes(), G.number_of_edges())

    logger.info("Building canonical records (with views)…")
    records = make_index_records(
        chunks,
        G=G,
        normalize_impl=normalize_impl,
        strip_literals=strip_literals_impl,
    )
    logger.info("Records: %d", len(records))

    logger.info("Preparing dual-view corpus (flatten)…")
    corpus = prepare_embedding_corpus(records, which=("intent", "impl"))
    logger.info("Corpus rows: %d", len(corpus))

    logger.info("Building two-view indices (metric=%s, index=%s)…", metric, index_type)
    indices = build_two_view_indices(
        corpus,
        embedder=embedder,
        metric=metric,
        index_type=index_type,
    )

    # Persist artifacts
    logger.info("Saving indices & artifacts to %s", out_dir)
    save_indices(indices, os.path.join(out_dir, "indices"))
    save_jsonl(records, os.path.join(out_dir, "records.jsonl"))
    save_jsonl(chunks, os.path.join(out_dir, "chunks.jsonl"))
    save_jsonl(calls,  os.path.join(out_dir, "calls.jsonl"))
    save_graph_json(G, os.path.join(out_dir, "graph.json"))

    return {
        "chunks": len(chunks),
        "calls": len(calls),
        "records": len(records),
        "corpus_rows": len(corpus),
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(),
        "out_dir": out_dir,
    }


def run_query(
    *,
    index_dir: str,
    records_path: str,
    embedder: Any,
    query: str,
    chunks_path: Optional[str] = None,
    graph_path: Optional[str] = None,
    top_k_per_view: int = 10,
    neighbor_depth: int = 1,
    use_lexical: bool = True,
) -> Dict[str, Any]:
    """
    Load indices/records and execute hybrid two-view retrieval (with RRF).
    Optionally load chunks and graph for lexical & graph expansion.
    """
    indices = load_indices(index_dir)
    records = load_jsonl(records_path)
    chunks = load_jsonl(chunks_path) if chunks_path else None
    G = load_graph_json(graph_path) if graph_path else None

    hits = hybrid_retrieve_two_view(
        query,
        indices=indices,
        embedder=embedder,
        chunks=chunks,
        G=G,
        top_k_per_view=top_k_per_view,
        neighbor_depth=neighbor_depth,
        use_lexical=use_lexical,
    )

    files = aggregate_by_file(hits, records)
    classes = aggregate_by_class(hits, records)

    return {
        "hits": hits,
        "top_files": files,
        "top_classes": classes,
    }
