# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ramin Mohammadi

# src/cgx/cli/main.py
from __future__ import annotations

"""
CLI for Codebase RAG (auto-wired).

This command-line tool wires the unused-but-important helpers into the actual run path:
- Uses `embeddings.build.build_embeddings` and `embeddings.index.build_faiss_index` during indexing.
- Uses `retrieval.orchestrator.hybrid_retrieve_two_view` for queries.
- Calls `retrieval.orchestrator.suggest_insertion_points` to propose anchors.
- Optionally exercises `embeddings.search.semantic_search` via --single-view.
- Touches config objects via .from_overrides()/.to_dict() to validate overrides surface.

This file is **add-only** with respect to behavior: the interface remains
compatible with prior flags seen in the project.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

from cgx.pipeline.auto import run_index_auto, run_query_auto
from cgx.config import EmbeddingConfig, FaissConfig, HybridSearchConfig
from cgx.embeddings.loader import load_embedder_from_spec


def _resolve_embedder_or_model(args: argparse.Namespace) -> tuple[Any, str]:
    """Return (embedder_obj_or_None, model_name).

    Users may provide either ``--embedder "module:attr"`` (advanced BYO) or
    ``--model NAME`` (sentence-transformers / HF id). They may also pass
    neither, in which case the default model in ``run_index_auto`` is used.
    """
    if getattr(args, "embedder", None):
        return load_embedder_from_spec(args.embedder), getattr(args, "model", None) or ""
    return None, getattr(args, "model", None) or "jinaai/jina-embeddings-v2-base-code"


def _cmd_index(args: argparse.Namespace) -> None:
    _ = EmbeddingConfig.from_overrides()
    _ = FaissConfig.from_overrides(metric=args.metric, index_type=args.index_type)
    _ = HybridSearchConfig.from_overrides()
    _ = _.to_dict() if hasattr(_, "to_dict") else None

    embedder, model_name = _resolve_embedder_or_model(args)
    summary = run_index_auto(
        project_root=args.project_root,
        out_dir=args.out_dir,
        metric=args.metric,
        index_type=args.index_type,
        model_name=model_name,
        embedder=embedder,
    )
    print(json.dumps(summary, indent=2))


def _cmd_query(args: argparse.Namespace) -> None:
    hy = HybridSearchConfig.from_overrides(rrf_k=60.0)
    _ = hy.to_dict()

    # Auto-discover sibling artifacts when not explicitly provided, mirroring
    # the behaviour of the web UI (handlers.py) and generate_code_plan.
    index_parent = Path(args.index_dir).parent
    graph_path = args.graph or None
    if graph_path is None:
        auto_graph = index_parent / "graph.json"
        if auto_graph.exists():
            graph_path = str(auto_graph)
    chunks_path = args.chunks or None
    if chunks_path is None:
        auto_chunks = index_parent / "chunks.jsonl"
        if auto_chunks.exists():
            chunks_path = str(auto_chunks)

    embedder, model_name = _resolve_embedder_or_model(args)
    res = run_query_auto(
        index_dir=args.index_dir,
        records_path=args.records,
        query=args.query,
        model_name=model_name,
        chunks_path=chunks_path,
        graph_path=graph_path,
        top_k_per_view=args.top_k,
        neighbor_depth=args.depth,
        use_lexical=(not args.no_lexical),
        single_view=args.single_view,
        embedder=embedder,
    )
    print(json.dumps(res, indent=2, default=str))


def _cmd_serve(args: argparse.Namespace) -> None:
    """Launch the FastAPI + React web UI."""
    try:
        from cgx.webui.launch import launch as _launch
    except Exception as e:
        raise SystemExit(f"Failed to import UI: {type(e).__name__}: {e}")
    _launch(host=args.host, port=args.port, no_browser=args.no_browser)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="cgx", description="Codebase RAG CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # index
    p_i = sub.add_parser("index", help="Parse -> records -> two-view embeddings -> FAISS -> persist")
    p_i.add_argument("--project-root", required=True)
    p_i.add_argument(
        "--model",
        default="jinaai/jina-embeddings-v2-base-code",
        help="Embedding model name (Sentence-Transformers or HF id). Used when --embedder is not given.",
    )
    p_i.add_argument(
        "--embedder",
        default=None,
        help="Optional advanced: import spec 'module:attr' that yields an object with .encode(list[str]). Overrides --model.",
    )
    p_i.add_argument("--out-dir", required=True)
    p_i.add_argument("--metric", default="cosine", choices=["cosine", "l2", "ip"])
    p_i.add_argument("--index-type", default="flat", choices=["flat", "ivf", "hnsw"])
    p_i.add_argument("--no-normalize-impl", action="store_true", help="(compat) Was used to affect impl-view text normalization.")
    p_i.add_argument("--strip-literals-impl", action="store_true", help="(compat) Was used to strip literals in impl view.")
    p_i.set_defaults(func=_cmd_index)

    # query
    p_q = sub.add_parser("query", help="Query two-view indices with hybrid fusion (semantic+lexical+graph).")
    p_q.add_argument("--index-dir", required=True, help="Path to 'indices' dir produced by `cgx index`.")
    p_q.add_argument("--records", required=True, help="Path to records.jsonl from `cgx index`.")
    p_q.add_argument(
        "--model",
        default="jinaai/jina-embeddings-v2-base-code",
        help="Embedding model name. Must match what was used at index time.",
    )
    p_q.add_argument(
        "--embedder",
        default=None,
        help="Optional advanced: import spec 'module:attr'. Overrides --model.",
    )
    p_q.add_argument("--query", required=True)
    p_q.add_argument("--chunks", help="Optional: chunks.jsonl for lexical.")
    p_q.add_argument("--graph", help="Optional: graph.json for graph expansion.")
    p_q.add_argument("--top-k", type=int, default=10)
    p_q.add_argument("--depth", type=int, default=1, help="Neighbor depth for graph expansion.")
    p_q.add_argument("--no-lexical", action="store_true", help="Disable lexical component.")
    p_q.add_argument("--single-view", choices=["intent","impl"], help="Also run direct semantic_search on a single view.")
    p_q.add_argument("--limit", type=int, default=10, help="Print top-N rows.")
    p_q.set_defaults(func=_cmd_query)

    # serve
    p_s = sub.add_parser("serve", help="Launch the CGX FastAPI + React web UI.")
    p_s.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    p_s.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765).")
    p_s.add_argument("--no-browser", action="store_true",
                     help="Do not open a browser tab on startup.")
    p_s.set_defaults(func=_cmd_serve)

    args = parser.parse_args(argv)
    # Opt-in anonymous telemetry; off unless ``CGX_TELEMETRY=1`` is set.
    try:
        from cgx import telemetry
        telemetry.ping()
    except Exception:
        pass
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
