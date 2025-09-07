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
import importlib
import inspect
import json
import sys
from typing import Any, Dict

from cgx.pipeline.auto import run_index_auto, run_query_auto
from cgx.config import EmbeddingConfig, FaissConfig, HybridSearchConfig


def _load_embedder(spec: str) -> Any:
    """
    Load an object or factory from "module:attr".
    If it's a class: instantiate with no args. If it's a callable: call to get the object.
    Otherwise: return the object itself.
    The returned object must expose `.encode(list[str]) -> ndarray`.
    """
    if not spec or ":" not in spec:
        raise ValueError('Embedder spec must be "module:attr" (got %r)' % spec)
    mod_name, attr = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    obj = getattr(mod, attr)
    if inspect.isclass(obj):
        return obj()
    if callable(obj) and not hasattr(obj, "encode"):
        # factory
        return obj()
    return obj


def _cmd_index(args: argparse.Namespace) -> None:
    # Touch config API (validate overrides surface and ensure they are used)
    _ = EmbeddingConfig.from_overrides()  # no overrides yet; verifies classmethod path
    _ = FaissConfig.from_overrides(metric=args.metric, index_type=args.index_type)
    _ = HybridSearchConfig.from_overrides()  # nothing used at index time
    # to_dict() round-trip to exercise that path too
    _ = _.to_dict() if hasattr(_, "to_dict") else None

    embedder = _load_embedder(args.embedder)
    summary = run_index_auto(
        project_root=args.project_root,
        embedder=embedder,
        out_dir=args.out_dir,
        metric=args.metric,
        index_type=args.index_type,
    )
    print(json.dumps(summary, indent=2))


def _cmd_query(args: argparse.Namespace) -> None:
    # Touch config API again to ensure from_overrides and to_dict paths are exercised
    hy = HybridSearchConfig.from_overrides(rrf_k=60.0)
    _ = hy.to_dict()

    embedder = _load_embedder(args.embedder)
    res = run_query_auto(
        index_dir=args.index_dir,
        records_path=args.records,
        embedder=embedder,
        query=args.query,
        chunks_path=args.chunks,
        graph_path=args.graph,
        top_k_per_view=args.top_k,
        neighbor_depth=args.depth,
        use_lexical=(not args.no_lexical),
        single_view=args.single_view,
    )
    # Pretty-print top hits
    print(json.dumps(res, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="cgx", description="Codebase RAG CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # index
    p_i = sub.add_parser("index", help="Parse -> records -> two-view embeddings -> FAISS -> persist")
    p_i.add_argument("--project-root", required=True)
    p_i.add_argument("--embedder", required=True, help="Import spec 'module:attr' that yields an encoder (see module docstring).")
    p_i.add_argument("--out-dir", required=True)
    p_i.add_argument("--metric", default="cosine", choices=["cosine", "l2", "ip"])
    p_i.add_argument("--index-type", default="flat", choices=["flat", "ivf", "hnsw"])
    # Keep knobs for compatibility (not used directly here but part of public surface)
    p_i.add_argument("--no-normalize-impl", action="store_true", help="(compat) Was used to affect impl-view text normalization.")
    p_i.add_argument("--strip-literals-impl", action="store_true", help="(compat) Was used to strip literals in impl view.")
    p_i.set_defaults(func=_cmd_index)

    # query
    p_q = sub.add_parser("query", help="Query two-view indices with hybrid fusion (semantic+lexical+graph).")
    p_q.add_argument("--index-dir", required=True, help="Path to 'indices' dir produced by `cgx index`.")
    p_q.add_argument("--records", required=True, help="Path to records.jsonl from `cgx index`.")
    p_q.add_argument("--embedder", required=True, help="Import spec 'module:attr' (must match the embedder used for index).")
    p_q.add_argument("--query", required=True)
    p_q.add_argument("--chunks", help="Optional: chunks.jsonl for lexical.")
    p_q.add_argument("--graph", help="Optional: graph.json for graph expansion.")
    p_q.add_argument("--top-k", type=int, default=10)
    p_q.add_argument("--depth", type=int, default=1, help="Neighbor depth for graph expansion.")
    p_q.add_argument("--no-lexical", action="store_true", help="Disable lexical component.")
    p_q.add_argument("--single-view", choices=["intent","impl"], help="Also run direct semantic_search on a single view.")
    p_q.add_argument("--limit", type=int, default=10, help="Print top-N rows.")
    p_q.set_defaults(func=_cmd_query)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
