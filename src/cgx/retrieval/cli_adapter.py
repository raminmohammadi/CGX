

# src/cgx/retrieval/cli_adapter.py
from __future__ import annotations

"""
CLI adapter to run the S7 HybridRetriever behind a --hybrid flag.

- Purely additive: if --hybrid isn't used, nothing else changes.
- Best-effort imports for S1/S4/S6/S7 components.
- Minimal defaults; override via flags if you like.
"""

from typing import Any, Dict, List, Optional
import argparse
import logging
import sys

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# ---------- flexible imports (don’t crash if some parts are absent) ----------
def _import_parse_codebase():
    try:
        from cgx.parse.ast_parser import parse_codebase  # preferred path
        return parse_codebase
    except Exception:
        pass
    try:
        from cgx.parse import parse_codebase  # alt
        return parse_codebase
    except Exception:
        pass
    try:
        # fallback if user exposed it under package root
        from cgx import parse_codebase  # type: ignore
        return parse_codebase
    except Exception:
        return None

def _import_build_graph():
    try:
        from cgx.graph.knowledge import build_knowledge_graph
        return build_knowledge_graph
    except Exception:
        pass
    try:
        from cgx.graph import build_knowledge_graph
        return build_knowledge_graph
    except Exception:
        return None

def _import_records():
    from cgx.embeddings.records import make_index_records
    return make_index_records

def _import_two_view_index():
    from cgx.retrieval.index import TwoViewIndex
    return TwoViewIndex

def _import_lexical_index():
    from cgx.retrieval.lexical import LexicalIndex
    return LexicalIndex

def _import_hybrid():
    from cgx.retrieval.hybrid import HybridRetriever, HybridConfig
    return HybridRetriever, HybridConfig


# ---------- minimal embedder loader (ST first, HF fallback) ----------
from cgx.embeddings.loader import load_embedder_from_model as _load_embedder


# ---------- tiny printer ----------
def _print_results(res: Dict[str, Any], *, top_chunks: int, top_groups: int):
    chunks = res.get("chunks", [])[:top_chunks]
    files  = res.get("files", [])[:top_groups]
    classes= res.get("classes", [])[:top_groups]

    if chunks:
        print("\n== Top chunks ==")
        for r in chunks:
            print(f"{r['rank']:>3}. {r['chunk_id']}  score={r['score']:.6f}")

    if files:
        print("\n== Top files ==")
        for i, f in enumerate(files, start=1):
            print(f"{i:>3}. {f['file']}  score={f['score']:.6f}")
            for m in f.get("members", [])[:5]:
                print(f"       - {m['chunk_id']} ({m['score']:.6f})")

    if classes:
        print("\n== Top classes ==")
        for i, c in enumerate(classes, start=1):
            print(f"{i:>3}. {c['class_id']}  score={c['score']:.6f}")
            for m in c.get("members", [])[:5]:
                print(f"       - {m['chunk_id']} ({m['score']:.6f})")
    print()


# ---------- public adapter API ----------
def register_hybrid_flag(parser: argparse.ArgumentParser) -> None:
    """
    Add the --hybrid flag + a few knobs to an existing argparse parser.
    Safe to call multiple times.
    """
    if any(a.option_strings == ["--hybrid"] for a in parser._actions):
        return  # already registered

    g = parser.add_argument_group("Hybrid retrieval (ADD-ONLY)")
    g.add_argument("--hybrid", action="store_true", help="Run hybrid retrieval pipeline")
    g.add_argument("--project-root", default=".", help="Project root (default: .)")
    g.add_argument("--query", help="Query text to search (required with --hybrid)")
    g.add_argument("--model-name", default="jinaai/jina-embeddings-v2-base-code",
                   help="HF/Sentence-Transformers model id for embeddings")
    g.add_argument("--top-k", type=int, default=30, help="Final top-K chunks to show (default: 30)")
    g.add_argument("--k-intent", type=int, default=50, help="ANN cutoff for intent view")
    g.add_argument("--k-impl", type=int, default=50, help="ANN cutoff for impl view")
    g.add_argument("--k-lex", type=int, default=50, help="Lexical cutoff")
    g.add_argument("--expand-top-n", type=int, default=10, help="Graph expansion seeds")
    g.add_argument("--expand-per-seed", type=int, default=12, help="Neighbors per seed")
    g.add_argument("--rrf-k", type=float, default=60.0, help="RRF stabilizer (higher=flatter)")
    g.add_argument("--no-graph", action="store_true", help="Skip graph building (faster)")


def maybe_run_hybrid(args: argparse.Namespace) -> bool:
    """
    If args.hybrid is set, run the pipeline and return True (caller may exit).
    Otherwise return False and let the existing CLI continue normally.
    """
    if not getattr(args, "hybrid", False):
        return False

    if not args.query:
        print("error: --query is required with --hybrid", file=sys.stderr)
        sys.exit(2)

    # 1) Parse codebase
    parse_codebase = _import_parse_codebase()
    if parse_codebase is None:
        print("error: could not import parse_codebase; ensure S1 is installed.", file=sys.stderr)
        sys.exit(2)
    chunks, calls = parse_codebase(args.project_root)
    logger.info("Parsed %d chunks, %d callsites", len(chunks), len(calls))

    # 2) Graph (optional)
    G = None
    if not getattr(args, "no_graph", False):
        build_knowledge_graph = _import_build_graph()
        if build_knowledge_graph:
            G = build_knowledge_graph(chunks, calls)
            logger.info("Graph nodes=%d, edges=%d", len(G.nodes) if G else 0, len(G.edges) if G else 0)
        else:
            logger.info("Graph builder not found; continuing without G")

    # 3) Records (S4)
    make_index_records = _import_records()
    records = make_index_records(chunks, G=G)
    logger.info("Records built: %d", len(records))

    # 4) Two-view ANN (S6)
    TwoViewIndex = _import_two_view_index()
    embedder = _load_embedder(args.model_name)
    tv = TwoViewIndex.from_records(records, embedder=embedder)  # uses default FAISS settings

    # 5) Lexical (S7)
    LexicalIndex = _import_lexical_index()
    lex = LexicalIndex.from_records(records)

    # 6) Hybrid orchestration (S7)
    HybridRetriever, HybridConfig = _import_hybrid()
    cfg = HybridConfig(
        k_intent=int(args.k_intent),
        k_impl=int(args.k_impl),
        k_lex=int(args.k_lex),
        expand_top_n=int(args.expand_top_n),
        expand_per_seed=int(args.expand_per_seed),
        rrf_k=float(args.rrf_k),
        top_k_chunks=int(args.top_k),
    )
    hyb = HybridRetriever(tv_index=tv, records=records, lexical_index=lex, G=G)
    res = hyb.search(args.query, embedder=embedder, cfg=cfg)

    _print_results(res, top_chunks=cfg.top_k_chunks, top_groups=20)
    return True


# Optional standalone entry (so you can also wire a separate console script if you want)
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser("cgx-hybrid")
    register_hybrid_flag(p)
    args = p.parse_args(argv)
    if maybe_run_hybrid(args):
        return 0
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
