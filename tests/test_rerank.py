"""Regression tests for the post-RRF rerank stage in HybridRetriever.

Covers:

* Bug fix: graph-only neighbors are surfaced in hits (previously they were
  silently dropped because the score-bump loop never appended new ids to
  ``fused``).
* Configurable bonuses: ``HybridConfig.graph_bonus`` and ``symbol_boost``
  control magnitudes and can be zeroed out to disable the respective bumps.
* Cross-encoder reranker hook: an injected fake reranker rearranges the head
  of the candidate pool and records provenance.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest

from cgx.retrieval.orchestrator import HybridConfig, HybridRetriever
from cgx.retrieval import reranker as reranker_mod


class _FakeView:
    """Minimal stand-in for TwoViewIndex used by HybridRetriever.search."""

    def __init__(self, hits_per_view: Dict[str, List[Dict[str, Any]]]) -> None:
        self._hits = hits_per_view

    def available_views(self) -> List[str]:
        return list(self._hits.keys())

    def search_view(self, view: str, query: str, *, embedder: Any, top_k: int):
        return self._hits[view][:top_k]


class _FakeGraph:
    """Trivial graph supporting ``in`` / successors / predecessors."""

    def __init__(self, edges: List[Tuple[str, str]]) -> None:
        self._succ: Dict[str, List[str]] = {}
        self._pred: Dict[str, List[str]] = {}
        for a, b in edges:
            self._succ.setdefault(a, []).append(b)
            self._pred.setdefault(b, []).append(a)
            self._succ.setdefault(b, [])
            self._pred.setdefault(a, [])

    def __contains__(self, item: str) -> bool:
        return item in self._succ or item in self._pred

    def successors(self, n: str) -> List[str]:
        return list(self._succ.get(n, []))

    def predecessors(self, n: str) -> List[str]:
        return list(self._pred.get(n, []))


def _records(*cids: str) -> List[Dict[str, Any]]:
    out = []
    for cid in cids:
        out.append({"id": cid, "name": cid.split("::")[-1], "file": cid.split("::")[0],
                    "code": f"def {cid.split('::')[-1]}(): pass"})
    return out


def _make_retriever(records, hits, edges=()):
    return HybridRetriever(
        tv_index=_FakeView(hits),
        records=records,
        lexical_index=None,
        chunks=[{"id": r["id"], "code": r["code"], "name": r["name"], "file": r["file"]} for r in records],
        G=_FakeGraph(list(edges)) if edges else None,
    )


def test_graph_only_neighbor_surfaces_in_hits():
    """A graph-only neighbor must appear in ``hits`` (was previously dropped)."""
    records = _records("pkg/a.py::function::seed", "pkg/b.py::function::neighbor")
    hits = {
        "intent": [{"chunk_id": "pkg/a.py::function::seed", "rank": 1, "score": 0.9}],
        "impl":   [{"chunk_id": "pkg/a.py::function::seed", "rank": 1, "score": 0.9}],
    }
    edges = [("pkg/a.py::function::seed", "pkg/b.py::function::neighbor")]
    r = _make_retriever(records, hits, edges)
    cfg = HybridConfig(k_intent=5, k_impl=5, k_lex=5, expand_top_n=2,
                       graph_depth=1, top_k_chunks=10, graph_bonus=0.2)
    out = r.search("seed", embedder=None, cfg=cfg)
    cids = [h["chunk_id"] for h in out["hits"]]
    assert "pkg/b.py::function::neighbor" in cids


def test_graph_bonus_zero_disables_neighbor_pull():
    records = _records("pkg/a.py::function::seed", "pkg/b.py::function::neighbor")
    hits = {
        "intent": [{"chunk_id": "pkg/a.py::function::seed", "rank": 1, "score": 0.9}],
        "impl":   [{"chunk_id": "pkg/a.py::function::seed", "rank": 1, "score": 0.9}],
    }
    edges = [("pkg/a.py::function::seed", "pkg/b.py::function::neighbor")]
    r = _make_retriever(records, hits, edges)
    cfg = HybridConfig(k_intent=5, k_impl=5, k_lex=5, expand_top_n=2,
                       graph_depth=1, top_k_chunks=10, graph_bonus=0.0)
    out = r.search("seed", embedder=None, cfg=cfg)
    cids = [h["chunk_id"] for h in out["hits"]]
    assert "pkg/b.py::function::neighbor" not in cids


def test_symbol_boost_respects_config():
    records = _records("pkg/a.py::function::needle", "pkg/b.py::function::hay")
    hits = {
        "intent": [{"chunk_id": "pkg/b.py::function::hay",    "rank": 1, "score": 0.9},
                   {"chunk_id": "pkg/a.py::function::needle", "rank": 2, "score": 0.5}],
        "impl":   [{"chunk_id": "pkg/b.py::function::hay",    "rank": 1, "score": 0.9}],
    }
    r = _make_retriever(records, hits)
    cfg = HybridConfig(k_intent=5, k_impl=5, k_lex=5, top_k_chunks=10,
                       graph_bonus=0.0, symbol_boost=10.0)
    out = r.search("explain `needle`", embedder=None, cfg=cfg)
    assert out["hits"][0]["chunk_id"] == "pkg/a.py::function::needle"
    assert out["hits"][0]["provenance"].get("symbol_match") is True


FOO = "pkg/a.py::function::foo"
BAR = "pkg/b.py::function::bar"


def _score_of(out, cid):
    for h in out["hits"]:
        if h["chunk_id"] == cid:
            return h["score"]
    raise AssertionError(f"{cid} not in hits")


def _foo_bar_retriever():
    """Retriever isolating the semantic + symbol-boost interaction: no chunks
    (so the regex fallback is off) and no lexical helpers on the records (so
    BM25 is empty). Only the intent/impl semantic ranks and the symbol boost
    decide the order -- exactly what the calibration governs."""
    records = _records(FOO, BAR)
    # bar tops both views; foo appears only at intent rank 2 (a weak semantic hit).
    hits = {
        "intent": [{"chunk_id": BAR, "rank": 1, "score": 0.9},
                   {"chunk_id": FOO, "rank": 2, "score": 0.3}],
        "impl":   [{"chunk_id": BAR, "rank": 1, "score": 0.9}],
    }
    return HybridRetriever(
        tv_index=_FakeView(hits), records=records,
        lexical_index=None, chunks=[], G=None,
    )


def test_bare_symbol_boost_does_not_dominate_by_orders_of_magnitude():
    """A bare symbol match adds ~one signal's worth (proportional to rrf_scale),
    not the old absolute +0.5 that was ~15x the top RRF score. ``bar`` is the
    strong hit (top of both views) and never matches the query token, so its
    score equals rrf_scale; ``foo`` matches the bare token ``foo`` from a lone
    weak semantic rank. Under the old absolute boost foo (~0.52) buried bar
    (~0.033); under the proportional boost bar stays on top."""
    r = _foo_bar_retriever()
    cfg = HybridConfig(k_intent=5, k_impl=5, k_lex=5, top_k_chunks=10,
                       graph_bonus=0.0, enable_reranker=False)
    out = r.search("the foo thing", embedder=None, cfg=cfg)
    foo_score, bar_score = _score_of(out, FOO), _score_of(out, BAR)
    assert out["hits"][0]["chunk_id"] == BAR  # strong hit stays on top
    # Proportional: a bare match is bounded near the fused scale, not 15x it.
    assert foo_score < 2.0 * bar_score
    prov = {h["chunk_id"]: h["provenance"] for h in out["hits"]}
    assert prov[FOO].get("symbol_match_kind") == "bare"


def test_quoted_symbol_boost_strictly_exceeds_bare():
    """Quoting a symbol (an explicit, intended reference) yields a strictly
    stronger boost than the same token appearing bare in prose."""
    cfg = HybridConfig(k_intent=5, k_impl=5, k_lex=5, top_k_chunks=10,
                       graph_bonus=0.0, enable_reranker=False)
    bare = _foo_bar_retriever().search("the foo thing", embedder=None, cfg=cfg)
    quoted = _foo_bar_retriever().search("the `foo` thing", embedder=None, cfg=cfg)
    # Only the boost magnitude differs between the two runs; quoted > bare.
    assert _score_of(quoted, FOO) > _score_of(bare, FOO)
    qprov = {h["chunk_id"]: h["provenance"] for h in quoted["hits"]}
    assert qprov[FOO].get("symbol_match_kind") == "quoted"
    # Explicitly named, foo now leads.
    assert quoted["hits"][0]["chunk_id"] == FOO


def test_file_aggregation_favors_single_best_chunk_over_member_count():
    """A file with ONE excellent chunk must outrank a file with many mediocre
    chunks. The corroboration bonus is capped at the group's best score, so
    ranking is anchored to relevance, not member count (the old unbounded
    decayed SUM let the crowded file win)."""
    a1 = "pkg/a.py::function::a1"
    bs = [f"pkg/b.py::function::b{i}" for i in range(1, 6)]
    records = _records(a1, *bs)
    # a1 tops both views (strong). b1..b5 are a crowd of weak intent hits.
    intent = [{"chunk_id": a1, "rank": 1, "score": 0.9}]
    intent += [{"chunk_id": b, "rank": i + 2, "score": 0.3} for i, b in enumerate(bs)]
    hits = {"intent": intent, "impl": [{"chunk_id": a1, "rank": 1, "score": 0.9}]}
    r = HybridRetriever(tv_index=_FakeView(hits), records=records,
                        lexical_index=None, chunks=[], G=None)
    cfg = HybridConfig(k_intent=10, k_impl=10, k_lex=10, top_k_chunks=20,
                       graph_bonus=0.0, enable_reranker=False)
    out = r.search("generic query terms", embedder=None, cfg=cfg)
    assert out["top_files"], "expected file aggregation"
    assert out["top_files"][0]["file"] == "pkg/a.py"


def test_reranker_hook_runs_and_records_provenance(monkeypatch):
    records = _records("pkg/a.py::function::alpha", "pkg/b.py::function::beta")
    hits = {
        "intent": [{"chunk_id": "pkg/a.py::function::alpha", "rank": 1, "score": 0.9},
                   {"chunk_id": "pkg/b.py::function::beta",  "rank": 2, "score": 0.5}],
        "impl":   [{"chunk_id": "pkg/a.py::function::alpha", "rank": 1, "score": 0.9}],
    }

    class FakeCE:
        # Make beta rank above alpha after reranking.
        def predict(self, pairs):
            return [1.0 if "beta" in p[1] else 0.0 for p in pairs]

    monkeypatch.setattr(reranker_mod, "get_default_reranker", lambda *a, **kw: FakeCE())
    r = _make_retriever(records, hits)
    cfg = HybridConfig(k_intent=5, k_impl=5, k_lex=5, top_k_chunks=10,
                       graph_bonus=0.0, symbol_boost=0.0,
                       enable_reranker=True, reranker_top_n=5, reranker_weight=1.0)
    out = r.search("anything", embedder=None, cfg=cfg)
    assert out["hits"][0]["chunk_id"] == "pkg/b.py::function::beta"
    assert "reranker_score" in out["hits"][0]["provenance"]


def test_reranker_silently_falls_back_when_no_model(monkeypatch):
    records = _records("pkg/a.py::function::alpha")
    hits = {"intent": [{"chunk_id": "pkg/a.py::function::alpha", "rank": 1, "score": 0.9}],
            "impl":   [{"chunk_id": "pkg/a.py::function::alpha", "rank": 1, "score": 0.9}]}
    monkeypatch.setattr(reranker_mod, "get_default_reranker", lambda *a, **kw: None)
    r = _make_retriever(records, hits)
    cfg = HybridConfig(enable_reranker=True, graph_bonus=0.0, symbol_boost=0.0)
    out = r.search("alpha", embedder=None, cfg=cfg)
    assert out["hits"]
