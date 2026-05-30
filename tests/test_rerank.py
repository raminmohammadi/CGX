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
