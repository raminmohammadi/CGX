"""Tests for cgx.graph.backend.CodeGraphBackend.

Covers wrapping/idempotency, the multi-digraph edge-attribute flattener,
BFS distances, undirected neighbor de-duplication, and error tolerance
when the underlying graph is missing or malformed.
"""

from __future__ import annotations

import pytest

nx = pytest.importorskip("networkx")

from cgx.graph.backend import CodeGraphBackend, _flatten_edge_attrs


def _multi_graph() -> "nx.MultiDiGraph":
    G = nx.MultiDiGraph()
    G.add_node("a", type="function", name="alpha")
    G.add_node("b", type="method", name="beta")
    G.add_node("c", type="unresolved", name="gamma")
    G.add_edge("a", "b", type="calls", internal=True)
    G.add_edge("a", "c", type="calls")
    G.add_edge("b", "a", type="defines")
    return G


def test_wrap_none_returns_none():
    assert CodeGraphBackend.wrap(None) is None


def test_wrap_is_idempotent():
    G = _multi_graph()
    b1 = CodeGraphBackend.wrap(G)
    b2 = CodeGraphBackend.wrap(b1)
    assert b1 is b2
    assert b1.raw is G


def test_has_node_and_adjacency():
    b = CodeGraphBackend.wrap(_multi_graph())
    assert b.has_node("a") and not b.has_node("zzz")
    assert sorted(b.successors("a")) == ["b", "c"]
    assert b.predecessors("a") == ["b"]


def test_undirected_neighbors_dedupes():
    G = nx.MultiDiGraph()
    G.add_edge("a", "b")
    G.add_edge("b", "a")  # bidirectional → b appears only once.
    b = CodeGraphBackend.wrap(G)
    assert b.undirected_neighbors("a") == ["b"]


def test_edge_attrs_handles_multidigraph_shape():
    b = CodeGraphBackend.wrap(_multi_graph())
    attrs = b.edge_attrs("a", "b")
    assert attrs.get("type") == "calls"
    assert attrs.get("internal") is True


def test_edge_attrs_handles_plain_digraph():
    G = nx.DiGraph()
    G.add_edge("a", "b", type="defines")
    b = CodeGraphBackend.wrap(G)
    assert b.edge_attrs("a", "b") == {"type": "defines"}


def test_edge_attrs_missing_returns_empty_dict():
    b = CodeGraphBackend.wrap(_multi_graph())
    assert b.edge_attrs("a", "no-such-node") == {}


def test_node_attrs_returns_copy():
    b = CodeGraphBackend.wrap(_multi_graph())
    attrs = b.node_attrs("a")
    assert attrs == {"type": "function", "name": "alpha"}
    attrs["mutated"] = True
    # The original graph must not see the mutation.
    assert "mutated" not in b.node_attrs("a")


def test_bfs_distances_respects_cutoff():
    G = nx.MultiDiGraph()
    G.add_edge("a", "b")
    G.add_edge("b", "c")
    G.add_edge("c", "d")
    b = CodeGraphBackend.wrap(G)
    out = b.bfs_distances("a", cutoff=2)
    assert out == {"a": 0, "b": 1, "c": 2}


def test_bfs_distances_missing_source():
    b = CodeGraphBackend.wrap(_multi_graph())
    assert b.bfs_distances("zzz", cutoff=5) == {}


def test_flatten_edge_attrs_shapes():
    assert _flatten_edge_attrs({}) == {}
    assert _flatten_edge_attrs(None) == {}
    assert _flatten_edge_attrs({"type": "calls"}) == {"type": "calls"}
    multi = {0: {"type": "calls"}, 1: {"type": "calls", "internal": True}}
    flat = _flatten_edge_attrs(multi)
    assert flat == {"type": "calls"}  # first sub-dict wins; deterministic


def test_backend_tolerates_broken_graph():
    class BrokenGraph:
        def __contains__(self, _):
            raise RuntimeError("boom")
        def successors(self, _):
            raise RuntimeError("boom")
        def predecessors(self, _):
            raise RuntimeError("boom")

    b = CodeGraphBackend.wrap(BrokenGraph())
    assert b.has_node("x") is False
    assert b.successors("x") == []
    assert b.predecessors("x") == []
    assert b.undirected_neighbors("x") == []


def test_helpers_consume_backend_directly():
    """embeddings.helpers must accept both raw nx and CodeGraphBackend inputs."""
    from cgx.embeddings.helpers import (
        _calls_degree, _calls_out_ids, _defines_children_ids, _neighbors_summary,
    )
    G = _multi_graph()
    b = CodeGraphBackend.wrap(G)
    assert _calls_degree(G, "a") == _calls_degree(b, "a")
    assert _calls_out_ids(G, "a") == _calls_out_ids(b, "a")
    assert _defines_children_ids(G, "b") == _defines_children_ids(b, "b")
    assert _neighbors_summary(G, "a") == _neighbors_summary(b, "a")
