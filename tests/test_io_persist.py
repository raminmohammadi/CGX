"""Tests for :mod:`cgx.io.persist` (JSONL + index + graph save/load)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from cgx.io.persist import (
    _NX, clear_indices_cache, load_graph_json, load_indices, load_jsonl,
    save_graph_json, save_indices, save_jsonl,
)


@pytest.fixture(autouse=True)
def _clear_index_cache():
    # Keep the in-process load_indices cache from leaking across tests.
    clear_indices_cache()
    yield
    clear_indices_cache()


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------
def test_save_and_load_jsonl_roundtrip(tmp_path):
    items = [{"a": 1}, {"b": 2, "c": [3, 4]}]
    p = tmp_path / "nested" / "out.jsonl"
    save_jsonl(items, str(p))
    assert p.exists()
    loaded = load_jsonl(str(p))
    assert loaded == items


def test_save_jsonl_skips_non_dict_items(tmp_path, caplog):
    p = tmp_path / "mixed.jsonl"
    save_jsonl([{"keep": True}, "not-a-dict", 42, {"also": True}], str(p))
    rows = load_jsonl(str(p))
    assert rows == [{"keep": True}, {"also": True}]


def test_save_jsonl_creates_parent_dir(tmp_path):
    p = tmp_path / "deep" / "deeper" / "file.jsonl"
    save_jsonl([{"x": 1}], str(p))
    assert p.exists()


def test_load_jsonl_returns_empty_for_empty_file(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    assert load_jsonl(str(p)) == []


def test_load_jsonl_skips_non_dict_lines(tmp_path):
    p = tmp_path / "mixed.jsonl"
    p.write_text('{"ok": true}\n[1, 2, 3]\n42\n{"also": true}\n')
    assert load_jsonl(str(p)) == [{"ok": True}, {"also": True}]


# ---------------------------------------------------------------------------
# Indices (rows-only path -- FAISS may be absent in test env)
# ---------------------------------------------------------------------------
def test_save_and_load_indices_writes_rows_and_meta(tmp_path):
    indices = {
        "metric": "cosine",
        "views": {
            "intent": {"rows": [{"chunk_id": "a", "text": "x"}], "index": None},
            "impl":   {"rows": [{"chunk_id": "b", "text": "y"}], "index": None},
        },
    }
    out_dir = tmp_path / "idx"
    save_indices(indices, str(out_dir))
    # meta.json + per-view rows.jsonl always present.
    assert (out_dir / "meta.json").exists()
    assert (out_dir / "intent.rows.jsonl").exists()
    assert (out_dir / "impl.rows.jsonl").exists()
    meta = json.loads((out_dir / "meta.json").read_text())
    assert meta["metric"] == "cosine"
    assert "intent" in meta["views"] and "impl" in meta["views"]

    loaded = load_indices(str(out_dir))
    assert loaded["metric"] == "cosine"
    assert loaded["views"]["intent"]["rows"] == [{"chunk_id": "a", "text": "x"}]
    assert loaded["views"]["impl"]["rows"] == [{"chunk_id": "b", "text": "y"}]


def test_save_indices_persists_provenance_metadata(tmp_path):
    # save_indices must record which model built the index (and when / for
    # which project) so a later query can pick the right embedder and detect a
    # dimension mismatch up front.
    indices = {
        "metric": "cosine",
        "embed_model": "BAAI/bge-m3",
        "index_type": "flat",
        "project_root": "/some/project",
        "indexed_at": "2026-07-14T18:00:00",
        "counts": {"intent": 3, "impl": 3},
        "views": {
            "intent": {"rows": [{"chunk_id": "a", "text": "x"}], "index": None},
            "impl":   {"rows": [{"chunk_id": "b", "text": "y"}], "index": None},
        },
    }
    out_dir = tmp_path / "idx"
    save_indices(indices, str(out_dir))
    meta = json.loads((out_dir / "meta.json").read_text())
    assert meta["embed_model"] == "BAAI/bge-m3"
    assert meta["project_root"] == "/some/project"
    assert meta["indexed_at"] == "2026-07-14T18:00:00"
    assert meta["counts"] == {"intent": 3, "impl": 3}

    loaded = load_indices(str(out_dir))
    assert loaded["embed_model"] == "BAAI/bge-m3"
    assert loaded["indexed_at"] == "2026-07-14T18:00:00"


def test_save_indices_defaults_indexed_at_when_absent(tmp_path):
    indices = {"metric": "cosine", "views": {}}
    out_dir = tmp_path / "idx2"
    save_indices(indices, str(out_dir))
    meta = json.loads((out_dir / "meta.json").read_text())
    assert meta["indexed_at"]  # stamped automatically


def test_load_indices_raises_on_missing_meta(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_indices(str(tmp_path))


def test_load_indices_caches_repeated_reads(tmp_path):
    # Second read of an unchanged index dir returns the memoized object.
    indices = {
        "metric": "cosine",
        "views": {
            "intent": {"rows": [{"chunk_id": "a", "text": "x"}], "index": None},
            "impl":   {"rows": [{"chunk_id": "b", "text": "y"}], "index": None},
        },
    }
    out_dir = tmp_path / "idx"
    save_indices(indices, str(out_dir))
    first = load_indices(str(out_dir))
    second = load_indices(str(out_dir))
    assert first is second  # served from cache, no re-parse


def test_load_indices_invalidates_on_meta_change(tmp_path):
    # Re-saving the index rewrites meta.json; the cache must not go stale.
    out_dir = tmp_path / "idx"
    save_indices(
        {"metric": "cosine",
         "views": {"intent": {"rows": [{"chunk_id": "a", "text": "x"}],
                              "index": None},
                   "impl": {"rows": [], "index": None}}},
        str(out_dir))
    first = load_indices(str(out_dir))
    assert first["views"]["intent"]["rows"] == [{"chunk_id": "a", "text": "x"}]

    # Bump mtime_ns forward so the stat-based key is guaranteed to differ
    # even on coarse-resolution filesystems.
    st = os.stat(out_dir / "meta.json")
    os.utime(out_dir / "meta.json",
             ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    save_indices(
        {"metric": "cosine",
         "views": {"intent": {"rows": [{"chunk_id": "z", "text": "new"}],
                              "index": None},
                   "impl": {"rows": [], "index": None}}},
        str(out_dir))
    reloaded = load_indices(str(out_dir))
    assert reloaded is not first
    assert reloaded["views"]["intent"]["rows"] == [{"chunk_id": "z", "text": "new"}]


def test_load_indices_use_cache_false_forces_fresh_read(tmp_path):
    out_dir = tmp_path / "idx"
    save_indices(
        {"metric": "cosine",
         "views": {"intent": {"rows": [{"chunk_id": "a", "text": "x"}],
                              "index": None},
                   "impl": {"rows": [], "index": None}}},
        str(out_dir))
    cached = load_indices(str(out_dir))
    fresh = load_indices(str(out_dir), use_cache=False)
    assert fresh is not cached
    assert fresh["views"]["intent"]["rows"] == cached["views"]["intent"]["rows"]


def test_save_indices_handles_missing_view_subtree(tmp_path):
    indices = {"metric": "cosine", "views": {}}
    out_dir = tmp_path / "empty_idx"
    save_indices(indices, str(out_dir))
    assert (out_dir / "meta.json").exists()
    loaded = load_indices(str(out_dir))
    assert loaded["views"]["intent"]["rows"] == []
    assert loaded["views"]["impl"]["rows"] == []


# ---------------------------------------------------------------------------
# Graph helpers (gated on networkx)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _NX, reason="networkx not installed")
def test_save_and_load_graph_json_roundtrips_nodes_and_edges(tmp_path):
    import networkx as nx
    G = nx.DiGraph()
    G.add_node("a", kind="module")
    G.add_node("b", kind="function")
    G.add_edge("a", "b", relation="contains")

    p = tmp_path / "graph.json"
    save_graph_json(G, str(p))
    assert p.exists()
    H = load_graph_json(str(p))
    assert set(H.nodes()) == {"a", "b"}
    assert H.has_edge("a", "b")
    assert H.nodes["a"]["kind"] == "module"


@pytest.mark.skipif(not _NX, reason="networkx not installed")
def test_load_graph_json_supports_links_key_for_backwards_compat(tmp_path):
    # Older artifacts use "links" instead of "edges" -- both must load.
    data = {
        "directed": True, "multigraph": False, "graph": {},
        "nodes": [{"id": "a"}, {"id": "b"}],
        "links": [{"source": "a", "target": "b"}],
    }
    p = tmp_path / "old_graph.json"
    p.write_text(json.dumps(data))
    H = load_graph_json(str(p))
    assert H.has_edge("a", "b")
