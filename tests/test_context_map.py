"""Tests for cgx.answer.context_map (tiered SLM source builder)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from cgx.answer.context_map import (
    build_tiered_context,
    classify_hits,
    format_neighbor_stub,
    load_records_by_id,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _make_hit(cid: str, *, depth: int = 0, score: float = 1.0) -> Dict[str, Any]:
    prov: Dict[str, Any] = {}
    if depth:
        prov["graph_depth"] = depth
    return {"chunk_id": cid, "score": score, "provenance": prov}


def _make_row(cid: str, text: str) -> Dict[str, Any]:
    return {"chunk_id": cid, "text": text, "view": "intent"}


def _make_record(cid: str, **fields: Any) -> Dict[str, Any]:
    base = {"id": cid, "type": "function", "name": cid.split("::")[-1]}
    base.update(fields)
    return base


@pytest.fixture
def synth_corpus():
    """Three primary chunks + two graph-expanded neighbors."""
    cmap = {
        "src/a.py::function::primary_one": _make_row(
            "src/a.py::function::primary_one",
            "\n".join(f"primary_one_line_{i}" for i in range(30)),
        ),
        "src/a.py::function::primary_two": _make_row(
            "src/a.py::function::primary_two",
            "\n".join(f"primary_two_line_{i}" for i in range(30)),
        ),
        "src/b.py::function::primary_three": _make_row(
            "src/b.py::function::primary_three",
            "primary_three_body",
        ),
        "src/c.py::function::neighbor_one": _make_row(
            "src/c.py::function::neighbor_one",
            "neighbor_one_body_text",
        ),
        "src/c.py::method::Helper.neighbor_two": _make_row(
            "src/c.py::method::Helper.neighbor_two",
            "neighbor_two_body_text",
        ),
    }
    records = {
        "src/a.py::function::primary_one": _make_record(
            "src/a.py::function::primary_one",
            signature="(x: int) -> int",
            doc_first_sentence="Add one to x.",
        ),
        "src/c.py::function::neighbor_one": _make_record(
            "src/c.py::function::neighbor_one",
            signature="(payload: dict) -> None",
            doc_first_sentence="Persist payload to disk.",
        ),
        "src/c.py::method::Helper.neighbor_two": _make_record(
            "src/c.py::method::Helper.neighbor_two",
            signature="(self, kind: str) -> str",
            doc_first_sentence="Resolve a kind to its display label.",
            class_name="Helper",
        ),
    }
    hits = [
        _make_hit("src/a.py::function::primary_one", score=2.0),
        _make_hit("src/a.py::function::primary_two", score=1.5),
        _make_hit("src/b.py::function::primary_three", score=1.0),
        _make_hit("src/c.py::function::neighbor_one", depth=1, score=0.5),
        _make_hit("src/c.py::method::Helper.neighbor_two", depth=2, score=0.3),
    ]
    return cmap, records, hits


# ---------------------------------------------------------------------------
# classify_hits
# ---------------------------------------------------------------------------
def test_classify_hits_splits_by_graph_depth():
    hits = [
        _make_hit("a", depth=0),
        _make_hit("b", depth=1),
        _make_hit("c"),  # no provenance
        _make_hit("d", depth=2),
    ]
    primary, neighbors = classify_hits(hits)
    assert [h["chunk_id"] for h in primary] == ["a", "c"]
    assert [h["chunk_id"] for h in neighbors] == ["b", "d"]


def test_classify_hits_handles_non_int_depth():
    # Non-integer / missing provenance must not raise.
    hits = [{"chunk_id": "x", "provenance": {"graph_depth": "1"}},
            {"chunk_id": "y", "provenance": None},
            {"chunk_id": "z"}]
    primary, neighbors = classify_hits(hits)
    assert [h["chunk_id"] for h in primary] == ["x", "y", "z"]
    assert neighbors == []


# ---------------------------------------------------------------------------
# format_neighbor_stub
# ---------------------------------------------------------------------------
def test_format_neighbor_stub_full():
    rec = {
        "name": "encode",
        "signature": "(self, x: int) -> bytes",
        "doc_first_sentence": "Encode x as bytes.",
        "class_name": "Codec",
    }
    out = format_neighbor_stub(rec, "encode")
    assert out.startswith("Codec.encode")
    assert "(self, x: int) -> bytes" in out
    assert "Encode x as bytes." in out


def test_format_neighbor_stub_drops_missing_components():
    rec = {"name": "save", "signature": "", "doc_first_sentence": "", "class_name": ""}
    assert format_neighbor_stub(rec, "save") == "save"
    rec = {"name": "save", "signature": "(p)", "doc_first_sentence": ""}
    assert format_neighbor_stub(rec, "save") == "save(p)"
    rec = {"name": "save", "signature": "", "doc_first_sentence": "Persist data."}
    assert format_neighbor_stub(rec, "save") == "save -- Persist data."


def test_format_neighbor_stub_falls_back_to_symbol():
    assert format_neighbor_stub(None, "lonely") == "lonely"
    assert format_neighbor_stub({}, "lonely") == "lonely"


# ---------------------------------------------------------------------------
# load_records_by_id
# ---------------------------------------------------------------------------
def test_load_records_by_id_roundtrip(tmp_path: Path):
    p = tmp_path / "records.jsonl"
    rows = [
        {"id": "src/x.py::function::a", "signature": "(z)"},
        {"id": "src/x.py::function::b"},
        {"not_a_record": True},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    by_id = load_records_by_id(str(p))
    assert set(by_id) == {"src/x.py::function::a", "src/x.py::function::b"}
    assert by_id["src/x.py::function::a"]["signature"] == "(z)"


def test_load_records_by_id_missing_path_returns_empty():
    assert load_records_by_id(None) == {}
    assert load_records_by_id("") == {}
    assert load_records_by_id("/no/such/file.jsonl") == {}


# ---------------------------------------------------------------------------
# build_tiered_context -- wiring & invariants
# ---------------------------------------------------------------------------
def test_build_tiered_context_orders_primary_before_neighbor(synth_corpus):
    cmap, records, hits = synth_corpus
    budget = {
        "primary_chars": 200, "neighbor_chars": 120,
        "primary_max": 4, "neighbor_max": 4,
        "total_chars": 10_000,
    }
    out = build_tiered_context(hits, cmap, records, budget=budget)
    tiers = [s["tier"] for s in out]
    # all primaries first, then all neighbors
    split = tiers.index("neighbor") if "neighbor" in tiers else len(tiers)
    assert all(t == "primary" for t in tiers[:split])
    assert all(t == "neighbor" for t in tiers[split:])
    assert tiers.count("primary") == 3
    assert tiers.count("neighbor") == 2


def test_build_tiered_context_neighbor_uses_stub_not_body(synth_corpus):
    cmap, records, hits = synth_corpus
    budget = {
        "primary_chars": 2_000, "neighbor_chars": 400,
        "primary_max": 8, "neighbor_max": 8,
        "total_chars": 20_000,
    }
    out = build_tiered_context(hits, cmap, records, budget=budget)
    neighbors = [s for s in out if s["tier"] == "neighbor"]
    assert neighbors, "expected at least one neighbor stub"
    for s in neighbors:
        # stub should be the formatted signature+doc string, NOT the raw body
        assert "neighbor_one_body_text" not in s["text"]
        assert "neighbor_two_body_text" not in s["text"]
        assert s["signature"]  # carried over from records


def test_build_tiered_context_enforces_total_chars_budget(synth_corpus):
    cmap, records, hits = synth_corpus
    # tiny ceiling forces truncation after the first primary
    budget = {
        "primary_chars": 500, "neighbor_chars": 200,
        "primary_max": 8, "neighbor_max": 8,
        "total_chars": 200,
    }
    out = build_tiered_context(hits, cmap, records, budget=budget)
    assert len(out) >= 1
    assert sum(len(s.get("text") or "") for s in out) <= 500 + 200  # at most one slot beyond cap


def test_build_tiered_context_caps_per_tier(synth_corpus):
    cmap, records, hits = synth_corpus
    budget = {
        "primary_chars": 1_000, "neighbor_chars": 200,
        "primary_max": 1, "neighbor_max": 1,
        "total_chars": 100_000,
    }
    out = build_tiered_context(hits, cmap, records, budget=budget)
    tiers = [s["tier"] for s in out]
    assert tiers.count("primary") == 1
    assert tiers.count("neighbor") == 1


def test_build_tiered_context_neighbor_stub_truncates(synth_corpus):
    cmap, records, hits = synth_corpus
    budget = {
        "primary_chars": 1_000, "neighbor_chars": 12,
        "primary_max": 4, "neighbor_max": 4,
        "total_chars": 100_000,
    }
    out = build_tiered_context(hits, cmap, records, budget=budget)
    for s in out:
        if s["tier"] == "neighbor":
            assert len(s["text"]) <= 12


def test_build_tiered_context_no_neighbors_returns_only_primaries():
    cmap = {"f::function::g": _make_row("f::function::g", "body")}
    records = {"f::function::g": _make_record("f::function::g", signature="(x)")}
    hits = [_make_hit("f::function::g")]
    out = build_tiered_context(
        hits, cmap, records,
        budget={"primary_chars": 100, "neighbor_chars": 50,
                "primary_max": 4, "neighbor_max": 4, "total_chars": 1_000},
    )
    assert len(out) == 1
    assert out[0]["tier"] == "primary"
    assert out[0]["signature"] == "(x)"  # backfilled from record
