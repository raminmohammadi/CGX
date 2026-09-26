"""Tests for cgx.answer.context_map (visibility-ladder SLM source builder)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from cgx.answer.context_map import (
    build_tiered_context,
    classify_hits,
    decide_visibility,
    format_neighbor_stub,
    load_records_by_id,
)


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------
def _make_hit(cid: str, *, depth: int = 0, score: float = 1.0,
              **prov: Any) -> Dict[str, Any]:
    p: Dict[str, Any] = dict(prov)
    if depth:
        p["graph_depth"] = depth
    return {"chunk_id": cid, "score": score, "provenance": p}


def _make_row(cid: str, text: str) -> Dict[str, Any]:
    return {"chunk_id": cid, "text": text, "view": "intent"}


def _make_record(cid: str, **fields: Any) -> Dict[str, Any]:
    base = {"id": cid, "type": "function", "name": cid.split("::")[-1]}
    base.update(fields)
    return base


def _ranked_hits(n: int) -> list:
    """n hits, RRF-ordered (descending score), no extra provenance signals."""
    return [_make_hit(f"src/m.py::function::f{i}", score=float(n - i))
            for i in range(n)]


_BUDGET = {
    "primary_chars": 300, "long_chars": 150, "neighbor_chars": 80,
    "primary_max": 8, "neighbor_max": 8, "total_chars": 100_000,
}


# ---------------------------------------------------------------------------
# classify_hits (retained legacy graph-depth split)
# ---------------------------------------------------------------------------
def test_classify_hits_splits_by_graph_depth():
    hits = [
        _make_hit("a", depth=0),
        _make_hit("b", depth=1),
        _make_hit("c"),
        _make_hit("d", depth=2),
    ]
    primary, neighbors = classify_hits(hits)
    assert [h["chunk_id"] for h in primary] == ["a", "c"]
    assert [h["chunk_id"] for h in neighbors] == ["b", "d"]


def test_classify_hits_handles_non_int_depth():
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
# decide_visibility -- the ladder
# ---------------------------------------------------------------------------
def test_decide_visibility_empty():
    assert decide_visibility([], budget=_BUDGET) == []


def test_decide_visibility_rank1_always_full():
    leveled = decide_visibility(_ranked_hits(10), budget=_BUDGET)
    assert leveled[0][1] == "full"


def test_decide_visibility_bands_by_quantile():
    # n=10: full_cut=2, long_cut=5, short_cut=8 (top 20/30/30/rest).
    levels = [lvl for _, lvl in decide_visibility(_ranked_hits(10), budget=_BUDGET)]
    assert levels[0] == "full" and levels[1] == "full"
    assert levels[2] == "long" and levels[4] == "long"
    assert levels[5] == "short" and levels[7] == "short"
    assert levels[8] == "hide" and levels[9] == "hide"


def test_decide_visibility_strong_signal_bumps_up():
    hits = _ranked_hits(10)
    hits[5]["provenance"]["symbol_match"] = True   # base short -> long
    hits[6]["provenance"]["intent_rank"] = 1       # base short -> long
    levels = [lvl for _, lvl in decide_visibility(hits, budget=_BUDGET)]
    assert levels[5] == "long"
    assert levels[6] == "long"


def test_decide_visibility_deep_or_demoted_bumps_down():
    hits = _ranked_hits(10)
    hits[2]["provenance"]["graph_depth"] = 3       # base long -> short
    hits[3]["provenance"]["scope_demoted"] = True  # base long -> short
    levels = [lvl for _, lvl in decide_visibility(hits, budget=_BUDGET)]
    assert levels[2] == "short"
    assert levels[3] == "short"


def test_decide_visibility_primary_max_caps_full():
    hits = _ranked_hits(10)
    for i in range(5):  # first five all want full via symbol_match
        hits[i]["provenance"]["symbol_match"] = True
    levels = [lvl for _, lvl in decide_visibility(hits, budget={**_BUDGET,
                                                                "primary_max": 2})]
    assert levels.count("full") == 2  # overflow demoted to long
    assert levels[2] == "long"


# ---------------------------------------------------------------------------
# build_tiered_context -- wiring & invariants
# ---------------------------------------------------------------------------
def _corpus(n: int, body: str = None):
    hits = _ranked_hits(n)
    cmap = {h["chunk_id"]: _make_row(
        h["chunk_id"],
        body if body is not None else "\n".join(f"body_{i}_{j}" for j in range(40)))
        for i, h in enumerate(hits)}
    records = {h["chunk_id"]: _make_record(
        h["chunk_id"], signature="(x)", doc_first_sentence="Does the thing.")
        for h in hits}
    return cmap, records, hits


def test_build_tiered_context_levels_and_ordering():
    cmap, records, hits = _corpus(10)
    out = build_tiered_context(hits, cmap, records, budget=_BUDGET)
    tiers = [s["tier"] for s in out]
    # only ladder levels appear; 'hide' never rendered, legacy names gone.
    assert set(tiers) <= {"full", "long", "short"}
    # most-visible first
    order = {"full": 0, "long": 1, "short": 2}
    assert tiers == sorted(tiers, key=lambda t: order[t])
    # rank-1 is full and rendered first with its real body.
    assert out[0]["tier"] == "full"
    assert out[0]["chunk_id"] == hits[0]["chunk_id"]
    assert "body_0_" in out[0]["text"]
    # hidden tail dropped -> fewer sources than hits.
    assert len(out) < len(hits)


def test_build_tiered_context_short_uses_stub_not_body():
    cmap, records, hits = _corpus(10, body="UNIQUE_BODY_MARKER")
    out = build_tiered_context(hits, cmap, records, budget=_BUDGET)
    shorts = [s for s in out if s["tier"] == "short"]
    assert shorts, "expected some short stubs at this corpus size"
    for s in shorts:
        assert "UNIQUE_BODY_MARKER" not in s["text"]  # stub, not raw body
        assert s["signature"]  # carried from records


def test_build_tiered_context_full_and_long_carry_bodies():
    cmap, records, hits = _corpus(10)
    out = build_tiered_context(hits, cmap, records, budget=_BUDGET)
    longs = [s for s in out if s["tier"] == "long"]
    assert longs
    # long bodies are shorter than full bodies (mid-window) but non-empty.
    assert all(s["text"] for s in longs)
    full = next(s for s in out if s["tier"] == "full")
    assert len(full["text"]) >= max(len(s["text"]) for s in longs)


def test_build_tiered_context_enforces_total_chars_budget():
    cmap, records, hits = _corpus(10, body="X" * 500)
    budget = {**_BUDGET, "primary_chars": 500, "long_chars": 300,
              "total_chars": 600}
    out = build_tiered_context(hits, cmap, records, budget=budget)
    assert len(out) >= 1
    total = sum(len(s.get("text") or "") for s in out)
    assert total <= 600 + 500  # at most one slot beyond the cap


def test_build_tiered_context_single_hit_is_full_with_backfilled_sig():
    hits = [_make_hit("f::function::g")]
    cmap = {"f::function::g": _make_row("f::function::g", "body")}
    records = {"f::function::g": _make_record("f::function::g", signature="(x)")}
    out = build_tiered_context(hits, cmap, records, budget=_BUDGET)
    assert len(out) == 1
    assert out[0]["tier"] == "full"
    assert out[0]["signature"] == "(x)"  # backfilled from record


def test_build_tiered_context_empty_hits():
    assert build_tiered_context([], {}, {}, budget=_BUDGET) == []
