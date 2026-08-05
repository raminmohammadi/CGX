"""Unit tests for the always-on index/retrieval pipeline metrics (Subsystem B)."""

from __future__ import annotations

import pytest

from cgx import metrics as m
from cgx.pipeline.instrument import (
    metered,
    record_index_build,
    record_retrieval,
)


def _reset():
    m.reset_for_tests()


def _counter(snap, name):
    return [s for s in snap["counters"] if s["name"] == name]


def _gauge(snap, name):
    return [s for s in snap["gauges"] if s["name"] == name]


def _hist(snap, name):
    return [s for s in snap["histograms"] if s["name"] == name]


def test_record_index_build_emits_counter_duration_and_gauges():
    _reset()
    result = {
        "incremental": True,
        "counts": {"intent": 42, "impl": 42},
        "parse": {"total_files": 7, "added": 3},
    }
    record_index_build(123.0, "ok", result)
    snap = m.snapshot()

    builds = _counter(snap, "cgx_index_builds_total")
    assert len(builds) == 1
    assert builds[0]["labels"] == {"mode": "incremental", "status": "ok"}
    assert builds[0]["value"] == 1

    dur = _hist(snap, "cgx_index_build_duration_ms")
    assert dur and dur[0]["count"] == 1 and dur[0]["sum"] == 123.0

    assert _gauge(snap, "cgx_index_records")[0]["value"] == 42
    assert _gauge(snap, "cgx_index_files")[0]["value"] == 7


def test_record_index_build_full_mode_and_error_status():
    _reset()
    record_index_build(10.0, "error", None)
    snap = m.snapshot()
    builds = _counter(snap, "cgx_index_builds_total")
    assert builds[0]["labels"] == {"mode": "full", "status": "error"}
    # No result -> no size gauges emitted.
    assert _gauge(snap, "cgx_index_records") == []


def test_record_retrieval_emits_counter_latency_and_candidates():
    _reset()
    record_retrieval(55.0, "ok", {"hits": [{"id": 1}, {"id": 2}, {"id": 3}]})
    snap = m.snapshot()

    q = _counter(snap, "cgx_retrieval_queries_total")
    assert q[0]["labels"] == {"status": "ok"} and q[0]["value"] == 1

    lat = _hist(snap, "cgx_retrieval_latency_ms")
    assert lat and lat[0]["count"] == 1 and lat[0]["sum"] == 55.0

    cand = _hist(snap, "cgx_retrieval_candidates")
    assert cand and cand[0]["count"] == 1 and cand[0]["sum"] == 3.0


def test_metered_success_passes_result_and_status_ok():
    _reset()
    seen = {}

    def recorder(elapsed_ms, status, result):
        seen["status"] = status
        seen["result"] = result
        seen["elapsed_ok"] = elapsed_ms >= 0

    @metered(recorder)
    def build(x):
        return {"counts": {"impl": x}}

    out = build(5)
    assert out == {"counts": {"impl": 5}}
    assert seen["status"] == "ok"
    assert seen["result"] == {"counts": {"impl": 5}}
    assert seen["elapsed_ok"] is True


def test_metered_error_records_status_and_reraises():
    _reset()
    seen = {}

    def recorder(elapsed_ms, status, result):
        seen["status"] = status
        seen["result"] = result

    @metered(recorder)
    def boom():
        raise ValueError("nope")

    with pytest.raises(ValueError):
        boom()
    assert seen["status"] == "error"
    assert seen["result"] is None


def test_metered_cancelled_status():
    _reset()
    seen = {}

    class IndexBuildCancelled(Exception):
        pass

    def recorder(elapsed_ms, status, result):
        seen["status"] = status

    @metered(recorder)
    def cancel():
        raise IndexBuildCancelled("stop")

    with pytest.raises(IndexBuildCancelled):
        cancel()
    assert seen["status"] == "cancelled"
