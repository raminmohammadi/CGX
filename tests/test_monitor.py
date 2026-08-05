"""Tests for the AIOps monitors (checks + store + Monitor façade)."""

from __future__ import annotations

import asyncio
import json

from cgx import metrics as m
from cgx.monitor import (
    Alert,
    AlertStore,
    Monitor,
    MonitorThresholds,
    check_cost_anomaly,
    check_groundedness,
    check_repair_health,
    check_retrieval_drift,
)

TH = MonitorThresholds()


def _codes(alerts):
    return {a.code for a in alerts}


# --------------------------------------------------------------------------
# Groundedness
# --------------------------------------------------------------------------
def test_groundedness_clean_answer_has_no_alerts():
    answer = {
        "answer_md": "It does X [[a]].",
        "citations": [{"chunk_id": "a"}],
        "confidence": 0.8,
        "debug": {"sources": [{"chunk_id": "a"}, {"chunk_id": "b"}]},
    }
    assert check_groundedness(answer, TH) == []


def test_groundedness_flags_invalid_citation_and_low_confidence():
    answer = {
        "answer_md": "Claim [[ghost]].",
        "citations": [{"chunk_id": "ghost"}],
        "confidence": 0.1,
        "debug": {"sources": [{"chunk_id": "a"}]},
    }
    codes = _codes(check_groundedness(answer, TH))
    assert "citation_invalid" in codes and "low_confidence" in codes


def test_groundedness_abstention_is_single_info_alert():
    answer = {"answer_md": "Not enough context.", "citations": [],
              "confidence": 0.2, "debug": {"sources": [{"chunk_id": "a"}]}}
    alerts = check_groundedness(answer, TH)
    assert [a.code for a in alerts] == ["answer_abstained"]
    assert alerts[0].severity == "info"


def test_groundedness_ungrounded_when_confident_but_uncited():
    answer = {"answer_md": "Confident claim.", "citations": [],
              "confidence": 0.9, "debug": {"sources": [{"chunk_id": "a"}]}}
    assert "ungrounded_answer" in _codes(check_groundedness(answer, TH))


# --------------------------------------------------------------------------
# Repair health
# --------------------------------------------------------------------------
def test_repair_failed_when_not_overall_ok():
    report = {"attempts": 1, "summary": {"overall_ok": False,
              "n_patches_failed": 1, "n_syntax_failed": 0}}
    codes = _codes(check_repair_health(report, TH))
    assert "repair_failed" in codes


def test_repair_churn_when_converged_at_budget():
    report = {"attempts": 2, "summary": {"overall_ok": True, "empty_plan": False}}
    assert _codes(check_repair_health(report, TH)) == {"repair_loop_churn"}


def test_repair_empty_plan_and_error():
    empty = {"attempts": 0, "summary": {"overall_ok": False, "empty_plan": True}}
    assert "empty_plan" in _codes(check_repair_health(empty, TH))
    assert _codes(check_repair_health({"error": "boom"}, TH)) == {"repair_error"}


# --------------------------------------------------------------------------
# Retrieval drift + cost
# --------------------------------------------------------------------------
def test_embedding_model_drift_short_circuits_score_check():
    cur = {"embed_model": "m2", "mean_top_score": 0.1}
    base = {"embed_model": "m1", "mean_top_score": 0.9}
    assert _codes(check_retrieval_drift(cur, base, TH)) == {"embedding_model_drift"}


def test_retrieval_score_drift():
    cur = {"embed_model": "m", "mean_top_score": 0.5}
    base = {"embed_model": "m", "mean_top_score": 0.9}
    assert "retrieval_score_drift" in _codes(check_retrieval_drift(cur, base, TH))


def test_cost_spike_and_error_rate():
    window = {"cost_usd": 4.0, "calls": 10, "errors": 5}
    base = {"cost_usd": 1.0}
    codes = _codes(check_cost_anomaly(window, base, TH))
    assert "cost_spike" in codes and "provider_error_rate" in codes


# --------------------------------------------------------------------------
# Store + Monitor façade
# --------------------------------------------------------------------------
def test_alert_store_roundtrip_and_filter():
    store = AlertStore(":memory:")
    store.record(Alert(code="low_confidence", severity="warning", message="x"))
    store.record(Alert(code="cost_spike", severity="warning", message="y"))
    assert len(store.recent()) == 2
    only = store.recent(code="cost_spike")
    assert len(only) == 1 and only[0]["code"] == "cost_spike"
    store.close()


def test_monitor_persists_and_emits_metrics():
    m.reset_for_tests()
    mon = Monitor(AlertStore(":memory:"))
    answer = {"answer_md": "Claim [[ghost]].", "citations": [{"chunk_id": "ghost"}],
              "confidence": 0.1, "debug": {"sources": [{"chunk_id": "a"}]}}
    alerts = mon.observe_answer(answer, run_id="run-1")
    assert alerts and mon.recent(code="citation_invalid")
    out = m.render_prometheus()
    assert "cgx_monitor_alerts_total" in out
    assert 'code="citation_invalid"' in out
    mon.close()


# --------------------------------------------------------------------------
# Read API route (/api/monitor/alerts)
# --------------------------------------------------------------------------
async def _asgi_get(app, path):
    scope = {"type": "http", "http_version": "1.1", "method": "GET",
             "path": path.split("?")[0], "raw_path": path.encode(),
             "query_string": path.split("?", 1)[1].encode() if "?" in path else b"",
             "headers": [], "client": ("test", 1), "server": ("test", 80),
             "scheme": "http", "root_path": ""}

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    out = {"status": None, "body": b""}

    async def send(message):
        if message["type"] == "http.response.start":
            out["status"] = message["status"]
        elif message["type"] == "http.response.body":
            out["body"] += message.get("body", b"")

    await app(scope, receive, send)
    return out


def test_monitor_alerts_route(monkeypatch):
    import cgx.monitor.monitor as mod
    from cgx.webui.server import create_app

    mon = Monitor(AlertStore(":memory:"))
    mon.observe_cost({"cost_usd": 9.0}, {"cost_usd": 1.0})
    monkeypatch.setattr(mod, "_DEFAULT", mon)

    out = asyncio.run(_asgi_get(create_app(), "/api/monitor/alerts?code=cost_spike"))
    assert out["status"] == 200
    body = json.loads(out["body"])
    assert body["count"] == 1
    assert body["alerts"][0]["code"] == "cost_spike"
    mon.close()
