"""Tests for the User Activity subsystem (run store + recorder + API)."""

from __future__ import annotations

import asyncio
import json

from cgx.activity import RunRecord, RunStore, record_run


def _mem_store() -> RunStore:
    return RunStore(":memory:")


# --------------------------------------------------------------------------
# Store: record / recent / get / summary
# --------------------------------------------------------------------------
def test_store_record_recent_and_get():
    store = _mem_store()
    rid = store.record(RunRecord(kind="ask", model="qwen", n_sources=3,
                                 n_citations=2, grounded=True, cost_usd=0.01,
                                 tokens_total=120, question="what is x?"))
    rows = store.recent()
    assert len(rows) == 1 and rows[0]["run_id"] == rid
    assert rows[0]["grounded"] is True and rows[0]["kind"] == "ask"

    got = store.get(rid)
    assert got and got["model"] == "qwen" and got["n_citations"] == 2
    assert store.get("nope") is None


def test_store_recent_filters_and_summary():
    store = _mem_store()
    store.record(RunRecord(kind="ask", cost_usd=0.02, tokens_total=100))
    store.record(RunRecord(kind="plan", cost_usd=0.05, tokens_total=200,
                           status="blocked"))
    store.record(RunRecord(kind="ask", cost_usd=0.03, tokens_total=50))

    assert len(store.recent(kind="ask")) == 2
    assert len(store.recent(status="blocked")) == 1

    s = store.summary()
    assert s["total"] == 3
    assert abs(s["cost_usd"] - 0.10) < 1e-9
    assert s["tokens_total"] == 350
    assert s["errors"] == 1  # the blocked plan
    assert s["by_kind"]["ask"]["runs"] == 2


# --------------------------------------------------------------------------
# Recorder: best-effort extraction from meta + never raises
# --------------------------------------------------------------------------
def test_record_run_extracts_from_meta():
    store = _mem_store()
    meta = {"model": "gpt", "prompt_version": "v1", "confidence": 0.9,
            "citations": [{"chunk_id": "a"}, {"chunk_id": "b"}],
            "tokens_total": 42, "cost_usd": 0.004, "intent": "explain"}
    record_run(kind="ask", run_id="run-x", meta=meta,
               sources=[{"chunk_id": "a"}], question="q", model="gpt",
               owner="alice", latency_ms=12.5, store=store)
    got = store.get("run-x")
    assert got["prompt_version"] == "v1" and got["n_citations"] == 2
    assert got["grounded"] is True and got["owner"] == "alice"
    assert got["labels"]["intent"] == "explain"


def test_record_run_swallows_errors():
    class Boom(RunStore):
        def record(self, rec):  # type: ignore[override]
            raise RuntimeError("db down")

    # Must not raise despite the failing store.
    record_run(kind="ask", run_id="r", store=Boom(":memory:"))


# --------------------------------------------------------------------------
# Agent-turn recorder: aggregate LLM_CALL facts into a kind="agent" run
# --------------------------------------------------------------------------
def test_record_agent_turn_aggregates_llm_facts():
    from cgx.activity import record_agent_turn
    from cgx.session.models import Fact, FactKind, Session

    store = _mem_store()
    session = Session.new(original_objective="add tests",
                          project_root="/tmp/proj")
    facts = [
        Fact.new(session.session_id, FactKind.LLM_CALL,
                 {"model": "qwen", "tokens_in": 10, "tokens_out": 5,
                  "tokens_total": 15, "cost_usd": 0.001}),
        Fact.new(session.session_id, FactKind.LLM_CALL,
                 {"model": "qwen", "tokens_in": 20, "tokens_out": 7,
                  "tokens_total": 27, "cost_usd": 0.002}),
    ]
    record_agent_turn(session=session, run_id="turn-1", facts=facts,
                      latency_ms=42.0, store=store)

    rows = store.recent(kind="agent")
    assert len(rows) == 1
    got = rows[0]
    assert got["kind"] == "agent" and got["run_id"] == "turn-1"
    assert got["project_root"] == "/tmp/proj" and got["model"] == "qwen"
    assert got["tokens_total"] == 42 and got["tokens_in"] == 30
    assert abs(got["cost_usd"] - 0.003) < 1e-9
    assert got["labels"]["mode"] == "explore"


def test_record_agent_turn_no_facts_is_zeroed_and_safe():
    from cgx.activity import record_agent_turn
    from cgx.session.models import Session

    store = _mem_store()
    session = Session.new(original_objective="noop", project_root="/tmp/p")
    record_agent_turn(session=session, run_id="turn-empty", facts=[],
                      store=store)
    got = store.get("turn-empty")
    assert got and got["kind"] == "agent"
    assert got["tokens_total"] == 0 and got["cost_usd"] == 0.0


# --------------------------------------------------------------------------
# API routes
# --------------------------------------------------------------------------
async def _asgi(app, method, path):
    body_parts: list = []
    scope = {"type": "http", "method": method, "path": path.split("?")[0],
             "raw_path": path.encode(), "query_string": (
                 path.split("?", 1)[1].encode() if "?" in path else b""),
             "headers": []}

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    status = {}

    async def send(msg):
        if msg["type"] == "http.response.start":
            status["code"] = msg["status"]
        elif msg["type"] == "http.response.body":
            body_parts.append(msg.get("body", b""))

    await app(scope, receive, send)
    return {"status": status.get("code"), "body": b"".join(body_parts)}


def test_activity_routes(monkeypatch):
    import cgx.activity.store as store_mod
    from cgx.webui.server import create_app

    store = _mem_store()
    store.record(RunRecord(kind="ask", run_id="run-1", model="qwen",
                           cost_usd=0.01, tokens_total=10))
    monkeypatch.setattr(store_mod, "_DEFAULT", store)
    app = create_app()

    r = asyncio.run(_asgi(app, "GET", "/api/activity/runs?kind=ask"))
    body = json.loads(r["body"])
    assert r["status"] == 200 and body["count"] == 1
    assert body["runs"][0]["run_id"] == "run-1"

    s = asyncio.run(_asgi(app, "GET", "/api/activity/summary"))
    assert json.loads(s["body"])["total"] == 1

    d = asyncio.run(_asgi(app, "GET", "/api/activity/runs/run-1"))
    detail = json.loads(d["body"])
    assert detail["run"]["run_id"] == "run-1"
    assert detail["feedback"] == [] and detail["alerts"] == []

    miss = asyncio.run(_asgi(app, "GET", "/api/activity/runs/nope"))
    assert miss["status"] == 404
