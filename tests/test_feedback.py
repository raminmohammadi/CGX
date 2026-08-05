"""Tests for the feedback loop (store + flywheel + API route)."""

from __future__ import annotations

import asyncio
import json

from cgx.feedback import (
    Feedback,
    FeedbackStore,
    export_eval_candidates,
    unify_with_lessons,
)


def _store() -> FeedbackStore:
    return FeedbackStore(":memory:")


# --------------------------------------------------------------------------
# Store round-trip + filters + stats
# --------------------------------------------------------------------------
def test_store_record_and_recent_filters():
    st = _store()
    st.record(Feedback(rating="up", run_id="run_a", kind="ask", question="q1"))
    st.record(Feedback(rating="down", run_id="run_b", kind="plan",
                       question="t1", comment="wrong file"))

    assert len(st.recent()) == 2
    assert len(st.recent(rating="down")) == 1
    assert st.recent(rating="down")[0]["comment"] == "wrong file"
    assert len(st.recent(kind="ask")) == 1
    assert st.recent(run_id="run_b")[0]["kind"] == "plan"
    st.close()


def test_store_stats_aggregation():
    st = _store()
    st.record(Feedback(rating="up", kind="ask"))
    st.record(Feedback(rating="up", kind="ask"))
    st.record(Feedback(rating="down", kind="plan"))
    stats = st.stats()
    assert stats["up"] == 2 and stats["down"] == 1 and stats["total"] == 3
    assert abs(stats["satisfaction"] - (2 / 3)) < 1e-9
    assert stats["by_kind"]["ask"]["up"] == 2
    assert stats["by_kind"]["plan"]["down"] == 1
    st.close()


# --------------------------------------------------------------------------
# Flywheel: export negatives as eval candidates
# --------------------------------------------------------------------------
def test_export_eval_candidates_shape_and_idempotency(tmp_path):
    st = _store()
    st.record(Feedback(rating="up", kind="ask", question="good"))
    st.record(Feedback(rating="down", kind="ask", question="ask q",
                       comment="missed a source"))
    st.record(Feedback(rating="down", kind="plan", question="plan task"))
    out = tmp_path / "cands.jsonl"

    res = export_eval_candidates(store=st, out_path=out)
    assert res["candidates"] == 2 and res["written"] == 2

    rows = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
    by_kind = {r["kind"]: r for r in rows}
    assert by_kind["ask"]["query"] == "ask q"
    assert by_kind["ask"]["source"] == "user_feedback"
    assert by_kind["plan"]["task"] == "plan task"

    # Second run must not duplicate already-exported rows.
    res2 = export_eval_candidates(store=st, out_path=out)
    assert res2["written"] == 0 and res2["skipped"] == 2
    st.close()


# --------------------------------------------------------------------------
# Flywheel: unify feedback + lessons
# --------------------------------------------------------------------------
def test_unify_with_lessons(tmp_path):
    from cgx.session import lessons as _lessons

    st = _store()
    st.record(Feedback(rating="down", kind="ask", run_id="r1",
                       comment="bad answer"))
    lp = tmp_path / "lessons.jsonl"
    _lessons.record_lesson(
        trigger_signature="ImportError: foo", classification="missing_dep",
        applied_fix={"strategy": "pin"}, scope={"stack": ["python"]},
        session_id="s1", path=lp)

    unified = unify_with_lessons(store=st, lessons_path=lp)
    assert unified["stats"]["down"] == 1
    assert unified["lessons_count"] == 1
    types = {s["type"] for s in unified["signals"]}
    assert types == {"feedback", "lesson"}
    st.close()


# --------------------------------------------------------------------------
# Route: /api/feedback (POST + GET + stats)
# --------------------------------------------------------------------------
async def _asgi(app, method, path, body=None):
    raw = json.dumps(body).encode() if body is not None else b""
    scope = {"type": "http", "http_version": "1.1", "method": method,
             "path": path.split("?")[0], "raw_path": path.encode(),
             "query_string": path.split("?", 1)[1].encode() if "?" in path else b"",
             "headers": [(b"content-type", b"application/json")],
             "client": ("test", 1), "server": ("test", 80),
             "scheme": "http", "root_path": ""}

    async def receive():
        return {"type": "http.request", "body": raw, "more_body": False}

    out = {"status": None, "body": b""}

    async def send(message):
        if message["type"] == "http.response.start":
            out["status"] = message["status"]
        elif message["type"] == "http.response.body":
            out["body"] += message.get("body", b"")

    await app(scope, receive, send)
    return out


def test_feedback_route_roundtrip(monkeypatch):
    import cgx.feedback.store as store_mod
    from cgx.webui.server import create_app

    st = FeedbackStore(":memory:")
    monkeypatch.setattr(store_mod, "_DEFAULT", st)
    app = create_app()

    bad = asyncio.run(_asgi(app, "POST", "/api/feedback", {"rating": "meh"}))
    assert bad["status"] == 422

    ok = asyncio.run(_asgi(app, "POST", "/api/feedback",
                           {"rating": "down", "run_id": "run_x", "kind": "ask",
                            "question": "q", "comment": "off-topic"}))
    assert ok["status"] == 200
    assert json.loads(ok["body"])["feedback_id"].startswith("fb_")

    listed = asyncio.run(_asgi(app, "GET", "/api/feedback?rating=down"))
    body = json.loads(listed["body"])
    assert body["count"] == 1 and body["feedback"][0]["run_id"] == "run_x"

    stats = asyncio.run(_asgi(app, "GET", "/api/feedback/stats"))
    assert json.loads(stats["body"])["down"] == 1
    st.close()
