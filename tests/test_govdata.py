"""Tests for data governance: PII scan/scrub, retention, erasure + API."""

from __future__ import annotations

import asyncio
import json
import time

from cgx.activity.store import RunRecord, RunStore
from cgx.feedback.store import Feedback, FeedbackStore
from cgx.govdata import (
    GovernanceConfig,
    erase_owner,
    erase_run,
    has_pii,
    purge_expired,
    scan_pii,
    scrub_mapping,
    scrub_pii,
)
from cgx.monitor.alerts import Alert, AlertStore

_SAMPLE = "mail a@b.com ip 10.0.0.1 card 4111 1111 1111 1111 tel +1 415 555 2671"


# --------------------------------------------------------------------------
# PII scan / scrub
# --------------------------------------------------------------------------
def test_scan_pii_non_overlapping():
    findings = {f["type"]: f["count"] for f in scan_pii(_SAMPLE)}
    # Each class counted once -- the card/ip runs are not also counted as phone.
    assert findings == {"email": 1, "card": 1, "ipv4": 1, "phone": 1}
    assert scan_pii("") == [] and not has_pii("nothing here")
    assert has_pii("reach a@b.com")


def test_scrub_pii_and_mapping():
    assert scrub_pii("reach a@b.com at 10.0.0.1") == "reach <email> at <ipv4>"
    assert scrub_pii(None) == "" and scrub_pii(123) == "123"
    nested = {"q": "a@b.com", "xs": ["10.0.0.1", {"k": "b@c.io"}], "n": 5}
    out = scrub_mapping(nested)
    assert out["q"] == "<email>" and out["xs"][0] == "<ipv4>"
    assert out["xs"][1]["k"] == "<email>" and out["n"] == 5


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------
def test_policy_from_env_and_apply(monkeypatch):
    monkeypatch.setenv("CGX_RETENTION_DAYS", "7")
    monkeypatch.setenv("CGX_STORE_FULL_TEXT", "false")
    monkeypatch.setenv("CGX_SCRUB_PII", "on")
    monkeypatch.setenv("CGX_PREVIEW_CAP", "10")
    p = GovernanceConfig.from_env()
    assert p.retention_days == 7 and p.store_full_text is False
    assert p.scrub_pii is True and p.preview_cap == 10
    # Scrub happens before the preview cap.
    assert p.apply_text_policy("mail a@b.com now") == "mail <emai"
    assert GovernanceConfig().apply_text_policy("keep me") == "keep me"


# --------------------------------------------------------------------------
# Per-store purge / delete primitives
# --------------------------------------------------------------------------
def test_store_purge_and_delete():
    now = time.time()
    runs = RunStore(":memory:")
    runs.record(RunRecord(kind="ask", run_id="r_old", owner="alice",
                          created_at=now - 10 * 86400))
    runs.record(RunRecord(kind="ask", run_id="r_new", owner="bob"))
    assert runs.purge(before=now - 5 * 86400) == 1
    assert runs.delete_owner("bob") == 1 and len(runs.recent()) == 0

    fb = FeedbackStore(":memory:")
    fb.record(Feedback(rating="down", run_id="r1", created_at=now - 10 * 86400))
    fb.record(Feedback(rating="up", run_id="r2"))
    assert fb.purge(before=now - 5 * 86400) == 1
    assert fb.delete_run("r2") == 1

    al = AlertStore(":memory:")
    al.record(Alert(code="c", severity="info", message="m", run_id="r1",
                    created_at=now - 10 * 86400))
    al.record(Alert(code="c", severity="info", message="m", run_id="r2"))
    assert al.purge(before=now - 5 * 86400) == 1
    assert al.delete_run("r2") == 1

    from cgx.governance.meter import UsageMeter
    um = UsageMeter(":memory:")
    um.record("alice", tokens_in=1, tokens_out=2, cost_usd=0.1)
    assert um.delete_owner("alice") == 1
    um.record("bob", tokens_in=1, tokens_out=2, cost_usd=0.1)
    assert um.purge(before=now + 1) == 1


# --------------------------------------------------------------------------
# Retention orchestrator (injected stores)
# --------------------------------------------------------------------------
def test_retention_purge_expired_and_erase():
    now = time.time()
    runs = RunStore(":memory:")
    runs.record(RunRecord(kind="ask", run_id="r1", owner="alice",
                          created_at=now - 10 * 86400))
    fb = FeedbackStore(":memory:")
    fb.record(Feedback(rating="down", run_id="r1", created_at=now - 10 * 86400))
    stores = {"activity": runs, "feedback": fb}

    # Zero retention_days disables the sweep entirely.
    assert purge_expired(GovernanceConfig(retention_days=0), stores=stores) == {}
    swept = purge_expired(GovernanceConfig(retention_days=1), stores=stores,
                          now=now)
    assert swept == {"activity": 1, "feedback": 1}

    runs.record(RunRecord(kind="ask", run_id="rx", owner="carol"))
    fb.record(Feedback(rating="down", run_id="rx"))
    assert erase_run("rx", stores=stores) == {"activity": 1, "feedback": 1}
    runs.record(RunRecord(kind="ask", run_id="ry", owner="dave"))
    assert erase_owner("dave", stores=stores) == {"activity": 1}


# --------------------------------------------------------------------------
# Write-path: record_run honours the PII scrub policy
# --------------------------------------------------------------------------
def test_record_run_scrubs_pii():
    from cgx.activity import record_run
    st = RunStore(":memory:")
    record_run(kind="ask", run_id="rr", question="email a@b.com",
               meta={"intent": "reach a@b.com"}, store=st,
               policy=GovernanceConfig(scrub_pii=True))
    got = st.get("rr")
    assert got["question"] == "email <email>"
    assert got["labels"]["intent"] == "reach <email>"


# --------------------------------------------------------------------------
# API routes
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


def test_govdata_routes(monkeypatch):
    from cgx.webui.server import create_app

    now = time.time()
    runs = RunStore(":memory:")
    runs.record(RunRecord(kind="ask", run_id="r1", owner="alice",
                          created_at=now - 10 * 86400))
    monkeypatch.setattr("cgx.govdata.retention._default_stores",
                        lambda: {"activity": runs})
    monkeypatch.delenv("CGX_RETENTION_DAYS", raising=False)
    app = create_app()

    pol = asyncio.run(_asgi(app, "GET", "/api/govdata/policy"))
    assert pol["status"] == 200
    assert json.loads(pol["body"])["retention_days"] == 0

    sc = asyncio.run(_asgi(app, "POST", "/api/govdata/scan",
                           {"text": "reach a@b.com"}))
    sbody = json.loads(sc["body"])
    assert sbody["total"] == 1 and sbody["scrubbed"] == "reach <email>"

    pg = asyncio.run(_asgi(app, "POST", "/api/govdata/purge",
                           {"retention_days": 1}))
    assert json.loads(pg["body"]) == {"ok": True, "deleted": {"activity": 1},
                                      "total": 1}

    runs.record(RunRecord(kind="ask", run_id="ry", owner="dave"))
    er = asyncio.run(_asgi(app, "POST", "/api/govdata/erase", {"owner": "dave"}))
    assert json.loads(er["body"])["deleted"] == {"activity": 1}

    bad = asyncio.run(_asgi(app, "POST", "/api/govdata/erase",
                           {"run_id": "x", "owner": "y"}))
    assert bad["status"] == 422
