"""Tests for the Admin logs/trace subsystem (metrics snapshot + routes)."""

from __future__ import annotations

import asyncio
import json

from cgx.metrics import MetricsRegistry


# --------------------------------------------------------------------------
# Metrics snapshot: structured, non-Prometheus view of every series
# --------------------------------------------------------------------------
def test_metrics_snapshot_shape():
    reg = MetricsRegistry()
    reg.inc("reqs", 2, route="/x", status="200")
    reg.inc("reqs", 1, route="/x", status="500")
    reg.set_gauge("ready", 1)
    reg.observe("dur_ms", 12.0, route="/x")

    snap = reg.snapshot()
    assert set(snap) == {"counters", "gauges", "histograms"}

    reqs = {tuple(sorted(c["labels"].items())): c["value"]
            for c in snap["counters"] if c["name"] == "reqs"}
    assert reqs[(("route", "/x"), ("status", "200"))] == 2.0
    assert reqs[(("route", "/x"), ("status", "500"))] == 1.0

    gauge = next(g for g in snap["gauges"] if g["name"] == "ready")
    assert gauge["value"] == 1.0

    hist = next(h for h in snap["histograms"] if h["name"] == "dur_ms")
    assert hist["count"] == 1 and hist["sum"] == 12.0
    assert hist["buckets"][-1][0] == "+Inf"


# --------------------------------------------------------------------------
# Minimal ASGI driver (no httpx dependency)
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


# --------------------------------------------------------------------------
# /api/admin/logs: reads project agent.log, redacts + filters, newest first
# --------------------------------------------------------------------------
def test_admin_logs_route_redacts_and_filters(tmp_path):
    from urllib.parse import quote

    from cgx.webui.server import create_app

    log_dir = tmp_path / ".cgx"
    log_dir.mkdir(parents=True)
    log = log_dir / "agent.log"
    log.write_text(
        json.dumps({"event": "trace_enter", "ts": 100.0,
                    "api_key": "sk-ABCDEFGHIJKLMNOP1234", "fn": "ask"}) + "\n" +
        json.dumps({"event": "trace_exit", "ts": 200.0, "elapsed_ms": 5}) + "\n",
        encoding="utf-8",
    )
    app = create_app()

    q = quote(str(tmp_path))
    r = asyncio.run(_asgi(app, "GET",
                          f"/api/admin/logs?project_root={q}&event=trace_enter"))
    body = json.loads(r["body"])
    assert r["status"] == 200 and body["count"] == 1
    row = body["logs"][0]
    assert row["event"] == "trace_enter"
    assert row["api_key"] == "<redacted>"
    assert "sk-ABCDEFGHIJKLMNOP1234" not in r["body"].decode()

    # No filter -> both lines, newest first (trace_exit has larger ts).
    both = asyncio.run(_asgi(app, "GET", f"/api/admin/logs?project_root={q}"))
    logs = json.loads(both["body"])["logs"]
    assert [x["event"] for x in logs] == ["trace_exit", "trace_enter"]


def test_admin_logs_missing_file_is_empty(tmp_path):
    from urllib.parse import quote

    from cgx.webui.server import create_app

    app = create_app()
    q = quote(str(tmp_path / "nope"))
    r = asyncio.run(_asgi(app, "GET", f"/api/admin/logs?project_root={q}"))
    assert r["status"] == 200 and json.loads(r["body"])["count"] == 0


# --------------------------------------------------------------------------
# /api/admin/metrics + /api/admin/overview
# --------------------------------------------------------------------------
def test_admin_metrics_and_overview_routes():
    from cgx import metrics as m
    from cgx.webui.server import create_app

    m.inc("cgx_http_requests_total", 4, route="/x", status="200", method="GET")
    m.inc("cgx_http_requests_total", 1, route="/x", status="500", method="GET")
    app = create_app()

    r = asyncio.run(_asgi(app, "GET", "/api/admin/metrics"))
    snap = json.loads(r["body"])
    assert r["status"] == 200 and set(snap) == {"counters", "gauges", "histograms"}

    o = asyncio.run(_asgi(app, "GET", "/api/admin/overview"))
    ov = json.loads(o["body"])
    assert o["status"] == 200
    assert set(ov) == {"activity", "http", "feedback", "alerts"}
    # The two counters above should be reflected in the http rollup.
    assert ov["http"]["requests"] >= 5
    assert ov["http"]["errors"] >= 1
    assert "by_severity" in ov["alerts"]
