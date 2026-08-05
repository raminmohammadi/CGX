"""Tests for /api/metrics, RED metrics, and the request-id middleware.

``httpx`` (and thus Starlette's ``TestClient``) is not a core dependency,
so the ASGI app is driven directly through a minimal scope/receive/send
harness -- no extra packages required.
"""

from __future__ import annotations

import asyncio
from typing import Dict, Optional

import pytest

from cgx import metrics as m
from cgx.webui.server import create_app


async def _asgi_get(app, path: str, headers: Optional[Dict[str, str]] = None):
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(k.lower().encode(), v.encode())
                    for k, v in (headers or {}).items()],
        "client": ("test", 123),
        "server": ("test", 80),
        "scheme": "http",
        "root_path": "",
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    out = {"status": None, "headers": {}, "body": b""}

    async def send(message):
        if message["type"] == "http.response.start":
            out["status"] = message["status"]
            out["headers"] = {k.decode().lower(): v.decode()
                              for k, v in message["headers"]}
        elif message["type"] == "http.response.body":
            out["body"] += message.get("body", b"")

    await app(scope, receive, send)
    return out


@pytest.fixture()
def app():
    m.reset_for_tests()
    yield create_app()
    m.reset_for_tests()


def _get(app, path, headers=None):
    return asyncio.run(_asgi_get(app, path, headers))


def test_metrics_endpoint_content_type(app):
    r = _get(app, "/api/metrics")
    assert r["status"] == 200
    assert r["headers"]["content-type"].startswith("text/plain; version=0.0.4")


def test_request_id_header_assigned(app):
    r = _get(app, "/api/metrics")
    assert r["headers"].get("x-request-id")


def test_request_id_header_propagated(app):
    r = _get(app, "/api/metrics", headers={"X-Request-ID": "fixed-id-123"})
    assert r["headers"].get("x-request-id") == "fixed-id-123"


def test_red_metrics_recorded(app):
    _get(app, "/api/status")
    body = _get(app, "/api/metrics")["body"].decode()
    assert "cgx_http_requests_total" in body
    assert "cgx_http_request_duration_ms_count" in body
    assert 'route="/api/status"' in body


def test_red_metrics_low_cardinality_uses_template(app):
    _get(app, "/api/does-not-exist-xyz")
    body = _get(app, "/api/metrics")["body"].decode()
    assert "does-not-exist-xyz" not in body
