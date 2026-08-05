"""Tests for liveness/readiness checks and the /healthz//readyz endpoints.

``httpx`` is not a core dependency, so the ASGI app is driven directly
through a minimal scope/receive/send harness (same approach as
``test_webui_metrics``).
"""

from __future__ import annotations

import asyncio
from typing import Dict, Optional

import pytest

from cgx import health, metrics as m
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


def _get(app, path, headers=None):
    return asyncio.run(_asgi_get(app, path, headers))


@pytest.fixture()
def app(tmp_path, monkeypatch):
    # Point config + session DB at a writable tmp dir so readiness is green
    # regardless of the host environment, and stub the provider probe so no
    # network call happens.
    monkeypatch.setenv("CGX_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home" / ".cgx").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "cgx.answer.ollama_discovery.health_check",
        lambda *a, **k: {"ok": True, "base_url": "http://x", "models_count": 0},
    )
    m.reset_for_tests()
    yield create_app()
    m.reset_for_tests()


# --- module-level checks --------------------------------------------------

def test_liveness_is_cheap_and_ok():
    out = health.liveness()
    assert out["status"] == "ok"
    assert out["checks"] == []


def test_config_dir_check_ok(tmp_path, monkeypatch):
    monkeypatch.setenv("CGX_CONFIG_DIR", str(tmp_path / "cfg"))
    r = health.check_config_dir()
    assert r["ok"] is True and r["critical"] is True
    assert r["detail"]["writable"] is True


def test_session_db_check_uses_memory_when_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".cgx").mkdir(parents=True, exist_ok=True)
    r = health.check_session_db()
    assert r["ok"] is True and r["critical"] is True


def test_provider_and_index_are_non_critical(monkeypatch):
    monkeypatch.setattr(
        "cgx.answer.ollama_discovery.health_check",
        lambda *a, **k: {"ok": False, "error": "boom"},
    )
    prov = health.check_provider()
    idx = health.check_index(index_dir="/nonexistent/xyz")
    assert prov["critical"] is False and prov["ok"] is False
    assert idx["critical"] is False and idx["ok"] is False


def test_readiness_green_when_critical_pass(tmp_path, monkeypatch):
    monkeypatch.setenv("CGX_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home" / ".cgx").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        "cgx.answer.ollama_discovery.health_check",
        lambda *a, **k: {"ok": False},  # provider down must NOT block readiness
    )
    report = health.readiness()
    assert report["ready"] is True
    assert report["status"] == "ready"


def test_readiness_red_when_config_dir_unwritable(monkeypatch):
    monkeypatch.setattr(health, "check_config_dir",
                        lambda: health._result("config_dir", False, critical=True))
    report = health.readiness()
    assert report["ready"] is False


# --- HTTP endpoints -------------------------------------------------------

def test_healthz_endpoint(app):
    r = _get(app, "/healthz")
    assert r["status"] == 200
    assert b'"status":"ok"' in r["body"].replace(b" ", b"")


def test_readyz_endpoint_200_when_ready(app):
    r = _get(app, "/readyz")
    assert r["status"] == 200
    body = r["body"].replace(b" ", b"")
    assert b'"ready":true' in body


def test_readyz_sets_ready_gauge(app):
    _get(app, "/readyz")
    body = _get(app, "/api/metrics")["body"].decode()
    assert "cgx_ready" in body
