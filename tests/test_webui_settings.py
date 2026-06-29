"""Route tests for ``/api/settings/trace`` (Phase TR.4).

The handlers are sync functions over an in-memory module-global flag,
so we call them directly rather than spinning up a TestClient -- matches
the convention used by :mod:`tests.test_webui_agent_session`.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from cgx import trace as tr
from cgx.webui.routes import settings as settings_route


@pytest.fixture(autouse=True)
def _reset_trace(monkeypatch):
    """Drop env-var pinning and runtime flag between cases."""
    monkeypatch.delenv("CGX_TRACE", raising=False)
    tr.reset_for_tests()
    yield
    tr.reset_for_tests()


def test_get_trace_defaults_off_runtime():
    s = settings_route.get_trace_settings()
    assert s.enabled is False
    assert s.source == "runtime"


def test_post_trace_flips_runtime_flag():
    enabled = settings_route.set_trace_settings(
        settings_route.TraceSettingsPatch(enabled=True))
    assert enabled.enabled is True
    assert enabled.source == "runtime"
    assert tr.is_trace_enabled() is True

    off = settings_route.set_trace_settings(
        settings_route.TraceSettingsPatch(enabled=False))
    assert off.enabled is False
    assert tr.is_trace_enabled() is False


def test_get_reports_env_pinning_when_cgx_trace_is_set(monkeypatch):
    monkeypatch.setenv("CGX_TRACE", "1")
    s = settings_route.get_trace_settings()
    assert s.enabled is True
    assert s.source == "env"


def test_post_returns_409_when_env_pinned(monkeypatch):
    monkeypatch.setenv("CGX_TRACE", "1")
    with pytest.raises(HTTPException) as exc:
        settings_route.set_trace_settings(
            settings_route.TraceSettingsPatch(enabled=False))
    assert exc.value.status_code == 409
    # Runtime state stays consistent with the env pin.
    assert tr.is_trace_enabled() is True


def test_post_409_also_fires_for_falsey_env_pin(monkeypatch):
    """``CGX_TRACE=0`` pins OFF; the UI still can't override it."""
    monkeypatch.setenv("CGX_TRACE", "0")
    with pytest.raises(HTTPException) as exc:
        settings_route.set_trace_settings(
            settings_route.TraceSettingsPatch(enabled=True))
    assert exc.value.status_code == 409
    assert tr.is_trace_enabled() is False
