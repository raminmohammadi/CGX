"""Unit tests for the curated function-call tracing module."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from cgx import trace as tr
from cgx.session import agent_log as al


@pytest.fixture(autouse=True)
def _reset_trace(monkeypatch):
    monkeypatch.delenv("CGX_TRACE", raising=False)
    tr.reset_for_tests()
    al.reset_for_tests()
    yield
    tr.reset_for_tests()
    al.reset_for_tests()


def _read_agent_log(root: Path) -> list[dict]:
    path = root / ".cgx" / "agent.log"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def test_toggle_default_off():
    assert tr.is_trace_enabled() is False
    assert tr.set_trace_enabled(True) is True
    assert tr.is_trace_enabled() is True
    tr.set_trace_enabled(False)
    assert tr.is_trace_enabled() is False


def test_env_pin_overrides_runtime(monkeypatch):
    monkeypatch.setenv("CGX_TRACE", "1")
    assert tr.is_trace_enabled() is True
    assert tr.trace_source() == "env"
    tr.set_trace_enabled(False)
    assert tr.is_trace_enabled() is True

    monkeypatch.setenv("CGX_TRACE", "0")
    assert tr.is_trace_enabled() is False
    tr.set_trace_enabled(True)
    assert tr.is_trace_enabled() is False


def test_traced_sync_emits_enter_exit(tmp_path: Path):
    token = tr.set_trace_context(
        session_id="s1", task_id="t1", project_root=str(tmp_path))
    tr.set_trace_enabled(True)
    try:
        @tr.traced("test", args=True)
        def f(x, y=2):
            return x + y

        assert f(1, y=5) == 6
    finally:
        tr.reset_trace_context(token)

    events = _read_agent_log(tmp_path)
    kinds = [e.get("event") for e in events]
    assert "trace_enter" in kinds
    assert "trace_exit" in kinds
    enter = next(e for e in events if e["event"] == "trace_enter")
    assert enter["category"] == "test"
    assert enter["session_id"] == "s1"
    assert enter["task_id"] == "t1"
    assert enter["args"]["x"] == "1"
    assert enter["args"]["y"] == "5"
    exit_evt = next(e for e in events if e["event"] == "trace_exit")
    assert exit_evt["elapsed_ms"] >= 0


def test_traced_error_emits_error_event(tmp_path: Path):
    token = tr.set_trace_context(project_root=str(tmp_path))
    tr.set_trace_enabled(True)
    try:
        @tr.traced("test")
        def boom():
            raise ValueError("nope")

        with pytest.raises(ValueError):
            boom()
    finally:
        tr.reset_trace_context(token)

    events = _read_agent_log(tmp_path)
    kinds = [e["event"] for e in events]
    assert "trace_enter" in kinds
    assert "trace_error" in kinds
    err = next(e for e in events if e["event"] == "trace_error")
    assert err["error_type"] == "ValueError"
    assert "nope" in err["error"]


def test_traced_async(tmp_path: Path):
    token = tr.set_trace_context(project_root=str(tmp_path))
    tr.set_trace_enabled(True)
    try:
        @tr.traced("test")
        async def af(x):
            await asyncio.sleep(0)
            return x * 2

        result = asyncio.run(af(3))
        assert result == 6
    finally:
        tr.reset_trace_context(token)

    events = _read_agent_log(tmp_path)
    kinds = [e["event"] for e in events]
    assert kinds.count("trace_enter") == 1
    assert kinds.count("trace_exit") == 1


def test_traced_off_is_noop(tmp_path: Path):
    token = tr.set_trace_context(project_root=str(tmp_path))
    tr.set_trace_enabled(False)
    try:
        @tr.traced("test")
        def f():
            return 42

        assert f() == 42
    finally:
        tr.reset_trace_context(token)

    assert _read_agent_log(tmp_path) == []


def test_context_propagates_into_nested_calls(tmp_path: Path):
    """Nested @traced calls inherit the outer set_trace_context frame."""
    token = tr.set_trace_context(
        session_id="s_outer", task_id="t_outer",
        project_root=str(tmp_path))
    tr.set_trace_enabled(True)
    try:
        @tr.traced("inner")
        def inner():
            return 7

        @tr.traced("outer")
        def outer():
            return inner()

        assert outer() == 7
    finally:
        tr.reset_trace_context(token)

    events = _read_agent_log(tmp_path)
    cats = [e.get("category") for e in events if e.get("event") == "trace_enter"]
    assert "outer" in cats and "inner" in cats
    # Every record carries the outer session/task ids (ContextVar inheritance).
    for e in events:
        if e.get("event") in ("trace_enter", "trace_exit"):
            assert e.get("session_id") == "s_outer"
            assert e.get("task_id") == "t_outer"


def test_context_propagates_across_async_nested_calls(tmp_path: Path):
    """Async @traced calls inherit the ContextVar via the event loop."""
    token = tr.set_trace_context(
        session_id="s_a", project_root=str(tmp_path))
    tr.set_trace_enabled(True)
    try:
        @tr.traced("inner")
        async def inner():
            await asyncio.sleep(0)
            return 1

        @tr.traced("outer")
        async def outer():
            return await inner() + 1

        assert asyncio.run(outer()) == 2
    finally:
        tr.reset_trace_context(token)

    events = _read_agent_log(tmp_path)
    cats = [e.get("category") for e in events if e.get("event") == "trace_enter"]
    assert "outer" in cats and "inner" in cats
    for e in events:
        if e.get("event") in ("trace_enter", "trace_exit"):
            assert e.get("session_id") == "s_a"


def test_emit_trace_helper_respects_toggle(tmp_path: Path):
    """The public :func:`emit_trace` helper is a no-op when tracing is off."""
    token = tr.set_trace_context(project_root=str(tmp_path))
    tr.set_trace_enabled(False)
    try:
        tr.emit_trace("http_request", method="GET", path="/x")
    finally:
        tr.reset_trace_context(token)
    assert _read_agent_log(tmp_path) == []

    token = tr.set_trace_context(project_root=str(tmp_path))
    tr.set_trace_enabled(True)
    try:
        tr.emit_trace("http_request", method="GET", path="/x")
    finally:
        tr.reset_trace_context(token)
    events = _read_agent_log(tmp_path)
    assert any(e.get("event") == "http_request" for e in events)


def test_request_id_propagates_into_records(tmp_path: Path):
    """A request_id set on the context appears on every emitted record."""
    token = tr.set_trace_context(
        project_root=str(tmp_path), request_id="req-42")
    tr.set_trace_enabled(True)
    try:
        @tr.traced("test")
        def f():
            return 1

        assert f() == 1
    finally:
        tr.reset_trace_context(token)

    events = _read_agent_log(tmp_path)
    assert events
    for e in events:
        if e.get("event") in ("trace_enter", "trace_exit"):
            assert e.get("request_id") == "req-42"


def test_otel_span_is_noop_when_disabled(tmp_path: Path, monkeypatch):
    """With CGX_OTEL unset, the decorator never touches OpenTelemetry."""
    monkeypatch.delenv("CGX_OTEL", raising=False)
    token = tr.set_trace_context(project_root=str(tmp_path))
    tr.set_trace_enabled(False)
    try:
        @tr.traced("test")
        def f():
            return 7

        # Neither trace nor OTel enabled -> fast path, value still returned.
        assert f() == 7
        assert tr._otel_enabled() is False
    finally:
        tr.reset_trace_context(token)


def test_fallback_logger_when_no_project_root(tmp_path: Path, monkeypatch):
    fake_home = tmp_path / "home"
    monkeypatch.setattr(tr, "_FALLBACK_DIR", fake_home / ".cgx")
    tr.set_trace_enabled(True)

    @tr.traced("test")
    def f():
        return 1

    assert f() == 1
    log_path = fake_home / ".cgx" / "cgx-trace.log"
    assert log_path.exists()
    content = log_path.read_text(encoding="utf-8")
    assert "trace_enter" in content
    assert "trace_exit" in content
