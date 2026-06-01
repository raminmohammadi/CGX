"""Tests for :mod:`cgx.webui.sse` (async generator bridge for SSE streams)."""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, AsyncIterator, Dict, List

import pytest

from cgx.webui import sse, task_store as ts


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    monkeypatch.setattr(ts, "_DB_PATH", tmp_path / "tasks.db")
    if ts._conn is not None:
        try:
            ts._conn.close()
        except Exception:
            pass
    monkeypatch.setattr(ts, "_conn", None)
    ts._cancel_events.clear()
    yield
    if ts._conn is not None:
        try:
            ts._conn.close()
        except Exception:
            pass
    monkeypatch.setattr(ts, "_conn", None)


async def _collect(agen: AsyncIterator[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    async for msg in agen:
        out.append(msg)
    return out


# ---------------------------------------------------------------------------
# _safe_json
# ---------------------------------------------------------------------------
def test_safe_json_serialises_normal_objects():
    out = sse._safe_json({"a": 1, "b": [1, 2]})
    assert json.loads(out) == {"a": 1, "b": [1, 2]}


def test_safe_json_falls_back_to_repr_for_unserialisable():
    class _Bad:
        def __repr__(self): return "<bad>"
        def __str__(self): return "<bad>"

    class _BoomEncoder:
        def __init__(self, *a, **kw): pass
        def encode(self, _): raise TypeError("nope")

    # default=str handles most objects; pass a circular dict to trigger the
    # outer try/except fallback path.
    d: Dict[str, Any] = {}
    d["self"] = d
    out = sse._safe_json(d)
    assert "_repr" in out


# ---------------------------------------------------------------------------
# bridge_generator
# ---------------------------------------------------------------------------
def test_bridge_forwards_events_and_terminates_with_done_frame():
    def gen():
        yield {"event": "task_start", "data": {"name": "a"}}
        yield {"event": "task_done", "data": {"name": "a"}}

    def to_event(item):
        return item

    async def go():
        return await _collect(sse.bridge_generator(
            gen, to_event=to_event, task_id=None, cancel_event=None,
        ))

    events = asyncio.run(go())
    assert events[-1] == {"event": "done", "data": "{}"}
    types = [e["event"] for e in events]
    assert types == ["task_start", "task_done", "done"]


def test_bridge_emits_error_frame_and_stops_when_generator_raises():
    def gen():
        yield {"event": "task_start", "data": {"name": "a"}}
        raise RuntimeError("oops")

    async def go():
        return await _collect(sse.bridge_generator(
            gen, to_event=lambda x: x, task_id=None, cancel_event=None,
        ))

    events = asyncio.run(go())
    types = [e["event"] for e in events]
    # task_start gets through; then error; no done after error (break).
    assert "error" in types
    assert "RuntimeError" in events[types.index("error")]["data"]


def test_bridge_persists_events_when_task_id_is_provided():
    task_id = ts.create_task("ask")

    def gen():
        yield {"event": "delta", "data": {"text": "hi"}}
        yield {"event": "final", "data": {"answer": "done"}}

    async def go():
        return await _collect(sse.bridge_generator(
            gen, to_event=lambda x: x, task_id=task_id, cancel_event=None,
        ))

    asyncio.run(go())

    stored = ts.get_task_events(task_id)
    types = [e["type"] for e in stored]
    assert "delta" in types and "final" in types
    # The final "done" frame is intentionally not persisted.
    assert "done" not in types
    # bridge should mark the still-running task as done on exit.
    assert ts.get_task(task_id)["status"] == "done"


def test_bridge_emits_cancelled_event_when_cancel_event_is_set():
    cancel = threading.Event()
    cancel.set()  # cancelled before any work begins

    def gen():
        yield {"event": "task_start", "data": {"name": "a"}}

    def to_event(item):
        # Production routes (see cgx.webui.routes.ask) tolerate both
        # ``(event, payload)`` tuples and pre-shaped dicts.
        if isinstance(item, tuple) and len(item) == 2:
            ev, payload = item
            return {"event": ev, "data": payload}
        return item

    async def go():
        return await _collect(sse.bridge_generator(
            gen, to_event=to_event, task_id=None, cancel_event=cancel,
        ))

    events = asyncio.run(go())
    types = [e["event"] for e in events]
    assert "cancelled" in types
    assert types[-1] == "done"


def test_bridge_with_task_id_records_error_payload():
    task_id = ts.create_task("ask")

    def gen():
        raise ValueError("kapow")

    async def go():
        return await _collect(sse.bridge_generator(
            gen, to_event=lambda x: x, task_id=task_id, cancel_event=None,
        ))

    asyncio.run(go())
    types = [e["type"] for e in ts.get_task_events(task_id)]
    assert "error" in types
