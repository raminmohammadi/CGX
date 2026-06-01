"""Tests for :mod:`cgx.webui.task_store` (SQLite task registry)."""

from __future__ import annotations

import time

import pytest

from cgx.webui import task_store as ts


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    """Point the module-level singleton at a tmp DB for each test."""
    monkeypatch.setattr(ts, "_DB_PATH", tmp_path / "tasks.db")
    # Force re-open against the new path.
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


def test_create_task_returns_unique_ids_and_persists_metadata():
    a = ts.create_task("ask", {"q": "hi"})
    b = ts.create_task("plan")
    assert a != b
    row = ts.get_task(a)
    assert row is not None
    assert row["type"] == "ask"
    assert row["status"] == "running"
    assert row["created_at"] > 0


def test_get_task_returns_none_for_unknown_id():
    assert ts.get_task("does-not-exist") is None


def test_finish_task_marks_status_done_and_clears_cancel_event():
    task_id = ts.create_task("ask")
    assert ts.get_cancel_event(task_id) is not None
    ts.finish_task(task_id)
    assert ts.get_task(task_id)["status"] == "done"
    assert ts.get_cancel_event(task_id) is None


def test_finish_task_with_error_records_failed_status():
    task_id = ts.create_task("ask")
    ts.finish_task(task_id, error="boom")
    row = ts.get_task(task_id)
    assert row["status"] == "failed"
    assert row["error"] == "boom"


def test_cancel_task_returns_true_when_event_exists_and_flips_status():
    task_id = ts.create_task("ask")
    assert ts.cancel_task(task_id) is True
    assert ts.get_cancel_event(task_id).is_set()
    assert ts.get_task(task_id)["status"] == "cancelled"


def test_cancel_task_returns_false_for_unknown_task():
    assert ts.cancel_task("nope") is False


def test_append_event_and_get_task_events_roundtrip():
    task_id = ts.create_task("agent")
    ts.append_event(task_id, "plan", {"goal": "x"})
    ts.append_event(task_id, "task_start", {"name": "t1"})
    ts.append_event(task_id, "task_done", None)

    events = ts.get_task_events(task_id)
    types = [e["type"] for e in events]
    assert types == ["plan", "task_start", "task_done"]
    assert events[0]["payload"] == {"goal": "x"}
    assert events[2]["payload"] == {}  # None payload normalises to {}


def test_get_task_events_handles_malformed_payload(tmp_path, monkeypatch):
    task_id = ts.create_task("ask")
    # Inject a row with invalid JSON via the underlying conn.
    conn = ts._get_conn()
    conn.execute(
        "INSERT INTO task_events (task_id, event_type, payload_json, at) "
        "VALUES (?, ?, ?, ?)",
        (task_id, "bad", "{not-json", time.time()),
    )
    conn.commit()
    events = ts.get_task_events(task_id)
    bad = [e for e in events if e["type"] == "bad"][0]
    assert bad["payload"] == {}


def test_list_tasks_returns_most_recent_first():
    a = ts.create_task("ask")
    time.sleep(0.01)
    b = ts.create_task("plan")
    time.sleep(0.01)
    c = ts.create_task("agent")
    rows = ts.list_tasks(limit=10)
    ids = [r["id"] for r in rows]
    assert ids[:3] == [c, b, a]


def test_list_tasks_respects_limit():
    for _ in range(5):
        ts.create_task("ask")
    rows = ts.list_tasks(limit=3)
    assert len(rows) == 3


def test_prune_old_tasks_removes_records_and_events_older_than_cutoff():
    old_id = ts.create_task("ask")
    ts.append_event(old_id, "msg", {"x": 1})
    # Backdate it directly.
    conn = ts._get_conn()
    conn.execute("UPDATE tasks SET created_at=? WHERE id=?",
                 (time.time() - 9999, old_id))
    conn.commit()

    fresh_id = ts.create_task("plan")
    removed = ts.prune_old_tasks(max_age_seconds=60)
    assert removed == 1
    assert ts.get_task(old_id) is None
    assert ts.get_task(fresh_id) is not None
    assert ts.get_task_events(old_id) == []


def test_prune_old_tasks_returns_zero_when_nothing_to_remove():
    ts.create_task("ask")
    assert ts.prune_old_tasks(max_age_seconds=9999) == 0
