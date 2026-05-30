"""Tests for the persistent chat-session store (``cgx.sessions``)."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture()
def fresh_sessions(tmp_path, monkeypatch):
    """Re-import ``cgx.sessions`` with CGX_CONFIG_DIR pointed at tmp_path."""
    monkeypatch.setenv("CGX_CONFIG_DIR", str(tmp_path))
    import cgx.sessions as ses  # noqa: WPS433
    importlib.reload(ses)
    yield ses
    importlib.reload(ses)


def test_create_session_persists_in_index(fresh_sessions):
    s = fresh_sessions.create_session("first")
    assert s.id and s.title == "first"
    listed = fresh_sessions.list_sessions()
    assert [m.id for m in listed] == [s.id]


def test_append_and_load_messages_roundtrip(fresh_sessions):
    s = fresh_sessions.create_session()
    fresh_sessions.append_message(s.id, "user", "Hello agent")
    fresh_sessions.append_message(s.id, "assistant", "Hi!",
                                  meta={"sources": [{"f": "x"}]})
    msgs = fresh_sessions.load_messages(s.id)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["meta"]["sources"][0]["f"] == "x"
    assert all("at" in m for m in msgs)


def test_appending_promotes_default_title_to_first_user_message(fresh_sessions):
    s = fresh_sessions.create_session()  # default title 'Untitled'
    assert fresh_sessions.list_sessions()[0].title == "Untitled"
    fresh_sessions.append_message(s.id, "user", "What does parse_codebase do?")
    assert fresh_sessions.list_sessions()[0].title.startswith("What does")
    # Second user message should not overwrite a non-default title.
    fresh_sessions.append_message(s.id, "user", "second")
    assert fresh_sessions.list_sessions()[0].title.startswith("What does")


def test_sessions_list_is_most_recent_first(fresh_sessions):
    a = fresh_sessions.create_session("A")
    b = fresh_sessions.create_session("B")
    fresh_sessions.append_message(a.id, "user", "ping")
    ids = [m.id for m in fresh_sessions.list_sessions()]
    # A just updated -> should now lead.
    assert ids[0] == a.id and ids[1] == b.id


def test_delete_session_removes_index_and_file(fresh_sessions):
    s = fresh_sessions.create_session("zap")
    p = fresh_sessions.SESSIONS_DIR / f"{s.id}.jsonl"
    assert p.exists()
    assert fresh_sessions.delete_session(s.id) is True
    assert not p.exists()
    assert fresh_sessions.list_sessions() == []
    # Deleting an unknown id reports False.
    assert fresh_sessions.delete_session("does-not-exist") is False


def test_append_message_to_unknown_session_raises(fresh_sessions):
    with pytest.raises(FileNotFoundError):
        fresh_sessions.append_message("nope", "user", "x")


def test_rename_session(fresh_sessions):
    s = fresh_sessions.create_session("old")
    assert fresh_sessions.rename_session(s.id, "new title") is True
    assert fresh_sessions.list_sessions()[0].title == "new title"


def test_load_messages_tolerates_corrupt_lines(fresh_sessions):
    s = fresh_sessions.create_session()
    fresh_sessions.append_message(s.id, "user", "real")
    # Manually append garbage.
    with (fresh_sessions.SESSIONS_DIR / f"{s.id}.jsonl").open("a") as f:
        f.write("this is not json\n")
    fresh_sessions.append_message(s.id, "assistant", "after")
    msgs = fresh_sessions.load_messages(s.id)
    assert [m["content"] for m in msgs] == ["real", "after"]


def test_atomic_index_write_uses_rename(fresh_sessions, tmp_path):
    s = fresh_sessions.create_session("a")
    # If the writer didn't rename atomically the .tmp file would linger.
    leftover = list(fresh_sessions.SESSIONS_DIR.glob("*.tmp"))
    assert leftover == []
    # The persisted index is valid JSON.
    with fresh_sessions.INDEX_PATH.open("r") as f:
        data = json.load(f)
    assert data["sessions"][0]["id"] == s.id
