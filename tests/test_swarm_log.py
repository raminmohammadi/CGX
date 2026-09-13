"""Regression coverage for swarm sub-agent telemetry (``swarm_beat``).

``swarm_beat`` is meant to persist one ``SWARM_BEAT`` fact per step so the live
agent UI can render per-sub-agent progress. It used to construct
``SessionStore(project_root)`` positionally -- but ``SessionStore``'s first
positional parameter is ``db_path``, so the project *directory* was handed to
``sqlite3.connect`` and ``__init__`` raised, which ``swarm_beat``'s best-effort
guard swallowed. Result: every beat was silently dropped and the UI never saw a
single sub-agent event. These tests pin the fixed behaviour: the beat both
persists to the session DB the webui reads and emits ``FACT_ADDED`` on the bus.
"""

import tempfile

from cgx.session.events import EventType, get_default_bus
from cgx.session.models import (
    FactKind, Session, SessionMode, TaskKind, TaskNode)
from cgx.session.store import SessionStore, default_db_path
from cgx.session.tasks import swarm_log
from cgx.trace import reset_trace_context, set_trace_context


def _seed_session(root):
    """Persist a session + task so the fact's FK parents exist."""
    store = SessionStore(project_root=root)
    sess = Session.new(original_objective="build", mode=SessionMode.SWARM,
                       project_root=root)
    store.save_session(sess)
    task = TaskNode.new(session_id=sess.session_id,
                        kind=TaskKind.SWARM_DEVELOPER, name="gen", description="")
    store.save_task(task)
    return store, sess, task


def test_swarm_beat_persists_to_session_db():
    """A beat lands as a SWARM_BEAT fact in the same DB the webui reads."""
    swarm_log._STORE_CACHE.clear()
    with tempfile.TemporaryDirectory() as root:
        store, sess, task = _seed_session(root)
        token = set_trace_context(session_id=sess.session_id, task_id=task.task_id)
        try:
            swarm_log.swarm_beat(root, "developer", "write",
                                 file="src/a.py", index=1, total=2, ok=True)
        finally:
            reset_trace_context(token)

        # The cached beat store must point at the very DB the runner/webui use.
        assert swarm_log._STORE_CACHE[root].path == default_db_path(root)
        beats = store.load_kb(sess.session_id).of_kind(FactKind.SWARM_BEAT)
        assert len(beats) == 1
        assert beats[0].content["phase"] == "write"
        assert beats[0].content["file"] == "src/a.py"


def test_swarm_beat_emits_fact_added_event():
    """A beat publishes FACT_ADDED so the SSE stream can forward it live."""
    swarm_log._STORE_CACHE.clear()
    with tempfile.TemporaryDirectory() as root:
        _store, sess, task = _seed_session(root)
        seen = []
        unsub = get_default_bus().subscribe("*", seen.append)
        token = set_trace_context(session_id=sess.session_id, task_id=task.task_id)
        try:
            swarm_log.swarm_beat(root, "tech_lead", "plan", goal="x")
        finally:
            reset_trace_context(token)
            unsub()

        facts = [e for e in seen
                 if e.type is EventType.FACT_ADDED and e.session_id == sess.session_id]
        assert len(facts) == 1
        assert (facts[0].payload.get("content") or {}).get("role") == "tech_lead"


def test_swarm_beat_never_raises_without_trace_context():
    """No session/task in context is a no-op, never an exception."""
    swarm_log._STORE_CACHE.clear()
    with tempfile.TemporaryDirectory() as root:
        # No set_trace_context: swarm_beat should quietly skip the DB write.
        swarm_log.swarm_beat(root, "developer", "generate", file="x.py")
