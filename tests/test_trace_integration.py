"""End-to-end check that a small session drives trace lines into agent.log.

Boots a real :class:`SessionRunner` against a tmp-path-rooted store with
a stub EXPLORE executor, flips the global trace flag on, and asserts the
project-local ``agent.log`` carries ``trace_enter`` / ``trace_exit``
records for the runner, the router, and the executor wrapper. This is
the integration counterpart to :mod:`tests.test_trace`'s unit tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cgx import trace as tr
from cgx.session import (
    Artifact,
    ArtifactKind,
    EventBus,
    Fact,
    FactKind,
    SessionRunner,
    SessionStore,
    TaskKind,
)
from cgx.session.agent_log import reset_for_tests as _reset_agent_log
from cgx.session.tasks.base import (
    ExecutorDeps,
    ExecutorResult,
    _REGISTRY,
    register_executor,
)


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    monkeypatch.delenv("CGX_TRACE", raising=False)
    tr.reset_for_tests()
    _reset_agent_log()
    snapshot = dict(_REGISTRY)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(snapshot)
    tr.reset_for_tests()
    _reset_agent_log()


def _read_log(root: Path) -> list[dict]:
    p = root / ".cgx" / "agent.log"
    if not p.exists():
        return []
    out: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _install_stub_explore() -> None:
    @register_executor(TaskKind.EXPLORE)
    def _explore(task, deps):
        artifact = Artifact.new(
            session_id=task.session_id,
            produced_by_task_id=task.task_id,
            kind=ArtifactKind.DIRECTIONS_LIST,
            content={"goal": task.inputs.get("goal"),
                     "options": [
                         {"chunk_id": "c1", "title": "A", "rationale": "a"},
                         {"chunk_id": "c2", "title": "B", "rationale": "b"},
                     ]},
        )
        facts = [
            Fact.new(task.session_id, FactKind.ANCHOR,
                     {"chunk_id": "c1", "title": "A"},
                     surfaced_in_task_id=task.task_id),
        ]
        return ExecutorResult(
            outputs={"options_count": 2,
                     "directions_artifact_id": artifact.artifact_id},
            facts=facts, artifact=artifact)


def test_session_emits_trace_records_when_toggle_on(tmp_path: Path):
    project_root = tmp_path / "proj"
    project_root.mkdir()
    db = tmp_path / "sessions.db"

    _install_stub_explore()
    tr.set_trace_enabled(True)

    store = SessionStore(db_path=db, bus=EventBus())
    try:
        runner = SessionRunner(store)
        session = runner.start_session(
            objective="improve retrieval",
            project_root=str(project_root))
        # Two steps drive EXPLORE -> DONE and the ASK_USER spawn -> READY.
        runner.run_next(session_id=session.session_id, deps=ExecutorDeps())
        runner.run_next(session_id=session.session_id, deps=ExecutorDeps())
    finally:
        store.close()

    events = _read_log(project_root)
    trace_events = [e for e in events if e.get("kind") == "trace"]
    assert trace_events, "expected at least one trace_* line in agent.log"

    by_category: dict[str, list[str]] = {}
    for e in trace_events:
        by_category.setdefault(e.get("category", ""), []).append(
            e.get("event", ""))

    # Runner entry point ran (run_next is @traced("runner")).
    assert "runner" in by_category
    assert "trace_enter" in by_category["runner"]
    assert "trace_exit" in by_category["runner"]

    # Router transition method ran (on_user_message / on_task_completed
    # / on_decision_recorded are @traced("router")).
    assert "router" in by_category
    assert "trace_enter" in by_category["router"]

    # Stub EXPLORE executor was auto-wrapped via register_executor.
    assert "executor" in by_category
    assert "trace_enter" in by_category["executor"]
    assert "trace_exit" in by_category["executor"]

    # Trace records carry session/task identifiers from the ContextVar.
    exec_events = [e for e in trace_events if e.get("category") == "executor"]
    assert all(e.get("session_id") == session.session_id for e in exec_events)
    assert all(e.get("task_id") for e in exec_events)


def test_session_emits_no_trace_records_when_toggle_off(tmp_path: Path):
    project_root = tmp_path / "proj"
    project_root.mkdir()
    db = tmp_path / "sessions.db"

    _install_stub_explore()
    # Default is OFF; assert that explicitly here.
    assert tr.is_trace_enabled() is False

    store = SessionStore(db_path=db, bus=EventBus())
    try:
        runner = SessionRunner(store)
        session = runner.start_session(
            objective="g", project_root=str(project_root))
        runner.run_next(session_id=session.session_id, deps=ExecutorDeps())
    finally:
        store.close()

    events = _read_log(project_root)
    assert all(e.get("kind") != "trace" for e in events), \
        "no trace_* events should fire when the toggle is off"
    # Non-trace agent-log emissions (e.g. task_started) still flow.
    assert any(e.get("event") == "task_started" for e in events)
