"""Tests for ``Session.skills`` and its pass-through into the plan/scaffold
executors.

Protects the "sessions without an explicit skill selection keep
auto-detecting from goal text" invariant: executors must call
``cgx.answer.engine``'s functions with ``skills=None`` (not ``[]``) when
the session has no skills set, and with the session's explicit list
otherwise.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cgx.session import (
    Artifact,
    ArtifactKind,
    EventBus,
    Session,
    SessionStore,
    TaskKind,
    TaskNode,
)
from cgx.session.models import SessionMode
from cgx.session.tasks.base import ExecutorDeps


@pytest.fixture()
def store(tmp_path: Path) -> SessionStore:
    db = tmp_path / "sessions.db"
    s = SessionStore(db_path=db, bus=EventBus())
    yield s
    s.close()


@pytest.fixture(autouse=True)
def _restore_task_registry():
    from cgx.session.tasks.base import _REGISTRY
    snapshot = dict(_REGISTRY)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(snapshot)


@pytest.fixture(autouse=True)
def _reset_agent_log():
    from cgx.session.agent_log import reset_for_tests
    reset_for_tests()
    yield
    reset_for_tests()


class _StubProvider:
    def chat(self, **kwargs):
        return {"content": ""}


def test_session_new_round_trips_skills_through_store(store):
    s = Session.new("add a graphql api", skills=["graphql", "fastapi"])
    store.save_session(s)
    got = store.get_session(s.session_id)
    assert got.skills == ["graphql", "fastapi"]


def test_session_new_defaults_skills_to_empty_list(store):
    s = Session.new("fix a bug")
    store.save_session(s)
    got = store.get_session(s.session_id)
    assert got.skills == []


def test_plan_change_passes_session_skills_to_generator(store, monkeypatch):
    from cgx.session.tasks.plan_change import run_plan_change

    session = Session.new("g", skills=["react", "fastapi"])
    store.save_session(session)

    captured = {}

    def fake_generate_code_plan(**kwargs):
        captured.update(kwargs)
        return {"plan_md": "p", "diffs": [], "citations": [], "confidence": 0.5}

    monkeypatch.setattr("cgx.answer.engine.generate_code_plan", fake_generate_code_plan)
    t = TaskNode.new(session.session_id, TaskKind.PLAN_CHANGE, "p",
                     inputs={"prior_goal": "add a graphql api"})
    result = run_plan_change(
        t, ExecutorDeps(provider=_StubProvider(), store=store,
                        index_dir="idx", records_path="rec"))
    assert result.failure is None
    assert captured["skills"] == ["react", "fastapi"]


def test_plan_change_passes_none_when_session_has_no_skills(store, monkeypatch):
    from cgx.session.tasks.plan_change import run_plan_change

    session = Session.new("g")
    store.save_session(session)

    captured = {}

    def fake_generate_code_plan(**kwargs):
        captured.update(kwargs)
        return {"plan_md": "p", "diffs": [], "citations": [], "confidence": 0.5}

    monkeypatch.setattr("cgx.answer.engine.generate_code_plan", fake_generate_code_plan)
    t = TaskNode.new(session.session_id, TaskKind.PLAN_CHANGE, "p",
                     inputs={"prior_goal": "fix a bug"})
    run_plan_change(
        t, ExecutorDeps(provider=_StubProvider(), store=store,
                        index_dir="idx", records_path="rec"))
    assert captured["skills"] is None


def test_decompose_passes_session_skills_to_manifest_planner(store, monkeypatch):
    from cgx.session.tasks.decompose import run_decompose

    session = Session.new("g", mode=SessionMode.GREENFIELD, skills=["fastapi"])
    store.save_session(session)

    captured = {}

    def fake_manifest(composed, provider, goal=None, skills=None):
        captured["skills"] = skills
        return {"plan_md": "p", "layers": [{"name": "app", "files": [
            {"path": "app.py", "description": "entry"}]}]}

    monkeypatch.setattr("cgx.answer.engine.plan_scaffold_manifest", fake_manifest)
    t = TaskNode.new(session.session_id, TaskKind.DECOMPOSE, "d",
                     inputs={"prior_goal": "build an api", "answers": {}})
    result = run_decompose(
        t, ExecutorDeps(provider=_StubProvider(), store=store))
    assert result.failure is None
    assert captured["skills"] == ["fastapi"]


def test_scaffold_passes_session_skills_to_generator(store, monkeypatch):
    from cgx.session.tasks.scaffold import run_scaffold

    session = Session.new("g", mode=SessionMode.GREENFIELD, skills=["tailwind"])
    store.save_session(session)
    plan = Artifact.new(
        session.session_id, "task_x", ArtifactKind.WORK_PLAN, {
            "prior_goal": "g", "composed_goal": "build ui", "answers": {},
            "plan_md": "##", "layers": [{"name": "app", "files": [
                {"path": "app.py", "description": "entry"}]}],
        })
    store.save_artifact(plan)

    captured = {}

    def fake_generate(path, description, provider, *,
                      layer=None, existing_files_with_content=None,
                      goal=None, skills=None, on_token=None,
                      depends_on=None, contracts=None):
        captured["skills"] = skills
        return {"file": path, "patch": f"+++ {path}\nx", "content": "x",
                "syntax_ok": True, "confidence": 0.9}

    monkeypatch.setattr("cgx.answer.engine.generate_single_scaffold_file", fake_generate)
    t = TaskNode.new(session.session_id, TaskKind.SCAFFOLD, "s",
                     inputs={"work_plan_artifact_id": plan.artifact_id})
    result = run_scaffold(
        t, ExecutorDeps(provider=_StubProvider(), store=store))
    assert result.failure is None
    assert captured["skills"] == ["tailwind"]
