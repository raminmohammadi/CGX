"""Route-handler integration tests for ``/api/agent-session``.

Drives the full write loop end-to-end (create -> decision with each
``chosen`` shape the React UI ships) so the typed ``build_decision``
validators stay in sync with the frontend contract.

Calls the async route handlers directly with Pydantic request models
rather than going through the ASGI transport, so the test suite stays
free of an ``httpx`` dependency. Pydantic still parses every payload,
so the wire shape is validated.

Stub executors are installed for every TaskKind to keep the loop free
of real LLM / index dependencies, and ``_build_deps`` is monkeypatched
so the route layer never tries to construct a provider.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any, Awaitable, Iterator, TypeVar

import pytest
from fastapi import HTTPException

from cgx.session import tasks as _tasks  # noqa: F401 - registers executors
from cgx.session.models import (
    Artifact, ArtifactKind, Fact, FactKind, TaskKind,
)
from cgx.session.tasks.base import (
    ExecutorDeps, ExecutorResult, _REGISTRY, register_executor,
)
from cgx.webui import models as wm
from cgx.webui.routes import agent_session as routes

# Captured at import time -- before the autouse ``_stub_build_deps`` fixture
# monkeypatches ``routes._build_deps`` -- so the profile-not-found test can
# drive the real dependency builder.
_REAL_BUILD_DEPS = routes._build_deps

T = TypeVar("T")


def _run(coro: Awaitable[T]) -> T:
    """Run a coroutine to completion in a fresh event loop."""
    return asyncio.new_event_loop().run_until_complete(coro)


# ----------------------------- fixtures -----------------------------

@pytest.fixture()
def project_root(tmp_path: Path) -> Path:
    return tmp_path / "proj"


@pytest.fixture(autouse=True)
def _reset_runners() -> Iterator[None]:
    """Close + clear the module-level runner cache between cases."""
    routes._RUNNERS.clear()
    routes._RUNNER_DB_IDS.clear()
    routes._SESSION_TO_RUNNER.clear()
    yield
    for r in list(routes._RUNNERS.values()):
        try:
            r.store.close()
        except Exception:
            pass
    routes._RUNNERS.clear()
    routes._RUNNER_DB_IDS.clear()
    routes._SESSION_TO_RUNNER.clear()


@pytest.fixture(autouse=True)
def _stub_build_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass provider construction; stub executors ignore deps anyway."""
    monkeypatch.setattr(
        routes, "_build_deps",
        lambda *a, **kw: ExecutorDeps(store=kw.get("store")))


@pytest.fixture(autouse=True)
def _restore_registry() -> Iterator[None]:
    """Snapshot the executor registry so stub installs don't leak."""
    snapshot = dict(_REGISTRY)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(snapshot)


@pytest.fixture(autouse=True)
def _drain_inline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the drain synchronously in tests.

    The direct-call shim drives each handler to completion on a throwaway
    event loop, so a backgrounded drain would never run. Forcing the
    inline path keeps snapshots reflecting a fully-drained pipeline.
    """
    monkeypatch.setattr(routes, "_RUN_DRAIN_IN_BACKGROUND", False)


@pytest.fixture()
def client() -> "_HandlerClient":
    """Direct-call shim that mirrors the routes' HTTP surface.

    Each method invokes the matching async handler with a Pydantic
    request model and returns the serialised state dict (same JSON the
    real /api/agent-session/* endpoints emit), translating HTTPException
    into ``status_code`` / ``detail`` so tests can assert on it.
    """
    return _HandlerClient()


class _HandlerClient:
    last_status: int = 200
    last_detail: Any = None

    def _call(self, coro: Awaitable[Any]) -> dict | list | None:
        try:
            result = _run(coro)
        except HTTPException as exc:
            self.last_status = exc.status_code
            self.last_detail = exc.detail
            return None
        self.last_status = 200
        self.last_detail = None
        return result.model_dump() if hasattr(result, "model_dump") else result

    def create(self, **body: Any) -> dict:
        # Default to explore mode so the EXPLORE-loop fixtures keep
        # exercising EXPLORE; greenfield-specific tests opt in by
        # passing ``mode="greenfield"`` (or omitting the override).
        body.setdefault("mode", "explore")
        return self._call(routes.create_session(
            wm.AgentSessionCreateRequest(**body)))  # type: ignore[arg-type]

    def get(self, sid: str, project_root: str | None = None) -> dict:
        return self._call(routes.get_session(sid, project_root=project_root))

    def list_sessions(self, project_root: str | None = None) -> list:
        return self._call(routes.list_agent_sessions(project_root=project_root))

    def message(self, sid: str, **body: Any) -> dict:
        return self._call(routes.post_message(
            sid, wm.AgentSessionMessageRequest(**body)))

    def decision(self, sid: str, **body: Any) -> dict | None:
        return self._call(routes.post_decision(
            sid, wm.AgentSessionDecisionRequest(**body)))

    def delete(self, sid: str, project_root: str | None = None) -> dict | None:
        return self._call(routes.delete_session(
            sid, project_root=project_root))

    def cancel(self, sid: str) -> dict | None:
        return self._call(routes.post_cancel(sid))


# ----------------------------- stubs -----------------------------

def _install_stubs() -> None:
    @register_executor(TaskKind.EXPLORE)
    def _explore(task, deps):
        art = Artifact.new(
            task.session_id, task.task_id, ArtifactKind.DIRECTIONS_LIST,
            {"goal": task.inputs.get("goal"),
             "options": [{"chunk_id": "c1", "title": "A", "rationale": "a"}]})
        facts = [Fact.new(task.session_id, FactKind.ANCHOR,
                          {"chunk_id": "c1", "title": "A"},
                          surfaced_in_task_id=task.task_id)]
        return ExecutorResult(
            outputs={"directions_artifact_id": art.artifact_id},
            facts=facts, artifact=art)

    @register_executor(TaskKind.INVESTIGATE)
    def _investigate(task, deps):
        art = Artifact.new(
            task.session_id, task.task_id, ArtifactKind.FINDINGS_BUNDLE,
            {"findings_md": "## what we found\n- thing"})
        return ExecutorResult(
            outputs={"findings_artifact_id": art.artifact_id}, artifact=art)

    @register_executor(TaskKind.RECOMMEND)
    def _recommend(task, deps):
        art = Artifact.new(
            task.session_id, task.task_id, ArtifactKind.RECOMMENDATION_LIST,
            {"recommendations": [
                {"id": "r1", "title": "Cache results", "rationale": "speed",
                 "kind": "plan_change"}]})
        return ExecutorResult(
            outputs={"recommendations_artifact_id": art.artifact_id},
            artifact=art)

    @register_executor(TaskKind.PLAN_CHANGE)
    def _plan(task, deps):
        art = Artifact.new(
            task.session_id, task.task_id, ArtifactKind.CODE_CHANGE_PLAN,
            {"plan_md": "## Plan\n- step", "diffs": [
                {"file": "x.py", "patch": "diff"}], "confidence": 0.7})
        return ExecutorResult(
            outputs={"plan_artifact_id": art.artifact_id}, artifact=art)

    @register_executor(TaskKind.APPLY)
    def _apply(task, deps):
        art = Artifact.new(
            task.session_id, task.task_id, ArtifactKind.APPLIED_CHANGES,
            {"plan_artifact_id": task.inputs.get("plan_artifact_id"),
             "applied_files": ["x.py"], "failed_files": [],
             "smoke_ok": True})
        return ExecutorResult(
            outputs={"apply_artifact_id": art.artifact_id}, artifact=art)

    @register_executor(TaskKind.VERIFY)
    def _verify(task, deps):
        art = Artifact.new(
            task.session_id, task.task_id, ArtifactKind.VERIFY_REPORT,
            {"ran": True, "tests_passed": True, "returncode": 0})
        return ExecutorResult(
            outputs={"verify_artifact_id": art.artifact_id,
                     "tests_passed": True}, artifact=art)


# ----------------------------- helpers -----------------------------

def _find_task(state: dict, *, kind: str,
               expected_kind: str | None = None) -> dict | None:
    """First task whose ``kind`` matches; optionally narrow by ``expected_kind``."""
    for t in state["tasks"]:
        if t["kind"] != kind:
            continue
        if (expected_kind is not None
                and t.get("inputs", {}).get("expected_kind") != expected_kind):
            continue
        return t
    return None


# ----------------------------- tests -----------------------------

def test_delete_session_removes_aggregate_and_404s_on_followups(
        client: _HandlerClient, project_root: Path) -> None:
    _install_stubs()
    state = client.create(objective="prune me",
                          project_root=str(project_root),
                          run_initial_task=True)
    sid = state["session"]["session_id"]
    # Sanity: the session is listed before deletion.
    assert any(s["session_id"] == sid
               for s in client.list_sessions(project_root=str(project_root)))
    resp = client.delete(sid, project_root=str(project_root))
    assert client.last_status == 200
    assert resp == {"deleted": sid}
    # Listing no longer includes it; subsequent reads return 404.
    assert not any(s["session_id"] == sid
                   for s in client.list_sessions(project_root=str(project_root)))
    client.get(sid, project_root=str(project_root))
    assert client.last_status == 404
    # Re-deleting yields 404 as well.
    client.delete(sid, project_root=str(project_root))
    assert client.last_status == 404


def test_delete_session_cancels_inflight_background_drain(
        client: _HandlerClient, project_root: Path) -> None:
    """DELETE stops a running drain so it can't outlive the session.

    A background drain whose task is still in flight when the session is
    deleted would otherwise write a child row against the now-gone parent
    and trip a FK error. Deletion must cancel the drain task, drop it from
    ``_SESSION_DRAINS``, and clear any pending cancel flag.
    """
    _install_stubs()
    routes._SESSION_DRAINS.clear()
    routes._CANCEL_REQUESTS.clear()
    state = client.create(objective="g", project_root=str(project_root),
                          run_initial_task=False)
    sid = state["session"]["session_id"]

    async def _never() -> None:
        await asyncio.sleep(60)

    async def _scenario() -> None:
        task = asyncio.create_task(_never())
        routes._SESSION_DRAINS[sid] = task
        resp = await routes.delete_session(sid, project_root=str(project_root))
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert resp == {"deleted": sid}
        assert task.cancelled()
        assert sid not in routes._SESSION_DRAINS
        assert sid not in routes._CANCEL_REQUESTS

    asyncio.new_event_loop().run_until_complete(_scenario())


def test_create_session_runs_initial_explore_and_spawns_ask(
        client: _HandlerClient, project_root: Path) -> None:
    _install_stubs()
    state = client.create(objective="improve retrieval",
                          project_root=str(project_root),
                          run_initial_task=True)
    assert client.last_status == 200
    assert state["session"]["original_objective"] == "improve retrieval"
    # EXPLORE has run and its ASK_USER(choose_path) successor was queued.
    explore = _find_task(state, kind="explore")
    assert explore is not None and explore["status"] == "done"
    ask = _find_task(state, kind="ask_user", expected_kind="choose_path")
    assert ask is not None and ask["status"] == "in_progress"
    # Surfaced anchor fact + directions artifact are visible to the UI.
    assert any(f["kind"] == "anchor" for f in state["facts"])
    assert any(a["kind"] == "directions_list" for a in state["artifacts"])


def test_choose_path_decision_drives_investigate_then_recommend(
        client: _HandlerClient, project_root: Path) -> None:
    _install_stubs()
    state = client.create(objective="g", project_root=str(project_root),
                          run_initial_task=True)
    sid = state["session"]["session_id"]
    ask = _find_task(state, kind="ask_user", expected_kind="choose_path")
    state = client.decision(sid, task_id=ask["task_id"],
                            chosen={"anchor_chunk_id": "c1", "title": "A"},
                            run_initial_task=True)
    # INVESTIGATE -> RECOMMEND -> ASK_USER(choose_recommendation) chain.
    inv = _find_task(state, kind="investigate")
    assert inv is not None and inv["status"] == "done"
    rec = _find_task(state, kind="recommend")
    assert rec is not None and rec["status"] == "done"
    pick = _find_task(state, kind="ask_user",
                      expected_kind="choose_recommendation")
    assert pick is not None and pick["status"] == "in_progress"


def test_full_write_loop_via_http(
        client: _HandlerClient, project_root: Path) -> None:
    """Explore -> path -> investigate -> recommend -> rec -> plan -> approve -> apply -> verify."""
    _install_stubs()
    state = client.create(objective="add caching",
                          project_root=str(project_root),
                          run_initial_task=True)
    sid = state["session"]["session_id"]

    # 1. choose_path
    ask_path = _find_task(state, kind="ask_user", expected_kind="choose_path")
    state = client.decision(sid, task_id=ask_path["task_id"],
                            chosen={"anchor_chunk_id": "c1", "title": "A"},
                            run_initial_task=True)

    # 2. choose_recommendation (plan_change)
    pick = _find_task(state, kind="ask_user",
                      expected_kind="choose_recommendation")
    state = client.decision(sid, task_id=pick["task_id"], chosen={
        "id": "r1", "title": "Cache results",
        "rationale": "speed", "kind": "plan_change"},
        run_initial_task=True)

    # PLAN_CHANGE has run and an APPROVE ASK is pending.
    plan_task = _find_task(state, kind="plan_change")
    assert plan_task is not None and plan_task["status"] == "done"
    approve = _find_task(state, kind="ask_user", expected_kind="approve")
    assert approve is not None and approve["status"] == "in_progress"
    assert approve["inputs"].get("plan_artifact_id")

    # 3. approve=True -> APPLY -> VERIFY (terminal)
    state = client.decision(sid, task_id=approve["task_id"],
                            chosen={"approved": True},
                            run_initial_task=True)
    apply_t = _find_task(state, kind="apply")
    assert apply_t is not None and apply_t["status"] == "done"
    verify_t = _find_task(state, kind="verify")
    assert verify_t is not None and verify_t["status"] == "done"
    kinds = {a["kind"] for a in state["artifacts"]}
    assert {"code_change_plan", "applied_changes",
            "verify_report"}.issubset(kinds)


def test_decline_approval_halts_loop(
        client: _HandlerClient, project_root: Path) -> None:
    _install_stubs()
    state = client.create(objective="g", project_root=str(project_root),
                          run_initial_task=True)
    sid = state["session"]["session_id"]
    ask_path = _find_task(state, kind="ask_user", expected_kind="choose_path")
    state = client.decision(sid, task_id=ask_path["task_id"],
                            chosen={"anchor_chunk_id": "c1", "title": "A"},
                            run_initial_task=True)
    pick = _find_task(state, kind="ask_user",
                      expected_kind="choose_recommendation")
    state = client.decision(sid, task_id=pick["task_id"], chosen={
        "id": "r1", "title": "Cache", "rationale": "",
        "kind": "plan_change"}, run_initial_task=True)
    approve = _find_task(state, kind="ask_user", expected_kind="approve")
    state = client.decision(sid, task_id=approve["task_id"],
                            chosen={"approved": False},
                            rationale="not yet", run_initial_task=True)
    assert _find_task(state, kind="apply") is None
    assert _find_task(state, kind="verify") is None
    approve_after = next(t for t in state["tasks"]
                         if t["task_id"] == approve["task_id"])
    assert approve_after["status"] == "done"


def test_decision_validation_rejects_missing_anchor(
        client: _HandlerClient, project_root: Path) -> None:
    _install_stubs()
    state = client.create(objective="g", project_root=str(project_root),
                          run_initial_task=True)
    sid = state["session"]["session_id"]
    ask = _find_task(state, kind="ask_user", expected_kind="choose_path")
    result = client.decision(sid, task_id=ask["task_id"], chosen={},
                             run_initial_task=False)
    assert result is None
    assert client.last_status == 400
    assert "anchor_chunk_id" in str(client.last_detail)


def test_get_snapshot_and_list_sessions(
        client: _HandlerClient, project_root: Path) -> None:
    _install_stubs()
    created = client.create(objective="g", project_root=str(project_root),
                            run_initial_task=False)
    sid = created["session"]["session_id"]
    # GET /{sid} returns a fresh snapshot with the same identifiers.
    snap = client.get(sid, project_root=str(project_root))
    assert snap["session"]["session_id"] == sid
    # GET / lists sessions filtered by project_root.
    listed = client.list_sessions(project_root=str(project_root))
    assert any(s["session_id"] == sid for s in listed)


def test_message_endpoint_runs_followup_explore(
        client: _HandlerClient, project_root: Path) -> None:
    """A followup message with no pending ASK spawns a sibling EXPLORE."""
    _install_stubs()
    # Walk to a state with no pending ASK by resolving choose_path then
    # picking the ``done`` recommendation kind (the router closes the
    # focus rather than spawning a new task).
    state = client.create(objective="g", project_root=str(project_root),
                          run_initial_task=True)
    sid = state["session"]["session_id"]
    ask = _find_task(state, kind="ask_user", expected_kind="choose_path")
    state = client.decision(sid, task_id=ask["task_id"],
                            chosen={"anchor_chunk_id": "c1", "title": "A"},
                            run_initial_task=True)
    pick = _find_task(state, kind="ask_user",
                      expected_kind="choose_recommendation")
    state = client.decision(sid, task_id=pick["task_id"], chosen={
        "id": "r0", "title": "Stop here", "rationale": "", "kind": "done"},
        run_initial_task=True)
    # No ASK pending now -- the followup message should spawn a sibling
    # EXPLORE, taking the count from 1 to 2.
    before_explores = len([t for t in state["tasks"]
                           if t["kind"] == "explore"])
    state = client.message(sid, message="different goal now",
                           run_initial_task=True)
    assert client.last_status == 200
    explores = [t for t in state["tasks"] if t["kind"] == "explore"]
    assert len(explores) > before_explores



# =====================================================================
# Greenfield route-level tests
# =====================================================================

def _install_greenfield_stubs() -> None:
    @register_executor(TaskKind.CLARIFY_REQUIREMENTS)
    def _clarify(task, deps):
        art = Artifact.new(
            task.session_id, task.task_id, ArtifactKind.REQUIREMENTS_SHEET,
            {"goal": task.inputs.get("goal"),
             "questions": [
                 {"id": "q1", "prompt": "Framework?",
                  "suggested": ["Flask"]},
                 {"id": "q2", "prompt": "Storage?",
                  "suggested": ["JSON on disk"]},
                 {"id": "q3", "prompt": "Auth?",
                  "suggested": ["None"]}],
             "source": "stub"})
        return ExecutorResult(
            outputs={"requirements_artifact_id": art.artifact_id,
                     "question_count": 3},
            artifact=art)

    @register_executor(TaskKind.DECOMPOSE)
    def _decompose(task, deps):
        art = Artifact.new(
            task.session_id, task.task_id, ArtifactKind.WORK_PLAN,
            {"prior_goal": task.inputs.get("prior_goal"),
             "composed_goal": task.inputs.get("prior_goal"),
             "answers": dict(task.inputs.get("answers") or {}),
             "plan_md": "## Plan\n- app.py",
             "layers": [{"name": "app", "files": [
                 {"path": "app.py", "description": "entry"}]}]})
        return ExecutorResult(
            outputs={"work_plan_artifact_id": art.artifact_id,
                     "file_count": 1, "layer_count": 1},
            artifact=art)

    @register_executor(TaskKind.SCAFFOLD)
    def _scaffold(task, deps):
        art = Artifact.new(
            task.session_id, task.task_id, ArtifactKind.SCAFFOLD_PATCHES,
            {"work_plan_artifact_id":
                task.inputs.get("work_plan_artifact_id"),
             "diffs": [{"file": "app.py", "patch": "+++ app.py\nbody"}],
             "generated": [{"file": "app.py", "bytes": 4, "layer": "app"}],
             "failed": []})
        return ExecutorResult(
            outputs={"scaffold_artifact_id": art.artifact_id,
                     "generated_count": 1, "failed_count": 0},
            artifact=art)

    @register_executor(TaskKind.APPLY)
    def _apply(task, deps):
        art = Artifact.new(
            task.session_id, task.task_id, ArtifactKind.APPLIED_CHANGES,
            {"plan_artifact_id": task.inputs.get("plan_artifact_id"),
             "source_artifact_kind": "scaffold_patches",
             "applied_files": ["app.py"], "failed_files": [],
             "smoke_ok": True})
        return ExecutorResult(
            outputs={"apply_artifact_id": art.artifact_id,
                     "applied_count": 1, "failed_count": 0}, artifact=art)

    @register_executor(TaskKind.BOOTSTRAP_ENV)
    def _bootstrap(task, deps):
        art = Artifact.new(
            task.session_id, task.task_id, ArtifactKind.BUILD_REPORT,
            {"apply_artifact_id": task.inputs.get("apply_artifact_id"),
             "project_type": "python",
             "venv_path": "/tmp/p/.venv",
             "python_exe": "/tmp/p/.venv/bin/python",
             "installed_from": ["requirements.txt"],
             "installed_packages": [], "failed_installs": [],
             "outcome": "succeeded", "pip_log_tail": "",
             "applied_files": ["app.py"]})
        return ExecutorResult(
            outputs={"build_artifact_id": art.artifact_id,
                     "outcome": "succeeded", "project_type": "python",
                     "venv_path": "/tmp/p/.venv",
                     "python_exe": "/tmp/p/.venv/bin/python",
                     "installed_count": 0, "failed_count": 0},
            artifact=art)

    @register_executor(TaskKind.API_CHECK)
    def _api_check(task, deps):
        art = Artifact.new(
            task.session_id, task.task_id, ArtifactKind.API_CHECK_REPORT,
            {"build_artifact_id": task.inputs.get("build_artifact_id"),
             "applied_files": ["app.py"], "references": [],
             "outcome": "skipped", "failed_references": [],
             "failure_signature": "", "probe_error": None})
        return ExecutorResult(
            outputs={"api_check_artifact_id": art.artifact_id,
                     "outcome": "skipped",
                     "failed_count": 0, "checked_count": 0,
                     "failure_signature": ""},
            artifact=art)

    @register_executor(TaskKind.SMOKE)
    def _smoke(task, deps):
        art = Artifact.new(
            task.session_id, task.task_id, ArtifactKind.SMOKE_REPORT,
            {"build_artifact_id": task.inputs.get("build_artifact_id"),
             "applied_files": ["app.py"], "modules": [],
             "outcome": "skipped", "failed_modules": [],
             "failure_signature": ""})
        return ExecutorResult(
            outputs={"smoke_artifact_id": art.artifact_id,
                     "outcome": "skipped",
                     "failed_count": 0, "tested_count": 0,
                     "failure_signature": ""},
            artifact=art)

    @register_executor(TaskKind.VERIFY)
    def _verify(task, deps):
        # Greenfield: no tests yet, so VERIFY emits a skipped report.
        art = Artifact.new(
            task.session_id, task.task_id, ArtifactKind.VERIFY_REPORT,
            {"ran": False, "tests_passed": False, "returncode": 0,
             "outcome": "skipped",
             "mode": task.inputs.get("mode") or "greenfield",
             "build_artifact_id": task.inputs.get("build_artifact_id"),
             "skipped_reason": "no tests discovered"})
        return ExecutorResult(
            outputs={"verify_artifact_id": art.artifact_id,
                     "ran": False, "tests_passed": False,
                     "outcome": "skipped"}, artifact=art)


def test_create_greenfield_session_auto_detects_empty_root(
        client: _HandlerClient, tmp_path: Path) -> None:
    """Empty project_root + no index -> mode resolves to greenfield."""
    _install_greenfield_stubs()
    state = client.create(
        objective="build a flask api that saves JSON to disk",
        project_root=str(tmp_path / "fresh"),
        mode=None,  # opt out of the default explore override
        run_initial_task=True)
    assert client.last_status == 200
    assert state["session"]["mode"] == "greenfield"
    # Root task is CLARIFY_REQUIREMENTS, not EXPLORE.
    clarify = _find_task(state, kind="clarify_requirements")
    assert clarify is not None and clarify["status"] == "done"
    ask = _find_task(state, kind="ask_user",
                     expected_kind="clarify_answers")
    assert ask is not None and ask["status"] == "in_progress"
    assert any(a["kind"] == "requirements_sheet"
               for a in state["artifacts"])


def test_create_greenfield_explicit_mode_overrides_detection(
        client: _HandlerClient, project_root: Path) -> None:
    """``mode='greenfield'`` is honored even when the project root exists."""
    _install_greenfield_stubs()
    project_root.mkdir(parents=True, exist_ok=True)
    (project_root / "existing.py").write_text("# pre-existing\n")
    state = client.create(objective="add a new module",
                          project_root=str(project_root),
                          mode="greenfield",
                          run_initial_task=True)
    assert state["session"]["mode"] == "greenfield"
    assert _find_task(state, kind="clarify_requirements") is not None
    assert _find_task(state, kind="explore") is None


def test_full_greenfield_loop_via_http(
        client: _HandlerClient, tmp_path: Path) -> None:
    """clarify -> answers -> decompose -> approve -> scaffold -> apply -> verify."""
    _install_greenfield_stubs()
    state = client.create(
        objective="flask json store", project_root=str(tmp_path / "p"),
        mode="greenfield", run_initial_task=True)
    sid = state["session"]["session_id"]

    # 1. Submit clarification answers.
    ans = _find_task(state, kind="ask_user",
                     expected_kind="clarify_answers")
    assert ans is not None
    state = client.decision(
        sid, task_id=ans["task_id"],
        chosen={"answers": {"q1": "Python + Flask",
                            "q2": "JSON on disk",
                            "q3": "None"}},
        run_initial_task=True)
    assert client.last_status == 200

    # 2. DECOMPOSE has run; approve_plan ASK is pending.
    dec = _find_task(state, kind="decompose")
    assert dec is not None and dec["status"] == "done"
    approve = _find_task(state, kind="ask_user",
                         expected_kind="approve_plan")
    assert approve is not None and approve["status"] == "in_progress"
    assert approve["inputs"].get("work_plan_artifact_id")

    # 3. Approve plan -> SCAFFOLD -> APPLY -> BOOTSTRAP_ENV -> VERIFY.
    state = client.decision(sid, task_id=approve["task_id"],
                            chosen={"approved": True},
                            run_initial_task=True)
    sc = _find_task(state, kind="scaffold")
    assert sc is not None and sc["status"] == "done"
    apply_t = _find_task(state, kind="apply")
    assert apply_t is not None and apply_t["status"] == "done"
    assert apply_t["inputs"].get("mode") == "greenfield"
    boot_t = _find_task(state, kind="bootstrap_env")
    assert boot_t is not None and boot_t["status"] == "done"
    assert boot_t["inputs"].get("apply_artifact_id")
    api_t = _find_task(state, kind="api_check")
    assert api_t is not None and api_t["status"] == "done"
    assert api_t["inputs"].get("build_artifact_id")
    smoke_t = _find_task(state, kind="smoke")
    assert smoke_t is not None and smoke_t["status"] == "done"
    assert smoke_t["inputs"].get("build_artifact_id")
    verify_t = _find_task(state, kind="verify")
    assert verify_t is not None and verify_t["status"] == "done"
    assert verify_t["inputs"].get("build_artifact_id")

    kinds = {a["kind"] for a in state["artifacts"]}
    assert {"requirements_sheet", "work_plan", "scaffold_patches",
            "applied_changes", "build_report", "api_check_report",
            "smoke_report", "verify_report"}.issubset(kinds)
    assert "directions_list" not in kinds


def _install_greenfield_stubs_with_api_repair() -> None:
    """Greenfield stubs whose API_CHECK fails once then regenerates.

    The first API_CHECK reports a hallucinated ``lance.connect`` symbol
    so the router spawns REPAIR; the REPAIR stub returns a
    ``strategy='regenerate'`` verdict, which re-queues a fresh SCAFFOLD
    and a second SCAFFOLD -> APPLY -> BOOTSTRAP_ENV -> API_CHECK ->
    SMOKE -> VERIFY cycle. The second API_CHECK is clean so the loop
    reaches VERIFY. The whole detour is 11 executor steps -- well past
    the old ``_drain_ready`` budget of 6.
    """
    _install_greenfield_stubs()
    api_calls = {"n": 0}

    @register_executor(TaskKind.API_CHECK)
    def _api_check_flaky(task, deps):
        api_calls["n"] += 1
        first = api_calls["n"] == 1
        failed_refs = [{"module": "lance", "name": "connect"}] if first else []
        outcome = "failed" if first else "skipped"
        sig = "api_check|lance.connect" if first else ""
        art = Artifact.new(
            task.session_id, task.task_id, ArtifactKind.API_CHECK_REPORT,
            {"build_artifact_id": task.inputs.get("build_artifact_id"),
             "applied_files": ["app.py"], "references": [],
             "outcome": outcome, "failed_references": failed_refs,
             "failure_signature": sig, "probe_error": None})
        return ExecutorResult(
            outputs={"api_check_artifact_id": art.artifact_id,
                     "outcome": outcome, "failed_count": len(failed_refs),
                     "checked_count": 1, "failure_signature": sig},
            artifact=art)

    @register_executor(TaskKind.REPAIR)
    def _repair_regenerate(task, deps):
        priors = task.inputs.get("prior_failure_signatures") \
            or ["api_check|lance.connect"]
        art = Artifact.new(
            task.session_id, task.task_id, ArtifactKind.REPAIR_PLAN,
            {"diffs": [], "classification": "api_check_failure",
             "rationale": "hallucinated lance.connect"})
        return ExecutorResult(
            outputs={"repair_artifact_id": art.artifact_id,
                     "classification": "api_check_failure",
                     "failure_signature": priors[-1],
                     "repair_attempt": task.inputs.get("repair_attempt", 1),
                     "diff_count": 0, "can_apply": False,
                     "strategy": "regenerate",
                     "extra_constraints": {
                         "kind": "api_check_failure",
                         "failed_references": [
                             {"module": "lance", "name": "connect"}],
                         "rationale": "regen"}},
            artifact=art)


def test_greenfield_api_check_repair_regenerate_drains_to_verify(
        client: _HandlerClient, tmp_path: Path) -> None:
    """A regenerate detour after an API_CHECK failure must reach VERIFY.

    Regression for the stall on session ``ses_e19caed6976d496b``: the
    drain budget used to be 6, exactly the happy-path pipeline length,
    so the first repair/regenerate cycle overran it and stranded the
    regenerated APPLY at READY, leaving the session stuck ``active``.
    """
    _install_greenfield_stubs_with_api_repair()
    state = client.create(
        objective="flask lancedb app", project_root=str(tmp_path / "p"),
        mode="greenfield", run_initial_task=True)
    sid = state["session"]["session_id"]

    ans = _find_task(state, kind="ask_user", expected_kind="clarify_answers")
    state = client.decision(
        sid, task_id=ans["task_id"],
        chosen={"answers": {"q1": "Flask", "q2": "LanceDB", "q3": "None"}},
        run_initial_task=True)
    approve = _find_task(state, kind="ask_user",
                         expected_kind="approve_plan")
    state = client.decision(sid, task_id=approve["task_id"],
                            chosen={"approved": True},
                            run_initial_task=True)

    # The regenerate cycle produced a second SCAFFOLD/APPLY pair, both
    # of which must have run (not the pre-fix stall where the second
    # APPLY was created READY and never dispatched).
    scaffolds = [t for t in state["tasks"] if t["kind"] == "scaffold"]
    assert len(scaffolds) == 2
    applies = [t for t in state["tasks"] if t["kind"] == "apply"]
    assert len(applies) == 2
    assert all(a["status"] == "done" for a in applies)
    repair_t = _find_task(state, kind="repair")
    assert repair_t is not None and repair_t["status"] == "done"
    verify_t = _find_task(state, kind="verify")
    assert verify_t is not None and verify_t["status"] == "done"
    # No task is left stranded at READY -- the stall symptom.
    assert not any(t["status"] == "ready" for t in state["tasks"])


def test_greenfield_decline_plan_halts_loop(
        client: _HandlerClient, tmp_path: Path) -> None:
    _install_greenfield_stubs()
    state = client.create(
        objective="flask app", project_root=str(tmp_path / "p"),
        mode="greenfield", run_initial_task=True)
    sid = state["session"]["session_id"]
    ans = _find_task(state, kind="ask_user",
                     expected_kind="clarify_answers")
    state = client.decision(
        sid, task_id=ans["task_id"],
        chosen={"answers": {"q1": "Flask"}},
        run_initial_task=True)
    approve = _find_task(state, kind="ask_user",
                         expected_kind="approve_plan")
    state = client.decision(sid, task_id=approve["task_id"],
                            chosen={"approved": False},
                            rationale="want a different stack",
                            run_initial_task=True)
    assert _find_task(state, kind="scaffold") is None
    assert _find_task(state, kind="apply") is None
    assert _find_task(state, kind="verify") is None
    approve_after = next(t for t in state["tasks"]
                         if t["task_id"] == approve["task_id"])
    assert approve_after["status"] == "done"


def test_greenfield_clarify_answers_validation_rejects_non_dict(
        client: _HandlerClient, tmp_path: Path) -> None:
    _install_greenfield_stubs()
    state = client.create(
        objective="flask", project_root=str(tmp_path / "p"),
        mode="greenfield", run_initial_task=True)
    sid = state["session"]["session_id"]
    ans = _find_task(state, kind="ask_user",
                     expected_kind="clarify_answers")
    result = client.decision(sid, task_id=ans["task_id"],
                             chosen={"answers": "not a dict"},
                             run_initial_task=False)
    assert result is None
    assert client.last_status == 400


def test_create_session_rejects_invalid_mode(
        client: _HandlerClient, tmp_path: Path) -> None:
    state = client.create(objective="g", project_root=str(tmp_path),
                          mode="bogus", run_initial_task=False)
    assert state is None
    assert client.last_status == 400
    assert "bogus" in str(client.last_detail)


# --------------------- P0: background drain + SSE ---------------------

def test_schedule_drain_backgrounds_and_coalesces(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A scheduled drain runs detached and a concurrent schedule coalesces.

    The POST handler must return without waiting for the drain, so
    ``_schedule_drain`` returns before the (stubbed) drain body runs, keeps
    exactly one task per session in ``_SESSION_DRAINS``, and clears the
    registry once the drain finishes.
    """
    monkeypatch.setattr(routes, "_RUN_DRAIN_IN_BACKGROUND", True)
    routes._SESSION_DRAINS.clear()
    order: list[str] = []

    async def _fake_drain(runner, session_id, deps, *, max_steps=64):
        order.append("start")
        await asyncio.sleep(0.02)
        order.append("end")

    monkeypatch.setattr(routes, "_drain_ready", _fake_drain)

    async def _scenario() -> None:
        await routes._schedule_drain(None, "sX", None)  # type: ignore[arg-type]
        # Returned before the drain body executed -> non-blocking.
        assert order == []
        first = routes._SESSION_DRAINS.get("sX")
        assert first is not None and not first.done()
        # A second schedule while the first runs must not spawn a duplicate.
        await routes._schedule_drain(None, "sX", None)  # type: ignore[arg-type]
        assert routes._SESSION_DRAINS.get("sX") is first
        await first
        await asyncio.sleep(0)  # let the done-callback fire
        assert order == ["start", "end"]
        assert "sX" not in routes._SESSION_DRAINS

    asyncio.new_event_loop().run_until_complete(_scenario())


def test_drain_ready_honours_cooperative_cancel(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A cancel requested mid-drive stops the loop before the next task.

    The flag is checked at the top of each iteration, so a cancel set
    while task N runs is honoured before task N+1 dispatches, and the
    flag is consumed (a later message can resume).
    """
    routes._CANCEL_REQUESTS.clear()
    calls: list[int] = []

    class _LiveStore:
        def get_session(self, sid: str) -> object:
            return object()  # session always present

    class _FakeRunner:
        store = _LiveStore()

        def run_next(self, *, session_id: str, deps: Any) -> object:
            calls.append(1)
            if len(calls) == 2:
                routes.request_cancel(session_id)
            return object()  # non-None -> the drain keeps looping

    async def _scenario() -> None:
        await routes._drain_ready(
            _FakeRunner(), "sC", None, max_steps=64)  # type: ignore[arg-type]

    asyncio.new_event_loop().run_until_complete(_scenario())
    # Tasks 1 and 2 ran; cancel set during task 2 -> checked at the top of
    # iteration 3 -> stop. Exactly two run_next calls, flag consumed.
    assert len(calls) == 2
    assert "sC" not in routes._CANCEL_REQUESTS


class _ToggleStore:
    """Store whose session presence flips off on demand (delete race)."""

    def __init__(self) -> None:
        self.exists = True

    def get_session(self, sid: str) -> object | None:
        return object() if self.exists else None


def test_drain_ready_stops_when_session_deleted_between_tasks() -> None:
    """A session deleted after task N halts the drive before task N+1."""
    routes._CANCEL_REQUESTS.clear()
    store = _ToggleStore()
    calls: list[int] = []

    class _FakeRunner:
        def __init__(self, s: _ToggleStore) -> None:
            self.store = s

        def run_next(self, *, session_id: str, deps: Any) -> object:
            calls.append(1)
            store.exists = False  # session removed out from under us
            return object()  # non-None -> would loop again but for the guard

    async def _scenario() -> None:
        await routes._drain_ready(
            _FakeRunner(store), "sD", None, max_steps=64)  # type: ignore[arg-type]

    asyncio.new_event_loop().run_until_complete(_scenario())
    # Iteration 2's vanished-session guard stops before dispatching again.
    assert len(calls) == 1


def test_drain_ready_swallows_error_when_session_deleted_mid_task() -> None:
    """A result-write error for a vanished session is a benign stop."""
    routes._CANCEL_REQUESTS.clear()
    store = _ToggleStore()

    class _FakeRunner:
        def __init__(self, s: _ToggleStore) -> None:
            self.store = s

        def run_next(self, *, session_id: str, deps: Any) -> object:
            store.exists = False  # deleted while this task ran
            raise RuntimeError("FOREIGN KEY constraint failed")

    async def _scenario() -> None:
        await routes._drain_ready(
            _FakeRunner(store), "sE", None, max_steps=64)  # type: ignore[arg-type]

    # Must not propagate -- the deleted session makes the FK error benign.
    asyncio.new_event_loop().run_until_complete(_scenario())


def test_drain_ready_reraises_error_when_session_still_exists() -> None:
    """A genuine error (session intact) still propagates out of the drain."""
    routes._CANCEL_REQUESTS.clear()

    class _LiveStore:
        def get_session(self, sid: str) -> object:
            return object()

    class _FakeRunner:
        store = _LiveStore()

        def run_next(self, *, session_id: str, deps: Any) -> object:
            raise RuntimeError("boom")

    async def _scenario() -> None:
        await routes._drain_ready(
            _FakeRunner(), "sF", None, max_steps=64)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.new_event_loop().run_until_complete(_scenario())


def test_post_cancel_flags_session_and_404s_on_unknown(
        client: _HandlerClient, project_root: Path) -> None:
    """POST /cancel flags a known session and 404s an unknown id."""
    _install_stubs()
    routes._CANCEL_REQUESTS.clear()
    state = client.create(objective="cancel me",
                          project_root=str(project_root),
                          run_initial_task=False)
    sid = state["session"]["session_id"]

    snap = client.cancel(sid)
    assert client.last_status == 200
    assert snap is not None and snap["session"]["session_id"] == sid
    assert sid in routes._CANCEL_REQUESTS

    assert client.cancel("nope-not-a-session") is None
    assert client.last_status == 404


def test_build_deps_missing_profile_is_400_not_500(
        project_root: Path) -> None:
    """P3.2: an unknown profile name surfaces as a clean 400, not a 500.

    ``provider_from_profile_name`` raises ``ValueError`` for a missing
    profile; ``_build_deps`` must translate that into an HTTP 400 with the
    underlying message so the UI can show a helpful error instead of a
    generic server fault.
    """
    provider = wm.ProviderConfig(use_profile=True,
                                 profile_name="does-not-exist-xyz")
    index = wm.IndexLocation()
    with pytest.raises(HTTPException) as exc:
        _REAL_BUILD_DEPS(provider, index, str(project_root), store=None)
    assert exc.value.status_code == 400
    assert "profile not found" in str(exc.value.detail).lower()


def test_session_events_streams_snapshot_then_partial(
        client: _HandlerClient, project_root: Path) -> None:
    """The SSE endpoint emits a snapshot first, then live bus events."""
    _install_stubs()
    state = client.create(objective="g", project_root=str(project_root),
                          run_initial_task=True)
    sid = state["session"]["session_id"]
    runner = routes._resolve_runner_for(sid)

    calls = {"n": 0}

    class _FakeRequest:
        async def is_disconnected(self) -> bool:
            calls["n"] += 1
            return calls["n"] > 1

    async def _scenario() -> None:
        resp = await routes.session_events(sid, _FakeRequest())  # type: ignore[arg-type]
        it = resp.body_iterator
        first = await it.__anext__()
        assert first["event"] == "snapshot"
        # Publish a progress beat for this session on the shared bus.
        runner.store.emit_task_progress(
            sid, "tX", {"index": 1, "total": 3, "path": "a.py"})
        second = await it.__anext__()
        assert second["event"] == "task.output_partial"
        payload = json.loads(second["data"])
        assert payload["payload"]["progress"]["path"] == "a.py"
        # Next disconnect check trips -> generator completes.
        with pytest.raises(StopAsyncIteration):
            await it.__anext__()

    asyncio.new_event_loop().run_until_complete(_scenario())


def test_emit_scaffold_progress_publishes_partial_with_eta(
        tmp_path: Path) -> None:
    """The scaffold helper publishes TASK_OUTPUT_PARTIAL and derives an ETA."""
    from cgx.session import SessionStore
    from cgx.session.events import EventType, get_default_bus
    from cgx.session.tasks.scaffold import _emit_scaffold_progress

    store = SessionStore(project_root=str(tmp_path / "sp"))
    seen: list[Any] = []
    unsub = get_default_bus().subscribe(
        EventType.TASK_OUTPUT_PARTIAL, lambda e: seen.append(e))

    class _Task:
        session_id = "s1"
        task_id = "t1"

    class _Deps:
        pass

    deps = _Deps()
    deps.store = store  # type: ignore[attr-defined]
    try:
        _emit_scaffold_progress(
            deps, _Task(), file="app/main.py", layer="api",  # type: ignore[arg-type]
            index=1, total=4, status="done", bytes=120, elapsed_ms=2000)
    finally:
        unsub()
        store.close()

    assert len(seen) == 1
    prog = seen[0].payload["progress"]
    assert prog["path"] == "app/main.py" and prog["index"] == 1
    # 2.0s/file * 3 remaining files.
    assert prog["eta_seconds"] == 6.0


def test_make_stream_beat_throttles_and_emits_stream_status(
        tmp_path: Path) -> None:
    """The on_token beat coalesces deltas into throttled ``stream`` frames."""
    from cgx.session import SessionStore
    from cgx.session.events import EventType, get_default_bus
    from cgx.session.tasks import scaffold as _scaffold

    store = SessionStore(project_root=str(tmp_path / "sb"))
    seen: list[Any] = []
    unsub = get_default_bus().subscribe(
        EventType.TASK_OUTPUT_PARTIAL, lambda e: seen.append(e))

    class _Task:
        session_id = "s1"
        task_id = "t1"

    class _Deps:
        pass

    deps = _Deps()
    deps.store = store  # type: ignore[attr-defined]
    try:
        beat = _scaffold._make_stream_beat(
            deps, _Task(), file="app/main.py", layer="api",  # type: ignore[arg-type]
            index=2, total=4)
        # First delta is throttled out (last==0 -> now-last < interval only
        # if clock is fresh); force determinism by pinning the interval to 0.
        orig = _scaffold._STREAM_BEAT_MIN_INTERVAL_S
        _scaffold._STREAM_BEAT_MIN_INTERVAL_S = 0.0
        try:
            beat("import os\n")
            beat("def main():\n")
        finally:
            _scaffold._STREAM_BEAT_MIN_INTERVAL_S = orig
    finally:
        unsub()
        store.close()

    assert seen, "at least one stream beat should be published"
    prog = seen[-1].payload["progress"]
    assert prog["status"] == "stream"
    assert prog["index"] == 2 and prog["total"] == 4
    assert prog["path"] == "app/main.py"
    # Cumulative streamed character count grows across deltas.
    assert prog["bytes"] == len("import os\n") + len("def main():\n")


def test_get_runner_reopens_store_when_db_file_is_gone(tmp_path: Path) -> None:
    """A cached store whose DB was moved away is rebuilt, not reused.

    The open fd survives the project folder being sent to the trash, so
    the live server kept writing session state into the unlinked file
    with no error (observed for Calculator2.7).
    """
    root = tmp_path / "proj"
    root.mkdir()
    first = routes._get_runner(str(root))
    assert Path(first.store.path).exists()
    routes._SESSION_TO_RUNNER["ses_stale"] = first

    shutil.move(str(root), str(tmp_path / "trashed"))
    second = routes._get_runner(str(root))

    assert second is not first
    assert Path(second.store.path).exists()
    # The session index must not keep pointing at the closed store.
    assert "ses_stale" not in routes._SESSION_TO_RUNNER


def test_get_runner_reuses_store_while_the_db_is_intact(tmp_path: Path) -> None:
    """The identity check must not defeat the per-root connection cache."""
    root = tmp_path / "proj"
    root.mkdir()
    first = routes._get_runner(str(root))
    assert routes._get_runner(str(root)) is first


def test_resolve_runner_for_rebuilds_a_stale_runner(tmp_path: Path) -> None:
    """Lookup by session id also goes through the staleness check."""
    root = tmp_path / "proj"
    root.mkdir()
    first = routes._get_runner(str(root))
    routes._SESSION_TO_RUNNER["ses_stale"] = first
    shutil.move(str(root), str(tmp_path / "trashed"))

    resolved = routes._resolve_runner_for("ses_stale")

    assert resolved is not first
    assert Path(resolved.store.path).exists()
