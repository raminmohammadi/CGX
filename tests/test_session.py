"""Phase 0 tests for the session-shaped agent backbone.

Covers the data layer in isolation: dataclass round-tripping through
SQLite, append-only fact semantics, decision-log integrity, and
synchronous event-bus delivery. The router, executors, and HTTP
surface arrive in later phases and have their own test modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from cgx.session import (
    Artifact,
    ArtifactKind,
    Decision,
    DecisionKind,
    Event,
    EventBus,
    EventType,
    Fact,
    FactKind,
    Router,
    Session,
    SessionRunner,
    SessionStatus,
    SessionStore,
    TaskKind,
    TaskNode,
    TaskNodeStatus,
)
from cgx.session.router import (
    AttachDecisionToTask,
    CreateTask,
    RecordDecision,
    UpdateSessionStatus,
    UpdateTaskStatus,
)
from cgx.session.tasks.ask import build_decision
from cgx.session.tasks.base import (
    ExecutorDeps,
    ExecutorResult,
    register_executor,
)


@pytest.fixture()
def store(tmp_path: Path) -> SessionStore:
    db = tmp_path / "sessions.db"
    bus = EventBus()
    s = SessionStore(db_path=db, bus=bus)
    yield s
    s.close()


@pytest.fixture()
def bus() -> EventBus:
    return EventBus()


@pytest.fixture(autouse=True)
def _restore_task_registry():
    """Snapshot the executor registry around every test.

    Phase 1 tests swap in stub executors via :func:`register_executor`;
    without this fixture the swap would leak to subsequent tests and
    -- more importantly -- to other test modules that rely on the real
    executors being present at import-registered defaults.
    """
    from cgx.session.tasks.base import _REGISTRY
    snapshot = dict(_REGISTRY)
    yield
    _REGISTRY.clear()
    _REGISTRY.update(snapshot)


@pytest.fixture(autouse=True)
def _reset_agent_log():
    """Close + drop cached agent.log handlers between tests.

    Tests use ``tmp_path``-rooted projects; without this the rotating
    handler from a previous test would point at a deleted directory and
    crash on the next emit.
    """
    from cgx.session.agent_log import reset_for_tests
    reset_for_tests()
    yield
    reset_for_tests()


# --------------------- models ---------------------

def test_session_new_truncates_long_title():
    s = Session.new("a" * 200)
    assert s.title.endswith("...")
    assert len(s.title) == 80
    assert s.original_objective == "a" * 200
    assert s.status is SessionStatus.ACTIVE


def test_tasknode_new_status_depends_on_blockers():
    ready = TaskNode.new("ses_x", TaskKind.EXPLORE, "explore")
    assert ready.status is TaskNodeStatus.READY
    blocked = TaskNode.new("ses_x", TaskKind.RECOMMEND, "rec",
                           blockers=["task_y"])
    assert blocked.status is TaskNodeStatus.BLOCKED
    assert blocked.blockers == ["task_y"]


def test_dataclass_to_dict_uses_enum_string_values():
    f = Fact.new("ses_x", FactKind.ANCHOR, {"chunk_id": "a.py::foo"})
    d = f.to_dict()
    assert d["kind"] == "anchor"
    assert d["content"]["chunk_id"] == "a.py::foo"
    assert d["stale"] is False


# --------------------- store round-trips ---------------------

def test_save_and_get_session_round_trip(store: SessionStore):
    s = Session.new("improve retrieval accuracy", project_root="/p")
    store.save_session(s)
    loaded = store.get_session(s.session_id)
    assert loaded is not None
    assert loaded.original_objective == s.original_objective
    assert loaded.project_root == "/p"
    assert loaded.status is SessionStatus.ACTIVE


def test_list_sessions_filters_by_project_root(store: SessionStore):
    a = Session.new("goal a", project_root="/p1")
    b = Session.new("goal b", project_root="/p2")
    store.save_session(a)
    store.save_session(b)
    ids_p1 = [x.session_id for x in
              store.list_sessions(project_root="/p1")]
    assert ids_p1 == [a.session_id]
    assert len(store.list_sessions()) == 2


def test_delete_session_cascades(store: SessionStore):
    s = Session.new("g")
    store.save_session(s)
    t = TaskNode.new(s.session_id, TaskKind.EXPLORE, "explore")
    store.save_task(t)
    f = Fact.new(s.session_id, FactKind.FILE, {"path": "a.py"})
    store.add_fact(f)
    assert store.delete_session(s.session_id) is True
    assert store.get_session(s.session_id) is None
    # Cascade should have removed dependent rows too.
    assert store.list_tasks(s.session_id) == []
    assert store.load_kb(s.session_id).facts == {}


def test_task_round_trip_preserves_inputs_and_status(store: SessionStore):
    s = Session.new("g"); store.save_session(s)
    t = TaskNode.new(s.session_id, TaskKind.RECOMMEND, "Recommend",
                     inputs={"prior_goal": "g"},
                     blockers=["task_a"])
    store.save_task(t)
    t.status = TaskNodeStatus.DONE
    t.outputs = {"recommendations": [{"id": 1, "title": "x"}]}
    store.save_task(t)
    again = store.get_task(t.task_id)
    assert again is not None
    assert again.status is TaskNodeStatus.DONE
    assert again.inputs["prior_goal"] == "g"
    assert again.outputs["recommendations"][0]["title"] == "x"
    assert again.blockers == ["task_a"]


def test_tasks_by_status(store: SessionStore):
    s = Session.new("g"); store.save_session(s)
    ready = TaskNode.new(s.session_id, TaskKind.EXPLORE, "e")
    blocked = TaskNode.new(s.session_id, TaskKind.RECOMMEND, "r",
                           blockers=[ready.task_id])
    store.save_task(ready)
    store.save_task(blocked)
    by_blocked = store.tasks_by_status(s.session_id,
                                       TaskNodeStatus.BLOCKED)
    assert [x.task_id for x in by_blocked] == [blocked.task_id]



# --------------------- knowledge base ---------------------

def test_kb_load_returns_inserted_facts_in_order(store: SessionStore):
    s = Session.new("g"); store.save_session(s)
    f1 = Fact.new(s.session_id, FactKind.FILE, {"path": "a.py"})
    f2 = Fact.new(s.session_id, FactKind.SYMBOL,
                  {"symbol": "foo", "path": "a.py"})
    store.add_fact(f1)
    store.add_fact(f2)
    kb = store.load_kb(s.session_id)
    assert set(kb.facts) == {f1.fact_id, f2.fact_id}
    assert kb.of_kind(FactKind.FILE)[0].content["path"] == "a.py"


def test_kb_find_anchor_locates_by_chunk_id(store: SessionStore):
    s = Session.new("g"); store.save_session(s)
    fact = Fact.new(s.session_id, FactKind.ANCHOR,
                    {"chunk_id": "src/a.py::foo", "path": "src/a.py"})
    store.add_fact(fact)
    kb = store.load_kb(s.session_id)
    hit = kb.find_anchor("src/a.py::foo")
    assert hit is not None and hit.fact_id == fact.fact_id


def test_mark_facts_stale_is_append_only(store: SessionStore):
    s = Session.new("g"); store.save_session(s)
    f = Fact.new(s.session_id, FactKind.FILE, {"path": "a.py"})
    store.add_fact(f)
    n = store.mark_facts_stale(s.session_id, [f.fact_id])
    assert n == 1
    kb = store.load_kb(s.session_id)
    # Content survives; only the staleness marker flips.
    assert kb.facts[f.fact_id].stale is True
    assert kb.facts[f.fact_id].content["path"] == "a.py"


# --------------------- decisions ---------------------

def test_decision_log_is_indexed_by_resolved_task(store: SessionStore):
    s = Session.new("g"); store.save_session(s)
    ask = TaskNode.new(s.session_id, TaskKind.ASK_USER,
                       "Which direction?")
    store.save_task(ask)
    dec = Decision.new(s.session_id, ask.task_id,
                       DecisionKind.CHOOSE_PATH,
                       question="Which direction?",
                       chosen={"anchor_chunk_id": "src/a.py::foo",
                               "title": "Foo"})
    store.record_decision(dec)
    log = store.load_decisions(s.session_id)
    assert dec.decision_id in log.decisions
    found = log.for_task(ask.task_id)
    assert found is not None
    assert found.chosen["anchor_chunk_id"] == "src/a.py::foo"


# --------------------- artifacts ---------------------

def test_artifact_round_trip(store: SessionStore):
    s = Session.new("g"); store.save_session(s)
    t = TaskNode.new(s.session_id, TaskKind.RECOMMEND, "rec")
    store.save_task(t)
    art = Artifact.new(s.session_id, t.task_id,
                       ArtifactKind.RECOMMENDATION_LIST,
                       {"recommendations": [{"id": 1, "title": "x"}]})
    store.save_artifact(art)
    again = store.get_artifact(art.artifact_id)
    assert again is not None
    assert again.kind is ArtifactKind.RECOMMENDATION_LIST
    listed = store.list_artifacts(s.session_id)
    assert [a.artifact_id for a in listed] == [art.artifact_id]


# --------------------- event bus ---------------------

def test_bus_delivers_to_type_specific_subscribers(bus: EventBus):
    received: List[Event] = []
    bus.subscribe(EventType.SESSION_CREATED, received.append)
    bus.publish(Event(EventType.SESSION_CREATED, "ses_a", {"x": 1}))
    bus.publish(Event(EventType.TASK_CREATED, "ses_a", {}))
    assert len(received) == 1
    assert received[0].session_id == "ses_a"
    assert received[0].payload == {"x": 1}


def test_bus_wildcard_subscriber_receives_all(bus: EventBus):
    received: List[Event] = []
    bus.subscribe("*", received.append)
    bus.publish(Event(EventType.SESSION_CREATED, "ses_a", {}))
    bus.publish(Event(EventType.TASK_CREATED, "ses_a", {}))
    assert [e.type for e in received] == [
        EventType.SESSION_CREATED, EventType.TASK_CREATED]


def test_bus_unsubscribe_thunk_removes_listener(bus: EventBus):
    received: List[Event] = []
    unsub = bus.subscribe(EventType.FACT_ADDED, received.append)
    bus.publish(Event(EventType.FACT_ADDED, "ses_a", {}))
    unsub()
    bus.publish(Event(EventType.FACT_ADDED, "ses_a", {}))
    assert len(received) == 1


def test_bus_subscriber_exception_does_not_block_others(bus: EventBus):
    received: List[Event] = []

    def bad(_evt: Event) -> None:
        raise RuntimeError("boom")

    bus.subscribe(EventType.TASK_CREATED, bad)
    bus.subscribe(EventType.TASK_CREATED, received.append)
    bus.publish(Event(EventType.TASK_CREATED, "ses_a", {}))
    assert len(received) == 1


# --------------------- store-to-bus integration ---------------------

def test_store_emits_session_created_then_updated(tmp_path: Path):
    bus = EventBus()
    seen: List[Event] = []
    bus.subscribe("*", seen.append)
    s_store = SessionStore(db_path=tmp_path / "db.sqlite", bus=bus)
    try:
        s = Session.new("g")
        s_store.save_session(s)
        s.title = "renamed"
        s_store.save_session(s)
    finally:
        s_store.close()
    types = [e.type for e in seen if e.session_id == s.session_id]
    assert EventType.SESSION_CREATED in types
    assert EventType.SESSION_UPDATED in types


def test_store_emits_task_status_changed_and_completed(store: SessionStore):
    seen: List[Event] = []
    store._bus.subscribe("*", seen.append)  # noqa: SLF001 - test seam
    s = Session.new("g"); store.save_session(s)
    t = TaskNode.new(s.session_id, TaskKind.EXPLORE, "e")
    store.save_task(t)
    t.status = TaskNodeStatus.IN_PROGRESS
    store.save_task(t)
    t.status = TaskNodeStatus.DONE
    store.save_task(t)
    types = [e.type for e in seen]
    assert EventType.TASK_CREATED in types
    assert EventType.TASK_STATUS_CHANGED in types
    assert EventType.TASK_COMPLETED in types


# =====================================================================
# Phase 1 -- router, executors, runner
# =====================================================================

# --------------------- router (no IO) ---------------------

def test_router_first_message_spawns_root_explore():
    session = Session.new("improve retrieval")
    plan = Router().on_user_message(
        session=session, message="improve retrieval", tasks=[])
    assert len(plan) == 1
    action = plan.actions[0]
    assert isinstance(action, CreateTask)
    assert action.task.kind is TaskKind.EXPLORE
    assert action.task.inputs["goal"] == "improve retrieval"
    assert action.task.parent_task_id is None


def test_router_followup_with_pending_ask_is_noop():
    session = Session.new("g")
    explore = TaskNode.new(session.session_id, TaskKind.EXPLORE, "e")
    explore.status = TaskNodeStatus.DONE
    ask = TaskNode.new(session.session_id, TaskKind.ASK_USER,
                       "pick", parent_task_id=explore.task_id,
                       inputs={"expected_kind": "choose_path"})
    plan = Router().on_user_message(
        session=session, message="ignore me", tasks=[explore, ask])
    assert len(plan) == 0


def test_router_followup_without_pending_ask_spawns_sibling_explore():
    session = Session.new("g")
    done = TaskNode.new(session.session_id, TaskKind.EXPLORE, "e")
    done.status = TaskNodeStatus.DONE
    plan = Router().on_user_message(
        session=session, message="next goal", tasks=[done])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    assert creates[0].task.kind is TaskKind.EXPLORE
    assert creates[0].task.inputs["goal"] == "next goal"


def test_router_explore_completion_spawns_ask_user():
    session = Session.new("g")
    explore = TaskNode.new(session.session_id, TaskKind.EXPLORE, "e",
                           inputs={"goal": "g"})
    explore.produced_artifact_id = "art_123"
    explore.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=explore, tasks=[explore])
    assert len(plan) == 1
    action = plan.actions[0]
    assert isinstance(action, CreateTask)
    assert action.task.kind is TaskKind.ASK_USER
    assert action.task.parent_task_id == explore.task_id
    assert action.task.inputs["directions_artifact_id"] == "art_123"
    assert action.task.inputs["expected_kind"] == "choose_path"


def test_router_decision_marks_ask_done_and_attaches():
    session = Session.new("g")
    ask = TaskNode.new(session.session_id, TaskKind.ASK_USER, "pick",
                       inputs={"expected_kind": "choose_path"})
    decision = Decision.new(
        session.session_id, ask.task_id, DecisionKind.CHOOSE_PATH,
        "pick one", {"anchor_chunk_id": "c1"})
    plan = Router().on_decision_recorded(
        session=session, decision=decision, tasks=[ask])
    kinds = [type(a) for a in plan.actions]
    assert RecordDecision in kinds
    assert AttachDecisionToTask in kinds
    upd = [a for a in plan.actions if isinstance(a, UpdateTaskStatus)]
    assert upd and upd[0].status is TaskNodeStatus.DONE
    assert upd[0].task_id == ask.task_id


def test_router_decision_against_wrong_task_kind_is_noop():
    session = Session.new("g")
    explore = TaskNode.new(session.session_id, TaskKind.EXPLORE, "e")
    decision = Decision.new(
        session.session_id, explore.task_id, DecisionKind.CHOOSE_PATH,
        "?", {"anchor_chunk_id": "x"})
    plan = Router().on_decision_recorded(
        session=session, decision=decision, tasks=[explore])
    assert len(plan) == 0


# --------------------- runner integration (stub executor) ---------------------

def _install_stub_explore():
    """Swap in a deterministic stub for the EXPLORE executor.

    The :func:`_restore_task_registry` autouse fixture reverts this
    after each test, so callers don't need to teardown.
    """
    @register_executor(TaskKind.EXPLORE)
    def _stub(task, deps):
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
            Fact.new(task.session_id, FactKind.ANCHOR,
                     {"chunk_id": "c2", "title": "B"},
                     surfaced_in_task_id=task.task_id),
        ]
        return ExecutorResult(
            outputs={"options_count": 2,
                     "directions_artifact_id": artifact.artifact_id},
            facts=facts, artifact=artifact)


def test_runner_start_session_persists_root_explore(store):
    _install_stub_explore()
    runner = SessionRunner(store)
    session = runner.start_session(objective="improve retrieval")
    tasks = store.list_tasks(session.session_id)
    assert len(tasks) == 1
    assert tasks[0].kind is TaskKind.EXPLORE
    assert tasks[0].status is TaskNodeStatus.READY
    assert session.root_task_id == tasks[0].task_id


def test_runner_executes_explore_then_spawns_ask(store):
    _install_stub_explore()
    runner = SessionRunner(store)
    session = runner.start_session(objective="g")
    task = runner.run_next(session_id=session.session_id,
                           deps=ExecutorDeps())
    assert task is not None
    assert task.kind is TaskKind.EXPLORE
    assert task.status is TaskNodeStatus.DONE
    assert task.produced_artifact_id is not None
    # Artifact + facts persisted.
    arts = store.list_artifacts(session.session_id)
    assert len(arts) == 1
    assert arts[0].kind is ArtifactKind.DIRECTIONS_LIST
    kb = store.load_kb(session.session_id)
    assert len([f for f in kb.facts.values()
                if f.kind is FactKind.ANCHOR]) == 2
    # ASK_USER successor was spawned by the router.
    tasks = store.list_tasks(session.session_id)
    asks = [t for t in tasks if t.kind is TaskKind.ASK_USER]
    assert len(asks) == 1
    assert asks[0].inputs["directions_artifact_id"] == arts[0].artifact_id


def test_runner_ask_user_stays_in_progress_then_decision_completes(store):
    _install_stub_explore()
    runner = SessionRunner(store)
    session = runner.start_session(objective="g")
    runner.run_next(session_id=session.session_id, deps=ExecutorDeps())
    runner.run_next(session_id=session.session_id, deps=ExecutorDeps())
    ask = next(t for t in store.list_tasks(session.session_id)
               if t.kind is TaskKind.ASK_USER)
    assert ask.status is TaskNodeStatus.IN_PROGRESS

    decision = build_decision(
        session_id=session.session_id, task=ask,
        chosen={"anchor_chunk_id": "c1", "title": "A"},
        rationale="prefer option A")
    runner.post_decision(session_id=session.session_id, decision=decision)

    ask_after = store.get_task(ask.task_id)
    assert ask_after.status is TaskNodeStatus.DONE
    assert decision.decision_id in ask_after.consumed_decision_ids
    log = store.load_decisions(session.session_id)
    assert decision.decision_id in log.decisions


def test_runner_run_next_returns_none_when_empty(store):
    _install_stub_explore()
    runner = SessionRunner(store)
    # No session, no tasks -- but we need a session to call run_next.
    session = runner.start_session(objective="g")
    runner.run_next(session_id=session.session_id, deps=ExecutorDeps())
    runner.run_next(session_id=session.session_id, deps=ExecutorDeps())
    # ASK_USER now sitting at IN_PROGRESS; nothing READY remains.
    assert runner.run_next(session_id=session.session_id,
                           deps=ExecutorDeps()) is None


def test_runner_failed_executor_marks_task_failed(store):
    @register_executor(TaskKind.EXPLORE)
    def _boom(task, deps):
        return ExecutorResult(failure="nope")

    runner = SessionRunner(store)
    session = runner.start_session(objective="g")
    task = runner.run_next(session_id=session.session_id,
                           deps=ExecutorDeps())
    assert task.status is TaskNodeStatus.FAILED
    assert task.error == "nope"
    # No successor spawned on failure.
    tasks = store.list_tasks(session.session_id)
    assert all(t.kind is TaskKind.EXPLORE for t in tasks)
    # Explore sessions keep their user-driven lifecycle -- still active.
    assert store.get_session(session.session_id).status \
        is SessionStatus.ACTIVE


def test_runner_greenfield_hard_failure_marks_session_failed(tmp_path, store):
    """A hard executor failure in greenfield ends the session FAILED.

    Hard failures (``ExecutorResult.failure``) never produce outputs, so
    the successor table can't run; the runner must still route them
    through :meth:`Router.on_task_failed` so the session reaches a
    terminal status instead of hanging in ``active``.
    """
    from cgx.session.models import SessionMode

    @register_executor(TaskKind.BOOTSTRAP_ENV)
    def _boom(task, deps):
        return ExecutorResult(failure="bootstrap failed: ['main']")

    session = Session.new("g", project_root=str(tmp_path),
                          mode=SessionMode.GREENFIELD)
    store.save_session(session)
    task = TaskNode.new(
        session.session_id, TaskKind.BOOTSTRAP_ENV, "bootstrap",
        inputs={"mode": SessionMode.GREENFIELD.value})
    task.status = TaskNodeStatus.READY
    store.save_task(task)

    out = SessionRunner(store).run_next(
        session_id=session.session_id, deps=ExecutorDeps())
    assert out.status is TaskNodeStatus.FAILED
    assert store.get_session(session.session_id).status \
        is SessionStatus.FAILED


# --------------------- E1: per-session budget ---------------------

def _seed_two_summarize(store, session):
    """Seed two independent READY work tasks with deterministic order.

    SUMMARIZE has no successor in the router table, so each completes
    without spawning children -- leaving the peer READY so the budget
    check bites on the second ``run_next``.
    """
    a = TaskNode.new(session.session_id, TaskKind.SUMMARIZE, "a")
    a.status = TaskNodeStatus.READY
    a.created_at = 1.0
    store.save_task(a)
    b = TaskNode.new(session.session_id, TaskKind.SUMMARIZE, "b")
    b.status = TaskNodeStatus.READY
    b.created_at = 2.0
    store.save_task(b)
    return a, b


def test_runner_task_budget_escalates_to_ask_user(store):
    @register_executor(TaskKind.SUMMARIZE)
    def _ok(task, deps):
        return ExecutorResult(outputs={"ok": True})

    session = Session.new("g", max_task_runs=1)
    store.save_session(session)
    a, b = _seed_two_summarize(store, session)
    runner = SessionRunner(store)

    # First work task runs and consumes the one-run budget.
    first = runner.run_next(session_id=session.session_id,
                            deps=ExecutorDeps())
    assert first.task_id == a.task_id
    assert first.status is TaskNodeStatus.DONE
    assert store.get_session(session.session_id).task_runs == 1

    # Second run trips the budget *before* dispatching -> escalation.
    out = runner.run_next(session_id=session.session_id, deps=ExecutorDeps())
    assert out.task_id == b.task_id
    assert out.status is TaskNodeStatus.BLOCKED
    sess = store.get_session(session.session_id)
    assert sess.status is SessionStatus.PAUSED
    asks = [t for t in store.list_tasks(session.session_id)
            if t.kind is TaskKind.ASK_USER]
    assert len(asks) == 1
    assert "task budget" in asks[0].inputs["reason"]

    # The loop quiesces: the only READY task left is the exempt ASK_USER,
    # which surfaces (IN_PROGRESS); nothing runs after it.
    surfaced = runner.run_next(session_id=session.session_id,
                               deps=ExecutorDeps())
    assert surfaced.kind is TaskKind.ASK_USER
    assert surfaced.status is TaskNodeStatus.IN_PROGRESS
    assert runner.run_next(session_id=session.session_id,
                           deps=ExecutorDeps()) is None
    # The exempt ASK_USER did not consume the budget.
    assert store.get_session(session.session_id).task_runs == 1


def test_runner_task_budget_headless_fails_terminally(store):
    @register_executor(TaskKind.SUMMARIZE)
    def _ok(task, deps):
        return ExecutorResult(outputs={"ok": True})

    session = Session.new("g", max_task_runs=1, headless=True)
    store.save_session(session)
    a, b = _seed_two_summarize(store, session)
    runner = SessionRunner(store)
    runner.run_next(session_id=session.session_id, deps=ExecutorDeps())
    out = runner.run_next(session_id=session.session_id, deps=ExecutorDeps())
    assert out.task_id == b.task_id
    assert out.status is TaskNodeStatus.ABANDONED
    assert store.get_session(session.session_id).status \
        is SessionStatus.FAILED
    # Headless mode never prompts a user.
    assert not [t for t in store.list_tasks(session.session_id)
                if t.kind is TaskKind.ASK_USER]
    assert runner.run_next(session_id=session.session_id,
                           deps=ExecutorDeps()) is None


def test_runner_wall_clock_budget_escalates(store):
    @register_executor(TaskKind.SUMMARIZE)
    def _ok(task, deps):
        return ExecutorResult(outputs={"ok": True})

    # A zero-second wall budget: the anchor is set on the first work
    # task, so the second run is always past the cap.
    session = Session.new("g", max_wall_seconds=0.0)
    store.save_session(session)
    a, b = _seed_two_summarize(store, session)
    runner = SessionRunner(store)
    first = runner.run_next(session_id=session.session_id,
                            deps=ExecutorDeps())
    assert first.status is TaskNodeStatus.DONE
    assert store.get_session(
        session.session_id).first_task_started_at is not None
    out = runner.run_next(session_id=session.session_id, deps=ExecutorDeps())
    assert out.status is TaskNodeStatus.BLOCKED
    sess = store.get_session(session.session_id)
    assert sess.status is SessionStatus.PAUSED
    asks = [t for t in store.list_tasks(session.session_id)
            if t.kind is TaskKind.ASK_USER]
    assert "time budget" in asks[0].inputs["reason"]


def test_runner_no_budget_configured_runs_all_tasks(store):
    @register_executor(TaskKind.SUMMARIZE)
    def _ok(task, deps):
        return ExecutorResult(outputs={"ok": True})

    session = Session.new("g")  # no caps configured
    store.save_session(session)
    a, b = _seed_two_summarize(store, session)
    runner = SessionRunner(store)
    runner.run_next(session_id=session.session_id, deps=ExecutorDeps())
    runner.run_next(session_id=session.session_id, deps=ExecutorDeps())
    tasks = {t.task_id: t for t in store.list_tasks(session.session_id)}
    assert tasks[a.task_id].status is TaskNodeStatus.DONE
    assert tasks[b.task_id].status is TaskNodeStatus.DONE
    assert store.get_session(session.session_id).status \
        is SessionStatus.ACTIVE
    assert not [t for t in store.list_tasks(session.session_id)
                if t.kind is TaskKind.ASK_USER]


def test_build_decision_rejects_choose_path_without_anchor():
    session = Session.new("g")
    ask = TaskNode.new(session.session_id, TaskKind.ASK_USER, "pick",
                       inputs={"expected_kind": "choose_path"})
    with pytest.raises(ValueError):
        build_decision(session_id=session.session_id, task=ask, chosen={})



# =====================================================================
# Phase 2 -- router CHOOSE_PATH -> INVESTIGATE -> RECOMMEND -> ASK
# =====================================================================

# --------------------- router (no IO) ---------------------

def test_router_choose_path_decision_spawns_investigate():
    session = Session.new("g")
    ask = TaskNode.new(session.session_id, TaskKind.ASK_USER, "pick",
                       inputs={"expected_kind": "choose_path",
                               "directions_artifact_id": "art_dirs",
                               "prior_goal": "improve retrieval"})
    decision = Decision.new(
        session.session_id, ask.task_id, DecisionKind.CHOOSE_PATH,
        "pick", {"anchor_chunk_id": "c1", "title": "A",
                 "rationale": "best fit"})
    plan = Router().on_decision_recorded(
        session=session, decision=decision, tasks=[ask])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    inv = creates[0].task
    assert inv.kind is TaskKind.INVESTIGATE
    assert inv.parent_task_id == ask.task_id
    assert inv.inputs["anchor_chunk_id"] == "c1"
    assert inv.inputs["title"] == "A"
    assert inv.inputs["prior_goal"] == "improve retrieval"
    assert inv.inputs["directions_artifact_id"] == "art_dirs"
    assert inv.inputs["decision_id"] == decision.decision_id


def test_router_investigate_completion_spawns_recommend():
    session = Session.new("g")
    inv = TaskNode.new(session.session_id, TaskKind.INVESTIGATE, "inv",
                      inputs={"anchor_chunk_id": "c1",
                              "prior_goal": "g", "title": "A"})
    inv.produced_artifact_id = "art_findings"
    inv.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=inv, tasks=[inv])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    rec = creates[0].task
    assert rec.kind is TaskKind.RECOMMEND
    assert rec.parent_task_id == inv.task_id
    assert rec.inputs["findings_artifact_id"] == "art_findings"
    assert rec.inputs["anchor_chunk_id"] == "c1"
    assert rec.inputs["prior_goal"] == "g"


def test_router_recommend_completion_spawns_ask_choose_recommendation():
    session = Session.new("g")
    rec = TaskNode.new(session.session_id, TaskKind.RECOMMEND, "rec",
                       inputs={"prior_goal": "g"})
    rec.produced_artifact_id = "art_recs"
    rec.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=rec, tasks=[rec])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    ask = creates[0].task
    assert ask.kind is TaskKind.ASK_USER
    assert ask.parent_task_id == rec.task_id
    assert ask.inputs["expected_kind"] == "choose_recommendation"
    assert ask.inputs["recommendations_artifact_id"] == "art_recs"
    assert ask.inputs["prior_goal"] == "g"


# --------------------- runner integration (stub executors) ---------------------

def _install_stub_investigate():
    @register_executor(TaskKind.INVESTIGATE)
    def _stub(task, deps):
        artifact = Artifact.new(
            session_id=task.session_id,
            produced_by_task_id=task.task_id,
            kind=ArtifactKind.FINDINGS_BUNDLE,
            content={
                "anchor_chunk_id": task.inputs.get("anchor_chunk_id"),
                "answer_md": "stub findings",
                "sources": [
                    {"chunk_id": task.inputs.get("anchor_chunk_id"),
                     "path": "pkg/mod.py", "symbol": "foo"},
                ],
            },
        )
        facts = [
            Fact.new(task.session_id, FactKind.SYMBOL,
                     {"symbol": "foo", "path": "pkg/mod.py"},
                     surfaced_in_task_id=task.task_id),
        ]
        return ExecutorResult(
            outputs={"findings_artifact_id": artifact.artifact_id},
            facts=facts, artifact=artifact)


def _install_stub_recommend():
    @register_executor(TaskKind.RECOMMEND)
    def _stub(task, deps):
        artifact = Artifact.new(
            session_id=task.session_id,
            produced_by_task_id=task.task_id,
            kind=ArtifactKind.RECOMMENDATION_LIST,
            content={
                "findings_artifact_id":
                    task.inputs.get("findings_artifact_id"),
                "recommendations": [
                    {"id": "r1", "title": "Investigate deeper",
                     "rationale": "see foo", "kind": "investigate_more",
                     "anchor_chunk_id":
                         task.inputs.get("anchor_chunk_id")},
                    {"id": "r2", "title": "Plan the change",
                     "rationale": "ready", "kind": "plan_change"},
                ],
            },
        )
        return ExecutorResult(
            outputs={"recommendations_artifact_id": artifact.artifact_id},
            artifact=artifact)


def test_runner_full_loop_explore_ask_investigate_recommend_ask(store):
    """Drive a session through the full Phase 2 continuation loop."""
    _install_stub_explore()
    _install_stub_investigate()
    _install_stub_recommend()
    runner = SessionRunner(store)
    session = runner.start_session(objective="improve retrieval")

    # 1. EXPLORE runs.
    runner.run_next(session_id=session.session_id, deps=ExecutorDeps())
    # 2. ASK_USER (choose_path) becomes IN_PROGRESS.
    runner.run_next(session_id=session.session_id, deps=ExecutorDeps())
    ask = next(t for t in store.list_tasks(session.session_id)
               if t.kind is TaskKind.ASK_USER
               and t.inputs.get("expected_kind") == "choose_path")
    assert ask.status is TaskNodeStatus.IN_PROGRESS

    # 3. User picks option 'c1' -> router spawns INVESTIGATE.
    decision = build_decision(
        session_id=session.session_id, task=ask,
        chosen={"anchor_chunk_id": "c1", "title": "A",
                "rationale": "a"})
    runner.post_decision(session_id=session.session_id, decision=decision)
    tasks = store.list_tasks(session.session_id)
    investigates = [t for t in tasks if t.kind is TaskKind.INVESTIGATE]
    assert len(investigates) == 1
    assert investigates[0].inputs["anchor_chunk_id"] == "c1"
    assert investigates[0].status is TaskNodeStatus.READY

    # 4. INVESTIGATE runs -> router spawns RECOMMEND.
    runner.run_next(session_id=session.session_id, deps=ExecutorDeps())
    tasks = store.list_tasks(session.session_id)
    recs = [t for t in tasks if t.kind is TaskKind.RECOMMEND]
    assert len(recs) == 1
    assert recs[0].status is TaskNodeStatus.READY
    assert recs[0].inputs["findings_artifact_id"] is not None

    # 5. RECOMMEND runs -> router spawns ASK_USER(CHOOSE_RECOMMENDATION).
    runner.run_next(session_id=session.session_id, deps=ExecutorDeps())
    tasks = store.list_tasks(session.session_id)
    pick_recs = [t for t in tasks if t.kind is TaskKind.ASK_USER
                 and t.inputs.get("expected_kind") == "choose_recommendation"]
    assert len(pick_recs) == 1
    assert pick_recs[0].inputs["recommendations_artifact_id"] is not None

    # 6. The CHOOSE_RECOMMENDATION ASK_USER reaches IN_PROGRESS then halts.
    runner.run_next(session_id=session.session_id, deps=ExecutorDeps())
    assert runner.run_next(
        session_id=session.session_id, deps=ExecutorDeps()) is None
    pick_after = store.get_task(pick_recs[0].task_id)
    assert pick_after.status is TaskNodeStatus.IN_PROGRESS

    # Artifacts persisted across the loop: DIRECTIONS, FINDINGS, RECS.
    arts = {a.kind for a in store.list_artifacts(session.session_id)}
    assert ArtifactKind.DIRECTIONS_LIST in arts
    assert ArtifactKind.FINDINGS_BUNDLE in arts
    assert ArtifactKind.RECOMMENDATION_LIST in arts


def test_runner_recommend_executor_needs_findings_artifact(store):
    """RECOMMEND fails cleanly when findings_artifact_id is missing."""
    from cgx.session.tasks.recommend import run_recommend
    session = Session.new("g")
    store.save_session(session)
    rec = TaskNode.new(session.session_id, TaskKind.RECOMMEND, "rec",
                       inputs={})
    store.save_task(rec)
    result = run_recommend(rec, ExecutorDeps(store=store))
    assert result.failure
    assert "findings_artifact_id" in result.failure


def test_runner_investigate_executor_needs_anchor(store):
    """INVESTIGATE fails cleanly when anchor_chunk_id is missing."""
    from cgx.session.tasks.investigate import run_investigate
    session = Session.new("g")
    store.save_session(session)
    inv = TaskNode.new(session.session_id, TaskKind.INVESTIGATE, "inv",
                       inputs={})
    store.save_task(inv)
    result = run_investigate(inv, ExecutorDeps())
    assert result.failure
    assert "anchor_chunk_id" in result.failure



# =====================================================================
# Phase 3 -- write loop: PLAN_CHANGE -> ASK(APPROVE) -> APPLY -> VERIFY
# =====================================================================

# --------------------- router (no IO) ---------------------

def test_router_choose_recommendation_plan_change_spawns_plan_change():
    session = Session.new("g")
    ask = TaskNode.new(
        session.session_id, TaskKind.ASK_USER, "pick rec",
        inputs={"expected_kind": "choose_recommendation",
                "recommendations_artifact_id": "art_recs",
                "findings_artifact_id": "art_findings",
                "prior_goal": "improve retrieval"})
    decision = Decision.new(
        session.session_id, ask.task_id,
        DecisionKind.CHOOSE_RECOMMENDATION, "pick rec",
        {"id": "r2", "title": "Add caching layer",
         "rationale": "speed up", "kind": "plan_change"})
    plan = Router().on_decision_recorded(
        session=session, decision=decision, tasks=[ask])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    pc = creates[0].task
    assert pc.kind is TaskKind.PLAN_CHANGE
    assert pc.parent_task_id == ask.task_id
    assert pc.name == "Add caching layer"
    assert pc.inputs["recommendation"]["kind"] == "plan_change"
    assert pc.inputs["prior_goal"] == "improve retrieval"
    assert pc.inputs["findings_artifact_id"] == "art_findings"
    assert pc.inputs["recommendations_artifact_id"] == "art_recs"
    assert pc.inputs["decision_id"] == decision.decision_id


def test_router_choose_recommendation_investigate_more_spawns_investigate():
    session = Session.new("g")
    ask = TaskNode.new(
        session.session_id, TaskKind.ASK_USER, "pick rec",
        inputs={"expected_kind": "choose_recommendation",
                "prior_goal": "g"})
    decision = Decision.new(
        session.session_id, ask.task_id,
        DecisionKind.CHOOSE_RECOMMENDATION, "pick rec",
        {"id": "r1", "title": "Dig deeper",
         "rationale": "more context", "kind": "investigate_more",
         "anchor_chunk_id": "c9"})
    plan = Router().on_decision_recorded(
        session=session, decision=decision, tasks=[ask])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    inv = creates[0].task
    assert inv.kind is TaskKind.INVESTIGATE
    assert inv.inputs["anchor_chunk_id"] == "c9"
    assert inv.inputs["title"] == "Dig deeper"
    assert inv.inputs["prior_goal"] == "g"


def test_router_choose_recommendation_ask_followup_spawns_freeform_ask():
    session = Session.new("g")
    ask = TaskNode.new(
        session.session_id, TaskKind.ASK_USER, "pick rec",
        inputs={"expected_kind": "choose_recommendation",
                "prior_goal": "g"})
    decision = Decision.new(
        session.session_id, ask.task_id,
        DecisionKind.CHOOSE_RECOMMENDATION, "pick rec",
        {"id": "r3", "title": "Need more from user",
         "rationale": "ambiguity", "kind": "ask_followup"})
    plan = Router().on_decision_recorded(
        session=session, decision=decision, tasks=[ask])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    followup = creates[0].task
    assert followup.kind is TaskKind.ASK_USER
    assert followup.inputs["expected_kind"] == "freeform"
    assert followup.inputs["from_recommendation"]["kind"] == "ask_followup"


def test_router_choose_recommendation_done_has_no_successor():
    session = Session.new("g")
    ask = TaskNode.new(
        session.session_id, TaskKind.ASK_USER, "pick rec",
        inputs={"expected_kind": "choose_recommendation"})
    decision = Decision.new(
        session.session_id, ask.task_id,
        DecisionKind.CHOOSE_RECOMMENDATION, "pick rec",
        {"id": "r4", "title": "Stop here", "rationale": "", "kind": "done"})
    plan = Router().on_decision_recorded(
        session=session, decision=decision, tasks=[ask])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert creates == []


def test_router_plan_change_completion_spawns_approve_ask():
    session = Session.new("g")
    pc = TaskNode.new(session.session_id, TaskKind.PLAN_CHANGE, "plan",
                      inputs={"prior_goal": "g",
                              "recommendation": {"kind": "plan_change"}})
    pc.produced_artifact_id = "art_plan"
    pc.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=pc, tasks=[pc])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    ask = creates[0].task
    assert ask.kind is TaskKind.ASK_USER
    assert ask.parent_task_id == pc.task_id
    assert ask.inputs["expected_kind"] == "approve"
    assert ask.inputs["plan_artifact_id"] == "art_plan"
    assert ask.inputs["prior_goal"] == "g"


def test_router_approve_true_spawns_apply():
    session = Session.new("g")
    ask = TaskNode.new(
        session.session_id, TaskKind.ASK_USER, "approve",
        inputs={"expected_kind": "approve",
                "plan_artifact_id": "art_plan",
                "prior_goal": "g"})
    decision = Decision.new(
        session.session_id, ask.task_id, DecisionKind.APPROVE,
        "approve", {"approved": True})
    plan = Router().on_decision_recorded(
        session=session, decision=decision, tasks=[ask])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    apply_task = creates[0].task
    assert apply_task.kind is TaskKind.APPLY
    assert apply_task.parent_task_id == ask.task_id
    assert apply_task.inputs["plan_artifact_id"] == "art_plan"
    assert apply_task.inputs["prior_goal"] == "g"
    assert apply_task.inputs["decision_id"] == decision.decision_id


def test_router_approve_false_has_no_successor():
    session = Session.new("g")
    ask = TaskNode.new(
        session.session_id, TaskKind.ASK_USER, "approve",
        inputs={"expected_kind": "approve",
                "plan_artifact_id": "art_plan"})
    decision = Decision.new(
        session.session_id, ask.task_id, DecisionKind.APPROVE,
        "approve", {"approved": False})
    plan = Router().on_decision_recorded(
        session=session, decision=decision, tasks=[ask])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert creates == []


def test_router_apply_completion_spawns_verify():
    session = Session.new("g")
    ap = TaskNode.new(session.session_id, TaskKind.APPLY, "apply",
                      inputs={"plan_artifact_id": "art_plan",
                              "prior_goal": "g"})
    ap.produced_artifact_id = "art_applied"
    ap.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=ap, tasks=[ap])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    ver = creates[0].task
    assert ver.kind is TaskKind.VERIFY
    assert ver.parent_task_id == ap.task_id
    assert ver.inputs["apply_artifact_id"] == "art_applied"
    assert ver.inputs["plan_artifact_id"] == "art_plan"


def test_router_verify_completion_has_no_successor():
    session = Session.new("g")
    ver = TaskNode.new(session.session_id, TaskKind.VERIFY, "verify",
                       inputs={})
    ver.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=ver, tasks=[ver])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert creates == []


def test_router_greenfield_apply_completion_spawns_bootstrap_env():
    """In greenfield mode, APPLY -> BOOTSTRAP_ENV (not VERIFY directly).

    The router branches on ``parent.inputs['mode']`` so explore-mode
    APPLY still goes straight to VERIFY (covered by the test above).
    """
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    ap = TaskNode.new(session.session_id, TaskKind.APPLY, "apply",
                      inputs={"plan_artifact_id": "art_plan",
                              "scaffold_artifact_id": "art_scaffold",
                              "prior_goal": "g",
                              "mode": SessionMode.GREENFIELD.value})
    ap.produced_artifact_id = "art_applied"
    ap.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=ap, tasks=[ap])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    boot = creates[0].task
    assert boot.kind is TaskKind.BOOTSTRAP_ENV
    assert boot.parent_task_id == ap.task_id
    assert boot.inputs["apply_artifact_id"] == "art_applied"
    assert boot.inputs["scaffold_artifact_id"] == "art_scaffold"
    assert boot.inputs["mode"] == "greenfield"


def test_router_bootstrap_env_completion_spawns_api_check():
    """BOOTSTRAP_ENV finishes -> API_CHECK runs with the build_artifact_id."""
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    boot = TaskNode.new(
        session.session_id, TaskKind.BOOTSTRAP_ENV, "bootstrap",
        inputs={"apply_artifact_id": "art_applied",
                "scaffold_artifact_id": "art_scaffold",
                "prior_goal": "g",
                "mode": SessionMode.GREENFIELD.value})
    boot.produced_artifact_id = "art_build"
    boot.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=boot, tasks=[boot])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    nxt = creates[0].task
    assert nxt.kind is TaskKind.API_CHECK
    assert nxt.parent_task_id == boot.task_id
    assert nxt.inputs["build_artifact_id"] == "art_build"
    assert nxt.inputs["apply_artifact_id"] == "art_applied"
    assert nxt.inputs["mode"] == "greenfield"


def test_router_api_check_passed_spawns_smoke():
    """API_CHECK with outcome=passed -> SMOKE runs."""
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    api = TaskNode.new(
        session.session_id, TaskKind.API_CHECK, "api",
        inputs={"build_artifact_id": "art_build",
                "apply_artifact_id": "art_applied",
                "mode": SessionMode.GREENFIELD.value})
    api.produced_artifact_id = "art_api"
    api.outputs = {"outcome": "passed", "failed_count": 0,
                   "checked_count": 3, "failure_signature": ""}
    api.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=api, tasks=[api])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    sm = creates[0].task
    assert sm.kind is TaskKind.SMOKE
    assert sm.parent_task_id == api.task_id
    assert sm.inputs["build_artifact_id"] == "art_build"
    assert sm.inputs["api_check_artifact_id"] == "art_api"
    assert sm.inputs["apply_artifact_id"] == "art_applied"
    assert sm.inputs["mode"] == "greenfield"


def test_router_api_check_skipped_spawns_smoke():
    """API_CHECK with outcome=skipped -> SMOKE still runs."""
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    api = TaskNode.new(
        session.session_id, TaskKind.API_CHECK, "api",
        inputs={"build_artifact_id": "art_build",
                "mode": SessionMode.GREENFIELD.value})
    api.produced_artifact_id = "art_api"
    api.outputs = {"outcome": "skipped", "failed_count": 0,
                   "checked_count": 0, "failure_signature": ""}
    api.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=api, tasks=[api])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    assert creates[0].task.kind is TaskKind.SMOKE


def test_router_api_check_failed_spawns_repair():
    """API_CHECK with outcome=failed -> REPAIR runs with api_check_artifact_id."""
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    api = TaskNode.new(
        session.session_id, TaskKind.API_CHECK, "api",
        inputs={"build_artifact_id": "art_build",
                "apply_artifact_id": "art_applied",
                "mode": SessionMode.GREENFIELD.value,
                "repair_attempt": 0,
                "prior_failure_signatures": []})
    api.produced_artifact_id = "art_api"
    api.outputs = {"outcome": "failed", "failed_count": 1,
                   "checked_count": 3,
                   "failure_signature": "api_check|werkzeug.urls.url_quote"}
    api.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=api, tasks=[api])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    rep = creates[0].task
    assert rep.kind is TaskKind.REPAIR
    assert rep.inputs["api_check_artifact_id"] == "art_api"
    assert rep.inputs["repair_attempt"] == 1
    assert "api_check|werkzeug.urls.url_quote" in rep.inputs[
        "prior_failure_signatures"]


def test_router_api_check_failed_skips_repair_when_flapping():
    """API_CHECK whose signature already appears in priors -> no spawn."""
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    sig = "api_check|werkzeug.urls.url_quote"
    api = TaskNode.new(
        session.session_id, TaskKind.API_CHECK, "api",
        inputs={"build_artifact_id": "art_build",
                "mode": SessionMode.GREENFIELD.value,
                "repair_attempt": 1,
                "prior_failure_signatures": [sig]})
    api.produced_artifact_id = "art_api"
    api.outputs = {"outcome": "failed", "failed_count": 1,
                   "checked_count": 3, "failure_signature": sig}
    api.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=api, tasks=[api])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert creates == []
    # C: a failed gate that declines to spawn REPAIR must resolve the
    # session terminally rather than leave the drain loop idle.
    status = [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]
    assert len(status) == 1
    assert status[0].status is SessionStatus.FAILED


def test_router_api_check_failed_budget_exhausted_terminates_session():
    """C: API_CHECK failed with the repair budget spent -> terminal FAILED."""
    from cgx.session.models import SessionMode
    from cgx.session.router import _REPAIR_BUDGET
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    api = TaskNode.new(
        session.session_id, TaskKind.API_CHECK, "api",
        inputs={"build_artifact_id": "art_build",
                "mode": SessionMode.GREENFIELD.value,
                "repair_attempt": _REPAIR_BUDGET,
                "prior_failure_signatures": []})
    api.produced_artifact_id = "art_api"
    api.outputs = {"outcome": "failed", "failed_count": 1,
                   "failure_signature": "api_check|werkzeug.urls.url_quote"}
    api.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=api, tasks=[api])
    assert [a for a in plan.actions if isinstance(a, CreateTask)] == []
    status = [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]
    assert len(status) == 1
    assert status[0].status is SessionStatus.FAILED


def test_router_api_check_passed_does_not_terminate_session():
    """C guard is scoped to failures: a passed gate spawns SMOKE, no status."""
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    api = TaskNode.new(
        session.session_id, TaskKind.API_CHECK, "api",
        inputs={"build_artifact_id": "art_build",
                "mode": SessionMode.GREENFIELD.value})
    api.produced_artifact_id = "art_api"
    api.outputs = {"outcome": "passed", "failed_count": 0}
    api.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=api, tasks=[api])
    assert not [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    assert creates[0].task.kind is TaskKind.SMOKE


def test_router_repair_install_deps_spawns_bootstrap_env():
    """REPAIR strategy=install_deps -> re-queue BOOTSTRAP_ENV (not regenerate)."""
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    sig = "api_check|flask.Flask"
    rep = TaskNode.new(
        session.session_id, TaskKind.REPAIR, "repair",
        inputs={"api_check_artifact_id": "art_api",
                "apply_artifact_id": "art_applied",
                "plan_artifact_id": "art_plan",
                "mode": SessionMode.GREENFIELD.value,
                "repair_attempt": 1,
                "prior_failure_signatures": [sig]})
    rep.produced_artifact_id = "art_plan_repair"
    rep.outputs = {"classification": "missing_dependency",
                   "strategy": "install_deps", "can_apply": False,
                   "diff_count": 0, "repair_attempt": 1,
                   "missing_modules": ["flask"],
                   "failure_signature": sig}
    rep.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=rep, tasks=[rep])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    boot = creates[0].task
    assert boot.kind is TaskKind.BOOTSTRAP_ENV
    assert boot.inputs["apply_artifact_id"] == "art_applied"
    assert boot.inputs["missing_modules"] == ["flask"]
    assert boot.inputs["repair_attempt"] == 1
    assert boot.inputs["prior_failure_signatures"] == [sig]


def test_router_smoke_passed_spawns_verify():
    """SMOKE finishes with outcome=passed -> VERIFY runs."""
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    sm = TaskNode.new(
        session.session_id, TaskKind.SMOKE, "smoke",
        inputs={"build_artifact_id": "art_build",
                "apply_artifact_id": "art_applied",
                "mode": SessionMode.GREENFIELD.value})
    sm.produced_artifact_id = "art_smoke"
    sm.outputs = {"outcome": "passed", "failed_count": 0,
                  "tested_count": 2, "failure_signature": ""}
    sm.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=sm, tasks=[sm])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    ver = creates[0].task
    assert ver.kind is TaskKind.VERIFY
    assert ver.parent_task_id == sm.task_id
    assert ver.inputs["build_artifact_id"] == "art_build"
    assert ver.inputs["smoke_artifact_id"] == "art_smoke"
    assert ver.inputs["mode"] == "greenfield"


def test_router_smoke_skipped_spawns_verify():
    """SMOKE finishes with outcome=skipped -> VERIFY still runs."""
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    sm = TaskNode.new(
        session.session_id, TaskKind.SMOKE, "smoke",
        inputs={"build_artifact_id": "art_build",
                "mode": SessionMode.GREENFIELD.value})
    sm.produced_artifact_id = "art_smoke"
    sm.outputs = {"outcome": "skipped", "failed_count": 0,
                  "tested_count": 0, "failure_signature": ""}
    sm.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=sm, tasks=[sm])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    assert creates[0].task.kind is TaskKind.VERIFY


def test_router_smoke_failed_spawns_repair():
    """SMOKE finishes with outcome=failed -> REPAIR with the smoke artifact."""
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    sm = TaskNode.new(
        session.session_id, TaskKind.SMOKE, "smoke",
        inputs={"build_artifact_id": "art_build",
                "apply_artifact_id": "art_applied",
                "mode": SessionMode.GREENFIELD.value})
    sm.produced_artifact_id = "art_smoke"
    sm.outputs = {"outcome": "failed", "failed_count": 1,
                  "tested_count": 2,
                  "failure_signature": "smoke_import|werkzeug"}
    sm.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=sm, tasks=[sm])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    rep = creates[0].task
    assert rep.kind is TaskKind.REPAIR
    assert rep.parent_task_id == sm.task_id
    assert rep.inputs["smoke_artifact_id"] == "art_smoke"
    assert rep.inputs["build_artifact_id"] == "art_build"
    assert rep.inputs["repair_attempt"] == 1
    assert "smoke_import|werkzeug" in rep.inputs["prior_failure_signatures"]


def test_router_smoke_failed_respects_repair_budget():
    """SMOKE failure does not spawn REPAIR once the budget is exhausted."""
    from cgx.session.models import SessionMode
    from cgx.session.router import _REPAIR_BUDGET
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    sm = TaskNode.new(
        session.session_id, TaskKind.SMOKE, "smoke",
        inputs={"build_artifact_id": "art_build",
                "mode": SessionMode.GREENFIELD.value,
                "repair_attempt": _REPAIR_BUDGET})
    sm.produced_artifact_id = "art_smoke"
    sm.outputs = {"outcome": "failed", "failed_count": 1,
                  "failure_signature": "smoke_import|werkzeug"}
    sm.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=sm, tasks=[sm])
    assert [a for a in plan.actions if isinstance(a, CreateTask)] == []
    status = [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]
    assert len(status) == 1
    assert status[0].status is SessionStatus.FAILED


def test_router_smoke_failed_flap_detector_blocks_repeat():
    """Repeating the same smoke failure signature is treated as a flap."""
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    sm = TaskNode.new(
        session.session_id, TaskKind.SMOKE, "smoke",
        inputs={"build_artifact_id": "art_build",
                "mode": SessionMode.GREENFIELD.value,
                "prior_failure_signatures": ["smoke_import|werkzeug"]})
    sm.produced_artifact_id = "art_smoke"
    sm.outputs = {"outcome": "failed", "failed_count": 1,
                  "failure_signature": "smoke_import|werkzeug"}
    sm.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=sm, tasks=[sm])
    assert [a for a in plan.actions if isinstance(a, CreateTask)] == []
    status = [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]
    assert len(status) == 1
    assert status[0].status is SessionStatus.FAILED


# --------------------- repair-loop router transitions ---------------------

def _greenfield_failed_verify(*, signature: str, outcome: str = "assertions_failed",
                              repair_attempt: int = 0, prior: list | None = None,
                              session=None) -> TaskNode:
    """Build a DONE VERIFY task that the repair successor should react to."""
    from cgx.session.models import SessionMode
    sess = session or Session.new("g", mode=SessionMode.GREENFIELD)
    ver = TaskNode.new(
        sess.session_id, TaskKind.VERIFY, "verify",
        inputs={"apply_artifact_id": "art_applied",
                "build_artifact_id": "art_build",
                "mode": SessionMode.GREENFIELD.value,
                "repair_attempt": repair_attempt,
                "prior_failure_signatures": list(prior or [])})
    ver.produced_artifact_id = "art_verify"
    ver.outputs = {"outcome": outcome, "failure_signature": signature,
                   "returncode": 1}
    ver.status = TaskNodeStatus.DONE
    return ver


def test_router_greenfield_verify_failure_spawns_repair():
    """A fixable VERIFY failure in greenfield -> REPAIR with carried inputs."""
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    ver = _greenfield_failed_verify(signature="abc123", session=session)
    plan = Router().on_task_completed(
        session=session, completed=ver, tasks=[ver])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    rep = creates[0].task
    assert rep.kind is TaskKind.REPAIR
    assert rep.parent_task_id == ver.task_id
    assert rep.inputs["verify_artifact_id"] == "art_verify"
    assert rep.inputs["build_artifact_id"] == "art_build"
    assert rep.inputs["mode"] == "greenfield"
    assert rep.inputs["repair_attempt"] == 1
    assert "abc123" in rep.inputs["prior_failure_signatures"]


def test_router_greenfield_verify_failed_outcome_spawns_repair():
    """A non-pytest ``failed`` VERIFY (e.g. npm build) -> REPAIR too."""
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    ver = _greenfield_failed_verify(
        signature="npmbuild1", outcome="failed", session=session)
    plan = Router().on_task_completed(
        session=session, completed=ver, tasks=[ver])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    assert creates[0].task.kind is TaskKind.REPAIR


def test_router_explore_verify_failure_does_not_spawn_repair():
    """REPAIR is greenfield-only; explore-mode VERIFY failures are terminal."""
    session = Session.new("g")
    ver = TaskNode.new(
        session.session_id, TaskKind.VERIFY, "verify",
        inputs={"mode": "explore"})
    ver.outputs = {"outcome": "assertions_failed",
                   "failure_signature": "x", "returncode": 1}
    ver.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=ver, tasks=[ver])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert creates == []
    # Explore-mode VERIFY keeps its own lifecycle -- no session status flip.
    assert not any(isinstance(a, UpdateSessionStatus) for a in plan.actions)


def test_router_verify_no_tests_collected_with_selection_spawns_repair():
    """pytest exit 5 with selected test files -> REPAIR (malformed tests)."""
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    ver = TaskNode.new(
        session.session_id, TaskKind.VERIFY, "verify",
        inputs={"apply_artifact_id": "art_applied",
                "build_artifact_id": "art_build",
                "mode": SessionMode.GREENFIELD.value,
                "repair_attempt": 0,
                "prior_failure_signatures": []})
    ver.produced_artifact_id = "art_verify"
    ver.outputs = {"outcome": "no_tests_collected", "failure_signature": "nt1",
                   "returncode": 5, "tests_selected_count": 1}
    ver.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=ver, tasks=[ver])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    assert creates[0].task.kind is TaskKind.REPAIR
    assert creates[0].task.inputs["verify_artifact_id"] == "art_verify"


def test_router_verify_no_tests_collected_empty_selection_is_terminal():
    """pytest exit 5 with no selected tests -> terminal (test-free project)."""
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    ver = TaskNode.new(
        session.session_id, TaskKind.VERIFY, "verify",
        inputs={"mode": SessionMode.GREENFIELD.value,
                "repair_attempt": 0, "prior_failure_signatures": []})
    ver.produced_artifact_id = "art_verify"
    ver.outputs = {"outcome": "no_tests_collected", "failure_signature": "nt0",
                   "returncode": 5, "tests_selected_count": 0}
    ver.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=ver, tasks=[ver])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert creates == []
    # A test-free greenfield project is a definitive failure, not success.
    status = [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]
    assert len(status) == 1
    assert status[0].session_id == session.session_id
    assert status[0].status is SessionStatus.FAILED


def test_router_verify_passed_has_no_successor():
    """A passing VERIFY in greenfield is terminal -- no REPAIR spawn."""
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    ver = TaskNode.new(
        session.session_id, TaskKind.VERIFY, "verify",
        inputs={"mode": SessionMode.GREENFIELD.value})
    ver.outputs = {"outcome": "passed", "failure_signature": "p",
                   "returncode": 0}
    ver.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=ver, tasks=[ver])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert creates == []
    # A passing suite completes the greenfield session.
    status = [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]
    assert len(status) == 1
    assert status[0].status is SessionStatus.COMPLETED


def test_router_on_task_failed_greenfield_marks_session_failed():
    """A hard task failure in greenfield transitions the session FAILED."""
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    task = TaskNode.new(
        session.session_id, TaskKind.BOOTSTRAP_ENV, "bootstrap",
        inputs={"mode": SessionMode.GREENFIELD.value})
    task.status = TaskNodeStatus.FAILED
    plan = Router().on_task_failed(
        session=session, failed=task, tasks=[task])
    status = [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]
    assert len(status) == 1
    assert status[0].status is SessionStatus.FAILED


def test_router_on_task_failed_explore_is_noop():
    """A hard task failure in explore mode keeps the user-driven lifecycle."""
    from cgx.session.models import SessionMode
    session = Session.new("e", mode=SessionMode.EXPLORE)
    task = TaskNode.new(session.session_id, TaskKind.EXPLORE, "explore")
    task.status = TaskNodeStatus.FAILED
    plan = Router().on_task_failed(
        session=session, failed=task, tasks=[task])
    assert list(plan.actions) == []


def test_router_on_task_failed_noop_when_already_terminal():
    """No duplicate transition once the session is already terminal."""
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    session.status = SessionStatus.FAILED
    task = TaskNode.new(
        session.session_id, TaskKind.BOOTSTRAP_ENV, "bootstrap",
        inputs={"mode": SessionMode.GREENFIELD.value})
    task.status = TaskNodeStatus.FAILED
    plan = Router().on_task_failed(
        session=session, failed=task, tasks=[task])
    assert list(plan.actions) == []


def test_router_verify_repeat_signature_refuses_repair():
    """Progress detector: same signature twice -> no REPAIR (loop guard)."""
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    ver = _greenfield_failed_verify(
        signature="dup", repair_attempt=1, prior=["dup"], session=session)
    plan = Router().on_task_completed(
        session=session, completed=ver, tasks=[ver])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert creates == []
    # Flap guard refuses another REPAIR -> the session fails terminally.
    status = [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]
    assert len(status) == 1
    assert status[0].status is SessionStatus.FAILED


def test_router_verify_exceeds_budget_refuses_repair():
    """Retry budget exhausted (>= 2 attempts) -> no REPAIR (loop guard)."""
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    ver = _greenfield_failed_verify(
        signature="new", repair_attempt=2, prior=["old1", "old2"],
        session=session)
    plan = Router().on_task_completed(
        session=session, completed=ver, tasks=[ver])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert creates == []
    # Budget exhausted -> no REPAIR, session fails terminally.
    status = [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]
    assert len(status) == 1
    assert status[0].status is SessionStatus.FAILED


def test_router_repair_with_diffs_spawns_apply():
    """REPAIR producing diffs -> APPLY carrying build_artifact_id + attempt."""
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    rep = TaskNode.new(
        session.session_id, TaskKind.REPAIR, "repair",
        inputs={"verify_artifact_id": "art_verify",
                "build_artifact_id": "art_build",
                "mode": SessionMode.GREENFIELD.value,
                "repair_attempt": 1,
                "prior_failure_signatures": ["abc123"]})
    rep.produced_artifact_id = "art_repair_plan"
    rep.outputs = {"can_apply": True, "classification": "unittest_pytest_mix",
                   "failure_signature": "abc123", "repair_attempt": 1,
                   "diff_count": 2}
    rep.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=rep, tasks=[rep])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    ap = creates[0].task
    assert ap.kind is TaskKind.APPLY
    assert ap.parent_task_id == rep.task_id
    assert ap.inputs["plan_artifact_id"] == "art_repair_plan"
    assert ap.inputs["build_artifact_id"] == "art_build"
    assert ap.inputs["repair_attempt"] == 1


def test_router_repair_without_diffs_fails_session_terminally():
    """REPAIR with empty plan -> terminal FAILED (no ASK_USER, no APPLY).

    Asking the user to hand-fix AI-generated code is never a valid
    recovery: once REPAIR cannot produce an applicable patch and neither
    the install-deps nor regenerate branch applies, the REPAIR node and
    the whole session go terminally ``FAILED``.
    """
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    rep = TaskNode.new(
        session.session_id, TaskKind.REPAIR, "repair",
        inputs={"verify_artifact_id": "art_verify",
                "mode": SessionMode.GREENFIELD.value,
                "repair_attempt": 1})
    rep.produced_artifact_id = "art_repair_plan"
    rep.outputs = {"can_apply": False, "classification": "unknown",
                   "failure_signature": "abc123", "diff_count": 0}
    rep.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=rep, tasks=[rep])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert creates == []
    task_fail = [a for a in plan.actions if isinstance(a, UpdateTaskStatus)
                 and a.status is TaskNodeStatus.FAILED]
    assert len(task_fail) == 1
    assert task_fail[0].task_id == rep.task_id
    assert task_fail[0].error and "unknown" in task_fail[0].error
    status = [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]
    assert len(status) == 1
    assert status[0].session_id == session.session_id
    assert status[0].status is SessionStatus.FAILED


def test_router_apply_from_repair_skips_bootstrap():
    """APPLY carrying build_artifact_id (from REPAIR) -> VERIFY (no BOOTSTRAP)."""
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    ap = TaskNode.new(
        session.session_id, TaskKind.APPLY, "apply",
        inputs={"plan_artifact_id": "art_repair_plan",
                "build_artifact_id": "art_build",
                "mode": SessionMode.GREENFIELD.value,
                "repair_attempt": 1,
                "prior_failure_signatures": ["abc123"]})
    ap.produced_artifact_id = "art_applied_repair"
    ap.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=ap, tasks=[ap])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    ver = creates[0].task
    assert ver.kind is TaskKind.VERIFY
    assert ver.inputs["build_artifact_id"] == "art_build"
    assert ver.inputs["repair_attempt"] == 1
    assert ver.inputs["prior_failure_signatures"] == ["abc123"]


# --------------------- build_decision validation ---------------------

def test_build_decision_rejects_choose_recommendation_with_bad_kind():
    session = Session.new("g")
    ask = TaskNode.new(session.session_id, TaskKind.ASK_USER, "pick",
                       inputs={"expected_kind": "choose_recommendation"})
    with pytest.raises(ValueError):
        build_decision(session_id=session.session_id, task=ask,
                       chosen={"kind": "no_such_kind"})


def test_build_decision_rejects_investigate_more_without_anchor():
    session = Session.new("g")
    ask = TaskNode.new(session.session_id, TaskKind.ASK_USER, "pick",
                       inputs={"expected_kind": "choose_recommendation"})
    with pytest.raises(ValueError):
        build_decision(session_id=session.session_id, task=ask,
                       chosen={"kind": "investigate_more"})


def test_build_decision_accepts_plan_change_recommendation():
    session = Session.new("g")
    ask = TaskNode.new(session.session_id, TaskKind.ASK_USER, "pick",
                       inputs={"expected_kind": "choose_recommendation"})
    d = build_decision(session_id=session.session_id, task=ask,
                       chosen={"kind": "plan_change", "title": "X"})
    assert d.kind is DecisionKind.CHOOSE_RECOMMENDATION
    assert d.chosen["kind"] == "plan_change"


def test_build_decision_rejects_approve_without_approved_key():
    session = Session.new("g")
    ask = TaskNode.new(session.session_id, TaskKind.ASK_USER, "ok?",
                       inputs={"expected_kind": "approve"})
    with pytest.raises(ValueError):
        build_decision(session_id=session.session_id, task=ask, chosen={})


def test_build_decision_accepts_approve_false():
    session = Session.new("g")
    ask = TaskNode.new(session.session_id, TaskKind.ASK_USER, "ok?",
                       inputs={"expected_kind": "approve"})
    d = build_decision(session_id=session.session_id, task=ask,
                       chosen={"approved": False})
    assert d.kind is DecisionKind.APPROVE
    assert d.chosen["approved"] is False


# --------------------- executor unit tests ---------------------

def test_apply_executor_needs_plan_artifact_id(store):
    from cgx.session.tasks.apply import run_apply
    session = Session.new("g", project_root="/tmp")
    store.save_session(session)
    ap = TaskNode.new(session.session_id, TaskKind.APPLY, "apply",
                      inputs={})
    store.save_task(ap)
    result = run_apply(ap, ExecutorDeps(project_root="/tmp", store=store))
    assert result.failure
    assert "plan_artifact_id" in result.failure


def test_apply_executor_rejects_missing_plan_artifact(store, tmp_path):
    from cgx.session.tasks.apply import run_apply
    session = Session.new("g", project_root=str(tmp_path))
    store.save_session(session)
    ap = TaskNode.new(session.session_id, TaskKind.APPLY, "apply",
                      inputs={"plan_artifact_id": "art_missing"})
    store.save_task(ap)
    result = run_apply(
        ap, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.failure
    assert "missing or wrong kind" in result.failure


def test_apply_executor_accepts_repair_plan(store, tmp_path):
    """A REPAIR_PLAN's diffs are applyable by the shared APPLY executor.

    Regression: the auto-repair loop wires REPAIR -> APPLY with the
    REPAIR_PLAN artifact, but APPLY's kind guard originally accepted only
    CODE_CHANGE_PLAN / SCAFFOLD_PATCHES, so a valid conftest.py pythonpath
    fix was rejected with "missing or wrong kind" and the run stalled.
    """
    from cgx.session.repair.locate import MissingPythonpathLocation
    from cgx.session.repair.propose import propose_missing_module_pythonpath
    from cgx.session.tasks.apply import run_apply
    diffs = propose_missing_module_pythonpath(
        tmp_path,
        [MissingPythonpathLocation(
            module_name="app", top_level="app", resolved_path="app")])
    assert diffs and diffs[0]["file"] == "conftest.py"

    session = Session.new("g", project_root=str(tmp_path))
    store.save_session(session)
    plan = Artifact.new(
        session_id=session.session_id, produced_by_task_id="t_repair",
        kind=ArtifactKind.REPAIR_PLAN,
        content={"classification": "missing_module_pythonpath",
                 "strategy": "patch", "diffs": diffs})
    store.save_artifact(plan)
    ap = TaskNode.new(session.session_id, TaskKind.APPLY, "apply",
                      inputs={"plan_artifact_id": plan.artifact_id})
    store.save_task(ap)
    result = run_apply(
        ap, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.failure is None
    assert result.artifact.content["applied_files"] == ["conftest.py"]
    assert not result.artifact.content["failed_files"]
    # failed_files is mirrored into outputs so the router's G1 regenerate
    # constraint can enumerate dropped files without loading the artifact.
    assert result.outputs["failed_files"] == []
    assert (tmp_path / "conftest.py").exists()


def test_verify_executor_requires_project_root(store):
    from cgx.session.tasks.verify import run_verify
    session = Session.new("g")
    store.save_session(session)
    ver = TaskNode.new(session.session_id, TaskKind.VERIFY, "verify",
                       inputs={})
    store.save_task(ver)
    result = run_verify(ver, ExecutorDeps(store=store))
    assert result.failure
    assert "project_root" in result.failure


def test_verify_executor_records_reproduce_cmd(
        tmp_path, store, monkeypatch):
    """run_verify populates content.reproduce_cmd with a paste-ready shell line."""
    from cgx.codegen.test_runner import TestRunOutcome
    from cgx.session.tasks.verify import run_verify

    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text("def test_x(): assert 1\n",
                                                  encoding="utf-8")
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.write_text("")
    venv_python.chmod(0o755)

    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    monkeypatch.setattr(
        "cgx.codegen.test_runner.run_tests_on_disk",
        lambda root, files, **_kw: TestRunOutcome(
            ran=True, returncode=1,
            stdout="1 failed", stderr="",
            tests_selected=[str(tmp_path / "tests" / "test_x.py")],
        ))

    t = TaskNode.new(session.session_id, TaskKind.VERIFY, "verify",
                     inputs={"applied_files": ["pkg/mod.py"],
                             "mode": SessionMode.EXPLORE.value})
    store.save_task(t)
    result = run_verify(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.failure is None
    content = result.artifact.content
    cmd = content["reproduce_cmd"]
    assert isinstance(cmd, str) and cmd
    # cd into project_root then invoke the venv python directly.
    assert f"cd {tmp_path.resolve()}" in cmd
    assert str(venv_python) in cmd
    assert "-m pytest -q --no-header" in cmd
    # The selected test is rendered relative to project_root.
    assert "tests/test_x.py" in cmd
    # Skipped runs (no tests selected) -> reproduce_cmd is None.
    monkeypatch.setattr(
        "cgx.codegen.test_runner.run_tests_on_disk",
        lambda root, files, **_kw: TestRunOutcome(
            ran=False, skipped_reason="no tests located"))
    t2 = TaskNode.new(session.session_id, TaskKind.VERIFY, "verify",
                      inputs={"applied_files": [],
                              "mode": SessionMode.EXPLORE.value})
    store.save_task(t2)
    result2 = run_verify(
        t2, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result2.artifact.content["reproduce_cmd"] is None


def test_verify_parses_junitxml_failures(tmp_path, store, monkeypatch):
    """run_verify populates content.failures from --junitxml output."""
    from cgx.codegen.test_runner import TestRunOutcome
    from cgx.session.tasks.verify import run_verify

    (tmp_path / "tests").mkdir()
    test_file = tmp_path / "tests" / "test_x.py"
    test_file.write_text("def test_x(): assert 0\n", encoding="utf-8")

    session = Session.new("g", mode=SessionMode.EXPLORE)
    store.save_session(session)

    # Stub run_tests_on_disk: when invoked, write a minimal junit XML to
    # whatever path was passed via --junitxml so the parser sees a real
    # file rather than an empty placeholder.
    def _fake_run(root, files, **kw):
        extra = kw.get("extra_pytest_args") or ()
        junit_path = None
        for arg in extra:
            if isinstance(arg, str) and arg.startswith("--junitxml="):
                junit_path = arg.split("=", 1)[1]
                break
        if junit_path:
            Path(junit_path).write_text(
                '<?xml version="1.0" encoding="utf-8"?>'
                '<testsuites><testsuite name="pytest" tests="2" failures="1" errors="1">'
                '<testcase classname="tests.test_x" name="test_x">'
                '<failure type="AssertionError" message="assert 0">'
                'Traceback (most recent call last):\n  assert 0\n</failure>'
                '</testcase>'
                '<testcase classname="tests.test_y" name="test_y">'
                '<error type="ImportError" message="cannot import name X">'
                'ImportError: cannot import name X\n</error>'
                '</testcase>'
                '</testsuite></testsuites>',
                encoding="utf-8",
            )
        return TestRunOutcome(
            ran=True, returncode=1,
            stdout="1 failed, 1 error", stderr="",
            tests_selected=[str(test_file)],
        )

    monkeypatch.setattr("cgx.codegen.test_runner.run_tests_on_disk", _fake_run)
    t = TaskNode.new(session.session_id, TaskKind.VERIFY, "verify",
                     inputs={"applied_files": ["pkg/mod.py"],
                             "mode": SessionMode.EXPLORE.value})
    store.save_task(t)
    result = run_verify(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.failure is None
    failures = result.artifact.content["failures"]
    assert isinstance(failures, list) and len(failures) == 2
    assert failures[0] == {
        "nodeid": "tests.test_x::test_x",
        "kind": "failure",
        "type": "AssertionError",
        "message": "assert 0",
        "traceback": (
            "Traceback (most recent call last):\n  assert 0"
        ),
    }
    assert failures[1]["nodeid"] == "tests.test_y::test_y"
    assert failures[1]["kind"] == "error"
    assert failures[1]["type"] == "ImportError"
    assert "cannot import name X" in failures[1]["message"]


def test_verify_failures_empty_when_junitxml_missing(
        tmp_path, store, monkeypatch):
    """A run that never writes the junit file degrades to failures=[]."""
    from cgx.codegen.test_runner import TestRunOutcome
    from cgx.session.tasks.verify import run_verify

    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text(
        "def test_x(): assert 1\n", encoding="utf-8")

    session = Session.new("g", mode=SessionMode.EXPLORE)
    store.save_session(session)

    monkeypatch.setattr(
        "cgx.codegen.test_runner.run_tests_on_disk",
        lambda root, files, **_kw: TestRunOutcome(
            ran=True, returncode=0,
            stdout="1 passed", stderr="",
            tests_selected=[str(tmp_path / "tests" / "test_x.py")],
        ))
    t = TaskNode.new(session.session_id, TaskKind.VERIFY, "verify",
                     inputs={"mode": SessionMode.EXPLORE.value})
    store.save_task(t)
    result = run_verify(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.failure is None
    assert result.artifact.content["failures"] == []


def test_verify_npm_only_build_failure_is_failed(
        tmp_path, store, monkeypatch):
    """A package.json-only project runs NpmRunner; a build break -> failed."""
    from cgx.codegen.test_runner import TestRunOutcome
    from cgx.codegen import test_runners
    from cgx.session.tasks.verify import run_verify

    # No python markers and no ``test_*.py`` -> pytest does not detect;
    # only NpmRunner applies, so a non-zero build is a real failure signal
    # rather than a silent "no tests -> skipped -> success".
    (tmp_path / "package.json").write_text(
        '{"name": "app", "scripts": {"build": "vite build"}}',
        encoding="utf-8")

    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)

    def _fake_run(self, root, files, **kw):
        return TestRunOutcome(
            ran=True, returncode=1,
            stdout="build failed", stderr="TS2345",
            tests_selected=["npm run build"])

    monkeypatch.setattr(test_runners.NpmRunner, "run", _fake_run)

    t = TaskNode.new(session.session_id, TaskKind.VERIFY, "verify",
                     inputs={"changed_files": ["src/App.jsx"],
                             "mode": SessionMode.GREENFIELD.value})
    store.save_task(t)
    result = run_verify(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.failure is None
    assert result.outputs["outcome"] == "failed"
    assert result.outputs["tests_passed"] is False
    assert result.outputs["ran"] is True
    content = result.artifact.content
    assert content["outcome"] == "failed"
    assert "build failed" in content["stdout"]
    # No pytest selection -> no paste-ready pytest reproduce line.
    assert content["reproduce_cmd"] is None


def test_verify_npm_only_build_pass_is_passed(
        tmp_path, store, monkeypatch):
    """A package.json-only project whose build/test succeeds -> passed."""
    from cgx.codegen.test_runner import TestRunOutcome
    from cgx.codegen import test_runners
    from cgx.session.tasks.verify import run_verify

    (tmp_path / "package.json").write_text(
        '{"name": "app", "scripts": {"build": "vite build"}}',
        encoding="utf-8")

    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)

    monkeypatch.setattr(
        test_runners.NpmRunner, "run",
        lambda self, root, files, **kw: TestRunOutcome(
            ran=True, returncode=0, stdout="built ok", stderr="",
            tests_selected=["npm run build"]))

    t = TaskNode.new(session.session_id, TaskKind.VERIFY, "verify",
                     inputs={"changed_files": ["src/App.jsx"],
                             "mode": SessionMode.GREENFIELD.value})
    store.save_task(t)
    result = run_verify(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.failure is None
    assert result.outputs["outcome"] == "passed"
    assert result.outputs["tests_passed"] is True


def test_verify_polyglot_npm_failure_surfaces_over_pytest_pass(
        tmp_path, store, monkeypatch):
    """In a polyglot repo, a failing npm build is not masked by green pytest."""
    from cgx.codegen.test_runner import TestRunOutcome
    from cgx.codegen import test_runners
    from cgx.session.tasks.verify import run_verify

    (tmp_path / "package.json").write_text(
        '{"name": "app", "scripts": {"build": "vite build"}}',
        encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text(
        "def test_x(): assert 1\n", encoding="utf-8")

    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)

    monkeypatch.setattr(
        "cgx.codegen.test_runner.run_tests_on_disk",
        lambda root, files, **_kw: TestRunOutcome(
            ran=True, returncode=0, stdout="1 passed", stderr="",
            tests_selected=[str(tmp_path / "tests" / "test_x.py")]))
    monkeypatch.setattr(
        test_runners.NpmRunner, "run",
        lambda self, root, files, **kw: TestRunOutcome(
            ran=True, returncode=1, stdout="build failed", stderr="",
            tests_selected=["npm run build"]))

    t = TaskNode.new(session.session_id, TaskKind.VERIFY, "verify",
                     inputs={"changed_files": ["src/App.jsx"],
                             "mode": SessionMode.GREENFIELD.value})
    store.save_task(t)
    result = run_verify(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.failure is None
    assert result.outputs["outcome"] == "failed"
    assert result.outputs["tests_passed"] is False


def test_plan_change_executor_needs_provider():
    from cgx.session.tasks.plan_change import run_plan_change
    session = Session.new("g")
    pc = TaskNode.new(session.session_id, TaskKind.PLAN_CHANGE, "plan",
                      inputs={"recommendation": {"title": "X"}})
    result = run_plan_change(
        pc, ExecutorDeps(index_dir="/tmp/idx", records_path="/tmp/r"))
    assert result.failure
    assert "LLM provider" in result.failure


# --------------------- runner integration (stub executors) ---------------------

def _install_stub_plan_change():
    @register_executor(TaskKind.PLAN_CHANGE)
    def _stub(task, deps):
        artifact = Artifact.new(
            session_id=task.session_id,
            produced_by_task_id=task.task_id,
            kind=ArtifactKind.CODE_CHANGE_PLAN,
            content={
                "plan_md": "## Plan\nstub",
                "diffs": [{"file": "pkg/mod.py", "patch": "stub diff"}],
                "citations": [],
                "confidence": 0.7,
                "prior_goal": task.inputs.get("prior_goal"),
                "recommendation": task.inputs.get("recommendation"),
            },
        )
        return ExecutorResult(
            outputs={"plan_artifact_id": artifact.artifact_id,
                     "diffs_count": 1},
            artifact=artifact)


def _install_stub_apply():
    @register_executor(TaskKind.APPLY)
    def _stub(task, deps):
        artifact = Artifact.new(
            session_id=task.session_id,
            produced_by_task_id=task.task_id,
            kind=ArtifactKind.APPLIED_CHANGES,
            content={
                "plan_artifact_id": task.inputs.get("plan_artifact_id"),
                "applied_files": ["pkg/mod.py"],
                "failed_files": [],
                "backup_dir": "/tmp/backup-fake",
                "smoke_ok": True,
                "diffs": [{"file": "pkg/mod.py", "patch": "stub diff"}],
            },
        )
        return ExecutorResult(
            outputs={"apply_artifact_id": artifact.artifact_id,
                     "applied_count": 1, "failed_count": 0,
                     "backup_dir": "/tmp/backup-fake"},
            artifact=artifact)


def _install_stub_verify():
    @register_executor(TaskKind.VERIFY)
    def _stub(task, deps):
        artifact = Artifact.new(
            session_id=task.session_id,
            produced_by_task_id=task.task_id,
            kind=ArtifactKind.VERIFY_REPORT,
            content={
                "apply_artifact_id": task.inputs.get("apply_artifact_id"),
                "plan_artifact_id": task.inputs.get("plan_artifact_id"),
                "changed_files": ["pkg/mod.py"],
                "ran": True, "tests_passed": True,
                "returncode": 0, "tests_selected": ["tests/test_mod.py"],
                "stdout": "1 passed", "stderr": "",
                "skipped_reason": None,
            },
        )
        return ExecutorResult(
            outputs={"verify_artifact_id": artifact.artifact_id,
                     "ran": True, "tests_passed": True,
                     "tests_selected_count": 1},
            artifact=artifact)


# --------------------- agent_log (project-local JSONL trace) ---------------------

def _read_agent_log(project_root: Path) -> list:
    """Return the parsed JSONL records from <root>/.cgx/agent.log."""
    import json
    path = project_root / ".cgx" / "agent.log"
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def test_agent_log_noop_when_project_root_is_none(tmp_path):
    """log_event with falsy project_root never touches disk."""
    from cgx.session.agent_log import log_event
    log_event(None, "task_started", session_id="ses_x")
    log_event("", "task_started", session_id="ses_x")
    assert not (tmp_path / ".cgx" / "agent.log").exists()


def test_agent_log_writes_jsonl_with_ts_and_event(tmp_path):
    """Each call emits one JSON object per line with ts + event fields."""
    from cgx.session.agent_log import log_event
    log_event(str(tmp_path), "task_started", session_id="ses_1",
              task_id="t_1", kind="apply")
    log_event(str(tmp_path), "task_completed", session_id="ses_1",
              task_id="t_1", kind="apply", duration_ms=42)
    records = _read_agent_log(tmp_path)
    assert len(records) == 2
    assert records[0]["event"] == "task_started"
    assert records[0]["session_id"] == "ses_1"
    assert records[0]["kind"] == "apply"
    assert isinstance(records[0]["ts"], (int, float))
    assert records[1]["event"] == "task_completed"
    assert records[1]["duration_ms"] == 42


def test_agent_log_mirrors_to_session_stable_path(tmp_path, monkeypatch):
    """P3.1: each line is mirrored to a session-stable path under config.

    The project-local log lives under the churn-prone project tree; the
    stable mirror lives at ``<config>/agent-sessions/<sid>/agent.log`` so it
    survives a re-scaffold that trashes the project directory. Both writes
    carry identical records (same ts + fields).
    """
    import json
    from cgx.session.agent_log import log_event, reset_for_tests
    cfg = tmp_path / "cfg"
    monkeypatch.setenv("CGX_CONFIG_DIR", str(cfg))
    reset_for_tests()
    proj = tmp_path / "proj"
    log_event(str(proj), "task_started", session_id="ses_9", kind="apply")

    stable = cfg / "agent-sessions" / "ses_9" / "agent.log"
    assert stable.is_file()
    stable_lines = [ln for ln in stable.read_text(
        encoding="utf-8").splitlines() if ln.strip()]
    assert len(stable_lines) == 1
    rec = json.loads(stable_lines[0])
    assert rec["event"] == "task_started"
    assert rec["session_id"] == "ses_9"
    assert rec["kind"] == "apply"
    # The project-local log carries the same record with the same timestamp.
    proj_records = _read_agent_log(proj)
    assert len(proj_records) == 1
    assert proj_records[0]["ts"] == rec["ts"]


def test_agent_log_no_stable_mirror_without_session_id(tmp_path, monkeypatch):
    """No ``session_id`` -> project-local only, no stray stable dir."""
    from cgx.session.agent_log import log_event, reset_for_tests
    cfg = tmp_path / "cfg"
    monkeypatch.setenv("CGX_CONFIG_DIR", str(cfg))
    reset_for_tests()
    proj = tmp_path / "proj"
    log_event(str(proj), "session_status_changed", status="running")
    assert _read_agent_log(proj)
    assert not (cfg / "agent-sessions").exists()


def test_runner_emits_task_lifecycle_events_to_agent_log(tmp_path, store):
    """A happy stub task produces task_started + task_completed lines."""
    @register_executor(TaskKind.EXPLORE)
    def _stub(task, deps):
        return ExecutorResult(outputs={"ok": True})

    session = Session.new("explore", project_root=str(tmp_path))
    store.save_session(session)
    t = TaskNode.new(session.session_id, TaskKind.EXPLORE, "explore")
    store.save_task(t)
    runner = SessionRunner(store)
    runner.run_next(session_id=session.session_id, deps=ExecutorDeps())

    records = _read_agent_log(tmp_path)
    events = [r["event"] for r in records]
    assert "task_started" in events
    assert "task_completed" in events
    started = next(r for r in records if r["event"] == "task_started")
    assert started["session_id"] == session.session_id
    assert started["task_id"] == t.task_id
    assert started["kind"] == "explore"


def test_runner_emits_task_failed_on_executor_result_failure(tmp_path, store):
    """An ExecutorResult.failure produces task_started + task_failed."""
    @register_executor(TaskKind.EXPLORE)
    def _stub(task, deps):
        return ExecutorResult(failure="planned failure: nope")

    session = Session.new("explore", project_root=str(tmp_path))
    store.save_session(session)
    t = TaskNode.new(session.session_id, TaskKind.EXPLORE, "explore")
    store.save_task(t)
    SessionRunner(store).run_next(
        session_id=session.session_id, deps=ExecutorDeps())

    records = _read_agent_log(tmp_path)
    events = [r["event"] for r in records]
    assert events == ["task_started", "task_failed"]
    assert records[1]["error"] == "planned failure: nope"


def test_runner_emits_executor_crashed_on_exception(tmp_path, store):
    """A raising executor produces task_started + executor_crashed + task_failed."""
    @register_executor(TaskKind.EXPLORE)
    def _stub(task, deps):
        raise RuntimeError("boom")

    session = Session.new("explore", project_root=str(tmp_path))
    store.save_session(session)
    t = TaskNode.new(session.session_id, TaskKind.EXPLORE, "explore")
    store.save_task(t)
    SessionRunner(store).run_next(
        session_id=session.session_id, deps=ExecutorDeps())

    records = _read_agent_log(tmp_path)
    events = [r["event"] for r in records]
    assert "task_started" in events
    assert "executor_crashed" in events
    assert "task_failed" in events
    crash = next(r for r in records if r["event"] == "executor_crashed")
    assert crash["exc_type"] == "RuntimeError"
    assert "boom" in crash["error"]


def test_runner_full_write_loop_plan_approve_apply_verify(store):
    """Drive the full Phase 3 write loop with stub executors.

    Pre-seeds an ``ASK_USER(CHOOSE_RECOMMENDATION)`` so the test bypasses
    the read-only loop and exercises just the write-side transitions
    introduced in Phase 3: a ``plan_change`` recommendation choice
    triggers PLAN_CHANGE; an APPROVE(True) triggers APPLY; APPLY
    completion spawns VERIFY; VERIFY is terminal.
    """
    _install_stub_plan_change()
    _install_stub_apply()
    _install_stub_verify()
    # Build the session directly (no start_session) so the auto-spawned
    # EXPLORE doesn't sit READY ahead of the write-loop tasks the test
    # is exercising.
    session = Session.new("add caching", project_root="/tmp/proj")
    store.save_session(session)
    runner = SessionRunner(store)

    # Seed the choose-recommendation ASK directly so we don't replay
    # the entire read-only loop; the upstream artifacts aren't needed
    # because we stub PLAN_CHANGE.
    pick = TaskNode.new(
        session.session_id, TaskKind.ASK_USER, "pick rec",
        inputs={"expected_kind": "choose_recommendation",
                "recommendations_artifact_id": "art_recs",
                "findings_artifact_id": "art_findings",
                "prior_goal": "add caching"})
    store.save_task(pick)

    # 1. User picks a plan_change recommendation -> spawns PLAN_CHANGE.
    decision = build_decision(
        session_id=session.session_id, task=pick,
        chosen={"id": "r1", "title": "Cache results",
                "rationale": "speed", "kind": "plan_change"})
    runner.post_decision(session_id=session.session_id, decision=decision)
    tasks = store.list_tasks(session.session_id)
    plans = [t for t in tasks if t.kind is TaskKind.PLAN_CHANGE]
    assert len(plans) == 1
    assert plans[0].status is TaskNodeStatus.READY

    # 2. PLAN_CHANGE runs -> spawns ASK_USER(APPROVE).
    runner.run_next(session_id=session.session_id, deps=ExecutorDeps())
    tasks = store.list_tasks(session.session_id)
    approves = [t for t in tasks if t.kind is TaskKind.ASK_USER
                and t.inputs.get("expected_kind") == "approve"]
    assert len(approves) == 1
    plan_artifact_id = approves[0].inputs["plan_artifact_id"]
    assert plan_artifact_id

    # 3. The APPROVE ASK_USER becomes IN_PROGRESS.
    runner.run_next(session_id=session.session_id, deps=ExecutorDeps())
    approve_ask = store.get_task(approves[0].task_id)
    assert approve_ask.status is TaskNodeStatus.IN_PROGRESS

    # 4. User approves -> spawns APPLY.
    approve_decision = build_decision(
        session_id=session.session_id, task=approve_ask,
        chosen={"approved": True})
    runner.post_decision(session_id=session.session_id,
                        decision=approve_decision)
    tasks = store.list_tasks(session.session_id)
    applies = [t for t in tasks if t.kind is TaskKind.APPLY]
    assert len(applies) == 1
    assert applies[0].inputs["plan_artifact_id"] == plan_artifact_id

    # 5. APPLY runs -> spawns VERIFY.
    runner.run_next(session_id=session.session_id, deps=ExecutorDeps())
    tasks = store.list_tasks(session.session_id)
    verifies = [t for t in tasks if t.kind is TaskKind.VERIFY]
    assert len(verifies) == 1
    apply_artifact_id = verifies[0].inputs["apply_artifact_id"]
    assert apply_artifact_id

    # 6. VERIFY runs -> terminal, no successor.
    runner.run_next(session_id=session.session_id, deps=ExecutorDeps())
    verify_task = store.get_task(verifies[0].task_id)
    assert verify_task.status is TaskNodeStatus.DONE
    assert runner.run_next(
        session_id=session.session_id, deps=ExecutorDeps()) is None

    # Artifacts persisted: PLAN, APPLIED, VERIFY (plus the seed EXPLORE
    # artifact from start_session, but that depends on whether the
    # initial EXPLORE ran -- assert the write-loop trio is present).
    kinds = {a.kind for a in store.list_artifacts(session.session_id)}
    assert ArtifactKind.CODE_CHANGE_PLAN in kinds
    assert ArtifactKind.APPLIED_CHANGES in kinds
    assert ArtifactKind.VERIFY_REPORT in kinds


def test_runner_decline_approval_halts_loop(store):
    """An ``approved: False`` decision should not spawn APPLY."""
    _install_stub_plan_change()
    _install_stub_apply()
    _install_stub_verify()
    session = Session.new("add caching", project_root="/tmp/proj")
    store.save_session(session)
    runner = SessionRunner(store)
    pick = TaskNode.new(
        session.session_id, TaskKind.ASK_USER, "pick rec",
        inputs={"expected_kind": "choose_recommendation",
                "prior_goal": "add caching"})
    store.save_task(pick)
    runner.post_decision(
        session_id=session.session_id,
        decision=build_decision(
            session_id=session.session_id, task=pick,
            chosen={"id": "r1", "title": "Cache", "rationale": "",
                    "kind": "plan_change"}))
    runner.run_next(session_id=session.session_id, deps=ExecutorDeps())
    approve_ask = next(t for t in store.list_tasks(session.session_id)
                       if t.kind is TaskKind.ASK_USER
                       and t.inputs.get("expected_kind") == "approve")
    runner.run_next(session_id=session.session_id, deps=ExecutorDeps())
    runner.post_decision(
        session_id=session.session_id,
        decision=build_decision(
            session_id=session.session_id, task=approve_ask,
            chosen={"approved": False},
            rationale="not yet"))

    # Decline -> no APPLY/VERIFY spawned, ASK is DONE.
    tasks = store.list_tasks(session.session_id)
    assert not [t for t in tasks if t.kind is TaskKind.APPLY]
    assert not [t for t in tasks if t.kind is TaskKind.VERIFY]
    approve_after = store.get_task(approve_ask.task_id)
    assert approve_after.status is TaskNodeStatus.DONE



# =====================================================================
# Greenfield -- CLARIFY -> DECOMPOSE -> ASK(APPROVE_PLAN) -> SCAFFOLD ->
# APPLY -> VERIFY
# =====================================================================

from cgx.session.models import SessionMode  # noqa: E402

# --------------------- mode detection ---------------------

def test_detect_mode_empty_project_root_is_greenfield(tmp_path: Path):
    from cgx.session.mode import detect_mode
    assert detect_mode(project_root=str(tmp_path)) is SessionMode.GREENFIELD


def test_detect_mode_missing_project_root_is_greenfield():
    from cgx.session.mode import detect_mode
    assert detect_mode(project_root=None) is SessionMode.GREENFIELD


def test_detect_mode_no_index_is_greenfield(tmp_path: Path):
    """A populated project with no FAISS meta still falls to greenfield."""
    from cgx.session.mode import detect_mode
    (tmp_path / "main.py").write_text("print('hi')\n")
    assert detect_mode(project_root=str(tmp_path)) is SessionMode.GREENFIELD


def test_detect_mode_with_index_is_explore(tmp_path: Path):
    from cgx.session.mode import detect_mode
    (tmp_path / "main.py").write_text("print('hi')\n")
    idx = tmp_path / "cgx_index"
    idx.mkdir()
    (idx / "meta.json").write_text("{}")
    rec = tmp_path / "records.jsonl"
    rec.write_text("{}\n")
    mode = detect_mode(project_root=str(tmp_path),
                       index_dir=str(idx), records_path=str(rec))
    assert mode is SessionMode.EXPLORE


# --------------------- router (no IO) ---------------------

def test_router_first_message_greenfield_spawns_clarify_root():
    session = Session.new("build a flask api", mode=SessionMode.GREENFIELD)
    plan = Router().on_user_message(
        session=session, message="build a flask api", tasks=[])
    assert len(plan) == 1
    action = plan.actions[0]
    assert isinstance(action, CreateTask)
    assert action.task.kind is TaskKind.CLARIFY_REQUIREMENTS
    assert action.task.inputs["goal"] == "build a flask api"
    assert action.task.parent_task_id is None


def test_router_clarify_completion_spawns_ask_clarify_answers():
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    clarify = TaskNode.new(
        session.session_id, TaskKind.CLARIFY_REQUIREMENTS, "clarify",
        inputs={"goal": "g"})
    clarify.produced_artifact_id = "art_req"
    clarify.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=clarify, tasks=[clarify])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    ask = creates[0].task
    assert ask.kind is TaskKind.ASK_USER
    assert ask.parent_task_id == clarify.task_id
    assert ask.inputs["expected_kind"] == "clarify_answers"
    assert ask.inputs["requirements_artifact_id"] == "art_req"
    assert ask.inputs["prior_goal"] == "g"


def test_router_clarify_answers_decision_spawns_decompose():
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    ask = TaskNode.new(
        session.session_id, TaskKind.ASK_USER, "answer",
        inputs={"expected_kind": "clarify_answers",
                "requirements_artifact_id": "art_req",
                "prior_goal": "build flask api"})
    decision = Decision.new(
        session.session_id, ask.task_id, DecisionKind.CLARIFY_ANSWERS,
        "answer", {"answers": {"q1": "Python + Flask",
                                "q2": "JSON on disk"}})
    plan = Router().on_decision_recorded(
        session=session, decision=decision, tasks=[ask])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    dec_task = creates[0].task
    assert dec_task.kind is TaskKind.DECOMPOSE
    assert dec_task.parent_task_id == ask.task_id
    assert dec_task.inputs["answers"] == {"q1": "Python + Flask",
                                          "q2": "JSON on disk"}
    assert dec_task.inputs["prior_goal"] == "build flask api"
    assert dec_task.inputs["requirements_artifact_id"] == "art_req"


def test_router_clarify_answers_with_empty_answers_has_no_successor():
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    ask = TaskNode.new(
        session.session_id, TaskKind.ASK_USER, "answer",
        inputs={"expected_kind": "clarify_answers"})
    decision = Decision.new(
        session.session_id, ask.task_id, DecisionKind.CLARIFY_ANSWERS,
        "answer", {"answers": {}})
    plan = Router().on_decision_recorded(
        session=session, decision=decision, tasks=[ask])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert creates == []


def test_router_decompose_completion_spawns_ask_approve_plan():
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    dec = TaskNode.new(
        session.session_id, TaskKind.DECOMPOSE, "decompose",
        inputs={"prior_goal": "g",
                "requirements_artifact_id": "art_req"})
    dec.produced_artifact_id = "art_plan"
    dec.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=dec, tasks=[dec])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    ask = creates[0].task
    assert ask.kind is TaskKind.ASK_USER
    assert ask.parent_task_id == dec.task_id
    assert ask.inputs["expected_kind"] == "approve_plan"
    assert ask.inputs["work_plan_artifact_id"] == "art_plan"
    assert ask.inputs["prior_goal"] == "g"
    assert ask.inputs["requirements_artifact_id"] == "art_req"


def test_router_approve_plan_true_spawns_scaffold():
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    ask = TaskNode.new(
        session.session_id, TaskKind.ASK_USER, "approve plan",
        inputs={"expected_kind": "approve_plan",
                "work_plan_artifact_id": "art_plan",
                "prior_goal": "g"})
    decision = Decision.new(
        session.session_id, ask.task_id, DecisionKind.APPROVE_PLAN,
        "approve plan", {"approved": True})
    plan = Router().on_decision_recorded(
        session=session, decision=decision, tasks=[ask])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    sc = creates[0].task
    assert sc.kind is TaskKind.SCAFFOLD
    assert sc.parent_task_id == ask.task_id
    assert sc.inputs["work_plan_artifact_id"] == "art_plan"
    assert sc.inputs["prior_goal"] == "g"
    assert sc.inputs["decision_id"] == decision.decision_id


def test_router_approve_plan_false_has_no_successor():
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    ask = TaskNode.new(
        session.session_id, TaskKind.ASK_USER, "approve plan",
        inputs={"expected_kind": "approve_plan",
                "work_plan_artifact_id": "art_plan"})
    decision = Decision.new(
        session.session_id, ask.task_id, DecisionKind.APPROVE_PLAN,
        "approve plan", {"approved": False})
    plan = Router().on_decision_recorded(
        session=session, decision=decision, tasks=[ask])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert creates == []


def test_router_scaffold_completion_spawns_apply_with_greenfield_mode():
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    sc = TaskNode.new(
        session.session_id, TaskKind.SCAFFOLD, "scaffold",
        inputs={"prior_goal": "g",
                "work_plan_artifact_id": "art_plan"})
    sc.produced_artifact_id = "art_scaffold"
    sc.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=sc, tasks=[sc])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    ap = creates[0].task
    assert ap.kind is TaskKind.APPLY
    assert ap.parent_task_id == sc.task_id
    assert ap.inputs["scaffold_artifact_id"] == "art_scaffold"
    assert ap.inputs["plan_artifact_id"] == "art_scaffold"
    assert ap.inputs["mode"] == SessionMode.GREENFIELD.value


# --------------------- build_decision validation ---------------------

def test_build_decision_rejects_clarify_answers_without_answers():
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    ask = TaskNode.new(session.session_id, TaskKind.ASK_USER, "ans",
                       inputs={"expected_kind": "clarify_answers"})
    with pytest.raises(ValueError):
        build_decision(session_id=session.session_id, task=ask, chosen={})


def test_build_decision_rejects_clarify_answers_with_non_dict_answers():
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    ask = TaskNode.new(session.session_id, TaskKind.ASK_USER, "ans",
                       inputs={"expected_kind": "clarify_answers"})
    with pytest.raises(ValueError):
        build_decision(session_id=session.session_id, task=ask,
                       chosen={"answers": "not a dict"})


def test_build_decision_accepts_clarify_answers_with_answers_dict():
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    ask = TaskNode.new(session.session_id, TaskKind.ASK_USER, "ans",
                       inputs={"expected_kind": "clarify_answers"})
    d = build_decision(
        session_id=session.session_id, task=ask,
        chosen={"answers": {"q1": "Python + Flask"}})
    assert d.kind is DecisionKind.CLARIFY_ANSWERS
    assert d.chosen["answers"] == {"q1": "Python + Flask"}


def test_build_decision_rejects_approve_plan_without_approved_key():
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    ask = TaskNode.new(session.session_id, TaskKind.ASK_USER, "ok?",
                       inputs={"expected_kind": "approve_plan"})
    with pytest.raises(ValueError):
        build_decision(session_id=session.session_id, task=ask, chosen={})


def test_build_decision_accepts_approve_plan_false():
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    ask = TaskNode.new(session.session_id, TaskKind.ASK_USER, "ok?",
                       inputs={"expected_kind": "approve_plan"})
    d = build_decision(session_id=session.session_id, task=ask,
                       chosen={"approved": False})
    assert d.kind is DecisionKind.APPROVE_PLAN
    assert d.chosen["approved"] is False


# --------------------- executor unit tests ---------------------

class _StubProvider:
    """Deterministic chat provider for executor unit tests."""

    def __init__(self, response: str):
        self._response = response
        self.calls: list = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {"content": self._response}


def test_clarify_executor_empty_goal_fails():
    from cgx.session.tasks.clarify_requirements import (
        run_clarify_requirements,
    )
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    t = TaskNode.new(session.session_id, TaskKind.CLARIFY_REQUIREMENTS,
                     "c", inputs={"goal": ""})
    result = run_clarify_requirements(t, ExecutorDeps())
    assert result.failure
    assert "empty goal" in result.failure


def test_clarify_executor_falls_back_when_no_provider():
    from cgx.session.tasks.clarify_requirements import (
        run_clarify_requirements,
    )
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    t = TaskNode.new(session.session_id, TaskKind.CLARIFY_REQUIREMENTS,
                     "c", inputs={"goal": "build a Flask app"})
    result = run_clarify_requirements(t, ExecutorDeps())
    assert result.failure is None
    assert result.artifact is not None
    assert result.artifact.kind is ArtifactKind.REQUIREMENTS_SHEET
    assert result.artifact.content["source"] == "fallback"
    assert len(result.artifact.content["questions"]) >= 3
    assert result.outputs["question_count"] >= 3


def test_clarify_executor_uses_llm_response_when_well_formed():
    import json
    from cgx.session.tasks.clarify_requirements import (
        run_clarify_requirements,
    )
    payload = json.dumps({"questions": [
        {"id": "q1", "prompt": "Which framework?",
         "suggested": ["FastAPI", "Flask"]},
        {"id": "q2", "prompt": "Which database?"},
        {"id": "q3", "prompt": "Need auth?"},
    ]})
    provider = _StubProvider(payload)
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    t = TaskNode.new(session.session_id, TaskKind.CLARIFY_REQUIREMENTS,
                     "c", inputs={"goal": "build a Flask app"})
    result = run_clarify_requirements(
        t, ExecutorDeps(provider=provider))
    assert result.failure is None
    assert result.artifact.content["source"] == "llm"
    assert [q["id"] for q in result.artifact.content["questions"]] == [
        "q1", "q2", "q3"]
    assert provider.calls  # provider was actually consulted


def test_clarify_executor_falls_back_when_llm_returns_too_few():
    import json
    from cgx.session.tasks.clarify_requirements import (
        run_clarify_requirements,
    )
    provider = _StubProvider(json.dumps({"questions": [
        {"id": "q1", "prompt": "only one"}]}))
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    t = TaskNode.new(session.session_id, TaskKind.CLARIFY_REQUIREMENTS,
                     "c", inputs={"goal": "build a Flask app"})
    result = run_clarify_requirements(
        t, ExecutorDeps(provider=provider))
    assert result.failure is None
    assert result.artifact.content["source"] == "fallback"


def test_chips_from_hint_derives_from_example_marker():
    from cgx.session.tasks.clarify_requirements import _chips_from_hint
    assert _chips_from_hint("e.g. Python + FastAPI, Node + Express") == [
        "Python + FastAPI", "Node + Express"]
    assert _chips_from_hint("Example: SQLite, PostgreSQL") == [
        "SQLite", "PostgreSQL"]
    assert _chips_from_hint("SQLite or Postgres") == ["SQLite", "Postgres"]
    # Plain instruction with no marker/list yields nothing.
    assert _chips_from_hint("Pick a storage layer") == []
    assert _chips_from_hint("") == []


def test_clarify_parse_backfills_chips_from_hint_when_suggested_missing():
    import json
    from cgx.session.tasks.clarify_requirements import (
        run_clarify_requirements,
    )
    payload = json.dumps({"questions": [
        {"id": "q1", "prompt": "Which framework?",
         "hint": "e.g. FastAPI, Flask"},
        {"id": "q2", "prompt": "Which database?",
         "hint": "Example: SQLite, Postgres"},
        {"id": "q3", "prompt": "Need auth?", "hint": "Pick a scope"},
    ]})
    provider = _StubProvider(payload)
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    t = TaskNode.new(session.session_id, TaskKind.CLARIFY_REQUIREMENTS,
                     "c", inputs={"goal": "build a Flask app"})
    result = run_clarify_requirements(t, ExecutorDeps(provider=provider))
    assert result.failure is None
    assert result.artifact.content["source"] == "llm"
    qs = {q["id"]: q for q in result.artifact.content["questions"]}
    assert qs["q1"]["suggested"] == ["FastAPI", "Flask"]
    assert qs["q2"]["suggested"] == ["SQLite", "Postgres"]
    # No example marker/list -> stays empty (no bogus chip).
    assert qs["q3"]["suggested"] == []


def test_decompose_executor_requires_provider_and_store(store):
    from cgx.session.tasks.decompose import run_decompose
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    t = TaskNode.new(session.session_id, TaskKind.DECOMPOSE, "d",
                     inputs={"prior_goal": "g",
                             "answers": {"q1": "Flask"}})
    # No provider in deps.
    result = run_decompose(t, ExecutorDeps(store=store))
    assert result.failure
    assert "LLM provider" in result.failure
    # No store in deps.
    result = run_decompose(t, ExecutorDeps(provider=_StubProvider("{}")))
    assert result.failure
    assert "store" in result.failure


def test_decompose_executor_happy_path_emits_work_plan(
        store, monkeypatch):
    from cgx.session.tasks import decompose as dec_mod
    from cgx.session.tasks.decompose import run_decompose
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    req = Artifact.new(
        session.session_id, "task_x", ArtifactKind.REQUIREMENTS_SHEET,
        {"goal": "g", "questions": [
            {"id": "q1", "prompt": "Framework?"}], "source": "llm"})
    store.save_artifact(req)

    def fake_manifest(composed, provider, goal=None):
        return {
            "plan_md": "## Plan\n- app.py\n- README.md",
            "layers": [{"name": "app", "files": [
                {"path": "app.py", "description": "entrypoint"},
                {"path": "README.md", "description": "docs"}]}],
        }

    monkeypatch.setattr("cgx.answer.engine.plan_scaffold_manifest",
                        fake_manifest)
    t = TaskNode.new(session.session_id, TaskKind.DECOMPOSE, "d",
                     inputs={"prior_goal": "build flask api",
                             "answers": {"q1": "Python + Flask"},
                             "requirements_artifact_id": req.artifact_id})
    result = run_decompose(
        t, ExecutorDeps(provider=_StubProvider(""), store=store))
    assert result.failure is None
    assert result.artifact.kind is ArtifactKind.WORK_PLAN
    layers = result.artifact.content["layers"]
    assert layers[0]["files"][0]["path"] == "app.py"
    assert result.outputs["file_count"] == 2
    assert result.outputs["layer_count"] == 1
    # The composed goal carried the user's answer through to the planner.
    assert "Python + Flask" in result.artifact.content["composed_goal"]


def test_decompose_executor_stores_contracts_on_work_plan(store, monkeypatch):
    """P0: a planner ``contracts`` block is normalized onto the WORK_PLAN."""
    from cgx.session.tasks.decompose import run_decompose
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)

    def fake_manifest(composed, provider, goal=None):
        return {
            "plan_md": "p",
            "contracts": {
                "endpoints": [{"method": "POST", "path": "/api/calc",
                               "request": {"expr": "str"}}],
                "functions": [{"signature": "evaluate(expr: str) -> float",
                               "module": "backend/calc.py"}],
                "junk": "dropped",
            },
            "layers": [{"name": "core", "files": [
                {"path": "backend/calc.py", "description": "core"}]}],
        }

    monkeypatch.setattr("cgx.answer.engine.plan_scaffold_manifest",
                        fake_manifest)
    t = TaskNode.new(session.session_id, TaskKind.DECOMPOSE, "d",
                     inputs={"prior_goal": "build a calc", "answers": {}})
    result = run_decompose(
        t, ExecutorDeps(provider=_StubProvider(""), store=store))
    assert result.failure is None
    contracts = result.artifact.content["contracts"]
    assert set(contracts.keys()) == {"endpoints", "functions"}
    assert contracts["endpoints"][0]["path"] == "/api/calc"
    # One endpoint + one function = two declared contract entries.
    assert result.outputs["contract_count"] == 2


def test_decompose_executor_defaults_contracts_to_empty(store, monkeypatch):
    """A planner that omits contracts stores an empty (not missing) block."""
    from cgx.session.tasks.decompose import run_decompose
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    monkeypatch.setattr(
        "cgx.answer.engine.plan_scaffold_manifest",
        lambda *a, **kw: {"plan_md": "p", "layers": [
            {"name": "core", "files": [
                {"path": "app.py", "description": "entry"}]}]})
    t = TaskNode.new(session.session_id, TaskKind.DECOMPOSE, "d",
                     inputs={"prior_goal": "g", "answers": {}})
    result = run_decompose(
        t, ExecutorDeps(provider=_StubProvider(""), store=store))
    assert result.failure is None
    assert result.artifact.content["contracts"] == {}
    assert result.outputs["contract_count"] == 0


def test_decompose_executor_empty_manifest_is_failure(store, monkeypatch):
    from cgx.session.tasks.decompose import run_decompose
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    monkeypatch.setattr(
        "cgx.answer.engine.plan_scaffold_manifest",
        lambda *a, **kw: {"plan_md": "", "layers": []})
    t = TaskNode.new(session.session_id, TaskKind.DECOMPOSE, "d",
                     inputs={"prior_goal": "g",
                             "answers": {"q1": "Flask"}})
    result = run_decompose(
        t, ExecutorDeps(provider=_StubProvider(""), store=store))
    assert result.failure
    assert "empty manifest" in result.failure


def _run_decompose_with_manifest(store, monkeypatch, manifest):
    from cgx.session.tasks.decompose import run_decompose
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    monkeypatch.setattr("cgx.answer.engine.plan_scaffold_manifest",
                        lambda *a, **kw: manifest)
    t = TaskNode.new(session.session_id, TaskKind.DECOMPOSE, "d",
                     inputs={"prior_goal": "g", "answers": {}})
    return run_decompose(
        t, ExecutorDeps(provider=_StubProvider(""), store=store))


def test_decompose_orders_files_by_dependency(store, monkeypatch):
    # src/app.py imports src/util.py but is declared first; the topo sort
    # must reorder so the dependency is generated before its consumer.
    result = _run_decompose_with_manifest(store, monkeypatch, {
        "plan_md": "p",
        "layers": [{"name": "core", "files": [
            {"path": "src/app.py", "description": "entry",
             "depends_on": ["src/util.py"]},
            {"path": "src/util.py", "description": "helpers"}]}],
    })
    assert result.failure is None
    paths = [f["path"] for f in result.artifact.content["layers"][0]["files"]]
    assert paths.index("src/util.py") < paths.index("src/app.py")


def test_decompose_preserves_order_without_dependency_hints(store, monkeypatch):
    # No depends_on anywhere -> declared order is preserved (stable sort).
    result = _run_decompose_with_manifest(store, monkeypatch, {
        "plan_md": "p",
        "layers": [{"name": "core", "files": [
            {"path": "src/a.py", "description": "a"},
            {"path": "src/b.py", "description": "b"},
            {"path": "src/c.py", "description": "c"}]}],
    })
    assert result.failure is None
    paths = [f["path"] for f in result.artifact.content["layers"][0]["files"]]
    assert paths == ["src/a.py", "src/b.py", "src/c.py"]


def test_decompose_fails_on_dangling_dependency(store, monkeypatch):
    result = _run_decompose_with_manifest(store, monkeypatch, {
        "plan_md": "p",
        "layers": [{"name": "core", "files": [
            {"path": "src/app.py", "description": "entry",
             "depends_on": ["src/missing.py"]}]}],
    })
    assert result.failure
    assert "dangling dependency" in result.failure
    assert "src/missing.py" in result.failure


def test_decompose_fails_on_dependency_cycle(store, monkeypatch):
    result = _run_decompose_with_manifest(store, monkeypatch, {
        "plan_md": "p",
        "layers": [{"name": "core", "files": [
            {"path": "a.py", "description": "a", "depends_on": ["b.py"]},
            {"path": "b.py", "description": "b", "depends_on": ["a.py"]}]}],
    })
    assert result.failure
    assert "circular dependency" in result.failure


def test_decompose_fails_when_no_source_entry_point(store, monkeypatch):
    result = _run_decompose_with_manifest(store, monkeypatch, {
        "plan_md": "p",
        "layers": [{"name": "meta", "files": [
            {"path": "README.md", "description": "docs"},
            {"path": "package.json", "description": "config"}]}],
    })
    assert result.failure
    assert "no runnable source" in result.failure


def test_decompose_fails_when_only_test_files(store, monkeypatch):
    result = _run_decompose_with_manifest(store, monkeypatch, {
        "plan_md": "p",
        "layers": [{"name": "tests", "files": [
            {"path": "tests/test_app.py", "description": "tests"},
            {"path": "README.md", "description": "docs"}]}],
    })
    assert result.failure
    assert "no runnable source" in result.failure


def test_scaffold_executor_missing_work_plan_fails(store):
    from cgx.session.tasks.scaffold import run_scaffold
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    t = TaskNode.new(session.session_id, TaskKind.SCAFFOLD, "s",
                     inputs={"work_plan_artifact_id": "art_missing"})
    result = run_scaffold(
        t, ExecutorDeps(provider=_StubProvider(""), store=store))
    assert result.failure
    assert "missing" in result.failure


def test_scaffold_executor_happy_path_accumulates_context(
        store, monkeypatch):
    from cgx.session.tasks.scaffold import run_scaffold
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    plan = Artifact.new(
        session.session_id, "task_x", ArtifactKind.WORK_PLAN, {
            "prior_goal": "g",
            "composed_goal": "build flask api",
            "answers": {},
            "plan_md": "##",
            "layers": [{"name": "app", "files": [
                {"path": "app.py", "description": "entry"},
                {"path": "README.md", "description": "docs"}]}],
        })
    store.save_artifact(plan)

    seen_contexts: list = []

    def fake_generate(path, description, provider, *,
                      layer=None, existing_files_with_content=None,
                      goal=None, on_token=None, depends_on=None,
                      contracts=None):
        seen_contexts.append(
            [c["path"] for c in (existing_files_with_content or [])])
        body = f"# {path}\nprint('{path}')\n"
        return {
            "file": path, "patch": f"+++ {path}\n{body}",
            "content": body, "syntax_ok": True, "confidence": 0.9,
        }

    monkeypatch.setattr("cgx.answer.engine.generate_single_scaffold_file",
                        fake_generate)
    t = TaskNode.new(session.session_id, TaskKind.SCAFFOLD, "s",
                     inputs={"work_plan_artifact_id": plan.artifact_id})
    result = run_scaffold(
        t, ExecutorDeps(provider=_StubProvider(""), store=store))
    assert result.failure is None
    assert result.artifact.kind is ArtifactKind.SCAFFOLD_PATCHES
    diffs = result.artifact.content["diffs"]
    assert [d["file"] for d in diffs] == ["app.py", "README.md"]
    assert result.outputs["generated_count"] == 2
    assert result.outputs["failed_count"] == 0
    # The README generation saw app.py as accumulated context (the order
    # is essential -- otherwise cross-file imports won't resolve).
    assert seen_contexts == [[], ["app.py"]]


def test_scaffold_threads_contracts_to_generator(store, monkeypatch):
    """P0: the WORK_PLAN contracts block is threaded into every file gen."""
    from cgx.session.tasks.scaffold import run_scaffold
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    contracts = {"endpoints": [{"method": "GET", "path": "/api/ping"}]}
    plan = Artifact.new(
        session.session_id, "task_x", ArtifactKind.WORK_PLAN, {
            "prior_goal": "g", "composed_goal": "build an api",
            "answers": {}, "plan_md": "", "contracts": contracts,
            "layers": [{"name": "app", "files": [
                {"path": "app.py", "description": "entry"}]}],
        })
    store.save_artifact(plan)

    seen: list = []

    def fake_generate(path, description, provider, *,
                      layer=None, existing_files_with_content=None,
                      goal=None, on_token=None, depends_on=None,
                      contracts=None):
        seen.append(contracts)
        return {"file": path, "patch": f"+++ {path}\nx",
                "content": "x", "syntax_ok": True}

    monkeypatch.setattr("cgx.answer.engine.generate_single_scaffold_file",
                        fake_generate)
    t = TaskNode.new(session.session_id, TaskKind.SCAFFOLD, "s",
                     inputs={"work_plan_artifact_id": plan.artifact_id})
    result = run_scaffold(
        t, ExecutorDeps(provider=_StubProvider(""), store=store))
    assert result.failure is None
    assert seen == [contracts]


def test_scaffold_passes_none_contracts_when_absent(store, monkeypatch):
    """A WORK_PLAN without contracts threads ``None`` (prompt unchanged)."""
    from cgx.session.tasks.scaffold import run_scaffold
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    plan = Artifact.new(
        session.session_id, "task_x", ArtifactKind.WORK_PLAN, {
            "prior_goal": "g", "composed_goal": "g", "answers": {},
            "plan_md": "",
            "layers": [{"name": "app", "files": [
                {"path": "app.py", "description": "entry"}]}],
        })
    store.save_artifact(plan)

    seen: list = []

    def fake_generate(path, *a, contracts=None, **kw):
        seen.append(contracts)
        return {"file": path, "patch": f"+++ {path}\nx",
                "content": "x", "syntax_ok": True}

    monkeypatch.setattr("cgx.answer.engine.generate_single_scaffold_file",
                        fake_generate)
    t = TaskNode.new(session.session_id, TaskKind.SCAFFOLD, "s",
                     inputs={"work_plan_artifact_id": plan.artifact_id})
    result = run_scaffold(
        t, ExecutorDeps(provider=_StubProvider(""), store=store))
    assert result.failure is None
    assert seen == [None]


def test_scaffold_executor_records_partial_failure(store, monkeypatch):
    from cgx.session.tasks.scaffold import run_scaffold
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    plan = Artifact.new(
        session.session_id, "task_x", ArtifactKind.WORK_PLAN, {
            "plan_md": "", "composed_goal": "g", "prior_goal": "g",
            "layers": [{"name": "app", "files": [
                {"path": "ok.py", "description": ""},
                {"path": "bad.py", "description": ""}]}],
        })
    store.save_artifact(plan)

    def fake_generate(path, *a, **kw):
        if path == "bad.py":
            raise RuntimeError("model timed out")
        return {"file": path, "patch": f"+++ {path}\nx",
                "content": "x", "syntax_ok": True}

    monkeypatch.setattr("cgx.answer.engine.generate_single_scaffold_file",
                        fake_generate)
    t = TaskNode.new(session.session_id, TaskKind.SCAFFOLD, "s",
                     inputs={"work_plan_artifact_id": plan.artifact_id})
    result = run_scaffold(
        t, ExecutorDeps(provider=_StubProvider(""), store=store))
    assert result.failure is None
    assert result.outputs["generated_count"] == 1
    assert result.outputs["failed_count"] == 1
    failed = result.artifact.content["failed"]
    assert failed[0]["file"] == "bad.py"
    assert "RuntimeError" in failed[0]["error"]
    # The failed list is surfaced in outputs so the router can enrich the
    # regenerate constraint with concrete per-file errors.
    assert result.outputs["failed"] == failed


def test_scaffold_syntax_invalid_file_becomes_explicit_failure(store,
                                                               monkeypatch):
    """A file the generator flags ``syntax_ok=False`` must not ship a diff.

    Instead of adding a broken diff to ``generated`` (which APPLY's own
    syntax gate would then silently drop), it becomes an explicit
    file-level failure carrying the generator's ``syntax_error`` so the
    router can turn it into a targeted regenerate (P2.1).
    """
    from cgx.session.tasks.scaffold import run_scaffold
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    plan = Artifact.new(
        session.session_id, "task_x", ArtifactKind.WORK_PLAN, {
            "plan_md": "", "composed_goal": "g", "prior_goal": "g",
            "layers": [{"name": "app", "files": [
                {"path": "ok.py", "description": ""},
                {"path": "broken.py", "description": ""}]}],
        })
    store.save_artifact(plan)

    def fake_generate(path, *a, **kw):
        if path == "broken.py":
            return {"file": path, "patch": f"+++ {path}\ndef f(:\n",
                    "content": "def f(:\n", "syntax_ok": False,
                    "syntax_error": "invalid syntax (line 1)"}
        return {"file": path, "patch": f"+++ {path}\nx",
                "content": "x", "syntax_ok": True}

    monkeypatch.setattr("cgx.answer.engine.generate_single_scaffold_file",
                        fake_generate)
    t = TaskNode.new(session.session_id, TaskKind.SCAFFOLD, "s",
                     inputs={"work_plan_artifact_id": plan.artifact_id})
    result = run_scaffold(
        t, ExecutorDeps(provider=_StubProvider(""), store=store))
    assert result.failure is None
    assert result.outputs["generated_count"] == 1
    assert result.outputs["failed_count"] == 1
    # The broken file is absent from the emitted diffs entirely.
    assert [d["file"] for d in result.artifact.content["diffs"]] == ["ok.py"]
    failed = result.artifact.content["failed"]
    assert failed[0]["file"] == "broken.py"
    assert "invalid syntax" in failed[0]["error"]


def test_scaffold_targeted_regenerate_reuses_good_diffs(store, monkeypatch):
    """regenerate_files -> only the failed path is generated; good diffs reused."""
    from cgx.answer.engine import _content_to_new_file_patch
    from cgx.session.tasks.scaffold import run_scaffold
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    plan = Artifact.new(
        session.session_id, "task_x", ArtifactKind.WORK_PLAN, {
            "prior_goal": "g", "composed_goal": "build a react calculator",
            "answers": {}, "plan_md": "",
            "layers": [{"name": "ui", "files": [
                {"path": "src/index.js", "description": "entry"},
                {"path": "src/App.jsx", "description": "root component"},
                {"path": "README.md", "description": "docs"}]}],
        })
    store.save_artifact(plan)
    prior = Artifact.new(
        session.session_id, "task_sc0", ArtifactKind.SCAFFOLD_PATCHES, {
            "diffs": [
                {"file": "src/index.js",
                 "patch": _content_to_new_file_patch(
                     "src/index.js", "import App from './App'\n")},
                {"file": "README.md",
                 "patch": _content_to_new_file_patch("README.md", "# Calc\n")},
                # A stale App.jsx diff that must NOT be reused -- it's the
                # file being regenerated.
                {"file": "src/App.jsx",
                 "patch": _content_to_new_file_patch("src/App.jsx", "BROKEN")},
            ],
        })
    store.save_artifact(prior)

    seen: list = []

    def fake_generate(path, description, provider, *,
                      layer=None, existing_files_with_content=None,
                      goal=None, on_token=None, depends_on=None,
                      contracts=None):
        seen.append({"path": path,
                     "context": [c["path"]
                                 for c in (existing_files_with_content or [])]})
        body = "export default function App(){return null}\n"
        return {"file": path, "patch": _content_to_new_file_patch(path, body),
                "content": body, "syntax_ok": True, "confidence": 0.9}

    monkeypatch.setattr("cgx.answer.engine.generate_single_scaffold_file",
                        fake_generate)
    t = TaskNode.new(
        session.session_id, TaskKind.SCAFFOLD, "s",
        inputs={"work_plan_artifact_id": plan.artifact_id,
                "regenerate_files": ["src/App.jsx"],
                "prior_scaffold_artifact_id": prior.artifact_id})
    result = run_scaffold(
        t, ExecutorDeps(provider=_StubProvider(""), store=store))
    assert result.failure is None
    # Only the failed file was generated; the good files were reused.
    assert [s["path"] for s in seen] == ["src/App.jsx"]
    # The regenerated file saw both reused good files as cross-file context
    # (so imports resolve), reconstructed from their patches -- never the
    # stale App.jsx.
    assert set(seen[0]["context"]) == {"src/index.js", "README.md"}
    # The emitted diff set carries all three files: two reused verbatim plus
    # the freshly regenerated one.
    files = {d["file"] for d in result.artifact.content["diffs"]}
    assert files == {"src/index.js", "src/App.jsx", "README.md"}
    reused = next(d for d in result.artifact.content["diffs"]
                  if d["file"] == "src/index.js")
    assert "import App from './App'" in reused["patch"]
    gen = {g["file"]: g for g in result.artifact.content["generated"]}
    assert gen["src/index.js"].get("reused") is True
    assert gen["src/App.jsx"].get("reused") is not True


def test_scaffold_targeted_regenerate_falls_back_without_prior(store, monkeypatch):
    """A regenerate_files marker with no resolvable prior -> whole-tree regen."""
    from cgx.session.tasks.scaffold import run_scaffold
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    plan = Artifact.new(
        session.session_id, "task_x", ArtifactKind.WORK_PLAN, {
            "prior_goal": "g", "composed_goal": "g", "answers": {},
            "plan_md": "",
            "layers": [{"name": "app", "files": [
                {"path": "a.py", "description": ""},
                {"path": "b.py", "description": ""}]}],
        })
    store.save_artifact(plan)

    seen: list = []

    def fake_generate(path, *a, **kw):
        seen.append(path)
        return {"file": path, "patch": f"+++ {path}\nx",
                "content": "x", "syntax_ok": True}

    monkeypatch.setattr("cgx.answer.engine.generate_single_scaffold_file",
                        fake_generate)
    t = TaskNode.new(
        session.session_id, TaskKind.SCAFFOLD, "s",
        inputs={"work_plan_artifact_id": plan.artifact_id,
                "regenerate_files": ["a.py"],
                "prior_scaffold_artifact_id": "art_missing"})
    result = run_scaffold(
        t, ExecutorDeps(provider=_StubProvider(""), store=store))
    assert result.failure is None
    # Prior artifact unresolvable -> every manifest file is regenerated.
    assert seen == ["a.py", "b.py"]


def test_scaffold_checkpoints_progress_after_each_layer(store, monkeypatch):
    """B4: the SCAFFOLD_PATCHES artifact is upserted after every layer.

    Each checkpoint carries the files generated so far with ``complete``
    False; only the returned artifact is finalised (``complete`` True),
    so a crash mid-run leaves resumable partial progress on disk.
    """
    import copy
    from cgx.session.tasks.scaffold import run_scaffold
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    plan = Artifact.new(
        session.session_id, "task_x", ArtifactKind.WORK_PLAN, {
            "composed_goal": "g", "prior_goal": "g",
            "layers": [
                {"name": "l1", "files": [{"path": "a.py", "description": ""}]},
                {"name": "l2", "files": [{"path": "b.py", "description": ""}]}],
        })
    store.save_artifact(plan)

    def fake_generate(path, *a, **kw):
        body = f"# {path}\n"
        return {"file": path, "patch": f"+++ {path}\n{body}",
                "content": body, "syntax_ok": True}

    monkeypatch.setattr("cgx.answer.engine.generate_single_scaffold_file",
                        fake_generate)

    checkpoints: list = []
    real_save = store.save_artifact

    def spy_save(artifact):
        if artifact.kind is ArtifactKind.SCAFFOLD_PATCHES:
            snap = copy.deepcopy(artifact.content)
            checkpoints.append(
                ([d["file"] for d in snap["diffs"]], snap.get("complete")))
        return real_save(artifact)

    monkeypatch.setattr(store, "save_artifact", spy_save)
    t = TaskNode.new(session.session_id, TaskKind.SCAFFOLD, "s",
                     inputs={"work_plan_artifact_id": plan.artifact_id})
    result = run_scaffold(
        t, ExecutorDeps(provider=_StubProvider(""), store=store))
    assert result.failure is None
    # One checkpoint per layer, each incomplete and cumulative.
    assert checkpoints == [(["a.py"], False), (["a.py", "b.py"], False)]
    # The returned artifact is the same row, finalised for the runner to save.
    assert result.artifact.content["complete"] is True
    assert result.outputs["generated_count"] == 2


def test_scaffold_resumes_from_checkpoint_skipping_done_files(
        store, monkeypatch):
    """B4: a resume pointer seeds checkpointed files and regenerates the rest."""
    from cgx.answer.engine import _content_to_new_file_patch
    from cgx.session.tasks.scaffold import run_scaffold
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    plan = Artifact.new(
        session.session_id, "task_x", ArtifactKind.WORK_PLAN, {
            "composed_goal": "g", "prior_goal": "g",
            "layers": [
                {"name": "l1", "files": [{"path": "a.py", "description": ""}]},
                {"name": "l2", "files": [{"path": "b.py", "description": ""}]}],
        })
    store.save_artifact(plan)
    # A crashed prior attempt that checkpointed only a.py.
    ckpt = Artifact.new(
        session.session_id, "crashed_task", ArtifactKind.SCAFFOLD_PATCHES, {
            "work_plan_artifact_id": plan.artifact_id,
            "diffs": [{"file": "a.py",
                       "patch": _content_to_new_file_patch("a.py", "# a\n")}],
            "generated": [{"file": "a.py"}], "failed": [], "complete": False,
        })
    store.save_artifact(ckpt)

    seen: list = []

    def fake_generate(path, *a, **kw):
        seen.append(path)
        body = f"# {path}\n"
        return {"file": path, "patch": f"+++ {path}\n{body}",
                "content": body, "syntax_ok": True}

    monkeypatch.setattr("cgx.answer.engine.generate_single_scaffold_file",
                        fake_generate)
    t = TaskNode.new(
        session.session_id, TaskKind.SCAFFOLD, "s",
        inputs={"work_plan_artifact_id": plan.artifact_id,
                "resume_scaffold_artifact_id": ckpt.artifact_id})
    result = run_scaffold(
        t, ExecutorDeps(provider=_StubProvider(""), store=store))
    assert result.failure is None
    # a.py resumed from the checkpoint; only b.py was regenerated.
    assert seen == ["b.py"]
    diffs = result.artifact.content["diffs"]
    assert [d["file"] for d in diffs] == ["a.py", "b.py"]
    gen = {g["file"]: g for g in result.artifact.content["generated"]}
    assert gen["a.py"].get("resumed") is True
    assert gen["b.py"].get("resumed") is not True
    assert result.outputs["generated_count"] == 2


def test_scaffold_resume_ignores_checkpoint_for_different_work_plan(
        store, monkeypatch):
    """B4: a checkpoint produced for a different work plan is not used to seed."""
    from cgx.answer.engine import _content_to_new_file_patch
    from cgx.session.tasks.scaffold import run_scaffold
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    plan = Artifact.new(
        session.session_id, "task_x", ArtifactKind.WORK_PLAN, {
            "composed_goal": "g", "prior_goal": "g",
            "layers": [{"name": "l1", "files": [
                {"path": "a.py", "description": ""},
                {"path": "b.py", "description": ""}]}],
        })
    store.save_artifact(plan)
    ckpt = Artifact.new(
        session.session_id, "crashed_task", ArtifactKind.SCAFFOLD_PATCHES, {
            "work_plan_artifact_id": "a_different_plan",
            "diffs": [{"file": "a.py",
                       "patch": _content_to_new_file_patch("a.py", "# a\n")}],
            "generated": [{"file": "a.py"}], "failed": [], "complete": False,
        })
    store.save_artifact(ckpt)

    seen: list = []

    def fake_generate(path, *a, **kw):
        seen.append(path)
        body = f"# {path}\n"
        return {"file": path, "patch": f"+++ {path}\n{body}",
                "content": body, "syntax_ok": True}

    monkeypatch.setattr("cgx.answer.engine.generate_single_scaffold_file",
                        fake_generate)
    t = TaskNode.new(
        session.session_id, TaskKind.SCAFFOLD, "s",
        inputs={"work_plan_artifact_id": plan.artifact_id,
                "resume_scaffold_artifact_id": ckpt.artifact_id})
    result = run_scaffold(
        t, ExecutorDeps(provider=_StubProvider(""), store=store))
    assert result.failure is None
    # Checkpoint's work plan mismatched -> nothing seeded, full manifest run.
    assert seen == ["a.py", "b.py"]


def test_scaffold_parallel_generation_preserves_order_and_cross_layer_context(
        store, monkeypatch):
    """B3: CGX_SCAFFOLD_CONCURRENCY fans out within a layer.

    Diffs stay in manifest order regardless of completion order, and every
    file in a later layer still sees the earlier layer's content -- but
    intra-layer siblings are NOT in each other's context (frozen snapshot),
    which is exactly what makes concurrent generation safe.
    """
    import threading
    monkeypatch.setenv("CGX_SCAFFOLD_CONCURRENCY", "4")
    from cgx.session.tasks.scaffold import run_scaffold
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    plan = Artifact.new(
        session.session_id, "task_x", ArtifactKind.WORK_PLAN, {
            "prior_goal": "g", "composed_goal": "build a react app",
            "answers": {}, "plan_md": "",
            "layers": [
                {"name": "base", "files": [
                    {"path": "src/config.js", "description": "config"}]},
                {"name": "ui", "files": [
                    {"path": "src/A.jsx", "description": "A"},
                    {"path": "src/B.jsx", "description": "B"},
                    {"path": "src/C.jsx", "description": "C"}]},
            ],
        })
    store.save_artifact(plan)

    lock = threading.Lock()
    seen: dict = {}
    completion_order: list = []
    # Force a non-manifest completion order (C, then B, then A) by chaining
    # per-file "done" events -- this proves the gather step re-sorts back
    # into manifest order rather than emitting in completion order.
    done = {p: threading.Event()
            for p in ("src/A.jsx", "src/B.jsx", "src/C.jsx")}

    def fake_generate(path, description, provider, *,
                      layer=None, existing_files_with_content=None, goal=None,
                      on_token=None, depends_on=None, contracts=None):
        with lock:
            seen[path] = [c["path"] for c in (existing_files_with_content or [])]
        if path == "src/B.jsx":
            done["src/C.jsx"].wait(2)
        elif path == "src/A.jsx":
            done["src/B.jsx"].wait(2)
        if path in done:
            with lock:
                completion_order.append(path)
            done[path].set()
        body = f"// {path}\n"
        return {"file": path, "patch": f"+++ {path}\n{body}",
                "content": body, "syntax_ok": True, "confidence": 0.9}

    monkeypatch.setattr("cgx.answer.engine.generate_single_scaffold_file",
                        fake_generate)
    t = TaskNode.new(session.session_id, TaskKind.SCAFFOLD, "s",
                     inputs={"work_plan_artifact_id": plan.artifact_id})
    result = run_scaffold(
        t, ExecutorDeps(provider=_StubProvider(""), store=store))
    assert result.failure is None
    assert result.outputs["generated_count"] == 4
    assert result.outputs["failed_count"] == 0
    # Emitted diffs follow manifest order even though C completed first.
    assert [d["file"] for d in result.artifact.content["diffs"]] == [
        "src/config.js", "src/A.jsx", "src/B.jsx", "src/C.jsx"]
    # Cross-layer context preserved: every ui file saw base/config.js.
    for p in ("src/A.jsx", "src/B.jsx", "src/C.jsx"):
        assert seen[p] == ["src/config.js"], p
    # Sanity: the events really did invert completion vs manifest order.
    assert completion_order == ["src/C.jsx", "src/B.jsx", "src/A.jsx"]


def test_scaffold_parallel_generation_aggregates_failures_in_order(
        store, monkeypatch):
    """B3: a mid-layer crash under parallelism is captured, order-stable."""
    monkeypatch.setenv("CGX_SCAFFOLD_CONCURRENCY", "3")
    from cgx.session.tasks.scaffold import run_scaffold
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    plan = Artifact.new(
        session.session_id, "task_x", ArtifactKind.WORK_PLAN, {
            "prior_goal": "g", "composed_goal": "g", "answers": {},
            "plan_md": "",
            "layers": [{"name": "app", "files": [
                {"path": "a.py", "description": ""},
                {"path": "b.py", "description": ""},
                {"path": "c.py", "description": ""}]}],
        })
    store.save_artifact(plan)

    def fake_generate(path, *a, **kw):
        if path == "b.py":
            raise RuntimeError("model timed out")
        return {"file": path, "patch": f"+++ {path}\nx",
                "content": "x", "syntax_ok": True}

    monkeypatch.setattr("cgx.answer.engine.generate_single_scaffold_file",
                        fake_generate)
    t = TaskNode.new(session.session_id, TaskKind.SCAFFOLD, "s",
                     inputs={"work_plan_artifact_id": plan.artifact_id})
    result = run_scaffold(
        t, ExecutorDeps(provider=_StubProvider(""), store=store))
    assert result.failure is None
    assert result.outputs["generated_count"] == 2
    assert result.outputs["failed_count"] == 1
    # Successes keep manifest order; the failed peer is dropped from diffs.
    assert [d["file"] for d in result.artifact.content["diffs"]] == [
        "a.py", "c.py"]
    failed = result.artifact.content["failed"]
    assert [f["file"] for f in failed] == ["b.py"]
    assert "RuntimeError" in failed[0]["error"]
    assert result.outputs["failed"] == failed


def test_scaffold_concurrency_is_provider_gated(monkeypatch):
    """P2.3: cloud providers fan out by default; local GPUs stay serial.

    The gate keys off the provider's ``parallel_scaffold_capable`` flag and
    only when ``CGX_SCAFFOLD_CONCURRENCY`` is unset; the env var overrides
    the gate in both directions.
    """
    from cgx.session.tasks.scaffold import (
        _CLOUD_SCAFFOLD_CONCURRENCY, _scaffold_concurrency)
    monkeypatch.delenv("CGX_SCAFFOLD_CONCURRENCY", raising=False)

    class _Local:
        parallel_scaffold_capable = False

    class _Cloud:
        parallel_scaffold_capable = True

    # Unset env: local (and a provider that lacks the flag entirely) stays
    # serial; a cloud-capable provider fans out to the bounded default.
    assert _scaffold_concurrency(None) == 1
    assert _scaffold_concurrency(_Local()) == 1
    assert _scaffold_concurrency(_Cloud()) == _CLOUD_SCAFFOLD_CONCURRENCY

    # Explicit env overrides the gate in both directions.
    monkeypatch.setenv("CGX_SCAFFOLD_CONCURRENCY", "1")
    assert _scaffold_concurrency(_Cloud()) == 1
    monkeypatch.setenv("CGX_SCAFFOLD_CONCURRENCY", "6")
    assert _scaffold_concurrency(_Local()) == 6
    # Malformed clamps to serial.
    monkeypatch.setenv("CGX_SCAFFOLD_CONCURRENCY", "nope")
    assert _scaffold_concurrency(_Cloud()) == 1


# --------------------- BOOTSTRAP_ENV executor unit tests ---------------------

def test_bootstrap_env_requires_project_root():
    from cgx.session.tasks.bootstrap_env import run_bootstrap_env
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    t = TaskNode.new(session.session_id, TaskKind.BOOTSTRAP_ENV, "boot",
                     inputs={})
    result = run_bootstrap_env(t, ExecutorDeps(store=object()))
    assert result.failure and "project_root" in result.failure


def test_bootstrap_env_skips_non_python_project(tmp_path, store):
    """No manifest -> project_type=unknown -> outcome=skipped, no venv work."""
    from cgx.session.tasks.bootstrap_env import run_bootstrap_env
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    t = TaskNode.new(session.session_id, TaskKind.BOOTSTRAP_ENV, "boot",
                     inputs={"mode": SessionMode.GREENFIELD.value})
    result = run_bootstrap_env(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.failure is None
    assert result.outputs["outcome"] == "skipped"
    assert result.outputs["project_type"] == "unknown"
    assert result.artifact.kind is ArtifactKind.BUILD_REPORT
    assert result.artifact.content["venv_path"] is None
    assert result.artifact.content["resolved_packages"] == []
    assert result.artifact.content["pip_freeze_text"] == ""
    assert "non-python" in (result.artifact.content.get("note") or "")


def test_bootstrap_env_python_takes_priority_over_package_json(tmp_path):
    """A polyglot repo bootstraps Python; package.json-only -> node."""
    from cgx.session.tasks.bootstrap_env import _detect_project_type
    (tmp_path / "package.json").write_text("{}", encoding="utf-8")
    assert _detect_project_type(tmp_path) == "node"
    (tmp_path / "requirements.txt").write_text("flask\n", encoding="utf-8")
    assert _detect_project_type(tmp_path) == "python"


def test_pin_transitive_caps_werkzeug_for_flask_22(tmp_path):
    """B: Flask 2.2.x without a Werkzeug pin gets ``werkzeug<2.3`` appended."""
    from cgx.session.tasks.bootstrap_env import _pin_transitive_constraints
    req = tmp_path / "requirements.txt"
    req.write_text("flask==2.2.2\ngunicorn==20.1.0\n", encoding="utf-8")
    added = _pin_transitive_constraints(tmp_path)
    assert added == ["werkzeug<2.3"]
    text = req.read_text(encoding="utf-8")
    assert "werkzeug<2.3" in text
    # Idempotent: a second pass sees the cap already present and no-ops.
    assert _pin_transitive_constraints(tmp_path) == []


def test_pin_transitive_noop_when_werkzeug_already_pinned(tmp_path):
    """B: an explicit Werkzeug constraint is respected (no double-cap)."""
    from cgx.session.tasks.bootstrap_env import _pin_transitive_constraints
    req = tmp_path / "requirements.txt"
    req.write_text("flask==2.2.2\nWerkzeug==2.2.3\n", encoding="utf-8")
    assert _pin_transitive_constraints(tmp_path) == []


def test_pin_transitive_noop_for_flask_3x(tmp_path):
    """B: Flask 3.x ships a compatible Werkzeug -- no cap is injected."""
    from cgx.session.tasks.bootstrap_env import _pin_transitive_constraints
    req = tmp_path / "requirements.txt"
    req.write_text("flask==3.0.0\n", encoding="utf-8")
    assert _pin_transitive_constraints(tmp_path) == []


def test_pin_transitive_noop_when_no_requirements(tmp_path):
    """B: a project without requirements.txt is a graceful no-op."""
    from cgx.session.tasks.bootstrap_env import _pin_transitive_constraints
    assert _pin_transitive_constraints(tmp_path) == []


def test_bootstrap_env_node_runs_npm_install(tmp_path, store, monkeypatch):
    """package.json-only -> project_type=node, bounded npm install runs."""
    from cgx.session.tasks import bootstrap_env as bs
    (tmp_path / "package.json").write_text(
        '{"name": "app", "scripts": {"build": "vite build"}}',
        encoding="utf-8")
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)

    calls: dict = {}

    def _fake_run(cmd, **kw):
        calls["cmd"] = list(cmd)
        (tmp_path / "node_modules").mkdir()

        class _P:
            returncode = 0
            stdout = "added 1 package"
            stderr = ""

        return _P()

    monkeypatch.setattr(bs.shutil, "which", lambda name: "/usr/bin/npm")
    monkeypatch.setattr(bs.subprocess, "run", _fake_run)

    t = TaskNode.new(session.session_id, TaskKind.BOOTSTRAP_ENV, "boot",
                     inputs={"applied_files": ["src/App.jsx"],
                             "mode": SessionMode.GREENFIELD.value})
    result = bs.run_bootstrap_env(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.failure is None
    assert result.outputs["project_type"] == "node"
    assert result.outputs["outcome"] == "succeeded"
    assert calls["cmd"][:2] == ["npm", "install"]
    content = result.artifact.content
    assert content["project_type"] == "node"
    # No venv/python for a node stack -> Python-only gates skip cleanly.
    assert content["python_exe"] is None
    assert content["venv_path"] is None


def test_bootstrap_env_node_npm_missing_is_skipped(tmp_path, store, monkeypatch):
    """No npm binary -> skipped, non-fatal (VERIFY still gives the signal)."""
    from cgx.session.tasks import bootstrap_env as bs
    (tmp_path / "package.json").write_text('{"name": "app"}', encoding="utf-8")
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)

    def _no_run(*a, **k):
        raise AssertionError("npm install must not run without npm present")

    monkeypatch.setattr(bs.shutil, "which", lambda name: None)
    monkeypatch.setattr(bs.subprocess, "run", _no_run)

    t = TaskNode.new(session.session_id, TaskKind.BOOTSTRAP_ENV, "boot",
                     inputs={"mode": SessionMode.GREENFIELD.value})
    result = bs.run_bootstrap_env(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.failure is None
    assert result.outputs["project_type"] == "node"
    assert result.outputs["outcome"] == "skipped"
    assert result.artifact.content["note"] == "npm not installed"


def test_bootstrap_env_node_install_failure_is_nonfatal(
        tmp_path, store, monkeypatch):
    """An offline npm install (rc!=0, no node_modules) -> skipped, non-fatal."""
    from cgx.session.tasks import bootstrap_env as bs
    (tmp_path / "package.json").write_text('{"name": "app"}', encoding="utf-8")
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)

    class _P:
        returncode = 1
        stdout = ""
        stderr = "npm ERR! ENOTFOUND registry.npmjs.org"

    monkeypatch.setattr(bs.shutil, "which", lambda name: "/usr/bin/npm")
    monkeypatch.setattr(bs.subprocess, "run", lambda *a, **k: _P())

    t = TaskNode.new(session.session_id, TaskKind.BOOTSTRAP_ENV, "boot",
                     inputs={"mode": SessionMode.GREENFIELD.value})
    result = bs.run_bootstrap_env(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.failure is None
    assert result.outputs["outcome"] == "skipped"
    assert "rc=1" in result.artifact.content["note"]
    assert "ENOTFOUND" in result.artifact.content["pip_log_tail"]


def test_bootstrap_env_provisions_venv_and_records_preflight(
        tmp_path, store, monkeypatch):
    """Happy path: requirements.txt present -> venv ready, preflight clean."""
    from cgx.session.tasks.bootstrap_env import run_bootstrap_env
    (tmp_path / "requirements.txt").write_text("flask\n", encoding="utf-8")
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.write_text("")
    venv_python.chmod(0o755)

    apply_art = Artifact.new(
        "sess_x", "task_apply", ArtifactKind.APPLIED_CHANGES,
        {"applied_files": ["app.py"], "failed_files": []})
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    apply_art.session_id = session.session_id
    store.save_artifact(apply_art)

    monkeypatch.setattr("cgx.codegen.test_runner.ensure_project_venv",
                        lambda root, timeout=300.0: str(venv_python))
    monkeypatch.setattr(
        "cgx.codegen.env_manager.preflight_install",
        lambda files, root, python=None: (["requests"], {"requests": True}))
    captured: dict = {}
    monkeypatch.setattr(
        "cgx.codegen.env_manager.update_requirements",
        lambda root, pkgs: captured.setdefault("pkgs", list(pkgs)))
    monkeypatch.setattr(
        "cgx.session.tasks.bootstrap_env._capture_pip_freeze",
        lambda exe, timeout=30.0: (
            [{"name": "flask", "version": "2.1.2"},
             {"name": "requests", "version": "2.32.3"}],
            "Flask==2.1.2\nrequests==2.32.3\n"))

    t = TaskNode.new(session.session_id, TaskKind.BOOTSTRAP_ENV, "boot",
                     inputs={"apply_artifact_id": apply_art.artifact_id,
                             "mode": SessionMode.GREENFIELD.value})
    result = run_bootstrap_env(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.failure is None
    assert result.outputs["outcome"] == "succeeded"
    assert result.outputs["venv_path"] == str(tmp_path / ".venv")
    assert result.outputs["installed_count"] == 1
    assert result.outputs["failed_count"] == 0
    assert captured["pkgs"] == ["requests"]
    content = result.artifact.content
    assert content["installed_packages"] == ["requests"]
    assert content["failed_installs"] == []
    assert "requirements.txt" in content["installed_from"]
    assert content["applied_files"] == ["app.py"]
    assert content["resolved_packages"] == [
        {"name": "flask", "version": "2.1.2"},
        {"name": "requests", "version": "2.32.3"}]
    assert "Flask==2.1.2" in content["pip_freeze_text"]


def test_bootstrap_env_surfaces_install_failures(tmp_path, store, monkeypatch):
    """A failed preflight install -> outcome=failed, executor reports failure."""
    from cgx.session.tasks.bootstrap_env import run_bootstrap_env
    (tmp_path / "requirements.txt").write_text("flask\n", encoding="utf-8")
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.write_text("")
    venv_python.chmod(0o755)

    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    monkeypatch.setattr("cgx.codegen.test_runner.ensure_project_venv",
                        lambda root, timeout=300.0: str(venv_python))
    monkeypatch.setattr(
        "cgx.codegen.env_manager.preflight_install",
        lambda files, root, python=None: (
            ["nonexistent-xyz"], {"nonexistent-xyz": False}))
    monkeypatch.setattr(
        "cgx.codegen.env_manager.update_requirements",
        lambda root, pkgs: None)
    monkeypatch.setattr(
        "cgx.session.tasks.bootstrap_env._capture_pip_freeze",
        lambda exe, timeout=30.0: ([], ""))

    t = TaskNode.new(session.session_id, TaskKind.BOOTSTRAP_ENV, "boot",
                     inputs={"applied_files": ["app.py"],
                             "mode": SessionMode.GREENFIELD.value})
    result = run_bootstrap_env(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.outputs["outcome"] == "failed"
    assert result.outputs["failed_count"] == 1
    assert result.failure and "nonexistent-xyz" in result.failure
    assert result.artifact.content["failed_installs"] == ["nonexistent-xyz"]


def test_bootstrap_env_fails_when_runtime_import_missing(
        tmp_path, store, monkeypatch):
    """Clean install but an unimportable dep -> outcome=failed (honesty gate).

    Mirrors the Test.7 regression: flask is "declared" so preflight has
    nothing to install, yet it never landed in the venv (a malformed
    requirements line aborted the batch install). The scaffold imports
    flask, so BOOTSTRAP must not report success.
    """
    from cgx.session.tasks.bootstrap_env import run_bootstrap_env
    (tmp_path / "requirements.txt").write_text(
        "flask==2.3.2\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("import flask\n", encoding="utf-8")
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.write_text("")
    venv_python.chmod(0o755)

    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    monkeypatch.setattr("cgx.codegen.test_runner.ensure_project_venv",
                        lambda root, timeout=300.0: str(venv_python))
    monkeypatch.setattr(
        "cgx.codegen.env_manager.preflight_install",
        lambda files, root, python=None: ([], {}))
    monkeypatch.setattr(
        "cgx.codegen.env_manager.update_requirements",
        lambda root, pkgs: None)
    monkeypatch.setattr(
        "cgx.session.tasks.bootstrap_env._capture_pip_freeze",
        lambda exe, timeout=30.0: ([], ""))
    # The scaffold's flask import does not resolve in the venv.
    monkeypatch.setattr(
        "cgx.codegen.env_manager._probe_importable",
        lambda names, python=None: set())

    t = TaskNode.new(session.session_id, TaskKind.BOOTSTRAP_ENV, "boot",
                     inputs={"applied_files": ["app.py"],
                             "mode": SessionMode.GREENFIELD.value})
    result = run_bootstrap_env(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.outputs["outcome"] == "failed"
    assert result.outputs["missing_import_count"] == 1
    assert result.failure and "flask" in result.failure
    assert result.artifact.content["missing_imports"] == ["flask"]


def test_bootstrap_env_no_venv_when_host_interpreter_returned(
        tmp_path, store, monkeypatch):
    """ensure_project_venv falling back to host sys.executable -> no_venv."""
    import sys
    from cgx.session.tasks.bootstrap_env import run_bootstrap_env
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n",
                                             encoding="utf-8")
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    monkeypatch.setattr("cgx.codegen.test_runner.ensure_project_venv",
                        lambda root, timeout=300.0: sys.executable)
    monkeypatch.setattr(
        "cgx.codegen.env_manager.preflight_install",
        lambda files, root, python=None: ([], {}))
    monkeypatch.setattr(
        "cgx.codegen.env_manager.update_requirements",
        lambda root, pkgs: None)

    t = TaskNode.new(session.session_id, TaskKind.BOOTSTRAP_ENV, "boot",
                     inputs={"applied_files": [],
                             "mode": SessionMode.GREENFIELD.value})
    result = run_bootstrap_env(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.failure is None
    assert result.outputs["outcome"] == "no_venv"
    assert result.artifact.content["venv_path"] is None


def test_bootstrap_env_resolves_applied_files_from_scaffold_artifact(
        tmp_path, store, monkeypatch):
    """When only scaffold_artifact_id is given, generated[].file is used."""
    from cgx.session.tasks.bootstrap_env import run_bootstrap_env
    (tmp_path / "requirements.txt").write_text("", encoding="utf-8")
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.write_text("")
    venv_python.chmod(0o755)

    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    scaffold_art = Artifact.new(
        session.session_id, "task_sc", ArtifactKind.SCAFFOLD_PATCHES,
        {"generated": [{"file": "app.py", "bytes": 12, "layer": "app"},
                       {"file": "tests/test_app.py", "bytes": 30,
                        "layer": "tests"}],
         "failed": [], "diffs": []})
    store.save_artifact(scaffold_art)

    seen: dict = {}

    def fake_preflight(files, root, python=None):
        seen["files"] = list(files)
        return ([], {})

    monkeypatch.setattr("cgx.codegen.test_runner.ensure_project_venv",
                        lambda root, timeout=300.0: str(venv_python))
    monkeypatch.setattr("cgx.codegen.env_manager.preflight_install",
                        fake_preflight)
    monkeypatch.setattr("cgx.codegen.env_manager.update_requirements",
                        lambda root, pkgs: None)
    monkeypatch.setattr(
        "cgx.session.tasks.bootstrap_env._capture_pip_freeze",
        lambda exe, timeout=30.0: ([], ""))

    t = TaskNode.new(
        session.session_id, TaskKind.BOOTSTRAP_ENV, "boot",
        inputs={"scaffold_artifact_id": scaffold_art.artifact_id,
                "mode": SessionMode.GREENFIELD.value})
    result = run_bootstrap_env(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.outputs["outcome"] == "succeeded"
    # Files are passed through as absolute paths under tmp_path.
    assert seen["files"] == [str(tmp_path / "app.py"),
                             str(tmp_path / "tests/test_app.py")]
    assert result.artifact.content["applied_files"] == \
        ["app.py", "tests/test_app.py"]


def test_bootstrap_env_captures_pip_freeze_on_succeeded(
        tmp_path, store, monkeypatch):
    """resolved_packages + pip_freeze_text are populated from the venv."""
    from cgx.session.tasks.bootstrap_env import run_bootstrap_env
    (tmp_path / "requirements.txt").write_text("flask\n", encoding="utf-8")
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.write_text("")
    venv_python.chmod(0o755)

    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    monkeypatch.setattr("cgx.codegen.test_runner.ensure_project_venv",
                        lambda root, timeout=300.0: str(venv_python))
    monkeypatch.setattr("cgx.codegen.env_manager.preflight_install",
                        lambda files, root, python=None: ([], {}))
    monkeypatch.setattr("cgx.codegen.env_manager.update_requirements",
                        lambda root, pkgs: None)

    seen_exe: dict = {}

    def fake_freeze(exe, timeout=30.0):
        seen_exe["exe"] = exe
        return ([{"name": "flask", "version": "2.1.2"},
                 {"name": "werkzeug", "version": "3.1.8"}],
                "Flask==2.1.2\nWerkzeug==3.1.8\n")

    monkeypatch.setattr(
        "cgx.session.tasks.bootstrap_env._capture_pip_freeze", fake_freeze)

    t = TaskNode.new(session.session_id, TaskKind.BOOTSTRAP_ENV, "boot",
                     inputs={"applied_files": ["app.py"],
                             "mode": SessionMode.GREENFIELD.value})
    result = run_bootstrap_env(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.outputs["outcome"] == "succeeded"
    assert seen_exe["exe"] == str(venv_python)
    content = result.artifact.content
    assert content["resolved_packages"] == [
        {"name": "flask", "version": "2.1.2"},
        {"name": "werkzeug", "version": "3.1.8"}]
    assert content["pip_freeze_text"] == \
        "Flask==2.1.2\nWerkzeug==3.1.8\n"


def test_bootstrap_env_pip_freeze_failure_is_graceful(
        tmp_path, store, monkeypatch):
    """A raising freeze does not fail BOOTSTRAP; fields default to empty."""
    from cgx.session.tasks.bootstrap_env import run_bootstrap_env
    (tmp_path / "requirements.txt").write_text("flask\n", encoding="utf-8")
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.write_text("")
    venv_python.chmod(0o755)

    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    monkeypatch.setattr("cgx.codegen.test_runner.ensure_project_venv",
                        lambda root, timeout=300.0: str(venv_python))
    monkeypatch.setattr("cgx.codegen.env_manager.preflight_install",
                        lambda files, root, python=None: ([], {}))
    monkeypatch.setattr("cgx.codegen.env_manager.update_requirements",
                        lambda root, pkgs: None)
    # Simulate the *internal* defensive behaviour: subprocess raised,
    # the helper swallowed it and returned empty.
    monkeypatch.setattr(
        "cgx.session.tasks.bootstrap_env._capture_pip_freeze",
        lambda exe, timeout=30.0: ([], ""))

    t = TaskNode.new(session.session_id, TaskKind.BOOTSTRAP_ENV, "boot",
                     inputs={"applied_files": ["app.py"],
                             "mode": SessionMode.GREENFIELD.value})
    result = run_bootstrap_env(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.failure is None
    assert result.outputs["outcome"] == "succeeded"
    assert result.artifact.content["resolved_packages"] == []
    assert result.artifact.content["pip_freeze_text"] == ""


def test_bootstrap_env_skips_pip_freeze_on_no_venv(
        tmp_path, store, monkeypatch):
    """no_venv outcome must not freeze the host interpreter."""
    import sys
    from cgx.session.tasks.bootstrap_env import run_bootstrap_env
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n",
                                             encoding="utf-8")
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    monkeypatch.setattr("cgx.codegen.test_runner.ensure_project_venv",
                        lambda root, timeout=300.0: sys.executable)
    monkeypatch.setattr("cgx.codegen.env_manager.preflight_install",
                        lambda files, root, python=None: ([], {}))
    monkeypatch.setattr("cgx.codegen.env_manager.update_requirements",
                        lambda root, pkgs: None)

    called: dict = {"n": 0}

    def boom(exe, timeout=30.0):
        called["n"] += 1
        raise AssertionError("pip freeze must not run when no_venv")

    monkeypatch.setattr(
        "cgx.session.tasks.bootstrap_env._capture_pip_freeze", boom)

    t = TaskNode.new(session.session_id, TaskKind.BOOTSTRAP_ENV, "boot",
                     inputs={"applied_files": [],
                             "mode": SessionMode.GREENFIELD.value})
    result = run_bootstrap_env(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.outputs["outcome"] == "no_venv"
    assert called["n"] == 0
    assert result.artifact.content["resolved_packages"] == []
    assert result.artifact.content["pip_freeze_text"] == ""


def test_pip_freeze_parser_handles_canonical_and_edge_lines():
    """Parser keeps name==version, drops -e/url/comment, PEP 503-normalises."""
    from cgx.session.tasks.bootstrap_env import _parse_pip_freeze
    raw = (
        "# pip freeze --all\n"
        "Flask==2.1.2\n"
        "Werkzeug==3.1.8\n"
        "Some_Package==1.0.0\n"
        "FOO.Bar==9.9\n"
        "-e git+https://example.com/x.git@deadbeef#egg=x\n"
        "weirdpkg @ file:///tmp/wheels/weirdpkg.whl\n"
        "spaces==1.0 ; python_version >= '3.10'\n"
        "\n"
    )
    parsed = _parse_pip_freeze(raw)
    assert parsed == [
        {"name": "flask",        "version": "2.1.2"},
        {"name": "werkzeug",     "version": "3.1.8"},
        {"name": "some-package", "version": "1.0.0"},
        {"name": "foo-bar",      "version": "9.9"},
        {"name": "spaces",       "version": "1.0"},
    ]


def test_capture_pip_freeze_swallows_subprocess_error(monkeypatch):
    """A raising subprocess.run -> ([], '') without propagating."""
    import cgx.session.tasks.bootstrap_env as be

    def boom(*args, **kwargs):
        raise OSError("ENOENT")

    monkeypatch.setattr(be.subprocess, "run", boom)
    parsed, raw = be._capture_pip_freeze("/no/such/python")
    assert parsed == []
    assert raw == ""


def test_capture_pip_freeze_swallows_nonzero_returncode(monkeypatch):
    """rc != 0 -> ([], '') without propagating."""
    import cgx.session.tasks.bootstrap_env as be

    class _Proc:
        returncode = 1
        stdout = b"oops"
        stderr = b""

    monkeypatch.setattr(be.subprocess, "run",
                        lambda *a, **k: _Proc())
    parsed, raw = be._capture_pip_freeze("/some/python")
    assert parsed == []
    assert raw == ""


# --------------------- SMOKE executor unit tests ---------------------

def test_smoke_requires_project_root():
    from cgx.session.tasks.smoke import run_smoke
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    t = TaskNode.new(session.session_id, TaskKind.SMOKE, "smoke", inputs={})
    result = run_smoke(t, ExecutorDeps(store=object()))
    assert result.failure and "project_root" in result.failure


def test_smoke_skipped_when_no_python_exe(tmp_path, store):
    """No build_artifact_id + no explicit python_exe -> outcome=skipped."""
    from cgx.session.tasks.smoke import run_smoke
    (tmp_path / "app.py").write_text("import os\n", encoding="utf-8")
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    t = TaskNode.new(session.session_id, TaskKind.SMOKE, "smoke",
                     inputs={"applied_files": ["app.py"]})
    result = run_smoke(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.failure is None
    assert result.outputs["outcome"] == "skipped"
    assert result.artifact.kind is ArtifactKind.SMOKE_REPORT
    assert result.artifact.content["modules"] == []


def test_smoke_skipped_when_only_stdlib_or_first_party(tmp_path, store):
    """All imports are stdlib or first-party -> nothing to probe."""
    from cgx.session.tasks.smoke import run_smoke
    (tmp_path / "app.py").write_text(
        "import os\nimport sys\nfrom myapp import helpers\n",
        encoding="utf-8")
    (tmp_path / "myapp").mkdir()
    (tmp_path / "myapp" / "__init__.py").write_text("", encoding="utf-8")
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    t = TaskNode.new(session.session_id, TaskKind.SMOKE, "smoke",
                     inputs={"applied_files": ["app.py"],
                             "python_exe": "/usr/bin/python3"})
    result = run_smoke(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.outputs["outcome"] == "skipped"
    assert result.outputs["tested_count"] == 0


def test_smoke_collects_only_third_party_imports(tmp_path):
    """The static collector filters stdlib + first-party + relative imports."""
    from cgx.session.tasks.smoke import _collect_third_party_imports
    (tmp_path / "app.py").write_text(
        "import os\n"
        "import sys\n"
        "import flask\n"
        "import requests.adapters\n"
        "from . import sibling\n"
        "from myapp import helpers\n"
        "from werkzeug.urls import url_quote\n",
        encoding="utf-8")
    (tmp_path / "myapp").mkdir()
    (tmp_path / "myapp" / "__init__.py").write_text("", encoding="utf-8")
    pkgs = _collect_third_party_imports(tmp_path, ["app.py"])
    assert "flask" in pkgs
    assert "requests" in pkgs
    assert "werkzeug" in pkgs
    assert "os" not in pkgs
    assert "sys" not in pkgs
    assert "myapp" not in pkgs


def test_smoke_runs_probes_and_records_failure(tmp_path, store, monkeypatch):
    """Each candidate is probed; a failing import flips outcome=failed."""
    from cgx.session.tasks import smoke as smoke_mod
    from cgx.session.tasks.smoke import run_smoke
    (tmp_path / "app.py").write_text(
        "import flask\nimport werkzeug\n", encoding="utf-8")
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    build_art = Artifact.new(
        session_id=session.session_id, produced_by_task_id="t_build",
        kind=ArtifactKind.BUILD_REPORT,
        content={"python_exe": "/fake/.venv/bin/python"})
    store.save_artifact(build_art)

    seen: list = []

    def fake_probe(python_exe, pkg, timeout):
        seen.append(pkg)
        if pkg == "werkzeug":
            return False, ("Traceback ...\n"
                           "ImportError: cannot import name 'url_quote' "
                           "from 'werkzeug.urls'")
        return True, ""

    monkeypatch.setattr(smoke_mod, "_probe_import", fake_probe)

    t = TaskNode.new(session.session_id, TaskKind.SMOKE, "smoke",
                     inputs={"applied_files": ["app.py"],
                             "build_artifact_id": build_art.artifact_id})
    result = run_smoke(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.failure is None
    assert result.outputs["outcome"] == "failed"
    assert result.outputs["failed_count"] == 1
    assert "werkzeug" in result.outputs["failure_signature"]
    modules = result.artifact.content["modules"]
    werkzeug_row = next(m for m in modules if m["name"] == "werkzeug")
    assert werkzeug_row["ok"] is False
    assert "url_quote" in werkzeug_row["stderr_tail"]
    assert set(seen) == {"flask", "werkzeug"}


def _node_smoke_task(store, tmp_path, *, build_script=True,
                     node_modules=True):
    """Build a package.json-only SMOKE task fixture for the JS build-smoke."""
    scripts = '{"build": "vite build"}' if build_script else "{}"
    (tmp_path / "package.json").write_text(
        '{"name": "app", "scripts": ' + scripts + "}", encoding="utf-8")
    if node_modules:
        (tmp_path / "node_modules").mkdir()
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    return TaskNode.new(session.session_id, TaskKind.SMOKE, "smoke",
                        inputs={"applied_files": ["src/App.jsx"],
                                "mode": SessionMode.GREENFIELD.value})


def test_smoke_node_build_failure_is_failed(tmp_path, store, monkeypatch):
    """A non-building JS app fails the SMOKE build-smoke honestly."""
    from cgx.session.tasks import smoke as smoke_mod
    t = _node_smoke_task(store, tmp_path)

    class _P:
        returncode = 1
        stdout = ""
        stderr = "src/App.jsx: TS2345 build error"

    monkeypatch.setattr(smoke_mod.shutil, "which", lambda name: "/usr/bin/npm")
    monkeypatch.setattr(smoke_mod.subprocess, "run", lambda *a, **k: _P())

    result = smoke_mod.run_smoke(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.failure is None
    assert result.outputs["outcome"] == "failed"
    bs = result.artifact.content["build_smoke"]
    assert bs["ok"] is False
    assert "TS2345" in bs["stderr_tail"]
    assert "npm run build" in result.outputs["failure_signature"]


def test_smoke_node_build_pass_is_passed(tmp_path, store, monkeypatch):
    """A clean JS build -> outcome=passed (chains on to VERIFY)."""
    from cgx.session.tasks import smoke as smoke_mod
    t = _node_smoke_task(store, tmp_path)

    class _P:
        returncode = 0
        stdout = "built"
        stderr = ""

    monkeypatch.setattr(smoke_mod.shutil, "which", lambda name: "/usr/bin/npm")
    monkeypatch.setattr(smoke_mod.subprocess, "run", lambda *a, **k: _P())

    result = smoke_mod.run_smoke(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.outputs["outcome"] == "passed"
    assert result.artifact.content["build_smoke"]["ok"] is True


def test_smoke_node_skips_without_node_modules(tmp_path, store, monkeypatch):
    """No provisioned node_modules -> build-smoke skips (no false failure)."""
    from cgx.session.tasks import smoke as smoke_mod
    t = _node_smoke_task(store, tmp_path, node_modules=False)

    def _no_run(*a, **k):
        raise AssertionError("build-smoke must not run without node_modules")

    monkeypatch.setattr(smoke_mod.shutil, "which", lambda name: "/usr/bin/npm")
    monkeypatch.setattr(smoke_mod.subprocess, "run", _no_run)

    result = smoke_mod.run_smoke(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.outputs["outcome"] == "skipped"
    assert result.artifact.content["build_smoke"] is None


def test_smoke_resolves_applied_files_from_apply_artifact(tmp_path, store):
    """When applied_files is not in inputs, fall back to the apply artifact."""
    from cgx.session.tasks.smoke import _resolve_applied_files
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    apply_art = Artifact.new(
        session_id=session.session_id, produced_by_task_id="t_apply",
        kind=ArtifactKind.APPLIED_CHANGES,
        content={"applied_files": ["src/a.py", "src/b.py"]})
    store.save_artifact(apply_art)
    t = TaskNode.new(session.session_id, TaskKind.SMOKE, "smoke",
                     inputs={"apply_artifact_id": apply_art.artifact_id})
    files = _resolve_applied_files(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert files == ["src/a.py", "src/b.py"]


# --------------------- API_CHECK executor unit tests ---------------------

def test_api_check_requires_project_root():
    from cgx.session.tasks.api_check import run_api_check
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    t = TaskNode.new(session.session_id, TaskKind.API_CHECK, "api",
                     inputs={})
    result = run_api_check(t, ExecutorDeps(store=object()))
    assert result.failure and "project_root" in result.failure


def test_api_check_skipped_when_no_python_exe(tmp_path, store):
    """No build_artifact_id + no explicit python_exe -> outcome=skipped."""
    from cgx.session.tasks.api_check import run_api_check
    (tmp_path / "app.py").write_text(
        "from werkzeug.urls import url_quote\n", encoding="utf-8")
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    t = TaskNode.new(session.session_id, TaskKind.API_CHECK, "api",
                     inputs={"applied_files": ["app.py"]})
    result = run_api_check(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.failure is None
    assert result.outputs["outcome"] == "skipped"
    assert result.artifact.kind is ArtifactKind.API_CHECK_REPORT


def test_api_check_collects_importfrom_and_attribute_refs(tmp_path):
    """Static collector grabs ImportFrom names + alias-qualified attrs."""
    from cgx.session.tasks.api_check import _collect_third_party_references
    (tmp_path / "app.py").write_text(
        "import os\n"
        "import numpy as np\n"
        "import flask\n"
        "from werkzeug.urls import url_quote, unquote\n"
        "from . import sibling\n"
        "from myapp import helpers\n"
        "x = np.zeros(3)\n"
        "y = flask.Flask(__name__)\n"
        "z = os.path.join('a', 'b')\n",
        encoding="utf-8")
    (tmp_path / "myapp").mkdir()
    (tmp_path / "myapp" / "__init__.py").write_text("", encoding="utf-8")
    order, refs = _collect_third_party_references(tmp_path, ["app.py"])
    assert ("werkzeug.urls", "url_quote") in order
    assert ("werkzeug.urls", "unquote") in order
    assert ("numpy", "zeros") in order
    assert ("flask", "Flask") in order
    # stdlib / first-party / relative imports must NOT appear.
    assert not any(m == "os" for m, _ in order)
    assert not any(m == "myapp" for m, _ in order)
    # Reference tracking records file + lineno.
    werkzeug_refs = refs[("werkzeug.urls", "url_quote")]
    assert werkzeug_refs and werkzeug_refs[0]["file"] == "app.py"


def test_api_check_runs_probes_and_records_failure(
        tmp_path, store, monkeypatch):
    """Each (module, name) is probed; a missing name flips outcome=failed."""
    from cgx.session.tasks import api_check as api_mod
    from cgx.session.tasks.api_check import run_api_check
    (tmp_path / "app.py").write_text(
        "from werkzeug.urls import url_quote\n"
        "import flask\n"
        "x = flask.Flask(__name__)\n",
        encoding="utf-8")
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    build_art = Artifact.new(
        session_id=session.session_id, produced_by_task_id="t_build",
        kind=ArtifactKind.BUILD_REPORT,
        content={"python_exe": "/fake/.venv/bin/python"})
    store.save_artifact(build_art)

    def fake_probe(python_exe, specs, timeout):
        rows = []
        for module, name in specs:
            if module == "werkzeug.urls" and name == "url_quote":
                rows.append({"module": module, "name": name, "ok": False,
                             "error": ("AttributeError: module "
                                       "'werkzeug.urls' has no attribute "
                                       "'url_quote'")})
            else:
                rows.append({"module": module, "name": name, "ok": True,
                             "error": ""})
        return rows, None

    monkeypatch.setattr(api_mod, "_probe_references", fake_probe)

    t = TaskNode.new(session.session_id, TaskKind.API_CHECK, "api",
                     inputs={"applied_files": ["app.py"],
                             "build_artifact_id": build_art.artifact_id})
    result = run_api_check(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.failure is None
    assert result.outputs["outcome"] == "failed"
    assert result.outputs["failed_count"] == 1
    assert "werkzeug.urls.url_quote" in result.outputs["failure_signature"]
    rows = result.artifact.content["references"]
    bad = next(r for r in rows
               if r["module"] == "werkzeug.urls" and r["name"] == "url_quote")
    assert bad["ok"] is False
    assert bad["references"] and bad["references"][0]["file"] == "app.py"


def test_api_check_splits_missing_dependency_from_hallucination(
        tmp_path, store, monkeypatch):
    """ModuleNotFoundError -> missing_dependency; AttributeError -> hallucination."""
    from cgx.session.tasks import api_check as api_mod
    from cgx.session.tasks.api_check import run_api_check
    (tmp_path / "app.py").write_text(
        "from flask import Flask\n"
        "from cerberus import Schema\n",
        encoding="utf-8")
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    build_art = Artifact.new(
        session_id=session.session_id, produced_by_task_id="t_build",
        kind=ArtifactKind.BUILD_REPORT,
        content={"python_exe": "/fake/.venv/bin/python"})
    store.save_artifact(build_art)

    def fake_probe(python_exe, specs, timeout):
        rows = []
        for module, name in specs:
            if module == "flask":
                rows.append({
                    "module": module, "name": name, "ok": False,
                    "error": "ModuleNotFoundError: No module named 'flask'"})
            elif module == "cerberus":
                rows.append({
                    "module": module, "name": name, "ok": False,
                    "error": ("AttributeError: module 'cerberus' has no "
                              "attribute 'Schema'")})
            else:
                rows.append({"module": module, "name": name, "ok": True,
                             "error": ""})
        return rows, None

    monkeypatch.setattr(api_mod, "_probe_references", fake_probe)
    t = TaskNode.new(session.session_id, TaskKind.API_CHECK, "api",
                     inputs={"applied_files": ["app.py"],
                             "build_artifact_id": build_art.artifact_id})
    result = run_api_check(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.outputs["outcome"] == "failed"
    assert result.outputs["missing_module_count"] == 1
    assert result.outputs["hallucinated_count"] == 1
    content = result.artifact.content
    assert content["missing_modules"] == ["flask"]
    assert content["hallucinated_references"] == [
        {"module": "cerberus", "name": "Schema"}]
    cats = {(r["module"], r["name"]): r.get("category")
            for r in content["references"] if not r["ok"]}
    assert cats[("flask", "Flask")] == "missing_dependency"
    assert cats[("cerberus", "Schema")] == "api_check_failure"


def test_api_check_probe_error_skips_outcome(tmp_path, store, monkeypatch):
    """A probe-level failure (e.g. missing python_exe) -> outcome=skipped."""
    from cgx.session.tasks import api_check as api_mod
    from cgx.session.tasks.api_check import run_api_check
    (tmp_path / "app.py").write_text(
        "from werkzeug.urls import url_quote\n", encoding="utf-8")
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    build_art = Artifact.new(
        session_id=session.session_id, produced_by_task_id="t_build",
        kind=ArtifactKind.BUILD_REPORT,
        content={"python_exe": "/fake/.venv/bin/python"})
    store.save_artifact(build_art)
    monkeypatch.setattr(
        api_mod, "_probe_references",
        lambda *a, **kw: ([], "FileNotFoundError: /fake/.venv/bin/python"))
    t = TaskNode.new(session.session_id, TaskKind.API_CHECK, "api",
                     inputs={"applied_files": ["app.py"],
                             "build_artifact_id": build_art.artifact_id})
    result = run_api_check(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.outputs["outcome"] == "skipped"
    assert result.artifact.content["probe_error"].startswith(
        "FileNotFoundError")


def test_repair_handles_api_check_report(tmp_path, store):
    """REPAIR(api_check_artifact_id) emits a non-applicable plan + rationale."""
    from cgx.session.tasks.repair import run_repair
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    report = Artifact.new(
        session_id=session.session_id, produced_by_task_id="t_api",
        kind=ArtifactKind.API_CHECK_REPORT,
        content={
            "outcome": "failed",
            "failed_references": [
                {"module": "werkzeug.urls", "name": "url_quote"},
            ],
            "failure_signature": "api_check|werkzeug.urls.url_quote",
        })
    store.save_artifact(report)
    t = TaskNode.new(
        session.session_id, TaskKind.REPAIR, "repair",
        inputs={"api_check_artifact_id": report.artifact_id,
                "repair_attempt": 1,
                "mode": SessionMode.GREENFIELD.value})
    result = run_repair(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.failure is None
    assert result.outputs["can_apply"] is False
    assert result.outputs["classification"] == "api_check_failure"
    assert ("api_check|werkzeug.urls.url_quote"
            == result.outputs["failure_signature"])
    plan = result.artifact
    assert plan.kind is ArtifactKind.REPAIR_PLAN
    assert "werkzeug.urls.url_quote" in plan.content["rationale"]
    # The rationale is folded into the regenerate goal as a constraint, so
    # it must be actionable (tell the model to drop/replace the symbol) and
    # must NOT leak internal control-flow language like ASK_USER.
    assert "REMOVE" in plan.content["rationale"]
    assert "ASK_USER" not in plan.content["rationale"]
    assert plan.content["diffs"] == []


def test_repair_api_check_missing_dependency_installs(tmp_path, store):
    """API_CHECK missing_modules -> strategy=install_deps, not regenerate."""
    from cgx.session.tasks.repair import run_repair
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    report = Artifact.new(
        session_id=session.session_id, produced_by_task_id="t_api",
        kind=ArtifactKind.API_CHECK_REPORT,
        content={
            "outcome": "failed",
            "failed_references": [
                {"module": "flask", "name": "Flask"},
                {"module": "flask_cors", "name": "CORS"},
            ],
            "missing_modules": ["flask", "flask_cors"],
            "hallucinated_references": [],
            "failure_signature": "api_check|flask.Flask,flask_cors.CORS",
        })
    store.save_artifact(report)
    t = TaskNode.new(
        session.session_id, TaskKind.REPAIR, "repair",
        inputs={"api_check_artifact_id": report.artifact_id,
                "repair_attempt": 1,
                "mode": SessionMode.GREENFIELD.value})
    result = run_repair(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.failure is None
    assert result.outputs["classification"] == "missing_dependency"
    assert result.outputs["strategy"] == "install_deps"
    assert result.outputs["can_apply"] is False
    assert result.outputs["missing_modules"] == ["flask", "flask_cors"]
    plan = result.artifact
    assert plan.kind is ArtifactKind.REPAIR_PLAN
    assert plan.content["strategy"] == "install_deps"
    assert plan.content["missing_modules"] == ["flask", "flask_cors"]
    assert "flask" in plan.content["rationale"]
    assert plan.content["diffs"] == []


# --------------------- runner integration (stub executors) ---------------------

def _install_stub_clarify():
    @register_executor(TaskKind.CLARIFY_REQUIREMENTS)
    def _stub(task, deps):
        artifact = Artifact.new(
            session_id=task.session_id,
            produced_by_task_id=task.task_id,
            kind=ArtifactKind.REQUIREMENTS_SHEET,
            content={"goal": task.inputs.get("goal"),
                     "questions": [
                         {"id": "q1", "prompt": "Framework?",
                          "suggested": ["Flask"]},
                         {"id": "q2", "prompt": "Storage?",
                          "suggested": ["JSON on disk"]},
                         {"id": "q3", "prompt": "Auth?",
                          "suggested": ["None"]}],
                     "source": "stub"})
        return ExecutorResult(
            outputs={"requirements_artifact_id": artifact.artifact_id,
                     "question_count": 3},
            artifact=artifact)


def _install_stub_decompose():
    @register_executor(TaskKind.DECOMPOSE)
    def _stub(task, deps):
        artifact = Artifact.new(
            session_id=task.session_id,
            produced_by_task_id=task.task_id,
            kind=ArtifactKind.WORK_PLAN,
            content={
                "prior_goal": task.inputs.get("prior_goal"),
                "composed_goal": task.inputs.get("prior_goal"),
                "answers": dict(task.inputs.get("answers") or {}),
                "plan_md": "## Plan\n- app.py",
                "layers": [{"name": "app", "files": [
                    {"path": "app.py", "description": "entry"}]}],
            })
        return ExecutorResult(
            outputs={"work_plan_artifact_id": artifact.artifact_id,
                     "file_count": 1, "layer_count": 1},
            artifact=artifact)


def _install_stub_scaffold():
    @register_executor(TaskKind.SCAFFOLD)
    def _stub(task, deps):
        artifact = Artifact.new(
            session_id=task.session_id,
            produced_by_task_id=task.task_id,
            kind=ArtifactKind.SCAFFOLD_PATCHES,
            content={
                "work_plan_artifact_id":
                    task.inputs.get("work_plan_artifact_id"),
                "prior_goal": task.inputs.get("prior_goal"),
                "diffs": [{"file": "app.py", "patch": "+++ app.py\nbody"}],
                "generated": [{"file": "app.py", "bytes": 4,
                               "layer": "app"}],
                "failed": [],
            })
        return ExecutorResult(
            outputs={"scaffold_artifact_id": artifact.artifact_id,
                     "generated_count": 1, "failed_count": 0},
            artifact=artifact)


def _install_stub_apply_greenfield():
    @register_executor(TaskKind.APPLY)
    def _stub(task, deps):
        artifact = Artifact.new(
            session_id=task.session_id,
            produced_by_task_id=task.task_id,
            kind=ArtifactKind.APPLIED_CHANGES,
            content={
                "plan_artifact_id": task.inputs.get("plan_artifact_id"),
                "source_artifact_kind": "scaffold_patches",
                "applied_files": ["app.py"],
                "failed_files": [],
                "backup_dir": "/tmp/backup-fake",
                "smoke_ok": True,
                "diffs": [{"file": "app.py", "patch": "+++ app.py\nbody"}],
            })
        return ExecutorResult(
            outputs={"apply_artifact_id": artifact.artifact_id,
                     "applied_count": 1, "failed_count": 0,
                     "backup_dir": "/tmp/backup-fake"},
            artifact=artifact)


def _install_stub_verify_greenfield():
    @register_executor(TaskKind.VERIFY)
    def _stub(task, deps):
        # Greenfield projects typically have no tests yet -- VERIFY
        # still emits a report (ran=False, no skipped_reason failure).
        artifact = Artifact.new(
            session_id=task.session_id,
            produced_by_task_id=task.task_id,
            kind=ArtifactKind.VERIFY_REPORT,
            content={
                "apply_artifact_id": task.inputs.get("apply_artifact_id"),
                "scaffold_artifact_id":
                    task.inputs.get("scaffold_artifact_id"),
                "build_artifact_id":
                    task.inputs.get("build_artifact_id"),
                "mode": task.inputs.get("mode") or "greenfield",
                "changed_files": ["app.py"],
                "ran": False, "tests_passed": False,
                "outcome": "skipped",
                "returncode": 0, "tests_selected": [],
                "stdout": "", "stderr": "",
                "skipped_reason": "no tests discovered",
            })
        return ExecutorResult(
            outputs={"verify_artifact_id": artifact.artifact_id,
                     "ran": False, "tests_passed": False,
                     "outcome": "skipped",
                     "tests_selected_count": 0},
            artifact=artifact)


def _install_stub_smoke_greenfield():
    @register_executor(TaskKind.SMOKE)
    def _stub(task, deps):
        artifact = Artifact.new(
            session_id=task.session_id,
            produced_by_task_id=task.task_id,
            kind=ArtifactKind.SMOKE_REPORT,
            content={
                "build_artifact_id": task.inputs.get("build_artifact_id"),
                "applied_files": ["app.py"],
                "modules": [],
                "outcome": "skipped",
                "failed_modules": [],
                "failure_signature": "",
            })
        return ExecutorResult(
            outputs={"smoke_artifact_id": artifact.artifact_id,
                     "outcome": "skipped",
                     "failed_count": 0,
                     "tested_count": 0,
                     "failure_signature": ""},
            artifact=artifact)


def _install_stub_api_check_greenfield():
    @register_executor(TaskKind.API_CHECK)
    def _stub(task, deps):
        artifact = Artifact.new(
            session_id=task.session_id,
            produced_by_task_id=task.task_id,
            kind=ArtifactKind.API_CHECK_REPORT,
            content={
                "build_artifact_id": task.inputs.get("build_artifact_id"),
                "applied_files": ["app.py"],
                "references": [],
                "outcome": "skipped",
                "failed_references": [],
                "failure_signature": "",
                "probe_error": None,
            })
        return ExecutorResult(
            outputs={"api_check_artifact_id": artifact.artifact_id,
                     "outcome": "skipped",
                     "failed_count": 0,
                     "checked_count": 0,
                     "failure_signature": ""},
            artifact=artifact)


def _install_stub_bootstrap_env_greenfield():
    @register_executor(TaskKind.BOOTSTRAP_ENV)
    def _stub(task, deps):
        artifact = Artifact.new(
            session_id=task.session_id,
            produced_by_task_id=task.task_id,
            kind=ArtifactKind.BUILD_REPORT,
            content={
                "apply_artifact_id": task.inputs.get("apply_artifact_id"),
                "scaffold_artifact_id":
                    task.inputs.get("scaffold_artifact_id"),
                "project_type": "python",
                "venv_path": "/tmp/proj/.venv",
                "python_exe": "/tmp/proj/.venv/bin/python",
                "installed_from": ["requirements.txt"],
                "installed_packages": [],
                "failed_installs": [],
                "outcome": "succeeded",
                "pip_log_tail": "",
                "applied_files": ["app.py"],
            })
        return ExecutorResult(
            outputs={"build_artifact_id": artifact.artifact_id,
                     "outcome": "succeeded",
                     "project_type": "python",
                     "venv_path": "/tmp/proj/.venv",
                     "python_exe": "/tmp/proj/.venv/bin/python",
                     "installed_count": 0,
                     "failed_count": 0},
            artifact=artifact)


def test_runner_full_greenfield_loop(store):
    """Drive the full greenfield path with stub executors.

    clarify -> ask(clarify_answers) -> decompose -> ask(approve_plan) ->
    scaffold -> apply -> verify (terminal).
    """
    _install_stub_clarify()
    _install_stub_decompose()
    _install_stub_scaffold()
    _install_stub_apply_greenfield()
    _install_stub_bootstrap_env_greenfield()
    _install_stub_api_check_greenfield()
    _install_stub_smoke_greenfield()
    _install_stub_verify_greenfield()
    runner = SessionRunner(store)
    session = runner.start_session(
        objective="build a Flask api that saves JSON to disk",
        project_root="/tmp/proj",
        mode=SessionMode.GREENFIELD)

    # Root task is CLARIFY_REQUIREMENTS, not EXPLORE.
    tasks = store.list_tasks(session.session_id)
    assert len(tasks) == 1
    assert tasks[0].kind is TaskKind.CLARIFY_REQUIREMENTS

    # 1. CLARIFY runs -> spawns ASK_USER(clarify_answers).
    runner.run_next(session_id=session.session_id, deps=ExecutorDeps())
    tasks = store.list_tasks(session.session_id)
    answer_ask = next(t for t in tasks if t.kind is TaskKind.ASK_USER
                      and t.inputs.get("expected_kind") == "clarify_answers")
    assert answer_ask.inputs["requirements_artifact_id"]

    # 2. The ASK reaches IN_PROGRESS.
    runner.run_next(session_id=session.session_id, deps=ExecutorDeps())
    answer_ask = store.get_task(answer_ask.task_id)
    assert answer_ask.status is TaskNodeStatus.IN_PROGRESS

    # 3. User submits answers -> spawns DECOMPOSE.
    answers_decision = build_decision(
        session_id=session.session_id, task=answer_ask,
        chosen={"answers": {"q1": "Python + Flask",
                            "q2": "JSON on disk",
                            "q3": "None"}})
    runner.post_decision(session_id=session.session_id,
                         decision=answers_decision)
    decompose = next(t for t in store.list_tasks(session.session_id)
                     if t.kind is TaskKind.DECOMPOSE)
    assert decompose.status is TaskNodeStatus.READY
    assert decompose.inputs["answers"]["q1"] == "Python + Flask"

    # 4. DECOMPOSE runs -> spawns ASK_USER(approve_plan).
    runner.run_next(session_id=session.session_id, deps=ExecutorDeps())
    approve_ask = next(t for t in store.list_tasks(session.session_id)
                       if t.kind is TaskKind.ASK_USER
                       and t.inputs.get("expected_kind") == "approve_plan")
    assert approve_ask.inputs["work_plan_artifact_id"]

    # 5. The APPROVE_PLAN ASK reaches IN_PROGRESS.
    runner.run_next(session_id=session.session_id, deps=ExecutorDeps())
    approve_ask = store.get_task(approve_ask.task_id)
    assert approve_ask.status is TaskNodeStatus.IN_PROGRESS

    # 6. User approves -> spawns SCAFFOLD.
    approve_decision = build_decision(
        session_id=session.session_id, task=approve_ask,
        chosen={"approved": True})
    runner.post_decision(session_id=session.session_id,
                         decision=approve_decision)
    scaffold = next(t for t in store.list_tasks(session.session_id)
                    if t.kind is TaskKind.SCAFFOLD)
    assert scaffold.status is TaskNodeStatus.READY

    # 7. SCAFFOLD runs -> spawns APPLY with mode=greenfield.
    runner.run_next(session_id=session.session_id, deps=ExecutorDeps())
    apply_t = next(t for t in store.list_tasks(session.session_id)
                   if t.kind is TaskKind.APPLY)
    assert apply_t.inputs["scaffold_artifact_id"]
    assert apply_t.inputs["mode"] == "greenfield"

    # 8. APPLY runs -> spawns BOOTSTRAP_ENV (greenfield-only edge).
    runner.run_next(session_id=session.session_id, deps=ExecutorDeps())
    boot_t = next(t for t in store.list_tasks(session.session_id)
                  if t.kind is TaskKind.BOOTSTRAP_ENV)
    assert boot_t.status is TaskNodeStatus.READY
    assert boot_t.inputs["apply_artifact_id"]
    assert boot_t.inputs["mode"] == "greenfield"

    # 9. BOOTSTRAP_ENV runs -> spawns API_CHECK with build_artifact_id.
    runner.run_next(session_id=session.session_id, deps=ExecutorDeps())
    api_t = next(t for t in store.list_tasks(session.session_id)
                 if t.kind is TaskKind.API_CHECK)
    assert api_t.status is TaskNodeStatus.READY
    assert api_t.inputs["build_artifact_id"]

    # 10. API_CHECK runs (skipped outcome) -> spawns SMOKE.
    runner.run_next(session_id=session.session_id, deps=ExecutorDeps())
    smoke_t = next(t for t in store.list_tasks(session.session_id)
                   if t.kind is TaskKind.SMOKE)
    assert smoke_t.status is TaskNodeStatus.READY
    assert smoke_t.inputs["build_artifact_id"]
    assert smoke_t.inputs["api_check_artifact_id"]

    # 11. SMOKE runs (skipped outcome) -> spawns VERIFY.
    runner.run_next(session_id=session.session_id, deps=ExecutorDeps())
    verify_t = next(t for t in store.list_tasks(session.session_id)
                    if t.kind is TaskKind.VERIFY)
    assert verify_t.status is TaskNodeStatus.READY
    assert verify_t.inputs["build_artifact_id"]
    assert verify_t.inputs["smoke_artifact_id"]

    # 12. VERIFY runs -> terminal; a skipped suite completes the session.
    runner.run_next(session_id=session.session_id, deps=ExecutorDeps())
    verify_after = store.get_task(verify_t.task_id)
    assert verify_after.status is TaskNodeStatus.DONE
    assert runner.run_next(
        session_id=session.session_id, deps=ExecutorDeps()) is None
    assert store.get_session(
        session.session_id).status is SessionStatus.COMPLETED

    # All eight greenfield artifacts present + clean separation from
    # the explore-loop kinds.
    kinds = {a.kind for a in store.list_artifacts(session.session_id)}
    assert ArtifactKind.REQUIREMENTS_SHEET in kinds
    assert ArtifactKind.WORK_PLAN in kinds
    assert ArtifactKind.SCAFFOLD_PATCHES in kinds
    assert ArtifactKind.APPLIED_CHANGES in kinds
    assert ArtifactKind.BUILD_REPORT in kinds
    assert ArtifactKind.API_CHECK_REPORT in kinds
    assert ArtifactKind.SMOKE_REPORT in kinds
    assert ArtifactKind.VERIFY_REPORT in kinds
    assert ArtifactKind.DIRECTIONS_LIST not in kinds


def test_runner_greenfield_reject_plan_halts_loop(store):
    """Declining the plan should keep SCAFFOLD/APPLY/VERIFY off the tree."""
    _install_stub_clarify()
    _install_stub_decompose()
    _install_stub_scaffold()
    runner = SessionRunner(store)
    session = runner.start_session(
        objective="build a Flask api", project_root="/tmp/proj",
        mode=SessionMode.GREENFIELD)
    runner.run_next(session_id=session.session_id, deps=ExecutorDeps())
    runner.run_next(session_id=session.session_id, deps=ExecutorDeps())
    answer_ask = next(t for t in store.list_tasks(session.session_id)
                      if t.kind is TaskKind.ASK_USER
                      and t.inputs.get("expected_kind") == "clarify_answers")
    runner.post_decision(
        session_id=session.session_id,
        decision=build_decision(
            session_id=session.session_id, task=answer_ask,
            chosen={"answers": {"q1": "Flask"}}))
    runner.run_next(session_id=session.session_id, deps=ExecutorDeps())
    runner.run_next(session_id=session.session_id, deps=ExecutorDeps())
    approve_ask = next(t for t in store.list_tasks(session.session_id)
                       if t.kind is TaskKind.ASK_USER
                       and t.inputs.get("expected_kind") == "approve_plan")
    runner.post_decision(
        session_id=session.session_id,
        decision=build_decision(
            session_id=session.session_id, task=approve_ask,
            chosen={"approved": False},
            rationale="want a different stack"))

    tasks = store.list_tasks(session.session_id)
    assert not [t for t in tasks if t.kind is TaskKind.SCAFFOLD]
    assert not [t for t in tasks if t.kind is TaskKind.APPLY]
    assert not [t for t in tasks if t.kind is TaskKind.VERIFY]
    approve_after = store.get_task(approve_ask.task_id)
    assert approve_after.status is TaskNodeStatus.DONE



# --------------------- repair module unit tests ---------------------

def test_classify_unittest_pytest_mix_from_traceback():
    """assertLogs AttributeError in pytest output -> unittest_pytest_mix."""
    from cgx.session.repair.classify import classify_verify_report
    content = {
        "outcome": "assertions_failed",
        "returncode": 1,
        "stdout": (
            "FAILED tests/test_app.py::TestThing::test_logs - "
            "AttributeError: 'TestThing' object has no attribute 'assertLogs'"
        ),
        "stderr": "",
    }
    assert classify_verify_report(content) == "unittest_pytest_mix"


def test_classify_unknown_for_plain_assertion_failure():
    """A regular assert failure has no auto-repair -> unknown."""
    from cgx.session.repair.classify import classify_verify_report
    content = {
        "outcome": "assertions_failed",
        "returncode": 1,
        "stdout": "E   assert 1 == 2\nE    +  where 1 = compute()",
        "stderr": "",
    }
    assert classify_verify_report(content) == "unknown"


def test_classify_skipped_outcomes_are_unknown():
    """Skipped / pytest_missing are env problems, not REPAIR."""
    from cgx.session.repair.classify import classify_verify_report
    for outcome in ("passed", "skipped", "pytest_missing"):
        assert classify_verify_report(
            {"outcome": outcome, "stdout": ""}) == "unknown"


def test_classify_no_tests_collected_is_empty_test_suite():
    """pytest exit 5 (selected files, 0 tests) -> repairable re-scaffold."""
    from cgx.session.repair.classify import classify_verify_report
    content = {
        "outcome": "no_tests_collected",
        "returncode": 5,
        "stdout": "\nno tests ran in 0.09s\n",
        "stderr": "",
    }
    assert classify_verify_report(content) == "empty_test_suite"


def test_failure_signature_stable_across_runs():
    """Same outcome + rc + first error line -> same signature."""
    from cgx.session.repair.classify import failure_signature
    a = {"outcome": "assertions_failed", "returncode": 1,
         "stdout": "E   AttributeError: 'X' object has no attribute 'assertLogs'\n"
                   "duration 0.42s"}
    b = {"outcome": "assertions_failed", "returncode": 1,
         "stdout": "E   AttributeError: 'X' object has no attribute 'assertLogs'\n"
                   "duration 0.87s"}
    assert failure_signature(a) == failure_signature(b)


def test_failure_signature_differs_on_different_error():
    from cgx.session.repair.classify import failure_signature
    a = {"outcome": "assertions_failed", "returncode": 1,
         "stdout": "E   AttributeError: ... 'assertLogs'"}
    b = {"outcome": "assertions_failed", "returncode": 1,
         "stdout": "E   AssertionError: 1 == 2"}
    assert failure_signature(a) != failure_signature(b)


def test_locate_unittest_pytest_mix_finds_offending_class(tmp_path: Path):
    from cgx.session.repair.locate import locate_unittest_pytest_mix
    rel = "tests/test_app.py"
    (tmp_path / "tests").mkdir()
    (tmp_path / rel).write_text(
        "class TestThing:\n"
        "    def test_logs(self):\n"
        "        with self.assertLogs('x'):\n"
        "            pass\n",
        encoding="utf-8",
    )
    locs = locate_unittest_pytest_mix(tmp_path, [rel])
    assert len(locs) == 1
    assert locs[0].rel_path == rel
    assert locs[0].class_name == "TestThing"
    assert "assertLogs" in locs[0].helpers


def test_locate_skips_class_already_inheriting_testcase(tmp_path: Path):
    from cgx.session.repair.locate import locate_unittest_pytest_mix
    rel = "tests/test_ok.py"
    (tmp_path / "tests").mkdir()
    (tmp_path / rel).write_text(
        "import unittest\n"
        "class TestOk(unittest.TestCase):\n"
        "    def test_logs(self):\n"
        "        with self.assertLogs('x'):\n"
        "            pass\n",
        encoding="utf-8",
    )
    assert locate_unittest_pytest_mix(tmp_path, [rel]) == []


def test_propose_unittest_pytest_mix_generates_diff(tmp_path: Path):
    from cgx.session.repair.locate import locate_unittest_pytest_mix
    from cgx.session.repair.propose import propose_unittest_pytest_mix
    rel = "tests/test_app.py"
    (tmp_path / "tests").mkdir()
    (tmp_path / rel).write_text(
        "class TestThing:\n"
        "    def test_logs(self):\n"
        "        with self.assertLogs('x'):\n"
        "            pass\n",
        encoding="utf-8",
    )
    locs = locate_unittest_pytest_mix(tmp_path, [rel])
    diffs = propose_unittest_pytest_mix(tmp_path, locs)
    assert len(diffs) == 1
    assert diffs[0]["file"] == rel
    patch = diffs[0]["patch"]
    assert patch.startswith("--- a/tests/test_app.py")
    assert "+++ b/tests/test_app.py" in patch
    assert "class TestThing(unittest.TestCase):" in patch
    assert "+import unittest" in patch


def test_propose_preserves_existing_bases(tmp_path: Path):
    """class Foo(Mixin): -> class Foo(Mixin, unittest.TestCase):"""
    from cgx.session.repair.locate import StyleMixLocation
    from cgx.session.repair.propose import propose_unittest_pytest_mix
    rel = "tests/test_app.py"
    (tmp_path / "tests").mkdir()
    (tmp_path / rel).write_text(
        "import unittest\n"
        "class Mixin: pass\n"
        "class TestThing(Mixin):\n"
        "    def t(self): self.assertEqual(1, 1)\n",
        encoding="utf-8",
    )
    locs = [StyleMixLocation(
        rel_path=rel, class_name="TestThing", class_lineno=3,
        helpers=frozenset({"assertEqual"}))]
    diffs = propose_unittest_pytest_mix(tmp_path, locs)
    assert len(diffs) == 1
    assert "class TestThing(Mixin, unittest.TestCase):" in diffs[0]["patch"]


def test_repair_executor_emits_repair_plan_artifact(store, tmp_path: Path):
    """End-to-end: failing VERIFY_REPORT -> REPAIR task -> REPAIR_PLAN diffs."""
    from cgx.session.tasks import repair as _repair_module  # noqa: F401
    rel = "tests/test_app.py"
    (tmp_path / "tests").mkdir()
    (tmp_path / rel).write_text(
        "class TestThing:\n"
        "    def test_logs(self):\n"
        "        with self.assertLogs('x'):\n"
        "            pass\n",
        encoding="utf-8",
    )
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    verify_task = TaskNode.new(
        session.session_id, TaskKind.VERIFY, "verify",
        inputs={"mode": SessionMode.GREENFIELD.value})
    verify_artifact = Artifact.new(
        session_id=session.session_id,
        produced_by_task_id=verify_task.task_id,
        kind=ArtifactKind.VERIFY_REPORT,
        content={
            "outcome": "assertions_failed",
            "returncode": 1,
            "changed_files": [rel],
            "tests_selected": [rel],
            "stdout": "AttributeError: 'TestThing' object has no attribute 'assertLogs'",
            "stderr": "",
        })
    store.save_task(verify_task)
    store.save_artifact(verify_artifact)
    repair_task = TaskNode.new(
        session.session_id, TaskKind.REPAIR, "repair",
        inputs={"verify_artifact_id": verify_artifact.artifact_id,
                "mode": SessionMode.GREENFIELD.value,
                "repair_attempt": 1})
    deps = ExecutorDeps(project_root=str(tmp_path), store=store)
    from cgx.session.tasks.base import _REGISTRY
    result = _REGISTRY[TaskKind.REPAIR](repair_task, deps)
    assert result.failure is None
    assert result.artifact is not None
    assert result.artifact.kind is ArtifactKind.REPAIR_PLAN
    assert result.outputs["classification"] == "unittest_pytest_mix"
    assert result.outputs["can_apply"] is True
    assert result.outputs["diff_count"] >= 1


def test_repair_executor_emits_empty_plan_for_unknown(store, tmp_path: Path):
    """Unclassifiable failure -> empty diffs + can_apply False (router escalates)."""
    from cgx.session.tasks import repair as _repair_module  # noqa: F401
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    verify_task = TaskNode.new(
        session.session_id, TaskKind.VERIFY, "verify", inputs={})
    verify_artifact = Artifact.new(
        session_id=session.session_id,
        produced_by_task_id=verify_task.task_id,
        kind=ArtifactKind.VERIFY_REPORT,
        content={"outcome": "assertions_failed", "returncode": 1,
                 "stdout": "E   assert 1 == 2", "stderr": ""})
    store.save_task(verify_task)
    store.save_artifact(verify_artifact)
    repair_task = TaskNode.new(
        session.session_id, TaskKind.REPAIR, "repair",
        inputs={"verify_artifact_id": verify_artifact.artifact_id,
                "mode": SessionMode.GREENFIELD.value,
                "repair_attempt": 1})
    deps = ExecutorDeps(project_root=str(tmp_path), store=store)
    from cgx.session.tasks.base import _REGISTRY
    result = _REGISTRY[TaskKind.REPAIR](repair_task, deps)
    assert result.failure is None
    assert result.outputs["classification"] == "unknown"
    assert result.outputs["can_apply"] is False
    assert result.outputs["diff_count"] == 0


def test_generate_repair_files_returns_validated_rewrites():
    """The engine helper keeps only changed, syntactically-valid rewrites."""
    import json as _json
    from cgx.answer.engine import generate_repair_files
    src = "def add(a, b):\n    return a - b\n"
    fixed = "def add(a, b):\n    return a + b\n"
    provider = _StubProvider(_json.dumps({"files": [
        {"path": "src/calc.py", "content": fixed},
        # unchanged file -> dropped
        {"path": "tests/test_calc.py", "content": "T"},
        # not in the supplied file list -> dropped
        {"path": "src/other.py", "content": "x = 1\n"},
        # broken Python -> dropped by the syntax gate
        {"path": "src/broken.py", "content": "def x(:\n"},
    ]}))
    out = generate_repair_files(
        provider,
        goal="a calculator",
        failure_text="E   assert 1 == 3",
        files=[
            {"path": "src/calc.py", "content": src},
            {"path": "tests/test_calc.py", "content": "T"},
            {"path": "src/broken.py", "content": "def x():\n    pass\n"},
        ],
    )
    assert out == {"src/calc.py": fixed}


def test_repair_executor_llm_logic_repair_emits_patch(store, tmp_path: Path):
    """unknown assertion failure + provider -> bounded LLM patch (can_apply)."""
    import json as _json
    from cgx.session.tasks import repair as _repair_module  # noqa: F401
    rel = "src/calc.py"
    (tmp_path / "src").mkdir()
    (tmp_path / rel).write_text(
        "def add(a, b):\n    return a - b\n", encoding="utf-8")
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    verify_task = TaskNode.new(
        session.session_id, TaskKind.VERIFY, "verify",
        inputs={"mode": SessionMode.GREENFIELD.value})
    verify_artifact = Artifact.new(
        session_id=session.session_id,
        produced_by_task_id=verify_task.task_id,
        kind=ArtifactKind.VERIFY_REPORT,
        content={
            "outcome": "assertions_failed",
            "returncode": 1,
            "changed_files": [rel],
            "stdout": "E   assert 1 == 3\nE    +  where 1 = add(2, 1)",
            "stderr": "",
        })
    store.save_task(verify_task)
    store.save_artifact(verify_artifact)
    provider = _StubProvider(_json.dumps({"files": [
        {"path": rel, "content": "def add(a, b):\n    return a + b\n"}]}))
    repair_task = TaskNode.new(
        session.session_id, TaskKind.REPAIR, "repair",
        inputs={"verify_artifact_id": verify_artifact.artifact_id,
                "mode": SessionMode.GREENFIELD.value,
                "repair_attempt": 1})
    deps = ExecutorDeps(
        project_root=str(tmp_path), store=store, provider=provider)
    from cgx.session.tasks.base import _REGISTRY
    result = _REGISTRY[TaskKind.REPAIR](repair_task, deps)
    assert result.failure is None
    assert result.outputs["classification"] == "unknown"
    assert result.outputs["can_apply"] is True
    assert result.outputs["diff_count"] == 1
    assert result.outputs["strategy"] == "patch"
    diffs = result.artifact.content["diffs"]
    assert diffs[0]["file"] == rel
    assert "return a + b" in diffs[0]["patch"]


def test_repair_executor_llm_logic_repair_respects_attempt_budget(
        store, tmp_path: Path):
    """Past the LLM-repair attempt cap the executor falls back to no patch."""
    import json as _json
    from cgx.session.tasks import repair as _repair_module  # noqa: F401
    rel = "src/calc.py"
    (tmp_path / "src").mkdir()
    (tmp_path / rel).write_text(
        "def add(a, b):\n    return a - b\n", encoding="utf-8")
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    verify_artifact = Artifact.new(
        session_id=session.session_id, produced_by_task_id="t_verify",
        kind=ArtifactKind.VERIFY_REPORT,
        content={
            "outcome": "assertions_failed",
            "returncode": 1,
            "changed_files": [rel],
            "stdout": "E   assert 1 == 3",
            "stderr": "",
        })
    store.save_artifact(verify_artifact)
    provider = _StubProvider(_json.dumps({"files": [
        {"path": rel, "content": "def add(a, b):\n    return a + b\n"}]}))
    repair_task = TaskNode.new(
        session.session_id, TaskKind.REPAIR, "repair",
        inputs={"verify_artifact_id": verify_artifact.artifact_id,
                "mode": SessionMode.GREENFIELD.value,
                "repair_attempt": 3})
    deps = ExecutorDeps(
        project_root=str(tmp_path), store=store, provider=provider)
    from cgx.session.tasks.base import _REGISTRY
    result = _REGISTRY[TaskKind.REPAIR](repair_task, deps)
    assert result.failure is None
    assert result.outputs["classification"] == "unknown"
    assert result.outputs["can_apply"] is False
    assert result.outputs["diff_count"] == 0
    assert provider.calls == []


def test_repair_executor_empty_test_suite_regenerates(store, tmp_path: Path):
    """no_tests_collected -> empty_test_suite classification + regenerate."""
    from cgx.session.tasks import repair as _repair_module  # noqa: F401
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    verify_task = TaskNode.new(
        session.session_id, TaskKind.VERIFY, "verify", inputs={})
    verify_artifact = Artifact.new(
        session_id=session.session_id,
        produced_by_task_id=verify_task.task_id,
        kind=ArtifactKind.VERIFY_REPORT,
        content={"outcome": "no_tests_collected", "returncode": 5,
                 "stdout": "\nno tests ran in 0.09s\n", "stderr": ""})
    store.save_task(verify_task)
    store.save_artifact(verify_artifact)
    repair_task = TaskNode.new(
        session.session_id, TaskKind.REPAIR, "repair",
        inputs={"verify_artifact_id": verify_artifact.artifact_id,
                "mode": SessionMode.GREENFIELD.value,
                "repair_attempt": 1})
    deps = ExecutorDeps(project_root=str(tmp_path), store=store)
    from cgx.session.tasks.base import _REGISTRY
    result = _REGISTRY[TaskKind.REPAIR](repair_task, deps)
    assert result.failure is None
    assert result.outputs["classification"] == "empty_test_suite"
    assert result.outputs["can_apply"] is False
    assert result.outputs["diff_count"] == 0
    assert result.outputs["strategy"] == "regenerate"
    assert result.outputs["extra_constraints"]["kind"] == "empty_test_suite"
    assert "top level" in result.artifact.content["rationale"]


def test_repair_executor_emits_smoke_repair_plan(store, tmp_path: Path):
    """SMOKE_REPORT input -> classification=smoke_import_failure, can_apply=False."""
    from cgx.session.tasks import repair as _repair_module  # noqa: F401
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    smoke_task = TaskNode.new(
        session.session_id, TaskKind.SMOKE, "smoke", inputs={})
    smoke_artifact = Artifact.new(
        session_id=session.session_id,
        produced_by_task_id=smoke_task.task_id,
        kind=ArtifactKind.SMOKE_REPORT,
        content={"outcome": "failed",
                 "failed_modules": ["werkzeug"],
                 "failure_signature": "smoke_import|werkzeug",
                 "modules": [{"name": "werkzeug", "ok": False,
                              "stderr_tail": "ImportError: url_quote"}]})
    store.save_task(smoke_task)
    store.save_artifact(smoke_artifact)
    repair_task = TaskNode.new(
        session.session_id, TaskKind.REPAIR, "repair",
        inputs={"smoke_artifact_id": smoke_artifact.artifact_id,
                "mode": SessionMode.GREENFIELD.value,
                "repair_attempt": 1})
    deps = ExecutorDeps(project_root=str(tmp_path), store=store)
    from cgx.session.tasks.base import _REGISTRY
    result = _REGISTRY[TaskKind.REPAIR](repair_task, deps)
    assert result.failure is None
    assert result.artifact.kind is ArtifactKind.REPAIR_PLAN
    assert result.outputs["classification"] == "smoke_import_failure"
    assert result.outputs["can_apply"] is False
    assert result.outputs["diff_count"] == 0
    assert result.artifact.content["failed_modules"] == ["werkzeug"]
    assert "werkzeug" in result.artifact.content["rationale"]


def test_repair_executor_emits_regenerate_for_build_smoke_failure(
        store, tmp_path: Path):
    """A JS build-smoke break -> strategy=regenerate with the build error."""
    from cgx.session.tasks import repair as _repair_module  # noqa: F401
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    smoke_task = TaskNode.new(
        session.session_id, TaskKind.SMOKE, "smoke", inputs={})
    smoke_artifact = Artifact.new(
        session_id=session.session_id,
        produced_by_task_id=smoke_task.task_id,
        kind=ArtifactKind.SMOKE_REPORT,
        content={"outcome": "failed",
                 "failed_modules": [],
                 "failure_signature": "smoke_import|npm run build --silent",
                 "modules": [],
                 "build_smoke": {"label": "npm run build --silent",
                                 "ok": False,
                                 "stderr_tail": "src/App.jsx: TS2345"}})
    store.save_task(smoke_task)
    store.save_artifact(smoke_artifact)
    repair_task = TaskNode.new(
        session.session_id, TaskKind.REPAIR, "repair",
        inputs={"smoke_artifact_id": smoke_artifact.artifact_id,
                "mode": SessionMode.GREENFIELD.value,
                "repair_attempt": 1})
    deps = ExecutorDeps(project_root=str(tmp_path), store=store)
    from cgx.session.tasks.base import _REGISTRY
    result = _REGISTRY[TaskKind.REPAIR](repair_task, deps)
    assert result.failure is None
    assert result.outputs["strategy"] == "regenerate"
    constraints = result.outputs["extra_constraints"]
    assert constraints["kind"] == "invalid_build_smoke"
    assert "TS2345" in constraints["build_error"]
    assert "build" in result.artifact.content["rationale"].lower()


def test_classify_missing_module_pythonpath_from_collection_error():
    """ModuleNotFoundError during collection -> missing_module_pythonpath."""
    from cgx.session.repair.classify import classify_verify_report
    content = {
        "outcome": "collection_error",
        "returncode": 2,
        "stdout": (
            "ImportError while importing test module 'tests/test_app.py'.\n"
            "tests/test_app.py:1: in <module>\n"
            "    from app import create_app\n"
            "E   ModuleNotFoundError: No module named 'app'\n"),
        "stderr": "",
    }
    assert classify_verify_report(content) == "missing_module_pythonpath"


def test_missing_module_names_extracts_dotted_targets():
    """Dotted module paths from pytest output are deduped + order-preserved."""
    from cgx.session.repair.classify import missing_module_names
    content = {"stdout": (
        "E   ModuleNotFoundError: No module named 'app'\n"
        "E   ModuleNotFoundError: No module named 'api.routes'\n"
        "E   ModuleNotFoundError: No module named 'app'\n")}
    assert missing_module_names(content) == ("app", "api.routes")


def test_locate_missing_module_pythonpath_resolves_project_dir(tmp_path: Path):
    """Top-level module exists on disk -> location returned."""
    from cgx.session.repair.locate import locate_missing_module_pythonpath
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "__init__.py").write_text("", encoding="utf-8")
    content = {"stdout": "E   ModuleNotFoundError: No module named 'app'\n"}
    locs = locate_missing_module_pythonpath(tmp_path, content)
    assert len(locs) == 1
    assert locs[0].top_level == "app"
    assert locs[0].resolved_path == "app"


def test_locate_missing_module_pythonpath_resolves_sibling_file(tmp_path: Path):
    """A sibling ``foo.py`` counts as a resolvable module."""
    from cgx.session.repair.locate import locate_missing_module_pythonpath
    (tmp_path / "foo.py").write_text("def bar(): return 1\n", encoding="utf-8")
    content = {"stdout": "E   ModuleNotFoundError: No module named 'foo'\n"}
    locs = locate_missing_module_pythonpath(tmp_path, content)
    assert len(locs) == 1
    assert locs[0].resolved_path == "foo.py"


def test_locate_missing_module_pythonpath_skips_third_party(tmp_path: Path):
    """A module that doesn't exist on disk is BOOTSTRAP_ENV's problem, not ours."""
    from cgx.session.repair.locate import locate_missing_module_pythonpath
    content = {"stdout": "E   ModuleNotFoundError: No module named 'flask'\n"}
    assert locate_missing_module_pythonpath(tmp_path, content) == []


def test_locate_missing_module_pythonpath_resolves_nested_submodule(
        tmp_path: Path):
    """A fully-resolvable dotted path (api.routes) is a real sys.path gap."""
    from cgx.session.repair.locate import locate_missing_module_pythonpath
    (tmp_path / "api").mkdir()
    (tmp_path / "api" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "api" / "routes.py").write_text("x = 1\n", encoding="utf-8")
    content = {"stdout": "E   ModuleNotFoundError: No module named 'api.routes'\n"}
    locs = locate_missing_module_pythonpath(tmp_path, content)
    assert len(locs) == 1
    assert locs[0].top_level == "api"


def test_locate_missing_module_pythonpath_skips_missing_leaf(tmp_path: Path):
    """tests/ exists but tests/auth.py does not -> not a pythonpath fix."""
    from cgx.session.repair.locate import locate_missing_module_pythonpath
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "__init__.py").write_text("", encoding="utf-8")
    content = {"stdout": "E   ModuleNotFoundError: No module named 'tests.auth'\n"}
    assert locate_missing_module_pythonpath(tmp_path, content) == []


def test_propose_missing_module_pythonpath_creates_new_conftest(tmp_path: Path):
    """No conftest.py at root -> diff creates one with sys.path snippet."""
    from cgx.session.repair.locate import MissingPythonpathLocation
    from cgx.session.repair.propose import propose_missing_module_pythonpath
    locs = [MissingPythonpathLocation(
        module_name="app", top_level="app", resolved_path="app")]
    diffs = propose_missing_module_pythonpath(tmp_path, locs)
    assert len(diffs) == 1
    assert diffs[0]["file"] == "conftest.py"
    patch = diffs[0]["patch"]
    assert "sys.path.insert(0, str(_HERE))" in patch
    assert "cgx-repair: missing_module_pythonpath" in patch


def test_propose_missing_module_pythonpath_prepends_existing_conftest(tmp_path: Path):
    """Existing conftest.py without the marker -> snippet is prepended."""
    from cgx.session.repair.locate import MissingPythonpathLocation
    from cgx.session.repair.propose import propose_missing_module_pythonpath
    (tmp_path / "conftest.py").write_text(
        "import pytest\n\n\n@pytest.fixture\ndef widget():\n    return 42\n",
        encoding="utf-8")
    locs = [MissingPythonpathLocation(
        module_name="app", top_level="app", resolved_path="app")]
    diffs = propose_missing_module_pythonpath(tmp_path, locs)
    assert len(diffs) == 1
    patch = diffs[0]["patch"]
    assert "import pytest" in patch
    assert "sys.path.insert(0, str(_HERE))" in patch


def test_propose_missing_module_pythonpath_no_op_when_marker_present(tmp_path: Path):
    """Repair already applied -> empty diffs (router will escalate)."""
    from cgx.session.repair.locate import MissingPythonpathLocation
    from cgx.session.repair.propose import propose_missing_module_pythonpath
    (tmp_path / "conftest.py").write_text(
        "# cgx-repair: missing_module_pythonpath\nimport sys\n",
        encoding="utf-8")
    locs = [MissingPythonpathLocation(
        module_name="app", top_level="app", resolved_path="app")]
    assert propose_missing_module_pythonpath(tmp_path, locs) == []


def test_repair_executor_emits_pythonpath_plan(store, tmp_path: Path):
    """End-to-end: ModuleNotFoundError VERIFY_REPORT -> conftest.py diff."""
    from cgx.session.tasks import repair as _repair_module  # noqa: F401
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "__init__.py").write_text("", encoding="utf-8")
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    verify_task = TaskNode.new(
        session.session_id, TaskKind.VERIFY, "verify",
        inputs={"mode": SessionMode.GREENFIELD.value})
    verify_artifact = Artifact.new(
        session_id=session.session_id,
        produced_by_task_id=verify_task.task_id,
        kind=ArtifactKind.VERIFY_REPORT,
        content={
            "outcome": "collection_error",
            "returncode": 2,
            "stdout": "E   ModuleNotFoundError: No module named 'app'\n",
            "stderr": "",
        })
    verify_task.produced_artifact_id = verify_artifact.artifact_id
    verify_task.status = TaskNodeStatus.DONE
    store.save_task(verify_task)
    store.save_artifact(verify_artifact)
    repair_task = TaskNode.new(
        session.session_id, TaskKind.REPAIR, "repair",
        inputs={"verify_artifact_id": verify_artifact.artifact_id,
                "mode": SessionMode.GREENFIELD.value,
                "repair_attempt": 1})
    deps = ExecutorDeps(project_root=str(tmp_path), store=store)
    from cgx.session.tasks.base import _REGISTRY
    result = _REGISTRY[TaskKind.REPAIR](repair_task, deps)
    assert result.failure is None
    assert result.artifact is not None
    assert result.artifact.kind is ArtifactKind.REPAIR_PLAN
    assert result.outputs["classification"] == "missing_module_pythonpath"
    assert result.outputs["can_apply"] is True
    assert result.outputs["diff_count"] == 1
    diffs = result.artifact.content["diffs"]
    assert diffs[0]["file"] == "conftest.py"


def test_repair_executor_missing_leaf_module_regenerates(store, tmp_path: Path):
    """tests/ exists but tests/auth.py does not -> regenerate, not patch."""
    from cgx.session.tasks import repair as _repair_module  # noqa: F401
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "__init__.py").write_text("", encoding="utf-8")
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    verify_task = TaskNode.new(
        session.session_id, TaskKind.VERIFY, "verify",
        inputs={"mode": SessionMode.GREENFIELD.value})
    verify_artifact = Artifact.new(
        session_id=session.session_id,
        produced_by_task_id=verify_task.task_id,
        kind=ArtifactKind.VERIFY_REPORT,
        content={
            "outcome": "collection_error",
            "returncode": 2,
            "stdout": (
                "E   ModuleNotFoundError: No module named 'tests.auth'\n"),
            "stderr": "",
        })
    verify_task.produced_artifact_id = verify_artifact.artifact_id
    verify_task.status = TaskNodeStatus.DONE
    store.save_task(verify_task)
    store.save_artifact(verify_artifact)
    repair_task = TaskNode.new(
        session.session_id, TaskKind.REPAIR, "repair",
        inputs={"verify_artifact_id": verify_artifact.artifact_id,
                "mode": SessionMode.GREENFIELD.value,
                "repair_attempt": 1})
    deps = ExecutorDeps(project_root=str(tmp_path), store=store)
    from cgx.session.tasks.base import _REGISTRY
    result = _REGISTRY[TaskKind.REPAIR](repair_task, deps)
    assert result.failure is None
    assert result.outputs["classification"] == "missing_module_pythonpath"
    assert result.outputs["can_apply"] is False
    assert result.outputs["diff_count"] == 0
    assert result.outputs["strategy"] == "regenerate"
    assert result.outputs["extra_constraints"]["kind"] == \
        "missing_module_pythonpath"


def test_classify_missing_fixture_from_pytest_traceback():
    """``fixture '<name>' not found`` -> missing_fixture classification."""
    from cgx.session.repair.classify import classify_verify_report
    content = {
        "outcome": "collection_error",
        "returncode": 2,
        "stdout": (
            "tests/test_widget.py:7: in test_widget\n"
            "    assert client.get('/')\n"
            "E       fixture 'client' not found\n"),
        "stderr": "",
    }
    assert classify_verify_report(content) == "missing_fixture"


def test_missing_fixture_names_extracts_targets_in_order():
    """Names from the traceback are deduped and order-preserving."""
    from cgx.session.repair.classify import missing_fixture_names
    content = {"stdout": (
        "E       fixture 'client' not found\n"
        "E       fixture 'db' not found\n"
        "E       fixture 'client' not found\n")}
    assert missing_fixture_names(content) == ("client", "db")


def test_locate_missing_fixture_finds_hoist_candidate(tmp_path: Path):
    """Locator picks up @pytest.fixture defs in test files."""
    from cgx.session.repair.locate import locate_missing_fixture
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_one.py").write_text(
        "import pytest\n\n@pytest.fixture\ndef client():\n    return object()\n",
        encoding="utf-8")
    content = {"stdout": "E       fixture 'client' not found\n"}
    locs = locate_missing_fixture(tmp_path, content)
    assert len(locs) == 1
    assert locs[0].fixture_name == "client"
    assert locs[0].source_rel_path == "tests/test_one.py"
    assert locs[0].target_rel_path == "tests/conftest.py"


def test_locate_missing_fixture_accepts_imported_decorator(tmp_path: Path):
    """``from pytest import fixture`` + ``@fixture`` is recognised."""
    from cgx.session.repair.locate import locate_missing_fixture
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_one.py").write_text(
        "from pytest import fixture\n\n@fixture(scope='module')\n"
        "def client():\n    return 1\n",
        encoding="utf-8")
    content = {"stdout": "E       fixture 'client' not found\n"}
    locs = locate_missing_fixture(tmp_path, content)
    assert len(locs) == 1
    assert locs[0].fixture_name == "client"


def test_locate_missing_fixture_skips_when_no_definition(tmp_path: Path):
    """Unknown fixture with no on-disk def returns an empty list."""
    from cgx.session.repair.locate import locate_missing_fixture
    (tmp_path / "tests").mkdir()
    content = {"stdout": "E       fixture 'ghost' not found\n"}
    assert locate_missing_fixture(tmp_path, content) == []


def test_locate_missing_fixture_skips_venv_subtree(tmp_path: Path):
    """Fixtures inside ``.venv`` must not be hoisted into conftest."""
    from cgx.session.repair.locate import locate_missing_fixture
    (tmp_path / ".venv" / "site-packages" / "pkg").mkdir(parents=True)
    (tmp_path / ".venv" / "site-packages" / "pkg" / "__init__.py").write_text(
        "import pytest\n\n@pytest.fixture\ndef client():\n    return 1\n",
        encoding="utf-8")
    content = {"stdout": "E       fixture 'client' not found\n"}
    assert locate_missing_fixture(tmp_path, content) == []


def test_locate_missing_fixture_target_falls_back_to_root(tmp_path: Path):
    """Without a ``tests/`` dir the hoist target is the project-root conftest."""
    from cgx.session.repair.locate import locate_missing_fixture
    (tmp_path / "support.py").write_text(
        "import pytest\n\n@pytest.fixture\ndef client():\n    return 1\n",
        encoding="utf-8")
    content = {"stdout": "E       fixture 'client' not found\n"}
    locs = locate_missing_fixture(tmp_path, content)
    assert len(locs) == 1
    assert locs[0].target_rel_path == "conftest.py"


def test_propose_missing_fixture_creates_conftest_with_hoisted_def(tmp_path: Path):
    """No existing conftest -> diff creates one with the fixture + pytest import."""
    from cgx.session.repair.locate import MissingFixtureLocation
    from cgx.session.repair.propose import propose_missing_fixture
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_one.py").write_text(
        "import pytest\n\n@pytest.fixture\ndef client():\n    return 'live'\n",
        encoding="utf-8")
    loc = MissingFixtureLocation(
        fixture_name="client",
        source_rel_path="tests/test_one.py",
        source_lineno=3,
        source_end_lineno=5,
        target_rel_path="tests/conftest.py",
    )
    diffs = propose_missing_fixture(tmp_path, [loc])
    assert len(diffs) == 1
    assert diffs[0]["file"] == "tests/conftest.py"
    patch = diffs[0]["patch"]
    assert "import pytest" in patch
    assert "# cgx-repair: missing_fixture client" in patch
    assert "def client():" in patch
    assert "return 'live'" in patch


def test_propose_missing_fixture_no_op_when_marker_present(tmp_path: Path):
    """Already-hoisted fixture -> empty diffs (router escalates)."""
    from cgx.session.repair.locate import MissingFixtureLocation
    from cgx.session.repair.propose import propose_missing_fixture
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "conftest.py").write_text(
        "import pytest\n\n# cgx-repair: missing_fixture client\n"
        "@pytest.fixture\ndef client():\n    return 1\n",
        encoding="utf-8")
    (tmp_path / "tests" / "test_one.py").write_text(
        "import pytest\n\n@pytest.fixture\ndef client():\n    return 1\n",
        encoding="utf-8")
    loc = MissingFixtureLocation(
        fixture_name="client",
        source_rel_path="tests/test_one.py",
        source_lineno=3,
        source_end_lineno=5,
        target_rel_path="tests/conftest.py",
    )
    assert propose_missing_fixture(tmp_path, [loc]) == []


def test_repair_executor_emits_missing_fixture_plan(store, tmp_path: Path):
    """End-to-end: fixture-not-found VERIFY_REPORT -> conftest hoist diff."""
    from cgx.session.tasks import repair as _repair_module  # noqa: F401
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_widget.py").write_text(
        "import pytest\n\n@pytest.fixture\ndef client():\n    return object()\n"
        "\ndef test_widget(client):\n    assert client is not None\n",
        encoding="utf-8")
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    verify_task = TaskNode.new(
        session.session_id, TaskKind.VERIFY, "verify",
        inputs={"mode": SessionMode.GREENFIELD.value})
    verify_artifact = Artifact.new(
        session_id=session.session_id,
        produced_by_task_id=verify_task.task_id,
        kind=ArtifactKind.VERIFY_REPORT,
        content={
            "outcome": "collection_error",
            "returncode": 2,
            "stdout": "E       fixture 'client' not found\n",
            "stderr": "",
        })
    verify_task.produced_artifact_id = verify_artifact.artifact_id
    verify_task.status = TaskNodeStatus.DONE
    store.save_task(verify_task)
    store.save_artifact(verify_artifact)
    repair_task = TaskNode.new(
        session.session_id, TaskKind.REPAIR, "repair",
        inputs={"verify_artifact_id": verify_artifact.artifact_id,
                "mode": SessionMode.GREENFIELD.value,
                "repair_attempt": 1})
    deps = ExecutorDeps(project_root=str(tmp_path), store=store)
    from cgx.session.tasks.base import _REGISTRY
    result = _REGISTRY[TaskKind.REPAIR](repair_task, deps)
    assert result.failure is None
    assert result.artifact is not None
    assert result.outputs["classification"] == "missing_fixture"
    assert result.outputs["can_apply"] is True
    assert result.outputs["diff_count"] == 1
    diffs = result.artifact.content["diffs"]
    assert diffs[0]["file"] == "tests/conftest.py"


def test_lint_test_style_returns_jsonable_issues(tmp_path: Path):
    """The preflight lint surfaces the same classes the REPAIR locator does."""
    from cgx.session.repair.locate import lint_test_style
    rel = "tests/test_app.py"
    (tmp_path / "tests").mkdir()
    (tmp_path / rel).write_text(
        "class TestThing:\n"
        "    def test_logs(self):\n"
        "        with self.assertLogs('x'):\n"
        "            pass\n",
        encoding="utf-8",
    )
    issues = lint_test_style(tmp_path, [rel])
    assert len(issues) == 1
    issue = issues[0]
    assert issue["kind"] == "unittest_pytest_mix"
    assert issue["file"] == rel
    assert issue["class_name"] == "TestThing"
    assert "assertLogs" in issue["helpers"]


def test_bootstrap_env_attaches_style_issues_to_build_report(tmp_path: Path, store):
    """BOOTSTRAP_ENV runs the style lint over applied test files."""
    from cgx.session.tasks import bootstrap_env as _bs_module  # noqa: F401
    (tmp_path / "requirements.txt").write_text("", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    rel_test = "tests/test_app.py"
    (tmp_path / rel_test).write_text(
        "class TestThing:\n"
        "    def test_logs(self):\n"
        "        with self.assertLogs('x'):\n"
        "            pass\n",
        encoding="utf-8",
    )
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    apply_artifact = Artifact.new(
        session_id=session.session_id,
        produced_by_task_id="t_apply",
        kind=ArtifactKind.APPLIED_CHANGES,
        content={"applied_files": [rel_test]},
    )
    store.save_artifact(apply_artifact)
    bs_task = TaskNode.new(
        session.session_id, TaskKind.BOOTSTRAP_ENV, "bootstrap",
        inputs={"mode": SessionMode.GREENFIELD.value,
                "apply_artifact_id": apply_artifact.artifact_id,
                "timeout_seconds": 5.0})
    deps = ExecutorDeps(project_root=str(tmp_path), store=store)
    from cgx.session.tasks.base import _REGISTRY
    result = _REGISTRY[TaskKind.BOOTSTRAP_ENV](bs_task, deps)
    assert result.artifact is not None
    assert result.artifact.kind is ArtifactKind.BUILD_REPORT
    issues = result.artifact.content.get("style_issues") or []
    assert len(issues) == 1
    assert issues[0]["file"] == rel_test
    assert issues[0]["class_name"] == "TestThing"
    assert result.outputs.get("style_issue_count") == 1


def test_bootstrap_env_style_issues_empty_when_no_test_files(tmp_path: Path, store):
    """Applied files with no test modules -> empty style_issues, count 0."""
    from cgx.session.tasks import bootstrap_env as _bs_module  # noqa: F401
    (tmp_path / "requirements.txt").write_text("", encoding="utf-8")
    (tmp_path / "app.py").write_text("def f(): return 1\n", encoding="utf-8")
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    apply_artifact = Artifact.new(
        session_id=session.session_id,
        produced_by_task_id="t_apply",
        kind=ArtifactKind.APPLIED_CHANGES,
        content={"applied_files": ["app.py"]},
    )
    store.save_artifact(apply_artifact)
    bs_task = TaskNode.new(
        session.session_id, TaskKind.BOOTSTRAP_ENV, "bootstrap",
        inputs={"mode": SessionMode.GREENFIELD.value,
                "apply_artifact_id": apply_artifact.artifact_id,
                "timeout_seconds": 5.0})
    deps = ExecutorDeps(project_root=str(tmp_path), store=store)
    from cgx.session.tasks.base import _REGISTRY
    result = _REGISTRY[TaskKind.BOOTSTRAP_ENV](bs_task, deps)
    assert result.artifact is not None
    assert result.artifact.content.get("style_issues") == []
    assert result.outputs.get("style_issue_count") == 0



def test_repair_loop_smoke_on_disk_unittest_pytest_mix(tmp_path: Path, store):
    """End-to-end smoke: VERIFY (fail) -> REPAIR -> apply on disk -> valid file.

    Drives the repair pipeline against a real on-disk file rather than
    mocked artifacts: the REPAIR executor produces a unified diff that
    the shared APPLY writer (``apply_diffs_to_disk``) writes to disk,
    after which the file must parse, inherit ``unittest.TestCase``, and
    import ``unittest`` at module level. This locks the moving parts
    (classify -> locate -> propose -> diff_apply -> disk_apply) against
    the wire format they hand off through.
    """
    import ast
    import importlib.util
    import sys

    from cgx.codegen.disk_apply import apply_diffs_to_disk
    from cgx.session.tasks import repair as _repair_module  # noqa: F401
    from cgx.session.tasks.base import _REGISTRY

    rel = "tests/test_widget.py"
    (tmp_path / "tests").mkdir()
    (tmp_path / rel).write_text(
        '"""Generated test module."""\n'
        "\n"
        "class TestWidget:\n"
        "    def test_logs(self):\n"
        "        with self.assertLogs('x'):\n"
        "            pass\n"
        "\n"
        "    def test_equal(self):\n"
        "        self.assertEqual(1 + 1, 2)\n",
        encoding="utf-8",
    )
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    verify_task = TaskNode.new(
        session.session_id, TaskKind.VERIFY, "verify",
        inputs={"mode": SessionMode.GREENFIELD.value})
    verify_artifact = Artifact.new(
        session_id=session.session_id,
        produced_by_task_id=verify_task.task_id,
        kind=ArtifactKind.VERIFY_REPORT,
        content={
            "outcome": "assertions_failed",
            "returncode": 1,
            "changed_files": [rel],
            "stdout": (
                "AttributeError: 'TestWidget' object has no "
                "attribute 'assertLogs'\n"),
            "stderr": "",
        })
    verify_task.produced_artifact_id = verify_artifact.artifact_id
    verify_task.status = TaskNodeStatus.DONE
    store.save_task(verify_task)
    store.save_artifact(verify_artifact)
    repair_task = TaskNode.new(
        session.session_id, TaskKind.REPAIR, "repair",
        inputs={"verify_artifact_id": verify_artifact.artifact_id,
                "mode": SessionMode.GREENFIELD.value,
                "repair_attempt": 1})
    deps = ExecutorDeps(project_root=str(tmp_path), store=store)
    result = _REGISTRY[TaskKind.REPAIR](repair_task, deps)
    assert result.failure is None
    assert result.artifact is not None
    assert result.outputs["classification"] == "unittest_pytest_mix"
    assert result.outputs["can_apply"] is True

    diffs = result.artifact.content["diffs"]
    apply_result = apply_diffs_to_disk(str(tmp_path), diffs)
    assert not apply_result.get("failed_files"), apply_result
    assert rel in apply_result.get("applied_files", [])

    fixed = (tmp_path / rel).read_text(encoding="utf-8")
    # Parses cleanly.
    tree = ast.parse(fixed)
    # Class header inherits unittest.TestCase (preserving structure).
    class_def = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    base_names = [
        ast.unparse(b) if hasattr(ast, "unparse") else getattr(b, "id", "")
        for b in class_def.bases
    ]
    assert any("unittest.TestCase" in n or n == "TestCase" for n in base_names), (
        base_names)
    # Module gained an ``import unittest`` statement.
    assert any(isinstance(n, ast.Import)
               and any(a.name == "unittest" for a in n.names)
               for n in tree.body)
    # And the rewritten file actually imports / instantiates -- proving
    # the repair produced runnable Python, not just plausible-looking text.
    mod_name = f"_repair_smoke_{tmp_path.name.replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(mod_name, tmp_path / rel)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        instance = module.TestWidget()
        assert hasattr(instance, "assertLogs"), (
            "TestWidget should now inherit unittest.TestCase helpers")
    finally:
        sys.modules.pop(mod_name, None)




# --------------------- Phase 3.2: third_party_import_break ---------------------

def test_classify_third_party_import_break_from_junit_failures():
    """`ImportError: cannot import name 'x' from 'pkg'` -> third_party_import_break."""
    from cgx.session.repair.classify import (
        classify_verify_report, third_party_import_breaks,
    )
    content = {
        "outcome": "collection_error",
        "returncode": 2,
        "failures": [{
            "nodeid": "tests.test_app::test_x",
            "kind": "error",
            "type": "ImportError",
            "message": "cannot import name 'url_quote' from 'werkzeug.urls'",
            "traceback": "  File '.../site-packages/flask/app.py'\n",
        }],
        "stdout": "",
        "stderr": "",
    }
    assert classify_verify_report(content) == "third_party_import_break"
    assert third_party_import_breaks(content) == (("url_quote", "werkzeug"),)


def test_classify_third_party_import_break_wins_over_missing_module():
    """Both patterns present in one blob -> import-name break takes priority."""
    from cgx.session.repair.classify import classify_verify_report
    content = {
        "outcome": "collection_error",
        "returncode": 2,
        "stdout": (
            "E   ModuleNotFoundError: No module named 'flask'\n"
            "E   ImportError: cannot import name 'url_quote' from 'werkzeug.urls'\n"
        ),
        "stderr": "",
    }
    assert classify_verify_report(content) == "third_party_import_break"


def test_pypi_client_uses_disk_cache(tmp_path):
    """Second call for the same package hits the cache, not the fetcher."""
    from cgx.session.repair.pypi_client import PyPIClient
    calls = []

    def fake_fetcher(url):
        calls.append(url)
        return b'{"info": {"name": "flask"}, "releases": {}, "urls": []}'

    client = PyPIClient(cache_dir=tmp_path / "cache", fetcher=fake_fetcher)
    assert client.get_package("Flask")["info"]["name"] == "flask"
    assert client.get_package("flask")["info"]["name"] == "flask"
    # Only one network call despite two get_package() invocations.
    assert len(calls) == 1


def test_pypi_client_returns_none_on_fetch_failure(tmp_path):
    """Network / decode errors degrade to None so the proposer can fall back."""
    from cgx.session.repair.pypi_client import PyPIClient

    def boom(url):
        raise OSError("network down")

    client = PyPIClient(cache_dir=tmp_path / "cache", fetcher=boom)
    assert client.get_package("flask") is None
    assert client.get_release("flask", "2.1.2") is None


def test_propose_third_party_pin_uses_declared_requires_dist(
        tmp_path: Path):
    """When the consumer declares an upper bound, reuse it verbatim."""
    from cgx.session.repair.propose import propose_third_party_pin
    from cgx.session.repair.pypi_client import PyPIClient

    (tmp_path / "requirements.txt").write_text(
        "flask==2.1.2\nwerkzeug>=2.0\n", encoding="utf-8")

    payloads = {
        "https://pypi.org/pypi/flask/2.1.2/json": {
            "info": {
                "name": "Flask",
                "requires_dist": ["Werkzeug<3,>=2.0", "Jinja2>=3.0"],
            },
            "urls": [{"upload_time_iso_8601": "2022-04-28T00:00:00Z"}],
        },
    }
    fetcher = lambda url: __import__("json").dumps(payloads[url]).encode("utf-8")
    client = PyPIClient(cache_dir=tmp_path / "cache", fetcher=fetcher)
    content = {
        "failures": [{
            "type": "ImportError",
            "message": "cannot import name 'url_quote' from 'werkzeug.urls'",
            "traceback": "  File '/x/site-packages/flask/app.py'\n",
        }],
    }
    diffs, decisions = propose_third_party_pin(
        tmp_path, content,
        pairs=(("url_quote", "werkzeug"),),
        installed_packages={"flask": "2.1.2", "werkzeug": "3.0.0"},
        pypi_client=client,
    )
    assert len(diffs) == 1
    assert diffs[0]["file"] == "requirements.txt"
    assert "Werkzeug<3,>=2.0" in diffs[0]["patch"]
    assert decisions and decisions[0]["consumer"] == "flask"
    assert decisions[0]["pin"] == "Werkzeug<3,>=2.0"


def test_propose_third_party_pin_falls_back_to_release_window(
        tmp_path: Path):
    """No declared peer constraint -> pick highest contemporary release."""
    import json
    from cgx.session.repair.propose import propose_third_party_pin
    from cgx.session.repair.pypi_client import PyPIClient

    (tmp_path / "requirements.txt").write_text(
        "flask==2.1.2\n", encoding="utf-8")
    payloads = {
        "https://pypi.org/pypi/flask/2.1.2/json": {
            "info": {"name": "Flask", "requires_dist": []},
            "urls": [{"upload_time_iso_8601": "2022-04-28T00:00:00Z"}],
        },
        "https://pypi.org/pypi/werkzeug/json": {
            "info": {"name": "Werkzeug"},
            "releases": {
                "2.0.3": [{"upload_time_iso_8601": "2022-02-01T00:00:00Z"}],
                "2.1.2": [{"upload_time_iso_8601": "2022-04-28T00:00:00Z"}],
                "3.0.0": [{"upload_time_iso_8601": "2024-09-01T00:00:00Z"}],
            },
        },
    }
    fetcher = lambda url: json.dumps(payloads[url]).encode("utf-8")
    client = PyPIClient(cache_dir=tmp_path / "cache", fetcher=fetcher)
    content = {
        "failures": [{
            "type": "ImportError",
            "message": "cannot import name 'url_quote' from 'werkzeug.urls'",
            "traceback": "  File '/x/site-packages/flask/app.py'\n",
        }],
    }
    diffs, decisions = propose_third_party_pin(
        tmp_path, content,
        pairs=(("url_quote", "werkzeug"),),
        installed_packages={"flask": "2.1.2", "werkzeug": "3.0.0"},
        pypi_client=client,
    )
    assert len(diffs) == 1
    assert "werkzeug==2.1.2" in diffs[0]["patch"]
    assert decisions[0]["pin"] == "werkzeug==2.1.2"


def test_propose_third_party_pin_skips_when_consumer_unknown(
        tmp_path: Path):
    """No traceback in failures -> can't identify consumer -> no diff."""
    from cgx.session.repair.propose import propose_third_party_pin
    from cgx.session.repair.pypi_client import PyPIClient

    client = PyPIClient(cache_dir=tmp_path / "cache",
                        fetcher=lambda url: b'{}')
    content = {"failures": [{
        "type": "ImportError",
        "message": "cannot import name 'X' from 'pkg'",
        "traceback": "",
    }]}
    diffs, decisions = propose_third_party_pin(
        tmp_path, content,
        pairs=(("X", "pkg"),),
        installed_packages={"pkg": "1.0.0"},
        pypi_client=client,
    )
    assert diffs == []
    assert decisions[0]["reason"].startswith("consumer package not detected")


def test_repair_executor_emits_third_party_pin_plan(
        store, tmp_path: Path):
    """Full VERIFY -> REPAIR pipeline produces a requirements.txt diff."""
    import json
    from cgx.session.repair.pypi_client import PyPIClient
    from cgx.session.tasks.repair import run_repair

    (tmp_path / "requirements.txt").write_text(
        "flask==2.1.2\n", encoding="utf-8")

    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)

    build_artifact = Artifact.new(
        session_id=session.session_id,
        produced_by_task_id="task-build",
        kind=ArtifactKind.BUILD_REPORT,
        content={
            "outcome": "succeeded",
            "resolved_packages": [
                {"name": "Flask", "version": "2.1.2"},
                {"name": "Werkzeug", "version": "3.0.0"},
            ],
        },
    )
    store.save_artifact(build_artifact)
    verify_content = {
        "outcome": "collection_error",
        "returncode": 2,
        "build_artifact_id": build_artifact.artifact_id,
        "failures": [{
            "nodeid": "tests.test_app::test_x",
            "type": "ImportError",
            "message": "cannot import name 'url_quote' from 'werkzeug.urls'",
            "traceback": "  File '/x/site-packages/flask/app.py'\n",
        }],
        "stdout": "",
        "stderr": "",
        "mode": SessionMode.GREENFIELD.value,
    }
    verify_artifact = Artifact.new(
        session_id=session.session_id,
        produced_by_task_id="task-verify",
        kind=ArtifactKind.VERIFY_REPORT,
        content=verify_content,
    )
    store.save_artifact(verify_artifact)

    payloads = {
        "https://pypi.org/pypi/flask/2.1.2/json": {
            "info": {"name": "Flask",
                     "requires_dist": ["Werkzeug<3,>=2.0"]},
            "urls": [{"upload_time_iso_8601": "2022-04-28T00:00:00Z"}],
        },
    }
    client = PyPIClient(
        cache_dir=tmp_path / "cache",
        fetcher=lambda url: json.dumps(payloads[url]).encode("utf-8"),
    )

    t = TaskNode.new(session.session_id, TaskKind.REPAIR, "repair",
                     inputs={
                         "verify_artifact_id": verify_artifact.artifact_id,
                         "mode": SessionMode.GREENFIELD.value,
                         "repair_attempt": 1,
                     })
    store.save_task(t)
    result = run_repair(t, ExecutorDeps(
        project_root=str(tmp_path), store=store,
        extra={"pypi_client": client}))
    assert result.failure is None
    assert result.outputs["classification"] == "third_party_import_break"
    assert result.outputs["can_apply"] is True
    diffs = result.artifact.content["diffs"]
    assert len(diffs) == 1 and diffs[0]["file"] == "requirements.txt"
    assert "Werkzeug<3,>=2.0" in diffs[0]["patch"]
    decisions = result.artifact.content["pin_decisions"]
    assert decisions[0]["consumer"] == "flask"



# --------------------- Phase 4.1: scaffold pin validator ---------------------

def test_is_requirements_path_recognises_canonical_layouts():
    from cgx.session.scaffold_validate import is_requirements_path
    assert is_requirements_path("requirements.txt")
    assert is_requirements_path("requirements-dev.txt")
    assert is_requirements_path("requirements/base.txt")
    assert is_requirements_path("requirements/dev.txt")
    assert not is_requirements_path("app/requirements.cfg")
    assert not is_requirements_path("src/app.py")
    assert not is_requirements_path("")


def test_validate_requirements_text_tightens_fragile_peer(tmp_path):
    """Pinned flask consumer -> werkzeug pin from requires_dist is appended."""
    import json
    from cgx.session.repair.pypi_client import PyPIClient
    from cgx.session.scaffold_validate import validate_requirements_text

    payloads = {
        "https://pypi.org/pypi/flask/2.1.2/json": {
            "info": {
                "name": "Flask",
                "requires_dist": [
                    "Werkzeug<3,>=2.0", "Jinja2<4,>=3.0",
                    "itsdangerous<3,>=2.0",
                ],
            },
        },
    }
    client = PyPIClient(
        cache_dir=tmp_path / "cache",
        fetcher=lambda url: json.dumps(payloads[url]).encode("utf-8"),
    )
    text = "flask==2.1.2\nrequests\n"
    new_text, adjustments = validate_requirements_text(
        text, pypi_client=client)
    assert new_text != text
    assert "Werkzeug<3,>=2.0" in new_text
    assert "Jinja2<4,>=3.0" in new_text
    by_peer = {a["peer"]: a for a in adjustments}
    assert by_peer["werkzeug"]["before"] is None
    assert by_peer["werkzeug"]["after"] == "Werkzeug<3,>=2.0"
    assert by_peer["werkzeug"]["consumer"] == "flask"


def test_validate_requirements_text_replaces_unbounded_peer_pin(tmp_path):
    """Existing ``werkzeug>=2.0`` (no upper bound) -> rewritten in-place."""
    import json
    from cgx.session.repair.pypi_client import PyPIClient
    from cgx.session.scaffold_validate import validate_requirements_text

    payloads = {
        "https://pypi.org/pypi/flask/2.1.2/json": {
            "info": {"name": "Flask",
                     "requires_dist": ["Werkzeug<3,>=2.0"]},
        },
    }
    client = PyPIClient(
        cache_dir=tmp_path / "cache",
        fetcher=lambda url: json.dumps(payloads[url]).encode("utf-8"),
    )
    text = "flask==2.1.2\nwerkzeug>=2.0\n"
    new_text, adjustments = validate_requirements_text(
        text, pypi_client=client)
    assert "werkzeug>=2.0\n" not in new_text
    assert "Werkzeug<3,>=2.0\n" in new_text
    assert adjustments[0]["before"] == "werkzeug>=2.0"
    assert adjustments[0]["after"] == "Werkzeug<3,>=2.0"


def test_validate_requirements_text_noop_when_consumer_unpinned(tmp_path):
    """Bare ``flask`` (no version) -> no PyPI lookup, no rewrite."""
    from cgx.session.repair.pypi_client import PyPIClient
    from cgx.session.scaffold_validate import validate_requirements_text

    calls: list = []

    def fetcher(url):
        calls.append(url)
        return b"{}"

    client = PyPIClient(cache_dir=tmp_path / "cache", fetcher=fetcher)
    text = "flask\nwerkzeug\n"
    new_text, adjustments = validate_requirements_text(
        text, pypi_client=client)
    assert new_text == text
    assert adjustments == []
    assert calls == []


def test_validate_requirements_text_degrades_when_pypi_fails(tmp_path):
    """PyPI fetch failure -> validator returns the original text."""
    from cgx.session.repair.pypi_client import PyPIClient
    from cgx.session.scaffold_validate import validate_requirements_text

    def boom(url):
        raise OSError("network down")

    client = PyPIClient(cache_dir=tmp_path / "cache", fetcher=boom)
    text = "flask==2.1.2\n"
    new_text, adjustments = validate_requirements_text(
        text, pypi_client=client)
    assert new_text == text
    assert adjustments == []


def test_validate_scaffold_diffs_rewrites_requirements_patch(tmp_path):
    """End-to-end: a requirements.txt diff is swapped with a tightened one."""
    import json
    from cgx.session.repair.pypi_client import PyPIClient
    from cgx.session.scaffold_validate import (
        _content_to_new_file_patch, validate_scaffold_diffs,
    )

    original_content = "flask==2.1.2\n"
    diffs = [
        {"file": "app.py", "patch": "--- /dev/null\n+++ b/app.py\n@@ -0,0 +1,1 @@\n+x"},
        {"file": "requirements.txt",
         "patch": _content_to_new_file_patch("requirements.txt", original_content)},
    ]
    file_contents = {"app.py": "x\n", "requirements.txt": original_content}

    payloads = {
        "https://pypi.org/pypi/flask/2.1.2/json": {
            "info": {"name": "Flask",
                     "requires_dist": ["Werkzeug<3,>=2.0"]},
        },
    }
    client = PyPIClient(
        cache_dir=tmp_path / "cache",
        fetcher=lambda url: json.dumps(payloads[url]).encode("utf-8"),
    )
    new_diffs, new_contents, adjustments = validate_scaffold_diffs(
        diffs, file_contents, pypi_client=client)
    # First diff (app.py) is untouched.
    assert new_diffs[0] == diffs[0]
    # Second diff (requirements.txt) has been rewritten.
    req_patch = new_diffs[1]["patch"]
    assert "Werkzeug<3,>=2.0" in req_patch
    assert new_contents["requirements.txt"] != original_content
    assert adjustments and adjustments[0]["file"] == "requirements.txt"


def test_validate_requirements_text_drops_stdlib_and_first_party(tmp_path):
    """stdlib (``sqlite3``) and first-party (``auth``) pins are stripped."""
    from cgx.session.repair.pypi_client import PyPIClient
    from cgx.session.scaffold_validate import validate_requirements_text

    client = PyPIClient(cache_dir=tmp_path / "cache", fetcher=lambda url: b"{}")
    text = "flask\nsqlite3==3.4.0\nauth\nrequests\n"
    new_text, adjustments = validate_requirements_text(
        text, pypi_client=client, first_party={"auth"})
    remaining = {ln.strip() for ln in new_text.splitlines() if ln.strip()}
    assert remaining == {"flask", "requests"}
    actions = {a["action"] for a in adjustments}
    assert actions == {"drop_stdlib", "drop_first_party"}


def test_validate_requirements_text_remaps_import_alias(tmp_path):
    """A ``jwt`` pin is rewritten to the real distribution ``PyJWT``."""
    from cgx.session.repair.pypi_client import PyPIClient
    from cgx.session.scaffold_validate import validate_requirements_text

    client = PyPIClient(cache_dir=tmp_path / "cache", fetcher=lambda url: b"{}")
    new_text, adjustments = validate_requirements_text(
        "jwt==2.1.0\n", pypi_client=client)
    assert new_text.strip() == "PyJWT"
    assert adjustments and adjustments[0]["action"] == "remap"
    assert adjustments[0]["after"] == "PyJWT"


def test_validate_scaffold_diffs_drops_first_party_module(tmp_path):
    """First-party set is derived from the generated paths, then applied."""
    from cgx.session.repair.pypi_client import PyPIClient
    from cgx.session.scaffold_validate import (
        _content_to_new_file_patch, validate_scaffold_diffs,
    )

    original = "flask\nauth\nsqlite3\n"
    diffs = [
        {"file": "backend/auth.py",
         "patch": _content_to_new_file_patch("backend/auth.py", "x = 1\n")},
        {"file": "requirements.txt",
         "patch": _content_to_new_file_patch("requirements.txt", original)},
    ]
    file_contents = {"backend/auth.py": "x = 1\n", "requirements.txt": original}
    client = PyPIClient(cache_dir=tmp_path / "cache", fetcher=lambda url: b"{}")
    _new_diffs, new_contents, adjustments = validate_scaffold_diffs(
        diffs, file_contents, pypi_client=client)
    remaining = {ln.strip()
                 for ln in new_contents["requirements.txt"].splitlines()
                 if ln.strip()}
    assert remaining == {"flask"}
    assert {a["action"] for a in adjustments} >= {
        "drop_first_party", "drop_stdlib"}


def test_scaffold_executor_tightens_requirements_pins(store, monkeypatch):
    """SCAFFOLD applies the validator and surfaces ``pin_adjustments``."""
    import json
    from cgx.session.repair.pypi_client import PyPIClient
    from cgx.session.tasks.scaffold import run_scaffold

    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    plan = Artifact.new(
        session.session_id, "task_x", ArtifactKind.WORK_PLAN, {
            "prior_goal": "g",
            "composed_goal": "build flask api",
            "answers": {},
            "plan_md": "",
            "layers": [{"name": "app", "files": [
                {"path": "app.py", "description": "entry"},
                {"path": "requirements.txt", "description": "deps"}]}],
        })
    store.save_artifact(plan)

    contents = {
        "app.py": "import flask\n",
        "requirements.txt": "flask==2.1.2\n",
    }

    def fake_generate(path, *a, **kw):
        body = contents[path]
        patch = (f"--- /dev/null\n+++ b/{path}\n"
                 f"@@ -0,0 +1,{len(body.splitlines())} @@\n"
                 + "\n".join(f"+{ln}" for ln in body.splitlines()))
        return {"file": path, "patch": patch, "content": body,
                "syntax_ok": True, "confidence": 1.0}

    monkeypatch.setattr("cgx.answer.engine.generate_single_scaffold_file",
                        fake_generate)

    payloads = {
        "https://pypi.org/pypi/flask/2.1.2/json": {
            "info": {"name": "Flask",
                     "requires_dist": ["Werkzeug<3,>=2.0"]},
        },
    }

    class _StubProv:
        def chat(self, *a, **kw):
            return ""

    t = TaskNode.new(session.session_id, TaskKind.SCAFFOLD, "s",
                     inputs={"work_plan_artifact_id": plan.artifact_id})
    store.save_task(t)
    import tempfile as _tmp
    with _tmp.TemporaryDirectory() as cache_root:
        client = PyPIClient(
            cache_dir=cache_root,
            fetcher=lambda url: json.dumps(payloads[url]).encode("utf-8"),
        )
        result = run_scaffold(t, ExecutorDeps(
            provider=_StubProv(), store=store,
            extra={"pypi_client": client}))
    assert result.failure is None
    pin_adjustments = result.artifact.content["pin_adjustments"]
    assert pin_adjustments and pin_adjustments[0]["peer"] == "werkzeug"
    assert pin_adjustments[0]["after"] == "Werkzeug<3,>=2.0"
    req_diff = next(d for d in result.artifact.content["diffs"]
                    if d["file"] == "requirements.txt")
    assert "Werkzeug<3,>=2.0" in req_diff["patch"]
    assert result.outputs["pin_adjustments_count"] == 1



# ------------- Phase 3.3: first-party import cross-check -------------

def test_cross_check_passes_when_imported_symbol_exists():
    """A ``from <local> import <name>`` that resolves -> no warnings."""
    from cgx.session.scaffold_validate import cross_check_first_party_imports
    contents = {
        "backend/auth.py": "def login():\n    return True\n",
        "backend/app.py": "from backend.auth import login\n",
    }
    assert cross_check_first_party_imports(contents) == []


def test_cross_check_flags_missing_first_party_symbol():
    """Importing a name absent from a generated module -> one warning."""
    from cgx.session.scaffold_validate import cross_check_first_party_imports
    contents = {
        "backend/auth.py": "def login():\n    return True\n",
        "backend/app.py": "from backend.auth import logout\n",
    }
    warnings = cross_check_first_party_imports(contents)
    assert len(warnings) == 1
    w = warnings[0]
    assert w["file"] == "backend/app.py"
    assert w["module"] == "backend.auth"
    assert w["name"] == "logout"


def test_cross_check_resolves_relative_imports():
    """``from . import name`` resolves against the importer's package."""
    from cgx.session.scaffold_validate import cross_check_first_party_imports
    ok = {
        "pkg/__init__.py": "",
        "pkg/helpers.py": "VALUE = 1\n",
        "pkg/app.py": "from .helpers import VALUE\n",
    }
    assert cross_check_first_party_imports(ok) == []
    bad = {
        "pkg/__init__.py": "",
        "pkg/helpers.py": "VALUE = 1\n",
        "pkg/app.py": "from .helpers import MISSING\n",
    }
    warnings = cross_check_first_party_imports(bad)
    assert [w["name"] for w in warnings] == ["MISSING"]


def test_cross_check_ignores_third_party_and_submodule_imports():
    """Third-party modules and generated submodules are never flagged."""
    from cgx.session.scaffold_validate import cross_check_first_party_imports
    contents = {
        "backend/__init__.py": "",
        "backend/models.py": "class User:\n    pass\n",
        "backend/app.py": (
            "from fastapi import FastAPI\n"
            "from backend import models\n"),
    }
    # ``FastAPI`` is third-party (no generated source) and ``models`` is a
    # generated submodule of ``backend`` -> both abstain.
    assert cross_check_first_party_imports(contents) == []


def test_cross_check_abstains_on_unparseable_target():
    """A target module that fails to parse -> no false positive."""
    from cgx.session.scaffold_validate import cross_check_first_party_imports
    contents = {
        "backend/auth.py": "def login(:\n",  # syntax error
        "backend/app.py": "from backend.auth import login\n",
    }
    assert cross_check_first_party_imports(contents) == []


def test_scaffold_executor_surfaces_import_warnings(store, monkeypatch):
    """SCAFFOLD runs the cross-check and surfaces ``import_warnings``."""
    from cgx.session.tasks.scaffold import run_scaffold

    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    plan = Artifact.new(
        session.session_id, "task_x", ArtifactKind.WORK_PLAN, {
            "prior_goal": "g",
            "composed_goal": "build api",
            "answers": {},
            "plan_md": "",
            "layers": [{"name": "app", "files": [
                {"path": "backend/auth.py", "description": "auth"},
                {"path": "backend/app.py", "description": "entry"}]}],
        })
    store.save_artifact(plan)

    contents = {
        "backend/auth.py": "def login():\n    return True\n",
        "backend/app.py": "from backend.auth import logout\n",
    }

    def fake_generate(path, *a, **kw):
        body = contents[path]
        patch = (f"--- /dev/null\n+++ b/{path}\n"
                 f"@@ -0,0 +1,{len(body.splitlines())} @@\n"
                 + "\n".join(f"+{ln}" for ln in body.splitlines()))
        return {"file": path, "patch": patch, "content": body,
                "syntax_ok": True, "confidence": 1.0}

    monkeypatch.setattr("cgx.answer.engine.generate_single_scaffold_file",
                        fake_generate)

    class _StubProv:
        def chat(self, *a, **kw):
            return ""

    t = TaskNode.new(session.session_id, TaskKind.SCAFFOLD, "s",
                     inputs={"work_plan_artifact_id": plan.artifact_id})
    store.save_task(t)
    result = run_scaffold(t, ExecutorDeps(provider=_StubProv(), store=store))
    assert result.failure is None
    warnings = result.artifact.content["import_warnings"]
    assert len(warnings) == 1
    assert warnings[0]["name"] == "logout"
    assert warnings[0]["module"] == "backend.auth"
    assert result.outputs["import_warnings_count"] == 1



# --------------------- Phase 5.1: LLM call tracing -------------------------

class _StubChatProvider:
    """Minimal LLMProvider used by the tracing tests."""

    def __init__(self, model: str = "stub-model",
                 reply: str = '{"ok": true}') -> None:
        self.model = model
        self._reply = reply
        self.calls: list = []

    def chat(self, messages, temperature=0.2, max_tokens=None,
             force_json=True, **kwargs):
        self.calls.append({"messages": messages, "temperature": temperature,
                           "max_tokens": max_tokens, "force_json": force_json,
                           "kwargs": kwargs})
        return {"content": self._reply}

    def chat_stream(self, messages, temperature=0.2, max_tokens=None,
                    **kwargs):
        yield self._reply


def test_tracing_provider_records_chat_as_fact():
    """Bound TracingProvider buffers one LLM_CALL fact per chat call."""
    from cgx.session.llm_trace import TracingProvider
    inner = _StubChatProvider(model="stub-1", reply="hi")
    tp = TracingProvider(inner)
    tp.bind("sess-A", "task-X")
    out = tp.chat([{"role": "user", "content": "ping"}],
                  temperature=0.7, max_tokens=64, force_json=False)
    assert out == {"content": "hi"}
    drained = tp.drain()
    assert len(drained) == 1
    fact = drained[0]
    assert fact.kind is FactKind.LLM_CALL
    assert fact.surfaced_in_task_id == "task-X"
    assert fact.session_id == "sess-A"
    assert fact.content["model"] == "stub-1"
    assert "ping" in fact.content["prompt"]
    assert fact.content["response"] == "hi"
    assert fact.content["sampling"]["temperature"] == 0.7
    assert fact.content["sampling"]["max_tokens"] == 64
    assert fact.content["sampling"]["force_json"] is False
    # drain clears the buffer.
    assert tp.drain() == []


def test_tracing_provider_records_streamed_response():
    """chat_stream accumulates deltas into the response fact."""
    from cgx.session.llm_trace import TracingProvider

    class _Streamer(_StubChatProvider):
        def chat_stream(self, messages, **kw):
            yield "hel"
            yield "lo"

    tp = TracingProvider(_Streamer(model="m"))
    tp.bind("s", "t")
    chunks = list(tp.chat_stream([{"role": "user", "content": "hi"}]))
    assert chunks == ["hel", "lo"]
    facts = tp.drain()
    assert len(facts) == 1 and facts[0].content["response"] == "hello"
    assert facts[0].content["streamed"] is True


def test_tracing_provider_records_chat_error():
    """A raised exception still produces an LLM_CALL fact with ``error``."""
    from cgx.session.llm_trace import TracingProvider

    class _Boom(_StubChatProvider):
        def chat(self, *a, **kw):
            raise RuntimeError("model down")

    tp = TracingProvider(_Boom())
    tp.bind("s", "t")
    with pytest.raises(RuntimeError):
        tp.chat([{"role": "user", "content": "x"}])
    facts = tp.drain()
    assert facts and facts[0].content["error"].startswith("RuntimeError")


def test_tracing_provider_unbound_calls_are_silent():
    """Calls made outside a bind/unbind window emit no facts."""
    from cgx.session.llm_trace import TracingProvider
    tp = TracingProvider(_StubChatProvider())
    tp.chat([{"role": "user", "content": "x"}])
    assert tp.drain() == []


def test_runner_persists_llm_call_facts_via_tracing(tmp_path):
    """run_next bind/drains the tracer; LLM_CALL facts land in the store."""
    from cgx.session import SessionRunner, SessionStore
    from cgx.session.llm_trace import TracingProvider
    from cgx.session.tasks.base import register_executor

    store = SessionStore(db_path=":memory:", project_root=str(tmp_path))
    runner = SessionRunner(store)
    session = Session.new(
        "trace", mode=SessionMode.EXPLORE, project_root=str(tmp_path))
    store.save_session(session)
    task = TaskNode.new(session.session_id, TaskKind.EXPLORE, "n", inputs={})
    task.status = TaskNodeStatus.READY
    store.save_task(task)

    @register_executor(TaskKind.EXPLORE)
    def _exec(t, deps):
        # Touch the provider so the tracer records a Fact.
        deps.provider.chat([{"role": "user", "content": "probe"}])
        return ExecutorResult(outputs={})

    try:
        provider = TracingProvider(_StubChatProvider(model="m"))
        result = runner.run_next(
            session_id=session.session_id,
            deps=ExecutorDeps(provider=provider, store=store,
                              project_root=str(tmp_path)))
        assert result is not None
        kb = store.load_kb(session.session_id)
        llm_facts = kb.of_kind(FactKind.LLM_CALL)
        assert len(llm_facts) == 1
        assert llm_facts[0].surfaced_in_task_id == task.task_id
        assert llm_facts[0].content["model"] == "m"
        assert "probe" in llm_facts[0].content["prompt"]
    finally:
        # Restore baseline EXPLORE executor for the rest of the suite.
        from cgx.session.tasks.explore import run_explore  # noqa: F401



# --------------------- Phase 6.1: branching repair (patch vs regenerate) ---

def test_select_repair_strategy_patches_small_diff_list():
    """Diffs <= _PATCH_DIFF_LIMIT keep the patch branch."""
    from cgx.session.tasks.repair import (
        _PATCH_DIFF_LIMIT,
        _select_repair_strategy,
    )
    diffs = [{"file": f"f{i}.py", "patch": "..."}
             for i in range(_PATCH_DIFF_LIMIT)]
    strategy, constraints = _select_repair_strategy(
        classification="unittest_pytest_mix", diffs=diffs,
        rationale="r", extra_plan_fields={}, locations_payload=[])
    assert strategy == "patch"
    assert constraints == {}


def test_select_repair_strategy_regenerates_when_no_diffs_and_unknown():
    """Empty diff list + regenerate-eligible class -> regenerate verdict."""
    from cgx.session.tasks.repair import _select_repair_strategy
    strategy, constraints = _select_repair_strategy(
        classification="unknown", diffs=[],
        rationale="rationale text",
        extra_plan_fields={}, locations_payload=[])
    assert strategy == "regenerate"
    assert constraints["kind"] == "unknown"
    assert constraints["rationale"] == "rationale text"


def test_select_repair_strategy_regenerates_missing_fixture_without_diffs():
    """missing_fixture with no locatable source fixture -> regenerate.

    When the required fixture was never authored anywhere the proposer
    cannot hoist it into conftest, so it emits zero diffs. Rather than
    dead-ending, that class re-scaffolds the offending test layer.
    """
    from cgx.session.tasks.repair import _select_repair_strategy
    strategy, constraints = _select_repair_strategy(
        classification="missing_fixture", diffs=[],
        rationale="fixture 'username' is never defined",
        extra_plan_fields={}, locations_payload=[])
    assert strategy == "regenerate"
    assert constraints["kind"] == "missing_fixture"


def test_select_repair_strategy_regenerates_when_patch_oversized():
    """Diff list above the limit forces regenerate with oversized hint."""
    from cgx.session.tasks.repair import (
        _PATCH_DIFF_LIMIT,
        _select_repair_strategy,
    )
    diffs = [{"file": f"f{i}.py", "patch": "..."}
             for i in range(_PATCH_DIFF_LIMIT + 2)]
    locations = [{"class_name": "TestX"}, {"class_name": "TestY"},
                 {"class_name": "TestX"}]
    strategy, constraints = _select_repair_strategy(
        classification="unittest_pytest_mix", diffs=diffs,
        rationale="oversized", extra_plan_fields={},
        locations_payload=locations)
    assert strategy == "regenerate"
    assert constraints["kind"] == "unittest_pytest_mix"
    assert constraints["affected_classes"] == ["TestX", "TestY"]
    assert constraints["oversized_patch"]["diff_count"] == _PATCH_DIFF_LIMIT + 2
    assert constraints["oversized_patch"]["limit"] == _PATCH_DIFF_LIMIT


def test_select_repair_strategy_patches_unknown_when_diffs_exist():
    """Regenerate-eligible class but with usable diffs stays on patch."""
    from cgx.session.tasks.repair import _select_repair_strategy
    strategy, constraints = _select_repair_strategy(
        classification="unknown",
        diffs=[{"file": "a.py", "patch": "..."}],
        rationale="r", extra_plan_fields={}, locations_payload=[])
    assert strategy == "patch"
    assert constraints == {}


def test_propose_regenerate_increments_attempt_and_accumulates_constraints():
    """propose_regenerate clones SCAFFOLD with bumped attempt + appended payload."""
    from cgx.session.repair.propose import propose_regenerate
    parent_id = "task-decompose"
    scaffold = TaskNode.new(
        "sess-A", TaskKind.SCAFFOLD, "scaffold",
        parent_task_id=parent_id,
        inputs={
            "work_plan_artifact_id": "art_plan",
            "regenerate_attempt": 0,
            "regenerate_constraints": [
                {"kind": "prior", "rationale": "old"},
            ],
        })
    fresh = propose_regenerate(
        scaffold,
        {"kind": "unittest_pytest_mix", "rationale": "mix",
         "affected_classes": ["TestX"]})
    assert fresh.kind is TaskKind.SCAFFOLD
    assert fresh.session_id == scaffold.session_id
    assert fresh.parent_task_id == parent_id
    assert fresh.task_id != scaffold.task_id
    assert fresh.inputs["regenerate_attempt"] == 1
    assert fresh.inputs["regenerated_from_task_id"] == scaffold.task_id
    assert fresh.inputs["work_plan_artifact_id"] == "art_plan"
    constraints = fresh.inputs["regenerate_constraints"]
    assert len(constraints) == 2
    assert constraints[0] == {"kind": "prior", "rationale": "old"}
    assert constraints[1]["kind"] == "unittest_pytest_mix"
    # Original task's inputs are unaffected (no aliasing).
    assert scaffold.inputs["regenerate_attempt"] == 0
    assert len(scaffold.inputs["regenerate_constraints"]) == 1


def _build_regenerate_chain(*, prior_regens: int = 0,
                            extra_descendants: bool = True):
    """Build a SCAFFOLD -> APPLY -> VERIFY -> REPAIR(regenerate) chain."""
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    scaffold = TaskNode.new(
        session.session_id, TaskKind.SCAFFOLD, "scaffold",
        inputs={"work_plan_artifact_id": "art_plan",
                "regenerate_attempt": prior_regens})
    scaffold.status = TaskNodeStatus.DONE
    apply_t = TaskNode.new(
        session.session_id, TaskKind.APPLY, "apply",
        parent_task_id=scaffold.task_id, inputs={})
    apply_t.status = TaskNodeStatus.DONE
    verify = TaskNode.new(
        session.session_id, TaskKind.VERIFY, "verify",
        parent_task_id=apply_t.task_id, inputs={})
    verify.status = TaskNodeStatus.DONE
    rep = TaskNode.new(
        session.session_id, TaskKind.REPAIR, "repair",
        parent_task_id=verify.task_id,
        inputs={"verify_artifact_id": "art_verify",
                "mode": SessionMode.GREENFIELD.value,
                "repair_attempt": 1})
    rep.produced_artifact_id = "art_repair_plan"
    rep.outputs = {
        "can_apply": False, "classification": "unittest_pytest_mix",
        "failure_signature": "sig", "repair_attempt": 1, "diff_count": 0,
        "strategy": "regenerate",
        "extra_constraints": {"kind": "unittest_pytest_mix",
                              "rationale": "mix",
                              "affected_classes": ["TestX"]},
    }
    rep.status = TaskNodeStatus.DONE
    tasks = [scaffold, apply_t, verify, rep]
    if extra_descendants:
        # A second pending APPLY off the same SCAFFOLD must also be abandoned.
        pending = TaskNode.new(
            session.session_id, TaskKind.APPLY, "apply-pending",
            parent_task_id=scaffold.task_id, inputs={})
        pending.status = TaskNodeStatus.READY
        tasks.append(pending)
    return session, scaffold, tasks, rep


def test_router_repair_regenerate_abandons_subtree_and_requeues_scaffold():
    """Regenerate verdict -> abandon descendants + create fresh SCAFFOLD."""
    session, scaffold, tasks, _rep = _build_regenerate_chain()
    plan = Router().on_task_completed(
        session=session, completed=tasks[-2], tasks=tasks)
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    abandons = [a for a in plan.actions
                if isinstance(a, UpdateTaskStatus)
                and a.status is TaskNodeStatus.ABANDONED]
    # Exactly one new SCAFFOLD spawned.
    assert len(creates) == 1
    new_scaffold = creates[0].task
    assert new_scaffold.kind is TaskKind.SCAFFOLD
    assert new_scaffold.parent_task_id == scaffold.parent_task_id
    assert new_scaffold.inputs["regenerate_attempt"] == 1
    assert new_scaffold.inputs["regenerated_from_task_id"] == scaffold.task_id
    payloads = new_scaffold.inputs["regenerate_constraints"]
    assert payloads and payloads[0]["kind"] == "unittest_pytest_mix"
    # Only the live pending child (not DONE descendants) is abandoned.
    abandoned_ids = {a.task_id for a in abandons}
    pending = [t for t in tasks if t.status is TaskNodeStatus.READY]
    assert {p.task_id for p in pending} <= abandoned_ids
    # DONE tasks must not be abandoned.
    done_ids = {t.task_id for t in tasks if t.status is TaskNodeStatus.DONE}
    assert abandoned_ids.isdisjoint(done_ids)


def test_router_repair_regenerate_budget_exhausted_fails_session():
    """Once the regenerate budget is hit the session fails terminally."""
    from cgx.session.router import _REGENERATE_BUDGET
    session, _scaffold, tasks, rep = _build_regenerate_chain(
        prior_regens=_REGENERATE_BUDGET)
    plan = Router().on_task_completed(
        session=session, completed=rep, tasks=tasks)
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    # Fallback path: no new SCAFFOLD and no ASK_USER -- the empty-diff
    # REPAIR fails the session terminally instead.
    assert creates == []
    abandons = [a for a in plan.actions
                if isinstance(a, UpdateTaskStatus)
                and a.status is TaskNodeStatus.ABANDONED]
    assert abandons == []
    status = [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]
    assert len(status) == 1
    assert status[0].status is SessionStatus.FAILED


def test_router_repair_regenerate_without_scaffold_ancestor_fails_session():
    """REPAIR with no SCAFFOLD on the ancestor chain -> terminal FAILED."""
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    # No SCAFFOLD anywhere -- REPAIR sits directly under a synthetic
    # VERIFY parent.
    verify = TaskNode.new(
        session.session_id, TaskKind.VERIFY, "verify", inputs={})
    verify.status = TaskNodeStatus.DONE
    rep = TaskNode.new(
        session.session_id, TaskKind.REPAIR, "repair",
        parent_task_id=verify.task_id,
        inputs={"verify_artifact_id": "art_verify",
                "mode": SessionMode.GREENFIELD.value,
                "repair_attempt": 1})
    rep.produced_artifact_id = "art_repair_plan"
    rep.outputs = {
        "can_apply": False, "classification": "unknown",
        "failure_signature": "sig", "diff_count": 0,
        "strategy": "regenerate", "extra_constraints": {"kind": "unknown"},
    }
    rep.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=rep, tasks=[verify, rep])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert creates == []
    status = [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]
    assert len(status) == 1
    assert status[0].status is SessionStatus.FAILED


def _build_apply_failed_chain(*, prior_regens: int = 0,
                              prior_replans: int = 0,
                              mode: str = "greenfield",
                              with_scaffold: bool = True,
                              survivors: int = 8):
    """Build a SCAFFOLD -> APPLY chain where APPLY dropped invalid files."""
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=(SessionMode.GREENFIELD
                                     if mode == "greenfield"
                                     else SessionMode.EXPLORE))
    tasks: List[TaskNode] = []
    parent_id = None
    if with_scaffold:
        scaffold = TaskNode.new(
            session.session_id, TaskKind.SCAFFOLD, "scaffold",
            inputs={"work_plan_artifact_id": "art_plan",
                    "prior_goal": "build flask api",
                    "regenerate_attempt": prior_regens,
                    "replan_attempt": prior_replans})
        scaffold.outputs = {"scaffold_artifact_id": "art_scaffold",
                            "generated_count": survivors, "failed_count": 1,
                            "failed": [{"file": "backend/main.py",
                                        "error": "generator returned empty "
                                                 "patch"}]}
        scaffold.status = TaskNodeStatus.DONE
        tasks.append(scaffold)
        parent_id = scaffold.task_id
    else:
        scaffold = None
    apply_t = TaskNode.new(
        session.session_id, TaskKind.APPLY, "apply",
        parent_task_id=parent_id,
        inputs={"mode": mode, "scaffold_artifact_id": "art_scaffold"})
    apply_t.outputs = {
        "apply_artifact_id": "art_applied",
        "applied_count": 1, "failed_count": 2,
        "failed_files": [
            {"file": "backend/models.py",
             "error": "python syntax: unexpected unindent (models.py, line 10)"},
            {"file": "tests/test_auth.py",
             "error": "python syntax: unexpected unindent (test_auth.py, "
                      "line 19)"}]}
    apply_t.status = TaskNodeStatus.DONE
    tasks.append(apply_t)
    return session, scaffold, apply_t, tasks


def test_router_apply_failed_files_regenerates_within_budget():
    """Greenfield APPLY with dropped files -> abandon subtree + re-scaffold."""
    session, scaffold, apply_t, tasks = _build_apply_failed_chain()
    plan = Router().on_task_completed(
        session=session, completed=apply_t, tasks=tasks)
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    new_scaffold = creates[0].task
    assert new_scaffold.kind is TaskKind.SCAFFOLD
    assert new_scaffold.parent_task_id == scaffold.parent_task_id
    assert new_scaffold.inputs["regenerate_attempt"] == 1
    payloads = new_scaffold.inputs["regenerate_constraints"]
    assert payloads and payloads[0]["kind"] == "invalid_scaffold_syntax"
    # The constraint must enumerate the concrete per-file failures from
    # both the SCAFFOLD (empty patch) and APPLY (syntax) so the retry has
    # actionable feedback rather than a bare count.
    rationale = payloads[0]["rationale"]
    assert "backend/models.py" in rationale
    assert "unexpected unindent" in rationale
    assert "tests/test_auth.py" in rationale
    assert "backend/main.py" in rationale
    # The structured list covers all three dropped files (deduped by path).
    listed = payloads[0]["failed_files"]
    assert len(listed) == 3
    assert any("backend/main.py" in e for e in listed)
    assert any("backend/models.py" in e for e in listed)
    assert any("tests/test_auth.py" in e for e in listed)
    # The session must not be failed while a regenerate is still possible.
    assert not [a for a in plan.actions
                if isinstance(a, UpdateSessionStatus)]
    # B2: the regenerate is targeted -- it names exactly the dropped files
    # (SCAFFOLD's empty patch + APPLY's two syntax failures) and points at
    # the prior SCAFFOLD_PATCHES artifact so its good diffs are reused.
    assert set(new_scaffold.inputs["regenerate_files"]) == {
        "backend/main.py", "backend/models.py", "tests/test_auth.py"}
    assert new_scaffold.inputs["prior_scaffold_artifact_id"] == "art_scaffold"


def test_router_apply_failed_files_budget_exhausted_escalates_to_replan():
    """C2: regenerate budget spent -> escalate once to a fresh DECOMPOSE."""
    from cgx.session.router import _REGENERATE_BUDGET
    session, scaffold, apply_t, tasks = _build_apply_failed_chain(
        prior_regens=_REGENERATE_BUDGET)
    plan = Router().on_task_completed(
        session=session, completed=apply_t, tasks=tasks)
    # No terminal failure yet -- the manifest is re-planned first.
    assert not [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    dec = creates[0].task
    assert dec.kind is TaskKind.DECOMPOSE
    assert dec.parent_task_id == scaffold.task_id
    assert dec.inputs["replan_attempt"] == 1
    # The revised goal keeps the objective and folds in the failure note.
    assert "build flask api" in dec.inputs["prior_goal"]
    assert "backend/models.py" in dec.inputs["prior_goal"]
    # The DONE APPLY is not abandoned (only live descendants would be).
    assert not [a for a in plan.actions
                if isinstance(a, UpdateTaskStatus)
                and a.status is TaskNodeStatus.ABANDONED]


def test_router_apply_failed_files_replan_budget_exhausted_proceeds_with_survivors():
    """B: budgets spent but survivors exist -> proceed on the normal
    APPLY -> BOOTSTRAP_ENV edge instead of discarding the run."""
    from cgx.session.router import _REGENERATE_BUDGET, _REPLAN_BUDGET
    session, _scaffold, apply_t, tasks = _build_apply_failed_chain(
        prior_regens=_REGENERATE_BUDGET, prior_replans=_REPLAN_BUDGET)
    plan = Router().on_task_completed(
        session=session, completed=apply_t, tasks=tasks)
    # No terminal failure: the successfully applied files carry forward.
    assert not [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    assert creates[0].task.kind is TaskKind.BOOTSTRAP_ENV


def test_router_apply_failed_files_replan_budget_exhausted_no_survivors_fails():
    """B: budgets spent and nothing generated cleanly -> terminal FAILED."""
    from cgx.session.router import _REGENERATE_BUDGET, _REPLAN_BUDGET
    session, _scaffold, apply_t, tasks = _build_apply_failed_chain(
        prior_regens=_REGENERATE_BUDGET, prior_replans=_REPLAN_BUDGET,
        survivors=0)
    plan = Router().on_task_completed(
        session=session, completed=apply_t, tasks=tasks)
    assert [a for a in plan.actions if isinstance(a, CreateTask)] == []
    status = [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]
    assert len(status) == 1
    assert status[0].status is SessionStatus.FAILED


def test_router_apply_failed_files_no_scaffold_ancestor_fails_session():
    """Greenfield APPLY with dropped files but no SCAFFOLD -> terminal FAILED."""
    session, _scaffold, apply_t, tasks = _build_apply_failed_chain(
        with_scaffold=False)
    plan = Router().on_task_completed(
        session=session, completed=apply_t, tasks=tasks)
    assert [a for a in plan.actions if isinstance(a, CreateTask)] == []
    status = [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]
    assert len(status) == 1
    assert status[0].status is SessionStatus.FAILED


def test_router_apply_failed_files_explore_mode_proceeds_normally():
    """Explore-mode APPLY keeps its APPLY -> VERIFY edge despite failed files."""
    session, _scaffold, apply_t, tasks = _build_apply_failed_chain(
        mode="explore", with_scaffold=False)
    plan = Router().on_task_completed(
        session=session, completed=apply_t, tasks=tasks)
    assert not [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    assert creates[0].task.kind is TaskKind.VERIFY


def _build_scaffold_failed_chain(*, prior_regens: int = 0,
                                 prior_replans: int = 0,
                                 failed_count: int = 2,
                                 with_pending_child: bool = True,
                                 survivors: int = 8):
    """Build a SCAFFOLD that dropped files, with an optional live APPLY child.

    Mirrors :func:`_build_apply_failed_chain` but the failure surfaces on
    the SCAFFOLD itself (LLM timeout / empty patch) while its surviving
    diffs would otherwise apply cleanly.
    """
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    parent = TaskNode.new(
        session.session_id, TaskKind.ASK_USER, "approve",
        inputs={"work_plan_artifact_id": "art_plan"})
    parent.status = TaskNodeStatus.DONE
    scaffold = TaskNode.new(
        session.session_id, TaskKind.SCAFFOLD, "scaffold",
        parent_task_id=parent.task_id,
        inputs={"work_plan_artifact_id": "art_plan",
                "prior_goal": "build a react calculator",
                "regenerate_attempt": prior_regens,
                "replan_attempt": prior_replans})
    scaffold.produced_artifact_id = "art_scaffold"
    failed = [
        {"file": "src/components/Calculator.jsx",
         "error": "ReadTimeout: read timed out (read timeout=300.0)"},
        {"file": "backend/main.py",
         "error": "generator returned empty patch"}][:failed_count]
    scaffold.outputs = {"scaffold_artifact_id": "art_scaffold",
                        "generated_count": survivors,
                        "failed_count": failed_count, "failed": failed}
    scaffold.status = TaskNodeStatus.DONE
    tasks = [parent, scaffold]
    if with_pending_child:
        apply_t = TaskNode.new(
            session.session_id, TaskKind.APPLY, "apply",
            parent_task_id=scaffold.task_id,
            inputs={"mode": "greenfield",
                    "scaffold_artifact_id": "art_scaffold"})
        apply_t.status = TaskNodeStatus.READY
        tasks.append(apply_t)
    return session, parent, scaffold, tasks


def test_router_scaffold_failed_files_regenerates_within_budget():
    """SCAFFOLD that dropped files -> abandon live subtree + re-scaffold."""
    session, parent, scaffold, tasks = _build_scaffold_failed_chain()
    plan = Router().on_task_completed(
        session=session, completed=scaffold, tasks=tasks)
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    new_scaffold = creates[0].task
    assert new_scaffold.kind is TaskKind.SCAFFOLD
    assert new_scaffold.parent_task_id == parent.task_id
    assert new_scaffold.inputs["regenerate_attempt"] == 1
    assert new_scaffold.inputs["regenerated_from_task_id"] == scaffold.task_id
    payloads = new_scaffold.inputs["regenerate_constraints"]
    assert payloads and payloads[0]["kind"] == "invalid_scaffold_syntax"
    # The constraint enumerates the concrete per-file SCAFFOLD failures
    # (timeout + empty patch) so the retry has actionable feedback.
    rationale = payloads[0]["rationale"]
    assert "src/components/Calculator.jsx" in rationale
    assert "read timed out" in rationale
    assert "backend/main.py" in rationale
    listed = payloads[0]["failed_files"]
    assert len(listed) == 2
    # The live APPLY child is abandoned so the stale subtree cannot run.
    abandoned = {a.task_id for a in plan.actions
                 if isinstance(a, UpdateTaskStatus)}
    pending = [t for t in tasks if t.status is TaskNodeStatus.READY]
    assert {p.task_id for p in pending} <= abandoned
    # No terminal failure while a regenerate is still possible.
    assert not [a for a in plan.actions
                if isinstance(a, UpdateSessionStatus)]
    # B2: the regenerate is targeted -- exactly the two dropped SCAFFOLD
    # files, pointed at the prior SCAFFOLD_PATCHES artifact for reuse.
    assert set(new_scaffold.inputs["regenerate_files"]) == {
        "src/components/Calculator.jsx", "backend/main.py"}
    assert new_scaffold.inputs["prior_scaffold_artifact_id"] == "art_scaffold"


def test_router_scaffold_failed_files_budget_exhausted_escalates_to_replan():
    """C2: SCAFFOLD regenerate budget spent -> escalate to a fresh DECOMPOSE."""
    from cgx.session.router import _REGENERATE_BUDGET
    session, _parent, scaffold, tasks = _build_scaffold_failed_chain(
        prior_regens=_REGENERATE_BUDGET)
    plan = Router().on_task_completed(
        session=session, completed=scaffold, tasks=tasks)
    assert not [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    dec = creates[0].task
    assert dec.kind is TaskKind.DECOMPOSE
    assert dec.parent_task_id == scaffold.task_id
    assert dec.inputs["replan_attempt"] == 1
    assert "build a react calculator" in dec.inputs["prior_goal"]
    assert "Calculator.jsx" in dec.inputs["prior_goal"]
    # The live APPLY descendant is abandoned so the stale subtree cannot run.
    abandoned = {a.task_id for a in plan.actions
                 if isinstance(a, UpdateTaskStatus)
                 and a.status is TaskNodeStatus.ABANDONED}
    pending = [t for t in tasks if t.status is TaskNodeStatus.READY]
    assert {p.task_id for p in pending} <= abandoned


def test_router_scaffold_failed_files_replan_budget_exhausted_proceeds_with_survivors():
    """B: budgets spent but survivors exist -> proceed on the normal
    SCAFFOLD -> APPLY edge instead of discarding the run."""
    from cgx.session.router import _REGENERATE_BUDGET, _REPLAN_BUDGET
    session, _parent, scaffold, tasks = _build_scaffold_failed_chain(
        prior_regens=_REGENERATE_BUDGET, prior_replans=_REPLAN_BUDGET,
        with_pending_child=False)
    plan = Router().on_task_completed(
        session=session, completed=scaffold, tasks=tasks)
    assert not [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    assert creates[0].task.kind is TaskKind.APPLY


def test_router_scaffold_failed_files_replan_budget_exhausted_no_survivors_fails():
    """B: budgets spent and nothing generated cleanly -> terminal FAILED."""
    from cgx.session.router import _REGENERATE_BUDGET, _REPLAN_BUDGET
    session, _parent, scaffold, tasks = _build_scaffold_failed_chain(
        prior_regens=_REGENERATE_BUDGET, prior_replans=_REPLAN_BUDGET,
        with_pending_child=False, survivors=0)
    plan = Router().on_task_completed(
        session=session, completed=scaffold, tasks=tasks)
    assert [a for a in plan.actions if isinstance(a, CreateTask)] == []
    status = [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]
    assert len(status) == 1
    assert status[0].status is SessionStatus.FAILED


def test_router_replan_attempt_threads_decompose_to_scaffold():
    """C2: replan_attempt propagates DECOMPOSE -> ASK(APPROVE_PLAN) -> SCAFFOLD."""
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    dec = TaskNode.new(
        session.session_id, TaskKind.DECOMPOSE, "revise",
        inputs={"prior_goal": "g", "requirements_artifact_id": "art_req",
                "replan_attempt": 1})
    dec.produced_artifact_id = "art_plan2"
    dec.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=dec, tasks=[dec])
    ask = [a.task for a in plan.actions
           if isinstance(a, CreateTask)][0]
    assert ask.kind is TaskKind.ASK_USER
    assert ask.inputs["expected_kind"] == "approve_plan"
    assert ask.inputs["replan_attempt"] == 1
    # Approving the revised plan carries replan_attempt onto the SCAFFOLD.
    decision = Decision.new(
        session.session_id, ask.task_id, DecisionKind.APPROVE_PLAN,
        "approve plan", {"approved": True})
    plan2 = Router().on_decision_recorded(
        session=session, decision=decision, tasks=[ask])
    sc = [a.task for a in plan2.actions if isinstance(a, CreateTask)][0]
    assert sc.kind is TaskKind.SCAFFOLD
    assert sc.inputs["replan_attempt"] == 1


def test_router_scaffold_clean_still_spawns_apply():
    """A SCAFFOLD with no dropped files keeps its SCAFFOLD -> APPLY edge."""
    session, _parent, scaffold, tasks = _build_scaffold_failed_chain(
        failed_count=0, with_pending_child=False)
    scaffold.outputs = {"scaffold_artifact_id": "art_scaffold",
                        "generated_count": 8, "failed_count": 0,
                        "failed": []}
    plan = Router().on_task_completed(
        session=session, completed=scaffold, tasks=tasks)
    assert not [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    assert creates[0].task.kind is TaskKind.APPLY


def _build_crashed_scaffold(*, prior_regens: int = 0,
                            with_pending_child: bool = True):
    """Build a SCAFFOLD that crashed mid-run (hard failure, no outputs).

    Mirrors :func:`_build_scaffold_failed_chain` but the SCAFFOLD is
    ``FAILED`` with no ``outputs`` (an LLM timeout / process kill), which
    is the state :meth:`Router.on_task_failed` sees for a crash.
    """
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    parent = TaskNode.new(
        session.session_id, TaskKind.ASK_USER, "approve",
        inputs={"work_plan_artifact_id": "art_plan"})
    parent.status = TaskNodeStatus.DONE
    scaffold = TaskNode.new(
        session.session_id, TaskKind.SCAFFOLD, "scaffold",
        parent_task_id=parent.task_id,
        inputs={"work_plan_artifact_id": "art_plan",
                "regenerate_attempt": prior_regens})
    scaffold.status = TaskNodeStatus.FAILED
    tasks = [parent, scaffold]
    if with_pending_child:
        child = TaskNode.new(
            session.session_id, TaskKind.APPLY, "apply",
            parent_task_id=scaffold.task_id,
            inputs={"mode": "greenfield"})
        child.status = TaskNodeStatus.READY
        tasks.append(child)
    return session, parent, scaffold, tasks


def test_router_on_task_failed_scaffold_resumes_from_checkpoint():
    """B4: a crashed SCAFFOLD with a checkpoint re-queues a resuming SCAFFOLD."""
    session, parent, scaffold, tasks = _build_crashed_scaffold()
    plan = Router().on_task_failed(
        session=session, failed=scaffold, tasks=tasks,
        resume_scaffold_artifact_id="art_ckpt")
    # No terminal failure while a resume is still possible.
    assert not [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    new_scaffold = creates[0].task
    assert new_scaffold.kind is TaskKind.SCAFFOLD
    assert new_scaffold.parent_task_id == parent.task_id
    assert new_scaffold.inputs["resume_scaffold_artifact_id"] == "art_ckpt"
    assert new_scaffold.inputs["regenerate_attempt"] == 1
    assert new_scaffold.inputs["regenerated_from_task_id"] == scaffold.task_id
    # The live APPLY child is abandoned so the stale subtree cannot run.
    abandoned = {a.task_id for a in plan.actions
                 if isinstance(a, UpdateTaskStatus)}
    pending = [t for t in tasks if t.status is TaskNodeStatus.READY]
    assert {p.task_id for p in pending} <= abandoned


def test_router_on_task_failed_scaffold_resume_budget_exhausted_fails():
    """B4: a re-crashed SCAFFOLD with no budget left fails the session."""
    from cgx.session.router import _REGENERATE_BUDGET
    session, _parent, scaffold, tasks = _build_crashed_scaffold(
        prior_regens=_REGENERATE_BUDGET, with_pending_child=False)
    plan = Router().on_task_failed(
        session=session, failed=scaffold, tasks=tasks,
        resume_scaffold_artifact_id="art_ckpt")
    assert [a for a in plan.actions if isinstance(a, CreateTask)] == []
    status = [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]
    assert len(status) == 1
    assert status[0].status is SessionStatus.FAILED


def test_router_on_task_failed_scaffold_without_checkpoint_fails():
    """B4: a crash with nothing checkpointed keeps the terminal-FAILED path."""
    session, _parent, scaffold, tasks = _build_crashed_scaffold(
        with_pending_child=False)
    plan = Router().on_task_failed(
        session=session, failed=scaffold, tasks=tasks,
        resume_scaffold_artifact_id=None)
    assert [a for a in plan.actions if isinstance(a, CreateTask)] == []
    status = [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]
    assert len(status) == 1
    assert status[0].status is SessionStatus.FAILED


def test_runner_scaffold_crash_resumes_from_checkpoint(tmp_path, store):
    """B4 end-to-end: the runner resolves a crashed SCAFFOLD's checkpoint.

    A SCAFFOLD that checkpoints one file then crashes must not fail the
    session terminally: the runner resolves the incomplete checkpoint and
    the router re-queues a resuming SCAFFOLD pointed at it.
    """
    from cgx.session.models import SessionMode

    @register_executor(TaskKind.SCAFFOLD)
    def _checkpoint_then_crash(task, deps):
        art = Artifact.new(
            task.session_id, task.task_id, ArtifactKind.SCAFFOLD_PATCHES, {
                "work_plan_artifact_id": "art_plan",
                "diffs": [{"file": "a.py", "patch": "+++ a.py\n# a\n"}],
                "generated": [{"file": "a.py"}], "failed": [],
                "complete": False,
            })
        deps.store.save_artifact(art)
        raise RuntimeError("model timed out mid-scaffold")

    session = Session.new("g", project_root=str(tmp_path),
                          mode=SessionMode.GREENFIELD)
    store.save_session(session)
    scaffold = TaskNode.new(
        session.session_id, TaskKind.SCAFFOLD, "scaffold",
        inputs={"mode": SessionMode.GREENFIELD.value,
                "work_plan_artifact_id": "art_plan"})
    scaffold.status = TaskNodeStatus.READY
    store.save_task(scaffold)

    out = SessionRunner(store).run_next(
        session_id=session.session_id, deps=ExecutorDeps(store=store))
    assert out.status is TaskNodeStatus.FAILED
    # Session stays active -- a resuming SCAFFOLD was queued, not a terminal fail.
    assert store.get_session(session.session_id).status \
        is SessionStatus.ACTIVE
    resumed = [t for t in store.list_tasks(session.session_id)
               if t.kind is TaskKind.SCAFFOLD
               and t.task_id != scaffold.task_id]
    assert len(resumed) == 1
    ckpt_id = resumed[0].inputs["resume_scaffold_artifact_id"]
    ckpt = store.get_artifact(ckpt_id)
    # The pointer resolves to the crashed task's incomplete checkpoint.
    assert ckpt.produced_by_task_id == scaffold.task_id
    assert ckpt.content["complete"] is False


def test_scaffold_augments_goal_with_regenerate_constraints(
        store, monkeypatch):
    """SCAFFOLD with regenerate_constraints injects them into the goal."""
    from cgx.session.tasks.scaffold import run_scaffold
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    plan = Artifact.new(
        session.session_id, "task_x", ArtifactKind.WORK_PLAN, {
            "prior_goal": "g",
            "composed_goal": "build flask api",
            "answers": {}, "plan_md": "",
            "layers": [{"name": "app", "files": [
                {"path": "app.py", "description": "entry"}]}],
        })
    store.save_artifact(plan)

    seen_goals: list = []

    def fake_generate(path, description, provider, *,
                      layer=None, existing_files_with_content=None,
                      goal=None, on_token=None, depends_on=None,
                      contracts=None):
        seen_goals.append(goal)
        body = "x = 1\n"
        return {"file": path, "patch": f"+++ {path}\n{body}",
                "content": body, "syntax_ok": True, "confidence": 0.9}

    monkeypatch.setattr("cgx.answer.engine.generate_single_scaffold_file",
                        fake_generate)
    t = TaskNode.new(
        session.session_id, TaskKind.SCAFFOLD, "s",
        inputs={"work_plan_artifact_id": plan.artifact_id,
                "regenerate_attempt": 1,
                "regenerate_constraints": [
                    {"kind": "unittest_pytest_mix",
                     "rationale": "Do not mix unittest.TestCase with "
                                  "pytest fixtures"},
                ]})
    result = run_scaffold(
        t, ExecutorDeps(provider=_StubProvider(""), store=store))
    assert result.failure is None
    assert seen_goals, "scaffold should have called the generator"
    augmented = seen_goals[0]
    assert "build flask api" in augmented
    assert "Prior-attempt failures" in augmented
    assert "unittest_pytest_mix" in augmented
    assert "Do not mix unittest.TestCase" in augmented


# --------------------- Phase 7.1: cross-session lessons store -------------

def test_record_lesson_appends_jsonl_row(tmp_path: Path):
    """record_lesson writes one JSON object per call to the configured path."""
    from cgx.session.lessons import load_lessons, record_lesson
    target = tmp_path / "lessons.jsonl"
    entry = record_lesson(
        trigger_signature="third_party_import_break|werkzeug",
        classification="third_party_import_break",
        applied_fix={"strategy": "patch", "files": ["requirements.txt"]},
        scope={"stack": ["flask"], "objective_keywords": ["api", "rest"]},
        session_id="sess-A", path=target)
    assert entry is not None
    assert target.exists()
    loaded = load_lessons(path=target)
    assert len(loaded) == 1
    saved = loaded[0]
    assert saved["trigger_signature"].startswith("third_party_import_break")
    assert saved["session_id"] == "sess-A"
    assert saved["applied_fix"]["files"] == ["requirements.txt"]
    # A second record appends instead of overwriting.
    record_lesson(
        trigger_signature="unittest_pytest_mix|abc",
        classification="unittest_pytest_mix",
        applied_fix={"strategy": "patch", "files": ["tests/test_x.py"]},
        scope={"objective_keywords": ["pytest"]},
        session_id="sess-B", path=target)
    assert len(load_lessons(path=target)) == 2


def test_record_lesson_rejects_empty_signature(tmp_path: Path):
    """A blank signature/classification yields no row + no file creation."""
    from cgx.session.lessons import record_lesson
    target = tmp_path / "lessons.jsonl"
    assert record_lesson(
        trigger_signature="", classification="unknown",
        applied_fix={}, scope={}, path=target) is None
    assert not target.exists()


def test_relevant_lessons_scores_by_stack_then_keywords(tmp_path: Path):
    """Stack matches outrank keyword-only matches; both must be > 0."""
    from cgx.session.lessons import record_lesson, relevant_lessons
    target = tmp_path / "lessons.jsonl"
    record_lesson(
        trigger_signature="sig-A", classification="third_party_import_break",
        applied_fix={}, scope={"stack": ["flask"],
                               "objective_keywords": ["rest"]},
        session_id="s1", path=target)
    record_lesson(
        trigger_signature="sig-B", classification="unittest_pytest_mix",
        applied_fix={}, scope={"objective_keywords": ["pytest", "fixture"]},
        session_id="s2", path=target)
    record_lesson(
        trigger_signature="sig-C", classification="missing_fixture",
        applied_fix={}, scope={"stack": ["django"],
                               "objective_keywords": ["orm"]},
        session_id="s3", path=target)
    # Objective with no stack overlap on django -> only A and B score.
    out = relevant_lessons(
        objective="build a REST endpoint with pytest fixtures",
        stack=["Flask"], path=target)
    sigs = [L["trigger_signature"] for L in out]
    # Stack match (Flask) ranks sig-A above keyword-only sig-B.
    assert sigs[0] == "sig-A"
    assert "sig-B" in sigs
    assert "sig-C" not in sigs


def test_relevant_lessons_empty_when_store_missing(tmp_path: Path):
    """Missing lessons file -> empty list (never raises)."""
    from cgx.session.lessons import relevant_lessons
    out = relevant_lessons(
        objective="anything", stack=["flask"],
        path=tmp_path / "does-not-exist.jsonl")
    assert out == []


def test_router_verify_pass_with_repair_ancestor_emits_record_lesson():
    """A VERIFY-pass downstream of REPAIR -> RecordLesson action."""
    from cgx.session.models import SessionMode
    from cgx.session.router import RecordLesson
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    scaffold = TaskNode.new(
        session.session_id, TaskKind.SCAFFOLD, "scaffold", inputs={})
    scaffold.status = TaskNodeStatus.DONE
    apply1 = TaskNode.new(
        session.session_id, TaskKind.APPLY, "apply",
        parent_task_id=scaffold.task_id, inputs={})
    apply1.status = TaskNodeStatus.DONE
    verify1 = TaskNode.new(
        session.session_id, TaskKind.VERIFY, "verify-1",
        parent_task_id=apply1.task_id, inputs={})
    verify1.status = TaskNodeStatus.DONE
    rep = TaskNode.new(
        session.session_id, TaskKind.REPAIR, "repair",
        parent_task_id=verify1.task_id, inputs={})
    rep.produced_artifact_id = "art_plan"
    rep.status = TaskNodeStatus.DONE
    apply2 = TaskNode.new(
        session.session_id, TaskKind.APPLY, "apply-2",
        parent_task_id=rep.task_id, inputs={})
    apply2.status = TaskNodeStatus.DONE
    verify2 = TaskNode.new(
        session.session_id, TaskKind.VERIFY, "verify-2",
        parent_task_id=apply2.task_id,
        inputs={"mode": SessionMode.GREENFIELD.value})
    verify2.outputs = {"outcome": "passed", "returncode": 0}
    verify2.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=verify2,
        tasks=[scaffold, apply1, verify1, rep, apply2, verify2])
    lessons_actions = [a for a in plan.actions
                       if isinstance(a, RecordLesson)]
    assert len(lessons_actions) == 1
    action = lessons_actions[0]
    assert action.verify_task_id == verify2.task_id
    assert action.repair_task_id == rep.task_id
    assert action.scaffold_task_id == scaffold.task_id


def test_router_verify_pass_without_repair_emits_no_lesson():
    """A fresh VERIFY-pass with no REPAIR upstream -> zero RecordLesson actions."""
    from cgx.session.models import SessionMode
    from cgx.session.router import RecordLesson
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    scaffold = TaskNode.new(
        session.session_id, TaskKind.SCAFFOLD, "scaffold", inputs={})
    scaffold.status = TaskNodeStatus.DONE
    apply_t = TaskNode.new(
        session.session_id, TaskKind.APPLY, "apply",
        parent_task_id=scaffold.task_id, inputs={})
    apply_t.status = TaskNodeStatus.DONE
    verify = TaskNode.new(
        session.session_id, TaskKind.VERIFY, "verify",
        parent_task_id=apply_t.task_id,
        inputs={"mode": SessionMode.GREENFIELD.value})
    verify.outputs = {"outcome": "passed", "returncode": 0}
    verify.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=verify,
        tasks=[scaffold, apply_t, verify])
    assert [a for a in plan.actions if isinstance(a, RecordLesson)] == []


def test_runner_writes_lesson_on_successful_repair_cycle(
        store, tmp_path: Path, monkeypatch):
    """End-to-end: SessionRunner._record_lesson lands a row in lessons.jsonl."""
    from cgx.session.lessons import load_lessons
    from cgx.session.router import RecordLesson
    lessons_file = tmp_path / "lessons.jsonl"
    monkeypatch.setenv("CGX_LESSONS_PATH", str(lessons_file))

    session = Session.new("g", mode=SessionMode.GREENFIELD)
    session.project_root = str(tmp_path)
    store.save_session(session)
    scaffold = TaskNode.new(
        session.session_id, TaskKind.SCAFFOLD, "scaffold",
        inputs={"prior_goal": "Build a Flask REST API with pytest"})
    scaffold.status = TaskNodeStatus.DONE
    store.save_task(scaffold)
    rep = TaskNode.new(
        session.session_id, TaskKind.REPAIR, "repair",
        parent_task_id=scaffold.task_id, inputs={})
    rep.status = TaskNodeStatus.DONE
    plan_artifact = Artifact.new(
        session_id=session.session_id,
        produced_by_task_id=rep.task_id,
        kind=ArtifactKind.REPAIR_PLAN,
        content={
            "failure_signature": "third_party_import_break|werkzeug",
            "classification": "third_party_import_break",
            "strategy": "patch",
            "diffs": [{"file": "requirements.txt", "patch": "..."}],
            "extra_constraints": {},
        })
    store.save_artifact(plan_artifact)
    rep.produced_artifact_id = plan_artifact.artifact_id
    store.save_task(rep)

    runner = SessionRunner(store=store)
    runner._record_lesson(  # type: ignore[attr-defined]
        session,
        RecordLesson(verify_task_id="v-1",
                     repair_task_id=rep.task_id,
                     scaffold_task_id=scaffold.task_id))
    lessons = load_lessons(path=lessons_file)
    assert len(lessons) == 1
    row = lessons[0]
    assert row["trigger_signature"] == "third_party_import_break|werkzeug"
    assert row["classification"] == "third_party_import_break"
    assert row["applied_fix"]["strategy"] == "patch"
    assert row["applied_fix"]["files"] == ["requirements.txt"]
    assert "rest" in row["scope"]["objective_keywords"]
    assert row["session_id"] == session.session_id


def test_scaffold_injects_relevant_lessons_into_goal(
        store, tmp_path: Path, monkeypatch):
    """SCAFFOLD with a matching lesson passes its rationale to the generator."""
    from cgx.session.lessons import record_lesson
    from cgx.session.tasks.scaffold import run_scaffold
    lessons_file = tmp_path / "lessons.jsonl"
    monkeypatch.setenv("CGX_LESSONS_PATH", str(lessons_file))
    record_lesson(
        trigger_signature="third_party_import_break|werkzeug",
        classification="third_party_import_break",
        applied_fix={"strategy": "patch",
                     "files": ["requirements.txt"]},
        scope={"stack": ["flask"],
               "objective_keywords": ["rest", "flask"]},
        session_id="sess-prior", path=lessons_file)

    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    plan = Artifact.new(
        session.session_id, "task_x", ArtifactKind.WORK_PLAN, {
            "prior_goal": "g",
            "composed_goal": "Build a Flask REST API",
            "answers": {}, "plan_md": "",
            "requirements_pins": ["flask==3.0.0"],
            "layers": [{"name": "app", "files": [
                {"path": "app.py", "description": "entry"}]}],
        })
    store.save_artifact(plan)

    seen_goals: list = []

    def fake_generate(path, description, provider, *,
                      layer=None, existing_files_with_content=None,
                      goal=None, on_token=None, depends_on=None,
                      contracts=None):
        seen_goals.append(goal)
        body = "x = 1\n"
        return {"file": path, "patch": f"+++ {path}\n{body}",
                "content": body, "syntax_ok": True, "confidence": 0.9}

    monkeypatch.setattr("cgx.answer.engine.generate_single_scaffold_file",
                        fake_generate)
    t = TaskNode.new(
        session.session_id, TaskKind.SCAFFOLD, "s",
        inputs={"work_plan_artifact_id": plan.artifact_id})
    result = run_scaffold(
        t, ExecutorDeps(provider=_StubProvider(""), store=store))
    assert result.failure is None
    assert seen_goals, "scaffold should have called the generator"
    augmented = seen_goals[0]
    assert "Build a Flask REST API" in augmented
    assert "Lessons from prior sessions" in augmented
    assert "third_party_import_break" in augmented
    assert "werkzeug" in augmented

