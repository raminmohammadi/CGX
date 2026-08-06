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
from cgx.session.actions import (
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


def test_diagnose_model_kinds_present_and_round_trip():
    # P2.1: the reasoning rung's typed additions exist with the exact
    # string values the pure router / store serialization depend on.
    assert TaskKind.DIAGNOSE.value == "diagnose"
    assert ArtifactKind.DIAGNOSIS.value == "diagnosis"
    assert FactKind.REPAIR_LEDGER.value == "repair_ledger"
    assert TaskKind("diagnose") is TaskKind.DIAGNOSE
    assert ArtifactKind("diagnosis") is ArtifactKind.DIAGNOSIS
    assert FactKind("repair_ledger") is FactKind.REPAIR_LEDGER

    ledger = Fact.new("ses_x", FactKind.REPAIR_LEDGER, {"attempts": []})
    assert ledger.to_dict()["kind"] == "repair_ledger"
    diag = Artifact.new("ses_x", "task_x", ArtifactKind.DIAGNOSIS,
                        {"minimal_action": "escalate"})
    assert diag.to_dict()["kind"] == "diagnosis"


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


def test_start_session_applies_greenfield_budget_defaults(store):
    """Greenfield runs are autonomous, so an unset cap gets the finite
    session-wide backstop; explore stays unlimited."""
    from cgx.session.budget import (
        GREENFIELD_MAX_TASK_RUNS,
        GREENFIELD_MAX_WALL_SECONDS,
    )
    from cgx.session.models import SessionMode

    runner = SessionRunner(store)
    gf = runner.start_session(objective="build a Flask api",
                              project_root="/tmp/proj",
                              mode=SessionMode.GREENFIELD)
    gf = store.get_session(gf.session_id)
    assert gf.max_task_runs == GREENFIELD_MAX_TASK_RUNS
    assert gf.max_wall_seconds == GREENFIELD_MAX_WALL_SECONDS

    # Explore is user-gated, not autonomous -- it stays unbounded.
    _install_stub_explore()
    ex = runner.start_session(objective="improve retrieval")
    ex = store.get_session(ex.session_id)
    assert ex.max_task_runs is None
    assert ex.max_wall_seconds is None


def test_start_session_explicit_cap_overrides_greenfield_default(store):
    """An explicit cap always wins over the greenfield default -- including
    a caller that opts a greenfield build back into a smaller budget."""
    from cgx.session.models import SessionMode

    runner = SessionRunner(store)
    gf = runner.start_session(objective="build a Flask api",
                              project_root="/tmp/proj",
                              mode=SessionMode.GREENFIELD,
                              max_task_runs=3, max_wall_seconds=42.0)
    gf = store.get_session(gf.session_id)
    assert gf.max_task_runs == 3
    assert gf.max_wall_seconds == 42.0


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
    from cgx.session.budget import REPAIR_BUDGET
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    api = TaskNode.new(
        session.session_id, TaskKind.API_CHECK, "api",
        inputs={"build_artifact_id": "art_build",
                "mode": SessionMode.GREENFIELD.value,
                "repair_attempt": REPAIR_BUDGET,
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


def test_router_api_check_terminal_records_honest_gate_reason():
    """H-B: a terminal gate failure attaches a concrete reason to the task.

    Without this the CLI epilogue only shows a bare "session failed
    (N done, 0 failed)". The router keeps the gate task DONE (it ran fine;
    its report is what failed) but records an ``error`` naming the gate,
    why it could not be repaired, and the failure signature so the reason
    is surfaced to the user.
    """
    from cgx.session.models import SessionMode
    from cgx.session.budget import REPAIR_BUDGET
    sig = "api_check|werkzeug.urls.url_quote"
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    api = TaskNode.new(
        session.session_id, TaskKind.API_CHECK, "api",
        inputs={"build_artifact_id": "art_build",
                "mode": SessionMode.GREENFIELD.value,
                "repair_attempt": REPAIR_BUDGET,
                "prior_failure_signatures": []})
    api.produced_artifact_id = "art_api"
    api.outputs = {"outcome": "failed", "failed_count": 1,
                   "failure_signature": sig}
    api.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=api, tasks=[api])
    task_updates = [a for a in plan.actions
                    if isinstance(a, UpdateTaskStatus)
                    and a.task_id == api.task_id]
    assert len(task_updates) == 1
    update = task_updates[0]
    # The gate task stays DONE -- it ran; its report is what failed.
    assert update.status is TaskNodeStatus.DONE
    assert update.error
    assert "api_check" in update.error
    assert "budget" in update.error
    assert sig in update.error
    # The honest reason precedes the terminal session status.
    session_status = [a for a in plan.actions
                      if isinstance(a, UpdateSessionStatus)]
    assert len(session_status) == 1
    assert session_status[0].status is SessionStatus.FAILED
    assert (plan.actions.index(update)
            < plan.actions.index(session_status[0]))


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


def test_router_repair_resolve_deps_spawns_bootstrap_env():
    """REPAIR strategy=resolve_deps -> re-queue BOOTSTRAP_ENV (not regenerate)."""
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    sig = "api_check|conflict:flask,werkzeug"
    rep = TaskNode.new(
        session.session_id, TaskKind.REPAIR, "repair",
        inputs={"api_check_artifact_id": "art_api",
                "apply_artifact_id": "art_applied",
                "plan_artifact_id": "art_plan",
                "mode": SessionMode.GREENFIELD.value,
                "repair_attempt": 1,
                "prior_failure_signatures": [sig]})
    rep.produced_artifact_id = "art_plan_repair"
    rep.outputs = {"classification": "dependency_conflict",
                   "strategy": "resolve_deps", "can_apply": False,
                   "diff_count": 0, "repair_attempt": 1,
                   "conflict_packages": ["flask", "werkzeug"],
                   "failure_signature": sig}
    rep.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=rep, tasks=[rep])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    boot = creates[0].task
    assert boot.kind is TaskKind.BOOTSTRAP_ENV
    assert boot.inputs["apply_artifact_id"] == "art_applied"
    assert boot.inputs["resolve_packages"] == ["flask", "werkzeug"]
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
    from cgx.session.budget import REPAIR_BUDGET
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    sm = TaskNode.new(
        session.session_id, TaskKind.SMOKE, "smoke",
        inputs={"build_artifact_id": "art_build",
                "mode": SessionMode.GREENFIELD.value,
                "repair_attempt": REPAIR_BUDGET})
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


def test_router_smoke_pass_threads_repair_budget_to_verify():
    """The repair budget survives the SMOKE -> VERIFY edge.

    A SMOKE that passes mid-repair-loop (e.g. after an install-deps
    round fixed the imports) hands off to VERIFY; dropping the counters
    on this edge silently reset the shared budget and re-opened the
    loop. The LoopBudget threading keeps the whole ledger intact.
    """
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    sm = TaskNode.new(
        session.session_id, TaskKind.SMOKE, "smoke",
        inputs={"build_artifact_id": "art_build",
                "mode": SessionMode.GREENFIELD.value,
                "repair_attempt": 2,
                "prior_failure_signatures": ["s1", "s2"],
                "prior_failing_counts": [5, 3],
                "prior_passing_counts": [1, 2]})
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
    assert ver.inputs["repair_attempt"] == 2
    assert ver.inputs["prior_failure_signatures"] == ["s1", "s2"]
    assert ver.inputs["prior_failing_counts"] == [5, 3]
    assert ver.inputs["prior_passing_counts"] == [1, 2]


# --------------------- repair-loop router transitions ---------------------

def _greenfield_failed_verify(*, signature: str, outcome: str = "assertions_failed",
                              repair_attempt: int = 0, prior: list | None = None,
                              failing_count: int | None = None,
                              prior_counts: list | None = None,
                              passing_count: int | None = None,
                              prior_passing: list | None = None,
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
                "prior_failure_signatures": list(prior or []),
                "prior_failing_counts": list(prior_counts or []),
                "prior_passing_counts": list(prior_passing or [])})
    ver.produced_artifact_id = "art_verify"
    ver.outputs = {"outcome": outcome, "failure_signature": signature,
                   "returncode": 1}
    if failing_count is not None:
        ver.outputs["failing_count"] = failing_count
    if passing_count is not None:
        ver.outputs["passing_count"] = passing_count
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


def test_router_verify_passed_spawns_runtime_verify():
    """P1: a passing greenfield VERIFY hands off to RUNTIME_VERIFY.

    The unit suite is green, but the session is not COMPLETED yet -- the
    app-boot gate must pass first, so the terminal status is deferred.
    """
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    ver = TaskNode.new(
        session.session_id, TaskKind.VERIFY, "verify",
        inputs={"mode": SessionMode.GREENFIELD.value,
                "build_artifact_id": "art_build",
                "apply_artifact_id": "art_apply",
                "scaffold_artifact_id": "art_scaffold"})
    ver.produced_artifact_id = "art_verify"
    ver.outputs = {"outcome": "passed", "failure_signature": "p",
                   "returncode": 0,
                   "js_tests_present": True, "js_tests_ran": False}
    ver.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=ver, tasks=[ver])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    rv = creates[0].task
    assert rv.kind is TaskKind.RUNTIME_VERIFY
    assert rv.inputs["verify_artifact_id"] == "art_verify"
    assert rv.inputs["build_artifact_id"] == "art_build"
    assert rv.inputs["apply_artifact_id"] == "art_apply"
    # P2: VERIFY's JS coverage signal is threaded forward so the runtime
    # gate's terminal action can fail closed on an unrun scaffolded suite.
    assert rv.inputs["js_tests_present"] is True
    assert rv.inputs["js_tests_ran"] is False
    # The session status is deferred to the runtime gate, not set here.
    assert not any(isinstance(a, UpdateSessionStatus) for a in plan.actions)


def test_router_verify_skipped_stays_terminal_completed():
    """A skipped (test-free) greenfield VERIFY completes without a boot gate."""
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    ver = TaskNode.new(
        session.session_id, TaskKind.VERIFY, "verify",
        inputs={"mode": SessionMode.GREENFIELD.value})
    ver.outputs = {"outcome": "skipped", "failure_signature": "",
                   "returncode": 5}
    ver.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=ver, tasks=[ver])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert creates == []
    status = [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]
    assert len(status) == 1
    assert status[0].status is SessionStatus.COMPLETED


def test_router_runtime_verify_passed_completes_session():
    """A booting app completes the greenfield session."""
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    rv = TaskNode.new(
        session.session_id, TaskKind.RUNTIME_VERIFY, "runtime",
        inputs={"mode": SessionMode.GREENFIELD.value})
    rv.outputs = {"outcome": "passed", "failure_signature": ""}
    rv.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=rv, tasks=[rv])
    assert [a for a in plan.actions if isinstance(a, CreateTask)] == []
    status = [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]
    assert len(status) == 1
    assert status[0].status is SessionStatus.COMPLETED


def test_router_runtime_verify_skipped_completes_session():
    """No detectable entry to boot is an explicit no-op -> COMPLETED."""
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    rv = TaskNode.new(
        session.session_id, TaskKind.RUNTIME_VERIFY, "runtime",
        inputs={"mode": SessionMode.GREENFIELD.value})
    rv.outputs = {"outcome": "skipped"}
    rv.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=rv, tasks=[rv])
    status = [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]
    assert len(status) == 1
    assert status[0].status is SessionStatus.COMPLETED


def test_router_runtime_verify_failed_spawns_repair():
    """#3: a boot failure within budget routes to REPAIR, not a terminal.

    The first RUNTIME_VERIFY boot failure hands off to a REPAIR carrying
    the RUNTIME_REPORT so the failing entry module can be re-authored; the
    session stays open (no status flip) while the repair chain runs.
    """
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    rv = TaskNode.new(
        session.session_id, TaskKind.RUNTIME_VERIFY, "runtime",
        inputs={"mode": SessionMode.GREENFIELD.value})
    rv.produced_artifact_id = "art_runtime"
    rv.outputs = {"outcome": "failed",
                  "failure_signature": "runtime_boot|app.py"}
    rv.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=rv, tasks=[rv])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    repair = creates[0].task
    assert repair.kind is TaskKind.REPAIR
    assert repair.inputs["runtime_artifact_id"] == "art_runtime"
    assert repair.inputs["repair_attempt"] == 1
    assert repair.inputs["prior_failure_signatures"] == ["runtime_boot|app.py"]
    # Session status is not flipped while a repair successor exists.
    assert not any(isinstance(a, UpdateSessionStatus) for a in plan.actions)


def test_router_runtime_verify_failed_budget_spent_fails_session():
    """A boot failure with the repair budget exhausted is a terminal FAILED."""
    from cgx.session.models import SessionMode
    from cgx.session.budget import REPAIR_BUDGET
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    rv = TaskNode.new(
        session.session_id, TaskKind.RUNTIME_VERIFY, "runtime",
        inputs={"mode": SessionMode.GREENFIELD.value,
                "repair_attempt": REPAIR_BUDGET})
    rv.produced_artifact_id = "art_runtime"
    rv.outputs = {"outcome": "failed",
                  "failure_signature": "runtime_boot|app.py"}
    rv.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=rv, tasks=[rv])
    assert [a for a in plan.actions if isinstance(a, CreateTask)] == []
    status = [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]
    assert len(status) == 1
    assert status[0].status is SessionStatus.FAILED


def test_router_runtime_verify_explore_is_noop():
    """RUNTIME_VERIFY is greenfield-only -- explore mode flips no status."""
    session = Session.new("e")
    rv = TaskNode.new(
        session.session_id, TaskKind.RUNTIME_VERIFY, "runtime",
        inputs={"mode": "explore"})
    rv.outputs = {"outcome": "failed"}
    rv.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=rv, tasks=[rv])
    assert not any(isinstance(a, UpdateSessionStatus) for a in plan.actions)


def test_router_runtime_verify_passed_but_js_unrun_fails_closed():
    """P2: a booting app with an unrun scaffolded JS suite is not green.

    The ses_4cbf963cdc67435a hole -- the Python half passed and the app
    booted, but the scaffolded React suite never executed, so "completed"
    would have been a false green. The terminal fails closed to FAILED.
    """
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    rv = TaskNode.new(
        session.session_id, TaskKind.RUNTIME_VERIFY, "runtime",
        inputs={"mode": SessionMode.GREENFIELD.value,
                "js_tests_present": True, "js_tests_ran": False})
    rv.outputs = {"outcome": "passed", "failure_signature": ""}
    rv.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=rv, tasks=[rv])
    assert [a for a in plan.actions if isinstance(a, CreateTask)] == []
    status = [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]
    assert len(status) == 1
    assert status[0].status is SessionStatus.FAILED


def test_router_runtime_verify_passed_with_js_ran_completes():
    """A booting app whose scaffolded JS suite actually ran is green."""
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    rv = TaskNode.new(
        session.session_id, TaskKind.RUNTIME_VERIFY, "runtime",
        inputs={"mode": SessionMode.GREENFIELD.value,
                "js_tests_present": True, "js_tests_ran": True})
    rv.outputs = {"outcome": "passed", "failure_signature": ""}
    rv.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=rv, tasks=[rv])
    status = [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]
    assert len(status) == 1
    assert status[0].status is SessionStatus.COMPLETED


def test_router_runtime_verify_skipped_with_server_entry_fails_closed():
    """P2: a boot that skipped while a server entry was on disk is not green.

    The whole-tree scan (P1c) surfaced a bootable ``backend/app.py``, but
    the gate skipped (e.g. no bootstrapped interpreter) so the server was
    never actually exercised. Completing green would repeat the
    ses_4cbf963cdc67435a blind spot, so the terminal fails closed.
    """
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    rv = TaskNode.new(
        session.session_id, TaskKind.RUNTIME_VERIFY, "runtime",
        inputs={"mode": SessionMode.GREENFIELD.value})
    rv.outputs = {"outcome": "skipped", "entry_files": ["backend/app.py"]}
    rv.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=rv, tasks=[rv])
    status = [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]
    assert len(status) == 1
    assert status[0].status is SessionStatus.FAILED


def test_router_verify_skipped_but_js_unrun_fails_closed():
    """P2: a test-free VERIFY that still carries an unrun JS suite is not green.

    A polyglot repo whose Python half is test-free (combined ``skipped``)
    but whose scaffolded JS suite was never executed reaches the VERIFY
    terminal directly (no boot gate). It must fail closed rather than
    report the write loop delivered a verified app.
    """
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    ver = TaskNode.new(
        session.session_id, TaskKind.VERIFY, "verify",
        inputs={"mode": SessionMode.GREENFIELD.value})
    ver.outputs = {"outcome": "skipped", "failure_signature": "",
                   "returncode": 5,
                   "js_tests_present": True, "js_tests_ran": False}
    ver.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=ver, tasks=[ver])
    assert [a for a in plan.actions if isinstance(a, CreateTask)] == []
    status = [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]
    assert len(status) == 1
    assert status[0].status is SessionStatus.FAILED


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


def _failed_decompose(session, *, retries=0, error="DECOMPOSE: bad plan"):
    task = TaskNode.new(
        session.session_id, TaskKind.DECOMPOSE, "decompose",
        inputs={"prior_goal": "build a todo app",
                "requirements_artifact_id": "art_req",
                "answers": {"q1": "a1"},
                "decompose_retry": retries})
    task.status = TaskNodeStatus.FAILED
    task.error = error
    return task


def test_router_on_task_failed_retryable_decompose_requeues_with_constraint():
    """A retryable DECOMPOSE failure re-queues with the failure folded in."""
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    task = _failed_decompose(session)
    plan = Router().on_task_failed(
        session=session, failed=task, tasks=[task], retryable=True)
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    retry = creates[0].task
    assert retry.kind is TaskKind.DECOMPOSE
    assert retry.parent_task_id == task.task_id
    assert retry.inputs["decompose_retry"] == 1
    assert retry.inputs["requirements_artifact_id"] == "art_req"
    assert retry.inputs["answers"] == {"q1": "a1"}
    # Original objective survives verbatim; failure is folded as constraint.
    assert retry.inputs["prior_goal"].startswith("build a todo app")
    assert "DECOMPOSE: bad plan" in retry.inputs["prior_goal"]
    # Session stays active -- no terminal transition alongside the retry.
    assert not any(isinstance(a, UpdateSessionStatus) for a in plan.actions)


def test_router_on_task_failed_retryable_decompose_budget_exhausted():
    """A second retryable DECOMPOSE failure ends the session terminally."""
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    task = _failed_decompose(session, retries=1)
    plan = Router().on_task_failed(
        session=session, failed=task, tasks=[task], retryable=True)
    assert not any(isinstance(a, CreateTask) for a in plan.actions)
    status = [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]
    assert len(status) == 1
    assert status[0].status is SessionStatus.FAILED


def test_router_on_task_failed_retryable_non_decompose_stays_terminal():
    """retryable only re-queues DECOMPOSE; other kinds fail terminally."""
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    task = TaskNode.new(
        session.session_id, TaskKind.BOOTSTRAP_ENV, "bootstrap",
        inputs={"mode": SessionMode.GREENFIELD.value})
    task.status = TaskNodeStatus.FAILED
    plan = Router().on_task_failed(
        session=session, failed=task, tasks=[task], retryable=True)
    assert not any(isinstance(a, CreateTask) for a in plan.actions)
    status = [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]
    assert len(status) == 1
    assert status[0].status is SessionStatus.FAILED


def test_router_on_task_failed_non_retryable_decompose_stays_terminal():
    """Without the retryable flag a DECOMPOSE failure stays terminal."""
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    task = _failed_decompose(session)
    plan = Router().on_task_failed(
        session=session, failed=task, tasks=[task])
    assert not any(isinstance(a, CreateTask) for a in plan.actions)
    status = [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]
    assert len(status) == 1
    assert status[0].status is SessionStatus.FAILED


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
    """Absolute repair cap exhausted -> no REPAIR (loop guard)."""
    from cgx.session.models import SessionMode
    from cgx.session.budget import REPAIR_BUDGET
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    ver = _greenfield_failed_verify(
        signature="new", repair_attempt=REPAIR_BUDGET,
        prior=[f"old{i}" for i in range(REPAIR_BUDGET)], session=session)
    plan = Router().on_task_completed(
        session=session, completed=ver, tasks=[ver])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert creates == []
    # Budget exhausted -> no REPAIR, session fails terminally.
    status = [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]
    assert len(status) == 1
    assert status[0].status is SessionStatus.FAILED


def test_router_verify_progress_drop_allows_repair_past_old_cap():
    """P2: a strictly-dropping failing-test count keeps the loop repairing.

    With the old 2-shot cap this third round (repair_attempt=2) would have
    terminated; the progress-aware budget lets it continue because the
    count fell 5 -> 3 -> 2, and threads the new count onto the trend.
    """
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    ver = _greenfield_failed_verify(
        signature="round3", repair_attempt=2, prior=["r1", "r2"],
        failing_count=2, prior_counts=[5, 3], session=session)
    plan = Router().on_task_completed(
        session=session, completed=ver, tasks=[ver])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    rep = creates[0].task
    assert rep.kind is TaskKind.REPAIR
    assert rep.inputs["prior_failing_counts"] == [5, 3, 2]
    assert rep.inputs["repair_attempt"] == 3
    # A spawned REPAIR defers the terminal status decision.
    assert not any(isinstance(a, UpdateSessionStatus) for a in plan.actions)


def test_router_verify_progress_stall_refuses_repair():
    """P2: a failing-test count that stops dropping ends the loop.

    The signature is brand new (the old flap guard would allow another
    REPAIR), but the count did not fall (3 -> 3), so the progress-aware
    gate declares a stall and the session fails terminally.
    """
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    ver = _greenfield_failed_verify(
        signature="fresh", repair_attempt=1, prior=["r1"],
        failing_count=3, prior_counts=[3], session=session)
    plan = Router().on_task_completed(
        session=session, completed=ver, tasks=[ver])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert creates == []
    status = [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]
    assert len(status) == 1
    assert status[0].status is SessionStatus.FAILED


def test_router_verify_passing_rise_rescues_flat_failing():
    """#5: a flat failing count still progresses when MORE tests pass.

    The failing count held at 3 (the P2 gate alone would call this a
    stall), but the passing count rose 2 -> 4 -- a round that fixed a test
    while another newly-collected one began failing. The coverage-aware
    ledger treats that as forward progress: REPAIR is spawned and both
    trends are threaded onto the next round.
    """
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    ver = _greenfield_failed_verify(
        signature="fresh", repair_attempt=1, prior=["r1"],
        failing_count=3, prior_counts=[3],
        passing_count=4, prior_passing=[2], session=session)
    plan = Router().on_task_completed(
        session=session, completed=ver, tasks=[ver])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    rep = creates[0].task
    assert rep.kind is TaskKind.REPAIR
    assert rep.inputs["prior_failing_counts"] == [3, 3]
    assert rep.inputs["prior_passing_counts"] == [2, 4]


def test_router_verify_stall_when_failing_flat_and_passing_flat():
    """#5: flat failing AND flat passing is a genuine stall -> terminal.

    A fresh signature would slip past the flap guard, but neither lever
    moved (failing 3 -> 3, passing 2 -> 2), so the coverage-aware gate
    ends the loop.
    """
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    ver = _greenfield_failed_verify(
        signature="fresh", repair_attempt=1, prior=["r1"],
        failing_count=3, prior_counts=[3],
        passing_count=2, prior_passing=[2], session=session)
    plan = Router().on_task_completed(
        session=session, completed=ver, tasks=[ver])
    assert [a for a in plan.actions if isinstance(a, CreateTask)] == []
    status = [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]
    assert len(status) == 1
    assert status[0].status is SessionStatus.FAILED


def test_router_collection_error_progress_drop_allows_repair():
    """P2/#5: a collection error whose erroring-module count drops repairs.

    ``failing_count`` on a ``collection_error`` is the number of modules
    erroring during collection (import fixes landing one at a time). A
    strictly-dropping count (2 -> 1) is real forward progress, so the loop
    continues and threads the new count onto the trend.
    """
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    ver = _greenfield_failed_verify(
        signature="cerr2", outcome="collection_error", repair_attempt=1,
        prior=["cerr1"], failing_count=1, prior_counts=[2], session=session)
    plan = Router().on_task_completed(
        session=session, completed=ver, tasks=[ver])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    rep = creates[0].task
    assert rep.kind is TaskKind.REPAIR
    assert rep.inputs["prior_failing_counts"] == [2, 1]


def test_router_collection_error_progress_stall_refuses_repair():
    """P2/#5: a collection error whose count stops dropping ends the loop.

    Before this fix a ``collection_error`` ignored ``failing_count`` and a
    fresh signature always slipped past the flap guard. Now a flat count
    (2 -> 2) is a stall even with a brand-new signature, so the session
    fails terminally instead of burning another round.
    """
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    ver = _greenfield_failed_verify(
        signature="cfresh", outcome="collection_error", repair_attempt=1,
        prior=["cerr1"], failing_count=2, prior_counts=[2], session=session)
    plan = Router().on_task_completed(
        session=session, completed=ver, tasks=[ver])
    assert [a for a in plan.actions if isinstance(a, CreateTask)] == []
    status = [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]
    assert len(status) == 1
    assert status[0].status is SessionStatus.FAILED


def test_router_repair_apply_verify_chain_carries_failing_counts():
    """P2/#5: the failing + passing ledgers survive REPAIR -> APPLY -> VERIFY."""
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    # REPAIR -> APPLY carries the trend forward.
    rep = TaskNode.new(
        session.session_id, TaskKind.REPAIR, "repair",
        inputs={"verify_artifact_id": "art_verify",
                "build_artifact_id": "art_build",
                "mode": SessionMode.GREENFIELD.value,
                "repair_attempt": 1,
                "prior_failure_signatures": ["r1"],
                "prior_failing_counts": [5, 3],
                "prior_passing_counts": [1, 3]})
    rep.produced_artifact_id = "art_repair_plan"
    rep.outputs = {"can_apply": True, "failure_signature": "r1",
                   "repair_attempt": 1, "diff_count": 2}
    rep.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=rep, tasks=[rep])
    ap = [a for a in plan.actions if isinstance(a, CreateTask)][0].task
    assert ap.kind is TaskKind.APPLY
    assert ap.inputs["prior_failing_counts"] == [5, 3]
    assert ap.inputs["prior_passing_counts"] == [1, 3]
    # APPLY -> VERIFY carries it the rest of the way.
    ap.produced_artifact_id = "art_applied"
    ap.status = TaskNodeStatus.DONE
    plan2 = Router().on_task_completed(
        session=session, completed=ap, tasks=[ap])
    ver = [a for a in plan2.actions if isinstance(a, CreateTask)][0].task
    assert ver.kind is TaskKind.VERIFY
    assert ver.inputs["prior_failing_counts"] == [5, 3]
    assert ver.inputs["prior_passing_counts"] == [1, 3]


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
    # The router's progress-aware repair budget reads this count.
    assert result.outputs["failing_count"] == 2
    # #5: coverage-ledger counts. Here junit reported 2 failures against a
    # single collected pytest node, so passing clamps at 0 (never negative).
    assert result.outputs["collected_count"] == 1
    assert result.outputs["passing_count"] == 0
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


def test_verify_collection_error_empty_junit_reports_no_false_progress(
        tmp_path, store, monkeypatch):
    """A collection error with no junit must not read as "0 failing".

    Regression for the false-success signal: pytest exit 4 (and often 2)
    writes no per-testcase junit, so ``failures`` is empty. Reporting
    ``failing_count: 0`` / ``passing_count: N`` there let the router's
    progress ledger read a total collection failure as forward progress
    and loop. The suite never executed, so passing must be 0 and failing
    must be unknown (``None``) -- the router then falls back to the
    signature-flap + REPAIR_BUDGET backstops.
    """
    from cgx.codegen.test_runner import TestRunOutcome
    from cgx.session.tasks.verify import run_verify

    (tmp_path / "tests").mkdir()
    test_file = tmp_path / "tests" / "test_x.py"
    test_file.write_text("def test_x(): assert 1\n", encoding="utf-8")

    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)

    monkeypatch.setattr(
        "cgx.codegen.test_runner.run_tests_on_disk",
        lambda root, files, **_kw: TestRunOutcome(
            ran=True, returncode=4,
            stdout="", stderr="ERROR: usage error during collection",
            tests_selected=[str(test_file)],
        ))
    t = TaskNode.new(session.session_id, TaskKind.VERIFY, "verify",
                     inputs={"mode": SessionMode.GREENFIELD.value})
    store.save_task(t)
    result = run_verify(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.failure is None
    assert result.outputs["outcome"] == "collection_error"
    # Nothing executed: no false "N passing", and failing is unknown so the
    # router's progress gate stays inconclusive rather than seeing 0 < prior.
    assert result.outputs["passing_count"] == 0
    assert result.outputs["failing_count"] is None
    assert result.outputs["collected_count"] == 1


def test_verify_collection_error_with_junit_errors_keeps_failing_count(
        tmp_path, store, monkeypatch):
    """A collection error that enumerates erroring modules keeps the trend.

    The router trusts a strictly-dropping ``failing_count`` on a
    ``collection_error`` as import fixes landing one module at a time, so
    a junit that actually lists erroring testcases must still surface the
    real count (here 2) -- only the empty-junit case degrades to ``None``.
    """
    from cgx.codegen.test_runner import TestRunOutcome
    from cgx.session.tasks.verify import run_verify

    (tmp_path / "tests").mkdir()
    test_file = tmp_path / "tests" / "test_x.py"
    test_file.write_text("def test_x(): assert 1\n", encoding="utf-8")

    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)

    def _fake_run(root, files, **kw):
        for arg in kw.get("extra_pytest_args") or ():
            if isinstance(arg, str) and arg.startswith("--junitxml="):
                Path(arg.split("=", 1)[1]).write_text(
                    '<?xml version="1.0" encoding="utf-8"?>'
                    '<testsuites><testsuite name="pytest" tests="2" errors="2">'
                    '<testcase classname="tests.test_x" name="test_x">'
                    '<error type="ImportError" message="no module a">'
                    'ImportError: no module a\n</error></testcase>'
                    '<testcase classname="tests.test_y" name="test_y">'
                    '<error type="ImportError" message="no module b">'
                    'ImportError: no module b\n</error></testcase>'
                    '</testsuite></testsuites>',
                    encoding="utf-8")
                break
        return TestRunOutcome(
            ran=True, returncode=2, stdout="", stderr="",
            tests_selected=[str(test_file)])

    monkeypatch.setattr(
        "cgx.codegen.test_runner.run_tests_on_disk", _fake_run)
    t = TaskNode.new(session.session_id, TaskKind.VERIFY, "verify",
                     inputs={"mode": SessionMode.GREENFIELD.value})
    store.save_task(t)
    result = run_verify(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.failure is None
    assert result.outputs["outcome"] == "collection_error"
    assert result.outputs["failing_count"] == 2
    assert result.outputs["passing_count"] == 0


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


def test_verify_npm_only_build_pass_is_no_tests(
        tmp_path, store, monkeypatch):
    """A build-only project that builds but wired up no tests -> no_tests.

    A passing *build* is not a passing *suite*: with no ``test`` script the
    NpmRunner ran only a build smoke (``ran_tests`` False), so VERIFY must
    report the honest ``no_tests`` rather than a false green ``passed``.
    """
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
            tests_selected=["npm run build"], ran_tests=False))

    t = TaskNode.new(session.session_id, TaskKind.VERIFY, "verify",
                     inputs={"changed_files": ["src/App.jsx"],
                             "mode": SessionMode.GREENFIELD.value})
    store.save_task(t)
    result = run_verify(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.failure is None
    assert result.outputs["outcome"] == "no_tests"
    assert result.outputs["tests_passed"] is False
    assert result.outputs["passing_count"] == 0


def test_verify_npm_real_test_pass_is_passed(
        tmp_path, store, monkeypatch):
    """A project whose real ``test`` script passes -> passed (ran_tests)."""
    from cgx.codegen.test_runner import TestRunOutcome
    from cgx.codegen import test_runners
    from cgx.session.tasks.verify import run_verify

    (tmp_path / "package.json").write_text(
        '{"name": "app", "scripts": {"test": "vitest run"}}',
        encoding="utf-8")

    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)

    monkeypatch.setattr(
        test_runners.NpmRunner, "run",
        lambda self, root, files, **kw: TestRunOutcome(
            ran=True, returncode=0, stdout="2 passed", stderr="",
            tests_selected=["npm test"], ran_tests=True))

    t = TaskNode.new(session.session_id, TaskKind.VERIFY, "verify",
                     inputs={"changed_files": ["src/App.jsx"],
                             "mode": SessionMode.GREENFIELD.value})
    store.save_task(t)
    result = run_verify(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.failure is None
    assert result.outputs["outcome"] == "passed"
    assert result.outputs["tests_passed"] is True


def test_classify_other_outcome_build_only_is_no_tests():
    """rc==0 build smoke -> no_tests; rc==0 real suite -> passed."""
    from cgx.codegen.test_runner import TestRunOutcome
    from cgx.session.tasks.verify import _classify_other_outcome

    build_only = TestRunOutcome(
        ran=True, returncode=0, tests_selected=["npm run build"],
        ran_tests=False)
    real_suite = TestRunOutcome(
        ran=True, returncode=0, tests_selected=["npm test"], ran_tests=True)
    assert _classify_other_outcome(build_only) == "no_tests"
    assert _classify_other_outcome(real_suite) == "passed"
    # A non-zero build is still a hard ``failed``, unaffected by ran_tests.
    broke = TestRunOutcome(
        ran=True, returncode=1, tests_selected=["npm run build"],
        ran_tests=False)
    assert _classify_other_outcome(broke) == "failed"


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


def test_verify_surfaces_js_tests_present_when_masked_by_pytest(
        tmp_path, store, monkeypatch):
    """Polyglot: a scaffolded-but-unrun JS suite is exposed, not hidden.

    The ses_4cbf963cdc67435a shape -- pytest passes while the React suite
    (present on disk) only got a build smoke. The combined token is still
    ``passed`` (build was green), but VERIFY must surface
    ``js_tests_present`` True / ``js_tests_ran`` False so P2 can fail closed.
    """
    from cgx.codegen.test_runner import TestRunOutcome
    from cgx.codegen import test_runners
    from cgx.session.tasks.verify import run_verify

    (tmp_path / "package.json").write_text(
        '{"name": "app", "scripts": {"build": "vite build"}}',
        encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "App.test.jsx").write_text(
        "test('x', () => {})\n", encoding="utf-8")
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
            ran=True, returncode=0, stdout="built ok", stderr="",
            tests_selected=["npm run build"], ran_tests=False,
            tests_present=True))

    t = TaskNode.new(session.session_id, TaskKind.VERIFY, "verify",
                     inputs={"changed_files": ["src/App.jsx"],
                             "mode": SessionMode.GREENFIELD.value})
    store.save_task(t)
    result = run_verify(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.failure is None
    assert result.outputs["outcome"] == "passed"
    assert result.outputs["js_tests_present"] is True
    assert result.outputs["js_tests_ran"] is False
    assert result.artifact.content["js_tests_present"] is True


def test_verify_js_tests_ran_true_for_real_suite(
        tmp_path, store, monkeypatch):
    """A real JS suite that executed records js_tests_ran True."""
    from cgx.codegen.test_runner import TestRunOutcome
    from cgx.codegen import test_runners
    from cgx.session.tasks.verify import run_verify

    (tmp_path / "package.json").write_text(
        '{"name": "app", "scripts": {"test": "vitest run"}}',
        encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "App.test.jsx").write_text(
        "test('x', () => {})\n", encoding="utf-8")

    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)

    monkeypatch.setattr(
        test_runners.NpmRunner, "run",
        lambda self, root, files, **kw: TestRunOutcome(
            ran=True, returncode=0, stdout="2 passed", stderr="",
            tests_selected=["npm test"], ran_tests=True, tests_present=True))

    t = TaskNode.new(session.session_id, TaskKind.VERIFY, "verify",
                     inputs={"changed_files": ["src/App.jsx"],
                             "mode": SessionMode.GREENFIELD.value})
    store.save_task(t)
    result = run_verify(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.failure is None
    assert result.outputs["outcome"] == "passed"
    assert result.outputs["js_tests_present"] is True
    assert result.outputs["js_tests_ran"] is True


def test_runtime_verify_skipped_without_python_exe(tmp_path, store):
    """No bootstrapped interpreter -> the boot gate is an explicit no-op."""
    from cgx.session.models import SessionMode
    from cgx.session.tasks.runtime_verify import run_runtime_verify
    (tmp_path / "app.py").write_text(
        "def create_app():\n    return object()\n", encoding="utf-8")
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    t = TaskNode.new(session.session_id, TaskKind.RUNTIME_VERIFY, "runtime",
                     inputs={"applied_files": ["app.py"],
                             "mode": SessionMode.GREENFIELD.value})
    store.save_task(t)
    result = run_runtime_verify(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.failure is None
    assert result.outputs["outcome"] == "skipped"
    assert result.outputs["tested_count"] == 0


def test_runtime_verify_skipped_without_entry_candidates(tmp_path, store):
    """A green suite with no bootable entry module skips the runtime gate."""
    import sys
    from cgx.session.models import SessionMode
    from cgx.session.tasks.runtime_verify import run_runtime_verify
    (tmp_path / "helpers.py").write_text("VALUE = 1\n", encoding="utf-8")
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    t = TaskNode.new(session.session_id, TaskKind.RUNTIME_VERIFY, "runtime",
                     inputs={"applied_files": ["helpers.py"],
                             "python_exe": sys.executable,
                             "mode": SessionMode.GREENFIELD.value})
    store.save_task(t)
    result = run_runtime_verify(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.failure is None
    assert result.outputs["outcome"] == "skipped"
    assert result.outputs["tested_count"] == 0


def test_runtime_verify_passed_when_app_boots(tmp_path, store):
    """A create_app factory that constructs cleanly -> passed with a probe."""
    import sys
    from cgx.session.models import SessionMode
    from cgx.session.tasks.runtime_verify import run_runtime_verify
    (tmp_path / "app.py").write_text(
        "def create_app():\n    return {'ok': True}\n", encoding="utf-8")
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    t = TaskNode.new(session.session_id, TaskKind.RUNTIME_VERIFY, "runtime",
                     inputs={"applied_files": ["app.py"],
                             "python_exe": sys.executable,
                             "mode": SessionMode.GREENFIELD.value})
    store.save_task(t)
    result = run_runtime_verify(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.failure is None
    assert result.outputs["outcome"] == "passed"
    assert result.outputs["tested_count"] == 1
    assert result.outputs["failed_count"] == 0
    assert result.artifact is not None
    assert result.artifact.kind is ArtifactKind.RUNTIME_REPORT
    assert result.artifact.content["probes"][0]["file"] == "app.py"


def test_runtime_verify_failed_on_import_time_error(tmp_path, store):
    """An import-time error the unit suite missed -> failed with a signature."""
    import sys
    from cgx.session.models import SessionMode
    from cgx.session.tasks.runtime_verify import run_runtime_verify
    # A raise at module load -- exactly the class of bug a passing unit
    # suite can miss when it never imports the entry module.
    (tmp_path / "main.py").write_text(
        "raise RuntimeError('boot boom')\n", encoding="utf-8")
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    t = TaskNode.new(session.session_id, TaskKind.RUNTIME_VERIFY, "runtime",
                     inputs={"applied_files": ["main.py"],
                             "python_exe": sys.executable,
                             "mode": SessionMode.GREENFIELD.value})
    store.save_task(t)
    result = run_runtime_verify(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.failure is None
    assert result.outputs["outcome"] == "failed"
    assert result.outputs["failed_count"] == 1
    assert result.outputs["failure_signature"] == "runtime_boot|main.py"
    probe = result.artifact.content["probes"][0]
    assert probe["ok"] is False
    assert probe["kind"] == "import_error"
    assert "boot boom" in probe["stderr_tail"]


def test_runtime_verify_probes_nested_entry_absent_from_applied_files(
        tmp_path, store):
    """P1c: a nested backend entry not in the last APPLY is still booted.

    The ses_4cbf963cdc67435a blind spot -- ``backend/app.py`` existed on
    disk but was absent from the final applied-files list, so the boot
    gate skipped and a broken server shipped green. The whole-tree scan
    must find and probe it regardless.
    """
    import sys
    from cgx.session.models import SessionMode
    from cgx.session.tasks.runtime_verify import run_runtime_verify
    (tmp_path / "backend").mkdir()
    (tmp_path / "backend" / "app.py").write_text(
        "from flask import Flask\napp = Flask(__name__)\n", encoding="utf-8")
    # The last APPLY only touched a frontend file (no .py entry).
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "App.jsx").write_text(
        "export default () => null\n", encoding="utf-8")
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    t = TaskNode.new(session.session_id, TaskKind.RUNTIME_VERIFY, "runtime",
                     inputs={"applied_files": ["src/App.jsx"],
                             "python_exe": sys.executable,
                             "mode": SessionMode.GREENFIELD.value})
    store.save_task(t)
    result = run_runtime_verify(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.failure is None
    assert "backend/app.py" in result.artifact.content["entry_files"]
    assert result.outputs["tested_count"] == 1


def test_runtime_verify_tree_scan_skips_vendored_dirs(tmp_path, store):
    """A dependency's own app.py under node_modules is never probed."""
    import sys
    from cgx.session.models import SessionMode
    from cgx.session.tasks.runtime_verify import run_runtime_verify
    (tmp_path / "node_modules" / "dep").mkdir(parents=True)
    (tmp_path / "node_modules" / "dep" / "app.py").write_text(
        "raise RuntimeError('should never boot')\n", encoding="utf-8")
    (tmp_path / "helpers.py").write_text("VALUE = 1\n", encoding="utf-8")
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    t = TaskNode.new(session.session_id, TaskKind.RUNTIME_VERIFY, "runtime",
                     inputs={"applied_files": ["helpers.py"],
                             "python_exe": sys.executable,
                             "mode": SessionMode.GREENFIELD.value})
    store.save_task(t)
    result = run_runtime_verify(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.failure is None
    assert result.outputs["outcome"] == "skipped"
    assert result.outputs["tested_count"] == 0


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


def test_agent_log_survives_project_dir_removed_under_handler(tmp_path, capfd):
    """A re-scaffold that trashes ``.cgx`` must not flood stderr or crash.

    Regression: the cached RotatingFileHandler kept a descriptor for the
    project-local ``agent.log``; when the project tree was regenerated the
    ``.cgx`` directory vanished, and every subsequent emit raised
    FileNotFoundError inside stdlib logging -- flooding stderr with
    ``--- Logging error ---`` tracebacks. ``_emit_to`` now detects the dead
    directory, rebuilds the handler (re-creating ``.cgx``), and keeps logging.
    """
    import shutil
    from cgx.session.agent_log import log_event

    proj = tmp_path / "proj"
    log_event(str(proj), "task_started", session_id="ses_r", kind="apply")
    assert _read_agent_log(proj)

    # Simulate the re-scaffold: the whole project tree (incl. .cgx) is trashed
    # while the rotating handler is still cached and pointing at the old path.
    shutil.rmtree(proj)
    assert not (proj / ".cgx").exists()

    _ = capfd.readouterr()  # drop anything emitted so far
    # The next emit must neither raise nor print a stdlib logging traceback.
    log_event(str(proj), "task_completed", session_id="ses_r", kind="apply")
    err = capfd.readouterr().err
    assert "--- Logging error ---" not in err
    assert "FileNotFoundError" not in err

    # The handler was rebuilt: the directory + log exist again and the second
    # record landed (the first was lost with the trashed tree, as expected).
    records = _read_agent_log(proj)
    assert [r["event"] for r in records] == ["task_completed"]


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


def test_router_scaffold_to_apply_threads_failure_signatures():
    """A regenerated SCAFFOLD's flap ledger flows into the new APPLY."""
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    sc = TaskNode.new(
        session.session_id, TaskKind.SCAFFOLD, "scaffold",
        inputs={"prior_goal": "g",
                "work_plan_artifact_id": "art_plan",
                "prior_failure_signatures": ["verify|collection_error|x"]})
    sc.produced_artifact_id = "art_scaffold"
    sc.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=sc, tasks=[sc])
    ap = [a.task for a in plan.actions if isinstance(a, CreateTask)][0]
    assert ap.kind is TaskKind.APPLY
    assert ap.inputs["prior_failure_signatures"] == [
        "verify|collection_error|x"]
    # A first-generation SCAFFOLD (no ledger) adds no key at all.
    sc2 = TaskNode.new(
        session.session_id, TaskKind.SCAFFOLD, "scaffold",
        inputs={"prior_goal": "g", "work_plan_artifact_id": "art_plan"})
    sc2.produced_artifact_id = "art_scaffold2"
    sc2.status = TaskNodeStatus.DONE
    plan2 = Router().on_task_completed(
        session=session, completed=sc2, tasks=[sc2])
    ap2 = [a.task for a in plan2.actions if isinstance(a, CreateTask)][0]
    assert "prior_failure_signatures" not in ap2.inputs


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

    def fake_manifest(composed, provider, goal=None, skills=None, **kwargs):
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
    assert result.artifact is not None
    assert result.artifact.kind is ArtifactKind.WORK_PLAN
    layers = result.artifact.content["layers"]
    # _order_manifest_layers regroups files into strict pipeline buckets
    # (models/utils -> core -> api/main -> tests); app.py lands in the
    # api bucket and README.md in the core bucket, so assert presence
    # across the flattened tree rather than a fixed slot.
    paths = [f["path"] for lay in layers for f in lay["files"]]
    assert "app.py" in paths
    assert result.outputs["file_count"] == 2
    assert result.outputs["layer_count"] == 2
    # The composed goal carried the user's answer through to the planner.
    assert "Python + Flask" in result.artifact.content["composed_goal"]


def test_decompose_executor_stores_contracts_on_work_plan(store, monkeypatch):
    """P0: a planner ``contracts`` block is normalized onto the WORK_PLAN."""
    from cgx.session.tasks.decompose import run_decompose
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)

    def fake_manifest(composed, provider, goal=None, skills=None, **kwargs):
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


# ------------- P1.1: deterministic scope calibration -------------

def test_estimate_scope_trivial_goal_is_tight_and_minimal():
    """A bare 'calculator' names no heavy feature -> trivial, tight ceiling."""
    from cgx.session.scope import estimate_scope
    profile = estimate_scope("build a small calculator")
    assert profile.complexity == "trivial"
    assert profile.max_files == 5
    assert profile.requested_features == ()
    # The injected constraint fences off the exact over-scoping we saw.
    assert "SCOPE CEILING" in profile.constraint
    assert "at most 5 files" in profile.constraint
    assert "a database, ORM, or migrations" in profile.constraint
    assert "browser/E2E tests" in profile.constraint


def test_estimate_scope_detects_requested_features_and_climbs():
    """Explicitly-named capabilities raise the tier and are NOT fenced off."""
    from cgx.session.scope import estimate_scope
    profile = estimate_scope(
        "a FastAPI service with a Postgres database, JWT auth and a React UI")
    assert profile.complexity == "complex"
    assert profile.max_files == 20
    assert set(profile.requested_features) == {
        "api_server", "database", "auth", "frontend"}
    # Requested categories must not appear in the "do NOT introduce" line.
    assert "a database, ORM, or migrations" not in profile.constraint
    assert "authentication or user accounts" not in profile.constraint


def test_clarify_records_project_complexity_on_requirements_sheet():
    """CLARIFY stamps the goal-level complexity tier onto the sheet."""
    from cgx.session.tasks.clarify_requirements import run_clarify_requirements
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    t = TaskNode.new(session.session_id, TaskKind.CLARIFY_REQUIREMENTS,
                     "c", inputs={"goal": "build a small calculator"})
    result = run_clarify_requirements(t, ExecutorDeps())
    assert result.failure is None
    assert result.artifact.content["project_complexity"] == "trivial"
    assert result.artifact.content["scope"]["max_files"] == 5
    assert result.outputs["project_complexity"] == "trivial"


def test_decompose_threads_scope_constraint_into_planner(store, monkeypatch):
    """DECOMPOSE passes the minimal-stack ceiling into the manifest planner
    and records the complexity tier on the WORK_PLAN."""
    from cgx.session.tasks.decompose import run_decompose
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    seen = {}

    def fake_manifest(composed, provider, goal=None, skills=None,
                      scope_constraint=None, **kwargs):
        seen["scope_constraint"] = scope_constraint
        return {"plan_md": "p", "layers": [{"name": "core", "files": [
            {"path": "calculator.py", "description": "core"},
            {"path": "tests/test_calculator.py", "description": "tests"}]}]}

    monkeypatch.setattr("cgx.answer.engine.plan_scaffold_manifest",
                        fake_manifest)
    t = TaskNode.new(session.session_id, TaskKind.DECOMPOSE, "d",
                     inputs={"prior_goal": "build a calculator", "answers": {}})
    result = run_decompose(
        t, ExecutorDeps(provider=_StubProvider(""), store=store))
    assert result.failure is None
    assert seen["scope_constraint"] and "SCOPE CEILING" in seen["scope_constraint"]
    assert result.artifact.content["project_complexity"] == "trivial"
    assert result.artifact.content["scope"]["max_files"] == 5
    assert result.outputs["project_complexity"] == "trivial"
    assert result.outputs["scope_max_files"] == 5


def test_scope_estimate_traced_to_agent_log_when_enabled(tmp_path):
    """With tracing toggled on, the scope estimate lands as a trace record."""
    from cgx.session.tasks.clarify_requirements import run_clarify_requirements
    from cgx.session import agent_log
    from cgx import trace as trace_mod

    agent_log.reset_for_tests()
    trace_mod.reset_for_tests()
    if trace_mod.is_trace_enabled():  # env-pinned: nothing to assert against
        return
    trace_mod.set_trace_enabled(True)
    token = trace_mod.set_trace_context(
        session_id="ses_scope", task_id="task_scope",
        project_root=str(tmp_path))
    try:
        session = Session.new("g", mode=SessionMode.GREENFIELD)
        t = TaskNode.new(session.session_id, TaskKind.CLARIFY_REQUIREMENTS,
                         "c", inputs={"goal": "build a small calculator"})
        run_clarify_requirements(t, ExecutorDeps())
    finally:
        trace_mod.reset_trace_context(token)
        trace_mod.reset_for_tests()
        agent_log.reset_for_tests()

    records = _read_agent_log(tmp_path)
    scope_events = [r for r in records if r.get("event") == "scope_estimate"]
    assert scope_events, "scope_estimate should be traced when enabled"
    assert scope_events[0]["stage"] == "clarify"
    assert scope_events[0]["complexity"] == "trivial"


# ------------- P1.2: bounded plan self-critique -------------

def _bloated_manifest():
    """A calculator manifest carrying a speculative DB + auth layer."""
    return {"plan_md": "p", "layers": [{"name": "core", "files": [
        {"path": "calculator.py", "description": "core arithmetic"},
        {"path": "database.py", "description": "speculative persistence"},
        {"path": "auth.py", "description": "speculative auth"},
        {"path": "tests/test_calculator.py", "description": "unit tests",
         "depends_on": ["calculator.py"]}]}]}


def test_critique_scaffold_manifest_flags_only_manifest_paths():
    """The engine helper returns flagged paths present in the manifest and
    drops hallucinated ones; a missing provider is a no-op."""
    import json as _json
    from cgx.answer.engine import critique_scaffold_manifest
    layers = _bloated_manifest()["layers"]
    provider = _StubProvider(_json.dumps(
        {"remove": ["database.py", "auth.py", "ghost.py"]}))
    flagged = critique_scaffold_manifest(
        "build a small calculator", layers, provider,
        scope_constraint="SCOPE CEILING")
    assert flagged == ["database.py", "auth.py"]
    assert critique_scaffold_manifest("g", layers, None) == []


def test_critique_scaffold_manifest_degrades_on_bad_reply():
    """A non-JSON / error reply yields no removals (today's behaviour)."""
    from cgx.answer.engine import critique_scaffold_manifest
    layers = _bloated_manifest()["layers"]
    assert critique_scaffold_manifest(
        "g", layers, _StubProvider("not json at all")) == []

    class _Boom:
        def chat(self, **kw):
            raise RuntimeError("model down")

    assert critique_scaffold_manifest("g", layers, _Boom()) == []


def test_decompose_applies_plan_critique_drops_speculative_files(
        store, monkeypatch):
    """DECOMPOSE folds the critique's safe removals out of the WORK_PLAN."""
    from cgx.session.tasks.decompose import run_decompose
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    monkeypatch.setattr("cgx.answer.engine.plan_scaffold_manifest",
                        lambda *a, **kw: _bloated_manifest())
    monkeypatch.setattr("cgx.answer.engine.critique_scaffold_manifest",
                        lambda *a, **kw: ["database.py", "auth.py"])
    t = TaskNode.new(session.session_id, TaskKind.DECOMPOSE, "d",
                     inputs={"prior_goal": "build a calculator", "answers": {}})
    result = run_decompose(
        t, ExecutorDeps(provider=_StubProvider(""), store=store))
    assert result.failure is None
    paths = {f["path"] for lay in result.artifact.content["layers"]
             for f in lay["files"]}
    assert "database.py" not in paths and "auth.py" not in paths
    assert "calculator.py" in paths
    assert "tests/test_calculator.py" in paths
    assert result.outputs["critique_removed"] == 2


def test_decompose_critique_guardrail_protects_depends_on_and_source(
        store, monkeypatch):
    """A critique that would drop the core module (a depends_on target and
    the only source file) is refused wholesale."""
    from cgx.session.tasks.decompose import run_decompose
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    monkeypatch.setattr("cgx.answer.engine.plan_scaffold_manifest",
                        lambda *a, **kw: _bloated_manifest())
    # The model over-reaches and flags the core module the tests depend on.
    monkeypatch.setattr("cgx.answer.engine.critique_scaffold_manifest",
                        lambda *a, **kw: ["calculator.py"])
    t = TaskNode.new(session.session_id, TaskKind.DECOMPOSE, "d",
                     inputs={"prior_goal": "build a calculator", "answers": {}})
    result = run_decompose(
        t, ExecutorDeps(provider=_StubProvider(""), store=store))
    assert result.failure is None
    paths = {f["path"] for lay in result.artifact.content["layers"]
             for f in lay["files"]}
    assert "calculator.py" in paths
    assert result.outputs["critique_removed"] == 0


def test_plan_critique_traced_to_agent_log_when_enabled(
        store, monkeypatch, tmp_path):
    """With tracing on, the critique outcome lands as a trace record."""
    from cgx.session.tasks.decompose import run_decompose
    from cgx.session import agent_log
    from cgx import trace as trace_mod

    agent_log.reset_for_tests()
    trace_mod.reset_for_tests()
    if trace_mod.is_trace_enabled():  # env-pinned: nothing to assert against
        return
    monkeypatch.setattr("cgx.answer.engine.plan_scaffold_manifest",
                        lambda *a, **kw: _bloated_manifest())
    monkeypatch.setattr("cgx.answer.engine.critique_scaffold_manifest",
                        lambda *a, **kw: ["database.py"])
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    trace_mod.set_trace_enabled(True)
    token = trace_mod.set_trace_context(
        session_id=session.session_id, task_id="task_dec",
        project_root=str(tmp_path))
    try:
        t = TaskNode.new(session.session_id, TaskKind.DECOMPOSE, "d",
                         inputs={"prior_goal": "build a calculator",
                                 "answers": {}})
        run_decompose(t, ExecutorDeps(provider=_StubProvider(""), store=store))
    finally:
        trace_mod.reset_trace_context(token)
        trace_mod.set_trace_enabled(False)

    records = _read_agent_log(tmp_path)
    events = [r for r in records if r.get("event") == "plan_critique"]
    assert events, "plan_critique should be traced when enabled"
    assert events[0]["stage"] == "decompose"
    assert events[0]["removed"] == ["database.py"]
    assert events[0]["removed_count"] == 1


# ------------- P1.3: coherence surgery as a plan-quality signal -------------

def _cyclic_py_manifest():
    """Three independent 2-cycles: the gate breaks three back-edges, so the
    surgery score clears COHERENCE_MUTATION_THRESHOLD. Pure Python (no
    client/server seam) so the exhausted-budget path reaches WORK_PLAN."""
    return {"plan_md": "p", "layers": [{"name": "core", "files": [
        {"path": "a.py", "description": "a", "depends_on": ["b.py"]},
        {"path": "b.py", "description": "b", "depends_on": ["a.py"]},
        {"path": "c.py", "description": "c", "depends_on": ["d.py"]},
        {"path": "d.py", "description": "d", "depends_on": ["c.py"]},
        {"path": "e.py", "description": "e", "depends_on": ["f.py"]},
        {"path": "f.py", "description": "f", "depends_on": ["e.py"]}]}]}


def test_decompose_reasks_when_coherence_surgery_is_heavy(store, monkeypatch):
    """A manifest the gate had to rewrite heavily is re-planned once rather
    than scaffolded, using the existing DECOMPOSE_RETRY_BUDGET path."""
    result = _run_decompose_with_manifest(
        store, monkeypatch, _cyclic_py_manifest())
    assert result.artifact is None
    assert result.retryable is True
    assert result.failure and "heavy structural repair" in result.failure
    assert "dependency cycle" in result.failure


def test_decompose_proceeds_after_reask_budget_exhausted(store, monkeypatch):
    """On the retry (budget spent) the repaired manifest ships -- P1.3 is
    never worse than today's in-place repair."""
    from cgx.session.tasks.decompose import run_decompose
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    monkeypatch.setattr("cgx.answer.engine.plan_scaffold_manifest",
                        lambda *a, **kw: _cyclic_py_manifest())
    t = TaskNode.new(session.session_id, TaskKind.DECOMPOSE, "d",
                     inputs={"prior_goal": "g", "answers": {},
                             "decompose_retry": 1})
    result = run_decompose(
        t, ExecutorDeps(provider=_StubProvider(""), store=store))
    assert result.failure is None
    assert result.outputs["coherence_surgery"] >= 3


def test_decompose_light_surgery_does_not_reask(store, monkeypatch):
    """A single routine repair stays below the threshold and ships."""
    result = _run_decompose_with_manifest(store, monkeypatch, {
        "plan_md": "p", "layers": [{"name": "core", "files": [
            {"path": "a.py", "description": "a", "depends_on": ["b.py"]},
            {"path": "b.py", "description": "b", "depends_on": ["a.py"]},
            {"path": "c.py", "description": "c"}]}]})
    assert result.failure is None
    assert result.outputs["coherence_surgery"] == 1


def test_plan_coherence_traced_to_agent_log_when_enabled(
        store, monkeypatch, tmp_path):
    """With tracing on, the coherence surgery tally lands as a trace record."""
    from cgx.session.tasks.decompose import run_decompose
    from cgx.session import agent_log
    from cgx import trace as trace_mod

    agent_log.reset_for_tests()
    trace_mod.reset_for_tests()
    if trace_mod.is_trace_enabled():  # env-pinned: nothing to assert against
        return
    monkeypatch.setattr("cgx.answer.engine.plan_scaffold_manifest",
                        lambda *a, **kw: _cyclic_py_manifest())
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    trace_mod.set_trace_enabled(True)
    token = trace_mod.set_trace_context(
        session_id=session.session_id, task_id="task_dec",
        project_root=str(tmp_path))
    try:
        t = TaskNode.new(session.session_id, TaskKind.DECOMPOSE, "d",
                         inputs={"prior_goal": "g", "answers": {}})
        run_decompose(t, ExecutorDeps(provider=_StubProvider(""), store=store))
    finally:
        trace_mod.reset_trace_context(token)
        trace_mod.set_trace_enabled(False)

    records = _read_agent_log(tmp_path)
    events = [r for r in records if r.get("event") == "plan_coherence"]
    assert events, "plan_coherence should be traced when enabled"
    assert events[0]["stage"] == "decompose"
    assert events[0]["broken_cycles"] == 3
    assert events[0]["surgery_score"] >= 3


# ------------- P1.4: dependency/test de-scoping -------------

def test_unrunnable_descope_needles_gated_on_request():
    """Browser/E2E needles fire only when the goal did NOT request E2E."""
    from cgx.session.scope import unrunnable_descope_needles
    needles = unrunnable_descope_needles(())
    assert "selenium" in needles and "e2e" in needles
    # An explicit request suppresses the de-scope entirely.
    assert unrunnable_descope_needles(("browser_e2e",)) == ()


def test_remove_from_requirements_symmetric_and_idempotent(tmp_path):
    """remove_from_requirements drops matched lines, keeps the rest, and is
    a no-op on a second run (the symmetric counterpart to update)."""
    from cgx.codegen.env_manager import remove_from_requirements
    req = tmp_path / "requirements.txt"
    req.write_text("flask==2.1\n# a comment\nselenium>=4.0\nrequests\n",
                   encoding="utf-8")
    removed = remove_from_requirements(str(tmp_path), ["Selenium"])
    assert removed == ["selenium"]
    text = req.read_text(encoding="utf-8")
    assert "selenium" not in text.lower()
    assert "flask==2.1" in text and "requests" in text
    assert "# a comment" in text
    # Already absent -> no-op (idempotent).
    assert remove_from_requirements(str(tmp_path), ["selenium"]) == []


def _e2e_manifest():
    """A calculator manifest carrying an unrequested selenium E2E suite
    alongside a real unit test, so the de-scope has a test to keep."""
    return {"plan_md": "p", "layers": [{"name": "core", "files": [
        {"path": "calculator.py", "description": "core arithmetic"},
        {"path": "tests/test_calculator.py", "description": "unit tests",
         "depends_on": ["calculator.py"]},
        {"path": "tests/test_e2e.py",
         "description": "selenium browser end-to-end flow",
         "depends_on": ["calculator.py"]}]}]}


def test_decompose_descopes_unrequested_e2e_files(store, monkeypatch):
    """A goal that never asked for E2E has its selenium suite de-scoped
    before scaffolding; the unit test and source survive."""
    result = _run_decompose_with_manifest(store, monkeypatch, _e2e_manifest())
    assert result.failure is None
    paths = {f["path"] for lay in result.artifact.content["layers"]
             for f in lay["files"]}
    assert "tests/test_e2e.py" not in paths
    assert "calculator.py" in paths
    assert "tests/test_calculator.py" in paths
    assert result.outputs["descoped_files"] == 1


def test_decompose_keeps_explicitly_requested_e2e(store, monkeypatch):
    """When the goal spells out Selenium E2E, the suite is kept (honoured)."""
    from cgx.session.tasks.decompose import run_decompose
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    monkeypatch.setattr("cgx.answer.engine.plan_scaffold_manifest",
                        lambda *a, **kw: _e2e_manifest())
    t = TaskNode.new(session.session_id, TaskKind.DECOMPOSE, "d",
                     inputs={"prior_goal":
                             "a calculator with selenium e2e tests",
                             "answers": {}})
    result = run_decompose(
        t, ExecutorDeps(provider=_StubProvider(""), store=store))
    assert result.failure is None
    paths = {f["path"] for lay in result.artifact.content["layers"]
             for f in lay["files"]}
    assert "tests/test_e2e.py" in paths
    assert result.outputs["descoped_files"] == 0


def test_plan_descope_traced_to_agent_log_when_enabled(
        store, monkeypatch, tmp_path):
    """With tracing on, the de-scope outcome lands as a trace record."""
    from cgx.session.tasks.decompose import run_decompose
    from cgx.session import agent_log
    from cgx import trace as trace_mod

    agent_log.reset_for_tests()
    trace_mod.reset_for_tests()
    if trace_mod.is_trace_enabled():  # env-pinned: nothing to assert against
        return
    monkeypatch.setattr("cgx.answer.engine.plan_scaffold_manifest",
                        lambda *a, **kw: _e2e_manifest())
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    trace_mod.set_trace_enabled(True)
    token = trace_mod.set_trace_context(
        session_id=session.session_id, task_id="task_dec",
        project_root=str(tmp_path))
    try:
        t = TaskNode.new(session.session_id, TaskKind.DECOMPOSE, "d",
                         inputs={"prior_goal": "g", "answers": {}})
        run_decompose(t, ExecutorDeps(provider=_StubProvider(""), store=store))
    finally:
        trace_mod.reset_trace_context(token)
        trace_mod.set_trace_enabled(False)

    records = _read_agent_log(tmp_path)
    events = [r for r in records if r.get("event") == "plan_descope"]
    assert events, "plan_descope should be traced when enabled"
    assert events[0]["stage"] == "decompose"
    assert events[0]["removed"] == ["tests/test_e2e.py"]
    assert events[0]["removed_count"] == 1


def test_bootstrap_descopes_dead_e2e_dependency(tmp_path):
    """A declared selenium that no applied file imports is scrubbed from
    requirements.txt; a package the code uses is left in place."""
    from cgx.session.tasks.bootstrap_env import _descope_dead_e2e_requirements
    (tmp_path / "requirements.txt").write_text(
        "flask==2.1\nselenium>=4.0\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("import flask\n", encoding="utf-8")
    removed = _descope_dead_e2e_requirements(tmp_path, ["app.py"])
    assert removed == ["selenium"]
    text = (tmp_path / "requirements.txt").read_text(encoding="utf-8")
    assert "selenium" not in text.lower() and "flask" in text


def test_bootstrap_keeps_imported_e2e_dependency(tmp_path):
    """selenium stays when an applied file actually imports it."""
    from cgx.session.tasks.bootstrap_env import _descope_dead_e2e_requirements
    (tmp_path / "requirements.txt").write_text(
        "selenium>=4.0\n", encoding="utf-8")
    (tmp_path / "test_e2e.py").write_text(
        "from selenium import webdriver\n", encoding="utf-8")
    assert _descope_dead_e2e_requirements(tmp_path, ["test_e2e.py"]) == []
    assert "selenium" in (tmp_path / "requirements.txt").read_text(
        encoding="utf-8").lower()


def test_dependency_descope_traced_to_agent_log_when_enabled(tmp_path):
    """With tracing on, the dead-dependency scrub lands as a trace record."""
    from cgx.session.tasks.bootstrap_env import _descope_dead_e2e_requirements
    from cgx.session import agent_log
    from cgx import trace as trace_mod

    agent_log.reset_for_tests()
    trace_mod.reset_for_tests()
    if trace_mod.is_trace_enabled():  # env-pinned: nothing to assert against
        return
    (tmp_path / "requirements.txt").write_text(
        "flask==2.1\nselenium>=4.0\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("import flask\n", encoding="utf-8")
    trace_mod.set_trace_enabled(True)
    token = trace_mod.set_trace_context(
        session_id="s", task_id="task_boot", project_root=str(tmp_path))
    try:
        removed = _descope_dead_e2e_requirements(tmp_path, ["app.py"])
    finally:
        trace_mod.reset_trace_context(token)
        trace_mod.set_trace_enabled(False)

    assert removed == ["selenium"]
    records = _read_agent_log(tmp_path)
    events = [r for r in records if r.get("event") == "dependency_descope"]
    assert events, "dependency_descope should be traced when enabled"
    assert events[0]["stage"] == "bootstrap_env"
    assert events[0]["removed"] == ["selenium"]
    assert events[0]["removed_count"] == 1


# ------------- P0a: mandatory cross-seam endpoint contracts -------------

def test_is_client_server_manifest_detects_and_abstains():
    from cgx.session.tasks.decompose import _is_client_server_manifest
    seam = [{"name": "app", "files": [
        {"path": "backend/app.py", "description": "Flask API with /calc route"},
        {"path": "src/App.jsx", "description": "React UI"}]}]
    assert _is_client_server_manifest(seam, None, "calc app") is True
    # Pure frontend -> no backend route.
    fe = [{"name": "ui", "files": [
        {"path": "src/App.jsx", "description": "React UI"}]}]
    assert _is_client_server_manifest(fe, None, "react app") is False
    # Pure backend -> no frontend caller.
    be = [{"name": "core", "files": [
        {"path": "backend/app.py", "description": "Flask API"}]}]
    assert _is_client_server_manifest(be, None, "flask api") is False


def test_backend_route_detected_via_skill_and_signal():
    from cgx.session.tasks.decompose import _has_backend_route
    files = [{"name": "x", "files": [
        {"path": "server/core.py", "description": "logic"}]}]
    # A framework skill promotes any .py file to a backend route.
    assert _has_backend_route(files, ["flask"], "") is True
    # No skill, no basename/description signal -> not a route.
    assert _has_backend_route(files, None, "") is False
    # A description signal alone qualifies.
    sig = [{"name": "x", "files": [
        {"path": "server/core.py",
         "description": "defines the REST API routes"}]}]
    assert _has_backend_route(sig, None, "") is True


def test_extract_endpoint_contracts_parses_provider_reply():
    from cgx.answer.engine import extract_endpoint_contracts
    reply = ('{"endpoints": [{"method": "post", "path": "/calculate", '
             '"request": {"num1": "number", "num2": "number", '
             '"operation": "str"}, "status": 201}]}')
    prov = _StubProvider(reply)
    eps = extract_endpoint_contracts("calc", [{"name": "a", "files": [
        {"path": "backend/app.py", "description": "flask"}]}], prov)
    assert len(eps) == 1 and eps[0]["path"] == "/calculate"
    # The success status survives contract normalization as an int.
    assert eps[0]["status"] == 201
    # No files -> nothing to extract; no provider -> abstain.
    assert extract_endpoint_contracts("calc", [], prov) == []
    assert extract_endpoint_contracts("calc", [{"name": "a", "files": [
        {"path": "backend/app.py"}]}], None) == []


def test_render_contracts_declares_success_status_and_message():
    """The endpoint contract prompt states the success status and message.

    Both the handler and the paired test are generated from this fragment,
    so the response contract (status + message) must appear verbatim.
    """
    from cgx.answer.engine import _render_contracts_for_prompt
    rendered = _render_contracts_for_prompt({"endpoints": [{
        "method": "POST", "path": "/register", "status": 201,
        "message": "user created"}]})
    assert "POST /register" in rendered
    assert "success_status=201" in rendered
    assert "user created" in rendered
    # A boolean/garbage status is dropped rather than rendered as 1/0.
    no_status = _render_contracts_for_prompt({"endpoints": [{
        "method": "GET", "path": "/health", "status": True}]})
    assert "success_status" not in no_status


def _cross_seam_manifest():
    return {"plan_md": "p", "layers": [{"name": "app", "files": [
        {"path": "backend/app.py",
         "description": "Flask API with /calculate route"},
        {"path": "src/App.jsx", "description": "React UI fetches /calculate"},
        {"path": "index.html", "description": "vite entry"},
        {"path": "vite.config.js", "description": "vite config"}]}]}


def test_decompose_cross_seam_requires_endpoints_fail_closed(store, monkeypatch):
    """A JSX+backend manifest with no endpoints contract fails closed."""
    from cgx.session.tasks.decompose import run_decompose
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    monkeypatch.setattr("cgx.answer.engine.plan_scaffold_manifest",
                        lambda *a, **kw: _cross_seam_manifest())
    monkeypatch.setattr("cgx.answer.engine.extract_endpoint_contracts",
                        lambda *a, **kw: [])
    t = TaskNode.new(session.session_id, TaskKind.DECOMPOSE, "d",
                     inputs={"prior_goal": "calc app", "answers": {}})
    result = run_decompose(
        t, ExecutorDeps(provider=_StubProvider(""), store=store))
    assert result.failure and "endpoints contract" in result.failure
    # Fail-closed is terminal, not a retry loop.
    assert result.retryable is False


def test_decompose_cross_seam_recovers_endpoints_via_extract(store, monkeypatch):
    """The bounded extract pass supplies the endpoints the planner omitted."""
    from cgx.session.tasks.decompose import run_decompose
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    monkeypatch.setattr("cgx.answer.engine.plan_scaffold_manifest",
                        lambda *a, **kw: _cross_seam_manifest())
    monkeypatch.setattr(
        "cgx.answer.engine.extract_endpoint_contracts",
        lambda goal, layers, provider: [
            {"method": "POST", "path": "/calculate",
             "request": {"num1": "number", "num2": "number",
                         "operation": "str"}}])
    t = TaskNode.new(session.session_id, TaskKind.DECOMPOSE, "d",
                     inputs={"prior_goal": "calc app", "answers": {}})
    result = run_decompose(
        t, ExecutorDeps(provider=_StubProvider(""), store=store))
    assert result.failure is None
    eps = result.artifact.content["contracts"]["endpoints"]
    assert eps[0]["path"] == "/calculate"
    assert result.outputs["contract_count"] == 1


def test_decompose_cross_seam_keeps_declared_endpoints(store, monkeypatch):
    """A planner that already declared endpoints skips the extract pass."""
    from cgx.session.tasks.decompose import run_decompose
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    manifest = _cross_seam_manifest()
    manifest["contracts"] = {
        "endpoints": [{"method": "POST", "path": "/calculate"}]}
    monkeypatch.setattr("cgx.answer.engine.plan_scaffold_manifest",
                        lambda *a, **kw: manifest)

    def _boom(*a, **kw):
        raise AssertionError("extract pass must not run when endpoints exist")
    monkeypatch.setattr(
        "cgx.answer.engine.extract_endpoint_contracts", _boom)
    t = TaskNode.new(session.session_id, TaskKind.DECOMPOSE, "d",
                     inputs={"prior_goal": "calc app", "answers": {}})
    result = run_decompose(
        t, ExecutorDeps(provider=_StubProvider(""), store=store))
    assert result.failure is None
    eps = result.artifact.content["contracts"]["endpoints"]
    assert eps[0]["path"] == "/calculate"


def test_decompose_python_only_manifest_not_cross_seam(store, monkeypatch):
    """A pure-Python manifest with no endpoints is NOT forced to fail."""
    from cgx.session.tasks.decompose import run_decompose
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    monkeypatch.setattr(
        "cgx.answer.engine.plan_scaffold_manifest",
        lambda *a, **kw: {"plan_md": "p", "layers": [{"name": "core", "files": [
            {"path": "app.py", "description": "Flask API"},
            {"path": "tests/test_app.py", "description": "tests"}]}]})

    def _boom(*a, **kw):
        raise AssertionError("extract must not run for a non-seam manifest")
    monkeypatch.setattr(
        "cgx.answer.engine.extract_endpoint_contracts", _boom)
    t = TaskNode.new(session.session_id, TaskKind.DECOMPOSE, "d",
                     inputs={"prior_goal": "flask api", "answers": {}})
    result = run_decompose(
        t, ExecutorDeps(provider=_StubProvider(""), store=store))
    assert result.failure is None
    assert result.artifact.content["contracts"] == {}


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
    # src/util.py buckets ahead of src/app.py (models/utils layer before
    # the api layer), so the dependency is still generated first; assert
    # the ordering across the flattened tree.
    paths = [f["path"] for lay in result.artifact.content["layers"]
             for f in lay["files"]]
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


def test_decompose_prunes_dangling_dependency(store, monkeypatch):
    # A phantom depends_on entry is pruned in place (not fatal): the
    # manifest still builds and the offending hint is dropped.
    result = _run_decompose_with_manifest(store, monkeypatch, {
        "plan_md": "p",
        "layers": [{"name": "core", "files": [
            {"path": "src/app.py", "description": "entry",
             "depends_on": ["src/util.py", "src/missing.py"]},
            {"path": "src/util.py", "description": "helpers"}]}],
    })
    assert result.failure is None
    files = [f for lay in result.artifact.content["layers"]
             for f in lay["files"]]
    app = next(f for f in files if f["path"] == "src/app.py")
    assert app["depends_on"] == ["src/util.py"]


def test_decompose_prunes_glob_dependency(store, monkeypatch):
    # A wildcard depends_on (a legitimate intent expressed wrong) is
    # pruned rather than sinking the whole re-plan.
    result = _run_decompose_with_manifest(store, monkeypatch, {
        "plan_md": "p",
        "layers": [{"name": "core", "files": [
            {"path": "src/App.jsx", "description": "entry",
             "depends_on": ["src/components/*.jsx"]}]}],
    })
    assert result.failure is None
    files = result.artifact.content["layers"][0]["files"]
    assert files[0]["depends_on"] == []


def test_decompose_breaks_dependency_cycle(store, monkeypatch):
    # depends_on is only an ordering hint, so a cycle (a routine slip
    # for small local models) is repaired in place -- the back-edge is
    # dropped -- instead of terminally failing the session.
    result = _run_decompose_with_manifest(store, monkeypatch, {
        "plan_md": "p",
        "layers": [{"name": "core", "files": [
            {"path": "a.py", "description": "a", "depends_on": ["b.py"]},
            {"path": "b.py", "description": "b", "depends_on": ["a.py"]}]}],
    })
    assert result.failure is None
    files = result.artifact.content["layers"][0]["files"]
    deps = {f["path"]: f.get("depends_on") or [] for f in files}
    # Exactly one edge survives; the surviving edge still orders the pair.
    assert sorted(len(d) for d in deps.values()) == [0, 1]
    paths = [f["path"] for f in files]
    kept_src = next(p for p, d in deps.items() if d)
    assert paths.index(deps[kept_src][0]) < paths.index(kept_src)


def test_decompose_breaks_self_and_three_node_cycle(store, monkeypatch):
    # A longer cycle (a -> b -> c -> a) is also broken deterministically
    # and the manifest proceeds to a valid topological order.
    result = _run_decompose_with_manifest(store, monkeypatch, {
        "plan_md": "p",
        "layers": [{"name": "core", "files": [
            {"path": "a.py", "description": "a", "depends_on": ["b.py"]},
            {"path": "b.py", "description": "b", "depends_on": ["c.py"]},
            {"path": "c.py", "description": "c", "depends_on": ["a.py"]}]}],
    })
    assert result.failure is None
    files = result.artifact.content["layers"][0]["files"]
    paths = [f["path"] for f in files]
    for f in files:
        for dep in f.get("depends_on") or []:
            assert paths.index(dep) < paths.index(f["path"])


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


def test_decompose_injects_missing_vite_entry_html(store, monkeypatch):
    # A Vite manifest without a root index.html cannot build at all, and
    # the regenerate loop can never add the file -- DECOMPOSE folds it in.
    result = _run_decompose_with_manifest(store, monkeypatch, {
        "plan_md": "p",
        "layers": [
            {"name": "ui", "files": [
                {"path": "src/main.jsx", "description": "entry"},
                {"path": "src/App.jsx", "description": "root component"}]},
            {"name": "config", "files": [
                {"path": "vite.config.js", "description": "vite config"},
                {"path": "package.json", "description": "manifest"}]},
        ],
    })
    assert result.failure is None
    layers = result.artifact.content["layers"]
    paths = [f["path"] for lay in layers for f in lay["files"]]
    assert "index.html" in paths
    assert result.outputs["file_count"] == 5


def test_decompose_keeps_existing_vite_entry_html(store, monkeypatch):
    # Already coherent -> no injection, no duplicate entry.
    result = _run_decompose_with_manifest(store, monkeypatch, {
        "plan_md": "p",
        "layers": [{"name": "ui", "files": [
            {"path": "index.html", "description": "entry html"},
            {"path": "src/main.jsx", "description": "entry"},
            {"path": "vite.config.js", "description": "vite config"}]}],
    })
    assert result.failure is None
    paths = [f["path"] for lay in result.artifact.content["layers"]
             for f in lay["files"]]
    assert paths.count("index.html") == 1


def test_missing_stack_entry_files_ignores_non_vite_manifests():
    from cgx.session.scaffold_validate import missing_stack_entry_files
    assert missing_stack_entry_files(
        ["src/index.js", "package.json", "public/index.html"]) == []
    assert missing_stack_entry_files(["backend/main.py"]) == []


def test_inject_stack_entry_relocates_misfiled_vite_entry():
    """``public/index.html`` is misfiled, not missing.

    Injecting a second node made SCAFFOLD generate near-identical
    boilerplate twice, and the duplicate-content gate then dropped the
    root entry -- the one file Vite cannot build without.
    """
    from cgx.session.tasks.decompose import _inject_stack_entry_files
    layers = [{"name": "ui", "files": [
        {"path": "vite.config.js", "description": "v"},
        {"path": "public/index.html", "description": "html"},
        {"path": "src/main.jsx", "description": "e",
         "depends_on": ["public/index.html"]},
    ]}]
    assert _inject_stack_entry_files(layers) == []
    files = layers[0]["files"]
    assert [f["path"] for f in files] == [
        "vite.config.js", "index.html", "src/main.jsx"]
    assert files[2]["depends_on"] == ["index.html"]
    assert "Vite entry HTML" in files[1]["description"]


def test_inject_stack_entry_still_injects_when_truly_absent():
    from cgx.session.tasks.decompose import _inject_stack_entry_files
    layers = [{"name": "ui", "files": [
        {"path": "vite.config.js", "description": "v"},
        {"path": "src/main.jsx", "description": "e"},
    ]}]
    assert _inject_stack_entry_files(layers) == ["index.html"]
    assert [f["path"] for f in layers[0]["files"]][-1] == "index.html"


def test_cross_language_depends_on_drops_unsatisfiable_pytest():
    """A pytest file planned against JSX sources can never be written.

    Verbatim from a real manifest: the planner laid a React frontend
    beside a Python backend and covered the components with pytest.
    SCAFFOLD invents a module name to import, the phantom-import gate
    rejects it, and each regenerate invents a different one -- three
    scaffold rounds and a replan to discard one planner slip.
    """
    from cgx.session.tasks.decompose import _validate_manifest_coherence
    layers = [{"name": "project", "files": [
        {"path": "src/App.jsx", "description": "a"},
        {"path": "src/main.jsx", "description": "m",
         "depends_on": ["src/App.jsx"]},
        {"path": "tests/test_main.py",
         "description": "Unit tests for main React components",
         "depends_on": ["src/main.jsx", "src/App.jsx"]},
        {"path": "backend/app.py", "description": "api"},
        {"path": "tests/test_app.py", "description": "t",
         "depends_on": ["backend/app.py"]},
    ]}]
    assert _validate_manifest_coherence(layers) is None
    paths = [f["path"] for f in layers[0]["files"]]
    assert "tests/test_main.py" not in paths
    # The same-language test is untouched.
    assert "tests/test_app.py" in paths
    assert layers[0]["files"][-1]["depends_on"] == ["backend/app.py"]


def test_cross_language_prune_spares_agnostic_and_nontest_files():
    """Only edges between two known, differing runtimes are cut.

    HTML/CSS/config/data are runtime-agnostic -- index.html referencing
    src/main.jsx and requirements.txt following backend/app.py are both
    correct, and the injected entry-point ordering depends on them
    surviving. A non-test file is never dropped: a mislinked module is
    still buildable, so only its bad edge goes.
    """
    from cgx.session.tasks.decompose import _validate_manifest_coherence
    layers = [{"name": "project", "files": [
        {"path": "index.html", "description": "h",
         "depends_on": ["src/main.jsx"]},
        {"path": "src/main.jsx", "description": "m"},
        {"path": "backend/app.py", "description": "a",
         "depends_on": ["src/main.jsx"]},
        {"path": "requirements.txt", "description": "r",
         "depends_on": ["backend/app.py"]},
    ]}]
    assert _validate_manifest_coherence(layers) is None
    by_path = {f["path"]: f.get("depends_on") for f in layers[0]["files"]}
    assert by_path["index.html"] == ["src/main.jsx"]
    assert by_path["requirements.txt"] == ["backend/app.py"]
    # Cut the edge, keep the module.
    assert by_path["backend/app.py"] == []


def test_cross_language_prune_keeps_partially_valid_test():
    """A test reaching one same-language module keeps that edge and lives."""
    from cgx.session.tasks.decompose import _validate_manifest_coherence
    layers = [{"name": "project", "files": [
        {"path": "backend/app.py", "description": "a"},
        {"path": "src/App.tsx", "description": "x"},
        {"path": "tests/test_app.py", "description": "t",
         "depends_on": ["backend/app.py", "src/App.tsx"]},
        {"path": "tests/App.test.ts", "description": "j",
         "depends_on": ["src/App.tsx"]},
    ]}]
    assert _validate_manifest_coherence(layers) is None
    by_path = {f["path"]: f.get("depends_on") for f in layers[0]["files"]}
    assert by_path["tests/test_app.py"] == ["backend/app.py"]
    # .ts -> .tsx is one family; nothing to cut.
    assert by_path["tests/App.test.ts"] == ["src/App.tsx"]


def test_import_coherence_error_names_the_project_modules():
    """A bare hallucinated root gets no locator without the inventory."""
    from cgx.session.tasks.scaffold import _import_coherence_failures
    layers = [{"files": [{"path": "backend/__init__.py"},
                         {"path": "backend/main.py"},
                         {"path": "tests/test_main.py"}]}]
    batch = [{"path": "tests/test_main.py",
              "content": "from app import client\n"}]
    failures = _import_coherence_failures(batch, layers, None)
    assert len(failures) == 1
    error = failures[0]["error"]
    assert "['app']" in error
    assert "backend.main" in error


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
                      contracts=None, skills=None, **kwargs):
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
    assert result.artifact is not None
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
                      contracts=None, skills=None, **kwargs):
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


def test_scaffold_import_coherence_gate_flags_hallucinated_import(
        store, monkeypatch):
    """A generated file importing a module that resolves nowhere fails.

    Mirrors the live failure: the model emitted ``from core import
    compute`` with no such module in the manifest, on disk, or in
    requirements.txt. Left alone the fabricated name reaches
    BOOTSTRAP_ENV where pip tries to install it and the session dies
    terminally; the gate fails the importer file here instead so the
    router's regenerate edge retries it with a concrete constraint.
    """
    from cgx.session.tasks.scaffold import run_scaffold
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    plan = Artifact.new(
        session.session_id, "task_x", ArtifactKind.WORK_PLAN, {
            "plan_md": "", "composed_goal": "g", "prior_goal": "g",
            "layers": [{"name": "app", "files": [
                {"path": "calc.py", "description": ""},
                {"path": "app.py", "description": ""}]}],
        })
    store.save_artifact(plan)

    bodies = {
        # Stdlib import only -> passes the gate.
        "calc.py": "import json\n\ndef add(a, b):\n    return a + b\n",
        # Sibling manifest import passes; the hallucinated module -- a bare
        # single-word root that resolves nowhere and reads as a
        # (possibly-forgotten) dependency, not a first-party module -- is
        # not synthesized and flags the file.
        "app.py": ("from calc import add\n"
                   "from zzhallucinatedcore import compute\n"),
    }

    def fake_generate(path, *a, **kw):
        return {"file": path, "patch": f"+++ {path}\nx",
                "content": bodies[path], "syntax_ok": True}

    monkeypatch.setattr("cgx.answer.engine.generate_single_scaffold_file",
                        fake_generate)
    t = TaskNode.new(session.session_id, TaskKind.SCAFFOLD, "s",
                     inputs={"work_plan_artifact_id": plan.artifact_id})
    result = run_scaffold(
        t, ExecutorDeps(provider=_StubProvider(""), store=store))
    assert result.failure is None
    assert result.outputs["failed_count"] == 1
    assert [d["file"] for d in result.artifact.content["diffs"]] == ["calc.py"]
    failed = result.artifact.content["failed"]
    assert failed[0]["file"] == "app.py"
    assert "zzhallucinatedcore" in failed[0]["error"]


def test_scaffold_circular_import_gate_breaks_cycle(store, monkeypatch):
    """Two generated modules importing each other fail one file per cycle.

    Mirrors the live failure (ses_c346e309fbcd4f16): backend/routes.py and
    backend/models.py imported each other, every per-file gate passed
    (each import resolves to a real sibling), and pytest collection died
    with "cannot import name ... from partially initialized module ...".
    The gate detects the cycle statically and fails the foundational
    member (models) so the regenerate constraint makes the dependency
    one-way.
    """
    from cgx.session.tasks.scaffold import run_scaffold
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    plan = Artifact.new(
        session.session_id, "task_x", ArtifactKind.WORK_PLAN, {
            "plan_md": "", "composed_goal": "g", "prior_goal": "g",
            "layers": [{"name": "backend", "files": [
                {"path": "backend/routes.py", "description": ""},
                {"path": "backend/models.py", "description": ""}]}],
        })
    store.save_artifact(plan)

    bodies = {
        "backend/routes.py": (
            "from backend.models import init_db\n\n\n"
            "def compute_expression(expr):\n"
            "    init_db()\n"
            "    return expr\n"),
        "backend/models.py": (
            "from backend.routes import compute_expression\n\n\n"
            "def init_db():\n"
            "    return compute_expression\n"),
    }

    def fake_generate(path, *a, **kw):
        return {"file": path, "patch": f"+++ {path}\nx",
                "content": bodies[path], "syntax_ok": True}

    monkeypatch.setattr("cgx.answer.engine.generate_single_scaffold_file",
                        fake_generate)
    t = TaskNode.new(session.session_id, TaskKind.SCAFFOLD, "s",
                     inputs={"work_plan_artifact_id": plan.artifact_id})
    result = run_scaffold(
        t, ExecutorDeps(provider=_StubProvider(""), store=store))
    assert result.failure is None
    assert result.outputs["failed_count"] == 1
    assert [d["file"] for d in result.artifact.content["diffs"]] == [
        "backend/routes.py"]
    failed = result.artifact.content["failed"]
    assert failed[0]["file"] == "backend/models.py"
    assert "circular import" in failed[0]["error"]
    assert "backend.routes" in failed[0]["error"]


def test_reconcile_js_dependencies_splices_missing_dep_into_package_json():
    """A runtime import missing from package.json is added to dependencies."""
    import json
    from cgx.answer.engine import _content_to_new_file_patch
    from cgx.session.tasks.scaffold import (
        _content_from_new_file_patch,
        _reconcile_js_dependencies,
    )

    pkg = json.dumps({"dependencies": {"react": ">=18", "react-dom": ">=18"}})
    diffs = [
        {"file": "package.json",
         "patch": _content_to_new_file_patch("package.json", pkg)},
        {"file": "src/App.jsx",
         "patch": _content_to_new_file_patch(
             "src/App.jsx",
             "import React from 'react';\nimport axios from 'axios';\n")},
    ]
    existing = [
        {"path": "package.json", "content": pkg},
        {"path": "src/App.jsx",
         "content": "import React from 'react';\nimport axios from 'axios';\n"},
    ]
    added = _reconcile_js_dependencies(
        diffs=diffs, existing_with_content=existing)
    assert added == ["axios"]
    # The rewritten diff carries the repaired manifest APPLY will write.
    rewritten = json.loads(
        _content_from_new_file_patch(diffs[0]["patch"]))
    assert rewritten["dependencies"]["axios"] == "*"
    assert rewritten["dependencies"]["react"] == ">=18"
    # The context copy is updated in place too, so later gates see the dep.
    assert json.loads(existing[0]["content"])["dependencies"]["axios"] == "*"


def test_reconcile_js_dependencies_noop_without_package_json():
    """No package.json in the bundle -> no-op, returns []."""
    from cgx.answer.engine import _content_to_new_file_patch
    from cgx.session.tasks.scaffold import _reconcile_js_dependencies
    diffs = [{"file": "src/App.jsx",
              "patch": _content_to_new_file_patch(
                  "src/App.jsx", "import axios from 'axios';\n")}]
    existing = [{"path": "src/App.jsx",
                 "content": "import axios from 'axios';\n"}]
    assert _reconcile_js_dependencies(
        diffs=diffs, existing_with_content=existing) == []


def test_synthesize_missing_frontend_stylesheet_stub():
    """An entry point importing an ungenerated ./index.css gets a stub."""
    from cgx.answer.engine import _content_to_new_file_patch
    from cgx.session.tasks.scaffold import (
        _content_from_new_file_patch,
        _synthesize_missing_frontend_stylesheets,
    )
    main = "import App from './App.jsx'\nimport './index.css'\n"
    diffs = [
        {"file": "src/main.jsx",
         "patch": _content_to_new_file_patch("src/main.jsx", main)},
        {"file": "src/App.jsx",
         "patch": _content_to_new_file_patch(
             "src/App.jsx", "export default 1\n")},
    ]
    generated = [{"file": "src/main.jsx"}, {"file": "src/App.jsx"}]
    existing = [
        {"path": "src/main.jsx", "content": main},
        {"path": "src/App.jsx", "content": "export default 1\n"},
    ]
    layers: list = []
    added = _synthesize_missing_frontend_stylesheets(
        diffs=diffs, generated=generated, existing_with_content=existing,
        layers=layers, project_root=None)
    assert added == ["src/index.css"]
    # Spliced into diffs + manifest so APPLY writes it and later gates see it.
    assert any(d["file"] == "src/index.css" for d in diffs)
    assert any(e["path"] == "src/index.css" for e in existing)
    stub = _content_from_new_file_patch(
        next(d["patch"] for d in diffs if d["file"] == "src/index.css"))
    assert "stylesheet stub" in stub


def test_synthesize_frontend_stylesheet_noop_when_present():
    """A stylesheet already generated is not clobbered with a stub."""
    from cgx.session.tasks.scaffold import (
        _synthesize_missing_frontend_stylesheets,
    )
    existing = [
        {"path": "src/main.jsx", "content": "import './index.css'\n"},
        {"path": "src/index.css", "content": "body{}\n"},
    ]
    assert _synthesize_missing_frontend_stylesheets(
        diffs=[], generated=[], existing_with_content=existing, layers=[],
        project_root=None) == []


def test_js_import_coherence_flags_missing_relative_script():
    """A relative script import with no generated sibling fails the importer."""
    from cgx.session.tasks.scaffold import _js_import_coherence_failures
    existing = [
        {"path": "src/main.jsx",
         "content": ("import App from './App.jsx'\n"
                     "import Missing from './Missing'\n")},
        {"path": "src/App.jsx", "content": "export default 1\n"},
    ]
    failures = _js_import_coherence_failures(existing, None)
    assert len(failures) == 1
    assert failures[0]["file"] == "src/main.jsx"
    assert "./Missing" in failures[0]["error"]
    # The resolvable sibling is not reported.
    assert "./App.jsx" not in failures[0]["error"]


def test_js_import_coherence_resolves_siblings_and_skips_assets():
    """Extensionless/index resolution passes; stylesheets and assets skip."""
    from cgx.session.tasks.scaffold import _js_import_coherence_failures
    existing = [
        {"path": "src/main.jsx",
         "content": ("import './index.css'\n"        # stylesheet -> pass A
                     "import logo from './logo.svg'\n"  # asset -> skipped
                     "import {u} from './util'\n"       # -> src/util.js
                     "import C from './comp'\n")},       # -> src/comp/index.jsx
        {"path": "src/util.js", "content": "export const u = 1\n"},
        {"path": "src/comp/index.jsx", "content": "export default 1\n"},
    ]
    assert _js_import_coherence_failures(existing, None) == []


# ---------------- P1a: JS test-harness coherence ----------------

def test_js_test_path_detection():
    from cgx.session.tasks.scaffold import _is_js_test_path
    assert _is_js_test_path("src/App.test.jsx") is True
    assert _is_js_test_path("src/util.spec.ts") is True
    assert _is_js_test_path("src/__tests__/App.jsx") is True
    assert _is_js_test_path("src/App.jsx") is False
    assert _is_js_test_path("src/App.test.py") is False


def test_synthesize_js_test_harness_backfills_react_project():
    """A React test file with a bare package.json gets a runnable harness."""
    import json
    from cgx.answer.engine import _content_to_new_file_patch  # noqa: F401
    from cgx.session.tasks.scaffold import (
        _content_from_new_file_patch,
        _synthesize_js_test_harness,
    )
    existing = [
        {"path": "package.json",
         "content": '{"name": "app", "dependencies": {"react": "^18.0.0"}}'},
        {"path": "src/App.jsx", "content": "export default () => null\n"},
        {"path": "src/App.test.jsx",
         "content": ("import { render } from '@testing-library/react'\n"
                     "import App from './App.jsx'\n")},
    ]
    diffs = [{"file": e["path"],
              "patch": _content_to_new_file_patch(e["path"], e["content"])}
             for e in existing]
    generated = [{"file": e["path"], "layer": "core", "bytes": 1}
                 for e in existing]
    layers: list = []
    touched = _synthesize_js_test_harness(
        diffs=diffs, generated=generated, existing_with_content=existing,
        layers=layers, project_root=None)
    assert "package.json" in touched
    assert "vitest.config.js" in touched
    assert "vitest.setup.js" in touched
    # package.json now carries a real test script + harness devDeps.
    pkg = json.loads(next(e["content"] for e in existing
                          if e["path"] == "package.json"))
    assert pkg["scripts"]["test"] == "vitest run"
    assert "vitest" in pkg["devDependencies"]
    assert "jsdom" in pkg["devDependencies"]
    assert "@testing-library/react" in pkg["devDependencies"]
    assert "@vitejs/plugin-react" in pkg["devDependencies"]
    # react stays where the model declared it, not duplicated into devDeps.
    assert "react" not in pkg["devDependencies"]
    # The config is jsdom + wired to the setup file, and rides the diff bundle.
    cfg = _content_from_new_file_patch(
        next(d["patch"] for d in diffs if d["file"] == "vitest.config.js"))
    assert "jsdom" in cfg and "vitest.setup.js" in cfg
    assert "plugin-react" in cfg
    # The synthesized setup wires jest-dom matchers AND a jest->vi alias so
    # a jest-dialect suite (jest.spyOn/jest.fn) runs under the harness.
    setup = next(e["content"] for e in existing
                 if e["path"] == "vitest.setup.js")
    assert "@testing-library/jest-dom" in setup
    assert "globalThis.jest = vi" in setup
    assert "import { vi } from 'vitest'" in setup


def test_vitest_setup_content_aliases_jest_to_vi():
    """The synthesized setup exposes a jest global backed by vi."""
    from cgx.session.tasks.scaffold import _vitest_setup_content
    setup = _vitest_setup_content()
    # jest-dom matchers are still imported for toBeInTheDocument() etc.
    assert "import '@testing-library/jest-dom';" in setup
    # vi is imported explicitly (robust even without globals) and aliased
    # onto the jest global so jest.spyOn/jest.fn resolve under vitest.
    assert "import { vi } from 'vitest';" in setup
    assert "globalThis.jest = vi;" in setup


def test_synthesize_js_test_harness_noop_without_tests():
    from cgx.session.tasks.scaffold import _synthesize_js_test_harness
    existing = [
        {"path": "package.json", "content": '{"name": "app"}'},
        {"path": "src/App.jsx", "content": "export default 1\n"},
    ]
    assert _synthesize_js_test_harness(
        diffs=[], generated=[], existing_with_content=existing, layers=[],
        project_root=None) == []


def test_synthesize_js_test_harness_preserves_existing_config_and_script():
    """A real test script and an existing vitest config are not clobbered."""
    from cgx.session.tasks.scaffold import _synthesize_js_test_harness
    existing = [
        {"path": "package.json",
         "content": ('{"name": "app", "scripts": {"test": "vitest"}, '
                     '"devDependencies": {"vitest": "^1.0.0"}}')},
        {"path": "vitest.config.js", "content": "export default {}\n"},
        {"path": "src/App.test.jsx", "content": "test('x', () => {})\n"},
    ]
    diffs = [{"file": e["path"], "patch": ""} for e in existing]
    touched = _synthesize_js_test_harness(
        diffs=diffs, generated=[], existing_with_content=existing,
        layers=[], project_root=None)
    # Config already present -> not re-synthesized; only devDeps may grow.
    assert "vitest.config.js" not in touched
    assert "vitest.setup.js" not in touched
    import json
    pkg = json.loads(next(e["content"] for e in existing
                          if e["path"] == "package.json"))
    assert pkg["scripts"]["test"] == "vitest"  # untouched


def test_synthesize_js_test_harness_skips_vue():
    from cgx.session.tasks.scaffold import _synthesize_js_test_harness
    existing = [
        {"path": "package.json", "content": '{"name": "app"}'},
        {"path": "src/App.vue", "content": "<template></template>\n"},
        {"path": "src/App.spec.js", "content": "test('x', () => {})\n"},
    ]
    assert _synthesize_js_test_harness(
        diffs=[], generated=[], existing_with_content=existing, layers=[],
        project_root=None) == []


def test_scaffold_synthesizes_missing_stylesheet_stub(store, monkeypatch):
    """run_scaffold backfills an omitted stylesheet so the tree builds."""
    from cgx.session.tasks.scaffold import run_scaffold
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    plan = Artifact.new(
        session.session_id, "task_x", ArtifactKind.WORK_PLAN, {
            "plan_md": "", "composed_goal": "g", "prior_goal": "g",
            "layers": [{"name": "app", "files": [
                {"path": "src/main.jsx", "description": ""}]}],
        })
    store.save_artifact(plan)
    body = ("import React from 'react'\n"
            "import './index.css'\n"
            "console.log(React)\n")

    def fake_generate(path, *a, **kw):
        return {"file": path, "patch": f"+++ {path}\nx",
                "content": body, "syntax_ok": True}

    monkeypatch.setattr("cgx.answer.engine.generate_single_scaffold_file",
                        fake_generate)
    t = TaskNode.new(session.session_id, TaskKind.SCAFFOLD, "s",
                     inputs={"work_plan_artifact_id": plan.artifact_id})
    result = run_scaffold(
        t, ExecutorDeps(provider=_StubProvider(""), store=store))
    assert result.failure is None
    files = [d["file"] for d in result.artifact.content["diffs"]]
    assert "src/index.css" in files
    assert result.outputs["failed_count"] == 0


def test_circular_import_failures_ignores_one_way_imports():
    """An acyclic first-party import graph produces no failures."""
    from cgx.session.tasks.scaffold import _circular_import_failures
    files = [
        {"path": "app/routes.py",
         "content": "from app.models import init_db\n"},
        {"path": "app/models.py", "content": "def init_db():\n    pass\n"},
    ]
    assert _circular_import_failures(files) == []


def test_circular_import_failures_detects_relative_cycle():
    """Cycles authored with relative imports are detected too."""
    from cgx.session.tasks.scaffold import _circular_import_failures
    files = [
        {"path": "pkg/alpha.py", "content": "from .beta import f\n"},
        {"path": "pkg/beta.py", "content": "from .alpha import g\n"},
    ]
    out = _circular_import_failures(files)
    # No foundational basename in the cycle -> first module in sorted
    # order is failed; exactly one file per cycle.
    assert [f["file"] for f in out] == ["pkg/alpha.py"]
    assert "circular import" in out[0]["error"]


def test_phantom_import_gate_flags_undefined_dotted_module():
    """``from backend.core import x`` with no backend/core.py is failed."""
    from cgx.session.tasks.scaffold import (
        _phantom_first_party_import_failures)
    files = [
        {"path": "backend/__init__.py", "content": ""},
        {"path": "backend/main.py",
         "content": "from backend.core import calculate\n"},
        {"path": "backend/models.py", "content": "X = 1\n"},
    ]
    layers = [{"files": [{"path": f["path"]} for f in files]}]
    out = _phantom_first_party_import_failures(files, layers)
    assert [f["file"] for f in out] == ["backend/main.py"]
    assert "backend.core" in out[0]["error"]
    assert "backend.models" in out[0]["error"]


def test_phantom_import_gate_flags_bare_import_of_packaged_module():
    """``from main import app`` when the module is backend/main.py."""
    from cgx.session.tasks.scaffold import (
        _phantom_first_party_import_failures)
    files = [
        {"path": "backend/__init__.py", "content": ""},
        {"path": "backend/main.py", "content": "app = 1\n"},
        {"path": "tests/test_main.py", "content": "from main import app\n"},
    ]
    layers = [{"files": [{"path": f["path"]} for f in files]}]
    out = _phantom_first_party_import_failures(files, layers)
    assert [f["file"] for f in out] == ["tests/test_main.py"]
    assert "backend.main" in out[0]["error"]


def test_phantom_import_gate_abstains_on_src_layout_and_manifest_peers():
    """Bare src/-layout imports and not-yet-generated manifest peers pass."""
    from cgx.session.tasks.scaffold import (
        _phantom_first_party_import_failures)
    files = [
        {"path": "src/calc.py", "content": "def add(a, b):\n    return a\n"},
        {"path": "tests/test_calc.py", "content": "from calc import add\n"},
    ]
    layers = [{"files": [{"path": f["path"]} for f in files]}]
    assert _phantom_first_party_import_failures(files, layers) == []
    # backend/routers/calculator.py is planned but not in this batch.
    files = [{"path": "backend/main.py",
              "content": "from backend.routers.calculator import router\n"}]
    layers = [{"files": [{"path": "backend/main.py"},
                         {"path": "backend/routers/calculator.py"}]}]
    assert _phantom_first_party_import_failures(files, layers) == []


def test_missing_first_party_imports_detects_module_under_test():
    """A test importing named symbols from an unplanned module is a candidate.

    Mirrors the live failure: the planner authored ``tests/test_app.py`` +
    ``tests/conftest.py`` importing ``backend.app`` but never planned
    ``backend/app.py``. The importers are correct; the module is missing, so
    it must be authored rather than dropped.
    """
    from cgx.session.tasks.scaffold import _missing_first_party_imports
    files = [
        {"path": "tests/test_app.py",
         "content": "from backend.app import app, compute\n"},
        {"path": "tests/conftest.py",
         "content": "from backend.app import app\n"},
    ]
    layers = [{"files": [{"path": f["path"]} for f in files]}]
    missing = _missing_first_party_imports(files, layers, None)
    assert set(missing) == {"backend.app"}
    rec = missing["backend.app"]
    assert rec["path"] == "backend/app.py"
    assert rec["symbols"] == {"app", "compute"}
    assert set(rec["importers"]) == {"tests/test_app.py", "tests/conftest.py"}


def test_missing_first_party_imports_ignores_stdlib_deps_and_self():
    """Stdlib, declared deps, and self-imports are never synthesis candidates."""
    from cgx.session.tasks.scaffold import _missing_first_party_imports
    files = [
        {"path": "requirements.txt", "content": "flask\n"},
        {"path": "tests/test_app.py",
         "content": ("import os\nfrom flask import Flask\n"
                     "from tests.test_app import helper\n")},
    ]
    layers = [{"files": [{"path": f["path"]} for f in files]}]
    assert _missing_first_party_imports(files, layers, None) == {}


def test_missing_first_party_imports_detects_source_named_import():
    """A snake_case named import from a *source* file is a synthesis candidate.

    Mirrors the live failure where the entry point imported an application
    module the plan omitted (``from calculation_service import compute`` in
    ``backend/app.py``). The compound root reads as first-party, so it is
    authored rather than dropping the importer.
    """
    from cgx.session.tasks.scaffold import _missing_first_party_imports
    files = [
        {"path": "backend/app.py",
         "content": "from calculation_service import compute\n"},
    ]
    layers = [{"files": [{"path": "backend/app.py"}]}]
    missing = _missing_first_party_imports(files, layers, None)
    assert set(missing) == {"calculation_service"}
    assert missing["calculation_service"]["path"] == "calculation_service.py"
    assert missing["calculation_service"]["symbols"] == {"compute"}


def test_missing_first_party_imports_skips_single_word_source_import():
    """A named single-word import from a source file is left to req repair.

    ``from cerberus import Schema`` in a non-test file names a plausible
    (forgotten) third-party dependency, not a first-party module: with no
    dotted path and no snake_case compound root there is no first-party
    signal, so it is not fabricated as an empty module here.
    """
    from cgx.session.tasks.scaffold import _missing_first_party_imports
    files = [{"path": "backend/app.py",
              "content": "from cerberus import Schema\n"}]
    layers = [{"files": [{"path": "backend/app.py"}]}]
    assert _missing_first_party_imports(files, layers, None) == {}


def test_missing_first_party_imports_ignores_common_thirdparty_frameworks():
    """Dotted third-party imports (werkzeug.security, sqlalchemy.orm) are never first-party."""
    from cgx.session.tasks.scaffold import _missing_first_party_imports
    files = [
        {"path": "backend/app.py",
         "content": ("from werkzeug.security import generate_password_hash\n"
                     "from sqlalchemy.orm import Session\n"
                     "from pydantic import BaseModel\n")},
        {"path": "tests/test_app.py",
         "content": "from werkzeug.security import check_password_hash\n"},
    ]
    layers = [{"files": [{"path": "backend/app.py"}, {"path": "tests/test_app.py"}]}]
    assert _missing_first_party_imports(files, layers, None) == {}


def test_scaffold_synthesizes_omitted_first_party_module(store, monkeypatch):
    """A planned test importing an unplanned app module gets the module authored.

    The manifest plans only ``tests/test_app.py`` (which imports
    ``backend.app``); the coherence/phantom gates would otherwise drop the
    test. The synthesis pass authors ``backend/app.py`` (plus its package
    marker) against the importer so the gates keep the test and the tree
    resolves.
    """
    from cgx.session.tasks.scaffold import run_scaffold
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    plan = Artifact.new(
        session.session_id, "task_x", ArtifactKind.WORK_PLAN, {
            "plan_md": "", "composed_goal": "g", "prior_goal": "g",
            "layers": [{"name": "app", "files": [
                {"path": "tests/test_app.py", "description": ""}]}],
        })
    store.save_artifact(plan)

    bodies = {
        "tests/test_app.py": (
            "from backend.app import compute\n\n\n"
            "def test_add():\n    assert compute(1, 1) == 2\n"),
        "backend/app.py": "def compute(a, b):\n    return a + b\n",
    }

    def fake_generate(path, *a, **kw):
        return {"file": path, "patch": f"+++ {path}\nx",
                "content": bodies[path], "syntax_ok": True}

    monkeypatch.setattr("cgx.answer.engine.generate_single_scaffold_file",
                        fake_generate)
    t = TaskNode.new(session.session_id, TaskKind.SCAFFOLD, "s",
                     inputs={"work_plan_artifact_id": plan.artifact_id})
    result = run_scaffold(
        t, ExecutorDeps(provider=_StubProvider(""), store=store))
    assert result.failure is None
    files = [d["file"] for d in result.artifact.content["diffs"]]
    assert "tests/test_app.py" in files
    assert "backend/app.py" in files
    assert "backend/__init__.py" in files
    assert result.artifact.content["failed"] == []


def test_synthesize_import_smoke_test_adds_probe_for_source_modules():
    """Planned tests all dropped -> a smoke test importing the survivors.

    The manifest planned ``tests/test_app.py`` but only source modules
    survived the gates; the synthesizer authors ``tests/test_smoke.py``
    that ``importlib.import_module()``s every first-party source module.
    """
    from cgx.session.tasks.scaffold import _synthesize_import_smoke_test
    diffs = [{"file": "backend/app.py", "patch": "x"}]
    generated = [{"file": "backend/app.py"}]
    existing = [
        {"path": "backend/__init__.py", "content": ""},
        {"path": "backend/app.py",
         "content": "def compute(a, b):\n    return a + b\n"},
    ]
    layers = [{"name": "app", "files": [
        {"path": "backend/__init__.py"},
        {"path": "backend/app.py"},
        {"path": "tests/test_app.py"}]}]
    added = _synthesize_import_smoke_test(
        diffs=diffs, generated=generated,
        existing_with_content=existing, layers=layers)
    assert added == "tests/test_smoke.py"
    smoke = next(e for e in existing if e["path"] == "tests/test_smoke.py")
    assert "backend.app" in smoke["content"]
    assert "importlib.import_module" in smoke["content"]
    # backend/__init__.py is a package marker, not a probe target.
    assert "'backend'," not in smoke["content"]
    assert "tests/test_smoke.py" in [d["file"] for d in diffs]


def test_synthesize_import_smoke_test_skipped_when_a_test_survives():
    """A model-authored test that survived the gates is never overridden."""
    from cgx.session.tasks.scaffold import _synthesize_import_smoke_test
    diffs: list = []
    generated: list = []
    existing = [
        {"path": "backend/app.py", "content": "x = 1\n"},
        {"path": "tests/test_app.py",
         "content": "def test_x():\n    assert True\n"},
    ]
    layers = [{"name": "app", "files": [
        {"path": "backend/app.py"}, {"path": "tests/test_app.py"}]}]
    assert _synthesize_import_smoke_test(
        diffs=diffs, generated=generated,
        existing_with_content=existing, layers=layers) is None
    assert diffs == []


def test_synthesize_import_smoke_test_skipped_when_no_test_was_planned():
    """A scaffold that never planned a test is left alone (no smoke test)."""
    from cgx.session.tasks.scaffold import _synthesize_import_smoke_test
    diffs = [{"file": "backend/app.py", "patch": "x"}]
    generated = [{"file": "backend/app.py"}]
    existing = [{"path": "backend/app.py", "content": "x = 1\n"}]
    layers = [{"name": "app", "files": [{"path": "backend/app.py"}]}]
    assert _synthesize_import_smoke_test(
        diffs=diffs, generated=generated,
        existing_with_content=existing, layers=layers) is None
    assert [d["file"] for d in diffs] == ["backend/app.py"]


def test_scaffold_synthesizes_smoke_test_when_planned_test_is_dropped(
        store, monkeypatch):
    """End-to-end: the planned test is unrecoverable, a smoke test replaces it.

    The manifest plans ``backend/app.py`` + ``tests/test_app.py``; the
    generated test imports an undefined dotted module so the phantom gate
    drops it, leaving a source module and no test. SCAFFOLD then synthesizes
    ``tests/test_smoke.py`` importing ``backend.app`` so VERIFY runs a real
    suite instead of reporting no_tests.
    """
    from cgx.session.tasks.scaffold import run_scaffold
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    plan = Artifact.new(
        session.session_id, "task_x", ArtifactKind.WORK_PLAN, {
            "plan_md": "", "composed_goal": "g", "prior_goal": "g",
            "layers": [{"name": "app", "files": [
                {"path": "backend/__init__.py", "description": ""},
                {"path": "backend/app.py", "description": ""},
                {"path": "tests/test_app.py", "description": ""}]}],
        })
    store.save_artifact(plan)

    bodies = {
        "backend/__init__.py": "",
        "backend/app.py": "def compute(a, b):\n    return a + b\n",
        # Imports a dotted first-party module that is never defined -> the
        # phantom gate drops this test (an unrecoverable model slip).
        "tests/test_app.py": "from backend.missing import gone\n",
    }

    def fake_generate(path, *a, **kw):
        return {"file": path, "patch": f"+++ {path}\nx",
                "content": bodies[path], "syntax_ok": True}

    monkeypatch.setattr("cgx.answer.engine.generate_single_scaffold_file",
                        fake_generate)
    t = TaskNode.new(session.session_id, TaskKind.SCAFFOLD, "s",
                     inputs={"work_plan_artifact_id": plan.artifact_id})
    result = run_scaffold(
        t, ExecutorDeps(provider=_StubProvider(""), store=store))
    assert result.failure is None
    assert result.outputs["smoke_test_synthesized"] == "tests/test_smoke.py"
    files = [d["file"] for d in result.artifact.content["diffs"]]
    assert "tests/test_app.py" not in files
    assert "tests/test_smoke.py" in files
    smoke = next(d for d in result.artifact.content["diffs"]
                 if d["file"] == "tests/test_smoke.py")
    assert "backend.app" in smoke["patch"]


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
                      contracts=None, skills=None, **kwargs):
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


def _carry_forward_plan(session):
    """A work plan whose manifest includes requirements.txt + one source."""
    return Artifact.new(
        session.session_id, "task_x", ArtifactKind.WORK_PLAN, {
            "prior_goal": "g", "composed_goal": "build a flask api",
            "answers": {}, "plan_md": "",
            "layers": [{"name": "app", "files": [
                {"path": "requirements.txt", "description": "deps"},
                {"path": "backend/app.py", "description": "flask app"}]}],
        })


def test_scaffold_carries_forward_locked_requirements_on_regenerate(
        tmp_path, store, monkeypatch):
    """A repair-locked requirements.txt survives a whole-tree regenerate.

    env_manager re-pinned requirements.txt to a conflict-free set and
    marked it locked; the regenerate must reuse that on-disk file verbatim
    (never re-emit the model's stale manifest) so the dependency fix holds.
    """
    from cgx.answer.engine import _content_to_new_file_patch
    from cgx.codegen.env_manager import mark_requirements_locked
    from cgx.session.tasks.scaffold import run_scaffold
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    plan = _carry_forward_plan(session)
    store.save_artifact(plan)
    # The resolved, conflict-free requirements.txt already on disk, marked
    # env-locked by the repair that re-pinned it.
    resolved = "flask==3.1.3\nwerkzeug==3.1.8\n"
    (tmp_path / "requirements.txt").write_text(resolved, encoding="utf-8")
    mark_requirements_locked(str(tmp_path))

    seen: list = []

    def fake_generate(path, *a, **kw):
        seen.append(path)
        body = "from flask import Flask\napp = Flask(__name__)\n"
        return {"file": path,
                "patch": _content_to_new_file_patch(path, body),
                "content": body, "syntax_ok": True, "confidence": 0.9}

    monkeypatch.setattr("cgx.answer.engine.generate_single_scaffold_file",
                        fake_generate)
    t = TaskNode.new(
        session.session_id, TaskKind.SCAFFOLD, "s",
        inputs={"work_plan_artifact_id": plan.artifact_id,
                "regenerated_from_task_id": "task_sc0"})
    result = run_scaffold(t, ExecutorDeps(
        provider=_StubProvider(""), store=store, project_root=str(tmp_path)))
    assert result.failure is None
    # requirements.txt was carried forward -- never handed to the model.
    assert "requirements.txt" in {d["file"]
                                  for d in result.artifact.content["diffs"]}
    assert seen == ["backend/app.py"]
    req_diff = next(d for d in result.artifact.content["diffs"]
                    if d["file"] == "requirements.txt")
    assert "flask==3.1.3" in req_diff["patch"]
    gen = {g["file"]: g for g in result.artifact.content["generated"]}
    assert gen["requirements.txt"].get("carried") is True


def test_scaffold_regenerates_requirements_without_lock_marker(
        tmp_path, store, monkeypatch):
    """No lock marker -> requirements.txt is regenerated normally (no carry)."""
    from cgx.answer.engine import _content_to_new_file_patch
    from cgx.session.tasks.scaffold import run_scaffold
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    plan = _carry_forward_plan(session)
    store.save_artifact(plan)
    # An on-disk requirements.txt but NO lock marker -> not env-managed.
    (tmp_path / "requirements.txt").write_text(
        "flask==2.0.1\n", encoding="utf-8")

    seen: list = []

    def fake_generate(path, *a, **kw):
        seen.append(path)
        return {"file": path,
                "patch": _content_to_new_file_patch(path, "x\n"),
                "content": "x\n", "syntax_ok": True, "confidence": 0.9}

    monkeypatch.setattr("cgx.answer.engine.generate_single_scaffold_file",
                        fake_generate)
    t = TaskNode.new(
        session.session_id, TaskKind.SCAFFOLD, "s",
        inputs={"work_plan_artifact_id": plan.artifact_id,
                "regenerated_from_task_id": "task_sc0"})
    result = run_scaffold(t, ExecutorDeps(
        provider=_StubProvider(""), store=store, project_root=str(tmp_path)))
    assert result.failure is None
    # Without the marker both manifest files are regenerated from the model.
    assert set(seen) == {"requirements.txt", "backend/app.py"}


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
                      on_token=None, depends_on=None, contracts=None,
                      skills=None, **kwargs):
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
    assert result.artifact is not None
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


def test_bootstrap_env_polyglot_provisions_python_and_node(
        tmp_path, store, monkeypatch):
    """A repo with both requirements.txt and package.json provisions BOTH.

    Part 5: previously ``python`` took priority and ``node_modules`` was
    left to VERIFY's best-effort install. Now BOOTSTRAP_ENV provisions the
    venv *and* runs ``npm install`` in the same pass, folding a ``node``
    sub-report into the BUILD_REPORT while keeping ``project_type=python``
    so the Python-only gates still key off it.
    """
    from cgx.session.tasks import bootstrap_env as bs
    (tmp_path / "requirements.txt").write_text("flask\n", encoding="utf-8")
    (tmp_path / "package.json").write_text(
        '{"name": "app", "scripts": {"build": "vite build"}}',
        encoding="utf-8")
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

    npm_calls: dict = {}

    def _fake_run(cmd, **kw):
        npm_calls["cmd"] = list(cmd)
        (tmp_path / "node_modules").mkdir()

        class _P:
            returncode = 0
            stdout = "added 1 package"
            stderr = ""

        return _P()

    monkeypatch.setattr(bs.shutil, "which", lambda name: "/usr/bin/npm")
    monkeypatch.setattr(bs.subprocess, "run", _fake_run)

    t = TaskNode.new(session.session_id, TaskKind.BOOTSTRAP_ENV, "boot",
                     inputs={"applied_files": ["app.py"],
                             "mode": SessionMode.GREENFIELD.value})
    result = bs.run_bootstrap_env(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.failure is None
    # Python remains the primary type/outcome so the downstream gates fire.
    assert result.outputs["project_type"] == "python"
    assert result.outputs["outcome"] == "succeeded"
    assert result.outputs["venv_path"] == str(tmp_path / ".venv")
    # Node was provisioned in the same pass.
    assert result.outputs["node_outcome"] == "succeeded"
    assert npm_calls["cmd"][:2] == ["npm", "install"]
    assert (tmp_path / "node_modules").is_dir()
    node = result.artifact.content["node"]
    assert node["outcome"] == "succeeded"
    assert node["note"] is None


def test_bootstrap_env_polyglot_node_failure_is_nonfatal(
        tmp_path, store, monkeypatch):
    """Node provisioning failure must not fail the Python bootstrap."""
    from cgx.session.tasks import bootstrap_env as bs
    (tmp_path / "requirements.txt").write_text("flask\n", encoding="utf-8")
    (tmp_path / "package.json").write_text('{"name": "app"}', encoding="utf-8")
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
    # npm absent -> node provisioning degrades to skipped, non-fatal.
    monkeypatch.setattr(bs.shutil, "which", lambda name: None)

    t = TaskNode.new(session.session_id, TaskKind.BOOTSTRAP_ENV, "boot",
                     inputs={"applied_files": ["app.py"],
                             "mode": SessionMode.GREENFIELD.value})
    result = bs.run_bootstrap_env(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.failure is None
    assert result.outputs["outcome"] == "succeeded"
    assert result.outputs["node_outcome"] == "skipped"
    assert result.artifact.content["node"]["note"] == "npm not installed"


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
    """A failed install of a *declared* dependency -> outcome=failed."""
    from cgx.session.tasks.bootstrap_env import run_bootstrap_env
    (tmp_path / "requirements.txt").write_text(
        "flask\nnonexistent-xyz\n", encoding="utf-8")
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
    assert "declared" in result.failure
    assert result.artifact.content["failed_installs"] == ["nonexistent-xyz"]
    assert result.artifact.content["uninstallable"] == []


def test_bootstrap_env_undeclared_install_failure_is_non_fatal(
        tmp_path, store, monkeypatch):
    """An undeclared scan-install failure is recorded, never terminal.

    Mirrors the live failure: the scaffold hallucinated ``from core
    import compute``, preflight scanned the import and pip-installing
    ``core`` failed, ending the session in a terminal bootstrap failure.
    An import discovered only by code scanning is a code problem, not an
    environment problem: record it as ``uninstallable`` and proceed so
    API_CHECK diagnoses it honestly (and routes it to a regenerate).
    """
    from cgx.session.tasks.bootstrap_env import run_bootstrap_env
    (tmp_path / "requirements.txt").write_text("flask\n", encoding="utf-8")
    (tmp_path / "app.py").write_text(
        "import flask\nfrom core import compute\n", encoding="utf-8")
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
        lambda files, root, python=None: (["core"], {"core": False}))
    monkeypatch.setattr(
        "cgx.codegen.env_manager.update_requirements",
        lambda root, pkgs: None)
    monkeypatch.setattr(
        "cgx.session.tasks.bootstrap_env._capture_pip_freeze",
        lambda exe, timeout=30.0: ([], ""))
    # flask resolves in the venv; ``core`` must be excluded from the
    # honesty gate because its install failure is already recorded.
    monkeypatch.setattr(
        "cgx.codegen.env_manager._probe_importable",
        lambda names, python=None: {"flask"})

    t = TaskNode.new(session.session_id, TaskKind.BOOTSTRAP_ENV, "boot",
                     inputs={"applied_files": ["app.py"],
                             "mode": SessionMode.GREENFIELD.value})
    result = run_bootstrap_env(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.failure is None
    assert result.outputs["outcome"] == "succeeded"
    assert result.outputs["failed_count"] == 0
    assert result.outputs["uninstallable_count"] == 1
    content = result.artifact.content
    assert content["uninstallable"] == ["core"]
    assert content["failed_installs"] == ["core"]
    assert content["missing_imports"] == []


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


def test_bootstrap_env_installs_requested_missing_modules(
        tmp_path, store, monkeypatch):
    """install_deps repair: ``missing_modules`` in inputs are installed
    even when the preflight file scan finds nothing to do."""
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
    seen: dict = {}

    def fake_find_missing(imports, root, python=None):
        seen["imports"] = sorted(imports)
        return ["uvicorn"]

    def fake_install(pkgs, python=None):
        seen["installed"] = list(pkgs)
        return {p: True for p in pkgs}

    monkeypatch.setattr(
        "cgx.codegen.env_manager.find_missing_python_packages",
        fake_find_missing)
    monkeypatch.setattr(
        "cgx.codegen.env_manager.install_packages", fake_install)
    captured: dict = {}
    monkeypatch.setattr(
        "cgx.codegen.env_manager.update_requirements",
        lambda root, pkgs: captured.setdefault("pkgs", list(pkgs)))
    monkeypatch.setattr(
        "cgx.session.tasks.bootstrap_env._capture_pip_freeze",
        lambda exe, timeout=30.0: ([], ""))

    t = TaskNode.new(session.session_id, TaskKind.BOOTSTRAP_ENV, "boot",
                     inputs={"applied_files": ["backend/app.py"],
                             "missing_modules": ["uvicorn"],
                             "mode": SessionMode.GREENFIELD.value})
    result = run_bootstrap_env(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.failure is None
    assert result.outputs["outcome"] == "succeeded"
    assert seen["imports"] == ["uvicorn"]
    assert seen["installed"] == ["uvicorn"]
    assert result.outputs["installed_count"] == 1
    assert result.artifact.content["installed_packages"] == ["uvicorn"]
    assert captured["pkgs"] == ["uvicorn"]


def test_bootstrap_env_installs_testclient_extra(
        tmp_path, store, monkeypatch):
    """TestClient usage in applied tests -> httpx installed up front.

    fastapi/starlette's TestClient needs httpx at import time but no
    first-party file imports it directly, so the file-scan preflight
    never installs it and VERIFY would die at collection.
    """
    from cgx.session.tasks.bootstrap_env import run_bootstrap_env
    (tmp_path / "requirements.txt").write_text("fastapi\n", encoding="utf-8")
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    venv_python = tmp_path / ".venv" / "bin" / "python"
    venv_python.write_text("")
    venv_python.chmod(0o755)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_api.py").write_text(
        "from fastapi.testclient import TestClient\n"
        "from backend.app import app\n\n"
        "client = TestClient(app)\n\n\n"
        "def test_ping():\n    assert client is not None\n",
        encoding="utf-8")

    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    monkeypatch.setattr("cgx.codegen.test_runner.ensure_project_venv",
                        lambda root, timeout=300.0: str(venv_python))
    monkeypatch.setattr("cgx.codegen.env_manager.preflight_install",
                        lambda files, root, python=None: ([], {}))
    seen: dict = {}

    def fake_find_missing(imports, root, python=None):
        seen["imports"] = sorted(imports)
        return sorted(imports)

    def fake_install(pkgs, python=None):
        seen["installed"] = list(pkgs)
        return {p: True for p in pkgs}

    monkeypatch.setattr(
        "cgx.codegen.env_manager.find_missing_python_packages",
        fake_find_missing)
    monkeypatch.setattr(
        "cgx.codegen.env_manager.install_packages", fake_install)
    captured: dict = {}
    monkeypatch.setattr(
        "cgx.codegen.env_manager.update_requirements",
        lambda root, pkgs: captured.setdefault("pkgs", list(pkgs)))
    monkeypatch.setattr(
        "cgx.session.tasks.bootstrap_env._capture_pip_freeze",
        lambda exe, timeout=30.0: ([], ""))
    monkeypatch.setattr(
        "cgx.session.tasks.bootstrap_env._verify_runtime_imports",
        lambda root, files, python_exe, skip_roots=frozenset(): [])

    t = TaskNode.new(session.session_id, TaskKind.BOOTSTRAP_ENV, "boot",
                     inputs={"applied_files": ["tests/test_api.py"],
                             "mode": SessionMode.GREENFIELD.value})
    result = run_bootstrap_env(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.failure is None
    assert result.outputs["outcome"] == "succeeded"
    assert seen["imports"] == ["httpx"]
    assert seen["installed"] == ["httpx"]
    assert result.artifact.content["installed_packages"] == ["httpx"]
    assert captured["pkgs"] == ["httpx"]


def test_testclient_extra_roots_detection(tmp_path):
    """Only fastapi/starlette testclient files trigger the httpx extra."""
    from cgx.session.tasks.bootstrap_env import _testclient_extra_roots
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_api.py").write_text(
        "from starlette.testclient import TestClient\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("import flask\n", encoding="utf-8")
    assert _testclient_extra_roots(tmp_path, ["app.py"]) == []
    assert _testclient_extra_roots(
        tmp_path, ["app.py", "tests/test_api.py"]) == ["httpx"]
    # Unreadable / missing files are skipped, never crash.
    assert _testclient_extra_roots(tmp_path, ["nope.py"]) == []


def test_bootstrap_env_requested_modules_already_satisfied(
        tmp_path, store, monkeypatch):
    """missing_modules that already import in the venv trigger no install."""
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
    monkeypatch.setattr(
        "cgx.codegen.env_manager.find_missing_python_packages",
        lambda imports, root, python=None: [])

    def boom(pkgs, python=None):
        raise AssertionError("install_packages must not run when the "
                             "requested modules already resolve")

    monkeypatch.setattr("cgx.codegen.env_manager.install_packages", boom)
    monkeypatch.setattr("cgx.codegen.env_manager.update_requirements",
                        lambda root, pkgs: None)
    monkeypatch.setattr(
        "cgx.session.tasks.bootstrap_env._capture_pip_freeze",
        lambda exe, timeout=30.0: ([], ""))

    t = TaskNode.new(session.session_id, TaskKind.BOOTSTRAP_ENV, "boot",
                     inputs={"applied_files": ["app.py"],
                             "missing_modules": ["flask"],
                             "mode": SessionMode.GREENFIELD.value})
    result = run_bootstrap_env(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.failure is None
    assert result.outputs["outcome"] == "succeeded"
    assert result.outputs["installed_count"] == 0


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
    assert result.artifact is not None
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

    def fake_probe(python_exe, pkg, timeout, root):
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


def test_smoke_node_build_error_head_survives_truncation(
        tmp_path, store, monkeypatch):
    """Vite/rolldown print the cause at the HEAD then a long generic stack.

    A tail-only clip drops the actionable line; the head+tail window must
    keep it so REPAIR's ``build_error`` constraint stays actionable.
    """
    from cgx.session.tasks import smoke as smoke_mod
    t = _node_smoke_task(store, tmp_path)
    head = "[UNRESOLVED_ENTRY] Cannot resolve entry module index.html."
    tail = "at CAC.<anonymous> (vite/dist/node/cli.js:776:3)"
    long_stack = "\n".join(f"    at frame_{i} (rolldown.mjs:{i}:1)"
                           for i in range(400))

    class _P:
        returncode = 1
        stdout = ""
        stderr = head + "\n" + long_stack + "\n" + tail

    monkeypatch.setattr(smoke_mod.shutil, "which", lambda name: "/usr/bin/npm")
    monkeypatch.setattr(smoke_mod.subprocess, "run", lambda *a, **k: _P())
    result = smoke_mod.run_smoke(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    bs = result.artifact.content["build_smoke"]
    assert bs["ok"] is False
    assert head in bs["stderr_tail"], "actionable cause (head) must survive"
    assert tail in bs["stderr_tail"], "trailing summary (tail) must survive"
    assert "...[truncated]..." in bs["stderr_tail"]


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
    assert result.artifact is not None
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

    def fake_probe(python_exe, specs, timeout, root):
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

    def fake_probe(python_exe, specs, timeout, root):
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


def test_api_check_splits_dependency_conflict_from_missing(
        tmp_path, store, monkeypatch):
    """``cannot import name ... from '<peer>'`` -> dependency_conflict.

    The package is installed but its own import chain broke on an
    incompatible peer major (the Flask 2.0.1 / Werkzeug 3 url_quote
    break). Distinct from an absent package; surfaced separately so the
    router re-resolves the env instead of regenerating valid code.
    """
    from cgx.session.tasks import api_check as api_mod
    from cgx.session.tasks.api_check import run_api_check
    (tmp_path / "app.py").write_text(
        "from flask import Flask\n", encoding="utf-8")
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    build_art = Artifact.new(
        session_id=session.session_id, produced_by_task_id="t_build",
        kind=ArtifactKind.BUILD_REPORT,
        content={"python_exe": "/fake/.venv/bin/python"})
    store.save_artifact(build_art)

    def fake_probe(python_exe, specs, timeout, root):
        rows = []
        for module, name in specs:
            if module == "flask":
                rows.append({
                    "module": module, "name": name, "ok": False,
                    "error": ("ImportError: cannot import name 'url_quote' "
                              "from 'werkzeug.urls'")})
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
    assert result.outputs["missing_module_count"] == 0
    assert result.outputs["conflict_count"] == 1
    # Consumer (flask) and the incompatible peer (werkzeug) are both
    # surfaced so the resolver can move the whole set to a consistent one.
    assert result.outputs["conflict_packages"] == ["flask", "werkzeug"]
    content = result.artifact.content
    assert content["conflict_packages"] == ["flask", "werkzeug"]
    assert content["missing_modules"] == []
    cats = {(r["module"], r["name"]): r.get("category")
            for r in content["references"] if not r["ok"]}
    assert cats[("flask", "Flask")] == "dependency_conflict"


def test_api_check_wrong_path_first_party_is_not_missing_dependency(
        tmp_path, store, monkeypatch):
    """A first-party module reached by the wrong import path is a
    hallucination (regenerate), never a ``missing_dependency`` install."""
    from cgx.session.tasks import api_check as api_mod
    from cgx.session.tasks.api_check import run_api_check
    (tmp_path / "backend").mkdir()
    (tmp_path / "backend" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "backend" / "app.py").write_text(
        "app = object()\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    # The test imports the app by the wrong path (bare ``app``) and also
    # references a genuinely-absent third-party package (``cerberus``).
    (tmp_path / "tests" / "test_app.py").write_text(
        "from app import app\n"
        "from cerberus import Schema\n",
        encoding="utf-8")
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    build_art = Artifact.new(
        session_id=session.session_id, produced_by_task_id="t_build",
        kind=ArtifactKind.BUILD_REPORT,
        content={"python_exe": "/fake/.venv/bin/python"})
    store.save_artifact(build_art)

    def fake_probe(python_exe, specs, timeout, root):
        rows = []
        for module, name in specs:
            if module == "app":
                rows.append({
                    "module": module, "name": name, "ok": False,
                    "error": "ModuleNotFoundError: No module named 'app'"})
            elif module == "cerberus":
                rows.append({
                    "module": module, "name": name, "ok": False,
                    "error": ("ModuleNotFoundError: No module named "
                              "'cerberus'")})
            else:
                rows.append({"module": module, "name": name, "ok": True,
                             "error": ""})
        return rows, None

    monkeypatch.setattr(api_mod, "_probe_references", fake_probe)
    t = TaskNode.new(
        session.session_id, TaskKind.API_CHECK, "api",
        inputs={"applied_files": ["backend/__init__.py", "backend/app.py",
                                  "tests/test_app.py"],
                "build_artifact_id": build_art.artifact_id})
    result = run_api_check(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.outputs["outcome"] == "failed"
    content = result.artifact.content
    # ``app`` exists on disk (backend/app.py) -> wrong-path hallucination,
    # not a package to install. Only ``cerberus`` is a genuine missing dep.
    assert content["missing_modules"] == ["cerberus"]
    cats = {(r["module"], r["name"]): r.get("category")
            for r in content["references"] if not r["ok"]}
    assert cats[("app", "app")] == "api_check_failure"
    assert cats[("cerberus", "Schema")] == "missing_dependency"


def test_api_check_uninstallable_root_is_hallucination(
        tmp_path, store, monkeypatch):
    """A root BOOTSTRAP_ENV already failed to pip-install is never a
    ``missing_dependency`` -- pip has proven it cannot satisfy the name,
    so an ``install_deps`` repair would loop. It routes to regenerate."""
    from cgx.session.tasks import api_check as api_mod
    from cgx.session.tasks.api_check import run_api_check
    (tmp_path / "app.py").write_text(
        "from core import compute\n", encoding="utf-8")
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    build_art = Artifact.new(
        session_id=session.session_id, produced_by_task_id="t_build",
        kind=ArtifactKind.BUILD_REPORT,
        content={"python_exe": "/fake/.venv/bin/python",
                 "uninstallable": ["core"]})
    store.save_artifact(build_art)

    def fake_probe(python_exe, specs, timeout, root):
        rows = [{"module": module, "name": name, "ok": False,
                 "error": "ModuleNotFoundError: No module named 'core'"}
                for module, name in specs]
        return rows, None

    monkeypatch.setattr(api_mod, "_probe_references", fake_probe)
    t = TaskNode.new(session.session_id, TaskKind.API_CHECK, "api",
                     inputs={"applied_files": ["app.py"],
                             "build_artifact_id": build_art.artifact_id})
    result = run_api_check(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.outputs["outcome"] == "failed"
    assert result.outputs["missing_module_count"] == 0
    assert result.outputs["hallucinated_count"] == 1
    content = result.artifact.content
    assert content["missing_modules"] == []
    assert content["hallucinated_references"] == [
        {"module": "core", "name": "compute"}]


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


def test_api_check_probe_isolated_from_server_cwd(tmp_path, monkeypatch):
    """The reference probe must not resolve modules from the server's cwd.

    Regression for the live failure where a probe launched from the CGX
    workspace resolved the launcher's root ``app.py`` instead of failing
    honestly with ``No module named 'app'``.
    """
    import sys
    from cgx.session.tasks.api_check import _probe_references
    shadow = tmp_path / "server_cwd"
    shadow.mkdir()
    (shadow / "app.py").write_text("app = object()\n", encoding="utf-8")
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(shadow)
    rows, probe_error = _probe_references(
        sys.executable, [("app", "app"), ("json", "loads")], 30.0, project)
    assert probe_error is None
    by_mod = {r["module"]: r for r in rows}
    assert by_mod["app"]["ok"] is False
    assert "No module named 'app'" in by_mod["app"]["error"]
    assert by_mod["json"]["ok"] is True


def test_smoke_probe_isolated_from_server_cwd(tmp_path, monkeypatch):
    """The import probe must not resolve modules from the server's cwd."""
    import sys
    from cgx.session.tasks.smoke import _probe_import
    shadow = tmp_path / "server_cwd"
    shadow.mkdir()
    (shadow / "fakedep.py").write_text("value = 1\n", encoding="utf-8")
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(shadow)
    ok, tail = _probe_import(sys.executable, "fakedep", 30.0, project)
    assert ok is False
    assert "No module named 'fakedep'" in tail
    ok, _ = _probe_import(sys.executable, "json", 30.0, project)
    assert ok is True


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


def test_repair_handles_runtime_report(tmp_path, store):
    """#3: REPAIR(runtime_artifact_id) -> regenerate with the boot error.

    A RUNTIME_REPORT boot failure has no test to patch, so REPAIR emits a
    non-applicable plan whose ``strategy='regenerate'`` and whose
    constraint carries the captured import/create_app traceback for the
    scaffold prompt to act on.
    """
    from cgx.session.tasks.repair import run_repair
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    report = Artifact.new(
        session_id=session.session_id, produced_by_task_id="t_runtime",
        kind=ArtifactKind.RUNTIME_REPORT,
        content={
            "outcome": "failed",
            "failed_entries": ["backend/app.py"],
            "failure_signature": "runtime_boot|backend/app.py",
            "probes": [{
                "file": "backend/app.py", "ok": False,
                "kind": "import_error",
                "stderr_tail": "NameError: name 'db' is not defined",
            }],
        })
    store.save_artifact(report)
    t = TaskNode.new(
        session.session_id, TaskKind.REPAIR, "repair",
        inputs={"runtime_artifact_id": report.artifact_id,
                "repair_attempt": 1,
                "mode": SessionMode.GREENFIELD.value})
    result = run_repair(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.failure is None
    assert result.outputs["can_apply"] is False
    assert result.outputs["classification"] == "runtime_failure"
    assert result.outputs["strategy"] == "regenerate"
    assert result.outputs["failure_signature"] == "runtime_boot|backend/app.py"
    plan = result.artifact
    assert plan.kind is ArtifactKind.REPAIR_PLAN
    assert plan.content["diffs"] == []
    assert plan.content["failed_entries"] == ["backend/app.py"]
    constraints = plan.content["extra_constraints"]
    assert constraints["kind"] == "runtime_failure"
    assert "NameError: name 'db' is not defined" in constraints["runtime_error"]
    assert "backend/app.py" in plan.content["rationale"]
    assert "ASK_USER" not in plan.content["rationale"]


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


def test_repair_api_check_dependency_conflict_resolves(tmp_path, store):
    """API_CHECK conflict_packages -> strategy=resolve_deps, not regenerate."""
    from cgx.session.tasks.repair import run_repair
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    report = Artifact.new(
        session_id=session.session_id, produced_by_task_id="t_api",
        kind=ArtifactKind.API_CHECK_REPORT,
        content={
            "outcome": "failed",
            "failed_references": [{"module": "flask", "name": "Flask"}],
            "missing_modules": [],
            "conflict_packages": ["flask", "werkzeug"],
            "conflict_references": [
                {"module": "flask", "name": "Flask",
                 "error": ("ImportError: cannot import name 'url_quote' "
                           "from 'werkzeug.urls'")}],
            "hallucinated_references": [],
            "failure_signature": "api_check|conflict:flask,werkzeug",
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
    assert result.outputs["classification"] == "dependency_conflict"
    assert result.outputs["strategy"] == "resolve_deps"
    assert result.outputs["can_apply"] is False
    assert result.outputs["conflict_packages"] == ["flask", "werkzeug"]
    plan = result.artifact
    assert plan.kind is ArtifactKind.REPAIR_PLAN
    assert plan.content["strategy"] == "resolve_deps"
    assert plan.content["conflict_packages"] == ["flask", "werkzeug"]
    assert "werkzeug" in plan.content["rationale"]
    assert plan.content["diffs"] == []


def test_repair_api_check_unresolved_module_regenerates(tmp_path, store):
    """An unresolved (wrong-path) module -> regenerate with path-fix guidance,
    not the 'remove the symbol' rationale used for absent attributes."""
    from cgx.session.tasks.repair import run_repair
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    report = Artifact.new(
        session_id=session.session_id, produced_by_task_id="t_api",
        kind=ArtifactKind.API_CHECK_REPORT,
        content={
            "outcome": "failed",
            "failed_references": [
                {"module": "app", "name": "app",
                 "error": "ModuleNotFoundError: No module named 'app'",
                 "category": "api_check_failure"},
            ],
            "missing_modules": [],
            "hallucinated_references": [{"module": "app", "name": "app"}],
            "failure_signature": "api_check|app.app",
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
    assert result.outputs["classification"] == "api_check_failure"
    assert result.outputs["strategy"] == "regenerate"
    assert result.outputs["can_apply"] is False
    rationale = result.artifact.content["rationale"]
    assert "could NOT be imported" in rationale
    assert "correct in-project path" in rationale
    assert "ASK_USER" not in rationale


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


def test_classify_assertion_drift_for_plain_assertion_failure():
    """A plain assert failure with no locator -> assertion_drift."""
    from cgx.session.repair.classify import classify_verify_report
    content = {
        "outcome": "assertions_failed",
        "returncode": 1,
        "stdout": "E   assert 1 == 2\nE    +  where 1 = compute()",
        "stderr": "",
    }
    assert classify_verify_report(content) == "assertion_drift"


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


def test_classify_undefined_name_from_collection_stderr():
    """A collection-time NameError is repairable, not `unknown`.

    Live failure: a conftest import chain died on ``enum.Enum`` with no
    ``import enum``. Pytest writes that to stderr and produces no junit
    XML, so the report carries no structured ``failures``.
    """
    from cgx.session.repair.classify import (
        classify_verify_report, undefined_names)
    content = {
        "outcome": "collection_error",
        "returncode": 4,
        "stdout": "",
        "stderr": (
            "ImportError while loading conftest 'tests/conftest.py'.\n"
            "tests/conftest.py:3: in <module>\n"
            "    from backend.main import app\n"
            "backend/main.py:13: in <module>\n"
            "    class Operation(str, enum.Enum):\n"
            "E   NameError: name 'enum' is not defined. Did you forget "
            "to import 'enum'"),
        "failures": [],
    }
    assert classify_verify_report(content) == "undefined_name"
    assert undefined_names(content) == ("enum",)


def test_classify_prefers_mechanical_class_over_undefined_name():
    """A run surfacing both keeps the classification with a locator."""
    from cgx.session.repair.classify import classify_verify_report
    content = {
        "outcome": "collection_error",
        "returncode": 2,
        "stdout": "ModuleNotFoundError: No module named 'backend'",
        "stderr": "NameError: name 'enum' is not defined",
    }
    assert classify_verify_report(content) == "missing_module_pythonpath"


def test_undefined_name_routes_to_regenerate():
    """No mechanical patch exists, so REPAIR must re-author the module."""
    from cgx.session.tasks.repair import _REGENERATE_CLASSES
    assert "undefined_name" in _REGENERATE_CLASSES


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


def test_traceback_source_files_extracts_both_frame_shapes():
    """pytest (``file.py:12``) + captured (``File "f.py"``) frames merge."""
    from cgx.session.repair.classify import traceback_source_files
    content = {
        "outcome": "assertions_failed",
        "returncode": 1,
        "stdout": (
            "tests/test_util.py:5: in test_scale\n"
            "    assert scale(2) == 4\n"
            "src/util.py:2: in scale\n"
            "    return x - 1\n"
            "E   assert 1 == 4"),
        "stderr": (
            'Traceback (most recent call last):\n'
            '  File "src/util.py", line 2, in scale\n'
            '    return x - 1\n'),
    }
    files = traceback_source_files(content)
    # order-preserving + de-duplicated across both regexes/streams
    assert files == ("src/util.py", "tests/test_util.py")


def test_traceback_source_files_empty_without_frames():
    """No frame shapes in the blob -> empty tuple (regenerate fallback)."""
    from cgx.session.repair.classify import traceback_source_files
    assert traceback_source_files(
        {"stdout": "E   assert 1 == 3", "stderr": ""}) == ()


# --------------------- FailureContext normalization (D1) -------------------

def test_failure_context_verify_normalizes_and_reuses_signature():
    """A VERIFY_REPORT folds into a FailureContext, reusing its cached
    ``failure_signature`` and localizing the traceback files."""
    from cgx.session.repair.context import FailureContext
    content = {
        "outcome": "assertions_failed",
        "returncode": 1,
        "stdout": (
            "tests/test_util.py:5: in test_scale\n"
            "    assert scale(2) == 4\n"
            "src/util.py:3: in scale\n"
            "    return x * 3\n"
            "E   AssertionError: assert 6 == 4\n"
        ),
        "stderr": "",
        "failure_signature": "cached-verify-sig",
    }
    fc = FailureContext.from_report(
        "verify", content, goal="a calculator",
        manifest_files=["src/util.py"], installed_packages=["pytest"])
    assert fc.gate == "verify"
    assert fc.classification == "assertion_drift"
    # The precomputed signature is trusted, not recomputed.
    assert fc.failure_signature == "cached-verify-sig"
    assert "AssertionError" in fc.failure_text
    assert fc.traceback_files == ("tests/test_util.py", "src/util.py")
    assert fc.installed_packages == ("pytest",)
    assert fc.manifest_files == ("src/util.py",)
    assert fc.goal == "a calculator"


def test_failure_context_computes_signature_when_absent():
    """No cached signature -> FailureContext recomputes it via classify."""
    from cgx.session.repair.classify import failure_signature
    from cgx.session.repair.context import FailureContext
    content = {"outcome": "assertions_failed", "returncode": 1,
               "stdout": "E   AssertionError: 1 == 2"}
    fc = FailureContext.from_report("verify", content)
    assert fc.failure_signature == failure_signature(content)


def test_failure_context_runtime_uses_runtime_text_and_frames():
    """A RUNTIME_REPORT folds its failing probe stderr tails into one blob
    and localizes the boot file from the captured traceback frame."""
    from cgx.session.repair.classify import classify_runtime_report
    from cgx.session.repair.context import FailureContext
    content = {
        "outcome": "failed",
        "probes": [
            {"file": "backend/app.py", "kind": "import_error", "ok": False,
             "stderr_tail": (
                 "Traceback (most recent call last):\n"
                 '  File "backend/app.py", line 10, in <module>\n'
                 "    db.init_app(app)\n"
                 "NameError: name 'db' is not defined")},
            {"file": "backend/ok.py", "kind": "import", "ok": True,
             "stderr_tail": ""},
        ],
        "failed_entries": ["backend/app.py"],
        "failure_signature": "runtime_boot|backend/app.py",
    }
    fc = FailureContext.from_report("runtime", content)
    assert fc.gate == "runtime"
    assert fc.classification == classify_runtime_report(content)
    assert "NameError: name 'db' is not defined" in fc.failure_text
    assert "backend/ok.py" not in fc.failure_text
    assert fc.traceback_files == ("backend/app.py",)


def test_failure_context_smoke_concatenates_failing_modules():
    """SMOKE has no classifier: it maps to ``unknown`` and its blob is the
    failing modules' stderr tails plus a failing build smoke."""
    from cgx.session.repair.context import FailureContext
    content = {
        "outcome": "failed",
        "modules": [
            {"module": "werkzeug", "ok": False,
             "stderr_tail": "ImportError: cannot import name 'url_quote'"},
            {"module": "flask", "ok": True, "stderr_tail": ""},
        ],
        "build_smoke": {"label": "vite build", "ok": False,
                        "stderr_tail": "[UNRESOLVED_ENTRY] index.html"},
        "failure_signature": "smoke_import|werkzeug",
    }
    fc = FailureContext.from_report("smoke", content)
    assert fc.gate == "smoke"
    assert fc.classification == "unknown"
    assert "url_quote" in fc.failure_text
    assert "vite build" in fc.failure_text
    assert "flask" not in fc.failure_text
    assert fc.failure_signature == "smoke_import|werkzeug"


def test_failure_context_api_check_renders_failed_references():
    """API_CHECK folds its failed references into ``module.name: error`` lines
    and appends any probe_error."""
    from cgx.session.repair.context import FailureContext
    content = {
        "outcome": "failed",
        "failed_references": [
            {"module": "werkzeug.urls", "name": "url_quote",
             "error": "ImportError: cannot import name 'url_quote'"},
        ],
        "probe_error": "probe crashed: exit 1",
        "failure_signature": "api_check|werkzeug.urls.url_quote",
    }
    fc = FailureContext.from_report("api_check", content)
    assert fc.gate == "api_check"
    assert fc.classification == "unknown"
    assert "werkzeug.urls.url_quote: ImportError" in fc.failure_text
    assert "probe crashed: exit 1" in fc.failure_text


def test_failure_context_explicit_classification_overrides_gate_default():
    """A caller-supplied classification wins over the derived token."""
    from cgx.session.repair.context import FailureContext
    fc = FailureContext.from_report(
        "smoke", {"outcome": "failed", "modules": []},
        classification="smoke_import_failure")
    assert fc.classification == "smoke_import_failure"


def test_failure_context_truncates_failure_text():
    """A pathological multi-thousand-line dump is bounded for small models."""
    from cgx.session.repair.context import FAILURE_TEXT_LIMIT, FailureContext
    huge = "E   AssertionError: boom\n" + ("x" * (FAILURE_TEXT_LIMIT * 2))
    fc = FailureContext.from_report(
        "verify", {"outcome": "assertions_failed", "stdout": huge})
    assert len(fc.failure_text) == FAILURE_TEXT_LIMIT


def test_failure_context_to_dict_is_json_friendly():
    """``to_dict`` renders the tuple fields as lists for tracing/persistence."""
    from cgx.session.repair.context import FailureContext
    fc = FailureContext.from_report(
        "verify", {"outcome": "assertions_failed", "stdout": "E   assert 0"},
        manifest_files=["a.py"], installed_packages=["pytest"])
    d = fc.to_dict()
    assert isinstance(d["traceback_files"], list)
    assert d["manifest_files"] == ["a.py"]
    assert d["installed_packages"] == ["pytest"]
    assert d["gate"] == "verify"


# --------------------- RepairLedger working memory (P2.4) -------------------

def test_repair_ledger_round_trips_content_and_normalizes_targets():
    """``from_content``/``to_content`` survive a store round trip; targets
    are de-duped and sorted so attempt identity is order-insensitive."""
    from cgx.session.repair.ledger import RepairLedger
    led = RepairLedger().append(
        "regenerate_files", ["b.py", "a.py", "a.py"], "sig1",
        rationale="try regen")
    content = led.to_content()
    assert content["attempts"][0]["targets"] == ["a.py", "b.py"]
    assert content["attempts"][0]["outcome"] == "pending"
    # Re-hydrating an equivalent (differently-ordered) targets list yields the
    # same normalized attempt.
    rebuilt = RepairLedger.from_content(
        {"attempts": [{"action": "regenerate_files",
                       "targets": ["b.py", "a.py"], "outcome": "pending",
                       "signature": "sig1"}]})
    assert rebuilt.attempts[0].targets == ("a.py", "b.py")


def test_repair_ledger_finalize_pending_resolves_by_signature():
    """The trailing pending attempt becomes ``still_failing`` when the live
    signature is unchanged, ``changed`` when it moved."""
    from cgx.session.repair.ledger import RepairLedger
    led = RepairLedger().append("add_dependency", ["flask"], "sigA")
    same = led.finalize_pending("sigA")
    assert same.attempts[-1].outcome == "still_failing"
    moved = led.finalize_pending("sigB")
    assert moved.attempts[-1].outcome == "changed"
    # A ledger with no pending tail is returned untouched.
    assert same.finalize_pending("sigZ") is same


def test_repair_ledger_has_attempted_only_blocks_dead_ends():
    """Only an identical ``(action, targets)`` with a ``still_failing``
    outcome blocks a re-proposal; pending / changed do not."""
    from cgx.session.repair.ledger import RepairLedger
    led = RepairLedger.from_content({"attempts": [
        {"action": "add_dependency", "targets": ["flask"],
         "outcome": "still_failing", "signature": "s"},
        {"action": "patch_files", "targets": ["a.py"],
         "outcome": "changed", "signature": "s"},
        {"action": "regenerate_files", "targets": ["b.py"],
         "outcome": "pending", "signature": "s"},
    ]})
    assert led.has_attempted("add_dependency", ["flask"]) is True
    # order-insensitive + a non-dead-end outcome does not block
    assert led.has_attempted("patch_files", ["a.py"]) is False
    assert led.has_attempted("regenerate_files", ["b.py"]) is False
    # a never-tried action is free
    assert led.has_attempted("remove_dependency", ["flask"]) is False


def test_loop_budget_threads_repair_ledger_fact_id():
    """The ledger id rides ``repair_chain_inputs`` and re-hydrates; a chain
    that never opened a ledger keeps the identical wire shape as before."""
    from cgx.session.budget import LoopBudget
    empty = LoopBudget.from_inputs({"repair_attempt": 1})
    assert empty.repair_ledger_fact_id is None
    assert "repair_ledger_fact_id" not in empty.repair_chain_inputs()
    threaded = empty.with_repair_ledger("fact_123")
    chain = threaded.repair_chain_inputs()
    assert chain["repair_ledger_fact_id"] == "fact_123"
    assert LoopBudget.from_inputs(chain).repair_ledger_fact_id == "fact_123"


# --------------------- DIAGNOSE executor (P2.3) -------------------

def _runtime_boot_report(session, produced_by, tail):
    """A failing RUNTIME_REPORT whose probe carries a boot traceback."""
    return Artifact.new(
        session_id=session.session_id, produced_by_task_id=produced_by,
        kind=ArtifactKind.RUNTIME_REPORT,
        content={"outcome": "failed",
                 "probes": [{"file": "app.py", "kind": "import",
                             "ok": False, "stderr_tail": tail}]})


def _smoke_fail_report(session, produced_by):
    """A failing SMOKE_REPORT -- classifies to the reasoning token ``unknown``."""
    return Artifact.new(
        session_id=session.session_id, produced_by_task_id=produced_by,
        kind=ArtifactKind.SMOKE_REPORT,
        content={"outcome": "failed",
                 "modules": [{"module": "app", "ok": False,
                              "stderr_tail": "ImportError: no module flask"}]})


def test_diagnose_deterministic_runtime_failure_targets_traceback_file(
        store, tmp_path: Path):
    """runtime_failure with a localized traceback -> model-free targeted regen."""
    from cgx.session.tasks.diagnose import run_diagnose
    (tmp_path / "app.py").write_text("import missing\n", encoding="utf-8")
    session = Session.new("build a flask app", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    art = _runtime_boot_report(
        session, "t_run",
        'Traceback (most recent call last):\n'
        '  File "app.py", line 1, in <module>\n'
        "ModuleNotFoundError: No module named 'missing'\n")
    store.save_artifact(art)
    t = TaskNode.new(session.session_id, TaskKind.DIAGNOSE, "diagnose",
                     inputs={"runtime_artifact_id": art.artifact_id})
    result = run_diagnose(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.failure is None
    assert result.artifact is not None
    assert result.artifact.kind is ArtifactKind.DIAGNOSIS
    assert result.outputs["minimal_action"] == "regenerate_files"
    assert result.outputs["target_files"] == ["app.py"]
    assert result.outputs["used_model"] is False
    assert result.artifact.content["confidence"] == 0.8


def test_diagnose_runtime_failure_without_disk_file_falls_back(
        store, tmp_path: Path):
    """A traceback file absent on disk is not deterministic -> ReAct/escalate."""
    from cgx.session.tasks.diagnose import run_diagnose
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    art = _runtime_boot_report(
        session, "t_run",
        'File "ghost.py", line 1, in <module>\nRuntimeError: boom\n')
    store.save_artifact(art)
    t = TaskNode.new(session.session_id, TaskKind.DIAGNOSE, "diagnose",
                     inputs={"runtime_artifact_id": art.artifact_id})
    result = run_diagnose(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    # No provider + no on-disk target -> the additive escalate fallback.
    assert result.outputs["minimal_action"] == "escalate"
    assert result.outputs["used_model"] is False


def test_diagnose_react_loop_emits_model_verdict(store, tmp_path: Path):
    """An ambiguous SMOKE failure + a provider verdict -> that minimal_action."""
    import json
    from cgx.session.tasks.diagnose import run_diagnose
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    art = _smoke_fail_report(session, "t_smoke")
    store.save_artifact(art)
    provider = _StubProvider(json.dumps({
        "minimal_action": "add_dependency", "root_cause": "flask not installed",
        "add_dependencies": ["flask"], "rationale": "import fails",
        "confidence": 0.9}))
    t = TaskNode.new(session.session_id, TaskKind.DIAGNOSE, "diagnose",
                     inputs={"smoke_artifact_id": art.artifact_id})
    result = run_diagnose(t, ExecutorDeps(
        project_root=str(tmp_path), store=store, provider=provider))
    assert result.failure is None
    assert result.outputs["minimal_action"] == "add_dependency"
    assert result.outputs["used_model"] is True
    assert result.artifact.content["add_dependencies"] == ["flask"]
    assert result.artifact.content["confidence"] == 0.9


def test_diagnose_react_runs_tool_then_verdict(store, tmp_path: Path):
    """The loop may call a read-only tool, fold the observation, then decide."""
    import json
    from cgx.session.tasks.diagnose import run_diagnose

    class _ScriptedProvider:
        def __init__(self, replies):
            self._replies = list(replies)
            self.calls: list = []

        def chat(self, **kwargs):
            self.calls.append(kwargs)
            return {"content": self._replies.pop(0)}

    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    art = _smoke_fail_report(session, "t_smoke")
    store.save_artifact(art)
    provider = _ScriptedProvider([
        json.dumps({"tool": "inspect_packages"}),
        json.dumps({"minimal_action": "escalate", "root_cause": "unclear",
                    "rationale": "not enough signal", "confidence": 0.1}),
    ])
    t = TaskNode.new(session.session_id, TaskKind.DIAGNOSE, "diagnose",
                     inputs={"smoke_artifact_id": art.artifact_id})
    result = run_diagnose(t, ExecutorDeps(
        project_root=str(tmp_path), store=store, provider=provider))
    assert result.failure is None
    assert len(provider.calls) == 2
    assert result.outputs["minimal_action"] == "escalate"
    assert result.outputs["used_model"] is True


def test_diagnose_no_provider_degrades_to_escalate(store, tmp_path: Path):
    """An ambiguous failure with no provider hands off to the regenerate path."""
    from cgx.session.tasks.diagnose import run_diagnose
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    art = _smoke_fail_report(session, "t_smoke")
    store.save_artifact(art)
    t = TaskNode.new(session.session_id, TaskKind.DIAGNOSE, "diagnose",
                     inputs={"smoke_artifact_id": art.artifact_id})
    result = run_diagnose(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.outputs["minimal_action"] == "escalate"
    assert result.outputs["used_model"] is False
    assert result.artifact.content["confidence"] == 0.0


def test_diagnose_garbled_output_degrades_to_escalate(store, tmp_path: Path):
    """A provider that returns non-JSON prose degrades cleanly to escalate."""
    from cgx.session.tasks.diagnose import run_diagnose
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    art = _smoke_fail_report(session, "t_smoke")
    store.save_artifact(art)
    t = TaskNode.new(session.session_id, TaskKind.DIAGNOSE, "diagnose",
                     inputs={"smoke_artifact_id": art.artifact_id})
    result = run_diagnose(t, ExecutorDeps(
        project_root=str(tmp_path), store=store,
        provider=_StubProvider("no json here, just prose")))
    assert result.outputs["minimal_action"] == "escalate"
    assert result.outputs["used_model"] is True


def test_diagnose_missing_source_report_fails(store, tmp_path: Path):
    """No runtime/api/smoke/verify id in inputs -> a clear hard failure."""
    from cgx.session.tasks.diagnose import run_diagnose
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    t = TaskNode.new(session.session_id, TaskKind.DIAGNOSE, "diagnose",
                     inputs={})
    result = run_diagnose(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.failure
    assert "source report" in result.failure


def test_diagnose_wrong_artifact_kind_fails(store, tmp_path: Path):
    """A source id pointing at the wrong artifact kind -> a hard failure."""
    from cgx.session.tasks.diagnose import run_diagnose
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    art = Artifact.new(
        session_id=session.session_id, produced_by_task_id="t_x",
        kind=ArtifactKind.WORK_PLAN, content={})
    store.save_artifact(art)
    t = TaskNode.new(session.session_id, TaskKind.DIAGNOSE, "diagnose",
                     inputs={"runtime_artifact_id": art.artifact_id})
    result = run_diagnose(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.failure
    assert "missing or wrong" in result.failure


def test_diagnose_verdict_traced_to_agent_log_when_enabled(
        store, tmp_path: Path):
    """With tracing on, the verdict + tool step land as agent-log records."""
    import json
    from cgx.session.tasks.diagnose import run_diagnose
    from cgx import trace as trace_mod

    class _ToolThenVerdict:
        def __init__(self):
            self._replies = [
                json.dumps({"tool": "inspect_packages"}),
                json.dumps({"minimal_action": "escalate",
                            "root_cause": "unclear", "confidence": 0.0}),
            ]

        def chat(self, **kwargs):
            return {"content": self._replies.pop(0)}

    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    art = _smoke_fail_report(session, "t_smoke")
    store.save_artifact(art)
    trace_mod.set_trace_enabled(True)
    token = trace_mod.set_trace_context(
        session_id=session.session_id, task_id="task_diag",
        project_root=str(tmp_path))
    try:
        t = TaskNode.new(session.session_id, TaskKind.DIAGNOSE, "diagnose",
                         inputs={"smoke_artifact_id": art.artifact_id})
        run_diagnose(t, ExecutorDeps(
            project_root=str(tmp_path), store=store,
            provider=_ToolThenVerdict()))
    finally:
        trace_mod.reset_trace_context(token)
        trace_mod.set_trace_enabled(False)

    records = _read_agent_log(tmp_path)
    verdicts = [r for r in records if r.get("event") == "diagnose_verdict"]
    steps = [r for r in records if r.get("event") == "diagnose_step"]
    assert verdicts, "diagnose_verdict should be traced when enabled"
    assert verdicts[0]["minimal_action"] == "escalate"
    assert verdicts[0]["used_model"] is True
    assert steps and steps[0]["tool"] == "inspect_packages"


# --------------------- DIAGNOSE repair ledger (P2.4) -------------------

def test_diagnose_emits_repair_ledger_fact_recording_the_proposal(
        store, tmp_path: Path):
    """Every round appends its proposed action to a REPAIR_LEDGER fact and
    threads that fact id forward via outputs."""
    import json
    from cgx.session.tasks.diagnose import run_diagnose
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    art = _smoke_fail_report(session, "t_smoke")
    store.save_artifact(art)
    provider = _StubProvider(json.dumps({
        "minimal_action": "add_dependency", "root_cause": "flask missing",
        "add_dependencies": ["flask"], "rationale": "import fails",
        "confidence": 0.9}))
    t = TaskNode.new(session.session_id, TaskKind.DIAGNOSE, "diagnose",
                     inputs={"smoke_artifact_id": art.artifact_id})
    result = run_diagnose(t, ExecutorDeps(
        project_root=str(tmp_path), store=store, provider=provider))
    ledgers = [f for f in result.facts if f.kind is FactKind.REPAIR_LEDGER]
    assert len(ledgers) == 1
    attempts = ledgers[0].content["attempts"]
    assert len(attempts) == 1
    assert attempts[0]["action"] == "add_dependency"
    assert attempts[0]["targets"] == ["flask"]
    assert attempts[0]["outcome"] == "pending"
    assert result.outputs["repair_ledger_fact_id"] == ledgers[0].fact_id


def test_diagnose_never_repeats_a_still_failing_action(store, tmp_path: Path):
    """A prior attempt that left the failure standing is a proven dead end:
    re-proposing it degrades to the additive escalate hand-off."""
    import json
    from cgx.session.tasks.diagnose import run_diagnose
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    art = _smoke_fail_report(session, "t_smoke")
    store.save_artifact(art)
    prior = Fact.new(session.session_id, FactKind.REPAIR_LEDGER, {"attempts": [
        {"action": "add_dependency", "targets": ["flask"],
         "outcome": "still_failing", "signature": "old"}]})
    store.add_fact(prior)
    provider = _StubProvider(json.dumps({
        "minimal_action": "add_dependency", "add_dependencies": ["flask"],
        "root_cause": "flask missing", "confidence": 0.9}))
    t = TaskNode.new(session.session_id, TaskKind.DIAGNOSE, "diagnose",
                     inputs={"smoke_artifact_id": art.artifact_id,
                             "repair_ledger_fact_id": prior.fact_id})
    result = run_diagnose(t, ExecutorDeps(
        project_root=str(tmp_path), store=store, provider=provider))
    assert result.outputs["minimal_action"] == "escalate"
    # the prior ledger is superseded (stale) and a fresh one threaded forward
    assert store.load_kb(session.session_id).facts[prior.fact_id].stale is True
    assert result.outputs["repair_ledger_fact_id"] != prior.fact_id


def test_diagnose_finalizes_prior_pending_attempt_across_rounds(
        store, tmp_path: Path):
    """Round 2 resolves round 1's pending proposal: an unchanged signature
    marks it ``still_failing`` and supersedes the old ledger fact."""
    import json
    from cgx.session.tasks.diagnose import run_diagnose
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    art = _smoke_fail_report(session, "t_smoke")
    store.save_artifact(art)
    provider = _StubProvider(json.dumps({
        "minimal_action": "add_dependency", "add_dependencies": ["flask"],
        "root_cause": "flask missing", "confidence": 0.9}))
    deps = ExecutorDeps(
        project_root=str(tmp_path), store=store, provider=provider)
    t1 = TaskNode.new(session.session_id, TaskKind.DIAGNOSE, "diagnose",
                      inputs={"smoke_artifact_id": art.artifact_id})
    r1 = run_diagnose(t1, deps)
    for f in r1.facts:
        store.add_fact(f)
    ledger_id = r1.outputs["repair_ledger_fact_id"]
    # Round 2 sees the same failure signature and the threaded ledger id.
    t2 = TaskNode.new(session.session_id, TaskKind.DIAGNOSE, "diagnose",
                      inputs={"smoke_artifact_id": art.artifact_id,
                              "repair_ledger_fact_id": ledger_id})
    r2 = run_diagnose(t2, deps)
    new_ledger = [f for f in r2.facts if f.kind is FactKind.REPAIR_LEDGER][0]
    outcomes = [a["outcome"] for a in new_ledger.content["attempts"]]
    assert outcomes[0] == "still_failing"
    assert store.load_kb(session.session_id).facts[ledger_id].stale is True


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


def test_repair_executor_emits_empty_plan_for_assertion_drift(
        store, tmp_path: Path):
    """Assertion drift with no locator/provider -> empty diffs + can_apply False."""
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
    assert result.outputs["classification"] == "assertion_drift"
    assert result.outputs["can_apply"] is False
    assert result.outputs["diff_count"] == 0


def test_repair_executor_assertion_drift_targets_impl_file(store, tmp_path: Path):
    """No patch + traceback naming a source file -> targeted regenerate.

    The failing test asserts a 201 the handler returns 200 for. With no
    provider the bounded LLM patch is a no-op, so the executor must fall
    back to a *targeted* regenerate of only the implementation file the
    traceback flows through (``src/handlers.py``) -- never the test that
    encodes the contract -- carrying the prior scaffold artifact id so the
    router can regenerate against it instead of the whole tree.
    """
    from cgx.session.tasks import repair as _repair_module  # noqa: F401
    src_rel = "src/handlers.py"
    test_rel = "tests/test_backend.py"
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / src_rel).write_text(
        "def login():\n    return '', 200\n", encoding="utf-8")
    (tmp_path / test_rel).write_text(
        "from src.handlers import login\n\n\ndef test_login():\n"
        "    assert login()[1] == 201\n", encoding="utf-8")
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    verify_artifact = Artifact.new(
        session_id=session.session_id, produced_by_task_id="t_verify",
        kind=ArtifactKind.VERIFY_REPORT,
        content={
            "outcome": "assertions_failed",
            "returncode": 1,
            "changed_files": [test_rel],
            "scaffold_artifact_id": "art_scaffold_1",
            "stdout": (
                "tests/test_backend.py:5: in test_login\n"
                "    assert login()[1] == 201\n"
                "src/handlers.py:2: in login\n"
                "    return '', 200\n"
                "E   assert 200 == 201"),
            "stderr": "",
        })
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
    assert result.outputs["classification"] == "assertion_drift"
    assert result.outputs["strategy"] == "regenerate"
    assert result.outputs["can_apply"] is False
    assert result.outputs["scaffold_artifact_id"] == "art_scaffold_1"
    ec = result.outputs["extra_constraints"]
    assert ec["target_files"] == [src_rel]
    # The test file that encodes the contract is never a regenerate target.
    assert test_rel not in ec["target_files"]


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


def test_generate_repair_files_flags_localized_files():
    """``localized_files`` adds a traceback note + per-file marker."""
    import json as _json
    from cgx.answer.engine import generate_repair_files
    src = "def add(a, b):\n    return a - b\n"
    fixed = "def add(a, b):\n    return a + b\n"
    provider = _StubProvider(_json.dumps({"files": [
        {"path": "src/calc.py", "content": fixed}]}))
    out = generate_repair_files(
        provider,
        goal="a calculator",
        failure_text="E   assert 1 == 3",
        files=[
            {"path": "src/calc.py", "content": src},
            {"path": "tests/test_calc.py", "content": "T"},
        ],
        localized_files=["src/calc.py"],
    )
    assert out == {"src/calc.py": fixed}
    prompt = provider.calls[0]["messages"][-1]["content"]
    assert "TRACEBACK LOCALIZATION" in prompt
    assert "src/calc.py" in prompt
    assert "(traceback points here)" in prompt


def test_repair_executor_llm_logic_repair_emits_patch(store, tmp_path: Path):
    """assertion-drift failure + provider -> bounded LLM patch (can_apply)."""
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
    assert result.outputs["classification"] == "assertion_drift"
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
                "repair_attempt": 5})
    deps = ExecutorDeps(
        project_root=str(tmp_path), store=store, provider=provider)
    from cgx.session.tasks.base import _REGISTRY
    result = _REGISTRY[TaskKind.REPAIR](repair_task, deps)
    assert result.failure is None
    assert result.outputs["classification"] == "assertion_drift"
    assert result.outputs["can_apply"] is False
    assert result.outputs["diff_count"] == 0
    assert provider.calls == []


def test_repair_executor_localizes_traceback_source_file(
        store, tmp_path: Path):
    """A source file named only in the traceback is repaired + flagged.

    ``changed_files`` lists the test file APPLY wrote, but the failure
    flows through ``src/util.py`` (named only in the traceback frames).
    The executor must pull that source file into the repair context,
    flag it to the provider, and emit a patch against it.
    """
    import json as _json
    from cgx.session.tasks import repair as _repair_module  # noqa: F401
    src_rel = "src/util.py"
    test_rel = "tests/test_util.py"
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / src_rel).write_text(
        "def scale(x):\n    return x - 1\n", encoding="utf-8")
    (tmp_path / test_rel).write_text(
        "from src.util import scale\n\n\ndef test_scale():\n"
        "    assert scale(2) == 4\n", encoding="utf-8")
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    verify_artifact = Artifact.new(
        session_id=session.session_id, produced_by_task_id="t_verify",
        kind=ArtifactKind.VERIFY_REPORT,
        content={
            "outcome": "assertions_failed",
            "returncode": 1,
            "changed_files": [test_rel],
            "stdout": (
                "tests/test_util.py:5: in test_scale\n"
                "    assert scale(2) == 4\n"
                "src/util.py:2: in scale\n"
                "    return x - 1\n"
                "E   assert 1 == 4"),
            "stderr": "",
        })
    store.save_artifact(verify_artifact)
    provider = _StubProvider(_json.dumps({"files": [
        {"path": src_rel, "content": "def scale(x):\n    return x * 2\n"}]}))
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
    assert result.outputs["can_apply"] is True
    diffs = result.artifact.content["diffs"]
    assert diffs[0]["file"] == src_rel
    assert "return x * 2" in diffs[0]["patch"]
    prompt = provider.calls[0]["messages"][-1]["content"]
    assert "TRACEBACK LOCALIZATION" in prompt
    assert "src/util.py" in prompt
    assert "(traceback points here)" in prompt


def test_repair_executor_retrieval_feeds_candidate_files(
        store, tmp_path: Path, monkeypatch):
    """#6: with an index wired in, retrieval fills the unused repair slots.

    ``changed_files`` names only the test file and the plain assertion
    failure carries no traceback frames, so the failure-localized
    candidates leave slots free. A stubbed hybrid-retrieval result points
    at ``src/helper.py``; the executor must pull that source file into the
    repair context and emit a patch against it.

    Retrieval is gated on the index manifest existing on disk (a
    greenfield project that was never indexed has none), so materialise a
    ``meta.json`` under ``index_dir`` to model an indexed project.
    """
    import json as _json
    from cgx.session.tasks import repair as _repair_module  # noqa: F401
    src_rel = "src/helper.py"
    test_rel = "tests/test_helper.py"
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / src_rel).write_text(
        "def scale(x):\n    return x - 1\n", encoding="utf-8")
    (tmp_path / test_rel).write_text(
        "def test_scale():\n    assert 1 == 3\n", encoding="utf-8")
    index_dir = tmp_path / "idx"
    index_dir.mkdir()
    (index_dir / "meta.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "cgx.pipeline.auto.run_query_auto",
        lambda **kwargs: {"top_files": [{"file": src_rel}]})
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    verify_artifact = Artifact.new(
        session_id=session.session_id, produced_by_task_id="t_verify",
        kind=ArtifactKind.VERIFY_REPORT,
        content={
            "outcome": "assertions_failed",
            "returncode": 1,
            "changed_files": [test_rel],
            "stdout": "E   assert 1 == 3",
            "stderr": "",
        })
    store.save_artifact(verify_artifact)
    provider = _StubProvider(_json.dumps({"files": [
        {"path": src_rel, "content": "def scale(x):\n    return x * 2\n"}]}))
    repair_task = TaskNode.new(
        session.session_id, TaskKind.REPAIR, "repair",
        inputs={"verify_artifact_id": verify_artifact.artifact_id,
                "mode": SessionMode.GREENFIELD.value,
                "repair_attempt": 1})
    deps = ExecutorDeps(
        project_root=str(tmp_path), store=store, provider=provider,
        index_dir=str(index_dir), records_path="rec")
    from cgx.session.tasks.base import _REGISTRY
    result = _REGISTRY[TaskKind.REPAIR](repair_task, deps)
    assert result.failure is None
    assert result.outputs["can_apply"] is True
    diffs = result.artifact.content["diffs"]
    assert any(d["file"] == src_rel for d in diffs)
    prompt = provider.calls[0]["messages"][-1]["content"]
    assert "src/helper.py" in prompt


def test_repair_executor_retrieval_noop_without_index(
        store, tmp_path: Path, monkeypatch):
    """#6: no index in deps -> retrieval is never invoked (greenfield)."""
    import json as _json
    from cgx.session.tasks import repair as _repair_module  # noqa: F401
    called = {"n": 0}

    def _boom(**kwargs):
        called["n"] += 1
        return {"top_files": []}

    monkeypatch.setattr("cgx.pipeline.auto.run_query_auto", _boom)
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
                "repair_attempt": 1})
    deps = ExecutorDeps(
        project_root=str(tmp_path), store=store, provider=provider)
    from cgx.session.tasks.base import _REGISTRY
    result = _REGISTRY[TaskKind.REPAIR](repair_task, deps)
    assert result.outputs["can_apply"] is True
    assert called["n"] == 0


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
    assert result.artifact is not None
    assert result.artifact.kind is ArtifactKind.REPAIR_PLAN
    assert result.outputs["classification"] == "smoke_import_failure"
    assert result.outputs["can_apply"] is False
    assert result.outputs["diff_count"] == 0
    assert result.artifact.content["failed_modules"] == ["werkzeug"]
    assert "werkzeug" in result.artifact.content["rationale"]


def test_repair_executor_emits_regenerate_for_build_smoke_failure(
        store, tmp_path: Path):
    """A JS build-smoke break -> strategy=regenerate with targeted files.
    
    If the error names a file (e.g. TS2345 in App.jsx), it should be extracted
    and used for a targeted AST regeneration instead of escalating.
    """
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
    
    # Touch the file so the target file extractor confirms it exists
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "App.jsx").touch()
    
    deps = ExecutorDeps(project_root=str(tmp_path), store=store)
    from cgx.session.tasks.base import _REGISTRY
    result = _REGISTRY[TaskKind.REPAIR](repair_task, deps)
    assert result.failure is None
    assert result.outputs["strategy"] == "regenerate"
    constraints = result.outputs["extra_constraints"]
    assert constraints["kind"] == "invalid_build_smoke"
    assert constraints["target_files"] == ["src/App.jsx"]
    assert "TS2345" in constraints["build_error"]
    assert "build" in result.artifact.content["rationale"].lower()


def test_repair_executor_names_unresolved_entry_module_as_missing_file(
        store, tmp_path: Path):
    """An unresolved entry module -> the absent file is named for the regenerate."""
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
                 "build_smoke": {
                     "label": "npm run build --silent",
                     "ok": False,
                     "stderr_tail": (
                         "error during build:\n"
                         "[UNRESOLVED_ENTRY] Cannot resolve entry module "
                         "index.html.\n")}})
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
    assert constraints["kind"] == "missing_entry_module"
    assert [f["path"] for f in constraints["missing_files"]] == ["index.html"]
    assert constraints["missing_files"][0]["description"]


def test_unresolved_entry_paths_parses_rollup_and_vite_wordings():
    from cgx.session.repair.classify import unresolved_entry_paths
    assert unresolved_entry_paths(
        "[UNRESOLVED_ENTRY] Cannot resolve entry module index.html.") == (
            "index.html",)
    assert unresolved_entry_paths(
        'Could not resolve entry module "./src/main.jsx".') == (
            "src/main.jsx",)
    # Deduplicated across repeats, and an unrelated build error is inert.
    assert unresolved_entry_paths(
        "Cannot resolve entry module index.html\n"
        "Cannot resolve entry module index.html\n") == ("index.html",)
    assert unresolved_entry_paths("src/App.jsx: TS2345") == ()


def test_unresolved_import_sources_parses_rollup_resolution_error():
    from cgx.session.repair.classify import unresolved_import_sources
    assert unresolved_import_sources(
        'Could not resolve "./index.css" from "src/main.jsx"') == (
            "src/main.jsx",)
    # De-duplicated across repeats; an unrelated build error is inert.
    assert unresolved_import_sources(
        'Could not resolve "./a" from "src/x.jsx"\n'
        'Could not resolve "./b" from "src/x.jsx"\n') == ("src/x.jsx",)
    assert unresolved_import_sources("src/App.jsx: TS2345") == ()


def test_repair_executor_targets_importer_for_build_resolution_error(
        store, tmp_path: Path):
    """A 'Could not resolve X from Y' build break names Y for a targeted regen."""
    from cgx.session.tasks import repair as _repair_module  # noqa: F401
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.jsx").write_text("import './index.css'\n")
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    smoke_task = TaskNode.new(
        session.session_id, TaskKind.SMOKE, "smoke", inputs={})
    smoke_artifact = Artifact.new(
        session_id=session.session_id,
        produced_by_task_id=smoke_task.task_id,
        kind=ArtifactKind.SMOKE_REPORT,
        content={"outcome": "failed", "failed_modules": [],
                 "failure_signature": "smoke_import|npm run build --silent",
                 "modules": [], "scaffold_artifact_id": "art_scaf",
                 "build_smoke": {
                     "label": "npm run build --silent", "ok": False,
                     "stderr_tail": ('error during build:\n'
                                     'Could not resolve "./index.css" '
                                     'from "src/main.jsx"\n')}})
    store.save_task(smoke_task)
    store.save_artifact(smoke_artifact)
    repair_task = TaskNode.new(
        session.session_id, TaskKind.REPAIR, "repair",
        inputs={"smoke_artifact_id": smoke_artifact.artifact_id,
                "mode": SessionMode.GREENFIELD.value, "repair_attempt": 1})
    deps = ExecutorDeps(project_root=str(tmp_path), store=store)
    from cgx.session.tasks.base import _REGISTRY
    result = _REGISTRY[TaskKind.REPAIR](repair_task, deps)
    assert result.failure is None
    assert result.outputs["strategy"] == "regenerate"
    constraints = result.outputs["extra_constraints"]
    assert constraints["kind"] == "invalid_build_smoke"
    assert constraints["target_files"] == ["src/main.jsx"]
    assert result.outputs["scaffold_artifact_id"] == "art_scaf"


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


def test_classify_requires_package_runtime_error():
    """``requires the <pkg> package to be installed`` -> missing_dependency.

    The live failure (starlette's TestClient guard for its optional
    httpx extra) names the exact pip package; it must route to an
    install, never a source regenerate.
    """
    from cgx.session.repair.classify import classify_verify_report
    content = {
        "outcome": "collection_error",
        "returncode": 2,
        "stdout": (
            "E   RuntimeError: The starlette.testclient module requires "
            "the httpx package to be installed.\n"
            "E   $ pip install httpx\n"),
        "stderr": "",
    }
    assert classify_verify_report(content) == "missing_dependency"


def test_classify_missing_dependency_wins_over_missing_module():
    """The guard's internal ModuleNotFoundError must not misroute the
    failure to missing_module_pythonpath (a source regenerate)."""
    from cgx.session.repair.classify import classify_verify_report
    content = {
        "outcome": "collection_error",
        "returncode": 2,
        "stdout": (
            "E   ModuleNotFoundError: No module named 'httpx'\n"
            "E   RuntimeError: The starlette.testclient module requires "
            "the httpx package to be installed.\n"),
        "stderr": "",
    }
    assert classify_verify_report(content) == "missing_dependency"


def test_required_package_names_extracts_and_dedupes():
    """Package names from the RuntimeError shape, ordered and deduped."""
    from cgx.session.repair.classify import required_package_names
    content = {"stdout": (
        "E   RuntimeError: The starlette.testclient module requires "
        "the httpx package to be installed.\n"
        "E   RuntimeError: The foo module requires the bar-baz package "
        "to be installed.\n"
        "E   RuntimeError: The starlette.testclient module requires "
        "the httpx package to be installed.\n")}
    assert required_package_names(content) == ("httpx", "bar-baz")


def test_classify_relative_import_beyond_top_level():
    """`attempted relative import beyond top-level package` -> its own token."""
    from cgx.session.repair.classify import classify_verify_report
    content = {
        "outcome": "collection_error",
        "returncode": 2,
        "stdout": (
            "ImportError while importing test module 'tests/test_app.py'.\n"
            "src/app.py:1: in <module>\n"
            "    from ..config import API_BASE\n"
            "E   ImportError: attempted relative import beyond top-level "
            "package\n"),
        "stderr": "",
    }
    assert classify_verify_report(content) == "relative_import_error"


def test_classify_relative_import_no_known_parent():
    """The `with no known parent package` phrasing maps to the same token."""
    from cgx.session.repair.classify import classify_verify_report
    content = {
        "outcome": "collection_error",
        "returncode": 2,
        "stdout": (
            "E   ImportError: attempted relative import with no known parent "
            "package\n"),
        "stderr": "",
    }
    assert classify_verify_report(content) == "relative_import_error"


def test_relative_import_error_routes_to_regenerate():
    """The classifier token forces a regenerate strategy, not an LLM patch."""
    from cgx.session.tasks.repair import _select_repair_strategy
    strategy, constraints = _select_repair_strategy(
        classification="relative_import_error", diffs=[],
        rationale="re-author the failing module", extra_plan_fields={},
        locations_payload=[])
    assert strategy == "regenerate"
    assert constraints["kind"] == "relative_import_error"


def test_classify_circular_import_partially_initialized():
    """The circular-import ImportError variant maps to its own token.

    The live failure fell through to ``unknown`` because "cannot import
    name 'x' from partially initialized module 'm'" does not match the
    plain from-'<pkg>' shape; the bounded LLM patch then produced a no-op
    diff and the session died. The dedicated token routes it to a
    regenerate instead.
    """
    from cgx.session.repair.classify import classify_verify_report
    content = {
        "outcome": "collection_error",
        "returncode": 2,
        "stdout": (
            "ImportError while importing test module 'tests/test_api.py'.\n"
            "backend/models.py:3: in <module>\n"
            "    from backend.routes import compute_expression\n"
            "E   ImportError: cannot import name 'compute_expression' from "
            "partially initialized module 'backend.routes' (most likely due "
            "to a circular import)\n"),
        "stderr": "",
    }
    assert classify_verify_report(content) == "circular_import"


def test_circular_import_modules_extracts_names():
    """Both message shapes yield the module names, deduped and ordered."""
    from cgx.session.repair.classify import circular_import_modules
    content = {"stdout": (
        "E   ImportError: cannot import name 'x' from partially initialized "
        "module 'backend.routes' (most likely due to a circular import)\n"
        "E   AttributeError: partially initialized module 'backend.models' "
        "has no attribute 'y' (most likely due to a circular import)\n"
        "E   ImportError: cannot import name 'z' from partially initialized "
        "module 'backend.routes' (most likely due to a circular import)\n")}
    assert circular_import_modules(content) == (
        "backend.routes", "backend.models")


def test_circular_import_routes_to_regenerate():
    """The token forces a regenerate carrying the cycle's module names."""
    from cgx.session.tasks.repair import _select_repair_strategy
    strategy, constraints = _select_repair_strategy(
        classification="circular_import", diffs=[],
        rationale="break the cycle",
        extra_plan_fields={"circular_modules": ["backend.routes"]},
        locations_payload=[])
    assert strategy == "regenerate"
    assert constraints["kind"] == "circular_import"
    assert constraints["modules"] == ["backend.routes"]


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


def test_classify_unrecognized_collection_error_is_first_class():
    """A collection_error that matches no classifier is its own token.

    Before this fix it fell back to ``unknown`` (-> silent regenerate);
    now it surfaces as ``collection_error`` so the executor can escalate.
    An assertion failure with no pattern maps to ``assertion_drift``.
    """
    from cgx.session.repair.classify import classify_verify_report
    cerr = {
        "outcome": "collection_error",
        "returncode": 4,
        "stdout": ("ERROR: usage: pytest [options]\n"
                   "pytest: error: unrecognized arguments: --foo\n"),
        "stderr": "",
    }
    assert classify_verify_report(cerr) == "collection_error"
    assert_fail = {
        "outcome": "assertions_failed",
        "returncode": 1,
        "stdout": "E   assert 1 == 2\n",
        "stderr": "",
    }
    assert classify_verify_report(assert_fail) == "assertion_drift"


def test_repair_executor_unrecognized_collection_error_escalates(
        store, tmp_path: Path):
    """An opaque collection_error escalates instead of regenerating.

    pytest could not collect the suite and no mechanical classifier
    matched; the REPAIR plan must carry strategy='escalate' /
    can_apply=False (so the router halts) rather than strategy='regenerate'
    (the whack-a-mole loop).
    """
    from cgx.session.tasks import repair as _repair_module  # noqa: F401
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
            "returncode": 4,
            "stdout": ("ERROR: usage: pytest [options]\n"
                       "pytest: error: unrecognized arguments: --foo\n"),
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
    assert result.outputs["classification"] == "collection_error"
    assert result.outputs["can_apply"] is False
    assert result.outputs["diff_count"] == 0
    assert result.outputs["strategy"] == "escalate"
    assert result.outputs["rationale"]
    assert result.artifact.content["extra_constraints"]["kind"] == \
        "collection_error"


def test_router_repair_escalate_terminates_not_regenerate():
    """An escalate verdict halts the session even with regenerate available.

    A REPAIR that escalated an unrecognized collection_error carries
    strategy='escalate' / can_apply=False. Despite a SCAFFOLD ancestor and
    unspent regenerate budget, the router must NOT re-scaffold; it fails
    the session and marks the REPAIR node FAILED with the rationale.
    """
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    scaffold = TaskNode.new(
        session.session_id, TaskKind.SCAFFOLD, "scaffold",
        inputs={"mode": SessionMode.GREENFIELD.value})
    scaffold.status = TaskNodeStatus.DONE
    repair = TaskNode.new(
        session.session_id, TaskKind.REPAIR, "repair",
        parent_task_id=scaffold.task_id,
        inputs={"mode": SessionMode.GREENFIELD.value})
    repair.produced_artifact_id = "art_plan"
    repair.outputs = {
        "classification": "collection_error",
        "strategy": "escalate",
        "can_apply": False,
        "failure_signature": "verify|collection_error|x",
        "rationale": "pytest could not collect the test suite.",
    }
    repair.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=repair, tasks=[scaffold, repair])
    assert [a for a in plan.actions if isinstance(a, CreateTask)] == []
    session_status = [a for a in plan.actions
                      if isinstance(a, UpdateSessionStatus)]
    assert len(session_status) == 1
    assert session_status[0].status is SessionStatus.FAILED
    task_status = [a for a in plan.actions
                   if isinstance(a, UpdateTaskStatus)]
    assert len(task_status) == 1
    assert task_status[0].status is TaskNodeStatus.FAILED
    assert "collection_error" in (task_status[0].error or "")
    assert "collect the test suite" in (task_status[0].error or "")


def test_repair_verify_missing_dependency_routes_to_install_deps(
        store, tmp_path: Path):
    """VERIFY missing_dependency -> strategy=install_deps, not regenerate.

    Mirrors the live loop (starlette TestClient's httpx guard): the
    failure names the exact pip package, so REPAIR must target the venv
    rather than regenerate source that never imports the package.
    """
    from cgx.session.tasks import repair as _repair_module  # noqa: F401
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
                "E   RuntimeError: The starlette.testclient module "
                "requires the httpx package to be installed.\n"
                "E   $ pip install httpx\n"),
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
    assert result.outputs["classification"] == "missing_dependency"
    assert result.outputs["strategy"] == "install_deps"
    assert result.outputs["can_apply"] is False
    assert result.outputs["missing_modules"] == ["httpx"]
    plan = result.artifact
    assert plan.kind is ArtifactKind.REPAIR_PLAN
    assert plan.content["strategy"] == "install_deps"
    assert plan.content["missing_modules"] == ["httpx"]
    assert plan.content["diffs"] == []
    assert "httpx" in plan.content["rationale"]


def test_repair_verify_pip_installable_missing_module_installs(
        store, tmp_path: Path):
    """A ModuleNotFoundError no project file claims -> install_deps.

    When the pythonpath locator finds nothing on disk for the missing
    name, the venv (not the source tree) is what's incomplete -- the
    repair routes to a package install instead of a doomed regenerate.
    """
    from cgx.session.tasks import repair as _repair_module  # noqa: F401
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
            "stdout": "E   ModuleNotFoundError: No module named 'httpx'\n",
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
    assert result.outputs["classification"] == "missing_dependency"
    assert result.outputs["strategy"] == "install_deps"
    assert result.outputs["missing_modules"] == ["httpx"]


def test_pip_installable_roots_skips_project_names(tmp_path):
    """Names claimed by files/dirs under the project root are excluded."""
    from cgx.session.tasks.repair import _pip_installable_roots
    (tmp_path / "app").mkdir()
    (tmp_path / "util.py").write_text("", encoding="utf-8")
    content = {"stdout": (
        "E   ModuleNotFoundError: No module named 'httpx'\n"
        "E   ModuleNotFoundError: No module named 'app.models'\n"
        "E   ModuleNotFoundError: No module named 'util'\n"
        "E   ModuleNotFoundError: No module named 'httpx.client'\n")}
    assert _pip_installable_roots(tmp_path, content) == ["httpx"]


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


def test_missing_fixture_names_rejects_non_fixture_names():
    """``self``/``cls``/``request`` are never fixtures a repair can add."""
    from cgx.session.repair.classify import missing_fixture_names
    content = {"stdout": (
        "E       fixture 'self' not found\n"
        "E       fixture 'cls' not found\n"
        "E       fixture 'request' not found\n")}
    assert missing_fixture_names(content) == ()
    # A real fixture alongside them still comes through.
    content = {"stdout": ("E       fixture 'self' not found\n"
                          "E       fixture 'client' not found\n")}
    assert missing_fixture_names(content) == ("client",)


def test_classify_does_not_call_self_a_missing_fixture():
    """A collected method reported as ``fixture 'self' not found`` is not
    a missing fixture: classifying it as one ordered a whole-tree
    regenerate to define a fixture that cannot exist."""
    from cgx.session.repair.classify import classify_verify_report
    content = {
        "outcome": "collection_error",
        "returncode": 2,
        "stdout": (
            "tests/test_widget.py:7: in test_widget\n"
            "E       fixture 'self' not found\n"),
        "stderr": "",
    }
    assert classify_verify_report(content) != "missing_fixture"


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


def test_import_name_breaks_keeps_full_dotted_module():
    """import_name_breaks preserves the module; the pin view collapses it."""
    from cgx.session.repair.classify import (
        import_name_breaks,
        third_party_import_breaks,
    )
    content = {
        "outcome": "collection_error",
        "returncode": 2,
        "stdout": (
            "E   ImportError: cannot import name 'login' from 'backend.auth'\n"
            "E   ImportError: cannot import name 'url_quote' from "
            "'werkzeug.urls'\n"),
        "stderr": "",
    }
    assert import_name_breaks(content) == (
        ("login", "backend.auth"), ("url_quote", "werkzeug.urls"))
    # The legacy top-level view the PyPI-pin path consumes is unchanged.
    assert third_party_import_breaks(content) == (
        ("login", "backend"), ("url_quote", "werkzeug"))


def test_repair_first_party_symbol_mismatch_regenerates_not_pins(
        store, tmp_path: Path):
    """A cannot-import-name from an on-disk module -> regenerate, not a pin.

    Mirrors ses_0408ac4084b04b4c: ``backend/auth.py`` imports cleanly but
    never defines ``login``. The pure classifier calls this
    ``third_party_import_break``; REPAIR sees the module resolves on disk and
    re-classifies it to a ``first_party_symbol_mismatch`` regenerate that
    names the missing symbol -- rather than a nonsensical PyPI pin against a
    package called ``backend`` that produces no diff and flaps the loop.
    """
    from cgx.session.tasks.repair import run_repair
    (tmp_path / "backend").mkdir()
    (tmp_path / "backend" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "backend" / "auth.py").write_text(
        "def logout():\n    return True\n", encoding="utf-8")
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    verify_content = {
        "outcome": "collection_error",
        "returncode": 2,
        "stdout": (
            "ImportError while importing test module 'tests/test_auth.py'.\n"
            "tests/test_auth.py:1: in <module>\n"
            "    from backend.auth import login\n"
            "E   ImportError: cannot import name 'login' from "
            "'backend.auth'\n"),
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
    t = TaskNode.new(session.session_id, TaskKind.REPAIR, "repair",
                     inputs={
                         "verify_artifact_id": verify_artifact.artifact_id,
                         "mode": SessionMode.GREENFIELD.value,
                         "repair_attempt": 1,
                     })
    store.save_task(t)
    result = run_repair(
        t, ExecutorDeps(project_root=str(tmp_path), store=store))
    assert result.failure is None
    assert result.outputs["classification"] == "first_party_symbol_mismatch"
    assert result.outputs["strategy"] == "regenerate"
    assert result.outputs["can_apply"] is False
    assert result.artifact.content["diffs"] == []
    constraints = result.outputs["extra_constraints"]
    assert constraints["kind"] == "first_party_symbol_mismatch"
    assert constraints["symbol_mismatches"] == [
        {"symbol": "login", "module": "backend.auth"}]
    # The rationale names the symbol + module and forbids a dependency pin.
    assert "login" in constraints["rationale"]
    assert "backend.auth" in constraints["rationale"]
    assert "do not add or pin" in constraints["rationale"].lower()


def test_repair_select_strategy_first_party_symbol_mismatch_regenerates():
    """The token is regenerate-eligible and carries the symbol mismatches."""
    from cgx.session.tasks.repair import _select_repair_strategy
    strategy, constraints = _select_repair_strategy(
        classification="first_party_symbol_mismatch", diffs=[],
        rationale="define the missing symbol",
        extra_plan_fields={"symbol_mismatches": [
            {"symbol": "login", "module": "backend.auth"}]},
        locations_payload=[])
    assert strategy == "regenerate"
    assert constraints["kind"] == "first_party_symbol_mismatch"
    assert constraints["symbol_mismatches"] == [
        {"symbol": "login", "module": "backend.auth"}]


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


def test_cross_check_flags_phantom_relative_import():
    """``from ..config import X`` where ``config`` was never generated."""
    from cgx.session.scaffold_validate import cross_check_first_party_imports
    contents = {
        "pkg/__init__.py": "",
        "pkg/sub/__init__.py": "",
        "pkg/sub/app.py": "from ..config import API_BASE\n",
    }
    warnings = cross_check_first_party_imports(contents)
    assert len(warnings) == 1
    w = warnings[0]
    assert w["file"] == "pkg/sub/app.py"
    assert w["module"] == "..config"
    assert w["name"] == "API_BASE"


def test_cross_check_flags_relative_import_beyond_top_level():
    """A relative import that walks above the top package -> warning.

    ``_resolve_from_target`` returns ``None`` for a level that exceeds the
    importer's package depth ("attempted relative import beyond top-level
    package" at runtime); the checker must flag it rather than abstain.
    """
    from cgx.session.scaffold_validate import cross_check_first_party_imports
    contents = {"app.py": "from ..config import API_BASE\n"}
    warnings = cross_check_first_party_imports(contents)
    assert [w["name"] for w in warnings] == ["API_BASE"]
    assert warnings[0]["module"] == "..config"


# ------------- #1: work-plan contract enforcement gate -------------

def test_contract_check_passes_when_all_satisfied():
    """Every declared endpoint/schema/function/constant is present -> []."""
    from cgx.session.scaffold_validate import check_contract_compliance
    contracts = {
        "endpoints": [{"method": "GET", "path": "/api/ping"}],
        "schemas": [{"name": "Thing"}],
        "functions": [{"name": "compute", "module": "src/core.py"}],
        "constants": [{"name": "API_BASE"}],
    }
    contents = {
        "src/core.py": (
            "API_BASE = '/api'\n\n"
            "class Thing:\n    pass\n\n"
            "def compute(a, b):\n    return a + b\n"),
        "src/app.py": (
            "@app.route('/api/ping')\ndef ping():\n    return 'ok'\n"),
    }
    assert check_contract_compliance(contents, contracts) == []


def test_contract_check_flags_missing_items():
    """A missing endpoint, schema, function and constant each warn once."""
    from cgx.session.scaffold_validate import check_contract_compliance
    contracts = {
        "endpoints": [{"method": "POST", "path": "/api/orders"}],
        "schemas": [{"name": "Order"}],
        "functions": [{"name": "place_order", "module": "src/core.py"}],
        "constants": [{"name": "MAX_ITEMS"}],
    }
    contents = {"src/core.py": "def unrelated():\n    return 1\n"}
    warnings = check_contract_compliance(contents, contracts)
    kinds = sorted(w["kind"] for w in warnings)
    assert kinds == ["constant", "endpoint", "function", "schema"]
    ep = next(w for w in warnings if w["kind"] == "endpoint")
    assert ep["name"] == "/api/orders" and ep["method"] == "POST"
    fn = next(w for w in warnings if w["kind"] == "function")
    assert fn["module"] == "src/core.py"


def test_contract_check_function_falls_back_to_any_module():
    """A function whose declared module was not generated is sought anywhere."""
    from cgx.session.scaffold_validate import check_contract_compliance
    contracts = {"functions": [
        {"name": "compute", "module": "src/missing.py"}]}
    # ``src/missing.py`` is not generated, so the check falls back to the
    # union of symbols across every generated module -> found, no warning.
    ok = {"src/core.py": "def compute():\n    return 1\n"}
    assert check_contract_compliance(ok, contracts) == []
    bad = {"src/core.py": "def other():\n    return 1\n"}
    warns = check_contract_compliance(bad, contracts)
    assert [w["name"] for w in warns] == ["compute"]


def test_contract_check_abstains_without_python_and_on_empty():
    """No Python files -> symbol checks abstain; empty contracts -> []."""
    from cgx.session.scaffold_validate import check_contract_compliance
    contracts = {
        "endpoints": [{"method": "GET", "path": "/health"}],
        "schemas": [{"name": "Widget"}],
    }
    # A pure JS scaffold: the endpoint scan still runs (path present ->
    # no warning), but the schema symbol check abstains (no AST).
    js = {"src/app.js": "app.get('/health', () => {})\n"}
    assert check_contract_compliance(js, contracts) == []
    # No contracts at all -> nothing to enforce.
    assert check_contract_compliance(js, None) == []
    assert check_contract_compliance(js, {}) == []


def test_contract_check_constant_import_only_does_not_satisfy():
    """A constant that is only *imported* (never assigned) still warns.

    ``from ..config import API_BASE`` re-exports the name into the module's
    namespace, but nothing in the generated tree actually assigns it, so the
    constant contract is unsatisfied and must be flagged for regeneration.
    """
    from cgx.session.scaffold_validate import check_contract_compliance
    contracts = {"constants": [{"name": "API_BASE"}]}
    import_only = {"src/core.py": "from ..config import API_BASE\n"}
    warns = check_contract_compliance(import_only, contracts)
    assert [w["kind"] for w in warns] == ["constant"]
    assert warns[0]["name"] == "API_BASE"
    # An actual assignment satisfies the same contract.
    assigned = {"src/core.py": "API_BASE = 'https://x'\n"}
    assert check_contract_compliance(assigned, contracts) == []


# ------------- P0b: client/server payload coherence gate -------------

def test_payload_coherence_flags_rename_mismatch():
    """The ses_4cbf963cdc67435a bug: client ``operator`` vs server ``operation``.

    The handler reads ``num1/num2/operation`` while the React client POSTs
    ``num1/num2/operator`` -- a rename in both directions -> one ``payload``
    warning naming the client file and the divergent keys.
    """
    from cgx.session.scaffold_validate import (
        check_client_server_payload_coherence,
    )
    contents = {
        "backend/app.py": (
            "@app.route('/calculate', methods=['POST'])\n"
            "def calc():\n"
            "    data = request.json\n"
            "    a = data.get('num1')\n"
            "    b = data.get('num2')\n"
            "    op = data.get('operation')\n"
            "    return {}\n"),
        "src/components/Calculator.jsx": (
            "fetch('/calculate', {\n"
            "  method: 'POST',\n"
            "  body: JSON.stringify({ num1, num2, operator }),\n"
            "})\n"),
    }
    warnings = check_client_server_payload_coherence(contents)
    assert len(warnings) == 1
    w = warnings[0]
    assert w["kind"] == "payload"
    assert w["name"] == "/calculate"
    assert w["file"] == "src/components/Calculator.jsx"
    assert w["server_file"] == "backend/app.py"
    assert "operator" in w["reason"] and "operation" in w["reason"]


def test_payload_coherence_ignores_subset_and_superset():
    """A body that merely omits/adds a field (one-directional) never fires."""
    from cgx.session.scaffold_validate import (
        check_client_server_payload_coherence,
    )
    handler = (
        "@app.route('/calculate', methods=['POST'])\n"
        "def calc():\n"
        "    data = request.json\n"
        "    a = data.get('num1')\n"
        "    b = data.get('num2')\n"
        "    op = data.get('operation')\n"
        "    return {}\n")
    # Exact match -> no warning.
    exact = {
        "backend/app.py": handler,
        "src/App.jsx": (
            "fetch('/calculate', { method: 'POST', body: "
            "JSON.stringify({ num1, num2, operation }) })\n"),
    }
    assert check_client_server_payload_coherence(exact) == []
    # Client omits an optional field (subset) -> still one-directional.
    subset = {
        "backend/app.py": handler,
        "src/App.jsx": (
            "fetch('/calculate', { method: 'POST', body: "
            "JSON.stringify({ num1, num2 }) })\n"),
    }
    assert check_client_server_payload_coherence(subset) == []


def test_payload_coherence_prefers_declared_contract():
    """When the WORK_PLAN declares the endpoint, its request schema wins.

    Even if the handler reads nothing the checker can see, the declared
    ``request`` keys stand in as the authoritative expected shape.
    """
    from cgx.session.scaffold_validate import (
        check_client_server_payload_coherence,
    )
    contracts = {"endpoints": [{
        "method": "POST", "path": "/calculate",
        "request": {"num1": 0, "num2": 0, "operation": ""}}]}
    contents = {
        "backend/app.py": (
            "@app.route('/calculate', methods=['POST'])\n"
            "def calc():\n    return {}\n"),
        "src/App.jsx": (
            "fetch('http://localhost:5000/calculate', { method: 'POST', "
            "body: JSON.stringify({ num1, num2, operator }) })\n"),
    }
    warnings = check_client_server_payload_coherence(contents, contracts)
    assert [w["name"] for w in warnings] == ["/calculate"]
    assert warnings[0]["expected_keys"] == ["num1", "num2", "operation"]


def test_payload_coherence_abstains_without_seam():
    """No backend route or no client fetch body -> nothing to compare."""
    from cgx.session.scaffold_validate import (
        check_client_server_payload_coherence,
    )
    # A route but no client fetch.
    assert check_client_server_payload_coherence({
        "backend/app.py": (
            "@app.route('/calculate', methods=['POST'])\n"
            "def calc():\n    return {}\n")}) == []
    # A fetch but no Python route.
    assert check_client_server_payload_coherence({
        "src/App.jsx": (
            "fetch('/calculate', { method: 'POST', body: "
            "JSON.stringify({ num1, num2 }) })\n")}) == []
    # Empty / non-dict input.
    assert check_client_server_payload_coherence({}) == []
    assert check_client_server_payload_coherence(None) == []


# ------------- P0c: response-contract status coherence ---------------

def test_response_coherence_flags_status_drift():
    """Contract declares 201 but the handler returns an explicit 200."""
    from cgx.session.scaffold_validate import (
        check_response_contract_coherence,
    )
    contracts = {"endpoints": [
        {"method": "POST", "path": "/register", "status": 201}]}
    contents = {
        "backend/app.py": (
            "@app.route('/register', methods=['POST'])\n"
            "def register():\n"
            "    data = request.json\n"
            "    return jsonify({'ok': True}), 200\n"),
    }
    warnings = check_response_contract_coherence(contents, contracts)
    assert len(warnings) == 1
    w = warnings[0]
    assert w["kind"] == "response"
    assert w["name"] == "/register"
    assert w["file"] == "backend/app.py"
    assert w["expected_status"] == 201
    assert w["found_statuses"] == [200]
    assert "201" in w["reason"]


def test_response_coherence_flags_implicit_200_vs_declared_201():
    """A handler with an implicit-200 success path but a declared 201."""
    from cgx.session.scaffold_validate import (
        check_response_contract_coherence,
    )
    contracts = {"endpoints": [
        {"method": "POST", "path": "/register", "status": 201}]}
    contents = {
        "backend/app.py": (
            "@app.route('/register', methods=['POST'])\n"
            "def register():\n"
            "    return jsonify({'ok': True})\n"),
    }
    warnings = check_response_contract_coherence(contents, contracts)
    assert len(warnings) == 1
    assert warnings[0]["expected_status"] == 201
    assert warnings[0]["found_statuses"] == []


def test_response_coherence_abstains_on_match_and_error_only():
    """No warning when the success status matches or cannot be judged."""
    from cgx.session.scaffold_validate import (
        check_response_contract_coherence,
    )
    # Handler returns the declared 201 explicitly -> coherent.
    match = {
        "backend/app.py": (
            "@app.route('/register', methods=['POST'])\n"
            "def register():\n"
            "    return jsonify({'ok': True}), 201\n"),
    }
    contracts_201 = {"endpoints": [
        {"method": "POST", "path": "/register", "status": 201}]}
    assert check_response_contract_coherence(match, contracts_201) == []
    # Implicit-200 success with an error branch, contract declares 200 ->
    # the explicit 400 is not a success code, so nothing to flag.
    implicit = {
        "backend/app.py": (
            "@app.route('/login', methods=['POST'])\n"
            "def login():\n"
            "    if not ok:\n"
            "        return jsonify({'error': 'bad'}), 400\n"
            "    return jsonify({'token': t})\n"),
    }
    contracts_200 = {"endpoints": [
        {"method": "POST", "path": "/login", "status": 200}]}
    assert check_response_contract_coherence(implicit, contracts_200) == []
    # No contract, no status, or no Python route -> abstain.
    assert check_response_contract_coherence(match, None) == []
    assert check_response_contract_coherence(match, {"endpoints": [
        {"method": "POST", "path": "/register"}]}) == []
    assert check_response_contract_coherence(
        {"src/App.jsx": "fetch('/register')\n"}, contracts_201) == []


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


def test_scaffold_executor_reconciles_import_warnings(store, monkeypatch):
    """#2: the coherence pass regenerates an importer so imports resolve."""
    from cgx.session.tasks.scaffold import run_scaffold

    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    plan = Artifact.new(
        session.session_id, "task_x", ArtifactKind.WORK_PLAN, {
            "prior_goal": "g", "composed_goal": "build api", "answers": {},
            "plan_md": "",
            "layers": [{"name": "app", "files": [
                {"path": "backend/auth.py", "description": "auth"},
                {"path": "backend/app.py", "description": "entry"}]}],
        })
    store.save_artifact(plan)

    calls = {"backend/app.py": 0}

    def fake_generate(path, description, provider, *, goal="", **kw):
        if path == "backend/auth.py":
            body = "def login():\n    return True\n"
        else:
            calls[path] += 1
            # First generation imports a symbol that does not exist; the
            # coherence regeneration (second call, carrying the mismatch
            # constraint) aligns to the real symbol.
            body = ("from backend.auth import logout\n" if calls[path] == 1
                    else "from backend.auth import login\n")
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
    assert result.outputs["reconciled_count"] == 1
    # After reconciliation the import resolves -> no residual warnings, and
    # the persisted diff carries the corrected import.
    assert result.artifact.content["import_warnings"] == []
    assert result.outputs["import_warnings_count"] == 0
    app_diff = next(d for d in result.artifact.content["diffs"]
                    if d["file"] == "backend/app.py")
    assert "import login" in app_diff["patch"]
    assert calls["backend/app.py"] == 2


def test_scaffold_coherence_noop_when_tree_resolves(store, monkeypatch):
    """#2: a tree whose imports already resolve is left untouched."""
    from cgx.session.tasks.scaffold import run_scaffold

    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    plan = Artifact.new(
        session.session_id, "task_x", ArtifactKind.WORK_PLAN, {
            "prior_goal": "g", "composed_goal": "build api", "answers": {},
            "plan_md": "",
            "layers": [{"name": "app", "files": [
                {"path": "backend/auth.py", "description": "auth"},
                {"path": "backend/app.py", "description": "entry"}]}],
        })
    store.save_artifact(plan)

    contents = {
        "backend/auth.py": "def login():\n    return True\n",
        "backend/app.py": "from backend.auth import login\n",
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
    assert result.outputs["reconciled_count"] == 0
    assert result.artifact.content["import_warnings"] == []


def test_scaffold_executor_surfaces_contract_warnings(store, monkeypatch):
    """SCAFFOLD enforces the WORK_PLAN contracts and surfaces mismatches."""
    from cgx.session.tasks.scaffold import run_scaffold

    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    contracts = {
        "endpoints": [{"method": "GET", "path": "/api/orders"}],
        "functions": [{"name": "place_order", "module": "backend/core.py"}],
    }
    plan = Artifact.new(
        session.session_id, "task_x", ArtifactKind.WORK_PLAN, {
            "prior_goal": "g",
            "composed_goal": "build api",
            "answers": {},
            "plan_md": "",
            "contracts": contracts,
            "layers": [{"name": "app", "files": [
                {"path": "backend/core.py", "description": "core"}]}],
        })
    store.save_artifact(plan)

    # The generated file honours neither the endpoint nor the function.
    contents = {"backend/core.py": "def unrelated():\n    return 1\n"}

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
    warns = result.artifact.content["contract_warnings"]
    assert sorted(w["kind"] for w in warns) == ["endpoint", "function"]
    assert result.outputs["contract_warnings_count"] == 2


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
    # Absent provider usage, tokens are estimated and cost is unknown.
    assert fact.content["token_source"] == "estimated"
    assert fact.content["cost_source"] == "unknown"
    assert fact.content["tokens_total"] >= 1
    # Every call carries a prompt version fingerprint (provenance join key);
    # run_id is None outside a run context.
    assert fact.content["prompt_version"]
    assert fact.content["run_id"] is None
    # drain clears the buffer.
    assert tp.drain() == []


def test_tracing_provider_stamps_run_id_from_trace_context():
    """A run_id on the trace context is stamped onto the LLM_CALL fact."""
    from cgx.registry import fingerprint
    from cgx.session.llm_trace import TracingProvider
    from cgx.trace import reset_trace_context, set_trace_context

    tp = TracingProvider(_StubChatProvider(model="m", reply="hi"))
    tp.bind("sess-A", "task-X")
    token = set_trace_context(run_id="run_abc123")
    try:
        tp.chat([{"role": "user", "content": "ping"}])
    finally:
        reset_trace_context(token)
    fact = tp.drain()[0]
    assert fact.content["run_id"] == "run_abc123"
    # The recorded version is the content fingerprint of the flattened prompt.
    assert fact.content["prompt_version"] == fingerprint("[user]\nping")


def test_tracing_provider_records_provider_usage_and_metrics():
    """Provider-reported usage is stored truthfully and mirrored to metrics."""
    from cgx import metrics as _metrics
    from cgx.session.llm_trace import TracingProvider

    class _UsageProvider(_StubChatProvider):
        def chat(self, messages, **kw):
            return {"content": "ok", "provider": "gemini",
                    "raw": {"usageMetadata": {"promptTokenCount": 100,
                                              "candidatesTokenCount": 20}}}

    _metrics.reset_for_tests()
    tp = TracingProvider(_UsageProvider(model="gemini-2.5-flash"))
    tp.bind("s", "t")
    tp.chat([{"role": "user", "content": "hi"}])
    fact = tp.drain()[0]
    assert fact.content["provider"] == "gemini"
    assert fact.content["token_source"] == "provider"
    assert fact.content["tokens_in"] == 100
    assert fact.content["tokens_out"] == 20
    # gemini-2.5-flash is in the default price table -> a real cost estimate.
    assert fact.content["cost_source"] == "default"
    assert fact.content["cost_usd"] > 0
    rendered = _metrics.render_prometheus()
    assert "cgx_llm_calls_total" in rendered
    assert "cgx_llm_tokens_total" in rendered
    assert "cgx_llm_call_latency_ms_count" in rendered
    _metrics.reset_for_tests()


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


def test_select_repair_strategy_assertion_drift_folds_target_files():
    """assertion_drift with named impl file(s) -> targeted regenerate."""
    from cgx.session.tasks.repair import _select_repair_strategy
    strategy, constraints = _select_repair_strategy(
        classification="assertion_drift", diffs=[],
        rationale="align handler to asserted contract",
        extra_plan_fields={"target_files": ["src/handlers.py"]},
        locations_payload=[])
    assert strategy == "regenerate"
    assert constraints["kind"] == "assertion_drift"
    assert constraints["target_files"] == ["src/handlers.py"]


def test_select_repair_strategy_assertion_drift_without_targets_whole_tree():
    """assertion_drift with no named impl file -> whole-tree regenerate."""
    from cgx.session.tasks.repair import _select_repair_strategy
    strategy, constraints = _select_repair_strategy(
        classification="assertion_drift", diffs=[], rationale="r",
        extra_plan_fields={}, locations_payload=[])
    assert strategy == "regenerate"
    assert "target_files" not in constraints


def test_assertion_impl_targets_excludes_test_files(tmp_path: Path):
    """Traceback source files minus the test modules -> impl targets."""
    from cgx.session.tasks.repair import _assertion_impl_targets
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "handlers.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "tests" / "test_backend.py").write_text(
        "x = 1\n", encoding="utf-8")
    content = {
        "outcome": "assertions_failed",
        "returncode": 1,
        "stdout": (
            "tests/test_backend.py:5: in test_login\n"
            "src/handlers.py:2: in login\n"
            "E   assert 200 == 201"),
        "stderr": "",
    }
    assert _assertion_impl_targets(content, tmp_path) == ["src/handlers.py"]


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
        extra_plan_fields={"missing_fixtures": ["client", "db"]},
        locations_payload=[])
    assert strategy == "regenerate"
    assert constraints["kind"] == "missing_fixture"
    # The names pytest reported ride along as a structured field so the
    # regenerated SCAFFOLD is told exactly which fixtures to author.
    assert constraints["missing_fixtures"] == ["client", "db"]


def test_fixture_rationale_instructs_authoring_when_no_definition():
    """No on-disk fixture -> the rationale is an actionable authoring order.

    Regression for the E2E halt where a model-authored Flask test requested
    a ``client`` fixture that no @pytest.fixture defined. The old rationale
    only *described* the gap, so the weak model re-emitted the same test.
    The rationale must now instruct the regenerate to author the fixture,
    add a conftest.py for shared fixtures, and build a web client fixture
    from the app's test client -- naming the missing fixture.
    """
    from cgx.session.tasks.repair import _fixture_rationale
    content = {"stdout": "E       fixture 'client' not found\n"}
    text = _fixture_rationale(content, [], has_diff=False).lower()
    assert "client" in text
    assert "conftest" in text
    assert "test_client" in text
    # It is an instruction to author, not merely a diagnosis.
    assert "author" in text


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


def test_propose_regenerate_carries_prior_failure_signatures():
    """The flap ledger survives the regenerate (merged and deduped)."""
    from cgx.session.repair.propose import propose_regenerate
    scaffold = TaskNode.new(
        "sess-A", TaskKind.SCAFFOLD, "scaffold",
        inputs={"work_plan_artifact_id": "art_plan",
                "prior_failure_signatures": ["sig-old"]})
    fresh = propose_regenerate(
        scaffold, {"kind": "missing_dependency", "rationale": "r"},
        prior_failure_signatures=["sig-old", "sig-new"])
    assert fresh.inputs["prior_failure_signatures"] == ["sig-old", "sig-new"]
    # Omitting the ledger leaves the cloned inputs untouched.
    fresh2 = propose_regenerate(
        scaffold, {"kind": "missing_dependency", "rationale": "r"})
    assert fresh2.inputs["prior_failure_signatures"] == ["sig-old"]
    # Original task's inputs are unaffected (no aliasing).
    assert scaffold.inputs["prior_failure_signatures"] == ["sig-old"]


def test_propose_regenerate_accumulates_additional_files():
    """additional_files survive and merge across regenerate attempts."""
    from cgx.session.repair.propose import propose_regenerate
    scaffold = TaskNode.new(
        "sess-A", TaskKind.SCAFFOLD, "scaffold",
        inputs={"work_plan_artifact_id": "art_plan"})
    first = propose_regenerate(
        scaffold, {"kind": "missing_entry_module", "rationale": "r"},
        additional_files=[{"path": "index.html", "description": "entry html"}])
    assert first.inputs["additional_files"] == [
        {"path": "index.html", "description": "entry html"}]
    # A second attempt that names a different file keeps both, and a
    # repeat of the first does not duplicate it.
    second = propose_regenerate(
        first, {"kind": "missing_entry_module", "rationale": "r"},
        additional_files=[{"path": "index.html", "description": "again"},
                          {"path": "src/main.jsx", "description": "entry js"}])
    assert [f["path"] for f in second.inputs["additional_files"]] == [
        "index.html", "src/main.jsx"]
    assert second.inputs["additional_files"][0]["description"] == "entry html"
    # No additional files -> the key is never introduced.
    plain = propose_regenerate(scaffold, {"kind": "unknown", "rationale": "r"})
    assert "additional_files" not in plain.inputs


def test_scaffold_with_additional_files_extends_last_layer():
    """Regenerate-added paths land in the manifest without mutating the plan."""
    from cgx.session.tasks.scaffold import _with_additional_files
    layers = [{"name": "ui", "files": [{"path": "src/main.jsx",
                                        "description": "entry"}]},
              {"name": "config", "files": [{"path": "vite.config.js",
                                            "description": "config"}]}]
    task = TaskNode.new(
        "sess-A", TaskKind.SCAFFOLD, "scaffold",
        inputs={"additional_files": [
            {"path": "index.html", "description": "entry html"},
            {"path": "src/main.jsx", "description": "already planned"}]})
    out, added = _with_additional_files(task, layers)
    assert added == ["index.html"]
    assert [f["path"] for f in out[1]["files"]] == ["vite.config.js",
                                                    "index.html"]
    # The caller's layers (and the stored work plan they came from) are
    # left untouched.
    assert [f["path"] for f in layers[1]["files"]] == ["vite.config.js"]


def test_scaffold_with_additional_files_noop_without_marker():
    from cgx.session.tasks.scaffold import _with_additional_files
    layers = [{"name": "ui", "files": [{"path": "src/main.jsx",
                                        "description": "entry"}]}]
    task = TaskNode.new("sess-A", TaskKind.SCAFFOLD, "scaffold", inputs={})
    out, added = _with_additional_files(task, layers)
    assert added == []
    assert out is layers


def _build_regenerate_chain(*, prior_regens: int = 0,
                            prior_repair_regens: int = 0,
                            extra_descendants: bool = True):
    """Build a SCAFFOLD -> APPLY -> VERIFY -> REPAIR(regenerate) chain."""
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    scaffold = TaskNode.new(
        session.session_id, TaskKind.SCAFFOLD, "scaffold",
        inputs={"work_plan_artifact_id": "art_plan",
                "regenerate_attempt": prior_regens,
                "repair_regenerate_attempt": prior_repair_regens})
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
    # A semantic-repair regenerate bumps its OWN counter, not just the
    # shared syntax-churn one, so the loop stays bounded across scaffolds.
    assert new_scaffold.inputs["repair_regenerate_attempt"] == 1
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


def test_router_repair_regenerate_threads_missing_files():
    """A missing-entry-module verdict grows the next SCAFFOLD's manifest."""
    session, _scaffold, tasks, rep = _build_regenerate_chain()
    rep.outputs = dict(rep.outputs)
    rep.outputs["classification"] = "smoke_import_failure"
    rep.outputs["extra_constraints"] = {
        "kind": "missing_entry_module",
        "rationale": "entry module absent",
        "missing_files": [{"path": "index.html",
                           "description": "entry html"}],
    }
    plan = Router().on_task_completed(
        session=session, completed=rep, tasks=tasks)
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    assert creates[0].task.inputs["additional_files"] == [
        {"path": "index.html", "description": "entry html"}]


def test_router_repair_regenerate_targets_build_smoke_importer():
    """A build-resolution verdict regenerates only the named importer."""
    session, scaffold, tasks, rep = _build_regenerate_chain()
    rep.outputs = dict(rep.outputs)
    rep.outputs["classification"] = "smoke_import_failure"
    rep.outputs["scaffold_artifact_id"] = "art_scaf"
    rep.outputs["extra_constraints"] = {
        "kind": "invalid_build_smoke",
        "rationale": "resolve",
        "build_error": "Could not resolve './index.css' from 'src/main.jsx'",
        "target_files": ["src/main.jsx"],
    }
    plan = Router().on_task_completed(
        session=session, completed=rep, tasks=tasks)
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    new_scaffold = creates[0].task
    assert new_scaffold.inputs["regenerate_files"] == ["src/main.jsx"]
    assert new_scaffold.inputs["prior_scaffold_artifact_id"] == "art_scaf"


def test_router_repair_regenerate_reuses_diagnosed_target_files():
    """C1: a diagnosed REPAIR that falls back to regenerate stays file-scoped.

    A REPAIR reached from a DIAGNOSE ``patch_files`` verdict carries the
    diagnosed implicated file(s) in its inputs. When the bounded patch is a
    no-op and the executor emits ``strategy=regenerate`` *without* the
    classifier naming ``target_files``, the router must reuse the diagnosed
    ``target_files`` for a scoped regenerate instead of nuking the tree.
    """
    session, _scaffold, tasks, rep = _build_regenerate_chain()
    rep.inputs = dict(rep.inputs)
    rep.inputs["target_files"] = ["src/handlers.py"]
    rep.outputs = dict(rep.outputs)
    rep.outputs["classification"] = "assertion_drift"
    rep.outputs["scaffold_artifact_id"] = "art_scaf"
    # The classifier named no file this round (whole-tree today).
    rep.outputs["extra_constraints"] = {"kind": "assertion_drift",
                                        "rationale": "drift"}
    plan = Router().on_task_completed(
        session=session, completed=rep, tasks=tasks)
    new_scaffold = [a.task for a in plan.actions
                    if isinstance(a, CreateTask)][0]
    assert new_scaffold.inputs["regenerate_files"] == ["src/handlers.py"]
    assert new_scaffold.inputs["prior_scaffold_artifact_id"] == "art_scaf"


def test_router_repair_regenerate_whole_tree_when_nothing_named():
    """Whole-tree stays the fallback when neither classifier nor DIAGNOSE
    named a file (no diagnosed inputs, no classifier target_files)."""
    session, _scaffold, tasks, rep = _build_regenerate_chain()
    rep.outputs = dict(rep.outputs)
    rep.outputs["classification"] = "unknown"
    rep.outputs["scaffold_artifact_id"] = "art_scaf"
    rep.outputs["extra_constraints"] = {"kind": "unknown"}
    plan = Router().on_task_completed(
        session=session, completed=rep, tasks=tasks)
    new_scaffold = [a.task for a in plan.actions
                    if isinstance(a, CreateTask)][0]
    assert not new_scaffold.inputs.get("regenerate_files")
    assert not new_scaffold.inputs.get("prior_scaffold_artifact_id")


def test_router_repair_regenerate_preserves_failure_signatures():
    """The failed chain's flap ledger survives into the new SCAFFOLD.

    Without this, a regenerate that reproduces the identical failure
    signature restarts with an empty ledger and the loop burns the whole
    regenerate budget on a fix that cannot work (live: ses_612f8d1c).
    """
    session, _scaffold, tasks, rep = _build_regenerate_chain()
    rep.inputs["prior_failure_signatures"] = ["sig-a", "sig-b"]
    plan = Router().on_task_completed(
        session=session, completed=rep, tasks=tasks)
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    new_scaffold = creates[0].task
    assert new_scaffold.kind is TaskKind.SCAFFOLD
    assert new_scaffold.inputs["prior_failure_signatures"] == [
        "sig-a", "sig-b"]


def test_actionable_contract_warnings_filters_endpoints_and_unattributed():
    from cgx.session.greenfield_edges import _actionable_contract_warnings
    got = _actionable_contract_warnings({"contract_warnings": [
        {"kind": "endpoint", "name": "/api/x", "method": "POST"},
        {"kind": "constant", "name": "API_BASE"},  # no module -> skip
        {"kind": "function", "name": "compute", "module": "src/core.py"},
        {"kind": "schema", "name": "User", "module": "src/db.py"},
        "not-a-dict",
    ]})
    assert {w["name"] for w in got} == {"compute", "User"}


def _make_clean_scaffold_with_contracts(warnings, *, prior_regens=0,
                                        failed_count=0):
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    scaffold = TaskNode.new(
        session.session_id, TaskKind.SCAFFOLD, "scaffold",
        inputs={"work_plan_artifact_id": "art_plan",
                "regenerate_attempt": prior_regens})
    scaffold.status = TaskNodeStatus.DONE
    scaffold.outputs = {"scaffold_artifact_id": "art_s",
                        "generated_count": 5, "failed_count": failed_count,
                        "contract_warnings": warnings}
    return session, scaffold


def test_router_scaffold_unmet_contract_regenerates_within_budget():
    # A clean scaffold whose named module lacks a declared function
    # regenerates with the unmet contract folded in (early return, so the
    # only created task is the fresh SCAFFOLD -- not the APPLY successor).
    session, scaffold = _make_clean_scaffold_with_contracts(
        [{"kind": "function", "name": "compute", "module": "src/core.py"}])
    plan = Router().on_task_completed(
        session=session, completed=scaffold, tasks=[scaffold])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    new_scaffold = creates[0].task
    assert new_scaffold.kind is TaskKind.SCAFFOLD
    assert new_scaffold.inputs["regenerate_attempt"] == 1
    payloads = new_scaffold.inputs["regenerate_constraints"]
    assert payloads and payloads[0]["kind"] == "unmet_contract"
    assert "compute" in payloads[0]["rationale"]


def test_router_scaffold_placeholder_endpoint_does_not_regenerate():
    # The exact shape from ses_fc44ba67d1cc4835: placeholder endpoints and
    # unattributed constants must NOT force a regenerate.
    from cgx.session.greenfield_edges import (
        _scaffold_contract_regenerate_actions,
    )
    _session, scaffold = _make_clean_scaffold_with_contracts([
        {"kind": "endpoint", "name": "/api/x", "method": "POST"},
        {"kind": "endpoint", "name": "/api/protected", "method": "GET"},
        {"kind": "constant", "name": "API_BASE"},
        {"kind": "constant", "name": "JWT_SECRET_KEY"},
    ])
    assert _scaffold_contract_regenerate_actions(scaffold, [scaffold]) == []


def test_router_scaffold_unmet_contract_non_terminal_when_budget_spent():
    # Budget spent -> helper returns nothing (non-terminal): the caller
    # then takes the normal SCAFFOLD -> APPLY edge instead of failing.
    from cgx.session.budget import REGENERATE_BUDGET
    from cgx.session.greenfield_edges import (
        _scaffold_contract_regenerate_actions,
    )
    _session, scaffold = _make_clean_scaffold_with_contracts(
        [{"kind": "function", "name": "compute", "module": "src/core.py"}],
        prior_regens=REGENERATE_BUDGET)
    assert _scaffold_contract_regenerate_actions(scaffold, [scaffold]) == []


def test_router_scaffold_unmet_contract_repeat_is_non_terminal():
    """The same contracts unmet twice ends the loop, budget or not.

    Round-trips the ledger the first regenerate records, so the test
    cannot drift from how the signature is derived.
    """
    from cgx.session.greenfield_edges import (
        _scaffold_contract_regenerate_actions,
    )
    _session, scaffold = _make_clean_scaffold_with_contracts(
        [{"kind": "function", "name": "compute", "module": "src/core.py"}])
    first = _scaffold_contract_regenerate_actions(scaffold, [scaffold])
    retry = [a for a in first if isinstance(a, CreateTask)][0].task
    signatures = retry.inputs["prior_failure_signatures"]
    assert signatures
    scaffold.inputs["prior_failure_signatures"] = signatures
    assert _scaffold_contract_regenerate_actions(scaffold, [scaffold]) == []


def test_router_scaffold_contract_skipped_when_files_dropped():
    # failed_count > 0 belongs to the dropped-files path; the contract
    # path must defer even if contract warnings are also present.
    from cgx.session.greenfield_edges import (
        _scaffold_contract_regenerate_actions,
    )
    _session, scaffold = _make_clean_scaffold_with_contracts(
        [{"kind": "function", "name": "compute", "module": "src/core.py"}],
        failed_count=1)
    assert _scaffold_contract_regenerate_actions(scaffold, [scaffold]) == []


# ------------- skill-structural regenerate gate -------------

def _make_clean_scaffold_with_skill_verdict(verdict, *, prior_regens=0,
                                            failed_count=0):
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    scaffold = TaskNode.new(
        session.session_id, TaskKind.SCAFFOLD, "scaffold",
        inputs={"work_plan_artifact_id": "art_plan",
                "regenerate_attempt": prior_regens})
    scaffold.status = TaskNodeStatus.DONE
    scaffold.outputs = {"scaffold_artifact_id": "art_s",
                        "generated_count": 5, "failed_count": failed_count,
                        "skill_verdict": verdict}
    return session, scaffold


def test_router_scaffold_skill_verdict_regenerates_within_budget():
    # A clean scaffold that recorded a fatal skill verdict regenerates with
    # the unmet-skill rationale folded in (early return, so the only created
    # task is the fresh SCAFFOLD -- not the APPLY successor).
    session, scaffold = _make_clean_scaffold_with_skill_verdict(
        {"skill": "react", "confidence": 0.9,
         "rationale": "scaffold has no .jsx/.tsx/.js/.ts files."})
    plan = Router().on_task_completed(
        session=session, completed=scaffold, tasks=[scaffold])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    new_scaffold = creates[0].task
    assert new_scaffold.kind is TaskKind.SCAFFOLD
    assert new_scaffold.inputs["regenerate_attempt"] == 1
    payloads = new_scaffold.inputs["regenerate_constraints"]
    assert payloads and payloads[-1]["kind"] == "unmet_skill"
    assert payloads[-1]["skill"] == "react"
    assert ".jsx" in payloads[-1]["rationale"]


def test_react_skill_python_only_scaffold_regenerates_via_router():
    # End-to-end: the real react skill judges a Python-only scaffold fatal,
    # and that verdict drives the router's whole-tree regenerate -- the
    # promise that a React goal never silently passes a Python-only output.
    from skills import detect_skills, validate_scaffold
    active = detect_skills("build a react app for tracking tasks")
    assert any(s.name == "react" for s in active)
    diffs = [{"file": "app.py", "patch": "print('hello')"},
             {"file": "requirements.txt", "patch": "flask\n"}]
    verdict = validate_scaffold(active, diffs, goal="build a react app")
    assert verdict is not None and verdict.skill == "react"

    session, scaffold = _make_clean_scaffold_with_skill_verdict(
        {"skill": verdict.skill, "confidence": verdict.confidence,
         "rationale": verdict.rationale})
    plan = Router().on_task_completed(
        session=session, completed=scaffold, tasks=[scaffold])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    assert creates[0].task.kind is TaskKind.SCAFFOLD
    payloads = creates[0].task.inputs["regenerate_constraints"]
    assert payloads[-1]["kind"] == "unmet_skill"


def test_router_scaffold_skill_skipped_when_files_dropped():
    # failed_count > 0 belongs to the dropped-files path; the skill gate
    # defers so the two regenerates never race on the same scaffold.
    from cgx.session.greenfield_edges import (
        _scaffold_skill_regenerate_actions,
    )
    _session, scaffold = _make_clean_scaffold_with_skill_verdict(
        {"skill": "react", "confidence": 0.9, "rationale": "no js"},
        failed_count=1)
    assert _scaffold_skill_regenerate_actions(scaffold, [scaffold]) == []


def test_router_scaffold_skill_non_terminal_when_budget_spent():
    # Budget spent -> helper returns nothing (non-terminal): the caller then
    # takes the normal SCAFFOLD -> APPLY edge instead of failing the session.
    from cgx.session.budget import REGENERATE_BUDGET
    from cgx.session.greenfield_edges import (
        _scaffold_skill_regenerate_actions,
    )
    _session, scaffold = _make_clean_scaffold_with_skill_verdict(
        {"skill": "react", "confidence": 0.9, "rationale": "no js"},
        prior_regens=REGENERATE_BUDGET)
    assert _scaffold_skill_regenerate_actions(scaffold, [scaffold]) == []


def test_router_scaffold_skill_repeat_is_non_terminal():
    """The same skill verdict twice stops the loop (flap backstop)."""
    from cgx.session.greenfield_edges import (
        _scaffold_skill_regenerate_actions,
    )
    _session, scaffold = _make_clean_scaffold_with_skill_verdict(
        {"skill": "react", "confidence": 0.9, "rationale": "no js"})
    first = _scaffold_skill_regenerate_actions(scaffold, [scaffold])
    retry = [a for a in first if isinstance(a, CreateTask)][0].task
    signatures = retry.inputs["prior_failure_signatures"]
    assert signatures
    scaffold.inputs["prior_failure_signatures"] = signatures
    assert _scaffold_skill_regenerate_actions(scaffold, [scaffold]) == []


def test_router_scaffold_no_skill_verdict_takes_normal_edge():
    # No verdict recorded -> the skill gate declines (empty), so the caller
    # falls through to the normal SCAFFOLD -> APPLY successor.
    from cgx.session.greenfield_edges import (
        _scaffold_skill_regenerate_actions,
    )
    _session, scaffold = _make_clean_scaffold_with_skill_verdict(None)
    assert _scaffold_skill_regenerate_actions(scaffold, [scaffold]) == []


# ------------- P0b: payload-mismatch targeted regenerate -------------

def _payload_warning(file="src/components/Calculator.jsx"):
    return {"kind": "payload", "name": "/calculate", "file": file,
            "server_file": "backend/app.py",
            "client_keys": ["num1", "num2", "operator"],
            "expected_keys": ["num1", "num2", "operation"],
            "reason": "client sends operator but endpoint expects operation"}


def test_actionable_payload_warnings_requires_client_file():
    from cgx.session.greenfield_edges import _actionable_payload_warnings
    got = _actionable_payload_warnings({"contract_warnings": [
        _payload_warning(),
        {"kind": "payload", "name": "/x"},  # no file -> skip
        {"kind": "function", "name": "compute", "module": "src/core.py"},
        "not-a-dict",
    ]})
    assert [w["file"] for w in got] == ["src/components/Calculator.jsx"]


def test_router_scaffold_payload_mismatch_targeted_regenerate():
    # A payload rename regenerates ONLY the offending client file against
    # the prior scaffold artifact, with the mismatch folded in.
    from cgx.session.greenfield_edges import (
        _scaffold_payload_regenerate_actions,
    )
    _session, scaffold = _make_clean_scaffold_with_contracts(
        [_payload_warning()])
    actions = _scaffold_payload_regenerate_actions(scaffold, [scaffold])
    creates = [a for a in actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    new_scaffold = creates[0].task
    assert new_scaffold.kind is TaskKind.SCAFFOLD
    assert new_scaffold.inputs["regenerate_files"] == [
        "src/components/Calculator.jsx"]
    assert new_scaffold.inputs["prior_scaffold_artifact_id"] == "art_s"
    constraint = new_scaffold.inputs["regenerate_constraints"][-1]
    assert constraint["kind"] == "payload_mismatch"
    assert "operation" in constraint["rationale"]


def test_router_scaffold_payload_skipped_when_files_dropped():
    # failed_count > 0 belongs to the dropped-files path; the payload path
    # defers so the two regenerates never race on the same scaffold.
    from cgx.session.greenfield_edges import (
        _scaffold_payload_regenerate_actions,
    )
    _session, scaffold = _make_clean_scaffold_with_contracts(
        [_payload_warning()], failed_count=1)
    assert _scaffold_payload_regenerate_actions(scaffold, [scaffold]) == []


def test_router_scaffold_payload_non_terminal_when_budget_spent():
    from cgx.session.budget import REGENERATE_BUDGET
    from cgx.session.greenfield_edges import (
        _scaffold_payload_regenerate_actions,
    )
    _session, scaffold = _make_clean_scaffold_with_contracts(
        [_payload_warning()], prior_regens=REGENERATE_BUDGET)
    assert _scaffold_payload_regenerate_actions(scaffold, [scaffold]) == []


def test_router_scaffold_payload_repeat_is_non_terminal():
    """The same payload mismatch twice stops the loop (flap backstop)."""
    from cgx.session.greenfield_edges import (
        _scaffold_payload_regenerate_actions,
    )
    _session, scaffold = _make_clean_scaffold_with_contracts(
        [_payload_warning()])
    first = _scaffold_payload_regenerate_actions(scaffold, [scaffold])
    retry = [a for a in first if isinstance(a, CreateTask)][0].task
    signatures = retry.inputs["prior_failure_signatures"]
    assert signatures
    scaffold.inputs["prior_failure_signatures"] = signatures
    assert _scaffold_payload_regenerate_actions(scaffold, [scaffold]) == []


def _response_warning(file="backend/app.py"):
    return {"kind": "response", "name": "/register", "file": file,
            "expected_status": 201, "found_statuses": [200],
            "reason": "handler returns 200 but the contract declares 201"}


def test_actionable_payload_warnings_accepts_response_kind():
    """The seam edge acts on both payload (client) and response (server)."""
    from cgx.session.greenfield_edges import _actionable_payload_warnings
    got = _actionable_payload_warnings({"contract_warnings": [
        _payload_warning(),
        _response_warning(),
        {"kind": "response", "name": "/x"},  # no file -> skip
    ]})
    assert [w["file"] for w in got] == [
        "src/components/Calculator.jsx", "backend/app.py"]


def test_router_scaffold_response_mismatch_targeted_regenerate():
    # A response-status drift regenerates ONLY the offending handler file
    # against the prior scaffold artifact, with the mismatch folded in.
    from cgx.session.greenfield_edges import (
        _scaffold_payload_regenerate_actions,
    )
    _session, scaffold = _make_clean_scaffold_with_contracts(
        [_response_warning()])
    actions = _scaffold_payload_regenerate_actions(scaffold, [scaffold])
    creates = [a for a in actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    new_scaffold = creates[0].task
    assert new_scaffold.kind is TaskKind.SCAFFOLD
    assert new_scaffold.inputs["regenerate_files"] == ["backend/app.py"]
    assert new_scaffold.inputs["prior_scaffold_artifact_id"] == "art_s"
    constraint = new_scaffold.inputs["regenerate_constraints"][-1]
    assert constraint["kind"] == "payload_mismatch"
    assert "201" in constraint["rationale"]


def test_router_repair_regenerate_budget_exhausted_fails_session():
    """Once the repair-regenerate budget is hit the session fails terminally."""
    from cgx.session.budget import REPAIR_REGENERATE_BUDGET
    session, _scaffold, tasks, rep = _build_regenerate_chain(
        prior_repair_regens=REPAIR_REGENERATE_BUDGET)
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


def test_router_repair_regenerate_allowed_after_syntax_budget_spent():
    """The ses_7c8b181873844f06 shape: a scaffold that spent its whole
    syntax-churn budget converging to a clean tree must still afford the
    FIRST semantic (api_check-driven) regenerate.
    """
    from cgx.session.budget import (REGENERATE_BUDGET,
                                    REPAIR_REGENERATE_BUDGET)
    assert REPAIR_REGENERATE_BUDGET >= 1
    session, scaffold, tasks, rep = _build_regenerate_chain(
        prior_regens=REGENERATE_BUDGET, prior_repair_regens=0)
    plan = Router().on_task_completed(
        session=session, completed=rep, tasks=tasks)
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    # The syntax budget being spent must NOT block the semantic repair.
    assert len(creates) == 1
    new_scaffold = creates[0].task
    assert new_scaffold.kind is TaskKind.SCAFFOLD
    assert new_scaffold.inputs["repair_regenerate_attempt"] == 1
    # No terminal FAILED -- the run gets its correctness rewrite.
    assert not [a for a in plan.actions
                if isinstance(a, UpdateSessionStatus)
                and a.status is SessionStatus.FAILED]


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
                              survivors: int = 8,
                              apply_failed_files=None):
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
    if apply_failed_files is None:
        apply_failed_files = [
            {"file": "backend/models.py",
             "error": "python syntax: unexpected unindent (models.py, line 10)"},
            {"file": "tests/test_auth.py",
             "error": "python syntax: unexpected unindent (test_auth.py, "
                      "line 19)"}]
    apply_t.outputs = {
        "apply_artifact_id": "art_applied",
        "applied_count": 1, "failed_count": len(apply_failed_files),
        "failed_files": apply_failed_files}
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
    # The retry carries the SCAFFOLD+APPLY failure signature so a repeat
    # of the identical drop is detectable on the next round.
    assert new_scaffold.inputs["prior_failure_signatures"]


def test_is_foundational_path_recognises_environment_manifests():
    from cgx.session.greenfield_edges import (
        _dropped_foundational_files,
        _is_foundational_path,
    )
    for p in ("requirements.txt", "requirements-dev.txt", "backend/package.json",
              "pyproject.toml", "setup.py", "setup.cfg"):
        assert _is_foundational_path(p), p
    for p in ("app.py", "src/components/App.jsx", "README.md", "tests/test_x.py"):
        assert not _is_foundational_path(p), p
    # De-duplicated, drawn from both dropped sources.
    got = _dropped_foundational_files(
        [{"file": "backend/main.py"}, {"file": "package.json"}],
        [{"file": "requirements.txt"}])
    assert set(got) == {"package.json", "requirements.txt"}


def test_router_apply_dropped_foundational_file_escalates_to_replan():
    """A dropped environment manifest bypasses the per-file regenerate loop
    (even with budget to spare) and re-plans the manifest instead."""
    session, scaffold, apply_t, tasks = _build_apply_failed_chain(
        apply_failed_files=[
            {"file": "package.json",
             "error": "JSON parse error: Expecting value: line 1"}])
    plan = Router().on_task_completed(
        session=session, completed=apply_t, tasks=tasks)
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    dec = creates[0].task
    # A re-plan (DECOMPOSE), NOT a regenerate SCAFFOLD.
    assert dec.kind is TaskKind.DECOMPOSE
    assert dec.inputs["replan_attempt"] == 1
    # The foundational note is folded into the revised goal.
    assert "package.json" in dec.inputs["prior_goal"]
    assert "manifest" in dec.inputs["prior_goal"]
    # No terminal failure while a re-plan is still possible.
    assert not [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]


def test_router_scaffold_dropped_foundational_file_escalates_to_replan():
    """Mirror of the APPLY guard on the SCAFFOLD dropped-file edge."""
    session, parent, scaffold, tasks = _build_scaffold_failed_chain(
        scaffold_failed_files=[
            {"file": "requirements.txt",
             "error": "not a valid pip requirements file"}])
    plan = Router().on_task_completed(
        session=session, completed=scaffold, tasks=tasks)
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    dec = creates[0].task
    assert dec.kind is TaskKind.DECOMPOSE
    assert "requirements.txt" in dec.inputs["prior_goal"]
    assert not [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]


def test_router_scaffold_foundational_empty_patch_regenerates_not_replan():
    """An empty-patch drop of a foundational manifest is a transient miss:
    it takes the targeted regenerate, NOT the heavy DECOMPOSE re-plan."""
    session, parent, scaffold, tasks = _build_scaffold_failed_chain(
        scaffold_failed_files=[
            {"file": "package.json",
             "error": "generator returned empty patch"}])
    plan = Router().on_task_completed(
        session=session, completed=scaffold, tasks=tasks)
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    new_scaffold = creates[0].task
    # A targeted regenerate SCAFFOLD, not a re-plan.
    assert new_scaffold.kind is TaskKind.SCAFFOLD
    assert "package.json" in new_scaffold.inputs["regenerate_files"]
    assert not [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]


def test_router_apply_foundational_empty_patch_regenerates_not_replan():
    """Mirror of the SCAFFOLD carve-out on the APPLY dropped-file edge."""
    session, scaffold, apply_t, tasks = _build_apply_failed_chain(
        apply_failed_files=[
            {"file": "package.json",
             "error": "generator returned empty patch"}])
    plan = Router().on_task_completed(
        session=session, completed=apply_t, tasks=tasks)
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    new_scaffold = creates[0].task
    assert new_scaffold.kind is TaskKind.SCAFFOLD
    assert "package.json" in new_scaffold.inputs["regenerate_files"]
    assert not [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]


def test_router_apply_failed_files_repeat_failure_skips_regenerate():
    """The same files dropped for the same reasons -> re-plan, not retry."""
    session, scaffold, apply_t, tasks = _build_apply_failed_chain()
    first = Router().on_task_completed(
        session=session, completed=apply_t, tasks=tasks)
    retry = [a for a in first.actions if isinstance(a, CreateTask)][0].task
    scaffold.inputs["prior_failure_signatures"] = (
        retry.inputs["prior_failure_signatures"])
    plan = Router().on_task_completed(
        session=session, completed=apply_t, tasks=tasks)
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    assert creates[0].task.kind is TaskKind.DECOMPOSE
    assert creates[0].task.inputs["replan_attempt"] == 1


def test_router_apply_failed_files_budget_exhausted_escalates_to_replan():
    """C2: regenerate budget spent -> escalate once to a fresh DECOMPOSE."""
    from cgx.session.budget import REGENERATE_BUDGET
    session, scaffold, apply_t, tasks = _build_apply_failed_chain(
        prior_regens=REGENERATE_BUDGET)
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
    from cgx.session.budget import REGENERATE_BUDGET, REPLAN_BUDGET
    session, _scaffold, apply_t, tasks = _build_apply_failed_chain(
        prior_regens=REGENERATE_BUDGET, prior_replans=REPLAN_BUDGET)
    plan = Router().on_task_completed(
        session=session, completed=apply_t, tasks=tasks)
    # No terminal failure: the successfully applied files carry forward.
    assert not [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    assert creates[0].task.kind is TaskKind.BOOTSTRAP_ENV


def test_router_apply_failed_files_replan_budget_exhausted_no_survivors_fails():
    """B: budgets spent and nothing generated cleanly -> terminal FAILED."""
    from cgx.session.budget import REGENERATE_BUDGET, REPLAN_BUDGET
    session, _scaffold, apply_t, tasks = _build_apply_failed_chain(
        prior_regens=REGENERATE_BUDGET, prior_replans=REPLAN_BUDGET,
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
                                 survivors: int = 8,
                                 scaffold_failed_files=None):
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
    if scaffold_failed_files is None:
        scaffold_failed_files = [
            {"file": "src/components/Calculator.jsx",
             "error": "ReadTimeout: read timed out (read timeout=300.0)"},
            {"file": "backend/main.py",
             "error": "generator returned empty patch"}][:failed_count]
    failed = scaffold_failed_files
    scaffold.outputs = {"scaffold_artifact_id": "art_scaffold",
                        "generated_count": survivors,
                        "failed_count": len(failed), "failed": failed}
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
    # The retry carries the failure signature so a repeat is detectable.
    from cgx.session.greenfield_edges import _scaffold_failure_signature
    assert new_scaffold.inputs["prior_failure_signatures"] == [
        _scaffold_failure_signature(scaffold.outputs["failed"])]


def test_router_scaffold_failed_files_repeat_failure_falls_back_to_ast():
    """A retry that reproduces the identical drop switches strategy.

    Budget remains, but re-running the same file-level generator on the
    same files failing the same gate only reproduces the drop, so the
    router escalates to a symbol-level AST regenerate -- a genuinely
    different strategy -- rather than spending another identical pass or
    prematurely re-planning the manifest.
    """
    from cgx.session.greenfield_edges import _scaffold_failure_signature
    session, _parent, scaffold, tasks = _build_scaffold_failed_chain()
    scaffold.inputs["prior_failure_signatures"] = [
        _scaffold_failure_signature(scaffold.outputs["failed"])]
    plan = Router().on_task_completed(
        session=session, completed=scaffold, tasks=tasks)
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    ast_regen = creates[0].task
    assert ast_regen.kind is TaskKind.AST_REGENERATE
    # Targeted at exactly the dropped files, reusing the prior scaffold's
    # good diffs rather than rebuilding the tree.
    assert set(ast_regen.inputs["regenerate_files"]) == {
        "src/components/Calculator.jsx", "backend/main.py"}
    assert ast_regen.inputs["prior_scaffold_artifact_id"] == "art_scaffold"
    # The AST fallback carries the flap signature so a repeat under the new
    # strategy is still detectable downstream.
    assert ast_regen.inputs["prior_failure_signatures"] == [
        _scaffold_failure_signature(scaffold.outputs["failed"])]
    # No re-plan and no terminal failure while the AST fallback is untried.
    assert not [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]
    # The live APPLY child is abandoned so the stale subtree cannot run.
    abandoned = {a.task_id for a in plan.actions
                 if isinstance(a, UpdateTaskStatus)}
    pending = [t for t in tasks if t.status is TaskNodeStatus.READY]
    assert {p.task_id for p in pending} <= abandoned


def test_scaffold_failure_signature_ignores_the_hallucinated_name():
    """Trading one invented module for another is not progress."""
    from cgx.session.greenfield_edges import _scaffold_failure_signature
    def sig(err):
        return _scaffold_failure_signature([{"file": "tests/test_main.py",
                                             "error": err}])
    assert sig("imports unknown module(s) ['app']: not in the manifest") == \
        sig("imports unknown module(s) ['api']: not in the manifest")
    assert sig("imports unknown module(s) ['app']") != sig("duplicate content")
    assert sig("x") != _scaffold_failure_signature(
        [{"file": "backend/main.py", "error": "x"}])
    assert _scaffold_failure_signature([]) == ""
    assert _scaffold_failure_signature([{"file": "", "error": "x"}]) == ""


def test_router_scaffold_failed_files_budget_exhausted_escalates_to_replan():
    """C2: SCAFFOLD regenerate budget spent -> escalate to a fresh DECOMPOSE."""
    from cgx.session.budget import REGENERATE_BUDGET
    session, _parent, scaffold, tasks = _build_scaffold_failed_chain(
        prior_regens=REGENERATE_BUDGET)
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
    from cgx.session.budget import REGENERATE_BUDGET, REPLAN_BUDGET
    session, _parent, scaffold, tasks = _build_scaffold_failed_chain(
        prior_regens=REGENERATE_BUDGET, prior_replans=REPLAN_BUDGET,
        with_pending_child=False)
    plan = Router().on_task_completed(
        session=session, completed=scaffold, tasks=tasks)
    assert not [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    assert creates[0].task.kind is TaskKind.APPLY


def test_router_scaffold_failed_files_replan_budget_exhausted_no_survivors_fails():
    """B: budgets spent and nothing generated cleanly -> terminal FAILED."""
    from cgx.session.budget import REGENERATE_BUDGET, REPLAN_BUDGET
    session, _parent, scaffold, tasks = _build_scaffold_failed_chain(
        prior_regens=REGENERATE_BUDGET, prior_replans=REPLAN_BUDGET,
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


def test_router_replan_carries_flap_ledger_into_new_decompose():
    """A re-plan must not restart the failure ledger from empty.

    Observed live: a re-planned manifest reproduced the identical
    ``smoke_import|npm run build --silent`` failure and, because the new
    chain had never seen the signature, spent a second full repair
    budget on it before the session finally failed.
    """
    from cgx.session.budget import REGENERATE_BUDGET
    session, _parent, scaffold, tasks = _build_scaffold_failed_chain(
        prior_regens=REGENERATE_BUDGET)
    scaffold.inputs["prior_failure_signatures"] = [
        "smoke_import|npm run build --silent"]
    plan = Router().on_task_completed(
        session=session, completed=scaffold, tasks=tasks)
    dec = [a.task for a in plan.actions if isinstance(a, CreateTask)][0]
    assert dec.kind is TaskKind.DECOMPOSE
    assert dec.inputs["prior_failure_signatures"] == [
        "smoke_import|npm run build --silent"]


def test_router_flap_ledger_threads_replan_decompose_to_scaffold():
    """The ledger rides DECOMPOSE -> ASK(APPROVE_PLAN) -> SCAFFOLD."""
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    dec = TaskNode.new(
        session.session_id, TaskKind.DECOMPOSE, "revise",
        inputs={"prior_goal": "g", "requirements_artifact_id": "art_req",
                "replan_attempt": 1,
                "prior_failure_signatures": ["sig-old"]})
    dec.produced_artifact_id = "art_plan2"
    dec.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=dec, tasks=[dec])
    ask = [a.task for a in plan.actions if isinstance(a, CreateTask)][0]
    assert ask.inputs["prior_failure_signatures"] == ["sig-old"]
    decision = Decision.new(
        session.session_id, ask.task_id, DecisionKind.APPROVE_PLAN,
        "approve plan", {"approved": True})
    plan2 = Router().on_decision_recorded(
        session=session, decision=decision, tasks=[ask])
    sc = [a.task for a in plan2.actions if isinstance(a, CreateTask)][0]
    assert sc.kind is TaskKind.SCAFFOLD
    assert sc.inputs["prior_failure_signatures"] == ["sig-old"]
    # And on down the regenerated write chain.
    sc.produced_artifact_id = "art_scaffold2"
    sc.status = TaskNodeStatus.DONE
    plan3 = Router().on_task_completed(
        session=session, completed=sc, tasks=[sc])
    ap = [a.task for a in plan3.actions if isinstance(a, CreateTask)][0]
    assert ap.kind is TaskKind.APPLY
    assert ap.inputs["prior_failure_signatures"] == ["sig-old"]


def test_router_replan_carries_regenerate_attempt_into_new_decompose():
    """A re-plan must not hand the revised manifest a fresh regenerate budget.

    Left to reset, a fresh DECOMPOSE -> SCAFFOLD would be born at
    ``regenerate_attempt=0`` and get a whole second ``REGENERATE_BUDGET``,
    multiplying the total syntax-churn budget by the number of re-plans.
    The spent count must ride the new DECOMPOSE so the regenerate budget
    stays a per-session ceiling.
    """
    from cgx.session.budget import REGENERATE_BUDGET
    session, _parent, scaffold, tasks = _build_scaffold_failed_chain(
        prior_regens=REGENERATE_BUDGET)
    plan = Router().on_task_completed(
        session=session, completed=scaffold, tasks=tasks)
    dec = [a.task for a in plan.actions if isinstance(a, CreateTask)][0]
    assert dec.kind is TaskKind.DECOMPOSE
    assert dec.inputs["regenerate_attempt"] == REGENERATE_BUDGET


def test_router_regenerate_attempt_threads_replan_decompose_to_scaffold():
    """regenerate_attempt rides DECOMPOSE -> ASK(APPROVE_PLAN) -> SCAFFOLD."""
    from cgx.session.budget import REGENERATE_BUDGET
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    dec = TaskNode.new(
        session.session_id, TaskKind.DECOMPOSE, "revise",
        inputs={"prior_goal": "g", "requirements_artifact_id": "art_req",
                "replan_attempt": 1,
                "regenerate_attempt": REGENERATE_BUDGET})
    dec.produced_artifact_id = "art_plan2"
    dec.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=dec, tasks=[dec])
    ask = [a.task for a in plan.actions if isinstance(a, CreateTask)][0]
    assert ask.inputs["regenerate_attempt"] == REGENERATE_BUDGET
    decision = Decision.new(
        session.session_id, ask.task_id, DecisionKind.APPROVE_PLAN,
        "approve plan", {"approved": True})
    plan2 = Router().on_decision_recorded(
        session=session, decision=decision, tasks=[ask])
    sc = [a.task for a in plan2.actions if isinstance(a, CreateTask)][0]
    assert sc.kind is TaskKind.SCAFFOLD
    assert sc.inputs["regenerate_attempt"] == REGENERATE_BUDGET


def test_router_first_run_scaffold_has_no_flap_ledger_key():
    """A non-re-planned approval leaves the ledger key off entirely."""
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    ask = TaskNode.new(
        session.session_id, TaskKind.ASK_USER, "approve",
        inputs={"expected_kind": "approve_plan",
                "work_plan_artifact_id": "art_plan"})
    decision = Decision.new(
        session.session_id, ask.task_id, DecisionKind.APPROVE_PLAN,
        "approve plan", {"approved": True})
    plan = Router().on_decision_recorded(
        session=session, decision=decision, tasks=[ask])
    sc = [a.task for a in plan.actions if isinstance(a, CreateTask)][0]
    assert "prior_failure_signatures" not in sc.inputs


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
    from cgx.session.budget import REGENERATE_BUDGET
    session, _parent, scaffold, tasks = _build_crashed_scaffold(
        prior_regens=REGENERATE_BUDGET, with_pending_child=False)
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


def test_ast_scaffold_parses_unified_skeleton_string(store):
    """AST_REGENERATE handles the unified ``project_skeleton`` string.

    ``generate_project_skeleton`` stores ``contracts['project_skeleton']``
    as a single unified script delimited by ``# --- <path> ---`` markers,
    not a per-path dict. The executor must split it back into per-file
    sections instead of crashing with ``AttributeError`` on ``str.get``.
    """
    import re as _re
    from cgx.session.models import SessionMode
    from cgx.session.tasks.ast_scaffold import run_ast_scaffold

    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)

    skeleton = (
        "# --- backend/app.py ---\n"
        "def create_app():\n"
        "    pass\n"
        "\n"
        "# --- backend/models.py ---\n"
        "class User:\n"
        "    pass\n"
    )
    plan = Artifact.new(
        session.session_id, "task_x", ArtifactKind.WORK_PLAN, {
            "prior_goal": "g",
            "composed_goal": "build a flask api",
            "contracts": {"project_skeleton": skeleton},
            "layers": [{"name": "app", "files": [
                {"path": "backend/app.py", "description": "entry"},
                {"path": "backend/models.py", "description": "models"}]}],
        })
    store.save_artifact(plan)

    class _AstProvider:
        def chat(self, messages=None, **kwargs):
            user = ""
            for m in (messages or []):
                if m.get("role") == "user":
                    user = m.get("content", "")
            if "file header" in user:
                return {"content": "import os\n"}
            match = _re.search(r"named `([^`]+)`", user)
            name = match.group(1) if match else "impl"
            return {"content": f"def {name}():\n    return 1\n"}

    t = TaskNode.new(
        session.session_id, TaskKind.AST_REGENERATE, "ast",
        inputs={"work_plan_artifact_id": plan.artifact_id,
                "composed_goal": "build a flask api"})
    store.save_task(t)
    result = run_ast_scaffold(
        t, ExecutorDeps(provider=_AstProvider(), store=store))

    assert result.failure is None
    content = result.artifact.content
    assert content["failed"] == []
    generated = {g["file"]: g["content"] for g in content["generated"]}
    assert set(generated) == {"backend/app.py", "backend/models.py"}
    # The per-file skeleton section was parsed and its symbol regenerated,
    # proving the unified string was split by path rather than mis-read.
    assert "def create_app" in generated["backend/app.py"]
    assert "def User" in generated["backend/models.py"]


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
                      contracts=None, skills=None, **kwargs):
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
    from cgx.session.actions import RecordLesson
    from cgx.session.models import SessionMode
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
    from cgx.session.actions import RecordLesson
    from cgx.session.models import SessionMode
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
    from cgx.session.actions import RecordLesson
    from cgx.session.lessons import load_lessons
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
                      contracts=None, skills=None, **kwargs):
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




# ---------------------------------------------------------------------------
# Failure-edge matrix (Phase 2). Table-driven suites asserting the loop's
# core liveness invariant: a greenfield session must never strand 'active'
# with no runnable work. Every TaskKind's hard-failure route through
# Router.on_task_failed and every budget-exhaustion path through
# Router.on_task_completed must either reach a terminal session status or
# spawn a recovery task. This is the class of test whose absence let the
# DECOMPOSE session-killer ship.
# ---------------------------------------------------------------------------


def _gf_session():
    from cgx.session.models import SessionMode
    return Session.new("g", mode=SessionMode.GREENFIELD)


def _hard_failed(session, kind, **extra_inputs):
    """Build a FAILED task of ``kind`` with the common greenfield wiring."""
    from cgx.session.models import SessionMode
    task = TaskNode.new(
        session.session_id, kind, f"{kind.value} under test",
        inputs={"mode": SessionMode.GREENFIELD.value,
                "prior_goal": "build a todo app", **extra_inputs})
    task.status = TaskNodeStatus.FAILED
    task.error = f"{kind.value} executor crashed"
    return task


def _done(task, **outputs):
    task.outputs = outputs
    task.status = TaskNodeStatus.DONE
    return task


def _assert_terminal_failed(plan):
    """The plan ends the session FAILED and spawns no further work."""
    assert [a for a in plan.actions if isinstance(a, CreateTask)] == []
    status = [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]
    assert [s.status for s in status] == [SessionStatus.FAILED]


def _assert_recovers_with(plan, kind):
    """The plan spawns exactly one ``kind`` task and no terminal status."""
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    assert creates[0].task.kind is kind
    assert not any(isinstance(a, UpdateSessionStatus) for a in plan.actions)


@pytest.mark.parametrize("kind", list(TaskKind), ids=lambda k: k.value)
def test_router_hard_failure_terminates_every_kind(kind):
    """on_task_failed: a bare hard failure is terminal for every TaskKind."""
    session = _gf_session()
    task = _hard_failed(session, kind)
    plan = Router().on_task_failed(
        session=session, failed=task, tasks=[task])
    _assert_terminal_failed(plan)


@pytest.mark.parametrize("kind", list(TaskKind), ids=lambda k: k.value)
def test_router_retryable_failure_only_requeues_decompose(kind):
    """on_task_failed(retryable): DECOMPOSE re-queues, the rest terminal."""
    session = _gf_session()
    task = _hard_failed(session, kind)
    plan = Router().on_task_failed(
        session=session, failed=task, tasks=[task], retryable=True)
    if kind is TaskKind.DECOMPOSE:
        _assert_recovers_with(plan, TaskKind.DECOMPOSE)
    else:
        _assert_terminal_failed(plan)


@pytest.mark.parametrize("kind", list(TaskKind), ids=lambda k: k.value)
def test_router_crash_checkpoint_only_resumes_scaffold(kind):
    """on_task_failed(resume ckpt): SCAFFOLD resumes, the rest terminal."""
    session = _gf_session()
    task = _hard_failed(session, kind)
    plan = Router().on_task_failed(
        session=session, failed=task, tasks=[task],
        resume_scaffold_artifact_id="art_ckpt")
    if kind is TaskKind.SCAFFOLD:
        _assert_recovers_with(plan, TaskKind.SCAFFOLD)
    else:
        _assert_terminal_failed(plan)


def test_router_crash_checkpoint_resume_budget_spent_is_terminal():
    """A second SCAFFOLD crash exhausts the regenerate budget -> FAILED."""
    from cgx.session.budget import REGENERATE_BUDGET
    session = _gf_session()
    task = _hard_failed(session, TaskKind.SCAFFOLD,
                        regenerate_attempt=REGENERATE_BUDGET)
    plan = Router().on_task_failed(
        session=session, failed=task, tasks=[task],
        resume_scaffold_artifact_id="art_ckpt")
    _assert_terminal_failed(plan)


# Builders for the completion-time budget-exhaustion matrix. Each returns
# (session, completed_task, tasks) shaped exactly as the runner would hand
# them to Router.on_task_completed.

def _exhausted_verify():
    from cgx.session.budget import REPAIR_BUDGET
    session = _gf_session()
    ver = _greenfield_failed_verify(
        signature="fresh-sig", repair_attempt=REPAIR_BUDGET, session=session)
    return session, ver, [ver]


def _exhausted_runtime_verify():
    from cgx.session.budget import REPAIR_BUDGET
    from cgx.session.models import SessionMode
    session = _gf_session()
    rv = TaskNode.new(
        session.session_id, TaskKind.RUNTIME_VERIFY, "runtime",
        inputs={"mode": SessionMode.GREENFIELD.value,
                "repair_attempt": REPAIR_BUDGET})
    rv.produced_artifact_id = "art_runtime"
    _done(rv, outcome="failed", failure_signature="runtime_boot|app.py")
    return session, rv, [rv]


def _exhausted_api_check():
    from cgx.session.budget import REPAIR_BUDGET
    from cgx.session.models import SessionMode
    session = _gf_session()
    api = TaskNode.new(
        session.session_id, TaskKind.API_CHECK, "api",
        inputs={"build_artifact_id": "art_build",
                "mode": SessionMode.GREENFIELD.value,
                "repair_attempt": REPAIR_BUDGET})
    api.produced_artifact_id = "art_api"
    _done(api, outcome="failed", failed_count=1,
          failure_signature="api_check|pkg.symbol")
    return session, api, [api]


def _exhausted_smoke():
    from cgx.session.budget import REPAIR_BUDGET
    from cgx.session.models import SessionMode
    session = _gf_session()
    sm = TaskNode.new(
        session.session_id, TaskKind.SMOKE, "smoke",
        inputs={"build_artifact_id": "art_build",
                "mode": SessionMode.GREENFIELD.value,
                "repair_attempt": REPAIR_BUDGET})
    sm.produced_artifact_id = "art_smoke"
    _done(sm, outcome="failed", failed_count=1,
          failure_signature="smoke_import|pkg")
    return session, sm, [sm]


def _dropped_files_scaffold(*, regen_spent, replan_spent, survivors):
    from cgx.session.budget import REGENERATE_BUDGET, REPLAN_BUDGET
    from cgx.session.models import SessionMode
    session = _gf_session()
    sc = TaskNode.new(
        session.session_id, TaskKind.SCAFFOLD, "scaffold",
        inputs={"mode": SessionMode.GREENFIELD.value,
                "prior_goal": "build a todo app",
                "regenerate_attempt":
                    REGENERATE_BUDGET if regen_spent else 0,
                "replan_attempt": REPLAN_BUDGET if replan_spent else 0})
    _done(sc, failed_count=1,
          failed=[{"file": "app.py", "error": "empty patch"}],
          generated_count=survivors,
          scaffold_artifact_id="art_scaffold")
    return session, sc, [sc]


def _apply_dropped_files_no_scaffold():
    from cgx.session.models import SessionMode
    session = _gf_session()
    ap = TaskNode.new(
        session.session_id, TaskKind.APPLY, "apply",
        inputs={"mode": SessionMode.GREENFIELD.value})
    _done(ap, failed_count=1,
          failed_files=[{"file": "app.py", "error": "syntax"}])
    return session, ap, [ap]


def _apply_dropped_files_budgets_spent():
    from cgx.session.budget import REGENERATE_BUDGET, REPLAN_BUDGET
    from cgx.session.models import SessionMode
    session = _gf_session()
    sc = TaskNode.new(
        session.session_id, TaskKind.SCAFFOLD, "scaffold",
        inputs={"mode": SessionMode.GREENFIELD.value,
                "prior_goal": "build a todo app",
                "regenerate_attempt": REGENERATE_BUDGET,
                "replan_attempt": REPLAN_BUDGET})
    _done(sc, generated_count=0, scaffold_artifact_id="art_scaffold")
    ap = TaskNode.new(
        session.session_id, TaskKind.APPLY, "apply",
        parent_task_id=sc.task_id,
        inputs={"mode": SessionMode.GREENFIELD.value})
    _done(ap, failed_count=1,
          failed_files=[{"file": "app.py", "error": "syntax"}])
    return session, ap, [sc, ap]


def _repair_no_patch():
    from cgx.session.models import SessionMode
    session = _gf_session()
    rep = TaskNode.new(
        session.session_id, TaskKind.REPAIR, "repair",
        inputs={"mode": SessionMode.GREENFIELD.value})
    _done(rep, can_apply=False, classification="unknown", strategy="patch")
    return session, rep, [rep]


def _repair_regenerate_budget_spent():
    from cgx.session.budget import REPAIR_REGENERATE_BUDGET
    from cgx.session.models import SessionMode
    session = _gf_session()
    sc = TaskNode.new(
        session.session_id, TaskKind.SCAFFOLD, "scaffold",
        inputs={"mode": SessionMode.GREENFIELD.value,
                "repair_regenerate_attempt": REPAIR_REGENERATE_BUDGET})
    _done(sc, scaffold_artifact_id="art_scaffold")
    rep = TaskNode.new(
        session.session_id, TaskKind.REPAIR, "repair",
        parent_task_id=sc.task_id,
        inputs={"mode": SessionMode.GREENFIELD.value})
    _done(rep, can_apply=False, strategy="regenerate",
          classification="logic_error", extra_constraints={})
    return session, rep, [sc, rep]


# Each row: (id, builder, expectation). Expectation "failed" asserts a
# terminal FAILED with no spawn; a TaskKind asserts a single recovery task
# of that kind with no terminal status. Either way the session never
# strands 'active' with nothing runnable.
_BUDGET_EXHAUSTION_CASES = [
    ("verify_repair_budget_spent", _exhausted_verify, "failed"),
    ("runtime_verify_repair_budget_spent",
     _exhausted_runtime_verify, "failed"),
    ("api_check_repair_budget_spent", _exhausted_api_check, "failed"),
    ("smoke_repair_budget_spent", _exhausted_smoke, "failed"),
    ("scaffold_dropped_files_regen_left",
     lambda: _dropped_files_scaffold(
         regen_spent=False, replan_spent=False, survivors=0),
     TaskKind.SCAFFOLD),
    ("scaffold_dropped_files_regen_spent_replans",
     lambda: _dropped_files_scaffold(
         regen_spent=True, replan_spent=False, survivors=0),
     TaskKind.DECOMPOSE),
    ("scaffold_dropped_files_all_spent_no_survivors",
     lambda: _dropped_files_scaffold(
         regen_spent=True, replan_spent=True, survivors=0),
     "failed"),
    ("scaffold_dropped_files_all_spent_with_survivors",
     lambda: _dropped_files_scaffold(
         regen_spent=True, replan_spent=True, survivors=3),
     TaskKind.APPLY),
    ("apply_dropped_files_no_scaffold_lineage",
     _apply_dropped_files_no_scaffold, "failed"),
    ("apply_dropped_files_all_budgets_spent",
     _apply_dropped_files_budgets_spent, "failed"),
    ("repair_no_patch_no_strategy_left", _repair_no_patch, "failed"),
    ("repair_regenerate_budget_spent_no_patch",
     _repair_regenerate_budget_spent, "failed"),
]


@pytest.mark.parametrize(
    "build,expect",
    [(b, e) for _, b, e in _BUDGET_EXHAUSTION_CASES],
    ids=[i for i, _, _ in _BUDGET_EXHAUSTION_CASES])
def test_router_budget_exhaustion_reaches_terminal_or_recovers(build, expect):
    """Every budget-exhaustion edge ends terminal or spawns recovery work."""
    session, completed, tasks = build()
    plan = Router().on_task_completed(
        session=session, completed=completed, tasks=tasks)
    if expect == "failed":
        _assert_terminal_failed(plan)
    else:
        _assert_recovers_with(plan, expect)


# --------------------- e2e greenfield smoke (scripted provider) ---------------------
#
# Unlike test_runner_full_greenfield_loop (which stubs every executor to
# pin the router's edges), this suite runs the REAL executors -- CLARIFY,
# DECOMPOSE, SCAFFOLD, APPLY, VERIFY, REPAIR -- against a tmp_path project
# with a scripted provider standing in for the local LLM. Only the
# environment-heavy kinds (BOOTSTRAP_ENV, API_CHECK, SMOKE,
# RUNTIME_VERIFY) are stubbed, so the whole loop -- including one injected
# failure + deterministic repair -- is exercised in CI without a GPU.

_E2E_CALCULATOR = (
    '"""Tiny calculator library."""\n'
    "\n"
    "\n"
    "def add(a, b):\n"
    "    return a + b\n"
    "\n"
    "\n"
    "def subtract(a, b):\n"
    "    return a - b\n"
)

# Deliberate unittest/pytest mix: a pytest-style class (no TestCase base)
# whose methods call self.assert* helpers. pytest collects it fine, then
# every test dies with AttributeError: ... 'assertEqual' -- the canonical
# ``unittest_pytest_mix`` failure the deterministic repair chain fixes by
# adding the unittest.TestCase base + import.
_E2E_BROKEN_TEST = (
    '"""Tests for the calculator."""\n'
    "\n"
    "from calculator import add, subtract\n"
    "\n"
    "\n"
    "class TestCalculator:\n"
    "    def test_add(self):\n"
    "        self.assertEqual(add(2, 3), 5)\n"
    "\n"
    "    def test_subtract(self):\n"
    "        self.assertEqual(subtract(5, 3), 2)\n"
)

_E2E_QUESTIONS = {
    "questions": [
        {"id": "q1", "prompt": "Which Python version?",
         "hint": "e.g. 3.11", "suggested": ["3.11"]},
        {"id": "q2", "prompt": "Which test framework?",
         "suggested": ["pytest"]},
        {"id": "q3", "prompt": "Any CLI needed?",
         "suggested": ["No"]},
    ]
}

_E2E_MANIFEST = {
    "plan_md": "## Plan\n- calculator.py\n- test_calculator.py",
    "layers": [
        {"name": "core", "files": [
            {"path": "calculator.py",
             "description": "add/subtract helpers"}]},
        {"name": "tests", "files": [
            {"path": "test_calculator.py",
             "description": "pytest suite for the calculator",
             "depends_on": ["calculator.py"]}]},
    ],
}


class _ScriptedLocalProvider:
    """Scripted stand-in for a local LLM.

    Routes each ``chat`` call on its declarative shape -- the
    ``json_schema`` object identity for CLARIFY/DECOMPOSE, the
    ``FILE TO GENERATE`` block for per-file scaffolding -- and returns
    canned JSON, so the real executors parse real provider replies
    without any network or GPU.
    """

    def __init__(self, files):
        self._files = dict(files)
        self.calls = []

    def chat(self, messages=None, **kwargs):
        import json as _json
        from cgx.answer.schemas import (
            CLARIFY_QUESTIONS_SCHEMA,
            MANIFEST_SCHEMA,
        )
        schema = kwargs.get("json_schema")
        user = ""
        for m in reversed(messages or []):
            if m.get("role") == "user":
                user = str(m.get("content") or "")
                break
        if schema is CLARIFY_QUESTIONS_SCHEMA:
            self.calls.append("clarify")
            return {"content": _json.dumps(_E2E_QUESTIONS)}
        if schema is MANIFEST_SCHEMA:
            self.calls.append("manifest")
            return {"content": _json.dumps(_E2E_MANIFEST)}
        marker = "FILE TO GENERATE:\nPath: "
        idx = user.find(marker)
        if idx >= 0:
            path = user[idx + len(marker):].split("\n", 1)[0].strip()
            self.calls.append(f"file:{path}")
            body = self._files.get(path)
            if body is None:
                body = ('"""Placeholder module."""\n'
                        if path.endswith(".py")
                        else f"placeholder for {path}\n")
            return {"content": _json.dumps({"content": body})}
        # Bounded plan self-critique (DECOMPOSE, P1.2) -- a plain chat call
        # asking for speculative files to drop. Return no removals so the
        # scripted manifest ships intact (today's degrade-to-safe behaviour).
        if "speculative files to remove" in user:
            self.calls.append("critique")
            return {"content": _json.dumps({"remove": []})}
        # Mandatory project-skeleton pass (DECOMPOSE) -- a plain chat call
        # carrying the manifest paths, no schema and no per-file marker.
        if "Manifest Paths:" in user:
            self.calls.append("skeleton")
            return {"content": "```python\npass\n```"}
        self.calls.append("unrouted")
        return {"content": "{}"}


def _install_stub_bootstrap_env_host_python():
    """BOOTSTRAP_ENV stub whose BUILD_REPORT points at the host python.

    The real VERIFY reads ``python_exe`` from the BUILD_REPORT and feeds
    it to the pytest subprocess; pointing it at ``sys.executable`` (which
    has pytest installed) lets the real test run happen without a real
    ``pip install`` bootstrap.
    """
    import sys

    @register_executor(TaskKind.BOOTSTRAP_ENV)
    def _stub(task, deps):
        artifact = Artifact.new(
            session_id=task.session_id,
            produced_by_task_id=task.task_id,
            kind=ArtifactKind.BUILD_REPORT,
            content={
                "apply_artifact_id": task.inputs.get("apply_artifact_id"),
                "project_type": "python",
                "venv_path": "",
                "python_exe": sys.executable,
                "installed_from": [],
                "installed_packages": [],
                "failed_installs": [],
                "outcome": "succeeded",
                "pip_log_tail": "",
            })
        return ExecutorResult(
            outputs={"build_artifact_id": artifact.artifact_id,
                     "outcome": "succeeded",
                     "project_type": "python",
                     "python_exe": sys.executable,
                     "installed_count": 0,
                     "failed_count": 0},
            artifact=artifact)


def _install_stub_runtime_verify_passed():
    @register_executor(TaskKind.RUNTIME_VERIFY)
    def _stub(task, deps):
        artifact = Artifact.new(
            session_id=task.session_id,
            produced_by_task_id=task.task_id,
            kind=ArtifactKind.RUNTIME_REPORT,
            content={"probes": [], "outcome": "passed",
                     "failure_signature": ""})
        return ExecutorResult(
            outputs={"runtime_artifact_id": artifact.artifact_id,
                     "outcome": "passed",
                     "failed_count": 0,
                     "failure_signature": ""},
            artifact=artifact)


def test_runner_full_greenfield_loop_with_recovery(tmp_path, store,
                                                   monkeypatch):
    """E2E smoke: a scripted provider drives a whole greenfield session.

    CLARIFY -> ASK(clarify_answers) -> DECOMPOSE -> ASK(approve_plan) ->
    SCAFFOLD -> APPLY -> [boot/api/smoke stubs] -> VERIFY(fails: injected
    unittest/pytest mix) -> REPAIR(patch) -> APPLY -> [stubs] ->
    VERIFY(passes) -> RUNTIME_VERIFY -> COMPLETED.

    The scaffold, apply, verify, and repair executors are the real ones:
    files land on disk under tmp_path and pytest genuinely runs twice
    (fail, then pass after the deterministic repair rewrites the test).
    """
    from cgx.session.models import SessionMode

    monkeypatch.setenv("CGX_LESSONS_PATH", str(tmp_path / "lessons.jsonl"))
    proj = tmp_path / "proj"
    proj.mkdir()

    _install_stub_bootstrap_env_host_python()
    _install_stub_api_check_greenfield()
    _install_stub_smoke_greenfield()
    _install_stub_runtime_verify_passed()

    provider = _ScriptedLocalProvider({
        "calculator.py": _E2E_CALCULATOR,
        "test_calculator.py": _E2E_BROKEN_TEST,
        "requirements.txt": "pytest\n",
        "README.md": "# Calculator\n\nTiny add/subtract library.\n",
    })
    deps = ExecutorDeps(project_root=str(proj), store=store,
                        provider=provider)
    runner = SessionRunner(store)
    session = runner.start_session(
        objective="build a small python calculator library with tests",
        project_root=str(proj),
        mode=SessionMode.GREENFIELD)

    def drain():
        """Run READY tasks until the loop quiesces or pauses on an ASK."""
        for _ in range(60):
            t = runner.run_next(session_id=session.session_id, deps=deps)
            if t is None:
                return
            if (t.kind is TaskKind.ASK_USER
                    and t.status is TaskNodeStatus.IN_PROGRESS):
                return
        raise AssertionError("session did not quiesce within 60 steps")

    # 1. CLARIFY runs against the scripted provider, pauses on the
    #    clarify_answers ASK.
    drain()
    ask = next(t for t in store.list_tasks(session.session_id)
               if t.kind is TaskKind.ASK_USER
               and t.inputs.get("expected_kind") == "clarify_answers")
    assert ask.status is TaskNodeStatus.IN_PROGRESS
    runner.post_decision(
        session_id=session.session_id,
        decision=build_decision(
            session_id=session.session_id, task=ask,
            chosen={"answers": {"q1": "3.11", "q2": "pytest",
                                "q3": "No"}}))

    # 2. DECOMPOSE plans the manifest, pauses on the approve_plan ASK.
    drain()
    approve = next(t for t in store.list_tasks(session.session_id)
                   if t.kind is TaskKind.ASK_USER
                   and t.inputs.get("expected_kind") == "approve_plan")
    assert approve.status is TaskNodeStatus.IN_PROGRESS
    runner.post_decision(
        session_id=session.session_id,
        decision=build_decision(
            session_id=session.session_id, task=approve,
            chosen={"approved": True}))

    # 3. Everything else runs unattended: scaffold + apply land real
    #    files, the first VERIFY fails on the injected mix, REPAIR
    #    patches it, and the second VERIFY + RUNTIME_VERIFY complete.
    drain()

    session_after = store.get_session(session.session_id)
    assert session_after.status is SessionStatus.COMPLETED

    tasks = store.list_tasks(session.session_id)
    assert all(t.status is not TaskNodeStatus.READY for t in tasks)

    # The verify ladder ran twice: fail, then pass after the repair.
    verifies = sorted((t for t in tasks if t.kind is TaskKind.VERIFY),
                      key=lambda t: t.started_at)
    assert len(verifies) == 2
    assert verifies[0].outputs["outcome"] in ("assertions_failed", "failed")
    assert verifies[0].outputs["failing_count"] == 2
    assert verifies[1].outputs["outcome"] == "passed"
    assert verifies[1].outputs["failing_count"] == 0

    # Exactly one deterministic repair, classified and applied.
    repairs = [t for t in tasks if t.kind is TaskKind.REPAIR]
    assert len(repairs) == 1
    assert repairs[0].outputs["classification"] == "unittest_pytest_mix"
    assert repairs[0].outputs["can_apply"] is True

    # The scaffold and the repair genuinely landed on disk.
    assert (proj / "calculator.py").read_text(
        encoding="utf-8") == _E2E_CALCULATOR
    fixed = (proj / "test_calculator.py").read_text(encoding="utf-8")
    assert "import unittest" in fixed
    assert "unittest.TestCase" in fixed

    # The provider was exercised through every scripted shape.
    assert "clarify" in provider.calls
    assert "manifest" in provider.calls
    assert "file:calculator.py" in provider.calls
    assert "file:test_calculator.py" in provider.calls
    assert "unrouted" not in provider.calls


# --- REPAIR retrieval-assist: missing-index degradation ---------------------


def test_repair_retrieval_skips_when_index_absent(tmp_path, caplog):
    """A greenfield project is never indexed, so ``index_dir`` points at a
    manifest that does not exist. Retrieval-assisted candidate fill must
    degrade to ``[]`` -- not raise / log a FileNotFoundError every round."""
    from cgx.session.tasks.repair import _retrieval_relevant_files

    index_dir = tmp_path / "cgx_index" / "indices"  # no meta.json on disk
    deps = ExecutorDeps(
        project_root=str(tmp_path),
        index_dir=str(index_dir),
        records_path=str(tmp_path / "records.jsonl"),
    )
    with caplog.at_level("ERROR", logger="cgx.session.tasks.repair"):
        out = _retrieval_relevant_files(
            deps, query="fix the failing assertion", root=tmp_path, limit=4)
    assert out == []
    # The old behaviour logged a crash at ERROR on every attempt; the
    # missing-index path is now a quiet debug-level skip.
    assert not [r for r in caplog.records if r.levelno >= 40]


def test_repair_retrieval_calls_query_when_index_present(tmp_path, monkeypatch):
    """When the manifest exists, retrieval is invoked and its ``top_files``
    are resolved to existing first-party paths."""
    from cgx.session.tasks import repair as repair_mod

    index_dir = tmp_path / "idx"
    index_dir.mkdir()
    (index_dir / "meta.json").write_text("{}", encoding="utf-8")
    (tmp_path / "calc.py").write_text("x = 1\n", encoding="utf-8")

    calls = {}

    def _fake_run_query_auto(**kwargs):
        calls.update(kwargs)
        return {"top_files": [{"file": "calc.py"}, {"file": "missing.py"}]}

    monkeypatch.setattr(
        "cgx.pipeline.auto.run_query_auto", _fake_run_query_auto)

    deps = ExecutorDeps(
        project_root=str(tmp_path),
        index_dir=str(index_dir),
        records_path=str(tmp_path / "records.jsonl"),
    )
    out = repair_mod._retrieval_relevant_files(
        deps, query="boom", root=tmp_path, limit=4)
    assert calls, "run_query_auto should have been called"
    # Only the existing first-party file survives resolution.
    assert out == ["calc.py"]


# --------------------- DIAGNOSE router wiring (P2.5) ---------------------
#
# Two seams: (1) the gate -> DIAGNOSE edges -- a reasoning-class
# ``classification`` on a failed VERIFY / RUNTIME_VERIFY routes to the
# DIAGNOSE rung instead of a mechanical REPAIR; (2) the
# ``_diagnose_dispatch_actions`` guard -- a completed DIAGNOSE verdict's
# ``minimal_action`` maps to exactly one deterministic successor. The
# router stays pure: it reads only ``classification`` / ``minimal_action``
# and the dependency / target-file lists the executor already placed in
# ``outputs``.


def _diagnose_completed(*, minimal_action: str, extra_outputs: dict | None = None,
                        repair_attempt: int = 1, prior_repair_regens: int = 0,
                        with_scaffold: bool = True):
    """Build a SCAFFOLD->APPLY->VERIFY->DIAGNOSE(DONE) chain.

    Returns ``(session, diagnose, tasks)``. The DIAGNOSE node carries the
    verdict ``minimal_action`` (+ any ``extra_outputs``) the dispatch guard
    reads; the SCAFFOLD ancestor lets the regenerate / escalate arms splice
    a scoped re-scaffold.
    """
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    tasks: list = []
    parent_id = None
    if with_scaffold:
        scaffold = TaskNode.new(
            session.session_id, TaskKind.SCAFFOLD, "scaffold",
            inputs={"work_plan_artifact_id": "art_plan",
                    "repair_regenerate_attempt": prior_repair_regens})
        scaffold.status = TaskNodeStatus.DONE
        apply_t = TaskNode.new(
            session.session_id, TaskKind.APPLY, "apply",
            parent_task_id=scaffold.task_id, inputs={})
        apply_t.status = TaskNodeStatus.DONE
        verify = TaskNode.new(
            session.session_id, TaskKind.VERIFY, "verify",
            parent_task_id=apply_t.task_id, inputs={})
        verify.status = TaskNodeStatus.DONE
        tasks += [scaffold, apply_t, verify]
        parent_id = verify.task_id
    diag = TaskNode.new(
        session.session_id, TaskKind.DIAGNOSE, "diagnose",
        parent_task_id=parent_id,
        inputs={"verify_artifact_id": "art_verify",
                "build_artifact_id": "art_build",
                "apply_artifact_id": "art_applied",
                "scaffold_artifact_id": "art_scaffold",
                "plan_artifact_id": "art_plan",
                "prior_goal": "g",
                "mode": SessionMode.GREENFIELD.value,
                "repair_attempt": repair_attempt})
    outputs = {"minimal_action": minimal_action,
               "verify_artifact_id": "art_verify",
               "repair_attempt": repair_attempt}
    outputs.update(extra_outputs or {})
    diag.outputs = outputs
    diag.status = TaskNodeStatus.DONE
    tasks.append(diag)
    return session, diag, tasks


# (minimal_action, extra outputs, expected successor kind)
_DIAGNOSE_DISPATCH_CASES = [
    ("patch_files", {"target_files": ["src/x.py"]}, TaskKind.REPAIR),
    ("add_dependency", {"add_dependencies": ["flask"]}, TaskKind.BOOTSTRAP_ENV),
    ("remove_dependency", {"remove_dependencies": ["selenium"]},
     TaskKind.BOOTSTRAP_ENV),
    ("adjust_manifest", {"target_files": ["src/x.py"]}, TaskKind.SCAFFOLD),
    ("regenerate_files", {"target_files": ["src/x.py"]}, TaskKind.SCAFFOLD),
    ("escalate", {}, TaskKind.SCAFFOLD),
]


@pytest.mark.parametrize(
    "action,extra,kind", _DIAGNOSE_DISPATCH_CASES,
    ids=[c[0] for c in _DIAGNOSE_DISPATCH_CASES])
def test_diagnose_dispatch_routes_each_minimal_action(action, extra, kind):
    """Every minimal_action verdict maps to exactly one successor kind."""
    session, diag, tasks = _diagnose_completed(
        minimal_action=action, extra_outputs=extra)
    plan = Router().on_task_completed(
        session=session, completed=diag, tasks=tasks)
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    assert creates[0].task.kind is kind


def test_diagnose_patch_files_spawns_targeted_repair():
    """patch_files -> REPAIR reusing the diagnosed source report + targets."""
    session, diag, tasks = _diagnose_completed(
        minimal_action="patch_files", repair_attempt=2,
        extra_outputs={"target_files": ["src/handlers.py"],
                       "repair_ledger_fact_id": "fact_led",
                       "repair_attempt": 2})
    plan = Router().on_task_completed(
        session=session, completed=diag, tasks=tasks)
    rep = [a.task for a in plan.actions if isinstance(a, CreateTask)][0]
    assert rep.kind is TaskKind.REPAIR
    assert rep.inputs["verify_artifact_id"] == "art_verify"
    assert rep.inputs["target_files"] == ["src/handlers.py"]
    # The gate-charged attempt + the ledger id ride the chain verbatim.
    assert rep.inputs["repair_attempt"] == 2
    assert rep.inputs["repair_ledger_fact_id"] == "fact_led"


def test_diagnose_add_dependency_spawns_bootstrap_install():
    """add_dependency -> BOOTSTRAP_ENV threading missing_modules to install."""
    session, diag, tasks = _diagnose_completed(
        minimal_action="add_dependency",
        extra_outputs={"add_dependencies": ["flask", "flask_cors"],
                       "repair_ledger_fact_id": "fact_led"})
    plan = Router().on_task_completed(
        session=session, completed=diag, tasks=tasks)
    boot = [a.task for a in plan.actions if isinstance(a, CreateTask)][0]
    assert boot.kind is TaskKind.BOOTSTRAP_ENV
    assert boot.inputs["missing_modules"] == ["flask", "flask_cors"]
    assert boot.inputs["repair_ledger_fact_id"] == "fact_led"


def test_diagnose_remove_dependency_spawns_bootstrap_descope():
    """remove_dependency -> BOOTSTRAP_ENV threading descope_packages (C3)."""
    session, diag, tasks = _diagnose_completed(
        minimal_action="remove_dependency",
        extra_outputs={"remove_dependencies": ["selenium", "playwright"]})
    plan = Router().on_task_completed(
        session=session, completed=diag, tasks=tasks)
    boot = [a.task for a in plan.actions if isinstance(a, CreateTask)][0]
    assert boot.kind is TaskKind.BOOTSTRAP_ENV
    assert boot.inputs["descope_packages"] == ["selenium", "playwright"]


def test_diagnose_regenerate_files_scopes_scaffold_to_targets():
    """regenerate_files -> a SCAFFOLD scoped to exactly the named files."""
    session, diag, tasks = _diagnose_completed(
        minimal_action="regenerate_files",
        extra_outputs={"target_files": ["src/main.py"],
                       "scaffold_artifact_id": "art_scaffold"})
    plan = Router().on_task_completed(
        session=session, completed=diag, tasks=tasks)
    sc = [a.task for a in plan.actions if isinstance(a, CreateTask)][0]
    assert sc.kind is TaskKind.SCAFFOLD
    assert sc.inputs["regenerate_files"] == ["src/main.py"]
    assert sc.inputs["prior_scaffold_artifact_id"] == "art_scaffold"


def test_diagnose_escalate_regenerates_whole_tree():
    """escalate -> today's whole-tree regenerate (no file scoping)."""
    session, diag, tasks = _diagnose_completed(minimal_action="escalate")
    plan = Router().on_task_completed(
        session=session, completed=diag, tasks=tasks)
    sc = [a.task for a in plan.actions if isinstance(a, CreateTask)][0]
    assert sc.kind is TaskKind.SCAFFOLD
    assert not sc.inputs.get("regenerate_files")


def test_diagnose_patch_files_without_source_report_escalates():
    """A malformed patch verdict (no source id) degrades to escalate."""
    session, diag, tasks = _diagnose_completed(minimal_action="patch_files")
    diag.outputs.pop("verify_artifact_id")
    plan = Router().on_task_completed(
        session=session, completed=diag, tasks=tasks)
    sc = [a.task for a in plan.actions if isinstance(a, CreateTask)][0]
    # No REPAIR -- the additive fallback re-scaffolds instead of stranding.
    assert sc.kind is TaskKind.SCAFFOLD


def test_diagnose_add_dependency_empty_list_escalates():
    """add_dependency with no packages degrades to escalate, never stranded."""
    session, diag, tasks = _diagnose_completed(
        minimal_action="add_dependency", extra_outputs={"add_dependencies": []})
    plan = Router().on_task_completed(
        session=session, completed=diag, tasks=tasks)
    sc = [a.task for a in plan.actions if isinstance(a, CreateTask)][0]
    assert sc.kind is TaskKind.SCAFFOLD


def test_diagnose_unknown_action_escalates():
    """An unrecognized (or empty) verdict falls through to escalate."""
    session, diag, tasks = _diagnose_completed(minimal_action="not_a_verdict")
    plan = Router().on_task_completed(
        session=session, completed=diag, tasks=tasks)
    sc = [a.task for a in plan.actions if isinstance(a, CreateTask)][0]
    assert sc.kind is TaskKind.SCAFFOLD


def test_diagnose_escalate_without_scaffold_lineage_fails_session():
    """escalate with no SCAFFOLD ancestor to regenerate fails terminally."""
    session, diag, tasks = _diagnose_completed(
        minimal_action="escalate", with_scaffold=False)
    plan = Router().on_task_completed(
        session=session, completed=diag, tasks=tasks)
    assert [a for a in plan.actions if isinstance(a, CreateTask)] == []
    status = [a for a in plan.actions if isinstance(a, UpdateSessionStatus)]
    assert len(status) == 1
    assert status[0].status is SessionStatus.FAILED


# --------------------- gate -> DIAGNOSE routing ---------------------


@pytest.mark.parametrize(
    "classification",
    ["assertion_drift", "collection_error", "unknown"])
def test_router_verify_reasoning_class_routes_to_diagnose(classification):
    """A reasoning-class VERIFY failure spawns DIAGNOSE, not a bare REPAIR."""
    session = Session.new("g")
    ver = _greenfield_failed_verify(signature="sig1", session=session)
    ver.outputs["classification"] = classification
    plan = Router().on_task_completed(
        session=session, completed=ver, tasks=[ver])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    diag = creates[0].task
    assert diag.kind is TaskKind.DIAGNOSE
    assert diag.inputs["verify_artifact_id"] == "art_verify"
    assert diag.inputs["classification"] == classification
    # The gate charged the round on this edge exactly as REPAIR would.
    assert diag.inputs["repair_attempt"] == 1
    assert diag.parent_task_id == ver.task_id


def test_router_verify_mechanical_class_stays_repair():
    """A mechanical classification keeps the fast path straight to REPAIR."""
    session = Session.new("g")
    ver = _greenfield_failed_verify(signature="sig1", session=session)
    ver.outputs["classification"] = "third_party_import_break"
    plan = Router().on_task_completed(
        session=session, completed=ver, tasks=[ver])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    assert creates[0].task.kind is TaskKind.REPAIR


def test_router_runtime_verify_reasoning_class_routes_to_diagnose():
    """A reasoning-class RUNTIME_VERIFY boot failure spawns DIAGNOSE."""
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    rv = TaskNode.new(
        session.session_id, TaskKind.RUNTIME_VERIFY, "runtime",
        inputs={"mode": SessionMode.GREENFIELD.value})
    rv.produced_artifact_id = "art_runtime"
    rv.outputs = {"outcome": "failed",
                  "failure_signature": "runtime_boot|app.py",
                  "classification": "runtime_failure"}
    rv.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=rv, tasks=[rv])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    diag = creates[0].task
    assert diag.kind is TaskKind.DIAGNOSE
    assert diag.inputs["runtime_artifact_id"] == "art_runtime"
    assert diag.inputs["classification"] == "runtime_failure"
    assert diag.inputs["repair_attempt"] == 1


# --------------------- P3.2: incremental RE_VERIFY (C2) ---------------------


def test_re_verify_kind_registered_and_executor_wired():
    """RE_VERIFY exists with the exact value and a discoverable executor."""
    import cgx.session.tasks  # noqa: F401 (populates the executor registry)
    from cgx.session.tasks.base import get_executor
    assert TaskKind.RE_VERIFY.value == "re_verify"
    assert TaskKind("re_verify") is TaskKind.RE_VERIFY
    assert get_executor(TaskKind.RE_VERIFY) is not None


def test_reverify_markers_only_for_verify_origin():
    """Only a VERIFY-origin diagnosis yields C2 markers; others run full chain."""
    from cgx.session.greenfield_edges import _reverify_markers
    session = Session.new("g")
    diag = TaskNode.new(session.session_id, TaskKind.DIAGNOSE, "d", inputs={})
    diag.outputs = {"smoke_artifact_id": "art_smoke"}
    assert _reverify_markers(diag) == {}
    diag.outputs = {"verify_artifact_id": "art_verify"}
    assert _reverify_markers(diag) == {
        "reverify_origin_gate": "verify", "reverify_report_id": "art_verify"}


def test_diagnose_patch_threads_reverify_markers_to_repair():
    """A VERIFY-origin patch verdict rides the C2 markers on to REPAIR."""
    session, diag, tasks = _diagnose_completed(
        minimal_action="patch_files",
        extra_outputs={"target_files": ["src/x.py"]})
    plan = Router().on_task_completed(
        session=session, completed=diag, tasks=tasks)
    rep = [a.task for a in plan.actions if isinstance(a, CreateTask)][0]
    assert rep.kind is TaskKind.REPAIR
    assert rep.inputs["reverify_origin_gate"] == "verify"
    assert rep.inputs["reverify_report_id"] == "art_verify"


def test_diagnose_add_dependency_threads_reverify_markers_to_bootstrap():
    """A VERIFY-origin add_dependency verdict rides the C2 markers on to boot."""
    session, diag, tasks = _diagnose_completed(
        minimal_action="add_dependency",
        extra_outputs={"add_dependencies": ["flask"]})
    plan = Router().on_task_completed(
        session=session, completed=diag, tasks=tasks)
    boot = [a.task for a in plan.actions if isinstance(a, CreateTask)][0]
    assert boot.kind is TaskKind.BOOTSTRAP_ENV
    assert boot.inputs["reverify_origin_gate"] == "verify"
    assert boot.inputs["reverify_report_id"] == "art_verify"


def test_repair_to_apply_threads_reverify_markers():
    """A REPAIR carrying C2 markers threads them (and scaffold id) on to APPLY."""
    session = Session.new("g")
    rep = TaskNode.new(
        session.session_id, TaskKind.REPAIR, "repair",
        inputs={"mode": "greenfield", "build_artifact_id": "art_build",
                "scaffold_artifact_id": "art_scaffold",
                "reverify_origin_gate": "verify",
                "reverify_report_id": "art_verify"})
    rep.produced_artifact_id = "art_plan"
    rep.outputs = {"can_apply": True, "failure_signature": "s",
                   "repair_attempt": 1}
    rep.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=rep, tasks=[rep])
    apply_t = [a.task for a in plan.actions if isinstance(a, CreateTask)][0]
    assert apply_t.kind is TaskKind.APPLY
    assert apply_t.inputs["reverify_origin_gate"] == "verify"
    assert apply_t.inputs["reverify_report_id"] == "art_verify"
    assert apply_t.inputs["scaffold_artifact_id"] == "art_scaffold"


def test_apply_with_reverify_marker_spawns_re_verify():
    """A DIAGNOSE-scoped patch's APPLY splices RE_VERIFY, not BOOTSTRAP/VERIFY."""
    session = Session.new("g")
    apply_t = TaskNode.new(
        session.session_id, TaskKind.APPLY, "apply",
        inputs={"mode": "greenfield", "build_artifact_id": "art_build",
                "scaffold_artifact_id": "art_scaffold",
                "reverify_origin_gate": "verify",
                "reverify_report_id": "art_verify"})
    apply_t.produced_artifact_id = "art_applied"
    apply_t.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=apply_t, tasks=[apply_t])
    rv = [a.task for a in plan.actions if isinstance(a, CreateTask)][0]
    assert rv.kind is TaskKind.RE_VERIFY
    assert rv.inputs["reverify_report_id"] == "art_verify"
    assert rv.inputs["build_artifact_id"] == "art_build"
    assert rv.inputs["apply_artifact_id"] == "art_applied"


def test_bootstrap_with_reverify_marker_spawns_re_verify():
    """A DIAGNOSE dependency fix's BOOTSTRAP_ENV splices RE_VERIFY, not API_CHECK."""
    session = Session.new("g")
    boot = TaskNode.new(
        session.session_id, TaskKind.BOOTSTRAP_ENV, "bootstrap",
        inputs={"mode": "greenfield", "apply_artifact_id": "art_applied",
                "scaffold_artifact_id": "art_scaffold",
                "reverify_origin_gate": "verify",
                "reverify_report_id": "art_verify"})
    boot.produced_artifact_id = "art_build"
    boot.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=boot, tasks=[boot])
    rv = [a.task for a in plan.actions if isinstance(a, CreateTask)][0]
    assert rv.kind is TaskKind.RE_VERIFY
    assert rv.inputs["build_artifact_id"] == "art_build"
    assert rv.inputs["apply_artifact_id"] == "art_applied"
    assert rv.inputs["reverify_report_id"] == "art_verify"


def test_re_verify_pass_hands_off_like_verify():
    """A green RE_VERIFY dispatches to RUNTIME_VERIFY exactly like VERIFY."""
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    rv = TaskNode.new(
        session.session_id, TaskKind.RE_VERIFY, "reverify",
        inputs={"mode": SessionMode.GREENFIELD.value})
    rv.produced_artifact_id = "art_reverify"
    rv.outputs = {"outcome": "passed"}
    rv.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=rv, tasks=[rv])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    assert creates[0].task.kind is TaskKind.RUNTIME_VERIFY


def test_re_verify_still_failing_reasoning_class_routes_to_diagnose():
    """A still-failing reasoning-class RE_VERIFY routes back to DIAGNOSE."""
    from cgx.session.models import SessionMode
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    rv = TaskNode.new(
        session.session_id, TaskKind.RE_VERIFY, "reverify",
        inputs={"mode": SessionMode.GREENFIELD.value})
    rv.produced_artifact_id = "art_reverify"
    rv.outputs = {"outcome": "assertions_failed",
                  "failure_signature": "sig_new",
                  "classification": "assertion_drift",
                  "returncode": 1, "tests_selected_count": 3,
                  "failing_count": 1}
    rv.status = TaskNodeStatus.DONE
    plan = Router().on_task_completed(
        session=session, completed=rv, tasks=[rv])
    creates = [a for a in plan.actions if isinstance(a, CreateTask)]
    assert len(creates) == 1
    diag = creates[0].task
    assert diag.kind is TaskKind.DIAGNOSE
    assert diag.inputs["verify_artifact_id"] == "art_reverify"


def test_re_verify_failing_test_files_selects_only_failed():
    """_failing_test_files narrows the selected set to the failing file(s)."""
    from cgx.session.tasks.re_verify import _failing_test_files
    content = {
        "tests_selected": ["tests/test_a.py", "tests/test_b.py"],
        "failures": [{"nodeid": "tests.test_b.TestX::test_it"}]}
    assert _failing_test_files(content) == ["tests/test_b.py"]


def test_re_verify_failing_test_files_falls_back_to_all_when_unresolved():
    """No resolvable failure file -> re-run the full selected set, never zero."""
    from cgx.session.tasks.re_verify import _failing_test_files
    content = {"tests_selected": ["tests/test_a.py"], "failures": []}
    assert _failing_test_files(content) == ["tests/test_a.py"]


# ---------------------------------------------------------------------
# Regression harness for session ``ses_fa6f72a9d3da4217`` (Calculator2).
#
# Each test encodes an input shape taken from that run's trace so the fix
# is proven against the real failure rather than a reconstruction.
# ---------------------------------------------------------------------

class _RecordingProvider:
    """Provider stub that records every prompt and replays canned text."""

    def __init__(self, replies=None):
        self.prompts: List[str] = []
        self._replies = list(replies or [])

    def chat(self, messages, force_json=False, **kwargs):
        self.prompts.append(messages[-1]["content"])
        return {"content": self._replies.pop(0) if self._replies
                else "import os\n"}


def test_ast_fallback_prompt_carries_the_project_goal(store):
    """The AST fallback must never prompt with an empty project goal.

    The router threads ``prior_goal``; the executor read ``composed_goal``,
    so every fallback prompt had a blank goal and the model invented a
    Flask app inside a FastAPI project.
    """
    from cgx.session.models import SessionMode
    from cgx.session.tasks.ast_scaffold import run_ast_scaffold
    session = Session.new("create a calculator app using fast-api and "
                          "react frontend", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    work_plan = Artifact.new(
        session_id=session.session_id, produced_by_task_id="t_plan",
        kind=ArtifactKind.WORK_PLAN,
        content={"contracts": {"project_skeleton":
                               "# --- tests/test_core.py ---\n"
                               "def test_compute():\n    ...\n"}})
    store.save_artifact(work_plan)
    provider = _RecordingProvider(
        ["import pytest\n", "def test_compute():\n    assert True\n"])
    task = TaskNode.new(
        session.session_id, TaskKind.AST_REGENERATE, "regenerate",
        inputs={"work_plan_artifact_id": work_plan.artifact_id,
                "prior_goal": ("create a calculator app using fast-api "
                               "and react frontend"),
                "regenerate_attempt": 2,
                "regenerate_files": ["tests/test_core.py"]})
    run_ast_scaffold(task, ExecutorDeps(store=store, provider=provider))
    assert provider.prompts, "the fallback issued no prompts"
    for prompt in provider.prompts:
        assert "create a calculator app using fast-api" in prompt, prompt
    # The header prompt must carry the file's skeleton as context; it was
    # hardcoded to "" while only the symbol prompt got the skeleton.
    assert "def test_compute" in provider.prompts[0]


def test_ast_fallback_rejects_empty_output(store):
    """The fallback must gate its own output before APPLY writes it.

    Prose replies leave the assembler with nothing to unparse, and the
    1-byte file that came out still shipped as ``generated``.
    """
    from cgx.session.models import SessionMode
    from cgx.session.tasks.ast_scaffold import run_ast_scaffold
    session = Session.new("g", mode=SessionMode.GREENFIELD)
    store.save_session(session)
    work_plan = Artifact.new(
        session_id=session.session_id, produced_by_task_id="t_plan",
        kind=ArtifactKind.WORK_PLAN,
        content={"contracts": {"project_skeleton":
                               "# --- tests/test_core.py ---\n"}})
    store.save_artifact(work_plan)
    provider = _RecordingProvider(["Sure! Here is what I would write."] * 3)
    task = TaskNode.new(
        session.session_id, TaskKind.AST_REGENERATE, "regenerate",
        inputs={"work_plan_artifact_id": work_plan.artifact_id,
                "prior_goal": "g",
                "regenerate_files": ["tests/test_core.py"]})
    result = run_ast_scaffold(
        task, ExecutorDeps(store=store, provider=provider))
    content = result.artifact.content
    assert content["generated"] == []
    assert [f["file"] for f in content["failed"]] == ["tests/test_core.py"]


def test_api_check_probe_resolves_lazy_submodules(tmp_path):
    """``from jose import jwt`` works but ``hasattr(jose, 'jwt')`` is False.

    The probe used ``hasattr`` alone, so any lazily-bound submodule was
    reported as a hallucinated attribute. ``concurrent.futures`` is the
    stdlib-only reproduction of the same shape.
    """
    import concurrent
    import sys
    from cgx.session.tasks.api_check import _probe_references
    assert not hasattr(concurrent, "futures"), (
        "precondition: concurrent.futures must be lazily bound")
    rows, probe_error = _probe_references(
        sys.executable,
        [("concurrent", "futures"), ("json", "no_such_symbol")],
        30.0, tmp_path)
    assert probe_error is None
    by_name = {(r["module"], r["name"]): r for r in rows}
    assert by_name[("concurrent", "futures")]["ok"] is True
    # A genuinely absent attribute must still fail, message intact.
    absent = by_name[("json", "no_such_symbol")]
    assert absent["ok"] is False
    assert "AttributeError" in absent["error"]


def test_bootstrap_unresolved_requested_roots_are_reported():
    """A root ``install_deps`` asked for that is *still* unimportable.

    This is the loop that killed the session: API_CHECK classified the
    hallucinated ``app`` module as ``missing_dependency``, BOOTSTRAP_ENV
    installed nothing, and the re-probe produced a byte-identical
    signature. Roots that survive an install round unimportable are
    recorded as ``uninstallable`` so API_CHECK calls them hallucinations.
    """
    import sys
    from cgx.session.tasks.bootstrap_env import _unresolved_requested_roots
    unresolved = _unresolved_requested_roots(
        ["json", "cgx_definitely_not_a_real_module"], sys.executable)
    assert unresolved == ["cgx_definitely_not_a_real_module"]


def test_task_from_json_tolerates_an_unknown_kind():
    """A legacy row must not 500 the snapshot endpoint forever.

    One persisted task with ``kind='agentic_repair'`` -- a value dropped
    from the enum -- made every GET on the session raise ValueError.
    """
    import json as _json
    from cgx.session.store import _task_from_json
    blob = _json.dumps({
        "task_id": "task_legacy", "session_id": "ses_legacy",
        "kind": "agentic_repair", "name": "legacy", "description": "",
        "status": "done", "inputs": {}, "created_at": 1.0})
    node = _task_from_json(blob)
    assert node.task_id == "task_legacy"
    assert node.kind is TaskKind.UNKNOWN
