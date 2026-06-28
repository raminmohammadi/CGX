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
                      goal=None):
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
                "mode": task.inputs.get("mode") or "greenfield",
                "changed_files": ["app.py"],
                "ran": False, "tests_passed": False,
                "returncode": 0, "tests_selected": [],
                "stdout": "", "stderr": "",
                "skipped_reason": "no tests discovered",
            })
        return ExecutorResult(
            outputs={"verify_artifact_id": artifact.artifact_id,
                     "ran": False, "tests_passed": False,
                     "tests_selected_count": 0},
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

    # 8. APPLY runs -> spawns VERIFY.
    runner.run_next(session_id=session.session_id, deps=ExecutorDeps())
    verify_t = next(t for t in store.list_tasks(session.session_id)
                    if t.kind is TaskKind.VERIFY)
    assert verify_t.status is TaskNodeStatus.READY

    # 9. VERIFY runs -> terminal.
    runner.run_next(session_id=session.session_id, deps=ExecutorDeps())
    verify_after = store.get_task(verify_t.task_id)
    assert verify_after.status is TaskNodeStatus.DONE
    assert runner.run_next(
        session_id=session.session_id, deps=ExecutorDeps()) is None

    # All five greenfield artifacts present + clean separation from
    # the explore-loop kinds.
    kinds = {a.kind for a in store.list_artifacts(session.session_id)}
    assert ArtifactKind.REQUIREMENTS_SHEET in kinds
    assert ArtifactKind.WORK_PLAN in kinds
    assert ArtifactKind.SCAFFOLD_PATCHES in kinds
    assert ArtifactKind.APPLIED_CHANGES in kinds
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
