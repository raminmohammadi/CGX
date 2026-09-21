"""Phase 2/3: the swarm router edges + terminal session actions.

The Tech Lead's validated plan spawns one Developer task per file; each
Developer spawns the next until every file is attempted, threading a
``file_index`` cursor and an accumulating ``failed_paths`` list. The last
Developer hands off to a single SWARM_VERIFY task, which owns the session's
terminal status -- COMPLETED only when the tree verified clean, FAILED
otherwise (a planning dead-end or a crashed executor also ends FAILED).
"""

from cgx.session.actions import CreateTask, UpdateSessionStatus
from cgx.session.models import (
    Session, SessionMode, SessionStatus, TaskKind, TaskNode)
from cgx.session.router import (
    Router, _swarm_developer_to_successors, _swarm_tech_lead_to_successors,
    _swarm_verify_to_successors)


def _tech_lead(outputs):
    t = TaskNode.new(session_id="s", kind=TaskKind.SWARM_TECH_LEAD,
                     name="plan", inputs={"goal": "g", "project_root": "/p"})
    t.outputs = outputs
    return t


def _developer(outputs):
    t = TaskNode.new(session_id="s", kind=TaskKind.SWARM_DEVELOPER, name="dev")
    t.outputs = outputs
    return t


def _verify(outputs):
    t = TaskNode.new(session_id="s", kind=TaskKind.SWARM_VERIFY, name="verify")
    t.outputs = outputs
    return t


def _statuses(plan):
    return [a.status for a in plan.actions if isinstance(a, UpdateSessionStatus)]


def _created(plan):
    return [a.task for a in plan.actions if isinstance(a, CreateTask)]


def test_tech_lead_spawns_first_developer():
    parent = _tech_lead({"work_plan_artifact_id": "art_1", "file_count": 3,
                         "swarm_paths": ["a.py", "b.py", "c.py"], "goal": "g",
                         "project_root": "/p"})
    kids = _swarm_tech_lead_to_successors(parent)
    assert len(kids) == 1
    dev = kids[0]
    assert dev.kind is TaskKind.SWARM_DEVELOPER
    assert dev.inputs["file_index"] == 0
    assert dev.inputs["work_plan_artifact_id"] == "art_1"
    assert dev.inputs["file_count"] == 3
    assert dev.inputs["failed_paths"] == []


def test_tech_lead_with_no_files_spawns_nothing():
    assert _swarm_tech_lead_to_successors(_tech_lead({"file_count": 0})) == []


def test_developer_spawns_next_until_last():
    parent = _developer({"work_plan_artifact_id": "art_1", "file_index": 0,
                         "file_count": 3, "failed_paths": ["a.py"],
                         "goal": "g", "project_root": "/p"})
    kids = _swarm_developer_to_successors(parent)
    assert len(kids) == 1
    assert kids[0].inputs["file_index"] == 1
    # failed_paths accumulates down the chain.
    assert kids[0].inputs["failed_paths"] == ["a.py"]


def test_developer_last_file_spawns_verify():
    parent = _developer({"work_plan_artifact_id": "art_1", "file_index": 2,
                         "file_count": 3, "failed_paths": ["b.py"],
                         "goal": "g", "project_root": "/p"})
    kids = _swarm_developer_to_successors(parent)
    assert len(kids) == 1
    verify = kids[0]
    assert verify.kind is TaskKind.SWARM_VERIFY
    assert verify.inputs["work_plan_artifact_id"] == "art_1"
    # failed_paths threads through to the tree-level verification stage.
    assert verify.inputs["failed_paths"] == ["b.py"]


def test_verify_is_terminal():
    assert _swarm_verify_to_successors(_verify({"verify_ok": True})) == []


def _session():
    return Session.new("obj", mode=SessionMode.SWARM)


def test_completed_verify_clean_completes_session():
    session = _session()
    completed = _verify({"verify_ok": True, "failed_paths": []})
    plan = Router().on_task_completed(
        session=session, completed=completed, tasks=[completed])
    assert _created(plan) == []
    assert _statuses(plan) == [SessionStatus.COMPLETED]


def test_completed_verify_not_ok_fails_session():
    session = _session()
    completed = _verify({"verify_ok": False, "coverage_gaps": ["b.py"]})
    plan = Router().on_task_completed(
        session=session, completed=completed, tasks=[completed])
    assert _statuses(plan) == [SessionStatus.FAILED]


def test_completed_developer_last_file_spawns_verify_no_terminal():
    session = _session()
    completed = _developer({"work_plan_artifact_id": "art_1", "file_index": 2,
                            "file_count": 3, "failed_paths": []})
    plan = Router().on_task_completed(
        session=session, completed=completed, tasks=[completed])
    created = _created(plan)
    assert len(created) == 1 and created[0].kind is TaskKind.SWARM_VERIFY
    assert _statuses(plan) == []


def test_completed_tech_lead_dead_end_fails_session():
    session = _session()
    completed = _tech_lead({"file_count": 0})
    plan = Router().on_task_completed(
        session=session, completed=completed, tasks=[completed])
    assert _created(plan) == []
    assert _statuses(plan) == [SessionStatus.FAILED]


def test_completed_developer_midchain_has_no_terminal_status():
    session = _session()
    completed = _developer({"work_plan_artifact_id": "art_1", "file_index": 0,
                            "file_count": 3, "failed_paths": []})
    plan = Router().on_task_completed(
        session=session, completed=completed, tasks=[completed])
    assert len(_created(plan)) == 1
    assert _statuses(plan) == []


def test_failed_swarm_task_fails_session():
    session = _session()
    failed = _developer({"file_index": 1, "file_count": 3})
    plan = Router().on_task_failed(
        session=session, failed=failed, tasks=[failed])
    assert _statuses(plan) == [SessionStatus.FAILED]


def test_completed_verify_partial_build_attaches_summary_error():
    from cgx.session.actions import UpdateTaskStatus
    session = _session()
    completed = _verify({
        "verify_ok": False,
        "summary": "partial build: built 14/15 files; tests failed: test_total_area",
    })
    plan = Router().on_task_completed(
        session=session, completed=completed, tasks=[completed])
    errs = [a for a in plan.actions if isinstance(a, UpdateTaskStatus)]
    assert len(errs) == 1 and "partial build" in errs[0].error
    assert SessionStatus.FAILED in _statuses(plan)


# --------------------- C2: question vs build intent routing ---------------------

def test_is_question_classifier():
    from cgx.session.mode import is_question
    assert is_question("how does the auth flow work?")
    assert is_question("Explain the retrieval pipeline")
    assert is_question("what is the entrypoint")
    assert not is_question("build a flask api")
    assert not is_question("add rate limiting to the webhook")
    assert not is_question("")
    # a build verb wins even with a trailing question mark
    assert not is_question("build a chatbot?")


def test_swarm_question_routes_to_explore_root():
    from cgx.session.models import TaskKind
    plan = Router().on_user_message(
        session=_session(), message="how does the login work?", tasks=[])
    created = _created(plan)
    assert len(created) == 1 and created[0].kind is TaskKind.EXPLORE


def test_swarm_build_routes_to_tech_lead_root():
    from cgx.session.models import TaskKind
    plan = Router().on_user_message(
        session=_session(), message="build a REST API for todos", tasks=[])
    created = _created(plan)
    assert len(created) == 1 and created[0].kind is TaskKind.SWARM_TECH_LEAD


# --------------------- plan-approval gate (Phase C) ---------------------

def test_tech_lead_gates_on_plan_approval_when_required():
    from cgx.session.models import DecisionKind
    parent = _tech_lead({"work_plan_artifact_id": "art_1", "file_count": 2,
                         "swarm_paths": ["a.py", "b.py"], "goal": "g",
                         "project_root": "/p"})
    parent.inputs["require_plan_approval"] = True
    kids = _swarm_tech_lead_to_successors(parent)
    assert len(kids) == 1
    ask = kids[0]
    assert ask.kind is TaskKind.ASK_USER
    assert ask.inputs["expected_kind"] == DecisionKind.APPROVE_PLAN.value
    seed = ask.inputs.get("swarm_dev_seed")
    assert isinstance(seed, dict) and seed["file_index"] == 0


def test_tech_lead_no_gate_when_not_required():
    parent = _tech_lead({"work_plan_artifact_id": "art_1", "file_count": 2,
                         "swarm_paths": ["a.py", "b.py"], "goal": "g",
                         "project_root": "/p"})  # require_plan_approval absent
    kids = _swarm_tech_lead_to_successors(parent)
    assert kids and kids[0].kind is TaskKind.SWARM_DEVELOPER


def test_approve_plan_swarm_seed_spawns_first_developer():
    from cgx.session.models import Decision, DecisionKind
    from cgx.session.router import _from_approve_plan
    seed = {"work_plan_artifact_id": "art_1", "file_index": 0, "file_count": 2,
            "failed_paths": [], "goal": "g", "project_root": "/p"}
    ask = TaskNode.new(session_id="s", kind=TaskKind.ASK_USER, name="approve",
                       inputs={"expected_kind": DecisionKind.APPROVE_PLAN.value,
                               "swarm_dev_seed": seed})
    dec = Decision.new(session_id="s", resolved_task_id=ask.task_id,
                       kind=DecisionKind.APPROVE_PLAN, question="approve?",
                       chosen={"approved": True})
    succ = _from_approve_plan(ask, dec)
    assert succ is not None and succ.kind is TaskKind.SWARM_DEVELOPER
    assert succ.inputs["file_index"] == 0


def test_decline_plan_swarm_spawns_no_successor():
    from cgx.session.models import Decision, DecisionKind
    from cgx.session.router import _from_approve_plan
    ask = TaskNode.new(session_id="s", kind=TaskKind.ASK_USER, name="approve",
                       inputs={"expected_kind": DecisionKind.APPROVE_PLAN.value,
                               "swarm_dev_seed": {"file_index": 0}})
    dec = Decision.new(session_id="s", resolved_task_id=ask.task_id,
                       kind=DecisionKind.APPROVE_PLAN, question="approve?",
                       chosen={"approved": False})
    assert _from_approve_plan(ask, dec) is None


# --------------------- existing-repo relevance routing (Phase C) ---------------------

def _assess(outputs, inputs=None):
    t = TaskNode.new(session_id="s", kind=TaskKind.SWARM_ASSESS, name="assess",
                     inputs=inputs or {"goal": "g", "project_root": "/p"})
    t.outputs = outputs
    return t


def test_assess_relevant_spawns_tech_lead():
    from cgx.session.router import _swarm_assess_to_successors
    kids = _swarm_assess_to_successors(
        _assess({"relevant": True, "goal": "g", "project_root": "/p"}))
    assert len(kids) == 1 and kids[0].kind is TaskKind.SWARM_TECH_LEAD
    assert kids[0].inputs["project_root"] == "/p"


def test_assess_irrelevant_asks_relocate():
    from cgx.session.models import DecisionKind
    from cgx.session.router import _swarm_assess_to_successors
    kids = _swarm_assess_to_successors(
        _assess({"relevant": False, "reason": "unrelated", "goal": "g",
                 "project_root": "/p"}))
    assert len(kids) == 1 and kids[0].kind is TaskKind.ASK_USER
    assert kids[0].inputs["expected_kind"] == DecisionKind.RELOCATE.value
    assert kids[0].inputs["current_project_root"] == "/p"


def test_relocate_new_path_reroots_tech_lead():
    from cgx.session.models import Decision, DecisionKind
    from cgx.session.router import _from_relocate
    ask = TaskNode.new(session_id="s", kind=TaskKind.ASK_USER, name="reloc",
                       inputs={"expected_kind": DecisionKind.RELOCATE.value,
                               "goal": "g", "current_project_root": "/old"})
    dec = Decision.new(session_id="s", resolved_task_id=ask.task_id,
                       kind=DecisionKind.RELOCATE, question="?",
                       chosen={"path": "/new"})
    succ = _from_relocate(ask, dec)
    assert succ.kind is TaskKind.SWARM_TECH_LEAD
    assert succ.inputs["project_root"] == "/new"


def test_relocate_proceed_here_uses_current_root():
    from cgx.session.models import Decision, DecisionKind
    from cgx.session.router import _from_relocate
    ask = TaskNode.new(session_id="s", kind=TaskKind.ASK_USER, name="reloc",
                       inputs={"expected_kind": DecisionKind.RELOCATE.value,
                               "goal": "g", "current_project_root": "/old"})
    dec = Decision.new(session_id="s", resolved_task_id=ask.task_id,
                       kind=DecisionKind.RELOCATE, question="?",
                       chosen={"proceed_here": True})
    assert _from_relocate(ask, dec).inputs["project_root"] == "/old"


def test_make_root_swarm_existing_repo_assesses(tmp_path):
    (tmp_path / "main.py").write_text("print(1)\n")
    session = Session.new("add a feature", mode=SessionMode.SWARM,
                          project_root=str(tmp_path))
    plan = Router().on_user_message(
        session=session, message="add a feature", tasks=[])
    created = _created(plan)
    assert len(created) == 1 and created[0].kind is TaskKind.SWARM_ASSESS


def test_make_root_swarm_empty_repo_builds_directly(tmp_path):
    session = Session.new("build an app", mode=SessionMode.SWARM,
                          project_root=str(tmp_path))  # empty dir
    plan = Router().on_user_message(
        session=session, message="build an app", tasks=[])
    created = _created(plan)
    assert len(created) == 1 and created[0].kind is TaskKind.SWARM_TECH_LEAD


def test_snapshot_repo_lists_source_and_skips_ignored(tmp_path):
    from cgx.session.tasks.swarm_assess import snapshot_repo
    (tmp_path / "a.py").write_text("x")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "cfg").write_text("x")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "b.js").write_text("x")
    files = snapshot_repo(str(tmp_path))
    assert "a.py" in files
    assert not any("node_modules" in f or ".git" in f for f in files)
