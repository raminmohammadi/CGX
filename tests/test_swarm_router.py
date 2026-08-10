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
