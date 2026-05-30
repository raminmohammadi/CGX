"""Tests for the multi-agent orchestration layer.

Covers the deterministic (no-LLM) fallback path of the Planner, the
Tracker state machine including capability dispatch + failure handling,
and the Judge's structural / LLM-mediated verdicts.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, List

import pytest

from cgx.agents import Judge, Planner, Tracker, run_agent
from cgx.agents.types import AgentEvent, Plan, Task, TaskKind, TaskStatus


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
def test_planner_fallback_change_intent_emits_plan_apply_verify_chain():
    plan = Planner(provider=None).plan("Add a CSV export function to reports")
    assert isinstance(plan, Plan)
    kinds = [t.kind for t in plan.tasks]
    assert kinds == [TaskKind.PLAN, TaskKind.APPLY, TaskKind.VERIFY]
    assert plan.tasks[0].criteria, "fallback plan task should have criteria"
    assert plan.tasks[1].criteria, "apply task should have criteria"
    assert plan.tasks[2].criteria, "verify task should have criteria"


def test_planner_fallback_question_intent_emits_ask_task():
    plan = Planner(provider=None).plan("What does parse_codebase do?")
    assert len(plan.tasks) == 1
    assert plan.tasks[0].kind == TaskKind.ASK


def test_planner_rejects_empty_goal():
    with pytest.raises(ValueError):
        Planner(provider=None).plan("")


def test_planner_fallback_verify_only_goal_emits_standalone_verify():
    plan = Planner(provider=None).plan("Do the tests pass?")
    kinds = [t.kind for t in plan.tasks]
    assert kinds == [TaskKind.VERIFY]
    assert plan.tasks[0].criteria


def test_planner_fallback_run_tests_phrase_emits_verify():
    plan = Planner(provider=None).plan("Run the tests and report results")
    kinds = [t.kind for t in plan.tasks]
    assert kinds == [TaskKind.VERIFY]


def test_planner_ensure_verb_routed_as_change_goal():
    # 'ensure' is now a change verb so QA goals that imply edits + tests
    # take the [plan, apply, verify] chain rather than ASK or VERIFY alone.
    plan = Planner(provider=None).plan(
        "Ensure all tests have the parameters they need"
    )
    kinds = [t.kind for t in plan.tasks]
    assert kinds == [TaskKind.PLAN, TaskKind.APPLY, TaskKind.VERIFY]


class _StubProvider:
    """Minimal LLMProvider stub recording chat() calls and returning a script."""

    def __init__(self, replies: List[Dict[str, Any]]) -> None:
        self.replies = list(replies)
        self.calls: List[Dict[str, Any]] = []

    def chat(self, *, messages, **kw):  # noqa: ANN001
        self.calls.append({"messages": messages, **kw})
        if not self.replies:
            return {"content": "", "error": "no-more-replies"}
        return self.replies.pop(0)


def test_planner_uses_llm_response_when_valid_json():
    provider = _StubProvider([{
        "content": json.dumps({"tasks": [
            {"description": "Locate report module", "kind": "search", "criteria": ["finds reports.py"]},
            {"description": "Add CSV exporter", "kind": "plan", "criteria": ["diff present"]},
        ]}),
    }])
    plan = Planner(provider=provider).plan("Add CSV export to reports")
    # Change goals get the apply+verify pair appended after the final plan.
    assert [t.kind for t in plan.tasks] == [
        TaskKind.SEARCH, TaskKind.PLAN, TaskKind.APPLY, TaskKind.VERIFY,
    ]
    assert plan.tasks[1].criteria == ["diff present"]


def test_planner_falls_back_when_llm_returns_garbage():
    provider = _StubProvider([{"content": "not json at all"}])
    plan = Planner(provider=provider).plan("Explain main()")
    assert len(plan.tasks) == 1
    assert plan.tasks[0].kind == TaskKind.ASK  # deterministic fallback path


def test_planner_caps_task_count():
    big = [{"description": f"step {i}", "kind": "ask"} for i in range(20)]
    provider = _StubProvider([{"content": json.dumps({"tasks": big})}])
    plan = Planner(provider=provider, max_tasks=3).plan("do many things")
    assert len(plan.tasks) == 3


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------
def _run_with(plan: Plan, caps, judge=None, stop_on_fail=True):
    events: List[AgentEvent] = []
    for ev in Tracker(caps, judge=judge, stop_on_fail=stop_on_fail).stream(plan):
        events.append(ev)
    return events


def test_tracker_runs_tasks_and_emits_events_in_order():
    plan = Plan(goal="g", tasks=[
        Task(description="q1", kind=TaskKind.ASK),
        Task(description="q2", kind=TaskKind.ASK),
    ])
    calls: List[str] = []

    def ask(q, **_):
        calls.append(q)
        return {"answer_md": f"answer for {q}"}

    events = _run_with(plan, {"ask": ask})
    kinds = [e.type for e in events]
    assert kinds[0] == "plan"
    assert kinds[-1] == "summary"
    assert calls == ["q1", "q2"]
    assert all(t.status == TaskStatus.DONE for t in plan.tasks)


def test_tracker_missing_capability_skips_task():
    plan = Plan(goal="g", tasks=[Task(description="x", kind=TaskKind.PLAN)])
    events = _run_with(plan, {})
    assert plan.tasks[0].status == TaskStatus.SKIPPED
    assert any(e.type == "task_skipped" for e in events)


def test_tracker_failure_halts_remaining_tasks_when_stop_on_fail():
    plan = Plan(goal="g", tasks=[
        Task(description="boom", kind=TaskKind.ASK),
        Task(description="never", kind=TaskKind.ASK),
    ])

    def ask(q, **_):
        if q == "boom":
            raise RuntimeError("kaboom")
        return {"answer_md": "ok"}

    events = _run_with(plan, {"ask": ask}, stop_on_fail=True)
    assert plan.tasks[0].status == TaskStatus.FAILED
    assert plan.tasks[1].status == TaskStatus.SKIPPED
    assert any(e.type == "task_failed" for e in events)


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------
def test_judge_structural_fail_when_plan_task_has_no_diff():
    # Hard-fail only when both plan_md AND diffs are absent.
    task = Task(description="t", kind=TaskKind.PLAN, criteria=["diff exists"],
                output={})
    v = Judge(provider=None).judge(task)
    assert v.verdict == "fail" and "diff" in v.rationale.lower()


def test_judge_softpass_plan_task_with_plan_md_but_no_diff():
    # plan_md present but no diffs → soft-pass (no LLM provider to judge further).
    task = Task(description="t", kind=TaskKind.PLAN, criteria=["provides a plan"],
                output={"plan_md": "1. identify tests\n2. fix placeholders"})
    v = Judge(provider=None).judge(task)
    assert v.verdict == "pass" and v.confidence <= 0.6


def test_judge_passes_ask_with_nonempty_answer_and_no_llm():
    task = Task(description="t", kind=TaskKind.ASK, criteria=["answers the question"],
                output={"answer_md": "Here is an answer with sources."})
    v = Judge(provider=None).judge(task)
    assert v.verdict == "pass"


def test_run_agent_end_to_end_no_llm_with_injected_capabilities():
    captured: Dict[str, Any] = {}

    def ask(q, **_):
        captured["q"] = q
        return {"answer_md": "stubbed"}

    plan = run_agent("Explain foo()",
                     capabilities={"ask": ask}, planner=Planner(provider=None))
    assert isinstance(plan, Plan)
    assert plan.tasks[0].status == TaskStatus.DONE
    assert captured["q"] == "Explain foo()"


# ---------------------------------------------------------------------------
# Planner kind-policy + name derivation
# ---------------------------------------------------------------------------
def test_planner_downgrades_plan_task_for_readonly_goal():
    provider = _StubProvider([{
        "content": json.dumps({"tasks": [
            {"description": "Locate vae.py", "kind": "search"},
            {"description": "Write a summary of vae.py", "kind": "plan",
             "criteria": ["diff present"]},
        ]}),
    }])
    plan = Planner(provider=provider).plan(
        "Summarize what src/perception/vae.py does in two short sentences.")
    kinds = [t.kind for t in plan.tasks]
    # The second task must be coerced away from PLAN since the goal is
    # purely explanatory; SEARCH stays untouched.
    assert TaskKind.PLAN not in kinds
    assert kinds == [TaskKind.SEARCH, TaskKind.ASK]
    downgraded = plan.tasks[1]
    assert downgraded.description == "Write a summary of vae.py"
    assert downgraded.criteria, "downgraded task should retain or seed criteria"


def test_planner_keeps_plan_task_for_change_goal():
    provider = _StubProvider([{
        "content": json.dumps({"tasks": [
            {"description": "Locate auth module", "kind": "search"},
            {"description": "Add OAuth callback handler", "kind": "plan",
             "criteria": ["diff present"]},
        ]}),
    }])
    plan = Planner(provider=provider).plan("Add an OAuth callback handler")
    assert [t.kind for t in plan.tasks] == [
        TaskKind.SEARCH, TaskKind.PLAN, TaskKind.APPLY, TaskKind.VERIFY,
    ]


def test_planner_uses_llm_name_when_present():
    provider = _StubProvider([{
        "content": json.dumps({"tasks": [
            {"name": "Find auth module", "description": "Locate auth-related files",
             "kind": "search"},
        ]}),
    }])
    plan = Planner(provider=provider).plan("Where is auth handled?")
    assert plan.tasks[0].name == "Find auth module"
    assert plan.tasks[0].description == "Locate auth-related files"


def test_planner_derives_name_from_description_when_missing():
    plan = Planner(provider=None).plan("Explain how parse_codebase works")
    t = plan.tasks[0]
    assert t.name, "fallback plan should derive a non-empty name"
    assert t.name.startswith("Explain how parse_codebase")


# ---------------------------------------------------------------------------
# Tracker progress events
# ---------------------------------------------------------------------------
def test_tracker_emits_task_progress_for_slow_capability():
    plan = Plan(goal="g", tasks=[Task(description="slow", kind=TaskKind.ASK)])

    started = threading.Event()
    release = threading.Event()

    def slow_ask(_q, **_):
        started.set()
        # Block long enough to trigger at least one heartbeat.
        release.wait(timeout=2.0)
        return {"answer_md": "done"}

    events: List[AgentEvent] = []
    tracker = Tracker({"ask": slow_ask}, progress_interval=0.05)
    stream = tracker.stream(plan)

    # Drain events on a background thread so we can release the capability
    # mid-flight once we've seen a heartbeat.
    def _drain() -> None:
        for ev in stream:
            events.append(ev)

    drainer = threading.Thread(target=_drain, daemon=True)
    drainer.start()
    assert started.wait(timeout=2.0), "capability never started"
    # Wait until at least one task_progress event lands.
    deadline = time.time() + 2.0
    while time.time() < deadline:
        if any(e.type == "task_progress" for e in events):
            break
        time.sleep(0.05)
    release.set()
    drainer.join(timeout=2.0)
    assert not drainer.is_alive(), "tracker stream did not complete"

    progress = [e for e in events if e.type == "task_progress"]
    assert progress, "expected at least one task_progress event"
    payload = progress[0].payload
    assert payload["task_id"] == plan.tasks[0].id
    assert isinstance(payload["elapsed"], float) and payload["elapsed"] >= 0
    # Sanity: the run must still terminate with task_done + summary.
    kinds = [e.type for e in events]
    assert kinds[-1] == "summary"
    assert "task_done" in kinds
    assert plan.tasks[0].status == TaskStatus.DONE


def test_tracker_progress_disabled_when_interval_non_positive():
    plan = Plan(goal="g", tasks=[Task(description="q", kind=TaskKind.ASK)])

    def ask(_q, **_):
        return {"answer_md": "ok"}

    events = list(Tracker({"ask": ask}, progress_interval=0).stream(plan))
    assert not any(e.type == "task_progress" for e in events)
    assert plan.tasks[0].status == TaskStatus.DONE


# ---------------------------------------------------------------------------
# Tracker dispatch for APPLY / VERIFY
# ---------------------------------------------------------------------------
def test_tracker_dispatches_apply_and_verify_with_prior_outputs():
    diffs = [{"file": "a.py", "patch": "--- a/a.py\n+++ b/a.py\n@@\n+x = 1\n"}]
    plan = Plan(goal="g", tasks=[
        Task(description="generate diff", kind=TaskKind.PLAN),
        Task(description="write to disk", kind=TaskKind.APPLY),
        Task(description="run tests", kind=TaskKind.VERIFY),
    ])

    received_apply: Dict[str, Any] = {}
    received_verify: Dict[str, Any] = {}

    def plan_cap(_text, **_):
        return {"plan_md": "...", "diffs": diffs}

    def apply_cap(prior, **_):
        received_apply["prior"] = prior
        return {"applied_files": ["a.py"], "failed_files": [],
                "backup_dir": "/tmp/bk", "diffs": diffs}

    def verify_cap(prior, **_):
        received_verify["prior"] = prior
        return {"ran": True, "tests_passed": True, "returncode": 0,
                "tests_selected": ["tests/test_a.py"], "stdout": "ok"}

    events = _run_with(
        plan, {"plan": plan_cap, "apply": apply_cap, "verify": verify_cap},
        stop_on_fail=False,
    )
    # APPLY received the PLAN output; VERIFY received both prior outputs.
    assert received_apply["prior"][0]["diffs"] == diffs
    assert any(p.get("applied_files") == ["a.py"] for p in received_verify["prior"])
    # The task_done events expose the new display fields.
    done = [e for e in events if e.type == "task_done"]
    apply_done = next(e for e in done if e.payload.get("kind") == "apply")
    verify_done = next(e for e in done if e.payload.get("kind") == "verify")
    assert apply_done.payload["output"]["applied_files"] == ["a.py"]
    assert apply_done.payload["output"]["backup_dir"] == "/tmp/bk"
    assert verify_done.payload["output"]["tests_passed"] is True
    assert verify_done.payload["output"]["tests_selected"] == ["tests/test_a.py"]


# ---------------------------------------------------------------------------
# Judge: structural checks on apply / verify outputs.
# ---------------------------------------------------------------------------
def test_judge_fails_apply_task_with_no_writes():
    task = Task(description="apply", kind=TaskKind.APPLY,
                criteria=["files written"],
                output={"applied_files": [], "failed_files": [
                    {"file": "a.py", "error": "patch failed"}]})
    v = Judge(provider=None).judge(task)
    assert v.verdict == "fail"


def test_judge_fails_verify_task_when_tests_failed():
    task = Task(description="verify", kind=TaskKind.VERIFY,
                criteria=["tests pass"],
                output={"ran": True, "tests_passed": False, "returncode": 1,
                        "tests_selected": ["tests/test_x.py"]})
    v = Judge(provider=None).judge(task)
    assert v.verdict == "fail"


def test_judge_passes_verify_task_when_tests_pass():
    task = Task(description="verify", kind=TaskKind.VERIFY,
                criteria=["tests pass"],
                output={"ran": True, "tests_passed": True, "returncode": 0,
                        "tests_selected": ["tests/test_x.py"]})
    v = Judge(provider=None).judge(task)
    assert v.verdict == "pass"
