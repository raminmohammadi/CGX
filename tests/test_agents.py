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


def test_planner_fallback_plan_carries_nonempty_rationale():
    # Every fallback routing branch should leave a human-readable rationale
    # on the Plan so the UI's "Plan Rationale" card has something to show
    # even when no LLM is wired in.
    for goal in [
        "Add a CSV export function to reports",
        "What does parse_codebase do?",
        "Do the tests pass?",
        "create a calculator using React + FastAPI",
    ]:
        plan = Planner(provider=None).plan(goal)
        assert plan.rationale.strip(), f"missing rationale for goal={goal!r}"
        assert plan.to_dict()["rationale"] == plan.rationale


def test_planner_llm_rationale_is_captured_on_plan():
    # When the provider returns a rationale alongside tasks, Planner.plan
    # must thread it through to the Plan dataclass unchanged.
    class _FakeProvider:
        def chat(self, *, messages, **_):
            payload = {
                "rationale": "Decomposed into UI + backend so each scaffold "
                             "covers exactly one layer.",
                "tasks": [
                    {"name": "Scaffold UI",
                     "description": "Generate React UI components",
                     "kind": "scaffold", "criteria": []},
                    {"name": "Scaffold API",
                     "description": "Generate FastAPI backend",
                     "kind": "scaffold", "criteria": []},
                ],
            }
            return {"content": json.dumps(payload)}

    plan = Planner(provider=_FakeProvider()).plan(
        "create a React calculator with FastAPI backend")
    assert "UI + backend" in plan.rationale
    assert plan.to_dict()["rationale"] == plan.rationale


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


def test_planner_fallback_calculator_with_tech_routed_as_scaffold():
    # Real-world phrasing that previously slipped past the noun-based
    # scaffold regex and was misrouted to PLAN against an empty index.
    plan = Planner(provider=None).plan(
        "create a calculator using React UI and python in which user can "
        "pass input and get the output. Ensure, that the UI looks fantasy"
    )
    kinds = [t.kind for t in plan.tasks]
    # SCAFFOLD path now emits SCAFFOLD_MANIFEST + APPLY + VERIFY.
    assert kinds[0] == TaskKind.SCAFFOLD_MANIFEST
    assert kinds[-2:] == [TaskKind.APPLY, TaskKind.VERIFY]
    # The original goal must be propagated as task input so the manifest
    # capability has the full technology-stack context.
    assert plan.tasks[0].inputs.get("goal", "").lower().startswith("create a calculator")


def test_planner_existing_codebase_hint_keeps_change_goal_path():
    # 'add a React component to our existing app' must NOT be treated as
    # a scaffold goal even though it pairs a scaffold-friendly verb with
    # a tech name -- the "existing app" hint pins it to the change path.
    plan = Planner(provider=None).plan(
        "Add a React component to our existing app"
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


def test_planner_trusts_llm_scaffold_tasks_for_ambiguous_goal():
    # When the regex doesn't recognise the goal as scaffold-y but the LLM
    # smartly emits scaffold tasks AND the goal has no existing-codebase
    # hint, the planner now defers to the LLM's decomposition instead of
    # silently dropping the scaffold tasks and forcing a PLAN chain.
    provider = _StubProvider([{
        "content": json.dumps({"tasks": [
            {"name": "UI layer", "description": "Generate React UI components",
             "kind": "scaffold"},
            {"name": "Backend", "description": "Generate Python backend",
             "kind": "scaffold"},
        ]}),
    }])
    # "implement a chess engine" -- no project noun, no tech in the regex,
    # but the LLM gave us scaffold tasks.
    plan = Planner(provider=provider).plan(
        "implement a chess engine that beats humans"
    )
    kinds = [t.kind for t in plan.tasks]
    # LLM emitted scaffold tasks; policy now routes to SCAFFOLD_MANIFEST + APPLY + VERIFY.
    assert kinds[0] == TaskKind.SCAFFOLD_MANIFEST
    assert kinds[-2:] == [TaskKind.APPLY, TaskKind.VERIFY]
    # Exactly one manifest task (which will inject per-file tasks at runtime).
    assert sum(1 for k in kinds if k == TaskKind.SCAFFOLD_MANIFEST) == 1


def test_planner_drops_llm_scaffold_tasks_for_existing_codebase_goal():
    # Counter-test: even with LLM scaffold tasks, an "existing app" goal
    # must stay on the change-goal path so we modify rather than recreate.
    provider = _StubProvider([{
        "content": json.dumps({"tasks": [
            {"description": "Generate new React UI", "kind": "scaffold"},
        ]}),
    }])
    plan = Planner(provider=provider).plan(
        "Add a chart panel to our existing dashboard app"
    )
    kinds = [t.kind for t in plan.tasks]
    assert TaskKind.SCAFFOLD not in kinds
    assert kinds[-3:] == [TaskKind.PLAN, TaskKind.APPLY, TaskKind.VERIFY]


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


def test_tracker_scaffold_file_failure_does_not_halt_siblings_or_apply():
    # A failed SCAFFOLD_FILE must not halt sibling SCAFFOLD_FILE tasks or
    # the trailing APPLY: the user's data on disk depends on the siblings
    # that succeeded, and APPLY is what writes them.
    plan = Plan(goal="g", tasks=[
        Task(description="bad", kind=TaskKind.SCAFFOLD_FILE,
             inputs={"path": "bad.py"}),
        Task(description="good", kind=TaskKind.SCAFFOLD_FILE,
             inputs={"path": "good.py"}),
        Task(description="write", kind=TaskKind.APPLY),
    ])

    def scaffold_file(desc, **kw):
        if desc == "bad":
            raise RuntimeError("empty")
        return {"diffs": [{"file": "good.py", "patch": "+ok"}]}

    def apply(_desc, **_kw):
        return {"applied_files": ["good.py"]}

    _run_with(plan, {"scaffold_file": scaffold_file, "apply": apply},
              stop_on_fail=True)
    assert plan.tasks[0].status == TaskStatus.FAILED
    assert plan.tasks[1].status == TaskStatus.DONE
    assert plan.tasks[2].status == TaskStatus.DONE


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


def test_judge_renders_scaffold_artifact_with_file_paths_and_previews():
    # The LLM judge previously saw json.dumps(out)[:4000] for scaffold
    # tasks, which truncated away the actual file content and led to
    # spurious "doesn't include input fields" rationales. The dedicated
    # SCAFFOLD renderer now exposes plan_md + file paths + per-file
    # content previews so the judge can ground its verdict.
    task = Task(
        description="Generate React calculator",
        kind=TaskKind.SCAFFOLD,
        criteria=["UI has input fields"],
        output={
            "plan_md": "## Fantasy Calculator\nReact + Python backend.",
            "diffs": [
                {"file": "src/App.jsx",
                 "patch": "import React from 'react'; export default function App(){return <input id='num1'/>}"},
                {"file": "package.json",
                 "patch": '{"name":"calc","dependencies":{"react":"^18.0.0"}}'},
                {"file": "backend/server.py",
                 "patch": "from flask import Flask\napp = Flask(__name__)"},
            ],
        },
    )
    art = Judge._render_artifact(task)
    assert "## Generated files (3)" in art
    assert "src/App.jsx" in art and "package.json" in art and "backend/server.py" in art
    # Real file content (not just paths) must appear so the LLM can judge it.
    assert "import React" in art
    assert "Fantasy Calculator" in art


def test_judge_scaffold_artifact_previews_source_files_before_metadata():
    # Real-world scaffolds emit README.md / requirements.txt / package.json
    # before the actual source files. The renderer must preview source code
    # first so the logic-bearing files fit inside the prompt budget.
    task = Task(
        description="Generate React UI components",
        kind=TaskKind.SCAFFOLD,
        criteria=["Supports +,-,*,/"],
        output={
            "plan_md": "calculator scaffold",
            "diffs": [
                {"file": "README.md", "patch": "# README\n" + ("readme " * 300)},
                {"file": "package.json",
                 "patch": '{"name":"calc"}\n' + ("# pad " * 200)},
                {"file": "src/Calculator.js",
                 "patch": "ADD_SUB_MUL_DIV_TOKEN\nfunction calc(a,b,op){"
                          "return op==='+'?a+b:op==='-'?a-b:op==='*'?a*b:a/b;}"},
            ],
        },
    )
    art = Judge._render_artifact(task)
    # The logic-bearing source file's content must appear, even though it
    # was emitted last in the diff list.
    assert "ADD_SUB_MUL_DIV_TOKEN" in art
    # The source-file preview section header must come before the
    # metadata-file preview header.
    assert art.index("### src/Calculator.js") < art.index("### README.md")


def test_judge_short_circuits_scaffold_on_structural_pass_without_calling_llm():
    # A passing structural check for SCAFFOLD (diffs exist + technology
    # match) must be trusted verbatim: the LLM judge is never invoked.
    # Local 3-7B judge models otherwise produce false-negative criteria
    # rejections against scaffolds that demonstrably satisfy them.
    called: Dict[str, Any] = {"n": 0}

    class _StubProvider:
        def chat(self, **kw):
            called["n"] += 1
            return {"content": '{"verdict":"fail","confidence":0.9,'
                               '"rationale":"hallucinated"}'}

    task = Task(
        description="Generate React UI components",
        kind=TaskKind.SCAFFOLD,
        criteria=["Has a calculator interface", "Supports +,-,*,/"],
        inputs={"goal": "create a calculator project with react UI and python backend"},
        output={"plan_md": "calc",
                "diffs": [{"file": "src/App.jsx", "patch": "import React from 'react';"},
                          {"file": "backend/main.py", "patch": "from fastapi import FastAPI"}]},
    )
    v = Judge(provider=_StubProvider()).judge(task)
    assert v.verdict == "pass"
    assert called["n"] == 0, "LLM judge must not be invoked when SCAFFOLD structural check passes"


def test_judge_still_fails_scaffold_with_tech_mismatch_via_structural():
    # The structural tech-match check must still hard-fail when a React
    # goal was answered with Python-only files -- this short-circuits
    # before the LLM and triggers the planner's re-plan loop.
    task = Task(
        description="Generate React UI components",
        kind=TaskKind.SCAFFOLD,
        criteria=["Has a calculator interface"],
        inputs={"goal": "create a calculator with react ui"},
        output={"diffs": [{"file": "app.py", "patch": "from flask import Flask"},
                          {"file": "main.py", "patch": "print('hi')"}]},
    )
    v = Judge(provider=None).judge(task)
    assert v.verdict == "fail"
    assert "React" in v.rationale or "Python" in v.rationale


def test_scaffold_existing_files_collects_paths_in_order_no_duplicates():
    from cgx.agents.loop import _scaffold_existing_files
    prior = [
        {"diffs": [{"file": "src/App.jsx", "patch": "+x"},
                   {"file": "package.json", "patch": "+x"}]},
        {"diffs": [{"file": "src/App.jsx", "patch": "+y"},  # dup
                   {"file": "backend/main.py", "patch": "+y"}]},
        {"answer_md": "no diffs here"},
    ]
    assert _scaffold_existing_files(prior) == [
        "src/App.jsx", "package.json", "backend/main.py",
    ]


def test_judge_llm_prompt_includes_user_goal_when_available():
    # The planner injects the original goal into task.inputs["goal"] so the
    # judge can ground per-task criteria in the full request. Make sure the
    # prompt sent to the LLM provider surfaces that goal.
    captured: Dict[str, Any] = {}

    class _StubProvider:
        def chat(self, **kw):
            captured["messages"] = kw.get("messages")
            return {"content": '{"verdict":"pass","confidence":0.9,'
                               '"rationale":"ok"}'}

    # plan_md only (no diffs) bypasses the SCAFFOLD structural
    # short-circuit and falls through to the LLM judge.
    task = Task(
        description="Generate React UI components",
        kind=TaskKind.SCAFFOLD,
        criteria=["Has a calculator interface"],
        inputs={"goal": "create a calculator project with react UI and python backend"},
        output={"plan_md": "calc design notes", "diffs": []},
    )
    v = Judge(provider=_StubProvider()).judge(task)
    assert v.verdict == "pass"
    user_msg = captured["messages"][1]["content"]
    assert "USER GOAL:" in user_msg
    assert "create a calculator project" in user_msg


# ---------------------------------------------------------------------------
# Fix #1: Plan.owned_files manifest
# ---------------------------------------------------------------------------
def test_plan_owned_files_initialises_empty():
    from cgx.agents.types import Plan
    p = Plan(goal="g")
    assert p.owned_files == {}
    assert "owned_files" in p.to_dict()


def test_tracker_populates_owned_files_after_apply():
    plan = Plan(goal="g", tasks=[
        Task(description="gen", kind=TaskKind.SCAFFOLD),
        Task(description="write", kind=TaskKind.APPLY),
    ])

    def scaffold(text, **_):
        return {"diffs": [{"file": "src/app.py", "patch": "+x=1"}]}

    def apply(prior, **_):
        return {"applied_files": ["src/app.py", "src/utils.py"],
                "failed_files": [{"file": "tests/bad.py", "error": "syntax"}],
                "diffs": [], "backup_dir": "/tmp/bk"}

    list(Tracker({"scaffold": scaffold, "apply": apply},
                 stop_on_fail=False).stream(plan))

    assert plan.owned_files.get("src/app.py") == "applied"
    assert plan.owned_files.get("src/utils.py") == "applied"
    assert plan.owned_files.get("tests/bad.py") == "failed"


def test_plan_to_dict_includes_owned_files():
    from cgx.agents.types import Plan
    p = Plan(goal="g")
    p.owned_files["src/a.py"] = "applied"
    d = p.to_dict()
    assert d["owned_files"] == {"src/a.py": "applied"}


# ---------------------------------------------------------------------------
# Fix #3: apply_failures included in recursive retry condition
# ---------------------------------------------------------------------------
def test_run_agent_retries_on_apply_failure():
    """When the re-plan's APPLY fails the loop must recurse for another attempt."""
    from cgx.agents.loop import run_agent
    attempt_counts: Dict[str, int] = {"apply": 0}

    def scaffold(text, **_):
        return {"diffs": [{"file": "a.py", "patch": "+x=1"}]}

    def apply(prior, **_):
        attempt_counts["apply"] += 1
        if attempt_counts["apply"] == 1:
            # First apply fails (smoke test failure).
            return {"applied_files": [], "failed_files": [
                {"file": "a.py", "error": "python syntax: SyntaxError"}],
                    "diffs": [{"file": "a.py", "patch": "+x=1"}], "backup_dir": None}
        # Second apply succeeds.
        return {"applied_files": ["a.py"], "failed_files": [],
                "diffs": [], "backup_dir": "/tmp/bk"}

    def verify(prior, **_):
        return {"ran": True, "tests_passed": True, "returncode": 0,
                "tests_selected": [], "stdout": ""}

    fixed_plan = run_agent(
        "create a project",
        capabilities={"scaffold": scaffold, "apply": apply, "verify": verify},
        planner=Planner(provider=None),
        project_root="/tmp/proj",
        max_retries=2,
        stop_on_fail=False,
    )
    assert attempt_counts["apply"] >= 2, (
        "apply must be called at least twice -- once for the original plan "
        "and once for the retry triggered by the apply failure"
    )


# ---------------------------------------------------------------------------
# Fix #4: _diagnose_failure
# ---------------------------------------------------------------------------
def test_diagnose_failure_detects_import_error():
    from cgx.agents.loop import _diagnose_failure
    failures = [{"stdout_tail": (
        "ERRORS\n"
        "ImportError while importing test module 'tests/test_calc.py'\n"
        "ModuleNotFoundError: No module named 'src.App'\n"
        "tests/test_calc.py:2: in <module>\n"
        "    from src.App import calculateResult\n"
    )}]
    d = _diagnose_failure(failures)
    assert d["error_type"] == "import_error"
    assert "src.App" in d["bad_imports"]
    assert "tests/test_calc.py" in d["responsible_files"]


def test_diagnose_failure_detects_syntax_error():
    from cgx.agents.loop import _diagnose_failure
    d = _diagnose_failure([{"stdout_tail": "SyntaxError: invalid syntax\nsrc/app.py:5"}])
    assert d["error_type"] == "syntax_error"


def test_diagnose_failure_detects_language_mismatch():
    from cgx.agents.loop import _diagnose_failure
    failures = [{"stdout_tail": (
        "python syntax: imports 'calculateResult' from 'src.App' but "
        "'src/App.jsx' is a JavaScript/JSX file, not a Python module\n"
        "tests/test_calc.py:2"
    )}]
    d = _diagnose_failure(failures)
    assert d["language_mismatch"] is True


def test_diagnose_failure_unknown_on_empty_output():
    from cgx.agents.loop import _diagnose_failure
    d = _diagnose_failure([{"stdout_tail": ""}])
    assert d["error_type"] == "unknown"
    assert d["responsible_files"] == []


# ---------------------------------------------------------------------------
# Fix #2: _build_fix_goal is targeted
# ---------------------------------------------------------------------------
def test_build_fix_goal_names_broken_files_and_preserves_safe():
    from cgx.agents.loop import _build_fix_goal
    from cgx.agents.types import Plan, Task, TaskKind

    plan = Plan(goal="create a calculator", tasks=[
        Task(description="apply", kind=TaskKind.APPLY,
             output={"applied_files": ["src/App.jsx", "src/index.js", "tests/test_calc.py"],
                     "failed_files": []}),
    ])
    plan.owned_files = {
        "src/App.jsx": "applied",
        "src/index.js": "applied",
        "tests/test_calc.py": "applied",
    }
    failures = [{"stdout_tail": (
        "ModuleNotFoundError: No module named 'src.App'\n"
        "tests/test_calc.py:2: in <module>\n"
        "    from src.App import calculateResult\n"
    ), "tests_selected": ["tests/test_calc.py"]}]

    goal = _build_fix_goal("create a calculator", failures, plan, "/tmp/proj")

    # Must name the broken file.
    assert "tests/test_calc.py" in goal
    # Must tell the LLM not to touch the safe files.
    assert "src/App.jsx" in goal
    assert "DO NOT CHANGE" in goal
    # Must surface the language-mismatch guidance.
    assert "JavaScript" in goal or "JSX" in goal or "Python module" in goal


def test_build_fix_goal_language_mismatch_includes_remediation():
    from cgx.agents.loop import _build_fix_goal
    from cgx.agents.types import Plan, Task, TaskKind

    plan = Plan(goal="g", tasks=[
        Task(description="apply", kind=TaskKind.APPLY,
             output={"applied_files": ["tests/test_calc.py"], "failed_files": []}),
    ])
    plan.owned_files = {"tests/test_calc.py": "applied"}
    failures = [{"stdout_tail": (
        "python syntax: imports 'calculateResult' from 'src.App' but "
        "'src/App.jsx' is a JavaScript/JSX file, not a Python module\n"
        "ModuleNotFoundError: No module named 'src.App'"
    )}]
    goal = _build_fix_goal("g", failures, plan, "/tmp/proj")
    assert "NEVER import" in goal or "never import" in goal.lower() or "Python backend" in goal


# ---------------------------------------------------------------------------
# Fix #5: cross-file coherence check
# ---------------------------------------------------------------------------
def test_coherence_check_catches_python_importing_jsx():
    from cgx.codegen.diff_apply import PatchResult
    from cgx.codegen.validate import check_cross_file_coherence

    results = [
        PatchResult(path="src/App.jsx", ok=True,
                    new_content="import React from 'react'; export default function App(){}"),
        PatchResult(path="tests/test_calc.py", ok=True,
                    new_content="from src.App import calculateResult\n\ndef test_basic():\n    assert calculateResult('2+2') == 4\n"),
    ]
    issues = check_cross_file_coherence(results)
    assert len(issues) == 1
    issue = issues[0]
    assert issue.path == "tests/test_calc.py"
    assert not issue.ok
    assert "JavaScript" in issue.error or "JSX" in issue.error


def test_coherence_check_passes_clean_python_batch():
    from cgx.codegen.diff_apply import PatchResult
    from cgx.codegen.validate import check_cross_file_coherence

    results = [
        PatchResult(path="backend/calc.py", ok=True,
                    new_content="def calculate(expr): return eval(expr)\n"),
        PatchResult(path="tests/test_calc.py", ok=True,
                    new_content="from backend.calc import calculate\ndef test_add(): assert calculate('1+1') == 2\n"),
    ]
    assert check_cross_file_coherence(results) == []


def test_coherence_check_ignores_failed_patches():
    from cgx.codegen.diff_apply import PatchResult
    from cgx.codegen.validate import check_cross_file_coherence

    results = [
        PatchResult(path="src/App.jsx", ok=False, new_content=None, error="patch failed"),
        PatchResult(path="tests/test_calc.py", ok=True,
                    new_content="from src.App import x\n"),
    ]
    # App.jsx patch failed so it's not "in the batch" for coherence purposes.
    issues = check_cross_file_coherence(results)
    assert issues == []


def test_coherence_check_detects_tsx_import():
    from cgx.codegen.diff_apply import PatchResult
    from cgx.codegen.validate import check_cross_file_coherence

    results = [
        PatchResult(path="src/Component.tsx", ok=True, new_content="export const X = () => null;"),
        PatchResult(path="tests/test_x.py", ok=True,
                    new_content="from src.Component import X\n"),
    ]
    issues = check_cross_file_coherence(results)
    assert len(issues) == 1
    assert "tsx" in issues[0].error.lower() or "javascript" in issues[0].error.lower()


# ---------------------------------------------------------------------------
# Fix #6: partial apply (write passing files even when some fail smoke check)
# ---------------------------------------------------------------------------
def test_partial_apply_writes_good_files_and_reports_bad(tmp_path):
    from cgx.codegen.disk_apply import apply_diffs_to_disk

    # One valid Python file and one with a syntax error.
    diffs = [
        {
            "file": "good.py",
            "patch": "--- /dev/null\n+++ b/good.py\n@@ -0,0 +1,1 @@\n+x = 1\n",
        },
        {
            "file": "bad.py",
            # Deliberately invalid Python -- unterminated string.
            "patch": '--- /dev/null\n+++ b/bad.py\n@@ -0,0 +1,1 @@\n+x = "unterminated\n',
        },
    ]
    result = apply_diffs_to_disk(str(tmp_path), diffs)

    # good.py must be written despite bad.py failing.
    assert "good.py" in result["applied_files"]
    assert (tmp_path / "good.py").exists()

    # bad.py must be in failed_files.
    failed = [f["file"] for f in result["failed_files"]]
    assert "bad.py" in failed
    assert not (tmp_path / "bad.py").exists()

    # smoke_ok is False because at least one file failed.
    assert result["smoke_ok"] is False


def test_all_good_files_sets_smoke_ok_true(tmp_path):
    from cgx.codegen.disk_apply import apply_diffs_to_disk

    diffs = [{
        "file": "ok.py",
        "patch": "--- /dev/null\n+++ b/ok.py\n@@ -0,0 +1,1 @@\n+x = 1\n",
    }]
    result = apply_diffs_to_disk(str(tmp_path), diffs)
    assert result["smoke_ok"] is True
    assert result["applied_files"] == ["ok.py"]
    assert result["failed_files"] == []



# ---------------------------------------------------------------------------
# Planner.plan_fix and retry routing
# ---------------------------------------------------------------------------
def test_plan_fix_emits_plan_apply_verify_only():
    plan = Planner(provider=None).plan_fix(
        "Fix test failures in project /tmp/x\nOriginal goal: create app for a calculator using React",
        broken_files=["backend/app.py"],
        already_good_files=["frontend/App.jsx", "package.json"],
        prior_owned_files={"backend/app.py": "applied",
                           "frontend/App.jsx": "applied",
                           "package.json": "applied"},
    )
    kinds = [t.kind for t in plan.tasks]
    assert kinds == [TaskKind.PLAN, TaskKind.APPLY, TaskKind.VERIFY], (
        "plan_fix must never re-route to SCAFFOLD even when the embedded goal "
        f"text contains 'create app for a calculator using React'; got {kinds}"
    )
    plan_task = plan.tasks[0]
    assert plan_task.inputs["target_files"] == ["backend/app.py"]
    assert plan_task.inputs["do_not_change"] == ["frontend/App.jsx", "package.json"]
    assert plan.owned_files == {
        "backend/app.py": "applied",
        "frontend/App.jsx": "applied",
        "package.json": "applied",
    }
    assert plan.rationale  # non-empty so the UI has something to render


def test_plan_fix_rejects_empty_goal():
    with pytest.raises(ValueError):
        Planner(provider=None).plan_fix("")


def test_plan_fix_handles_missing_broken_files():
    plan = Planner(provider=None).plan_fix("Fix something")
    plan_task = plan.tasks[0]
    assert "target_files" not in plan_task.inputs
    assert "do_not_change" not in plan_task.inputs
    assert plan.owned_files == {}


def test_already_good_files_filters_broken_and_only_applied():
    from cgx.agents.loop import _already_good_files

    plan = Plan(goal="g", tasks=[])
    plan.owned_files = {
        "a.py": "applied",
        "b.py": "applied",
        "broken.py": "applied",   # should be filtered: it's in broken_files
        "c.py": "failed",          # should be filtered: status != applied
    }
    assert _already_good_files(plan, ["broken.py"]) == ["a.py", "b.py"]


def test_apply_broken_files_dedupes_and_preserves_order():
    from cgx.agents.loop import _apply_broken_files

    failures = [
        {"failed_files": [
            {"file": "a.py", "error": "x"},
            {"file": "b.py", "error": "y"},
        ]},
        {"failed_files": [
            {"file": "a.py", "error": "z"},   # duplicate
            {"file": "c.py", "error": "q"},
        ]},
    ]
    assert _apply_broken_files(failures) == ["a.py", "b.py", "c.py"]


# ---------------------------------------------------------------------------
# Tracker skip for SCAFFOLD_FILE targeting already-applied paths
# ---------------------------------------------------------------------------
def test_tracker_skips_scaffold_file_when_target_already_applied():
    plan = Plan(goal="g", tasks=[
        Task(description="re-emit frontend/App.jsx",
             kind=TaskKind.SCAFFOLD_FILE,
             inputs={"path": "frontend/App.jsx"}),
        Task(description="generate new backend/app.py",
             kind=TaskKind.SCAFFOLD_FILE,
             inputs={"path": "backend/app.py"}),
    ])
    plan.owned_files = {"frontend/App.jsx": "applied"}
    calls: List[str] = []

    def scaffold_file(desc, **kw):
        calls.append(str(kw.get("path") or desc))
        return {"file": kw.get("path"), "patch": "--- /dev/null\n+++ a\n"}

    events = _run_with(plan, {"scaffold_file": scaffold_file})
    statuses = [t.status for t in plan.tasks]
    assert statuses[0] == TaskStatus.SKIPPED, (
        "SCAFFOLD_FILE targeting an already-applied path must be skipped, "
        f"got {statuses[0]}"
    )
    assert statuses[1] == TaskStatus.DONE
    assert calls == ["backend/app.py"], (
        "scaffold capability must not be invoked for already-applied files"
    )
    skipped_events = [e for e in events if e.type == "task_skipped"]
    assert any("already applied" in str(e.payload.get("reason") or "")
               for e in skipped_events)


def test_tracker_does_not_skip_scaffold_file_when_target_failed_previously():
    plan = Plan(goal="g", tasks=[
        Task(description="regen broken file",
             kind=TaskKind.SCAFFOLD_FILE,
             inputs={"path": "broken.py"}),
    ])
    plan.owned_files = {"broken.py": "failed"}
    calls: List[str] = []

    def scaffold_file(desc, **kw):
        calls.append(str(kw.get("path") or desc))
        return {"file": kw.get("path"), "patch": ""}

    _run_with(plan, {"scaffold_file": scaffold_file})
    assert plan.tasks[0].status == TaskStatus.DONE
    assert calls == ["broken.py"]


# ---------------------------------------------------------------------------
# loop._build_default_capabilities.plan: strips kwargs unknown to engine
# ---------------------------------------------------------------------------
def test_plan_capability_strips_target_files_and_folds_into_task_text(monkeypatch):
    """``plan_fix`` attaches ``target_files`` / ``do_not_change`` to the PLAN
    task's inputs. The Tracker forwards inputs as **kwargs, but
    ``generate_code_plan`` doesn't accept them -- they must be stripped and
    folded into the task text rather than raising TypeError.
    """
    from cgx.agents.loop import _build_default_capabilities

    captured: Dict[str, Any] = {}

    def fake_generate_code_plan(index_dir, records_path, task, provider,
                                *, project_root=None, self_test=False,
                                run_tests=False, max_retries=1,
                                skills=None, **_ignored):
        # Strict signature: target_files / do_not_change would TypeError here.
        captured["task"] = task
        captured["project_root"] = project_root
        captured["self_test"] = self_test
        captured["skills"] = skills
        return {"plan_md": "ok"}

    monkeypatch.setattr(
        "cgx.answer.engine.generate_code_plan", fake_generate_code_plan,
    )
    # Avoid touching the on-disk symbol map.
    monkeypatch.setattr(
        "cgx.codegen.symbol_map.build_symbol_context_prompt",
        lambda _p: "",
    )

    caps = _build_default_capabilities(
        provider=object(),
        index_dir="/tmp/idx",
        records_path="/tmp/records.jsonl",
        project_root="/tmp/proj",
    )
    out = caps["plan"](
        "Fix the failing tests",
        target_files=["backend/app.py"],
        do_not_change=["package.json"],
        skills=["react"],
    )
    assert out == {"plan_md": "ok"}
    assert "backend/app.py" in captured["task"]
    assert "package.json" in captured["task"]
    assert captured["skills"] == ["react"]
    assert captured["project_root"] == "/tmp/proj"
    assert captured["self_test"] is True


# ---------------------------------------------------------------------------
# loop._build_scaffold_retry_plan: SCAFFOLD_FILE failures retry via
# SCAFFOLD_FILE tasks (no engine / no retriever index required)
# ---------------------------------------------------------------------------
def test_build_scaffold_retry_plan_emits_scaffold_apply_verify_only():
    """When a SCAFFOLD_FILE task fails (e.g. empty content), the retry plan
    must regenerate the failed file via another SCAFFOLD_FILE task -- never
    a PLAN task -- because PLAN's engine path reads a FAISS index that
    doesn't exist for freshly-scaffolded user projects.
    """
    from cgx.agents.loop import _build_scaffold_retry_plan

    original = Plan(
        goal="create a calculator with react UI",
        tasks=[
            Task(
                description="Generate tests/App.test.jsx",
                kind=TaskKind.SCAFFOLD_FILE,
                name="Generate tests/App.test.jsx",
                inputs={"path": "tests/App.test.jsx",
                        "file_description": "React test file",
                        "layer": "tests",
                        "goal": "create a calculator with react UI",
                        "skills": ["react"]},
                status=TaskStatus.FAILED,
            ),
        ],
        owned_files={"backend/main.py": "applied", "src/App.jsx": "applied"},
    )

    retry = _build_scaffold_retry_plan(
        original, ["tests/App.test.jsx"], "Fix failing scaffold files",
    )
    kinds = [t.kind for t in retry.tasks]
    assert kinds == [TaskKind.SCAFFOLD_FILE, TaskKind.APPLY, TaskKind.VERIFY]
    assert TaskKind.PLAN not in kinds, (
        "scaffold-retry plan must not emit a PLAN task -- that would require "
        "a FAISS retriever index that doesn't exist for fresh user projects."
    )
    scaffold = retry.tasks[0]
    assert scaffold.inputs["path"] == "tests/App.test.jsx"
    # Original inputs preserved so generate_single_scaffold_file has context.
    assert scaffold.inputs["layer"] == "tests"
    assert scaffold.inputs["goal"] == "create a calculator with react UI"
    assert scaffold.inputs["skills"] == ["react"]
    # Previously-applied files are carried forward so the Tracker can skip
    # any stray scaffold for them.
    assert retry.owned_files == {"backend/main.py": "applied",
                                 "src/App.jsx": "applied"}
    # Linear dependency chain.
    for i in range(1, len(retry.tasks)):
        assert retry.tasks[i].dependencies == [retry.tasks[i - 1].id]


def test_run_agent_scaffold_file_retry_does_not_require_index(monkeypatch):
    """End-to-end: a SCAFFOLD_FILE failure on attempt 1 must trigger a
    SCAFFOLD_FILE-based retry, not a PLAN-based one. Regression for the
    ``FileNotFoundError: meta.json`` crash when no index is built yet.
    """
    from cgx.agents.loop import run_agent

    scaffold_calls: List[Dict[str, Any]] = []
    apply_calls: List[List[Dict[str, Any]]] = []
    verify_calls: List[List[Dict[str, Any]]] = []

    def scaffold_file(desc, **kw):
        scaffold_calls.append({"desc": desc, "path": kw.get("path")})
        # First call (initial) returns empty → judge will fail it.
        # Second call (retry) returns valid content.
        if len(scaffold_calls) == 1:
            return {"diffs": [], "path": kw.get("path"), "content": ""}
        return {
            "diffs": [{"file": kw.get("path"),
                       "new_content": "export default function App() {}\n"}],
            "path": kw.get("path"),
            "content": "export default function App() {}\n",
        }

    def apply(prior, **kw):
        apply_calls.append(prior)
        return {"applied_files": ["tests/App.test.jsx"], "failed_files": []}

    def verify(prior, **kw):
        verify_calls.append(prior)
        return {"ran": True, "tests_passed": True, "returncode": 0,
                "tests_selected": [], "stdout": "", "stderr": ""}

    # If anything reaches the ``plan`` capability the test must fail --
    # that's the path that crashed on the missing FAISS index.
    def plan_should_never_be_called(*a, **kw):
        raise AssertionError(
            "scaffold-retry must not invoke the plan capability"
        )

    capabilities = {
        "scaffold_file": scaffold_file,
        "apply": apply,
        "verify": verify,
        "plan": plan_should_never_be_called,
    }

    # Drive a deterministic single SCAFFOLD_FILE plan so we don't depend
    # on Planner's branching for this test.
    init_plan = Plan(
        goal="generate a tiny react test",
        tasks=[
            Task(description="Generate tests/App.test.jsx",
                 kind=TaskKind.SCAFFOLD_FILE,
                 name="Generate tests/App.test.jsx",
                 inputs={"path": "tests/App.test.jsx",
                         "file_description": "React test file",
                         "layer": "tests",
                         "goal": "generate a tiny react test"},
                 criteria=["File has non-stub content.",
                           "File passes syntax validation."]),
        ],
    )

    class _StubPlanner:
        def plan(self, _goal):
            return init_plan

    final = run_agent(
        "generate a tiny react test",
        provider=None,
        capabilities=capabilities,
        planner=_StubPlanner(),
        project_root="/tmp/does-not-exist-but-not-read",
        stream=False, max_retries=1,
    )
    # Two SCAFFOLD_FILE calls: 1 initial (empty) + 1 retry (valid).
    assert len(scaffold_calls) == 2
    assert all(c["path"] == "tests/App.test.jsx" for c in scaffold_calls)
    # Retry plan reached APPLY + VERIFY.
    assert len(apply_calls) == 1
    assert len(verify_calls) == 1
    assert final is not None


# ---------------------------------------------------------------------------
# Judge: APPLY soft-pass when there were no diffs to write
# ---------------------------------------------------------------------------
def test_judge_passes_apply_when_no_diffs_to_apply():
    """When every upstream SCAFFOLD_FILE soft-failed, the ``apply``
    capability returns ``applied_files=[]`` AND ``failed_files=[]`` AND
    surfaces ``error="no diffs found in prior task outputs"``. The Judge
    must treat that as a benign no-op (previously-applied files remain on
    disk) rather than a hard failure that halts the plan and the run.
    """
    task = Task(description="apply", kind=TaskKind.APPLY,
                criteria=["files written"],
                output={"applied_files": [], "failed_files": [],
                        "diffs": [],
                        "error": "no diffs found in prior task outputs"})
    v = Judge(provider=None).judge(task)
    assert v.verdict == "pass"
    assert "no new diffs" in v.rationale.lower()


def test_judge_still_fails_apply_when_truly_nothing_written():
    """The soft-pass must be narrow: an APPLY that produced 0 writes
    *without* the explicit "no diffs" error is still a hard failure
    (e.g. a real bug in disk_apply that silently dropped the work).
    """
    task = Task(description="apply", kind=TaskKind.APPLY,
                criteria=["files written"],
                output={"applied_files": [], "failed_files": []})
    v = Judge(provider=None).judge(task)
    assert v.verdict == "fail"
    assert "no files" in v.rationale.lower()


# ---------------------------------------------------------------------------
# Index-availability check: prevents plan_fix from crashing on a freshly
# scaffolded user project (no FAISS index built yet).
# ---------------------------------------------------------------------------
def test_plan_fix_index_available_false_for_missing_or_empty(tmp_path):
    """Sanity-check the helper used by the retry loop to decide whether
    ``planner.plan_fix`` can be invoked. It must return False when the
    index dir is None, missing, or has no ``meta.json`` so the loop can
    fall back to a path that doesn't depend on retrieval.
    """
    from cgx.agents.loop import _plan_fix_index_available
    assert _plan_fix_index_available(None) is False
    assert _plan_fix_index_available("") is False
    assert _plan_fix_index_available(str(tmp_path / "does-not-exist")) is False
    # Empty directory: no meta.json.
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _plan_fix_index_available(str(empty)) is False
    # Directory with meta.json: usable.
    populated = tmp_path / "populated"
    populated.mkdir()
    (populated / "meta.json").write_text("{}", encoding="utf-8")
    assert _plan_fix_index_available(str(populated)) is True


def test_run_agent_does_not_mask_verify_failure_when_no_index_available(tmp_path):
    """When a verify task fails and the FAISS index isn't available,
    the retry loop must NOT call ``plan_fix`` (which would crash on
    missing ``meta.json``), AND must NOT hide the failure by demoting
    VERIFY to SKIPPED. The plan is returned with VERIFY still FAILED
    so the caller (and user) sees the real test output.
    """
    from cgx.agents.loop import run_agent

    project_root = tmp_path / "proj"
    project_root.mkdir()
    # Sentinel guards: plan_fix must NEVER be reached.
    plan_fix_calls: List[Any] = []

    class _GuardPlanner(Planner):
        def plan_fix(self, *a, **kw):  # type: ignore[override]
            plan_fix_calls.append((a, kw))
            raise AssertionError("plan_fix must not be called when no index is available")

    def scaffold(text, **_):
        return {"diffs": [{"file": "a.py", "patch": "+x=1"}]}

    def apply(prior, **_):
        return {"applied_files": ["a.py"], "failed_files": [],
                "diffs": [{"file": "a.py", "patch": "+x=1"}],
                "backup_dir": "/tmp/bk"}

    def verify(prior, **_):
        # Simulate a real test-collection failure (rc=2) whose stderr
        # does NOT look like an unrecoverable sandbox import error.
        return {"ran": True, "tests_passed": False, "returncode": 2,
                "tests_selected": ["tests/test_a.py"],
                "stdout": "E   AssertionError: oops",
                "stderr": ""}

    final = run_agent(
        "create a project",
        capabilities={"scaffold": scaffold, "apply": apply, "verify": verify},
        planner=_GuardPlanner(provider=None),
        project_root=str(project_root),
        index_dir=None,           # No FAISS index available.
        records_path=None,
        max_retries=1,
        stop_on_fail=False,
    )
    assert plan_fix_calls == [], "plan_fix must not be called without an index"
    # VERIFY must stay FAILED -- the test ran, failed, and the caller must see
    # the real outcome. Demoting to SKIPPED hides failures from the user.
    verifies = [t for t in final.tasks if t.kind == TaskKind.VERIFY]
    assert verifies, "plan should have at least one verify task"
    assert any(t.status == TaskStatus.FAILED for t in verifies), (
        "verify failure must remain FAILED (not hidden as SKIPPED) when "
        f"plan_fix is unavailable; got statuses={[t.status for t in verifies]}"
    )


def test_run_agent_routes_apply_failure_through_scaffold_retry_when_no_index(tmp_path):
    """When apply fails and there's no FAISS index, the retry must use
    ``_build_scaffold_retry_plan`` (which doesn't require an index)
    instead of ``planner.plan_fix`` (which would crash on missing
    ``meta.json``).
    """
    from cgx.agents.loop import run_agent

    project_root = tmp_path / "proj"
    project_root.mkdir()
    plan_fix_calls: List[Any] = []

    class _GuardPlanner(Planner):
        def plan_fix(self, *a, **kw):  # type: ignore[override]
            plan_fix_calls.append((a, kw))
            raise AssertionError("plan_fix must not be called when no index is available")

    apply_count = {"n": 0}

    def scaffold(text, **_):
        return {"diffs": [{"file": "a.py", "patch": "+x=1"}]}

    def scaffold_file(text, **_):
        return {"diffs": [{"file": "a.py", "patch": "+def f():\n    return 1\n"}]}

    def apply(prior, **_):
        apply_count["n"] += 1
        if apply_count["n"] == 1:
            return {"applied_files": [], "failed_files": [
                {"file": "a.py", "error": "python syntax: SyntaxError"}],
                    "diffs": [{"file": "a.py", "patch": "+x=1"}],
                    "backup_dir": None}
        return {"applied_files": ["a.py"], "failed_files": [],
                "diffs": [], "backup_dir": "/tmp/bk"}

    def verify(prior, **_):
        return {"ran": True, "tests_passed": True, "returncode": 0,
                "tests_selected": [], "stdout": ""}

    run_agent(
        "create a project",
        capabilities={"scaffold": scaffold, "scaffold_file": scaffold_file,
                      "apply": apply, "verify": verify},
        planner=_GuardPlanner(provider=None),
        project_root=str(project_root),
        index_dir=None,
        records_path=None,
        max_retries=1,
        stop_on_fail=False,
    )
    assert plan_fix_calls == [], "plan_fix must not be called without an index"
    assert apply_count["n"] >= 2, (
        "apply must run at least twice (original + scaffold-retry plan's apply)"
    )
