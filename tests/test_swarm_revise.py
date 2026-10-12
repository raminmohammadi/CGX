"""The feedback loop: user feedback on a finished swarm build becomes a precise,
verified revision of that build (``cgx.session.revise`` + ``SWARM_REVISE``).

Covers the pure core (change-vs-question classification, build discovery,
deterministic localization, theme switch), the executor (model refinement,
grounded apply with the unchanged-retry, the revision report and plan
history), the router edges (reopen a finished build, chain to SWARM_VERIFY,
restore the prior verdict when nothing changed) and the CLI continuation.
"""

from __future__ import annotations

import json
import os

import pytest

from cgx.session.actions import CreateTask, UpdateSessionStatus, UpdateTaskStatus
from cgx.session.models import (
    Artifact, ArtifactKind, Session, SessionMode, SessionStatus, TaskKind, TaskNode,
    TaskNodeStatus)
from cgx.session.revise import (
    is_change_request, latest_build, rank_files, relevant_lines, split_identifier,
    theme_from_feedback, unified_diff)
from cgx.session.router import Router, _swarm_revise_to_successors
from cgx.session.tasks import swarm_revise as sr
from cgx.session.tasks.base import ExecutorDeps


# ----------------------------- fixtures: a small React build -----------------------------

_FILES = {
    "src/main.jsx": "import React from 'react';\nimport { BrowserRouter } from 'react-router';\n"
                    "import App from './App.jsx';\nimport './styles/tokens.css';\nimport './styles/app.css';\n",
    "src/App.jsx": "import { Routes, Route } from 'react-router';\nimport Chat from './pages/Chat.jsx';\n"
                   "import Settings from './pages/Settings.jsx';\nimport TopBar from './components/TopBar.jsx';\n"
                   "export default function App() {\n  return (<div className=\"app\"><TopBar />"
                   "<Routes><Route path=\"/\" element={<Chat />} /><Route path=\"/settings\" element={<Settings />} />"
                   "</Routes></div>);\n}\n",
    "src/components/TopBar.jsx": "import { NavLink } from 'react-router';\nexport default function TopBar() {\n"
                                 "  return (<header className=\"topbar\"><span className=\"brand\">Chatbot</span>"
                                 "<nav><NavLink to=\"/\">Chat</NavLink><NavLink to=\"/settings\">Settings</NavLink></nav>"
                                 "<span className=\"badge\">Not Connected</span></header>);\n}\n",
    "src/components/Composer.jsx": "export default function Composer({ onSend }) {\n"
                                   "  return (<form className=\"composer\"><textarea placeholder=\"Type a message\" />"
                                   "<button type=\"submit\">Send</button></form>);\n}\n",
    "src/components/SettingsForm.jsx": "import { saveSettings } from '../lib/storage.js';\n"
                                       "export default function SettingsForm() {\n  return (<form className=\"settings-form\">"
                                       "<label>API Address<input /></label><label>API Key<input type=\"password\" /></label>"
                                       "<button>Save</button></form>);\n}\n",
    "src/pages/Chat.jsx": "import Composer from '../components/Composer.jsx';\nexport default function Chat() {\n"
                          "  return (<main className=\"chat\"><Composer onSend={() => {}} /></main>);\n}\n",
    "src/pages/Settings.jsx": "import SettingsForm from '../components/SettingsForm.jsx';\n"
                              "export default function Settings() {\n  return (<main><h1>Settings</h1><SettingsForm /></main>);\n}\n",
    "src/lib/api.js": "export async function sendMessage(baseUrl, apiKey, text) {\n  return fetch(baseUrl);\n}\n",
    "src/lib/storage.js": "export function saveSettings(s) { localStorage.setItem('s', JSON.stringify(s)); }\n"
                          "export function loadSettings() { return JSON.parse(localStorage.getItem('s') || '{}'); }\n",
    "src/styles/app.css": ".app { display: grid; }\n.topbar { max-width: 960px; margin: 0 auto; }\n"
                          ".composer { display: flex; }\n",
    "src/styles/tokens.css": ":root { --color-bg: #fff; }\n",
    "README.md": "# Chatbot\n\nRun `npm run dev`.\n",
}
_LAYERS = [{"name": "app", "files": [
    {"path": p, "description": {
        "src/components/TopBar.jsx": "Top bar with brand, nav links and a connection badge",
        "src/components/Composer.jsx": "Chat composer: textarea plus Send button",
        "src/lib/storage.js": "Persist settings in localStorage",
        "src/lib/api.js": "API client with bearer auth",
    }.get(p, f"{p} for the chatbot app"), "depends_on": []}
    for p in _FILES]}]


def _design():
    from cgx.design import build_brief
    brief = build_brief("create a chatbot app with an API key settings tab and a chat page",
                        ["react"])
    d = brief.to_dict()
    d["tokens_path"] = "src/styles/tokens.css"
    d["global_css_path"] = "src/styles/app.css"
    d["app_path"] = "src/App.jsx"
    d["entry_path"] = "src/main.jsx"
    d["api_client_path"] = "src/lib/api.js"
    d["storage_path"] = "src/lib/storage.js"
    return d


def _write_tree(root) -> None:
    for p, src in _FILES.items():
        full = root / p
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(src, encoding="utf-8")


def _plan(root, design=None):
    return {"goal": "create a chatbot app", "project_root": str(root),
            "paths": list(_FILES), "layers": _LAYERS, "contracts": {},
            "skills": ["react"], "design": design or _design()}


def _specs():
    from cgx.session.tasks.swarm_plan import plan_specs
    return plan_specs({"layers": _LAYERS})


# ----------------------------- pure: classification -----------------------------

def test_is_change_request_vs_question():
    assert is_change_request("make the top bar span the full width")
    assert is_change_request("can you make the header bigger?")
    assert is_change_request("why is the settings page blank?")
    assert is_change_request("the Send button should be aligned with the textarea")
    assert not is_change_request("how does the login work?")
    assert not is_change_request("what is the entrypoint")
    assert not is_change_request("")


def test_split_identifier():
    assert split_identifier("SettingsForm.jsx") == ["settings", "form"]
    assert split_identifier("app.css") == ["app"]
    assert split_identifier("message_bubble") == ["message", "bubble"]


# ----------------------------- pure: build discovery -----------------------------

def _task(kind, status, inputs=None, outputs=None, created_at=0.0):
    t = TaskNode.new(session_id="s", kind=kind, name=kind.value, inputs=inputs or {})
    t.status = status
    t.outputs = outputs
    t.created_at = created_at
    return t


def _built_tasks(verify_ok=True):
    return [
        _task(TaskKind.SWARM_TECH_LEAD, TaskNodeStatus.DONE,
              inputs={"goal": "g", "project_root": "/p"},
              outputs={"work_plan_artifact_id": "wp1", "file_count": 2, "goal": "g",
                       "project_root": "/p"}, created_at=1),
        _task(TaskKind.SWARM_DEVELOPER, TaskNodeStatus.DONE,
              inputs={"work_plan_artifact_id": "wp1"},
              outputs={"work_plan_artifact_id": "wp1", "file_index": 0}, created_at=2),
        _task(TaskKind.SWARM_VERIFY, TaskNodeStatus.DONE,
              inputs={"work_plan_artifact_id": "wp1"},
              outputs={"verify_ok": verify_ok, "verify_report_artifact_id": "vr1",
                       "project_root": "/p"}, created_at=3),
    ]


def test_latest_build_reads_plan_report_and_verdict():
    b = latest_build(_built_tasks())
    assert b is not None
    assert b.work_plan_artifact_id == "wp1"
    assert b.verify_report_artifact_id == "vr1"
    assert b.last_verify_ok is True
    assert b.goal == "g" and b.project_root == "/p"
    assert b.revisions == 0 and b.in_flight is False


def test_latest_build_requires_written_files_and_tracks_in_flight():
    only_plan = [_task(TaskKind.SWARM_TECH_LEAD, TaskNodeStatus.DONE,
                       outputs={"work_plan_artifact_id": "wp1"})]
    assert latest_build(only_plan) is None
    tasks = _built_tasks() + [_task(TaskKind.SWARM_DEVELOPER, TaskNodeStatus.READY,
                                    inputs={"work_plan_artifact_id": "wp1"}, created_at=4)]
    assert latest_build(tasks).in_flight is True
    tasks = _built_tasks() + [_task(TaskKind.SWARM_REVISE, TaskNodeStatus.DONE,
                                    inputs={"work_plan_artifact_id": "wp1"},
                                    outputs={"changed": ["a"]}, created_at=4)]
    assert latest_build(tasks).revisions == 1
    assert latest_build([]) is None


# ----------------------------- pure: localization -----------------------------

def _rank(feedback, **kw):
    return rank_files(feedback, list(_FILES), _specs(), _design(), dict(_FILES), **kw)


def test_rank_names_the_component_the_feedback_mentions():
    ranked = _rank("make the top bar span the full width of the page")
    assert ranked[0].path == "src/components/TopBar.jsx"
    assert any("top bar" in r for r in ranked[0].reasons)
    # a layout complaint also implicates the app stylesheet
    assert "src/styles/app.css" in [c.path for c in ranked]
    # the deterministic tokens stylesheet is never a target
    assert "src/styles/tokens.css" not in [c.path for c in ranked]


def test_rank_uses_quoted_ui_copy_and_elements():
    ranked = _rank("the 'Send' button should be level with the textarea")
    assert ranked[0].path == "src/components/Composer.jsx"
    assert any("Send" in r for r in ranked[0].reasons)


def test_rank_maps_a_failing_route_report_to_its_files():
    ranked = _rank("the settings page is blank", failing_routes=["/settings"])
    paths = [c.path for c in ranked]
    assert paths[0] == "src/pages/Settings.jsx"
    assert "src/components/SettingsForm.jsx" in paths  # second ring: what the page renders


def test_rank_maps_concerns_to_file_roles():
    assert _rank("the README should explain how to configure the API key")[0].path == "README.md"
    paths = [c.path for c in _rank("remember the API key after a reload")]
    assert "src/lib/storage.js" in paths
    assert _rank("make the fonts larger and the colours more vivid")[0].path == "src/styles/app.css"


def test_rank_returns_nothing_for_unrelated_text():
    assert _rank("xyzzy plugh") == []


def test_theme_from_feedback():
    d = _design()
    d["theme"] = "conversational"
    assert theme_from_feedback("use a dark theme please", d) in ("midnight-console", "bold-dark")
    assert theme_from_feedback("switch to the Midnight Console theme", d) == "midnight-console"
    assert theme_from_feedback("make the header bigger", d) is None
    d["theme"] = "midnight-console"
    assert theme_from_feedback("use the midnight console theme", d) is None
    assert theme_from_feedback("dark mode", None) is None


def test_relevant_lines_and_diff():
    lines = relevant_lines(_FILES["src/styles/app.css"], "the topbar should be full width")
    assert "2:" in lines and ".topbar" in lines
    diff = unified_diff("a.css", ".a{x:1}\n", ".a{x:2}\n")
    assert "-.a{x:1}" in diff and "+.a{x:2}" in diff


# ----------------------------- executor -----------------------------

class _Art:
    def __init__(self, kind, content, aid):
        self.kind = kind
        self.content = content
        self.artifact_id = aid


class _Store:
    def __init__(self, plan, report=None):
        self.arts = {"wp": _Art(ArtifactKind.WORK_PLAN, plan, "wp")}
        if report is not None:
            self.arts["vr"] = _Art(ArtifactKind.SWARM_VERIFY_REPORT, report, "vr")
        self.saved = []
        self.progress = []

    def get_artifact(self, aid):
        return self.arts.get(aid)

    def save_artifact(self, art):
        self.saved.append(art)

    def emit_task_progress(self, sid, tid, progress):
        self.progress.append(progress)


class _Provider:
    """Answers the localization prompt with ``reply`` (JSON text) or raises."""

    def __init__(self, reply=None, raise_=False):
        self.reply = reply
        self.raise_ = raise_
        self.calls = []

    def chat(self, **kw):
        self.calls.append(kw)
        if self.raise_:
            raise RuntimeError("offline")
        return {"content": self.reply or ""}


class _Outcome:
    def __init__(self, path, content, ok=True, method="full-file", error=""):
        self.path, self.content, self.ok, self.method, self.error = path, content, ok, method, error


def _run(tmp_path, feedback, provider, store=None, **inputs):
    deps = ExecutorDeps(project_root=str(tmp_path), provider=provider, store=store)
    task = TaskNode.new(session_id="s", kind=TaskKind.SWARM_REVISE, name="revise",
                        inputs={"feedback": feedback, "work_plan_artifact_id": "wp",
                                "verify_report_artifact_id": "vr", "prior_status": "completed",
                                "revision_index": 1, **inputs})
    return sr.swarm_revise(task, deps), task


def test_executor_applies_model_localized_change_and_reports(tmp_path, monkeypatch):
    _write_tree(tmp_path)
    store = _Store(_plan(tmp_path), report={"runtime": {"routes": []}})
    provider = _Provider(json.dumps({
        "changes": [{"path": "src/components/TopBar.jsx",
                     "instruction": "add a `topbar--full` modifier class to the header",
                     "why": "it renders the header"}],
        "note": "Widen the top bar to the full page width."}))
    seen = {}

    def _gen(**kw):
        seen.update(kw)
        return _Outcome(kw["path"], _FILES[kw["path"]].replace('className="topbar"',
                                                             'className="topbar topbar--full"'))
    monkeypatch.setattr(sr, "generate_file", _gen)

    res, task = _run(tmp_path, "make the top bar span the full width", provider, store)
    assert res.failure is None
    assert res.outputs["changed"] == ["src/components/TopBar.jsx"]
    assert res.outputs["note"] == "Widen the top bar to the full page width."
    # the revision prompt carried the feedback + the per-file instruction, grounded on the file
    assert "USER FEEDBACK ON THE RUNNING APP: make the top bar" in seen["description"]
    assert "REQUIRED CHANGE IN THIS FILE: add a `topbar--full` modifier" in seen["description"]
    assert seen["modify_existing"] is True and seen["existing_limit"] == sr._REVISE_EXISTING_LIMIT
    assert seen["design"]["applies"] is True
    # written to disk
    assert "topbar--full" in (tmp_path / "src/components/TopBar.jsx").read_text()
    # the report says what changed where, and why
    art = res.artifact
    assert art.kind is ArtifactKind.SWARM_REVISION
    c = art.content
    assert c["status"] == "applied" and c["method"] == "model"
    # The deterministic layout diagnosis found the real cap (.topbar max-width)
    # and queued its fix first; the stub returned that file unchanged, which is
    # reported honestly rather than counted as done.
    diag_row = c["changes"][0]
    assert diag_row["source"] == "diagnosis" and diag_row["path"] == "src/styles/app.css"
    assert ".topbar { max-width: none; }" in diag_row["instruction"] and diag_row["outcome"] == "unchanged"
    row = next(ch for ch in c["changes"] if ch["path"] == "src/components/TopBar.jsx")
    assert row["outcome"] == "changed"
    assert "+" in row["diff"] and "topbar--full" in row["diff"]
    # the diff touched what the instruction named -> applied in one attempt
    assert row["covered"] == ["topbar--full"] and row["partial"] is False and c["parts"] == [
        "make the top bar span the full width"]
    assert row["why"] == "it renders the header"
    assert c["candidates"][0]["path"] == "src/components/TopBar.jsx"
    assert c["unchanged"] == ["src/styles/app.css"]
    # history appended to the plan artifact for the next round
    assert store.saved and store.saved[-1].content["revisions"][0]["feedback"].startswith("make the top bar")
    # the localizing prompt saw the ranked candidate with its source lines
    prompt = provider.calls[0]["messages"][1]["content"]
    assert "### src/components/TopBar.jsx" in prompt and "relevant lines" in prompt
    assert "ALL PLANNED FILES" in prompt
    # live progress was emitted for the UI
    assert any(p["layer"] == "revise" and p["status"] == "done" for p in store.progress)


def test_executor_falls_back_to_scan_when_model_fails(tmp_path, monkeypatch):
    _write_tree(tmp_path)
    store = _Store(_plan(tmp_path))
    monkeypatch.setattr(sr, "generate_file",
                        lambda **kw: _Outcome(kw["path"], _FILES[kw["path"]] + "/* changed */\n"))
    res, _ = _run(tmp_path, "the 'Send' button should be level with the textarea",
                  _Provider(raise_=True), store)
    assert res.artifact.content["method"] == "scan"
    assert "src/components/Composer.jsx" in res.outputs["changed"]
    # the raw feedback became the instruction
    assert res.artifact.content["changes"][0]["instruction"].startswith("the 'Send' button")


def test_executor_reports_unchanged_file_instead_of_claiming_success(tmp_path, monkeypatch):
    _write_tree(tmp_path)
    store = _Store(_plan(tmp_path))
    calls = []

    def _same(**kw):
        calls.append(kw["description"])
        return _Outcome(kw["path"], _FILES[kw["path"]])  # identical body
    monkeypatch.setattr(sr, "generate_file", _same)
    provider = _Provider(json.dumps({"changes": [
        {"path": "src/components/Composer.jsx", "instruction": "align the button"}], "note": ""}))
    res, _ = _run(tmp_path, "align the Send button with the textarea", provider, store)
    assert res.outputs["changed"] == []
    assert res.outputs["unchanged"] == ["src/components/Composer.jsx"]
    assert res.artifact.content["status"] == "nothing_changed"
    assert "unchanged" in res.outputs["reason"]
    # one firmer re-ask happened
    assert len(calls) == 2 and "returned the file UNCHANGED" in calls[1]


def test_executor_switches_theme_deterministically(tmp_path, monkeypatch):
    _write_tree(tmp_path)
    design = _design()
    design["theme"] = "conversational"
    store = _Store(_plan(tmp_path, design))
    monkeypatch.setattr(sr, "generate_file",
                        lambda **kw: pytest.fail("no model regeneration for a theme switch"))
    res, _ = _run(tmp_path, "use a dark theme", _Provider(json.dumps({"changes": []})), store)
    assert res.outputs["changed"] == ["src/styles/tokens.css"]
    t = res.artifact.content["theme"]
    assert t["from"] == "conversational" and t["to"] in ("midnight-console", "bold-dark")
    body = (tmp_path / "src/styles/tokens.css").read_text()
    assert "--color-bg" in body and len(body) > 500
    assert store.saved[-1].content["design"]["theme"] == t["to"]


def test_executor_gives_up_cleanly_when_feedback_cannot_be_localized(tmp_path, monkeypatch):
    _write_tree(tmp_path)
    store = _Store(_plan(tmp_path))
    monkeypatch.setattr(sr, "generate_file", lambda **kw: pytest.fail("nothing to apply"))
    res, _ = _run(tmp_path, "xyzzy plugh", _Provider(json.dumps({"changes": []})), store)
    assert res.failure is None
    assert res.outputs["changed"] == []
    assert "Could not map the feedback" in res.outputs["reason"]
    assert res.artifact.content["status"] == "unlocalized"


def test_executor_ignores_unplanned_and_tokens_paths_from_model(tmp_path, monkeypatch):
    _write_tree(tmp_path)
    store = _Store(_plan(tmp_path))
    monkeypatch.setattr(sr, "generate_file",
                        lambda **kw: _Outcome(kw["path"], _FILES[kw["path"]] + "/* x */\n"))
    provider = _Provider(json.dumps({"changes": [
        {"path": "src/styles/tokens.css", "instruction": "change colours"},
        {"path": "src/components/Ghost.jsx", "instruction": "invent"},
        {"path": "src/styles/app.css", "instruction": "widen .topbar"}], "note": "n"}))
    res, _ = _run(tmp_path, "make the top bar full width", provider, store)
    assert res.outputs["changed"] == ["src/styles/app.css"]


# ----------------------------- router -----------------------------

def _swarm_session(status=SessionStatus.COMPLETED):
    s = Session.new("create a chatbot app", mode=SessionMode.SWARM, project_root="/p")
    s.status = status
    return s


def _created(plan):
    return [a.task for a in plan.actions if isinstance(a, CreateTask)]


def _statuses(plan):
    return [a.status for a in plan.actions if isinstance(a, UpdateSessionStatus)]


def test_feedback_on_completed_build_reopens_and_spawns_revise():
    tasks = _built_tasks()
    plan = Router().on_user_message(session=_swarm_session(),
                                    message="make the header bigger", tasks=tasks)
    created = _created(plan)
    assert len(created) == 1 and created[0].kind is TaskKind.SWARM_REVISE
    t = created[0]
    assert t.inputs["feedback"] == "make the header bigger"
    assert t.inputs["work_plan_artifact_id"] == "wp1"
    assert t.inputs["verify_report_artifact_id"] == "vr1"
    assert t.inputs["prior_status"] == "completed"
    assert t.inputs["revision_index"] == 1
    assert t.parent_task_id is None and t.name == "Apply feedback #1"
    assert _statuses(plan) == [SessionStatus.ACTIVE]


def test_feedback_on_failed_build_remembers_prior_verdict():
    plan = Router().on_user_message(session=_swarm_session(SessionStatus.FAILED),
                                    message="fix the blank settings page",
                                    tasks=_built_tasks(verify_ok=False))
    assert _created(plan)[0].inputs["prior_status"] == "failed"


def test_question_on_built_session_still_investigates():
    plan = Router().on_user_message(session=_swarm_session(),
                                    message="how does the login work?", tasks=_built_tasks())
    assert _created(plan)[0].kind is TaskKind.EXPLORE


def test_feedback_while_build_in_flight_is_refused():
    tasks = _built_tasks() + [_task(TaskKind.SWARM_DEVELOPER, TaskNodeStatus.IN_PROGRESS,
                                    inputs={"work_plan_artifact_id": "wp1"}, created_at=9)]
    plan = Router().on_user_message(session=_swarm_session(SessionStatus.ACTIVE),
                                    message="make it blue", tasks=tasks)
    assert plan.actions == []


def test_message_without_a_build_takes_the_normal_root(tmp_path):
    s = Session.new("x", mode=SessionMode.SWARM, project_root=str(tmp_path))
    only_plan = [_task(TaskKind.SWARM_TECH_LEAD, TaskNodeStatus.DONE,
                       outputs={"work_plan_artifact_id": "wp1", "file_count": 0})]
    plan = Router().on_user_message(session=s, message="build a todo app", tasks=only_plan)
    assert _created(plan)[0].kind is TaskKind.SWARM_TECH_LEAD


def test_revise_with_changes_chains_to_verify():
    parent = _task(TaskKind.SWARM_REVISE, TaskNodeStatus.DONE,
                   inputs={"work_plan_artifact_id": "wp1", "revision_index": 2, "goal": "g",
                           "project_root": "/p"},
                   outputs={"changed": ["a.jsx"], "failed_paths": [], "goal": "g",
                            "project_root": "/p", "work_plan_artifact_id": "wp1",
                            "revision_artifact_id": "rev1", "revision_index": 2})
    kids = _swarm_revise_to_successors(parent)
    assert len(kids) == 1 and kids[0].kind is TaskKind.SWARM_VERIFY
    assert kids[0].parent_task_id == parent.task_id
    assert kids[0].inputs["work_plan_artifact_id"] == "wp1"
    assert kids[0].inputs["revision_artifact_id"] == "rev1"
    assert kids[0].inputs["revision_index"] == 2
    assert _swarm_revise_to_successors(_task(
        TaskKind.SWARM_REVISE, TaskNodeStatus.DONE, outputs={"changed": []})) == []


def test_revise_that_changed_nothing_restores_prior_verdict_with_reason():
    completed = _task(TaskKind.SWARM_REVISE, TaskNodeStatus.DONE,
                      inputs={"prior_status": "completed"},
                      outputs={"changed": [], "reason": "Could not map the feedback to any file"})
    plan = Router().on_task_completed(session=_swarm_session(SessionStatus.ACTIVE),
                                      completed=completed, tasks=[completed])
    assert _statuses(plan) == [SessionStatus.COMPLETED]
    errs = [a for a in plan.actions if isinstance(a, UpdateTaskStatus)]
    assert len(errs) == 1 and "Could not map" in errs[0].error
    assert errs[0].status is TaskNodeStatus.DONE


def test_crashed_revision_restores_prior_verdict():
    failed = _task(TaskKind.SWARM_REVISE, TaskNodeStatus.FAILED,
                   inputs={"prior_status": "completed"})
    plan = Router().on_task_failed(session=_swarm_session(SessionStatus.ACTIVE),
                                   failed=failed, tasks=[failed])
    assert _statuses(plan) == [SessionStatus.COMPLETED]


# ----------------------------- CLI continuation -----------------------------

def test_cli_agent_events_continues_a_finished_swarm_build(monkeypatch):
    from cgx.cli.tui import ops

    session = _swarm_session()
    tasks = _built_tasks()
    for t in tasks:
        t.session_id = session.session_id

    class _FakeStore:
        def get_session(self, sid):
            return session if sid == session.session_id else None

        def list_tasks(self, sid):
            return tasks

    class _FakePlan:
        actions = [object()]

    class _FakeRunner:
        store = _FakeStore()
        posted = []
        started = []

        def post_message(self, *, session_id, message):
            self.posted.append((session_id, message))
            return _FakePlan()

        def start_session(self, **kw):
            self.started.append(kw)
            return session

    runner = _FakeRunner()
    monkeypatch.setattr(ops, "_session_runner", lambda root: runner)
    monkeypatch.setattr(ops, "_session_deps", lambda state, **kw: object())
    monkeypatch.setattr(ops, "_drive_session",
                        lambda state, r, sid, deps, **kw: iter([("session_done", {"status": "x"})]))

    class _State:
        project_root = "/p"
        agent_session_id = session.session_id
        pending_ask = None

    events = list(ops.agent_events(_State(), "make the header bigger"))
    assert runner.posted == [(session.session_id, "make the header bigger")]
    assert runner.started == []
    assert events[0][0] == "status" and "revising" in events[0][1]["message"] or events[0][1]["message"]


def test_cli_agent_events_refuses_feedback_mid_build(monkeypatch):
    from cgx.cli.tui import ops

    session = _swarm_session(SessionStatus.ACTIVE)
    tasks = _built_tasks() + [_task(TaskKind.SWARM_DEVELOPER, TaskNodeStatus.READY,
                                    inputs={"work_plan_artifact_id": "wp1"}, created_at=9)]

    class _FakeStore:
        def get_session(self, sid):
            return session

        def list_tasks(self, sid):
            return tasks

    class _FakePlan:
        actions = []

    class _FakeRunner:
        store = _FakeStore()

        def post_message(self, **kw):
            return _FakePlan()

    monkeypatch.setattr(ops, "_session_runner", lambda root: _FakeRunner())
    monkeypatch.setattr(ops, "_session_deps", lambda state, **kw: object())

    class _State:
        project_root = "/p"
        agent_session_id = session.session_id
        pending_ask = None

    events = list(ops.agent_events(_State(), "make it blue"))
    assert events == [("error", {"message": ("this build is still running; wait for it to "
                                             "finish (or stop it) before sending feedback")})]


def test_cli_renders_revision_and_verdict_events():
    from cgx.cli.tui.ops import map_event
    r = map_event("revision", {"changed": ["a.jsx"], "unchanged": ["b.css"], "note": "Widened the bar",
                               "index": 2}, enabled=False)
    assert "revision #2" in r.text and "changed a.jsx" in r.text and "unchanged: b.css" in r.text
    assert "Widened the bar" in r.text
    r = map_event("verdict", {"summary": "verified: built 3/3 files", "ok": True}, enabled=False)
    assert "verified: built 3/3 files" in r.text


# ----------------------------- CLI review (cgx session preview) -----------------------------

def test_build_review_summarises_verdict_screens_and_revisions(tmp_path):
    from cgx.cli.commands.session import build_review

    class _A:
        def __init__(self, kind, content, created_at):
            self.kind, self.content, self.created_at = kind, content, created_at

    arts = [
        _A(ArtifactKind.WORK_PLAN, {"design": {"app_root": ""}}, 1),
        _A(ArtifactKind.SWARM_VERIFY_REPORT,
           {"verify_ok": False, "summary": "partial build: ...", "runtime": {"routes": [
               {"route": "/", "mounted": True, "errors": [], "screenshot": "/p/.cgx/screenshots/index.png"}]}}, 2),
        _A(ArtifactKind.SWARM_REVISION,
           {"revision_index": 1, "feedback": "widen the bar", "status": "applied",
            "note": "done", "changed": ["a.css"], "unchanged": [], "failed": []}, 3),
        _A(ArtifactKind.SWARM_VERIFY_REPORT,
           {"verify_ok": True, "summary": "verified: built 2/2 files", "runtime": {"routes": [
               {"route": "/", "mounted": True, "errors": [], "screenshot": "/p/.cgx/screenshots/index.png"},
               {"route": "/settings", "mounted": False, "errors": ["boom"], "screenshot": None}]}}, 4),
    ]

    class _S:
        def list_artifacts(self, sid):
            return arts

    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "index.html").write_text("<html></html>")
    review = build_review(_S(), "ses_1", str(tmp_path))
    assert review["verify_ok"] is True and review["summary"] == "verified: built 2/2 files"
    assert [(r["route"], r["ok"]) for r in review["routes"]] == [("/", True), ("/settings", False)]
    assert review["revisions"] == [{"index": 1, "feedback": "widen the bar", "status": "applied",
                                    "note": "done", "changed": ["a.css"], "unchanged": [], "failed": []}]
    assert review["dist"] == str(tmp_path / "dist")
    empty = build_review(type("S", (), {"list_artifacts": lambda self, sid: []})(), "x", str(tmp_path / "nope"))
    assert empty["summary"] is None and empty["routes"] == [] and empty["dist"] is None


def test_cli_session_preview_verb_is_registered():
    from cgx.cli.main import build_parser
    parser = build_parser()
    ns = parser.parse_args(["session", "preview", "ses_1", "--no-serve", "--project-root", "/p"])
    assert ns.session_cmd == "preview" and ns.no_serve is True and ns.sid == "ses_1"


# ----------------------------- multi-part feedback, CSS awareness, apply coverage -----------------------------

def test_split_feedback_into_actionable_parts():
    from cgx.session.revise import split_feedback
    assert split_feedback(
        "make the top bar span the full width of the page, and put the Send button on the "
        "same row as the textarea, aligned to its bottom edge") == [
        "make the top bar span the full width of the page",
        "put the Send button on the same row as the textarea, aligned to its bottom edge"]
    assert split_feedback("The header is too small. Also the footer text is cut off; and add a logo") == [
        "The header is too small", "Also the footer text is cut off", "and add a logo"]
    assert split_feedback("make it blue") == ["make it blue"]
    assert split_feedback("1) bigger font 2) darker background") == ["bigger font", "darker background"]
    assert split_feedback("") == []


def test_instruction_tokens_and_diff_coverage():
    from cgx.session.revise import diff_covers, instruction_tokens
    toks = instruction_tokens("Remove the max-width from the .topbar class and set width 100%; "
                              "pass the `disabled` prop to Composer")
    assert ".topbar" in toks and "max-width" in toks and "100%" in toks
    assert "disabled" in toks  # quoted identifiers count; a lone Capitalised word does not
    assert "Remove" not in toks
    diff = "--- a/x.css\n+++ b/x.css\n-.topbar { max-width: 820px; }\n+.topbar { width: 100%; }\n"
    assert set(diff_covers(diff, toks)) == {".topbar", "max-width", "100%"}
    assert diff_covers("+.composer { align-items: flex-end }", toks) == []
    assert instruction_tokens("improve the header") == []


def test_css_rules_and_relevant_blocks_show_the_constraining_parent():
    from cgx.session.revise import component_classes, css_rules, relevant_css_blocks, rules_for_classes
    css = ("/* shell */\n.app { display: flex; flex-direction: column; align-items: center; }\n"
           ".main { width: 100%; max-width: var(--chat-max-width); }\n"
           "@media (max-width: 768px) {\n  .topbar { padding: 0 8px; }\n}\n"
           ".btn { color: red }\n")
    rules = css_rules(css)
    assert [r["selector"] for r in rules] == [".app", ".main", ".topbar", ".btn"]
    assert rules[2]["media"].startswith("@media") and rules[2]["line"] == 5
    # a width complaint pulls in .main (body mentions width) and, via the shell's
    # classes, the .app container whose align-items:center narrows the header
    blocks = relevant_css_blocks(css, "make the top bar span the full width", ["app", "topbar"])
    assert ".app {" in blocks and ".main {" in blocks and ".topbar {" in blocks and ".btn" not in blocks
    assert component_classes('<header className="topbar"><nav className="nav nav--x">') == ["topbar", "nav", "nav--x"]
    assert [r["selector"] for r in rules_for_classes(css, ["topbar"])] == [".topbar"]


def test_executor_covers_every_feedback_part(tmp_path, monkeypatch):
    """A two-part message: the model answers only part 2 -> part 1 is added from its own scan."""
    _write_tree(tmp_path)
    store = _Store(_plan(tmp_path))
    applied = []

    def _gen(**kw):
        applied.append(kw["description"])
        return _Outcome(kw["path"], (_safe_read_now(tmp_path, kw["path"]) or "") + "\n/* changed */\n")
    monkeypatch.setattr(sr, "generate_file", _gen)
    provider = _Provider(json.dumps({"changes": [
        {"path": "src/styles/app.css", "part": 2,
         "instruction": "set align-items: flex-end on .composer"}], "note": "n"}))
    res, _ = _run(tmp_path,
                  "make the top bar span the full width of the page, and put the Send button "
                  "level with the textarea", provider, store)
    c = res.artifact.content
    assert c["parts"] == ["make the top bar span the full width of the page",
                          "put the Send button level with the textarea"]
    parts_done = sorted((ch["part"], ch["path"]) for ch in c["changes"])
    assert (1, "src/components/TopBar.jsx") in parts_done   # added for the uncovered part
    assert (2, "src/styles/app.css") in parts_done
    assert c["method"] == "model+scan"
    assert any(ch["source"] == "scan:part" for ch in c["changes"])
    # the localizing prompt listed the parts and the CSS rules behind the shell
    prompt = provider.calls[0]["messages"][1]["content"]
    assert "FEEDBACK PARTS" in prompt and "1) make the top bar" in prompt
    assert "styles for this component and its container" in prompt
    assert ".topbar {" in prompt


def _safe_read_now(root, path):
    full = root / path
    return full.read_text() if full.is_file() else None


def test_apply_reasks_when_the_diff_misses_what_the_instruction_names(tmp_path, monkeypatch):
    _write_tree(tmp_path)
    store = _Store(_plan(tmp_path))
    calls = []

    def _gen(**kw):
        calls.append(kw["description"])
        cur = _safe_read_now(tmp_path, kw["path"]) or ""
        if len(calls) == 1:
            return _Outcome(kw["path"], cur.replace(".composer { display: flex; }",
                                                  ".composer { display: flex; gap: 4px; }"))
        return _Outcome(kw["path"], cur.replace("max-width: 960px", "width: 100%"))
    monkeypatch.setattr(sr, "generate_file", _gen)
    provider = _Provider(json.dumps({"changes": [
        {"path": "src/styles/app.css", "part": 1,
         "instruction": "in .topbar replace max-width: 960px with width: 100%"}], "note": "n"}))
    res, _ = _run(tmp_path, "make the top bar full width", provider, store)
    row = res.artifact.content["changes"][0]
    assert len(calls) == 2 and "did NOT touch" in calls[1] and ".topbar" in calls[1]
    assert row["outcome"] == "changed" and row["partial"] is False
    assert ".topbar" in row["covered"] and "max-width" in row["covered"]
    assert "width: 100%" in (tmp_path / "src/styles/app.css").read_text()
    assert "Caution" not in res.outputs["note"]


def test_apply_reports_partial_when_the_instruction_is_never_honoured(tmp_path, monkeypatch):
    _write_tree(tmp_path)
    store = _Store(_plan(tmp_path))
    monkeypatch.setattr(sr, "generate_file", lambda **kw: _Outcome(
        kw["path"], (_safe_read_now(tmp_path, kw["path"]) or "") + "\n.extra { color: red }\n"))
    provider = _Provider(json.dumps({"changes": [
        {"path": "src/styles/app.css", "part": 1,
         "instruction": "remove max-width from .topbar"}], "note": ""}))
    res, _ = _run(tmp_path, "make the top bar full width", provider, store)
    c = res.artifact.content
    assert c["changes"][0]["partial"] is True and c["partial"] == ["src/styles/app.css"]
    assert "Caution" in res.outputs["note"] and ".topbar" in res.outputs["note"]
    # the file changed, so the revision still counts as applied and goes to verification
    assert res.outputs["changed"] == ["src/styles/app.css"]
    assert store.saved[-1].content["revisions"][0]["instructions"] == {
        "src/styles/app.css": ["remove max-width from .topbar"]}


def test_verifier_regenerates_revised_files_as_modifications(tmp_path, monkeypatch):
    from cgx.session.tasks import swarm_verify as sv
    seen = {}

    def _gen(**kw):
        seen.update(kw)
        return type("O", (), {"ok": True, "content": "x", "method": "full-file"})()
    monkeypatch.setattr(sv, "generate_file", _gen)
    monkeypatch.setattr(sv, "edit_file", lambda *a, **k: "ok")
    notes = sv._revision_notes({"revisions": [
        {"feedback": "make the top bar full width", "changed": ["a.css", "b.jsx"],
         "instructions": {"a.css": ["in .app set align-items: stretch"]}}]})
    assert notes == {"a.css": ["in .app set align-items: stretch"],
                     "b.jsx": ["make the top bar full width"]}
    sv._regenerate(["a.css"], {"a.css": {"description": "app styles"}}, {}, "g", str(tmp_path),
                   object(), ["a.css", "b.jsx"], ["react"], directives={"a.css": "define .x"},
                   revisions=notes)
    assert seen["modify_existing"] is True
    assert seen["existing_limit"] == sv._REVISED_EXISTING_LIMIT
    assert "USER FEEDBACK ALREADY APPLIED TO THIS FILE" in seen["description"]
    assert "align-items: stretch" in seen["description"] and "define .x" in seen["description"]
    # a file no revision touched is regenerated from its spec as before
    sv._regenerate(["b.css"], {"b.css": {"description": "other"}}, {}, "g", str(tmp_path),
                   object(), ["b.css"], ["react"], revisions=notes)
    assert seen["modify_existing"] is False and "ALREADY APPLIED" not in seen["description"]


# ----------------------------- visual acceptance of the feedback + bounded auto follow-up -----------------------------

def test_parse_feedback_verdict():
    from cgx.design.vision import parse_feedback_verdict
    v = parse_feedback_verdict("NOT SATISFIED: the header is still only as wide as its content", routes=["index"], model="m")
    assert v["satisfied"] is False and v["verdict"] == "not_satisfied"
    assert v["evidence"].startswith("the header is still") and v["routes"] == ["index"] and v["model"] == "m"
    assert parse_feedback_verdict("`SATISFIED`")["satisfied"] is True
    assert parse_feedback_verdict("UNDECIDABLE: behaviour")["satisfied"] is None
    assert parse_feedback_verdict("")["verdict"] == "undecidable"


def test_check_feedback_sends_before_and_after_images(tmp_path, monkeypatch):
    from cgx.design import vision
    (tmp_path / "b.png").write_bytes(b"before")
    (tmp_path / "a.png").write_bytes(b"after")
    sent = {}

    class _Resp:
        def json(self):
            return {"message": {"content": "NOT SATISFIED: still narrow"}}

    class _Req:
        @staticmethod
        def post(url, json=None, timeout=None):
            sent["url"] = url
            sent["json"] = json
            return _Resp()
    monkeypatch.setitem(__import__("sys").modules, "requests", _Req)
    v = vision.check_feedback("make the top bar full width", {"index": str(tmp_path / "b.png")},
                              {"index": str(tmp_path / "a.png")}, provider=None,
                              model=("http://localhost:11434", "qwen3-vl:8b"))
    assert v["satisfied"] is False and v["evidence"] == "still narrow" and v["routes"] == ["index"]
    msg = sent["json"]["messages"][0]
    assert len(msg["images"]) == 2 and "index BEFORE, index AFTER" in msg["content"]
    assert "make the top bar full width" in msg["content"]
    assert vision.check_feedback("x", {}, {}, provider=None, model=("u", "m")) is None


def test_verifier_runs_the_feedback_check_and_marks_the_revision(tmp_path, monkeypatch):
    from cgx.session.tasks import swarm_verify as sv

    before_dir = tmp_path / ".cgx" / "screenshots" / "before-rev1"
    before_dir.mkdir(parents=True)
    (before_dir / "index.png").write_bytes(b"b")
    after_png = tmp_path / ".cgx" / "screenshots" / "index.png"
    after_png.write_bytes(b"a")

    class _Rev:
        content = {"feedback": "make the top bar full width",
                   "before_screenshots": {"index": str(before_dir / "index.png")}}
        artifact_id = "rev1"

    class _S:
        saved = []

        def get_artifact(self, aid):
            return _Rev() if aid == "rev1" else None

        def save_artifact(self, art):
            self.saved.append(art)

    seen = {}
    import cgx.design.vision as vision
    monkeypatch.setattr(vision, "check_feedback", lambda fb, before, after, provider, model=None: (
        seen.update({"fb": fb, "before": before, "after": after})
        or {"satisfied": False, "verdict": "not_satisfied", "evidence": "header still narrow",
            "routes": ["index"], "model": "qwen3-vl:8b"}))
    store = _S()
    deps = ExecutorDeps(project_root=str(tmp_path), provider=object(), store=store)
    task = TaskNode.new(session_id="s", kind=TaskKind.SWARM_VERIFY, name="v",
                        inputs={"revision_artifact_id": "rev1", "feedback": "make the top bar full width"})
    smoke = {"ran": True, "result": {"routes": [{"route": "/", "screenshot": str(after_png)}]}}
    verdict = sv._check_feedback_addressed(task, deps, smoke, str(tmp_path))
    assert verdict["satisfied"] is False
    assert seen["fb"] == "make the top bar full width"
    assert seen["before"] == {"index": str(before_dir / "index.png")} and seen["after"] == {"index": str(after_png)}
    assert store.saved and store.saved[0].content["visual_check"]["evidence"] == "header still narrow"
    # not a revision verify / no smoke -> no check
    assert sv._check_feedback_addressed(TaskNode.new(session_id="s", kind=TaskKind.SWARM_VERIFY, name="v"),
                                        deps, smoke, str(tmp_path)) is None
    assert sv._check_feedback_addressed(task, deps, None, str(tmp_path)) is None
    summary = sv._verify_summary(["a"], ["a"], [], [], {"outcome": "passed"}, True,
                                 feedback_check=verdict)
    assert "feedback NOT visibly addressed (header still narrow)" in summary
    assert "feedback visibly addressed" in sv._verify_summary(
        ["a"], ["a"], [], [], {"outcome": "passed"}, True, feedback_check={"satisfied": True})


def _verify_after_revision(satisfied, auto_round=0, budget=1):
    return _task(TaskKind.SWARM_VERIFY, TaskNodeStatus.DONE,
                 inputs={"revision_artifact_id": "rev1", "revision_index": 1,
                         "feedback": "make the top bar full width", "prior_status": "completed",
                         "auto_round": auto_round, "auto_rounds_max": budget,
                         "work_plan_artifact_id": "wp1", "goal": "g", "project_root": "/p"},
                 outputs={"verify_ok": True, "verify_report_artifact_id": "vr2",
                          "revision_artifact_id": "rev1", "revision_index": 1,
                          "feedback": "make the top bar full width",
                          "feedback_satisfied": satisfied,
                          "feedback_evidence": "header still narrow",
                          "auto_round": auto_round, "auto_rounds_max": budget,
                          "work_plan_artifact_id": "wp1", "goal": "g", "project_root": "/p",
                          "summary": "verified: ...; feedback NOT visibly addressed"})


def test_unmet_feedback_spawns_one_automatic_follow_up_revision():
    from cgx.session.router import _swarm_verify_auto_followup
    kids = _swarm_verify_auto_followup(_verify_after_revision(False))
    assert len(kids) == 1 and kids[0].kind is TaskKind.SWARM_REVISE
    k = kids[0]
    assert k.name == "Apply feedback #2 (auto follow-up)"
    assert k.inputs["auto_round"] == 1 and k.inputs["auto_rounds_max"] == 1
    assert k.inputs["revision_index"] == 2 and k.inputs["prior_status"] == "completed"
    assert k.inputs["work_plan_artifact_id"] == "wp1" and k.inputs["verify_report_artifact_id"] == "vr2"
    assert k.inputs["feedback"].startswith("make the top bar full width")
    assert "automatic follow-up #1" in k.inputs["feedback"] and "header still narrow" in k.inputs["feedback"]
    # budget spent, satisfied, undecidable, or a plain build verify -> terminal as before
    assert _swarm_verify_auto_followup(_verify_after_revision(False, auto_round=1)) == []
    assert _swarm_verify_auto_followup(_verify_after_revision(True)) == []
    assert _swarm_verify_auto_followup(_verify_after_revision(None)) == []
    assert _swarm_verify_auto_followup(_verify_after_revision(False, budget=0)) == []
    plain = _task(TaskKind.SWARM_VERIFY, TaskNodeStatus.DONE, outputs={"verify_ok": True})
    assert _swarm_verify_auto_followup(plain) == []


def test_router_keeps_session_open_while_the_follow_up_runs():
    completed = _verify_after_revision(False)
    plan = Router().on_task_completed(session=_swarm_session(SessionStatus.ACTIVE),
                                      completed=completed, tasks=[completed])
    assert [a.task.kind for a in plan.actions if isinstance(a, CreateTask)] == [TaskKind.SWARM_REVISE]
    assert _statuses(plan) == []
    # once the budget is spent the verify is terminal again
    done = _verify_after_revision(False, auto_round=1)
    plan = Router().on_task_completed(session=_swarm_session(SessionStatus.ACTIVE),
                                      completed=done, tasks=[done])
    assert _statuses(plan) == [SessionStatus.COMPLETED]


def test_feedback_autofix_rounds_env(monkeypatch):
    from cgx.session.router import feedback_autofix_rounds
    monkeypatch.delenv("CGX_FEEDBACK_AUTOFIX", raising=False)
    assert feedback_autofix_rounds() == 1
    monkeypatch.setenv("CGX_FEEDBACK_AUTOFIX", "off")
    assert feedback_autofix_rounds() == 0
    monkeypatch.setenv("CGX_FEEDBACK_AUTOFIX", "2")
    assert feedback_autofix_rounds() == 2
    monkeypatch.setenv("CGX_FEEDBACK_AUTOFIX", "9")
    assert feedback_autofix_rounds() == 3


def test_revise_snapshots_screenshots_before_applying(tmp_path, monkeypatch):
    _write_tree(tmp_path)
    shots = tmp_path / ".cgx" / "screenshots"
    shots.mkdir(parents=True)
    (shots / "index.png").write_bytes(b"png")
    store = _Store(_plan(tmp_path))
    monkeypatch.setattr(sr, "generate_file",
                        lambda **kw: _Outcome(kw["path"], _FILES[kw["path"]] + "/* x */\n"))
    provider = _Provider(json.dumps({"changes": [
        {"path": "src/styles/app.css", "part": 1, "instruction": "widen .topbar"}], "note": "n"}))
    res, _ = _run(tmp_path, "make the top bar full width", provider, store)
    before = res.artifact.content["before_screenshots"]
    assert set(before) == {"index"} and os.path.isfile(before["index"])
    assert before["index"].endswith("before-rev1/index.png")
    assert res.outputs["auto_round"] == 0


def test_revise_successor_threads_feedback_and_auto_budget():
    parent = _task(TaskKind.SWARM_REVISE, TaskNodeStatus.DONE,
                   inputs={"feedback": "f", "prior_status": "failed", "auto_round": 0,
                           "auto_rounds_max": 1, "work_plan_artifact_id": "wp1"},
                   outputs={"changed": ["a"], "failed_paths": [], "revision_artifact_id": "r1",
                            "revision_index": 1, "auto_round": 0, "auto_rounds_max": 1})
    kid = _swarm_revise_to_successors(parent)[0]
    assert kid.inputs["feedback"] == "f" and kid.inputs["prior_status"] == "failed"
    assert kid.inputs["auto_round"] == 0 and kid.inputs["auto_rounds_max"] == 1


# ----------------------------- grounding the instruction in the current file -----------------------------

def test_grounding_notes_call_out_selectors_the_file_does_not_define():
    from cgx.session.tasks.swarm_revise import grounding_notes
    css = "body { display: flex; align-items: center; }\n.main { width: 100%; }\n"
    notes = grounding_notes("in .app set align-items: stretch so the header spans the page", css,
                            "src/styles/app.css")
    assert len(notes) == 1 and "`.app` has NO rule in this file yet" in notes[0] and "ADD a new" in notes[0]
    assert grounding_notes("in .main remove width", css, "src/styles/app.css") == []
    # ``.main`` must not be confused with ``.main-x``
    assert grounding_notes("edit .main-x", css, "src/styles/app.css")[0].startswith("`.main-x` has NO rule")
    jsx = "export default function TopBar() { return <header className=\"topbar\" />; }\n"
    assert grounding_notes("pass the `disabled` prop into TopBar", jsx, "src/components/TopBar.jsx") == [
        "`disabled` does not occur in this file yet: ADD it as the change requires."]


def test_apply_prompt_includes_grounding_note_and_never_asks_to_echo(tmp_path, monkeypatch):
    _write_tree(tmp_path)
    store = _Store(_plan(tmp_path))
    seen = []

    def _gen(**kw):
        seen.append(kw["description"])
        cur = _safe_read_now(tmp_path, kw["path"]) or ""
        return _Outcome(kw["path"], cur + "\n.header-row { width: 100%; align-items: stretch; }\n")
    monkeypatch.setattr(sr, "generate_file", _gen)
    provider = _Provider(json.dumps({"changes": [
        {"path": "src/styles/app.css", "part": 1,
         "instruction": "in .header-row set align-items: stretch for the header row"}], "note": "n"}))
    # an alignment ask (no width complaint) -> no deterministic width diagnosis interferes
    res, _ = _run(tmp_path, "align the items in the header row to the end", provider, store)
    assert len(seen) == 1
    assert "NOTE: `.header-row` has NO rule in this file yet" in seen[0]
    assert "Never return the file unchanged" in seen[0]
    assert res.outputs["changed"] == ["src/styles/app.css"]
    assert res.artifact.content["changes"][0]["covered"] == [".header-row", "align-items"]


def test_cli_renders_each_completion_once():
    from cgx.cli.tui import ops
    from cgx.session.events import Event, EventType

    class _Ev:
        def __init__(self, type_, payload):
            self.type, self.payload, self.session_id = type_, payload, "s"
    done = _Ev(EventType.TASK_COMPLETED, {"task_id": "t1", "kind": "swarm_revise",
                                          "outputs": {"changed": ["a"], "note": "n", "revision_index": 1}})
    seen = set()
    first = list(ops._bus_to_tui(done, seen))
    again = list(ops._bus_to_tui(done, seen))
    assert [e[0] for e in first] == ["task_done", "revision"]
    assert again == []
    # without a de-dup set the translation stays stateless (legacy callers)
    assert [e[0] for e in ops._bus_to_tui(done)] == ["task_done", "revision"]


# ----------------------------- DOM ancestry context + decisive layout judge -----------------------------

def test_ancestor_chain_walks_importers_to_the_document_root():
    from cgx.session.revise import ancestor_chain, mount_selector, rules_for_selector
    from cgx.session.tasks.swarm_revise import _importers_of, _ancestry_block
    contents = dict(_FILES)
    contents["index.html"] = '<!doctype html><html><body><div id="root"></div><script type="module" src="/src/main.jsx"></script></body></html>'
    contents["src/main.jsx"] = "import { createRoot } from 'react-dom/client';\nimport App from './App.jsx';\ncreateRoot(document.getElementById('root')).render(<App />);\n"
    contents["src/styles/app.css"] = "body { display: flex; justify-content: center; align-items: center; }\n.topbar { max-width: 960px; }\n"
    paths = list(contents)
    assert mount_selector(contents) == "#root"
    chain = ancestor_chain("src/components/TopBar.jsx", contents, paths, importers_of=_importers_of)
    assert [r["selector"] for r in chain] == ["html", "body", "#root", ".app", ".topbar"]
    assert chain[3]["source"] == "src/App.jsx" and chain[4]["source"] == "src/components/TopBar.jsx"
    block = _ancestry_block(chain, contents, paths, "src/styles/tokens.css")
    assert "body  (document)" in block and "justify-content: center" in block
    assert "#root  (mount point (entry / index.html)) -- NO rule in any stylesheet" in block
    assert ".app  (src/App.jsx) -- NO rule" in block
    assert [r["selector"] for r in rules_for_selector(contents["src/styles/app.css"], "body")] == ["body"]
    assert rules_for_selector(".topbar-x { a: b }\n", ".topbar") == []


def test_feedback_check_reasks_once_when_a_layout_verdict_hedges(tmp_path, monkeypatch):
    from cgx.design import vision
    (tmp_path / "a.png").write_bytes(b"after")
    replies = iter(["UNDECIDABLE", "NOT SATISFIED: the header still stops well short of the page edges"])
    calls = []

    class _Resp:
        def __init__(self, text):
            self._t = text

        def json(self):
            return {"message": {"content": self._t}}

    class _Req:
        @staticmethod
        def post(url, json=None, timeout=None):
            calls.append(json)
            return _Resp(next(replies))
    monkeypatch.setitem(__import__("sys").modules, "requests", _Req)
    v = vision.check_feedback("make the top bar span the full width", {}, {"index": str(tmp_path / "a.png")},
                              provider=None, model=("http://localhost:11434", "qwen3-vl:8b"))
    assert v["satisfied"] is False and v.get("reasked") is True
    assert len(calls) == 2 and len(calls[1]["messages"]) == 3
    assert "decidable from the screenshots" in calls[1]["messages"][2]["content"]
    assert "ALWAYS judgeable" in calls[0]["messages"][0]["content"]


def test_width_diagnosis_names_the_centring_ancestor_not_the_element():
    from cgx.session.revise import chain_with_rules, diagnose_width_constraints
    chain = [{"selector": "html", "source": "document"}, {"selector": "body", "source": "document"},
             {"selector": "#root", "source": "mount"}, {"selector": ".app", "source": "App.jsx"},
             {"selector": ".topbar", "source": "TopBar.jsx"}]
    css = {"app.css": "body { display: flex; justify-content: center; align-items: center; height: 100vh; }\n"
                      ".app { display: flex; flex-direction: column; align-items: stretch; width: 100%; }\n"
                      "@media (max-width: 600px) { .app { width: auto; } }\n"}
    rich = chain_with_rules(chain, css)
    assert rich[1]["rules"]["justify-content"] == "center" and rich[3]["rules"]["width"] == "100%"
    diag = diagnose_width_constraints(rich, "the top bar still does not span the full width of the page")
    assert diag[0]["ring"] == "body" and diag[0]["problem"].startswith("body is a ROW flex container")
    assert diag[0]["fix"] == "body { display: block; }"
    assert any(d["ring"] == "#root" and d["fix"] == "#root { width: 100%; }" for d in diag)
    assert not any(d["ring"] == ".topbar" for d in diag)
    # a column flex container that centres its children, and a max-width cap
    css2 = {"a.css": ".app { display: flex; flex-direction: column; align-items: center; }\n.main { max-width: 820px; }\n"}
    rich2 = chain_with_rules([{"selector": ".app", "source": "x"}, {"selector": ".main", "source": "y"}], css2)
    diag2 = diagnose_width_constraints(rich2, "make the header full width")
    assert diag2[0]["fix"] == ".app { align-items: stretch; }"
    assert diag2[1]["fix"] == ".main { max-width: none; }"
    # not a width complaint -> no diagnosis
    assert diagnose_width_constraints(rich, "make the Send button blue") == []


def test_noop_css_instruction_is_detected():
    from cgx.session.revise import instruction_is_noop
    sheets = {"app.css": ".app { display: flex; align-items: stretch; width: 100%; }\nbody { display: flex; }\n"}
    assert instruction_is_noop("in .app set align-items: stretch so the header spans the page", sheets)
    assert instruction_is_noop("add width: 100% to .app", sheets)
    assert instruction_is_noop("in .app set align-items: center", sheets) is None
    assert instruction_is_noop("in body set display: block", sheets) is None
    assert instruction_is_noop("in #root set width: 100%", sheets) is None  # no rule yet -> not a no-op
    assert instruction_is_noop("improve the header", sheets) is None


def test_executor_drops_noop_instruction_and_fixes_the_diagnosed_ring(tmp_path, monkeypatch):
    _write_tree(tmp_path)
    (tmp_path / "index.html").write_text('<html><body><div id="root"></div></body></html>')
    css = ("body { display: flex; justify-content: center; align-items: center; }\n"
           ".app { display: flex; flex-direction: column; align-items: stretch; width: 100%; }\n"
           ".topbar { padding: 0; }\n")
    (tmp_path / "src/styles/app.css").write_text(css)
    plan = _plan(tmp_path)
    plan["paths"] = list(_FILES) + ["index.html"]
    store = _Store(plan)
    seen = []

    def _gen(**kw):
        seen.append(kw["description"])
        cur = _safe_read_now(tmp_path, kw["path"]) or ""
        return _Outcome(kw["path"], cur.replace("body { display: flex;", "body { display: block;"))
    monkeypatch.setattr(sr, "generate_file", _gen)
    provider = _Provider(json.dumps({"changes": [
        {"path": "src/styles/app.css", "part": 1,
         "instruction": "in .app set align-items: stretch so the header spans the page"}], "note": "n"}))
    res, _ = _run(tmp_path, "the top bar still does not span the full width of the page", provider, store)
    c = res.artifact.content
    assert c["method"] == "model+diagnosis"
    assert len(c["changes"]) == 1
    ch = c["changes"][0]
    assert ch["source"] == "diagnosis" and ch["path"] == "src/styles/app.css"
    assert "body { display: block; }" in ch["instruction"] and "ROW flex container" in ch["instruction"]
    assert ch["outcome"] == "changed"
    assert "display: block" in (tmp_path / "src/styles/app.css").read_text()
    assert "LAYOUT DIAGNOSIS" in provider.calls[0]["messages"][1]["content"]
