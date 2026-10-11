"""The design layer wired through the swarm: Tech Lead plan injection, the
Developer's deterministic tokens file + per-file prompt block, and the
Verifier's design checks / regen directives / gate."""

from __future__ import annotations

import json

import pytest

import cgx.answer.engine as engine
from cgx import design
from cgx.session.models import TaskKind, TaskNode
from cgx.session.tasks import swarm_generate as sg
from cgx.session.tasks import swarm_tech_lead as tl
from cgx.session.tasks import swarm_verify as sv
from cgx.session.tasks.base import ExecutorDeps
from cgx.session.tasks.swarm_plan import ensure_ui_scaffolding, normalize_plan, ordered_paths

GOAL = ("create an app for a chatbot in which user provides a API address and a key on the "
        "profile tab and then navigates to the main page and can interact with the chat. "
        "Use React frontend")


# --------------------------------------------------------------------------
# plan-level
# --------------------------------------------------------------------------

def test_ensure_ui_scaffolding_injects_design_system_files_and_respects_vetoes():
    plan = normalize_plan({"goal": GOAL, "layers": [{"name": "ui", "files": [
        {"path": "frontend/index.html", "description": "entry"},
        {"path": "frontend/package.json", "description": "manifest"},
        {"path": "frontend/src/main.jsx", "description": "mount"},
        {"path": "frontend/src/App.jsx", "description": "shell"},
        {"path": "frontend/src/Chat.jsx", "description": "chat"}]}]})
    brief = design.build_brief(GOAL, ["react"]).resolved_for_paths(ordered_paths(plan))
    out = ensure_ui_scaffolding(plan, brief.to_dict())
    paths = set(ordered_paths(out))
    assert {"frontend/src/styles/tokens.css", "frontend/src/styles/app.css",
            "frontend/src/lib/api.js", "frontend/src/lib/storage.js"} <= paths
    # dependency edges only reference planned paths (never a dangling hint)
    for lay in out["layers"]:
        for f in lay["files"]:
            assert all(d in paths for d in f.get("depends_on", []))
    # non-UI brief / None brief -> untouched
    assert ensure_ui_scaffolding(plan, None) is plan
    assert ensure_ui_scaffolding(plan, {"applies": False}) is plan

    class _Veto:
        def forbids_scaffold_path(self, path):
            return path.endswith("storage.js")
    vetoed = ensure_ui_scaffolding(plan, brief.to_dict(), skills=[_Veto()])
    assert "frontend/src/lib/storage.js" not in set(ordered_paths(vetoed))


class _PlanProvider:
    """Returns a minimal (deliberately design-less) plan; records the system prompt."""

    def __init__(self):
        self.systems = []

    def chat(self, messages=None, **kw):
        self.systems.append(messages[0]["content"])
        return {"content": json.dumps({"goal": "chat app", "layers": [{"name": "ui", "files": [
            {"path": "frontend/index.html", "description": "entry"},
            {"path": "frontend/package.json", "description": "manifest"},
            {"path": "frontend/vite.config.js", "description": "vite"},
            {"path": "frontend/src/main.jsx", "description": "mount"},
            {"path": "frontend/src/App.jsx", "description": "shell"},
            {"path": "frontend/src/pages/Chat.jsx", "description": "chat"},
            {"path": "frontend/src/pages/Settings.jsx", "description": "settings"}]}],
            "contracts": {}})}


def test_tech_lead_injects_spec_into_prompt_and_persists_brief(tmp_path, monkeypatch):
    monkeypatch.setattr("cgx.session.tasks.swarm_apiground.ground_external_dependencies",
                        lambda *a, **k: {})
    provider = _PlanProvider()
    deps = ExecutorDeps(project_root=str(tmp_path), provider=provider)
    task = TaskNode.new(session_id="s", kind=TaskKind.SWARM_TECH_LEAD, name="plan",
                        inputs={"goal": GOAL, "project_root": str(tmp_path)})
    res = tl.swarm_tech_lead(task, deps)
    assert res.artifact is not None, res.outputs
    content = res.artifact.content
    assert provider.systems and "UI/UX PRODUCT SPEC" in provider.systems[0]
    assert "Chat / AI assistant app" in provider.systems[0]
    brief = content["design"]
    assert brief and brief["applies"] and brief["archetype"] == "chat-assistant"
    assert brief["app_root"] == "frontend" and brief["tokens_path"] == "frontend/src/styles/tokens.css"
    paths = set(content["paths"])
    assert {"frontend/src/styles/tokens.css", "frontend/src/styles/app.css",
            "frontend/src/lib/api.js", "frontend/src/lib/storage.js"} <= paths
    assert res.outputs["design"]["theme"] == "conversational"


def test_tech_lead_non_ui_goal_has_no_design(tmp_path, monkeypatch):
    monkeypatch.setattr("cgx.session.tasks.swarm_apiground.ground_external_dependencies",
                        lambda *a, **k: {})

    class _P:
        def chat(self, messages=None, **kw):
            assert "UI/UX PRODUCT SPEC" not in messages[0]["content"]
            return {"content": json.dumps({"goal": "cli", "layers": [{"name": "src", "files": [
                {"path": "src/rename.py", "description": "cli"},
                {"path": "requirements.txt", "description": "deps"},
                {"path": "README.md", "description": "readme"}]}], "contracts": {}})}
    deps = ExecutorDeps(project_root=str(tmp_path), provider=_P())
    task = TaskNode.new(session_id="s", kind=TaskKind.SWARM_TECH_LEAD, name="plan",
                        inputs={"goal": "a python cli tool that renames files", "project_root": str(tmp_path)})
    res = tl.swarm_tech_lead(task, deps)
    assert res.artifact is not None and res.artifact.content["design"] is None


# --------------------------------------------------------------------------
# developer / engine
# --------------------------------------------------------------------------

def test_generate_file_emits_tokens_deterministically_without_the_model(tmp_path):
    brief = design.build_brief(GOAL, ["react"]).to_dict()

    class _Boom:
        def chat(self, *a, **k):
            raise AssertionError("the model must not be called for the tokens file")
    out = sg.generate_file(path="src/styles/tokens.css", description="tokens", depends_on=[],
                           contracts={}, goal=GOAL, root=str(tmp_path), provider=_Boom(),
                           design=brief)
    assert out.ok and out.method == "template" and "--color-primary" in out.content


def test_engine_adds_design_block_and_budget_for_ui_files(monkeypatch):
    brief = design.build_brief(GOAL, ["react"]).to_dict()
    seen = {}

    class _P:
        def chat(self, messages=None, **kw):
            seen["system"] = messages[0]["content"]
            seen["max_tokens"] = kw.get("max_tokens")
            return {"content": json.dumps({"content": "import { useState } from 'react';\n"
                                           "export default function Chat() { return (<section className=\"chat\">"
                                           "<h1>Chat</h1></section>); }\n"})}
    monkeypatch.setattr(engine, "get_capability_tier", lambda p: "medium", raising=False)
    res = engine.generate_single_scaffold_file("src/pages/Chat.jsx", "chat screen", _P(), goal=GOAL,
                                               skills=["react"], design=brief)
    assert res["syntax_ok"] and "COMPONENT RULES" in seen["system"] and "DESIGN SYSTEM" in seen["system"]
    # tokens file short-circuits in the engine too
    tok = engine.generate_single_scaffold_file("src/styles/tokens.css", "tokens", _P(), goal=GOAL,
                                               skills=["react"], design=brief)
    assert tok["syntax_ok"] and ":root" in tok["content"]
    # a non-UI file gets no design block
    seen.clear()

    class _Py:
        def chat(self, messages=None, **kw):
            seen["system"] = messages[0]["content"]
            return {"content": json.dumps({"content": "def main():\n    return 1\n"})}
    engine.generate_single_scaffold_file("backend/app.py", "entry", _Py(), goal="a flask api",
                                         skills=["flask"], design=brief)
    assert "COMPONENT RULES" not in seen["system"] and "DESIGN SYSTEM" not in seen["system"]


# --------------------------------------------------------------------------
# verifier
# --------------------------------------------------------------------------

class _Art:
    def __init__(self, content):
        self.content = content
        self.artifact_id = "wp"


class _Store:
    def __init__(self, plan):
        self._plan = plan

    def get_artifact(self, _aid):
        return _Art(self._plan)


class _Provider:
    pass


UNSTYLED = ("import { useState } from 'react';\nexport default function Chat() {\n"
            "  const [m, setM] = useState([]);\n  return (<div><h2>Chat</h2><div>{m.map((x) => <div key={x}>{x}</div>)}</div>"
            "<input value='' /><button>Send</button></div>);\n}\n")
STYLED = ("import { useState } from 'react';\nexport default function Chat() {\n"
          "  const [m, setM] = useState([]);\n  return (<section className=\"chat\"><h2 className=\"page-title\">Chat</h2>"
          "{m.length === 0 ? <div className=\"empty-state\">Say hi</div> : m.map((x) => <div className=\"bubble\" key={x}>{x}</div>)}"
          "<form className=\"composer\"><label htmlFor=\"c\" className=\"visually-hidden\">Message</label>"
          "<textarea id=\"c\" className=\"textarea\" /><button className=\"btn btn-primary\">Send</button></form></section>);\n}\n")
ENTRY = ("import { StrictMode } from 'react';\nimport { createRoot } from 'react-dom/client';\n"
         "import { BrowserRouter } from 'react-router';\nimport './styles/tokens.css';\nimport './styles/app.css';\n"
         "import App from './App.jsx';\ncreateRoot(document.getElementById('root')).render(<StrictMode><BrowserRouter><App/></BrowserRouter></StrictMode>);\n")
APP = ("import { Routes, Route, NavLink } from 'react-router';\nimport Chat from './pages/Chat.jsx';\n"
       "export default function App() { return (<div className=\"app-shell\"><header className=\"topbar\"><nav className=\"nav\">"
       "<NavLink className=\"nav-link\" to=\"/\">Chat</NavLink></nav></header><main className=\"main\"><Routes>"
       "<Route path=\"/\" element={<Chat/>}/></Routes></main></div>); }\n")
STORAGE = "export function loadSettings() { try { return JSON.parse(localStorage.getItem('s')) || {}; } catch { return {}; } }\n"
APP_CSS = ".chat { display: grid; } @media (max-width: 768px) { .chat { display: block; } }\n"


def _ui_tree(root, chat_body):
    files = {
        "index.html": ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
                       'content="width=device-width, initial-scale=1"><title>Chatbot</title></head><body>'
                       '<div id="root"></div><script type="module" src="/src/main.jsx"></script></body></html>'),
        "src/main.jsx": ENTRY, "src/App.jsx": APP, "src/pages/Chat.jsx": chat_body,
        "src/lib/storage.js": STORAGE, "src/styles/app.css": APP_CSS,
    }
    for rel, body in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    return list(files)


def _run_verify(tmp_path, paths, brief, monkeypatch, regen_body):
    monkeypatch.setattr(sv, "_run_env_dryrun", lambda paths, root: {"ran": False, "outcome": "skipped"})
    monkeypatch.setenv("CGX_DESIGN_CRITIC", "0")
    regen_calls = []

    def fake_generate_file(**kw):
        regen_calls.append(kw)
        return sg.GenerationOutcome(kw["path"], regen_body, True, "full-file")
    monkeypatch.setattr(sv, "generate_file", fake_generate_file)
    plan = {"goal": GOAL, "project_root": str(tmp_path), "paths": paths,
            "layers": [{"name": "ui", "files": [{"path": p, "description": p} for p in paths]}],
            "contracts": {}, "skills": ["react"], "design": brief}
    deps = ExecutorDeps(project_root=str(tmp_path), provider=_Provider(), store=_Store(plan))
    task = TaskNode.new(session_id="s", kind=TaskKind.SWARM_VERIFY, name="verify",
                        inputs={"work_plan_artifact_id": "wp", "failed_paths": []})
    return sv.swarm_verify(task, deps), regen_calls


def test_verify_writes_missing_tokens_and_regenerates_unstyled_component_with_directive(tmp_path, monkeypatch):
    paths = _ui_tree(tmp_path, UNSTYLED)
    brief = design.build_brief(GOAL, ["react"]).resolved_for_paths(paths).to_dict()
    res, regen = _run_verify(tmp_path, paths, brief, monkeypatch, regen_body=STYLED)
    # the tokens stylesheet the plan dropped was written deterministically
    assert (tmp_path / "src/styles/tokens.css").is_file()
    assert res.artifact.content["design"]["tokens_written"] == "src/styles/tokens.css"
    # the unstyled screen was regenerated with the design directive threaded in
    assert any(c["path"] == "src/pages/Chat.jsx" and "REQUIRED FIX" in c["description"]
               and "className" in c["description"] and c.get("design") for c in regen)
    assert res.outputs["verify_ok"] is True
    assert res.artifact.content["design"]["fatal_remaining"] == 0
    assert "design ok" in res.outputs["summary"]


def test_verify_gate_fails_when_design_defect_survives(tmp_path, monkeypatch):
    paths = _ui_tree(tmp_path, UNSTYLED)
    brief = design.build_brief(GOAL, ["react"]).resolved_for_paths(paths).to_dict()
    # the regen keeps producing the unstyled body -> fatal issue survives the rounds
    res, regen = _run_verify(tmp_path, paths, brief, monkeypatch, regen_body=UNSTYLED)
    assert regen, "verify must have attempted a regeneration"
    assert res.outputs["verify_ok"] is False
    assert "D03-unstyled-component" in res.outputs["summary"]
    # ... but only advisory when the gate is switched off
    monkeypatch.setenv("CGX_DESIGN_GATE", "0")
    res2, _ = _run_verify(tmp_path, paths, brief, monkeypatch, regen_body=UNSTYLED)
    assert res2.outputs["verify_ok"] is True
    assert res2.artifact.content["design"]["fatal_remaining"] >= 1


def test_verify_non_ui_plan_unchanged(tmp_path, monkeypatch):
    (tmp_path / "models.py").write_text("class User:\n    pass\n", encoding="utf-8")
    monkeypatch.setattr(sv, "_run_env_dryrun", lambda paths, root: {"ran": False, "outcome": "skipped"})
    plan = {"goal": "g", "project_root": str(tmp_path), "paths": ["models.py"], "layers": [],
            "contracts": {}}
    deps = ExecutorDeps(project_root=str(tmp_path), provider=_Provider(), store=_Store(plan))
    task = TaskNode.new(session_id="s", kind=TaskKind.SWARM_VERIFY, name="verify",
                        inputs={"work_plan_artifact_id": "wp", "failed_paths": []})
    res = sv.swarm_verify(task, deps)
    assert res.outputs["verify_ok"] is True
    assert res.artifact.content["design"] == {"applies": False}
    assert "design" not in res.outputs["summary"]
