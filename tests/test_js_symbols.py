"""JS/TS symbol intelligence: module surface, grounding views, import hints,
the Verifier's symbol gate, the React skill's deterministic repairs and the
tool-call leak hardening -- all against the defects a real 14B run shipped."""

from __future__ import annotations

import json

import pytest

import cgx.answer.engine as engine
from cgx.session import js_symbols as js
from cgx.session.tasks import swarm_generate as sg
from cgx.session.tasks.swarm_skeleton import render_import_hints
from skills.react import ReactSkill

STORAGE = """const KEY = 'chatbot-';
export const loadSettings = () => { try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch (e) { return {}; } };
export const saveSettings = (settings) => { localStorage.setItem(KEY, JSON.stringify(settings)); };
export function clearAll() {}
"""
API = """import { loadSettings } from './storage';
export async function sendMessage(message) { return message; }
export const testConnection = async () => true;
"""
FORM = """import React, { useState } from 'react';
const SettingsForm = ({ onSave }) => { const [v, setV] = useState(''); return (<form className="card" onSubmit={onSave}><input className="input" value={v} onChange={(e) => setV(e.target.value)} /></form>); };
export default SettingsForm;
"""
SETTINGS_BAD = """import React, { useState, useEffect } from 'react';
import { storage } from '../lib/storage';
import { api } from '../lib/api';
import { SettingsForm } from '../components/SettingsForm';
import './Settings.css';
const Settings = () => { useEffect(() => { storage.loadSettings(); }, []); return (<div className="page"><SettingsForm onSave={() => api.sendMessage('x')} /></div>); };
export default Settings;
"""
MAIN_BAD = """import React from 'react';
import { createRoot } from 'react-dom/client';
import './styles/tokens.css';
import { BrowserRouter } from 'react-router-dom';
import App from './App';
createRoot(document.getElementById('root')).render(<StrictMode><BrowserRouter><App /></BrowserRouter></StrictMode>);
"""
APP_BAD = """import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Settings from './pages/Settings';
const App = () => (<Router><Routes><Route path="/" element={<Settings />} /></Routes></Router>);
export default App;
"""
SHELL_BAD = """import React from 'react';
import { Link } from 'react-router';
const AppShell = ({ children }) => { const s = storage.loadSettings(); return (<div className="app-shell"><nav className="nav"><Link to="/">Chat</Link></nav>{children}</div>); };
export default AppShell;
"""
TREE = {"src/lib/storage.js": STORAGE, "src/lib/api.js": API, "src/components/SettingsForm.jsx": FORM,
        "src/pages/Settings.jsx": SETTINGS_BAD, "src/main.jsx": MAIN_BAD, "src/App.jsx": APP_BAD,
        "src/components/AppShell.jsx": SHELL_BAD, "src/styles/tokens.css": ":root{}"}


def test_exports_and_imports_are_parsed():
    ex = js.js_exports(STORAGE)
    assert ex["named"] == ["loadSettings", "saveSettings", "clearAll"] and not ex["has_default"]
    ex = js.js_exports(FORM)
    assert ex["default"] == "SettingsForm" and ex["has_default"] and ex["named"] == []
    assert js.js_exports("export default function App() {}")["default"] == "App"
    assert js.js_exports("const a = 1; const b = 2; export { a, b as bee };")["named"] == ["a", "bee"]
    assert js.js_exports("module.exports = { x, y };")["named"] == ["x", "y"]
    stmts = js.js_imports(SETTINGS_BAD)
    assert [s.spec for s in stmts] == ["react", "../lib/storage", "../lib/api", "../components/SettingsForm", "./Settings.css"]
    assert stmts[0].default == "React" and [n for n, _ in stmts[0].named] == ["useState", "useEffect"]
    assert stmts[-1].side_effect
    assert js.js_imports("import * as ns from './x';")[0].namespace == "ns"


def test_grounding_views_show_the_real_surface(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "storage.js").write_text(STORAGE, encoding="utf-8")
    from cgx.session.tasks.swarm_ground import describe_file, file_skeleton, list_symbols
    syms = list_symbols("src/storage.js", str(tmp_path))
    assert {s["name"] for s in syms} == {"loadSettings", "saveSettings", "clearAll"}
    assert "export const loadSettings = () =>" in file_skeleton("src/storage.js", str(tmp_path))
    assert describe_file("src/storage.js", str(tmp_path))["symbols"]


def test_import_hints_for_js_dependencies():
    hint = render_import_hints("src/pages/Settings.jsx", {
        "src/lib/storage.js": STORAGE, "src/components/SettingsForm.jsx": FORM, "src/styles/app.css": "x"})
    assert "import { loadSettings, saveSettings, clearAll } from '../lib/storage.js';" in hint
    assert "import SettingsForm from '../components/SettingsForm.jsx';" in hint
    assert render_import_hints("backend/app.py", {"src/lib/storage.js": STORAGE}) == ""


def test_symbol_gate_catches_the_live_run_defects():
    rows = js.js_symbol_breaks(TREE, list(TREE), entry_path="src/main.jsx", app_path="src/App.jsx")
    by_file = {}
    for r in rows:
        by_file.setdefault(r["file"], []).append(r)
    reasons = " ".join(r["reason"] for r in by_file["src/pages/Settings.jsx"])
    assert "{ storage }" in reasons and "{ api }" in reasons and "{ SettingsForm }" in reasons
    directives = " ".join(r["directive"] for r in by_file["src/pages/Settings.jsx"])
    assert "import { loadSettings, saveSettings, clearAll } from '../lib/storage';" in directives
    assert "import SettingsForm from '../components/SettingsForm';" in directives
    assert any("StrictMode" in r["reason"] for r in by_file["src/main.jsx"])
    assert any("nested routers" in r["reason"] for r in by_file["src/App.jsx"])
    assert any("storage" in r["reason"] and "used like an imported module" in r["directive"]
               for r in by_file["src/components/AppShell.jsx"])
    # every row targets the importer with an additive directive
    assert all(r["define_in"] == r["file"] and r["directive"] for r in rows)


def test_symbol_gate_abstains_on_clean_or_unknown_surfaces():
    good = {"src/lib/storage.js": STORAGE,
            "src/pages/Settings.jsx": ("import { useState } from 'react';\nimport { loadSettings } from '../lib/storage';\n"
                                       "export default function Settings() { const [v] = useState(loadSettings()); return (<div className=\"page\">{v}</div>); }\n")}
    assert js.js_symbol_breaks(good, list(good)) == []
    # a dependency whose exports cannot be read -> no opinion on its importers
    odd = {"src/lib/weird.js": "/* generated */ Object.assign(exports, build());",
           "src/a.jsx": "import { thing } from './lib/weird';\nexport default function A() { return (<p className=\"x\">{thing}</p>); }\n"}
    assert js.js_symbol_breaks(odd, list(odd)) == []


def test_react_skill_repairs_router_import_dangling_css_and_missing_react_imports():
    files = [{"path": "package.json", "content": json.dumps({"dependencies": {"react": "^19", "react-router": "^7"}})},
             {"path": "src/main.jsx", "content": MAIN_BAD},
             {"path": "src/pages/Settings.jsx", "content": SETTINGS_BAD},
             {"path": "src/styles/tokens.css", "content": ":root{}"}]
    fixes = ReactSkill().repair_scaffold(files)
    main = fixes["src/main.jsx"]
    assert "from 'react-router'" in main and "react-router-dom" not in main
    assert "import React, { StrictMode } from 'react';" in main
    settings = fixes["src/pages/Settings.jsx"]
    assert "./Settings.css" not in settings
    # a stylesheet that exists is kept; nothing to fix -> file absent from fixes
    keep = [{"path": "src/a.jsx", "content": "import './a.css';\nexport default function A() { return (<p className=\"x\"/>); }\n"},
            {"path": "src/a.css", "content": ".x{}"}]
    assert ReactSkill().repair_scaffold(keep) == {}
    # no react import line at all -> one is prepended
    bare = [{"path": "src/b.jsx", "content": "export default function B() { return (<Fragment><p className=\"y\"/></Fragment>); }\n"}]
    assert ReactSkill().repair_scaffold(bare)["src/b.jsx"].startswith("import { Fragment } from 'react';")


class _ToolHappyProvider:
    """Always answers with a tool call until told the budget is exhausted."""

    def __init__(self):
        self.calls = 0
        self.saw_exhausted = False

    def chat(self, messages=None, **kw):
        self.calls += 1
        if any("TOOL BUDGET EXHAUSTED" in str(m.get("content")) for m in messages if m.get("role") == "user"):
            self.saw_exhausted = True
            return {"content": json.dumps({"content": "export default function TopBar() { return (<header className=\"topbar\"/>); }"})}
        return {"content": '<call_tool name="file_skeleton">{"path": "src/components/TopBar.jsx"}</call_tool>'}


def test_tool_wrapper_exhaustion_turn_and_self_file_lookup(tmp_path):
    prov = _ToolHappyProvider()
    w = sg.ToolWrapper(prov, str(tmp_path), tools=("file_skeleton",), target_path="src/components/TopBar.jsx")
    res = w.chat(messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "write it"}])
    assert prov.saw_exhausted and "<call_tool" not in res["content"]
    assert prov.calls == sg._MAX_TOOL_ITERS + 1


def test_engine_rejects_leaked_tool_calls(monkeypatch):
    # JSON mode: the model wraps the stray tool call INSIDE the content field.
    replies = iter([
        {"content": json.dumps({"content": '<call_tool name="file_skeleton">{"path": "src/components/TopBar.jsx"}</call_tool>'})},
        {"content": json.dumps({"content": "export default function TopBar() { return (<header className=\"topbar\"/>); }\n"})},
    ])

    class _P:
        def chat(self, messages=None, **kw):
            return next(replies)
    res = engine.generate_single_scaffold_file("src/components/TopBar.jsx", "top bar", _P(), goal="chat app",
                                               skills=["react"])
    assert res["syntax_ok"] and "<call_tool" not in res["content"] and "topbar" in res["content"]


def test_regate_rejects_tool_calls_and_broken_jsx():
    assert "tool call" in (sg._regate_repaired("src/x.jsx", '<call_tool name="a">{}</call_tool>', {}, None, ".") or "")
    assert sg._regate_repaired("src/x.jsx", "export default function X() { return (<p className=\"a\"/>); }\n", {}, None, ".") is None


def test_duplicate_route_tables_flag_the_non_app_owner():
    tree = {
        "src/main.jsx": "import { createRoot } from 'react-dom/client';\nimport { BrowserRouter } from 'react-router';\nimport App from './App.jsx';\ncreateRoot(document.getElementById('root')).render(<BrowserRouter><App /></BrowserRouter>);\n",
        "src/App.jsx": "import { Routes, Route } from 'react-router';\nimport Shell from './components/AppShell.jsx';\nexport default function App() { return (<Shell><Routes><Route path=\"/\" element={<p className=\"a\"/>} /></Routes></Shell>); }\n",
        "src/components/AppShell.jsx": "import { Routes, Route, Link } from 'react-router';\nexport default function Shell({ children }) { return (<div className=\"app-shell\"><Routes><Route path=\"/\" element={<p className=\"b\"/>} /></Routes>{children}</div>); }\n",
    }
    rows = js.js_symbol_breaks(tree, list(tree), entry_path="src/main.jsx", app_path="src/App.jsx")
    assert [r["file"] for r in rows] == ["src/components/AppShell.jsx"]
    assert "owns the single route table" in rows[0]["directive"]


def test_js_broken_files_treats_tool_leaks_and_syntax_errors_as_gaps():
    from cgx.session.tasks.swarm_verify import _js_broken_files
    contents = {
        "src/components/TopBar.jsx": '<call_tool name="file_skeleton">{"path": "x"}</call_tool>',
        "src/components/Empty.jsx": "",
        "src/ok.jsx": "export default function Ok() { return (<p className=\"ok\"/>); }\n",
        "backend/app.py": "<call_tool name='x'></call_tool>",  # python is the Python gate's job
    }
    broken = _js_broken_files(contents)
    assert "src/components/TopBar.jsx" in broken and "src/components/Empty.jsx" in broken
    assert "src/ok.jsx" not in broken and "backend/app.py" not in broken


def test_prop_contract_breaks_both_directions():
    tree = {
        "src/components/EmptyState.jsx": "export default function EmptyState() { return (<div className=\"empty-state\"><h2>No messages yet</h2></div>); }\n",
        "src/components/ConnectionStatus.jsx": "const ConnectionStatus = ({ isConnected }) => (<div className=\"badge\">{isConnected ? 'on' : 'off'}</div>);\nexport default ConnectionStatus;\n",
        "src/pages/Chat.jsx": "import EmptyState from '../components/EmptyState.jsx';\nimport ConnectionStatus from '../components/ConnectionStatus.jsx';\nexport default function Chat() { return (<section className=\"chat\"><ConnectionStatus /><EmptyState headline=\"Configure your API\" suggestions={['a']} /></section>); }\n",
    }
    rows = js.js_symbol_breaks(tree, list(tree))
    by_file = {r["file"]: r for r in rows}
    assert "ignores the props" in by_file["src/components/EmptyState.jsx"]["reason"]
    assert "headline, suggestions" in by_file["src/components/EmptyState.jsx"]["directive"]
    assert "without any props" in by_file["src/pages/Chat.jsx"]["reason"] and "isConnected" in by_file["src/pages/Chat.jsx"]["directive"]
    # a component taking a props object / rest accepts anything; children never counts as required
    fine = {
        "src/components/Shell.jsx": "export default function Shell({ children }) { return (<div className=\"app-shell\">{children}</div>); }\n",
        "src/components/Card.jsx": "export default function Card(props) { return (<div className=\"card\">{props.title}</div>); }\n",
        "src/App.jsx": "import Shell from './components/Shell.jsx';\nimport Card from './components/Card.jsx';\nexport default function App() { return (<Shell><Card title=\"x\"/></Shell>); }\n",
    }
    assert js.js_symbol_breaks(fine, list(fine)) == []


def test_tool_leak_triggers_a_tool_free_full_file_retry(tmp_path, monkeypatch):
    """When the full-file rung keeps leaking <call_tool>, the ladder retries once
    with NO tools advertised (so the reply can only be the file) before any
    semantic repair could turn the tag into a placeholder component."""
    calls = []

    def fake(path, description, provider, **kw):
        calls.append(tuple(getattr(provider, "tools", ())))
        if provider.tools:
            return {"file": path, "content": '<call_tool name="x">{}</call_tool>', "syntax_ok": False,
                    "syntax_error": "content contains a <call_tool>/<tool_response> tag -- a tool call is not file content"}
        return {"file": path, "content": "export default function TopBar() { return (<header className=\"topbar\">Chatbot</header>); }\n",
                "syntax_ok": True}
    monkeypatch.setattr(engine, "generate_single_scaffold_file", fake)
    out = sg.generate_file(path="src/components/TopBar.jsx", description="top bar", depends_on=[], contracts={},
                           goal="chat app", root=str(tmp_path), provider=object())
    assert out.ok and out.method == "full-file" and "topbar" in out.content
    assert calls[-1] == () and len(calls) == 3  # two tool-enabled attempts, then the tool-free one


def test_prop_passed_but_not_accepted_is_flagged_on_the_component():
    tree = {
        "src/components/Composer.jsx": "export default function Composer({ onSend }) { return (<form className=\"composer\" onSubmit={onSend}><button className=\"btn\">Send</button></form>); }\n",
        "src/pages/Chat.jsx": "import Composer from '../components/Composer.jsx';\nexport default function Chat() { return (<section className=\"chat\"><Composer onSend={() => 1} disabled={true} /></section>); }\n",
    }
    rows = js.js_symbol_breaks(tree, list(tree))
    assert [r["file"] for r in rows] == ["src/components/Composer.jsx"]
    assert "disabled" in rows[0]["reason"] and "disabled" in rows[0]["directive"]
    # a spread or a plain props object accepts anything
    tree["src/components/Composer.jsx"] = "export default function Composer({ onSend, ...rest }) { return (<form className=\"composer\" {...rest} onSubmit={onSend}/>); }\n"
    assert js.js_symbol_breaks(tree, list(tree)) == []


def test_missing_router_provider_is_flagged_on_the_entry():
    tree = {
        "src/main.jsx": "import { createRoot } from 'react-dom/client';\nimport App from './App.jsx';\ncreateRoot(document.getElementById('root')).render(<App />);\n",
        "src/App.jsx": "import { Routes, Route } from 'react-router';\nexport default function App() { return (<main className=\"main\"><Routes><Route path=\"/\" element={<p className=\"a\"/>} /></Routes></main>); }\n",
    }
    rows = js.js_symbol_breaks(tree, list(tree), entry_path="src/main.jsx", app_path="src/App.jsx")
    assert [r["file"] for r in rows] == ["src/main.jsx"] and "BrowserRouter" in rows[0]["directive"]
    tree["src/main.jsx"] = ("import { createRoot } from 'react-dom/client';\nimport { BrowserRouter } from 'react-router';\nimport App from './App.jsx';\n"
                            "createRoot(document.getElementById('root')).render(<BrowserRouter><App /></BrowserRouter>);\n")
    assert js.js_symbol_breaks(tree, list(tree), entry_path="src/main.jsx", app_path="src/App.jsx") == []


def test_react_repair_rewrites_undeclared_namespace_calls_into_named_imports():
    files = [
        {"path": "src/lib/storage.js", "content": STORAGE},
        {"path": "src/lib/api.js", "content": API},
        {"path": "src/App.jsx", "content": ("import React from 'react';\nimport { Link } from 'react-router';\n"
                                            "export default function App() { const s = storage.loadSettings(); storage.saveSettings(s); "
                                            "api.sendMessage('x'); return (<div className=\"app-shell\">{s.apiKey}</div>); }\n")},
        # a namespace whose module is NOT in the bundle must be left alone (the gate reports it)
        {"path": "src/pages/Chat.jsx", "content": "export default function Chat() { ghost.run(); return (<p className=\"c\"/>); }\n"},
    ]
    fixes = ReactSkill().repair_scaffold(files)
    app = fixes["src/App.jsx"]
    assert "import { loadSettings, saveSettings } from './lib/storage.js';" in app
    assert "import { sendMessage } from './lib/api.js';" in app
    assert "loadSettings()" in app and "saveSettings(s)" in app and "sendMessage('x')" in app
    assert "storage.loadSettings" not in app and "storage.saveSettings" not in app and "api.sendMessage" not in app
    assert "src/pages/Chat.jsx" not in fixes
    # merging into an existing partial import of the same module
    files2 = [{"path": "src/lib/storage.js", "content": STORAGE},
              {"path": "src/Settings.jsx", "content": ("import { saveSettings } from './lib/storage.js';\n"
                                                        "export default function S() { const v = storage.loadSettings(); saveSettings(v); return (<p className=\"s\"/>); }\n")}]
    out = ReactSkill().repair_scaffold(files2)["src/Settings.jsx"]
    assert out.count("from './lib/storage.js'") == 1 and "import { saveSettings, loadSettings } from './lib/storage.js';" in out


def test_prop_drift_rows_are_non_gating_but_import_breaks_gate():
    from cgx.session.tasks.swarm_verify import _gating
    tree = {
        "src/components/Composer.jsx": "export default function Composer({ onSend }) { return (<form className=\"composer\" onSubmit={onSend}/>); }\n",
        "src/pages/Chat.jsx": "import Composer from '../components/Composer.jsx';\nimport { ghostFn } from '../components/Composer.jsx';\nexport default function Chat() { return (<section className=\"chat\"><Composer onSend={() => 1} disabled /></section>); }\n",
    }
    rows = js.js_symbol_breaks(tree, list(tree))
    kinds = {(r["kind"], r.get("gating", True)) for r in rows}
    assert ("js_props", False) in kinds and ("js_symbol", True) in kinds
    assert all(r["kind"] == "js_symbol" for r in _gating(rows))
