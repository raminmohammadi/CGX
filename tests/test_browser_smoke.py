"""Runtime smoke for built frontends: browser discovery, SPA serving, DOM/console
parsing, and the Verifier's blank-route repair round -- all with a fake browser."""

from __future__ import annotations

import os
import stat

import pytest

from cgx.codegen import browser_smoke as bs

FAKE_BROWSER = """#!/bin/sh
# Fake Chromium: prints a DOM for --dump-dom, writes a file for --screenshot=, logs console lines.
url=""
for a in "$@"; do case "$a" in http*) url="$a";; esac; done
for a in "$@"; do case "$a" in --screenshot=*) echo png > "${a#--screenshot=}";; esac; done
case "$url" in
  */settings)
    echo '<html><body><div id="root"></div></body></html>'
    echo '[1:2:0101/000000.000:INFO:CONSOLE(1)] "Uncaught TypeError: Cannot read properties of undefined (reading '"'"'apiAddress'"'"')", source: http://x/assets/index.js (1)' >&2 ;;
  *)
    echo '<html><body><div id="root"><header class="topbar">Chatbot</header><main>No messages yet</main></div></body></html>' ;;
esac
"""


@pytest.fixture()
def fake_browser(tmp_path, monkeypatch):
    b = tmp_path / "chrome"
    b.write_text(FAKE_BROWSER, encoding="utf-8")
    b.chmod(b.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("CGX_BROWSER", str(b))
    monkeypatch.delenv("CGX_BROWSER_SMOKE", raising=False)
    return b


def _dist(root):
    d = root / "dist"
    d.mkdir(parents=True, exist_ok=True)
    (d / "index.html").write_text("<html><body><div id='root'></div></body></html>", encoding="utf-8")
    return d


def test_parsers():
    ok, text = bs._mounted('<html><body><div id="root"><h1>Hi there friend</h1></div></body></html>')
    assert ok and "Hi there" in text
    blank, _ = bs._mounted('<html><body><div id="root"></div><script>x</script></body></html>')
    assert not blank
    errs = bs._parse_console_errors('[1:2:0101/0:INFO:CONSOLE(3)] "Uncaught ReferenceError: x is not defined", source: a (3)\n'
                                    '[1:2:0101/0:INFO:CONSOLE(4)] "Download the React DevTools", source: a (4)')
    assert errs == ["Uncaught ReferenceError: x is not defined"]
    routes = bs.routes_from_sources({"src/App.jsx": '<Route path="/chat" element={<Chat/>}/><Route path="/settings" element={<S/>}/><Route path="*" element={<C/>}/>'})
    assert routes == ["/", "/chat", "/settings"]


def test_skips_without_browser_or_bundle(tmp_path, monkeypatch):
    monkeypatch.setenv("CGX_BROWSER", str(tmp_path / "nope"))
    assert not bs.run_browser_smoke(str(tmp_path), ["/"]).ran
    monkeypatch.setenv("CGX_BROWSER_SMOKE", "0")
    assert "disabled" in bs.run_browser_smoke(str(tmp_path), ["/"]).skipped_reason


def test_fake_browser_run_reports_blank_route_and_screenshots(tmp_path, fake_browser):
    _dist(tmp_path)
    res = bs.run_browser_smoke(str(tmp_path), ["/", "/settings"])
    assert res.ran and not res.ok
    by = {r.route: r for r in res.results}
    assert by["/"].ok and by["/"].screenshot and os.path.isfile(by["/"].screenshot)
    assert not by["/settings"].mounted and "apiAddress" in by["/settings"].errors[0]
    assert res.as_dict()["routes"][1]["route"] == "/settings"


def test_verifier_runs_a_repair_round_for_a_blank_route(tmp_path, fake_browser, monkeypatch):
    from cgx import design
    from cgx.session.tasks import swarm_verify as sv
    from cgx.session.tasks import swarm_generate as sg
    goal = "a chatbot app with an API key settings tab, React"
    files = {
        "src/main.jsx": "import { createRoot } from 'react-dom/client';\nimport './styles/tokens.css';\nimport App from './App.jsx';\ncreateRoot(document.getElementById('root')).render(<App/>);\n",
        "src/App.jsx": "import { Routes, Route, NavLink } from 'react-router';\nimport Chat from './pages/Chat.jsx';\nimport Settings from './pages/Settings.jsx';\nexport default function App() { return (<div className=\"app-shell\"><nav className=\"nav\"><NavLink className=\"nav-link\" to=\"/\">Chat</NavLink></nav><Routes><Route path=\"/\" element={<Chat/>}/><Route path=\"/settings\" element={<Settings/>}/></Routes></div>); }\n",
        "src/pages/Chat.jsx": "export default function Chat() { return (<section className=\"chat\"><div className=\"empty-state\">hi</div></section>); }\n",
        "src/pages/Settings.jsx": "import { saveSettings } from '../lib/storage.js';\nexport default function Settings() { const s = saveSettings(); return (<form className=\"card\"><label htmlFor=\"u\" className=\"label\">URL</label><input id=\"u\" className=\"input\" value={s.apiAddress}/></form>); }\n",
        "src/lib/storage.js": "export function saveSettings(s) { localStorage.setItem('s', JSON.stringify(s)); }\nexport function loadSettings() { return JSON.parse(localStorage.getItem('s')) || {}; }\n",
        "src/styles/tokens.css": ":root{--color-primary:#000}", "src/styles/app.css": ".chat{} @media (max-width:768px){.chat{}}",
        "index.html": '<!doctype html><html><head><meta name="viewport" content="width=device-width"></head><body><div id="root"></div></body></html>',
    }
    for rel, body in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    _dist(tmp_path)
    paths = list(files)
    brief = design.build_brief(goal, ["react"]).resolved_for_paths(paths).to_dict()
    monkeypatch.setattr(sv, "_run_env_dryrun", lambda paths, root: {"ran": True, "outcome": "passed", "output": ""})
    monkeypatch.setenv("CGX_DESIGN_CRITIC", "0")
    repaired = {}

    def fake_dynamic_repair(env, targets, contents, goal, root, provider, paths, specs, *a, **k):
        repaired["targets"] = list(targets)
        repaired["output"] = env.get("output")
        # the "repair" fixes the file; the fake browser still reports /settings blank
        # (it keys on the URL), so the verdict must stay red and name the route.
        return True
    monkeypatch.setattr(sv, "_dynamic_repair", fake_dynamic_repair)

    class _Store:
        def get_artifact(self, _):
            class _A:
                content = {"goal": goal, "project_root": str(tmp_path), "paths": paths,
                           "layers": [{"name": "ui", "files": [{"path": p, "description": p} for p in paths]}],
                           "contracts": {}, "skills": ["react"], "design": brief}
                artifact_id = "wp"
            return _A()

    class _Provider:
        pass
    from cgx.session.models import TaskKind, TaskNode
    from cgx.session.tasks.base import ExecutorDeps
    deps = ExecutorDeps(project_root=str(tmp_path), provider=_Provider(), store=_Store())
    task = TaskNode.new(session_id="s", kind=TaskKind.SWARM_VERIFY, name="verify",
                        inputs={"work_plan_artifact_id": "wp", "failed_paths": []})
    res = sv.swarm_verify(task, deps)
    assert repaired["targets"][:1] == ["src/pages/Settings.jsx"] and "src/lib/storage.js" in repaired["targets"]
    assert "RUNTIME SMOKE FAILURE" in repaired["output"] and "apiAddress" in repaired["output"]
    assert res.outputs["verify_ok"] is False
    assert "runtime: /settings blank/crashed" in res.outputs["summary"]
    assert res.artifact.content["runtime"]["routes"][1]["route"] == "/settings"
    assert (tmp_path / ".cgx" / "screenshots" / "settings.png").is_file()


def test_dom_error_collector_is_injected_and_parsed(tmp_path):
    # the SPA handler injects the collector into served HTML
    d = _dist(tmp_path)
    server, port = bs._serve(d)
    try:
        import urllib.request
        html = urllib.request.urlopen(f"http://127.0.0.1:{port}/settings", timeout=5).read().decode()
    finally:
        server.shutdown(); server.server_close()
    assert "__cgx_errors" in html and "<div id='root'>" in html  # SPA fallback served index.html
    # the collector's DOM payload is read back as errors and excluded from the mounted text
    dom = ('<html><body><div id="root"></div><div id="__cgx_errors" hidden>'
           '["Uncaught Error: useRoutes() may be used only in the context of a &lt;Router&gt; component."]</div></body></html>')
    assert bs._parse_console_errors("", dom) == ["Uncaught Error: useRoutes() may be used only in the context of a <Router> component."]
    mounted, _ = bs._mounted(dom)
    assert not mounted
