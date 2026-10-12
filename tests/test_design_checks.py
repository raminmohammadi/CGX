"""Deterministic UI-quality checks over generated file text (cgx.design.checks)."""

from __future__ import annotations

from cgx import design

GOAL = ("create an app for a chatbot in which user provides a API address and a key on the "
        "profile tab and then navigates to the main page and can interact with the chat. "
        "Use React frontend")


def _brief():
    return design.build_brief(GOAL, ["react"]).resolved_for_paths(
        ["frontend/index.html", "frontend/src/main.jsx", "frontend/src/App.jsx"])


def _ids(issues):
    return {i.id for i in issues}


# The exact demo output: unstyled, console.log-only profile, hard-coded /api/chat,
# deprecated ReactDOM.render, no stylesheet, toggle-button navigation.
DEMO_FILES = {
    "frontend/index.html": ('<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
                            '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
                            '<title>Chatbot App</title></head><body><div id="root"></div>'
                            '<script type="module" src="/src/main.jsx"></script></body></html>'),
    "frontend/src/main.jsx": ("import React from 'react';\nimport ReactDOM from 'react-dom';\n"
                              "import App from './App';\n"
                              "ReactDOM.render(<App />, document.getElementById('root'));\n"),
    "frontend/src/App.jsx": ("import React, { useState } from 'react';\nimport Profile from './Profile';\n"
                             "import Chat from './Chat';\nconst App = () => {\n  const [showProfile, setShowProfile] = useState(true);\n"
                             "  return (\n    <div>\n      <button onClick={() => setShowProfile(!showProfile)}>"
                             "{showProfile ? 'Go to Chat' : 'Go to Profile'}</button>\n"
                             "      {showProfile ? <Profile /> : <Chat />}\n    </div>\n  );\n};\nexport default App;\n"),
    "frontend/src/Profile.jsx": ("import React, { useState } from 'react';\nconst Profile = () => {\n"
                                 "  const [apiKey, setApiKey] = useState('');\n"
                                 "  const handleSubmit = (event) => {\n    event.preventDefault();\n"
                                 "    console.log('API Key:', apiKey);\n  };\n  return (\n    <div>\n      <h2>Profile</h2>\n"
                                 "      <form onSubmit={handleSubmit}>\n        <div><label>API Key:<input type=\"text\" "
                                 "value={apiKey} onChange={(e) => setApiKey(e.target.value)} /></label></div>\n"
                                 "        <button type=\"submit\">Save</button>\n      </form>\n    </div>\n  );\n};\n"
                                 "export default Profile;\n"),
    "frontend/src/Chat.jsx": ("import React, { useState } from 'react';\nconst Chat = () => {\n"
                              "  const [messages, setMessages] = useState([]);\n  const [input, setInput] = useState('');\n"
                              "  const send = async () => {\n    const response = await fetch('/api/chat', { method: 'POST', "
                              "body: JSON.stringify({ message: input }) });\n    const data = await response.json();\n"
                              "    setMessages((m) => [...m, { text: data.response }]);\n  };\n  return (\n"
                              "    <div className=\"chat-container\">\n      <div className=\"message-list\">"
                              "{messages.map((m, i) => <div key={i}>{m.text}</div>)}</div>\n"
                              "      <input type=\"text\" value={input} onChange={(e) => setInput(e.target.value)} />\n"
                              "      <button onClick={send}>Send</button>\n    </div>\n  );\n};\nexport default Chat;\n"),
}


def test_demo_output_is_rejected_for_the_right_reasons():
    issues = design.check_tree(_brief(), DEMO_FILES)
    fatal = _ids(design.fatal_issues(issues))
    assert "D01-no-stylesheet" in fatal
    assert "D03-unstyled-component" in fatal          # App.jsx / Profile.jsx have no className
    assert "D04-no-navigation" in fatal               # toggle button, no router / nav
    assert "D06-legacy-render" in fatal               # ReactDOM.render on React 19
    assert "D07-hardcoded-api" in fatal               # fetch('/api/chat')
    assert "D09-log-only-handler" in fatal            # Profile submit only console.logs
    # D08 persistence: the demo never touches localStorage
    assert "D08-no-persistence" in fatal
    # every fatal issue names a planned file + an additive directive
    for i in design.fatal_issues(issues):
        assert i.regen_file or i.file
        assert len(i.directive) > 20


def test_directives_target_the_right_files():
    issues = design.fatal_issues(design.check_tree(_brief(), DEMO_FILES))
    by_id = {i.id: i for i in issues}
    assert by_id["D07-hardcoded-api"].file == "frontend/src/Chat.jsx"
    assert by_id["D09-log-only-handler"].file == "frontend/src/Profile.jsx"
    assert by_id["D06-legacy-render"].file == "frontend/src/main.jsx"
    assert by_id["D01-no-stylesheet"].regen_file == "frontend/src/styles/tokens.css"
    rows = design.issues_as_warnings(issues)
    assert all(r["kind"] == "design" and r["define_in"] and r["directive"] for r in rows)


GOOD_FILES = {
    "frontend/index.html": DEMO_FILES["frontend/index.html"],
    "frontend/src/styles/tokens.css": ":root { --color-primary: #6D28D9; } .btn { }",
    "frontend/src/styles/app.css": ".chat { display: grid; } @media (max-width: 768px) { .chat { display: block; } }",
    "frontend/src/main.jsx": ("import { StrictMode } from 'react';\nimport { createRoot } from 'react-dom/client';\n"
                              "import { BrowserRouter } from 'react-router';\nimport './styles/tokens.css';\n"
                              "import './styles/app.css';\nimport App from './App.jsx';\n"
                              "createRoot(document.getElementById('root')).render(<StrictMode><BrowserRouter>"
                              "<App /></BrowserRouter></StrictMode>);\n"),
    "frontend/src/App.jsx": ("import { Routes, Route, NavLink } from 'react-router';\nimport Chat from './pages/Chat.jsx';\n"
                             "import Settings from './pages/Settings.jsx';\nexport default function App() {\n  return (\n"
                             "    <div className=\"app-shell\"><header className=\"topbar\"><nav className=\"nav\">"
                             "<NavLink className=\"nav-link\" to=\"/\">Chat</NavLink></nav></header>\n"
                             "    <main className=\"main\"><Routes><Route path=\"/\" element={<Chat />} />"
                             "<Route path=\"/settings\" element={<Settings />} /></Routes></main></div>\n  );\n}\n"),
    "frontend/src/pages/Chat.jsx": ("import { useState } from 'react';\nimport { sendMessage } from '../lib/api.js';\n"
                                    "export default function Chat() {\n  const [messages, setMessages] = useState([]);\n"
                                    "  const [loading, setLoading] = useState(false);\n  const [error, setError] = useState(null);\n"
                                    "  return (<section className=\"chat\">{messages.length === 0 ? <div className=\"empty-state\">"
                                    "<h3>Start a conversation</h3></div> : messages.map((m) => <div className=\"bubble\" key={m.id}>{m.text}</div>)}"
                                    "<form className=\"composer\" onSubmit={() => sendMessage(messages)}><label htmlFor=\"msg\" className=\"visually-hidden\">Message</label>"
                                    "<textarea id=\"msg\" className=\"textarea\" /><button className=\"btn btn-primary\">Send</button></form></section>);\n}\n"),
    "frontend/src/pages/Settings.jsx": ("import { useState } from 'react';\nimport { saveSettings } from '../lib/storage.js';\n"
                                        "export default function Settings() {\n  const [url, setUrl] = useState('');\n"
                                        "  const onSubmit = (e) => { e.preventDefault(); saveSettings({ url }); };\n"
                                        "  return (<form className=\"card\" onSubmit={onSubmit}><div className=\"field\"><label htmlFor=\"url\" className=\"label\">API address</label>"
                                        "<input id=\"url\" className=\"input\" value={url} onChange={(e) => setUrl(e.target.value)} /></div>"
                                        "<button className=\"btn btn-primary\">Save</button></form>);\n}\n"),
    "frontend/src/lib/api.js": ("import { loadSettings } from './storage.js';\nexport async function sendMessage(messages) {\n"
                                "  const { url, key } = loadSettings();\n  const res = await fetch(url, { headers: { Authorization: `Bearer ${key}` } });\n"
                                "  if (!res.ok) throw new Error(`HTTP ${res.status}`);\n  return res.json();\n}\n"),
    "frontend/src/lib/storage.js": ("const KEY = 'chatbot.settings';\nexport function loadSettings() { try { return JSON.parse("
                                    "localStorage.getItem(KEY)) || {}; } catch { return {}; } }\n"
                                    "export function saveSettings(s) { localStorage.setItem(KEY, JSON.stringify(s)); }\n"),
}


def test_well_built_tree_has_no_fatal_issues():
    issues = design.check_tree(_brief(), GOOD_FILES)
    assert design.fatal_issues(issues) == [], [i.as_dict() for i in design.fatal_issues(issues)]


def test_checks_are_noops_for_non_ui_briefs_and_never_raise():
    assert design.check_tree(design.build_brief("a python cli", []), DEMO_FILES) == []
    assert design.check_tree(None, DEMO_FILES) == []
    # malformed content never raises; a lone unstyled jsx legitimately has issues
    assert isinstance(design.check_tree(_brief(), {"weird.jsx": None}), list)


def test_html_framework_checks_link_and_viewport():
    b = design.build_brief("a landing page website for a bakery", [])
    issues = design.check_tree(b, {"index.html": "<html><head></head><body><header class=\"topbar\"></header></body></html>",
                                   "css/tokens.css": ":root{}", "css/styles.css": ".hero{}"})
    ids = _ids(design.fatal_issues(issues))
    assert "D02-stylesheet-not-linked" in ids and "D05-no-viewport" in ids


def test_stub_component_is_fatal_but_real_components_pass():
    files = dict(GOOD_FILES)
    files["frontend/src/components/TopBar.jsx"] = ("import React from 'react';\nconst TopBar = () => {\n  return (\n    <div>\n"
                                                   "      {/* Top Bar Content */}\n    </div>\n  );\n};\nexport default TopBar;\n")
    issues = design.check_tree(_brief(), files)
    stubs = [i for i in issues if i.id == "D17-stub-component"]
    assert [i.file for i in stubs] == ["frontend/src/components/TopBar.jsx"] and stubs[0].severity == "fatal"
    # single-element components with content or expressions are not stubs
    files["frontend/src/components/TopBar.jsx"] = "export default function TopBar({ title }) { return (<header className=\"topbar\">{title}</header>); }\n"
    files["frontend/src/components/Spinner.jsx"] = "export default function Spinner() { return <span className=\"spinner\" aria-label=\"Loading\" />; }\n"
    assert not [i for i in design.check_tree(_brief(), files) if i.id == "D17-stub-component"]


def test_secret_logging_is_fatal_and_dead_links_are_warnings():
    files = dict(GOOD_FILES)
    files["frontend/src/lib/api.js"] = GOOD_FILES["frontend/src/lib/api.js"] + "console.log('key', settings.apiKey);\n"
    files["frontend/src/App.jsx"] = GOOD_FILES["frontend/src/App.jsx"].replace(
        '<NavLink className="nav-link" to="/">Chat</NavLink>',
        '<NavLink className="nav-link" to="/">Chat</NavLink><NavLink className="nav-link" to="/history">History</NavLink>')
    issues = design.check_tree(_brief(), files)
    ids = {i.id: i for i in issues}
    assert ids["D18-secret-logged"].severity == "fatal" and ids["D18-secret-logged"].file == "frontend/src/lib/api.js"
    assert ids["D20-dead-link"].severity == "warning" and "/history" in ids["D20-dead-link"].message
    assert "D20-dead-link" not in {i.id for i in design.check_tree(_brief(), GOOD_FILES)}
