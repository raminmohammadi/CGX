"""The design layer's brief: UI detection, archetype/theme/framework selection,
canonical paths, product naming, and the deterministic tokens stylesheet."""

from __future__ import annotations

import pytest

from cgx import design
from cgx.design import themes

DEMO_GOAL = ("create an app for a chatbot in which user provides a API address and a key "
             "on the profile tab and then navigates to the main page and can interact "
             "with the chat. Use React frontend")


def test_demo_goal_resolves_to_a_chat_assistant_brief():
    b = design.build_brief(DEMO_GOAL, ["react"])
    assert b.applies
    assert b.framework == "react" and b.styling == "css-tokens"
    assert b.archetype == "chat-assistant"
    assert b.theme == "conversational"
    assert b.external_api and b.secrets and b.multi_screen and b.persistence
    assert b.product_name == "Chatbot"
    # canonical UI files at the repo root (no backend named)
    assert b.app_root == ""
    assert b.tokens_path == "src/styles/tokens.css"
    assert b.entry_path == "src/main.jsx" and b.app_path == "src/App.jsx"
    assert b.api_client_path == "src/lib/api.js" and b.storage_path == "src/lib/storage.js"
    # archetype requirements + universal rules + external-API rule are all carried
    assert any("localStorage" in r for r in b.requirements)
    assert any("Bearer" in n for n in b.integration_notes)


@pytest.mark.parametrize("goal,expect", [
    ("a python fastapi rest api for a todo list", False),
    ("a python cli tool that renames files", False),
    ("a library for parsing csv files", False),
    ("Build a kanban task board web app, dark mode. Use Vue.", True),
    ("a landing page website for a coffee shop with menu and contact form", True),
    ("an analytics dashboard showing sales KPIs with charts, React + Tailwind", True),
])
def test_applies_only_to_ui_bearing_goals(goal, expect):
    assert design.applies(goal, []) is expect


def test_framework_defaults_and_styling():
    assert design.build_brief("Build a kanban board. Use Vue.", []).framework == "vue"
    assert design.build_brief("a landing page website for a bakery", []).framework == "html"
    assert design.build_brief("an expense tracker app", []).framework == "react"  # app -> React default
    b = design.build_brief("an analytics dashboard, React + Tailwind", [])
    assert b.styling == "tailwind" and b.tokens_path == "src/index.css" and b.global_css_path == ""
    assert b.archetype == "analytics-dashboard" and b.theme == "midnight-console"


def test_named_product_and_dark_request():
    assert design.build_brief("build an app called PulseNote for taking notes", []).product_name == "PulseNote"
    assert design.build_brief("a todo app with dark mode", []).theme == "midnight-console"


def test_backend_goal_places_ui_under_frontend_and_resolves_to_plan_root():
    b = design.build_brief("a React frontend with a Flask backend for notes", ["react", "flask"])
    assert b.app_root == "frontend" and b.tokens_path == "frontend/src/styles/tokens.css"
    # the plan is authoritative once it exists
    r = b.resolved_for_paths(["ui/index.html", "ui/src/App.tsx", "backend/app.py"])
    assert r.app_root == "ui" and r.typescript and r.entry_path == "ui/src/main.tsx"


def test_ui_files_to_inject_only_adds_missing_canonical_files():
    b = design.build_brief(DEMO_GOAL, ["react"]).resolved_for_paths(
        ["frontend/index.html", "frontend/src/App.jsx", "frontend/src/main.jsx"])
    have = ["frontend/index.html", "frontend/package.json", "frontend/vite.config.js",
            "frontend/src/App.jsx", "frontend/src/main.jsx", "frontend/src/Chat.jsx", "README.md"]
    inject = {f["path"] for f in design.ui_files_to_inject(b, have)}
    assert inject == {"frontend/src/styles/tokens.css", "frontend/src/styles/app.css",
                      "frontend/src/lib/api.js", "frontend/src/lib/storage.js"}
    # idempotent: nothing left once injected
    assert design.ui_files_to_inject(b, have + sorted(inject)) == []


def test_plan_problems_flags_a_ui_plan_with_no_screens():
    b = design.build_brief(DEMO_GOAL, ["react"])
    assert design.plan_problems(b, ["index.html", "package.json", "src/main.jsx", "src/App.jsx"])
    assert design.plan_problems(b, ["index.html", "src/main.jsx", "src/App.jsx",
                                    "src/pages/Chat.jsx", "src/pages/Settings.jsx"]) == []
    assert design.plan_problems(design.build_brief("a python cli", []), []) == []


def test_every_theme_palette_meets_aa_where_it_is_used_for_text():
    for t in themes.THEMES:
        p = t.palette
        assert themes.contrast_ratio(p["text"], p["bg"]) >= 4.5, t.key
        assert themes.contrast_ratio(p["muted"], p["bg"]) >= 4.5, t.key
        assert themes.contrast_ratio(p["muted"], p["surface"]) >= 4.5, t.key
        assert themes.contrast_ratio(p["primary_contrast"], p["primary"]) >= 4.5, t.key


def test_tokens_stylesheet_is_complete_and_contrast_safe():
    b = design.build_brief(DEMO_GOAL, ["react"])
    css = design.tokens_content(b)
    for needle in (":root", "--color-primary", "--color-link", "--font-heading", "--space-4",
                   "--radius-md", "--shadow-sm", ".btn-primary", ".card", ".input", ".topbar",
                   ".nav-link", ".empty-state", ".spinner", "@media (max-width: 768px)",
                   ":focus-visible", "fonts.googleapis.com"):
        assert needle in css, needle
    # a decorative-only accent never becomes the link colour
    playful = themes.tokens_stylesheet(themes.theme_by_key("playful-vibrant"))
    assert "--color-link: #2B1B4A" in playful
    # tailwind flavour is a v4 entry with @theme tokens
    tw = design.tokens_content(design.build_brief("an analytics dashboard, React + Tailwind", []))
    assert '@import "tailwindcss";' in tw and "@theme" in tw and "--color-primary" in tw


def test_prompt_blocks_are_bounded_and_file_specific():
    b = design.build_brief(DEMO_GOAL, ["react"])
    pb = design.planner_block(b)
    assert "UI/UX PRODUCT SPEC" in pb and "src/styles/tokens.css" in pb and len(pb) < 7000
    comp = design.file_block(b, "src/pages/Chat.jsx", "chat screen")
    assert "COMPONENT RULES" in comp and "DESIGN SYSTEM" in comp and "composer" in comp.lower()
    entry = design.file_block(b, "src/main.jsx", "entry")
    assert "createRoot" in entry and "ReactDOM.render" in entry and "BrowserRouter" in entry
    api = design.file_block(b, "src/lib/api.js", "api client")
    assert "Bearer" in api and "fetch" in api
    assert design.file_block(b, "src/styles/tokens.css", "tokens") == ""  # generated, never prompted
    assert design.file_block(b, "backend/app.py", "flask app") == ""
    assert design.file_block(b, "tests/app.test.jsx", "test") == ""
    assert design.file_kind(b, "src/styles/app.css") == "stylesheet"
    assert design.file_kind(b, "package.json") == "package"
