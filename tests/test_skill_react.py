"""Tests for the ReactSkill: detection, the new scaffold validators (JSX in a
plain .js/.ts that imports react, and a dangling index.html <script src> entry),
and the validate_plan false-positive fix that no longer vetoes small incremental
edits which omit index.html/package.json."""

from __future__ import annotations

import pytest

import skills as registry
from skills.react import ReactSkill


@pytest.fixture()
def skill() -> ReactSkill:
    return ReactSkill()


def _diff(path, content=""):
    return {"path": path, "content": content}


# --- detection -------------------------------------------------------------

@pytest.mark.parametrize("goal", [
    "build a React app for todos",
    "a react.js dashboard",
    "create a calculator using React and vite",
])
def test_detects_react_goals(skill, goal):
    assert skill.detect(goal) >= 0.9


def test_abstains_on_react_native(skill):
    assert skill.detect("build a React Native mobile app") == 0.0


def test_jsx_and_typo_are_soft_signals(skill):
    assert skill.detect("write a jsx component") == pytest.approx(0.6)
    assert skill.detect("build a reach frontend") == pytest.approx(0.6)
    assert skill.detect("") == 0.0


def test_registered_and_detected_via_registry():
    detected = [s.name for s in registry.detect_skills("build a React app")]
    assert "react" in detected


# --- prompt content --------------------------------------------------------

def test_scaffold_prompt_teaches_vite_layout(skill):
    p = skill.scaffold_system_prompt()
    assert "main.jsx" in p and "index.html" in p and "package.json" in p


# --- scaffold validation: happy path --------------------------------------

def _sanctioned_diffs():
    return [
        _diff("index.html",
              "<!doctype html><html><body><div id=\"root\"></div>"
              "<script type=\"module\" src=\"/src/main.jsx\"></script>"
              "</body></html>"),
        _diff("package.json",
              "{\"dependencies\":{\"react\":\"^18\",\"react-dom\":\"^18\"}}"),
        _diff("vite.config.js",
              "import react from '@vitejs/plugin-react'\n"
              "export default { plugins: [react()] }\n"),
        _diff("src/main.jsx",
              "import React from 'react'\n"
              "import { createRoot } from 'react-dom/client'\n"
              "import App from './App.jsx'\n"
              "createRoot(document.getElementById('root')).render(<App />)\n"),
        _diff("src/App.jsx",
              "import React from 'react'\n"
              "export default function App() {\n"
              "  return (<div>hi</div>)\n}\n"),
        _diff("tests/App.test.jsx",
              "import { render } from '@testing-library/react'\n"
              "import App from '../src/App.jsx'\n"
              "test('renders', () => { render(<App />) })\n"),
    ]


def test_validate_scaffold_passes_sanctioned_layout(skill):
    assert skill.validate_scaffold(_sanctioned_diffs(), goal="todos") is None


# --- scaffold validation: existing fatal checks preserved ------------------

def test_no_js_files_rejected(skill):
    v = skill.validate_scaffold([_diff("README.md", "# hi")])
    assert v is not None and not v.passed and ".jsx" in v.rationale


def test_all_python_rejected(skill):
    v = skill.validate_scaffold([
        _diff("app.py", "print('hi')"),
        _diff("src/main.jsx", "import React from 'react'\n"),
    ])
    # the .jsx presence means the source is not all-python; must pass here
    assert v is None


# --- scaffold validation: NEW jsx-in-plain-js fatal ------------------------

def test_jsx_in_js_importing_react_is_fatal(skill):
    diffs = [
        _diff("src/App.js",
              "import React from 'react'\n"
              "export default function App() {\n"
              "  return (<div>hello</div>)\n}\n"),
        _diff("index.html",
              "<script type=\"module\" src=\"/src/main.jsx\"></script>"),
        _diff("src/main.jsx", "import React from 'react'\n"),
        _diff("package.json", "{\"dependencies\":{\"react\":\"^18\"}}"),
    ]
    v = skill.validate_scaffold(diffs)
    assert v is not None and not v.passed
    assert "src/App.js" in v.rationale and ".jsx/.tsx" in v.rationale


def test_jsx_closing_tag_in_ts_importing_react_is_fatal(skill):
    diffs = [
        _diff("src/widget.ts",
              "import { useState } from 'react'\n"
              "export const W = () => <span>x</span>\n"),
    ]
    v = skill.validate_scaffold(diffs)
    assert v is not None and not v.passed and "src/widget.ts" in v.rationale


def test_jsx_in_proper_jsx_file_is_fine(skill):
    # Same JSX, correct extension -> no jsx-in-plain-js veto.
    v = skill.validate_scaffold(_sanctioned_diffs())
    assert v is None


def test_react_import_without_jsx_is_fine(skill):
    # A plain .js that imports react but has no JSX (e.g. a hook/util) is OK.
    diffs = [
        _diff("src/useCounter.js",
              "import { useState } from 'react'\n"
              "export function useCounter() { return useState(0) }\n"),
        _diff("index.html",
              "<script type=\"module\" src=\"/src/main.jsx\"></script>"),
        _diff("src/main.jsx", "import React from 'react'\n"),
        _diff("package.json", "{}"),
    ]
    assert skill.validate_scaffold(diffs) is None


def test_jsx_in_js_without_react_import_is_not_flagged(skill):
    # No react import -> the file is not a React source; don't fatally reject.
    diffs = [
        _diff("src/tpl.js",
              "export const html = '<div></div>'\n"
              "function f(a, b) { return (a < b) }\n"),
        _diff("src/main.jsx", "import React from 'react'\n"),
        _diff("index.html",
              "<script type=\"module\" src=\"/src/main.jsx\"></script>"),
        _diff("package.json", "{}"),
    ]
    assert skill.validate_scaffold(diffs) is None


def test_vite_config_importing_plugin_react_is_not_flagged(skill):
    # vite.config.js imports @vitejs/plugin-react, not 'react' -> no false hit.
    diffs = [
        _diff("vite.config.js",
              "import react from '@vitejs/plugin-react'\n"
              "export default { plugins: [react()] }\n"),
        _diff("src/main.jsx", "import React from 'react'\n"),
        _diff("index.html",
              "<script type=\"module\" src=\"/src/main.jsx\"></script>"),
        _diff("package.json", "{}"),
    ]
    assert skill.validate_scaffold(diffs) is None


# --- scaffold validation: NEW dangling index.html entry fatal --------------

def test_dangling_html_entry_is_fatal(skill):
    diffs = [
        _diff("index.html",
              "<script type=\"module\" src=\"/src/main.jsx\"></script>"),
        # generated entry is .tsx, but index.html asks for main.jsx
        _diff("src/main.tsx", "import React from 'react'\n"),
        _diff("src/App.tsx",
              "import React from 'react'\n"
              "export default function App() { return (<div/>) }\n"),
        _diff("package.json", "{}"),
    ]
    v = skill.validate_scaffold(diffs)
    assert v is not None and not v.passed
    assert "/src/main.jsx" in v.rationale


def test_resolved_html_entry_passes(skill):
    diffs = [
        _diff("index.html",
              "<script type=\"module\" src=\"/src/main.jsx\"></script>"),
        _diff("src/main.jsx", "import React from 'react'\n"),
        _diff("package.json", "{}"),
    ]
    assert skill.validate_scaffold(diffs) is None


def test_external_script_src_is_ignored(skill):
    diffs = [
        _diff("index.html",
              "<script src=\"https://cdn.example.com/react.js\"></script>"
              "<script type=\"module\" src=\"/src/main.jsx\"></script>"),
        _diff("src/main.jsx", "import React from 'react'\n"),
        _diff("package.json", "{}"),
    ]
    assert skill.validate_scaffold(diffs) is None


# --- validate_plan: incremental-edit false-positive fix --------------------

def test_plan_incremental_single_component_no_veto(skill):
    # A 1-file edit to an existing component must NOT be vetoed for missing
    # index.html / package.json -- they already live in the repo.
    assert skill.validate_plan([_diff("src/components/Button.jsx")]) is None


def test_plan_incremental_two_files_no_veto(skill):
    assert skill.validate_plan([
        _diff("src/App.jsx"),
        _diff("src/components/Button.jsx"),
    ]) is None


def test_plan_full_scaffold_missing_entry_is_fatal(skill):
    # >= 3 files reads as a fresh scaffold: the missing Vite entry chain
    # (index.html + package.json) is still a fatal veto.
    v = skill.validate_plan([
        _diff("src/main.jsx"),
        _diff("src/App.jsx"),
        _diff("src/components/List.jsx"),
    ])
    assert v is not None and not v.passed
    assert "index.html" in v.rationale and "package.json" in v.rationale


def test_plan_manifest_marker_triggers_entry_check(skill):
    # Only 2 files, but package.json marks it as a scaffold -> still require
    # the missing index.html.
    v = skill.validate_plan([
        _diff("package.json"),
        _diff("src/App.jsx"),
    ])
    assert v is not None and not v.passed and "index.html" in v.rationale


def test_plan_complete_scaffold_passes(skill):
    assert skill.validate_plan([
        _diff("index.html"),
        _diff("package.json"),
        _diff("src/App.jsx"),
        _diff("src/main.jsx"),
    ]) is None


def test_plan_no_js_is_none(skill):
    assert skill.validate_plan([_diff("README.md"), _diff("styles.css")]) is None


def test_plan_none_on_empty(skill):
    assert skill.validate_plan([]) is None


# --- warnings --------------------------------------------------------------

def test_warns_when_no_test_file(skill):
    warns = skill.scaffold_warnings([
        _diff("src/App.jsx", "import React from 'react'\n"),
        _diff("src/main.jsx", "import React from 'react'\n"),
    ])
    assert warns and warns[0].severity == "warning"


def test_no_warning_when_js_test_present(skill):
    assert skill.scaffold_warnings([
        _diff("src/App.jsx", "import React from 'react'\n"),
        _diff("tests/App.test.jsx", "test('x', () => {})\n"),
    ]) == []
