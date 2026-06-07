"""Tests for the top-level ``skills`` package and its integration with
the Planner, the answer engine, and the Judge.

The skills package is intentionally light on its own dependencies so the
tests here exercise it directly without any LLM provider stubbing
beyond what the Judge integration tests already do.
"""

from __future__ import annotations

from typing import Any, Dict, List

import skills
from skills.base import SKILL_DETECT_THRESHOLD, Skill, SkillVerdict


# ---------------------------------------------------------------------------
# Registry shape
# ---------------------------------------------------------------------------
def test_registry_contains_every_documented_skill():
    names = {s.name for s in skills.SKILLS}
    expected = {
        "react", "nextjs", "vue", "tailwind",
        "fastapi", "flask", "django", "express",
        "python_cli", "sqlite",
    }
    assert expected.issubset(names), (
        f"missing skills: {sorted(expected - names)}"
    )


def test_registry_skills_have_required_attributes():
    for s in skills.SKILLS:
        assert isinstance(s, Skill)
        assert s.name and s.name == s.name.lower()
        assert s.role in {"frontend", "backend", "fullstack", "data",
                          "cli", "style", "infra"}


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def test_detect_react_fastapi_combo():
    detected = skills.detect_skills(
        "create a calculator using React UI and FastAPI backend in python")
    names = [s.name for s in detected]
    assert "react" in names and "fastapi" in names


def test_detect_returns_empty_for_unrelated_goal():
    assert skills.detect_skills("what does the auth module do?") == []
    assert skills.detect_skills("") == []


def test_detect_react_native_does_not_fire_react():
    detected = [s.name for s in skills.detect_skills(
        "build a React Native mobile app")]
    assert "react" not in detected


def test_detect_python_cli_skips_when_web_framework_mentioned():
    # A goal that names both "python CLI" and "FastAPI" should route to
    # FastAPI, not the CLI skill -- the CLI skill abstains on web mentions.
    detected = [s.name for s in skills.detect_skills(
        "build a python CLI tool that wraps a FastAPI service")]
    assert "fastapi" in detected
    assert "python_cli" not in detected


def test_skills_by_names_silently_skips_unknown():
    resolved = skills.skills_by_names(["react", "does-not-exist", "fastapi"])
    assert [s.name for s in resolved] == ["react", "fastapi"]


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------
def test_compose_scaffold_prompt_joins_active_skills():
    react = skills.skills_by_names(["react"])
    prompt = skills.compose_scaffold_prompt(react)
    assert "React" in prompt and "Vite" in prompt
    assert "main.jsx" in prompt


def test_compose_scaffold_prompt_empty_for_no_skills():
    assert skills.compose_scaffold_prompt([]) == ""


# ---------------------------------------------------------------------------
# Per-skill validators
# ---------------------------------------------------------------------------
def _diff(file: str, patch: str = "") -> Dict[str, Any]:
    return {"file": file, "patch": patch}


def test_react_validator_fails_when_no_js_files():
    react = skills.skills_by_names(["react"])
    diffs = [_diff("backend/main.py", "from fastapi import FastAPI")]
    v = skills.validate_scaffold(react, diffs)
    assert v is not None and not v.passed
    assert "react" in v.skill.lower() or "React" in v.rationale


def test_react_validator_passes_when_jsx_present():
    react = skills.skills_by_names(["react"])
    diffs = [_diff("src/App.jsx", "import React from 'react';"),
             _diff("package.json", '{"dependencies":{"react":"^18"}}')]
    assert skills.validate_scaffold(react, diffs) is None


def test_fastapi_validator_requires_requirements_file():
    fa = skills.skills_by_names(["fastapi"])
    diffs = [_diff("backend/main.py",
                   "from fastapi import FastAPI\napp = FastAPI()")]
    v = skills.validate_scaffold(fa, diffs)
    assert v is not None and not v.passed
    assert "requirements" in v.rationale.lower()


def test_tailwind_validator_fails_without_config():
    tw = skills.skills_by_names(["tailwind"])
    diffs = [_diff("src/index.css",
                   "@tailwind base;\n@tailwind components;\n@tailwind utilities;")]
    v = skills.validate_scaffold(tw, diffs)
    assert v is not None and not v.passed
    assert "tailwind.config" in v.rationale


# ---------------------------------------------------------------------------
# Non-fatal "missing tests" warnings
# ---------------------------------------------------------------------------
def test_collect_warnings_flags_missing_tests_for_react_scaffold():
    react = skills.skills_by_names(["react"])
    diffs = [_diff("src/App.jsx", "import React from 'react';"),
             _diff("package.json", '{"dependencies":{"react":"^18"}}')]
    warnings = skills.collect_scaffold_warnings(react, diffs)
    assert len(warnings) == 1
    w = warnings[0]
    assert w.severity == "warning" and not w.passed
    assert w.skill == "react"
    assert "test" in w.rationale.lower()


def test_collect_warnings_silent_when_react_test_present():
    react = skills.skills_by_names(["react"])
    diffs = [_diff("src/App.jsx", "import React from 'react';"),
             _diff("tests/App.test.jsx", "import {render} from '@testing-library/react'"),
             _diff("package.json", "{}")]
    assert skills.collect_scaffold_warnings(react, diffs) == []


def test_collect_warnings_flags_missing_tests_for_python_backend():
    fa = skills.skills_by_names(["fastapi"])
    diffs = [_diff("backend/main.py",
                   "from fastapi import FastAPI\napp = FastAPI()"),
             _diff("backend/requirements.txt", "fastapi\nuvicorn\n")]
    warnings = skills.collect_scaffold_warnings(fa, diffs)
    assert len(warnings) == 1
    assert warnings[0].severity == "warning"
    assert "test" in warnings[0].rationale.lower()


def test_collect_warnings_silent_when_python_test_present():
    fa = skills.skills_by_names(["fastapi"])
    diffs = [_diff("backend/main.py",
                   "from fastapi import FastAPI\napp = FastAPI()"),
             _diff("backend/requirements.txt", "fastapi\n"),
             _diff("tests/test_main.py",
                   "from fastapi.testclient import TestClient")]
    assert skills.collect_scaffold_warnings(fa, diffs) == []


def test_warnings_are_not_returned_by_validate_scaffold():
    # validate_scaffold must only surface fatal failures; warnings live
    # in collect_scaffold_warnings so they don't fail the task.
    react = skills.skills_by_names(["react"])
    diffs = [_diff("src/App.jsx", "import React from 'react';"),
             _diff("package.json", "{}")]
    # No tests \u2192 a warning exists, but validate_scaffold sees nothing.
    assert skills.validate_scaffold(react, diffs) is None
    assert skills.collect_scaffold_warnings(react, diffs)


def test_judge_surfaces_scaffold_warning_in_rationale_without_failing():
    # End-to-end: a React scaffold with no tests passes but the verdict
    # rationale tells the operator a test file is missing.
    from cgx.agents.judge import Judge
    from cgx.agents.types import Task, TaskKind
    task = Task(
        description="Generate a React calculator",
        kind=TaskKind.SCAFFOLD,
        criteria=["renders a calculator"],
        inputs={"skills": ["react"], "goal": "create a React calculator"},
        output={"plan_md": "react calc",
                "diffs": [_diff("src/App.jsx", "import React from 'react';"),
                          _diff("package.json", "{}")]},
    )
    v = Judge(provider=None).judge(task)
    assert v.verdict == "pass"
    assert "warning" in v.rationale.lower() and "test" in v.rationale.lower()


# ---------------------------------------------------------------------------
# Planner integration
# ---------------------------------------------------------------------------
def test_planner_attaches_detected_skill_names_to_scaffold_inputs():
    from cgx.agents.planner import Planner
    plan = Planner(provider=None).plan(
        "create a React calculator with FastAPI backend")
    manifest_tasks = [t for t in plan.tasks if t.kind.value == "scaffold_manifest"]
    assert manifest_tasks, "expected a scaffold_manifest task"
    attached = manifest_tasks[0].inputs.get("skills") or []
    assert "react" in attached and "fastapi" in attached
