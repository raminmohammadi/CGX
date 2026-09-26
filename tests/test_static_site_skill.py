"""Tests for the StaticSiteSkill: detection, plan/scaffold validation, and
the local link/asset resolver."""

from __future__ import annotations

import pytest

import skills as registry
from skills.static_site import StaticSiteSkill


@pytest.fixture()
def skill() -> StaticSiteSkill:
    return StaticSiteSkill()


# --- detection -------------------------------------------------------------

@pytest.mark.parametrize("goal", [
    "build a static HTML website",
    "make me a landing page",
    "a portfolio site in plain html/css/js",
    "multi-page marketing site, no framework",
    "vanilla js website",
])
def test_detects_static_goals(skill, goal):
    assert skill.detect(goal) >= 0.9


@pytest.mark.parametrize("goal", [
    "build a React site",
    "a Next.js landing page",
    "portfolio site with Vue",
    "svelte web app",
])
def test_does_not_fire_when_framework_named(skill, goal):
    assert skill.detect(goal) == 0.0


def test_registered_and_detected_via_registry():
    detected = [s.name for s in registry.detect_skills("build a static html site")]
    assert "static_site" in detected
    # A React goal must NOT pull in the static-site skill.
    react = [s.name for s in registry.detect_skills("build a react site")]
    assert "static_site" not in react


# --- scaffold validation ---------------------------------------------------

def _diff(path, patch=""):
    return {"file": path, "patch": patch}


def test_validate_scaffold_requires_index_html(skill):
    v = skill.validate_scaffold([_diff("about.html"), _diff("css/style.css")])
    assert v is not None and not v.passed
    assert "index.html" in v.rationale


def test_validate_scaffold_rejects_framework_leak(skill):
    v = skill.validate_scaffold([_diff("index.html"), _diff("package.json"),
                                 _diff("src/App.jsx")])
    assert v is not None and not v.passed
    assert "package.json" in v.rationale or "App.jsx" in v.rationale


def test_validate_scaffold_passes_clean_site(skill):
    diffs = [
        _diff("index.html",
              '<link rel="stylesheet" href="css/style.css">'
              '<a href="about.html">About</a>'
              '<script src="js/script.js"></script>'),
        _diff("about.html", '<link rel="stylesheet" href="css/style.css">'),
        _diff("css/style.css", "body { margin: 0; }"),
        _diff("js/script.js", "console.log('hi');"),
    ]
    assert skill.validate_scaffold(diffs) is None


def test_validate_scaffold_flags_broken_link(skill):
    diffs = [
        _diff("index.html", '<a href="missing.html">Gone</a>'
                            '<link rel="stylesheet" href="css/style.css">'),
        _diff("css/style.css", "body{}"),
    ]
    v = skill.validate_scaffold(diffs)
    assert v is not None and not v.passed
    assert "missing.html" in v.rationale


def test_broken_link_ignores_external_and_anchors(skill):
    diffs = [
        _diff("index.html",
              '<a href="https://example.com">ext</a>'
              '<a href="#top">anchor</a>'
              '<a href="mailto:x@y.z">mail</a>'
              '<link rel="stylesheet" href="css/style.css">'),
        _diff("css/style.css", "body{}"),
    ]
    assert skill.validate_scaffold(diffs) is None


def test_relative_link_resolution_with_subdir(skill):
    # A page under pages/ linking ../index.html must resolve.
    diffs = [
        _diff("index.html", ""),
        _diff("pages/about.html", '<a href="../index.html">Home</a>'),
    ]
    assert skill.validate_scaffold(diffs) is None


def test_css_url_reference_checked(skill):
    diffs = [
        _diff("index.html", '<link rel="stylesheet" href="css/style.css">'),
        _diff("css/style.css", "body { background: url('../assets/bg.png'); }"),
    ]
    v = skill.validate_scaffold(diffs)
    assert v is not None and "bg.png" in v.rationale


# --- plan validation -------------------------------------------------------

def test_validate_plan_requires_index(skill):
    v = skill.validate_plan([_diff("about.html")])
    assert v is not None and "index.html" in v.rationale


def test_validate_plan_rejects_package_json(skill):
    v = skill.validate_plan([_diff("index.html"), _diff("package.json")])
    assert v is not None and "package.json" in v.rationale


def test_scaffold_warning_when_no_css(skill):
    warns = skill.scaffold_warnings([_diff("index.html")])
    assert warns and warns[0].severity == "warning"
