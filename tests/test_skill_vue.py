"""Tests for the VueSkill: detection, the new paths-only validate_plan gate,
and the Vite-specific scaffold checks (@vitejs/plugin-vue registration and the
package.json build trio). High-precision: legitimate Vue/Nuxt outputs and plain
component modifications must never be fatally rejected."""

from __future__ import annotations

import pytest

import skills as registry
from skills.vue import VueSkill


@pytest.fixture()
def skill() -> VueSkill:
    return VueSkill()


def _diff(file: str, patch: str = ""):
    return {"file": file, "patch": patch}


# --- detection -------------------------------------------------------------

@pytest.mark.parametrize("goal", [
    "build a Vue 3 dashboard",
    "a Vue.js single page app",
    "make a Nuxt marketing site",
])
def test_detects_vue_goals(skill, goal):
    assert skill.detect(goal) >= 0.9


def test_registered_and_detected_via_registry():
    detected = [s.name for s in registry.detect_skills("build a vue 3 app")]
    assert "vue" in detected


# --- validate_plan (paths only) -------------------------------------------

def test_plan_passes_with_full_vite_entry(skill):
    diffs = [_diff("frontend/index.html"), _diff("frontend/vite.config.js"),
             _diff("frontend/package.json"), _diff("frontend/src/main.js"),
             _diff("frontend/src/App.vue")]
    assert skill.validate_plan(diffs) is None


def test_plan_passes_with_typescript_entry(skill):
    diffs = [_diff("index.html"), _diff("vite.config.ts"),
             _diff("package.json"), _diff("src/main.ts")]
    assert skill.validate_plan(diffs) is None


def test_plan_vetoes_missing_vite_config(skill):
    # Skeleton is clearly being built (has package.json + main.js) but the
    # vite.config -- required to register the Vue plugin -- is absent.
    diffs = [_diff("index.html"), _diff("package.json"),
             _diff("src/main.js"), _diff("src/App.vue")]
    v = skill.validate_plan(diffs)
    assert v is not None and not v.passed
    assert "vite.config" in v.rationale


def test_plan_vetoes_missing_index_html(skill):
    diffs = [_diff("vite.config.js"), _diff("package.json"),
             _diff("src/main.js")]
    v = skill.validate_plan(diffs)
    assert v is not None and not v.passed and "index.html" in v.rationale


def test_plan_vetoes_missing_main_mount(skill):
    diffs = [_diff("index.html"), _diff("vite.config.js"),
             _diff("package.json")]
    v = skill.validate_plan(diffs)
    assert v is not None and not v.passed and "main" in v.rationale


def test_plan_vetoes_missing_package_json(skill):
    diffs = [_diff("index.html"), _diff("vite.config.js"),
             _diff("src/main.js")]
    v = skill.validate_plan(diffs)
    assert v is not None and not v.passed and "package.json" in v.rationale


def test_plan_ignores_component_only_modification(skill):
    # No skeleton/manifest file touched -> a plain "add a component" change,
    # not a scaffold. Must not be rejected for lacking the build entry.
    diffs = [_diff("src/components/Chart.vue"),
             _diff("src/components/Table.vue")]
    assert skill.validate_plan(diffs) is None


def test_plan_ignores_pure_backend_plan(skill):
    diffs = [_diff("backend/app.py"), _diff("requirements.txt")]
    assert skill.validate_plan(diffs) is None


def test_plan_skips_nuxt_by_goal(skill):
    # A Nuxt app legitimately has no vite.config/index.html/src/main.js.
    diffs = [_diff("nuxt.config.ts"), _diff("app.vue"),
             _diff("package.json"), _diff("pages/index.vue")]
    assert skill.validate_plan(diffs, goal="build a Nuxt blog") is None


def test_plan_skips_nuxt_by_config_file(skill):
    diffs = [_diff("nuxt.config.js"), _diff("app.vue"),
             _diff("package.json")]
    assert skill.validate_plan(diffs) is None


def test_plan_via_registry(skill):
    vue = registry.skills_by_names(["vue"])
    diffs = [_diff("index.html"), _diff("package.json"), _diff("src/main.js")]
    v = registry.validate_plan(vue, diffs)
    assert v is not None and not v.passed and v.skill == "vue"


# --- validate_scaffold: plugin-vue registration ----------------------------

_GOOD_PKG = ('{"dependencies":{"vue":"^3.4.0"},'
             '"devDependencies":{"vite":"^5.0.0",'
             '"@vitejs/plugin-vue":"^5.0.0"}}')
_GOOD_VITE = ("import { defineConfig } from 'vite'\n"
              "import vue from '@vitejs/plugin-vue'\n"
              "export default defineConfig({ plugins: [vue()] })\n")


def test_scaffold_passes_clean_vite_vue(skill):
    diffs = [
        _diff("index.html", '<div id="app"></div>'),
        _diff("vite.config.js", _GOOD_VITE),
        _diff("package.json", _GOOD_PKG),
        _diff("src/main.js", "import { createApp } from 'vue'"),
        _diff("src/App.vue", "<template><div/></template>"),
    ]
    assert skill.validate_scaffold(diffs) is None


def test_scaffold_fatal_when_plugin_vue_missing(skill):
    # .vue present but vite.config never registers @vitejs/plugin-vue -> the
    # build cannot compile the SFCs.
    diffs = [
        _diff("index.html", ""),
        _diff("vite.config.js",
              "import { defineConfig } from 'vite'\n"
              "export default defineConfig({ plugins: [] })\n"),
        _diff("package.json", _GOOD_PKG),
        _diff("src/App.vue", "<template><div/></template>"),
    ]
    v = skill.validate_scaffold(diffs)
    assert v is not None and not v.passed
    assert "@vitejs/plugin-vue" in v.rationale


def test_scaffold_fatal_when_no_vite_config_at_all(skill):
    diffs = [
        _diff("index.html", ""),
        _diff("package.json", _GOOD_PKG),
        _diff("src/App.vue", "<template><div/></template>"),
    ]
    v = skill.validate_scaffold(diffs)
    assert v is not None and not v.passed and "@vitejs/plugin-vue" in v.rationale


# --- validate_scaffold: package.json build trio ----------------------------

def test_scaffold_fatal_when_package_json_missing_deps(skill):
    diffs = [
        _diff("vite.config.js", _GOOD_VITE),
        _diff("package.json", '{"dependencies":{"vue":"^3.4.0"}}'),
        _diff("src/App.vue", "<template><div/></template>"),
    ]
    v = skill.validate_scaffold(diffs)
    assert v is not None and not v.passed
    assert "vite" in v.rationale and "@vitejs/plugin-vue" in v.rationale


def test_scaffold_pkg_check_does_not_confuse_vitejs_for_vite(skill):
    # "@vitejs/plugin-vue" contains the substring "vite"; the vite-dep check
    # must still flag a missing top-level `vite` devDependency.
    pkg = ('{"dependencies":{"vue":"^3.4.0"},'
           '"devDependencies":{"@vitejs/plugin-vue":"^5.0.0"}}')
    diffs = [
        _diff("vite.config.js", _GOOD_VITE),
        _diff("package.json", pkg),
        _diff("src/App.vue", "<template><div/></template>"),
    ]
    v = skill.validate_scaffold(diffs)
    assert v is not None and "vite (devDependencies)" in v.rationale


def test_scaffold_vue_dep_not_matched_by_vue_test_utils(skill):
    # "@vue/test-utils" must not satisfy the required top-level `vue` dep.
    pkg = ('{"devDependencies":{"vite":"^5.0.0",'
           '"@vitejs/plugin-vue":"^5.0.0","@vue/test-utils":"^2.0.0"}}')
    diffs = [
        _diff("vite.config.js", _GOOD_VITE),
        _diff("package.json", pkg),
        _diff("src/App.vue", "<template><div/></template>"),
    ]
    v = skill.validate_scaffold(diffs)
    assert v is not None and "vue (^3, dependencies)" in v.rationale


# --- validate_scaffold: existing behavior preserved ------------------------

def test_scaffold_fatal_when_no_frontend_source(skill):
    v = skill.validate_scaffold([_diff("backend/main.py", "from fastapi import x")])
    assert v is not None and not v.passed and ".vue" in v.rationale


# --- validate_scaffold: Nuxt is exempt from the Vite checks ----------------

def test_scaffold_skips_vite_checks_for_nuxt(skill):
    # Nuxt scaffold: .vue files, no vite.config, no @vitejs/plugin-vue -- all
    # legitimate. Must not be fatally rejected.
    diffs = [
        _diff("nuxt.config.ts", "export default defineNuxtConfig({})"),
        _diff("app.vue", "<template><NuxtPage/></template>"),
        _diff("pages/index.vue", "<template><h1>hi</h1></template>"),
        _diff("package.json", '{"dependencies":{"nuxt":"^3.10.0"}}'),
    ]
    assert skill.validate_scaffold(diffs, goal="build a Nuxt site") is None


# --- scaffold warnings preserved -------------------------------------------

def test_scaffold_warns_when_no_test_file(skill):
    warns = skill.scaffold_warnings([_diff("src/App.vue"), _diff("package.json")])
    assert warns and warns[0].severity == "warning"


def test_scaffold_no_warning_when_spec_present(skill):
    diffs = [_diff("src/App.vue"), _diff("tests/App.spec.js")]
    assert skill.scaffold_warnings(diffs) == []
