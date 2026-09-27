"""Tests for the version-aware, deadlock-safe TailwindSkill validator.

Covers the three legitimate Tailwind setups -- CDN/build-less, v4
(`@import "tailwindcss";`), and v3 (config + PostCSS + directives) -- plus
the specific static_site deadlock the skill must NOT create.
"""

from __future__ import annotations

import pytest

import skills as registry
from skills.tailwind import TailwindSkill


@pytest.fixture()
def skill() -> TailwindSkill:
    return TailwindSkill()


def _diff(path, patch=""):
    return {"file": path, "patch": patch}


# --- detection (unchanged behavior) ---------------------------------------

@pytest.mark.parametrize("goal", [
    "style it with Tailwind",
    "use TailwindCSS for the UI",
    "a landing page using tailwind css",
])
def test_detects_tailwind_goals(skill, goal):
    assert skill.detect(goal) >= 0.9


def test_does_not_detect_unrelated(skill):
    assert skill.detect("build a plain css site") == 0.0


def test_registered_and_prompt_is_version_aware(skill):
    # Registry still finds it, and the prompt now teaches all three setups.
    assert "tailwind" in [s.name for s in registry.detect_skills("use tailwind")]
    prompt = skill.scaffold_system_prompt()
    assert "cdn.tailwindcss.com" in prompt          # CDN / build-less path
    assert '@import "tailwindcss"' in prompt          # v4 path
    assert "tailwind.config.js" in prompt             # v3 path


# --- (1) v4 acceptance: a single @import is a complete setup ---------------

def test_v4_import_needs_no_config(skill):
    diffs = [
        _diff("package.json", '{"devDependencies":{"tailwindcss":"^4"}}'),
        _diff("src/index.css", '@import "tailwindcss";'),
        _diff("src/main.jsx", "import './index.css';"),
    ]
    assert skill.validate_scaffold(diffs) is None


def test_v4_import_single_quotes_ok(skill):
    diffs = [
        _diff("package.json", '{"devDependencies":{"tailwindcss":"^4"}}'),
        _diff("app.css", "@import 'tailwindcss';"),
    ]
    assert skill.validate_scaffold(diffs) is None


# --- (2a) CDN / build-less: abstain, never reject --------------------------

def test_cdn_script_url_abstains(skill):
    diffs = [
        _diff("index.html",
              '<script src="https://cdn.tailwindcss.com"></script>'
              '<div class="p-4 text-center">hi</div>'),
        _diff("css/style.css", ".hero{gap:1rem}"),
    ]
    assert skill.validate_scaffold(diffs) is None


def test_cdn_generic_script_src_abstains(skill):
    diffs = [
        _diff("index.html", '<script src="/vendor/tailwind.min.js"></script>'),
    ]
    assert skill.validate_scaffold(diffs) is None


def test_no_package_json_is_buildless_and_abstains(skill):
    # A static site: no package.json anywhere -> no build system to configure.
    diffs = [
        _diff("index.html", '<div class="grid grid-cols-2">a</div>'),
        _diff("css/style.css", "body{margin:0}"),
    ]
    assert skill.validate_scaffold(diffs) is None


# --- (2b) the exact tailwind + static_site deadlock ------------------------

def test_static_site_plus_tailwind_do_not_deadlock():
    """static_site emits a CDN-Tailwind, package.json-free site; the tailwind
    skill must accept it (return None) so neither skill fatally rejects the
    other's valid output."""
    from skills.static_site import StaticSiteSkill

    site_diffs = [
        {"file": "index.html", "patch": (
            '<!DOCTYPE html><html lang="en"><head>'
            '<script src="https://cdn.tailwindcss.com"></script>'
            '<link rel="stylesheet" href="css/style.css"></head>'
            '<body class="bg-slate-900">'
            '<a href="js/script.js"></a></body></html>')},
        {"file": "css/style.css", "patch": "body{margin:0}"},
        {"file": "js/script.js", "patch": "console.log('hi');"},
    ]
    # static_site accepts its own output ...
    assert StaticSiteSkill().validate_scaffold(site_diffs) is None
    # ... and tailwind must NOT reject it.
    assert TailwindSkill().validate_scaffold(site_diffs) is None

    # And via the registry fatal-verdict gate (warnings excluded): no fatal.
    active = registry.skills_by_names(["static_site", "tailwind"])
    assert registry.validate_scaffold(active, site_diffs) is None


# --- (3) build-based v3: missing config is a WARNING, never fatal ----------

def test_build_based_missing_config_is_warning_not_fatal(skill):
    diffs = [
        _diff("package.json", '{"devDependencies":{"tailwindcss":"^3"}}'),
        _diff("src/index.css",
              "@tailwind base;\n@tailwind components;\n@tailwind utilities;"),
    ]
    v = skill.validate_scaffold(diffs)
    assert v is not None and not v.passed
    assert v.severity == "warning"          # downgraded from the old fatal error
    assert "tailwind.config" in v.rationale


def test_missing_config_warning_not_returned_by_fatal_gate(skill):
    # The registry fatal gate must skip the warning -> no regenerate.
    diffs = [
        _diff("package.json", '{"devDependencies":{"tailwindcss":"^3"}}'),
        _diff("src/index.css", "@tailwind base;"),
    ]
    active = registry.skills_by_names(["tailwind"])
    assert registry.validate_scaffold(active, diffs) is None
    warnings = registry.collect_scaffold_warnings(active, diffs)
    assert any(w.severity == "warning" and "tailwind.config" in w.rationale
               for w in warnings)


def test_build_based_config_without_directives_warns(skill):
    diffs = [
        _diff("package.json", '{"devDependencies":{"tailwindcss":"^3"}}'),
        _diff("tailwind.config.js", "module.exports = { content: [] }"),
        _diff("postcss.config.js", "module.exports={plugins:{tailwindcss:{}}}"),
    ]
    v = skill.validate_scaffold(diffs)
    assert v is not None and not v.passed and v.severity == "warning"


def test_complete_v3_setup_passes(skill):
    diffs = [
        _diff("package.json", '{"devDependencies":{"tailwindcss":"^3",'
                              '"postcss":"^8","autoprefixer":"^10"}}'),
        _diff("tailwind.config.js",
              "module.exports = { content: ['./index.html'] }"),
        _diff("postcss.config.js", "module.exports={plugins:{tailwindcss:{}}}"),
        _diff("src/index.css",
              "@tailwind base;\n@tailwind components;\n@tailwind utilities;"),
        _diff("src/main.jsx", "import './index.css';"),
    ]
    assert skill.validate_scaffold(diffs) is None


# --- edge cases ------------------------------------------------------------

def test_empty_diffs_abstain(skill):
    assert skill.validate_scaffold([]) is None


def test_no_validate_plan_opinion(skill):
    # Tailwind never rejects a plan (base default) -> deadlock-safe on plans.
    assert skill.validate_plan([_diff("index.html")]) is None
    assert skill.validate_plan([]) is None
