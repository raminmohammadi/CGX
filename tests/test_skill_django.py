"""Tests for the DjangoSkill: detection, prompt guidance, and the plan /
scaffold validators that prevent Django's common codegen failures --
missing project files, unpinned deps whose import name differs from the pip
name, and pytest-style tests that plain pytest cannot collect because no
config sets DJANGO_SETTINGS_MODULE.
"""

from __future__ import annotations

import pytest

import skills as registry
from skills.django import DjangoSkill


@pytest.fixture()
def skill() -> DjangoSkill:
    return DjangoSkill()


def _d(path, patch=""):
    """A diff row. ``patch`` is read by both base and structural helpers."""
    return {"file": path, "patch": patch}


def _p(path):
    """A paths-only diff row (validate_plan is fed paths only)."""
    return {"file": path}


# --- detection -------------------------------------------------------------

@pytest.mark.parametrize("goal", [
    "build a Django blog",
    "a Django REST Framework API for todos",
    "DRF backend with token auth",
    "django admin dashboard",
])
def test_detects_django_goals(skill, goal):
    assert skill.detect(goal) >= 0.9


@pytest.mark.parametrize("goal", [
    "build a Flask API",
    "a FastAPI service",
    "what does the auth module do?",
    "",
])
def test_abstains_on_non_django(skill, goal):
    assert skill.detect(goal) == 0.0


def test_registered_and_detected_via_registry():
    names = [s.name for s in registry.detect_skills("build a django rest api")]
    assert "django" in names


# --- prompt guidance -------------------------------------------------------

def test_scaffold_prompt_covers_fix_spec_points(skill):
    p = skill.scaffold_system_prompt()
    assert "INSTALLED_APPS" in p
    assert "include(" in p
    assert "models.Model" in p
    assert "DJANGO_SETTINGS_MODULE" in p
    # import != pip mapping is spelled out
    assert "djangorestframework" in p and "django-cors-headers" in p
    assert "rest_framework" in p and "corsheaders" in p


def test_plan_prompt_mentions_settings_module_and_include(skill):
    p = skill.plan_system_prompt()
    assert "DJANGO_SETTINGS_MODULE" in p
    assert "include(" in p


# --- validate_plan (paths only) -------------------------------------------

def test_plan_greenfield_missing_settings_and_urls_is_fatal(skill):
    v = skill.validate_plan([_p("manage.py"), _p("app/views.py")])
    assert v is not None and not v.passed
    assert "settings.py" in v.rationale and "urls.py" in v.rationale


def test_plan_greenfield_missing_urls_is_fatal(skill):
    v = skill.validate_plan([_p("manage.py"), _p("proj/settings.py")])
    assert v is not None and not v.passed
    assert "urls.py" in v.rationale


def test_plan_tests_without_pytest_config_is_fatal(skill):
    diffs = [_p("manage.py"), _p("proj/settings.py"), _p("proj/urls.py"),
             _p("tests/test_views.py")]
    v = skill.validate_plan(diffs)
    assert v is not None and not v.passed
    assert "DJANGO_SETTINGS_MODULE" in v.rationale


def test_plan_tests_with_pytest_ini_passes(skill):
    diffs = [_p("manage.py"), _p("proj/settings.py"), _p("proj/urls.py"),
             _p("tests/test_views.py"), _p("pytest.ini")]
    assert skill.validate_plan(diffs) is None


def test_plan_tests_with_conftest_passes(skill):
    diffs = [_p("manage.py"), _p("proj/settings.py"), _p("proj/urls.py"),
             _p("tests/test_views.py"), _p("conftest.py")]
    assert skill.validate_plan(diffs) is None


def test_plan_complete_scaffold_without_tests_passes(skill):
    diffs = [_p("manage.py"), _p("proj/settings.py"), _p("proj/urls.py"),
             _p("blog/models.py"), _p("requirements.txt")]
    assert skill.validate_plan(diffs) is None


def test_plan_incremental_edit_is_not_rejected(skill):
    # No manage.py in the plan -> an edit to an existing project, not a fresh
    # scaffold. Must NOT be fatally rejected for lacking a pytest config etc.
    assert skill.validate_plan([_p("proj/settings.py")]) is None
    assert skill.validate_plan(
        [_p("blog/views.py"), _p("blog/tests.py")]) is None
    assert skill.validate_plan(
        [_p("blog/tests/test_views.py")]) is None
    assert skill.validate_plan([]) is None


# --- validate_scaffold (sees content) -------------------------------------

def test_scaffold_no_python_is_fatal(skill):
    v = skill.validate_scaffold([_d("README.md", "# hi")])
    assert v is not None and not v.passed


def test_scaffold_missing_manage_py_is_fatal(skill):
    v = skill.validate_scaffold([_d("proj/settings.py", "import django")])
    assert v is not None and not v.passed
    assert "manage.py" in v.rationale


def test_scaffold_no_django_reference_is_fatal(skill):
    v = skill.validate_scaffold([_d("manage.py", "import os"),
                                 _d("proj/settings.py", "SECRET = 1")])
    assert v is not None and not v.passed
    assert "django" in v.rationale.lower()


def test_scaffold_undeclared_django_is_fatal(skill):
    # Django imported but no requirements.txt at all -> ships broken.
    diffs = [
        _d("manage.py", "import django\nfrom django.core.management import x"),
        _d("proj/settings.py", "from django.urls import path"),
    ]
    v = skill.validate_scaffold(diffs)
    assert v is not None and not v.passed
    assert "Django" in v.rationale


def test_scaffold_undeclared_drf_is_fatal(skill):
    diffs = [
        _d("manage.py", "import django"),
        _d("proj/settings.py", "from django.urls import path"),
        _d("api/views.py", "from rest_framework.views import APIView"),
        _d("requirements.txt", "Django>=4.2\n"),
    ]
    v = skill.validate_scaffold(diffs)
    assert v is not None and not v.passed
    # reports the PIP name, not the import name
    assert "djangorestframework" in v.rationale


def test_scaffold_undeclared_corsheaders_is_fatal(skill):
    diffs = [
        _d("manage.py", "import django"),
        _d("proj/settings.py", "import corsheaders\nfrom django.urls import path"),
        _d("requirements.txt", "Django>=4.2\n"),
    ]
    v = skill.validate_scaffold(diffs)
    assert v is not None and not v.passed
    assert "django-cors-headers" in v.rationale


def test_scaffold_tests_without_settings_module_is_warning(skill):
    # Structurally fine, deps declared, but pytest tests with no
    # DJANGO_SETTINGS_MODULE anywhere -> advisory warning, NOT fatal.
    diffs = [
        _d("manage.py", "import django"),
        _d("proj/settings.py", "from django.urls import path"),
        _d("requirements.txt", "Django>=4.2\n"),
        _d("tests/test_views.py", "def test_ok():\n    assert True\n"),
    ]
    v = skill.validate_scaffold(diffs)
    assert v is not None and not v.passed and v.severity == "warning"
    assert "DJANGO_SETTINGS_MODULE" in v.rationale


def test_scaffold_tests_with_pytest_ini_settings_passes(skill):
    diffs = [
        _d("manage.py", "import django"),
        _d("proj/settings.py", "from django.urls import path"),
        _d("requirements.txt", "Django>=4.2\n"),
        _d("tests/test_views.py", "def test_ok():\n    assert True\n"),
        _d("pytest.ini", "[pytest]\nDJANGO_SETTINGS_MODULE = proj.settings\n"),
    ]
    assert skill.validate_scaffold(diffs) is None


def test_scaffold_conftest_settings_configure_counts(skill):
    diffs = [
        _d("manage.py", "import django"),
        _d("proj/settings.py", "from django.urls import path"),
        _d("requirements.txt", "Django>=4.2\n"),
        _d("tests/test_views.py", "def test_ok():\n    assert True\n"),
        _d("conftest.py", "from django.conf import settings\n"
                          "settings.configure(DEBUG=True)\n"),
    ]
    assert skill.validate_scaffold(diffs) is None


# --- scaffold_warnings -----------------------------------------------------

def test_scaffold_warning_when_no_tests(skill):
    diffs = [_d("manage.py", "import django"),
             _d("proj/settings.py", "import django")]
    warns = skill.scaffold_warnings(diffs)
    assert warns and warns[0].severity == "warning"


def test_no_warning_when_django_style_tests_present(skill):
    diffs = [_d("manage.py", "import django"),
             _d("blog/tests.py", "from django.test import TestCase")]
    assert skill.scaffold_warnings(diffs) == []


# --- registry integration --------------------------------------------------

def test_settings_module_warning_is_not_surfaced_fatally():
    """The warning must live in collect_scaffold_warnings, not fail the task."""
    dj = registry.skills_by_names(["django"])
    diffs = [
        _d("manage.py", "import django"),
        _d("proj/settings.py", "from django.urls import path"),
        _d("requirements.txt", "Django>=4.2\n"),
        _d("tests/test_views.py", "def test_ok():\n    assert True\n"),
    ]
    # Fatal helper skips the warning...
    assert registry.validate_scaffold(dj, diffs) is None
    # ...collect_scaffold_warnings surfaces it.
    warns = registry.collect_scaffold_warnings(dj, diffs)
    assert any("DJANGO_SETTINGS_MODULE" in w.rationale for w in warns)


def test_validate_plan_via_registry_fires_for_greenfield():
    dj = registry.skills_by_names(["django"])
    diffs = [_p("manage.py"), _p("proj/settings.py"), _p("proj/urls.py"),
             _p("tests/test_views.py")]
    v = registry.validate_plan(dj, diffs, goal="django blog")
    assert v is not None and not v.passed and v.skill == "django"
