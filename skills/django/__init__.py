

"""Django backend skill."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from skills._structural import undeclared_python_deps
from skills.base import (
    Skill,
    SkillVerdict,
    file_paths,
    file_with_content,
    has_python_test_file,
)

_DJANGO_RE = re.compile(r"\bdjango\b", re.IGNORECASE)
_DRF_RE = re.compile(r"\bdjango\s*rest\s*framework\b|\bdrf\b", re.IGNORECASE)

# Django deps whose import name differs from the pip package name (import!=pip),
# used to flag a use that was never declared (a phantom dep that ships broken).
# ``django`` itself is 1:1 but is included so a scaffold that imports it without
# pinning it in requirements.txt is caught the same way.
_DJANGO_DEP_PIP = {
    "django": "Django",
    "rest_framework": "djangorestframework",
    "corsheaders": "django-cors-headers",
}

# Files created once by ``django-admin startproject`` and never touched by an
# incremental change -- their presence marks a *fresh project* scaffold rather
# than an edit to an existing one. Used to gate the greenfield-only plan check
# so a plain PLAN_CHANGE (which only lists the touched files) is never rejected.
_GREENFIELD_MARKER = "manage.py"

# Where a plain-pytest run can be told which settings module to use. At plan
# time (paths only) we can only confirm one of these files is planned; the
# content-level DJANGO_SETTINGS_MODULE check lives in validate_scaffold.
_PYTEST_CONFIG_FILES = ("pytest.ini", "conftest.py", "setup.cfg", "tox.ini",
                        "pyproject.toml")


class DjangoSkill(Skill):
    name = "django"
    role = "backend"
    aliases = ("Django", "DRF", "Django REST Framework")
    description = "Django / Django REST Framework Python backend."

    def detect(self, goal: str) -> float:
        g = goal or ""
        if _DJANGO_RE.search(g) or _DRF_RE.search(g):
            return 0.95
        return 0.0

    def scaffold_system_prompt(self) -> str:
        return (
            "BACKEND -- Django project\n"
            "- Project layout: <project>/manage.py at root, <project>/<project>/"
            "settings.py + urls.py + wsgi.py for the site, one app folder per "
            "feature with models.py, views.py, urls.py, apps.py.\n"
            "- settings.py must populate INSTALLED_APPS, DATABASES (sqlite3 "
            "by default), MIDDLEWARE, ROOT_URLCONF, TEMPLATES. Register EVERY "
            "app you create in INSTALLED_APPS (e.g. add \"blog\" or "
            "\"blog.apps.BlogConfig\") -- an unregistered app's models, "
            "migrations and admin never load.\n"
            "- MODELS are CLASSES subclassing `django.db.models.Model` in each "
            "app's models.py (`class Post(models.Model): title = "
            "models.CharField(max_length=200)`), one file per app -- never a "
            "bare dict or module-level function.\n"
            "- urls.py at project level uses `path(...)` and `include(...)` to "
            "MOUNT each app's urlpatterns (`path(\"blog/\", "
            "include(\"blog.urls\"))`); define per-app routes in that app's own "
            "urls.py. A view is only reachable once its app's urls are wired in "
            "via include().\n"
            "- requirements.txt must pin `Django`, plus `djangorestframework` "
            "when the goal uses DRF/REST and `django-cors-headers` when CORS is "
            "needed. NOTE the import name differs from the pip name: you `import "
            "rest_framework` but pip-install `djangorestframework`, and `import "
            "corsheaders` but pip-install `django-cors-headers` -- pin the PIP "
            "name and add the app to INSTALLED_APPS.\n"
            "- Views: class-based (`generic.ListView` / `APIView`) preferred "
            "for CRUD; function views fine for simple cases.\n"
            "- Provide an initial migration file when generating models.\n"
            "- TESTS: the verifier runs PLAIN `pytest`, which cannot import "
            "Django settings on its own -- add a pytest runner config that sets "
            "DJANGO_SETTINGS_MODULE (a pytest.ini / setup.cfg / pyproject.toml "
            "with `DJANGO_SETTINGS_MODULE = <project>.settings`, or a conftest.py "
            "that does `os.environ.setdefault(\"DJANGO_SETTINGS_MODULE\", ...)` "
            "then `django.setup()`). Without it every test errors out at "
            "collection with ImproperlyConfigured."
        )

    def plan_system_prompt(self) -> str:
        return (
            "When modifying a Django project:\n"
            "- New routes register in the relevant app's urls.py and are "
            "included from the project urls.py via `include(...)`.\n"
            "- New apps must be added to INSTALLED_APPS in settings.py.\n"
            "- Schema changes go through `python manage.py makemigrations` -- "
            "include the generated migration file in the plan.\n"
            "- Use the ORM (`Model.objects...`); avoid raw SQL unless the "
            "existing code already uses it.\n"
            "- A fresh Django project must plan manage.py, a settings.py and a "
            "project-level urls.py. Whenever you plan pytest-style tests "
            "(tests/*.py or test_*.py), also plan a pytest runner config "
            "(pytest.ini / setup.cfg / pyproject.toml / conftest.py) that sets "
            "DJANGO_SETTINGS_MODULE, or plain pytest fails to collect them."
        )

    def validate_scaffold(self, diffs: List[Dict[str, Any]],
                          goal: str = "") -> Optional[SkillVerdict]:
        paths = file_paths(diffs)
        if not paths:
            return None
        if not any(p.endswith(".py") for p in paths):
            return SkillVerdict(
                passed=False, confidence=0.9,
                rationale=("Django skill: scaffold has no Python files."),
            )
        has_manage = any(p.endswith("manage.py") for p in paths)
        has_settings = any(p.endswith("settings.py") for p in paths)
        if not (has_manage and has_settings):
            return SkillVerdict(
                passed=False, confidence=0.85,
                rationale=("Django skill: scaffold is missing manage.py "
                           "and/or settings.py. A Django project needs both."),
            )
        if file_with_content(diffs, "django") is None:
            return SkillVerdict(
                passed=False, confidence=0.85,
                rationale=("Django skill: no generated file imports or "
                           "references `django`."),
            )
        # A dep imported but never pinned ships broken at install/import. Note
        # the import name (rest_framework, corsheaders) differs from the pip
        # name (djangorestframework, django-cors-headers).
        missing = undeclared_python_deps(diffs, _DJANGO_DEP_PIP)
        if missing:
            names = ", ".join(sorted({m["pip"] for m in missing}))
            return SkillVerdict(
                passed=False, confidence=0.8,
                rationale=(f"Django skill: {names} is imported but not pinned in "
                           "requirements.txt. Add it (the pip name can differ "
                           "from the import name: rest_framework->"
                           "djangorestframework, corsheaders->"
                           "django-cors-headers) and register it in "
                           "INSTALLED_APPS."),
            )
        # Advisory: pytest-style tests that plain pytest cannot collect because
        # no config sets DJANGO_SETTINGS_MODULE (warning -- doesn't regenerate).
        if has_python_test_file(paths) and not self._has_settings_module(diffs):
            return SkillVerdict(
                passed=False, confidence=0.7, severity="warning",
                rationale=("Django skill: pytest-style tests were generated but "
                           "no config sets DJANGO_SETTINGS_MODULE. Plain pytest "
                           "then fails at collection with ImproperlyConfigured. "
                           "Add a pytest.ini/setup.cfg/pyproject.toml with "
                           "`DJANGO_SETTINGS_MODULE = <project>.settings`, or a "
                           "conftest.py that sets it and calls django.setup()."),
            )
        return None

    def validate_plan(self, diffs: List[Dict[str, Any]],
                      goal: str = "") -> Optional[SkillVerdict]:
        # Paths-only at plan time. Only fire on a *greenfield* Django scaffold
        # (manage.py present -- boilerplate that is never touched by an
        # incremental change), so a plain PLAN_CHANGE that edits a subset of an
        # existing project is never fatally rejected.
        paths = [p.replace("\\", "/") for p in file_paths(diffs)]
        if not paths:
            return None
        bases = {p.rsplit("/", 1)[-1] for p in paths}
        if _GREENFIELD_MARKER not in bases:
            return None
        # The other two files every Django site must have.
        missing_core = [f for f in ("settings.py", "urls.py") if f not in bases]
        if missing_core:
            return SkillVerdict(
                passed=False, confidence=0.8,
                rationale=("Django skill: a Django project scaffold (manage.py "
                           "planned) is missing " + " and ".join(missing_core)
                           + ". A project needs manage.py, a settings.py, and a "
                           "project-level urls.py wiring apps via include()."),
            )
        # plain pytest cannot collect Django tests without DJANGO_SETTINGS_MODULE
        # (ImproperlyConfigured), and the Swarm verifier runs plain pytest -- so
        # a scaffold that plans pytest-style tests must also plan a runner config
        # that can carry it.
        if has_python_test_file(paths) and not (
                set(_PYTEST_CONFIG_FILES) & bases):
            return SkillVerdict(
                passed=False, confidence=0.8,
                rationale=("Django skill: pytest-style tests are planned but no "
                           "pytest runner config (pytest.ini / conftest.py / "
                           "setup.cfg / pyproject.toml) is. Plain pytest can't "
                           "collect Django tests and errors with "
                           "ImproperlyConfigured. Add one that sets "
                           "DJANGO_SETTINGS_MODULE = <project>.settings."),
            )
        return None

    def scaffold_warnings(self, diffs: List[Dict[str, Any]],
                          goal: str = "") -> List[SkillVerdict]:
        paths = file_paths(diffs)
        if not paths or not any(p.endswith(".py") for p in paths):
            return []
        # Django ships test discovery via `manage.py test` on app/tests.py
        # or tests/ directories; honour both conventions.
        if has_python_test_file(paths) or any(
            p.endswith("tests.py") for p in paths
        ):
            return []
        return [SkillVerdict(
            passed=False, confidence=0.7, severity="warning",
            rationale=("Django skill: no test file generated. Add "
                       "<app>/tests.py or tests/test_<app>.py exercising "
                       "the views and models with django.test.TestCase."),
        )]

    @staticmethod
    def _has_settings_module(diffs: List[Dict[str, Any]]) -> bool:
        """True if any diff body points pytest at a Django settings module.

        Best-effort content check (validate_scaffold sees patch/content text):
        a pytest.ini/setup.cfg/pyproject.toml carrying ``DJANGO_SETTINGS_MODULE``
        or a conftest.py calling ``settings.configure(...)`` both count.
        """
        return (file_with_content(diffs, "DJANGO_SETTINGS_MODULE") is not None
                or file_with_content(diffs, "settings.configure") is not None)


__all__ = ["DjangoSkill"]
