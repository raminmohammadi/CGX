"""Django backend skill."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from skills.base import (
    Skill, SkillVerdict, file_paths, file_with_content, has_python_test_file,
)


_DJANGO_RE = re.compile(r"\bdjango\b", re.IGNORECASE)
_DRF_RE = re.compile(r"\bdjango\s*rest\s*framework\b|\bdrf\b", re.IGNORECASE)


class DjangoSkill(Skill):
    name = "django"
    role = "backend"
    aliases = ("Django", "DRF", "Django REST Framework")

    def detect(self, goal: str) -> float:
        g = goal or ""
        if _DJANGO_RE.search(g) or _DRF_RE.search(g):
            return 0.95
        return 0.0

    def scaffold_system_prompt(self) -> str:
        return (
            "BACKEND — Django project\n"
            "- Project layout: <project>/manage.py at root, <project>/<project>/"
            "settings.py + urls.py + wsgi.py for the site, one app folder per "
            "feature with models.py, views.py, urls.py, apps.py.\n"
            "- settings.py must populate INSTALLED_APPS, DATABASES (sqlite3 "
            "by default), MIDDLEWARE, ROOT_URLCONF, TEMPLATES.\n"
            "- urls.py at project level uses `path(...)` and `include(...)` "
            "to mount per-app urlpatterns.\n"
            "- requirements.txt must pin `django` (and `djangorestframework` "
            "when the goal mentions DRF or REST APIs).\n"
            "- Views: class-based (`generic.ListView` / `APIView`) preferred "
            "for CRUD; function views fine for simple cases.\n"
            "- Provide an initial migration file when generating models."
        )

    def plan_system_prompt(self) -> str:
        return (
            "When modifying a Django project:\n"
            "- New routes register in the relevant app's urls.py and are "
            "included from the project urls.py.\n"
            "- Schema changes go through `python manage.py makemigrations` — "
            "include the generated migration file in the plan.\n"
            "- Use the ORM (`Model.objects...`); avoid raw SQL unless the "
            "existing code already uses it."
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


__all__ = ["DjangoSkill"]
