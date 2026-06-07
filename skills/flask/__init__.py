

"""Flask backend skill."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from skills.base import (
    Skill, SkillVerdict, file_paths, file_with_content, has_python_test_file,
)


_FLASK_RE = re.compile(r"\bflask\b", re.IGNORECASE)


class FlaskSkill(Skill):
    name = "flask"
    role = "backend"
    aliases = ("Flask",)

    def detect(self, goal: str) -> float:
        if _FLASK_RE.search(goal or ""):
            return 0.95
        return 0.0

    def scaffold_system_prompt(self) -> str:
        return (
            "BACKEND -- Flask service\n"
            "- Single application module at backend/app.py (or app/__init__.py "
            "if using the application-factory pattern) creating "
            "`app = Flask(__name__)` and exposing routes with "
            "`@app.route(\"/...\", methods=[...])`.\n"
            "- Return JSON via `flask.jsonify(...)` for API endpoints.\n"
            "- requirements.txt must pin `flask` (and `flask-cors` when this "
            "service is paired with a separate frontend skill).\n"
            "- Provide a `if __name__ == \"__main__\":` block that calls "
            "`app.run(host=\"0.0.0.0\", port=5000, debug=False)`.\n"
            "- Tests under tests/test_*.py using `app.test_client()`."
        )

    def plan_system_prompt(self) -> str:
        return (
            "When modifying a Flask project:\n"
            "- Attach new routes to the existing `app` (or blueprint) via "
            "`@app.route` decorators; don't create a parallel Flask() "
            "instance.\n"
            "- Use blueprints for grouping related routes when the file "
            "grows past ~5 endpoints."
        )

    def validate_scaffold(self, diffs: List[Dict[str, Any]],
                          goal: str = "") -> Optional[SkillVerdict]:
        paths = file_paths(diffs)
        if not paths:
            return None
        if not any(p.endswith(".py") for p in paths):
            return SkillVerdict(
                passed=False, confidence=0.9,
                rationale=("Flask skill: scaffold has no Python files. "
                           "Flask requires .py modules."),
            )
        if file_with_content(diffs, "flask") is None:
            return SkillVerdict(
                passed=False, confidence=0.85,
                rationale=("Flask skill: no generated file imports or "
                           "references `flask`. Add backend/app.py with "
                           "`from flask import Flask`."),
            )
        if not any(p.endswith("requirements.txt")
                   or p.endswith("pyproject.toml") for p in paths):
            return SkillVerdict(
                passed=False, confidence=0.8,
                rationale=("Flask skill: scaffold is missing "
                           "requirements.txt (or pyproject.toml) pinning "
                           "`flask`."),
            )
        return None

    def scaffold_warnings(self, diffs: List[Dict[str, Any]],
                          goal: str = "") -> List[SkillVerdict]:
        paths = file_paths(diffs)
        if not paths or not any(p.endswith(".py") for p in paths):
            return []
        if has_python_test_file(paths):
            return []
        return [SkillVerdict(
            passed=False, confidence=0.7, severity="warning",
            rationale=("Flask skill: no test file generated. Add a "
                       "tests/test_app.py using `app.test_client()` to "
                       "exercise the registered routes."),
        )]


__all__ = ["FlaskSkill"]
