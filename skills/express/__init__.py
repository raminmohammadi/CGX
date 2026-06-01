"""Express.js (Node) backend skill."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from skills.base import (
    Skill, SkillVerdict, file_paths, file_with_content, has_js_test_file,
)


_EXPRESS_RE = re.compile(r"\bexpress(?:\.?js)?\b", re.IGNORECASE)


class ExpressSkill(Skill):
    name = "express"
    role = "backend"
    aliases = ("Express", "Express.js", "ExpressJS")

    def detect(self, goal: str) -> float:
        if _EXPRESS_RE.search(goal or ""):
            return 0.95
        return 0.0

    def scaffold_system_prompt(self) -> str:
        return (
            "BACKEND — Express.js (Node) service\n"
            "- Single entry module at server/index.js (or src/server.js) "
            "creating `const app = express()` and listening on `process.env."
            "PORT || 3000`.\n"
            "- Routes attach via `app.get('/...', handler)` / "
            "`app.post(...)`; group related routes under `express.Router()` "
            "in routes/<name>.js when there are more than a couple.\n"
            "- Use `express.json()` middleware for JSON bodies and `cors()` "
            "when this service is paired with a separate frontend skill.\n"
            "- package.json dependencies must include `express` (^4); "
            "`cors` when relevant. Scripts: `start` → `node server/index.js`, "
            "`dev` → `nodemon server/index.js` (when devDependency added).\n"
            "- Do NOT mix Python files into this service."
        )

    def plan_system_prompt(self) -> str:
        return (
            "When modifying an Express project:\n"
            "- Attach new routes to the existing `app` or a Router; don't "
            "create a parallel express() instance.\n"
            "- Keep middleware order: body parsers and CORS before route "
            "handlers."
        )

    def validate_scaffold(self, diffs: List[Dict[str, Any]],
                          goal: str = "") -> Optional[SkillVerdict]:
        paths = file_paths(diffs)
        if not paths:
            return None
        if not any(p.endswith((".js", ".mjs", ".cjs", ".ts")) for p in paths):
            return SkillVerdict(
                passed=False, confidence=0.9,
                rationale=("Express skill: scaffold has no Node source "
                           "files (.js/.ts)."),
            )
        if file_with_content(diffs, "express") is None:
            return SkillVerdict(
                passed=False, confidence=0.85,
                rationale=("Express skill: no generated file imports or "
                           "requires `express`."),
            )
        if not any(p.endswith("package.json") for p in paths):
            return SkillVerdict(
                passed=False, confidence=0.85,
                rationale=("Express skill: scaffold is missing package.json "
                           "with `express` dependency."),
            )
        return None

    def scaffold_warnings(self, diffs: List[Dict[str, Any]],
                          goal: str = "") -> List[SkillVerdict]:
        paths = file_paths(diffs)
        if not paths or not any(p.endswith((".js", ".mjs", ".cjs", ".ts"))
                                for p in paths):
            return []
        if has_js_test_file(paths):
            return []
        return [SkillVerdict(
            passed=False, confidence=0.7, severity="warning",
            rationale=("Express skill: no test file generated. Add a "
                       "tests/app.test.js using `supertest` to exercise "
                       "the routes."),
        )]


__all__ = ["ExpressSkill"]
