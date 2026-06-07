

"""FastAPI backend skill."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from skills.base import (
    Skill, SkillVerdict, file_paths, file_with_content, has_python_test_file,
)


_FASTAPI_RE = re.compile(r"\bfast\s*api\b", re.IGNORECASE)


class FastAPISkill(Skill):
    name = "fastapi"
    role = "backend"
    aliases = ("FastAPI", "Fast API")

    def detect(self, goal: str) -> float:
        g = goal or ""
        if _FASTAPI_RE.search(g):
            return 0.95
        # "python backend" alone is ambiguous; abstain.
        return 0.0

    def scaffold_system_prompt(self) -> str:
        return (
            "BACKEND -- FastAPI service\n"
            "- Single application module at backend/main.py (or app/main.py) "
            "creating `app = FastAPI()` and exposing routes with "
            "`@app.get`/`@app.post`.\n"
            "- Use pydantic models in backend/models.py for request/response "
            "schemas when the endpoint accepts a JSON body.\n"
            "- requirements.txt must pin `fastapi` and `uvicorn[standard]` "
            "with compatible versions.\n"
            "- Add CORSMiddleware allowing the frontend origin when this "
            "service is paired with a frontend skill (React/Vue/Next.js).\n"
            "- Provide a `if __name__ == \"__main__\":` block that runs "
            "`uvicorn.run(app, host=\"0.0.0.0\", port=8000)` so the file is "
            "runnable directly.\n"
            "- Tests under tests/test_*.py using `fastapi.testclient.TestClient`."
        )

    def plan_system_prompt(self) -> str:
        return (
            "When modifying a FastAPI project:\n"
            "- New endpoints attach to the existing `app` instance via "
            "decorators; don't create a parallel FastAPI() instance.\n"
            "- Add pydantic schemas to the existing models module rather "
            "than inlining dicts."
        )

    def validate_scaffold(self, diffs: List[Dict[str, Any]],
                          goal: str = "") -> Optional[SkillVerdict]:
        paths = file_paths(diffs)
        if not paths:
            return None
        if not any(p.endswith(".py") for p in paths):
            return SkillVerdict(
                passed=False, confidence=0.9,
                rationale=("FastAPI skill: scaffold has no Python files. "
                           "FastAPI requires .py modules."),
            )
        if file_with_content(diffs, "fastapi") is None:
            return SkillVerdict(
                passed=False, confidence=0.85,
                rationale=("FastAPI skill: no generated file imports or "
                           "references `fastapi`. Add backend/main.py with "
                           "`from fastapi import FastAPI`."),
            )
        has_req = any(p.endswith("requirements.txt")
                      or p.endswith("pyproject.toml")
                      for p in paths)
        if not has_req:
            return SkillVerdict(
                passed=False, confidence=0.8,
                rationale=("FastAPI skill: scaffold is missing "
                           "requirements.txt (or pyproject.toml) pinning "
                           "`fastapi` and `uvicorn`."),
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
            rationale=("FastAPI skill: no test file generated. Add a "
                       "tests/test_main.py using "
                       "`fastapi.testclient.TestClient` to exercise the "
                       "exposed routes."),
        )]


__all__ = ["FastAPISkill"]
