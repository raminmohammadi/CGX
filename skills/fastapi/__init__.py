

"""FastAPI backend skill."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from skills._structural import (
    any_body_matches,
    app_backimports,
    undeclared_python_deps,
)
from skills.base import (
    Skill,
    SkillVerdict,
    file_paths,
    has_python_test_file,
)

_FASTAPI_RE = re.compile(r"\bfast\s*api\b", re.IGNORECASE)

# Application-entrypoint basenames for a FastAPI project: the module that
# constructs `app = FastAPI()` and wires the routers via include_router.
_APP_ENTRYPOINTS = ("main.py", "app.py", "server.py", "asgi.py")

# Third-party imports whose absence from requirements ships a broken service.
# NOTE: `pydantic` is deliberately NOT here -- it is a hard (transitive)
# dependency of `fastapi`, so importing it while only `fastapi` is pinned still
# installs & imports fine; a fatal reject there would be a false positive.
# `sqlalchemy` is not pulled in by fastapi, so an undeclared use genuinely
# breaks at install/import. Declaring pydantic is still taught in the prompt.
_FASTAPI_DEP_PIP = {"sqlalchemy": "sqlalchemy"}


def _is_router_module(path: str) -> bool:
    """True when ``path`` looks like an APIRouter route-group module.

    High precision: fires on a module under a ``routers/`` (or ``routes/``)
    package, or a basename of ``router(s).py`` / ``route(s).py`` /
    ``*_router.py`` / ``*_routes.py``. Used paths-only at plan time to require
    an app entrypoint (``main.py``) that would ``include_router`` them.
    """
    p = path.replace("\\", "/")
    name = p.rsplit("/", 1)[-1]
    if not name.endswith(".py"):
        return False
    parents = p.split("/")[:-1]
    if any(seg in ("routers", "routes") for seg in parents):
        return True
    stem = name[:-3]
    if stem in ("router", "routers", "route", "routes"):
        return True
    return stem.endswith(("_router", "_routers", "_route", "_routes"))


class FastAPISkill(Skill):
    name = "fastapi"
    role = "backend"
    aliases = ("FastAPI", "Fast API")
    description = "FastAPI Python backend service with pydantic models and uvicorn."

    def detect(self, goal: str) -> float:
        g = goal or ""
        if _FASTAPI_RE.search(g):
            return 0.95
        # "python backend" alone is ambiguous; abstain.
        return 0.0

    def scaffold_system_prompt(self) -> str:
        return (
            "BACKEND -- FastAPI service\n"
            "STRUCTURE (split the app from its routers -- the circular import "
            "this avoids is the #1 FastAPI codegen failure):\n"
            "- Create the FastAPI app in `backend/main.py` (or app/main.py): "
            "`app = FastAPI()`, then wire each route group with "
            "`app.include_router(...)`. main.py imports the routers; the "
            "routers NEVER import main.\n"
            "- Group routes as `APIRouter` instances in their OWN modules "
            "(`router = APIRouter(prefix=\"/items\", tags=[\"items\"])`, "
            "`@router.get(...)`); main.py does "
            "`from backend.routers.items import router as items_router` then "
            "`app.include_router(items_router)`.\n"
            "- A router/model module MUST NEVER do `from backend.main import "
            "app` (or import `app`/`engine`/`db` back from main) -- that is a "
            "circular import that breaks even pytest collection. Put shared DB "
            "objects (`engine`, `SessionLocal`, `Base`, `get_db`) in their OWN "
            "module `backend/db.py` (or database.py), import them from there, "
            "and inject the session into routes with "
            "`db: Session = Depends(get_db)`.\n"
            "- Request/response schemas are pydantic models in "
            "`backend/models.py` (or schemas.py). SQLAlchemy ORM model CLASSES "
            "live with the declarative `Base` in db.py/models.py, importing "
            "`Base` from db.py -- not from main.\n"
            "DEPENDENCIES: requirements.txt (or pyproject.toml) MUST pin "
            "`fastapi`, `uvicorn[standard]` and `pydantic`; add `sqlalchemy` "
            "whenever it is imported. Any third-party import used in the code "
            "must be declared or the service ships broken at install/import.\n"
            "- Add CORSMiddleware allowing the frontend origin when this "
            "service is paired with a frontend skill (React/Vue/Next.js).\n"
            "- Provide a `if __name__ == \"__main__\":` block in main.py that "
            "runs `uvicorn.run(app, host=\"0.0.0.0\", port=8000)` so the file "
            "is runnable directly.\n"
            "- Tests under tests/test_*.py using "
            "`fastapi.testclient.TestClient(app)`."
        )

    def plan_system_prompt(self) -> str:
        return (
            "When planning or modifying a FastAPI project:\n"
            "- Plan the FastAPI app in `backend/main.py` (`app = FastAPI()` "
            "calling `app.include_router(...)`), one `APIRouter` module per "
            "route group under backend/routers/, pydantic schemas in "
            "backend/models.py, and shared DB objects "
            "(engine/SessionLocal/Base/get_db) in backend/db.py.\n"
            "- Router/model/db modules MUST NOT import `app` back from "
            "backend.main -- that is a circular import. main.py imports the "
            "routers, never the reverse; get shared deps from backend/db.py "
            "and inject them with `Depends(get_db)`.\n"
            "- New endpoints attach to an existing APIRouter (or the app) via "
            "decorators; never create a parallel FastAPI() instance.\n"
            "- Declare pydantic schema classes under contracts.schemas (not "
            "functions), and list fastapi / uvicorn / pydantic (and sqlalchemy "
            "when used) in third_party_dependencies whenever they are used."
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
        # A REAL FastAPI application instance, not merely the word "fastapi"
        # somewhere (an import line alone never constructs the app).
        if not any_body_matches(diffs, r"FastAPI\(", only_ext=(".py",)):
            return SkillVerdict(
                passed=False, confidence=0.85,
                rationale=("FastAPI skill: no real FastAPI application "
                           "instance. Add backend/main.py with "
                           "`app = FastAPI()` and wire routes with "
                           "`app.include_router(...)`."),
            )
        # THE structural failure: a router/model module importing app (or
        # engine/db) back from the app entrypoint = circular import that
        # breaks even pytest collection. Exclude test modules: a test doing
        # `from backend.main import app` (the canonical TestClient pattern) is
        # a leaf nothing imports back, so it is not a cycle -- filtering it
        # keeps this verdict high-precision.
        cyc = [c for c in app_backimports(diffs)
               if not has_python_test_file([c["file"]])
               and c["file"].rsplit("/", 1)[-1] != "conftest.py"]
        if cyc:
            worst = cyc[0]
            return SkillVerdict(
                passed=False, confidence=0.9,
                rationale=(f"FastAPI skill: circular import -- {worst['file']} "
                           f"does `from ...{worst['from_module']} import "
                           f"{worst['imported']}`. A router/model module must "
                           "not import app/engine/db back from the app "
                           "entrypoint (main.py). main.py should "
                           "include_router the routers; move shared DB objects "
                           "to backend/db.py and import them from there."),
            )
        # A third-party import used but never declared -> ships broken.
        missing = undeclared_python_deps(diffs, _FASTAPI_DEP_PIP)
        if missing:
            names = ", ".join(sorted({m["pip"] for m in missing}))
            return SkillVerdict(
                passed=False, confidence=0.8,
                rationale=(f"FastAPI skill: {names} is imported but not pinned "
                           "in requirements.txt (or pyproject.toml). Add it and "
                           "list it in third_party_dependencies."),
            )
        has_req = any(p.endswith("requirements.txt")
                      or p.endswith("pyproject.toml")
                      for p in paths)
        if not has_req:
            return SkillVerdict(
                passed=False, confidence=0.8,
                rationale=("FastAPI skill: scaffold is missing "
                           "requirements.txt (or pyproject.toml) pinning "
                           "`fastapi`, `uvicorn` and `pydantic`."),
            )
        return None

    def validate_plan(self, diffs: List[Dict[str, Any]],
                      goal: str = "") -> Optional[SkillVerdict]:
        # Paths-only at plan time. When APIRouter modules are planned, there
        # MUST be an app entrypoint (main.py) to include_router them; a bag of
        # routers with nothing mounting them is an unrunnable app.
        paths = [p.replace("\\", "/") for p in file_paths(diffs)]
        if not paths:
            return None
        py = [p for p in paths if p.endswith(".py")]
        if not py:
            return None
        bases = {p.rsplit("/", 1)[-1] for p in py}
        has_entry = any(b in _APP_ENTRYPOINTS for b in bases)
        if not has_entry and any(_is_router_module(p) for p in py):
            return SkillVerdict(
                passed=False, confidence=0.7,
                rationale=("FastAPI skill: APIRouter modules are planned but "
                           "there is no app entrypoint (backend/main.py) to "
                           "mount them. Add a main.py that creates "
                           "`app = FastAPI()` and calls "
                           "`app.include_router(...)` for each router."),
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
                       "`fastapi.testclient.TestClient(app)` to exercise the "
                       "exposed routes."),
        )]


__all__ = ["FastAPISkill"]
