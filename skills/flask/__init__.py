

"""Flask backend skill."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from skills._structural import (
    any_body_matches,
    app_backimports,
    blueprint_prefix_collisions,
    undeclared_python_deps,
    undefined_model_refs,
)
from skills.base import Skill, SkillVerdict, file_paths, has_python_test_file

_FLASK_RE = re.compile(r"\bflask\b", re.IGNORECASE)

# Flask extensions whose import name differs from the pip package name, used to
# flag a use that was never declared (a phantom dependency that ships broken).
_FLASK_EXT_PIP = {
    "flask_sqlalchemy": "flask-sqlalchemy",
    "flask_cors": "flask-cors",
    "flask_migrate": "flask-migrate",
    "flask_login": "flask-login",
    "flask_jwt_extended": "flask-jwt-extended",
    "flask_marshmallow": "flask-marshmallow",
}

# Persistence intent in the goal -> the plan should include an extensions.py +
# models.py so routes never import db/app back from the app module.
_DB_INTENT = ("db", "database", "model", "sqlalchemy", "persist", "member",
              "order", "payment", "user", "account", "store", "record")


class FlaskSkill(Skill):
    name = "flask"
    role = "backend"
    aliases = ("Flask",)
    description = "Flask Python backend service (WSGI, Blueprints)."

    def detect(self, goal: str) -> float:
        if _FLASK_RE.search(goal or ""):
            return 0.95
        return 0.0

    def scaffold_system_prompt(self) -> str:
        return (
            "BACKEND -- Flask service\n"
            "STRUCTURE (avoid circular imports -- the #1 Flask codegen failure):\n"
            "- Put extension objects in their OWN module `backend/extensions.py`, "
            "constructed WITHOUT an app: `db = SQLAlchemy()` (and `cors = CORS()`, "
            "etc.).\n"
            "- Build the app in `backend/app.py` with an application factory: "
            "`def create_app(): app = Flask(__name__); db.init_app(app); "
            "register blueprints; return app` (a module-level `app = create_app()` "
            "for `flask run` is fine).\n"
            "- Group routes as Blueprints in their own modules "
            "(`bp = Blueprint(\"members\", __name__)`, `@bp.route(...)`); app.py "
            "imports the blueprints and calls `app.register_blueprint(bp)`.\n"
            "- A route/model/db module MUST NEVER do `from backend.app import app` "
            "or `from backend.app import db` -- that is a circular import that "
            "breaks even pytest collection. Import `db` from `backend.extensions`, "
            "and use `flask.current_app` if you need the app inside a request.\n"
            "MODELS are CLASSES: `class Member(db.Model): ...` in backend/models.py, "
            "importing `db` from backend.extensions. Declare each model under the "
            "plan's contracts.schemas, NEVER under contracts.functions.\n"
            "DEPENDENCIES: any use of flask-sqlalchemy "
            "(`from flask_sqlalchemy import SQLAlchemy`) or flask-cors "
            "(`from flask_cors import CORS`) MUST be pinned in requirements.txt AND "
            "listed in the plan's third_party_dependencies. Import names "
            "(flask_sqlalchemy, flask_cors) differ from pip names "
            "(flask-sqlalchemy, flask-cors).\n"
            "- Return JSON via `flask.jsonify(...)`. requirements.txt pins `flask` "
            "(plus the extensions above whenever they are imported).\n"
            "- Provide `if __name__ == \"__main__\": app.run(host=\"0.0.0.0\", "
            "port=5000, debug=False)`.\n"
            "- Tests under tests/test_*.py: build the app via `create_app()`, wrap "
            "DB setup in `with app.app_context(): db.create_all()`, and drive "
            "routes with `app.test_client()`."
        )

    def plan_system_prompt(self) -> str:
        return (
            "When planning or modifying a Flask project:\n"
            "- Plan `backend/extensions.py` (holds `db = SQLAlchemy()` etc.), "
            "`backend/app.py` (a create_app factory calling db.init_app and "
            "registering blueprints), `backend/models.py` (model CLASSES importing "
            "db from extensions), and one blueprint module per route group.\n"
            "- Route/model/db modules MUST NOT import `app` or `db` from "
            "backend.app -- that is a circular import. Import db from "
            "backend.extensions.\n"
            "- Declare model classes under contracts.schemas (not functions), and "
            "list flask / flask-sqlalchemy / flask-cors in "
            "third_party_dependencies whenever they are used.\n"
            "- Attach new routes to a Blueprint (or the existing app); never create "
            "a parallel Flask() instance."
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
        # A REAL application instance, not merely the word "flask" somewhere.
        if not any_body_matches(
                diffs, r"Flask\(\s*(?:__name__|import_name)|def\s+create_app",
                only_ext=(".py",)):
            return SkillVerdict(
                passed=False, confidence=0.85,
                rationale=("Flask skill: no real Flask application instance. Add "
                           "backend/app.py with `app = Flask(__name__)` or a "
                           "`create_app()` factory."),
            )
        # THE case-study failure: a leaf module importing app/db back from the
        # app entrypoint = circular import that breaks even test collection.
        cyc = app_backimports(diffs)
        if cyc:
            worst = cyc[0]
            return SkillVerdict(
                passed=False, confidence=0.9,
                rationale=(f"Flask skill: circular import -- {worst['file']} does "
                           f"`from ...{worst['from_module']} import "
                           f"{worst['imported']}`. A route/model module must not "
                           "import app/db back from the app entrypoint. Move `db` "
                           "to backend/extensions.py, import it from there, and "
                           "register blueprints in create_app()."),
            )
        # A model/table referenced by routes/tests but defined by NO db.Model
        # (the coffee-shop failure: raw `INSERT INTO orders` + `Order.query`
        # with only a Member model). The fix is ADDITIVE and lives in models.py
        # -- so carry the definer + a column-precise directive, not a bare fail,
        # so verify regenerates models.py to DEFINE it rather than stripping the
        # reference. Columns are mined from the INSERT when available.
        undef = undefined_model_refs(diffs)
        if undef:
            specs = []
            for u in undef:
                cols = ", ".join(u["columns"]) if u["columns"] else \
                    "an integer primary key plus the fields it is used with"
                tbl = f" (table '{u['table']}')" if u.get("table") else ""
                specs.append(f"`class {u['name']}(db.Model)`{tbl} with columns: "
                             f"{cols}")
            define_in = undef[0]["define_in"]
            directive = (
                "This module MUST DEFINE these referenced-but-missing models "
                "(import db from backend.extensions; give each a primary key and "
                "db.Column fields matching how it is used; ADD them, do not "
                "remove any reference): " + "; ".join(specs) + ". A raw-SQL table "
                "must have a matching model so db.create_all() creates it.")
            names = ", ".join(u["name"] for u in undef)
            return SkillVerdict(
                passed=False, confidence=0.9,
                rationale=(f"Flask skill: model/table referenced but never "
                           f"defined ({names}). Define it in {define_in} so "
                           "db.create_all() creates the table -- otherwise every "
                           "query raises OperationalError: no such table. "
                           + directive),
                regen_file=define_in, directive=directive,
            )
        # A blueprint route repeating its own url_prefix -> /api/api/... (every
        # request 404s). Deterministically fixable, so this is a hard verdict
        # whose repair the skill's repair_scaffold applies without a model call.
        collisions = blueprint_prefix_collisions(diffs)
        if collisions:
            worst = collisions[0]
            return SkillVerdict(
                passed=False, confidence=0.85,
                rationale=(f"Flask skill: route '{worst['route']}' already starts "
                           f"with the blueprint's url_prefix '{worst['prefix']}', "
                           f"so the live path is '{worst['prefix']}{worst['route']}'"
                           " (every request 404s). Drop the prefix from the route "
                           "decorator (the url_prefix already namespaces it)."),
                regen_file=worst["file"],
            )
        # Extensions used but never declared -> ships broken at install/import.
        missing = undeclared_python_deps(diffs, _FLASK_EXT_PIP)
        if missing:
            names = ", ".join(sorted({m["pip"] for m in missing}))
            return SkillVerdict(
                passed=False, confidence=0.8,
                rationale=(f"Flask skill: {names} is imported but not pinned in "
                           "requirements.txt. Add it (note pip name differs from "
                           "the import name) and list it in "
                           "third_party_dependencies."),
            )
        if not any(p.endswith("requirements.txt")
                   or p.endswith("pyproject.toml") for p in paths):
            return SkillVerdict(
                passed=False, confidence=0.8,
                rationale=("Flask skill: scaffold is missing requirements.txt "
                           "(or pyproject.toml) pinning `flask`."),
            )
        return None

    def validate_plan(self, diffs: List[Dict[str, Any]],
                      goal: str = "") -> Optional[SkillVerdict]:
        # Paths-only at plan time. When the goal needs persistence, steer to the
        # extensions.py + models.py layout so routes never import db/app back.
        paths = [p.replace("\\", "/") for p in file_paths(diffs)]
        if not paths:
            return None
        bases = {p.rsplit("/", 1)[-1] for p in paths}
        lg = (goal or "").lower()
        if any(w in lg for w in _DB_INTENT) \
                and "extensions.py" not in bases and "models.py" not in bases:
            return SkillVerdict(
                passed=False, confidence=0.7,
                rationale=("Flask skill: a Flask app with persistence must plan a "
                           "backend/extensions.py (holding `db = SQLAlchemy()`) and "
                           "a backend/models.py (model classes importing db from "
                           "extensions), so routes never import db/app back from "
                           "the app module (a circular import). Add them."),
            )
        return None

    def repair_scaffold(self, files: List[Dict[str, Any]]) -> Dict[str, str]:
        """Deterministically fix the double-prefix bug -- no model call.

        For every route that repeats its blueprint's ``url_prefix`` (e.g.
        ``@bp.route("/api/orders")`` under ``url_prefix="/api"``), strip the
        prefix from the decorator so the live path is ``/api/orders`` again. This
        is certain and mechanical, so it runs before any LLM repair round.
        """
        collisions = blueprint_prefix_collisions(files)
        if not collisions:
            return {}
        bodies = {(d.get("file") or d.get("path")): (d.get("content")
                  or d.get("patch") or "")
                  for d in files if isinstance(d, dict)}
        by_file: Dict[str, List[Dict[str, str]]] = {}
        for c in collisions:
            by_file.setdefault(c["file"], []).append(c)
        out: Dict[str, str] = {}
        for path, cols in by_file.items():
            body = bodies.get(path)
            if not isinstance(body, str) or not body:
                continue
            new = body
            for c in cols:
                prefix, route = c["prefix"], c["route"]
                # The route with the leading prefix stripped (keep a leading /).
                stripped = route[len(prefix):] or "/"
                if not stripped.startswith("/"):
                    stripped = "/" + stripped
                # Replace only inside the @bp.route("...") decorator literal, so
                # an identical string elsewhere is never touched.
                for q in ("'", '"'):
                    new = new.replace(f".route({q}{route}{q}",
                                      f".route({q}{stripped}{q}")
            if new != body:
                out[path] = new
        return out

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
                       "tests/test_app.py that builds the app via create_app() "
                       "and uses `app.test_client()` to exercise the routes."),
        )]


__all__ = ["FlaskSkill"]
