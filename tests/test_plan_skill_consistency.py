"""Guards against plan-gate vs skill contradictions (the coffee-shop failure).

The failure: for a "static HTML frontend + Flask backend" goal, ensure_scaffolding
force-injected `frontend/package.json` (any .js -> "node" component), while the
active static_site skill rejects any package.json -- an unwinnable
inject-then-reject loop that failed the session after 3 attempts. These tests
lock in the three fixes: skill-aware scaffolding (veto), narrower node
detection (vanilla .js != node), and the invariant-vs-skill consistency check
that catches the whole class.
"""

from __future__ import annotations

from cgx.session.tasks.swarm_plan import (
    _langs_present,
    ensure_scaffolding,
    ensure_test_coverage,
    normalize_plan,
    ordered_paths,
)
from cgx.session.tasks.swarm_tech_lead import _rejection_stalled
from skills import detect_skills, validate_plan
from skills.static_site import StaticSiteSkill

_COFFEE_GOAL = ("build a coffee shop html website with a python flask backend "
                "for member info and payments")


def _coffee_plan():
    # The SANCTIONED Flask layout: db in extensions.py, model classes in
    # models.py importing db from extensions, a create_app app.py, and a
    # blueprint route module -- no routes<->app cycle.
    return normalize_plan({"goal": _COFFEE_GOAL, "layers": [
        {"name": "frontend", "files": [
            {"path": "frontend/index.html", "description": "home page"},
            {"path": "frontend/style.css", "description": "styles"},
            {"path": "frontend/script.js", "description": "vanilla interactivity"},
        ]},
        {"name": "backend", "files": [
            {"path": "backend/extensions.py", "description": "db = SQLAlchemy()"},
            {"path": "backend/models.py", "description": "model classes",
             "depends_on": ["backend/extensions.py"]},
            {"path": "backend/app.py", "description": "create_app factory",
             "depends_on": ["backend/extensions.py", "backend/models.py"]},
            {"path": "backend/routes/members.py", "description": "members blueprint",
             "depends_on": ["backend/extensions.py"]},
        ]},
    ]})


def _finalize(plan, skills):
    return ensure_scaffolding(ensure_test_coverage(normalize_plan(plan)),
                              skills=skills)


# --- narrower node detection (fix 3b) -----------------------------------

def test_vanilla_js_is_not_node():
    plan = normalize_plan({"goal": "static site", "layers": [{"name": "ui", "files": [
        {"path": "frontend/index.html", "description": "home"},
        {"path": "frontend/script.js", "description": "vanilla script"},
    ]}]})
    assert _langs_present(plan)["node"] is False


def test_jsx_is_node():
    plan = normalize_plan({"goal": "react app", "layers": [{"name": "ui", "files": [
        {"path": "src/App.jsx", "description": "component"}]}]})
    assert _langs_present(plan)["node"] is True


def test_declared_manifest_is_node():
    plan = normalize_plan({"goal": "js", "layers": [{"name": "ui", "files": [
        {"path": "app/main.js", "description": "entry"},
        {"path": "app/package.json", "description": "manifest"}]}]})
    assert _langs_present(plan)["node"] is True


# --- skill veto (fix 3a) -------------------------------------------------

def test_static_site_forbids_framework_files():
    s = StaticSiteSkill()
    assert s.forbids_scaffold_path("frontend/package.json") is True
    assert s.forbids_scaffold_path("frontend/App.jsx") is True
    assert s.forbids_scaffold_path("frontend/index.html") is False
    assert s.forbids_scaffold_path("style.css") is False


def test_scaffolder_respects_skill_veto_even_when_node():
    # A .ts source makes node=True, so ensure_scaffolding WOULD inject a
    # package.json -- but a vetoing skill must filter it out.
    plan = normalize_plan({"goal": "x", "layers": [{"name": "ui", "files": [
        {"path": "frontend/index.html", "description": "home"},
        {"path": "frontend/app.ts", "description": "typed script"}]}]})
    out = ensure_scaffolding(plan, skills=[StaticSiteSkill()])
    paths = {f["path"] for lay in out["layers"] for f in lay["files"]}
    assert not any(p.endswith("package.json") for p in paths)


# --- the coffee-shop regression + consistency invariant (fix 2) ---------

def test_coffee_shop_plan_no_package_json_injected():
    skills = detect_skills(_COFFEE_GOAL)
    names = {getattr(s, "name", "") for s in skills}
    assert "static_site" in names and "flask" in names, names
    plan = _finalize(_coffee_plan(), skills)
    paths = ordered_paths(plan)
    assert not any(p.endswith("package.json") for p in paths), paths
    # requirements.txt still injected for the Python backend.
    assert any(p.endswith("requirements.txt") for p in paths)


def test_scaffolded_plan_passes_active_skill_validators():
    """The invariant that makes this class of bug impossible: the deterministic
    scaffolder's output must never trip an active skill's validate_plan."""
    cases = [
        _COFFEE_GOAL,
        "build a static html landing page website for a bakery",
        "build a python flask rest api for a todo list",
    ]
    plans = [_coffee_plan, _coffee_plan, lambda: normalize_plan(
        {"goal": "flask api", "layers": [{"name": "api", "files": [
            {"path": "app.py", "description": "flask app"},
            {"path": "routes.py", "description": "routes",
             "depends_on": ["app.py"]}]}]})]
    for goal, plan_fn in zip(cases, plans, strict=True):
        skills = detect_skills(goal)
        final = _finalize(plan_fn(), skills)
        verdict = validate_plan(skills, [{"path": p} for p in ordered_paths(final)],
                                goal)
        assert verdict is None or verdict.passed, (goal, verdict)


# --- convergence detection (fix 1) --------------------------------------

def test_rejection_stalled():
    assert _rejection_stalled([]) is False
    assert _rejection_stalled([frozenset({"a"})]) is False
    assert _rejection_stalled([frozenset({"a"}), frozenset({"b"})]) is False
    assert _rejection_stalled([frozenset({"a"}), frozenset({"a"})]) is True
    # progress then repeat: only the final repeat stalls.
    assert _rejection_stalled(
        [frozenset({"a"}), frozenset({"b"}), frozenset({"b"})]) is True


# --- Flask skill hardening (the SWARM_VERIFY case study) -----------------

def test_flask_scaffold_catches_circular_import():
    from skills.flask import FlaskSkill
    diffs = [
        {"path": "backend/app.py",
         "content": "from flask import Flask\nfrom backend.routes.members import bp\napp = Flask(__name__)\n"},
        {"path": "backend/routes/members.py",
         "content": "from backend.app import app, db\nbp = None\n"},
        {"path": "requirements.txt", "content": "flask\n"},
    ]
    v = FlaskSkill().validate_scaffold(diffs, goal="members")
    assert v is not None and not v.passed and "circular import" in v.rationale


def test_flask_scaffold_catches_undeclared_extension():
    from skills.flask import FlaskSkill
    diffs = [
        {"path": "backend/app.py",
         "content": ("from flask import Flask\nfrom flask_sqlalchemy import "
                     "SQLAlchemy\napp = Flask(__name__)\ndb = SQLAlchemy(app)\n")},
        {"path": "requirements.txt", "content": "flask\n"},
    ]
    v = FlaskSkill().validate_scaffold(diffs, goal="members")
    assert v is not None and not v.passed and "flask-sqlalchemy" in v.rationale


def test_flask_sanctioned_layout_passes():
    from skills.flask import FlaskSkill
    diffs = [
        {"path": "backend/extensions.py",
         "content": "from flask_sqlalchemy import SQLAlchemy\ndb = SQLAlchemy()\n"},
        {"path": "backend/models.py",
         "content": "from backend.extensions import db\nclass Member(db.Model):\n    id = db.Column(db.Integer, primary_key=True)\n"},
        {"path": "backend/routes/members.py",
         "content": "from flask import Blueprint, jsonify\nfrom backend.extensions import db\nbp = Blueprint('m', __name__)\n"},
        {"path": "backend/app.py",
         "content": "from flask import Flask\nfrom backend.extensions import db\ndef create_app():\n    app = Flask(__name__)\n    db.init_app(app)\n    return app\n"},
        {"path": "requirements.txt", "content": "flask\nflask-sqlalchemy\n"},
    ]
    assert FlaskSkill().validate_scaffold(diffs, goal="members") is None


def test_flask_plan_steers_missing_extensions():
    from skills import detect_skills as _ds
    from skills import validate_plan as _vp
    bad = normalize_plan({"goal": _COFFEE_GOAL, "layers": [{"name": "b", "files": [
        {"path": "backend/app.py", "description": "flask app"},
        {"path": "backend/routes.py", "description": "routes"}]}]})
    skills = _ds(_COFFEE_GOAL)
    v = _vp(skills, [{"path": p} for p in ordered_paths(bad)], _COFFEE_GOAL)
    assert v is not None and not v.passed and "extensions" in v.rationale


def test_import_to_pypi_maps_flask_extensions():
    from cgx.codegen.env_manager import _IMPORT_TO_PYPI
    assert _IMPORT_TO_PYPI.get("flask_sqlalchemy") == "flask-sqlalchemy"
    assert _IMPORT_TO_PYPI.get("flask_cors") == "flask-cors"
    assert _IMPORT_TO_PYPI.get("rest_framework") == "djangorestframework"


def test_swarm_structural_scan_gates_circular_import(tmp_path):
    """The Swarm verify path now HARD-gates a routes<->app cycle (previously it
    burned dynamic-repair rounds and reported a generic failure)."""
    from cgx.session.tasks.swarm_verify import _structural_scan
    contents = {
        "backend/app.py": ("from flask import Flask\n"
                           "from backend.routes.members import bp\n"
                           "app = Flask(__name__)\n"),
        "backend/routes/members.py": ("from backend.app import app, db\n"
                                      "bp = None\n"),
    }
    paths = list(contents)
    gaps, import_w, phantom_w, contract_w = _structural_scan(
        paths, contents, {}, str(tmp_path), ["flask"], "members orders")
    kinds = {w.get("kind") for w in import_w}
    assert import_w, "cycle must gate (structural_ok = not (gaps or import_w))"
    assert "circular_import" in kinds or "skill_verdict" in kinds
