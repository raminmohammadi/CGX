"""Structural detectors for the coffee-shop failure class.

These lock in the three deterministic Layer-1 checks that turn "burned 12 min of
LLM repair then failed" into "detected + targeted the fix instantly":

* an ORM model / raw-SQL table referenced but defined by no ``db.Model``
  (``undefined_model_refs`` -- mines the columns from the ``INSERT`` so the
  regen of ``models.py`` is additive and precise, never symptom-stripping);
* a blueprint route repeating its own ``url_prefix`` across the import-alias hop
  (``blueprint_prefix_collisions`` -- the ``/api/api/...`` 404 bug);
* a relative SQLite URI (``relative_sqlite_uris`` -- ``unable to open db file``).

The fixtures are the actual coffee-shop bodies from the failed run.
"""

from __future__ import annotations

from skills._structural import (
    blueprint_prefix_collisions,
    defined_orm_models,
    relative_sqlite_uris,
    undefined_model_refs,
)

_MODELS = "from backend.extensions import db\n\n\nclass Member(db.Model):\n    id = db.Column(db.Integer, primary_key=True)\n    name = db.Column(db.String(100), nullable=False)\n"

_ORDERS = (
    'from flask import Blueprint, request, jsonify\n'
    'from backend.models import Member\n'
    'from backend.extensions import cors, db\n\n'
    'bp = Blueprint("orders", __name__)\n\n'
    'def create_order(member_id, items):\n'
    '    with db.session.begin():\n'
    '        member = db.session.get(Member, member_id)\n'
    '        order_id = db.session.execute(db.text("SELECT MAX(order_id) FROM orders")).scalar() or 0\n'
    '        db.session.execute(db.text("INSERT INTO orders (order_id, member_id, items, payment_link) VALUES (?, ?, ?, ?)"), (order_id, member_id, str(items), "x"))\n'
    '    return {"order_id": order_id}\n\n'
    '@bp.route("/api/orders", methods=["POST"])\n'
    'def api_create_order():\n'
    '    return jsonify(create_order(1, []))\n'
)

_MEMBERS = (
    'from flask import Blueprint, request, jsonify\n'
    'from backend.models import Member\n'
    'from backend.extensions import db\n\n'
    "bp = Blueprint('members', __name__)\n\n"
    "@bp.route('/api/members', methods=['POST'])\n"
    'def create_member_route():\n'
    '    return jsonify({})\n'
)

_APP = (
    'from flask import Flask\n'
    'from backend.extensions import cors, db, init_db\n'
    'from backend.routes.members import bp as members_bp\n'
    'from backend.routes.orders import bp as orders_bp\n\n'
    'def create_app():\n'
    '    app = Flask(__name__)\n'
    "    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///./data/app.db'\n"
    '    init_db(app)\n'
    "    app.register_blueprint(members_bp, url_prefix='/api')\n"
    "    app.register_blueprint(orders_bp, url_prefix='/api')\n"
    '    return app\n'
)


def _diffs():
    return [
        {"path": "backend/models.py", "content": _MODELS},
        {"path": "backend/routes/orders.py", "content": _ORDERS},
        {"path": "backend/routes/members.py", "content": _MEMBERS},
        {"path": "backend/app.py", "content": _APP},
    ]


def test_defined_models_include_member_not_order():
    defined = defined_orm_models(_diffs())
    assert "Member" in defined and defined["Member"].endswith("models.py")
    assert "Order" not in defined and "order" not in defined


def test_undefined_order_detected_with_mined_columns():
    refs = undefined_model_refs(_diffs())
    orders = [r for r in refs if r["name"] == "Order"]
    assert orders, f"Order should be flagged as undefined; got {refs}"
    o = orders[0]
    # columns mined straight from the INSERT so the models.py regen is precise
    assert o["columns"] == ["order_id", "member_id", "items", "payment_link"]
    assert o["table"] == "orders"
    assert o["define_in"] == "backend/models.py"      # retarget to the definer
    # Member is defined -> never flagged (no false positive on session.get(Member))
    assert not any(r["name"] == "Member" for r in refs)


def test_orm_ref_to_a_bound_nonmodel_is_not_flagged():
    # `Session.query(...)` where Session is IMPORTED (a real, non-Model class)
    # must NOT be mistaken for a missing model -- a false FATAL would block a
    # valid build. The raw-SQL branch still catches genuinely-missing tables.
    diffs = [
        {"path": "svc.py", "content":
            "from sqlalchemy.orm import Session\n"
            "from myapp.helpers import Report\n"
            "def f(s):\n    Report.query.all()\n"
            "    return Session.query(object)\n"},
    ]
    assert undefined_model_refs(diffs) == []


def test_double_prefix_detected_across_alias_hop():
    hits = blueprint_prefix_collisions(_diffs())
    routes = sorted(h["route"] for h in hits)
    assert routes == ["/api/members", "/api/orders"], hits
    assert all(h["prefix"] == "/api" for h in hits)


def test_no_prefix_collision_when_routes_are_relative():
    good = [
        {"path": "app.py", "content":
            "from routes.m import bp as mbp\n"
            "app.register_blueprint(mbp, url_prefix='/api')\n"},
        {"path": "routes/m.py", "content":
            "bp = Blueprint('m', __name__)\n@bp.route('/members')\ndef f(): ...\n"},
    ]
    assert blueprint_prefix_collisions(good) == []


def test_plain_english_ui_strings_are_not_mistaken_for_sql():
    # Regression: a route returning user copy that merely CONTAINS select/update/
    # from must NOT mint phantom tables (which would be a FATAL blocking a valid
    # build). The literal has to BE a SQL statement (start with a DML verb).
    diffs = [
        {"path": "r.py", "content":
            "from flask import jsonify\n"
            "def v():\n"
            "    return jsonify({'msg': 'Please select a category from the "
            "list to update your cart'})\n"
            "def w():\n    return 'Insert your details into the form below'\n"},
    ]
    assert undefined_model_refs(diffs) == []


def test_prefixless_blueprint_with_absolute_route_not_flagged():
    # Regression: an admin blueprint registered with NO url_prefix, whose route
    # is the absolute '/api/stats', must not be falsely matched against an
    # UNRELATED '/api' registration of a different blueprint that shares the
    # near-universal variable name `bp`.
    diffs = [
        {"path": "app.py", "content":
            "from routes.public import bp\n"
            "from routes.admin import bp as admin_bp\n"
            "app.register_blueprint(bp, url_prefix='/api')\n"
            "app.register_blueprint(admin_bp)\n"},
        {"path": "routes/public.py", "content":
            "bp = Blueprint('public', __name__)\n@bp.route('/api/list')\ndef a(): ...\n"},
        {"path": "routes/admin.py", "content":
            "bp = Blueprint('admin', __name__)\n@bp.route('/api/stats')\ndef s(): ...\n"},
    ]
    hits = blueprint_prefix_collisions(diffs)
    files = sorted(h["file"] for h in hits)
    assert files == ["routes/public.py"], hits   # admin's absolute route is fine
    # And the repair only rewrites the colliding blueprint's file.
    from skills.flask import FlaskSkill
    assert list(FlaskSkill().repair_scaffold(diffs)) == ["routes/public.py"]


def test_relative_sqlite_detected_absolute_and_memory_ignored():
    assert relative_sqlite_uris(_diffs())[0]["uri"] == "sqlite:///./data/app.db"
    ok = [
        {"path": "a.py", "content": "URI='sqlite:////abs/app.db'"},
        {"path": "b.py", "content": "URI='sqlite:///:memory:'"},
    ]
    assert relative_sqlite_uris(ok) == []


def test_relative_sqlite_is_surfaced_as_advisory_not_fatal():
    # The detector must be WIRED (finding #7 was that it was dead code): a
    # relative URI rides along as a non-fatal warning that steers a regen.
    from skills.flask import FlaskSkill
    diffs = [
        {"path": "backend/app.py", "content":
            "from flask import Flask\n"
            "def create_app():\n    app = Flask(__name__)\n"
            "    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///./data/app.db'\n"
            "    return app\n"},
        {"path": "tests/test_app.py", "content": "def test_x(): pass\n"},
        {"path": "requirements.txt", "content": "flask\n"},
    ]
    s = FlaskSkill()
    assert s.validate_scaffold(diffs) is None      # not fatal
    warns = s.scaffold_warnings(diffs)
    assert any("relative SQLite URI" in w.rationale and w.severity == "warning"
               for w in warns)
