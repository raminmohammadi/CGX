"""VERIFY: targeted autonomous fixes + converge-then-escalate (fail fast).

The coffee-shop run wrote every file, then VERIFY burned 12 min / 5 repair
rounds and still failed because it (a) never ran the deterministic fixers,
(b) regenerated the *failing test* instead of the file that should DEFINE the
missing model, and (c) had no sense of non-convergence so it repaired the same
signature five times. These lock in the fixes:

* the Layer-1 deterministic autofix rewrites a Flask double-prefix in place;
* a referenced-but-undefined model retargets the regen to the DEFINER
  (``models.py``) with a column-precise, additive directive -- not the test;
* a failure signature unchanged across rounds escalates once then stops, so a
  stuck run fails in ~2 rounds, not 5.
"""

from __future__ import annotations

from cgx.session.models import TaskKind, TaskNode
from cgx.session.tasks import swarm_verify as sv
from cgx.session.tasks.base import ExecutorDeps


class _Art:
    def __init__(self, content):
        self.content = content
        self.artifact_id = "wp"


class _Store:
    def __init__(self, plan):
        self._plan = plan

    def get_artifact(self, _aid):
        return _Art(self._plan)

    def add_fact(self, *_a, **_k):
        pass


class _Provider:
    pass


def _plan(root, paths, skills=None, layers=None):
    return {"goal": "coffee shop with members and orders",
            "project_root": str(root), "paths": paths,
            "skills": skills or [], "layers": layers or [], "contracts": {}}


def _run(tmp_path, plan):
    deps = ExecutorDeps(project_root=str(tmp_path), provider=_Provider(),
                        store=_Store(plan))
    task = TaskNode.new(session_id="s", kind=TaskKind.SWARM_VERIFY,
                        name="verify", inputs={"work_plan_artifact_id": "wp"})
    return sv.swarm_verify(task, deps)


def _write_flask_tree(tmp_path):
    (tmp_path / "backend").mkdir()
    (tmp_path / "backend" / "routes").mkdir()
    (tmp_path / "backend" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "backend" / "routes" / "__init__.py").write_text(
        "", encoding="utf-8")
    (tmp_path / "backend" / "extensions.py").write_text(
        "from flask_sqlalchemy import SQLAlchemy\ndb = SQLAlchemy()\n",
        encoding="utf-8")
    (tmp_path / "backend" / "models.py").write_text(
        "from backend.extensions import db\n\n\n"
        "class Member(db.Model):\n    id = db.Column(db.Integer, "
        "primary_key=True)\n", encoding="utf-8")
    # orders.py references an `orders` table with no matching model + a route
    # that repeats the /api prefix it is registered under.
    (tmp_path / "backend" / "routes" / "orders.py").write_text(
        'from flask import Blueprint, jsonify\n'
        'from backend.extensions import db\n\n'
        'bp = Blueprint("orders", __name__)\n\n'
        'def create_order():\n'
        '    db.session.execute(db.text("INSERT INTO orders (order_id, '
        'member_id, items, payment_link) VALUES (?, ?, ?, ?)"))\n\n'
        '@bp.route("/api/orders", methods=["POST"])\n'
        'def api_create_order():\n    return jsonify({})\n',
        encoding="utf-8")
    (tmp_path / "backend" / "app.py").write_text(
        'from flask import Flask\n'
        'from backend.extensions import db\n'
        'from backend.routes.orders import bp as orders_bp\n\n'
        'def create_app():\n    app = Flask(__name__)\n'
        '    db.init_app(app)\n'
        '    app.register_blueprint(orders_bp, url_prefix="/api")\n'
        '    return app\n', encoding="utf-8")
    (tmp_path / "requirements.txt").write_text(
        "flask\nflask-sqlalchemy\npytest\n", encoding="utf-8")
    return ["backend/extensions.py", "backend/models.py",
            "backend/routes/orders.py", "backend/app.py", "requirements.txt"]


def test_double_prefix_is_deterministically_autofixed_in_place(tmp_path):
    paths = _write_flask_tree(tmp_path)
    # No LLM regen needed for the autofix; make it a no-op so the test is
    # hermetic and only the deterministic pass can change the route.
    changed = sv._apply_deterministic_repairs(paths, str(tmp_path), ["flask"])
    assert "backend/routes/orders.py" in changed
    body = (tmp_path / "backend" / "routes" / "orders.py").read_text()
    assert '.route("/orders"' in body       # prefix stripped from the decorator
    assert "/api/orders" not in body        # no more double-prefix at runtime


def test_missing_model_retargets_regen_to_the_definer_with_columns(
        tmp_path, monkeypatch):
    paths = _write_flask_tree(tmp_path)
    seen = []

    def _fake_generate(**kw):
        seen.append((kw["path"], kw.get("description", "")))
        # Only the definer regen "heals": grow models.py to DEFINE Order.
        if kw["path"] == "backend/models.py":
            content = (
                "from backend.extensions import db\n\n\n"
                "class Member(db.Model):\n    id = db.Column(db.Integer, "
                "primary_key=True)\n\n\n"
                "class Order(db.Model):\n"
                "    order_id = db.Column(db.Integer, primary_key=True)\n"
                "    member_id = db.Column(db.Integer)\n"
                "    items = db.Column(db.String)\n"
                "    payment_link = db.Column(db.String)\n")
            return type("O", (), {"ok": True, "content": content})()
        return type("O", (), {"ok": True, "content": ""})()

    monkeypatch.setattr(sv, "generate_file", _fake_generate)
    monkeypatch.setattr(sv, "_run_env_dryrun",
                        lambda paths, root: {"ran": False, "outcome": "skipped"})
    res = _run(tmp_path, _plan(tmp_path, paths, skills=["flask"]))

    regen_paths = [p for p, _ in seen]
    assert "backend/models.py" in regen_paths, regen_paths
    # NEVER the referencing file -- the definer is regenerated, additively.
    assert "backend/routes/orders.py" not in regen_paths
    # The directive that steered the models.py regen carried the mined columns.
    models_dirs = [d for p, d in seen if p == "backend/models.py"]
    assert any("Order" in d and "order_id" in d and "payment_link" in d
               for d in models_dirs), models_dirs
    # Once Order is defined the structural break clears -> verifies clean.
    assert res.artifact.content["import_warnings"] == []
    assert res.outputs["verify_ok"] is True
    assert "class Order(db.Model)" in \
        (tmp_path / "backend" / "models.py").read_text()


def test_converge_then_escalate_stops_fast_not_five_rounds(
        tmp_path, monkeypatch):
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text(
        "def test_foo():\n    assert False\n", encoding="utf-8")
    paths = ["app.py", "tests/test_x.py"]

    # Same DEFECT every round, but the prose jiggles (a repair-temperature
    # nonce) so byte-equality never fires -- only the signature guard can stop.
    calls = {"n": 0}

    def _env(paths, root):
        calls["n"] += 1
        return {"ran": True, "outcome": "failed",
                "output": ("FAILED tests/test_x.py::test_foo\n"
                           "sqlalchemy.exc.OperationalError: no such table "
                           f"orders\n# repair nonce {calls['n']}")}

    monkeypatch.setattr(sv, "_run_env_dryrun", _env)
    monkeypatch.setattr(sv, "generate_file",
                        lambda **kw: type("O", (), {"ok": True,
                                                     "content": ""})())
    # Repairer declines (returns nothing changed) so the loop relies on the
    # convergence guard, not on a lucky repair.
    monkeypatch.setattr(sv, "_dynamic_repair", lambda *a, **k: [])
    res = _run(tmp_path, _plan(tmp_path, paths))

    # Old behaviour burned all 5 rounds; the signature guard escalates once then
    # bails -> at most 2 dynamic rounds.
    assert res.artifact.content["dynamic_regen_rounds"] <= 2
    assert res.outputs["verify_ok"] is False


def test_hard_signature_excludes_tests_includes_errors_and_toolcodes():
    # Pure assertion failure -> no HARD signature (soft only); iterating on it is
    # progress, not a stall.
    assert sv._hard_signature("FAILED tests/t.py::test_a\nassert 1 == 2") == \
        frozenset()
    # Python error + missing symbol -> hard.
    h = sv._hard_signature("OperationalError: no such table orders")
    assert any(t.startswith("err:") for t in h)
    assert any(t.startswith("miss:") for t in h)
    # Non-Python toolchains fingerprint via their error CODES (else empty ->
    # spuriously 'stalled'): tsc / rustc / MSVC.
    assert sv._hard_signature("src/x.ts(3,5): error TS2322: bad") != frozenset()
    assert sv._hard_signature("error[E0308]: mismatched types") != frozenset()


def test_soft_only_failure_uses_full_repair_budget(tmp_path, monkeypatch):
    # A pure assertion failure (no hard signature) must NOT trip the stall guard:
    # it gets the full repair-temperature ramp, not a 2-round cut.
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_text(
        "def test_foo():\n    assert False\n", encoding="utf-8")
    calls = {"n": 0}

    def _env(paths, root):
        calls["n"] += 1
        return {"ran": True, "outcome": "failed",
                "output": (f"FAILED tests/test_x.py::test_foo\n"
                           f"assert {calls['n']} == {calls['n'] + 1}")}

    monkeypatch.setattr(sv, "_run_env_dryrun", _env)
    monkeypatch.setattr(sv, "generate_file",
                        lambda **kw: type("O", (), {"ok": True,
                                                     "content": ""})())
    monkeypatch.setattr(sv, "_dynamic_repair", lambda *a, **k: [])
    res = _run(tmp_path, _plan(tmp_path, ["app.py", "tests/test_x.py"]))
    # No hard signal -> full budget (5), unlike the hard-stall case (<=2).
    assert res.artifact.content["dynamic_regen_rounds"] == \
        sv._MAX_DYNAMIC_REPAIR_ROUNDS


def test_failure_signature_is_temperature_invariant():
    a = ("FAILED tests/t.py::test_a\nOperationalError: no such table orders\n"
         "# nonce 1  at 0x7f00")
    b = ("FAILED tests/t.py::test_a\nOperationalError: no such table orders\n"
         "# nonce 999  at 0x9999")
    assert sv._failure_signature(a) == sv._failure_signature(b)
    # A genuinely different defect changes the signature.
    c = "FAILED tests/t.py::test_b\nValueError: bad\n"
    assert sv._failure_signature(a) != sv._failure_signature(c)
