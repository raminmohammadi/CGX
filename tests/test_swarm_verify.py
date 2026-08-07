"""Phase 3: the tree-level verification ladder (SWARM_VERIFY).

Once every file has been attempted, VERIFY scans the whole tree for coverage
gaps (a planned file missing or unparseable) and import breaks (a first-party
``from X import name`` naming a symbol no file defines). A gap or break names a
concrete file, so a bounded regeneration loop re-runs the generation ladder on
just those files before the stage reports. The environment dry-run is stubbed
here so the tests stay hermetic and focus on the structural logic.
"""

import pytest

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


class _Provider:
    """Non-None placeholder; the generation ladder itself is monkeypatched."""


@pytest.fixture(autouse=True)
def _skip_env(monkeypatch):
    monkeypatch.setattr(sv, "_run_env_dryrun",
                        lambda paths, root: {"ran": False, "outcome": "skipped"})


def _plan(root, paths, layers=None, contracts=None):
    return {"goal": "g", "project_root": str(root), "paths": paths,
            "layers": layers or [], "contracts": contracts or {}}


def _run(tmp_path, plan, failed_paths=None):
    store = _Store(plan)
    deps = ExecutorDeps(project_root=str(tmp_path), provider=_Provider(),
                        store=store)
    task = TaskNode.new(session_id="s", kind=TaskKind.SWARM_VERIFY,
                        name="verify",
                        inputs={"work_plan_artifact_id": "wp",
                                "failed_paths": failed_paths or []})
    return sv.swarm_verify(task, deps)


def test_clean_tree_verifies_ok(tmp_path):
    (tmp_path / "models.py").write_text("class User:\n    pass\n",
                                        encoding="utf-8")
    (tmp_path / "app.py").write_text(
        "from models import User\n\n\ndef run():\n    return User\n",
        encoding="utf-8")
    res = _run(tmp_path, _plan(tmp_path, ["models.py", "app.py"]))
    assert res.outputs["verify_ok"] is True
    assert res.outputs["coverage_gaps"] == []
    assert res.artifact.content["import_warnings"] == []


def test_missing_file_is_a_coverage_gap(tmp_path, monkeypatch):
    (tmp_path / "models.py").write_text("class User:\n    pass\n",
                                         encoding="utf-8")
    # app.py is planned but never written; regeneration also fails to write it.
    monkeypatch.setattr(
        sv, "generate_file",
        lambda **kw: type("O", (), {"ok": False, "content": ""})())
    res = _run(tmp_path, _plan(tmp_path, ["models.py", "app.py"]))
    assert res.outputs["verify_ok"] is False
    assert "app.py" in res.outputs["coverage_gaps"]
    assert res.artifact.content["regen_rounds"] == sv._MAX_VERIFY_ROUNDS


def test_import_break_triggers_regeneration_that_heals(tmp_path, monkeypatch):
    (tmp_path / "models.py").write_text("class User:\n    pass\n",
                                        encoding="utf-8")
    # app.py imports a symbol models.py does not define -> import break.
    (tmp_path / "app.py").write_text(
        "from models import Ghost\n", encoding="utf-8")

    def _heal(**kw):
        # The regen ladder rewrites app.py to import a symbol that exists.
        return type("O", (), {"ok": True,
                              "content": "from models import User\n"})()
    monkeypatch.setattr(sv, "generate_file", _heal)
    plan = _plan(tmp_path, ["models.py", "app.py"],
                 layers=[{"name": "core", "files": [
                     {"path": "app.py", "description": "entry",
                      "depends_on": ["models.py"]}]}])
    res = _run(tmp_path, plan)
    assert res.outputs["verify_ok"] is True
    assert res.artifact.content["regen_rounds"] >= 1
    assert (tmp_path / "app.py").read_text() == "from models import User\n"


def test_unwritten_developer_file_fails_verify(tmp_path):
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    res = _run(tmp_path, _plan(tmp_path, ["app.py"]), failed_paths=["app.py"])
    assert res.outputs["verify_ok"] is False


def test_no_provider_fails_fast(tmp_path):
    deps = ExecutorDeps(project_root=str(tmp_path), provider=None,
                        store=_Store(_plan(tmp_path, [])))
    task = TaskNode.new(session_id="s", kind=TaskKind.SWARM_VERIFY,
                        name="verify", inputs={})
    res = sv.swarm_verify(task, deps)
    assert res.failure and not res.retryable


def test_misrooted_first_party_import_fails_verify(tmp_path, monkeypatch):
    # A first-party import that resolves against neither root nor root/src is
    # a break the basename-blind symbol check abstained on; the resolver now
    # names the file and, when regeneration cannot heal it, verify is red.
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "app.py").write_text(
        "from pkg.missing import thing\n", encoding="utf-8")
    monkeypatch.setattr(
        sv, "generate_file",
        lambda **kw: type("O", (), {"ok": False, "content": ""})())
    res = _run(tmp_path, _plan(tmp_path, ["pkg/app.py"]))
    assert res.outputs["verify_ok"] is False
    mods = [w["module"] for w in res.artifact.content["import_warnings"]]
    assert "pkg.missing" in mods


def test_dynamic_regen_heals_red_but_structurally_clean_tree(
        tmp_path, monkeypatch):
    # The tree parses and its imports resolve on paper, yet the suite is red
    # with a ModuleNotFoundError -> parse it, regenerate the implicated file
    # once, and dry-run again (which now passes).
    (tmp_path / "app.py").write_text(
        "import os\n\nx = os.getcwd()\n", encoding="utf-8")
    calls = {"env": 0}

    def fake_env(paths, root):
        calls["env"] += 1
        if calls["env"] == 1:
            return {"ran": True, "outcome": "failed",
                    "output": "ModuleNotFoundError: No module named 'app'"}
        return {"ran": True, "outcome": "passed", "output": ""}
    monkeypatch.setattr(sv, "_run_env_dryrun", fake_env)

    healed = {"n": 0}

    def heal(**kw):
        healed["n"] += 1
        return type("O", (), {"ok": True, "content": "x = 1\n"})()
    monkeypatch.setattr(sv, "generate_file", heal)

    res = _run(tmp_path, _plan(tmp_path, ["app.py"]))
    assert res.artifact.content["dynamic_regen_rounds"] == 1
    assert healed["n"] == 1
    assert res.outputs["verify_ok"] is True
