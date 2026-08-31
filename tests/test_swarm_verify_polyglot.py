"""Stage 3: the swarm verifier runs the polyglot test/build runner.

Kept separate from ``test_swarm_verify.py`` because that module's autouse
fixture stubs ``_run_env_dryrun`` -- here we exercise the real one.
"""

from cgx.codegen.test_runner import TestRunOutcome
from cgx.session.tasks import swarm_verify as sv


def test_env_dryrun_uses_polyglot_runner(tmp_path, monkeypatch):
    calls = {}

    def fake_run(root, changed, **kw):
        calls["root"] = root
        return TestRunOutcome(ran=True, returncode=1, stdout="npm build boom",
                              stderr="", tests_selected=["npm:build"])

    import cgx.codegen.test_runners as _tr
    monkeypatch.setattr(_tr, "run_project_tests", fake_run)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "App.jsx").write_text("export default 1\n",
                                              encoding="utf-8")
    report = sv._run_env_dryrun(["src/App.jsx"], str(tmp_path))
    assert calls  # the polyglot runner was invoked
    assert report["outcome"] == "failed"
    assert "npm build boom" in report["output"]


def test_env_dryrun_passed_when_runner_green(tmp_path, monkeypatch):
    import cgx.codegen.test_runners as _tr
    monkeypatch.setattr(_tr, "run_project_tests",
                        lambda root, changed, **kw: TestRunOutcome(
                            ran=True, returncode=0, stdout="ok"))
    report = sv._run_env_dryrun(["src/App.jsx"], str(tmp_path))
    assert report["outcome"] == "passed"


def test_reconcile_python_adds_imported_and_implied_deps(tmp_path, monkeypatch):
    # Systemic: requirements is reconciled from what the source imports. A
    # directly-imported package (requests) AND a framework-implied one (httpx,
    # for FastAPI TestClient, never imported) are both installed + pinned.
    (tmp_path / "app.py").write_text(
        "import requests\nfrom fastapi import FastAPI\napp=FastAPI()\n",
        encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("fastapi\npytest\n",
                                               encoding="utf-8")
    installed = {}
    import cgx.codegen.env_manager as em
    monkeypatch.setattr(em, "install_packages",
                        lambda dists, python=None: installed.update(
                            {d: True for d in dists}) or {d: True for d in dists})
    sv._reconcile_python_requirements(["app.py"], str(tmp_path))
    reqs = (tmp_path / "requirements.txt").read_text()
    assert "requests" in installed and "httpx" in installed
    assert "requests" in reqs and "httpx" in reqs


def test_reconcile_python_noop_for_stdlib_only(tmp_path, monkeypatch):
    (tmp_path / "m.py").write_text("import os, json, sys\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("pytest\n", encoding="utf-8")
    called = {"n": 0}
    import cgx.codegen.env_manager as em
    monkeypatch.setattr(em, "install_packages",
                        lambda dists, python=None: called.update(n=called["n"]+1) or {})
    sv._reconcile_python_requirements(["m.py"], str(tmp_path))
    assert called["n"] == 0  # nothing third-party to install


def test_reconcile_node_installs_undeclared_imports(tmp_path, monkeypatch):
    import shutil, cgx.session.tasks.swarm_verify as _sv
    monkeypatch.setattr(shutil, "which", lambda _n: "/usr/bin/npm")
    fe = tmp_path / "frontend"
    (fe / "src").mkdir(parents=True)
    (fe / "package.json").write_text(
        '{"dependencies": {"react": "^18"}}', encoding="utf-8")
    (fe / "src" / "App.jsx").write_text(
        "import axios from 'axios';\nimport React from 'react';\n",
        encoding="utf-8")
    calls = {}
    import subprocess
    def fake_run(cmd, **kw):
        calls["cmd"] = cmd; calls["cwd"] = kw.get("cwd")
        return type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})()
    monkeypatch.setattr(subprocess, "run", fake_run)
    _sv._reconcile_node_dependencies(str(tmp_path))
    # axios (undeclared) is installed; react (declared) is not re-added.
    assert "axios" in calls["cmd"] and "react" not in calls["cmd"]
    assert calls["cwd"] == str(fe)


def test_phantom_third_party_is_advisory_not_gating(tmp_path):
    # `import uvicorn` not listed in contracts must be an *advisory* phantom
    # warning, not a hard import break -- so a working, installed dependency
    # the model under-declared can't sink a build whose tests pass.
    contents = {"main.py": "import uvicorn\nx = 1\n"}
    gaps, import_breaks, phantom, contract = sv._structural_scan(
        ["main.py"], contents, {"third_party_dependencies": []}, str(tmp_path))
    assert any(w.get("module") == "uvicorn" for w in phantom)
    assert import_breaks == []   # not a hard, gating signal


def test_repair_context_includes_js_files():
    # Fix B: a red JS build must offer the implicated frontend file to the
    # repairer -- the context is no longer .py-only.
    paths = ["backend/app.py", "frontend/src/App.jsx", "frontend/src/main.jsx"]
    # No localization -> all source files (py + js) are eligible.
    ctx = sv._repair_context_paths([], paths, {})
    assert "frontend/src/App.jsx" in ctx and "backend/app.py" in ctx
    # Localized to a jsx file -> it is kept (previously dropped as non-.py).
    ctx2 = sv._repair_context_paths(["frontend/src/App.jsx"], paths, {})
    assert "frontend/src/App.jsx" in ctx2
