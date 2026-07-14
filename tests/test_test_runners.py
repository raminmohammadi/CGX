"""Tests for the pluggable multi-language test-runner registry.

Covers stack detection (pytest vs npm, polyglot), npm script/command
selection, graceful degradation when a toolchain is absent, and the
end-to-end pytest path plus outcome aggregation.
"""

from __future__ import annotations

import sys
from pathlib import Path

from cgx.codegen.test_runner import TestRunOutcome
from cgx.codegen.test_runners import (
    NpmRunner,
    PytestRunner,
    TestRunner,
    _npm_script_command,
    detect_test_runners,
    run_project_tests,
)


def _write(root: Path, rel: str, body: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------
def test_pytest_runner_detects_marker_file(tmp_path):
    _write(tmp_path, "pyproject.toml", "[project]\nname='x'\n")
    assert PytestRunner().detect(str(tmp_path)) is True


def test_pytest_runner_detects_via_test_module(tmp_path):
    # No marker files, but a test module is enough to qualify.
    _write(tmp_path, "pkg/test_thing.py", "def test_ok():\n    assert True\n")
    assert PytestRunner().detect(str(tmp_path)) is True


def test_npm_runner_detects_package_json(tmp_path):
    _write(tmp_path, "package.json", '{"name": "x"}')
    assert NpmRunner().detect(str(tmp_path)) is True
    # A pure JS project has no Python markers/tests.
    assert PytestRunner().detect(str(tmp_path)) is False


def test_detect_test_runners_polyglot(tmp_path):
    _write(tmp_path, "pyproject.toml", "[project]\nname='x'\n")
    _write(tmp_path, "package.json", '{"name": "x", "scripts": {"build": "tsc"}}')
    names = {r.name for r in detect_test_runners(str(tmp_path))}
    assert names == {"pytest", "npm"}


def test_detect_test_runners_empty_project(tmp_path):
    assert detect_test_runners(str(tmp_path)) == []


# --------------------------------------------------------------------------
# npm command selection
# --------------------------------------------------------------------------
def test_npm_command_prefers_real_test_script(tmp_path):
    _write(tmp_path, "package.json", '{"scripts": {"test": "vitest run", "build": "tsc"}}')
    assert _npm_script_command(str(tmp_path)) == ["npm", "test", "--silent"]


def test_npm_command_falls_back_to_build_for_placeholder(tmp_path):
    _write(
        tmp_path, "package.json",
        '{"scripts": {"test": "echo \\"Error: no test specified\\" && exit 1",'
        ' "build": "tsc"}}',
    )
    assert _npm_script_command(str(tmp_path)) == ["npm", "run", "build", "--silent"]


def test_npm_command_none_when_no_usable_script(tmp_path):
    _write(tmp_path, "package.json", '{"name": "x"}')
    assert _npm_script_command(str(tmp_path)) is None


def test_npm_runner_skips_when_npm_missing(tmp_path, monkeypatch):
    _write(tmp_path, "package.json", '{"scripts": {"build": "tsc"}}')
    monkeypatch.setattr("cgx.codegen.test_runners.shutil.which", lambda _: None)
    outcome = NpmRunner().run(str(tmp_path), [])
    assert outcome.ran is False
    assert "npm not installed" in (outcome.skipped_reason or "")


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def test_run_project_tests_runs_pytest_end_to_end(tmp_path):
    _write(tmp_path, "pyproject.toml", "[project]\nname='x'\n")
    _write(tmp_path, "mymod.py", "def add(a, b):\n    return a + b\n")
    _write(
        tmp_path, "tests/test_mymod.py",
        "from mymod import add\n\ndef test_add():\n    assert add(1, 2) == 3\n",
    )
    outcome = run_project_tests(
        str(tmp_path), ["mymod.py"], python_exe=sys.executable,
    )
    assert outcome.ran is True
    assert outcome.returncode == 0
    assert outcome.tests_selected


def test_run_project_tests_no_runner_is_soft_skip(tmp_path):
    outcome = run_project_tests(str(tmp_path), [])
    assert outcome.ran is False
    assert "no test runner detected" in (outcome.skipped_reason or "")


class _FakeRunner(TestRunner):
    def __init__(self, name, outcome):
        self.name = name
        self._outcome = outcome

    def detect(self, project_root):
        return True

    def run(self, project_root, changed_files, *, timeout_seconds=180.0, python_exe=None):
        return self._outcome


def test_run_project_tests_aggregates_worst_case_returncode(tmp_path):
    passing = TestRunOutcome(ran=True, returncode=0, stdout="ok", tests_selected=["a"])
    failing = TestRunOutcome(ran=True, returncode=1, stderr="boom", tests_selected=["b"])
    outcome = run_project_tests(
        str(tmp_path), [],
        runners=[_FakeRunner("pytest", passing), _FakeRunner("npm", failing)],
    )
    assert outcome.ran is True
    assert outcome.returncode == 1
    assert outcome.tests_selected == ["a", "b"]
    assert "== npm ==" in outcome.stderr
