"""Tests for the PythonCliSkill: detection, the testable-entrypoint prompt,
and the third-party-import -> dependency-manifest FATAL gate."""

from __future__ import annotations

import pytest

import skills as registry
from skills.python_cli import PythonCliSkill


@pytest.fixture()
def skill() -> PythonCliSkill:
    return PythonCliSkill()


def _diff(path, patch=""):
    return {"file": path, "patch": patch}


# --- detection (preserved behavior) ---------------------------------------

@pytest.mark.parametrize("goal", [
    "build a python cli tool",
    "a command-line utility in python",
    "python script to rename files",
])
def test_detects_python_cli_goals(skill, goal):
    assert skill.detect(goal) >= 0.9


def test_cli_noun_only_is_weaker(skill):
    assert skill.detect("a command-line tool") == 0.55


@pytest.mark.parametrize("goal", [
    "build a Flask cli command",
    "a FastAPI command-line admin tool",
    "django management script",
])
def test_abstains_when_web_framework_named(skill, goal):
    assert skill.detect(goal) == 0.0


def test_registered_and_detected_via_registry():
    detected = [s.name for s in registry.detect_skills("python cli tool")]
    assert "python_cli" in detected


# --- prompt mandates a testable entrypoint --------------------------------

def test_prompt_mandates_testable_entrypoint(skill):
    p = skill.scaffold_system_prompt()
    assert "def main(argv=None):" in p
    assert "parser.parse_args(argv)" in p
    assert "raise SystemExit(main())" in p
    # console_scripts entry when packaged
    assert "console_scripts" in p or "[project.scripts]" in p


# --- validate_scaffold: existing checks preserved -------------------------

def test_validate_scaffold_no_diffs_is_no_opinion(skill):
    assert skill.validate_scaffold([]) is None


def test_validate_scaffold_requires_python(skill):
    v = skill.validate_scaffold([_diff("README.md"), _diff("data.csv")])
    assert v is not None and not v.passed
    assert "Python" in v.rationale


def test_validate_scaffold_requires_entrypoint_markers(skill):
    # A .py file with none of argparse / console_scripts / __main__.
    v = skill.validate_scaffold([_diff("util.py", "x = 1\n")])
    assert v is not None and not v.passed
    assert "argparse" in v.rationale


# --- validate_scaffold: third-party import -> manifest FATAL --------------

def test_third_party_import_without_manifest_is_fatal(skill):
    diffs = [_diff(
        "cli.py",
        "import argparse\nimport requests\n\n"
        "def main(argv=None):\n"
        "    args = argparse.ArgumentParser().parse_args(argv)\n"
        "    return 0\n"
        'if __name__ == "__main__":\n    raise SystemExit(main())\n',
    )]
    v = skill.validate_scaffold(diffs)
    assert v is not None and not v.passed
    assert v.severity == "error"
    assert "requests" in v.rationale


def test_third_party_import_with_requirements_txt_passes(skill):
    diffs = [
        _diff("cli.py", "import argparse\nimport requests\n"),
        _diff("requirements.txt", "requests>=2\n"),
    ]
    assert skill.validate_scaffold(diffs) is None


def test_third_party_import_with_pyproject_passes(skill):
    diffs = [
        _diff("src/tool/cli.py", "import argparse\nimport click\n"),
        _diff("pyproject.toml", "[project]\ndependencies=['click']\n"),
    ]
    assert skill.validate_scaffold(diffs) is None


def test_setup_py_and_setup_cfg_count_as_manifest(skill):
    for manifest in ("setup.py", "setup.cfg"):
        diffs = [
            _diff("cli.py", "import argparse\nimport rich\n"),
            _diff(manifest, "rich\n"),
        ]
        assert skill.validate_scaffold(diffs) is None, manifest


def test_nested_requirements_txt_counts(skill):
    diffs = [
        _diff("app/cli.py", "import argparse\nimport httpx\n"),
        _diff("app/requirements.txt", "httpx\n"),
    ]
    assert skill.validate_scaffold(diffs) is None


# --- validate_scaffold: pure-stdlib CLIs remain allowed -------------------

def test_pure_stdlib_cli_needs_no_manifest(skill):
    diffs = [_diff(
        "cli.py",
        "import argparse\nimport sys\nimport json\nimport pathlib\n"
        "from collections import Counter\n\n"
        "def main(argv=None):\n"
        "    args = argparse.ArgumentParser().parse_args(argv)\n"
        "    return 0\n"
        'if __name__ == "__main__":\n    raise SystemExit(main())\n',
    )]
    assert skill.validate_scaffold(diffs) is None


def test_local_package_import_is_not_third_party(skill):
    diffs = [
        _diff("src/mypkg/cli.py",
              "import argparse\nfrom mypkg.core import run\nimport sys\n"),
        _diff("src/mypkg/core.py", "def run():\n    return 0\n"),
        _diff("src/mypkg/__init__.py", ""),
    ]
    assert skill.validate_scaffold(diffs) is None


def test_relative_import_is_not_third_party(skill):
    diffs = [
        _diff("cli.py",
              "import argparse\nfrom . import helpers\nfrom .util import fn\n"),
        _diff("helpers.py", ""),
        _diff("util.py", "def fn():\n    return 1\n"),
    ]
    assert skill.validate_scaffold(diffs) is None


def test_pytest_in_tests_does_not_demand_manifest(skill):
    # A stdlib-only CLI whose only non-stdlib import is pytest in its tests
    # must still count as manifest-free.
    diffs = [
        _diff("cli.py", "import argparse\ndef main(argv=None):\n    return 0\n"),
        _diff("tests/test_cli.py",
              "import pytest\nfrom cli import main\n\n"
              "def test_runs():\n    assert main([]) == 0\n"),
    ]
    assert skill.validate_scaffold(diffs) is None


# --- helper precision ------------------------------------------------------

def test_third_party_imports_lists_sorted_and_deduped(skill):
    diffs = [_diff(
        "cli.py",
        "import argparse\nimport requests\nimport click\nimport requests\n",
    )]
    tp = skill._third_party_imports(diffs, ["cli.py"])
    assert set(tp) == {"requests", "click"}
    v = skill.validate_scaffold(diffs)
    # rationale lists them sorted: click before requests
    assert v is not None
    assert v.rationale.index("click") < v.rationale.index("requests")


def test_has_manifest_recognizes_known_names(skill):
    assert skill._has_manifest(["pyproject.toml"])
    assert skill._has_manifest(["a/b/requirements.txt"])
    assert not skill._has_manifest(["cli.py", "README.md"])


# --- warnings preserved ----------------------------------------------------

def test_warns_when_no_test_file(skill):
    warns = skill.scaffold_warnings([_diff("cli.py", "import argparse\n")])
    assert warns and warns[0].severity == "warning"


def test_no_warning_when_test_present(skill):
    diffs = [_diff("cli.py", "import argparse\n"),
             _diff("tests/test_cli.py", "def test_x():\n    pass\n")]
    assert skill.scaffold_warnings(diffs) == []
