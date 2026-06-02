"""End-to-end-ish tests for the self-testing codegen pipeline.

These tests use a tiny on-disk project and never touch an LLM.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from cgx.codegen.pipeline import build_retry_feedback, validate_and_test
from cgx.codegen.disk_apply import apply_diffs_to_disk, rollback_from_backup
from cgx.codegen.test_runner import discover_all_tests, run_pytest_paths


def _make_project(root: Path) -> None:
    (root / "pkg").mkdir(parents=True, exist_ok=True)
    (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pkg" / "mod.py").write_text(
        textwrap.dedent(
            """
            def add(a, b):
                return a + b
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (root / "tests").mkdir(parents=True, exist_ok=True)
    (root / "tests" / "__init__.py").write_text("", encoding="utf-8")
    (root / "tests" / "test_mod.py").write_text(
        textwrap.dedent(
            """
            from pkg.mod import add

            def test_add():
                assert add(2, 3) == 5
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def test_validate_and_test_new_file_passes(tmp_path: Path) -> None:
    _make_project(tmp_path)
    plan = textwrap.dedent(
        """
        ## Plan
        Add a `mul` function.

        ## Diffs
        ```diff path=pkg/extra.py
        --- /dev/null
        +++ b/pkg/extra.py
        @@
        +def mul(a, b):
        +    return a * b
        ```
        """
    ).strip()
    report = validate_and_test(str(tmp_path), plan, run_tests=False)
    assert report.summary["n_patches_ok"] >= 1
    assert report.summary["n_syntax_failed"] == 0


def test_validate_and_test_flags_empty_plan_as_failure(tmp_path: Path) -> None:
    # When the LLM emits prose but no fenced diff blocks, the self-test
    # must hard-fail (not silently "pass" with 0/0 patches) so the engine's
    # retry loop fires and the Judge sees an honest failure.
    _make_project(tmp_path)
    plan = textwrap.dedent(
        """
        ## Plan
        We will refactor everything to be cleaner.

        1. Identify files.
        2. Apply changes.
        """
    ).strip()
    report = validate_and_test(str(tmp_path), plan, run_tests=False)
    assert report.summary["n_targets"] == 0
    assert report.summary["empty_plan"] is True
    assert report.summary["overall_ok"] is False
    feedback = build_retry_feedback(report)
    assert "no diff blocks were parsed" in feedback


def test_validate_and_test_catches_syntax_error(tmp_path: Path) -> None:
    _make_project(tmp_path)
    plan = textwrap.dedent(
        """
        ## Diffs
        ```diff path=pkg/broken.py
        --- /dev/null
        +++ b/pkg/broken.py
        @@
        +def oops(
        +    return 1
        ```
        """
    ).strip()
    report = validate_and_test(str(tmp_path), plan, run_tests=False)
    assert report.summary["n_patches_ok"] >= 1
    assert report.summary["n_syntax_failed"] >= 1
    feedback = build_retry_feedback(report)
    assert "syntax error" in feedback.lower()


def test_validate_and_test_runs_impacted_tests(tmp_path: Path) -> None:
    pytest.importorskip("pytest")
    _make_project(tmp_path)
    plan = textwrap.dedent(
        """
        ## Diffs
        ```diff path=pkg/extra.py
        --- /dev/null
        +++ b/pkg/extra.py
        @@
        +def mul(a, b):
        +    return a * b
        ```
        """
    ).strip()
    report = validate_and_test(str(tmp_path), plan, run_tests=True, timeout_seconds=60.0)
    # With no test files referencing pkg/extra.py, the runner should mark
    # the run as skipped (no impacted tests) rather than failing.
    assert report.tests is not None
    assert (not report.tests.ran) or report.tests.returncode == 0



# --------------------------------------------------------------------------
# rollback_from_backup
# --------------------------------------------------------------------------

_EDIT_DIFF = textwrap.dedent(
    """
    --- a/pkg/mod.py
    +++ b/pkg/mod.py
    @@ -1,2 +1,2 @@
     def add(a, b):
    -    return a + b
    +    return a + b  # edited
    """
).lstrip()

_NEW_FILE_DIFF = textwrap.dedent(
    """
    --- /dev/null
    +++ b/pkg/extra.py
    @@
    +def mul(a, b):
    +    return a * b
    """
).lstrip()


def test_rollback_restores_existing_and_deletes_new(tmp_path: Path) -> None:
    _make_project(tmp_path)
    original = (tmp_path / "pkg" / "mod.py").read_text(encoding="utf-8")

    res = apply_diffs_to_disk(str(tmp_path), [
        {"file": "pkg/mod.py", "patch": _EDIT_DIFF},
        {"file": "pkg/extra.py", "patch": _NEW_FILE_DIFF},
    ])
    assert not res["failed_files"], res["failed_files"]
    assert res["backup_dir"]
    # Sanity-check the apply landed.
    assert "edited" in (tmp_path / "pkg" / "mod.py").read_text(encoding="utf-8")
    assert (tmp_path / "pkg" / "extra.py").exists()

    out = rollback_from_backup(str(tmp_path), res["backup_dir"])
    assert "pkg/mod.py" in out["restored_files"]
    assert "pkg/extra.py" in out["deleted_files"]
    assert out["failed_files"] == []
    assert out.get("error") in (None, "")
    # Existing file restored byte-for-byte.
    assert (tmp_path / "pkg" / "mod.py").read_text(encoding="utf-8") == original
    # Newly-created file removed.
    assert not (tmp_path / "pkg" / "extra.py").exists()


def test_rollback_missing_backup_dir_errors(tmp_path: Path) -> None:
    _make_project(tmp_path)
    out = rollback_from_backup(str(tmp_path), str(tmp_path / ".averix-backups" / "missing"))
    assert out["restored_files"] == []
    assert out["deleted_files"] == []
    assert "does not exist" in (out.get("error") or "")


def test_rollback_rejects_backup_outside_project_root(tmp_path: Path) -> None:
    _make_project(tmp_path)
    outside = tmp_path.parent / "elsewhere"
    outside.mkdir(parents=True, exist_ok=True)
    out = rollback_from_backup(str(tmp_path), str(outside))
    assert out["restored_files"] == []
    assert "outside project_root" in (out.get("error") or "")


# --------------------------------------------------------------------------
# Standalone verify path: discover_all_tests + run_pytest_paths
# --------------------------------------------------------------------------
def test_discover_all_tests_globs_tests_dir(tmp_path: Path) -> None:
    _make_project(tmp_path)
    (tmp_path / "tests" / "subdir").mkdir()
    (tmp_path / "tests" / "subdir" / "test_nested.py").write_text(
        "def test_nested(): assert 1 == 1\n", encoding="utf-8",
    )
    # Decoy: not a test_*.py file, should not be picked up.
    (tmp_path / "tests" / "helpers.py").write_text("x = 1\n", encoding="utf-8")
    found = discover_all_tests(str(tmp_path))
    names = sorted(Path(p).name for p in found)
    assert names == ["test_mod.py", "test_nested.py"]


def test_run_pytest_paths_executes_discovered_tests(tmp_path: Path) -> None:
    pytest.importorskip("pytest")
    _make_project(tmp_path)
    discovered = discover_all_tests(str(tmp_path))
    outcome = run_pytest_paths(str(tmp_path), discovered, timeout_seconds=60.0)
    assert outcome.ran
    assert outcome.returncode == 0
    assert outcome.tests_selected == discovered


def test_run_pytest_paths_resolves_first_party_imports_without_packaging(
    tmp_path: Path,
) -> None:
    """A freshly-scaffolded project with ``backend/`` + ``tests/`` and no
    ``pyproject.toml`` / ``conftest.py`` must still import its own modules
    under pytest. ``run_pytest_paths`` should set ``PYTHONPATH`` so
    ``from backend.calculator import …`` resolves.
    """
    pytest.importorskip("pytest")
    (tmp_path / "backend").mkdir()
    (tmp_path / "backend" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "backend" / "calculator.py").write_text(
        "def add(a, b):\n    return a + b\n", encoding="utf-8",
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_calculator.py").write_text(
        textwrap.dedent(
            """
            from backend.calculator import add

            def test_add():
                assert add(2, 3) == 5
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    discovered = discover_all_tests(str(tmp_path))
    outcome = run_pytest_paths(str(tmp_path), discovered, timeout_seconds=60.0)
    assert outcome.ran
    assert outcome.returncode == 0, (
        f"pytest should import backend.* via PYTHONPATH; "
        f"stdout={outcome.stdout!r} stderr={outcome.stderr!r}"
    )



# ---------------------------------------------------------------------------
# env_manager: import-root → PyPI distribution-name resolution. A naive
# split on "." reduces ``import google.generativeai`` to ``google``,
# which is not a pip-installable distribution. The mapping translates
# the dotted form to ``google-generativeai`` so preflight installs the
# right wheel.
# ---------------------------------------------------------------------------
def test_extract_imports_python_captures_namespace_dotted_form():
    from cgx.codegen.env_manager import _extract_imports_python

    src = "import google.generativeai as genai\n"
    roots = _extract_imports_python(src)
    assert "google" in roots
    assert "google.generativeai" in roots


def test_extract_imports_python_captures_from_namespace_form():
    from cgx.codegen.env_manager import _extract_imports_python

    src = "from google.generativeai import GenerativeModel\n"
    roots = _extract_imports_python(src)
    assert "google" in roots
    assert "google.generativeai" in roots


def _force_import_miss(monkeypatch):
    """Make ``find_missing_python_packages``'s import-probe always miss.

    The function does a live ``__import__(name)`` to skip packages that
    happen to be installed in the current interpreter. Tests need a
    deterministic answer independent of which extras the contributor
    has locally, so we stub the probe to raise ``ImportError`` for
    every name.
    """
    import builtins
    real = builtins.__import__

    def _fake(name, *a, **kw):
        # Allow the cgx.codegen.env_manager module's own ``import ast``
        # / ``import json`` etc. to keep working — only fail on the
        # third-party names the function probes for.
        if name in {
            "PIL", "cv2", "sklearn", "bs4", "yaml", "skimage",
            "Crypto", "magic", "dateutil", "dotenv", "jose", "git",
            "OpenSSL", "serial", "usb",
            "google", "google.generativeai", "google.cloud",
            "google.oauth2", "google.auth", "google.api_core",
        }:
            raise ImportError(name)
        return real(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _fake)


def test_find_missing_packages_maps_google_generativeai(tmp_path, monkeypatch):
    from cgx.codegen.env_manager import find_missing_python_packages

    _force_import_miss(monkeypatch)
    (tmp_path / "requirements.txt").write_text("", encoding="utf-8")
    imports = {"google", "google.generativeai"}
    missing = find_missing_python_packages(imports, str(tmp_path))
    # Bare ``google`` namespace root is pruned; the dotted form resolves
    # to the proper PyPI name.
    assert "google" not in missing
    assert "google-generativeai" in missing


def test_find_missing_packages_maps_well_known_aliases(tmp_path, monkeypatch):
    from cgx.codegen.env_manager import find_missing_python_packages

    _force_import_miss(monkeypatch)
    (tmp_path / "requirements.txt").write_text("", encoding="utf-8")
    imports = {"PIL", "cv2", "sklearn", "bs4", "yaml"}
    missing = find_missing_python_packages(imports, str(tmp_path))
    # Every alias must be reported under its PyPI distribution name.
    assert "Pillow" in missing
    assert "opencv-python" in missing
    assert "scikit-learn" in missing
    assert "beautifulsoup4" in missing
    assert "PyYAML" in missing
    # And the import-time names must NOT appear (those aren't pip names).
    assert "PIL" not in missing
    assert "cv2" not in missing


def test_find_missing_packages_respects_declared_pypi_name(tmp_path, monkeypatch):
    from cgx.codegen.env_manager import find_missing_python_packages

    _force_import_miss(monkeypatch)
    # When the user already declared ``google-generativeai``, the agent
    # must NOT re-install it on a subsequent run.
    (tmp_path / "requirements.txt").write_text(
        "google-generativeai>=0.3.0\n", encoding="utf-8",
    )
    imports = {"google", "google.generativeai"}
    missing = find_missing_python_packages(imports, str(tmp_path))
    assert "google-generativeai" not in missing
