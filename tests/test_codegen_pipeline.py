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
    out = rollback_from_backup(str(tmp_path), str(tmp_path / ".cgx-backups" / "missing"))
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


def test_partial_failure_logs_each_dropped_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    # A batch mixing one valid file with one that fails the syntax smoke
    # test must write the good file, drop the bad one, and -- critically --
    # name the dropped file plus its error in the log so a paused run is
    # diagnosable after the fact (previously only an aggregate count was
    # logged, leaving the dropped path unrecoverable).
    _make_project(tmp_path)
    good = textwrap.dedent(
        """
        --- /dev/null
        +++ b/pkg/good.py
        @@
        +def ok():
        +    return 1
        """
    ).lstrip()
    bad = textwrap.dedent(
        """
        --- /dev/null
        +++ b/pkg/bad.py
        @@
        +def oops(
        +    return 1
        """
    ).lstrip()

    with caplog.at_level("WARNING", logger="cgx.codegen.disk_apply"):
        res = apply_diffs_to_disk(str(tmp_path), [
            {"file": "pkg/good.py", "patch": good},
            {"file": "pkg/bad.py", "patch": bad},
        ])

    assert res["smoke_ok"] is False
    assert "pkg/good.py" in res["applied_files"]
    dropped = {f["file"] for f in res["failed_files"]}
    assert "pkg/bad.py" in dropped
    # The good file landed; the broken one was never written.
    assert (tmp_path / "pkg" / "good.py").exists()
    assert not (tmp_path / "pkg" / "bad.py").exists()
    # The dropped path + its error must appear in the warning stream.
    assert any(
        "dropped" in r.message and "pkg/bad.py" in r.getMessage()
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records]


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
    """Force the import probe to report every candidate as missing.

    ``find_missing_python_packages`` probes the target interpreter (the
    project venv) to skip packages that are already importable. Tests
    need a deterministic answer independent of which extras the
    contributor has installed locally, so we stub the probe to return an
    empty importable-set -- i.e. everything is missing.
    """
    from cgx.codegen import env_manager

    monkeypatch.setattr(
        env_manager, "_probe_importable",
        lambda names, python=None: set())


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


def test_find_missing_packages_skips_importable(tmp_path, monkeypatch):
    from cgx.codegen import env_manager

    # An already-importable package must NOT be reported (idempotency),
    # regardless of whether it is declared in requirements.txt.
    monkeypatch.setattr(
        env_manager, "_probe_importable",
        lambda names, python=None: {"google.generativeai"})
    (tmp_path / "requirements.txt").write_text(
        "google-generativeai>=0.3.0\n", encoding="utf-8",
    )
    imports = {"google", "google.generativeai"}
    missing = env_manager.find_missing_python_packages(imports, str(tmp_path))
    assert "google-generativeai" not in missing


def test_find_missing_packages_skips_nested_first_party(tmp_path, monkeypatch):
    from cgx.codegen.env_manager import find_missing_python_packages

    _force_import_miss(monkeypatch)
    (tmp_path / "requirements.txt").write_text("", encoding="utf-8")
    # ``main`` lives at backend/main.py -- a first-party module nested
    # under a package dir, not a flat or src-layout module.
    backend = tmp_path / "backend"
    backend.mkdir()
    (backend / "main.py").write_text("app = 1\n", encoding="utf-8")
    missing = find_missing_python_packages({"main"}, str(tmp_path))
    # It is not a PyPI distribution, so it must never be reported for
    # installation despite the empty requirements.txt.
    assert "main" not in missing


def test_find_missing_packages_ignores_venv_when_detecting_local(
        tmp_path, monkeypatch):
    from cgx.codegen.env_manager import find_missing_python_packages

    _force_import_miss(monkeypatch)
    (tmp_path / "requirements.txt").write_text("", encoding="utf-8")
    # A same-named module buried inside a pruned dir (.venv) must NOT be
    # treated as first-party -- ``flask`` here is a real dependency.
    vend = tmp_path / ".venv" / "lib" / "flask"
    vend.mkdir(parents=True)
    (vend / "flask.py").write_text("", encoding="utf-8")
    missing = find_missing_python_packages({"flask"}, str(tmp_path))
    assert "flask" in missing


def test_find_missing_packages_reports_declared_but_uninstalled(
        tmp_path, monkeypatch):
    from cgx.codegen import env_manager

    # requirements.txt declares flask, but it isn't importable in the
    # target venv -- e.g. a malformed/unresolvable line aborted the batch
    # ``pip install -r``. Importability is authoritative, so flask must
    # still be reported missing rather than silently trusted.
    monkeypatch.setattr(
        env_manager, "_probe_importable",
        lambda names, python=None: set())
    (tmp_path / "requirements.txt").write_text(
        "flask==2.3.2\n", encoding="utf-8")
    missing = env_manager.find_missing_python_packages(
        {"flask"}, str(tmp_path))
    assert "flask" in missing


def test_requirement_name_parses_specifiers_and_skips_flags():
    from cgx.codegen.env_manager import _requirement_name
    assert _requirement_name("flask==2.0.1") == "flask"
    assert _requirement_name("Flask-Cors>=3.0  # cors") == "Flask-Cors"
    assert _requirement_name("uvicorn[standard]==0.29.0") == "uvicorn"
    assert _requirement_name("# a comment") is None
    assert _requirement_name("-r base.txt") is None
    assert _requirement_name("   ") is None


def test_repin_requirements_pins_declared_to_installed_versions():
    from cgx.codegen.env_manager import _repin_requirements
    text = (
        "# pinned by the model\n"
        "flask==2.0.1\n"
        "flask-cors>=3.0.9  # cors extra\n"
        "gunicorn\n")
    installed = {"flask": "3.1.3", "flask_cors": "6.0.1",
                 "werkzeug": "3.1.8"}
    out = _repin_requirements(text, installed)
    lines = out.splitlines()
    # Comment preserved; stale pins rewritten to the resolved versions;
    # the inline comment survives; an undeclared installed dep is not added.
    assert lines[0] == "# pinned by the model"
    assert lines[1] == "flask==3.1.3"
    assert lines[2] == "flask-cors==6.0.1  # cors extra"
    # gunicorn has no installed version in the map -> left verbatim.
    assert lines[3] == "gunicorn"
    assert "werkzeug" not in out


def test_requirements_lock_marker_roundtrip(tmp_path):
    """mark_requirements_locked -> requirements_locked True; absent -> False."""
    from cgx.codegen.env_manager import (
        mark_requirements_locked, requirements_locked)
    root = str(tmp_path)
    assert requirements_locked(root) is False
    mark_requirements_locked(root)
    assert requirements_locked(root) is True
    # The marker lives under the hidden .cgx dir, never in the project tree.
    assert (tmp_path / ".cgx" / "requirements.locked").is_file()


def test_resolve_conflict_marks_requirements_locked(tmp_path, monkeypatch):
    """A successful conflict re-resolve re-pins AND marks the file env-locked.

    The lock marker is what lets a later whole-tree regenerate carry the
    resolved pins forward instead of re-emitting the model's stale manifest.
    """
    from cgx.codegen import env_manager
    (tmp_path / "requirements.txt").write_text(
        "flask==2.0.1\n", encoding="utf-8")

    class _Proc:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(env_manager.subprocess, "run",
                        lambda *a, **k: _Proc())
    monkeypatch.setattr(env_manager, "_pip_freeze_versions",
                        lambda py: {"flask": "3.1.3", "werkzeug": "3.1.8"})
    summary = env_manager.resolve_dependency_conflict(
        str(tmp_path), ["flask", "werkzeug"])
    assert summary["upgraded"] is True
    # The stale pin was rewritten to the resolved, conflict-free version...
    assert "flask==3.1.3" in (tmp_path / "requirements.txt").read_text()
    # ...and the file is now marked env-locked so a regenerate preserves it.
    assert env_manager.requirements_locked(str(tmp_path)) is True


# -- _apply_hunks context-verification regression tests -----------------

def _two_func_src() -> str:
    return textwrap.dedent(
        """
        def add(a, b):
            return a + b

        def sub(a, b):
            return a - b
        """
    ).lstrip()


def test_apply_hunks_exact_match_applies(tmp_path: Path) -> None:
    from cgx.codegen.diff_apply import PatchTarget, apply_diffs_in_memory

    (tmp_path / "m.py").write_text(_two_func_src(), encoding="utf-8")
    diff = textwrap.dedent(
        """
        @@ -1,2 +1,2 @@
         def add(a, b):
        -    return a + b
        +    return a + b + 1
        """
    ).strip()
    res = apply_diffs_in_memory(str(tmp_path), [PatchTarget(path="m.py", diff_text=diff)])[0]
    assert res.ok
    assert res.rejected_hunks == []
    assert "return a + b + 1" in (res.new_content or "")
    # The second function must be untouched.
    assert "def sub(a, b):" in (res.new_content or "")
    assert "return a - b" in (res.new_content or "")


def test_apply_hunks_drifted_line_numbers_fuzzy_locates(tmp_path: Path) -> None:
    from cgx.codegen.diff_apply import PatchTarget, apply_diffs_in_memory

    (tmp_path / "m.py").write_text(_two_func_src(), encoding="utf-8")
    # @@ line numbers are wildly wrong, but the context+deletion uniquely
    # identifies the sub() function.
    diff = textwrap.dedent(
        """
        @@ -42,2 +42,2 @@
         def sub(a, b):
        -    return a - b
        +    return a - b - 1
        """
    ).strip()
    res = apply_diffs_in_memory(str(tmp_path), [PatchTarget(path="m.py", diff_text=diff)])[0]
    assert res.ok, f"fuzzy locate should have rescued the drifted hunk: {res.error}"
    assert "return a - b - 1" in (res.new_content or "")
    assert "return a + b" in (res.new_content or "")


def test_apply_hunks_hallucinated_context_is_rejected(tmp_path: Path) -> None:
    from cgx.codegen.diff_apply import PatchTarget, apply_diffs_in_memory

    original = _two_func_src()
    (tmp_path / "m.py").write_text(original, encoding="utf-8")
    # Context lines that don't exist anywhere in the file: must NOT silently
    # overwrite real content (this was the pre-fix bug).
    diff = textwrap.dedent(
        """
        @@ -1,3 +1,3 @@
         def NONEXISTENT_func():
        -    return None
        +    return 'red'
        """
    ).strip()
    res = apply_diffs_in_memory(str(tmp_path), [PatchTarget(path="m.py", diff_text=diff)])[0]
    assert not res.ok
    assert len(res.rejected_hunks) == 1
    # Original content must be preserved byte-for-byte.
    assert res.new_content == original


def test_apply_hunks_ambiguous_match_is_rejected(tmp_path: Path) -> None:
    from cgx.codegen.diff_apply import PatchTarget, apply_diffs_in_memory

    # Two identical blocks; a hunk that matches both with wrong line numbers
    # is ambiguous -- must reject rather than guess.
    src = textwrap.dedent(
        """
        def a():
            return 1

        def b():
            return 1
        """
    ).lstrip()
    (tmp_path / "m.py").write_text(src, encoding="utf-8")
    diff = textwrap.dedent(
        """
        @@ -999,1 +999,1 @@
        -    return 1
        +    return 2
        """
    ).strip()
    res = apply_diffs_in_memory(str(tmp_path), [PatchTarget(path="m.py", diff_text=diff)])[0]
    assert not res.ok
    assert len(res.rejected_hunks) == 1
    assert res.new_content == src


# ---------------------------------------------------------------------------
# JS/TS/TSX syntax gate: tree-sitter-backed deterministic validation. Grammars
# ship with the ``parsers`` extra; skip the real-parse cases when absent so the
# suite still runs on a minimal install (the degradation case is exercised via
# monkeypatch below and needs no grammar).
# ---------------------------------------------------------------------------
def _require_grammar(language: str) -> None:
    from cgx.parser.treesitter_base import treesitter_available

    if not treesitter_available(language):
        pytest.skip(f"tree-sitter grammar for {language!r} unavailable")


def test_validate_js_source_accepts_valid() -> None:
    _require_grammar("javascript")
    from cgx.codegen.validate import validate_js_ts_source

    src = "export function add(a, b) {\n  return a + b;\n}\n"
    diag = validate_js_ts_source("src/add.js", src, "javascript")
    assert diag.ok
    assert diag.language == "javascript"


def test_validate_js_source_flags_syntax_error() -> None:
    _require_grammar("javascript")
    from cgx.codegen.validate import validate_js_ts_source

    # Unclosed function body.
    src = "export function add(a, b) {\n  return a + b;\n"
    diag = validate_js_ts_source("src/add.js", src, "javascript")
    assert not diag.ok
    assert diag.line is not None


def test_validate_tsx_accepts_jsx() -> None:
    _require_grammar("tsx")
    from cgx.codegen.validate import validate_js_ts_source

    src = (
        "export const App = (): JSX.Element => {\n"
        "  return <div className=\"x\">hi</div>;\n"
        "};\n"
    )
    diag = validate_js_ts_source("src/App.tsx", src, "tsx")
    assert diag.ok


def test_validate_ts_flags_syntax_error() -> None:
    _require_grammar("typescript")
    from cgx.codegen.validate import validate_js_ts_source

    src = "const x: number = ;\n"
    diag = validate_js_ts_source("src/x.ts", src, "typescript")
    assert not diag.ok


def test_validate_js_ts_source_degrades_when_grammar_unavailable(monkeypatch) -> None:
    # No ``parsers`` extra installed -> skip the gate rather than hard-fail,
    # mirroring the YAML validator when PyYAML is absent.
    import cgx.parser.treesitter_base as ts_base
    from cgx.codegen.validate import validate_js_ts_source

    monkeypatch.setattr(ts_base, "_get_ts_parser", lambda language: None)
    diag = validate_js_ts_source("src/add.js", "this is not js {{{", "javascript")
    assert diag.ok
    assert "unavailable" in (diag.error or "")


def test_validate_patch_results_routes_js_through_gate() -> None:
    _require_grammar("javascript")
    from cgx.codegen.diff_apply import PatchResult
    from cgx.codegen.validate import validate_patch_results

    good = PatchResult(path="src/ok.js", ok=True,
                       new_content="export const x = 1;\n")
    bad = PatchResult(path="src/bad.jsx", ok=True,
                      new_content="export function f() {\n  return (\n")
    diags = {d.path: d for d in validate_patch_results([good, bad])}
    assert diags["src/ok.js"].ok
    assert not diags["src/bad.jsx"].ok
    assert diags["src/bad.jsx"].language == "javascript"
