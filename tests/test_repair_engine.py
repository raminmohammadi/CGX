"""General failure localization + repair-context selection (repair_engine).

These are the language-agnostic primitives that replaced the verifier's
Python/pytest-specific error scanning: localize by planned-path appearance
(boundary-guarded), map a missing-module name to its planned file, and select a
focused, defect-centred repair context.
"""

from cgx.session import repair_engine as re_


# ---------------------- localize_paths ----------------------

def test_localize_matches_path_at_line_start():
    assert re_.localize_paths("app.py:1: NameError", ["app.py"]) == ["app.py"]


def test_localize_matches_planned_suffix_with_separator_boundary():
    # Runner prints tests/test_app.py; planned path is the basename test_app.py.
    out = "tests/test_app.py:4: in <module>\nTypeError: boom"
    got = re_.localize_paths(out, ["app.py", "test_app.py"])
    assert got == ["test_app.py"]          # app.py inside test_app.py is NOT matched


def test_localize_rejects_word_boundary_false_positive():
    # a.py must not match inside data.py.
    assert re_.localize_paths("data.py:2: error", ["a.py"]) == []


def test_localize_longest_first_suppresses_generic_suffix():
    out = "sub/app.py:3: Error"
    got = re_.localize_paths(out, ["app.py", "sub/app.py"])
    assert got == ["sub/app.py"]           # specific path claims the span


def test_localize_empty_when_no_path_named():
    assert re_.localize_paths("AssertionError: 0 == 2", ["app.py"]) == []


# ---------------------- python_module_targets ----------------------

def test_module_target_from_module_not_found():
    out = "ModuleNotFoundError: No module named 'app'"
    assert re_.python_module_targets(out, ["app.py"]) == ["app.py"]


def test_module_target_src_stripped_variant():
    out = "ModuleNotFoundError: No module named 'foo'"
    assert re_.python_module_targets(out, ["src/foo.py"]) == ["src/foo.py"]


def test_module_target_dotted_package():
    out = "ImportError: cannot import name X from 'pkg.sub'"
    assert re_.python_module_targets(out, ["pkg/sub.py"]) == ["pkg/sub.py"]


def test_module_target_empty_without_import_error():
    assert re_.python_module_targets("AssertionError: nope", ["app.py"]) == []


# ---------------------- neighbor_context ----------------------

_EXTS = (".py", ".js", ".jsx", ".ts", ".tsx")


def test_neighbor_localized_plus_depends_on():
    paths = ["src/base62.py", "src/store.py", "src/shortener.py",
             "tests/test_shortener.py"]
    specs = {"src/shortener.py": {"depends_on": ["src/store.py"]}}
    ctx = re_.neighbor_context(["src/shortener.py"], paths, specs, _EXTS)
    assert ctx == ["src/shortener.py", "src/store.py"]


def test_neighbor_includes_reverse_deps():
    # store.py is localized; shortener depends on it -> pull the caller in too.
    paths = ["src/store.py", "src/shortener.py"]
    specs = {"src/shortener.py": {"depends_on": ["src/store.py"]}}
    ctx = re_.neighbor_context(["src/store.py"], paths, specs, _EXTS)
    assert ctx == ["src/store.py", "src/shortener.py"]


def test_neighbor_falls_back_to_all_source_without_localization():
    paths = ["a.py", "b.py", "README.md", "requirements.txt"]
    assert re_.neighbor_context([], paths, {}, _EXTS) == ["a.py", "b.py"]


def test_neighbor_includes_js_sources():
    paths = ["backend/app.py", "frontend/src/App.jsx"]
    ctx = re_.neighbor_context([], paths, {}, _EXTS)
    assert "frontend/src/App.jsx" in ctx and "backend/app.py" in ctx
