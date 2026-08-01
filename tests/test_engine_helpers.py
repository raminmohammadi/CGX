"""Tests for engine.py pure-Python helpers (no LLM / embedder)."""

import json
from typing import Any, Dict, List

from cgx.answer.engine import (
    SYSTEM_PROMPTS,
    _extract_json_object,
    _get_system_prompt,
    _normalize_scaffold_path,
    _parse_plan_freeform,
    _window_text,
    generate_project_scaffold,
)


def test_window_text_centers_on_focus():
    text = "\n".join(f"line {i}" for i in range(60))
    out = _window_text(text, ["line 30"], max_chars=200, context_lines=3)
    assert "line 30" in out
    assert "line 0" not in out  # window should not start at the top
    assert "line 59" not in out


def test_window_text_falls_back_when_no_match():
    text = "abc\ndef\nghi"
    out = _window_text(text, ["nope"], max_chars=100)
    assert out == text


def test_extract_json_object_balanced():
    text = 'prose {\n"a": 1, "b": "}"\n} trailing'
    obj = _extract_json_object(text)
    assert obj == {"a": 1, "b": "}"}


def test_extract_json_object_empty_on_garbage():
    assert _extract_json_object("not json at all") == {}


def test_get_system_prompt_known_and_fallback():
    for mode in SYSTEM_PROMPTS:
        assert isinstance(_get_system_prompt(mode), str)
    # unknown mode falls back to the default SYSTEM string
    default = _get_system_prompt("definitely-not-a-mode")
    assert "senior codebase assistant" in default.lower()


def test_parse_plan_freeform_extracts_diffs():
    text = (
        "## Plan\n"
        "Add a hello function.\n\n"
        "## Diffs\n"
        "```diff path=src/foo.py\n"
        "--- a/src/foo.py\n"
        "+++ b/src/foo.py\n"
        "@@\n"
        "+def hello():\n"
        "+    return 1\n"
        "```\n"
        "Cite as [[src/foo.py::function::hello]]"
    )
    parsed = _parse_plan_freeform(text)
    assert parsed["diffs"] and parsed["diffs"][0]["file"] == "src/foo.py"
    assert "def hello" in parsed["diffs"][0]["patch"]
    assert parsed["citations"] and parsed["citations"][0]["chunk_id"].startswith("src/foo.py")


# ---------------------------------------------------------------------------
# Scaffold path discipline
# ---------------------------------------------------------------------------
def test_normalize_scaffold_path_strips_project_name_prefix():
    # Weak LLMs frequently prepend a top-level project folder despite the
    # prompt explicitly forbidding it. The normaliser must strip the
    # stray prefix so APPLY lands files in the agreed root.
    assert _normalize_scaffold_path("calculator/src/App.jsx", None) == "src/App.jsx"
    assert _normalize_scaffold_path("my-app/backend/main.py", None) == "backend/main.py"


def test_normalize_scaffold_path_keeps_canonical_roots_untouched():
    for p in ("src/App.jsx", "backend/main.py", "tests/test_app.py",
              "public/index.html", "docs/README.md", "scripts/build.sh"):
        assert _normalize_scaffold_path(p, None) == p


def test_normalize_scaffold_path_honours_sibling_established_root():
    # If a sibling task already established a non-canonical top dir
    # (e.g. ``api/``), later tasks should extend it rather than have
    # their paths rewritten away from it.
    existing = ["api/server.py", "api/routes.py"]
    assert _normalize_scaffold_path("api/handlers.py", existing) == "api/handlers.py"


def test_normalize_scaffold_path_handles_leading_slashes_and_dots():
    assert _normalize_scaffold_path("./src/App.jsx", None) == "src/App.jsx"
    assert _normalize_scaffold_path("/src/App.jsx", None) == "src/App.jsx"
    # No slash means nothing to strip.
    assert _normalize_scaffold_path("README.md", None) == "README.md"


class _OneShotProvider:
    """Stub provider that returns a single canned chat reply."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: List[Dict[str, Any]] = []

    def chat(self, messages, **kw):  # noqa: ANN001 -- duck type
        self.calls.append({"messages": messages, **kw})
        return {"content": self.content}


def test_generate_project_scaffold_strips_prepended_project_folder():
    reply = json.dumps({
        "plan_md": "Calculator UI",
        "files": [
            {"path": "calculator/src/App.jsx", "content": "export default 1"},
            {"path": "calculator/package.json", "content": "{}"},
        ],
    })
    out = generate_project_scaffold("React calc", _OneShotProvider(reply))
    files = sorted(d["file"] for d in out["diffs"])
    assert files == ["package.json", "src/App.jsx"], files


def test_generate_project_scaffold_skips_existing_and_lists_them_in_prompt():
    reply = json.dumps({
        "plan_md": "Backend",
        "files": [
            # Already produced by sibling -- must be dropped.
            {"path": "src/App.jsx", "content": "export default 1"},
            {"path": "backend/main.py", "content": "print('ok')"},
        ],
    })
    provider = _OneShotProvider(reply)
    existing = ["src/App.jsx", "package.json"]
    out = generate_project_scaffold(
        "FastAPI backend", provider, existing_files=existing,
    )
    files = [d["file"] for d in out["diffs"]]
    assert files == ["backend/main.py"], files
    # The user prompt must surface the existing-files list so the LLM
    # has a chance to coordinate even before the post-filter runs.
    user_msg = provider.calls[0]["messages"][1]["content"]
    assert "EXISTING FILES" in user_msg
    assert "src/App.jsx" in user_msg



# ---------------------------------------------------------------------------
# _summarize_file_for_context and its language-specific helpers
# ---------------------------------------------------------------------------
from cgx.answer.engine import (  # noqa: E402
    _summarize_file_for_context,
    _summarize_json,
    _summarize_python,
    _summarize_textual,
)


def test_summarize_python_keeps_imports_and_signatures_elides_bodies():
    src = (
        "import os\n"
        "from typing import List\n"
        "\n"
        "CONST = 1\n"
        "\n"
        "def add(a: int, b: int) -> int:\n"
        "    # implementation detail the LLM does not need\n"
        "    result = a + b\n"
        "    return result\n"
        "\n"
        "class Calc:\n"
        "    def __init__(self) -> None:\n"
        "        self.x = 0\n"
        "    def total(self, n: int) -> int:\n"
        "        for _ in range(n):\n"
        "            self.x += 1\n"
        "        return self.x\n"
    )
    out = _summarize_python(src)
    assert "import os" in out
    assert "from typing import List" in out
    assert "CONST = 1" in out
    assert "def add(a: int, b: int) -> int:" in out
    assert "class Calc:" in out
    assert "def __init__(self) -> None:" in out
    assert "def total(self, n: int) -> int:" in out
    # Bodies must be replaced with an ellipsis, not the real implementation.
    assert "result = a + b" not in out
    assert "self.x += 1" not in out
    assert "..." in out


def test_summarize_python_returns_empty_on_syntax_error():
    assert _summarize_python("def broken(:\n  pass\n") == ""


def test_summarize_json_lists_top_level_keys():
    src = json.dumps({"name": "calc", "version": "0.1", "scripts": {"test": "x"}})
    out = _summarize_json(src)
    assert out.startswith("{")
    assert "'name'" in out and "'version'" in out and "'scripts'" in out
    # Implementation details (values) must not be echoed.
    assert "0.1" not in out and "calc" not in out


def test_summarize_json_handles_arrays_and_invalid():
    assert _summarize_json("[1, 2, 3]") == "[ array of 3 item(s) ]"
    assert _summarize_json("not json") == ""


def test_summarize_textual_extracts_jsx_signatures():
    src = (
        "import React from 'react'\n"
        "import { useState } from 'react'\n"
        "\n"
        "function Header(props) {\n"
        "  return <h1>{props.title}</h1>\n"
        "}\n"
        "\n"
        "export const Button = ({ onClick }) => {\n"
        "  const [count, setCount] = useState(0)\n"
        "  return <button onClick={onClick}>{count}</button>\n"
        "}\n"
        "\n"
        "export default function App() {\n"
        "  return <div />\n"
        "}\n"
    )
    out = _summarize_textual(src)
    assert "import React from 'react'" in out
    assert "import { useState } from 'react'" in out
    assert "function Header(props) {" in out
    assert "export const Button" in out
    assert "export default function App() {" in out
    # Bodies must not leak through.
    assert "<h1>" not in out
    assert "useState(0)" not in out


def test_summarize_file_for_context_dispatches_by_extension():
    py = _summarize_file_for_context("a.py", "def f():\n    return 1\n")
    assert "def f():" in py and "return 1" not in py

    js = _summarize_file_for_context("a.jsx", "export function X(){return 1}\n")
    assert "export function X()" in js

    js2 = _summarize_file_for_context("unknown.ext",
                                      "function foo() { return 1 }\n")
    assert "function foo()" in js2


def test_summarize_file_for_context_truncates_to_max_chars():
    big = "import x\n" + "\n".join(f"def f{i}(): pass" for i in range(500))
    out = _summarize_file_for_context("big.py", big, max_chars=300)
    assert len(out) <= 400  # cap + trailing marker
    assert "summary truncated" in out


def test_summarize_file_for_context_empty_input():
    assert _summarize_file_for_context("a.py", "") == ""


# ---------------------------------------------------------------------------
# _inject_python_package_inits: ensures every Python source directory in the
# manifest gets an ``__init__.py`` so pytest can import first-party modules
# without sys.path tricks. Excludes ``tests/`` (pytest convention) and
# root-level .py files (no parent dir).
# ---------------------------------------------------------------------------
def test_inject_python_package_inits_adds_marker_for_each_package_dir():
    from cgx.answer.engine import _inject_python_package_inits
    layers = [
        {"name": "backend", "files": [
            {"path": "backend/main.py", "description": "entry"},
            {"path": "backend/calculator.py", "description": "math"},
        ]},
        {"name": "tests", "files": [
            {"path": "tests/test_main.py", "description": "test"},
        ]},
    ]
    out = _inject_python_package_inits(layers)
    paths = [f["path"] for lay in out for f in (lay.get("files") or [])]
    # Source dir gets a marker; tests/ does NOT (pytest convention).
    assert "backend/__init__.py" in paths
    assert "tests/__init__.py" not in paths
    # Existing files are preserved untouched.
    assert "backend/main.py" in paths
    assert "tests/test_main.py" in paths


def test_inject_python_package_inits_walks_nested_packages():
    from cgx.answer.engine import _inject_python_package_inits
    layers = [
        {"name": "backend", "files": [
            {"path": "backend/utils/helpers.py", "description": "helpers"},
        ]},
    ]
    out = _inject_python_package_inits(layers)
    paths = [f["path"] for lay in out for f in (lay.get("files") or [])]
    # Both ancestor directories get markers.
    assert "backend/__init__.py" in paths
    assert "backend/utils/__init__.py" in paths


def test_inject_python_package_inits_is_idempotent():
    from cgx.answer.engine import _inject_python_package_inits
    layers = [
        {"name": "backend", "files": [
            {"path": "backend/__init__.py", "description": "marker"},
            {"path": "backend/main.py", "description": "entry"},
        ]},
    ]
    out = _inject_python_package_inits(layers)
    paths = [f["path"] for lay in out for f in (lay.get("files") or [])]
    # No duplicate marker injected.
    assert paths.count("backend/__init__.py") == 1


def test_inject_python_package_inits_skips_root_level_and_tests_subdirs():
    from cgx.answer.engine import _inject_python_package_inits
    layers = [
        {"name": "root", "files": [
            {"path": "manage.py", "description": "root entry"},
            {"path": "tests/backend/test_x.py", "description": "test"},
        ]},
    ]
    out = _inject_python_package_inits(layers)
    paths = [f["path"] for lay in out for f in (lay.get("files") or [])]
    # Root-level .py has no parent dir to mark; tests/* is excluded.
    assert "tests/__init__.py" not in paths
    assert "tests/backend/__init__.py" not in paths
    assert all(not p.endswith("__init__.py") for p in paths)


def test_inject_python_package_inits_noop_for_non_python_manifest():
    from cgx.answer.engine import _inject_python_package_inits
    layers = [
        {"name": "ui", "files": [
            {"path": "src/App.jsx", "description": "React"},
            {"path": "package.json", "description": "npm"},
        ]},
    ]
    out = _inject_python_package_inits(layers)
    paths = [f["path"] for lay in out for f in (lay.get("files") or [])]
    assert all(not p.endswith("__init__.py") for p in paths)


# ---------------------------------------------------------------------------
# generate_single_scaffold_file: __init__.py path short-circuits the LLM
# call and returns canned non-empty content so the Judge's "no content"
# gate passes.
# ---------------------------------------------------------------------------
def test_generate_single_scaffold_file_short_circuits_init_py():
    from cgx.answer.engine import generate_single_scaffold_file

    class _Boom:
        def chat(self, *a, **kw):
            raise AssertionError("provider must not be called for __init__.py")

    out = generate_single_scaffold_file(
        "backend/__init__.py", "package marker", _Boom(),
        layer="backend",
    )
    assert out["file"] == "backend/__init__.py"
    assert out["syntax_ok"] is True
    assert out["patch"], "patch must be non-empty for Judge to pass"
    assert out["content"].strip(), "content must be non-empty"
    assert "backend" in out["content"]



# ---------------------------------------------------------------------------
# _inject_python_package_inits: top-level src/ is a sys.path root in the
# standard "src layout", not a package -- so it must NOT get an
# __init__.py. Subpackages under src/ still do.
# ---------------------------------------------------------------------------
def test_inject_python_package_inits_skips_top_level_src():
    from cgx.answer.engine import _inject_python_package_inits
    layers = [
        {"name": "src", "files": [
            {"path": "src/app.py", "description": "entry"},
            {"path": "src/chat_manager.py", "description": "manager"},
        ]},
    ]
    out = _inject_python_package_inits(layers)
    paths = [f["path"] for lay in out for f in (lay.get("files") or [])]
    assert "src/__init__.py" not in paths


def test_inject_python_package_inits_marks_subpackages_under_src():
    from cgx.answer.engine import _inject_python_package_inits
    layers = [
        {"name": "src", "files": [
            {"path": "src/app.py", "description": "entry"},
            {"path": "src/models/user.py", "description": "model"},
            {"path": "src/services/db.py", "description": "service"},
        ]},
    ]
    out = _inject_python_package_inits(layers)
    paths = [f["path"] for lay in out for f in (lay.get("files") or [])]
    assert "src/__init__.py" not in paths
    # Subpackages still get their package marker.
    assert "src/models/__init__.py" in paths
    assert "src/services/__init__.py" in paths


# ---------------------------------------------------------------------------
# _inject_required_manifest_files: when a Python manifest places source
# under src/, a root conftest.py is injected so pytest can import the
# modules by their flat name.
# ---------------------------------------------------------------------------
def test_inject_required_files_adds_conftest_for_python_src_layout():
    from cgx.answer.engine import _inject_required_manifest_files
    layers = [
        {"name": "core", "files": [
            {"path": "src/app.py", "description": "entry"},
        ]},
        {"name": "tests", "files": [
            {"path": "tests/test_app.py", "description": "test"},
        ]},
    ]
    out = _inject_required_manifest_files(
        layers, goal="Build a python web app", skill_names=["python"],
    )
    paths = [f["path"] for lay in out for f in (lay.get("files") or [])]
    assert "conftest.py" in paths


def test_inject_required_files_skips_conftest_when_no_src_python():
    from cgx.answer.engine import _inject_required_manifest_files
    # Python backend without src/ layout -- no conftest.py needed because
    # the existing pytest convention already handles backend/ imports.
    layers = [
        {"name": "backend", "files": [
            {"path": "backend/main.py", "description": "entry"},
        ]},
    ]
    out = _inject_required_manifest_files(
        layers, goal="Build a python fastapi backend", skill_names=["python", "fastapi"],
    )
    paths = [f["path"] for lay in out for f in (lay.get("files") or [])]
    assert "conftest.py" not in paths


def test_inject_required_files_conftest_idempotent_when_already_present():
    from cgx.answer.engine import _inject_required_manifest_files
    layers = [
        {"name": "core", "files": [
            {"path": "src/app.py", "description": "entry"},
            {"path": "conftest.py", "description": "user-provided bootstrap"},
        ]},
    ]
    out = _inject_required_manifest_files(
        layers, goal="python web app", skill_names=["python"],
    )
    paths = [f["path"] for lay in out for f in (lay.get("files") or [])]
    assert paths.count("conftest.py") == 1


# ---------------------------------------------------------------------------
# _inject_readme: every project gets a top-level README.md, generated LAST
# so it can summarise the real, already-generated files.
# ---------------------------------------------------------------------------
def test_inject_readme_appends_when_absent():
    from cgx.answer.engine import _inject_readme
    layers = [
        {"name": "core", "files": [{"path": "src/app.py", "description": "entry"}]},
    ]
    out = _inject_readme(layers, goal="Build a thing")
    paths = [f["path"] for lay in out for f in (lay.get("files") or [])]
    assert paths[-1] == "README.md"
    assert out[-1]["name"] == "docs"


def test_inject_readme_moves_existing_readme_to_end():
    from cgx.answer.engine import _inject_readme
    layers = [
        {"name": "docs", "files": [{"path": "README.md", "description": "old"}]},
        {"name": "core", "files": [{"path": "src/app.py", "description": "entry"}]},
    ]
    out = _inject_readme(layers)
    paths = [f["path"] for lay in out for f in (lay.get("files") or [])]
    # Exactly one README, and it is the final file generated.
    assert paths.count("README.md") == 1
    assert paths[-1] == "README.md"


def test_inject_readme_is_idempotent():
    from cgx.answer.engine import _inject_readme
    layers = [{"name": "core", "files": [{"path": "src/app.py", "description": "e"}]}]
    once = _inject_readme(layers)
    twice = _inject_readme(once)
    paths = [f["path"] for lay in twice for f in (lay.get("files") or [])]
    assert paths.count("README.md") == 1
    assert paths[-1] == "README.md"


# ---------------------------------------------------------------------------
# _inject_required_test_file: every greenfield manifest must ship at least
# one test so the verify step has a runnable self-correction signal.
# ---------------------------------------------------------------------------
def test_inject_required_test_file_adds_pytest_smoke_for_python():
    from cgx.answer.engine import _inject_required_test_file
    layers = [
        {"name": "backend", "files": [
            {"path": "backend/main.py", "description": "FastAPI entry"},
        ]},
    ]
    out = _inject_required_test_file(
        layers, goal="build a python fastapi backend", skill_names=["python", "fastapi"],
    )
    paths = [f["path"] for lay in out for f in (lay.get("files") or [])]
    assert "tests/test_smoke.py" in paths


def test_inject_required_test_file_adds_js_test_for_react():
    from cgx.answer.engine import _inject_required_test_file
    layers = [
        {"name": "ui", "files": [
            {"path": "package.json", "description": "npm manifest"},
            {"path": "src/App.jsx", "description": "React App"},
        ]},
    ]
    out = _inject_required_test_file(
        layers, goal="build a react calculator ui", skill_names=["react"],
    )
    paths = [f["path"] for lay in out for f in (lay.get("files") or [])]
    # Extension mirrors the project's own source files (.jsx here).
    assert "tests/app.test.jsx" in paths


def test_inject_required_test_file_is_noop_when_test_present():
    from cgx.answer.engine import _inject_required_test_file
    layers = [
        {"name": "core", "files": [{"path": "src/app.py", "description": "entry"}]},
        {"name": "tests", "files": [
            {"path": "tests/test_app.py", "description": "existing test"},
        ]},
    ]
    out = _inject_required_test_file(layers, goal="python app", skill_names=["python"])
    paths = [f["path"] for lay in out for f in (lay.get("files") or [])]
    assert paths.count("tests/test_app.py") == 1
    assert "tests/test_smoke.py" not in paths


# ---------------------------------------------------------------------------
# _is_pytest_test_path / _has_collectable_pytest_test: the deterministic
# gate that stops an empty test suite (pytest exit 5) from stalling the
# verify->repair loop.
# ---------------------------------------------------------------------------
def test_is_pytest_test_path_matches_conventions():
    from cgx.answer.engine import _is_pytest_test_path
    assert _is_pytest_test_path("tests/test_app.py")
    assert _is_pytest_test_path("test_foo.py")
    assert _is_pytest_test_path("pkg/foo_test.py")
    # Non-tests: conftest, helpers, source, and non-Python files.
    assert not _is_pytest_test_path("tests/conftest.py")
    assert not _is_pytest_test_path("tests/helper.py")
    assert not _is_pytest_test_path("src/app.py")
    assert not _is_pytest_test_path("tests/test_app.txt")


def test_has_collectable_pytest_test_detects_top_level_function():
    from cgx.answer.engine import _has_collectable_pytest_test
    assert _has_collectable_pytest_test("def test_x():\n    assert True\n")
    assert _has_collectable_pytest_test("async def test_y():\n    assert 1\n")


def test_has_collectable_pytest_test_detects_test_class_method():
    from cgx.answer.engine import _has_collectable_pytest_test
    src = "class TestThing:\n    def test_a(self):\n        assert True\n"
    assert _has_collectable_pytest_test(src)


def test_has_collectable_pytest_test_rejects_app_logic_and_nested():
    from cgx.answer.engine import _has_collectable_pytest_test
    # App logic reimplemented under a test path -- the observed failure.
    app = ("def init_db():\n    return True\n\n"
           "def register(u, p):\n    return u\n")
    assert not _has_collectable_pytest_test(app)
    # test_* nested inside a fixture is invisible to pytest.
    nested = ("import pytest\n\n@pytest.fixture\ndef client():\n"
              "    def test_inner():\n        assert True\n    return 1\n")
    assert not _has_collectable_pytest_test(nested)
    # Unparseable module collects nothing.
    assert not _has_collectable_pytest_test("def (:\n")


class _QueueProvider:
    """Fake provider returning queued ``{"content": ...}`` JSON payloads."""

    model = "gpt-4o"  # cloud-tier so get_summary_budget stays generous

    def __init__(self, payloads: List[str]):
        self._payloads = list(payloads)
        self._payloads_last = payloads[-1] if payloads else ""
        self.calls = 0

    def chat(self, *args: Any, **kwargs: Any) -> Dict[str, str]:
        self.calls += 1
        body = self._payloads.pop(0) if self._payloads else self._payloads_last
        return {"content": json.dumps({"content": body})}


_APP_LOGIC = ("import sqlite3\n\n"
              "def init_db():\n    return True\n\n"
              "def register(u, p):\n    return u\n")
_REAL_TESTS = ("from app import init_db\n\n"
               "def test_init_db():\n    assert init_db() is True\n\n"
               "def test_register():\n    assert True\n\n"
               "def test_more():\n    assert 2 + 2 == 4\n")


def test_scaffold_test_file_retries_when_no_collectable_test():
    from cgx.answer.engine import generate_single_scaffold_file
    provider = _QueueProvider([_APP_LOGIC, _REAL_TESTS])
    out = generate_single_scaffold_file(
        "tests/test_app.py", "tests for the app", provider,
        layer="tests", goal="build an app",
    )
    assert provider.calls == 2, "gate must trigger exactly one retry"
    assert out["syntax_ok"] is True
    assert "def test_init_db" in out["content"]
    assert out["patch"]


def test_scaffold_test_file_flags_when_retry_still_empty():
    from cgx.answer.engine import generate_single_scaffold_file
    provider = _QueueProvider([_APP_LOGIC, _APP_LOGIC])
    out = generate_single_scaffold_file(
        "tests/test_app.py", "tests for the app", provider,
        layer="tests", goal="build an app",
    )
    assert provider.calls == 2
    assert out["syntax_ok"] is False
    assert "collectable" in out.get("syntax_error", "")


_BROKEN_PY = "def compute(a, b:\n    return a + b\n"  # missing ')'
_VALID_PY = "def compute(a, b):\n    return a + b\n"


def test_scaffold_py_syntax_error_retries_and_recovers():
    from cgx.answer.engine import generate_single_scaffold_file
    provider = _QueueProvider([_BROKEN_PY, _VALID_PY])
    out = generate_single_scaffold_file(
        "src/calculator.py", "calculator core", provider,
        layer="core", goal="build a calculator",
    )
    assert provider.calls == 2, "syntax gate must trigger one retry"
    assert out["syntax_ok"] is True
    assert out["content"] == _VALID_PY
    assert out["patch"]


def test_scaffold_py_syntax_error_flags_when_retry_still_broken():
    from cgx.answer.engine import generate_single_scaffold_file
    provider = _QueueProvider([_BROKEN_PY, _BROKEN_PY])
    out = generate_single_scaffold_file(
        "src/calculator.py", "calculator core", provider,
        layer="core", goal="build a calculator",
    )
    assert provider.calls == 2
    assert out["syntax_ok"] is False
    assert out.get("syntax_error")


def test_new_file_body_from_patch_roundtrips_and_rejects_modifications():
    from cgx.answer.engine import (
        _content_to_new_file_patch, _new_file_body_from_patch)
    body = "line 1\nline 2\n\nline 4"
    assert _new_file_body_from_patch(
        _content_to_new_file_patch("a/b.txt", body)) == body
    # An empty new file round-trips to an empty body.
    assert _new_file_body_from_patch(
        _content_to_new_file_patch("e.txt", "")) == ""
    # A modification diff (context/removal lines) is not losslessly
    # reversible into a whole-file body -> None.
    mod = ("--- a/x.py\n+++ b/x.py\n@@ -1,2 +1,2 @@\n"
           " keep\n-old\n+new\n")
    assert _new_file_body_from_patch(mod) is None
    # Not a diff at all, or empty -> None.
    assert _new_file_body_from_patch("just some text") is None
    assert _new_file_body_from_patch("") is None


class _FreeformFallbackProvider:
    """Empty primary JSON content -> forces the freeform fallback path."""

    stream_json_capable = False

    def __init__(self, freeform_body: str):
        self._freeform = freeform_body
        self.calls = 0

    def chat(self, *args: Any, **kwargs: Any) -> Dict[str, str]:
        self.calls += 1
        # Primary JSON-mode call yields no usable ``content`` -> fallback.
        if kwargs.get("force_json"):
            return {"content": "{}"}
        return {"content": self._freeform}


def test_scaffold_freeform_fallback_recovers_body_not_diff_header():
    """The freeform fallback must hand back the file body, not the wrapped
    unified diff -- otherwise the diff header leaks in as content and trips
    the 'content is a unified-diff header' guard (the public/index.html
    regen failure)."""
    from cgx.answer.engine import generate_single_scaffold_file
    html = ('<!doctype html>\n<html>\n  <head><title>Calc</title></head>\n'
            '  <body><div id="root"></div></body>\n</html>')
    freeform = "```html path=public/index.html\n" + html + "\n```"
    provider = _FreeformFallbackProvider(freeform)
    out = generate_single_scaffold_file(
        "public/index.html", "app entry html", provider,
        layer="frontend", goal="build a react app",
    )
    assert provider.calls == 2, "primary (empty) then freeform fallback"
    assert out["syntax_ok"] is True
    assert "unified-diff header" not in (out.get("syntax_error") or "")
    assert out["content"].startswith("<!doctype html>")
    assert "--- /dev/null" not in out["content"]
    assert 'id="root"' in out["content"]


def test_unwrap_wrapping_code_fence_variants():
    from cgx.answer.engine import _unwrap_wrapping_code_fence as unwrap
    # Whole-wrap with trailing prose after the closing fence.
    assert unwrap("```python\ndef f():\n    return 1\n```\n\nDone.") == (
        "def f():\n    return 1")
    # Unclosed fence (model forgot the closing ```).
    assert unwrap("```python\ndef f():\n    return 1") == (
        "def f():\n    return 1")
    # Short natural-language preamble before the opening fence.
    assert unwrap("Here is the file:\n```py\nx = 1\n```") == "x = 1"
    # No fence at all -> content is returned untouched.
    assert unwrap("x = 1\n") == "x = 1\n"
    # A docstring-embedded fence must NOT be stripped (guard on triple
    # quotes in the preamble) so real Python survives verbatim.
    doc = '"""Doc.\n\n```python\nnope\n```\n"""\nx = 1\n'
    assert unwrap(doc) == doc


def test_format_syntax_error_surfaces_offending_line():
    from cgx.answer.engine import _format_syntax_error
    e = SyntaxError("invalid syntax")
    e.lineno = 1
    e.text = "```python\n"
    msg = _format_syntax_error(e)
    assert "line 1" in msg
    assert "```python" in msg


def test_format_syntax_error_without_text_is_base():
    from cgx.answer.engine import _format_syntax_error
    assert _format_syntax_error(SyntaxError("invalid syntax")) == (
        str(SyntaxError("invalid syntax")))


def test_scaffold_fence_wrapped_python_parses_without_retry():
    """A fence-wrapped body is unwrapped before the syntax gate, so a
    valid file no longer burns a retry on a bogus line-1 error."""
    from cgx.answer.engine import generate_single_scaffold_file
    wrapped = "```python\n" + _VALID_PY + "```\n\nLet me know if you need more."
    provider = _QueueProvider([wrapped])
    out = generate_single_scaffold_file(
        "src/calculator.py", "calculator core", provider,
        layer="core", goal="build a calculator",
    )
    assert provider.calls == 1, "unwrap must avoid a line-1 syntax retry"
    assert out["syntax_ok"] is True
    assert out["content"] == _VALID_PY.rstrip("\n")
    assert out["patch"]


class _RecordingQueueProvider(_QueueProvider):
    """Queue provider that also records each call's ``messages`` list."""

    def __init__(self, payloads: List[str]):
        super().__init__(payloads)
        self.messages: List[Any] = []

    def chat(self, *args: Any, **kwargs: Any) -> Dict[str, str]:
        msgs = kwargs.get("messages")
        if msgs is None and args:
            msgs = args[0]
        self.messages.append(msgs)
        return super().chat(*args, **kwargs)


def test_scaffold_py_syntax_retry_prompt_is_minimal():
    """The targeted retry drops the prior-file digest, ships the broken file.

    The first attempt carries the ``ALREADY GENERATED FILES`` context so
    imports resolve; the syntax retry must instead send only the offending
    file plus the parser error (P1.1), reclaiming the O(files) digest tax.
    """
    from cgx.answer.engine import generate_single_scaffold_file
    prior = {"path": "src/db.py",
             "content": "def unique_prior_marker():\n    return 42\n"}
    provider = _RecordingQueueProvider([_BROKEN_PY, _VALID_PY])
    out = generate_single_scaffold_file(
        "src/calculator.py", "calculator core", provider,
        layer="core", goal="build a calculator",
        existing_files_with_content=[prior],
    )
    assert provider.calls == 2
    assert out["syntax_ok"] is True and out["content"] == _VALID_PY
    first_user = provider.messages[0][1]["content"]
    retry_user = provider.messages[1][1]["content"]
    # First attempt: full context (digest of the prior file present).
    assert "ALREADY GENERATED FILES" in first_user
    assert "unique_prior_marker" in first_user
    # Retry: minimal -- no digest block, just the broken bytes + error.
    assert "ALREADY GENERATED FILES" not in retry_user
    assert "unique_prior_marker" not in retry_user
    assert "BROKEN FILE:" in retry_user
    assert _BROKEN_PY.strip() in retry_user


def _chunk(text: str, n: int = 5) -> List[str]:
    """Split ``text`` into ``n`` roughly-equal pieces (fake stream deltas)."""
    step = max(1, len(text) // n)
    return [text[i:i + step] for i in range(0, len(text), step)]


class _StreamProvider:
    """Provider whose ``chat_stream`` yields queued JSON deltas.

    Advertises ``stream_json_capable`` so the scaffold generator takes the
    streaming path when handed an ``on_token``; ``chat`` is the blocking
    fallback and records its own call count.
    """

    model = "gpt-4o"  # cloud-tier so the summary budget stays generous
    stream_json_capable = True

    def __init__(self, stream_chunks: List[str], chat_body: str = ""):
        self._chunks = list(stream_chunks)
        self._chat_body = chat_body
        self.stream_calls = 0
        self.chat_calls = 0

    def chat_stream(self, messages: Any, temperature: float = 0.2,
                    max_tokens: Any = None, force_json: bool = False,
                    **kwargs: Any):
        self.stream_calls += 1
        assert force_json is True, "scaffold stream must request JSON mode"
        for chunk in self._chunks:
            yield chunk

    def chat(self, *args: Any, **kwargs: Any) -> Dict[str, str]:
        self.chat_calls += 1
        return {"content": json.dumps({"content": self._chat_body})}


def test_scaffold_streams_tokens_and_parses_result():
    """With ``on_token`` the file is streamed; deltas reach the callback."""
    from cgx.answer.engine import generate_single_scaffold_file
    body = json.dumps({"content": _VALID_PY})
    provider = _StreamProvider(_chunk(body))
    seen: List[str] = []
    out = generate_single_scaffold_file(
        "src/calculator.py", "calculator core", provider,
        layer="core", goal="build a calculator",
        on_token=seen.append,
    )
    assert provider.stream_calls == 1
    assert provider.chat_calls == 0, "streaming must not also block-call chat"
    assert out["syntax_ok"] is True and out["content"] == _VALID_PY
    assert "".join(seen) == body, "every delta must reach on_token in order"


def test_scaffold_stream_falls_back_to_chat_when_unparseable():
    """A stream that yields non-JSON degrades to the reliable chat call."""
    from cgx.answer.engine import generate_single_scaffold_file
    provider = _StreamProvider(["not json ", "at all"], chat_body=_VALID_PY)
    out = generate_single_scaffold_file(
        "src/calculator.py", "calculator core", provider,
        layer="core", goal="build a calculator",
        on_token=lambda _d: None,
    )
    assert provider.stream_calls == 1
    assert provider.chat_calls == 1, "unparseable stream must fall back"
    assert out["syntax_ok"] is True and out["content"] == _VALID_PY


def test_scaffold_without_on_token_never_streams():
    """No callback -> the legacy blocking path, even on a stream provider."""
    from cgx.answer.engine import generate_single_scaffold_file
    provider = _StreamProvider([], chat_body=_VALID_PY)
    out = generate_single_scaffold_file(
        "src/calculator.py", "calculator core", provider,
        layer="core", goal="build a calculator",
    )
    assert provider.stream_calls == 0
    assert provider.chat_calls == 1
    assert out["content"] == _VALID_PY


# Unbalanced JSX: the return expression is never closed. Parsed with the
# tree-sitter ``javascript`` grammar this reports has_error, which the
# inline gate turns into one hardened retry.
_BROKEN_JSX = (
    "import React from 'react'\n"
    "export default function App() {\n"
    "  return (\n"
    "    <div><h1>Calc</h1>\n"
    "}\n"
)
_VALID_JSX = (
    "import React from 'react'\n"
    "export default function App() {\n"
    "  return (\n"
    "    <div><h1>Calc</h1></div>\n"
    "  );\n"
    "}\n"
)


def test_scaffold_jsx_syntax_error_retries_and_recovers():
    from cgx.answer.engine import generate_single_scaffold_file
    provider = _QueueProvider([_BROKEN_JSX, _VALID_JSX])
    out = generate_single_scaffold_file(
        "src/App.jsx", "root component", provider,
        layer="ui", goal="build a react calculator",
    )
    assert provider.calls == 2, "JS syntax gate must trigger one retry"
    assert out["syntax_ok"] is True
    assert out["content"] == _VALID_JSX
    assert out["patch"]


def test_scaffold_jsx_syntax_error_flags_when_retry_still_broken():
    from cgx.answer.engine import generate_single_scaffold_file
    provider = _QueueProvider([_BROKEN_JSX, _BROKEN_JSX])
    out = generate_single_scaffold_file(
        "src/App.jsx", "root component", provider,
        layer="ui", goal="build a react calculator",
    )
    assert provider.calls == 2
    assert out["syntax_ok"] is False
    assert out.get("syntax_error")


_BAD_SYMBOL_TESTS = ("from app import generate_token\n\n"
                     "def test_x():\n    assert generate_token()\n")


def test_scaffold_retries_on_undefined_first_party_symbol():
    from cgx.answer.engine import generate_single_scaffold_file
    provider = _QueueProvider([_BAD_SYMBOL_TESTS, _REAL_TESTS])
    out = generate_single_scaffold_file(
        "tests/test_app.py", "tests for the app", provider,
        layer="tests", goal="build an app",
        existing_files_with_content=[{"path": "app.py", "content": _APP_LOGIC}],
    )
    assert provider.calls == 2, "symbol gate must trigger exactly one retry"
    assert out["syntax_ok"] is True
    assert "generate_token" not in out["content"]


def test_scaffold_flags_when_first_party_symbol_still_undefined():
    from cgx.answer.engine import generate_single_scaffold_file
    provider = _QueueProvider([_BAD_SYMBOL_TESTS, _BAD_SYMBOL_TESTS])
    out = generate_single_scaffold_file(
        "tests/test_app.py", "tests for the app", provider,
        layer="tests", goal="build an app",
        existing_files_with_content=[{"path": "app.py", "content": _APP_LOGIC}],
    )
    assert provider.calls == 2
    assert out["syntax_ok"] is False
    assert "generate_token" in out.get("syntax_error", "")


# ---------------------------------------------------------------------------
# generate_single_scaffold_file: conftest.py short-circuits the LLM and
# emits a deterministic sys.path-bootstrap body.
# ---------------------------------------------------------------------------
def test_generate_single_scaffold_file_short_circuits_conftest():
    from cgx.answer.engine import generate_single_scaffold_file

    class _Boom:
        def chat(self, *a, **kw):
            raise AssertionError("provider must not be called for conftest.py")

    out = generate_single_scaffold_file(
        "conftest.py", "pytest bootstrap", _Boom(),
    )
    assert out["file"] == "conftest.py"
    assert out["syntax_ok"] is True
    assert out["patch"], "patch must be non-empty"
    body = out["content"]
    assert "sys.path.insert" in body
    assert '"src"' in body or "'src'" in body


# ---------------------------------------------------------------------------
# generate_single_scaffold_file: pure-boilerplate files (.gitignore, ...)
# short-circuit the LLM with a deterministic template (P1.3).
# ---------------------------------------------------------------------------
def test_generate_single_scaffold_file_short_circuits_trivial_boilerplate():
    from cgx.answer.engine import generate_single_scaffold_file

    class _Boom:
        def chat(self, *a, **kw):
            raise AssertionError("provider must not be called for boilerplate")

    cases = {
        ".gitignore": "__pycache__/",
        ".dockerignore": ".git",
        ".gitattributes": "text=auto",
        ".editorconfig": "root = true",
    }
    for path, marker in cases.items():
        out = generate_single_scaffold_file(path, "boilerplate", _Boom())
        assert out["file"].rsplit("/", 1)[-1] == path
        assert out["syntax_ok"] is True
        assert out["patch"], "patch must be non-empty"
        assert marker in out["content"]


def test_generate_single_scaffold_file_depends_on_scopes_digest():
    """With ``depends_on`` the digest carries only the declared deps."""
    from cgx.answer.engine import generate_single_scaffold_file

    dep = {"path": "src/dep.py",
           "content": "def dep_marker_fn():\n    return 1\n"}
    other = {"path": "src/other.py",
             "content": "def other_marker_fn():\n    return 2\n"}

    scoped = _RecordingQueueProvider([_VALID_PY])
    generate_single_scaffold_file(
        "src/calculator.py", "calculator core", scoped,
        layer="core", goal="build a calculator",
        existing_files_with_content=[dep, other],
        depends_on=["src/dep.py"],
    )
    scoped_user = scoped.messages[0][1]["content"]
    assert "dep_marker_fn" in scoped_user
    assert "other_marker_fn" not in scoped_user

    # No depends_on -> legacy full-context digest (both siblings present).
    full = _RecordingQueueProvider([_VALID_PY])
    generate_single_scaffold_file(
        "src/calculator.py", "calculator core", full,
        layer="core", goal="build a calculator",
        existing_files_with_content=[dep, other],
    )
    full_user = full.messages[0][1]["content"]
    assert "dep_marker_fn" in full_user
    assert "other_marker_fn" in full_user


def test_trivial_boilerplate_content_returns_none_for_normal_files():
    from cgx.answer.engine import _trivial_boilerplate_content

    assert _trivial_boilerplate_content("src/app.py") is None
    assert _trivial_boilerplate_content("README.md") is None
    assert _trivial_boilerplate_content(".gitignore") is not None
    assert _trivial_boilerplate_content("nested/dir/.gitignore") is not None


# ---------------------------------------------------------------------------
# P0 contract-first planning: plan_scaffold_manifest emits a normalized
# ``contracts`` block and generate_single_scaffold_file threads it into the
# per-file prompt so cross-file interfaces are declared, not re-derived.
# ---------------------------------------------------------------------------
def test_plan_scaffold_manifest_normalizes_and_returns_contracts():
    from cgx.answer.engine import plan_scaffold_manifest

    reply = json.dumps({
        "plan_md": "flask + react calculator",
        "contracts": {
            "endpoints": [
                {"method": "post", "path": "/api/calc",
                 "request": {"expr": "str"},
                 "response": {"result": "number"},
                 "description": "evaluate an expression"},
                {},  # malformed (empty) -> dropped by the normalizer
            ],
            "schemas": [{"name": "CalcRequest", "fields": {"expr": "str"}}],
            "functions": [{"name": "evaluate",
                           "signature": "evaluate(expr: str) -> float",
                           "module": "backend/calc.py"}],
            "constants": [{"name": "API_BASE", "value": "/api"}],
            "junk": "unknown category ignored",
        },
        "layers": [{"name": "core", "files": [
            {"path": "backend/calc.py", "description": "core"}]}],
    })
    out = plan_scaffold_manifest(
        "build a calculator", _OneShotProvider(reply), goal="build a calculator")
    contracts = out["contracts"]
    assert set(contracts.keys()) == {
        "endpoints", "schemas", "functions", "constants"}
    # The empty endpoint dict is dropped; the well-formed one survives.
    assert len(contracts["endpoints"]) == 1
    assert contracts["endpoints"][0]["path"] == "/api/calc"
    # Unknown top-level categories never make it into the stored block.
    assert "junk" not in contracts


def test_generate_single_scaffold_file_threads_contracts_into_prompt():
    from cgx.answer.engine import generate_single_scaffold_file

    contracts = {
        "endpoints": [{"method": "GET", "path": "/api/ping",
                       "response": {"ok": "bool"}}],
        "functions": [{"signature": "ping() -> dict", "module": "app.py"}],
    }
    provider = _RecordingQueueProvider([_VALID_PY])
    generate_single_scaffold_file(
        "app.py", "flask app", provider,
        layer="core", goal="build an api", contracts=contracts,
    )
    user = provider.messages[0][1]["content"]
    assert "PROJECT CONTRACTS" in user
    assert "GET /api/ping" in user
    assert "ping() -> dict" in user

    # No contracts -> the section is absent (prompt unchanged).
    plain = _RecordingQueueProvider([_VALID_PY])
    generate_single_scaffold_file(
        "app.py", "flask app", plain,
        layer="core", goal="build an api",
    )
    assert "PROJECT CONTRACTS" not in plain.messages[0][1]["content"]


# ---------------------------------------------------------------------
# clarify_paths structured generation
# ---------------------------------------------------------------------


def _make_sources(n: int) -> List[Dict[str, Any]]:
    """Build ``n`` distinct (path, chunk_id, symbol) source rows."""
    out: List[Dict[str, Any]] = []
    for i in range(n):
        out.append({
            "chunk_id": f"/repo/src/mod_{i}.py::func::do_{i}",
            "path": f"src/mod_{i}.py",
            "symbol": f"do_{i}",
            "text": f"def do_{i}():\n    pass\n",
        })
    return out


def test_clarify_candidates_group_by_file_and_collect_symbols():
    from cgx.answer.engine import _clarify_candidates_from_sources
    sources = [
        {"chunk_id": "/a.py::f::foo", "path": "a.py", "symbol": "foo"},
        {"chunk_id": "/a.py::f::bar", "path": "a.py", "symbol": "bar"},
        {"chunk_id": "/b.py::f::baz", "path": "b.py", "symbol": "baz"},
    ]
    cands = _clarify_candidates_from_sources(sources)
    assert [c["path"] for c in cands] == ["a.py", "b.py"]
    # First chunk_id seen per file wins (preserves retrieval order).
    assert cands[0]["chunk_id"] == "/a.py::f::foo"
    assert "foo" in cands[0]["symbols"] and "bar" in cands[0]["symbols"]


def test_clarify_candidates_caps_at_max():
    from cgx.answer.engine import _clarify_candidates_from_sources
    sources = _make_sources(12)
    cands = _clarify_candidates_from_sources(sources, max_candidates=5)
    assert len(cands) == 5


def test_validate_clarify_options_filters_unknown_and_dedupes():
    from cgx.answer.engine import _validate_clarify_options
    candidates = [{"chunk_id": "A"}, {"chunk_id": "B"}, {"chunk_id": "C"}]
    raw = [
        {"title": "t1", "rationale": "r1", "chunk_id": "A"},
        {"title": "dup", "rationale": "r1b", "chunk_id": "A"},   # dedupe
        {"title": "t2", "rationale": "r2", "chunk_id": "B"},
        {"title": "bogus", "rationale": "r", "chunk_id": "ZZZ"},  # not allowed
        "not a dict",
        {"title": "t3", "rationale": "r3", "chunk_id": "C"},
    ]
    out = _validate_clarify_options(raw, candidates)
    assert [o["chunk_id"] for o in out] == ["A", "B", "C"]


def test_render_clarify_markdown_has_three_sections():
    from cgx.answer.engine import _render_clarify_markdown
    md = _render_clarify_markdown(
        "You want to improve indexing.",
        [
            {"title": "Add reranker", "rationale": "More precise.", "chunk_id": "X"},
            {"title": "Tune chunking", "rationale": "Better recall.", "chunk_id": "Y"},
            {"title": "Hybrid fusion", "rationale": "Stronger ranking.", "chunk_id": "Z"},
        ],
        "Which matters most: accuracy or latency?",
    )
    assert "You want to improve indexing." in md
    assert "**Possible directions:**" in md
    assert "1. **Add reranker**" in md and "[[X]]" in md
    assert "2. **Tune chunking**" in md and "[[Y]]" in md
    assert "3. **Hybrid fusion**" in md and "[[Z]]" in md
    assert "_Which matters most" in md


class _StubClarifyProvider:
    """Returns the queued JSON payloads in order from ``chat`` calls."""

    def __init__(self, payloads: List[Dict[str, Any]]) -> None:
        self._payloads = list(payloads)
        self.calls: List[List[Dict[str, str]]] = []

    def chat(self, messages, temperature=0.2, max_tokens=None,
             force_json=True, **_kwargs):
        self.calls.append(list(messages))
        if not self._payloads:
            return {"content": "{}"}
        return {"content": json.dumps(self._payloads.pop(0))}


def test_answer_clarify_paths_happy_path_renders_three_options():
    from cgx.answer.engine import _answer_clarify_paths
    sources = _make_sources(4)
    prep = {"sources": sources, "merged_hits": []}
    provider = _StubClarifyProvider([{
        "restatement": "You want to improve indexing accuracy.",
        "options": [
            {"title": "Better embedder",
             "rationale": "Use a stronger model.",
             "chunk_id": sources[0]["chunk_id"]},
            {"title": "Add reranker",
             "rationale": "Refine top-N.",
             "chunk_id": sources[1]["chunk_id"]},
            {"title": "Adjust chunking",
             "rationale": "Capture more context.",
             "chunk_id": sources[2]["chunk_id"]},
        ],
        "follow_up_question": "Accuracy or latency first?",
    }])
    out = _answer_clarify_paths(prep, "improve indexing accuracy", provider, root=None)
    assert "**Possible directions:**" in out["answer_md"]
    assert out["answer_md"].count("\n1. **") == 1
    assert "Better embedder" in out["answer_md"]
    assert "Add reranker" in out["answer_md"]
    assert "Adjust chunking" in out["answer_md"]
    assert "Accuracy or latency first?" in out["answer_md"]
    assert len(out["citations"]) == 3
    assert out["confidence"] >= 0.7
    assert len(provider.calls) == 1, "happy path must not retry"


def test_answer_clarify_paths_retries_when_first_reply_is_thin():
    from cgx.answer.engine import _answer_clarify_paths
    sources = _make_sources(5)
    prep = {"sources": sources, "merged_hits": []}
    # First reply: only 1 valid option (others reference unknown chunk_ids).
    # Second reply (retry): 3 valid options.
    provider = _StubClarifyProvider([
        {
            "restatement": "thin reply",
            "options": [
                {"title": "ok", "rationale": "r", "chunk_id": sources[0]["chunk_id"]},
                {"title": "bad", "rationale": "r", "chunk_id": "made-up"},
            ],
            "follow_up_question": "q?",
        },
        {
            "restatement": "fuller reply",
            "options": [
                {"title": "A", "rationale": "ra", "chunk_id": sources[0]["chunk_id"]},
                {"title": "B", "rationale": "rb", "chunk_id": sources[1]["chunk_id"]},
                {"title": "C", "rationale": "rc", "chunk_id": sources[2]["chunk_id"]},
            ],
            "follow_up_question": "which one?",
        },
    ])
    out = _answer_clarify_paths(prep, "open-ended goal", provider, root=None)
    assert len(provider.calls) == 2, "must retry exactly once on thin reply"
    assert len(out["citations"]) == 3
    assert "fuller reply" in out["answer_md"]


def test_answer_clarify_paths_backfills_from_candidates_when_model_fails_twice():
    from cgx.answer.engine import _answer_clarify_paths
    sources = _make_sources(4)
    prep = {"sources": sources, "merged_hits": []}
    # Both replies fail to produce any valid options.
    provider = _StubClarifyProvider([
        {"restatement": "", "options": [], "follow_up_question": ""},
        {"restatement": "", "options": "garbage", "follow_up_question": ""},
    ])
    out = _answer_clarify_paths(prep, "improve indexing", provider, root=None)
    assert len(provider.calls) == 2
    # Backfill must produce at least 3 deterministic options drawn from
    # the top candidates so the user sees concrete pointers even when
    # the model fails twice.
    assert len(out["citations"]) >= 3
    assert "**Possible directions:**" in out["answer_md"]
    for s in sources[:3]:
        assert s["chunk_id"] in out["answer_md"]
    # The deterministic backfill renders a generic "Review <stem>" title
    # so the user can distinguish it from a model-authored option.
    assert "Review" in out["answer_md"]


def test_answer_clarify_paths_empty_sources_degrades_gracefully():
    from cgx.answer.engine import _answer_clarify_paths
    prep = {"sources": [], "merged_hits": []}
    provider = _StubClarifyProvider([])  # must NOT be called
    out = _answer_clarify_paths(prep, "improve indexing", provider, root=None)
    assert provider.calls == []
    assert out["citations"] == []
    assert out["confidence"] <= 0.2
    assert "Re-index" in out["answer_md"] or "narrow" in out["answer_md"]
