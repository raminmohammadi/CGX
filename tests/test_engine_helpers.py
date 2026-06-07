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
