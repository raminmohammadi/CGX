"""Tests for the AST-anchored insertion planner."""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

from cgx.codegen.ast_insert import (
    AstInsertSpec,
    build_unified_diff,
    plan_ast_insertion,
    plan_ast_insertion_from_suggestion,
)


def _write(root: Path, rel: str, body: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(body).lstrip("\n"), encoding="utf-8")
    return p


def test_module_insert_after_anchor(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/mod.py", """
        def alpha():
            return 1


        def gamma():
            return 3
    """)
    spec = AstInsertSpec(
        rel_path="pkg/mod.py",
        code="def beta():\n    return 2\n",
        anchor_symbol="alpha",
    )
    res = plan_ast_insertion(str(tmp_path), spec)
    assert res.ok and res.new_content
    assert res.new_content.index("def beta") < res.new_content.index("def gamma")
    assert res.new_content.index("def alpha") < res.new_content.index("def beta")
    ast.parse(res.new_content)


def test_module_insert_appends_when_anchor_missing(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/mod.py", "def a():\n    return 1\n")
    spec = AstInsertSpec(
        rel_path="pkg/mod.py",
        code="def b():\n    return 2\n",
        anchor_symbol="nonexistent",
    )
    res = plan_ast_insertion(str(tmp_path), spec)
    assert res.ok
    assert res.new_content.index("def a") < res.new_content.index("def b")


def test_class_insert_after_sibling_method(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/c.py", """
        class Foo:
            def one(self):
                return 1

            def three(self):
                return 3
    """)
    spec = AstInsertSpec(
        rel_path="pkg/c.py",
        code="def two(self):\n    return 2\n",
        class_name="Foo",
        anchor_symbol="one",
    )
    res = plan_ast_insertion(str(tmp_path), spec)
    assert res.ok, res.error
    tree = ast.parse(res.new_content)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    names = [getattr(n, "name", None) for n in cls.body]
    assert names == ["one", "two", "three"]


def test_dedupe_skips_existing_symbol(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/m.py", "def keep():\n    return 1\n")
    spec = AstInsertSpec(
        rel_path="pkg/m.py",
        code="def keep():\n    return 99\n",
    )
    res = plan_ast_insertion(str(tmp_path), spec)
    # nothing-to-do path returns ok=True with the original content unchanged
    assert res.ok
    assert res.new_content == res.original_content
    assert "already defined" in (res.error or "")


def test_non_python_path_is_rejected(tmp_path: Path) -> None:
    spec = AstInsertSpec(rel_path="README.md", code="x = 1\n")
    res = plan_ast_insertion(str(tmp_path), spec)
    assert not res.ok and "Python file" in (res.error or "")


def test_syntax_error_snippet(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/m.py", "def a():\n    return 1\n")
    spec = AstInsertSpec(rel_path="pkg/m.py", code="def broken(:\n")
    res = plan_ast_insertion(str(tmp_path), spec)
    assert not res.ok and "SyntaxError" in (res.error or "")


def test_new_file_creation(tmp_path: Path) -> None:
    spec = AstInsertSpec(
        rel_path="pkg/new.py",
        code="def hello():\n    return 'hi'\n",
    )
    res = plan_ast_insertion(str(tmp_path), spec)
    assert res.ok and res.is_new_file
    ast.parse(res.new_content)
    assert "def hello" in res.new_content


def test_class_not_found(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/m.py", "def a():\n    return 1\n")
    spec = AstInsertSpec(
        rel_path="pkg/m.py",
        code="def f(self):\n    pass\n",
        class_name="Missing",
    )
    res = plan_ast_insertion(str(tmp_path), spec)
    assert not res.ok and "Missing" in (res.error or "")


def test_comment_preservation(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/m.py", "def a():\n    return 1\n")
    spec = AstInsertSpec(
        rel_path="pkg/m.py",
        code="# important note\ndef b():\n    return 2\n",
        anchor_symbol="a",
    )
    res = plan_ast_insertion(str(tmp_path), spec)
    assert res.ok and "# important note" in res.new_content


def test_suggestion_bridge_class_container(tmp_path: Path) -> None:
    target = _write(tmp_path, "pkg/c.py", """
        class Bar:
            def one(self):
                return 1
    """)
    suggestion = {
        "container_type": "class",
        "container_id": f"{target}::class::Bar",
        "anchors": {
            "similar_signature_neighbor": f"{target}::method::Bar.one",
            "likely_caller": None,
        },
        "score": 0.9,
    }
    res = plan_ast_insertion_from_suggestion(
        str(tmp_path), suggestion, "def two(self):\n    return 2\n",
    )
    assert res.ok, res.error
    tree = ast.parse(res.new_content)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    assert [getattr(n, "name", None) for n in cls.body] == ["one", "two"]


def test_build_unified_diff_round_trip(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/m.py", "def a():\n    return 1\n")
    spec = AstInsertSpec(rel_path="pkg/m.py", code="def b():\n    return 2\n")
    res = plan_ast_insertion(str(tmp_path), spec)
    diff = build_unified_diff(res)
    assert diff.startswith("--- a/pkg/m.py")
    assert "+++ b/pkg/m.py" in diff
    assert "+def b():" in diff


def test_container_id_rejects_nested_class(tmp_path: Path) -> None:
    target = _write(tmp_path, "pkg/c.py", "class Outer:\n    class Inner:\n        pass\n")
    suggestion = {
        "container_type": "class",
        "container_id": f"{target}::class::Outer.Inner",
        "anchors": {"likely_caller": None, "similar_signature_neighbor": None},
        "score": 0.5,
    }
    res = plan_ast_insertion_from_suggestion(
        str(tmp_path), suggestion, "def x(self):\n    pass\n",
    )
    assert not res.ok


def test_line_anchored_class_insertion_uses_loc(tmp_path: Path) -> None:
    """anchor_loc.end_line drives the splice without re-walking the AST."""
    _write(tmp_path, "pkg/c.py", """
        class Foo:
            def one(self):
                return 1

            def three(self):
                return 3
    """)
    # ``one`` ends at line 3 in the dedented source (1-indexed).
    spec = AstInsertSpec(
        rel_path="pkg/c.py",
        code="def two(self):\n    return 2\n",
        class_name="Foo",
        anchor_symbol="one",
        anchor_loc={"start_line": 2, "end_line": 3, "indent_col": 4},
    )
    res = plan_ast_insertion(str(tmp_path), spec)
    assert res.ok, res.error
    tree = ast.parse(res.new_content)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    assert [getattr(n, "name", None) for n in cls.body] == ["one", "two", "three"]


def test_line_anchored_module_insertion_uses_loc(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/mod.py", """
        def alpha():
            return 1


        def gamma():
            return 3
    """)
    # ``alpha`` body ends at line 2 in the dedented file.
    spec = AstInsertSpec(
        rel_path="pkg/mod.py",
        code="def beta():\n    return 2\n",
        anchor_symbol="alpha",
        anchor_loc={"start_line": 1, "end_line": 2, "indent_col": 0},
    )
    res = plan_ast_insertion(str(tmp_path), spec)
    assert res.ok and res.new_content
    assert res.new_content.index("def alpha") < res.new_content.index("def beta")
    assert res.new_content.index("def beta") < res.new_content.index("def gamma")
    ast.parse(res.new_content)


def test_line_anchored_out_of_range_falls_back(tmp_path: Path) -> None:
    """An end_line beyond the file's line count must fall back to AST walk."""
    _write(tmp_path, "pkg/c.py", """
        class Foo:
            def one(self):
                return 1
    """)
    spec = AstInsertSpec(
        rel_path="pkg/c.py",
        code="def two(self):\n    return 2\n",
        class_name="Foo",
        anchor_symbol="one",
        anchor_loc={"start_line": 999, "end_line": 999, "indent_col": 4},
    )
    res = plan_ast_insertion(str(tmp_path), spec)
    assert res.ok, res.error
    tree = ast.parse(res.new_content)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    assert [getattr(n, "name", None) for n in cls.body] == ["one", "two"]


def test_suggestion_bridge_threads_loc_through(tmp_path: Path) -> None:
    """plan_ast_insertion_from_suggestion forwards similar_signature_neighbor_loc."""
    target = _write(tmp_path, "pkg/c.py", """
        class Bar:
            def one(self):
                return 1
    """)
    suggestion = {
        "container_type": "class",
        "container_id": f"{target}::class::Bar",
        "anchors": {
            "similar_signature_neighbor": f"{target}::method::Bar.one",
            "similar_signature_neighbor_loc": {
                "start_line": 2, "end_line": 3, "indent_col": 4,
            },
            "likely_caller": None,
            "likely_caller_loc": None,
        },
        "score": 0.9,
    }
    res = plan_ast_insertion_from_suggestion(
        str(tmp_path), suggestion, "def two(self):\n    return 2\n",
    )
    assert res.ok, res.error
    tree = ast.parse(res.new_content)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    assert [getattr(n, "name", None) for n in cls.body] == ["one", "two"]
