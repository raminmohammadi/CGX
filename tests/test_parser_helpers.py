"""Unit tests for the module-level AST helpers hoisted out of
``parse_codebase``.

These helpers were previously defined inside the ``parse_codebase`` closure
and therefore unreachable from any test. They are pure functions of their
inputs; the tests here pin their observable contracts so future refactors
cannot silently change record shapes.
"""

from __future__ import annotations

import ast

import pytest

from cgx.parser.parse_codebase import (
    _build_file_code_stub,
    _class_signature,
    _collect_top_level_members,
    _comments_by_line,
    _comments_in_span,
    _dotted_attr,
    _infer_type,
    _param_list,
    _parse_docstring,
    _signature_str,
    _unparse,
    _value_preview,
)


def _parse(src: str) -> ast.Module:
    return ast.parse(src)


# ---------- _unparse / _class_signature / _signature_str -----------------


def test_unparse_handles_none():
    assert _unparse(None) is None


def test_unparse_roundtrips_attribute():
    expr = ast.parse("foo.bar.baz", mode="eval").body
    assert _unparse(expr) == "foo.bar.baz"


def test_class_signature_with_bases():
    tree = _parse("class Foo(A, B.C): pass")
    cls = tree.body[0]
    assert _class_signature(cls) == "class Foo(A, B.C)"


def test_class_signature_without_bases():
    tree = _parse("class Bare: pass")
    cls = tree.body[0]
    assert _class_signature(cls) == "class Bare"


def test_signature_str_full_shapes():
    tree = _parse(
        "def f(a, b: int = 1, /, c: str = 'x', *args, d: float, e=2, **kw): pass"
    )
    fn = tree.body[0]
    sig = _signature_str(fn.args)
    assert sig.startswith("(") and sig.endswith(")")
    assert "/" in sig
    assert "*args" in sig and "**kw" in sig
    assert "d: float" in sig and "e=2" in sig


# ---------- _param_list ---------------------------------------------------


def test_param_list_kinds():
    tree = _parse("def f(a, /, b: int = 2, *args, c, **kw): pass")
    fn = tree.body[0]
    params = _param_list(fn.args)
    kinds = [p["kind"] for p in params]
    assert kinds == ["posonly", "pos_or_kw", "vararg", "kwonly", "kwarg"]
    b = next(p for p in params if p["name"] == "b")
    assert b["annotation"] == "int" and b["default"] == "2"


# ---------- _dotted_attr / _infer_type / _value_preview -------------------


def test_dotted_attr_recursive():
    expr = ast.parse("a.b.c", mode="eval").body
    assert _dotted_attr(expr) == "a.b.c"


@pytest.mark.parametrize(
    "src,expected",
    [
        ("1", "int"),
        ("'x'", "str"),
        ("None", "NoneType"),
        ("[1, 2]", "list"),
        ("(1,)", "tuple"),
        ("{1}", "set"),
        ("{'a': 1}", "dict"),
        ("foo()", "foo()"),
        ("x", "Symbol:x"),
        ("a.b", "Attr:a.b"),
    ],
)
def test_infer_type(src, expected):
    node = ast.parse(src, mode="eval").body
    assert _infer_type(node) == expected


def test_value_preview_truncates():
    node = ast.parse("'" + "x" * 500 + "'", mode="eval").body
    out = _value_preview(node, maxlen=20)
    assert out is not None and out.endswith("...") and len(out) == 20


# ---------- comments ------------------------------------------------------


def test_comments_by_line_and_span():
    src = "# top\nx = 1  # inline\n# mid\ny = 2\n"
    cmap = _comments_by_line(src)
    assert 1 in cmap and 2 in cmap and 3 in cmap
    span = _comments_in_span(cmap, 1, 2)
    assert any("top" in c for c in span)
    assert any("inline" in c for c in span)


# ---------- docstring parser ---------------------------------------------


def test_parse_docstring_google_style():
    doc = (
        "Adds two numbers.\n\n"
        "Args:\n    x (int): First number\n    y: Second number\n\n"
        "Returns:\n    int: The sum\n"
    )
    parsed = _parse_docstring(doc)
    assert parsed is not None
    assert parsed["summary"] == "Adds two numbers."
    names = [p["name"] for p in parsed["params"]]
    assert names == ["x", "y"]
    assert parsed["params"][0]["type"] == "int"
    assert parsed["returns"] == "int: The sum"


def test_parse_docstring_none_input():
    assert _parse_docstring(None) is None
    assert _parse_docstring("") is None


# ---------- top-level members + file stub --------------------------------


def test_collect_top_level_members_and_stub():
    src = (
        '"""mod doc."""\n'
        "from os import path\n"
        "VERSION: str = '1.0'\n"
        "def hello(name: str) -> None: ...\n"
        "class Foo(Base): ...\n"
    )
    tree = ast.parse(src)
    members = _collect_top_level_members(tree, src)
    assert [f["name"] for f in members["functions"]] == ["hello"]
    assert [c["name"] for c in members["classes"]] == ["Foo"]
    assert members["imports"] == ["from os import path"]
    assert members["globals"][0]["name"] == "VERSION"
    assert members["globals"][0]["annotation"] == "str"

    stub = _build_file_code_stub(ast.get_docstring(tree), members)
    assert '"""mod doc."""' in stub
    assert "from os import path" in stub
    # signature_str does not include the return annotation; only param list.
    assert "def hello(name: str): ..." in stub
    assert "class Foo(Base): ..." in stub
