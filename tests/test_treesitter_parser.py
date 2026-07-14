"""Tests for the tree-sitter-backed JavaScript / TypeScript parsers.

These exercise the multi-language ingestion path added behind the
``BaseParser`` seam. They skip cleanly when the optional
``tree_sitter_language_pack`` dependency (the ``parsers`` extra) is not
installed, mirroring how the registry gates registration on availability.
"""

from __future__ import annotations

import os
import textwrap

import pytest

pytest.importorskip("tree_sitter_language_pack")

from cgx.parser.js_ts_parser import (  # noqa: E402
    JavaScriptParser,
    TSXParser,
    TypeScriptParser,
)
from cgx.parser.parse_codebase import _PARSER_REGISTRY, parse_codebase  # noqa: E402
from cgx.parser.treesitter_base import treesitter_available  # noqa: E402

_IDENTITY_KEYS = (
    "id", "type", "name", "file", "module_path", "code",
    "start_line", "end_line", "col_offset", "meta",
)

_JS = textwrap.dedent(
    """
    import { helper } from './util';

    export function greet(name, count = 1) {
      return helper(name);
    }

    const add = (a, b) => a + b;

    class Animal extends Base {
      constructor(n) { this.n = n; }
      speak() { return describe(this.n); }
      static async *make() { yield new Animal('x'); }
    }
    """
).lstrip()


def _by_type(chunks, t):
    return [c for c in chunks if c["type"] == t]


def test_javascript_parser_extracts_expected_chunks(tmp_path):
    fp = tmp_path / "sample.js"
    fp.write_text(_JS)
    chunks, calls = JavaScriptParser().parse_file(str(fp), _JS, str(tmp_path))

    types = sorted({c["type"] for c in chunks})
    assert types == ["class", "file", "function"]
    for c in chunks:
        for key in _IDENTITY_KEYS:
            assert key in c, f"missing {key} on {c.get('id')}"

    names = {c["name"] for c in _by_type(chunks, "function")}
    # top-level function, arrow-function const, and class methods.
    assert {"greet", "add", "constructor", "speak", "make"} <= names


def test_javascript_method_carries_class_and_kinds(tmp_path):
    fp = tmp_path / "sample.js"
    chunks, _calls = JavaScriptParser().parse_file(str(fp), _JS, str(tmp_path))
    speak = next(c for c in chunks
                 if c["type"] == "function" and c["name"] == "speak")
    assert speak["meta"]["is_method"] is True
    assert speak["meta"]["class_name"] == "Animal"
    assert "::method::Animal.speak" in speak["id"]

    make = next(c for c in chunks
                if c["type"] == "function" and c["name"] == "make")
    assert make["meta"]["is_async"] is True
    assert make["meta"]["is_generator"] is True
    assert make["meta"]["method_kind"] == "staticmethod"

    ctor = next(c for c in chunks
                if c["type"] == "function" and c["name"] == "constructor")
    assert ctor["meta"]["method_kind"] == "constructor"


def test_javascript_captures_call_relations(tmp_path):
    fp = tmp_path / "sample.js"
    chunks, calls = JavaScriptParser().parse_file(str(fp), _JS, str(tmp_path))
    callee_names = {cr.get("callee_name") for cr in calls}
    # ``helper`` (bare call) and ``describe`` (call inside a method) captured.
    assert {"helper", "describe"} <= callee_names
    # Every call relation is anchored to a real caller chunk id.
    chunk_ids = {c["id"] for c in chunks}
    for cr in calls:
        assert cr["caller_id"] in chunk_ids


def test_javascript_class_records_base(tmp_path):
    fp = tmp_path / "sample.js"
    chunks, _calls = JavaScriptParser().parse_file(str(fp), _JS, str(tmp_path))
    animal = next(c for c in chunks if c["type"] == "class")
    assert animal["name"] == "Animal"
    assert "Base" in animal["meta"]["bases"]


_TS = textwrap.dedent(
    """
    interface Shape { area(): number; }

    export function identity<T>(x: T): T { return x; }

    class Circle implements Shape {
      radius: number = 0;
      area(): number { return compute(this.radius); }
    }

    enum Color { Red, Green, Blue }
    """
).lstrip()


def test_typescript_parser_maps_interface_and_enum_to_classes(tmp_path):
    fp = tmp_path / "sample.ts"
    chunks, _calls = TypeScriptParser().parse_file(str(fp), _TS, str(tmp_path))
    class_names = {c["name"] for c in _by_type(chunks, "class")}
    # interface, class, and enum are all indexed as class-like containers.
    assert {"Shape", "Circle", "Color"} <= class_names

    identity = next(c for c in chunks
                    if c["type"] == "function" and c["name"] == "identity")
    # Generic function still yields a function chunk with a return type.
    assert identity["meta"]["returns_annotation"] is not None

    area = next(c for c in chunks
                if c["type"] == "function" and c["name"] == "area"
                and c["meta"].get("class_name") == "Circle")
    assert "::method::Circle.area" in area["id"]


def test_tsx_parser_handles_jsx(tmp_path):
    src = textwrap.dedent(
        """
        function App() {
          return <div>{render()}</div>;
        }
        """
    ).lstrip()
    fp = tmp_path / "App.tsx"
    chunks, calls = TSXParser().parse_file(str(fp), src, str(tmp_path))
    fn_names = {c["name"] for c in _by_type(chunks, "function")}
    assert "App" in fn_names
    assert any(cr.get("callee_name") == "render" for cr in calls)


# ---------- registry + project-level dispatch ---------------------------


def test_registry_registers_available_languages():
    if treesitter_available("javascript"):
        assert isinstance(_PARSER_REGISTRY.get(".js"), JavaScriptParser)
        assert isinstance(_PARSER_REGISTRY.get(".jsx"), JavaScriptParser)
    if treesitter_available("typescript"):
        assert isinstance(_PARSER_REGISTRY.get(".ts"), TypeScriptParser)
    if treesitter_available("tsx"):
        assert isinstance(_PARSER_REGISTRY.get(".tsx"), TSXParser)


def test_parse_codebase_mixes_python_and_javascript(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n")
    (tmp_path / "b.js").write_text("function bar() { return foo(); }\n")
    chunks, calls = parse_codebase(str(tmp_path))
    files = {os.path.basename(c["file"]) for c in chunks if c["type"] == "file"}
    assert files == {"a.py", "b.js"}
    fn_names = {c["name"] for c in chunks if c["type"] == "function"}
    assert {"foo", "bar"} <= fn_names
    assert any(cr.get("callee_name") == "foo" for cr in calls)
