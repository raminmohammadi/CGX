"""Tests for the BaseParser / PythonASTParser / registry seam.

Verifies that:

* ``BaseParser`` is an ABC with the expected contract;
* ``PythonASTParser`` registers ``.py`` and produces the same per-file
  chunks/calls that the project walker now aggregates;
* the ``_PARSER_REGISTRY`` dispatches by extension and silently skips
  unknown extensions (so the existing ``.py``-only behavior is preserved);
* the seam is robust to unparseable source.
"""

from __future__ import annotations

import os
import textwrap

import pytest

from cgx.parser.base import BaseParser
from cgx.parser.parse_codebase import _PARSER_REGISTRY, _parse_python_module, parse_codebase
from cgx.parser.python_parser import PythonASTParser
from cgx.parser.schema import CallRelation, CodeChunk  # noqa: F401  (import smoke)


# ---------- BaseParser contract -----------------------------------------


def test_base_parser_is_abstract():
    with pytest.raises(TypeError):
        BaseParser()  # type: ignore[abstract]


def test_base_parser_requires_parse_file():
    class Incomplete(BaseParser):
        extensions = (".py",)

    with pytest.raises(TypeError):
        Incomplete()  # type: ignore[abstract]


# ---------- Registry ----------------------------------------------------


def test_registry_contains_python_parser():
    parser = _PARSER_REGISTRY.get(".py")
    assert isinstance(parser, PythonASTParser)


def test_registry_keys_are_lowercase_with_dot():
    for ext in _PARSER_REGISTRY.keys():
        assert ext.startswith(".") and ext == ext.lower()


def test_python_parser_advertises_py_extension():
    assert ".py" in PythonASTParser.extensions


# ---------- PythonASTParser per-file output -----------------------------


_SAMPLE = textwrap.dedent(
    '''
    """Sample module."""

    import math


    class Calc:
        def add(self, x: int, y: int) -> int:
            return x + y


    def square_root(z: float) -> float:
        return math.sqrt(z)
    '''
).lstrip()


def test_python_parser_parse_file_returns_expected_chunk_types(tmp_path):
    fp = tmp_path / "example.py"
    fp.write_text(_SAMPLE)
    parser = PythonASTParser()
    chunks, calls = parser.parse_file(str(fp), _SAMPLE, str(tmp_path))
    types = sorted({c["type"] for c in chunks})
    # Methods are tagged "function" with meta.is_method=True (see visitor).
    assert types == ["class", "file", "function"]
    # Required identity fields are present on every chunk.
    for c in chunks:
        for key in ("id", "type", "name", "file", "module_path", "code",
                    "start_line", "end_line", "col_offset", "meta"):
            assert key in c, f"missing {key} on {c.get('id')}"
    # File chunk anchors at line 1.
    file_chunk = next(c for c in chunks if c["type"] == "file")
    assert file_chunk["start_line"] == 1
    assert file_chunk["end_line"] >= 1
    # Method chunks expose the parent class via meta.
    method = next(c for c in chunks if c["type"] == "function" and c["name"] == "add")
    assert method["meta"].get("is_method") is True
    assert method["meta"].get("class_name") == "Calc"
    # A call relation for math.sqrt was captured.
    callee_names = {(cr.get("callee_name"), cr.get("callee_fullname")) for cr in calls}
    assert ("sqrt", "math.sqrt") in callee_names


def test_python_parser_handles_syntax_errors_gracefully(tmp_path):
    fp = tmp_path / "broken.py"
    src = "def broken(:\n    pass\n"
    parser = PythonASTParser()
    chunks, calls = parser.parse_file(str(fp), src, str(tmp_path))
    assert chunks == [] and calls == []


def test_parse_python_module_matches_parser_parse_file(tmp_path):
    fp = tmp_path / "example.py"
    fp.write_text(_SAMPLE)
    direct = _parse_python_module(str(fp), _SAMPLE, str(tmp_path))
    via_parser = PythonASTParser().parse_file(str(fp), _SAMPLE, str(tmp_path))
    assert direct == via_parser  # exact equality, including ordering


# ---------- Project-level dispatch --------------------------------------


def test_parse_codebase_skips_files_without_registered_parser(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n")
    (tmp_path / "skip_me.txt").write_text("not source code")
    (tmp_path / "data.json").write_text('{"x": 1}')
    chunks, _calls = parse_codebase(str(tmp_path))
    files = {os.path.basename(c["file"]) for c in chunks if c["type"] == "file"}
    assert files == {"a.py"}


def test_parse_codebase_aggregates_across_files(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    return bar()\n")
    (tmp_path / "b.py").write_text("def bar():\n    return 2\n")
    chunks, calls = parse_codebase(str(tmp_path))
    # Both functions are aggregated from their respective files.
    fn_names = {c["name"] for c in chunks if c["type"] == "function"}
    assert fn_names == {"foo", "bar"}
    fn_files = {os.path.basename(c["file"]) for c in chunks if c["type"] == "function"}
    assert fn_files == {"a.py", "b.py"}
    # Call relations carry across files via the merged list.
    assert any(cr.get("callee_name") == "bar" for cr in calls)
    # Cross-file post-processing runs: foo gets calls_out_top attached.
    foo = next(c for c in chunks if c["type"] == "function" and c["name"] == "foo")
    assert "calls_out_top" in foo["meta"]
