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
from cgx.parser.markdown_parser import MarkdownParser
from cgx.parser.parse_codebase import _PARSER_REGISTRY, _parse_python_module, parse_codebase
from cgx.parser.python_parser import PythonASTParser
from cgx.parser.schema import CallRelation, CodeChunk  # noqa: F401  (import smoke)
from cgx.embeddings.records import make_index_records


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


# ---------- Markdown / RST documentation parser -------------------------


_README = textwrap.dedent(
    """
    # Project Title

    Intro paragraph describing the project.

    ## Installation

    Run ``pip install foo`` to install.

    ## Usage

    Import it and call ``foo.run()``.
    """
).lstrip()


def test_registry_contains_markdown_parser():
    for ext in (".md", ".markdown", ".mdx", ".rst"):
        assert isinstance(_PARSER_REGISTRY.get(ext), MarkdownParser)


def test_markdown_parser_chunks_by_heading(tmp_path):
    fp = tmp_path / "README.md"
    fp.write_text(_README)
    chunks, calls = MarkdownParser().parse_file(str(fp), _README, str(tmp_path))
    # No call graph for prose.
    assert calls == []
    # One file chunk plus one doc chunk per heading section.
    types = sorted({c["type"] for c in chunks})
    assert types == ["doc", "file"]
    doc_titles = {c["name"] for c in chunks if c["type"] == "doc"}
    assert {"Installation", "Usage"}.issubset(doc_titles)
    # Every emitted chunk is stamped with doc provenance.
    for c in chunks:
        assert c["meta"].get("source_kind") == "doc"
        for key in ("id", "type", "name", "file", "code",
                    "start_line", "end_line", "col_offset", "meta"):
            assert key in c, f"missing {key} on {c.get('id')}"
    # File chunk anchors at line 1 and lists the section titles.
    file_chunk = next(c for c in chunks if c["type"] == "file")
    assert file_chunk["start_line"] == 1
    assert "Installation" in file_chunk["code"]


def test_markdown_parser_handles_setext_headings(tmp_path):
    src = "Title\n=====\n\nbody line\n\nSub\n---\n\nmore\n"
    chunks, _calls = MarkdownParser().parse_file(str(tmp_path / "d.md"), src, str(tmp_path))
    titles = {c["name"] for c in chunks if c["type"] == "doc"}
    assert {"Title", "Sub"}.issubset(titles)


def test_markdown_parser_ignores_empty_source(tmp_path):
    chunks, calls = MarkdownParser().parse_file(str(tmp_path / "e.md"), "   \n\n", str(tmp_path))
    assert chunks == [] and calls == []


def test_parse_codebase_indexes_readme_and_prunes_vendored_docs(tmp_path):
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n")
    (tmp_path / "README.md").write_text(_README)
    # Vendored generated-site output must be pruned by DEFAULT_IGNORE_DIRS.
    site = tmp_path / "_site"
    site.mkdir()
    (site / "vendored.md").write_text("# Vendored\n\nnoise\n")
    chunks, _calls = parse_codebase(str(tmp_path))
    doc_files = {os.path.basename(c["file"]) for c in chunks
                 if c.get("meta", {}).get("source_kind") == "doc"}
    assert "README.md" in doc_files
    assert "vendored.md" not in doc_files


def test_source_kind_provenance_on_records(tmp_path):
    (tmp_path / "a.py").write_text('"""mod."""\n\ndef foo():\n    return 1\n')
    (tmp_path / "README.md").write_text(_README)
    chunks, _calls = parse_codebase(str(tmp_path))
    recs = make_index_records(chunks, G=None)
    by_kind = {}
    for r in recs:
        by_kind.setdefault(r.get("source_kind"), set()).add(os.path.basename(r.get("file") or ""))
    # Code records carry "code"; doc records carry "doc".
    assert "README.md" in by_kind.get("doc", set())
    assert "a.py" in by_kind.get("code", set())
    # Every record has an explicit provenance (no None leaking through).
    assert all(r.get("source_kind") in ("code", "doc") for r in recs)
