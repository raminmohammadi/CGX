"""Unit tests for the symmetric sub-word tokenizer.

The tokenizer is the contract between the indexer and the BM25 querier, so
these tests pin both the splitting rules and the dedup/order semantics of
``expand_with_subwords``.
"""

from __future__ import annotations

import pytest

from cgx.retrieval.tokenize import (
    expand_with_subwords,
    split_identifier,
    tokenize_text,
)


@pytest.mark.parametrize(
    "name,expected",
    [
        ("databaseReconnect", ["database", "reconnect"]),
        ("parse_input_args", ["parse", "input", "args"]),
        ("Calculator", ["calculator"]),
        ("HTTPSConnection", ["https", "connection"]),
        ("URLParser", ["url", "parser"]),
        ("XMLHttpRequest", ["xml", "http", "request"]),
        ("snake_case_name", ["snake", "case", "name"]),
        ("Mixed_HTTPRequest", ["mixed", "http", "request"]),
        ("already.dotted/path-name", ["already", "dotted", "path", "name"]),
        ("ALLCAPS", ["allcaps"]),
        ("lower", ["lower"]),
        ("a", ["a"]),
        ("", []),
    ],
)
def test_split_identifier_cases(name, expected):
    assert split_identifier(name) == expected


def test_split_identifier_handles_none_gracefully():
    assert split_identifier(None) == []  # type: ignore[arg-type]


def test_expand_with_subwords_keeps_original_first():
    out = expand_with_subwords(["databaseReconnect"])
    # original lower-cased form leads, sub-words follow in left-to-right order
    assert out[0] == "databasereconnect"
    assert out[1:] == ["database", "reconnect"]


def test_expand_with_subwords_dedupes_stably():
    out = expand_with_subwords(["foo", "fooBar", "foo", "bar"])
    # ``foo`` (from first arg + sub-word of ``fooBar``) collapses to one entry,
    # and the second ``foo`` / the trailing ``bar`` add no duplicates either.
    assert out == ["foo", "foobar", "bar"]


def test_expand_with_subwords_min_len_drops_short_subwords_but_keeps_originals():
    out = expand_with_subwords(["a", "aBc"], min_len=2)
    # ``a`` is kept because the caller explicitly passed it (the rule is
    # documented as "originals always kept"). Sub-words of "aBc" -> "a", "bc";
    # "a" is below min_len so dropped (it's a sub-word, not an original);
    # "bc" meets min_len so kept.
    assert out == ["a", "abc", "bc"]


def test_tokenize_text_splits_runs_and_expands():
    out = tokenize_text("call databaseReconnect from parse_input_args!")
    # The runs are: ["call", "databaseReconnect", "from", "parse_input_args"]
    # Each is expanded; non-word chars (!) act as separators.
    assert out[:3] == ["call", "databasereconnect", "database"]
    assert "reconnect" in out
    assert "parse_input_args" in out
    assert "parse" in out and "input" in out and "args" in out


def test_tokenize_text_empty_input():
    assert tokenize_text("") == []
    assert tokenize_text(None) == []  # type: ignore[arg-type]


def test_tokenize_text_respects_min_len_for_subwords():
    # min_len=3 should drop two-letter sub-words but keep the original tokens
    out = tokenize_text("ifElseBlock", min_len=3)
    assert "ifelseblock" in out
    # ``if`` is a 2-letter sub-word -> dropped by min_len
    assert "if" not in out
    assert "else" in out and "block" in out
