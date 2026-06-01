"""Tests for :mod:`cgx.webui.helpers` (provider builder + payload helpers)."""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import pytest

from cgx.webui.helpers import (
    build_provider, diffs_payload, json_safe, maybe_extract_zip,
    provider_from_profile_name, report_summary, stringify,
)


# ---------------------------------------------------------------------------
# build_provider
# ---------------------------------------------------------------------------
def test_build_provider_returns_ollama_for_ollama_kind():
    from cgx.answer.providers import OllamaProvider
    p = build_provider(kind="ollama", model="llama3", base_url="http://localhost:11434/v1")
    assert isinstance(p, OllamaProvider)
    assert p.model == "llama3"
    # OllamaProvider stores the base URL with any trailing slash stripped.
    assert "11434" in p.base_url
    assert not p.base_url.endswith("/")


def test_build_provider_returns_openai_compat_for_anything_else():
    from cgx.answer.providers import OpenAICompatProvider
    p = build_provider(kind="openai", model="gpt-4o-mini",
                       base_url="https://api.openai.com/v1", api_key="sk-test")
    assert isinstance(p, OpenAICompatProvider)
    assert p.model == "gpt-4o-mini"


def test_build_provider_handles_default_base_url_for_ollama():
    p = build_provider(kind="ollama", model="m", base_url="")
    assert "11434" in p.base_url


def test_build_provider_forwards_rate_limit_and_retries():
    p = build_provider(kind="openai", model="m", base_url="http://x", api_key=None,
                       rate_limit=2.0, max_retries=4)
    # The provider stores retries on the private ``_max_retries`` field and
    # constructs a RateLimiter when ``rate_limit`` is a positive number.
    assert getattr(p, "_max_retries", None) == 4
    assert getattr(p, "_limiter", None) is not None


# ---------------------------------------------------------------------------
# provider_from_profile_name
# ---------------------------------------------------------------------------
def test_provider_from_profile_name_raises_for_unknown():
    with pytest.raises(ValueError):
        provider_from_profile_name("__does_not_exist__profile__")


# ---------------------------------------------------------------------------
# maybe_extract_zip
# ---------------------------------------------------------------------------
def test_maybe_extract_zip_returns_none_for_missing_path():
    assert maybe_extract_zip(None) is None
    assert maybe_extract_zip("/tmp/this/should/not/exist.zip") is None


def test_maybe_extract_zip_returns_inner_dir_when_single_root(tmp_path):
    src = tmp_path / "proj"
    src.mkdir()
    (src / "hello.py").write_text("print('hi')\n")
    zip_path = tmp_path / "p.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(src / "hello.py", "proj/hello.py")
    out = maybe_extract_zip(str(zip_path))
    assert out is not None
    # Single top-level dir is unwrapped.
    assert os.path.basename(out) == "proj"
    assert (Path(out) / "hello.py").exists()


def test_maybe_extract_zip_returns_tmpdir_when_multiple_roots(tmp_path):
    zip_path = tmp_path / "p.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("a.txt", "a")
        zf.writestr("b.txt", "b")
    out = maybe_extract_zip(str(zip_path))
    assert out is not None
    assert (Path(out) / "a.txt").exists()
    assert (Path(out) / "b.txt").exists()


# ---------------------------------------------------------------------------
# json_safe
# ---------------------------------------------------------------------------
def test_json_safe_passes_through_jsonable_objects():
    assert json_safe({"a": 1, "b": [1, "two"]}) == {"a": 1, "b": [1, "two"]}
    assert json_safe(42) == 42


def test_json_safe_coerces_numpy_like_objects():
    class _Scalar:
        def item(self): return 7
    assert json_safe(_Scalar()) == 7


def test_json_safe_recurses_into_nested_unserializable():
    class _Opaque:
        def __repr__(self): return "<opaque>"
    out = json_safe({"obj": _Opaque(), "lst": [_Opaque(), 1]})
    assert out["obj"] == "<opaque>"
    assert out["lst"] == ["<opaque>", 1]


# ---------------------------------------------------------------------------
# stringify
# ---------------------------------------------------------------------------
def test_stringify_returns_string_passthrough():
    assert stringify("hello") == "hello"


def test_stringify_extracts_content_key_from_dict():
    assert stringify({"content": "abc", "ignored": 1}) == "abc"


def test_stringify_falls_back_to_json_dump_for_unstructured_dict():
    out = stringify({"foo": "bar"})
    assert "foo" in out and "bar" in out


def test_stringify_joins_list_entries_with_newlines():
    assert stringify(["a", "b", "c"]) == "a\nb\nc"


# ---------------------------------------------------------------------------
# diffs_payload + report_summary
# ---------------------------------------------------------------------------
def test_diffs_payload_normalises_path_and_diff_keys():
    diffs = [
        {"file": "a.py", "patch": "...patch..."},
        {"path": "b.py", "diff": "...diff..."},
        "ignored",  # non-dict entries are skipped
    ]
    out = diffs_payload(diffs)
    assert out == [
        {"file": "a.py", "patch": "...patch..."},
        {"file": "b.py", "patch": "...diff..."},
    ]


def test_diffs_payload_uses_unknown_when_path_missing():
    out = diffs_payload([{"patch": "x"}])
    assert out[0]["file"] == "(unknown)"


def test_report_summary_returns_none_for_empty_report():
    assert report_summary(None) is None
    assert report_summary({}) is None


def test_report_summary_filters_to_json_safe():
    class _Opaque:
        def __repr__(self): return "<x>"
    out = report_summary({"summary": {"obj": _Opaque()}, "ok": True})
    assert out["ok"] is True
    assert out["summary"]["obj"] == "<x>"
