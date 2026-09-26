"""Tests for repo-level context-file loading (CGX.md / AGENT.md)."""

from __future__ import annotations

import pytest

from cgx import context_files as cf


@pytest.fixture(autouse=True)
def _clear_cache():
    cf.clear_cache()
    yield
    cf.clear_cache()


def test_no_root_or_missing_file_returns_empty(tmp_path):
    assert cf.load_project_context(None) == ""
    assert cf.load_project_context(str(tmp_path)) == ""
    assert cf.find_context_file(str(tmp_path)) is None


def test_loads_cgx_md_at_root(tmp_path):
    (tmp_path / "CGX.md").write_text("Use tabs, not spaces.", encoding="utf-8")
    assert cf.load_project_context(str(tmp_path)) == "Use tabs, not spaces."
    assert cf.find_context_file(str(tmp_path)).name == "CGX.md"


def test_precedence_cgx_over_agent(tmp_path):
    (tmp_path / "CGX.md").write_text("cgx wins", encoding="utf-8")
    (tmp_path / "AGENT.md").write_text("agent loses", encoding="utf-8")
    assert cf.load_project_context(str(tmp_path)) == "cgx wins"


def test_falls_back_to_dot_cgx_dir(tmp_path):
    (tmp_path / ".cgx").mkdir()
    (tmp_path / ".cgx" / "CGX.md").write_text("hidden rules", encoding="utf-8")
    assert cf.load_project_context(str(tmp_path)) == "hidden rules"


def test_truncation_caps_length(tmp_path):
    (tmp_path / "CGX.md").write_text("x" * 100, encoding="utf-8")
    out = cf.load_project_context(str(tmp_path), max_chars=20)
    assert "truncated" in out
    assert len(out) < 100


def test_cache_invalidates_on_mtime_change(tmp_path):
    import os
    import time

    p = tmp_path / "CGX.md"
    p.write_text("first", encoding="utf-8")
    assert cf.load_project_context(str(tmp_path)) == "first"

    # Rewrite with a distinctly newer mtime so the cache refreshes.
    p.write_text("second", encoding="utf-8")
    os.utime(p, (time.time() + 10, time.time() + 10))
    assert cf.load_project_context(str(tmp_path)) == "second"


def test_default_context_path(tmp_path):
    assert cf.default_context_path(str(tmp_path)).name == "CGX.md"
