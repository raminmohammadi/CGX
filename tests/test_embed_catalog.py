"""Tests for :mod:`cgx.embeddings.catalog`."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from cgx.embeddings import catalog as cat


def test_catalog_contains_expected_models():
    names = [m.name for m in cat.EMBED_MODEL_CATALOG]
    assert "jinaai/jina-embeddings-v2-base-code" in names
    assert "BAAI/bge-m3" in names
    assert "Qwen/Qwen3-Embedding-8B" in names


def test_catalog_entries_have_required_metadata():
    for m in cat.EMBED_MODEL_CATALOG:
        assert m.name and m.label and m.description
        assert m.dim > 0
        assert m.max_tokens > 0
        assert m.size_gb > 0
        assert m.kind == "huggingface"


def test_find_by_name_returns_entry_or_none():
    hit = cat.find_by_name("BAAI/bge-m3")
    assert hit is not None and hit.label.startswith("BGE-M3")
    assert cat.find_by_name("does/not-exist") is None


# ---------------------------------------------------------------------------
# is_cached / _hf_cache_root
# ---------------------------------------------------------------------------
def _make_snapshot(cache_root: Path, model_name: str) -> Path:
    """Create the on-disk layout HF Hub uses for a downloaded snapshot."""
    safe = "models--" + model_name.replace("/", "--")
    snap = cache_root / safe / "snapshots" / "deadbeef"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    return snap


def test_is_cached_false_when_no_snapshot(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "hub"))
    assert cat.is_cached("BAAI/bge-m3") is False


def test_is_cached_true_when_snapshot_present(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    hub = tmp_path / "hub"
    hub.mkdir()
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    _make_snapshot(hub, "BAAI/bge-m3")
    assert cat.is_cached("BAAI/bge-m3") is True


def test_is_cached_false_when_snapshots_dir_is_empty(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    hub = tmp_path / "hub"
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(hub))
    safe = "models--BAAI--bge-m3"
    (hub / safe / "snapshots").mkdir(parents=True)
    assert cat.is_cached("BAAI/bge-m3") is False


def test_hf_cache_root_honors_hf_home_over_hub_cache(tmp_path, monkeypatch):
    """HF_HOME wins over HUGGINGFACE_HUB_CACHE and gets a ``/hub`` suffix."""
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "other"))
    root = cat._hf_cache_root()
    assert root == tmp_path / "hf" / "hub"


def test_hf_cache_root_defaults_to_user_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    root = cat._hf_cache_root()
    # Default lives under ``~/.cache/huggingface/hub`` — only assert the suffix
    # so the test doesn't depend on the runner's $HOME.
    assert root.parts[-3:] == (".cache", "huggingface", "hub")
