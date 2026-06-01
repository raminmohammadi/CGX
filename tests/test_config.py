"""Tests for :mod:`cgx.config` (typed config dataclasses + env parsing)."""

from __future__ import annotations

import pytest

from cgx.config import (
    EmbeddingConfig, FaissConfig, HybridSearchConfig,
    _as_bool, _as_float, _as_int, _env,
)


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------
def test_as_int_accepts_strings_and_falls_back_to_default():
    assert _as_int("42", default=0) == 42
    assert _as_int("not-an-int", default=7) == 7
    assert _as_int(None, default=3) == 3
    assert _as_int(3.9, default=0) == 3


def test_as_float_accepts_strings_and_falls_back_to_default():
    assert _as_float("0.5", default=0.0) == 0.5
    assert _as_float("nope", default=1.5) == 1.5
    assert _as_float(None, default=2.0) == 2.0


@pytest.mark.parametrize("v,expected", [
    ("1", True), ("true", True), ("yes", True), ("Y", True), ("on", True),
    ("0", False), ("false", False), ("no", False), ("OFF", False),
    ("", False),  # falls back to default=False
    (True, True), (False, False),
])
def test_as_bool_recognises_common_strings(v, expected):
    assert _as_bool(v, default=False) is expected


def test_as_bool_returns_default_for_unknown_strings():
    assert _as_bool("maybe", default=True) is True
    assert _as_bool("idk", default=False) is False


def test_env_returns_default_when_unset(monkeypatch):
    monkeypatch.delenv("CGX_TEST_VAR_DOES_NOT_EXIST", raising=False)
    assert _env("CGX_TEST_VAR_DOES_NOT_EXIST", "fallback") == "fallback"


def test_env_treats_empty_string_as_unset(monkeypatch):
    monkeypatch.setenv("CGX_TEST_EMPTY", "")
    assert _env("CGX_TEST_EMPTY", "fallback") == "fallback"


def test_env_returns_set_value(monkeypatch):
    monkeypatch.setenv("CGX_TEST_VAR", "value")
    assert _env("CGX_TEST_VAR", "fallback") == "value"


# ---------------------------------------------------------------------------
# EmbeddingConfig
# ---------------------------------------------------------------------------
def test_embedding_config_uses_env_overrides(monkeypatch):
    monkeypatch.setenv("CGX_EMBED_MODEL", "custom-model")
    monkeypatch.setenv("CGX_EMBED_BATCH", "128")
    monkeypatch.setenv("CGX_EMBED_MAXLEN", "4096")
    cfg = EmbeddingConfig()
    assert cfg.model_name == "custom-model"
    assert cfg.batch_size == 128
    assert cfg.max_length == 4096


def test_embedding_config_from_overrides_applies_only_known_fields():
    cfg = EmbeddingConfig.from_overrides(batch_size=256, unknown_field="ignored")
    assert cfg.batch_size == 256
    assert not hasattr(cfg, "unknown_field")


def test_embedding_config_to_dict_roundtrips_fields():
    cfg = EmbeddingConfig(model_name="m", batch_size=8, max_length=128, device="cpu")
    d = cfg.to_dict()
    assert d["model_name"] == "m"
    assert d["batch_size"] == 8
    assert d["max_length"] == 128
    assert d["device"] == "cpu"


# ---------------------------------------------------------------------------
# FaissConfig
# ---------------------------------------------------------------------------
def test_faiss_config_defaults_when_env_unset(monkeypatch):
    for var in ("CGX_FAISS_METRIC", "CGX_FAISS_INDEX", "CGX_FAISS_NLIST",
                "CGX_FAISS_NPROBE", "CGX_FAISS_GPU"):
        monkeypatch.delenv(var, raising=False)
    cfg = FaissConfig()
    assert cfg.metric == "cosine"
    assert cfg.index == "flat"
    assert cfg.nlist == 1024
    assert cfg.use_gpu is False


def test_faiss_config_picks_up_hnsw_knobs(monkeypatch):
    monkeypatch.setenv("CGX_FAISS_INDEX", "hnsw")
    monkeypatch.setenv("CGX_FAISS_HNSW_M", "16")
    monkeypatch.setenv("CGX_FAISS_HNSW_EFS", "32")
    cfg = FaissConfig()
    assert cfg.index == "hnsw"
    assert cfg.M == 16
    assert cfg.efSearch == 32


def test_faiss_config_from_overrides_and_to_dict():
    cfg = FaissConfig.from_overrides(metric="l2", nlist=128)
    assert cfg.metric == "l2"
    assert cfg.nlist == 128
    assert cfg.to_dict()["metric"] == "l2"


# ---------------------------------------------------------------------------
# HybridSearchConfig
# ---------------------------------------------------------------------------
def test_hybrid_search_config_defaults():
    cfg = HybridSearchConfig.from_overrides()
    assert isinstance(cfg.top_k, int) and cfg.top_k > 0
    assert isinstance(cfg.rrf_k, float)
    assert isinstance(cfg.build_graph, bool)


def test_hybrid_search_config_env_overrides(monkeypatch):
    monkeypatch.setenv("CGX_TOP_K", "42")
    monkeypatch.setenv("CGX_RRF_K", "30")
    monkeypatch.setenv("CGX_BUILD_GRAPH", "0")
    cfg = HybridSearchConfig()
    assert cfg.top_k == 42
    assert cfg.rrf_k == 30.0
    assert cfg.build_graph is False


def test_hybrid_search_config_from_overrides_applies_known_fields_only():
    cfg = HybridSearchConfig.from_overrides(top_k=99, garbage="x")
    assert cfg.top_k == 99
    assert "garbage" not in cfg.to_dict()
