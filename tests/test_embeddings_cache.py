"""Tests for the content-addressed embedding cache."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from cgx.embeddings.cache import embed_with_cache, hash_text, load_cache, save_cache


def _fake_encode_factory():
    """Deterministic 8-d fake encoder + call counter."""
    calls = {"n": 0, "batches": []}

    def encode(texts):
        calls["n"] += len(texts)
        calls["batches"].append(list(texts))
        out = np.zeros((len(texts), 8), dtype=np.float32)
        for i, t in enumerate(texts):
            h = hashlib.sha1(t.encode("utf-8")).digest()
            for j in range(8):
                out[i, j] = (h[j] / 255.0) - 0.5
        return out

    return encode, calls


def test_hash_text_is_stable():
    assert hash_text("foo") == hash_text("foo")
    assert hash_text("foo") != hash_text("bar")


def test_first_call_misses_then_subsequent_call_hits(tmp_path):
    cache_path = str(tmp_path / "emb.npz")
    encode, calls = _fake_encode_factory()
    texts = ["alpha", "beta", "gamma"]

    embs1, stats1 = embed_with_cache(
        texts, encode_fn=encode, cache_path=cache_path,
        model_name="fake", normalize=False)
    assert embs1.shape == (3, 8)
    assert stats1 == {"hits": 0, "misses": 3, "dim": 8}
    assert calls["n"] == 3

    # Second call with the same texts should not touch the encoder.
    embs2, stats2 = embed_with_cache(
        texts, encode_fn=encode, cache_path=cache_path,
        model_name="fake", normalize=False)
    assert np.allclose(embs1, embs2)
    assert stats2 == {"hits": 3, "misses": 0, "dim": 8}
    assert calls["n"] == 3


def test_partial_overlap_only_embeds_new_rows(tmp_path):
    cache_path = str(tmp_path / "emb.npz")
    encode, calls = _fake_encode_factory()

    embed_with_cache(["a", "b"], encode_fn=encode, cache_path=cache_path,
                     model_name="fake", normalize=False)
    calls["batches"].clear()

    embs, stats = embed_with_cache(
        ["a", "c", "b", "d"], encode_fn=encode, cache_path=cache_path,
        model_name="fake", normalize=False)
    assert stats["hits"] == 2 and stats["misses"] == 2
    assert embs.shape == (4, 8)
    # The encoder must have been called with exactly the new texts.
    assert calls["batches"] == [["c", "d"]]


def test_changing_model_name_invalidates_cache(tmp_path):
    cache_path = str(tmp_path / "emb.npz")
    encode, calls = _fake_encode_factory()
    embed_with_cache(["x"], encode_fn=encode, cache_path=cache_path,
                     model_name="fake-v1", normalize=False)
    calls["batches"].clear()

    embed_with_cache(["x"], encode_fn=encode, cache_path=cache_path,
                     model_name="fake-v2", normalize=False)
    # The new model_name must force a re-embed.
    assert calls["batches"] == [["x"]]


def test_load_cache_returns_empty_when_meta_mismatch(tmp_path):
    cache_path = str(tmp_path / "emb.npz")
    save_cache(cache_path, {"k": np.zeros(4, dtype=np.float32)},
               model_name="m", dim=4, normalize=False)
    # Wrong dim.
    assert load_cache(cache_path, expected_meta={
        "version": 1, "model_name": "m", "dim": 8, "normalize": False}) == {}
    # Right dim, wrong model.
    assert load_cache(cache_path, expected_meta={
        "version": 1, "model_name": "n", "dim": 4, "normalize": False}) == {}
    # Matching meta.
    got = load_cache(cache_path, expected_meta={
        "version": 1, "model_name": "m", "dim": 4, "normalize": False})
    assert list(got.keys()) == ["k"]


def test_load_cache_missing_file_is_empty_not_error(tmp_path):
    assert load_cache(str(tmp_path / "nope.npz"),
                      expected_meta={"version": 1, "model_name": "x",
                                     "dim": 1, "normalize": False}) == {}


def test_encode_fn_bad_shape_raises(tmp_path):
    cache_path = str(tmp_path / "emb.npz")

    def bad_encode(texts):
        return np.zeros((len(texts) + 1, 4), dtype=np.float32)

    with pytest.raises(RuntimeError):
        embed_with_cache(["a"], encode_fn=bad_encode, cache_path=cache_path,
                         model_name="x", normalize=False)
