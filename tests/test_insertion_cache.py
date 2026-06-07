"""Tests for the exemplar-corpus cache in suggest_insertion_points.

The cache memoizes the per-records ``name + docstring`` matrix so repeated
queries against an unchanged records list don't re-encode the corpus. The
query embedding itself always runs (it depends on the question).
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

from cgx.retrieval import orchestrator
from cgx.retrieval.orchestrator import (
    _build_exemplar_corpus,
    _clear_insertion_corpus_cache,
    _insertion_corpus_key,
    suggest_insertion_points,
)


class CountingEmbedder:
    """Deterministic embedder that counts encode() invocations."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.calls: List[List[str]] = []

    def encode(self, texts: List[str]) -> np.ndarray:
        self.calls.append(list(texts))
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for j, ch in enumerate((t or " ").lower()):
                out[i, j % self.dim] += float(ord(ch) % 17) / 17.0
            n = float(np.linalg.norm(out[i])) + 1e-12
            out[i] /= n
        return out


def _records():
    return [
        {
            "id": "f1", "type": "file", "name": "alpha.py",
            "docstring": "alpha module", "schema_version": 3,
            "defines_children_ids": ["f1::function::go"],
        },
        {
            "id": "f1::class::Foo", "type": "class", "name": "Foo",
            "docstring": "the foo class", "schema_version": 3,
            "defines_children_ids": [],
        },
        {
            "id": "f1::function::go", "type": "function", "name": "go",
            "docstring": "do it", "schema_version": 3, "signature": "go()",
            "calls_in_count": 1, "parent_file_id": "f1",
        },
    ]


@pytest.fixture(autouse=True)
def _clear_cache():
    _clear_insertion_corpus_cache()
    yield
    _clear_insertion_corpus_cache()


def test_corpus_encoded_once_for_repeated_queries():
    recs = _records()
    emb = CountingEmbedder()
    suggest_insertion_points("first query", [{"chunk_id": "f1::function::go"}], recs, embedder=emb)
    n_after_first = len(emb.calls)
    suggest_insertion_points("second query", [{"chunk_id": "f1::function::go"}], recs, embedder=emb)
    n_after_second = len(emb.calls)
    # First call: 1 query encode + 1 corpus encode = 2.
    # Second call: 1 query encode only (corpus cached) = 1.
    assert n_after_first == 2
    assert n_after_second == 3


def test_cache_invalidates_when_records_list_replaced():
    emb = CountingEmbedder()
    suggest_insertion_points("q", [{"chunk_id": "f1::function::go"}], _records(), embedder=emb)
    # Different list object → different id() → cache miss → corpus re-encoded.
    suggest_insertion_points("q", [{"chunk_id": "f1::function::go"}], _records(), embedder=emb)
    # Two query encodes + two corpus encodes.
    assert len(emb.calls) == 4


def test_cache_invalidates_when_schema_version_differs():
    emb = CountingEmbedder()
    recs = _records()
    _build_exemplar_corpus(recs, emb)
    assert len(emb.calls) == 1
    for r in recs:
        r["schema_version"] = 99
    _build_exemplar_corpus(recs, emb)
    # Same list id but schema_version differs → cache miss.
    assert len(emb.calls) == 2


def test_cache_invalidates_when_embedder_differs():
    recs = _records()
    emb_a = CountingEmbedder()
    emb_b = CountingEmbedder()
    _build_exemplar_corpus(recs, emb_a)
    _build_exemplar_corpus(recs, emb_b)
    assert len(emb_a.calls) == 1
    assert len(emb_b.calls) == 1


def test_cache_bounded_fifo_eviction():
    emb = CountingEmbedder()
    # Fill the cache past its max with distinct records lists.
    recs_lists = [_records() for _ in range(orchestrator._INSERTION_CORPUS_CACHE_MAX + 2)]
    for recs in recs_lists:
        _build_exemplar_corpus(recs, emb)
    assert len(orchestrator._INSERTION_CORPUS_CACHE) == orchestrator._INSERTION_CORPUS_CACHE_MAX
    # Oldest two keys were evicted; encoding their records again is a miss.
    n_before = len(emb.calls)
    _build_exemplar_corpus(recs_lists[0], emb)
    assert len(emb.calls) == n_before + 1


def test_cache_key_shape_is_deterministic():
    recs = _records()
    emb = CountingEmbedder()
    k1 = _insertion_corpus_key(recs, emb)
    k2 = _insertion_corpus_key(recs, emb)
    assert k1 == k2
    assert len(k1) == 4
    assert k1[1] == len(recs)
    assert k1[2] == 3  # schema_version on the first record


def test_empty_corpus_caches_empty_result_no_redundant_encode():
    """Records with no file/class entries skip the encode entirely."""
    emb = CountingEmbedder()
    recs = [{"id": "fn", "type": "function", "name": "go", "docstring": "", "schema_version": 3}]
    mat, ids = _build_exemplar_corpus(recs, emb)
    assert ids == []
    assert mat.shape == (0, 0)
    # No encode call when corpus is empty.
    assert emb.calls == []
    _build_exemplar_corpus(recs, emb)
    assert emb.calls == []
