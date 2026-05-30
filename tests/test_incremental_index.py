"""Integration test for the incremental-indexing cache in ``run_index_auto``."""

from __future__ import annotations

import hashlib
import textwrap
from pathlib import Path
from typing import List

import numpy as np
import pytest

pytest.importorskip("faiss")

from cgx.pipeline.auto import run_index_auto  # noqa: E402


class CountingHashEmbedder:
    """Deterministic embedder that records every text it sees."""

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim
        self.encoded: List[str] = []

    def encode(self, texts: List[str]) -> np.ndarray:
        self.encoded.extend(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            h = hashlib.sha256((t or " ").encode("utf-8")).digest()
            for j in range(self.dim):
                out[i, j] = (h[j % len(h)] / 255.0) - 0.5
            n = np.linalg.norm(out[i]) + 1e-12
            out[i] /= n
        return out


def _write_project(root: Path, files: dict) -> None:
    for rel, body in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(textwrap.dedent(body), encoding="utf-8")


def test_second_index_run_hits_the_embedding_cache(tmp_path):
    project = tmp_path / "proj"
    out_dir = tmp_path / "out"
    _write_project(project, {
        "pkg/__init__.py": "",
        "pkg/a.py": '''
            def add(a, b):
                """sum two numbers"""
                return a + b
        ''',
        "pkg/b.py": '''
            def sub(a, b):
                """subtract"""
                return a - b
        ''',
    })

    emb = CountingHashEmbedder()
    result1 = run_index_auto(
        project_root=str(project), out_dir=str(out_dir),
        embedder=emb, model_name="fake-test",
    )
    first_call_count = len(emb.encoded)
    assert first_call_count > 0
    assert result1["incremental"] is True
    cache1 = result1["embedding_cache"]
    assert all(stats["misses"] > 0 and stats["hits"] == 0
               for stats in cache1.values())

    # Second run with no file changes: encoder must not be invoked again.
    emb.encoded.clear()
    result2 = run_index_auto(
        project_root=str(project), out_dir=str(out_dir),
        embedder=emb, model_name="fake-test",
    )
    assert emb.encoded == [], (
        "expected zero encoder calls on unchanged corpus, "
        f"got {len(emb.encoded)} texts")
    cache2 = result2["embedding_cache"]
    assert all(stats["misses"] == 0 and stats["hits"] > 0
               for stats in cache2.values())


def test_changing_one_file_only_reembeds_its_chunks(tmp_path):
    project = tmp_path / "proj"
    out_dir = tmp_path / "out"
    _write_project(project, {
        "pkg/__init__.py": "",
        "pkg/a.py": "def f(): return 1\n",
        "pkg/b.py": "def g(): return 2\n",
    })

    emb = CountingHashEmbedder()
    run_index_auto(project_root=str(project), out_dir=str(out_dir),
                   embedder=emb, model_name="fake-test")
    baseline_count = len(emb.encoded)
    assert baseline_count > 0
    emb.encoded.clear()

    # Modify only b.py.
    (project / "pkg" / "b.py").write_text("def g(): return 99\n", encoding="utf-8")
    result = run_index_auto(project_root=str(project), out_dir=str(out_dir),
                            embedder=emb, model_name="fake-test")

    # Far fewer texts than the original baseline; at least one cache miss
    # for the modified chunks.
    assert 0 < len(emb.encoded) < baseline_count
    cache = result["embedding_cache"]
    total_hits = sum(s["hits"] for s in cache.values())
    total_misses = sum(s["misses"] for s in cache.values())
    assert total_hits > 0 and total_misses > 0


def test_incremental_flag_can_be_disabled(tmp_path):
    project = tmp_path / "proj"
    out_dir = tmp_path / "out"
    _write_project(project, {"pkg/__init__.py": "", "pkg/a.py": "x = 1\n"})
    emb = CountingHashEmbedder()
    result = run_index_auto(project_root=str(project), out_dir=str(out_dir),
                            embedder=emb, model_name="fake-test",
                            incremental=False)
    assert result["incremental"] is False
    assert result["embedding_cache"] == {}
    # Disabling the cache should also leave no cache file on disk.
    assert not (out_dir / "emb_cache_intent.npz").exists()
    assert not (out_dir / "emb_cache_impl.npz").exists()
