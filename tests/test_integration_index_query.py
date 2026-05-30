"""End-to-end smoke test: parse -> index -> query, using a deterministic
fake embedder so no model download or GPU is required.

Skipped automatically if ``faiss`` (required by the indexer) is not installed
in the test environment.
"""

from __future__ import annotations

import hashlib
import textwrap
from pathlib import Path
from typing import List

import numpy as np
import pytest

pytest.importorskip("faiss")

from cgx.pipeline.auto import run_index_auto, run_query_auto  # noqa: E402


class HashEmbedder:
    """Deterministic, dependency-free embedder for tests.

    Maps each input string to a fixed-size float32 vector derived from a
    SHA-256 digest. Two identical inputs produce identical vectors, and
    similar inputs produce similar-but-not-identical vectors (sufficient
    for exercising the retrieval pipeline end to end).
    """

    def __init__(self, dim: int = 32) -> None:
        self.dim = int(dim)

    def encode(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            tokens = (t or " ").lower().split()
            for tok in tokens:
                h = hashlib.sha256(tok.encode("utf-8")).digest()
                # Spread token contribution across the vector.
                for j in range(self.dim):
                    out[i, j] += (h[j % len(h)] / 255.0) - 0.5
            n = np.linalg.norm(out[i]) + 1e-12
            out[i] /= n
        return out


def _make_mini_project(root: Path) -> None:
    (root / "pkg").mkdir(parents=True, exist_ok=True)
    (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pkg" / "math_ops.py").write_text(
        textwrap.dedent(
            '''
            """Tiny math utilities used by the integration test."""

            def add(a, b):
                """Return the sum of two numbers."""
                return a + b


            def multiply(a, b):
                """Return the product of two numbers."""
                return a * b
            '''
        ).lstrip(),
        encoding="utf-8",
    )
    (root / "pkg" / "strings.py").write_text(
        textwrap.dedent(
            '''
            """String helpers."""

            def shout(text):
                """Return TEXT uppercased with an exclamation mark."""
                return text.upper() + "!"
            '''
        ).lstrip(),
        encoding="utf-8",
    )


def test_index_then_query_with_fake_embedder(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    out_dir = tmp_path / "out"
    _make_mini_project(project)

    embedder = HashEmbedder(dim=32)

    index_result = run_index_auto(
        str(project),
        str(out_dir),
        metric="cosine",
        index_type="flat",
        embedder=embedder,
    )

    # The project has at least one chunk per view (intent + impl).
    assert "counts" in index_result
    assert any(c > 0 for c in index_result["counts"].values())
    for key in ("indices", "records", "chunks"):
        assert Path(index_result["out"][key]).exists()

    query_result = run_query_auto(
        index_dir=index_result["out"]["indices"],
        records_path=index_result["out"]["records"],
        query="how do I add two numbers?",
        chunks_path=index_result["out"]["chunks"],
        graph_path=index_result["out"]["graph"],
        embedder=embedder,
        top_k_per_view=5,
        neighbor_depth=1,
    )

    assert "hits" in query_result
    assert isinstance(query_result["hits"], list)
    assert query_result["debug"]["lexical_forced"] is True
    # The `add` function should be discoverable somewhere in the hits or top files.
    flat = " ".join(
        str(h.get("chunk_id", "") or h.get("name", "") or "")
        for h in query_result["hits"]
    ) + " " + " ".join(str(f) for f in query_result.get("top_files", []))
    assert "add" in flat or "math_ops" in flat
