"""Golden-output snapshots that pin the shape of records, suggest_insertion_points
output, and hybrid retrieval hits over a tiny synthetic Python repo.

The snapshots intentionally compare only *keys* and *chunk ids*, not values, so
they fail only on shape regressions (added/removed/renamed fields) or on hit-ID
churn caused by tokenizer/scoring changes. Regenerate via SNAPSHOT_REGEN=1.
"""

from __future__ import annotations

import hashlib
import json
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from cgx.parser.parse_codebase import parse_codebase
from cgx.embeddings.records import SCHEMA_VERSION, make_index_records

SNAP_DIR = Path(__file__).parent / "snapshots"


class HashEmbedder:
    """Deterministic, dependency-free embedder mirroring test_integration_index_query."""

    def __init__(self, dim: int = 32) -> None:
        self.dim = int(dim)

    def encode(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in (t or " ").lower().split():
                h = hashlib.sha256(tok.encode("utf-8")).digest()
                for j in range(self.dim):
                    out[i, j] += (h[j % len(h)] / 255.0) - 0.5
            n = np.linalg.norm(out[i]) + 1e-12
            out[i] /= n
        return out


def _make_synth_repo(root: Path) -> None:
    pkg = root / "pkg"
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "db.py").write_text(
        textwrap.dedent(
            '''
            """Database helpers."""

            def databaseReconnect(retries: int = 3) -> bool:
                """Reconnect with bounded retries."""
                return retries > 0


            def parse_input_args(argv):
                """Parse command-line args."""
                return list(argv)
            '''
        ).lstrip(),
        encoding="utf-8",
    )
    (pkg / "calc.py").write_text(
        textwrap.dedent(
            '''
            """Calculator."""

            class Calculator:
                """A trivial calculator."""

                def add(self, a, b):
                    """Add two numbers."""
                    return a + b

                def multiply(self, a, b):
                    """Multiply two numbers."""
                    return a * b
            '''
        ).lstrip(),
        encoding="utf-8",
    )


def _load_snapshot(name: str) -> Dict[str, Any]:
    p = SNAP_DIR / name
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _save_snapshot(name: str, data: Dict[str, Any]) -> None:
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    (SNAP_DIR / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _assert_or_regen(name: str, observed: Dict[str, Any]) -> None:
    if os.environ.get("SNAPSHOT_REGEN") == "1":
        _save_snapshot(name, observed)
        return
    expected = _load_snapshot(name)
    if not expected:
        _save_snapshot(name, observed)
        return
    assert observed == expected, (
        f"Snapshot drift for {name}.\n"
        f"Run with SNAPSHOT_REGEN=1 to regenerate.\n"
        f"Expected: {json.dumps(expected, indent=2, sort_keys=True)[:800]}\n"
        f"Observed: {json.dumps(observed, indent=2, sort_keys=True)[:800]}"
    )


def test_schema_version_constant_is_present():
    assert isinstance(SCHEMA_VERSION, int) and SCHEMA_VERSION >= 1


def test_record_key_snapshot(tmp_path: Path) -> None:
    proj = tmp_path / "proj"
    _make_synth_repo(proj)
    chunks, _calls = parse_codebase(str(proj))
    recs = make_index_records(chunks, G=None)

    # All records must carry schema_version.
    assert recs, "expected non-empty records for synthetic repo"
    for r in recs:
        assert r.get("schema_version") == SCHEMA_VERSION

    # Snapshot the *set of keys* per record type. Stable across tokenizer/ranking changes.
    by_type: Dict[str, List[str]] = {}
    for r in recs:
        keys = sorted(k for k in r.keys() if k != "vec_intent" and k != "vec_impl")
        by_type.setdefault(str(r.get("type")), keys)
    observed = {"by_type_keys": {k: sorted(set(v)) for k, v in by_type.items()}}
    _assert_or_regen("record_keys.json", observed)


def _walk_keys(obj: Any, prefix: str = "") -> List[str]:
    out: List[str] = []
    if isinstance(obj, dict):
        for k in sorted(obj.keys()):
            path = f"{prefix}.{k}" if prefix else k
            out.append(path)
            out.extend(_walk_keys(obj[k], path))
    elif isinstance(obj, list) and obj:
        out.extend(_walk_keys(obj[0], f"{prefix}[]"))
    return out


def test_suggest_insertion_points_shape(tmp_path: Path) -> None:
    from cgx.retrieval.orchestrator import suggest_insertion_points

    proj = tmp_path / "proj"
    _make_synth_repo(proj)
    chunks, _calls = parse_codebase(str(proj))
    recs = make_index_records(chunks, G=None)

    # Fake fused hits: the first function record acts as the only seed.
    seed_id = next((r["id"] for r in recs if r.get("type") == "function"), recs[0]["id"])
    fused_hits = [{"chunk_id": seed_id, "score": 1.0, "rank": 1}]

    out = suggest_insertion_points(
        "add a new helper",
        fused_hits,
        recs,
        k_candidates=3,
        k_exemplars=1,
        embedder=None,
        G=None,
    )
    # Snapshot the recursive set of keys, ignoring values.
    observed = {"sample_keys": sorted(set(_walk_keys(out)))}
    _assert_or_regen("suggest_insertion_points_shape.json", observed)


def test_hybrid_retrieve_top_k_chunk_ids(tmp_path: Path) -> None:
    pytest.importorskip("faiss")
    from cgx.pipeline.auto import run_index_auto, run_query_auto

    proj = tmp_path / "proj"
    out_dir = tmp_path / "out"
    _make_synth_repo(proj)
    emb = HashEmbedder(dim=32)
    index_result = run_index_auto(
        str(proj), str(out_dir), metric="cosine", index_type="flat", embedder=emb,
    )
    q = run_query_auto(
        index_dir=index_result["out"]["indices"],
        records_path=index_result["out"]["records"],
        query="add two numbers",
        chunks_path=index_result["out"]["chunks"],
        graph_path=index_result["out"]["graph"],
        embedder=emb,
        top_k_per_view=5,
        neighbor_depth=0,
    )
    # Snapshot the *set* of top-3 chunk ids (sorted, path-normalised). Top-3
    # is the deterministic prefix on this synthetic repo; positions 4+ can
    # swap on BM25 tiebreaks because parse_codebase iterates ``os.walk`` in
    # filesystem-inode order, and that's outside the tokenizer's contract.
    top3 = sorted({str(h.get("chunk_id")) for h in (q.get("hits") or [])[:3]})
    norm = sorted({cid.split("/proj/", 1)[-1] for cid in top3})
    _assert_or_regen("hybrid_top3_ids.json", {"top3_relative_ids": norm})


def test_meta_json_carries_schema_version(tmp_path: Path) -> None:
    pytest.importorskip("faiss")
    from cgx.pipeline.auto import run_index_auto

    proj = tmp_path / "proj"
    out_dir = tmp_path / "out"
    _make_synth_repo(proj)
    res = run_index_auto(
        str(proj), str(out_dir), metric="cosine", index_type="flat",
        embedder=HashEmbedder(dim=16),
    )
    meta_path = Path(res["out"]["indices"]) / "meta.json"
    assert meta_path.exists(), "save_indices must write meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta.get("schema_version") == SCHEMA_VERSION


def test_camel_case_subword_query_hits_identifier(tmp_path: Path) -> None:
    """A query for a single sub-word ("reconnect") must surface the chunk
    whose identifier is ``databaseReconnect``. This exercises the symmetry
    of the sub-word tokenizer end-to-end (indexer + BM25 querier)."""
    pytest.importorskip("faiss")
    from cgx.pipeline.auto import run_index_auto, run_query_auto

    proj = tmp_path / "proj"
    out_dir = tmp_path / "out"
    _make_synth_repo(proj)
    emb = HashEmbedder(dim=32)
    idx = run_index_auto(
        str(proj), str(out_dir), metric="cosine", index_type="flat", embedder=emb,
    )
    q = run_query_auto(
        index_dir=idx["out"]["indices"],
        records_path=idx["out"]["records"],
        query="reconnect",
        chunks_path=idx["out"]["chunks"],
        graph_path=idx["out"]["graph"],
        embedder=emb,
        top_k_per_view=10,
        neighbor_depth=0,
    )
    hit_ids = [str(h.get("chunk_id") or "") for h in (q.get("hits") or [])]
    assert any("databaseReconnect" in cid for cid in hit_ids), (
        "Sub-word query 'reconnect' did not surface databaseReconnect; "
        f"hits were: {hit_ids}"
    )
