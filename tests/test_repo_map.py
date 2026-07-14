"""Tests for the hierarchical repo map (cgx.answer.repo_map).

Covers map generation from index records (package/file/symbol hierarchy),
fingerprint-based caching + invalidation, and the budgeted text renderer.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from cgx.answer.repo_map import (
    build_or_load_repo_map,
    build_repo_map,
    fingerprint_records,
    load_repo_map,
    render_repo_map,
)
from cgx.embeddings.records import make_index_records
from cgx.graph.build_graph import build_knowledge_graph
from cgx.parser.parse_codebase import parse_codebase


def _records_for(root: Path):
    chunks, calls = parse_codebase(str(root))
    G = build_knowledge_graph(chunks, calls)
    return make_index_records(chunks, G)


def _write(root: Path, rel: str, body: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(body), encoding="utf-8")


def _sample_project(root: Path) -> None:
    _write(root, "pkg/__init__.py", "")
    _write(root, "pkg/calc.py", '''
        """Calculator utilities."""

        class Calc:
            """Adds numbers."""
            def add(self, a, b):
                return a + b
            def sub(self, a, b):
                return a - b

        def helper(x):
            """A free function."""
            return x
    ''')
    _write(root, "pkg/util/io.py", '''
        """IO helpers."""
        def read(path):
            return open(path).read()
    ''')


def test_build_repo_map_has_hierarchy(tmp_path):
    _sample_project(tmp_path)
    records = _records_for(tmp_path)
    rmap = build_repo_map(records)

    assert rmap["stats"]["n_files"] >= 2
    # Files are indexed with their symbols.
    by_path = {f["path"]: f for f in rmap["files"]}
    calc = next(f for p, f in by_path.items() if p.endswith("calc.py"))
    kinds = {(s["kind"], s["name"]) for s in calc["symbols"]}
    assert ("class", "Calc") in kinds
    assert ("function", "helper") in kinds
    # Methods are folded into their class as a count, not top-level symbols.
    calc_cls = next(s for s in calc["symbols"]
                    if s["kind"] == "class" and s["name"] == "Calc")
    assert calc_cls["methods"] == 2
    assert calc["summary"] == "Calculator utilities."

    # Packages roll up files under their directory. The shared ``pkg/`` prefix
    # collapses into the map root, so top-level files sit under "." and the
    # nested directory appears as "util".
    assert rmap["root"] == "pkg"
    pkg_paths = {p["path"] for p in rmap["packages"]}
    assert "." in pkg_paths
    assert "util" in pkg_paths


def test_fingerprint_changes_when_summary_changes(tmp_path):
    _sample_project(tmp_path)
    fp1 = fingerprint_records(_records_for(tmp_path))
    # Change a docstring -> summary material changes -> fingerprint changes.
    _write(tmp_path, "pkg/util/io.py", '''
        """Totally different summary."""
        def read(path):
            return open(path).read()
    ''')
    fp2 = fingerprint_records(_records_for(tmp_path))
    assert fp1 != fp2


def test_build_or_load_caches_and_invalidates(tmp_path):
    _sample_project(tmp_path)
    cache = tmp_path / "repo_map.json"
    records = _records_for(tmp_path)

    rmap1 = build_or_load_repo_map(records, cache_path=str(cache))
    assert cache.exists()
    fp1 = rmap1["fingerprint"]

    # Same records -> cache hit returns an identical map.
    rmap2 = build_or_load_repo_map(records, cache_path=str(cache))
    assert rmap2["fingerprint"] == fp1
    assert rmap2 == load_repo_map(str(cache))

    # Change the tree -> fingerprint drifts -> cache is rebuilt + re-persisted.
    _write(tmp_path, "pkg/new.py", "def brand_new():\n    return 1\n")
    records2 = _records_for(tmp_path)
    rmap3 = build_or_load_repo_map(records2, cache_path=str(cache))
    assert rmap3["fingerprint"] != fp1
    assert load_repo_map(str(cache))["fingerprint"] == rmap3["fingerprint"]


def test_load_repo_map_rejects_version_mismatch(tmp_path):
    _sample_project(tmp_path)
    cache = tmp_path / "repo_map.json"
    build_or_load_repo_map(_records_for(tmp_path), cache_path=str(cache))
    import json
    data = json.loads(cache.read_text())
    data["version"] = 999
    cache.write_text(json.dumps(data))
    assert load_repo_map(str(cache)) is None


def test_render_repo_map_is_budgeted(tmp_path):
    _sample_project(tmp_path)
    rmap = build_repo_map(_records_for(tmp_path))
    text = render_repo_map(rmap, max_chars=10_000)
    assert "Repo map:" in text
    assert "class Calc" in text
    assert "def helper" in text
    # Budget is enforced.
    tiny = render_repo_map(rmap, max_chars=80)
    assert len(tiny) <= 80 + len("\n... (truncated)")
    assert tiny.endswith("(truncated)")
