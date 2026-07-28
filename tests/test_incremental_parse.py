"""Tests for the content-addressed incremental parser.

Covers add / modify / delete / move cases, cache reuse (unchanged files are
not re-parsed), byte-for-byte equivalence with a full ``parse_codebase`` run,
and cache-version invalidation.
"""

from __future__ import annotations

import json
from pathlib import Path

from cgx.parser.incremental import (
    PARSE_CACHE_VERSION,
    incremental_parse_codebase,
    load_parse_cache,
)
from cgx.parser.parse_codebase import parse_codebase


def _fn_files(chunks):
    return {(c["name"], Path(c["file"]).name)
            for c in chunks if c["type"] == "function"}


def _write(root: Path, rel: str, body: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body, encoding="utf-8")


def test_first_run_matches_full_parse(tmp_path):
    proj = tmp_path / "proj"
    _write(proj, "a.py", "def foo():\n    return bar()\n")
    _write(proj, "b.py", "def bar():\n    return 2\n")
    cache = tmp_path / "parse_cache.json"

    chunks, calls, stats = incremental_parse_codebase(
        str(proj), cache_path=str(cache))
    full_chunks, full_calls = parse_codebase(str(proj))

    # Identical output to a full parse (including ordering).
    assert chunks == full_chunks
    assert calls == full_calls
    assert stats["added"] == 2 and stats["reparsed"] == 2
    assert stats["unchanged"] == 0 and stats["deleted"] == 0
    assert cache.exists()


def test_second_run_reuses_cache_without_reparsing(tmp_path):
    proj = tmp_path / "proj"
    _write(proj, "a.py", "def foo():\n    return 1\n")
    _write(proj, "b.py", "def bar():\n    return 2\n")
    cache = tmp_path / "parse_cache.json"

    incremental_parse_codebase(str(proj), cache_path=str(cache))
    chunks, _calls, stats = incremental_parse_codebase(
        str(proj), cache_path=str(cache))

    assert stats["unchanged"] == 2
    assert stats["reparsed"] == 0
    assert stats["added"] == 0 and stats["modified"] == 0
    # Reused output still equals a fresh full parse.
    full_chunks, _ = parse_codebase(str(proj))
    assert chunks == full_chunks


def test_modify_one_file_only_reparses_it(tmp_path):
    proj = tmp_path / "proj"
    _write(proj, "a.py", "def foo():\n    return 1\n")
    _write(proj, "b.py", "def bar():\n    return 2\n")
    cache = tmp_path / "parse_cache.json"

    incremental_parse_codebase(str(proj), cache_path=str(cache))
    _write(proj, "b.py", "def bar():\n    return 99\n")
    chunks, _calls, stats = incremental_parse_codebase(
        str(proj), cache_path=str(cache))

    assert stats["modified"] == 1
    assert stats["unchanged"] == 1
    assert stats["reparsed"] == 1
    # The changed body is reflected in the returned chunks.
    bar = next(c for c in chunks
               if c["type"] == "function" and c["name"] == "bar")
    assert "99" in bar["code"]
    # Equivalent to a full parse of the new tree.
    full_chunks, _ = parse_codebase(str(proj))
    assert chunks == full_chunks


def test_delete_file_drops_its_chunks(tmp_path):
    proj = tmp_path / "proj"
    _write(proj, "a.py", "def foo():\n    return 1\n")
    _write(proj, "b.py", "def bar():\n    return 2\n")
    cache = tmp_path / "parse_cache.json"

    incremental_parse_codebase(str(proj), cache_path=str(cache))
    (proj / "b.py").unlink()
    chunks, _calls, stats = incremental_parse_codebase(
        str(proj), cache_path=str(cache))

    assert stats["deleted"] == 1
    assert _fn_files(chunks) == {("foo", "a.py")}
    # Deleted path is gone from the persisted cache.
    assert "b.py" not in load_parse_cache(str(cache))


def test_move_file_is_detected_and_reparsed(tmp_path):
    proj = tmp_path / "proj"
    _write(proj, "a.py", "def foo():\n    return 1\n")
    cache = tmp_path / "parse_cache.json"

    incremental_parse_codebase(str(proj), cache_path=str(cache))
    (proj / "a.py").rename(proj / "b.py")
    chunks, _calls, stats = incremental_parse_codebase(
        str(proj), cache_path=str(cache))

    assert stats["deleted"] == 1
    assert stats["added"] == 1
    assert stats["moved"] == 1
    # Chunk identity follows the new path (id/module_path are path-derived).
    assert _fn_files(chunks) == {("foo", "b.py")}
    foo = next(c for c in chunks if c["type"] == "function")
    assert foo["file"].endswith("b.py")


def test_cache_version_mismatch_forces_full_reparse(tmp_path):
    proj = tmp_path / "proj"
    _write(proj, "a.py", "def foo():\n    return 1\n")
    cache = tmp_path / "parse_cache.json"

    incremental_parse_codebase(str(proj), cache_path=str(cache))
    # Corrupt the version so the loader discards the cache.
    data = json.loads(cache.read_text())
    data["version"] = PARSE_CACHE_VERSION + 999
    cache.write_text(json.dumps(data))
    assert load_parse_cache(str(cache)) == {}

    _chunks, _calls, stats = incremental_parse_codebase(
        str(proj), cache_path=str(cache))
    # Nothing reused -> treated as a fresh add.
    assert stats["added"] == 1 and stats["unchanged"] == 0
