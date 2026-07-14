"""Incremental parsing keyed on file content hash.

Mirrors the embedding cache design (:mod:`cgx.embeddings.cache`): a persisted,
content-addressed manifest lets re-index runs skip re-parsing files whose bytes
are unchanged. Only added/modified files are dispatched to their parser;
unchanged files reuse their cached per-file ``(chunks, calls)``; deleted files
are dropped. The global cross-file post-processing (call dedup + reverse edges,
:func:`cgx.parser.parse_codebase._finalize_calls`) always re-runs over the
merged set so per-chunk call metadata (``calls_out_top`` / ``called_by_count``)
stays correct even when only one file changed.

On-disk layout (JSON at ``cache_path``)::

    parse_cache.json
      ├── version        : int   -- PARSE_CACHE_VERSION (structure of this file)
      ├── schema_version : int   -- cgx.embeddings.records.SCHEMA_VERSION
      └── files          : { rel_path: {mtime, sha, chunks, calls} }

The cache stores *raw* per-file parse output (before ``_finalize_calls``) so the
reverse-edge metadata is never baked into a cached entry; it is recomputed from
the merged corpus on every run.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from cgx.embeddings.records import SCHEMA_VERSION
from cgx.logging_setup import get_logger
from cgx.parser.parse_codebase import _finalize_calls, _iter_source_files

logger = get_logger("parser.incremental")

# Bump when the structure of the cache file itself changes (independent of the
# chunk schema, which is tracked by SCHEMA_VERSION). Either mismatch discards
# the cache and forces a full re-parse.
PARSE_CACHE_VERSION = 1


def _hash_source(text: str) -> str:
    """Return the sha256 hex digest of ``text`` (the content-address key)."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _empty_stats() -> Dict[str, int]:
    return {
        "added": 0, "modified": 0, "deleted": 0, "unchanged": 0,
        "moved": 0, "reparsed": 0, "total_files": 0,
    }


def load_parse_cache(cache_path: str) -> Dict[str, Dict[str, Any]]:
    """Load the per-file cache from ``cache_path``.

    Returns the ``{rel_path: entry}`` mapping, or an empty dict when the file
    is missing, unreadable, or written by an incompatible cache/schema version
    (in which case the caller re-parses everything).
    """
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as exc:
        logger.warning("parse cache unreadable (%s); ignoring: %s", cache_path, exc)
        return {}
    if not isinstance(data, dict):
        return {}
    if int(data.get("version", -1)) != PARSE_CACHE_VERSION:
        logger.info("parse cache version mismatch; discarding")
        return {}
    if int(data.get("schema_version", -1)) != SCHEMA_VERSION:
        logger.info("parse cache schema_version mismatch; discarding")
        return {}
    files = data.get("files")
    return files if isinstance(files, dict) else {}


def save_parse_cache(cache_path: str, files: Dict[str, Dict[str, Any]]) -> None:
    """Persist the per-file ``files`` mapping to ``cache_path`` atomically."""
    payload = {
        "version": PARSE_CACHE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "files": files,
    }
    tmp = f"{cache_path}.tmp"
    try:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, cache_path)
    except Exception as exc:
        logger.warning("failed to persist parse cache %s: %s", cache_path, exc)
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def incremental_parse_codebase(
    project_root: str,
    *,
    cache_path: str,
    ignore_patterns: Optional[List[str]] = None,
    max_file_bytes: Optional[int] = None,
    follow_symlinks: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    """Parse ``project_root`` incrementally, reusing an on-disk parse cache.

    Only files whose content hash changed (or that are new) are dispatched to
    their parser; unchanged files reuse their cached per-file output. Deleted
    files drop out of the cache. Returns ``(chunks, calls, stats)`` where
    ``chunks``/``calls`` are byte-for-byte equivalent to what a full
    :func:`cgx.parser.parse_codebase.parse_codebase` would produce for the same
    tree, and ``stats`` counts added/modified/deleted/unchanged/moved files.

    Moves are reported for observability but still trigger a re-parse: chunk
    ``id`` and ``module_path`` are path-derived, so a relocated file cannot
    reuse a cached entry keyed on the old path.
    """
    abs_root = os.path.abspath(project_root)
    prev_files = load_parse_cache(cache_path)
    new_files: Dict[str, Dict[str, Any]] = {}
    stats = _empty_stats()

    for parser, filepath, source_code in _iter_source_files(
        project_root,
        ignore_patterns=ignore_patterns,
        max_file_bytes=max_file_bytes,
        follow_symlinks=follow_symlinks,
    ):
        try:
            rel = os.path.relpath(filepath, abs_root)
        except Exception:
            rel = filepath
        sha = _hash_source(source_code)
        try:
            mtime = os.path.getmtime(filepath)
        except Exception:
            mtime = 0.0

        prior = prev_files.get(rel)
        if prior is not None and prior.get("sha") == sha:
            # Unchanged: reuse the cached raw per-file output.
            new_files[rel] = {
                "mtime": mtime, "sha": sha,
                "chunks": prior.get("chunks", []),
                "calls": prior.get("calls", []),
            }
            stats["unchanged"] += 1
            continue

        # Added or modified: re-parse. Skip (do not cache) on parser failure,
        # mirroring parse_codebase's per-file error handling.
        try:
            file_chunks, file_calls = parser.parse_file(filepath, source_code, project_root)
        except Exception as exc:
            logger.error("Parser failed for %s: %s", filepath, exc)
            continue
        new_files[rel] = {
            "mtime": mtime, "sha": sha,
            "chunks": file_chunks, "calls": file_calls,
        }
        stats["reparsed"] += 1
        if prior is None:
            stats["added"] += 1
        else:
            stats["modified"] += 1

    # Deletions: any previously cached path we did not see this walk.
    deleted_paths = [p for p in prev_files if p not in new_files]
    stats["deleted"] = len(deleted_paths)

    # Move detection (informational): a newly added path whose content hash
    # matches a just-deleted path is almost certainly a rename.
    deleted_shas = {prev_files[p].get("sha") for p in deleted_paths}
    for p, entry in new_files.items():
        if p not in prev_files and entry.get("sha") in deleted_shas:
            stats["moved"] += 1

    # Persist the raw cache *before* finalization so cached entries never carry
    # globally-computed reverse-edge metadata.
    save_parse_cache(cache_path, new_files)

    merged_chunks: List[Dict[str, Any]] = []
    merged_calls: List[Dict[str, Any]] = []
    for entry in new_files.values():
        merged_chunks.extend(entry.get("chunks", []))
        merged_calls.extend(entry.get("calls", []))

    chunks, calls = _finalize_calls(merged_chunks, merged_calls)
    stats["total_files"] = len(new_files)
    logger.info(
        "incremental parse: +%d ~%d -%d =%d (moved=%d) over %d files",
        stats["added"], stats["modified"], stats["deleted"],
        stats["unchanged"], stats["moved"], stats["total_files"],
    )
    return chunks, calls, stats


__all__ = [
    "incremental_parse_codebase",
    "load_parse_cache",
    "save_parse_cache",
    "PARSE_CACHE_VERSION",
]
