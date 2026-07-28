

# src/cgx/io/persist.py
from __future__ import annotations

"""
Persistence utilities for indices, records, chunks, calls, and graphs (ADD-ONLY).

All saves are explicit, human-readable where possible (JSON/JSONL), and
gracefully degrade when optional deps (e.g., FAISS) are missing.
"""

import json
import os
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

try:
    import faiss  # type: ignore
    _FAISS = True
except Exception:
    faiss = None  # type: ignore
    _FAISS = False

try:
    import networkx as nx  # type: ignore
    from networkx.readwrite import json_graph  # type: ignore
    _NX = True
except Exception:
    nx = None  # type: ignore
    json_graph = None  # type: ignore
    _NX = False


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _now_iso() -> str:
    """Local, timezone-aware completion timestamp for the index manifest."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def save_jsonl(items: List[Dict[str, Any]], path: str) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            if not isinstance(it, dict):
                logger.warning("save_jsonl: skipping non-dict item %r", it)
                continue
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def load_jsonl(path: str):
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(obj)
    return out




def save_indices(indices: Dict[str, Any], out_dir: str) -> None:
    """
    Save per-view FAISS indices (if present) and metadata/rows.
    Layout:
      out_dir/
        meta.json
        intent.index, intent.rows.jsonl
        impl.index,   impl.rows.jsonl
    """
    _ensure_dir(out_dir)

    # Import here to avoid a hard cyclic dependency at module load time.
    from cgx.embeddings.records import SCHEMA_VERSION

    # Manifest carries enough provenance for a later query to pick the right
    # embedder without guessing: the embed model + its vector dim (so a
    # 768-vs-1024 mismatch is caught up front), when the build completed, and
    # which project it indexed (so tools can warn when pointed at a stale or
    # foreign index).
    meta = {
        "schema_version": SCHEMA_VERSION,
        "metric": indices.get("metric"),
        "embed_model": indices.get("embed_model"),
        "index_type": indices.get("index_type"),
        "project_root": indices.get("project_root"),
        "indexed_at": indices.get("indexed_at") or _now_iso(),
        "counts": indices.get("counts"),
        "views": {},
    }

    for view in ("intent", "impl"):
        v = indices.get("views", {}).get(view) or {}
        rows = v.get("rows") or []
        idx = v.get("index")
        vdir = out_dir
        # rows
        save_jsonl(rows, os.path.join(vdir, f"{view}.rows.jsonl"))
        # index
        if idx is not None and _FAISS:
            try:
                faiss.write_index(idx, os.path.join(vdir, f"{view}.index"))  # type: ignore
                meta["views"][view] = {
                    "has_index": True,
                    "dim": int(getattr(idx, "d", 0)) or None,
                }
            except Exception as e:
                logger.warning("Failed to save FAISS index for view %s: %s", view, e)
                meta["views"][view] = {"has_index": False}
        else:
            meta["views"][view] = {"has_index": False}

    # Top-level dim mirror (either view; both share the same embedder).
    meta["embed_dim"] = (meta["views"].get("intent", {}).get("dim")
                         or meta["views"].get("impl", {}).get("dim"))

    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# In-process load_indices cache
# ---------------------------------------------------------------------------
#
# FAISS indices + per-view row JSONL are immutable between re-index runs, yet
# a single interactive ask reloads them from disk more than once (once in
# ``run_query_auto`` and again in ``_prepare_answer_request``); plan/agent
# runs repeat the read on every task. Reading FAISS off disk and re-parsing
# thousands of row lines on each call is a real latency tax.
#
# We memoize the parsed result keyed by the absolute index dir plus the
# ``meta.json`` mtime + size. ``save_indices`` always rewrites ``meta.json``
# last, so a fresh build naturally changes the key and invalidates the entry.
# The cache is bounded with FIFO eviction and can be disabled via
# ``CGX_DISABLE_INDEX_CACHE=1`` for debugging. Callers treat the returned dict
# as read-only (rows/index are wrapped, never mutated in place), so sharing the
# cached object is safe.
_INDEX_CACHE_MAX = 4
_INDEX_CACHE: "Dict[str, Tuple[Tuple[int, int], Dict[str, Any]]]" = {}
_INDEX_CACHE_ORDER: "List[str]" = []
_INDEX_CACHE_LOCK = threading.Lock()


def _index_cache_key(in_dir: str) -> Optional[Tuple[str, Tuple[int, int]]]:
    """Return ``(abs_dir, (mtime_ns, size))`` for the dir's meta.json, or None."""
    try:
        st = os.stat(os.path.join(in_dir, "meta.json"))
    except OSError:
        return None
    return os.path.abspath(in_dir), (st.st_mtime_ns, st.st_size)


def clear_indices_cache() -> None:
    """Drop all cached indices (test hook / manual invalidation)."""
    with _INDEX_CACHE_LOCK:
        _INDEX_CACHE.clear()
        _INDEX_CACHE_ORDER.clear()


def load_indices(in_dir: str, *, use_cache: bool = True) -> Dict[str, Any]:
    """
    Load indices and rows from a directory created by save_indices.
    Will set index=None if FAISS is unavailable or the file is missing.

    Results are memoized per index dir keyed on ``meta.json``'s mtime+size so
    repeated queries against an unchanged index skip the disk read entirely.
    A re-index rewrites ``meta.json`` and so invalidates the entry. Set
    ``use_cache=False`` (or ``CGX_DISABLE_INDEX_CACHE=1``) to force a fresh read.
    """
    cache_enabled = use_cache and os.environ.get("CGX_DISABLE_INDEX_CACHE") not in ("1", "true", "True")
    cache_key = _index_cache_key(in_dir) if cache_enabled else None
    if cache_key is not None:
        abs_dir, stamp = cache_key
        with _INDEX_CACHE_LOCK:
            cached = _INDEX_CACHE.get(abs_dir)
            if cached is not None and cached[0] == stamp:
                return cached[1]

    result = _load_indices_from_disk(in_dir)

    if cache_key is not None:
        abs_dir, stamp = cache_key
        with _INDEX_CACHE_LOCK:
            _INDEX_CACHE[abs_dir] = (stamp, result)
            if abs_dir in _INDEX_CACHE_ORDER:
                _INDEX_CACHE_ORDER.remove(abs_dir)
            _INDEX_CACHE_ORDER.append(abs_dir)
            while len(_INDEX_CACHE_ORDER) > _INDEX_CACHE_MAX:
                evict = _INDEX_CACHE_ORDER.pop(0)
                _INDEX_CACHE.pop(evict, None)
    return result


def _load_indices_from_disk(in_dir: str) -> Dict[str, Any]:
    """Uncached parse of an index directory (see :func:`load_indices`)."""
    with open(os.path.join(in_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    result = {
        "metric": meta.get("metric"),
        "embed_model": meta.get("embed_model"),
        "embed_dim": meta.get("embed_dim"),
        "index_type": meta.get("index_type"),
        "project_root": meta.get("project_root"),
        "indexed_at": meta.get("indexed_at"),
        "counts": meta.get("counts"),
        "schema_version": meta.get("schema_version"),
        "views": {},
    }
    for view in ("intent", "impl"):
        rows = load_jsonl(os.path.join(in_dir, f"{view}.rows.jsonl"))
        index_path = os.path.join(in_dir, f"{view}.index")
        idx = None
        if _FAISS and os.path.exists(index_path):
            try:
                idx = faiss.read_index(index_path)  # type: ignore
            except Exception as e:
                logger.warning("Failed to load FAISS index for %s: %s", view, e)
        result["views"][view] = {"index": idx, "rows": rows, "ids": None, "meta": None}
    return result


def save_graph_json(G, path: str) -> None:
    """Save graph as NetworkX node-link JSON (portable)."""
    if not _NX:
        raise RuntimeError("networkx is required to save/load graphs.")
    _ensure_dir(os.path.dirname(path) or ".")
    data = json_graph.node_link_data(G)  # type: ignore
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def load_graph_json(path: str):
    if not _NX:
        raise RuntimeError("networkx is required to save/load graphs.")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # networkx>=3.4 changed the default edges key in node_link_data from
    # "links" to "edges". Honour whichever key the saved file uses so older
    # graph.json artifacts continue to load.
    edges_key = "edges" if isinstance(data, dict) and "edges" in data else "links"
    return json_graph.node_link_graph(data, edges=edges_key)  # type: ignore
