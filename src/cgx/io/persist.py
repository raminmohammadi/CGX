

# src/cgx/io/persist.py
from __future__ import annotations

"""
Persistence utilities for indices, records, chunks, calls, and graphs (ADD-ONLY).

All saves are explicit, human-readable where possible (JSON/JSONL), and
gracefully degrade when optional deps (e.g., FAISS) are missing.
"""

import json
import os
from typing import Any, Dict, List, Optional

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

    meta = {
        "metric": indices.get("metric"),
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
                meta["views"][view] = {"has_index": True}
            except Exception as e:
                logger.warning("Failed to save FAISS index for view %s: %s", view, e)
                meta["views"][view] = {"has_index": False}
        else:
            meta["views"][view] = {"has_index": False}

    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_indices(in_dir: str) -> Dict[str, Any]:
    """
    Load indices and rows from a directory created by save_indices.
    Will set index=None if FAISS is unavailable or the file is missing.
    """
    with open(os.path.join(in_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)

    result = {"metric": meta.get("metric"), "views": {}}
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
