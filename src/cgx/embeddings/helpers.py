# src/cgx/embeddings/records.py
from __future__ import annotations

"""
S4 — Deterministic record builder and two-view embedding corpus.

This module is ADDITIVE and safe: it does not modify existing parse/graph code or
assume any embedding model. It turns your parsed chunks (+ optional graph) into:

1) Canonical index records (one per chunk) with the exact, deterministic fields
   you outlined (identity, graph anchors, semantics, metrics, search helpers).
2) A flat embedding corpus with two rows per chunk (view='intent' and 'impl'),
   containing the text to embed and basic accounting (tokens estimate, mapping).

Primary entrypoints:
- make_index_records(chunks, G=None, ...)
- prepare_embedding_corpus(records, which=('intent','impl'))

Both are pure functions and return NEW data structures.
"""

from cgx.logging_setup import get_logger
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover
    nx = None  # we guard for None at call sites

from cgx.embeddings.views import (
    build_intent_view,
    build_implementation_view,
)

logger = get_logger(__name__)


# ---------------------------
# Small deterministic helpers
# ---------------------------

def _safe_get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Safely retrieve a nested value from a dictionary using a dotted path.

    Args:
        d (Dict[str, Any]): Dictionary to traverse.
        path (str): Dot-delimited key path (e.g., "meta.class_name").
        default (Any): Value to return if path is missing.

    Returns:
        Any: Value found at the path, or `default` if not found.
    """
    cur = d
    for p in path.split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur

def _lc(s: Optional[str]) -> str:
    """
    Lowercase a string, returning an empty string if None.

    Args:
        s (Optional[str]): Input string.

    Returns:
        str: Lowercased string.
    """
    return (s or "").lower()

def _split_tokens(s: str) -> List[str]:
    """
    Split a string into tokens on non-alphanumeric characters.

    Args:
        s (str): Input string.

    Returns:
        List[str]: Lowercased tokens, filtering out empties.
    """
    return [t for t in re.split(r"[^A-Za-z0-9_]+", s.lower()) if t]

def _ngrams(tokens: List[str], n: int) -> List[str]:
    """
    Generate n-grams from a list of tokens.

    Args:
        tokens (List[str]): Token sequence.
        n (int): N-gram size.

    Returns:
        List[str]: List of space-joined n-grams.
    """
    if n <= 1:
        return tokens[:]
    out = []
    for i in range(0, max(0, len(tokens) - n + 1)):
        out.append(" ".join(tokens[i:i+n]))
    return out

def _estimate_tokens(s: str) -> int:
    """
    Fast, deterministic token estimate (not model-specific).

    Heuristic: 1 token per ~4 chars, clamped >= 1 for non-empty strings.

    Args:
        s (str): Input text.

    Returns:
        int: Estimated token count.
    """
    if not s:
        return 0
    est = max(1, int(math.ceil(len(s) / 4.0)))
    return est

def _attribute_roots_read(meta: Dict[str, Any]) -> List[str]:
    """
    Extract root attributes accessed via `self.<attr>` from metadata.

    Args:
        meta (Dict[str, Any]): Metadata dictionary.

    Returns:
        List[str]: Sorted unique root attributes.
    """
    roots = set()
    try:
        reads = meta.get("attributes_used") or meta.get("reads") or []
        for dotted in reads if isinstance(reads, list) else []:
            if isinstance(dotted, str) and dotted.startswith("self."):
                after = dotted.split("self.", 1)[1]
                root = after.split(".", 1)[0]
                if root:
                    roots.add(root)
    except Exception:
        pass
    return sorted(roots)

def _normalize_raises(meta: Dict[str, Any]) -> List[str]:
    """
    Normalize exceptions raised into a consistent list of names.

    Args:
        meta (Dict[str, Any]): Metadata containing "raises".

    Returns:
        List[str]: Sorted list of exception names.
    """
    out: List[str] = []
    rs = meta.get("raises") or []
    if isinstance(rs, list):
        for r in rs:
            if isinstance(r, str) and r:
                out.append(r)
            elif isinstance(r, dict):
                nm = r.get("name")
                if nm:
                    out.append(str(nm))
    return sorted(set(out))

def _imports_full(meta: Dict[str, Any]) -> List[str]:
    """
    Collect full import paths from metadata.

    Args:
        meta (Dict[str, Any]): Metadata containing "imports_used".

    Returns:
        List[str]: Sorted list of fully qualified imports.
    """
    fulls = set()
    imps = meta.get("imports_used")
    if isinstance(imps, dict):
        for full in imps.values():
            if isinstance(full, str) and full:
                fulls.add(full)
    elif isinstance(imps, list):
        for full in imps:
            if isinstance(full, str) and full:
                fulls.add(full)
    return sorted(fulls)

def _parent_class_id(chunk: Dict[str, Any]) -> Optional[str]:
    """
    Get the parent class node id for a method/function chunk.

    Args:
        chunk (Dict[str, Any]): Chunk with type and metadata.

    Returns:
        Optional[str]: Parent class id, or None if not applicable.
    """
    if (chunk.get("type") == "method") or (chunk.get("type") == "function" and "::method::" in chunk.get("id","")):
        meta = chunk.get("meta") or {}
        cls = meta.get("class_name")
        if cls:
            return f"{chunk['file']}::class::{cls}"
    return None

def _defines_children_ids(G, node_id: str, limit: int = 10_000) -> List[str]:
    """
    Collect children nodes defined by a given node.

    Args:
        G: Graph object (networkx).
        node_id (str): Node identifier.
        limit (int): Maximum number of children.

    Returns:
        List[str]: Child node ids defined by this node.
    """
    out: List[str] = []
    try:
        if G is None or node_id not in G:
            return out
        cnt = 0
        for succ in G.successors(node_id):
            ed = G[node_id][succ]
            etype = ed.get("type") if isinstance(ed, dict) and not any(isinstance(v, dict) for v in ed.values()) \
                    else (list(ed.values())[0].get("type") if isinstance(ed, dict) and ed else None)
            if etype == "defines":
                out.append(succ)
                cnt += 1
                if cnt >= limit:
                    break
    except Exception:
        return out
    return out

def _calls_out_ids(G, node_id: str) -> Tuple[List[str], List[str]]:
    """
    Collect outgoing call targets from a node.

    Args:
        G: Graph object (networkx).
        node_id (str): Node identifier.

    Returns:
        Tuple[List[str], List[str]]: 
            - internal_targets: ids of project functions/methods/lambdas.
            - unresolved_names: unresolved function names.
    """
    internal: List[str] = []
    unresolved: List[str] = []
    try:
        if G is None or node_id not in G:
            return internal, unresolved
        for succ in G.successors(node_id):
            ed = G[node_id][succ]
            attrs = ed if isinstance(ed, dict) and not any(isinstance(v, dict) for v in ed.values()) \
                    else (list(ed.values())[0] if isinstance(ed, dict) and ed else {})
            if attrs.get("type") == "calls":
                st = G.nodes[succ].get("type")
                if attrs.get("internal") is True and st in {"function", "method", "lambda"}:
                    internal.append(succ)
                elif st == "unresolved":
                    name = G.nodes[succ].get("name")
                    if isinstance(name, str) and name:
                        unresolved.append(name)
        return sorted(set(internal)), sorted(set(unresolved))
    except Exception:
        return internal, unresolved

def _calls_degree(G, node_id: str) -> Tuple[int, int]:
    """
    Count number of incoming and outgoing call edges.

    Args:
        G: Graph object (networkx).
        node_id (str): Node identifier.

    Returns:
        Tuple[int, int]: (calls_in_count, calls_out_count)
    """
    cin = cout = 0
    try:
        if G is None or node_id not in G:
            return 0, 0
        for pred in G.predecessors(node_id):
            ed = G[pred][node_id]
            attrs = ed if isinstance(ed, dict) and not any(isinstance(v, dict) for v in ed.values()) \
                    else (list(ed.values())[0] if isinstance(ed, dict) and ed else {})
            if attrs.get("type") == "calls":
                cin += 1
        for succ in G.successors(node_id):
            ed = G[node_id][succ]
            attrs = ed if isinstance(ed, dict) and not any(isinstance(v, dict) for v in ed.values()) \
                    else (list(ed.values())[0] if isinstance(ed, dict) and ed else {})
            if attrs.get("type") == "calls":
                cout += 1
        return int(cin), int(cout)
    except Exception:
        return 0, 0

def _neighbors_summary(G, node_id: str, max_n: int = 64) -> List[Tuple[str, str]]:
    """
    Collect a deterministic summary of neighbors of a node.

    Args:
        G: Graph object (networkx).
        node_id (str): Node identifier.
        max_n (int): Maximum neighbors to return.

    Returns:
        List[Tuple[str, str]]: List of (edge_type, neighbor_id) tuples.
    """
    out: List[Tuple[str, str]] = []
    try:
        if G is None or node_id not in G:
            return out
        for u in G.predecessors(node_id):
            ed = G[u][node_id]
            attrs = ed if isinstance(ed, dict) and not any(isinstance(v, dict) for v in ed.values()) \
                    else (list(ed.values())[0] if isinstance(ed, dict) and ed else {})
            et = attrs.get("type", "")
            out.append((et, u))
        for v in G.successors(node_id):
            ed = G[node_id][v]
            attrs = ed if isinstance(ed, dict) and not any(isinstance(v2, dict) for v2 in ed.values()) \
                    else (list(ed.values())[0] if isinstance(ed, dict) and ed else {})
            et = attrs.get("type", "")
            out.append((et, v))
        out = sorted({(et, nid) for et, nid in out})[:max_n]
        return out
    except Exception:
        return out


def _lexical_helpers(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build lexical helper fields (lowercased and n-grams) for a chunk.

    Args:
        chunk (Dict[str, Any]): Code chunk dictionary.

    Returns:
        Dict[str, Any]: Dictionary with lowercased fields and n-grams.
    """
    name = _lc(chunk.get("name"))
    cid = _lc(chunk.get("id"))
    file = _lc(chunk.get("file"))
    meta = chunk.get("meta") or {}
    cls = _lc(meta.get("class_name"))
    sig = _lc(meta.get("signature"))

    toks = _split_tokens(" ".join([t for t in [name, cid, file, cls, sig] if t]))
    unigrams = sorted(set(toks))
    bigrams = sorted(set(_ngrams(toks, 2)))

    return {
        "name_lc": name,
        "id_lc": cid,
        "file_lc": file,
        "class_name_lc": cls,
        "signature_lc": sig,
        "ngrams_1": unigrams,
        "ngrams_2": bigrams,
    }
