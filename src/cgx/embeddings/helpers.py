

# src/cgx/embeddings/records.py
from __future__ import annotations

"""
S4 -- Deterministic record builder and two-view embedding corpus.

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
    _attribute_roots_read,  # canonical home; re-exported here for back-compat
)
from cgx.retrieval.tokenize import tokenize_text
from cgx.graph.backend import CodeGraphBackend

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
    Split a string into lower-cased tokens, expanding camelCase/snake_case
    identifiers into their sub-words so the BM25 index can be probed by
    partial names at query time.

    Example: ``"databaseReconnect parse_input_args"`` ->
    ``["databasereconnect", "database", "reconnect",
       "parse_input_args", "parse", "input", "args"]``.

    The original (lower-cased) form is always kept first so exact-name
    matches are not weakened by the expansion.
    """
    return tokenize_text(s, min_len=1)

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
        G: Graph object (networkx or CodeGraphBackend).
        node_id (str): Node identifier.
        limit (int): Maximum number of children.

    Returns:
        List[str]: Child node ids defined by this node.
    """
    out: List[str] = []
    backend = CodeGraphBackend.wrap(G)
    if backend is None or not backend.has_node(node_id):
        return out
    cnt = 0
    for succ in backend.successors(node_id):
        if backend.edge_attrs(node_id, succ).get("type") == "defines":
            out.append(succ)
            cnt += 1
            if cnt >= limit:
                break
    return out

def _calls_out_ids(G, node_id: str) -> Tuple[List[str], List[str]]:
    """
    Collect outgoing call targets from a node.

    Args:
        G: Graph object (networkx or CodeGraphBackend).
        node_id (str): Node identifier.

    Returns:
        Tuple[List[str], List[str]]:
            - internal_targets: ids of project functions/methods/lambdas.
            - unresolved_names: unresolved function names.
    """
    internal: List[str] = []
    unresolved: List[str] = []
    backend = CodeGraphBackend.wrap(G)
    if backend is None or not backend.has_node(node_id):
        return internal, unresolved
    for succ in backend.successors(node_id):
        attrs = backend.edge_attrs(node_id, succ)
        if attrs.get("type") != "calls":
            continue
        node_attrs = backend.node_attrs(succ)
        st = node_attrs.get("type")
        if attrs.get("internal") is True and st in {"function", "method", "lambda"}:
            internal.append(succ)
        elif st == "unresolved":
            name = node_attrs.get("name")
            if isinstance(name, str) and name:
                unresolved.append(name)
    return sorted(set(internal)), sorted(set(unresolved))

def _calls_degree(G, node_id: str) -> Tuple[int, int]:
    """
    Count number of incoming and outgoing call edges.

    Args:
        G: Graph object (networkx or CodeGraphBackend).
        node_id (str): Node identifier.

    Returns:
        Tuple[int, int]: (calls_in_count, calls_out_count)
    """
    backend = CodeGraphBackend.wrap(G)
    if backend is None or not backend.has_node(node_id):
        return 0, 0
    cin = cout = 0
    for pred in backend.predecessors(node_id):
        if backend.edge_attrs(pred, node_id).get("type") == "calls":
            cin += 1
    for succ in backend.successors(node_id):
        if backend.edge_attrs(node_id, succ).get("type") == "calls":
            cout += 1
    return int(cin), int(cout)

def _neighbors_summary(G, node_id: str, max_n: int = 64) -> List[Tuple[str, str]]:
    """
    Collect a deterministic summary of neighbors of a node.

    Args:
        G: Graph object (networkx or CodeGraphBackend).
        node_id (str): Node identifier.
        max_n (int): Maximum neighbors to return.

    Returns:
        List[Tuple[str, str]]: List of (edge_type, neighbor_id) tuples.
    """
    backend = CodeGraphBackend.wrap(G)
    if backend is None or not backend.has_node(node_id):
        return []
    pairs: List[Tuple[str, str]] = []
    for u in backend.predecessors(node_id):
        pairs.append((backend.edge_attrs(u, node_id).get("type", ""), u))
    for v in backend.successors(node_id):
        pairs.append((backend.edge_attrs(node_id, v).get("type", ""), v))
    return sorted({(et, nid) for et, nid in pairs})[:max_n]


# Cap the number of content tokens indexed per chunk so a single pathological
# multi-thousand-line file cannot dominate the BM25 postings. Function/method
# chunks are far below this; only very large classes/files approach it.
_MAX_CONTENT_TOKENS = 4000


def _file_stem(path: str) -> str:
    """Basename without directory or extension: ``a/b/client.py`` -> ``client``.

    Used instead of the full path so filename queries still work without
    polluting every posting with directory/machine-path tokens (which share a
    huge common prefix and add near-zero-IDF noise to every chunk's vector and
    posting list)."""
    base = (path or "").replace("\\", "/").rsplit("/", 1)[-1]
    return base.rsplit(".", 1)[0] if "." in base else base


def _doc_parsed_text(meta: Dict[str, Any]) -> str:
    """Flatten the structured docstring (summary/description/params/returns/raises)
    into searchable text. Belt-and-suspenders for chunk types (e.g. file stubs)
    whose ``code`` segment omits the real body/docstring."""
    dp = meta.get("doc_parsed")
    if not isinstance(dp, dict):
        return ""
    parts: List[str] = []
    for key in ("summary", "description", "returns"):
        v = dp.get(key)
        if isinstance(v, str) and v:
            parts.append(v)
    for key in ("params", "raises"):
        seq = dp.get(key)
        if isinstance(seq, list):
            for it in seq:
                if isinstance(it, dict):
                    for kk in ("name", "type", "description", "desc"):
                        vv = it.get(kk)
                        if isinstance(vv, str) and vv:
                            parts.append(vv)
                elif isinstance(it, str):
                    parts.append(it)
    return " ".join(parts)


def _lexical_helpers(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build lexical helper fields (lowercased identity fields + BM25 posting tokens).

    ``ngrams_1`` is the frequency-preserving unigram stream the BM25 index scores
    over. Historically it covered ONLY identifiers (name/id/file/class/signature),
    so the lexical arm was blind to every docstring, comment, and code body -- a
    query phrased in behaviour/domain terms rather than the exact symbol name hit
    nothing. It now also tokenizes the chunk's source segment (``code`` -- which
    already includes the docstring, inline comments, and body) plus the flattened
    structured docstring, so content queries retrieve. For ``doc`` chunks ``code``
    is the section prose, so documentation bodies become searchable too.

    Tokens are NOT deduplicated into a set anymore: repeats are what give BM25 a
    real term-frequency signal (previously every term collapsed to tf=1). The
    absolute path is dropped from the posting stream (only the file *stem* is
    kept) to avoid polluting every posting with shared directory/machine tokens.
    Bigrams stay scoped to the short high-value identity+summary fields so phrase
    matches keep precision without exploding the index over full bodies.

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
    code = chunk.get("code") or ""
    docstring = meta.get("docstring") or ""
    stem = _file_stem(chunk.get("file") or "")

    # Identity tokens: symbol name, class, signature, file stem. High-precision;
    # these also seed the bigram phrase index.
    identity_text = " ".join([t for t in [name, cls, sig, stem] if t])
    identity_toks = _split_tokens(identity_text)

    # Content tokens: source segment (docstring + comments + body) and the
    # flattened structured docstring. Capped to bound index size.
    content_text = " ".join([t for t in [docstring, _doc_parsed_text(meta), code] if t])
    content_toks = _split_tokens(content_text)[:_MAX_CONTENT_TOKENS]

    # Frequency-preserving unigram stream (identity first so exact-name repeats
    # keep their weight), then content. No set() dedup -> real BM25 tf.
    unigrams = identity_toks + content_toks

    # Bigrams over the short, high-signal fields only (name/class/signature +
    # docstring summary) -- phrase precision without full-body explosion.
    from cgx.embeddings.views import _doc_first_sentence as _dfs  # local: avoid import cycle
    summary = _dfs(meta) if isinstance(meta, dict) else ""
    bigram_src = _split_tokens(" ".join([t for t in [name, cls, sig, summary] if t]))
    bigrams = sorted(set(_ngrams(bigram_src, 2)))

    return {
        "name_lc": name,
        "id_lc": cid,
        "file_lc": file,
        "class_name_lc": cls,
        "signature_lc": sig,
        "ngrams_1": unigrams,
        "ngrams_2": bigrams,
    }
