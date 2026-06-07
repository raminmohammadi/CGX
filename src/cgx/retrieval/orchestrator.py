

from __future__ import annotations

import concurrent.futures

"""
Two-view retrieval orchestrator (ALL SIGNALS REQUIRED + IMPACT ANALYSIS).

Provides deterministic, auditable retrieval by orchestrating:
  • ANN indices for both views ("intent", "impl") from S4 records/corpus.
  • Semantic retrieval via TwoViewIndex.
  • Lexical retrieval via LexicalIndex (BM25-lite) or a regex fallback on chunks.
  • Graph expansion on the knowledge graph.
  • RRF fusion of all lists (semantic+lexical+graph-boost).
  • Aggregation of results to files/classes.
  • Insertion-point suggestions.
  • Change impact analysis (files likely affected by editing a target symbol).

Nothing is optional: if a signal is unavailable (e.g., no G), we still
execute the step (no-ops) so the pipeline shape is always the same.
"""

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from dataclasses import dataclass

import math
import re
import numpy as np
# networkx ops are funneled through CodeGraphBackend (see cgx.graph.backend)
# so this module no longer touches the nx API directly.

from cgx.logging_setup import get_logger
logger = get_logger("orchestration")

# canonical imports
from cgx.retrieval.ann_numpy import build_ann_index
from cgx.retrieval.index import TwoViewIndex, ViewSlice
from cgx.retrieval.lexical import LexicalIndex
from cgx.retrieval.rrf import rrf_fuse
from cgx.retrieval.tokenize import expand_with_subwords
from cgx.graph.backend import CodeGraphBackend


# ---------------------------
# Helpers / config
# ---------------------------

def _records_map(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(r.get("id")): r for r in records if isinstance(r, dict) and r.get("id")}


# ---------------------------
# Insertion-point exemplar corpus cache
# ---------------------------
#
# suggest_insertion_points re-encodes the same per-records "name + docstring"
# corpus on every call. For repeated queries against an unchanged records
# list (the common interactive case) the matrix is invariant -- only the
# query embedding changes. We memoize ``(mat, ids)`` keyed by the records
# list identity, its length, the schema_version of the first record, and
# the embedder's object identity. The cache is bounded with FIFO eviction
# so long-running processes don't grow without bound.
_INSERTION_CORPUS_CACHE_MAX = 8
_INSERTION_CORPUS_CACHE: "Dict[Tuple[int, int, int, int], Tuple[Any, List[str]]]" = {}
_INSERTION_CORPUS_ORDER: "List[Tuple[int, int, int, int]]" = []


def _insertion_corpus_key(
    records: List[Dict[str, Any]], embedder: Any,
) -> Tuple[int, int, int, int]:
    """Build a cheap, structural cache key for the exemplar corpus.

    Uses :func:`id` on the records list + embedder rather than hashing
    their contents; in-process callers that reuse the same objects hit
    the cache, while a rebuilt records list naturally misses (its ``id``
    differs and/or its length differs).
    """
    sv = 0
    for r in records:
        if isinstance(r, dict):
            sv = int(r.get("schema_version") or 0)
            break
    return (id(records), len(records), sv, id(embedder))


def _build_exemplar_corpus(
    records: List[Dict[str, Any]], embedder: Any,
) -> Tuple[Any, List[str]]:
    """Encode the file/class ``name + docstring`` corpus once, with caching."""
    key = _insertion_corpus_key(records, embedder)
    cached = _INSERTION_CORPUS_CACHE.get(key)
    if cached is not None:
        return cached
    texts: List[str] = []
    ids: List[str] = []
    for r in records:
        if not isinstance(r, dict):
            continue
        if r.get("type") in {"file", "class"}:
            desc = (r.get("name") or "") + " " + (r.get("docstring") or "")
            if desc.strip():
                texts.append(desc)
                ids.append(r.get("id"))
    if not texts:
        mat = np.zeros((0, 0), dtype=np.float32)
    else:
        mat = embedder.encode(texts)
    _INSERTION_CORPUS_CACHE[key] = (mat, ids)
    _INSERTION_CORPUS_ORDER.append(key)
    while len(_INSERTION_CORPUS_ORDER) > _INSERTION_CORPUS_CACHE_MAX:
        evict = _INSERTION_CORPUS_ORDER.pop(0)
        _INSERTION_CORPUS_CACHE.pop(evict, None)
    return mat, ids


def _clear_insertion_corpus_cache() -> None:
    """Test hook: drop all cached exemplar corpora."""
    _INSERTION_CORPUS_CACHE.clear()
    _INSERTION_CORPUS_ORDER.clear()

# Shared stopword set for filtering identifier-like tokens out of NL questions.
# Kept here so both the engine and the orchestrator agree on what counts as a
# real symbol token. Quoted/backticked tokens bypass this filter on purpose.
SYMBOL_STOPWORDS = frozenset({
    "from", "import", "which", "that", "this", "will", "would", "could",
    "should", "can", "may", "might", "if", "else", "elif", "for", "while",
    "with", "def", "class", "return", "true", "false", "none", "and", "or",
    "not", "is", "in", "on", "to", "by", "of", "at", "as", "do", "does", "did",
    "what", "where", "when", "who", "why", "how", "whose", "whom",
    "the", "a", "an", "it", "its", "there", "their", "they", "them",
    "are", "was", "were", "be", "been", "being", "has", "have", "had",
    "function", "functions", "method", "methods", "code", "codebase",
    "file", "files", "module", "modules", "repo", "repository", "package",
    "please", "show", "tell", "give", "find", "list", "describe", "explain",
    "any", "all", "some", "into", "onto", "out", "over", "under", "about",
    "between", "across", "after", "before", "than", "then", "so", "such",
    "use", "used", "uses", "using",
})

_MIN_SYMBOL_LEN = 3


def _extract_symbol_tokens(q: str) -> List[str]:
    """
    Extract candidate identifier tokens from a natural-language query.

    Quoted/backticked names always pass through. Bare identifiers are filtered
    against SYMBOL_STOPWORDS and required to be at least _MIN_SYMBOL_LEN chars.
    Returned list is deduplicated (lowercased) and sorted longest-first for
    stable downstream matching.
    """
    q = q or ""
    quoted = re.findall(r"[`\"]([A-Za-z_][A-Za-z0-9_]*)[`\"]", q)
    bare = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", q)
    # Keep the raw (case-preserving) tokens first so split_identifier can see
    # camel/Pascal boundaries; lower-casing + dedup happens inside
    # ``expand_with_subwords``. Quoted tokens are intentionally not stopword-
    # filtered (the user explicitly delimited them).
    raw: List[str] = []
    raw.extend(quoted)
    for t in bare:
        tl = t.lower()
        if len(tl) < _MIN_SYMBOL_LEN:
            continue
        if tl in SYMBOL_STOPWORDS:
            continue
        raw.append(t)
    # Sub-word expansion (symmetric with the indexer): "databaseReconnect"
    # produces ["databasereconnect", "database", "reconnect"] so partial-name
    # queries can still hit BM25 postings and trigger symbol_boost on full
    # '::'-segment matches of the chunk id.
    expanded = expand_with_subwords(raw, min_len=_MIN_SYMBOL_LEN)
    # Drop stopwords that may have leaked in via sub-word expansion (e.g.
    # an identifier split that produces a generic word). Sub-words shorter
    # than _MIN_SYMBOL_LEN were already filtered by ``expand_with_subwords``.
    expanded = [t for t in expanded if t not in SYMBOL_STOPWORDS]
    # Stable longest-first ordering for downstream regex/substring matching.
    expanded.sort(key=lambda s: (-len(s), s))
    return expanded


def _cid_segments(cid: str) -> List[str]:
    """Return the '::'-separated segments of a chunk id, lowercased."""
    return [seg.lower() for seg in str(cid).split("::") if seg]

@dataclass
class HybridConfig:
    k_intent: int = 50
    k_impl: int = 50
    k_lex: int = 50

    expand_top_n: int = 10
    expand_per_seed: int = 12   # accepted for cli_adapter compatibility
    graph_depth: int = 1
    relation_types: Optional[List[str]] = None  # ditto: edge-type filter
    rrf_k: float = 60.0

    top_k_chunks: int = 50
    top_k_files: int = 20
    top_k_classes: int = 20

    agg_alpha: float = 0.5
    agg_decay: float = 0.75
    agg_max_per_group: int = 6

    # Rerank knobs (post-RRF). The defaults preserve historical behavior.
    # Symbol-match boost applied to chunks whose name/id matches a quoted or
    # identifier-like token from the query. RRF scores at k=60 sit around
    # 0.01–0.05 per signal, so 0.5 deliberately dominates when the user
    # explicitly names a symbol.
    symbol_boost: float = 0.5
    # Graph-proximity bonus added to chunks reachable from top seeds. The
    # actual per-chunk increment is ``graph_bonus / (depth + 1)`` so closer
    # neighbors get more credit. Set to 0.0 to disable.
    graph_bonus: float = 0.2

    # Optional cross-encoder reranker over the top-N fused chunks. Disabled
    # by default to keep the no-torch path clean. When enabled, the rest of
    # the fused ordering is preserved as a tiebreaker for chunks below
    # ``reranker_top_n``.
    enable_reranker: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_n: int = 30
    reranker_weight: float = 1.0


# ---------------------------
# Public API
# ---------------------------

__all__ = [
    "build_two_view_indices",
    "hybrid_retrieve_two_view",
    "aggregate_by_file",
    "aggregate_by_class",
    "suggest_insertion_points",
    "analyze_change_impact",
]


# ---------------------------
# Index building
# ---------------------------


def hybrid_retrieve_two_view(
    query: str,
    *,
    indices: Optional[Any] = None,
    records: Optional[List[Dict[str, Any]]] = None,
    embedder: Any,
    chunks: Optional[List[Dict[str, Any]]] = None,
    G=None,
    top_k_per_view: int = 10,
    neighbor_depth: int = 1,
    use_lexical: bool = True,  # kept for API compatibility; we always include lexical anyway
    lexical_index: Optional[LexicalIndex] = None,
    enable_reranker: Optional[bool] = None,
    reranker_model: Optional[str] = None,
    reranker_top_n: Optional[int] = None,
    reranker_weight: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Hybrid retrieval over two views (intent + impl), lexical, and graph--then fuse.
    Nothing is optional; if a signal is unavailable, the step becomes a no-op.

    Parameters mirror the original API so existing callers (e.g., run_query_auto)
    work unchanged.
    """
    if query is None:
        query = ""

    # Wrap dict/persisted indices into a TwoViewIndex if needed
    tv = _two_view_index_from_dict(indices, records=records, embedder=embedder)

    # Ensure we have records/chunks lists (even if empty) for consistent behavior
    recs = records or []
    chs = chunks or []

    # Internal candidate pools are deliberately decoupled from the caller's
    # display-time `top_k_per_view`. RRF/graph expansion/symbol boosting only
    # work when each signal contributes a healthy pool. We use generous
    # internal pools (~50) and only restrict the final returned list to
    # `top_k_per_view`. Callers can still cap display by post-slicing.
    k_pool = max(int(top_k_per_view), 50)
    cfg = HybridConfig(
        k_intent=k_pool,
        k_impl=k_pool,
        k_lex=k_pool,
        expand_top_n=max(10, int(top_k_per_view)),
        graph_depth=max(0, neighbor_depth),
        rrf_k=60.0,
        top_k_chunks=max(int(top_k_per_view), 20),
        top_k_files=max(int(top_k_per_view), 20),
        top_k_classes=max(int(top_k_per_view), 20),
        agg_alpha=0.5,
        agg_decay=0.75,
        agg_max_per_group=6,
    )
    # Reranker knobs are opt-in: keep the dataclass defaults (disabled) unless
    # the caller explicitly threads a profile-derived value through.
    if enable_reranker is not None:
        cfg.enable_reranker = bool(enable_reranker)
    if reranker_model is not None:
        cfg.reranker_model = str(reranker_model)
    if reranker_top_n is not None:
        cfg.reranker_top_n = int(reranker_top_n)
    if reranker_weight is not None:
        cfg.reranker_weight = float(reranker_weight)

    retriever = HybridRetriever(
        tv_index=tv,
        records=recs,
        lexical_index=lexical_index,
        chunks=chs,
        G=G,
    )

    out = retriever.search(query, embedder=embedder, cfg=cfg)

    # out already includes hits/top_files/top_classes; anchors left empty by design here.
    # keep the exact return shape expected by callers.
    return {
        "hits": out.get("hits", []),
        "top_files": out.get("top_files", []),
        "top_classes": out.get("top_classes", []),
        "anchors": out.get("anchors", []),
    }


def build_two_view_indices(
    records: List[Dict[str, Any]],
    *,
    embedder: Any,
    metric: str = "cosine",
    index_type: str = "flat",
) -> TwoViewIndex:
    """
    Build a TwoViewIndex from parsed records (intent + impl views).
    """
    return TwoViewIndex.from_records(
        records,
        embedder=embedder,
        index_builder=build_ann_index,
        metric=metric,
        index_type=index_type,
    )


# ---------------------------
# Utility to wrap persisted dict indices
# ---------------------------

def _two_view_index_from_dict(
    indices: Any,
    records: Optional[List[Dict[str, Any]]] = None,
    *,
    embedder: Any = None,
) -> TwoViewIndex:
    """
    If given a TwoViewIndex, return it.
    If given a dict (loaded from disk), wrap its ANN indices + rows
    into a TwoViewIndex without re-embedding.
    """
    if isinstance(indices, TwoViewIndex):
        return indices
    if not isinstance(indices, dict):
        raise TypeError(f"indices must be dict or TwoViewIndex, got {type(indices)}")

    views = indices.get("views", {})
    intent = views.get("intent") or {}
    impl   = views.get("impl") or {}

    if not (intent.get("rows") or impl.get("rows")):
        raise ValueError("_two_view_index_from_dict: no rows available in indices dict.")

    tv = TwoViewIndex()

    if intent.get("rows") and intent.get("index") is not None:
        tv._intent = ViewSlice(
            rows=intent["rows"],
            ids=np.arange(len(intent["rows"]), dtype=np.int64),
            index=intent["index"],
            meta=intent.get("meta") or {},
            dim=None,
        )

    if impl.get("rows") and impl.get("index") is not None:
        tv._impl = ViewSlice(
            rows=impl["rows"],
            ids=np.arange(len(impl["rows"]), dtype=np.int64),
            index=impl["index"],
            meta=impl.get("meta") or {},
            dim=None,
        )

    # Build chunk->rows mapping
    tv._chunk_id_to_rows = {}
    for view_name, vs in (("intent", tv._intent), ("impl", tv._impl)):
        if vs is None:
            continue
        for ridx, row in enumerate(vs.rows):
            cid = row.get("chunk_id")
            if isinstance(cid, str):
                tv._chunk_id_to_rows.setdefault(cid, {}).setdefault(view_name, []).append(ridx)

    return tv


# ---------------------------
# Hybrid retrieval (ALL SIGNALS)
# ---------------------------

class HybridRetriever:
    """
    Orchestrates multi-signal retrieval across codebase views.

    This retriever fuses several complementary signals to rank code chunks:
      • Two-view ANN (semantic embeddings): searches both the "intent" view
        (docstrings, signatures, comments) and the "impl" view (code bodies).
      • Lexical search: BM25 over records + regex fallback over raw chunks
        to catch exact string matches not well-covered by embeddings.
      • Graph expansion: traverses the knowledge graph outward from strong
        matches to include related symbols (callers, callees, class members).
      • Reciprocal Rank Fusion (RRF): merges ranked lists from all signals
        into a unified set of candidate chunks.
      • Symbol boosting: explicitly rewards matches where the query mentions
        an identifier by name.
      • Aggregation: groups chunk-level results into files and classes to
        provide higher-level context for suggested changes.

    Attributes
    ----------
    tv : TwoViewIndex
        Semantic index for intent and impl views.
    records : list[dict]
        Canonical record list for all chunks (identity + metadata).
    lex : LexicalIndex or None
        Lexical index (BM25-like) over the records, built on demand.
    chunks : list[dict]
        Raw parsed chunks, used for regex fallback matching.
    G : networkx.Graph or None
        Knowledge graph of code entities and relations.

    Usage
    -----
    >>> retriever = HybridRetriever(tv_index=two_view, records=records, chunks=chunks, G=graph)
    >>> results = retriever.search("optimize matrix multiply", embedder=my_embedder)
    >>> results["hits"][0]
    {
        "chunk_id": "ml/linalg.py::function::matmul_optimized",
        "score": 13.42,
        "rank": 1,
        "provenance": {"intent_rank": 1, "lexical_count": 2, "graph_depth": 0}
    }

    This shows how multiple signals contributed to the final score/provenance
    of a retrieved chunk.
    """

    def __init__(
        self,
        *,
        tv_index: TwoViewIndex,
        records: List[Dict[str, Any]],
        lexical_index: Optional[LexicalIndex] = None,
        chunks: Optional[List[Dict[str, Any]]] = None,
        G=None
    ) -> None:
        
        self.tv = tv_index
        self.records = records or []
        self.lex = lexical_index
        self.chunks = chunks or []
        self.G = G
        self._rec_by_id = _records_map(self.records)

    def _expand_multi_hop(
        self,
        seeds: List[str],
        *,
        max_depth: int = 1,
    ) -> Dict[str, int]:
        """
        Perform a breadth-first multi-hop expansion in the knowledge graph (self.G).

        Starting from a set of seed nodes, this walks outward up to `max_depth`
        and records how far each discovered node is from the seeds.
        The traversal is undirected in spirit: both successors (outgoing edges)
        and predecessors (incoming edges) are considered.

        Parameters
        ----------
        seeds : list[str]
            Node IDs (chunk_ids) to start expansion from.
        max_depth : int, default=1
            Maximum hop distance to explore from the seeds.

        Returns
        -------
        dict[str, int]
            Mapping of discovered node_id → distance (number of hops from
            the nearest seed). The seeds themselves are not included in the output.

        Example
        -------
        Suppose the graph has edges:
            A -> B,  B -> C,  C -> D

        >>> self._expand_multi_hop(["A"], max_depth=2)
        {"B": 1, "C": 2}

        Here:
          - "B" is 1 hop away from "A"
          - "C" is 2 hops away from "A"
          - "D" is not included because it is 3 hops away (> max_depth).
        """
        depths: Dict[str, int] = {}
        backend = CodeGraphBackend.wrap(self.G)
        if backend is None or not seeds:
            return depths
        q: List[Tuple[str, int]] = [(sid, 0) for sid in seeds if backend.has_node(sid)]
        visited = set(sid for sid, _ in q)
        while q:
            nid, d = q.pop(0)
            if d >= max_depth:
                continue
            for nbr in backend.undirected_neighbors(nid):
                if nbr in visited:
                    continue
                visited.add(nbr)
                depths[str(nbr)] = d + 1
                q.append((nbr, d + 1))
        return depths

    def search(
        self,
        query: str,
        *,
        embedder: Any,
        cfg: Optional[HybridConfig] = None,
    ) -> Dict[str, Any]:
        """
        Run a hybrid retrieval over all available signals (semantic, lexical, graph).

        The search pipeline executes the following steps in order:
          1. Semantic search on both views ("intent" and "impl") using ANN indices.
          2. Lexical search:
             - BM25-like scoring over records (names, docstrings, code snippets).
             - Regex fallback over raw chunk text to catch literal mentions.
          3. Reciprocal Rank Fusion (RRF) of all ranked lists to unify results.
          4. Graph expansion: multi-hop traversal from top seeds to pull in
             related chunks (callers, callees, class members).
          5. Symbol boosting: increase scores when query explicitly mentions
             an identifier that matches a chunk’s name/id.
          6. Final aggregation:
             - Chunk-level hits with provenance for each contributing signal.
             - Grouped summaries by file and by class.

        Parameters
        ----------
        query : str
            The natural language or keyword query string to search for.
        embedder : Any
            Embedding model with `.encode(list[str]) -> np.ndarray` used for
            semantic retrieval on both views.
        cfg : HybridConfig, optional
            Configuration object controlling top-k cutoffs, RRF strength,
            graph depth, and aggregation parameters. If not provided,
            a default `HybridConfig()` is used.

        Returns
        -------
        dict
            Dictionary containing:
              • "hits": list of per-chunk results
                [{"chunk_id": str, "score": float, "rank": int, "provenance": {...}}, ...]
              • "top_files": aggregated file-level results
              • "top_classes": aggregated class-level results
              • "anchors": legacy placeholder (empty; use suggest_insertion_points)

        Example
        -------
        >>> retriever.search("optimize matrix multiply", embedder=my_embedder)
        {
            "hits": [
                {"chunk_id": "ml/linalg.py::function::matmul_optimized",
                 "score": 13.42, "rank": 1,
                 "provenance": {"intent_rank": 1, "lexical_count": 2, "graph_depth": 0}}
            ],
            "top_files": [{"file": "ml/linalg.py", "score": 18.5, "members": [...]}],
            "top_classes": [],
            "anchors": []
        }
        """        
        cfg = cfg or HybridConfig()
        lists_for_rrf: List[List[Dict[str, Any]]] = []
        provenance: Dict[str, Dict[str, Any]] = {}

        # --- semantic per view (intent + impl) -- run both views in parallel ---
        available_views = [(v, k) for v, k in (("intent", cfg.k_intent), ("impl", cfg.k_impl))
                           if v in self.tv.available_views()]

        def _search_view(view_k):
            view, k = view_k
            return view, self.tv.search_view(view, query, embedder=embedder, top_k=k)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as _ex:
            view_results = list(_ex.map(_search_view, available_views))

        for view, hits in view_results:
            lists_for_rrf.append([{"chunk_id": h["chunk_id"], "rank": h["rank"]} for h in hits])
            for h in hits:
                provenance.setdefault(h["chunk_id"], {}).update({
                    f"{view}_rank": h["rank"],
                    f"{view}_score": h["score"],
                })

        logger.debug("provenance %s", provenance)
        
        # --- lexical (required): BM25 over records; fallback regex over chunks ---
        """
        Perform lexical retrieval in two stages:

        1. BM25 search (preferred):
           - Build a lightweight BM25 index (LexicalIndex) from all records.
           - Query it with the text form of the user query.
           - Returns top-k chunks ranked by keyword overlap frequency and rarity.

        2. Regex fallback (safety net):
           - If BM25 finds nothing useful, or as additional evidence,
             extract symbol-like tokens from the query (identifiers, quoted names).
           - Build a regex to scan multiple chunk fields (code, name, id, file, docstring).
           - Count matches per chunk and rank chunks by frequency.

        Both result sets (BM25 and regex) are added into the fusion lists for RRF.

        Example
        -------
        Suppose the query is "optimize matrix multiplication".

        - BM25 may surface chunks whose text contains words like
          "optimize", "matrix", "multiplication" across docstrings or code.

        - Regex fallback may trigger if the user types a symbol-like query such as
          "MatMulOptimizer". Even if BM25 doesn't weight it well, the regex will
          directly match that exact substring in identifiers, ensuring the chunk
          still gets boosted into the candidate pool.
        """
        # Always try BM25 on records
        lex_hits = []
        if self.records:
            self.lex = self.lex or LexicalIndex.from_records(self.records)
            lex_hits = self.lex.search(query, top_k=cfg.k_lex)

        # regex fallback adds additional evidence if present
        regex_hits = []
        if self.chunks:
            tokens = _extract_symbol_tokens(query)
            if not tokens:
                tokens = [t for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", query)]
            if tokens:
                pat = r"(" + "|".join(re.escape(t) for t in tokens) + r")"
                rx = re.compile(pat, re.IGNORECASE)
                counts: Dict[str, int] = {}
                for ch in self.chunks:
                    cid = str(ch.get("id") or "")
                    if not cid:
                        continue
                    total = 0
                    for f in ("code", "name", "id", "file", "meta.docstring"):
                        cur: Any = ch
                        for p in f.split("."):
                            cur = cur.get(p, "") if isinstance(cur, dict) else ""
                        s = str(cur or "")
                        if s:
                            total += len(list(rx.finditer(s)))
                    if total:
                        counts[cid] = total
                        provenance.setdefault(cid, {}).update({"lexical_count": total})
                if counts:
                    sorted_ids = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
                    regex_hits = [{"chunk_id": cid, "rank": i+1} for i, (cid, _) in enumerate(sorted_ids[:cfg.k_lex])]

        # Add lexical lists to fusion inputs (both sources if available)
        if lex_hits:
            lists_for_rrf.append([{"chunk_id": h["chunk_id"], "rank": h["rank"]} for h in lex_hits])
            for h in lex_hits:
                provenance.setdefault(h["chunk_id"], {}).update({"lexical_count": 1})
        if regex_hits:
            lists_for_rrf.append(regex_hits)

        if not lists_for_rrf:
            return {"hits": [], "top_files": [], "top_classes": [], "anchors": []}

        logger.debug("lists_for_rrf %s", lists_for_rrf)


        # --- RRF fusion of all lists ---
        """
        Apply Reciprocal Rank Fusion (RRF) to combine all retrieval signals.

        Each list in `lists_for_rrf` comes from a different signal:
          - semantic intent hits
          - semantic implementation hits
          - lexical BM25 hits
          - regex fallback hits
          - (graph seeds may be injected later)

        RRF assigns a score to each chunk_id based on its rank in each list:
            score = Σ (1 / (k + rank))
        where `k` is a constant (cfg.rrf_k, usually 60).

        Intuition:
          - Items that appear in multiple lists get higher total scores.
          - High ranks (rank=1,2,3…) contribute more than low ranks.
          - This allows a symbol that is “pretty good” across all views
            to outrank something that is “excellent” in only one view.

        Example
        -------
        Suppose "foo()" appears:
          - rank 2 in semantic-intent results,
          - rank 5 in semantic-impl results,
          - rank 10 in BM25 lexical.

        With k=60, the score is:
          1/(60+2) + 1/(60+5) + 1/(60+10)
          ≈ 0.016 + 0.016 + 0.015 = 0.047

        Another symbol that only shows up rank 1 in a single list would get:
          1/(60+1) ≈ 0.016

        So "foo()" is ranked higher overall, because it was consistently
        present across different signals, which is exactly the desired effect.
        """
        fused = rrf_fuse(lists_for_rrf, k=cfg.rrf_k, top_k=cfg.top_k_chunks + cfg.expand_top_n)

        logger.debug("fused %s", fused)

        # --- graph expansion (required shape; no-op if G is None) ---
        """
        Expand the top fused chunks into their neighbors on the knowledge graph.

        Process:
          1. Take the top-N "seed" chunks after initial fusion
             (cfg.expand_top_n controls N).
          2. Perform a breadth-first expansion up to `cfg.graph_depth`
             over both predecessors and successors in the graph G.
          3. For each neighbor found:
             - record the depth (hop count) in provenance,
             - apply a small score bonus inversely proportional to depth
               (closer neighbors get more credit).

        Purpose:
          This step brings in chunks that may not match the query text directly
          but are structurally related to strong hits (e.g., callers, callees,
          parent/child classes). It ensures retrieval respects *code structure*
          in addition to text/embeddings.

        Example
        -------
        Query: "optimize matrix multiply"

        - Semantic search finds `matmul_optimized()` and ranks it high.
        - Graph expansion walks one hop outward:
            • Finds `train_model()` which calls `matmul_optimized()`.
            • Finds helper function `pack_tensors()` used inside `matmul_optimized()`.

        These related chunks get pulled in with a mild bonus,
        so they appear in the candidate set for fusion/aggregation,
        even if they didn't match the text query strongly.
        """
        seeds = [cid for cid, _ in fused[:cfg.expand_top_n]]
        graph_depths = self._expand_multi_hop(seeds, max_depth=cfg.graph_depth)
        # Fix: previously this loop only updated scores for chunks already in
        # ``fused``, so graph-only neighbors (discovered via expansion but not
        # surfaced by semantic/lexical) were silently dropped. We now both
        # bump existing chunks and append graph-only neighbors with a base
        # score of ``graph_bonus / (depth + 1)``.
        in_fused = {c for c, _ in fused}
        for cid, depth in graph_depths.items():
            # Keep graph-only neighbors restricted to chunks we actually have
            # records for; otherwise they can't be aggregated to file/class.
            if cid not in self._rec_by_id and cid not in in_fused:
                continue
            provenance.setdefault(cid, {}).update({"graph_depth": depth})
        if cfg.graph_bonus > 0.0 and graph_depths:
            bumped: List[Tuple[str, float]] = []
            for c, sc in fused:
                d = graph_depths.get(c)
                bumped.append((c, sc + cfg.graph_bonus / (d + 1)) if d is not None else (c, sc))
            fused = bumped
            for cid, depth in graph_depths.items():
                if cid in in_fused:
                    continue
                if cid not in self._rec_by_id:
                    continue
                fused.append((cid, cfg.graph_bonus / (depth + 1)))

        logger.debug("fused after graph %s", fused)

        # --- symbol boosting (quote or exact name matches) ---
        """
        Give an extra score boost to chunks whose symbol names directly match
        identifiers mentioned in the query.

        Process:
          - Extract symbol-like tokens from the query (e.g., names inside quotes
            or identifiers such as `build_two_view_indices`).
          - For each candidate chunk:
              • Look at its recorded `name` and its `chunk_id`.
              • If either exactly matches or contains one of the tokens,
                boost its score by a fixed amount (+0.5).
          - Record in provenance that a symbol match occurred.

        Purpose:
          This ensures that when the user explicitly names a function/class/variable,
          that chunk (and its file) reliably surfaces near the top,
          even if embeddings/lexical overlap are weaker.

        Example
        -------
        Query: "Where is `build_two_view_indices` used?"

        - `_extract_symbol_tokens` → ["build_two_view_indices"]
        - A chunk with name = "build_two_view_indices" gets a +0.5 score boost.
        - Even if it was originally ranked lower by embeddings,
          this boost ensures it appears as a top candidate,
          since the user explicitly mentioned it.
        """
        sym_tokens = _extract_symbol_tokens(query)
        if sym_tokens and cfg.symbol_boost > 0.0:
            tok_set = set(sym_tokens)
            boosted: Dict[str, float] = {}
            for cid, _ in list(fused):
                rec = self._rec_by_id.get(cid) or {}
                nm = str(rec.get("name") or "").lower()
                segs = set(_cid_segments(cid))
                # Require exact match against the record name OR a full '::'
                # segment of the chunk id, never a bare substring (avoids
                # short tokens like 'add' matching 'add_calls_edges' etc.).
                if (nm and nm in tok_set) or (segs & tok_set):
                    provenance.setdefault(cid, {})["symbol_match"] = True
                    boosted[cid] = cfg.symbol_boost
            if boosted:
                fused = [(c, sc + boosted.get(c, 0.0)) for c, sc in fused]

        logger.debug("fused after sym_tokens %s", fused)

        # --- optional cross-encoder reranker (opt-in, post-RRF) ---
        if cfg.enable_reranker and fused:
            try:
                from cgx.retrieval.reranker import rerank_chunks
                fused = rerank_chunks(
                    query=query,
                    fused=fused,
                    rec_by_id=self._rec_by_id,
                    chunks=self.chunks,
                    model_name=cfg.reranker_model,
                    top_n=cfg.reranker_top_n,
                    weight=cfg.reranker_weight,
                    provenance=provenance,
                )
            except Exception as e:
                logger.warning("Cross-encoder reranker disabled (%s: %s); falling back to RRF order.",
                               type(e).__name__, e)

        # --- finalize chunks ---
        """
        Take the fused list of chunk IDs and scores, sort them by score,
        and keep only the top N (cfg.top_k_chunks).

        Each result is packaged with:
          - chunk_id: the unique identifier for the code chunk
          - score   : final fused score (float)
          - rank    : position after sorting
          - provenance: dictionary of how/why this chunk scored
            (semantic ranks/scores, lexical matches, graph depth, symbol boost, etc.)

        Example
        -------
        A chunk "foo.py::func::bar" might get:
          {
            "chunk_id": "foo.py::func::bar",
            "score": 0.83,
            "rank": 3,
            "provenance": {
              "intent_rank": 2,
              "impl_rank": 5,
              "lexical_count": 1,
              "graph_depth": 1,
              "symbol_match": True
            }
          }
        """
        chunk_scores: List[Tuple[str, float]] = sorted(fused, key=lambda kv: -kv[1])[:cfg.top_k_chunks]
        chunk_results = []
        for i, (cid, sc) in enumerate(chunk_scores, start=1):
            prov = provenance.get(cid, {})
            chunk_results.append({"chunk_id": cid, "score": float(sc), "rank": i, "provenance": prov})

        # --- aggregate to files/classes ---
        """
        After ranking chunks, aggregate them at higher levels of granularity:
          - By file: group chunks by their source file.
          - By class: group chunks by their parent class (if any).

        The `_aggregate_group` helper:
          - Groups chunks by key (file or class).
          - Takes the best chunk score + a decayed contribution from other
            chunks in the same group.
          - Produces a total score per group along with its top members.

        This allows surfacing *containers* (files/classes) most relevant
        to the query, not just individual chunks.
        """

        def _file_of(cid: str) -> Optional[str]:
            """Return the file path of a chunk, or None if unknown."""
            r = self._rec_by_id.get(cid)
            return r.get("file") if r else None

        def _class_of(cid: str) -> Optional[str]:
            """Return the parent class id of a chunk, or None if not in a class."""
            r = self._rec_by_id.get(cid)
            return r.get("parent_class_id") if r else None

        def _aggregate_group(
            pairs: List[Tuple[str, float]],
            key_fn,
            alpha: float = cfg.agg_alpha,
            decay: float = cfg.agg_decay,
            max_per_group: int = cfg.agg_max_per_group
        ) -> List[Tuple[str, float, List[Tuple[str, float]]]]:
            """
            Group chunk scores by a container key (file or class).

            - `best`: highest chunk score in the group.
            - `extra`: weighted sum of other members' scores,
               with geometric decay (decay^i).
            - `total`: best + alpha * extra.

            Returns a sorted list of (group_id, total_score, members).
            """
            by: Dict[str, List[Tuple[str, float]]] = {}
            for cid, s in pairs:
                gid = key_fn(cid)
                if gid:
                    by.setdefault(gid, []).append((cid, s))
            out: List[Tuple[str, float, List[Tuple[str, float]]]] = []
            for gid, items in by.items():
                items = sorted(items, key=lambda kv: (-kv[1], kv[0]))
                best = items[0][1]
                extra = sum((decay ** (i - 1)) * sc for i, (_, sc) in enumerate(items[1:max_per_group], start=1))
                total = best + alpha * extra
                out.append((gid, float(total), items[:max_per_group]))
            out.sort(key=lambda kv: (-kv[1], kv[0]))
            return out

        files = _aggregate_group(chunk_scores, key_fn=_file_of)
        classes = _aggregate_group(chunk_scores, key_fn=_class_of)

        # --- format aggregated results ---
        """
        Convert the grouped results into dictionaries suitable for downstream use.

        Example output
        --------------
        "top_files": [
          {
            "file": "utils/math_ops.py",
            "score": 1.23,
            "members": [
              {"chunk_id": "utils/math_ops.py::func::matmul", "score": 0.83},
              {"chunk_id": "utils/math_ops.py::func::normalize", "score": 0.40}
            ]
          }
        ]
        """
        file_results = [{
            "file": gid,
            "score": float(sc),
            "members": [{"chunk_id": cid, "score": float(s)} for cid, s in items]
        } for gid, sc, items in files[:cfg.top_k_files] if gid]

        class_results = [{
            "class_id": gid,
            "score": float(sc),
            "members": [{"chunk_id": cid, "score": float(s)} for cid, s in items]
        } for gid, sc, items in classes[:cfg.top_k_classes] if gid]

        return {
            "hits": chunk_results,
            "top_files": file_results,
            "top_classes": class_results,
            "anchors": [],  # legacy placeholder (use suggest_insertion_points if needed)
        }


# ---------------------------
# Aggregation helpers (public)
# ---------------------------

def _record_map(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(r.get("id")): r for r in records if isinstance(r, dict) and r.get("id")}

def aggregate_by_file(
    fused_hits: List[Dict[str, Any]],
    records: List[Dict[str, Any]],
    *,
    top_symbols_per_file: int = 2,
    centrality_bonus: float = 0.15,
) -> List[Dict[str, Any]]:
    rec_map = _record_map(records)
    buckets: Dict[str, List[Tuple[str, float]]] = {}

    for hit in fused_hits:
        cid = hit.get("chunk_id")
        sc = float(hit.get("score", 0.0))
        r = rec_map.get(str(cid))
        if not r:
            continue
        f = r.get("file")
        if not f:
            continue
        buckets.setdefault(f, []).append((str(cid), sc))

    out: List[Dict[str, Any]] = []
    for f, pairs in buckets.items():
        total = sum(sc for _, sc in pairs)
        deg = sum(
            int(rec_map.get(cid, {}).get("calls_in_count", 0))
            + int(rec_map.get(cid, {}).get("calls_out_count", 0))
            for cid, _ in pairs
        )
        bonus = centrality_bonus * math.log1p(deg)
        pairs_sorted = sorted(pairs, key=lambda kv: kv[1], reverse=True)
        top_syms = [cid for cid, _ in pairs_sorted[: int(top_symbols_per_file)]]
        out.append({"file": f, "score": float(total + bonus), "top_symbols": top_syms})

    out.sort(key=lambda d: d["score"], reverse=True)
    return out


def aggregate_by_class(
    fused_hits: List[Dict[str, Any]],
    records: List[Dict[str, Any]],
    *,
    top_methods_per_class: int = 2,
) -> List[Dict[str, Any]]:
    rec_map = _record_map(records)
    buckets: Dict[str, List[Tuple[str, float]]] = {}

    for hit in fused_hits:
        cid = str(hit.get("chunk_id"))
        sc = float(hit.get("score", 0.0))
        r = rec_map.get(cid)
        if not r:
            continue
        parent_cls = r.get("parent_class_id")
        if parent_cls:
            buckets.setdefault(parent_cls, []).append((cid, sc))

    out: List[Dict[str, Any]] = []
    for cls_id, pairs in buckets.items():
        total = sum(sc for _, sc in pairs)
        pairs_sorted = sorted(pairs, key=lambda kv: kv[1], reverse=True)
        top_methods = [cid for cid, _ in pairs_sorted[: int(top_methods_per_class)]]
        out.append({"class_id": cls_id, "score": float(total), "top_methods": top_methods})

    out.sort(key=lambda d: d["score"], reverse=True)
    return out


# ---------------------------
# Insertion point suggestions (uses networkx)
# ---------------------------

def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 0.0
    return float(len(A & B)) / max(1, len(A | B))


def suggest_insertion_points(
    query: str,
    fused_hits: List[Dict[str, Any]],
    records: List[Dict[str, Any]],
    *,
    k_candidates: int = 5,
    k_exemplars: int = 5,
    embedder: Optional[Any] = None,
    G=None,
) -> List[Dict[str, Any]]:
    """
    Suggest where in the codebase a new change should be inserted.
    Combines retrieval overlap, semantic similarity, graph proximity, signature overlap.
    """
    rec_map = _record_map(records)

    exemplar_ids = [str(h.get("chunk_id")) for h in fused_hits[: int(k_exemplars)]]

    # Signals from exemplars
    imports, attrs, params = set(), set(), []
    for cid in exemplar_ids:
        r = rec_map.get(str(cid)) or {}
        imports.update([imp for imp in r.get("imports_used") or [] if isinstance(imp, str)])
        attrs.update([ar for ar in r.get("attributes_used_root_reads") or [] if isinstance(ar, str)])
        sig = r.get("signature") or ""
        m = re.search(r"\((.*)\)", sig)
        if m:
            inner = m.group(1)
            names = [p.strip().split(":")[0].split("=")[0] for p in inner.split(",") if p.strip()]
            params.extend([n for n in names if n and n != "self"])
    sigs = {"imports": sorted(imports), "attributes": sorted(attrs), "param_names": sorted(params)}

    exemplar_containers: set[str] = set()
    for cid in exemplar_ids:
        rec = rec_map.get(cid) or {}
        parent = rec.get("parent_class_id") or rec.get("file")
        if parent:
            exemplar_containers.add(parent)

    # semantic similarity to container names/docstrings
    sem_scores: Dict[str, float] = {}
    if embedder is not None:
        q_emb = embedder.encode([query])[0]
        # The exemplar corpus (file/class name + docstring) only depends on
        # the records list; cache it across repeated calls so a long-running
        # interactive session doesn't re-encode it per query.
        mat, ids = _build_exemplar_corpus(records, embedder)
        if ids:
            sims = np.dot(mat, q_emb) / (np.linalg.norm(mat, axis=1) * np.linalg.norm(q_emb) + 1e-9)
            for i, rid in enumerate(ids):
                sem_scores[rid] = float(sims[i])

    # graph proximity to exemplars
    graph_scores: Dict[str, float] = {}
    backend = CodeGraphBackend.wrap(G)
    if backend is not None and exemplar_ids:
        for cid in exemplar_ids:
            if not backend.has_node(cid):
                continue
            lengths = backend.bfs_distances(cid, cutoff=2)
            for nid, dist in lengths.items():
                rec = rec_map.get(nid) or {}
                parent = rec.get("parent_class_id") or rec.get("file")
                if parent:
                    score = 1.0 / (1 + dist)
                    graph_scores[parent] = max(graph_scores.get(parent, 0.0), score)

    # baseline overlap by imports/attrs/signatures
    def _best_sig_overlap(child_ids: List[str]) -> float:
        best = 0.0
        for cid in child_ids:
            rr = rec_map.get(cid) or {}
            sig = rr.get("signature") or ""
            m = re.search(r"\((.*)\)", sig)
            if not m:
                continue
            names = [p.strip().split(":")[0].split("=")[0] for p in m.group(1).split(",") if p.strip()]
            names = [n for n in names if n and n != "self"]
            best = max(best, _jaccard(names, sigs["param_names"]))
        return best

    file_scores: Dict[str, float] = {}
    class_scores: Dict[str, float] = {}
    children_by_container: Dict[str, List[str]] = {}

    for r in records:
        rid = r.get("id")
        if not rid:
            continue
        rtype = r.get("type")
        children_by_container[rid] = list(r.get("defines_children_ids") or [])
        if rtype == "file":
            s_imp = _jaccard(r.get("imports_used", []), sigs["imports"])
            s_att = _jaccard(r.get("attributes_used_root_reads", []), sigs["attributes"])
            base = 0.55 * s_imp + 0.45 * s_att + 0.3 * _best_sig_overlap(children_by_container[rid])
            file_scores[rid] = base
        elif rtype == "class":
            s_imp = _jaccard(r.get("imports_used", []), sigs["imports"])
            s_att = _jaccard(r.get("attributes_used_root_reads", []), sigs["attributes"])
            base = 0.55 * s_imp + 0.45 * s_att + 0.3 * _best_sig_overlap(children_by_container[rid])
            class_scores[rid] = base

    def _final_score(rid: str, base: float) -> float:
        score = base
        if rid in exemplar_containers:
            score += 0.4
        score += 0.3 * sem_scores.get(rid, 0.0)
        score += 0.2 * graph_scores.get(rid, 0.0)
        return score

    file_ranked = sorted(((rid, _final_score(rid, sc)) for rid, sc in file_scores.items()),
                         key=lambda kv: (-kv[1], kv[0]))[:k_candidates]

    class_ranked = sorted(((rid, _final_score(rid, sc)) for rid, sc in class_scores.items()),
                          key=lambda kv: (-kv[1], kv[0]))[:k_candidates]

    def _likely_caller(child_ids: List[str]) -> Optional[str]:
        best_id, best_deg = None, -1
        for cid in child_ids:
            rr = rec_map.get(cid) or {}
            deg = int(rr.get("calls_in_count", 0))
            if deg > best_deg:
                best_id, best_deg = cid, deg
        return best_id

    def _similar_signature_neighbor(child_ids: List[str]) -> Optional[str]:
        best_id, best_sim = None, -1.0
        for cid in child_ids:
            rr = rec_map.get(cid) or {}
            sig = rr.get("signature") or ""
            m = re.search(r"\((.*)\)", sig)
            if not m:
                continue
            names = [p.strip().split(":")[0].split("=")[0] for p in m.group(1).split(",") if p.strip()]
            names = [n for n in names if n and n != "self"]
            sim = _jaccard(names, sigs["param_names"])
            if sim > best_sim:
                best_id, best_sim = cid, sim
        return best_id

    def _loc_for(cid: Optional[str]) -> Optional[Dict[str, int]]:
        """Return start_line/end_line/indent_col from the record map (v3 schema).

        Yields ``None`` when the chunk id is missing, unknown, or the record
        has no usable line span (start_line == 0). Downstream consumers
        (ast_insert) treat ``None`` as "no anchor" and fall back to AST walks.
        """
        if not cid:
            return None
        rr = rec_map.get(str(cid)) or {}
        sl = int(rr.get("start_line") or 0)
        el = int(rr.get("end_line") or 0)
        if sl <= 0 or el <= 0:
            return None
        return {
            "start_line": sl,
            "end_line": el,
            "indent_col": int(rr.get("col_offset") or 0),
        }

    out: List[Dict[str, Any]] = []
    for rid, sc in class_ranked:
        kids = children_by_container.get(rid, [])
        lc = _likely_caller(kids)
        ss = _similar_signature_neighbor(kids)
        out.append({
            "container_type": "class",
            "container_id": rid,
            "score": float(sc),
            "anchors": {
                "likely_caller": lc,
                "likely_caller_loc": _loc_for(lc),
                "similar_signature_neighbor": ss,
                "similar_signature_neighbor_loc": _loc_for(ss),
            }
        })
    for rid, sc in file_ranked:
        kids = children_by_container.get(rid, [])
        lc = _likely_caller(kids)
        ss = _similar_signature_neighbor(kids)
        out.append({
            "container_type": "file",
            "container_id": rid,
            "score": float(sc),
            "anchors": {
                "likely_caller": lc,
                "likely_caller_loc": _loc_for(lc),
                "similar_signature_neighbor": ss,
                "similar_signature_neighbor_loc": _loc_for(ss),
            }
        })

    out.sort(key=lambda d: (-d["score"], d["container_type"], d["container_id"]))
    return out


# ---------------------------
# Change impact analysis (NEW)
# ---------------------------

def analyze_change_impact(
    symbol_query: str,
    fused_hits: List[Dict[str, Any]],
    records: List[Dict[str, Any]],
    G=None,
    *,
    depth: int = 2,
    max_results: int = 20,
) -> List[Dict[str, Any]]:
    """
    Given a (likely) symbol name in `symbol_query` (e.g., "build_two_view_indices"),
    compute a ranked list of *files* likely impacted if that symbol changes.

    Algorithm (deterministic):
      1) Identify seed chunks:
         - exact name matches in records (case-insensitive)
         - otherwise top fused hits filtered by symbol tokens
      2) Collect owning file(s) of seeds (direct impact +1.0).
      3) Traverse the graph up to `depth`: callers, callees, defines edges.
         - For each reached chunk, add its owning file with decayed weights.
      4) Add light lexical boost to files whose file path contains the token.
      5) Return unique files sorted by total score, with reasons.

    Returns:
      [{ "file": <path>, "score": float, "reasons": {...} }, ...]
    """
    rec_map = _record_map(records)
    tokens = _extract_symbol_tokens(symbol_query)
    if not tokens:
        return []

    # (1) seeds by exact name match
    seeds: List[str] = []
    wanted = tokens[0]
    for r in records:
        nm = str(r.get("name") or "").lower()
        if nm == wanted:
            cid = str(r.get("id") or "")
            if cid:
                seeds.append(cid)

    # fallback: top fused hits whose name matches a token or whose chunk_id
    # contains the token as a full '::'-segment
    if not seeds:
        tok_set = set(tokens)
        for h in fused_hits:
            cid = str(h.get("chunk_id") or "")
            if not cid:
                continue
            r = rec_map.get(cid) or {}
            nm = str(r.get("name") or "").lower()
            segs = set(_cid_segments(cid))
            if (nm and nm in tok_set) or (segs & tok_set):
                seeds.append(cid)
            if len(seeds) >= 10:
                break

    if not seeds:
        return []

    # (2) owning files of seeds
    file_scores: Dict[str, float] = {}
    file_reasons: Dict[str, Dict[str, Any]] = {}
    def _bump(file_path: str, amt: float, reason: str):
        file_scores[file_path] = file_scores.get(file_path, 0.0) + amt
        file_reasons.setdefault(file_path, {}).setdefault(reason, 0)
        file_reasons[file_path][reason] += 1

    for cid in seeds:
        r = rec_map.get(cid) or {}
        f = r.get("file")
        if f:
            _bump(f, 1.0, "seed_owner")

    # (3) graph walk (breadth-first) over chunks → accumulate owning files
    backend = CodeGraphBackend.wrap(G)
    if backend is not None:
        frontier = [(cid, 0) for cid in seeds if backend.has_node(cid)]
        visited = set([cid for cid, _ in frontier])
        while frontier:
            nid, d = frontier.pop(0)
            if d >= depth:
                continue
            for nb in backend.undirected_neighbors(nid):
                if nb in visited:
                    continue
                visited.add(nb)
                frontier.append((nb, d + 1))
                rr = rec_map.get(nb) or {}
                f = rr.get("file")
                if f:
                    _bump(f, 0.6 / (d + 1), "graph_neighbor")

    # (4) lexical tweak based on filename/token overlap
    for f in list(file_scores.keys()):
        lower = f.lower()
        if any(t in lower for t in tokens):
            _bump(f, 0.15, "filename_token")

    out = [{"file": f, "score": float(sc), "reasons": file_reasons.get(f, {})}
           for f, sc in file_scores.items()]
    out.sort(key=lambda d: (-d["score"], d["file"]))
    return out[:max_results]
