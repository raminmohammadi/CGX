from __future__ import annotations

"""
Two-view hybrid retrieval orchestrator (UNIFIED).

Consumes:
  - TwoViewIndex (S6) for ANN over 'intent' and 'impl'
  - LexicalIndex (BM25-lite) from S7 lexical.py, OR fallback regex over chunks
  - S4 records/chunks for metadata + grouping (file/class)
  - Optional knowledge graph G for 1-hop/multi-hop expansion

Outputs:
  {
    "chunks":  [ {chunk_id, score, rank, provenance}, ... ],
    "files":   [ {file, score, members:[...]}, ... ],
    "classes": [ {class_id, score, members:[...]}, ... ]
  }

Provenance per chunk includes:
  - intent_rank, intent_score
  - impl_rank, impl_score
  - lexical_count
  - graph_depth
  - symbol_match
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Iterable, Callable
import logging
import re
from collections import deque

from .index import TwoViewIndex
from .lexical import LexicalIndex
from .rrf import rrf_fuse

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# ---------------------------
# helpers
# ---------------------------

def _records_map(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(r.get("id")): r for r in records if isinstance(r.get("id"), str) and r.get("id")}


def _expand_multi_hop(
    seeds: List[str],
    *,
    G,
    max_depth: int = 1,
    relation_types: Optional[Iterable[str]] = None,
) -> Dict[str, int]:
    """Multi-hop graph expansion with optional relation-type filtering."""
    depths: Dict[str, int] = {}
    if G is None:
        return depths
    qd = deque([(sid, 0) for sid in seeds if sid in G])
    visited = set(seeds)

    while qd:
        nid, d = qd.popleft()
        if d >= max_depth:
            continue
        for nbr in list(G.successors(nid)) + list(G.predecessors(nid)):
            if nbr in visited:
                continue
            edge_ok = True
            if relation_types:
                attrs = G[nid].get(nbr) or G.get(nid, {}).get(nbr, {})
                etypes = [ed.get("type") for ed in (attrs.values() if isinstance(attrs, dict) else []) if isinstance(ed, dict)]
                if etypes and not any(et in relation_types for et in etypes):
                    edge_ok = False
            if not edge_ok:
                continue
            visited.add(nbr)
            qd.append((nbr, d + 1))
            depths[str(nbr)] = d + 1
    return depths


def _aggregate_group(
    scores_by_chunk: List[Tuple[str, float]],
    *,
    key_fn,
    alpha: float = 0.5,
    decay: float = 0.75,
    max_per_group: int = 6
) -> List[Tuple[str, float, List[Tuple[str, float]]]]:
    """Aggregate chunk scores to groups using: best + alpha * sum(decay^(i-1)*next_i)."""
    by_group: Dict[str, List[Tuple[str, float]]] = {}
    for cid, s in scores_by_chunk:
        gid = key_fn(cid)
        if not gid:
            continue
        by_group.setdefault(gid, []).append((cid, s))

    out: List[Tuple[str, float, List[Tuple[str, float]]]] = []
    for gid, items in by_group.items():
        items = sorted(items, key=lambda kv: (-kv[1], kv[0]))
        best = items[0][1]
        extra = sum((decay ** (i - 1)) * sc for i, (_, sc) in enumerate(items[1:max_per_group], start=1))
        total = best + alpha * extra
        out.append((gid, float(total), items[:max_per_group]))

    out.sort(key=lambda kv: (-kv[1], kv[0]))
    return out


def _extract_symbol_tokens(q: str) -> List[str]:
    q = q or ""
    quoted = re.findall(r"[`\"]([A-Za-z_][A-Za-z0-9_]*)[`\"]", q)
    bare = re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", q)
    return list({t.lower() for t in quoted + bare})


# ---------------------------
# config
# ---------------------------

@dataclass
class HybridConfig:
    k_intent: int = 50
    k_impl: int = 50
    k_lex: int = 50

    expand_top_n: int = 10
    expand_per_seed: int = 12
    graph_depth: int = 1
    relation_types: Optional[List[str]] = None

    rrf_k: float = 60.0

    top_k_chunks: int = 50
    top_k_files: int = 20
    top_k_classes: int = 20

    agg_alpha: float = 0.5
    agg_decay: float = 0.75
    agg_max_per_group: int = 6


# ---------------------------
# retriever
# ---------------------------

class HybridRetriever:
    """
    Orchestrates two-view ANN + lexical (index or regex) + multi-hop graph expansion
    with RRF fusion, symbol boosting, provenance, and file/class aggregation.
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
        self.records = records
        self.lex = lexical_index
        self.chunks = chunks or []
        self.G = G
        self._rec_by_id = _records_map(records)

    def search(
        self,
        query: str,
        *,
        embedder: Any,
        cfg: Optional[HybridConfig] = None,
        lexical_search_fn: Optional[Callable[..., List[Dict[str, Any]]]] = None,
        lex_fields: Iterable[str] = ("code", "name", "id", "file", "meta.docstring"),
        lex_regex: bool = True,
        lex_case_sensitive: bool = False,
        lex_whole_word: bool = False,
    ) -> Dict[str, Any]:
        cfg = cfg or HybridConfig()
        lists_for_rrf: List[List[Dict[str, Any]]] = []
        provenance: Dict[str, Dict[str, Any]] = {}

        # --- semantic per view ---
        for view, k in (("intent", cfg.k_intent), ("impl", cfg.k_impl)):
            if view in self.tv.available_views():
                hits = self.tv.search_view(view, query, embedder=embedder, top_k=k)
                lists_for_rrf.append([{"chunk_id": h["chunk_id"], "rank": h["rank"]} for h in hits])
                for h in hits:
                    provenance.setdefault(h["chunk_id"], {}).update({
                        f"{view}_rank": h["rank"],
                        f"{view}_score": h["score"],
                    })

        # --- lexical ---
        if cfg.k_lex > 0:
            if self.lex is not None:
                lex_hits = self.lex.search(query, top_k=cfg.k_lex)
                lists_for_rrf.append([{"chunk_id": h["chunk_id"], "rank": h["rank"]} for h in lex_hits])
                for h in lex_hits:
                    provenance.setdefault(h["chunk_id"], {}).update({"lexical_count": 1})
            elif self.chunks:
                tokens = _extract_symbol_tokens(query)
                if not tokens:
                    tokens = [t for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", query)]
                if tokens:
                    def _mk_rx(tok: str, whole: bool) -> str:
                        return (r"\b" + re.escape(tok) + r"\b") if whole else re.escape(tok)
                    pat = r"(" + "|".join(_mk_rx(t, bool(lex_whole_word)) for t in tokens) + r")"
                    rx = re.compile(pat, 0 if bool(lex_case_sensitive) else re.IGNORECASE)

                    def _field_get(d: Dict[str, Any], dotted: str) -> str:
                        cur: Any = d
                        for p in dotted.split("."):
                            cur = cur.get(p, "") if isinstance(cur, dict) else ""
                        return str(cur or "")

                    lex_counts: Dict[str, int] = {}
                    for ch in self.chunks:
                        cid = str(ch.get("id") or "")
                        if not cid:
                            continue
                        total = 0
                        for f in tuple(lex_fields):
                            s = _field_get(ch, f)
                            if s:
                                total += len(list(rx.finditer(s)))
                        if total:
                            lex_counts[cid] = total
                            provenance.setdefault(cid, {}).update({"lexical_count": total})
                    if lex_counts:
                        sorted_ids = sorted(lex_counts.items(), key=lambda kv: (-kv[1], kv[0]))
                        lists_for_rrf.append([{"chunk_id": cid, "rank": i+1} for i, (cid, _) in enumerate(sorted_ids)])

        if not lists_for_rrf:
            return {"chunks": [], "files": [], "classes": []}

        # --- fusion ---
        fused = rrf_fuse(lists_for_rrf, k=cfg.rrf_k, top_k=cfg.top_k_chunks + cfg.expand_top_n)

        # --- graph expansion ---
        seeds = [cid for cid, _ in fused[:cfg.expand_top_n]]
        graph_depths = _expand_multi_hop(
            seeds, G=self.G,
            max_depth=cfg.graph_depth,
            relation_types=cfg.relation_types,
        )
        for cid, depth in graph_depths.items():
            provenance.setdefault(cid, {}).update({"graph_depth": depth})
            fused = [(c, sc + 0.2 / (depth + 1) if c == cid else sc) for c, sc in fused]

        # --- symbol boosting ---
        sym_tokens = _extract_symbol_tokens(query)
        for cid in list(provenance.keys()):
            rec = self._rec_by_id.get(cid) or {}
            nm = str(rec.get("name") or "").lower()
            if any(t in cid.lower() or nm == t for t in sym_tokens):
                provenance[cid]["symbol_match"] = True
                fused = [(c, sc + 0.5 if c == cid else sc) for c, sc in fused]

        # --- final chunks ---
        chunk_scores: List[Tuple[str, float]] = sorted(fused, key=lambda kv: -kv[1])[:cfg.top_k_chunks]
        chunk_results = []
        for i, (cid, sc) in enumerate(chunk_scores, start=1):
            prov = provenance.get(cid, {})
            chunk_results.append({"chunk_id": cid, "score": float(sc), "rank": i, "provenance": prov})

        # --- aggregate to files/classes ---
        def file_key(cid: str) -> Optional[str]:
            rec = self._rec_by_id.get(cid)
            return rec.get("file") if rec else None

        def class_key(cid: str) -> Optional[str]:
            rec = self._rec_by_id.get(cid)
            return rec.get("parent_class_id") if rec else None

        files = _aggregate_group(chunk_scores, key_fn=file_key,
                                 alpha=cfg.agg_alpha, decay=cfg.agg_decay,
                                 max_per_group=cfg.agg_max_per_group)
        classes = _aggregate_group(chunk_scores, key_fn=class_key,
                                   alpha=cfg.agg_alpha, decay=cfg.agg_decay,
                                   max_per_group=cfg.agg_max_per_group)

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
            "hits": chunk_results,          # was "chunks"
            "top_files": file_results,      # was "files"
            "top_classes": class_results,   # was "classes"
            "anchors": []                   # keep placeholder for insertion points
        }
