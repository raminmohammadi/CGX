# src/cgx/retrieval/hybrid.py
from __future__ import annotations

"""
Two-view hybrid retrieval orchestrator (ADD-ONLY).

Consumes:
  - TwoViewIndex (S6) for ANN over 'intent' and 'impl'
  - LexicalIndex (BM25-lite) from S7 lexical.py
  - S4 records for metadata + grouping (file/class)
  - Optional knowledge graph G for 1-hop expansion (safe to omit)

Outputs:
  - chunk_results: ranked list of chunks
  - file_results:  aggregated ranking by file
  - class_results: aggregated ranking by parent class

No changes to existing modules; pure consumer layer.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import logging

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

def _neighbors_ids_from_record(rec: Dict[str, Any], max_n: int = 32) -> List[str]:
    out: List[str] = []
    for tup in rec.get("neighbors_summary") or []:
        try:
            edge_type, nid = tup
        except Exception:
            continue
        if isinstance(nid, str) and nid:
            out.append(nid)
        if len(out) >= max_n:
            break
    return out

def _neighbors_ids_from_graph(G, cid: str, max_n: int = 32) -> List[str]:
    out: List[str] = []
    if G is None or cid not in G:
        return out
    try:
        # predecessors + successors, stable sort by (edge_type, node_id)
        pairs = []
        for u in G.predecessors(cid):
            attrs = G[u][cid]
            if isinstance(attrs, dict) and any(isinstance(v, dict) for v in attrs.values()):
                for ed in attrs.values():
                    et = ed.get("type", "")
                    pairs.append((et, u))
            else:
                et = attrs.get("type", "")
                pairs.append((et, u))
        for v in G.successors(cid):
            attrs = G[cid][v]
            if isinstance(attrs, dict) and any(isinstance(v2, dict) for v2 in attrs.values()):
                for ed in attrs.values():
                    et = ed.get("type", "")
                    pairs.append((et, v))
            else:
                et = attrs.get("type", "")
                pairs.append((et, v))
        pairs = sorted({(et, nid) for et, nid in pairs})
        for _, nid in pairs[:max_n]:
            out.append(nid)
    except Exception:
        return out
    return out

def _expand_1hop(seed_ids: List[str], *, records_by_id: Dict[str, Dict[str, Any]], G=None, per_seed: int = 16) -> List[str]:
    """
    Deterministic 1-hop expansion: for each seed, append up to `per_seed` neighbors
    (graph if provided, else neighbor summary from the record).
    Output is a flattened list (duplicates removed while preserving order).
    """
    seen = set(seed_ids)
    out: List[str] = []
    for sid in seed_ids:
        rec = records_by_id.get(sid) or {}
        neigh = _neighbors_ids_from_graph(G, sid, per_seed) if G is not None else _neighbors_ids_from_record(rec, per_seed)
        for nid in neigh:
            if nid not in seen:
                out.append(nid)
                seen.add(nid)
    return out

def _aggregate_group(scores_by_chunk: List[Tuple[str, float]], *, key_fn, alpha: float = 0.5, decay: float = 0.75, max_per_group: int = 6) -> List[Tuple[str, float, List[Tuple[str, float]]]]:
    """
    Aggregate chunk scores to groups using:
      group_score = best + alpha * sum(decay^(i-1) * next_i)
    Deterministic: ties broken by group id.
    """
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
        extra = 0.0
        for i, (_, sc) in enumerate(items[1:max_per_group], start=1):
            extra += (decay ** (i - 1)) * sc
        total = best + alpha * extra
        out.append((gid, float(total), items[:max_per_group]))

    out.sort(key=lambda kv: (-kv[1], kv[0]))
    return out


# ---------------------------
# public api
# ---------------------------

@dataclass
class HybridConfig:
    # per-view cutoffs
    k_intent: int = 50
    k_impl: int = 50
    k_lex: int = 50

    # graph expansion
    expand_top_n: int = 10      # expand the top-N fused seeds
    expand_per_seed: int = 12   # up to M neighbors per seed

    # RRF stabilizer
    rrf_k: float = 60.0

    # final cutoffs
    top_k_chunks: int = 50
    top_k_files: int = 20
    top_k_classes: int = 20

    # aggregation recipe
    agg_alpha: float = 0.5
    agg_decay: float = 0.75
    agg_max_per_group: int = 6


class HybridRetriever:
    """
    Orchestrates two-view ANN + lexical + (optional) 1-hop graph expansion with RRF fusion.
    """

    def __init__(
        self,
        *,
        tv_index: TwoViewIndex,
        records: List[Dict[str, Any]],
        lexical_index: Optional[LexicalIndex] = None,
        G=None
    ) -> None:
        self.tv = tv_index
        self.records = records
        self.lex = lexical_index
        self.G = G
        self._rec_by_id = _records_map(records)

    # ---- main search ----

    def search(
        self,
        query: str,
        *,
        embedder: Any,
        cfg: Optional[HybridConfig] = None
    ) -> Dict[str, Any]:
        cfg = cfg or HybridConfig()

        lists_for_rrf: List[List[Dict[str, Any]]] = []

        # 1) intent ANN
        if "intent" in self.tv.available_views():
            intent_hits = self.tv.search_view(
                "intent",
                query,
                embedder=embedder,
                top_k=cfg.k_intent,
            )
            # convert to rank list
            lists_for_rrf.append([{"chunk_id": h["chunk_id"], "rank": h["rank"]} for h in intent_hits])

        # 2) impl ANN
        if "impl" in self.tv.available_views():
            impl_hits = self.tv.search_view(
                "impl",
                query,
                embedder=embedder,
                top_k=cfg.k_impl,
            )
            lists_for_rrf.append([{"chunk_id": h["chunk_id"], "rank": h["rank"]} for h in impl_hits])

        # 3) lexical
        if self.lex is not None:
            lex_hits = self.lex.search(query, top_k=cfg.k_lex)
            # scores -> ranks internally in RRF helper if we pass scores; but we keep explicit ranks for stability
            lists_for_rrf.append([{"chunk_id": h["chunk_id"], "rank": h["rank"]} for h in lex_hits])

        # If nothing to fuse, return empty result
        if not lists_for_rrf:
            return {"chunks": [], "files": [], "classes": []}

        # 4) initial fusion
        fused = rrf_fuse(lists_for_rrf, k=cfg.rrf_k, top_k=cfg.top_k_chunks + cfg.expand_top_n)

        # 5) 1-hop expansion from top-N seeds (optional)
        if cfg.expand_top_n > 0:
            seeds = [cid for cid, _ in fused[:cfg.expand_top_n]]
            expanded = _expand_1hop(seeds, records_by_id=self._rec_by_id, G=self.G, per_seed=cfg.expand_per_seed)
            # make an expansion list where neighbors follow seeds deterministically
            exp_list = [{"chunk_id": cid} for cid in expanded]
            lists_for_rrf.append(exp_list)
            fused = rrf_fuse(lists_for_rrf, k=cfg.rrf_k, top_k=cfg.top_k_chunks)

        # chunk results (with simple normalized score for display)
        chunk_scores: List[Tuple[str, float]] = fused[:cfg.top_k_chunks]
        chunk_results = [{"chunk_id": cid, "score": float(sc), "rank": i + 1} for i, (cid, sc) in enumerate(chunk_scores)]

        # 6) aggregate to files/classes
        def file_key(cid: str) -> Optional[str]:
            rec = self._rec_by_id.get(cid)
            return rec.get("file") if rec else None

        def class_key(cid: str) -> Optional[str]:
            rec = self._rec_by_id.get(cid)
            return rec.get("parent_class_id") if rec else None

        files = _aggregate_group(chunk_scores, key_fn=file_key, alpha=cfg.agg_alpha, decay=cfg.agg_decay, max_per_group=cfg.agg_max_per_group)
        classes = _aggregate_group(chunk_scores, key_fn=class_key, alpha=cfg.agg_alpha, decay=cfg.agg_decay, max_per_group=cfg.agg_max_per_group)

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
            "chunks": chunk_results,
            "files": file_results,
            "classes": class_results,
        }
