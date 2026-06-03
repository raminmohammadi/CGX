

# src/cgx/retrieval/rrf.py
from __future__ import annotations

"""
Reciprocal Rank Fusion (RRF) utilities.

RRF combines multiple ranked lists into a single, robust ordering with:
    score(id) = sum_{lists} 1 / (k + rank_in_list(id))

This module offers tiny, deterministic helpers that accept:
- sequences of ids (order = rank)
- sequences of {id, rank} dicts
- sequences of {id, score} dicts (converted to ranks per list)

We do NOT assume any embedding model or index; this purely fuses ranks.
"""

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple, Union, Any, Optional
import logging

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

Id = str
Rank = int
Score = float
ListLike = Sequence[Union[Id, Mapping[str, Any]]]


def _as_rank_map(lst: ListLike) -> Dict[Id, Rank]:
    """
    Convert a list into {id: rank} (1-based). Supports:
      - ["a","b","c"]
      - [{"id":"a","rank":2}, {"id":"b","rank":1}]
      - [{"chunk_id":"a","score":0.9}, ...]  -> ranks by descending score
    """
    if not lst:
        return {}

    # case 1: pure id sequence
    if isinstance(lst[0], str):
        return {item: i + 1 for i, item in enumerate(lst)}  # type: ignore[arg-type]

    # case 2/3: dicts
    if isinstance(lst[0], Mapping):
        first = lst[0]  # type: ignore[index]
        if "rank" in first:
            # explicit rank
            items = [(str(d.get("id") or d.get("chunk_id")), int(d["rank"])) for d in lst]  # type: ignore[index]
            # ensure smallest rank is best (1-based)
            return {i: r for i, r in items if i}
        elif "score" in first:
            # convert scores to ranks (higher score -> better rank)
            scored: List[Tuple[Id, Score]] = []
            for d in lst:  # type: ignore[index]
                cid = str(d.get("id") or d.get("chunk_id") or "")
                if not cid:
                    continue
                s = float(d.get("score", 0.0))
                scored.append((cid, s))
            scored.sort(key=lambda t: (-t[1], t[0]))
            return {cid: i + 1 for i, (cid, _) in enumerate(scored)}
        elif "id" in first or "chunk_id" in first:
            # implicit by order
            ids = [str(d.get("id") or d.get("chunk_id")) for d in lst]  # type: ignore[index]
            return {cid: i + 1 for i, cid in enumerate([x for x in ids if x])}

    # fallback
    logger.debug("rrf._as_rank_map: unrecognized list shape; treating as empty.")
    return {}


def rrf_fuse(
    lists: Sequence[ListLike],
    *,
    k: float = 60.0,
    top_k: Optional[int] = None
) -> List[Tuple[Id, float]]:
    """
    Fuse multiple ranked lists with RRF.

    Parameters
    ----------
    lists : sequence of list-likes
        Each list-like is converted to {id: rank}. Missing ids in a list are ignored.
    k : float, default 60.0
        Stabilizer; larger values flatten the contribution of low ranks.
    top_k : int | None
        Optional cut after fusion.

    Returns
    -------
    list[(id, score)]
        Descending by fused score; ties broken by id for determinism.
    """
    rank_maps: List[Dict[Id, Rank]] = [_as_rank_map(lst) for lst in lists if lst]
    if not rank_maps:
        return []

    scores: Dict[Id, float] = {}
    for rm in rank_maps:
        for cid, r in rm.items():
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + float(r))

    fused = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return fused[:top_k] if isinstance(top_k, int) and top_k > 0 else fused
