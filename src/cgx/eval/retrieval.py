

"""Offline retrieval evaluation against an in-repo golden dataset.

The harness indexes a small pinned sample repo with a deterministic,
dependency-free embedder, replays each golden query through the real
``run_query_auto`` retrieval path, and scores the ranked hits with the pure
metrics in :mod:`cgx.eval.metrics`. Relevance is matched by machine-independent
chunk-id *fragments* (e.g. ``calc.py::method::Calculator.add``) so a golden
file authored on one machine scores identically on another.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Sequence

import numpy as np

from cgx.eval import metrics as M
from cgx.logging_setup import get_logger

logger = get_logger(__name__)


class DeterministicEmbedder:
    """Hash-based embedder: reproducible, offline, and torch-free."""

    def __init__(self, dim: int = 32) -> None:
        self.dim = int(dim)

    def encode(self, texts: Sequence[str]) -> "np.ndarray":
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in (t or " ").lower().split():
                h = hashlib.sha256(tok.encode("utf-8")).digest()
                for j in range(self.dim):
                    out[i, j] += (h[j % len(h)] / 255.0) - 0.5
            out[i] /= (np.linalg.norm(out[i]) + 1e-12)
        return out


def _relevances(hit_ids: Sequence[str], relevant: Sequence[str]) -> List[float]:
    """Binary relevance per ranked hit (1.0 if it contains any golden fragment)."""
    return [
        1.0 if any(frag in str(cid) for frag in relevant) else 0.0
        for cid in hit_ids
    ]


def _distinct_found(hit_ids: Sequence[str], relevant: Sequence[str], k: int) -> int:
    """How many distinct golden fragments appear anywhere in the top ``k`` hits."""
    top = [str(c) for c in list(hit_ids)[:k]]
    return sum(1 for frag in relevant if any(frag in c for c in top))


def evaluate_query(
    hit_ids: Sequence[str],
    relevant: Sequence[str],
    k_values: Sequence[int] = (1, 5, 10),
) -> Dict[str, float]:
    """Score a single ranked hit list against its golden relevance set."""
    rels = _relevances(hit_ids, relevant)
    n_rel = len(relevant)
    scores: Dict[str, float] = {"mrr": M.reciprocal_rank(rels)}
    for k in k_values:
        scores[f"recall@{k}"] = (
            _distinct_found(hit_ids, relevant, k) / n_rel if n_rel else 0.0
        )
        scores[f"precision@{k}"] = M.precision_at_k(rels, k)
        scores[f"ndcg@{k}"] = M.ndcg_at_k(rels, k)
    return scores


def build_sample_index(sample_repo: str, out_dir: str, embedder: Any) -> Dict[str, str]:
    """Index ``sample_repo`` for evaluation; returns the artifact path map.

    Requires faiss (imported transitively by the pipeline); callers gate on it.
    """
    from cgx.pipeline.auto import run_index_auto

    res = run_index_auto(
        sample_repo, out_dir, metric="cosine", index_type="flat", embedder=embedder,
    )
    return res["out"]


def evaluate_retrieval(
    golden: Sequence[Dict[str, Any]],
    artifacts: Dict[str, str],
    embedder: Any,
    *,
    top_k_per_view: int = 10,
    k_values: Sequence[int] = (1, 5, 10),
) -> Dict[str, Any]:
    """Replay every golden query and aggregate ranking metrics (mean over queries)."""
    from cgx.pipeline.auto import run_query_auto

    per_query: List[Dict[str, Any]] = []
    for item in golden:
        q = str(item.get("query", ""))
        relevant = list(item.get("relevant", []))
        out = run_query_auto(
            index_dir=artifacts["indices"],
            records_path=artifacts["records"],
            query=q,
            chunks_path=artifacts.get("chunks"),
            graph_path=artifacts.get("graph"),
            embedder=embedder,
            top_k_per_view=top_k_per_view,
            neighbor_depth=0,
        )
        hit_ids = [h.get("chunk_id") for h in (out.get("hits") or [])]
        scores = evaluate_query(hit_ids, relevant, k_values)
        per_query.append({"query": q, "scores": scores})
        logger.info("eval.retrieval: query=%r scores=%s", q, scores)

    keys = per_query[0]["scores"].keys() if per_query else []
    aggregate = {
        key: M.mean([row["scores"][key] for row in per_query]) for key in keys
    }
    return {
        "n_queries": len(per_query),
        "per_query": per_query,
        "aggregate": aggregate,
    }
