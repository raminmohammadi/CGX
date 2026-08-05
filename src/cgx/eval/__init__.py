

"""CGX offline evaluation harness (retrieval + codegen) and CI quality gate.

Only the dependency-free metric primitives are imported eagerly here; the
retrieval and codegen harnesses pull in heavier modules (faiss, the codegen
pipeline) lazily so ``import cgx.eval`` stays cheap and side-effect free.
"""

from __future__ import annotations

from cgx.eval.metrics import (
    dcg_at_k,
    mean,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

__all__ = [
    "recall_at_k",
    "precision_at_k",
    "reciprocal_rank",
    "dcg_at_k",
    "ndcg_at_k",
    "mean",
]
