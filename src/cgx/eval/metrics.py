

"""Pure ranking-quality metrics for offline retrieval evaluation.

Every function operates on a ranked list of relevance grades (``0`` = not
relevant, ``> 0`` = relevant; larger = more relevant) already ordered by the
system under test, plus -- where a recall denominator is needed -- the total
number of relevant items in the corpus. The module is intentionally
dependency-free (stdlib only) so the eval gate runs in every CI job, including
the "core" matrix that has no torch / faiss installed.
"""

from __future__ import annotations

import math
from typing import Sequence


def recall_at_k(relevances: Sequence[float], n_relevant_total: int, k: int) -> float:
    """Fraction of all relevant items that appear in the top ``k`` results."""
    if n_relevant_total <= 0 or k <= 0:
        return 0.0
    found = sum(1 for r in list(relevances)[:k] if r > 0)
    return min(found, n_relevant_total) / n_relevant_total


def precision_at_k(relevances: Sequence[float], k: int) -> float:
    """Fraction of the top ``k`` results that are relevant."""
    if k <= 0:
        return 0.0
    found = sum(1 for r in list(relevances)[:k] if r > 0)
    return found / k


def reciprocal_rank(relevances: Sequence[float]) -> float:
    """``1 / rank`` of the first relevant hit (0.0 if none)."""
    for i, r in enumerate(relevances, start=1):
        if r > 0:
            return 1.0 / i
    return 0.0


def dcg_at_k(relevances: Sequence[float], k: int) -> float:
    """Discounted cumulative gain over the top ``k`` (graded relevance)."""
    dcg = 0.0
    for i, r in enumerate(list(relevances)[:k], start=1):
        if r > 0:
            dcg += (2.0 ** r - 1.0) / math.log2(i + 1)
    return dcg


def ndcg_at_k(relevances: Sequence[float], k: int) -> float:
    """DCG@k normalised by the ideal DCG@k (0.0 when no relevant items)."""
    idcg = dcg_at_k(sorted(relevances, reverse=True), k)
    if idcg <= 0.0:
        return 0.0
    return dcg_at_k(relevances, k) / idcg


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean; ``0.0`` for an empty sequence (never divides by zero)."""
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0
