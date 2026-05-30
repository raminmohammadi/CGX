"""Backwards-compatible re-exports for the hybrid retriever.

The canonical implementation lives in :mod:`cgx.retrieval.orchestrator`. This
module used to host a parallel, slightly diverging copy of ``HybridRetriever``
that was reachable only through ``cgx.retrieval.cli_adapter``. Maintaining two
implementations meant fixes (e.g. graph-only neighbors being dropped during
rerank) had to be applied twice and one path kept lagging.

To keep ``python -m cgx.retrieval.cli_adapter --hybrid ...`` working without
behavioral drift, this module now re-exports the orchestrator's classes.
"""

from __future__ import annotations

from cgx.retrieval.orchestrator import HybridConfig, HybridRetriever

__all__ = ["HybridConfig", "HybridRetriever"]
