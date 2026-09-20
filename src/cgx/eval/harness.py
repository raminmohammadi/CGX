

"""Orchestrate the offline evals and enforce release-gate thresholds.

``run_gate`` loads the golden datasets + thresholds from an ``evals/``
directory, runs the codegen and recovery evals (always -- no heavy deps) and
the retrieval eval (only when faiss is importable), then checks every
configured threshold.
It returns ``(report, ok)`` where ``ok`` is False if any *ran* metric fell
below its floor; a metric whose section was skipped never fails the gate.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List, Tuple

from cgx.eval import codegen as _codegen
from cgx.eval import recovery as _recovery
from cgx.eval import retrieval as _retrieval
from cgx.logging_setup import get_logger

logger = get_logger(__name__)


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resolve_codegen_plans(golden: List[Dict[str, Any]], evals_dir: str) -> None:
    """Inline each case's plan text, reading ``plan_file`` when present."""
    for item in golden:
        if "plan_text" not in item and item.get("plan_file"):
            with open(os.path.join(evals_dir, item["plan_file"]), encoding="utf-8") as f:
                item["plan_text"] = f.read()


def _faiss_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("faiss") is not None


def _check_thresholds(
    section: str, aggregate: Dict[str, float], floors: Dict[str, float],
) -> List[str]:
    """Return a list of human-readable failure strings for one eval section."""
    failures: List[str] = []
    for metric, floor in (floors or {}).items():
        got = aggregate.get(metric)
        if got is None:
            failures.append(f"{section}.{metric}: metric not produced")
        elif got + 1e-9 < float(floor):
            failures.append(f"{section}.{metric}: {got:.3f} < {float(floor):.3f}")
    return failures


def run_gate(evals_dir: str) -> Tuple[Dict[str, Any], bool]:
    """Run every eval and evaluate the thresholds; returns (report, passed)."""
    with open(os.path.join(evals_dir, "thresholds.json"), encoding="utf-8") as f:
        thresholds = json.load(f)

    report: Dict[str, Any] = {"sections": {}, "failures": []}

    # --- Codegen (always runs; core deps only) -----------------------------
    cg_golden = _load_jsonl(os.path.join(evals_dir, "codegen_golden.jsonl"))
    _resolve_codegen_plans(cg_golden, evals_dir)
    sample_repo = os.path.join(evals_dir, "sample_repo")
    cg = _codegen.evaluate_codegen(cg_golden, sample_repo)
    report["sections"]["codegen"] = cg
    report["failures"] += _check_thresholds(
        "codegen", cg["aggregate"], thresholds.get("codegen", {}),
    )

    # --- Recovery (always runs; deterministic, provider-free) --------------
    rc_golden = _load_jsonl(os.path.join(evals_dir, "recovery_golden.jsonl"))
    rc = _recovery.evaluate_recovery(rc_golden)
    report["sections"]["recovery"] = rc
    report["failures"] += _check_thresholds(
        "recovery", rc["aggregate"], thresholds.get("recovery", {}),
    )
    # --- Lexical retrieval (always runs; NO torch/faiss needed) ------------
    # Isolates the BM25/lexical arm against content queries whose signal lives
    # in docstrings/comments/bodies rather than symbol names. The fused eval
    # below uses a bag-of-words embedder that masks lexical-coverage gaps, so
    # this is the honest measure of "does the lexical index cover content".
    lex_golden_path = os.path.join(evals_dir, "retrieval_lexical_golden.jsonl")
    lex_repo = os.path.join(evals_dir, "retrieval_repo")
    if os.path.exists(lex_golden_path) and os.path.isdir(lex_repo):
        lx_golden = _load_jsonl(lex_golden_path)
        records = _retrieval.build_sample_records(lex_repo)
        lx = _retrieval.evaluate_retrieval_lexical(lx_golden, records)
        report["sections"]["retrieval_lexical"] = lx
        report["failures"] += _check_thresholds(
            "retrieval_lexical", lx["aggregate"],
            thresholds.get("retrieval_lexical", {}),
        )
    else:
        report["sections"]["retrieval_lexical"] = {"skipped": "no lexical golden set"}

    if _faiss_available():
        rt_golden = _load_jsonl(os.path.join(evals_dir, "retrieval_golden.jsonl"))
        # Prefer the real embedding model when its deps are installed so the
        # fused eval measures true semantic quality; fall back to the
        # deterministic bag-of-words embedder (torch-free) otherwise. The
        # deterministic embedder is a WEAK proxy (perfect bag-of-words over the
        # full chunk text), so it validates fusion/ranking wiring but not
        # semantic recall -- hence the lexical gate above and the real path here.
        embedder, embedder_kind = _resolve_eval_embedder()
        with tempfile.TemporaryDirectory(prefix="cgx-eval-") as tmp:
            artifacts = _retrieval.build_sample_index(sample_repo, tmp, embedder)
            rt = _retrieval.evaluate_retrieval(rt_golden, artifacts, embedder)
        rt["embedder_kind"] = embedder_kind
        report["sections"]["retrieval"] = rt
        report["failures"] += _check_thresholds(
            "retrieval", rt["aggregate"], thresholds.get("retrieval", {}),
        )
    else:
        report["sections"]["retrieval"] = {"skipped": "faiss not installed"}
        logger.warning("eval.harness: faiss absent -- retrieval gate skipped")

    return report, not report["failures"]


def _resolve_eval_embedder() -> Tuple[Any, str]:
    """Return ``(embedder, kind)`` for the fused retrieval eval.

    Deterministic (bag-of-words, torch-free) BY DEFAULT so ``run_gate`` stays
    fast, reproducible, and CI-safe -- and, critically, never loads a torch
    model into the same process as faiss, which on some setups (notably macOS,
    where faiss-cpu and torch ship incompatible OpenMP runtimes) segfaults the
    interpreter. Set ``CGX_EVAL_REAL_EMBEDDER=1`` to opt into a real
    sentence-transformers measurement (true semantic recall) where the deps and
    environment support it.
    """
    if os.environ.get("CGX_EVAL_REAL_EMBEDDER"):
        import importlib.util
        if importlib.util.find_spec("sentence_transformers") is not None:
            try:
                from sentence_transformers import SentenceTransformer

                class _STEmbedder:
                    model_name = "jinaai/jina-embeddings-v2-base-code"

                    def __init__(self) -> None:
                        self._m = SentenceTransformer(
                            self.model_name, trust_remote_code=True,
                        )

                    def encode(self, texts):  # type: ignore[no-untyped-def]
                        return self._m.encode(list(texts), normalize_embeddings=False)

                return _STEmbedder(), "sentence-transformers/jina-v2-code"
            except Exception as e:  # pragma: no cover - env dependent
                logger.warning("eval.harness: real embedder unavailable (%s); "
                               "using deterministic fallback", e)
    return _retrieval.DeterministicEmbedder(dim=32), "deterministic-bow"
