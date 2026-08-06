

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
    if _faiss_available():
        rt_golden = _load_jsonl(os.path.join(evals_dir, "retrieval_golden.jsonl"))
        embedder = _retrieval.DeterministicEmbedder(dim=32)
        with tempfile.TemporaryDirectory(prefix="cgx-eval-") as tmp:
            artifacts = _retrieval.build_sample_index(sample_repo, tmp, embedder)
            rt = _retrieval.evaluate_retrieval(rt_golden, artifacts, embedder)
        report["sections"]["retrieval"] = rt
        report["failures"] += _check_thresholds(
            "retrieval", rt["aggregate"], thresholds.get("retrieval", {}),
        )
    else:
        report["sections"]["retrieval"] = {"skipped": "faiss not installed"}
        logger.warning("eval.harness: faiss absent -- retrieval gate skipped")

    return report, not report["failures"]
