

"""Offline codegen-quality evaluation against an in-repo golden dataset.

Each golden case pairs a pinned plan (free-form markdown wrapping fenced
``diff path=...`` blocks) with an expectation about whether the change is
acceptable. The harness feeds every plan through the real
``cgx.codegen.pipeline.validate_and_test`` path (diff parse -> in-memory apply
-> language-aware syntax check) and aggregates the parse / syntax / overall
pass rates that a release gate can enforce. Tests are not executed by default
so the gate stays hermetic and fast.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from cgx.eval import metrics as M
from cgx.logging_setup import get_logger

logger = get_logger(__name__)


def score_report(summary: Dict[str, Any], expect: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce a :class:`CodegenReport` summary to boolean quality signals."""
    parsed = int(summary.get("n_targets", 0)) > 0
    patches_ok = (
        int(summary.get("n_patches_failed", 0)) == 0
        and int(summary.get("n_patches_ok", 0)) > 0
    )
    syntax_ok = int(summary.get("n_syntax_failed", 0)) == 0
    overall_ok = bool(summary.get("overall_ok"))
    expected = bool(expect.get("overall_ok", True))
    return {
        "parsed": parsed,
        "patches_ok": patches_ok,
        "syntax_ok": syntax_ok,
        "overall_ok": overall_ok,
        "expected_overall_ok": expected,
        "matched_expectation": overall_ok == expected,
    }


def aggregate_codegen(per_item: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Mean pass rates across all scored codegen cases."""
    return {
        "parse_rate": M.mean([1.0 if r["parsed"] else 0.0 for r in per_item]),
        "syntax_pass_rate": M.mean([1.0 if r["syntax_ok"] else 0.0 for r in per_item]),
        "overall_ok_rate": M.mean([1.0 if r["overall_ok"] else 0.0 for r in per_item]),
        "expectation_match_rate": M.mean(
            [1.0 if r["matched_expectation"] else 0.0 for r in per_item]
        ),
    }


def evaluate_codegen(
    golden: Sequence[Dict[str, Any]],
    sample_repo: str,
    *,
    run_tests: bool = False,
) -> Dict[str, Any]:
    """Validate every golden plan against ``sample_repo`` and aggregate metrics."""
    from cgx.codegen.pipeline import validate_and_test

    per_item: List[Dict[str, Any]] = []
    for item in golden:
        plan_text = str(item.get("plan_text", ""))
        report = validate_and_test(sample_repo, plan_text, run_tests=run_tests)
        row = {"id": item.get("id")}
        row.update(score_report(report.summary, item.get("expect", {})))
        per_item.append(row)
        logger.info(
            "eval.codegen: id=%s overall_ok=%s matched=%s",
            row["id"], row["overall_ok"], row["matched_expectation"],
        )
    return {
        "n_items": len(per_item),
        "per_item": per_item,
        "aggregate": aggregate_codegen(per_item),
    }
