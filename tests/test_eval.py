"""Tests for the offline eval harness (metrics, codegen, retrieval, gate)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cgx.eval import metrics as M
from cgx.eval import codegen as CG
from cgx.eval import recovery as RC
from cgx.eval import retrieval as RT
from cgx.eval.harness import run_gate

EVALS_DIR = str(Path(__file__).resolve().parents[1] / "evals")


def _load_recovery_golden():
    import json
    path = Path(EVALS_DIR) / "recovery_golden.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --------------------------------------------------------------------------
# Pure ranking metrics (run everywhere -- no heavy deps)
# --------------------------------------------------------------------------
def test_recall_and_precision_at_k():
    rels = [0, 1, 0, 1]  # 2 relevant in a corpus of 3 relevant total
    assert M.recall_at_k(rels, n_relevant_total=3, k=4) == pytest.approx(2 / 3)
    assert M.recall_at_k(rels, n_relevant_total=3, k=1) == pytest.approx(0.0)
    assert M.precision_at_k(rels, k=2) == pytest.approx(0.5)
    # Guards: empty / non-positive inputs never divide by zero.
    assert M.recall_at_k(rels, 0, 4) == 0.0
    assert M.precision_at_k(rels, 0) == 0.0


def test_reciprocal_rank_and_ndcg():
    assert M.reciprocal_rank([0, 0, 1]) == pytest.approx(1 / 3)
    assert M.reciprocal_rank([0, 0, 0]) == 0.0
    # A perfectly ranked list has nDCG 1.0; a reversed one is strictly worse.
    assert M.ndcg_at_k([1, 1, 0], k=3) == pytest.approx(1.0)
    assert M.ndcg_at_k([0, 1, 1], k=3) < 1.0
    assert M.ndcg_at_k([0, 0, 0], k=3) == 0.0


def test_mean_handles_empty():
    assert M.mean([]) == 0.0
    assert M.mean([1.0, 3.0]) == pytest.approx(2.0)


# --------------------------------------------------------------------------
# Retrieval scoring (pure part -- fragment matching against ranked ids)
# --------------------------------------------------------------------------
def test_evaluate_query_matches_by_fragment():
    hit_ids = [
        "/abs/pkg/db.py::function::parse_input_args",
        "/abs/pkg/calc.py::method::Calculator.add",
    ]
    scores = RT.evaluate_query(
        hit_ids, ["calc.py::method::Calculator.add"], k_values=(1, 2),
    )
    assert scores["recall@2"] == pytest.approx(1.0)
    assert scores["recall@1"] == pytest.approx(0.0)  # relevant hit is at rank 2
    assert scores["mrr"] == pytest.approx(0.5)


# --------------------------------------------------------------------------
# Codegen scoring (pure) + end-to-end validate over the sample repo
# --------------------------------------------------------------------------
def test_score_report_flags_expectation_mismatch():
    ok_summary = {"n_targets": 1, "n_patches_ok": 1, "n_patches_failed": 0,
                  "n_syntax_failed": 0, "overall_ok": True}
    row = CG.score_report(ok_summary, {"overall_ok": True})
    assert row["parsed"] and row["syntax_ok"] and row["matched_expectation"]
    # Expected a failure but the plan validated -> mismatch.
    row2 = CG.score_report(ok_summary, {"overall_ok": False})
    assert row2["matched_expectation"] is False


def test_evaluate_codegen_over_golden_cases():
    golden = [
        {"id": "good", "plan_text": (
            "```diff path=pkg/extra.py\n--- /dev/null\n+++ b/pkg/extra.py\n"
            "@@ -0,0 +1,2 @@\n+def hello():\n+    return \"hi\"\n```"
        ), "expect": {"overall_ok": True}},
        {"id": "empty", "plan_text": "Just prose, no diffs.",
         "expect": {"overall_ok": False}},
    ]
    sample_repo = str(Path(EVALS_DIR) / "sample_repo")
    out = CG.evaluate_codegen(golden, sample_repo)
    assert out["n_items"] == 2
    assert out["aggregate"]["expectation_match_rate"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Recovery scoring (pure, provider-free -- mirrors the real router dispatch)
# --------------------------------------------------------------------------
def test_resolve_recovery_scopes_mechanical_and_localized_failures():
    # A unittest/pytest mix is a mechanical patch, never a regenerate.
    patch = RC.resolve_recovery({
        "gate": "verify",
        "content": {"outcome": "assertions_failed", "failures": [{
            "type": "AttributeError",
            "message": "'T' object has no attribute 'assertEqual'",
            "traceback": "tests/test_calc.py:11: in test_add"}]}})
    assert patch["action"] == "patch_files"
    # Assertion drift localizes to the impl file (test frames are filtered out).
    drift = RC.resolve_recovery({
        "gate": "verify",
        "content": {"outcome": "assertions_failed", "failures": [{
            "type": "AssertionError", "message": "assert 200 == 201",
            "traceback": ("tests/test_api.py:15: in test_create\n"
                          "app/api.py:42: in create_item")}]}})
    assert drift["action"] == "regenerate_files"
    assert drift["targets"] == ["app/api.py"]


def test_resolve_recovery_falls_back_to_whole_tree_without_impl():
    # Only the test frame is named -> honest whole-tree fallback.
    out = RC.resolve_recovery({
        "gate": "verify",
        "content": {"outcome": "assertions_failed", "failures": [{
            "type": "AssertionError", "message": "assert False",
            "traceback": "tests/test_logic.py:20: in test_invariant"}]}})
    assert out["action"] == "regenerate_whole_tree"
    assert out["action"] not in RC.SCOPED_ACTIONS


def test_evaluate_recovery_over_golden_corpus():
    golden = _load_recovery_golden()
    out = RC.evaluate_recovery(golden)
    assert out["n_items"] == len(golden)
    agg = out["aggregate"]
    # Every golden case must resolve to its pinned expected action.
    assert agg["action_match_rate"] == pytest.approx(1.0)
    # Scoped recovery must dominate, and cost must beat the whole-tree baseline.
    assert agg["scoped_recovery_rate"] >= 0.7
    assert agg["mean_rounds_to_green"] < agg["baseline_mean_rounds"]
    assert agg["mean_tokens"] < agg["baseline_mean_tokens"]


# --------------------------------------------------------------------------
# Full gate: codegen + recovery always run; retrieval only when faiss present.
# --------------------------------------------------------------------------
def test_run_gate_passes_on_repo_golden():
    report, ok = run_gate(EVALS_DIR)
    assert ok, f"gate failed: {report['failures']}"
    assert "codegen" in report["sections"]
    assert report["sections"]["codegen"]["aggregate"]["expectation_match_rate"] == 1.0
    assert "recovery" in report["sections"]
    assert report["sections"]["recovery"]["aggregate"]["action_match_rate"] == 1.0


def test_retrieval_end_to_end_over_sample_repo(tmp_path):
    pytest.importorskip("faiss")
    golden = [
        {"query": "add two numbers", "relevant": ["calc.py::method::Calculator.add"]},
        {"query": "reconnect database",
         "relevant": ["db.py::function::databaseReconnect"]},
    ]
    embedder = RT.DeterministicEmbedder(dim=32)
    sample_repo = str(Path(EVALS_DIR) / "sample_repo")
    artifacts = RT.build_sample_index(sample_repo, str(tmp_path / "idx"), embedder)
    out = RT.evaluate_retrieval(golden, artifacts, embedder)
    assert out["n_queries"] == 2
    assert out["aggregate"]["recall@10"] == pytest.approx(1.0)
    assert out["aggregate"]["mrr"] > 0.0
