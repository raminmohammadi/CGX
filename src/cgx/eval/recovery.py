

"""Offline recovery-quality evaluation against a golden failure corpus (E1).

Each golden case pins a *real* gate failure (a VERIFY / RUNTIME report content
dict, a build-smoke stderr, or an over-scoped de-scope scenario) drawn from the
sessions in ``docs/Agent.md``. The eval runs the failure through the **real**
deterministic decision surface -- ``cgx.session.repair.classify`` for the
classification token, its traceback / build-error extraction helpers for the
concrete fix targets, and the router's own ``DIAGNOSE_CLASSIFICATIONS`` /
regenerate-class constants -- and maps the outcome to a recovery-action class.

The point is regression protection for the Phase 1-3 overhaul: a change that
makes a scoped-fixable failure fall back to a whole-tree regenerate flips the
resolved action here and the gate fails. It is intentionally provider-free (no
LLM), so it runs in every CI job alongside the codegen eval. ``rounds`` /
``tokens`` come from a transparent per-action cost model so the gate can track
the scoped-vs-whole-tree savings the overhaul was built to deliver.

Because the whole eval is provider-free, it also doubles as the **degradation
floor** for the DIAGNOSE reasoning rung (E2): it measures exactly the behavior
the agent guarantees when the LLM is unavailable. Two guardrails lock that
floor to the release gate -- ``never_worse_rate`` (every resolved action,
including the provider-outage ``escalate`` fallback, costs no more than the old
whole-tree ladder) and ``determinism_ok`` (the resolver is pure, so re-running
the corpus yields byte-identical verdicts and the router stays replayable).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from cgx.eval import metrics as M
from cgx.logging_setup import get_logger
from cgx.session.repair.classify import (
    DIAGNOSE_CLASSIFICATIONS,
    classify_runtime_report,
    classify_verify_report,
    generic_build_error_files,
    runtime_failure_text,
    traceback_source_files,
    unresolved_entry_paths,
    unresolved_import_sources,
)
from cgx.session.scope import unrunnable_descope_needles

logger = get_logger(__name__)

# Recovery-action taxonomy. A *scoped* action fixes only what broke (a patch, a
# dependency add/remove, or a targeted regenerate of the localized file[s]); a
# *whole-tree* action is the old ladder's nuke-and-regenerate / escalate.
SCOPED_ACTIONS = frozenset(
    {"patch_files", "install_deps", "remove_dependency", "regenerate_files"})
WHOLE_TREE_ACTIONS = frozenset({"regenerate_whole_tree", "escalate"})

# Classifications with a mechanical locator -> a scoped patch.
_PATCH_CLASSES = frozenset(
    {"unittest_pytest_mix", "missing_module_pythonpath", "missing_fixture"})
# Classifications the fix is a package install / pin, not a source edit.
_INSTALL_CLASSES = frozenset({"missing_dependency", "third_party_import_break"})
# Regenerate classes the router does *not* scope to a file set (no target_files
# marker in ``_select_repair_strategy``): they re-author the whole subtree.
_WHOLE_TREE_CLASSES = frozenset(
    {"circular_import", "first_party_symbol_mismatch", "relative_import_error",
     "undefined_name", "empty_test_suite"})

# Deterministic cost model: rounds-to-green + provider tokens a recovery action
# class typically spends. A scoped action converges in one round on cheap
# targeted context; a whole-tree regenerate re-authors the tree over several
# rounds (and historically reproduced the identical miss).
_ACTION_COST: Dict[str, Dict[str, int]] = {
    "patch_files": {"rounds": 1, "tokens": 1200},
    "install_deps": {"rounds": 1, "tokens": 400},
    "remove_dependency": {"rounds": 1, "tokens": 300},
    "regenerate_files": {"rounds": 1, "tokens": 3500},
    "regenerate_whole_tree": {"rounds": 3, "tokens": 18000},
    "escalate": {"rounds": 3, "tokens": 18000},
}
_WHOLE_TREE_BASELINE = _ACTION_COST["regenerate_whole_tree"]


def _is_test_file(rel: str) -> bool:
    norm = (rel or "").replace("\\", "/")
    base = norm.rsplit("/", 1)[-1]
    return (base.startswith("test_") or base.endswith("_test.py")
            or norm.startswith("tests/") or "/tests/" in norm)


def _scoped_targets(gate: str, classification: str,
                    content: Dict[str, Any]) -> List[str]:
    """Concrete fix targets the real extraction helpers recover from a failure."""
    if gate == "build":
        stderr = str(content.get("stderr") or "")
        return (list(unresolved_import_sources(stderr))
                or list(unresolved_entry_paths(stderr))
                or list(generic_build_error_files(stderr)))
    if gate == "runtime":
        return list(traceback_source_files({"stdout": runtime_failure_text(content)}))
    files = list(traceback_source_files(content))
    # assertion_drift re-authors only the implementation the failure reached
    # (tests encode the contract); collection_error keeps the raw frame set.
    if classification == "assertion_drift":
        return [f for f in files if not _is_test_file(f)]
    return files


def _map_action(gate: str, classification: str, targets: Sequence[str]) -> str:
    if gate == "build":
        return "regenerate_files" if targets else "escalate"
    if classification in _INSTALL_CLASSES:
        return "install_deps"
    if classification in _PATCH_CLASSES:
        return "patch_files"
    if classification in _WHOLE_TREE_CLASSES:
        return "regenerate_whole_tree"
    if classification in DIAGNOSE_CLASSIFICATIONS:
        if classification == "unknown":
            return "regenerate_whole_tree"
        return "regenerate_files" if targets else "regenerate_whole_tree"
    return "regenerate_whole_tree"


def resolve_recovery(case: Dict[str, Any]) -> Dict[str, Any]:
    """Drive the real classifier + extractors to a recovery-action verdict."""
    gate = str(case.get("gate") or "verify").strip()
    content = case.get("content") or {}
    if gate == "descope":
        needles = unrunnable_descope_needles(
            tuple(case.get("requested_features") or ()))
        blob = " ".join(str(v) for v in content.values()).lower()
        hit = any(n in blob for n in needles)
        action = "remove_dependency" if hit else "regenerate_whole_tree"
        return {"classification": "unrunnable_dependency", "targets": [],
                "action": action}
    if gate == "runtime":
        classification = classify_runtime_report(content)
    elif gate == "build":
        classification = "build_error"
    else:
        classification = classify_verify_report(content)
    targets = _scoped_targets(gate, classification, content)
    return {"classification": classification, "targets": targets,
            "action": _map_action(gate, classification, targets)}


def score_case(case: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve one case and reduce it to boolean + cost quality signals."""
    resolved = resolve_recovery(case)
    action = resolved["action"]
    cost = _ACTION_COST.get(action, _WHOLE_TREE_BASELINE)
    expected = str(case.get("expect_action") or "").strip()
    # Degradation guardrail (E2): a recovery path must never cost *more* than
    # the old nuke-and-regenerate ladder. The whole-tree action is the
    # baseline, so every resolved action -- including the provider-free
    # ``escalate`` fallback -- has to converge in no more rounds/tokens.
    never_worse = (cost["rounds"] <= _WHOLE_TREE_BASELINE["rounds"]
                   and cost["tokens"] <= _WHOLE_TREE_BASELINE["tokens"])
    return {
        "id": case.get("id"),
        "gate": case.get("gate"),
        "classification": resolved["classification"],
        "action": action,
        "scoped": action in SCOPED_ACTIONS,
        "expected_action": expected,
        "matched_action": (not expected) or action == expected,
        "never_worse": never_worse,
        "rounds": cost["rounds"],
        "tokens": cost["tokens"],
    }


def aggregate_recovery(per_item: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Scoped-recovery rate, action-match rate, and cost savings vs the ladder."""
    base_rounds = float(_WHOLE_TREE_BASELINE["rounds"])
    base_tokens = float(_WHOLE_TREE_BASELINE["tokens"])
    mean_rounds = M.mean([float(r["rounds"]) for r in per_item])
    mean_tokens = M.mean([float(r["tokens"]) for r in per_item])
    return {
        "scoped_recovery_rate":
            M.mean([1.0 if r["scoped"] else 0.0 for r in per_item]),
        "action_match_rate":
            M.mean([1.0 if r["matched_action"] else 0.0 for r in per_item]),
        "mean_rounds_to_green": mean_rounds,
        "mean_tokens": mean_tokens,
        "baseline_mean_rounds": base_rounds,
        "baseline_mean_tokens": base_tokens,
        "rounds_saved_rate":
            (base_rounds - mean_rounds) / base_rounds if base_rounds else 0.0,
        "tokens_saved_rate":
            (base_tokens - mean_tokens) / base_tokens if base_tokens else 0.0,
        "never_worse_rate":
            M.mean([1.0 if r["never_worse"] else 0.0 for r in per_item]),
    }


def _resolution_signature(golden: Sequence[Dict[str, Any]]) -> tuple:
    """A hashable projection of every case's resolved verdict + cost."""
    return tuple(
        (r["id"], r["classification"], r["action"], r["rounds"], r["tokens"])
        for r in (score_case(c) for c in golden)
    )


def check_recovery_determinism(
        golden: Sequence[Dict[str, Any]], repeats: int = 3) -> bool:
    """Determinism guardrail: the resolver is pure -- re-running the corpus
    yields byte-identical verdicts every time (no LLM, no ordering leak, no
    hidden state), so the router's dispatch stays replayable and testable.
    """
    first = _resolution_signature(golden)
    return all(_resolution_signature(golden) == first for _ in range(max(0, repeats - 1)))


def evaluate_recovery(golden: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Score every golden failure and aggregate the recovery-quality metrics."""
    per_item = [score_case(c) for c in golden]
    for r in per_item:
        logger.info(
            "eval.recovery: id=%s class=%s action=%s scoped=%s matched=%s "
            "never_worse=%s",
            r["id"], r["classification"], r["action"], r["scoped"],
            r["matched_action"], r["never_worse"],
        )
    aggregate = aggregate_recovery(per_item)
    # Determinism is a gate-checked invariant, surfaced as a 0/1 metric so a
    # regression that introduces nondeterminism fails the release gate.
    aggregate["determinism_ok"] = (
        1.0 if check_recovery_determinism(golden) else 0.0)
    return {
        "n_items": len(per_item),
        "per_item": per_item,
        "aggregate": aggregate,
    }
