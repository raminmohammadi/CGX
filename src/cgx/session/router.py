

"""Deterministic Router for the session-shaped agent backbone.

The router replaces ``Planner.plan`` from the legacy agent loop. It is
pure Python with no LLM calls and no IO: every method takes the
current session state plus an event and returns a :class:`RouterPlan`
of typed actions that the caller applies to the store.

Three entry points cover every transition in Phase 1:

* :meth:`on_user_message` -- user posts a fresh objective or a
  follow-up message to a session.
* :meth:`on_task_completed` -- an executor finished a task; the
  router decides what to spawn next based on ``parent.kind``.
* :meth:`on_decision_recorded` -- user resolved an ``ASK_USER`` task
  via a typed :class:`Decision`; the router marks it done and spawns
  the successor.

Phase 1 wires EXPLORE -> ASK_USER. Later phases extend the
``TASK_SUCCESSOR`` table with INVESTIGATE / RECOMMEND / PLAN_CHANGE /
APPLY / VERIFY without changing the router's shape.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from cgx.session.actions import (
    AttachDecisionToTask,
    CreateTask,
    RecordDecision,
    RouterAction,
    RouterPlan,
    UpdateSessionStatus,
    UpdateTaskStatus,
)
from cgx.session.budget import LoopBudget
from cgx.session.greenfield_edges import (
    _apply_failed_files_actions,
    _decompose_retry_actions,
    _diagnose_dispatch_actions,
    _make_budget_ask,
    _repair_install_deps_actions,
    _repair_regenerate_actions,
    _repair_resolve_deps_actions,
    _repair_terminal_failure_actions,
    _scaffold_contract_regenerate_actions,
    _scaffold_failed_files_actions,
    _scaffold_payload_regenerate_actions,
    _scaffold_resume_actions,
    _scaffold_skill_regenerate_actions,
    _verify_lesson_actions,
)
from cgx.session.repair.classify import DIAGNOSE_CLASSIFICATIONS
from cgx.session.models import (
    Decision,
    DecisionKind,
    Session,
    SessionMode,
    SessionStatus,
    TaskKind,
    TaskNode,
    TaskNodeStatus,
)
from cgx.trace import traced

logger = logging.getLogger(__name__)


# --------------------- successor table ---------------------

def _explore_to_ask(parent: TaskNode) -> List[TaskNode]:
    """Spawn the ``ASK_USER`` follow-up for a finished EXPLORE."""
    artifact_id = parent.produced_artifact_id
    return [TaskNode.new(
        session_id=parent.session_id,
        kind=TaskKind.ASK_USER,
        name="Pick a direction",
        description=("Choose which of the surfaced directions to "
                     "investigate first."),
        parent_task_id=parent.task_id,
        inputs={
            "expected_kind": DecisionKind.CHOOSE_PATH.value,
            "directions_artifact_id": artifact_id,
            "prior_goal": parent.inputs.get("goal"),
        },
    )]


def _investigate_to_recommend(parent: TaskNode) -> List[TaskNode]:
    """Spawn the ``RECOMMEND`` follow-up for a finished INVESTIGATE."""
    return [TaskNode.new(
        session_id=parent.session_id,
        kind=TaskKind.RECOMMEND,
        name="Recommend next steps",
        description=("Synthesize concrete next-step recommendations "
                     "from the investigation findings."),
        parent_task_id=parent.task_id,
        inputs={
            "findings_artifact_id": parent.produced_artifact_id,
            "anchor_chunk_id": parent.inputs.get("anchor_chunk_id"),
            "prior_goal": (parent.inputs.get("prior_goal")
                           or parent.inputs.get("goal")),
            "title": parent.inputs.get("title"),
        },
    )]


def _recommend_to_ask(parent: TaskNode) -> List[TaskNode]:
    """Spawn the ``ASK_USER`` (CHOOSE_RECOMMENDATION) for a finished RECOMMEND."""
    return [TaskNode.new(
        session_id=parent.session_id,
        kind=TaskKind.ASK_USER,
        name="Pick a recommendation",
        description=("Choose which of the surfaced recommendations to "
                     "act on next."),
        parent_task_id=parent.task_id,
        inputs={
            "expected_kind": DecisionKind.CHOOSE_RECOMMENDATION.value,
            "recommendations_artifact_id": parent.produced_artifact_id,
            "findings_artifact_id":
                parent.inputs.get("findings_artifact_id"),
            "prior_goal": parent.inputs.get("prior_goal"),
        },
    )]


def _plan_change_to_ask(parent: TaskNode) -> List[TaskNode]:
    """Spawn the ``ASK_USER`` (APPROVE) gate for a finished PLAN_CHANGE."""
    return [TaskNode.new(
        session_id=parent.session_id,
        kind=TaskKind.ASK_USER,
        name="Approve change plan",
        description=("Review the proposed code-change plan and decide "
                     "whether to apply it."),
        parent_task_id=parent.task_id,
        inputs={
            "expected_kind": DecisionKind.APPROVE.value,
            "plan_artifact_id": parent.produced_artifact_id,
            "prior_goal": parent.inputs.get("prior_goal"),
            "recommendation": parent.inputs.get("recommendation"),
        },
    )]


def _apply_to_verify(parent: TaskNode) -> List[TaskNode]:
    """Spawn the post-APPLY successor.

    Greenfield projects need a runtime environment before pytest can
    even collect: a freshly-scaffolded Flask app has nothing installed
    yet, so VERIFY would always fail at collection time. We splice
    BOOTSTRAP_ENV in between -- it provisions ``.venv`` and installs
    declared + dynamically-imported dependencies, then its own
    successor (see :func:`_bootstrap_to_verify`) spawns VERIFY.

    Explore-mode sessions keep the direct APPLY -> VERIFY edge: the
    working tree's runtime is the user's existing venv, not something
    we manage.

    Repair cycles in greenfield mode also skip BOOTSTRAP_ENV: the venv
    has already been provisioned in the original pass, and the upstream
    REPAIR carries the prior ``build_artifact_id`` forward through
    APPLY.inputs. Re-bootstrapping would just spend time reinstalling
    the same packages.

    A DIAGNOSE-driven scoped patch carries a ``reverify_origin_gate``
    marker (design §C2): the venv is already live and every other gate
    already passed, so instead of VERIFY we splice RE_VERIFY to re-run
    only the origin report's failing test file(s).
    """
    inputs = parent.inputs or {}
    if str(inputs.get("reverify_origin_gate") or "").strip():
        return _re_verify_node(
            parent, build_artifact_id=inputs.get("build_artifact_id"),
            apply_artifact_id=parent.produced_artifact_id)
    mode = str(parent.inputs.get("mode") or "").strip()
    has_build_artifact = bool(
        str(parent.inputs.get("build_artifact_id") or "").strip())
    budget = LoopBudget.from_inputs(parent.inputs)
    if mode == SessionMode.GREENFIELD.value and not has_build_artifact:
        return [TaskNode.new(
            session_id=parent.session_id,
            kind=TaskKind.BOOTSTRAP_ENV,
            name="Bootstrap project environment",
            description=("Create a project venv and install declared + "
                         "undeclared dependencies so VERIFY can run."),
            parent_task_id=parent.task_id,
            inputs={
                "apply_artifact_id": parent.produced_artifact_id,
                "plan_artifact_id": parent.inputs.get("plan_artifact_id"),
                "scaffold_artifact_id":
                    parent.inputs.get("scaffold_artifact_id"),
                "prior_goal": parent.inputs.get("prior_goal"),
                "mode": mode,
                **budget.repair_chain_inputs(),
            },
        )]
    return [TaskNode.new(
        session_id=parent.session_id,
        kind=TaskKind.VERIFY,
        name="Verify applied changes",
        description=("Run impacted tests against the working tree to "
                     "validate the applied changes."),
        parent_task_id=parent.task_id,
        inputs={
            "apply_artifact_id": parent.produced_artifact_id,
            "plan_artifact_id": parent.inputs.get("plan_artifact_id"),
            "scaffold_artifact_id": parent.inputs.get("scaffold_artifact_id"),
            "build_artifact_id": parent.inputs.get("build_artifact_id"),
            "prior_goal": parent.inputs.get("prior_goal"),
            "mode": parent.inputs.get("mode"),
            **budget.repair_chain_inputs(),
        },
    )]


def _bootstrap_to_api_check(parent: TaskNode) -> List[TaskNode]:
    """Spawn API_CHECK once the project environment is provisioned.

    Always runs in greenfield mode (the only path that creates a
    BOOTSTRAP_ENV node). API_CHECK statically walks the applied files
    and resolves every third-party ``from <pkg> import <name>`` and
    aliased ``pkg.attr`` access under the bootstrapped venv. Its
    successor (see :func:`_api_check_to_smoke_or_repair`) then chains
    SMOKE on pass / skip, or REPAIR on a hallucinated symbol.

    A DIAGNOSE-driven dependency fix (add/remove) carries a
    ``reverify_origin_gate`` marker (design §C2): the re-provisioned venv
    is all the fix needed, so instead of re-probing API_CHECK/SMOKE we
    splice RE_VERIFY to re-run only the origin report's failing test(s).
    """
    inputs = parent.inputs or {}
    if str(inputs.get("reverify_origin_gate") or "").strip():
        return _re_verify_node(
            parent, build_artifact_id=parent.produced_artifact_id,
            apply_artifact_id=inputs.get("apply_artifact_id"))
    return [TaskNode.new(
        session_id=parent.session_id,
        kind=TaskKind.API_CHECK,
        name="Probe third-party API references",
        description=("Resolve every third-party symbol the applied files "
                     "reference under the bootstrapped venv to fail fast "
                     "on hallucinated names before SMOKE/VERIFY."),
        parent_task_id=parent.task_id,
        inputs={
            "build_artifact_id": parent.produced_artifact_id,
            "apply_artifact_id": parent.inputs.get("apply_artifact_id"),
            "plan_artifact_id": parent.inputs.get("plan_artifact_id"),
            "scaffold_artifact_id":
                parent.inputs.get("scaffold_artifact_id"),
            "prior_goal": parent.inputs.get("prior_goal"),
            "mode": parent.inputs.get("mode"),
            **LoopBudget.from_inputs(parent.inputs).repair_chain_inputs(),
        },
    )]


# Outcomes that REPAIR knows how to attempt a fix for on an API_CHECK
# report. Only ``failed`` is repairable; ``passed`` and ``skipped``
# chain to SMOKE.
_REPAIRABLE_API_CHECK_OUTCOMES = frozenset({"failed"})

# How many times one API_CHECK failure signature may repeat before the
# loop is a proven dead end. Two rungs: the first repeat re-enters REPAIR
# with ``repair_escalation`` set so it abandons the strategy that just
# no-opped, the second gives up. The shared ``repair_attempt`` cap still
# bounds the loop independently.
_API_CHECK_SIGNATURE_LADDER = 2


def _api_check_to_smoke_or_repair(parent: TaskNode) -> List[TaskNode]:
    """Spawn SMOKE on a clean API_CHECK; REPAIR on a hallucinated symbol.

    Mirrors :func:`_smoke_to_verify_or_repair`: ``passed`` / ``skipped``
    hand off to SMOKE with the API_CHECK report carried forward;
    ``failed`` routes to REPAIR with the API_CHECK_REPORT as the source
    artifact, gated by the shared retry budget + flap detector.
    """
    outputs = parent.outputs or {}
    outcome = str(outputs.get("outcome") or "").strip()
    mode = str(parent.inputs.get("mode") or "").strip()
    budget = LoopBudget.from_inputs(parent.inputs)
    if outcome not in _REPAIRABLE_API_CHECK_OUTCOMES:
        return [TaskNode.new(
            session_id=parent.session_id,
            kind=TaskKind.SMOKE,
            name="Smoke-test environment imports",
            description=("Import each third-party top-level package the "
                         "applied files declare to fail fast on dependency "
                         "breakage before VERIFY runs pytest."),
            parent_task_id=parent.task_id,
            inputs={
                "build_artifact_id": parent.inputs.get("build_artifact_id"),
                "api_check_artifact_id": parent.produced_artifact_id,
                "apply_artifact_id": parent.inputs.get("apply_artifact_id"),
                "plan_artifact_id": parent.inputs.get("plan_artifact_id"),
                "scaffold_artifact_id":
                    parent.inputs.get("scaffold_artifact_id"),
                "prior_goal": parent.inputs.get("prior_goal"),
                "mode": mode,
                **budget.repair_chain_inputs(),
            },
        )]
    if mode != SessionMode.GREENFIELD.value:
        return []
    if budget.repair_exhausted:
        return []
    new_signature = str(outputs.get("failure_signature") or "").strip()
    if not new_signature:
        failed = outputs.get("failed_count")
        new_signature = f"api_check_failed|count={failed}"
    # A repeated signature used to end the session outright. One rung of
    # escalation first: the first round may pick install_deps, and when
    # pip cannot satisfy the name (session ses_fa6f72a9d3da4217 asked it
    # for a hallucinated ``app``) that round is a no-op that reproduces
    # the identical signature. REPAIR reads ``repair_escalation`` and
    # switches to a regenerate that removes the offending import; a
    # second repeat is a genuine dead end.
    escalation = budget.signature_repeats(new_signature)
    if escalation >= _API_CHECK_SIGNATURE_LADDER:
        return []
    return [TaskNode.new(
        session_id=parent.session_id,
        kind=TaskKind.REPAIR,
        name="Repair hallucinated API references",
        description=("Classify the upstream API_CHECK failure and propose "
                     "a targeted patch (typically a rename, an import "
                     "rewrite, or a dependency pin) the shared APPLY "
                     "executor can write."),
        parent_task_id=parent.task_id,
        inputs={
            "api_check_artifact_id": parent.produced_artifact_id,
            "build_artifact_id": parent.inputs.get("build_artifact_id"),
            "apply_artifact_id": parent.inputs.get("apply_artifact_id"),
            "prior_goal": parent.inputs.get("prior_goal"),
            "mode": mode,
            "repair_escalation": escalation,
            **budget.spend_repair(new_signature).repair_chain_inputs(),
        },
    )]


# Outcomes that REPAIR knows how to attempt a fix for on a SMOKE_REPORT.
# Only ``failed`` is repairable; ``passed`` and ``skipped`` chain to VERIFY.
_REPAIRABLE_SMOKE_OUTCOMES = frozenset({"failed"})


def _smoke_to_verify_or_repair(parent: TaskNode) -> List[TaskNode]:
    """Spawn VERIFY on a clean smoke run; REPAIR on an import failure.

    SMOKE only runs in greenfield mode (it's only ever spawned by
    :func:`_bootstrap_to_smoke`). On ``passed`` / ``skipped`` we hand
    off to VERIFY with the same inputs we would have forwarded from
    BOOTSTRAP_ENV. On ``failed`` -- a third-party import broke under
    the bootstrapped venv -- we route to REPAIR with the SMOKE_REPORT
    as the source artifact, gated by the same retry budget and
    flap-detector used by the VERIFY-driven repair loop.
    """
    outputs = parent.outputs or {}
    outcome = str(outputs.get("outcome") or "").strip()
    mode = str(parent.inputs.get("mode") or "").strip()
    budget = LoopBudget.from_inputs(parent.inputs)
    if outcome not in _REPAIRABLE_SMOKE_OUTCOMES:
        return [TaskNode.new(
            session_id=parent.session_id,
            kind=TaskKind.VERIFY,
            name="Verify applied changes",
            description=("Run tests under the project's bootstrapped venv "
                         "to validate the applied changes."),
            parent_task_id=parent.task_id,
            inputs={
                "build_artifact_id": parent.inputs.get("build_artifact_id"),
                "smoke_artifact_id": parent.produced_artifact_id,
                "apply_artifact_id": parent.inputs.get("apply_artifact_id"),
                "plan_artifact_id": parent.inputs.get("plan_artifact_id"),
                "scaffold_artifact_id":
                    parent.inputs.get("scaffold_artifact_id"),
                "prior_goal": parent.inputs.get("prior_goal"),
                "mode": mode,
                **budget.repair_chain_inputs(),
            },
        )]
    if mode != SessionMode.GREENFIELD.value:
        return []
    if budget.repair_exhausted:
        return []
    new_signature = str(outputs.get("failure_signature") or "").strip()
    if not new_signature:
        failed = outputs.get("failed_count")
        new_signature = f"smoke_failed|count={failed}"
    if budget.seen(new_signature):
        return []
    return [TaskNode.new(
        session_id=parent.session_id,
        kind=TaskKind.REPAIR,
        name="Repair failed smoke imports",
        description=("Classify the upstream SMOKE import failure and "
                     "propose a targeted patch (typically a dependency "
                     "pin) the shared APPLY executor can write."),
        parent_task_id=parent.task_id,
        inputs={
            "smoke_artifact_id": parent.produced_artifact_id,
            "build_artifact_id": parent.inputs.get("build_artifact_id"),
            "apply_artifact_id": parent.inputs.get("apply_artifact_id"),
            "prior_goal": parent.inputs.get("prior_goal"),
            "mode": mode,
            **budget.spend_repair(new_signature).repair_chain_inputs(),
        },
    )]


def _clarify_requirements_to_ask(parent: TaskNode) -> List[TaskNode]:
    """Spawn the ASK_USER(CLARIFY_ANSWERS) for a finished CLARIFY_REQUIREMENTS."""
    return [TaskNode.new(
        session_id=parent.session_id,
        kind=TaskKind.ASK_USER,
        name="Answer requirements questions",
        description=("Answer the clarifying questions so the agent can "
                     "scaffold a project tailored to your needs."),
        parent_task_id=parent.task_id,
        inputs={
            "expected_kind": DecisionKind.CLARIFY_ANSWERS.value,
            "requirements_artifact_id": parent.produced_artifact_id,
            "prior_goal": parent.inputs.get("goal"),
        },
    )]


def _decompose_to_ask(parent: TaskNode) -> List[TaskNode]:
    """Spawn the ASK_USER(APPROVE_PLAN) gate for a finished DECOMPOSE.

    A re-planned DECOMPOSE carries the failed chain's flap ledger
    (``prior_failure_signatures``) and the spent ``regenerate_attempt``
    count; both ride the approval gate down to the new SCAFFOLD so the
    revised manifest's chain is not amnesiac about failures the previous
    manifest already produced and cannot silently reset the syntax-churn
    regenerate budget to zero.
    """
    inputs: Dict[str, Any] = {
        "expected_kind": DecisionKind.APPROVE_PLAN.value,
        "work_plan_artifact_id": parent.produced_artifact_id,
        "prior_goal": parent.inputs.get("prior_goal"),
        "requirements_artifact_id":
            parent.inputs.get("requirements_artifact_id"),
        "replan_attempt": parent.inputs.get("replan_attempt"),
        "regenerate_attempt": parent.inputs.get("regenerate_attempt"),
    }
    signatures = list(LoopBudget.from_inputs(
        parent.inputs).prior_failure_signatures)
    if signatures:
        inputs["prior_failure_signatures"] = signatures
    return [TaskNode.new(
        session_id=parent.session_id,
        kind=TaskKind.ASK_USER,
        name="Approve work plan",
        description=("Review the proposed file manifest and approve to "
                     "begin scaffolding."),
        parent_task_id=parent.task_id,
        inputs=inputs,
    )]


def _coerce_count(value: Any) -> Optional[int]:
    """Best-effort ``int`` coercion; returns ``None`` for missing/garbage."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _repair_progress_stalled(
        new_count: Optional[int],
        prior_counts: List[int],
        new_passing: Optional[int] = None,
        prior_passing: Optional[List[int]] = None) -> bool:
    """True when the coverage-aware progress ledger shows no forward step.

    ``new_count`` is the number of tests failing in the just-finished
    VERIFY; ``prior_counts`` is the ordered history of failing counts from
    earlier rounds of the *same* repair loop. The primary progress signal
    is the failing count strictly dropping round over round.

    ``new_passing`` / ``prior_passing`` extend that with the passing-test
    trend (#5): a round that did not lower the failing count can still be
    real forward progress if it made *more* tests pass than the previous
    round -- e.g. it fixed one assertion while a newly-unskipped test began
    failing, holding the failing count flat. Such a round is NOT a stall.
    So the loop is stalled only when neither lever moved forward: the
    failing count did not drop AND the passing count did not rise.

    A missing failing count (``None`` -- e.g. a non-assertion outcome
    where a test count is not a meaningful progress signal) is
    inconclusive and never on its own declares a stall: the caller still
    applies the signature-flap backstop and the absolute
    :data:`~cgx.session.budget.REPAIR_BUDGET` cap.
    """
    if new_count is None or not prior_counts:
        return False
    failing_dropped = new_count < prior_counts[-1]
    passing_rose = (new_passing is not None
                    and bool(prior_passing)
                    and new_passing > prior_passing[-1])
    return not (failing_dropped or passing_rose)


# Outcomes that REPAIR knows how to attempt a fix for. ``passed`` and
# ``skipped`` are terminal -- they're not failures. ``pytest_missing``
# is BOOTSTRAP_ENV's job, not REPAIR's. ``no_tests_collected`` is
# repairable only when test files were actually selected but pytest
# collected zero tests (malformed tests -- see
# :func:`_verify_to_repair_or_terminal`); a genuinely test-free project
# still terminates cleanly. ``failed`` is a non-pytest runner (e.g. an
# ``npm`` build/test) that exited non-zero: it classifies as ``unknown``
# in the repair classifier, which routes to a re-scaffold (regenerate)
# so a JS/TS build break is not a silent false success. ``no_tests`` is
# deliberately absent: a JS/TS project whose build passes but wired up no
# tests has nothing for a regenerate to mechanically fix, so it is not
# repairable (it terminates honestly rather than looping).
_REPAIRABLE_VERIFY_OUTCOMES = frozenset({
    "assertions_failed",
    "collection_error",
    "no_tests_collected",
    "failed",
})


# Terminal VERIFY outcomes that mean the greenfield write loop delivered
# a working suite. Everything else that reaches a terminal VERIFY (with
# no REPAIR spawned) is a definitive failure -- never a "success" and
# never an ASK_USER prompt. ``skipped`` counts as success because it is
# an explicit opt-out, not a broken suite. ``no_tests`` is deliberately
# absent: a passing build with no test suite is not a verified suite, so
# it fails honestly instead of reporting a false green.
_VERIFY_SUCCESS_OUTCOMES = frozenset({"passed", "skipped"})

# How many times one VERIFY failure signature may repeat before the loop
# is a proven dead end. Mirrors :data:`_API_CHECK_SIGNATURE_LADDER`, and
# applies only to the one classification whose first round can legitimately
# no-op (``missing_dependency`` -> install_deps).
_VERIFY_SIGNATURE_LADDER = 2


def _verify_to_repair_or_terminal(parent: TaskNode) -> List[TaskNode]:
    """Spawn REPAIR after a fixable VERIFY failure; otherwise terminal.

    Triggers only in greenfield mode (auto-apply is part of the
    greenfield contract; explore-mode write loops keep their existing
    approval gates). Progress is judged two ways, both read off the
    parent's inputs so the router stays IO-free:

    * a failing-test-count *trend* (``prior_failing_counts``): for a real
      ``assertions_failed`` outcome the loop keeps going only while the
      count strictly drops round over round (see
      :func:`_repair_progress_stalled`) -- the primary, progress-aware
      guard that lets a genuinely-improving hard task iterate further;
    * a failure-*signature* flap backstop (``prior_failure_signatures``):
      if the just-finished VERIFY's signature already appears in the
      list the loop is churning and we refuse another REPAIR. This still
      covers non-assertion outcomes where a test count is meaningless.

    Both sit under the absolute :data:`~cgx.session.budget.REPAIR_BUDGET`
    cap. The attempt
    counter lives in ``parent.inputs["repair_attempt"]`` (incremented by
    the REPAIR -> APPLY -> VERIFY chain), so the router can read it
    without walking the task tree.
    """
    mode = str(parent.inputs.get("mode") or "").strip()
    if mode != SessionMode.GREENFIELD.value:
        return []
    outputs = parent.outputs or {}
    outcome = str(outputs.get("outcome") or "").strip()
    if outcome not in _REPAIRABLE_VERIFY_OUTCOMES:
        return []
    # ``no_tests_collected`` (pytest exit 5) is only a failure when
    # pytest actually selected test files but found zero test functions
    # in them (malformed tests -- e.g. ``def test_*`` nested inside a
    # fixture). When nothing was selected the project simply has no
    # tests yet, which is a clean terminal state, not a repair trigger.
    if (outcome == "no_tests_collected"
            and int(outputs.get("tests_selected_count") or 0) <= 0):
        return []
    budget = LoopBudget.from_inputs(parent.inputs)
    if budget.repair_exhausted:
        return []
    # Coverage-aware progress gate (#5): for a real test failure the
    # failing-test count is a truer progress signal than failure-signature
    # identity -- a loop can keep churning fresh signatures while fixing
    # nothing. Keep repairing while the failing count strictly drops OR the
    # passing count strictly rises (a round that fixed one test while
    # another newly began failing held the failing count flat but still
    # made forward progress). A stall (neither lever moved forward) ends
    # the loop. The failing-count trend is trusted for ``assertions_failed``
    # *and* ``collection_error``: for a collection error ``failing_count`` is
    # the number of modules erroring during collection (e.g. import fixes
    # landing one module at a time), so a strictly-dropping count is genuine
    # forward progress that should buy another round under the budget. The
    # ``passing_count`` lever stays ``assertions_failed``-only ("M passing"
    # is meaningless when nothing collected); every other outcome falls back
    # to the signature-flap backstop below.
    new_count = (_coerce_count(outputs.get("failing_count"))
                 if outcome in ("assertions_failed", "collection_error")
                 else None)
    prior_counts = list(budget.prior_failing_counts)
    new_passing = (_coerce_count(outputs.get("passing_count"))
                   if outcome == "assertions_failed" else None)
    prior_passing = list(budget.prior_passing_counts)
    # Read the VERIFY_REPORT's failure_signature lazily by deferring to
    # the classifier; the router stays free of I/O by using a precomputed
    # signature stashed by the runner-style ``outputs``. Falls back to a
    # ``returncode``+ ``outcome`` composite so a missing signature still
    # gives the progress detector a stable token to compare.
    new_signature = str(outputs.get("failure_signature") or "").strip()
    if not new_signature:
        new_signature = f"{outcome}|rc={outputs.get('returncode')}"
    classification = str(outputs.get("classification") or "").strip()
    # A repeated signature (and a flat failing count) normally ends the
    # loop. One exception, one rung: a ``missing_dependency`` round routes
    # to install_deps, and when pip cannot satisfy the name (a hallucinated
    # ``httpx2``) that round is a no-op reproducing the identical signature
    # *and* the identical count -- both guards fire on a failure the loop
    # has never actually attempted to fix. REPAIR reads
    # ``repair_escalation`` and switches to a regenerate that drops the
    # un-installable import; a second repeat is a genuine dead end. Every
    # other classification keeps the strict guards.
    escalation = budget.signature_repeats(new_signature)
    dependency_rung = (classification == "missing_dependency"
                       and 0 < escalation < _VERIFY_SIGNATURE_LADDER)
    if not dependency_rung:
        if _repair_progress_stalled(
                new_count, prior_counts, new_passing, prior_passing):
            return []
        if escalation:
            return []
    spent = budget.spend_repair(
        new_signature, failing_count=new_count, passing_count=new_passing)
    verify_artifact_id = parent.produced_artifact_id
    # A reasoning-class failure (design §8.1) routes to the DIAGNOSE rung
    # instead of a mechanical REPAIR: the gate emits a ``classification``
    # token so the pure router gates by membership test alone. Mechanical
    # tokens keep the fast path straight to REPAIR. The already-spent budget
    # (same attempt + flap accounting the REPAIR path uses) threads verbatim.
    if classification in DIAGNOSE_CLASSIFICATIONS:
        return [_diagnose_node(
            parent, source_key="verify_artifact_id",
            source_id=verify_artifact_id, classification=classification,
            budget=spent, mode=mode)]
    return [TaskNode.new(
        session_id=parent.session_id,
        kind=TaskKind.REPAIR,
        name="Repair failed verification",
        description=("Classify the upstream VERIFY failure and propose a "
                     "targeted patch the shared APPLY executor can write."),
        parent_task_id=parent.task_id,
        inputs={
            "verify_artifact_id": verify_artifact_id,
            "build_artifact_id": parent.inputs.get("build_artifact_id"),
            "apply_artifact_id": parent.inputs.get("apply_artifact_id"),
            "prior_goal": parent.inputs.get("prior_goal"),
            "mode": mode,
            "repair_escalation": escalation,
            **spent.repair_chain_inputs(),
        },
    )]


def _diagnose_node(parent: TaskNode, *, source_key: str, source_id: str,
                   classification: str, budget: LoopBudget,
                   mode: str) -> TaskNode:
    """Spawn the DIAGNOSE reasoning rung for a reasoning-class gate failure.

    A gate failure whose ``classification`` is in
    :data:`~cgx.session.repair.classify.DIAGNOSE_CLASSIFICATIONS` routes
    here instead of straight to a mechanical REPAIR. DIAGNOSE reasons over
    the concrete failure + the real repo + the repair ledger and emits a
    typed verdict the :func:`_diagnose_dispatch_actions` guard maps to a
    scoped successor. ``budget`` is the already-spent :class:`LoopBudget`
    (the gate charged the round on this edge exactly as the REPAIR path
    would), threaded verbatim so the reasoning rung stays under the shared
    :data:`~cgx.session.budget.REPAIR_BUDGET` and carries the ledger id
    forward. ``source_key`` names the report id key DIAGNOSE reads
    (``verify_artifact_id`` / ``runtime_artifact_id`` / ...).
    """
    return TaskNode.new(
        session_id=parent.session_id,
        kind=TaskKind.DIAGNOSE,
        name="Diagnose the gate failure",
        description=("Reason over the concrete gate failure, the real repo, "
                     "and what earlier rounds already tried, then emit one "
                     "minimal-action verdict the pure router dispatches."),
        parent_task_id=parent.task_id,
        inputs={
            source_key: source_id,
            "classification": classification,
            "build_artifact_id": parent.inputs.get("build_artifact_id"),
            "apply_artifact_id": parent.inputs.get("apply_artifact_id"),
            "scaffold_artifact_id": parent.inputs.get("scaffold_artifact_id"),
            "plan_artifact_id": parent.inputs.get("plan_artifact_id"),
            "prior_goal": parent.inputs.get("prior_goal"),
            "mode": mode,
            **budget.repair_chain_inputs(),
        },
    )


def _verify_successors(parent: TaskNode) -> List[TaskNode]:
    """Route a finished VERIFY to RUNTIME_VERIFY, REPAIR, or a terminal.

    Greenfield + a *passing* unit suite hands off to RUNTIME_VERIFY: the
    tests the model wrote are green, but the app itself may still fail to
    boot (an import-time error, a bad ``create_app`` wiring), so a
    runtime gate runs before the session is declared COMPLETED. Every
    other case -- a fixable failure, a skipped/test-free suite, or an
    explore-mode VERIFY -- keeps the existing repair-or-terminal path.
    """
    mode = str(parent.inputs.get("mode") or "").strip()
    outputs = parent.outputs or {}
    outcome = str(outputs.get("outcome") or "").strip()
    if mode == SessionMode.GREENFIELD.value and outcome == "passed":
        return _runtime_verify_node(parent)
    return _verify_to_repair_or_terminal(parent)


def _re_verify_node(parent: TaskNode, *, build_artifact_id: Optional[str],
                    apply_artifact_id: Optional[str]) -> List[TaskNode]:
    """Spawn the incremental RE_VERIFY gate for a scoped fix (design §C2).

    Carries the origin ``reverify_report_id`` (the failing VERIFY_REPORT
    whose recorded test file(s) get re-run) plus the artifact ids the
    executor needs to resolve the venv python and stamp provenance. The
    shared loop budget threads verbatim so a still-failing re-run keeps
    the same attempt/flap accounting the full VERIFY path uses.
    """
    inputs = parent.inputs or {}
    budget = LoopBudget.from_inputs(inputs)
    return [TaskNode.new(
        session_id=parent.session_id,
        kind=TaskKind.RE_VERIFY,
        name="Re-verify the scoped fix",
        description=("Re-run pytest against only the test file(s) the origin "
                     "VERIFY report recorded as failing, skipping the full "
                     "bootstrap -> API-check -> smoke -> verify chain."),
        parent_task_id=parent.task_id,
        inputs={
            "reverify_report_id": inputs.get("reverify_report_id"),
            "reverify_origin_gate": inputs.get("reverify_origin_gate"),
            "build_artifact_id": build_artifact_id,
            "apply_artifact_id": apply_artifact_id,
            "scaffold_artifact_id": inputs.get("scaffold_artifact_id"),
            "plan_artifact_id": inputs.get("plan_artifact_id"),
            "prior_goal": inputs.get("prior_goal"),
            "mode": inputs.get("mode") or SessionMode.GREENFIELD.value,
            **budget.repair_chain_inputs(),
        },
    )]


def _re_verify_successors(parent: TaskNode) -> List[TaskNode]:
    """Dispatch a finished RE_VERIFY exactly like VERIFY (design §C2).

    RE_VERIFY emits a VERIFY_REPORT identical in shape, so the same
    green -> RUNTIME_VERIFY / still-failing -> DIAGNOSE-or-REPAIR edges
    apply unchanged.
    """
    return _verify_successors(parent)


def _runtime_verify_node(parent: TaskNode) -> List[TaskNode]:
    """Spawn the RUNTIME_VERIFY gate carrying the upstream artifact ids."""
    return [TaskNode.new(
        session_id=parent.session_id,
        kind=TaskKind.RUNTIME_VERIFY,
        name="Boot the scaffolded app",
        description=("Import-and-call smoke each entry module under the "
                     "bootstrapped venv to confirm the app actually runs, "
                     "not just that its unit tests pass."),
        parent_task_id=parent.task_id,
        inputs={
            "verify_artifact_id": parent.produced_artifact_id,
            "build_artifact_id": parent.inputs.get("build_artifact_id"),
            "apply_artifact_id": parent.inputs.get("apply_artifact_id"),
            "scaffold_artifact_id": parent.inputs.get("scaffold_artifact_id"),
            "plan_artifact_id": parent.inputs.get("plan_artifact_id"),
            "prior_goal": parent.inputs.get("prior_goal"),
            "mode": SessionMode.GREENFIELD.value,
            # Thread VERIFY's JS coverage signal forward so the terminal
            # fail-closed policy (P2) can see, at RUNTIME_VERIFY completion,
            # whether a scaffolded JS suite was present but never executed --
            # RUNTIME_VERIFY's own outputs do not carry it.
            "js_tests_present": (parent.outputs or {}).get("js_tests_present"),
            "js_tests_ran": (parent.outputs or {}).get("js_tests_ran"),
            # Thread the shared repair budget through the runtime gate so a
            # boot failure can route to REPAIR under the same attempt cap +
            # flap detector as the pre-VERIFY gates (#3).
            **LoopBudget.from_inputs(parent.inputs).repair_chain_inputs(),
        },
    )]


# RUNTIME_REPORT outcomes REPAIR knows how to attempt a fix for: a hard
# boot failure. ``passed`` / ``skipped`` complete the session and never
# reach this edge.
_REPAIRABLE_RUNTIME_OUTCOMES = frozenset({"failed", "timeout", "error"})


def _runtime_verify_to_repair_or_terminal(parent: TaskNode) -> List[TaskNode]:
    """Spawn REPAIR on a boot failure; otherwise terminal (#3).

    A ``passed`` / ``skipped`` RUNTIME_VERIFY spawns no successor and the
    session COMPLETES via :func:`_runtime_verify_terminal_session_actions`.
    A hard boot outcome (``failed`` / ``timeout`` / ``error``) routes to
    REPAIR with the RUNTIME_REPORT as the source artifact, gated by the
    same shared retry budget and failure-signature flap detector used by
    the SMOKE/API_CHECK gates. When the budget is spent or the signature
    flaps the helper declines to spawn REPAIR and the terminal action
    marks the session FAILED. Explore mode never reaches RUNTIME_VERIFY.
    """
    outputs = parent.outputs or {}
    outcome = str(outputs.get("outcome") or "").strip()
    mode = str(parent.inputs.get("mode") or "").strip()
    if mode != SessionMode.GREENFIELD.value:
        return []
    if outcome not in _REPAIRABLE_RUNTIME_OUTCOMES:
        return []
    budget = LoopBudget.from_inputs(parent.inputs)
    if budget.repair_exhausted:
        return []
    new_signature = str(outputs.get("failure_signature") or "").strip()
    if not new_signature:
        failed = outputs.get("failed_count")
        new_signature = f"runtime_failed|count={failed}"
    if budget.seen(new_signature):
        return []
    spent = budget.spend_repair(new_signature)
    # Reasoning-class boot failures route to DIAGNOSE (design §8.1); a
    # mechanical token keeps the fast path straight to REPAIR.
    classification = str(outputs.get("classification") or "").strip()
    if classification in DIAGNOSE_CLASSIFICATIONS:
        return [_diagnose_node(
            parent, source_key="runtime_artifact_id",
            source_id=parent.produced_artifact_id,
            classification=classification, budget=spent, mode=mode)]
    return [TaskNode.new(
        session_id=parent.session_id,
        kind=TaskKind.REPAIR,
        name="Repair failed app boot",
        description=("Classify the upstream RUNTIME_VERIFY boot failure and "
                     "re-author the failing entry module(s) so the app "
                     "imports and starts, not just passes its unit tests."),
        parent_task_id=parent.task_id,
        inputs={
            "runtime_artifact_id": parent.produced_artifact_id,
            "build_artifact_id": parent.inputs.get("build_artifact_id"),
            "apply_artifact_id": parent.inputs.get("apply_artifact_id"),
            "scaffold_artifact_id": parent.inputs.get("scaffold_artifact_id"),
            "prior_goal": parent.inputs.get("prior_goal"),
            "mode": mode,
            **spent.repair_chain_inputs(),
        },
    )]


def _repair_to_apply_or_ask(parent: TaskNode) -> List[TaskNode]:
    """Spawn APPLY when REPAIR produced an applicable patch.

    The empty-diff path (``can_apply`` False) is handled earlier in
    :meth:`Router.on_task_completed` by
    :func:`_repair_terminal_failure_actions`, which marks the session
    terminally ``FAILED`` rather than asking the user to hand-fix
    AI-generated code. This function therefore only ever spawns the
    APPLY successor for an applicable patch; it returns an empty list
    defensively if it is somehow reached with no patch.
    """
    outputs = parent.outputs or {}
    can_apply = bool(outputs.get("can_apply"))
    signature = str(outputs.get("failure_signature") or "")
    budget = LoopBudget.from_inputs(parent.inputs)
    attempt = int(outputs.get("repair_attempt")
                  or budget.repair_attempt or 1)
    budget = budget.with_repair_attempt(attempt).with_signature(signature)
    if not can_apply:
        return []
    return [TaskNode.new(
        session_id=parent.session_id,
        kind=TaskKind.APPLY,
        name="Apply repair patch",
        description=("Write the auto-generated repair diffs to the "
                     "working tree."),
        parent_task_id=parent.task_id,
        inputs={
            "plan_artifact_id": parent.produced_artifact_id,
            "build_artifact_id": parent.inputs.get("build_artifact_id"),
            "scaffold_artifact_id": parent.inputs.get("scaffold_artifact_id"),
            "prior_goal": parent.inputs.get("prior_goal"),
            "mode": parent.inputs.get("mode"),
            # A DIAGNOSE-scoped patch threads its C2 re-verify markers on to
            # APPLY so its successor (:func:`_apply_to_verify`) can splice
            # RE_VERIFY; a mechanical repair leaves these unset (safe no-op).
            "reverify_report_id": parent.inputs.get("reverify_report_id"),
            "reverify_origin_gate": parent.inputs.get("reverify_origin_gate"),
            **budget.repair_chain_inputs(),
        },
    )]


def _coverage_gap(completed: TaskNode) -> Optional[str]:
    """Return a coverage gap that must block a green completion (P2).

    Terminal fail-closed policy. A gate whose ``outcome`` reads
    ``passed`` / ``skipped`` can still hide a blind spot that means the
    delivered app was never actually exercised -- exactly what let
    ses_4cbf963cdc67435a ship "completed" with a broken, untested React
    half. Two such gaps block the green:

    * **JS tests present but unrun** -- a scaffolded JS suite exists on
      disk (``js_tests_present``) yet no JS runner executed a real suite
      (``js_tests_ran`` falsy), so a passing Python half was masking an
      unrun React suite. The flags are read from ``outputs`` (a VERIFY
      terminal's own report) with an ``inputs`` fallback (threaded onto
      RUNTIME_VERIFY by :func:`_runtime_verify_node`, whose own outputs
      do not carry them).
    * **Boot skipped with a server entry present** -- a RUNTIME_VERIFY
      that ``skipped`` while its whole-tree scan still surfaced a bootable
      entry (``entry_files`` non-empty) never actually booted a server the
      tree clearly contains (typically a missing bootstrapped interpreter,
      not an absent app).

    Returns a short reason token, or ``None`` when the terminal is
    honestly green. These are unrecoverable-by-regeneration coverage
    gaps (a missing toolchain / interpreter, not broken source), so the
    caller fails the session closed rather than re-queue a loop that
    would re-hit the identical environmental miss.
    """
    outputs = completed.outputs or {}
    inputs = completed.inputs or {}

    def _flag(key: str) -> bool:
        if key in outputs:
            return bool(outputs.get(key))
        return bool(inputs.get(key))

    if _flag("js_tests_present") and not _flag("js_tests_ran"):
        return "js_tests_present_but_unrun"
    if (completed.kind is TaskKind.RUNTIME_VERIFY
            and str(outputs.get("outcome") or "").strip() == "skipped"
            and outputs.get("entry_files")):
        return "runtime_boot_skipped_with_server_entry"
    return None


def _verify_terminal_session_actions(
        completed: TaskNode) -> List[RouterAction]:
    """Set the session's terminal status for a greenfield VERIFY.

    Called only when a VERIFY finished without spawning a REPAIR
    successor. In greenfield mode a passing (or skipped) suite means the
    write loop delivered working code -> ``COMPLETED``; any other
    terminal outcome (assertions still failing after the repair budget,
    a flapping signature, a collection error with no fixable cause, or
    no tests at all) is a definitive ``FAILED`` -- never a silent
    "success". A terminal that would read green is additionally held to
    the P2 fail-closed policy: an unrun scaffolded JS suite
    (:func:`_coverage_gap`) downgrades it to ``FAILED`` rather than
    reporting a false green. Explore-mode sessions keep their own
    lifecycle, so this returns an empty list for them.
    """
    mode = str(completed.inputs.get("mode") or "").strip()
    if mode != SessionMode.GREENFIELD.value:
        return []
    outputs = completed.outputs or {}
    outcome = str(outputs.get("outcome") or "").strip()
    green = outcome in _VERIFY_SUCCESS_OUTCOMES and not _coverage_gap(completed)
    status = SessionStatus.COMPLETED if green else SessionStatus.FAILED
    return [UpdateSessionStatus(session_id=completed.session_id,
                                status=status)]


# Terminal RUNTIME_VERIFY outcomes that mean the greenfield write loop
# delivered an app that actually boots. ``skipped`` (no detectable entry
# module to boot) counts as success -- it is an explicit no-op, not a
# broken app. Everything else (``failed`` / ``timeout`` / ``error``) is a
# definitive failure.
_RUNTIME_VERIFY_SUCCESS_OUTCOMES = frozenset({"passed", "skipped"})


def _runtime_verify_terminal_session_actions(
        completed: TaskNode) -> List[RouterAction]:
    """Set the greenfield session's terminal status after RUNTIME_VERIFY.

    A booting app (``passed``) -- or a run with no detectable entry to
    boot (``skipped``) -- COMPLETES the session; any hard boot outcome
    (``failed`` / ``timeout`` / ``error``) is a definitive ``FAILED``.
    A terminal that would read green is additionally held to the P2
    fail-closed policy (:func:`_coverage_gap`): an unrun scaffolded JS
    suite (threaded forward from VERIFY) or a boot that ``skipped`` while
    a server entry was present on disk downgrades it to ``FAILED`` rather
    than reporting a false green. Mirrors
    :func:`_verify_terminal_session_actions`; explore mode never reaches
    RUNTIME_VERIFY, so this is a no-op there.
    """
    mode = str(completed.inputs.get("mode") or "").strip()
    if mode != SessionMode.GREENFIELD.value:
        return []
    outputs = completed.outputs or {}
    outcome = str(outputs.get("outcome") or "").strip()
    green = (outcome in _RUNTIME_VERIFY_SUCCESS_OUTCOMES
             and not _coverage_gap(completed))
    status = SessionStatus.COMPLETED if green else SessionStatus.FAILED
    return [UpdateSessionStatus(session_id=completed.session_id,
                                status=status)]


def _preverify_gate_terminal_actions(
        completed: TaskNode) -> List[RouterAction]:
    """Terminate a greenfield run when a pre-VERIFY gate stalls.

    API_CHECK / SMOKE hand off to their successor on ``passed`` /
    ``skipped`` (SMOKE, then VERIFY) and to REPAIR on ``failed`` -- but
    only while the shared repair budget holds and the failure signature
    keeps changing. Once the budget is spent or the signature flaps, the
    gate helper declines to spawn REPAIR and returns no successor. A
    ``failed`` gate with no successor is a genuine dead end (the applied
    files reference symbols that cannot resolve, and repairing them is no
    longer making progress); without an explicit transition the drain
    loop would exit with the session still ``active`` -- idle, with no
    terminal status the UI can settle on. Mirroring
    :func:`_verify_terminal_session_actions`, end the session ``FAILED``
    so the run resolves instead of hanging. The gate task stays ``DONE``
    (it ran fine; its *report* is what failed) but records a concrete
    ``error`` -- which gate, why it could not be repaired, and the failure
    signature -- so the CLI epilogue surfaces the real reason instead of a
    bare "session failed (N done, 0 failed)". A non-``failed`` gate that
    somehow produced no successor is left untouched (empty list) so the
    normal edge is not overridden. Explore-mode keeps its own lifecycle.
    """
    mode = str(completed.inputs.get("mode") or "").strip()
    if mode != SessionMode.GREENFIELD.value:
        return []
    outputs = completed.outputs or {}
    outcome = str(outputs.get("outcome") or "").strip()
    if outcome != "failed":
        return []
    gate = completed.kind.value
    signature = str(outputs.get("failure_signature") or "").strip()
    budget = LoopBudget.from_inputs(completed.inputs)
    why = ("the automated repair budget is exhausted" if budget.repair_exhausted
           else "the same failure keeps recurring (no further progress)")
    reason = f"{gate} failed and could not be repaired: {why}"
    if signature:
        reason = f"{reason} [{signature}]"
    return [
        UpdateTaskStatus(task_id=completed.task_id,
                         status=completed.status, error=reason),
        UpdateSessionStatus(session_id=completed.session_id,
                            status=SessionStatus.FAILED),
    ]


def _scaffold_to_apply(parent: TaskNode) -> List[TaskNode]:
    """Spawn the APPLY follow-up for a finished SCAFFOLD.

    A repair-driven regenerate folds the failed chain's flap ledger
    (``prior_failure_signatures``) into the SCAFFOLD's inputs; thread it
    into the new APPLY -> ... -> VERIFY chain so a regenerated tree that
    reproduces the identical failure is stopped by ``budget.seen()``
    instead of looping. The repair *attempt* counter deliberately does
    not survive (each regenerated tree gets a fresh repair budget,
    bounded by ``repair_regenerate_attempt``).
    """
    inputs: Dict[str, Any] = {
        "scaffold_artifact_id": parent.produced_artifact_id,
        "plan_artifact_id": parent.produced_artifact_id,
        "prior_goal": parent.inputs.get("prior_goal"),
        "mode": SessionMode.GREENFIELD.value,
    }
    signatures = list(LoopBudget.from_inputs(
        parent.inputs).prior_failure_signatures)
    if signatures:
        inputs["prior_failure_signatures"] = signatures
    return [TaskNode.new(
        session_id=parent.session_id,
        kind=TaskKind.APPLY,
        name="Write scaffolded files to disk",
        description=("Apply the generated file contents to the working "
                     "tree."),
        parent_task_id=parent.task_id,
        inputs=inputs,
    )]


# Maps the parent's kind to a function that produces the successor
# tasks. Greenfield kinds (CLARIFY_REQUIREMENTS, DECOMPOSE, SCAFFOLD,
# BOOTSTRAP_ENV) chain to ASK_USER / APPLY / VERIFY respectively. A
# passing greenfield VERIFY hands off to RUNTIME_VERIFY (the app-boot
# gate); a booting RUNTIME_VERIFY is terminal while a boot failure routes
# to REPAIR under the shared budget (#3).
TASK_SUCCESSOR = {
    TaskKind.EXPLORE: _explore_to_ask,
    TaskKind.INVESTIGATE: _investigate_to_recommend,
    TaskKind.RECOMMEND: _recommend_to_ask,
    TaskKind.PLAN_CHANGE: _plan_change_to_ask,
    TaskKind.APPLY: _apply_to_verify,
    TaskKind.CLARIFY_REQUIREMENTS: _clarify_requirements_to_ask,
    TaskKind.DECOMPOSE: _decompose_to_ask,
    TaskKind.SCAFFOLD: _scaffold_to_apply,
    TaskKind.AST_REGENERATE: _scaffold_to_apply,
    TaskKind.BOOTSTRAP_ENV: _bootstrap_to_api_check,
    TaskKind.API_CHECK: _api_check_to_smoke_or_repair,
    TaskKind.SMOKE: _smoke_to_verify_or_repair,
    TaskKind.VERIFY: _verify_successors,
    TaskKind.RE_VERIFY: _re_verify_successors,
    TaskKind.RUNTIME_VERIFY: _runtime_verify_to_repair_or_terminal,
    TaskKind.REPAIR: _repair_to_apply_or_ask,
}


# --------------------- the router ---------------------

# Completion-time overrides, consulted in declaration order before the
# TASK_SUCCESSOR table lookup. Each entry pairs a task kind with a guard
# that returns the pre-empting actions, or ``[]`` to decline so the
# chain falls through to the next guard and finally the normal
# table-driven edge. The guard bodies live in
# :mod:`cgx.session.greenfield_edges`.
_CompletionGuard = Callable[[TaskNode, List[TaskNode]], List[RouterAction]]

_COMPLETION_GUARDS: Tuple[Tuple[TaskKind, _CompletionGuard], ...] = (
    (TaskKind.REPAIR,
     lambda completed, tasks: _repair_install_deps_actions(completed)),
    (TaskKind.REPAIR,
     lambda completed, tasks: _repair_resolve_deps_actions(completed)),
    (TaskKind.REPAIR, _repair_regenerate_actions),
    (TaskKind.REPAIR,
     lambda completed, tasks: _repair_terminal_failure_actions(completed)),
    # DIAGNOSE always dispatches (the escalate arm regenerates or fails
    # terminally), so this guard never declines and no TASK_SUCCESSOR entry
    # is needed -- the reasoning rung is never stranded.
    (TaskKind.DIAGNOSE, _diagnose_dispatch_actions),
    (TaskKind.SCAFFOLD, _scaffold_failed_files_actions),
    (TaskKind.SCAFFOLD, _scaffold_payload_regenerate_actions),
    (TaskKind.SCAFFOLD, _scaffold_contract_regenerate_actions),
    (TaskKind.SCAFFOLD, _scaffold_skill_regenerate_actions),
    (TaskKind.APPLY, _apply_failed_files_actions),
)


class Router:
    """State machine for session-task transitions.

    The router is stateless across calls: it takes a snapshot of the
    session + task list on every invocation. That makes it cheap to
    run inside a request handler and trivial to unit-test.
    """

    @traced("router")
    def on_user_message(self, *, session: Session, message: str,
                        tasks: List[TaskNode]) -> RouterPlan:
        """Decide how to react to a user message.

        Phase 1 contract:
        * No tasks yet -> spawn root EXPLORE.
        * Existing pending ASK_USER -> caller should route the message
          to ``on_decision_recorded`` instead; this method returns an
          empty plan so the caller can detect the case.
        * Otherwise -> spawn a sibling EXPLORE under the current root
          (treats the message as a course-correction objective).
        """
        plan = RouterPlan()
        message = (message or "").strip()
        if not message:
            return plan
        if not tasks:
            plan.actions.append(CreateTask(_make_root(session, message)))
            return plan
        pending_ask = _first_pending_ask(tasks)
        if pending_ask is not None:
            return plan
        plan.actions.append(CreateTask(_make_root(session, message)))
        return plan

    @traced("router")
    def on_task_completed(self, *, session: Session,
                          completed: TaskNode,
                          tasks: List[TaskNode]) -> RouterPlan:
        """Spawn successors for a task that just finished.

        The dispatch is table-driven via :data:`TASK_SUCCESSOR`; a
        missing entry is a no-op (used for terminal kinds like
        ``VERIFY`` or for kinds whose successor lives in a later
        phase).

        The greenfield failure-recovery overrides (the REPAIR verdict
        splices, the dropped-file and contract regenerates for
        SCAFFOLD/APPLY) run *before* the table lookup as an explicit
        guard chain, :data:`_COMPLETION_GUARDS`: guards are consulted
        in declaration order, the first one returning actions pre-empts
        the normal edge, and a guard that declines returns ``[]`` so
        the chain falls through to the next guard and finally the
        successor table.
        """
        plan = RouterPlan()
        for kind, guard in _COMPLETION_GUARDS:
            if completed.kind is not kind:
                continue
            override = guard(completed, tasks)
            if override:
                plan.actions.extend(override)
                return plan
        if completed.kind in (TaskKind.VERIFY, TaskKind.RE_VERIFY):
            plan.actions.extend(_verify_lesson_actions(completed, tasks))
        spawn = TASK_SUCCESSOR.get(completed.kind)
        if spawn is None:
            return plan
        children = spawn(completed)
        for child in children:
            plan.actions.append(CreateTask(child))
        if (completed.kind in (TaskKind.VERIFY, TaskKind.RE_VERIFY)
                and not children):
            plan.actions.extend(
                _verify_terminal_session_actions(completed))
        if completed.kind is TaskKind.RUNTIME_VERIFY and not children:
            plan.actions.extend(
                _runtime_verify_terminal_session_actions(completed))
        if (completed.kind in (TaskKind.API_CHECK, TaskKind.SMOKE)
                and not children):
            plan.actions.extend(
                _preverify_gate_terminal_actions(completed))
        return plan

    @traced("router")
    def on_task_failed(self, *, session: Session,
                       failed: TaskNode,
                       tasks: List[TaskNode],
                       resume_scaffold_artifact_id: Optional[str] = None,
                       retryable: bool = False) -> RouterPlan:
        """Transition a session to terminal ``FAILED`` on a hard failure.

        A *hard* failure is an executor that returned
        ``ExecutorResult.failure`` or crashed: it never produced
        ``outputs``, so the ``outputs``-keyed successor table cannot
        run and :meth:`on_task_completed` is never reached. Without an
        explicit terminal transition the greenfield session would hang
        in ``active`` with a dead FAILED leaf and no successor (e.g. a
        BOOTSTRAP_ENV whose ``pip install`` failed). Greenfield write
        loops must always reach a terminal status, so any unrecoverable
        hard failure ends the session ``FAILED`` -- asking the user to
        hand-fix AI-generated code is never a valid recovery.

        Two recoverable cases:

        * (B4) a SCAFFOLD that crashed or timed out *mid-run* after
          checkpointing some files. When the runner resolves the crashed
          task's incomplete SCAFFOLD_PATCHES checkpoint and threads its
          id via ``resume_scaffold_artifact_id``, and the shared
          regenerate budget is not spent, re-queue a fresh SCAFFOLD that
          resumes from that checkpoint (regenerating only the remainder)
          instead of discarding every completed file.
        * a DECOMPOSE whose executor marked the failure ``retryable``
          (a plan-quality problem such as an empty or unbuildable
          manifest). Bounded by
          :data:`~cgx.session.budget.DECOMPOSE_RETRY_BUDGET`, re-queue
          a fresh DECOMPOSE with the failure message folded into its
          goal so the planner LLM sees the constraint it must satisfy.

        Budget-exhausted or non-recoverable failures fall through to the
        terminal ``FAILED``. Explore-mode sessions keep their user-driven
        lifecycle (the caller may post a follow-up objective), so this
        returns an empty plan for them, and it is a no-op if the session
        is already in a terminal status.
        """
        plan = RouterPlan()
        if session.mode is not SessionMode.GREENFIELD:
            return plan
        if session.status in (SessionStatus.COMPLETED,
                              SessionStatus.FAILED,
                              SessionStatus.ABANDONED):
            return plan
        resume_actions = _scaffold_resume_actions(
            failed, tasks, resume_scaffold_artifact_id)
        if resume_actions:
            plan.actions.extend(resume_actions)
            return plan
        if retryable:
            retry_actions = _decompose_retry_actions(failed)
            if retry_actions:
                plan.actions.extend(retry_actions)
                return plan
        plan.actions.append(UpdateSessionStatus(
            session_id=session.session_id,
            status=SessionStatus.FAILED))
        return plan

    @traced("router")
    def on_decision_recorded(self, *, session: Session,
                             decision: Decision,
                             tasks: List[TaskNode]) -> RouterPlan:
        """Apply a user decision to its blocking ASK_USER and unblock.

        Records the decision, marks the ASK_USER ``DONE``, attaches the
        decision to its consumed list, and spawns the typed successor
        the decision implies (Phase 2: ``CHOOSE_PATH`` -> INVESTIGATE).
        """
        plan = RouterPlan()
        ask = _find(tasks, decision.resolved_task_id)
        if ask is None or ask.kind is not TaskKind.ASK_USER:
            logger.warning("router: decision %s targets non-ask task %s",
                           decision.decision_id, decision.resolved_task_id)
            return plan
        plan.actions.append(RecordDecision(decision))
        plan.actions.append(AttachDecisionToTask(
            task_id=ask.task_id, decision_id=decision.decision_id))
        plan.actions.append(UpdateTaskStatus(
            task_id=ask.task_id, status=TaskNodeStatus.DONE))
        successor = _decision_successor(ask, decision)
        if successor is not None:
            plan.actions.append(CreateTask(successor))
        return plan

    @traced("router")
    def on_budget_exhausted(self, *, session: Session,
                            over_task: TaskNode,
                            tasks: List[TaskNode],
                            reason: str = "") -> RouterPlan:
        """Halt an autonomous loop that hit its per-session budget.

        The runner detects budget exhaustion (task-run count or
        wall-clock) *before* dispatching the next work task and asks the
        router how to stop. An interactive session pauses: every
        still-READY work task is set ``BLOCKED`` so the drain loop cannot
        re-pick it, a fresh ``ASK_USER`` surfaces the exhaustion, and the
        session goes ``PAUSED``. A ``headless`` session has no user to
        prompt, so the loop ends terminally ``FAILED`` with the READY
        work abandoned -- never silently looping past its budget.
        """
        plan = RouterPlan()
        ready_work = [t for t in tasks
                      if t.status is TaskNodeStatus.READY
                      and t.kind is not TaskKind.ASK_USER]
        if session.headless:
            for t in ready_work:
                plan.actions.append(UpdateTaskStatus(
                    task_id=t.task_id, status=TaskNodeStatus.ABANDONED,
                    error=f"session budget exhausted: {reason}"))
            plan.actions.append(UpdateSessionStatus(
                session_id=session.session_id, status=SessionStatus.FAILED))
            return plan
        for t in ready_work:
            plan.actions.append(UpdateTaskStatus(
                task_id=t.task_id, status=TaskNodeStatus.BLOCKED))
        plan.actions.append(CreateTask(_make_budget_ask(over_task, reason)))
        plan.actions.append(UpdateSessionStatus(
            session_id=session.session_id, status=SessionStatus.PAUSED))
        return plan


# --------------------- helpers ---------------------


def _make_root(session: Session, message: str) -> TaskNode:
    """Pick the root task kind based on the session's mode."""
    if session.mode is SessionMode.GREENFIELD:
        return _make_root_clarify(session, message)
    return _make_root_explore(session, message)


def _make_root_explore(session: Session, message: str) -> TaskNode:
    return TaskNode.new(
        session_id=session.session_id,
        kind=TaskKind.EXPLORE,
        name="Explore directions",
        description=("Survey the codebase for directions that bear on "
                     "the user's objective."),
        inputs={"goal": message,
                "original_objective": session.original_objective,
                "project_root": session.project_root},
    )


def _make_root_clarify(session: Session, message: str) -> TaskNode:
    return TaskNode.new(
        session_id=session.session_id,
        kind=TaskKind.CLARIFY_REQUIREMENTS,
        name="Clarify project requirements",
        description=("Surface clarifying questions about the desired "
                     "tech stack, scope, and target environment."),
        inputs={"goal": message,
                "original_objective": session.original_objective,
                "project_root": session.project_root},
    )


def _first_pending_ask(tasks: List[TaskNode]) -> Optional[TaskNode]:
    for t in tasks:
        if (t.kind is TaskKind.ASK_USER and
                t.status in (TaskNodeStatus.READY,
                             TaskNodeStatus.BLOCKED,
                             TaskNodeStatus.PENDING,
                             TaskNodeStatus.IN_PROGRESS)):
            return t
    return None


def _find(tasks: List[TaskNode], task_id: str) -> Optional[TaskNode]:
    for t in tasks:
        if t.task_id == task_id:
            return t
    return None


def _decision_successor(ask: TaskNode,
                        decision: Decision) -> Optional[TaskNode]:
    """Return the task a decision implies, or ``None`` for noop kinds.

    * ``CHOOSE_PATH`` -> ``INVESTIGATE`` anchored on the chosen chunk_id.
    * ``CHOOSE_RECOMMENDATION`` -> dispatched by the recommendation
      ``kind`` token: ``investigate_more`` reopens the investigate loop,
      ``plan_change`` enters the write loop, ``ask_followup`` spawns a
      freeform follow-up, ``done`` closes the focus (no successor).
    * ``APPROVE`` -> ``APPLY`` when ``approved`` is true; ``None`` on a
      decline so the user can pivot via a fresh objective.
    * ``FREEFORM`` -> ``None`` (handled as a new user message by the
      caller).
    """
    if decision.kind is DecisionKind.CHOOSE_PATH:
        anchor = str(decision.chosen.get("anchor_chunk_id") or "").strip()
        if not anchor:
            return None
        return TaskNode.new(
            session_id=ask.session_id,
            kind=TaskKind.INVESTIGATE,
            name="Investigate selected direction",
            description=("Deeper retrieval anchored on the chosen "
                         "direction's chunk_id."),
            parent_task_id=ask.task_id,
            inputs={
                "anchor_chunk_id": anchor,
                "title": decision.chosen.get("title"),
                "rationale": decision.chosen.get("rationale"),
                "prior_goal": ask.inputs.get("prior_goal"),
                "directions_artifact_id":
                    ask.inputs.get("directions_artifact_id"),
                "decision_id": decision.decision_id,
            },
        )
    if decision.kind is DecisionKind.CHOOSE_RECOMMENDATION:
        return _from_choose_recommendation(ask, decision)
    if decision.kind is DecisionKind.APPROVE:
        return _from_approve(ask, decision)
    if decision.kind is DecisionKind.CLARIFY_ANSWERS:
        return _from_clarify_answers(ask, decision)
    if decision.kind is DecisionKind.APPROVE_PLAN:
        return _from_approve_plan(ask, decision)
    return None


def _from_choose_recommendation(ask: TaskNode,
                                decision: Decision) -> Optional[TaskNode]:
    rec_kind = str(decision.chosen.get("kind") or "").strip()
    title = decision.chosen.get("title")
    rationale = decision.chosen.get("rationale")
    prior_goal = ask.inputs.get("prior_goal")
    if rec_kind == "investigate_more":
        anchor = str(decision.chosen.get("anchor_chunk_id") or "").strip()
        if not anchor:
            return None
        return TaskNode.new(
            session_id=ask.session_id,
            kind=TaskKind.INVESTIGATE,
            name="Investigate further",
            description=("Follow-up investigation anchored on the "
                         "recommended chunk."),
            parent_task_id=ask.task_id,
            inputs={
                "anchor_chunk_id": anchor,
                "title": title,
                "rationale": rationale,
                "prior_goal": prior_goal,
                "decision_id": decision.decision_id,
            },
        )
    if rec_kind == "plan_change":
        return TaskNode.new(
            session_id=ask.session_id,
            kind=TaskKind.PLAN_CHANGE,
            name=str(title or "Plan code change"),
            description=("Propose a concrete code-change plan + diffs "
                         "for the chosen recommendation."),
            parent_task_id=ask.task_id,
            inputs={
                "prior_goal": prior_goal,
                "recommendation": dict(decision.chosen),
                "anchor_chunk_id": decision.chosen.get("anchor_chunk_id"),
                "findings_artifact_id":
                    ask.inputs.get("findings_artifact_id"),
                "recommendations_artifact_id":
                    ask.inputs.get("recommendations_artifact_id"),
                "decision_id": decision.decision_id,
            },
        )
    if rec_kind == "ask_followup":
        return TaskNode.new(
            session_id=ask.session_id,
            kind=TaskKind.ASK_USER,
            name=str(title or "Follow-up question"),
            description=str(rationale or title
                            or "Provide additional input."),
            parent_task_id=ask.task_id,
            inputs={
                "expected_kind": DecisionKind.FREEFORM.value,
                "prior_goal": prior_goal,
                "from_recommendation": dict(decision.chosen),
                "decision_id": decision.decision_id,
            },
        )
    # ``done`` (and any unknown token) -> no successor; the session
    # focus closes here and the caller can post a fresh message to
    # start a new direction.
    return None


def _from_approve(ask: TaskNode,
                  decision: Decision) -> Optional[TaskNode]:
    approved = bool(decision.chosen.get("approved"))
    if not approved:
        return None
    plan_artifact_id = str(ask.inputs.get("plan_artifact_id") or "").strip()
    if not plan_artifact_id:
        return None
    return TaskNode.new(
        session_id=ask.session_id,
        kind=TaskKind.APPLY,
        name="Apply change plan",
        description="Write the approved diffs to the working tree.",
        parent_task_id=ask.task_id,
        inputs={
            "plan_artifact_id": plan_artifact_id,
            "prior_goal": ask.inputs.get("prior_goal"),
            "decision_id": decision.decision_id,
        },
    )


def _from_clarify_answers(ask: TaskNode,
                          decision: Decision) -> Optional[TaskNode]:
    """Spawn DECOMPOSE once the user has answered the clarifying questions."""
    answers = decision.chosen.get("answers")
    if not isinstance(answers, dict) or not answers:
        return None
    return TaskNode.new(
        session_id=ask.session_id,
        kind=TaskKind.DECOMPOSE,
        name="Decompose into a work plan",
        description=("Turn the user's answers into a structured file "
                     "manifest the scaffold step can iterate."),
        parent_task_id=ask.task_id,
        inputs={
            "prior_goal": ask.inputs.get("prior_goal"),
            "requirements_artifact_id":
                ask.inputs.get("requirements_artifact_id"),
            "answers": dict(answers),
            "decision_id": decision.decision_id,
        },
    )


def _from_approve_plan(ask: TaskNode,
                       decision: Decision) -> Optional[TaskNode]:
    """Spawn SCAFFOLD when the user approves the work plan.

    Carries the flap ledger and the spent ``regenerate_attempt`` count a
    re-plan folded into the approval gate, so :func:`_scaffold_to_apply`
    threads them down the new chain: a revised manifest that reproduces an
    already-seen failure is stopped by ``budget.seen()`` instead of
    spending a fresh repair budget, and the syntax-churn regenerate budget
    stays a per-session ceiling rather than resetting to zero per manifest.
    """
    if not bool(decision.chosen.get("approved")):
        return None
    work_plan_artifact_id = str(
        ask.inputs.get("work_plan_artifact_id") or "").strip()
    if not work_plan_artifact_id:
        return None
    inputs: Dict[str, Any] = {
        "work_plan_artifact_id": work_plan_artifact_id,
        "requirements_artifact_id":
            ask.inputs.get("requirements_artifact_id"),
        "prior_goal": ask.inputs.get("prior_goal"),
        "replan_attempt": ask.inputs.get("replan_attempt"),
        "regenerate_attempt": ask.inputs.get("regenerate_attempt"),
        "decision_id": decision.decision_id,
    }
    signatures = list(LoopBudget.from_inputs(
        ask.inputs).prior_failure_signatures)
    if signatures:
        inputs["prior_failure_signatures"] = signatures
    return TaskNode.new(
        session_id=ask.session_id,
        kind=TaskKind.SCAFFOLD,
        name="Generate scaffolded files",
        description=("Generate the content for each file in the work "
                     "plan, layer by layer."),
        parent_task_id=ask.task_id,
        inputs=inputs,
    )
