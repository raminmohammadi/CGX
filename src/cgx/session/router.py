

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
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Union

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


# --------------------- typed router actions ---------------------

@dataclass
class CreateTask:
    """Persist ``task`` as a new node in the tree."""
    task: TaskNode


@dataclass
class UpdateTaskStatus:
    """Transition ``task_id`` to ``status``; optionally clear blockers.

    ``error`` is recorded on the task when the transition is to
    ``FAILED`` so the UI can surface why the node went red (e.g. a
    REPAIR that could not produce a patch).
    """
    task_id: str
    status: TaskNodeStatus
    clear_blockers: bool = False
    error: Optional[str] = None


@dataclass
class UpdateSessionStatus:
    """Transition a session to a terminal (or paused) lifecycle status.

    Emitted by the router when a greenfield write loop reaches a
    definitive end: ``COMPLETED`` when VERIFY passes, ``FAILED`` when
    verification fails and no automated recovery (patch / regenerate /
    install-deps) remains. Asking the user to hand-fix AI-generated
    code is never a valid recovery, so exhaustion is terminal, not a
    prompt.
    """
    session_id: str
    status: SessionStatus


@dataclass
class RecordDecision:
    """Persist ``decision`` against the decision log."""
    decision: Decision


@dataclass
class AttachDecisionToTask:
    """Append ``decision_id`` to ``task_id``'s ``consumed_decision_ids``."""
    task_id: str
    decision_id: str


@dataclass
class RecordLesson:
    """Persist a successful REPAIR -> VERIFY-pass pair as a cross-session lesson.

    The router emits this when a VERIFY succeeds with a REPAIR on the
    ancestor chain (Phase 7.1). The runner resolves the artifacts and
    writes via :func:`cgx.session.lessons.record_lesson`.
    """
    verify_task_id: str
    repair_task_id: str
    scaffold_task_id: Optional[str] = None


RouterAction = Union[CreateTask, UpdateTaskStatus, UpdateSessionStatus,
                     RecordDecision, AttachDecisionToTask, RecordLesson]


@dataclass
class RouterPlan:
    """The list of state changes the router wants the caller to apply.

    The caller is responsible for ordering writes (the actions are
    already topologically ordered by construction: creates before
    updates, decisions before attaches).
    """
    actions: List[RouterAction] = field(default_factory=list)

    def __iter__(self):
        return iter(self.actions)

    def __len__(self) -> int:
        return len(self.actions)

    def extend(self, more: Iterable[RouterAction]) -> None:
        self.actions.extend(more)


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
    """
    mode = str(parent.inputs.get("mode") or "").strip()
    has_build_artifact = bool(
        str(parent.inputs.get("build_artifact_id") or "").strip())
    repair_attempt = int(parent.inputs.get("repair_attempt") or 0)
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
            "repair_attempt": repair_attempt,
            "prior_failure_signatures":
                list(parent.inputs.get("prior_failure_signatures") or []),
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
    """
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
            "prior_failure_signatures":
                list(parent.inputs.get("prior_failure_signatures") or []),
            "repair_attempt": int(parent.inputs.get("repair_attempt") or 0),
        },
    )]


# Outcomes that REPAIR knows how to attempt a fix for on an API_CHECK
# report. Only ``failed`` is repairable; ``passed`` and ``skipped``
# chain to SMOKE.
_REPAIRABLE_API_CHECK_OUTCOMES = frozenset({"failed"})


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
                "prior_failure_signatures":
                    list(parent.inputs.get("prior_failure_signatures")
                         or []),
                "repair_attempt":
                    int(parent.inputs.get("repair_attempt") or 0),
            },
        )]
    if mode != SessionMode.GREENFIELD.value:
        return []
    repair_attempt = int(parent.inputs.get("repair_attempt") or 0)
    if repair_attempt >= _REPAIR_BUDGET:
        return []
    new_signature = str(outputs.get("failure_signature") or "").strip()
    if not new_signature:
        failed = outputs.get("failed_count")
        new_signature = f"api_check_failed|count={failed}"
    prior = list(parent.inputs.get("prior_failure_signatures") or [])
    if new_signature in prior:
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
            "repair_attempt": repair_attempt + 1,
            "prior_failure_signatures": prior + [new_signature],
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
            },
        )]
    if mode != SessionMode.GREENFIELD.value:
        return []
    repair_attempt = int(parent.inputs.get("repair_attempt") or 0)
    if repair_attempt >= _REPAIR_BUDGET:
        return []
    new_signature = str(outputs.get("failure_signature") or "").strip()
    if not new_signature:
        failed = outputs.get("failed_count")
        new_signature = f"smoke_failed|count={failed}"
    prior = list(parent.inputs.get("prior_failure_signatures") or [])
    if new_signature in prior:
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
            "repair_attempt": repair_attempt + 1,
            "prior_failure_signatures": prior + [new_signature],
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
    """Spawn the ASK_USER(APPROVE_PLAN) gate for a finished DECOMPOSE."""
    return [TaskNode.new(
        session_id=parent.session_id,
        kind=TaskKind.ASK_USER,
        name="Approve work plan",
        description=("Review the proposed file manifest and approve to "
                     "begin scaffolding."),
        parent_task_id=parent.task_id,
        inputs={
            "expected_kind": DecisionKind.APPROVE_PLAN.value,
            "work_plan_artifact_id": parent.produced_artifact_id,
            "prior_goal": parent.inputs.get("prior_goal"),
            "requirements_artifact_id":
                parent.inputs.get("requirements_artifact_id"),
            "replan_attempt": parent.inputs.get("replan_attempt"),
        },
    )]


_REPAIR_BUDGET = 2

# Maximum number of targeted regenerate attempts per SCAFFOLD ancestor
# chain. Each attempt re-generates only the files that dropped, seeding
# the survivors from the prior checkpoint, so a retry is fast and does
# not disturb good work. A local/flaky model routinely drops a single
# file to a read timeout or an empty patch, so a shallow budget escalated
# straight to a disruptive re-plan (and re-approval); a few in-place
# retries clear the common case before the manifest is ever blamed.
_REGENERATE_BUDGET = 3

# Maximum number of *re-plan* escalations per session. When a SCAFFOLD or
# APPLY spends its per-manifest regenerate budget the manifest itself is
# the suspect (not the generation of any single file), so the router
# escalates once to a fresh DECOMPOSE that revises the plan with the
# accumulated failure folded into its goal. When the re-plan budget is
# also spent the router does NOT discard the run: as long as the partial
# scaffold produced survivors it proceeds along the normal edge (APPLY
# writes them, VERIFY judges them) rather than failing terminally and
# throwing away every successfully generated file. Only a scaffold that
# produced nothing usable is a genuine dead end.
_REPLAN_BUDGET = 1


# Outcomes that REPAIR knows how to attempt a fix for. ``passed`` and
# ``skipped`` are terminal -- they're not failures. ``pytest_missing``
# is BOOTSTRAP_ENV's job, not REPAIR's. ``no_tests_collected`` is
# repairable only when test files were actually selected but pytest
# collected zero tests (malformed tests -- see
# :func:`_verify_to_repair_or_terminal`); a genuinely test-free project
# still terminates cleanly. ``failed`` is a non-pytest runner (e.g. an
# ``npm`` build/test) that exited non-zero: it classifies as ``unknown``
# in the repair classifier, which routes to a re-scaffold (regenerate)
# so a JS/TS build break is not a silent false success.
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
# an explicit opt-out, not a broken suite.
_VERIFY_SUCCESS_OUTCOMES = frozenset({"passed", "skipped"})


def _verify_to_repair_or_terminal(parent: TaskNode) -> List[TaskNode]:
    """Spawn REPAIR after a fixable VERIFY failure; otherwise terminal.

    Triggers only in greenfield mode (auto-apply is part of the
    greenfield contract; explore-mode write loops keep their existing
    approval gates). The progress detector reads
    ``prior_failure_signatures`` off the parent: if the just-finished
    VERIFY's signature already appears in the list, the loop is
    flapping and we refuse to spawn another REPAIR.

    The retry budget is :data:`_REPAIR_BUDGET` attempts. The attempt
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
    repair_attempt = int(parent.inputs.get("repair_attempt") or 0)
    if repair_attempt >= _REPAIR_BUDGET:
        return []
    # Read the VERIFY_REPORT's failure_signature lazily by deferring to
    # the classifier; the router stays free of I/O by using a precomputed
    # signature stashed by the runner-style ``outputs``. Falls back to a
    # ``returncode``+ ``outcome`` composite so a missing signature still
    # gives the progress detector a stable token to compare.
    new_signature = str(outputs.get("failure_signature") or "").strip()
    if not new_signature:
        new_signature = f"{outcome}|rc={outputs.get('returncode')}"
    prior = list(parent.inputs.get("prior_failure_signatures") or [])
    if new_signature in prior:
        return []
    verify_artifact_id = parent.produced_artifact_id
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
            "repair_attempt": repair_attempt + 1,
            "prior_failure_signatures": prior + [new_signature],
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
    attempt = int(outputs.get("repair_attempt")
                  or parent.inputs.get("repair_attempt") or 1)
    prior = list(parent.inputs.get("prior_failure_signatures") or [])
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
            "prior_goal": parent.inputs.get("prior_goal"),
            "mode": parent.inputs.get("mode"),
            "repair_attempt": attempt,
            "prior_failure_signatures": (
                prior if signature in prior else prior + [signature]),
        },
    )]


def _repair_terminal_failure_actions(
        completed: TaskNode) -> List[RouterAction]:
    """Fail the session when REPAIR has no automated recovery left.

    Reached from :meth:`Router.on_task_completed` only after the
    install-deps and regenerate branches have both declined. A REPAIR
    that produced no applicable patch (``can_apply`` False) means every
    automated path -- patch, regenerate, dependency install -- is
    exhausted. Asking the user to hand-edit AI-generated code is never a
    valid recovery, so the loop terminates: the REPAIR node goes
    ``FAILED`` (carrying the classification for the UI) and the whole
    session flips to ``FAILED``. Returns an empty list when the patch is
    applicable so the caller falls through to the APPLY successor.
    """
    outputs = completed.outputs or {}
    if bool(outputs.get("can_apply")):
        return []
    classification = str(outputs.get("classification") or "unknown")
    error = ("Automated repair could not produce a patch "
             f"(classification={classification}); no regenerate or "
             "dependency-install path remained.")
    return [
        UpdateTaskStatus(task_id=completed.task_id,
                         status=TaskNodeStatus.FAILED, error=error),
        UpdateSessionStatus(session_id=completed.session_id,
                            status=SessionStatus.FAILED),
    ]


def _verify_terminal_session_actions(
        completed: TaskNode) -> List[RouterAction]:
    """Set the session's terminal status for a greenfield VERIFY.

    Called only when a VERIFY finished without spawning a REPAIR
    successor. In greenfield mode a passing (or skipped) suite means the
    write loop delivered working code -> ``COMPLETED``; any other
    terminal outcome (assertions still failing after the repair budget,
    a flapping signature, a collection error with no fixable cause, or
    no tests at all) is a definitive ``FAILED`` -- never a silent
    "success". Explore-mode sessions keep their own lifecycle, so this
    returns an empty list for them.
    """
    mode = str(completed.inputs.get("mode") or "").strip()
    if mode != SessionMode.GREENFIELD.value:
        return []
    outputs = completed.outputs or {}
    outcome = str(outputs.get("outcome") or "").strip()
    status = (SessionStatus.COMPLETED
              if outcome in _VERIFY_SUCCESS_OUTCOMES
              else SessionStatus.FAILED)
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
    so the run resolves instead of hanging. A non-``failed`` gate that
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
    return [UpdateSessionStatus(session_id=completed.session_id,
                                status=SessionStatus.FAILED)]


def _scaffold_to_apply(parent: TaskNode) -> List[TaskNode]:
    """Spawn the APPLY follow-up for a finished SCAFFOLD."""
    return [TaskNode.new(
        session_id=parent.session_id,
        kind=TaskKind.APPLY,
        name="Write scaffolded files to disk",
        description=("Apply the generated file contents to the working "
                     "tree."),
        parent_task_id=parent.task_id,
        inputs={
            "scaffold_artifact_id": parent.produced_artifact_id,
            "plan_artifact_id": parent.produced_artifact_id,
            "prior_goal": parent.inputs.get("prior_goal"),
            "mode": SessionMode.GREENFIELD.value,
        },
    )]


# Maps the parent's kind to a function that produces the successor
# tasks. Greenfield kinds (CLARIFY_REQUIREMENTS, DECOMPOSE, SCAFFOLD,
# BOOTSTRAP_ENV) chain to ASK_USER / APPLY / VERIFY respectively.
# VERIFY is terminal.
TASK_SUCCESSOR = {
    TaskKind.EXPLORE: _explore_to_ask,
    TaskKind.INVESTIGATE: _investigate_to_recommend,
    TaskKind.RECOMMEND: _recommend_to_ask,
    TaskKind.PLAN_CHANGE: _plan_change_to_ask,
    TaskKind.APPLY: _apply_to_verify,
    TaskKind.CLARIFY_REQUIREMENTS: _clarify_requirements_to_ask,
    TaskKind.DECOMPOSE: _decompose_to_ask,
    TaskKind.SCAFFOLD: _scaffold_to_apply,
    TaskKind.BOOTSTRAP_ENV: _bootstrap_to_api_check,
    TaskKind.API_CHECK: _api_check_to_smoke_or_repair,
    TaskKind.SMOKE: _smoke_to_verify_or_repair,
    TaskKind.VERIFY: _verify_to_repair_or_terminal,
    TaskKind.REPAIR: _repair_to_apply_or_ask,
}


# --------------------- the router ---------------------

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

        Phase 6.1 splices a special branch *before* the table lookup:
        when a finished REPAIR carries ``outputs.strategy=='regenerate'``
        and a SCAFFOLD ancestor exists within budget, the router walks
        the chain, marks the abandoned subtree, and re-queues a fresh
        SCAFFOLD instead of taking the patch path.

        A REPAIR carrying ``outputs.strategy=='install_deps'`` (a
        missing-dependency verdict from API_CHECK) is spliced the same
        way: the router re-queues BOOTSTRAP_ENV so preflight installs
        the absent package(s) and API_CHECK re-probes, rather than
        regenerating code that references valid APIs.

        A finished greenfield APPLY that dropped any invalid-syntax file
        (``outputs.failed_count > 0``) is likewise spliced before the
        table: proceeding with a missing module guarantees a collection
        error, so the router re-scaffolds within budget (or ends the
        session terminally ``FAILED`` when it cannot).

        A finished SCAFFOLD that dropped any file (``outputs.failed_count
        > 0`` from an LLM timeout or empty patch) is spliced the same way
        so the tree is never applied with a required file simply absent.
        """
        plan = RouterPlan()
        if completed.kind is TaskKind.REPAIR:
            install_actions = _repair_install_deps_actions(completed)
            if install_actions:
                plan.actions.extend(install_actions)
                return plan
            regen_actions = _repair_regenerate_actions(completed, tasks)
            if regen_actions:
                plan.actions.extend(regen_actions)
                return plan
            fail_actions = _repair_terminal_failure_actions(completed)
            if fail_actions:
                plan.actions.extend(fail_actions)
                return plan
        if completed.kind is TaskKind.SCAFFOLD:
            dropped_actions = _scaffold_failed_files_actions(completed, tasks)
            if dropped_actions:
                plan.actions.extend(dropped_actions)
                return plan
        if completed.kind is TaskKind.APPLY:
            dropped_actions = _apply_failed_files_actions(completed, tasks)
            if dropped_actions:
                plan.actions.extend(dropped_actions)
                return plan
        if completed.kind is TaskKind.VERIFY:
            plan.actions.extend(_verify_lesson_actions(completed, tasks))
        spawn = TASK_SUCCESSOR.get(completed.kind)
        if spawn is None:
            return plan
        children = spawn(completed)
        for child in children:
            plan.actions.append(CreateTask(child))
        if completed.kind is TaskKind.VERIFY and not children:
            plan.actions.extend(
                _verify_terminal_session_actions(completed))
        if (completed.kind in (TaskKind.API_CHECK, TaskKind.SMOKE)
                and not children):
            plan.actions.extend(
                _preverify_gate_terminal_actions(completed))
        return plan

    @traced("router")
    def on_task_failed(self, *, session: Session,
                       failed: TaskNode,
                       tasks: List[TaskNode],
                       resume_scaffold_artifact_id: Optional[str] = None
                       ) -> RouterPlan:
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

        One recoverable case (B4): a SCAFFOLD that crashed or timed out
        *mid-run* after checkpointing some files. When the runner resolves
        the crashed task's incomplete SCAFFOLD_PATCHES checkpoint and
        threads its id via ``resume_scaffold_artifact_id``, and the shared
        regenerate budget is not spent, re-queue a fresh SCAFFOLD that
        resumes from that checkpoint (regenerating only the remainder)
        instead of discarding every completed file. Budget-exhausted or
        checkpoint-less crashes fall through to the terminal ``FAILED``.

        Explore-mode sessions keep their user-driven lifecycle (the
        caller may post a follow-up objective), so this returns an empty
        plan for them, and it is a no-op if the session is already in a
        terminal status.
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


def _make_budget_ask(over_task: TaskNode, reason: str) -> TaskNode:
    """Build the ASK_USER that surfaces a paused-on-budget session."""
    prior_goal = (over_task.inputs.get("prior_goal")
                  or over_task.inputs.get("goal"))
    return TaskNode.new(
        session_id=over_task.session_id,
        kind=TaskKind.ASK_USER,
        name="Session budget exhausted",
        description=(f"The session hit its {reason} before finishing. "
                     "Autonomous work is paused -- review the progress so "
                     "far and decide whether to continue or stop."),
        parent_task_id=over_task.parent_task_id,
        inputs={
            "expected_kind": DecisionKind.FREEFORM.value,
            "reason": reason,
            "prior_goal": prior_goal,
            "over_task_id": over_task.task_id,
        },
    )

def _repair_install_deps_actions(
        completed: TaskNode) -> List[RouterAction]:
    """Return the router actions that execute an install-deps verdict.

    An ``install_deps`` verdict (set by the REPAIR executor for an
    API_CHECK ``missing_dependency`` failure) tells the router to
    re-provision the environment rather than rewrite code: it re-queues
    a BOOTSTRAP_ENV whose preflight installs the absent third-party
    imports and syncs requirements.txt. BOOTSTRAP_ENV's own successor
    (:func:`_bootstrap_to_api_check`) then re-probes the same symbols,
    so a successful install flows straight back into SMOKE/VERIFY while
    the shared ``repair_attempt`` + ``prior_failure_signatures`` budget
    on API_CHECK prevents an install loop. Returns an empty list for
    any other strategy so the dispatcher falls through to the regenerate
    / patch / ASK_USER paths.
    """
    outputs = completed.outputs or {}
    strategy = str(outputs.get("strategy") or "").strip()
    if strategy != "install_deps":
        return []
    inputs = completed.inputs or {}
    repair_attempt = int(outputs.get("repair_attempt")
                         or inputs.get("repair_attempt") or 1)
    prior = list(inputs.get("prior_failure_signatures") or [])
    missing = [str(m) for m in outputs.get("missing_modules") or []]
    boot = TaskNode.new(
        session_id=completed.session_id,
        kind=TaskKind.BOOTSTRAP_ENV,
        name="Install missing dependencies",
        description=("Re-provision the project venv to install the "
                     "third-party package(s) the applied files import "
                     "but that are absent from the environment, then "
                     "re-probe via API_CHECK."),
        parent_task_id=completed.task_id,
        inputs={
            "apply_artifact_id": inputs.get("apply_artifact_id"),
            "plan_artifact_id": inputs.get("plan_artifact_id"),
            "scaffold_artifact_id": inputs.get("scaffold_artifact_id"),
            "prior_goal": inputs.get("prior_goal"),
            "mode": inputs.get("mode") or SessionMode.GREENFIELD.value,
            "missing_modules": missing,
            "repair_attempt": repair_attempt,
            "prior_failure_signatures": prior,
        },
    )
    return [CreateTask(boot)]


def _repair_regenerate_actions(completed: TaskNode,
                               tasks: List[TaskNode]) -> List[RouterAction]:
    """Return the router actions that execute a regenerate verdict.

    A regenerate verdict (set by the REPAIR executor when patching is
    impossible or too large to be safe) tells the router to abandon
    the failing subtree under the nearest SCAFFOLD ancestor and
    re-queue a fresh SCAFFOLD with the constraint payload folded into
    its inputs. The dispatcher in :meth:`Router.on_task_completed`
    falls back to the regular patch / ASK_USER table-driven path when
    this function returns an empty list, so the four early-exit cases
    below (wrong strategy, no SCAFFOLD ancestor, budget exhausted, or
    nothing to abandon) degrade gracefully.
    """
    from cgx.session.repair.propose import propose_regenerate  # local import: dep direction

    outputs = completed.outputs or {}
    strategy = str(outputs.get("strategy") or "").strip()
    if strategy != "regenerate":
        return []
    extra_constraints = outputs.get("extra_constraints")
    if not isinstance(extra_constraints, dict):
        extra_constraints = {}
    scaffold = _find_scaffold_ancestor(completed, tasks)
    if scaffold is None:
        return []
    prior_regens = int(scaffold.inputs.get("regenerate_attempt") or 0)
    if prior_regens >= _REGENERATE_BUDGET:
        return []
    abandon_targets = _collect_descendants(scaffold.task_id, tasks)
    actions: List[RouterAction] = []
    skip_states = {TaskNodeStatus.DONE, TaskNodeStatus.FAILED,
                   TaskNodeStatus.ABANDONED}
    for t in abandon_targets:
        if t.status in skip_states:
            continue
        actions.append(UpdateTaskStatus(
            task_id=t.task_id, status=TaskNodeStatus.ABANDONED))
    new_scaffold = propose_regenerate(scaffold, extra_constraints)
    actions.append(CreateTask(new_scaffold))
    return actions


def _apply_failed_files_actions(completed: TaskNode,
                                tasks: List[TaskNode]) -> List[RouterAction]:
    """Regenerate (or terminally fail) a greenfield APPLY that dropped files.

    The APPLY executor refuses to write a file whose source does not
    parse as valid Python, recording it under ``failed_files`` and
    surfacing a non-zero ``failed_count`` while still applying the rest.
    Continuing to BOOTSTRAP_ENV / VERIFY with a core module silently
    missing guarantees a downstream collection error, so any greenfield
    APPLY that dropped a file re-scaffolds within
    :data:`_REGENERATE_BUDGET` instead of limping forward. When the
    regenerate budget is spent the router escalates once to a revised
    manifest via :func:`_replan_or_fail` (a fresh DECOMPOSE); when the
    re-plan budget is also spent that helper proceeds with the survivors
    rather than discarding the run, and only fails terminally when nothing
    usable was generated. When no SCAFFOLD ancestor exists the session
    fails terminally -- it cannot re-scaffold a tree it cannot find.
    Returns an empty list for explore mode or a clean apply so the
    dispatcher takes the normal APPLY -> VERIFY edge.
    """
    from cgx.session.repair.propose import propose_regenerate  # dep direction

    outputs = completed.outputs or {}
    mode = str(completed.inputs.get("mode") or "").strip()
    if mode != SessionMode.GREENFIELD.value:
        return []
    failed_count = int(outputs.get("failed_count") or 0)
    if failed_count <= 0:
        return []
    scaffold = _find_scaffold_ancestor(completed, tasks)
    if scaffold is None:
        return [UpdateSessionStatus(
            session_id=completed.session_id, status=SessionStatus.FAILED)]
    scaffold_outputs = scaffold.outputs or {}
    constraint = _invalid_scaffold_constraint(
        failed_count,
        apply_failed=outputs.get("failed_files"),
        scaffold_failed=scaffold_outputs.get("failed"))
    prior_regens = int(scaffold.inputs.get("regenerate_attempt") or 0)
    if prior_regens >= _REGENERATE_BUDGET:
        return _replan_or_fail(
            completed, tasks, scaffold=scaffold,
            failure_note=str(constraint.get("rationale") or ""))
    regen_files = _failed_scaffold_paths(
        scaffold_outputs.get("failed"), outputs.get("failed_files"))
    prior_id = str(
        scaffold_outputs.get("scaffold_artifact_id") or "").strip()
    actions: List[RouterAction] = []
    skip_states = {TaskNodeStatus.DONE, TaskNodeStatus.FAILED,
                   TaskNodeStatus.ABANDONED}
    for t in _collect_descendants(scaffold.task_id, tasks):
        if t.status in skip_states:
            continue
        actions.append(UpdateTaskStatus(
            task_id=t.task_id, status=TaskNodeStatus.ABANDONED))
    actions.append(CreateTask(propose_regenerate(
        scaffold, constraint,
        regenerate_files=regen_files,
        prior_scaffold_artifact_id=prior_id)))
    return actions


def _fold_failure_into_goal(prior_goal: str, failure_note: str) -> str:
    """Append a re-plan failure note to a goal so DECOMPOSE can react.

    The revised goal keeps the original objective verbatim and adds a
    short, explicit note describing why the prior manifest could not be
    scaffolded so the planner restructures the plan (drop the offending
    file, split a layer, pick a simpler stack) instead of re-emitting the
    same broken manifest.
    """
    goal = (prior_goal or "").strip()
    note = (failure_note or "").strip()
    if not note:
        return goal
    banner = ("The previous plan could not be scaffolded. Revise the file "
              "manifest to avoid this failure: " + note)
    return f"{goal}\n\n{banner}" if goal else banner


def _replan_or_fail(completed: TaskNode, tasks: List[TaskNode], *,
                    scaffold: Optional[TaskNode],
                    failure_note: str) -> List[RouterAction]:
    """Escalate an exhausted regenerate budget to a fresh DECOMPOSE.

    When a SCAFFOLD/APPLY has spent its per-manifest
    :data:`_REGENERATE_BUDGET` the manifest itself is the suspect, not
    the generation of any single file. Before failing the session
    terminally the router escalates *once* (capped by
    :data:`_REPLAN_BUDGET`) to a revised plan: it abandons the live
    subtree under the failing SCAFFOLD and spawns a fresh DECOMPOSE whose
    goal folds in ``failure_note`` so the planner can restructure the
    manifest. The ``replan_attempt`` counter threads DECOMPOSE ->
    ASK_USER(APPROVE_PLAN) -> SCAFFOLD (and copies verbatim across
    :func:`propose_regenerate` retries), so a second exhaustion on the
    revised manifest falls through to terminal ``FAILED``. Returns the
    terminal-fail action when no SCAFFOLD lineage exists or the re-plan
    budget is already spent.
    """
    fail = [UpdateSessionStatus(
        session_id=completed.session_id, status=SessionStatus.FAILED)]
    if scaffold is None:
        return fail
    prior_replans = int(scaffold.inputs.get("replan_attempt") or 0)
    if prior_replans >= _REPLAN_BUDGET:
        # Budgets spent. Rather than discard every file that generated
        # cleanly, proceed with the survivors on the normal edge (the
        # empty return lets the dispatcher take SCAFFOLD -> APPLY /
        # APPLY -> VERIFY) whenever the partial scaffold produced output;
        # the dropped files are already surfaced to the UI via the
        # scaffold ``failed_count`` progress beats. Only a scaffold that
        # produced nothing usable is a terminal dead end.
        survivors = int((scaffold.outputs or {}).get("generated_count") or 0)
        return [] if survivors > 0 else fail
    prior_goal = str(scaffold.inputs.get("prior_goal") or "").strip()
    decompose = _find_ancestor_by_kind(scaffold, tasks, TaskKind.DECOMPOSE)
    answers: Dict[str, object] = {}
    if decompose is not None:
        prior_answers = decompose.inputs.get("answers")
        if isinstance(prior_answers, dict):
            answers = dict(prior_answers)
    new_decompose = TaskNode.new(
        session_id=completed.session_id,
        kind=TaskKind.DECOMPOSE,
        name="Revise the work plan",
        description=("Re-plan the file manifest after the prior plan's "
                     "scaffold could not be generated cleanly."),
        parent_task_id=scaffold.task_id,
        inputs={
            "prior_goal": _fold_failure_into_goal(prior_goal, failure_note),
            "requirements_artifact_id":
                scaffold.inputs.get("requirements_artifact_id"),
            "answers": answers,
            "replan_attempt": prior_replans + 1,
        },
    )
    actions: List[RouterAction] = []
    skip_states = {TaskNodeStatus.DONE, TaskNodeStatus.FAILED,
                   TaskNodeStatus.ABANDONED}
    for t in _collect_descendants(scaffold.task_id, tasks):
        if t.status in skip_states:
            continue
        actions.append(UpdateTaskStatus(
            task_id=t.task_id, status=TaskNodeStatus.ABANDONED))
    actions.append(CreateTask(new_decompose))
    return actions


def _scaffold_failed_files_actions(completed: TaskNode,
                                   tasks: List[TaskNode]) -> List[RouterAction]:
    """Regenerate (or terminally fail) a SCAFFOLD that dropped files.

    The SCAFFOLD executor records every file whose generation crashed
    (e.g. an LLM read timeout) or returned an empty patch under
    ``outputs.failed`` while still emitting the survivors -- so a partial
    generation returns *success* with ``failed_count > 0``. Proceeding to
    APPLY / VERIFY with a required module never written guarantees a
    downstream collection error the test-repair loop cannot fix: there is
    nothing to patch because the file is simply absent. Mirroring
    :func:`_apply_failed_files_actions`, any SCAFFOLD that dropped a file
    re-scaffolds within :data:`_REGENERATE_BUDGET`, folding the concrete
    per-file errors into the regenerate constraint so the retry has
    actionable feedback. When that budget is spent the router escalates
    once to a revised manifest via :func:`_replan_or_fail` (a fresh
    DECOMPOSE); when the re-plan budget is also spent that helper proceeds
    with the survivors on the normal SCAFFOLD -> APPLY edge rather than
    discarding them, failing terminally only when nothing usable was
    generated. Returns an empty list for a clean scaffold so the
    dispatcher takes the normal SCAFFOLD -> APPLY edge.
    """
    from cgx.session.repair.propose import propose_regenerate  # dep direction

    outputs = completed.outputs or {}
    failed_count = int(outputs.get("failed_count") or 0)
    if failed_count <= 0:
        return []
    constraint = _invalid_scaffold_constraint(
        failed_count, apply_failed=None,
        scaffold_failed=outputs.get("failed"))
    prior_regens = int(completed.inputs.get("regenerate_attempt") or 0)
    if prior_regens >= _REGENERATE_BUDGET:
        return _replan_or_fail(
            completed, tasks, scaffold=completed,
            failure_note=str(constraint.get("rationale") or ""))
    regen_files = _failed_scaffold_paths(outputs.get("failed"), None)
    prior_id = str(outputs.get("scaffold_artifact_id") or "").strip()
    actions: List[RouterAction] = []
    skip_states = {TaskNodeStatus.DONE, TaskNodeStatus.FAILED,
                   TaskNodeStatus.ABANDONED}
    for t in _collect_descendants(completed.task_id, tasks):
        if t.status in skip_states:
            continue
        actions.append(UpdateTaskStatus(
            task_id=t.task_id, status=TaskNodeStatus.ABANDONED))
    actions.append(CreateTask(propose_regenerate(
        completed, constraint,
        regenerate_files=regen_files,
        prior_scaffold_artifact_id=prior_id)))
    return actions


def _scaffold_resume_actions(
        failed: TaskNode, tasks: List[TaskNode],
        resume_scaffold_artifact_id: Optional[str]) -> List[RouterAction]:
    """Re-queue a SCAFFOLD that crashed mid-run to resume from a checkpoint.

    A SCAFFOLD executor checkpoints its SCAFFOLD_PATCHES artifact after
    every layer, so a crash or timeout leaves the completed files under
    an incomplete checkpoint. When the runner resolves that checkpoint
    and threads its id here, and the shared regenerate budget is not
    spent, abandon any live descendants and re-queue a fresh SCAFFOLD
    carrying ``resume_scaffold_artifact_id`` -- the new attempt seeds
    every checkpointed file and regenerates only the remainder, so the
    completed work is not discarded. The incremented ``regenerate_attempt``
    doubles as the crash-loop guard: a second crash exhausts the budget
    and falls through to terminal ``FAILED``. Returns an empty list (the
    caller then ends the session ``FAILED``) for a non-SCAFFOLD failure,
    an absent checkpoint, or a spent budget.
    """
    from cgx.session.repair.propose import propose_regenerate  # dep direction

    if failed.kind is not TaskKind.SCAFFOLD:
        return []
    resume_id = str(resume_scaffold_artifact_id or "").strip()
    if not resume_id:
        return []
    prior_regens = int(failed.inputs.get("regenerate_attempt") or 0)
    if prior_regens >= _REGENERATE_BUDGET:
        return []
    actions: List[RouterAction] = []
    skip_states = {TaskNodeStatus.DONE, TaskNodeStatus.FAILED,
                   TaskNodeStatus.ABANDONED}
    for t in _collect_descendants(failed.task_id, tasks):
        if t.status in skip_states:
            continue
        actions.append(UpdateTaskStatus(
            task_id=t.task_id, status=TaskNodeStatus.ABANDONED))
    actions.append(CreateTask(propose_regenerate(
        failed, {}, resume_scaffold_artifact_id=resume_id)))
    return actions


def _invalid_scaffold_constraint(
        failed_count: int,
        *, apply_failed: object,
        scaffold_failed: object) -> Dict[str, object]:
    """Build the ``invalid_scaffold_syntax`` regenerate constraint.

    Enumerates each dropped file with its concrete error so the next
    SCAFFOLD gets actionable feedback rather than a bare count. Draws
    from two sources, both shaped ``{"file", "error"}``: the SCAFFOLD's
    own ``failed`` generations (e.g. an empty patch for a missing
    entrypoint) and APPLY's ``failed_files`` (files whose source did not
    parse and were skipped before write). De-duplicated by path and
    capped so the constraint stays prompt-sized.
    """
    seen: set = set()
    details: List[str] = []
    for entry in (list(scaffold_failed or []) + list(apply_failed or [])):
        if not isinstance(entry, dict):
            continue
        path = str(entry.get("file") or "").strip()
        if not path or path in seen:
            continue
        seen.add(path)
        err = str(entry.get("error") or "").strip()
        details.append(f"{path} ({err})" if err else path)
        if len(details) >= 12:
            break
    files_blurb = "; ".join(details) if details else f"{failed_count} file(s)"
    rationale = (
        "The previous attempt was abandoned because these generated files "
        f"were invalid and dropped before write: {files_blurb}. Regenerate "
        "each dropped file so it parses as valid Python: keep decorated "
        "defs indented inside their class/function body, use consistent "
        "indentation and complete statements, avoid stray or trailing "
        "commas, define every referenced module and symbol, and import only "
        "modules that exist in this project or its declared dependencies.")
    return {
        "kind": "invalid_scaffold_syntax",
        "rationale": rationale,
        "failed_files": details,
    }


def _failed_scaffold_paths(scaffold_failed: object,
                           apply_failed: object) -> List[str]:
    """Return the de-duplicated file paths dropped by SCAFFOLD/APPLY.

    Draws from the same two ``{"file", "error"}`` sources as
    :func:`_invalid_scaffold_constraint` -- the SCAFFOLD's own ``failed``
    generations and APPLY's ``failed_files`` -- but returns just the
    paths so the router can hand SCAFFOLD a targeted regenerate set
    (regenerate only these; reuse every prior-good diff).
    """
    out: List[str] = []
    seen: set = set()
    for entry in (list(scaffold_failed or []) + list(apply_failed or [])):
        if not isinstance(entry, dict):
            continue
        path = str(entry.get("file") or "").strip()
        if not path or path in seen:
            continue
        seen.add(path)
        out.append(path)
    return out


def _verify_lesson_actions(completed: TaskNode,
                           tasks: List[TaskNode]) -> List[RouterAction]:
    """Emit :class:`RecordLesson` when a VERIFY-pass repairs a prior failure.

    Phase 7.1: a VERIFY whose outputs say ``outcome=passed`` and whose
    ancestor chain includes a REPAIR is, by construction, a successful
    repair cycle -- the REPAIR's diff (or its regenerate's fresh
    SCAFFOLD output) is what brought the test suite back to green. We
    surface that pair to the runner via a single :class:`RecordLesson`
    action carrying the VERIFY id, the REPAIR id (most recent ancestor),
    and the SCAFFOLD id if one exists on the chain (used as the
    lesson's ``scope`` provenance).
    """
    outputs = completed.outputs or {}
    outcome = str(outputs.get("outcome") or "").strip()
    if outcome != "passed":
        return []
    by_id = {t.task_id: t for t in tasks}
    repair: Optional[TaskNode] = None
    scaffold: Optional[TaskNode] = None
    cur_id = completed.parent_task_id
    visited: set = set()
    while cur_id and cur_id not in visited:
        visited.add(cur_id)
        cur = by_id.get(cur_id)
        if cur is None:
            break
        if repair is None and cur.kind is TaskKind.REPAIR:
            repair = cur
        if scaffold is None and cur.kind is TaskKind.SCAFFOLD:
            scaffold = cur
        cur_id = cur.parent_task_id
    if repair is None:
        return []
    return [RecordLesson(
        verify_task_id=completed.task_id,
        repair_task_id=repair.task_id,
        scaffold_task_id=scaffold.task_id if scaffold else None,
    )]


def _find_ancestor_by_kind(start: TaskNode, tasks: List[TaskNode],
                           kind: TaskKind) -> Optional[TaskNode]:
    """Walk up ``parent_task_id`` chain to the nearest task of ``kind``."""
    by_id = {t.task_id: t for t in tasks}
    visited: set = set()
    cur_id = start.parent_task_id
    while cur_id and cur_id not in visited:
        visited.add(cur_id)
        cur = by_id.get(cur_id)
        if cur is None:
            return None
        if cur.kind is kind:
            return cur
        cur_id = cur.parent_task_id
    return None


def _find_scaffold_ancestor(start: TaskNode,
                            tasks: List[TaskNode]) -> Optional[TaskNode]:
    """Walk up ``parent_task_id`` chain to the nearest SCAFFOLD task."""
    return _find_ancestor_by_kind(start, tasks, TaskKind.SCAFFOLD)


def _collect_descendants(root_task_id: str,
                         tasks: List[TaskNode]) -> List[TaskNode]:
    """Return every task whose ancestor chain includes ``root_task_id``.

    Bread-first walk over ``parent_task_id`` edges; the root itself is
    not included in the result -- only its successors are abandoned.
    """
    children_by_parent: Dict[str, List[TaskNode]] = {}
    for t in tasks:
        if t.parent_task_id:
            children_by_parent.setdefault(t.parent_task_id, []).append(t)
    out: List[TaskNode] = []
    queue: List[str] = [root_task_id]
    while queue:
        pid = queue.pop(0)
        for child in children_by_parent.get(pid, []):
            out.append(child)
            queue.append(child.task_id)
    return out


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
    """Spawn SCAFFOLD when the user approves the work plan."""
    if not bool(decision.chosen.get("approved")):
        return None
    work_plan_artifact_id = str(
        ask.inputs.get("work_plan_artifact_id") or "").strip()
    if not work_plan_artifact_id:
        return None
    return TaskNode.new(
        session_id=ask.session_id,
        kind=TaskKind.SCAFFOLD,
        name="Generate scaffolded files",
        description=("Generate the content for each file in the work "
                     "plan, layer by layer."),
        parent_task_id=ask.task_id,
        inputs={
            "work_plan_artifact_id": work_plan_artifact_id,
            "requirements_artifact_id":
                ask.inputs.get("requirements_artifact_id"),
            "prior_goal": ask.inputs.get("prior_goal"),
            "replan_attempt": ask.inputs.get("replan_attempt"),
            "decision_id": decision.decision_id,
        },
    )
