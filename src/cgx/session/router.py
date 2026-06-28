

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
from typing import Iterable, List, Optional, Union

from cgx.session.models import (
    Decision,
    DecisionKind,
    Session,
    SessionMode,
    TaskKind,
    TaskNode,
    TaskNodeStatus,
)

logger = logging.getLogger(__name__)


# --------------------- typed router actions ---------------------

@dataclass
class CreateTask:
    """Persist ``task`` as a new node in the tree."""
    task: TaskNode


@dataclass
class UpdateTaskStatus:
    """Transition ``task_id`` to ``status``; optionally clear blockers."""
    task_id: str
    status: TaskNodeStatus
    clear_blockers: bool = False


@dataclass
class RecordDecision:
    """Persist ``decision`` against the decision log."""
    decision: Decision


@dataclass
class AttachDecisionToTask:
    """Append ``decision_id`` to ``task_id``'s ``consumed_decision_ids``."""
    task_id: str
    decision_id: str


RouterAction = Union[CreateTask, UpdateTaskStatus, RecordDecision,
                     AttachDecisionToTask]


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


def _bootstrap_to_verify(parent: TaskNode) -> List[TaskNode]:
    """Spawn VERIFY once the project environment is provisioned.

    Always runs in greenfield mode (the only path that creates a
    BOOTSTRAP_ENV node). VERIFY reads the BUILD_REPORT to pick the
    project venv's python for the pytest subprocess.
    """
    return [TaskNode.new(
        session_id=parent.session_id,
        kind=TaskKind.VERIFY,
        name="Verify applied changes",
        description=("Run tests under the project's bootstrapped venv "
                     "to validate the applied changes."),
        parent_task_id=parent.task_id,
        inputs={
            "build_artifact_id": parent.produced_artifact_id,
            "apply_artifact_id": parent.inputs.get("apply_artifact_id"),
            "plan_artifact_id": parent.inputs.get("plan_artifact_id"),
            "scaffold_artifact_id":
                parent.inputs.get("scaffold_artifact_id"),
            "prior_goal": parent.inputs.get("prior_goal"),
            "mode": parent.inputs.get("mode"),
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
        },
    )]


_REPAIR_BUDGET = 2


# Outcomes that REPAIR knows how to attempt a fix for. ``passed``,
# ``skipped``, and ``no_tests_collected`` are terminal -- they're not
# failures. ``pytest_missing`` is BOOTSTRAP_ENV's job, not REPAIR's.
_REPAIRABLE_VERIFY_OUTCOMES = frozenset({
    "assertions_failed",
    "collection_error",
})


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
    """Spawn APPLY when REPAIR produced diffs; otherwise ASK_USER.

    The empty-diff path fires when classification returned ``unknown``
    or the locator could not pin the failure to a concrete span. The
    user-facing ASK_USER carries the classification + rationale so the
    operator can decide whether to edit the failing file by hand,
    abandon the session, or post a fresh objective.
    """
    outputs = parent.outputs or {}
    can_apply = bool(outputs.get("can_apply"))
    classification = str(outputs.get("classification") or "unknown")
    signature = str(outputs.get("failure_signature") or "")
    attempt = int(outputs.get("repair_attempt")
                  or parent.inputs.get("repair_attempt") or 1)
    prior = list(parent.inputs.get("prior_failure_signatures") or [])
    if not can_apply:
        return [TaskNode.new(
            session_id=parent.session_id,
            kind=TaskKind.ASK_USER,
            name="Repair could not produce a patch",
            description=("Automated repair did not yield a diff; review "
                         "the rationale and decide how to proceed."),
            parent_task_id=parent.task_id,
            inputs={
                "expected_kind": DecisionKind.FREEFORM.value,
                "repair_artifact_id": parent.produced_artifact_id,
                "classification": classification,
                "prior_goal": parent.inputs.get("prior_goal"),
            },
        )]
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
    TaskKind.BOOTSTRAP_ENV: _bootstrap_to_verify,
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

    def on_task_completed(self, *, session: Session,
                          completed: TaskNode,
                          tasks: List[TaskNode]) -> RouterPlan:
        """Spawn successors for a task that just finished.

        The dispatch is table-driven via :data:`TASK_SUCCESSOR`; a
        missing entry is a no-op (used for terminal kinds like
        ``VERIFY`` or for kinds whose successor lives in a later
        phase).
        """
        plan = RouterPlan()
        spawn = TASK_SUCCESSOR.get(completed.kind)
        if spawn is None:
            return plan
        for child in spawn(completed):
            plan.actions.append(CreateTask(child))
        return plan

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
            "decision_id": decision.decision_id,
        },
    )
