

"""ASK_USER pseudo-executor.

``ASK_USER`` tasks don't compute anything synchronously -- they pause
the session until the user supplies a :class:`Decision`. The runner
calls this executor when ``ASK_USER`` becomes ``READY`` so it can
transition to ``IN_PROGRESS`` with a serialised view of the open
question for the UI to render.

The actual resolution lives in :func:`apply_decision`, which the route
layer calls when the user posts a chip click or a freeform answer.
The function validates the decision against the task, then asks the
:class:`Router` what to spawn next. It does *not* touch the store --
the runner applies the returned :class:`RouterPlan` so writes stay
sequenced through one chokepoint.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from cgx.session.models import (
    Decision,
    DecisionKind,
    TaskKind,
    TaskNode,
)
from cgx.session.actions import RouterPlan
from cgx.session.router import Router
from cgx.session.tasks.base import (
    ExecutorDeps,
    ExecutorResult,
    register_executor,
)

logger = logging.getLogger(__name__)


@register_executor(TaskKind.ASK_USER)
def run_ask_user(task: TaskNode, deps: ExecutorDeps) -> ExecutorResult:
    """Surface the question payload; do not block.

    The runner keeps the task at ``IN_PROGRESS`` after this returns;
    progression to ``DONE`` happens via :func:`apply_decision`.
    """
    expected_kind = str(task.inputs.get("expected_kind")
                        or DecisionKind.FREEFORM.value)
    return ExecutorResult(outputs={
        "awaiting_decision": True,
        "expected_kind": expected_kind,
        "question": task.description or task.name,
        "directions_artifact_id": task.inputs.get("directions_artifact_id"),
    })


_ALLOWED_RECOMMENDATION_KINDS = {
    "investigate_more", "plan_change", "ask_followup", "done",
}


def build_decision(*, session_id: str, task: TaskNode,
                   chosen: Dict[str, Any],
                   rationale: str | None = None) -> Decision:
    """Construct a typed :class:`Decision` for ``task``.

    Validates that ``chosen`` carries the slot required by
    ``task.inputs["expected_kind"]``:

    * ``choose_path`` -> non-empty ``anchor_chunk_id``.
    * ``choose_recommendation`` -> ``kind`` is one of the four
      recommendation tokens; ``investigate_more`` additionally
      requires ``anchor_chunk_id``.
    * ``approve`` -> ``approved`` boolean must be present.
    * ``freeform`` -> no required slots.
    """
    if task.kind is not TaskKind.ASK_USER:
        raise ValueError(
            f"build_decision called on non-ASK_USER task {task.task_id}")
    kind = DecisionKind(task.inputs.get("expected_kind")
                       or DecisionKind.FREEFORM.value)
    if kind is DecisionKind.CHOOSE_PATH:
        anchor = str(chosen.get("anchor_chunk_id") or "").strip()
        if not anchor:
            raise ValueError(
                "choose_path decision requires non-empty anchor_chunk_id")
    elif kind is DecisionKind.CHOOSE_RECOMMENDATION:
        rec_kind = str(chosen.get("kind") or "").strip()
        if rec_kind not in _ALLOWED_RECOMMENDATION_KINDS:
            raise ValueError(
                "choose_recommendation decision requires kind in "
                f"{sorted(_ALLOWED_RECOMMENDATION_KINDS)}; got {rec_kind!r}")
        if rec_kind == "investigate_more" and not str(
                chosen.get("anchor_chunk_id") or "").strip():
            raise ValueError(
                "investigate_more recommendation requires "
                "non-empty anchor_chunk_id")
    elif kind is DecisionKind.APPROVE:
        if "approved" not in chosen:
            raise ValueError(
                "approve decision requires an 'approved' boolean")
    elif kind is DecisionKind.CLARIFY_ANSWERS:
        answers = chosen.get("answers")
        if not isinstance(answers, dict) or not answers:
            raise ValueError(
                "clarify_answers decision requires a non-empty "
                "'answers' dict (question_id -> answer text)")
    elif kind is DecisionKind.APPROVE_PLAN:
        if "approved" not in chosen:
            raise ValueError(
                "approve_plan decision requires an 'approved' boolean")
    return Decision.new(
        session_id=session_id,
        resolved_task_id=task.task_id,
        kind=kind,
        question=task.description or task.name,
        chosen=chosen,
        rationale=rationale,
    )


def apply_decision(*, router: Router, session, tasks: List[TaskNode],
                   decision: Decision) -> RouterPlan:
    """Hand the decision to the router and return its plan.

    Thin wrapper -- exists so the route layer doesn't depend on the
    router import directly and so future logic (e.g. multi-step
    decision validation) has one place to live.
    """
    return router.on_decision_recorded(
        session=session, decision=decision, tasks=tasks)
