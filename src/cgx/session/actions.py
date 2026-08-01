"""Typed router actions and the plan that carries them.

The :class:`~cgx.session.router.Router` never mutates state directly:
every method returns a :class:`RouterPlan` -- an ordered list of the
typed actions below -- that the caller (the runner or a route handler)
applies to the store. Keeping the action vocabulary in its own module
lets the router and the greenfield edge helpers share it without a
circular import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Union

from cgx.session.models import Decision, SessionStatus, TaskNode, TaskNodeStatus


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
