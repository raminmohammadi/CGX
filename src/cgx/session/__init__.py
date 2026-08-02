

"""Session-shaped agent backbone -- CGX's core agent loop.

This package owns the persistent, cross-turn state for the
session-based agent loop: ``Session`` objects, the task tree
(``TaskNode``), an append-only ``KnowledgeBase``, the ``DecisionLog``,
and produced ``Artifact``s, plus the Router, task executors, and the
``SessionRunner`` that drives them. The web UI (``/api/agent-session``)
and the terminal dashboard are its consumers.

The autonomous greenfield write loop is designed to be *honest* and
*halting*. Honest: a non-executing ``VERIFY`` outcome
(``collection_error`` / ``timeout`` / ``pytest_missing``) reports
``passing_count=0`` so the router never mistakes "nothing ran" for
forward progress, and ``collection_error`` is a first-class REPAIR
classification that escalates to ``ASK_USER`` instead of silently
regenerating. Halting: every recovery loop is bounded by a typed
``LoopBudget`` (``cgx.session.budget``), the whole session is capped by
``GREENFIELD_MAX_TASK_RUNS`` / ``GREENFIELD_MAX_WALL_SECONDS``, and the
per-loop ledgers are carried across the ``DECOMPOSE -> approve_plan ->
SCAFFOLD`` re-plan so a re-plan cannot reset a spent budget. A dropped
foundational manifest (``requirements*.txt`` / ``pyproject.toml`` /
``setup.{py,cfg}`` / ``package.json``) escalates straight to a re-plan
rather than burning the per-file regenerate budget.
"""

from __future__ import annotations

from cgx.session.models import (
    Artifact,
    ArtifactKind,
    Decision,
    DecisionKind,
    DecisionLog,
    Fact,
    FactKind,
    KnowledgeBase,
    Session,
    SessionMode,
    SessionStatus,
    TaskKind,
    TaskNode,
    TaskNodeStatus,
)
from cgx.session.events import Event, EventBus, EventType, get_default_bus
from cgx.session.mode import detect_mode
from cgx.session.actions import RouterPlan
from cgx.session.router import Router
from cgx.session.runner import SessionRunner
from cgx.session.store import SessionStore

__all__ = [
    "Artifact",
    "ArtifactKind",
    "Decision",
    "DecisionKind",
    "DecisionLog",
    "Event",
    "EventBus",
    "EventType",
    "Fact",
    "FactKind",
    "KnowledgeBase",
    "Router",
    "RouterPlan",
    "Session",
    "SessionMode",
    "SessionRunner",
    "SessionStatus",
    "SessionStore",
    "TaskKind",
    "TaskNode",
    "TaskNodeStatus",
    "detect_mode",
    "get_default_bus",
]
