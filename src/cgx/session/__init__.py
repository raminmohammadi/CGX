

"""Session-shaped agent backbone (Phase 0 of the redesign).

This package owns the persistent, cross-turn state for the new
session-based agent loop: ``Session`` objects, the task tree
(``TaskNode``), an append-only ``KnowledgeBase``, the ``DecisionLog``,
and produced ``Artifact``s. Phase 0 ships the data layer only -- the
Router, executors, and UI integration arrive in later phases.

Nothing in :mod:`cgx.agents` imports from here yet. The new
``/api/session/*`` routes (Phase 1) will be the first consumer.
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
from cgx.session.router import Router, RouterPlan
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
