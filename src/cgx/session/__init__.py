

"""Session-shaped agent backbone -- CGX's core agent loop.

This package owns the persistent, cross-turn state for the
session-based agent loop: ``Session`` objects, the task tree
(``TaskNode``), an append-only ``KnowledgeBase``, the ``DecisionLog``,
and produced ``Artifact``s, plus the Router, task executors, and the
``SessionRunner`` that drives them. The web UI (``/api/agent-session``)
and the terminal dashboard are its consumers.
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
