

"""In-process event bus for session state changes.

The bus is the single seam through which :mod:`cgx.session.store`,
the router (Phase 1+), and the SSE bridge (Phase 1+) communicate.
Subscribers register callbacks for specific event types or the
wildcard ``"*"``; publishes are synchronous so the store always sees
its own writes reflected on read-back paths driven by callbacks.

The bus is intentionally process-local and dependency-free; no
asyncio, no threading-safe queue. Phase 1 wraps it with an SSE
adapter that buffers per-connection.
"""

from __future__ import annotations

import enum
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class EventType(str, enum.Enum):
    """All event kinds the session layer can emit.

    Phase 0 ships the constants; the store and router populate them
    in later phases. Keeping the enum in one place makes the SSE
    bridge's switch-on-type trivial.
    """
    SESSION_CREATED = "session.created"
    SESSION_UPDATED = "session.updated"
    TASK_CREATED = "task.created"
    TASK_STATUS_CHANGED = "task.status_changed"
    TASK_OUTPUT_PARTIAL = "task.output_partial"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    DECISION_RECORDED = "decision.recorded"
    FACT_ADDED = "fact.added"
    FACT_STALE = "fact.stale"
    ARTIFACT_CREATED = "artifact.created"


@dataclass
class Event:
    """Single bus message.

    ``payload`` is whatever the producer wants to ship; subscribers
    are responsible for downcasting. ``session_id`` is duplicated out
    of ``payload`` so subscribers can filter cheaply.
    """
    type: EventType
    session_id: str
    payload: Dict[str, Any] = field(default_factory=dict)
    at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["type"] = self.type.value
        return d


Subscriber = Callable[[Event], None]


class EventBus:
    """Synchronous fan-out bus.

    Thread-safe: subscribe / unsubscribe / publish are all guarded
    by an :class:`RLock` so callback re-entry (a subscriber that
    publishes during its own handler) doesn't deadlock.
    """

    _WILDCARD = "*"

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._subs: Dict[str, List[Subscriber]] = {}

    def subscribe(self, event_type: Any, callback: Subscriber) -> Callable[[], None]:
        """Register ``callback`` for ``event_type`` (or "*" for all).

        Returns an unsubscribe thunk; callers should retain it for
        teardown (tests, SSE disconnect, etc).
        """
        key = self._normalise(event_type)
        with self._lock:
            self._subs.setdefault(key, []).append(callback)

        def _unsub() -> None:
            self.unsubscribe(event_type, callback)

        return _unsub

    def unsubscribe(self, event_type: Any, callback: Subscriber) -> None:
        key = self._normalise(event_type)
        with self._lock:
            subs = self._subs.get(key)
            if not subs:
                return
            try:
                subs.remove(callback)
            except ValueError:
                pass
            if not subs:
                self._subs.pop(key, None)

    def publish(self, event: Event) -> None:
        """Deliver ``event`` to type-specific then wildcard subscribers.

        Subscriber exceptions are caught and logged so a single bad
        listener cannot poison the rest of the fan-out.
        """
        key = event.type.value
        with self._lock:
            targets = list(self._subs.get(key, []))
            targets.extend(self._subs.get(self._WILDCARD, []))
        for cb in targets:
            try:
                cb(event)
            except Exception as e:  # pragma: no cover - logged not raised
                logger.warning("event_bus: subscriber raised on %s: %s: %s",
                               key, type(e).__name__, e)

    def clear(self) -> None:
        """Drop every subscriber. For tests only."""
        with self._lock:
            self._subs.clear()

    @classmethod
    def _normalise(cls, event_type: Any) -> str:
        if isinstance(event_type, EventType):
            return event_type.value
        if event_type == cls._WILDCARD:
            return cls._WILDCARD
        return str(event_type)


_default_bus: Optional[EventBus] = None
_default_bus_lock = threading.Lock()


def get_default_bus() -> EventBus:
    """Process-wide bus shared by the store and webui adapters."""
    global _default_bus
    with _default_bus_lock:
        if _default_bus is None:
            _default_bus = EventBus()
        return _default_bus
