

"""SessionRunner -- the orchestrator the HTTP routes call.

The runner sits between the deterministic :class:`Router` (state
transitions, no IO) and the :class:`SessionStore` (persistence, no
business logic). All write paths funnel through here so a single
sequencer enforces:

* Router plans applied in order: creates, decisions, attaches,
  status updates.
* Per-session locking so two requests for the same session can't
  interleave half-applied plans.
* Centralised executor dispatch + failure handling.

Routes never touch the router or the store directly; they call
``runner.start_session``, ``runner.post_message``,
``runner.post_decision``, and ``runner.run_next``.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Optional

from cgx.session.models import (
    Decision,
    Session,
    SessionMode,
    TaskKind,
    TaskNode,
    TaskNodeStatus,
)
from cgx.session.router import (
    AttachDecisionToTask,
    CreateTask,
    RecordDecision,
    Router,
    RouterPlan,
    UpdateTaskStatus,
)
from cgx.session.store import SessionStore
from cgx.session.tasks.base import ExecutorDeps, ExecutorResult, dispatch

logger = logging.getLogger(__name__)


class SessionRunner:
    """Coordinator that owns the store + router for a project."""

    def __init__(self, store: SessionStore,
                 router: Optional[Router] = None) -> None:
        self._store = store
        self._router = router or Router()
        # Per-session locks so concurrent requests targeting the same
        # session serialise their store writes. ``defaultdict`` would
        # race on the first-access creation under threads; the dict +
        # outer lock pattern is the standard fix.
        self._locks: Dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    @property
    def store(self) -> SessionStore:
        return self._store

    @property
    def router(self) -> Router:
        return self._router

    # ----------------------- public API -----------------------

    def start_session(self, *, objective: str,
                      project_root: Optional[str] = None,
                      title: Optional[str] = None,
                      mode: SessionMode = SessionMode.EXPLORE) -> Session:
        """Create a session + its root task and persist both.

        ``mode`` decides whether the router seeds an EXPLORE root (default,
        existing codebase) or a CLARIFY_REQUIREMENTS root (greenfield).
        """
        session = Session.new(original_objective=objective,
                              project_root=project_root, title=title,
                              mode=mode)
        self._store.save_session(session)
        plan = self._router.on_user_message(
            session=session, message=objective, tasks=[])
        self._apply_plan(session, plan)
        # Link the root task back to the session for convenient lookup.
        roots = [t for t in self._store.list_tasks(session.session_id)
                 if t.parent_task_id is None]
        if roots and session.root_task_id is None:
            session.root_task_id = roots[0].task_id
            self._store.save_session(session)
        return session

    def post_message(self, *, session_id: str, message: str) -> RouterPlan:
        session = self._require_session(session_id)
        with self._lock_for(session_id):
            tasks = self._store.list_tasks(session_id)
            plan = self._router.on_user_message(
                session=session, message=message, tasks=tasks)
            self._apply_plan(session, plan)
        return plan

    def post_decision(self, *, session_id: str,
                      decision: Decision) -> RouterPlan:
        session = self._require_session(session_id)
        with self._lock_for(session_id):
            tasks = self._store.list_tasks(session_id)
            plan = self._router.on_decision_recorded(
                session=session, decision=decision, tasks=tasks)
            self._apply_plan(session, plan)
        return plan

    def run_next(self, *, session_id: str,
                 deps: ExecutorDeps) -> Optional[TaskNode]:
        """Pick the oldest READY task and execute it.

        Returns the task in its post-run state, or ``None`` if nothing
        was ready. ASK_USER tasks stay at ``IN_PROGRESS`` (waiting for
        the user) rather than ``DONE``; the runner skips them on
        subsequent calls until a decision arrives.
        """
        session = self._require_session(session_id)
        with self._lock_for(session_id):
            task = self._pick_ready(session_id)
            if task is None:
                return None
            return self._execute(session, task, deps)

    # ----------------------- internals -----------------------

    def _require_session(self, session_id: str) -> Session:
        s = self._store.get_session(session_id)
        if s is None:
            raise LookupError(f"unknown session {session_id!r}")
        return s

    def _lock_for(self, session_id: str) -> threading.Lock:
        with self._locks_guard:
            lock = self._locks.get(session_id)
            if lock is None:
                lock = threading.Lock()
                self._locks[session_id] = lock
            return lock

    def _pick_ready(self, session_id: str) -> Optional[TaskNode]:
        ready = self._store.tasks_by_status(
            session_id, TaskNodeStatus.READY)
        return ready[0] if ready else None

    def _execute(self, session: Session, task: TaskNode,
                 deps: ExecutorDeps) -> TaskNode:
        """Run a task through its executor and persist the outcome."""
        task.status = TaskNodeStatus.IN_PROGRESS
        task.started_at = time.time()
        self._store.save_task(task)

        try:
            result = dispatch(task, deps)
        except LookupError as exc:
            logger.warning("runner: %s", exc)
            return self._mark_failed(task, str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("runner: executor crashed")
            return self._mark_failed(
                task, f"{type(exc).__name__}: {exc}")

        # Persist facts even on failure -- they're append-only context
        # that may help the user diagnose what went wrong.
        for fact in result.facts:
            self._store.add_fact(fact)

        if result.failure:
            return self._mark_failed(task, result.failure)

        if result.artifact is not None:
            self._store.save_artifact(result.artifact)
            task.produced_artifact_id = result.artifact.artifact_id

        task.outputs = dict(result.outputs or {})

        if task.kind is TaskKind.ASK_USER:
            # Stays IN_PROGRESS until apply_decision posts a Decision.
            self._store.save_task(task)
            return task

        task.status = TaskNodeStatus.DONE
        task.completed_at = time.time()
        self._store.save_task(task)

        tasks_after = self._store.list_tasks(session.session_id)
        successor_plan = self._router.on_task_completed(
            session=session, completed=task, tasks=tasks_after)
        self._apply_plan(session, successor_plan)
        return task

    def _mark_failed(self, task: TaskNode, message: str) -> TaskNode:
        task.status = TaskNodeStatus.FAILED
        task.error = message
        task.completed_at = time.time()
        self._store.save_task(task)
        return task

    def _apply_plan(self, session: Session, plan: RouterPlan) -> None:
        """Apply the router's actions to the store, in order.

        Creates and decision records happen before status updates so a
        spawned child is visible to subscribers by the time a parent
        flips to ``DONE``.
        """
        for action in plan:
            if isinstance(action, CreateTask):
                self._store.save_task(action.task)
            elif isinstance(action, RecordDecision):
                self._store.record_decision(action.decision)
            elif isinstance(action, AttachDecisionToTask):
                t = self._store.get_task(action.task_id)
                if t is None:
                    logger.warning(
                        "runner: AttachDecision targets missing %s",
                        action.task_id)
                    continue
                if action.decision_id not in t.consumed_decision_ids:
                    t.consumed_decision_ids.append(action.decision_id)
                    self._store.save_task(t)
            elif isinstance(action, UpdateTaskStatus):
                t = self._store.get_task(action.task_id)
                if t is None:
                    logger.warning(
                        "runner: UpdateTaskStatus targets missing %s",
                        action.task_id)
                    continue
                t.status = action.status
                if action.clear_blockers:
                    t.blockers = []
                if action.status is TaskNodeStatus.DONE:
                    t.completed_at = time.time()
                self._store.save_task(t)
            else:  # pragma: no cover - exhaustive at the type level
                logger.warning("runner: unknown action %r", action)

