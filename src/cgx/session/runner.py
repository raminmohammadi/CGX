

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
from typing import Any, Dict, List, Optional

from cgx.session.models import (
    ArtifactKind,
    Decision,
    Session,
    SessionMode,
    TaskKind,
    TaskNode,
    TaskNodeStatus,
)
from cgx.session.actions import (
    AttachDecisionToTask,
    CreateTask,
    RecordDecision,
    RecordLesson,
    RouterPlan,
    UpdateSessionStatus,
    UpdateTaskStatus,
)
from cgx.session.router import Router
from cgx.session.agent_log import log_event
from cgx.session.store import SessionStore
from cgx.session.tasks.base import ExecutorDeps, ExecutorResult, dispatch
from cgx.trace import (
    reset_trace_context,
    set_trace_context,
    traced as _traced,
)

logger = logging.getLogger(__name__)

_LLM_TASK_KINDS = {
    TaskKind.EXPLORE,
    TaskKind.INVESTIGATE,
    TaskKind.RECOMMEND,
    TaskKind.PLAN_CHANGE,
    TaskKind.REPAIR,
    TaskKind.CLARIFY_REQUIREMENTS,
    TaskKind.DECOMPOSE,
    TaskKind.SCAFFOLD,
}

# GPU inference throttle: serialise heavy LLM generation tasks to protect
# local GPU VRAM.
_GPU_INFERENCE_SEMAPHORE = threading.Semaphore(1)


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
                      mode: SessionMode = SessionMode.EXPLORE,
                      max_task_runs: Optional[int] = None,
                      max_wall_seconds: Optional[float] = None,
                      headless: bool = False,
                      skills: Optional[List[str]] = None) -> Session:
        """Create a session + its root task and persist both.

        ``mode`` decides whether the router seeds an EXPLORE root (default,
        existing codebase) or a CLARIFY_REQUIREMENTS root (greenfield).
        ``max_task_runs`` / ``max_wall_seconds`` cap the session's
        autonomous work (``None`` = unlimited); ``headless`` makes budget
        exhaustion terminal ``FAILED`` instead of pausing on an ASK_USER.
        ``skills`` pins the plan/scaffold executors to an explicit skill
        list instead of auto-detecting from the objective text.
        """
        session = Session.new(original_objective=objective,
                              project_root=project_root, title=title,
                              mode=mode, max_task_runs=max_task_runs,
                              max_wall_seconds=max_wall_seconds,
                              headless=headless, skills=skills)
        self._store.save_session(session)
        # Trace the seed-router call inside the session's context so the
        # on_user_message records land in the project agent.log.
        token = set_trace_context(
            session_id=session.session_id, project_root=session.project_root)
        try:
            plan = self._router.on_user_message(
                session=session, message=objective, tasks=[])
            self._apply_plan(session, plan)
        finally:
            reset_trace_context(token)
        # Link the root task back to the session for convenient lookup.
        roots = [t for t in self._store.list_tasks(session.session_id)
                 if t.parent_task_id is None]
        if roots and session.root_task_id is None:
            session.root_task_id = roots[0].task_id
            self._store.save_session(session)
        return session

    def post_message(self, *, session_id: str, message: str) -> RouterPlan:
        session = self._require_session(session_id)
        # Set the trace context *before* the @_traced wrapper fires so the
        # runner's own enter/exit records route to <project>/.cgx/agent.log
        # instead of the global fallback log.
        token = set_trace_context(
            session_id=session_id, project_root=session.project_root)
        try:
            return self._post_message_traced(session=session, message=message)
        finally:
            reset_trace_context(token)

    @_traced("runner")
    def _post_message_traced(self, *, session: Session,
                             message: str) -> RouterPlan:
        with self._lock_for(session.session_id):
            tasks = self._store.list_tasks(session.session_id)
            plan = self._router.on_user_message(
                session=session, message=message, tasks=tasks)
            self._apply_plan(session, plan)
        return plan

    def post_decision(self, *, session_id: str,
                      decision: Decision) -> RouterPlan:
        session = self._require_session(session_id)
        token = set_trace_context(
            session_id=session_id, project_root=session.project_root)
        try:
            return self._post_decision_traced(
                session=session, decision=decision)
        finally:
            reset_trace_context(token)

    @_traced("runner")
    def _post_decision_traced(self, *, session: Session,
                              decision: Decision) -> RouterPlan:
        with self._lock_for(session.session_id):
            tasks = self._store.list_tasks(session.session_id)
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
        token = set_trace_context(
            session_id=session_id, project_root=session.project_root)
        try:
            return self._run_next_traced(session=session, deps=deps)
        finally:
            reset_trace_context(token)

    @_traced("runner")
    def _run_next_traced(self, *, session: Session,
                         deps: ExecutorDeps) -> Optional[TaskNode]:
        with self._lock_for(session.session_id):
            task = self._pick_ready(session.session_id)
            if task is None:
                return None
            # The ASK_USER pause primitive is exempt from the budget so an
            # escalation prompt (and any pending user gate) can still be
            # surfaced once the cap is reached.
            if task.kind is not TaskKind.ASK_USER:
                reason = self._budget_reason(session)
                if reason is not None:
                    return self._escalate_budget(session, task, reason)
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

    def _budget_reason(self, session: Session) -> Optional[str]:
        """Return a human-readable cause when ``session`` is over budget.

        ``None`` means the session still has headroom on both axes (or
        neither cap is configured). The task-run cap is checked first so
        its message wins when both trip on the same call.
        """
        max_runs = session.max_task_runs
        if max_runs is not None and session.task_runs >= max_runs:
            unit = "task run" if max_runs == 1 else "task runs"
            return f"task budget ({max_runs} {unit})"
        max_secs = session.max_wall_seconds
        started = session.first_task_started_at
        if (max_secs is not None and started is not None
                and (time.time() - started) >= max_secs):
            return f"time budget ({max_secs:g}s)"
        return None

    def _escalate_budget(self, session: Session, task: TaskNode,
                         reason: str) -> TaskNode:
        """Stop an over-budget loop and route the escalation via the router.

        Returns the over-budget task in its post-escalation state (BLOCKED
        for an interactive pause, ABANDONED for a headless terminal fail)
        so the drain loop sees no lingering READY work and quiesces.
        """
        log_event(session.project_root, "session_budget_exhausted",
                  session_id=session.session_id, task_id=task.task_id,
                  reason=reason, task_runs=session.task_runs,
                  headless=session.headless)
        tasks_after = self._store.list_tasks(session.session_id)
        plan = self._router.on_budget_exhausted(
            session=session, over_task=task, tasks=tasks_after, reason=reason)
        self._apply_plan(session, plan)
        return self._store.get_task(task.task_id) or task

    def _execute(self, session: Session, task: TaskNode,
                 deps: ExecutorDeps) -> TaskNode:
        """Run a task through its executor and persist the outcome."""
        task.status = TaskNodeStatus.IN_PROGRESS
        task.started_at = time.time()
        self._store.save_task(task)
        # Charge the per-session budget for real work only; the ASK_USER
        # pause primitive is free so escalation/user gates don't consume
        # the cap. The wall-clock anchor starts on the first work task.
        if task.kind is not TaskKind.ASK_USER:
            session.task_runs += 1
            if session.first_task_started_at is None:
                session.first_task_started_at = task.started_at
            self._store.save_session(session)
        log_event(session.project_root, "task_started",
                  session_id=session.session_id, task_id=task.task_id,
                  kind=task.kind.value, name=task.name)

        provider = getattr(deps, "provider", None)
        traced = provider is not None and hasattr(provider, "bind") \
            and hasattr(provider, "drain")
        if traced:
            provider.bind(session.session_id, task.task_id)
        # Phase TR.2: propagate session/task/project_root into the trace
        # ContextVar so nested @traced calls land in the right agent.log.
        # Scope spans the entire executor + successor-router call so the
        # ``on_task_completed`` trace records also land in the project log.
        trace_token = set_trace_context(
            session_id=session.session_id,
            task_id=task.task_id,
            project_root=session.project_root,
        )
        try:
            try:
                if task.kind in _LLM_TASK_KINDS and provider is not None:
                    with _GPU_INFERENCE_SEMAPHORE:
                        result = dispatch(task, deps)
                else:
                    result = dispatch(task, deps)
            except LookupError as exc:
                logger.warning("runner: %s", exc)
                log_event(session.project_root, "executor_missing",
                          session_id=session.session_id,
                          task_id=task.task_id,
                          kind=task.kind.value, error=str(exc))
                if traced:
                    for fact in provider.drain():
                        self._store.add_fact(fact)
                return self._fail_and_route(session, task, str(exc))
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("runner: executor crashed")
                log_event(session.project_root, "executor_crashed",
                          session_id=session.session_id,
                          task_id=task.task_id,
                          kind=task.kind.value,
                          exc_type=type(exc).__name__, error=str(exc))
                if traced:
                    for fact in provider.drain():
                        self._store.add_fact(fact)
                return self._fail_and_route(
                    session, task, f"{type(exc).__name__}: {exc}")

            # Persist facts even on failure -- they're append-only context
            # that may help the user diagnose what went wrong. Tracing
            # facts are drained from the provider after the executor
            # returns so both executor-emitted facts and LLM-call facts
            # land in the same persistence loop.
            if traced:
                for fact in provider.drain():
                    self._store.add_fact(fact)
            for fact in result.facts:
                self._store.add_fact(fact)

            if result.failure:
                return self._fail_and_route(session, task, result.failure,
                                            retryable=result.retryable)

            if result.artifact is not None:
                self._store.save_artifact(result.artifact)
                task.produced_artifact_id = result.artifact.artifact_id

            task.outputs = dict(result.outputs or {})

            if task.kind is TaskKind.ASK_USER:
                # Stays IN_PROGRESS until apply_decision posts a Decision.
                self._store.save_task(task)
                log_event(session.project_root, "task_waiting_user",
                          session_id=session.session_id, task_id=task.task_id,
                          kind=task.kind.value)
                return task

            task.status = TaskNodeStatus.DONE
            task.completed_at = time.time()
            self._store.save_task(task)
            log_event(session.project_root, "task_completed",
                      session_id=session.session_id, task_id=task.task_id,
                      kind=task.kind.value,
                      artifact_id=task.produced_artifact_id,
                      duration_ms=int(((task.completed_at or 0.0)
                                       - (task.started_at or 0.0)) * 1000))

            tasks_after = self._store.list_tasks(session.session_id)
            successor_plan = self._router.on_task_completed(
                session=session, completed=task, tasks=tasks_after)
            self._apply_plan(session, successor_plan)
            return task
        finally:
            if traced:
                provider.unbind()
            reset_trace_context(trace_token)

    def _mark_failed(self, session: Session, task: TaskNode,
                     message: str) -> TaskNode:
        task.status = TaskNodeStatus.FAILED
        task.error = message
        task.completed_at = time.time()
        self._store.save_task(task)
        log_event(session.project_root, "task_failed",
                  session_id=session.session_id, task_id=task.task_id,
                  kind=task.kind.value, error=message)
        return task

    def _fail_and_route(self, session: Session, task: TaskNode,
                        message: str, *, retryable: bool = False) -> TaskNode:
        """Mark ``task`` FAILED and route the hard failure through the router.

        A hard failure (executor ``result.failure`` or a crash) produces
        no ``outputs``, so the ``on_task_completed`` successor table never
        runs. Feeding it to :meth:`Router.on_task_failed` lets a
        greenfield session reach its terminal ``FAILED`` status instead of
        hanging in ``active`` with a dead FAILED leaf. Explore-mode
        sessions get an empty plan, preserving their user-driven
        lifecycle. ``retryable`` forwards the executor's own verdict that
        the failure is plan-quality (an LLM retry could fix it); crashes
        and missing-executor failures are never retryable.
        """
        self._mark_failed(session, task, message)
        tasks_after = self._store.list_tasks(session.session_id)
        resume_id = self._resume_checkpoint_id(session, task)
        plan = self._router.on_task_failed(
            session=session, failed=task, tasks=tasks_after,
            resume_scaffold_artifact_id=resume_id,
            retryable=retryable)
        self._apply_plan(session, plan)
        return task

    def _resume_checkpoint_id(self, session: Session,
                              task: TaskNode) -> Optional[str]:
        """Return the id of ``task``'s incomplete SCAFFOLD checkpoint, if any.

        A SCAFFOLD executor upserts its SCAFFOLD_PATCHES artifact after
        every layer (B4), so a crash mid-run leaves an incomplete
        checkpoint (``content.complete`` falsy) whose ``produced_by_task_id``
        is the crashed task. Resolving it here lets the IO-free router
        decide whether to resume from it rather than discard the completed
        files. Returns ``None`` for a non-SCAFFOLD task or when no
        incomplete checkpoint exists (a clean crash with nothing written).
        """
        if task.kind is not TaskKind.SCAFFOLD:
            return None
        try:
            artifacts = self._store.list_artifacts(session.session_id)
        except Exception:  # pragma: no cover - defensive: store best-effort
            logger.exception("runner: list_artifacts failed resolving resume")
            return None
        for art in artifacts:
            if art.produced_by_task_id != task.task_id:
                continue
            if art.kind is not ArtifactKind.SCAFFOLD_PATCHES:
                continue
            if (art.content or {}).get("complete"):
                continue
            return art.artifact_id
        return None

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
                if action.error is not None:
                    t.error = action.error
                if action.status in (TaskNodeStatus.DONE,
                                     TaskNodeStatus.FAILED):
                    t.completed_at = time.time()
                self._store.save_task(t)
            elif isinstance(action, UpdateSessionStatus):
                self._apply_session_status(session, action)
            elif isinstance(action, RecordLesson):
                self._record_lesson(session, action)
            else:  # pragma: no cover - exhaustive at the type level
                logger.warning("runner: unknown action %r", action)

    def _apply_session_status(self, session: Session,
                              action: UpdateSessionStatus) -> None:
        """Persist a router-driven session lifecycle transition.

        The router emits :class:`UpdateSessionStatus` when a greenfield
        write loop reaches a definitive end (VERIFY passed -> COMPLETED;
        verification failed with no automated recovery -> FAILED). The
        target is normally the in-flight ``session``; fall back to a
        store lookup so a mismatched id still lands.
        """
        target = session
        if session.session_id != action.session_id:
            fetched = self._store.get_session(action.session_id)
            if fetched is None:
                logger.warning(
                    "runner: UpdateSessionStatus targets missing %s",
                    action.session_id)
                return
            target = fetched
        target.status = action.status
        self._store.save_session(target)
        log_event(target.project_root, "session_status_changed",
                  session_id=target.session_id, status=action.status.value)

    def delete_session_lock(self, session_id: str) -> None:
        """Evict the lock for a session to prevent memory leaks."""
        with self._locks_guard:
            self._locks.pop(session_id, None)

    def _record_lesson(self, session: Session,
                       action: RecordLesson) -> None:
        """Resolve a :class:`RecordLesson` action into a lessons.jsonl row.

        Best-effort: missing artifacts, malformed REPAIR_PLAN content,
        or disk failures are logged and swallowed so a learning hiccup
        cannot break the rest of the router plan.
        """
        from cgx.session.lessons import (
            extract_objective_keywords,
            record_lesson,
        )

        repair = self._store.get_task(action.repair_task_id)
        if repair is None or not repair.produced_artifact_id:
            return
        plan_artifact = self._store.get_artifact(repair.produced_artifact_id)
        if plan_artifact is None:
            return
        content = plan_artifact.content or {}
        signature = str(content.get("failure_signature") or "").strip()
        classification = str(content.get("classification") or "").strip()
        if not signature or not classification:
            return
        diffs = content.get("diffs") or []
        applied_fix: Dict[str, Any] = {
            "strategy": content.get("strategy") or (
                "patch" if diffs else "regenerate"),
            "diff_count": len(diffs) if isinstance(diffs, list) else 0,
            "files": sorted({d.get("file") for d in diffs
                             if isinstance(d, dict) and d.get("file")})
            if isinstance(diffs, list) else [],
            "extra_constraints": content.get("extra_constraints") or {},
        }
        scope: Dict[str, Any] = {}
        if action.scaffold_task_id:
            scaffold = self._store.get_task(action.scaffold_task_id)
            if scaffold is not None:
                goal = (scaffold.inputs or {}).get("prior_goal") or ""
                scope["objective_keywords"] = extract_objective_keywords(goal)
                pkgs = (scaffold.inputs or {}).get("stack_packages") or []
                if isinstance(pkgs, list):
                    scope["stack"] = [str(p) for p in pkgs]
        try:
            record_lesson(
                trigger_signature=signature,
                classification=classification,
                applied_fix=applied_fix,
                scope=scope,
                session_id=session.session_id,
            )
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("runner: record_lesson failed: %s", exc)

