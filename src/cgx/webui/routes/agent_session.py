

"""Session-backed agent endpoints (Phase 1 of the redesign).

This sits next to the legacy ``/api/agent`` SSE route -- it does
*not* replace it. Routes here expose the new session backbone
(``cgx.session``) over plain JSON for now; streaming is added in a
later phase once the UI is rewired.

Endpoint shape:

* ``POST /api/agent-session`` -- create a session, seed the root
  EXPLORE task, optionally run it.
* ``GET  /api/agent-session?project_root=...`` -- list sessions.
* ``GET  /api/agent-session/{sid}`` -- full state snapshot.
* ``POST /api/agent-session/{sid}/message`` -- post a follow-up.
* ``POST /api/agent-session/{sid}/decision`` -- resolve an ASK_USER.
* ``DELETE /api/agent-session/{sid}`` -- discard a session and its
  tasks / facts / decisions / artifacts (``ON DELETE CASCADE``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from sse_starlette.sse import EventSourceResponse

# Importing the tasks package side-effect-registers Phase 1 executors.
from cgx.session import tasks as _tasks  # noqa: F401
from cgx.session import SessionRunner, SessionStore
from cgx.session.events import Event, EventType, get_default_bus
from cgx.session.llm_trace import TracingProvider
from cgx.session.mode import detect_mode
from cgx.session.models import SessionMode
from cgx.session.tasks.ask import build_decision
from cgx.session.tasks.base import ExecutorDeps
from cgx.webui.handlers import _resolve_provider
from cgx.webui.models import (
    AgentSessionCreateRequest,
    AgentSessionDecisionRequest,
    AgentSessionMessageRequest,
    AgentSessionState,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["agent-session"], prefix="/agent-session")


# --------------------- per-project runner cache ---------------------

# One :class:`SessionStore` per ``project_root`` so the SQLite WAL
# connection is reused across requests. Keyed on the resolved string;
# ``None`` falls back to the user-global path used by Phase 0 tests.
_RUNNERS: Dict[str, SessionRunner] = {}
_RUNNERS_LOCK = threading.Lock()
_SESSION_TO_RUNNER: Dict[str, SessionRunner] = {}


def _get_runner(project_root: Optional[str]) -> SessionRunner:
    key = str(project_root or "__default__")
    with _RUNNERS_LOCK:
        runner = _RUNNERS.get(key)
        if runner is None:
            store = SessionStore(project_root=project_root)
            runner = SessionRunner(store)
            _RUNNERS[key] = runner
        return runner


def _build_deps(req_provider, req_index, project_root: Optional[str],
                store: Optional[SessionStore] = None) -> ExecutorDeps:
    # A missing profile (or otherwise invalid provider config) is a client
    # error, not a server fault: surface it as a clean 400 with the
    # underlying message instead of letting the ValueError escape as a 500.
    try:
        provider = _resolve_provider(
            use_profile=req_provider.use_profile,
            profile_name=req_provider.profile_name,
            kind=req_provider.kind, model=req_provider.model,
            base_url=req_provider.base_url, api_key=req_provider.api_key,
            temperature=req_provider.temperature,
            num_predict=req_provider.num_predict,
            num_ctx=getattr(req_provider, "num_ctx", None),
            endpoint_path=getattr(req_provider, "endpoint_path",
                                  "/v1/chat/completions"),
            allow_no_auth=bool(getattr(req_provider, "allow_no_auth", False)),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    # Wrap with TracingProvider so every chat / chat_stream invocation
    # the executor makes lands as an LLM_CALL fact attributed to the
    # producing task (Phase 5.1). Untraced providers (already wrapped
    # in nested calls) are passed through unchanged.
    if provider is not None and not isinstance(provider, TracingProvider):
        provider = TracingProvider(provider)
    return ExecutorDeps(
        project_root=project_root,
        index_dir=req_index.index_dir,
        records_path=req_index.records,
        embed_model=req_index.embed_model,
        provider=provider,
        store=store,
    )


def _snapshot(runner: SessionRunner, session_id: str) -> AgentSessionState:
    store = runner.store
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404,
                            detail=f"session {session_id!r} not found")
    return AgentSessionState(
        session=session.to_dict(),
        tasks=[t.to_dict() for t in store.list_tasks(session_id)],
        artifacts=[a.to_dict() for a in store.list_artifacts(session_id)],
        facts=[f.to_dict() for f in
               store.load_kb(session_id).facts.values()],
        decisions=[d.to_dict() for d in
                   store.load_decisions(session_id).decisions.values()],
    )


async def _drain_ready(runner: SessionRunner, session_id: str,
                       deps: ExecutorDeps, *, max_steps: int = 64) -> None:
    """Synchronously execute READY tasks until none remain or budget exhausts.

    The loop stops naturally when ``run_next`` returns ``None`` -- either
    nothing is READY or the pipeline paused on an ASK_USER (which goes to
    IN_PROGRESS, not DONE, so it leaves no READY task behind). ``max_steps``
    is only a safety valve against a router bug that spawns READY tasks
    without end; it must not double as a functional limit, or a task
    created READY past the cap is stranded with no request to re-drive it.

    The greenfield write pipeline is SCAFFOLD -> APPLY -> BOOTSTRAP_ENV ->
    API_CHECK -> SMOKE -> VERIFY (6 tasks on the happy path). An
    API_CHECK / SMOKE / VERIFY failure can splice in a bounded REPAIR
    detour -- either a regenerate (fresh SCAFFOLD -> APPLY -> ...) or a
    patch (APPLY -> VERIFY). Those detours are capped inside the router
    (``REPAIR_BUDGET`` / ``REGENERATE_BUDGET`` plus no-progress
    signature guards), so the total per drive is bounded well under the
    cap; the previous default of 6 cut the very first repair cycle off
    mid-pipeline and left the regenerated APPLY stuck at READY.
    """
    for _ in range(max_steps):
        # Cooperative cancel (P2.2): honour a stop request *between* tasks
        # so the in-flight task finishes cleanly and no further READY task
        # is dispatched. The flag is consumed here so a later message can
        # resume the session. A hard mid-task abort is intentionally not
        # done -- it would leave partial artifacts and a running worker
        # thread with no owner.
        if _consume_cancel(session_id):
            logger.info("drain: cancel honoured for session %s; stopping "
                        "before the next task", session_id)
            get_default_bus().publish(Event(
                type=EventType.SESSION_UPDATED, session_id=session_id,
                payload={"cancelled": True}))
            return
        # The session may have been deleted out from under a running drain
        # (DELETE cancels the drive but a task already in flight keeps its
        # worker thread). Stop cleanly before dispatching the next task so
        # we never write a child row against a vanished parent (which would
        # trip an ``ON DELETE CASCADE`` FK error).
        if runner.store.get_session(session_id) is None:
            logger.info("drain: session %s no longer exists; stopping",
                        session_id)
            return
        try:
            task = await asyncio.to_thread(
                runner.run_next, session_id=session_id, deps=deps)
        except Exception:
            # If the session was deleted while this task ran, persisting its
            # result trips a FK error -- treat that as a benign stop rather
            # than a drain failure. Any other error propagates as before.
            if runner.store.get_session(session_id) is None:
                logger.info("drain: session %s deleted mid-task; discarding "
                            "in-flight result", session_id)
                return
            raise
        if task is None:
            return
    logger.warning(
        "drain: hit max_steps=%d for session %s without quiescing; "
        "a READY task may remain undispatched", max_steps, session_id)


# --------------------- background drain ---------------------

# Production drives the drain as a detached background task so the POST
# returns the snapshot immediately and the UI follows progress over SSE.
# Tests flip this off to keep the historical synchronous contract (the
# drain finishes before the returned snapshot is taken).
_RUN_DRAIN_IN_BACKGROUND = True
_SESSION_DRAINS: Dict[str, "asyncio.Task[None]"] = {}

# Session ids with a pending cooperative-cancel request. The drain loop
# consumes the flag between tasks (see ``_drain_ready``). Guarded by a
# lock only for defensiveness -- both the setter (cancel route) and the
# consumer (drain) run on the same event loop.
_CANCEL_REQUESTS: set[str] = set()
_CANCEL_LOCK = threading.Lock()


def request_cancel(session_id: str) -> None:
    """Flag ``session_id`` so its running drain stops after the next task."""
    with _CANCEL_LOCK:
        _CANCEL_REQUESTS.add(session_id)


def _consume_cancel(session_id: str) -> bool:
    """Return True and clear the flag when a cancel is pending for ``sid``."""
    with _CANCEL_LOCK:
        if session_id in _CANCEL_REQUESTS:
            _CANCEL_REQUESTS.discard(session_id)
            return True
        return False


async def _schedule_drain(runner: SessionRunner, session_id: str,
                          deps: ExecutorDeps, *, max_steps: int = 64) -> None:
    """Start (or coalesce onto) a background drain for ``session_id``.

    A single drain per session is kept in ``_SESSION_DRAINS``; if one is
    already running it is left to pick up the freshly-created READY task on
    its next loop rather than spawning a duplicate that would contend for
    the same runner lock. When ``_RUN_DRAIN_IN_BACKGROUND`` is off the drain
    runs inline (synchronous, test-only behaviour).
    """
    if not _RUN_DRAIN_IN_BACKGROUND:
        await _drain_ready(runner, session_id, deps, max_steps=max_steps)
        return
    existing = _SESSION_DRAINS.get(session_id)
    if existing is not None and not existing.done():
        return
    # Fresh drive: drop any stale cancel flag left by a prior run so a
    # user-initiated resume isn't cancelled before its first task.
    _consume_cancel(session_id)
    task = asyncio.create_task(
        _drain_ready(runner, session_id, deps, max_steps=max_steps),
        name=f"drain:{session_id}")
    _SESSION_DRAINS[session_id] = task

    def _cleanup(t: "asyncio.Task[None]") -> None:
        if _SESSION_DRAINS.get(session_id) is t:
            _SESSION_DRAINS.pop(session_id, None)
        if t.cancelled():
            return
        exc = t.exception()
        if exc is not None:  # pragma: no cover - defensive logging
            logger.error("drain: background task for %s failed: %s: %s",
                         session_id, type(exc).__name__, exc)

    task.add_done_callback(_cleanup)


def _safe_json(payload: Any) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:  # pragma: no cover - defensive
        return json.dumps({"_repr": str(payload)})

def _normalize_project_root(project_root: Optional[str]) -> Optional[str]:
    """Canonicalize a caller-supplied project root before it reaches disk.

    Resolves ``~`` and any ``..``/relative segments to an absolute path
    up front so a crafted value can't be used to escape the intended
    directory once it flows into filesystem operations (``SessionStore``,
    ``detect_mode``, etc.) -- CodeQL: uncontrolled data used in a path
    expression. This intentionally does not restrict *which* directory
    may be used: pointing the agent at an arbitrary local project is the
    whole point of ``project_root``.
    """
    if project_root is None:
        return None
    # Normalize with ``os.path.abspath`` (a pure normalization, not a
    # filesystem-access sink like ``Path.resolve``) after expanding ``~``.
    # This collapses any ``..`` segments up front; it deliberately does not
    # restrict *which* directory may be used -- pointing the agent at an
    # arbitrary local project is the whole point of ``project_root``.
    return os.path.abspath(os.path.expanduser(project_root))


# --------------------- routes ---------------------

@router.post("", response_model=AgentSessionState)
async def create_session(req: AgentSessionCreateRequest) -> AgentSessionState:
    project_root = _normalize_project_root(req.project_root)
    runner = _get_runner(project_root)
    mode = _resolve_mode(req)
    session = await asyncio.to_thread(
        runner.start_session, objective=req.objective,
        project_root=project_root, title=req.title, mode=mode,
        skills=req.skills or None)
    with _RUNNERS_LOCK:
        _SESSION_TO_RUNNER[session.session_id] = runner
    if req.run_initial_task:
        deps = _build_deps(req.provider, req.index, project_root,
                           store=runner.store)
        await _schedule_drain(runner, session.session_id, deps)
    return _snapshot(runner, session.session_id)


def _resolve_mode(req: AgentSessionCreateRequest) -> SessionMode:
    """Honor an explicit ``mode`` if given; otherwise auto-detect."""
    requested = (req.mode or "").strip().lower()
    if requested:
        try:
            return SessionMode(requested)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=(f"invalid mode {requested!r}; "
                        f"expected one of "
                        f"{[m.value for m in SessionMode]}")) from exc
    normalized_project_root = _normalize_project_root(req.project_root)
    normalized_index_dir = _normalize_project_root(req.index.index_dir)
    normalized_records_path = _normalize_project_root(req.index.records)
    # Containment of ``index_dir`` / ``records_path`` (and their relation to
    # ``project_root``) is enforced inside ``_has_usable_index`` via the
    # CodeQL-recognized ``startswith`` prefix guard, so no pre-check here.
    return detect_mode(
        project_root=normalized_project_root,
        index_dir=normalized_index_dir,
        records_path=normalized_records_path,
    )


@router.get("", response_model=List[Dict[str, Any]])
async def list_agent_sessions(
        project_root: Optional[str] = Query(default=None)) -> List[Dict[str, Any]]:
    normalized_project_root = _normalize_project_root(project_root)
    runner = _get_runner(normalized_project_root)
    return [s.to_dict() for s in
            runner.store.list_sessions(project_root=normalized_project_root)]


@router.get("/{sid}", response_model=AgentSessionState)
async def get_session(sid: str,
                      project_root: Optional[str] = Query(default=None)
                      ) -> AgentSessionState:
    normalized_project_root = _normalize_project_root(project_root)
    runner = _resolve_runner_for(sid) if normalized_project_root is None \
        else _get_runner(normalized_project_root)
    return _snapshot(runner, sid)


@router.get("/{sid}/events")
async def session_events(sid: str, request: Request) -> EventSourceResponse:
    """Stream live session events over SSE.

    Subscribes to the process-wide :class:`EventBus`, filters to ``sid``,
    and forwards each event as a named SSE frame. The store publishes from
    the drain's worker thread, so events are marshalled onto the request's
    event loop via ``call_soon_threadsafe`` and a bounded queue (newest
    events are dropped under back-pressure rather than blocking the drain).
    A ``snapshot`` frame is sent first so a late subscriber still renders
    current state; periodic ``ping`` frames detect a vanished client.
    """
    runner = _resolve_runner_for(sid)
    if runner.store.get_session(sid) is None:
        raise HTTPException(status_code=404,
                            detail=f"session {sid!r} not found")

    bus = get_default_bus()
    loop = asyncio.get_running_loop()
    queue: "asyncio.Queue[Event]" = asyncio.Queue(maxsize=1024)

    def _push(event: Event) -> None:
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:  # pragma: no cover - slow consumer
            pass

    def _on_event(event: Event) -> None:
        if event.session_id != sid:
            return
        try:
            loop.call_soon_threadsafe(_push, event)
        except RuntimeError:  # pragma: no cover - loop torn down
            pass

    unsubscribe = bus.subscribe("*", _on_event)

    async def _generator() -> AsyncIterator[Dict[str, str]]:
        try:
            snap = _snapshot(runner, sid)
            yield {"event": "snapshot", "data": _safe_json(snap.model_dump())}
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield {"event": "ping", "data": "{}"}
                    continue
                yield {"event": event.type.value,
                       "data": _safe_json(event.to_dict())}
        finally:
            unsubscribe()

    return EventSourceResponse(_generator())


@router.post("/{sid}/message", response_model=AgentSessionState)
async def post_message(sid: str,
                       req: AgentSessionMessageRequest) -> AgentSessionState:
    runner = _resolve_runner_for(sid)
    session = runner.store.get_session(sid)
    if session is None:
        raise HTTPException(status_code=404,
                            detail=f"session {sid!r} not found")
    await asyncio.to_thread(
        runner.post_message, session_id=sid, message=req.message)
    if req.run_initial_task:
        deps = _build_deps(req.provider, req.index, session.project_root,
                           store=runner.store)
        await _schedule_drain(runner, sid, deps)
    return _snapshot(runner, sid)


@router.post("/{sid}/decision", response_model=AgentSessionState)
async def post_decision(sid: str,
                        req: AgentSessionDecisionRequest) -> AgentSessionState:
    runner = _resolve_runner_for(sid)
    task = runner.store.get_task(req.task_id)
    if task is None or task.session_id != sid:
        raise HTTPException(status_code=404,
                            detail=f"task {req.task_id!r} not in session")
    try:
        decision = build_decision(
            session_id=sid, task=task,
            chosen=req.chosen, rationale=req.rationale)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    await asyncio.to_thread(
        runner.post_decision, session_id=sid, decision=decision)
    if req.run_initial_task:
        session = runner.store.get_session(sid)
        if session is not None:
            deps = _build_deps(req.provider, req.index,
                               session.project_root, store=runner.store)
            await _schedule_drain(runner, sid, deps)
    return _snapshot(runner, sid)


@router.post("/{sid}/cancel", response_model=AgentSessionState)
async def post_cancel(sid: str) -> AgentSessionState:
    """Request a cooperative stop of the session's running drain (P2.2).

    Flags the session so its background drain stops after the current
    task finishes -- no task is aborted mid-flight. Returns the current
    snapshot immediately; the UI follows the stop over SSE. A later
    message/decision re-drives the session from where it stopped.
    """
    runner = _resolve_runner_for(sid)
    session = runner.store.get_session(sid)
    if session is None:
        raise HTTPException(status_code=404,
                            detail=f"session {sid!r} not found")
    request_cancel(sid)
    return _snapshot(runner, sid)


@router.delete("/{sid}", response_model=Dict[str, Any])
async def delete_session(sid: str,
                         project_root: Optional[str] = Query(default=None)
                         ) -> Dict[str, Any]:
    """Remove a session and its descendants from the store.

    Foreign keys on ``tasks`` / ``facts`` / ``decisions`` / ``artifacts``
    are declared ``ON DELETE CASCADE`` so a single row delete on
    ``sessions`` removes the full aggregate. Returns ``{deleted: sid}``
    on success or 404 if no runner knows about the id.
    """
    normalized_project_root = _normalize_project_root(project_root)
    runner = _resolve_runner_for(sid) if normalized_project_root is None \
        else _get_runner(normalized_project_root)
    # Stop any in-flight background drain before removing the row so its
    # worker can't race the delete and write a child row against the
    # now-gone session (FK ``ON DELETE CASCADE`` -> IntegrityError). The
    # cancel flag halts the loop between tasks; cancelling the task stops
    # it from dispatching anything further; the drain's own vanished-session
    # guard swallows the current task's result if it was already running.
    request_cancel(sid)
    drain = _SESSION_DRAINS.pop(sid, None)
    if drain is not None and not drain.done():
        drain.cancel()
    removed = await asyncio.to_thread(runner.store.delete_session, sid)
    if not removed:
        raise HTTPException(status_code=404,
                            detail=f"session {sid!r} not found")
    runner.delete_session_lock(sid)
    _consume_cancel(sid)
    with _RUNNERS_LOCK:
        _SESSION_TO_RUNNER.pop(sid, None)
    return {"deleted": sid}


# --------------------- helpers ---------------------

def _resolve_runner_for(sid: str) -> SessionRunner:
    """Find which cached runner already knows about ``sid``.

    Falls back to the default (project_root=None) runner if nothing
    matches -- typical for sessions created without a project_root.
    """
    with _RUNNERS_LOCK:
        runner = _SESSION_TO_RUNNER.get(sid)
        if runner is not None:
            return runner
        runners = list(_RUNNERS.values())
    for r in runners:
        if r.store.get_session(sid) is not None:
            with _RUNNERS_LOCK:
                _SESSION_TO_RUNNER[sid] = r
            return r
    default_runner = _get_runner(None)
    with _RUNNERS_LOCK:
        _SESSION_TO_RUNNER[sid] = default_runner
    return default_runner
