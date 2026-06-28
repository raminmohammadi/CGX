

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
import logging
import threading
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

# Importing the tasks package side-effect-registers Phase 1 executors.
from cgx.session import tasks as _tasks  # noqa: F401
from cgx.session import SessionRunner, SessionStore
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
                       deps: ExecutorDeps, *, max_steps: int = 6) -> None:
    """Synchronously execute READY tasks until none remain or budget exhausts.

    Capped at ``max_steps`` so a runaway router doesn't peg the request
    thread. ASK_USER tasks intentionally stop the loop -- they go to
    IN_PROGRESS rather than DONE. The greenfield write loop spawns up
    to five tasks per decision (SCAFFOLD -> APPLY -> BOOTSTRAP_ENV ->
    SMOKE -> VERIFY), so the default budget is sized to cover it with
    one step of headroom.
    """
    for _ in range(max_steps):
        task = await asyncio.to_thread(
            runner.run_next, session_id=session_id, deps=deps)
        if task is None:
            return


# --------------------- routes ---------------------

@router.post("", response_model=AgentSessionState)
async def create_session(req: AgentSessionCreateRequest) -> AgentSessionState:
    runner = _get_runner(req.project_root)
    mode = _resolve_mode(req)
    session = await asyncio.to_thread(
        runner.start_session, objective=req.objective,
        project_root=req.project_root, title=req.title, mode=mode)
    if req.run_initial_task:
        deps = _build_deps(req.provider, req.index, req.project_root,
                           store=runner.store)
        await _drain_ready(runner, session.session_id, deps)
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
    return detect_mode(
        project_root=req.project_root,
        index_dir=req.index.index_dir,
        records_path=req.index.records,
    )


@router.get("", response_model=List[Dict[str, Any]])
async def list_agent_sessions(
        project_root: Optional[str] = Query(default=None)) -> List[Dict[str, Any]]:
    runner = _get_runner(project_root)
    return [s.to_dict() for s in
            runner.store.list_sessions(project_root=project_root)]


@router.get("/{sid}", response_model=AgentSessionState)
async def get_session(sid: str,
                      project_root: Optional[str] = Query(default=None)
                      ) -> AgentSessionState:
    runner = _resolve_runner_for(sid) if project_root is None \
        else _get_runner(project_root)
    return _snapshot(runner, sid)


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
        await _drain_ready(runner, sid, deps)
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
            await _drain_ready(runner, sid, deps)
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
    runner = _resolve_runner_for(sid) if project_root is None \
        else _get_runner(project_root)
    removed = await asyncio.to_thread(runner.store.delete_session, sid)
    if not removed:
        raise HTTPException(status_code=404,
                            detail=f"session {sid!r} not found")
    return {"deleted": sid}


# --------------------- helpers ---------------------

def _resolve_runner_for(sid: str) -> SessionRunner:
    """Find which cached runner already knows about ``sid``.

    Falls back to the default (project_root=None) runner if nothing
    matches -- typical for sessions created without a project_root.
    """
    with _RUNNERS_LOCK:
        runners = list(_RUNNERS.values())
    for r in runners:
        if r.store.get_session(sid) is not None:
            return r
    return _get_runner(None)
