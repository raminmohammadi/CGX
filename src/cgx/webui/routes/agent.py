"""Agent SSE route — streams Planner/Tracker/Judge events."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from cgx.webui import task_store
from cgx.webui.handlers import get_agent_plan, stream_agent
from cgx.webui.models import AgentRequest
from cgx.webui.sse import bridge_generator

logger = logging.getLogger(__name__)
router = APIRouter(tags=["agent"])


@router.post("/agent")
async def agent(req: AgentRequest) -> EventSourceResponse:
    pcfg = req.provider
    idx = req.index

    task_id = task_store.create_task("agent", {
        "goal": req.goal,
        "model": pcfg.model,
        "project_root": req.project_root,
    })
    cancel_event = task_store.get_cancel_event(task_id)
    logger.info("agent route: task_id=%s goal=%r", task_id, req.goal[:60])

    def factory():
        return stream_agent(
            index_dir=idx.index_dir, records=idx.records,
            goal=req.goal, embed_model=idx.embed_model,
            use_profile=pcfg.use_profile, profile_name=pcfg.profile_name,
            kind=pcfg.kind, model=pcfg.model, base_url=pcfg.base_url,
            api_key=pcfg.api_key, temperature=pcfg.temperature,
            num_predict=pcfg.num_predict,
            project_root=req.project_root, stop_on_fail=req.stop_on_fail,
            endpoint_path=getattr(pcfg, "endpoint_path", "/v1/chat/completions"),
            allow_no_auth=bool(getattr(pcfg, "allow_no_auth", False)),
            cancel_event=cancel_event,
        )

    def to_event(item):
        try:
            ev, payload = item
        except Exception:
            ev, payload = "message", {"raw": str(item)}
        return {"event": ev, "data": payload}

    return EventSourceResponse(
        bridge_generator(factory, to_event=to_event,
                         task_id=task_id, cancel_event=cancel_event)
    )


@router.post("/agent/plan")
async def agent_plan_only(req: AgentRequest) -> Dict[str, Any]:
    """Run the Planner only and return the plan as JSON — no task execution.

    Used by the *Review plan* and *Plan only* execution modes in the UI
    so the DAG panel can be shown before the user commits to running the
    full agent loop.
    """
    pcfg = req.provider
    idx = req.index
    result: Dict[str, Any] = await asyncio.to_thread(
        get_agent_plan,
        index_dir=idx.index_dir, records=idx.records,
        goal=req.goal, embed_model=idx.embed_model,
        use_profile=pcfg.use_profile, profile_name=pcfg.profile_name,
        kind=pcfg.kind, model=pcfg.model, base_url=pcfg.base_url,
        api_key=pcfg.api_key, temperature=pcfg.temperature,
        num_predict=pcfg.num_predict,
        project_root=req.project_root,
    )
    return result
