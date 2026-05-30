"""Plan SSE route — streams sketch thoughts → final plan + diffs."""

from __future__ import annotations

import logging

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from cgx.webui import task_store
from cgx.webui.handlers import stream_plan
from cgx.webui.models import PlanRequest
from cgx.webui.sse import bridge_generator

logger = logging.getLogger(__name__)
router = APIRouter(tags=["plan"])


@router.post("/plan")
async def plan(req: PlanRequest) -> EventSourceResponse:
    pcfg = req.provider
    idx = req.index

    task_id = task_store.create_task("plan", {
        "task": req.task,
        "model": pcfg.model,
        "self_test": req.self_test,
        "project_root": req.project_root,
    })
    cancel_event = task_store.get_cancel_event(task_id)
    logger.info("plan route: task_id=%s task=%r", task_id, req.task[:60])

    def factory():
        return stream_plan(
            index_dir=idx.index_dir, records=idx.records,
            task=req.task, embed_model=idx.embed_model,
            use_profile=pcfg.use_profile, profile_name=pcfg.profile_name,
            kind=pcfg.kind, model=pcfg.model, base_url=pcfg.base_url,
            api_key=pcfg.api_key, temperature=pcfg.temperature,
            num_predict=pcfg.num_predict,
            self_test=req.self_test, run_tests=req.run_tests,
            project_root=req.project_root,
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
