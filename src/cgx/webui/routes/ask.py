"""Ask SSE route — streams intent → thought deltas → final answer."""

from __future__ import annotations

import logging

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from cgx.webui import task_store
from cgx.webui.handlers import stream_ask
from cgx.webui.models import AskRequest
from cgx.webui.sse import bridge_generator

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ask"])


@router.post("/ask")
async def ask(req: AskRequest) -> EventSourceResponse:
    pcfg = req.provider
    idx = req.index

    task_id = task_store.create_task("ask", {
        "question": req.question,
        "model": pcfg.model,
        "session_id": req.session_id,
    })
    cancel_event = task_store.get_cancel_event(task_id)
    logger.info("ask route: task_id=%s question=%r", task_id, req.question[:60])

    final: dict = {"answer_md": "", "sources": [], "meta": {}, "intent": {}}

    def factory():
        for ev, payload in stream_ask(
            index_dir=idx.index_dir, records=idx.records,
            question=req.question, embed_model=idx.embed_model,
            use_profile=pcfg.use_profile, profile_name=pcfg.profile_name,
            kind=pcfg.kind, model=pcfg.model, base_url=pcfg.base_url,
            api_key=pcfg.api_key, temperature=pcfg.temperature,
            num_predict=pcfg.num_predict,
            endpoint_path=getattr(pcfg, "endpoint_path", "/v1/chat/completions"),
            allow_no_auth=bool(getattr(pcfg, "allow_no_auth", False)),
            cancel_event=cancel_event,
        ):
            if ev == "intent":
                final["intent"] = payload
            elif ev == "answer":
                final["answer_md"] = payload.get("answer_md", "")
                final["sources"] = payload.get("sources", [])
                final["meta"] = payload.get("meta", {})
            yield ev, payload

        # Session persistence.
        sid = (req.session_id or "").strip()
        if sid and req.question.strip() and final["answer_md"]:
            try:
                from cgx import sessions as _sessions
                _sessions.append_message(sid, "user", req.question)
                _sessions.append_message(
                    sid, "assistant", final["answer_md"],
                    meta={"intent": final.get("intent"),
                          "sources": final.get("sources")},
                )
            except Exception as _e:
                logger.warning("session persistence failed for %s: %s", sid, _e)

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
