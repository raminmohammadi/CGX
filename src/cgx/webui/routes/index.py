"""Index build: ZIP upload + SSE-streamed progress."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import uuid

from fastapi import APIRouter, File, UploadFile
from sse_starlette.sse import EventSourceResponse

from cgx.webui import task_store
from cgx.webui.handlers import stream_index
from cgx.webui.models import IndexBuildRequest
from cgx.webui.sse import bridge_generator

logger = logging.getLogger(__name__)
router = APIRouter(tags=["index"])


@router.post("/index/upload")
async def upload_zip(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        return {"error": "no filename"}
    tmpdir = tempfile.mkdtemp(prefix="averix_upload_")
    safe = f"{uuid.uuid4().hex[:8]}.zip"
    target = os.path.join(tmpdir, safe)
    with open(target, "wb") as f:
        shutil.copyfileobj(file.file, f)
    logger.info("index upload: saved %r -> %r (%d bytes)",
                file.filename, target, os.path.getsize(target))
    return {"path": target, "original_name": file.filename,
            "size_bytes": os.path.getsize(target)}


@router.post("/index/build")
async def build_index(req: IndexBuildRequest) -> EventSourceResponse:
    task_id = task_store.create_task("index", {
        "project_root": req.project_root,
        "out_dir": req.out_dir,
        "embed_model": req.embed_model,
    })
    cancel_event = task_store.get_cancel_event(task_id)
    logger.info("index build route: task_id=%s project_root=%r", task_id, req.project_root)

    def factory():
        return stream_index(
            project_root=req.project_root,
            out_dir=req.out_dir,
            embed_model=req.embed_model,
            metric=req.metric,
            index_type=req.index_type,
            zip_path=req.zip_path,
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
