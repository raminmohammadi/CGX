# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ramin Mohammadi

"""Task registry REST endpoints.

GET  /api/tasks           — list recent tasks
GET  /api/tasks/{id}      — get task record + status
GET  /api/tasks/{id}/events — full event log for replay after tab switch
DELETE /api/tasks/{id}    — cancel a running task
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from cgx.webui import task_store

logger = logging.getLogger(__name__)
router = APIRouter(tags=["tasks"])


@router.get("/tasks")
def list_tasks(limit: int = 50):
    return {"tasks": task_store.list_tasks(limit=limit)}


@router.get("/tasks/{task_id}")
def get_task(task_id: str):
    task = task_store.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return task


@router.get("/tasks/{task_id}/events")
def get_task_events(task_id: str):
    task = task_store.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    events = task_store.get_task_events(task_id)
    return {"task_id": task_id, "status": task["status"], "events": events}


@router.delete("/tasks/{task_id}")
def cancel_task(task_id: str):
    cancelled = task_store.cancel_task(task_id)
    if not cancelled:
        raise HTTPException(status_code=404, detail="task not found or already finished")
    logger.info("tasks route: cancelled task id=%s", task_id)
    return {"task_id": task_id, "status": "cancelled"}
