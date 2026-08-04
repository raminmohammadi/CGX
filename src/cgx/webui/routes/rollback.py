

"""Rollback REST route -- undo a prior :func:`apply_diffs_to_disk` run.

The agent's APPLY task records its per-run backup directory in the
response payload (``output.backup_dir``). The UI surfaces this path to
the user as an "Undo" button; on click it POSTs the project root + the
backup directory here and we walk the mirror to restore originals.
"""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, HTTPException

from cgx.codegen.disk_apply import rollback_from_backup
from cgx.logging_setup import sanitize_for_log
from cgx.webui.models import RollbackRequest

logger = logging.getLogger(__name__)
router = APIRouter(tags=["rollback"])


@router.post("/rollback")
def rollback(req: RollbackRequest) -> dict:
    if not (req.project_root or "").strip():
        raise HTTPException(status_code=400, detail="project_root is required")
    if not (req.backup_dir or "").strip():
        raise HTTPException(status_code=400, detail="backup_dir is required")

    workspace_root = (os.environ.get("CGX_WORKSPACE_ROOT") or "").strip()
    if not workspace_root:
        raise HTTPException(status_code=500, detail="CGX_WORKSPACE_ROOT is not configured")

    workspace_root_real = os.path.realpath(workspace_root)
    project_root_real = os.path.realpath(req.project_root)
    if not (
        project_root_real == workspace_root_real
        or project_root_real.startswith(workspace_root_real + os.sep)
    ):
        raise HTTPException(status_code=400, detail="project_root is outside allowed workspace")

    backup_dir_value = req.backup_dir.strip()
    if os.path.isabs(backup_dir_value):
        backup_real = os.path.realpath(backup_dir_value)
        if not (
            backup_real == project_root_real
            or backup_real.startswith(project_root_real + os.sep)
        ):
            raise HTTPException(status_code=400, detail="backup_dir is outside project_root")
        safe_backup_dir = backup_real
    else:
        safe_backup_dir = backup_dir_value

    try:
        result = rollback_from_backup(project_root_real, safe_backup_dir)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    logger.info("rollback: restored=%d deleted=%d failed=%d err=%r",
                len(result.get("restored_files") or []),
                len(result.get("deleted_files") or []),
                len(result.get("failed_files") or []),
                sanitize_for_log(result.get("error")))
    return result
