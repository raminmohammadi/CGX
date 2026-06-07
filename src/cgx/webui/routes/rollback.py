

"""Rollback REST route -- undo a prior :func:`apply_diffs_to_disk` run.

The agent's APPLY task records its per-run backup directory in the
response payload (``output.backup_dir``). The UI surfaces this path to
the user as an "Undo" button; on click it POSTs the project root + the
backup directory here and we walk the mirror to restore originals.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from cgx.codegen.disk_apply import rollback_from_backup
from cgx.webui.models import RollbackRequest

logger = logging.getLogger(__name__)
router = APIRouter(tags=["rollback"])


@router.post("/rollback")
def rollback(req: RollbackRequest) -> dict:
    if not (req.project_root or "").strip():
        raise HTTPException(status_code=400, detail="project_root is required")
    if not (req.backup_dir or "").strip():
        raise HTTPException(status_code=400, detail="backup_dir is required")
    try:
        result = rollback_from_backup(req.project_root, req.backup_dir)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    logger.info("rollback: restored=%d deleted=%d failed=%d err=%r",
                len(result.get("restored_files") or []),
                len(result.get("deleted_files") or []),
                len(result.get("failed_files") or []),
                result.get("error"))
    return result
