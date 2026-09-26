

"""Read/write a project's CGX.md context file (the CLAUDE.md analogue).

The chatbot and agents inject this file's contents on every turn for the
project it lives in (see :mod:`cgx.answer.instructions`). These endpoints
let the UI show and edit it. Writes always target a fixed filename
(``CGX.md``) at the given ``project_root`` -- or the existing context file
if one is already present -- so the filename itself can't be used for path
traversal. ``project_root`` is a user-supplied local path, consistent with
the rest of the agent surface's local-first, accepted-risk posture.
"""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, HTTPException, Query

from cgx import context_files as _cf
from cgx.webui.models import ContextFileWriteRequest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["context-file"])

# Same generous ceiling the loader applies; keeps a pathological paste out
# of the repo. The injection layer trims further to the model's budget.
_MAX_WRITE_CHARS = 64_000


def _require_root(project_root: str) -> str:
    root = (project_root or "").strip()
    if not root:
        raise HTTPException(status_code=400, detail="project_root is required")
    return root


@router.get("/context-file")
def get_context_file(project_root: str = Query("")) -> dict:
    """Return the project's context file (content + which candidate matched)."""
    root = _require_root(project_root)
    found = _cf.find_context_file(root)
    if found is not None:
        try:
            content = found.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            raise HTTPException(status_code=500,
                                detail=f"could not read context file: {e}") from e
        return {
            "project_root": root,
            "exists": True,
            "path": str(found),
            "filename": found.name,
            "content": content,
            "candidates": list(_cf.CONTEXT_FILE_CANDIDATES),
        }
    default = _cf.default_context_path(root)
    return {
        "project_root": root,
        "exists": False,
        "path": str(default),
        "filename": _cf.DEFAULT_CONTEXT_FILENAME,
        "content": "",
        "candidates": list(_cf.CONTEXT_FILE_CANDIDATES),
    }


@router.put("/context-file")
def write_context_file(req: ContextFileWriteRequest) -> dict:
    """Create or replace the project's context file."""
    root = _require_root(req.project_root)
    if len(req.content or "") > _MAX_WRITE_CHARS:
        raise HTTPException(
            status_code=422,
            detail={"error_kind": "too_large",
                    "error_detail": f"context file exceeds {_MAX_WRITE_CHARS} characters"})
    # Write to the existing file if one is present, else the default CGX.md.
    target = _cf.find_context_file(root) or _cf.default_context_path(root)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(req.content, encoding="utf-8")
    except OSError as e:
        raise HTTPException(status_code=500,
                            detail=f"could not write context file: {e}") from e
    _cf.clear_cache()
    return {"path": str(target), "filename": target.name,
            "bytes": len(req.content.encode("utf-8"))}


@router.delete("/context-file")
def delete_context_file(project_root: str = Query("")) -> dict:
    """Delete the project's context file, if present."""
    root = _require_root(project_root)
    found = _cf.find_context_file(root)
    if found is None:
        raise HTTPException(status_code=404, detail="no context file to delete")
    try:
        os.remove(found)
    except OSError as e:
        raise HTTPException(status_code=500,
                            detail=f"could not delete context file: {e}") from e
    _cf.clear_cache()
    return {"deleted": str(found)}
