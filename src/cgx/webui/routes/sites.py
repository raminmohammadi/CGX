

"""Managed site-workspace CRUD for the Site Studio.

Lets the UI create a named site (folder under ~/.cgx/sites/<slug>) without
the user typing an absolute path, and list existing ones to reopen. The
actual build runs as a normal agent session pointed at the returned
project_root.
"""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException

from cgx.webui import workspace
from cgx.webui.models import SiteCreateRequest, SiteInfo

router = APIRouter(tags=["sites"])


@router.get("/sites", response_model=List[SiteInfo])
def list_sites() -> List[SiteInfo]:
    return [SiteInfo(**s) for s in workspace.list_sites()]


@router.post("/sites", response_model=SiteInfo, status_code=201)
def create_site(req: SiteCreateRequest) -> SiteInfo:
    name = (req.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    try:
        info = workspace.create_site(name)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return SiteInfo(slug=info["slug"], project_root=info["project_root"],
                    has_index=False, modified=0.0)
