

"""Site Studio backend: managed workspaces + one-shot generate/revise + preview.

The Studio does NOT drive the generic greenfield pipeline (which manufactures
thin, generic stubs for "build me a website"). Instead it calls a focused
single-shot generator that emits one self-contained ``index.html`` -- which
previews instantly, has no local files to break, and produces a real site.
Feedback revises that file in place. Preview serves the site read-only for a
sandboxed iframe.
"""

from __future__ import annotations

import logging
import os
from typing import List

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from cgx.answer.static_site_gen import generate_site, refine_site
from cgx.webui import workspace
from cgx.webui.helpers import build_provider, provider_from_profile_name
from cgx.webui.models import (
    GeneratedSite,
    ProviderConfig,
    SiteCreateRequest,
    SiteGenerateRequest,
    SiteInfo,
    SiteReviseRequest,
)
from cgx.webui.preview import serve_web_file

logger = logging.getLogger(__name__)

router = APIRouter(tags=["sites"])

# A full page is a few thousand tokens; the ProviderConfig default (1024)
# truncates it into a thin stub, so site generation forces a large budget.
_SITE_NUM_PREDICT = 8000


def _provider(pc: ProviderConfig):
    """Build an LLM provider from a request's ProviderConfig."""
    if pc.use_profile and pc.profile_name:
        return provider_from_profile_name(pc.profile_name)
    return build_provider(
        kind=pc.kind, model=pc.model, base_url=pc.base_url, api_key=pc.api_key,
        temperature=0.4, num_predict=max(int(pc.num_predict or 0), _SITE_NUM_PREDICT),
        num_ctx=pc.num_ctx, endpoint_path=pc.endpoint_path,
        allow_no_auth=pc.allow_no_auth)


def _write_index(project_root: str, html: str) -> int:
    path = os.path.join(project_root, "index.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return len(html.encode("utf-8"))


@router.get("/sites/themes")
def list_themes() -> dict:
    """Curated design themes for the Studio's theme picker."""
    from cgx.answer.web_themes import theme_summaries
    return {"themes": theme_summaries()}


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


@router.post("/sites/generate", response_model=GeneratedSite)
def generate(req: SiteGenerateRequest) -> GeneratedSite:
    name = (req.name or "").strip()
    brief = (req.brief or "").strip()
    if not name or not brief:
        raise HTTPException(status_code=400, detail="name and brief are required")
    try:
        info = workspace.create_site(name)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    try:
        prov = _provider(req.provider)
        theme_key = (req.theme or "").strip() or None
        html = generate_site(brief, prov, flavor=req.flavor, theme_key=theme_key)
        # Agentic design pass: critique the draft against the theme and
        # self-repair once before it reaches the user.
        html = refine_site(html, prov, flavor=req.flavor, theme_key=theme_key,
                           rounds=1)
    except Exception as e:  # noqa: BLE001
        logger.exception("site generate failed")
        raise HTTPException(status_code=502,
                            detail=f"generation failed: {type(e).__name__}: {e}") from e
    nbytes = _write_index(info["project_root"], html)
    return GeneratedSite(slug=info["slug"], project_root=info["project_root"],
                         entry="index.html", bytes=nbytes)


@router.post("/sites/revise", response_model=GeneratedSite)
def revise(req: SiteReviseRequest) -> GeneratedSite:
    feedback = (req.feedback or "").strip()
    if not feedback:
        raise HTTPException(status_code=400, detail="feedback is required")
    try:
        root = workspace.site_path(req.slug)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    index = root / "index.html"
    if not index.is_file():
        raise HTTPException(status_code=404, detail="site has no index.html to revise")
    current = index.read_text(encoding="utf-8", errors="replace")
    try:
        html = generate_site("", _provider(req.provider), flavor=req.flavor,
                             theme_key=(req.theme or "").strip() or None,
                             current_html=current, feedback=feedback)
    except Exception as e:  # noqa: BLE001
        logger.exception("site revise failed")
        raise HTTPException(status_code=502,
                            detail=f"revision failed: {type(e).__name__}: {e}") from e
    nbytes = _write_index(str(root), html)
    return GeneratedSite(slug=req.slug, project_root=str(root),
                         entry="index.html", bytes=nbytes)


@router.get("/sites/{slug}/preview", include_in_schema=False, response_model=None)
@router.get("/sites/{slug}/preview/{path:path}", include_in_schema=False,
            response_model=None)
def preview(slug: str, path: str = "") -> Response:
    try:
        root = workspace.site_path(slug)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return serve_web_file(os.path.realpath(str(root)), path)
