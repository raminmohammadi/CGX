

"""Skill CRUD -- list built-in + custom skills, create/update/delete custom ones.

Every skill's source is readable (built-ins via :func:`skills.read_skill_source`,
which locates their file with :mod:`inspect`); only user-authored custom
skills under ``~/.cgx/skills/`` can be created, edited, or deleted. Write
endpoints run the submitted source through
:func:`skills.loader.validate_skill_source` (syntax check, then a bounded
subprocess dry-import + ``detect()`` probe) before persisting anything.
"""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException

import skills as _skills
from skills.loader import (
    delete_custom_skill,
    delete_markdown_skill,
    read_custom_skill_source,
    read_markdown_skill_source,
    save_custom_skill,
    save_markdown_skill,
    validate_markdown_skill_source,
    validate_skill_source,
)
from cgx.webui.models import (
    MarkdownSkillCreateRequest,
    MarkdownSkillUpdateRequest,
    SkillCreateRequest,
    SkillSummary,
    SkillUpdateRequest,
)


router = APIRouter(tags=["skills"])


def _builtin_names() -> set:
    return {s.name.lower() for s in _skills.SKILLS}


def _markdown_summary(meta: dict) -> SkillSummary:
    """Build a listing entry for a freshly-saved global markdown skill."""
    return SkillSummary(
        name=meta.get("name", ""),
        role=meta.get("role", "infra"),
        aliases=list(meta.get("aliases", ()) or ()),
        description=meta.get("description", ""),
        is_custom=True,
        format="markdown",
        surfaces=list(meta.get("surfaces", ()) or ()),
        always_on=bool(meta.get("always_on", False)),
        scope="global",
    )


@router.get("/skills", response_model=List[SkillSummary])
def list_skills() -> List[SkillSummary]:
    return [SkillSummary(**d) for d in _skills.describe_skills()]


@router.get("/skills/{name}/source")
def get_skill_source(name: str) -> dict:
    src = _skills.read_skill_source(name)
    if src is None:
        raise HTTPException(status_code=404, detail=f"skill {name!r} not found")
    return {"name": name, "source": src}


@router.post("/skills", response_model=SkillSummary, status_code=201)
def create_skill(req: SkillCreateRequest) -> SkillSummary:
    result = validate_skill_source(req.source, known_names=_skills.known_skill_names())
    if not result.ok:
        raise HTTPException(
            status_code=422,
            detail={"error_kind": result.error_kind, "error_detail": result.error_detail},
        )
    meta = result.meta or {}
    save_custom_skill(meta["name"], req.source)
    return SkillSummary(**meta, is_custom=True)


@router.put("/skills/{name}", response_model=SkillSummary)
def update_skill(name: str, req: SkillUpdateRequest) -> SkillSummary:
    if name.lower() in _builtin_names():
        raise HTTPException(status_code=400, detail="cannot edit a built-in skill")
    if read_custom_skill_source(name) is None:
        raise HTTPException(status_code=404, detail=f"custom skill {name!r} not found")
    result = validate_skill_source(
        req.source, known_names=_skills.known_skill_names(exclude=name)
    )
    if not result.ok:
        raise HTTPException(
            status_code=422,
            detail={"error_kind": result.error_kind, "error_detail": result.error_detail},
        )
    meta = result.meta or {}
    if meta.get("name") != name:
        raise HTTPException(
            status_code=422,
            detail={
                "error_kind": "name_mismatch",
                "error_detail": (
                    f"submitted source defines name {meta.get('name')!r}, "
                    f"expected {name!r} -- rename via delete + create instead"
                ),
            },
        )
    save_custom_skill(name, req.source)
    return SkillSummary(**meta, is_custom=True)


@router.post("/skills/markdown", response_model=SkillSummary, status_code=201)
def create_markdown_skill(req: MarkdownSkillCreateRequest) -> SkillSummary:
    """Create a global markdown (``SKILL.md``) skill. No code is executed."""
    result = validate_markdown_skill_source(
        req.content, known_names=_skills.known_skill_names(), name_hint=req.name)
    if not result.ok:
        raise HTTPException(
            status_code=422,
            detail={"error_kind": result.error_kind, "error_detail": result.error_detail},
        )
    meta = result.meta or {}
    save_markdown_skill(meta["name"], req.content)
    return _markdown_summary(meta)


@router.put("/skills/markdown/{name}", response_model=SkillSummary)
def update_markdown_skill(name: str, req: MarkdownSkillUpdateRequest) -> SkillSummary:
    if name.lower() in _builtin_names():
        raise HTTPException(status_code=400, detail="cannot edit a built-in skill")
    if read_markdown_skill_source(name) is None:
        raise HTTPException(status_code=404, detail=f"markdown skill {name!r} not found")
    result = validate_markdown_skill_source(
        req.content, known_names=_skills.known_skill_names(exclude=name),
        name_hint=name)
    if not result.ok:
        raise HTTPException(
            status_code=422,
            detail={"error_kind": result.error_kind, "error_detail": result.error_detail},
        )
    meta = result.meta or {}
    if meta.get("name") != name:
        raise HTTPException(
            status_code=422,
            detail={
                "error_kind": "name_mismatch",
                "error_detail": (
                    f"frontmatter defines name {meta.get('name')!r}, "
                    f"expected {name!r} -- rename via delete + create instead"
                ),
            },
        )
    save_markdown_skill(name, req.content)
    return _markdown_summary(meta)


@router.delete("/skills/{name}")
def remove_skill(name: str) -> dict:
    if name.lower() in _builtin_names():
        raise HTTPException(status_code=400, detail="cannot delete a built-in skill")
    # A name is unique across formats (collision-checked on save), so try both
    # the Python-skill file and the markdown-skill directory.
    if delete_custom_skill(name) or delete_markdown_skill(name):
        return {"deleted": name}
    raise HTTPException(status_code=404, detail=f"custom skill {name!r} not found")
