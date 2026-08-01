

"""Agent Profile CRUD -- list / upsert / delete saved {task, skills} bundles.

Distinct from ``routes/profiles.py`` (LLM connection presets). See
:mod:`cgx.answer.agent_profiles` for the storage layer.
"""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException

from cgx.answer.agent_profiles import (
    AgentProfile,
    delete_agent_profile,
    list_agent_profiles,
    save_agent_profile,
)
from cgx.webui.models import AgentProfileSummary, AgentProfileUpsertRequest


router = APIRouter(tags=["agent-profiles"])


def _to_summary(p: AgentProfile) -> AgentProfileSummary:
    return AgentProfileSummary(
        name=p.name, objective=p.objective, project_root=p.project_root,
        mode=p.mode, skills=list(p.skills),
    )


@router.get("/agent-profiles", response_model=List[AgentProfileSummary])
def get_agent_profiles() -> List[AgentProfileSummary]:
    return [_to_summary(p) for p in list_agent_profiles()]


@router.put("/agent-profiles/{name}", response_model=AgentProfileSummary)
def upsert_agent_profile(name: str, req: AgentProfileUpsertRequest) -> AgentProfileSummary:
    if not name.strip():
        raise HTTPException(status_code=400, detail="agent profile name is required")
    if req.name != name:
        req = req.model_copy(update={"name": name})
    if not req.objective.strip():
        raise HTTPException(status_code=400, detail="objective is required")
    profile = AgentProfile(
        name=req.name.strip(),
        objective=req.objective.strip(),
        project_root=req.project_root.strip(),
        mode=req.mode,
        skills=list(req.skills),
    )
    save_agent_profile(profile)
    return _to_summary(profile)


@router.delete("/agent-profiles/{name}")
def remove_agent_profile(name: str) -> dict:
    ok = delete_agent_profile(name)
    if not ok:
        raise HTTPException(status_code=404, detail=f"agent profile {name!r} not found")
    return {"deleted": name}
