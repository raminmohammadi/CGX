

"""Profile CRUD — list / upsert / delete saved provider configurations.

API keys are sent in the upsert body and forwarded to
:func:`cgx.answer.profiles.save_profile` which persists them via the OS
keyring (or a permissioned file fallback). The list endpoint never
returns key material, only the ``has_api_key`` flag.
"""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException

from cgx.answer.profiles import (
    Profile,
    delete_profile,
    list_profiles,
    save_profile,
)
from cgx.webui.models import ProfileSummary, ProfileUpsertRequest


router = APIRouter(tags=["profiles"])


def _to_summary(p: Profile) -> ProfileSummary:
    return ProfileSummary(
        name=p.name, kind=p.kind, model=p.model, base_url=p.base_url,
        has_api_key=p.has_api_key, temperature=p.temperature,
        num_predict=p.num_predict,
        endpoint_path=getattr(p, "endpoint_path", "/v1/chat/completions"),
        allow_no_auth=bool(getattr(p, "allow_no_auth", False)),
    )


@router.get("/profiles", response_model=List[ProfileSummary])
def get_profiles() -> List[ProfileSummary]:
    return [_to_summary(p) for p in list_profiles()]


@router.put("/profiles/{name}", response_model=ProfileSummary)
def upsert_profile(name: str, req: ProfileUpsertRequest) -> ProfileSummary:
    if not name.strip():
        raise HTTPException(status_code=400, detail="profile name is required")
    if req.name != name:
        # Body and path name disagree — prefer the path for idempotency.
        req = req.model_copy(update={"name": name})
    try:
        p = Profile(
            name=req.name.strip(),
            kind=req.kind,
            model=req.model.strip(),
            base_url=req.base_url.strip(),
            temperature=float(req.temperature),
            num_predict=int(req.num_predict),
            endpoint_path=getattr(req, "endpoint_path", "/v1/chat/completions") or "/v1/chat/completions",
            allow_no_auth=bool(getattr(req, "allow_no_auth", False)),
        )
        save_profile(p, api_key=(req.api_key or None))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{type(e).__name__}: {e}")
    # Re-read to pick up the persisted has_api_key bit.
    for live in list_profiles():
        if live.name == p.name:
            return _to_summary(live)
    return _to_summary(p)


@router.delete("/profiles/{name}")
def remove_profile(name: str) -> dict:
    ok = delete_profile(name)
    if not ok:
        raise HTTPException(status_code=404, detail=f"profile {name!r} not found")
    return {"deleted": name}
