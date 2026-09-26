"""Route-handler tests for the markdown-skill endpoints on ``/api/skills``."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from skills import loader as skill_loader
from cgx.webui.models import (
    MarkdownSkillCreateRequest,
    MarkdownSkillUpdateRequest,
)
from cgx.webui.routes import skills as routes


_VALID_MD = (
    "---\n"
    "name: refunds\n"
    "description: How refunds work here.\n"
    "triggers: [refund, chargeback]\n"
    "surfaces: [chat]\n"
    "---\n"
    "Every refund must reference an original charge id."
)


@pytest.fixture()
def md_dir(tmp_path, monkeypatch):
    d = tmp_path / "skills"
    d.mkdir()
    monkeypatch.setattr(skill_loader, "CUSTOM_SKILLS_DIR", d)
    monkeypatch.setattr(skill_loader, "_md_cache", {})
    monkeypatch.setattr(skill_loader, "_cache_signature", None)
    monkeypatch.setattr(skill_loader, "_cache", [])
    return d


def test_create_markdown_skill_persists_and_lists(md_dir):
    created = routes.create_markdown_skill(MarkdownSkillCreateRequest(content=_VALID_MD))
    assert created.name == "refunds"
    assert created.format == "markdown"
    assert created.is_custom is True
    assert "chat" in created.surfaces

    listed = {s.name: s for s in routes.list_skills()}
    assert listed["refunds"].format == "markdown"


def test_get_source_returns_markdown(md_dir):
    routes.create_markdown_skill(MarkdownSkillCreateRequest(content=_VALID_MD))
    out = routes.get_skill_source("refunds")
    assert "original charge id" in out["source"]


def test_create_markdown_rejects_empty_body(md_dir):
    with pytest.raises(HTTPException) as exc:
        routes.create_markdown_skill(
            MarkdownSkillCreateRequest(content="---\nname: x\n---\n   "))
    assert exc.value.status_code == 422
    assert exc.value.detail["error_kind"] == "empty_body"


def test_create_markdown_rejects_collision_with_builtin(md_dir):
    with pytest.raises(HTTPException) as exc:
        routes.create_markdown_skill(
            MarkdownSkillCreateRequest(content="---\nname: react\n---\nBody."))
    assert exc.value.detail["error_kind"] == "name_collision"


def test_update_markdown_skill(md_dir):
    routes.create_markdown_skill(MarkdownSkillCreateRequest(content=_VALID_MD))
    updated = routes.update_markdown_skill(
        "refunds",
        MarkdownSkillUpdateRequest(content=_VALID_MD.replace("original charge", "ORDER")))
    assert updated.name == "refunds"
    assert "ORDER id" in routes.get_skill_source("refunds")["source"]


def test_update_markdown_rejects_rename(md_dir):
    routes.create_markdown_skill(MarkdownSkillCreateRequest(content=_VALID_MD))
    with pytest.raises(HTTPException) as exc:
        routes.update_markdown_skill(
            "refunds",
            MarkdownSkillUpdateRequest(content="---\nname: other\n---\nBody."))
    assert exc.value.detail["error_kind"] == "name_mismatch"


def test_update_markdown_404_for_missing(md_dir):
    with pytest.raises(HTTPException) as exc:
        routes.update_markdown_skill(
            "ghost", MarkdownSkillUpdateRequest(content="---\nname: ghost\n---\nB."))
    assert exc.value.status_code == 404


def test_delete_handles_markdown(md_dir):
    routes.create_markdown_skill(MarkdownSkillCreateRequest(content=_VALID_MD))
    out = routes.remove_skill("refunds")
    assert out == {"deleted": "refunds"}
    with pytest.raises(HTTPException):
        routes.get_skill_source("refunds")


def test_delete_builtin_rejected(md_dir):
    with pytest.raises(HTTPException) as exc:
        routes.remove_skill("react")
    assert exc.value.status_code == 400
