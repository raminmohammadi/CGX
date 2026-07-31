"""Route-handler tests for ``/api/skills``.

Calls the route functions directly (matching the convention used by
``tests/test_webui_settings.py``) rather than spinning up a TestClient.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

import skills as _skills
from skills import loader as skill_loader
from cgx.webui.models import SkillCreateRequest, SkillUpdateRequest
from cgx.webui.routes import skills as routes


_VALID_SOURCE = '''
from skills.base import Skill

class GraphQLSkill(Skill):
    name = "graphql"
    role = "backend"
    aliases = ("GraphQL",)
    description = "GraphQL API layer."

    def detect(self, goal: str) -> float:
        return 0.9 if "graphql" in (goal or "").lower() else 0.0
'''


@pytest.fixture()
def custom_skills_dir(tmp_path, monkeypatch):
    d = tmp_path / "skills"
    d.mkdir()
    monkeypatch.setattr(skill_loader, "CUSTOM_SKILLS_DIR", d)
    monkeypatch.setattr(skill_loader, "_cache_signature", None)
    monkeypatch.setattr(skill_loader, "_cache", [])
    return d


def test_list_skills_includes_builtins(custom_skills_dir):
    out = routes.list_skills()
    names = {s.name for s in out}
    assert "react" in names and "fastapi" in names
    assert all(not s.is_custom for s in out)


def test_create_skill_persists_and_lists(custom_skills_dir):
    created = routes.create_skill(SkillCreateRequest(source=_VALID_SOURCE))
    assert created.name == "graphql"
    assert created.is_custom is True

    names = {s.name for s in routes.list_skills()}
    assert "graphql" in names


def test_create_skill_rejects_syntax_error(custom_skills_dir):
    with pytest.raises(HTTPException) as exc:
        routes.create_skill(SkillCreateRequest(source="def broken(:"))
    assert exc.value.status_code == 422
    assert exc.value.detail["error_kind"] == "syntax_error"


def test_create_skill_rejects_builtin_name_collision(custom_skills_dir):
    source = _VALID_SOURCE.replace('name = "graphql"', 'name = "react"')
    with pytest.raises(HTTPException) as exc:
        routes.create_skill(SkillCreateRequest(source=source))
    assert exc.value.status_code == 422
    assert exc.value.detail["error_kind"] == "name_collision"


def test_get_skill_source_roundtrips_custom_skill(custom_skills_dir):
    routes.create_skill(SkillCreateRequest(source=_VALID_SOURCE))
    got = routes.get_skill_source("graphql")
    assert got["source"] == _VALID_SOURCE


def test_get_skill_source_readable_for_builtin(custom_skills_dir):
    got = routes.get_skill_source("react")
    assert got["name"] == "react"
    assert "class ReactSkill" in got["source"]


def test_get_skill_source_404_for_unknown(custom_skills_dir):
    with pytest.raises(HTTPException) as exc:
        routes.get_skill_source("does-not-exist")
    assert exc.value.status_code == 404


def test_update_skill_overwrites_source(custom_skills_dir):
    routes.create_skill(SkillCreateRequest(source=_VALID_SOURCE))
    updated_source = _VALID_SOURCE.replace(
        'description = "GraphQL API layer."',
        'description = "GraphQL API layer (v2)."',
    )
    updated = routes.update_skill("graphql", SkillUpdateRequest(source=updated_source))
    assert updated.description == "GraphQL API layer (v2)."


def test_update_skill_rejects_builtin(custom_skills_dir):
    with pytest.raises(HTTPException) as exc:
        routes.update_skill("react", SkillUpdateRequest(source=_VALID_SOURCE))
    assert exc.value.status_code == 400


def test_delete_skill_rejects_builtin(custom_skills_dir):
    with pytest.raises(HTTPException) as exc:
        routes.remove_skill("react")
    assert exc.value.status_code == 400


def test_delete_skill_404_for_unknown(custom_skills_dir):
    with pytest.raises(HTTPException) as exc:
        routes.remove_skill("does-not-exist")
    assert exc.value.status_code == 404


def test_delete_custom_skill_removes_it(custom_skills_dir):
    routes.create_skill(SkillCreateRequest(source=_VALID_SOURCE))
    result = routes.remove_skill("graphql")
    assert result == {"deleted": "graphql"}
    assert "graphql" not in {s.name for s in routes.list_skills()}
