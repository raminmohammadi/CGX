"""Tests for the Agent Profile store and its ``/api/agent-profiles`` routes.

Unlike ``tests/test_profiles.py`` (which reloads ``cgx.answer.profiles``
under a monkeypatched ``CGX_CONFIG_DIR``), this patches the module's path
constants *in place* instead of reloading: ``cgx.webui.routes.agent_profiles``
imports ``save_agent_profile``/etc. by reference at its own import time, so
a reload of ``cgx.answer.agent_profiles`` would leave those already-bound
names pointing at the pre-reload module -- silently writing to the real
``~/.cgx`` instead of the test's tmp dir. In-place attribute patching
avoids that hazard since both modules share the same function objects.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException


@pytest.fixture()
def agent_profiles_module(tmp_path, monkeypatch):
    import cgx.answer.agent_profiles as agent_profiles
    monkeypatch.setattr(agent_profiles, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(agent_profiles, "AGENT_PROFILES_PATH",
                        tmp_path / "agent_profiles.json")
    return agent_profiles


def test_save_and_list_agent_profile_roundtrip(agent_profiles_module):
    AP = agent_profiles_module
    AP.save_agent_profile(AP.AgentProfile(
        name="graphql-api", objective="Add a GraphQL API",
        project_root="/repo", mode="explore", skills=["graphql", "fastapi"],
    ))
    profiles = AP.list_agent_profiles()
    assert [p.name for p in profiles] == ["graphql-api"]
    got = AP.get_agent_profile("graphql-api")
    assert got is not None
    assert got.skills == ["graphql", "fastapi"]
    assert got.mode == "explore"


def test_delete_agent_profile(agent_profiles_module):
    AP = agent_profiles_module
    AP.save_agent_profile(AP.AgentProfile(name="p1", objective="do a thing"))
    assert AP.delete_agent_profile("p1") is True
    assert AP.get_agent_profile("p1") is None
    assert AP.delete_agent_profile("p1") is False


def test_agent_profile_defaults(agent_profiles_module):
    AP = agent_profiles_module
    AP.save_agent_profile(AP.AgentProfile(name="minimal", objective="do it"))
    got = AP.get_agent_profile("minimal")
    assert got.project_root == ""
    assert got.mode == ""
    assert got.skills == []


# --------------------------- route layer ---------------------------

def test_upsert_agent_profile_route(agent_profiles_module):
    from cgx.webui.models import AgentProfileUpsertRequest
    from cgx.webui.routes import agent_profiles as routes

    result = routes.upsert_agent_profile(
        "graphql-api",
        AgentProfileUpsertRequest(
            name="graphql-api", objective="Add a GraphQL API",
            project_root="/repo", mode="explore", skills=["graphql"],
        ),
    )
    assert result.name == "graphql-api"
    assert result.skills == ["graphql"]
    assert {p.name for p in routes.get_agent_profiles()} == {"graphql-api"}


def test_upsert_agent_profile_requires_objective(agent_profiles_module):
    from cgx.webui.models import AgentProfileUpsertRequest
    from cgx.webui.routes import agent_profiles as routes

    with pytest.raises(HTTPException) as exc:
        routes.upsert_agent_profile(
            "p1", AgentProfileUpsertRequest(name="p1", objective="   "))
    assert exc.value.status_code == 400


def test_remove_agent_profile_route_404_when_missing(agent_profiles_module):
    from cgx.webui.routes import agent_profiles as routes

    with pytest.raises(HTTPException) as exc:
        routes.remove_agent_profile("does-not-exist")
    assert exc.value.status_code == 404
