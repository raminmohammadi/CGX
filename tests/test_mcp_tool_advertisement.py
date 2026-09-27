"""MCP tool-advertisement guard.

MCP tools (``mcp_list_servers`` / ``mcp_list_tools`` / ``mcp_call``) are
registered in the shared tool registry unconditionally, but are only
*advertised* to an agent role when at least one MCP server is enabled in
``~/.cgx/mcp.json``. Every agent context that runs a tool-calling loop and can
research/fetch -- the Swarm developer (``_dev_tools``, also used by the
Greenfield build via ``swarm_developer``) and the tech-lead planner
(``_planner_tools``) -- must advertise them when configured, and must fall back
to the built-in ``fetch_url`` when not.

(Non-tool-loop tasks -- explore/investigate/ask, plan_change/apply, the
deterministic scaffolder -- have no tool loop, so MCP does not apply there.)

This test pins that contract so a refactor can't silently drop MCP from a
generation/planning role.
"""

from __future__ import annotations

import pytest

_MCP_TOOLS = {"mcp_list_servers", "mcp_list_tools", "mcp_call"}


@pytest.fixture
def mcp_enabled(monkeypatch):
    # mcp_tools_if_configured() imports enabled_servers lazily, so patching the
    # source symbol is enough. Return a non-empty list to simulate a server.
    monkeypatch.setattr("cgx.mcp.config.enabled_servers",
                        lambda *a, **k: [object()], raising=True)


@pytest.fixture
def mcp_disabled(monkeypatch):
    monkeypatch.setattr("cgx.mcp.config.enabled_servers",
                        lambda *a, **k: [], raising=True)


def test_dev_tools_advertise_mcp_when_configured(mcp_enabled):
    from cgx.session.tasks.swarm_generate import _dev_tools
    tools = set(_dev_tools(None))
    assert _MCP_TOOLS <= tools, "developer role must advertise MCP tools when a server is enabled"
    assert "fetch_url" not in tools, "fetch_url should be dropped in favor of mcp_call"


def test_dev_tools_fallback_without_mcp(mcp_disabled):
    from cgx.session.tasks.swarm_generate import _dev_tools
    tools = set(_dev_tools(None))
    assert not (_MCP_TOOLS & tools), "no MCP tools should be advertised when no server is enabled"
    assert "fetch_url" in tools, "fetch_url is the fallback fetch path without MCP"


def test_planner_tools_advertise_mcp_when_configured(mcp_enabled):
    from cgx.session.tasks.swarm_tech_lead import _planner_tools
    tools = set(_planner_tools())
    assert _MCP_TOOLS <= tools, "tech-lead planner must advertise MCP tools when a server is enabled"
    assert "fetch_url" not in tools
    assert "search_web" in tools, "search_web stays for URL discovery"


def test_planner_tools_fallback_without_mcp(mcp_disabled):
    from cgx.session.tasks.swarm_tech_lead import _planner_tools
    tools = set(_planner_tools())
    assert not (_MCP_TOOLS & tools)
    assert {"search_web", "fetch_url"} <= tools
