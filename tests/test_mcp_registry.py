"""Tests for the MCP config loader and tool registration/degradation."""

import json

from cgx.mcp.config import (
    MCPServerConfig, enabled_servers, load_mcp_config)
from cgx.mcp.manager import call_tool, list_servers, list_tools, register_mcp_tools


def _write_config(tmp_path, servers):
    p = tmp_path / "mcp.json"
    p.write_text(json.dumps({"servers": servers}), encoding="utf-8")
    return p


def test_load_missing_config_is_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("CGX_MCP_CONFIG", str(tmp_path / "nope.json"))
    assert load_mcp_config() == []


def test_load_and_filter_enabled(tmp_path, monkeypatch):
    _write_config(tmp_path, [
        {"name": "fetch", "transport": "stdio", "command": "uvx",
         "args": ["mcp-server-fetch"], "enabled": True},
        {"name": "off", "transport": "http", "url": "http://x", "enabled": False},
    ])
    monkeypatch.setenv("CGX_MCP_CONFIG", str(tmp_path / "mcp.json"))
    all_servers = load_mcp_config()
    assert {s.name for s in all_servers} == {"fetch", "off"}
    assert [s.name for s in enabled_servers()] == ["fetch"]


def test_bearer_auth_header(monkeypatch):
    monkeypatch.setenv("DOCS_TOKEN", "secret123")
    s = MCPServerConfig.from_dict({
        "name": "docs", "transport": "http", "url": "http://x",
        "auth": {"type": "bearer", "token_env": "DOCS_TOKEN"}})
    assert s.auth_header() == {"Authorization": "Bearer secret123"}


def test_list_servers_no_config(tmp_path, monkeypatch):
    monkeypatch.setenv("CGX_MCP_CONFIG", str(tmp_path / "none.json"))
    out = list_servers({}, None)
    assert "No MCP servers configured" in out


def test_list_servers_lists_configured(tmp_path, monkeypatch):
    _write_config(tmp_path, [
        {"name": "fetch", "transport": "stdio", "command": "uvx", "args": []}])
    monkeypatch.setenv("CGX_MCP_CONFIG", str(tmp_path / "mcp.json"))
    out = list_servers({}, None)
    assert "fetch" in out


def test_call_unknown_server(tmp_path, monkeypatch):
    monkeypatch.setenv("CGX_MCP_CONFIG", str(tmp_path / "mcp.json"))
    _write_config(tmp_path, [])
    assert "Unknown or disabled" in call_tool(
        {"server": "ghost", "tool": "x"}, None)


def test_register_mcp_tools_idempotent():
    from cgx.session.tasks.tool_registry import REGISTRY
    register_mcp_tools()
    register_mcp_tools()
    for name in ("mcp_list_servers", "mcp_list_tools", "mcp_call"):
        assert REGISTRY.get(name) is not None


def test_mcp_tools_advertised_only_when_configured(tmp_path, monkeypatch):
    from cgx.session.tasks.swarm_tools import mcp_tools_if_configured
    # No config -> not advertised.
    monkeypatch.setenv("CGX_MCP_CONFIG", str(tmp_path / "empty.json"))
    _write_config(tmp_path, [])
    # need a distinct empty file
    (tmp_path / "empty.json").write_text('{"servers": []}', encoding="utf-8")
    assert mcp_tools_if_configured() == ()
    # With a server -> advertised.
    _write_config(tmp_path, [
        {"name": "fetch", "transport": "stdio", "command": "uvx", "args": []}])
    monkeypatch.setenv("CGX_MCP_CONFIG", str(tmp_path / "mcp.json"))
    assert "mcp_call" in mcp_tools_if_configured()
