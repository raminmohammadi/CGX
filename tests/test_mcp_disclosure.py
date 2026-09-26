"""Tests for MCP tiered disclosure: Tier-2 schema + pre-dispatch arg validation."""

from __future__ import annotations

import cgx.mcp.manager as manager

_SCHEMA = {
    "type": "object",
    "properties": {"n": {"type": "number"}, "name": {"type": "string"}},
    "required": ["n"],
}


def _seed(monkeypatch):
    monkeypatch.setattr(manager, "_TOOL_SCHEMAS", {"srv": {"t": _SCHEMA}})


def test_args_validation_conforms(monkeypatch):
    _seed(monkeypatch)
    assert manager._args_schema_violations("srv", "t", {"n": 1}) == []


def test_args_validation_missing_required(monkeypatch):
    _seed(monkeypatch)
    errs = manager._args_schema_violations("srv", "t", {"name": "x"})
    assert any("n" in e for e in errs)


def test_args_validation_wrong_type(monkeypatch):
    _seed(monkeypatch)
    errs = manager._args_schema_violations("srv", "t", {"n": "not-a-number"})
    assert any("number" in e for e in errs)


def test_args_validation_no_schema_is_noop(monkeypatch):
    _seed(monkeypatch)
    assert manager._args_schema_violations("srv", "unknown", {}) == []
    assert manager._args_schema_violations("other", "t", {}) == []


def test_describe_tool_returns_cached_schema(monkeypatch):
    _seed(monkeypatch)
    out = manager.describe_tool({"server": "srv", "tool": "t"}, None)
    assert "srv/t" in out
    assert '"type": "object"' in out
    assert '"required"' in out


def test_describe_tool_requires_tool_name(monkeypatch):
    _seed(monkeypatch)
    assert manager.describe_tool({"server": "srv"}, None).startswith(
        "mcp_describe_tool requires")


def test_describe_tool_unknown_server_reports_missing(monkeypatch):
    monkeypatch.setattr(manager, "_TOOL_SCHEMAS", {})
    # Unknown/unconfigured server: list_tools returns an error, cache stays
    # empty, and describe_tool reports the missing schema (no live SDK needed).
    out = manager.describe_tool({"server": "nope", "tool": "t"}, None)
    assert "No schema" in out


def test_register_adds_describe_tool():
    from cgx.session.tasks.tool_registry import REGISTRY
    manager.register_mcp_tools()
    assert REGISTRY.get("mcp_describe_tool") is not None
    assert REGISTRY.get("mcp_call") is not None
