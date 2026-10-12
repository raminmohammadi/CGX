"""The curated MCP catalog, the roster write helpers, the secret store and the
shared ops used by both the web routes and the ``cgx mcp`` CLI."""

from __future__ import annotations

import json
import re

import pytest

from cgx.mcp import catalog as cat
from cgx.mcp import config as cfg
from cgx.mcp import ops
from cgx.mcp import secrets as sec

_PLACEHOLDER = re.compile(r"\$\{([A-Z0-9_]+)\}")


@pytest.fixture()
def roster(tmp_path, monkeypatch):
    """Isolated roster + secrets file; the OS keyring is never touched."""
    monkeypatch.setenv("CGX_MCP_CONFIG", str(tmp_path / "mcp.json"))
    monkeypatch.setenv("CGX_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(sec, "_keyring", lambda: None)
    for var in ("BRAVE_API_KEY", "GITHUB_PERSONAL_ACCESS_TOKEN", "SUPABASE_ACCESS_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


# --------------------------------------------------------------------------
# catalog integrity
# --------------------------------------------------------------------------

def test_catalog_entries_are_well_formed():
    ids = [e.id for e in cat.CATALOG]
    assert len(ids) == len(set(ids)), "duplicate catalog ids"
    assert len(ids) >= 18
    for e in cat.CATALOG:
        assert e.category in cat.CATEGORIES, e.id
        assert e.status in ("official", "vendor", "community", "archived"), e.id
        assert e.risk in ("read", "write", "execute"), e.id
        assert e.description and e.agent_hint and e.docs_url.startswith("https://"), e.id
        if e.transport == "http":
            assert e.url.startswith("https://") and not e.command, e.id
        else:
            assert e.command in ("npx", "uvx", "docker"), e.id
            assert e.args, e.id
        # every ${PLACEHOLDER} is either a declared param or a declared env var
        declared = {p["name"] for p in e.params} | {v["name"] for v in e.env}
        for text in list(e.args) + [e.url]:
            for ph in _PLACEHOLDER.findall(text or ""):
                assert ph in declared, (e.id, ph)
        if e.auth:
            assert e.auth.get("type") == "bearer" and e.auth.get("token_env") in declared, e.id


def test_instantiate_fills_params_and_keeps_env_references():
    entry = cat.catalog_entry("supabase")
    assert cat.missing_params(entry, {}) == ["PROJECT_REF"]
    server = cat.instantiate(entry, params={"PROJECT_REF": "abc123"})
    assert server["transport"] == "http" and "project_ref=abc123" in server["url"]
    assert server["auth"]["token_env"] == "SUPABASE_ACCESS_TOKEN" and "env" not in server
    assert server["catalog_id"] == "supabase" and server["agent_hint"]
    gh = cat.instantiate(cat.catalog_entry("github"))
    assert gh["transport"] == "http" and gh["auth"]["token_env"] == "GITHUB_PERSONAL_ACCESS_TOKEN"
    assert "env" not in gh
    # a stdio entry references only its REQUIRED vars, plus optional ones the user supplied
    ctx = cat.instantiate(cat.catalog_entry("context7"))
    assert ctx["env"] == {}
    ctx = cat.instantiate(cat.catalog_entry("context7"), env_values={"CONTEXT7_API_KEY": "k"})
    assert ctx["env"] == {"CONTEXT7_API_KEY": "${CONTEXT7_API_KEY}"}
    mongo = cat.instantiate(cat.catalog_entry("mongodb"))
    assert mongo["env"] == {"MDB_MCP_CONNECTION_STRING": "${MDB_MCP_CONNECTION_STRING}"} and "--readOnly" in mongo["args"]


# --------------------------------------------------------------------------
# roster + secrets
# --------------------------------------------------------------------------

def test_add_remove_toggle_roundtrip(roster):
    row = ops.add_from_request(catalog_id="fetch")
    assert row["name"] == "fetch" and row["ready"] and row["catalog"]["status"] == "official"
    with pytest.raises(ValueError, match="already exists"):
        ops.add_from_request(catalog_id="fetch")
    ops.add_from_request(catalog_id="fetch", name="fetch2", enabled=False)
    names = {s.name: s.enabled for s in cfg.load_mcp_config()}
    assert names == {"fetch": True, "fetch2": False}
    assert ops.toggle("fetch", False)["ok"] and not cfg.get_server("fetch").enabled
    assert ops.toggle("ghost", True)["ok"] is False
    assert ops.remove("fetch2")["ok"] and ops.remove("fetch2")["ok"] is False
    data = json.loads((roster / "mcp.json").read_text())
    assert [s["name"] for s in data["servers"]] == ["fetch"]


def test_missing_params_and_bad_definitions_are_rejected(roster):
    with pytest.raises(ValueError, match="missing required parameter"):
        ops.add_from_request(catalog_id="filesystem")
    with pytest.raises(ValueError, match="unknown catalog entry"):
        ops.add_from_request(catalog_id="nope")
    with pytest.raises(ValueError, match="needs a url"):
        ops.add_from_request(name="x", transport="http")
    with pytest.raises(ValueError, match="server name"):
        ops.add_from_request(name="bad name!", transport="stdio", command="npx")


def test_secrets_never_land_in_the_roster_and_resolve_at_launch(roster):
    row = ops.add_from_request(catalog_id="brave-search", env={"BRAVE_API_KEY": "brv-secret"})
    text = (roster / "mcp.json").read_text()
    assert "brv-secret" not in text and "${BRAVE_API_KEY}" in text
    assert row["ready"] and row["missing_env"] == []
    server = cfg.get_server("brave-search")
    env, missing = server.resolved_env()
    assert env == {"BRAVE_API_KEY": "brv-secret"} and missing == []
    # the secrets file is private and keyed per server/var
    secrets_file = roster / "mcp-secrets.json"
    assert secrets_file.is_file() and json.loads(secrets_file.read_text()) == {"brave-search/BRAVE_API_KEY": "brv-secret"}
    # clearing + removing
    assert ops.set_env("brave-search", {"BRAVE_API_KEY": ""})["missing_env"] == ["BRAVE_API_KEY"]
    assert ops.remove("brave-search")["ok"]


def test_missing_env_reported_and_process_env_wins(roster, monkeypatch):
    ops.add_from_request(catalog_id="github")
    row = ops.server_rows()[0]
    assert row["missing_env"] == ["GITHUB_PERSONAL_ACCESS_TOKEN"] and not row["ready"]
    server = cfg.get_server("github")
    assert server.auth_header() == {}
    monkeypatch.setenv("GITHUB_PERSONAL_ACCESS_TOKEN", "ghp_x")
    assert server.auth_header() == {"Authorization": "Bearer ghp_x"}
    assert server.missing_env() == []


def test_set_env_on_custom_server_registers_the_variable(roster):
    ops.add_from_request(name="mytool", transport="stdio", command="npx", args=["-y", "mytool"])
    out = ops.set_env("mytool", {"MYTOOL_TOKEN": "t0k"})
    assert out["stored"] == {"MYTOOL_TOKEN": "file"} and out["missing_env"] == []
    assert cfg.get_server("mytool").env == {"MYTOOL_TOKEN": "${MYTOOL_TOKEN}"}
    with pytest.raises(ValueError):
        ops.set_env("ghost", {"X": "y"})


def test_optional_env_never_blocks_readiness(roster):
    row = ops.add_from_request(catalog_id="context7")
    assert row["ready"] and row["missing_env"] == [] and row.get("env", {}) == {}
    row = ops.add_from_request(catalog_id="memory", env={"MEMORY_FILE_PATH": "/tmp/mem.jsonl"})
    assert row["ready"] and row["env"] == {"MEMORY_FILE_PATH": "${MEMORY_FILE_PATH}"}


def test_catalog_payload_marks_installed(roster):
    ops.add_from_request(catalog_id="context7")
    payload = ops.catalog_payload()
    by_id = {e["id"]: e for e in payload["servers"]}
    assert by_id["context7"]["installed"] is True and by_id["fetch"]["installed"] is False
    assert payload["categories"]["docs"] and by_id["context7"]["category_label"]
    assert {e["category"] for e in ops.catalog_payload("database")["servers"]} == {"database"}


# --------------------------------------------------------------------------
# agent-facing
# --------------------------------------------------------------------------

def test_prompt_block_lists_enabled_servers_with_hints(roster):
    from cgx.mcp.manager import servers_prompt_block
    from cgx.session.tasks.swarm_tools import mcp_prompt_block, mcp_tools_if_configured
    assert servers_prompt_block() == "" and mcp_tools_if_configured() == ()
    ops.add_from_request(catalog_id="context7")
    ops.add_from_request(catalog_id="playwright", enabled=False)
    # a hand-written entry matching a catalog id inherits its hint
    cfg.add_server(cfg.MCPServerConfig(name="fetch", command="uvx", args=["mcp-server-fetch"]))
    block = mcp_prompt_block()
    assert "context7:" in block and "documentation" in block
    assert "fetch:" in block and "real documentation page" in block
    assert "playwright" not in block  # disabled servers are not advertised
    assert mcp_tools_if_configured() == ("mcp_list_servers", "mcp_list_tools", "mcp_describe_tool", "mcp_call")


def test_test_server_reports_missing_env_without_connecting(roster):
    from cgx.mcp.manager import test_server
    ops.add_from_request(catalog_id="github")
    out = test_server("github")
    assert out["ok"] is False and out["missing_env"] == ["GITHUB_PERSONAL_ACCESS_TOKEN"]
    assert "missing value" in out["error"]
    assert test_server("ghost")["error"].startswith("unknown MCP server")


# --------------------------------------------------------------------------
# routes + CLI go through the same ops
# --------------------------------------------------------------------------

def test_routes_add_test_toggle_remove(roster):
    from fastapi import HTTPException
    from cgx.webui.routes import mcp as routes
    out = routes.add(routes.AddServerBody(catalog_id="filesystem", params={"ALLOWED_DIR": "/tmp/x"}))
    assert out["name"] == "filesystem" and "/tmp/x" in out["args"]
    with pytest.raises(HTTPException) as ei:
        routes.add(routes.AddServerBody(catalog_id="filesystem", params={"ALLOWED_DIR": "/tmp/x"}))
    assert ei.value.status_code == 409
    with pytest.raises(HTTPException) as ei:
        routes.add(routes.AddServerBody(catalog_id="filesystem"))
    assert ei.value.status_code == 400
    assert routes.catalog()["servers"] and routes.servers()["servers"][0]["name"] == "filesystem"
    assert routes.toggle(routes.ToggleBody(name="filesystem", enabled=False))["ok"]
    assert routes.set_env("filesystem", routes.EnvBody(env={"FOO": "bar"}))["ok"]
    assert routes.remove("filesystem")["ok"]
    with pytest.raises(HTTPException):
        routes.remove("filesystem")


def test_cli_parses_catalog_verbs(roster, capsys):
    from cgx.cli.main import build_parser
    parser = build_parser()
    args = parser.parse_args(["mcp", "add", "filesystem", "--param", "ALLOWED_DIR=/tmp/y", "--disabled"])
    args.func(args)
    assert cfg.get_server("filesystem").enabled is False
    args = parser.parse_args(["mcp", "catalog", "--category", "docs", "--table"])
    args.func(args)
    assert "context7" in capsys.readouterr().out
    args = parser.parse_args(["mcp", "list", "--table"])
    args.func(args)
    assert "filesystem" in capsys.readouterr().out
    args = parser.parse_args(["mcp", "rm", "filesystem"])
    args.func(args)
    assert cfg.get_server("filesystem") is None
