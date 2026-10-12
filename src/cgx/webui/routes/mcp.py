"""MCP server management surface.

Lists the configured MCP tool servers (from ``~/.cgx/mcp.json``) with their
catalog metadata and secret status, exposes the curated **catalog** of
well-known developer servers, and lets the operator add a server from the
catalog (or a custom definition), set its secrets, test the connection,
toggle it, or remove it. All logic lives in :mod:`cgx.mcp.ops` so the
``cgx mcp`` CLI does exactly the same things.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from cgx.mcp import ops
from cgx.mcp.config import default_config_path

router = APIRouter(tags=["mcp"])


@router.get("/mcp/servers")
def servers() -> dict:
    """Configured MCP servers (with catalog metadata + missing secrets) and SDK availability."""
    return {
        "sdk_installed": ops.sdk_installed(),
        "config_path": str(default_config_path()),
        "servers": ops.server_rows(),
    }


@router.get("/mcp/catalog")
def catalog(category: Optional[str] = None) -> dict:
    """The curated catalog of developer MCP servers (one-click add templates)."""
    return ops.catalog_payload(category)


class AddServerBody(BaseModel):
    """Add from the catalog (``catalog_id`` + ``params``/``env``) or a custom server."""
    catalog_id: Optional[str] = None
    name: Optional[str] = None
    params: Dict[str, str] = Field(default_factory=dict)
    # Secret values to store in the keyring/secret file -- never in the roster.
    env: Dict[str, str] = Field(default_factory=dict)
    enabled: bool = True
    replace: bool = False
    transport: Optional[str] = None
    command: Optional[str] = None
    args: List[str] = Field(default_factory=list)
    url: Optional[str] = None
    auth: Dict[str, Any] = Field(default_factory=dict)
    description: str = ""
    agent_hint: str = ""


@router.post("/mcp/servers")
def add(body: AddServerBody) -> dict:
    try:
        return ops.add_from_request(**body.model_dump())
    except ValueError as exc:
        code = 409 if "already exists" in str(exc) else 400
        raise HTTPException(status_code=code, detail=str(exc)) from exc


@router.delete("/mcp/servers/{name}")
def remove(name: str) -> dict:
    out = ops.remove(name)
    if not out["ok"]:
        raise HTTPException(status_code=404, detail=f"server {name!r} not found")
    return out


class EnvBody(BaseModel):
    env: Dict[str, str]


@router.post("/mcp/servers/{name}/env")
def set_env(name: str, body: EnvBody) -> dict:
    """Store (or clear, with an empty value) a server's secret variables."""
    try:
        return ops.set_env(name, body.env)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/mcp/servers/{name}/test")
def test(name: str) -> dict:
    """Connect to the server and list its tools (or explain why it cannot)."""
    from cgx.mcp.manager import test_server
    return test_server(name)


class ToggleBody(BaseModel):
    name: str
    enabled: bool


@router.post("/mcp/toggle")
def toggle(body: ToggleBody) -> dict:
    """Enable or disable a configured server, persisting to the JSON roster."""
    return ops.toggle(body.name, body.enabled)
