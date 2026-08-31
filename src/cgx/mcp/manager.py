"""Lazy MCP client + the three registry tools the swarm agents call.

Discovery is lazy (``mcp_list_servers`` -> ``mcp_list_tools`` -> ``mcp_call``)
so many servers never flood the prompt. Each ``mcp_call`` opens a fresh session
for the target server, invokes the tool, and closes -- simple and stateless,
matching the swarm's per-turn synchronous execution. ``tools/list`` results are
cached briefly per server to avoid re-listing on every discovery.

The ``mcp`` SDK is optional: import failures surface as an actionable message
rather than crashing the generation loop.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from cgx.mcp.config import MCPServerConfig, enabled_servers

logger = logging.getLogger(__name__)

_SDK_HINT = ("The MCP SDK is not installed. Install it with "
             "'pip install cgx[mcp]' to use MCP tool servers.")

# (tools_json, fetched_at) per server name.
_TOOLS_CACHE: Dict[str, Tuple[str, float]] = {}
_TOOLS_TTL = 300.0


def _have_sdk() -> bool:
    try:
        import mcp  # noqa: F401
        return True
    except Exception:
        return False


def _find_server(name: str) -> Optional[MCPServerConfig]:
    name = (name or "").strip()
    for s in enabled_servers():
        if s.name == name:
            return s
    return None


def list_servers(_args: Dict[str, Any], _ctx: Any) -> str:
    """List configured, enabled MCP servers (works without the SDK)."""
    servers = enabled_servers()
    if not servers:
        return ("No MCP servers configured. Add them to ~/.cgx/mcp.json "
                "(see docs).")
    lines = [f"- {s.name} ({s.transport}"
             + (f": {s.url}" if s.url else f": {s.command} {' '.join(s.args)}")
             + ")"
             for s in servers]
    return "Configured MCP servers:\n" + "\n".join(lines)


def list_tools(args: Dict[str, Any], _ctx: Any) -> str:
    """List the tools a given server exposes (cached ~5 min)."""
    name = str(args.get("server") or "")
    server = _find_server(name)
    if server is None:
        return f"Unknown or disabled MCP server: {name!r}"
    cached = _TOOLS_CACHE.get(name)
    if cached and (time.time() - cached[1]) < _TOOLS_TTL:
        return cached[0]
    if not _have_sdk():
        return _SDK_HINT
    try:
        import asyncio
        result = asyncio.run(_list_tools_async(server))
    except Exception as exc:  # pragma: no cover - depends on live server
        return f"Failed to list tools for {name}: {type(exc).__name__}: {exc}"
    _TOOLS_CACHE[name] = (result, time.time())
    return result


def call_tool(args: Dict[str, Any], _ctx: Any) -> str:
    """Invoke ``tool`` on ``server`` with ``arguments`` (a dict)."""
    name = str(args.get("server") or "")
    tool = str(args.get("tool") or "")
    arguments = args.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except Exception:
            arguments = {"input": arguments}
    server = _find_server(name)
    if server is None:
        return f"Unknown or disabled MCP server: {name!r}"
    if not tool:
        return "mcp_call requires a 'tool' name."
    if not _have_sdk():
        return _SDK_HINT
    try:
        import asyncio
        return asyncio.run(_call_tool_async(server, tool, arguments))
    except Exception as exc:  # pragma: no cover - depends on live server
        return f"MCP call failed: {type(exc).__name__}: {exc}"


# --------------------- async SDK bridge ---------------------

async def _open_session(server: MCPServerConfig):
    """Yield an initialized ClientSession for ``server`` (stdio or http).

    Returned as an async context manager tuple so callers use it with
    ``async with``. Kept in one place so both list and call share transport
    setup.
    """
    from mcp import ClientSession  # type: ignore
    if server.transport == "http":
        from mcp.client.streamable_http import streamablehttp_client  # type: ignore
        return ("http", streamablehttp_client(server.url or "",
                                              headers=server.auth_header()),
                ClientSession)
    from mcp import StdioServerParameters  # type: ignore
    from mcp.client.stdio import stdio_client  # type: ignore
    params = StdioServerParameters(command=server.command or "",
                                   args=list(server.args or []))
    return ("stdio", stdio_client(params), ClientSession)


async def _list_tools_async(server: MCPServerConfig) -> str:
    kind, transport_cm, ClientSession = await _open_session(server)
    async with transport_cm as streams:
        read, write = streams[0], streams[1]
        async with ClientSession(read, write) as session:
            await session.initialize()
            resp = await session.list_tools()
            tools = getattr(resp, "tools", []) or []
            if not tools:
                return f"{server.name}: (no tools)"
            lines = [f"- {t.name}: {getattr(t, 'description', '') or ''}"
                     for t in tools]
            return f"Tools on {server.name}:\n" + "\n".join(lines)


async def _call_tool_async(server: MCPServerConfig, tool: str,
                           arguments: Dict[str, Any]) -> str:
    kind, transport_cm, ClientSession = await _open_session(server)
    async with transport_cm as streams:
        read, write = streams[0], streams[1]
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, arguments)
            parts: List[str] = []
            for item in getattr(result, "content", []) or []:
                text = getattr(item, "text", None)
                parts.append(text if text is not None else str(item))
            return "\n".join(parts) if parts else "(no content)"


def register_mcp_tools() -> None:
    """Register the three MCP tools on the default swarm registry.

    Idempotent. HIGH risk on ``mcp_call`` so the approval gate intercepts any
    external effect; discovery tools are LOW (read-only).
    """
    from cgx.session.tasks.tool_registry import (
        REGISTRY, RiskLevel, ToolSpec)
    REGISTRY.register(ToolSpec(
        name="mcp_list_servers", risk=RiskLevel.LOW, arg_hint="{}",
        description="List configured external MCP tool servers.",
        handler=list_servers))
    REGISTRY.register(ToolSpec(
        name="mcp_list_tools", risk=RiskLevel.LOW,
        arg_hint='{"server": "..."}',
        description="List the tools a named MCP server exposes.",
        handler=list_tools))
    REGISTRY.register(ToolSpec(
        name="mcp_call", risk=RiskLevel.HIGH,
        arg_hint='{"server": "...", "tool": "...", "arguments": {}}',
        description="Call a tool on an MCP server with a JSON arguments object.",
        handler=call_tool))
