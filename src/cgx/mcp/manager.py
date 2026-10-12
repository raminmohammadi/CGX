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

from cgx.mcp.config import MCPServerConfig, enabled_servers, get_server

logger = logging.getLogger(__name__)

# Wall-clock cap on one connect+list/call round trip so a hung stdio child or
# an unreachable HTTP endpoint cannot stall a generation turn indefinitely.
_CALL_TIMEOUT_S = 60.0

_SDK_HINT = ("The MCP SDK is not installed. Install it with "
             "'pip install cgx[mcp]' to use MCP tool servers.")

# (tools_json, fetched_at) per server name.
_TOOLS_CACHE: Dict[str, Tuple[str, float]] = {}
_TOOLS_TTL = 300.0

# Tiered disclosure (JEV #4): the model sees only tool names + descriptions
# (Tier 1) until it picks one; the full argument schema is fetched on demand
# (Tier 2, mcp_describe_tool) and re-checked before dispatch, so a wrong
# argument type is an actionable error rather than a silent server failure.
# {server -> {tool -> inputSchema}}, populated whenever a server is listed.
_TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {}


def _args_schema_violations(server: str, tool: str,
                            arguments: Dict[str, Any]) -> List[str]:
    """Best-effort validation of MCP call arguments against the cached schema.

    Uses the same draft-07 subset validator the rest of CGX uses; unknown
    JSON-Schema keywords are ignored (no false rejects). Empty list when no
    object schema is known for the tool or the arguments conform.
    """
    schema = (_TOOL_SCHEMAS.get(server) or {}).get(tool)
    if not isinstance(schema, dict) or schema.get("type") != "object":
        return []
    try:
        from cgx.answer.schemas import validate_json_schema
        return validate_json_schema(arguments, schema)
    except Exception:  # pragma: no cover - validation is best-effort
        return []


def _have_sdk() -> bool:
    try:
        import mcp  # noqa: F401
        return True
    except Exception:
        return False


def _missing_env(server: Any) -> List[str]:
    """Unresolved ``${VAR}`` references for ``server`` (tolerates stand-ins)."""
    fn = getattr(server, "missing_env", None)
    if not callable(fn):
        return []
    try:
        return list(fn() or [])
    except Exception:  # pragma: no cover - never block a call on this check
        return []


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
        return ("No MCP servers configured. Add them from the MCP page / "
                "`cgx mcp add <catalog-id>` (see docs).")
    lines = []
    for s in servers:
        what = s.agent_hint or s.description
        lines.append(f"- {s.name} ({s.transport})" + (f": {what}" if what else ""))
    return "Configured MCP servers:\n" + "\n".join(lines)


def servers_prompt_block() -> str:
    """One prompt block telling the agent WHICH servers exist and WHEN to use them.

    ``mcp_list_servers`` only answers when asked; a local model that never asks
    never learns a docs server or a browser is one call away. Rendering each
    enabled server's ``agent_hint`` into the system prompt closes that gap.
    Empty when no server is enabled.
    """
    try:
        servers = enabled_servers()
    except Exception:  # pragma: no cover
        return ""
    if not servers:
        return ""
    lines = ["MCP SERVERS AVAILABLE (external tools). Workflow: mcp_list_tools "
             '{"server": "<name>"} to see a server\'s tools, mcp_describe_tool for a '
             "tool's arguments, then mcp_call. Use them when the task matches:"]
    for s in servers:
        hint = s.agent_hint or s.description
        if not hint:
            # A hand-written roster entry that matches a catalog server (by id
            # or name) inherits the catalog's hint.
            try:
                from cgx.mcp.catalog import catalog_entry
                entry = catalog_entry(s.catalog_id or "") or catalog_entry(s.name)
                hint = entry.agent_hint if entry else ""
            except Exception:  # pragma: no cover
                hint = ""
        lines.append(f"- {s.name}: {hint or f'{s.transport} server'}")
    return "\n".join(lines)


def test_server(name: str, *, timeout: float = 30.0) -> Dict[str, Any]:
    """Connect to ``name`` (enabled or not) and list its tools; never raises.

    Returns ``{"ok": bool, "tools": [{"name", "description"}], "error": str,
    "missing_env": [...], "sdk_installed": bool}`` -- the MCP page's "Test"
    button and ``cgx mcp test``.
    """
    server = get_server(name)
    result: Dict[str, Any] = {"ok": False, "tools": [], "error": "",
                              "missing_env": [], "sdk_installed": _have_sdk()}
    if server is None:
        result["error"] = f"unknown MCP server {name!r}"
        return result
    missing = server.missing_env()
    result["missing_env"] = missing
    if missing:
        result["error"] = ("missing value(s) for " + ", ".join(missing)
                           + " -- set them on the MCP page or export them")
        return result
    if not _have_sdk():
        result["error"] = _SDK_HINT
        return result
    try:
        import asyncio
        tools = asyncio.run(asyncio.wait_for(_list_tools_structured(server), timeout))
    except Exception as exc:  # pragma: no cover - depends on a live server
        msg = f"{type(exc).__name__}: {exc}".strip()
        if (server.command or "") == "npx":
            msg += (" -- if this is an npm E401/auth error, your ~/.npmrc points npm at a private "
                    "registry; set CGX_NPM_REGISTRY=https://registry.npmjs.org/ for MCP servers")
        result["error"] = msg
        return result
    result["ok"] = True
    result["tools"] = tools
    # Refresh the Tier-1 cache so the agent sees the same listing.
    if tools:
        text = (f"Tools on {server.name}:\n"
                + "\n".join(f"- {t['name']}: {t.get('description') or ''}" for t in tools)
                + "\n(Call mcp_describe_tool for a tool's argument schema before mcp_call.)")
        _TOOLS_CACHE[server.name] = (text, time.time())
    return result


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
    missing = _missing_env(server)
    if missing:
        return (f"MCP server {name!r} is missing value(s) for {', '.join(missing)}; "
                "set them on the MCP page (or export them) and retry.")
    try:
        import asyncio
        result = asyncio.run(asyncio.wait_for(_list_tools_async(server), _CALL_TIMEOUT_S))
    except Exception as exc:  # pragma: no cover - depends on live server
        return f"Failed to list tools for {name}: {type(exc).__name__}: {exc}"
    # Tool descriptions are server-controlled text that lands in the model's
    # context -- an injection vector too. Screen them like any untrusted blob.
    try:
        from cgx.guardrails.injection import screen_untrusted
        result = screen_untrusted(result, origin=f"MCP {name} tool list")
    except Exception:  # pragma: no cover - guardrail is best-effort
        pass
    _TOOLS_CACHE[name] = (result, time.time())
    return result


def describe_tool(args: Dict[str, Any], ctx: Any) -> str:
    """Tier-2 disclosure: the JSON argument schema for ONE MCP tool, on demand.

    The model sees only names + descriptions (Tier 1) until it picks a tool;
    this returns that tool's full ``inputSchema`` so it can construct correct
    arguments -- and :func:`call_tool` re-validates against it, so a wrong
    argument type becomes an actionable error rather than a silent failure.
    """
    name = str(args.get("server") or "")
    tool = str(args.get("tool") or "")
    if not tool:
        return "mcp_describe_tool requires a 'tool' name."
    if name not in _TOOL_SCHEMAS:
        # Populate the schema cache by listing the server's tools first.
        list_tools({"server": name}, ctx)
    schema = (_TOOL_SCHEMAS.get(name) or {}).get(tool)
    if schema is None:
        return (f"No schema for tool {tool!r} on server {name!r}. "
                "Run mcp_list_tools first, or check the names.")
    return (f"Argument schema for {name}/{tool}:\n"
            + json.dumps(schema, indent=2)
            + "\nConstruct 'arguments' to match, then call mcp_call.")


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
    # Egress policy: an MCP tool that fetches a URL (e.g. mcp-server-fetch) must
    # obey the same SSRF block + domain allowlist as the built-in fetch_url, so
    # switching to MCP doesn't lose the "don't reach internal / arbitrary hosts"
    # guarantee. Screen every URL-valued argument before dispatching.
    try:
        from cgx.guardrails.net import find_urls, url_refusal
        for u in find_urls(arguments):
            refusal = url_refusal(u)
            if refusal:
                return (f"[BLOCKED] MCP {name}/{tool} refused: {refusal}. "
                        "Adjust the URL, or set CGX_FETCH_ALLOWLIST / "
                        "CGX_FETCH_ALLOW_ANY to permit it.")
    except Exception:  # pragma: no cover - guardrail is best-effort
        pass
    # Tier-2 arg validation against the cached inputSchema: a wrong argument
    # type/shape is an actionable error, not a silent server failure (JEV #4).
    violations = _args_schema_violations(name, tool, arguments)
    if violations:
        return (f"[INVALID ARGS] MCP {name}/{tool}: " + "; ".join(violations[:8])
                + ". Call mcp_describe_tool to see the expected schema.")
    missing = _missing_env(server)
    if missing:
        return (f"MCP server {name!r} is missing value(s) for {', '.join(missing)}; "
                "set them on the MCP page (or export them) and retry.")
    try:
        import asyncio
        result = asyncio.run(asyncio.wait_for(
            _call_tool_async(server, tool, arguments), _CALL_TIMEOUT_S))
    except Exception as exc:  # pragma: no cover - depends on live server
        return f"MCP call failed: {type(exc).__name__}: {exc}"
    # An MCP tool result (e.g. a fetched web page) is untrusted third-party
    # content -- screen it through the shared prompt-injection guardrail before
    # it enters the model's context, exactly as the built-in fetch/search do.
    try:
        from cgx.guardrails.injection import screen_untrusted
        return screen_untrusted(result, origin=f"MCP {name}/{tool}")
    except Exception:  # pragma: no cover - guardrail is best-effort
        return result


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
        url, _missing = server.resolved_url()
        return ("http", streamablehttp_client(url or server.url or "",
                                              headers=server.auth_header()),
                ClientSession)
    from mcp import StdioServerParameters  # type: ignore
    from mcp.client.stdio import stdio_client  # type: ignore
    # ``${VAR}`` references in args/env resolve from the environment, then the
    # secret store, at launch -- the roster never carries a secret. The child
    # inherits the SDK's safe default environment (PATH, HOME ...) so npx/uvx
    # can run, plus the server's declared variables.
    args, _m1 = server.resolved_args()
    env_vars, _m2 = server.resolved_env()
    # A ~/.npmrc pointing npm at a private registry (an expired corporate token
    # is common) makes every `npx -y <public package>` fail with E401. The
    # catalog's npm packages are all public, so an operator can pin the registry
    # for MCP children with CGX_NPM_REGISTRY without touching their npmrc.
    import os as _os
    if (server.command or "") in ("npx", "npm") and _os.environ.get("CGX_NPM_REGISTRY"):
        env_vars = {**env_vars, "NPM_CONFIG_REGISTRY": _os.environ["CGX_NPM_REGISTRY"]}
    child_env = None
    if env_vars:
        try:
            from mcp.client.stdio import get_default_environment  # type: ignore
            child_env = {**get_default_environment(), **env_vars}
        except Exception:  # pragma: no cover - older SDKs
            import os as _os
            child_env = {**_os.environ, **env_vars}
    params = StdioServerParameters(command=server.command or "", args=list(args),
                                   env=child_env)
    return ("stdio", stdio_client(params), ClientSession)


async def _list_tools_async(server: MCPServerConfig) -> str:
    kind, transport_cm, ClientSession = await _open_session(server)
    async with transport_cm as streams:
        read, write = streams[0], streams[1]
        async with ClientSession(read, write) as session:
            await session.initialize()
            resp = await session.list_tools()
            tools = getattr(resp, "tools", []) or []
            # Cache each tool's argument schema for Tier-2 disclosure +
            # pre-dispatch validation, but keep the Tier-1 listing to
            # names + descriptions so the prompt stays small.
            _TOOL_SCHEMAS[server.name] = {
                t.name: (getattr(t, "inputSchema", None) or {}) for t in tools}
            if not tools:
                return f"{server.name}: (no tools)"
            lines = [f"- {t.name}: {getattr(t, 'description', '') or ''}"
                     for t in tools]
            return (f"Tools on {server.name}:\n" + "\n".join(lines)
                    + "\n(Call mcp_describe_tool for a tool's argument schema "
                    "before mcp_call.)")


async def _list_tools_structured(server: MCPServerConfig) -> List[Dict[str, str]]:
    """Connect and return ``[{name, description}]`` (also fills the schema cache)."""
    kind, transport_cm, ClientSession = await _open_session(server)
    async with transport_cm as streams:
        read, write = streams[0], streams[1]
        async with ClientSession(read, write) as session:
            await session.initialize()
            resp = await session.list_tools()
            tools = getattr(resp, "tools", []) or []
            _TOOL_SCHEMAS[server.name] = {
                t.name: (getattr(t, "inputSchema", None) or {}) for t in tools}
            return [{"name": t.name, "description": (getattr(t, "description", "") or "")[:300]}
                    for t in tools]


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
    """Register the four MCP tools on the default swarm registry.

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
        name="mcp_describe_tool", risk=RiskLevel.LOW,
        arg_hint='{"server": "...", "tool": "..."}',
        description="Show one MCP tool's JSON argument schema (Tier-2, on demand).",
        handler=describe_tool))
    REGISTRY.register(ToolSpec(
        name="mcp_call", risk=RiskLevel.HIGH,
        arg_hint='{"server": "...", "tool": "...", "arguments": {}}',
        description="Call a tool on an MCP server with a JSON arguments object.",
        handler=call_tool))
