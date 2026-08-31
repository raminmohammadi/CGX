"""MCP (Model Context Protocol) integration for CGX.

Gives the swarm agents access to external MCP tool servers through three
registry tools (``mcp_list_servers`` / ``mcp_list_tools`` / ``mcp_call``) that
discover servers and their tools *lazily*, so a large fleet of MCP servers does
not flood the model's context with every tool schema up front.

The server roster is a local JSON config (``~/.cgx/mcp.json``) -- adding a tool
server is a config edit, no code change -- keeping CGX local-first: nothing is
contacted until the agent explicitly lists or calls a server.

The ``mcp`` SDK is an optional dependency (extra ``mcp``). Everything degrades
gracefully: without it, ``mcp_list_servers`` still works from config and the
call tools return an actionable "install cgx[mcp]" message.
"""

from cgx.mcp.config import MCPServerConfig, load_mcp_config  # noqa: F401
from cgx.mcp.manager import register_mcp_tools  # noqa: F401
