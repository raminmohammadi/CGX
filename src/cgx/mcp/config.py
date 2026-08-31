"""Local MCP server roster (``~/.cgx/mcp.json``).

Config-driven so adding an MCP tool server never requires touching code -- the
user's second requirement for the MCP feature. Pure stdlib (no ``mcp`` SDK) so
it is importable and testable regardless of whether the optional dependency is
installed.

Schema::

    {
      "servers": [
        {"name": "fetch", "transport": "stdio",
         "command": "uvx", "args": ["mcp-server-fetch"], "enabled": true},
        {"name": "docs", "transport": "http",
         "url": "http://localhost:3000/mcp", "enabled": true,
         "auth": {"type": "bearer", "token_env": "DOCS_TOKEN"}}
      ]
    }
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def default_config_path() -> Path:
    """Location of the roster; overridable via ``CGX_MCP_CONFIG``."""
    override = os.environ.get("CGX_MCP_CONFIG")
    if override:
        return Path(override).expanduser()
    return Path(os.environ.get("CGX_CONFIG_DIR")
                or (Path.home() / ".cgx")).expanduser() / "mcp.json"


@dataclass
class MCPServerConfig:
    """One configured MCP server."""

    name: str
    transport: str = "stdio"  # "stdio" | "http"
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    url: Optional[str] = None
    enabled: bool = True
    auth: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MCPServerConfig":
        return cls(
            name=str(d.get("name") or "").strip(),
            transport=str(d.get("transport") or "stdio").strip().lower(),
            command=d.get("command"),
            args=list(d.get("args") or []),
            url=d.get("url"),
            enabled=bool(d.get("enabled", True)),
            auth=dict(d.get("auth") or {}),
        )

    def auth_header(self) -> Dict[str, str]:
        """Resolve an Authorization header from the auth spec, if any.

        Only a ``bearer`` token sourced from an env var is supported (keeps
        secrets out of the JSON); returns an empty dict otherwise.
        """
        if self.auth.get("type") == "bearer":
            token_env = self.auth.get("token_env")
            token = os.environ.get(token_env or "", "")
            if token:
                return {"Authorization": f"Bearer {token}"}
        return {}


def load_mcp_config(path: Optional[Path] = None) -> List[MCPServerConfig]:
    """Load and validate the server roster; ``[]`` when absent or malformed."""
    p = path or default_config_path()
    try:
        if not p.exists():
            return []
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    servers = data.get("servers") if isinstance(data, dict) else None
    if not isinstance(servers, list):
        return []
    out: List[MCPServerConfig] = []
    for entry in servers:
        if isinstance(entry, dict) and entry.get("name"):
            out.append(MCPServerConfig.from_dict(entry))
    return out


def enabled_servers(path: Optional[Path] = None) -> List[MCPServerConfig]:
    return [s for s in load_mcp_config(path) if s.enabled]
