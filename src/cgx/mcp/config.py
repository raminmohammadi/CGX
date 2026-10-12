"""Local MCP server roster (``~/.cgx/mcp.json``).

Config-driven so adding an MCP tool server never requires touching code. Pure
stdlib (no ``mcp`` SDK) so it is importable and testable regardless of whether
the optional dependency is installed.

Schema::

    {
      "servers": [
        {"name": "fetch", "transport": "stdio",
         "command": "uvx", "args": ["mcp-server-fetch"], "enabled": true,
         "catalog_id": "fetch", "description": "...", "agent_hint": "..."},
        {"name": "github", "transport": "http",
         "url": "https://api.githubcopilot.com/mcp/", "enabled": true,
         "auth": {"type": "bearer", "token_env": "GITHUB_PERSONAL_ACCESS_TOKEN"}},
        {"name": "brave", "transport": "stdio", "command": "npx",
         "args": ["-y", "@brave/brave-search-mcp-server"],
         "env": {"BRAVE_API_KEY": "${BRAVE_API_KEY}"}}
      ]
    }

Secrets never live in this file. ``env`` values and ``args`` may reference
``${VAR}``; at launch each reference is resolved from the CGX process
environment, then from the secret store (:mod:`cgx.mcp.secrets`). ``auth`` for
an HTTP server names the variable holding the bearer token the same way.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def default_config_path() -> Path:
    """Location of the roster; overridable via ``CGX_MCP_CONFIG``."""
    override = os.environ.get("CGX_MCP_CONFIG")
    if override:
        return Path(override).expanduser()
    return Path(os.environ.get("CGX_CONFIG_DIR")
                or (Path.home() / ".cgx")).expanduser() / "mcp.json"


def _resolve_ref(server: str, var: str) -> Optional[str]:
    try:
        from cgx.mcp.secrets import resolve_env
        return resolve_env(server, var)
    except Exception:  # pragma: no cover - secrets layer is best-effort
        return os.environ.get(var)


def _expand(server: str, text: str) -> Tuple[str, List[str]]:
    """Replace ``${VAR}`` references in ``text``; return (expanded, missing vars)."""
    missing: List[str] = []

    def _sub(m: "re.Match[str]") -> str:
        var = m.group(1)
        val = _resolve_ref(server, var)
        if val is None or val == "":
            missing.append(var)
            return ""
        return val

    return _REF_RE.sub(_sub, text or ""), missing


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
    #: Environment for a stdio child process; values may be ``${VAR}`` refs.
    env: Dict[str, str] = field(default_factory=dict)
    #: Catalog template this server was added from (``None`` for hand-written).
    catalog_id: Optional[str] = None
    description: str = ""
    #: One line telling the agent WHEN to use this server (prompt-injected).
    agent_hint: str = ""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MCPServerConfig":
        env = d.get("env") or {}
        return cls(
            name=str(d.get("name") or "").strip(),
            transport=str(d.get("transport") or "stdio").strip().lower(),
            command=d.get("command"),
            args=[str(a) for a in (d.get("args") or [])],
            url=d.get("url"),
            enabled=bool(d.get("enabled", True)),
            auth=dict(d.get("auth") or {}),
            env={str(k): str(v) for k, v in env.items()} if isinstance(env, dict) else {},
            catalog_id=(str(d["catalog_id"]) if d.get("catalog_id") else None),
            description=str(d.get("description") or ""),
            agent_hint=str(d.get("agent_hint") or ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Roster row (omits empty optional fields; never contains secret values)."""
        out: Dict[str, Any] = {"name": self.name, "transport": self.transport,
                               "enabled": bool(self.enabled)}
        if self.transport == "http":
            out["url"] = self.url or ""
            if self.auth:
                out["auth"] = dict(self.auth)
        else:
            out["command"] = self.command or ""
            out["args"] = list(self.args)
            if self.env:
                out["env"] = dict(self.env)
        for key in ("catalog_id", "description", "agent_hint"):
            val = getattr(self, key)
            if val:
                out[key] = val
        return out

    # ---- launch-time resolution ----------------------------------------
    def referenced_vars(self) -> List[str]:
        """Every ``${VAR}`` this server references (env values, args, auth)."""
        seen: List[str] = []
        texts = list(self.env.values()) + list(self.args) + [self.url or ""]
        for t in texts:
            for m in _REF_RE.finditer(t or ""):
                if m.group(1) not in seen:
                    seen.append(m.group(1))
        tok = self.auth.get("token_env") if self.auth.get("type") == "bearer" else None
        if tok and tok not in seen:
            seen.append(str(tok))
        return seen

    def resolved_env(self) -> Tuple[Dict[str, str], List[str]]:
        """Child-process env with references resolved; plus the unresolved names."""
        out: Dict[str, str] = {}
        missing: List[str] = []
        for k, v in self.env.items():
            val, miss = _expand(self.name, v)
            if miss:
                missing.extend(m for m in miss if m not in missing)
                continue
            out[k] = val
        return out, missing

    def resolved_args(self) -> Tuple[List[str], List[str]]:
        out: List[str] = []
        missing: List[str] = []
        for a in self.args:
            val, miss = _expand(self.name, a)
            if miss:
                missing.extend(m for m in miss if m not in missing)
            out.append(val)
        return out, missing

    def resolved_url(self) -> Tuple[str, List[str]]:
        return _expand(self.name, self.url or "")

    def missing_env(self) -> List[str]:
        """Referenced variables with no value in the environment or secret store."""
        missing: List[str] = []
        for var in self.referenced_vars():
            if not _resolve_ref(self.name, var):
                missing.append(var)
        return missing

    def auth_header(self) -> Dict[str, str]:
        """Resolve an Authorization header from the auth spec, if any.

        Only a ``bearer`` token is supported; the token is read from the named
        environment variable, then the secret store (never from the JSON).
        """
        if self.auth.get("type") == "bearer":
            token_env = str(self.auth.get("token_env") or "")
            token = _resolve_ref(self.name, token_env) if token_env else None
            if token:
                return {"Authorization": f"Bearer {token}"}
        return {}


def _read_roster(p: Path) -> Dict[str, Any]:
    try:
        if not p.exists():
            return {"servers": []}
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"servers": []}
    if not isinstance(data, dict) or not isinstance(data.get("servers"), list):
        return {"servers": []}
    return data


def load_mcp_config(path: Optional[Path] = None) -> List[MCPServerConfig]:
    """Load and validate the server roster; ``[]`` when absent or malformed."""
    p = path or default_config_path()
    data = _read_roster(p)
    out: List[MCPServerConfig] = []
    for entry in data.get("servers") or []:
        if isinstance(entry, dict) and entry.get("name"):
            out.append(MCPServerConfig.from_dict(entry))
    return out


def enabled_servers(path: Optional[Path] = None) -> List[MCPServerConfig]:
    return [s for s in load_mcp_config(path) if s.enabled]


def save_mcp_config(servers: List[MCPServerConfig], path: Optional[Path] = None) -> Path:
    """Persist the roster (pretty JSON, 0600-ish directory is the caller's job)."""
    p = path or default_config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"servers": [s.to_dict() for s in servers]}, indent=2) + "\n",
                 encoding="utf-8")
    return p


def get_server(name: str, path: Optional[Path] = None) -> Optional[MCPServerConfig]:
    name = (name or "").strip()
    for s in load_mcp_config(path):
        if s.name == name:
            return s
    return None


def add_server(server: MCPServerConfig, *, replace: bool = False,
               path: Optional[Path] = None) -> MCPServerConfig:
    """Append (or replace) a server in the roster. Raises ``ValueError`` on a
    duplicate name unless ``replace`` is set, or on an invalid definition."""
    name = (server.name or "").strip()
    if not name or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", name):
        raise ValueError("server name must be 1-64 chars: letters, digits, '.', '_' or '-'")
    server.name = name
    if server.transport not in ("stdio", "http"):
        raise ValueError("transport must be 'stdio' or 'http'")
    if server.transport == "http" and not (server.url or "").strip():
        raise ValueError("an http server needs a url")
    if server.transport == "stdio" and not (server.command or "").strip():
        raise ValueError("a stdio server needs a command")
    servers = load_mcp_config(path)
    existing = [i for i, s in enumerate(servers) if s.name == name]
    if existing and not replace:
        raise ValueError(f"an MCP server named {name!r} already exists")
    if existing:
        servers[existing[0]] = server
    else:
        servers.append(server)
    save_mcp_config(servers, path)
    return server


def remove_server(name: str, path: Optional[Path] = None) -> bool:
    servers = load_mcp_config(path)
    kept = [s for s in servers if s.name != (name or "").strip()]
    if len(kept) == len(servers):
        return False
    save_mcp_config(kept, path)
    return True


def set_enabled(name: str, enabled: bool, path: Optional[Path] = None) -> bool:
    servers = load_mcp_config(path)
    hit = False
    for s in servers:
        if s.name == (name or "").strip():
            s.enabled = bool(enabled)
            hit = True
    if hit:
        save_mcp_config(servers, path)
    return hit
