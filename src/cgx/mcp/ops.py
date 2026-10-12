"""Roster operations shared by the web routes and the ``cgx mcp`` CLI.

Both surfaces must do exactly the same thing (CLI/UI parity), so the logic
lives here once: list servers with their catalog metadata and missing-secret
status, add a server from a catalog template or a custom definition (storing
any supplied secrets in :mod:`cgx.mcp.secrets`, never in the roster), remove a
server (and its secrets), and set secrets for an existing server.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from cgx.mcp import secrets as _secrets
from cgx.mcp.catalog import CATEGORIES, catalog_entries, catalog_entry, instantiate, missing_params
from cgx.mcp.config import (
    MCPServerConfig, add_server, default_config_path, load_mcp_config, remove_server,
    set_enabled)

__all__ = ["sdk_installed", "server_row", "server_rows", "catalog_payload", "add_from_request",
           "remove", "set_env", "toggle"]


def sdk_installed() -> bool:
    try:
        import mcp  # noqa: F401
        return True
    except Exception:
        return False


def _catalog_meta(server: MCPServerConfig) -> Optional[Dict[str, Any]]:
    entry = catalog_entry(server.catalog_id or "") or catalog_entry(server.name)
    if entry is None:
        return None
    return {"id": entry.id, "name": entry.name, "vendor": entry.vendor, "category": entry.category,
            "category_label": CATEGORIES.get(entry.category, entry.category), "docs_url": entry.docs_url,
            "status": entry.status, "risk": entry.risk,
            "env": entry.env, "agent_hint": entry.agent_hint, "description": entry.description}


def server_row(server: MCPServerConfig) -> Dict[str, Any]:
    """A roster server as the UI/CLI see it (metadata + secret status, no values)."""
    row = server.to_dict()
    meta = _catalog_meta(server)
    if meta:
        row.setdefault("description", meta["description"])
        row.setdefault("agent_hint", meta["agent_hint"])
    row["catalog"] = meta
    row["env_vars"] = server.referenced_vars()
    row["missing_env"] = server.missing_env()
    row["ready"] = not row["missing_env"]
    # The UI shows WHICH variables exist, never their values.
    if "env" in row:
        row["env"] = {k: v for k, v in row["env"].items()}
    return row


def server_rows() -> List[Dict[str, Any]]:
    return [server_row(s) for s in load_mcp_config()]


def catalog_payload(category: Optional[str] = None) -> Dict[str, Any]:
    installed = {s.catalog_id or s.name for s in load_mcp_config()}
    entries = catalog_entries(category)
    for e in entries:
        e["installed"] = e["id"] in installed
        e["category_label"] = CATEGORIES.get(e["category"], e["category"])
    return {"categories": CATEGORIES, "servers": entries, "config_path": str(default_config_path()),
            "sdk_installed": sdk_installed()}


def _store_secrets(name: str, env: Dict[str, str]) -> Dict[str, str]:
    stored: Dict[str, str] = {}
    for var, value in (env or {}).items():
        var = str(var).strip()
        if not var:
            continue
        if value is None or str(value) == "":
            _secrets.delete_secrets(name, var)
            stored[var] = "deleted"
        else:
            stored[var] = _secrets.set_secret(name, var, str(value))
    return stored


def add_from_request(*, catalog_id: Optional[str] = None, name: Optional[str] = None,
                     params: Optional[Dict[str, str]] = None, env: Optional[Dict[str, str]] = None,
                     enabled: bool = True, replace: bool = False,
                     transport: Optional[str] = None, command: Optional[str] = None,
                     args: Optional[List[str]] = None, url: Optional[str] = None,
                     auth: Optional[Dict[str, Any]] = None, description: str = "",
                     agent_hint: str = "") -> Dict[str, Any]:
    """Add a server from a catalog template (``catalog_id``) or a custom definition.

    Raises ``ValueError`` with a user-facing message on a missing parameter, a
    duplicate name or an invalid definition. Supplied ``env`` values are stored
    as secrets under the server's name; the roster only keeps ``${VAR}`` refs.
    """
    params = dict(params or {})
    if catalog_id:
        entry = catalog_entry(catalog_id)
        if entry is None:
            raise ValueError(f"unknown catalog entry {catalog_id!r}")
        missing = missing_params(entry, params)
        if missing:
            raise ValueError("missing required parameter(s): " + ", ".join(missing))
        server = MCPServerConfig.from_dict(instantiate(entry, name=name, params=params, enabled=enabled,
                                                       env_values=env))
    else:
        if not name:
            raise ValueError("a custom server needs a name")
        transport = (transport or "stdio").strip().lower()
        custom_env = {k: "${" + k + "}" for k in (env or {})} if transport == "stdio" else {}
        server = MCPServerConfig(
            name=name.strip(), transport=transport, command=(command or "").strip() or None,
            args=[str(a) for a in (args or [])], url=(url or "").strip() or None, enabled=enabled,
            auth=dict(auth or {}), env=custom_env, description=description or "", agent_hint=agent_hint or "")
    added = add_server(server, replace=replace)
    _store_secrets(added.name, env or {})
    return server_row(added)


def remove(name: str) -> Dict[str, Any]:
    removed = remove_server(name)
    n_secrets = _secrets.delete_secrets(name) if removed else 0
    return {"ok": removed, "name": name, "removed_secrets": n_secrets}


def set_env(name: str, env: Dict[str, str]) -> Dict[str, Any]:
    servers = {s.name: s for s in load_mcp_config()}
    server = servers.get((name or "").strip())
    if server is None:
        raise ValueError(f"server {name!r} not found")
    stored = _store_secrets(server.name, env)
    # A custom stdio server learns about a new variable the first time it is set.
    if server.transport == "stdio":
        changed = False
        for var in (env or {}):
            if var and var not in server.env and env.get(var):
                server.env[var] = "${" + var + "}"
                changed = True
        if changed:
            add_server(server, replace=True)
    return {"ok": True, "name": server.name, "stored": stored,
            "missing_env": MCPServerConfig.from_dict(server.to_dict()).missing_env()}


def toggle(name: str, enabled: bool) -> Dict[str, Any]:
    if not set_enabled(name, enabled):
        return {"ok": False, "reason": f"server {name!r} not found"}
    return {"ok": True, "name": name, "enabled": bool(enabled)}
