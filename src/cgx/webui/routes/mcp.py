"""MCP server management surface.

Lists the configured MCP tool servers (from ``~/.cgx/mcp.json``) and reports
whether the optional SDK is installed. Read-oriented: editing the roster is a
config-file edit (the feature's "easy to extend" contract), so this exposes
listing plus a lightweight enable/disable toggle written back to the file.
"""

from __future__ import annotations

import json

from fastapi import APIRouter
from pydantic import BaseModel

from cgx.mcp.config import default_config_path, load_mcp_config

router = APIRouter(tags=["mcp"])


def _sdk_installed() -> bool:
    try:
        import mcp  # noqa: F401
        return True
    except Exception:
        return False


@router.get("/mcp/servers")
def servers() -> dict:
    """Configured MCP servers and SDK availability."""
    cfg = load_mcp_config()
    return {
        "sdk_installed": _sdk_installed(),
        "config_path": str(default_config_path()),
        "servers": [
            {"name": s.name, "transport": s.transport, "url": s.url,
             "command": s.command, "enabled": s.enabled}
            for s in cfg
        ],
    }


class ToggleBody(BaseModel):
    name: str
    enabled: bool


@router.post("/mcp/toggle")
def toggle(body: ToggleBody) -> dict:
    """Enable or disable a configured server, persisting to the JSON roster."""
    path = default_config_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"ok": False, "reason": "no readable mcp.json"}
    changed = False
    for entry in data.get("servers", []):
        if isinstance(entry, dict) and entry.get("name") == body.name:
            entry["enabled"] = body.enabled
            changed = True
    if not changed:
        return {"ok": False, "reason": f"server {body.name!r} not found"}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return {"ok": True}
