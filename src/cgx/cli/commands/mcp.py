"""``cgx mcp`` -- list / enable / disable configured MCP tool servers.

Mirrors the web-UI MCP routes (``src/cgx/webui/routes/mcp.py``):

* ``list``  -> ``GET  /mcp/servers`` (roster + SDK availability + config path,
  via :func:`cgx.mcp.config.load_mcp_config` + :func:`default_config_path`;
  SDK presence is probed by attempting ``import mcp``).
* ``enable`` / ``disable`` NAME -> ``POST /mcp/toggle`` (persist the ``enabled``
  flag back to the JSON roster; the request body's ``enabled`` bool is expressed
  by the verb rather than a flag).
* ``tools`` NAME mirrors the swarm's ``mcp_list_tools`` registry tool
  (:func:`cgx.mcp.manager.list_tools`).

Same engine calls as the routes, invoked directly -- never through
``cgx.webui`` -- so the base ``cgx`` install needs no FastAPI. The ``mcp`` SDK
is optional: ``list`` works without it (``sdk_installed`` just reports
``false``); ``tools`` needs the SDK plus a live, enabled server.
"""

from __future__ import annotations

import argparse

from cgx.cli import _render

_LIST_COLUMNS = [
    ("NAME", "name"),
    ("TRANSPORT", "transport"),
    ("ENABLED", "enabled"),
    ("URL", "url"),
    ("COMMAND", "command"),
]


def _sdk_installed() -> bool:
    """Whether the optional ``mcp`` SDK is importable (mirrors the route)."""
    try:
        import mcp  # noqa: F401
        return True
    except Exception:
        return False


def register(sub) -> None:
    p = sub.add_parser("mcp", help="Manage MCP tool servers (list/enable/disable/tools).")
    verbs = p.add_subparsers(dest="mcp_cmd", required=True)

    p_list = verbs.add_parser("list", help="List configured MCP servers + SDK status.")
    p_list.add_argument("--table", action="store_true",
                        help="Human-readable table instead of JSON.")
    p_list.set_defaults(func=_list)

    p_enable = verbs.add_parser("enable", help="Enable a configured server (persisted to mcp.json).")
    p_enable.add_argument("name")
    p_enable.set_defaults(func=_enable)

    p_disable = verbs.add_parser("disable", help="Disable a configured server (persisted to mcp.json).")
    p_disable.add_argument("name")
    p_disable.set_defaults(func=_disable)

    p_tools = verbs.add_parser("tools", help="List the tools an enabled server exposes (needs the SDK + a live server).")
    p_tools.add_argument("name")
    p_tools.set_defaults(func=_tools)


def _list(args: argparse.Namespace) -> None:
    from cgx.mcp.config import default_config_path, load_mcp_config
    servers = [
        {"name": s.name, "transport": s.transport, "url": s.url,
         "command": s.command, "enabled": s.enabled}
        for s in load_mcp_config()
    ]
    # --table shows the server roster; JSON (default) mirrors GET /mcp/servers
    # exactly -- servers wrapped with sdk_installed + config_path metadata.
    if getattr(args, "table", False):
        _render.emit(servers, table=True, columns=_LIST_COLUMNS)
        return
    _render.print_json({
        "sdk_installed": _sdk_installed(),
        "config_path": str(default_config_path()),
        "servers": servers,
    })


def _toggle(name: str, enabled: bool) -> None:
    """Persist ``enabled`` for the named server (mirrors POST /mcp/toggle)."""
    import json

    from cgx.mcp.config import default_config_path
    path = default_config_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        _render.die("no readable mcp.json", code=3)
    changed = False
    for entry in data.get("servers", []):
        if isinstance(entry, dict) and entry.get("name") == name:
            entry["enabled"] = enabled
            changed = True
    if not changed:
        _render.die(f"server {name!r} not found", code=2)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    _render.print_json({"ok": True, "name": name, "enabled": enabled})


def _enable(args: argparse.Namespace) -> None:
    _toggle(args.name, True)


def _disable(args: argparse.Namespace) -> None:
    _toggle(args.name, False)


def _tools(args: argparse.Namespace) -> None:
    from cgx.mcp import manager
    # manager.list_tools returns a ready-to-read text listing (or an actionable
    # message when the SDK is absent / the server is unknown or disabled); print
    # it raw rather than JSON-wrapping it.
    print(manager.list_tools({"server": args.name}, None))

