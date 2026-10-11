"""``cgx mcp`` -- manage MCP tool servers from the terminal.

Mirrors the web-UI MCP routes (``src/cgx/webui/routes/mcp.py``) verb for verb,
through the same :mod:`cgx.mcp.ops` functions -- never through ``cgx.webui``,
so the base ``cgx`` install needs no FastAPI:

* ``list``      -> ``GET  /mcp/servers``   (roster + catalog metadata + missing secrets)
* ``catalog``   -> ``GET  /mcp/catalog``   (curated developer servers)
* ``add``       -> ``POST /mcp/servers``   (from the catalog, with --param / --env)
* ``add-custom``-> ``POST /mcp/servers``   (a hand-written stdio/http server)
* ``rm``        -> ``DELETE /mcp/servers/{name}``
* ``set-env``   -> ``POST /mcp/servers/{name}/env``
* ``test``      -> ``POST /mcp/servers/{name}/test``
* ``enable`` / ``disable`` -> ``POST /mcp/toggle``
* ``tools``     -> the swarm's ``mcp_list_tools`` registry tool

The ``mcp`` SDK is optional: everything except ``test``/``tools`` works without it.
"""

from __future__ import annotations

import argparse
from typing import Dict, List

from cgx.cli import _render

_LIST_COLUMNS = [
    ("NAME", "name"),
    ("TRANSPORT", "transport"),
    ("ENABLED", "enabled"),
    ("READY", "ready"),
    ("MISSING_ENV", "missing_env"),
    ("CATALOG", "catalog_id"),
]
_CATALOG_COLUMNS = [
    ("ID", "id"),
    ("NAME", "name"),
    ("CATEGORY", "category"),
    ("STATUS", "status"),
    ("INSTALLED", "installed"),
    ("DESCRIPTION", "description"),
]


def _kv(pairs: List[str], label: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in pairs or []:
        if "=" not in item:
            _render.die(f"{label} expects KEY=VALUE, got {item!r}", code=3)
        k, v = item.split("=", 1)
        out[k.strip()] = v
    return out


def register(sub) -> None:
    p = sub.add_parser("mcp", help="Manage MCP tool servers (list/catalog/add/rm/test/set-env/enable/disable/tools).")
    verbs = p.add_subparsers(dest="mcp_cmd", required=True)

    p_list = verbs.add_parser("list", help="List configured MCP servers + SDK status.")
    p_list.add_argument("--table", action="store_true", help="Human-readable table instead of JSON.")
    p_list.set_defaults(func=_list)

    p_cat = verbs.add_parser("catalog", help="Show the curated catalog of developer MCP servers.")
    p_cat.add_argument("--category", default=None, help="Filter by category id (e.g. docs, database).")
    p_cat.add_argument("--table", action="store_true", help="Human-readable table instead of JSON.")
    p_cat.set_defaults(func=_catalog)

    p_add = verbs.add_parser("add", help="Add a server from the catalog (cgx mcp catalog).")
    p_add.add_argument("catalog_id")
    p_add.add_argument("--name", default=None, help="Roster name (default: the catalog id).")
    p_add.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                       help="Fill a ${PARAM} placeholder (repeatable).")
    p_add.add_argument("--env", action="append", default=[], metavar="VAR=VALUE",
                       help="Store a secret/env value for the server (repeatable; kept in the keyring).")
    p_add.add_argument("--disabled", action="store_true", help="Add but keep hidden from the agent.")
    p_add.add_argument("--replace", action="store_true", help="Overwrite a same-named server.")
    p_add.set_defaults(func=_add)

    p_custom = verbs.add_parser("add-custom", help="Add a hand-written stdio or http server.")
    p_custom.add_argument("name")
    p_custom.add_argument("--transport", default="stdio", choices=["stdio", "http"])
    p_custom.add_argument("--command", default=None, help="stdio: executable to launch.")
    p_custom.add_argument("--arg", action="append", default=[], help="stdio: argument (repeatable).")
    p_custom.add_argument("--url", default=None, help="http: streamable HTTP endpoint.")
    p_custom.add_argument("--token-env", default=None,
                          help="http: env var holding the bearer token (value via --env).")
    p_custom.add_argument("--env", action="append", default=[], metavar="VAR=VALUE",
                          help="Secret/env value passed to the server (repeatable).")
    p_custom.add_argument("--description", default="")
    p_custom.add_argument("--agent-hint", default="", help="One line telling the agent when to use it.")
    p_custom.add_argument("--disabled", action="store_true")
    p_custom.add_argument("--replace", action="store_true")
    p_custom.set_defaults(func=_add_custom)

    p_rm = verbs.add_parser("rm", help="Remove a server (and its stored secrets).")
    p_rm.add_argument("name")
    p_rm.set_defaults(func=_rm)

    p_env = verbs.add_parser("set-env", help="Store secret/env values for a server (VAR= clears).")
    p_env.add_argument("name")
    p_env.add_argument("pairs", nargs="+", metavar="VAR=VALUE")
    p_env.set_defaults(func=_set_env)

    p_test = verbs.add_parser("test", help="Connect to a server and list its tools.")
    p_test.add_argument("name")
    p_test.set_defaults(func=_test)

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
    from cgx.mcp import ops
    from cgx.mcp.config import default_config_path
    rows = ops.server_rows()
    if getattr(args, "table", False):
        _render.emit([{**r, "missing_env": ",".join(r.get("missing_env") or []) or "-"} for r in rows],
                     table=True, columns=_LIST_COLUMNS)
        return
    _render.print_json({"sdk_installed": ops.sdk_installed(),
                        "config_path": str(default_config_path()), "servers": rows})


def _catalog(args: argparse.Namespace) -> None:
    from cgx.mcp import ops
    payload = ops.catalog_payload(getattr(args, "category", None))
    if getattr(args, "table", False):
        _render.emit(payload["servers"], table=True, columns=_CATALOG_COLUMNS)
        return
    _render.print_json(payload)


def _add(args: argparse.Namespace) -> None:
    from cgx.mcp import ops
    try:
        row = ops.add_from_request(
            catalog_id=args.catalog_id, name=args.name, params=_kv(args.param, "--param"),
            env=_kv(args.env, "--env"), enabled=not args.disabled, replace=args.replace)
    except ValueError as exc:
        _render.die(str(exc), code=3)
    _render.print_json(row)


def _add_custom(args: argparse.Namespace) -> None:
    from cgx.mcp import ops
    auth = {"type": "bearer", "token_env": args.token_env} if args.token_env else {}
    try:
        row = ops.add_from_request(
            name=args.name, transport=args.transport, command=args.command, args=list(args.arg or []),
            url=args.url, auth=auth, env=_kv(args.env, "--env"), enabled=not args.disabled,
            replace=args.replace, description=args.description, agent_hint=args.agent_hint)
    except ValueError as exc:
        _render.die(str(exc), code=3)
    _render.print_json(row)


def _rm(args: argparse.Namespace) -> None:
    from cgx.mcp import ops
    out = ops.remove(args.name)
    if not out["ok"]:
        _render.die(f"server {args.name!r} not found", code=2)
    _render.print_json(out)


def _set_env(args: argparse.Namespace) -> None:
    from cgx.mcp import ops
    try:
        _render.print_json(ops.set_env(args.name, _kv(args.pairs, "set-env")))
    except ValueError as exc:
        _render.die(str(exc), code=2)


def _test(args: argparse.Namespace) -> None:
    from cgx.mcp.manager import test_server
    out = test_server(args.name)
    _render.print_json(out)
    if not out.get("ok"):
        raise SystemExit(1)


def _toggle(name: str, enabled: bool) -> None:
    from cgx.mcp import ops
    out = ops.toggle(name, enabled)
    if not out.get("ok"):
        _render.die(out.get("reason") or f"server {name!r} not found", code=2)
    _render.print_json(out)


def _enable(args: argparse.Namespace) -> None:
    _toggle(args.name, True)


def _disable(args: argparse.Namespace) -> None:
    _toggle(args.name, False)


def _tools(args: argparse.Namespace) -> None:
    from cgx.mcp import manager
    # manager.list_tools returns a ready-to-read text listing (or an actionable
    # message when the SDK is absent / the server is unknown or disabled).
    print(manager.list_tools({"server": args.name}, None))
