"""CLI <-> Web-UI parity guard.

The `cgx` CLI must expose the same capabilities as the web UI. Each web-UI
feature is a route module under ``cgx/webui/routes/``; this test discovers
them all and asserts every one is either:

  * mapped to a CLI command group (``PARITY_MAP``), or
  * explicitly exempted with a reason (``EXEMPT`` -- web-internal plumbing or a
    tracked known gap).

If someone adds a new route module, this test fails until they add a matching
CLI command (preferred) or record why it has none. That is the ratchet that
keeps the CLI from drifting behind the UI again.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from cgx.cli.main import build_parser

ROUTES_DIR = Path(__file__).resolve().parents[1] / "src" / "cgx" / "webui" / "routes"

# route module (under cgx/webui/routes/) -> CLI command noun that covers it.
PARITY_MAP = {
    "activity": "activity",
    "admin": "admin",
    "agent_profiles": "agent-profile",
    "agent_session": "session",
    "approvals": "approvals",
    "ask": "ask",
    "context_file": "context",
    "embed": "model",        # model embed-list / embed-pull
    "feedback": "feedback",
    "govdata": "govdata",
    "hardware": "model",     # model matrix / fit
    "health": "health",
    "index": "index",
    "mcp": "mcp",
    "metrics": "metrics",
    "monitor": "monitor",
    "plan": "plan",
    "profiles": "profile",
    "rollback": "rollback",
    "settings": "trace",     # settings/trace runtime toggle
    "setup": "model",        # model list / pull
    "sites": "site",
    "skills": "skills",
    "status": "status",
    "usage": "usage",
}

# Route modules with no CLI command, each with a rationale.
EXEMPT = {
    "tasks": "web-internal async task/SSE plumbing for the SPA; no user-facing CLI verb",
    "sessions": "UI chat-history convenience store; the CLI's `ask` is stateless "
                "(tracked gap -- add `cgx chat` if history is wanted on the CLI)",
}


def _cli_nouns() -> set[str]:
    parser = build_parser()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices.keys())
    raise AssertionError("cgx parser exposes no subcommands")


def _route_modules() -> set[str]:
    return {
        p.stem for p in ROUTES_DIR.glob("*.py")
        if p.stem != "__init__"
    }


def test_every_route_is_mapped_or_exempt():
    """A new web-UI route must be given a CLI command or an explicit exemption."""
    routes = _route_modules()
    covered = set(PARITY_MAP) | set(EXEMPT)
    missing = routes - covered
    assert not missing, (
        "New web-UI route module(s) with no CLI command or exemption: "
        f"{sorted(missing)}. Add a `cgx <noun>` command (preferred) and map it in "
        "PARITY_MAP, or record a rationale in EXEMPT."
    )


@pytest.mark.parametrize("route,noun", sorted(PARITY_MAP.items()))
def test_mapped_command_is_registered(route: str, noun: str):
    """Every mapped CLI noun is actually registered on the parser."""
    assert noun in _cli_nouns(), (
        f"web-UI route {route!r} maps to `cgx {noun}` but that command is not "
        "registered (see cgx.cli.commands.COMMAND_MODULES / cgx.cli.main)."
    )


def test_command_modules_import_and_register():
    """Every module in COMMAND_MODULES imports and registers without error."""
    from cgx.cli import commands
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    commands.register_all(sub)  # raises if any module is broken
    assert set(commands.COMMAND_MODULES), "no parity command modules registered"
