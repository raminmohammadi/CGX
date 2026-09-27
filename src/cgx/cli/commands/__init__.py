"""Parity subcommands for the ``cgx`` CLI.

Each module here exposes a ``register(subparsers)`` function that attaches a
noun-verb command group (``cgx skills ...``, ``cgx site ...``) whose leaves
are thin adapters over the same engine functions the web-UI routes call --
never over ``cgx.webui`` -- so the base ``cgx`` install needs no FastAPI.

New command groups are added by dropping a module here and listing it in
``COMMAND_MODULES``. ``register_all`` is invoked once from
:func:`cgx.cli.main.main`.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

# Ordered list of command-group modules. Each must define ``register``.
COMMAND_MODULES: list[str] = [
    "skills",
    "profile",
    "agent_profile",
    "site",
    "session",
    "context",
    "mcp",
    "activity",
    "usage",
    "feedback",
    "monitor",
    "govdata",
    "admin",
    "model",
    "trace",
    "metrics",
    "health",
    "rollback",
    "approvals",
]


def register_all(subparsers: Any) -> None:
    """Import and register every command group onto ``subparsers``."""
    for name in COMMAND_MODULES:
        import_module(f"cgx.cli.commands.{name}").register(subparsers)
