"""``cgx trace`` -- inspect / flip the runtime function-call trace toggle.

Mirrors GET/POST ``/api/settings/trace`` (see ``webui/routes/settings.py``),
backed by :mod:`cgx.trace`. When ``CGX_TRACE`` pins the flag from the
environment, ``on``/``off`` refuse to change it (matching the route's 409).
"""

from __future__ import annotations

import argparse

from cgx.cli import _render


def register(sub) -> None:
    p = sub.add_parser("trace", help="Runtime trace toggle (status/on/off).")
    verbs = p.add_subparsers(dest="trace_cmd", required=True)
    verbs.add_parser("status", help="Show the trace state and how it's pinned.").set_defaults(func=_status)
    verbs.add_parser("on", help="Enable tracing (refused if pinned by CGX_TRACE).").set_defaults(func=_on)
    verbs.add_parser("off", help="Disable tracing (refused if pinned by CGX_TRACE).").set_defaults(func=_off)


def _status(args: argparse.Namespace) -> None:
    from cgx.trace import is_trace_enabled, trace_source
    _render.print_json({"enabled": is_trace_enabled(), "source": trace_source()})


def _apply(enabled: bool) -> None:
    from cgx.trace import is_trace_enabled, set_trace_enabled, trace_source
    if trace_source() == "env":
        _render.die("trace toggle is pinned by the CGX_TRACE environment variable; "
                    "unset it to control the flag here", code=3)
    set_trace_enabled(enabled)
    _render.print_json({"enabled": is_trace_enabled(), "source": trace_source()})


def _on(args: argparse.Namespace) -> None:
    _apply(True)


def _off(args: argparse.Namespace) -> None:
    _apply(False)
