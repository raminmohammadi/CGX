"""``cgx usage`` -- per-owner LLM cost & quota usage (read-only).

Mirrors the cost & quota usage read API (``src/cgx/webui/routes/usage.py``)
exactly: same ``cgx.governance`` calls, same current-day-window semantics.
``show`` reports one owner's current-day totals merged with their budget
state (``GET /usage``); ``summary`` reports per-owner totals for the day for
the admin cost dashboard (``GET /usage/summary``). Read-only -- spend is
recorded by the governed provider on the ask/plan + agent-session paths, never
by these commands.
"""

from __future__ import annotations

import argparse

from cgx.cli import _render

# Columns for the ``summary`` table -- keys are :meth:`UsageMeter.totals` fields.
_SUMMARY_COLUMNS = [
    ("OWNER", "owner"),
    ("DAY", "day"),
    ("CALLS", "calls"),
    ("TOKENS_IN", "tokens_in"),
    ("TOKENS_OUT", "tokens_out"),
    ("TOKENS", "tokens_total"),
    ("COST_USD", "cost_usd"),
]


def register(sub) -> None:
    p = sub.add_parser("usage",
                       help="Per-owner LLM cost & quota usage (show/summary).")
    verbs = p.add_subparsers(dest="usage_cmd", required=True)

    p_show = verbs.add_parser(
        "show", help="One owner's current-day totals + budget state.")
    p_show.add_argument("--owner", default=None,
                        help="Owner id (default: the resolved active owner).")
    p_show.add_argument("--day", default=None,
                        help="UTC day bucket YYYY-MM-DD (default: today).")
    p_show.set_defaults(func=_show)

    p_summary = verbs.add_parser(
        "summary", help="Per-owner totals for the day (admin cost dashboard).")
    p_summary.add_argument("--day", default=None,
                           help="UTC day bucket YYYY-MM-DD (default: today).")
    p_summary.add_argument("--table", action="store_true",
                           help="Human-readable table instead of JSON.")
    p_summary.set_defaults(func=_summary)


def _show(args: argparse.Namespace) -> None:
    try:
        from cgx.governance import get_default_quota_manager, resolve_owner
        mgr = get_default_quota_manager()
        who = args.owner or resolve_owner()
        status = mgr.check(who, enforce=False)
        status.update(mgr.meter.totals(who, day=args.day))
    except Exception as exc:  # defensive: mirror the route's 500 read-failure path
        _render.die(f"could not read usage: {exc}")
    _render.print_json(status)


def _summary(args: argparse.Namespace) -> None:
    try:
        from cgx.governance import get_default_quota_manager
        rows = get_default_quota_manager().meter.summary(day=args.day)
    except Exception:  # defensive: mirror the route's graceful degrade to []
        rows = []
    _render.emit(rows, table=getattr(args, "table", False),
                 columns=_SUMMARY_COLUMNS)

