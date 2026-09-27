"""``cgx activity`` -- inspect recorded ask/plan/agent runs.

Mirrors the web-UI activity routes (``src/cgx/webui/routes/activity.py``):
read-only views over the per-run observation store
(:class:`cgx.activity.RunStore`, reached via ``get_default_run_store()``).
``runs`` is the filtered, most-recent-first list; ``summary`` is the aggregate
totals for the dashboard; ``show`` joins one run to its feedback (Subsystem H)
and monitor/guardrail alerts (Subsystems G/K). Runs are recorded by the
ask/plan/agent handlers -- this group only reads them. Calls the same engine
functions the route calls, directly (never through ``cgx.webui``).
"""

from __future__ import annotations

import argparse

from cgx.cli import _render

# Columns for the ``runs --table`` view (mirrors the run dict keys).
_RUN_COLUMNS = [
    ("RUN_ID", "run_id"),
    ("KIND", "kind"),
    ("MODEL", "model"),
    ("STATUS", "status"),
    ("TOKENS", "tokens_total"),
    ("COST", "cost_usd"),
    ("OWNER", "owner"),
    ("CREATED", "created_at"),
]


def register(sub) -> None:
    p = sub.add_parser(
        "activity", help="Inspect recorded runs (runs/summary/show).")
    verbs = p.add_subparsers(dest="activity_cmd", required=True)

    p_runs = verbs.add_parser("runs", help="List recent runs, most-recent first.")
    p_runs.add_argument("--kind", default=None,
                        help="Filter by run kind (ask/plan/agent).")
    p_runs.add_argument("--owner", default=None, help="Filter by run owner.")
    p_runs.add_argument("--status", default=None,
                        help="Filter by status (e.g. ok/error).")
    p_runs.add_argument("--limit", type=int, default=100,
                        help="Max rows to return (1-500, default 100).")
    p_runs.add_argument("--table", action="store_true",
                        help="Human-readable table instead of JSON.")
    p_runs.set_defaults(func=_runs)

    p_sum = verbs.add_parser(
        "summary", help="Aggregate run counts + cost/token totals.")
    p_sum.set_defaults(func=_summary)

    p_show = verbs.add_parser(
        "show", help="Show one run joined to its feedback + alerts.")
    p_show.add_argument("run_id")
    p_show.set_defaults(func=_show)


def _runs(args: argparse.Namespace) -> None:
    from cgx.activity import get_default_run_store
    # Mirror the route's ``Query(100, ge=1, le=500)`` bound -> 422 == code 3.
    if args.limit < 1 or args.limit > 500:
        _render.die("--limit must be between 1 and 500", code=3)
    rows = get_default_run_store().recent(
        limit=args.limit, kind=args.kind, owner=args.owner, status=args.status)
    if getattr(args, "table", False):
        _render.emit(rows, table=True, columns=_RUN_COLUMNS)
    else:
        # Route body shape: {"runs": rows, "count": len(rows)}.
        _render.print_json({"runs": rows, "count": len(rows)})


def _summary(args: argparse.Namespace) -> None:
    from cgx.activity import get_default_run_store
    _render.print_json(get_default_run_store().summary())


def _show(args: argparse.Namespace) -> None:
    from cgx.activity import get_default_run_store
    run = get_default_run_store().get(args.run_id)
    if run is None:
        _render.die(f"run {args.run_id!r} not found", code=2)  # route 404

    # Feedback + alerts are optional subsystems: best-effort, exactly like the
    # route -- a missing/failed lookup yields an empty list, never an error.
    feedback: list = []
    try:
        from cgx.feedback import get_default_store
        feedback = get_default_store().recent(run_id=args.run_id, limit=50)
    except Exception:
        feedback = []

    alerts: list = []
    try:
        from cgx.monitor import get_default_monitor
        alerts = [a for a in get_default_monitor().recent(limit=500)
                  if a.get("run_id") == args.run_id]
    except Exception:
        alerts = []

    _render.print_json({"run": run, "feedback": feedback, "alerts": alerts})

