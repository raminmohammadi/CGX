"""``cgx monitor`` -- list AIOps alerts (and run a check on demand).

Mirrors the web-UI monitor route (``src/cgx/webui/routes/monitor.py``):
``alerts`` is the CLI twin of ``GET /monitor/alerts``, driving the same
process-wide :class:`cgx.monitor.Monitor` via ``get_default_monitor()`` and
its read-only ``recent()`` proxy with the identical ``limit/severity/code/
since`` filters. ``check`` has no web twin -- it drives the monitor's
``observe_*`` observers to run one quality/drift/cost check over a supplied
JSON payload, persisting any findings exactly as the live ask/plan paths do
(so they then surface under ``alerts``).
"""

from __future__ import annotations

import argparse
import json

from cgx.cli import _render

# Compact triage columns for the alert list/table (a subset of each record).
_ALERT_COLUMNS = [
    ("CREATED", "created_at"),
    ("SEVERITY", "severity"),
    ("CODE", "code"),
    ("VALUE", "value"),
    ("THRESHOLD", "threshold"),
    ("MESSAGE", "message"),
]

# ``check --kind`` -> (Monitor observer method, whether a --baseline is required).
_CHECK_KINDS = {
    "answer": ("observe_answer", False),
    "codegen": ("observe_codegen", False),
    "retrieval": ("observe_retrieval", True),
    "cost": ("observe_cost", True),
}


def register(sub) -> None:
    p = sub.add_parser("monitor", help="AIOps alerts (list recent, run a check).")
    verbs = p.add_subparsers(dest="monitor_cmd", required=True)

    p_alerts = verbs.add_parser(
        "alerts", help="List recent alerts (most recent first), optionally filtered.")
    p_alerts.add_argument("--limit", type=int, default=100,
                          help="Max alerts to return (1-1000, default 100).")
    p_alerts.add_argument("--severity", default=None,
                          help="Filter by severity (info/warning/critical).")
    p_alerts.add_argument("--code", default=None,
                          help="Filter by alert code (e.g. low_confidence).")
    p_alerts.add_argument("--since", type=float, default=None,
                          help="Only alerts created at/after this epoch time.")
    p_alerts.add_argument("--table", action="store_true",
                          help="Human-readable table instead of JSON.")
    p_alerts.set_defaults(func=_alerts)

    p_check = verbs.add_parser(
        "check", help="Run one monitor check over a JSON payload and record findings.")
    p_check.add_argument("--kind", required=True, choices=sorted(_CHECK_KINDS),
                         help="Which check to run (answer/codegen/retrieval/cost).")
    p_check.add_argument("--file", "-f", default=None,
                         help="Payload JSON file (default: read stdin).")
    p_check.add_argument("--baseline", default=None,
                         help="Baseline JSON file (required for retrieval/cost).")
    p_check.add_argument("--run-id", default=None,
                         help="Optional run id to tag the resulting alerts.")
    p_check.add_argument("--table", action="store_true",
                         help="Human-readable table instead of JSON.")
    p_check.set_defaults(func=_check)


def _alerts(args: argparse.Namespace) -> None:
    from cgx.monitor import get_default_monitor
    if not 1 <= args.limit <= 1000:  # mirrors the route's Query(ge=1, le=1000) -> 422
        _render.die("--limit must be between 1 and 1000", code=3)
    alerts = get_default_monitor().recent(
        limit=args.limit, severity=args.severity, code=args.code, since=args.since)
    _render.emit(alerts, table=getattr(args, "table", False), columns=_ALERT_COLUMNS)


def _check(args: argparse.Namespace) -> None:
    from cgx.monitor import get_default_monitor
    method_name, needs_baseline = _CHECK_KINDS[args.kind]
    payload = _load_json(_render.read_text_input(args.file), "payload")
    observe = getattr(get_default_monitor(), method_name)
    if needs_baseline:
        if not args.baseline:
            _render.die(f"--baseline is required for --kind {args.kind}", code=3)
        baseline = _load_json(_render.read_text_input(args.baseline), "baseline")
        alerts = observe(payload, baseline, run_id=args.run_id)
    else:
        alerts = observe(payload, run_id=args.run_id)
    _render.emit([a.to_dict() for a in alerts],
                 table=getattr(args, "table", False), columns=_ALERT_COLUMNS)


def _load_json(text: str, what: str) -> dict:
    """Parse ``text`` as a JSON object, or ``die`` with a validation error."""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        _render.die(f"{what} is not valid JSON: {exc}", code=3)
    if not isinstance(obj, dict):
        _render.die(f"{what} must be a JSON object", code=3)
    return obj

