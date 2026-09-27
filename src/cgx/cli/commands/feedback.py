"""``cgx feedback`` -- record + inspect thumbs up/down ratings.

Mirrors the web-UI feedback routes (``src/cgx/webui/routes/feedback.py``)
exactly: same engine calls into :mod:`cgx.feedback`, same rating validation,
same optional filters. ``add`` mirrors ``POST /feedback`` (build a
:class:`~cgx.feedback.Feedback` and persist it through the process-wide
:func:`~cgx.feedback.get_default_store`), ``list`` mirrors ``GET /feedback``
and ``stats`` mirrors ``GET /feedback/stats``. Writes share the one feedback
DB (``~/.cgx/feedback.db``) the web UI writes through.
"""

from __future__ import annotations

import argparse

from cgx.cli import _render

_LIST_COLUMNS = [
    ("ID", "feedback_id"),
    ("RATING", "rating"),
    ("KIND", "kind"),
    ("RUN", "run_id"),
    ("MODEL", "model"),
    ("QUESTION", "question"),
    ("COMMENT", "comment"),
]


def register(sub) -> None:
    p = sub.add_parser(
        "feedback",
        help="Record + inspect thumbs up/down ratings (stats/list/add).")
    verbs = p.add_subparsers(dest="feedback_cmd", required=True)

    p_stats = verbs.add_parser(
        "stats", help="Aggregate up/down counts + satisfaction.")
    p_stats.add_argument(
        "--since", type=float, default=None,
        help="Only count feedback recorded at/after this epoch time.")
    p_stats.set_defaults(func=_stats)

    p_list = verbs.add_parser(
        "list", help="List recent feedback (most recent first).")
    p_list.add_argument("--rating", default=None,
                        help="Filter by rating ('up' or 'down').")
    p_list.add_argument("--kind", default=None,
                        help="Filter by kind ('ask' or 'plan').")
    p_list.add_argument("--run-id", default=None,
                        help="Filter to a single run's ratings.")
    p_list.add_argument("--since", type=float, default=None,
                        help="Only rows recorded at/after this epoch time.")
    p_list.add_argument("--limit", type=int, default=100,
                        help="Max rows to return (1-1000, default: 100).")
    p_list.add_argument("--table", action="store_true",
                        help="Human-readable table instead of JSON.")
    p_list.set_defaults(func=_list)

    p_add = verbs.add_parser("add", help="Record one thumbs up/down rating.")
    p_add.add_argument("--rating", required=True, metavar="up|down",
                       help="The rating: 'up' or 'down' (required).")
    p_add.add_argument("--run-id", default=None,
                       help="Run id echoed from the ask/plan meta.")
    p_add.add_argument("--session-id", default=None,
                       help="Session id the rating belongs to.")
    p_add.add_argument("--kind", default="ask",
                       help="What was rated: 'ask' or 'plan' (default: ask).")
    p_add.add_argument("--comment", default=None,
                       help="Optional free-text comment.")
    p_add.add_argument("--question", default=None,
                       help="The question/task that was rated.")
    p_add.add_argument("--answer-preview", default=None,
                       help="Preview of the rated answer (truncated to 2000).")
    p_add.add_argument("--model", default=None,
                       help="Model that produced the rated result.")
    p_add.add_argument("--prompt-version", default=None,
                       help="Prompt version echoed from the ask/plan meta.")
    p_add.add_argument("--label", action="append", default=None,
                       metavar="KEY=VALUE",
                       help="Attach a label (repeatable).")
    p_add.set_defaults(func=_add)


def _stats(args: argparse.Namespace) -> None:
    from cgx.feedback import get_default_store
    _render.print_json(get_default_store().stats(since=args.since))


def _list(args: argparse.Namespace) -> None:
    from cgx.feedback import get_default_store
    if not 1 <= args.limit <= 1000:
        _render.die("--limit must be between 1 and 1000", code=3)
    rows = get_default_store().recent(
        limit=args.limit, rating=args.rating, kind=args.kind,
        run_id=args.run_id, since=args.since)
    _render.emit(rows, table=getattr(args, "table", False),
                 columns=_LIST_COLUMNS)


def _add(args: argparse.Namespace) -> None:
    from cgx.feedback import Feedback, get_default_store
    rating = (args.rating or "").strip().lower()
    if rating not in ("up", "down"):
        _render.die("rating must be 'up' or 'down'", code=3)
    fb = Feedback(
        rating=rating, run_id=args.run_id, session_id=args.session_id,
        kind=(args.kind or "ask"), comment=(args.comment or ""),
        question=(args.question or ""),
        answer_preview=(args.answer_preview or "")[:2000],
        model=args.model, prompt_version=args.prompt_version,
        labels=_parse_labels(args.label),
    )
    fid = get_default_store().record(fb)
    _render.print_json({"ok": True, "feedback_id": fid})


def _parse_labels(pairs) -> dict:
    """Parse repeatable ``--label KEY=VALUE`` flags into the labels dict."""
    labels: dict = {}
    for item in pairs or []:
        key, sep, value = item.partition("=")
        if not sep:
            _render.die(f"--label must be KEY=VALUE, got {item!r}", code=3)
        labels[key] = value
    return labels

