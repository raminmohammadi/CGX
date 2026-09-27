"""``cgx approvals`` -- list / resolve human-in-the-loop approval requests.

Mirrors ``/api/approvals/pending`` and ``/api/approvals/resolve``
(:mod:`cgx.session.approval`). The approval gate registry is *in-process*:
``resolve`` only affects a session gate active in this same process (e.g. an
embedding/library use of the agent loop). Across separate processes there is
no shared gate to resolve -- for unattended terminal runs use
``cgx agent --approve`` instead, which prompts inline.
"""

from __future__ import annotations

import argparse

from cgx.cli import _render

_COLUMNS = [
    ("REQUEST", "request_id"),
    ("SESSION", "session_id"),
    ("TOOL", "tool"),
    ("RISK", "risk"),
]


def register(sub) -> None:
    p = sub.add_parser("approvals", help="List/resolve pending approval requests (in-process).")
    verbs = p.add_subparsers(dest="approvals_cmd", required=True)

    p_pend = verbs.add_parser("pending", help="List requests awaiting a decision.")
    p_pend.add_argument("--table", action="store_true", help="Human table instead of JSON.")
    p_pend.set_defaults(func=_pending)

    p_res = verbs.add_parser("resolve", help="Approve or deny a pending request.")
    p_res.add_argument("--session-id", required=True)
    p_res.add_argument("--request-id", required=True)
    grp = p_res.add_mutually_exclusive_group(required=True)
    grp.add_argument("--approve", dest="approved", action="store_true", help="Approve the request.")
    grp.add_argument("--deny", dest="approved", action="store_false", help="Deny the request.")
    p_res.add_argument("--reason", default="", help="Optional rationale recorded with the decision.")
    p_res.set_defaults(func=_resolve)


def _pending(args: argparse.Namespace) -> None:
    from cgx.session.approval import all_pending
    _render.emit(all_pending(), table=getattr(args, "table", False), columns=_COLUMNS)


def _resolve(args: argparse.Namespace) -> None:
    from cgx.session.approval import ApprovalDecision, get_gate
    gate = get_gate(args.session_id)
    if gate is None:
        _render.die("no active gate for session (approval gates are in-process only)", code=2)
    ok = gate.resolve(args.request_id,
                      ApprovalDecision(approved=args.approved, reason=args.reason))
    _render.print_json({"ok": ok})
