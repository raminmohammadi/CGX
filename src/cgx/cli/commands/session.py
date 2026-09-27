"""``cgx session`` -- inspect and manage session-shaped agent runs.

Mirrors the read/management half of the agent-session routes
(``src/cgx/webui/routes/agent_session.py``) over :class:`cgx.session.SessionStore`:
``list`` -> GET /agent-session, ``show`` -> GET /agent-session/{sid} (the full
tasks/artifacts/facts/decisions snapshot), ``rm`` -> DELETE /agent-session/{sid}.

*Driving* a session (creating one, taking a turn, answering a decision) is the
job of ``cgx agent "<goal>"`` (one unattended turn, resumable) and the
interactive ``cgx dash``; this group is for scriptable inspection and cleanup
of the sessions those produce (persisted to ``<project>/.cgx/sessions.db``).
"""

from __future__ import annotations

import argparse
import os

from cgx.cli import _render

_LIST_COLUMNS = [
    ("SESSION_ID", "session_id"),
    ("STATUS", "status"),
    ("MODE", "mode"),
    ("TITLE", "title"),
]


def register(sub) -> None:
    p = sub.add_parser("session", help="Inspect/manage agent sessions (list/show/rm).")
    verbs = p.add_subparsers(dest="session_cmd", required=True)

    p_list = verbs.add_parser("list", help="List agent sessions for a project.")
    p_list.add_argument("--project-root", default=None,
                        help="Project whose sessions.db to read (default: cwd).")
    p_list.add_argument("--table", action="store_true", help="Human table instead of JSON.")
    p_list.set_defaults(func=_list)

    p_show = verbs.add_parser("show", help="Full snapshot: tasks, artifacts, facts, decisions.")
    p_show.add_argument("sid")
    p_show.add_argument("--project-root", default=None)
    p_show.set_defaults(func=_show)

    p_rm = verbs.add_parser("rm", help="Delete a session and all its records.")
    p_rm.add_argument("sid")
    p_rm.add_argument("--project-root", default=None)
    p_rm.set_defaults(func=_rm)

    # --- driving verbs (stream live; Ctrl-C cancels, exit 130) ---
    from cgx.cli.main import _add_provider_flags  # provider/index/mode flags

    p_send = verbs.add_parser(
        "send", help="Post a follow-up message to a session and drive one turn.")
    p_send.add_argument("sid")
    p_send.add_argument("message", nargs="+", help="The message to send.")
    p_send.add_argument("--auto", action="store_true",
                        help="Auto-answer clarify/approval questions with defaults "
                             "(unattended, like `cgx agent`).")
    _add_provider_flags(p_send)
    p_send.set_defaults(func=_send)

    p_decide = verbs.add_parser(
        "decide", help="Answer an open decision task with a structured choice.")
    p_decide.add_argument("sid")
    p_decide.add_argument("--task-id", required=True,
                          help="The ASK_USER task id awaiting a decision (see `show`).")
    p_decide.add_argument("--chosen", required=True,
                          help='Decision payload as JSON, e.g. \'{"approved": true}\'.')
    p_decide.add_argument("--rationale", default=None, help="Optional rationale.")
    p_decide.add_argument("--auto", action="store_true",
                          help="Auto-answer any subsequent questions with defaults.")
    _add_provider_flags(p_decide)
    p_decide.set_defaults(func=_decide)


def _root(args: argparse.Namespace) -> str:
    """Absolute project root -- sessions persist their root as an absolute path,
    so the store path *and* the list filter must be absolute to match."""
    return os.path.abspath(args.project_root or os.getcwd())


def _open_store(project_root: str):
    from cgx.session import SessionStore
    return SessionStore(project_root=project_root)


def _list(args: argparse.Namespace) -> None:
    root = _root(args)
    # A non-existent project_root has no sessions; don't materialize a .cgx
    # workspace just to list (mirrors the route's read-only guard).
    if not os.path.isdir(root):
        _render.emit([], table=getattr(args, "table", False), columns=_LIST_COLUMNS)
        return
    store = _open_store(root)
    try:
        rows = [s.to_dict() for s in store.list_sessions(project_root=root)]
    finally:
        store.close()
    _render.emit(rows, table=getattr(args, "table", False), columns=_LIST_COLUMNS)


def _show(args: argparse.Namespace) -> None:
    store = _open_store(_root(args))
    try:
        session = store.get_session(args.sid)
        if session is None:
            _render.die(f"session {args.sid!r} not found", code=2)
        state = {
            "session": session.to_dict(),
            "tasks": [t.to_dict() for t in store.list_tasks(args.sid)],
            "artifacts": [a.to_dict() for a in store.list_artifacts(args.sid)],
            "facts": [f.to_dict() for f in store.load_kb(args.sid).facts.values()],
            "decisions": [d.to_dict() for d in
                          store.load_decisions(args.sid).decisions.values()],
        }
    finally:
        store.close()
    _render.print_json(state)


def _rm(args: argparse.Namespace) -> None:
    store = _open_store(_root(args))
    try:
        removed = store.delete_session(args.sid)
    finally:
        store.close()
    if removed:
        _render.print_json({"deleted": args.sid})
    else:
        _render.die(f"session {args.sid!r} not found", code=2)


def _resume_state(args: argparse.Namespace):
    """A DashboardState pinned to an existing session for a resume/drive turn.

    The session's stored project_root must match, so run from the project dir
    or pass ``--project-root``. Reuses the same provider/index resolution and
    streaming path as ``cgx agent``.
    """
    from cgx.cli.main import _state_from_args
    state = _state_from_args(args)
    state.agent_session_id = args.sid
    return state


def _send(args: argparse.Namespace) -> None:
    from cgx.cli.main import _run_cli_stream
    from cgx.cli.tui import ops
    state = _resume_state(args)
    # Guard so a typo'd sid errors rather than silently starting a new session
    # (agent_events treats a missing session as "start fresh").
    store = _open_store(state.project_root)
    try:
        exists = store.get_session(args.sid) is not None
    finally:
        store.close()
    if not exists:
        _render.die(f"session {args.sid!r} not found "
                    f"(run from the project dir or pass --project-root)", code=2)
    message = " ".join(args.message)
    _run_cli_stream(lambda ce: ops.agent_events(
        state, message, index_dir=args.index_dir, records=args.records,
        auto=args.auto, cancel_event=ce))


def _decide(args: argparse.Namespace) -> None:
    import json

    from cgx.cli.main import _run_cli_stream
    from cgx.cli.tui import ops
    try:
        chosen = json.loads(args.chosen)
    except json.JSONDecodeError as exc:
        _render.die(f"--chosen must be valid JSON: {exc}", code=3)
    if not isinstance(chosen, dict):
        _render.die("--chosen must be a JSON object (e.g. '{\"approved\": true}')", code=3)
    state = _resume_state(args)
    _run_cli_stream(lambda ce: ops.decide_events(
        state, args.task_id, chosen, rationale=args.rationale,
        index_dir=args.index_dir, records=args.records, auto=args.auto,
        cancel_event=ce))
