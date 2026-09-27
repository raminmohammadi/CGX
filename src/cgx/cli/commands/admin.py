"""``cgx admin`` -- operator read surface + trace-log housekeeping.

Mirrors the web-UI admin routes (``src/cgx/webui/routes/admin.py``):

* ``overview``   -> ``GET /admin/overview``  (activity + http + feedback +
  alerts rollup -- the audit-lite health view).
* ``logs``       -> ``GET /admin/logs``      (newest-first, server-side
  *redacted* slice of the JSONL trace log, with the durable session-stable
  mirror fallback).
* ``purge-logs`` -> ``DELETE /admin/logs``   (delete trace/log files only;
  requires an explicit scope -- never a bare purge).
* ``metrics``    -> ``GET /admin/metrics``   (structured in-process snapshot).

The route's read/delete pipeline is route-local (allow-list resolution
against the activity store, JSONL parse + :func:`cgx.redact.redact_mapping`
scrubbing, stable-mirror fallback, the HTTP-counter rollup); none of it is
wrapped by an engine function, so it is replicated here over the *same*
engine calls -- never through ``cgx.webui``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cgx.cli import _render

# Route parity: GET /admin/logs bounds ``limit`` at ``Query(200, ge=1, le=2000)``.
_LOGS_LIMIT_DEFAULT = 200
_LOGS_LIMIT_MAX = 2000

# Session ids are the ``<sid>:<ms>`` prefix of an activity ``run_id`` (minted
# by CGX, never argument input), but are still validated against this strict
# charset before a session-stable mirror path is built -- mirrors the route.
_SAFE_SESSION_ID = re.compile(r"^[A-Za-z0-9_.-]+$")

# Compact triage columns for ``logs --table`` (a subset of each trace record).
_LOG_COLUMNS = [
    ("TS", "ts"),
    ("EVENT", "event"),
    ("CATEGORY", "category"),
    ("FN", "fn"),
    ("SESSION", "session_id"),
    ("ELAPSED_MS", "elapsed_ms"),
]


def register(sub) -> None:
    p = sub.add_parser(
        "admin", help="Operator surface (overview/logs/purge-logs/metrics).")
    verbs = p.add_subparsers(dest="admin_cmd", required=True)

    verbs.add_parser(
        "overview",
        help="Audit-lite health rollup (activity + http + feedback + alerts).",
    ).set_defaults(func=_overview)

    p_logs = verbs.add_parser(
        "logs", help="Newest-first, redacted slice of the JSONL trace log.")
    p_logs.add_argument(
        "--project-root", default=None,
        help="Read this project's .cgx/agent.log (must be a known project "
             "root; anything else falls back to the global log).")
    p_logs.add_argument("--event", default=None,
                        help="Substring-match against the record 'event' field.")
    p_logs.add_argument("--since", type=float, default=None,
                        help="Only records with ts at/after this epoch time.")
    p_logs.add_argument(
        "--limit", type=int, default=_LOGS_LIMIT_DEFAULT,
        help=f"Max records to return (1-{_LOGS_LIMIT_MAX}, "
             f"default {_LOGS_LIMIT_DEFAULT}).")
    p_logs.add_argument("--table", action="store_true",
                        help="Human-readable table instead of JSON.")
    p_logs.set_defaults(func=_logs)

    p_purge = verbs.add_parser(
        "purge-logs",
        help="Delete trace-log files only (requires an explicit scope flag).")
    # Destructive, so an explicit scope is mandatory -- a bare ``purge-logs``
    # is refused (die code 3) so nothing is ever deleted implicitly. The three
    # selectors mirror the route's three delete scopes; kept mutually exclusive.
    scope = p_purge.add_mutually_exclusive_group()
    scope.add_argument("--project-root", default=None,
                       help="Purge just this project's agent.log "
                            "(must be a known project root).")
    scope.add_argument("--all", action="store_true",
                       help="Purge the global fallback log AND every known "
                            "project's agent.log.")
    scope.add_argument("--fallback", action="store_true",
                       help="Purge only the global fallback trace log.")
    p_purge.set_defaults(func=_purge_logs)

    verbs.add_parser(
        "metrics", help="Structured snapshot of every in-process metric.",
    ).set_defaults(func=_metrics)


# ----- overview --------------------------------------------------------------

def _overview(args: argparse.Namespace) -> None:
    """Fold activity (C), alerts (G) and feedback (H) into one health view.

    Each subsystem is best-effort exactly like the route: a missing/failed
    lookup degrades to an empty payload rather than failing the command.
    """
    activity: Dict[str, Any] = {}
    feedback: Dict[str, Any] = {}
    alerts_recent: List[Dict[str, Any]] = []
    try:
        from cgx.activity import get_default_run_store
        activity = get_default_run_store().summary()
    except Exception:
        pass
    try:
        from cgx.monitor import get_default_monitor
        alerts_recent = get_default_monitor().recent(limit=100)
    except Exception:
        pass
    try:
        from cgx.feedback import get_default_store
        feedback = get_default_store().stats()
    except Exception:
        pass
    by_sev: Dict[str, int] = {}
    for a in alerts_recent:
        sev = str(a.get("severity", "info"))
        by_sev[sev] = by_sev.get(sev, 0) + 1
    _render.print_json({
        "activity": activity,
        "http": _http_totals(),
        "feedback": feedback,
        "alerts": {"total": len(alerts_recent), "by_severity": by_sev,
                   "recent": alerts_recent[:10]},
    })


def _http_totals() -> Dict[str, float]:
    """Fold the ``cgx_http_requests_total`` counter into requests/5xx errors."""
    total = errors = 0.0
    try:
        from cgx import metrics as _m
        for c in _m.snapshot().get("counters", []):  # type: ignore[union-attr]
            if c.get("name") != "cgx_http_requests_total":
                continue
            v = float(c.get("value") or 0)
            total += v
            if str(c.get("labels", {}).get("status", "")).startswith("5"):
                errors += v
    except Exception:
        pass
    return {"requests": total, "errors": errors}


# ----- logs (read) -----------------------------------------------------------

def _logs(args: argparse.Namespace) -> None:
    """Newest-first, redacted trace slice; mirrors GET /admin/logs.

    Reads the project-local ``<root>/.cgx/agent.log`` first, then -- when that
    is empty and a project root was given (the common case after a greenfield
    tree is re-scaffolded and takes its local log with it) -- falls back to the
    durable session-stable mirror(s) so the trace still resolves.
    """
    if not 1 <= args.limit <= _LOGS_LIMIT_MAX:  # route Query(ge=1, le=2000) -> 422
        _render.die(f"--limit must be between 1 and {_LOGS_LIMIT_MAX}", code=3)
    path = _log_path(args.project_root)
    rows = _read_jsonl(path, limit=args.limit, event=args.event, since=args.since)
    source = str(path)
    if not rows and args.project_root:
        mirror_rows, mirror_source = _read_stable_mirrors(
            args.project_root, limit=args.limit, event=args.event, since=args.since)
        if mirror_rows:
            rows, source = mirror_rows, mirror_source
    if getattr(args, "table", False):
        _render.emit(rows, table=True, columns=_LOG_COLUMNS)
    else:
        _render.print_json({"source": source, "logs": rows, "count": len(rows)})


def _known_project_roots() -> List[str]:
    """Distinct project roots recorded in the activity store (trusted set).

    The trace reader and the ``--all`` purge both draw their roots from here,
    never from raw input, so a caller-supplied ``--project-root`` can only ever
    resolve to a path CGX itself produced.
    """
    roots: List[str] = []
    try:
        from cgx.activity import get_default_run_store
        seen = set()
        for run in get_default_run_store().recent(limit=500):
            root = run.get("project_root")
            if root and root not in seen:
                seen.add(root)
                roots.append(root)
    except Exception:
        pass
    return roots


def _log_path(project_root: Optional[str]) -> Path:
    """Trace source: a project's ``.cgx/agent.log`` or the global fallback.

    ``project_root`` must **exactly match** (after ``realpath``) a root the
    activity store already recorded; anything else (unknown path, traversal,
    symlink) falls back to the global log, and the returned path is built from
    the trusted stored root -- never the raw argument.
    """
    from cgx.trace import fallback_trace_log_path
    if not project_root:
        return fallback_trace_log_path()
    try:
        want = os.path.realpath(os.fspath(project_root))
    except OSError:
        return fallback_trace_log_path()
    for known in _known_project_roots():
        try:
            real_known = os.path.realpath(known)
        except OSError:
            continue
        if real_known == want:
            return Path(real_known) / ".cgx" / "agent.log"
    return fallback_trace_log_path()


def _session_ids_for_root(project_root: str) -> List[str]:
    """Session ids of ``kind='agent'`` runs that recorded ``project_root``.

    Drawn from the same trusted activity allow-list; a run's ``run_id`` is
    ``<session_id>:<ms>`` so the prefix is the session id, canonicalised
    against a strict charset before it locates a mirror path.
    """
    try:
        want = os.path.realpath(os.fspath(project_root))
    except OSError:
        return []
    out: List[str] = []
    seen = set()
    try:
        from cgx.activity import get_default_run_store
        for run in get_default_run_store().recent(limit=1000, kind="agent"):
            root = run.get("project_root")
            if not root:
                continue
            try:
                if os.path.realpath(root) != want:
                    continue
            except OSError:
                continue
            sid = str(run.get("run_id") or "").split(":", 1)[0]
            if sid and sid not in seen and _SAFE_SESSION_ID.match(sid):
                seen.add(sid)
                out.append(sid)
    except Exception:
        pass
    return out


def _row_ts(rec: Dict[str, Any]) -> float:
    try:
        return float(rec.get("ts", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _read_stable_mirrors(project_root: str, *, limit: int, event: Optional[str],
                         since: Optional[float]) -> Tuple[List[Dict[str, Any]], str]:
    """Read the durable session-stable trace mirror(s) for ``project_root``.

    Consulted only when the project-local ``agent.log`` is gone. Session ids
    come from the trusted activity allow-list, so no raw argument reaches the
    filesystem read. Returns ``(rows_newest_first, source_label)``.
    """
    from cgx.session.agent_log import stable_trace_log_path
    merged: List[Dict[str, Any]] = []
    used: List[Path] = []
    for sid in _session_ids_for_root(project_root):
        chunk = _read_jsonl(stable_trace_log_path(sid), limit=limit,
                            event=event, since=since)
        if chunk:
            merged.extend(chunk)
            used.append(stable_trace_log_path(sid))
    if not merged:
        return [], ""
    merged.sort(key=_row_ts, reverse=True)
    source = str(used[0]) if len(used) == 1 else f"{len(used)} session mirrors"
    return merged[:limit], source


def _read_jsonl(path: Path, *, limit: int, event: Optional[str],
                since: Optional[float]) -> List[Dict[str, Any]]:
    """Parse a JSONL trace file newest-first, redacted and filtered.

    Every record is passed through :func:`cgx.redact.redact_mapping` before it
    leaves the process -- the same server-side scrub the route applies -- so a
    secret that slipped into a prompt/response preview can never surface here.
    """
    from cgx.redact import redact_mapping
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:  # missing file is the normal case; read errors are defensive
        return []
    out: List[Dict[str, Any]] = []
    for raw in reversed(lines):
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except Exception:
            continue
        if not isinstance(rec, dict):
            continue
        if event and event not in str(rec.get("event", "")):
            continue
        if since is not None:
            try:
                if float(rec.get("ts", 0)) < since:
                    continue
            except (TypeError, ValueError):
                pass
        out.append(redact_mapping(rec))
        if len(out) >= limit:
            break
    return out


# ----- purge-logs (delete) ---------------------------------------------------

def _purge_logs(args: argparse.Namespace) -> None:
    """Delete trace/log files only -- mirrors DELETE /admin/logs scope handling.

    Destructive, so an explicit scope is mandatory (``--all`` /
    ``--project-root`` / ``--fallback``); a bare ``purge-logs`` is refused. A
    ``--project-root`` must match a root the activity store recorded, else the
    purge is a not-found. Deletion is delegated to helpers that only ever
    unlink files literally named ``agent.log`` / ``cgx-trace.log`` (+ rotation
    backups), refuse symlinks, and require a regular file.
    """
    from cgx.session.agent_log import delete_project_trace_log
    from cgx.trace import delete_fallback_trace_log
    removed = 0
    if args.all:
        targets: List[str] = ["fallback"]
        removed += delete_fallback_trace_log()
        for root in _known_project_roots():
            n = delete_project_trace_log(root)
            if n:
                removed += n
                targets.append(root)
        _render.print_json({"deleted": removed, "scope": "all", "targets": targets})
    elif args.project_root:
        want = os.path.realpath(os.fspath(args.project_root))
        # Resolve to the matching trusted root and delete via *that* value,
        # never the raw argument, so nothing caller-supplied reaches the unlink.
        match = next(
            (r for r in _known_project_roots() if os.path.realpath(r) == want), None)
        if match is None:
            _render.die(
                f"unknown project root {args.project_root!r} "
                "(not recorded in the activity store); nothing purged", code=2)
        removed += delete_project_trace_log(match)
        _render.print_json(
            {"deleted": removed, "scope": "single", "targets": [match]})
    elif args.fallback:
        removed += delete_fallback_trace_log()
        _render.print_json(
            {"deleted": removed, "scope": "fallback", "targets": ["fallback"]})
    else:
        _render.die(
            "purge-logs requires an explicit scope: "
            "--project-root PATH, --all, or --fallback", code=3)


# ----- metrics ---------------------------------------------------------------

def _metrics(args: argparse.Namespace) -> None:
    """Structured (non-Prometheus) snapshot of every in-process metric."""
    from cgx import metrics as _m
    _render.print_json(_m.snapshot())

