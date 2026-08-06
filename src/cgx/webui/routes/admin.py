"""Admin logs / trace explorer read API (Subsystem D).

Read-only operator surface that stitches the other observability stores
into one admin view: the JSONL function-call trace log (trace explorer),
a structured metrics snapshot, and an audit-lite health overview that
folds activity (C), alerts (G) and feedback (H) into one payload.

Every log line is passed through :func:`cgx.redact.redact_mapping` before
it leaves the process, so secrets that slipped into a prompt/response
preview can never reach the admin UI even if they reached disk.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from cgx import metrics as _metrics
from cgx.redact import redact_mapping
from cgx.session.agent_log import delete_project_trace_log
from cgx.trace import delete_fallback_trace_log, fallback_trace_log_path

logger = logging.getLogger(__name__)
router = APIRouter(tags=["admin"])


def _known_project_roots() -> List[str]:
    """Distinct project roots seen in the activity store (trusted internal).

    The request never supplies the root set, so an attacker cannot inject an
    arbitrary path -- the trace reader and the ``scope=all`` delete sweep
    both draw their project roots from here, not from raw request input.
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
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("admin: project-root enumeration failed: %s", e)
    return roots


def _log_path(project_root: Optional[str]) -> Path:
    """Trace source: a project's ``.cgx/agent.log`` or the global fallback.

    ``project_root`` is caller-controlled, so it is never used to build a
    filesystem path directly. Instead it must **exactly match** (after
    :func:`os.path.realpath` canonicalisation) one of the project roots the
    activity store has already recorded -- a closed allow-list of paths CGX
    itself produced. Any value not on that list (including traversal or
    symlink tricks) falls back to the global log, so untrusted input can
    never steer the reader at an arbitrary file.
    """
    if not project_root:
        return fallback_trace_log_path()
    safe = project_root.replace("\r", "\\r").replace("\n", "\\n")
    try:
        want = os.path.realpath(os.fspath(project_root))
    except OSError:  # pragma: no cover - defensive
        logger.warning("admin log path rejected (bad path): %s", safe)
        return fallback_trace_log_path()
    for known in _known_project_roots():
        try:
            if os.path.realpath(known) == want:
                # Trailing components are compile-time constants.
                return Path(want) / ".cgx" / "agent.log"
        except OSError:  # pragma: no cover - defensive
            continue
    logger.warning("admin log path rejected (unknown project root): %s", safe)
    return fallback_trace_log_path()


def _read_jsonl(path: Path, *, limit: int, event: Optional[str],
                since: Optional[float]) -> List[Dict[str, Any]]:
    """Parse a JSONL trace file, newest first, redacted and filtered."""
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except FileNotFoundError:
        return []
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("admin log read failed: %s", e)
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


@router.get("/admin/logs")
def admin_logs(
    limit: int = Query(200, ge=1, le=2000),
    event: Optional[str] = Query(None),
    since: Optional[float] = Query(None),
    project_root: Optional[str] = Query(None),
) -> JSONResponse:
    """Newest-first, server-side-redacted slice of the JSONL trace log."""
    path = _log_path(project_root)
    rows = _read_jsonl(path, limit=limit, event=event, since=since)
    return JSONResponse({"source": str(path), "logs": rows, "count": len(rows)})


@router.delete("/admin/logs")
def delete_logs(
    project_root: Optional[str] = Query(None),
    scope: Optional[str] = Query(None),
) -> JSONResponse:
    """Delete trace/log files only -- never any other file on the machine.

    Three modes, in precedence order:

    * ``scope=all`` -- purge the global fallback log plus every project
      ``agent.log`` known to the activity store.
    * ``project_root=<path>`` -- purge just that project's ``agent.log``.
    * neither -- purge just the global fallback log.

    A caller-supplied ``project_root`` must **exactly match** (after
    canonicalisation) a project root the activity store already recorded --
    the same closed allow-list the trace reader uses -- so untrusted input
    cannot steer the delete at an arbitrary path. On top of that, deletion
    is delegated to helpers that only ever unlink files literally named
    ``cgx-trace.log`` / ``agent.log`` (plus rotation backups), refuse
    symlinks, and require a regular file.
    """
    removed = 0
    targets: List[str] = []
    if scope == "all":
        removed += delete_fallback_trace_log()
        targets.append("fallback")
        for root in _known_project_roots():
            n = delete_project_trace_log(root)
            if n:
                removed += n
                targets.append(root)
    elif project_root:
        want = os.path.realpath(os.fspath(project_root))
        known = any(os.path.realpath(r) == want for r in _known_project_roots())
        if not known:
            safe = project_root.replace("\r", "\\r").replace("\n", "\\n")
            logger.warning("admin delete rejected (unknown project root): %s", safe)
            return JSONResponse({"deleted": 0, "scope": "single",
                                 "targets": [], "rejected": True})
        removed += delete_project_trace_log(project_root)
        targets.append(project_root)
    else:
        removed += delete_fallback_trace_log()
        targets.append("fallback")
    return JSONResponse({"deleted": removed, "scope": scope or "single",
                         "targets": targets})


@router.get("/admin/metrics")
def admin_metrics() -> JSONResponse:
    """Structured (non-Prometheus) snapshot of every in-process metric."""
    return JSONResponse(_metrics.snapshot())


def _http_totals() -> Dict[str, float]:
    total = errors = 0.0
    try:
        for c in _metrics.snapshot().get("counters", []):  # type: ignore[union-attr]
            if c.get("name") != "cgx_http_requests_total":
                continue
            v = float(c.get("value") or 0)
            total += v
            if str(c.get("labels", {}).get("status", "")).startswith("5"):
                errors += v
    except Exception:  # pragma: no cover - defensive
        pass
    return {"requests": total, "errors": errors}


@router.get("/admin/overview")
def admin_overview() -> JSONResponse:
    """Audit-lite: fold activity, alerts and feedback into one health view."""
    activity: Dict[str, Any] = {}
    feedback: Dict[str, Any] = {}
    alerts_recent: List[Dict[str, Any]] = []
    try:
        from cgx.activity import get_default_run_store
        activity = get_default_run_store().summary()
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("admin overview: activity summary failed: %s", e)
    try:
        from cgx.monitor import get_default_monitor
        alerts_recent = get_default_monitor().recent(limit=100)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("admin overview: alerts failed: %s", e)
    try:
        from cgx.feedback import get_default_store
        feedback = get_default_store().stats()
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("admin overview: feedback stats failed: %s", e)
    by_sev: Dict[str, int] = {}
    for a in alerts_recent:
        sev = str(a.get("severity", "info"))
        by_sev[sev] = by_sev.get(sev, 0) + 1
    return JSONResponse({
        "activity": activity,
        "http": _http_totals(),
        "feedback": feedback,
        "alerts": {"total": len(alerts_recent), "by_severity": by_sev,
                   "recent": alerts_recent[:10]},
    })
