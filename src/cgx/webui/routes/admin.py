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
from cgx.trace import fallback_trace_log_path

logger = logging.getLogger(__name__)
router = APIRouter(tags=["admin"])


def _log_path(project_root: Optional[str]) -> Path:
    """Trace source: a project's ``.cgx/agent.log`` or the global fallback."""
    if project_root:
        base = Path.cwd().resolve()
        raw = Path(os.fspath(project_root))
        if raw.is_absolute():
            logger.warning("admin log path rejected absolute input: %s", project_root)
            return fallback_trace_log_path()
        rel = Path(os.path.normpath(os.fspath(project_root)))
        if rel.parts and rel.parts[0] == "..":
            logger.warning("admin log path rejected traversal input: %s", project_root)
            return fallback_trace_log_path()
        candidate = (base / rel).resolve()
        try:
            candidate.relative_to(base)
        except ValueError:
            safe_project_root = project_root.replace("\r", "\\r").replace("\n", "\\n")
            logger.warning("admin log path rejected outside base: %s", safe_project_root)
            return fallback_trace_log_path()
        return candidate / ".cgx" / "agent.log"
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
