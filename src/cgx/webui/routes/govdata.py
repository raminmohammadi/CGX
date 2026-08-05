

"""Data-governance API (Subsystem M).

Operator-facing surface over the retention + PII layer:

* ``GET  /api/govdata/policy`` -- the resolved :class:`GovernanceConfig`
  (TTL days, full-vs-preview, PII toggle) so the admin page can show what is
  in force.
* ``POST /api/govdata/purge``  -- run a TTL retention sweep across every
  observation store; returns ``{store: rows_deleted}``.
* ``POST /api/govdata/erase``  -- right-to-erasure by ``run_id`` or ``owner``.
* ``POST /api/govdata/scan``   -- audit a text snippet for PII (non-destructive)
  and return a scrubbed preview.

Every action is best-effort and defensive: a failing store is skipped, and
handler errors surface as a 4xx/5xx JSON body rather than propagating.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from cgx.webui.models import GovEraseRequest, GovPurgeRequest, GovScanRequest

logger = logging.getLogger(__name__)
router = APIRouter(tags=["govdata"])


@router.get("/govdata/policy")
def get_policy() -> JSONResponse:
    """Return the currently-resolved data-governance policy."""
    try:
        from cgx.govdata import GovernanceConfig
        p = GovernanceConfig.from_env()
        return JSONResponse({
            "retention_days": p.retention_days,
            "store_full_text": p.store_full_text,
            "scrub_pii": p.scrub_pii,
            "preview_cap": p.preview_cap,
        })
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("get_policy failed: %s", e)
        return JSONResponse({"detail": "could not resolve policy"},
                            status_code=500)


@router.post("/govdata/purge")
def purge(req: GovPurgeRequest) -> JSONResponse:
    """Sweep rows older than the retention window across all stores."""
    try:
        from cgx.govdata import GovernanceConfig, purge_expired
        policy = GovernanceConfig.from_env()
        if req.retention_days is not None:
            from dataclasses import replace
            policy = replace(policy, retention_days=max(0, int(req.retention_days)))
        deleted = purge_expired(policy)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("purge failed: %s", e)
        return JSONResponse({"detail": "purge failed"}, status_code=500)
    return JSONResponse({"ok": True, "deleted": deleted,
                         "total": sum(deleted.values())})


@router.post("/govdata/erase")
def erase(req: GovEraseRequest) -> JSONResponse:
    """Delete every row tied to one ``run_id`` or one ``owner``."""
    run_id = (req.run_id or "").strip()
    owner = (req.owner or "").strip()
    if bool(run_id) == bool(owner):
        return JSONResponse(
            {"detail": "provide exactly one of run_id or owner"},
            status_code=422)
    try:
        if run_id:
            from cgx.govdata import erase_run
            deleted = erase_run(run_id)
        else:
            from cgx.govdata import erase_owner
            deleted = erase_owner(owner)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("erase failed: %s", e)
        return JSONResponse({"detail": "erase failed"}, status_code=500)
    return JSONResponse({"ok": True, "deleted": deleted,
                         "total": sum(deleted.values())})


@router.post("/govdata/scan")
def scan(req: GovScanRequest) -> JSONResponse:
    """Return PII counts + a scrubbed preview for ``text`` (non-destructive)."""
    try:
        from cgx.govdata import scan_pii, scrub_pii
        findings = scan_pii(req.text)
        return JSONResponse({
            "findings": findings,
            "total": sum(f["count"] for f in findings),
            "scrubbed": scrub_pii(req.text),
        })
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("scan failed: %s", e)
        return JSONResponse({"detail": "scan failed"}, status_code=500)
