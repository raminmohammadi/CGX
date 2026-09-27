"""``cgx govdata`` -- data-governance policy / PII scan / retention purge / erase.

Mirrors the web-UI data-governance route
(``src/cgx/webui/routes/govdata.py``, Subsystem M) exactly: the same engine
calls into :mod:`cgx.govdata`, the same env-resolved :class:`GovernanceConfig`
with a one-sweep ``retention_days`` override on purge, and the same
"exactly one of run_id/owner" guard on erase. Like the route, every action is
best-effort -- the retention driver skips a failing/missing store -- and any
handler-level failure surfaces as an ``error:`` line + nonzero exit rather
than a traceback. Output is JSON by default; ``--table`` renders the list-shaped
parts (scan findings, per-store deleted counts) as a compact table.
"""

from __future__ import annotations

import argparse

from cgx.cli import _render

# PII classes + counts from ``scan`` rendered under ``--table``.
_FINDING_COLUMNS = [
    ("TYPE", "type"),
    ("COUNT", "count"),
]

# ``{store: rows_deleted}`` from purge/erase rendered under ``--table``.
_DELETED_COLUMNS = [
    ("STORE", "store"),
    ("DELETED", "deleted"),
]


def register(sub) -> None:
    p = sub.add_parser(
        "govdata",
        help="Data governance: policy/scan/purge/erase (retention + PII).")
    verbs = p.add_subparsers(dest="govdata_cmd", required=True)

    p_policy = verbs.add_parser(
        "policy", help="Show the resolved data-governance policy.")
    p_policy.set_defaults(func=_policy)

    p_scan = verbs.add_parser(
        "scan", help="Scan text for PII (non-destructive) + scrubbed preview.")
    p_scan.add_argument("--file", "-f", default=None,
                        help="Text file to scan (default: read stdin).")
    p_scan.add_argument("--table", action="store_true",
                        help="Render PII findings as a table instead of JSON.")
    p_scan.set_defaults(func=_scan)

    p_purge = verbs.add_parser(
        "purge", help="Sweep observation rows older than the retention window.")
    p_purge.add_argument("--retention-days", type=int, default=None,
                         help="Override the ambient retention window for this sweep.")
    p_purge.add_argument("--table", action="store_true",
                         help="Render per-store deleted counts as a table.")
    p_purge.set_defaults(func=_purge)

    p_erase = verbs.add_parser(
        "erase", help="Right-to-erasure by run id or owner (exactly one).")
    p_erase.add_argument("--run-id", default=None,
                         help="Erase every row tied to this run id.")
    p_erase.add_argument("--owner", default=None,
                         help="Erase every row tied to this owner.")
    p_erase.add_argument("--table", action="store_true",
                         help="Render per-store deleted counts as a table.")
    p_erase.set_defaults(func=_erase)


def _policy(args: argparse.Namespace) -> None:
    # GET /api/govdata/policy -- resolve the env-driven GovernanceConfig.
    try:
        from cgx.govdata import GovernanceConfig
        p = GovernanceConfig.from_env()
        _render.print_json({
            "retention_days": p.retention_days,
            "store_full_text": p.store_full_text,
            "scrub_pii": p.scrub_pii,
            "preview_cap": p.preview_cap,
        })
    except Exception as e:  # defensive -- mirrors the route's 500 fallback
        _render.die(f"could not resolve policy: {e}")


def _scan(args: argparse.Namespace) -> None:
    # POST /api/govdata/scan -- audit ``text`` for PII + return a scrub preview.
    text = _render.read_text_input(args.file)
    try:
        from cgx.govdata import scan_pii, scrub_pii
        findings = scan_pii(text)
        result = {
            "findings": findings,
            "total": sum(f["count"] for f in findings),
            "scrubbed": scrub_pii(text),
        }
        if getattr(args, "table", False):
            _render.emit(result["findings"], table=True, columns=_FINDING_COLUMNS)
        else:
            _render.print_json(result)
    except Exception as e:  # defensive -- mirrors the route's 500 fallback
        _render.die(f"scan failed: {e}")


def _purge(args: argparse.Namespace) -> None:
    # POST /api/govdata/purge -- TTL sweep, with an optional one-off override.
    try:
        from cgx.govdata import GovernanceConfig, purge_expired
        policy = GovernanceConfig.from_env()
        if args.retention_days is not None:
            from dataclasses import replace
            policy = replace(policy, retention_days=max(0, int(args.retention_days)))
        deleted = purge_expired(policy)
    except Exception as e:  # defensive -- mirrors the route's 500 fallback
        _render.die(f"purge failed: {e}")
    _emit_deleted(args, deleted)


def _erase(args: argparse.Namespace) -> None:
    # POST /api/govdata/erase -- delete every row tied to one run_id or owner.
    run_id = (args.run_id or "").strip()
    owner = (args.owner or "").strip()
    if bool(run_id) == bool(owner):  # 422: exactly one selector required
        _render.die("provide exactly one of --run-id or --owner", code=3)
    try:
        if run_id:
            from cgx.govdata import erase_run
            deleted = erase_run(run_id)
        else:
            from cgx.govdata import erase_owner
            deleted = erase_owner(owner)
    except Exception as e:  # defensive -- mirrors the route's 500 fallback
        _render.die(f"erase failed: {e}")
    _emit_deleted(args, deleted)


def _emit_deleted(args: argparse.Namespace, deleted: dict) -> None:
    """Emit a ``{store: rows_deleted}`` result as JSON (default) or a table."""
    if getattr(args, "table", False):
        rows = [{"store": k, "deleted": v} for k, v in deleted.items()]
        _render.emit(rows, table=True, columns=_DELETED_COLUMNS)
    else:
        _render.print_json({"ok": True, "deleted": deleted,
                            "total": sum(deleted.values())})

