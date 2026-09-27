"""``cgx health`` -- liveness + readiness probes.

Mirrors the root ``/healthz`` and ``/readyz`` endpoints (see
``webui/routes/health.py``), backed by :mod:`cgx.health`. Prints both
reports as JSON and exits non-zero when a critical subsystem is not ready,
so it composes in shell health checks.
"""

from __future__ import annotations

import argparse

from cgx.cli import _render


def register(sub) -> None:
    p = sub.add_parser("health", help="Print liveness + readiness; exit non-zero when not ready.")
    p.add_argument("--project-root", default=None,
                   help="Project whose session DB / index are checked (default: none).")
    p.set_defaults(func=_health)


def _health(args: argparse.Namespace) -> None:
    from cgx import health as _h
    live = _h.liveness()
    ready = _h.readiness(project_root=getattr(args, "project_root", None))
    _render.print_json({"liveness": live, "readiness": ready})
    if not ready.get("ready", False):
        raise SystemExit(1)
