"""``cgx metrics`` -- dump the in-process metrics registry.

Prometheus text exposition by default (mirrors ``GET /api/metrics``); pass
``--json`` for the structured snapshot the admin page uses (``GET
/api/admin/metrics``). Backed by :mod:`cgx.metrics`.
"""

from __future__ import annotations

import argparse

from cgx.cli import _render


def register(sub) -> None:
    p = sub.add_parser("metrics", help="Dump the metrics registry (Prometheus text, or --json).")
    p.add_argument("--json", action="store_true",
                   help="Emit the JSON snapshot instead of Prometheus text.")
    p.set_defaults(func=_dump)


def _dump(args: argparse.Namespace) -> None:
    from cgx import metrics as _m
    if getattr(args, "json", False):
        _render.print_json(_m.snapshot())
    else:
        print(_m.render_prometheus(), end="")
