"""``cgx rollback`` -- undo an APPLY run from its backup directory.

Mirrors ``POST /api/rollback`` (``cgx.codegen.disk_apply.rollback_from_backup``).
The agent's APPLY task records its per-run backup dir in
``output.backup_dir``; pass that here to restore the originals.

Unlike the web route -- which sandboxes writes to ``CGX_WORKSPACE_ROOT`` --
the CLI runs as the user against a project they name directly, so it does
not require that env var. It still guards that an *absolute* backup dir
lives inside the project root.
"""

from __future__ import annotations

import argparse
import os

from cgx.cli import _render


def register(sub) -> None:
    p = sub.add_parser("rollback", help="Restore files from an APPLY backup directory.")
    p.add_argument("--project-root", required=True, help="Project whose changes to undo.")
    p.add_argument("--backup-dir", required=True,
                   help="Backup dir recorded by the APPLY task (output.backup_dir).")
    p.set_defaults(func=_rollback)


def _rollback(args: argparse.Namespace) -> None:
    from cgx.codegen.disk_apply import rollback_from_backup
    project_root_real = os.path.realpath(args.project_root)
    if not os.path.isdir(project_root_real):
        _render.die(f"project_root not found: {args.project_root!r}", code=2)
    backup = args.backup_dir.strip()
    if os.path.isabs(backup):
        backup_real = os.path.realpath(backup)
        if not (backup_real == project_root_real
                or backup_real.startswith(project_root_real + os.sep)):
            _render.die("backup_dir is outside project_root", code=3)
        backup = backup_real
    try:
        result = rollback_from_backup(project_root_real, backup)
    except ValueError as exc:
        _render.die(str(exc), code=3)
    _render.print_json(result)
