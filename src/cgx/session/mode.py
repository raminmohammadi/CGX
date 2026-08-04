

"""Session mode auto-detection.

A session enters ``GREENFIELD`` mode when there is nothing for the
FAISS-backed loop to read: an empty or missing ``project_root``,
or a ``project_root`` that doesn't yet have a usable index built.
Everything else stays in the default ``EXPLORE`` mode.

This is deliberately a small, dependency-free helper so route layers,
the router, and tests can all call it without dragging the answer
engine or FAISS in.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Optional

from cgx.session.models import SessionMode

logger = logging.getLogger(__name__)


# Directory names ignored when counting "source-like" files. Mirrors the
# excludes used by the legacy parser/indexer so a fresh clone of an
# existing project (e.g. ``.git`` checked out, ``node_modules`` cached)
# is still classified as greenfield when there's no real source yet.
_IGNORE_DIRS = frozenset({
    ".git", ".hg", ".svn",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".venv", "venv", "env",
    "node_modules", ".idea", ".vscode",
    ".cgx", ".cgx-backups",
    "dist", "build", ".next", ".cache",
})


def _has_usable_index(index_dir: Optional[str],
                      records_path: Optional[str]) -> bool:
    """True iff ``index_dir`` + ``records_path`` look like a real index.

    A "usable" index has both the FAISS meta + the records file. We
    don't try to load FAISS here -- a meta.json + non-empty records is
    a strong-enough signal and stays import-free.
    """
    if not index_dir or not records_path:
        return False
    try:
        # ``index_dir`` and ``records_path`` are independent, caller-chosen
        # locations (siblings by default, e.g. ``/tmp/cgx_index/indices`` and
        # ``/tmp/cgx_index/records.jsonl``), so there is no shared root to
        # contain them under. Normalize each with ``os.path.realpath`` to
        # collapse ``..`` segments and follow symlinks before the stat calls.
        base = os.path.realpath(index_dir)
        meta = os.path.join(base, "meta.json")
        rec = os.path.realpath(records_path)
        return (os.path.isfile(meta) and os.path.isfile(rec)
                and os.path.getsize(rec) > 0)
    except (OSError, ValueError):
        return False


def _project_is_empty(project_root: Optional[str], *,
                      ignore: Iterable[str] = _IGNORE_DIRS,
                      threshold: int = 1) -> bool:
    """True iff ``project_root`` has fewer than ``threshold`` source files.

    Walks one level under ``project_root`` (cheap, deterministic) and
    counts entries that are not in ``ignore``. A missing directory
    counts as empty.
    """
    if not project_root:
        return True
    root = Path(project_root)
    if not root.exists():
        return True
    ignore_set = set(ignore)
    count = 0
    try:
        for entry in os.scandir(root):
            if entry.name in ignore_set or entry.name.startswith("."):
                continue
            count += 1
            if count >= threshold:
                return False
    except OSError as exc:
        logger.warning("mode: scandir(%s) failed: %s", root, exc)
        return True
    return count < threshold


def detect_mode(*, project_root: Optional[str] = None,
                index_dir: Optional[str] = None,
                records_path: Optional[str] = None) -> SessionMode:
    """Pick a :class:`SessionMode` from the request inputs.

    Rules (first match wins):

    1. ``project_root`` is missing or empty (no non-ignored entries)
       -> ``GREENFIELD``.
    2. No usable FAISS index visible at ``index_dir`` /
       ``records_path`` -> ``GREENFIELD``.
    3. Otherwise -> ``EXPLORE``.

    The greenfield path neither requires nor builds an index; it walks
    the working tree and generates files directly via the scaffold
    engine.
    """
    if _project_is_empty(project_root):
        return SessionMode.GREENFIELD
    if not _has_usable_index(index_dir, records_path):
        return SessionMode.GREENFIELD
    return SessionMode.EXPLORE
