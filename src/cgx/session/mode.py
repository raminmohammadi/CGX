

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
                      records_path: Optional[str],
                      project_root: Optional[str] = None) -> bool:
    """True iff ``index_dir`` + ``records_path`` look like a real index.

    A "usable" index has both the FAISS meta + the records file. We
    don't try to load FAISS here -- a meta.json + non-empty records is
    a strong-enough signal and stays import-free.
    """
    if not index_dir or not records_path:
        return False
    try:
        # Normalize paths first. ``os.path.realpath`` collapses ``..`` and
        # follows symlinks -- a pure canonicalization, unlike ``Path.resolve``
        # which CodeQL models as a filesystem-access sink.
        base = os.path.realpath(index_dir)
        meta = os.path.join(base, "meta.json")
        rec = os.path.realpath(records_path)

        # SafeAccessCheck (CodeQL path-injection): confirm both artifacts
        # stay within the index workspace root via the recognized
        # ``startswith(prefix + os.sep)`` prefix guard *before* any
        # filesystem access. ``records_path`` sits beside ``index_dir`` in
        # the default layout (``/tmp/cgx_index/indices`` +
        # ``/tmp/cgx_index/records.jsonl``), so the shared root is the parent
        # of ``index_dir``; ``meta.json`` lives directly under ``index_dir``.
        root = os.path.realpath(os.path.dirname(base))
        prefix = root + os.sep
        if not meta.startswith(prefix):
            return False
        if not rec.startswith(prefix):
            return False

        # If a project root is known, additionally require both artifacts to
        # live under it, again via the recognized prefix guard.
        if project_root:
            project_prefix = os.path.realpath(project_root) + os.sep
            if not meta.startswith(project_prefix):
                return False
            if not rec.startswith(project_prefix):
                return False

        # Rebuild the paths that reach a filesystem sink by joining the
        # trusted workspace ``root`` with the containment-checked relative
        # segment. The values flowing into ``os.path.isfile`` /
        # ``os.path.getsize`` are thus anchored on ``root`` rather than on
        # the raw request input -- CodeQL: uncontrolled data used in a path
        # expression.
        safe_meta = os.path.join(root, os.path.relpath(meta, root))
        safe_rec = os.path.join(root, os.path.relpath(rec, root))

        return (os.path.isfile(safe_meta) and os.path.isfile(safe_rec)
                and os.path.getsize(safe_rec) > 0)
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
    if not _has_usable_index(index_dir, records_path, project_root):
        return SessionMode.GREENFIELD
    return SessionMode.EXPLORE
