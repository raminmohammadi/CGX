

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
from typing import Iterable, Optional

from cgx.logging_setup import sanitize_for_log
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
        # Normalize first. ``os.path.realpath`` collapses ``..`` and follows
        # symlinks -- a pure canonicalization, unlike ``Path.resolve`` which
        # CodeQL models as a filesystem-access sink.
        base = os.path.realpath(index_dir)
        meta = os.path.realpath(os.path.join(base, "meta.json"))
        rec = os.path.realpath(records_path)

        # Confine both artifacts to a single canonical container before any
        # filesystem access: the ``project_root`` when known, otherwise the
        # index workspace root (the parent of ``index_dir`` -- ``records_path``
        # sits beside ``index_dir`` in the default layout). ``prefix`` ends in
        # a separator so a sibling like ``/proj-evil`` can't pass the check
        # for ``/proj``.
        container = (os.path.realpath(project_root) if project_root
                     else os.path.realpath(os.path.dirname(base)))
        prefix = container + os.sep

        # CodeQL path-injection barrier: a direct ``startswith`` prefix guard
        # on the exact canonical string handed to each filesystem sink, so the
        # tainted request input can never reach ``os.path.isfile`` /
        # ``os.path.getsize`` without first being confined to ``container``.
        if not meta.startswith(prefix):
            return False
        if not rec.startswith(prefix):
            return False

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
    # Canonicalize away ``..``/symlinks, then confirm the result is an
    # absolute path before any filesystem access. ``os.path.realpath`` is a
    # pure normalization (not a filesystem-access sink like ``Path.resolve``);
    # the ``startswith`` check is the CodeQL path-injection barrier on the
    # exact string handed to ``os.path.exists`` / ``os.scandir``. Pointing the
    # agent at an arbitrary *absolute* local directory is intentional, so the
    # guard rejects only non-canonical (relative/traversal) roots.
    root = os.path.realpath(project_root)
    if not root.startswith(os.sep):
        return True
    if not os.path.exists(root):
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
        logger.warning("mode: scandir(%s) failed: %s",
                       sanitize_for_log(root), type(exc).__name__)
        return True
    return count < threshold


# Imperative verbs that mark an objective as "build/change this", so it is a
# task even if phrased with a leading question word ("how about you build...").
_BUILD_VERBS = frozenset({
    "build", "create", "add", "implement", "generate", "make", "scaffold",
    "write", "develop", "refactor", "fix", "setup", "bootstrap", "port",
    "migrate", "convert", "wire", "rewrite", "extend", "delete", "remove",
    "update", "rename",
})

# Sentence starts that mark an objective as a read-only question ("explain how
# X works") rather than a build request.
_QUESTION_STARTS = (
    "how ", "how's", "what ", "what's", "why ", "where ", "where's", "when ",
    "which ", "who ", "does ", "do ", "is ", "are ", "can ", "could ",
    "should ", "would ", "explain", "describe", "summarize", "summarise",
    "tell me", "list ", "show me", "walk me", "where is", "what is",
)


def is_question(objective: Optional[str]) -> bool:
    """True when ``objective`` reads as a read-only question, not a build task.

    Deterministic (no LLM): a leading imperative build verb ("build ...",
    "add ...") is always a task; otherwise a trailing ``?`` or a question-word
    opener marks it as something to *answer*. Lets the single Swarm mode route
    "how does auth work?" to read-only investigation while "add rate limiting"
    still builds.
    """
    t = (objective or "").strip().lower()
    if not t:
        return False
    first = t.split(None, 1)[0].rstrip(":,")
    if first in _BUILD_VERBS:
        return False
    if t.endswith("?"):
        return True
    return t.startswith(_QUESTION_STARTS)


def detect_mode(*, project_root: Optional[str] = None,
                index_dir: Optional[str] = None,
                records_path: Optional[str] = None) -> SessionMode:
    """Pick a :class:`SessionMode` from the request inputs.

    Rules (first match wins):

    1. ``project_root`` is missing or empty (no non-ignored entries)
       -> ``SWARM`` (build a new project from the objective).
    2. No usable FAISS index visible at ``index_dir`` /
       ``records_path`` -> ``GREENFIELD``.
    3. Otherwise -> ``EXPLORE``.

    Phase C soft-hide: an empty ``project_root`` (nothing to overwrite) now
    defaults to the Swarm builder rather than the older GREENFIELD scaffold
    pipeline -- the Swarm is being unified into the single end-to-end build
    mode. A *non-empty but unindexed* project still routes to GREENFIELD for
    now: neither pipeline should overwrite an existing tree, and Swarm's
    surgical edit path lands in a later step (C3). GREENFIELD/EXPLORE remain
    fully reachable when a mode is passed explicitly.
    """
    if _project_is_empty(project_root):
        return SessionMode.SWARM
    if not _has_usable_index(index_dir, records_path, project_root):
        return SessionMode.GREENFIELD
    return SessionMode.EXPLORE
