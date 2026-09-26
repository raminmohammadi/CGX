"""Repo-level instruction files -- CGX's answer to ``CLAUDE.md``.

A project can drop a plain-markdown context file at its root (or under
``.cgx/``) describing house style, domain rules, "always do X / never do
Y" conventions, ports, glossaries -- anything the chatbot and the agents
should honor on *every* turn without the user restating it. Unlike a
skill (which activates on triggers), this file is **always on** for the
project it lives in.

    CGX.md            (preferred, repo root -- commit it, share with the team)
    AGENT.md / AGENTS.md
    .cgx/CGX.md       (if you'd rather keep it out of the repo root)
    .cgx/AGENT.md

The first file found wins. Content is size-capped here at load time; the
per-surface injection layer trims further to the model's context budget so
a long instruction file never silently truncates retrieved citations on a
small local model.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "CONTEXT_FILE_CANDIDATES",
    "DEFAULT_CONTEXT_FILENAME",
    "DIRECTORY_NOTES_FILENAME",
    "find_context_file",
    "load_project_context",
    "load_directory_notes",
    "default_context_path",
    "clear_cache",
]

#: Per-directory "footguns" file (JEV conditional instructions): loaded only
#: when a task touches a file in (or under) that directory, so a rule that
#: matters for ``billing/`` never costs context while working elsewhere.
DIRECTORY_NOTES_FILENAME: str = "GOTCHAS.md"

#: Search order, relative to the project root. First match wins.
CONTEXT_FILE_CANDIDATES: Tuple[str, ...] = (
    "CGX.md",
    "AGENT.md",
    "AGENTS.md",
    os.path.join(".cgx", "CGX.md"),
    os.path.join(".cgx", "AGENT.md"),
)

#: Where the UI creates a new context file when none exists yet.
DEFAULT_CONTEXT_FILENAME: str = "CGX.md"

#: Load-time ceiling. Generous; the injection layer applies the real,
#: model-tier-scaled budget. Guards against a pathological multi-MB file.
_MAX_CONTEXT_CHARS: int = 16_000

# mtime cache: realpath -> (mtime, text). Keeps per-turn loads cheap without
# re-reading the file on every chat/agent call.
_cache: Dict[str, Tuple[float, str]] = {}


def clear_cache() -> None:
    """Drop the mtime cache (tests / after an in-app edit)."""
    _cache.clear()


def find_context_file(project_root: Optional[str]) -> Optional[Path]:
    """Return the first existing context file under ``project_root``, if any."""
    if not project_root:
        return None
    try:
        root = Path(project_root)
    except (TypeError, ValueError):
        return None
    for rel in CONTEXT_FILE_CANDIDATES:
        candidate = root / rel
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def default_context_path(project_root: str) -> Path:
    """Path where a new context file should be created for ``project_root``."""
    return Path(project_root) / DEFAULT_CONTEXT_FILENAME


def load_project_context(project_root: Optional[str],
                         max_chars: int = _MAX_CONTEXT_CHARS) -> str:
    """Return the project's context-file text (capped), or ``""``.

    Reads are mtime-cached and fail soft: an unreadable file yields ``""``
    rather than raising, so a bad context file never breaks answering.
    """
    path = find_context_file(project_root)
    if path is None:
        return ""
    key = os.path.realpath(str(path))
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return ""
    cached = _cache.get(key)
    if cached is not None and cached[0] == mtime:
        text = cached[1]
    else:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001
            logger.warning("context_files: failed to read %s: %s: %s",
                           path, type(e).__name__, e)
            return ""
        _cache[key] = (mtime, text)
    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n\n[... context file truncated ...]"
    return text


def load_directory_notes(
    project_root: Optional[str],
    files_touched,
    *,
    max_chars: int = 2000,
    per_file_chars: int = 800,
) -> str:
    """Return concatenated ``GOTCHAS.md`` notes for the dirs a task touches, or ``""``.

    JEV conditional instructions (Section VIII): instead of loading every rule
    up front, a directory's footguns file is pulled in only when the task
    touches a file in (or under) that directory. Notes are gathered
    nearest-directory-first, mtime-cached, and bounded. Intended to be appended
    at the TAIL of the instruction preamble so the always-on CGX.md + skills
    prefix stays byte-identical across tasks (prompt-cache friendly). Fail-soft.
    """
    if not project_root or not files_touched:
        return ""
    try:
        root = Path(project_root).resolve()
    except (TypeError, ValueError, OSError):
        return ""

    ordered_dirs: list = []
    seen_dirs: set = set()
    for f in files_touched:
        if not f:
            continue
        try:
            raw = str(f)
            p = (Path(raw) if os.path.isabs(raw) else root / raw).resolve()
        except (ValueError, OSError):
            continue
        cur = p.parent if p.suffix else p
        while True:
            try:
                cur.relative_to(root)
            except ValueError:
                break  # climbed out of the repo
            if cur not in seen_dirs:
                seen_dirs.add(cur)
                ordered_dirs.append(cur)
            if cur == root:
                break
            cur = cur.parent

    parts: list = []
    used = 0
    for d in ordered_dirs:
        gp = d / DIRECTORY_NOTES_FILENAME
        key = os.path.realpath(str(gp))
        try:
            if not gp.is_file():
                continue
            mtime = gp.stat().st_mtime
        except OSError:
            continue
        cached = _cache.get(key)
        if cached is not None and cached[0] == mtime:
            text = cached[1]
        else:
            try:
                text = gp.read_text(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001 - a bad notes file is skipped
                continue
            _cache[key] = (mtime, text)
        text = text.strip()
        if not text:
            continue
        if len(text) > per_file_chars:
            text = text[:per_file_chars].rstrip() + " [...]"
        try:
            rel = d.relative_to(root).as_posix() or "."
        except ValueError:  # pragma: no cover - guarded above
            rel = str(d)
        block = f"### {rel}/{DIRECTORY_NOTES_FILENAME}\n{text}"
        if parts and used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)
