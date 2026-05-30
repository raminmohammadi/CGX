"""Write LLM-proposed diffs to the user's working tree.

This module is the only place in Averix that touches the real filesystem
on behalf of the agent. To keep the operation recoverable, every original
file is mirrored into a per-run backup directory before its contents are
overwritten. Callers receive the list of applied and failed files plus
the absolute backup path so the user can roll back if needed.

The actual diff parsing + in-memory hunk application is delegated to
:mod:`cgx.codegen.diff_apply`; this module sequences that with the
syntax validator (smoke test) and the disk I/O step.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

from cgx.codegen.diff_apply import (
    PatchTarget,
    apply_diffs_in_memory,
    parse_fenced_diffs,
)
from cgx.codegen.validate import validate_patch_results

logger = logging.getLogger(__name__)


def _dedupe_diffs(diffs: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    """Drop entries that repeat the same ``(file, patch)`` payload.

    Smaller planner models sometimes echo the same diff multiple times
    inside a single plan output; applying each copy in sequence writes
    duplicated imports / hunks to disk and triggers Judge rejections
    downstream. Order is preserved.
    """
    out: List[Dict[str, str]] = []
    seen: set = set()
    for d in diffs:
        if not isinstance(d, dict):
            continue
        fp = str(d.get("file") or d.get("path") or "").strip()
        patch = str(d.get("patch") or d.get("diff") or "")
        if not fp or not patch:
            continue
        key = (fp, patch)
        if key in seen:
            continue
        seen.add(key)
        out.append({"file": fp, "patch": patch})
    return out


def _to_targets(diffs: Sequence[Dict[str, str]]) -> List[PatchTarget]:
    targets: List[PatchTarget] = []
    seen: set = set()
    for d in diffs:
        if not isinstance(d, dict):
            continue
        fp = str(d.get("file") or d.get("path") or "").strip()
        patch = str(d.get("patch") or d.get("diff") or "")
        if not fp or not patch:
            continue
        # Tolerate callers that hand us full markdown blocks rather than
        # raw unified diffs by routing through the fenced-block parser.
        parsed = parse_fenced_diffs(patch)
        if parsed:
            for p in parsed:
                key = (p.path or fp, p.diff_text)
                if key in seen:
                    continue
                seen.add(key)
                targets.append(PatchTarget(path=p.path or fp, diff_text=p.diff_text))
        else:
            key = (fp, patch)
            if key in seen:
                continue
            seen.add(key)
            targets.append(PatchTarget(path=fp, diff_text=patch))
    return targets


def _build_file_tree(rel_paths: List[str]) -> str:
    """Return a markdown-style file tree from a list of relative paths."""
    if not rel_paths:
        return ""
    # Build a nested dict representing the directory tree.
    tree: Dict[str, Any] = {}
    for p in sorted(rel_paths):
        parts = Path(p).parts
        node = tree
        for part in parts:
            node = node.setdefault(part, {})

    lines: List[str] = []

    def _render(node: Dict[str, Any], prefix: str) -> None:
        items = sorted(node.keys(), key=lambda k: (not node[k], k.lower()))
        for i, name in enumerate(items):
            connector = "└── " if i == len(items) - 1 else "├── "
            child = node[name]
            if child:  # directory
                lines.append(f"{prefix}{connector}{name}/")
                extension = "    " if i == len(items) - 1 else "│   "
                _render(child, prefix + extension)
            else:
                lines.append(f"{prefix}{connector}{name}")

    _render(tree, "")
    return "\n".join(lines)


def apply_diffs_to_disk(
    project_root: str,
    diffs: Sequence[Dict[str, str]],
    *,
    allow_new_files: bool = True,
    backup_root: str = ".averix-backups",
) -> Dict[str, Any]:
    """Apply ``diffs`` to ``project_root`` after a syntax smoke test.

    Parameters
    ----------
    project_root
        Absolute path of the working tree to modify.
    diffs
        List of ``{"file": rel_path, "patch": unified_diff_text}`` entries
        (the shape emitted by the ``plan`` capability).
    allow_new_files
        When True, additive-only diffs for non-existent files create them.
    backup_root
        Directory (relative to ``project_root``) under which originals
        are mirrored before being overwritten.

    Returns
    -------
    dict with keys ``applied_files``, ``failed_files``, ``backup_dir``,
    ``diffs`` (the input diffs echoed back for UI rendering), and
    ``smoke_ok``.
    """
    root = Path(project_root).resolve()
    if root.exists() and not root.is_dir():
        raise ValueError(f"project_root is not a directory: {project_root}")
    root.mkdir(parents=True, exist_ok=True)

    deduped_diffs = _dedupe_diffs(diffs)
    targets = _to_targets(deduped_diffs)
    if not targets:
        return {"applied_files": [], "failed_files": [], "diffs": deduped_diffs,
                "backup_dir": None, "smoke_ok": False,
                "error": "no parseable diffs"}

    # Step 1: in-memory apply + syntax validation (smoke test).
    patches = apply_diffs_in_memory(str(root), targets, allow_new_files=allow_new_files)
    diagnostics = validate_patch_results(patches)
    diag_by_path = {d.path: d for d in diagnostics}

    failed_files: List[Dict[str, str]] = []
    for p in patches:
        if not p.ok:
            failed_files.append({"file": p.path, "error": p.error or "patch failed"})
            continue
        diag = diag_by_path.get(p.path)
        if diag is not None and not diag.ok:
            failed_files.append({
                "file": p.path,
                "error": f"{diag.language} syntax: {diag.error}",
            })

    if failed_files:
        logger.warning("apply_diffs_to_disk: smoke test failed for %d file(s); not writing",
                       len(failed_files))
        return {
            "applied_files": [], "failed_files": failed_files,
            "diffs": deduped_diffs, "backup_dir": None, "smoke_ok": False,
        }

    # Step 2: prepare a backup mirror.
    run_id = time.strftime("%Y%m%d-%H%M%S")
    backup_dir = root / backup_root / run_id
    backup_dir.mkdir(parents=True, exist_ok=True)

    applied: List[str] = []
    for p in patches:
        if not p.ok or p.new_content is None:
            continue
        rel = _normalize_rel(p.path, root)
        if rel is None:
            failed_files.append({"file": p.path,
                                 "error": "refusing to write outside project_root"})
            continue
        dest = root / rel
        if not _is_under(dest, root):
            failed_files.append({"file": p.path,
                                 "error": "refusing to write outside project_root"})
            continue
        if dest.exists():
            mirror = backup_dir / rel
            mirror.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dest, mirror)
        else:
            # Record an explicit ``.new`` marker so rollback can delete it.
            mirror = backup_dir / (str(rel) + ".new")
            mirror.parent.mkdir(parents=True, exist_ok=True)
            mirror.write_text("", encoding="utf-8")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(p.new_content, encoding="utf-8")
        applied.append(str(rel))
        logger.info("apply_diffs_to_disk: wrote %s (%d bytes)",
                    rel, len(p.new_content))

    return {
        "applied_files": applied, "failed_files": failed_files,
        "diffs": deduped_diffs, "backup_dir": str(backup_dir),
        "project_tree": _build_file_tree(applied),
        "smoke_ok": True,
    }


def _normalize_rel(path_str: str, root: Path) -> Path | None:
    """Normalize a target path into one relative to ``root``.

    The LLM sometimes emits absolute paths (e.g. ``/home/u/proj/src/x.py``)
    in fenced-diff headers. ``Path("/root") / "/abs"`` collapses to
    ``"/abs"`` in pathlib, which causes ``shutil.copy2(dest, mirror)`` to
    receive the same path twice (``SameFileError``). We coerce absolute
    paths inside ``root`` back to their project-relative form and reject
    anything that escapes the tree (``..``, absolute paths outside root).
    """
    s = (path_str or "").strip()
    # Strip a single leading ``./`` (or repeated ``./`` segments) — but
    # never ``lstrip("./")`` which is a character-set strip and would
    # eat the leading ``/`` of an absolute path, turning ``/home/u/x.py``
    # into ``home/u/x.py`` and causing the writer to mirror the absolute
    # path under root.
    while s.startswith("./"):
        s = s[2:]
    if not s:
        return None
    candidate = Path(s)
    if candidate.is_absolute():
        try:
            rel = candidate.resolve().relative_to(root)
        except (ValueError, OSError):
            return None
        return rel
    # Relative: resolve under root and confirm it still sits within root.
    try:
        resolved = (root / candidate).resolve()
        rel = resolved.relative_to(root)
    except (ValueError, OSError):
        return None
    return rel


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
        return True
    except Exception:
        return False



def rollback_from_backup(
    project_root: str,
    backup_dir: str,
) -> Dict[str, Any]:
    """Undo an earlier :func:`apply_diffs_to_disk` run.

    Walks every entry under ``backup_dir`` and either restores the
    mirrored original to its project-relative location or — for files
    that were created by the apply (mirrored as ``<rel>.new`` empty
    markers) — deletes the file the apply wrote.

    Parameters
    ----------
    project_root
        Working tree the original apply targeted.
    backup_dir
        Path produced by ``apply_diffs_to_disk`` (the value returned in
        the response's ``backup_dir`` field). Must sit inside
        ``project_root``; absolute paths outside the tree are rejected.

    Returns
    -------
    dict with keys ``restored_files`` (list[str], project-relative
    paths whose contents were rewritten from the mirror),
    ``deleted_files`` (list[str], paths that were removed because they
    were new), ``failed_files`` (list[{"file": str, "error": str}]),
    and ``error`` (top-level message when the backup directory itself
    is missing or out of bounds).
    """
    root = Path(project_root).resolve()
    if not root.is_dir():
        raise ValueError(f"project_root is not a directory: {project_root}")

    backup_path = Path(backup_dir)
    if not backup_path.is_absolute():
        backup_path = (root / backup_path)
    try:
        backup_path = backup_path.resolve()
    except OSError:
        return {"restored_files": [], "deleted_files": [], "failed_files": [],
                "error": f"backup_dir not resolvable: {backup_dir}"}
    if not _is_under(backup_path, root):
        return {"restored_files": [], "deleted_files": [], "failed_files": [],
                "error": "backup_dir is outside project_root"}
    if not backup_path.is_dir():
        return {"restored_files": [], "deleted_files": [], "failed_files": [],
                "error": f"backup_dir does not exist: {backup_dir}"}

    restored: List[str] = []
    deleted: List[str] = []
    failed: List[Dict[str, str]] = []

    for entry in sorted(backup_path.rglob("*")):
        if not entry.is_file():
            continue
        try:
            rel_in_backup = entry.relative_to(backup_path)
        except ValueError:
            continue

        rel_str = str(rel_in_backup)
        is_new_marker = rel_str.endswith(".new")
        rel_for_target = rel_str[:-4] if is_new_marker else rel_str

        target_rel = _normalize_rel(rel_for_target, root)
        if target_rel is None:
            failed.append({"file": rel_str,
                           "error": "refusing to touch path outside project_root"})
            continue
        target = root / target_rel
        if not _is_under(target, root):
            failed.append({"file": str(target_rel),
                           "error": "refusing to touch path outside project_root"})
            continue

        try:
            if is_new_marker:
                if target.exists():
                    target.unlink()
                deleted.append(str(target_rel))
                logger.info("rollback_from_backup: deleted %s", target_rel)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(entry, target)
                restored.append(str(target_rel))
                logger.info("rollback_from_backup: restored %s", target_rel)
        except OSError as e:
            failed.append({"file": str(target_rel), "error": str(e)})

    return {"restored_files": restored, "deleted_files": deleted,
            "failed_files": failed}
