

"""Write LLM-proposed diffs to the user's working tree.

This module is the only place in CGX that touches the real filesystem
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
from cgx.codegen.validate import check_cross_file_coherence, validate_patch_results
from cgx.trace import traced

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


@traced("codegen")
def apply_diffs_to_disk(
    project_root: str,
    diffs: Sequence[Dict[str, str]],
    *,
    allow_new_files: bool = True,
    backup_root: str = ".cgx-backups",
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
    root_real = os.path.realpath(project_root)
    if os.path.exists(root_real) and not os.path.isdir(root_real):
        raise ValueError(f"project_root is not a directory: {project_root}")
    os.makedirs(root_real, exist_ok=True)
    root = Path(root_real)

    deduped_diffs = _dedupe_diffs(diffs)
    targets = _to_targets(deduped_diffs)
    if not targets:
        return {"applied_files": [], "failed_files": [], "diffs": deduped_diffs,
                "backup_dir": None, "smoke_ok": False,
                "error": "no parseable diffs"}

    # Step 1: in-memory apply + syntax validation (smoke test) + cross-file coherence.
    patches = apply_diffs_in_memory(str(root), targets, allow_new_files=allow_new_files)
    diagnostics = validate_patch_results(patches)
    coherence_issues = check_cross_file_coherence(patches, project_root=str(root))
    # Merge coherence issues into the per-path diagnostic map; they take
    # precedence over a "ok" structural diagnostic for the same file.
    diag_by_path = {d.path: d for d in diagnostics}
    for ci in coherence_issues:
        diag_by_path[ci.path] = ci

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
        logger.warning(
            "apply_diffs_to_disk: smoke test failed for %d file(s); "
            "writing %d passing file(s) and reporting partial failure",
            len(failed_files),
            sum(1 for p in patches if p.ok and p.new_content is not None
                and p.path not in {f["file"] for f in failed_files}),
        )
        for f in failed_files:
            logger.warning("apply_diffs_to_disk: dropped %s: %s",
                           f["file"], f["error"])

    # Step 2: prepare a backup mirror (always, so passing files are safely backed up).
    run_id = time.strftime("%Y%m%d-%H%M%S")
    backup_dir = root / backup_root / run_id
    backup_dir.mkdir(parents=True, exist_ok=True)

    failed_paths = {f["file"] for f in failed_files}
    applied: List[str] = []
    for p in patches:
        if not p.ok or p.new_content is None:
            continue
        if p.path in failed_paths:
            continue  # skip files that failed smoke check
        dest = _normalize_rel(p.path, root_real)
        # SafeAccessCheck: confirm the normalized target stays under the
        # project root before any filesystem access (CodeQL path-injection).
        if dest is None or not dest.startswith(root_real + os.sep):
            failed_files.append({"file": p.path,
                                 "error": "refusing to write outside project_root"})
            continue
        rel = os.path.relpath(dest, root_real)
        if os.path.exists(dest):
            mirror = os.path.join(str(backup_dir), rel)
            os.makedirs(os.path.dirname(mirror), exist_ok=True)
            shutil.copy2(dest, mirror)
        else:
            # Record an explicit ``.new`` marker so rollback can delete it.
            mirror = os.path.join(str(backup_dir), rel + ".new")
            os.makedirs(os.path.dirname(mirror), exist_ok=True)
            with open(mirror, "w", encoding="utf-8") as fh:
                fh.write("")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w", encoding="utf-8") as fh:
            fh.write(p.new_content)
        applied.append(rel)
        logger.info("apply_diffs_to_disk: wrote %s (%d bytes)",
                    rel, len(p.new_content))

    return {
        "applied_files": applied, "failed_files": failed_files,
        "diffs": deduped_diffs, "backup_dir": str(backup_dir),
        "project_tree": _build_file_tree(applied),
        "smoke_ok": len(failed_files) == 0,
    }


def _normalize_rel(path_str: str, root_real: str) -> str | None:
    """Return an absolute, normalized path for ``path_str`` under ``root``.

    ``root_real`` must already be an ``os.path.realpath`` result. The LLM
    sometimes emits absolute paths (e.g. ``/home/u/proj/src/x.py``) in
    fenced-diff headers, so we coerce absolute paths inside ``root`` back
    to the project tree and reject anything that escapes it.

    Normalization uses ``os.path.normpath`` -- a pure string operation
    that never touches the filesystem, so it is safe on untrusted input,
    unlike ``Path.resolve`` which is itself a filesystem-access sink.
    Callers must still apply a ``startswith`` containment guard before
    touching the returned path.
    """
    s = (path_str or "").strip()
    # Strip leading ``./`` segments without ``lstrip("./")`` (a character
    # strip that would eat the leading ``/`` of an absolute path, turning
    # ``/home/u/x.py`` into ``home/u/x.py``).
    while s.startswith("./"):
        s = s[2:]
    if not s:
        return None
    if os.path.isabs(s):
        full = os.path.normpath(s)
    else:
        full = os.path.normpath(os.path.join(root_real, s))
    # Containment is enforced by callers via a ``startswith`` guard on the
    # normalized result; return it unconditionally here.
    return full


def rollback_from_backup(
    project_root: str,
    backup_dir: str,
) -> Dict[str, Any]:
    """Undo an earlier :func:`apply_diffs_to_disk` run.

    Walks every entry under ``backup_dir`` and either restores the
    mirrored original to its project-relative location or -- for files
    that were created by the apply (mirrored as ``<rel>.new`` empty
    markers) -- deletes the file the apply wrote.

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
    root_real = os.path.realpath(project_root)
    if not os.path.isdir(root_real):
        raise ValueError(f"project_root is not a directory: {project_root}")

    if os.path.isabs(backup_dir):
        backup_real = os.path.normpath(backup_dir)
    else:
        backup_real = os.path.normpath(os.path.join(root_real, backup_dir))
    # Follow symlinks, then confirm containment before any filesystem access.
    backup_real = os.path.realpath(backup_real)
    if not backup_real.startswith(root_real + os.sep):
        return {"restored_files": [], "deleted_files": [], "failed_files": [],
                "error": "backup_dir is outside project_root"}
    if not os.path.isdir(backup_real):
        return {"restored_files": [], "deleted_files": [], "failed_files": [],
                "error": f"backup_dir does not exist: {backup_dir}"}

    restored: List[str] = []
    deleted: List[str] = []
    failed: List[Dict[str, str]] = []

    for entry in sorted(Path(backup_real).rglob("*")):
        if not entry.is_file():
            continue
        rel_str = os.path.relpath(str(entry), backup_real)
        is_new_marker = rel_str.endswith(".new")
        rel_for_target = rel_str[:-4] if is_new_marker else rel_str

        target = _normalize_rel(rel_for_target, root_real)
        # SafeAccessCheck: never touch a path outside the project root.
        if target is None or not target.startswith(root_real + os.sep):
            failed.append({"file": rel_str,
                           "error": "refusing to touch path outside project_root"})
            continue
        target_rel = os.path.relpath(target, root_real)

        try:
            if is_new_marker:
                if os.path.exists(target):
                    os.remove(target)
                deleted.append(target_rel)
                logger.info("rollback_from_backup: deleted %s", target_rel)
            else:
                os.makedirs(os.path.dirname(target), exist_ok=True)
                shutil.copy2(str(entry), target)
                restored.append(target_rel)
                logger.info("rollback_from_backup: restored %s", target_rel)
        except OSError as e:
            # Log the underlying OS error (with type) server-side only; the
            # returned payload is surfaced in the UI, so expose a generic
            # message rather than a stack-trace-bearing exception string.
            logger.warning("rollback_from_backup: failed on %s: %s: %s",
                           target_rel, type(e).__name__, e)
            failed.append({"file": target_rel,
                           "error": "could not restore file"})

    return {"restored_files": restored, "deleted_files": deleted,
            "failed_files": failed}
