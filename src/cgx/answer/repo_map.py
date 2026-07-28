"""Hierarchical repo map -- cheap whole-repo context for the Planner.

Built deterministically from the index records (:func:`cgx.embeddings.records.
make_index_records`), so it reuses the ``intent``-side summary material already
computed at index time (module/class/function docstrings, signatures, line
spans) without embedding or re-walking the AST. The map is a package -> file ->
symbol tree plus roll-up counts:

    build_repo_map(records) -> {version, schema_version, root, fingerprint,
                                stats, packages[], files[]}

It is content-addressed via ``fingerprint`` (a hash over each record's identity
+ summary fields) so :func:`build_or_load_repo_map` can skip a rebuild when the
persisted map still matches the current records, and rebuild when they drift.
:func:`render_repo_map` renders a budgeted text tree suitable for a prompt.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional

from cgx.embeddings.records import SCHEMA_VERSION
from cgx.logging_setup import get_logger

logger = get_logger("repo_map")

# Structure of the map/cache file (independent of the chunk SCHEMA_VERSION).
REPO_MAP_VERSION = 1


def _common_root(paths: List[str]) -> Optional[str]:
    """Best-effort common directory of ``paths`` for repo-relative display."""
    real = [p for p in paths if p]
    if not real:
        return None
    if len(real) == 1:
        return os.path.dirname(real[0]) or None
    try:
        return os.path.commonpath(real) or None
    except Exception:
        return None


def _rel(path: str, root: Optional[str]) -> str:
    if not root:
        return path
    try:
        return os.path.relpath(path, root)
    except Exception:
        return path


def fingerprint_records(records: List[Dict[str, Any]]) -> str:
    """Stable hash over the identity + summary fields that shape the map."""
    payload = []
    for r in records:
        payload.append((
            r.get("id"), r.get("type"), r.get("name"), r.get("file"),
            r.get("class_name"), r.get("module_path"), r.get("signature"),
            r.get("doc_first_sentence"),
            int(r.get("start_line") or 0), int(r.get("end_line") or 0),
        ))
    payload.sort(key=lambda t: (str(t[0]),))
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8", errors="replace")).hexdigest()


def build_repo_map(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build the hierarchical repo map from index records (pure, deterministic)."""
    root = _common_root([r.get("file") for r in records])
    # file path -> accumulator
    files: Dict[str, Dict[str, Any]] = {}

    def _acc(fp: str) -> Dict[str, Any]:
        return files.setdefault(fp, {
            "path": _rel(fp, root), "module_path": None, "summary": "",
            "n_loc": 0, "_classes": {}, "_functions": [],
        })

    for r in records:
        fp = r.get("file")
        if not fp:
            continue
        acc = _acc(fp)
        t = r.get("type")
        if t == "file":
            acc["summary"] = r.get("doc_first_sentence") or ""
            acc["module_path"] = r.get("module_path")
            acc["n_loc"] = int((r.get("metrics") or {}).get("n_loc") or 0)
        elif t == "class":
            acc["_classes"][r.get("name")] = {
                "kind": "class", "name": r.get("name"),
                "doc": r.get("doc_first_sentence") or "",
                "start_line": int(r.get("start_line") or 0),
                "end_line": int(r.get("end_line") or 0),
                "methods": 0,
            }
        elif t == "function":
            cls_name = r.get("class_name")
            if cls_name:
                cls = acc["_classes"].get(cls_name)
                if cls is not None:
                    cls["methods"] += 1
                    continue
            acc["_functions"].append({
                "kind": "function", "name": r.get("name"),
                "signature": r.get("signature") or "",
                "doc": r.get("doc_first_sentence") or "",
                "start_line": int(r.get("start_line") or 0),
                "end_line": int(r.get("end_line") or 0),
                "called_by_count": int(r.get("calls_in_count") or 0),
            })

    file_entries: List[Dict[str, Any]] = []
    for acc in files.values():
        symbols = list(acc.pop("_classes").values()) + acc.pop("_functions")
        symbols.sort(key=lambda s: (s.get("start_line") or 0, s.get("name") or ""))
        acc["symbols"] = symbols
        acc["n_symbols"] = len(symbols)
        file_entries.append(acc)
    file_entries.sort(key=lambda f: f.get("path") or "")

    # Package roll-up: group files by their (relative) directory.
    packages: Dict[str, Dict[str, Any]] = {}
    for fe in file_entries:
        pkg = os.path.dirname(fe["path"]) or "."
        pe = packages.setdefault(pkg, {"path": pkg, "files": [], "n_symbols": 0})
        pe["files"].append(fe["path"])
        pe["n_symbols"] += fe["n_symbols"]
    package_entries = sorted(packages.values(), key=lambda p: p["path"])

    return {
        "version": REPO_MAP_VERSION,
        "schema_version": SCHEMA_VERSION,
        "root": os.path.basename(root) if root else "",
        "fingerprint": fingerprint_records(records),
        "stats": {
            "n_files": len(file_entries),
            "n_symbols": sum(f["n_symbols"] for f in file_entries),
            "n_packages": len(package_entries),
        },
        "packages": package_entries,
        "files": file_entries,
    }


def save_repo_map(path: str, repo_map: Dict[str, Any]) -> None:
    """Persist ``repo_map`` to ``path`` atomically (best-effort)."""
    tmp = f"{path}.tmp"
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(repo_map, f)
        os.replace(tmp, path)
    except Exception as exc:
        logger.warning("failed to persist repo map %s: %s", path, exc)
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def load_repo_map(path: str) -> Optional[Dict[str, Any]]:
    """Load a persisted repo map, or ``None`` if missing/incompatible."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    except Exception as exc:
        logger.warning("repo map unreadable (%s); ignoring: %s", path, exc)
        return None
    if not isinstance(data, dict):
        return None
    if int(data.get("version", -1)) != REPO_MAP_VERSION:
        return None
    if int(data.get("schema_version", -1)) != SCHEMA_VERSION:
        return None
    return data


def build_or_load_repo_map(
    records: List[Dict[str, Any]], *, cache_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the repo map, reusing ``cache_path`` when its fingerprint matches.

    The map is derived deterministically from ``records``; the cache exists so
    the Planner can load it without re-deriving. A stored map is reused only
    when its ``fingerprint`` still equals the current records' fingerprint,
    otherwise it is rebuilt and re-persisted (cache invalidation).
    """
    current_fp = fingerprint_records(records)
    if cache_path:
        cached = load_repo_map(cache_path)
        if cached is not None and cached.get("fingerprint") == current_fp:
            logger.info("repo map cache hit (%s)", cache_path)
            return cached
    repo_map = build_repo_map(records)
    if cache_path:
        save_repo_map(cache_path, repo_map)
    return repo_map


def render_repo_map(
    repo_map: Dict[str, Any], *,
    max_chars: int = 4000,
    max_files: int = 200,
    max_symbols_per_file: int = 12,
) -> str:
    """Render a compact, budgeted text tree of the repo for prompt context.

    Groups files under their package and lists each file's top-level symbols
    (classes with method counts, functions with signatures). Truncates to
    ``max_chars`` so it stays a bounded, whole-repo overview for the Planner.
    """
    stats = repo_map.get("stats", {})
    lines: List[str] = [
        f"Repo map: {stats.get('n_files', 0)} files, "
        f"{stats.get('n_symbols', 0)} symbols, "
        f"{stats.get('n_packages', 0)} packages"
    ]
    files_by_path = {f["path"]: f for f in repo_map.get("files", [])}
    shown = 0
    for pkg in repo_map.get("packages", []):
        if shown >= max_files:
            break
        lines.append(f"\n{pkg['path']}/")
        for fp in pkg.get("files", []):
            if shown >= max_files:
                break
            fe = files_by_path.get(fp)
            if not fe:
                continue
            shown += 1
            summary = (fe.get("summary") or "").strip()
            head = f"  {os.path.basename(fp)}"
            if summary:
                head += f" -- {summary}"
            lines.append(head)
            for sym in fe.get("symbols", [])[:max_symbols_per_file]:
                if sym["kind"] == "class":
                    n = sym.get("methods", 0)
                    lines.append(f"    class {sym['name']} ({n} methods)")
                else:
                    sig = sym.get("signature") or "()"
                    lines.append(f"    def {sym['name']}{sig}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n... (truncated)"
    return text


__all__ = [
    "build_repo_map",
    "build_or_load_repo_map",
    "load_repo_map",
    "save_repo_map",
    "render_repo_map",
    "fingerprint_records",
    "REPO_MAP_VERSION",
]
