"""Read-only introspection ("grounding") for the swarm Developer.

Before a file is written, the Developer must see the *real* public surface of
the files it depends on -- their imports, function/class signatures and
docstrings -- so it imports symbols that exist instead of inventing them (the
single largest source of broken imports in the first swarm cut). These helpers
wrap the deterministic summarizers in :mod:`cgx.parser.parse_codebase` and are
safe to expose as Developer tools: they only read, never write, and refuse
paths that escape the project root.
"""

from __future__ import annotations

import ast
import os
from typing import Any, Dict, List, Optional

from cgx.parser.parse_codebase import (
    _build_file_code_stub,
    _collect_top_level_members,
)

# Grounding blocks are injected into a prompt, so bound the size of any single
# dependency summary; a widely-imported module must not swamp the context.
_DEP_LIMIT = 2000


def _safe_read(path: str, root: str) -> Optional[str]:
    """Return the text of ``root/path`` or ``None`` (missing / unsafe / binary).

    Paths that are absolute or climb out of the root are refused -- the same
    guard the AST fallback applies -- so a grounding lookup can never read
    outside the project tree.
    """
    rel = str(path or "").strip()
    if not rel or os.path.isabs(rel) or ".." in rel.split(os.sep):
        return None
    full = os.path.normpath(os.path.join(root, rel))
    if not full.startswith(os.path.normpath(root)):
        return None
    try:
        with open(full, "r", encoding="utf-8") as fh:
            return fh.read()
    except (OSError, UnicodeDecodeError):
        return None


def _summarize(source: str) -> Optional[Dict[str, Any]]:
    """Parse Python source into ``{docstring, members}`` or ``None``."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    try:
        module_doc = ast.get_docstring(tree)
    except Exception:  # pragma: no cover - defensive
        module_doc = None
    return {"docstring": module_doc,
            "members": _collect_top_level_members(tree, source)}


def file_skeleton(path: str, root: str = ".") -> str:
    """A compact ``docstring + imports + signature stubs`` view of a file.

    Empty when the file is missing. When present but unparseable (a partial
    write), fall back to the first lines of raw source so the caller still has
    something concrete to ground against.
    """
    source = _safe_read(path, root)
    if source is None:
        return ""
    summary = _summarize(source)
    if summary is None:
        head = source.strip().splitlines()[:20]
        return "\n".join(head)
    return _build_file_code_stub(summary["docstring"], summary["members"])


def list_symbols(path: str, root: str = ".") -> List[Dict[str, str]]:
    """Top-level functions and classes of a file as ``{name, kind, signature}``."""
    source = _safe_read(path, root)
    summary = _summarize(source) if source is not None else None
    if summary is None:
        return []
    members = summary["members"]
    out: List[Dict[str, str]] = []
    for f in members.get("functions", []):
        out.append({"name": f["name"], "kind": "function",
                    "signature": f"def {f['name']}{f.get('signature', '()')}"})
    for c in members.get("classes", []):
        out.append({"name": c["name"], "kind": "class",
                    "signature": c.get("signature", f"class {c['name']}")})
    return out


def get_signature(path: str, name: str, root: str = ".") -> Optional[str]:
    """The signature line of one top-level symbol, or ``None`` if absent."""
    for sym in list_symbols(path, root):
        if sym["name"] == name:
            return sym["signature"]
    return None


def describe_file(path: str, root: str = ".") -> Dict[str, Any]:
    """A structured summary of a file for the Developer to reason over."""
    source = _safe_read(path, root)
    if source is None:
        return {"path": path, "exists": False, "docstring": None,
                "symbols": [], "imports": [], "skeleton": ""}
    summary = _summarize(source)
    members = summary["members"] if summary else {}
    return {
        "path": path,
        "exists": True,
        "parses": summary is not None,
        "docstring": summary["docstring"] if summary else None,
        "imports": list(members.get("imports", [])),
        "symbols": list_symbols(path, root),
        "skeleton": file_skeleton(path, root),
    }


def ground_dependencies(dep_paths: List[str], root: str = ".",
                        limit: int = _DEP_LIMIT) -> str:
    """A prompt-ready block of the public surface of a file's dependencies.

    One ``# From <dep>:`` section per readable dependency, each bounded so no
    single dependency can dominate the prompt. Missing dependencies (not yet
    written) are simply skipped -- the toposort guarantees a real dependency is
    written before its consumer, so a gap here means an ordering-optional edge.
    """
    parts: List[str] = []
    for dep in dep_paths:
        skel = file_skeleton(dep, root).strip()
        if not skel:
            continue
        if len(skel) > limit:
            skel = skel[:limit] + "\n# ... (truncated)"
        parts.append(f"# From {dep}:\n{skel}")
    return "\n\n".join(parts).strip()
