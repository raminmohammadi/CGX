"""Deterministic import invariants for the swarm generation/verify ladder.

Two correctness invariants a weak model routinely violates and that must never
reach disk unchallenged:

1. **No phantom imports** -- :func:`unused_imports` / :func:`strip_unused_imports`
   flag (and remove) module-level imports whose bound name is never referenced.
   A hallucinated ``from pydantic_settings import BaseSettings`` in a trivial
   ``is_prime`` module is the canonical case: harmless-looking, but it makes the
   module unimportable when the phantom dependency is absent, so the whole tree
   fails at collection time. ``__future__`` imports, star imports and package
   ``__init__`` re-exports are deliberately never touched.

2. **First-party imports must resolve** -- :func:`resolve_first_party_imports`
   resolves every first-party ``import``/``from ... import`` against the *same*
   roots pytest sees (the project root and, when present, ``root/src`` -- the
   layout :func:`cgx.codegen.test_runner._pytest_env` wires onto ``PYTHONPATH``).
   A first-party import that resolves against neither root names the importing
   file so the verify ladder can regenerate it, instead of relying on a
   basename-blind symbol scan that silently abstained on a misrooted module.
"""

from __future__ import annotations

import ast
from typing import Any, Dict, List, Optional, Set, Tuple

from cgx.session.scaffold_validate import _first_party_names, _module_name_for_path


def _is_init(path: str) -> bool:
    return (path or "").replace("\\", "/").endswith("__init__.py")


def _module_imports(tree: ast.Module) -> List[Tuple[ast.AST, List[Tuple[str, str]]]]:
    """Top-level import nodes as ``(node, [(bound_name, dotted_source), ...])``.

    ``__future__`` imports and ``from x import *`` are omitted: the former are
    mandatory and bind nothing meaningful, the latter binds names we cannot see.
    """
    out: List[Tuple[ast.AST, List[Tuple[str, str]]]] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            binds = [((a.asname or a.name.split(".")[0]), a.name)
                     for a in node.names]
            out.append((node, binds))
        elif isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                continue
            if any(a.name == "*" for a in node.names):
                continue
            binds = [((a.asname or a.name), a.name) for a in node.names]
            out.append((node, binds))
    return out


def _used_names(tree: ast.Module) -> Set[str]:
    """Names loaded anywhere in the module, plus any listed in ``__all__``."""
    used: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used.add(node.id)
    for node in tree.body:
        targets = (node.targets if isinstance(node, ast.Assign)
                   else [node.target] if isinstance(node, ast.AnnAssign) else [])
        if not any(isinstance(t, ast.Name) and t.id == "__all__"
                   for t in targets):
            continue
        val = getattr(node, "value", None)
        if isinstance(val, (ast.List, ast.Tuple, ast.Set)):
            for elt in val.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    used.add(elt.value)
    return used


def unused_imports(content: str, *, path: str = "") -> List[str]:
    """Return the module-level imported names that are never used.

    Best-effort and precise: a parse failure abstains (the syntax gate owns
    that), and ``__init__.py`` is skipped wholesale because its imports are
    conventionally re-exports.
    """
    if _is_init(path):
        return []
    try:
        tree = ast.parse(content or "")
    except SyntaxError:
        return []
    used = _used_names(tree)
    unused: List[str] = []
    for _node, binds in _module_imports(tree):
        for bound, _src in binds:
            if bound not in used and bound not in unused:
                unused.append(bound)
    return unused


def strip_unused_imports(content: str, *, path: str = "") -> Tuple[str, List[str]]:
    """Remove provably-unused module-level imports; return ``(text, removed)``.

    Keeps used aliases within a mixed import statement (``from m import a, b``
    where only ``b`` is unused becomes ``from m import a``) and drops the whole
    statement when every alias is unused. Non-import lines, formatting and
    comments elsewhere are preserved verbatim.
    """
    if _is_init(path):
        return content, []
    try:
        tree = ast.parse(content or "")
    except SyntaxError:
        return content, []
    used = _used_names(tree)
    lines = (content or "").splitlines()
    removed: List[str] = []
    edits: List[Tuple[int, int, Optional[str]]] = []
    for node, binds in _module_imports(tree):
        keep = [a for a in node.names
                if (a.asname or a.name.split(".")[0]
                    if isinstance(node, ast.Import) else a.asname or a.name) in used]
        if len(keep) == len(node.names):
            continue
        for bound, _src in binds:
            if bound not in used:
                removed.append(bound)
        end = getattr(node, "end_lineno", node.lineno)
        if keep:
            new = ast.Import(names=keep) if isinstance(node, ast.Import) else \
                ast.ImportFrom(module=node.module, names=keep, level=node.level)
            edits.append((node.lineno, end, ast.unparse(ast.fix_missing_locations(new))))
        else:
            edits.append((node.lineno, end, None))
    if not edits:
        return content, []
    for start, end, replacement in sorted(edits, key=lambda e: e[0], reverse=True):
        block = ([replacement] if replacement is not None else [])
        lines[start - 1:end] = block
    trailing = "\n" if (content or "").endswith("\n") else ""
    return "\n".join(lines) + trailing, removed


# --------------------- first-party import resolution ---------------------

def _importable_modules(paths: List[str]) -> Set[str]:
    """Dotted module names importable given the paths and pytest's rooting.

    Each planned ``.py`` contributes its dotted name (``src/foo.py`` ->
    ``src.foo``) and, because :func:`_pytest_env` also prepends ``root/src``,
    the ``src``-stripped variant (``foo``). Package prefixes are included so a
    ``from pkg import sub`` where ``pkg`` is only an implicit namespace still
    resolves against a generated ``pkg/sub.py``.
    """
    mods: Set[str] = set()
    for p in paths or []:
        m = _module_name_for_path(p)
        if not m:
            continue
        variants = [m]
        if m.startswith("src."):
            variants.append(m[len("src."):])
        for v in variants:
            parts = v.split(".")
            for i in range(1, len(parts) + 1):
                mods.add(".".join(parts[:i]))
    return mods


def _import_targets(tree: ast.Module) -> List[str]:
    """Absolute (non-relative) dotted import targets used by a module."""
    targets: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            targets.append(node.module)
    return targets


def resolve_first_party_imports(
        file_contents: Dict[str, str],
        paths: List[str], root: str = "") -> List[Dict[str, Any]]:
    """Flag first-party imports that resolve against neither root nor root/src.

    ``root`` is accepted for symmetry with the verify caller but resolution is
    purely path-based (the generated manifest is the source of truth), so the
    check is deterministic and hermetic. A parse failure abstains.
    """
    first_party = _first_party_names(list(paths))
    importable = _importable_modules(list(paths))
    warnings: List[Dict[str, Any]] = []
    for path, content in (file_contents or {}).items():
        if _module_name_for_path(path) is None or not isinstance(content, str):
            continue
        try:
            tree = ast.parse(content)
        except SyntaxError:
            continue
        for target in _import_targets(tree):
            top = target.split(".")[0]
            if top not in first_party:
                continue  # third-party / stdlib -> abstain
            if target in importable:
                continue
            warnings.append({
                "file": path,
                "module": target,
                "reason": (f"first-party import {target!r} resolves against "
                           "neither the project root nor root/src"),
            })
    return warnings
