"""Deterministic per-file scaffolding hints that shrink the model's job.

``generate_single_scaffold_file`` already tells the model WHICH symbols a file
must define (``_render_required_symbols_for_file``) and WHICH symbols its
dependencies export (the "AVAILABLE PROJECT MODULE SYMBOLS" block). The one gap
a weak model still trips on is assembling the exact *import statement*: whether
to write ``from store import X`` or ``from src.store import X``, and which
symbol names actually exist. This module closes that gap deterministically from
the real on-disk dependency source, so the model copies a correct line instead
of recalling a plausible-but-wrong one.

Scope is honest: this pre-writes imports for the Python dependency graph (the
only ecosystem whose module-name resolution + public surface the harness can
compute today). A non-Python file gets nothing here -- its cross-file breaks
are caught structurally by :mod:`cgx.session.cross_ref` instead. It is an
ADVISORY prompt hint, never a gate: a wrong or unused suggestion is stripped by
the developer ladder's phantom-import sanitiser, so it can only help.
"""

from __future__ import annotations

from typing import Dict

from cgx.session.scaffold_validate import _module_name_for_path, _top_level_symbols


def _flat_module(path: str) -> str:
    """The import name the harness's Python discipline prefers for ``path``.

    ``src/`` is a sys.path ROOT (stripped: ``src/store.py`` -> ``store``); a
    real package dir is kept (``backend/models.py`` -> ``backend.models``),
    matching the import rules in ``_SINGLE_FILE_SYSTEM``.
    """
    mod = _module_name_for_path(path) or ""
    if mod.startswith("src."):
        mod = mod[len("src."):]
    return mod


def render_import_hints(path: str, dep_sources: Dict[str, str]) -> str:
    """Ready-to-use import lines for a file's Python dependencies, or ``""``.

    ``dep_sources`` maps each ``depends_on`` path to its full on-disk source.
    For every Python dependency, emit ``from <flat module> import <public
    symbols>`` using the names the dependency actually DEFINES (not re-exports),
    so both the module path and the symbol names are grounded in reality. Skips
    a dependency whose source does not parse or exposes no public symbol
    (abstain rather than emit a guess). Returns ``""`` for a non-Python target
    or when nothing usable can be derived.
    """
    if not path.endswith(".py"):
        return ""
    lines = []
    for dep_path, src in (dep_sources or {}).items():
        if not dep_path.endswith(".py") or not isinstance(src, str) or not src:
            continue
        mod = _flat_module(dep_path)
        if not mod:
            continue
        defined = _top_level_symbols(src, include_imports=False)
        if not defined:
            continue
        public = sorted(s for s in defined if not s.startswith("_"))
        if not public:
            continue
        lines.append(f"from {mod} import {', '.join(public)}")
    if not lines:
        return ""
    return ("SUGGESTED IMPORTS (these are the correct, verified statements to "
            "import from THIS file's dependencies -- use them exactly; import "
            "only the names you actually use):\n" + "\n".join(lines))


__all__ = ["render_import_hints"]
