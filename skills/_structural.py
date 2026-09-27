"""Shared structural checks for skill validators (and the Swarm verify gate).

Skill validators only ever see the diffs -- paths plus patch/content text -- so
these are best-effort textual checks over that bundle, not a full project
import-graph analysis. The heavy cross-file circular-import detection on the
Swarm path reuses the AST/SCC detector in
:func:`cgx.session.tasks.scaffold._circular_import_failures`; these helpers
cover the single-pattern, per-skill checks that every backend skill was
otherwise re-deriving by hand:

* :func:`app_backimports` -- a leaf module (route/model/db) importing an
  app-object (``app`` / ``db`` / ``engine`` ...) back from the application
  entrypoint, i.e. the classic Flask/FastAPI import cycle.
* :func:`undeclared_python_deps` -- a third-party import used in the code but
  absent from every ``requirements.txt`` / ``pyproject.toml`` in the diffs.
* :func:`body_of` / :func:`any_body_matches` -- small utilities so a validator
  can look at real content (e.g. confirm a genuine ``Flask(__name__)`` instance
  exists) rather than the mere presence of a substring anywhere in the tree.

All functions are pure and defensive: malformed diffs yield empty results, never
an exception, so a validator can call them without its own try/except.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def _body(d: Dict[str, Any]) -> str:
    """The textual body of one diff row, across the shapes CGX uses."""
    if not isinstance(d, dict):
        return ""
    return str(d.get("patch") or d.get("diff") or d.get("content")
               or d.get("new_content") or "")


def _path(d: Dict[str, Any]) -> str:
    return str(d.get("file") or d.get("path") or "").replace("\\", "/")


def body_of(diffs: List[Dict[str, Any]], needle: str) -> Optional[str]:
    """Return the body of the diff whose path equals or ends with ``needle``."""
    n = (needle or "").replace("\\", "/")
    for d in diffs or []:
        p = _path(d)
        if p == n or p.endswith("/" + n) or p.rsplit("/", 1)[-1] == n:
            return _body(d)
    return None


def any_body_matches(diffs: List[Dict[str, Any]], pattern: str,
                     *, only_ext: Optional[tuple] = None) -> bool:
    """True if any diff body matches ``pattern`` (optionally restricted by ext)."""
    rx = re.compile(pattern)
    for d in diffs or []:
        if only_ext and not _path(d).endswith(only_ext):
            continue
        if rx.search(_body(d)):
            return True
    return False


# Application-entrypoint module basenames: importing the app object *back* from
# one of these, from a non-entrypoint module, is the circular-import anti-pattern.
_ENTRY_STEMS = ("app", "main", "server", "wsgi", "asgi")
_ENTRY_BASENAMES = tuple(f"{s}.py" for s in _ENTRY_STEMS) + ("__init__.py",)
_APP_OBJECTS = ("app", "db", "engine", "celery", "cache", "socketio", "ma", "api")


def app_backimports(
    diffs: List[Dict[str, Any]],
    *,
    app_objects: tuple = _APP_OBJECTS,
) -> List[Dict[str, str]]:
    """Leaf ``.py`` modules importing an app-object back from the entrypoint.

    Returns ``[{file, imported, from_module}]`` -- high precision: fires only on
    ``from [pkg.]<entry> import ...<app_object>...`` inside a module that is not
    itself an entrypoint. This is the exact shape of the Flask case-study cycle
    (``backend/routes/members.py: from backend.app import app, db``).
    """
    imp_re = re.compile(
        r"^\s*from\s+([\w.]*\b(?:app|main|server|wsgi|asgi))\s+import\s+([^\n#]+)",
        re.M)
    out: List[Dict[str, str]] = []
    for d in diffs or []:
        path = _path(d)
        if not path.endswith(".py"):
            continue
        if path.rsplit("/", 1)[-1] in _ENTRY_BASENAMES:
            continue  # the entrypoint itself legitimately defines these
        body = _body(d)
        for m in imp_re.finditer(body):
            from_mod, names = m.group(1), m.group(2)
            hit = [s for s in app_objects if re.search(rf"\b{s}\b", names)]
            if hit:
                out.append({"file": path, "imported": ", ".join(hit),
                            "from_module": from_mod})
    return out


_PY_IMPORT_RE = re.compile(r"^\s*(?:import|from)\s+([\w.]+)", re.M)
_REQ_FILES = ("requirements.txt", "pyproject.toml", "setup.py", "setup.cfg")


def undeclared_python_deps(
    diffs: List[Dict[str, Any]],
    import_to_pip: Dict[str, str],
) -> List[Dict[str, str]]:
    """Third-party imports (per ``import_to_pip``) used but never declared.

    ``import_to_pip`` maps an import name (``flask_sqlalchemy``) to its pip name
    (``flask-sqlalchemy``). A match is "declared" if the pip name (or the import
    name) appears in any requirements/pyproject body in the diffs. Returns
    ``[{module, pip, file}]`` for the missing ones (deduped by pip name).
    """
    req = "\n".join(_body(d) for d in (diffs or [])
                    if _path(d).rsplit("/", 1)[-1] in _REQ_FILES).lower()
    missing: List[Dict[str, str]] = []
    seen: set = set()
    for d in diffs or []:
        path = _path(d)
        if not path.endswith(".py"):
            continue
        for m in _PY_IMPORT_RE.finditer(_body(d)):
            mod = m.group(1)
            for imp_name, pip in import_to_pip.items():
                if (mod == imp_name or mod.startswith(imp_name + ".")) \
                        and pip not in seen:
                    if pip.lower() not in req and imp_name.lower() not in req:
                        seen.add(pip)
                        missing.append({"module": imp_name, "pip": pip,
                                        "file": path})
    return missing


__all__ = [
    "body_of", "any_body_matches", "app_backimports", "undeclared_python_deps",
]
