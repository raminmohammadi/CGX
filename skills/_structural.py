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


# ------------------------- ORM model / table coherence -------------------------
#
# The generic first-party import checker
# (:func:`cgx.session.scaffold_validate.cross_check_first_party_imports`) catches
# ``from x.models import Order`` when ``Order`` is undefined -- but it is blind to
# the two *other* ways generated code references a model that was never created:
#
#   * an ORM-position use of a class name (``Order.query``, ``db.session.get(Order,
#     ...)``) with no ``class Order(db.Model)`` anywhere, and
#   * a *raw-SQL* table (``INSERT INTO orders (...)`` / ``... FROM orders``) that no
#     model maps to, so ``db.create_all()`` never creates it -> runtime
#     ``OperationalError: no such table``.
#
# Both are exactly the coffee-shop failure: routes/tests reference an ``Order`` /
# ``orders`` that ``models.py`` never defines, and the old repair loop *stripped*
# the reference (symptom) instead of *adding the model* (cause). These helpers let
# a skill detect the gap AND mine the columns straight from the SQL, so verify can
# drive a targeted, additive regen of the defining module instead of a blind pass.

# A ``class X(db.Model)`` / ``class X(Base)`` / ``class X(Model)`` declaration.
_MODEL_CLASS_RE = re.compile(
    r"^\s*class\s+(\w+)\s*\(\s*[^)]*\b(?:db\.Model|Model|Base)\b", re.M)
# An explicit ``__tablename__ = "orders"`` inside a model body.
_TABLENAME_RE = re.compile(r"__tablename__\s*=\s*['\"](\w+)['\"]")
# ORM-position uses of a *capitalised* name: ``Order.query``, ``db.session.get(
# Order, ...)``, ``db.session.query(Order)``, ``session.query(Order)``.
_ORM_USE_RE = re.compile(
    r"\b([A-Z]\w+)\.query\b"
    r"|\bsession\.get\(\s*([A-Z]\w+)"
    r"|\bsession\.query\(\s*([A-Z]\w+)")
# Raw-SQL table references, scanned ONLY inside string literals that actually
# contain a SQL DML verb -- so a Python ``from x import y`` (which reads like a
# SQL ``FROM x`` to a naive regex) can never be mistaken for a table.
# ``INSERT INTO t (cols...)`` additionally yields the column list.
_STRING_LIT_RE = re.compile(r"['\"]([^'\"\n]*)['\"]")
_SQL_DML_RE = re.compile(r"\b(?:INSERT|SELECT|UPDATE|DELETE)\b", re.I)
_SQL_INSERT_RE = re.compile(
    r"INSERT\s+INTO\s+([A-Za-z_]\w*)\s*\(([^)]*)\)", re.I)
_SQL_TABLE_RE = re.compile(
    r"(?:FROM|JOIN|UPDATE|INTO|DELETE\s+FROM)\s+([A-Za-z_]\w*)", re.I)


def _norm_table(name: str) -> str:
    """Fold a class/table name to a comparison key: lowercased, de-pluralised."""
    n = (name or "").lower()
    if n.endswith("ies"):
        return n[:-3] + "y"
    return n[:-1] if n.endswith("s") else n


def defined_orm_models(diffs: List[Dict[str, Any]]) -> Dict[str, str]:
    """Map every declared ORM model to the diff path that defines it.

    A "model" is a class deriving from ``db.Model`` / ``Model`` / ``Base``. The
    return also folds in each model's ``__tablename__`` (when declared) and the
    Flask-SQLAlchemy default table name, so a raw-SQL table reference can be
    matched against a model even when the names differ (``class Order`` ->
    table ``order``/``orders``).
    """
    defined: Dict[str, str] = {}
    for d in diffs or []:
        path = _path(d)
        if not path.endswith(".py"):
            continue
        body = _body(d)
        for m in _MODEL_CLASS_RE.finditer(body):
            cls = m.group(1)
            defined[cls] = path
            defined[_norm_table(cls)] = path
        for m in _TABLENAME_RE.finditer(body):
            defined[_norm_table(m.group(1))] = path
    return defined


def _models_target(diffs: List[Dict[str, Any]],
                   defined: Dict[str, str]) -> str:
    """Best path for the module that SHOULD define models (for a regen target).

    An existing ``models.py`` (or the file already defining models) wins;
    otherwise fall back to a conventional ``models.py`` beside the app package.
    """
    for d in diffs or []:
        p = _path(d)
        if p.rsplit("/", 1)[-1] == "models.py":
            return p
    if defined:
        # Reuse whatever file already holds models.
        return sorted(set(defined.values()))[0]
    # Conventional default under the first python package seen.
    for d in diffs or []:
        p = _path(d)
        if "/" in p and p.endswith(".py"):
            return p.rsplit("/", 1)[0] + "/models.py"
    return "models.py"


def undefined_model_refs(
        diffs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Models/tables referenced by the code but defined by no ORM model.

    Returns ``[{name, table, columns, referenced_in, define_in, via}]`` -- one
    entry per missing model, deduped by folded name. ``columns`` is mined from
    ``INSERT INTO t (c1, c2, ...)`` when available (the precise spec for an
    additive regen of the defining module); empty otherwise. High precision: a
    reference that matches ANY defined model (by class name or table name, incl.
    singular/plural) is never flagged.
    """
    defined = defined_orm_models(diffs)
    known = set(defined) | {_norm_table(k) for k in defined}
    target = _models_target(diffs, defined)
    out: Dict[str, Dict[str, Any]] = {}

    def _note(display: str, key: str, ref_path: str, via: str,
              table: str = "", columns: Optional[List[str]] = None) -> None:
        if key in known:
            return
        cur = out.get(key)
        if cur is None:
            out[key] = {"name": display, "table": table, "columns": columns or [],
                        "referenced_in": [ref_path], "define_in": target,
                        "via": via}
        else:
            if ref_path not in cur["referenced_in"]:
                cur["referenced_in"].append(ref_path)
            if columns and not cur["columns"]:
                cur["columns"] = columns
            if table and not cur["table"]:
                cur["table"] = table

    for d in diffs or []:
        path = _path(d)
        if not path.endswith(".py"):
            continue
        body = _body(d)
        # 1) ORM-position class references (Order.query, session.get(Order, ...)).
        for m in _ORM_USE_RE.finditer(body):
            cls = m.group(1) or m.group(2) or m.group(3)
            if not cls:
                continue
            _note(cls, _norm_table(cls), path, via="orm")
        # 2) Raw-SQL tables -- only within string literals that carry a DML verb,
        #    with columns mined from any INSERT.
        sql_blobs = [s for s in _STRING_LIT_RE.findall(body)
                     if _SQL_DML_RE.search(s)]
        cols_by_table: Dict[str, List[str]] = {}
        for blob in sql_blobs:
            for m in _SQL_INSERT_RE.finditer(blob):
                cols = [c.strip() for c in m.group(2).split(",") if c.strip()]
                cols_by_table.setdefault(_norm_table(m.group(1)), cols)
        for blob in sql_blobs:
            for m in _SQL_TABLE_RE.finditer(blob):
                tbl = m.group(1)
                key = _norm_table(tbl)
                _note(key.capitalize(), key, path, via="sql", table=tbl,
                      columns=cols_by_table.get(key))
    return list(out.values())


# ------------------------- Flask routing / config coherence -------------------------
#
# Two deterministically-fixable Flask bugs the old loop left to a weak model:
#   * a route decorated ``/api/x`` on a blueprint *also* registered with
#     ``url_prefix="/api"`` -> the live path is ``/api/api/x`` (every request 404s);
#   * a *relative* SQLite URI (``sqlite:///./data/app.db``) -> ``unable to open
#     database file`` whenever the process CWD differs from the app dir.
# Detected here (pure), fixed deterministically by the skill's ``repair_scaffold``.

_REGISTER_BP_RE = re.compile(
    r"register_blueprint\(\s*(\w+)[^)]*?url_prefix\s*=\s*['\"]([^'\"]+)['\"]")
_ROUTE_RE = re.compile(r"@(\w+)\.route\(\s*['\"]([^'\"]+)['\"]")
# ``from a.b.c import bp as members_bp`` / ``from a.b.c import bp`` -- the alias
# hop between where a blueprint is registered and where it is defined+decorated.
_IMPORT_ALIAS_RE = re.compile(
    r"^\s*from\s+([\w.]+)\s+import\s+(\w+)(?:\s+as\s+(\w+))?", re.M)
_SQLITE_REL_RE = re.compile(
    r"sqlite:///(?!/)(?!:memory:)(\.?/?[^'\"]*\.(?:db|sqlite3?|sqlite))")


def _module_of(path: str) -> str:
    """Dotted module name for a diff path (``a/b/c.py`` -> ``a.b.c``)."""
    p = (path or "").replace("\\", "/")
    if p.endswith("/__init__.py"):
        p = p[: -len("/__init__.py")]
    elif p.endswith(".py"):
        p = p[:-3]
    return p.strip("/").replace("/", ".")


def blueprint_prefix_collisions(
        diffs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Routes whose path already begins with the blueprint's ``url_prefix``.

    Correlates ``register_blueprint(bp, url_prefix=P)`` with each
    ``@bp.route(path)`` -- resolving the common **import-alias hop**
    (``from routes.members import bp as members_bp`` then
    ``register_blueprint(members_bp, url_prefix="/api")``, while the decorator in
    ``members.py`` is ``@bp.route("/api/members")``) -- and flags routes where
    ``path`` starts with ``P``, the double-prefix that makes the live URL
    ``P + P + ...`` (every request 404s). Returns ``[{file, prefix, route}]``.
    """
    # 1) alias used at registration -> (source module, original local name).
    alias_to_src: Dict[str, tuple] = {}
    for d in diffs or []:
        for m in _IMPORT_ALIAS_RE.finditer(_body(d)):
            src_mod, orig, alias = m.group(1), m.group(2), m.group(3)
            alias_to_src[alias or orig] = (src_mod, orig)
    # 2) registered url_prefix per blueprint variable (as named at registration).
    prefixes: Dict[str, str] = {}
    for d in diffs or []:
        for m in _REGISTER_BP_RE.finditer(_body(d)):
            prefixes[m.group(1)] = "/" + m.group(2).strip("/")
    if not prefixes:
        return []
    # 3) route decorators, indexed by (module, local var) so an aliased
    #    registration resolves to the file+var that actually carries the routes.
    routes_by_key: Dict[tuple, List[tuple]] = {}
    for d in diffs or []:
        path = _path(d)
        if not path.endswith(".py"):
            continue
        mod = _module_of(path)
        for m in _ROUTE_RE.finditer(_body(d)):
            routes_by_key.setdefault((mod, m.group(1)), []).append(
                (path, m.group(2)))
    out: List[Dict[str, str]] = []
    seen: set = set()
    for regvar, prefix in prefixes.items():
        src = alias_to_src.get(regvar)
        # Candidate (module, var) keys the registration could point at: the
        # resolved import source, plus a same-file registration on the bare var.
        cand = set()
        if src:
            cand.add(src)  # (module, orig local name)
        for (mod, var) in routes_by_key:
            if var == regvar or (src and var == src[1] and mod == src[0]):
                cand.add((mod, var))
        for key in cand:
            for (path, route) in routes_by_key.get(key, []):
                r = "/" + route.strip("/")
                if r == prefix or r.startswith(prefix + "/"):
                    sig = (path, route, prefix)
                    if sig not in seen:
                        seen.add(sig)
                        out.append({"file": path, "prefix": prefix,
                                    "route": route})
    return out


def relative_sqlite_uris(diffs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Config lines binding SQLAlchemy to a *relative* SQLite path.

    Returns ``[{file, uri}]`` for each ``sqlite:///./x.db`` (or ``sqlite:///x.db``)
    -- a CWD-dependent path that fails to open under pytest. An absolute path or
    ``sqlite:///:memory:`` is never flagged.
    """
    out: List[Dict[str, str]] = []
    for d in diffs or []:
        path = _path(d)
        if not path.endswith(".py"):
            continue
        for m in _SQLITE_REL_RE.finditer(_body(d)):
            out.append({"file": path, "uri": m.group(0)})
    return out


__all__ = [
    "body_of", "any_body_matches", "app_backimports", "undeclared_python_deps",
    "defined_orm_models", "undefined_model_refs", "blueprint_prefix_collisions",
    "relative_sqlite_uris",
]
