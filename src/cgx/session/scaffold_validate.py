


"""PyPI-aware ``requirements.txt`` pin validator for SCAFFOLD.

Phase 4.1 prevention gate: before SCAFFOLD persists its diff bundle,
inspect any ``requirements.txt`` the generator emitted and tighten
upper bounds on known-fragile peers using PyPI metadata. The curated
:data:`FRAGILE_PEERS` table maps a consumer package (e.g. ``flask``)
to peers whose unbounded releases have historically broken the
consumer (``werkzeug``, ``jinja2``, ``itsdangerous``, ``click``).

For each pinned consumer in the file, we fetch ``info.requires_dist``
from PyPI and reuse any declared constraint on a fragile peer verbatim;
unpinned consumers or PyPI fetch failures degrade to no-op so SCAFFOLD
never blocks on transient network errors. Reuses
:class:`cgx.session.repair.pypi_client.PyPIClient` (cache + DI hook)
from Phase 3.2.
"""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from cgx.session.repair.pypi_client import PyPIClient

logger = logging.getLogger(__name__)


# Consumer (key) -> peers (normalised lower-case) we will enforce the
# consumer's declared ``requires_dist`` constraint on. Kept small and
# conservative; expand only when a real failure case justifies adding
# a peer (each addition costs one PyPI round-trip per scaffold).
FRAGILE_PEERS: Dict[str, List[str]] = {
    "flask": ["werkzeug", "jinja2", "itsdangerous", "click"],
    "alembic": ["sqlalchemy"],
    "scipy": ["numpy"],
    "pydantic": ["pydantic-core"],
}


_REQ_BASENAMES = {"requirements.txt", "requirements-dev.txt"}

_PIN_RE = re.compile(
    r"^\s*([A-Za-z][A-Za-z0-9_.\-]*)\s*([=~<>!].*)?\s*$"
)
_EXACT_RE = re.compile(r"^==\s*([0-9][A-Za-z0-9_.\-]*)\s*$")


def is_requirements_path(path: str) -> bool:
    """Return True when ``path`` looks like a pip requirements file."""
    p = (path or "").strip().lower()
    if not p:
        return False
    base = Path(p).name
    if base in _REQ_BASENAMES:
        return True
    parts = Path(p).parts
    return len(parts) >= 2 and parts[0] == "requirements" and base.endswith(".txt")


def _normalise_name(raw: str) -> str:
    return (raw or "").strip().lower().replace("_", "-")


def _parse_pin_line(line: str) -> Optional[Tuple[str, str, str]]:
    """Return ``(raw_name, spec, normalised_name)`` or ``None``."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    # Skip ``-r foo.txt`` / ``-c constraints.txt`` includes verbatim.
    if stripped.startswith("-"):
        return None
    m = _PIN_RE.match(stripped)
    if not m:
        return None
    name = m.group(1)
    spec = (m.group(2) or "").strip()
    return name, spec, _normalise_name(name)


def _exact_version(spec: str) -> Optional[str]:
    """Extract ``X.Y.Z`` from ``==X.Y.Z`` (and nothing else)."""
    m = _EXACT_RE.match(spec or "")
    return m.group(1) if m else None


def _peer_constraint_from_requires_dist(
        info: Dict[str, Any], peer_key: str) -> Optional[str]:
    """Pull the consumer's declared constraint for ``peer_key`` out of
    ``info.requires_dist``. Returns ``"<canonical_name><spec>"`` (no
    surrounding spaces) or ``None`` when no constraint is declared.
    """
    reqs = info.get("requires_dist") or []
    if not isinstance(reqs, list):
        return None
    for raw in reqs:
        if not isinstance(raw, str):
            continue
        head = raw.split(";", 1)[0].strip()
        if not head:
            continue
        m = re.match(r"^([A-Za-z][A-Za-z0-9_.\-]*)\s*(.*)$", head)
        if not m:
            continue
        if _normalise_name(m.group(1)) != peer_key:
            continue
        rest = m.group(2).strip()
        if not rest:
            continue
        return f"{m.group(1)}{rest}"
    return None


def _replace_or_append_pin(lines: List[str], peer_key: str,
                           constraint: str) -> List[str]:
    """Replace the existing peer line in-place or append ``constraint``."""
    out: List[str] = []
    replaced = False
    for raw in lines:
        parsed = _parse_pin_line(raw)
        if parsed and parsed[2] == peer_key:
            ending = "\n" if raw.endswith("\n") else ""
            out.append(constraint + ending)
            replaced = True
        else:
            out.append(raw)
    if not replaced:
        if out and not out[-1].endswith("\n"):
            out[-1] = out[-1] + "\n"
        out.append(constraint + "\n")
    return out


def _content_to_new_file_patch(path: str, content: str) -> str:
    """Render ``content`` as a new-file unified diff (mirrors engine.py).

    Kept in lockstep with ``cgx.answer.engine._content_to_new_file_patch``
    so the swapped diff round-trips through ``apply_diffs_to_disk`` the
    same way the generator's original diff did.
    """
    lines = content.splitlines()
    n = len(lines)
    header = f"--- /dev/null\n+++ b/{path}\n@@ -0,0 +1,{n} @@\n"
    body = "\n".join(f"+{line}" for line in lines)
    return header + (body if body else "+")


def _first_party_names(paths: List[str]) -> Set[str]:
    """Derive first-party module names from generated scaffold paths.

    A top-level directory (``backend/`` -> ``backend``) and every Python
    module basename (``backend/auth.py`` -> ``auth``) are treated as
    first-party. The ``tests`` tree is excluded so a bona-fide test
    dependency is never mistaken for a project module. Names are
    normalised so they compare directly against ``requirements.txt``
    pins.
    """
    out: Set[str] = set()
    for p in paths:
        s = (p or "").strip().replace("\\", "/")
        parts = [x for x in s.split("/") if x]
        if not parts:
            continue
        top = parts[0]
        if top and top.lower() not in ("tests", "test"):
            out.add(_normalise_name(top))
        if s.endswith(".py"):
            base = parts[-1][:-3]
            if base and base != "__init__":
                out.add(_normalise_name(base))
    return out


def _sanitize_requirements_lines(
        lines: List[str],
        first_party: Set[str],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Drop stdlib/first-party pins and remap import-name aliases.

    A ``requirements.txt`` that pins a stdlib module (``sqlite3``), a
    first-party project module (``auth``), or an import name whose PyPI
    distribution differs (``jwt`` -> ``PyJWT``) either aborts
    ``pip install -r`` outright or silently installs the wrong
    distribution (so ``import jwt`` succeeds but ``jwt.encode`` is
    missing). This pass rewrites those lines before APPLY persists the
    file. Returns ``(new_lines, actions)`` where each action is a
    ``{action, name, before, after, source}`` record.
    """
    # Lazy import: keeps the module-import graph flat and avoids paying
    # for env_manager at scaffold-validate import time.
    from cgx.codegen.env_manager import _IMPORT_TO_PYPI, _STDLIB_TOP

    out: List[str] = []
    actions: List[Dict[str, Any]] = []
    for raw in lines:
        parsed = _parse_pin_line(raw)
        if not parsed:
            out.append(raw)
            continue
        name, _spec, key = parsed
        ending = "\n" if raw.endswith("\n") else ""
        if name.lower().replace("-", "_") in _STDLIB_TOP:
            actions.append({"action": "drop_stdlib", "name": name,
                            "before": raw.strip(), "after": None,
                            "source": "sanitizer"})
            continue
        if key in first_party:
            actions.append({"action": "drop_first_party", "name": name,
                            "before": raw.strip(), "after": None,
                            "source": "sanitizer"})
            continue
        alias = _IMPORT_TO_PYPI.get(name) or _IMPORT_TO_PYPI.get(name.lower())
        if alias and _normalise_name(alias) != key:
            # Drop the (wrong-package) version spec: a pin valid for the
            # shadowing distribution is unlikely to resolve for the real
            # one, so let pip pick a compatible release.
            out.append(alias + ending)
            actions.append({"action": "remap", "name": name,
                            "before": raw.strip(), "after": alias,
                            "source": "sanitizer"})
            continue
        out.append(raw)
    return out, actions


def validate_requirements_text(
        text: str, *,
        pypi_client: PyPIClient,
        first_party: Optional[Set[str]] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Sanitize a ``requirements.txt`` body and tighten fragile peers.

    First strips stdlib/first-party pins and remaps import-name aliases
    (see :func:`_sanitize_requirements_lines`), then tightens fragile
    peer pins against PyPI ``requires_dist``. Returns
    ``(new_text, adjustments)``; ``adjustments`` records every sanitizer
    action and peer rewrite. When nothing changed (and PyPI lookups
    fail or find nothing to tighten) the returned ``new_text`` is
    identical to ``text`` and ``adjustments`` is empty.
    """
    if not text:
        return text, []
    lines: List[str] = text.splitlines(keepends=True)

    adjustments: List[Dict[str, Any]] = []
    lines, san_actions = _sanitize_requirements_lines(
        lines, first_party or set())
    adjustments.extend(san_actions)

    def _index() -> Dict[str, Tuple[str, str]]:
        out: Dict[str, Tuple[str, str]] = {}
        for raw in lines:
            parsed = _parse_pin_line(raw)
            if not parsed:
                continue
            _, spec, key = parsed
            out.setdefault(key, (raw.rstrip("\n"), spec))
        return out

    pins = _index()
    for consumer_key, peers in FRAGILE_PEERS.items():
        consumer_pin = pins.get(consumer_key)
        if not consumer_pin:
            continue
        version = _exact_version(consumer_pin[1])
        if not version:
            continue
        release = pypi_client.get_release(consumer_key, version)
        if release is None:
            logger.debug(
                "scaffold pin validator: PyPI lookup failed for %s==%s",
                consumer_key, version)
            continue
        info = (release or {}).get("info") or {}
        for peer_key in peers:
            constraint = _peer_constraint_from_requires_dist(info, peer_key)
            if not constraint:
                continue
            existing = pins.get(peer_key)
            existing_spec = existing[1] if existing else None
            # When the existing line already carries the consumer's
            # declared constraint verbatim, skip -- no churn.
            if existing_spec and constraint.strip().lower() == (
                    existing[0].strip().lower()):
                continue
            lines = _replace_or_append_pin(lines, peer_key, constraint)
            adjustments.append({
                "consumer": consumer_key,
                "consumer_version": version,
                "peer": peer_key,
                "before": existing[0].strip() if existing else None,
                "after": constraint,
                "source": "requires_dist",
            })
            pins = _index()
    if not adjustments:
        return text, []
    return "".join(lines), adjustments


def validate_scaffold_diffs(
        diffs: List[Dict[str, str]],
        file_contents: Dict[str, str], *,
        pypi_client: PyPIClient,
) -> Tuple[List[Dict[str, str]], Dict[str, str], List[Dict[str, Any]]]:
    """Run :func:`validate_requirements_text` over each requirements file.

    ``file_contents`` is a ``{path: content}`` map (e.g. derived from
    the scaffold loop's ``existing_files_with_content``). For every
    diff whose ``file`` is a requirements path AND for which we have
    the source content, we rewrite the patch in place. Returns the
    possibly-rewritten ``(diffs, file_contents, adjustments)``.
    """
    if not diffs:
        return diffs, file_contents, []
    new_diffs = list(diffs)
    new_contents = dict(file_contents)
    all_adjustments: List[Dict[str, Any]] = []
    # First-party module names are drawn from every generated path so the
    # sanitizer can strip a project module that leaked into requirements.
    first_party = _first_party_names(
        list(new_contents.keys())
        + [str(e.get("file") or "") for e in new_diffs])
    for idx, entry in enumerate(new_diffs):
        path = str(entry.get("file") or "").strip()
        if not path or not is_requirements_path(path):
            continue
        original = new_contents.get(path)
        if not isinstance(original, str) or not original:
            continue
        rewritten, adjustments = validate_requirements_text(
            original, pypi_client=pypi_client, first_party=first_party)
        if not adjustments or rewritten == original:
            continue
        new_diffs[idx] = {
            "file": path,
            "patch": _content_to_new_file_patch(path, rewritten),
        }
        new_contents[path] = rewritten
        for adj in adjustments:
            adj["file"] = path
            all_adjustments.append(adj)
    return new_diffs, new_contents, all_adjustments


# --------------------- first-party import cross-check ---------------------

def _module_name_for_path(path: str) -> Optional[str]:
    """Dotted module name for a generated ``.py`` path, or ``None``.

    ``pkg/sub/mod.py`` -> ``pkg.sub.mod``; ``pkg/__init__.py`` -> ``pkg``.
    Non-Python paths return ``None``.
    """
    s = (path or "").strip().replace("\\", "/")
    if not s.endswith(".py"):
        return None
    parts = [x for x in s.split("/") if x and x != "."]
    if not parts:
        return None
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][:-3]
    return ".".join(parts) if parts else None


def _top_level_symbols(content: str,
                       include_imports: bool = True) -> Optional[Set[str]]:
    """Top-level names a ``from mod import name`` could bind, or ``None``.

    ``None`` signals the module could not be parsed, so the caller must
    abstain rather than flag a false positive. When ``include_imports`` is
    ``False`` the names merely *bound* by an ``import`` / ``from ... import``
    are omitted, leaving only the names the module actually *defines*
    (functions, classes, assignments). Contract checks that require a name
    to be authored -- not just re-exported -- use that stricter view.
    """
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return None
    names: Set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for tgt in node.targets:
                for nm in _assign_names(tgt):
                    names.add(nm)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                names.add(node.target.id)
        elif isinstance(node, ast.Import):
            if include_imports:
                for a in node.names:
                    names.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if include_imports:
                for a in node.names:
                    if a.name != "*":
                        names.add(a.asname or a.name)
    return names


def _assign_names(target: ast.AST) -> List[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        out: List[str] = []
        for e in target.elts:
            out.extend(_assign_names(e))
        return out
    return []


def _resolve_from_target(node: ast.ImportFrom, importer: str,
                         is_pkg_init: bool) -> Optional[str]:
    """Dotted target module of a ``from ... import`` (absolute or relative)."""
    if node.level:
        parts = importer.split(".") if importer else []
        base = parts if is_pkg_init else parts[:-1]
        up = node.level - 1  # level 1 == current package
        if up > len(base):
            return None
        base = base[:len(base) - up]
        target = base + (node.module.split(".") if node.module else [])
        return ".".join(target) if target else None
    return node.module or None


def cross_check_first_party_imports(
        file_contents: Dict[str, str]) -> List[Dict[str, Any]]:
    """Flag ``from <first-party> import <name>`` where ``name`` is absent.

    Best-effort, Python-only static check run after SCAFFOLD: for every
    generated ``.py`` file, resolve each ``from ... import`` whose target
    module is itself a generated file and verify the imported names are
    defined there (or are generated submodules). Third-party and
    unresolved imports are ignored, and any parse failure abstains, so
    the check never fails the scaffold -- it only surfaces
    ``{file, module, name, reason}`` warnings the router can act on.
    """
    modules: Dict[str, str] = {}
    packages: Set[str] = set()
    for path in file_contents:
        mod = _module_name_for_path(path)
        if mod is None:
            continue
        modules[mod] = path
        parts = mod.split(".")
        for i in range(1, len(parts)):
            packages.add(".".join(parts[:i]))
        if path.replace("\\", "/").endswith("__init__.py"):
            packages.add(mod)

    warnings: List[Dict[str, Any]] = []
    symbol_cache: Dict[str, Optional[Set[str]]] = {}
    for path, content in file_contents.items():
        importer = _module_name_for_path(path)
        if importer is None or not isinstance(content, str):
            continue
        try:
            tree = ast.parse(content)
        except SyntaxError:
            continue
        is_pkg_init = path.replace("\\", "/").endswith("__init__.py")
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            target = _resolve_from_target(node, importer, is_pkg_init)
            if node.level:
                # Relative imports can only reference first-party siblings in
                # the same package tree, so -- unlike absolute/third-party
                # targets -- an unresolved one is never something to abstain
                # on. Flag it when resolution fell off the top of the tree
                # (``_resolve_from_target`` -> ``None``: "attempted relative
                # import beyond top-level package") or the resolved module was
                # never generated (a phantom sibling), so the router
                # regenerates instead of shipping a scaffold that cannot
                # import. A target that resolves to a generated package with
                # no ``__init__`` still abstains via the fall-through below.
                if target is None or (
                        target not in modules and target not in packages):
                    spec = "." * node.level + (node.module or "")
                    for alias in node.names:
                        warnings.append({
                            "file": path,
                            "module": spec,
                            "name": alias.name,
                            "reason": (f"relative import {spec!r} does not "
                                       "resolve to a generated module"),
                        })
                    continue
            if target is None:
                continue
            target_path = modules.get(target)
            if target_path is None:
                # A first-party package with no generated ``__init__`` (or a
                # third-party module): no source to verify against -> abstain.
                continue
            if target_path not in symbol_cache:
                symbol_cache[target_path] = _top_level_symbols(
                    file_contents.get(target_path) or "")
            defined = symbol_cache[target_path]
            if defined is None:
                continue
            for alias in node.names:
                name = alias.name
                if name == "*":
                    continue
                sub = f"{target}.{name}"
                if sub in modules or sub in packages:
                    continue  # importing a generated submodule/subpackage
                if name not in defined:
                    warnings.append({
                        "file": path,
                        "module": target,
                        "name": name,
                        "reason": f"{name!r} not defined in {target_path}",
                    })
    return warnings


# --------------------- work-plan contract enforcement ---------------------

def check_contract_compliance(
        file_contents: Dict[str, str],
        contracts: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flag WORK_PLAN contract items no generated file satisfies.

    Best-effort static gate run after SCAFFOLD (companion to
    :func:`cross_check_first_party_imports`): checks each declared shared
    interface against the generated file set so a mismatch is caught here,
    not only when VERIFY runs the suite. Four categories are checked:

    * ``endpoints`` -- the declared ``path`` string must appear verbatim
      in some generated file (route decorators embed the path literally,
      so this scan is language-agnostic);
    * ``schemas`` -- a class/name ``name`` must be defined in some
      generated Python module;
    * ``functions`` -- a function named ``name`` must be defined, in the
      declared ``module`` when it is a generated Python file, otherwise in
      any generated module;
    * ``constants`` -- a module-level name ``name`` must be assigned in
      some generated Python module.

    The symbol checks only run when at least one Python module parsed (a
    pure JS/TS scaffold has no AST here, so they abstain); the endpoint
    substring scan always runs. Any item that cannot be evaluated is
    skipped, so the gate never fails a scaffold on its own -- it returns
    ``{kind, name, reason, ...}`` warnings the router can turn into a
    targeted regenerate constraint.
    """
    if not isinstance(contracts, dict) or not contracts:
        return []

    per_module: Dict[str, Set[str]] = {}
    all_symbols: Set[str] = set()
    # Names actually *defined* (assigned / functions / classes) across every
    # module, excluding names merely re-exported via an import. Constants must
    # be assigned, not just imported, to satisfy their contract.
    assigned_symbols: Set[str] = set()
    for path, content in file_contents.items():
        if not isinstance(content, str):
            continue
        if _module_name_for_path(path) is None:
            continue
        syms = _top_level_symbols(content)
        if syms is None:
            continue
        per_module[path.replace("\\", "/")] = syms
        all_symbols |= syms
        defined = _top_level_symbols(content, include_imports=False)
        if defined is not None:
            assigned_symbols |= defined

    have_python = bool(per_module)
    haystack = "\n".join(
        c for c in file_contents.values() if isinstance(c, str))

    warnings: List[Dict[str, Any]] = []

    for ep in contracts.get("endpoints") or []:
        if not isinstance(ep, dict):
            continue
        path = str(ep.get("path") or "").strip()
        if not path:
            continue
        if path not in haystack:
            warnings.append({
                "kind": "endpoint",
                "name": path,
                "method": str(ep.get("method") or "").strip().upper(),
                "reason": (f"declared endpoint {path!r} not found in any "
                           "generated file"),
            })

    if have_python:
        for sc in contracts.get("schemas") or []:
            if not isinstance(sc, dict):
                continue
            name = str(sc.get("name") or "").strip()
            if not name:
                continue
            if name not in all_symbols:
                warnings.append({
                    "kind": "schema",
                    "name": name,
                    "reason": (f"declared schema {name!r} has no definition "
                               "in any generated module"),
                })

        for fn in contracts.get("functions") or []:
            if not isinstance(fn, dict):
                continue
            name = str(fn.get("name") or "").strip()
            if not name:
                continue
            module = str(fn.get("module") or "").strip().replace("\\", "/")
            scope = per_module.get(module)
            if scope is not None:
                if name not in scope:
                    warnings.append({
                        "kind": "function",
                        "name": name,
                        "module": module,
                        "reason": (f"declared function {name!r} not defined "
                                   f"in {module}"),
                    })
            elif name not in all_symbols:
                warnings.append({
                    "kind": "function",
                    "name": name,
                    "module": module or None,
                    "reason": (f"declared function {name!r} not defined in "
                               "any generated module"),
                })

        for c in contracts.get("constants") or []:
            if not isinstance(c, dict):
                continue
            name = str(c.get("name") or "").strip()
            if not name:
                continue
            if name not in assigned_symbols:
                warnings.append({
                    "kind": "constant",
                    "name": name,
                    "reason": (f"declared constant {name!r} not assigned in "
                               "any generated module"),
                })

    return warnings


# ------------------- client/server payload coherence -------------------

# Frontend source extensions whose ``fetch(...)`` calls we scan for the
# JSON body they POST to a backend route.
_CLIENT_EXTS: Tuple[str, ...] = (
    ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".vue")

# A Flask/FastAPI route decorator: captures the literal path so the same
# endpoint can be matched on the client side.
_ROUTE_DECORATOR_RE = re.compile(
    r"@\s*[\w.]+\.(?:route|get|post|put|patch|delete)\(\s*"
    r"['\"]([^'\"]+)['\"]")

# Request-body key reads inside a handler window: ``data.get('k')`` /
# ``request.json['k']`` / ``body.get(\"k\")`` and friends. Broad enough for
# the dict-access style weak models emit, scoped to the handler window so a
# stray ``.get`` elsewhere is not mistaken for a payload key.
_PY_GET_READ_RE = re.compile(r"\.get\(\s*['\"]([A-Za-z_][\w-]*)['\"]")
_PY_SUBSCRIPT_READ_RE = re.compile(
    r"(?:data|body|payload|json|args|form|values|params)"
    r"\s*\[\s*['\"]([A-Za-z_][\w-]*)['\"]\s*\]")

# A ``fetch(URL`` call and the JSON body it stringifies. The body regex is
# deliberately flat (no nested braces) so only simple, confidently-parsed
# payloads are compared; anything richer abstains rather than risk a false
# positive.
_FETCH_URL_RE = re.compile(r"fetch\(\s*[`'\"]([^`'\"]+)[`'\"]")
_STRINGIFY_BODY_RE = re.compile(r"JSON\.stringify\(\s*\{([^{}]*)\}")
_FETCH_BODY_WINDOW = 600
_JS_KEY_RE = re.compile(r"^['\"]?([A-Za-z_$][\w$]*)['\"]?\s*(?::|$)")


def _js_object_keys(inner: str) -> Set[str]:
    """Keys of a flat JS object literal body (``{a, b: 1, 'c': x}``)."""
    keys: Set[str] = set()
    for part in inner.split(","):
        part = part.strip()
        if not part or part.startswith("."):  # spread ``...rest``
            continue
        m = _JS_KEY_RE.match(part)
        if m:
            keys.add(m.group(1))
    return keys


def _python_route_reads(
        file_contents: Dict[str, str]) -> List[Dict[str, Any]]:
    """Map each Flask/FastAPI route to the request-body keys its handler reads.

    Returns one ``{path, file, reads}`` record per route decorator, where
    ``reads`` is the set of literal keys the handler pulls out of the
    request body within its window (up to the next route decorator or EOF).
    Best-effort and regex-based -- a file that yields nothing simply
    contributes no records.
    """
    out: List[Dict[str, Any]] = []
    for path, content in file_contents.items():
        if not isinstance(content, str) or not path.endswith(".py"):
            continue
        matches = list(_ROUTE_DECORATOR_RE.finditer(content))
        for i, m in enumerate(matches):
            route = m.group(1).strip()
            if len(route) < 2:
                continue
            end = (matches[i + 1].start() if i + 1 < len(matches)
                   else len(content))
            window = content[m.start():end]
            reads = set(_PY_GET_READ_RE.findall(window))
            reads |= set(_PY_SUBSCRIPT_READ_RE.findall(window))
            out.append({"path": route, "file": path, "reads": reads})
    return out


def _js_fetch_sends(
        file_contents: Dict[str, str]) -> List[Dict[str, Any]]:
    """Map each ``fetch(URL, {body: JSON.stringify({...})})`` to its keys.

    Returns one ``{url, file, sends}`` record per fetch call that carries a
    flat JSON body; calls without a confidently-parsed body are skipped.
    """
    out: List[Dict[str, Any]] = []
    for path, content in file_contents.items():
        if not isinstance(content, str):
            continue
        if not any(path.endswith(ext) for ext in _CLIENT_EXTS):
            continue
        for m in _FETCH_URL_RE.finditer(content):
            url = m.group(1).strip()
            window = content[m.start():m.start() + _FETCH_BODY_WINDOW]
            body = _STRINGIFY_BODY_RE.search(window)
            if not body:
                continue
            sends = _js_object_keys(body.group(1))
            if sends:
                out.append({"url": url, "file": path, "sends": sends})
    return out


def _url_matches_route(url: str, route: str) -> bool:
    """True when a client fetch ``url`` targets backend ``route``.

    A route path (``/calculate``) matches any URL that contains it as a
    substring -- covering bare paths, ``/api``-prefixed proxies and
    absolute ``http://host:port/calculate`` forms alike.
    """
    return bool(route) and route in url


def check_client_server_payload_coherence(
        file_contents: Dict[str, str],
        contracts: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Flag a JS ``fetch`` body whose keys disagree with the backend handler.

    The JS<->Python analogue of :func:`cross_check_first_party_imports`:
    for every backend route that a frontend ``fetch`` also targets, compare
    the keys the client POSTs against the keys the handler reads (and, when
    the WORK_PLAN declares the endpoint, against its ``request`` schema).
    A *rename* disagreement -- the client sends a key the server never
    reads while the server reads a key the client never sends -- is the
    high-precision signal (e.g. ``operator`` vs ``operation``); a body that
    is merely a superset/subset is left alone so the gate does not fire on
    an optional field. Returns ``{kind: 'payload', name, file, reason, ...}``
    warnings the router can turn into a targeted regenerate of the client
    file; never raises (callers still wrap defensively).
    """
    if not isinstance(file_contents, dict) or not file_contents:
        return []
    routes = _python_route_reads(file_contents)
    if not routes:
        return []
    sends = _js_fetch_sends(file_contents)
    if not sends:
        return []

    # Declared request keys per endpoint path (authoritative when present).
    declared: Dict[str, Set[str]] = {}
    if isinstance(contracts, dict):
        for ep in contracts.get("endpoints") or []:
            if not isinstance(ep, dict):
                continue
            p = str(ep.get("path") or "").strip()
            req = ep.get("request")
            if p and isinstance(req, dict) and req:
                declared[p] = {str(k) for k in req.keys()}

    warnings: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()
    for route in routes:
        rpath = route["path"]
        reads = route["reads"]
        # The declared contract is authoritative when present; otherwise the
        # keys the handler actually reads stand in as the expected shape.
        expected = declared.get(rpath) or reads
        if not expected:
            continue
        for call in sends:
            if not _url_matches_route(call["url"], rpath):
                continue
            client = call["sends"]
            client_only = client - expected
            server_only = expected - client
            # Require divergence in *both* directions: a rename (the actual
            # bug), not a client that merely omits or adds an optional field.
            if not (client_only and server_only):
                continue
            key = (call["file"], rpath)
            if key in seen:
                continue
            seen.add(key)
            warnings.append({
                "kind": "payload",
                "name": rpath,
                "file": call["file"],
                "server_file": route["file"],
                "client_keys": sorted(client),
                "expected_keys": sorted(expected),
                "reason": (
                    f"client {call['file']} POSTs {sorted(client_only)} to "
                    f"{rpath} but the endpoint expects {sorted(server_only)} "
                    "-- align the request body keys with the backend "
                    "contract"),
            })
    return warnings


# ------------------ response-contract status coherence -----------------

# Explicit HTTP status codes a Python handler sets: a trailing status in a
# return tuple (``return jsonify(...), 201``) or a keyword form
# (``status_code=201`` / ``status=201``, covering both FastAPI decorators
# and Flask ``Response(..., status=201)``). Deliberately narrow: an implicit
# 200 (a bare ``return``) is never inferred, so an absent status abstains
# rather than guesses.
_PY_RETURN_STATUS_RE = re.compile(r"return\b[^\n]*?,\s*(\d{3})\b")
_PY_STATUS_KW_RE = re.compile(r"status(?:_code)?\s*=\s*(\d{3})")


def _python_route_statuses(
        file_contents: Dict[str, str]) -> List[Dict[str, Any]]:
    """Map each Flask/FastAPI route to the explicit status codes it sets.

    Returns one ``{path, file, statuses}`` record per route decorator, where
    ``statuses`` is the set of integer HTTP status codes the handler
    explicitly returns within its window (a trailing status in a return
    tuple or a ``status_code=`` / ``status=`` keyword). A handler that sets
    none contributes an empty set -- an implicit 200 is never inferred.
    Best-effort and regex-based, mirroring :func:`_python_route_reads`.
    """
    out: List[Dict[str, Any]] = []
    for path, content in file_contents.items():
        if not isinstance(content, str) or not path.endswith(".py"):
            continue
        matches = list(_ROUTE_DECORATOR_RE.finditer(content))
        for i, m in enumerate(matches):
            route = m.group(1).strip()
            if len(route) < 2:
                continue
            end = (matches[i + 1].start() if i + 1 < len(matches)
                   else len(content))
            window = content[m.start():end]
            statuses: Set[int] = set()
            for rx in (_PY_RETURN_STATUS_RE, _PY_STATUS_KW_RE):
                for s in rx.findall(window):
                    try:
                        statuses.add(int(s))
                    except ValueError:  # pragma: no cover - regex is \d{3}
                        continue
            out.append({"path": route, "file": path, "statuses": statuses})
    return out


def check_response_contract_coherence(
        file_contents: Dict[str, str],
        contracts: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Flag a handler whose success status disagrees with the declared one.

    Response-contract coherence (P0c): the ``endpoints`` contract may carry
    a success ``status`` (e.g. ``201`` for a create). Both the paired test
    and the handler are generated against it, so a handler that returns a
    *different* 2xx status is the same test<->implementation drift the
    assertion-repair path chases -- but caught statically here, before
    VERIFY runs the suite. For each declared endpoint carrying a 2xx
    ``status`` that matches a generated Python route:

    * the handler declares explicit 2xx status(es) and the declared one is
      not among them (contract 201, handler returns 200) -> mismatch;
    * the handler sets no explicit 2xx status (its success path is an
      implicit 200) and the declared status is not 200 -> mismatch.

    Otherwise it abstains. Emits ``{kind: "response", file, name,
    expected_status, found_statuses, reason}`` warnings the router folds
    into a targeted regenerate of the offending handler file. Returns an
    empty list when there is no contract, no Python route, or nothing to
    confidently compare.
    """
    if not isinstance(contracts, dict) or not contracts:
        return []
    if not isinstance(file_contents, dict) or not file_contents:
        return []
    declared: Dict[str, int] = {}
    for ep in contracts.get("endpoints") or []:
        if not isinstance(ep, dict):
            continue
        p = str(ep.get("path") or "").strip()
        raw = ep.get("status")
        if not p or isinstance(raw, bool):
            continue
        try:
            code = int(raw)
        except (TypeError, ValueError):
            continue
        if 200 <= code < 300:
            declared[p] = code
    if not declared:
        return []
    routes = _python_route_statuses(file_contents)
    if not routes:
        return []
    warnings: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()
    for route in routes:
        rpath = route["path"]
        code = declared.get(rpath)
        if code is None:
            continue
        handler_2xx = {s for s in route["statuses"] if 200 <= s < 300}
        if handler_2xx:
            if code in handler_2xx:
                continue
            reason = (f"handler {route['file']} returns HTTP "
                      f"{sorted(handler_2xx)} for {rpath} but the endpoint "
                      f"contract declares success status {code} -- align the "
                      "handler's success response to the declared status")
        else:
            if code == 200:
                continue
            reason = (f"handler {route['file']} sets no explicit success "
                      f"status for {rpath} (implicit 200) but the endpoint "
                      f"contract declares {code} -- return the declared "
                      "status explicitly")
        key = (route["file"], rpath)
        if key in seen:
            continue
        seen.add(key)
        warnings.append({
            "kind": "response",
            "name": rpath,
            "file": route["file"],
            "expected_status": code,
            "found_statuses": sorted(handler_2xx),
            "reason": reason,
        })
    return warnings


# --------------------- manifest stack requirements ---------------------

# Entry files a toolchain requires but a planner routinely forgets, keyed
# by the manifest path that proves the toolchain is in play. Each rule is
# ``(trigger basenames, required path, alternatives, description)``: when
# any trigger appears in the manifest and neither the required path nor
# any alternative does, the required path is missing and the build cannot
# resolve its entry module. Kept to cases where the omission is *fatal*
# and the fix is a single well-known file -- a rule that merely encodes a
# style preference would add files nobody asked for.
_STACK_ENTRY_RULES: Tuple[Tuple[Tuple[str, ...], str,
                                Tuple[str, ...], str], ...] = (
    (
        ("vite.config.js", "vite.config.ts", "vite.config.mjs",
         "vite.config.cjs"),
        "index.html",
        (),
        ("Vite entry HTML at the project root: loads the app's script "
         "entry point (e.g. <script type=\"module\" src=\"/src/main.jsx\">) "
         "and provides the mount element the entry point renders into."),
    ),
)


def stack_entry_description(path: str) -> str:
    """Return the manifest description to use for entry file ``path``.

    Shared by the DECOMPOSE injection and the REPAIR regenerate path so a
    file added late (because the bundler could not resolve it) is
    described exactly as it would have been had the planner remembered
    it. Falls back to a generic entry-module description for paths no
    rule covers.
    """
    want = str(path or "").strip().replace("\\", "/").lstrip("./")
    for _triggers, required, _alternatives, description in _STACK_ENTRY_RULES:
        if required == want:
            return description
    return (f"Build entry module {want!r}: the bundler resolves the "
            "application from this file, so it must exist and load the "
            "app's script entry point.")


def missing_stack_entry_files(paths: Sequence[str]) -> List[Dict[str, str]]:
    """Return the entry files ``paths`` implies but does not contain.

    Deterministic manifest gate: a Vite manifest without a root
    ``index.html`` cannot build at all (``[UNRESOLVED_ENTRY] Cannot
    resolve entry module index.html``), and no amount of re-authoring the
    files that *are* planned can fix it -- the fix is a file that must
    exist. Rather than let SCAFFOLD generate an unbuildable tree and burn
    the repair budget discovering it, DECOMPOSE folds the missing entries
    straight into the manifest.

    Returns one ``{path, description}`` dict per missing entry, in rule
    order; an empty list when the manifest is already coherent.
    """
    have = {str(p or "").strip().replace("\\", "/").lstrip("./")
            for p in paths}
    basenames = {p.rsplit("/", 1)[-1] for p in have}
    out: List[Dict[str, str]] = []
    for triggers, required, alternatives, description in _STACK_ENTRY_RULES:
        if not basenames & set(triggers):
            continue
        if required in have or any(a in have for a in alternatives):
            continue
        out.append({"path": required, "description": description})
    return out
