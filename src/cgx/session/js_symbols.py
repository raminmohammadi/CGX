"""JS/TS module surface + cross-file symbol checks -- the Python gates' twin.

The Python side of the swarm grounds every file on the *real* exports of its
dependencies (``list_symbols``, ``render_import_hints``, the first-party
symbol gate). The JS/TS side had none of that: ``list_symbols`` returned ``[]``
for a ``.jsx`` file, no import hints were rendered, and the structural scan
only checked that a relative import *resolved to a file* -- never that the
imported names existed. A weak model therefore guessed (``import { storage }
from './storage'`` against a module exporting functions; ``StrictMode`` used
but never imported; two nested ``<BrowserRouter>``) and the tree built or
booted broken.

This module closes that gap deterministically with regex-level parsing of the
ES-module surface (good enough for generated code; it abstains whenever it
is unsure):

* :func:`js_exports` -- the names a module exports (default + named).
* :func:`js_imports` -- every import statement's specifier and bound names.
* :func:`symbol_list` / :func:`skeleton` -- the grounding views the Developer
  tools show for a JS/TS file.
* :func:`import_hints` -- ready-to-paste import lines for a file's JS deps.
* :func:`js_symbol_breaks` -- gating rows for the Verifier: a named/default
  import no dependency exports, a React/router identifier used but never
  imported, a ``<Router>`` nested inside the entry's ``<BrowserRouter>``.

Never raises; every function degrades to "no opinion" on odd input.
"""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

__all__ = [
    "JS_EXTS", "ImportStmt", "js_exports", "js_imports", "symbol_list", "skeleton",
    "import_hints", "js_symbol_breaks", "resolve_relative", "is_js_path",
]

JS_EXTS: Tuple[str, ...] = (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs")
_CODE_EXTS: Tuple[str, ...] = JS_EXTS + (".vue",)

# ``export`` may follow another statement on the same line, so anchor on a
# non-identifier boundary rather than the start of a line.
_RE_EXPORT_DECL = re.compile(
    r"(?<![\w$.])export\s+(?:async\s+)?(?:const|let|var|function\*?|class|enum|interface|type)\s+"
    r"([A-Za-z_$][\w$]*)")
_RE_EXPORT_DEFAULT_NAMED = re.compile(
    r"(?<![\w$.])export\s+default\s+(?:async\s+)?(?:function\*?|class)\s+([A-Za-z_$][\w$]*)")
_RE_EXPORT_DEFAULT_ANY = re.compile(r"(?<![\w$.])export\s+default\b")
_RE_EXPORT_DEFAULT_IDENT = re.compile(r"(?<![\w$.])export\s+default\s+([A-Za-z_$][\w$]*)\s*;?\s*$", re.M)
_RE_EXPORT_LIST = re.compile(r"(?<![\w$.])export\s*\{([^}]*)\}\s*;?")
_RE_CJS_OBJ = re.compile(r"module\.exports\s*=\s*\{([^}]*)\}")
_RE_CJS_PROP = re.compile(r"(?:module\.)?exports\.([A-Za-z_$][\w$]*)\s*=")

_RE_IMPORT = re.compile(
    r"^\s*import\s+(?P<clause>[^;'\"]*?)\s*from\s*['\"](?P<spec>[^'\"]+)['\"]\s*;?", re.M)
_RE_IMPORT_SIDE = re.compile(r"^\s*import\s*['\"](?P<spec>[^'\"]+)['\"]\s*;?", re.M)
_RE_IMPORT_TYPE = re.compile(r"^\s*import\s+type\b", re.M)

_RE_DECL = re.compile(
    r"\b(?:function\*?|class|const|let|var)\s+([A-Za-z_$][\w$]*)")
_RE_DESTRUCTURE = re.compile(r"\b(?:const|let|var)\s*[\[{]([^=]*?)[\]}]\s*=")
_RE_JSX_TAG = re.compile(r"<([A-Z][\w$]*)\b")
_RE_FUNC_PARAMS = re.compile(r"\(([^)]*)\)\s*(?:=>|\{)")

# Identifiers a React/Vite app uses that MUST be imported; value = the module.
REACT_API: Dict[str, str] = {
    "StrictMode": "react", "Fragment": "react", "Suspense": "react", "lazy": "react",
    "useState": "react", "useEffect": "react", "useRef": "react", "useMemo": "react",
    "useCallback": "react", "useContext": "react", "useReducer": "react", "createContext": "react",
    "forwardRef": "react", "memo": "react", "useId": "react", "useLayoutEffect": "react",
    "createRoot": "react-dom/client", "hydrateRoot": "react-dom/client",
    "BrowserRouter": "react-router", "HashRouter": "react-router", "Routes": "react-router",
    "Route": "react-router", "Link": "react-router", "NavLink": "react-router", "Navigate": "react-router",
    "Outlet": "react-router", "useNavigate": "react-router", "useLocation": "react-router",
    "useParams": "react-router", "useSearchParams": "react-router", "RouterProvider": "react-router",
}
_ROUTER_WRAPPERS = ("BrowserRouter", "HashRouter", "Router", "MemoryRouter")


def is_js_path(path: str) -> bool:
    return (path or "").lower().endswith(JS_EXTS)


# --------------------------------------------------------------------------
# module surface
# --------------------------------------------------------------------------

def _strip_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", "", src or "", flags=re.S)
    return re.sub(r"(?m)^\s*//.*$", "", src)


def collapse_braces(src: str, rounds: int = 12) -> str:
    """Replace every balanced ``{...}`` with ``{}`` so JSX attribute expressions
    (``onClick={() => x > 1}``) cannot fool a ``<Tag ...>`` regex with an inner ``>``."""
    out = src or ""
    for _ in range(rounds):
        nxt = re.sub(r"\{[^{}]*\}", "{}", out)
        if nxt == out:
            break
        out = nxt
    return out


def js_exports(content: str) -> Dict[str, Any]:
    """``{"named": [...], "default": name|None, "has_default": bool}`` for a module.

    ``default`` is the exported identifier when the module writes
    ``export default Name`` / ``export default function Name``; an anonymous
    default (``export default () => ...``) sets ``has_default`` only.
    """
    src = _strip_comments(content)
    named: List[str] = []
    for m in _RE_EXPORT_DECL.finditer(src):
        if m.group(1) not in named:
            named.append(m.group(1))
    for m in _RE_EXPORT_LIST.finditer(src):
        for part in m.group(1).split(","):
            part = part.strip()
            if not part:
                continue
            if " as " in part:
                local, alias = (x.strip() for x in part.split(" as ", 1))
                exported = alias
            else:
                exported = part
            if exported == "default":
                continue
            if exported and exported not in named:
                named.append(exported)
    for m in _RE_CJS_OBJ.finditer(src):
        for part in m.group(1).split(","):
            name = part.split(":")[0].strip()
            if re.fullmatch(r"[A-Za-z_$][\w$]*", name or "") and name not in named:
                named.append(name)
    for m in _RE_CJS_PROP.finditer(src):
        if m.group(1) not in named:
            named.append(m.group(1))
    default: Optional[str] = None
    m = _RE_EXPORT_DEFAULT_NAMED.search(src)
    if m:
        default = m.group(1)
    else:
        m = _RE_EXPORT_DEFAULT_IDENT.search(src)
        if m:
            default = m.group(1)
    has_default = bool(_RE_EXPORT_DEFAULT_ANY.search(src)) or any(
        "default as" in m.group(1) or re.search(r"\bdefault\b", m.group(1)) for m in _RE_EXPORT_LIST.finditer(src))
    return {"named": named, "default": default, "has_default": has_default}


@dataclass
class ImportStmt:
    spec: str
    default: Optional[str] = None
    named: List[Tuple[str, str]] = field(default_factory=list)  # (imported, local)
    namespace: Optional[str] = None
    side_effect: bool = False
    type_only: bool = False

    def bound_names(self) -> List[str]:
        out = [self.default] if self.default else []
        out += [local for _, local in self.named]
        if self.namespace:
            out.append(self.namespace)
        return out


def js_imports(content: str) -> List[ImportStmt]:
    """Every import statement in a module (ES modules; CommonJS ``require`` is skipped)."""
    src = _strip_comments(content)
    out: List[ImportStmt] = []
    for m in _RE_IMPORT.finditer(src):
        clause = (m.group("clause") or "").strip()
        spec = m.group("spec").strip()
        stmt = ImportStmt(spec=spec, type_only=clause.startswith("type "))
        if clause.startswith("type "):
            clause = clause[5:].strip()
        if clause.startswith("* as "):
            stmt.namespace = clause[5:].strip()
        else:
            brace = clause.find("{")
            head = clause if brace < 0 else clause[:brace]
            head = head.strip().rstrip(",").strip()
            if head and re.fullmatch(r"[A-Za-z_$][\w$]*", head):
                stmt.default = head
            if brace >= 0:
                inner = clause[brace + 1: clause.rfind("}")] if "}" in clause else clause[brace + 1:]
                for part in inner.split(","):
                    part = part.strip()
                    if not part:
                        continue
                    if part.startswith("type "):
                        part = part[5:].strip()
                    if " as " in part:
                        imported, local = (x.strip() for x in part.split(" as ", 1))
                    else:
                        imported = local = part
                    if imported:
                        stmt.named.append((imported, local))
        out.append(stmt)
    for m in _RE_IMPORT_SIDE.finditer(src):
        out.append(ImportStmt(spec=m.group("spec").strip(), side_effect=True))
    return out


def _declared_names(content: str) -> FrozenSet[str]:
    src = _strip_comments(content)
    names = set(_RE_DECL.findall(src))
    for m in _RE_DESTRUCTURE.finditer(src):
        for part in re.split(r"[,\s]+", m.group(1)):
            part = part.split(":")[-1].split("=")[0].strip()
            if re.fullmatch(r"[A-Za-z_$][\w$]*", part or ""):
                names.add(part)
    for m in _RE_FUNC_PARAMS.finditer(src):
        for part in re.split(r"[,\s{}\[\]]+", m.group(1)):
            part = part.split(":")[0].split("=")[0].strip()
            if re.fullmatch(r"[A-Za-z_$][\w$]*", part or ""):
                names.add(part)
    for stmt in js_imports(content):
        names.update(stmt.bound_names())
    return frozenset(names)


def symbol_list(content: str) -> List[Dict[str, str]]:
    """``[{name, kind, signature}]`` for the exported surface of a JS/TS module."""
    src = _strip_comments(content)
    out: List[Dict[str, str]] = []
    ex = js_exports(content)
    for name in ex["named"]:
        sig = _signature_for(src, name) or name
        out.append({"name": name, "kind": "export", "signature": f"export {sig}"})
    if ex["has_default"]:
        name = ex["default"] or "(anonymous)"
        sig = _signature_for(src, ex["default"]) if ex["default"] else None
        out.append({"name": name, "kind": "default export",
                    "signature": f"export default {sig or name}"})
    return out


def _signature_for(src: str, name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    m = re.search(r"(?:async\s+)?function\*?\s+" + re.escape(name) + r"\s*\(([^)]*)\)", src)
    if m:
        return f"function {name}({' '.join(m.group(1).split())})"
    m = re.search(r"\b(?:const|let|var)\s+" + re.escape(name) + r"\s*=\s*(?:async\s*)?\(([^)]*)\)\s*=>", src)
    if m:
        return f"const {name} = ({' '.join(m.group(1).split())}) =>"
    m = re.search(r"\b(?:const|let|var)\s+" + re.escape(name) + r"\s*=\s*(?:async\s+)?([A-Za-z_$][\w$]*)\s*=>", src)
    if m:
        return f"const {name} = ({m.group(1)}) =>"
    if re.search(r"\bclass\s+" + re.escape(name) + r"\b", src):
        return f"class {name}"
    if re.search(r"\b(?:const|let|var)\s+" + re.escape(name) + r"\b", src):
        return f"const {name}"
    return None


def skeleton(content: str, path: str = "") -> str:
    """A compact ``imports + export signatures`` view of a JS/TS module."""
    lines: List[str] = []
    for stmt in js_imports(content):
        if stmt.side_effect:
            lines.append(f"import '{stmt.spec}';")
            continue
        parts = []
        if stmt.default:
            parts.append(stmt.default)
        if stmt.namespace:
            parts.append(f"* as {stmt.namespace}")
        if stmt.named:
            parts.append("{ " + ", ".join(i if i == l else f"{i} as {l}" for i, l in stmt.named) + " }")
        lines.append(f"import {', '.join(parts)} from '{stmt.spec}';")
    syms = symbol_list(content)
    if syms:
        lines.append("")
        lines.append(f"// exports of {path or 'this module'}:")
        lines.extend(s["signature"] + ";" for s in syms)
    else:
        lines.append("// (no exports detected)")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# resolution + hints
# --------------------------------------------------------------------------

def resolve_relative(spec: str, importer: str, manifest: Sequence[str]) -> Optional[str]:
    """The manifest path a relative ``spec`` (from ``importer``) resolves to, else ``None``."""
    if not spec or spec[0] not in (".", "/"):
        return None
    files = {p.replace("\\", "/").lstrip("./") for p in manifest if p}
    base = posixpath.dirname(importer.replace("\\", "/"))
    target = posixpath.normpath(posixpath.join(base, spec)) if spec[0] == "." else spec.lstrip("/")
    if target.startswith(".."):
        return None
    target = target.lstrip("./") or target
    if target in files:
        return target
    for e in _CODE_EXTS:
        if target + e in files:
            return target + e
        idx = posixpath.join(target, "index" + e)
        if idx in files:
            return idx
    return None


def _rel_spec(importer: str, dep: str) -> str:
    rel = posixpath.relpath(dep.replace("\\", "/"), posixpath.dirname(importer.replace("\\", "/")) or ".")
    return rel if rel.startswith(".") else "./" + rel


def import_hints(path: str, dep_sources: Dict[str, str]) -> str:
    """Ready-to-paste import lines for a JS/TS file's JS/TS dependencies, or ``""``."""
    if not is_js_path(path):
        return ""
    lines: List[str] = []
    for dep_path, src in (dep_sources or {}).items():
        if not is_js_path(dep_path) or not isinstance(src, str) or not src.strip():
            continue
        if dep_path == path:
            continue
        ex = js_exports(src)
        if not ex["named"] and not ex["has_default"]:
            continue
        spec = _rel_spec(path, dep_path)
        parts: List[str] = []
        if ex["has_default"]:
            parts.append(ex["default"] or posixpath.splitext(posixpath.basename(dep_path))[0])
        if ex["named"]:
            parts.append("{ " + ", ".join(ex["named"]) + " }")
        lines.append(f"import {', '.join(parts)} from '{spec}';")
    if not lines:
        return ""
    return ("SUGGESTED IMPORTS (verified against the dependencies' REAL exports -- "
            "copy the names exactly; import only what you use; these modules export "
            "NOTHING else):\n" + "\n".join(lines))


# --------------------------------------------------------------------------
# gating rows for the Verifier
# --------------------------------------------------------------------------

def _row(file: str, reason: str, directive: str, kind: str = "js_symbol",
         module: str = "", gating: bool = True) -> Dict[str, Any]:
    """A regen row. ``gating=False`` rows (prop-contract drift) steer a
    regeneration round but never fail the verdict on their own -- the build and
    the runtime smoke decide."""
    return {"file": file, "define_in": file, "module": module, "kind": kind,
            "reason": reason, "directive": directive, "gating": gating}


def js_symbol_breaks(contents: Dict[str, str], paths: Sequence[str], *,
                     entry_path: str = "", app_path: str = "") -> List[Dict[str, Any]]:
    """Cross-file JS/TS symbol defects as regen rows (``define_in`` = the importer).

    * a named import no resolved dependency exports / a default import from a
      module with no default export -- the directive lists the real exports;
    * a React / React Router identifier used (JSX tag or call) but never
      imported or declared in the file;
    * a ``<Router>`` wrapper in the app shell while the entry already wraps the
      app in one (React Router refuses nested routers -> blank page).
    Abstains per file on anything it cannot parse confidently.
    """
    rows: List[Dict[str, Any]] = []
    try:
        manifest = [p for p in paths if p]
        js_files = {p: c for p, c in (contents or {}).items()
                    if is_js_path(p) and isinstance(c, str) and c.strip()}
        for importer, src in js_files.items():
            rows.extend(_import_breaks(importer, src, js_files, manifest))
            if importer.lower().endswith((".jsx", ".tsx")) or "react" in src:
                rows.extend(_undefined_identifier_breaks(importer, src, manifest))
        rows.extend(_nested_router_breaks(js_files, entry_path, app_path))
        rows.extend(_missing_router_breaks(js_files, entry_path))
        rows.extend(_duplicate_routes_breaks(js_files, app_path))
        rows.extend(_prop_contract_breaks(js_files, manifest))
    except Exception:  # pragma: no cover - checks are best-effort
        return rows
    return rows


_RE_COMPONENT_PARAMS = re.compile(
    r"(?:export\s+default\s+function\s+[A-Z]\w*|(?:const|function)\s+(?P<name>[A-Z]\w*)\s*=?\s*(?:\([^)]*\)\s*=>)?)"
    r"\s*\(\s*(?P<params>[^)]*)\)")


def _component_props(src: str, name: str) -> Optional[Tuple[bool, List[str]]]:
    """(takes_props, destructured names) for the default-exported component ``name``."""
    body = _strip_comments(src)
    pats = [r"function\s+" + re.escape(name) + r"\s*\(([^)]*)\)",
            r"(?:const|let|var)\s+" + re.escape(name) + r"\s*=\s*(?:React\.memo\()?(?:async\s*)?\(([^)]*)\)\s*=>",
            r"(?:const|let|var)\s+" + re.escape(name) + r"\s*=\s*\(?\s*([A-Za-z_$][\w$]*)\s*\)?\s*=>"]
    for pat in pats:
        m = re.search(pat, body)
        if not m:
            continue
        params = (m.group(1) or "").strip()
        if not params:
            return (False, [])
        if params.startswith("{"):
            inner = params.strip("{}")
            names = [p.split(":")[0].split("=")[0].strip() for p in inner.split(",")]
            names = [n for n in names if re.fullmatch(r"[A-Za-z_$][\w$]*", n or "")]
            if "..." in inner:
                return (True, [])  # a rest element accepts anything
            return (True, names)
        return (True, [])  # ``props`` object: accepts anything
    return None


def _prop_contract_breaks(js_files: Dict[str, str], manifest: Sequence[str]) -> List[Dict[str, Any]]:
    """Caller/component prop drift: props passed but ignored, or required but omitted."""
    rows: List[Dict[str, Any]] = []
    for comp_path, comp_src in js_files.items():
        ex = js_exports(comp_src)
        name = ex.get("default")
        if not name or not name[0].isupper():
            continue
        info = _component_props(comp_src, name)
        if info is None:
            continue
        takes_props, declared = info
        for caller, src in js_files.items():
            if caller == comp_path:
                continue
            # the caller must import THIS file's default export under that name
            bound = None
            for stmt in js_imports(src):
                if stmt.default and resolve_relative(stmt.spec, caller, manifest) == comp_path:
                    bound = stmt.default
                    break
            if not bound:
                continue
            usages = list(re.finditer(r"<" + re.escape(bound) + r"\b([^>]*?)/?>",
                                      collapse_braces(_strip_comments(src)), re.S))
            if not usages:
                continue
            passed: List[str] = []
            spread = False
            for u in usages:
                attrs = u.group(1)
                if "{..." in attrs or "{}" in re.sub(r"=\s*\{\}", "", attrs):
                    spread = True  # a bare {} left by collapsing a {...rest} spread
                # attribute names: everything that is not a quoted string or a
                # collapsed expression -- so bare boolean props (``disabled``) count
                clean = re.sub(r"\"[^\"]*\"|'[^']*'|\{\}|=", " ", attrs)
                passed += re.findall(r"(?<![\w$-])[A-Za-z_$][\w$-]*", clean)
            passed = sorted({p for p in passed if p not in ("key", "ref")})
            if takes_props and declared and passed and not spread:
                # Props the caller passes that the component never destructures
                # (``<Composer disabled />`` into ``({ onSend }) =>``) are silently
                # dropped -- a disabled state or callback that never happens.
                unused = [p for p in passed if p not in declared and p != "children"]
                if unused:
                    rows.append(_row(
                        comp_path,
                        f"{comp_path} ({name}) does not accept the props its caller {caller} passes: "
                        f"{', '.join(unused)} (it destructures only {', '.join(declared)})",
                        f"Accept and USE the props callers pass -- {', '.join(unused)} -- in {name}: add them to "
                        "the destructured signature and wire them into the markup/handlers (e.g. a `disabled` "
                        "prop disables the controls).", kind="js_props", gating=False))
            if not takes_props and passed:
                rows.append(_row(
                    comp_path,
                    f"{comp_path} ({name}) ignores the props its caller {caller} passes: {', '.join(passed)}",
                    f"Accept and RENDER the props callers pass -- {', '.join(passed)} -- in {name} "
                    f"(destructure them in the signature and use them in the markup; keep sensible defaults).",
                    kind="js_props", gating=False))
            elif takes_props and declared and not passed and not spread:
                required = [d for d in declared if d.lower() != "children"]
                if required:
                    rows.append(_row(
                        caller,
                        f"{caller} renders <{bound} /> without any props although {comp_path} destructures "
                        f"{', '.join(required)}",
                        f"Pass the props {name} needs ({', '.join(required)}) where <{bound} /> is rendered in "
                        f"{caller}, wired to real state (not placeholders).", kind="js_props", gating=False))
    return rows


_ROUTER_PROVIDER_RE = re.compile(r"<(?:BrowserRouter|HashRouter|MemoryRouter|Router|RouterProvider)\b|createBrowserRouter\(|createHashRouter\(")


def _missing_router_breaks(js_files: Dict[str, str], entry_path: str) -> List[Dict[str, Any]]:
    """``<Routes>``/``<Route>``/``<Link>`` used but no Router provider anywhere: React
    Router throws at mount and the whole app renders blank."""
    uses = [p for p, c in js_files.items() if re.search(r"<(?:Routes|Route|Link|NavLink)\b", c)
            or re.search(r"\buse(?:Navigate|Location|Params)\s*\(", c)]
    if not uses:
        return []
    if any(_ROUTER_PROVIDER_RE.search(c) for c in js_files.values()):
        return []
    entry = entry_path if entry_path in js_files else next(
        (p for p, c in js_files.items() if "createRoot" in c or "createApp" in c), uses[0])
    return [_row(
        entry,
        f"routing is used in {', '.join(sorted(uses)[:4])} but no <BrowserRouter> (or RouterProvider) wraps the "
        "app -- React Router throws at mount and every route renders a blank page",
        f"Wrap the app in a router in {entry}: import {{ BrowserRouter }} from 'react-router' and render "
        "<StrictMode><BrowserRouter><App /></BrowserRouter></StrictMode>.")]


def _duplicate_routes_breaks(js_files: Dict[str, str], app_path: str) -> List[Dict[str, Any]]:
    """More than one file renders a ``<Routes>`` table: screens mount twice."""
    owners = [p for p, c in js_files.items() if re.search(r"<Routes\b", c)]
    if len(owners) < 2:
        return []
    keep = app_path if app_path in owners else next(
        (p for p in owners if p.rsplit("/", 1)[-1].lower().startswith("app.")), owners[0])
    rows: List[Dict[str, Any]] = []
    for p in owners:
        if p == keep:
            continue
        rows.append(_row(
            p,
            f"{p} renders its own <Routes> although {keep} already owns the route table -- every "
            "screen mounts twice and navigation state splits",
            f"Remove <Routes>/<Route> (and the page imports) from {p}; render {{children}} where the "
            f"routed screen should appear. {keep} owns the single route table."))
    return rows


def _import_breaks(importer: str, src: str, js_files: Dict[str, str],
                   manifest: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for stmt in js_imports(src):
        if stmt.side_effect or stmt.type_only or stmt.spec[0] not in (".", "/"):
            continue
        target = resolve_relative(stmt.spec, importer, manifest)
        if not target or target not in js_files:
            continue  # unresolved modules are cross_ref's job; non-JS targets abstain
        ex = js_exports(js_files[target])
        if not ex["named"] and not ex["has_default"]:
            continue  # could not read the surface -> abstain
        missing = [i for i, _ in stmt.named if i != "default" and i not in ex["named"]]
        bad_default = bool(stmt.default) and not ex["has_default"]
        if not missing and not bad_default:
            continue
        exported = []
        if ex["has_default"]:
            exported.append(f"default export {ex['default'] or '(anonymous)'}")
        if ex["named"]:
            exported.append("named exports: " + ", ".join(ex["named"]))
        what = []
        if missing:
            what.append(f"imports {{ {', '.join(missing)} }}")
        if bad_default:
            what.append(f"imports a default '{stmt.default}'")
        correct = []
        if ex["has_default"]:
            correct.append(ex["default"] or posixpath.splitext(posixpath.basename(target))[0])
        if ex["named"]:
            correct.append("{ " + ", ".join(ex["named"]) + " }")
        rows.append(_row(
            importer,
            f"{importer} {' and '.join(what)} from '{stmt.spec}', but {target} has "
            f"{'; '.join(exported)}",
            f"Fix the import of '{stmt.spec}': {target} has {'; '.join(exported)}. Write "
            f"`import {', '.join(correct)} from '{stmt.spec}';` and update every call site to use those "
            f"exact names (e.g. call the named functions directly instead of through a namespace object).",
            module=stmt.spec))
    return rows


_HTML_TAGS = frozenset("""a abbr address area article aside audio b base bdi bdo blockquote body br button canvas
caption cite code col colgroup data datalist dd del details dfn dialog div dl dt em embed fieldset figcaption figure
footer form h1 h2 h3 h4 h5 h6 head header hgroup hr html i iframe img input ins kbd label legend li link main map
mark menu meta meter nav noscript object ol optgroup option output p param picture pre progress q rp rt ruby s samp
script section select slot small source span strong style sub summary sup svg table tbody td template textarea
tfoot th thead time title tr track u ul var video wbr path circle rect line polyline polygon g defs use text""".split())


_GLOBALS = frozenset("""window document console localStorage sessionStorage indexedDB JSON Math Object Array
Promise Date Number String Boolean Symbol Map Set WeakMap WeakSet Reflect Proxy Error TypeError RangeError fetch
setTimeout clearTimeout setInterval clearInterval requestAnimationFrame cancelAnimationFrame navigator location
history crypto performance Intl structuredClone globalThis queueMicrotask atob btoa URL URLSearchParams
AbortController FormData Blob File FileReader Headers Request Response TextEncoder TextDecoder WebSocket
EventSource Notification process require module exports this super arguments e event evt err error ev args
props state ref el node item row entry msg message data res response result value val key index i j k acc prev
next cur curr obj arr list items options opts config settings params query body text str num id name""".split())
_RE_NS_CALL = re.compile(r"(?<![\w$.])([a-z_$][\w$]*)\.[A-Za-z_$][\w$]*\s*\(")


def _undefined_identifier_breaks(importer: str, src: str,
                                 manifest: Sequence[str]) -> List[Dict[str, Any]]:
    declared = _declared_names(src)
    body = _strip_comments(src)
    missing: Dict[str, str] = {}
    # A lowercase namespace that is called like a module (``storage.load()``)
    # but is neither imported, declared, a parameter nor a browser global.
    for ns in sorted(set(_RE_NS_CALL.findall(body))):
        if ns in declared or ns in _GLOBALS or len(ns) < 2:
            continue
        missing[ns] = "module"
    # React / router APIs used as calls or JSX tags.
    for name, module in REACT_API.items():
        if name in declared:
            continue
        if re.search(r"<" + name + r"\b", body) or re.search(r"\b" + name + r"\s*\(", body):
            missing[name] = module
    # Capitalised JSX tags: must be imported or declared in the file.
    for tag in sorted(set(_RE_JSX_TAG.findall(body))):
        if tag in declared or tag.lower() in _HTML_TAGS or tag in missing or tag in REACT_API:
            continue
        if tag in ("React",):
            continue
        missing[tag] = "component"
    if not missing:
        return []
    fixes = []
    by_module: Dict[str, List[str]] = {}
    for name, module in missing.items():
        by_module.setdefault(module, []).append(name)
    for module, names in by_module.items():
        if module == "component":
            fixes.append(f"import the component(s) {', '.join(names)} from their file(s) in the PROJECT MANIFEST "
                         "(or define them in this file)")
        elif module == "module":
            fixes.append(f"'{', '.join(names)}' is used like an imported module but nothing imports or defines "
                         "it -- import the EXACT exported names from the helper module in the PROJECT MANIFEST "
                         "(see its listed exports) and call them directly")
        else:
            fixes.append(f"import {{ {', '.join(names)} }} from '{module}'")
    return [_row(
        importer,
        f"{importer} uses {', '.join(sorted(missing))} without importing or declaring them "
        "(ReferenceError at runtime -> blank page)",
        "Add the missing imports at the top of the file: " + "; ".join(fixes) + ".")]


def _nested_router_breaks(js_files: Dict[str, str], entry_path: str,
                          app_path: str) -> List[Dict[str, Any]]:
    entry = js_files.get(entry_path, "") if entry_path else ""
    if not entry:
        # find an entry that wraps a Router
        for p, c in js_files.items():
            if "createRoot" in c and any(w in c for w in _ROUTER_WRAPPERS):
                entry, entry_path = c, p
                break
    if not entry or not any(re.search(r"<" + w + r"\b", entry) for w in _ROUTER_WRAPPERS):
        return []
    rows: List[Dict[str, Any]] = []
    for p, c in js_files.items():
        if p == entry_path:
            continue
        if any(re.search(r"<" + w + r"\b", c) for w in _ROUTER_WRAPPERS):
            rows.append(_row(
                p,
                f"{p} wraps its tree in a Router although {entry_path} already renders <BrowserRouter> -- "
                "React Router throws on nested routers (blank page)",
                f"Remove the Router/BrowserRouter wrapper (and its import) from {p}; keep <Routes>/<Route>/"
                f"<NavLink> only. {entry_path} already provides the router."))
    return rows
