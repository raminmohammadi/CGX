

"""React frontend skill.

Detects goals naming React (but not React Native), supplies a
Vite-based scaffold prompt, and validates that scaffold outputs
actually contain JS/TS source files rather than a Python fallback.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from skills._structural import any_body_matches, body_of
from skills.base import (
    Skill,
    SkillVerdict,
    file_paths,
    has_any_ext,
    has_js_test_file,
)

_REACT_NATIVE_RE = re.compile(r"\breact\s*native\b", re.IGNORECASE)
_REACT_RE = re.compile(r"\breact(?:\.?js)?\b", re.IGNORECASE)
_JSX_RE = re.compile(r"\b(?:jsx|tsx)\b", re.IGNORECASE)
# Common typo: "reach" instead of "react". Only fire when paired with a
# frontend-context word so we don't false-match the ordinary English verb
# ("extend the reach of the API").
_REACT_TYPO_RE = re.compile(
    r"\breach\b\s+(?:frontend|front-end|ui|app|js|jsx|tsx|hooks?|components?|sfc)\b",
    re.IGNORECASE,
)

# Plain .js/.ts extensions -- deliberately NOT .jsx/.tsx. Vite/esbuild only
# transforms JSX in the x-suffixed files, so JSX here is a build failure.
_PLAIN_JS_EXTS = (".js", ".ts")

# An import of the ``react`` module proper (not a plugin such as
# ``@vitejs/plugin-react`` nor a subpath like ``react/jsx-runtime``): the
# quote must sit immediately either side of ``react``.
_REACT_MODULE_IMPORT_RE = (
    r"""(?:from\s+['"]react['"]"""
    r"""|require\(\s*['"]react['"]\s*\)"""
    r"""|import\s+['"]react['"])"""
)
# JSX markers, exactly the two the spec calls out: a closing tag (``</div``,
# ``</App``) or a parenthesised JSX return (``return (<Foo``). Both are absent
# from ordinary JS/TS (``return (a < b)`` opens with an identifier, not ``<``;
# TS generics close with ``>`` never ``</``), so pairing them with a real
# ``react`` import keeps this high precision.
_JSX_BODY_RE = r"(?:</\s*[A-Za-z])|(?:return\s*\(\s*<[A-Za-z])"

# A local ``<script src=...>`` in index.html (module entry). Group 1 is the src.
_HTML_SCRIPT_SRC_RE = re.compile(
    r"""<script\b[^>]*\bsrc\s*=\s*['"]([^'"]+)['"]""", re.IGNORECASE)

# Files a diff only carries when it is standing up a fresh Vite project (as
# opposed to editing existing components). index.html is the HTML build entry.
_SCAFFOLD_MARKERS = frozenset(
    {"index.html", "package.json", "vite.config.js", "vite.config.ts"})


def _basename(path: str) -> str:
    return path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].lower()


def _looks_like_full_scaffold(paths: List[str]) -> bool:
    """True when the diff reads as a fresh scaffold rather than an edit.

    A full scaffold is either >= 3 files or carries a build/manifest entry
    file (index.html / package.json / vite.config.*). A 1-2 file component
    edit carries neither -- its index.html and package.json already live in
    the repo and simply aren't part of the diff -- so the Vite-entry-chain
    veto must not fire on it.
    """
    if len(paths) >= 3:
        return True
    names = {_basename(p) for p in paths}
    return bool(names & _SCAFFOLD_MARKERS)


class ReactSkill(Skill):
    name = "react"
    role = "frontend"
    aliases = ("React", "react.js", "ReactJS")
    description = "Vite + React frontend: functional components, hooks, JSX/TSX."

    def detect(self, goal: str) -> float:
        g = goal or ""
        # React Native is a distinct ecosystem -- don't fire on it.
        if _REACT_NATIVE_RE.search(g):
            return 0.0
        if _REACT_RE.search(g):
            return 0.95
        if _JSX_RE.search(g):
            return 0.6
        if _REACT_TYPO_RE.search(g):
            return 0.6
        return 0.0

    def scaffold_system_prompt(self) -> str:
        return (
            "FRONTEND -- React project (React 19 + Vite)\n"
            "- Layout (relative to the frontend component root): index.html (Vite "
            "entry: viewport meta, a specific <title>, `<div id=\"root\"></div>`, "
            "`<script type=\"module\" src=\"/src/main.jsx\"></script>`); "
            "src/main.jsx mounts the app; src/App.jsx is the app shell (header/nav "
            "+ routes); src/pages/<Screen>.jsx holds ONE file per screen; "
            "src/components/<Piece>.jsx holds shared UI pieces; src/lib/*.js holds "
            "non-UI modules (API client, storage helpers); src/styles/*.css holds "
            "the stylesheets. No webpack/babel config files.\n"
            "- package.json: `react` + `react-dom` ^19 under dependencies "
            "(`react-router` ^7 too when the app has more than one screen); "
            "`vite` + `@vitejs/plugin-react` under devDependencies, plus `vitest`, "
            "`jsdom` and `@testing-library/react` when tests are planned; "
            "`\"type\": \"module\"`; scripts dev/build/preview/test.\n"
            "- vite.config.js with the React plugin (add `test: { environment: "
            "'jsdom' }` when tests exist).\n"
            "- The entry MUST use `import { createRoot } from 'react-dom/client'` "
            "and `createRoot(document.getElementById('root')).render(<StrictMode>"
            "<App /></StrictMode>)`; wrap in `<BrowserRouter>` (from 'react-router') "
            "for multi-screen apps. `ReactDOM.render` was REMOVED in React 19 and "
            "renders a blank page -- never use it.\n"
            "- main.jsx imports the project stylesheet(s) (e.g. "
            "`import './styles/tokens.css'`) so styles actually load.\n"
            "- Functional components and hooks only (useState, useEffect, custom "
            "hooks); no class components. Style with className against the "
            "project stylesheet -- never ship unstyled markup.\n"
            "- Do NOT emit Python files for the UI layer."
        )

    def plan_system_prompt(self) -> str:
        return (
            "When modifying a React project:\n"
            "- Preserve hook ordering rules and component composition.\n"
            "- New components go under src/components/ as .jsx files.\n"
            "- Don't introduce class components into a hooks-based codebase."
        )

    def validate_plan(self, diffs: List[Dict[str, Any]],
                      goal: str = "") -> Optional[SkillVerdict]:
        """Veto a React plan missing the pieces a Vite build requires.

        A React plan with source files but no ``index.html`` (Vite's build
        entry) or no ``package.json`` cannot build -- the exact failure that
        slipped through before. Returning a fatal verdict re-asks the planner
        with this reason instead of generating an unbuildable tree.

        This only applies to a *fresh scaffold*. An incremental edit that
        touches one or two component files legitimately omits index.html and
        package.json (they already exist in the repo); vetoing it was a false
        positive, so :func:`_looks_like_full_scaffold` gates the check.
        """
        paths = file_paths(diffs)
        js = [p for p in paths
              if p.lower().endswith((".jsx", ".tsx", ".js", ".ts"))]
        if not js:
            return None  # no React source planned; not this skill's concern
        if not _looks_like_full_scaffold(paths):
            return None  # incremental edit -- entry chain already in the repo
        lower = [p.lower() for p in paths]
        missing: List[str] = []
        if not any(p.rsplit("/", 1)[-1] == "index.html" for p in lower):
            missing.append("an index.html entry (Vite's build entry, beside "
                           "the frontend src/)")
        if not any(p.rsplit("/", 1)[-1] == "package.json" for p in lower):
            missing.append("a package.json")
        if missing:
            return SkillVerdict(
                passed=False, confidence=0.9,
                rationale=("React skill: the plan has React source but is "
                           "missing " + " and ".join(missing)
                           + "; a Vite build needs them."))
        return None

    def validate_scaffold(self, diffs: List[Dict[str, Any]],
                          goal: str = "") -> Optional[SkillVerdict]:
        paths = file_paths(diffs)
        if not paths:
            return None
        js_exts = (".jsx", ".tsx", ".js", ".ts")
        if not has_any_ext(paths, js_exts):
            return SkillVerdict(
                passed=False, confidence=0.9,
                rationale=("React skill: scaffold has no .jsx/.tsx/.js/.ts "
                           "files. Regenerate with src/App.jsx + "
                           "src/main.jsx + package.json."),
            )
        non_meta = [p for p in paths
                    if not p.lower().endswith((".md", ".txt", ".cfg", ".ini",
                                               ".toml", ".yml", ".yaml",
                                               ".json", ".lock"))]
        if non_meta and all(p.lower().endswith(".py") for p in non_meta):
            return SkillVerdict(
                passed=False, confidence=0.9,
                rationale=("React skill: every source file is Python -- the "
                           "scaffold ignored the React requirement."),
            )
        bad = self._jsx_in_plain_js(diffs)
        if bad:
            return SkillVerdict(
                passed=False, confidence=0.9,
                rationale=("React skill: " + bad + " imports react and "
                           "contains JSX but is a .js/.ts file -- Vite/esbuild "
                           "does not transform JSX there. Rename it to "
                           ".jsx/.tsx and update its imports/index.html."),
            )
        dangling = self._dangling_html_entry(diffs)
        if dangling:
            return SkillVerdict(
                passed=False, confidence=0.9,
                rationale=("React skill: index.html loads "
                           "<script src=\"" + dangling + "\"> but no generated "
                           "file provides that entry. Emit the referenced entry "
                           "(e.g. src/main.jsx) or fix the script src path."),
            )
        return None

    # ---- validation helpers ------------------------------------------
    @staticmethod
    def _jsx_in_plain_js(diffs: List[Dict[str, Any]]) -> Optional[str]:
        """Path of a .js/.ts file that both imports react and shows JSX.

        High precision: the SAME diff body must match the react import AND a
        JSX marker (checked per-file by feeding a one-item list to
        :func:`any_body_matches`), and the path must be plain .js/.ts, never
        the sanctioned .jsx/.tsx.
        """
        for d in diffs or []:
            ps = file_paths([d])
            if not ps:
                continue
            pl = ps[0].lower()
            if not pl.endswith(_PLAIN_JS_EXTS):
                continue
            one = [d]
            if (any_body_matches(one, _REACT_MODULE_IMPORT_RE,
                                 only_ext=_PLAIN_JS_EXTS)
                    and any_body_matches(one, _JSX_BODY_RE,
                                         only_ext=_PLAIN_JS_EXTS)):
                return ps[0]
        return None

    @staticmethod
    def _dangling_html_entry(diffs: List[Dict[str, Any]]) -> Optional[str]:
        """The src of an index.html module script whose JS entry is missing.

        Returns ``None`` (no opinion) when the diff carries no index.html, when
        the referenced src is external/non-JS, or when a generated file
        provides that entry basename. Only a truly dangling entry -- the script
        names a JS entry no generated file supplies -- is reported, so the fatal
        verdict cannot fire on a legitimate scaffold.
        """
        html = body_of(diffs, "index.html")
        if not html:
            return None
        basenames = {_basename(p) for p in file_paths(diffs)}
        for m in _HTML_SCRIPT_SRC_RE.finditer(html):
            src = m.group(1).strip()
            if "://" in src or src.startswith("//"):
                continue  # external/CDN script, not the app entry
            if not src.lower().endswith((".js", ".jsx", ".ts", ".tsx")):
                continue  # not a JS module entry
            ref = _basename(src)
            if ref and ref not in basenames:
                return src
        return None

    # ---- deterministic repairs (Layer-1, no model call) ----------------
    def repair_scaffold(self, files: List[Dict[str, Any]]) -> Dict[str, str]:
        """Fix the mechanical frontend defects a weak model ships every run.

        * ``react-router-dom`` imports when ``package.json`` depends on
          ``react-router`` (v7 is one package) -- the build cannot resolve the
          module otherwise;
        * a relative stylesheet import (``./Chat.css``) whose file was never
          generated -- the bundle aborts on it; the shared stylesheets carry
          the styles, so the dangling import is dropped;
        * a React / React Router API (``StrictMode``, ``useState`` ...) used
          but never imported -- a ReferenceError and a blank page.
        Pure text rewrites, each certain; returns ``{path: new_content}``.
        """
        from cgx.session import js_symbols as _js
        bundle = {str(f.get("path") or f.get("file") or "").replace("\\", "/"):
                  str(f.get("content") or f.get("patch") or "") for f in (files or [])}
        bundle = {p: c for p, c in bundle.items() if p}
        deps: set = set()
        for p, c in bundle.items():
            if p.rsplit("/", 1)[-1] == "package.json":
                try:
                    import json as _json
                    pkg = _json.loads(c)
                    deps |= set((pkg.get("dependencies") or {}).keys())
                    deps |= set((pkg.get("devDependencies") or {}).keys())
                except Exception:
                    continue
        fixes: Dict[str, str] = {}
        for path, content in bundle.items():
            if not _js.is_js_path(path):
                continue
            new = content
            if ("react-router" in deps and "react-router-dom" not in deps
                    and "react-router-dom" in new):
                new = new.replace("'react-router-dom'", "'react-router'").replace(
                    '"react-router-dom"', '"react-router"')
            new = _strip_dangling_stylesheet_imports(new, path, bundle)
            new = _add_missing_react_imports(new, path)
            new = _fix_namespace_calls(new, path, bundle)
            if new != content:
                fixes[path] = new
        return fixes

    def scaffold_warnings(self, diffs: List[Dict[str, Any]],
                          goal: str = "") -> List[SkillVerdict]:
        paths = file_paths(diffs)
        if not paths or not has_any_ext(paths, (".jsx", ".tsx", ".js", ".ts")):
            return []
        if has_js_test_file(paths):
            return []
        return [SkillVerdict(
            passed=False, confidence=0.7, severity="warning",
            rationale=("React skill: no test file generated. Add a "
                       "tests/<Component>.test.jsx that exercises the "
                       "primary component with @testing-library/react."),
        )]


_STYLE_IMPORT_LINE_RE = re.compile(
    r"""^\s*import\s+['"](\.[^'"]+\.(?:css|scss|sass|less|styl))['"]\s*;?\s*$\n?""", re.M)


def _strip_dangling_stylesheet_imports(content: str, path: str,
                                       bundle: Dict[str, str]) -> str:
    """Drop ``import './x.css'`` lines whose stylesheet is not in the bundle."""
    import posixpath

    def _keep(m: "re.Match[str]") -> str:
        spec = m.group(1)
        target = posixpath.normpath(posixpath.join(posixpath.dirname(path), spec)).lstrip("./")
        return m.group(0) if target in bundle else ""

    return _STYLE_IMPORT_LINE_RE.sub(_keep, content)


def _add_missing_react_imports(content: str, path: str) -> str:
    """Import React / router APIs the file uses but never imported."""
    from cgx.session import js_symbols as _js
    if not path.lower().endswith((".jsx", ".tsx", ".js", ".ts")):
        return content
    declared = _js._declared_names(content)
    body = _js._strip_comments(content)
    needed: Dict[str, List[str]] = {}
    for name, module in _js.REACT_API.items():
        if name in declared:
            continue
        if re.search(r"<" + name + r"\b", body) or re.search(r"\b" + name + r"\s*\(", body):
            needed.setdefault(module, []).append(name)
    if not needed:
        return content
    out = content
    for module, names in needed.items():
        line_re = re.compile(r"""^(\s*import\s+)([^;'"\n]*?)(\s*from\s*['"]""" + re.escape(module) + r"""['"]\s*;?)""", re.M)
        m = line_re.search(out)
        if m and "{" in m.group(2):
            clause = m.group(2)
            inner_start, inner_end = clause.index("{") + 1, clause.rindex("}")
            inner = clause[inner_start:inner_end].strip()
            merged = ", ".join([x for x in [inner] if x] + names)
            new_clause = clause[:inner_start] + " " + merged + " " + clause[inner_end:]
            out = out[:m.start(2)] + new_clause + out[m.end(2):]
        elif m:
            new_clause = m.group(2).strip() + ", { " + ", ".join(names) + " }"
            out = out[:m.start(2)] + new_clause + out[m.end(2):]
        else:
            out = f"import {{ {', '.join(names)} }} from '{module}';\n" + out
    return out


_IMPORT_LINE_RE = re.compile(r"^\s*import\b[^;\n]*(?:;|$)", re.M)


def _fix_namespace_calls(content: str, path: str, bundle: Dict[str, str]) -> str:
    """Rewrite ``storage.loadSettings()`` (no ``storage`` in scope) into a named
    import + direct call when a sibling module named ``storage`` exports it.

    The single most repeated weak-model habit on the frontend: calling helpers
    through a module-like namespace that was never imported (a ReferenceError
    and a blank page). Only rewrites when EVERY call on that namespace resolves
    to a named export of exactly one bundle module whose basename matches, so
    the rewrite is certain; anything else is left to the symbol gate + model.
    """
    from cgx.session import js_symbols as _js
    import posixpath
    body = _js._strip_comments(content)
    declared = _js._declared_names(content)
    uses: Dict[str, set] = {}
    for m in re.finditer(r"(?<![\w$.])([a-z_$][\w$]*)\.([A-Za-z_$][\w$]*)\s*\(", body):
        ns, fn = m.group(1), m.group(2)
        if ns in declared or ns in _js._GLOBALS or len(ns) < 2:
            continue
        uses.setdefault(ns, set()).add(fn)
    if not uses:
        return content
    out = content
    for ns, fns in uses.items():
        targets = [p for p in bundle if _js.is_js_path(p)
                   and posixpath.splitext(posixpath.basename(p))[0] == ns and p != path]
        if len(targets) != 1:
            continue
        target = targets[0]
        exports = set(_js.js_exports(bundle[target])["named"])
        if not fns <= exports:
            continue
        names = sorted(fns)
        out = re.sub(r"(?<![\w$.])" + re.escape(ns) + r"\.(" + "|".join(map(re.escape, names)) + r")\s*\(",
                     r"\1(", out)
        spec = _js._rel_spec(path, target)
        # merge into an existing import of the same module, else add one
        merged = False
        for stmt in _js.js_imports(out):
            if stmt.side_effect or _js.resolve_relative(stmt.spec, path, list(bundle)) != target:
                continue
            have = {i for i, _ in stmt.named}
            missing = [n for n in names if n not in have]
            if not missing:
                merged = True
                break
            pat = re.compile(r"(import\s+(?:[^;'\"]*?)\{)([^}]*)(\}\s*from\s*['\"]" + re.escape(stmt.spec) + r"['\"])")
            m = pat.search(out)
            if m:
                inner = m.group(2).strip()
                new_inner = ", ".join([x for x in [inner] if x] + missing)
                out = out[:m.start(2)] + " " + new_inner + " " + out[m.end(2):]
                merged = True
            break
        if not merged:
            line = f"import {{ {', '.join(names)} }} from '{spec}';\n"
            imports = list(_IMPORT_LINE_RE.finditer(out))
            if imports:
                last = imports[-1]
                out = out[:last.end()] + "\n" + line.rstrip("\n") + out[last.end():]
            else:
                out = line + out
    return out


__all__ = ["ReactSkill"]
