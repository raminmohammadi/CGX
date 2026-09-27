

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
            "FRONTEND -- React project\n"
            "- Use a modern Vite-style layout: src/main.jsx mounts the app, "
            "src/App.jsx is the root component, src/components/*.jsx for "
            "individual UI pieces. No webpack/babel config files.\n"
            "- index.html at the project root with a single "
            "`<div id=\"root\"></div>` and "
            "`<script type=\"module\" src=\"/src/main.jsx\"></script>`.\n"
            "- package.json must list `react` and `react-dom` (^18) under "
            "dependencies and `vite` + `@vitejs/plugin-react` under "
            "devDependencies. Include `scripts.dev`, `scripts.build`, "
            "`scripts.preview`.\n"
            "- vite.config.js with the React plugin.\n"
            "- Use functional components and hooks (useState, useEffect). "
            "No class components.\n"
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


__all__ = ["ReactSkill"]
