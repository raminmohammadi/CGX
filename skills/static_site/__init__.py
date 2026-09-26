"""Static HTML/CSS/JS website skill (no framework, no build step).

Detects goals asking for a plain multi-page website -- a landing page,
portfolio, marketing/brochure site, docs page -- and steers the generator
AWAY from the default Vite/React assumption baked into the base scaffold
prompts toward a coherent, build-less static site: ``index.html`` + one
file per page, a single shared stylesheet, one vanilla-JS file, a header/
footer repeated identically across pages, and relative links that only
point at files the plan actually emits.

Its validators enforce exactly that contract: an ``index.html`` must
exist, no framework/build files may leak in, and every local
``href``/``src``/``link`` reference must resolve to an emitted file
(broken links are the number-one way a "finished" static site is actually
broken). It also owns the ``ask``/``plan`` surfaces so the chatbot and the
change-planner give static-site-appropriate guidance.
"""

from __future__ import annotations

import posixpath
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from skills.base import Skill, SkillVerdict, file_paths

# --- detection -------------------------------------------------------------

# Explicit "plain/static HTML site" signals.
_STATIC_RE = re.compile(
    r"\b(static\s+(?:site|website|web\s?page|html)|"
    r"plain\s+html|vanilla\s+(?:html|js|javascript)|"
    r"no[\s-]*framework|html\s*/\s*css\s*/\s*js|html\s+css\s+js|"
    r"multi[\s-]*page\s+(?:site|website)|landing\s+page|"
    r"brochure\s+site|marketing\s+site|portfolio\s+(?:site|website|page))\b",
    re.IGNORECASE,
)
# Weaker signal: "html website / html site / website in html".
_HTML_SITE_RE = re.compile(
    r"\bhtml\b.{0,20}\b(?:site|website|page)\b|"
    r"\b(?:site|website|page)\b.{0,20}\bhtml\b",
    re.IGNORECASE,
)
# Frameworks that own the frontend instead -- never fire against these.
_FRAMEWORK_RE = re.compile(
    r"\b(react|next\.?js|nextjs|vue|nuxt|svelte(?:kit)?|angular|"
    r"solid\.?js|preact|remix|astro|gatsby)\b",
    re.IGNORECASE,
)

_FRAMEWORK_FILES = ("package.json", "vite.config.js", "vite.config.ts",
                    "webpack.config.js", "next.config.js", "tsconfig.json")
_FRAMEWORK_EXTS = (".jsx", ".tsx", ".vue", ".svelte", ".ts")

# Local reference extraction from generated HTML/CSS.
_HREF_SRC_RE = re.compile(
    r"""(?:href|src)\s*=\s*["']([^"'#?]+)["']""", re.IGNORECASE)
_CSS_URL_RE = re.compile(r"""url\(\s*["']?([^"')#?]+)["']?\s*\)""", re.IGNORECASE)


def _is_external(ref: str) -> bool:
    r = ref.strip().lower()
    return (
        not r
        or r.startswith(("http://", "https://", "//", "data:", "mailto:",
                         "tel:", "javascript:", "#"))
    )


class StaticSiteSkill(Skill):
    name = "static_site"
    role = "frontend"
    aliases = ("static-site", "html-site", "static html", "vanilla web")
    description = ("Plain multi-page HTML/CSS/JS website -- no framework, no "
                   "build step, shared header/footer, relative links.")

    def detect(self, goal: str) -> float:
        g = goal or ""
        if _FRAMEWORK_RE.search(g):
            return 0.0  # a named framework owns this build
        if _STATIC_RE.search(g):
            return 0.95
        if _HTML_SITE_RE.search(g):
            return 0.9
        return 0.0

    # --- prompt composition ------------------------------------------------
    def _layout_rules(self) -> str:
        return (
            "STATIC HTML SITE -- no framework, no build step.\n"
            "IMPORTANT: ignore any earlier instruction to use Vite, React, a "
            "bundler, or a `/src/main.jsx` entry. This is a plain, build-less "
            "website served by opening the files directly.\n"
            "- `index.html` MUST exist at the project root and be the home page.\n"
            "- One `.html` file per page at the root (e.g. index.html, "
            "about.html, contact.html). Do NOT create a build tool, "
            "package.json, node_modules, or any test file.\n"
            "- Exactly one shared stylesheet at `css/style.css`; every page "
            "links it with `<link rel=\"stylesheet\" href=\"css/style.css\">`.\n"
            "- Put any interactivity in one `js/script.js`, loaded with "
            "`<script src=\"js/script.js\" defer></script>`. Vanilla DOM APIs "
            "only -- no imports, no framework, no CDN framework tags.\n"
            "- Every page shares an IDENTICAL <header> (with a <nav>) and "
            "<footer>. The nav links to the other pages using RELATIVE paths "
            "(href=\"about.html\"), and every href/src you write MUST point at "
            "a file that is actually part of this site.\n"
            "- Each page is a complete document: <!DOCTYPE html>, "
            "<html lang=\"en\">, <head> with <meta charset> + responsive "
            "<meta name=\"viewport\"> + a descriptive <title>, and <body>.\n"
            "- Semantic, accessible markup (header/nav/main/section/footer, "
            "alt text on images, labelled controls). Responsive CSS "
            "(mobile-first, flexbox/grid, a max-width container).\n"
            "- Reference images from `assets/` with relative paths; if an "
            "image file is not generated, use a CSS background/gradient or an "
            "inline SVG placeholder rather than linking a missing file."
        )

    def scaffold_system_prompt(self) -> str:
        return self._layout_rules()

    def plan_system_prompt(self) -> str:
        return (
            "STATIC HTML SITE plan rules:\n"
            "- Plan `index.html` at the root plus one .html file per page, a "
            "single `css/style.css`, and `js/script.js` if any interactivity "
            "is needed.\n"
            "- Do NOT plan package.json, a bundler config, or test files.\n"
            "- When adding a page, also add its nav link to every existing "
            "page's shared header."
        )

    def ask_system_prompt(self) -> str:
        return (
            "This project is a plain static HTML/CSS/JS site (no framework/"
            "build). When explaining or changing it, keep it build-less: edit "
            "the .html/.css/.js files directly, keep the header/footer "
            "identical across pages, and use relative links."
        )

    # --- validation --------------------------------------------------------
    def _emitted_and_refs(
        self, diffs: List[Dict[str, Any]]
    ) -> Tuple[Set[str], List[Tuple[str, str]]]:
        """Return (emitted normalized paths, [(referencing_file, ref)])."""
        emitted: Set[str] = {
            posixpath.normpath(p).lstrip("./") for p in file_paths(diffs)
        }
        refs: List[Tuple[str, str]] = []
        for d in diffs or []:
            if not isinstance(d, dict):
                continue
            path = str(d.get("file") or d.get("path") or "")
            low = path.lower()
            body = str(d.get("patch") or d.get("diff") or d.get("content") or "")
            if low.endswith((".html", ".htm")):
                found = _HREF_SRC_RE.findall(body)
            elif low.endswith(".css"):
                found = _CSS_URL_RE.findall(body)
            else:
                continue
            for ref in found:
                if not _is_external(ref):
                    refs.append((path, ref))
        return emitted, refs

    def _broken_links(self, diffs: List[Dict[str, Any]]) -> List[str]:
        emitted, refs = self._emitted_and_refs(diffs)
        broken: List[str] = []
        for src_file, ref in refs:
            base = posixpath.dirname(src_file)
            target = posixpath.normpath(posixpath.join(base, ref)).lstrip("./")
            if target and target not in emitted and f"{target}" not in broken:
                broken.append(f"{ref} (referenced by {src_file})")
        return broken

    def validate_plan(self, diffs: List[Dict[str, Any]],
                      goal: str = "") -> Optional[SkillVerdict]:
        paths = [p.lower() for p in file_paths(diffs)]
        if not paths:
            return None
        if not any(p.rsplit("/", 1)[-1] == "index.html" for p in paths):
            return SkillVerdict(
                passed=False, confidence=0.9,
                rationale=("static_site skill: the plan has no index.html at "
                           "the root -- a static site needs a home page."))
        leaked = [p for p in paths
                  if p.rsplit("/", 1)[-1] in _FRAMEWORK_FILES
                  or p.endswith(_FRAMEWORK_EXTS)]
        if leaked:
            return SkillVerdict(
                passed=False, confidence=0.85,
                rationale=("static_site skill: this is a build-less static "
                           "site; drop framework/build files: "
                           + ", ".join(sorted(set(leaked)))))
        return None

    def validate_scaffold(self, diffs: List[Dict[str, Any]],
                          goal: str = "") -> Optional[SkillVerdict]:
        paths = file_paths(diffs)
        if not paths:
            return None
        lower = [p.lower() for p in paths]
        if not any(p.rsplit("/", 1)[-1] == "index.html" for p in lower):
            return SkillVerdict(
                passed=False, confidence=0.9,
                rationale=("static_site skill: no index.html at the project "
                           "root. Regenerate a build-less site with index.html "
                           "as the home page."))
        leaked = [p for p in lower
                  if p.rsplit("/", 1)[-1] in _FRAMEWORK_FILES
                  or p.endswith(_FRAMEWORK_EXTS)]
        if leaked:
            return SkillVerdict(
                passed=False, confidence=0.85,
                rationale=("static_site skill: framework/build files leaked "
                           "into a plain static site -- remove: "
                           + ", ".join(sorted(set(leaked)))))
        broken = self._broken_links(diffs)
        if broken:
            return SkillVerdict(
                passed=False, confidence=0.85,
                rationale=("static_site skill: broken local references -- "
                           "these point at files the site does not include: "
                           + "; ".join(broken[:8])))
        return None

    def scaffold_warnings(self, diffs: List[Dict[str, Any]],
                          goal: str = "") -> List[SkillVerdict]:
        paths = [p.lower() for p in file_paths(diffs)]
        if not any(p.endswith((".html", ".htm")) for p in paths):
            return []
        warns: List[SkillVerdict] = []
        if not any(p.rsplit("/", 1)[-1] == "style.css" or p.endswith(".css")
                   for p in paths):
            warns.append(SkillVerdict(
                passed=False, confidence=0.6, severity="warning",
                rationale=("static_site skill: no stylesheet was generated -- "
                           "the site will be unstyled.")))
        return warns
