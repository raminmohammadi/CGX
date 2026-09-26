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
    description = ("Static HTML/CSS/JS website -- no build step, but CDN "
                   "libraries welcome (Tailwind, Alpine, fonts). Renders "
                   "instantly, deploys anywhere.")

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
            "STATIC WEBSITE -- no build step (no bundler/npm/package.json).\n"
            "IMPORTANT: ignore any earlier instruction to use Vite, a bundler, "
            "a `/src/main.jsx` entry, OR to split the app into 'core_logic'/"
            "'auth'/'api' layers or to add unit-test files. This is a "
            "front-end website, not a backend app: there is NO auth, NO server "
            "logic, and NO test files -- do not plan or generate any.\n"
            "CONTENT & SCOPE -- build a REAL, substantial site, not a stub. A "
            "landing page must be a rich single `index.html` with several full "
            "sections in this order: a header/nav, a compelling HERO (headline "
            "+ subtext + call-to-action), 2-4 content sections appropriate to "
            "the topic (e.g. for a coffee shop: featured menu with items + "
            "prices, about/story, opening hours, gallery, a location/contact "
            "block), and a footer -- with real, specific copy (not 'Lorem "
            "ipsum' and not a single bare form). Only build separate pages when "
            "the request clearly needs them.\n"
            "- `index.html` MUST exist at the project root and be the home page. "
            "Each page is a complete document: <!DOCTYPE html>, "
            "<html lang=\"en\">, <head> with <meta charset> + responsive "
            "<meta name=\"viewport\"> + a descriptive <title>, and <body>.\n"
            "STYLING & LIBRARIES -- follow the styling approach named in the "
            "goal. You MAY use CDN-delivered libraries via <link>/<script> tags "
            "(they need no build): e.g. Tailwind CSS Play CDN "
            "(https://cdn.tailwindcss.com), Alpine.js, htmx, and Google Fonts. "
            "Do NOT emit files that imply a build: no package.json, no "
            "vite/webpack config, no `.jsx`/`.tsx`/`.vue` single-file "
            "components, no test files. Prefer one shared `css/style.css` for "
            "hand-written CSS; with Tailwind, keep only small custom overrides "
            "there.\n"
            "INTERACTIVITY -- put custom scripts in `js/script.js` "
            "(`<script src=\"js/script.js\" defer></script>`), or use Alpine/"
            "htmx attributes inline. Plain ES in the browser -- no bundler "
            "imports.\n"
            "SHARED HEADER/FOOTER (avoid copy-paste drift):\n"
            "- For a SMALL site, prefer a SINGLE page (index.html) with in-page "
            "sections and a nav that links to `#section` anchors -- then there "
            "is no header to duplicate.\n"
            "- For a genuinely MULTI-PAGE site, define the header+footer markup "
            "ONCE in `js/layout.js` (as a string) and inject it into "
            "`<header id=\"site-header\"></header>` / "
            "`<footer id=\"site-footer\"></footer>` placeholders that every page "
            "includes (load `js/layout.js` with `defer`). Editing the nav then "
            "means editing one file, and it works when opening files directly. "
            "Do NOT hand-duplicate the same header markup across pages.\n"
            "LINKS -- inter-page and asset links use RELATIVE paths "
            "(href=\"about.html\", href=\"css/style.css\"); every local href/src "
            "you write MUST point at a file that is part of this site.\n"
            "IMAGERY -- you CANNOT create real photo/raster files, so NEVER "
            "reference a local `.jpg`/`.jpeg`/`.png`/`.gif`/`.webp` file (it "
            "would 404). Instead use: inline `<svg>` illustrations/icons, CSS "
            "gradients/patterns, emoji, or a REMOTE image URL "
            "(https://images.unsplash.com/... , https://picsum.photos/... , "
            "https://placehold.co/...). A standalone `.svg` file you write the "
            "markup for is fine.\n"
            "QUALITY -- semantic, accessible markup (header/nav/main/section/"
            "footer, alt text, labelled controls), responsive (mobile-first, "
            "flexbox/grid). A contact form with no backend should either post "
            "to a named form service (e.g. Formspree) or be clearly marked as "
            "a front-end mockup."
        )

    def scaffold_system_prompt(self) -> str:
        return self._layout_rules()

    def plan_system_prompt(self) -> str:
        return (
            "STATIC SITE plan rules (no build step):\n"
            "- Plan `index.html` at the root. For a small site prefer a single "
            "page with in-page sections; for a multi-page site plan one .html "
            "per page plus `js/layout.js` for the shared header/footer.\n"
            "- Plan a single `css/style.css`; `js/script.js` for interactivity.\n"
            "- Do NOT plan package.json, a bundler config, or test files. CDN "
            "libraries (Tailwind/Alpine/fonts) are fine and add no files.\n"
            "- When adding a page, wire it into the shared nav (in js/layout.js "
            "for multi-page sites)."
        )

    def ask_system_prompt(self) -> str:
        return (
            "This project is a static site with NO build step (edit the "
            ".html/.css/.js directly; CDN libraries like Tailwind/Alpine are "
            "fine, but never add a bundler/package.json). Use relative links, "
            "and if the site shares a header/footer via js/layout.js, change "
            "it there rather than per page."
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
        # Broken links are advisory, NOT fatal: SCAFFOLD can only regenerate
        # files the manifest already lists, so a reference to a file the plan
        # omitted is unfixable by regenerate -- a fatal verdict would just
        # death-spiral the build into FAILED. Surface it as a warning (and the
        # on-disk StaticSiteRunner still gives VERIFY a real signal).
        broken = self._broken_links(diffs)
        if broken:
            return SkillVerdict(
                passed=False, confidence=0.85, severity="warning",
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
