

"""Next.js fullstack skill (App Router preferred).

Detects ``next.js``/``nextjs`` mentions, supplies an App Router scaffold
prompt, and validates that the output contains route files plus a
package.json declaring ``next`` as a dependency.

The scaffold validator additionally enforces the App Router server/client
boundary: a file under ``app/`` that reaches for client-only React or browser
APIs (``useState``/``useEffect``/``useRef``, JSX event handlers, ``window``/
``document``) without a leading ``'use client'`` directive is a Server
Component that cannot possibly render -- the single most common Next.js codegen
failure -- so it is rejected with an actionable message.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from skills.base import (
    Skill,
    SkillVerdict,
    file_paths,
    has_any_ext,
    has_js_test_file,
)

_NEXT_RE = re.compile(r"\bnext\.?js\b|\bnextjs\b", re.IGNORECASE)

# Root-level Next.js config filenames (any one satisfies the "next config" need).
_NEXT_CONFIG_NAMES = (
    "next.config.js", "next.config.mjs", "next.config.cjs", "next.config.ts",
)

# Client-only React hooks (CALLED, not merely imported) and JSX event-handler
# props. In the App Router every module under app/ is a Server Component by
# default; any of these forces a Client Component and REQUIRES a leading
# 'use client' directive. There is no isomorphic escape hatch for these, so
# matching one without the directive is a high-precision failure.
_HOOK_HANDLER_RE = re.compile(
    r"\buse(?:State|Effect|LayoutEffect|InsertionEffect|Ref|Reducer|Context|"
    r"Callback|Memo|ImperativeHandle|Transition|DeferredValue|"
    r"SyncExternalStore)\s*\("
    r"|\bon[A-Z][A-Za-z]*=\{"          # JSX event prop, e.g. onClick={...}
)
# Browser globals. Weaker signal than hooks: a module may legitimately touch
# these behind a `typeof window`-style guard and stay isomorphic, so this only
# fires when no such guard is present.
_BROWSER_GLOBAL_RE = re.compile(
    r"\bwindow\.|\bdocument\.|\blocalStorage\b|\bsessionStorage\b|\bnavigator\.")
_ISOMORPHIC_GUARD_RE = re.compile(r"typeof\s+(?:window|document|globalThis)")


def _diff_body(d: Dict[str, Any]) -> str:
    """Raw textual body of a diff row, across the shapes CGX uses."""
    if not isinstance(d, dict):
        return ""
    return str(d.get("patch") or d.get("diff") or d.get("content")
               or d.get("new_content") or "")


def _diff_path(d: Dict[str, Any]) -> str:
    if not isinstance(d, dict):
        return ""
    return str(d.get("file") or d.get("path") or "").replace("\\", "/")


def _decomment(body: str) -> str:
    """Best-effort strip of ``/* ... */`` and ``//`` comments.

    Used so the server/client checks look at real code, not a license header
    or an eslint pragma. Over-stripping (e.g. a ``//`` inside a URL string)
    only ever hides a signal -- a false negative, which is the safe direction
    for a high-precision fatal verdict.
    """
    no_block = re.sub(r"/\*.*?\*/", "", body or "", flags=re.S)
    out: List[str] = []
    for raw in no_block.splitlines():
        idx = raw.find("//")
        out.append(raw if idx == -1 else raw[:idx])
    return "\n".join(out)


def _first_line_is_use_client(body: str) -> bool:
    """True when the first non-blank line of ``body`` is a 'use client' directive.

    ``body`` is expected to be comment-stripped already. Accepts single/double
    quotes or backticks and an optional trailing semicolon.
    """
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        norm = line.rstrip(";").strip().strip("'\"`").strip()
        return norm.lower() == "use client"
    return False


class NextJsSkill(Skill):
    name = "nextjs"
    role = "fullstack"
    aliases = ("Next.js", "NextJS", "next")
    description = "Next.js App Router fullstack framework (React + server routes)."

    def detect(self, goal: str) -> float:
        if _NEXT_RE.search(goal or ""):
            return 0.95
        return 0.0

    def scaffold_system_prompt(self) -> str:
        return (
            "FULLSTACK -- Next.js project\n"
            "ROUTER CHOICE (pick ONE, never mix the two in one tree):\n"
            "- App Router (PREFERRED for new work): routes live under app/. "
            "app/layout.tsx (or .jsx) is a REQUIRED root layout that renders "
            "<html> and <body> and wraps every route; app/page.tsx is the "
            "index page; nested folders are nested routes; app/<route>/page.tsx "
            "is a route's page.\n"
            "- Pages Router (legacy alternative): routes are files under "
            "pages/ (pages/index.tsx, pages/about.tsx); pages/_app.tsx and "
            "pages/_document.tsx are the shells. Do NOT also create an app/ "
            "tree.\n"
            "SERVER vs CLIENT COMPONENTS (App Router -- the #1 Next.js codegen "
            "bug):\n"
            "- Every file under app/ is a SERVER Component by default: it may "
            "be async and read data/DB directly, but it may NOT call React "
            "hooks (useState/useEffect/useRef/useReducer/useContext/...), "
            "attach event handlers (onClick/onChange/...), or touch browser "
            "globals (window/document/localStorage).\n"
            "- A component that needs ANY of those MUST be a Client Component: "
            "put the exact directive `'use client'` as the VERY FIRST line of "
            "the file, above every import. Without it the build fails with "
            "\"You're importing a component that needs useState ... but none of "
            "its parents are marked with 'use client'\".\n"
            "- Keep client components small (a leaf that owns the interactive "
            "bit); fetch data in the server component and pass it down as "
            "props. In the Pages Router components are client by default and "
            "the 'use client' directive is NOT used.\n"
            "- API routes live under app/api/<route>/route.ts as named "
            "GET/POST exports (server-only -- never add 'use client' there).\n"
            "- package.json must list `next` (^14), `react` and `react-dom` "
            "(^18) under dependencies. Scripts: `dev` -> `next dev`, "
            "`build` -> `next build`, `start` -> `next start`.\n"
            "- next.config.js (or .mjs) at project root, even if empty.\n"
            "- tsconfig.json when generating TypeScript variants.\n"
            "- Do NOT add webpack config -- Next.js owns bundling.\n"
            "- Do NOT add a separate Vite/CRA setup."
        )

    def plan_system_prompt(self) -> str:
        return (
            "When modifying a Next.js project:\n"
            "- New pages go under app/<route>/page.tsx (App Router) or "
            "pages/<route>.tsx (Pages Router) -- match whichever the project "
            "already uses.\n"
            "- App Router needs a root app/layout.tsx (renders <html>/<body>); "
            "keep it if it exists, add it if an app/ page is introduced.\n"
            "- Server components by default; add 'use client' as the file's "
            "FIRST line (above imports) whenever hooks, event handlers, or "
            "browser APIs are needed."
        )

    def validate_scaffold(self, diffs: List[Dict[str, Any]],
                          goal: str = "") -> Optional[SkillVerdict]:
        paths = file_paths(diffs)
        if not paths:
            return None
        has_router = any(
            (p.startswith(("app/", "src/app/")) and (
                p.endswith("page.tsx") or p.endswith("page.jsx")
                or p.endswith("layout.tsx") or p.endswith("layout.jsx")
                or p.endswith("route.ts") or p.endswith("route.js")))
            or p.startswith(("pages/", "src/pages/"))
            for p in paths
        )
        if not has_router and not has_any_ext(paths, (".tsx", ".jsx")):
            return SkillVerdict(
                passed=False, confidence=0.85,
                rationale=("Next.js skill: scaffold has no app/ or pages/ "
                           "route files. Regenerate using App Router layout "
                           "(app/page.tsx + app/layout.tsx)."),
            )
        if not any(p.endswith("package.json") for p in paths):
            return SkillVerdict(
                passed=False, confidence=0.85,
                rationale=("Next.js skill: scaffold is missing package.json "
                           "with `next` dependency."),
            )
        # Server/client boundary: an app/ Server Component using client-only
        # features without a leading 'use client' cannot render.
        boundary = self._client_boundary_violation(diffs)
        if boundary is not None:
            return boundary
        return None

    def _client_boundary_violation(
            self, diffs: List[Dict[str, Any]]) -> Optional[SkillVerdict]:
        for d in diffs or []:
            p = _diff_path(d)
            if not (p.startswith("app/") or p.startswith("src/app/")):
                continue
            if not p.lower().endswith((".tsx", ".jsx", ".ts", ".js")):
                continue
            base = p.rsplit("/", 1)[-1].lower()
            # API route handlers are server-only by contract; test files use
            # client APIs (render/hooks) legitimately without the directive.
            if base.startswith("route.") or ".test." in base or ".spec." in base:
                continue
            body = _decomment(_diff_body(d))
            if not body:
                continue
            feature = bool(_HOOK_HANDLER_RE.search(body))
            if not feature and _BROWSER_GLOBAL_RE.search(body) \
                    and not _ISOMORPHIC_GUARD_RE.search(body):
                feature = True
            if feature and not _first_line_is_use_client(body):
                return SkillVerdict(
                    passed=False, confidence=0.9,
                    rationale=(
                        f"Next.js skill: {p} uses client-only React/browser "
                        "features (a hook such as useState/useEffect/useRef, a "
                        "JSX event handler like onClick, or window/document) but "
                        "app/ files are Server Components by default. Add the "
                        "exact directive `'use client'` as the VERY FIRST line "
                        "of the file, above all imports, or the build fails."),
                )
        return None

    def validate_plan(self, diffs: List[Dict[str, Any]],
                      goal: str = "") -> Optional[SkillVerdict]:
        """Veto a Next.js plan missing the pieces a build requires (paths only).

        When the plan introduces an App Router page (app/**/page.*), Next.js
        mandates a root app/layout.* (it renders <html>/<body>). Either router
        also needs a package.json and a root next.config.* -- omitting them
        yields a tree that cannot install or build.
        """
        paths = [p.replace("\\", "/") for p in file_paths(diffs)]
        if not paths:
            return None

        def _base(p: str) -> str:
            return p.rsplit("/", 1)[-1].lower()

        js_ext = (".tsx", ".jsx", ".ts", ".js")
        app_pages = [
            p for p in paths
            if (p.startswith("app/") or p.startswith("src/app/"))
            and _base(p).startswith("page.") and p.lower().endswith(js_ext)
        ]
        router_src = app_pages or [
            p for p in paths
            if p.startswith(("app/", "src/app/", "pages/", "src/pages/"))
            and p.lower().endswith(js_ext)
        ]
        if not router_src:
            return None  # nothing Next.js-shaped is planned; abstain

        missing: List[str] = []
        if app_pages:
            root_layouts = {
                f"{root}/layout{ext}"
                for root in ("app", "src/app")
                for ext in js_ext
            }
            if not any(p in root_layouts for p in paths):
                missing.append("a root app/layout.tsx (the App Router requires "
                               "a root layout rendering <html> and <body>)")
        if not any(_base(p) == "package.json" for p in paths):
            missing.append("a package.json declaring `next`")
        if not any(_base(p) in _NEXT_CONFIG_NAMES for p in paths):
            missing.append("a next.config.js (or .mjs/.ts) at the project root")

        if missing:
            return SkillVerdict(
                passed=False, confidence=0.85,
                rationale=("Next.js skill: the plan has Next.js routes but is "
                           "missing " + "; ".join(missing) + "."),
            )
        return None

    def scaffold_warnings(self, diffs: List[Dict[str, Any]],
                          goal: str = "") -> List[SkillVerdict]:
        paths = file_paths(diffs)
        if not paths or not has_any_ext(paths, (".tsx", ".jsx", ".ts", ".js")):
            return []
        if has_js_test_file(paths):
            return []
        return [SkillVerdict(
            passed=False, confidence=0.7, severity="warning",
            rationale=("Next.js skill: no test file generated. Add a "
                       "tests/<page>.test.tsx using Jest + "
                       "@testing-library/react to render the route."),
        )]


__all__ = ["NextJsSkill"]
