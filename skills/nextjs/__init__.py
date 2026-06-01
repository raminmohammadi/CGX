"""Next.js fullstack skill (App Router preferred).

Detects ``next.js``/``nextjs`` mentions, supplies an App Router scaffold
prompt, and validates that the output contains route files plus a
package.json declaring ``next`` as a dependency.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from skills.base import (
    Skill, SkillVerdict, file_paths, has_any_ext, has_js_test_file,
)


_NEXT_RE = re.compile(r"\bnext\.?js\b|\bnextjs\b", re.IGNORECASE)


class NextJsSkill(Skill):
    name = "nextjs"
    role = "fullstack"
    aliases = ("Next.js", "NextJS", "next")

    def detect(self, goal: str) -> float:
        if _NEXT_RE.search(goal or ""):
            return 0.95
        return 0.0

    def scaffold_system_prompt(self) -> str:
        return (
            "FULLSTACK — Next.js project (App Router)\n"
            "- Use the App Router layout: app/layout.tsx (or .jsx) wraps "
            "every route; app/page.tsx is the index page; nested folders "
            "are nested routes.\n"
            "- API routes live under app/api/<route>/route.ts as named "
            "GET/POST exports.\n"
            "- package.json must list `next` (^14), `react` and `react-dom` "
            "(^18) under dependencies. Scripts: `dev` → `next dev`, "
            "`build` → `next build`, `start` → `next start`.\n"
            "- next.config.js (or .mjs) at project root, even if empty.\n"
            "- tsconfig.json when generating TypeScript variants.\n"
            "- Do NOT add webpack config — Next.js owns bundling.\n"
            "- Do NOT add a separate Vite/CRA setup."
        )

    def plan_system_prompt(self) -> str:
        return (
            "When modifying a Next.js project:\n"
            "- New pages go under app/<route>/page.tsx (App Router) or "
            "pages/<route>.tsx (Pages Router) — match whichever the project "
            "already uses.\n"
            "- Server components by default; add 'use client' only when "
            "hooks or browser APIs are needed."
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
