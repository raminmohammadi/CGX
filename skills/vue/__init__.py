

"""Vue 3 frontend skill (with Nuxt detection as a co-mention)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from skills.base import (
    Skill, SkillVerdict, file_paths, has_any_ext, has_js_test_file,
)


_VUE_RE = re.compile(r"\bvue(?:\.?js)?\b", re.IGNORECASE)
_NUXT_RE = re.compile(r"\bnuxt\b", re.IGNORECASE)


class VueSkill(Skill):
    name = "vue"
    role = "frontend"
    aliases = ("Vue", "Vue.js", "VueJS", "Nuxt")

    def detect(self, goal: str) -> float:
        g = goal or ""
        if _VUE_RE.search(g) or _NUXT_RE.search(g):
            return 0.95
        return 0.0

    def scaffold_system_prompt(self) -> str:
        return (
            "FRONTEND -- Vue 3 project\n"
            "- Use Vite + Vue 3 with single-file components (.vue).\n"
            "- src/main.js mounts the app; src/App.vue is the root SFC; "
            "src/components/*.vue for individual pieces.\n"
            "- index.html at the project root with `<div id=\"app\"></div>` "
            "and `<script type=\"module\" src=\"/src/main.js\"></script>`.\n"
            "- package.json lists `vue` (^3) under dependencies and "
            "`vite` + `@vitejs/plugin-vue` under devDependencies.\n"
            "- vite.config.js with the Vue plugin.\n"
            "- Use the Composition API (`<script setup>`) by default.\n"
            "- Do NOT emit Python files for the UI layer."
        )

    def validate_scaffold(self, diffs: List[Dict[str, Any]],
                          goal: str = "") -> Optional[SkillVerdict]:
        paths = file_paths(diffs)
        if not paths:
            return None
        if not has_any_ext(paths, (".vue", ".js", ".ts")):
            return SkillVerdict(
                passed=False, confidence=0.9,
                rationale=("Vue skill: scaffold has no .vue/.js/.ts files. "
                           "Regenerate with src/App.vue + src/main.js + "
                           "package.json."),
            )
        non_meta = [p for p in paths
                    if not p.lower().endswith((".md", ".txt", ".cfg", ".ini",
                                               ".toml", ".yml", ".yaml",
                                               ".json", ".lock"))]
        if non_meta and all(p.lower().endswith(".py") for p in non_meta):
            return SkillVerdict(
                passed=False, confidence=0.9,
                rationale=("Vue skill: every source file is Python -- the "
                           "scaffold ignored the Vue requirement."),
            )
        return None

    def scaffold_warnings(self, diffs: List[Dict[str, Any]],
                          goal: str = "") -> List[SkillVerdict]:
        paths = file_paths(diffs)
        if not paths or not has_any_ext(paths, (".vue", ".js", ".ts")):
            return []
        if has_js_test_file(paths):
            return []
        return [SkillVerdict(
            passed=False, confidence=0.7, severity="warning",
            rationale=("Vue skill: no test file generated. Add a "
                       "tests/<Component>.spec.js using Vitest + "
                       "@vue/test-utils to mount and assert."),
        )]


__all__ = ["VueSkill"]
