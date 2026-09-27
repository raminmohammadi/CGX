

"""Vue 3 frontend skill (with Nuxt detection as a co-mention)."""

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

_VUE_RE = re.compile(r"\bvue(?:\.?js)?\b", re.IGNORECASE)
_NUXT_RE = re.compile(r"\bnuxt\b", re.IGNORECASE)

# vite.config files hold the build config; the Vue plugin must be registered
# there for .vue single-file components to compile at all.
_VITE_CONFIGS = ("vite.config.js", "vite.config.ts")


class VueSkill(Skill):
    name = "vue"
    role = "frontend"
    aliases = ("Vue", "Vue.js", "VueJS", "Nuxt")
    description = "Vue 3 (Composition API) frontend, with Nuxt co-detection."

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

    @staticmethod
    def _is_nuxt(goal: str, paths: List[str]) -> bool:
        """True when the goal/plan is a Nuxt app rather than a plain Vite+Vue app.

        Nuxt owns the build (no ``vite.config``/``index.html``/``src/main.*``
        entry and Vue is wired internally, not via ``@vitejs/plugin-vue``), so
        the Vite-shaped requirements below must NOT fire for it -- otherwise a
        perfectly valid Nuxt scaffold gets fatally rejected.
        """
        if _NUXT_RE.search(goal or ""):
            return True
        for p in paths:
            base = p.replace("\\", "/").rsplit("/", 1)[-1].lower()
            if base.startswith("nuxt.config"):
                return True
        return False

    def validate_plan(self, diffs: List[Dict[str, Any]],
                      goal: str = "") -> Optional[SkillVerdict]:
        """Veto a Vue plan that builds an incomplete Vite app skeleton.

        Paths-only at plan time. A Vite + Vue app needs four build pieces --
        ``index.html`` (Vite's entry), a ``vite.config.(js|ts)``, a
        ``package.json``, and a ``src/main.(js|ts)`` mount. The check fires
        only when the plan is actually authoring/altering that skeleton
        (i.e. it already touches at least one of those files), so a plain
        "add a component" modification (only ``.vue`` under components/) is
        never rejected. Nuxt plans are skipped -- they have a different layout.
        """
        paths = [p.replace("\\", "/") for p in file_paths(diffs)]
        if not paths or self._is_nuxt(goal, paths):
            return None
        bases = {p.rsplit("/", 1)[-1].lower() for p in paths}
        has_index = "index.html" in bases
        has_vite = bool(bases & set(_VITE_CONFIGS))
        has_pkg = "package.json" in bases
        has_main = bool(bases & {"main.js", "main.ts"})
        # Only opine once the plan is clearly building the app skeleton; a plan
        # touching none of these is a component edit, not a scaffold.
        if not (has_index or has_vite or has_pkg or has_main):
            return None
        missing: List[str] = []
        if not has_index:
            missing.append("an index.html entry (Vite's build entry)")
        if not has_vite:
            missing.append("a vite.config.js/ts registering @vitejs/plugin-vue")
        if not has_pkg:
            missing.append("a package.json")
        if not has_main:
            missing.append("a src/main.js/ts that mounts the app")
        if missing:
            return SkillVerdict(
                passed=False, confidence=0.9,
                rationale=("Vue skill: the plan builds a Vue app skeleton but "
                           "is missing " + " and ".join(missing)
                           + "; a Vite + Vue build needs them."),
            )
        return None

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
        # The remaining checks are Vite-specific; Nuxt owns its own build.
        if self._is_nuxt(goal, paths):
            return None
        if has_any_ext(paths, (".vue",)):
            # .vue single-file components DO NOT compile unless the Vue plugin
            # is registered in vite.config -- a missing plugin hard-fails the
            # build with an unhelpful "failed to parse" error, so gate on it.
            if not any_body_matches(diffs, r"plugin-vue",
                                    only_ext=_VITE_CONFIGS):
                return SkillVerdict(
                    passed=False, confidence=0.9,
                    rationale=("Vue skill: .vue single-file components are "
                               "present but no vite.config registers "
                               "@vitejs/plugin-vue. Vite cannot compile .vue "
                               "files without it and the build hard-fails. Add "
                               "`import vue from '@vitejs/plugin-vue'` and "
                               "`plugins: [vue()]` to vite.config.js."),
                )
            # package.json must declare the build trio, or install/build fails.
            pkg = body_of(diffs, "package.json")
            if pkg:
                missing = []
                if not re.search(r'"vue"\s*:', pkg):
                    missing.append("vue (^3, dependencies)")
                if "@vitejs/plugin-vue" not in pkg:
                    missing.append("@vitejs/plugin-vue (devDependencies)")
                if not re.search(r'"vite"\s*:', pkg):
                    missing.append("vite (devDependencies)")
                if missing:
                    return SkillVerdict(
                        passed=False, confidence=0.85,
                        rationale=("Vue skill: package.json does not declare "
                                   + ", ".join(missing) + ". A Vite + Vue "
                                   "build needs `vue`, `vite` and "
                                   "`@vitejs/plugin-vue`; missing them breaks "
                                   "npm install / the build. Add them."),
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
