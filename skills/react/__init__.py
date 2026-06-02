# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ramin Mohammadi

"""React frontend skill.

Detects goals naming React (but not React Native), supplies a
Vite-based scaffold prompt, and validates that scaffold outputs
actually contain JS/TS source files rather than a Python fallback.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from skills.base import (
    Skill, SkillVerdict, file_paths, has_any_ext, has_js_test_file,
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


class ReactSkill(Skill):
    name = "react"
    role = "frontend"
    aliases = ("React", "react.js", "ReactJS")

    def detect(self, goal: str) -> float:
        g = goal or ""
        # React Native is a distinct ecosystem — don't fire on it.
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
            "FRONTEND — React project\n"
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
                rationale=("React skill: every source file is Python — the "
                           "scaffold ignored the React requirement."),
            )
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
