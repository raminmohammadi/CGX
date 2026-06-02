# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ramin Mohammadi

"""Tailwind CSS styling skill.

Tailwind is an addon: it composes with whatever frontend skill is also
active. It contributes a configuration-prompt fragment and validates
that the scaffold actually includes a Tailwind config + the directives.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from skills.base import Skill, SkillVerdict, file_paths, file_with_content


_TAILWIND_RE = re.compile(r"\btailwind(?:css)?\b", re.IGNORECASE)


class TailwindSkill(Skill):
    name = "tailwind"
    role = "style"
    aliases = ("Tailwind", "TailwindCSS", "Tailwind CSS")

    def detect(self, goal: str) -> float:
        if _TAILWIND_RE.search(goal or ""):
            return 0.95
        return 0.0

    def scaffold_system_prompt(self) -> str:
        return (
            "STYLE — Tailwind CSS\n"
            "- Add a tailwind.config.js at project root with a `content` "
            "array covering `./index.html` and `./src/**/*.{js,jsx,ts,tsx,vue}`.\n"
            "- Add a postcss.config.js declaring `tailwindcss` and "
            "`autoprefixer` plugins.\n"
            "- Add an entry CSS file (src/index.css or src/main.css) that "
            "starts with `@tailwind base; @tailwind components; @tailwind "
            "utilities;` and is imported from src/main.{js,jsx}.\n"
            "- package.json devDependencies must include `tailwindcss`, "
            "`postcss`, and `autoprefixer`.\n"
            "- Use Tailwind utility classes in markup — do NOT also emit "
            "redundant custom CSS for the same elements."
        )

    def validate_scaffold(self, diffs: List[Dict[str, Any]],
                          goal: str = "") -> Optional[SkillVerdict]:
        paths = file_paths(diffs)
        if not paths:
            return None
        has_cfg = any(p.endswith("tailwind.config.js")
                      or p.endswith("tailwind.config.ts")
                      or p.endswith("tailwind.config.cjs")
                      or p.endswith("tailwind.config.mjs")
                      for p in paths)
        if not has_cfg:
            return SkillVerdict(
                passed=False, confidence=0.85,
                rationale=("Tailwind skill: scaffold is missing "
                           "tailwind.config.js. Add it at project root."),
            )
        # Verify the @tailwind directives are present in some CSS file.
        has_directives = file_with_content(diffs, "@tailwind base") is not None
        if not has_directives:
            return SkillVerdict(
                passed=False, confidence=0.8,
                rationale=("Tailwind skill: no CSS file contains the "
                           "`@tailwind base; @tailwind components; "
                           "@tailwind utilities;` directives."),
            )
        return None


__all__ = ["TailwindSkill"]
