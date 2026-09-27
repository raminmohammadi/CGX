

"""Tailwind CSS styling skill.

Tailwind is an addon: it composes with whatever frontend skill is also
active. It contributes a configuration-prompt fragment and validates that
the scaffold actually wires Tailwind up correctly -- but Tailwind has THREE
legitimate setups, and the validator must accept all of them instead of
hard-coding the v3 "tailwind.config.js + postcss + @tailwind directives"
recipe:

* **Build-less / CDN** -- a static site loads the Play CDN
  (``<script src="https://cdn.tailwindcss.com">``) and just uses utility
  classes. There is NO ``package.json``, NO config, NO PostCSS. This is
  exactly the shape the ``static_site`` skill emits, so requiring a config
  here would deadlock the two skills (each fatally rejecting the other's
  valid output). Tailwind therefore ABSTAINS whenever the site is build-less.
* **Built, Tailwind v4** -- a single ``@import "tailwindcss";`` in the entry
  CSS is a complete setup; v4 needs NO ``tailwind.config.js`` and NO
  ``@tailwind base/components/utilities`` directives.
* **Built, Tailwind v3** -- the classic ``tailwind.config.js`` + PostCSS +
  ``@tailwind base; @tailwind components; @tailwind utilities;`` recipe.

Because a missing v3 config can be a legitimately v4/CDN project rather than a
real defect, the only non-abstaining verdicts are advisory (``severity=
"warning"``) -- never a fatal regenerate.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from skills._structural import any_body_matches
from skills.base import Skill, SkillVerdict, file_paths

_TAILWIND_RE = re.compile(r"\btailwind(?:css)?\b", re.IGNORECASE)

# A Tailwind config file, any of the four extensions Tailwind accepts.
_CONFIG_SUFFIXES = (
    "tailwind.config.js", "tailwind.config.ts",
    "tailwind.config.cjs", "tailwind.config.mjs",
)

# Positive "this is loaded via a CDN, no build step" signals. Matched over the
# whole diff body (all extensions) because they live in HTML <head>.
_CDN_TAILWIND_RE = r"(?i)cdn\.tailwindcss\.com"
_CDN_SCRIPT_RE = (
    r"""(?i)<script[^>]*\bsrc\s*=\s*['"][^'"]*tailwind[^'"]*['"]"""
)
# Tailwind v4: a single `@import "tailwindcss";` is a complete setup.
_V4_IMPORT_RE = r"""(?i)@import\s+['"]tailwindcss"""
# Tailwind v3: the classic entry directives.
_V3_DIRECTIVE_RE = r"(?i)@tailwind\s+(?:base|components|utilities)"


class TailwindSkill(Skill):
    name = "tailwind"
    role = "style"
    aliases = ("Tailwind", "TailwindCSS", "Tailwind CSS")
    description = "Tailwind CSS utility-first styling addon."

    def detect(self, goal: str) -> float:
        if _TAILWIND_RE.search(goal or ""):
            return 0.95
        return 0.0

    def scaffold_system_prompt(self) -> str:
        return (
            "STYLE -- Tailwind CSS (pick the ONE setup that matches the "
            "project; do not mix them)\n"
            "- BUILD-LESS / static site or CDN (no bundler, no package.json): "
            "load the Play CDN with "
            "`<script src=\"https://cdn.tailwindcss.com\"></script>` in <head> "
            "and use utility classes in the markup. Do NOT add "
            "tailwind.config, postcss.config, or package.json -- the CDN needs "
            "none of them.\n"
            "- BUILT project, Tailwind v4 (preferred for new builds): add "
            "`tailwindcss` and `@tailwindcss/postcss` (or the "
            "`@tailwindcss/vite` plugin) to package.json, and put a SINGLE "
            "`@import \"tailwindcss\";` at the top of the entry CSS "
            "(src/index.css). v4 needs NO tailwind.config.js and NO "
            "`@tailwind base/components/utilities` directives.\n"
            "- BUILT project, Tailwind v3: add a tailwind.config.js at project "
            "root with a `content` array covering `./index.html` and "
            "`./src/**/*.{js,jsx,ts,tsx,vue}`, a postcss.config.js declaring "
            "`tailwindcss` and `autoprefixer`, list "
            "`tailwindcss`/`postcss`/`autoprefixer` in package.json "
            "devDependencies, and start the entry CSS with `@tailwind base; "
            "@tailwind components; @tailwind utilities;` (imported from "
            "src/main.{js,jsx}).\n"
            "- Use Tailwind utility classes in markup -- do NOT also emit "
            "redundant custom CSS for the same elements."
        )

    # --- validation --------------------------------------------------------
    def _is_buildless(self, diffs: List[Dict[str, Any]],
                      paths: List[str]) -> bool:
        """True when Tailwind is being used with no build step.

        Either a CDN Tailwind is loaded (definitive), or there is no
        ``package.json`` anywhere in the diffs (no build system -> the config
        files a build would need are irrelevant). In both cases the skill must
        not require -- or reject over -- a missing config, otherwise it
        deadlocks with the build-less ``static_site`` skill.
        """
        if any_body_matches(diffs, _CDN_TAILWIND_RE):
            return True
        if any_body_matches(diffs, _CDN_SCRIPT_RE):
            return True
        has_pkg = any(p.replace("\\", "/").rsplit("/", 1)[-1] == "package.json"
                      for p in paths)
        return not has_pkg

    def validate_scaffold(self, diffs: List[Dict[str, Any]],
                          goal: str = "") -> Optional[SkillVerdict]:
        paths = file_paths(diffs)
        if not paths:
            return None

        # (A) Build-less / CDN Tailwind needs no config, no PostCSS, no build
        # directives -- abstain entirely (this is the static_site skill's
        # valid output; requiring a config here deadlocks the two skills).
        if self._is_buildless(diffs, paths):
            return None

        # (B) Tailwind v4: a single `@import "tailwindcss";` is a complete,
        # valid setup -- no tailwind.config.js and no @tailwind directives are
        # required. Accept it.
        if any_body_matches(diffs, _V4_IMPORT_RE):
            return None

        # (C) A build-based project that is not v4: expect the v3 recipe. A
        # missing piece is only ADVISORY (never fatal): the project could be a
        # v4/CDN setup whose defining file simply is not in this diff slice, so
        # a fatal regenerate would risk rejecting legitimate output.
        has_cfg = any(p.endswith(_CONFIG_SUFFIXES) for p in paths)
        if not has_cfg:
            return SkillVerdict(
                passed=False, confidence=0.7, severity="warning",
                rationale=("Tailwind skill: no tailwind.config.js and no "
                           "`@import \"tailwindcss\";` (v4) were found for what "
                           "looks like a build-based project. Add a "
                           "tailwind.config.js (v3), switch to a single "
                           "`@import \"tailwindcss\";` (v4), or load the Play "
                           "CDN for a build-less site."),
            )
        # Config present -> v3: the entry directives should exist somewhere.
        if not any_body_matches(diffs, _V3_DIRECTIVE_RE):
            return SkillVerdict(
                passed=False, confidence=0.65, severity="warning",
                rationale=("Tailwind skill: a tailwind.config.js is present but "
                           "no CSS file contains the `@tailwind base; "
                           "@tailwind components; @tailwind utilities;` "
                           "directives (v3) or `@import \"tailwindcss\";` (v4)."),
            )
        return None


__all__ = ["TailwindSkill"]
