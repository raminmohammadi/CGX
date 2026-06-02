# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ramin Mohammadi

"""Python command-line tool skill.

Fires on goals describing a CLI / script / command-line tool written in
Python. Distinct from the backend Python skills (FastAPI/Flask/Django)
because the expected layout and dependencies differ.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from skills.base import (
    Skill, SkillVerdict, file_paths, file_with_content, has_python_test_file,
)


_CLI_NOUNS = re.compile(
    r"\b(cli|command[\s-]*line|script|tool|utility)\b", re.IGNORECASE
)
_PYTHON_RE = re.compile(r"\bpython\b", re.IGNORECASE)
# Frameworks that disqualify "this is a CLI" — those skills will fire
# instead.
_WEB_RE = re.compile(
    r"\b(fastapi|flask|django|react|vue|next\.?js|express)\b", re.IGNORECASE
)


class PythonCliSkill(Skill):
    name = "python_cli"
    role = "cli"
    aliases = ("Python CLI", "python script", "argparse")

    def detect(self, goal: str) -> float:
        g = goal or ""
        # Web framework wins; CLI abstains so it doesn't compose
        # contradictory advice into the scaffold prompt.
        if _WEB_RE.search(g):
            return 0.0
        has_python = bool(_PYTHON_RE.search(g))
        has_cli_noun = bool(_CLI_NOUNS.search(g))
        if has_python and has_cli_noun:
            return 0.9
        if has_cli_noun:
            return 0.55
        return 0.0

    def scaffold_system_prompt(self) -> str:
        return (
            "CLI — Python command-line tool\n"
            "- Single entry script at src/<package>/cli.py (or just "
            "<package>/__main__.py) parsing args with `argparse`.\n"
            "- Provide a `main()` function returning an int exit code, and "
            "an `if __name__ == \"__main__\":` block calling "
            "`sys.exit(main())`.\n"
            "- Use a console_scripts entry point in pyproject.toml so the "
            "tool installs as a callable command.\n"
            "- Tests under tests/test_*.py exercising `main([...])` with "
            "fake argv.\n"
            "- requirements.txt or pyproject.toml pins runtime dependencies; "
            "keep the dependency set minimal."
        )

    def plan_system_prompt(self) -> str:
        return (
            "When modifying a Python CLI:\n"
            "- New subcommands extend the existing argparse parser "
            "(subparsers); don't create a parallel parser.\n"
            "- Keep `main()` return-code semantics: 0 success, non-zero "
            "failure."
        )

    def validate_scaffold(self, diffs: List[Dict[str, Any]],
                          goal: str = "") -> Optional[SkillVerdict]:
        paths = file_paths(diffs)
        if not paths:
            return None
        if not any(p.endswith(".py") for p in paths):
            return SkillVerdict(
                passed=False, confidence=0.9,
                rationale=("Python CLI skill: scaffold has no Python "
                           "files."),
            )
        # Require either argparse use or a clear console_scripts entry.
        has_argparse = file_with_content(diffs, "argparse") is not None
        has_entry = file_with_content(diffs, "console_scripts") is not None
        has_main_guard = file_with_content(diffs, "__main__") is not None
        if not (has_argparse or has_entry or has_main_guard):
            return SkillVerdict(
                passed=False, confidence=0.75,
                rationale=("Python CLI skill: no file uses `argparse`, "
                           "declares a `console_scripts` entry, or has an "
                           "`if __name__ == \"__main__\"` block."),
            )
        return None

    def scaffold_warnings(self, diffs: List[Dict[str, Any]],
                          goal: str = "") -> List[SkillVerdict]:
        paths = file_paths(diffs)
        if not paths or not any(p.endswith(".py") for p in paths):
            return []
        if has_python_test_file(paths):
            return []
        return [SkillVerdict(
            passed=False, confidence=0.7, severity="warning",
            rationale=("Python CLI skill: no test file generated. Add a "
                       "tests/test_cli.py that calls `main([...])` with "
                       "fake argv."),
        )]


__all__ = ["PythonCliSkill"]
