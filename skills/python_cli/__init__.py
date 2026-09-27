

"""Python command-line tool skill.

Fires on goals describing a CLI / script / command-line tool written in
Python. Distinct from the backend Python skills (FastAPI/Flask/Django)
because the expected layout and dependencies differ.
"""

from __future__ import annotations

import re
import sys
from typing import Any, Dict, List, Optional

from skills.base import (
    Skill,
    SkillVerdict,
    file_paths,
    file_with_content,
    has_python_test_file,
)

_CLI_NOUNS = re.compile(
    r"\b(cli|command[\s-]*line|script|tool|utility)\b", re.IGNORECASE
)
_PYTHON_RE = re.compile(r"\bpython\b", re.IGNORECASE)
# Frameworks that disqualify "this is a CLI" -- those skills will fire instead.
_WEB_RE = re.compile(
    r"\b(fastapi|flask|django|react|vue|next\.?js|express)\b", re.IGNORECASE
)

# Top-level `import x` / `from x import ...` -- raw file content, so the
# statement sits at column 0 (matches the diff shape the rest of the skills
# validate against). Relative imports (`from . import`) capture an empty top.
_IMPORT_RE = re.compile(r"^\s*(?:import|from)\s+([\w.]+)", re.M)

# The set of names that count as "already available", so importing them never
# demands a runtime dependency manifest. The Python stdlib (queried live so it
# tracks the interpreter) plus a curated fallback for the handful of names we
# rely on even if ``stdlib_module_names`` is somehow unavailable.
_STDLIB: frozenset = frozenset(getattr(sys, "stdlib_module_names", ())) | {
    "__future__", "argparse", "sys", "os", "json", "re", "pathlib", "typing",
    "collections", "itertools", "functools", "dataclasses", "subprocess",
    "shutil", "logging", "math", "random", "datetime", "time", "csv", "io",
    "textwrap", "enum", "abc", "contextlib", "tempfile", "glob", "hashlib",
    "sqlite3", "unittest", "configparser", "urllib", "http", "socket",
    "threading", "asyncio", "string", "decimal", "warnings", "traceback",
    "inspect", "importlib", "pprint", "shlex", "signal", "getpass",
}

# Build/test tooling a scaffold assumes is present in the dev environment --
# importing these is NOT the "ships uninstallable" signal we gate on, so a
# pure-stdlib CLI whose only non-stdlib import is `pytest` in its tests still
# counts as manifest-free.
_ASSUMED_PRESENT: frozenset = frozenset({
    "pytest", "_pytest", "setuptools", "pkg_resources",
})

# A dependency manifest by any of these names satisfies the "declare your
# runtime deps" requirement. Broader than flask/fastapi (which only look at
# requirements.txt / pyproject.toml) so a scaffold that pins deps in setup.py
# or setup.cfg is not fatally rejected.
_MANIFEST_SUFFIXES = ("requirements.txt", "pyproject.toml", "setup.py",
                      "setup.cfg")


class PythonCliSkill(Skill):
    name = "python_cli"
    role = "cli"
    aliases = ("Python CLI", "python script", "argparse")
    description = "Python command-line tool (argparse/click-style entry point)."

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
            "CLI -- Python command-line tool\n"
            "- Single entry script at src/<package>/cli.py (or just "
            "<package>/__main__.py) parsing args with `argparse`.\n"
            "TESTABLE ENTRYPOINT (the #1 Python-CLI codegen failure -- a test "
            "must be able to call `main([...])` without a TypeError):\n"
            "- Define `def main(argv=None):` and parse with "
            "`args = parser.parse_args(argv)` -- NOT the implicit "
            "`parser.parse_args()` that reads `sys.argv` directly. Threading "
            "`argv` through lets a test call `main([\"--flag\", \"x\"])` "
            "deterministically, while `argv=None` still falls back to "
            "`sys.argv[1:]` for real invocation.\n"
            "- `main()` returns an int exit code (0 success, non-zero "
            "failure); never call `sys.exit()`/`raise SystemExit` from inside "
            "`main()` itself.\n"
            "- End the module with `if __name__ == \"__main__\":` then "
            "`raise SystemExit(main())` so the process exit status is the "
            "return code and merely importing the module (as tests do) never "
            "runs `main()`.\n"
            "- If the tool is packaged, add a console_scripts entry under "
            "`[project.scripts]` in pyproject.toml (e.g. "
            "`mytool = \"mypackage.cli:main\"`) so it installs as a callable "
            "command.\n"
            "- Tests under tests/test_*.py call `main([...])` with fake argv "
            "and assert on the returned exit code / captured stdout.\n"
            "DEPENDENCIES: any third-party import (requests, click, rich, "
            "httpx, ...) MUST be pinned in requirements.txt or pyproject.toml "
            "-- otherwise the tool installs broken. A pure-stdlib CLI (only "
            "argparse/sys/os/json/...) needs no manifest; keep the dependency "
            "set minimal."
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
        # A third-party import with no dependency manifest ships an
        # uninstallable tool. Pure-stdlib CLIs are fine (no manifest needed) --
        # this is the requirements.txt/pyproject presence gate flask/fastapi
        # have but python_cli lacked, scoped to "only when a dep is used".
        third_party = self._third_party_imports(diffs, paths)
        if third_party and not self._has_manifest(paths):
            mods = ", ".join(sorted(third_party))
            return SkillVerdict(
                passed=False, confidence=0.8,
                rationale=(f"Python CLI skill: third-party import(s) {mods} "
                           "used but the scaffold has no requirements.txt or "
                           "pyproject.toml pinning them -- the tool installs "
                           "broken. Add a dependency manifest (a pure-stdlib "
                           "CLI needs none)."),
            )
        return None

    # ---- Dependency-manifest helpers ---------------------------------
    @staticmethod
    def _has_manifest(paths: List[str]) -> bool:
        """True when the scaffold ships a dependency-declaring manifest."""
        return any(
            p.replace("\\", "/").endswith(_MANIFEST_SUFFIXES) for p in paths
        )

    @classmethod
    def _local_module_names(cls, paths: List[str]) -> set:
        """Names importable from within the scaffold itself.

        Every directory component and ``.py`` file stem in the scaffold is a
        candidate local module/package, so an intra-project import is never
        mistaken for a third-party dependency. Intentionally broad: leniency
        here only shrinks the (fatal) third-party set, protecting precision.
        """
        local: set = set()
        for p in paths:
            pl = p.replace("\\", "/")
            if not pl.endswith(".py"):
                continue
            parts = pl.split("/")
            for seg in parts[:-1]:
                if seg and seg not in (".", ".."):
                    local.add(seg)
            stem = parts[-1][:-3]  # strip ".py"
            if stem and stem != "__init__":
                local.add(stem)
        return local

    @classmethod
    def _third_party_imports(cls, diffs: List[Dict[str, Any]],
                             paths: List[str]) -> Dict[str, str]:
        """Top-level modules imported that are neither stdlib nor local.

        Returns ``{module: file}`` (deduped by top-level module). Relative
        imports, stdlib, assumed-present dev/test tooling, and names that
        resolve to a file/dir in the scaffold are all excluded, so a match is
        a genuine external dependency.
        """
        local = cls._local_module_names(paths)
        found: Dict[str, str] = {}
        for d in diffs or []:
            if not isinstance(d, dict):
                continue
            path = str(d.get("file") or d.get("path") or "").replace("\\", "/")
            if not path.endswith(".py"):
                continue
            body = str(d.get("patch") or d.get("diff") or d.get("content")
                       or d.get("new_content") or "")
            for m in _IMPORT_RE.finditer(body):
                top = m.group(1).split(".", 1)[0]
                if not top:  # relative import (`from . import x`)
                    continue
                if top in _STDLIB or top in _ASSUMED_PRESENT or top in local:
                    continue
                found.setdefault(top, path)
        return found

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
