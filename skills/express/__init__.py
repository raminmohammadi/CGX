

"""Express.js (Node) backend skill."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterator, List, Optional, Tuple

from skills._structural import body_of
from skills.base import (
    Skill,
    SkillVerdict,
    file_paths,
    file_with_content,
    has_js_test_file,
)

_EXPRESS_RE = re.compile(r"\bexpress(?:\.?js)?\b", re.IGNORECASE)

# ESM `import ... from '...'` statement. Line-anchored so it never matches a
# `// import ...` comment or a ` * import ...` JSDoc line, and it requires the
# `from` clause so dynamic `import(...)` / `await import(...)` (valid in
# CommonJS) is left alone.
_ESM_IMPORT_RE = re.compile(r"^\s*import\b[^\n]*\bfrom\b", re.M)
# package.json opting into ESM.
_TYPE_MODULE_RE = re.compile(r'"type"\s*:\s*"module"')
# Creation of an express.Router() (or a destructured Router()).
_ROUTER_RE = re.compile(r"\b(?:express\.)?Router\s*\(\s*\)")
# Any export mechanism (CommonJS or ESM) -- used to confirm a router is
# actually exported so the entry file can mount it.
_EXPORT_RE = re.compile(
    r"module\.exports|exports\.|"
    r"export\s+(?:default|const|let|var|function|async|\{|\*)")
# Signs a module is the app entry (creates the app / listens / wires an `app`),
# NOT a pure Router module that must export -- such a file may build a router
# and mount it locally without exporting.
_APP_ENTRY_RE = re.compile(r"\bexpress\s*\(\s*\)|\.listen\s*\(|\bapp\s*\.")

# Extensions whose module system is governed by package.json "type" (so an ESM
# `import` in them is only valid when the project opted into ESM). `.mjs` is
# always ESM and `.ts`/`.tsx` is transpiled, so all are excluded from the mix
# check.
_CJS_GOVERNED_EXT = (".js", ".jsx", ".cjs")
# Node source extensions a Router module can live in.
_ROUTER_EXT = (".js", ".mjs", ".cjs", ".jsx", ".ts")
# A route module in the plan lives under a routes/ (or route/) directory.
_ROUTE_MODULE_RE = re.compile(r"(?:^|/)routes?/[^/]+\.(?:js|mjs|cjs|jsx|ts)$",
                              re.IGNORECASE)
# Acceptable entry-module basenames (permissive on the stem so a legitimately
# named entry -- main.js, www -- is never fatally rejected).
_ENTRY_BASENAMES = {
    f"{stem}.{ext}"
    for stem in ("index", "app", "server", "main")
    for ext in ("js", "mjs", "cjs", "ts")
} | {"www"}


def _rows(diffs: List[Dict[str, Any]]) -> Iterator[Tuple[str, str]]:
    """Yield ``(path, body)`` for each diff row across the shapes CGX uses."""
    for d in diffs or []:
        if not isinstance(d, dict):
            continue
        path = str(d.get("file") or d.get("path") or "").replace("\\", "/")
        body = str(d.get("patch") or d.get("diff") or d.get("content")
                   or d.get("new_content") or "")
        yield path, body


class ExpressSkill(Skill):
    name = "express"
    role = "backend"
    aliases = ("Express", "Express.js", "ExpressJS")
    description = "Express.js Node backend service."

    def detect(self, goal: str) -> float:
        if _EXPRESS_RE.search(goal or ""):
            return 0.95
        return 0.0

    def scaffold_system_prompt(self) -> str:
        return (
            "BACKEND -- Express.js (Node) service\n"
            "- Single entry module at server/index.js (or src/server.js) "
            "creating `const app = express()` and listening on `process.env."
            "PORT || 3000`.\n"
            "- Routes attach via `app.get('/...', handler)` / "
            "`app.post(...)`; group related routes under `express.Router()` "
            "in routes/<name>.js when there are more than a couple.\n"
            "- Use `express.json()` middleware for JSON bodies and `cors()` "
            "when this service is paired with a separate frontend skill.\n"
            "- package.json dependencies must include `express` (^4); "
            "`cors` when relevant. Scripts: `start` → `node server/index.js`, "
            "`dev` → `nodemon server/index.js` (when devDependency added).\n"
            "MODULE SYSTEM (pick ONE and use it in EVERY file -- do not mix):\n"
            "- If package.json declares `\"type\": \"module\"`, use ESM "
            "everywhere: `import express from 'express'` and, for a router "
            "module, `export default router`. Never use `require(...)` or "
            "`module.exports` in that project.\n"
            "- Otherwise (no `\"type\": \"module\"`, the CommonJS default) use "
            "CommonJS everywhere: `const express = require('express')` and "
            "`module.exports = router`. Never use `import ... from` / `export` "
            "syntax -- Node throws `Cannot use import statement outside a "
            "module`.\n"
            "- Every routes/*.js module MUST end by exporting its router "
            "(`module.exports = router`, or `export default router` under ESM) "
            "so the entry file can `app.use(...)` it.\n"
            "- Do NOT mix Python files into this service."
        )

    def plan_system_prompt(self) -> str:
        return (
            "When modifying an Express project:\n"
            "- Attach new routes to the existing `app` or a Router; don't "
            "create a parallel express() instance.\n"
            "- Keep middleware order: body parsers and CORS before route "
            "handlers.\n"
            "- When the plan introduces routes/*.js modules, it must also "
            "include a package.json and an entry module (index.js/app.js/"
            "server.js) that creates the express() app and mounts the routers."
        )

    def validate_scaffold(self, diffs: List[Dict[str, Any]],
                          goal: str = "") -> Optional[SkillVerdict]:
        paths = file_paths(diffs)
        if not paths:
            return None
        if not any(p.endswith((".js", ".mjs", ".cjs", ".ts")) for p in paths):
            return SkillVerdict(
                passed=False, confidence=0.9,
                rationale=("Express skill: scaffold has no Node source "
                           "files (.js/.ts)."),
            )
        if file_with_content(diffs, "express") is None:
            return SkillVerdict(
                passed=False, confidence=0.85,
                rationale=("Express skill: no generated file imports or "
                           "requires `express`."),
            )

        pkg = body_of(diffs, "package.json")
        if pkg is None:
            return SkillVerdict(
                passed=False, confidence=0.85,
                rationale=("Express skill: scaffold is missing package.json "
                           "with `express` dependency."),
            )
        if '"express"' not in pkg:
            return SkillVerdict(
                passed=False, confidence=0.85,
                rationale=("Express skill: package.json does not declare "
                           "`express` in its dependencies. Add "
                           '`"express": "^4.x"` under dependencies.'),
            )
        if '"scripts"' not in pkg or not re.search(r'"start"\s*:', pkg):
            return SkillVerdict(
                passed=False, confidence=0.8,
                rationale=("Express skill: package.json has no `scripts.start`. "
                           'Add `"scripts": {"start": "node server/index.js"}` '
                           "so the service can be launched."),
            )

        # CommonJS-vs-ESM consistency: a CommonJS project (no "type":"module")
        # that uses ESM `import ... from` throws at load. Only meaningful when
        # package.json is present (checked above) and did NOT opt into ESM.
        if not _TYPE_MODULE_RE.search(pkg):
            for path, body in _rows(diffs):
                if not path.endswith(_CJS_GOVERNED_EXT):
                    continue
                if _ESM_IMPORT_RE.search(body):
                    return SkillVerdict(
                        passed=False, confidence=0.9,
                        rationale=(
                            f"Express skill: {path} uses ESM `import ... from` "
                            "but package.json does not declare "
                            '`"type": "module"` (CommonJS is the default), so '
                            "Node throws `Cannot use import statement outside a "
                            'module`. Either add `"type": "module"` to '
                            "package.json (and use import/export everywhere) or "
                            "convert this file to `const x = require('...')` / "
                            "`module.exports`."),
                    )

        # A router module that never exports its router can't be mounted.
        exported = self._unexported_router(diffs)
        if exported is not None:
            return SkillVerdict(
                passed=False, confidence=0.85,
                rationale=(
                    f"Express skill: {exported} creates an express.Router() but "
                    "never exports it, so the entry file cannot mount it. End "
                    "the module with `module.exports = router` (or "
                    "`export default router` under ESM)."),
            )
        return None

    def _unexported_router(self,
                           diffs: List[Dict[str, Any]]) -> Optional[str]:
        """Path of a pure Router module that never exports its router, or None.

        High precision: fires only when a file creates an ``express.Router()``,
        has no export of any kind, is not itself the app entry (does not create
        the app / listen / wire an ``app`` object), and is not a test file.
        """
        for path, body in _rows(diffs):
            if not path.endswith(_ROUTER_EXT):
                continue
            if has_js_test_file([path]) or "__tests__" in path:
                continue
            if not _ROUTER_RE.search(body):
                continue
            if _APP_ENTRY_RE.search(body):
                continue  # app entry / mounts locally -- not a pure Router
            if not _EXPORT_RE.search(body):
                return path
        return None

    def validate_plan(self, diffs: List[Dict[str, Any]],
                      goal: str = "") -> Optional[SkillVerdict]:
        """Paths-only: a plan with route modules needs a manifest + an entry.

        When the plan defines ``routes/*.js`` modules but omits package.json or
        an entry module (index.js/app.js/server.js) that creates the app and
        mounts them, the tree cannot run. Re-ask the planner with that reason.
        """
        paths = [p.replace("\\", "/") for p in file_paths(diffs)]
        if not paths:
            return None
        if not any(_ROUTE_MODULE_RE.search(p) for p in paths):
            return None  # no route modules planned; not this skill's concern
        lower = [p.lower() for p in paths]
        missing: List[str] = []
        if not any(p.rsplit("/", 1)[-1] == "package.json" for p in lower):
            missing.append("a package.json")
        has_entry = any(
            p.rsplit("/", 1)[-1] in _ENTRY_BASENAMES
            and not _ROUTE_MODULE_RE.search(p)
            for p in lower
        )
        if not has_entry:
            missing.append("an entry module (index.js/app.js/server.js) that "
                           "creates the express() app and mounts the routers")
        if missing:
            return SkillVerdict(
                passed=False, confidence=0.8,
                rationale=("Express skill: the plan defines route modules but "
                           "is missing " + " and ".join(missing) + "."),
            )
        return None

    def scaffold_warnings(self, diffs: List[Dict[str, Any]],
                          goal: str = "") -> List[SkillVerdict]:
        paths = file_paths(diffs)
        if not paths or not any(p.endswith((".js", ".mjs", ".cjs", ".ts"))
                                for p in paths):
            return []
        if has_js_test_file(paths):
            return []
        return [SkillVerdict(
            passed=False, confidence=0.7, severity="warning",
            rationale=("Express skill: no test file generated. Add a "
                       "tests/app.test.js using `supertest` to exercise "
                       "the routes."),
        )]


__all__ = ["ExpressSkill"]
