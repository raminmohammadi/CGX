"""Tests for the ExpressSkill: detection, prompt guidance, CommonJS/ESM
consistency, router-export and package.json/entry gates on scaffold and plan.
"""

from __future__ import annotations

import pytest

import skills as registry
from skills.express import ExpressSkill


@pytest.fixture()
def skill() -> ExpressSkill:
    return ExpressSkill()


def _diff(path: str, patch: str = "") -> dict:
    return {"file": path, "patch": patch}


_PKG_CJS = ('{"name":"api","scripts":{"start":"node server/index.js"},'
            '"dependencies":{"express":"^4.18.2"}}')
_PKG_ESM = ('{"name":"api","type":"module",'
            '"scripts":{"start":"node server/index.js"},'
            '"dependencies":{"express":"^4.18.2"}}')


# --- detection -------------------------------------------------------------

@pytest.mark.parametrize("goal", [
    "build an Express.js API",
    "a node express backend for a todo list",
    "REST service with ExpressJS",
])
def test_detects_express_goals(skill, goal):
    assert skill.detect(goal) >= 0.9


def test_does_not_fire_on_unrelated_goal(skill):
    assert skill.detect("build a flask api") == 0.0
    assert skill.detect("") == 0.0


def test_registered_and_detected_via_registry():
    names = [s.name for s in registry.detect_skills("build an express api")]
    assert "express" in names


# --- prompt guidance -------------------------------------------------------

def test_scaffold_prompt_has_module_system_guidance(skill):
    p = skill.scaffold_system_prompt()
    assert '"type": "module"' in p
    assert "module.exports" in p and "export default" in p
    assert "require(" in p


# --- CommonJS-vs-ESM consistency (FATAL) -----------------------------------

def test_esm_import_without_type_module_is_fatal(skill):
    diffs = [
        _diff("server/index.js",
              "import express from 'express'\n"
              "const app = express()\napp.listen(3000)\n"),
        _diff("package.json", _PKG_CJS),
    ]
    v = skill.validate_scaffold(diffs)
    assert v is not None and not v.passed and v.severity == "error"
    assert "server/index.js" in v.rationale


def test_esm_import_ok_when_type_module_declared(skill):
    diffs = [
        _diff("server/index.js",
              "import express from 'express'\n"
              "const app = express()\napp.listen(3000)\n"),
        _diff("package.json", _PKG_ESM),
    ]
    assert skill.validate_scaffold(diffs) is None


def test_commonjs_require_passes(skill):
    diffs = [
        _diff("server/index.js",
              "const express = require('express')\n"
              "const app = express()\napp.listen(3000)\n"),
        _diff("package.json", _PKG_CJS),
    ]
    assert skill.validate_scaffold(diffs) is None


def test_dynamic_import_is_not_flagged(skill):
    # `await import(...)` is legal in CommonJS -- must not trip the mix check.
    diffs = [
        _diff("server/index.js",
              "const express = require('express')\n"
              "const app = express()\n"
              "async function load(){ const m = await import('node:fs'); }\n"
              "app.listen(3000)\n"),
        _diff("package.json", _PKG_CJS),
    ]
    assert skill.validate_scaffold(diffs) is None


def test_import_in_comment_is_not_flagged(skill):
    diffs = [
        _diff("server/index.js",
              "// import express from 'express' -- old style\n"
              "const express = require('express')\n"
              "const app = express()\napp.listen(3000)\n"),
        _diff("package.json", _PKG_CJS),
    ]
    assert skill.validate_scaffold(diffs) is None


def test_mjs_esm_import_not_flagged_even_without_type_module(skill):
    # .mjs is always an ES module regardless of package.json "type".
    diffs = [
        _diff("server/index.mjs",
              "import express from 'express'\n"
              "const app = express()\napp.listen(3000)\n"),
        _diff("package.json", _PKG_CJS),
    ]
    assert skill.validate_scaffold(diffs) is None


# --- router-export gate (FATAL) --------------------------------------------

def test_router_without_export_is_fatal(skill):
    diffs = [
        _diff("server/index.js",
              "const express = require('express')\n"
              "const app = express()\napp.listen(3000)\n"),
        _diff("routes/users.js",
              "const express = require('express')\n"
              "const router = express.Router()\n"
              "router.get('/', (req, res) => res.json([]))\n"),
        _diff("package.json", _PKG_CJS),
    ]
    v = skill.validate_scaffold(diffs)
    assert v is not None and not v.passed
    assert "routes/users.js" in v.rationale


def test_router_with_commonjs_export_passes(skill):
    diffs = [
        _diff("server/index.js",
              "const express = require('express')\n"
              "const app = express()\napp.listen(3000)\n"),
        _diff("routes/users.js",
              "const express = require('express')\n"
              "const router = express.Router()\n"
              "router.get('/', (req, res) => res.json([]))\n"
              "module.exports = router\n"),
        _diff("package.json", _PKG_CJS),
    ]
    assert skill.validate_scaffold(diffs) is None


def test_router_with_esm_default_export_passes(skill):
    diffs = [
        _diff("server/index.js",
              "import express from 'express'\n"
              "const app = express()\napp.listen(3000)\n"),
        _diff("routes/users.js",
              "import express from 'express'\n"
              "const router = express.Router()\n"
              "router.get('/', (req, res) => res.json([]))\n"
              "export default router\n"),
        _diff("package.json", _PKG_ESM),
    ]
    assert skill.validate_scaffold(diffs) is None


def test_router_exported_as_named_const_passes(skill):
    diffs = [
        _diff("server/index.js",
              "import express from 'express'\n"
              "const app = express()\napp.listen(3000)\n"),
        _diff("routes/users.js",
              "import express from 'express'\n"
              "export const router = express.Router()\n"
              "router.get('/', (req, res) => res.json([]))\n"),
        _diff("package.json", _PKG_ESM),
    ]
    assert skill.validate_scaffold(diffs) is None


def test_entry_creating_local_router_not_flagged(skill):
    # The app entry may build a sub-router and mount it locally without export.
    diffs = [
        _diff("server/index.js",
              "const express = require('express')\n"
              "const app = express()\n"
              "const router = express.Router()\n"
              "router.get('/', (req, res) => res.send('ok'))\n"
              "app.use('/api', router)\n"
              "app.listen(3000)\n"),
        _diff("package.json", _PKG_CJS),
    ]
    assert skill.validate_scaffold(diffs) is None


def test_test_file_router_without_export_not_flagged(skill):
    diffs = [
        _diff("server/index.js",
              "const express = require('express')\n"
              "const app = express()\napp.listen(3000)\n"),
        _diff("tests/app.test.js",
              "const express = require('express')\n"
              "const router = express.Router()\n"
              "// exercised inline, not exported\n"),
        _diff("package.json", _PKG_CJS),
    ]
    assert skill.validate_scaffold(diffs) is None


# --- package.json requirements (FATAL) -------------------------------------

def test_missing_package_json_is_fatal(skill):
    diffs = [
        _diff("server/index.js",
              "const express = require('express')\n"
              "const app = express()\napp.listen(3000)\n"),
    ]
    v = skill.validate_scaffold(diffs)
    assert v is not None and not v.passed and "package.json" in v.rationale


def test_package_json_without_express_dep_is_fatal(skill):
    diffs = [
        _diff("server/index.js",
              "const express = require('express')\n"
              "const app = express()\napp.listen(3000)\n"),
        _diff("package.json",
              '{"name":"api","scripts":{"start":"node server/index.js"},'
              '"dependencies":{"cors":"^2"}}'),
    ]
    v = skill.validate_scaffold(diffs)
    assert v is not None and not v.passed
    assert "express" in v.rationale and "dependencies" in v.rationale


def test_package_json_without_start_script_is_fatal(skill):
    diffs = [
        _diff("server/index.js",
              "const express = require('express')\n"
              "const app = express()\napp.listen(3000)\n"),
        _diff("package.json",
              '{"name":"api","dependencies":{"express":"^4.18.2"}}'),
    ]
    v = skill.validate_scaffold(diffs)
    assert v is not None and not v.passed and "scripts.start" in v.rationale


def test_no_node_source_is_fatal(skill):
    v = skill.validate_scaffold([_diff("README.md", "# api")])
    assert v is not None and not v.passed


def test_empty_diffs_no_opinion(skill):
    assert skill.validate_scaffold([]) is None
    assert skill.validate_plan([]) is None


def test_clean_scaffold_passes(skill):
    diffs = [
        _diff("server/index.js",
              "const express = require('express')\n"
              "const usersRouter = require('../routes/users')\n"
              "const app = express()\n"
              "app.use(express.json())\n"
              "app.use('/users', usersRouter)\n"
              "app.listen(process.env.PORT || 3000)\n"),
        _diff("routes/users.js",
              "const express = require('express')\n"
              "const router = express.Router()\n"
              "router.get('/', (req, res) => res.json([]))\n"
              "module.exports = router\n"),
        _diff("package.json", _PKG_CJS),
        _diff("tests/app.test.js",
              "const request = require('supertest')\n"),
    ]
    assert skill.validate_scaffold(diffs) is None


# --- plan validation (paths only) ------------------------------------------

def test_plan_route_module_without_package_json_and_entry_is_fatal(skill):
    v = skill.validate_plan([_diff("routes/users.js")])
    assert v is not None and not v.passed
    assert "package.json" in v.rationale and "entry" in v.rationale


def test_plan_route_module_missing_only_entry_is_fatal(skill):
    v = skill.validate_plan([_diff("routes/users.js"), _diff("package.json")])
    assert v is not None and not v.passed
    assert "entry" in v.rationale and "package.json" not in v.rationale


def test_plan_route_index_does_not_count_as_entry(skill):
    # routes/index.js is a route module, not the app entry.
    v = skill.validate_plan([_diff("routes/index.js"), _diff("package.json")])
    assert v is not None and not v.passed and "entry" in v.rationale


def test_plan_complete_layout_passes(skill):
    diffs = [
        _diff("package.json"),
        _diff("server/index.js"),
        _diff("routes/users.js"),
    ]
    assert skill.validate_plan(diffs) is None


def test_plan_app_entry_variant_passes(skill):
    diffs = [
        _diff("package.json"),
        _diff("app.js"),
        _diff("routes/users.js"),
    ]
    assert skill.validate_plan(diffs) is None


def test_plan_without_route_modules_no_opinion(skill):
    # No routes/ modules planned -> the skill abstains (e.g. incremental edit).
    assert skill.validate_plan([_diff("server/index.js")]) is None
    assert skill.validate_plan([_diff("package.json")]) is None


# --- warnings --------------------------------------------------------------

def test_scaffold_warning_when_no_test(skill):
    warns = skill.scaffold_warnings([_diff("server/index.js")])
    assert warns and warns[0].severity == "warning"


def test_scaffold_no_warning_when_test_present(skill):
    diffs = [_diff("server/index.js"), _diff("tests/app.test.js")]
    assert skill.scaffold_warnings(diffs) == []
