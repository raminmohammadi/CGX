"""Tests for the NextJsSkill: detection, prompt guidance, the App Router
server/client boundary check in validate_scaffold, and the paths-only
validate_plan (root layout + package.json + next config)."""

from __future__ import annotations

import pytest

import skills as registry
from skills.nextjs import NextJsSkill


@pytest.fixture()
def skill() -> NextJsSkill:
    return NextJsSkill()


def _diff(path, body=""):
    return {"file": path, "patch": body}


# --- detection -------------------------------------------------------------

@pytest.mark.parametrize("goal", [
    "build a Next.js dashboard",
    "make me a nextjs blog",
    "a next.js app with server routes",
])
def test_detects_nextjs_goals(skill, goal):
    assert skill.detect(goal) >= 0.9


def test_does_not_fire_on_unrelated_goal(skill):
    assert skill.detect("build a flask api") == 0.0
    assert skill.detect("") == 0.0


def test_registered_and_detected_via_registry():
    names = [s.name for s in registry.detect_skills("build a next.js app")]
    assert "nextjs" in names


# --- prompt guidance -------------------------------------------------------

def test_scaffold_prompt_teaches_router_choice_and_use_client(skill):
    p = skill.scaffold_system_prompt()
    # App Router vs Pages Router pick.
    assert "App Router" in p and "Pages Router" in p
    # server-vs-client components + the 'use client' directive rule.
    assert "'use client'" in p
    assert "Server Component" in p or "SERVER Component" in p
    assert "FIRST line" in p


# --- validate_scaffold: server/client boundary ----------------------------

_PKG = _diff("package.json", '{"dependencies": {"next": "^14"}}')
_LAYOUT = _diff("app/layout.tsx",
                "export default function RootLayout({children}) {"
                "return <html><body>{children}</body></html>}")


def test_client_component_without_directive_is_fatal(skill):
    page = _diff("app/page.tsx",
                 "import {useState} from 'react'\n"
                 "export default function Page(){\n"
                 "  const [n, setN] = useState(0)\n"
                 "  return <button>{n}</button>\n}")
    v = skill.validate_scaffold([page, _LAYOUT, _PKG])
    assert v is not None and not v.passed and v.severity == "error"
    assert "use client" in v.rationale and "app/page.tsx" in v.rationale


def test_onclick_handler_without_directive_is_fatal(skill):
    page = _diff("app/dashboard/page.tsx",
                 "export default function Page(){\n"
                 "  return <button onClick={() => alert('hi')}>go</button>\n}")
    v = skill.validate_scaffold([page, _LAYOUT, _PKG])
    assert v is not None and not v.passed
    assert "app/dashboard/page.tsx" in v.rationale


def test_client_component_with_directive_first_passes(skill):
    page = _diff("app/page.tsx",
                 "'use client'\n"
                 "import {useState} from 'react'\n"
                 "export default function Page(){\n"
                 "  const [n, setN] = useState(0)\n"
                 "  return <button onClick={() => setN(n + 1)}>{n}</button>\n}")
    assert skill.validate_scaffold([page, _LAYOUT, _PKG]) is None


def test_directive_after_imports_is_fatal(skill):
    # 'use client' must be the VERY FIRST line; below an import it does nothing.
    page = _diff("app/page.tsx",
                 "import {useEffect} from 'react'\n"
                 "'use client'\n"
                 "export default function Page(){ useEffect(() => {}, []) }")
    v = skill.validate_scaffold([page, _LAYOUT, _PKG])
    assert v is not None and not v.passed


def test_directive_after_license_block_comment_passes(skill):
    page = _diff("app/page.tsx",
                 "/**\n * (c) 2026 Acme\n */\n"
                 "'use client';\n"
                 "import {useRef} from 'react'\n"
                 "export default function Page(){ const r = useRef(null); "
                 "return <div ref={r}/> }")
    assert skill.validate_scaffold([page, _LAYOUT, _PKG]) is None


def test_pure_server_component_passes(skill):
    # async server component fetching data, no client features -> fine.
    page = _diff("app/page.tsx",
                 "export default async function Page(){\n"
                 "  const data = await fetch('https://api').then(r => r.json())\n"
                 "  return <main>{data.title}</main>\n}")
    assert skill.validate_scaffold([page, _LAYOUT, _PKG]) is None


def test_route_handler_using_request_is_not_flagged(skill):
    # API route handlers are server-only and never carry 'use client'.
    route = _diff("app/api/hello/route.ts",
                  "export async function GET(){ return Response.json({ok: true}) }")
    # add a page so structural checks pass, plus manifest.
    page = _diff("app/page.tsx", "export default function P(){ return <p>hi</p> }")
    assert skill.validate_scaffold([route, page, _PKG]) is None


def test_isomorphic_window_guard_not_flagged(skill):
    # window behind a typeof guard is isomorphic-safe; no 'use client' needed.
    mod = _diff("app/lib/env.ts",
                "export const isBrowser = typeof window !== 'undefined' "
                "&& !!window.location")
    page = _diff("app/page.tsx", "export default function P(){ return <p>hi</p> }")
    assert skill.validate_scaffold([mod, page, _PKG]) is None


def test_unguarded_window_use_is_fatal(skill):
    mod = _diff("app/widget.tsx",
                "export function W(){ document.title = 'x'; return null }")
    page = _diff("app/page.tsx", "export default function P(){ return <p>hi</p> }")
    v = skill.validate_scaffold([mod, page, _PKG])
    assert v is not None and not v.passed


def test_pages_router_hooks_are_not_flagged(skill):
    # Pages Router components are client by default; no directive required.
    page = _diff("pages/index.tsx",
                 "import {useState} from 'react'\n"
                 "export default function Home(){ const [n] = useState(0); "
                 "return <button onClick={() => {}}>{n}</button> }")
    assert skill.validate_scaffold([page, _PKG]) is None


# --- validate_scaffold: preserved structural checks ------------------------

def test_missing_router_files_is_fatal(skill):
    v = skill.validate_scaffold([_diff("README.md"), _diff("notes.txt")])
    assert v is not None and not v.passed
    assert "route files" in v.rationale


def test_missing_package_json_is_fatal(skill):
    v = skill.validate_scaffold([_diff("app/page.tsx", "export default () => null")])
    assert v is not None and not v.passed
    assert "package.json" in v.rationale


def test_empty_diffs_abstains(skill):
    assert skill.validate_scaffold([]) is None


# --- validate_plan ---------------------------------------------------------

def test_plan_app_page_without_layout_is_fatal(skill):
    v = skill.validate_plan([
        _diff("app/page.tsx"), _diff("package.json"), _diff("next.config.js"),
    ])
    assert v is not None and not v.passed
    assert "layout" in v.rationale.lower()


def test_plan_app_page_missing_manifest_and_config(skill):
    v = skill.validate_plan([_diff("app/page.tsx"), _diff("app/layout.tsx")])
    assert v is not None and not v.passed
    assert "package.json" in v.rationale
    assert "next.config" in v.rationale


def test_plan_complete_app_router_passes(skill):
    assert skill.validate_plan([
        _diff("app/layout.tsx"),
        _diff("app/page.tsx"),
        _diff("package.json"),
        _diff("next.config.mjs"),
    ]) is None


def test_plan_nested_layout_does_not_satisfy_root(skill):
    # A layout nested under a route is not the required ROOT layout.
    v = skill.validate_plan([
        _diff("app/page.tsx"),
        _diff("app/dashboard/layout.tsx"),
        _diff("package.json"),
        _diff("next.config.js"),
    ])
    assert v is not None and not v.passed and "layout" in v.rationale.lower()


def test_plan_pages_router_requires_manifest_and_config(skill):
    v = skill.validate_plan([_diff("pages/index.tsx")])
    assert v is not None and not v.passed
    assert "package.json" in v.rationale and "next.config" in v.rationale
    # Pages Router does NOT require an app/ root layout.
    assert "layout" not in v.rationale.lower()


def test_plan_pages_router_complete_passes(skill):
    assert skill.validate_plan([
        _diff("pages/index.tsx"),
        _diff("package.json"),
        _diff("next.config.js"),
    ]) is None


def test_plan_abstains_when_no_nextjs_source(skill):
    assert skill.validate_plan([_diff("app.py"), _diff("requirements.txt")]) is None
    assert skill.validate_plan([]) is None


# --- scaffold warnings -----------------------------------------------------

def test_warns_when_no_test_file(skill):
    warns = skill.scaffold_warnings([_diff("app/page.tsx"), _diff("package.json")])
    assert warns and warns[0].severity == "warning"


def test_no_warning_when_test_present(skill):
    warns = skill.scaffold_warnings([
        _diff("app/page.tsx"), _diff("package.json"),
        _diff("tests/page.test.tsx"),
    ])
    assert warns == []
