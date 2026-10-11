"""Render a :class:`~cgx.design.brief.DesignBrief` into prompt blocks.

Two consumers:

* :func:`planner_block` -- for the Tech Lead / manifest planner: the product
  spec (archetype, screens, components, UX requirements) plus the FILE PLAN a
  UI build must include, so the plan carries a design system, an app shell, a
  router, an API client and a storage helper instead of two bare components.
* :func:`file_block` -- for the per-file generator: guidance tailored to the
  kind of file being written (entry, app shell, component/page, stylesheet,
  index.html, API client, storage helper, package.json, test), with the theme
  rendered as hard constraints. Returns ``""`` for files the design layer does
  not own (backend Python, docs, ...).

Both are plain text sized for a 14B local model's context window.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from cgx.design.brief import DesignBrief
from cgx.design.themes import theme_by_key, theme_prompt_block

__all__ = ["planner_block", "file_block", "file_kind", "is_ui_file", "UI_SOURCE_EXT"]

UI_SOURCE_EXT = (".jsx", ".tsx", ".vue", ".svelte")
_STYLE_EXT = (".css", ".scss", ".sass", ".less")


def _brief(b: Any) -> DesignBrief:
    return b if isinstance(b, DesignBrief) else DesignBrief.from_dict(b)


def file_kind(brief: Any, path: str) -> str:
    """Classify ``path`` for prompt/check purposes.

    One of: ``tokens`` | ``stylesheet`` | ``html`` | ``entry`` | ``app`` |
    ``component`` | ``api`` | ``storage`` | ``package`` | ``config`` | ``test``
    | ``script`` | ``other``.
    """
    b = _brief(brief)
    p = (path or "").replace("\\", "/")
    low = p.lower()
    base = low.rsplit("/", 1)[-1]
    if p == b.tokens_path:
        return "tokens"
    if low.endswith(_STYLE_EXT):
        return "stylesheet"
    if re.search(r"\.(test|spec)\.(jsx?|tsx?|mjs)$", low) or "/__tests__/" in low or low.startswith("tests/"):
        if low.endswith((".js", ".jsx", ".ts", ".tsx", ".mjs")):
            return "test"
    if base == "package.json":
        return "package"
    if base in ("vite.config.js", "vite.config.ts", "vitest.config.js", "vitest.config.ts", "tsconfig.json",
                "postcss.config.js", "tailwind.config.js", "next.config.js", "next.config.mjs", "eslint.config.js"):
        return "config"
    if base.endswith(".html"):
        return "html"
    if p == b.entry_path:
        return "entry"
    if p == b.app_path:
        return "app"
    if p == b.api_client_path or re.search(r"/(lib|services|api|utils)/[^/]*(api|client|http|request)[^/]*\.(js|ts)$", low):
        return "api"
    if p == b.storage_path or re.search(r"/(lib|utils|hooks|store)/[^/]*(storage|settings|persist|store)[^/]*\.(js|ts)$", low):
        return "storage"
    if (b.sample_data_path and p == b.sample_data_path) or re.search(r"/(lib|data)/[^/]*(sample|seed|mock|fixture)[^/]*\.(js|ts|json)$", low):
        return "sample"
    if low.endswith(UI_SOURCE_EXT):
        return "component"
    if low.endswith((".js", ".ts", ".mjs")):
        if b.framework == "html" or "/components/" in low or "/pages/" in low or "/views/" in low or "/hooks/" in low:
            return "component" if ("/components/" in low or "/pages/" in low or "/views/" in low) else "script"
        return "script"
    return "other"


def is_ui_file(brief: Any, path: str) -> bool:
    """Files whose output budget and prompt the design layer should shape."""
    return file_kind(brief, path) in ("tokens", "stylesheet", "html", "entry", "app", "component",
                                      "api", "storage", "script", "sample")


def _lines(items: List[str], bullet: str = "- ") -> str:
    return "\n".join(f"{bullet}{it}" for it in items if it)


def _required_files(b: DesignBrief) -> List[str]:
    r = b.ui_dir()
    jsx = "tsx" if b.typescript else "jsx"
    js = "ts" if b.typescript else "js"
    files: List[str] = []
    if b.framework == "nextjs":
        files += [f"{b.tokens_path} -- design tokens + base styles (CGX generates this file; just list it)",
                  f"{b.entry_path} -- root layout importing globals.css, with the app shell (header/nav)",
                  f"{b.app_path} -- home page"]
        files += [f"{r}app/<screen>/page.{jsx} -- one route per screen"]
    elif b.framework == "vue":
        files += [f"{b.html_path} -- Vite entry (<div id=\"app\">, module script)",
                  f"{r}package.json -- vue ^3, vue-router ^4 (multi-screen), vite, @vitejs/plugin-vue, vitest",
                  f"{r}vite.config.{js} -- Vue plugin",
                  f"{b.tokens_path} -- design tokens + base styles (CGX generates this file; just list it)",
                  (f"{b.global_css_path} -- app-level layout/component styles using var(--...)" if b.global_css_path else ""),
                  f"{b.entry_path} -- createApp(App).use(router).mount('#app'); imports BOTH stylesheets",
                  f"{b.app_path} -- app shell: header/nav + <RouterView/>",
                  f"{r}src/router/index.{js} -- routes (one per screen)" if b.multi_screen else "",
                  f"{r}src/views/<Screen>.vue -- one per screen", f"{r}src/components/<Piece>.vue -- shared pieces"]
    elif b.framework == "html" and b.server_rendered:
        files += [f"{b.html_path} -- base template: <!doctype html>, viewport meta, title block, links the tokens stylesheet "
                  "then the app stylesheet from static/css, a semantic header/nav/main/footer with {% block content %}, "
                  "loads static/js/app.js with defer",
                  f"{r}templates/<page>.html -- one template per screen extending base.html",
                  f"{b.tokens_path} -- design tokens + base styles (CGX generates this file; just list it)",
                  f"{b.global_css_path} -- page/section/component styles using var(--...) with @media rules",
                  f"{b.entry_path} -- progressive-enhancement behaviour (nav toggle, forms, fetch to the app's own routes) as a plain ES module"]
    elif b.framework == "html":
        files += [f"{b.html_path} -- the page(s): semantic header/nav/main/footer, viewport meta, links css/tokens.css then css/styles.css, loads js/app.js with defer",
                  f"{b.tokens_path} -- design tokens + base styles (CGX generates this file; just list it)",
                  f"{b.global_css_path} -- page/section/component styles using var(--...) with @media rules",
                  f"{b.entry_path} -- behaviour (nav toggle, forms, state) as plain ES modules"]
    else:
        files += [f"{b.html_path} -- Vite entry (<div id=\"root\">, module script, viewport meta, title)",
                  f"{r}package.json -- react ^19, react-dom ^19{', react-router ^7' if b.multi_screen else ''}, vite, @vitejs/plugin-react, vitest + @testing-library/react + jsdom (devDependencies); scripts dev/build/preview/test",
                  f"{r}vite.config.{js} -- React plugin (+ test.environment 'jsdom' when tests are planned)",
                  f"{b.tokens_path} -- design tokens + base styles (CGX generates this file; just list it)",
                  (f"{b.global_css_path} -- app-level layout/component styles using var(--...)" if b.global_css_path else ""),
                  f"{b.entry_path} -- createRoot(...).render(<StrictMode>{'<BrowserRouter>' if b.multi_screen else ''}<App/>...); imports the stylesheet(s)",
                  f"{b.app_path} -- app shell: header/nav{' + <Routes>' if b.multi_screen else ''}, page container",
                  f"{r}src/pages/<Screen>.{jsx} -- one per screen (named after the screen)",
                  f"{r}src/components/<Piece>.{jsx} -- shared UI pieces (each with a sibling .css when it needs own styles)"]
    if b.external_api:
        files.append(f"{b.api_client_path} -- the ONLY module that calls fetch: reads base URL + key from storage, "
                     "builds headers, normalises errors")
    if b.persistence:
        files.append(f"{b.storage_path} -- load/save helpers over localStorage (settings, user data) with safe JSON parsing")
    if b.sample_data_path:
        files.append(f"{b.sample_data_path} -- realistic seed data (12-30 records: names, categories, ISO dates, amounts) "
                     "exported as arrays + helpers; the storage helper seeds it on first run so screens render populated")
    return [f for f in files if f]


def _planner_requirements(b: DesignBrief) -> List[str]:
    """Archetype-specific requirements first, then a few universal ones (bounded)."""
    from cgx.design.brief import UNIVERSAL_REQUIREMENTS
    universal = set(UNIVERSAL_REQUIREMENTS)
    specific = [r for r in b.requirements if r not in universal]
    general = [r for r in b.requirements if r in universal]
    return specific[:8] + general[:4]


def planner_block(brief: Any) -> str:
    """The product + design spec and file-plan requirements for the planner."""
    b = _brief(brief)
    if not b.applies:
        return ""
    theme = theme_by_key(b.theme)
    fw = {"react": "React 19 + Vite", "vue": "Vue 3 + Vite", "nextjs": "Next.js (app router)",
          "html": ("server-rendered HTML templates + static CSS/JS" if b.server_rendered
                   else "plain HTML + CSS + ES-module JS (no build step)")}.get(b.framework, b.framework)
    styling = ("Tailwind v4 via the generated @theme tokens" if b.styling == "tailwind"
               else "a generated design-tokens stylesheet (CSS variables + base component classes)")
    root_note = (f"The UI component lives under '{b.app_root}/' (it has its own package.json/index.html there)."
                 if b.app_root else "The UI component lives at the project root.")
    parts = [
        f"UI/UX PRODUCT SPEC (binding -- the plan MUST realise all of it):",
        f"- Product: {b.product_name}. Archetype: {b.archetype_name}. Stack: {fw}; styling: {styling}; "
        f"theme: {theme.name if theme else b.theme}.",
        f"- Layout: {b.layout}",
        f"- Screens: {'; '.join(b.screens)}.",
        f"- Components to plan as real files: {'; '.join(c.split(' (')[0] for c in b.components)}.",
        "- UX requirements the implementation will be CHECKED against:\n" + _lines(_planner_requirements(b), "  * "),
    ]
    if b.integration_notes:
        parts.append("- Integration rules:\n" + _lines(b.integration_notes, "  * "))
    parts.append(f"FILE PLAN REQUIREMENTS -- {root_note} Include these files (exact paths) plus one file per screen "
                 "and per reusable component:\n" + _lines(_required_files(b), "  * "))
    parts.append("Do NOT plan a Python backend, database, auth layer or tests the objective did not ask for. "
                 "In contracts, declare the storage shape (schemas: settings/data keys) and the API client functions "
                 "(functions) so every screen uses the same names.")
    return "\n".join(parts)


def _common_ui_rules(b: DesignBrief) -> str:
    return (
        "UI COMPLETENESS (overrides 'add nothing unasked' for UI files): a page/component must be COMPLETE and "
        "POLISHED -- full markup with semantic structure, all handlers wired, loading/empty/error/success states, "
        "accessible labels, and styling hooks (className) on every element group. Aim for a real product screen, "
        "not a demo. Typical size: 60-220 lines. Never leave a handler that only console.logs, never call "
        "alert(), never ship placeholder copy or TODOs."
    )


def _component_hints(b: DesignBrief, path: str) -> str:
    base = path.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
    hints: List[str] = []
    screens_low = " ".join(b.screens).lower()
    if any(k in base for k in ("chat", "message", "conversation")) and b.archetype == "chat-assistant":
        hints.append("Chat screen: `main.chat` column with a scrollable `.message-list` (auto-scroll via a ref on "
                     "the last message), `.bubble.user` right-aligned in primary, `.bubble.assistant` left-aligned on "
                     "surface with border; a typing indicator while awaiting; an empty state with a headline and 3 "
                     "suggestion chips when there are no messages; a sticky `.composer` with an auto-growing textarea "
                     "(Enter sends, Shift+Enter newline) and a `.btn.btn-primary` Send button that shows a spinner "
                     "and is disabled while sending or when settings are missing.")
    if any(k in base for k in ("setting", "profile", "config", "preference")):
        hints.append("Settings screen: `.container-narrow` with a `.card`; each input in a `.field` with a `.label` "
                     "and `.hint`; the key field is type=password with a show/hide toggle button (aria-label); Save "
                     "persists via the storage helper and shows a `.alert.alert--success` confirmation; a 'Test "
                     "connection' button calls the API client and reports the result inline; validate the URL "
                     "(must start with http) before saving.")
    if any(k in base for k in ("app", "shell", "layout", "nav", "header", "topbar", "sidebar")):
        hints.append("App shell: `<header class=\"topbar\">` with `.brand` (mark + product name) and a `<nav "
                     "class=\"nav\">` of links (`.nav-link`, active class on the current route), then `<main "
                     "class=\"main\">` holding the routed screen; show a connection/status badge in the header when "
                     "the app has settings.")
    if any(k in base for k in ("list", "table", "grid", "board", "column")):
        hints.append("Collections: render a `.empty-state` (icon/headline/hint/action) when there are no items, "
                     "skeleton rows while loading, and an `.alert.alert--error` with Retry on failure.")
    if "form" in base or "wizard" in base or "step" in base:
        hints.append("Forms: every input in a `.field` with `.label`; inline validation messages in "
                     "`.hint` (or `.alert--error`); submit button disabled while pending; success feedback inline.")
    if not hints and "settings" in screens_low and b.multi_screen:
        hints.append("This is one screen of a multi-screen app: give it a `.page-header` (title + subtitle) and "
                     "compose from `.card` blocks; use the shared storage/API modules rather than fetch/localStorage directly.")
    return "\n".join(f"- {h}" for h in hints)


def _stylesheet_hints(b: DesignBrief) -> str:
    """Archetype-specific layout classes the app stylesheet must define."""
    hints = {
        "analytics-dashboard": "LAYOUT CLASSES TO DEFINE: .kpi-grid { display:grid; grid-template-columns: repeat(auto-fit, "
                               "minmax(220px, 1fr)); gap: var(--space-4) } (tiles side by side on desktop, never stacked), "
                               ".kpi-tile (label small muted uppercase, value var(--text-3xl) tabular-nums, delta badge), "
                               ".chart-card with a .chart-area of fixed height 280-320px and an SVG that fills it, "
                               ".chart-legend row, .table-wrap { overflow-x:auto }, .filter-bar (flex, wraps, gap).",
        "admin-dashboard": "LAYOUT CLASSES TO DEFINE: .app-shell--sidebar grid (sidebar + content), .stat-grid "
                           "(auto-fit minmax(220px,1fr)), .toolbar (search + filters + primary action in one row), "
                           ".table-wrap { overflow-x:auto } with sticky th, .row-actions, .drawer/.modal panel styles.",
        "chat-assistant": "LAYOUT CLASSES TO DEFINE: .topbar spans the FULL viewport width (the chat column is centred "
                          "INSIDE main, not the header); .chat { display:grid; grid-template-rows: 1fr auto; height: "
                          "calc(100vh - var(--header-height)) }, .message-list (overflow-y:auto), .bubble/.bubble.user/"
                          ".bubble.assistant (max-width 72%, radius, fills), .composer (sticky bottom card: textarea + "
                          "button aligned on one row with gap, the textarea growing, the button never overflowing).",
        "kanban-productivity": "LAYOUT CLASSES TO DEFINE: .board { display:grid; grid-auto-flow:column; grid-auto-columns: "
                               "minmax(280px, 1fr); gap: var(--space-4); overflow-x:auto }, .column (surface, header with "
                               "count badge), .task-card (hover lift), .add-form inline row.",
        "landing-marketing": "LAYOUT CLASSES TO DEFINE: .hero (min-height 70vh grid, gradient background from the theme), "
                             ".section (padding var(--space-16) 0), .features-grid (auto-fit minmax(260px,1fr)), "
                             ".pricing-grid, .testimonial, .cta-band, .footer-grid.",
        "ecommerce-catalog": "LAYOUT CLASSES TO DEFINE: .catalog { display:grid; grid-template-columns: 240px 1fr } "
                             "(stacks under 900px), .product-grid (auto-fill minmax(220px,1fr)), .product-card with a "
                             "fixed-ratio .media box (gradient placeholder), .cart-drawer (fixed right panel).",
        "developer-tool": "LAYOUT CLASSES TO DEFINE: .workspace { display:grid; grid-template-columns: 1fr 1fr } (stacks "
                          "under 900px), .pane (surface, header, mono body), .toolbar, .status-bar (sticky bottom, mono).",
    }
    return hints.get(b.archetype, "")


def file_block(brief: Any, path: str, description: str = "") -> str:
    """Design guidance for generating ``path``; ``""`` when not a UI file."""
    b = _brief(brief)
    if not b.applies:
        return ""
    kind = file_kind(b, path)
    if kind in ("other", "test", "config") or kind == "tokens":
        return ""
    theme = theme_by_key(b.theme)
    theme_block = theme_prompt_block(theme, styling=b.styling) if theme else ""
    tokens_rel = b.tokens_path
    css_rel = b.global_css_path
    jsx = "tsx" if b.typescript else "jsx"
    head = (f"PRODUCT CONTEXT: {b.product_name} -- {b.archetype_name}. Screens: {'; '.join(b.screens)}. "
            f"Layout: {b.layout}")
    blocks: List[str] = [head]

    if kind == "package":
        deps = ["\"react\": \"^19.0.0\"", "\"react-dom\": \"^19.0.0\""]
        if b.multi_screen:
            deps.append("\"react-router\": \"^7.0.0\"")
        if b.framework == "vue":
            deps = ["\"vue\": \"^3.5.0\""] + (["\"vue-router\": \"^4.4.0\""] if b.multi_screen else [])
        dev = (["\"vite\": \"^6.0.0\"", "\"@vitejs/plugin-vue\": \"^5.0.0\""] if b.framework == "vue"
               else ["\"vite\": \"^6.0.0\"", "\"@vitejs/plugin-react\": \"^4.3.0\""])
        dev += ["\"vitest\": \"^3.0.0\"", "\"jsdom\": \"^26.0.0\"", "\"@testing-library/react\": \"^16.0.0\"",
                "\"@testing-library/jest-dom\": \"^6.0.0\""] if b.framework != "vue" else ["\"vitest\": \"^3.0.0\"", "\"jsdom\": \"^26.0.0\"", "\"@vue/test-utils\": \"^2.4.0\""]
        if b.styling == "tailwind":
            dev += ["\"tailwindcss\": \"^4.0.0\"", "\"@tailwindcss/vite\": \"^4.0.0\""]
        blocks.append("PACKAGE RULES: dependencies: " + ", ".join(deps) + ". devDependencies: " + ", ".join(dev)
                      + ". scripts: {\"dev\": \"vite\", \"build\": \"vite build\", \"preview\": \"vite preview\", "
                        "\"test\": \"vitest run\"}. \"type\": \"module\". Exact JSON, no comments.")
        return "\n\n".join(blocks)

    if kind == "html":
        if b.framework == "html" and b.server_rendered:
            blocks.append(
                "TEMPLATE RULES: a base template owns <!doctype html>, <html lang=\"en\">, <meta charset>, <meta "
                "name=\"viewport\" content=\"width=device-width, initial-scale=1\">, a {% block title %}, links the "
                "tokens stylesheet THEN the app stylesheet via the framework's static helper (url_for('static', ...) / "
                "{% static %}), <header class=\"topbar\"> (brand + <nav class=\"nav\">), <main class=\"main\">{% block "
                "content %}{% endblock %}</main>, <footer>, and <script type=\"module\" src=... defer>. Page templates "
                "extend it and fill the blocks with real copy, .card groups, labelled .field inputs and .btn.btn-primary "
                "actions; render empty states when a list is empty.")
        elif b.framework == "html":
            blocks.append(
                "HTML PAGE RULES: <!doctype html>, <html lang=\"en\">, <meta charset>, <meta name=\"viewport\" "
                "content=\"width=device-width, initial-scale=1\">, a specific <title>, <meta name=\"theme-color\">; "
                f"link css/tokens.css THEN css/styles.css; <script type=\"module\" src=\"js/app.js\" defer>. Body: "
                "<header class=\"topbar\"> (brand + <nav class=\"nav\"> with anchor/page links), <main> with "
                "sections (each with an id, an h2 and real copy), <footer>. Use the tokens classes (.container, "
                ".btn.btn-primary, .card, .grid, .field, .label, .input). No local image files -- use inline SVG or "
                "CSS gradients for visuals. Forms have labels and a success message element.")
        else:
            blocks.append(
                f"INDEX.HTML RULES: <!doctype html>, <html lang=\"en\">, <meta charset=\"UTF-8\">, <meta name=\"viewport\" "
                f"content=\"width=device-width, initial-scale=1.0\">, <title>{b.product_name}</title>, <meta "
                f"name=\"theme-color\">, <body><div id=\"{'app' if b.framework == 'vue' else 'root'}\"></div><script "
                f"type=\"module\" src=\"/src/{'main.' + ('ts' if b.typescript else 'js') if b.framework == 'vue' else 'main.' + jsx}\"></script>. "
                "Nothing else -- fonts and styles come from the stylesheet imported by the entry module.")
        blocks.append(theme_block)
        return "\n\n".join(blocks)

    if kind == "entry":
        if b.framework == "vue":
            blocks.append(f"ENTRY RULES: import {tokens_rel.split('/', 1)[-1] if b.app_root else tokens_rel} and "
                          f"{css_rel or ''} via relative paths ('./styles/tokens.css', './styles/app.css'), create the "
                          "app with createApp(App), install the router when present, mount('#app').")
        elif b.framework == "nextjs":
            blocks.append("LAYOUT RULES: import './globals.css'; export metadata {title, description}; render <html "
                          "lang=\"en\"><body> with the app shell (header/nav) and {children} inside <main class=\"main\">.")
        elif b.framework == "html":
            blocks.append("APP SCRIPT RULES: plain ES module; wire the nav toggle, forms (validate, persist via the "
                          "storage module, show success), and any interactive sections; no framework imports.")
        else:
            router = "import { BrowserRouter } from 'react-router'; wrap <App /> in <BrowserRouter>; " if b.multi_screen else ""
            css_imports = f"import './styles/tokens.css'; " + (f"import './styles/app.css'; " if css_rel else "")
            if b.styling == "tailwind":
                css_imports = "import './index.css'; "
            blocks.append(
                "ENTRY RULES (React 19): import { StrictMode } from 'react'; import { createRoot } from "
                f"'react-dom/client'; {css_imports}{router}createRoot(document.getElementById('root')).render(<StrictMode>"
                "...</StrictMode>). NEVER use ReactDOM.render (removed in React 19 -- the page would stay blank).")
        blocks.append(theme_block)
        return "\n\n".join(blocks)

    if kind == "app":
        if b.framework == "react":
            nav = ("Routes: import { Routes, Route, NavLink } from 'react-router'; one <Route> per screen under "
                   "<main class=\"main\">; <NavLink className=\"nav-link\"> for each screen (react-router adds the "
                   "'active' class automatically); unknown routes redirect home."
                   if b.multi_screen else "Single screen: render the page inside <main className=\"main\">.")
            blocks.append(
                f"APP SHELL RULES: <div className=\"app-shell\"><header className=\"topbar\"><div className=\"brand\">"
                f"<span className=\"brand-mark\" aria-hidden=\"true\"/>{b.product_name}</div><nav className=\"nav\" "
                f"aria-label=\"Primary\">...</nav>{'<ConnectionStatus/>' if b.external_api else ''}</header>...</div>. {nav} "
                "Load persisted settings/data on mount (storage helper) and pass them down or expose via context.")
        elif b.framework == "vue":
            blocks.append("APP SHELL RULES: <header class=\"topbar\"> with .brand and <nav class=\"nav\"> of "
                          "<RouterLink class=\"nav-link\" active-class=\"active\">, then <main class=\"main\"><RouterView/></main>.")
        blocks.append(_common_ui_rules(b))
        blocks.append(theme_block)
        return "\n\n".join(blocks)

    if kind == "component":
        rules = [
            _common_ui_rules(b),
            "Use the shared storage module for persistence and the shared API client for requests -- do not call "
            "localStorage or fetch directly inside a screen unless this file IS that module.",
            "Structure: a root element with a descriptive className, `.page-header` (h1/h2 + subtitle) for screens, "
            "`.card` groups, `.field/.label/.input` for inputs, `.btn.btn-primary` for the main action, `.btn.btn-ghost` "
            "for secondary actions, `.empty-state` / `.spinner` / `.alert.alert--error` for states.",
            "If this component needs styles beyond the shared classes, import a sibling stylesheet named after it "
            "(e.g. './Chat.css') ONLY if that file is in the PROJECT MANIFEST; otherwise put the rules in the shared "
            "app stylesheet's classes and reuse them.",
        ]
        hints = _component_hints(b, path)
        if hints:
            rules.append("Screen-specific guidance:\n" + hints)
        if b.integration_notes and file_kind(b, path) != "api":
            rules.append("Integration: " + " ".join(b.integration_notes[:1]))
        blocks.append("COMPONENT RULES:\n" + _lines(rules))
        blocks.append("UX REQUIREMENTS (checked):\n" + _lines(b.requirements[13:22] if len(b.requirements) > 13 else b.requirements))
        blocks.append(theme_block)
        return "\n\n".join(blocks)

    if kind == "stylesheet":
        blocks.append(_stylesheet_hints(b))
        blocks.append(
            f"STYLESHEET RULES: the tokens file ({tokens_rel}) already defines :root variables, base element styles, "
            "focus rings and the classes .container .stack .row .grid .topbar .brand .nav .nav-link .sidebar .main "
            ".page-header .card .btn .btn-primary .btn-ghost .btn-danger .field .label .hint .input .textarea "
            ".badge .alert .empty-state .spinner .skeleton .table. Do NOT redefine those or any colour. Write the "
            "APP-SPECIFIC layout and component styles for this product (e.g. .chat, .message-list, .bubble.user, "
            ".bubble.assistant, .composer, .typing-dots, .settings-card, .kpi-grid ...) using ONLY var(--color-*), "
            "var(--space-*), var(--radius-*), var(--shadow-*), var(--font-*), var(--text-*). Include hover/focus/"
            "disabled states, a @media (max-width: 768px) block, and keyframes for any animation. Target 80-200 lines.")
        blocks.append(theme_block)
        return "\n\n".join(blocks)

    if kind == "api":
        notes = " ".join(b.integration_notes) if b.integration_notes else (
            "Read the base URL (and key, if any) from the storage helper; build JSON requests with "
            "'Content-Type: application/json' and 'Authorization: Bearer <key>' when a key exists.")
        blocks.append(
            "API CLIENT RULES: this is the ONLY module that calls fetch. Export small async functions the screens "
            "use (e.g. sendMessage(messages, settings), testConnection(settings)). " + notes +
            " Normalise errors: throw an Error whose message includes the HTTP status and the response text "
            "(trimmed); handle network failures ('Could not reach <url>'); never log the key; support an "
            "AbortController timeout (~30s). No React imports here.")
        return "\n\n".join(blocks)

    if kind == "storage":
        blocks.append(
            "STORAGE RULES: thin, safe helpers over window.localStorage with a namespaced key prefix: "
            "loadSettings()/saveSettings(partial)/clearSettings(), plus load/save for the app's data (e.g. "
            "conversation history, items). JSON.parse inside try/catch with defaults; export the default settings "
            "shape; guard for environments without localStorage. No React imports; no fetch.")
        return "\n\n".join(blocks)

    if kind == "sample":
        blocks.append(
            "SAMPLE DATA RULES: export realistic, internally consistent seed data for this product (12-30 records "
            "with real-looking names/titles, 4-8 categories, ISO dates spread over the last 90 days, plausible amounts "
            "or counts as numbers, stable string ids) plus small helpers (e.g. totals by category, by month). Plain "
            "ES module, no React, no fetch. The storage helper seeds this on first run; screens never show 0/NaN.")
        return "\n\n".join(blocks)
    if kind == "script":
        blocks.append(_common_ui_rules(b))
        blocks.append(theme_block)
        return "\n\n".join(blocks)
    return ""
