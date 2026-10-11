"""The design brief: one deterministic product + design spec per UI build.

``build_brief(goal, skills)`` turns a plain-language objective into a
:class:`DesignBrief` -- which framework and styling approach to use, which
product archetype applies (screens, components, UX requirements), which curated
theme, and the canonical file paths the UI must ship (tokens stylesheet, entry,
app shell, API client, storage helper). Everything is derived from the goal
text and the active skills, never from the model, so a weak model receives the
same high-quality constraints every run.

The brief is a plain dict on the wire (``to_dict`` / ``from_dict``) so it can be
persisted in the WORK_PLAN artifact and threaded to the Developer and the
Verifier unchanged.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from cgx.design.archetypes import GENERIC, archetype_by_key, select_archetype
from cgx.design.themes import select_theme, theme_by_key

__all__ = ["DesignBrief", "applies", "build_brief", "detect_framework",
           "UNIVERSAL_REQUIREMENTS", "FRONTEND_SKILLS"]

FRONTEND_SKILLS = ("react", "nextjs", "vue", "tailwind")

# Goal phrasing that marks a UI-bearing build even when no framework is named.
_UI_RE = re.compile(
    r"\b(ui|gui|frontend|front-end|front end|web ?app|webapp|website|web site|site|landing ?page|"
    r"page|pages|dashboard|interface|screen|screens|tab|tabs|form|forms|chat|chatbot|browser|"
    r"html|css|react|vue|next\.?js|svelte|tailwind|component|components|widget|portal|"
    r"single.page|spa|responsive|mobile.friendly|app for|an app|the app|web application)\b",
    re.IGNORECASE)
# A bare "app" is UI-bearing unless the goal is clearly server/CLI-only
# ("a flask app", "a cli app").
_APP_RE = re.compile(r"\bapps?\b|\bapplication\b", re.IGNORECASE)
_BACKEND_ONLY_RE = re.compile(
    r"\b(backend|back-end|server|rest api|api server|flask|fastapi|django|express|"
    r"microservice|worker|cli|command.line|library|sdk|script|daemon|graphql)\b", re.IGNORECASE)
# Goals that are explicitly NOT a UI deliverable.
_NO_UI_RE = re.compile(
    r"\b(cli|command.line|library|sdk|package|rest api only|api only|backend only|microservice|"
    r"cron|daemon|script|scraper|etl|pipeline|data pipeline|lambda|serverless function|"
    r"terminal app|tui)\b", re.IGNORECASE)
_BACKEND_RE = re.compile(r"\b(backend|server|flask|fastapi|django|express|node\.?js|python api|rest api|"
                         r"graphql|database|postgres|sqlite|mongo)\b", re.IGNORECASE)
_HTML_RE = re.compile(r"\b(static|plain html|html/css|html css|vanilla js|no framework|landing ?page|"
                      r"website|web site|brochure|portfolio site)\b", re.IGNORECASE)
_EXTERNAL_API_RE = re.compile(
    r"\b(api (address|url|endpoint|base|key|token)|base url|endpoint|api key|access token|bearer|"
    r"openai|anthropic|gemini|ollama|llm|third.party api|external api|rest endpoint|webhook)\b",
    re.IGNORECASE)
_SECRET_RE = re.compile(r"\b(api key|secret|token|password|credential)s?\b", re.IGNORECASE)
_PERSIST_RE = re.compile(r"\b(save|saved|persist|remember|store|stored|history|local ?storage|"
                         r"offline|profile|settings|preferences|bookmark|favorites?)\b", re.IGNORECASE)
_MULTI_RE = re.compile(r"\b(tab|tabs|pages|screens|navigate|navigates|navigation|sidebar|menu|"
                       r"settings|profile|login|dashboard|detail|wizard|steps?)\b", re.IGNORECASE)
_TAILWIND_RE = re.compile(r"\btailwind(?:css)?\b", re.IGNORECASE)
# Server frameworks that render their own HTML (Jinja/Django templates, EJS):
# with no frontend framework named, the UI is server-rendered templates +
# static CSS/JS, not a React build.
_SERVER_FW_RE = re.compile(r"\b(flask|fastapi|django|express|rails|laravel|php|jinja)\b", re.IGNORECASE)
# Explicit UI words (a bare "app" is handled separately -- see ``applies``).
_STRONG_UI_RE = re.compile(
    r"\b(ui|gui|frontend|front-end|front end|web ?app|webapp|website|web site|landing ?page|"
    r"page|pages|dashboard|interface|screen|screens|tab|tabs|form|forms|chat|chatbot|browser|"
    r"html|css|react|vue|next\.?js|svelte|tailwind|component|components|widget|portal|"
    r"single.page|spa|responsive|mobile.friendly|web application|templates?)\b", re.IGNORECASE)
_TS_RE = re.compile(r"\b(typescript|tsx|\.ts)\b", re.IGNORECASE)

# Rules every UI build follows regardless of archetype -- imperative, concrete,
# and (mostly) checkable from source text by :mod:`cgx.design.checks`.
UNIVERSAL_REQUIREMENTS: List[str] = [
    "Ship a real app shell: a header/top bar with the product name and navigation, and a centred content "
    "container -- never a bare <div> holding raw controls",
    "Every screen has a visible title and clear hierarchy (heading, supporting text, grouped controls, cards)",
    "All styling goes through the generated design tokens and stylesheet classes; no unstyled HTML controls, "
    "no inline style objects for layout, no hard-coded hex colours in components",
    "Every async operation and every list has explicit loading, empty, error and success states in the UI",
    "Every input has a visible <label> (or aria-label for icon/search inputs); placeholders are hints, not labels; "
    "secrets use type=password with a show/hide toggle",
    "Forms validate before submit, disable the submit button while pending, and show success/error feedback "
    "inline -- never alert(), never console.log as the only effect of a handler",
    "User settings and user-created data persist (localStorage unless a backend is declared) and restore on load",
    "Multi-screen apps use a router (or an explicit tab/nav pattern) with the active item highlighted",
    "Responsive: usable at 375px and 1280px; stacks under 768px; no horizontal overflow",
    "Accessible: semantic landmarks (header/nav/main), focus-visible styles, aria-labels on icon-only buttons, "
    "sufficient contrast (guaranteed by the tokens)",
    "Buttons have hover/active/disabled styles and ~160ms transitions; destructive actions confirm",
    "Copy is real and specific to the product -- no lorem ipsum, no 'Click here', no TODO/placeholder text",
    "Keyboard works: Enter submits/sends, Escape closes dialogs, focus is managed after navigation",
    "Exactly one h1 per screen; multi-screen apps include a catch-all route that redirects to the main screen",
    "Secrets are never logged; show a masked value (last 4 characters) where a key must be displayed",
]


@dataclass
class DesignBrief:
    applies: bool = False
    framework: str = "react"          # react | vue | nextjs | html
    styling: str = "css-tokens"       # css-tokens | tailwind
    typescript: bool = False
    product_name: str = "App"
    archetype: str = GENERIC.key
    archetype_name: str = GENERIC.name
    layout: str = ""
    screens: List[str] = field(default_factory=list)
    components: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    integration_notes: List[str] = field(default_factory=list)
    theme: str = "clean-tech"
    app_root: str = ""                # "" (repo root) or e.g. "frontend"
    multi_screen: bool = True
    external_api: bool = False
    persistence: bool = True
    secrets: bool = False
    #: HTML rendered by a server framework (Flask/Django/FastAPI templates):
    #: tokens/styles under static/, pages under templates/.
    server_rendered: bool = False
    # canonical UI files (relative to the repo root)
    tokens_path: str = ""
    global_css_path: str = ""
    html_path: str = ""
    entry_path: str = ""
    app_path: str = ""
    api_client_path: str = ""
    storage_path: str = ""
    #: Realistic seed data for data-bearing archetypes with no backend (so the
    #: first render shows a populated dashboard/table, never "0 / NaN%").
    sample_data_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "DesignBrief":
        if not isinstance(d, dict):
            return cls()
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    # ---- path conventions -------------------------------------------------
    def with_app_root(self, app_root: str) -> "DesignBrief":
        """Return a copy with the UI component root (and derived paths) set."""
        b = DesignBrief.from_dict(self.to_dict())
        b.app_root = (app_root or "").strip("/")
        _assign_paths(b)
        return b

    def ui_dir(self) -> str:
        """Prefix (with trailing slash) under which the UI component lives."""
        return (self.app_root + "/") if self.app_root else ""

    def resolved_for_paths(self, paths: Iterable[str]) -> "DesignBrief":
        """Re-anchor the brief on a concrete plan: UI root + JS/TS flavour.

        Call once the planner has produced paths so the canonical files land
        inside the UI component (``frontend/src/...`` when the app lives under
        ``frontend/``) and match the plan's ``.tsx``/``.jsx`` choice.
        """
        ps = [str(p).replace("\\", "/") for p in (paths or [])]
        b = DesignBrief.from_dict(self.to_dict())
        if not ps or not b.applies:
            return b
        b.app_root = _ui_root_from_paths(ps)
        if any(p.endswith((".ts", ".tsx")) for p in ps):
            b.typescript = True
        _assign_paths(b)
        return b


#: Archetypes whose screens are data-bearing: they get a seeded sample-data
#: module when no backend is declared, so the first render is populated.
DATA_ARCHETYPES = ("analytics-dashboard", "admin-dashboard", "ecommerce-catalog",
                   "kanban-productivity", "content-site")


def _assign_paths(b: DesignBrief) -> None:
    r = b.ui_dir()
    jsx = "tsx" if b.typescript else "jsx"
    js = "ts" if b.typescript else "js"
    seed = b.archetype in DATA_ARCHETYPES and not b.external_api
    if b.framework == "nextjs":
        b.tokens_path = f"{r}app/globals.css"
        b.global_css_path = ""
        b.html_path = ""
        b.entry_path = f"{r}app/layout.{jsx}"
        b.app_path = f"{r}app/page.{jsx}"
        b.api_client_path = f"{r}lib/api.{js}"
        b.storage_path = f"{r}lib/storage.{js}"
        b.sample_data_path = f"{r}lib/sampleData.{js}" if seed else ""
    elif b.framework == "vue":
        b.tokens_path = f"{r}src/styles/tokens.css" if b.styling != "tailwind" else f"{r}src/index.css"
        b.global_css_path = f"{r}src/styles/app.css" if b.styling != "tailwind" else ""
        b.html_path = f"{r}index.html"
        b.entry_path = f"{r}src/main.{js}"
        b.app_path = f"{r}src/App.vue"
        b.api_client_path = f"{r}src/lib/api.{js}"
        b.storage_path = f"{r}src/lib/storage.{js}"
        b.sample_data_path = f"{r}src/lib/sampleData.{js}" if seed else ""
    elif b.framework == "html" and b.server_rendered:
        b.tokens_path = f"{r}static/css/tokens.css"
        b.global_css_path = f"{r}static/css/styles.css"
        b.html_path = f"{r}templates/base.html"
        b.entry_path = f"{r}static/js/app.js"
        b.app_path = f"{r}templates/base.html"
        b.api_client_path = f"{r}static/js/api.js"
        b.storage_path = f"{r}static/js/storage.js"
    elif b.framework == "html":
        b.tokens_path = f"{r}css/tokens.css"
        b.global_css_path = f"{r}css/styles.css"
        b.html_path = f"{r}index.html"
        b.entry_path = f"{r}js/app.js"
        b.app_path = f"{r}index.html"
        b.api_client_path = f"{r}js/api.js"
        b.storage_path = f"{r}js/storage.js"
    else:  # react (Vite)
        b.tokens_path = f"{r}src/styles/tokens.css" if b.styling != "tailwind" else f"{r}src/index.css"
        b.global_css_path = f"{r}src/styles/app.css" if b.styling != "tailwind" else ""
        b.html_path = f"{r}index.html"
        b.entry_path = f"{r}src/main.{jsx}"
        b.app_path = f"{r}src/App.{jsx}"
        b.api_client_path = f"{r}src/lib/api.{js}"
        b.storage_path = f"{r}src/lib/storage.{js}"
        b.sample_data_path = f"{r}src/lib/sampleData.{js}" if seed else ""


def detect_framework(goal: str, skills: Iterable[str]) -> str:
    """Resolve the UI framework: named skill > named framework > sensible default.

    With no framework named, a *website/landing page* defaults to plain
    HTML/CSS/JS (zero toolchain risk) and an *app* defaults to React + Vite (the
    industry default a senior engineer would reach for).
    """
    names = {str(s).lower() for s in (skills or [])}
    low = (goal or "").lower()
    if "nextjs" in names or re.search(r"\bnext\.?js\b", low):
        return "nextjs"
    if "vue" in names or re.search(r"\bvue(?:\.js)?\b", low):
        return "vue"
    if "react" in names or re.search(r"\breact(?:\.?js)?\b", low):
        return "react"
    if _SERVER_FW_RE.search(low) or (names & {"flask", "fastapi", "django", "express"}):
        return "html"  # server-rendered templates + static assets
    if _HTML_RE.search(low) and not re.search(r"\b(app|dashboard|tool|chat)\b", low):
        return "html"
    if re.search(r"\b(html|static)\b", low) and not re.search(r"\b(react|vue|next)\b", low):
        return "html"
    return "react"


def applies(goal: str, skills: Optional[Iterable[str]] = None) -> bool:
    """True when the objective has a user-facing UI the design layer should own."""
    names = {str(s).lower() for s in (skills or [])}
    if names & set(FRONTEND_SKILLS):
        return True
    low = goal or ""
    strong = bool(_STRONG_UI_RE.search(low))
    if _NO_UI_RE.search(low) and not strong:
        return False
    # "a flask app for todos" / "a fastapi service": server-side unless the goal
    # also names a UI (pages, templates, frontend, dashboard ...).
    if _BACKEND_ONLY_RE.search(low) and not strong:
        return False
    if strong or _UI_RE.search(low):
        return True
    # "a todo app" / "an expense tracker app": an app is a UI.
    return bool(_APP_RE.search(low))


_NAME_STOP = {"main", "home", "new", "this", "that", "profile", "settings", "web", "simple", "basic", "app",
              "page", "user", "users", "the", "my", "our", "your", "chat page", "main page", "landing", "first",
              "single", "another", "same", "second", "front", "frontend", "back", "backend"}


def _clean_name(raw: str) -> str:
    cand = re.sub(r"\b(simple|basic|small|new|modern|react|vue|web|nice|beautiful|fully|functional|responsive)\b",
                  "", raw, flags=re.I)
    cand = re.sub(r"\s+", " ", cand).strip(" -")
    return cand


def _product_name(goal: str, archetype_name: str) -> str:
    """A short, title-cased product name derived from the goal (never a generic word)."""
    low = (goal or "").strip()
    m = re.search(r"\b(?:called|named|titled)\s+[\"']?([A-Za-z][\w\-]*(?:\s+[A-Z][\w\-]*){0,2})", low)
    if m:
        cand = _clean_name(m.group(1))
        if cand and cand.lower() not in _NAME_STOP:
            return cand if cand[:1].isupper() else cand.title()
    patterns = (
        r"\b(?:app|application|website|site|dashboard|tool|page|ui)\s+for\s+(?:an?\s+|the\s+)?([a-z][\w\- ]{2,28}?)(?=\s+(?:in|that|which|where|with|using|so|to|and|,|\.)|$)",
        r"\b(?:an?|the)\s+([a-z][\w\- ]{2,28}?)\s+(?:app|application|website|site|dashboard|tool|page)\b",
        r"\b(?:build|create|make|develop)\s+(?:me\s+)?(?:an?\s+)?([a-z][\w\- ]{2,28}?)\s+(?:app|website|site|dashboard|tool)\b",
    )
    for pat in patterns:
        for m in re.finditer(pat, low, re.IGNORECASE):
            cand = _clean_name(m.group(1))
            if 2 < len(cand) <= 30 and cand.lower() not in _NAME_STOP and \
                    not any(w in _NAME_STOP for w in [cand.lower()]):
                words = cand.split()
                if len(words) <= 4:
                    return cand.title()
    base = archetype_name.split("/")[0].strip()
    return base.title() if base else "App"


def build_brief(goal: str, skills: Optional[Iterable[str]] = None, *,
                plan_paths: Optional[Iterable[str]] = None,
                explicit_theme: Optional[str] = None,
                explicit_archetype: Optional[str] = None) -> DesignBrief:
    """Derive the full design brief for ``goal`` (deterministic; never raises).

    ``plan_paths`` (when the plan already exists) fixes the UI component root
    and the JS/TS flavour; without it the root is inferred from the goal (a
    goal that also names a backend puts the UI under ``frontend/``).
    """
    skills = [str(s) for s in (skills or [])]
    if not applies(goal, skills):
        return DesignBrief(applies=False)
    low = goal or ""
    arch = archetype_by_key(explicit_archetype) if explicit_archetype else None
    arch = arch or select_archetype(low)
    framework = detect_framework(low, skills)
    styling = "tailwind" if ("tailwind" in {s.lower() for s in skills} or _TAILWIND_RE.search(low)) else "css-tokens"
    theme = select_theme(low, explicit_theme, archetype_key=arch.key)
    external_api = bool(_EXTERNAL_API_RE.search(low)) or arch.key == "chat-assistant"
    secrets = bool(_SECRET_RE.search(low)) or arch.key == "chat-assistant"
    persistence = True  # universal: settings/data survive a reload
    multi_screen = bool(_MULTI_RE.search(low)) or len(arch.screens) > 1
    typescript = bool(_TS_RE.search(low))
    server_rendered = framework == "html" and bool(
        _SERVER_FW_RE.search(low) or ({s.lower() for s in skills} & {"flask", "fastapi", "django", "express"}))
    paths = [str(p).replace("\\", "/") for p in (plan_paths or [])]
    if paths:
        typescript = typescript or any(p.endswith((".ts", ".tsx")) for p in paths)
        app_root = _ui_root_from_paths(paths)
    else:
        backend = bool(_BACKEND_RE.search(low)) and framework != "nextjs" and not server_rendered
        app_root = "frontend" if backend else ""
    reqs = list(UNIVERSAL_REQUIREMENTS) + list(arch.ux_requirements)
    if arch.key in DATA_ARCHETYPES and not external_api:
        reqs.insert(len(UNIVERSAL_REQUIREMENTS),
                    "With no backend declared, the storage helper seeds realistic sample data from the sample-data "
                    "module on first run (12-30 records with real-looking names, categories, dates and amounts) so every "
                    "screen renders populated -- never '0', 'NaN' or an empty chart -- and a 'Sample data' badge is shown")
    if external_api:
        reqs.append("Every request to the user-configured API reads the base URL and key from the saved settings, "
                    "sends them as headers, and reports failures with the server's status and message")
    brief = DesignBrief(
        applies=True, framework=framework, styling=styling, typescript=typescript,
        product_name=_product_name(low, arch.name), archetype=arch.key, archetype_name=arch.name,
        layout=arch.layout, screens=list(arch.screens), components=list(arch.components),
        requirements=reqs, integration_notes=list(arch.integration_notes), theme=theme.key,
        app_root=app_root, multi_screen=multi_screen, external_api=external_api,
        persistence=persistence, secrets=secrets, server_rendered=server_rendered)
    _assign_paths(brief)
    return brief


def _ui_root_from_paths(paths: List[str]) -> str:
    """Directory holding the UI component: where index.html lives, else the
    parent of the JS ``src/`` tree, else the repo root."""
    for p in paths:
        if p.rsplit("/", 1)[-1].lower() == "index.html":
            return p.rsplit("/", 1)[0] if "/" in p else ""
    for p in paths:
        if "/templates/" in p:
            return p.split("/templates/", 1)[0]
        if p.startswith("templates/"):
            return ""
    for p in paths:
        if p.endswith((".jsx", ".tsx", ".vue")) or (p.endswith((".js", ".ts")) and "/src/" in p):
            if "/src/" in p:
                return p.split("/src/", 1)[0]
            if p.startswith("src/"):
                return ""
    return ""


def theme_for(brief: DesignBrief):
    """The :class:`~cgx.design.themes.Theme` object a brief refers to."""
    return theme_by_key(brief.theme) or select_theme("", None)
