"""Product archetypes: what a *real* version of a requested app looks like.

A coding model given "an app for a chatbot with an API-key profile tab" has
no notion of what a finished chat product contains -- so it ships a form that
``console.log``s the key and two bare buttons. A senior product designer, by
contrast, reaches for a known archetype: a chat app has a message column with
a sticky composer, a settings screen that persists and masks the key, empty /
loading / error states, keyboard shortcuts, and so on.

This module encodes those archetypes deterministically. :func:`select_archetype`
scores the goal text against each archetype's keywords and returns the best
match (or the generic web-app fallback); the brief then carries the archetype's
layout, screens, component inventory and CHECKABLE UX requirements into the
planner prompt, the per-file generator prompt, and the verifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

__all__ = ["Archetype", "ARCHETYPES", "GENERIC", "select_archetype", "archetype_by_key"]


@dataclass
class Archetype:
    key: str
    name: str
    match_keywords: List[str]
    layout: str
    screens: List[str]
    components: List[str]
    ux_requirements: List[str]
    default_theme: str = "clean-tech"
    #: Domain integration guidance a senior engineer would apply (e.g. how a
    #: chat client should talk to an OpenAI-compatible endpoint).
    integration_notes: List[str] = field(default_factory=list)


ARCHETYPES: List[Archetype] = [
    Archetype(
        key="chat-assistant", name="Chat / AI assistant app",
        match_keywords=["chat", "chatbot", "chat bot", "assistant", "conversation", "messaging", "messenger",
                        "llm", "gpt", "ai assistant", "copilot", "support bot", "helpdesk", "q&a bot"],
        layout=("App shell with a slim top bar (product name, nav to Chat and Settings, connection status "
                "badge). The Chat screen is a full-height column: a scrollable message list with user and "
                "assistant bubbles on alternate sides, and a sticky composer (auto-growing textarea + primary "
                "Send button) pinned to the bottom. Settings is a centred card (max-width ~640px)."),
        screens=["Chat (main)", "Settings / Profile (API address + API key)"],
        components=["AppShell/TopBar with nav + ConnectionStatus badge", "MessageList (auto-scrolls to newest)",
                    "MessageBubble (user vs assistant styling, timestamp, markdown-safe text)",
                    "Composer (textarea, Enter sends, Shift+Enter newline, disabled while sending)",
                    "TypingIndicator (three animated dots while awaiting a reply)",
                    "EmptyState (welcome headline + 3 suggestion chips that prefill the composer)",
                    "SettingsForm (URL input, masked key input with show/hide, Save, Test connection, saved confirmation)",
                    "ErrorBanner/inline error with Retry"],
        ux_requirements=[
            "API address and key persist in localStorage and are loaded on start; the Chat screen shows a clear "
            "'configure your API in Settings' empty state with a link until both are set",
            "Sending appends the user message immediately, shows a typing indicator, then appends the assistant "
            "reply; a failed request shows an inline error with the server message and a Retry action",
            "The composer sends on Enter (Shift+Enter inserts a newline), clears after send, is disabled while a "
            "request is in flight, and the Send button has a loading state",
            "The message list auto-scrolls to the newest message and distinguishes user vs assistant bubbles",
            "Conversation history persists in localStorage with a 'New chat'/'Clear' action",
            "The API key input is type=password with a show/hide toggle and is never logged",
        ],
        default_theme="conversational",
        integration_notes=[
            "Treat the configured address as an OpenAI-compatible chat endpoint: if the URL has no path (or ends "
            "with /v1) append /chat/completions as needed; POST JSON {model, messages:[{role, content}]}; read "
            "choices[0].message.content; send 'Authorization: Bearer <key>' and 'Content-Type: application/json'",
            "Surface non-2xx responses as readable errors (status + response text); never hard-code '/api/chat'",
            "Keep the API client in one module (lib/api.js) that reads settings from storage; screens never call fetch directly",
        ]),
    Archetype(
        key="admin-dashboard", name="Admin / CRUD dashboard",
        match_keywords=["admin", "admin dashboard", "admin panel", "crud", "manage", "management", "inventory", "records", "customers",
                        "orders", "catalog admin", "back office", "backoffice", "data table", "employees", "students",
                        "library system", "booking", "reservations", "crm"],
        layout=("Sidebar navigation (brand + section links) with a top bar (page title, search, primary action). "
                "Content area: page header with title/subtitle and a primary 'New …' button, stat cards row, then a "
                "data table card with search/filter controls, row actions and pagination. Create/edit happens in a "
                "modal or side drawer with a validated form."),
        screens=["Overview (stat cards + recent activity)", "List (table with search/filter/sort)",
                 "Detail / Edit (form in a drawer or modal)", "Settings"],
        components=["Sidebar + TopBar", "StatCard", "DataTable (sortable headers, row hover, empty row state)",
                    "SearchInput + FilterChips", "Modal/Drawer with Form + validation messages", "Toast notifications",
                    "ConfirmDialog for destructive actions", "Pagination", "EmptyState"],
        ux_requirements=[
            "Every list has loading (skeleton rows), empty (guidance + create action) and error states",
            "Create/edit forms validate required fields inline and disable Submit while saving; success shows a toast",
            "Delete asks for confirmation and can be cancelled",
            "Data persists (localStorage when no backend is declared, otherwise the declared API) and reloads on start",
            "Active navigation item is highlighted and the page title matches the current screen",
        ],
        default_theme="midnight-console"),
    Archetype(
        key="analytics-dashboard", name="Analytics / metrics dashboard",
        match_keywords=["analytics", "analytics dashboard", "metrics", "kpi", "kpis", "charts", "chart", "monitoring", "report", "reports",
                        "statistics", "stats", "visualize", "visualization", "insights", "telemetry"],
        layout=("Top bar with title, date-range/filter controls and a refresh action. A responsive card grid: a row "
                "of KPI tiles (value, delta, sparkline), then chart cards (SVG line/bar charts drawn in-house unless a "
                "chart library is requested) and a breakdown table."),
        screens=["Overview (KPIs + charts)", "Detail / drill-down", "Settings / data source"],
        components=["TopBar + FilterBar (date range, segment select)", "KpiTile (value, delta badge, sparkline)",
                    "ChartCard (title, legend, SVG chart, empty/loading states)", "Legend", "DataTable", "EmptyState"],
        ux_requirements=[
            "Charts render from real data arrays with axes/labels and a legend; no placeholder images",
            "Every card has loading and empty states; numbers use locale formatting and units",
            "Filters update all cards consistently; the active filter is visible",
            "Layout reflows to a single column under 768px",
        ],
        default_theme="midnight-console"),
    Archetype(
        key="kanban-productivity", name="Tasks / notes / productivity tool",
        match_keywords=["todo", "to-do", "task", "tasks", "kanban", "board", "notes", "note taking", "habit",
                        "planner", "checklist", "project management", "pomodoro", "reminder", "journal", "tracker"],
        layout=("Top bar (brand, view switcher, add action) over the workspace: a kanban board of equal-width columns "
                "with cards, or a filterable list with grouping. Adding uses an inline form; editing opens a side "
                "panel/modal. Counts per column/group are visible."),
        screens=["Board / List (main)", "Item detail / edit", "Settings"],
        components=["TopBar", "Column with header + count", "TaskCard (title, meta, tags, actions)",
                    "AddItemForm (inline, Enter to add)", "FilterChips / search", "EmptyState per column", "Toast"],
        ux_requirements=[
            "Items persist in localStorage and restore on load",
            "Adding with Enter works; empty input is rejected with inline feedback",
            "Completing/moving/deleting an item gives immediate visual feedback",
            "Every column/list shows an empty state when it has no items",
        ],
        default_theme="calm-productivity"),
    Archetype(
        key="landing-marketing", name="Marketing / landing website",
        match_keywords=["landing", "landing page", "marketing", "homepage", "home page", "website for", "brochure",
                        "startup site", "agency", "coffee shop", "restaurant", "bakery", "gym", "salon", "clinic",
                        "studio", "company site", "business site", "promo", "launch page", "small business"],
        layout=("Sticky header (logo, section nav, CTA button) over stacked full-width sections: hero (headline, "
                "sub-headline, primary + secondary CTA), feature grid (3-4 cards with icons), social proof / "
                "testimonials, pricing or menu/services, FAQ or about, contact/CTA band, footer with columns."),
        screens=["Home (sections)", "Contact (form) or anchor section"],
        components=["Header + MobileMenu", "Hero", "FeatureCard", "Testimonial", "PricingCard / MenuItem", "CTABand",
                    "ContactForm with validation + success state", "Footer"],
        ux_requirements=[
            "Sections have real, specific copy for the business (no lorem ipsum)",
            "Anchor navigation works and the header stays usable on mobile (collapsible menu)",
            "The contact form validates fields and shows a success message without a page reload",
            "Hero/section backgrounds use CSS gradients or inline SVG -- no missing local image files",
        ],
        default_theme="clean-tech"),
    Archetype(
        key="ecommerce-catalog", name="Shop / product catalog",
        match_keywords=["shop", "store", "e-commerce", "ecommerce", "products", "product catalog", "cart", "checkout",
                        "marketplace", "storefront", "menu ordering", "order online"],
        layout=("Header with logo, search and a cart button with a count badge. Catalog: filter sidebar/chips + "
                "responsive product grid; product detail page/modal; slide-in cart drawer with line items, "
                "quantity steppers and totals; checkout form."),
        screens=["Catalog", "Product detail", "Cart", "Checkout"],
        components=["Header + SearchInput + CartButton(badge)", "ProductCard (image area, name, price, add button)",
                    "FilterSidebar/Chips", "CartDrawer + QuantityStepper + totals", "CheckoutForm", "EmptyState"],
        ux_requirements=[
            "The cart persists in localStorage, shows an empty state, and totals update live",
            "Adding to cart gives immediate feedback (badge count + toast)",
            "Product images use CSS gradient/SVG placeholders or remote URLs -- never missing local files",
            "Checkout validates fields and shows an order confirmation",
        ],
        default_theme="playful-vibrant"),
    Archetype(
        key="form-wizard", name="Form / calculator / wizard tool",
        match_keywords=["form", "survey", "wizard", "questionnaire", "multi-step", "calculator", "converter",
                        "estimator", "quiz", "generator", "configurator", "signup form", "intake"],
        layout=("Centred card (max-width ~640px) under a minimal top bar: an optional step/progress indicator, one "
                "focused group of fields per step, Back/Next or Calculate actions, and a result/summary card."),
        screens=["Input / steps", "Result / summary"],
        components=["Stepper/ProgressBar", "FieldGroup with labels + hints", "ValidationMessage", "ResultCard",
                    "Reset action"],
        ux_requirements=[
            "Inputs are labelled, validated inline, and the primary action is disabled until valid",
            "Results update clearly with formatted values and units; a reset returns to a clean state",
            "Keyboard: Enter advances/submits; focus moves sensibly between steps",
        ],
        default_theme="calm-productivity"),
    Archetype(
        key="saas-app", name="Account-based SaaS web app",
        match_keywords=["login", "log in", "signup", "sign up", "auth", "authentication", "account", "profile page",
                        "settings page", "subscription", "saas", "workspace", "team", "user management", "register"],
        layout=("Auth screens as a centred card with brand mark; the authenticated app uses a sidebar shell (brand, "
                "nav, user menu) with a top bar and content pages. Settings is a tabbed/sectioned page with a danger "
                "zone."),
        screens=["Login / Sign up", "Home / Dashboard", "Settings / Profile"],
        components=["AuthCard with PasswordInput (show/hide)", "Sidebar + TopBar + UserMenu/Avatar", "SettingsSections",
                    "Tabs", "DangerZone with ConfirmDialog", "Toast"],
        ux_requirements=[
            "Auth forms validate inline, mask passwords with a show/hide toggle, and show submission errors",
            "Session/profile state persists (localStorage when no backend) and protected screens redirect when signed out",
            "Settings changes save with visible success feedback",
        ],
        default_theme="clean-tech"),
    Archetype(
        key="content-site", name="Docs / blog / content site",
        match_keywords=["blog", "docs", "documentation", "articles", "wiki", "knowledge base", "markdown", "posts",
                        "news", "magazine", "reader", "newsletter", "changelog"],
        layout=("Header with brand, nav and search; content column with reading typography (max ~72ch) and a sticky "
                "table-of-contents sidebar on wide screens; index pages use card lists with tags and dates."),
        screens=["Index / listing", "Article / page", "Search"],
        components=["Header + Search", "ArticleCard (title, excerpt, date, tags)", "TOC", "Prose styles", "TagChip",
                    "Pagination", "EmptyState for no results"],
        ux_requirements=[
            "Reading view uses a comfortable measure (max ~72 characters), clear hierarchy and code block styling",
            "Search/filter has an empty-results state",
            "Navigation shows the current section",
        ],
        default_theme="clean-tech"),
    Archetype(
        key="developer-tool", name="Developer tool / console",
        match_keywords=["playground", "console", "editor", "regex", "json formatter", "formatter", "api client",
                        "debugger", "terminal", "log viewer", "devtool", "sql runner", "query runner", "rest client",
                        "diff viewer", "linter", "cli ui", "repl"],
        layout=("Top bar with title, mode/toolbar actions and a status indicator; the workspace is a resizable split: "
                "input/editor pane (monospace) on the left, output/result pane on the right (stacked on mobile); a "
                "status bar shows state, timing or errors."),
        screens=["Workspace (input + output)", "History", "Settings"],
        components=["TopBar + Toolbar", "CodeEditor (monospace textarea with line wrap)", "OutputPanel with copy "
                    "action and empty state", "StatusBar", "Tabs", "HistoryList"],
        ux_requirements=[
            "Input and output use the mono font; output has a Copy action and clear error rendering",
            "Run/Format executes on Ctrl/Cmd+Enter as well as the button; the button shows a busy state",
            "History/settings persist in localStorage",
        ],
        default_theme="midnight-console"),
    Archetype(
        key="portfolio", name="Portfolio / personal site",
        match_keywords=["portfolio", "personal site", "personal website", "resume", "cv", "photographer", "designer",
                        "showcase", "freelancer", "artist"],
        layout=("Minimal header; hero with name, role and CTA; project grid with hover states; about section; contact "
                "section/footer with links."),
        screens=["Home (sections)"],
        components=["Header", "Hero", "ProjectCard", "SkillList/Badges", "ContactLinks", "Footer"],
        ux_requirements=["Real copy and project entries (no lorem ipsum)",
                         "Project cards have hover states and consistent image areas (gradient/SVG placeholders)",
                         "Anchor nav + responsive layout"],
        default_theme="bold-dark"),
    Archetype(
        key="game", name="Browser game",
        match_keywords=["game", "tic-tac-toe", "tic tac toe", "snake", "memory game", "puzzle", "trivia", "wordle",
                        "minesweeper", "sudoku", "2048", "hangman", "flashcards"],
        layout=("Centred stage: a title, a score/status HUD, the board or play area, and a controls row (new game, "
                "difficulty). Rules/help in a collapsible panel."),
        screens=["Play", "Results / high scores"],
        components=["HUD (score, timer, status)", "Board + Cell", "Controls", "ResultOverlay/Modal", "HelpPanel"],
        ux_requirements=["Game state and high score persist in localStorage",
                         "Win/lose/draw is announced clearly with a restart action",
                         "Keyboard controls where natural; touch-friendly targets (>= 44px)"],
        default_theme="playful-vibrant"),
]

GENERIC = Archetype(
    key="generic-web-app", name="Web application",
    match_keywords=[],
    layout=("App shell: top bar with brand and navigation, a main content area in a centred container with a page "
            "header (title + subtitle + primary action), and cards for each functional block."),
    screens=["Home (main task)", "Settings"],
    components=["AppShell/TopBar + Nav", "PageHeader", "Card", "Form with validation", "Button variants", "EmptyState",
                "Toast/inline feedback"],
    ux_requirements=[
        "Every async action and list has loading, empty and error states",
        "User data/settings persist in localStorage (or the declared backend) and restore on load",
        "Forms are labelled, validated inline, and give success feedback",
    ],
    default_theme="clean-tech")


def archetype_by_key(key: Optional[str]) -> Optional[Archetype]:
    for a in ARCHETYPES:
        if a.key == (key or ""):
            return a
    return GENERIC if key == GENERIC.key else None


def select_archetype(goal: str) -> Archetype:
    """Best-matching archetype for ``goal`` (deterministic keyword score).

    Multi-word keywords weigh double (they are more specific); ties are broken
    by list order. Falls back to :data:`GENERIC` when nothing matches.
    """
    low = f" {(goal or '').lower()} "
    best, best_score = None, 0
    for a in ARCHETYPES:
        score = 0
        for kw in a.match_keywords:
            needle = kw.lower()
            if needle in low:
                score += 2 if " " in needle else 1
        if score > best_score:
            best, best_score = a, score
    return best or GENERIC
