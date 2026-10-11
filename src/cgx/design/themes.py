"""Curated design-system themes + the deterministic tokens stylesheet.

Design quality cannot be prompted out of a mid-size local model: asked to
"make it beautiful" it improvises gray-on-white, unstyled controls and no
hierarchy. So the *design* comes from a curated library of hand-tuned themes
(real palette hex, a Google Font pairing, spacing/radius/shadow scales and
component treatment), and generation is CONSTRAINED to the selected theme:

* :func:`select_theme` picks a theme deterministically from the brief
  (keyword score, archetype default, explicit key) -- a coffee shop reliably
  gets a warm espresso palette, a developer console a dark blue one, with no
  model choice involved.
* :func:`tokens_stylesheet` renders the theme into a COMPLETE, professional
  tokens stylesheet (``:root`` variables, base element styles, focus rings, a
  small set of ready-made component classes -- ``.btn-primary``, ``.card``,
  ``.input``, ``.app-shell`` ... -- and responsive rules). CGX writes this file
  itself, with no model call, the same way it writes ``conftest.py``: the weak
  model then *uses* a design system instead of inventing one.
* :func:`theme_prompt_block` renders the theme into hard constraints for the
  per-file generator so component styles use ``var(--color-*)`` and never
  hard-code colours.

Every palette pair below was checked against WCAG AA (>= 4.5:1 for body text
on the page/surface, and for the primary button label on the primary colour).
Where an accent does not clear body-text contrast its ``style`` text restricts
it to decorative/large use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

__all__ = [
    "Theme", "THEMES", "DEFAULT_THEME_KEY", "select_theme", "theme_by_key",
    "theme_prompt_block", "theme_summaries", "tokens_stylesheet",
    "tailwind_entry_stylesheet", "contrast_ratio",
]


@dataclass
class Theme:
    key: str
    name: str
    vibe: str
    match_keywords: List[str]
    palette: Dict[str, str]  # bg, surface, text, muted, primary, primary_contrast, accent, border
    fonts: Dict[str, str]    # heading, body, google_link (+ optional mono)
    style: str
    dark: bool = False
    #: Archetype keys this theme is the natural default for.
    default_for: List[str] = field(default_factory=list)


def _lum(hex_color: str) -> float:
    """Relative luminance of a ``#rrggbb`` colour (WCAG 2.x formula)."""
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    def _lin(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def contrast_ratio(fg: str, bg: str) -> float:
    """WCAG contrast ratio between two ``#rrggbb`` colours (>= 1.0)."""
    try:
        l1, l2 = _lum(fg), _lum(bg)
    except Exception:
        return 1.0
    hi, lo = max(l1, l2), min(l1, l2)
    return round((hi + 0.05) / (lo + 0.05), 2)


def _mix(hex_a: str, hex_b: str, t: float) -> str:
    """Linear mix of two colours (``t`` = weight of ``hex_b``)."""
    a = hex_a.lstrip("#"); b = hex_b.lstrip("#")
    out = []
    for i in (0, 2, 4):
        ca, cb = int(a[i:i + 2], 16), int(b[i:i + 2], 16)
        out.append(round(ca + (cb - ca) * t))
    return "#" + "".join(f"{max(0, min(255, v)):02X}" for v in out)


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


THEMES: List[Theme] = [
    Theme(
        key="warm-artisan", name="Warm Artisan",
        vibe="Cozy hand-crafted warmth -- espresso, cream, and caramel with an editorial serif voice.",
        match_keywords=["coffee", "coffee shop", "cafe", "espresso", "bakery", "artisan", "tea house",
                        "roastery", "patisserie", "brunch", "restaurant", "chocolate", "handmade", "deli"],
        palette={"bg": "#FBF6EE", "surface": "#FFFDF9", "text": "#2B1D14", "muted": "#6F5B4E",
                 "primary": "#8A4B2A", "primary_contrast": "#FFFFFF", "accent": "#A85A15",
                 "border": "#E7DAC8"},
        fonts={"heading": "Fraunces", "body": "Nunito Sans",
               "google_link": "https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,500;0,9..144,600;0,9..144,700;1,9..144,500&family=Nunito+Sans:wght@400;500;600;700&display=swap"},
        style=("Rounded, tactile, paper-warm. RADIUS 6px inputs / 14px cards / pill buttons. Soft espresso-tinted "
               "shadows (never black). 8px spacing base; sections 96px desktop / 56px mobile; max width 1120px. "
               "Headings Fraunces 600-700 with tight leading; body Nunito Sans at 1.65 line-height. Primary button "
               "solid #8A4B2A with white text (6.7:1), pill, darkens on hover. Links/eyebrows in accent #A85A15 "
               "(4.7:1 on the cream). Hero backgrounds are CSS gradients only -- never remote images."),
        dark=False, default_for=["landing-marketing"]),
    Theme(
        key="clean-tech", name="Clean Tech",
        vibe="Crisp, modern, trustworthy -- an airy indigo-accented SaaS look on cool neutrals and generous whitespace.",
        match_keywords=["saas", "startup", "software", "agency", "product", "platform", "b2b", "landing",
                        "fintech", "cloud", "signup", "login", "account", "subscription"],
        palette={"bg": "#F7F8FA", "surface": "#FFFFFF", "text": "#0F172A", "muted": "#5A6478",
                 "primary": "#4F46E5", "primary_contrast": "#FFFFFF", "accent": "#7C3AED",
                 "border": "#E2E8F0"},
        fonts={"heading": "Space Grotesk", "body": "Inter",
               "google_link": "https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap"},
        style=("Geometric, airy, confident. RADIUS 12px cards/inputs, 10px buttons, pill badges. Layered soft shadows; "
               "hover lifts with an indigo glow. 8px spacing scale; max width 1120px. Headings Space Grotesk 600-700 with "
               "-0.02em tracking; body Inter 16-17px / 1.6; muted copy #5A6478 (5.6:1). Primary button solid #4F46E5 "
               "with white text (6.3:1). Eyebrow labels in accent #7C3AED (5.4:1). Sections alternate #F7F8FA / #FFFFFF."),
        dark=False, default_for=["saas-app", "content-site", "generic-web-app"]),
    Theme(
        key="bold-dark", name="Neon Noir",
        vibe="Cinematic near-black canvas lit by an electric violet-to-teal glow -- loud, confident, after-dark.",
        match_keywords=["creative studio", "portfolio", "nightlife", "gaming", "music", "product launch",
                        "events", "dj", "photography", "brand", "film", "neon"],
        palette={"bg": "#08080C", "surface": "#12121B", "text": "#EAEAF2", "muted": "#9EA0B8",
                 "primary": "#B383FF", "primary_contrast": "#0A0A12", "accent": "#2DE2C9",
                 "border": "#26263A"},
        fonts={"heading": "Space Grotesk", "body": "Inter",
               "google_link": "https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap"},
        style=("RADIUS 14px cards/inputs, pill buttons. Depth via layered dark shadows plus a violet halo on primary "
               "actions. 8px spacing base; sections 96-120px; max width 1120px. Primary button solid #B383FF with "
               "#0A0A12 text (7.1:1), uppercase label tracking, gradient shift on hover. Accent teal #2DE2C9 (11:1 on "
               "the dark ground) for links, active nav and kickers. Muted #9EA0B8 (7.2:1). Hero: radial violet/teal "
               "glows over #08080C plus a faint dot grid -- pure CSS, no images."),
        dark=True, default_for=["portfolio"]),
    Theme(
        key="elegant-luxury", name="Elegant Luxury",
        vibe="Hushed, deep-toned refinement with warm gold accents -- a fine-dining room or a jeweler's velvet tray.",
        match_keywords=["fine dining", "hotel", "resort", "jewelry", "jeweler", "fashion", "boutique", "spa",
                        "luxury", "couture", "atelier", "champagne", "concierge", "wine", "winery", "gallery"],
        palette={"bg": "#14110E", "surface": "#1F1A16", "text": "#F2EBE0", "muted": "#A89A88",
                 "primary": "#C9A24B", "primary_contrast": "#14110E", "accent": "#E4CE9B",
                 "border": "#342C24"},
        fonts={"heading": "Cormorant Garamond", "body": "Jost",
               "google_link": "https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Jost:wght@300;400;500;600&display=swap"},
        style=("Restrained radius (2px buttons / 6px cards, nothing pill-shaped). Near-invisible warm shadows; emphasis "
               "from 1px #342C24 hairlines and gold rules. Generous spacing (clamp(5rem,10vw,9rem) sections). Headings "
               "Cormorant Garamond 500/600 at large scale against Jost 300 body (1.7 line-height); uppercase Jost 500 "
               "with .22em tracking for eyebrows and nav. Primary button solid gold #C9A24B with #14110E text (7.8:1), "
               "hover to champagne #E4CE9B. Gold and champagne both clear AA on the dark ground (7-12:1)."),
        dark=True, default_for=[]),
    Theme(
        key="fresh-natural", name="Fresh & Natural",
        vibe="Calm, organic wellness -- soft sage greens and warm cream with airy, rounded, breathable layouts.",
        match_keywords=["wellness", "salon", "plants", "eco", "clinic", "yoga", "organic", "health", "beauty",
                        "nutrition", "garden", "sustainability", "meditation", "skincare", "botanical", "fitness",
                        "therapy", "recipe", "recipes", "food"],
        palette={"bg": "#F7F4ED", "surface": "#FFFFFF", "text": "#22302A", "muted": "#5F6B60",
                 "primary": "#3F7150", "primary_contrast": "#FFFFFF", "accent": "#D98E5A",
                 "border": "#E4DECE"},
        fonts={"heading": "Poppins", "body": "Nunito Sans",
               "google_link": "https://fonts.googleapis.com/css2?family=Poppins:wght@500;600;700&family=Nunito+Sans:wght@400;600;700&display=swap"},
        style=("Rounded, soft, airy. RADIUS 20px cards/images, pill buttons/inputs, 12px chips. Gentle green-tinted "
               "shadows with a 1px #E4DECE border. 8px spacing scale; sections 96px; max width 1120px. Headings Poppins "
               "600/700; body Nunito Sans at 1.7 line-height; muted #5F6B60 (5.1:1). Primary button solid #3F7150 with "
               "white text (5.7:1), pill. IMPORTANT: terracotta accent #D98E5A is DECORATIVE ONLY (2.5:1 on light) -- "
               "icon glyphs, chip fills with dark ink, top-accent bars -- never body/link text; use primary #3F7150 "
               "for coloured text."),
        dark=False, default_for=[]),
    Theme(
        key="playful-vibrant", name="Playful Vibrant",
        vibe="Bright, bouncy and joyful -- candy-colored energy with chunky rounded type and confetti-soft shapes.",
        match_keywords=["kids", "children", "party", "food truck", "festival", "playful", "fun", "ice cream",
                        "toys", "birthday", "carnival", "game", "games", "quiz", "trivia", "puzzle"],
        palette={"bg": "#FFFDF7", "surface": "#FFFFFF", "text": "#2B1B4A", "muted": "#6B6480",
                 "primary": "#FF4D8D", "primary_contrast": "#2B1B4A", "accent": "#00BFA6",
                 "border": "#F0E6D8"},
        fonts={"heading": "Baloo 2", "body": "Nunito",
               "google_link": "https://fonts.googleapis.com/css2?family=Baloo+2:wght@500;600;700;800&family=Nunito:wght@400;500;600;700&display=swap"},
        style=("Rounded and chunky. RADIUS pill buttons/inputs, 24px cards, 12px chips. Soft coloured lifted shadows; "
               "buttons carry a chunky bottom press-shadow that collapses on :active. 8px spacing base; max width "
               "1120px. Headings Baloo 2 700-800; body Nunito 17px / 1.7; muted #6B6480 (5.5:1); default text #2B1B4A. "
               "COLOUR RULES: pink #FF4D8D is a FILL / large-display colour (4.9:1 with #2B1B4A ink on it) -- never "
               "small body or link text on white; teal #00BFA6 and amber #FFC93C are decorative fills only with "
               "#2B1B4A ink over them. Text links use #2B1B4A with a pink underline."),
        dark=False, default_for=["game", "ecommerce-catalog"]),
    # ---- app-oriented themes ------------------------------------------------
    Theme(
        key="midnight-console", name="Midnight Console",
        vibe="A focused dark developer console -- deep navy canvas, cool blue actions, cyan highlights, crisp mono data.",
        match_keywords=["dashboard", "console", "monitoring", "metrics", "analytics", "developer", "devtool",
                        "terminal", "logs", "admin", "ops", "observability", "dark", "dark mode", "editor",
                        "playground", "api client", "sql"],
        palette={"bg": "#0B1020", "surface": "#121A2E", "text": "#E6EAF2", "muted": "#9AA4B8",
                 "primary": "#4F8CFF", "primary_contrast": "#0B1020", "accent": "#22D3EE",
                 "border": "#1F2A44"},
        fonts={"heading": "Inter", "body": "Inter", "mono": "JetBrains Mono",
               "google_link": "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap"},
        style=("Dense but breathable. RADIUS 10px cards/inputs, 8px buttons, 999px status pills. Depth from a 1px "
               "#1F2A44 border plus a subtle 0 1px 0 rgba(255,255,255,.04) inner top highlight and 0 12px 32px "
               "rgba(0,0,0,.45) shadows. 8px spacing base (8/12/16/24/32/48); sidebar 260px; content max width 1280px. "
               "Headings Inter 600 with -0.01em tracking; body Inter 14-15px / 1.55; numbers, ids, logs and code in "
               "JetBrains Mono. Contrast (verified): text #E6EAF2 on #0B1020 15.6:1; muted #9AA4B8 on #0B1020 7.4:1 and "
               "on surface #121A2E 6.4:1; primary button #4F8CFF with #0B1020 text 5.9:1; accent cyan #22D3EE on #0B1020 "
               "10.4:1 so it is safe for links, active nav, focus rings and live indicators. Primary hover lightens to "
               "#6BA0FF. Success #34D399, warning #FBBF24, danger #F87171 (all >= 7:1 on the canvas). Hero/empty-state "
               "backgrounds: radial-gradient(60% 80% at 20% 0%, rgba(79,140,255,.18), transparent 60%) over #0B1020 "
               "plus a faint 24px grid of rgba(255,255,255,.04) lines -- no images."),
        dark=True, default_for=["analytics-dashboard", "developer-tool", "admin-dashboard"]),
    Theme(
        key="calm-productivity", name="Calm Productivity",
        vibe="A quiet, trustworthy light workspace -- soft blue-gray canvas, clear blue actions, teal success cues.",
        match_keywords=["todo", "to-do", "task", "tasks", "kanban", "notes", "planner", "productivity", "habit",
                        "checklist", "project management", "calendar", "form", "survey", "wizard", "crm",
                        "inventory", "booking", "tracker", "budget", "expense", "manage"],
        palette={"bg": "#F6F7FB", "surface": "#FFFFFF", "text": "#1C2333", "muted": "#5B6478",
                 "primary": "#2563EB", "primary_contrast": "#FFFFFF", "accent": "#0F766E",
                 "border": "#E3E7EF"},
        fonts={"heading": "Manrope", "body": "Inter",
               "google_link": "https://fonts.googleapis.com/css2?family=Manrope:wght@600;700;800&family=Inter:wght@400;500;600&display=swap"},
        style=("Clean and calm. RADIUS 12px cards, 8px inputs/buttons, pill chips. Shadows barely there: 0 1px 2px "
               "rgba(28,35,51,.06), 0 8px 24px rgba(28,35,51,.06); cards also carry a 1px #E3E7EF border. 8px spacing "
               "base; page padding 32px desktop / 16px mobile; content max width 1200px; kanban columns 300px. "
               "Headings Manrope 700 with -0.01em tracking; body Inter 15px / 1.6. Contrast (verified): text #1C2333 "
               "on #F6F7FB 14.6:1; muted #5B6478 on #F6F7FB 5.5:1 and on white 6.0:1; primary button #2563EB with "
               "white text 5.2:1, hover #1D4ED8; accent teal #0F766E on #F6F7FB 5.1:1 so it is safe for success text, "
               "links and completed states. Danger #B91C1C (6.5:1), warning text #B45309 (4.8:1). Column/empty-state "
               "surfaces use #EEF1F7. Subtle 150ms hover lifts; drag handles in muted."),
        dark=False, default_for=["kanban-productivity", "form-wizard"]),
    Theme(
        key="conversational", name="Conversational",
        vibe="A modern messaging canvas -- warm off-white, violet user bubbles, calm assistant cards, sky-blue links.",
        match_keywords=["chat", "chatbot", "assistant", "conversation", "messaging", "messenger", "llm", "gpt",
                        "ai assistant", "copilot", "support bot", "helpdesk"],
        palette={"bg": "#FAFAFC", "surface": "#FFFFFF", "text": "#17171F", "muted": "#6B6F80",
                 "primary": "#6D28D9", "primary_contrast": "#FFFFFF", "accent": "#0369A1",
                 "border": "#E6E6EE"},
        fonts={"heading": "DM Sans", "body": "Inter",
               "google_link": "https://fonts.googleapis.com/css2?family=DM+Sans:wght@500;600;700&family=Inter:wght@400;500;600&display=swap"},
        style=("Soft and legible. RADIUS 16px message bubbles (user bubbles square off the bottom-right corner, "
               "assistant bubbles the bottom-left), 12px cards, 10px inputs, pill buttons/chips. Shadows: composer and "
               "cards 0 1px 2px rgba(23,23,31,.05), 0 6px 20px rgba(23,23,31,.06). 8px spacing base; message column max "
               "width 820px centred; composer sticky at the bottom with 16px padding. Headings DM Sans 600/700; body "
               "Inter 15px / 1.6; timestamps and meta 12px muted. Contrast (verified): text #17171F on #FAFAFC 17.6:1; "
               "muted #6B6F80 on #FAFAFC 4.7:1 (use for meta only, never long copy); primary #6D28D9 user bubble with "
               "white text 7.1:1; accent #0369A1 links on #FAFAFC 5.7:1. Assistant bubbles are #FFFFFF with a 1px "
               "#E6E6EE border; user bubbles are solid primary. Typing indicator: three 6px dots in muted with a "
               "staggered 1.2s bounce. Danger #B91C1C, success #047857 (both >= 6:1). Empty state: centred headline, "
               "three suggestion chips, soft radial violet glow rgba(109,40,217,.08) -- CSS only, no images."),
        dark=False, default_for=["chat-assistant"]),
]

DEFAULT_THEME_KEY = "clean-tech"


def theme_by_key(key: Optional[str]) -> Optional[Theme]:
    for t in THEMES:
        if t.key == (key or ""):
            return t
    return None


def select_theme(brief: str, explicit_key: Optional[str] = None,
                 archetype_key: Optional[str] = None) -> Theme:
    """Pick the best-fitting theme deterministically.

    Precedence: an explicit key (user choice) > a keyword hit in the brief
    (a "coffee shop dashboard" is still warm) > the archetype's default theme
    > :data:`DEFAULT_THEME_KEY`. A "dark" request pulls toward the dark themes.
    """
    t = theme_by_key(explicit_key)
    if t:
        return t
    low = (brief or "").lower()
    wants_dark = "dark" in low
    best: Optional[Theme] = None
    best_score = 0
    for cand in THEMES:
        score = sum(2 if " " in kw else 1 for kw in cand.match_keywords if kw in low)
        if wants_dark and cand.dark:
            score += 1
        if score > best_score:
            best, best_score = cand, score
    if best is not None and best_score >= 2:
        return best
    for cand in THEMES:
        if archetype_key and archetype_key in cand.default_for:
            if not wants_dark or cand.dark or archetype_key == "chat-assistant":
                return cand
    if wants_dark:
        return theme_by_key("midnight-console") or THEMES[0]
    return best or theme_by_key(DEFAULT_THEME_KEY) or THEMES[0]


def theme_summaries() -> List[Dict[str, str]]:
    """Lightweight listing for a UI theme picker / docs."""
    return [{"key": t.key, "name": t.name, "vibe": t.vibe,
             "primary": t.palette["primary"], "bg": t.palette["bg"],
             "accent": t.palette["accent"], "dark": str(t.dark).lower()}
            for t in THEMES]


def _semantic_colors(theme: Theme) -> Dict[str, str]:
    """Success/warning/danger triplet tuned for the theme's scheme."""
    if theme.dark:
        return {"success": "#34D399", "warning": "#FBBF24", "danger": "#F87171"}
    return {"success": "#047857", "warning": "#B45309", "danger": "#B91C1C"}


def tokens_stylesheet(theme: Theme, *, app_name: str = "") -> str:
    """Render the complete, deterministic tokens + base stylesheet for ``theme``.

    This is written to disk by CGX itself (no model call). It gives every
    generated UI a professional baseline: palette/type/spacing/radius/shadow
    variables, sane element defaults, focus rings, a handful of ready-made
    component classes, and responsive rules. Component files extend it with
    ``var(--...)`` -- they never redefine colours.
    """
    p = theme.palette
    sem = _semantic_colors(theme)
    heading = theme.fonts.get("heading", "Inter")
    body = theme.fonts.get("body", "Inter")
    mono = theme.fonts.get("mono", "JetBrains Mono")
    link = theme.fonts.get("google_link", "")
    if mono not in link and "JetBrains" not in link:
        link = link + ("&" if "?" in link else "?") + "family=JetBrains+Mono:wght@400;500&display=swap" \
            if link else "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap"
    shadow_ink = p["text"] if not theme.dark else "#000000"
    hover = _mix(p["primary"], "#FFFFFF" if theme.dark else "#000000", 0.12)
    # Link / active-nav colour must clear AA for small text on the page bg: the
    # accent when it does, else the primary, else the text colour -- so a
    # decorative-only accent (terracotta, candy pink) never becomes link text.
    link_color = next((c for c in (p["accent"], p["primary"], p["text"])
                       if contrast_ratio(c, p["bg"]) >= 4.5), p["text"])
    subtle = _mix(p["bg"], p["text"], 0.04)
    primary_soft = _rgba(p["primary"], 0.12)
    label = f" ({app_name})" if app_name else ""
    return f"""/* Design tokens -- generated by CGX from the "{theme.name}" theme{label}.
   {theme.vibe}
   Rules: use these variables and classes everywhere; never hard-code colours
   in components. Extend with component styles in app.css / *.module.css. */
@import url("{link}");

:root {{
  color-scheme: {"dark" if theme.dark else "light"};
  /* palette */
  --color-bg: {p["bg"]};
  --color-surface: {p["surface"]};
  --color-surface-2: {subtle};
  --color-text: {p["text"]};
  --color-muted: {p["muted"]};
  --color-primary: {p["primary"]};
  --color-primary-hover: {hover};
  --color-primary-soft: {primary_soft};
  --color-primary-contrast: {p["primary_contrast"]};
  --color-accent: {p["accent"]};
  --color-link: {link_color};
  --color-border: {p["border"]};
  --color-success: {sem["success"]};
  --color-warning: {sem["warning"]};
  --color-danger: {sem["danger"]};
  --color-danger-soft: {_rgba(sem["danger"], 0.12)};
  /* typography */
  --font-heading: "{heading}", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-body: "{body}", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-mono: "{mono}", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  --text-xs: 0.75rem;
  --text-sm: 0.875rem;
  --text-base: 1rem;
  --text-lg: 1.125rem;
  --text-xl: 1.375rem;
  --text-2xl: 1.75rem;
  --text-3xl: 2.25rem;
  --text-4xl: clamp(2.5rem, 4vw + 1rem, 3.5rem);
  /* spacing (4px base) */
  --space-1: 0.25rem;
  --space-2: 0.5rem;
  --space-3: 0.75rem;
  --space-4: 1rem;
  --space-5: 1.25rem;
  --space-6: 1.5rem;
  --space-8: 2rem;
  --space-10: 2.5rem;
  --space-12: 3rem;
  --space-16: 4rem;
  /* radius + shadows */
  --radius-sm: 6px;
  --radius-md: 10px;
  --radius-lg: 16px;
  --radius-pill: 999px;
  --shadow-sm: 0 1px 2px {_rgba(shadow_ink, 0.08 if not theme.dark else 0.5)};
  --shadow-md: 0 2px 4px {_rgba(shadow_ink, 0.05 if not theme.dark else 0.4)}, 0 10px 28px -8px {_rgba(shadow_ink, 0.16 if not theme.dark else 0.6)};
  --shadow-lg: 0 20px 48px -16px {_rgba(shadow_ink, 0.22 if not theme.dark else 0.7)};
  /* layout */
  --container: 1120px;
  --container-narrow: 720px;
  --sidebar-width: 260px;
  --header-height: 60px;
  --transition: 160ms ease;
}}

/* ---- base ------------------------------------------------------------- */
*, *::before, *::after {{ box-sizing: border-box; }}
html, body, #root, #app {{ height: 100%; }}
body {{
  margin: 0;
  background: var(--color-bg);
  color: var(--color-text);
  font-family: var(--font-body);
  font-size: var(--text-base);
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
}}
h1, h2, h3, h4 {{
  font-family: var(--font-heading);
  line-height: 1.15;
  letter-spacing: -0.01em;
  margin: 0 0 var(--space-3);
  color: var(--color-text);
}}
h1 {{ font-size: var(--text-3xl); }}
h2 {{ font-size: var(--text-2xl); }}
h3 {{ font-size: var(--text-xl); }}
h4 {{ font-size: var(--text-lg); }}
p {{ margin: 0 0 var(--space-3); }}
a {{ color: var(--color-link); text-decoration: none; }}
a:hover {{ text-decoration: underline; text-underline-offset: 3px; }}
code, pre, kbd {{ font-family: var(--font-mono); font-size: 0.92em; }}
pre {{
  background: var(--color-surface-2);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  padding: var(--space-4);
  overflow: auto;
}}
button, input, select, textarea {{ font: inherit; color: inherit; }}
button {{ cursor: pointer; }}
button:disabled {{ cursor: not-allowed; opacity: 0.6; }}
img, svg, video {{ max-width: 100%; display: block; }}
:focus-visible {{ outline: 2px solid var(--color-primary); outline-offset: 2px; border-radius: var(--radius-sm); }}
::selection {{ background: var(--color-primary-soft); }}

/* ---- layout helpers ------------------------------------------------------ */
.container {{ width: 100%; max-width: var(--container); margin: 0 auto; padding: 0 var(--space-6); }}
.container-narrow {{ width: 100%; max-width: var(--container-narrow); margin: 0 auto; padding: 0 var(--space-6); }}
.stack {{ display: flex; flex-direction: column; gap: var(--space-4); }}
.row {{ display: flex; align-items: center; gap: var(--space-3); }}
.row-between {{ display: flex; align-items: center; justify-content: space-between; gap: var(--space-3); }}
.grid {{ display: grid; gap: var(--space-4); grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); }}
.muted {{ color: var(--color-muted); }}
.text-sm {{ font-size: var(--text-sm); }}
.text-xs {{ font-size: var(--text-xs); }}
.mono {{ font-family: var(--font-mono); }}
.visually-hidden {{ position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }}

/* ---- app shell ------------------------------------------------------------ */
.app-shell {{ min-height: 100%; display: flex; flex-direction: column; }}
.app-shell--sidebar {{ display: grid; grid-template-columns: var(--sidebar-width) 1fr; min-height: 100%; }}
.topbar {{
  height: var(--header-height);
  display: flex; align-items: center; justify-content: space-between; gap: var(--space-4);
  padding: 0 var(--space-6);
  background: var(--color-surface);
  border-bottom: 1px solid var(--color-border);
  position: sticky; top: 0; z-index: 20;
}}
.brand {{ display: flex; align-items: center; gap: var(--space-2); font-family: var(--font-heading); font-weight: 700; font-size: var(--text-lg); color: var(--color-text); }}
.brand-mark {{ width: 28px; height: 28px; border-radius: 8px; background: linear-gradient(135deg, var(--color-primary), var(--color-accent)); }}
.nav {{ display: flex; align-items: center; gap: var(--space-1); }}
.nav-link {{
  display: inline-flex; align-items: center; gap: var(--space-2);
  padding: var(--space-2) var(--space-3);
  border-radius: var(--radius-sm);
  color: var(--color-muted); font-weight: 500; font-size: var(--text-sm);
  transition: background var(--transition), color var(--transition);
}}
.nav-link:hover {{ color: var(--color-text); background: var(--color-surface-2); text-decoration: none; }}
.nav-link.active, .nav-link[aria-current="page"] {{ color: var(--color-link); background: var(--color-primary-soft); }}
.sidebar {{ background: var(--color-surface); border-right: 1px solid var(--color-border); padding: var(--space-6) var(--space-4); display: flex; flex-direction: column; gap: var(--space-2); }}
.sidebar .nav {{ flex-direction: column; align-items: stretch; }}
.main {{ flex: 1; padding: var(--space-8) var(--space-6); width: 100%; }}
.page-header {{ display: flex; align-items: flex-end; justify-content: space-between; gap: var(--space-4); margin-bottom: var(--space-6); }}
.page-title {{ margin: 0; }}
.page-subtitle {{ color: var(--color-muted); margin: var(--space-1) 0 0; }}

/* ---- components ---------------------------------------------------------- */
.card {{
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  padding: var(--space-6);
  box-shadow: var(--shadow-sm);
}}
.card--hover {{ transition: transform var(--transition), box-shadow var(--transition); }}
.card--hover:hover {{ transform: translateY(-2px); box-shadow: var(--shadow-md); }}
.btn {{
  display: inline-flex; align-items: center; justify-content: center; gap: var(--space-2);
  padding: 0.625rem 1rem;
  border-radius: var(--radius-md);
  border: 1px solid transparent;
  font-weight: 600; font-size: var(--text-sm); line-height: 1.2;
  background: var(--color-surface); color: var(--color-text);
  transition: background var(--transition), border-color var(--transition), transform var(--transition), box-shadow var(--transition);
}}
.btn:hover {{ background: var(--color-surface-2); }}
.btn:active {{ transform: translateY(1px); }}
.btn-primary {{ background: var(--color-primary); color: var(--color-primary-contrast); border-color: var(--color-primary); box-shadow: var(--shadow-sm); }}
.btn-primary:hover {{ background: var(--color-primary-hover); border-color: var(--color-primary-hover); }}
.btn-ghost {{ background: transparent; border-color: var(--color-border); color: var(--color-text); }}
.btn-ghost:hover {{ background: var(--color-surface-2); }}
.btn-danger {{ background: var(--color-danger-soft); color: var(--color-danger); border-color: transparent; }}
.btn-sm {{ padding: 0.375rem 0.75rem; font-size: var(--text-xs); }}
.btn-icon {{ padding: 0.5rem; border-radius: var(--radius-md); }}
.field {{ display: flex; flex-direction: column; gap: var(--space-2); }}
.label {{ font-size: var(--text-sm); font-weight: 600; color: var(--color-text); }}
.hint {{ font-size: var(--text-xs); color: var(--color-muted); }}
.input, .select, .textarea {{
  width: 100%;
  padding: 0.625rem 0.875rem;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  background: var(--color-surface);
  color: var(--color-text);
  transition: border-color var(--transition), box-shadow var(--transition);
}}
.input::placeholder, .textarea::placeholder {{ color: var(--color-muted); opacity: 0.8; }}
.input:focus, .select:focus, .textarea:focus {{ outline: none; border-color: var(--color-primary); box-shadow: 0 0 0 3px var(--color-primary-soft); }}
.input[aria-invalid="true"] {{ border-color: var(--color-danger); }}
.textarea {{ min-height: 96px; resize: vertical; }}
.badge {{ display: inline-flex; align-items: center; gap: var(--space-1); padding: 0.125rem 0.625rem; border-radius: var(--radius-pill); font-size: var(--text-xs); font-weight: 600; background: var(--color-primary-soft); color: var(--color-link); }}
.badge--success {{ background: {_rgba(sem["success"], 0.14)}; color: var(--color-success); }}
.badge--warning {{ background: {_rgba(sem["warning"], 0.16)}; color: var(--color-warning); }}
.badge--danger {{ background: var(--color-danger-soft); color: var(--color-danger); }}
.alert {{ padding: var(--space-3) var(--space-4); border-radius: var(--radius-md); border: 1px solid var(--color-border); background: var(--color-surface-2); font-size: var(--text-sm); }}
.alert--error {{ background: var(--color-danger-soft); border-color: transparent; color: var(--color-danger); }}
.alert--success {{ background: {_rgba(sem["success"], 0.12)}; border-color: transparent; color: var(--color-success); }}
.empty-state {{
  display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center;
  gap: var(--space-3); padding: var(--space-12) var(--space-6);
  border: 1px dashed var(--color-border); border-radius: var(--radius-lg);
  color: var(--color-muted);
}}
.empty-state h3 {{ color: var(--color-text); margin: 0; }}
.spinner {{
  width: 18px; height: 18px; border-radius: 50%;
  border: 2px solid var(--color-border); border-top-color: var(--color-primary);
  animation: spin 0.8s linear infinite;
}}
@keyframes spin {{ to {{ transform: rotate(360deg); }} }}
.skeleton {{ background: linear-gradient(90deg, var(--color-surface-2), var(--color-border), var(--color-surface-2)); background-size: 200% 100%; animation: shimmer 1.4s infinite; border-radius: var(--radius-sm); }}
@keyframes shimmer {{ to {{ background-position: -200% 0; }} }}
.divider {{ border: 0; border-top: 1px solid var(--color-border); margin: var(--space-6) 0; }}
.table {{ width: 100%; border-collapse: collapse; font-size: var(--text-sm); }}
.table th {{ text-align: left; font-weight: 600; color: var(--color-muted); padding: var(--space-3); border-bottom: 1px solid var(--color-border); }}
.table td {{ padding: var(--space-3); border-bottom: 1px solid var(--color-border); }}
.table tr:hover td {{ background: var(--color-surface-2); }}

/* ---- responsive ---------------------------------------------------------- */
@media (max-width: 768px) {{
  .app-shell--sidebar {{ grid-template-columns: 1fr; }}
  .sidebar {{ border-right: 0; border-bottom: 1px solid var(--color-border); flex-direction: row; overflow-x: auto; }}
  .sidebar .nav {{ flex-direction: row; }}
  .main {{ padding: var(--space-5) var(--space-4); }}
  .page-header {{ flex-direction: column; align-items: flex-start; }}
  .topbar {{ padding: 0 var(--space-4); }}
  h1 {{ font-size: var(--text-2xl); }}
}}
@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{ animation-duration: 0.01ms !important; transition-duration: 0.01ms !important; }}
}}
"""


def tailwind_entry_stylesheet(theme: Theme) -> str:
    """Tailwind v4 entry stylesheet carrying the theme as ``@theme`` tokens.

    One ``@import "tailwindcss";`` is a complete v4 setup; the ``@theme`` block
    exposes the palette and fonts as utilities (``bg-primary``, ``text-muted``,
    ``font-heading`` ...) so the model styles with the curated system instead
    of improvising gray-on-white.
    """
    p = theme.palette
    sem = _semantic_colors(theme)
    link = theme.fonts.get("google_link", "")
    return f"""/* Tailwind v4 entry -- generated by CGX from the "{theme.name}" theme.
   {theme.vibe}
   Use the token utilities below (bg-primary, text-muted, font-heading ...);
   do not hard-code colours. */
@import url("{link}");
@import "tailwindcss";

@theme {{
  --color-bg: {p["bg"]};
  --color-surface: {p["surface"]};
  --color-text: {p["text"]};
  --color-muted: {p["muted"]};
  --color-primary: {p["primary"]};
  --color-primary-contrast: {p["primary_contrast"]};
  --color-accent: {p["accent"]};
  --color-border: {p["border"]};
  --color-success: {sem["success"]};
  --color-warning: {sem["warning"]};
  --color-danger: {sem["danger"]};
  --font-heading: "{theme.fonts.get("heading", "Inter")}", ui-sans-serif, system-ui, sans-serif;
  --font-body: "{theme.fonts.get("body", "Inter")}", ui-sans-serif, system-ui, sans-serif;
  --font-mono: "{theme.fonts.get("mono", "JetBrains Mono")}", ui-monospace, monospace;
  --radius-md: 10px;
  --radius-lg: 16px;
}}

@layer base {{
  html {{ color-scheme: {"dark" if theme.dark else "light"}; }}
  body {{ @apply bg-bg text-text font-body antialiased; }}
  h1, h2, h3, h4 {{ @apply font-heading tracking-tight; }}
  :focus-visible {{ @apply outline-2 outline-primary outline-offset-2; }}
}}
"""


def theme_prompt_block(theme: Theme, *, styling: str = "css-tokens") -> str:
    """Render the theme into hard design constraints for the generator."""
    p = theme.palette
    scheme = "DARK theme (dark backgrounds, light text)" if theme.dark else "LIGHT theme"
    if styling == "tailwind":
        how = ("Style with Tailwind utilities backed by the generated @theme tokens: "
               "bg-bg / bg-surface / text-text / text-muted / bg-primary text-primary-contrast / "
               "text-accent / border-border / font-heading / font-body. Never type raw hex colours.")
    else:
        how = ("Style with the generated tokens stylesheet: use its classes (.topbar .brand .nav-link "
               ".main .page-header .card .btn .btn-primary .btn-ghost .field .label .input .textarea "
               ".badge .alert .empty-state .spinner .table .container .stack .row) and CSS variables "
               "(var(--color-primary), var(--color-surface), var(--color-border), var(--color-muted), "
               "var(--space-4), var(--radius-md), var(--shadow-sm), var(--font-heading)). Never type raw "
               "hex colours in a component or stylesheet.")
    return (
        f"DESIGN SYSTEM (binding) -- theme \"{theme.name}\": {theme.vibe} This is a {scheme}.\n"
        f"- Palette (already defined as tokens): bg {p['bg']}, surface {p['surface']}, text {p['text']}, "
        f"muted {p['muted']}, primary {p['primary']} (label colour {p['primary_contrast']}), accent "
        f"{p['accent']}, border {p['border']}.\n"
        f"- Fonts (already loaded by the tokens stylesheet): headings '{theme.fonts.get('heading')}', "
        f"body '{theme.fonts.get('body')}'.\n"
        f"- {how}\n"
        f"- Visual language: {theme.style}"
    )
