"""Focused single-shot generator for the Site Studio.

The generic greenfield pipeline (manifest -> per-file) manufactures thin,
generic stubs for a "build me a website" request. For a static site a single
strong, design-oriented prompt that emits ONE self-contained ``index.html``
produces a far better result -- and, being self-contained (styles + scripts
inline, imagery inline/CDN/remote), it has no local files to break, previews
instantly, and needs no build. This module is what the Studio calls; the
`static_site` skill + verifier still guard Agent-Loop static builds.
"""

from __future__ import annotations

import logging
import re
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["generate_site", "FLAVOR_STYLE", "DEFAULT_MAX_TOKENS"]

DEFAULT_MAX_TOKENS = 8000

# Per-flavor styling directive spliced into the system prompt.
FLAVOR_STYLE = {
    "simple": (
        "STYLING: hand-write clean, modern CSS inside a single <style> block "
        "in <head> (NO external CSS/JS files, NO CDN -- it must work opened "
        "directly from disk). Use a tasteful color palette, system fonts, "
        "flexbox/grid, generous spacing."
    ),
    "modern": (
        "STYLING: use Tailwind CSS via the Play CDN "
        "(<script src=\"https://cdn.tailwindcss.com\"></script> in <head>) and "
        "a Google Font. Aim for a polished, modern, editorial look: strong "
        "type hierarchy, cohesive palette, rounded cards, subtle shadows, "
        "hover/transition states."
    ),
    "interactive": (
        "STYLING: use Tailwind CSS via the Play CDN and a Google Font, and add "
        "Alpine.js via CDN (<script defer "
        "src=\"https://unpkg.com/alpinejs@3/dist/cdn.min.js\"></script>) for "
        "interactivity (mobile nav toggle, tabs, accordions, simple state). "
        "Polished, modern, responsive."
    ),
}

_BASE_RULES = (
    "You are an elite web designer and front-end developer. Produce ONE "
    "complete, self-contained, production-quality `index.html` document for "
    "the website the user describes.\n"
    "HARD REQUIREMENTS:\n"
    "- Output ONLY the HTML document, starting with <!DOCTYPE html>. No prose, "
    "no explanations, no markdown code fences.\n"
    "- Everything inline in this one file: no references to local .css/.js "
    "files, no build step, no package.json.\n"
    "CONTRAST & LEGIBILITY (critical): every piece of text MUST be clearly "
    "readable -- NEVER light text on a light background or dark on dark. Give "
    "each section an explicit background (a solid color or gradient) and put "
    "contrasting text on it. Double-check the hero.\n"
    "OFFLINE-SAFE VISUALS: assume there is NO network. Build backgrounds and "
    "decoration from CSS gradients, solid colors, CSS patterns, and inline "
    "<svg> -- these always render. Do NOT depend on a remote image to carry a "
    "section: if you use a remote <img>/background-image (Unsplash/picsum/"
    "placehold.co), ALWAYS put a colored gradient BEHIND it so the section "
    "still looks designed and text stays legible when the image fails to "
    "load. NEVER reference a local image file (it would 404).\n"
    "FULL, RICH PAGE: no large empty voids -- the page must feel complete from "
    "top to bottom with substantial, REAL, specific copy (never 'Lorem "
    "ipsum'). Include a sticky header with nav; a bold hero (headline + "
    "subtext + CTA) that fills the first screen with a designed background; "
    "then 4-6 full content sections with genuine detail appropriate to the "
    "topic (e.g. featured menu with named items + prices, our story, "
    "gallery, testimonials, hours/location); and a footer.\n"
    "WORKING NAV: if the request mentions multiple 'pages' (e.g. menu, order, "
    "contact), realise each as a full <section id=\"...\"> in THIS page, and "
    "make every nav link an in-page anchor (href=\"#menu\") that points to a "
    "section that actually exists. Every link must resolve.\n"
    "- Semantic, accessible, fully responsive (mobile-first).\n"
)

_REVISE_RULES = (
    "You are an elite web designer editing an existing single-file website. "
    "Apply the user's requested change and return the ENTIRE updated "
    "index.html document (starting with <!DOCTYPE html>, no fences, no prose). "
    "Preserve everything that works; change only what the feedback asks for, "
    "plus anything needed to keep it coherent. Keep it a single self-contained "
    "file with the same no-local-files / inline-or-remote-imagery rules."
)


def _strip_fences(text: str) -> str:
    t = (text or "").strip()
    # Drop a leading ```html / ``` fence and a trailing ``` if present.
    t = re.sub(r"^```[a-zA-Z0-9]*\s*\n", "", t)
    t = re.sub(r"\n```\s*$", "", t)
    # If the model prepended prose, cut to the first doctype/html tag.
    m = re.search(r"<!DOCTYPE html>|<html[ >]", t, re.IGNORECASE)
    if m and m.start() > 0:
        t = t[m.start():]
    return t.strip()


def generate_site(
    brief: str,
    provider: Any,
    *,
    flavor: str = "modern",
    theme_key: Optional[str] = None,
    current_html: Optional[str] = None,
    feedback: Optional[str] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Return a complete self-contained ``index.html`` for ``brief``.

    Generation is constrained to a curated design theme (selected from the
    brief, or ``theme_key``) so the palette/fonts are professional rather than
    model-improvised. In *revise* mode (``current_html`` + ``feedback``) the
    document is edited per the feedback, keeping the same theme. Raises
    ``ValueError`` if the model returns nothing usable.
    """
    from cgx.answer.web_themes import select_theme, theme_prompt_block
    theme = select_theme(brief or (feedback or ""), theme_key)
    theme_block = theme_prompt_block(theme)
    style = FLAVOR_STYLE.get(flavor, FLAVOR_STYLE["modern"])
    if current_html and feedback:
        system = f"{_REVISE_RULES}\n{style}\n\n{theme_block}"
        user = (
            f"CURRENT index.html:\n{current_html}\n\n"
            f"REQUESTED CHANGE:\n{feedback}\n\n"
            "Return the full updated index.html."
        )
    else:
        system = f"{_BASE_RULES}{style}\n\n{theme_block}"
        user = f"Build the website for: {brief.strip()}."

    resp = provider.chat(
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.4, force_json=False, max_tokens=int(max_tokens),
    )
    html = _strip_fences(resp.get("content") if isinstance(resp, dict) else "")
    if "<" not in html or "html" not in html.lower():
        raise ValueError("model did not return an HTML document")
    return html


# ---------------------------------------------------------------------------
# Agentic design critique -> self-repair. A one-shot ships its first draft;
# the agent reviews its own output against a rubric and revises until it
# passes (bounded). Text-only for now (no headless browser to screenshot);
# a vision model on a rendered screenshot is the future upgrade.
# ---------------------------------------------------------------------------

_CRITIC_SYSTEM = (
    "You are a strict senior design reviewer. You are given an HTML document "
    "and the design theme it MUST follow. Judge whether the page looks "
    "professionally designed and actually uses the theme. Check for these "
    "common AI failure modes:\n"
    "- Low-contrast / invisible text (light-on-light, dark-on-dark).\n"
    "- Bland: not using the theme palette (--primary/--accent), default look, "
    "gray-on-white.\n"
    "- Weak hierarchy / not using the theme fonts.\n"
    "- Empty voids / thin content / too few sections.\n"
    "- A hero with no designed background (should use the theme gradient).\n"
    "- Missing hover/spacing polish.\n"
    "Reply with EITHER the single word PASS (if it is genuinely well-designed "
    "and on-theme), OR up to 6 short, SPECIFIC, actionable fixes as plain "
    "lines (no numbering, no prose)."
)


def critique_site(html: str, theme: Any, provider: Any) -> List[str]:
    """Return a list of specific design fixes, or ``[]`` when it passes."""
    try:
        from cgx.answer.web_themes import theme_prompt_block
        resp = provider.chat(
            messages=[{"role": "system", "content": _CRITIC_SYSTEM},
                      {"role": "user", "content":
                       f"THEME:\n{theme_prompt_block(theme)}\n\nHTML:\n{html}"}],
            temperature=0.0, force_json=False, max_tokens=600)
        text = (resp.get("content") if isinstance(resp, dict) else "") or ""
    except Exception as e:  # noqa: BLE001
        logger.debug("design critique failed: %s", e)
        return []
    if text.strip().upper().startswith("PASS"):
        return []
    fixes: List[str] = []
    for ln in text.splitlines():
        s = ln.strip().lstrip("-*•0123456789. ").strip()
        if len(s) > 4:
            fixes.append(s)
    return fixes[:6]


def refine_site(
    html: str, provider: Any, *, flavor: str = "modern",
    theme_key: Optional[str] = None, rounds: int = 1,
) -> str:
    """Critique the page and self-repair design issues (bounded rounds)."""
    from cgx.answer.web_themes import select_theme
    theme = select_theme("", theme_key) if theme_key else select_theme(html[:400])
    for _ in range(max(0, rounds)):
        fixes = critique_site(html, theme, provider)
        if not fixes:
            break
        feedback = ("A design review flagged these issues -- fix ALL of them "
                    "while keeping the theme and everything that works:\n- "
                    + "\n- ".join(fixes))
        try:
            html = generate_site("", provider, flavor=flavor,
                                 theme_key=theme.key, current_html=html,
                                 feedback=feedback)
        except ValueError:
            break
    return html
