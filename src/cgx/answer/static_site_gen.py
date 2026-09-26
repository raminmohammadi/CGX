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
from typing import Any, Optional

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
    "- IMAGERY you may use ONLY: inline <svg>, CSS gradients/patterns, emoji, "
    "or REMOTE image URLs (https://images.unsplash.com/... , "
    "https://picsum.photos/... , https://placehold.co/...). NEVER reference a "
    "local image file (it would 404).\n"
    "- Substantial, REAL, specific copy tailored to the topic -- never 'Lorem "
    "ipsum', never a single bare form. Include, as appropriate to the request: "
    "a sticky header with nav; a bold hero (headline + subtext + CTA); 3-5 "
    "content sections with genuine detail (e.g. features/menu/services with "
    "names + specifics, an about/story, testimonials, pricing, gallery, "
    "hours/location); a contact section; and a footer.\n"
    "- Semantic, accessible, fully responsive (mobile-first). Smooth in-page "
    "anchor nav.\n"
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
    current_html: Optional[str] = None,
    feedback: Optional[str] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Return a complete self-contained ``index.html`` for ``brief``.

    In *revise* mode (``current_html`` + ``feedback`` given) the current
    document is edited per the feedback and the full updated document is
    returned. Raises ``ValueError`` if the model returns nothing usable.
    """
    style = FLAVOR_STYLE.get(flavor, FLAVOR_STYLE["modern"])
    if current_html and feedback:
        system = f"{_REVISE_RULES}\n{style}"
        user = (
            f"CURRENT index.html:\n{current_html}\n\n"
            f"REQUESTED CHANGE:\n{feedback}\n\n"
            "Return the full updated index.html."
        )
    else:
        system = f"{_BASE_RULES}{style}"
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
