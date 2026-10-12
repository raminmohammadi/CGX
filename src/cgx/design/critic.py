"""LLM design critic: one bounded review of the UI files against the brief.

The deterministic checks catch what is *mechanically* wrong (no stylesheet,
no className, no persistence). A design review catches what is merely *weak*:
a settings page that is a single input, a chat without a composer, no
hierarchy, no states. The critic sees the brief and the UI files, and returns
concrete per-file fixes the verifier threads into targeted regenerations
(``REQUIRED FIX`` directives) -- never a blind re-ask.

Bounded on purpose (one call, temperature 0, capped input/output) so it fits a
local model and cannot loop. Disabled with ``CGX_DESIGN_CRITIC=0``.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List

from cgx.design.brief import DesignBrief
from cgx.design.prompts import file_kind

logger = logging.getLogger(__name__)

__all__ = ["critic_enabled", "critique_ui", "parse_fixes", "MAX_FIXES"]

MAX_FIXES = 6
_PER_FILE_CHARS = 3500
_TOTAL_CHARS = 16000

_SYSTEM = (
    "You are a senior product designer and front-end lead reviewing a generated UI against its product "
    "spec. Judge COMPLETENESS and POLISH: app shell + navigation, visual hierarchy, every required screen and "
    "component present, loading/empty/error/success states, labelled inputs, persisted settings, correct "
    "integration, real copy, responsive layout, use of the design tokens/classes. Ignore code style.\n"
    "Output rules: if the UI meets the spec well, reply exactly PASS. Otherwise reply with up to "
    f"{MAX_FIXES} lines, each formatted as `<file path>: <one concrete, additive fix>`, most important first. "
    "Only name files that exist in the list you were given. No prose, no headings, no markdown."
)


def critic_enabled() -> bool:
    return (os.environ.get("CGX_DESIGN_CRITIC") or "1").strip().lower() not in ("0", "false", "no", "off")


def _brief(b: Any) -> DesignBrief:
    return b if isinstance(b, DesignBrief) else DesignBrief.from_dict(b)


def _ui_files(b: DesignBrief, files: Dict[str, str]) -> Dict[str, str]:
    keep = {}
    for p, c in (files or {}).items():
        k = file_kind(b, p)
        if k in ("component", "app", "entry", "stylesheet", "html", "api", "storage", "script"):
            keep[p] = c or ""
    return keep


def critique_ui(brief: Any, files: Dict[str, str], provider: Any) -> Dict[str, str]:
    """Return ``{path: fix directive}`` (empty when PASS / unavailable)."""
    b = _brief(brief)
    if not b.applies or provider is None:
        return {}
    ui = _ui_files(b, files)
    if not ui:
        return {}
    budget = _TOTAL_CHARS
    blocks: List[str] = []
    for p in sorted(ui, key=lambda x: (file_kind(b, x) != "component", x)):
        body = ui[p]
        if len(body) > _PER_FILE_CHARS:
            body = body[:_PER_FILE_CHARS] + "\n... [truncated]"
        if budget - len(body) < 0:
            break
        budget -= len(body)
        blocks.append(f"### {p}\n{body}")
    spec = (f"Product: {b.product_name} ({b.archetype_name}). Layout: {b.layout}\nScreens: {'; '.join(b.screens)}\n"
            f"Components: {'; '.join(b.components)}\nRequirements:\n- " + "\n- ".join(b.requirements[-12:]))
    if b.integration_notes:
        spec += "\nIntegration:\n- " + "\n- ".join(b.integration_notes)
    user = (f"SPEC:\n{spec}\n\nFILES ({len(blocks)} shown; only these paths may be named):\n"
            + "\n".join(f"- {p}" for p in ui) + "\n\n" + "\n\n".join(blocks))
    try:
        resp = provider.chat(messages=[{"role": "system", "content": _SYSTEM},
                                       {"role": "user", "content": user}],
                             temperature=0.0, force_json=False, max_tokens=700)
        text = str((resp or {}).get("content") or "") if isinstance(resp, dict) else ""
    except Exception as exc:  # pragma: no cover - provider failure is non-fatal
        logger.debug("design critic failed: %s", exc)
        return {}
    return parse_fixes(text, list(ui))


def parse_fixes(text: str, known_paths: List[str]) -> Dict[str, str]:
    """Parse ``<file>: <fix>`` lines into ``{known path: fix}`` (``PASS`` -> ``{}``)."""
    text = (text or "").strip()
    if not text or text.upper().startswith("PASS"):
        return {}
    fixes: Dict[str, str] = {}
    known = {p.lower(): p for p in known_paths}
    for ln in text.splitlines():
        s = ln.strip().lstrip("-*•0123456789. ").strip().strip("`")
        if ":" not in s:
            continue
        path, fix = s.split(":", 1)
        path = path.strip().strip("`").replace("\\", "/")
        fix = fix.strip()
        if len(fix) < 8:
            continue
        match = known.get(path.lower())
        if match is None:
            # tolerate a bare basename
            for low, real in known.items():
                if low.endswith("/" + path.lower()) or low == path.lower():
                    match = real
                    break
        if match is None:
            continue
        fixes[match] = (fixes[match] + " " + fix).strip() if match in fixes else fix
        if len(fixes) >= MAX_FIXES:
            break
    return fixes
