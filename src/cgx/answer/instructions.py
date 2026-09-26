"""Compose the user-instruction preamble injected into LLM system prompts.

This is the single place that turns a project's ``CGX.md`` context file and
its active skills into a bounded block of plain system-prompt text. It is
used by the Ask/chat path and the read-only agent tasks so both honor the
same house rules and domain knowledge.

Two hard rules, both for the sake of small local models:

* **Deterministic, not tool-driven.** Skills are resolved by explicit pin
  or keyword ``detect()`` -- never by asking the model to choose.
* **Bounded.** The block is capped to ``max_chars`` and, at the call site,
  its length is subtracted from the retrieval SOURCES budget, so a long
  ``CGX.md`` can never silently push a model's citations past ``num_ctx``.
"""

from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

__all__ = ["build_instruction_preamble"]

_CGX_HEADER = (
    "PROJECT INSTRUCTIONS (from CGX.md -- always follow these for this repo; "
    "they take precedence over generic defaults, but never invent facts):"
)
_SKILLS_HEADER = (
    "PROJECT SKILLS (user-authored guidance; apply the ones relevant to the "
    "question):"
)
_NOTES_HEADER = (
    "DIRECTORY NOTES (footguns for the specific directories this task touches -- "
    "follow them for files in those dirs):"
)
_TRUNC = "\n[... truncated ...]"


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(_TRUNC))].rstrip() + _TRUNC


def build_instruction_preamble(
    project_root: Optional[str],
    question: str,
    pinned_skills: Optional[List[str]] = None,
    *,
    surface: str = "chat",
    max_chars: int = 4000,
    files_touched: Optional[List[str]] = None,
) -> str:
    """Return the composed ``CGX.md`` + active-skills preamble, or ``""``.

    ``surface`` is currently ``"chat"`` (conversational answering + read-only
    agent tasks). ``pinned_skills`` forces a specific skill set; when ``None``
    skills are keyword-detected from ``question`` (plus any ``always_on`` /
    per-repo skills). ``files_touched`` (JEV conditional instructions) pulls in
    per-directory ``GOTCHAS.md`` notes for exactly the dirs those files live in,
    appended at the TAIL so the always-on CGX.md + skills prefix stays stable.
    Never raises -- a failure yields ``""``.
    """
    if max_chars <= 0:
        return ""

    cgx_text = ""
    try:
        from cgx.context_files import load_project_context
        cgx_text = load_project_context(project_root)
    except Exception as e:  # noqa: BLE001
        logger.debug("instructions: CGX.md load failed: %s", e)

    skill_text = ""
    try:
        import skills as _sk
        if pinned_skills:
            active = _sk.skills_by_names(list(pinned_skills), project_root=project_root)
        else:
            active = _sk.detect_skills(question or "", project_root=project_root)
        if surface == "chat":
            skill_text = _sk.compose_ask_prompt(active)
        elif surface == "plan":
            skill_text = _sk.compose_plan_prompt(active)
        elif surface == "scaffold":
            skill_text = _sk.compose_scaffold_prompt(active)
    except Exception as e:  # noqa: BLE001
        logger.debug("instructions: skill composition failed: %s", e)

    # Conditional directory notes (JEV Section VIII): footguns for exactly the
    # dirs this task touches. Loaded on its own small budget so it never eats
    # the CGX.md/skills budget, and appended at the TAIL below.
    notes_text = ""
    if files_touched:
        try:
            from cgx.context_files import load_directory_notes
            notes_text = load_directory_notes(
                project_root, files_touched,
                max_chars=min(1500, max(300, max_chars // 3)))
        except Exception as e:  # noqa: BLE001
            logger.debug("instructions: directory notes failed: %s", e)

    if not cgx_text and not skill_text and not notes_text:
        return ""

    # Budget split: keep CGX.md primary (it is the always-on repo contract),
    # give skills the remainder. Reserve a little for the headers.
    overhead = len(_CGX_HEADER) + len(_SKILLS_HEADER) + 8
    usable = max(0, max_chars - overhead)
    cgx_cap = usable if not skill_text else int(usable * 0.6)
    cgx_text = _clip(cgx_text, cgx_cap)
    skill_cap = usable - len(cgx_text)
    skill_text = _clip(skill_text, skill_cap)

    parts: List[str] = []
    if cgx_text:
        parts.append(f"{_CGX_HEADER}\n{cgx_text}")
    if skill_text:
        parts.append(f"{_SKILLS_HEADER}\n{skill_text}")
    if notes_text:
        parts.append(f"{_NOTES_HEADER}\n{notes_text}")
    return "\n\n".join(parts)
