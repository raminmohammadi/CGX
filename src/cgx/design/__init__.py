"""The CGX design layer: product + design intelligence for UI-bearing builds.

Why this exists: asked for "an app for a chatbot with an API-key profile tab
and a chat page", the agent produced an unstyled white page with two buttons,
a form that ``console.log``'d the key and a chat posting to a hard-coded
``/api/chat`` -- and the verifier called it *verified*. Nothing in the
pipeline knew what a real product looks like.

This package teaches that, deterministically, so it works for weak local
models:

* :mod:`cgx.design.archetypes` -- what a finished chat app / dashboard /
  landing page / tool contains (layout, screens, components, checkable UX
  requirements, integration rules).
* :mod:`cgx.design.themes` -- curated, contrast-verified design systems and a
  tokens stylesheet CGX writes itself (variables + base component classes).
* :mod:`cgx.design.brief` -- ``build_brief(goal, skills)``: one spec per build
  (framework, styling, archetype, theme, canonical file paths), persisted in
  the WORK_PLAN and threaded to the Developer and Verifier.
* :mod:`cgx.design.prompts` -- planner and per-file prompt blocks.
* :mod:`cgx.design.checks` -- plan invariants (inject the design-system files)
  and deterministic UI-quality checks the verifier gates on.
* :mod:`cgx.design.critic` -- one bounded LLM design review producing targeted
  regen directives.
"""

from cgx.design.archetypes import ARCHETYPES, GENERIC, Archetype, select_archetype  # noqa: F401
from cgx.design.brief import (  # noqa: F401
    UNIVERSAL_REQUIREMENTS, DesignBrief, applies, build_brief, detect_framework)
from cgx.design.checks import (  # noqa: F401
    DesignIssue, check_tree, fatal_issues, is_tokens_path, issues_as_warnings,
    plan_problems, tokens_content, ui_files_to_inject)
from cgx.design.critic import critic_enabled, critique_ui  # noqa: F401
from cgx.design.prompts import file_block, file_kind, is_ui_file, planner_block  # noqa: F401
from cgx.design.themes import (  # noqa: F401
    THEMES, Theme, contrast_ratio, select_theme, theme_by_key, theme_summaries,
    tokens_stylesheet)


def gate_enabled() -> bool:
    """Whether fatal design issues gate the verify verdict (``CGX_DESIGN_GATE``)."""
    import os
    return (os.environ.get("CGX_DESIGN_GATE") or "1").strip().lower() not in ("0", "false", "no", "off")


__all__ = [
    "ARCHETYPES", "GENERIC", "Archetype", "select_archetype",
    "UNIVERSAL_REQUIREMENTS", "DesignBrief", "applies", "build_brief", "detect_framework",
    "DesignIssue", "check_tree", "fatal_issues", "is_tokens_path", "issues_as_warnings",
    "plan_problems", "tokens_content", "ui_files_to_inject",
    "critic_enabled", "critique_ui", "gate_enabled",
    "file_block", "file_kind", "is_ui_file", "planner_block",
    "THEMES", "Theme", "contrast_ratio", "select_theme", "theme_by_key", "theme_summaries",
    "tokens_stylesheet",
]
