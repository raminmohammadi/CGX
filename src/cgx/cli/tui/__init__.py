

"""CGX interactive terminal dashboard.

A stdlib-only, Gemini-CLI / Qwen-Code style REPL: an ASCII banner, a
bordered input box, top/bottom status bars, and slash commands. Plain
messages are routed through the Planner -> Tracker -> Judge loop
(:func:`cgx.agents.loop.run_agent`) so a single prompt can ask a
question, plan an edit, or scaffold a new project.

Rendering (:mod:`cgx.cli.tui.render`) is kept as pure string-returning
functions so it can be unit-tested without a terminal; the interactive
shell lives in :mod:`cgx.cli.tui.app`.
"""

from __future__ import annotations

from cgx.cli.tui.app import Dashboard, run_dashboard

__all__ = ["Dashboard", "run_dashboard"]
