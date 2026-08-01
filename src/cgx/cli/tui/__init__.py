

"""CGX interactive terminal dashboard.

A stdlib-only, Gemini-CLI / Qwen-Code style REPL: an ASCII banner, a
bordered input box, top/bottom status bars, and slash commands. Plain
messages drive the session agent loop (:mod:`cgx.session`) so a single
prompt can start a session, answer an open question, or post a
follow-up objective.

Rendering (:mod:`cgx.cli.tui.render`) is kept as pure string-returning
functions so it can be unit-tested without a terminal; the interactive
shell lives in :mod:`cgx.cli.tui.app`.
"""

from __future__ import annotations

from cgx.cli.tui.app import Dashboard, run_dashboard

__all__ = ["Dashboard", "run_dashboard"]
