

"""Pure string renderers for the CGX dashboard.

Every function here returns a ``str`` and never touches stdin/stdout, so
the layout can be asserted in unit tests. Colour is controlled by the
``enabled`` flag threaded down from :func:`cgx.cli.tui.ansi.color_enabled`.
"""

from __future__ import annotations

import os
import re
from typing import List, Optional

from cgx.cli.tui import ansi

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# "ANSI Shadow" block-letters for CGX.
_BANNER_ROWS: List[str] = [
    " ██████╗ ██████╗ ██╗  ██╗",
    "██╔════╝██╔════╝ ╚██╗██╔╝",
    "██║     ██║  ███╗ ╚███╔╝ ",
    "██║     ██║   ██║ ██╔██╗ ",
    "╚██████╗╚██████╔╝██╔╝ ██╗",
    " ╚═════╝ ╚═════╝ ╚═╝  ╚═╝",
]


def visible_len(text: str) -> int:
    """Length of ``text`` ignoring ANSI escape sequences."""
    return len(_ANSI_RE.sub("", text))


def render_banner(*, enabled: bool = True) -> str:
    """The gradient CGX word-mark."""
    out = []
    for i, row in enumerate(_BANNER_ROWS):
        code = ansi.GRADIENT[min(i, len(ansi.GRADIENT) - 1)]
        out.append(ansi.fg(row, code, enabled=enabled))
    return "\n".join(out)


def render_tips(*, enabled: bool = True) -> str:
    """Getting-started hints shown under the banner."""
    head = ansi.bold("Tips for getting started:", enabled=enabled)
    cmd = lambda s: ansi.paint(s, "cyan", enabled=enabled)
    lines = [
        head,
        f"1. {cmd('/index')} builds the code graph so answers are grounded.",
        f"2. {cmd('/ask')} a question for a fast, read-only grounded answer.",
        f"3. Type a change to make and the agent plans + executes it live.",
        f"4. Press {cmd('Ctrl-C')} to cancel a running task; {cmd('/quit')} exits.",
    ]
    return "\n".join(lines)


def abbreviate_path(path: str) -> str:
    """Collapse ``$HOME`` to ``~`` for a compact status bar."""
    try:
        home = os.path.expanduser("~")
        ap = os.path.abspath(path)
        if home and ap.startswith(home):
            return "~" + ap[len(home):]
        return ap
    except Exception:
        return path


def _three_seg(left: str, mid: str, right: str, width: int) -> str:
    """Justify three coloured segments across ``width`` columns."""
    lv, mv, rv = visible_len(left), visible_len(mid), visible_len(right)
    slack = width - (lv + mv + rv)
    if slack < 2:
        # Too narrow for the middle segment -- drop it and keep the ends.
        slack2 = width - (lv + rv)
        if slack2 < 1:
            return left
        return left + (" " * slack2) + right
    gap_l = slack // 2
    gap_r = slack - gap_l
    return left + (" " * gap_l) + mid + (" " * gap_r) + right


def render_status_bar(
    *, cwd: str, index_state: str, model: str, context_pct: int,
    width: int, enabled: bool = True,
) -> str:
    """Top/bottom status line: cwd | index state | model (ctx%)."""
    left = ansi.paint(abbreviate_path(cwd), "grey", enabled=enabled)
    mid = ansi.paint(index_state, "magenta", enabled=enabled)
    model_txt = model or "no model"
    pct = max(0, min(int(context_pct), 100))
    right = ansi.paint(
        f"{model_txt} ({pct}% context left)", "green", enabled=enabled)
    return _three_seg(left, mid, right, width)


def render_input_box(
    placeholder: str, *, width: int, enabled: bool = True,
) -> str:
    """A rounded, single-line input box (drawn above the live prompt)."""
    inner = max(10, width - 4)
    ph = placeholder[: inner - 2]
    top = "╭" + ("─" * (width - 2)) + "╮"
    body = "│ " + ansi.dim("> " + ph, enabled=enabled)
    body += " " * max(0, inner - visible_len("> " + ph))
    body += " │"
    bot = "╰" + ("─" * (width - 2)) + "╯"
    tint = lambda s: ansi.paint(s, "blue", enabled=enabled)
    return "\n".join([tint(top), tint("│") + body[1:-1] + tint("│"), tint(bot)])


def render_help() -> str:
    """Plain (uncoloured) command reference used by ``/help``."""
    rows = [
        ("/help", "Show this command reference."),
        ("/ask <question>", "Fast read-only grounded answer (streams live)."),
        ("/index [path]", "Build/refresh the code graph for the project."),
        ("/project <path>", "Switch the active project directory."),
        ("/model <name>", "Set the model for the current provider."),
        ("/provider <name>", "Use a saved profile, or ollama|openai|gemini."),
        ("/status", "Show provider, index, and hardware status."),
        ("/serve", "Launch the web UI (FastAPI + React)."),
        ("/clear", "Clear the screen and scrollback."),
        ("/quit, /exit", "Leave the dashboard."),
    ]
    w = max(len(c) for c, _ in rows)
    lines = ["Commands:"]
    lines += [f"  {c.ljust(w)}   {d}" for c, d in rows]
    lines.append("")
    lines.append("Anything else runs the agent (plan → execute), streaming")
    lines.append("progress live. Press Ctrl-C to cancel the running task.")
    return "\n".join(lines)
