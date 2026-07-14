

"""ANSI styling helpers for the terminal dashboard.

All colour output is opt-in and self-disabling: when stdout is not a TTY,
``NO_COLOR`` is set, ``CGX_NO_COLOR`` is set, or ``TERM=dumb``, the
helpers return their input unchanged. This keeps piped/redirected output
clean and makes the renderers deterministic under test.
"""

from __future__ import annotations

import os
import shutil
import sys
from typing import List

RESET = "\x1b[0m"

# Braille spinner frames used for the transient "working" indicator.
SPINNER_FRAMES: List[str] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

# 256-colour ramp used for the banner gradient (yellow -> orange -> red).
GRADIENT: List[int] = [226, 220, 214, 208, 202, 166]

# Named foreground colours (256-colour codes) used across the dashboard.
COLORS = {
    "yellow": 220,
    "orange": 208,
    "red": 196,
    "green": 42,
    "cyan": 44,
    "blue": 39,
    "magenta": 170,
    "grey": 245,
    "dim": 240,
    "white": 255,
}


def color_enabled(stream=None) -> bool:
    """Return ``True`` when it is safe to emit ANSI colour codes."""
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("CGX_NO_COLOR") is not None:
        return False
    if os.environ.get("CGX_FORCE_COLOR") is not None:
        return True
    if (os.environ.get("TERM") or "").lower() == "dumb":
        return False
    st = stream if stream is not None else sys.stdout
    try:
        return bool(st.isatty())
    except Exception:
        return False


def fg(text: str, code: int, *, enabled: bool = True) -> str:
    """Wrap ``text`` in a 256-colour foreground escape when ``enabled``."""
    if not enabled or not text:
        return text
    return f"\x1b[38;5;{int(code)}m{text}{RESET}"


def paint(text: str, name: str, *, enabled: bool = True) -> str:
    """Colour ``text`` by a named colour from :data:`COLORS`."""
    return fg(text, COLORS.get(name, COLORS["white"]), enabled=enabled)


def bold(text: str, *, enabled: bool = True) -> str:
    if not enabled or not text:
        return text
    return f"\x1b[1m{text}{RESET}"


def dim(text: str, *, enabled: bool = True) -> str:
    if not enabled or not text:
        return text
    return f"\x1b[2m{text}{RESET}"


def term_width(default: int = 80) -> int:
    """Best-effort terminal column count, clamped to a sane range."""
    try:
        cols = shutil.get_terminal_size((default, 24)).columns
    except Exception:
        cols = default
    return max(40, min(int(cols), 120))


def clear_screen() -> str:
    """Return the escape sequence that clears the screen + homes the cursor."""
    return "\x1b[2J\x1b[H"


def clear_line() -> str:
    """Return the escape sequence that returns to col 0 and erases the line."""
    return "\r\x1b[K"
