"""Prepend SPDX/copyright headers to CGX source files.

Idempotent: re-running the script is a no-op once headers are in place.
Designed to be safe against shebangs, encoding cookies, and module
docstrings (PEP 263 / PEP 257). Run from the repo root::

    python scripts/_add_spdx_headers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HEADER_LINES = (
    "# SPDX-License-Identifier: MIT",
    "# Copyright (c) 2026 Ramin Mohammadi",
)

ROOTS = ("src/cgx", "skills")


def needs_header(text: str) -> bool:
    head = "\n".join(text.splitlines()[:6])
    return "SPDX-License-Identifier" not in head


def insert_header(text: str) -> str:
    lines = text.splitlines(keepends=True)
    insert_at = 0

    # Preserve shebang (must be line 0) and PEP 263 encoding cookie
    # (must appear within the first two lines).
    if lines and lines[0].startswith("#!"):
        insert_at = 1
    if len(lines) > insert_at and lines[insert_at].lstrip().startswith("# -*-"):
        insert_at += 1
    if (
        len(lines) > insert_at
        and lines[insert_at].lstrip().startswith("# coding")
    ):
        insert_at += 1

    header = "".join(line + "\n" for line in HEADER_LINES)
    # Add a blank separator line only when the following content isn't
    # already a blank line — keeps black/ruff-formatted files stable.
    if insert_at < len(lines) and lines[insert_at].strip() != "":
        header += "\n"
    return "".join(lines[:insert_at]) + header + "".join(lines[insert_at:])


def process(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if not needs_header(text):
        return False
    path.write_text(insert_header(text), encoding="utf-8")
    return True


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    changed = 0
    scanned = 0
    for root in ROOTS:
        base = repo / root
        if not base.exists():
            continue
        for py in base.rglob("*.py"):
            scanned += 1
            if process(py):
                changed += 1
                print(f"+ {py.relative_to(repo)}")
    print(f"\n{changed}/{scanned} file(s) updated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
