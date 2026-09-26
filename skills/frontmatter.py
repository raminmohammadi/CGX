"""Minimal YAML-frontmatter parser for markdown skills.

Markdown skills (``SKILL.md``) open with a small ``---``-fenced block of
key/value metadata, mirroring the Anthropic *Agent Skills* format::

    ---
    name: my-skill
    description: One line about when to use this.
    triggers: [foo, bar]
    surfaces: [chat, plan]
    always_on: false
    ---
    <instruction body ...>

We deliberately do **not** depend on PyYAML: the top-level ``skills``
package is kept import-light so it works in air-gapped installs and never
drags a third-party parser onto the hot detection path. This parser
covers the small, flat subset the skill schema needs -- scalars, inline
lists (``[a, b]``), block lists (``- item``), booleans and numbers -- and
nothing else. Anything it cannot represent is returned as a plain string,
which the :class:`~skills.markdown_skill.MarkdownSkill` adapter validates.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

__all__ = ["parse_frontmatter", "split_frontmatter"]

_FENCE = "---"


def _coerce_scalar(raw: str) -> Any:
    """Turn a bare scalar token into a bool / int / float / str."""
    s = raw.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    low = s.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("null", "none", "~", ""):
        return None
    # Integers / floats -- but never mangle things like "1.2.3" or "3b".
    try:
        if s.lstrip("-").isdigit():
            return int(s)
    except ValueError:
        pass
    try:
        f = float(s)
        # Reject values float() accepts but we want as strings (inf/nan).
        if s.lower() not in ("inf", "-inf", "nan", "+inf"):
            return f
    except ValueError:
        pass
    return s


def _parse_inline_list(raw: str) -> List[Any]:
    """Parse an inline flow list: ``[a, "b, c", 3]``."""
    inner = raw.strip()[1:-1].strip()
    if not inner:
        return []
    items: List[str] = []
    buf: List[str] = []
    quote = ""
    for ch in inner:
        if quote:
            if ch == quote:
                quote = ""
            else:
                buf.append(ch)
            continue
        if ch in ("'", '"'):
            quote = ch
            continue
        if ch == ",":
            items.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    items.append("".join(buf))
    return [_coerce_scalar(x) for x in items if x.strip() != ""]


def split_frontmatter(text: str) -> Tuple[str, str]:
    """Return ``(frontmatter_text, body)``.

    ``frontmatter_text`` is empty when the document does not open with a
    ``---`` fence on its first non-BOM line.
    """
    if not text:
        return "", ""
    # Tolerate a leading BOM and blank lines before the fence.
    stripped = text.lstrip("﻿")
    lines = stripped.splitlines()
    idx = 0
    while idx < len(lines) and lines[idx].strip() == "":
        idx += 1
    if idx >= len(lines) or lines[idx].strip() != _FENCE:
        return "", text
    # Find the closing fence.
    for j in range(idx + 1, len(lines)):
        if lines[j].strip() == _FENCE:
            fm = "\n".join(lines[idx + 1:j])
            body = "\n".join(lines[j + 1:])
            return fm, body
    # Unterminated fence -> treat the whole thing as body (no metadata).
    return "", text


def parse_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    """Parse a markdown document's frontmatter into ``(meta, body)``.

    ``meta`` is ``{}`` when there is no frontmatter block. Parsing never
    raises: malformed lines are skipped so one bad line can't sink a whole
    skill. The caller (:class:`~skills.markdown_skill.MarkdownSkill`) is
    responsible for validating required fields and value types.
    """
    fm, body = split_frontmatter(text)
    meta: Dict[str, Any] = {}
    if not fm:
        return meta, body.strip("\n")

    lines = fm.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        i += 1
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value == "":
            # Possibly a block list on following indented "- item" lines.
            block: List[Any] = []
            while i < n:
                nxt = lines[i]
                if nxt.strip().startswith("- "):
                    block.append(_coerce_scalar(nxt.strip()[2:]))
                    i += 1
                elif nxt.strip() == "":
                    i += 1
                else:
                    break
            meta[key] = block if block else None
        elif value.startswith("[") and value.endswith("]"):
            meta[key] = _parse_inline_list(value)
        else:
            meta[key] = _coerce_scalar(value)
    return meta, body.strip("\n")
