"""Wrapper-tolerant parser for swarm agent action blocks.

Small local models (e.g. ``qwen2.5-coder:7b-instruct``) routinely omit the
``<tool_call>`` envelope, emit the inner ``<name>`` block bare, or wrap the
whole reply in a Markdown ``` fence. The first swarm cut parsed with
``text.find("<tool_call>")``; when the envelope was missing it saw no action
and re-prompted until the loop budget drained. This module parses a reply into
a typed :class:`SwarmAction` that honours a well-formed *intent* even when the
*envelope* is imperfect.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# Field tags a tool call may carry. Extracted verbatim (DOTALL) from the reply.
_FIELD_TAGS = ("query", "command", "path", "content", "find", "replace")

# Content-bearing tags a truncated reply often leaves unclosed; for these we
# fall back to "everything until end of string" so a cut-off file still lands.
_UNCLOSED_OK = ("content", "replace", "find")

_FENCE_RE = re.compile(r"\A\s*```[a-zA-Z0-9_+-]*[ \t]*\n?|\n?```\s*\Z")


@dataclass
class SwarmAction:
    """A single parsed action from a swarm agent reply.

    ``kind`` is one of ``tool_call``/``delegate``/``finish``/``report``/
    ``none``. For a tool call, ``name`` is the tool and ``fields`` its
    arguments; for the terminal blocks, ``text`` is the body. ``error``
    is set when a tool was named but is not allowed for the role.
    """

    kind: str = "none"
    name: Optional[str] = None
    fields: Dict[str, str] = field(default_factory=dict)
    text: str = ""
    error: Optional[str] = None


def strip_fence(s: str) -> str:
    """Remove a single surrounding Markdown code fence, if present."""
    if "```" not in s:
        return s
    return _FENCE_RE.sub("", s).strip()


def _find_tag(tag: str, text: str) -> Optional[str]:
    """Return the (stripped) body of the first ``<tag>...</tag>`` in ``text``.

    Falls back to an unclosed ``<tag>...`` for the content-bearing tags so a
    reply truncated mid-file still yields the partial body.
    """
    m = re.search(rf"<{tag}\s*>(.*?)</{tag}\s*>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    if tag in _UNCLOSED_OK:
        m = re.search(rf"<{tag}\s*>(.*)\Z", text, re.DOTALL)
        if m:
            return m.group(1).strip()
    return None


def _terminal_blocks(kind: str, text: str) -> List[Tuple[int, str]]:
    """All ``(start_offset, body)`` matches for a terminal block ``kind``."""
    out = [(m.start(), m.group(1).strip())
           for m in re.finditer(rf"<{kind}\s*>(.*?)</{kind}\s*>", text,
                                re.DOTALL)]
    if out:
        return out
    m = re.search(rf"<{kind}\s*>(.*)\Z", text, re.DOTALL)
    return [(m.start(), m.group(1).strip())] if m else []


def parse_swarm_action(reply: str, *, allowed_tools: Sequence[str],
                       allow_delegate: bool = False,
                       allow_finish: bool = False,
                       allow_report: bool = False) -> SwarmAction:
    """Parse a swarm reply into one :class:`SwarmAction`, tolerant of envelope.

    Precedence: an allowed tool call wins first (the agent asked for data);
    otherwise, among the enabled terminal blocks, the one appearing *last*
    in the reply wins -- models reason first and conclude last, so a
    ``<finish>`` that follows a stray ``<delegate>`` actually terminates the
    loop instead of re-delegating forever.
    """
    raw = reply or ""

    name = _find_tag("name", raw)
    if name is not None:
        if name in allowed_tools:
            fields: Dict[str, str] = {}
            for tag in _FIELD_TAGS:
                val = _find_tag(tag, raw)
                if val is not None:
                    fields[tag] = strip_fence(val) if tag == "content" else val
            return SwarmAction(kind="tool_call", name=name, fields=fields)
        return SwarmAction(kind="none", name=name,
                           error=f"unknown or disallowed tool: {name!r}")

    candidates: List[Tuple[int, str, str]] = []
    for kind, ok in (("delegate", allow_delegate),
                     ("finish", allow_finish),
                     ("report", allow_report)):
        if not ok:
            continue
        for start, body in _terminal_blocks(kind, raw):
            candidates.append((start, kind, body))
    if candidates:
        _, kind, body = max(candidates, key=lambda c: c[0])
        return SwarmAction(kind=kind, text=body)
    return SwarmAction(kind="none")
