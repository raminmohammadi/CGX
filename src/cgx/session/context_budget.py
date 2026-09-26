"""Deterministic num_ctx headroom guard for in-task ReAct loops (JEV #2: CACHE).

CGX assembles context per query rather than accumulating a transcript -- except
inside a couple of bounded ReAct loops (the Swarm Tech Lead planner and
DIAGNOSE) that genuinely append tool observations turn by turn. Left uncapped,
one large tool response overruns the model's ``num_ctx`` and the server
silently truncates the TAIL, dropping the model's own latest reasoning.

This module answers the paper's CACHE question deterministically -- no model
call, no added latency: given the effective context window, does the running
prompt still fit? If not, keep the stable HEAD (system + objective)
byte-identical so the llama.cpp/Ollama prompt cache stays warm, cap oversized
bodies, and elide the OLDEST middle observations while always preserving the
most recent turns the model needs to continue. Probabilities and model-scored
reuse are deliberately NOT used here: a fit check is exact, and a wrong "reuse"
would feed a truncated prompt -- a real bug, not a ranking miss.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

Messages = List[Dict[str, Any]]

# Coarse chars-per-token heuristic. We only need order-of-magnitude fit, and
# over-estimating is safe (it trims slightly earlier); this keeps the guard
# tokenizer-free and provider-agnostic.
_CHARS_PER_TOKEN = 4
_ELIDE = "\n...[trimmed to fit the context window]...\n"
_ELIDE_MSG = "[earlier tool observations elided to fit the context window]"


def estimate_tokens(text: str) -> int:
    """Coarse token estimate for ``text`` (ceil of chars / 4)."""
    n = len(text or "")
    return (n + _CHARS_PER_TOKEN - 1) // _CHARS_PER_TOKEN


def messages_tokens(messages: Messages) -> int:
    """Coarse token estimate for a message list (sum of content estimates)."""
    return sum(estimate_tokens(str(m.get("content", ""))) for m in messages)


def cap_text(text: str, max_chars: int) -> str:
    """Truncate ``text`` to ``max_chars`` with a middle-out elision marker.

    A tool response's head (what ran) and tail (the error / final lines) are
    usually the informative parts, so we keep both ends and elide the middle,
    rather than a blunt head cut that drops the stack-trace tail.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= len(_ELIDE) + 8:
        return text[:max_chars]
    room = max_chars - len(_ELIDE)
    head = (room * 2) // 3
    tail = room - head
    return text[:head] + _ELIDE + (text[-tail:] if tail > 0 else "")


def _head_end(messages: Messages) -> int:
    """Index past the byte-identical stable prefix (system + first objective)."""
    n = len(messages)
    if n == 0:
        return 0
    if messages[0].get("role") == "system":
        return min(2, n)  # system + first user/objective turn
    return min(1, n)


def fit_react_messages(
    messages: Messages,
    *,
    effective_ctx: int,
    reserve_output: int = 1024,
    keep_recent: int = 4,
    per_msg_cap_chars: Optional[int] = None,
) -> Messages:
    """Return a NEW message list trimmed to fit ``effective_ctx - reserve_output``.

    Invariants (see module docstring): the stable head -- ``messages[0]``
    (system) and the first objective turn -- is preserved byte-identical, and
    the last ``keep_recent`` messages are always kept. Oversized non-head bodies
    are capped to ``per_msg_cap_chars`` first; if the estimate still overflows,
    the OLDEST middle turns are dropped and replaced by a single elision marker.
    The input list is not mutated.
    """
    msgs: Messages = [dict(m) for m in messages]
    n = len(msgs)
    if n == 0:
        return msgs
    budget = max(256, int(effective_ctx) - max(0, int(reserve_output)))
    head_end = _head_end(msgs)

    # Step 1: cap oversized bodies outside the protected head.
    if per_msg_cap_chars:
        for i in range(head_end, n):
            body = str(msgs[i].get("content", ""))
            if len(body) > per_msg_cap_chars:
                msgs[i]["content"] = cap_text(body, per_msg_cap_chars)

    if messages_tokens(msgs) <= budget:
        return msgs

    # Step 2: drop the oldest middle turns, always keeping head + recent tail.
    tail_start = max(head_end, n - max(0, keep_recent))
    head = msgs[:head_end]
    middle = msgs[head_end:tail_start]
    tail = msgs[tail_start:]
    dropped = False
    while middle and messages_tokens(head + middle + tail) > budget:
        middle.pop(0)
        dropped = True
    if dropped:
        return head + [{"role": "user", "content": _ELIDE_MSG}] + middle + tail
    return head + middle + tail
