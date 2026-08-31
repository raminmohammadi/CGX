"""A single tool registry + tolerant dispatcher for the swarm agents.

The first swarm cut carried *three* incompatible tool-call designs -- a
hardcoded ``if/elif`` chain in :mod:`swarm_generate`, a second one in
:mod:`swarm_tech_lead`, and a JSON protocol in :mod:`diagnose` -- plus a
fourth, better, but entirely unused tolerant parser (the old ``swarm_parse``).
Adding a tool meant editing the dispatch, the allow-list, *and* the
hand-written tool descriptions in every system prompt, so the agent's view of
its own toolset drifted from what the code could actually run.

This module makes tools *declarative*. A :class:`ToolSpec` bundles a tool's
name, its LLM-facing description (auto-injected into the prompt so the agent
always knows what exists and how to call it), its handler, and a
:class:`RiskLevel` used by the human-in-the-loop approval gate. One
:class:`ToolRegistry` owns them; :func:`parse_tool_calls` extracts **every**
``<call_tool>`` block in a reply (tolerant of quote style and whitespace), and
:meth:`ToolRegistry.dispatch` runs one through the (optional) approval gate.

Adding a new native tool -- or an MCP tool -- is now a single ``register``
call; nothing in the generation loop changes.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """How dangerous a tool is, for the approval gate to key on.

    ``LOW``  -- read-only introspection (query the index, read a file skeleton).
    ``MEDIUM`` -- reaches the network or reads broadly (web search).
    ``HIGH`` -- executes code, writes files, or drives an external MCP effect.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ToolContext:
    """Everything a tool handler may need, passed uniformly at dispatch time."""

    root: str
    deps: Any = None
    log_root: Optional[str] = None
    approval_gate: Any = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolSpec:
    """A declaratively-registered tool.

    ``handler`` takes the parsed ``args`` dict and a :class:`ToolContext` and
    returns a string the loop feeds back to the model. ``arg_hint`` is a short
    JSON example shown in the prompt so the model gets the argument shape right.
    """

    name: str
    description: str
    handler: Callable[[Dict[str, Any], ToolContext], str]
    risk: RiskLevel = RiskLevel.LOW
    arg_hint: str = "{}"


@dataclass
class ToolCall:
    """One parsed ``<call_tool>`` request."""

    name: str
    args: Dict[str, Any]
    raw_args: str


# Tolerant of: name="x" / name='x' / name = x, arbitrary whitespace, and a
# body that is JSON args (``{...}``) or empty. Captures every occurrence.
_CALL_RE = re.compile(
    r"<call_tool\s+name\s*=\s*['\"]?([\w.-]+)['\"]?\s*>(.*?)</call_tool>",
    re.DOTALL | re.IGNORECASE,
)


def parse_tool_calls(text: str) -> List[ToolCall]:
    """Extract every ``<call_tool>`` block in ``text`` as a :class:`ToolCall`.

    Unlike the old ``re.search`` (which matched only the first block and failed
    on single quotes or stray whitespace), this iterates all matches and parses
    the body as JSON when possible, else keeps it as a ``{"input": <raw>}`` bag
    so a non-JSON payload still reaches the handler rather than being dropped.
    """
    calls: List[ToolCall] = []
    for m in _CALL_RE.finditer(text or ""):
        name = m.group(1).strip()
        raw = (m.group(2) or "").strip()
        args: Dict[str, Any] = {}
        if raw:
            try:
                parsed = json.loads(raw)
                args = parsed if isinstance(parsed, dict) else {"input": parsed}
            except Exception:
                args = {"input": raw}
        calls.append(ToolCall(name=name, args=args, raw_args=raw))
    return calls


class ToolRegistry:
    """A named collection of :class:`ToolSpec` with dispatch + prompt help."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        """Add (or replace) a tool by name."""
        self._tools[spec.name] = spec

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools)

    def describe_for_prompt(self,
                            names: Optional[Sequence[str]] = None) -> str:
        """Render the tool list block injected into a role's system prompt.

        Passing ``names`` restricts the block to a subset (a role sees only its
        allowed tools); omitting it describes every registered tool. This is
        what keeps the agent's advertised toolset in lock-step with what
        ``dispatch`` can actually run.
        """
        chosen = [self._tools[n] for n in (names or self.names())
                  if n in self._tools]
        if not chosen:
            return ""
        lines = ["Tools available:"]
        for spec in chosen:
            lines.append(f"- {spec.name}{spec.arg_hint}: {spec.description}")
        lines.append(
            'To call a tool, output EXACTLY: '
            '<call_tool name="tool_name">{"arg": "val"}</call_tool> '
            "and wait for the <tool_response> before continuing.")
        return "\n".join(lines)

    def dispatch(self, call: ToolCall, ctx: ToolContext) -> str:
        """Run one tool call through the approval gate and its handler.

        Returns the handler's string result, an ``Unknown tool`` message for an
        unregistered name, a denial notice when the approval gate rejects a
        risky call, or a ``Tool error`` string on any handler exception -- the
        loop must never crash on a bad tool call.
        """
        spec = self._tools.get(call.name)
        if spec is None:
            return (f"Unknown tool: {call.name!r}. "
                    f"Available: {', '.join(self.names())}")
        # Prefer an explicit gate on the context; otherwise use the
        # context-local gate a front-end installed for this session (if any).
        gate = ctx.approval_gate
        if gate is None:
            try:
                from cgx.session.approval import current_gate
                gate = current_gate()
            except Exception:  # pragma: no cover - approval module optional
                gate = None
        if gate is not None:
            decision = gate.request(spec.name, call.args, spec.risk)
            if not decision.approved:
                return (f"Tool call '{spec.name}' was not approved: "
                        f"{decision.reason or 'denied by policy'}")
        try:
            return str(spec.handler(call.args, ctx))
        except Exception as exc:  # pragma: no cover - handler is best-effort
            logger.exception("tool %s failed", spec.name)
            return f"Tool error: {type(exc).__name__}: {exc}"


# The process-wide default registry. Native tools register at import of
# :mod:`swarm_tools`; MCP tools register when the MCP layer is configured.
REGISTRY = ToolRegistry()
