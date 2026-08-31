"""Human-in-the-loop approval gate for risky agent tool calls.

CGX's swarm tools include arbitrary code execution (``run_python_probe``), file
writes, and -- once MCP is configured -- calls that reach the outside world,
guarded today only by a wall-clock timeout. This module adds an opt-in gate: a
risky :class:`~cgx.session.tasks.tool_registry.ToolSpec` is intercepted at
dispatch and must be approved (by a human) before it runs.

Design notes:

* The gate is **opt-in**. When no gate is active (the default), dispatch runs
  tools exactly as before -- nothing here changes existing behavior until a
  front-end installs a gate via :func:`use_gate`.
* Mode is env-driven (``CGX_APPROVAL_MODE`` = ``off`` | ``risky`` | ``all``,
  default ``risky``): ``risky`` gates MEDIUM/HIGH tools, ``all`` gates every
  tool, ``off`` disables gating even when a gate is installed.
* A request blocks the calling worker on a :class:`threading.Event` until a
  decision arrives or the TTL elapses (auto-reject) -- the fail-safe default is
  to *deny*, never to silently run an unapproved risky call.
* A ``responder`` (a blocking callable, e.g. a terminal prompt) resolves inline;
  otherwise a front-end resolves out-of-band via :meth:`ApprovalGate.resolve`
  after reading :meth:`ApprovalGate.pending` (the web/SSE path).
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from cgx.session.tasks.tool_registry import RiskLevel


class ApprovalMode(str, Enum):
    OFF = "off"
    RISKY = "risky"
    ALL = "all"


def mode_from_env() -> ApprovalMode:
    """Read ``CGX_APPROVAL_MODE`` (default ``risky``), tolerant of junk values."""
    raw = (os.environ.get("CGX_APPROVAL_MODE") or "risky").strip().lower()
    try:
        return ApprovalMode(raw)
    except ValueError:
        return ApprovalMode.RISKY


@dataclass
class ApprovalDecision:
    """The outcome of an approval request."""

    approved: bool
    reason: str = ""


@dataclass
class ApprovalRequest:
    """A pending request awaiting a human decision."""

    request_id: str
    tool: str
    args: Dict[str, Any]
    risk: RiskLevel
    created_at: float
    event: threading.Event = field(default_factory=threading.Event)
    decision: Optional[ApprovalDecision] = None

    def as_dict(self) -> Dict[str, Any]:
        return {"request_id": self.request_id, "tool": self.tool,
                "args": self.args, "risk": self.risk.value,
                "created_at": self.created_at}


# Default auto-reject window (seconds): a request nobody answers is denied.
_DEFAULT_TTL = 1800


class ApprovalGate:
    """Decides whether a tool call may run, asking a human for risky ones."""

    def __init__(self, mode: Optional[ApprovalMode] = None,
                 ttl_seconds: int = _DEFAULT_TTL,
                 responder: Optional[Callable[[ApprovalRequest],
                                              ApprovalDecision]] = None,
                 on_request: Optional[Callable[[ApprovalRequest], None]] = None,
                 session_id: Optional[str] = None) -> None:
        self.mode = mode or mode_from_env()
        self.ttl = ttl_seconds
        self.responder = responder
        self.on_request = on_request  # notify hook (e.g. emit an SSE event)
        self.session_id = session_id
        self._pending: Dict[str, ApprovalRequest] = {}
        self._lock = threading.Lock()

    def _needs_approval(self, risk: RiskLevel) -> bool:
        if self.mode is ApprovalMode.OFF:
            return False
        if self.mode is ApprovalMode.ALL:
            return True
        return risk in (RiskLevel.MEDIUM, RiskLevel.HIGH)  # RISKY

    def request(self, tool: str, args: Dict[str, Any],
                risk: RiskLevel) -> ApprovalDecision:
        """Block until ``tool`` is approved/denied; auto-reject after the TTL."""
        if not self._needs_approval(risk):
            return ApprovalDecision(True, "auto-approved")
        req = ApprovalRequest(request_id="apr_" + uuid.uuid4().hex[:12],
                              tool=tool, args=args, risk=risk,
                              created_at=time.time())
        with self._lock:
            self._pending[req.request_id] = req
        if self.on_request is not None:
            try:
                self.on_request(req)
            except Exception:  # pragma: no cover - notify is best-effort
                pass
        # Inline responder (e.g. terminal prompt) resolves immediately.
        if self.responder is not None:
            try:
                decision = self.responder(req)
            except Exception as exc:  # pragma: no cover
                decision = ApprovalDecision(False, f"responder error: {exc}")
            self.resolve(req.request_id, decision)
            return decision
        # Out-of-band resolution (web/SSE): wait for resolve() or TTL.
        got = req.event.wait(self.ttl)
        with self._lock:
            self._pending.pop(req.request_id, None)
        if not got or req.decision is None:
            return ApprovalDecision(False, "auto-rejected: no decision within "
                                    f"{self.ttl}s")
        return req.decision

    def resolve(self, request_id: str, decision: ApprovalDecision) -> bool:
        """Record a decision for a pending request; wakes its waiter."""
        with self._lock:
            req = self._pending.get(request_id)
        if req is None:
            return False
        req.decision = decision
        req.event.set()
        return True

    def pending(self) -> List[Dict[str, Any]]:
        """Snapshot of currently pending requests (for a UI to render)."""
        with self._lock:
            return [r.as_dict() for r in self._pending.values()]


# --------------------- context-local active gate ---------------------
# The gate is installed per session without threading it through every executor
# and generation-ladder signature: dispatch falls back to the context-local
# gate when its ToolContext carries none. ``use_gate`` scopes activation.

_gate_var: ContextVar[Optional[ApprovalGate]] = ContextVar(
    "cgx_approval_gate", default=None)

# Process-global fallback: the CLI drives generation in a worker thread, where a
# ContextVar set on the main thread is not visible, so a single-user CLI run
# installs the gate here instead. The context-local gate always wins when set.
_default_gate: Optional[ApprovalGate] = None


def set_default_gate(gate: Optional[ApprovalGate]) -> None:
    """Install a process-global fallback gate (used by the CLI)."""
    global _default_gate
    _default_gate = gate


def current_gate() -> Optional[ApprovalGate]:
    """The gate active in this context, the process default, or ``None``."""
    return _gate_var.get() or _default_gate


def set_gate(gate: Optional[ApprovalGate]):
    """Install ``gate`` as the context-local active gate; returns the token."""
    return _gate_var.set(gate)


def reset_gate(token) -> None:
    """Reset the context-local gate to what it was before ``set_gate``."""
    try:
        _gate_var.reset(token)
    except Exception:  # pragma: no cover - token from another context
        pass


class use_gate:
    """Context manager to activate a gate for the enclosed block."""

    def __init__(self, gate: Optional[ApprovalGate]) -> None:
        self._gate = gate
        self._token = None

    def __enter__(self) -> Optional[ApprovalGate]:
        self._token = _gate_var.set(self._gate)
        return self._gate

    def __exit__(self, *exc: Any) -> None:
        if self._token is not None:
            _gate_var.reset(self._token)


# --------------------- CLI responder ---------------------

def terminal_responder(req: ApprovalRequest) -> ApprovalDecision:
    """Block on a terminal y/n prompt (the ``cgx agent`` approval front-end)."""
    import sys
    prompt = (f"\n[approval] {req.risk.value.upper()} tool '{req.tool}' "
              f"args={req.args}\n  Approve? [y/N] ")
    try:
        sys.stderr.write(prompt)
        sys.stderr.flush()
        answer = sys.stdin.readline().strip().lower()
    except Exception as exc:  # pragma: no cover - no tty
        return ApprovalDecision(False, f"no input: {exc}")
    if answer in ("y", "yes"):
        return ApprovalDecision(True, "approved at terminal")
    return ApprovalDecision(False, "denied at terminal")


# --------------------- session -> gate registry (web path) ---------------------
# The web approvals route resolves decisions out-of-band, so it needs to find
# the gate a running session installed. Keyed by session id; best-effort.

_ACTIVE_GATES: Dict[str, ApprovalGate] = {}
_ACTIVE_LOCK = threading.Lock()


def register_gate(session_id: str, gate: ApprovalGate) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE_GATES[session_id] = gate


def unregister_gate(session_id: str) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE_GATES.pop(session_id, None)


def get_gate(session_id: str) -> Optional[ApprovalGate]:
    with _ACTIVE_LOCK:
        return _ACTIVE_GATES.get(session_id)


def all_pending() -> List[Dict[str, Any]]:
    """Every pending request across active sessions (tagged with session_id)."""
    out: List[Dict[str, Any]] = []
    with _ACTIVE_LOCK:
        gates = list(_ACTIVE_GATES.items())
    for sid, gate in gates:
        for req in gate.pending():
            out.append({**req, "session_id": sid})
    return out
