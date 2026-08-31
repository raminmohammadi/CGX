"""Human-in-the-loop approval endpoints.

A running swarm session with an installed approval gate blocks any risky tool
call (arbitrary code execution, file writes, external MCP effects) until a human
decides. This surface lets the web UI list what is waiting and resolve it; the
gate wakes the blocked worker as soon as a decision is posted. No auth --
CGX is local-first single-user (see the ``owner`` cost bucket in governance).
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from cgx.session.approval import ApprovalDecision, all_pending, get_gate

router = APIRouter(tags=["approvals"])


class ResolveBody(BaseModel):
    session_id: str
    request_id: str
    approved: bool
    reason: str = ""


@router.get("/approvals/pending")
def pending() -> dict:
    """Every approval request currently awaiting a decision."""
    return {"pending": all_pending()}


@router.post("/approvals/resolve")
def resolve(body: ResolveBody) -> dict:
    """Approve or deny a pending request, unblocking its worker."""
    gate = get_gate(body.session_id)
    if gate is None:
        return {"ok": False, "reason": "no active gate for session"}
    ok = gate.resolve(body.request_id,
                      ApprovalDecision(approved=body.approved,
                                       reason=body.reason))
    return {"ok": ok}
