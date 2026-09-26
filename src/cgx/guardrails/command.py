"""Programmable, content-inspecting permission policy for agent actions (JEV #5).

Answers the paper's PERMISSIONS question -- "should this action run?" -- over
the action NAME *plus its inspected argument / script CONTENT*, not the name
alone, returning a typed ``allow`` / ``ask`` / ``deny`` verdict with a
confidence and a human-readable reason. It closes the gap where CGX's approval
gate keyed only on a tool's static RiskLevel: a ``run_python_probe`` that opens
a socket, reads ``~/.ssh``, or embeds a secret looked exactly like a benign
probe.

Deterministic -- regex / substring + the shared sensitivity scorer + the
existing SSRF and secret-literal guardrails; no model call, ~0 latency on the
hot dispatch path. Enforcement is env-gated and FAIL-OPEN:

* ``CGX_POLICY_MODE`` = ``off`` | ``advisory`` (default) | ``enforce``.
* ``advisory`` enforces only HARD security denies (SSRF, sensitive-path,
  restricted-file, secret literal -- confidence >= 0.9); softer signals are
  logged, not blocked.
* ``enforce`` additionally blocks soft denies (network egress outside a deploy
  task) and lets a clear ``allow`` skip the human gate to cut prompts.
* An unrecognised action is ``ask`` (defers to the human gate), never a silent
  allow. The SSRF hard-block is un-downgradable; only a not-allowlisted but
  public host is left to the fetch guard / human gate.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any, List, Optional
from urllib.parse import urlparse

from cgx.answer.sensitivity import Sensitivity, score_paths
from cgx.guardrails.net import find_urls, host_is_blocked
from cgx.guardrails.output import scan_secret_literals

logger = logging.getLogger(__name__)

# Confidence at/above which a deny is a HARD security block (enforced even in
# advisory mode). Softer denies (egress heuristics) sit below it.
HARD_DENY = 0.9

# Network-egress tokens in exec'd code (a plain URL argument is handled by the
# SSRF/allowlist guard, not here, so fetch_url is not mis-flagged).
_EGRESS = re.compile(
    r"\b(socket\.|urllib|urlopen|requests\.(get|post|put|delete|patch|request)"
    r"|http\.client|httpx\.|aiohttp|smtplib|ftplib|paramiko"
    r"|subprocess.*\b(curl|wget|nc|ncat)\b|/dev/tcp/)", re.I)

# Sensitive filesystem targets referenced anywhere in the args/content.
_SENSITIVE_FS = re.compile(
    r"(~|\$HOME)?/?\.ssh\b|/?\.aws\b|/?\.netrc\b|/?\.git-credentials\b"
    r"|\.env(\.[\w.-]+)?\b|id_rsa\b|\.pem\b|\.p12\b", re.I)


@dataclass
class ActionPolicy:
    """Typed permission decision: verdict + confidence + reason (JEV #5)."""

    verdict: str            # "allow" | "ask" | "deny"
    confidence: float
    reason: str

    @property
    def is_hard_deny(self) -> bool:
        return self.verdict == "deny" and self.confidence >= HARD_DENY


def policy_mode() -> str:
    """``off`` | ``advisory`` (default) | ``enforce`` from ``CGX_POLICY_MODE``."""
    m = (os.environ.get("CGX_POLICY_MODE") or "advisory").strip().lower()
    return m if m in ("off", "advisory", "enforce") else "advisory"


def _strings(value: Any) -> List[str]:
    """Flatten every string in a (possibly nested) argument value."""
    out: List[str] = []

    def _walk(v: Any) -> None:
        if isinstance(v, str):
            out.append(v)
        elif isinstance(v, dict):
            for x in v.values():
                _walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                _walk(x)

    _walk(value)
    return out


def _looks_like_path(s: str) -> bool:
    """Heuristic: a short string that looks like a filesystem path, not prose."""
    s = s.strip()
    if not s or len(s) > 400 or "\n" in s:
        return False
    if s[:7].lower() == "http://" or s[:8].lower() == "https://":
        return False
    return ("/" in s or "\\" in s) or bool(re.search(r"\.\w{1,6}$", s))


def evaluate_action(
    action: str,
    args: Any,
    *,
    risk: Any = None,
    task_scope: Optional[str] = None,
) -> ActionPolicy:
    """Return an :class:`ActionPolicy` for running ``action`` with ``args``."""
    contents = _strings(args)
    blob = "\n".join(contents)

    # 1) SSRF hard-block on any URL in the args/content -- un-downgradable.
    for url in find_urls(args):
        host = urlparse(url).hostname or ""
        if host_is_blocked(host):
            return ActionPolicy("deny", 0.99,
                                f"SSRF-blocked host in URL {url!r}")

    # 2) Sensitive filesystem access (~/.ssh, .env, keys) anywhere in args.
    if _SENSITIVE_FS.search(blob):
        return ActionPolicy("deny", 0.95,
                            "action references a sensitive path (.ssh/.env/key)")

    # 3) File-path arguments the shared scorer rates RESTRICTED.
    paths = [c for c in contents if _looks_like_path(c)]
    if paths and score_paths(paths) is Sensitivity.RESTRICTED:
        return ActionPolicy("deny", 0.9,
                            "action targets restricted files (secrets/env/infra)")

    # 4) A secret-shaped literal about to be written/executed.
    for c in contents:
        if scan_secret_literals(c):
            return ActionPolicy("deny", 0.9,
                                "content contains a secret-shaped literal")

    # 5) Network egress inside exec'd code, outside a deploy-scoped task (soft).
    if _EGRESS.search(blob) and (task_scope or "").strip().lower() != "deploy":
        return ActionPolicy("deny", 0.7,
                            "network egress in executed code outside a deploy task")

    # 6) Otherwise defer: a read-only (LOW) action is a clear allow; anything
    #    else is 'ask' so the human gate stays authoritative.
    risk_val = getattr(risk, "value", None) or (str(risk) if risk else "")
    if risk_val == "low":
        return ActionPolicy("allow", 0.6, "read-only action")
    return ActionPolicy("ask", 0.5, "no policy rule matched; defer to approval gate")


def gate_action(
    action: str,
    args: Any,
    *,
    risk: Any = None,
    task_scope: Optional[str] = None,
) -> Optional[str]:
    """Return a refusal reason if policy blocks ``action``, else ``None``.

    Applies :func:`policy_mode`: ``off`` never blocks; ``advisory`` blocks only
    HARD security denies; ``enforce`` also blocks soft denies. The caller keeps
    its existing human approval gate for everything this lets through.
    """
    mode = policy_mode()
    if mode == "off":
        return None
    pol = evaluate_action(action, args, risk=risk, task_scope=task_scope)
    block = pol.is_hard_deny or (mode == "enforce" and pol.verdict == "deny")
    if block:
        logger.warning("policy denied %s: %s", action, pol.reason)
        return pol.reason
    if pol.verdict == "deny":  # soft deny in advisory mode: log only
        logger.info("policy advisory (not enforced) for %s: %s", action, pol.reason)
    return None


__all__ = [
    "ActionPolicy", "evaluate_action", "gate_action", "policy_mode", "HARD_DENY",
]
