

"""GovernedProvider: the quota-enforcing LLMProvider wrapper (Subsystem I).

Wraps any :class:`cgx.answer.providers.LLMProvider` so every ``chat`` /
``chat_stream`` invocation is (1) *pre-checked* against the owner's current-day
budget -- an already-exceeded ceiling raises
:class:`~cgx.governance.budget.BudgetExceeded` before the call runs -- and
(2) *metered* afterwards: tokens (provider-reported or estimated) and the
estimated USD cost are recorded to the :class:`QuotaManager` and mirrored to
metrics. Non-intercepted attribute access is forwarded to the inner provider
via ``__getattr__`` so the wrapper is a drop-in replacement (mirrors
:class:`cgx.session.llm_trace.TracingProvider`).
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

from cgx import usage as _usage
from cgx.governance.budget import BudgetExceeded
from cgx.governance.manager import QuotaManager, get_default_quota_manager, resolve_owner


def _flatten(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    out: List[str] = []
    for m in messages:
        if isinstance(m, dict):
            out.append(str(m.get("content") or ""))
    return "\n\n".join(out)


class GovernedProvider:
    """LLMProvider wrapper that enforces per-owner budgets and meters usage."""

    def __init__(self, inner: Any, *,
                 manager: Optional[QuotaManager] = None) -> None:
        self._inner = inner
        self._manager = manager or get_default_quota_manager()

    @property
    def inner(self) -> Any:
        return self._inner

    @property
    def model(self) -> Optional[str]:
        return getattr(self._inner, "model", None)

    # --- intercepts ---------------------------------------------------
    def chat(self, messages: List[Dict[str, str]],
             temperature: float = 0.2,
             max_tokens: Optional[int] = None,
             *,
             force_json: bool,
             **kwargs: Any) -> Dict[str, Any]:
        owner = resolve_owner()
        self._manager.check(owner)  # hard-stop before spending, may raise
        resp = self._inner.chat(
            messages, temperature=temperature, max_tokens=max_tokens,
            force_json=force_json, **kwargs)
        self._meter(owner, messages, resp)
        return resp

    def chat_stream(self, messages: List[Dict[str, str]],
                    temperature: float = 0.2,
                    max_tokens: Optional[int] = None,
                    *,
                    force_json: bool,
                    **kwargs: Any) -> Iterator[str]:
        owner = resolve_owner()
        self._manager.check(owner)  # hard-stop before spending, may raise
        chunks: List[str] = []
        try:
            for delta in self._inner.chat_stream(
                    messages, temperature=temperature,
                    max_tokens=max_tokens, force_json=force_json, **kwargs):
                chunks.append(str(delta))
                yield delta
        finally:
            self._meter(owner, messages, {"content": "".join(chunks)})

    def __getattr__(self, name: str) -> Any:
        # Only consulted when normal lookup fails, so the explicit overrides
        # above keep priority; everything else (rate limiter, provider knobs,
        # ``stream_json_capable``, etc.) delegates to the inner provider.
        return getattr(self._inner, name)

    # --- internals ----------------------------------------------------
    def _meter(self, owner: str, messages: Any,
               response: Optional[Dict[str, Any]]) -> None:
        """Record one call's tokens + cost. Never breaks the call itself."""
        try:
            prompt_text = _flatten(messages)
            response_text = ""
            if isinstance(response, dict):
                response_text = str(response.get("content") or "")
            usage = _usage.extract_usage(
                response, prompt_text=prompt_text, response_text=response_text)
            cost = _usage.estimate_cost(
                self.model, usage["tokens_in"], usage["tokens_out"])
            self._manager.record_usage(
                owner, tokens_in=usage["tokens_in"],
                tokens_out=usage["tokens_out"], cost_usd=cost["cost_usd"],
                model=self.model, provider=self._provider_name(response))
        except Exception:  # pragma: no cover - metering must never break a call
            pass

    def _provider_name(self, response: Any) -> str:
        if isinstance(response, dict) and response.get("provider"):
            return str(response["provider"])
        cls = type(self._inner).__name__.lower()
        for name in ("ollama", "gemini", "openai"):
            if name in cls:
                return name if name != "openai" else "openai-compat"
        return cls or "unknown"


def govern(provider: Any, *,
           manager: Optional[QuotaManager] = None) -> Any:
    """Wrap ``provider`` in a :class:`GovernedProvider` when governance is on.

    A ``None`` provider, an already-governed provider, or a disabled config
    (``CGX_BUDGET_ENABLED=0``) is returned unchanged so callers can wrap
    unconditionally at the provider choke-point.
    """
    if provider is None or isinstance(provider, GovernedProvider):
        return provider
    mgr = manager or get_default_quota_manager()
    if not mgr.config.enabled:
        return provider
    return GovernedProvider(provider, manager=mgr)


__all__ = ["GovernedProvider", "govern", "BudgetExceeded"]
