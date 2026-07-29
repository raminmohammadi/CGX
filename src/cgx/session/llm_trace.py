


"""LLM call tracing as :class:`Fact` records.

Wraps an :class:`cgx.answer.providers.LLMProvider` so every ``chat``
invocation produces a typed :class:`Fact` of kind
``FactKind.LLM_CALL`` attributed to the currently-executing task. The
runner binds ``(session_id, task_id)`` before each dispatch via
:meth:`TracingProvider.bind` and drains the accumulated facts via
:meth:`TracingProvider.drain` after the executor returns, folding them
into :attr:`ExecutorResult.facts` so they persist alongside any
executor-emitted facts. Untraced providers (anything without ``bind``
+ ``drain``) are passed through unchanged.

The recorded payload contains the flattened prompt, the response
content, the model id, the per-call latency, and the sampling
parameters. Long prompts / responses are truncated symmetrically to
keep store rows bounded; the full byte counts remain visible via
``prompt_chars`` / ``response_chars``.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

from cgx.redact import redact_text
from cgx.session.models import Fact, FactKind
from cgx.trace import emit_llm_call


_PROMPT_CHAR_CAP = 8_000
_RESPONSE_CHAR_CAP = 8_000


def _truncate(text: str, cap: int) -> str:
    if not isinstance(text, str):
        return ""
    if len(text) <= cap:
        return text
    head = cap // 2
    tail = cap - head - 32
    return f"{text[:head]}\n…[{len(text) - cap} chars truncated]…\n{text[-tail:]}"


def _flatten_messages(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    out: List[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "user")
        content = str(m.get("content") or "")
        out.append(f"[{role}]\n{content}")
    return "\n\n".join(out)


class TracingProvider:
    """LLMProvider wrapper that records every chat call as a Fact.

    Forwards all non-intercepted attribute access to the wrapped
    provider via ``__getattr__`` so the wrapper is a drop-in
    replacement: model_caps probes, prompt builders that look at
    ``provider.model``, and any provider-specific tuning hooks keep
    working without modification.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._facts: List[Fact] = []
        self._binding: Optional[Tuple[str, str]] = None

    # --- public binding api used by the runner ------------------------
    @property
    def inner(self) -> Any:
        return self._inner

    @property
    def model(self) -> Optional[str]:
        return getattr(self._inner, "model", None)

    def bind(self, session_id: str, task_id: str) -> None:
        self._binding = (session_id, task_id)

    def unbind(self) -> None:
        self._binding = None

    def drain(self) -> List[Fact]:
        out, self._facts = self._facts, []
        return out

    # --- intercepts ---------------------------------------------------
    def chat(self, messages: List[Dict[str, str]],
             temperature: float = 0.2,
             max_tokens: Optional[int] = None,
             force_json: bool = True,
             **kwargs: Any) -> Dict[str, Any]:
        t0 = time.perf_counter()
        try:
            resp = self._inner.chat(
                messages, temperature=temperature,
                max_tokens=max_tokens, force_json=force_json, **kwargs)
        except Exception as exc:
            self._record(
                messages, error=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                temperature=temperature, max_tokens=max_tokens,
                force_json=force_json, extras=kwargs)
            raise
        self._record(messages, response=resp,
                     latency_ms=(time.perf_counter() - t0) * 1000.0,
                     temperature=temperature, max_tokens=max_tokens,
                     force_json=force_json, extras=kwargs)
        return resp

    def chat_stream(self, messages: List[Dict[str, str]],
                    temperature: float = 0.2,
                    max_tokens: Optional[int] = None,
                    **kwargs: Any) -> Iterator[str]:
        t0 = time.perf_counter()
        chunks: List[str] = []
        try:
            for delta in self._inner.chat_stream(
                    messages, temperature=temperature,
                    max_tokens=max_tokens, **kwargs):
                chunks.append(str(delta))
                yield delta
        finally:
            self._record(
                messages, response={"content": "".join(chunks)},
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                temperature=temperature, max_tokens=max_tokens,
                force_json=False, extras=dict(kwargs), streamed=True)

    def __getattr__(self, name: str) -> Any:
        # __getattr__ is only consulted when normal lookup fails, so the
        # explicit overrides above (``chat`` / ``chat_stream`` / ``model``
        # / etc.) keep priority. Everything else delegates to the inner
        # provider so provider-specific knobs (rate limiter, extra
        # options) stay reachable.
        return getattr(self._inner, name)

    # --- internals ----------------------------------------------------
    def _record(self, messages: Any, *,
                response: Optional[Dict[str, Any]] = None,
                error: Optional[str] = None,
                latency_ms: float = 0.0,
                temperature: float = 0.2,
                max_tokens: Optional[int] = None,
                force_json: bool = True,
                extras: Optional[Dict[str, Any]] = None,
                streamed: bool = False) -> None:
        if self._binding is None:
            return
        session_id, task_id = self._binding
        # Redact before truncation so a secret split across the truncation
        # boundary can't survive in either half of the stored row.
        prompt_text = _truncate(
            redact_text(_flatten_messages(messages)), _PROMPT_CHAR_CAP)
        response_text = ""
        if isinstance(response, dict):
            response_text = _truncate(
                redact_text(str(response.get("content") or "")),
                _RESPONSE_CHAR_CAP)
        sampling: Dict[str, Any] = {
            "temperature": float(temperature),
            "max_tokens": max_tokens,
            "force_json": bool(force_json),
        }
        if extras:
            for k, v in extras.items():
                if isinstance(v, (str, int, float, bool, type(None))):
                    sampling.setdefault(k, v)
        content: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt_text,
            "response": response_text,
            "prompt_chars": len(prompt_text),
            "response_chars": len(response_text),
            "latency_ms": round(latency_ms, 2),
            "sampling": sampling,
            "streamed": bool(streamed),
        }
        if error:
            content["error"] = redact_text(error)
        fact = Fact.new(
            session_id=session_id,
            kind=FactKind.LLM_CALL,
            content=content,
            surfaced_in_task_id=task_id,
        )
        self._facts.append(fact)
        # Also drop a bounded preview into the trace timeline, correlated
        # to the full stored payload via ``fact_id`` (+ the task_id already
        # on the trace context), so a debugger sees the call inline.
        emit_llm_call(
            component="session", model=self.model, messages=messages,
            response=response, error=error, latency_ms=latency_ms,
            sampling=sampling, streamed=streamed, fact_id=fact.fact_id)
