"""LLM I/O tracing for the ``run_agent`` build loop.

Unlike the session subsystem -- which wraps its provider in
:class:`cgx.session.llm_trace.TracingProvider` to persist every call as a
:class:`~cgx.session.models.Fact` in the SQLite store -- the
Planner→Tracker→Judge build loop has no store. :class:`LLMTraceProvider`
gives that loop equivalent *debuggability* by emitting a bounded,
redacted ``llm_call`` trace record for every ``chat`` / ``chat_stream``
invocation. Those records land in ``<project_root>/.cgx/agent.log`` (or
the fallback trace log) alongside the loop's control-flow events, so a
prompt/response preview sits next to the planner/judge decision it drove.

The wrapper is a drop-in replacement: all non-intercepted attribute
access delegates to the inner provider via ``__getattr__``.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterator, List, Optional

from cgx.trace import emit_llm_call


class LLMTraceProvider:
    """Provider wrapper that emits a redacted ``llm_call`` trace per call."""

    def __init__(self, inner: Any, *, component: str = "agent_loop") -> None:
        self._inner = inner
        self._component = component

    @property
    def inner(self) -> Any:
        return self._inner

    @property
    def model(self) -> Optional[str]:
        return getattr(self._inner, "model", None)

    def chat(self, messages: List[Dict[str, str]],
             temperature: float = 0.2,
             max_tokens: Optional[int] = None,
             force_json: bool = True,
             **kwargs: Any) -> Dict[str, Any]:
        t0 = time.perf_counter()
        sampling = self._sampling(temperature, max_tokens, force_json, kwargs)
        try:
            resp = self._inner.chat(
                messages, temperature=temperature,
                max_tokens=max_tokens, force_json=force_json, **kwargs)
        except Exception as exc:
            emit_llm_call(
                component=self._component, model=self.model,
                messages=messages, error=f"{type(exc).__name__}: {exc}",
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                sampling=sampling)
            raise
        emit_llm_call(
            component=self._component, model=self.model,
            messages=messages, response=resp,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            sampling=sampling)
        return resp

    def chat_stream(self, messages: List[Dict[str, str]],
                    temperature: float = 0.2,
                    max_tokens: Optional[int] = None,
                    **kwargs: Any) -> Iterator[str]:
        t0 = time.perf_counter()
        chunks: List[str] = []
        sampling = self._sampling(temperature, max_tokens, None, kwargs)
        try:
            for delta in self._inner.chat_stream(
                    messages, temperature=temperature,
                    max_tokens=max_tokens, **kwargs):
                chunks.append(str(delta))
                yield delta
        finally:
            emit_llm_call(
                component=self._component, model=self.model,
                messages=messages, response={"content": "".join(chunks)},
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                sampling=sampling, streamed=True)

    def __getattr__(self, name: str) -> Any:
        # Consulted only when normal lookup fails, so the explicit
        # overrides above keep priority; everything else delegates to the
        # inner provider so provider-specific knobs stay reachable.
        return getattr(self._inner, name)

    @staticmethod
    def _sampling(temperature: float, max_tokens: Optional[int],
                  force_json: Optional[bool],
                  extras: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {"temperature": temperature,
                               "max_tokens": max_tokens}
        if force_json is not None:
            out["force_json"] = bool(force_json)
        for k, v in (extras or {}).items():
            out.setdefault(k, v)
        return out
