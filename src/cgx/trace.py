"""Curated function-call tracing for the agent loop (Option A).

A single global toggle controls whether ``@traced`` decorators emit
``enter`` / ``exit`` records. When off, the decorator's hot path is a
single ``bool`` check so production overhead stays negligible.

Trace records are routed to the project-local ``agent.log`` via
:mod:`cgx.session.agent_log` whenever a session has set the
``trace_context`` ContextVar. Outside any session (HTTP routes,
batch CLI, retrieval/codegen called directly), records fall through
to ``~/.cgx/cgx-trace.log`` written by a dedicated rotating logger.

The toggle has three layers, in order of precedence:

1. ``$CGX_TRACE`` env var (``1``/``true``/``yes`` pins ON, ``0``/``false``
   pins OFF). When set, ``set_trace_enabled`` becomes a no-op.
2. Runtime flag (default OFF) flipped via the settings endpoint.
3. The decorator otherwise short-circuits before any work.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import inspect
import json
import logging
import os
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar

# ``cgx.session.agent_log`` is imported lazily inside :func:`_emit` to
# avoid a circular import: many modules in ``cgx.session`` apply
# ``@traced`` at module load time and would re-enter this module before
# the session package finishes initialising.

TRACE = 5
logging.addLevelName(TRACE, "TRACE")

F = TypeVar("F", bound=Callable[..., Any])

_ENV_VAR = "CGX_TRACE"
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}

# Runtime toggle. ``None`` = unset, fall back to env / default OFF.
_runtime_enabled: Optional[bool] = None


def _env_pin() -> Optional[bool]:
    raw = os.environ.get(_ENV_VAR, "").strip().lower()
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    return None


def is_trace_enabled() -> bool:
    """Cheap hot-path check used by every ``@traced`` call."""
    pin = _env_pin()
    if pin is not None:
        return pin
    return bool(_runtime_enabled)


def set_trace_enabled(enabled: bool) -> bool:
    """Flip the runtime flag. Ignored (returns current state) when env-pinned."""
    global _runtime_enabled
    if _env_pin() is not None:
        return is_trace_enabled()
    _runtime_enabled = bool(enabled)
    return _runtime_enabled


def trace_source() -> str:
    """``'env'`` when ``$CGX_TRACE`` pins the flag, else ``'runtime'``."""
    return "env" if _env_pin() is not None else "runtime"


# Per-task context propagated via contextvars so nested @traced calls
# (including async ones) inherit the active session/task/project root.
trace_context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "cgx_trace_context", default={})


def set_trace_context(
    *,
    session_id: Optional[str] = None,
    task_id: Optional[str] = None,
    project_root: Optional[str] = None,
) -> contextvars.Token:
    """Push a new trace context frame. Returns a token for ``reset_trace_context``."""
    cur = dict(trace_context.get() or {})
    if session_id is not None:
        cur["session_id"] = session_id
    if task_id is not None:
        cur["task_id"] = task_id
    if project_root is not None:
        cur["project_root"] = project_root
    return trace_context.set(cur)


def reset_trace_context(token: contextvars.Token) -> None:
    trace_context.reset(token)


# ----- Fallback global trace log (no project_root in context) ----------------

_FALLBACK_DIR = Path.home() / ".cgx"
_FALLBACK_FILE = "cgx-trace.log"
_FALLBACK_MAX_BYTES = 2 * 1024 * 1024
_FALLBACK_BACKUPS = 3
_fallback_logger: Optional[logging.Logger] = None


class _JsonlFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = getattr(record, "trace_event", None) or {"event": record.msg}
        try:
            return json.dumps(payload, default=str, ensure_ascii=False)
        except Exception:  # pragma: no cover - defensive
            return json.dumps({"event": "trace_format_error"})


def _get_fallback_logger() -> Optional[logging.Logger]:
    global _fallback_logger
    if _fallback_logger is not None:
        return _fallback_logger
    try:
        _FALLBACK_DIR.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            str(_FALLBACK_DIR / _FALLBACK_FILE),
            maxBytes=_FALLBACK_MAX_BYTES,
            backupCount=_FALLBACK_BACKUPS,
            encoding="utf-8",
        )
        handler.setFormatter(_JsonlFormatter())
        handler.setLevel(TRACE)
        lg = logging.getLogger("cgx.trace.fallback")
        lg.setLevel(TRACE)
        # Avoid duplicate handlers across module reloads / pytest workers.
        if not any(isinstance(h, RotatingFileHandler) for h in lg.handlers):
            lg.addHandler(handler)
        lg.propagate = False
        _fallback_logger = lg
        return lg
    except Exception:  # pragma: no cover - defensive
        return None


def _emit(event: str, **fields: Any) -> None:
    """Route one trace record to the project agent log or the fallback file."""
    ctx = trace_context.get() or {}
    payload: Dict[str, Any] = {
        "ts": time.time(),
        "event": event,
        "kind": "trace",
    }
    if "session_id" in ctx:
        payload["session_id"] = ctx["session_id"]
    if "task_id" in ctx:
        payload["task_id"] = ctx["task_id"]
    payload.update(fields)
    project_root = ctx.get("project_root")
    if project_root:
        try:
            from cgx.session.agent_log import log_event as _agent_log_event
            _agent_log_event(project_root, event, **{k: v for k, v in payload.items()
                                                    if k not in ("ts", "event")})
            return
        except Exception:
            pass
    lg = _get_fallback_logger()
    if lg is None:
        return
    rec = logging.LogRecord(
        name="cgx.trace", level=TRACE, pathname=__file__, lineno=0,
        msg=event, args=None, exc_info=None)
    rec.trace_event = payload
    try:
        lg.handle(rec)
    except Exception:  # pragma: no cover - defensive
        pass


def _arg_summary(func: Callable[..., Any], args: tuple, kwargs: dict) -> Dict[str, Any]:
    """Best-effort, bounded summary of call args for trace records."""
    try:
        sig = inspect.signature(func)
        bound = sig.bind_partial(*args, **kwargs)
        out: Dict[str, Any] = {}
        for name, val in bound.arguments.items():
            if name in ("self", "cls"):
                continue
            try:
                s = repr(val)
            except Exception:
                s = f"<{type(val).__name__}>"
            if len(s) > 200:
                s = s[:197] + "..."
            out[name] = s
        return out
    except Exception:
        return {}


def traced(category: str, *, args: bool = False) -> Callable[[F], F]:
    """Decorator: emit ``enter`` / ``exit`` (or ``error``) trace records.

    The decorator is a no-op when tracing is disabled -- the wrapper still
    runs but skips all formatting / emission work. ``category`` groups
    related entry points (e.g. ``"router"``, ``"executor"``, ``"llm"``).

    Set ``args=True`` to include a bounded ``repr()`` summary of the
    function arguments in the enter record. Default OFF because most
    high-volume entry points have large pydantic payloads.
    """
    def decorate(func: F) -> F:
        qual = f"{func.__module__}.{func.__qualname__}"
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def aw(*a: Any, **kw: Any) -> Any:
                if not is_trace_enabled():
                    return await func(*a, **kw)
                t0 = time.perf_counter()
                enter: Dict[str, Any] = {"category": category, "fn": qual}
                if args:
                    enter["args"] = _arg_summary(func, a, kw)
                _emit("trace_enter", **enter)
                try:
                    result = await func(*a, **kw)
                except BaseException as exc:
                    _emit("trace_error", category=category, fn=qual,
                          elapsed_ms=int((time.perf_counter() - t0) * 1000),
                          error_type=type(exc).__name__, error=str(exc)[:300])
                    raise
                _emit("trace_exit", category=category, fn=qual,
                      elapsed_ms=int((time.perf_counter() - t0) * 1000))
                return result
            return aw  # type: ignore[return-value]

        @functools.wraps(func)
        def sw(*a: Any, **kw: Any) -> Any:
            if not is_trace_enabled():
                return func(*a, **kw)
            t0 = time.perf_counter()
            enter: Dict[str, Any] = {"category": category, "fn": qual}
            if args:
                enter["args"] = _arg_summary(func, a, kw)
            _emit("trace_enter", **enter)
            try:
                result = func(*a, **kw)
            except BaseException as exc:
                _emit("trace_error", category=category, fn=qual,
                      elapsed_ms=int((time.perf_counter() - t0) * 1000),
                      error_type=type(exc).__name__, error=str(exc)[:300])
                raise
            _emit("trace_exit", category=category, fn=qual,
                  elapsed_ms=int((time.perf_counter() - t0) * 1000))
            return result
        return sw  # type: ignore[return-value]
    return decorate


def emit_trace(event: str, **fields: Any) -> None:
    """Public alias of :func:`_emit` for callers outside the decorator.

    Skips emission when the global trace flag is off. Used by HTTP
    middleware and other non-function entry points that still want to
    appear in the curated trace stream.
    """
    if not is_trace_enabled():
        return
    _emit(event, **fields)


_LLM_PREVIEW_CAP = 240


def emit_llm_call(
    *,
    component: str,
    model: Optional[str] = None,
    messages: Any = None,
    response: Any = None,
    error: Optional[str] = None,
    latency_ms: float = 0.0,
    sampling: Optional[Dict[str, Any]] = None,
    streamed: bool = False,
    fact_id: Optional[str] = None,
) -> None:
    """Emit a bounded, redacted ``llm_call`` trace record.

    Used by the session-store wrapper
    (:class:`cgx.session.llm_trace.TracingProvider`). Only prompt/response
    *previews* and byte counts land in the trace; the full payload (when
    persisted) lives in the session store, correlated via ``fact_id`` +
    the ``task_id`` already carried on the trace context.
    """
    if not is_trace_enabled():
        return
    from cgx.redact import flatten_messages, preview_text

    prompt = flatten_messages(messages)
    if isinstance(response, dict):
        resp_text = str(response.get("content") or "")
    elif isinstance(response, str):
        resp_text = response
    else:
        resp_text = ""
    fields: Dict[str, Any] = {
        "component": component,
        "model": model,
        "prompt_chars": len(prompt),
        "response_chars": len(resp_text),
        "prompt_preview": preview_text(prompt, _LLM_PREVIEW_CAP),
        "response_preview": preview_text(resp_text, _LLM_PREVIEW_CAP),
        "latency_ms": round(float(latency_ms), 2),
        "streamed": bool(streamed),
    }
    if sampling:
        fields["sampling"] = {
            k: v for k, v in sampling.items()
            if isinstance(v, (str, int, float, bool, type(None)))
        }
    if error:
        fields["error"] = preview_text(str(error), 300)
    if fact_id:
        fields["fact_id"] = fact_id
    _emit("llm_call", **fields)


def reset_for_tests() -> None:
    """Reset runtime flag + fallback logger. Test-only."""
    global _runtime_enabled, _fallback_logger
    _runtime_enabled = None
    if _fallback_logger is not None:
        for h in list(_fallback_logger.handlers):
            try:
                h.close()
            except Exception:
                pass
            _fallback_logger.removeHandler(h)
    _fallback_logger = None
