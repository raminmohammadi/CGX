# src/cgx/answer/providers.py
from __future__ import annotations
import os
import json
from typing import Any, Dict, Iterator, List, Optional
import requests  # intentionally re-exported; UI imports this to ensure dependency

from cgx.answer.ratelimit import RateLimiter, request_with_retry

DEFAULT_TIMEOUT = float(os.environ.get("CGX_HTTP_TIMEOUT", "120"))


def _build_limiter(rate_limit: Optional[float]) -> Optional[RateLimiter]:
    """Return a RateLimiter when ``rate_limit`` is a positive number, else None.

    ``None`` / ``0`` / negative values produce a no-op (no limiter) so that
    existing callers see zero behavioral change unless they opt in.
    """
    if rate_limit is None:
        return None
    try:
        rate = float(rate_limit)
    except Exception:
        return None
    return RateLimiter(rate=rate) if rate > 0 else None

class LLMProvider:
    """Abstract base for LLM chat providers.

    All providers accept `force_json` (default True). When False, providers
    must NOT request constrained JSON output, so callers can emit free-form
    text (e.g. unified diffs) without backslash/quote escaping artefacts.
    """
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        force_json: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Yield incremental text deltas. Default fallback: call :meth:`chat`
        once and yield the whole content. Concrete providers override for
        real token streaming (used by the UI 'thought process' panel)."""
        out = self.chat(
            messages, temperature=temperature, max_tokens=max_tokens,
            force_json=False, **kwargs,
        )
        text = out.get("content", "") if isinstance(out, dict) else ""
        if text:
            yield text


class OllamaProvider(LLMProvider):
    """
    Calls a local Ollama server (default http://localhost:11434) with JSON mode enforced.

    Notes
    -----
    - We set `format: "json"` so the model must emit a single JSON object.
    - We disable streaming to get a single response payload.
    - We pass through `system`, `user`, `assistant` roles as-is.
    """
    def __init__(
        self,
        model: str = "qwen2.5-coder:3b",
        base_url: str = "http://localhost:11434",
        timeout: float = DEFAULT_TIMEOUT,
        extra_options: Optional[Dict[str, Any]] = None,
        rate_limit: Optional[float] = None,
        max_retries: int = 0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.extra_options = extra_options or {}
        # Rate limiting + retry are opt-in. Ollama is local by default so the
        # defaults here are no-ops; remote OpenAI-compatible setups typically
        # want a small ``rate_limit`` and ``max_retries=3``.
        self._limiter = _build_limiter(rate_limit)
        self._max_retries = max(0, int(max_retries))

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        force_json: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/api/chat"
        # Ollama accepts options.*; we keep it minimal and override with caller extras.
        options = {"temperature": float(temperature)}
        options.update(self.extra_options)
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if force_json:
            # Constrains the sampler to emit a single JSON object.
            payload["format"] = "json"
        if max_tokens is not None:
            # Ollama supports `num_predict` as a cap on tokens to generate
            options["num_predict"] = int(max_tokens)

        resp = request_with_retry(
            lambda: requests.post(url, json=payload, timeout=self.timeout),
            limiter=self._limiter,
            max_retries=self._max_retries,
        )
        try:
            resp.raise_for_status()
        except Exception as e:
            return {"content": "", "error": f"Ollama HTTP {resp.status_code}: {e}", "raw": getattr(resp, "text", "")}

        data = resp.json()
        # When `format=json`, Ollama returns the assistant content as a JSON string.
        content = ""
        if isinstance(data, dict):
            msg = data.get("message") or {}
            content = msg.get("content", "") or ""
            # Defensive: Sometimes models include stray text around JSON; return raw too.
        return {
            "content": content,
            "provider": "ollama",
            "model": self.model,
            "raw": data,
        }

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream deltas from Ollama via NDJSON lines on /api/chat."""
        url = f"{self.base_url}/api/chat"
        options = {"temperature": float(temperature)}
        options.update(self.extra_options)
        if max_tokens is not None:
            options["num_predict"] = int(max_tokens)
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": options,
        }
        try:
            with requests.post(url, json=payload, timeout=self.timeout, stream=True) as resp:
                resp.raise_for_status()
                for raw in resp.iter_lines(decode_unicode=True):
                    if not raw:
                        continue
                    try:
                        obj = json.loads(raw)
                    except Exception:
                        continue
                    msg = obj.get("message") or {}
                    delta = msg.get("content") or ""
                    if delta:
                        yield delta
                    if obj.get("done"):
                        break
        except Exception as e:
            yield f"\n[stream error: {type(e).__name__}: {e}]"


class OpenAICompatProvider(LLMProvider):
    """
    Talks to an OpenAI-compatible /v1/chat/completions endpoint.

    We request JSON via response_format={'type': 'json_object'} when the server supports it.
    If the server rejects that field, we fall back to plain text and let the caller parse.
    """
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        headers: Optional[Dict[str, str]] = None,
        extra_options: Optional[Dict[str, Any]] = None,
        rate_limit: Optional[float] = None,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or ""
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        if headers:
            self.headers.update(headers)
        # Overrides applied on every chat() call (e.g. temperature, max_tokens).
        self.extra_options = extra_options or {}
        # Remote endpoints can return 429/5xx; opt in to retries by default
        # for OpenAI-compatible providers (still no-op'd when rate_limit is None).
        self._limiter = _build_limiter(rate_limit)
        self._max_retries = max(0, int(max_retries))

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        force_json: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/v1/chat/completions"
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
            "stream": False,
        }
        if max_tokens is not None:
            body["max_tokens"] = int(max_tokens)
        # Per-instance overrides win over per-call defaults so the UI can pin
        # temperature/max_tokens regardless of internal engine defaults.
        if self.extra_options:
            body.update(self.extra_options)

        def _do_post(b: Dict[str, Any]):
            return request_with_retry(
                lambda: requests.post(url, json=b, headers=self.headers, timeout=self.timeout),
                limiter=self._limiter,
                max_retries=self._max_retries,
            )

        if force_json:
            # Prefer strict JSON responses if supported; fall back on 4xx.
            try:
                body["response_format"] = {"type": "json_object"}
                resp = _do_post(body)
                if resp.status_code >= 400:
                    raise RuntimeError(f"HTTP {resp.status_code}")
            except Exception:
                body.pop("response_format", None)
                resp = _do_post(body)
        else:
            resp = _do_post(body)

        try:
            resp.raise_for_status()
        except Exception as e:
            return {"content": "", "error": f"OpenAICompat HTTP {resp.status_code}: {e}", "raw": getattr(resp, "text", "")}

        data = resp.json()
        content = ""
        try:
            content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
        except Exception:
            content = ""

        return {
            "content": content,
            "provider": "openai-compat",
            "model": self.model,
            "raw": data,
        }

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream deltas from OpenAI-compatible SSE chat completions."""
        url = f"{self.base_url}/v1/chat/completions"
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
            "stream": True,
        }
        if max_tokens is not None:
            body["max_tokens"] = int(max_tokens)
        if self.extra_options:
            body.update(self.extra_options)
            body["stream"] = True
        try:
            with requests.post(url, json=body, headers=self.headers, timeout=self.timeout, stream=True) as resp:
                resp.raise_for_status()
                for raw in resp.iter_lines(decode_unicode=True):
                    if not raw:
                        continue
                    line = raw.strip()
                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if not line or line == "[DONE]":
                        if line == "[DONE]":
                            break
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    delta = (choices[0].get("delta") or {}).get("content") or ""
                    if delta:
                        yield delta
                    if choices[0].get("finish_reason"):
                        break
        except Exception as e:
            yield f"\n[stream error: {type(e).__name__}: {e}]"