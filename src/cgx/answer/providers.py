

# src/cgx/answer/providers.py
from __future__ import annotations
import logging
import os
import json
import re
from typing import Any, Dict, Iterator, List, Optional
import requests  # intentionally re-exported; UI imports this to ensure dependency

from cgx.answer.ratelimit import RateLimiter, request_with_retry

DEFAULT_TIMEOUT = float(os.environ.get("CGX_HTTP_TIMEOUT", "120"))

logger = logging.getLogger(__name__)


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


_MISSING_GEMINI_KEY_MSG = (
    "Gemini API key not configured. Provide one via the UI profile, the "
    "request body, or the GEMINI_API_KEY environment variable."
)


class GeminiProvider(LLMProvider):
    """Google Gemini via the official REST API (generativelanguage.googleapis.com).

    Maps CGX's internal ``messages`` format to Gemini's ``contents`` +
    ``systemInstruction`` format so the orchestration layer stays provider-agnostic.
    """

    _BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str = "",
        timeout: float = DEFAULT_TIMEOUT,
        rate_limit: Optional[float] = None,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or ""
        self.timeout = timeout
        self._limiter = _build_limiter(rate_limit)
        self._max_retries = max(0, int(max_retries))

    def _url(self, method: str = "generateContent") -> str:
        return f"{self._BASE}/{self.model}:{method}?key={self.api_key}"

    @staticmethod
    def _map_messages(
        messages: List[Dict[str, str]],
    ) -> tuple:
        """Split CGX messages into (system_text, gemini_contents)."""
        system_parts: List[str] = []
        contents: List[Dict] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_parts.append(content)
            else:
                gemini_role = "user" if role == "user" else "model"
                # Merge consecutive same-role turns (Gemini requires alternating).
                if contents and contents[-1]["role"] == gemini_role:
                    contents[-1]["parts"][0]["text"] += "\n" + content
                else:
                    contents.append({"role": gemini_role, "parts": [{"text": content}]})
        return "\n\n".join(system_parts), contents

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        force_json: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        if not self.api_key:
            return {
                "content": "",
                "error": _MISSING_GEMINI_KEY_MSG,
                "raw": "",
            }
        system_text, contents = self._map_messages(messages)
        body: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"temperature": float(temperature)},
        }
        if system_text:
            body["systemInstruction"] = {"parts": [{"text": system_text}]}
        if max_tokens:
            body["generationConfig"]["maxOutputTokens"] = int(max_tokens)
        if force_json:
            # Gemini supports constrained JSON output via responseMimeType.
            body["generationConfig"]["responseMimeType"] = "application/json"

        resp = request_with_retry(
            lambda: requests.post(self._url(), json=body, timeout=self.timeout),
            limiter=self._limiter,
            max_retries=self._max_retries,
        )
        try:
            resp.raise_for_status()
        except Exception as e:
            return {
                "content": "",
                "error": f"Gemini HTTP {resp.status_code}: {self._scrub_secret(str(e))}",
                "raw": getattr(resp, "text", ""),
            }
        data = resp.json()
        content = ""
        try:
            content = data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception:
            pass
        result: Dict[str, Any] = {
            "content": content, "provider": "gemini", "model": self.model, "raw": data,
        }
        if not content:
            reason = self._diagnose_empty_response(data)
            if reason:
                logger.warning("Gemini returned empty content: %s", reason)
                result["error"] = reason
        return result

    @staticmethod
    def _scrub_secret(text: str) -> str:
        """Redact ``key=<value>`` query parameters so the API key never appears
        in logs or error payloads. Matches Gemini's URL-style key parameter
        until the next ``&``, whitespace, or end-of-string. Empty values
        are rewritten to ``<missing>`` so logs disambiguate a redacted key
        from a request that never carried one."""
        if not text:
            return text

        def _sub(m: "re.Match[str]") -> str:
            return m.group(1) + ("<redacted>" if m.group(2) else "<missing>")

        return re.sub(r"([?&]key=)([^&\s]*)", _sub, text)

    @staticmethod
    def _diagnose_empty_response(data: Any) -> str:
        """Return a human-readable reason when the Gemini response carries no text.

        Inspects the documented Gemini error/feedback shapes — top-level
        ``error.message``, ``promptFeedback.blockReason``, the candidate's
        ``finishReason`` (``SAFETY`` / ``MAX_TOKENS`` / ``RECITATION`` /
        ``OTHER``), and any non-``NEGLIGIBLE`` ``safetyRatings`` — so the
        caller can surface why the call produced no content instead of
        silently degrading to an empty plan.
        """
        if not isinstance(data, dict):
            return ""
        err = data.get("error")
        if isinstance(err, dict) and err.get("message"):
            status = err.get("status") or err.get("code") or ""
            return f"Gemini API error: {err['message']}" + (
                f" (status={status})" if status else ""
            )
        pf = data.get("promptFeedback") or {}
        if isinstance(pf, dict) and pf.get("blockReason"):
            return f"Prompt blocked: {pf['blockReason']}"
        candidates = data.get("candidates") or []
        if not candidates:
            return "No candidates returned (possibly blocked by safety filters)"
        cand = candidates[0] if isinstance(candidates[0], dict) else {}
        finish = cand.get("finishReason") or ""
        if finish and finish.upper() not in ("STOP", "FINISH_REASON_UNSPECIFIED"):
            blocked: List[str] = []
            for r in cand.get("safetyRatings") or []:
                if isinstance(r, dict) and r.get("blocked"):
                    blocked.append(str(r.get("category") or "unknown"))
            extra = f" (safety categories: {', '.join(blocked)})" if blocked else ""
            return f"Response truncated: finishReason={finish}{extra}"
        return "Response contained no text parts"

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream via Gemini's streamGenerateContent SSE endpoint."""
        if not self.api_key:
            yield f"\n[stream error: {_MISSING_GEMINI_KEY_MSG}]"
            return
        system_text, contents = self._map_messages(messages)
        body: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"temperature": float(temperature)},
        }
        if system_text:
            body["systemInstruction"] = {"parts": [{"text": system_text}]}
        if max_tokens:
            body["generationConfig"]["maxOutputTokens"] = int(max_tokens)
        url = self._url("streamGenerateContent") + "&alt=sse"
        try:
            with requests.post(url, json=body, timeout=self.timeout, stream=True) as resp:
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
                        delta = obj["candidates"][0]["content"]["parts"][0]["text"]
                        if delta:
                            yield delta
                    except Exception:
                        continue
        except Exception as e:
            yield f"\n[stream error: {type(e).__name__}: {self._scrub_secret(str(e))}]"


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
        endpoint_path: str = "/v1/chat/completions",
        allow_no_auth: bool = False,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.endpoint_path = endpoint_path or "/v1/chat/completions"
        self.allow_no_auth = bool(allow_no_auth)
        self.api_key = api_key or (os.environ.get("OPENAI_API_KEY") or "") if not allow_no_auth else (api_key or "")
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
        path = self.endpoint_path if self.endpoint_path.startswith("/") else "/" + self.endpoint_path
        url = f"{self.base_url}{path}"
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
        path = self.endpoint_path if self.endpoint_path.startswith("/") else "/" + self.endpoint_path
        url = f"{self.base_url}{path}"
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