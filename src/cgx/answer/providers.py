# src/cgx/answer/providers.py
from __future__ import annotations
import os
import json
from typing import Any, Dict, List, Optional
import requests  # intentionally re-exported; UI imports this to ensure dependency

DEFAULT_TIMEOUT = float(os.environ.get("CGX_HTTP_TIMEOUT", "120"))

class LLMProvider:
    """Abstract base for LLM chat providers."""
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        raise NotImplementedError


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
        model: str = "qwen2.5:7b-instruct",
        base_url: str = "http://localhost:11434",
        timeout: float = DEFAULT_TIMEOUT,
        extra_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.extra_options = extra_options or {}

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
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
            # CRITICAL: force JSON output from the model tokenizer/sampler
            "format": "json",
        }
        if max_tokens is not None:
            # Ollama supports `num_predict` as a cap on tokens to generate
            options["num_predict"] = int(max_tokens)

        resp = requests.post(url, json=payload, timeout=self.timeout)
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

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
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

        # Prefer strict JSON responses if supported
        try:
            body["response_format"] = {"type": "json_object"}
            resp = requests.post(url, json=body, headers=self.headers, timeout=self.timeout)
            # Some servers return 400 for unknown field; if so, retry without it.
            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code}")
        except Exception:
            body.pop("response_format", None)
            resp = requests.post(url, json=body, headers=self.headers, timeout=self.timeout)

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