"""Construct LLM providers from a small, surface-agnostic knob set.

This factory lives in the engine (``cgx.answer``) rather than under
``cgx.webui`` so both the web UI *and* the CLI can build providers without
importing the ``ui``/FastAPI extra. ``cgx.webui.helpers`` re-exports these
names for backward compatibility.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from cgx.answer.model_caps import get_model_context_window
from cgx.answer.profiles import get_profile, load_api_key
from cgx.answer.providers import (
    GeminiProvider,
    LLMProvider,
    OllamaProvider,
    OpenAICompatProvider,
)

# Conservative auto-cap for the Ollama KV-cache. Ollama defaults to 2048-4096
# without an explicit ``num_ctx``; bumping to 8K covers the common "tell me
# about this project" round-trip without ballooning VRAM on 8 GB GPUs running
# a 12 B model. Users can override per-profile to push higher when they have
# the headroom.
DEFAULT_OLLAMA_NUM_CTX_CAP = 8_192


def _effective_ollama_num_ctx(model: str, override: Optional[int]) -> int:
    """Resolve the ``num_ctx`` to send to Ollama for ``model``.

    ``override`` wins when positive. Otherwise the model's registry-reported
    window is clamped to :data:`DEFAULT_OLLAMA_NUM_CTX_CAP` so a 256 K-window
    model doesn't accidentally force CPU offload on a modest GPU.
    """
    if override is not None and int(override) > 0:
        return int(override)
    window = get_model_context_window(model)
    return min(int(window), DEFAULT_OLLAMA_NUM_CTX_CAP)


def build_provider(
    *,
    kind: str,
    model: str,
    base_url: str,
    api_key: Optional[str] = None,
    temperature: float = 0.2,
    num_predict: int = 1024,
    num_ctx: Optional[int] = None,
    rate_limit: Optional[float] = None,
    max_retries: Optional[int] = None,
    endpoint_path: str = "/v1/chat/completions",
    allow_no_auth: bool = False,
) -> LLMProvider:
    """Construct a provider with per-call overrides for temperature/tokens.

    Supports five kinds:
      - ``ollama``       -- local Ollama server
      - ``openai-compat``-- OpenAI or any /v1/chat/completions-compatible API
      - ``gemini``       -- Google Gemini via REST
      - ``huggingface``  -- Hugging Face Inference Providers (OpenAI-compatible router)
      - ``custom``       -- OpenAI-compatible with custom host, path, and optional auth-bypass
    """
    ollama_opts: Dict[str, Any] = {"temperature": float(temperature),
                                   "num_predict": int(num_predict)}
    openai_opts: Dict[str, Any] = {"temperature": float(temperature),
                                   "max_tokens": int(num_predict)}
    rl_kwargs: Dict[str, Any] = {}
    if rate_limit is not None:
        rl_kwargs["rate_limit"] = float(rate_limit)
    if max_retries is not None:
        rl_kwargs["max_retries"] = int(max_retries)

    if kind == "ollama":
        base = (base_url or "http://localhost:11434").replace("/v1", "").rstrip("/")
        ctx = _effective_ollama_num_ctx(model, num_ctx)
        if ctx is not None:
            ollama_opts["num_ctx"] = ctx
        return OllamaProvider(model=model, base_url=base,
                              extra_options=ollama_opts, **rl_kwargs)

    if kind == "gemini":
        return GeminiProvider(
            model=model or "gemini-2.5-flash",
            api_key=api_key or "",
            **rl_kwargs,
        )

    if kind == "huggingface":
        # Hugging Face Inference Providers expose an OpenAI-compatible router at
        # https://router.huggingface.co/v1, so ``OpenAICompatProvider`` works
        # verbatim. The only HF-specifics are the fixed host and the ``hf_...``
        # bearer token, which we also accept from the standard HF env vars so a
        # user who exported one doesn't have to paste it into a profile.
        hf_key = (api_key or os.environ.get("HF_TOKEN")
                  or os.environ.get("HUGGINGFACEHUB_API_TOKEN") or None)
        return OpenAICompatProvider(
            model=model,
            base_url="https://router.huggingface.co",
            api_key=hf_key,
            extra_options=openai_opts,
            endpoint_path="/v1/chat/completions",
            **rl_kwargs,
        )

    # "openai-compat" and "custom" both use OpenAICompatProvider; "custom"
    # additionally supports a non-standard endpoint path and auth bypass.
    eff_endpoint_path = endpoint_path or "/v1/chat/completions"
    return OpenAICompatProvider(
        model=model,
        base_url=(base_url or "").rstrip("/"),
        api_key=api_key or None,
        extra_options=openai_opts,
        endpoint_path=eff_endpoint_path,
        allow_no_auth=bool(allow_no_auth),
        **rl_kwargs,
    )


def provider_from_profile_name(name: str) -> LLMProvider:
    """Resolve a saved profile by name into a ready-to-use provider."""
    p = get_profile(name)
    if p is None:
        raise ValueError(f"Profile not found: {name!r}")
    api_key = load_api_key(p.name) if p.has_api_key else None
    return build_provider(
        kind=p.kind, model=p.model, base_url=p.base_url, api_key=api_key,
        temperature=p.temperature, num_predict=p.num_predict,
        num_ctx=getattr(p, "num_ctx", None),
        rate_limit=getattr(p, "rate_limit", None),
        max_retries=getattr(p, "max_retries", None),
        endpoint_path=getattr(p, "endpoint_path", "/v1/chat/completions"),
        allow_no_auth=getattr(p, "allow_no_auth", False),
    )
