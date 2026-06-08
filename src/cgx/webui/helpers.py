

"""Shared helpers for the CGX web UI handlers."""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from cgx.answer.model_caps import get_model_context_window
from cgx.answer.profiles import Profile, get_profile, load_api_key
from cgx.answer.providers import (
    GeminiProvider, LLMProvider, OllamaProvider, OpenAICompatProvider,
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

    Supports four kinds:
      - ``ollama``       -- local Ollama server
      - ``openai-compat``-- OpenAI or any /v1/chat/completions-compatible API
      - ``gemini``       -- Google Gemini via REST
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
        ollama_opts["num_ctx"] = _effective_ollama_num_ctx(model, num_ctx)
        return OllamaProvider(model=model, base_url=base,
                              extra_options=ollama_opts, **rl_kwargs)

    if kind == "gemini":
        return GeminiProvider(
            model=model or "gemini-2.5-flash",
            api_key=api_key or "",
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


def maybe_extract_zip(path: Optional[str]) -> Optional[str]:
    """Extract an uploaded ``.zip`` into a temp dir; return root path."""
    if not path or not os.path.exists(path):
        return None
    tmpdir = tempfile.mkdtemp(prefix="cgx_zip_")
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(tmpdir)
    entries = [p for p in Path(tmpdir).iterdir()]
    if len(entries) == 1 and entries[0].is_dir():
        return str(entries[0])
    return tmpdir


def json_safe(obj: Any) -> Any:
    """Best-effort coercion of nested objects into JSON-serialisable form."""
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        pass
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def stringify(value: Any) -> str:
    """Render any LLM-returned ``answer_md``-shaped value into a string."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("content", "text", "markdown", "md"):
            v = value.get(key)
            if isinstance(v, str) and v:
                return v
        try:
            return json.dumps(value, ensure_ascii=False, indent=2)
        except Exception:
            return str(value)
    if isinstance(value, list):
        return "\n".join(stringify(v) for v in value)
    return str(value)


def diffs_payload(diffs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Normalise plan diffs into a uniform ``[{file, patch}]`` list.

    The original Gradio surface formatted these as markdown for a single
    blob; the React diff viewer renders one card per file so we return
    structured records instead.
    """
    out: List[Dict[str, str]] = []
    for d in diffs or []:
        if not isinstance(d, dict):
            continue
        out.append({
            "file": str(d.get("file") or d.get("path") or "(unknown)"),
            "patch": str(d.get("patch") or d.get("diff") or ""),
        })
    return out


def report_summary(report: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Surface only the structured bits of a codegen report.

    The React Plan page renders the report itself; we just pass the
    dict through after filtering to JSON-safe values.
    """
    if not report:
        return None
    return json_safe(report)
