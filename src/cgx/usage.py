"""Truthful token and cost accounting for LLM calls.

Two concerns, kept deliberately honest:

* **Tokens.** :func:`extract_usage` prefers the token counts the provider
  itself returns (Gemini ``usageMetadata``, OpenAI-compatible ``usage``,
  Ollama ``prompt_eval_count`` / ``eval_count``). Only when the backend
  reports nothing do we fall back to a coarse ``~4 chars/token`` estimate,
  and the result is tagged ``token_source="estimated"`` so downstream
  cost/quota logic never silently treats a guess as ground truth.

* **Cost.** :func:`estimate_cost` multiplies tokens by a per-model price
  table (USD per 1M tokens). Prices are volatile, so the built-in table is
  a small set of approximate list prices that is fully overridable via the
  ``CGX_MODEL_PRICING`` env var (a JSON map ``{model: {"in": x, "out": y}}``
  in USD per 1M tokens). Unknown models return ``0.0`` with
  ``cost_source="unknown"`` rather than a fabricated number.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

# Approximate public list prices (USD per 1,000,000 tokens) as of 2026.
# These are intentionally conservative defaults and are overridden by the
# CGX_MODEL_PRICING env var. Keys are matched by case-insensitive prefix.
_DEFAULT_PRICING: Dict[str, Tuple[float, float]] = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-1.5-pro": (1.25, 5.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.0),
}


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def extract_usage(response: Any, *, prompt_text: str = "",
                  response_text: str = "") -> Dict[str, Any]:
    """Return ``{tokens_in, tokens_out, tokens_total, token_source}``.

    ``token_source`` is ``"provider"`` when the backend reported usage,
    otherwise ``"estimated"``.
    """
    raw: Any = None
    if isinstance(response, dict):
        raw = response.get("raw")
    tin = tout = None
    if isinstance(raw, dict):
        # Gemini: usageMetadata.{promptTokenCount,candidatesTokenCount}
        um = raw.get("usageMetadata")
        if isinstance(um, dict):
            tin = um.get("promptTokenCount")
            tout = um.get("candidatesTokenCount")
        # OpenAI-compatible: usage.{prompt_tokens,completion_tokens}
        usage = raw.get("usage")
        if isinstance(usage, dict):
            tin = usage.get("prompt_tokens", tin)
            tout = usage.get("completion_tokens", tout)
        # Ollama: prompt_eval_count / eval_count at the top level.
        if raw.get("prompt_eval_count") is not None:
            tin = raw.get("prompt_eval_count")
        if raw.get("eval_count") is not None:
            tout = raw.get("eval_count")
    if isinstance(tin, (int, float)) and isinstance(tout, (int, float)):
        tin_i, tout_i = int(tin), int(tout)
        return {"tokens_in": tin_i, "tokens_out": tout_i,
                "tokens_total": tin_i + tout_i, "token_source": "provider"}
    tin_i = _estimate_tokens(prompt_text)
    tout_i = _estimate_tokens(response_text)
    return {"tokens_in": tin_i, "tokens_out": tout_i,
            "tokens_total": tin_i + tout_i, "token_source": "estimated"}


def _env_pricing() -> Dict[str, Tuple[float, float]]:
    """Parse the ``CGX_MODEL_PRICING`` JSON override into a price table."""
    table: Dict[str, Tuple[float, float]] = {}
    raw = os.environ.get("CGX_MODEL_PRICING", "").strip()
    if not raw:
        return table
    try:
        data = json.loads(raw)
    except Exception:
        return table
    if isinstance(data, dict):
        for model, spec in data.items():
            if isinstance(spec, dict):
                try:
                    table[str(model).lower()] = (
                        float(spec.get("in", 0.0)), float(spec.get("out", 0.0)))
                except Exception:
                    continue
    return table


def _lookup_price(model: Optional[str],
                  table: Dict[str, Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    if not model:
        return None
    m = model.lower()
    if m in table:
        return table[m]
    # Prefix match so "gpt-4o-2024-08-06" resolves to "gpt-4o".
    best: Optional[Tuple[float, float]] = None
    best_len = -1
    for key, price in table.items():
        if m.startswith(key) and len(key) > best_len:
            best, best_len = price, len(key)
    return best


def estimate_cost(model: Optional[str], tokens_in: int,
                  tokens_out: int) -> Dict[str, Any]:
    """Return ``{cost_usd, cost_source}`` for the given token counts.

    ``cost_source`` is ``"config"`` when priced from the env override,
    ``"default"`` from the built-in table, or ``"unknown"`` (cost 0.0)
    when the model is not priced.
    """
    env = _env_pricing()
    # Env overrides win; the source reflects which table actually priced it.
    price = _lookup_price(model, env)
    source = "config"
    if price is None:
        price = _lookup_price(model, _DEFAULT_PRICING)
        source = "default"
    if price is None:
        return {"cost_usd": 0.0, "cost_source": "unknown"}
    cost = (tokens_in / 1_000_000.0) * price[0] + (tokens_out / 1_000_000.0) * price[1]
    return {"cost_usd": round(cost, 6), "cost_source": source}
