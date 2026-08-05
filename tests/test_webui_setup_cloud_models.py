"""Tests for the cloud_models discovery endpoint in cgx.webui.routes.setup."""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import patch

import pytest

from cgx.webui.routes.setup import (
    CloudModelsRequest,
    _GEMINI_FALLBACK,
    _HF_FALLBACK,
    _OPENAI_FALLBACK,
    _sanitize_local_name,
    cloud_models,
    hf_models,
)


class _FakeResp:
    def __init__(self, json_data: Dict[str, Any], status_code: int = 200):
        self._json = json_data
        self.status_code = status_code
        self.content = b"x" if json_data else b""

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            from requests import HTTPError
            raise HTTPError(
                f"{self.status_code} Client Error: Bad for url: "
                f"https://x/v1beta/models?key=SECRET123"
            )

    def json(self) -> Dict[str, Any]:
        return self._json


def _gemini_payload() -> Dict[str, Any]:
    return {"models": [
        {"name": "models/gemini-2.5-flash",
         "supportedGenerationMethods": ["generateContent", "countTokens"]},
        {"name": "models/gemini-2.5-pro",
         "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-2.5-flash-image",
         "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/embedding-001",
         "supportedGenerationMethods": ["embedContent"]},
        {"name": "models/gemini-1.5-flash",
         "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/lyria-2",
         "supportedGenerationMethods": ["generateContent"]},
    ]}


def _openai_payload() -> Dict[str, Any]:
    return {"data": [
        {"id": "gpt-4o-mini"}, {"id": "gpt-4o"},
        {"id": "text-embedding-3-small"},
        {"id": "whisper-1"},
        {"id": "dall-e-3"},
        {"id": "o3-mini"},
        {"id": "chatgpt-4o-latest"},
    ]}


def test_gemini_lists_current_models_and_strips_models_prefix():
    with patch("requests.get", return_value=_FakeResp(_gemini_payload())):
        r = cloud_models(CloudModelsRequest(kind="gemini", api_key="k"))
    assert "gemini-2.5-flash" in r.choices
    assert "gemini-2.5-pro" in r.choices
    assert "gemini-1.5-flash" in r.choices  # legacy is still returned by API
    assert r.recommended_default == "gemini-2.5-flash"
    # Image / embedding / non-gemini families filtered out.
    assert "gemini-2.5-flash-image" not in r.choices
    assert "embedding-001" not in r.choices
    assert "lyria-2" not in r.choices
    # Names never carry the "models/" namespace prefix.
    assert all(not c.startswith("models/") for c in r.choices)


def test_openai_filters_to_chat_only_models():
    with patch("requests.get", return_value=_FakeResp(_openai_payload())):
        r = cloud_models(CloudModelsRequest(
            kind="openai-compat", api_key="sk-x",
            base_url="https://api.openai.com",
        ))
    assert "gpt-4o-mini" in r.choices
    assert "gpt-4o" in r.choices
    assert "o3-mini" in r.choices
    assert "chatgpt-4o-latest" in r.choices
    assert "text-embedding-3-small" not in r.choices
    assert "whisper-1" not in r.choices
    assert "dall-e-3" not in r.choices
    assert r.recommended_default == "gpt-4o-mini"


def test_no_api_key_falls_back_to_static_current_models_gemini():
    r = cloud_models(CloudModelsRequest(kind="gemini"))
    assert r.choices == list(_GEMINI_FALLBACK)
    assert r.recommended_default == "gemini-2.5-flash"
    assert "gemini-1.5-flash" not in r.choices


def test_no_api_key_falls_back_to_static_current_models_openai():
    r = cloud_models(CloudModelsRequest(kind="openai-compat"))
    assert r.choices == list(_OPENAI_FALLBACK)
    assert r.recommended_default == "gpt-4o-mini"


def test_unknown_kind_returns_empty():
    r = cloud_models(CloudModelsRequest(kind="bogus"))
    assert r.choices == []
    assert r.recommended_default == ""


def test_gemini_http_error_falls_back_without_leaking_key():
    # An API call that 4xx's should never crash the endpoint nor expose the
    # key -- the dropdown stays populated from the static fallback.
    err_resp = _FakeResp({}, status_code=404)
    err_resp.content = b""
    with patch("requests.get", return_value=err_resp):
        r = cloud_models(CloudModelsRequest(kind="gemini", api_key="SECRET123"))
    assert r.choices == list(_GEMINI_FALLBACK)
    assert "SECRET123" not in repr(r.choices)


def test_resolve_key_from_saved_profile(monkeypatch: pytest.MonkeyPatch):
    # Ensure a missing inline key is satisfied by the profile store lookup.
    from cgx.webui.routes import setup as setup_mod

    monkeypatch.setattr(setup_mod, "load_api_key", lambda name: "from-store" if name == "p1" else None)
    captured: Dict[str, Any] = {}

    def fake_get(url, *_args, **kwargs):
        captured["url"] = url
        captured["headers"] = kwargs.get("headers") or {}
        return _FakeResp(_gemini_payload())

    with patch("requests.get", side_effect=fake_get):
        cloud_models(CloudModelsRequest(kind="gemini", profile_name="p1"))
    # The resolved key is sent via the x-goog-api-key header, never the URL
    # query string, so it can't leak into request logs/proxies.
    assert captured["headers"].get("x-goog-api-key") == "from-store"
    assert "from-store" not in captured["url"]


# --------------------------- Hugging Face ---------------------------

def _hf_inference_payload() -> Dict[str, Any]:
    return {"data": [
        {"id": "openai/gpt-oss-20b",
         "architecture": {"input_modalities": ["text"],
                          "output_modalities": ["text"]}},
        {"id": "Qwen/Qwen2.5-Coder-32B-Instruct",
         "architecture": {"input_modalities": ["text"],
                          "output_modalities": ["text"]}},
        {"id": "black-forest-labs/FLUX.1-dev",
         "architecture": {"input_modalities": ["text"],
                          "output_modalities": ["image"]}},
    ]}


def _hf_hub_payload():
    return [
        {"id": "unsloth/Qwen3-Coder-GGUF", "downloads": 1000, "likes": 42,
         "pipeline_tag": "text-generation", "gated": False,
         "siblings": [{"rfilename": "model-Q4_K_M.gguf"},
                      {"rfilename": "model-Q8_0.gguf"},
                      {"rfilename": "README.md"}]},
    ]


def test_huggingface_lists_text_models_and_drops_image_only():
    with patch("requests.get", return_value=_FakeResp(_hf_inference_payload())):
        r = cloud_models(CloudModelsRequest(kind="huggingface"))
    assert "openai/gpt-oss-20b" in r.choices
    assert "Qwen/Qwen2.5-Coder-32B-Instruct" in r.choices
    # Image-output model is filtered out of the chat dropdown.
    assert "black-forest-labs/FLUX.1-dev" not in r.choices
    assert r.recommended_default == "Qwen/Qwen2.5-Coder-32B-Instruct"


def test_huggingface_falls_back_to_static_list_on_error():
    from requests import ConnectionError as _ConnErr
    with patch("requests.get", side_effect=_ConnErr("boom")):
        r = cloud_models(CloudModelsRequest(kind="huggingface"))
    assert r.choices == list(_HF_FALLBACK)
    assert r.recommended_default in _HF_FALLBACK


def test_hf_models_parses_hub_and_builds_pull_tags():
    with patch("requests.get", return_value=_FakeResp(_hf_hub_payload())):
        r = hf_models(search="qwen", sort="downloads", limit=5)
    assert len(r.models) == 1
    m = r.models[0]
    assert m.id == "unsloth/Qwen3-Coder-GGUF"
    assert m.pull_tag == "hf.co/unsloth/Qwen3-Coder-GGUF"
    # Quant labels are extracted from the .gguf siblings (README ignored).
    assert m.quants == ["Q4_K_M", "Q8_0"]
    assert m.downloads == 1000 and m.likes == 42


def test_hf_models_uses_fixed_host_and_sanitizes_sort():
    captured: Dict[str, Any] = {}

    def fake_get(url, *_args, **kwargs):
        captured["url"] = url
        captured["params"] = kwargs.get("params") or {}
        return _FakeResp(_hf_hub_payload())

    with patch("requests.get", side_effect=fake_get):
        hf_models(search="x", sort="../etc/passwd", limit=9999)
    # Host is the fixed Hub constant; an out-of-allowlist sort is coerced to the
    # Hub's camelCase default and the limit is clamped, so no attacker value
    # reaches the outbound request.
    assert captured["url"] == "https://huggingface.co/api/models"
    assert captured["params"]["sort"] == "trendingScore"
    assert captured["params"]["limit"] == 100
    assert captured["params"]["filter"] == "gguf"


def test_hf_models_translates_trending_sort_to_camelcase():
    captured: Dict[str, Any] = {}

    def fake_get(url, *_args, **kwargs):
        captured["params"] = kwargs.get("params") or {}
        return _FakeResp(_hf_hub_payload())

    # The Hub rejects snake_case ``trending_score`` with HTTP 400; the friendly
    # key the UI sends must be translated to the camelCase the Hub expects.
    with patch("requests.get", side_effect=fake_get):
        hf_models(search="", sort="trending_score", limit=40)
    assert captured["params"]["sort"] == "trendingScore"


def test_hf_models_returns_empty_on_error():
    from requests import ConnectionError as _ConnErr
    with patch("requests.get", side_effect=_ConnErr("down")):
        r = hf_models()
    assert r.models == []


# ------------------- local-name sanitisation (HF pull re-alias) -------------

@pytest.mark.parametrize("raw,expected", [
    # Bare name gets the implicit ``:latest`` tag.
    ("Ornith-1.0-9B-GGUF", "Ornith-1.0-9B-GGUF:latest"),
    # Explicit quant tag is preserved.
    ("ornith-1.0-9b-gguf:q4_k_m", "ornith-1.0-9b-gguf:q4_k_m"),
    # Empty / whitespace -> None (no rename requested).
    ("", None),
    ("   ", None),
    # A registry-style ``namespace/repo`` is rejected: it must be a single
    # bare model name, so it can't smuggle a host/namespace into /api/copy.
    ("ornith-ai/Ornith-1.0-9B-GGUF", None),
    # ``hf.co/...`` (contains '/') is likewise rejected.
    ("hf.co/ornith-ai/Ornith-1.0-9B-GGUF", None),
    # Odd characters are rejected rather than silently mangled.
    ("bad name!", None),
    ("../evil", None),
])
def test_sanitize_local_name(raw, expected):
    assert _sanitize_local_name(raw) == expected
