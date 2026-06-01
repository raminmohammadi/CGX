"""Tests for the cloud_models discovery endpoint in cgx.webui.routes.setup."""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import patch

import pytest

from cgx.webui.routes.setup import (
    CloudModelsRequest,
    _GEMINI_FALLBACK,
    _OPENAI_FALLBACK,
    cloud_models,
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
    # key — the dropdown stays populated from the static fallback.
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

    def fake_get(url, *_args, **_kwargs):
        captured["url"] = url
        return _FakeResp(_gemini_payload())

    with patch("requests.get", side_effect=fake_get):
        cloud_models(CloudModelsRequest(kind="gemini", profile_name="p1"))
    assert "key=from-store" in captured["url"]
