"""Tests for the /api/embed/* routes in :mod:`cgx.webui.routes.embed`."""

from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from cgx.webui.routes.embed import (
    EmbedPullRequest,
    _prepare_online_pull,
    _restore_online_pull,
    embed_pull,
    list_embed_models,
)


# ---------------------------------------------------------------------------
# GET /api/embed/models
# ---------------------------------------------------------------------------
def test_list_embed_models_returns_catalog_with_cached_flag(monkeypatch):
    # Force every entry to look "not cached" so the test doesn't depend on
    # what the developer happens to have in ~/.cache/huggingface.
    monkeypatch.setattr(
        "cgx.webui.routes.embed.is_cached", lambda name: False
    )
    r = list_embed_models()
    assert r.recommended_default == "jinaai/jina-embeddings-v2-base-code"
    names = [c.name for c in r.choices]
    assert "jinaai/jina-embeddings-v2-base-code" in names
    assert "BAAI/bge-m3" in names
    assert "Qwen/Qwen3-Embedding-8B" in names
    assert all(c.cached is False for c in r.choices)


def test_list_embed_models_reports_cached_when_probe_says_so(monkeypatch):
    monkeypatch.setattr(
        "cgx.webui.routes.embed.is_cached",
        lambda name: name == "BAAI/bge-m3",
    )
    r = list_embed_models()
    by_name = {c.name: c for c in r.choices}
    assert by_name["BAAI/bge-m3"].cached is True
    assert by_name["jinaai/jina-embeddings-v2-base-code"].cached is False


# ---------------------------------------------------------------------------
# POST /api/embed/pull (SSE)
# ---------------------------------------------------------------------------
def _sibling(name: str, size: int) -> SimpleNamespace:
    return SimpleNamespace(rfilename=name, size=size)


def _fake_model_info(siblings: List[SimpleNamespace]):
    """Build the (HfApi, hf_hub_download) replacement pair used in patches."""
    class _FakeApi:
        def model_info(self, repo_id, files_metadata=True):
            return SimpleNamespace(siblings=list(siblings))

    calls: List[str] = []

    def fake_download(repo_id, filename, **_kw):
        calls.append(filename)
        return f"/tmp/fake-cache/{repo_id}/{filename}"

    return _FakeApi, fake_download, calls


async def _drain(response) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    async for frame in response.body_iterator:
        out.append(frame)
    return out


def test_embed_pull_streams_progress_and_done_on_success():
    siblings = [
        _sibling("config.json", 1_000),
        _sibling("tokenizer.json", 5_000),
        _sibling("model.safetensors", 2_000_000),
        # The .bin duplicate should be skipped when .safetensors is present.
        _sibling("pytorch_model.bin", 2_000_000),
        # Runtime variants should also be filtered.
        _sibling("model.onnx", 1_500_000),
    ]
    FakeApi, fake_download, calls = _fake_model_info(siblings)

    with patch("huggingface_hub.HfApi", FakeApi), \
         patch("huggingface_hub.hf_hub_download", side_effect=fake_download):

        async def run():
            resp = await embed_pull(EmbedPullRequest(model="fake/model"))
            return await _drain(resp)

        frames = asyncio.run(run())

    events = [f["event"] for f in frames]
    payloads = [
        json.loads(f["data"]) if f["data"] else {} for f in frames
    ]

    # Last frame is always the terminal ``done`` event.
    assert events[-1] == "done"
    # Every other frame is a ``progress`` event.
    assert set(events[:-1]) == {"progress"}

    statuses = [p.get("status", "") for p in payloads[:-1]]
    assert any("Resolving" in s for s in statuses)
    assert any(s == "success" for s in statuses)

    # The .bin / .onnx siblings must have been filtered out.
    assert calls == ["config.json", "tokenizer.json", "model.safetensors"]

    # The cumulative ``completed`` counter is monotonically non-decreasing
    # and ends at the announced ``total``.
    success = next(p for p in payloads if p.get("status") == "success")
    assert success["total"] == 1_000 + 5_000 + 2_000_000
    assert success["completed"] == success["total"]

    progress_completed = [
        p["completed"] for p in payloads if p.get("status", "") not in ("", "error")
    ]
    assert progress_completed == sorted(progress_completed)


def test_prepare_and_restore_online_pull_round_trip(monkeypatch):
    """Pulls must override the launcher's offline mode then restore it."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    import huggingface_hub.constants as hc
    monkeypatch.setattr(hc, "HF_HUB_OFFLINE", True)

    prev = _prepare_online_pull()
    try:
        # Inside the pull window the worker thread sees an online state.
        assert os.environ.get("HF_HUB_OFFLINE") is None
        assert os.environ.get("TRANSFORMERS_OFFLINE") is None
        assert hc.HF_HUB_OFFLINE is False
    finally:
        _restore_online_pull(prev)

    # Caller's offline state is reinstated for downstream indexing code.
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    assert hc.HF_HUB_OFFLINE is True


def test_embed_pull_emits_error_frame_when_hub_call_raises():
    class _BoomApi:
        def model_info(self, *_a, **_kw):
            raise RuntimeError("hub offline")

    with patch("huggingface_hub.HfApi", _BoomApi), \
         patch("huggingface_hub.hf_hub_download", side_effect=AssertionError):

        async def run():
            resp = await embed_pull(EmbedPullRequest(model="fake/model"))
            return await _drain(resp)

        frames = asyncio.run(run())

    payloads = [json.loads(f["data"]) if f["data"] else {} for f in frames]
    err = next((p for p in payloads if p.get("status") == "error"), None)
    assert err is not None
    assert "hub offline" in err.get("error", "")
    # ``done`` still terminates the stream so the SSE client can close cleanly.
    assert frames[-1]["event"] == "done"
