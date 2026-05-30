"""Tests for cgx.answer.ollama_discovery. Network calls are stubbed."""

from __future__ import annotations

from typing import Any, Dict

import pytest

from cgx.answer import ollama_discovery as od


class _Resp:
    def __init__(self, status: int, payload: Dict[str, Any]):
        self.status_code = status
        self.ok = 200 <= status < 300
        self._payload = payload

    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def test_list_installed_models_parses_tags(monkeypatch):
    payload = {
        "models": [
            {"name": "qwen2.5-coder:3b", "size": 1234,
             "details": {"family": "qwen", "parameter_size": "3B"}},
            {"name": "llama3.2:3b-instruct"},
            "garbage",
        ]
    }
    monkeypatch.setattr(od.requests, "get",
                        lambda *a, **kw: _Resp(200, payload))
    out = od.list_installed_models("http://localhost:11434")
    names = [m["name"] for m in out]
    assert "qwen2.5-coder:3b" in names
    assert "llama3.2:3b-instruct" in names


def test_list_installed_models_handles_failure(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("connection refused")
    monkeypatch.setattr(od.requests, "get", boom)
    assert od.list_installed_models() == []


def test_health_check_ok(monkeypatch):
    monkeypatch.setattr(od.requests, "get",
                        lambda *a, **kw: _Resp(200, {"models": [{"name": "x"}]}))
    out = od.health_check("http://localhost:11434")
    assert out["ok"] is True
    assert out["models_count"] == 1


def test_health_check_failure(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("nope")
    monkeypatch.setattr(od.requests, "get", boom)
    out = od.health_check()
    assert out["ok"] is False
    assert "error" in out


def test_recommend_default_model_prefers_installed(monkeypatch):
    monkeypatch.setattr(
        od, "list_installed_models",
        lambda *a, **kw: [{"name": "qwen2.5-coder:1.5b"}],
    )
    monkeypatch.setattr(od, "detect_hardware",
                        lambda: {"ram_gb": 8.0, "gpu_vram_gb": None})
    pick = od.recommend_default_model()
    assert pick == "qwen2.5-coder:1.5b"


def test_model_choices_dedups_and_orders(monkeypatch):
    monkeypatch.setattr(
        od, "list_installed_models",
        lambda *a, **kw: [{"name": "qwen2.5-coder:3b"}, {"name": "custom:latest"}],
    )
    choices = od.model_choices()
    assert choices[:2] == ["qwen2.5-coder:3b", "custom:latest"]
    # ladder appended without duplicating installed entries
    assert "qwen2.5-coder:1.5b" in choices
    assert choices.count("qwen2.5-coder:3b") == 1
