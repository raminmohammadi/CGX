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



def test_sort_model_choices_clusters_by_family():
    # Deliberately scrambled order spanning multiple families / versions.
    names = [
        "llama3.2:3b-instruct",
        "gemma4:12b",
        "qwen2.5-coder:7b-instruct",
        "gemma2:2b",
        "deepseek-r1:7b",
        "gemma4:e2b",
        "qwen2.5:7b-instruct",
        "gemma3:1b",
        "llama3.1:8b-instruct",
        "deepseek-coder:6.7b",
        "gemma4:e4b",
        "qwen2.5-coder:1.5b",
        "phi3.5:3.8b-mini-instruct",
        "custom:latest",
    ]
    out = od.sort_model_choices_by_family(names)

    # Each family appears as one contiguous run (no interleaving).
    def _root(n: str) -> str:
        import re
        m = re.match(r"^([a-z]+)", n)
        return m.group(1) if m else n
    seen: list[str] = []
    for n in out:
        r = _root(n)
        if not seen or seen[-1] != r:
            seen.append(r)
    assert len(seen) == len(set(seen)), \
        f"families not contiguous in sorted output: {out}"

    # All gemma variants together, ordered by version (2 → 3 → 4) then size.
    gemma = [n for n in out if n.startswith("gemma")]
    assert gemma == [
        "gemma2:2b",
        "gemma3:1b",
        "gemma4:e2b",  # e2b → 2.0 via size-hint regex
        "gemma4:e4b",  # e4b → 4.0
        "gemma4:12b",
    ], f"gemma order wrong: {gemma}"

    # Within deepseek: alphabetical sub-family (coder before r1).
    deepseek = [n for n in out if n.startswith("deepseek")]
    assert deepseek == ["deepseek-coder:6.7b", "deepseek-r1:7b"]


def test_sort_model_choices_uses_params_lookup_for_e_variants():
    # When an explicit lookup is supplied, it overrides the size-hint regex.
    names = ["gemma4:e2b", "gemma4:e4b", "gemma4:12b"]
    out = od.sort_model_choices_by_family(
        names,
        {"gemma4:e2b": 2.0, "gemma4:e4b": 4.0, "gemma4:12b": 12.0},
    )
    assert out == ["gemma4:e2b", "gemma4:e4b", "gemma4:12b"]
