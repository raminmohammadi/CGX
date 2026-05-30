"""Tests for the provider profile store."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest


@pytest.fixture()
def profiles_module(tmp_path, monkeypatch):
    monkeypatch.setenv("CGX_CONFIG_DIR", str(tmp_path))
    import cgx.answer.profiles as profiles
    importlib.reload(profiles)
    # Force file-backed secret store for deterministic tests.
    monkeypatch.setattr(profiles, "_keyring", lambda: None)
    return profiles


def test_save_and_load_profile_roundtrip(profiles_module):
    P = profiles_module
    P.save_profile(
        P.Profile(name="alpha", kind="ollama", model="qwen2.5-coder:3b",
                  base_url="http://localhost:11434"),
        api_key=None,
    )
    profs = P.list_profiles()
    assert any(p.name == "alpha" for p in profs)
    got = P.get_profile("alpha")
    assert got is not None
    assert got.kind == "ollama"
    assert got.has_api_key is False


def test_save_profile_with_api_key_persists_secret(profiles_module, tmp_path):
    P = profiles_module
    P.save_profile(
        P.Profile(name="cloud", kind="openai-compat", model="gpt-4o",
                  base_url="https://api.example.com"),
        api_key="sk-secret-value",
    )
    got = P.get_profile("cloud")
    assert got is not None and got.has_api_key is True
    secret = P.load_api_key("cloud")
    assert secret == "sk-secret-value"
    # Secrets file must exist with restrictive perms on POSIX.
    sec_path = Path(os.environ["CGX_CONFIG_DIR"]) / "secrets.json"
    assert sec_path.exists()
    mode = oct(sec_path.stat().st_mode)[-3:]
    assert mode in {"600", "640"}  # tolerate umask defaults


def test_delete_profile_removes_secret(profiles_module):
    P = profiles_module
    P.save_profile(
        P.Profile(name="temp", kind="ollama", model="qwen2.5-coder:3b",
                  base_url="http://localhost:11434"),
        api_key="will-be-deleted",
    )
    assert P.load_api_key("temp") == "will-be-deleted"
    assert P.delete_profile("temp") is True
    assert P.get_profile("temp") is None
    assert P.load_api_key("temp") is None
