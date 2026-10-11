"""Vision critic: model discovery, screenshot review parsing, and the Verifier's
bounded vision round -- with the Ollama HTTP API mocked."""

from __future__ import annotations

import json

import pytest

from cgx import design
from cgx.design import vision


class _Resp:
    def __init__(self, payload):
        self._p = payload

    def json(self):
        return self._p


def test_resolve_vision_model_auto_and_explicit(monkeypatch):
    calls = []

    def fake_get(url, timeout=0):
        calls.append(url)
        return _Resp({"models": [{"name": "qwen2.5-coder:14b"}, {"name": "llava:13b"}, {"name": "qwen3-vl:8b"}]})
    monkeypatch.setattr(vision, "requests", type("R", (), {"get": staticmethod(fake_get)}), raising=False)
    import sys
    monkeypatch.setitem(sys.modules, "requests", vision.requests)

    class _P:
        base_url = "http://localhost:11434"
    monkeypatch.delenv("CGX_VISION_MODEL", raising=False)
    assert vision.resolve_vision_model(_P()) == ("http://localhost:11434", "qwen3-vl:8b")  # 'vl' preferred
    monkeypatch.setenv("CGX_VISION_MODEL", "llava")
    assert vision.resolve_vision_model(_P()) == ("http://localhost:11434", "llava:13b")
    monkeypatch.setenv("CGX_VISION_MODEL", "off")
    assert vision.resolve_vision_model(_P()) is None
    monkeypatch.setenv("CGX_VISION_MODEL", "auto")

    def no_vision(url, timeout=0):
        return _Resp({"models": [{"name": "qwen2.5-coder:14b"}]})
    monkeypatch.setattr(vision.requests, "get", staticmethod(no_vision))
    assert vision.resolve_vision_model(_P()) is None


def test_critique_screenshots_posts_images_and_parses_fixes(tmp_path, monkeypatch):
    shot = tmp_path / "index.png"
    shot.write_bytes(b"\x89PNG fake")
    posted = {}

    class _Req:
        @staticmethod
        def post(url, json=None, timeout=0):
            posted["url"] = url
            posted["body"] = json
            return _Resp({"message": {"content": "src/styles/app.css: make the .topbar span the full width\n"
                                                 "src/components/Composer.jsx: align the Send button with the textarea\n"
                                                 "nope.jsx: ignored"}})
    import sys
    monkeypatch.setitem(sys.modules, "requests", _Req)
    brief = design.build_brief("a chatbot app with an API key settings tab, React", ["react"])
    fixes = vision.critique_screenshots(brief, {"/": str(shot)}, ["src/styles/app.css", "src/components/Composer.jsx"],
                                        provider=None, model=("http://localhost:11434", "qwen3-vl:8b"))
    assert fixes == {"src/styles/app.css": "make the .topbar span the full width",
                     "src/components/Composer.jsx": "align the Send button with the textarea"}
    assert posted["url"].endswith("/api/chat") and posted["body"]["model"] == "qwen3-vl:8b"
    assert len(posted["body"]["messages"][0]["images"]) == 1 and "Chatbot" in posted["body"]["messages"][0]["content"]
    # PASS -> no fixes; non-UI brief -> no call
    _Req.post = staticmethod(lambda url, json=None, timeout=0: _Resp({"message": {"content": "PASS"}}))
    assert vision.critique_screenshots(brief, {"/": str(shot)}, ["src/styles/app.css"], None,
                                       model=("http://x", "m")) == {}
    assert vision.critique_screenshots(design.build_brief("a python cli", []), {"/": str(shot)}, [], None) == {}


def test_parse_fixes_shared_with_text_critic():
    from cgx.design.critic import parse_fixes
    assert parse_fixes("PASS", ["a.jsx"]) == {}
    assert parse_fixes("- `a.jsx`: add a page header\n- b.jsx: something else", ["src/a.jsx"]) == {"src/a.jsx": "add a page header"}
