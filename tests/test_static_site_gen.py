"""Tests for the Site Studio one-shot generator + the generate/revise/preview
routes (with a fake provider, no real model)."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from cgx.answer.static_site_gen import generate_site
from cgx.webui import workspace
from cgx.webui.models import (
    ProviderConfig, SiteGenerateRequest, SiteReviseRequest,
)
from cgx.webui.routes import sites as sroutes


_HTML = "<!DOCTYPE html>\n<html lang=\"en\"><head><title>T</title></head><body><h1>Hi</h1></body></html>"


class _FakeProvider:
    def __init__(self, out: str):
        self.out = out
        self.calls = []

    def chat(self, *, messages, temperature=0.2, force_json=False, max_tokens=None):
        self.calls.append({"messages": messages, "max_tokens": max_tokens})
        return {"content": self.out}


# --- generator -------------------------------------------------------------

def test_generate_returns_clean_html():
    html = generate_site("a bakery site", _FakeProvider(_HTML), flavor="modern")
    assert html.startswith("<!DOCTYPE html>")
    assert "<h1>Hi</h1>" in html


def test_generate_strips_markdown_fences_and_prose():
    fenced = "Sure! Here it is:\n```html\n" + _HTML + "\n```"
    html = generate_site("x", _FakeProvider(fenced))
    assert html.startswith("<!DOCTYPE html>")
    assert "```" not in html and "Sure!" not in html


def test_generate_rejects_non_html():
    with pytest.raises(ValueError):
        generate_site("x", _FakeProvider("I cannot do that."))


def test_revise_mode_passes_current_and_feedback():
    fp = _FakeProvider(_HTML)
    generate_site("", fp, current_html="<html>old</html>", feedback="make it blue")
    user_msg = fp.calls[0]["messages"][1]["content"]
    assert "old" in user_msg and "make it blue" in user_msg


def test_generate_requests_large_token_budget_via_route(monkeypatch, tmp_path):
    # The route must not let the default 1024 num_predict truncate a page.
    monkeypatch.setattr(workspace, "SITES_DIR", tmp_path / "sites")
    captured = {}

    def fake_build_provider(**kw):
        captured.update(kw)
        return _FakeProvider(_HTML)

    monkeypatch.setattr(sroutes, "build_provider", fake_build_provider)
    sroutes.generate(SiteGenerateRequest(
        name="Toast", brief="a bakery", flavor="modern",
        provider=ProviderConfig(num_predict=1024)))
    assert captured["num_predict"] >= 8000


# --- routes ----------------------------------------------------------------

@pytest.fixture()
def sites_dir(tmp_path, monkeypatch):
    d = tmp_path / "sites"
    monkeypatch.setattr(workspace, "SITES_DIR", d)
    monkeypatch.setattr(sroutes, "_provider", lambda pc: _FakeProvider(_HTML))
    return d


def test_generate_route_writes_index(sites_dir):
    res = sroutes.generate(SiteGenerateRequest(
        name="Coffee Shop", brief="cozy coffee shop", flavor="modern",
        provider=ProviderConfig()))
    assert res.slug == "coffee-shop"
    idx = sites_dir / "coffee-shop" / "index.html"
    assert idx.is_file() and "<h1>Hi</h1>" in idx.read_text()
    assert res.bytes > 0


def test_generate_requires_name_and_brief(sites_dir):
    with pytest.raises(HTTPException) as exc:
        sroutes.generate(SiteGenerateRequest(name="", brief="x",
                                             provider=ProviderConfig()))
    assert exc.value.status_code == 400


def test_revise_route_updates_index(sites_dir):
    sroutes.generate(SiteGenerateRequest(name="Shop", brief="shop",
                                         provider=ProviderConfig()))
    monkeypatch_out = "<!DOCTYPE html><html><body><h1>Revised</h1></body></html>"
    sroutes._provider = lambda pc: _FakeProvider(monkeypatch_out)  # type: ignore
    res = sroutes.revise(SiteReviseRequest(slug="shop", feedback="change it",
                                           provider=ProviderConfig()))
    idx = sites_dir / "shop" / "index.html"
    assert "Revised" in idx.read_text()
    assert res.slug == "shop"


def test_revise_404_when_no_site(sites_dir):
    with pytest.raises(HTTPException) as exc:
        sroutes.revise(SiteReviseRequest(slug="ghost", feedback="x",
                                         provider=ProviderConfig()))
    assert exc.value.status_code == 404


def test_preview_serves_index(sites_dir):
    sroutes.generate(SiteGenerateRequest(name="Prev", brief="p",
                                         provider=ProviderConfig()))
    resp = sroutes.preview("prev", "")
    assert resp.status_code == 200


def test_preview_blocks_traversal(sites_dir):
    sroutes.generate(SiteGenerateRequest(name="Prev2", brief="p",
                                         provider=ProviderConfig()))
    with pytest.raises(HTTPException) as exc:
        sroutes.preview("prev2", "../../../../etc/passwd")
    assert exc.value.status_code in (403, 404)
