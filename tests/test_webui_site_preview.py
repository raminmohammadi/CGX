"""Tests for the site-preview file server, the sites workspace, and the
path-containment / content-type-allowlist guards."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from cgx.webui import workspace
from cgx.webui.models import SiteCreateRequest
from cgx.webui.routes import agent_session as asroutes
from cgx.webui.routes import sites as sites_routes


# --- workspace / sites -----------------------------------------------------

@pytest.fixture()
def sites_dir(tmp_path, monkeypatch):
    d = tmp_path / "sites"
    monkeypatch.setattr(workspace, "SITES_DIR", d)
    return d


def test_slugify():
    assert workspace.slugify("My Cool Site!") == "my-cool-site"
    assert workspace.slugify("   ") == "site"
    assert workspace.slugify("A" * 100).startswith("a")


def test_create_and_list_site(sites_dir):
    info = sites_routes.create_site(SiteCreateRequest(name="Portfolio Site"))
    assert info.slug == "portfolio-site"
    assert info.project_root.endswith("portfolio-site")
    listed = sites_routes.list_sites()
    assert any(s.slug == "portfolio-site" for s in listed)


def test_create_site_requires_name(sites_dir):
    with pytest.raises(HTTPException) as exc:
        sites_routes.create_site(SiteCreateRequest(name="   "))
    assert exc.value.status_code == 400


def test_site_path_rejects_traversal(sites_dir):
    with pytest.raises(ValueError):
        workspace.site_path("../evil")


# --- preview serving -------------------------------------------------------

class _FakeSession:
    def __init__(self, root):
        self.project_root = str(root)


class _FakeStore:
    def __init__(self, session):
        self._s = session

    def get_session(self, sid):
        return self._s


class _FakeRunner:
    def __init__(self, session):
        self.store = _FakeStore(session)


@pytest.fixture()
def served_site(tmp_path, monkeypatch):
    root = tmp_path / "site"
    root.mkdir()
    (root / "index.html").write_text("<h1>Home</h1>", encoding="utf-8")
    (root / "style.css").write_text("body{}", encoding="utf-8")
    (root / "secret.env").write_text("TOKEN=abc", encoding="utf-8")
    session = _FakeSession(root)
    monkeypatch.setattr(asroutes, "_resolve_runner_for",
                        lambda sid: _FakeRunner(session))
    return root


def test_preview_serves_index(served_site):
    resp = asroutes.preview_index("sid1")
    assert resp.status_code == 200
    assert resp.path == str(served_site / "index.html")


def test_preview_serves_css_with_type(served_site):
    resp = asroutes.preview_file("sid1", "style.css")
    assert resp.media_type == "text/css"


def test_preview_blocks_non_web_extension(served_site):
    # A secret .env must never be readable through the preview endpoint.
    with pytest.raises(HTTPException) as exc:
        asroutes.preview_file("sid1", "secret.env")
    assert exc.value.status_code == 404


def test_preview_blocks_path_traversal(served_site):
    with pytest.raises(HTTPException) as exc:
        asroutes.preview_file("sid1", "../../../../etc/passwd")
    assert exc.value.status_code in (403, 404)


def test_preview_404_for_missing_file(served_site):
    with pytest.raises(HTTPException) as exc:
        asroutes.preview_file("sid1", "nope.html")
    assert exc.value.status_code == 404


def test_preview_requires_project_root(monkeypatch):
    session = _FakeSession("")
    monkeypatch.setattr(asroutes, "_resolve_runner_for",
                        lambda sid: _FakeRunner(session))
    with pytest.raises(HTTPException) as exc:
        asroutes.preview_index("sid1")
    assert exc.value.status_code == 400


def test_preview_404_for_unknown_session(monkeypatch):
    monkeypatch.setattr(asroutes, "_resolve_runner_for",
                        lambda sid: _FakeRunner(None))
    with pytest.raises(HTTPException) as exc:
        asroutes.preview_index("ghost")
    assert exc.value.status_code == 404
