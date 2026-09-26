"""Route-handler tests for ``/api/context-file`` (CGX.md editor)."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from cgx import context_files as cf
from cgx.webui.models import ContextFileWriteRequest
from cgx.webui.routes import context_file as routes


@pytest.fixture(autouse=True)
def _clear():
    cf.clear_cache()
    yield
    cf.clear_cache()


def test_get_when_absent_reports_default_path(tmp_path):
    out = routes.get_context_file(str(tmp_path))
    assert out["exists"] is False
    assert out["filename"] == "CGX.md"
    assert out["content"] == ""


def test_write_then_get_roundtrip(tmp_path):
    routes.write_context_file(
        ContextFileWriteRequest(project_root=str(tmp_path), content="House rules."))
    assert (tmp_path / "CGX.md").read_text() == "House rules."
    out = routes.get_context_file(str(tmp_path))
    assert out["exists"] is True
    assert out["content"] == "House rules."


def test_write_updates_existing_agent_md_in_place(tmp_path):
    (tmp_path / "AGENT.md").write_text("old", encoding="utf-8")
    routes.write_context_file(
        ContextFileWriteRequest(project_root=str(tmp_path), content="new"))
    # Should overwrite the existing AGENT.md, not create a second CGX.md.
    assert (tmp_path / "AGENT.md").read_text() == "new"
    assert not (tmp_path / "CGX.md").exists()


def test_write_requires_project_root():
    with pytest.raises(HTTPException) as exc:
        routes.write_context_file(ContextFileWriteRequest(project_root="", content="x"))
    assert exc.value.status_code == 400


def test_delete(tmp_path):
    routes.write_context_file(
        ContextFileWriteRequest(project_root=str(tmp_path), content="x"))
    out = routes.delete_context_file(str(tmp_path))
    assert "deleted" in out
    assert not (tmp_path / "CGX.md").exists()


def test_delete_404_when_absent(tmp_path):
    with pytest.raises(HTTPException) as exc:
        routes.delete_context_file(str(tmp_path))
    assert exc.value.status_code == 404
