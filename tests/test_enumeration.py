"""Tests for deterministic endpoint enumeration (cgx.answer.enumeration)."""

from __future__ import annotations

from cgx.answer.enumeration import (
    answer_endpoint_enumeration,
    collect_endpoints,
    extract_subject_terms,
    render_enumeration,
)


def _rec(cid, file, name, route):
    return {"id": cid, "file": file, "name": name, "route": route}


def _corpus():
    return [
        _rec("app/scanai/scanai.py::function::a", "app/scanai/scanai.py", "a",
             {"methods": ["GET"], "path": "/scan"}),
        _rec("app/scanai/scanai.py::function::b", "app/scanai/scanai.py", "b",
             {"methods": ["POST"], "path": "/scan/{id}"}),
        _rec("app/other/x.py::function::c", "app/other/x.py", "c",
             {"methods": ["GET"], "path": "/other"}),
        # Route-less function must never be counted.
        _rec("app/scanai/scanai.py::function::d", "app/scanai/scanai.py", "d", None),
    ]


def test_extract_subject_terms_drops_boilerplate():
    assert extract_subject_terms("how many api endpoints does scanai have?") == ["scanai"]
    assert extract_subject_terms("list all endpoints") == []
    assert "route" not in extract_subject_terms("list the routes in auth")


def test_collect_endpoints_filters_and_scopes():
    recs = _corpus()
    assert len(collect_endpoints(recs, ["scanai"])) == 2
    assert len(collect_endpoints(recs, [])) == 3
    # Route-less record is excluded regardless of subject.
    assert all(e["chunk_id"].endswith(("::a", "::b")) for e in collect_endpoints(recs, ["scanai"]))


def test_collect_endpoints_dedupes_identical_entries():
    dup = _rec("dup::function::a", "a.py", "a", {"methods": ["GET"], "path": "/x"})
    out = collect_endpoints([dup, dict(dup)], [])
    assert len(out) == 1


def test_collect_endpoints_sorted_by_path():
    recs = _corpus()
    paths = [e["path"] for e in collect_endpoints(recs, [])]
    assert paths == sorted(paths)


def test_render_enumeration_shape_and_citations():
    eps = collect_endpoints(_corpus(), ["scanai"])
    out = render_enumeration(eps, ["scanai"], scoped=True)
    assert out["debug"]["endpoint_count"] == 2
    assert out["confidence"] == 0.95
    assert out["answer_md"].startswith("**2 API endpoints** matching `scanai`")
    assert "GET /scan" in out["answer_md"]
    assert len(out["citations"]) == 2
    assert all("chunk_id" in c for c in out["citations"])


def test_render_enumeration_zero_endpoints():
    out = render_enumeration([], ["scanai"], scoped=True)
    assert out["debug"]["endpoint_count"] == 0
    assert "no API endpoints" in out["answer_md"]
    assert out["citations"] == []


def test_answer_scoped_then_fallback():
    recs = _corpus()
    scoped = answer_endpoint_enumeration(recs, "how many api endpoints does scanai have?")
    assert scoped["debug"]["endpoint_count"] == 2

    # Subject that matches nothing falls back to enumerating all endpoints.
    fallback = answer_endpoint_enumeration(recs, "how many api endpoints does zzz have?")
    assert fallback["debug"]["endpoint_count"] == 3


def test_answer_empty_index():
    out = answer_endpoint_enumeration([], "list all endpoints")
    assert out["debug"]["endpoint_count"] == 0
    assert "no API endpoints" in out["answer_md"]
