"""Phase 9 -- reranker profile policy.

Covers:

* ``default_reranker_for_kind`` returns the correct per-kind default
  (cloud kinds opt in; local / custom kinds opt out).
* ``resolve_enable_reranker`` honours an explicit boolean and otherwise
  falls back to the kind-derived default.
* ``Profile.enable_reranker`` round-trips through ``save_profile`` /
  ``list_profiles`` for ``True``, ``False``, and ``None`` (None means
  "auto" and must not be persisted as a literal value).
* ``hybrid_retrieve_two_view`` threads the flag into the internal
  ``HybridConfig``: when disabled the order is deterministic RRF;
  when enabled the cross-encoder reorders the head.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Tuple

import pytest

from cgx.retrieval import orchestrator as orch_mod
from cgx.retrieval import reranker as reranker_mod
from cgx.retrieval.orchestrator import hybrid_retrieve_two_view


# --------------------------------------------------------------------------- #
# Profile field + policy helpers
# --------------------------------------------------------------------------- #


@pytest.fixture()
def profiles_module(tmp_path, monkeypatch):
    monkeypatch.setenv("CGX_CONFIG_DIR", str(tmp_path))
    import cgx.answer.profiles as profiles
    importlib.reload(profiles)
    monkeypatch.setattr(profiles, "_keyring", lambda: None)
    return profiles


@pytest.mark.parametrize("kind,expected", [
    ("ollama", False),
    ("custom", False),
    ("openai-compat", True),
    ("gemini", True),
    ("OpenAI-Compat", True),   # case-insensitive
    ("", False),
])
def test_default_reranker_for_kind(profiles_module, kind, expected):
    assert profiles_module.default_reranker_for_kind(kind) is expected


def test_resolve_enable_reranker_explicit_wins(profiles_module):
    P = profiles_module
    # Cloud kind, explicit False → False.
    prof = P.Profile(name="x", kind="gemini", model="m", base_url="u",
                     enable_reranker=False)
    assert P.resolve_enable_reranker(prof) is False
    # Local kind, explicit True → True.
    prof2 = P.Profile(name="y", kind="ollama", model="m", base_url="u",
                      enable_reranker=True)
    assert P.resolve_enable_reranker(prof2) is True


def test_resolve_enable_reranker_falls_back_to_kind(profiles_module):
    P = profiles_module
    prof = P.Profile(name="x", kind="ollama", model="m", base_url="u")
    assert P.resolve_enable_reranker(prof) is False
    prof2 = P.Profile(name="y", kind="openai-compat", model="m", base_url="u")
    assert P.resolve_enable_reranker(prof2) is True


def test_profile_round_trip_persists_enable_reranker(profiles_module):
    P = profiles_module
    # Explicit True is persisted.
    P.save_profile(P.Profile(name="on", kind="ollama", model="m",
                             base_url="u", enable_reranker=True))
    # Explicit False is persisted.
    P.save_profile(P.Profile(name="off", kind="gemini", model="m",
                             base_url="u", enable_reranker=False))
    # None ("auto") is not persisted as a literal value.
    P.save_profile(P.Profile(name="auto", kind="gemini", model="m",
                             base_url="u", enable_reranker=None))

    by_name = {p.name: p for p in P.list_profiles()}
    assert by_name["on"].enable_reranker is True
    assert by_name["off"].enable_reranker is False
    assert by_name["auto"].enable_reranker is None
    # The kind-derived default still drives the resolved flag.
    assert P.resolve_enable_reranker(by_name["auto"]) is True


# --------------------------------------------------------------------------- #
# Threading: hybrid_retrieve_two_view → HybridConfig
# --------------------------------------------------------------------------- #


class _RecordingRetriever:
    """Captures the ``cfg`` HybridRetriever.search receives."""

    captured: List[Any] = []

    def __init__(self, *a, **kw) -> None:
        pass

    def search(self, query: str, *, embedder: Any, cfg: Any = None) -> Dict[str, Any]:
        _RecordingRetriever.captured.append(cfg)
        return {"hits": [], "top_files": [], "top_classes": [], "anchors": []}


@pytest.fixture()
def patched_retriever(monkeypatch):
    _RecordingRetriever.captured = []
    monkeypatch.setattr(orch_mod, "HybridRetriever", _RecordingRetriever)
    # Stub the TwoViewIndex wrapper so we don't need a real index on disk.
    monkeypatch.setattr(orch_mod, "_two_view_index_from_dict",
                        lambda indices, records=None, embedder=None: object())
    yield _RecordingRetriever


@pytest.mark.parametrize("enable,expected", [
    (None, False),    # default: off
    (False, False),
    (True, True),
])
def test_hybrid_retrieve_threads_enable_reranker(patched_retriever, enable, expected):
    out = hybrid_retrieve_two_view(
        "q",
        indices={"views": {}},
        records=[],
        embedder=object(),
        chunks=None,
        G=None,
        top_k_per_view=10,
        enable_reranker=enable,
    )
    assert out == {"hits": [], "top_files": [], "top_classes": [], "anchors": []}
    cfg = patched_retriever.captured[-1]
    assert cfg.enable_reranker is expected


def test_hybrid_retrieve_threads_reranker_knobs(patched_retriever):
    hybrid_retrieve_two_view(
        "q", indices={"views": {}}, records=[], embedder=object(),
        chunks=None, G=None, top_k_per_view=10,
        enable_reranker=True,
        reranker_model="cross-encoder/custom",
        reranker_top_n=7,
        reranker_weight=0.25,
    )
    cfg = patched_retriever.captured[-1]
    assert cfg.enable_reranker is True
    assert cfg.reranker_model == "cross-encoder/custom"
    assert cfg.reranker_top_n == 7
    assert cfg.reranker_weight == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# End-to-end: deterministic order vs reranked order
# --------------------------------------------------------------------------- #


class _FakeView:
    def __init__(self, hits_per_view: Dict[str, List[Dict[str, Any]]]) -> None:
        self._hits = hits_per_view

    def available_views(self) -> List[str]:
        return list(self._hits.keys())

    def search_view(self, view: str, query: str, *, embedder: Any, top_k: int):
        return self._hits[view][:top_k]


def _records(*cids: str) -> List[Dict[str, Any]]:
    return [
        {"id": cid, "name": cid.split("::")[-1], "file": cid.split("::")[0],
         "code": f"def {cid.split('::')[-1]}(): pass"}
        for cid in cids
    ]


def _run_with_reranker(monkeypatch, enable: bool) -> List[str]:
    """Run retrieval with the same fixture and toggle just ``enable_reranker``."""
    cids = ("pkg/a.py::function::alpha", "pkg/b.py::function::beta")
    records = _records(*cids)
    hits = {
        "intent": [{"chunk_id": cids[0], "rank": 1, "score": 0.9},
                   {"chunk_id": cids[1], "rank": 2, "score": 0.5}],
        "impl":   [{"chunk_id": cids[0], "rank": 1, "score": 0.9}],
    }

    class _FakeCE:
        # Bias beta above alpha when reranking runs.
        def predict(self, pairs):
            return [1.0 if "beta" in p[1] else 0.0 for p in pairs]

    monkeypatch.setattr(reranker_mod, "get_default_reranker",
                        lambda *a, **kw: _FakeCE())
    r = orch_mod.HybridRetriever(
        tv_index=_FakeView(hits), records=records,
        lexical_index=None,
        chunks=[{"id": rec["id"], "code": rec["code"],
                 "name": rec["name"], "file": rec["file"]} for rec in records],
        G=None,
    )
    cfg = orch_mod.HybridConfig(
        k_intent=5, k_impl=5, k_lex=5, top_k_chunks=10,
        graph_bonus=0.0, symbol_boost=0.0,
        enable_reranker=enable, reranker_top_n=5, reranker_weight=1.0,
    )
    out = r.search("anything", embedder=None, cfg=cfg)
    return [h["chunk_id"] for h in out["hits"]]


def test_deterministic_order_when_reranker_disabled(monkeypatch):
    order_off = _run_with_reranker(monkeypatch, enable=False)
    # alpha leads in every signal → RRF must keep it first, deterministically.
    assert order_off[0].endswith("::alpha")
    # Repeat to confirm stability.
    assert _run_with_reranker(monkeypatch, enable=False) == order_off


def test_reranker_reorders_head_when_enabled(monkeypatch):
    order_on = _run_with_reranker(monkeypatch, enable=True)
    # The fake CE biases beta above alpha post-rerank.
    assert order_on[0].endswith("::beta")
