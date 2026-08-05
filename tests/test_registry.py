"""Tests for cgx.registry (prompt versions, run ids, index lineage)."""

from __future__ import annotations

import json

from cgx import registry


def test_fingerprint_is_stable_and_content_addressed():
    a = registry.fingerprint("hello world")
    assert a == registry.fingerprint("hello world")  # deterministic
    assert a != registry.fingerprint("hello world!")  # content-sensitive
    assert len(a) == 12
    assert len(registry.fingerprint("x", length=8)) == 8
    # Empty/None normalise to the same stable hash rather than raising.
    assert registry.fingerprint("") == registry.fingerprint(None)  # type: ignore[arg-type]


def test_new_run_id_prefixed_and_unique():
    a, b = registry.new_run_id(), registry.new_run_id()
    assert a.startswith("run_") and b.startswith("run_")
    assert a != b


def test_prompt_registry_versions_by_content():
    reg = registry.PromptRegistry()
    v1 = reg.register("greet", "hello")
    assert reg.version_of("greet") == v1 == registry.fingerprint("hello")
    # Re-register with new text -> new version (last-writer-wins by content).
    v2 = reg.register("greet", "hello there")
    assert v2 != v1
    assert reg.version_of("missing") is None
    man = reg.manifest()
    assert man["greet"]["version"] == v2
    assert man["greet"]["chars"] == len("hello there")


def test_register_known_prompts_is_idempotent_and_nonempty():
    reg1 = registry.register_known_prompts()
    reg2 = registry.register_known_prompts()
    assert reg1 is reg2  # cached, loaded once
    man = reg1.manifest()
    # The engine defines intent-conditioned ask prompts; at least one should
    # have been registered under the ``ask:`` namespace.
    assert any(name.startswith("ask:") for name in man)


def test_cgx_version_is_a_string():
    assert isinstance(registry.cgx_version(), str)
    assert registry.cgx_version()  # non-empty


def test_git_revision_none_for_non_repo(tmp_path):
    assert registry.git_revision(str(tmp_path)) is None
    assert registry.git_revision(None) is None


def test_build_index_lineage_shape():
    lin = registry.build_index_lineage(
        project_root=None, embed_model="m", embed_dim=768,
        index_type="flat", metric="cosine")
    assert lin["index_id"].startswith("idx_")
    assert lin["cgx_version"]
    assert lin["git_revision"] is None
    assert lin["embed_model"] == "m"
    assert lin["embed_dim"] == 768
    assert lin["index_type"] == "flat"
    assert lin["metric"] == "cosine"


def test_lineage_round_trips_through_save_and_load_indices(tmp_path):
    from cgx.io.persist import (
        clear_indices_cache, load_indices, save_indices,
    )

    clear_indices_cache()
    indices = {
        "metric": "cosine",
        "lineage": registry.build_index_lineage(
            embed_model="BAAI/bge-m3", index_type="flat", metric="cosine"),
        "views": {
            "intent": {"rows": [{"chunk_id": "a"}], "index": None},
            "impl": {"rows": [], "index": None},
        },
    }
    idx_id = indices["lineage"]["index_id"]
    out_dir = tmp_path / "idx"
    save_indices(indices, str(out_dir))
    meta = json.loads((out_dir / "meta.json").read_text())
    assert meta["lineage"]["index_id"] == idx_id
    assert meta["lineage"]["cgx_version"]

    loaded = load_indices(str(out_dir))
    assert loaded["lineage"]["index_id"] == idx_id
