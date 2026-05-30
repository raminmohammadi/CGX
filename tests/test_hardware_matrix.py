"""Tests for the offline hardware / trade-off matrix."""

from __future__ import annotations

from cgx.answer.hardware_matrix import (
    LOCAL_MODEL_CATALOG,
    TRADEOFFS,
    compute_local_fit,
    tradeoffs_rows,
)


def test_catalog_is_non_empty_and_well_typed():
    assert len(LOCAL_MODEL_CATALOG) >= 4
    required = {"name", "params_b", "min_ram_gb", "recommended_vram_gb",
                "ctx_window", "family", "notes"}
    for entry in LOCAL_MODEL_CATALOG:
        assert required <= set(entry), f"missing fields in {entry}"
        assert isinstance(entry["params_b"], (int, float)) and entry["params_b"] > 0
        assert isinstance(entry["min_ram_gb"], (int, float)) and entry["min_ram_gb"] > 0
        assert entry["family"] in {"coder", "general"}


def test_compute_local_fit_unknown_hardware_marks_all_unknown():
    rows = compute_local_fit({})
    assert rows, "expected at least one row"
    assert all(r["fit"].startswith("❓") for r in rows)


def test_compute_local_fit_huge_machine_fits_everything():
    rows = compute_local_fit({"ram_gb": 256.0, "gpu_vram_gb": 80.0})
    assert all(r["fit"].startswith("✅") for r in rows), \
        f"expected all fits, got {[r['fit'] for r in rows]}"


def test_compute_local_fit_tiny_machine_rejects_large_models():
    # 8 GB RAM, no GPU: should reject 14B, accept 1.5B comfortably.
    rows = compute_local_fit({"ram_gb": 8.0, "gpu_vram_gb": 0.0})
    by_name = {r["model"]: r for r in rows}
    assert "qwen2.5-coder:14b-instruct" in by_name
    assert by_name["qwen2.5-coder:14b-instruct"]["fit"].startswith("❌")
    assert by_name["qwen2.5-coder:1.5b"]["fit"].startswith("✅")


def test_compute_local_fit_tight_vram_flagged_as_tight():
    # 32 GB RAM (plenty) but a tiny 2 GB GPU — the 7B coder needs 8 GB VRAM.
    rows = compute_local_fit({"ram_gb": 32.0, "gpu_vram_gb": 2.0})
    by_name = {r["model"]: r for r in rows}
    assert by_name["qwen2.5-coder:7b-instruct"]["fit"].startswith("⚠️")


def test_compute_local_fit_rows_sorted_by_params():
    rows = compute_local_fit({"ram_gb": 64.0})
    params = [r["params_b"] for r in rows]
    assert params == sorted(params)


def test_compute_local_fit_row_shape_matches_catalog():
    # One annotated row per catalog entry; no duplicates / drops.
    rows = compute_local_fit({"ram_gb": 16.0})
    assert len(rows) == len(LOCAL_MODEL_CATALOG)
    assert {r["model"] for r in rows} == {e["name"] for e in LOCAL_MODEL_CATALOG}


def test_tradeoffs_rows_are_well_formed():
    rows = tradeoffs_rows()
    assert rows == TRADEOFFS  # value-equal but independent list
    assert rows is not TRADEOFFS  # defensive copy
    for r in rows:
        assert set(r) == {"dimension", "local", "cloud", "winner"}
        assert r["winner"] in {"local", "cloud", "tie"}


def test_tradeoffs_cover_privacy_and_cost():
    dims = {r["dimension"].lower() for r in tradeoffs_rows()}
    assert any("privacy" in d for d in dims)
    assert any("cost" in d for d in dims)
