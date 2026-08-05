"""Tests for the hardware route: matrix merges installed models; hf_fit."""

from __future__ import annotations

from unittest.mock import patch

from cgx.webui.routes import hardware


_HW = {"ram_gb": 64.0, "gpu_vram_gb": 24.0}


def test_matrix_flags_installed_catalog_rows():
    installed = [{"name": "qwen2.5-coder:7b-instruct", "parameter_size": "7.6B"}]
    with patch.object(hardware.ollama_discovery, "detect_hardware", return_value=_HW), \
         patch.object(hardware.ollama_discovery, "list_installed_models", return_value=installed):
        resp = hardware.matrix()
    by_name = {r.model: r for r in resp.rows}
    assert by_name["qwen2.5-coder:7b-instruct"].installed is True
    # A catalogue model the user hasn't pulled is not flagged.
    assert by_name["qwen2.5-coder:1.5b"].installed is False


def test_matrix_appends_installed_only_models():
    installed = [{"name": "my-custom:8b", "parameter_size": "8.2B", "family": "llama"}]
    catalog_names = {"qwen2.5-coder:1.5b"}
    with patch.object(hardware.ollama_discovery, "detect_hardware", return_value=_HW), \
         patch.object(hardware.ollama_discovery, "list_installed_models", return_value=installed):
        resp = hardware.matrix()
    by_name = {r.model: r for r in resp.rows}
    assert "my-custom:8b" in by_name
    row = by_name["my-custom:8b"]
    assert row.installed is True
    assert row.params_b == 8.2
    assert row.notes == "installed locally"
    # Catalogue entries are still present alongside the appended row.
    assert catalog_names <= set(by_name)


def test_matrix_survives_unreachable_ollama():
    with patch.object(hardware.ollama_discovery, "detect_hardware", return_value=_HW), \
         patch.object(hardware.ollama_discovery, "list_installed_models", return_value=[]):
        resp = hardware.matrix()
    assert resp.rows
    assert all(r.installed is False for r in resp.rows)


def test_hf_fit_rejects_bad_repo():
    with patch.object(hardware.ollama_discovery, "detect_hardware", return_value=_HW):
        resp = hardware.hf_fit("not-a-valid-repo")
    assert resp.fit == "unknown"
    assert "invalid repo" in resp.reason


def test_hf_fit_uses_safetensors_total():
    spec = {"safetensors": {"total": 7_000_000_000}, "pipeline_tag": "text-generation"}
    with patch.object(hardware.ollama_discovery, "detect_hardware", return_value=_HW), \
         patch.object(hardware, "_hf_model_spec", return_value=spec):
        resp = hardware.hf_fit("Qwen/Qwen2.5-Coder-7B-Instruct")
    assert resp.params_source == "safetensors"
    assert resp.params_b == 7.0
    assert resp.fit == "fits"
    assert resp.pipeline_tag == "text-generation"


def test_hf_fit_falls_back_to_name_when_spec_missing():
    with patch.object(hardware.ollama_discovery, "detect_hardware", return_value=_HW), \
         patch.object(hardware, "_hf_model_spec", side_effect=RuntimeError("boom")):
        resp = hardware.hf_fit("someone/Model-14B-GGUF")
    assert resp.params_source == "name"
    assert resp.params_b == 14.0
    assert resp.fit in {"fits", "tight", "won't fit"}


def test_hf_fit_unknown_when_no_params_anywhere():
    with patch.object(hardware.ollama_discovery, "detect_hardware", return_value=_HW), \
         patch.object(hardware, "_hf_model_spec", return_value={}):
        resp = hardware.hf_fit("someone/mystery-model")
    assert resp.params_source == "unknown"
    assert resp.params_b == 0.0
    assert resp.fit == "unknown"
