"""Tests for the hardware-aware embedding budget (cgx.embeddings.autotune).

The regression these guard against: indexing a large repo with the local torch
embedder froze laptops because (batch=64, max_length=8192) needs ~206 GB of
O(L^2) attention. The tuner must never hand back a memory-catastrophic pair,
must scale to the machine, and must honour explicit env overrides.
"""

from __future__ import annotations

import pytest

from cgx.embeddings.autotune import resolve_embed_budget, _N_HEADS


def _attn_peak_gb(batch: int, max_length: int) -> float:
    return batch * _N_HEADS * (max_length ** 2) * 4 / (1024 ** 3)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv("CGX_EMBED_MAXLEN", raising=False)
    monkeypatch.delenv("CGX_EMBED_BATCH", raising=False)


_MACHINES = [
    ("m5_unified_48", {"ram_gb": 48.0, "gpu_vram_gb": 48.0, "is_unified_memory": True, "torch_mps_available": True}),
    ("air_unified_8", {"ram_gb": 8.0, "gpu_vram_gb": 8.0, "is_unified_memory": True, "torch_mps_available": True}),
    ("cpu_16", {"ram_gb": 16.0, "gpu_vram_gb": 0.0}),
    ("cuda_8_of_32", {"ram_gb": 32.0, "gpu_vram_gb": 8.0, "torch_cuda_available": True, "is_unified_memory": False}),
    ("a100_80_of_256", {"ram_gb": 256.0, "gpu_vram_gb": 80.0, "torch_cuda_available": True, "is_unified_memory": False}),
    ("unknown", {}),
]


@pytest.mark.parametrize("name,hw", _MACHINES)
def test_budget_never_memory_catastrophic(name, hw):
    b = resolve_embed_budget(hw)
    # Attention peak must stay small -- the whole point. (Old crash was ~192 GB.)
    peak = _attn_peak_gb(b["batch_size"], b["max_length"])
    assert peak <= 4.0, f"{name}: attention peak {peak:.1f} GB too high"
    # Sane, positive, capped knobs.
    assert 1 <= b["batch_size"] <= 32
    assert 256 <= b["max_length"] <= 2048  # auto never exceeds the 2048 cap


def test_bigger_memory_gets_at_least_as_much_headroom():
    small = resolve_embed_budget(_MACHINES[1][1])   # 8 GB unified
    big = resolve_embed_budget(_MACHINES[0][1])      # 48 GB unified
    # More memory -> not a smaller max_length (monotonic-ish scaling).
    assert big["max_length"] >= small["max_length"]


def test_cuda_sizes_to_vram_not_system_ram():
    # 8 GB VRAM on a 256 GB box must be bounded by VRAM, not RAM.
    tiny_vram = resolve_embed_budget(
        {"ram_gb": 256.0, "gpu_vram_gb": 8.0, "torch_cuda_available": True, "is_unified_memory": False})
    assert tiny_vram["max_length"] <= 1024
    assert _attn_peak_gb(tiny_vram["batch_size"], tiny_vram["max_length"]) <= 1.5


def test_env_overrides_win(monkeypatch):
    monkeypatch.setenv("CGX_EMBED_MAXLEN", "512")
    monkeypatch.setenv("CGX_EMBED_BATCH", "4")
    b = resolve_embed_budget({"ram_gb": 48.0, "is_unified_memory": True})
    assert b["max_length"] == 512 and b["batch_size"] == 4
    assert "maxlen=env" in b["basis"] and "batch=env" in b["basis"]


def test_pinned_maxlen_rederives_safe_batch(monkeypatch):
    # A user forcing the full 8192 window must still get a memory-safe batch.
    monkeypatch.setenv("CGX_EMBED_MAXLEN", "8192")
    b = resolve_embed_budget({"ram_gb": 48.0, "is_unified_memory": True})
    assert b["max_length"] == 8192
    assert _attn_peak_gb(b["batch_size"], b["max_length"]) <= 4.0


def test_detection_failure_falls_back_safely(monkeypatch):
    # If hardware detection raises, resolve must not crash the build.
    monkeypatch.setattr(
        "cgx.answer.ollama_discovery.detect_hardware",
        lambda: (_ for _ in ()).throw(RuntimeError("no probe")),
    )
    b = resolve_embed_budget()  # hardware=None -> triggers detection
    assert b["batch_size"] >= 1 and b["max_length"] >= 256
    assert _attn_peak_gb(b["batch_size"], b["max_length"]) <= 4.0
