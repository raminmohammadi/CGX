# Hardware compute matrix

This document explains the static catalogue + trade-off table that
backs the **📊 Hardware** tab in the CGX UI. Source of truth:

- `src/cgx/answer/hardware_matrix.py` — the Python module
- `docs/hardware_matrix.json` — same data, exported for tooling

Numbers are deliberate approximations for **4-bit quantised
GGUF / AWQ-style inference** (the format Ollama serves by default).
They are intended for UI sorting and a "will this fit?" sanity check,
not capacity planning.

## Local-model catalogue

The catalogue lists 8 locally-runnable models across two families
(`coder`, `general`):

| Model                              | Params (B) | Min RAM (GB) | Rec. VRAM (GB) | Ctx window | Family   | Notes                                          |
|------------------------------------|-----------:|-------------:|---------------:|-----------:|----------|------------------------------------------------|
| `qwen2.5-coder:1.5b`               |        1.5 |          4.0 |            2.0 |     32 768 | coder    | smallest viable coder; CPU-friendly            |
| `qwen2.5-coder:3b`                 |        3.0 |          6.0 |            4.0 |     32 768 | coder    | balanced default for code Q&A                  |
| `qwen2.5-coder:7b-instruct`        |        7.0 |         10.0 |            8.0 |     32 768 | coder    | higher-quality coder; sweet spot on 16 GB GPUs |
| `qwen2.5-coder:14b-instruct`       |       14.0 |         20.0 |           16.0 |     32 768 | coder    | near-cloud coder quality; needs ≥16 GB VRAM    |
| `llama3.2:3b-instruct`             |        3.0 |          6.0 |            4.0 |    131 072 | general  | long context, light general-purpose            |
| `llama3.1:8b-instruct`             |        8.0 |         12.0 |            8.0 |    131 072 | general  | general-purpose with strong reasoning          |
| `qwen2.5:7b-instruct`              |        7.0 |         10.0 |            8.0 |     32 768 | general  | general-purpose alternative to llama 8b        |
| `phi3.5:3.8b-mini-instruct`        |        3.8 |          6.0 |            4.0 |    131 072 | general  | small, long-context, low-RAM                   |

### Fit verdict

`compute_local_fit(hw)` computes an **effective budget** in GB and
classifies each entry:

```text
effective_budget = max(ram_gb, gpu_vram_gb * 2.0)   when a GPU is present
effective_budget = ram_gb                            otherwise
```

| Symbol | Condition                                                                                       |
|--------|-------------------------------------------------------------------------------------------------|
| ❓     | `effective_budget == 0` (probe returned nothing — UI hasn't run *Detect hardware* yet).         |
| ❌     | `effective_budget < min_ram_gb * 0.9`. The model won't fit; not even tight.                     |
| ⚠️     | GPU present but `gpu_vram_gb < recommended_vram_gb * 0.75`, **or** budget within 1.2× min RAM. |
| ✅     | Budget ≥ 1.2× min RAM **and** GPU VRAM (if any) meets ≥75% of the recommendation.              |

The `reason` column on each row reports the exact comparison behind
the verdict so you can sanity-check the model against your own
machine without trusting the UI's symbol.

### Adding or tweaking a model

Edit `LOCAL_MODEL_CATALOG` in `src/cgx/answer/hardware_matrix.py`,
keep the field schema, and run:

```bash
PYTHONPATH=$PWD/src python -c "
import json
from cgx.answer.hardware_matrix import LOCAL_MODEL_CATALOG, TRADEOFFS
with open('docs/hardware_matrix.json', 'w') as f:
    json.dump({'local_model_catalog': LOCAL_MODEL_CATALOG,
               'tradeoffs': TRADEOFFS}, f, indent=2)
"
```

The pytest suite (`tests/test_hardware_matrix.py`) asserts:

- All entries have the required fields.
- Rows are sorted by `params_b`.
- The 14B coder is rejected on an 8 GB-RAM CPU-only machine.
- The 7B coder is flagged as tight on 32 GB RAM + 2 GB VRAM.
- All entries fit on a 256 GB / 80 GB VRAM workstation.

## Local vs cloud trade-offs

The `TRADEOFFS` table is intentionally editorial — short opinionated
strings about each axis. `winner ∈ {local, cloud, tie}`.

| Dimension                     | Local                                                               | Cloud                                                                                          | Winner |
|-------------------------------|---------------------------------------------------------------------|------------------------------------------------------------------------------------------------|--------|
| Privacy / data egress         | Prompts + code never leave the machine.                             | Prompts + retrieved snippets go to the provider; subject to their data policy.                 | local  |
| Marginal cost / token         | Electricity only; zero per-call cost once the model is downloaded.  | Pay-per-token; cost scales linearly with usage and context length.                             | local  |
| Quality ceiling               | Capped by what fits on your hardware (≈14B params on a 16 GB GPU).  | Access to frontier models (100B+ params, long context, tool-use).                              | cloud  |
| Latency (cold)                | First token after model load (seconds on small, minutes on large).  | Sub-second TTFT in steady state; spikes during provider load.                                  | tie    |
| Latency (warm)                | Predictable; bound by local GPU/CPU.                                | Variable; subject to rate limits + network round-trip.                                         | local  |
| Offline use                   | Works on a plane / air-gapped network.                              | Requires connectivity.                                                                         | local  |
| Setup effort                  | Install Ollama, pull a model (~GB-scale download).                  | Sign up, mint an API key, paste into a profile.                                                | cloud  |
| Operational risk              | Your machine = your SLO.                                            | Vendor outages / price changes / model deprecations.                                           | local  |

## Caveats

- The catalogue numbers describe **4-bit quantised** inference. Full
  FP16 weights require roughly 2.5–3× more memory.
- VRAM budget is doubled when projecting onto the system-RAM budget
  (`max(ram, vram * 2)`) because partial offload typically works once
  ~half the model fits on the GPU. This is a heuristic, not a
  guarantee — extremely small system RAM will still slow you down via
  KV-cache pressure.
- The trade-off table is editorial. If your privacy posture allows
  cloud (or your hardware can host a 70B model), the verdict for a
  given dimension may legitimately flip.

## Programmatic access

```python
from cgx.answer.hardware_matrix import (
    LOCAL_MODEL_CATALOG,
    compute_local_fit,
    tradeoffs_rows,
)
from cgx.answer.ollama_discovery import detect_hardware

hw = detect_hardware()                # {'ram_gb': ..., 'gpu_vram_gb': ...}
rows = compute_local_fit(hw)          # [{ ... fit: '✅ fits', reason: ... }, ...]
fits = [r for r in rows if r["fit"].startswith("✅")]

print("Recommended models:")
for r in fits:
    print(f"  - {r['model']:35s}  ({r['family']}, {r['params_b']}B)")

print("\nLocal-vs-cloud:")
for t in tradeoffs_rows():
    print(f"  {t['dimension']:30s}  -> {t['winner']}")
```
