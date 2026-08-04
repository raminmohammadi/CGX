# Hardware compute matrix

This document explains the static catalogue + trade-off table that
backs the **📊 Hardware** tab in the CGX UI. Source of truth:

- `src/cgx/answer/hardware_matrix.py` -- the Python module
- `docs/hardware_matrix.json` -- same data, exported for tooling

For how to read the verdicts in the UI, see
[usage § Hardware-aware model picker](usage.md#9-hardware-aware-model-picker-hardware-tab).

Numbers are deliberate approximations for **4-bit quantised
GGUF / AWQ-style inference** (the format Ollama serves by default).
They are intended for UI sorting and a "will this fit?" sanity check,
not capacity planning.

<details>
<summary>

## Local-model catalogue
</summary>

The catalogue lists 21 locally-runnable models across three families
(`coder`, `general`, `reasoning`). Rows are grouped by family and, within
each family, ascend by `params_b` -- the exact order `compute_local_fit`
returns and the Hardware tab renders:

| Model | Params (B) | Min RAM (GB) | Rec. VRAM (GB) | Ctx window | Family | Notes |
|-------|-----------:|-------------:|---------------:|-----------:|--------|-------|
| `qwen2.5-coder:1.5b` | 1.5 | 4.0 | 2.0 | 32 768 | coder | smallest viable coder; CPU-friendly |
| `qwen2.5-coder:3b` | 3.0 | 6.0 | 4.0 | 32 768 | coder | balanced default for code Q&A |
| `deepseek-coder:6.7b` | 6.7 | 10.0 | 8.0 | 16 384 | coder | strong FIM; good code completion on 8 GB GPU |
| `qwen2.5-coder:7b-instruct` | 7.0 | 10.0 | 8.0 | 32 768 | coder | higher-quality coder; sweet spot on 16 GB GPUs |
| `qwen2.5-coder:14b-instruct` | 14.0 | 20.0 | 16.0 | 32 768 | coder | near-cloud coder quality; needs ≥16 GB VRAM |
| `deepseek-coder-v2:16b` | 16.0 | 20.0 | 16.0 | 163 840 | coder | MoE architecture; very long context; needs ≥16 GB VRAM |
| `gemma3:1b` | 1.0 | 3.0 | 2.0 | 32 768 | general | ultra-light; runs CPU-only on any modern laptop |
| `gemma2:2b` | 2.0 | 4.0 | 2.0 | 8 192 | general | efficient small model; good quality-per-GB |
| `gemma4:e2b` | 2.0 | 8.0 | 8.0 | 131 072 | general | Effective 2B; ~7.2 GB on disk; mobile/edge tier |
| `llama3.2:3b-instruct` | 3.0 | 6.0 | 4.0 | 131 072 | general | long context, light general-purpose |
| `phi3.5:3.8b-mini-instruct` | 3.8 | 6.0 | 4.0 | 131 072 | general | small, long-context, low-RAM |
| `gemma3:4b` | 4.0 | 6.0 | 4.0 | 131 072 | general | capable laptop model with very long context |
| `gemma4:e4b` | 4.0 | 12.0 | 10.0 | 131 072 | general | Effective 4B (gemma4:latest alias); ~9.6 GB on disk |
| `qwen2.5:7b-instruct` | 7.0 | 10.0 | 8.0 | 32 768 | general | general-purpose alternative to llama 8b |
| `llama3.1:8b-instruct` | 8.0 | 12.0 | 8.0 | 131 072 | general | general-purpose with strong reasoning |
| `gemma2:9b` | 9.0 | 12.0 | 8.0 | 8 192 | general | high-quality general; sweet spot on 12 GB RAM |
| `gemma4:12b` | 12.0 | 10.0 | 8.0 | 262 144 | general | Workstation dense; ~7.6 GB on disk at default quant |
| `deepseek-r1:1.5b` | 1.5 | 4.0 | 2.0 | 65 536 | reasoning | tiny reasoning model; runs on most laptops |
| `deepseek-r1:7b` | 7.0 | 10.0 | 8.0 | 65 536 | reasoning | chain-of-thought reasoning; solid on 8 GB GPU |
| `gemma4:26b` | 26.0 | 22.0 | 18.0 | 262 144 | reasoning | MoE (4B active/token); ~18 GB on disk |
| `gemma4:31b` | 31.0 | 24.0 | 24.0 | 262 144 | reasoning | Dense; ~20 GB on disk; near-cloud quality |

<details>
<summary>

### Fit verdict
</summary>

`compute_local_fit(hw)` computes an **effective budget** in GB and
classifies each entry:

```text
effective_budget = max(ram_gb, gpu_vram_gb * 2.0)   when a GPU is present
effective_budget = ram_gb                            otherwise
```

| Symbol | Condition                                                                                       |
|--------|-------------------------------------------------------------------------------------------------|
| ❓     | `effective_budget == 0` (probe returned nothing -- UI hasn't run *Detect hardware* yet).         |
| ❌     | `effective_budget < min_ram_gb * 0.9`. The model won't fit; not even tight.                     |
| ⚠️     | GPU present but `gpu_vram_gb < recommended_vram_gb * 0.75`, **or** budget within 1.2× min RAM. |
| ✅     | Budget ≥ 1.2× min RAM **and** GPU VRAM (if any) meets ≥75% of the recommendation.              |

The `reason` column on each row reports the exact comparison behind
the verdict so you can sanity-check the model against your own
machine without trusting the UI's symbol.

</details>
<details>
<summary>

### Adding or tweaking a model
</summary>

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

- All entries have the required fields and are well-typed
  (`test_catalog_is_non_empty_and_well_typed`).
- Rows are grouped by family (`coder → general → reasoning`) and
  ascend by `params_b` within each family
  (`test_compute_local_fit_rows_grouped_by_family_then_params`).
- Large models are rejected on a tiny machine, and tight-VRAM
  configurations are flagged
  (`test_compute_local_fit_tiny_machine_rejects_large_models`,
  `test_compute_local_fit_tight_vram_flagged_as_tight`).
- Everything fits on a 256 GB / 80 GB-VRAM workstation, and unknown
  hardware marks every row `❓`
  (`test_compute_local_fit_huge_machine_fits_everything`,
  `test_compute_local_fit_unknown_hardware_marks_all_unknown`).

</details>

</details>
<details>
<summary>

## Local vs cloud trade-offs
</summary>

The `TRADEOFFS` table is intentionally editorial -- short opinionated
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

</details>
<details>
<summary>

## Caveats
</summary>

- The catalogue numbers describe **4-bit quantised** inference. Full
  FP16 weights require roughly 2.5–3× more memory.
- VRAM budget is doubled when projecting onto the system-RAM budget
  (`max(ram, vram * 2)`) because partial offload typically works once
  ~half the model fits on the GPU. This is a heuristic, not a
  guarantee -- extremely small system RAM will still slow you down via
  KV-cache pressure.
- The trade-off table is editorial. If your privacy posture allows
  cloud (or your hardware can host a 70B model), the verdict for a
  given dimension may legitimately flip.

</details>
<details>
<summary>

## Programmatic access
</summary>

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

</details>
