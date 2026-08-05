# Configuration and Tuning

This page collects the knobs that shape retrieval quality, indexing
speed, provider behaviour, and where CGX stores its state. Defaults are
reasonable out of the box — reach for these when you want more.

---

## Tuning hybrid retrieval

`HybridConfig` (in `cgx.retrieval.orchestrator`) controls post-RRF
reranking. Each signal can be disabled or amplified independently:

| Field             | Default | Effect |
|-------------------|---------|--------|
| `graph_bonus`     | `0.2`   | RRF-scaled bump for chunks reached via the import/call graph. `0.0` ignores graph-only neighbors. |
| `symbol_boost`    | `0.5`   | Bonus for chunks whose identifier or path matches a query token. |
| `enable_reranker` | `False` | Run a cross-encoder over the top-N fused chunks. |
| `reranker_model`  | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Hugging Face model id. |
| `reranker_top_n`  | `30`    | How many head candidates to re-score. |
| `reranker_weight` | `1.0`   | Convex blend of cross-encoder vs RRF score (`1.0` = CE only). |

```python
from cgx.retrieval.orchestrator import HybridConfig
cfg = HybridConfig(enable_reranker=True, reranker_top_n=20, graph_bonus=0.3)
```

The reranker lazy-loads `sentence_transformers` only when
`enable_reranker=True`; without the ML stack it silently falls back to
the RRF order. Install `requirements-ml.txt` to opt in.

When `graph_bonus > 0` surfaces neighbors, the answer pipeline switches to
the two-tier **Code Map** prompt automatically — see **[[How It Works]]**.

---

## Environment variables

CGX reads configuration from the environment (empty string counts as
unset). The most useful:

| Variable | Default | Purpose |
|----------|---------|---------|
| `CGX_CONFIG_DIR` | `~/.cgx` (`%USERPROFILE%\.cgx` on Windows) | Where profiles, secrets, sessions, and caches live. |
| `CGX_HOST` / `CGX_PORT` | `127.0.0.1` / `8765` | Web-UI bind address (also `--host` / `--port`). |
| `CGX_EMBED_MODEL` | `jinaai/jina-embeddings-v2-base-code` | Local embedding model. |
| `CGX_EMBED_DEVICE` | auto (CUDA > MPS > CPU) | Force `cpu`, `cuda`, or `mps`. |
| `CGX_EMBED_BATCH` / `CGX_EMBED_MAXLEN` | `64` / `8192` | Embedding batch size / max tokens. |
| `CGX_TELEMETRY` | `0` (off) | Set `1` to enable the anonymous startup ping. |
| `NO_COLOR` / `CGX_NO_COLOR` / `CGX_FORCE_COLOR` | — | Terminal colour control. |

**FAISS knobs** (`CGX_FAISS_*`): `METRIC` (`cosine`), `INDEX`
(`flat`/`ivf`/`hnsw`), `NLIST`, `NPROBE`, `HNSW_M`, `HNSW_EFC`,
`HNSW_EFS`, `GPU`.

**Retrieval knobs** (`CGX_*`): `TOP_K` (30), `K_INTENT`/`K_IMPL`/`K_LEX`
(50), `EXPAND_TOP_N` (10), `EXPAND_PER_SEED` (12), `RRF_K` (60.0),
`BUILD_GRAPH` (1).

These map to the dataclasses in
[`src/cgx/config.py`](https://github.com/raminmohammadi/Averix/blob/main/src/cgx/config.py) and can also be set via
`.from_overrides(...)`.

---

## Incremental indexing

`run_index_auto` is incremental by default at **two** layers:

- **Parse layer** — `parse_cache.json` keyed on each file's mtime/sha, so
  only edited files are re-parsed.
- **Embedding layer** — per-view `emb_cache_<view>.npz` keyed on
  `sha256(corpus_text)`, so only changed chunks reach the embedder.

```python
result = run_index_auto(project_root="./", out_dir="/tmp/cgx_index")
print(result["incremental"])       # True
print(result["embedding_cache"])
# {'intent': {'hits': 412, 'misses': 5, 'dim': 768}, ...}
```

`hits` = chunks served from cache; `misses` = chunks re-embedded;
`hits + misses` = chunks in the view. The cache is auto-invalidated when
the embedding `model_name`, `dim`, or `normalize` flag changes — no risk
of stale vectors. Force a clean rebuild with `incremental=False`.

> Not to be confused with retrieval `hits` (the top-k chunks a query
> returns). Cache hits measure index reuse; retrieval hits measure search
> results.

---

## Rate limits & retries (per profile)

```python
from cgx.answer.profiles import Profile, save_profile
save_profile(Profile(
    name="my-cloud",
    kind="openai-compat",
    model="gpt-4o-mini",
    base_url="https://api.openai.com/v1",
    rate_limit=2.0,   # 2 req/sec, bucket capacity = rate
    max_retries=4,    # default 0; 4 ≈ ~30s backoff ceiling
))
```

`rate_limit=None` (default) makes the limiter a no-op. See
**[[Providers and Models]]**.

---

## Session budgets (greenfield agent)

Greenfield sessions seed finite backstops so an autonomous build always
halts: `max_task_runs` and `max_wall_seconds`, plus per-loop repair /
regenerate budgets (`REPAIR_BUDGET=4`, `REGENERATE_BUDGET=3`, …) in
`cgx.session.budget.LoopBudget`. See **[[Session Based Agent]]**.

---

## See also

- **[[How It Works]]** — what each stage does.
- **[[Providers and Models]]** — provider-side configuration.
- [`docs/usage.md`](https://github.com/raminmohammadi/Averix/blob/main/docs/usage.md#5-tune-retrieval-optional).
