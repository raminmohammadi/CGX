# Architecture

This page is the contributor's map of the codebase: the top-level
packages under `src/cgx/`, how they compose end-to-end, and where to look
when you want to change something. For the exhaustive internals see
[`docs/architecture.md`](https://github.com/raminmohammadi/Averix/blob/main/docs/architecture.md) and
[`docs/Agent.md`](https://github.com/raminmohammadi/Averix/blob/main/docs/Agent.md).

---

## End-to-end flow

```
        parser/            embeddings/ + graph/        retrieval/
 files -----------> chunks -------------------> index -----------> ranked
        (AST/TS)         (2 views + graph)      (FAISS)   (hybrid)  chunks
                                                                      |
                                                                      v
 answer/  <-------------------------------------------------  prompt assembly
 (LLM providers, engine)                                        (Code Map)
    |
    +--> ask   (read-only grounded answer)
    +--> plan  --> codegen/ (validate + self-test)
    +--> session/ (agent DAG) --> codegen/ (apply, backups, verify)
```

Two surfaces sit on top: the **CLI** (`cli/`) and the **web UI**
(`webui/`, a FastAPI + SSE backend serving a React SPA).

---

## Top-level packages

| Package | Responsibility |
|---------|----------------|
| `parser/` | Language-aware parsing (`python_parser`, `js_ts_parser`, `markdown_parser`) on a tree-sitter base, plus incremental parse caching. Emits typed chunks + symbol/call metadata. |
| `embeddings/` | Builds the two views (**intent** summaries, **impl** code), embeds them (`build`, `loader`, `catalog`), and manages the per-view embedding cache (`cache`). |
| `graph/` | Import/call graph construction (`build_graph`, `backend`, `aggregation`) used for graph-expansion during retrieval. |
| `retrieval/` | Hybrid search: FAISS ANN (`index`), lexical/BM25 (`lexical`), RRF fusion (`rrf`), optional cross-encoder (`reranker`), all wired by the `orchestrator`. |
| `answer/` | LLM provider abstraction (`providers`), the answer/plan `engine`, prompt/context assembly (`context_map`, `repo_map`), profiles, rate limiting, hardware matrix. |
| `codegen/` | Diff parsing/apply (`diff_apply`, `disk_apply`), validation (`validate`), self-test loop (`pipeline`, `test_runner(s)`), dynamic deps (`env_manager`), AST insertion (`ast_insert`), AST scaffolding fallback (`ast_gluer`). |
| `session/` | The persistent agent orchestrator: `router` (pure state machine), `runner`, `store`, `tasks/` (executors), `repair/`, `budget`, `mode`. |
| `pipeline/` | High-level entry points that wire parsing → embedding → indexing (`auto.run_index_auto`, `run`). |
| `webui/` | FastAPI app (`server`), `routes/`, SSE (`sse`), task store, and the launcher (`launch`). Serves the built SPA from `static/`. |
| `cli/` | The `cgx` command (`main`) and the interactive terminal dashboard (`tui/`). |
| `io/` | Persistence for indices, records, chunks, and graphs (`persist`). |
| Cross-cutting | `config` (env-driven dataclasses), `trace` / `redact` / `metrics` (observability), `telemetry` (opt-in ping), `logging_setup`. |
| MLOps layer | `metrics`, `health`, `registry`, `usage`, `activity`, `eval`, `monitor`, `feedback`, `governance`, `guardrails`, `govdata` — the production observability/governance subsystems. See **[[MLOps and Production]]**. |

`skills/` lives at the **repo root** (not under `src/cgx/`) — a
plug-and-play registry consumed by `answer.engine` and the session
executors. See **[[Skills Registry]]**.

---

## The two retrieval views

Every chunk is embedded twice:

- **intent** — a natural-language summary of what the code does.
- **impl** — the (optionally skeletonized) source itself.

Queries search both views; results are fused with **Reciprocal Rank
Fusion**, boosted by graph neighbors and symbol matches, and optionally
re-scored by a cross-encoder. Full detail in **[[How It Works]]**.

---

## The session orchestrator

The agent is a **persistent DAG of typed tasks** with three clean layers:

- `router.Router` — a pure, LLM-free, I/O-free state machine returning a
  typed `RouterPlan`. Deterministic and replayable.
- `runner.SessionRunner` — per-session lock, executor dispatch, failure
  handling, persistence, and the session-budget circuit breaker.
- `tasks/*` — one executor per `TaskKind`; the **only** layer that calls
  the LLM or touches disk.

State persists to `<project_root>/.cgx/sessions.db`. Full walkthrough:
**[[Session Based Agent]]**.

---

## Design invariants

These hold across the codebase; respect them when contributing:

- **Torch-free core.** No top-level import of `torch`, `transformers`, or
  `sentence_transformers` inside `src/cgx/` — keep them lazy inside
  function scope so the base install stays light.
- **Local-first.** No new default egress path; cloud calls are always
  opt-in. See **[[Privacy and Security]]**.
- **Pure router.** All LLM/I/O stays in executors, never in routing.
- **Prevent, don't just recover.** Greenfield plans are calibrated to a
  scope ceiling, self-critiqued, and de-scoped of speculative /
  sandbox-unrunnable work at `DECOMPOSE` / `BOOTSTRAP_ENV` time, so the
  recovery ladder has less to fix downstream. See **[[Session Based Agent]]**.
- **Diagnose before you regenerate.** A mechanical failure still takes an
  instant `REPAIR` fast path, but a *reasoning-class* failure routes to a
  `DIAGNOSE` rung that reasons over the failure, the repo, and a
  `REPAIR_LEDGER` of already-tried actions, then emits a single
  `minimal_action` verdict the pure router maps to a targeted fix (patch,
  dependency install/de-scope, or a **scoped** regenerate) instead of
  nuking the whole tree. Deterministic-first: it degrades to `escalate`
  on any provider outage. See **[[Session Based Agent]]**.
- **Re-verify only what broke.** When a diagnosed fix originated from a
  `VERIFY` failure, the router splices a **RE_VERIFY** task that re-runs
  pytest against only the failing test file(s) instead of replaying the
  whole `BOOTSTRAP_ENV → API_CHECK → SMOKE → VERIFY` chain — the venv is
  already provisioned and every other gate already passed. Non-`VERIFY`
  origins run the full chain, so behavior is never worse than before. See
  **[[Session Based Agent]]**.
- **Additive persistence.** Index/record writers are add-only and
  degrade gracefully when optional deps (FAISS, ML stack) are absent.
- **Skills add no agent-layer edits.** New technology support is a
  single-folder change.

---

## Where to make a change

| Goal | Start here |
|------|-----------|
| Support a new framework | `skills/<name>/` — **[[Skills Registry]]** |
| Add/adjust a provider | `answer/providers.py` |
| Tune retrieval | `retrieval/orchestrator.py` (`HybridConfig`) |
| Change a codegen safeguard | `codegen/validate.py`, `codegen/disk_apply.py` |
| Add an agent task/transition | `session/router.py` + `session/tasks/` |
| Add an HTTP endpoint | `webui/routes/` + register in `webui/server.py` |
| Add a CLI subcommand | `cli/main.py` |

---

## See also

- **[[How It Works]]** — the retrieval pipeline in depth.
- **[[Session Based Agent]]** — the orchestrator internals.
- **[[MLOps and Production]]** — the observability/governance/deploy layer.
- **[[Contributing]]** — dev setup, tests, and the PR checklist.
- [`docs/architecture.md`](https://github.com/raminmohammadi/Averix/blob/main/docs/architecture.md) — the full reference.
