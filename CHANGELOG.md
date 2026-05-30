# Changelog

All notable changes are documented here. Versions follow semver-ish.

## Unreleased — Scaffold prompt fix: frontend technology bias

### Fixed
- **`_SCAFFOLD_SYSTEM` Python bias** (`cgx.answer.engine`): the scaffold
  system prompt previously contained unconditional Python-specific
  instructions (`conftest.py`, `sys.path`, pytest import examples) that
  caused the LLM to generate Flask/Python files even when the goal
  explicitly requested a React or other frontend project. The
  Python-specific block is now gated under *"For PYTHON projects only"*.
  A matching *"For FRONTEND projects (React, Vue, etc.)"* section was
  added that explicitly instructs the LLM to generate component files
  (`App.jsx`, `index.js`) rather than webpack/babel build-tooling, and
  to omit Python files entirely.
- **`_SCAFFOLD_FREEFORM_SYSTEM` example bias** (`cgx.answer.engine`): the
  freeform fallback prompt's example fenced block was hardcoded as
  `` ```python path=src/main.py `` — swapped to a language-neutral
  `` ```<language> path=<relative/path/to/file> `` placeholder so the
  fallback path does not bias the model toward Python when JSON mode is
  unavailable.
- **Judge blind to technology mismatch** (`cgx.agents.judge`): the
  structural check for `scaffold` tasks previously passed any output that
  contained generated files, regardless of whether those files matched
  the requested technology. A React-specific check was added: when the
  task goal or description mentions "react", the judge now hard-fails
  (0.9 confidence) if no `.js`/`.jsx`/`.tsx`/`.ts` files are present or
  if all non-config files are Python — producing a clear rationale that
  triggers a proper retry with corrected instructions.

## Unreleased — New-project scaffold, agent kind-policy fixes

### Added
- **`TaskKind.SCAFFOLD`** (`cgx.agents.types`): new task kind that routes
  to `cgx.answer.engine.generate_project_scaffold`. The planner emits a
  `[scaffold, apply, verify]` chain whenever the goal describes a
  *new-project* request ("create a new FastAPI app", "build from scratch",
  "generate a CLI tool", etc.). No existing index is required — the LLM
  generates all files from a plain-language idea.
- **`generate_project_scaffold(idea, provider)`** (`cgx.answer.engine`):
  LLM function that produces a complete project from a free-text idea.
  Prefers JSON mode with `{plan_md, files: [{path, content}]}` output;
  falls back to free-form fenced blocks (`` ```<language> path=... `` ``).
  File contents are converted to `--- /dev/null` new-file unified diffs
  so the existing `apply_diffs_to_disk` pipeline writes them to
  `project_root` unchanged.
- **`_SCAFFOLD_RE` + `_goal_is_scaffold()`** (`cgx.agents.planner`):
  regex-based scaffold-goal detector (checked before the change-verb
  regex so "create a new project" doesn't accidentally route to `plan`).
- **`scaffold` capability** (`cgx.agents.loop._build_default_capabilities`):
  calls `generate_project_scaffold`; does not call `_need_index()` so
  it works without an existing codebase.

### Fixed
- **Judge PLAN structural check too strict** (`cgx.agents.judge`):
  the `_structural_check` for `plan` tasks previously hard-failed (90%
  confidence) whenever `diffs` was empty, even when `plan_md` had
  meaningful content. Local LLMs that produce a narrative plan without
  diff blocks now fall through to the LLM judge instead of being
  rejected outright. Only the case where *both* `plan_md` and `diffs`
  are absent triggers a hard structural fail.

### Changed
- `cgx.agents.planner.SYSTEM_PROMPT`: extended to describe the `scaffold`
  kind and updated routing rules (new-project goals → `scaffold` only;
  existing-codebase change goals → `plan`).
- `cgx.agents.planner._enforce_kind_policy`: scaffold goals are
  intercepted first (before verify-only and change-verb checks) and
  always produce a clean `[scaffold, apply, verify]` chain.
- `cgx.agents.tracker._summarize_task_output` / `_extract_display_output`:
  SCAFFOLD tasks surface "N file(s) generated + preview" in the timeline
  row and reuse the PLAN rendering for the diff viewer panel.
- All relevant docstrings and module-level docs updated.

## Unreleased — Observability, task registry, tab persistence, parallel execution

### Added
- **Startup logging** (`launch.py`): `setup_logging(INFO)` is now called once at
  process start so every major operation emits structured `[INFO]`/`[WARNING]`
  lines to stdout — handlers (`ask`, `plan`, `agent`, `index`), tracker (each
  `task_start` / `task_done` / `task_fail`), planner (LLM call, task count,
  fallback activation), and the SSE bridge.
- **SQLite task registry** (`src/cgx/webui/task_store.py`): every SSE request
  (`ask`, `plan`, `agent`, `index`) now creates a row in `~/.cgx/tasks.db`.
  All emitted SSE events are persisted per-task so the frontend can replay them
  on tab switch. An in-memory `threading.Event` per task supports cancellation.
- **Task REST API** (`src/cgx/webui/routes/tasks.py`, mounted at `/api/tasks`):
  - `GET /api/tasks` — list recent tasks (up to 50).
  - `GET /api/tasks/{id}` — retrieve a single task record.
  - `GET /api/tasks/{id}/events` — full persisted event log for replay.
  - `DELETE /api/tasks/{id}` — cancel a running task (sets its `threading.Event`).
- **SSE bridge cancellation** (`src/cgx/webui/sse.py`): `bridge_generator()` now
  accepts `task_id` and `cancel_event` parameters. All handlers (`stream_ask`,
  `stream_plan`, `stream_agent`, `stream_index`) check `cancel_event.is_set()`
  at each yield point and terminate the stream cleanly when set.
- **Cancel / Stop buttons**: every streaming page now renders an abort button
  while busy — **Stop** on Ask, **Cancel** on Plan, Agent, and Index — that
  closes the SSE connection and sets the cancel event.
- **Tab persistence** (`frontend/src/store/tasks.ts`,
  `frontend/src/lib/connections.ts`):
  - `tasks.ts` — Zustand store backed by `sessionStorage` that holds streaming
    state per page: agent (tasks / events / phase / summary), ask (messages),
    plan (thought / planMd / diff / report), index (progress / result).
  - `connections.ts` — module-level `Map<string, SseConnection>` holding live
    SSE connections outside the React component lifecycle; switching tabs
    unmounts the component but leaves the SSE connection streaming and updating
    the Zustand store so state is fully intact on remount.
- **Sidebar running indicators**: the left navigation sidebar shows an animated
  spinner next to any tab that currently has a running task, driven by the
  Zustand tasks store.
- **Parallel two-view indexing** (`src/cgx/pipeline/auto.py`,
  `run_index_auto()`): intent-view and impl-view FAISS index builds now run
  concurrently inside a `ThreadPoolExecutor`, reducing total indexing time
  roughly 2× on multi-core machines.
- **Parallel semantic search** (`src/cgx/retrieval/orchestrator.py`,
  `HybridRetriever.search()`): intent-view and impl-view ANN searches now run
  concurrently inside a `ThreadPoolExecutor`. RRF fusion and result ordering
  are unchanged.

### Changed
- `bridge_generator()` in `src/cgx/webui/sse.py` signature extended with
  `task_id: str` and `cancel_event: threading.Event`.
- `run_index_auto()` in `src/cgx/pipeline/auto.py` now dispatches both view
  builds to threads rather than building them sequentially.
- `HybridRetriever.search()` in `src/cgx/retrieval/orchestrator.py` now
  dispatches both ANN searches to threads rather than running them sequentially.

## Unreleased — Agent loop polish

### Added
- **Planner kind-policy enforcement** (`cgx.agents.planner._enforce_kind_policy`):
  every plan emitted by the LLM is post-validated and any `plan` task is
  downgraded to `ask` when the goal text doesn't match the change-verb
  regex, so informational queries no longer trigger expensive
  code-generation work.
- **Task short titles** (`Task.name`): planner schema now asks for
  `{name, description, kind, criteria}`; `_derive_name()` distils a clean
  title from the first sentence when the LLM omits it. Threaded through
  `_fallback_plan` and surfaced in `task_start` / `task_done` payloads.
- **Live progress heartbeats** (`cgx.agents.tracker`): each capability
  runs in a worker thread and the Tracker yields a `task_progress`
  `AgentEvent` every `progress_interval` seconds (default `2.0`) with
  `{task_id, name, kind, elapsed}`. `run_agent` forwards the parameter
  end-to-end.
- **React Agent UI** (`frontend/src/pages/AgentPage.tsx`): new
  `PlanTasksHeader` (clipboard icon + count pill), vertical
  `TaskTimelineRow` with status circles (pending / pulsing run / done
  check / failed cross / skipped dash), bold task names, and a live
  elapsed-seconds badge driven by `task_progress`.
- **Audience-specific flowcharts** under `docs/diagrams/` (`flow_user.svg`,
  `flow_developer.svg`, `flow_company.svg`) plus a `docs/flowcharts.md`
  index linked from the README, architecture, and usage docs.
- **Tests** (`tests/test_agents.py`): coverage for kind-policy downgrade,
  `_derive_name`, threaded dispatch, and `task_progress` event emission.

### Changed
- `AgentEvent` union now includes `task_progress`; `docs/architecture.md`,
  `docs/usage.md`, and `README.md` updated to list the full event set.
- `task_start` / `task_done` SSE payloads now carry `name` alongside
  `description`.

## 0.2.0 — Phase 2 (current)

### Added
- **Self-testing code generation** (`cgx.codegen`): unified-diff parser,
  in-memory dry-apply, AST-based syntax validation, sandboxed
  pytest-impact runner, and a feedback-driven retry loop in
  `generate_code_plan`.
- **Intent-conditioned system prompts** (`SYSTEM_PROMPTS`) for
  `symbol_explain`, `howto`, `change_plan`, `symbol_location`,
  `line_number`, and `overview` modes.
- **Snippet windowing** (`_window_text`) trims SOURCES to the lines
  surrounding the focus symbol, cutting prompt size 5–10× while keeping
  the relevant region.
- **Provider streaming** (`LLMProvider.chat_stream`) with real
  implementations on Ollama (`/api/chat` NDJSON) and OpenAI-compatible
  endpoints (SSE).
- **Provider profile store** (`cgx.answer.profiles`) with OS keyring
  backing when available and a `0600` file fallback under `~/.cgx/`.
- **Ollama discovery** (`cgx.answer.ollama_discovery`): installed-model
  listing, health check, hardware probing, and a hardware-aware
  recommended-default model picker.
- **Gradio UI overhaul** (`cgx.ui`): five-tab product layout (Setup,
  Index, Ask, Plan, Profiles), streaming thought-process panel, diff
  viewer, soft theme.
- `averix-ui` console entry point.
- `docs/architecture.md` and `docs/usage.md`.
- GitHub Actions CI (`.github/workflows/ci.yml`) running pytest +
  py-compile on 3.10 / 3.11 / 3.12.
- Pytest suite covering codegen pipeline, intent detection, profile
  store, snippet windowing, and Ollama discovery (non-network paths).
- End-to-end integration test that runs `run_index_auto` →
  `run_query_auto` against a tiny on-disk project with a deterministic
  hash-based fake embedder (no model download, no GPU).
- `LICENSE` file (MIT).
- **Optional cross-encoder reranker** (`cgx.retrieval.reranker`) gated
  behind `HybridConfig.enable_reranker`. Defaults to
  `cross-encoder/ms-marco-MiniLM-L-6-v2`, lazy-loads
  `sentence_transformers`, and silently falls back to the RRF order if
  the model can't be loaded.
- `HybridConfig.symbol_boost`, `graph_bonus`, `enable_reranker`,
  `reranker_model`, `reranker_top_n`, `reranker_weight`,
  `expand_per_seed`, `relation_types` — previously hard-coded magic
  numbers are now tunable.
- Rerank regression tests (`tests/test_rerank.py`) covering the
  graph-only-neighbor fix, config-driven boosts, and the cross-encoder
  hook with an injected fake model.
- **`requirements-ml.txt`** for the optional embedding / reranker stack
  (`torch`, `transformers`, `sentence-transformers`). The base
  `requirements.txt` is now torch-free.
- CI workflow split into a **core** matrix (Python 3.10/3.11/3.12, no
  torch) that asserts the lazy-import path stays clean, plus an optional
  **ml** job that exercises the embedding/reranker stack.
- **Multi-agent orchestration** (`cgx.agents`): a Planner that decomposes
  a goal into ordered atomic tasks, a Tracker state machine that
  executes each task by dispatching to the existing Ask / Plan / Search
  capabilities, and a Judge that validates outputs against
  per-task criteria (with both LLM and structural fallbacks).
  Exposed via :func:`cgx.agents.run_agent` and a new **🤖 Agent** tab
  in the Gradio UI that streams a live execution log.
- **Anonymous opt-in telemetry** (`cgx.telemetry`): single startup ping
  carrying only a random installation ID and the package version. Off
  by default; toggled via the `CGX_TELEMETRY=1` environment variable.
- **Persistent privacy banner** at the top of the Gradio UI and a new
  *Privacy & data flow* section in `README.md` confirming that all
  parsing, embedding, indexing, retrieval, and session storage stay
  local.
- **Client-side rate limiter + retry** (`cgx.answer.ratelimit`):
  token-bucket throttling plus exponential-backoff retry on HTTP 429 /
  5xx for the OpenAI-compatible and Ollama providers, with optional
  `rate_limit` / `max_retries` fields persisted on each `Profile`.
- **Execution graph visualizer** (`cgx.agents.viz`): the Agent tab now
  renders a live status table and HTML DAG of the planner's tasks
  alongside the streaming event log.
- **Persistent chat sessions** (`cgx.sessions`): JSONL-backed thread
  store under `~/.cgx/sessions/` and a session sidebar in the Ask tab
  for creating, listing, and resuming historical conversations.
- **Incremental indexing** (`cgx.embeddings.cache`): content-addressed
  embedding cache (sha256 of the corpus text → vector) persisted as a
  per-view `.npz`. `run_index_auto` now reuses cached vectors for
  unchanged chunks and only invokes the embedder on misses; reports
  `embedding_cache` hit/miss stats in its return dict. Disable with
  `incremental=False`.
- **Hardware / trade-offs dashboard** (`cgx.answer.hardware_matrix` +
  new **📊 Hardware** UI tab): offline catalogue of 8 locally-runnable
  models annotated against detected RAM/VRAM with a ✅ / ⚠️ / ❌ fit
  verdict, plus an editorial local-vs-cloud comparison across privacy,
  cost, quality ceiling, latency, offline use, setup, and operational
  risk. Exported as `docs/hardware_matrix.json` for downstream tooling.
- **VS Code extension scaffold** (`extension/`): minimal TypeScript
  extension exposing **Averix: Open UI** / **Averix: Reload UI** that
  host the running Gradio server in a webview panel. Server URL is
  configurable via the `averix.ui.url` setting. Source-only scaffold;
  not packaged into a `.vsix` from the repo.

### Changed
- `cgx.embeddings.build` no longer imports `torch` /
  `sentence_transformers` / `transformers` at module load; they are
  loaded lazily inside `build_embeddings`. The UI and any BYO-embedder
  path now work on machines without the ML stack installed.
- Removed the legacy `app_gradio_llm.py`; `cgx ui` / `averix-ui` /
  `app.py` all launch `cgx.ui.app.build_demo()` directly.
- `pyproject.toml`: corrected package layout, declared new optional
  extras (`codegen`, `keyring`, `dev`), added `averix-ui` script.
- `OllamaProvider` default model is now `qwen2.5-coder:3b`.
- `parse_codebase` honours `.gitignore`, default ignore globs, a 1 MB
  file-size cap, and skips symlinks by default.
- `LLMProvider.chat` gained a `force_json` toggle; `generate_code_plan`
  falls back to free-form output when JSON-mode mangles unified diffs.
- **Gradio 6.0 compatibility:** moved `theme=gr.themes.Soft()` out of
  the `Blocks(...)` constructor and into `launch()` (the constructor
  arg was deprecated and emits a `UserWarning` on 6.x).
- `src/cgx/retrieval/hybrid.py` is now a thin re-export of
  `cgx.retrieval.orchestrator.{HybridRetriever,HybridConfig}` instead of
  hosting a parallel ~420-line implementation. `cli_adapter` (the
  standalone `python -m cgx.retrieval.cli_adapter --hybrid …` path)
  keeps working but now shares a single source of truth with
  `run_query_auto`.

### Fixed
- **Reranking dropped graph-only neighbors.** In
  `HybridRetriever.search`, the post-RRF graph-bonus loop only updated
  scores for chunks already in `fused`, so neighbors discovered via
  graph expansion never appeared in `hits` despite being recorded in
  `provenance`. They are now appended (provided a record exists), and
  the score update is no longer O(N²) per neighbor.
- Mixed `src.cgx` vs `cgx` import paths.
- Intent detection mis-routing (`change`/`add` matching before
  symbol-targeted phrasing).
- Symbol-token substring false positives in the orchestrator.
- Hardcoded `top_k_per_view=3` in the Gradio app.
- `generate_code_plan` retrieving by index order instead of hybrid
  retrieval.
- Graph callers/callees walk now filters by `type=='calls'` edges only.
- JSON extraction uses a balanced-brace scanner.

## 0.1.x
- Initial hybrid-retrieval RAG prototype.
