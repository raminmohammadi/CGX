# Architecture

Averix is structured as a small set of cooperating layers under `cgx.*`.

## Layers

```
cgx.parser              — language-aware tree walker → chunk records
cgx.graph               — NetworkX call/containment graph over chunks
cgx.embeddings          — two-view (intent / impl) corpora, FAISS indices
cgx.embeddings.cache    — content-addressed embedding cache (.npz)
cgx.retrieval           — hybrid retriever (semantic + BM25 + graph) + RRF
cgx.answer              — LLM providers, intent detection, prompt registry
cgx.answer.ratelimit    — token-bucket limiter + 429/5xx retry
cgx.answer.profiles     — provider config + keyring-backed secret store
cgx.answer.hardware_matrix — offline local-model catalogue + tradeoffs
cgx.answer.ollama_discovery — installed-model listing + hardware probe
cgx.codegen             — diff parse / dry-apply / syntax & test validation
cgx.pipeline            — high-level orchestrators (run_index_auto, run_query_auto)
cgx.agents              — Planner / Tracker / Judge multi-agent loop
cgx.agents.viz          — DAG + status-table renderers for the Agent tab
cgx.sessions            — append-only JSONL conversation store
cgx.telemetry           — opt-in anonymous startup ping
cgx.webui.task_store    — SQLite task registry + threading.Event cancel tokens
cgx.webui.routes.tasks  — REST API for task list / get / event-replay / cancel
cgx.cli / cgx.ui        — terminal + Gradio surfaces
```

## Data flow

1. `parse_codebase` walks the repo (respecting `.gitignore`, ignore globs
   and a 1 MB file-size cap) and emits one chunk per file/class/function.
2. `build_knowledge_graph` derives a NetworkX graph with `calls`, `module`,
   `attr` and `defined_in` edges.
3. `make_index_records` materialises chunk records, then
   `prepare_embedding_corpus` builds two views:
   - **intent** — NL-friendly summary (docstrings, names).
   - **impl** — implementation text (signatures + bodies).
4. `build_embeddings` + `build_faiss_index` persist per-view ANN indices.
   Embeddings flow through a content-addressed cache
   (`cgx.embeddings.cache`, one `.npz` per view) keyed on the sha256 of
   the corpus text. Subsequent `run_index_auto` invocations re-embed
   only changed chunks; the cache is invalidated automatically when the
   embedding model or its dim/normalisation flag changes. Disable with
   `run_index_auto(..., incremental=False)`. **Both the intent-view and
   impl-view builds now run concurrently** inside a `ThreadPoolExecutor`
   in `run_index_auto()`, reducing total indexing time roughly 2× on
   multi-core machines. `build_embeddings` auto-detects CUDA > MPS > CPU
   at runtime and selects the fastest available device.
5. `HybridRetriever.search` (in `cgx.retrieval.orchestrator`) fuses
   three signals using Reciprocal Rank Fusion, then applies a
   tunable post-fusion rerank pass. **The intent-view and impl-view ANN
   searches now run concurrently** inside a `ThreadPoolExecutor`; the
   results are joined before RRF fusion so ordering is unchanged:
   - semantic search on each view (intent + impl),
   - BM25 lexical search,
   - graph expansion from the top hits (`graph_bonus`),
   - symbol-match boosting (`symbol_boost`) for chunks whose identifier
     or path matches a token in the question,
   - optional cross-encoder rerank (`enable_reranker`) that lazy-loads
     `sentence_transformers` and silently falls back to the RRF order
     if the ML stack is absent.
6. `answer_with_llm` selects an intent-conditioned system prompt
   (`SYSTEM_PROMPTS`), builds line-windowed SOURCES around the focus
   symbol, and asks the configured `LLMProvider` for a JSON answer.
7. `generate_code_plan` does the same but routes through a
   diff-aware retry loop in `cgx.codegen.pipeline.validate_and_test`.

## Self-test loop

`validate_and_test` is the orchestrator:

1. `parse_fenced_diffs` extracts ` ```diff path=... ``` blocks.
2. `apply_diffs_in_memory` projects each patch onto the current file.
3. `validate_patch_results` runs `ast.parse` over Python targets.
4. `run_impacted_tests` (when enabled) copies the project into a
   temporary directory, materialises the diffs, and runs
   `pytest <impacted_files>` with a timeout.
5. On failure, `build_retry_feedback` summarises the breakage and the
   loop in `generate_code_plan` re-asks the model in free-form mode.

The whole report is returned under `parsed["codegen_report"]` and rendered
in the UI as a markdown table.

## Multi-agent loop

`cgx.agents` adds an orchestration layer on top of the single-shot
`answer_with_llm` / `generate_code_plan` / `generate_project_scaffold`
entry points:

1. **Planner** (`cgx.agents.planner.Planner`) — decomposes a user goal
   into an ordered list of atomic `Task`s. Prefers the LLM with a
   strict JSON schema (1–5 tasks, each with `name` (short title),
   `description` (imperative sentence),
   `kind ∈ {ask, plan, scaffold, search, summarize, apply, verify}`,
   and plain-English `criteria`); falls back to a deterministic
   single-task plan derived from `detect_intent` when no provider is
   available or the model returns garbage. When the LLM omits `name`,
   `_derive_name()` distils a clean title from the first sentence of
   the description. A post-validation step (`_enforce_kind_policy`)
   applies three routing rules in priority order:
   - **Scaffold goals** ("create a new project", "from scratch", etc.)
     → always `[scaffold, apply, verify]`; no index required.
   - **Verify-only goals** ("do the tests pass?") → `[verify]`.
   - **Read-only goals** (no change verb) → any `plan` task is
     downgraded to `ask`, preventing expensive code-gen on informational
     questions.
   - **Code-change goals** → `apply` + `verify` appended after the
     final `plan` task.
2. **Tracker** (`cgx.agents.tracker.Tracker`) — drives the plan
   task-by-task, dispatching each kind to a caller-supplied capability
   callable (`ask`, `plan`, `scaffold`, `search`, `summarize`, `apply`,
   `verify`). `ask`, `plan`, `scaffold`, and `search` receive the task
   description as their first argument; `summarize`, `apply`, and
   `verify` receive the list of all prior task outputs. Each capability
   runs in a worker thread so the loop can emit a `task_progress`
   heartbeat every `progress_interval` seconds (default `2.0`) carrying
   `{task_id, name, kind, elapsed}` — the UI uses this as a live
   "running for Ns" counter. On completion the Tracker invokes the Judge
   and emits one of `task_done` / `task_failed` / `task_skipped`. The
   full `AgentEvent` set is: `plan`, `task_start`, `task_progress`,
   `task_done`, `task_failed`, `task_skipped`, `judge`, `summary`.
3. **Judge** (`cgx.agents.judge.Judge`) — validates each completed task
   against its criteria. Performs cheap structural short-circuits before
   optionally asking the LLM for a strict `{verdict, confidence,
   rationale}` JSON verdict. Per-kind rules:
   - `ask`: hard-fail when `answer_md` is empty.
   - `plan`: hard-fail only when *both* `plan_md` and `diffs` are
     absent; when `plan_md` exists but `diffs` is empty (e.g. a local
     LLM that produced a narrative plan), passes to LLM judge.
   - `scaffold`: hard-fail when both `plan_md` and `diffs` are absent.
     When files are present, a technology-match check runs first: if the
     goal mentions "react", the judge hard-fails (0.9 confidence) if no
     `.js`/`.jsx`/`.tsx`/`.ts` files were generated or if all
     non-config files are `.py` — preventing a Flask/Python output from
     silently passing a React scaffold request. If the technology check
     passes, falls through to the LLM judge.
   - `search`: structural pass when `hits > 0` (LLM judge not invoked).
   - `apply`: fail when `failed_files` is non-empty or nothing was
     written; pass when `applied_files` is non-empty.
   - `verify`: trusts pytest exit code directly; soft-pass on "no
     impacted tests".

The high-level `cgx.agents.run_agent(goal, …, progress_interval=2.0)`
wires all three to the default capabilities backed by the existing
engine, and is exposed via the **🤖 Agent** tab in the React UI. See
[flowcharts.md](flowcharts.md) for a visual breakdown of the loop and
the event timeline.

The **SSE bridge** (`cgx.webui.sse.bridge_generator`) records every
emitted `AgentEvent` into the task registry (`cgx.webui.task_store`) so
the frontend can replay the full event log when the user switches back to
the Agent tab. The bridge also accepts a `cancel_event: threading.Event`;
when it is set (e.g. via `DELETE /api/tasks/{id}`) the generator
terminates cleanly between yields.

## Task registry

`cgx.webui.task_store` is a lightweight SQLite store (database at
`~/.cgx/tasks.db`) that records every SSE operation — `ask`, `plan`,
`agent`, and `index` — from the moment the request arrives to the final
`done` / `error` event.

**Schema** (simplified):

- `tasks` table — one row per operation: `id` (UUID), `kind`, `status`
  (`running` / `done` / `cancelled` / `error`), `created_at`,
  `updated_at`, `goal` / `query` text.
- `task_events` table — one row per SSE event: `task_id`, `seq`,
  `event_type`, `payload` (JSON), `ts`.

**Cancellation**: a module-level `dict[str, threading.Event]` maps each
running `task_id` to its cancel token. Setting the event causes
`bridge_generator()` to break out of its yield loop and emit a
`cancelled` event. The cancel event is cleared automatically when the
task ends.

**REST API** (`cgx.webui.routes.tasks`, mounted at `/api/tasks`):

| Method   | Path                      | Description                                     |
|----------|---------------------------|-------------------------------------------------|
| `GET`    | `/api/tasks`              | List up to 50 most-recent tasks (newest first). |
| `GET`    | `/api/tasks/{id}`         | Retrieve a single task record.                  |
| `GET`    | `/api/tasks/{id}/events`  | Return the full ordered event log for replay.   |
| `DELETE` | `/api/tasks/{id}`         | Cancel a running task (no-op if already done).  |

## Persistent sessions

`cgx.sessions` is stdlib-only and stores conversation history under
`~/.cgx/sessions/` (or `$CGX_CONFIG_DIR/sessions/`):

- `index.json` — list of `SessionMeta(id, title, created_at, updated_at,
  message_count)` headers.
- `<uuid>.jsonl` — append-only message stream, one JSON object per line
  with fields `role`, `content`, `at` (unix time), `meta`.

All writes go through a temp file + `os.replace` for atomicity. The
public API (`create_session`, `append_message`, `get_messages`,
`list_sessions`, `delete_session`, `rename_session`) is what the Ask
tab calls on every interaction; failures are swallowed so chat is
never broken by a session-store I/O error.

## Rate limiting

`cgx.answer.ratelimit` adds two primitives shared by every HTTP-backed
provider:

- `RateLimiter(rate, capacity)` — token bucket guarded by a
  `threading.Lock`. `acquire()` is called before each request;
  `rate <= 0` makes the limiter a no-op so the existing call sites
  keep their pre-feature behaviour when no profile config is set.
- `request_with_retry(func, *, limiter, max_retries)` — wraps a
  callable returning a `requests.Response`. Retries on HTTP **429**
  and **5xx** using exponential backoff with jitter, honouring the
  `Retry-After` header when present.

`Profile.rate_limit` (req/sec) and `Profile.max_retries` are
serialised with the rest of the provider config so cloud profiles
keep their per-tenant budget across sessions.

## Hardware / model matrix

`cgx.answer.hardware_matrix` is a pure-data offline module:

- `LOCAL_MODEL_CATALOG` — 8 locally-runnable models with `name`,
  `params_b`, `min_ram_gb`, `recommended_vram_gb`, `ctx_window`,
  `family`, and a one-line `notes` blurb.
- `compute_local_fit(hw)` — annotates the catalogue with a verdict
  string (`✅ fits` / `⚠️ tight` / `❌ won't fit` / `❓ unknown`) and a
  `reason`. Uses an "effective budget" of
  `max(ram_gb, gpu_vram_gb * 2.0)` when a GPU is detected.
- `TRADEOFFS` — eight editorial rows comparing local vs cloud across
  privacy, marginal cost, quality ceiling, cold + warm latency,
  offline use, setup effort, and operational risk.

The data is exported as `docs/hardware_matrix.json` and documented in
`docs/hardware_matrix.md`. The Hardware tab in the UI is a thin view
on top of these two functions; no network call is ever made from the
tab.

## Telemetry

`cgx.telemetry.ping()` is invoked once from `cgx.ui.app.launch()`. It
returns immediately unless `CGX_TELEMETRY=1` is set. The opt-in
payload contains **only** a random install UUID (cached in
`~/.cgx/install_id`) and the Averix package version — no prompts, no
code, no file paths, no model names, no PII. Implementation is ~50
lines; review it before opting in.

## Observability

`setup_logging(INFO)` is called once at server startup in `launch.py`,
configuring the root logger with a timestamped formatter. Every major
operation then emits structured log lines to stdout:

| Module / layer                   | What it logs                                                             |
|----------------------------------|--------------------------------------------------------------------------|
| `cgx.webui` handlers             | Request received, SSE stream started/ended, error details.               |
| `cgx.webui.task_store`           | Task created, status transitions (`running→done/cancelled/error`).       |
| `cgx.agents.tracker`             | `task_start`, `task_done`, `task_fail` for each task in the plan.        |
| `cgx.agents.planner`             | LLM planning call dispatched, task count returned, fallback activated.   |
| `cgx.webui.sse` (SSE bridge)     | Stream opened, each event type forwarded, cancellation detected.         |

Log lines use `[INFO]` and `[WARNING]` severity and include the logger
name so they can be filtered in production with standard `logging`
configuration.

## React frontend

The React frontend (`frontend/src/`) supplements the server-side layers
with two client-side modules introduced for tab persistence:

- **`frontend/src/store/tasks.ts`** — Zustand store backed by
  `sessionStorage`. Holds the in-flight streaming state for each page:
  agent (tasks / events / phase / summary), ask (messages), plan
  (thought / planMd / diff / report), index (progress / result).
  Components read from this store on mount, so a previously running task
  is immediately visible when the user returns to a tab.

- **`frontend/src/lib/connections.ts`** — module-level
  `Map<string, SseConnection>` that owns live SSE connections outside
  the React component lifecycle. When a component unmounts (tab switch),
  the connection continues streaming and writing into the Zustand store.
  When the component remounts, it reads the accumulated state. This
  eliminates the need to re-issue requests after tab switches.

The left sidebar reads the Zustand tasks store to determine which tabs
have a running task and renders an animated spinner next to those tabs.

## Security model

- Embedder loading via `module:attr` performs `importlib.import_module`,
  which runs the target module's top-level code. Pass trusted specs only.
- File walks honour `.gitignore` patterns, default ignore globs, a 1 MB
  size cap, and skip symbolic links by default.
- API keys live in the OS keyring when available
  (`pip install -e ".[keyring]"`) and otherwise in `~/.cgx/secrets.json`
  with `0600` permissions. They are never echoed back through tool output
  or LLM transcripts.
- The VS Code extension scaffold (`extension/`) frames the Gradio UI
  in a webview with a tight CSP (`frame-src` restricted to
  `http://localhost:*` and `http://127.0.0.1:*`) and a sandboxed
  iframe; the configured `averix.ui.url` value is HTML-escaped before
  interpolation.
