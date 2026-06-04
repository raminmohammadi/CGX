# Architecture

CGX is structured as a small set of cooperating layers under `cgx.*`.

## Layers

```
cgx.parser              — language-aware tree walker → chunk records
cgx.parser.module_path  — repo-root-aware dotted-name resolver for imports
cgx.graph               — NetworkX call/containment graph over chunks
cgx.graph.aggregation   — node/edge projections consumed by retrieval + viz
cgx.embeddings          — two-view (intent / impl) corpora, FAISS indices
cgx.embeddings.loader   — shared embedder loaders (spec / model_name, lazy ML)
cgx.embeddings.cache    — content-addressed embedding cache (.npz)
cgx.retrieval           — hybrid retriever (semantic + BM25 + graph) + RRF
cgx.answer              — LLM providers, intent detection, prompt registry
cgx.answer.providers    — OllamaProvider, OpenAICompatProvider, GeminiProvider
cgx.answer.ratelimit    — token-bucket limiter + 429/5xx retry
cgx.answer.profiles     — provider config + keyring-backed secret store
cgx.answer.hardware_matrix — offline local-model catalogue + tradeoffs
cgx.answer.ollama_discovery — installed-model listing + hardware probe
cgx.codegen             — diff parse / dry-apply / syntax & test validation
cgx.codegen.ast_insert  — AST-anchored insertion planner (sibling-anchor → PatchResult)
cgx.codegen.disk_apply  — write applied diffs to disk + per-run backup mirror
cgx.codegen.env_manager — pre-flight dependency scan, pip-install, requirements update
cgx.codegen.symbol_map  — symbol-table context builder for working-memory injection
cgx.io.persist          — JSON/JSONL/FAISS writers shared by the index pipeline
cgx.pipeline            — high-level orchestrators (run_index_auto, run_query_auto)
cgx.agents              — Planner / Tracker / Judge multi-agent loop
cgx.agents.viz          — DAG + status-table renderers for the Agent tab
cgx.sessions            — append-only JSONL conversation store
cgx.telemetry           — opt-in anonymous startup ping
cgx.logging_setup       — shared setup_logging() invoked once from launch.py
cgx.webui.task_store    — SQLite task registry + threading.Event cancel tokens
cgx.webui.routes.tasks  — REST API for task list / get / event-replay / cancel
cgx.webui.routes.rollback — POST /api/rollback restores from an apply backup dir
cgx.webui.routes.setup  — discovery endpoints + POST /api/provider/ping
cgx.cli / cgx.webui     — terminal + FastAPI/React surfaces (uvicorn on :8765)
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

## Structured AST insertion

`cgx.codegen.ast_insert` provides an additive, AST-anchored alternative
to text diffs for the common "insert a new def into this container after
this sibling" case. The retrieval side already speaks in terms of
containers and sibling anchors (`suggest_insertion_points` returns
`{container_type, container_id, anchors}`), and this module bridges
that signal into the same `PatchResult` shape that
`apply_diffs_in_memory` and `validate_patch_results` consume:

1. `AstInsertSpec(rel_path, code, class_name=None, anchor_symbol=None,
   dedupe=True)` declares the target (module level or a named
   top-level class) and the snippet to splice in.
2. `plan_ast_insertion` reads the target file, parses it with the
   stdlib `ast` module, locates the anchor sibling's `end_lineno`,
   detects the container's body indentation, re-emits the snippet via
   `ast.get_source_segment` so user-supplied comments and formatting
   survive, and splices it in. The result is re-parsed before being
   returned, so a broken splice produces `ok=False` rather than a
   corrupted file. Nothing is written to disk.
3. `plan_ast_insertion_from_suggestion(project_root, suggestion, code)`
   accepts a single item from `suggest_insertion_points` directly,
   resolves the `container_id` (including the `::class::<Name>`
   suffix), and prefers the `similar_signature_neighbor` anchor over
   the `likely_caller`.
4. `build_unified_diff(patch_result)` renders the plan as a standard
   unified diff so it routes back through `parse_fenced_diffs` /
   `apply_diffs_to_disk` / `validate_patch_results` without any
   special-casing.

The module is purely additive: it does not modify any existing
signature in `diff_apply`, `validate`, `disk_apply`, or
`orchestrator`. Callers can keep using the text-diff path; the AST
path is opt-in via the entry points above.

## LLM Providers

`cgx.answer.providers` exposes four concrete `LLMProvider` subclasses,
all sharing a uniform `chat()` / `chat_stream()` interface so the
orchestration layer never knows which backend it is talking to.

| Class | Kind string | Notes |
|---|---|---|
| `OllamaProvider` | `"ollama"` | Local Ollama via `/api/chat`; JSON mode via `format: "json"`. |
| `OpenAICompatProvider` | `"openai-compat"` or `"custom"` | Any `/v1/chat/completions`-compatible endpoint. Accepts `endpoint_path` to override the path suffix and `allow_no_auth=True` to skip Bearer auth (private subnets). |
| `GeminiProvider` | `"gemini"` | Google Gemini via `generativelanguage.googleapis.com`. Maps CGX's `messages` to Gemini's `contents` + `systemInstruction`; merges consecutive same-role turns; uses `responseMimeType: "application/json"` for JSON mode; streams via `streamGenerateContent`. |

**Profile persistence**: `cgx.answer.profiles.Profile` stores `kind`,
`model`, `base_url`, `temperature`, `num_predict`, `has_api_key`,
`rate_limit`, `max_retries`, `endpoint_path`, and `allow_no_auth`.
`build_provider` in `cgx.webui.helpers` instantiates the correct class
from a profile so every code path (inline config, saved profile) goes
through one factory.

**Live connection test** (`POST /api/provider/ping`): returns
`{ok, latency_ms, error}` after hitting the provider's cheapest
available endpoint (Ollama `GET /api/tags`; Gemini `generateContent`
with `maxOutputTokens: 1`; custom `OPTIONS` / `HEAD` on the endpoint
path). Used by the Settings page Ping button.

## Dynamic Dependency Management

`cgx.codegen.env_manager` intercepts the gap between code generation and
test execution: a generated file may import a package the model chose
that is not listed in `requirements.txt`.

**Pipeline** (called inside the `verify` capability before `pytest`):

1. `scan_imports(generated_files)` — AST walk (`.py`) or regex
   (`.js`/`.ts`) to collect top-level import roots.
2. `find_missing_python_packages(imports, project_root)` — cross-references
   against `requirements.txt`; then probes live importability; skips the
   full CPython stdlib (50+ top-level names).
3. `install_packages(packages)` — `pip install --quiet` per missing
   package; records success/failure per name.
4. If tests pass, `update_requirements(project_root, installed)` appends
   new packages to `requirements.txt` idempotently.

Failures are logged but never abort the test run — the model may have
misspelled the package name, in which case pytest still runs and gives
the retry loop a real `ImportError` to diagnose.

## Symbol Table Context

`cgx.codegen.symbol_map` builds a compressed working-memory map from
the same JSONL records the retrieval layer uses, then injects it into
every `plan`-kind LLM prompt so local models stop re-implementing helpers
that already exist.

**`build_symbol_map(records_path)`** reads chunk IDs in the form
`path::kind::symbol`, normalises paths to project-relative form (anchors
on `src/`, `tests/`, `app/`, `backend/`), and returns
`{relative_path: [symbol, …]}`.

**`format_symbol_map(symbol_map)`** renders a compact block:
```
# AVAILABLE CONTEXT (Do not redefine these):
File: src/db.py -> get_connection(), close_connection()
File: src/utils.py -> hash_password(str), verify_token(str)
```
Capped at 60 files × 20 symbols each to stay within the prompt budget of
a 7B model.

**`fetch_symbol_source(records_path, symbol_name)`** is the AST-RAG
on-demand path: the retry loop calls it when a generated call site fails
the syntax smoke test (e.g. `verify_token` called with the wrong
signature) to pull the exact function source text from the records file
and inject it into the re-try prompt.

## Multi-agent loop

`cgx.agents` adds an orchestration layer on top of the single-shot
`answer_with_llm` / `generate_code_plan` / `generate_project_scaffold`
entry points:

1. **Planner** (`cgx.agents.planner.Planner`) — decomposes a user goal
   into an ordered list of atomic `Task`s. Prefers the LLM with a
   strict JSON schema (1–5 tasks, each with `name` (short title),
   `description` (imperative sentence),
   `kind ∈ {ask, plan, scaffold, scaffold_manifest, scaffold_file,
   search, summarize, apply, verify, fill_logic}`,
   and plain-English `criteria`); falls back to a deterministic
   single-task plan derived from `detect_intent` when no provider is
   available or the model returns garbage. When the LLM omits `name`,
   `_derive_name()` distils a clean title from the first sentence of
   the description. A post-validation step (`_enforce_kind_policy`)
   applies four routing rules in priority order:
   - **Scaffold goals** → always
     `[scaffold_manifest, apply, verify]`; no index required. The
     manifest capability returns a layered file list and the Tracker
     injects one `scaffold_file` task per planned file before `apply`
     runs, giving the UI per-file progress and letting each generation
     call stay focused on a single output. Detection
     accepts three independent signals (see `_goal_is_scaffold`):
     (a) the `_SCAFFOLD_RE` regex — a scaffold verb (`create`,
     `build`, `generate`, `scaffold`, `bootstrap`, `init`, …) within
     5 tokens of a project noun (`app`, `project`, `cli`, `tool`,
     `library`, `calculator`, `dashboard`, `todo`, `blog`, `game`,
     `chat`, `editor`, `tracker`, `portfolio`, `landing page`,
     `form`, `page`, `site`, `gui`, `interface`, `ui`, `bot`, etc.),
     OR explicit `from scratch` / `new <project-noun>` phrasing;
     (b) a scaffold verb paired with a framework or language name
     from `_TECH_RE` (`react`, `vue`, `angular`, `svelte`, `next.js`,
     `fastapi`, `flask`, `django`, `express`, `tkinter`, `pyqt`,
     `electron`, `streamlit`, `react native`, `flutter`, `rails`,
     `spring`, `python`, `typescript`, `rust`, `go`, `tailwind`,
     etc.) — covers prompts like *"create a calculator using React"*;
     (c) the LLM emitted at least one `scaffold` task and the goal
     has no existing-codebase hint (`_EXISTING_CODE_HINT_RE`:
     `existing`, `our app`, `legacy`, `refactor`, `modify`,
     `fix the bug`, …); (d) a scaffold verb together with at least one
     supported, non-style skill firing via `skills.detect_skills` —
     the more precise of the verb-paired signals because it only
     matches technologies CGX actually has dedicated handling for
     (see the [Skills](#skills) section). The scaffold branch always
     emits a single `scaffold_manifest` task (the per-file `scaffold_file`
     tasks are injected at runtime by the Tracker from the manifest
     output) followed by a fresh `apply` + `verify` pair. Every
     scaffold-family task receives the full original goal under
     `task.inputs["goal"]` plus a `task.inputs["skills"]` list of
     detected skill names so the scaffold capability and the Judge
     can both compose / validate against the same technology context.
     PLAN tasks for code-change goals receive the same
     `task.inputs["skills"]` attachment.
   - **Verify-only goals** ("do the tests pass?") → `[verify]`.
   - **Read-only goals** (no change verb) → any `plan` task is
     downgraded to `ask`, preventing expensive code-gen on informational
     questions.
   - **Code-change goals** → any stray `scaffold` tasks are dropped
     (we modify the existing codebase rather than recreate it), then
     `apply` + `verify` are appended after the final `plan` task.
   Each branch emits a single `[INFO]` log line
   (`Planner: kind-policy SCAFFOLD/VERIFY-ONLY/READ-ONLY/CHANGE-GOAL
   path`) so the operator can read the routing decision in the
   server terminal.
2. **Tracker** (`cgx.agents.tracker.Tracker`) — drives the plan
   task-by-task, dispatching each kind to a caller-supplied capability
   callable (`ask`, `plan`, `scaffold`, `scaffold_manifest`,
   `scaffold_file`, `search`, `summarize`, `apply`, `verify`,
   `fill_logic`). `ask`, `plan`, `scaffold`, `scaffold_manifest`,
   `scaffold_file`, `search`, and `fill_logic` receive the task
   description as their first argument; `summarize`, `apply`, and
   `verify` receive the list of all prior task outputs. `fill_logic`
   additionally reads `file_path`, `function_name`, and optional
   `skeleton` from `task.inputs`; `scaffold_file` reads `path`,
   `description`, and `layer` from `task.inputs`. Each capability
   runs in a worker thread so the loop can emit a `task_progress`
   heartbeat every `progress_interval` seconds (default `2.0`) carrying
   `{task_id, name, kind, elapsed}` — the UI uses this as a live
   "running for Ns" counter. When a `scaffold_manifest` task returns
   an `inject_tasks` list, the Tracker splices those `scaffold_file`
   tasks into the plan immediately after the manifest task so the
   downstream `apply` step sees the full file batch. After every
   successful `apply` task the Tracker updates `plan.owned_files`
   (a `dict[str, "applied"|"failed"]`) so the retry loop always knows
   which files are on disk and which still need fixing. On completion
   the Tracker invokes the Judge and emits one of `task_done` /
   `task_failed` / `task_skipped`. The full `AgentEvent` set is:
   `plan`, `task_start`, `task_progress`, `task_done`, `task_failed`,
   `task_skipped`, `judge`, `summary`.
3. **Judge** (`cgx.agents.judge.Judge`) — validates each completed task
   against its criteria. Performs cheap structural short-circuits before
   optionally asking the LLM for a strict `{verdict, confidence,
   rationale}` JSON verdict. Per-kind rules:
   - `ask`: hard-fail when `answer_md` is empty.
   - `plan`: hard-fail only when *both* `plan_md` and `diffs` are
     absent; when `plan_md` exists but `diffs` is empty (e.g. a local
     LLM that produced a narrative plan), passes to LLM judge.
   - `scaffold`: hard-fail when both `plan_md` and `diffs` are absent.
     When files are present, every active skill (resolved from
     `task.inputs["skills"]` or re-detected from the goal) runs its
     `validate_scaffold(diffs)` check via the
     [Skills](#skills) registry. The first failing `SkillVerdict`
     short-circuits to a Judge fail with the skill's rationale
     prefixed by `[<skill>]` (e.g. `[react] React skill: scaffold has
     no .jsx/.tsx/.js/.ts files`). Skills that abstain or pass let
     the Judge fall through to a structural pass — and from there to
     the SCAFFOLD short-circuit in `judge()` which skips the LLM
     grader entirely (local 3-7B judge models routinely fabricate
     criteria-based fails against scaffolds that demonstrably satisfy
     them). The artifact passed to the LLM judge (used only when
     diffs are absent but `plan_md` is present) is rendered by a
     dedicated SCAFFOLD branch of `_render_artifact`: it surfaces
     `plan_md`, the full list of generated file paths, and a per-file
     content preview (up to 6 files, each capped to keep the prompt
     small).
   - `scaffold_manifest`: hard-fail when the manifest is empty or has
     no relative paths; pass when at least one layer with one file is
     returned (LLM judge skipped — the per-file `scaffold_file` tasks
     carry their own verdicts).
   - `scaffold_file`: hard-fail when the generated file content is
     empty or only contains stubs; otherwise pass (the file's syntax
     is smoke-tested by `apply` downstream).
   - `search`: structural pass when `hits > 0` (LLM judge not invoked).
   - `apply`: fail when `failed_files` is non-empty (partial write —
     passing files are written, failing files are skipped); pass when
     `applied_files` is non-empty and `failed_files` is empty.
     `smoke_ok` in the return value is `True` only when all files passed.
     A per-run backup directory under `<project_root>/.cgx-backups/`
     is created before the first overwrite and returned as
     `backup_dir`; the rollback REST route restores from it on demand.
   - `verify`: trusts pytest exit code directly; soft-pass on "no
     impacted tests".

The high-level `cgx.agents.run_agent(goal, …, progress_interval=2.0)`
wires all three to the default capabilities backed by the existing
engine, and is exposed via the **🤖 Agent** tab in the React UI. See
[flowcharts.md](flowcharts.md) for a visual breakdown of the loop and
the event timeline.

### Retry loop

`_stream_with_retry` in `cgx.agents.loop` handles all failure paths in
priority order and recurses up to `max_retries` times:

1. **Verify failures** — test stdout/stderr is parsed by `_diagnose_failure`
   to classify the error type (`import_error`, `syntax_error`,
   `logic_error`, `unknown`) and extract responsible files from
   tracebacks. `_build_fix_goal` then emits a *targeted* re-plan goal
   that names exactly the broken files and instructs the LLM not to
   touch the files that are already correct (read from `plan.owned_files`).
   When a Python test imports a JS/JSX module (e.g. `from src.App import
   calculateResult` where `src/App.jsx` exists), the diagnosis detects the
   language mismatch and the fix goal explicitly offers two remediation
   paths: create a Python backend module, or replace the test with a
   JS test.

   **10-line buffer rule (Phase 4)**: `_extract_error_snippet` locates
   the first line-number reference in the traceback, opens the failing
   file, and extracts lines `[lineno−5 … lineno+5]` with an
   `# <-- ERROR HERE` annotation. `_build_fix_goal` embeds this snippet
   as a focused `` ```python `` block instead of dumping the full log,
   keeping the prompt tight enough for 7B models to act on it precisely.

2. **Apply failures** — when the smoke check or cross-file coherence
   check rejects generated files, the failing-file list is forwarded to
   `_build_apply_fix_goal` which tells the LLM to regenerate only those
   files with valid syntax.  **Passing files are already on disk** so
   nothing already correct is lost. Apply failures trigger a recursive
   retry.
3. **Scaffold / plan generation failures** — Judge rejections on the
   code-generation step trigger `_build_core_fix_goal`.

### Cross-file coherence check

`cgx.codegen.validate.check_cross_file_coherence` runs as part of the
`apply_diffs_to_disk` smoke-test step (before anything is written).  It
walks every `.py` file in the patch batch, parses its `import` statements
via `ast`, and flags any `from X.Y import Z` where `X/Y.jsx`, `X/Y.tsx`,
`X/Y.js`, or `X/Y.ts` is present in the same batch or on disk under
`project_root`.  A flagged import causes that Python file to be added to
`failed_files`, triggering the retry loop with a language-mismatch
diagnosis.

### Partial apply

`apply_diffs_to_disk` previously rejected the entire batch if any file
failed the smoke check.  It now writes files that pass validation and
records the ones that failed in `failed_files`.  `smoke_ok` is `True`
only when every file passed.  This means a retry only needs to
regenerate the failing file(s) — the correct files are already on disk.

The **SSE bridge** (`cgx.webui.sse.bridge_generator`) records every
emitted `AgentEvent` into the task registry (`cgx.webui.task_store`) so
the frontend can replay the full event log when the user switches back to
the Agent tab. The bridge also accepts a `cancel_event: threading.Event`;
when it is set (e.g. via `DELETE /api/tasks/{id}`) the generator
terminates cleanly between yields.

## Skills

The `skills/` package at the repo root holds modular technology-specific
knowledge bundles. Each skill lives in its own folder under
`skills/<name>/` (e.g. `skills/react/`, `skills/fastapi/`) and
implements the `skills.base.Skill` protocol with four orthogonal
responsibilities:

| Method | Purpose |
|--------|---------|
| `detect(goal) -> float`              | Return `[0.0, 1.0]` confidence that *goal* involves this technology. Scores at or above `SKILL_DETECT_THRESHOLD` (0.5) activate the skill. |
| `scaffold_system_prompt() -> str`    | Prompt fragment appended to `_SCAFFOLD_SYSTEM` when generating a brand-new project that uses this technology. |
| `plan_system_prompt() -> str`        | Prompt fragment appended to the plan-time system prompt for code-change goals. |
| `validate_scaffold(diffs, goal)`     | Inspect the produced diffs and return a `SkillVerdict` (or `None` for "no opinion"). Drives the Judge's structural pass/fail. |
| `validate_plan(diffs, goal)`         | Same shape as `validate_scaffold` but for plan tasks. |

**Registry**: `skills/__init__.py` declares the `SKILLS` list (one
instance per registered skill) and exposes the dispatchers
`detect_skills(goal)`, `compose_scaffold_prompt(active)`,
`compose_plan_prompt(active)`, `validate_scaffold(active, diffs)`,
`validate_plan(active, diffs)`, and `skills_by_names(names)`. The
initial bundle covers React, Next.js, Vue, Tailwind, FastAPI, Flask,
Django, Express, Python CLI, and SQLite.

**Wiring**:

- `cgx.agents.planner` calls `detect_skills(goal)` and attaches the
  resulting name list to every SCAFFOLD and PLAN task's
  `task.inputs["skills"]`. The planner also uses skill detection as a
  secondary scaffold-routing signal (verb + supported skill → SCAFFOLD)
  alongside the broader regex (`_TECH_RE`) that keeps coverage for
  unsupported frameworks.
- `cgx.answer.engine.generate_project_scaffold` /
  `generate_code_plan` accept a `skills=` kwarg. They resolve the
  Planner-attached names back to instances (or re-detect from the
  goal), then concatenate `compose_*_prompt(active)` onto the base
  system prompt with an `ACTIVE SKILLS:` header.
- `cgx.agents.judge._structural_check` runs `validate_scaffold(active,
  diffs)` for SCAFFOLD tasks and `validate_plan(active, diffs)` for
  PLAN tasks. A failing verdict is converted to `Verdict(verdict=
  "fail")` with the rationale prefixed by `[<skill>]` so the operator
  can see which skill rejected the artifact. A passing or abstaining
  verdict falls through to the Judge's existing logic.

Adding a new skill: create `skills/<name>/__init__.py` with a single
`Skill` subclass, import it from `skills/__init__.py`, and append an
instance to `SKILLS`. No agent-layer changes are required.

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

## Apply rollback

`cgx.codegen.disk_apply.apply_diffs_to_disk` mirrors every file it is
about to overwrite into a timestamped directory under
`<project_root>/.cgx-backups/<run_id>/` before writing. The path is
returned as `output["backup_dir"]` on the `apply` task and surfaced in
the UI as an **Undo** button.

`cgx.webui.routes.rollback` exposes `POST /api/rollback` which accepts
`{project_root, backup_dir}` and calls
`cgx.codegen.disk_apply.rollback_from_backup` to restore originals and
delete files that did not exist before the run. The response is
`{restored_files, deleted_files, failed_files, error}`.

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

`cgx.telemetry.ping()` is invoked once from `cgx.webui.launch.launch()`. It
returns immediately unless `CGX_TELEMETRY=1` is set. The opt-in
payload contains **only** a random install UUID (cached in
`~/.cgx/install_id`) and the CGX package version — no prompts, no
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
| `cgx.agents.planner`             | LLM planning call dispatched, task count returned, fallback activated, kind-policy routing branch (`SCAFFOLD` / `VERIFY-ONLY` / `READ-ONLY` / `CHANGE-GOAL`). |
| `cgx.agents.judge`               | Structural verdict per task; LLM-judge invocation outcome.               |
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
- The VS Code extension scaffold (`extension/`) frames the CGX web UI
  in a webview with a tight CSP (`frame-src` restricted to
  `http://localhost:*` and `http://127.0.0.1:*`) and a sandboxed
  iframe; the configured `cgx.ui.url` value is HTML-escaped before
  interpolation.
