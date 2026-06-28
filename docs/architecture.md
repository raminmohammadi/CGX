# Architecture

CGX is structured as a small set of cooperating layers under `cgx.*`.

## Layers

```
cgx.parser              -- language-aware tree walker → chunk records
cgx.parser.schema       -- TypedDicts pinning the chunk + call-relation shapes
cgx.parser.base         -- BaseParser ABC; the per-language seam
cgx.parser.python_parser -- PythonASTParser; the only registered parser today
cgx.parser.module_path  -- repo-root-aware dotted-name resolver for imports
cgx.graph               -- NetworkX call/containment graph over chunks
cgx.graph.aggregation   -- node/edge projections consumed by retrieval + viz
cgx.graph.backend       -- CodeGraphBackend facade isolating retrieval from nx
cgx.embeddings          -- two-view (intent / impl) corpora, FAISS indices
cgx.embeddings.loader   -- shared embedder loaders (spec / model_name, lazy ML)
cgx.embeddings.cache    -- content-addressed embedding cache (.npz)
cgx.retrieval           -- hybrid retriever (semantic + BM25 + graph) + RRF
cgx.retrieval.tokenize  -- symmetric sub-word tokenizer (camelCase / snake_case)
cgx.answer              -- LLM providers, intent detection, prompt registry
cgx.answer.context_map  -- tiered ("Code Map") SOURCES builder for the prompt
cgx.answer.model_caps   -- model-window-aware budgets consumed by context_map
cgx.answer.providers    -- OllamaProvider, OpenAICompatProvider, GeminiProvider
cgx.answer.ratelimit    -- token-bucket limiter + 429/5xx retry
cgx.answer.profiles     -- provider config + keyring-backed secret store
cgx.answer.hardware_matrix -- offline local-model catalogue + tradeoffs
cgx.answer.ollama_discovery -- installed-model listing + hardware probe
cgx.codegen             -- diff parse / dry-apply / syntax & test validation
cgx.codegen.ast_insert  -- AST-anchored insertion planner (sibling-anchor → PatchResult)
cgx.codegen.disk_apply  -- write applied diffs to disk + per-run backup mirror
cgx.codegen.env_manager -- pre-flight dependency scan, pip-install, requirements update
cgx.codegen.symbol_map  -- symbol-table context builder for working-memory injection
cgx.io.persist          -- JSON/JSONL/FAISS writers shared by the index pipeline
cgx.pipeline            -- high-level orchestrators (run_index_auto, run_query_auto)
cgx.agents              -- Planner / Tracker / Judge multi-agent loop (legacy /agent-legacy)
cgx.agents.viz          -- DAG + status-table renderers for the Agent tab
cgx.session             -- session-shaped agent backbone (default /agent)
cgx.session.models      -- Session, TaskNode, Fact, Artifact, Decision dataclasses
cgx.session.router      -- deterministic state machine (TASK_SUCCESSOR table)
cgx.session.runner      -- SessionRunner: per-session lock, executor dispatch
cgx.session.store       -- SQLite persistence at <project_root>/.cgx/sessions.db
cgx.session.mode        -- detect_mode: explore vs greenfield auto-detection
cgx.session.tasks       -- registered executors (EXPLORE/INVESTIGATE/RECOMMEND/CLARIFY_REQUIREMENTS/DECOMPOSE/SCAFFOLD/BOOTSTRAP_ENV/APPLY/VERIFY/REPAIR/...)
cgx.session.repair      -- deterministic, LLM-free classifier / locator / proposer for the REPAIR executor
cgx.sessions            -- append-only JSONL conversation store (Ask tab history)
cgx.telemetry           -- opt-in anonymous startup ping
cgx.logging_setup       -- shared setup_logging() invoked once from launch.py
cgx.webui.task_store    -- SQLite task registry + threading.Event cancel tokens
cgx.webui.routes.tasks  -- REST API for task list / get / event-replay / cancel
cgx.webui.routes.rollback -- POST /api/rollback restores from an apply backup dir
cgx.webui.routes.setup  -- discovery endpoints + POST /api/provider/ping
cgx.webui.routes.agent_session -- /api/agent-session/* JSON routes for the session backbone
cgx.cli / cgx.webui     -- terminal + FastAPI/React surfaces (uvicorn on :8765)
```

## Data flow

1. `parse_codebase` walks the repo (respecting `.gitignore`, ignore globs
   and a 1 MB file-size cap) and emits one chunk per file/class/function.
2. `build_knowledge_graph` derives a NetworkX graph with `calls`, `module`,
   `attr` and `defined_in` edges.
3. `make_index_records` materialises chunk records, then
   `prepare_embedding_corpus` builds two views:
   - **intent** -- NL-friendly summary (docstrings, names).
   - **impl** -- implementation text (signatures + bodies).
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
   When the retriever surfaced graph-expanded neighbors the prompt
   SOURCES are built by `cgx.answer.context_map.build_tiered_context`
   (see [Tiered SLM context (Code Map)](#tiered-slm-context-code-map))
   so neighbor chunks contribute compact stubs instead of full bodies.
7. `generate_code_plan` does the same but routes through a
   diff-aware retry loop in `cgx.codegen.pipeline.validate_and_test`.

## Tiered SLM context (Code Map)

`cgx.answer.context_map` builds the prompt-time SOURCES list as **two
tiers** so that the small / mid-sized LLMs CGX targets locally
(3B–8B Ollama models, etc.) don't have their entire context window
consumed by graph-expanded neighbors that only need to be *named*, not
*read*. The retriever already labels neighbors via
`HybridRetriever._expand_multi_hop`, which writes
`provenance.graph_depth = d` (1, 2, …) on every hit that was pulled in
by walking the call/import graph from a primary seed; everything else
is depth `0`.

**Classification rule** (`classify_hits`): a hit with
`provenance.graph_depth >= 1` goes into the `neighbor` tier; anything
else (semantic match, BM25 match, symbol-boosted seed) is `primary`.

**Tier shapes**

- **Primary** sources reuse `engine._as_sources_with_meta` unchanged
  and carry the focus-windowed code body. They get the larger per-chunk
  char cap so the LLM has enough text to ground its answer or its
  diff.
- **Neighbor** sources are rendered by `format_neighbor_stub` as
  ``[class.]name(signature) -- doc_first_sentence``. Each component is
  dropped silently when the record doesn't supply it (e.g. methods
  prepend their class, free functions don't). The signature, first-doc
  sentence, and parent class name are pulled from `records.jsonl` via
  `load_records_by_id(records_path)`. If the record carries no
  enrichment at all the stub falls back to a trimmed view-text slice
  so the neighbor stays visible.

Each emitted source dict carries the same keys as the legacy
`_as_sources_with_meta` output (`chunk_id`, `path`, `kind`, `symbol`,
`signature`, `start_line`, `end_line`, `parent_class_id`, `text`,
`hit_meta`) **plus** a `tier` field of either `"primary"` or
`"neighbor"`. `engine._fmt_source` consults `tier` and, when it equals
`neighbor`, appends `tier=neighbor` to the bracketed metadata line so
the LLM sees an explicit hint that the block is a structural reference,
not the focal body.

**Budget** (`cgx.answer.model_caps.get_context_map_budget`)

`get_context_map_budget(provider)` returns a five-key dict scaled by
the provider's model context window. The same window-resolution logic
already used by `get_summary_budget` (small / mid / large / huge tiers
at 16K / 64K / 200K boundaries) drives a separate, more generous set
of numbers tuned for SOURCES rather than summary prose:

| Window           | `primary_chars` | `neighbor_chars` | `primary_max` | `neighbor_max` | `total_chars` |
|------------------|-----------------|------------------|---------------|----------------|---------------|
| < 16 K (small)   | 900             | 220              | 8             | 12             | 6 000         |
| < 64 K (mid)     | 1 400           | 320              | 12            | 24             | 18 000        |
| < 200 K (large)  | 2 200           | 420              | 20            | 40             | 48 000        |
| ≥ 200 K (huge)   | 3 500           | 520              | 32            | 60             | 120 000       |

The five keys are consumed exactly once each by `build_tiered_context`:
`primary_chars` and `primary_max` go through `_as_sources_with_meta`
(per-chunk char window, total chunk count); `neighbor_chars` and
`neighbor_max` cap the stub length and the neighbor count respectively;
`total_chars` is enforced as a deterministic global ceiling -- the
builder walks the ordered list primary-first and drops trailing items
once the cumulative `text` length would exceed the cap (the first item
is always kept). Call sites never hard-code budget numbers; they always
go through `get_context_map_budget(provider)`.

**Ordering** is fixed: primary sources first (in retrieval order), then
neighbor stubs. This makes the prompt deterministic for a given hit
list + provider pair and keeps citation indices stable across reruns.

**Wiring**

`cgx.answer.engine` activates the Code Map opportunistically -- only
when the retriever actually surfaced a neighbor. Both entry points
share the same gate:

```python
_has_neighbors = any(
    int(((h.get("provenance") or {}) if isinstance(h, dict) else {})
        .get("graph_depth", 0) or 0) >= 1
    for h in hits
)
if _has_neighbors:
    from cgx.answer.context_map import build_tiered_context, load_records_by_id
    from cgx.answer.model_caps import get_context_map_budget
    sources = build_tiered_context(
        hits, cmap, load_records_by_id(records_path),
        budget=get_context_map_budget(provider),
        focus_terms=focus_terms or None,
    )
else:
    sources = _as_sources_with_meta(hits, cmap, max_chunks=…, max_chars=…)
```

`answer_with_llm` runs this branch in the SOURCES-build phase (engine.py
around line 730), and `generate_code_plan` runs the same branch
unchanged (engine.py around line 915). When `_has_neighbors` is false
the legacy single-tier builder is used, so questions whose hit list
contains no graph-expanded chunks behave bit-identically to the
pre-Phase-3 prompt format.

**Public API** (`cgx.answer.context_map`)

| Symbol                                        | Purpose |
|-----------------------------------------------|---------|
| `load_records_by_id(records_path)`            | Read `records.jsonl` and key it by `id`. Returns `{}` on missing / unreadable file (the caller treats absence as "no enrichment"). |
| `classify_hits(hits)`                         | Split a hit list into `(primary, neighbors)` using the `graph_depth >= 1` rule. |
| `format_neighbor_stub(record, symbol)`        | Compose the `[class.]name(signature) -- doc_first_sentence` string with silent drop of missing parts. Pure: no I/O. |
| `build_tiered_context(hits, cmap, records_by_id, *, budget, focus_terms=None)` | The full builder. Returns the final source-dict list, each carrying a `tier` key. |

The module owns a lazy import of `engine._as_sources_with_meta` to
avoid a load-time cycle (`engine` imports `context_map`, and
`context_map` reuses the legacy primary-tier formatter from `engine`).

**Tests**: `tests/test_context_map.py` exercises the classifier rule,
the stub formatter (with and without each enrichment field), the
budget contract (per-chunk char clipping, primary/neighbor caps,
total-chars ceiling, ordering), and the engine-level branch
(deterministic switch on `_has_neighbors`).

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
`rate_limit`, `max_retries`, `endpoint_path`, `allow_no_auth`, and
`enable_reranker`. `build_provider` in `cgx.webui.helpers` instantiates
the correct class from a profile so every code path (inline config,
saved profile) goes through one factory.

**Reranker policy** (`enable_reranker`): an `Optional[bool]` whose
`None` value means "auto" and resolves through
`default_reranker_for_kind(kind)` -- cloud kinds (`openai-compat`,
`gemini`) opt in by default, local / private kinds (`ollama`, `custom`)
opt out. Explicit `True` / `False` on the profile wins.
`resolve_enable_reranker(profile)` is the single helper that returns
the effective flag. The value threads from `Profile` →
`cgx.pipeline.auto.run_query_auto(enable_reranker=…)` →
`cgx.retrieval.orchestrator.hybrid_retrieve_two_view(enable_reranker=…,
reranker_model=…, reranker_top_n=…, reranker_weight=…)` →
`HybridConfig`. `None` at the threading layer preserves the
`HybridConfig` defaults (reranker off), so existing callers that don't
pass the flag keep their pre-feature behaviour.

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

1. `scan_imports(generated_files)` -- AST walk (`.py`) or regex
   (`.js`/`.ts`) to collect top-level import roots.
2. `find_missing_python_packages(imports, project_root)` -- cross-references
   against `requirements.txt`; then probes live importability; skips the
   full CPython stdlib (50+ top-level names).
3. `install_packages(packages)` -- `pip install --quiet` per missing
   package; records success/failure per name.
4. If tests pass, `update_requirements(project_root, installed)` appends
   new packages to `requirements.txt` idempotently.

Failures are logged but never abort the test run -- the model may have
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

## Session-shaped agent (`cgx.session`)

The default Agent UI at `/agent` is backed by a **stateful, session-based**
orchestrator that is independent of the batch `cgx.agents` loop below.
Where the batch loop commits a full plan up front and streams it to
completion, the session shape persists a DAG of typed tasks under
`<project_root>/.cgx/sessions.db` and progresses one task at a time
with structured human-in-the-loop checkpoints. The two shapes share
the same retrieval, codegen, and provider stacks; they differ in the
state model, the interaction model, and the execution model.

### Session modes (`cgx.session.mode`)

A session runs in one of two **modes** -- `explore` (modify an
existing codebase) or `greenfield` (scaffold a new project from
scratch). The mode is fixed at session creation and dictates which
root task the router seeds and which executor branch the runner
walks. `cgx.session.mode.detect_mode(project_root)` is the
deterministic auto-detector; callers can override it explicitly via
the `POST /api/agent-session` body.

* **`explore`** -- the project root exists, is non-empty, and has a
  usable FAISS index under `<root>/cgx_index/meta.json`. The session
  seeds `EXPLORE` and walks the retrieval-grounded loop below.
* **`greenfield`** -- the project root is missing, empty, or has no
  index. The session seeds `CLARIFY_REQUIREMENTS` and walks the
  scaffold loop (`clarify → decompose → scaffold → apply → verify`).
  No retrieval is performed; everything is goal-driven.

The detector prefers `greenfield` whenever the project signals are
ambiguous, on the principle that mis-seeding `EXPLORE` against a
missing index crashes immediately, whereas mis-seeding
`CLARIFY_REQUIREMENTS` against an existing codebase still produces
useful clarification questions.

### Data model (`cgx.session.models`)

Plain :mod:`dataclasses` (no Pydantic at the core layer -- Pydantic
stays at the webui wire boundary). JSON-serialise via `to_dict()`,
matching the convention already used by `cgx.agents.types` and
`cgx.sessions`.

| Type           | Purpose |
|----------------|---------|
| `Session`      | Root aggregate: `original_objective`, `project_root`, `root_task_id`, `status`, timestamps. |
| `TaskNode`     | One node in the per-session DAG. Carries `kind`, `name`, `description`, `parent_task_id`, `status`, `inputs`, `outputs`, `produced_artifact_id`, `consumed_decision_ids`, `error`, lifecycle timestamps. |
| `Fact`         | Append-only piece of session knowledge (`FILE` / `SYMBOL` / `PARAMETER` / `ANCHOR`). Updates set `stale=True` rather than mutating `content`. |
| `Artifact`     | Typed output produced by a finished task. Explore-mode kinds: `DIRECTIONS_LIST`, `FINDINGS_BUNDLE`, `RECOMMENDATION_LIST`, `CODE_CHANGE_PLAN`. Greenfield-mode kinds: `REQUIREMENTS_SHEET`, `WORK_PLAN`, `SCAFFOLD_PATCHES`, `BUILD_REPORT`, `REPAIR_PLAN`. Shared write-loop kinds: `APPLIED_CHANGES`, `VERIFY_REPORT`, `SESSION_DIGEST`. |
| `Decision`     | Structured record of a user choice resolving an `ASK_USER`. Downstream tasks reference decisions by `decision_id`. |
| `KnowledgeBase` / `DecisionLog` | Per-session views over the facts and decisions tables. |

`TaskKind` values:

* Explore loop: `EXPLORE`, `INVESTIGATE`, `RECOMMEND`, `PLAN_CHANGE`.
* Greenfield loop: `CLARIFY_REQUIREMENTS`, `DECOMPOSE`, `SCAFFOLD`,
  `BOOTSTRAP_ENV`, `REPAIR`.
* Shared: `APPLY`, `VERIFY`, `ASK_USER`, plus utility kinds
  `SEARCH` / `SUMMARIZE`.

`TaskNodeStatus` runs through
`PENDING → BLOCKED → READY → IN_PROGRESS → DONE`/`FAILED`/`ABANDONED`;
`ASK_USER` deliberately stays `IN_PROGRESS` after its executor runs
until a `Decision` arrives.

### Router (`cgx.session.router`)

`Router` replaces `Planner.plan` from the batch loop. **Pure Python,
no LLM calls, no I/O**: every method takes the current session state
plus an event and returns a `RouterPlan` of typed actions
(`CreateTask`, `UpdateTaskStatus`, `RecordDecision`,
`AttachDecisionToTask`) that the caller applies to the store.

Three entry points cover every transition:

* `on_user_message(session, message, tasks)` -- no tasks yet → spawn
  the root task. In explore mode this is `EXPLORE`; in greenfield
  mode it is `CLARIFY_REQUIREMENTS`. Pending `ASK_USER` open →
  return empty plan so the caller routes the message to
  `on_decision_recorded` instead. Otherwise (explore mode) → spawn
  a sibling `EXPLORE` under the current root (course-correction
  objective). Greenfield mode does not spawn course-correction
  siblings: a follow-up message is treated as freeform context.
* `on_task_completed(session, completed, tasks)` -- dispatch via the
  `TASK_SUCCESSOR` table:

  ```
  # Explore loop
  EXPLORE              -> ASK_USER(choose_path)
  INVESTIGATE          -> RECOMMEND
  RECOMMEND            -> ASK_USER(choose_recommendation)
  PLAN_CHANGE          -> ASK_USER(approve)

  # Greenfield loop
  CLARIFY_REQUIREMENTS -> ASK_USER(clarify_answers)
  DECOMPOSE            -> ASK_USER(approve_plan)
  SCAFFOLD             -> APPLY (mode=greenfield in inputs)
  APPLY (greenfield)   -> BOOTSTRAP_ENV (threads apply / scaffold ids)
  BOOTSTRAP_ENV        -> VERIFY (threads build_artifact_id)

  # Shared write-loop tail (explore mode keeps the direct edge)
  APPLY (explore)      -> VERIFY
  VERIFY (passed)      -> (terminal)

  # Autonomous repair loop (greenfield only)
  VERIFY (assertions_failed | collection_error)
                       -> REPAIR  (when repair_attempt < 2 and
                                   failure_signature not in prior_failure_signatures)
  REPAIR (can_apply)   -> APPLY  (carries build_artifact_id forward
                                   so BOOTSTRAP_ENV is skipped)
  REPAIR (empty plan)  -> ASK_USER(freeform)
  APPLY (repair)       -> VERIFY (no BOOTSTRAP_ENV)
  ```
* `on_decision_recorded(session, decision, tasks)` -- record the
  decision, attach it to the resolved `ASK_USER`, mark the `ASK_USER`
  `DONE`, and spawn the typed successor implied by `decision.kind` +
  `decision.chosen`:

  | `DecisionKind`            | Successor task |
  |---------------------------|----------------|
  | `CHOOSE_PATH`             | `INVESTIGATE` anchored on `chosen.anchor_chunk_id` |
  | `CHOOSE_RECOMMENDATION` `kind=investigate_more` | `INVESTIGATE` on the recommended anchor |
  | `CHOOSE_RECOMMENDATION` `kind=plan_change` | `PLAN_CHANGE` carrying the full recommendation |
  | `CHOOSE_RECOMMENDATION` `kind=ask_followup` | Freeform `ASK_USER` |
  | `CHOOSE_RECOMMENDATION` `kind=done` | No successor (focus closes; a follow-up message spawns a fresh sibling EXPLORE) |
  | `APPROVE` `approved=true` | `APPLY` against the staged `plan_artifact_id` |
  | `APPROVE` `approved=false` | No successor (user can pivot via a fresh objective) |
  | `CLARIFY_ANSWERS`         | `DECOMPOSE` carrying the answers + the prior goal |
  | `APPROVE_PLAN` `approved=true` | `SCAFFOLD` against the approved `WORK_PLAN` artifact |
  | `APPROVE_PLAN` `approved=false` | No successor (loop halts; user can restart with a new objective) |
  | `FREEFORM`                 | None (handled as a new user message by the caller) |

### Runner (`cgx.session.runner`)

`SessionRunner` is the orchestrator the HTTP routes call. It sits
between the deterministic `Router` (state transitions, no IO) and the
`SessionStore` (persistence, no business logic). All write paths
funnel through it so a single sequencer enforces:

* **Router plans applied in order**: creates and decision records
  happen before status updates so a spawned child is visible to
  subscribers by the time a parent flips to `DONE`.
* **Per-session locking** (`Dict[str, threading.Lock]` guarded by an
  outer lock) so concurrent requests for the same session can't
  interleave half-applied plans.
* **Centralised executor dispatch + failure handling**: missing
  executor → `LookupError` → task transitions to `FAILED` with a
  helpful message; uncaught executor exception → same path with the
  exception class + message. Facts surfaced before the error are
  still persisted.

Public API: `start_session(objective, project_root, title)`,
`post_message(session_id, message)`, `post_decision(session_id,
decision)`, `run_next(session_id, deps)`. Routes never touch the
router or the store directly.

### Executors (`cgx.session.tasks`)

An **executor** is a pure function `(TaskNode, ExecutorDeps) ->
ExecutorResult` registered via `@register_executor(TaskKind.X)` in
`cgx.session.tasks.base`. Each kind has at most one executor;
importing the `cgx.session.tasks` package side-effect-registers them
all. Executors do **not** write to the store directly -- the runner
persists their outputs (`task.outputs`), facts (`store.add_fact`),
and artifacts (`store.save_artifact` + `task.produced_artifact_id`)
after the call. This keeps executors easy to unit-test without a
database and gives the runner a single place to enforce ordering.

Concrete executors:

| Module                              | Produces |
|-------------------------------------|----------|
| `tasks/explore.py`                  | `DIRECTIONS_LIST` artifact + `ANCHOR` facts (one per option). Bypasses `answer_with_llm`'s Markdown round-trip by reading the structured `debug["options"]` field directly. *(explore mode)* |
| `tasks/investigate.py`              | `FINDINGS_BUNDLE` artifact anchored on the chosen chunk. *(explore mode)* |
| `tasks/recommend.py`                | `RECOMMENDATION_LIST` artifact with typed `kind` per recommendation (`investigate_more` / `plan_change` / `ask_followup` / `done`). *(explore mode)* |
| `tasks/plan_change.py`              | `CODE_CHANGE_PLAN` artifact via `generate_code_plan`. *(explore mode)* |
| `tasks/clarify_requirements.py`     | `REQUIREMENTS_SHEET` artifact: 3–6 clarification questions (LLM-emitted with a deterministic fallback bank). *(greenfield mode)* |
| `tasks/decompose.py`                | `WORK_PLAN` artifact (`plan_md` + layered file manifest) via `cgx.answer.engine.plan_scaffold_manifest`, with the user's clarify answers folded into the goal text. *(greenfield mode)* |
| `tasks/scaffold.py`                 | `SCAFFOLD_PATCHES` artifact: walks the `WORK_PLAN` layers, calls `cgx.answer.engine.generate_single_scaffold_file` per entry while accumulating sibling-file context, captures per-file failures into a `failed` list rather than aborting. *(greenfield mode)* |
| `tasks/apply.py`                    | `APPLIED_CHANGES` artifact via `apply_diffs_to_disk` (same `backup_dir` mechanic as the batch loop). Accepts either `CODE_CHANGE_PLAN` (explore) or `SCAFFOLD_PATCHES` (greenfield) as the upstream artifact. |
| `tasks/bootstrap_env.py`            | `BUILD_REPORT` artifact: detects project type, calls `cgx.codegen.test_runner.ensure_project_venv` to create/refresh `.venv` and install declared deps, then `cgx.codegen.env_manager.preflight_install` for undeclared imports (successful adds are appended back to `requirements.txt` via `update_requirements`). After preflight, runs `cgx.session.repair.locate.lint_test_style` over the applied test files (paths starting with `tests/` or basenames starting with `test_`) and attaches the result as a `style_issues` list (`{kind, file, class_name, lineno, helpers}`) on the artifact; the lint is informational and does not change the outcome -- it names the issue ahead of `VERIFY` so the UI can surface it before REPAIR auto-fixes. Surfaces an `outcome` token (`succeeded` / `failed` / `no_venv` / `skipped` / `partial`) plus `python_exe` for VERIFY to consume. *(greenfield mode)* |
| `tasks/verify.py`                   | `VERIFY_REPORT` artifact via the impacted-tests runner. Reads `python_exe` from the upstream `BUILD_REPORT` (when present) so pytest runs inside the project venv; classifies pytest's exit code into an `outcome` token (`passed` / `assertions_failed` / `collection_error` / `no_tests_collected` / `timeout` / `pytest_missing` / `skipped`) so environment failures are distinguishable from real assertion failures. Also computes and stores a `failure_signature` (sha1 of outcome + returncode + first error line, truncated) so the router's progress detector can compare attempts without re-reading the artifact. In greenfield mode, "no tests discovered yet" reports `ran=False` + `skipped_reason` instead of failing. |
| `tasks/repair.py`                   | `REPAIR_PLAN` artifact: reads the upstream `VERIFY_REPORT`, classifies the failure via `cgx.session.repair.classify` (deterministic, LLM-free; v1 covers `unittest_pytest_mix`, `missing_module_pythonpath`, and `missing_fixture`), locates offending classes/modules/fixtures via `cgx.session.repair.locate` (AST scan + project-root resolution), and emits unified diffs via `cgx.session.repair.propose` shaped for the shared APPLY executor. The `missing_module_pythonpath` proposer creates (or prepends to) a project-root `conftest.py` carrying a marker comment + `sys.path.insert(0, str(Path(__file__).parent))`. The `missing_fixture` proposer scans the tree (skipping `.venv` / cache / build dirs) for an `@pytest.fixture`-decorated function matching the missing name and hoists its verbatim source span into `tests/conftest.py` (or root `conftest.py` when no `tests/` dir exists), gated by a `# cgx-repair: missing_fixture <name>` marker. Content carries `classification`, `failure_signature`, `repair_attempt`, `rationale`, `locations`, and `diffs`. Empty diffs (classification `unknown`, or proposer marker already present) escalate via the router to `ASK_USER(freeform)`. *(greenfield mode)* |
| `tasks/ask.py`                      | Pseudo-executor: surfaces the question payload; the runner keeps the task at `IN_PROGRESS` until `build_decision` consumes a user reply. |

`ExecutorDeps` carries optional `project_root`, `index_dir`,
`records_path`, `embed_model`, `provider`, `store`, and an `extra`
dict. Executors validate the fields they need and return
`ExecutorResult(failure=...)` if a required dep is missing rather
than raising.

### Decision contract (`cgx.session.tasks.ask.build_decision`)

The HTTP route layer calls `build_decision(session_id, task, chosen,
rationale)` on every incoming decision; the function validates
`chosen` against `task.inputs["expected_kind"]` and raises
`ValueError` (rendered as `400`) on mismatch. The validated shapes
are:

| `expected_kind`         | Required slots in `chosen` |
|-------------------------|----------------------------|
| `choose_path`           | `anchor_chunk_id` (non-empty) |
| `choose_recommendation` | `kind ∈ {investigate_more, plan_change, ask_followup, done}`; `kind=investigate_more` additionally requires `anchor_chunk_id` |
| `approve`               | `approved` (boolean) |
| `clarify_answers`       | `answers` (non-empty dict keyed by question id) |
| `approve_plan`          | `approved` (boolean) |
| `freeform`              | (no required slots) |

The frontend's per-form components in
`frontend/src/components/agent/AskUserForm.tsx` post exactly these
shapes, and `tests/test_webui_agent_session.py` pins the contract
end-to-end against the route handlers.

### Persistence (`cgx.session.store`)

`SessionStore` is a thin SQLite wrapper. One database file per
project root at `<project_root>/.cgx/sessions.db` (or
`~/.cgx/sessions.db` when no project root is provided -- typical for
interactive scripts and tests with a tmp `HOME`). Tables: `sessions`,
`tasks`, `facts`, `decisions`, `artifacts`. Each row stores the
dataclass as a JSON blob plus a few indexed columns (session_id,
status, timestamps) so common queries don't have to parse JSON.
Connections use WAL mode (`PRAGMA journal_mode=WAL`) for concurrent
reader tolerance; `PRAGMA foreign_keys=ON` cascades deletes.

Writes funnel through `store.publish` so the in-process `EventBus`
stays in sync with the on-disk state; the bus is what a future
streaming surface will subscribe to.

### HTTP surface (`cgx.webui.routes.agent_session`)

JSON-only, mounted at `/api/agent-session` next to the legacy SSE
route at `/api/agent` -- it does **not** replace it.

| Method | Path                                  | Purpose |
|--------|---------------------------------------|---------|
| `POST` | `/api/agent-session`                  | Create a session, seed the root task (`EXPLORE` in explore mode, `CLARIFY_REQUIREMENTS` in greenfield mode), optionally drain READY tasks. Accepts an optional `mode: "explore" | "greenfield"`; falls back to `cgx.session.mode.detect_mode` when absent. Returns the full snapshot. |
| `GET`  | `/api/agent-session?project_root=...` | List sessions for a project. |
| `GET`  | `/api/agent-session/{sid}`            | Full state snapshot (`session + tasks + artifacts + facts + decisions`). |
| `POST` | `/api/agent-session/{sid}/message`    | Post a follow-up message. Spawns a sibling `EXPLORE` when no `ASK_USER` is open. |
| `POST` | `/api/agent-session/{sid}/decision`   | Resolve a pending `ASK_USER` with a typed `{task_id, chosen, rationale?}` payload. |
| `DELETE` | `/api/agent-session/{sid}`          | Discard a session and its aggregate (tasks / facts / decisions / artifacts) via SQLite `ON DELETE CASCADE`. Returns `{deleted: sid}` or 404. |

A per-`project_root` runner cache (`_RUNNERS` in the route module)
reuses one `SessionStore` (and its SQLite WAL connection) across
requests. Mutating endpoints await `runner.run_next` in a thread
loop (`_drain_ready`, capped at four steps) so the request returns
once the synchronous tasks finish or an `ASK_USER` pauses the loop.

Every mutating endpoint returns `AgentSessionState`
(`cgx.webui.models`) so the React UI can render the updated tree in
one round-trip. There is no SSE on this surface today -- the UI
polls while any task is `IN_PROGRESS` other than an `ASK_USER`.

### React UI (`/agent`)

`frontend/src/pages/AgentPage.tsx` is the session-shaped page;
modular components live under `frontend/src/components/agent/`:

* `SessionLauncher.tsx` -- create a session (objective + project
  root + mode picker: *auto / explore / greenfield*; *auto* defers
  to `detect_mode`).
* `TaskTree.tsx` -- hierarchical DAG renderer keyed on
  `parent_task_id`; depth-based indentation, status icons, selection
  highlighting. Orphaned tasks re-surface at the top level so a
  malformed snapshot still renders. Includes badges for the
  greenfield kinds (`clarify`, `decompose`, `scaffold`).
* `ActiveTask.tsx` + `AskUserForm.tsx` -- detail panel for the
  selected task; dispatches on `expected_kind` to one of
  `ChoosePathForm`, `ChooseRecommendationForm`, `ApproveForm`,
  `ClarifyAnswersForm`, `ApprovePlanForm`, `FreeformForm`. Each
  form posts the typed `chosen` payload `build_decision` expects.
  `ClarifyAnswersForm` reads the linked `REQUIREMENTS_SHEET` and
  renders one labeled textarea per question; `ApprovePlanForm`
  renders the layered file manifest with `[Approve & Scaffold |
  Reject]`.
* `SidePanel.tsx` + `ArtifactPreview.tsx` -- tabbed Knowledge-Base
  and Artifacts view; per-kind artifact renderers, including
  dedicated bodies for `requirements_sheet`, `work_plan`, and
  `scaffold_patches`.
* `LiveView.tsx` -- right-hand active-task pane with polling and a
  mode badge in the session header (`explore` / `greenfield`).

`frontend/src/store/agentSession.ts` (Zustand + `persist` middleware,
key `cgx-agent-session`) holds the active session id, the selected
task id, and the three-column layout state (session-bar / task-tree
/ side-panel widths plus the collapsed flag for each rail) in
`localStorage` so a tab switch or reload comes back to the same
view and the same panel sizing; the snapshot itself is reloaded
from `/api/agent-session/{sid}` on mount rather than cached
client-side. Resize handles (`ResizeHandle.tsx`) clamp widths to
the bounds exported from the store and collapsed rails render in
place of a panel when its column would otherwise crowd a narrow
viewport.

`frontend/src/lib/api.ts` exports a typed `ApiError extends Error`
(`status`, `path`, `body`) that `jsonReq` throws on non-2xx
responses. `AgentPage.loadState` branches on
`e instanceof ApiError && e.status === 404` for the active id:
the persisted `activeId` / `selectedTaskId` are cleared, the
in-memory snapshot is dropped, and `refreshSessions()` reruns so
the launcher empty-state takes over. This makes the
`cgx-agent-session` store self-correcting against out-of-band
session deletion or a `project_root` switch to a different SQLite
file -- the loop that would otherwise re-fire the same 404 on every
mount terminates after one round-trip.

The legacy `AgentPage` lives at `frontend/src/pages/AgentLegacyPage.tsx`
and is mounted at `/agent-legacy`; it still drives the
`/api/agent` SSE stream against the batch `cgx.agents` loop
described below.

### Testing

* Core: unit tests over `models.py`, `store.py`, `router.py`,
  `runner.py`, `mode.py`, and each executor under
  `tests/test_session*.py`. Greenfield coverage includes
  mode-detection edge cases, router transitions for the
  `CLARIFY → DECOMPOSE → SCAFFOLD → APPLY → VERIFY` chain, the
  reject-plan halt path, and end-to-end runner walks with stub
  executors.
* HTTP: `tests/test_webui_agent_session.py` drives both write
  loops directly against the FastAPI route handlers through a
  `_HandlerClient` shim. Explore path: `EXPLORE` → choose_path →
  `INVESTIGATE` → `RECOMMEND` → choose_recommendation(plan_change)
  → `PLAN_CHANGE` → approve → `APPLY` → `VERIFY`. Greenfield path:
  explicit `mode="greenfield"` on session creation →
  `CLARIFY_REQUIREMENTS` → clarify_answers → `DECOMPOSE` →
  approve_plan → `SCAFFOLD` → `APPLY` → `VERIFY`. Pydantic still
  parses every payload (so the wire shape is validated) but no
  ASGI transport is required, keeping the test suite free of an
  `httpx` dependency. Stub executors are registered for every
  `TaskKind`.
* UI: `frontend/src/components/agent/AskUserForm.test.tsx` and
  `TaskTree.test.tsx` cover the React decision-posting contract and
  the DAG renderer respectively.

---

## Multi-agent loop

`cgx.agents` adds an orchestration layer on top of the single-shot
`answer_with_llm` / `generate_code_plan` / `generate_project_scaffold`
entry points:

1. **Planner** (`cgx.agents.planner.Planner`) -- decomposes a user goal
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
     (a) the `_SCAFFOLD_RE` regex -- a scaffold verb (`create`,
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
     etc.) -- covers prompts like *"create a calculator using React"*;
     (c) the LLM emitted at least one `scaffold` task and the goal
     has no existing-codebase hint (`_EXISTING_CODE_HINT_RE`:
     `existing`, `our app`, `legacy`, `refactor`, `modify`,
     `fix the bug`, …); (d) a scaffold verb together with at least one
     supported, non-style skill firing via `skills.detect_skills` --
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
2. **Tracker** (`cgx.agents.tracker.Tracker`) -- drives the plan
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
   `{task_id, name, kind, elapsed}` -- the UI uses this as a live
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
3. **Judge** (`cgx.agents.judge.Judge`) -- validates each completed task
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
     the Judge fall through to a structural pass -- and from there to
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
     returned (LLM judge skipped -- the per-file `scaffold_file` tasks
     carry their own verdicts).
   - `scaffold_file`: hard-fail when the generated file content is
     empty or only contains stubs; otherwise pass (the file's syntax
     is smoke-tested by `apply` downstream).
   - `search`: structural pass when `hits > 0` (LLM judge not invoked).
   - `apply`: fail when `failed_files` is non-empty (partial write --
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
engine, and is exposed at `/agent-legacy` in the React UI (and via
the `cgx agent` CLI). The default `/agent` route uses the
session-shaped backbone above; the batch loop is preserved unchanged
for callers that prefer fire-and-forget semantics. See
[flowcharts.md](flowcharts.md) for a visual breakdown of the loop and
the event timeline.

### Retry loop

`_stream_with_retry` in `cgx.agents.loop` handles all failure paths in
priority order and recurses up to `max_retries` times:

1. **Verify failures** -- test stdout/stderr is parsed by `_diagnose_failure`
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

2. **Apply failures** -- when the smoke check or cross-file coherence
   check rejects generated files, the failing-file list is forwarded to
   `_build_apply_fix_goal` which tells the LLM to regenerate only those
   files with valid syntax.  **Passing files are already on disk** so
   nothing already correct is lost. Apply failures trigger a recursive
   retry.
3. **Scaffold / plan generation failures** -- Judge rejections on the
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
regenerate the failing file(s) -- the correct files are already on disk.

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
`~/.cgx/tasks.db`) that records every SSE operation -- `ask`, `plan`,
`agent`, and `index` -- from the moment the request arrives to the final
`done` / `error` event.

**Schema** (simplified):

- `tasks` table -- one row per operation: `id` (UUID), `kind`, `status`
  (`running` / `done` / `cancelled` / `error`), `created_at`,
  `updated_at`, `goal` / `query` text.
- `task_events` table -- one row per SSE event: `task_id`, `seq`,
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

- `index.json` -- list of `SessionMeta(id, title, created_at, updated_at,
  message_count)` headers.
- `<uuid>.jsonl` -- append-only message stream, one JSON object per line
  with fields `role`, `content`, `at` (unix time), `meta`.

All writes go through a temp file + `os.replace` for atomicity. The
public API (`create_session`, `append_message`, `get_messages`,
`list_sessions`, `delete_session`, `rename_session`) is what the Ask
tab calls on every interaction; failures are swallowed so chat is
never broken by a session-store I/O error.

## Rate limiting

`cgx.answer.ratelimit` adds two primitives shared by every HTTP-backed
provider:

- `RateLimiter(rate, capacity)` -- token bucket guarded by a
  `threading.Lock`. `acquire()` is called before each request;
  `rate <= 0` makes the limiter a no-op so the existing call sites
  keep their pre-feature behaviour when no profile config is set.
- `request_with_retry(func, *, limiter, max_retries)` -- wraps a
  callable returning a `requests.Response`. Retries on HTTP **429**
  and **5xx** using exponential backoff with jitter, honouring the
  `Retry-After` header when present.

`Profile.rate_limit` (req/sec) and `Profile.max_retries` are
serialised with the rest of the provider config so cloud profiles
keep their per-tenant budget across sessions.

## Hardware / model matrix

`cgx.answer.hardware_matrix` is a pure-data offline module:

- `LOCAL_MODEL_CATALOG` -- 8 locally-runnable models with `name`,
  `params_b`, `min_ram_gb`, `recommended_vram_gb`, `ctx_window`,
  `family`, and a one-line `notes` blurb.
- `compute_local_fit(hw)` -- annotates the catalogue with a verdict
  string (`✅ fits` / `⚠️ tight` / `❌ won't fit` / `❓ unknown`) and a
  `reason`. Uses an "effective budget" of
  `max(ram_gb, gpu_vram_gb * 2.0)` when a GPU is detected.
- `TRADEOFFS` -- eight editorial rows comparing local vs cloud across
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
`~/.cgx/install_id`) and the CGX package version -- no prompts, no
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

- **`frontend/src/store/tasks.ts`** -- Zustand store backed by
  `sessionStorage`. Holds the in-flight streaming state for each page:
  agent (tasks / events / phase / summary), ask (messages), plan
  (thought / planMd / diff / report), index (progress / result).
  Components read from this store on mount, so a previously running task
  is immediately visible when the user returns to a tab.

- **`frontend/src/lib/connections.ts`** -- module-level
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
