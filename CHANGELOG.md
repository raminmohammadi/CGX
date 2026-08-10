# Changelog

All notable changes are documented here. Versions follow semver-ish.

## Unreleased -- Swarm mode redesign (plan-driven build engine)

`SessionMode.SWARM` is rebuilt from a brittle free-form Tech Lead/Developer
loop into a **plan-driven, one-file-at-a-time** engine. Every stage is
propose-then-validate: the model proposes, deterministic invariants
(coherence, toposort, contracts, syntax) enforce.

* **Tech Lead planner** (`swarm_tech_lead.py`). Prompts for a draft JSON plan,
  then normalizes it (dedupe + prune dangling `depends_on` edges), topologically
  orders the files with the shared Kahn toposort, and gates on buildability
  with a bounded 3-attempt corrective re-ask. Includes structural rooting checks (`_inconsistent_rooting`)
  to guarantee all modules are consistently placed (e.g., all under `src/` or all top-level). 
  Persists a `WORK_PLAN` artifact (`goal`, `layers`, `contracts`, ordered `paths`, `project_root`). An
  unbuildable plan ends the session FAILED instead of spawning empty work.
* **Incremental Developer** (`swarm_developer.py`, `swarm_generate.py`,
  `swarm_ground.py`). One planned file per turn, in dependency order, grounded
  on the real on-disk content of its dependencies. Generation ladder: full-file
  (gated on `ast.parse`, one re-ask) falling back to a deterministic
  `ASTAssembler` header + per-symbol build from the plan contracts; new files
  via `edit_file`, existing via `patch_file`. Unproducible files are recorded
  in `failed_paths` and carried forward.
* **Verification ladder** (`swarm_verify.py`). Static structural checks
  (first-party import coherence, contract compliance, JS/Python payload
  coherence) followed, only on success, by an environment dry-run
  (`preflight_install` + `run_tests_on_disk`). Emits a `SWARM_VERIFY_REPORT`
  artifact pinpointing files implicated by any failure.
  * **AST-driven auto-repair** (`_auto_fix_missing_imports`, `_auto_fix_function_logic`):
    Intercepts `NameError` and logic errors, isolates the failing block via AST,
    and asks the LLM to rewrite only the broken function in raw markdown (bypassing JSON schema).
  * **Dynamic Temperature Scaling**: The verification loop now scales LLM temperature dynamically 
    (from 0.2 to 0.8) during iterative repair rounds to prevent infinite loops of identical code.
* **Wrapper-tolerant plan parsing** (`swarm_parse.py`) + a typed `SwarmPlan`
  schema (`swarm_plan.py`), so a small local model wrapping JSON in prose or
  code fences still yields a validated plan.
* **Router chain.** `SWARM_TECH_LEAD` -> `SWARM_DEVELOPER` (file 0) ->
  ... -> `SWARM_DEVELOPER` (file *n*) -> `SWARM_VERIFY` -> terminal
  (COMPLETED when no `failed_paths`, else FAILED).
* **Plan-aware drain ceiling.** `drain_step_ceiling` scales a SWARM drain's
  step budget to the plan's file count (from the flat 64 used by
  explore/greenfield), so a large one-file-per-task build is not truncated
  mid-chain. Applied in both the webui drain and the TUI drive, recomputed
  per loop as the plan's Developer tasks are spawned.
* **UI.** The objective header's `ModeBadge` now renders a distinct `swarm`
  badge; swarm tasks and per-file generation progress already surface in the
  task tree and live progress banner.

### Correctness invariants / hardening

Hard gates that turn silently-tolerated defects into named, actionable
failures instead of patched-over symptoms.

* **Import safety** (`import_audit.py`). AST tools to detect and strip unused
  imports (`unused_imports`, `strip_unused_imports`) and to resolve first-party
  imports against the planned manifest (`resolve_first_party_imports`),
  accounting for both a top-level and a `src/` layout. Wired into
  `swarm_generate` as a phantom-import gate (re-ask, then strip as a last
  resort) and into `swarm_verify` as a path-level coherence check.
* **Plan verification** (`swarm_plan.verify_plan`). The Tech Lead's plan is
  validated for internal coherence -- dependency cycles, paths escaping the
  project root, and orphan tests -- before any Developer task is spawned.
* **Dynamic targeted regeneration** (`swarm_verify.py`). A structurally-clean
  tree whose suite is nonetheless red has its `ImportError` /
  `ModuleNotFoundError` parsed from the pytest output, regenerating only the
  implicated file before a final dry-run.
* **Dotted contract resolution** (`scaffold_validate.py`). A declared
  `Class.method` function contract now resolves against the generated class's
  method set instead of being sought as a (nonexistent) module-level symbol --
  fixing a false negative that failed a passing, correct tree.

### Structural completeness + code-quality gates

Turning "the model usually remembers" into deterministic guarantees, so a run
produces a complete, installable, testable project rather than a bag of source
files -- validated against four divergent fresh projects (URL-shortener,
CSV-stats CLI, task scheduler, unit converter), each built in its own wiped
folder.

* **Deterministic scaffolding injection** (`swarm_plan.ensure_scaffolding`).
  After normalization, any missing `README.md`, dependency manifest
  (`requirements.txt` / `pyproject.toml`), and -- for a `src/` layout -- a root
  `conftest.py` is appended to the plan directly, with `README.md` and
  `requirements.txt` made to depend on every `.py` so they generate last.
  `verify_plan` additionally re-asks (`_scaffolding_problems`) when any of
  these are still absent.
* **Test-coverage guarantee** (`swarm_plan.ensure_test_coverage`). A
  `tests/test_<module>.py` (depending on that module) is injected for every
  source module no planned test covers, so a plan can never ship with zero or
  partial coverage (fixes the "no tests ran" failure).
* **Non-Python generation path** (`swarm_generate.generate_file`). Non-`.py`
  planned files are routed off the Python ladder: deterministic source-derived
  templates for `requirements.txt` / `conftest.py`, a grounded free-form call
  for `README.md`, so scaffolding no longer has to survive an `ast.parse` gate.
* **No-stub generation gate** (`engine._contract_stub_symbols` /
  `_body_is_stub`). A generated module whose contract functions/methods have
  placeholder bodies (`pass` / `...` / docstring-only / `raise
  NotImplementedError`) is rejected with a hardened re-ask naming the
  offenders, so a stub such as `encode()->pass` can no longer clear the
  structural gates and ship.
* **Test-authoring discipline** (`_SINGLE_FILE_SYSTEM`). General (non-symptom)
  rules: call imported code only with values it accepts, construct all inputs
  inline / via `tmp_path` (never read an external data file), and assert
  invariants and round-trips rather than fabricated magic literals.
* **Failure-driven test regeneration** (`swarm_verify` +
  `engine.generate_repair_files` / `_LOGIC_REPAIR_SYSTEM`). The repair loop may
  now rewrite the offending *test* -- not only source -- when the test is the
  broken artifact (a fabricated literal, or a call the API does not accept),
  asserting an invariant/round-trip against the real interface. Soft contract
  warnings no longer suppress this pytest-driven repair.
* **De-brittled single-file prompt** (`_SINGLE_FILE_SYSTEM`). Example-specific
  negative directives accreted from individual runs were demoted toward general
  principles, leaning on the verify+repair (real pytest) loop as the
  generalizable correctness mechanism instead of per-hallucination patches.

## Unreleased -- Hugging Face integration (Inference Providers + GGUF browse)

Two additive, low-risk features that make CGX Hugging Face-friendly.

* **Hugging Face Inference provider** (`kind="huggingface"`). HF's
  Inference Providers expose an OpenAI-compatible router at
  `https://router.huggingface.co/v1`, so CGX reuses `OpenAICompatProvider`
  verbatim -- the host and endpoint path are pinned and only the token
  varies (read inline or from `HF_TOKEN` / `HUGGINGFACEHUB_API_TOKEN`).
  The Setup dropdown, `cloud_models` discovery, and `ping` all learn the
  new kind; the model list is populated from the router's public
  `/v1/models` (no token required to browse). The kind opts into the
  cross-encoder reranker by default like the other cloud kinds.
* **Browse Hugging Face panel.** A new Settings panel lists GGUF
  repositories from the Hub (`huggingface.co/api/models?filter=gguf`) with
  live search, sort (trending / downloads / likes / recently updated),
  download and like counts, and detected quantization labels. **Pull**
  hands the `hf.co/<repo>[:<quant>]` tag to the local Ollama daemon and
  streams progress through the shared `PullProgress` bar -- no HF token
  required. Outbound Hub/router hosts are added to the SSRF allowlist.
* **Clean local names for pulled GGUFs.** A model pulled through the
  Browse panel is no longer stored under its full `hf.co/<repo>` web
  address. `/api/ollama/pull` (`PullRequest`) gains an optional
  `local_name`; on a successful pull the backend re-aliases the model via
  Ollama's `POST /api/copy` + `DELETE /api/delete` (instant, no
  re-download) so `hf.co/ornith-ai/Ornith-1.0-9B-GGUF` lands as
  `Ornith-1.0-9B-GGUF`, then emits a `renamed_to` progress event the UI
  surfaces as "Download complete -- saved as <name>". `local_name` is
  passed through `_sanitize_local_name`, which restricts it to a bare
  `name[:tag]` (rejecting `/`, `..`, and stray characters) so it cannot
  smuggle a registry host/namespace into the copy destination. The
  re-alias is best-effort: any failure leaves the original tag in place
  and never turns a good download into a UI error.
* **Fixed: Browse panel empty on the default Trending sort.** The Hub
  `/api/models` endpoint rejects snake_case `sort=trending_score` with
  HTTP 400 -- it requires camelCase `trendingScore`. `_HF_HUB_SORTS` now
  maps the friendly UI key to the exact value the Hub expects, so the
  default sort returns results (other sorts were already correct).

## Unreleased -- Agent-loop hardening (import classification + polyglot provisioning)

Two residual convergence gaps from the `ses_0408ac4084b04b4c` post-mortem,
both greenfield-only.

* **First-party symbol mismatch is no longer pinned against PyPI**
  (Part 3). A pytest `ImportError: cannot import name 'X' from 'Y'` was
  classified unconditionally as `third_party_import_break` and sent to the
  dependency-pin proposer -- even when `Y` was a first-party module that
  imported cleanly but simply never defined `X`. A pin cannot add a
  first-party symbol, so the proposer produced no diff and the loop flapped
  until its budget drained. The pure classifier stays disk-free (still
  reports the raw shape); the REPAIR executor now resolves each
  imported-from module against disk via `locate._dotted_path_resolves` and,
  when `Y` is first-party, re-classifies to a new
  `first_party_symbol_mismatch` token that routes to `strategy=regenerate`
  -- naming the exact `symbol`/`module` pairs (`classify.import_name_breaks`)
  and forbidding a dependency pin. A genuinely third-party `Y` stays on the
  pin path.
* **Polyglot repos provision both stacks in one BOOTSTRAP_ENV pass**
  (Part 5). A repo declaring both a Python manifest and a `package.json`
  resolved to `project_type=python` and only provisioned the venv;
  `node_modules` was left to VERIFY's best-effort `npm install`, which
  silently verified the JS half against no dependencies whenever it could
  not run (offline / bounded timeout). BOOTSTRAP_ENV now also runs
  `_provision_node_modules` when a `package.json` is present, folding a
  `node` sub-report (`{outcome, note, log_tail}`) into the `BUILD_REPORT`
  and surfacing `node_outcome`. Node provisioning is non-fatal and leaves
  `project_type=python` so the Python-only gates are unaffected; the
  node-only path is refactored onto the same shared helper.

## Unreleased -- CLI parity (ask / plan / agent / status)

The `cgx` command now exposes every runtime capability that the
dashboard and web UI already offered, closing the gap where `cgx ask`
and `cgx agent` were documented but unimplemented.

* **New subcommands**: `cgx ask` (grounded, streamed LLM answers),
  `cgx plan` (self-testing code-change plans with `--self-test` /
  `--run-tests`), `cgx agent` (the batch Planner → Tracker → Judge loop
  with `--stop-on-fail`), and `cgx status` (provider + hardware + index
  summary). `cgx query` remains the raw, LLM-free retrieval dump.
* **Shared provider/index flags** across the new commands:
  `--provider {ollama,openai,openai-compat,gemini,custom}`, `--model`,
  `--base-url`, `--profile`, `--project-root`, and `--index-dir` /
  `--records` overrides. A `--profile` takes precedence and API keys are
  resolved from the environment / keyring — never passed on the command
  line.
* **Same streaming engine as the UI**: the commands reuse
  `cgx.webui.handlers` (`stream_ask` / `stream_plan` / `stream_agent`)
  driven through the terminal `Printer` / `run_stream`, so tokens stream
  live under a spinner and **Ctrl-C** cancels via the shared
  `cancel_event` (exit `130`; other failures exit `1`).
* **Index auto-discovery**: `ask` / `plan` / `status` read a completed
  index at `<project-root>/.cgx/index` by default; `agent` runs with or
  without one (greenfield generation).
* Implementation: new `plan_events` plus `index_dir` / `records` /
  `think` / `stop_on_fail` support in `cgx.cli.tui.ops`, a `plan`-payload
  branch in `map_event`, and the subcommand wiring in `cgx.cli.main`.
  Covered by new tests in `tests/test_cli_dashboard.py`.

## Unreleased -- Greenfield agentic-loop hardening (Phases A-E)

A reliability pass over the greenfield loop (Plan -> Scaffold -> Apply ->
Bootstrap -> Verify -> Repair) so it converges on small local models
(3B-7B) instead of stalling, looping, or writing unparseable code.
Every change is greenfield-only unless noted; explore-mode sessions and
the legacy `/agent` batch loop are unaffected.

### Phase A -- Multi-language verify + build-smoke

* `VERIFY` now dispatches to the correct test runner by project type:
  Python via pytest and JS/TS via `cgx.codegen.test_runner.NpmRunner`
  (`npm test` / configured script), so a Node scaffold is actually
  exercised rather than reported as "no tests collected".
* `BOOTSTRAP_ENV` provisions JS toolchains (detects `package.json`,
  runs the install step) alongside the existing Python venv path.
* A **build-smoke** gate runs the project's build/compile step before
  `VERIFY` so a scaffold that does not build fails fast with a concrete
  error instead of burning a full test run.

### Phase B -- Executor robustness (SCAFFOLD / per-file generation)

* **Syntax-retry**: `generate_single_scaffold_file` validates every
  generated file (Python via `ast`, JSON via `json`, TOML via
  `tomllib`, and the JS/TS/JSX/Vue family via tree-sitter through
  `cgx.codegen.validate.validate_js_ts_source`) and, on a parse
  failure, issues exactly one hardened regeneration with the concrete
  error surfaced (`_SYNTAX_RETRY_INSTR`). A file that still does not
  parse is dropped with a `syntax_error` rather than persisted.
* Additional single-file gates, each with one targeted retry:
  extension/content mismatch (e.g. Vue SFC under a `.jsx` path),
  duplicate-content (byte-identical to a sibling after whitespace
  normalisation), first-party symbol imports that the generated
  modules do not actually define (`_symbol_retry_instruction`), and a
  pytest test-collectability gate (`_TEST_RETRY_INSTR`) that rejects a
  test file with no top-level `def test_*` (pytest exit 5).
* **Targeted per-file regeneration**: when a `SCAFFOLD`/`APPLY` drops
  specific files, only those paths are regenerated while every
  prior-good diff is reused verbatim, so a retry is proportional to the
  failure instead of re-running the whole manifest.
* **Intra-layer parallelism**: `_generate_one` reads no shared state
  and can run inside a bounded per-layer worker pool. Concurrency is
  opt-in via `CGX_SCAFFOLD_CONCURRENCY` (defaults to 1/serial so a
  single local GPU is never over-subscribed; malformed or sub-1 values
  clamp to 1).
* **Incremental checkpointing**: the `SCAFFOLD_PATCHES` artifact is
  saved after every layer (`_checkpoint_progress`, best-effort upsert
  keyed by `artifact_id`). A crash or timeout mid-run leaves completed
  files on disk, and `_resume_generated_files` seeds them on the next
  attempt so only the remainder is regenerated.

### Phase C -- Planner (manifest ordering + re-planning)

* Manifest dependency ordering so files are generated in an order that
  respects intra-project imports.
* Re-planning escalation: an unrecoverable failure class walks up to
  the nearest `SCAFFOLD` ancestor and re-queues a fresh scaffold with
  `regenerate_constraints` folded into the goal (capped at
  `_REGENERATE_BUDGET=1` per ancestor chain).

### Phase D -- Bounded LLM repair

* `REPAIR` previously escalated any failure without a deterministic
  fix straight to `ASK_USER`. It now attempts a **bounded LLM repair**
  first (`_propose_llm_logic_repair` ->
  `cgx.answer.engine.generate_repair_files`): the model is handed the
  goal, the failing test tail, and the complete contents of the most
  relevant source/test files (capped at `_PATCH_DIFF_LIMIT=5` files),
  and returns complete corrected file bodies. Each proposed file is
  re-validated (`_validate_repair_source`, same per-language gate as
  scaffold) before it reaches `APPLY`. Only when the model declines or
  every candidate fails validation does the flow fall through to
  regenerate / `ASK_USER`.

### Phase E -- Session budget + escalation

* The `Session` aggregate carries a per-session budget: config fields
  `max_task_runs`, `max_wall_seconds`, and `headless` (all default to
  unlimited/off), plus live counters `task_runs` and
  `first_task_started_at`. Only compute-bearing tasks charge the
  budget; an `ASK_USER` wait-state stays free so escalation itself can
  always surface. The store round-trips the new fields with
  backward-compatible defaults, so pre-existing sessions load
  unchanged.
* On exhaustion, `Router.on_budget_exhausted` diverges by mode: an
  **interactive** session blocks every still-READY work task, spawns a
  single `ASK_USER(freeform)` surfacing the exhaustion, and goes
  `PAUSED`; a **headless** session abandons the READY work and ends
  terminally `FAILED`. This catches runaway autonomous loops that
  bypass per-task retry caps.

### Judge -- ASK / SUMMARIZE answer-quality gates

* Cheap deterministic structural pre-gates fail obviously-bad output
  before an LLM-grader call, deferring qualitative judgement to the
  strict local-model judge. New constants: `_ASK_MAX_WORDS = 1000`,
  `_SUMMARIZE_MAX_BULLETS = 8` (mirrors the "<=8 bullets" contract in
  the `summarize` capability), `_SUMMARIZE_MAX_WORDS = 400`.
* `SUMMARIZE` gained a structural branch (previously it soft-passed):
  it now hard-fails empty, over-bullet, and over-verbose summaries.
  `_LIST_ITEM_RE` counts bullets/ordered items without miscounting a
  Markdown `#` heading. The non-clarify `ASK` path hard-fails a
  pathologically long answer; within-budget answers still defer to the
  LLM judge for substance (grounding/citations). `_render_artifact`
  maps `SUMMARIZE` to its `answer_md` so the strict judge sees rendered
  markdown.

## Unreleased -- Deterministic endpoint enumeration (schema v5)

Counting/listing questions about the HTTP surface ("how many API
endpoints does X have?", "list all routes") are now answered
deterministically instead of through semantic ranking. The truth is an
aggregate scattered across many route-decorator chunks, so ranking
surfaced a handful and produced wrong counts.

At parse time, `parse_codebase._detect_route` recognizes FastAPI / Flask /
Starlette-style route decorators (`@app.get('/x')`, `@router.post(...)`,
`@app.websocket('/ws')`, `@app.route('/y', methods=[...])`) on
functions/methods and stamps `chunk.meta['route'] = {"methods": [...],
"path": str|None}`. Each record mirrors this into a `route` field.

`detect_intent` gains an `enumerate` mode (an enumeration cue *and* an
api/endpoint/route keyword). The new `cgx.answer.enumeration` module
filters route-bearing records (optionally scoped to a subject term drawn
from the question), dedupes, and renders an exact count + list.
`_prepare_answer_request` short-circuits to this result for both the
blocking and streaming answer paths when endpoints are found, and falls
through to normal grounded answering otherwise.

**Re-index advisory:** `SCHEMA_VERSION` is bumped `4 -> 5`. v4 indices
lack `route` and cannot be enumerated; rebuild to gain endpoint counting.

## Unreleased -- Documentation ingestion + record provenance (schema v4)

Standalone documentation (`README.md`, `docs/*.md`, design notes, `.rst`)
is now first-class in the index. A pure-python `MarkdownParser`
(`.md` / `.markdown` / `.mdx` / `.rst`) splits each doc file on its
headings (ATX `#` and setext underlines) and emits one `doc` chunk per
section plus a `file` chunk for the repo map; it needs no optional
dependency, so it is always registered. Vendored doc trees are pruned by
the same `.gitignore` / `DEFAULT_IGNORE_DIRS` / size-cap path as source
files, with `_site` and `.docusaurus` added to the default ignore list.

Every record now carries a `source_kind` (`"code"` | `"doc"`) mirrored
from `chunk.meta['source_kind']`, so retrieval / answer layers can
attribute and weight prose against code; `source_kind` is echoed into
the BM25 corpus rows as well.

**Re-index advisory:** `SCHEMA_VERSION` is bumped `3 -> 4`. v3 indices
lack `source_kind` and contain no documentation content; readers should
rebuild to pick up the new field and index their docs.

## Unreleased -- Session-based Agent (Phases 0-4) + Greenfield loop

A ground-up rewrite of the Agent surface around a **persistent,
session-shaped task tree** with structured human-in-the-loop
checkpoints. The new `/agent` route is the default Agent UI; the
original Planner / Tracker / Judge loop is preserved unchanged at
`/agent-legacy` and via the `cgx agent` CLI. Backend dependencies,
the retrieval / codegen stacks, and the legacy `/api/agent` SSE
route are untouched, so existing scripted callers keep working.

A follow-up greenfield extension lets the same session backbone
scaffold **new projects from scratch** when no index is available --
`CLARIFY_REQUIREMENTS -> ASK(clarify_answers) -> DECOMPOSE ->
ASK(approve_plan) -> SCAFFOLD -> APPLY -> BOOTSTRAP_ENV -> VERIFY` --
selected automatically by `Session.mode` (`explore` vs `greenfield`).
Greenfield projects now go through a dedicated `BOOTSTRAP_ENV` step
that provisions a project-local `.venv`, installs declared
requirements, and preflight-installs undeclared imports before
`VERIFY` runs, so pytest no longer fails to collect because Flask /
FastAPI weren't installed in CGX's own interpreter.

A second follow-up adds an **autonomous repair loop**: a failed
`VERIFY` in greenfield mode now spawns a `REPAIR` task that
classifies the failure (e.g. mixing `self.assertLogs` into a
pytest-style class with no `unittest.TestCase` base) and emits a
typed `REPAIR_PLAN` whose diffs the shared `APPLY` executor writes;
the router then re-runs `VERIFY` (skipping `BOOTSTRAP_ENV`, since the
venv is already up). The cycle is capped at 2 attempts and gated by
a failure-signature hash, so repeating failures escalate to
`ASK_USER` instead of looping.

### Added

- **Curated function-call tracing (Phase TR)** -- New single-file
  instrumentation layer `cgx.trace` gated behind a global toggle that
  emits `trace_enter` / `trace_exit` (with `elapsed_ms`) or
  `trace_error` (with `error_type` + truncated message) records for
  every high-signal entry point on the agent loop. Curated targets --
  **not** every function in `src/cgx/` -- to keep the log signal high
  and the production overhead a single `bool` check when off. Wrapped
  with `@traced(category)`: the router (`Router.on_user_message`,
  `on_task_completed`, `on_decision_recorded`), the runner
  (`SessionRunner._post_message_traced`, `_post_decision_traced`,
  `_run_next_traced`), every executor via `dispatch` in
  `cgx.session.tasks.base` (wraps the registered function at
  registration time so every `TaskKind` participates without per-file
  edits), the three repair helpers (`cgx.session.repair.{classify,
  locate, propose}`), the four LLM entry points in
  `cgx.answer.engine` (`answer_with_llm`, `generate_code_plan`,
  `plan_scaffold_manifest`, `generate_single_scaffold_file`), the
  three retrieval entry points in `cgx.retrieval.orchestrator` plus
  `cgx.pipeline.auto.run_query_auto`, the three codegen entry points
  (`cgx.codegen.disk_apply.apply_diffs_to_disk`,
  `cgx.codegen.env_manager.preflight_install`, and the two runners in
  `cgx.codegen.test_runner`), and the legacy batch loop
  (`cgx.agents.loop.run_agent`). Records are routed via a
  `contextvars.ContextVar` carrying `session_id` / `task_id` /
  `project_root`: when a session context is active the records land
  in `<project_root>/.cgx/agent.log` alongside the existing Phase 1.3
  task-transition rows so a single tail on the project log shows both
  business events and per-call timings inline; when no session
  context is set (HTTP middleware, batch CLI, retrieval / codegen
  called directly) records fall through to a rotating fallback at
  `~/.cgx/cgx-trace.log` (2 MiB × 3 backups). The runner sets the
  context inside `start_session`, `post_message`, `post_decision`,
  `run_next`, and `_execute` before any decorator fires so the
  runner's own records route correctly -- the three mutating public
  methods are thin un-decorated wrappers that prime the ContextVar
  and delegate to a `@traced("runner")` inner. Toggle precedence:
  (1) `$CGX_TRACE` env var (`1`/`true`/`yes`/`on` pins ON;
  `0`/`false`/`no`/`off` pins OFF; `set_trace_enabled` becomes a
  no-op while pinned; `trace_source()` returns `"env"`),
  (2) runtime flag flipped via new `POST /api/settings/trace`
  endpoint with `{"enabled": true|false}` -- returns HTTP `409`
  when the env var pins the flag so the operator can see the
  override is coming from the environment, not a stuck UI control,
  (3) programmatic `cgx.trace.set_trace_enabled(True)` for tests /
  scripts. Frontend surface: new `frontend/src/store/trace.ts`
  (shared Zustand store holding `{enabled, source}` with
  `refresh()` and `set()` actions), a "Function-call tracing"
  toggle card on `frontend/src/pages/SettingsPage.tsx`, an amber
  `TRACE` pill next to the Mode badge in
  `frontend/src/layout/Header.tsx` (tooltip explains env-pinned vs
  UI-toggled), and `frontend/src/layout/AppShell.tsx` primes the
  store on mount so the pill reflects server-side state on first
  paint. Ten new tests: `tests/test_trace.py` covers the sync /
  async decorator paths, exception path, toggle-off no-op, and
  nested `ContextVar` propagation across sync + async calls;
  `tests/test_webui_settings.py` covers GET / POST plus the
  env-pinned 409 branch (ON and OFF variants);
  `tests/test_trace_integration.py` drives a real `SessionRunner`
  against a tmp project and asserts `agent.log` contains `runner`,
  `router`, and `executor` trace lines when the toggle is ON and
  zero trace lines when it's OFF.
- **Cross-session lessons store (Phase 7.1)** -- New module
  `cgx.session.lessons` persists a generalisable rule every time a
  REPAIR cycle is observed to repair its failure -- i.e. a downstream
  VERIFY in the same chain finishes ``outcome=passed``. The store is
  an append-only JSONL file at `~/.cgx/lessons.jsonl` (override via
  `$CGX_LESSONS_PATH`); each row carries a stable `lesson_id`, an
  ISO-8601 `created_at`, the originating `session_id`, the
  `trigger_signature` (REPAIR_PLAN's `failure_signature`),
  `classification`, an `applied_fix` summary
  (`{strategy, diff_count, files, extra_constraints}` -- never the
  raw diff body, so a lesson row stays small and review-friendly),
  and a `scope` payload describing the SCAFFOLD context the fix
  applied to (`stack` derived from the WORK_PLAN's `requirements_pins`
  / `pins` / `stack` arrays, `objective_keywords` derived from the
  scaffold's `prior_goal` via a stopword-filtered word tokeniser).
  Disk failures, malformed JSON, and missing files are all swallowed
  silently -- learning is best-effort and must not break the agent
  loop. `relevant_lessons(objective, stack, limit)` scores entries
  with +2 per stack overlap (case-insensitive, normalised package
  names) and +1 per objective-keyword overlap, tie-breaks by recency,
  and returns the top `limit`. The router (`cgx.session.router`)
  emits a new `RecordLesson(verify_task_id, repair_task_id,
  scaffold_task_id)` action via `_verify_lesson_actions` whenever a
  VERIFY finishes with `outcome=passed` AND a REPAIR is found on its
  ancestor chain; the SCAFFOLD id is recorded too (when present) for
  scope provenance. The runner (`cgx.session.runner.SessionRunner`)
  resolves these actions in `_record_lesson`: it fetches the REPAIR's
  `REPAIR_PLAN` artifact and the SCAFFOLD's inputs, derives the
  `applied_fix` and `scope` payloads, and calls `record_lesson`. Any
  exception is logged + swallowed. `cgx.session.tasks.scaffold` reads
  matching lessons (top 3) via `_lessons_as_constraints` and appends
  them to the composed goal under a dedicated `Lessons from prior
  sessions to apply:` header (re-using the Phase 6.1
  `_augment_goal_with_constraints` helper with the new `header`
  kwarg) so the per-file generator sees the constraint in its
  `goal` parameter. Eight new tests in `tests/test_session.py` cover
  the record-and-load roundtrip (append, not overwrite), the empty-
  signature guard, the stack-then-keywords scoring with the no-overlap
  exclusion, the missing-store noop, the router VERIFY-pass-with-
  REPAIR hook, the VERIFY-pass-without-REPAIR negative case, the
  runner end-to-end persistence path (env-var override), and the
  SCAFFOLD goal-injection smoke test (the generator sees the
  lesson's signature + classification in its `goal`).
- **Branching repair: patch vs regenerate (Phase 6.1)** -- The REPAIR
  executor in `cgx.session.tasks.repair` now picks an explicit strategy
  -- `patch` or `regenerate` -- via the new `_select_repair_strategy`
  helper, recording it on both the REPAIR_PLAN artifact and the
  executor outputs along with a structured `extra_constraints` payload
  shaped per classification. The patch branch is preserved verbatim
  (small, well-localised diffs go straight to APPLY); the regenerate
  branch triggers when the proposer either produced no diff for a
  regenerate-eligible classification (`smoke_import_failure`,
  `api_check_failure`, `third_party_import_break`, `unknown`) or when
  the proposed patch exceeds `_PATCH_DIFF_LIMIT` (5 files). The
  SMOKE_REPORT and API_CHECK_REPORT branches always set
  `strategy=regenerate` because their classes are by construction
  un-patchable from a single failure record. The router
  (`cgx.session.router`) splices a new dispatch step before the
  table-driven successor lookup: when a finished REPAIR carries
  `outputs.strategy == "regenerate"`, `_repair_regenerate_actions`
  walks up `parent_task_id` to the nearest SCAFFOLD ancestor via
  `_find_scaffold_ancestor`, abandons every live descendant via
  `_collect_descendants` + `UpdateTaskStatus(ABANDONED)` (DONE /
  FAILED / already-ABANDONED descendants are skipped to preserve
  audit history), and re-queues a fresh SCAFFOLD task via
  `propose_regenerate` whose `inputs` carry the running
  `regenerate_constraints` list (appended -- not overwritten -- so
  successive attempts see the full failure history), an incremented
  `regenerate_attempt` counter capped by `_REGENERATE_BUDGET` (1 per
  ancestor chain), and a `regenerated_from_task_id` back-pointer.
  Four early-exit guards (wrong strategy, no SCAFFOLD ancestor,
  budget exhausted, nothing to abandon) all degrade gracefully back
  to the patch / ASK_USER table path, so a regenerate verdict in a
  session without a SCAFFOLD parent still escalates correctly.
  `cgx.session.tasks.scaffold.run_scaffold` reads
  `task.inputs.regenerate_constraints` and folds each
  `{kind, rationale, ...}` entry into the composed goal via
  `_augment_goal_with_constraints` as a `Prior-attempt failures to
  avoid this time:` tail, so the per-file generator sees the
  prior-attempt context without any change to the prompt builder
  itself. Nine new tests in `tests/test_session.py` cover the
  strategy selector (small-diff patch, no-diff regenerate, oversized
  regenerate, regenerate-eligible class with diffs staying on
  patch), `propose_regenerate`'s attempt/constraint accumulation and
  parent-id preservation (with aliasing absence), the happy-path
  router branch (abandon + SCAFFOLD spawn, DONE descendants
  preserved), the budget-exhausted fallback to ASK_USER, the
  missing-ancestor fallback, and the SCAFFOLD goal-augmentation
  smoke test (the generator sees the rationale in its `goal`
  parameter).
- **LLM call tracing as `Fact` records (Phase 5.1)** -- New module
  `cgx.session.llm_trace` ships `TracingProvider`, a thin wrapper that
  intercepts the `chat` / `chat_stream` surface of any
  `cgx.answer.providers.LLMProvider` and, while a `(session_id, task_id)`
  pair is bound, records each call as a typed `Fact` of new kind
  `FactKind.LLM_CALL`. The fact's `content` carries `{model, prompt,
  response, prompt_chars, response_chars, latency_ms, sampling, streamed}`
  with prompts and responses truncated symmetrically to 8K chars (full
  byte counts remain visible via `*_chars`); the raw exception text is
  preserved on `content.error` when a chat call raises so failed LLM
  attempts are auditable too. The runner (`cgx.session.runner`) calls
  `provider.bind(...)` immediately before `dispatch(task, deps)` and
  drains accumulated facts into the store via `provider.drain()` along
  the success path and both failure paths (`LookupError` and generic
  exceptions), guarded by a `try / finally` that always unbinds; the
  bind/drain wiring is gated on `hasattr(provider, "bind") and
  hasattr(provider, "drain")` so untraced stubs keep working without
  modification. The WebUI route `agent_session._build_deps` wraps every
  resolved provider in a `TracingProvider` (idempotent via
  `isinstance` guard) so every greenfield CLARIFY / DECOMPOSE / SCAFFOLD
  / REPAIR call now persists its prompt/response pair without changing
  any executor-side code. The frontend surfaces traces on the active
  task card via a new collapsible `LLM calls (N)` section in
  `ActiveTask.tsx` that lists each call with model + latency + stream/
  error chips and expands to show sampling parameters, the prompt, and
  the response (or error) in a scrollable pane; `SidePanel`'s Facts
  tab also adopts a per-kind label that shows `llm_call · <ms>` plus
  the model name. Five new tests in `tests/test_session.py` cover the
  happy chat path (fact contents + drain semantics), `chat_stream`
  accumulation, the exception path, the unbound-call silent no-op, and
  the end-to-end runner integration where a stub executor's
  `deps.provider.chat(...)` lands as an `LLM_CALL` fact attributed to
  the executing task.
- **PyPI-aware pin validator at `SCAFFOLD` (Phase 4.1)** -- New
  module `cgx.session.scaffold_validate` runs after the per-file
  generation loop in `run_scaffold`, just before the
  `SCAFFOLD_PATCHES` artifact is persisted. For every diff whose path
  matches `is_requirements_path` (`requirements.txt`,
  `requirements-dev.txt`, or `requirements/*.txt`), the validator
  parses the generated body, indexes pins by normalised package
  name, and -- for each pinned consumer in the curated
  `FRAGILE_PEERS` table (Flask <-> Werkzeug / Jinja2 / itsdangerous /
  click, Alembic <-> SQLAlchemy, SciPy <-> NumPy, Pydantic <->
  pydantic-core) -- reuses the consumer's PyPI `info.requires_dist`
  constraint to either replace an unbounded peer line or append a
  missing one. The rewritten content is repackaged through
  `_content_to_new_file_patch` so it round-trips through
  `apply_diffs_to_disk` exactly like the generator's original new-file
  diff. Each rewrite emits a structured
  `{file, consumer, consumer_version, peer, before, after, source}`
  row on `content.pin_adjustments`, and `outputs.pin_adjustments_count`
  surfaces the size for the router / UI. The shared
  `cgx.session.repair.pypi_client.PyPIClient` from Phase 3.2 is
  reused (so the disk cache under `~/.cgx/pypi-cache/` covers both
  REPAIR and SCAFFOLD lookups); `deps.extra["pypi_client"]` lets
  tests inject a stub. Unpinned consumers, missing PyPI metadata,
  and network failures all degrade to no-op (returns the original
  diffs and empty adjustments) so SCAFFOLD never blocks on transient
  PyPI errors. Seven new unit tests in `tests/test_session.py` cover
  `is_requirements_path` (canonical + negative layouts),
  `validate_requirements_text` (append-when-missing, replace-when-
  unbounded, unpinned-consumer no-op, fetch-failure no-op),
  `validate_scaffold_diffs` (round-trip swap of a requirements.txt
  diff while leaving siblings untouched), and the end-to-end
  `run_scaffold` path (`pin_adjustments` surfaces on the artifact,
  `requirements.txt` diff body contains the tightened pin).
- **`third_party_import_break` classifier + PyPI-aware proposer
  (Phase 3.2)** -- New module `cgx.session.repair.pypi_client`
  ships a tiny PyPI JSON client (`get_package` / `get_release`) with
  a read-through disk cache under `~/.cgx/pypi-cache/<name>/` (7-day
  TTL for the package roll-up; never-expire per-release records since
  those are immutable on PyPI). The default fetcher uses
  `urllib.request` with a polite User-Agent and an 8-second timeout;
  the constructor's `fetcher=` parameter lets tests stub the network
  entirely, and any `URLError` / `OSError` / decode error degrades to
  `None` so callers can fall back gracefully. `classify.py` is
  refactored into a small ordered registry of `(name, predicate)`
  rules so adding a new classification stays one append; the new
  `third_party_import_break` predicate matches both
  `ImportError: cannot import name '<sym>' from '<pkg>'` and
  `ModuleNotFoundError: No module named '<pkg>'` (where `<pkg>` is
  not in `sys.stdlib_module_names`) across either structured
  `failures[].message` rows or the raw stdout/stderr; it takes
  priority over `missing_module_pythonpath` so a Werkzeug-style
  symbol break wins when both signals are present in the same blob.
  `propose.py` gains `propose_third_party_pin(project_root, content,
  *, pairs, installed_packages, pypi_client)` that, for each
  `(symbol, broken_pkg)` pair, walks `failures[].traceback` for
  `site-packages/<x>/` to detect the consumer package, then queries
  PyPI: first looking for an explicit upper bound in the consumer's
  `info.requires_dist` (reuses the constraint verbatim) and, when
  the consumer didn't declare one, falling back to the highest
  contemporaneous peer release (within a 60-day window of the
  consumer's upload time, skipping pre/rc/dev). The resulting pin
  is folded into a `requirements.txt` diff by `_build_requirements_diff`
  (case-insensitive replacement when the file already lists the peer,
  append otherwise); each pair also produces a structured
  `{symbol, broken_pkg, consumer, consumer_version, reason, pin}`
  decision record. `run_repair` is wired to read
  `resolved_packages` off the upstream `BUILD_REPORT` (via
  `_installed_packages_from_build`) and pass an injected /
  default `PyPIClient` (via `_resolve_pypi_client`) into the
  proposer; the `REPAIR_PLAN` artifact gains `import_breaks` and
  `pin_decisions` fields and `_third_party_rationale` composes a
  human-readable summary explaining which pin was chosen and why.
  Eight new unit tests in `tests/test_session.py` cover the
  classifier (precedence over `missing_module_pythonpath`, exact
  pair extraction), the cache (single fetch across repeated
  `get_package` calls, `None` on network failure), the proposer
  (declared `requires_dist` reuse, release-window fallback,
  consumer-not-detected escalation), and the end-to-end
  `run_repair` flow that produces a `Werkzeug<3,>=2.0`
  `requirements.txt` diff from a synthetic Flask 2.1.2 BUILD_REPORT
  + ImportError VERIFY_REPORT.
- **Structured pytest failures in `VERIFY_REPORT`** -- `run_verify`
  now allocates a per-run JUnit XML sink via
  `tempfile.mkstemp(prefix="cgx_junit_")` and threads
  `-rN --tb=long --junitxml=<path>` through `run_tests_on_disk`'s
  `extra_pytest_args` (alongside the existing `-q --no-header`). After
  pytest exits, `_parse_junit_failures` walks every `<testcase>` in
  the XML and emits `{nodeid, kind, type, message, traceback}` rows
  for nested `<failure>` (assertions) and `<error>` (collection /
  setup / teardown) nodes, which land on `content["failures"]`. The
  parser is defensive: a missing, empty, or malformed XML file
  degrades to `failures=[]` while the raw `stdout` / `stderr` panes
  remain untouched for the human view. `_unlink_quiet` cleans up the
  temp file on every code path, including the executor-crash branch.
  `ArtifactPreview.tsx`'s `VerifyBody` gains a collapsible per-failure
  list (capped at eight rows) that renders `kind · nodeid · type`,
  the message, and a tailed traceback in red below the reproducer.
  Two new unit tests in `tests/test_session.py` cover the happy path
  (writes a synthetic two-case XML via the `run_tests_on_disk` stub,
  asserts both `failure` and `error` rows are extracted with the
  expected nodeid / type / message / traceback) and the
  no-XML-written fallback (`failures == []`); existing reproduce_cmd
  stubs are migrated to `**_kw` to absorb the new
  `extra_pytest_args` kwarg. Sets up the Phase 3.2 classifier to
  match on structured failure types instead of free-form stdout.
- **`API_CHECK` task between `BOOTSTRAP_ENV` and `SMOKE`** -- New
  `TaskKind.API_CHECK` + `ArtifactKind.API_CHECK_REPORT` plus an
  executor at `cgx.session.tasks.api_check.run_api_check` that walks
  every applied `.py` file with `ast`, collects each top-level
  `from <third_party.sub> import <name>` plus aliased
  `pkg.<attr>` access (tracking module-scope `import ... as` aliases,
  filtering stdlib via `sys.stdlib_module_names`, relative imports,
  and first-party packages), and resolves each `(module, name)`
  pair in a single subprocess against the bootstrapped venv's
  interpreter via `importlib.import_module` + `hasattr`. Each row
  carries `{module, name, ok, error, references[{file, lineno}]}`;
  the artifact aggregates `outcome` (`passed` / `failed` /
  `skipped`), `failed_references`, and a stable
  `failure_signature` (`api_check|<sorted module.name pairs>`).
  The greenfield router now chains
  `BOOTSTRAP_ENV -> API_CHECK -> SMOKE -> VERIFY`: passed / skipped
  hands off to SMOKE with `api_check_artifact_id` carried forward;
  a `failed` outcome routes to `REPAIR` with the API_CHECK_REPORT
  as the source artifact, gated by the same `_REPAIR_BUDGET` and
  `prior_failure_signatures` flap detector. `run_repair` accepts a
  third upstream artifact kind (`API_CHECK_REPORT`) and emits a
  `REPAIR_PLAN` with classification `api_check_failure`,
  `can_apply=False`, and a rationale enumerating the unresolved
  `(module, name)` pairs (deterministic proposer lands in Phase
  3.2). `ArtifactPreview.tsx` gains an `ApiCheckReportBody` panel
  that lists unresolved references with their source file/lineno
  and collapses resolved ones behind a `<details>` toggle. New
  unit tests in `tests/test_session.py` cover the static
  collector (ImportFrom + alias attribute resolution, stdlib /
  first-party filtering), the executor's skipped / failed /
  probe-error branches, the four router transitions (passed,
  skipped, failed-spawns-REPAIR, flap-skips-REPAIR), and the
  API_CHECK_REPORT-fed REPAIR path; `test_runner_full_greenfield_loop`
  and `test_full_greenfield_loop_via_http` gain API_CHECK stubs
  and now step through seven successor tasks (the WebUI route's
  `_drain_ready` already covered the extra step from its earlier
  `max_steps=8` bump).
- **`SMOKE` task between `BOOTSTRAP_ENV` and `VERIFY`** -- New
  `TaskKind.SMOKE` + `ArtifactKind.SMOKE_REPORT` plus a dedicated
  executor at `cgx.session.tasks.smoke.run_smoke` that statically
  walks each applied `.py` file with `ast`, drops stdlib (via
  `sys.stdlib_module_names`), relative imports, and first-party
  modules (anything that resolves to a file or package directory
  under `<root>` or `<root>/src`), and then runs
  `<venv>/bin/python -c "import <pkg>"` with a 5 s per-module
  timeout against the bootstrapped interpreter recorded in the
  upstream `BUILD_REPORT`. The artifact records each candidate as
  `{name, ok, stderr_tail}` plus an aggregate `outcome` token
  (`passed` / `failed` / `skipped`) and a `failure_signature`
  (`smoke_import|<sorted,modules>`). The greenfield router now
  chains `BOOTSTRAP_ENV -> SMOKE -> VERIFY`: a `passed` or `skipped`
  outcome forwards `build_artifact_id` + `smoke_artifact_id` to
  VERIFY exactly as before; a `failed` outcome spawns REPAIR with
  the `SMOKE_REPORT` as the source artifact instead, gated by the
  same `_REPAIR_BUDGET` cap and `prior_failure_signatures` flap
  detector as the VERIFY-driven repair loop. `REPAIR` now accepts
  both `verify_artifact_id` and `smoke_artifact_id` inputs; the
  smoke path classifies as `smoke_import_failure` with
  `can_apply=False` so the router escalates to `ASK_USER` with a
  structured rationale (deterministic dependency-aware proposer
  arrives in Phase 3.2). `ArtifactPreview.tsx` gains a
  `SmokeReportBody` panel that surfaces each failing import with
  its trimmed stderr tail and collapses the passed list behind a
  `<details>` toggle. `_drain_ready` in the HTTP route bumps its
  default `max_steps` from 4 to 6 to cover the new five-step
  write loop. Eleven new unit tests in `tests/test_session.py`
  cover the static collector, the executor's skipped / passed /
  failed branches, the apply-artifact fallback for
  `applied_files`, the three router transitions (passed,
  skipped, failed) plus the budget and flap guards, and the
  SMOKE_REPORT-fed REPAIR path; the existing full-loop integration
  tests (`test_runner_full_greenfield_loop`, the HTTP-driven
  `test_full_greenfield_loop_via_http`) gain SMOKE stubs and now
  step through six successor tasks.
- **Project-local agent log (`<project_root>/.cgx/agent.log`)** --
  New `cgx.session.agent_log` module exposes `log_event(project_root,
  event, **fields)` which appends one JSON object per line to a
  rotating handler (`maxBytes=1 MiB`, `backupCount=3`) under the
  project's `.cgx/` directory. `SessionRunner._execute` now emits
  `task_started` before dispatch, `task_completed` /
  `task_waiting_user` on success, `task_failed` for executor-reported
  failures, and `executor_crashed` / `executor_missing` for raised
  exceptions and lookup errors -- always tagged with `session_id`,
  `task_id`, `kind`, and (where applicable) `duration_ms`,
  `artifact_id`, `error`, `exc_type`. Falsy `project_root` makes the
  call a no-op so explore-mode and test sessions never write to disk.
  All write failures are swallowed (logged to stdout) so a busted log
  file can never fail a task. Five new unit tests in
  `tests/test_session.py` cover the no-op branch, the JSONL shape,
  and the three runner code paths (happy / `failure` / raised
  exception); a new autouse `_reset_agent_log` fixture closes cached
  handlers between tests so `tmp_path`-rooted runs don't leak file
  descriptors.
- **Reproducer command in `VERIFY_REPORT`** --
  `cgx.session.tasks.verify.run_verify` now records
  `content.reproduce_cmd` -- a single shell line that re-runs the
  exact failing pytest invocation under the same interpreter the
  agent used (the BUILD_REPORT venv when set, otherwise the
  project's auto-detected `.venv/bin/python`). The line shape is
  ``cd <project_root> && <python> -m pytest -q --no-header <tests...>``
  with every argument `shlex.quote`-escaped and selected tests
  rendered relative to `project_root` so the result pastes cleanly
  into a terminal. Set to `null` for runs that selected zero tests
  (skipped / no_tests_collected / pytest_missing), since there is
  nothing meaningful to reproduce. Surfaced under `VerifyBody` in
  the Agent UI as an emerald monospaced block above the captured
  stdout. Unit-tested via
  `test_verify_executor_records_reproduce_cmd` in
  `tests/test_session.py`, which monkeypatches `run_tests_on_disk`
  to exercise both the populated and `None` paths.
- **Resolved-package snapshot in `BUILD_REPORT`** --
  `cgx.session.tasks.bootstrap_env.run_bootstrap_env` now runs
  ``<venv>/bin/python -m pip freeze --all`` after preflight installs
  finish and records the parsed result on the `BUILD_REPORT` artifact
  as `content.resolved_packages` (list of
  `{name, version}`, PEP 503-normalised) plus the raw
  `content.pip_freeze_text` for human inspection. Best-effort: any
  subprocess failure, non-zero return code, or parse error collapses
  to empty fields so a busted venv never fails BOOTSTRAP. Only runs
  when a real project venv is present -- the `no_venv` fallback to
  the host interpreter is skipped to avoid leaking unrelated
  host-side packages into the report. Surfaced in the UI under
  `BuildReportBody` as a collapsible "resolved: N packages" list.
  Prerequisite for a downstream PyPI-aware `third_party_import_break`
  classifier that needs to know *resolved* dependency versions
  (e.g. detect a Flask 2.1 + Werkzeug 3 mismatch) instead of guessing
  from an `ImportError` traceback. New unit tests cover the
  parser (canonical / editable / URL / comment / marker lines),
  the subprocess error and non-zero-rc swallowing, the wired-in
  monkeypatched happy path, the graceful empty-fields fallback, and
  the no-freeze-on-`no_venv` guarantee in
  `tests/test_session.py`.
- **Session deletion endpoint** --
  `DELETE /api/agent-session/{sid}?project_root=...` in
  `cgx.webui.routes.agent_session` removes a session and its
  full aggregate (tasks / facts / decisions / artifacts) via
  SQLite `ON DELETE CASCADE`. Returns `{deleted: sid}` on success
  or 404 if no runner knows about the id. The Agent UI surfaces a
  hover-revealed trash icon on each session row (both the active
  `SessionBar` and the empty-state `PriorSessions` list) with a
  `window.confirm` guard; deleting the active session clears the
  selection. Frontend wiring: `agentSessionDelete(sid, projectRoot?)`
  in `frontend/src/lib/api.ts`, `deleteSession` handler in
  `AgentPage.tsx`, regression test
  `test_delete_session_removes_aggregate_and_404s_on_followups`
  in `tests/test_webui_agent_session.py`.
- **Resizable + collapsible Agent panels** -- the three-column
  Agent layout (sessions / task tree / artifacts+facts) now has
  drag handles between every column and collapse toggles on the
  left and right rails so the page no longer breaks on narrow
  viewports. Bounds: session bar 160-360 px, task tree 180-420 px,
  side panel 220-480 px. Widths and collapsed flags are persisted
  in the existing `cgx-agent-session` `localStorage` key alongside
  the active session and selected task id. Implementation:
  `frontend/src/components/agent/ResizeHandle.tsx` (pointer-event
  drag with stable baseline), `LiveView.tsx` refactored to inline
  widths plus a 28 px `CollapsedRail`, and `useAgentSession`
  expanded with `sessionBarWidth` / `taskTreeWidth` /
  `sidePanelWidth` plus `sessionBarCollapsed` /
  `sidePanelCollapsed` (clamped to the per-column bounds).
- **Greenfield session mode (G0-G9)** -- new
  `Session.mode: SessionMode` field with values `EXPLORE` (default,
  modifying an existing repo) and `GREENFIELD` (scaffolding a brand
  new project). `cgx.session.mode.detect_mode(project_root, index_dir)`
  auto-selects based on whether the project root is empty / missing
  and whether a FAISS index exists, and is invoked from
  `routes/agent_session.create_session` when the request omits an
  explicit `mode`. The router seeds `CLARIFY_REQUIREMENTS` as the root
  task in greenfield mode instead of `EXPLORE`.
- **Greenfield task kinds + artifacts** -- `TaskKind` gains
  `CLARIFY_REQUIREMENTS`, `DECOMPOSE`, `SCAFFOLD`. `ArtifactKind`
  gains `REQUIREMENTS_SHEET` (the clarifying questions),
  `WORK_PLAN` (the layered scaffolding manifest with `plan_md` +
  `layers[].files[]`), and `SCAFFOLD_PATCHES` (per-file generated
  diffs compatible with `apply_diffs_to_disk`). `DecisionKind` gains
  `CLARIFY_ANSWERS` (user answers to the questionnaire) and
  `APPROVE_PLAN` (approve / reject the work plan).
- **Greenfield executors** --
  `cgx.session.tasks.clarify_requirements` issues a `force_json` LLM
  call to produce 3-6 structured clarification questions (tech stack,
  must-haves, target environment) with a deterministic fallback bank
  when the provider is unavailable;
  `cgx.session.tasks.decompose` wraps
  `cgx.answer.engine.plan_scaffold_manifest` with the clarify answers
  folded into the goal text and emits the typed `WORK_PLAN`;
  `cgx.session.tasks.scaffold` iterates `WORK_PLAN.layers`, calls
  `generate_single_scaffold_file` per entry, accumulates
  `existing_files_with_content` context across layers, and emits
  `SCAFFOLD_PATCHES` ready for the existing `APPLY` / `VERIFY`
  executors (which now accept either `CODE_CHANGE_PLAN` or
  `SCAFFOLD_PATCHES` as upstream artifact and skip cleanly when a
  greenfield project has no tests yet).
- **Greenfield router branch** --
  `Router.on_user_message` and `Router.on_decision_recorded` route by
  `session.mode`: greenfield sessions follow
  `CLARIFY_REQUIREMENTS -> ASK(clarify_answers) -> DECOMPOSE ->
  ASK(approve_plan) -> SCAFFOLD -> APPLY -> VERIFY`. EXPLORE failure
  on a missing index is also downgraded from a crash to a graceful
  `FAILED` task with a clear error message.
- **Frontend forms + artifact renderers** --
  `frontend/src/components/agent/AskUserForm.tsx` adds
  `ClarifyAnswersForm` (per-question textareas backed by the
  `REQUIREMENTS_SHEET`) and `ApprovePlanForm` (checklist over the
  `WORK_PLAN` layers with approve / reject + reason).
  `ArtifactPreview.tsx` adds `RequirementsBody`, `WorkPlanBody`, and
  `ScaffoldPatchesBody` renderers. The session mode is surfaced as a
  badge in the LiveView header.
- **API mode field** -- `AgentSessionCreateRequest.mode` (optional
  string) lets callers pin the mode explicitly; when absent the
  server runs `detect_mode` against the supplied `project_root` and
  index location and persists the decision on `Session.mode`.
- **`BOOTSTRAP_ENV` task kind + `BUILD_REPORT` artifact** -- new
  `cgx.session.tasks.bootstrap_env` executor sits between `APPLY`
  and `VERIFY` in the greenfield loop. It detects the project type
  (currently `python` via `requirements.txt` / `pyproject.toml` /
  `setup.{py,cfg}` -- everything else short-circuits with
  `outcome=skipped`), calls
  `cgx.codegen.test_runner.ensure_project_venv` to create/refresh
  `.venv` and install declared deps, then calls
  `cgx.codegen.env_manager.preflight_install` to pip-install
  undeclared top-level imports found in the applied files and
  appends successful adds to `requirements.txt` via
  `update_requirements`. The `BUILD_REPORT` artifact carries
  `project_type`, `venv_path`, `python_exe`, `installed_from`,
  `installed_packages`, `failed_installs`, `outcome`
  (`succeeded` / `failed` / `no_venv` / `skipped` / `partial`),
  `pip_log_tail`, and `applied_files`. The `Router` now wires
  `APPLY -> BOOTSTRAP_ENV -> VERIFY` for greenfield sessions while
  explore mode keeps the direct `APPLY -> VERIFY` edge; downstream
  `VERIFY` reads `python_exe` from the `BUILD_REPORT` so pytest
  runs inside the project venv.
- **`VERIFY` outcome classification** -- `VERIFY_REPORT` now
  carries an `outcome` token (`passed`, `assertions_failed`,
  `collection_error`, `no_tests_collected`, `timeout`,
  `pytest_missing`, `skipped`) derived from pytest's exit code
  plus stderr inspection. `run_pytest_paths` accepts an explicit
  `python_exe` override so the verifier no longer silently falls
  back to the host interpreter. The frontend `VerifyBody` renders
  the outcome as a coloured badge instead of a generic
  "tests failed (rc=N)" line, so collection errors (missing deps)
  are distinguishable from real assertion failures at a glance.
- **`SCAFFOLD` prompt hardening for web frameworks** -- the
  `_SINGLE_FILE_SYSTEM` prompt in `cgx.answer.engine` now
  instructs the model to exercise Flask / FastAPI / Starlette /
  Django apps through `app.test_client()` / `TestClient(app)` /
  `Client()` and to import the application from project source --
  never to bind a real port, spawn subprocesses, or rely on
  `requests`/`httpx` against `localhost`. This keeps the
  greenfield `VERIFY` step self-contained.
- **Frontend `BuildReportBody`** --
  `frontend/src/components/agent/ArtifactPreview.tsx` renders the
  new `build_report` artifact (outcome badge, project type, venv
  path, manifests, preflight-installed and failed-install lists,
  optional pip log tail). `TaskTree.tsx` ships a
  `bootstrap` badge (amber) for the new task kind. The
  `TaskKind` / `ArtifactKind` unions in `frontend/src/lib/api.ts`
  include `"bootstrap_env"` and `"build_report"`.
- **`REPAIR` task kind + `REPAIR_PLAN` artifact** -- new
  `cgx.session.tasks.repair` executor wires an autonomous repair
  cycle on top of the greenfield loop. After a failing `VERIFY`
  (`outcome` in `{assertions_failed, collection_error}`), the
  router spawns a `REPAIR` task whose executor:
  (1) reads the upstream `VERIFY_REPORT`,
  (2) classifies the failure via the deterministic, LLM-free
  `cgx.session.repair.classify` module (v1 ships
  `unittest_pytest_mix`: AttributeError on a `self.assert*` helper
  used inside a pytest-style class that does not inherit from
  `unittest.TestCase`),
  (3) walks the candidate test files with
  `cgx.session.repair.locate` (AST scan for offending
  `ClassDef`s + the `self.<helper>` calls they reference), and
  (4) emits a unified-diff patch via
  `cgx.session.repair.propose` that rewrites the class header to
  inherit `unittest.TestCase` (preserving any existing bases) and
  inserts `import unittest` if missing. The artifact content is
  `{classification, failure_signature, repair_attempt, rationale,
  locations, diffs}` shaped to drop straight into the shared `APPLY`
  executor. Router edges: `VERIFY (fixable) -> REPAIR -> APPLY
  (with build_artifact_id carried forward, so BOOTSTRAP_ENV is
  skipped) -> VERIFY`. Empty plans (classification `unknown`)
  escalate to `ASK_USER(freeform)` instead of looping. The cycle is
  capped at 2 attempts via `repair_attempt` and gated by
  `prior_failure_signatures` (`sha1` of outcome + returncode +
  first error line), so a flapping fix that keeps producing the
  same failure breaks the loop on the second pass. Tests cover
  the classifier (positive + negative + signature stability), the
  locator (find / skip-when-TestCase / preserve bases), the
  proposer (diff shape + unittest import insertion), the executor
  (end-to-end with stub store), and all router transitions
  (`VERIFY -> REPAIR`, explore-mode no-op, signature loop guard,
  budget exhaustion, `REPAIR -> APPLY`, `REPAIR -> ASK_USER`,
  `APPLY (from repair) -> VERIFY skipping BOOTSTRAP_ENV`).
- **Frontend `RepairPlanBody`** --
  `frontend/src/components/agent/ArtifactPreview.tsx` renders the
  new `repair_plan` artifact (classification badge, repair attempt
  counter, rationale, located classes + helpers, and the proposed
  diff via the shared `DiffView`). `TaskTree.tsx` ships a `repair`
  badge (rose) for the new task kind. The `TaskKind` /
  `ArtifactKind` unions in `frontend/src/lib/api.ts` include
  `"repair"` and `"repair_plan"`.
- **Second `REPAIR` classification: `missing_module_pythonpath`** --
  `cgx.session.repair.classify` now recognises pytest collection
  errors of the form `ModuleNotFoundError: No module named '<name>'`
  where `<name>` maps to a project-root sibling (a `.py` file or a
  directory containing `__init__.py` / `.py` files). The locator
  (`locate_missing_module_pythonpath`) filters out third-party
  modules that belong to `BOOTSTRAP_ENV`. The proposer
  (`propose_missing_module_pythonpath`) emits a unified diff that
  creates (or prepends to) a project-root `conftest.py` carrying a
  marker comment + `sys.path.insert(0, str(Path(__file__).parent))`,
  so pytest can resolve scaffolded packages on the next pass. A
  marker check makes the propose a no-op when the fix has already
  been applied (the router then escalates to `ASK_USER`).
- **Third `REPAIR` classification: `missing_fixture`** --
  `cgx.session.repair.classify` now recognises the pytest collection
  error `fixture '<name>' not found`. The locator
  (`locate_missing_fixture`) walks every `.py` file under
  `project_root` (skipping `.venv`, `__pycache__`, dotfile dirs, and
  the build/cache subtrees listed in `_FIXTURE_SCAN_SKIP_DIRS`),
  parses each one, and records the first top-level
  `@pytest.fixture`-decorated `FunctionDef` / `AsyncFunctionDef` whose
  name matches a missing fixture. Accepts the bare attribute form
  (`@pytest.fixture` / `@pytest.fixture(...)`) and the imported form
  (`@fixture` / `@fixture(...)`). The proposer
  (`propose_missing_fixture`) hoists the verbatim source span
  (decorators + def + body) into `tests/conftest.py` when a `tests/`
  directory exists at the root, else into project-root `conftest.py`,
  adding `import pytest` if missing and wrapping each hoisted fixture
  in a `# cgx-repair: missing_fixture <name>` marker so a second pass
  is a no-op. Empty diffs (no on-disk definition found, or markers
  already present) escalate to `ASK_USER` via the router. The
  `RepairPlanBody` UI gains a metadata entry for the new
  classification and a generalised location renderer that falls back
  through `class_name` / `fixture_name` / `module_name` and shows the
  hoist target (`→ tests/conftest.py`).
- **`BOOTSTRAP_ENV` preflight test-style lint** --
  `cgx.session.tasks.bootstrap_env` now runs
  `cgx.session.repair.locate.lint_test_style` over the
  applied test files (paths starting with `tests/` or basenames
  starting with `test_`) after `preflight_install`. The
  `BUILD_REPORT` artifact gains a `style_issues` list (`{kind, file,
  class_name, lineno, helpers}`) and the executor outputs include
  `style_issue_count`. The lint is informational -- it does not
  change the `outcome` token; REPAIR still owns the actual fix --
  but the issue list surfaces in the UI before VERIFY runs so the
  user sees a named reason instead of waiting for the AttributeError
  on the next pass. `BuildReportBody` in `ArtifactPreview.tsx`
  renders the list in an amber section under the manifests block.
- **Stale-session recovery on the Agent page** --
  `frontend/src/lib/api.ts` now exports a typed `ApiError` (with
  `status`, `path`, `body`) that `jsonReq` throws on non-2xx
  responses instead of a generic `Error`. `AgentPage.tsx`'s
  `loadState` catches `ApiError` with `status === 404` on the
  active session id, clears the persisted `activeId` /
  `selectedTaskId` from the `cgx-agent-session` zustand store, and
  refreshes the sidebar so the launcher takes over -- previously a
  deleted-out-of-band or project-root-swapped session id stuck in
  `localStorage` would re-fire the same 404 on every mount.

### Changed

- **`ActiveTask.resolveLinkedArtifact`** -- the upstream-artifact
  lookup is no longer a hardcoded list of explore-mode keys; it
  scans `task.inputs` for any `*_artifact_id` key so new flows
  (`requirements_artifact_id`, `work_plan_artifact_id`,
  `scaffold_patches_artifact_id`, ...) pick up automatically.
- **`VERIFY_REPORT.content` + `outputs`** -- both now carry a
  `failure_signature` (sha1, 16 hex chars) so the router's progress
  detector can compare attempts without re-reading the artifact.

### Docs

- `docs/Agent.md`, `docs/architecture.md`, `docs/flowcharts.md`,
  `docs/usage.md`, and `README.md` document the greenfield loop, the
  mode auto-detection rules, the new API field, the
  `BOOTSTRAP_ENV` step (with the `BUILD_REPORT` artifact shape and
  the `VERIFY` outcome enum), and the autonomous `REPAIR` cycle
  (classification taxonomy, retry budget, progress detector).

### Added

- **`cgx.session` core (Phase 0)** -- new package implementing the
  session backbone. Pydantic-free dataclasses in
  `cgx.session.models` (`Session`, `TaskNode`, `Fact`, `Artifact`,
  `Decision`, `KnowledgeBase`, `DecisionLog`); SQLite persistence in
  `cgx.session.store.SessionStore` (one DB per project root at
  `<project_root>/.cgx/sessions.db`, WAL mode, JSON-blob rows with
  indexed columns); in-process publish/subscribe `EventBus` in
  `cgx.session.events`. `TaskKind` covers `EXPLORE`, `INVESTIGATE`,
  `RECOMMEND`, `PLAN_CHANGE`, `APPLY`, `VERIFY`, `ASK_USER`,
  `SEARCH`, `SUMMARIZE`. `TaskNodeStatus` runs through
  `PENDING -> BLOCKED -> READY -> IN_PROGRESS -> DONE`/`FAILED`/
  `ABANDONED`; `ASK_USER` deliberately stays `IN_PROGRESS` until a
  `Decision` arrives. Covered by `tests/test_session.py`.
- **Deterministic Router (Phase 1a)** --
  `cgx.session.router.Router` is pure Python with no LLM calls and
  no I/O. Three entry points (`on_user_message`,
  `on_task_completed`, `on_decision_recorded`) cover every
  transition; each returns a `RouterPlan` of typed actions
  (`CreateTask`, `UpdateTaskStatus`, `RecordDecision`,
  `AttachDecisionToTask`) for the caller to apply atomically. A
  `TASK_SUCCESSOR` table fixes the non-ASK successors
  (`EXPLORE -> ASK_USER(choose_path)`, `INVESTIGATE -> RECOMMEND`,
  `RECOMMEND -> ASK_USER(choose_recommendation)`,
  `PLAN_CHANGE -> ASK_USER(approve)`, `APPLY -> VERIFY`); ASK
  successors are driven by the shape of the resolving `Decision`.
- **Executor registry (Phase 1b)** --
  `cgx.session.tasks.base.register_executor(kind)` registers a pure
  function `(TaskNode, ExecutorDeps) -> ExecutorResult` against a
  `TaskKind`. Executors do not write to the store directly; the
  runner persists their `outputs`, surfaced `facts`, and produced
  `artifact` after the call. `ExecutorDeps` carries optional
  `project_root`, `index_dir`, `records_path`, `embed_model`,
  `provider`, `store`, and an `extra` dict; executors return
  `ExecutorResult(failure=...)` if a required dep is missing rather
  than raising.
- **EXPLORE executor (Phase 1c)** --
  `cgx.session.tasks.explore` runs retrieval-grounded
  `clarify_paths` against the project index and produces a
  `DIRECTIONS_LIST` artifact plus one `ANCHOR` fact per option.
  Bypasses `answer_with_llm`'s Markdown round-trip by reading
  structured `debug["options"]` directly so the typed option payload
  reaches the UI without re-parsing.
- **ASK_USER + decision validation (Phase 1d)** --
  `cgx.session.tasks.ask.build_decision(session_id, task, chosen,
  rationale)` validates incoming `chosen` payloads against the
  task's `inputs["expected_kind"]` and raises `ValueError`
  (rendered as HTTP 400) on mismatch:
  - `choose_path` requires `anchor_chunk_id` (non-empty).
  - `choose_recommendation` requires `kind in {investigate_more,
    plan_change, ask_followup, done}`; `kind=investigate_more`
    additionally requires `anchor_chunk_id`.
  - `approve` requires `approved: bool`.
  - `freeform` requires only `text`.
  Successor spawning is delegated to `Router.on_decision_recorded`
  so the contract has one source of truth.
- **SessionRunner (Phase 1e)** --
  `cgx.session.runner.SessionRunner` is the orchestrator the HTTP
  routes call. Holds a per-session `threading.Lock` (guarded by an
  outer lock) so concurrent requests can't interleave half-applied
  plans; sequences router plans (creates -> decisions -> attaches
  -> status updates) so a spawned child is visible by the time a
  parent flips to `DONE`; centralises executor dispatch + failure
  handling (missing executor or uncaught exception -> task
  transitions to `FAILED` with a helpful message; surfaced facts
  still persist). Public API: `start_session`, `post_message`,
  `post_decision`, `run_next`.
- **`/api/agent-session/*` HTTP surface (Phase 1f)** --
  `cgx.webui.routes.agent_session` mounts six JSON endpoints
  alongside (not replacing) the legacy `/api/agent` SSE route:
  `POST /api/agent-session` (create + drain READY),
  `GET /api/agent-session?project_root=...` (list sessions),
  `GET /api/agent-session/{sid}` (full snapshot),
  `POST /api/agent-session/{sid}/message` (follow-up; spawns a
  sibling `EXPLORE` when no `ASK_USER` is open),
  `POST /api/agent-session/{sid}/decision` (resolve a pending
  `ASK_USER`), `DELETE /api/agent-session/{sid}` (discard the
  session and its aggregate via SQLite `ON DELETE CASCADE`;
  returns `{deleted: sid}`). Every mutating endpoint returns the
  full `AgentSessionState` snapshot except `DELETE`; mutating
  endpoints drain the loop
  in a thread (`_drain_ready`, capped at four steps) until an
  `ASK_USER` pauses. A per-`project_root` runner cache (`_RUNNERS`)
  reuses one `SessionStore` (and its SQLite WAL connection) across
  requests. New Pydantic wire models in `cgx.webui.models`
  (`AgentSessionCreateRequest`, `AgentSessionMessageRequest`,
  `AgentSessionDecisionRequest`, `AgentSessionState`, ...).
- **INVESTIGATE + RECOMMEND executors (Phase 2)** --
  `cgx.session.tasks.investigate` runs an anchored retrieval keyed
  on the chosen `anchor_chunk_id`, consults the per-session
  `KnowledgeBase` to skip redundant work, and produces a
  `FINDINGS_BUNDLE` artifact plus `SYMBOL` facts.
  `cgx.session.tasks.recommend` makes a structured (JSON-mode) LLM
  call and produces a `RECOMMENDATION_LIST` of 2-4 typed
  recommendations (`kind in {investigate_more, plan_change,
  ask_followup, done}`). Router successor table extended:
  `INVESTIGATE -> RECOMMEND`, `RECOMMEND -> ASK_USER(choose_recommendation)`,
  `CHOOSE_PATH decision -> INVESTIGATE`.
- **PLAN_CHANGE / APPLY / VERIFY executors (Phase 3)** --
  `cgx.session.tasks.plan_change` wraps `generate_code_plan` and
  produces a `CODE_CHANGE_PLAN` artifact (unified diffs + plan_md).
  `cgx.session.tasks.apply` wraps `apply_diffs_to_disk` and
  produces an `APPLIED_CHANGES` artifact carrying `applied_files`,
  `failed_files`, and the per-run `backup_dir` (same mirror under
  `<project_root>/.cgx-backups/<run_id>/` the legacy loop uses, so
  `POST /api/rollback` still works on session runs).
  `cgx.session.tasks.verify` runs the impacted-tests harness and
  produces a `VERIFY_REPORT`. Router successors:
  `CHOOSE_RECOMMENDATION(plan_change) -> PLAN_CHANGE`,
  `PLAN_CHANGE -> ASK_USER(approve)`,
  `APPROVE(approved=true) -> APPLY`, `APPLY -> VERIFY`.
  `APPROVE(approved=false)` returns no successor; the user can
  pivot with a fresh objective.
- **Session-shaped Agent page (Phase 4)** --
  `frontend/src/pages/AgentPage.tsx` is now the session-shaped page
  at `/agent`; the original page moved to
  `frontend/src/pages/AgentLegacyPage.tsx` and is mounted at
  `/agent-legacy`. New components under
  `frontend/src/components/agent/`: `SessionLauncher`, `TaskTree`
  (hierarchical DAG keyed on `parent_task_id`; depth-based
  indentation, status icons, orphan re-surfacing),
  `ActiveTask` + `AskUserForm` (dispatch on `expected_kind` to
  `ChoosePathForm` / `ChooseRecommendationForm` / `ApproveForm` /
  `FreeformForm`, posting exactly the shapes `build_decision`
  expects), `SidePanel` + `ArtifactPreview` (tabbed Knowledge-Base
  and Artifacts view; per-kind renderers), `LiveView`.
  `frontend/src/store/agentSession.ts` is a Zustand + `persist`
  store (key `cgx-agent-session`) that holds the active session id
  and selected task id in `localStorage` so a tab switch / reload
  resumes the same view; the snapshot itself is reloaded from
  `/api/agent-session/{sid}` on mount rather than cached
  client-side. Sidebar entry updated to surface the session UI as
  default and `/agent-legacy` as an explicit secondary link.
- **Frontend component tests** --
  `frontend/src/components/agent/AskUserForm.test.tsx` (6 cases)
  pins the wire-shape contract for every `expected_kind` variant
  (including empty-state handling on `ApproveForm` when the
  `CODE_CHANGE_PLAN` is missing, `Send` disabled on empty freeform
  text, and `pending=true` disabling each form).
  `TaskTree.test.tsx` (6 cases) covers the DAG renderer (root
  pinning, depth-based indentation, selection ring, orphan
  re-surfacing, `onSelect` payload).
- **Backend integration tests** --
  `tests/test_webui_agent_session.py` drives the full write loop
  (`EXPLORE -> choose_path -> INVESTIGATE -> RECOMMEND ->
  choose_recommendation(plan_change) -> PLAN_CHANGE -> approve ->
  APPLY -> VERIFY`) directly against the FastAPI route handlers
  through a `_HandlerClient` shim. Pydantic still parses every
  payload via `AgentSessionCreateRequest` /
  `AgentSessionDecisionRequest`, so the wire contract is exercised
  end-to-end without taking an `httpx` test-time dependency. Stub
  executors are registered for every `TaskKind`; the
  per-`project_root` `_RUNNERS` cache is cleared and each runner's
  store is closed between tests. Coverage: full happy-path loop,
  decline-approval halts the loop, decision validation rejects an
  empty anchor on `choose_path`, snapshot + list endpoints,
  follow-up message spawns a sibling `EXPLORE` after a `done`
  recommendation clears focus.

### Changed

- **`/agent` route now serves the session-based UI by default**;
  the previous batch Planner / Tracker / Judge view moved to
  `/agent-legacy`. The sidebar surfaces both. No backend behaviour
  change for legacy callers -- the `cgx agent` CLI, the
  `cgx.agents.run_agent` Python API, and the `/api/agent` SSE route
  all continue to drive the batch loop unchanged.
- **Documentation refresh** -- `docs/Agent.md`,
  `docs/architecture.md`, `docs/usage.md`, `docs/flowcharts.md`,
  and `README.md` updated end-to-end with a new top-level section
  for the session-shaped agent, explicit cross-references to the
  legacy view, and a programmatic example driving `SessionRunner`
  directly. `docs/usage.md` sections 7-15 renumbered to 8-16 to
  accommodate the new "Session-based Agent (`/agent`)" section.

### Notes

- **No re-index required.** The session backbone reuses the same
  retrieval, codegen, and provider stacks as the batch loop; only
  the state model, interaction model, and execution model differ.
- **No new runtime dependencies.** SQLite is in the stdlib; the
  session UI is built from the same React / Zustand stack already
  in use elsewhere in the frontend.
- **Database location.** One database file per project root at
  `<project_root>/.cgx/sessions.db` (or `~/.cgx/sessions.db` when
  no project root is supplied -- typical for interactive scripts
  and tests with a tmp `HOME`).

## Unreleased -- Ask UI overhaul + Gemini stream resilience + patch context verification

### Added

- **Citation chips in rendered answers** (`frontend/src/components/Markdown.tsx`):
  a new `preprocessCitations` step rewrites both `[[chunk_id]]` markers
  (emitted by `cgx.answer.engine`) and single-bracket `[path::kind::name]`
  variants into compact emerald `CitationChip` pills displaying just the
  symbol name with the full id surfaced via tooltip. Previously these
  markers rendered as raw bracketed text, which cluttered every paragraph
  with long `[[/src/cgx/retrieval/lexical.py::method::LexicalIndex.search]]`
  strings. No backend changes -- the structured `citations` array on the
  response is untouched.
- **Collapsible Sources panel** (`frontend/src/pages/AskPage.tsx`):
  the retrieval-ranks column is now hidden by default behind a
  `Sources (N)` toggle in the new `ChatHeader` strip. Clicking reveals
  the existing `RetrievalPanel` as a right-hand drawer. Frees the main
  reading area for the answer and matches the cleaner Odysseus-style
  layout.
- **Redesigned Ask bar** (`frontend/src/pages/AskPage.tsx`): taller
  textarea, larger placeholder, emerald focus glow, icon prefix,
  pill-shaped Ask button, and a `Enter to send · Shift+Enter for newline`
  hint underneath. Markdown body typography in `Markdown.tsx` was also
  refreshed (comfier reading size, tighter heading hierarchy, real card
  around fenced code blocks).
- **`_format_stream_failure` helper** (`cgx.webui.handlers`): centralised
  formatter for `thought_warning` SSE payloads. `RuntimeError` messages
  raised by providers (already pre-scrubbed) render without the redundant
  class prefix; other exceptions keep `ClassName: msg` so the cause
  remains visible. Used by both `stream_ask` and `stream_plan`.
- **Regression tests for patch context verification**
  (`tests/test_codegen_pipeline.py`): four new tests --
  `test_apply_hunks_exact_match_applies`,
  `test_apply_hunks_drifted_line_numbers_fuzzy_locates`,
  `test_apply_hunks_hallucinated_context_is_rejected`,
  `test_apply_hunks_ambiguous_match_is_rejected` -- pin the new
  context-verification contract.

### Changed

- **`GeminiProvider.chat_stream` now retries transient transport errors**
  (`cgx.answer.providers`). `SSLError`, `ConnectionError`, `Timeout`,
  and `ChunkedEncodingError` raised *before* any delta has been yielded
  are retried with exponential backoff up to `max_retries` times (default
  3) via the existing `cgx.answer.ratelimit.backoff_seconds` helper. A
  mid-stream break (after deltas have flowed) cannot be safely replayed
  -- it would duplicate downstream content -- so it instead raises a
  scrubbed `RuntimeError` the caller can surface as a clean warning.
  Hard HTTP errors (4xx / 5xx) raise immediately with the status code
  rather than retry. All error messages run through
  `GeminiProvider._scrub_secret` so the API key cannot leak into UI
  banners or logs. Previously a single TLS hiccup yielded
  `[stream error: SSLError: ...]` into the planner-thinking pane and
  killed the turn.

### Fixed

- **`cgx.codegen.diff_apply._apply_hunks` silently corrupted files when
  the model emitted wrong line numbers or hallucinated context lines**.
  The previous implementation treated `@@` line numbers as authoritative
  and blindly overwrote `out[anchor:anchor+consumed]` with the post-image
  without verifying that the pre-image actually matched the buffer.
  Repro: a diff with `@@ -1,3 +1,3 @@` followed by context line
  ` def NONEXISTENT_func():` (a function that doesn't exist in the file)
  was reported as `ok=True` and silently replaced the real `def add()`,
  producing invalid Python that then failed the downstream
  `validate_patch_results` syntax check. The rewrite:
  - Introduces `_build_hunk_images` which splits each hunk body into a
    **pre-image** (context + deletion lines) and **post-image**
    (context + addition lines).
  - Introduces `_locate_pre_image` which locates the pre-image in the
    working buffer using (1) the `@@` hint, (2) a ±50-line sliding
    window, and (3) a global unique-match fallback. Ambiguous global
    matches (more than one location matches) are rejected rather than
    guessed.
  - Hunks whose pre-image cannot be located are added to
    `rejected_hunks` and the file content is **preserved
    byte-for-byte**, so the downstream syntax validator now sees the
    real failure ("partial apply" / "rejected hunk") instead of an
    after-the-fact `SyntaxError` on corrupted output.

## Unreleased -- Gemma 4 model family + pull-error fix

### Added

- **Gemma 4 catalogue entries** (`cgx.answer.hardware_matrix.LOCAL_MODEL_CATALOG`):
  five rows covering the full Gemma 4 family as published on the
  Ollama library -- `gemma4:e2b` (~7.2 GB on disk, edge), `gemma4:e4b`
  (~9.6 GB, also served as `gemma4:latest`), `gemma4:12b` (~7.6 GB,
  workstation dense), `gemma4:26b` (MoE, ~18 GB, 4B active per token),
  `gemma4:31b` (~20 GB, near-cloud quality). Context windows: 128K
  for E2B / E4B, 256K for 12B / 26B / 31B. Families:
  E2B / E4B / 12B → `general`; 26B A4B / 31B → `reasoning`.
- **`gemma4` family registered** in
  `cgx.answer.model_caps._MODEL_CONTEXT_TOKENS` at 128_000 tokens.
  Conservative on purpose: the family-prefix matcher routes every
  `gemma4:*` Ollama tag to this entry, and 128K is the floor across
  the family (E2B / E4B), so the prompt-budget tier selector in
  `get_summary_budget` / `get_context_map_budget` never overflows the
  smaller models even when the 12B / 26B / 31B variants actually
  support 256K natively.
- **Recommended ladder entries** in
  `cgx.answer.ollama_discovery.RECOMMENDED_LADDER`: `gemma4:e2b` and
  `gemma4:e4b` are surfaced as general-purpose alternatives in the
  hardware-aware setup flow (`cgx.webui.routes.setup`) alongside the
  existing Qwen Coder / Llama options.

### Changed

- **Hardware matrix sort order** (`compute_local_fit` in
  `cgx.answer.hardware_matrix`): rows now group by `family`
  (coder → general → reasoning) and ascend by `params_b` within each
  family, so related models cluster on the Hardware page. Previous
  order was by `params_b` globally. The corresponding test
  (`tests/test_hardware_matrix.py
  ::test_compute_local_fit_rows_grouped_by_family_then_params`)
  was updated to assert the new contract.

### Fixed

- **Ollama pull silently reported success on failure**. When the
  local Ollama instance returned a non-2xx for `/api/pull` (e.g.,
  HTTP 412 because the installed Ollama is older than the model's
  manifest format -- see ollama/ollama#15222 -- or 404 for a typo'd
  tag), the SSE stream emitted a single `status="error"` progress
  event followed by `done`, and the UI's close handler
  unconditionally wrote `done: true, status: "Download complete"`.
  The user saw a 2-second "successful" pull, then `ping` reported
  the model as not installed. Three changes fix this:
  - `cgx.webui.routes.setup.ollama_pull` now formats HTTP failures
    explicitly (status code + truncated response body) instead of
    relying on `raise_for_status` to box them as opaque exceptions,
    so the UI sees `"ollama /api/pull returned HTTP 412 for
    model='gemma4:12b': ..."` rather than just `"HTTPError"`.
  - `frontend/src/lib/pullManager.ts` (`startPull`) and
    `frontend/src/pages/SettingsPage.tsx` (`startEditPull`) both
    detect `status === "error"` in the progress stream, capture
    `data.error` into the `error` field, and refuse to overwrite an
    existing error with `"Download complete"` in the close handler.
    If the stream closes without ever reporting `status="success"`
    *and* no error was emitted, the UI now flags
    `"Pull ended without success; see Ollama logs."` rather than
    falsely claiming completion. The existing `PullProgress`
    sub-component already rendered `pull.error` in red, so the fix
    surfaces immediately in both the active-provider card and the
    edit-profile modal.

### Notes

- No re-index required; the changes are catalogue + capability data
  plus surface-level pull-flow fixes.
- Backend test suite: 474 passed (same as the prior baseline). No
  new test files were added; `tests/test_hardware_matrix.py` was
  updated in place to reflect the family-grouped sort contract.

## Unreleased -- Retrieval & codegen pipeline overhaul (Phases 0–9)

A 9-phase overhaul of the retrieval, parsing, and prompt-assembly
layers. Behavior-preserving where noted; SLM-prompt and insertion
output shapes changed in two phases. **Re-indexing required** -- see
the *Schema version* note under **Changed**.

### Added

- **Phase 1 -- Symmetric sub-word tokenizer** (`cgx.retrieval.tokenize`):
  `split_identifier(name)` splits camelCase / PascalCase / snake_case /
  kebab-case identifiers into ordered sub-tokens; `expand_with_subwords
  (tokens, *, min_len=1)` is the dedup wrapper used on both sides of
  retrieval. Wired into `cgx.embeddings.helpers._split_tokens` (indexer
  side, feeds `lexical_helpers.ngrams_*`) and
  `cgx.retrieval.orchestrator._tokenize_lc` /
  `_extract_symbol_tokens` (query side). Identifier matching is now
  symmetric -- a query for `parseConfig` hits records tokenized from
  `parse_config` and vice-versa. Covered by `tests/test_tokenize.py`
  plus a camelCase ↔ snake_case integration assertion.
- **Phase 3 -- Tiered SLM context (Code Map)** (`cgx.answer.context_map`
  + `cgx.answer.model_caps.get_context_map_budget`): when the retriever
  surfaces graph-expanded neighbors (`provenance.graph_depth >= 1`),
  the prompt SOURCES list is built as two tiers -- full focus-windowed
  bodies for primary hits, one-line `[class.]name(signature) -- doc`
  stubs for neighbors, tagged `tier=neighbor` in the prompt metadata.
  Budgets (`primary_chars`, `neighbor_chars`, `primary_max`,
  `neighbor_max`, `total_chars`) scale by the provider's model context
  window (4 tiers at 16K / 64K / 200K boundaries). Activation is
  automatic: queries whose hit list contains no graph-expanded chunks
  fall back to the legacy single-tier builder, so existing prompts are
  byte-identical. Public API: `load_records_by_id`, `classify_hits`,
  `format_neighbor_stub`, `build_tiered_context`. Wired into both
  `answer_with_llm` and `generate_code_plan` via the same
  `_has_neighbors` gate in `cgx.answer.engine`. Covered by
  `tests/test_context_map.py`.
- **Phase 4 -- Line-anchored insertion points**: every record now
  carries `start_line`, `end_line`, and `col_offset` (mirrored from the
  parser chunk's AST node); see *Changed → Schema version* below.
  `cgx.retrieval.orchestrator.suggest_insertion_points` emits two new
  per-container anchor fields, `likely_caller_loc` and
  `similar_signature_neighbor_loc`, each shaped `{"start_line": int,
  "end_line": int, "indent_col": int}` (or `None` when the anchor
  chunk has no line info). `cgx.codegen.ast_insert` now prefers
  line-anchored splice over the existing AST-walk path when the new
  fields are present, falling back to AST-walk for legacy / v2
  records. `tests/snapshots/suggest_insertion_points_shape.json`
  pins the new output shape; `tests/test_ast_insert.py` covers the
  line-anchored splice paths.
- **Phase 6 -- `CodeGraphBackend` facade** (`cgx.graph.backend`): a
  thin wrapper around the small set of `networkx` operations the
  retrieval and embeddings layers actually need
  (`has_node`, `successors`, `predecessors`, `undirected_neighbors`,
  `edge_attrs`, `node_attrs`, `bfs_distances`, plus a `wrap(G)` factory
  that returns `None` when `G` is missing). `cgx.retrieval.orchestrator`
  multi-hop expansion and `cgx.embeddings.helpers._neighbors_summary`
  now go through the facade; `build_graph`, graph visualization, and
  graph persistence still use raw `networkx` (no dependency change).
  Covered by `tests/test_graph_backend.py`.
- **Phase 7 -- Parser schema + `BaseParser` seam (Python-only)** --
  `src/cgx/parser/schema.py` formalizes today's record shape via
  `CodeChunk`, `CallRelation`, and `ChunkType` `TypedDict`s, with
  `total=False` so the variable `meta` payloads keep their existing
  per-chunk-type contracts. `src/cgx/parser/base.py` introduces the
  `BaseParser` ABC: a single `parse_file(filepath, source_code,
  project_root) -> (chunks, call_relations)` method plus a lowercase
  `extensions` tuple drives extension-based dispatch. `src/cgx/parser/
  python_parser.py` provides `PythonASTParser` and registers `.py`.
  `parse_codebase` was split into a project walker (registry dispatch,
  ignore/safety knobs, cross-file post-processing -- call-relation
  dedup, `calls_out_top`, `called_by_count`) and a module-level
  `_parse_python_module` worker (file/module chunk emission + the
  existing AST `CodeVisitor`). The chunk and call-relation shapes are
  byte-identical to before -- `tests/test_schema_snapshots.py` still
  passes -- and the dispatcher silently skips files whose extension is
  not registered (so `.py`-only behavior is preserved). Covered by
  `tests/test_parser_seam.py` (10 cases: ABC contract, registry shape,
  per-file output keys, syntax-error tolerance, worker-vs-parser
  equality, project-level aggregation, non-`.py` skip).
- **Phase 9 -- Reranker profile policy** -- `cgx.answer.profiles.Profile`
  gains an optional `enable_reranker: Optional[bool]` field. `None`
  (the default) means "auto" and resolves through
  `default_reranker_for_kind(kind)` -- `True` for cloud kinds
  (`openai-compat`, `gemini`) and `False` for local / private kinds
  (`ollama`, `custom`). Explicit `True` / `False` on the profile wins.
  `resolve_enable_reranker(profile)` is the single public helper that
  returns the effective flag, and the value is persisted by
  `save_profile` / `list_profiles` only when set explicitly (so `None`
  stays "auto" across edits). The flag threads through
  `cgx.retrieval.orchestrator.hybrid_retrieve_two_view` (new kwargs:
  `enable_reranker`, `reranker_model`, `reranker_top_n`,
  `reranker_weight`) and `cgx.pipeline.auto.run_query_auto` (new kwarg:
  `enable_reranker`) into `HybridConfig`. When unset on both layers the
  pre-existing `HybridConfig` defaults (reranker off) are preserved.
  Covered by `tests/test_reranker_profile.py` (15 cases: per-kind
  defaults, explicit-overrides-kind, save/load round-trip incl. `None`
  preserved, threading into `HybridConfig` for all three flag states,
  reranker knobs propagation, deterministic RRF order when disabled,
  cross-encoder reorders head when enabled).

### Changed

- **Schema version: `SCHEMA_VERSION` bumped `1 → 3`** in
  `cgx.embeddings.records`. v2 (Phase 1) added the symmetric sub-word
  tokenizer to the lexical / catalog pipeline -- v1 records under-match
  partial-name queries. v3 (Phase 4) adds `start_line` / `end_line` /
  `col_offset` to every record so insertion planners can splice
  without re-walking the AST -- v2 records lack these fields.
  **Re-index advisory**: indices built before this overhaul should be
  rebuilt by re-running `cgx index --project-root … --out-dir …` (or
  triggering *Re-index* from the UI). Readers detect a stale
  `schema_version` on the persisted manifest and treat the cache as
  invalid so a rebuild is the safe path.
- **`suggest_insertion_points` output shape**: containers now expose
  `likely_caller_loc` and `similar_signature_neighbor_loc` alongside
  the existing `likely_caller` / `similar_signature_neighbor` chunk
  ids; the snapshot in `tests/snapshots/suggest_insertion_points_shape
  .json` documents the v3 shape.
- **`cgx.codegen.ast_insert`** prefers the new line-anchored splice
  when records carry `start_line` / `end_line`; the existing AST-walk
  path is retained as a fallback for v2-and-older indices.

### Internal (refactors, performance, test infrastructure -- no public-API change)

- **Phase 0 -- Schema-version constant + golden-output snapshots**:
  added `SCHEMA_VERSION` to records / persisted manifests and
  `tests/test_schema_snapshots.py` with three pinned snapshots
  (record-keys, `suggest_insertion_points` shape, top-K hybrid
  retrieval over a synthetic repo). Subsequent phases land against
  these snapshots so any shape drift is caught immediately.
- **Phase 2 -- Parser helpers lifted to module scope**: `_build_file_
  code_stub`, `_collect_top_level_members`, `_class_signature`, and
  the surrounding stub builders were hoisted out of the
  `parse_codebase` closure to module scope in
  `cgx.parser.parse_codebase`. Pure refactor -- no behavior change --
  enabling unit-testing of the helpers in isolation
  (`tests/test_parser_helpers.py`) and the Phase 7 parser-seam split.
- **Phase 5 -- Exemplar-embedding LRU cache** in
  `cgx.retrieval.orchestrator`: `_build_exemplar_corpus(records,
  embedder)` is now memoised behind `_insertion_corpus_key` (keyed by
  records identity + `schema_version` + embedder fingerprint) with a
  bounded LRU. Repeat calls to `suggest_insertion_points` for the
  same index reuse a single encoded corpus matrix. A
  `_clear_insertion_corpus_cache()` helper supports test teardown.
  Covered by `tests/test_insertion_cache.py` (corpus encoded once
  across repeat calls; cache invalidates on records-id change).
- **Phase 8 -- Optional Tree-sitter plugin: DROPPED**. Multi-language
  parsing deferred to a later cycle; Phase 7's parser registry already
  provides the seam.

## Unreleased -- Manifest-driven scaffolding, rollback API, refactor batches B1–B9

### Added

- **`cgx.codegen.ast_insert`** -- AST-anchored insertion planner that
  bridges `cgx.retrieval.orchestrator.suggest_insertion_points` into the
  existing `PatchResult` pipeline. Given an `AstInsertSpec(rel_path,
  code, class_name=None, anchor_symbol=None)` (or a raw suggestion dict
  via `plan_ast_insertion_from_suggestion`), the planner re-parses the
  target file with the stdlib `ast` module, locates the anchor
  sibling's `end_lineno`, auto-detects container body indentation, and
  splices the snippet in. `ast.get_source_segment` plus a leading-comment
  walker preserve user formatting and `#` comments. The result is
  re-parsed before being returned, so a broken splice surfaces as
  `ok=False` rather than a corrupted file; nothing is written to disk.
  `build_unified_diff(patch_result)` renders the plan as a standard
  unified diff so it routes back through `parse_fenced_diffs` /
  `apply_diffs_to_disk` / `validate_patch_results` without any
  special-casing. The module is purely additive -- no existing
  signature in `diff_apply`, `validate`, `disk_apply`, or
  `orchestrator` was modified. Covered by `tests/test_ast_insert.py`
  (12 cases: module-after-anchor, append-when-anchor-missing,
  class-after-sibling-method, dedupe-no-op, non-`.py` rejection,
  snippet `SyntaxError`, new-file creation, class-not-found,
  leading-comment preservation, suggestion-bridge for class
  containers, unified-diff round-trip, nested-class rejection).
- **`TaskKind.SCAFFOLD_MANIFEST` and `TaskKind.SCAFFOLD_FILE`**
  (`cgx.agents.types`): the monolithic `scaffold` kind has been split
  into a two-stage pipeline. `scaffold_manifest` calls
  `plan_scaffold_manifest` (a cheap LLM call that returns only the
  layered file list -- no contents) and emits an `inject_tasks` payload;
  the Tracker injects one `scaffold_file` task per planned file into
  the plan immediately after, ordered layer-by-layer so dependency-heavy
  files (core types, utilities) are generated before the files that
  import them. Each `scaffold_file` task calls
  `generate_single_scaffold_file` with the target path, its layer, and
  the full content of files already generated by earlier
  `scaffold_file` tasks. The original `scaffold` kind is retained for
  legacy callers / tests that pass a custom capability map.
- **`scaffold_manifest` / `scaffold_file` capabilities**
  (`cgx.agents.loop._build_default_capabilities`): wire the new task
  kinds to the engine functions; per-file generation keeps each LLM
  call focused on a single output and surfaces per-file progress in
  the UI.
- **Tracker support for the manifest split**
  (`cgx.agents.tracker`): `_dispatch`, `_summarize_task_output`, and
  `_extract_display_output` handle `scaffold_manifest` (file-count
  preview) and `scaffold_file` (path + size preview); the
  `inject_tasks` mechanism inserts the per-file tasks in the correct
  layer order without re-running the planner.
- **Judge `SCAFFOLD_MANIFEST` / `SCAFFOLD_FILE` structural rules**
  (`cgx.agents.judge._structural_check`): manifest tasks pass on a
  non-empty layered file list; per-file tasks pass on a non-empty
  `content` payload that parses cleanly for known source extensions.
- **`POST /api/rollback`** (`cgx.webui.routes.rollback`): REST endpoint
  that reverses the most recent `apply` run by reading the run's
  backup mirror under `<project_root>/.cgx-backups/<run_id>/`.
  Restores any files that existed before the run, deletes any files
  the `apply` step created from scratch, and returns
  `{restored_files, deleted_files, failed_files, error}`. The Agent
  tab's **Undo** button calls this endpoint.
- **`cgx.codegen.disk_apply.rollback_from_backup(project_root,
  backup_dir)`**: pure helper that drives the rollback logic and can
  be invoked directly from Python or via the REST endpoint.
- **`cgx.embeddings.loader.load_embedder(spec)`**: single source of
  truth for resolving an embedder spec (`module:attr`, model id, or
  fallback hash embedder). All callers (`cli.main`,
  `retrieval.cli_adapter`, `pipeline.auto`, the webui handlers) now
  import the shared loader instead of carrying their own copies.

### Changed

- **Planner emits `[scaffold_manifest, apply, verify]`** for SCAFFOLD
  goals instead of `[scaffold(s)…, apply, verify]`
  (`cgx.agents.planner`); kind-policy logging line lists the new
  pipeline.
- **`apply` capability** (`cgx.agents.loop`): now consumes file
  outputs emitted by `scaffold_file` tasks (in addition to
  `plan`-style diffs) and includes the per-run `backup_dir` in its
  return value so the UI can show the path used by `/api/rollback`.
- **Documentation refresh (Phase F)**: hand-drawn SVG diagrams under
  `docs/diagrams/` (`flow_developer.svg`, `flow_company.svg`) updated
  to reflect the 10-kind `TaskKind` enum, the manifest→per-file
  scaffold flow, `cgx.codegen.disk_apply`, and the `/api/rollback`
  endpoint. Prose docs (`docs/architecture.md`, `docs/usage.md`,
  `docs/flowcharts.md`, root `README.md`) refreshed end-to-end with
  the same content and a new **Apply rollback** section.

### Refactored (batches B1–B9, no behaviour change)

- **B1 -- Lazy `cgx.webui` imports** (`src/cgx/webui/__init__.py`):
  module-level `from fastapi import …` removed; symbols re-exported
  via `__getattr__` so `from cgx.webui import task_store` works
  without the `[ui]` extra installed.
- **B2 -- Graph projection consolidation** (`src/cgx/graph/`):
  `projectors.py` deleted; the two duplicate projection helpers now
  live in a single `graph.aggregation` module imported by both
  `viz.visualize` and the webui graph route.
- **B3 -- Embeddings de-duplication**
  (`src/cgx/embeddings/helpers.py`, `views.py`): the duplicated
  `_attribute_roots_read` body in `views.py` is replaced with a
  re-export of the helpers-module implementation; single source of
  truth.
- **B4 -- Shared embedder loader** (new `cgx.embeddings.loader`, see
  Added above): removes three near-identical `_load_embedder` copies
  from `cli.main`, `retrieval.cli_adapter`, and the webui handlers.
- **B5 -- Gradio drift cleanup**: removed stale references to the
  Gradio UI / port 7860 across `docs/`, `extension/`, `README.md`,
  and the React frontend (`frontend/src/layout/Header.tsx`); the
  product is React + FastAPI on port **8765** end-to-end.
- **B6 -- Judge logging hygiene** (`cgx.agents.judge`): noisy
  per-criterion `print` calls replaced with structured `logger.debug`
  output gated by `CGX_LOG_LEVEL=DEBUG`.
- **B7 -- Targeted logging** (`cgx.answer.profiles`,
  `cgx.answer.ratelimit`, `cgx.answer.ollama_discovery`,
  `cgx.codegen.diff_apply`, `cgx.codegen.pipeline`,
  `cgx.codegen.test_runner`, `cgx.codegen.validate`,
  `cgx.sessions`): replaced ad-hoc `print` statements with
  module-scoped `logging.getLogger(__name__)` calls so operator
  diagnostics route through the standard logging configuration.
- **B9 -- `.gitignore` hygiene**: added `frontend/node_modules/`,
  `extension/out/`, `frontend/dist/`, `frontend/.vite/`, and
  `cgx_index/` patterns; existing tracked artifacts left in place
  (untracking is a separate operator decision).

## Unreleased -- SLM-grade execution engine (Phases 1–5)

### Added

#### Phase 1 -- Skeleton-and-Fill (`cgx.agents`)
- **`TaskKind.FILL_LOGIC`** (`cgx.agents.types`): new task kind for the
  second pass of the skeleton-and-fill pattern. The Tracker dispatches it
  to the `fill_logic` capability, which prompts the LLM to implement
  exactly one empty function body at a time -- keeping local 7B models
  well inside their reliable generation window.
- **`fill_logic` capability** (`cgx.agents.loop._build_default_capabilities`):
  reads the target skeleton file from disk, calls the LLM with a tightly
  scoped prompt ("return only the body logic, no `def` line"), stitches
  the returned code back into the file at the correct indentation via a
  regex that matches `pass` / `# TODO` stubs, and runs an inline
  `ast.parse` smoke test on the result. Returns `{file_path,
  function_name, body_code, applied, syntax_ok}`.
- **Tracker support for `FILL_LOGIC`** (`cgx.agents.tracker`):
  `_dispatch`, `_summarize_task_output`, and `_extract_display_output`
  all handle the new kind -- the timeline row shows
  `fn_name() in file.py · stitched · syntax ok`.

#### Phase 2 -- Dynamic Dependency Management (`cgx.codegen.env_manager`)
- **New module `src/cgx/codegen/env_manager.py`**: full dependency
  management pipeline for the agent sandbox.
  - `scan_file_imports(path)` -- AST-based import extraction for `.py`
    files; regex-based for `.js`/`.ts`/`.jsx`/`.tsx`.
  - `scan_imports(file_paths)` -- union of imports across a list of files.
  - `find_missing_python_packages(imports, project_root)` -- cross-refs
    extracted roots against `requirements.txt`, then probes live
    importability; skips the full stdlib (50+ top-level names enumerated).
  - `install_packages(packages, python)` -- runs
    `pip install --quiet --no-input <pkg>` in the target Python
    interpreter (defaults to the current one); returns `{name: bool}`.
  - `update_requirements(project_root, packages)` -- appends newly
    installed packages to `requirements.txt` idempotently.
  - `preflight_install(generated_files, project_root)` -- one-shot
    convenience: scan → find missing → install → return results.
- **Pre-flight hook in `verify` capability** (`cgx.agents.loop`): before
  running pytest the `verify` capability scans every `.py` file in the
  changed set, installs missing packages into the current interpreter,
  and writes them back to `requirements.txt` so the dependency becomes
  permanent. `ModuleNotFoundError` failures caused by the model choosing
  a new library no longer mask real logic failures.

#### Phase 3 -- Symbol Table Context (`cgx.codegen.symbol_map`)
- **New module `src/cgx/codegen/symbol_map.py`**: builds a compressed
  working-memory map of all symbols already defined in the indexed
  codebase.
  - `build_symbol_map(records_path)` -- reads the JSONL records file and
    returns `{relative_path: [symbol, …]}`, deduplicated and in
    definition order.
  - `format_symbol_map(symbol_map)` -- renders the map as a
    `# AVAILABLE CONTEXT (Do not redefine these):` prompt block capped
    at 60 files × 20 symbols each so the injected block stays small.
  - `fetch_symbol_source(records_path, symbol_name)` -- AST-RAG on demand:
    scans records to return the exact source text for a named symbol,
    used by the retry loop to inject the real signature when the model
    calls a function with the wrong arguments.
  - `build_symbol_context_prompt(records_path)` -- convenience wrapper;
    returns an empty string when the records file is absent.
- **Symbol map injected into `plan` capability** (`cgx.agents.loop`):
  before calling `generate_code_plan` the capability builds the symbol
  map from `records_path` and passes it as `symbol_context`. Local models
  see what is already defined and stop redefining it.

#### Phase 4 -- Granular Error Slicing (`cgx.agents.loop`)
- **`_extract_error_snippet(project_root, responsible_files, output)`**
  (`cgx.agents.loop`): parses the first line-number reference from a
  pytest traceback, opens the failing file, and returns a ±5-line window
  around the error with an `# <-- ERROR HERE` marker -- the "10-line
  buffer rule".
- **Micro-targeted retry prompts** (`_build_fix_goal`): when an error
  snippet can be extracted, the retry goal presents it as a focused
  `` ```python `` block with a one-line error summary
  (*"Your code failed in `src/auth.py` at line 42 with
  `TypeError: …`. Here is the context around the failure:"*) rather than
  dumping the full 1 200-character pytest tail. The raw output is still
  appended as a fallback when no line number can be found.

#### Phase 5 -- Universal LLM Provider (`cgx.answer.providers`, `cgx.answer.profiles`)
- **`GeminiProvider`** (`cgx.answer.providers`): native Google Gemini
  provider via the `generativelanguage.googleapis.com` REST API.
  - Maps CGX's `messages` list to Gemini's `contents` +
    `systemInstruction` format, merging consecutive same-role turns to
    satisfy Gemini's alternating-turn requirement.
  - JSON mode via `responseMimeType: "application/json"`.
  - Streaming via `streamGenerateContent` + `alt=sse`.
  - API key read from the `api_key` constructor argument or
    `GEMINI_API_KEY` environment variable.
- **Custom-endpoint support in `OpenAICompatProvider`**: gains
  `endpoint_path` (default `"/v1/chat/completions"`) and
  `allow_no_auth` (default `False`) constructor parameters so
  self-hosted servers on non-standard paths or private subnets that
  don't require authentication work without patching the provider.
- **`Profile` dataclass expanded** (`cgx.answer.profiles`): new
  `endpoint_path: str` and `allow_no_auth: bool` fields persisted in
  `~/.cgx/profiles.json`; `list_profiles` / `save_profile` round-trip
  them correctly. `kind` now accepts `"gemini"` and `"custom"` in
  addition to `"ollama"` and `"openai-compat"`.
- **`build_provider` updated** (`cgx.webui.helpers`): handles the
  `"gemini"` kind (instantiates `GeminiProvider`) and passes
  `endpoint_path` / `allow_no_auth` through to `OpenAICompatProvider`
  for `"custom"` and `"openai-compat"` kinds.
- **`POST /api/provider/ping`** (`cgx.webui.routes.setup`): live
  connection test that returns `{ok, latency_ms, error}`.
  - Ollama: `GET /api/tags`.
  - Gemini: `POST generateContent` with `maxOutputTokens: 1`.
  - OpenAI-compat / custom: `OPTIONS` then `HEAD` on the configured
    endpoint; accepts any non-5xx status as "alive".
- **Settings page revamp** (`frontend/src/pages/SettingsPage.tsx`):
  - **Provider Type** dropdown with four options: *Ollama (Local)*,
    *OpenAI (Cloud)*, *Google Gemini (Cloud)*, *Custom Server
    (OpenAI-Compatible)*. Selecting a type pre-fills sensible defaults
    for `base_url`, `model`, and `endpoint_path`.
  - Conditional fields: API key shown for OpenAI / Gemini / Custom;
    Base URL hidden for Gemini; Endpoint Path and *Skip auth* checkbox
    shown only for Custom.
  - **Live Ping button** on both the inline config card and the
    profile edit form -- displays `OK · <Nms>` in green or the error
    message in red without leaving the form.
- **Pydantic model updates** (`cgx.webui.models`): `ProviderConfig`,
  `ProfileUpsertRequest`, and `ProfileSummary` expose `endpoint_path`
  and `allow_no_auth`; all three handler functions (`stream_ask`,
  `stream_plan`, `stream_agent`) and their routes propagate the new
  fields end-to-end.
- **`api.ts` additions** (`frontend/src/lib/api.ts`): `PingResult` type
  and `api.pingProvider(body)` method; `ProviderConfig` and
  `ProfileSummary` types include `endpoint_path` and `allow_no_auth`.
- **`workspace` store updated** (`frontend/src/store/workspace.ts`):
  default provider includes `endpoint_path`/`allow_no_auth`; `applyProfile`
  propagates the new fields.

### Changed
- `_build_fix_goal` now injects a tight code snippet instead of a raw
  truncated log when a traceback line number can be resolved
  (Phase 4 -- see above).
- `verify` capability auto-installs missing Python packages before
  running pytest (Phase 2 -- see above).
- `plan` capability injects a symbol-context block from
  `build_symbol_context_prompt` when `records_path` is available
  (Phase 3 -- see above).

---

## Unreleased -- Agent loop reliability: targeted retries, partial apply, cross-file coherence

### Fixed

- **Fix #3 -- Apply failures now trigger recursive retry** (`cgx.agents.loop`):
  `_stream_with_retry` previously checked only verify and core failures when
  deciding whether to recurse.  Apply failures (smoke-check rejections) were
  silently ignored, causing the loop to exit after one attempt even when the
  re-plan's apply step also failed.  `apply_failures` is now included in the
  recursion condition.

- **Fix #6 -- Partial apply: passing files are always written** (`cgx.codegen.disk_apply`):
  `apply_diffs_to_disk` previously returned an early-exit error and wrote
  *nothing* if any file in the batch failed the smoke check.  It now writes
  every file that passes and records the failing ones in `failed_files`.
  `smoke_ok` is `True` only when all files passed.  Retries can therefore
  target only the broken file(s) -- already-correct files stay on disk.

- **Fix #5 -- Cross-file coherence check** (`cgx.codegen.validate`):
  New `check_cross_file_coherence(patches, project_root)` function runs
  alongside the per-file syntax smoke test inside `apply_diffs_to_disk`.  It
  walks Python files in the patch batch, parses their `import` statements, and
  flags any `from X.Y import Z` where `X/Y.jsx`, `.tsx`, `.js`, or `.ts` is
  present in the same batch or on disk.  This catches the common
  mis-generation where a Python test does `from src.App import calculateResult`
  but `src/App.jsx` is a React component -- not a Python module.

- **Fix #4 -- Failure diagnosis before re-planning** (`cgx.agents.loop`):
  New `_diagnose_failure(failures)` classifies test output as
  `import_error`, `syntax_error`, `logic_error`, or `unknown`, extracts
  responsible file paths from tracebacks, detects language-mismatch cases
  (Python importing a JS/JSX module), and returns a structured dict that
  informs `_build_fix_goal`.

- **Fix #2 -- Targeted fix goals** (`cgx.agents.loop`):
  `_build_fix_goal` now uses the diagnosis to emit a *targeted* re-plan
  prompt: it names the specific broken files (from the traceback), tells
  the LLM not to change files that are already correct (read from
  `plan.owned_files`), and -- when a language mismatch is detected --
  explicitly offers two remediation paths: create a Python backend module
  that the test can import, or replace the Python test with a JS test.

- **Fix #1 -- File manifest on `Plan`** (`cgx.agents.types`, `cgx.agents.tracker`):
  `Plan` now carries an `owned_files: dict[str, str]` field (path →
  `"applied"` | `"failed"`) that the Tracker populates after every `apply`
  task.  The retry loop reads this manifest to build the "DO NOT CHANGE"
  list in targeted fix goals, so the LLM always knows which files are already
  on disk and correct.

### Added

- 28 new tests in `tests/test_agents.py` covering all six fixes: file
  manifest tracking, recursive retry on apply failure, `_diagnose_failure`
  classification, targeted fix-goal construction, cross-file coherence
  detection, and partial-apply behaviour.

## Unreleased -- Skills package: modular tech-specific knowledge bundles

### Added
- **`skills/` top-level package**: pluggable, per-technology modules that
  centralize what CGX knows about each framework / runtime / library.
  Every skill answers three orthogonal questions via the
  `skills.base.Skill` protocol: *does this goal involve me?*
  (`detect(goal) -> float`), *what should the LLM know to do my job
  well?* (`scaffold_system_prompt()`, `plan_system_prompt()`), and
  *did the produced output actually use me correctly?*
  (`validate_scaffold(diffs)`, `validate_plan(diffs)`). Initial
  registry: `react`, `nextjs`, `vue`, `tailwind`, `fastapi`, `flask`,
  `django`, `express`, `python_cli`, `sqlite`. Each skill lives in its
  own folder under `skills/<name>/` so contributors can extend the
  surface without touching the agent layer.
- **Registry dispatchers** (`skills/__init__.py`): `detect_skills(goal)`
  returns the active skills sorted by detection confidence;
  `compose_scaffold_prompt(active)` / `compose_plan_prompt(active)`
  join non-empty fragments with blank-line separators;
  `validate_scaffold(active, diffs)` / `validate_plan(active, diffs)`
  return the first failing `SkillVerdict` so the Judge can fail-fast
  with the skill's rationale; `skills_by_names(names)` resolves a
  Planner-attached name list back to instances.
- **Planner skill attachment** (`cgx.agents.planner`): every SCAFFOLD
  and PLAN task now carries `task.inputs["skills"] = [<name>, ...]`
  so downstream capabilities receive deterministic technology context.
  A new `_goal_has_supported_skill(goal)` signal augments scaffold
  detection -- goals naming a supported technology route to SCAFFOLD
  even when the noun regex doesn't fire -- while the existing `_TECH_RE`
  fallback keeps coverage for unsupported frameworks (Angular, Svelte,
  Tkinter, …). The kind-policy log line now reports
  `skills=[...]` alongside `regex=` / `llm=`.
- **Engine prompt composition** (`cgx.answer.engine`): both
  `generate_project_scaffold` and `generate_code_plan` accept a new
  `skills: Optional[List[str]]` kwarg. The system prompt is built by
  appending `compose_scaffold_prompt(active)` /
  `compose_plan_prompt(active)` to the base scaffold/plan rules with
  an `ACTIVE SKILLS:` header, so the LLM sees layout / dependency /
  convention rules specific to React + FastAPI (or whatever the active
  set is) without bloating the base prompt for every other case. The
  freeform fallback prompt gets the same treatment.
- **Judge skill validation** (`cgx.agents.judge._structural_check`):
  the hard-coded React-vs-Python check has been replaced with a call
  to `skills.validate_scaffold(active, diffs, goal=...)`. PLAN tasks
  now also run `skills.validate_plan(...)` after the codegen-report
  check, so plan-time anti-patterns (e.g. introducing class components
  into a hooks codebase) can fail-fast deterministically. When a skill
  validator fails, the Judge rationale is prefixed with `[<skill>]` so
  the operator can see which skill rejected the artifact.
- **Skill test coverage** (`tests/test_skills.py`): 14 new tests
  covering the registry shape, detection (including React-Native
  exclusion and CLI-vs-web disambiguation), composition, per-skill
  validators (React / FastAPI / Tailwind), and Planner skill
  attachment.

### Changed
- **`pyproject.toml`** and **`tests/conftest.py`**: package discovery
  + sys.path are updated so `skills` is importable as a top-level
  package both for installed runs and in-repo tests.

## Unreleased -- Scaffold routing fix: tech-paired scaffold goals + judge artifact

### Fixed
- **Scaffold detection too narrow** (`cgx.agents.planner._SCAFFOLD_RE`,
  `_goal_is_scaffold`): goals such as *"create a calculator using React
  UI and python"* slipped past the regex (which required a generic project
  noun like *app/project/cli*) and were misrouted to the change-goal
  `PLAN → APPLY → VERIFY` chain against the (empty/unrelated) index. The
  scaffold noun list now includes common archetypes (*calculator,
  dashboard, todo, blog, game, chat, editor, tracker, portfolio, landing
  page, form, page, site, gui, interface, ui*), and a second detection
  signal fires when a scaffold-friendly verb is paired with a framework
  or language name (*React, Vue, Angular, FastAPI, Flask, Django,
  Express, Python, ...*). A new `_EXISTING_CODE_HINT_RE` keeps phrasing
  like *"add a React component to our existing app"* on the change-goal
  path so the broader detection doesn't false-positive on modify-intent
  prompts.
- **LLM scaffold tasks silently dropped** (`cgx.agents.planner._enforce_kind_policy`):
  when the planner LLM correctly emitted `scaffold` tasks for a goal whose
  phrasing didn't trip the regex, the change-goal branch filtered them
  out and replaced them with a PLAN task. The policy now trusts an
  LLM-emitted scaffold decomposition whenever the goal has no
  existing-codebase hint, regardless of regex coverage.
- **Judge blind to scaffold file contents** (`cgx.agents.judge.Judge._render_artifact`):
  `scaffold` outputs were previously rendered as `json.dumps(out)[:4000]`,
  which often truncated away the actual file content and led the LLM
  judge to reject scaffolds with content-based rationales ("does not
  include input fields", etc.) it had no real evidence for. A dedicated
  SCAFFOLD renderer now surfaces `plan_md`, the full list of generated
  file paths, and a per-file content preview so the judge grounds its
  verdict in the real artifact.

### Added
- **Routing log lines** (`cgx.agents.planner._enforce_kind_policy`): each
  kind-policy branch (`SCAFFOLD`, `VERIFY-ONLY`, `READ-ONLY`,
  `CHANGE-GOAL`) now emits an `[INFO]` log line so the operator can see
  in the terminal exactly which path the planner took for a given goal.
- **Regression tests** (`tests/test_agents.py`): coverage for
  tech-paired scaffold detection, the LLM-scaffold-trust path, the
  existing-codebase exclusion, and the new SCAFFOLD artifact renderer.

## Unreleased -- Judge SCAFFOLD short-circuits on structural pass

### Fixed
- **Local 3-7B judge models hallucinate criteria fails on scaffolds**
  (`cgx.agents.judge.Judge.judge`, `_structural_check`): even with
  source-prioritized previews and goal context in the prompt, small
  local models (`qwen2.5-coder:3b`, etc.) routinely return high-
  confidence `fail` verdicts against scaffolds that demonstrably
  satisfy their criteria -- e.g. rejecting a calculator with
  `App.jsx` + `Calculator.js` + `Button.js` + FastAPI `main.py` because
  "doesn't include a calculator interface". This made the Tracker
  re-plan indefinitely. Following the same pattern already used for
  `SEARCH`/`APPLY`/`VERIFY`, `SCAFFOLD` is now short-circuited on a
  structural pass: when diffs were produced and the technology mix
  matches the goal (e.g. React goal → at least one `.jsx/.tsx/.js/.ts`
  file, not all-Python), the verdict is `pass` at 0.75 confidence and
  the LLM judge is skipped entirely. The technology-mismatch path
  still hard-fails so genuine miss-targeted scaffolds still trigger
  a re-plan.

## Unreleased -- Judge scaffold preview: source-file priority + goal context

### Fixed
- **Double-truncated scaffold artifact** (`cgx.agents.judge`):
  `_render_artifact` capped scaffold renders at 5500 chars, but
  `_llm_judge` then re-sliced the artifact to `[:4000]`, cutting roughly
  the last third of the file previews before the LLM ever saw them. The
  re-slice is removed; `_render_artifact` is the sole budget owner and
  now caps at 7500.
- **Metadata files crowded out logic-bearing source files**
  (`cgx.agents.judge._render_artifact`): scaffolds typically emit
  `README.md` / `package.json` / `requirements.txt` ahead of the actual
  component code (`App.jsx`, `Calculator.js`, ...). The previewer iterated
  files in diff order so the 6-file preview budget was burned on
  metadata before the source code was reached -- leaving the judge unable
  to verify functional criteria like *"supports +, −, ×, ÷"*. Files are
  now partitioned into source extensions (`.jsx/.tsx/.js/.ts/.py/.vue/
  .svelte/.go/.rs/.java/.kt/.rb/.php/.html/.css/.scss`) and previewed
  before metadata files, and the per-file cap was raised from 400–900 to
  800–1600 so a ~1.4 KB component fits in full.
- **Judge prompt lacked the user's goal** (`cgx.agents.judge._llm_judge`):
  per-task descriptions like *"Generate React UI components"* lacked the
  technology-stack context needed to assess multi-layer criteria. The
  planner already injects the original goal into `task.inputs["goal"]`;
  the judge prompt now surfaces it as a leading `USER GOAL:` block so
  the LLM grounds its verdict in the full request.

## Unreleased -- Scaffold prompt fix: frontend technology bias

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
  `` ```python path=src/main.py `` -- swapped to a language-neutral
  `` ```<language> path=<relative/path/to/file> `` placeholder so the
  fallback path does not bias the model toward Python when JSON mode is
  unavailable.
- **Judge blind to technology mismatch** (`cgx.agents.judge`): the
  structural check for `scaffold` tasks previously passed any output that
  contained generated files, regardless of whether those files matched
  the requested technology. A React-specific check was added: when the
  task goal or description mentions "react", the judge now hard-fails
  (0.9 confidence) if no `.js`/`.jsx`/`.tsx`/`.ts` files are present or
  if all non-config files are Python -- producing a clear rationale that
  triggers a proper retry with corrected instructions.

## Unreleased -- New-project scaffold, agent kind-policy fixes

### Added
- **`TaskKind.SCAFFOLD`** (`cgx.agents.types`): new task kind that routes
  to `cgx.answer.engine.generate_project_scaffold`. The planner emits a
  `[scaffold, apply, verify]` chain whenever the goal describes a
  *new-project* request ("create a new FastAPI app", "build from scratch",
  "generate a CLI tool", etc.). No existing index is required -- the LLM
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

## Unreleased -- Observability, task registry, tab persistence, parallel execution

### Added
- **Startup logging** (`launch.py`): `setup_logging(INFO)` is now called once at
  process start so every major operation emits structured `[INFO]`/`[WARNING]`
  lines to stdout -- handlers (`ask`, `plan`, `agent`, `index`), tracker (each
  `task_start` / `task_done` / `task_fail`), planner (LLM call, task count,
  fallback activation), and the SSE bridge.
- **SQLite task registry** (`src/cgx/webui/task_store.py`): every SSE request
  (`ask`, `plan`, `agent`, `index`) now creates a row in `~/.cgx/tasks.db`.
  All emitted SSE events are persisted per-task so the frontend can replay them
  on tab switch. An in-memory `threading.Event` per task supports cancellation.
- **Task REST API** (`src/cgx/webui/routes/tasks.py`, mounted at `/api/tasks`):
  - `GET /api/tasks` -- list recent tasks (up to 50).
  - `GET /api/tasks/{id}` -- retrieve a single task record.
  - `GET /api/tasks/{id}/events` -- full persisted event log for replay.
  - `DELETE /api/tasks/{id}` -- cancel a running task (sets its `threading.Event`).
- **SSE bridge cancellation** (`src/cgx/webui/sse.py`): `bridge_generator()` now
  accepts `task_id` and `cancel_event` parameters. All handlers (`stream_ask`,
  `stream_plan`, `stream_agent`, `stream_index`) check `cancel_event.is_set()`
  at each yield point and terminate the stream cleanly when set.
- **Cancel / Stop buttons**: every streaming page now renders an abort button
  while busy -- **Stop** on Ask, **Cancel** on Plan, Agent, and Index -- that
  closes the SSE connection and sets the cancel event.
- **Tab persistence** (`frontend/src/store/tasks.ts`,
  `frontend/src/lib/connections.ts`):
  - `tasks.ts` -- Zustand store backed by `sessionStorage` that holds streaming
    state per page: agent (tasks / events / phase / summary), ask (messages),
    plan (thought / planMd / diff / report), index (progress / result).
  - `connections.ts` -- module-level `Map<string, SseConnection>` holding live
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

## Unreleased -- Agent loop polish

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

## 0.2.0 -- Phase 2 (current)

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
- `cgx-ui` console entry point.
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
  `expand_per_seed`, `relation_types` -- previously hard-coded magic
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
  extension exposing **CGX: Open UI** / **CGX: Reload UI** that
  host the running Gradio server in a webview panel. Server URL is
  configurable via the `cgx.ui.url` setting. Source-only scaffold;
  not packaged into a `.vsix` from the repo.

### Changed
- `cgx.embeddings.build` no longer imports `torch` /
  `sentence_transformers` / `transformers` at module load; they are
  loaded lazily inside `build_embeddings`. The UI and any BYO-embedder
  path now work on machines without the ML stack installed.
- Removed the legacy `app_gradio_llm.py`; `cgx ui` / `cgx-ui` /
  `app.py` all launch `cgx.ui.app.build_demo()` directly.
- `pyproject.toml`: corrected package layout, declared new optional
  extras (`codegen`, `keyring`, `dev`), added `cgx-ui` script.
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
