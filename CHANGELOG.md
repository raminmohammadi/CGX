# Changelog

All notable changes are documented here. Versions follow semver-ish.

## Unreleased — Gemma 4 model family + pull-error fix

### Added

- **Gemma 4 catalogue entries** (`cgx.answer.hardware_matrix.LOCAL_MODEL_CATALOG`):
  five rows covering the full Gemma 4 family as published on the
  Ollama library — `gemma4:e2b` (~7.2 GB on disk, edge), `gemma4:e4b`
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
  manifest format — see ollama/ollama#15222 — or 404 for a typo'd
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

## Unreleased — Retrieval & codegen pipeline overhaul (Phases 0–9)

A 9-phase overhaul of the retrieval, parsing, and prompt-assembly
layers. Behavior-preserving where noted; SLM-prompt and insertion
output shapes changed in two phases. **Re-indexing required** — see
the *Schema version* note under **Changed**.

### Added

- **Phase 1 — Symmetric sub-word tokenizer** (`cgx.retrieval.tokenize`):
  `split_identifier(name)` splits camelCase / PascalCase / snake_case /
  kebab-case identifiers into ordered sub-tokens; `expand_with_subwords
  (tokens, *, min_len=1)` is the dedup wrapper used on both sides of
  retrieval. Wired into `cgx.embeddings.helpers._split_tokens` (indexer
  side, feeds `lexical_helpers.ngrams_*`) and
  `cgx.retrieval.orchestrator._tokenize_lc` /
  `_extract_symbol_tokens` (query side). Identifier matching is now
  symmetric — a query for `parseConfig` hits records tokenized from
  `parse_config` and vice-versa. Covered by `tests/test_tokenize.py`
  plus a camelCase ↔ snake_case integration assertion.
- **Phase 3 — Tiered SLM context (Code Map)** (`cgx.answer.context_map`
  + `cgx.answer.model_caps.get_context_map_budget`): when the retriever
  surfaces graph-expanded neighbors (`provenance.graph_depth >= 1`),
  the prompt SOURCES list is built as two tiers — full focus-windowed
  bodies for primary hits, one-line `[class.]name(signature) — doc`
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
- **Phase 4 — Line-anchored insertion points**: every record now
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
- **Phase 6 — `CodeGraphBackend` facade** (`cgx.graph.backend`): a
  thin wrapper around the small set of `networkx` operations the
  retrieval and embeddings layers actually need
  (`has_node`, `successors`, `predecessors`, `undirected_neighbors`,
  `edge_attrs`, `node_attrs`, `bfs_distances`, plus a `wrap(G)` factory
  that returns `None` when `G` is missing). `cgx.retrieval.orchestrator`
  multi-hop expansion and `cgx.embeddings.helpers._neighbors_summary`
  now go through the facade; `build_graph`, graph visualization, and
  graph persistence still use raw `networkx` (no dependency change).
  Covered by `tests/test_graph_backend.py`.
- **Phase 7 — Parser schema + `BaseParser` seam (Python-only)** —
  `src/cgx/parser/schema.py` formalizes today's record shape via
  `CodeChunk`, `CallRelation`, and `ChunkType` `TypedDict`s, with
  `total=False` so the variable `meta` payloads keep their existing
  per-chunk-type contracts. `src/cgx/parser/base.py` introduces the
  `BaseParser` ABC: a single `parse_file(filepath, source_code,
  project_root) -> (chunks, call_relations)` method plus a lowercase
  `extensions` tuple drives extension-based dispatch. `src/cgx/parser/
  python_parser.py` provides `PythonASTParser` and registers `.py`.
  `parse_codebase` was split into a project walker (registry dispatch,
  ignore/safety knobs, cross-file post-processing — call-relation
  dedup, `calls_out_top`, `called_by_count`) and a module-level
  `_parse_python_module` worker (file/module chunk emission + the
  existing AST `CodeVisitor`). The chunk and call-relation shapes are
  byte-identical to before — `tests/test_schema_snapshots.py` still
  passes — and the dispatcher silently skips files whose extension is
  not registered (so `.py`-only behavior is preserved). Covered by
  `tests/test_parser_seam.py` (10 cases: ABC contract, registry shape,
  per-file output keys, syntax-error tolerance, worker-vs-parser
  equality, project-level aggregation, non-`.py` skip).
- **Phase 9 — Reranker profile policy** — `cgx.answer.profiles.Profile`
  gains an optional `enable_reranker: Optional[bool]` field. `None`
  (the default) means "auto" and resolves through
  `default_reranker_for_kind(kind)` — `True` for cloud kinds
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
  tokenizer to the lexical / catalog pipeline — v1 records under-match
  partial-name queries. v3 (Phase 4) adds `start_line` / `end_line` /
  `col_offset` to every record so insertion planners can splice
  without re-walking the AST — v2 records lack these fields.
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

### Internal (refactors, performance, test infrastructure — no public-API change)

- **Phase 0 — Schema-version constant + golden-output snapshots**:
  added `SCHEMA_VERSION` to records / persisted manifests and
  `tests/test_schema_snapshots.py` with three pinned snapshots
  (record-keys, `suggest_insertion_points` shape, top-K hybrid
  retrieval over a synthetic repo). Subsequent phases land against
  these snapshots so any shape drift is caught immediately.
- **Phase 2 — Parser helpers lifted to module scope**: `_build_file_
  code_stub`, `_collect_top_level_members`, `_class_signature`, and
  the surrounding stub builders were hoisted out of the
  `parse_codebase` closure to module scope in
  `cgx.parser.parse_codebase`. Pure refactor — no behavior change —
  enabling unit-testing of the helpers in isolation
  (`tests/test_parser_helpers.py`) and the Phase 7 parser-seam split.
- **Phase 5 — Exemplar-embedding LRU cache** in
  `cgx.retrieval.orchestrator`: `_build_exemplar_corpus(records,
  embedder)` is now memoised behind `_insertion_corpus_key` (keyed by
  records identity + `schema_version` + embedder fingerprint) with a
  bounded LRU. Repeat calls to `suggest_insertion_points` for the
  same index reuse a single encoded corpus matrix. A
  `_clear_insertion_corpus_cache()` helper supports test teardown.
  Covered by `tests/test_insertion_cache.py` (corpus encoded once
  across repeat calls; cache invalidates on records-id change).
- **Phase 8 — Optional Tree-sitter plugin: DROPPED**. Multi-language
  parsing deferred to a later cycle; Phase 7's parser registry already
  provides the seam.

## Unreleased — Manifest-driven scaffolding, rollback API, refactor batches B1–B9

### Added

- **`cgx.codegen.ast_insert`** — AST-anchored insertion planner that
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
  special-casing. The module is purely additive — no existing
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
  layered file list — no contents) and emits an `inject_tasks` payload;
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

- **B1 — Lazy `cgx.webui` imports** (`src/cgx/webui/__init__.py`):
  module-level `from fastapi import …` removed; symbols re-exported
  via `__getattr__` so `from cgx.webui import task_store` works
  without the `[ui]` extra installed.
- **B2 — Graph projection consolidation** (`src/cgx/graph/`):
  `projectors.py` deleted; the two duplicate projection helpers now
  live in a single `graph.aggregation` module imported by both
  `viz.visualize` and the webui graph route.
- **B3 — Embeddings de-duplication**
  (`src/cgx/embeddings/helpers.py`, `views.py`): the duplicated
  `_attribute_roots_read` body in `views.py` is replaced with a
  re-export of the helpers-module implementation; single source of
  truth.
- **B4 — Shared embedder loader** (new `cgx.embeddings.loader`, see
  Added above): removes three near-identical `_load_embedder` copies
  from `cli.main`, `retrieval.cli_adapter`, and the webui handlers.
- **B5 — Gradio drift cleanup**: removed stale references to the
  Gradio UI / port 7860 across `docs/`, `extension/`, `README.md`,
  and the React frontend (`frontend/src/layout/Header.tsx`); the
  product is React + FastAPI on port **8765** end-to-end.
- **B6 — Judge logging hygiene** (`cgx.agents.judge`): noisy
  per-criterion `print` calls replaced with structured `logger.debug`
  output gated by `CGX_LOG_LEVEL=DEBUG`.
- **B7 — Targeted logging** (`cgx.answer.profiles`,
  `cgx.answer.ratelimit`, `cgx.answer.ollama_discovery`,
  `cgx.codegen.diff_apply`, `cgx.codegen.pipeline`,
  `cgx.codegen.test_runner`, `cgx.codegen.validate`,
  `cgx.sessions`): replaced ad-hoc `print` statements with
  module-scoped `logging.getLogger(__name__)` calls so operator
  diagnostics route through the standard logging configuration.
- **B9 — `.gitignore` hygiene**: added `frontend/node_modules/`,
  `extension/out/`, `frontend/dist/`, `frontend/.vite/`, and
  `cgx_index/` patterns; existing tracked artifacts left in place
  (untracking is a separate operator decision).

## Unreleased — SLM-grade execution engine (Phases 1–5)

### Added

#### Phase 1 — Skeleton-and-Fill (`cgx.agents`)
- **`TaskKind.FILL_LOGIC`** (`cgx.agents.types`): new task kind for the
  second pass of the skeleton-and-fill pattern. The Tracker dispatches it
  to the `fill_logic` capability, which prompts the LLM to implement
  exactly one empty function body at a time — keeping local 7B models
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
  all handle the new kind — the timeline row shows
  `fn_name() in file.py · stitched · syntax ok`.

#### Phase 2 — Dynamic Dependency Management (`cgx.codegen.env_manager`)
- **New module `src/cgx/codegen/env_manager.py`**: full dependency
  management pipeline for the agent sandbox.
  - `scan_file_imports(path)` — AST-based import extraction for `.py`
    files; regex-based for `.js`/`.ts`/`.jsx`/`.tsx`.
  - `scan_imports(file_paths)` — union of imports across a list of files.
  - `find_missing_python_packages(imports, project_root)` — cross-refs
    extracted roots against `requirements.txt`, then probes live
    importability; skips the full stdlib (50+ top-level names enumerated).
  - `install_packages(packages, python)` — runs
    `pip install --quiet --no-input <pkg>` in the target Python
    interpreter (defaults to the current one); returns `{name: bool}`.
  - `update_requirements(project_root, packages)` — appends newly
    installed packages to `requirements.txt` idempotently.
  - `preflight_install(generated_files, project_root)` — one-shot
    convenience: scan → find missing → install → return results.
- **Pre-flight hook in `verify` capability** (`cgx.agents.loop`): before
  running pytest the `verify` capability scans every `.py` file in the
  changed set, installs missing packages into the current interpreter,
  and writes them back to `requirements.txt` so the dependency becomes
  permanent. `ModuleNotFoundError` failures caused by the model choosing
  a new library no longer mask real logic failures.

#### Phase 3 — Symbol Table Context (`cgx.codegen.symbol_map`)
- **New module `src/cgx/codegen/symbol_map.py`**: builds a compressed
  working-memory map of all symbols already defined in the indexed
  codebase.
  - `build_symbol_map(records_path)` — reads the JSONL records file and
    returns `{relative_path: [symbol, …]}`, deduplicated and in
    definition order.
  - `format_symbol_map(symbol_map)` — renders the map as a
    `# AVAILABLE CONTEXT (Do not redefine these):` prompt block capped
    at 60 files × 20 symbols each so the injected block stays small.
  - `fetch_symbol_source(records_path, symbol_name)` — AST-RAG on demand:
    scans records to return the exact source text for a named symbol,
    used by the retry loop to inject the real signature when the model
    calls a function with the wrong arguments.
  - `build_symbol_context_prompt(records_path)` — convenience wrapper;
    returns an empty string when the records file is absent.
- **Symbol map injected into `plan` capability** (`cgx.agents.loop`):
  before calling `generate_code_plan` the capability builds the symbol
  map from `records_path` and passes it as `symbol_context`. Local models
  see what is already defined and stop redefining it.

#### Phase 4 — Granular Error Slicing (`cgx.agents.loop`)
- **`_extract_error_snippet(project_root, responsible_files, output)`**
  (`cgx.agents.loop`): parses the first line-number reference from a
  pytest traceback, opens the failing file, and returns a ±5-line window
  around the error with an `# <-- ERROR HERE` marker — the "10-line
  buffer rule".
- **Micro-targeted retry prompts** (`_build_fix_goal`): when an error
  snippet can be extracted, the retry goal presents it as a focused
  `` ```python `` block with a one-line error summary
  (*"Your code failed in `src/auth.py` at line 42 with
  `TypeError: …`. Here is the context around the failure:"*) rather than
  dumping the full 1 200-character pytest tail. The raw output is still
  appended as a fallback when no line number can be found.

#### Phase 5 — Universal LLM Provider (`cgx.answer.providers`, `cgx.answer.profiles`)
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
    profile edit form — displays `OK · <Nms>` in green or the error
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
  (Phase 4 — see above).
- `verify` capability auto-installs missing Python packages before
  running pytest (Phase 2 — see above).
- `plan` capability injects a symbol-context block from
  `build_symbol_context_prompt` when `records_path` is available
  (Phase 3 — see above).

---

## Unreleased — Agent loop reliability: targeted retries, partial apply, cross-file coherence

### Fixed

- **Fix #3 — Apply failures now trigger recursive retry** (`cgx.agents.loop`):
  `_stream_with_retry` previously checked only verify and core failures when
  deciding whether to recurse.  Apply failures (smoke-check rejections) were
  silently ignored, causing the loop to exit after one attempt even when the
  re-plan's apply step also failed.  `apply_failures` is now included in the
  recursion condition.

- **Fix #6 — Partial apply: passing files are always written** (`cgx.codegen.disk_apply`):
  `apply_diffs_to_disk` previously returned an early-exit error and wrote
  *nothing* if any file in the batch failed the smoke check.  It now writes
  every file that passes and records the failing ones in `failed_files`.
  `smoke_ok` is `True` only when all files passed.  Retries can therefore
  target only the broken file(s) — already-correct files stay on disk.

- **Fix #5 — Cross-file coherence check** (`cgx.codegen.validate`):
  New `check_cross_file_coherence(patches, project_root)` function runs
  alongside the per-file syntax smoke test inside `apply_diffs_to_disk`.  It
  walks Python files in the patch batch, parses their `import` statements, and
  flags any `from X.Y import Z` where `X/Y.jsx`, `.tsx`, `.js`, or `.ts` is
  present in the same batch or on disk.  This catches the common
  mis-generation where a Python test does `from src.App import calculateResult`
  but `src/App.jsx` is a React component — not a Python module.

- **Fix #4 — Failure diagnosis before re-planning** (`cgx.agents.loop`):
  New `_diagnose_failure(failures)` classifies test output as
  `import_error`, `syntax_error`, `logic_error`, or `unknown`, extracts
  responsible file paths from tracebacks, detects language-mismatch cases
  (Python importing a JS/JSX module), and returns a structured dict that
  informs `_build_fix_goal`.

- **Fix #2 — Targeted fix goals** (`cgx.agents.loop`):
  `_build_fix_goal` now uses the diagnosis to emit a *targeted* re-plan
  prompt: it names the specific broken files (from the traceback), tells
  the LLM not to change files that are already correct (read from
  `plan.owned_files`), and — when a language mismatch is detected —
  explicitly offers two remediation paths: create a Python backend module
  that the test can import, or replace the Python test with a JS test.

- **Fix #1 — File manifest on `Plan`** (`cgx.agents.types`, `cgx.agents.tracker`):
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

## Unreleased — Skills package: modular tech-specific knowledge bundles

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
  detection — goals naming a supported technology route to SCAFFOLD
  even when the noun regex doesn't fire — while the existing `_TECH_RE`
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

## Unreleased — Scaffold routing fix: tech-paired scaffold goals + judge artifact

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

## Unreleased — Judge SCAFFOLD short-circuits on structural pass

### Fixed
- **Local 3-7B judge models hallucinate criteria fails on scaffolds**
  (`cgx.agents.judge.Judge.judge`, `_structural_check`): even with
  source-prioritized previews and goal context in the prompt, small
  local models (`qwen2.5-coder:3b`, etc.) routinely return high-
  confidence `fail` verdicts against scaffolds that demonstrably
  satisfy their criteria — e.g. rejecting a calculator with
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

## Unreleased — Judge scaffold preview: source-file priority + goal context

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
  metadata before the source code was reached — leaving the judge unable
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
