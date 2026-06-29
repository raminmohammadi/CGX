# The CGX Agent

This document describes the **agent layer** of CGX -- the component that
turns a natural-language goal into grounded answers, recommendations,
and applied code changes against a real codebase. It is intended for
community contributors who want to understand how the agent works
today and where it could be pushed further.

CGX ships **two agent shapes**:

* **Session-based agent** (default) -- a persistent DAG of typed tasks
  with structured human-in-the-loop checkpoints. Code under
  [`src/cgx/session/`](../src/cgx/session/); HTTP at
  `/api/agent-session/*`; UI at `/agent`. Designed for multi-turn
  exploration that may or may not end in a code change.
* **Batch agent** (legacy) -- a one-shot Planner → Tracker → Judge
  loop that commits a full plan up front and streams it over SSE. Code
  under [`src/cgx/agents/`](../src/cgx/agents/); HTTP at `/api/agent`;
  UI preserved at `/agent-legacy`. Sections 2–8 below describe this
  shape; it is still the entry point exposed by
  [`cgx.agents.run_agent`](../src/cgx/agents/loop.py) and the CLI.

The two shapes coexist: the session backbone reuses the same retrieval,
codegen, and provider stacks, so a `PLAN_CHANGE` task in a session goes
through the same `generate_code_plan` path that the batch agent's
`plan` task does. They differ in **state model** (persistent vs.
batch), **interaction model** (typed decisions vs. fire-and-forget),
and **execution model** (deterministic router + per-kind executors
vs. LLM-emitted plan + capability dispatch).

---

## 1A. Session-Based Agent (default)

The session-based agent treats every interaction as part of an
ongoing **Session**: a persistent record of the user's objective, the
task tree the router has spawned, the facts surfaced into the
knowledge base, the artifacts produced by finished tasks, and the
decision log of typed user choices. State survives process restarts;
the user can return to a session days later and pick up where they
left off.

### 1A.1 Why a different shape?

The batch agent treats every goal as a one-shot job: plan up front,
execute, judge, return. That works for well-scoped goals ("add
docstrings", "create a FastAPI todo app") but breaks down for
exploratory work ("how should we refactor the parser layer?") where
the user wants to **see options before committing to a direction**,
and where the conversation may branch repeatedly before any code is
written. The session-based agent makes those branches first-class:
every divergence point is an explicit `ASK_USER` task with a typed
decision contract, and every choice is replayable.

A session runs in one of two **modes** (`Session.mode`):

* **`explore`** -- modify an existing, indexed codebase. The router
  starts with an `EXPLORE` task that runs retrieval-grounded
  `clarify_paths` against the FAISS index. Used for refactors,
  bug-fixes, feature additions.
* **`greenfield`** -- scaffold a new project from scratch. The router
  starts with `CLARIFY_REQUIREMENTS` (no index, no retrieval) and
  walks the user through a structured clarify → decompose → approve
  → scaffold loop. Used when the project root is empty, doesn't
  exist yet, or has no FAISS index.

Mode is resolved in `cgx.session.mode.detect_mode`: empty / missing
`project_root` → greenfield; populated root but no usable
`meta.json` + records file → greenfield; everything else → explore.
The webui route honors an explicit `mode` field on the create
request and falls back to `detect_mode` only when none is given.

### 1A.2 Data model

All session state lives under `src/cgx/session/models.py` as plain
:mod:`dataclasses` (no Pydantic at the core layer -- Pydantic stays at
the webui wire boundary in `cgx.webui.models`).

| Type           | Purpose                                                                 |
|----------------|-------------------------------------------------------------------------|
| `Session`      | Root aggregate: original objective, project root, root task id, status. |
| `TaskNode`     | One node in the per-session DAG; carries `kind`, `inputs`, `outputs`, `parent_task_id`, `produced_artifact_id`, `consumed_decision_ids`, lifecycle timestamps. |
| `Fact`         | An append-only piece of session knowledge (`FILE`, `SYMBOL`, `PARAMETER`, `ANCHOR`, or `LLM_CALL`). `LLM_CALL` facts (**Phase 5.1**) carry `{provider, model, sampling_params, prompt, response, latency_ms, tokens_in, tokens_out, source_task_id, role}` recorded by every LLM call site in `cgx.answer.engine`, `clarify_requirements.py`, `decompose.py`, and `scaffold.py`. Updates set `stale=True` rather than mutating `content`. |
| `Artifact`     | A typed output produced by a finished task (e.g. `DIRECTIONS_LIST`, `FINDINGS_BUNDLE`, `CODE_CHANGE_PLAN`). Survives across turns. |
| `Decision`     | Structured record of a user choice resolving an `ASK_USER` task; downstream tasks reference decisions by id rather than re-parsing free text. |
| `KnowledgeBase`| Per-session view over the facts table.                                  |
| `DecisionLog`  | Per-session view over the decisions table.                              |

Task kinds (`TaskKind`):

| Kind                    | Role |
|-------------------------|------|
| `EXPLORE`               | Survey the codebase for directions that bear on the user's objective. Produces a `DIRECTIONS_LIST` artifact and anchor facts. *(explore mode)* |
| `INVESTIGATE`           | Deeper retrieval anchored on a single chunk; produces a `FINDINGS_BUNDLE`. *(explore mode)* |
| `RECOMMEND`             | Synthesize concrete next-step recommendations from the investigation; produces a `RECOMMENDATION_LIST`. *(explore mode)* |
| `CLARIFY_REQUIREMENTS`  | Structured LLM call (or deterministic fallback) producing 3–6 clarification questions about the user's greenfield goal; emits a `REQUIREMENTS_SHEET`. *(greenfield mode)* |
| `DECOMPOSE`             | Folds the user's clarify answers into the goal, runs `plan_scaffold_manifest`, and emits a `WORK_PLAN` (`plan_md` + layered file manifest). *(greenfield mode)* |
| `SCAFFOLD`              | Walks the `WORK_PLAN` layers, calls `generate_single_scaffold_file` per entry while accumulating sibling-file context, emits `SCAFFOLD_PATCHES`. *(greenfield mode)* |
| `PLAN_CHANGE`           | Turn an approved recommendation into a unified-diff change plan; produces a `CODE_CHANGE_PLAN`. *(explore mode)* |
| `APPLY`                 | Write an approved plan's (or scaffold's) diffs to disk; produces `APPLIED_CHANGES` (with `backup_dir`). |
| `BOOTSTRAP_ENV`         | Provision a project-local `.venv`, install declared requirements, and preflight-install undeclared imports found in the applied files; produces a `BUILD_REPORT` carrying `project_type`, `venv_path`, `python_exe`, `installed_from`, `installed_packages` (parsed `[{name, version}, …]` from `pip freeze --all`, **Phase 1.1**), `freeze_text` (the raw freeze output for diagnostics), `failed_installs`, an `outcome` token (`succeeded` / `failed` / `no_venv` / `skipped` / `partial`), and a `style_issues` list populated by an AST lint over the applied test files (catches `self.assert*` calls in non-`TestCase` classes ahead of `VERIFY`; informational, does not change the outcome). The `installed_packages` snapshot is what the Phase 3.2 PyPI-aware repair proposer reads to compute corrective pins. *(greenfield mode)* |
| `API_CHECK`             | After `BOOTSTRAP_ENV`, statically walks every applied file and resolves every `from <third_party> import <name>` and aliased `pkg.attr` access under the bootstrapped venv via `importlib` + `inspect.getmembers`. Unresolved references surface as a structured `API_CHECK_REPORT` (`outcome ∈ {passed, failed, skipped}`, `unresolved: [{file, line, module, name}]`, `failure_signature`). A clean run chains to `SMOKE`; `failed` routes to `REPAIR` carrying the `API_CHECK_REPORT` as the source artifact. *(greenfield mode, **Phase 2.2**)* |
| `SMOKE`                 | Cheap fail-fast gate between `API_CHECK` and `VERIFY`: spawns `<venv>/bin/python -c "import <pkg>"` for every top-level module the applied files declare, with a 30s wall-clock budget. Produces a `SMOKE_REPORT` (`outcome`, `imports: [{module, ok, stderr_tail}]`, `failure_signature`). On `passed` / `skipped` chains to `VERIFY`; on `failed` routes to `REPAIR` (typical trigger: `ImportError: cannot import name 'url_quote' from 'werkzeug.urls'` -- the Flask/Werkzeug peer pin mismatch that motivated the whole plan). *(greenfield mode, **Phase 2.1**)* |
| `VERIFY`                | Run impacted tests against the working tree; produces a `VERIFY_REPORT` whose `outcome` token classifies pytest's exit code (`passed` / `assertions_failed` / `collection_error` / `no_tests_collected` / `timeout` / `pytest_missing` / `skipped`). Uses `BUILD_REPORT.python_exe` when available so pytest runs inside the project venv, not CGX's interpreter. Pytest is now invoked with `--junitxml=<tmp> -rN --tb=long` and the XML is parsed via stdlib `xml.etree` into a structured `failures: [{nodeid, type, message, traceback}]` list (**Phase 3.1**) so the classifier can consume types rather than re-regexing stdout. Also surfaces `reproduce_cmd` -- a single `shlex.quote`-escaped shell line that re-runs the exact failing pytest invocation under the project venv (**Phase 1.2**) -- and a `failure_signature` (sha1 of outcome + returncode + first error line) used by the autonomous repair loop. |
| `REPAIR`                | Classify a failed `VERIFY` / `SMOKE` / `API_CHECK` and emit a typed `REPAIR_PLAN` (diffs + rationale + located classes + `strategy` + `extra_constraints`). The classifier is a small registry in `cgx.session.repair.classify`; v1 ships: `unittest_pytest_mix` (rewrite class header to inherit `unittest.TestCase`), `missing_module_pythonpath` (create/extend project-root `conftest.py` so pytest can resolve scaffolded packages), `missing_fixture` (hoist an `@pytest.fixture` definition into `tests/conftest.py` or project-root `conftest.py`), `hallucinated_api` (rename / drop the broken symbol surfaced by `API_CHECK`), and `third_party_import_break` (**Phase 3.2** -- recognises `ImportError: cannot import name '<x>' from '<pkg>'` and `ModuleNotFoundError` for third-party modules, then `propose_third_party_pin` queries the PyPI JSON API via `cgx.session.repair.pypi_client` -- with an on-disk cache under `~/.cgx/pypi-cache/` -- to compute a corrective version pin and emit a `requirements.txt` diff). Unknown failures yield an empty plan that escalates to `ASK_USER(freeform)`. Repair has two branches (**Phase 6.1**): `strategy=patch` writes the proposed diffs through the shared `APPLY` executor (≤5 diffs in a patchable class); `strategy=regenerate` abandons the failed `SCAFFOLD` subtree and re-queues a fresh `SCAFFOLD` with `regenerate_constraints` folded into the goal so the per-file generator avoids the failure mode this time. Greenfield-only; capped at 2 attempts per session and gated by `failure_signature` to break flapping loops; the regenerate branch is additionally capped at one re-scaffold per ancestor chain. |
| `ASK_USER`              | Structured pause; carries an `expected_kind` indicating which decision contract the UI must satisfy. |
| `SEARCH` / `SUMMARIZE`  | Utility kinds the router may interleave. |

### 1A.3 The Router

`cgx.session.router.Router` is the central state machine. It is **pure
Python with no LLM calls and no I/O**: every method takes the current
session state plus an event and returns a `RouterPlan` of typed
actions (`CreateTask`, `UpdateTaskStatus`, `RecordDecision`,
`AttachDecisionToTask`, `RecordLesson`) that the caller applies to
the store.

Three entry points cover every transition:

* `on_user_message(session, message, tasks)` -- user posts a fresh
  objective or a follow-up. If no tasks exist, spawn the root task:
  `EXPLORE` in explore mode, `CLARIFY_REQUIREMENTS` in greenfield
  mode. If a pending `ASK_USER` is open, return an empty plan so
  the caller can route the message to `on_decision_recorded`
  instead. Otherwise spawn a sibling EXPLORE (explore mode) under
  the current root (treats the message as a course-correction
  objective).
* `on_task_completed(session, completed, tasks)` -- an executor
  finished a task; dispatch via the `TASK_SUCCESSOR` table:
  - Explore loop: `EXPLORE → ASK_USER(choose_path)`,
    `INVESTIGATE → RECOMMEND`,
    `RECOMMEND → ASK_USER(choose_recommendation)`,
    `PLAN_CHANGE → ASK_USER(approve)`,
    `APPLY → VERIFY`. `VERIFY` is terminal.
  - Greenfield loop: `CLARIFY_REQUIREMENTS → ASK_USER(clarify_answers)`,
    `DECOMPOSE → ASK_USER(approve_plan)`,
    `SCAFFOLD → APPLY` (with `mode=greenfield` threaded into the
    APPLY inputs),
    `APPLY → BOOTSTRAP_ENV` (greenfield-only edge, threading
    `apply_artifact_id` and `scaffold_artifact_id` through inputs),
    `BOOTSTRAP_ENV → API_CHECK` (**Phase 2.2**, threading
    `build_artifact_id`),
    `API_CHECK (passed | skipped) → SMOKE` (**Phase 2.1**),
    `API_CHECK (failed) → REPAIR` (subject to the shared retry
    budget + flap guard, carrying `API_CHECK_REPORT` as the source
    artifact),
    `SMOKE (passed | skipped) → VERIFY`,
    `SMOKE (failed) → REPAIR` (same guards, `SMOKE_REPORT` as
    source).  Explore mode keeps the direct `APPLY → VERIFY` edge
    and never spawns `API_CHECK` / `SMOKE`.
  - Repair loop (greenfield only): `VERIFY (assertions_failed |
    collection_error) → REPAIR` (when `repair_attempt < 2` and the
    new `failure_signature` is not in `prior_failure_signatures`);
    `REPAIR (strategy=patch, can_apply) → APPLY` (carrying
    `build_artifact_id` forward so BOOTSTRAP_ENV is skipped);
    `REPAIR (strategy=regenerate) → SCAFFOLD` (**Phase 6.1** -- the
    router walks up to the nearest `SCAFFOLD` ancestor, marks every
    live descendant `ABANDONED`, and re-queues a fresh `SCAFFOLD`
    via `propose_regenerate` with the failure-derived
    `regenerate_constraints` appended to its `inputs`; capped at
    one re-scaffold per ancestor chain by `_REGENERATE_BUDGET=1`);
    `REPAIR (empty plan) → ASK_USER(freeform)`. The cycle re-enters
    `VERIFY` and either terminates (`passed`) or escalates once the
    budget / signature guard fires.
  - Lesson-recording (**Phase 7.1**): whenever a `VERIFY` finishes
    with `outcome=passed` AND a `REPAIR` is found on its ancestor
    chain, the router emits a `RecordLesson(verify_task_id,
    repair_task_id, scaffold_task_id?)` action. The runner resolves
    it into a `record_lesson(...)` call against
    `~/.cgx/lessons.jsonl` (override via `$CGX_LESSONS_PATH`) so
    future `SCAFFOLD` runs in any session can re-use the rule.
* `on_decision_recorded(session, decision, tasks)` -- user resolved
  an `ASK_USER` via a typed `Decision`. The router records the
  decision, attaches it to the ASK_USER, marks the ASK_USER `DONE`,
  and spawns the successor implied by the decision shape (see §1A.5).

### 1A.4 The Runner and executors

`cgx.session.runner.SessionRunner` sits between the router and the
SQLite-backed `SessionStore`. All write paths funnel through it so a
single sequencer enforces:

* Router plans applied in order: creates, decisions, attaches,
  status updates, lesson recording.
* Per-session locking so concurrent requests for the same session
  can't interleave half-applied plans.
* Centralised executor dispatch + failure handling.
* Project-local JSONL agent log: one line per task transition + per
  executor exception, written to `<project_root>/.cgx/agent.log`
  via a rotating handler wired in `cgx.logging_setup`
  (**Phase 1.3**). The handler is best-effort -- log failures are
  swallowed so they never break the loop.
* `RecordLesson` resolution: when the router emits one, the runner
  fetches the `REPAIR_PLAN` artifact and the SCAFFOLD's inputs,
  derives the `applied_fix` (`{strategy, diff_count, files,
  extra_constraints}`) and `scope` (`{stack, objective_keywords}`)
  payloads, and calls `cgx.session.lessons.record_lesson` against
  `~/.cgx/lessons.jsonl`. Any exception is logged + swallowed.

An **executor** is a pure function `(TaskNode, ExecutorDeps) ->
ExecutorResult` registered against a `TaskKind` via
`@register_executor` in `cgx.session.tasks.base`. Each kind has at
most one executor; importing the `cgx.session.tasks` package
side-effect-registers them all (`explore.py`, `ask.py`,
`investigate.py`, `recommend.py`, `plan_change.py`,
`clarify_requirements.py`, `decompose.py`, `scaffold.py`,
`apply.py`, `bootstrap_env.py`, `api_check.py`, `smoke.py`,
`verify.py`, `repair.py`).

Executors do **not** write to the store directly -- the runner
persists their outputs, facts, and artifacts after the call. This
keeps executors easy to unit-test without a database and gives the
runner a single place to enforce ordering. LLM-issuing executors
(`clarify_requirements`, `decompose`, `scaffold`, and the
`cgx.answer.engine` helpers they call) additionally record an
`LLM_CALL` fact per provider invocation via
`cgx.session.llm_trace.trace_llm_call` so the UI's task card can
surface model name, sampling params, latency, and a redacted
prompt/response preview (**Phase 5.1**).

### 1A.5 Decision contract

Every `ASK_USER` task carries `inputs["expected_kind"]` indicating
which `DecisionKind` the UI must satisfy. The frontend posts
`{task_id, chosen, rationale?}` to
`/api/agent-session/{sid}/decision`; the route layer calls
`build_decision` in `cgx.session.tasks.ask` which validates the
slot shape and raises `400` on mismatch:

| `expected_kind`           | `chosen` shape                                                                            | Successor spawned by router |
|---------------------------|-------------------------------------------------------------------------------------------|-----------------------------|
| `choose_path`             | `{anchor_chunk_id: str, title?: str}`                                                     | `INVESTIGATE` on the chosen anchor |
| `choose_recommendation`   | `{id, title, rationale, kind, anchor_chunk_id?}` where `kind ∈ {investigate_more, plan_change, ask_followup, done}` | `INVESTIGATE` / `PLAN_CHANGE` / freeform `ASK_USER` / no successor |
| `approve`                 | `{approved: bool}`                                                                        | `APPLY` on approval; no successor on decline (user can pivot via a fresh objective) |
| `clarify_answers`         | `{answers: {question_id: str}}` (non-empty dict)                                          | `DECOMPOSE` carrying the answers + the prior goal |
| `approve_plan`            | `{approved: bool}` (against a `WORK_PLAN` artifact)                                       | `SCAFFOLD` on approval; no successor on decline |
| `freeform`                | `{text: str}`                                                                             | None (handled as a new user message by the caller) |

A `done` recommendation closes the session focus and lets a follow-up
message spawn a fresh sibling EXPLORE -- this is how the user signals
"I'm satisfied with what we found; let's move on".

### 1A.6 Greenfield walk-through

A greenfield session for *"build a Python app with a Flask API and a
frontend where users enter their information and the server saves it
as JSON on disk"* walks through eight executor calls and two user
checkpoints. Every line below corresponds to one row in the session
store; the React UI renders each as a node in the task tree.

1. `CLARIFY_REQUIREMENTS` runs on the raw objective. The executor
   calls the configured LLM with a JSON-forced prompt asking for
   3–6 clarification questions about stack, storage strategy,
   schema, and target environment. If the provider is unavailable
   or returns fewer than three well-formed questions, the executor
   falls back to a deterministic question bank so the loop never
   stalls on a network blip. Output: a `REQUIREMENTS_SHEET`
   artifact (`questions: [{id, prompt, hint?, suggested?}, …]`,
   `source: "llm" | "fallback"`).

2. `ASK_USER(clarify_answers)` is spawned by the router. The UI
   renders one textarea per question and posts
   `{answers: {q1: …, q2: …, …}}`. `build_decision` rejects empty
   payloads or non-dict `answers` with HTTP 400.

3. `DECOMPOSE` runs once the answers are in. The executor folds
   each `Q: A` pair into the goal text, calls
   `cgx.answer.engine.plan_scaffold_manifest`, and emits a
   `WORK_PLAN` artifact carrying `plan_md` (human-readable
   markdown) plus `layers: [{name, files: [{path, description}]}]`.
   An empty manifest is treated as a failure so the router can
   spawn a retry rather than push the session into a broken state.

4. `ASK_USER(approve_plan)` is spawned by the router. The UI shows
   the layered file list and a `[Approve & Scaffold | Reject]`
   pair. Rejection halts the loop and keeps `SCAFFOLD` / `APPLY`
   / `VERIFY` off the tree.

5. `SCAFFOLD` walks the approved manifest. For each file it calls
   `cgx.answer.engine.generate_single_scaffold_file` with the
   previously generated files passed as
   `existing_files_with_content` so cross-file imports resolve
   correctly. Each generation produces a unified diff; failures
   are captured into `failed: [{file, error}]` rather than
   aborting the loop, so a partial scaffold is recoverable. Before
   the executor writes `requirements.txt` it runs the **Phase 4.1**
   PyPI-aware pin validator (`cgx.session.scaffold_validate`): for
   every declared pin it queries the package metadata through the
   shared PyPI client + on-disk cache, inspects `requires_dist`,
   and auto-tightens upper bounds on a curated peer table
   (`Flask ↔ Werkzeug`, `Pydantic v1 ↔ v2`, `NumPy ↔ SciPy`,
   `SQLAlchemy ↔ alembic`) so a hard `Flask==2.1.2` no longer pulls
   in a Werkzeug 3.x that breaks `url_quote` at import time. The
   executor also queries `cgx.session.lessons.relevant_lessons`
   (filtered by the WORK_PLAN's stack + the goal's keyword
   tokeniser) and folds the top-3 hits into the per-file generator
   goal under a `Lessons from prior sessions to apply:` header so
   cross-session knowledge influences this run before the first
   token is generated (**Phase 7.1**). Output: a `SCAFFOLD_PATCHES`
   artifact (`diffs`, `generated`, `failed`).

6. `APPLY` consumes the `SCAFFOLD_PATCHES` artifact -- the
   executor accepts either `CODE_CHANGE_PLAN` (explore mode) or
   `SCAFFOLD_PATCHES` (greenfield) as upstream artifact -- and
   writes the diffs via `cgx.codegen.disk_apply.apply_diffs_to_disk`
   under a session-tagged backup directory. Successors carry
   `mode=greenfield` in their inputs.

7. `BOOTSTRAP_ENV` provisions a project-local runtime. For Python
   projects (detected via `requirements.txt` / `pyproject.toml` /
   `setup.{py,cfg}`) it calls
   `cgx.codegen.test_runner.ensure_project_venv` to create or
   refresh `.venv` and pip-install declared requirements, then
   `cgx.codegen.env_manager.preflight_install` to detect undeclared
   top-level imports in the applied files and install them into the
   same venv; successful adds are appended back to
   `requirements.txt` via `update_requirements` so the manifest
   stays in sync. At the end of the run the executor calls
   `<venv>/bin/pip freeze --all` and stores the parsed
   `installed_packages: [{name, version}, …]` plus the raw
   `freeze_text` on the report (**Phase 1.1**) so the repair
   classifier has the resolved peer-dep graph available without
   re-shelling pip. The executor emits a `BUILD_REPORT` artifact
   with the venv path, the manifests installed from, the list of
   preflight-installed and failed packages, and a single `outcome`
   token the UI surfaces as a coloured badge. Non-Python projects
   short-circuit with `outcome=skipped` so the loop still reaches
   `VERIFY`.

8. `API_CHECK` (**Phase 2.2**) statically walks every applied file
   under the bootstrapped venv and resolves each
   `from <third_party> import <name>` and aliased `pkg.attr`
   reference via `importlib` + `inspect.getmembers`. Unresolved
   references surface in an `API_CHECK_REPORT` (`outcome`,
   `unresolved: [{file, line, module, name}]`,
   `failure_signature`). The point is to catch hallucinated symbols
   -- the LLM imagining a `flask.legacy.url_quote` that does not
   exist -- before pytest even starts collection. A clean report
   hands off to `SMOKE`; `failed` routes to `REPAIR` carrying the
   `API_CHECK_REPORT` as the source artifact.

9. `SMOKE` (**Phase 2.1**) is the cheapest possible runtime gate:
   `<venv>/bin/python -c "import <pkg>"` for every top-level
   module the applied files declare, capped at a 30s wall-clock
   budget for the whole batch. Result is a `SMOKE_REPORT`
   (`outcome`, `imports: [{module, ok, stderr_tail}]`,
   `failure_signature`). On `passed` / `skipped` the router chains
   to `VERIFY`; on `failed` (e.g. `ImportError: cannot import name
   'url_quote' from 'werkzeug.urls'`) it routes to `REPAIR` with
   the `SMOKE_REPORT` as the source artifact, gated by the same
   retry budget + flap detector as the VERIFY-driven loop.

10. `VERIFY` is the same executor as the explore loop, but it now
    classifies pytest's exit code into an explicit `outcome` token
    (`passed` / `assertions_failed` / `collection_error` /
    `no_tests_collected` / `timeout` / `pytest_missing` /
    `skipped`) and reads `python_exe` from the upstream
    `BUILD_REPORT` when present, so a `ModuleNotFoundError: flask`
    shows up as `collection_error` rather than masquerading as a
    real test failure. Pytest is invoked with `--junitxml=<tmp>
    -rN --tb=long` (**Phase 3.1**) and the XML is parsed into a
    structured `failures: [{nodeid, type, message, traceback}]`
    list so the classifier consumes typed records rather than
    re-regexing stdout. The report also carries
    `reproduce_cmd` -- a single `shlex.quote`-escaped shell line
    that re-runs the exact failing pytest invocation under the
    project venv (**Phase 1.2**) -- which the UI renders above the
    stdout pane. When `mode=greenfield` and no tests have been
    discovered yet the report carries `ran=False` with a
    `skipped_reason` so the UI marks the loop terminal-clean.

Four router-level guardrails keep the loop honest:

* The `ASK(approve_plan)` checkpoint is mandatory. Even if the
  manifest looks great, the user has to confirm before any file is
  written -- this is the inflection point where a greenfield
  session can pivot at zero disk cost.
* If `SCAFFOLD` records any entry in `failed`, the result still
  surfaces a `SCAFFOLD_PATCHES` artifact -- the `failed` list is
  preserved so the UI can show which file slipped and the user can
  choose whether to apply the partial scaffold or restart.
* Phases 2.1 / 2.2 keep `BOOTSTRAP_ENV` isolated from `VERIFY`:
  environment failures surface as `outcome=failed` / `no_venv` on
  the `BUILD_REPORT` (with a `pip_log_tail` for diagnosis),
  hallucinated APIs surface on the `API_CHECK_REPORT`, and
  third-party import breakage surfaces on the `SMOKE_REPORT` --
  each as a structured signal in <1 s rather than an opaque pytest
  collection error 30 s later.
* The autonomous `REPAIR` cycle is greenfield-only and bounded by
  three orthogonal guards. The retry budget (`repair_attempt`
  capped at 2) prevents the loop from monopolising the session.
  The progress detector (a sha1 over the verify outcome,
  returncode, and first error line, tracked in
  `prior_failure_signatures` on every downstream task) refuses a
  second attempt when the signature matches a prior failure, so a
  fix that "succeeds" but leaves the same crash in place escalates
  to `ASK_USER` instead of looping forever. The
  `_REGENERATE_BUDGET=1` cap on `propose_regenerate` (**Phase
  6.1**) prevents the patch-vs-regenerate branch from re-
  scaffolding the same subtree more than once.

### 1A.7 Persistence

`cgx.session.store.SessionStore` is a thin SQLite wrapper. One
database file holds every session for a given project root, at
`<project_root>/.cgx/sessions.db` (or `~/.cgx/sessions.db` when no
project root is provided). Tables: `sessions`, `tasks`, `facts`,
`decisions`, `artifacts`. Each row stores the dataclass as a JSON
blob plus a few indexed columns (session_id, status, timestamps) so
common queries don't have to parse JSON. Connections use WAL mode for
concurrent reader tolerance.

Three sibling files live alongside `sessions.db`:

* `<project_root>/.cgx/agent.log` -- rotating JSONL agent log
  (one line per task transition + per executor exception),
  written by the project-local handler wired in
  `cgx.logging_setup` (**Phase 1.3**). Survives the lifetime of
  the project, not the session, so it's also the place to look
  when a session row was deleted but the user wants to know what
  happened.
* `~/.cgx/lessons.jsonl` -- cross-session lessons store
  (**Phase 7.1**). Append-only JSONL; each row carries
  `{lesson_id, created_at, session_id, trigger_signature,
  classification, applied_fix: {strategy, diff_count, files,
  extra_constraints}, scope: {stack, objective_keywords}}`. The
  path is `~/.cgx/lessons.jsonl` by default; override via
  `$CGX_LESSONS_PATH` (used by tests for isolation). Disk
  failures, malformed JSON, and missing files are all swallowed
  silently -- learning is best-effort and must not break the
  agent loop.
* `~/.cgx/pypi-cache/` -- on-disk JSON cache for
  `cgx.session.repair.pypi_client` (**Phases 3.2 + 4.1**), keyed
  by `{pkg}/{version}.json`. Reused by both the SCAFFOLD pin
  validator and the third-party-import repair proposer.

### 1A.8 HTTP surface

`cgx.webui.routes.agent_session` mounts the session API at
`/api/agent-session`:

| Method | Path                              | Purpose |
|--------|-----------------------------------|---------|
| `POST` | `/api/agent-session`              | Create a session, seed the root task (`EXPLORE` or `CLARIFY_REQUIREMENTS` depending on mode), optionally drain READY tasks. Accepts an optional `mode: "explore" | "greenfield"`; falls back to `detect_mode` when absent. |
| `GET`  | `/api/agent-session?project_root` | List sessions for a project. |
| `GET`  | `/api/agent-session/{sid}`        | Full state snapshot (`session + tasks + artifacts + facts + decisions`). |
| `POST` | `/api/agent-session/{sid}/message` | Post a follow-up message (spawns a sibling EXPLORE when no ASK is open). |
| `POST` | `/api/agent-session/{sid}/decision` | Resolve a pending ASK_USER with a typed `chosen` payload. |
| `DELETE` | `/api/agent-session/{sid}` | Discard a session and its full aggregate (tasks / facts / decisions / artifacts) via SQLite `ON DELETE CASCADE`. Returns `{deleted: sid}` or 404. |

Every mutating endpoint returns the full snapshot so the React UI can
render the updated tree in one round-trip. There is no SSE on this
surface today -- the UI polls while any task is in-flight (running
executors, not pending `ASK_USER` tasks).

A per-`project_root` runner cache (`_RUNNERS` in
`agent_session.py`) reuses one `SessionStore` (and its SQLite WAL
connection) across requests.

### 1A.9 React UI (`/agent`)

`frontend/src/pages/AgentPage.tsx` is the session-shaped page; modular
components live under `frontend/src/components/agent/`:

* `SessionLauncher.tsx` -- create a new session (objective + project
  root + mode picker: *auto / explore / greenfield*). *Auto* defers
  to `detect_mode`; the two explicit choices override it.
* `TaskTree.tsx` -- hierarchical DAG renderer keyed on
  `parent_task_id`; depth-based indentation, status icons, selection
  highlighting. Orphaned tasks re-surface at the top level. Carries
  badges for the greenfield kinds (`clarify`, `decompose`,
  `scaffold`).
* `ActiveTask.tsx` -- detail panel for the currently selected task;
  shows description, status, outputs, and (for `ASK_USER`) the
  appropriate decision form.
* `AskUserForm.tsx` -- dispatch on `expected_kind` to one of
  `ChoosePathForm`, `ChooseRecommendationForm`, `ApproveForm`,
  `ClarifyAnswersForm`, `ApprovePlanForm`, `FreeformForm`. Each
  form posts the typed `chosen` payload that `build_decision`
  expects. `ClarifyAnswersForm` reads the linked
  `REQUIREMENTS_SHEET` and renders one labeled textarea per
  question; `ApprovePlanForm` renders the layered file manifest
  with `[Approve & Scaffold | Reject]`.
* `ArtifactPreview.tsx` -- per-kind renderers for the artifacts the
  panel can carry; the greenfield kinds (`requirements_sheet`,
  `work_plan`, `scaffold_patches`) have dedicated bodies showing
  questions, layered manifests, and generated/failed file lists
  with combined unified diffs.
* `LiveView.tsx` -- the right-hand active-task pane, with a mode
  badge in the session header (`explore` / `greenfield`) so the
  loop shape is visible at a glance.
* `SidePanel.tsx` -- tabbed view of the Knowledge Base (facts) and
  Artifacts; `ArtifactPreview.tsx` renders each artifact by kind.

Selection and active-session id are persisted to `localStorage` via
`frontend/src/store/agentSession.ts` (Zustand + `persist`) so a tab
switch / reload comes back to the same view. `AgentPage.loadState`
catches the typed `ApiError` exported from `frontend/src/lib/api.ts`
and, on `status === 404` for the active id, clears the persisted
`activeId` / `selectedTaskId` and refreshes the sidebar -- so a
session deleted out-of-band or a `project_root` swap to a different
SQLite file lands the user on the launcher instead of re-firing the
same 404 on every mount.

### 1A.10 Where to look for what

| To understand…                    | Read… |
|-----------------------------------|-------|
| The state model                   | `src/cgx/session/models.py` |
| Mode auto-detection               | `src/cgx/session/mode.py` :: `detect_mode` |
| Transitions / successor table     | `src/cgx/session/router.py` |
| The runner sequencer              | `src/cgx/session/runner.py` |
| Persistence schema                | `src/cgx/session/store.py` |
| Project-local agent log (Phase 1.3) | `src/cgx/session/agent_log.py`, `src/cgx/logging_setup.py` |
| Explore-mode executors            | `src/cgx/session/tasks/{explore,investigate,recommend,plan_change}.py` |
| Greenfield executors              | `src/cgx/session/tasks/{clarify_requirements,decompose,scaffold,bootstrap_env,api_check,smoke,repair}.py` |
| Shared write executors            | `src/cgx/session/tasks/{apply,verify,ask}.py` |
| Repair classify / locate / propose | `src/cgx/session/repair/{classify,locate,propose}.py` |
| PyPI client + cache (Phases 3.2 / 4.1) | `src/cgx/session/repair/pypi_client.py` |
| Scaffold-time pin validator (Phase 4.1) | `src/cgx/session/scaffold_validate.py` |
| LLM tracing (Phase 5.1)           | `src/cgx/session/llm_trace.py` |
| Cross-session lessons (Phase 7.1) | `src/cgx/session/lessons.py` |
| Decision validation               | `src/cgx/session/tasks/ask.py` :: `build_decision` |
| HTTP surface                      | `src/cgx/webui/routes/agent_session.py` |
| Wire models                       | `src/cgx/webui/models.py` :: `AgentSession*` |
| UI                                | `frontend/src/pages/AgentPage.tsx` + `frontend/src/components/agent/` |
| Integration tests                 | `tests/test_webui_agent_session.py`, `tests/test_session.py` |

The remainder of this document (sections 1–8 below) describes the
**legacy batch agent**, which still backs `/agent-legacy`, the
`cgx agent` CLI, and `cgx.agents.run_agent`.

---

## 1. Agent Type

CGX ships **one** agent: a single-actor, multi-role **orchestrator**
that operates strictly **local-first**.

* **Single-actor, not multi-agent.** There is one logical agent. Inside
  it, three cooperating roles share the same process and the same LLM
  provider -- they are not independent agents communicating over a bus.
* **Plan-and-execute, not ReAct.** A full plan is committed up front;
  the executor does not call the LLM mid-task to decide the next step.
  Retries re-enter the planner, they do not branch off it.
* **Capability-dispatched.** The agent never makes raw shell or file
  calls. Each task is routed to a named **capability** callable
  (`ask`, `plan`, `scaffold`, `apply`, `verify`, …). Callers can
  replace the capability table with stubs for tests or sandboxing.
* **Local-first by default.** The agent runs against a local Ollama
  daemon, a local FAISS index, and a local working tree. No cloud
  service is required, no telemetry is emitted, and the entire loop
  works offline once models and the index are present. Cloud
  providers (Gemini, OpenAI-compatible) are opt-in.
* **Streaming, not blocking.** The default UI path streams
  `AgentEvent` records over SSE so the user sees plan, per-task start,
  heartbeats, completion, judge verdicts, and retry transitions as
  they happen.

---

## 2. Architecture -- Planner → Tracker → Judge

```
                   ┌─────────┐
        goal ────▶ │ Planner │ ──▶ Plan(tasks=[…], rationale)
                   └─────────┘
                        │
                        ▼
                   ┌─────────┐        capability table
                   │ Tracker │ ──▶  ask/plan/scaffold/search/
                   └─────────┘      summarize/apply/verify/fill_logic
                        │
                ┌───────┴────────┐
                ▼                ▼
           AgentEvent…       ┌───────┐
           (SSE stream)      │ Judge │ -- verdict + rationale per task
                             └───────┘
                        │
                  failures? ──▶ planner.plan_fix() ──▶ retry plan
```

* **`Planner`** ([`planner.py`](../src/cgx/agents/planner.py)) -- asks
  the LLM for a strict-JSON plan (`{rationale, tasks:[{name,
  description, kind, criteria}]}`), then runs `_enforce_kind_policy()`
  to route the goal down one of four branches: **SCAFFOLD**,
  **PLAN+APPLY+VERIFY**, **VERIFY-only**, or read-only
  **SEARCH/ASK/SUMMARIZE**. When the LLM is absent or returns
  unparseable output, a deterministic fallback consults
  `cgx.answer.intent.detect_intent` and emits a one-task plan that
  matches legacy single-shot behaviour.
* **`Tracker`** ([`tracker.py`](../src/cgx/agents/tracker.py)) -- index
  loop over `plan.tasks` (so tasks injected mid-run, like
  `SCAFFOLD_MANIFEST` expanding into per-file tasks, are visited).
  Dispatches each task to its capability, emits heartbeats every
  `progress_interval` seconds while a capability is blocked, and
  persists `task.output` + `task.judge` back into the plan.
* **`Judge`** ([`judge.py`](../src/cgx/agents/judge.py)) -- validates
  each completed task against its `criteria` list. LLM-grounded when a
  provider is available; otherwise heuristic (artifact shape +
  per-skill structural checks). Returns `{verdict, rationale,
  confidence}`. A `fail` verdict on a hard-fail kind aborts the plan
  (subject to `_SOFT_FAIL_KINDS`, currently only `SCAFFOLD_FILE`,
  which continues so partial scaffolds survive).

The whole loop is wired in
[`run_agent()`](../src/cgx/agents/loop.py). The streaming variant
adds `_stream_with_retry` on top, which re-enters the planner when
`verify` or `apply` fails and renames the next plan's events from
`plan` to `retry_plan` so the UI appends rather than replaces.

---

## 3. Agentic Capabilities (Task Kinds)

The agent's atomic operations are enumerated in
[`TaskKind`](../src/cgx/agents/types.py). Each kind maps 1:1 to a
capability in the default capability table built by
`_build_default_capabilities`.

| Kind                 | Purpose                                                              | Backing function |
|----------------------|----------------------------------------------------------------------|------------------|
| `search`             | Retrieve code chunks from the FAISS index for the current goal.      | `cgx.pipeline.auto.run_query_auto` |
| `ask`                | Answer a grounded natural-language question over the indexed code.   | `cgx.answer.engine.answer_with_llm` |
| `summarize`          | Condense prior task outputs into ≤8 bullets via the LLM.             | inline `provider.chat` call |
| `plan`               | Produce a unified-diff change plan against an **existing** codebase. | `cgx.answer.engine.generate_code_plan` |
| `scaffold`           | Generate a brand-new project from scratch (no index required).       | `cgx.answer.engine.generate_project_scaffold` |
| `scaffold_manifest`  | Cheap LLM call that returns only the file list for a new project.    | injects `scaffold_file` tasks into the running plan |
| `scaffold_file`      | Generate exactly **one** file given its spec + sibling context.      | per-file scaffold call |
| `fill_logic`         | Phase-2 of skeleton-and-fill: replace empty bodies in a skeleton.    | targeted edit call |
| `apply`              | Write a prior `plan`/`scaffold` diff set to disk + smoke-test.       | `cgx.codegen.disk_apply.apply_diffs_to_disk` |
| `verify`             | Run impacted (or all) pytest tests against the working tree.         | `cgx.codegen.test_runner.run_tests_on_disk` / `run_pytest_paths` |

The kinds are intentionally **coarse** -- each one is the cheapest
unit of work that still produces a verifiable artifact. There is no
"call this Python function" or "edit this hunk" primitive; the agent
expresses fine-grained intent through the prompt to the underlying
capability, not through more atomic tools.

---

## 4. Agent Style

The behavioural choices that distinguish the CGX agent from a generic
"LLM + tools" loop:

* **Local-first, offline-capable.** No external API is required for
  any capability. The default provider is `OllamaProvider`; the
  default retrieval stack is on-disk FAISS + a JSONL record store.
  Cloud providers (`GeminiProvider`, `OpenAICompatProvider`) are
  optional drop-ins through the same `LLMProvider` interface.
* **Plan-first, with a deterministic safety net.** The LLM is asked
  for a strict-JSON plan; if it returns malformed JSON, no JSON, or
  no tasks, `Planner._fallback_plan` synthesises a single-task plan
  from intent classification so the agent never deadlocks on a bad
  model response.
* **Skill-aware decomposition.** The `skills/` package contributes
  three signals: (1) it influences the planner's SCAFFOLD vs. PLAN
  routing decision; (2) it injects technology-specific instructions
  into the system prompts of `plan`/`scaffold`; (3) it adds
  per-skill structural checks to the Judge (e.g. "a React scaffold
  must include a `package.json` and an `App.jsx`/`App.tsx`").
* **Diff-shaped output, always.** Even scaffolds are emitted as
  `--- /dev/null` new-file unified diffs so the `apply` capability
  has a single code path for both new and edited files.
* **Retrieval is a task, not a side-effect.** When the goal mentions
  a file, symbol, or behaviour, the planner is expected to emit an
  explicit `search` task whose hits feed downstream `ask`/`plan`
  tasks via `prior_outputs`. This keeps index access auditable and
  lets the UI render the retrieval result.
* **Verify is the contract.** Code-change goals always terminate in
  a `verify` task. The plan is only "complete" when `verify`
  succeeds -- or when its failure is classified as unrecoverable
  sandbox / `sys.path` noise (see §5).
* **Errors are structured, not opaque.** Tracker exceptions are
  caught, surfaced as `task_failed` events, persisted on
  `task.error`, and (where applicable) post-processed by
  `_diagnose_failure` so the retry loop can quote the offending file
  and line back to the LLM.
* **UI feedback is incremental.** Long-running capabilities emit
  `task_progress` heartbeats every two seconds; the React Agent page
  consumes these to keep the timeline alive without polling.

---

## 5. The Retry Loop and Self-Correction

`run_agent` re-enters the planner up to `max_retries` times (default
`1`) when the first plan ends with failures. The retry path is
**targeted, not blanket**:

1. **Failure classification.** `_extract_verify_failures`,
   `_extract_apply_failures`, and `_extract_core_failures` partition
   the failed tasks. `_diagnose_failure` parses pytest tracebacks and
   classifies the error as `import_error`, `syntax_error`,
   `logic_error`, or `unknown`, then extracts the responsible
   project-relative file paths.
2. **Sandbox-failure short-circuit.**
   `_verify_failure_is_unrecoverable` detects pytest collection
   errors (`rc == 2`) caused by `ModuleNotFoundError` on a first-party
   project directory. These are packaging / `sys.path` issues the LLM
   cannot fix by regenerating code, so the retry is skipped and the
   `verify` task is demoted to "complete with warnings".
3. **Targeted regeneration.**
   * If only `scaffold_file` tasks failed, the loop builds a
     scaffold-retry plan (`_build_scaffold_retry_plan`) that
     regenerates only the broken files and preserves the siblings
     already on disk via `plan.owned_files`.
   * If `verify` failed against an existing codebase, the loop calls
     `planner.plan_fix(fix_goal, broken_files=…, already_good_files=…)`
     which constrains the new PLAN task to a `target_files` /
     `do_not_change` allow-list folded into the prompt.
4. **The 10-line buffer rule.** `_extract_error_snippet` pulls ±5
   lines around the first traceback line from the failing file and
   injects them into the retry prompt. Small models (3B-class) drown
   in full tracebacks; a tight snippet keeps them focused.
5. **Streaming continuity.** Retry plans are streamed under the
   `retry_plan` event so the UI appends new task rows instead of
   replacing the original timeline; a `retry_start` event carries
   the human-readable reason.

The retry is bounded: one re-plan by default. There is no open-ended
"keep trying until it works" loop, because every retry costs an LLM
call and a test run, and unbounded retries against a 3B-class model
diverge faster than they converge.

---

## 6. Integration Surfaces

* **Web UI.** The `/api/agent` SSE endpoint streams `AgentEvent`
  records. The React Agent page renders the plan DAG, per-task
  status, judge verdicts, and the rationale card from the `plan`
  event payload. Visual helpers live in
  [`viz.py`](../src/cgx/agents/viz.py).
* **CLI.** `cgx agent "<goal>"` is the terminal entrypoint; it
  consumes the same stream and prints a compact task table.
* **Programmatic.**

  ```python
  from cgx.agents import run_agent
  from cgx.answer.providers import OllamaProvider

  prov = OllamaProvider(model="qwen2.5-coder:3b")

  for event in run_agent(
      goal="Add docstrings to every public function in cgx.parser",
      provider=prov,
      index_dir="/tmp/cgx_index/indices",
      records_path="/tmp/cgx_index/records.jsonl",
      project_root=".",
      stream=True,
  ):
      print(event.type, event.payload)
  ```

  Tests inject their own capability map to bypass the LLM and disk
  entirely -- see `tests/test_agents_*` and the example in
  [`docs/usage.md`](usage.md#programmatic-use).

---

## 7. Rooms for Improvement

The current design is deliberately conservative -- one actor, one
plan, one retry, no live tool-calling. That makes the loop legible
and reproducible, but it leaves clear headroom. The items below are
the most impactful next steps the maintainers and community have
identified; contributions are welcome on any of them.

### 7.1 Orchestration

* **Parallel task execution.** The Tracker walks tasks sequentially
  even when they have no data dependency. `Task.dependencies` already
  carries the DAG edges; a topological scheduler that runs
  independent tasks (e.g. multiple `scaffold_file` siblings, or a
  `search` task in parallel with a `summarize`) would significantly
  cut wall-clock time on multi-layer scaffolds. Streaming would need
  per-task lanes in the SSE protocol.
* **True multi-agent split.** The "Planner / Tracker / Judge" roles
  share one provider today. A reviewer / critic role with a
  different (possibly larger, possibly slower) model could review
  PLAN outputs before APPLY, in the style of a reflective
  critic-actor pair. This is independent of parallelism: the agent
  would still be a single orchestrator, but its sub-roles would each
  speak through their own provider configuration.
* **Plan revision mid-stream.** Today a plan is committed up front
  and only re-planned at the end. A capability that lets the
  Tracker request a plan amendment (e.g. "this scaffold revealed I
  need a new layer") would close the gap between plan-and-execute
  and ReAct, without giving up the structured plan event the UI
  depends on.
* **Unbounded retry budgets with confidence gating.** `max_retries`
  is a hard cap. Replacing it with a confidence-weighted budget
  (e.g. "keep retrying while the Judge's confidence is trending up")
  would let strong models converge on harder problems without
  letting weak models loop indefinitely.

### 7.2 Tool-Use Expansion

* **Finer-grained file ops.** The agent only writes whole-file diffs
  via `apply`. Adding a `patch` / `rename` / `delete` task kind with
  its own Judge contract would let small models make surgical
  changes the current diff-only pipeline forces them to express as
  full-file rewrites.
* **Shell execution as a first-class capability.** `verify` runs
  pytest, but there is no general `run_command` kind. A sandboxed
  shell capability -- gated on a per-command allow-list and confined
  to the project venv -- would unlock `npm install`, `cargo check`,
  `tsc --noEmit`, and other language-native verifiers that are
  currently impossible to plan.
* **HTTP / network capability.** Goals like "fetch the OpenAPI spec
  at URL X and generate a client" cannot be expressed today. A
  bounded `fetch` capability with a domain allow-list would open
  the door without breaking the air-gapped default (it would be
  off unless the user opts in).
* **Browser / headless rendering.** For UI-heavy scaffolds, a
  Playwright-backed `screenshot` or `dom_snapshot` capability would
  let the Judge verify *visual* criteria rather than just structural
  ones. This is high-value for React / Vue / Svelte goals where the
  current Judge can only check file structure.
* **IDE / LSP integration.** The agent currently runs blind to
  language-server diagnostics. Piping `pyright` / `tsc` / `gopls`
  output into the apply-time smoke test would catch type errors
  before they reach `verify` and waste a full test run.

### 7.3 Planner Quality

* **Learned routing instead of regex-based intent.**
  `_SCAFFOLD_RE`, `_CHANGE_VERB_RE`, `_TECH_RE`, and
  `_VERIFY_ONLY_RE` are brittle. A small classifier (logistic
  regression on goal embeddings against a labelled corpus of past
  agent runs) would generalise better and degrade more gracefully
  than the current regex cascade.
* **Plan-rationale grounding.** The planner emits a free-text
  `rationale` but it is not currently validated. The Judge could
  cross-check that every claim in the rationale ("the goal needs a
  React UI, FastAPI backend, and pytest suite") matches at least
  one task -- catching planner hallucination at zero extra LLM cost.
* **Goal disambiguation.** Ambiguous goals collapse to whatever the
  LLM picks. A pre-planner clarification step ("Did you mean to
  modify the existing project at `./` or to create a new project?")
  driven by the `_EXISTING_CODE_HINT_RE` signal would prevent the
  whole-plan misroute that is currently the most expensive failure
  mode.

### 7.4 Verifier and Sandbox Hardening

* **Container-isolated `verify`.** Today `verify` runs pytest in the
  project's own venv, but on the host filesystem and host user.
  Running it inside a rootless Podman / Docker container with a
  read-only mount of the staging directory would close the gap
  where generated code can execute arbitrary code at collection
  time.
* **Resource caps.** There is no CPU / memory / wall-time cap on
  generated test runs beyond `timeout_seconds`. Cgroup limits or
  the equivalent on macOS would prevent an infinite-loop test from
  saturating the user's machine.
* **Verifier diversity.** `verify` is pytest-only. Adding language
  detectors that pick `vitest`, `jest`, `cargo test`, `go test`,
  `phpunit`, etc. based on the project manifest would make the
  contract real for non-Python scaffolds (which today get a
  `verify` task that finds nothing to run).
* **Coverage-aware test selection.** `discover_all_tests` runs the
  whole suite when there is no APPLY history. A coverage map keyed
  on the changed files would let `verify` run only the impacted
  subset even on first-touch goals.

### 7.5 Skills System

* **Skill discovery from disk.** Skills are hand-registered in
  `skills/__init__.py`. A `skills/` directory scan with a
  registration decorator would let third-party packages contribute
  skills without forking the registry.
* **Skill versioning.** Skill detection is binary (does it fire?).
  Versioning ("React 18 vs. React 19") would let the system prompts
  and Judge checks track upstream changes without conditional
  branches inside each skill module.
* **Cross-skill conflict resolution.** Today a goal can legitimately
  trigger React + FastAPI + SQLite + Tailwind at once. There is no
  explicit conflict layer when two skills disagree (e.g. two
  competing build tools). A conflict matrix consulted at planning
  time would let the planner ask for clarification instead of
  emitting a plan that mixes incompatible toolchains.

### 7.6 Memory and Context

* **Cross-run memory.** Each `run_agent` invocation starts fresh.
  Persisting a per-project "agent memory" (what worked, what didn't,
  which files the user reverted) would let the planner avoid
  repeating known-bad approaches. The `.cgx_runs/` directory is the
  natural home for this.
* **Symbol-map freshness.** `build_symbol_context_prompt` reads
  from the records store at plan time, but does not detect when the
  user has edited the project since the last index. An auto-reindex
  trigger on `mtime` changes would prevent the planner from
  emitting diffs that conflict with files it cannot see.
* **Citation-grounded answers.** `ask` outputs include citations,
  but the Judge does not currently penalise an answer that fails to
  cite a hit. Tightening that contract would reduce hallucinated
  references in read-only flows.

### 7.7 Observability

* **Structured event log on disk.** SSE events are streamed to the
  UI but not persisted. Writing them to `.cgx_runs/<plan_id>.jsonl`
  would give users a complete replay log per run -- essential for
  bug reports and for the cross-run memory item above.
* **Cost / token accounting.** There is no per-task token or
  wall-time accounting surfaced in the UI. Adding it would help
  users tell whether a slow run is dominated by planning, a single
  scaffold call, or verify execution -- and would let the planner
  cost-budget its own decomposition.
* **Trace export.** OpenTelemetry-compatible trace export (opt-in,
  off by default to preserve the air-gapped guarantee) would let
  teams running CGX in production tie agent runs to their existing
  observability stack.

---

## 8. Where to Start Contributing

If you want to land your first change in the agent layer, the
easiest on-ramps are:

* Add a new **skill** (see [`CONTRIBUTING.md`](../CONTRIBUTING.md)).
  Skills are the lowest-coupling extension point: one file, one
  test file, no changes to the orchestrator.
* Add a new **capability** by extending `TaskKind`, wiring it into
  `_build_default_capabilities`, and adding a Judge branch. The
  `fill_logic` capability is the most recent example of this
  pattern and is a good template.
* Improve a **diagnoser** in `loop.py` -- `_diagnose_failure` and
  `_extract_error_snippet` are pure functions over failure
  payloads, easy to unit-test, and produce immediate user-visible
  quality gains in the retry loop.
* Improve the **planner prompt** in `planner.py::SYSTEM_PROMPT` and
  add a regression test under `tests/test_agents_planner.py` that
  pins the new behaviour against the deterministic fallback.

See [`docs/architecture.md`](architecture.md) for the broader
system context and [`docs/book.md`](book.md) for the deep technical
history of the pipeline.
