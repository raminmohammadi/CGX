# Session-Based Agent

The **Agent** tab (`/agent`) and the `cgx agent` CLI are backed by a
**persistent, session-shaped** orchestrator in `cgx.session`. Every
interaction belongs to a `Session` whose state survives process restarts
under `<project_root>/.cgx/sessions.db`. The agent works toward a goal one
typed task at a time, pausing at every branch so **you approve each
decision** — nothing reaches disk until you say so.

This page is the practical guide. The full internals live in
[`docs/Agent.md`](https://github.com/raminmohammadi/Averix/blob/main/docs/Agent.md) and **[[Architecture]]**.

---

## Two modes

The mode is auto-detected by `detect_mode(project_root)` at session
creation (or set explicitly via the launcher / API):

- **Explore** — the project root exists and has a usable FAISS index.
  Use it for exploratory questions ("how should we refactor this layer?")
  and grounded code changes on an existing codebase.
- **Greenfield** — the project root is missing, empty, or has no index.
  Use it to scaffold a brand-new project from a plain-language idea; the
  agent clarifies requirements, plans a layered file manifest, and only
  writes after you approve.

The detector prefers **greenfield** when signals are ambiguous, because
mis-seeding explore against a missing index fails immediately while
greenfield still produces useful clarification questions.

---

## The explore loop

```
EXPLORE -> ASK_USER(choose_path)
              -> INVESTIGATE -> RECOMMEND -> ASK_USER(choose_recommendation)
                                                -> investigate_more (loop)
                                                -> plan_change
                                                     -> PLAN_CHANGE
                                                        -> ASK_USER(approve)
                                                           -> APPLY -> VERIFY
                                                -> ask_followup / done
```

- **EXPLORE** produces a `DIRECTIONS_LIST` and one `ANCHOR` fact per
  option. You pick a direction.
- **INVESTIGATE** runs a deeper anchored retrieval → `FINDINGS_BUNDLE`.
- **RECOMMEND** synthesises typed next steps, each with a `kind`
  (`investigate_more` / `plan_change` / `ask_followup` / `done`) that
  decides what the router spawns next.
- The `plan_change` path produces a `CODE_CHANGE_PLAN`, gates on an
  **APPROVE** checkpoint, applies the diffs (backed up under
  `.cgx-backups/`), and runs `VERIFY`.

## The greenfield loop

```
CLARIFY_REQUIREMENTS -> ASK_USER(clarify_answers)
   -> DECOMPOSE (scope ceiling + self-critique + de-scope
                 + coherence re-ask + contracts + layers)
        -> ASK_USER(approve_plan)
        -> SCAFFOLD (contract + coherence gates) -> APPLY
             |-(on failure)-> AST_REGENERATE -> APPLY
             -> BOOTSTRAP_ENV -> API_CHECK -> SMOKE -> VERIFY
                  -> (passed) RUNTIME_VERIFY -> COMPLETED
                  -> (mechanical failure) REPAIR --> APPLY (re-enters loop)
                  -> (reasoning-class failure) DIAGNOSE
                        --> {REPAIR | BOOTSTRAP_ENV | scoped SCAFFOLD | regenerate}
                        --> (verify-origin fix) RE_VERIFY (only failing test file(s))
```

- **CLARIFY_REQUIREMENTS** asks 3–6 questions about stack, storage,
  schema, target environment → `REQUIREMENTS_SHEET`. A deterministic
  **scope calibration** (P1.1, `estimate_scope`, no LLM) also reads how
  much project the goal actually asks for and stamps a
  `project_complexity` tier + `scope` block onto the sheet.
- **DECOMPOSE** folds your answers into the goal and emits a `WORK_PLAN`. Four **prevention** passes keep the plan honest before it ships: **(P1.1) scope ceiling** — the calibrated "minimal viable stack" is threaded into `plan_scaffold_manifest` as a hard constraint, so a plain "calculator" cannot pull in a DB + auth + Selenium + React; **(P1.2) plan self-critique** — one bounded LLM pass drops speculative files the goal never asked for (guarded, and a no-op when the provider is unavailable); **(P1.4) de-scoping** — a deterministic pass drops any *sandbox-unrunnable* capability the goal never requested (today browser/E2E suites that cannot run without a display); and **(P1.3) coherence surgery signal** — a manifest the deterministic coherence gate had to *heavily* repair is re-planned once (under `DECOMPOSE_RETRY_BUDGET`) instead of scaffolding the churn. The manifest is then **programmatically bucketed** into 4 strict topological layers (Models -> Core -> API -> Tests) to enforce dependency resolution, and a dedicated **Project Skeleton** pass generates interface stubs for all files, embedding them into a `contracts` block that every subsequent file generation step must adhere to (alongside shared endpoints / schemas / functions / constants).
- **SCAFFOLD** generates each file strictly layer-by-layer, injecting the Project Skeleton and prior layer context, then runs
  best-effort static gates (import coherence, contract compliance,
  client/server payload & response coherence) before anything is written. If generating the file as a whole fails repeatedly, the router escalates to **AST_REGENERATE** to parse the skeleton and prompt the LLM symbol-by-symbol (function by function).
- **BOOTSTRAP_ENV** provisions a project-local environment (a `.venv` for
  Python, the JS toolchain for `package.json`) and installs both declared
  and detected-undeclared dependencies. A **dead-dependency de-scope**
  (P1.4) preflight also scrubs a browser/E2E distribution (Selenium /
  Playwright / …) that `requirements.txt` declares but no applied file
  imports, so an unrunnable, unused package never slows the install.
- **VERIFY** runs the project's tests through a pluggable runner registry
  (pytest for Python; `npm test` / `npm run build` for JS/TS — a polyglot
  repo is verified in one pass).
- **RUNTIME_VERIFY** actually **boots** each detected entry (`app.py` /
  `main.py` / a Flask/FastAPI app) — turning "the tests pass" into "the
  app runs".

---

## Autonomous repair (greenfield)

A failed `VERIFY` spawns a `REPAIR` task that first tries deterministic,
LLM-free classifiers — a test class missing `unittest.TestCase`, a
`ModuleNotFoundError` for a project module, a missing fixture, a
third-party import break fixed via a PyPI version pin or auto-installation of missing packages. It also parses generic frontend build errors (Webpack/TypeScript/ESLint) to extract filenames for highly targeted repairs, safely escalating untargetable build breaks to save tokens. For ordinary
logic/assertion failures with no mechanical fix, it falls back to a
**bounded LLM repair** that rewrites the smallest set of files (≤5) and
re-validates their syntax before applying.

**The DIAGNOSE reasoning rung (Phase 2).** Mechanical fixes still take an
instant `REPAIR` fast path, but a *reasoning-class* failure
(`assertion_drift` / `collection_error` / `unknown` / `runtime_failure`)
— exactly the cases that used to jump straight to a whole-tree
regenerate — is first handed to a **DIAGNOSE** task, the reasoning rung
between a mechanical patch and a full regenerate. It normalizes the
upstream report into one `FailureContext`, runs a **bounded, read-only
ReAct loop** (`DIAGNOSE_STEPS=3` — read file / grep / inspect installed
packages) under the shared `REPAIR_BUDGET`, and consults a
`REPAIR_LEDGER` of already-tried actions so it never repeats a prior
round. It emits a typed `DIAGNOSIS` carrying a single `minimal_action`
verdict — `patch_files`, `add_dependency`, `remove_dependency`,
`adjust_manifest`, `regenerate_files`, or `escalate` — that the **pure**
router maps to exactly one successor (`REPAIR`, a `BOOTSTRAP_ENV`
install/de-scope, a **scoped** `SCAFFOLD` that regenerates only the named
files, or today's whole-tree regenerate / re-plan). It is
**deterministic-first**: a provider outage or garbled reply degrades
cleanly to `escalate`, so the rung is never worse than before.

**Incremental RE_VERIFY (Phase 3.2, C2).** When a `DIAGNOSE`-driven fix
originated from a `VERIFY` failure, the router does not replay the whole
`BOOTSTRAP_ENV → API_CHECK → SMOKE → VERIFY` chain — the venv is already
provisioned and every other gate already passed. Instead the fix builders
stamp a marker onto the `REPAIR` / `BOOTSTRAP_ENV` node that rides down
the fix chain, and at the edge that would re-enter verification the router
splices a **RE_VERIFY** task that re-runs pytest against **only** the
origin report's failing test file(s). It emits a shape-identical
`VERIFY_REPORT`, so a green re-run hands off to `RUNTIME_VERIFY` and a
still-failing reasoning-class outcome routes back to `DIAGNOSE` under the
shared budget. Non-`VERIFY` origins (and the scoped-scaffold arm) carry no
marker and run the full chain, so behavior is never worse than before.

The loop continues **only while the failing-test count strictly drops**
(absolute ceiling of 4 rounds) and is gated by a failure-signature hash,
so identical repeated failures escalate rather than loop forever. A
session-level budget (`max_task_runs` / `max_wall_seconds`) backstops the
whole run: interactive sessions pause on an `ASK_USER` when spent,
headless ones end terminally `FAILED`.

Greenfield "green" is **fail-closed**: a session is downgraded to
`FAILED` if a scaffolded JS suite was never run, or a bootable server was
never booted — so a `completed` session provably ran every suite and
booted every detected server.

---

## The three modules that own every transition

| Module | Responsibility |
|--------|----------------|
| `cgx.session.router.Router` | Pure-Python deterministic state machine — **no LLM calls, no I/O**. Returns a typed `RouterPlan` of actions (`CreateTask`, `UpdateTaskStatus`, `RecordDecision`, …). Reads `session.mode` to pick the root task and successor chain. |
| `cgx.session.runner.SessionRunner` | Per-session lock, executor dispatch, failure handling, persistence sequencing, and the session-budget circuit breaker. |
| `cgx.session.tasks.*` | One registered executor per `TaskKind` (explore, greenfield, and shared kinds like `APPLY` / `VERIFY` / `ASK_USER`). |

Keeping the router pure means every branch is deterministic and
replayable, and the LLM only ever runs inside executors.

---

## The decision contract

Every `ASK_USER` task carries `inputs.expected_kind` naming the payload
the route layer accepts. The UI forms post exactly these shapes:

| `expected_kind` | `chosen` shape |
|-----------------|----------------|
| `choose_path` | `{anchor_chunk_id, title?}` |
| `choose_recommendation` | `{id, title, rationale, kind, anchor_chunk_id?}` |
| `approve` | `{approved: boolean}` |
| `clarify_answers` | `{answers: {[question_id]: string}}` (non-empty) |
| `approve_plan` | `{approved: boolean}` |
| `freeform` | `{text: string}` |

A mismatch returns HTTP `400` with a clear error so the UI can surface
the failure without re-posting.

---

## HTTP surface

The API at `/api/agent-session/*` has eight endpoints — seven JSON
(create / list / get / message / decision / cancel / delete) plus a
`GET /{sid}/events` **SSE** stream. Mutating endpoints return the full
`AgentSessionState` snapshot so the UI re-renders the task tree in one
round-trip. The UI follows a running session over SSE and falls back to
polling only when the stream is unhealthy.

---

## Drive it programmatically (no UI)

```python
from cgx.answer.providers import OllamaProvider
from cgx.session import SessionRunner, SessionStore
import cgx.session.tasks  # noqa: F401 -- registers executors
from cgx.session.tasks.base import ExecutorDeps

store = SessionStore(project_root="/path/to/proj")
runner = SessionRunner(store)
session = runner.start_session(
    objective="how should we refactor the parser layer?",
    project_root="/path/to/proj",
)
deps = ExecutorDeps(
    project_root="/path/to/proj",
    index_dir="/tmp/cgx_index/indices",
    records_path="/tmp/cgx_index/records.jsonl",
    provider=OllamaProvider(model="qwen2.5-coder:3b"),
    store=store,
)
task = runner.run_next(session_id=session.session_id, deps=deps)
# `task` is now an ASK_USER waiting on a `choose_path` decision.
```

---

## Safety

- **Nothing is written until you approve.** `APPLY` and scaffold writes
  happen only after a checkpoint, inside the configured Project Root.
- **Every overwrite is backed up** under
  `<project_root>/.cgx-backups/<run_id>/`; the whole run is reversible via
  `POST /api/rollback` (the UI's **Undo** button).
- Set the Project Root deliberately — see **[[Privacy and Security]]**.

---

## See also

- **[[Self Testing Code Generation]]** — the validation loop `PLAN_CHANGE`
  and `SCAFFOLD` rely on.
- **[[Skills Registry]]** — how greenfield scaffolding becomes
  technology-aware.
- [`docs/Agent.md`](https://github.com/raminmohammadi/Averix/blob/main/docs/Agent.md) — the exhaustive internals.
- [`docs/flowcharts.md`](https://github.com/raminmohammadi/Averix/blob/main/docs/flowcharts.md#session-shaped-write-loop-agent)
  — the write-loop diagrams.
