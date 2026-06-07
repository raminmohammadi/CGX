# The CGX Agent

This document describes the **agent layer** of CGX — the component that
turns a single natural-language goal into an executed plan against a
real codebase. It is intended for community contributors who want to
understand how the agent works today and where it could be pushed
further.

The code lives in [`src/cgx/agents/`](../src/cgx/agents/). The public
entrypoint is [`cgx.agents.run_agent`](../src/cgx/agents/loop.py).

---

## 1. Agent Type

CGX ships **one** agent: a single-actor, multi-role **orchestrator**
that operates strictly **local-first**.

* **Single-actor, not multi-agent.** There is one logical agent. Inside
  it, three cooperating roles share the same process and the same LLM
  provider — they are not independent agents communicating over a bus.
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

## 2. Architecture — Planner → Tracker → Judge

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
           (SSE stream)      │ Judge │ — verdict + rationale per task
                             └───────┘
                        │
                  failures? ──▶ planner.plan_fix() ──▶ retry plan
```

* **`Planner`** ([`planner.py`](../src/cgx/agents/planner.py)) — asks
  the LLM for a strict-JSON plan (`{rationale, tasks:[{name,
  description, kind, criteria}]}`), then runs `_enforce_kind_policy()`
  to route the goal down one of four branches: **SCAFFOLD**,
  **PLAN+APPLY+VERIFY**, **VERIFY-only**, or read-only
  **SEARCH/ASK/SUMMARIZE**. When the LLM is absent or returns
  unparseable output, a deterministic fallback consults
  `cgx.answer.intent.detect_intent` and emits a one-task plan that
  matches legacy single-shot behaviour.
* **`Tracker`** ([`tracker.py`](../src/cgx/agents/tracker.py)) — index
  loop over `plan.tasks` (so tasks injected mid-run, like
  `SCAFFOLD_MANIFEST` expanding into per-file tasks, are visited).
  Dispatches each task to its capability, emits heartbeats every
  `progress_interval` seconds while a capability is blocked, and
  persists `task.output` + `task.judge` back into the plan.
* **`Judge`** ([`judge.py`](../src/cgx/agents/judge.py)) — validates
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

The kinds are intentionally **coarse** — each one is the cheapest
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
  succeeds — or when its failure is classified as unrecoverable
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
  entirely — see `tests/test_agents_*` and the example in
  [`docs/usage.md`](usage.md#programmatic-use).

---

## 7. Rooms for Improvement

The current design is deliberately conservative — one actor, one
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
  shell capability — gated on a per-command allow-list and confined
  to the project venv — would unlock `npm install`, `cargo check`,
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
  one task — catching planner hallucination at zero extra LLM cost.
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
  would give users a complete replay log per run — essential for
  bug reports and for the cross-run memory item above.
* **Cost / token accounting.** There is no per-task token or
  wall-time accounting surfaced in the UI. Adding it would help
  users tell whether a slow run is dominated by planning, a single
  scaffold call, or verify execution — and would let the planner
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
* Improve a **diagnoser** in `loop.py` — `_diagnose_failure` and
  `_extract_error_snippet` are pure functions over failure
  payloads, easy to unit-test, and produce immediate user-visible
  quality gains in the retry loop.
* Improve the **planner prompt** in `planner.py::SYSTEM_PROMPT` and
  add a regression test under `tests/test_agents_planner.py` that
  pins the new behaviour against the deterministic fallback.

See [`docs/architecture.md`](architecture.md) for the broader
system context and [`docs/book.md`](book.md) for the deep technical
history of the pipeline.
