# Swarm Agent

Swarm is a **plan-driven, one-file-at-a-time** build engine. Instead of a
free-form agent loop, the work is split into three router-driven roles and
every stage is **propose-then-validate**: the model proposes, deterministic
invariants enforce.

Swarm is **polyglot**. It was originally Python-only; it now plans, generates,
and verifies **multi-language, multi-component projects** — e.g. a
Flask/FastAPI/Django backend paired with a React/Vue/Next/Express frontend,
plus CLIs and libraries. Python and JS/TS/Node are **first-class**; new stacks
are added by shipping a skill plus a test runner (see **Extending the stack
matrix** below), not by touching the swarm itself.

## How It Works

When you start a session in `swarm` mode, the router drives three roles:

### 1. The Tech Lead (`SWARM_TECH_LEAD`)
Authors and validates the build plan.
- **Skill-aware planning.** Before drafting, the Tech Lead resolves the goal's
  active skills — auto-detected from the goal, or an explicit session /
  Agent-Profile pin — and injects their guidance into the plan prompt. A
  *"Flask + React"* goal therefore plans **both** components. Each active
  skill's validator can **veto** a plan that omits required files, so a plan
  that skips a whole component is rejected rather than half-built. See
  **[[Skills Registry]]**.
- Prompts the model for a draft JSON plan (files, `depends_on` edges,
  per-file contracts).
- Normalizes it (dedupe + prune dangling edges), topologically orders the
  files, and gates on buildability -- with a bounded 3-attempt corrective
  re-ask if the draft is unbuildable.
- **Guarantees a complete project deterministically, per ecosystem.** Rather
  than trust a weak model to remember boilerplate, the plan is auto-completed
  for each component's language: a Python component gets a `requirements.txt`
  (and, for a `src/` layout, a root `conftest.py`), while a JS/TS component
  gets a `package.json`. A `README.md` and a `tests/test_<module>.py` for
  every untested source module are injected as before. A final `verify_plan`
  gate rejects unsafe paths, dependency cycles, orphan tests, and
  still-missing scaffolding.
- **Mixed layouts are valid.** A `backend/` + `frontend/` tree is legal; the
  old rule forcing every module to live uniformly under `src/` or top-level is
  now **Python-only** — it constrains a Python component's rooting but does not
  reject a multi-component repo.
- Persists a `WORK_PLAN` artifact and hands the ordered file list to the
  Developer chain. If no buildable plan can be produced, the session ends
  FAILED rather than spawning empty work.

### 2. The Developer (`SWARM_DEVELOPER`)
Implements **exactly one planned file per turn**, in dependency order.
- **Skill-guided, language-aware generation.** Each file is generated under the
  guidance of the component's active skill(s), and the ladder branches on the
  file's language rather than assuming Python.
- Grounds each file on the *real on-disk content* of its dependencies so it
  sees the actual sibling symbols.
- Routes non-source files off the code ladder: `requirements.txt` / `conftest.py`
  (Python) and `package.json` (JS/TS) come from deterministic source-derived
  templates, `README.md` from a grounded free-form call.
- Runs a generation ladder for source files: full-file generation (gated on a
  real `ast.parse`, one re-ask) falling back to a deterministic AST assembler
  that builds the module header + each required symbol from the plan
  contracts. It features advanced auto-repair mechanisms:
  - **AST Import Injector**: Identifies missing standard library or first-party
    imports and injects them directly into the AST, bypassing the LLM.
  - **Contract Renegotiation**: If a signature changes during implementation,
    the contract is dynamically renegotiated rather than failing the build.
  - **Semantic Repair Fallback**: For more complex logical errors, a targeted
    fallback repair is attempted. This is language-aware — it works on JS/TS
    sources, not just Python.
- Applies two hard code-quality gates on the parsed source: a **phantom-import
  gate** (a provably-unused import fails the file) and a **no-stub gate** (a
  contract function or method whose body is just `pass` / `...` / a docstring /
  `raise NotImplementedError` is rejected with a re-ask naming it), so a stub
  can no longer pass structurally and break the suite the instant a test calls
  it.
- Writes new files with `edit_file` and edits existing ones with
  `patch_file`; emits per-file progress. Any file it cannot produce is
  recorded in `failed_paths` and carried forward.

### 3. The Verifier (`SWARM_VERIFY`)
Runs a graded verification ladder over the finished tree.
- **Manifest reconciliation** before any build or test. The Verifier scans each
  component's source imports and makes its manifest actually declare and install
  them — a Python component's third-party imports are installed and pinned into
  `requirements.txt`; a JS/TS component's bare imports are `npm install`ed into
  its `package.json`. This is **one general mechanism for any package**, not a
  per-package rule. FastAPI/Starlette additionally imply `httpx` — its
  `TestClient` needs it at runtime but never imports it — via a single bounded
  framework map, so the dynamic suite doesn't fail on a missing test-only dep.
- **Manifest co-location.** A `package.json` the model left at the repo root
  while the app lives under `frontend/` is moved to the component root next to
  `index.html`, so the Vite build resolves its entry instead of failing to find
  one.
- **Static** structural checks next. Only **two** findings gate the build:
  test-coverage gaps (a planned test that defines nothing to run) and
  first-party import breaks. **Phantom third-party imports** (an import not in
  the plan's declared dependencies) and **contract warnings** are now
  **advisory**, not gating. Rationale: reconciliation installs every real
  import, so a truly hallucinated package fails the real install anyway, and a
  legitimate-but-under-declared dependency (e.g. `uvicorn`) must not fail a
  build whose tests actually pass — **the real build/test is the authority**.
  Named files still drive a bounded targeted regeneration.
- **Dynamic** dry-run only if static passes: install missing dependencies,
  then run the **polyglot test runner** instead of a pytest-only pass. A Python
  component is exercised with `pytest`; a JS/TS component gets a real
  build/test gate via `npm test` / `npm run build`. The runner now **discovers
  `package.json` in component subdirs** (a monorepo `frontend/`) and installs
  with `--legacy-peer-deps`, so an imperfect peer-version pin doesn't abort the
  install. A mixed-root repo is verified **as a whole** today — full
  per-component isolation (separate roots and verdicts) is a planned follow-up.
- **Failure-driven repair** when the tree is clean but the suite is red: 
  - **AST-Driven Auto Repair**: Missing imports and specific logical bugs are isolated via Python's AST and repaired using surgical string-injections (bypassing the strict JSON response format which smaller models struggle with).
  - **Dynamic Repair**: For complex structural issues, the pytest output and implicated files are fed back to the model which returns corrected complete files.
  - **Dynamic Temperature**: All iterative repair loops scale the LLM's temperature incrementally (0.2 -> 0.8) on each round to prevent infinite repetitive output cycles.
  - **Larger repair budget**: the loop was raised from **3 to 5 rounds** (with the temperature ramp above) to give a weaker local model more chances to converge on a genuine logic bug.
  repair may fix **either side** -- when a test asserts a value the goal never
  specified or calls the API wrongly, the *test* is rewritten to assert an
  invariant or round-trip instead of forcing impossible source. Tests
  themselves are authored to construct inputs inline and assert
  invariants/round-trips rather than fabricated literals.
- Produces a verification report that pinpoints the files implicated by any
  failure.
- **Honest by design.** A small local model can still emit a logic bug the
  bounded repair can't fix; when that happens the session honestly reports
  **FAILED** — it never falsely passes a red suite.

## Tools: one registry, tolerant dispatch

Every swarm role now shares a single **tool registry**
(`cgx.session.tasks.tool_registry`). A `ToolSpec` bundles a tool's name, its
LLM-facing description, its handler, a `RiskLevel`, and a short `arg_hint`.
This replaces the three previously-divergent hardcoded dispatch chains (one
each in the Tech Lead, Developer, and diagnose paths), so the agent's view of
its own toolset can no longer drift from what the code can actually run.

- **Auto-injected descriptions**: tool help is generated from the registry and
  injected into the agent's system prompt (`REGISTRY.describe_for_prompt(...)`),
  so the model always knows which tools exist and how to call them.
- **Tolerant parsing**: `parse_tool_calls(text)` extracts **every**
  `<call_tool name="...">{json}</call_tool>` block in a reply (the old parser
  matched only the first and broke on quote style or stray whitespace).
- **Native tools**: `run_python_probe` (HIGH), `file_skeleton`, `list_symbols`,
  `query_codebase` (LOW), and `search_web` (MEDIUM). The dead `bash_repl` and
  `patch_file` tools and the unused `swarm_parse` module were removed.

Adding a native — or MCP — tool is now a single `register` call; nothing in the
generation loop changes.

### MCP tools and human approval

When at least one **MCP server** is configured (`~/.cgx/mcp.json`), the swarm
also gains `mcp_list_servers`, `mcp_list_tools`, and `mcp_call` (HIGH risk).
Discovery is lazy so many servers don't flood the prompt, and MCP tools are only
advertised when a server is actually configured. See **[[Providers and Models]]**.

Risky tool calls (code execution, file writes, MCP calls) can be gated behind an
opt-in **human-in-the-loop approval** step — off by default, enabled with
`cgx agent --approve` or the `CGX_APPROVAL_MODE` env var. See
**[[Privacy and Security]]** and **[[Configuration and Tuning]]**.

## Running a Swarm Session

Use the `--mode swarm` flag in the CLI:

```bash
python -m cgx.cli.main agent --model qwen2.5-coder:7b-instruct --mode swarm "your objective here"
```

## Extending the stack matrix

The swarm's language coverage is **data, not code**. Python and JS/TS/Node are
first-class today; you add a new stack by shipping two things and touching
nothing in the swarm itself:

- a **skill** under `skills/` — teaches planning and scaffolding about the
  technology and supplies the validator that vetoes an incomplete plan (see
  **[[Skills Registry]]**), and
- a **test runner** in `test_runners.py` — gives the Verifier a real
  build/test gate for that ecosystem.

## Correctness & robustness fixes

- **No false FAILED verdict**: a file recorded in `failed_paths` but
  successfully regenerated by Verify no longer sinks the session — the verdict
  is reconciled against the final on-disk structural scan.
- **No mid-chain contract truncation**: a renegotiated contract is now
  **merged** (superset) into the plan rather than replacing it, so contracts for
  not-yet-generated files are never dropped.
- **Bounded context growth**: tool responses and injected dependency bodies are
  truncated, so a long chain can't grow the prompt without limit.
- **Debate rationale recorded**: debate mode now records the judge's reasoning,
  not just the A/B letter.

In the **[[Web UI Guide]]** the swarm view was fixed too: Developer tasks now
sit as siblings under the Tech Lead (no runaway per-file indentation), the live
progress banner reads the correct phase (Planning / Verifying, not always
"Generating"), and the Facts feed shows role- and phase-labeled beats with
concise summaries.

## Advantages
- **Bounded, deterministic invariants**: coherence, toposort, contract,
  syntax, phantom-import, and no-stub gates catch a bad plan or a broken file
  before it propagates.
- **A complete, runnable project every time**: README, the per-ecosystem
  dependency manifest (`requirements.txt` / `package.json`), a root
  `conftest.py` for a Python `src/` layout, and one test per module are
  injected into the plan deterministically, so a run never ships uninstallable
  or untestable output — for a Python **or** a JS/TS component.
- **Grounded generation**: each file is generated against the true on-disk
  symbols of its dependencies, not a guess.
- **Self-correcting**: a red suite on a structurally-clean tree feeds its own
  failure output back to the model to repair the offending source *or* test.
- **Reduced context pressure**: one file per turn keeps prompts small, so a
  local model (e.g. Ollama) is far less likely to hit read timeouts.
- **Plan-aware execution budget**: the drain ceiling scales to the plan's
  file count so large builds are not truncated mid-chain.
