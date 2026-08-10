# Swarm Agent

Swarm is a **plan-driven, one-file-at-a-time** build engine. Instead of a
free-form agent loop, the work is split into three router-driven roles and
every stage is **propose-then-validate**: the model proposes, deterministic
invariants enforce.

## How It Works

When you start a session in `swarm` mode, the router drives three roles:

### 1. The Tech Lead (`SWARM_TECH_LEAD`)
Authors and validates the build plan.
- Prompts the model for a draft JSON plan (files, `depends_on` edges,
  per-file contracts).
- Normalizes it (dedupe + prune dangling edges), topologically orders the
  files, and gates on buildability -- with a bounded 3-attempt corrective
  re-ask if the draft is unbuildable.
- **Guarantees a complete project deterministically.** Rather than trust a
  weak model to remember boilerplate, the plan is auto-completed: a
  `README.md`, a dependency manifest (`requirements.txt`), and -- for a
  `src/` layout -- a root `conftest.py` are injected if missing, and a
  `tests/test_<module>.py` is injected for every source module that has no
  test. A final `verify_plan` gate rejects unsafe paths, dependency cycles,
  orphan tests, and inconsistent routing structures (all modules must uniformly live under `src/` or top-level), along with any still-missing scaffolding.
- Persists a `WORK_PLAN` artifact and hands the ordered file list to the
  Developer chain. If no buildable plan can be produced, the session ends
  FAILED rather than spawning empty work.

### 2. The Developer (`SWARM_DEVELOPER`)
Implements **exactly one planned file per turn**, in dependency order.
- Grounds each file on the *real on-disk content* of its dependencies so it
  sees the actual sibling symbols.
- Routes non-source files off the code ladder: `requirements.txt` and
  `conftest.py` come from deterministic source-derived templates, `README.md`
  from a grounded free-form call.
- Runs a generation ladder for source files: full-file generation (gated on a
  real `ast.parse`, one re-ask) falling back to a deterministic AST assembler
  that builds the module header + each required symbol from the plan
  contracts. It features advanced auto-repair mechanisms:
  - **AST Import Injector**: Identifies missing standard library or first-party
    imports and injects them directly into the AST, bypassing the LLM.
  - **Contract Renegotiation**: If a signature changes during implementation,
    the contract is dynamically renegotiated rather than failing the build.
  - **Semantic Repair Fallback**: For more complex logical errors, a targeted
    fallback repair is attempted.
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
- **Static** structural checks first: test-coverage gaps (a planned test that
  defines nothing to run is a gap), first-party import coherence, and contract
  compliance. Named files drive a bounded targeted regeneration.
- **Dynamic** dry-run only if static passes: install missing dependencies,
  then run the impacted tests.
- **Failure-driven repair** when the tree is clean but the suite is red: 
  - **AST-Driven Auto Repair**: Missing imports and specific logical bugs are isolated via Python's AST and repaired using surgical string-injections (bypassing the strict JSON response format which smaller models struggle with).
  - **Dynamic Repair**: For complex structural issues, the pytest output and implicated files are fed back to the model which returns corrected complete files.
  - **Dynamic Temperature**: All iterative repair loops scale the LLM's temperature incrementally (0.2 -> 0.8) on each round to prevent infinite repetitive output cycles.
  repair may fix **either side** -- when a test asserts a value the goal never
  specified or calls the API wrongly, the *test* is rewritten to assert an
  invariant or round-trip instead of forcing impossible source. Tests
  themselves are authored to construct inputs inline and assert
  invariants/round-trips rather than fabricated literals.
- Produces a verification report that pinpoints the files implicated by any
  failure.

## Running a Swarm Session

Use the `--mode swarm` flag in the CLI:

```bash
python -m cgx.cli.main agent --model qwen2.5-coder:7b-instruct --mode swarm "your objective here"
```

## Advantages
- **Bounded, deterministic invariants**: coherence, toposort, contract,
  syntax, phantom-import, and no-stub gates catch a bad plan or a broken file
  before it propagates.
- **A complete, runnable project every time**: README, dependency manifest,
  root `conftest.py`, and one test per module are injected into the plan
  deterministically, so a run never ships uninstallable or untestable output.
- **Grounded generation**: each file is generated against the true on-disk
  symbols of its dependencies, not a guess.
- **Self-correcting**: a red suite on a structurally-clean tree feeds its own
  failure output back to the model to repair the offending source *or* test.
- **Reduced context pressure**: one file per turn keeps prompts small, so a
  local model (e.g. Ollama) is far less likely to hit read timeouts.
- **Plan-aware execution budget**: the drain ceiling scales to the plan's
  file count so large builds are not truncated mid-chain.
