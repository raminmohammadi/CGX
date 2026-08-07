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
- Persists a `WORK_PLAN` artifact and hands the ordered file list to the
  Developer chain. If no buildable plan can be produced, the session ends
  FAILED rather than spawning empty work.

### 2. The Developer (`SWARM_DEVELOPER`)
Implements **exactly one planned file per turn**, in dependency order.
- Grounds each file on the *real on-disk content* of its dependencies so it
  sees the actual sibling symbols.
- Runs a generation ladder: full-file generation (gated on a real
  `ast.parse`, one re-ask) falling back to a deterministic AST assembler that
  builds the module header + each required symbol from the plan contracts.
- Writes new files with `edit_file` and edits existing ones with
  `patch_file`; emits per-file progress. Any file it cannot produce is
  recorded in `failed_paths` and carried forward.

### 3. The Verifier (`SWARM_VERIFY`)
Runs a graded verification ladder over the finished tree.
- **Static** structural checks first: first-party import coherence, contract
  compliance, and JS/Python payload coherence.
- **Dynamic** dry-run only if static passes: install missing dependencies,
  then run the impacted tests. Produces a verification report that pinpoints
  the files implicated by any failure.

## Running a Swarm Session

Use the `--mode swarm` flag in the CLI:

```bash
python -m cgx.cli.main agent --model qwen2.5-coder:7b-instruct --mode swarm "your objective here"
```

## Advantages
- **Bounded, deterministic invariants**: coherence, toposort, contract, and
  syntax gates catch a bad plan or a broken file before it propagates.
- **Grounded generation**: each file is generated against the true on-disk
  symbols of its dependencies, not a guess.
- **Reduced context pressure**: one file per turn keeps prompts small, so a
  local model (e.g. Ollama) is far less likely to hit read timeouts.
- **Plan-aware execution budget**: the drain ceiling scales to the plan's
  file count so large builds are not truncated mid-chain.
