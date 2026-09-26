# JEV — the typed decision layer

JEV is CGX's answer to a simple observation: a coding agent is a loop around a
model, and the leverage is **not in the loop** — it is in *what the loop feeds
the model on each turn*. Those per-turn choices ("which chunks should this
prompt show?", "should this command run?", "which tool, with what arguments?")
are usually made implicitly, by an append-only transcript. JEV makes them
**explicit and typed**.

> **What "JEV" is here.** A small decision layer that sits *beside* the model:
> `cgx.answer.jev.decide(...)`. The harness hands it a fixed **question** plus
> the current **state**, and it returns a **typed** answer — an enum choice, a
> score, or a bool — with a probability. Because the answer is typed rather
> than prose, the harness can validate it, apply a threshold, and branch
> **deterministically**. JEV *decides*; the model, tools, and plain code do the
> work.

CGX was already most of the way here: it assembles context per query through a
hybrid retrieval engine rather than accumulating a transcript. JEV formalizes
the remaining micro-decisions as typed decision functions.

---

## The six decision points

| Decision point | Question to JEV | Typed answer | In CGX |
|---|---|---|---|
| **Context** | How visible should this chunk be for this query? | `hide` / `short` / `long` / `full` | ✅ shipped — visibility ladder |
| **Cache** | Reuse the running context or rebuild it? | fits / trim (num_ctx) | ✅ shipped — deterministic guard |
| **Tools** | Which tool fits this intent, with what arguments? | schema on demand + validation | ✅ shipped — tiered MCP disclosure |
| **Permissions** | Should this command run? | `allow` / `ask` / `deny` + confidence | ✅ shipped — programmable policy |
| **Security** | Which files will this task touch? | `public` / `standard` / `restricted` | ✅ shipped — sensitivity scorer |
| **Routing** | Can this subtask leave the main model? | choice + cost estimate | ⏳ deferred (see *Local-first*) |

---

## The core: `jev.decide()`

`cgx.answer.jev.decide(provider, prompt, schema, *, decision_key=…, prob_key=…,
prefer_logprobs=…)` returns a `TypedDecision(value, probability, raw, violations,
ok)`:

- It calls the provider with **schema-constrained decoding** (`force_json` +
  the JSON schema), reusing `cgx.answer.schemas.validate_json_schema` and the
  provider translators already in the codebase.
- It **re-checks** the parsed reply against the schema and issues **one bounded
  corrective re-ask** on a violation.
- On a provider crash or unparseable reply it returns `ok=False` with an empty
  result, so callers keep their existing safe default. It never raises.

Two decisions already route through it today: DIAGNOSE's per-turn output
(`diagnose._diagnose_call`, now schema-constrained so a weak model can't emit an
out-of-enum action) and the SWARM_ASSESS relevance verdict (`swarm_assess`).

---

## What ships today

### Context — the visibility ladder
`cgx.answer.context_map.decide_visibility` assigns every retrieved chunk a level
— `full` / `long` / `short` / `hide` — banded by its relevance to *this* query,
and `build_tiered_context` renders each level accordingly. It is now the
**default** assembly path for **every** query (the old graph-neighbour gate is
removed). The rank-1 hit is always `full`; weak matches collapse to a one-line
`name(signature) — doc` stub or drop out. Since reading dominates the token
budget, spending a small window on the right chunks is the single biggest lever.
Deterministic and model-free by default.

### Cache — the num_ctx guard
`cgx.session.context_budget.fit_react_messages` guards the agent's in-task
reasoning loops (the Swarm Tech Lead planner and DIAGNOSE). It keeps the
system + objective prefix **byte-identical** (so the local prompt cache stays
warm) and elides the **oldest** tool observations before they overflow the
window — so a large tool result can no longer silently truncate the model's own
reasoning off the tail. It is an exact fit check, not a model-scored guess.

### Permissions — programmable, content-inspecting policy
`cgx.guardrails.command.evaluate_action` judges a command by its **content**,
not just its name: SSRF hosts, `~/.ssh`/`.env`, secret-shaped literals, and
network egress in executed code are inspected before it runs, returning a typed
`allow` / `ask` / `deny` + confidence. It runs at the single tool-dispatch
choke point (`tool_registry.dispatch`) before the human gate. Env-gated by
`CGX_POLICY_MODE` (`off` | `advisory` [default] | `enforce`): advisory enforces
only hard security denies; enforce also blocks soft denies and lets a clear
read-only allow skip the gate. The SSRF hard-block is un-downgradable.

### Security — one shared sensitivity scorer
`cgx.answer.sensitivity.score_paths` rates the files a task touches as
`public` / `standard` / `restricted` (`.env*`, `~/.ssh`, keys, infra state …).
This is the "trust" axis: it feeds the permission policy today, and gates cloud
routing later. Built once, consumed everywhere, so the two can't drift.

### Tools — tiered disclosure
The model sees one-line tool descriptions; it fetches a tool's full argument
schema only once it has selected that tool (`mcp_describe_tool`), and the
harness validates the arguments against the schema before the call
(`mcp/manager.py`). A wrong argument type becomes an actionable error, not a
silent server-side failure.

### Instructions — conditional context
`cgx.context_files.load_directory_notes` loads a directory's `GOTCHAS.md` only
when a task actually touches a file in (or under) that directory, appended at
the **tail** so the always-on `CGX.md` prefix stays byte-identical. Small-window
models stop paying for rules that don't apply to the work in front of them.

### Real local logprobs (opt-in)
A local model's self-reported confidence is uncalibrated, and Ollama exposes no
logprobs. Point CGX at a llama.cpp `llama-server` / vLLM / LM Studio endpoint
(`CGX_JEV_BASE_URL`) and a typed decision carries the **true constrained-choice
logprob** — the calibration input for turning thresholds on. Ollama stays the
default for everything else. See [jev-decision-provider.md](jev-decision-provider.md).

---

## Design principles

- **Deterministic by default.** Every thousands-per-session decision runs as
  plain code over signals CGX already has (RRF ranks, provenance, path globs).
  A model call is added only where it clearly pays for itself, behind a flag.
- **Probability gates stay off until calibrated.** A local probability is a
  ranking hint, never a gate, until a reliability curve is mined from
  `emit_trace` history vs actual outcomes.
- **Prefix-cache safe.** Anything that trims context keeps the stable prompt
  prefix byte-identical and only appends/elides at the tail.
- **Reproducibility preserved.** CGX stays plan-driven and replayable; any
  model-answered decision is flagged and kept out of the determinism-critical
  eval paths (`eval/recovery.py`).

## Local-first framing (and what's deferred)

The paper JEV synthesizes prices routing in cloud-API dollars. On a local-first
tool that inverts: with **one** local model there is nowhere cheaper to route,
and local is already the most private option — so **routing** and
**trust-routing** largely collapse. What survives — context, cache, tools,
permissions, conditional instructions — is *more* valuable locally, because the
binding constraint is the `num_ctx`-clamped context window.

Deferred, by design: model routing (only pays off with multiple local tiers or
cloud opt-in), calibrated probability gates, and an LLM tool-router (only worth
it behind a large registered MCP fleet).

## Enabling the opt-in pieces

```sh
# Programmable permissions: advisory (default) enforces hard security denies;
# enforce also blocks soft denies and auto-allows clear read-only calls.
export CGX_POLICY_MODE=enforce

# Real constrained-choice logprobs from a local llama-server / vLLM endpoint.
export CGX_JEV_BASE_URL="http://127.0.0.1:8080"   # see docs/jev-decision-provider.md
```

## Where it lives

| Piece | Module |
|---|---|
| Typed decision core | `cgx.answer.jev` |
| Canonical schemas + validation | `cgx.answer.schemas` |
| Visibility ladder | `cgx.answer.context_map`, `cgx.answer.model_caps` |
| num_ctx guard | `cgx.session.context_budget` |
| Programmable permissions | `cgx.guardrails.command`, `cgx.session.tasks.tool_registry` |
| Sensitivity scorer | `cgx.answer.sensitivity` |
| Tiered tool disclosure | `cgx.mcp.manager` |
| Conditional instructions | `cgx.context_files`, `cgx.answer.instructions` |
| Local logprob provider | `cgx.answer.providers` (`OpenAICompatProvider`) |
