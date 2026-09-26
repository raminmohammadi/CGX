# JEV — the typed decision layer

A coding agent is a loop around a model, and the leverage is **not in the
loop** — it is in *what the loop feeds the model on each turn*. Those per-turn
choices ("which chunks should this prompt show?", "should this command run?",
"which tool, with what arguments?") are usually made implicitly, by an
append-only transcript. **JEV makes them explicit and typed.**

> **In one sentence.** JEV is a small decision layer beside the model
> (`cgx.answer.jev.decide`): the harness hands it a fixed **question** plus the
> current **state**, and it returns a **typed** answer — an enum choice, a
> score, or a bool — with a probability, so the harness can validate it, apply a
> threshold, and branch **deterministically** instead of parsing prose. JEV
> *decides*; the model, tools, and plain code do the work.

CGX was already most of the way here: it assembles context per query through a
hybrid retrieval engine (see **[[How It Works]]**) rather than accumulating a
transcript. JEV formalizes the remaining micro-decisions.

---

## The six decision points

| Decision point | Question to JEV | Typed answer | In CGX |
|---|---|---|---|
| **Context** | How visible should this chunk be for this query? | hide / short / long / full | ✅ visibility ladder |
| **Cache** | Reuse the running context or rebuild it? | fits / trim (`num_ctx`) | ✅ deterministic guard |
| **Tools** | Which tool fits this intent, with what arguments? | schema on demand + validate | ✅ tiered MCP disclosure |
| **Permissions** | Should this command run? | allow / ask / deny + confidence | ✅ programmable policy |
| **Security** | Which files will this task touch? | public / standard / restricted | ✅ sensitivity scorer |
| **Routing** | Can this subtask leave the main model? | choice + cost estimate | ⏳ deferred |

---

## What ships today

- **Visibility ladder (Context).** Every retrieved chunk is shown `full`,
  `long`, `short`, or hidden — banded by its relevance to *this* query, for
  *every* query. The top hit is always full; weak matches collapse to a one-line
  signature stub or drop out. Reading dominates the token budget, so spending a
  small window on the right chunks is the single biggest lever.
- **num_ctx guard (Cache).** Inside the agent's reasoning loops the system +
  objective prefix stays byte-identical (warm prompt cache) while the *oldest*
  tool observations are elided before they overflow the window — so a large
  tool result can't silently truncate the model's own reasoning.
- **Programmable permissions.** A command is judged by its **content**, not
  just its name: SSRF hosts, `~/.ssh`/`.env`, secret-shaped literals, and
  network egress in executed code are inspected before it runs, returning a
  typed `allow` / `ask` / `deny` + confidence. Complements the human approval
  gate — see **[[Privacy and Security]]**.
- **Tiered tool disclosure.** The model sees one-line tool descriptions; it
  fetches a tool's full argument schema only once it picks that tool, and the
  harness validates arguments before the call. See **[[Providers and
  Models#mcp-tool-servers]]**.
- **Sensitivity scorer (Security).** One shared scorer rates the files a task
  touches `public` / `standard` / `restricted` — the "trust" axis that gates
  permissions today and cloud routing later.
- **Conditional instructions.** A directory's `GOTCHAS.md` loads only when a
  task touches a file in that directory, appended at the tail so the always-on
  `CGX.md` prefix stays stable. Related: **[[Skills Registry]]**.
- **Real local logprobs (opt-in).** Point CGX at a llama.cpp `llama-server` /
  vLLM endpoint and a typed decision carries the true constrained-choice
  logprob instead of a self-reported number.

---

## Design principles

- **Deterministic by default.** Every thousands-per-session decision runs as
  plain code over signals CGX already has (RRF ranks, provenance, path globs).
  A model call is added only where it clearly pays for itself, behind a flag.
- **Probability gates stay off until calibrated.** A local model's confidence
  is uncalibrated (and Ollama exposes no logprobs), so it is a ranking hint —
  never a gate — until a reliability curve is mined from trace history.
- **Reproducibility preserved.** CGX stays plan-driven and replayable; any
  model-answered decision is flagged and kept out of the determinism-critical
  paths.

## Local-first framing

On a local-first tool, **routing** and **trust-routing** largely collapse —
with one local model there is nowhere cheaper to route, and local is already the
most private option. What survives (context, cache, tools, permissions,
conditional instructions) is *more* valuable locally, because the binding
constraint is the `num_ctx`-clamped context window. Model routing is deferred by
design; it only pays off with multiple local tiers or a cloud opt-in.

## Enabling the opt-in pieces

```sh
# Programmable permissions: advisory (default) enforces hard security denies;
# enforce also blocks soft denies and auto-allows clear read-only calls.
export CGX_POLICY_MODE=enforce

# Real constrained-choice logprobs from a local llama-server / vLLM endpoint.
export CGX_JEV_BASE_URL="http://127.0.0.1:8080"
```

See **[[Configuration and Tuning]]** for all environment knobs.

---

## Deep dives

- [`docs/jev.md`](https://github.com/raminmohammadi/CGX/blob/main/docs/jev.md) —
  the full write-up, module map, and status of each decision point.
- [`docs/jev-decision-provider.md`](https://github.com/raminmohammadi/CGX/blob/main/docs/jev-decision-provider.md)
  — running a local `llama-server` for real logprobs, and calibration.
