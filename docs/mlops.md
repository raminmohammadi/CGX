# MLOps & production operations

CGX ships with a layer of MLOps subsystems that make a local-first RAG /
code-generation tool observable, governable, and deployable without giving up
the zero-config, stdlib-first philosophy of the rest of the project. Every
store is SQLite (WAL, `$CGX_CONFIG_DIR`-aware, path-injection-guarded), every
metric is collected in-process, and every recorder is best-effort -- an
observability failure can never break an ask/plan/agent request.

This document is the operator-facing map of those subsystems. For the request
pipeline itself see [architecture.md](architecture.md); for containers and
Kubernetes see [../deploy/README.md](../deploy/README.md).

**Contents:** [Provenance join key](#provenance-join-key) ·
[Observability](#observability-metrics--tracing) ·
[User activity](#user-activity) · [Admin explorer](#admin-explorer) ·
[Evaluation](#evaluation) · [Lineage & registry](#lineage--registry) ·
[AIOps monitoring](#aiops-monitoring) · [Feedback loop](#feedback-loop) ·
[Cost & quota governance](#cost--quota-governance) ·
[Reliability & health](#reliability--health) ·
[Guardrails & safety](#guardrails--safety) ·
[Data governance](#data-governance-retention--pii) ·
[Packaging & deployment](#packaging--deployment) ·
[Env var reference](#environment-variable-reference) ·
[Stores](#persistent-stores)

---

## Provenance join key

Every ask/plan/agent run mints a **`run_id`** (`cgx.registry.new_run_id`) and
stamps it -- together with the resolved `model` and the active
`prompt_version` -- onto the trace context and into the response `meta`. That
single key is the join used across the whole MLOps layer: the LLM tracer, the
activity store, monitor alerts, and user feedback all key on the same
`run_id`, so a single run can be followed from its metrics to its grounding
signals to the thumbs-down that later turned it into an eval candidate.

---

## Observability (metrics & tracing)

**Subsystem B.** `cgx.metrics` is a stdlib-only, always-on metrics registry
with a Prometheus text exporter (`version=0.0.4`) rendered by
`render_prometheus` and scraped at **`GET /api/metrics`**. It supports
counters, gauges, and histograms; labels are keyword args keyed by their
sorted `(name, value)` tuple. RED-style LLM metrics
(`cgx_llm_calls_total`, `cgx_llm_call_latency_ms`, `cgx_llm_tokens_total`,
cost) are emitted from `cgx.session.llm_trace` independently of the trace
toggle, alongside monitor (`cgx_monitor_alerts_total`) and guardrail
(`cgx_guardrail_events_total`) series.

`cgx.trace` is the curated `@traced` function-call tracer over the router,
runner, executors, repair helpers, and the LLM / retrieval / codegen entry
points. It is **off by default** (the hot path is a single bool check) and
toggled by `CGX_TRACE` (env pin) or the settings endpoint. In-session records
route to the project-local `agent.log`; outside a session they fall through to
`~/.cgx/cgx-trace.log`. Set `CGX_OTEL=1` to also emit spans over OpenTelemetry
when the SDK is installed.

Each `@traced` entry point emits an `enter` record (`{category, fn}`, plus a
bounded `args` repr when `args=True`), then either an `exit` record with
`elapsed_ms` or a `trace_error` record with `error_type` + a truncated message.
Categories group the entry points -- `router`, `llm`, `retrieval`, `codegen`,
`pipeline`, `executor`, and `repair.*`. On top of the span records,
`cgx.session.llm_trace` emits one **`llm_call`** record per model call carrying
the full provenance and, crucially, the **full redacted prompt and response**
(`prompt_full` / `response_full`, alongside `prompt_preview` /
`response_preview`, `model`, `latency_ms`, `streamed`, char counts, `sampling`,
and the `fact_id` / `run_id` join keys). This is what turns the trace explorer
from a request log into a full record of every LLM interaction.

The **agent loop** produces these `llm_call` records natively (the session
runner wraps its provider with `TracingProvider`). The web-UI **Ask** and
**Plan** paths do the same: `stream_ask` / `stream_plan` set the `run_id` and
`project_root` on the trace context and wrap their provider with the tracing
shim (`_traced_provider`), so a traced ask/plan writes the same rich `llm_call`
records into that project's `agent.log`. With tracing on, every LLM interaction
across agent, ask, and plan is therefore reviewable end to end.

`cgx.redact` masks credential-shaped literals (API keys, bearer tokens,
`key=value` pairs, provider key prefixes) before any text reaches a log, a
trace, a store, or the admin API -- defence-in-depth so a secret echoed into a
prompt/response preview never persists.

---

## User activity

**Subsystem C.** `cgx.activity` persists one `RunRecord` per ask/plan run
**and one per agent turn** to `activity.db`: the provenance keys, the grounding
signals already computed by the answer pipeline (sources / citations /
confidence / grounded), and the token / cost / latency accounting. `record_run`
is the best-effort recorder the web handlers call once a run's `meta` is
assembled; stored text passes through the data-governance policy (PII scrub +
preview cap) first. Agent turns take a parallel path: when a drive quiesces
(nothing READY / paused on an ASK_USER), `record_agent_turn` aggregates that
turn's freshly-drained `LLM_CALL` facts into a `kind="agent"` record keyed on
the session's `project_root` -- wired into both the web drain
(`webui.routes.agent_session._drain_ready`) and the CLI drive
(`cli.tui.ops._drive_session`). Besides surfacing agent runs on the Activity
page, this registers the session's project root on the trace explorer's
project allow-list (`admin._known_project_roots`, derived from `activity.db`),
which is what lets the Trace tab's **Source** dropdown offer that project's
`agent.log`. The Activity page reads:

- `GET /api/activity/runs` -- filterable list of runs.
- `GET /api/activity/runs/{run_id}` -- one run joined to its feedback + alerts.
- `GET /api/activity/summary` -- roll-up counts / cost / satisfaction.

---

## Admin explorer

**Subsystem D.** `cgx.webui.routes.admin` is a read-only operator surface that
stitches the observability stores into one admin view. Every line is passed
through `cgx.redact.redact_mapping` before it leaves the process.

- `GET /api/admin/logs` -- the JSONL function-call trace log (trace explorer),
  newest first, redacted, with `event` / `since` / `limit` filters. The
  **source** is chosen by the optional `project_root` query param: omit it to
  read the global fallback (`~/.cgx/cgx-trace.log`, i.e. HTTP / CLI records),
  or pass a project root to read that project's `<root>/.cgx/agent.log` with
  the rich `llm_call` (full prompt + response), router, executor, codegen,
  scaffold, and repair records.
- `GET /api/admin/metrics` -- a structured snapshot of the metrics registry.
- `GET /api/admin/overview` -- an audit-lite health view folding activity (C),
  alerts (G) and feedback (H) into one payload.
- `DELETE /api/admin/logs` -- purge trace/log files. Three modes, in
  precedence order: `?scope=all` clears the global fallback **and** every
  project `agent.log` known to the activity store; `?project_root=<path>`
  clears just that project's `agent.log`; with neither, it clears just the
  global fallback. Returns `{deleted, scope, targets}`.

**Deletion security.** The delete path is deliberately narrow so a
caller-supplied `project_root` can never be leveraged to remove anything but a
CGX trace log. Deletion is delegated to `cgx.trace.delete_fallback_trace_log`
and `cgx.session.agent_log.delete_project_trace_log`, which only ever unlink a
file whose basename is literally `cgx-trace.log` / `agent.log` (plus the
numbered rotation backups) -- the filename is a compile-time constant, so no
`../` traversal can escape to an arbitrary path. Each candidate must be an
existing **regular file** (`lstat` + `S_ISREG`); **symlinks are refused**
outright, so a planted `agent.log -> /etc/shadow` cannot redirect the unlink.
The `scope=all` sweep enumerates its project roots from the activity store
(trusted internal data), never from the request. The live rotating file handle
is closed before unlink and re-opened lazily on the next write.

The **Ops hub** (`/ops` → **Trace**) is the UI over this API: a source selector
(Global vs. any project seen in recent activity), an `event` filter, a
category breakdown, an "HTTP hidden" toggle, and click-through to a record's
full redacted prompt/response. It also carries **Delete** (current source) and
**Delete all** controls, each behind a confirmation that spells out exactly
what is removed; deletion only ever touches trace/log files.

---

## Evaluation

**Subsystem E.** `cgx.eval` is an offline evaluation harness plus a CI quality
gate. The dependency-free metric primitives (`recall_at_k`, `precision_at_k`,
`reciprocal_rank`, `ndcg_at_k`, `mean`) import eagerly; the retrieval and
codegen harnesses lazy-load the heavier pipeline. Golden sets live under
`evals/` (`retrieval_golden.jsonl`, `codegen_golden.jsonl`, and a small
`sample_repo/`). Run it as a module:

```bash
python -m cgx.eval retrieval --golden evals/retrieval_golden.jsonl
python -m cgx.eval codegen   --golden evals/codegen_golden.jsonl
```

The CI workflow runs the harness against the golden sets and fails the build
when a metric regresses below its threshold, so retrieval / codegen quality is
gated the same way tests are.

---

## Lineage & registry

**Subsystem F.** `cgx.registry` provides three provenance primitives:
`fingerprint` hashes a prompt template to a short, stable content id and
`PromptRegistry` maps stable names to their current fingerprint;
`new_run_id` mints the per-execution join key; and `build_index_lineage`
captures the CGX version, the indexed repo's git revision, and the embedder
identity so a stale or foreign index is detectable after the fact. All probes
are best-effort and never raise.

---

## AIOps monitoring

**Subsystem G.** `cgx.monitor` turns already-computed pipeline signals into
persisted, metric-exported `Alert` records. The `Monitor` façade runs pure
`check_*` functions over each run and writes findings to the `AlertStore`
(`monitor.db`):

- `check_groundedness` -- low citation coverage / confidence on an answer.
- `check_retrieval_drift` -- a drop in retrieval score distribution.
- `check_cost_anomaly` -- a cost spike against a rolling window.
- `check_repair_health` -- excessive autonomous-repair attempts.

Thresholds are env-tunable (`CGX_MON_*`) and alerts surface at
`GET /api/monitor/alerts` and on the admin overview. Guardrail findings
(Subsystem K) are recorded as `guardrail_*` alerts in the same store.

---

## Feedback loop

**Subsystem H.** `cgx.feedback` captures thumbs up/down + comments on
ask/plan results, joined to the provenance keys, in `feedback.db`. The
`flywheel` closes the loop: `export_eval_candidates` drains down-votes into an
idempotent JSONL of candidate rows for the Subsystem E golden sets (path
overridable via `CGX_EVAL_CANDIDATES_PATH`), and `unify_with_lessons` merges
feedback with the cross-session `lessons.jsonl` into one negative-signal view.

- `POST /api/feedback` -- record a rating + optional comment for a `run_id`.
- `GET /api/feedback` -- recent feedback, filterable.
- `GET /api/feedback/stats` -- satisfaction aggregation.

---

## Cost & quota governance

**Subsystem I.** `cgx.usage` does truthful token + cost accounting:
`extract_usage` prefers provider-reported token counts and tags a fallback
estimate as `token_source="estimated"`; `estimate_cost` multiplies by a
per-model price table overridable with `CGX_MODEL_PRICING` (JSON map, USD per
1M tokens), returning `0.0`/`cost_source="unknown"` for unpriced models rather
than a fabricated number.

`cgx.governance` turns that accounting into an enforceable budget. A
`GovernedProvider` (wired via `govern` at the request choke-point) checks each
call against the owner's day ceiling -- soft-warn then hard-stop
(`BudgetExceeded`) -- and meters actual spend into `UsageMeter` (`usage.db`).
`QuotaManager` ties config + meter + metrics together. Budgets are configured
with `CGX_BUDGET_*`; the read API is:

- `GET /api/usage` -- per-owner usage + budget state.
- `GET /api/usage/summary` -- all-owner roll-up for the admin dashboard.

---

## Reliability & health

**Subsystem J.** `cgx.health` backs two Kubernetes-style probes, deliberately
split by cost and meaning:

- `GET /healthz` (**liveness**) -- the process is up and the event loop
  responsive; touches nothing external, so a provider outage or slow volume
  can't trigger a restart loop.
- `GET /readyz` (**readiness**) -- the config dir is writable and the SQLite
  driver + session-DB parent accept a connection. Provider reachability and
  index presence are *reported* but do **not** gate readiness, so read-only
  surfaces stay serviceable when a backend is down.

Probes never echo secrets or raw exception text (only the exception type).
Prometheus SLO recording + alerting rules live in
[`deploy/prometheus/cgx-slo-rules.yml`](../deploy/prometheus/cgx-slo-rules.yml).

---

## Guardrails & safety

**Subsystem K.** `cgx.guardrails` adds three defensive layers around the LLM
call path. **Input** (`injection`) applies prompt-injection heuristics to the
user's question/task and to *retrieved* repo chunks (indirect injection).
**Output** (`output`) flags secret-shaped literals in generated code and diff
targets that escape `project_root`. The **kill-switch** (`assert_llm_enabled`,
`CGX_LLM_DISABLED`) is an operator panic button enforced at the provider
choke-point. Findings are advisory by default: mirrored to metrics + the AIOps
alert store and surfaced in response `meta`, without mutating the prompt or
silently dropping a request. Tunable with `CGX_GUARDRAIL_*`.

---

## Data governance (retention & PII)

**Subsystem M.** `cgx.govdata` layers a data-lifecycle policy over the
observation stores. `GovernanceConfig.from_env()` resolves TTL, full-vs-preview
text, and the PII toggle; the `pii` module scans/scrubs PII (email, card,
IPv4, phone) in a fixed, non-overlapping order, complementing the credential
redaction in `cgx.redact`; and `retention` sweeps expired rows and honours
per-run / per-owner erasure across every store.

- `GET  /api/govdata/policy` -- the resolved policy in force.
- `POST /api/govdata/purge` -- TTL sweep; returns `{store: rows_deleted}`.
- `POST /api/govdata/erase` -- right-to-erasure by `run_id` or `owner`.
- `POST /api/govdata/scan` -- audit a snippet for PII (non-destructive).

Retention can be driven periodically from a `CronJob` (see the deploy guide).

---

## Packaging & deployment

**Subsystem L.** A multi-stage `Dockerfile`, a `docker-compose.yml` bundling
CGX + Prometheus + Grafana, and a Helm chart under `deploy/helm/cgx`. Full
build / run / scale walkthrough -- including the SQLite single-writer caveat on
the shared config volume -- is in [../deploy/README.md](../deploy/README.md).

---

## Environment-variable reference

| Variable | Subsystem | Effect |
|----------|-----------|--------|
| `CGX_CONFIG_DIR` | all | Root for every SQLite store + config (default `~/.cgx`). |
| `CGX_TRACE` | B | Pin the `@traced` tracer on/off (overrides the runtime flag). |
| `CGX_OTEL` | B | Emit OpenTelemetry spans when the SDK is present. |
| `CGX_MON_*` | G | Monitor thresholds (min confidence / citation coverage, drift drop, cost spike ratio, max error rate, max repair attempts). |
| `CGX_EVAL_CANDIDATES_PATH` | H | Override the flywheel eval-candidate JSONL path. |
| `CGX_MODEL_PRICING` | I | JSON `{model: {"in": x, "out": y}}` price overrides (USD / 1M tokens). |
| `CGX_BUDGET_ENABLED` / `CGX_BUDGET_DAILY_TOKENS` / `CGX_BUDGET_DAILY_COST_USD` / `CGX_BUDGET_SOFT_RATIO` / `CGX_BUDGET_OWNERS` | I | Per-owner day budgets + soft-warn ratio. |
| `CGX_GUARDRAIL_ENABLED` / `CGX_GUARDRAIL_SCAN_INPUT` / `CGX_GUARDRAIL_SCAN_OUTPUT` / `CGX_GUARDRAIL_BLOCK_SECRETS` | K | Guardrail layer toggles. |
| `CGX_LLM_DISABLED` | K | Kill-switch: reject every LLM call at the choke-point. |
| `CGX_RETENTION_DAYS` | M | TTL for observation rows (retention sweep). |
| `CGX_SCRUB_PII` / `CGX_STORE_FULL_TEXT` / `CGX_PREVIEW_CAP` | M | PII scrubbing, full-vs-preview text, preview length cap. |

---

## Persistent stores

All live under `$CGX_CONFIG_DIR` (SQLite, WAL):

| File | Subsystem | Contents |
|------|-----------|----------|
| `activity.db` | C | One `RunRecord` per ask/plan run and per agent turn. |
| `monitor.db` | G / K | AIOps + guardrail `Alert` records. |
| `feedback.db` | H | Thumbs up/down + comments. |
| `usage.db` | I | Per-owner / per-day token + cost meter. |

The pre-existing `sessions.db` (agent state) and `tasks.db` (task registry)
are covered in [architecture.md](architecture.md).
