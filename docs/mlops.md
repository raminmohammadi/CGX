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

`cgx.redact` masks credential-shaped literals (API keys, bearer tokens,
`key=value` pairs, provider key prefixes) before any text reaches a log, a
trace, a store, or the admin API -- defence-in-depth so a secret echoed into a
prompt/response preview never persists.

---

## User activity

**Subsystem C.** `cgx.activity` persists one `RunRecord` per ask/plan run to
`activity.db`: the provenance keys, the grounding signals already computed by
the answer pipeline (sources / citations / confidence / grounded), and the
token / cost / latency accounting. `record_run` is the best-effort recorder
the web handlers call once a run's `meta` is assembled; stored text passes
through the data-governance policy (PII scrub + preview cap) first. The
Activity page reads:

- `GET /api/activity/runs` -- filterable list of runs.
- `GET /api/activity/runs/{run_id}` -- one run joined to its feedback + alerts.
- `GET /api/activity/summary` -- roll-up counts / cost / satisfaction.

---

## Admin explorer

**Subsystem D.** `cgx.webui.routes.admin` is a read-only operator surface that
stitches the observability stores into one admin view. Every line is passed
through `cgx.redact.redact_mapping` before it leaves the process.

- `GET /api/admin/logs` -- the JSONL function-call trace log (trace explorer).
- `GET /api/admin/metrics` -- a structured snapshot of the metrics registry.
- `GET /api/admin/overview` -- an audit-lite health view folding activity (C),
  alerts (G) and feedback (H) into one payload.

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
| `activity.db` | C | One `RunRecord` per ask/plan run. |
| `monitor.db` | G / K | AIOps + guardrail `Alert` records. |
| `feedback.db` | H | Thumbs up/down + comments. |
| `usage.db` | I | Per-owner / per-day token + cost meter. |

The pre-existing `sessions.db` (agent state) and `tasks.db` (task registry)
are covered in [architecture.md](architecture.md).
