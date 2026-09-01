# MLOps and Production

Beyond the request pipeline, CGX ships a production **MLOps layer** for
running the service as a real deployment: observability, evaluation,
monitoring, governance, and packaging. It follows the same local-first,
stdlib-first, zero-config philosophy as the rest of CGX — every store is
SQLite (WAL, `$CGX_CONFIG_DIR`-aware), metrics are collected in-process, and
every recorder is best-effort so an observability failure can never break an
ask/plan/agent request.

This page is the subsystem-level reference. For a field guide to the **`/ops`
web hub** — every tab, stat card, donut, gauge, bar list and button, and the
subsystem each maps to — see **[[Ops and Observability]]**. The exhaustive
operator guide (endpoints, env vars, store layout) is
[`docs/mlops.md`](https://github.com/raminmohammadi/CGX/blob/main/docs/mlops.md);
deployment specifics live in
[`deploy/README.md`](https://github.com/raminmohammadi/CGX/blob/main/deploy/README.md).

---

## The provenance join key

Every ask/plan/agent run mints a **`run_id`** (`cgx.registry`) and stamps it —
with the resolved `model` and `prompt_version` — onto the trace context and
the response `meta`. That single key ties metrics, activity, monitor alerts,
and feedback together, so any run can be followed from its latency to its
grounding signals to the thumbs-down that later became an eval candidate.

---

## The subsystems at a glance

| Area | Modules | Surface |
|------|---------|---------|
| **Observability** | `cgx.metrics`, `cgx.trace`, `cgx.redact` | Prometheus text at `GET /api/metrics`; `@traced` tracer (`CGX_TRACE`) with secret redaction |
| **User activity** | `cgx.activity` (`activity.db`) | `GET /api/activity/{runs,runs/{id},summary}` |
| **Admin explorer** | `cgx.webui.routes.admin` | `GET /api/admin/{logs,metrics,overview}` + `DELETE /api/admin/logs` (server-side redacted) |
| **Evaluation** | `cgx.eval`, `evals/` | `python -m cgx.eval`; CI quality gate (retrieval + codegen + **recovery** golden sets) |
| **Lineage** | `cgx.registry` | prompt fingerprint / `run_id` / index lineage |
| **AIOps monitoring** | `cgx.monitor` (`monitor.db`) | drift/quality/cost `Alert`s at `GET /api/monitor/alerts` |
| **Feedback** | `cgx.feedback` (`feedback.db`) | `GET/POST /api/feedback`, `/api/feedback/stats` |
| **Cost & quota** | `cgx.governance` (`usage.db`), `cgx.usage` | per-owner budgets; `GET /api/usage`, `/api/usage/summary` |
| **Reliability** | `cgx.health` | `GET /healthz` (liveness), `/readyz` (readiness) |
| **Guardrails** | `cgx.guardrails` | injection / secret-output / path-escape checks + LLM kill-switch |
| **Data governance** | `cgx.govdata` | retention, right-to-erasure, PII scan/scrub; `GET/POST /api/govdata/*` |
| **Packaging** | `Dockerfile`, `docker-compose.yml`, `deploy/helm/cgx` | container / Compose / Helm |

---

## Observability

`cgx.metrics` is an always-on, stdlib-only registry (counters, gauges,
histograms) with a Prometheus text exporter scraped at `GET /api/metrics`.
RED-style LLM series (`cgx_llm_calls_total`, `cgx_llm_call_latency_ms`,
`cgx_llm_tokens_total`, cost) are emitted from the LLM tracer regardless of the
trace toggle. `cgx.trace` is the curated `@traced` function-call tracer — off
by default, flipped on with `CGX_TRACE` — routing to the project `agent.log`
in-session or `~/.cgx/cgx-trace.log` otherwise. `cgx.redact` masks
credential-shaped literals before anything reaches a log, trace, store, or the
admin API.

With tracing on, every model call also emits a rich **`llm_call`** record
carrying the **full redacted prompt and response** plus `model` / `latency_ms`
/ token counts and the `run_id` join key — beside the `enter` / `exit` /
`error` span records (categories: `router`, `llm`, `retrieval`, `codegen`,
`pipeline`, `executor`, `repair.*`). The agent loop produces these natively,
and the web-UI **Ask** / **Plan** paths do too: they set `project_root` on the
trace context and wrap their provider with the tracing shim, so a traced
ask/plan writes the same records into that project's `agent.log`. In addition to standard spans, long-running tasks like the AST symbol-level generator (`ast_scaffold`) emit live progress beats under the `ast_fallback` layer directly to the dashboard.

### Trace explorer + delete (Ops hub)

The **Ops hub** (`/ops` → **Trace**) is the UI over the admin read API. A
**source** selector switches between the Global fallback (HTTP / CLI records)
and any project's `agent.log` (the rich LLM / router / codegen / scaffold /
ast_fallback / repair records); records are newest-first, redacted server-side, filterable by
event and category, and each opens to its full prompt/response. The section
also exposes **Delete** (current source) and **Delete all** controls backed by
`DELETE /api/admin/logs`.

Deletion is **hard-limited to trace/log files** — it only ever unlinks files
literally named `cgx-trace.log` / `agent.log` (and their rotation backups),
requires a regular file, and **refuses symlinks**, so a caller-supplied
`project_root` can never be used to delete anything else on the machine. The
`scope=all` sweep enumerates project roots from the activity store, never from
the request. See [[Privacy and Security]] for the full threat model.

---

## Monitoring, feedback, and evaluation

`cgx.monitor` turns computed signals into persisted `Alert`s — groundedness,
retrieval drift, cost anomalies, repair-loop health — tunable with `CGX_MON_*`.
Down-votes captured by `cgx.feedback` flow through the **flywheel** into
candidate rows for the offline **eval** golden sets under `evals/`, which the
CI job gates on so retrieval / codegen quality can't silently regress.

The gate also carries a **recovery** section (`cgx.eval.recovery`,
`recovery_golden.jsonl`): a provider-free regression guard for the recovery
ladder. Each case pins a real gate failure and is resolved through the *same*
deterministic decision surface the router uses (`cgx.session.repair.classify`
plus its traceback / build-error extractors), so a change that makes a
scoped-fixable failure fall back to a whole-tree regenerate flips the resolved
action and fails the build. Because it is LLM-free it doubles as the
**degradation floor** for the `DIAGNOSE` rung and enforces two guardrails:
`never_worse_rate` (no resolved action — including the provider-outage
`escalate` fallback — costs more than the old whole-tree ladder) and
`determinism_ok` (re-resolving the corpus is byte-identical, so the router
stays pure). Both are floored at `1.0`. See **[[Session Based Agent]]** for the
ladder itself.

---

## Governance and safety

`cgx.usage` does truthful token + cost accounting (override prices with
`CGX_MODEL_PRICING`), and `cgx.governance` enforces per-owner day budgets with
soft-warn then hard-stop at the provider choke-point (`CGX_BUDGET_*`).
`cgx.guardrails` adds prompt-injection heuristics (direct and indirect),
secret-in-output / path-escape checks, and an operator kill-switch
(`CGX_LLM_DISABLED`); findings are advisory and surfaced in `meta`.
`cgx.govdata` layers retention TTL (`CGX_RETENTION_DAYS`), right-to-erasure,
and a PII scan/scrub pass (`CGX_SCRUB_PII`, `CGX_STORE_FULL_TEXT`) over the
observation stores. Tuning details are on **[[Configuration and Tuning]]**.

---

## Reliability and deployment

`cgx.health` backs Kubernetes-style probes: `GET /healthz` (liveness, touches
nothing external) and `GET /readyz` (readiness — config dir writable, SQLite
usable; provider/index are reported but do not gate). The
[`deploy/`](https://github.com/raminmohammadi/CGX/blob/main/deploy/README.md)
tree packages CGX as a multi-stage Docker image, a Compose stack (CGX +
Prometheus + Grafana with provisioned SLO rules and a starter dashboard), and a
Helm chart with the probes, a PVC for the config volume, and an optional
`ServiceMonitor`.

> **Scaling caveat.** Every store is SQLite (single-writer) on the config
> volume, so the default chart pins observation data to one pod via a
> `ReadWriteOnce` PVC. Multi-replica writes need an external store — see the
> deploy guide.

---

## See also

- **[[Ops and Observability]]** — the `/ops` hub, tab by tab.
- [`docs/mlops.md`](https://github.com/raminmohammadi/CGX/blob/main/docs/mlops.md) — the full operator reference.
- [`deploy/README.md`](https://github.com/raminmohammadi/CGX/blob/main/deploy/README.md) — build / run / scale.
- **[[Configuration and Tuning]]** — the MLOps environment variables.
- **[[Privacy and Security]]** — redaction, secrets, and egress boundaries.
- **[[Architecture]]** — where these modules sit in the codebase.
