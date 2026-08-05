# MLOps and Production

Beyond the request pipeline, CGX ships a production **MLOps layer** for
running the service as a real deployment: observability, evaluation,
monitoring, governance, and packaging. It follows the same local-first,
stdlib-first, zero-config philosophy as the rest of CGX — every store is
SQLite (WAL, `$CGX_CONFIG_DIR`-aware), metrics are collected in-process, and
every recorder is best-effort so an observability failure can never break an
ask/plan/agent request.

The exhaustive operator guide (endpoints, env vars, store layout) is
[`docs/mlops.md`](https://github.com/raminmohammadi/Averix/blob/main/docs/mlops.md);
deployment specifics live in
[`deploy/README.md`](https://github.com/raminmohammadi/Averix/blob/main/deploy/README.md).

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
| **Admin explorer** | `cgx.webui.routes.admin` | `GET /api/admin/{logs,metrics,overview}` (server-side redacted) |
| **Evaluation** | `cgx.eval`, `evals/` | `python -m cgx.eval`; CI quality gate |
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

---

## Monitoring, feedback, and evaluation

`cgx.monitor` turns computed signals into persisted `Alert`s — groundedness,
retrieval drift, cost anomalies, repair-loop health — tunable with `CGX_MON_*`.
Down-votes captured by `cgx.feedback` flow through the **flywheel** into
candidate rows for the offline **eval** golden sets under `evals/`, which the
CI job gates on so retrieval / codegen quality can't silently regress.

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
[`deploy/`](https://github.com/raminmohammadi/Averix/blob/main/deploy/README.md)
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

- [`docs/mlops.md`](https://github.com/raminmohammadi/Averix/blob/main/docs/mlops.md) — the full operator reference.
- [`deploy/README.md`](https://github.com/raminmohammadi/Averix/blob/main/deploy/README.md) — build / run / scale.
- **[[Configuration and Tuning]]** — the MLOps environment variables.
- **[[Privacy and Security]]** — redaction, secrets, and egress boundaries.
- **[[Architecture]]** — where these modules sit in the codebase.
