# CGX packaging & deployment (Subsystem L)

Container image, a local observability stack, and a Helm chart for running the
CGX web service in production. The server is the FastAPI app in
`cgx.webui.server:app`; it serves the built React SPA same-origin and exposes
Kubernetes-style probes at `/healthz` (liveness) and `/readyz` (readiness) plus
a Prometheus scrape at `/api/metrics`.

## Image

Multi-stage `Dockerfile` at the repo root: stage 1 builds the SPA into
`src/cgx/webui/static`; stage 2 installs the package on `python:3.12-slim`,
runs as a non-root user (uid 10001), and persists state under `/data`
(`CGX_CONFIG_DIR`).

```bash
# core + web UI (default)
docker build -t cgx:latest .

# + ML extras (torch / faiss / sentence-transformers) -- large image
docker build --build-arg CGX_EXTRAS=all -t cgx:ml .

docker run --rm -p 8765:8765 -v cgx-data:/data cgx:latest
# open http://localhost:8765
```

`CGX_EXTRAS` maps to the `pyproject.toml` optional-dependency groups
(`ui`, `all`, ...). Config and all observation SQLite DBs (activity, feedback,
alerts, usage, monitor) live under the `/data` volume.

## Local stack (Compose)

`docker-compose.yml` runs CGX + Prometheus + Grafana:

```bash
docker compose up --build
#   CGX UI      http://localhost:8765
#   Prometheus  http://localhost:9090
#   Grafana     http://localhost:3000  (admin / $GRAFANA_ADMIN_PASSWORD, default "admin")
```

Prometheus (`deploy/prometheus/prometheus.yml`) scrapes `cgx:8765/api/metrics`
and loads the SLO recording + alerting rules from
`deploy/prometheus/cgx-slo-rules.yml`. Grafana is auto-provisioned with the
Prometheus datasource and a starter **CGX Overview** dashboard (RED, LLM
latency/cost, AIOps alerts) from `deploy/grafana/provisioning/`.

> Set a real `GRAFANA_ADMIN_PASSWORD` before exposing Grafana anywhere.

## Kubernetes (Helm)

Chart skeleton in `deploy/helm/cgx`:

```bash
helm install cgx deploy/helm/cgx \
  --set image.repository=myrepo/cgx --set image.tag=0.2.0

# optional surfaces
helm upgrade cgx deploy/helm/cgx \
  --set ingress.enabled=true --set ingress.hosts[0].host=cgx.example.com \
  --set autoscaling.enabled=true \
  --set serviceMonitor.enabled=true      # Prometheus Operator
```

The Deployment wires the liveness/readiness probes to `/healthz` and `/readyz`,
mounts a PVC at `/data` (toggle with `persistence.enabled`), and annotates pods
for Prometheus scraping. Tune resources, probes, ingress, and HPA in
`values.yaml`.

## Scaling & worker notes

- **Web tier** — the request path (ask/plan streaming, retrieval, provider
  calls) is CPU/IO-bound and largely stateless per request. Scale horizontally
  with `replicaCount` / the HPA. Run one uvicorn worker per pod and add pods
  rather than threads so autoscaling and rollout stay clean.
- **State lives on the config volume** — every store (activity, feedback,
  alerts, usage, monitor, sessions) is SQLite under `CGX_CONFIG_DIR`. SQLite is
  single-writer: a `ReadWriteOnce` PVC pins those DBs to one pod, so the default
  chart keeps observation data pod-local. For multi-replica writes, point the
  stores at an external DB / shared service (future work) rather than a shared
  `ReadWriteMany` SQLite file, which will contend under concurrent writers.
- **Retention** — set `CGX_RETENTION_DAYS` (Subsystem M) and run the
  `purge_expired` sweep (via `POST /api/govdata/purge`) from a `CronJob` or the
  admin page so observation DBs stay bounded.
- **Embeddings / ML** — the `all` image bundles torch/faiss and is much larger;
  give those pods more memory and, where available, a GPU. Prefer a warmed
  model cache (`HF_HUB_OFFLINE`) baked into the image or a shared read-only
  volume to avoid cold-start downloads.
- **Long-running agent sessions** — session work can outlive a single HTTP
  request; keep session state on the persistent volume and drain gracefully on
  rollout (respect the probe grace periods) so in-flight runs are not cut off.
