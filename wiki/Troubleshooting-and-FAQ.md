# Troubleshooting and FAQ

Quick fixes for the most common issues, then answers to questions that
come up a lot. If something here doesn't resolve it, turn on
[function-call tracing](#how-do-i-debug-a-session-that-fails-in-a-weird-place)
and re-run the failing step.

---

## Common issues

### `cgx: command not found`
The console script installs with the package. Reinstall in editable mode
and make sure your virtualenv is active:

```bash
pip install -e ".[dev,codegen]"
```

### Ollama connection errors on Ask / Plan / Agent
CGX defaults to Ollama at `http://localhost:11434`. Start it and pull a
model, then use the **Ping** button (Setup tab) or `cgx status` to
verify:

```bash
ollama serve            # in another terminal
ollama pull qwen2.5-coder:3b
cgx status --provider ollama
```

### `ModuleNotFoundError` for `faiss` / `torch` / `sentence_transformers`
The core install is deliberately light. FAISS-backed indexing and the
embedding/reranker stack are optional:

```bash
pip install -r requirements.txt        # core (includes FAISS)
pip install -r requirements-ml.txt     # embeddings + reranker
```

Without the ML stack, the reranker silently falls back to RRF order — it
does not error.

### "index not found" from `ask` / `plan` / `status`
These commands auto-discover an index at `<project-root>/.cgx/index`.
Build one first, or point at it explicitly:

```bash
cgx index --project-root . --out-dir .cgx/index
# or: cgx ask "..." --index-dir <dir>/indices --records <dir>/records.jsonl
```

### The web UI shows "frontend bundle not found"
The React SPA needs to be built once into `src/cgx/webui/static/`:

```bash
cd frontend && npm install && npm run build
```

### A cloud provider returns 401 / auth errors
Keys are read from the environment or the keyring-backed profile store,
**never** from the command line. Set `OPENAI_API_KEY` / `GEMINI_API_KEY`
(or save them via a profile) and re-**Ping**. See
**[[Providers and Models]]**.

### Re-indexing seems to re-embed everything
Indexing is incremental by default. A full re-embed happens when the
embedding `model_name`, `dim`, or `normalize` flag changed (the cache is
auto-invalidated), or when you passed `incremental=False`. See
**[[Configuration and Tuning]]**.

---

## FAQ

### Does my code leave my machine?
Not with the default setup. Parsing, embedding, indexing, retrieval, and
Ollama inference are all local. Code only leaves your machine if you
select a **cloud provider** (its prompt includes retrieved snippets) or
enable opt-in telemetry. Full detail: **[[Privacy and Security]]**.

### Is telemetry on?
No — it is **off by default** and only fires when `CGX_TELEMETRY=1` **and**
a collector URL are set. Even then the payload is just a random install
UUID + the CGX version.

### Do I need a GPU?
No. The embedder auto-detects CUDA > MPS > CPU. Use the **Hardware** tab
(or `cgx status`) to see which local models fit your RAM/VRAM. Cloud
providers need no local GPU at all.

### Can I use it fully offline?
Yes, once the embedding model has been downloaded once by Hugging Face.
Stay on the Ollama provider and CGX makes no outbound calls.

### Will the agent change my files without asking?
No. Nothing reaches disk until you approve a checkpoint, every overwrite
is backed up under `<project_root>/.cgx-backups/<run_id>/`, and a run is
reversible via the **Undo** button (`POST /api/rollback`). See
**[[Session Based Agent]]**.

### Is the web server safe to expose on my network?
It has **no built-in authentication**. Keep it on loopback, or put it
behind an auth-enforcing reverse proxy on a trusted network. See
**[[Privacy and Security]]**.

### Where does CGX store its state?
Under `~/.cgx/` (or `$CGX_CONFIG_DIR`): profiles, secrets, sessions,
caches, and backups. FAISS indices and JSONL records live wherever you
pointed `--out-dir`.

### How do I add support for a new framework?
Write a **skill** — a single self-contained folder under `skills/` (or a
private `.py` under `~/.cgx/skills/`), with no agent-layer edits. See
**[[Skills Registry]]**.

### Do I need to declare a swarm build's dependencies myself?
No. Before it builds or tests, the swarm **reconciles each component's
manifest** against its actual source imports: a Python component's third-party
imports are installed and pinned into `requirements.txt`, and a JS/TS
component's bare imports are `npm install`ed into its `package.json` (installs
use `--legacy-peer-deps` so an imperfect peer pin can't abort them). One
general mechanism covers any package; FastAPI/Starlette also pull in `httpx`
for its `TestClient`. A `package.json` left at the repo root while the app is
under `frontend/` is moved next to `index.html` so the Vite build resolves it.
See **[[Swarm Agent]]**.

### My swarm build reported FAILED — did it silently ship broken code?
No. A small local model can occasionally emit a logic bug the bounded repair
loop (raised to **5 rounds**) can't fix. When that happens the session reports
**FAILED honestly rather than passing a red suite** — so a FAILED verdict means
the tests genuinely did not pass, not that CGX gave up quietly. Note that
phantom third-party imports and contract mismatches are **advisory** now: since
reconciliation installs every real import, the real build/test is the
authority, and a truly hallucinated package fails the real install anyway. See
**[[Swarm Agent]]**.

### How do I debug a session that fails in a weird place?
Turn on **function-call tracing**: flip the toggle on the `/settings`
page or export `CGX_TRACE=1` before `cgx serve`, then re-run the failing
step. Curated `trace_enter` / `trace_exit` / `trace_error` records
(router, runner, executors, repair, LLM, retrieval, codegen) land in the
project-local `agent.log` (or `~/.cgx/cgx-trace.log` outside a session).
Only redacted previews are recorded — never full secrets.

### My agent runs but the Ops Activity and Trace tabs stay empty
Agent turns are recorded to `activity.db` when a drive **quiesces** — the
turn finishes with nothing READY, or it pauses on an ASK_USER. Only then
does the Activity tab show the run and does the project appear in the Trace
tab's **Source** dropdown (the dropdown's project list is drawn from
`activity.db`). So:

- Let a turn reach a natural stop; a drive that is still executing tasks
  has not recorded yet.
- The Trace records themselves also need tracing **on** (`CGX_TRACE=1` or
  the settings toggle) — without it the `agent.log` is empty even though
  the project now appears in the dropdown.
- If you select the project but the trace stays empty, its local
  `agent.log` may be gone — a greenfield tree that was re-scaffolded takes
  its `<root>/.cgx/agent.log` along. The reader then falls back to the
  durable **session-stable mirror** at
  `~/.cgx/agent-sessions/<session_id>/agent.log`, so the trace still
  resolves as long as tracing was on when the turn ran.
- Recording is best-effort and never breaks a run, so a telemetry failure
  is logged and swallowed rather than surfaced.

### Why does the Cost tab show $0.00 for my model?
The estimated cost is `0.0` when the model is not in the price table
(`cost_source="unknown"`) — typically a local model. Provide rates via the
`CGX_MODEL_PRICING` env var (a JSON map of USD per 1M tokens, e.g.
`{"my-local-model": {"in": 1.0, "out": 2.0}}`) to get a non-zero figure.
See **[[Configuration and Tuning]]**.

---

## See also

- **[[Installation]]** · **[[Quick Start]]** — get up and running.
- **[[Configuration and Tuning]]** — the full list of knobs.
- **[[Privacy and Security]]** — data-flow and safety details.
- [`docs/usage.md`](https://github.com/raminmohammadi/Averix/blob/main/docs/usage.md) — the exhaustive usage reference.
