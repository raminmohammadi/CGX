<p align="center">
  <a href="https://github.com/raminmohammadi/CGX/actions/workflows/ci.yml?branch=main"><img src="https://github.com/raminmohammadi/CGX/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI status"></a>
  <a href="https://github.com/raminmohammadi/CGX/releases"><img src="https://img.shields.io/github/v/release/raminmohammadi/CGX?label=RELEASE" alt="GitHub release"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/raminmohammadi/CGX?color=blue" alt="MIT License"></a>
</p>

# CGX -- Code Graph eXecution

**Local-first codebase RAG and self-testing code-generation platform.**

CGX indexes a code repository, retrieves grounded context via a hybrid
engine (semantic + lexical + graph), and asks a local or remote LLM to
answer questions or produce **self-tested** code change plans. It is
model-agnostic and ships with a React/Vite web UI served by a FastAPI
backend that streams progress over Server-Sent Events.

- **Local-first.** Indexing, embedding, retrieval, sessions, and
  telemetry **never leave the machine.** Works fully offline with
  [Ollama](https://ollama.com/).
- **Universal LLM provider.** Ollama (local), OpenAI-compatible
  endpoints, native **Google Gemini**, or any self-hosted server with a
  custom IP, path, and optional auth-bypass -- switchable from the Settings
  tab with a live **Ping** latency check. API keys live in your OS keyring.
- **Hybrid retrieval.** Two-view semantic + BM25 + graph expansion,
  fused with Reciprocal Rank Fusion and an optional cross-encoder
  rerank.
- **Multi-agent orchestration.** A Planner / Tracker / Judge loop
  decomposes complex requests into atomic tasks (`ask`, `plan`,
  `scaffold`, `scaffold_manifest`, `scaffold_file`, `search`, `summarize`,
  `apply`, `verify`, `fill_logic`) and validates each artefact before
  moving on (`cgx.agents`, **🤖 Agent** tab). The planner routes scaffold
  goals through a manifest-then-per-file chain, downgrades expensive
  code-gen tasks to plain Q&A for read-only goals, and the tracker
  streams live `task_progress` heartbeats so the UI never looks frozen on
  long LLM calls. See [docs/flowcharts.md](docs/flowcharts.md) for a visual.
- **New project generation.** Give CGX a plain-language idea
  (e.g. *"create a FastAPI todo app"* or *"create a React calculator
  app"*), set a destination directory as Project Root, and the
  `scaffold_manifest → scaffold_file × N → apply → verify` chain
  generates a complete, working project from scratch -- no existing
  codebase or index required. The `apply` step writes a per-run backup
  mirror under `<project_root>/.cgx-backups/` so the whole run can
  be undone via `POST /api/rollback`.
- **Modular skills registry** (`skills/`). Each supported technology
  lives in its own folder (`skills/react/`, `skills/fastapi/`,
  `skills/nextjs/`, `skills/vue/`, `skills/tailwind/`, `skills/flask/`,
  `skills/django/`, `skills/express/`, `skills/python_cli/`,
  `skills/sqlite/`) and bundles three things: detection from the goal,
  the prompt fragment the LLM sees while generating, and a structural
  validator the Judge runs against the produced diffs. Multi-skill
  goals compose naturally -- *"React UI + FastAPI backend"* activates
  both, so the scaffold prompt carries both layouts and the Judge
  refuses to silently pass an output that only honours one half. Adding
  a new framework is a single-folder change with no agent-layer edits.
  See [docs/usage.md](docs/usage.md#generating-a-new-project-from-scratch)
  for the full table and [docs/architecture.md](docs/architecture.md#skills)
  for the protocol.
- **Persistent chat sessions.** Conversations are saved as JSONL
  threads under `~/.cgx/sessions/`; resume them later from the Ask
  tab's session sidebar.
- **Self-testing code generation.** Diffs are parsed, syntax-checked,
  and optionally run against impacted pytest tests in a sandbox before
  being surfaced. The sandbox now auto-installs missing Python packages
  before running pytest (`cgx.codegen.env_manager`) so a model choosing
  a new library doesn't mask real failures.
- **Symbol table context.** Before generating a change plan, CGX
  injects a compressed `# AVAILABLE CONTEXT` map of every symbol already
  defined in the indexed codebase (`cgx.codegen.symbol_map`), preventing
  local models from re-implementing helpers that already exist.
- **Granular error slicing.** Retry prompts include ±5 lines of source
  context around the first traceback line number rather than a raw
  1 200-character pytest dump, keeping small models focused on the precise
  failure site.
- **Incremental indexing.** A content-addressed embedding cache
  (per-view `.npz` keyed on sha256 of the corpus text) makes
  re-indexing a touched-only-a-few-files repo nearly instant.
- **Hardware-aware model picker.** The Hardware tab reports
  ✅/⚠️/❌ verdicts for ~8 local models against your detected RAM/VRAM
  and shows a local-vs-cloud trade-off table.
- **Client-side rate limiting + 429 retry** on every provider, with
  per-profile budgets persisted alongside the model config.
- **Thought-process panel.** Live streaming of the model's reasoning
  sketch, followed by the final grounded answer.
- **VS Code extension scaffold** (`extension/`) that hosts the
  CGX web UI inside an editor webview.
- **Task registry & cancel.** Every operation is tracked in
  `~/.cgx/tasks.db`; cancel any running task with
  `DELETE /api/tasks/{id}` or the in-UI Cancel button.
- **Cancel button on every tab.** Stop a streaming request mid-flight
  from Ask (Stop), Plan, Agent, or Index (Cancel).
- **Tab persistence.** Switching between tabs mid-task no longer loses
  the running view -- state is held in a session-scoped Zustand store
  (`frontend/src/store/tasks.ts`) and the SSE stream continues in the
  background via `frontend/src/lib/connections.ts`.
- 🖥️ **Terminal observability.** All operations emit structured
  `[INFO]`/`[WARNING]` log lines to stdout from startup
  (`setup_logging(INFO)` in `launch.py`).
- ⚡ **Parallel two-view execution.** FAISS index building
  (`run_index_auto`) and semantic retrieval (`HybridRetriever.search`)
  both run the intent and impl views concurrently via
  `ThreadPoolExecutor`.

---

## Install

CGX has a **small core** and a **separately-installable ML stack**. Pick
the path that matches how you plan to use it.

CGX runs natively on **Linux**, **macOS** (Intel and Apple Silicon),
and **Windows**, on Python 3.10 / 3.11 / 3.12. The only OS-specific
step is venv activation; everything else (CLI, UI, indexing, agent
loop) is identical across platforms. See [Platform notes](#platform-notes)
for Apple Silicon (Metal) and Windows-specific paths.

### Core install (no torch)

Use this if you'll point CGX at an Ollama server or an OpenAI-compatible
endpoint and supply your own embeddings via a BYO embedder callable.

```bash
git clone <your fork>
cd cgx
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install -e ".[codegen]"
```

This installs FAISS, FastAPI/Uvicorn, NetworkX, and the codegen pieces
but skips `torch` / `transformers` / `sentence-transformers` entirely.
Heavy ML modules are imported lazily, so the UI and CLI work out of the
box.

### Full install (with local embeddings)

Use this if you want CGX to load the default Jina embedding model
locally and/or run the optional cross-encoder reranker. Activate the
venv first as shown above, then:

```bash
pip install -r requirements.txt -r requirements-ml.txt
pip install -e ".[codegen]"
# or, equivalently, via extras:
# pip install -e ".[ui,embeddings,faiss,codegen]"
```

Optional extras:

| Extra        | Adds                                              |
|--------------|---------------------------------------------------|
| `ui`         | FastAPI + Uvicorn web UI backend                  |
| `embeddings` | `sentence-transformers`, `transformers`, `torch`  |
| `faiss`      | `faiss-cpu` (large speedup over numpy fallback)   |
| `codegen`    | `unidiff` (stricter diff parsing)                 |
| `keyring`    | OS keyring for API-key storage                    |
| `dev`        | `pytest`, `ruff`, `mypy`                          |

Pull a small local model (recommended default):

```bash
ollama pull qwen2.5-coder:3b
```

### Platform notes

- **Linux** -- no extra steps. NVIDIA users wanting GPU embeddings or
  rerank need a CUDA-enabled `torch` build, **and the wheel's CUDA
  series must match your driver**. The default `pip install torch`
  from PyPI tracks the newest CUDA release, which is frequently ahead
  of installed drivers and silently falls back to CPU at runtime.
  Check `nvidia-smi`'s "CUDA Version" column, then install the
  matching wheel:
  ```bash
  # Driver supports CUDA 12.8 (most 5xx series drivers):
  pip install --index-url https://download.pytorch.org/whl/cu128 torch
  # Older drivers: substitute cu124 / cu121 as appropriate. See
  # https://pytorch.org/get-started/locally/ for the full matrix.
  ```
  Symptom of a mismatch: `torch.cuda.is_available()` is False despite
  `nvidia-smi` reporting the GPU, embeddings run on CPU (~10x slower),
  and the index metadata shows `used_gpu: false`.
- **macOS -- Intel** -- CPU-only by default; same install path as Linux.
- **macOS -- Apple Silicon** -- works natively on arm64. The embedding
  model loads on CPU by default; to use the Metal backend, install the
  ML extras and set `CGX_EMBED_DEVICE=mps` before launching:
  ```bash
  CGX_EMBED_DEVICE=mps cgx-ui
  ```
  Ollama also runs natively on Apple Silicon -- no Rosetta needed.
- **Windows** -- use PowerShell or `cmd.exe`. The venv activates with
  `.venv\Scripts\Activate.ps1` (PowerShell) or `.venv\Scripts\activate.bat`
  (cmd). The CGX config directory resolves to `%USERPROFILE%\.cgx`
  (override with `CGX_CONFIG_DIR`). The `0600` file-permission fallback
  used for `~/.cgx/secrets.json` is a POSIX no-op on NTFS, so install
  the `keyring` extra (`pip install -e ".[keyring]"`) so API keys are
  stored in Windows Credential Manager instead of a plain file.

---

## Quick start

### UI (recommended)

```bash
cgx-ui               # after `pip install -e ".[ui]"`
# or
python app.py
# or via the unified CLI
cgx serve
```

### Binding & remote access

`cgx-ui` (and `python app.py` / `cgx serve`) bind the FastAPI server
to `127.0.0.1:8765` by default, so the UI is only reachable from the
same host. Override with `--host` / `--port` flags or the `CGX_HOST` /
`CGX_PORT` environment variables:

```bash
cgx-ui --host 0.0.0.0 --port 8765
# or
CGX_HOST=0.0.0.0 CGX_PORT=8765 cgx-ui
```

The server has **no built-in authentication** -- anything that can
reach the bound `host:port` can drive the agent loop, read sessions,
and write to disk under the configured Project Root. Bind to a
non-loopback address only on a trusted LAN/VPN (Tailscale, WireGuard,
…) or behind a reverse proxy that adds auth (Caddy, nginx + basic
auth, oauth2-proxy, …). Do not expose port 8765 directly to the
public internet.

Tabs (left → right):

1. **Setup** -- choose a **Provider Type** (Ollama, OpenAI, Google
   Gemini, or Custom Server), fill in the model and credentials, and click
   **Ping** to verify the connection with a live latency check. Detect
   hardware (RAM + GPU VRAM) and tune sampling parameters. Save named
   profiles; API keys are stored in your OS keyring.
2. **Index** -- point at a project root or upload a `.zip`. Honours
   `.gitignore` and a 1 MB file-size cap; emits `indices/`,
   `records.jsonl`, `chunks.jsonl`, `graph.json` and per-view
   `emb_cache_<view>.npz` for incremental re-indexing. Intent and impl
   views are indexed in parallel. A **Cancel** button is available while
   indexing is in progress.
3. **Ask** -- natural-language question with a streaming "thought
   process" panel and a final grounded answer. Sidebar holds the
   **session list** (➕ New / 🗑️ Delete / dropdown to resume an existing
   thread). A **Stop** button halts the stream mid-flight; switching
   tabs preserves the answer in progress.
4. **Plan** -- request a change plan; optionally tick *Validate diffs*
   and *Run impacted tests* to have CGX self-check its own output
   before returning. The full self-test report renders inline. A
   **Cancel** button is available while planning is in progress; tab
   switching is non-destructive.
5. **Agent** -- give CGX a goal, watch the **Planner → Tracker →
   Judge** loop decompose it into 1–5 atomic tasks, dispatch each task
   to a capability (`ask`, `plan`, `scaffold`, `search`, `summarize`,
   `apply`, `verify`), and judge the artefact against per-task criteria.
   For goals like *"create a new FastAPI project"* the planner emits a
   `scaffold → apply → verify` chain that generates a complete project
   from scratch in the Project Root directory. Live event log,
   task-status table, and DAG view of the plan. A **Cancel** button is
   available while the loop is running; tab switching keeps the agent
   running and state is fully restored on return. The sidebar shows an
   animated spinner next to this tab while a task is active.
6. **Hardware** -- click **Detect hardware** to annotate the local
   model catalogue with ✅/⚠️/❌ fit verdicts against your machine. The
   second table shows the editorial local-vs-cloud trade-off across
   privacy, cost, quality ceiling, latency, offline use, setup effort,
   and operational risk. Pure-offline; no network calls fire from this
   tab.
7. **Profiles** -- save provider configurations for any supported
   provider kind (`ollama`, `openai-compat`, `gemini`, `custom`). Custom
   profiles expose an **Endpoint Path** field and a **Skip auth** toggle
   for private-subnet servers. API keys are persisted in the OS keyring
   when available, otherwise in a `0600`-permissioned file under
   `~/.cgx/`. Optional per-profile `rate_limit` (req/sec) and
   `max_retries` apply automatically to every call made by that profile.

### CLI

```bash
cgx index --project-root /path/to/repo --out-dir /tmp/cgx_index
cgx query --index-dir /tmp/cgx_index/indices \
          --records  /tmp/cgx_index/records.jsonl \
          --query "What does parse_codebase do?"
```

### Python

```python
from cgx.pipeline.auto import run_index_auto, run_query_auto
from cgx.answer.engine import answer_with_llm, generate_code_plan
from cgx.answer.providers import OllamaProvider, GeminiProvider, OpenAICompatProvider

run_index_auto(project_root="./", out_dir="/tmp/cgx_index")

# Local Ollama
prov = OllamaProvider(model="qwen2.5-coder:3b")

# Google Gemini
# prov = GeminiProvider(model="gemini-1.5-flash", api_key="YOUR_KEY")

# Custom self-hosted server (no auth, non-standard path)
# prov = OpenAICompatProvider(
#     model="my-model", base_url="http://100.10.20.10:8080",
#     endpoint_path="/completion", allow_no_auth=True,
# )

ans = answer_with_llm(
    "/tmp/cgx_index/indices",
    "/tmp/cgx_index/records.jsonl",
    "What does parse_codebase do?",
    prov,
)
print(ans["answer_md"])
```

---

## How it works

Three picture-first views of the same system live in
[docs/flowcharts.md](docs/flowcharts.md):

- **For users** ([flow_user.svg](docs/diagrams/flow_user.svg)) -- the
  install → index → ask/plan/agent → grounded-answer journey.
- **For developers** ([flow_developer.svg](docs/diagrams/flow_developer.svg)) --
  the Planner → Tracker → Judge loop, the capability dispatch table,
  and the full SSE event timeline (including `task_progress`).
- **For companies** ([flow_company.svg](docs/diagrams/flow_company.svg)) --
  trust boundaries: what stays on the local machine, where credentials
  live, and the single opt-in egress path to a remote LLM.

---

## Tuning hybrid retrieval

`HybridConfig` (in `cgx.retrieval.orchestrator`) exposes the knobs that
shape post-RRF reranking. The defaults are reasonable, but each signal can
be disabled or amplified independently:

| Field             | Default | Effect                                                    |
|-------------------|---------|-----------------------------------------------------------|
| `graph_bonus`     | `0.2`   | Score bump (RRF-scaled) for chunks reached via the import/call graph. Set to `0.0` to ignore graph-only neighbors. |
| `symbol_boost`    | `0.5`   | RRF-scaled bonus for chunks whose identifier or file path matches a token in the question. |
| `enable_reranker` | `False` | Run an optional cross-encoder over the top-N fused chunks. |
| `reranker_model`  | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Hugging Face model id. |
| `reranker_top_n`  | `30`    | How many head candidates to re-score.                     |
| `reranker_weight` | `1.0`   | Convex blend between cross-encoder and RRF score (`1.0` = CE only). |

The reranker lazy-loads `sentence_transformers` only when
`enable_reranker=True`; if the ML stack isn't installed it silently falls
back to the RRF order. Install it via `requirements-ml.txt` to opt in.

```python
from cgx.retrieval.orchestrator import HybridConfig
cfg = HybridConfig(enable_reranker=True, reranker_top_n=20, graph_bonus=0.3)
```

When `graph_bonus > 0` surfaces neighbors of the top hits, the answer
pipeline automatically switches to a **two-tier "Code Map" prompt**:
direct matches keep their full code bodies, while graph-expanded
neighbors collapse to one-line `name(signature) -- docstring` stubs
tagged `tier=neighbor`. This keeps small local models (3B/7B Ollama,
etc.) from spending their entire context window on structural
references they only need to *know about*. The per-tier budget scales
by the provider's model window -- see
[docs/usage.md § Tiered SOURCES (Code Map)](docs/usage.md#tiered-sources-code-map)
and the architecture doc for the full treatment.

---

## Self-testing code generation

When you tick **Validate diffs** in the Plan tab (or pass `self_test=True`
to `generate_code_plan`), CGX will:

1. Parse fenced ```diff path=...``` blocks from the model output.
2. Dry-apply each diff in memory.
3. Run `ast.parse` on the projected file contents.
4. If **Run impacted tests** is enabled, copy the project to a sandbox,
   materialise the diffs, and run pytest scoped to impacted files.
5. If anything fails, retry once with the concrete failures as feedback.

The full report is attached to the result as `codegen_report` and rendered
under the plan in the UI.

---

## Multi-agent orchestration

For requests that don't fit into a single Ask or Plan round-trip,
CGX ships a Planner → Tracker → Judge loop in `cgx.agents`:

1. The **Planner** decomposes your goal into 1–5 ordered atomic
   `Task`s, each tagged with a short `name`, a `description`, a `kind`
   (`ask`, `plan`, `scaffold`, `search`, `summarize`, `apply`,
   `verify`, `fill_logic`) and plain-English `criteria`. It prefers a strict JSON
   plan from the LLM but falls back to a deterministic single-task
   plan derived from `cgx.answer.intent.detect_intent` when no
   provider is available. A kind-policy pass:
   - Routes *new-project* goals to a `scaffold → apply → verify` chain.
   - Downgrades `plan` → `ask` for read-only goals so informational
     queries don't pay for code-generation work.
   - Appends `apply` + `verify` after the final `plan` or `scaffold`
     task so generated code always reaches disk and gets tested.
2. The **Tracker** is a state machine that walks the plan task by
   task, dispatching each one to the matching capability callable on a
   worker thread. It emits `AgentEvent`s (`plan`, `task_start`,
   `task_progress`, `task_done`, `task_failed`, `task_skipped`,
   `judge`, `summary`) that stream as SSE into the UI. `task_progress`
   ticks every `progress_interval` seconds (default `2.0`) with the
   elapsed running time.
3. The **Judge** validates each completed task against its criteria
   with cheap structural short-circuits (*search* passes when
   `hits > 0`; *plan* hard-fails only when both `plan_md` and `diffs`
   are absent; *scaffold* hard-fails only when no files were produced)
   before optionally asking the LLM for a strict
   `{verdict, confidence, rationale}` JSON.

Use it from the **🤖 Agent** tab, or programmatically:

```python
from cgx.agents import run_agent
from cgx.answer.providers import OllamaProvider

prov = OllamaProvider(model="qwen2.5-coder:3b")

# Modify an existing codebase -- stream=True yields AgentEvent objects.
for event in run_agent(
    goal="Add docstrings to every public function in cgx.parser",
    provider=prov,
    index_dir="/tmp/cgx_index/indices",
    records_path="/tmp/cgx_index/records.jsonl",
    project_root="./",
    stop_on_fail=True,
    stream=True,
):
    print(event.type, event.payload)

# Generate a brand-new project -- no index required.
for event in run_agent(
    goal="Create a FastAPI todo app with SQLite and pytest tests",
    provider=prov,
    project_root="/tmp/my_todo_app",   # destination directory
    stream=True,
):
    print(event.type, event.payload)

# stream=False (default) blocks until done and returns the final Plan.
plan = run_agent(
    goal="Add docstrings to every public function in cgx.parser",
    provider=prov, index_dir="/tmp/cgx_index/indices",
    records_path="/tmp/cgx_index/records.jsonl",
)
for task in plan.tasks:
    print(task.kind, task.status, task.output)
```

The Agent tab renders the same `AgentEvent` stream as a live status
table + DAG (`src/cgx/agents/viz.py`).

---

## Persistent chat sessions

The Ask tab's sidebar manages local conversation history:

- **➕ New** -- creates a session, returns a UUID, and starts an empty
  thread.
- **🗑️ Delete** -- removes the selected session file.
- Selecting a session from the dropdown renders prior turns inline
  and routes new questions through that thread; user + assistant
  turns are appended automatically as the answer stream finishes.

Storage layout (under `~/.cgx/sessions/`, or `$CGX_CONFIG_DIR/sessions/`):

```
~/.cgx/sessions/
├── index.json                 # session headers (id, title, counts, timestamps)
└── <session-uuid>.jsonl       # one append-only message per line
```

Programmatic access:

```python
from cgx import sessions
meta = sessions.create_session(title="refactor parse_codebase")
sessions.append_message(meta.id, role="user", content="What does it return?")
for m in sessions.list_sessions():
    print(m.id, m.title, m.message_count)
```

Sessions are stdlib-only (no extra deps) and written atomically via
`os.replace`.

---

## Incremental indexing

`run_index_auto` is incremental by default. On every re-index it
consults a per-view content-addressed cache that lives next to the
FAISS indices:

```
<out_dir>/
├── indices/...
├── records.jsonl
├── emb_cache_intent.npz       # ← cache, keyed on sha256(corpus_text)
└── emb_cache_impl.npz
```

The cache stores `{sha256(corpus_text): np.ndarray}` pairs. Unchanged
chunks reuse their cached vectors; only modified chunks reach the
embedder. The cache is auto-invalidated when the embedding
`model_name`, `dim`, or `normalize` flag changes -- there is no risk of
serving stale vectors against a different model.

Inspect the hit/miss ratio:

```python
result = run_index_auto(project_root="./", out_dir="/tmp/cgx_index")
print(result["incremental"])         # True
print(result["embedding_cache"])
# {'intent': {'hits': 412, 'misses': 5, 'dim': 768},
#  'impl':   {'hits': 410, 'misses': 7, 'dim': 768}}
```

Disable for a clean rebuild:

```python
run_index_auto(project_root="./", out_dir="/tmp/cgx_index", incremental=False)
```

---

## Hardware-aware model picker

The **📊 Hardware** tab annotates a static catalogue of 8
locally-runnable models against the RAM/VRAM detected by
`cgx.answer.ollama_discovery.detect_hardware()`. Each row reports:

| Column        | Meaning                                                                                        |
|---------------|------------------------------------------------------------------------------------------------|
| `model`       | Ollama tag (e.g. `qwen2.5-coder:3b`, `llama3.1:8b-instruct`).                                  |
| `params_b`    | Approx parameter count in billions.                                                            |
| `min_ram_gb`  | Lower bound for 4-bit quantised inference.                                                     |
| `rec_vram_gb` | VRAM at which throughput is smooth.                                                            |
| `ctx_window`  | Maximum prompt window the model advertises.                                                    |
| `family`      | `coder` or `general`.                                                                          |
| `fit`         | ✅ *fits* / ⚠️ *tight* / ❌ *won't fit* against your detected budget.                          |
| `reason`      | The numeric comparison behind the verdict.                                                     |

The second table shows the editorial local-vs-cloud trade-off across
**privacy, marginal cost, quality ceiling, cold/warm latency,
offline use, setup effort, and operational risk**. Every number is
computed locally -- opening this tab does **not** make any network
call. The same data is exported as
[`docs/hardware_matrix.json`](docs/hardware_matrix.json) for downstream
tooling and documented in
[`docs/hardware_matrix.md`](docs/hardware_matrix.md).

---

## Rate limiting and retries

Every HTTP-backed provider goes through `cgx.answer.ratelimit`, which
adds a thread-safe token-bucket limiter plus exponential-backoff
retry (honouring `Retry-After` when present) on HTTP **429** and
**5xx** responses.

Configure per-profile in the **Profiles** tab (or programmatically):

```python
from cgx.answer.profiles import Profile, save_profile
save_profile(Profile(
    name="my-cloud",
    kind="openai-compat",
    model="gpt-4o-mini",
    base_url="https://api.openai.com/v1",
    rate_limit=2.0,   # 2 requests/sec, bucket capacity = rate
    max_retries=4,    # default is 0 (no retry); 4 ≈ ~30s ceiling
))
```

Setting `rate_limit=None` (the default) makes the limiter a no-op so
existing call sites keep their pre-feature behaviour.

---

## VS Code extension scaffold

[`extension/`](extension/) is a minimal TypeScript extension that hosts
the running CGX web UI inside a VS Code webview panel. It is **not**
packaged into a `.vsix` from the repo -- build it locally:

```bash
cd extension
npm install
npm run compile
# then press F5 in VS Code to launch an Extension Development Host
```

Commands contributed: **CGX: Open UI**, **CGX: Reload UI**.
The server URL is read from the `cgx.ui.url` setting (default
`http://localhost:8765`). The extension does not spawn the server --
start it with `cgx-ui` (or `python app.py`) first.

See [`extension/README.md`](extension/README.md) for the full setup.

---

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for a deeper dive.

---

## Privacy & data flow

CGX is built around **local-first** processing. The following table is
the complete list of network egress paths in the product:

| Activity                          | Network egress? | Where it goes                                     |
|-----------------------------------|-----------------|---------------------------------------------------|
| Parsing, embedding, indexing      | **No**          | All on-device.                                    |
| Hybrid retrieval / reranking      | **No**          | All on-device.                                    |
| Asking a question / planning code | Yes             | Only the LLM endpoint you configure.              |
| Local LLM (default: Ollama)       | Yes (loopback)  | `http://localhost:11434` -- never leaves your box. |
| OpenAI-compatible providers       | Yes             | The exact base URL / endpoint path you configure. |
| Google Gemini provider            | Yes             | `generativelanguage.googleapis.com` only.         |
| Session history, profiles, cache  | **No**          | `~/.cgx/` (locked-down `0600` files).             |
| Anonymous startup telemetry       | **Opt-in**      | Disabled by default; see below.                   |

### Server access & secrets

- **No authentication on the local API.** The FastAPI server does not
  ship with login, tokens, or CSRF protection -- any process that can
  reach the bound `host:port` can drive the agent loop, read sessions,
  and write to disk under the configured Project Root. This is safe at
  the default `127.0.0.1:8765` loopback bind; do not bind to `0.0.0.0`
  (via `--host`, `CGX_HOST`, or otherwise) without putting an auth-
  enforcing reverse proxy in front. See
  [Binding & remote access](#binding--remote-access).
- **Disk-writing capabilities.** `apply` and `scaffold_file` tasks
  write inside the configured **Project Root**. Every overwrite is
  mirrored under `<project_root>/.cgx-backups/<run_id>/` and the whole
  run can be undone via `POST /api/rollback`. Set the Project Root
  deliberately -- a stray value lets the agent write anywhere the
  launching user can.
- **Secrets at rest.** API keys go to the OS keyring when the
  `keyring` extra is installed: macOS **Keychain**, GNOME
  **Keyring** / KDE **KWallet** on Linux, **Windows Credential
  Manager** on Windows. The fallback is `~/.cgx/secrets.json` with
  `0600` permissions on POSIX. On Windows NTFS the POSIX bits are not
  enforced, so install the `keyring` extra for production use.
- **Config directory hardening.** `~/.cgx/` is chmodded to `0700` on
  POSIX once a profile is saved. Override the location on any OS with
  the `CGX_CONFIG_DIR` environment variable; it resolves to
  `%USERPROFILE%\.cgx` by default on Windows.

### Telemetry

A single, anonymous startup ping is available for measuring active
installs. It is **off by default** and contains *only* a random install
UUID generated on first run and the CGX version -- no prompts, no
code, no file paths, no model names, no PII.

Enable:
```bash
export CGX_TELEMETRY=1
```
Disable: unset the variable, or set `CGX_TELEMETRY=0`. To rotate the
install id, delete `~/.cgx/install_id` and restart.

The exact payload shape and source live in
[`src/cgx/telemetry.py`](src/cgx/telemetry.py).

---

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

The suite covers parser, embeddings cache, hybrid retrieval / rerank,
codegen pipeline, agents (planner / tracker / judge / viz), sessions,
hardware matrix, rate limiter, telemetry, profiles, and an end-to-end
index → query smoke test with a deterministic fake embedder (no model
download, no GPU). Expected count is in the high 90s and grows with
each feature.

CI is configured in [`.github/workflows/ci.yml`](.github/workflows/ci.yml)
as a **two-job matrix**:

- **core** -- runs on Python 3.10 / 3.11 / 3.12 with only
  `requirements.txt`. Asserts the lazy-import path stays clean (no
  hard dependency on `torch`).
- **ml** (optional) -- installs `requirements-ml.txt` too and exercises
  the embedding + reranker stack.

---

## License

MIT.
