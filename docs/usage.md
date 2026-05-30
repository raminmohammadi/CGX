# Usage

## 0. Install

Averix splits dependencies into a **core** layer (always required) and an
**ML extras** layer (only needed for local embedding models and the
optional cross-encoder reranker).

```bash
# Core only — small, no torch:
pip install -r requirements.txt
pip install -e ".[codegen]"

# Add ML extras for local Jina embeddings / reranker:
pip install -r requirements-ml.txt
```

The UI and the Ollama / remote-LLM answer paths run on the core install;
`torch`, `transformers`, and `sentence_transformers` are imported lazily
only when an embedding or reranker step is actually invoked.

## 1. Pick a provider

### Ollama (default, local)

```bash
ollama serve   # in another terminal
ollama pull qwen2.5-coder:3b
```

The default base URL is `http://localhost:11434`. The Setup tab's
**Ping Ollama** button exercises `GET /api/tags` and prints the result.

### OpenAI-compatible (cloud or self-hosted)

Provide a base URL pointing at `/v1/chat/completions`-compatible
endpoint (OpenAI, Groq, Together, DeepSeek, vLLM, etc.) and an API key.
Save it as a Profile to keep the key in your OS keyring.

## 2. Index a project

```bash
averix index --project-root /path/to/repo --out-dir /tmp/averix_index
```

or use the **Index** tab in the UI, which also accepts a `.zip` upload.

Artefacts written under `out_dir`:

```
indices/                   # FAISS files + per-view metadata (.npy + .json)
records.jsonl              # canonical records (one per chunk)
chunks.jsonl               # raw parser chunks
graph.json                 # NetworkX node-link graph
emb_cache_intent.npz       # content-addressed embedding cache (intent view)
emb_cache_impl.npz         # content-addressed embedding cache (impl view)
```

### Parallel two-view build and GPU detection

`run_index_auto()` builds the intent-view and impl-view FAISS indices
concurrently using a `ThreadPoolExecutor`, roughly halving indexing time
on multi-core machines. `build_embeddings()` auto-detects the best
available compute device at runtime (CUDA > MPS > CPU) — no manual
configuration is needed.

The **Index** tab in the UI displays a **Cancel** button while indexing
is in progress; clicking it terminates the SSE stream cleanly.

### Incremental re-indexing

The two `emb_cache_*.npz` files make re-indexing cheap. Each file
stores `{sha256(corpus_text): np.ndarray}` pairs; on the next
`run_index_auto` call, unchanged chunks reuse their cached vectors and
only modified chunks reach the embedder.

```python
from cgx.pipeline.auto import run_index_auto
result = run_index_auto(project_root=".", out_dir="/tmp/averix_index")
print(result["incremental"])       # True
print(result["embedding_cache"])
# {'intent': {'hits': 412, 'misses': 5,  'dim': 768},
#  'impl':   {'hits': 410, 'misses': 7,  'dim': 768}}
```

The cache is invalidated automatically when the embedding `model_name`,
`dim`, or `normalize` flag changes — there is no risk of serving stale
vectors against a different model.

Force a clean rebuild:

```python
run_index_auto(project_root=".", out_dir="/tmp/averix_index",
               incremental=False)
```

Implementation lives in `src/cgx/embeddings/cache.py`.

## 3. Ask a question

```bash
averix query --index-dir /tmp/averix_index/indices \
             --records  /tmp/averix_index/records.jsonl \
             --query    "What does parse_codebase do?"
```

Or open the **Ask** tab. The streaming panel shows the model's
reasoning sketch; the structured grounded answer (with citations and a
debug payload) appears below it.

A **Stop** button is visible while the stream is in progress; clicking
it closes the SSE connection and cancels the running task. Switching to
another tab mid-stream does **not** lose the answer — the connection
keeps streaming in the background and the accumulated messages are
restored when you return to the Ask tab.

## 4. Generate a change plan

The **Plan** tab accepts a free-form task description. Recommended
options:

- ✅ **Validate diffs** — parses + dry-applies fenced diffs and runs
  `ast.parse` on each affected Python file.
- ✅ **Run impacted tests** — copies the project into a sandbox,
  materialises the diffs, and runs pytest scoped to the impacted files.

Failures feed a one-shot retry. The full report is rendered as a
markdown table under the plan and is also available as
`result["codegen_report"]` when called programmatically.

A **Cancel** button is shown while planning is in progress; clicking it
closes the SSE connection and terminates the backend stream. Tab
switching is non-destructive — the plan output accumulated so far is
preserved in session state and displayed when you return.

## 5. Tune retrieval (optional)

The hybrid retriever fuses semantic + lexical + graph signals via
Reciprocal Rank Fusion. The post-fusion rerank stage is controlled by
`HybridConfig` in `cgx.retrieval.orchestrator`:

```python
from cgx.retrieval.orchestrator import HybridConfig
cfg = HybridConfig(
    # graph-aware reranking — pulls in neighbors of top hits.
    graph_depth=1,
    graph_bonus=0.2,        # set 0.0 to ignore graph-only neighbors
    # symbol-match bonus — rewards files/funcs whose name appears
    # verbatim in the query.
    symbol_boost=0.5,       # 0.0 disables
    # optional cross-encoder rerank over the top-N RRF hits.
    enable_reranker=True,
    reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    reranker_top_n=30,
    reranker_weight=1.0,    # 1.0 = pure cross-encoder, 0.0 = pure RRF
)
```

`enable_reranker=True` lazy-loads `sentence_transformers`. If the ML
extras aren't installed, the call silently falls back to the RRF order
— no crash, just no rerank. Install `requirements-ml.txt` to opt in.

Each chunk's `provenance` dict in the search result records which signals
fired (`semantic_intent`, `semantic_impl`, `lexical`, `graph_depth`,
`symbol_match`, `reranker_score`), so the **Ask** tab's "thought process"
panel shows exactly why a chunk ranked where it did.

## 6. Multi-agent orchestration (Agent tab)

For requests that don't fit into a single Ask or Plan round-trip, the
**🤖 Agent** tab runs a Planner → Tracker → Judge loop:

A picture-first overview lives in [flowcharts.md](flowcharts.md) — the
"general user" SVG matches the UI flow described below.

1. **Planner** decomposes the goal into 1–5 ordered `Task`s with a short
   `name`, a `description`, a `kind` (see table below), and plain-English
   `criteria`. The LLM is asked for a strict JSON plan; on failure the
   planner falls back to a deterministic single-task plan. A kind-policy
   pass applies routing rules in order:
   - *New-project goals* → `[scaffold, apply, verify]` (no index needed).
   - *Verify-only goals* ("do tests pass?") → `[verify]`.
   - *Read-only goals* → any `plan` downgraded to `ask`.
   - *Code-change goals* → `apply` + `verify` appended after `plan`.

2. **Tracker** walks the plan task by task, dispatching each `kind`
   to a capability on a worker thread and streaming `AgentEvent`s:
   `plan`, `task_start`, `task_progress`, `task_done`, `task_failed`,
   `task_skipped`, `judge`, `summary`. `task_progress` fires every 2 s
   with `{task_id, name, kind, elapsed}`.

3. **Judge** validates each completed task: structural short-circuits
   first, then optionally an LLM verdict `{verdict, confidence, rationale}`.

| Kind | What it does | Index required? |
|------|-------------|-----------------|
| `ask` | Answer a question grounded in the indexed code | Yes |
| `plan` | Produce a unified-diff change plan for an existing codebase | Yes |
| `scaffold` | Generate a **complete new project** from a plain-language idea | **No** |
| `search` | Retrieve relevant code chunks | Yes |
| `summarize` | Condense prior outputs | No |
| `apply` | Write diffs from `plan`/`scaffold` to `project_root` on disk | No |
| `verify` | Run impacted pytest tests | No |

### Generating a new project from scratch

Set **Project Root** to the target directory (it will be created if absent),
then describe the project idea as the goal:

> *"Create a FastAPI todo app with SQLite storage and a full pytest suite"*
> *"Create a React calculator app with input fields and result display"*

The planner will emit `[scaffold, apply, verify]`. The scaffold task calls
`generate_project_scaffold`, which asks the LLM to design and code all
project files. The apply task writes them to `project_root`. The verify
task runs the generated tests.

**Technology detection.** The scaffold prompt contains separate
instruction blocks for frontend projects (React, Vue, Angular, Svelte,
etc.) and Python projects. When you name a frontend framework, the LLM
is instructed to generate component files (`src/App.jsx`, `src/index.js`,
`package.json`) using Create React App or Vite conventions, and to omit
Python files entirely. The Judge enforces this: a React scaffold that
produces Python-only files is hard-failed immediately, triggering a
retry with the corrected technology instructions.

### UI controls

- **Goal** textbox — free-form English.
- **Project Root** — destination directory for `apply` / `scaffold`.
- **Stop on first failure** toggle — halts the loop on `task_failed`.
- **Cancel** button — closes the SSE connection; partial event log preserved.
- **Plan Tasks panel** — vertical rail showing status circle, bold name,
  elapsed-seconds badge, and Judge rationale per task.
- **Live event log** — collapsible JSON stream of every `AgentEvent`.
- **Tab persistence** — SSE stream continues in the background; state
  restored from Zustand on return.

### Programmatic use

```python
from cgx.agents import run_agent
from cgx.answer.providers import OllamaProvider

prov = OllamaProvider(model="qwen2.5-coder:3b")

# Modify an existing codebase (stream=True yields AgentEvent objects).
for event in run_agent(
    goal="Add docstrings to every public function in cgx.parser",
    provider=prov,
    index_dir="/tmp/averix_index/indices",
    records_path="/tmp/averix_index/records.jsonl",
    project_root=".",
    stop_on_fail=True,
    stream=True,
):
    print(event.type, event.payload)

# Generate a brand-new project — no index required.
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
    provider=prov,
    index_dir="/tmp/averix_index/indices",
    records_path="/tmp/averix_index/records.jsonl",
)
for task in plan.tasks:
    print(task.kind, task.status, task.output)
```

For tests and custom flows, inject your own capability map:

```python
from cgx.agents import run_agent, Planner, Judge

caps = {
    "ask":      lambda q, **_: {"answer_md": "stubbed answer"},
    "plan":     lambda q, **_: {"plan_md": "...", "diffs": []},
    "scaffold": lambda q, **_: {"plan_md": "...", "diffs": []},
    "search":   lambda q, **_: {"hits": []},
    "summarize":lambda prior, **_: {"answer_md": "summary"},
    "apply":    lambda prior, **_: {"applied_files": [], "failed_files": []},
    "verify":   lambda prior, **_: {"ran": False, "skipped_reason": "stub"},
}
plan = run_agent(goal="...", capabilities=caps,
                 planner=Planner(provider=None), judge=Judge(provider=None))
```

## 7. Persistent chat sessions (Ask tab sidebar)

The Ask tab's sidebar holds the local conversation store:

- **➕ New** creates a session, sets it as the active thread, and
  starts an empty history.
- **🗑️ Delete** removes the selected session file.
- Selecting an entry from the dropdown renders prior turns inline.

User and assistant turns are appended automatically as each answer
stream finishes (failed answers — those starting with `ERROR` — are
not persisted). The `meta` blob on each assistant message captures the
detected intent and the cited sources for later inspection.

Storage layout (under `~/.cgx/sessions/`, overridable via the
`CGX_CONFIG_DIR` env var):

```
~/.cgx/sessions/
├── index.json                 # session headers: id, title, message_count, ts
└── <uuid>.jsonl               # one JSON message per line, append-only
```

Programmatic API (`cgx.sessions`, stdlib-only):

```python
from cgx import sessions
m = sessions.create_session(title="refactor parse_codebase")
sessions.append_message(m.id, role="user", content="What does it return?")
sessions.append_message(m.id, role="assistant", content="A tuple of...")
for header in sessions.list_sessions():
    print(header.id, header.title, header.message_count)
sessions.delete_session(m.id)
```

Writes go through `os.replace` so a crash mid-write cannot corrupt
either the index or a thread file.

## 8. Hardware-aware model picker (Hardware tab)

Click **🧠 Detect hardware** to populate the local-model fit table.
The computation is pure-local — it reads the RAM/VRAM detected by
`cgx.answer.ollama_discovery.detect_hardware()` and compares against
the static catalogue in `cgx.answer.hardware_matrix.LOCAL_MODEL_CATALOG`.

Verdict semantics:

| Symbol | Meaning                                                           |
|--------|-------------------------------------------------------------------|
| ✅     | Budget ≥ 1.2× the model's minimum RAM and any GPU has ≥0.75× the recommended VRAM. |
| ⚠️     | Within 20% of the minimum RAM, or GPU VRAM below the recommended threshold.        |
| ❌     | Budget is less than 90% of the model's minimum RAM — won't fit.   |
| ❓     | No RAM / VRAM detected; the verdict is suppressed.                |

The "effective budget" used to compare against `min_ram_gb` is
`max(ram_gb, gpu_vram_gb * 2.0)` when a GPU is present, otherwise just
`ram_gb` (see `_effective_budget_gb` in
`src/cgx/answer/hardware_matrix.py`).

The second table is the editorial **local vs cloud** trade-off across
privacy, marginal cost, quality ceiling, cold and warm latency,
offline use, setup effort, and operational risk. Every value is a
short string and `winner ∈ {local, cloud, tie}` — see
[`docs/hardware_matrix.md`](hardware_matrix.md) for the rationale
behind each row.

The same data is exported as `docs/hardware_matrix.json` for
downstream tooling.

## 9. Rate limiting and retries

Every HTTP-backed provider (Ollama and OpenAI-compatible) goes through
`cgx.answer.ratelimit`, which provides:

- A thread-safe token-bucket limiter — `RateLimiter(rate=…)`. Set
  `rate=0` (or `None` at the profile level) to make it a no-op.
- Exponential-backoff retry with jitter, honouring `Retry-After` when
  the server provides one. Triggers on **HTTP 429** and **5xx**.

Configure per-profile in the **Profiles** tab, or programmatically:

```python
from cgx.answer.profiles import Profile, save_profile
save_profile(Profile(
    name="my-cloud",
    kind="openai-compat",
    model="gpt-4o-mini",
    base_url="https://api.openai.com/v1",
    rate_limit=2.0,      # requests/sec; bucket capacity == rate
    max_retries=4,       # 0 = no retry (default)
))
```

When you load a profile from the UI, the provider is instantiated
with the persisted `rate_limit` / `max_retries`, so the budget is
applied transparently to every subsequent call.

## 10. Anonymous telemetry (opt-in)

`cgx.telemetry` ships an ultra-light startup ping that exists solely
to count active installs (MAU/DAU). It is **off by default** and emits
*only* a random install UUID + the Averix version — no prompts, no
code, no file paths, no model names, no PII.

Enable:

```bash
export CGX_TELEMETRY=1
```

Disable: unset the variable, or set `CGX_TELEMETRY=0`. To rotate the
install id, delete `~/.cgx/install_id` and restart. The full payload
shape lives in `src/cgx/telemetry.py`; review it before opting in.

## 11. VS Code extension

[`extension/`](../extension/) hosts the Gradio UI inside a VS Code
webview. The scaffold ships source-only; build it locally:

```bash
cd extension
npm install
npm run compile        # emits out/extension.js
```

Then open the `extension/` folder in VS Code and press **F5** to
launch an Extension Development Host. Run **Averix: Open UI** from
the command palette. The URL is read from the `averix.ui.url` setting
(default `http://localhost:7860`).

The extension does *not* spawn the server — start it first with
`averix-ui` (or `python app.py`) from the repo root.

To produce a `.vsix` for side-loading:

```bash
npm install -g @vscode/vsce
vsce package        # → averix-0.0.1.vsix
```

## 12. Terminal logging

All operations emit structured log lines to stdout from the moment the
server starts. `setup_logging(INFO)` is called once in `launch.py` and
configures the root logger with a timestamped, module-prefixed formatter.

What each module logs:

| Module                     | Log lines emitted                                            |
|----------------------------|--------------------------------------------------------------|
| Handlers (ask/plan/agent/index) | Request received; SSE stream opened / closed; errors.  |
| `cgx.webui.task_store`     | Task created; status transitions (running → done/cancelled/error). |
| `cgx.agents.tracker`       | `task_start`, `task_done`, `task_fail` for each planned task. |
| `cgx.agents.planner`       | LLM call dispatched, task count returned, fallback activated. |
| SSE bridge (`cgx.webui.sse`) | Stream opened; each event forwarded; cancellation detected. |

Severity levels used are `[INFO]` for normal progress and `[WARNING]`
for recoverable issues (e.g. LLM fallback, missing cancel token). To
increase verbosity set the `CGX_LOG_LEVEL` environment variable, or
call `setup_logging` with the desired level before importing other cgx
modules.

## 13. Task REST API

Every SSE operation creates a task record in `~/.cgx/tasks.db` (via
`cgx.webui.task_store`). The REST API mounted at `/api/tasks` lets you
inspect or cancel tasks programmatically:

| Method   | Path                      | Description                                          |
|----------|---------------------------|------------------------------------------------------|
| `GET`    | `/api/tasks`              | List up to 50 most-recent tasks (newest first).      |
| `GET`    | `/api/tasks/{id}`         | Retrieve a single task record (status, kind, goal).  |
| `GET`    | `/api/tasks/{id}/events`  | Full ordered event log — use this for tab replay.    |
| `DELETE` | `/api/tasks/{id}`         | Cancel a running task; no-op if already completed.   |

Example — cancel a running task:

```bash
curl -X DELETE http://localhost:7860/api/tasks/<task-id>
```

Example — replay the event log after switching tabs:

```bash
curl http://localhost:7860/api/tasks/<task-id>/events | jq '.[].event_type'
```

The in-UI **Cancel / Stop** buttons on each tab call
`DELETE /api/tasks/{id}` under the hood.

## 14. Safety defaults

- **Plan tab** — Averix never writes to your project directory during
  plan generation. The "Run impacted tests" sandbox uses a temporary
  copy; disk writes only happen when you explicitly click **Apply**.
- **Agent tab (apply / scaffold tasks)** — the `apply` task *does*
  write diffs to `project_root`. A timestamped backup of every
  original file is created under `<project_root>/.averix-backups/`
  before any file is overwritten, so you can roll back with
  `apply_diffs_to_disk`'s companion `rollback_from_backup()` helper.
- Embedder specs (`module:attr`) execute Python on import — only use
  modules you trust.
- API keys live in your OS keyring (or `~/.cgx/secrets.json` with
  `0600` permissions); they are never echoed back through the UI or
  LLM transcripts.
- Session files live under `~/.cgx/sessions/` and inherit the user's
  umask. Once a profile has been saved (or the profile store has been
  initialised), `cgx.answer.profiles._ensure_dir` chmods `~/.cgx/` to
  `0700` (owner-only). Override the root via `CGX_CONFIG_DIR`.
