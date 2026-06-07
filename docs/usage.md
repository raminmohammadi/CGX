# Usage

## 0. Install

CGX splits dependencies into a **core** layer (always required) and an
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

CGX supports four provider kinds, all configurable from the **⚙️ Setup**
tab's **Provider Type** dropdown. A **Ping** button appears on both the
inline config card and the profile edit form — it performs a live
connection test and reports latency or the exact error message.

### Ollama (default, local)

```bash
ollama serve   # in another terminal
ollama pull qwen2.5-coder:3b
```

Set **Provider Type → Ollama (Local)**. The default base URL is
`http://localhost:11434`. Ping exercises `GET /api/tags`.

### OpenAI (cloud)

Set **Provider Type → OpenAI (Cloud)**, enter your `OPENAI_API_KEY`,
and choose a model (`gpt-4o-mini`, `gpt-4o`, etc.). The default base URL
is `https://api.openai.com`. Any OpenAI-compatible endpoint (Groq,
Together, DeepSeek, vLLM, etc.) also works here.

### Google Gemini (cloud)

Set **Provider Type → Google Gemini (Cloud)**, enter your
`GEMINI_API_KEY`, and choose a model (`gemini-1.5-flash`,
`gemini-1.5-pro`, etc.). Ping sends a minimal `generateContent` request
with `maxOutputTokens: 1` to verify the key and model are valid.

Programmatic usage:

```python
from cgx.answer.providers import GeminiProvider
prov = GeminiProvider(model="gemini-1.5-flash", api_key="YOUR_KEY")
# or set GEMINI_API_KEY in the environment and omit api_key
```

### Custom Server (OpenAI-Compatible)

Set **Provider Type → Custom Server (OpenAI-Compatible)** to configure a
self-hosted model endpoint:

- **Host IP/URL** — e.g. `http://100.10.20.10:8080`
- **Endpoint Path** — the exact path suffix, e.g. `/completion` or
  `/v1/chat/completions` (default)
- **Bearer Token** — optional; leave blank and tick **Skip auth** for
  servers on private subnets that do not require authentication

```python
from cgx.answer.providers import OpenAICompatProvider
prov = OpenAICompatProvider(
    model="my-model",
    base_url="http://100.10.20.10:8080",
    endpoint_path="/completion",
    allow_no_auth=True,
)
```

Save any provider configuration as a named **Profile**; the profile
persists `endpoint_path` and `allow_no_auth` alongside the other fields.
API keys are stored in the OS keyring when available, otherwise in
`~/.cgx/secrets.json` with `0600` permissions.

## 2. Index a project

```bash
cgx index --project-root /path/to/repo --out-dir /tmp/cgx_index
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
result = run_index_auto(project_root=".", out_dir="/tmp/cgx_index")
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
run_index_auto(project_root=".", out_dir="/tmp/cgx_index",
               incremental=False)
```

Implementation lives in `src/cgx/embeddings/cache.py`.

## 3. Ask a question

```bash
cgx query --index-dir /tmp/cgx_index/indices \
          --records  /tmp/cgx_index/records.jsonl \
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

### Tiered SOURCES (Code Map)

When the retriever's graph expansion (`graph_depth >= 1`) pulls in
callers, callees, or import-neighbors of the top hits, CGX switches the
prompt-time SOURCES list to a **two-tier "Code Map"** instead of
packing every chunk with a full code body:

- **Primary tier** — chunks that matched directly (semantic / lexical /
  symbol-boosted). Rendered with the focus-windowed code body, exactly
  as before.
- **Neighbor tier** — chunks reached by walking the call/import graph
  one or more hops from a primary seed. Rendered as a one-line stub:
  `[class.]name(signature) — first sentence of docstring`. Each part
  drops silently when the record doesn't carry it. The block is tagged
  `tier=neighbor` in the prompt metadata so the LLM treats it as a
  structural reference rather than the focal body.

This kicks in automatically — there is no flag to flip. If a query's
top results don't trigger graph expansion (e.g. very short queries, or
`graph_bonus=0.0`), the prompt falls back to the legacy single-tier
SOURCES list and behaves bit-identically to earlier CGX versions.

**Why it matters**: when running against a local 3B/7B model with a 16K
or 32K context window, a half-dozen graph-expanded neighbors can blow
the entire prompt budget on code that the model only needs to *know
exists*. Stubs keep that structural context visible (the model can
still cite the chunk and reason about the call shape) while reserving
the bulk of the window for the bodies that actually need to be read.

The per-tier budget scales by the provider's advertised context
window — see `get_context_map_budget` in
`src/cgx/answer/model_caps.py`. The defaults are:

| Window         | Per primary chunk | Per neighbor stub | Max primary | Max neighbors | Total cap |
|----------------|-------------------|-------------------|-------------|---------------|-----------|
| < 16 K         | 900 chars         | 220 chars         | 8           | 12            | 6 000     |
| < 64 K         | 1 400 chars       | 320 chars         | 12          | 24            | 18 000    |
| < 200 K        | 2 200 chars       | 420 chars         | 20          | 40            | 48 000    |
| ≥ 200 K        | 3 500 chars       | 520 chars         | 32          | 60            | 120 000   |

Ordering is deterministic: primary first (in retrieval order), then
neighbors. The total-chars cap is enforced as a hard ceiling — once
the cumulative body length would exceed it, trailing items are
dropped, so citation indices stay stable across reruns.

The architecture doc has the full developer-facing treatment under
[Tiered SLM context (Code Map)](architecture.md#tiered-slm-context-code-map),
including the classifier rule, the `cgx.answer.context_map` public
API, and the engine-level activation gate.

## 6. Multi-agent orchestration (Agent tab)

For requests that don't fit into a single Ask or Plan round-trip, the
**🤖 Agent** tab runs a Planner → Tracker → Judge loop:

A picture-first overview lives in [flowcharts.md](flowcharts.md) — the
"general user" SVG matches the UI flow described below.

1. **Planner** decomposes the goal into 1–6 ordered `Task`s with a short
   `name`, a `description`, a `kind` (see table below), and plain-English
   `criteria`. The LLM is asked for a strict JSON plan; on failure the
   planner falls back to a deterministic single-task plan. A kind-policy
   pass applies routing rules in order:
   - *New-project goals* → `[scaffold_manifest, apply, verify]` (no
     index needed). A goal is recognised as a new-project request when
     any of three signals fires: a scaffold verb + project noun
     (`create a FastAPI app`, `build a calculator`, `bootstrap a
     CLI`), a scaffold verb paired with a framework / language name
     from a curated list (`create a calculator using React`, `build a
     todo app with FastAPI`), or the planner LLM emitted `scaffold`
     task(s) and the goal has no existing-codebase hint. Phrases like
     *"add a React component to our existing app"* are kept on the
     change-goal path. The manifest call returns a layered file plan
     (core logic, UI, tests, config); the Tracker then injects one
     `scaffold_file` task per planned file before `apply` runs, so
     each generation call stays focused on a single output and the UI
     shows per-file progress.
   - *Verify-only goals* ("do tests pass?") → `[verify]`.
   - *Read-only goals* → any `plan` downgraded to `ask`.
   - *Code-change goals* → `apply` + `verify` appended after `plan`;
     any stray `scaffold` tasks are dropped so we modify rather than
     recreate the existing codebase.

2. **Tracker** walks the plan task by task, dispatching each `kind`
   to a capability on a worker thread and streaming `AgentEvent`s:
   `plan`, `task_start`, `task_progress`, `task_done`, `task_failed`,
   `task_skipped`, `judge`, `summary`. `task_progress` fires every 2 s
   with `{task_id, name, kind, elapsed}`.

3. **Judge** validates each completed task: structural short-circuits
   first (per kind), then optionally an LLM verdict
   `{verdict, confidence, rationale}`. For `scaffold` tasks the artifact
   shown to the LLM judge includes the plan summary, the full list of
   generated file paths, and a per-file content preview so verdicts are
   grounded in the actual code rather than a truncated JSON keyset.

| Kind | What it does | Index required? |
|------|-------------|-----------------|
| `ask` | Answer a question grounded in the indexed code | Yes |
| `plan` | Produce a unified-diff change plan for an existing codebase | Yes |
| `scaffold` | One-shot new-project generation (legacy, kept for tests / callers passing a custom capability map) | **No** |
| `scaffold_manifest` | Plan the file layout of a new project; emits an `inject_tasks` list of `scaffold_file` tasks | **No** |
| `scaffold_file` | Generate exactly one file given its path, layer, and the content of prior generated files | **No** |
| `search` | Retrieve relevant code chunks | Yes |
| `summarize` | Condense prior outputs | No |
| `apply` | Write diffs from `plan` / `scaffold_file` outputs to `project_root` on disk (with backup mirror) | No |
| `verify` | Run impacted pytest tests (with pre-flight dep install) | No |
| `fill_logic` | Fill a single empty function body in an existing skeleton file | No |

### Generating a new project from scratch

Set **Project Root** to the target directory (it will be created if absent),
then describe the project idea as the goal:

> *"Create a FastAPI todo app with SQLite storage and a full pytest suite"*
> *"Create a React calculator app with input fields and result display"*

The planner emits `[scaffold_manifest, apply, verify]`. The
`scaffold_manifest` task calls `plan_scaffold_manifest` (a cheap LLM
call that returns just the layered file list, no contents) and the
Tracker injects one `scaffold_file` task per planned file into the
plan immediately after, ordered layer-by-layer so dependency-heavy
files (core types, utilities) are generated before the files that
import them. Each `scaffold_file` task calls
`generate_single_scaffold_file` with the target path, its layer, and
the full content of files already generated by earlier
`scaffold_file` tasks. The `apply` task writes every generated file
to `project_root` (creating a backup mirror under
`<project_root>/.cgx-backups/<run_id>/` for any pre-existing file
it overwrites). The `verify` task runs the generated tests.

**Skills the system knows about.** CGX ships with a registry of
per-technology *skills* (`skills/<name>/`) that activate automatically
when the goal mentions them. Each active skill (a) injects technology-
specific instructions into the scaffold / plan system prompt and
(b) runs a structural validator on the produced diffs so a goal asking
for React can never silently pass a Python-only output.

| Skill        | Activates on (examples)                                  | Enforces in scaffold output |
|--------------|----------------------------------------------------------|-----------------------------|
| `react`      | "react app", "react component", "react ui"               | At least one `.jsx`/`.tsx`/`.js`/`.ts` file; `package.json` with `react` dep |
| `nextjs`     | "next.js", "nextjs app", "app router"                    | `next.config.*` or `pages/` / `app/` directory |
| `vue`        | "vue app", "vue 3", "single-file component"              | At least one `.vue` file |
| `tailwind`   | "tailwind", "tailwind css", "utility-first"              | `tailwind.config.*` present |
| `fastapi`    | "fastapi", "fastapi backend", "python rest api"          | At least one Python file importing `fastapi` |
| `flask`      | "flask", "flask app", "flask api"                        | At least one Python file importing `flask` |
| `django`     | "django", "django app", "django project"                 | `manage.py` and at least one `settings.py` |
| `express`    | "express", "express.js", "node backend"                  | `package.json` with `express` dep |
| `python_cli` | "python cli", "argparse cli", "command line tool"        | Entrypoint with `argparse` / `click` / `typer` |
| `sqlite`     | "sqlite", "sqlite db", "sqlite storage"                  | `sqlite3` / `aiosqlite` import or `.db` reference |

Multi-skill goals compose naturally — *"Create a calculator project
with a React UI and a FastAPI backend"* activates both `react` and
`fastapi`, so the LLM sees both prompt fragments and the Judge runs
both validators against the produced diffs. To extend the registry,
add a folder under `skills/<name>/` with a single `Skill` subclass and
append it to `SKILLS` in `skills/__init__.py`; no agent-layer changes
are required. See [architecture.md](architecture.md#skills) for the
full protocol.

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
    index_dir="/tmp/cgx_index/indices",
    records_path="/tmp/cgx_index/records.jsonl",
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
    index_dir="/tmp/cgx_index/indices",
    records_path="/tmp/cgx_index/records.jsonl",
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

## 7. SLM-grade execution engine

The following features are active by default whenever the Agent loop
runs and require no extra configuration. They are particularly important
when using a local 7B model that would otherwise hallucinate dependencies
or degrade on large files.

### Skeleton-and-Fill (`fill_logic` task kind)

Instead of asking a small model to write a 300-line file in one shot, the
planner can decompose generation into two phases:

1. **Scaffold** — write the file structure (imports, class/function
   signatures, `pass` stubs).
2. **Fill** — one `fill_logic` task per empty function body, prompting
   the model with *"Implement the body for `fn_name`. Return only the
   logic."*

The `fill_logic` capability finds each `pass` / `# TODO` stub in the
skeleton file via a regex match on the function signature, stitches the
returned body into the file at the correct indentation, and runs an
inline `ast.parse` smoke test. The timeline row reports
`fn_name() in file.py · stitched · syntax ok`.

### Dynamic Dependency Management (`cgx.codegen.env_manager`)

Before running pytest in the verify step, the agent scans every
generated `.py` file for `import` statements and cross-references them
against `requirements.txt`. Any new package the model chose but did not
declare is installed with `pip install --quiet` before tests run.

If the tests pass, the newly installed packages are appended to
`requirements.txt` so the dependency becomes permanent.

```
Generated src/auth.py imports: bcrypt
bcrypt not in requirements.txt → pip install bcrypt → tests run → OK
requirements.txt updated: +bcrypt
```

Failures (e.g. a misspelled package name) are logged but never abort
the run — pytest still executes and gives the retry loop a real
`ModuleNotFoundError` to diagnose rather than a false pass.

### Symbol Table Context (`cgx.codegen.symbol_map`)

Before dispatching a `plan` task the agent injects a compressed map of
every symbol already defined in the indexed codebase:

```
# AVAILABLE CONTEXT (Do not redefine these):
File: src/db.py -> get_connection(), close_connection()
File: src/utils.py -> hash_password(str), verify_token(str)
```

This is built from the same `records.jsonl` file the retrieval layer
uses (`build_symbol_map`) and formatted to stay well inside the context
window of a 7B model (capped at 60 files × 20 symbols).

When a retry is triggered by a wrong-arguments call, the retry loop uses
`fetch_symbol_source(records_path, symbol_name)` to inject the exact
source code of the failing function so the model sees the real signature.

### Granular Error Slicing (10-line buffer)

Instead of dumping the full 1 200-character pytest log into the retry
prompt, `_build_fix_goal` extracts exactly ±5 lines around the first
traceback line number:

```
Your code failed in `src/auth.py` at line 42 with `TypeError: expected str`.
Here is the context around the failure:
```python
37.  def login(req):
...
42.     hash = hash_password(req.body)  # <-- ERROR HERE
...
47.
```
Fix this specific issue only.
```

This keeps the retry prompt under ~500 tokens and lets small models
focus on the precise failure site rather than guessing from a wall of
pytest output.

## 8. Persistent chat sessions (Ask tab sidebar)

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

## 9. Hardware-aware model picker (Hardware tab)

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

## 10. Rate limiting and retries

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

## 11. Anonymous telemetry (opt-in)

`cgx.telemetry` ships an ultra-light startup ping that exists solely
to count active installs (MAU/DAU). It is **off by default** and emits
*only* a random install UUID + the CGX version — no prompts, no
code, no file paths, no model names, no PII.

Enable:

```bash
export CGX_TELEMETRY=1
```

Disable: unset the variable, or set `CGX_TELEMETRY=0`. To rotate the
install id, delete `~/.cgx/install_id` and restart. The full payload
shape lives in `src/cgx/telemetry.py`; review it before opting in.

## 12. VS Code extension

[`extension/`](../extension/) hosts the CGX web UI inside a VS Code
webview. The scaffold ships source-only; build it locally:

```bash
cd extension
npm install
npm run compile        # emits out/extension.js
```

Then open the `extension/` folder in VS Code and press **F5** to
launch an Extension Development Host. Run **CGX: Open UI** from
the command palette. The URL is read from the `cgx.ui.url` setting
(default `http://localhost:8765`).

The extension does *not* spawn the server — start it first with
`cgx-ui` (or `python app.py`) from the repo root.

To produce a `.vsix` for side-loading:

```bash
npm install -g @vscode/vsce
vsce package        # → cgx-0.0.1.vsix
```

## 13. Terminal logging

All operations emit structured log lines to stdout from the moment the
server starts. `setup_logging(INFO)` is called once in `launch.py` and
configures the root logger with a timestamped, module-prefixed formatter.

What each module logs:

| Module                     | Log lines emitted                                            |
|----------------------------|--------------------------------------------------------------|
| Handlers (ask/plan/agent/index) | Request received; SSE stream opened / closed; errors.  |
| `cgx.webui.task_store`     | Task created; status transitions (running → done/cancelled/error). |
| `cgx.agents.tracker`       | `task_start`, `task_done`, `task_fail` for each planned task. |
| `cgx.agents.planner`       | LLM call dispatched, task count returned, fallback activated, kind-policy routing branch (`SCAFFOLD` / `VERIFY-ONLY` / `READ-ONLY` / `CHANGE-GOAL`). |
| `cgx.agents.judge`         | Structural verdict per task; LLM judge invocation outcome.   |
| SSE bridge (`cgx.webui.sse`) | Stream opened; each event forwarded; cancellation detected. |

Example lines for a scaffold goal (*"create a calculator using React UI
and python"*):

```
[INFO] cgx.agents.planner: Planner: kind-policy SCAFFOLD path (regex=True llm=False)
[INFO] cgx.agents.planner: Planner: plan ready id=… tasks=4 kinds=['scaffold', 'scaffold', 'apply', 'verify']
[INFO] cgx.agents.tracker: task_start kind=scaffold name='Generate React UI'
```

Severity levels used are `[INFO]` for normal progress and `[WARNING]`
for recoverable issues (e.g. LLM fallback, missing cancel token). To
increase verbosity set the `CGX_LOG_LEVEL` environment variable, or
call `setup_logging` with the desired level before importing other cgx
modules.

## 14. Task REST API

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
curl -X DELETE http://localhost:8765/api/tasks/<task-id>
```

Example — replay the event log after switching tabs:

```bash
curl http://localhost:8765/api/tasks/<task-id>/events | jq '.[].event_type'
```

The in-UI **Cancel / Stop** buttons on each tab call
`DELETE /api/tasks/{id}` under the hood.

### Rollback an `apply` run

When the Agent's `apply` task writes to `project_root`, it first
mirrors every overwritten file into a timestamped directory under
`<project_root>/.cgx-backups/<run_id>/` and returns the path on
the task output as `backup_dir`. To undo the run, POST to
`/api/rollback`:

```bash
curl -X POST http://localhost:8765/api/rollback \
  -H 'Content-Type: application/json' \
  -d '{"project_root": "/path/to/proj",
       "backup_dir": "/path/to/proj/.cgx-backups/<run_id>"}'
```

The response shape is
`{restored_files, deleted_files, failed_files, error}`. Files that
existed before the run are restored from the backup; files the
`apply` step created from scratch are deleted. The same call powers
the **Undo** button surfaced by the Agent tab after a successful
apply.

## 15. Safety defaults

- **Plan tab** — CGX never writes to your project directory during
  plan generation. The "Run impacted tests" sandbox uses a temporary
  copy; disk writes only happen when you explicitly click **Apply**.
- **Agent tab (apply / scaffold tasks)** — the `apply` task *does*
  write diffs to `project_root`. A timestamped backup of every
  original file is created under `<project_root>/.cgx-backups/`
  before any file is overwritten, so you can roll back with
  `cgx.codegen.disk_apply.rollback_from_backup()` directly or via
  `POST /api/rollback` (see [Rollback an `apply` run](#rollback-an-apply-run)).
- Embedder specs (`module:attr`) execute Python on import — only use
  modules you trust.
- API keys live in your OS keyring (or `~/.cgx/secrets.json` with
  `0600` permissions); they are never echoed back through the UI or
  LLM transcripts.
- Session files live under `~/.cgx/sessions/` and inherit the user's
  umask. Once a profile has been saved (or the profile store has been
  initialised), `cgx.answer.profiles._ensure_dir` chmods `~/.cgx/` to
  `0700` (owner-only). Override the root via `CGX_CONFIG_DIR`.
