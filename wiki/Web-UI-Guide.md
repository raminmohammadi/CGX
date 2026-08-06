# Web UI Guide

The web UI is a React/Vite SPA served by a FastAPI backend that streams
progress over Server-Sent Events (SSE). Launch it with:

```bash
cgx-ui            # or: python app.py   |   cgx serve
```

It binds to `127.0.0.1:8765` by default; open <http://localhost:8765>.

> **Binding & remote access.** The server has **no built-in
> authentication** — anything that can reach the bound `host:port` can
> drive the agent loop, read sessions, and write to disk under the
> configured Project Root. Bind to a non-loopback address only on a
> trusted LAN/VPN or behind an auth-enforcing reverse proxy. See
> **[[Privacy and Security]]**. Override with `--host` / `--port` or the
> `CGX_HOST` / `CGX_PORT` environment variables.

---

## The left sidebar

Navigation is a grouped sidebar, not a flat tab row. Each group holds one or
more pages; the active page is highlighted, and pages with a running
task show a live indicator.

| Group | Page (route) | What it does |
|-------|--------------|--------------|
| — | **Overview** (`/`) | Landing dashboard: provider, index and session status at a glance. |
| **Converse** | **Contextual Ask** (`/ask`) | Grounded Q&A with citations. |
| **Build** | **Self-Testing Plan** (`/plan`) | Change plans that self-validate. |
| **Build** | **Agent Loop** (`/agent`) | The multi-step session agent. |
| **Retrieval** | **Incremental Index** (`/index`) | Index a repo / upload a `.zip`. |
| **Observability** | **Ops & Observability** (`/ops`) | The MLOps hub (10 tabs). |
| **System** | **Profiles & Setup** (`/settings`) | Provider, profiles, Hugging Face, tracing, hardware. |

The sidebar also holds the **session list** (➕ New / 🗑️ Delete / dropdown to
resume a thread) used by Ask, Plan and the Agent.

### Contextual Ask (`/ask`)
Natural-language question with a streaming **thought-process** panel and a
final grounded answer with citations. A **Stop** button halts the stream
mid-flight; switching pages preserves the answer in progress. Chat history is
stored as JSONL under `~/.cgx/sessions/`.

### Self-Testing Plan (`/plan`)
Request a change plan. Optionally tick **Validate diffs** and **Run impacted
tests** to have CGX self-check its own output before returning; the full
self-test report renders inline. A **Cancel** button is available; switching
pages is non-destructive. See **[[Self Testing Code Generation]]**.

### Agent Loop (`/agent`)
The **session-based** agentic surface. Start a session with an objective, pick
a **mode** (auto / explore / greenfield), and watch the agent walk the chain,
pausing at every branch for a typed decision. The task tree shows the full DAG
with status icons; a side panel surfaces the Knowledge Base (facts) and
Artifacts. Nothing reaches disk until you tick the approval checkpoint, and an
**Undo** button rolls the run back via `POST /api/rollback`. It also carries
its own sub-tabs for **Agent Profiles**, **Skills** and **New Skill**. Full
walkthrough: **[[Session Based Agent]]**.

### Incremental Index (`/index`)
Point at a project root or upload a `.zip`. Honours `.gitignore` and a 1 MB
file-size cap; emits `indices/`, `records.jsonl`, `chunks.jsonl`, `graph.json`,
and per-view `emb_cache_<view>.npz` for incremental re-indexing. Intent and
impl views build in parallel. A **Cancel** button is available while indexing
runs.

### Ops & Observability (`/ops`)
The unified **observability hub** over the MLOps layer — ten tabs of metrics,
activity, alerts, cost, feedback, governance, health and a **Trace explorer**.
Because users found the density confusing, the hub now has its own field
guide that names **every card, chart and button** tab by tab:
**[[Ops and Observability]]** (deep dive) and **[[MLOps and Production]]** (the
subsystems behind it).

## Profiles & Setup (`/settings`)

The settings page is a left category list. Search filters the categories.

### Active Provider
Choose a **Provider Type** (Ollama, OpenAI, Google Gemini, Hugging Face, or
Custom Server), fill in the model and credentials, and click **Ping** to verify
the connection with a live latency check. API keys are stored in your OS
keyring. See **[[Providers and Models]]**.

### Saved Profiles
Save provider configurations for any supported kind, then **Use / Edit /
Delete** them. Custom profiles expose an **Endpoint Path** field and a **Skip
auth** toggle for private-subnet servers. Optional per-profile `rate_limit` and
`max_retries` apply automatically to every call made by that profile. Use
**New profile** (bottom of the category list) to start one.

### Browse Hugging Face
Search the Hub for GGUF repositories, size them against your hardware with
**Check fit**, and **Pull** them straight into your local Ollama daemon (no HF
token needed). Full panel breakdown in **[[Providers and Models]]**.

### Observability
A single **tracing toggle**. Turning it on is equivalent to `CGX_TRACE=1`: from
then on, ask/plan/agent runs write rich `@traced` records (each LLM call with
its full redacted prompt + response) into the project's `agent.log`, which the
Ops hub's **Trace** tab reads.

### Hardware
Click **Detect Hardware Budget** to probe RAM + GPU VRAM and annotate the local
model catalogue with ✅/⚠️/❌ fit verdicts, flag models you've already
downloaded, size any Hugging Face repo, and show a local-vs-cloud trade-off
grid. The catalogue is pure-offline. Details in **[[Providers and Models]]**.

---

## Cross-cutting UI behaviour

- **Cancel/Stop on every tab.** Halt a streaming request mid-flight from
  Ask (Stop), Plan, Agent, or Index (Cancel). This flips a shared
  `cancel_event` so the backend stops between tokens rather than the
  process dying.
- **Tab persistence.** Switching tabs mid-task does not lose the running
  view — state is held in a session-scoped store and the SSE stream
  continues in the background.
- **Task registry.** Every operation is tracked in `~/.cgx/tasks.db`;
  cancel any running task with `DELETE /api/tasks/{id}` or the in-UI
  Cancel button.

---

## VS Code extension

`extension/` is a minimal TypeScript extension that hosts the running CGX
web UI inside a VS Code webview. It is not packaged into a `.vsix` from
the repo — build it locally:

```bash
cd extension
npm install
npm run compile
# then press F5 in VS Code to launch an Extension Development Host
```

Commands contributed: **CGX: Open UI**, **CGX: Reload UI**. The server
URL is read from the `cgx.ui.url` setting (default
`http://localhost:8765`); the extension does not spawn the server, so
start it with `cgx-ui` first. See
[`extension/README.md`](https://github.com/raminmohammadi/Averix/blob/main/extension/README.md).

---

## Prefer the terminal?

Every UI capability is also available from the **[[CLI Reference]]** and
the interactive dashboard (`cgx` / `cgx dash`).
