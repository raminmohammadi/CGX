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

## The tabs (left → right)

### 1. Setup
Choose a **Provider Type** (Ollama, OpenAI, Google Gemini, Hugging Face,
or Custom Server), fill in the model and credentials, and click **Ping**
to verify the connection with a live latency check. Detect hardware
(RAM + GPU VRAM), tune sampling parameters, and save named profiles. API
keys are stored in your OS keyring. A **Browse Hugging Face** panel lists
GGUF repositories from the Hub and pulls them straight into your local
Ollama daemon. See **[[Providers and Models]]**.

### 2. Index
Point at a project root or upload a `.zip`. Honours `.gitignore` and a
1 MB file-size cap; emits `indices/`, `records.jsonl`, `chunks.jsonl`,
`graph.json`, and per-view `emb_cache_<view>.npz` for incremental
re-indexing. Intent and impl views build in parallel. A **Cancel** button
is available while indexing runs.

### 3. Ask
Natural-language question with a streaming **thought-process** panel and a
final grounded answer with citations. The sidebar holds the **session
list** (➕ New / 🗑️ Delete / dropdown to resume a thread). A **Stop**
button halts the stream mid-flight; switching tabs preserves the answer
in progress. Chat history is stored as JSONL under `~/.cgx/sessions/`.

### 4. Plan
Request a change plan. Optionally tick **Validate diffs** and **Run
impacted tests** to have CGX self-check its own output before returning;
the full self-test report renders inline. A **Cancel** button is
available; tab switching is non-destructive. See
**[[Self Testing Code Generation]]**.

### 5. Agent (`/agent`)
The **session-based** view and the default agentic surface. Start a
session with an objective, pick a **mode** (auto / explore / greenfield),
and watch the agent walk the appropriate chain, pausing at every branch
for a typed decision. The task tree shows the full DAG with status icons
and depth-based indentation; a side panel surfaces the Knowledge Base
(facts) and Artifacts. Nothing reaches disk until you tick the approval
checkpoint, and an **Undo** button rolls the run back via
`POST /api/rollback`. Full walkthrough: **[[Session Based Agent]]**.

### 6. Hardware
Click **Detect hardware** to annotate the local model catalogue with
✅/⚠️/❌ fit verdicts against your machine, plus a local-vs-cloud
trade-off table. Pure-offline — no network calls fire from this tab. See
the picker details in **[[Providers and Models]]**.

### 7. Profiles
Save provider configurations for any supported kind. Custom profiles
expose an **Endpoint Path** field and a **Skip auth** toggle for
private-subnet servers. Optional per-profile `rate_limit` and
`max_retries` apply automatically to every call made by that profile.

### 8. Ops (`/ops`)
The unified **observability hub** over the MLOps layer: live metrics,
pipeline/subsystem cards, and a **Trace explorer**. The explorer reads the
`@traced` function-call log (enable tracing with `CGX_TRACE` or the Settings
toggle) and lets you switch **source** between the Global fallback (HTTP / CLI
records) and any project's `agent.log` — the latter holds the rich records:
each LLM call with its **full prompt + response**, plus router, executor,
codegen, scaffold, and repair spans. Records are newest-first, redacted
server-side, filterable by event and category, with an "HTTP hidden" toggle and
click-through detail. **Ask** and **Plan** runs are traced into their project's
log too, not just the agent. **Delete** (current source) and **Delete all**
controls purge trace/log files only — never any other file — each behind a
confirmation. Deep dive: **[[MLOps and Production]]**.

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
