# Privacy and Security

CGX is **local-first by design**. Parsing, embedding, indexing,
retrieval, prompt assembly, session storage, backups, and (with the
default Ollama provider) LLM inference all run on your machine. This page
is the honest, complete account of what stays local, what can leave, and
how to keep the system safe.

---

## What stays on your machine

By default the **entire pipeline** runs on your host:

- **Parsing & graph building** — tree-sitter / AST, no network.
- **Embeddings** — the model is downloaded once by Hugging Face on first
  use and then cached locally; embedding itself is offline.
- **FAISS indices & JSONL records** — written wherever you point
  `--out-dir` (typically `<project_root>/.cgx/index`).
- **Sessions, tasks, caches, backups, profiles, secrets** — under
  `~/.cgx/` (or `$CGX_CONFIG_DIR`).
- **LLM inference with Ollama** — served over `localhost:11434`; nothing
  leaves the machine.

There is **no telemetry by default** (see below).

---

## What can leave your machine (egress)

CGX only makes outbound network calls in these cases:

| Path | When it happens |
|------|-----------------|
| **Cloud LLM provider** | You explicitly select OpenAI, Gemini, Hugging Face Inference, an OpenAI-compatible endpoint, or a custom server. Your prompt (which includes retrieved code snippets) is sent to that provider. |
| **Browse Hugging Face** | The Settings panel lists GGUF repositories from `huggingface.co/api/models`; **Pull** delegates the download to your local Ollama daemon (`hf.co/<repo>`), which then re-aliases the model to a clean local name via loopback `POST /api/copy` + `DELETE /api/delete` (no extra outbound traffic). |
| **Provider Ping** | The Setup tab / `POST /api/provider/ping` runs a live reachability check against the configured provider. |
| **Model discovery** | Listing available models for a cloud provider, restricted to an allow-list of approved hosts (SSRF barrier in `webui/routes/setup.py`). |
| **First-run model download** | Hugging Face downloads the embedding model once, then caches it. |
| **Opt-in telemetry** | Only when `CGX_TELEMETRY=1` **and** a collector URL are set. |

If you stay on Ollama and never opt into telemetry, CGX makes **no
outbound calls** beyond the one-time embedding-model download.

---

## Telemetry (opt-in, off by default)

`cgx.telemetry.ping()` is a ~50-line, best-effort startup ping that
exists solely to count active installs. It is **off unless you set
`CGX_TELEMETRY=1`**, and even then only fires when a collector URL is
configured (`CGX_TELEMETRY_URL`; the default endpoint is empty).

The entire payload is:

```json
{"install_id": "<random UUID4>", "cgx_version": "<version>", "event": "startup"}
```

No prompts, code, file paths, model names, or PII — ever. The install id
lives at `~/.cgx/install_id`; delete it to rotate. The POST runs in a
daemon thread with a 2 s timeout and swallows every exception. Review
[`src/cgx/telemetry.py`](https://github.com/raminmohammadi/Averix/blob/main/src/cgx/telemetry.py) before opting in.

---

## Where credentials live

API keys and bearer tokens are **never passed on the command line** and
are stored, in order of preference:

1. **OS keyring** — used automatically when the `keyring` extra is
   installed.
2. **`~/.cgx/secrets.json`** — the fallback, created with `0600`
   permissions via `os.open(..., 0o600)` (not a write-then-chmod race).

Cloud providers read keys from the environment (`OPENAI_API_KEY`,
`GEMINI_API_KEY`) or the keyring-backed profile store. Error strings are
scrubbed of Gemini-style `?key=...` query params
(`GeminiProvider._scrub_secret`) so a failed HTTP call never leaks a key
into a log or SSE payload.

---

## The web server has no authentication

`cgx serve` / `cgx-ui` binds to `127.0.0.1:8765` by default and ships
with **no built-in auth**. Anything that can reach the bound `host:port`
can drive the agent loop, read your sessions, and write files under the
configured Project Root.

- Keep the bind on **loopback** for single-user local use.
- Only bind to a non-loopback address (`--host` / `CGX_HOST`) on a
  **trusted LAN/VPN** or **behind an auth-enforcing reverse proxy**.
- CORS is permissive only for the Vite dev origin (`localhost:5173`);
  production builds are same-origin.

---

## Write scope & reversibility

The agent only writes inside the **Project Root** you configure, and only
**after you approve** a checkpoint — nothing reaches disk implicitly.

- Every overwrite is mirrored under
  `<project_root>/.cgx-backups/<run_id>/`.
- A whole run is reversible via `POST /api/rollback` (the UI **Undo**
  button).
- Set the Project Root deliberately; treat it as the blast radius.

---

## Human-in-the-loop approval for risky tools

The swarm agent's tools include arbitrary code execution
(`run_python_probe`), file writes, and — once MCP is configured — calls that
reach the outside world. An **opt-in approval gate**
(`cgx.session.approval`) can intercept these before they run.

- **Off by default.** With no gate installed, tools dispatch exactly as before
  — enabling the gate is the only behavior change.
- **Mode is env-driven** via `CGX_APPROVAL_MODE`: `off`, `risky` (the default
  when a gate is active — gates MEDIUM/HIGH tools), or `all` (gates every tool).
- **Fail-safe.** A request **blocks** the worker until it is approved or denied;
  if nobody answers within the TTL (default 1800 s) it **auto-rejects** — an
  unanswered risky call is never silently run.
- **Terminal**: `cgx agent --approve <goal>` prompts `y/N` before each risky
  call. **Web/SSE**: a front-end reads `GET /api/approvals/pending` and resolves
  with `POST /api/approvals/resolve`.

See **[[Configuration and Tuning]]** for the env var and **[[CLI Reference]]**
for the flag.

## MCP tool servers (local config, env-held secrets)

The optional **MCP (Model Context Protocol)** tool layer follows the same
local-first, secrets-in-env posture as the rest of CGX:

- The server roster is a **local JSON file** at `~/.cgx/mcp.json` (override with
  `CGX_MCP_CONFIG`). Adding a tool server is a config edit, not a code change.
- **Bearer tokens are never stored in the JSON.** A server entry names an env
  var via `token_env`; the token is read from that variable at call time.
- MCP calls (`mcp_call`) are HIGH risk, so they are gated by the approval gate
  above when it is active. See **[[Providers and Models]]**.

Configuring an MCP server can cause egress to whatever endpoint that server
talks to — treat each server you add as a deliberate outbound path.

---

## Tracing & redaction

Function-call tracing (`CGX_TRACE=1` or the `/settings` toggle) writes
`@traced` span records and, per LLM call, an `llm_call` record with the
**full prompt and response** — but everything is passed through
`cgx.redact` first, so credential-shaped literals are masked before they
reach disk or the admin API. Records land in the project-local
`<root>/.cgx/agent.log` (or the global `~/.cgx/cgx-trace.log` for
HTTP / CLI activity) and **never leave the machine**. Tracing is off by
default. See `cgx.trace` and `cgx.redact`.

You can purge these trace logs from **Ops → Trace** (or
`DELETE /api/admin/logs`). That path is hard-limited to trace/log files:
it only unlinks files literally named `agent.log` / `cgx-trace.log` (and
their rotation backups), requires a regular file, and refuses symlinks —
so a supplied `project_root` can never be turned into a delete of any
other file on the machine.

---

## See also

- **[[Providers and Models]]** — provider selection and credential flow.
- **[[Web UI Guide]]** — binding and remote-access notes.
- **[[Session Based Agent]]** — approval checkpoints and rollback.
- [`docs/architecture.md` § Telemetry / Privacy](https://github.com/raminmohammadi/Averix/blob/main/docs/architecture.md).
