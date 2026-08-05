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
