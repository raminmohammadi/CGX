# CLI Reference

The `cgx` command exposes every capability as a scriptable subcommand.
Run `cgx <command> --help` for the authoritative flag list; this page is
the reference. Bare `cgx` (or `cgx dash`) opens the interactive
dashboard.

```bash
cgx <command> [flags]
```

---

## Core commands at a glance

| Command      | What it does                                                   |
|--------------|----------------------------------------------------------------|
| `cgx`        | Open the interactive dashboard (same as `cgx dash`).           |
| `cgx index`  | Parse → embed (two views) → FAISS → persist an index on disk.  |
| `cgx query`  | Raw hybrid retrieval; prints ranked chunks as JSON (no LLM).   |
| `cgx ask`    | Grounded, read-only LLM answer streamed to the terminal.       |
| `cgx plan`   | Generate a code-change plan (`plan_md` + structured diffs).    |
| `cgx agent`  | Run one unattended turn of the session agent loop.             |
| `cgx status` | Print provider, hardware, and index status.                    |
| `cgx serve`  | Launch the FastAPI + React web UI (`--host` / `--port`).       |
| `cgx dash`   | Launch the interactive terminal dashboard.                     |

`ask`, `plan`, `agent`, and `status` stream tokens/events live under a
Braille spinner and **cancel on Ctrl-C** (exit code `130`); any other
failure exits `1`. Colour follows the TTY / `NO_COLOR` /
`CGX_FORCE_COLOR` rules, so piping to a file yields clean text.

## Management & operations commands

Everything the web UI can do is also a scriptable subcommand. These follow a
`git`-style **noun-verb** shape, print **JSON by default** (add `--table` on
`list`-style verbs for a human view), and map errors to a non-zero exit code
(`2` = not found, `3` = validation/forbidden). Each is a thin adapter over the
same engine the web UI's REST routes call.

| Command group | Verbs | Mirrors |
|---------------|-------|---------|
| `cgx skills` | `list` · `show` · `create` · `edit` · `delete` (`--markdown`) | Skills registry |
| `cgx profile` | `list` · `show` · `add` · `rm` | LLM connection presets |
| `cgx agent-profile` | `list` · `show` · `add` · `rm` | Saved `{objective, skills}` bundles |
| `cgx context` | `show` · `edit` · `rm` | Repo `CGX.md` context file |
| `cgx site` | `list` · `new` · `generate` · `revise` · `path` · `themes` | Site Studio |
| `cgx session` | `list` · `show` · `rm` · `send` · `decide` | Agent sessions: inspect, drive, decide |
| `cgx model` | `list` · `matrix` · `fit` · `pull` · `embed-list` · `embed-pull` | Model discovery / pull |
| `cgx mcp` | `list` · `enable` · `disable` · `tools` | MCP tool servers |
| `cgx activity` | `runs` · `summary` · `show` | Per-run observation store |
| `cgx usage` | `show` · `summary` | Cost & quota governance |
| `cgx feedback` | `stats` · `list` · `add` | Thumbs-up/down feedback |
| `cgx monitor` | `alerts` | AIOps quality/drift/cost alerts |
| `cgx govdata` | `policy` · `scan` · `purge` · `erase` | Data governance / PII |
| `cgx admin` | `overview` · `logs` · `metrics` · `purge-logs` | Admin logs / metrics |
| `cgx trace` | `status` · `on` · `off` | Runtime trace toggle |
| `cgx approvals` | `pending` · `resolve` | Human-in-the-loop approvals (in-process) |
| `cgx rollback` | (restore from an APPLY backup dir) | Undo applied changes |
| `cgx health` | (liveness + readiness; non-zero exit when not ready) | `/healthz` · `/readyz` |
| `cgx metrics` | (Prometheus text, or `--json` snapshot) | `/api/metrics` |

Run `cgx <group> --help` and `cgx <group> <verb> --help` for the authoritative
flags. Notes: `session send`/`decide` **drive** an existing session (stream live,
Ctrl-C cancels) — `send` posts a follow-up message, `decide --task-id --chosen
'{"approved": true}'` answers an open decision; run from the project dir (or pass
`--project-root`) so the session id resolves. `cgx agent` (unattended) and
`cgx dash` (interactive) remain the ways to *start* a session; `approvals resolve` acts
on an in-process gate, so for terminal runs prefer `cgx agent --approve`;
`govdata purge`/`erase` and `admin purge-logs` are destructive and require an
explicit scope flag.

---

## Shared provider & index flags

`ask`, `plan`, `agent`, and `status` accept a common flag set:

| Flag             | Default                  | Purpose |
|------------------|--------------------------|---------|
| `--project-root` | current directory        | Project whose index is read / written. |
| `--provider`     | `ollama`                 | `ollama`, `openai`, `openai-compat`, `gemini`, `huggingface`, or `custom`. |
| `--model`        | provider default         | LLM name; always overrides a profile's model. |
| `--base-url`     | `http://localhost:11434` | Provider endpoint. |
| `--profile`      | none                     | A saved profile — takes precedence over `--provider`/`--base-url`. |
| `--index-dir`    | auto-discovered          | Override the FAISS `indices/` directory. |
| `--records`      | auto-discovered          | Override the `records.jsonl` path. |

**Provider resolution.** `--profile` wins (its `kind`, `model`,
`base_url`), with `--model` still able to override the model. Otherwise
explicit flags are used. For `ollama` with no model given, CGX picks a
hardware-appropriate default. **API keys are never passed on the command
line** — cloud providers read `OPENAI_API_KEY` / `GEMINI_API_KEY` or the
keyring-backed profile store.

**Index discovery.** `ask` / `plan` / `status` look for a completed index
at `<project-root>/.cgx/index`; `agent` uses it when present and can also
run without one for greenfield generation.

---

## `cgx index`

```bash
cgx index --project-root . --out-dir .cgx/index
```

Key flags: `--project-root` (required), `--out-dir` (required),
`--metric {cosine,l2,ip}`, `--index-type {flat,ivf,hnsw}`,
`--model NAME`, `--embedder module:attr` (BYO embedder). Prints a JSON
summary including incremental-cache hit/miss counts.

## `cgx query`

```bash
cgx query --index-dir .cgx/index/indices \
          --records  .cgx/index/records.jsonl \
          --query "What does parse_codebase do?"
```

Raw hybrid retrieval as JSON — no LLM. Useful for debugging retrieval or
piping into other tooling. Flags include `--top-k`, `--depth` (graph
expansion), `--no-lexical`, `--single-view {intent,impl}`, `--limit`.

## `cgx ask`

```bash
cgx ask "What does parse_codebase do?" --think
```

Streams a read-only, citation-grounded answer. `--think` also streams the
model's reasoning sketch. The multi-word question is positional (quoting
optional).

## `cgx plan`

```bash
cgx plan "Add a --json flag to the query command" --self-test --run-tests
```

Emits a markdown plan plus structured diffs. `--self-test` validates and
repairs its own diffs (parse + dry-apply + `ast.parse`); `--run-tests`
executes impacted tests in a sandbox. See
**[[Self Testing Code Generation]]**.

## `cgx agent`

```bash
cgx agent "Add docstrings to every public function in cgx.parser"
```

Runs one unattended turn of the session agent loop (**[[Session Based
Agent]]**) in auto-answer mode: clarify questions take the first
suggested option and approvals are approved, so the turn never blocks;
questions with no safe default end the turn with the question printed.
With no discoverable index the loop scaffolds a brand-new project into
`--project-root`. The session persists to
`<project_root>/.cgx/sessions.db` and can be resumed from the UI or
dashboard.

`--approve` installs an opt-in **human-in-the-loop gate**: before any risky
tool call (code execution, file writes, MCP calls) the run prompts for `y/N` at
the terminal and blocks until you answer. Off by default, so unattended runs are
unchanged. The gated risk level is controlled by `CGX_APPROVAL_MODE` — see
**[[Privacy and Security]]** and **[[Configuration and Tuning]]**.

## `cgx status`

```bash
cgx status --provider ollama
```

Prints the resolved provider/model, a live Ollama reachability check,
detected RAM/VRAM, and whether the project's index is built (with build
timestamp and embedding model).

---

## The interactive dashboard

Run bare `cgx` (or `cgx dash`) for a full-screen REPL that unifies
indexing, questions, and the agent loop. Type a plain message to route it
through the session agent; lines starting with `/` are commands:

| Command | Action |
|---------|--------|
| `/help` | Show the command reference. |
| `/ask <q>` | Fast, read-only grounded answer (streams live). |
| `/index [path]` | Build/refresh the code graph. |
| `/project <path>` | Switch the active project directory. |
| `/model <name>` · `/provider <name>` | Set model / provider (or a saved profile). |
| `/status` · `/serve` · `/clear` · `/quit` | Status / launch UI / clear / exit. |

The dashboard is stdlib-only (ANSI escapes, no `rich`/`textual`), so it
works cleanly over SSH. **Ctrl-C** cancels the running task; `/quit` or
Ctrl-D leaves.

Full reference:
[`docs/usage.md`](https://github.com/raminmohammadi/CGX/blob/main/docs/usage.md#the-cli-non-interactive-subcommands).
