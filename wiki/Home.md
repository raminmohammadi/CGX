# CGX Wiki — Code Graph eXecution

**Local-first codebase RAG and self-testing code-generation platform.**

CGX indexes a repository, retrieves grounded context through a hybrid
engine (semantic + lexical + graph), and asks a local or remote LLM to
answer questions or produce **self-tested** code-change plans. It is
model-agnostic and ships with a React/Vite web UI served by a FastAPI
backend that streams progress over Server-Sent Events.

Point CGX at a repo and ask in plain English. Whether you are onboarding
to an unfamiliar codebase, planning a refactor, or scaffolding a
brand-new project, CGX grounds every answer and every code change in
your actual source — with citations to the exact files and lines — and
keeps it all on your machine unless you explicitly opt into a cloud
model.

---

## Start here

Pick the path that matches what you need right now.

| I want to…                                   | Go to |
|----------------------------------------------|-------|
| Install CGX                                  | **[[Installation]]** |
| Index my first repo and ask a question       | **[[Quick Start]]** |
| Understand how CGX works internally           | **[[How It Works]]** |
| Learn the web UI tab-by-tab                   | **[[Web UI Guide]]** |
| Script CGX from the terminal                  | **[[CLI Reference]]** |
| Drive the multi-step agent                    | **[[Session Based Agent]]** |
| Generate and auto-validate code changes       | **[[Self Testing Code Generation]]** |
| Teach CGX a new framework                     | **[[Skills Registry]]** |
| Choose a provider / model for my hardware     | **[[Providers and Models]]** |
| Tune retrieval, caching, or rate limits       | **[[Configuration and Tuning]]** |
| Understand what leaves my machine             | **[[Privacy and Security]]** |
| Run CGX in production (metrics, deploy)        | **[[MLOps and Production]]** |
| Read the internals as a contributor           | **[[Architecture]]** · **[[Contributing]]** |
| Fix a problem                                 | **[[Troubleshooting and FAQ]]** |

---

## Who is this for?

- **New users** — start with **[[Installation]]** and **[[Quick Start]]**.
  You do not need to understand the internals to use CGX productively.
- **Power users** — tune retrieval and scripting through
  **[[Configuration and Tuning]]**, the **[[CLI Reference]]**, and the
  **[[Providers and Models]]** hardware picker.
- **Contributors** — **[[Architecture]]**, **[[Session Based Agent]]**,
  and **[[Contributing]]** explain how CGX works and how to extend it.
  The easiest first contribution is a new **Skill**.

---

## What makes CGX different

- **Local-first.** Parsing, embedding, retrieval, sessions, and
  telemetry never leave the machine. Works fully offline with
  [Ollama](https://ollama.com/).
- **Universal LLM provider.** Ollama (local), OpenAI-compatible
  endpoints, native Google Gemini, Hugging Face Inference, or any
  self-hosted server — switchable at runtime with a live latency
  **Ping** check.
- **Hybrid retrieval.** Two-view semantic + BM25 + graph expansion,
  fused with Reciprocal Rank Fusion and an optional cross-encoder rerank.
- **Session-based agent.** Describe a goal; the agent works toward it one
  step at a time, pausing at every branch so **you approve each decision**.
  It explores an existing codebase or scaffolds a new project from a
  plain-language idea.
- **Self-testing code generation.** Diffs are parsed, syntax-checked, and
  optionally run against impacted tests in a sandbox before you ever see
  them.
- **Modular skills registry.** Each supported technology lives in its own
  folder and bundles detection, prompt guidance, and a structural
  validator. Adding a framework is a single-folder change.

See the full list in the project [README](https://github.com/raminmohammadi/Averix/blob/main/README.md#highlights).

---

## How the pieces fit together

```
parse → graph → embed → retrieve → answer / codegen
                                  ↑
                     session agent orchestrates one typed task at a time
```

The **[[How It Works]]** page walks this pipeline end to end, and
**[[Architecture]]** is the contributor-facing deep dive.

---

## Project facts

| | |
|---|---|
| **License** | MIT |
| **Language** | Python 3.10 / 3.11 / 3.12 (core is torch-free) |
| **Platforms** | Linux, macOS (Intel + Apple Silicon), Windows |
| **UI** | React/Vite SPA served by FastAPI on `127.0.0.1:8765` |
| **Default local model** | `qwen2.5-coder:3b` via Ollama |
| **Import package** | `cgx` (CLI: `cgx`, UI: `cgx-ui`) |

---

## Reference docs

The wiki is the curated, navigable entry point. The in-repo `docs/` set
holds the authoritative deep dives that individual wiki pages link into:

- [`docs/architecture.md`](https://github.com/raminmohammadi/Averix/blob/main/docs/architecture.md) — full architecture reference
- [`docs/mlops.md`](https://github.com/raminmohammadi/Averix/blob/main/docs/mlops.md) — production MLOps operator guide
- [`docs/usage.md`](https://github.com/raminmohammadi/Averix/blob/main/docs/usage.md) — exhaustive usage guide
- [`docs/Agent.md`](https://github.com/raminmohammadi/Averix/blob/main/docs/Agent.md) — session-agent internals
- [`docs/flowcharts.md`](https://github.com/raminmohammadi/Averix/blob/main/docs/flowcharts.md) — audience-specific diagrams
- [`docs/hardware_matrix.md`](https://github.com/raminmohammadi/Averix/blob/main/docs/hardware_matrix.md) — model fit matrix

> **A note on naming.** This wiki uses **CGX**, matching `README.md`,
> `pyproject.toml`, the `docs/` set, and the `cgx` import package. If the
> project is rebranding (the `averix` build metadata suggests it may be),
> the pages can be updated in one pass.
