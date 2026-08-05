# Quick Start

This page takes you from a fresh install to a grounded answer in a few
minutes. If you have not installed CGX yet, see **[[Installation]]**.

You can drive CGX from three surfaces that share one engine: the **web
UI**, the **CLI**, and a **Python API**. Start with whichever fits your
workflow.

---

## 1. Start a local model (recommended)

```bash
ollama serve                 # in a separate terminal
ollama pull qwen2.5-coder:3b # small, capable coder model
```

CGX defaults to Ollama at `http://localhost:11434`. To use a cloud
provider instead, see **[[Providers and Models]]**.

---

## 2. Launch the web UI

```bash
cgx-ui            # after `pip install -e ".[ui]"`
# or
python app.py
# or
cgx serve
```

The server binds to `127.0.0.1:8765` by default, so the UI is reachable
only from the same host. Open <http://localhost:8765> and work left to
right through the tabs:

1. **Setup** — choose a provider, fill in the model/credentials, and
   click **Ping** to verify the connection with a live latency check.
2. **Index** — point at a project root (or upload a `.zip`) and build the
   code graph.
3. **Ask** — type a natural-language question and read the streamed,
   cited answer.

The full tab-by-tab tour is in the **[[Web UI Guide]]**.

---

## 3. Or use the CLI

```bash
# 1. Build an index into the auto-discovered .cgx/index location
cgx index --project-root . --out-dir .cgx/index

# 2. Ask a grounded, streamed question over that index
cgx ask "What does parse_codebase do?" --think

# 3. Generate a self-tested code-change plan
cgx plan "Add a --json flag to the query command" --self-test --run-tests

# 4. Run one unattended turn of the session agent
cgx agent "Add docstrings to every public function in cgx.parser"
```

`ask`, `plan`, `agent`, and `status` auto-discover the index at
`<project-root>/.cgx/index`, stream tokens live, and cancel cleanly on
**Ctrl-C**. Full details: **[[CLI Reference]]**.

Prefer a terminal cockpit? Run bare `cgx` (or `cgx dash`) for the
interactive dashboard.

---

## 4. Or call the Python API

```python
from cgx.pipeline.auto import run_index_auto
from cgx.answer.engine import answer_with_llm
from cgx.answer.providers import OllamaProvider

run_index_auto(project_root="./", out_dir="/tmp/cgx_index")

prov = OllamaProvider(model="qwen2.5-coder:3b")
ans = answer_with_llm(
    "/tmp/cgx_index/indices",
    "/tmp/cgx_index/records.jsonl",
    "What does parse_codebase do?",
    prov,
)
print(ans["answer_md"])
```

---

## What just happened?

Indexing walked your repo, chunked it per file/class/function, built a
call/containment graph, and embedded two views (intent + implementation)
into FAISS. Asking ran **hybrid retrieval** (semantic + lexical + graph),
fused the results, and handed a grounded, line-windowed context to the
LLM — which returned an answer with citations to the exact files and
lines.

The **[[How It Works]]** page explains this pipeline in depth.

---

## Where things live

| Artifact | Location |
|----------|----------|
| Index (FAISS, records, graph, caches) | `<project-root>/.cgx/index/` |
| Chat sessions (Ask tab history)       | `~/.cgx/sessions/` |
| Agent sessions                        | `<project-root>/.cgx/sessions.db` |
| Provider profiles                     | `~/.cgx/profiles.json` |
| Secrets (fallback)                    | `~/.cgx/secrets.json` (`0600`) or OS keyring |
| Apply backups                         | `<project-root>/.cgx-backups/<run_id>/` |

Override the config directory anywhere with `CGX_CONFIG_DIR`.

---

## Next steps

- **[[Web UI Guide]]** — every tab explained.
- **[[Session Based Agent]]** — multi-step, human-in-the-loop work.
- **[[Providers and Models]]** — pick the right model for your hardware.
- **[[Configuration and Tuning]]** — sharpen retrieval quality.
