# Installation

CGX has a **small core** and a **separately-installable ML stack**. Pick
the path that matches how you plan to use it, then continue to
**[[Quick Start]]**.

CGX runs natively on **Linux**, **macOS** (Intel and Apple Silicon), and
**Windows**, on Python **3.10 / 3.11 / 3.12**. The only OS-specific step
is virtual-env activation; the CLI, UI, indexing, and agent loop are
identical across platforms.

---

## Prerequisites

- Python 3.10–3.12 and `pip`.
- (Recommended) [Ollama](https://ollama.com/) for fully-offline local
  models.
- (Optional) A CUDA-capable GPU for faster local embeddings/rerank.

---

## Core install (no torch)

Use this if you will point CGX at an Ollama server or an
OpenAI-compatible endpoint and either rely on the local embedder later or
supply your own embeddings via a BYO embedder callable.

```bash
git clone https://github.com/raminmohammadi/CGX.git
cd CGX
python -m venv .venv

# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell): .venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install -e ".[codegen]"
```

This installs FAISS, FastAPI/Uvicorn, NetworkX, and the codegen pieces
but **skips** `torch` / `transformers` / `sentence-transformers`. The
heavy ML modules are imported lazily, so the UI and CLI work out of the
box.

---

## Full install (with local embeddings)

Use this to load the default Jina embedding model locally and/or run the
optional cross-encoder reranker. Activate the venv first, then:

```bash
pip install -r requirements.txt -r requirements-ml.txt
pip install -e ".[codegen]"
# or, equivalently, via extras:
# pip install -e ".[ui,embeddings,faiss,codegen]"
```

Pull a small local model (the recommended default):

```bash
ollama pull qwen2.5-coder:3b
```

---

## Optional extras

Install any of these with `pip install -e ".[<extra>]"`:

| Extra        | Adds                                                   |
|--------------|--------------------------------------------------------|
| `ui`         | FastAPI + Uvicorn web-UI backend                       |
| `embeddings` | `sentence-transformers`, `transformers`, `torch`       |
| `faiss`      | `faiss-cpu` (large speedup over the numpy fallback)    |
| `codegen`    | `unidiff` (stricter diff parsing)                      |
| `keyring`    | OS keyring for API-key storage                         |
| `parsers`    | `tree-sitter` grammars for JavaScript / TypeScript / TSX |
| `mcp`        | Model Context Protocol SDK for external tool servers (see **[[Providers and Models]]**) |
| `viz`        | `matplotlib` (regenerate the docs book diagrams)       |
| `dev`        | `pytest`, `ruff`, `mypy`                               |

Without the `parsers` extra CGX still indexes Python; other languages are
simply skipped rather than failing the pipeline.

---

## Platform notes

### Linux
No extra steps for CPU. NVIDIA users who want GPU embeddings need a
**CUDA-matched** `torch` wheel — the default PyPI wheel tracks the newest
CUDA series and often silently falls back to CPU. Check the "CUDA
Version" column of `nvidia-smi`, then install the matching wheel:

```bash
# Driver supports CUDA 12.8 (most 5xx-series drivers):
pip install --index-url https://download.pytorch.org/whl/cu128 torch
```

Symptom of a mismatch: `torch.cuda.is_available()` is `False` while
`nvidia-smi` shows the GPU, embeddings run ~10× slower on CPU, and index
metadata reports `used_gpu: false`.

### macOS — Intel
CPU-only by default; same path as Linux.

### macOS — Apple Silicon
Works natively on arm64. The embedding model loads on CPU by default; to
use the Metal backend install the ML extras and set the device:

```bash
CGX_EMBED_DEVICE=mps cgx-ui
```

Ollama also runs natively on Apple Silicon (no Rosetta needed).

### Windows
Use PowerShell or `cmd.exe`. The venv activates with
`.venv\Scripts\Activate.ps1` (PowerShell) or `.venv\Scripts\activate.bat`
(cmd). The config directory resolves to `%USERPROFILE%\.cgx` (override
with `CGX_CONFIG_DIR`). The `0600` secrets-file fallback is a no-op on
NTFS, so install the `keyring` extra so API keys land in **Windows
Credential Manager** instead of a plain file.

---

## Verify the install

```bash
cgx status          # provider + hardware + index summary
cgx --help          # list every subcommand
```

Next: **[[Quick Start]]** to build your first index and ask a question.
