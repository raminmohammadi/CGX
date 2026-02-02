# Codebase RAG

This README explains how the repository works **end‑to‑end** and exactly **how to run it**.  
It also documents the main modules, data flow, CLI flags, expected inputs/outputs, and API surfaces you can call from code.

> The system is **model‑agnostic**. Bring your own embedder exposing `.encode(list[str]) -> np.ndarray`.

---

## Which file do I run? (Entry Point)

Use the **CLI** at `cgx/cli/main.py`. It now routes through a new **auto‑wired pipeline** that connects all key components.

```bash
# Make the src/ layout importable
export PYTHONPATH="$PWD/src:$PYTHONPATH"

# 1) INDEX — parse → graph → records → two-view embeddings → FAISS → persist
python -m cgx.cli.main index   --project-root /path/to/your/codebase   --embedder "myproj.embed:make_model"   --out-dir /tmp/cgx_index   --metric cosine   --index-type flat

# 2) QUERY — hybrid retrieval (semantic + optional lexical + graph) + aggregation + insertion anchors
python -m cgx.cli.main query   --index-dir /tmp/cgx_index/indices   --records /tmp/cgx_index/records.jsonl   --embedder "myproj.embed:make_model"   --query "How do we add a new FastAPI route?"
```

Under the hood the CLI calls the **auto‑wired** pipeline:

- `src/cgx/pipeline/auto.py`  
  - `run_index_auto(project_root, embedder, out_dir, metric, index_type)`  
  - `run_query_auto(index_dir, records_path, embedder, query, ...)`

Your **original** pipeline remains unchanged and available:

- `src/cgx/pipeline/run.py`  
  - `run_index(...)`  
  - `run_query(...)`

You can keep using these or delegate them to the auto‑wired versions if you want one canonical path.

---

## Architecture & Data Flow

1. **Parse → Chunks**  
   `cgx.parser.parse_codebase.parse_codebase(project_root)`  
   Produces canonical **chunks** (files/classes/functions/methods) and, if available, basic call edges.

2. **Graph**  
   `cgx.graph.build_graph.build_knowledge_graph(chunks, calls=None)`  
   Builds a NetworkX **knowledge graph** over code entities and relations (calls/modules/attrs/etc.).

3. **Records & Two‑View Corpus**  
   - `cgx.embeddings.records.make_index_records(chunks, G)` → **records** (deterministic S4‑style)  
   - `cgx.embeddings.records.prepare_embedding_corpus(records, which=('intent','impl'))` → **corpus**  
     - **intent** view: NL‑friendly summary (names/docstrings/comments)  
     - **impl** view: implementation‑centric text (code/signatures), optionally normalized

4. **Embeddings & FAISS per view**  
   - Embeddings (explicitly exercised): `cgx.embeddings.build.build_embeddings(...)`  
   - ANN index (explicitly exercised): `cgx.embeddings.index.build_faiss_index(...)`  
   - Persist per‑view artifacts via `cgx.io.persist.save_indices(...)`

5. **Retrieval, Fusion & Post‑processing**  
   - **Hybrid two‑view** retrieval (semantic on both views + optional lexical + optional graph) with RRF fusion:  
     `cgx.retrieval.orchestrator.hybrid_retrieve_two_view(...)`  
   - Aggregate to implementation units:  
     - `cgx.retrieval.orchestrator.aggregate_by_file(...)`  
     - `cgx.retrieval.orchestrator.aggregate_by_class(...)`  
   - Suggest **insertion points** for new code:  
     - `cgx.retrieval.orchestrator.suggest_insertion_points(query, results, records)`

---

## CLI Reference

### `index`

Builds two FAISS indices (one per view) and saves metadata + rows + records.

**Flags**

- `--project-root` (required): Repository to index.  
- `--embedder` (required): Import spec `"module:attr"` that yields an object with `.encode(list[str]) -> ndarray`.  
  - Class → instantiated with no args.  
  - Callable (factory) → called to produce the object.  
  - Pre‑instantiated object (module attr) → used directly.  
- `--out-dir` (required): Output directory.  
- `--metric` (default `cosine`): One of `cosine|l2|ip`.  
- `--index-type` (default `flat`): One of `flat|ivf|hnsw`.  
- Compatibility flags (kept for UX continuity; not required):  
  `--no-normalize-impl`, `--strip-literals-impl`.

**Output Layout**

```
/out-dir/
  ├── indices/
  │   ├── meta.json
  │   ├── intent.index
  │   ├── intent.rows.jsonl
  │   ├── impl.index
  │   └── impl.rows.jsonl
  └── records.jsonl
```

### `query`

Runs hybrid two‑view retrieval + file/class aggregation + insertion‑point suggestions.  
Optionally, runs a **single‑view** semantic helper for debugging.

**Flags**

- `--index-dir` (required): Directory containing `indices/` from the `index` step.  
- `--records` (required): Path to `records.jsonl`.  
- `--embedder` (required): Same import spec used at index time.  
- `--query` (required): User question / task.  
- `--chunks`: Optional `chunks.jsonl` to power lexical search.  
- `--graph`: Optional JSON graph (if you wish to include graph expansion).  
- `--top-k` (default 10): Per‑view semantic top‑k.  
- `--depth` (default 1): Graph neighbor depth (if graph expansion is enabled).  
- `--no-lexical`: Disable lexical component.  
- `--single-view {intent,impl}`: Also run the `semantic_search(...)` helper on a single view and return its top‑k.

**Output (printed JSON)**

- `hits` — fused top‑k chunks with ranks/scores.  
- `top_files` — aggregated by file.  
- `top_classes` — aggregated by class.  
- `anchors` — suggested insertion points (deterministic overlap signals).  
- `single_view` — optional block (when `--single-view` is provided).

---

## Programmatic Usage

### Auto‑wired pipeline (same as the CLI)

```python
from cgx.pipeline.auto import run_index_auto, run_query_auto

# Build indices
summary = run_index_auto(
    project_root="/path/to/code",
    embedder=make_model(),          # object with .encode(list[str]) -> ndarray
    out_dir="/tmp/cgx_index",
    metric="cosine",
    index_type="flat",
)

# Query with hybrid fusion + anchors
results = run_query_auto(
    index_dir="/tmp/cgx_index/indices",
    records_path="/tmp/cgx_index/records.jsonl",
    embedder=make_model(),
    query="How to add JWT validation?",
    top_k_per_view=10,
    neighbor_depth=1,
    use_lexical=True,
    single_view=None,               # or "intent"/"impl"
)
```

### Legacy pipeline (kept intact)

```python
from cgx.pipeline.run import run_index, run_query
# These remain available and unchanged.
```

---

## Configuration Objects

Typed configs live in `cgx/config.py` and support a simple overrides surface:

- `EmbeddingConfig.from_overrides(...).to_dict()`  
- `FaissConfig.from_overrides(metric="cosine", index_type="flat").to_dict()`  
- `HybridSearchConfig.from_overrides(rrf_k=60.0, ...).to_dict()`

> Some fields may also read environment variables (see `cgx/config.py` for exact names).

---

## Embedder Contract (BYO Model)

Any embedder works as long as it implements:

```python
.encode(list[str]) -> numpy.ndarray  # shape (N, D), dtype float32 preferred
```

**Tips**

- For `cosine`/`ip` metrics, L2‑normalize vectors across rows (both for index and query).  
- Reuse model/tokenizer across calls; batch requests to avoid overhead.

---

## Troubleshooting & Tips

- **PYTHONPATH**: Always export `PYTHONPATH="$PWD/src:$PYTHONPATH"` for src‑layout imports.  
- **Missing FAISS**: The persist layer degrades gracefully; install FAISS for best performance.  
- **Graph Optionality**: Hybrid retrieval works without a graph; provide one to enable graph expansion.  
- **Large Repos**: Consider `ivf`/`hnsw` for larger corpora; tune `nlist`, `efSearch`, etc., if exposed by your build.  
- **Determinism**: Records and row order are deterministic; indices map back to stable record IDs written in `*.rows.jsonl`.

---

## Module Map

- Parsing — `cgx.parser.parse_codebase.parse_codebase`  
- Graph — `cgx.graph.build_graph.build_knowledge_graph`  
- Records — `cgx.embeddings.records.make_index_records`, `prepare_embedding_corpus`  
- Embeddings — `cgx.embeddings.build.build_embeddings`  
- Index — `cgx.embeddings.index.build_faiss_index`  
- Orchestrator — `cgx.retrieval.orchestrator.hybrid_retrieve_two_view`, `aggregate_by_file`, `aggregate_by_class`, `suggest_insertion_points`  
- Persistence — `cgx.io.persist.save_indices/load_indices/save_jsonl/load_jsonl`  
- CLI — `cgx.cli.main`  
- Pipeline — **auto**: `cgx.pipeline.auto.run_index_auto`, `run_query_auto`; **legacy**: `cgx.pipeline.run.run_index`, `run_query`

---


# Codebase RAG — Full Guide (Capabilities & Usage)

This project indexes an entire **codebase** and lets you **ask questions**, **find the right places to modify**, and **add new functionality that fits** the existing patterns. It does this by parsing code into canonical chunks, building a two‑view embedding index (intent & implementation), optionally expanding across a code graph, and fusing multiple signals for grounded retrieval. It also suggests **insertion points** to help you place new code safely.

> **Bring‑Your‑Own Embedder** (BYOE). Any model works as long as it exposes `.encode(list[str]) -> numpy.ndarray`.

---

## What you can do

- **Ask questions about the codebase** in natural language (e.g., “Where is JWT verification implemented?”)
- **Find where to add new functionality** (e.g., “Where should I add CSV export for reports?”)
- **Reuse patterns** (e.g., “Show me canonical logging setup and usage across services”)
- **Discover APIs & contracts** (e.g., “Which class validates requests?”)
- **Explore related code** using optional **graph expansion** (follow callers/callees/imports)
- **Get insertion anchors** for new code (files/classes/locations most likely to be correct)

The system returns:
- `hits` (top chunks)
- `top_files` (file rollups)
- `top_classes` (class rollups)
- `anchors` (suggested insertion points)

---

## Which file do I run? (Entry Point)

Use the **CLI** at `cgx/cli/main.py`. It routes through the **auto‑wired pipeline** that connects all major components.

```bash
# Make the src/ layout importable
export PYTHONPATH="$PWD/src:$PYTHONPATH"

# 1) INDEX — parse → graph → records → two‑view (intent/impl) embeddings → FAISS → persist
python -m cgx.cli.main index   --project-root /path/to/your/codebase   --embedder "myproj.embed:make_model"   --out-dir /tmp/cgx_index   --metric cosine   --index-type flat

# 2) QUERY — hybrid retrieval (semantic + optional lexical + graph) + aggregation + insertion anchors
python -m cgx.cli.main query   --index-dir /tmp/cgx_index/indices   --records /tmp/cgx_index/records.jsonl   --embedder "myproj.embed:make_model"   --query "How do we add a new FastAPI route?"
```

**Under the hood** the CLI calls the **auto‑wired** pipeline:
- `src/cgx/pipeline/auto.py`
  - `run_index_auto(project_root, embedder, out_dir, metric, index_type)`
  - `run_query_auto(index_dir, records_path, embedder, query, ...)`

Your **original** pipeline remains unchanged and available:
- `src/cgx/pipeline/run.py`
  - `run_index(...)`
  - `run_query(...)`

---

## Install / Environment

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Make src/ layout importable for local development
export PYTHONPATH="$PWD/src:$PYTHONPATH"
# (Alternatively, pip install -e . if you have a proper pyproject/setup)
```

**Embedding Model** — provide an object created from `"module:attr"`:
- If it’s a **class**: it will be instantiated with no arguments.
- If it’s a **callable factory**: it will be called to obtain the object.
- If it’s an **object**: it will be used directly.
- The object must expose: `.encode(list[str]) -> numpy.ndarray`

---

## CLI Reference

### `index`
Builds two FAISS indices (one per view) and saves metadata + rows + records.

**Flags**
- `--project-root` (required): Repository to index
- `--embedder` (required): Import spec `"module:attr"` returning an object with `.encode(...)`
- `--out-dir` (required): Output directory
- `--metric` (default `cosine`): `cosine|l2|ip`
- `--index-type` (default `flat`): `flat|ivf|hnsw`
- Compatibility flags (kept for UX continuity): `--no-normalize-impl`, `--strip-literals-impl`

**Output Layout**
```
/out-dir/
  ├── indices/
  │   ├── meta.json
  │   ├── intent.index
  │   ├── intent.rows.jsonl
  │   ├── impl.index
  │   └── impl.rows.jsonl
  └── records.jsonl
```

### `query`
Runs **hybrid two‑view** retrieval + file/class aggregation + insertion‑point suggestions.  
Optionally, runs a **single‑view** semantic helper for debugging.

**Flags**
- `--index-dir` (required): Directory containing `indices/` from the `index` step
- `--records` (required): Path to `records.jsonl`
- `--embedder` (required): Same import spec used at index time
- `--query` (required): The question or task
- `--chunks`: Optional `chunks.jsonl` to power lexical search
- `--graph`: Optional JSON graph to enable graph expansion
- `--top-k` (default 10): Per‑view semantic top‑k
- `--depth` (default 1): Graph neighbor depth
- `--no-lexical`: Disable lexical component
- `--single-view {intent,impl}`: Also run `semantic_search(...)` on a single view

**Printed JSON**
- `hits` — fused top‑k chunks with ranks/scores
- `top_files` — aggregated by file
- `top_classes` — aggregated by class
- `anchors` — suggested insertion points
- `single_view` — optional block when `--single-view` is provided

---

## Programmatic Usage

### Auto‑wired (same execution path as the CLI)

```python
from cgx.pipeline.auto import run_index_auto, run_query_auto

# Build indices
summary = run_index_auto(
    project_root="/path/to/code",
    embedder=make_model(),          # object with .encode(list[str]) -> ndarray
    out_dir="/tmp/cgx_index",
    metric="cosine",
    index_type="flat",
)

# Query with hybrid fusion + anchors
results = run_query_auto(
    index_dir="/tmp/cgx_index/indices",
    records_path="/tmp/cgx_index/records.jsonl",
    embedder=make_model(),
    query="How to add JWT validation?",
    top_k_per_view=10,
    neighbor_depth=1,
    use_lexical=True,
    single_view=None,               # or "intent"/"impl"
)
```

### Legacy pipeline (kept intact)

```python
from cgx.pipeline.run import run_index, run_query
# Same responsibilities as auto-wired; available if you prefer legacy names.
```

---

## How it works (Architecture)

1) **Parsing → Chunks**  
`cgx.parser.parse_codebase.parse_codebase(project_root)` produces canonical chunks representing files/classes/functions/methods (and optionally call edges).

2) **Graph (optional)**  
`cgx.graph.build_graph.build_knowledge_graph(chunks, calls=None)` creates a NetworkX graph of entities and relations (calls/imports/attributes).

3) **Deterministic Records & Two‑View Corpus**  
- `cgx.embeddings.records.make_index_records(chunks, G)` → records (stable IDs, metadata)  
- `cgx.embeddings.records.prepare_embedding_corpus(records, which=('intent','impl'))` → corpus rows
  - **intent** view = NL‑friendly (names/docstrings/comments)
  - **impl** view = code‑focused (signatures/bodies)

4) **Two‑view Embeddings & FAISS**  
- `cgx.embeddings.build.build_embeddings(...)` encodes text per view
- `cgx.embeddings.index.build_faiss_index(...)` builds an ANN index per view
- Metadata + row mappings persisted with `cgx.io.persist.save_indices(...)`

5) **Hybrid Retrieval & Post‑Processing**  
- `cgx.retrieval.orchestrator.hybrid_retrieve_two_view(...)`: semantic on both views + optional lexical + optional graph → fused with RRF
- Aggregates: `aggregate_by_file(...)`, `aggregate_by_class(...)`
- Anchors: `suggest_insertion_points(query, results, records)` for safe code placement

---

## BYO Embedder (Contract & Tips)

**Contract**: any object with
```python
.encode(list[str]) -> numpy.ndarray  # shape (N, D), dtype float32 preferred
```

**Tips**
- For `cosine`/`ip`, L2‑normalize vectors across rows for both index and query.
- Batch large inputs to avoid overhead; reuse model/tokenizer across calls.

---

## Scenarios (Copy/Paste)

- **Find where to add a feature** (CSV export):
  ```bash
  python -m cgx.cli.main query     --index-dir /tmp/cgx_index/indices     --records /tmp/cgx_index/records.jsonl     --embedder "myproj.embed:make_model"     --query "Where should I add CSV export for reports? Show helpers and similar code paths."
  ```

- **Follow canonical logging pattern**:
  ```bash
  python -m cgx.cli.main query     --index-dir /tmp/cgx_index/indices     --records /tmp/cgx_index/records.jsonl     --embedder "myproj.embed:make_model"     --query "Find canonical logging setup and usage patterns across services"     --single-view impl
  ```

- **New OAuth provider** with awareness of neighbors:
  ```bash
  python -m cgx.cli.main query     --index-dir /tmp/cgx_index/indices     --records /tmp/cgx_index/records.jsonl     --embedder "myproj.embed:make_model"     --query "Add new OAuth provider: where to plug in config, handlers, and tests?"     --depth 2
  ```

---

## Troubleshooting

- **PYTHONPATH**: Ensure `export PYTHONPATH="$PWD/src:$PYTHONPATH"` for src‑layout imports.
- **FAISS**: If FAISS isn’t present, ensure the `requirements.txt` includes a CPU build (or install via conda).
- **Graph**: Not required. Provide one only if you want graph expansion.
- **Index / Query Mismatch**: Use the **same embedder** for querying as you used for indexing.
- **Large Repos**: Consider `--index-type ivf|hnsw` for scale; tune advanced params if exposed by your FAISS build.

---

## Module Map (for navigation)

- **Parsing** — `cgx.parser.parse_codebase.parse_codebase`
- **Graph** — `cgx.graph.build_graph.build_knowledge_graph`
- **Records** — `cgx.embeddings.records.make_index_records`, `prepare_embedding_corpus`
- **Embeddings** — `cgx.embeddings.build.build_embeddings`
- **ANN Index** — `cgx.embeddings.index.build_faiss_index`
- **Retrieval** — `cgx.retrieval.orchestrator.hybrid_retrieve_two_view`, `aggregate_by_file`, `aggregate_by_class`, `suggest_insertion_points`
- **Persistence** — `cgx.io.persist.save_indices/load_indices/save_jsonl/load_jsonl`
- **CLI** — `cgx.cli.main`
- **Pipelines** — **auto**: `cgx.pipeline.auto.run_index_auto`, `run_query_auto`; **legacy**: `cgx.pipeline.run.run_index`, `run_query`

---

## License & Contributing
