# CGX — Flowcharts

Three audience-specific views of the same system. Each SVG is hand-authored,
scales cleanly, and renders inline on GitHub.

---

## For users

![CGX user flow](diagrams/flow_user.svg)

Install once, point CGX at a repo, then ask questions or request changes in
plain English. The **Ask** tab returns a streaming, cited explanation; the
**Plan** tab returns a self-tested code-change diff; the **Agent** tab handles
larger goals — including generating brand-new projects from scratch — by
decomposing them into 1–6 atomic tasks with live progress. Everything runs
locally by default — cloud LLMs are strictly opt-in.

---

## For developers

![CGX developer flow](diagrams/flow_developer.svg)

`cgx.agents.run_agent` wires the **Planner → Tracker → Judge** loop. The
Planner asks the LLM for a strict-JSON
`[{name, description, kind, criteria}]` plan (ten kinds:
`ask`, `plan`, `scaffold`, `scaffold_manifest`, `scaffold_file`,
`search`, `summarize`, `apply`, `verify`, `fill_logic`) and applies
`_enforce_kind_policy()` to route the goal down one of four branches:
**SCAFFOLD** (new-project goals — detected via `_SCAFFOLD_RE`, a verb
paired with `_TECH_RE`, a verb paired with a supported skill from
`skills.detect_skills`, or LLM-emitted `scaffold` tasks with no
existing-codebase hint) — emits `[scaffold_manifest, apply, verify]`
where the manifest's runtime output injects one `scaffold_file` task
per planned file before `apply` runs, **VERIFY-ONLY**, **READ-ONLY**
(any `plan` task downgraded to `ask`), or **CHANGE-GOAL** (`apply` +
`verify` appended after `plan`). The Planner also attaches
`task.inputs["skills"]` to every SCAFFOLD/PLAN task so downstream
capabilities receive deterministic technology context.

The Tracker dispatches each task's `kind` to a capability on a worker
thread and yields `task_progress {task_id, elapsed}` events every
`progress_interval` (2.0 s default). The `plan` capability injects a
compressed **symbol-table** map (`cgx.codegen.symbol_map`) into the LLM
prompt so local models stop re-implementing helpers that already exist;
the `verify` capability runs `cgx.codegen.env_manager.preflight_install`
to auto-`pip install` any missing imports and append them to
`requirements.txt`; the `apply` capability performs a **partial apply**
that writes passing files and records failing files in `failed_files`,
plus a **cross-file coherence** check that catches a Python test
importing a `.jsx` module before anything hits disk. After every apply
the Tracker updates `plan.owned_files[path] = "applied" | "failed"` so
the retry loop knows what is already correct on disk.

The Judge runs cheap structural short-circuits per kind. For SCAFFOLD
and PLAN it consults the active **skills/** validators (`react`,
`nextjs`, `vue`, `tailwind`, `fastapi`, `flask`, `django`, `express`,
`python_cli`, `sqlite`); a failing `SkillVerdict` short-circuits to a
Judge fail with the skill name prefixed (`[react] …`). A scaffold that
passes structural + skill checks short-circuits to `pass` without
invoking the LLM judge — small local models hallucinate criteria fails
too often on demonstrably-correct scaffolds. When the LLM judge is
invoked, the SCAFFOLD branch of `_render_artifact` exposes `plan_md`,
the generated file list, and source-prioritised per-file previews
(capped at 7.5 KB total) so the verdict is grounded in the real code.

On verify failure `_stream_with_retry` calls `_diagnose_failure` to
classify the error (`import_error`, `syntax_error`, `logic_error`),
extracts a ±5-line snippet around the traceback line with
`# <-- ERROR HERE` (the **10-line buffer rule**), and emits a targeted
re-plan goal that names exactly the broken files and tells the LLM not
to touch the files already in `plan.owned_files`. Apply failures and
Judge rejections trigger the same recursive retry (up to `max_retries`).
The loop emits a final `summary` event and all events
(`plan`, `task_start`, `task_progress`, `task_done`, `task_skipped`,
`task_failed`, `judge`, `summary`) stream as SSE to `AgentPage.tsx`,
persisted into the SQLite task registry (`~/.cgx/tasks.db`) for replay
on tab switch. Every routing branch, skill attachment, and judge
verdict is written to stdout as `[INFO]` log lines.

### Inside the retrieval & codegen capabilities

The boxes labelled **search / ask / plan** in the developer diagram
hide a layered pipeline that is documented in detail in
[architecture.md](architecture.md) and exercised by the test suite.
The notes below are a quick map from the diagram to the modules.

The `search` box calls `cgx.pipeline.auto.run_query_auto`, which
fans out two ANN queries (intent view + impl view) against FAISS,
unions them with a BM25 lexical retriever, and fuses with Reciprocal
Rank Fusion. Identifier matching is **symmetric** — both indexer
(`cgx.embeddings.helpers`) and query (`cgx.retrieval.orchestrator`)
sides go through `cgx.retrieval.tokenize.split_identifier`, so a
query for `parseConfig` and an index entry for `parse_config` agree.
The fused head is optionally re-scored by a cross-encoder; the
**reranker is automatically on for cloud profiles** (OpenAI-compat,
Gemini) and off for local / air-gapped profiles, governed by
`cgx.answer.profiles.resolve_enable_reranker`. Graph expansion
walks one or two hops from the top hits via
`cgx.graph.backend.CodeGraphBackend`, which is a thin facade over
the small set of `networkx` operations the orchestrator actually
needs (decoupling retrieval from the graph library so a future
backend swap is local).

The `ask` and `plan` boxes call `cgx.answer.engine.answer_with_llm`
and `generate_code_plan` respectively. Both detect whether the
retriever surfaced graph-expanded neighbors (any hit with
`provenance.graph_depth >= 1`) and, when present, build the prompt
SOURCES list with `cgx.answer.context_map.build_tiered_context`
instead of the legacy single-tier builder. Direct matches keep their
focus-windowed code body (the **primary tier**); graph-discovered
neighbors collapse to one-line stubs of the form
`[class.]name(signature) — doc_first_sentence`, tagged
`tier=neighbor` in the prompt metadata (the **neighbor tier**). The
per-tier budget scales by the provider's model context window via
`cgx.answer.model_caps.get_context_map_budget`, so small local
models don't spend their whole window on structural references they
only need to *know* about.

The `plan` box's diff-application stage routes through
`cgx.codegen.ast_insert`, which can now prefer **line-anchored
splicing** when records carry the new `start_line` / `end_line` /
`col_offset` fields (schema v3) and falls back to its existing
AST-walk path for older indices. The companion anchor fields
`likely_caller_loc` and `similar_signature_neighbor_loc` are
emitted by `cgx.retrieval.orchestrator.suggest_insertion_points`
so an insertion target can be located without re-parsing the file.

The parser side is fronted by a small registry
(`cgx.parser.python_parser.PythonASTParser` registering for `.py`
via the `BaseParser` ABC in `cgx.parser.base`). The project walker
in `parse_codebase` dispatches on file extension; non-`.py` files
are silently skipped today. Adding a language later means writing a
new `BaseParser` subclass and registering its extensions — no
changes to the orchestrator or codegen layers.

---

## For companies

![CGX trust boundaries](diagrams/flow_company.svg)

Source code, embeddings, FAISS indices, chat sessions, the SQLite task
registry (`~/.cgx/tasks.db`), and the embedding cache all live on the
local machine under `~/.cgx/` and `indices/`. The agent loop runs
in-process and streams SSE over localhost; the task registry persists
every event so the UI can replay a tab on remount and `DELETE
/api/tasks/{id}` can cancel a running stream — there is no analytics
or telemetry channel. Credentials live in the OS keyring when
available (`0600`-permissioned file fallback) and are never echoed to
event payloads or tool-call arguments. The only opt-in egress is when
a profile points at a remote provider — **OpenAI-compatible**, **Google
Gemini**, or a **custom** OpenAI-shape endpoint (with optional
`allow_no_auth` for private subnets) — in which case the prompt plus
the retrieved snippets are sent; the repository, indices, sessions,
and task registry are not. `POST /api/provider/ping` performs a
liveness check (e.g. Gemini `generateContent` with `maxOutputTokens:
1`, Ollama `GET /api/tags`) and returns only `{ok, latency_ms,
error}`. Air-gapped operation is the default once an Ollama model is
pulled.
