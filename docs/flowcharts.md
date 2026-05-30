# Averix — Flowcharts

Three audience-specific views of the same system. Each SVG is hand-authored,
scales cleanly, and renders inline on GitHub.

---

## For users

![Averix user flow](diagrams/flow_user.svg)

Install once, point Averix at a repo, then ask questions or request changes in
plain English. The **Ask** tab returns a streaming, cited explanation; the
**Plan** tab returns a self-tested code-change diff; the **Agent** tab handles
larger goals by decomposing them into 1–5 atomic tasks with live progress.
Everything runs locally by default — cloud LLMs are strictly opt-in.

---

## For developers

![Averix developer flow](diagrams/flow_developer.svg)

`cgx.agents.run_agent` wires the **Planner → Tracker → Judge** loop. The
Planner asks the LLM for a strict-JSON `[{name, description, kind, criteria}]`
plan and applies `_enforce_kind_policy()` to downgrade `plan` → `ask` for
read-only goals. The Tracker dispatches each task's `kind` to a capability
(`search`, `ask`, `plan`, `summarize`) on a worker thread and yields
`task_progress {task_id, elapsed}` events on a configurable
`progress_interval` so the React UI never looks frozen. The Judge runs cheap
structural short-circuits before optionally asking the LLM for a
`{verdict, confidence, rationale}` verdict, then the loop emits a final
`summary` event. All events stream as SSE to `AgentPage.tsx`.

---

## For companies

![Averix trust boundaries](diagrams/flow_company.svg)

Source code, embeddings, FAISS indices, chat sessions, and the embedding cache
all live on the local machine under `~/.cgx/` and `indices/`. The agent loop
runs in-process and streams SSE over localhost — there is no analytics or
telemetry channel. Credentials live in the OS keyring when available
(`0600`-permissioned file fallback) and are never echoed to event payloads or
tool-call arguments. The only opt-in egress is when a profile points at a
remote OpenAI-compatible endpoint, in which case the prompt plus the retrieved
snippets are sent — the repository, indices, and sessions are not.
Air-gapped operation is the default once an Ollama model is pulled.
