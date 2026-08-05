# Contributing

Thanks for wanting to improve CGX. This page is the practical
contributor guide; the canonical source is
[`CONTRIBUTING.md`](https://github.com/raminmohammadi/Averix/blob/main/CONTRIBUTING.md) in the repo root.

---

## Ways to contribute

- **Add a Skill** — the easiest, highest-impact change. Teach CGX a new
  framework or tool by dropping one self-contained folder under
  `skills/`, with **no agent-layer edits**. Full walkthrough:
  **[[Skills Registry]]**.
- **Improve the core** — retrieval, codegen, the session loop, and the
  web UI all live under `src/cgx/`. See **[[Architecture]]**.
- **Sharpen the docs** — fixes to `README.md`, the `docs/` set, this
  wiki, or inline docstrings are always welcome.

---

## Local setup

```bash
git clone <your fork>
cd cgx
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev,codegen]"
pytest -q
```

The core install is intentionally **torch-free**. Install
`requirements-ml.txt` only when you need to exercise the embedding +
reranker stack locally.

---

## Project layout

- `src/cgx/` — runtime library (`session/`, `answer/`, `codegen/`,
  `embeddings/`, `retrieval/`, `graph/`, `parser/`, `pipeline/`,
  `webui/`, `cli/`, `io/`).
- `skills/` — pluggable per-technology bundles consumed by
  `cgx.answer.engine` and the session executors (`cgx.session.tasks`).
- `frontend/` — React/Vite SPA bundled into `src/cgx/webui/static/`.
- `extension/` — VS Code webview extension scaffold.
- `tests/` — pytest suite mirroring the `src/cgx/` package structure.

A deeper tour of each package is in **[[Architecture]]**.

---

## Running tests

```bash
pytest -q                      # full core matrix
pytest tests/test_skills_react.py -q   # a single module
ruff check src tests           # lint
```

New code paths (skills, codegen, sessions, retrieval, …) should ship
with a test. When you change the public surface, update the affected
docs (`README.md`, `docs/architecture.md`, `docs/usage.md`, and this
wiki).

---

## Pull-request checklist

Before opening a PR, confirm:

- [ ] `pytest -q` is green (core matrix; ML extras optional).
- [ ] `ruff check src tests` reports no new errors.
- [ ] New code paths include a test.
- [ ] **No top-level imports** of `torch`, `transformers`, or
  `sentence_transformers` inside `src/cgx/` — keep them lazy inside
  function scopes so the core install stays torch-free.
- [ ] **No secrets** (API keys, bearer tokens) appear in commits, logs,
  SSE payloads, or test fixtures.
- [ ] Docs touched when the public surface changes.

---

## Security when touching credentials

CGX stores credentials in the OS keyring when `keyring` is installed and
otherwise in `~/.cgx/secrets.json` with `0600` permissions. When your
change touches that path:

- Never echo a secret through a tool argument, log line, SSE payload, or
  error message.
- Scrub Gemini-style `?key=...` URLs before propagating exceptions — see
  `GeminiProvider._scrub_secret` in
  [`src/cgx/answer/providers.py`](https://github.com/raminmohammadi/Averix/blob/main/src/cgx/answer/providers.py).
- Use `os.open(..., 0o600)` (not `write_text` + `chmod`) when creating
  any file that may hold secret material — see `_write_json` in
  [`src/cgx/answer/profiles.py`](https://github.com/raminmohammadi/Averix/blob/main/src/cgx/answer/profiles.py).

Report security issues privately via the repository's **"Report a
vulnerability"** workflow rather than a public issue. More detail:
**[[Privacy and Security]]**.

---

## License

By contributing, you agree that your contribution is released under the
**MIT** license that covers the rest of the repository.

---

## See also

- **[[Skills Registry]]** — the primary contribution surface.
- **[[Architecture]]** — where each subsystem lives.
- [`CONTRIBUTING.md`](https://github.com/raminmohammadi/Averix/blob/main/CONTRIBUTING.md) — the canonical guide.
