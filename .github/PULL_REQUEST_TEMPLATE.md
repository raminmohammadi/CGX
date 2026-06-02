<!-- Thanks for contributing to CGX! Please keep PRs focused: one
feature / fix / refactor per PR. The Skills Registry is the primary
plug-and-play surface — see CONTRIBUTING.md for the protocol. -->

## Summary

<!-- One-paragraph description of the change and why. Link any related
issues with `Fixes #123` or `Refs #123`. -->

## Type of change

- [ ] New skill (adds a module under `skills/`)
- [ ] Bug fix
- [ ] New feature
- [ ] Refactor / internal cleanup
- [ ] Documentation only
- [ ] CI / build / dependency change

## Checklist

Mirrors the contributor checklist in [`CONTRIBUTING.md`](../CONTRIBUTING.md):

- [ ] `pytest -q` is green (core matrix; ML extras optional).
- [ ] `ruff check src tests` reports no new errors.
- [ ] New code paths include a test (skills, codegen, agents,
      retrieval, sessions, etc.).
- [ ] No top-level imports of `torch`, `transformers`, or
      `sentence_transformers` inside `src/cgx/` — keep them lazy
      inside function scopes so the core install stays torch-free.
- [ ] No secrets (API keys, bearer tokens) appear in commits, logs,
      SSE payloads, or test fixtures.
- [ ] Docs touched when the public surface changes (`README.md`,
      `docs/architecture.md`, `docs/usage.md`).
- [ ] If the change affects the web UI, the frontend was rebuilt
      (`cd frontend && npm run build`) so the served assets in
      `src/cgx/webui/static/` match the source.

## Screenshots / logs (optional)

<!-- Drop UI screenshots, terminal output, or SSE traces here when
they help reviewers understand the change. Never paste real API keys
or production logs — scrub anything that looks like a credential. -->
