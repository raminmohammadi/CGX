# Contributing to CGX

Thanks for your interest in CGX (Code Graph eXecution). This guide
covers the day-to-day developer workflow and the **Skills Registry** —
the primary, plug-and-play contribution surface.

## Local setup

```bash
git clone <your fork>
cd cgx
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev,codegen]"
pytest -q
```

The core install is intentionally torch-free; install
`requirements-ml.txt` only when you need to exercise the embedding +
reranker stack locally.

## Project layout

- `src/cgx/` — runtime library (`agents/`, `answer/`, `codegen/`,
  `embeddings/`, `retrieval/`, `pipeline/`, `webui/`, `cli/`).
- `skills/` — pluggable per-technology bundles consumed by
  `cgx.agents.planner`, `cgx.answer.engine`, and `cgx.agents.judge`.
- `frontend/` — React/Vite SPA bundled into `src/cgx/webui/static/`.
- `extension/` — VS Code webview extension scaffold.
- `tests/` — pytest suite (mirrors `src/cgx/` package structure).

## Adding a new Skill

A *skill* answers three questions for one technology: *does this goal
involve me?*, *what should the LLM know to do my job well?*, and *did
the produced output actually use me correctly?*. The contract lives in
[`skills/base.py`](skills/base.py).

### 1. Create the skill module

Add `skills/<name>/__init__.py` (or `skills/<name>.py` for a single-
file skill) exposing one subclass of `skills.base.Skill`:

```python
from typing import Any, Dict, List, Optional
from skills.base import Skill, SkillVerdict, file_paths, has_any_ext


class SvelteSkill(Skill):
    name = "svelte"
    role = "frontend"
    aliases = ("svelte", "sveltekit")

    def detect(self, goal: str) -> float:
        g = (goal or "").lower()
        if "sveltekit" in g:
            return 0.95
        if "svelte" in g:
            return 0.85
        return 0.0

    def scaffold_system_prompt(self) -> str:
        return (
            "SVELTE: Produce at least one `.svelte` component and a "
            "`package.json` with the `svelte` dependency. Place routes "
            "under `src/routes/` when SvelteKit is requested."
        )

    def validate_scaffold(
        self, diffs: List[Dict[str, Any]], goal: str = ""
    ) -> Optional[SkillVerdict]:
        paths = file_paths(diffs)
        if not has_any_ext(paths, (".svelte",)):
            return SkillVerdict(
                passed=False, confidence=0.9,
                rationale="No .svelte component file was produced.",
            )
        return None
```

### 2. Register the skill

Append an instance to `SKILLS` in
[`skills/__init__.py`](skills/__init__.py); also import the class at
the top of that file. The registry order only affects diagnostic
logging — multi-skill goals (*"React UI + FastAPI backend"*) activate
every matching skill.

### 3. Test the skill

Add `tests/test_skills_<name>.py` covering:

- `detect()` returns >= 0.5 for representative goals and 0.0 for
  unrelated goals.
- `scaffold_system_prompt()` is non-empty.
- `validate_scaffold()` fails on an empty diff list and passes on a
  diff payload that contains a representative file.

Run `pytest tests/test_skills_<name>.py -q` before opening a PR.

### Skill design rules

- **No agent-layer edits.** A new skill must not require changes to
  `cgx.agents.*` or `cgx.answer.engine`. If you find yourself doing
  that, the abstraction is missing — open an issue first.
- **Validators are structural, not stylistic.** A failing verdict
  should mean *"the output cannot possibly satisfy this technology"*,
  not *"the code style is wrong"*. Use `severity="warning"` for
  advisory checks (missing tests, missing README, etc.).
- **Confidence in `[0.0, 1.0]`.** Stay below 0.5 for ambiguous
  matches; the threshold lives in `SKILL_DETECT_THRESHOLD`.

## Pull-request checklist

- [ ] `pytest -q` is green (core matrix; ML extras optional).
- [ ] `ruff check src tests` reports no new errors.
- [ ] New code paths include a test (skills, codegen, agents,
  retrieval, sessions, etc.).
- [ ] No top-level imports of `torch`, `transformers`, or
  `sentence_transformers` inside `src/cgx/` — keep them lazy inside
  function scopes so the core install stays torch-free.
- [ ] No secrets (API keys, bearer tokens) appear in commits,
  logs, SSE payloads, or test fixtures.
- [ ] Docs touched when the public surface changes (`README.md`,
  `docs/architecture.md`, `docs/usage.md`).

## Security

CGX stores credentials in the OS keyring when `keyring` is installed
and otherwise in `~/.cgx/secrets.json` with `0600` permissions. When
adding code that touches that path:

- Never echo a secret value through a tool argument, log line, SSE
  payload, or error message.
- Scrub Gemini-style `?key=...` URLs before propagating exceptions —
  see `GeminiProvider._scrub_secret` in
  [`src/cgx/answer/providers.py`](src/cgx/answer/providers.py).
- Use `os.open(..., 0o600)` (not `Path.write_text` followed by
  `chmod`) when creating any file that may hold secret material; see
  `_write_json` in
  [`src/cgx/answer/profiles.py`](src/cgx/answer/profiles.py).

Report security issues privately via the repository's "Report a
vulnerability" workflow rather than opening a public issue.

## License

By contributing, you agree that your contribution will be released
under the MIT license that covers the rest of the repository.
