# Skills Registry

A **skill** is a self-contained bundle of everything CGX knows about one
technology (framework, runtime, library, build tool). Skills make
scaffolding and planning **technology-aware** — and adding support for a
new framework is a **single-folder change** with no agent-layer edits.
This is the easiest, highest-impact way to contribute.

Each skill answers three questions for its technology:

1. **Does this goal involve me?** — `detect(goal) -> float`
2. **What should the LLM know to do my job well?** —
   `scaffold_system_prompt()` / `plan_system_prompt()`
3. **Did the produced output actually use me correctly?** —
   `validate_scaffold(diffs, goal)` / `validate_plan(diffs, goal)`

The contract lives in [`skills/base.py`](https://github.com/raminmohammadi/CGX/blob/main/skills/base.py).

---

## Built-in skills

| Role | Skills |
|------|--------|
| Frontend | `react`, `nextjs`, `vue` |
| Backend (Python) | `fastapi`, `flask`, `django` |
| Backend (Node) | `express` |
| CLI / scripts | `python_cli` |
| Data layer | `sqlite` |
| Styling | `tailwind` |

Multi-skill goals compose naturally — *"React UI + FastAPI backend"*
activates both, so the scaffold prompt carries both layouts and both
validators run.

---

## How skills are consumed

- `cgx.answer.engine` calls `detect_skills(goal)` and composes the
  matching skills' prompt fragments into the scaffold / plan system
  prompt.
- `cgx.session.tasks.scaffold` runs `validate_scaffold` over the produced
  diffs. A **fatal** verdict (e.g. a React goal that emitted no JS/TS
  source) drives a whole-tree regenerate rather than silently applying a
  wrong-shaped output.
- `cgx.session.tasks.plan_change` runs `validate_plan` and surfaces the
  verdict at the approval gate.
- The **[[Swarm Agent]]**'s Tech Lead is skill-aware: it resolves the goal's
  skills (auto-detected, or an explicit session / Agent-Profile pin) and
  injects their guidance, so a *"Flask + React"* goal plans **both**
  components. Each active skill's `validate_plan` can **veto** a plan that
  omits its required files.

Detection is scored: a skill's `detect(goal)` must meet
`SKILL_DETECT_THRESHOLD` to activate. Verdicts marked
`severity="warning"` are advisory (missing tests/README) and do not fail
the scaffold; only non-warning failures do.

---

## Two ways to ship a skill

- **Built-in** — a folder under `skills/`, registered in
  `skills/__init__.py`. Contribute these upstream.
- **Private / local-only** — drop a single `Skill` subclass in a `.py`
  file under `~/.cgx/skills/`. It is discovered at runtime by
  [`skills/loader.py`](https://github.com/raminmohammadi/CGX/blob/main/skills/loader.py) and participates in detection,
  prompt composition, and validation exactly like a built-in — no repo
  edits required.

---

## Author a new skill

### 1. Create the module

Add `skills/<name>/__init__.py` (a single-file `skills/<name>.py` also
works) exposing one `Skill` subclass:

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

### 2. Register it (built-in only)

Import the class at the top of [`skills/__init__.py`](https://github.com/raminmohammadi/CGX/blob/main/skills/__init__.py)
and append an instance to `SKILLS`. Registry order only affects
diagnostic logging.

### 3. Test it

Add `tests/test_skills_<name>.py` covering:

- `detect()` returns ≥ 0.5 for representative goals and 0.0 for unrelated
  ones,
- `scaffold_system_prompt()` is non-empty,
- `validate_scaffold()` fails on an empty diff list and passes on a diff
  payload containing a representative file.

```bash
pytest tests/test_skills_<name>.py -q
```

---

## Design rules

- **No agent-layer edits.** A new skill must not require changes to
  `cgx.session.*` or `cgx.answer.engine`. If it does, the abstraction is
  missing — open an issue first.
- **Validators are structural, not stylistic.** A failing verdict should
  mean *"the output cannot possibly satisfy this technology"*, not *"the
  style is wrong"*. Use `severity="warning"` for advisory checks.
- **Confidence in `[0.0, 1.0]`.** Stay below 0.5 for ambiguous matches.

---

## See also

- **[[Contributing]]** — the full PR checklist.
- **[[Session Based Agent]]** — where scaffold validation runs.
- [`CONTRIBUTING.md`](https://github.com/raminmohammadi/CGX/blob/main/CONTRIBUTING.md#adding-a-new-skill).
