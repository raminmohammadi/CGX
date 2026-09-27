# Skills & the CGX.md context file

CGX lets you teach the assistant about *your* project without touching the
codebase: drop in a **markdown skill** or a repo-level **`CGX.md`** context
file, and both the chatbot (Ask) and the agents pick them up automatically.

Everything here is **local** and works with any model — including small
local ones (the default is a ~3B Ollama coder). Activation is deterministic
(keywords / always-on / an explicit pin), never model-guessed, and injected
text is size-capped so it can't crowd out the code CGX retrieves to answer.

---

## `CGX.md` — always-on project context (the `CLAUDE.md` analogue)

Create one of these at your repo (first match wins):

```
CGX.md            # preferred — commit it, share it with the team
AGENT.md
AGENTS.md
.cgx/CGX.md
.cgx/AGENT.md
```

Its contents are prepended to the assistant's instructions on **every turn**
for that project. Use it for conventions, a domain glossary, and hard
"always / never" rules:

```markdown
# Project context

## Conventions
- This is a WEX substantiation service; "claim" always means an FSA/HSA claim.
- Money is stored in integer cents, never floats.

## Always / never
- Never propose changes that touch `legacy/` — it is frozen.
- Always cite the exact file and line when explaining behavior.
```

Edit it in the UI (**Agent → Context File**) or via the API, or just commit
the file. It's mtime-cached, so edits take effect on the next question.

---

## Markdown skills (`SKILL.md`)

A skill is a folder of guidance that turns on only when it's relevant. It
plugs into the same registry as CGX's built-in technology skills, so it
shows up in the Skills library and can be pinned to a session.

### Layout

```
<skills-dir>/<name>/SKILL.md      # preferred
<skills-dir>/<name>.md            # quick single-file form
```

- **Global (personal):** `~/.cgx/skills/<name>/SKILL.md` — manage these in
  the UI (**Agent → New Skill**, "Markdown" format).
- **Per-repo (shared):** `<repo>/.cgx/skills/<name>/SKILL.md` — commit them
  so your team gets the same skills. (The `.cgx/` directory is excluded from
  the retrieval index, so these instructions never pollute search results.)

### The `SKILL.md` format

```markdown
---
name: substantiation            # optional; defaults to the folder name
description: EOB / Rx required fields and repayment flow.
triggers: [substantiation, EOB, receipt, claim, Rx]   # deterministic activation
trigger_regex: "\\bIRS\\b"      # optional regex, OR-ed with triggers
surfaces: [chat, plan]          # where it applies (see below); default: chat
roles: [backend, data]          # optional, for grouping/filtering
always_on: false                # true = active every turn, ignore triggers
priority: 0                     # higher sorts first when several match
---
Every substantiation must cite the EOB's date-of-service, provider name,
patient responsibility, and service description. A credit-card receipt is
NOT sufficient for medical claims. On denial, the participant must repay...
```

Only the body and a `name` are required; everything else has sane defaults.

### How activation works (deterministic — safe for weak models)

A skill is active for a question/goal when **any** of these hold:

1. `always_on: true`, or
2. a `triggers` keyword (whole-word, case-insensitive) or `trigger_regex`
   matches the text, or
3. you **pin** it to the session (Agent → Run / a profile).

No tool-calling and no model-side selection are involved, so even a tiny
local model uses your skills reliably. If a skill declares no triggers and
isn't `always_on`, it's **pin-only**.

### `surfaces` — where the body is injected

| surface    | reaches                                                    |
|------------|------------------------------------------------------------|
| `chat`     | the Ask chatbot and read-only agent steps (investigate, recommend) |
| `plan`     | code-change planning                                       |
| `scaffold` | greenfield project generation                              |
| `all`      | all of the above                                           |

Default is `chat`, so authoring a knowledge skill never perturbs the
JSON-strict codegen prompts unless you opt in.

### Optional: catch bad output, and bind to touched files

A codegen skill (`surfaces: [scaffold]` / `[plan]`) can go beyond steering the
prompt and **validate** the generated diffs declaratively — so it can *reject* a
bad output, not just describe the right one. All fields are optional:

```markdown
---
name: house-flask
surfaces: [scaffold, plan]
forbid_files: ["package.json", "*.env"]          # no diff file may match
require_files: ["backend/extensions.py"]         # scaffold must include one
forbid_patch_regex: ["from .*app import .*db"]   # no diff body may match
require_patch_regex: ["create_app"]              # some diff body must match
validate_surfaces: [scaffold]                    # default: this skill's non-chat surfaces
context_globs: ["billing/**", "*.tf"]            # activate when a task touches these
context_exts: [".css", ".scss"]                  # ...or files with these extensions
---
```

- **File checks** run over the diff paths; **patch-regex checks** over the diff
  bodies. A violation is a fatal verdict that drives a corrective regenerate,
  the same as a built-in skill's validator. (`require_*` runs on a full
  scaffold; on an incremental **plan** edit only `forbid_files` applies, so a
  small edit is never rejected for lacking a required file.)
- `context_globs` / `context_exts` are **condition-bound activation**: the skill
  turns on when the current task touches a matching file, even without a goal
  keyword — useful for a "when working under `billing/`" or "when editing CSS"
  house rule.
- Regexes are validated when you save the skill; an uncompilable pattern is
  rejected with a clear error.

---

## Managing skills & context via the API

```
GET    /api/skills                       # built-in + custom (Python + markdown)
GET    /api/skills/{name}/source
POST   /api/skills/markdown              # { content, name? }  create markdown skill
PUT    /api/skills/markdown/{name}       # { content }         update
DELETE /api/skills/{name}                # delete a custom skill (either format)

GET    /api/context-file?project_root=…  # read CGX.md
PUT    /api/context-file                 # { project_root, content }  create/replace
DELETE /api/context-file?project_root=…
```

Markdown skills and `CGX.md` are validated statically (frontmatter fields,
size, name uniqueness) — **no code is executed**, unlike the Python skill
format, which runs in a sandboxed probe. That makes markdown the
recommended format for most users.

---

## Notes & limits

- Injected instructions are capped and their length is subtracted from the
  retrieval budget, so a long `CGX.md`/skill can't silently truncate the
  citations CGX needs on a small-`num_ctx` model. Keep them tight.
- `CGX.md` injection into the JSON-strict scaffold/plan codegen prompts is
  intentionally conservative; use a skill with `surfaces: [scaffold]` /
  `[plan]` to guide code generation specifically.
- Python skills (`~/.cgx/skills/*.py`) still work and are unchanged; see the
  [architecture docs](architecture.md#skills). Markdown is simply the
  code-free alternative.
