# Self-Testing Code Generation

CGX does not just emit diffs and hope. When you ask for a change plan, it
can **parse, dry-apply, syntax-check, and test** its own output in a
sandbox before you ever see it — and retry with the concrete failures as
feedback. This is what backs the **Plan** tab's *Validate diffs* / *Run
impacted tests* options, the `cgx plan --self-test --run-tests` flags, and
the agent's `PLAN_CHANGE` / `SCAFFOLD` tasks.

---

## The self-test loop

When *Validate diffs* is on (or `self_test=True` is passed to
`generate_code_plan`), CGX runs `validate_and_test`:

1. **Parse** fenced ```` ```diff path=... ```` blocks from the model
   output (`parse_fenced_diffs`).
2. **Dry-apply** each diff in memory (`apply_diffs_in_memory`) — projecting
   the patch onto the current file without touching disk.
3. **Syntax-gate** the projected files (`validate_patch_results`):
   `ast.parse` over Python targets, and a tree-sitter parse over JS / TS /
   TSX targets (skipped gracefully without the `parsers` extra). Grounding
   the check in a real parser — not the model's self-report — keeps
   quality flat across providers.
4. **Run impacted tests** (when enabled): copy the project into a
   temporary sandbox, materialise the diffs, and run `pytest` scoped to
   the impacted files with a timeout.
5. **Retry once** on failure: `build_retry_feedback` summarises the
   breakage and the loop re-asks the model with that feedback.

The full report is attached to the result as `codegen_report` and
rendered under the plan in the UI as a markdown table.

---

## Granular error slicing

Retry prompts include **±5 lines of source context** around the first
traceback line number rather than a raw 1,200-character pytest dump. That
keeps small local models focused on the precise failure site instead of
drowning in log noise.

---

## Dynamic dependency management

A generated file may import a package the model chose that is not in
`requirements.txt`. Before `pytest` runs, `cgx.codegen.env_manager`:

1. **Scans imports** in the generated files (AST for `.py`, regex for
   `.js`/`.ts`).
2. **Finds missing packages** by cross-referencing `requirements.txt`,
   probing live importability, and skipping the CPython stdlib.
3. **Installs** each missing package with `pip install --quiet`.
4. **Updates `requirements.txt`** idempotently when tests pass.

Failures are logged but never abort the run — a misspelled package still
lets pytest produce a real `ImportError` for the retry loop to diagnose.

---

## Structured AST insertion (opt-in)

For the common "insert a new function into this container after this
sibling" case, `cgx.codegen.ast_insert` offers an **additive,
AST-anchored** alternative to text diffs:

- `AstInsertSpec(rel_path, code, class_name=None, anchor_symbol=None)`
  declares the target and snippet.
- `plan_ast_insertion` parses the file, locates the anchor sibling,
  detects the container's indentation, re-emits the snippet via
  `ast.get_source_segment` (so comments/formatting survive), splices it
  in, and **re-parses** — a broken splice returns `ok=False` rather than a
  corrupted file. Nothing is written to disk.
- `build_unified_diff` renders the plan as a standard unified diff so it
  routes back through the same parse/apply/validate path.

The module is purely additive; the text-diff path is untouched.

---

## Apply safeguards

When a plan is finally applied to disk (via the agent's `APPLY` task or
the disk-apply path):

- Every overwrite is mirrored under `<project_root>/.cgx-backups/<run_id>/`.
- The whole run is reversible via `POST /api/rollback` (the UI **Undo**
  button).
- Any file whose projected source does not parse is **dropped** and
  recorded in `failed_files` rather than written broken — in greenfield
  mode this drives a targeted re-scaffold. See **[[Session Based Agent]]**.

---

## Try it

```bash
cgx plan "Add a --json flag to the query command" --self-test --run-tests
```

Or tick **Validate diffs** + **Run impacted tests** in the Plan tab.

---

## See also

- **[[Session Based Agent]]** — where codegen runs inside a task DAG.
- **[[Skills Registry]]** — structural validators that gate scaffolds.
- [`docs/architecture.md` § Self-test loop](https://github.com/raminmohammadi/Averix/blob/main/docs/architecture.md#self-test-loop).
