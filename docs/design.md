# Design-aware builds -- the `cgx.design` layer

Asked to *"create an app for a chatbot where the user enters an API address
and key on a profile tab, then chats on the main page (React)"*, the agent
used to ship an unstyled white page with two `<button>`s, a Profile form that
`console.log`'d the key and a Chat that posted to a hard-coded `/api/chat` --
and the Verifier called it **verified**. Nothing in the pipeline knew what a
real product looks like: the planner planned *files*, the per-file prompt was
Python import discipline plus "add nothing unasked", and verification checked
syntax, imports and `npm run build`.

`cgx.design` fixes this generally, not per-example. It gives every UI-bearing
build the judgement a senior product designer + front-end lead would apply,
encoded so that **weak local models can follow it**: deterministic selection,
curated token systems, a stylesheet CGX writes itself, hard constraints in the
prompts, and text-level checks the Verifier gates on. No browser is required.

## How it flows through the swarm

```
goal ──► build_brief() ──► DesignBrief ─┬─► Tech Lead   : planner_block() + ensure_ui_scaffolding() + plan_problems()
                                        ├─► Developer   : tokens.css written deterministically; file_block() per UI file
                                        └─► Verifier    : check_tree() ─► regen directives ─► critique_ui() ─► design gate
```

1. **Brief** (`cgx/design/brief.py`). `build_brief(goal, skills)` derives, from
   the goal text and active skills only:
   * whether the objective is UI-bearing (`applies`): frontend skills, UI
     words (page, dashboard, chat, website, form ...), or a bare "app" that is
     not explicitly server/CLI-side;
   * the **framework**: a named one (React / Vue / Next.js), server-rendered
     templates for Flask/Django/FastAPI goals, plain HTML for a website, and
     React + Vite as the default for an "app";
   * the **styling**: a generated design-tokens stylesheet, or Tailwind v4
     `@theme` tokens when Tailwind is requested;
   * the **archetype** (`archetypes.py`): chat/assistant app, admin/CRUD
     dashboard, analytics dashboard, tasks/notes tool, landing site, shop,
     form/wizard tool, account-based SaaS app, docs/blog site, developer tool,
     portfolio, browser game, or a generic web app -- each with a layout,
     screens, a component inventory, *checkable* UX requirements and
     integration rules (e.g. how a chat client talks to an OpenAI-compatible
     endpoint with a bearer key);
   * the **theme** (`themes.py`): one of nine contrast-verified design systems
     picked by keyword, archetype default and a "dark" request;
   * for data-bearing archetypes with no backend, a **sample-data module** the
     storage helper seeds on first run, so dashboards and tables render
     populated instead of `0` / `NaN%`;
   * flags (`external_api`, `secrets`, `persistence`, `multi_screen`) and the
     **canonical UI files**: tokens stylesheet, app stylesheet, HTML entry,
     entry module, app shell, API client, storage helper -- anchored on the
     plan's real UI root once the plan exists (`resolved_for_paths`).

   The brief is a plain dict persisted in the `WORK_PLAN` artifact
   (`content["design"]`) so all three roles share one spec.

2. **Tech Lead** (`swarm_tech_lead.py`). The planner prompt gains the
   *UI/UX PRODUCT SPEC* (`planner_block`): product, archetype, layout,
   screens, components, the requirements the build will be checked against and
   the exact files the plan must include. After the draft, `ensure_ui_scaffolding`
   injects any canonical file the model omitted (like `ensure_scaffolding`
   does for README/manifests), and `plan_problems` rejects a UI plan with no
   screen components. A goal that named no framework activates the React
   skill when the brief resolves to React.

3. **Developer** (`swarm_generate.py` / `engine.py`). The **tokens stylesheet
   is written by CGX, with no model call** -- a complete, professional
   baseline (`:root` palette/type/spacing/radius/shadow variables, element
   defaults, focus rings, `.topbar .nav-link .card .btn-primary .field .input
   .empty-state .spinner .table ...`, responsive rules; or a Tailwind v4 entry
   with `@theme` tokens). Every other UI file gets `file_block()` appended to
   its system prompt: product context, rules for its *kind* (entry / app shell
   / component / stylesheet / API client / storage / package.json / HTML), the
   theme as hard constraints, and an explicit *UI COMPLETENESS* rule that
   overrides the "add nothing unasked" minimalism for UI files. UI files also
   get a 1.5x output-token budget so a complete screen is not truncated.

4. **Verifier** (`swarm_verify.py`). Beside the structural scan,
   `check_tree()` runs the deterministic UI checks (below). Fatal issues become
   regen targets with an additive `REQUIRED FIX` directive, exactly like an
   import break. If the plan lost the tokens file, CGX writes it. Then one
   bounded **design critic** call (`critique_ui`, temperature 0) reviews the
   UI files against the brief and returns per-file fixes that are regenerated
   the same way, before the real build/test dry-run validates the result. A
   fatal design issue that survives the rounds fails the verdict
   (`CGX_DESIGN_GATE=0` makes it advisory) and is named in the summary
   (`design: 1 fatal issue(s) (D03-unstyled-component)`).

The greenfield path (`DECOMPOSE` -> `SCAFFOLD`) gets the same brief: the
manifest planner receives the spec and injects the canonical files, the
per-file generator receives `file_block`, the tokens file is deterministic,
and design issues ride along as advisory warnings on the scaffold artifact.

## Deterministic checks

| id | severity | what it catches |
|----|----------|-----------------|
| `D01-no-stylesheet` | fatal | no stylesheet in the tree |
| `D02-stylesheet-not-imported` / `-linked` | fatal | tokens never loaded (entry imports none; HTML links none) |
| `D03-unstyled-component` | fatal | a component renders elements with zero `className` |
| `D04-no-navigation` | fatal | multi-screen app with neither a router nor a `<nav>` |
| `D05-no-viewport` | fatal | HTML document without a viewport meta |
| `D06-legacy-render` | fatal | `ReactDOM.render` (removed in React 19 -- blank page) |
| `D07-hardcoded-api` | fatal | `fetch('/api/...')` although the user configures the endpoint |
| `D08-no-persistence` | fatal | nothing persists settings/data (no `localStorage` etc.) |
| `D09-log-only-handler` | fatal | a form submit whose only effect is `console.log` |
| `D17-stub-component` | fatal | a component whose JSX renders nothing (placeholder wrapper) |
| `D10` .. `D16` | warning | hard-coded hex colours, unlabelled inputs, fetch without loading/error state, lists without an empty state, `alert()`, placeholder copy, no `@media` rules |

Warnings never gate; they are reported (`design.issues` in the verify report)
and handed to the critic.

## Themes

| key | scheme | default for |
|-----|--------|-------------|
| `conversational` | light | chat / assistant apps |
| `calm-productivity` | light | tasks, notes, forms, wizards |
| `midnight-console` | dark | dashboards, analytics, developer tools, admin |
| `clean-tech` | light | SaaS apps, docs/blog, generic web apps |
| `warm-artisan` | light | cafés, bakeries, restaurants (landing sites) |
| `bold-dark` | dark | portfolios, creative, "neon" |
| `elegant-luxury` | dark | hospitality, fashion, jewelry |
| `fresh-natural` | light | wellness, food, eco |
| `playful-vibrant` | light | games, kids, shops |

Every palette pair used for text clears WCAG AA (there is a test for it), and
the tokens stylesheet derives a `--color-link` that always clears AA so a
decorative-only accent never becomes link text. A goal containing "dark"
steers to a dark theme.

## Configuration

| variable | default | effect |
|----------|---------|--------|
| `CGX_DESIGN_CRITIC` | `1` | run the one-call LLM design review in the Verifier |
| `CGX_DESIGN_GATE` | `1` | fatal design issues fail the verify verdict (`0` = advisory) |

## Extending

* **A new archetype**: add an `Archetype` to `ARCHETYPES` with specific
  `match_keywords` (multi-word keywords weigh double), a precise `layout`,
  `screens`, `components`, *checkable* `ux_requirements` and a
  `default_theme`.
* **A new theme**: add a `Theme` with a full palette, a Google Font pairing and
  a dense `style` paragraph; `tests/test_design_brief.py` enforces AA contrast.
* **A new check**: add a `DesignIssue` in `checks._check_tree` with an id, a
  severity, the file to regenerate and an additive directive.
* **Per-project taste**: a markdown skill (see [skills-and-context.md](skills-and-context.md))
  with `surfaces: [scaffold, plan]` can add house rules on top of the brief.

## Runtime smoke (optional browser)

Text-level checks cannot see a render-time crash (a form that calls the wrong
helper, dereferences `undefined` and unmounts the whole tree -- a blank page
behind a green build). When a Chromium-family browser is installed (Chrome,
Chromium, Edge, Brave; or `CGX_BROWSER=/path/to/binary`), the Verifier serves
the production bundle from a loopback port, loads every `<Route path>` plus `/`
headlessly (`--headless=new --dump-dom`), and treats an empty root or an
uncaught error as a failure (the served page carries a tiny collector that
mirrors uncaught errors and `console.error` into the DOM, so the real
message -- e.g. `useLocation() may be used only in the context of a <Router>`
-- reaches the repair prompt on every platform): one bounded repair round against the
route's component and its first-party imports, a rebuild, and a re-check. A
route that stays blank fails the verdict (`runtime: /settings blank/crashed`
in the summary). Screenshots of every route land in
`<project>/.cgx/screenshots/`. Nothing is downloaded; without a browser the
step is skipped and reported as such. Disable with `CGX_BROWSER_SMOKE=0`.

## Vision critic (optional local vision model)

With screenshots in hand, a local **vision model** can judge what no text check
can: a header squeezed into the content column, a composer overflowing its
card, flat hierarchy. When the active provider is Ollama and a vision-capable
model is installed (`qwen3-vl`, `llava`, `minicpm-v`, `gemma3` ... --
`CGX_VISION_MODEL=auto` picks one, or name it; `off` disables), the Verifier
shows it the brief plus up to three route screenshots and asks for concrete
per-file CSS/markup fixes, regenerates those files, rebuilds and re-runs the
smoke. One round, after the runtime smoke passed; taste never gates the
verdict, the re-check does.

## Frontend symbol intelligence

The JS/TS side is grounded like the Python side (`cgx/session/js_symbols.py`):
the Developer's `list_symbols` / `file_skeleton` show a module's real exports,
import hints are rendered for JS dependencies, and the Verifier gates on a
named/default import no dependency exports, a React or router identifier used
but never imported, a lowercase namespace called like a module but never
imported, a `<Router>` nested inside the entry's `<BrowserRouter>`, duplicate
`<Routes>` tables, and prop-contract drift (a component ignoring the props its
caller passes, or a caller omitting the props a component destructures). Each
row regenerates the right file with the real surface as its directive; the
certain cases (`react-router-dom` -> `react-router`, dangling `./X.css`
imports, missing `StrictMode`/hook imports) are fixed without a model call.

## Limits

Deterministic checks and a one-shot critic cannot judge taste. For a richer
visual review, enable the [Playwright MCP server](mcp.md): the agent can open
the built app, interact with it and screenshot it; a vision-capable model can
then critique the screenshots the smoke step already saves.
