# Site Studio — build a website by describing it

**Site Studio** (nav: **Build → Site Studio**, route `/studio`) turns a
plain-English description into a working static **HTML/CSS/JS** website, renders
it **live** in a sandboxed preview beside the build, and lets you refine it by
giving feedback — all locally, on any model (small local ones included). No
framework, no build step, and the output is **real files on disk**.

For framework sites (React / Vue / Next), use the **[[Session Based Agent]]**
with the matching skill instead — Site Studio is specifically for plain static
sites.

---

## The loop

1. **Describe.** Give the site a name and say what it should contain — *"a
   coffee-shop site with menu, story, hours and a contact page; dark, modern,
   responsive."* The folder is created for you under `~/.cgx/sites/<slug>` — no
   path to type. (You can still point the Agent Loop at any absolute path.)
2. **Collaborate.** CGX asks a few clarifying questions (pages, style, sections)
   and shows you the file plan to **approve** before it writes anything.
3. **Build.** It generates a coherent multi-page site — `index.html` plus a
   page per route, one shared `css/style.css` and `js/script.js`, an identical
   header/footer on every page, and relative links that only point at pages
   that exist.
4. **Render.** The site appears live in a sandboxed preview pane (desktop or
   mobile width), reloading automatically each time files are written.
5. **Feedback → fix.** Type *"make the header sticky"*, *"use a blue hero"*,
   *"add a pricing page"* in the feedback box; CGX revises the files and the
   preview reloads. Repeat until you're happy.

The generated files are ordinary files in the site's folder — open, edit,
deploy, or commit them like any other static site.

---

## How it stays correct (even on small models)

- A built-in **`static_site` skill** steers generation to a build-less
  multi-page layout and *validates* the result: it requires an `index.html`,
  rejects stray framework/build files (React/Vite/`package.json`), and resolves
  every local `href`/`src`/`link` to a file the site actually includes — the
  number-one cause of a "finished" site being broken. A failed check drives a
  targeted regenerate automatically.
- Activation is **deterministic** (the skill is pinned for Studio builds), so
  even a small local model produces a real static site rather than a React app.
  See **[[Skills Registry]]** for how skills work.
- The site never gets a pointless test file, `package.json`, or bundler.

## Preview safety

The preview is served **read-only** from the site's folder over a loopback
route that (a) refuses to leave the project directory (path-containment guard)
and (b) only serves web content (`.html/.css/.js/.svg`/images/fonts), so it can
never read other project files. It renders inside a **sandboxed iframe**
(`allow-scripts`, *not* `allow-same-origin`), so generated JavaScript runs but
is isolated in its own origin — it cannot touch the CGX app, its cookies, or its
API. See **[[Privacy and Security]]**.

## Notes

- Site Studio drives the normal greenfield agent loop with the `static_site`
  skill pinned and plan-approval on, so anything the agent loop can do (cancel,
  resume, inspect the task graph) applies here too.
- Rendering uses **your** browser (the sandboxed iframe) — no headless browser
  or extra dependency is installed.

---

## Deep dive

- [`docs/site-studio.md`](https://github.com/raminmohammadi/CGX/blob/main/docs/site-studio.md)
  — the authoritative reference.
