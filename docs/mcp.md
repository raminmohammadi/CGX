# MCP tool servers

CGX's agents can call external tools exposed over the
[Model Context Protocol](https://modelcontextprotocol.io) (MCP). This is how
the agent reaches beyond the local codebase -- library documentation, a real
browser, GitHub, databases, payments, error trackers -- without CGX growing a
bespoke connector for each.

Design goals:

- **Local-first.** Servers are declared in a local JSON roster; nothing is
  contacted until the agent explicitly lists or calls a server. Secrets never
  touch the roster.
- **Agent-aware.** Enabled servers are advertised to the Tech Lead and the
  Developer through the shared tool registry **with a one-line hint of when
  to use each**, so a local model knows a docs server or a browser is one call
  away instead of guessing.
- **One click.** A curated **catalog** of the MCP servers developers actually
  use is built in; add one from the MCP page or `cgx mcp add <id>`.
- **Context-frugal.** Discovery is lazy and tiered (`list_servers` ->
  `list_tools` -> `describe_tool` -> `call`), so a fleet of servers never
  floods the model's context with every tool schema up front.

## Install

The MCP SDK is an optional dependency:

```
pip install "cgx[mcp]"
```

Without it, CGX still runs; the roster, the catalog and `mcp_list_servers`
work, and the call tools return a message telling you to install the extra.
Launchers: stdio servers from the catalog run with `npx` (Node) or `uvx`
(Python's `uv`), so one of those must be on your `PATH`.

## The catalog

| id | server | category | needs |
|----|--------|----------|-------|
| `filesystem` | Filesystem (reference) | files | `ALLOWED_DIR` |
| `git` | Git (reference) | source control | `REPO_PATH` |
| `fetch` | Fetch (reference) | web | -- |
| `memory` | Memory (reference) | reasoning | optional `MEMORY_FILE_PATH` |
| `sequential-thinking` | Sequential Thinking (reference) | reasoning | -- |
| `time` | Time (reference) | utility | -- |
| `github` | GitHub (hosted) | source control | `GITHUB_PERSONAL_ACCESS_TOKEN` |
| `playwright` | Playwright (Microsoft) | browser | -- |
| `brave-search` | Brave Search | web | `BRAVE_API_KEY` |
| `context7` | Context7 (Upstash) | docs | optional `CONTEXT7_API_KEY` |
| `aws-docs` | AWS Documentation (AWS Labs) | docs | -- |
| `microsoft-learn` | Microsoft Learn (hosted) | docs | -- |
| `markitdown` | MarkItDown (Microsoft) | utility | -- |
| `postgres` | PostgreSQL (postgres-mcp, pinned to the 1.x MCP SDK until 0.3.1 ships) | database | `DATABASE_URI` |
| `sqlite` | SQLite (reference, archived) | database | `DB_PATH` |
| `mongodb` | MongoDB | database | `MDB_MCP_CONNECTION_STRING` |
| `redis` | Redis | database | `REDIS_URL` |
| `supabase` | Supabase (hosted, read-only, PAT bearer) | backend | `PROJECT_REF`, `SUPABASE_ACCESS_TOKEN` |
| `stripe` | Stripe | payments | `STRIPE_SECRET_KEY` |
| `sentry` | Sentry | observability | `SENTRY_ACCESS_TOKEN` |
| `figma` | Figma (Framelink) | design | `FIGMA_API_KEY` |
| `notion` | Notion (legacy self-hosted server; the remote successor is OAuth-only) | communication | `NOTION_TOKEN` |

Each entry records its vendor, maintenance status (`official` / `vendor` /
`community` / `archived`), what its tools can do (`read` / `write` /
`execute`), the exact launch command, the parameters and environment variables
it needs, a docs link, and an **agent hint** -- the sentence injected into the
agent's prompt (e.g. *"get the current API/usage docs for a library you are
about to use"*). The full list is `cgx mcp catalog --table` or
`GET /api/mcp/catalog`. Servers that only support an OAuth browser login are
deliberately not listed (CGX runs unattended). Every entry was fact-checked
against its npm/PyPI/GitHub page (package exists, launch form and variable
names are the documented ones, bearer/PAT auth where HTTP).

> **npm on machines with a private registry.** If your `~/.npmrc` points npm
> at a corporate registry (an expired token is a common cause of `E401`), the
> `npx`-launched servers will fail to install. Set
> `CGX_NPM_REGISTRY=https://registry.npmjs.org/` so MCP children use the public
> registry without touching your npmrc; `cgx mcp test <name>` prints this hint
> when it sees the failure.

## Adding a server

**Web UI** -- *Control -> MCP Servers*: pick a catalog card, click **Add**,
fill any parameters (an allowed directory, a project ref) and secrets (masked),
and the server appears in your roster. **Test** connects and lists its tools;
**Secrets** sets or rotates tokens; the toggle hides a server from the agent
without deleting it; **Remove** deletes it and its secrets. **Add custom
server** takes a hand-written stdio command or HTTP URL.

**CLI** (same engine functions, no server needed):

```bash
cgx mcp catalog --table                              # what is available
cgx mcp add context7                                 # one click
cgx mcp add filesystem --param ALLOWED_DIR=$HOME/src # fill a placeholder
cgx mcp add github --env GITHUB_PERSONAL_ACCESS_TOKEN=ghp_...   # secret -> keyring
cgx mcp set-env brave-search BRAVE_API_KEY=...       # set/rotate later ("VAR=" clears)
cgx mcp test context7                                # connect + list tools
cgx mcp add-custom docs --transport http --url https://docs.internal/mcp --token-env DOCS_TOKEN --agent-hint "look up our internal API docs"
cgx mcp list --table                                 # roster with READY / MISSING_ENV
cgx mcp disable playwright && cgx mcp enable playwright
cgx mcp rm sqlite
```

### Secrets

Values given on the MCP page or via `--env` / `set-env` are stored in your OS
keyring (Keychain / Secret Service / Credential Manager) under the service
`cgx-mcp`, or -- when no keyring backend exists -- in a `0600`
`~/.cgx/mcp-secrets.json`. The roster only ever contains **references**:

```json
{
  "servers": [
    {"name": "fetch", "transport": "stdio", "command": "uvx",
     "args": ["mcp-server-fetch"], "enabled": true, "catalog_id": "fetch"},
    {"name": "brave-search", "transport": "stdio", "command": "npx",
     "args": ["-y", "@brave/brave-search-mcp-server"],
     "env": {"BRAVE_API_KEY": "${BRAVE_API_KEY}"}, "enabled": true},
    {"name": "github", "transport": "http",
     "url": "https://api.githubcopilot.com/mcp/", "enabled": true,
     "auth": {"type": "bearer", "token_env": "GITHUB_PERSONAL_ACCESS_TOKEN"}}
  ]
}
```

At launch each `${VAR}` in `env`, `args`, `url` or `auth.token_env` resolves
from the CGX process environment first (so an exported variable keeps
working), then from the secret store. A server with an unresolved variable
shows **needs VAR** in the UI (`READY = false` on the CLI) and the agent's
`mcp_call` returns an actionable message instead of a cryptic failure.

Roster fields:

| Field | Meaning |
|-------|---------|
| `name` | Unique server name the agent references. |
| `transport` | `stdio` (spawn a local process) or `http` (streamable HTTP). |
| `command` / `args` | For `stdio`: the process to launch; `args` may use `${VAR}`. |
| `env` | For `stdio`: variables passed to the child, as `${VAR}` references. |
| `url` | For `http`: the server endpoint. |
| `auth` | Optional. `{"type": "bearer", "token_env": "ENV_NAME"}`. |
| `enabled` | `false` keeps a server configured but hidden from the agent. |
| `catalog_id` / `description` / `agent_hint` | Metadata the catalog filled in; `agent_hint` is what the agent is told. |

The roster path is `~/.cgx/mcp.json` (override with `CGX_MCP_CONFIG`).

## Built-in web tools vs. an MCP fetch server

The swarm's **Tech Lead and Developer** always have two built-in web tools, so
they can implement a real third-party API instead of guessing or shipping a
stub:

- `search_web {"query": "..."}` -- returns each result's title, **URL**, and
  snippet (best-effort DuckDuckGo scrape).
- `fetch_url {"url": "https://..."}` -- fetches an http(s) page and returns its
  readable text (HTML reduced to text; size capped).

**Where it's allowed to look (so it never reads the wrong place):**

- **Scheme:** http/https only.
- **SSRF hard-block (always on, not configurable):** loopback/private/link-local/
  reserved hosts are refused on the initial request *and on every redirect hop*.
- **Domain allowlist (default on):** `fetch_url` only reads from trusted
  documentation / package-registry / code-host / major-provider domains.
  `CGX_FETCH_ALLOWLIST="docs.mycorp.com"` adds domains;
  `CGX_FETCH_ALLOW_ANY=1` permits any public host (still SSRF-blocked).
- **Prompt-injection screening:** every fetched page, search result blob, MCP
  tool listing and MCP tool result runs through `cgx.guardrails.injection`.

When an MCP server is enabled the built-in `fetch_url` is dropped in favour of
`mcp_call` (the `fetch` catalog server returns clean Markdown); `search_web`
stays for URL discovery. URL-valued MCP arguments obey the same egress policy.

## How the agent uses them

Once at least one server is enabled, both roles see four tools:

- `mcp_list_servers` -- list configured servers (LOW risk).
- `mcp_list_tools {"server": "..."}` -- a server's tools, names + descriptions, cached ~5 min (LOW).
- `mcp_describe_tool {"server": "...", "tool": "..."}` -- one tool's JSON argument schema, on demand (LOW).
- `mcp_call {"server": "...", "tool": "...", "arguments": {}}` -- invoke a tool
  (HIGH risk -- gated by the approval gate when it is enabled; arguments are
  validated against the schema first).

and a prompt block listing the enabled servers with their hints:

```
MCP SERVERS AVAILABLE (external tools). Workflow: mcp_list_tools {"server": "<name>"} ...
- context7: get the current API/usage docs for a library or framework you are about to use
- playwright: open the generated app in a browser to verify it renders and behaves ...
```

## Web UI / API

The **MCP Servers** page (`/mcp`, under "Control") shows the roster and the
catalog. Backing endpoints:

- `GET /api/mcp/servers` -- roster rows with catalog metadata, `env_vars`, `missing_env`, `ready`.
- `GET /api/mcp/catalog[?category=]` -- the catalog (with `installed` flags).
- `POST /api/mcp/servers` -- add from the catalog (`catalog_id`, `params`, `env`) or a custom server.
- `POST /api/mcp/servers/{name}/env` -- store/clear secret values.
- `POST /api/mcp/servers/{name}/test` -- connect and list tools.
- `POST /api/mcp/toggle` -- enable/disable.
- `DELETE /api/mcp/servers/{name}` -- remove (and delete its secrets).

The CLI mirrors every route (`cgx mcp list|catalog|add|add-custom|rm|set-env|test|enable|disable|tools`).
