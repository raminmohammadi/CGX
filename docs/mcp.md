# MCP tool servers

CGX's swarm agents can call external tools exposed over the
[Model Context Protocol](https://modelcontextprotocol.io) (MCP). This is how
the agent reaches beyond the local codebase -- web fetch, browser automation,
external data sources -- without CGX growing a bespoke connector for each.

Design goals:

- **Local-first.** Servers are declared in a local JSON file; nothing is
  contacted until the agent explicitly lists or calls a server.
- **Agent-aware.** Once a server is configured, the three MCP tools are
  advertised to the swarm through the shared tool registry, so the agent knows
  they exist and how to call them. No prompt editing required.
- **Easy to extend.** Adding a tool server is a one-line edit to a JSON file --
  no code change, no redeploy.
- **Context-frugal.** Discovery is lazy (`list_servers` -> `list_tools` ->
  `call`), so a large fleet of servers never floods the model's context with
  every tool schema up front.

## Install

The MCP SDK is an optional dependency:

```
pip install "cgx[mcp]"
```

Without it, CGX still runs; `mcp_list_servers` works from config, and the call
tools return a message telling you to install the extra.

## Configure servers

Create `~/.cgx/mcp.json` (override the path with the `CGX_MCP_CONFIG` env var):

```json
{
  "servers": [
    {
      "name": "fetch",
      "transport": "stdio",
      "command": "uvx",
      "args": ["mcp-server-fetch"],
      "enabled": true
    },
    {
      "name": "docs",
      "transport": "http",
      "url": "http://localhost:3000/mcp",
      "enabled": true,
      "auth": {"type": "bearer", "token_env": "DOCS_TOKEN"}
    }
  ]
}
```

Fields:

| Field       | Meaning |
|-------------|---------|
| `name`      | Unique server name the agent references. |
| `transport` | `stdio` (spawn a local process) or `http` (streamable HTTP). |
| `command` / `args` | For `stdio`: the process to launch. |
| `url`       | For `http`: the server endpoint. |
| `enabled`   | Set `false` to keep a server configured but hidden from the agent. |
| `auth`      | Optional. `{"type": "bearer", "token_env": "ENV_NAME"}` reads the token from an environment variable -- **secrets are never stored in the JSON**. |

## Built-in web tools vs. an MCP fetch server

The swarm's **Tech Lead and Developer** always have two built-in web tools, so
they can implement a real third-party API instead of guessing or shipping a
stub:

- `search_web {"query": "..."}` -- returns each result's title, **URL**, and
  snippet (best-effort DuckDuckGo scrape).
- `fetch_url {"url": "https://..."}` -- fetches an http(s) page and returns its
  readable text (HTML reduced to text; size capped). This is how the Developer
  **reads the actual docs** for, say, the Gemini or Stripe SDK and writes the
  real call.

**Where it's allowed to look (so it never reads the wrong place):**

- **Scheme:** http/https only.
- **SSRF hard-block (always on, not configurable):** loopback/private/link-local/
  reserved hosts are refused on the initial request *and on every redirect hop*
  -- the agent can't reach `localhost`, your LAN, or cloud metadata
  (`169.254.169.254`), even via a public URL that 302-redirects inward.
- **Domain allowlist (default on):** `fetch_url` only reads from trusted
  documentation / package-registry / code-host / major-provider domains
  (python.org, pypi.org, npmjs.com, developer.mozilla.org, github.com,
  stackoverflow.com, ai.google.dev, docs.anthropic.com, learn.microsoft.com,
  docs.aws.amazon.com, palletsprojects.com, react.dev, stripe.com, …). A host
  that isn't a suffix of one of these is refused with a message telling you how
  to allow it. Controls:
  - `CGX_FETCH_ALLOWLIST="docs.mycorp.com,internal-wiki.example"` -- add domains.
  - `CGX_FETCH_ALLOW_ANY=1` -- permit any public host (still SSRF-blocked).

> Fetched page text is treated as **reference data, not instructions.** Keeping
> the agent on reputable docs domains is also the main mitigation against a
> page trying to inject instructions or bad code -- widen the allowlist only to
> sources you trust.

These are dependency-free and always on, but best-effort (a scrape can miss and
`fetch_url` returns raw page text). For **more reliable retrieval**, enable an
MCP fetch/docs server -- e.g. the reference `mcp-server-fetch`, which returns
clean, readable Markdown for a URL:

```json
{
  "servers": [
    {"name": "fetch", "transport": "stdio",
     "command": "uvx", "args": ["mcp-server-fetch"], "enabled": true}
  ]
}
```

With it enabled, the agent additionally sees the `mcp_*` tools and can pull
higher-fidelity documentation. A dedicated docs server (e.g. an HTTP MCP that
indexes a library's reference) is the most robust option for a fleet that
integrates many SDKs.

> Note: reliable docs retrieval removes the *"the model didn't know the API"*
> excuse for a stub, but turning docs into correct, version-matched code still
> depends on the model's strength -- see [Agent.md](Agent.md).

## How the agent uses them

The swarm sees three tools once at least one server is enabled:

- `mcp_list_servers` -- list configured servers (LOW risk).
- `mcp_list_tools {"server": "..."}` -- list a server's tools, cached ~5 min (LOW).
- `mcp_call {"server": "...", "tool": "...", "arguments": {}}` -- invoke a tool
  (HIGH risk -- gated by the approval gate when it is enabled; see
  [Agent.md](Agent.md)).

## Web UI / API

The **MCP Servers** page (`/mcp`, under "Control" in the sidebar) lists the
configured servers, shows whether the SDK is installed and the config path, and
toggles a server on/off. Backing endpoints:

- `GET /api/mcp/servers` -- configured servers plus whether the SDK is installed.
- `POST /api/mcp/toggle` -- enable/disable a server (persisted to the JSON roster).

## Adding a new tool later

Drop another entry into `~/.cgx/mcp.json` and restart the agent. Because the
tools flow through the shared registry, no CGX code changes -- the agent simply
starts seeing the new server on its next `mcp_list_servers` call.
