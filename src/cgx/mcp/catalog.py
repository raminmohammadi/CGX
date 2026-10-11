"""Curated catalog of MCP servers software developers actually use.

The MCP page offers these as one-click "Add" cards; ``cgx mcp add <id>`` does
the same from the CLI. Each entry is a *template*: the launch command (stdio)
or URL (http), the parameters the user must supply (an allowed directory, a
database URL ...), the environment variables it needs (secrets are stored via
:mod:`cgx.mcp.secrets`, never in the roster), and an ``agent_hint`` -- one
line telling the coding agent WHEN to reach for the server, which CGX injects
into the Tech Lead / Developer prompts so a local model knows a docs server or
a browser is available instead of guessing.

Every entry names a real, maintained package; ``status`` distinguishes the
official reference servers, vendor-maintained servers, popular community
servers and archived packages (kept only when still installable and useful).
``${PARAM}`` placeholders in ``args``/``url``/``env`` are filled from the
user's parameters at add-time; ``${ENV_VAR}`` references stay in the roster and
are resolved at launch from the environment or the secret store.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["CatalogEntry", "CATALOG", "CATEGORIES", "catalog_entries", "catalog_entry",
           "instantiate", "missing_params"]

CATEGORIES: Dict[str, str] = {
    "filesystem": "Files & workspace",
    "vcs": "Source control",
    "web": "Web search & fetch",
    "browser": "Browser automation",
    "docs": "Library & platform docs",
    "database": "Databases",
    "backend": "Backend platforms",
    "observability": "Errors & monitoring",
    "design": "Design",
    "payments": "Payments",
    "communication": "Team communication",
    "reasoning": "Reasoning & memory",
    "utility": "Utilities",
}


@dataclass
class CatalogEntry:
    id: str
    name: str
    vendor: str
    category: str
    description: str
    agent_hint: str
    transport: str = "stdio"                 # stdio | http
    command: str = ""
    args: List[str] = field(default_factory=list)
    url: str = ""
    env: List[Dict[str, Any]] = field(default_factory=list)       # {name, description, secret, required}
    params: List[Dict[str, Any]] = field(default_factory=list)    # {name, description, placeholder, required}
    auth: Dict[str, Any] = field(default_factory=dict)            # http: {"type": "bearer", "token_env": "..."}
    docs_url: str = ""
    status: str = "official"                  # official | vendor | community | archived
    risk: str = "read"                        # read | write | execute  (what the tools can do)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


CATALOG: List[CatalogEntry] = [
    # ---- official reference servers (modelcontextprotocol/servers) --------------
    CatalogEntry(
        id="filesystem", name="Filesystem", vendor="Model Context Protocol (reference)", category="filesystem",
        description="Read, search and edit files under directories you allow.",
        agent_hint="read or edit files outside the project tree the user explicitly allowed (configs, docs, data)",
        command="npx", args=["-y", "@modelcontextprotocol/server-filesystem", "${ALLOWED_DIR}"],
        params=[{"name": "ALLOWED_DIR", "description": "Directory the server may access (absolute path)",
                 "placeholder": "/Users/me/projects", "required": True}],
        docs_url="https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem", status="official",
        risk="write"),
    CatalogEntry(
        id="git", name="Git", vendor="Model Context Protocol (reference)", category="vcs",
        description="Inspect and operate a local Git repository: status, diff, log, branches, commits.",
        agent_hint="inspect repository history/diffs or stage and commit changes in the local repo",
        command="uvx", args=["mcp-server-git", "--repository", "${REPO_PATH}"],
        params=[{"name": "REPO_PATH", "description": "Path of the Git repository", "placeholder": "/Users/me/projects/app",
                 "required": True}],
        docs_url="https://github.com/modelcontextprotocol/servers/tree/main/src/git", status="official", risk="write"),
    CatalogEntry(
        id="fetch", name="Fetch", vendor="Model Context Protocol (reference)", category="web",
        description="Fetch a web page and return it as clean Markdown (readable docs, release notes, READMEs).",
        agent_hint="read the real documentation page of a library/API before implementing against it",
        command="uvx", args=["mcp-server-fetch"],
        docs_url="https://github.com/modelcontextprotocol/servers/tree/main/src/fetch", status="official", risk="read"),
    CatalogEntry(
        id="memory", name="Memory", vendor="Model Context Protocol (reference)", category="reasoning",
        description="A persistent knowledge-graph memory (entities, relations, observations) across sessions.",
        agent_hint="remember or recall project facts, decisions and user preferences across runs",
        command="npx", args=["-y", "@modelcontextprotocol/server-memory"],
        env=[{"name": "MEMORY_FILE_PATH", "description": "Optional path of the JSONL memory file (default lives inside the "
              "npx cache and is not durable -- set e.g. ~/.cgx/mcp-memory.jsonl)", "secret": False, "required": False}],
        docs_url="https://github.com/modelcontextprotocol/servers/tree/main/src/memory", status="official", risk="write"),
    CatalogEntry(
        id="sequential-thinking", name="Sequential Thinking", vendor="Model Context Protocol (reference)",
        category="reasoning",
        description="A structured step-by-step reasoning scratchpad for breaking down hard problems.",
        agent_hint="plan a multi-step change or debug a hard failure before writing code",
        command="npx", args=["-y", "@modelcontextprotocol/server-sequential-thinking"],
        docs_url="https://github.com/modelcontextprotocol/servers/tree/main/src/sequentialthinking",
        status="official", risk="read"),
    CatalogEntry(
        id="time", name="Time", vendor="Model Context Protocol (reference)", category="utility",
        description="Current time and timezone conversions.",
        agent_hint="get the current date/time or convert between timezones",
        command="uvx", args=["mcp-server-time"],
        docs_url="https://github.com/modelcontextprotocol/servers/tree/main/src/time", status="official", risk="read"),
    # ---- source control / issues ---------------------------------------------
    CatalogEntry(
        id="github", name="GitHub", vendor="GitHub", category="vcs",
        description="Repositories, issues, pull requests, code search and Actions via GitHub's hosted MCP server.",
        agent_hint="look up or create GitHub issues/PRs, read files from other repos, search code on GitHub",
        transport="http", url="https://api.githubcopilot.com/mcp/",
        auth={"type": "bearer", "token_env": "GITHUB_PERSONAL_ACCESS_TOKEN"},
        env=[{"name": "GITHUB_PERSONAL_ACCESS_TOKEN", "description": "Fine-grained personal access token with the repo scopes you want to allow",
              "secret": True, "required": True}],
        docs_url="https://github.com/github/github-mcp-server", status="vendor", risk="write"),
    # ---- web / browser / docs ------------------------------------------------
    CatalogEntry(
        id="playwright", name="Playwright", vendor="Microsoft", category="browser",
        description="Drive a real browser: open pages, click, fill forms, take screenshots, read the accessibility tree.",
        agent_hint="open the generated app in a browser to verify it renders and behaves, or scrape a page that needs JS",
        command="npx", args=["-y", "@playwright/mcp@latest"],
        docs_url="https://github.com/microsoft/playwright-mcp", status="vendor", risk="execute"),
    CatalogEntry(
        id="brave-search", name="Brave Search", vendor="Brave", category="web",
        description="Web and local search via the Brave Search API.",
        agent_hint="search the web for current documentation, error messages or package names",
        command="npx", args=["-y", "@brave/brave-search-mcp-server", "--transport", "stdio"],
        env=[{"name": "BRAVE_API_KEY", "description": "Brave Search API key", "secret": True, "required": True}],
        docs_url="https://github.com/brave/brave-search-mcp-server", status="vendor", risk="read"),
    CatalogEntry(
        id="context7", name="Context7", vendor="Upstash", category="docs",
        description="Version-specific, up-to-date documentation and code examples for thousands of libraries.",
        agent_hint="get the current API/usage docs for a library or framework you are about to use",
        command="npx", args=["-y", "@upstash/context7-mcp"],
        env=[{"name": "CONTEXT7_API_KEY", "description": "Optional API key for higher rate limits", "secret": True,
              "required": False}],
        docs_url="https://github.com/upstash/context7", status="vendor", risk="read"),
    CatalogEntry(
        id="aws-docs", name="AWS Documentation", vendor="AWS Labs", category="docs",
        description="Search and read AWS documentation pages as Markdown.",
        agent_hint="look up an AWS service API, limit or configuration detail",
        command="uvx", args=["awslabs.aws-documentation-mcp-server@latest"],
        docs_url="https://awslabs.github.io/mcp/servers/aws-documentation-mcp-server", status="vendor", risk="read"),
    CatalogEntry(
        id="microsoft-learn", name="Microsoft Learn", vendor="Microsoft", category="docs",
        description="Search official Microsoft/Azure/.NET documentation on Microsoft Learn.",
        agent_hint="look up .NET, Azure, TypeScript or Windows platform documentation",
        transport="http", url="https://learn.microsoft.com/api/mcp",
        docs_url="https://github.com/MicrosoftDocs/mcp", status="vendor", risk="read"),
    CatalogEntry(
        id="markitdown", name="MarkItDown", vendor="Microsoft", category="utility",
        description="Convert PDFs, Office documents, images and HTML into Markdown the model can read.",
        agent_hint="read a PDF/DOCX/XLSX spec or design document the user provided",
        command="uvx", args=["markitdown-mcp"],
        docs_url="https://github.com/microsoft/markitdown/tree/main/packages/markitdown-mcp", status="vendor",
        risk="read"),
    # ---- databases ------------------------------------------------------------
    CatalogEntry(
        id="postgres", name="PostgreSQL", vendor="Crystal DBA (community)", category="database",
        description="Query a PostgreSQL database, explore its schema, explain plans and analyse index health.",
        agent_hint="inspect the real schema or run read-only SQL against the project's Postgres database",
        # PyPI 0.3.0 declares mcp[cli]>=1.5 with no upper bound and breaks on the
        # 2.x SDK (upstream fix merged 2026-08, unreleased); the --with pin is the
        # documented workaround until 0.3.1 ships.
        command="uvx", args=["--with", "mcp[cli]<2", "postgres-mcp", "--access-mode=restricted"],
        env=[{"name": "DATABASE_URI", "description": "postgresql://user:pass@host:5432/db", "secret": True,
              "required": True}],
        docs_url="https://github.com/crystaldba/postgres-mcp", status="community", risk="read"),
    CatalogEntry(
        id="sqlite", name="SQLite", vendor="Model Context Protocol (reference, archived)", category="database",
        description="Query and modify a local SQLite database file.",
        agent_hint="inspect or query the project's local SQLite database file",
        command="uvx", args=["mcp-server-sqlite", "--db-path", "${DB_PATH}"],
        params=[{"name": "DB_PATH", "description": "Path to the .db file", "placeholder": "./data/app.db", "required": True}],
        docs_url="https://github.com/modelcontextprotocol/servers-archived/tree/main/src/sqlite", status="archived",
        risk="write"),
    CatalogEntry(
        id="mongodb", name="MongoDB", vendor="MongoDB", category="database",
        description="Explore collections, run queries and aggregations against MongoDB / Atlas.",
        agent_hint="inspect collections or run queries against the project's MongoDB",
        command="npx", args=["-y", "mongodb-mcp-server@latest", "--readOnly"],
        env=[{"name": "MDB_MCP_CONNECTION_STRING", "description": "mongodb+srv://user:pass@cluster/db (passed via env, "
              "never on the command line)", "secret": True, "required": True}],
        docs_url="https://github.com/mongodb-js/mongodb-mcp-server", status="vendor", risk="read"),
    CatalogEntry(
        id="redis", name="Redis", vendor="Redis", category="database",
        description="Read and write keys, hashes, lists, streams and vectors in a Redis instance.",
        agent_hint="inspect cache/queue state in the project's Redis",
        command="uvx", args=["--from", "redis-mcp-server@latest", "redis-mcp-server", "--url", "${REDIS_URL}"],
        env=[{"name": "REDIS_URL", "description": "redis://user:pass@host:6379/0", "secret": True, "required": True}],
        docs_url="https://github.com/redis/mcp-redis", status="vendor", risk="write"),
    # ---- backend platforms / payments / observability --------------------------
    CatalogEntry(
        id="supabase", name="Supabase", vendor="Supabase", category="backend",
        description="Inspect a Supabase project (read-only): tables, SQL, migrations, edge functions, logs, docs.",
        agent_hint="inspect the app's Supabase schema, policies and functions",
        transport="http", url="https://mcp.supabase.com/mcp?project_ref=${PROJECT_REF}&read_only=true",
        auth={"type": "bearer", "token_env": "SUPABASE_ACCESS_TOKEN"},
        params=[{"name": "PROJECT_REF", "description": "Supabase project ref (from the project URL)", "placeholder": "abcdefghijklmnop",
                 "required": True}],
        env=[{"name": "SUPABASE_ACCESS_TOKEN", "description": "Scoped personal access token (dashboard -> Account -> Access Tokens)",
              "secret": True, "required": True}],
        docs_url="https://supabase.com/docs/guides/getting-started/mcp", status="vendor", risk="read"),
    CatalogEntry(
        id="stripe", name="Stripe", vendor="Stripe", category="payments",
        description="Create and inspect customers, products, prices, payment links and subscriptions; search Stripe docs.",
        agent_hint="implement or debug a Stripe integration against the real (test-mode) account and docs",
        command="npx", args=["-y", "@stripe/mcp", "--api-key=${STRIPE_SECRET_KEY}"],
        env=[{"name": "STRIPE_SECRET_KEY", "description": "Restricted or test-mode secret key (sk_test_...)",
              "secret": True, "required": True}],
        docs_url="https://github.com/stripe/ai/tree/main/tools/modelcontextprotocol", status="vendor",
        risk="write"),
    CatalogEntry(
        id="sentry", name="Sentry", vendor="Sentry", category="observability",
        description="Query issues, events, stack traces and releases from your Sentry organisation.",
        agent_hint="pull the real stack trace for a production error before fixing it",
        command="npx", args=["-y", "@sentry/mcp-server@latest", "--access-token=${SENTRY_ACCESS_TOKEN}"],
        env=[{"name": "SENTRY_ACCESS_TOKEN", "description": "User auth token (org:read, project:read, event:read)",
              "secret": True, "required": True},
             {"name": "SENTRY_HOST", "description": "Self-hosted Sentry hostname (optional)", "secret": False,
              "required": False}],
        docs_url="https://github.com/getsentry/toolkit", status="vendor", risk="read"),
    # ---- design / communication -------------------------------------------------
    CatalogEntry(
        id="figma", name="Figma (Framelink)", vendor="Framelink (community)", category="design",
        description="Read layout, component and style data from a Figma file so the agent can implement a design faithfully.",
        agent_hint="implement a screen from a Figma design the user linked",
        command="npx", args=["-y", "figma-developer-mcp", "--figma-api-key=${FIGMA_API_KEY}", "--stdio"],
        env=[{"name": "FIGMA_API_KEY", "description": "Figma personal access token", "secret": True, "required": True}],
        docs_url="https://github.com/GLips/Figma-Context-MCP", status="community", risk="read"),
    CatalogEntry(
        id="notion", name="Notion", vendor="Notion", category="communication",
        description="Search and read Notion pages and databases; create and update pages. Legacy self-hosted server: "
                    "Notion now recommends its OAuth-only Remote MCP (not usable unattended); this token-based package "
                    "still works but is no longer actively maintained.",
        agent_hint="read a spec, requirements doc or task list the team keeps in Notion",
        command="npx", args=["-y", "@notionhq/notion-mcp-server"],
        env=[{"name": "NOTION_TOKEN", "description": "Internal integration token (ntn_...)", "secret": True,
              "required": True}],
        docs_url="https://github.com/makenotion/notion-mcp-server", status="archived", risk="write"),
]


def catalog_entries(category: Optional[str] = None) -> List[Dict[str, Any]]:
    """Catalog as plain dicts (optionally one category)."""
    return [e.to_dict() for e in CATALOG if not category or e.category == category]


def catalog_entry(entry_id: str) -> Optional[CatalogEntry]:
    for e in CATALOG:
        if e.id == (entry_id or "").strip():
            return e
    return None


def missing_params(entry: CatalogEntry, params: Dict[str, str]) -> List[str]:
    """Required ``${PARAM}`` placeholders the caller did not supply."""
    return [p["name"] for p in entry.params
            if p.get("required") and not str((params or {}).get(p["name"]) or "").strip()]


def _fill(text: str, params: Dict[str, str]) -> str:
    out = text
    for k, v in (params or {}).items():
        out = out.replace("${" + k + "}", str(v))
    return out


def instantiate(entry: CatalogEntry, *, name: Optional[str] = None,
                params: Optional[Dict[str, str]] = None,
                enabled: bool = True,
                env_values: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """A roster entry (``mcp.json`` server dict) for ``entry`` with params filled.

    ``${PARAM}`` placeholders are replaced now; ``${ENV_VAR}`` references are
    kept for the entry's REQUIRED env vars plus any optional one the user
    supplied a value for (``env_values``), so the value is resolved at launch
    from the environment or the secret store (never written to the roster). An
    optional variable nobody set is simply not referenced, so it can never show
    up as "missing".
    """
    params = dict(params or {})
    supplied = {k for k, v in (env_values or {}).items() if str(v or "").strip()}
    env_names = {e["name"] for e in entry.env if e.get("required") or e["name"] in supplied}
    server: Dict[str, Any] = {
        "name": (name or entry.id).strip(),
        "transport": entry.transport,
        "enabled": bool(enabled),
        "catalog_id": entry.id,
        "description": entry.description,
        "agent_hint": entry.agent_hint,
    }
    if entry.transport == "http":
        server["url"] = _fill(entry.url, params)
        if entry.auth:
            server["auth"] = dict(entry.auth)
    else:
        server["command"] = entry.command
        server["args"] = [_fill(a, params) for a in entry.args]
        # Declared env vars are passed to the child process by reference.
        server["env"] = {n: "${" + n + "}" for n in sorted(env_names)}
    return server
