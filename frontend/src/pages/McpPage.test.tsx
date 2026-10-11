import { describe, expect, it, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

vi.mock("../lib/api", () => {
  const servers = {
    sdk_installed: true,
    config_path: "/home/u/.cgx/mcp.json",
    servers: [
      {
        name: "github", transport: "http", enabled: true, url: "https://api.githubcopilot.com/mcp/",
        catalog_id: "github", description: "GitHub", agent_hint: "look up issues",
        catalog: { id: "github", name: "GitHub", vendor: "GitHub", category: "vcs", category_label: "Source control",
                   docs_url: "https://x", status: "vendor", risk: "write",
                   env: [{ name: "GITHUB_PERSONAL_ACCESS_TOKEN", description: "PAT", secret: true, required: true }] },
        env_vars: ["GITHUB_PERSONAL_ACCESS_TOKEN"], missing_env: ["GITHUB_PERSONAL_ACCESS_TOKEN"], ready: false,
      },
    ],
  };
  const catalog = {
    categories: { web: "Web search & fetch", docs: "Library & platform docs" },
    config_path: "/home/u/.cgx/mcp.json",
    sdk_installed: true,
    servers: [
      { id: "fetch", name: "Fetch", vendor: "MCP", category: "web", description: "Fetch pages", agent_hint: "read docs",
        transport: "stdio", command: "uvx", args: ["mcp-server-fetch"], env: [], params: [], docs_url: "https://d",
        status: "official", risk: "read", installed: false },
      { id: "context7", name: "Context7", vendor: "Upstash", category: "docs", description: "Library docs",
        agent_hint: "get current API docs", transport: "stdio", command: "npx", args: ["-y", "@upstash/context7-mcp"],
        env: [{ name: "CONTEXT7_API_KEY", description: "optional", secret: true, required: false }], params: [],
        docs_url: "https://c", status: "vendor", risk: "read", installed: true },
    ],
  };
  return {
    api: {
      mcpServers: vi.fn(async () => servers),
      mcpCatalog: vi.fn(async () => catalog),
      mcpAdd: vi.fn(async (body: any) => ({ ...body, env_vars: [], missing_env: [], ready: true })),
      mcpRemove: vi.fn(async () => ({ ok: true, removed_secrets: 0 })),
      mcpSetEnv: vi.fn(async () => ({ ok: true, stored: {}, missing_env: [] })),
      mcpTest: vi.fn(async () => ({ ok: false, tools: [], error: "missing value(s) for GITHUB_PERSONAL_ACCESS_TOKEN", missing_env: ["GITHUB_PERSONAL_ACCESS_TOKEN"], sdk_installed: true })),
      mcpToggle: vi.fn(async () => ({ ok: true })),
    },
  };
});

import McpPage, { buildAddBody, missingRequired } from "./McpPage";
import { api } from "../lib/api";

describe("McpPage", () => {
  beforeEach(() => vi.clearAllMocks());

  it("renders the roster with secret status and the catalog with agent hints", async () => {
    render(<McpPage />);
    await waitFor(() => expect(screen.getByTestId("server-github")).toBeInTheDocument());
    expect(screen.getByText(/needs GITHUB_PERSONAL_ACCESS_TOKEN/)).toBeInTheDocument();
    expect(screen.getByTestId("catalog-fetch")).toHaveTextContent("read docs");
    expect(screen.getByTestId("catalog-context7")).toHaveTextContent("installed");
  });

  it("adds a parameter-less catalog server in one click", async () => {
    render(<McpPage />);
    await waitFor(() => screen.getByTestId("catalog-fetch"));
    fireEvent.click(screen.getByTestId("catalog-fetch").querySelector("button")!);
    await waitFor(() => expect(api.mcpAdd).toHaveBeenCalledWith(expect.objectContaining({ catalog_id: "fetch" })));
    expect(api.mcpServers).toHaveBeenCalledTimes(2); // reloaded after adding
  });

  it("tests a server and shows the error inline", async () => {
    render(<McpPage />);
    await waitFor(() => screen.getByTestId("server-github"));
    fireEvent.click(screen.getByText("Test"));
    await waitFor(() => expect(screen.getByText(/missing value\(s\)/)).toBeInTheDocument());
  });
});

describe("buildAddBody / missingRequired", () => {
  const entry: any = {
    id: "supabase", params: [{ name: "PROJECT_REF", required: true }],
    env: [{ name: "SUPABASE_ACCESS_TOKEN", required: true, secret: true }, { name: "OPT", required: false }],
  };
  it("flags required params and env until filled", () => {
    expect(missingRequired(entry, { params: {}, env: {} })).toEqual(["PROJECT_REF", "SUPABASE_ACCESS_TOKEN"]);
    expect(missingRequired(entry, { params: { PROJECT_REF: "abc" }, env: { SUPABASE_ACCESS_TOKEN: "t" } })).toEqual([]);
  });
  it("drops blank values and keeps the catalog id", () => {
    const body = buildAddBody(entry, { name: " ", params: { PROJECT_REF: " abc " }, env: { SUPABASE_ACCESS_TOKEN: "t", OPT: "" } });
    expect(body).toEqual({ catalog_id: "supabase", name: undefined, params: { PROJECT_REF: "abc" }, env: { SUPABASE_ACCESS_TOKEN: "t" }, enabled: true });
  });
});
