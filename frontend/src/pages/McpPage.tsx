import { useCallback, useEffect, useState } from "react";
import { RefreshCw, Server } from "lucide-react";
import { api, type McpServersResponse } from "../lib/api";
import { Card, CardHeader } from "../components/Card";
import { Pill } from "../components/Pill";
import { Toggle } from "../components/Input";
import { EmptyState } from "../components/EmptyState";

// MCP (Model Context Protocol) tool-server management. Servers are declared in
// a local ~/.cgx/mcp.json roster; this page lists them, shows whether the
// optional SDK is installed, and lets the operator enable/disable a server
// (persisted back to the JSON file). Adding a new server is still a config
// edit -- this surface intentionally does not create servers.
export default function McpPage() {
  const [data, setData] = useState<McpServersResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      setData(await api.mcpServers());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const toggle = async (name: string, enabled: boolean) => {
    setBusy(name);
    try {
      await api.mcpToggle({ name, enabled });
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const servers = data?.servers ?? [];

  return (
    <div className="p-6 space-y-6 overflow-y-auto h-full max-w-6xl">
      <CardHeader
        eyebrow="Control"
        title="MCP tool servers"
        description="External Model Context Protocol tool servers the swarm agent can call."
        right={
          <button className="av-btn-ghost" onClick={() => void load()}>
            <RefreshCw className="h-3.5 w-3.5" /> Refresh
          </button>
        }
      />

      <div className="flex items-center gap-3 text-xs text-slate-400">
        <Pill tone={data?.sdk_installed ? "neon" : "amber"}>
          {data?.sdk_installed ? "SDK installed" : "SDK not installed"}
        </Pill>
        {!data?.sdk_installed && (
          <span>Install with <code className="text-slate-300">pip install "cgx[mcp]"</code></span>
        )}
        {data?.config_path && (
          <span className="ml-auto font-mono text-[11px] text-slate-500">
            {data.config_path}
          </span>
        )}
      </div>

      {error && (
        <p className="text-xs text-red-400 font-mono">{error}</p>
      )}

      <Card padded>
        <CardHeader
          eyebrow="Roster"
          title="Configured servers"
          description="Toggle a server to advertise or hide it from the agent."
        />
        <div className="mt-3 space-y-2">
          {servers.length === 0 && !loading && (
            <EmptyState
              icon={<Server className="h-5 w-5" />}
              title="No MCP servers configured"
              description="Add servers to ~/.cgx/mcp.json, then refresh."
            />
          )}
          {servers.map((s) => (
            <div
              key={s.name}
              className="flex items-center gap-3 rounded border border-white/5 bg-slate-950/60 px-3 py-2"
            >
              <div className="flex-1 min-w-0">
                <p className="text-sm text-slate-100 font-medium truncate">
                  {s.name}
                  <span className="ml-2 text-[10px] uppercase tracking-wider text-slate-500">
                    {s.transport}
                  </span>
                </p>
                <p className="text-[11px] font-mono text-slate-500 truncate">
                  {s.url || s.command || "(no endpoint)"}
                </p>
              </div>
              <Toggle
                checked={s.enabled}
                label={busy === s.name ? "…" : s.enabled ? "enabled" : "disabled"}
                onChange={(v) => void toggle(s.name, v)}
              />
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
