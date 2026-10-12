import { useCallback, useEffect, useMemo, useState } from "react";
import {
  BookOpen,
  Check,
  ExternalLink,
  KeyRound,
  Plus,
  RefreshCw,
  Server,
  ShieldAlert,
  Trash2,
  Wrench,
} from "lucide-react";
import {
  api,
  type McpAddBody,
  type McpCatalogEntry,
  type McpCatalogResponse,
  type McpServer,
  type McpServersResponse,
  type McpTestResult,
} from "../lib/api";
import { Card, CardHeader } from "../components/Card";
import { Pill, type PillTone } from "../components/Pill";
import { Field, TextInput, Toggle } from "../components/Input";
import { EmptyState } from "../components/EmptyState";
import { TabBar } from "../components/TabBar";
import { Button } from "../components/Button";
import { cn } from "../lib/utils";

// MCP (Model Context Protocol) tool servers. Two halves:
//   * the ROSTER (~/.cgx/mcp.json): servers the agent can reach, with their
//     secret status, a Test button that lists the server's tools, secrets
//     management (values go to the keyring -- never the JSON), toggle, remove;
//   * the CATALOG: well-known developer servers (official reference servers,
//     GitHub, Playwright, Context7, databases, Stripe, Sentry ...) added in
//     one click after the user supplies any parameters/secrets.
// The agent is told which servers exist and WHEN to use each (agent_hint) in
// its prompts, so a local model actually reaches for them.

const STATUS_TONE: Record<string, PillTone> = {
  official: "neon",
  vendor: "purple",
  community: "slate",
  archived: "amber",
};
const RISK_TONE: Record<string, PillTone> = { read: "slate", write: "amber", execute: "red" };

export function buildAddBody(
  entry: McpCatalogEntry,
  values: { name?: string; params: Record<string, string>; env: Record<string, string> },
): McpAddBody {
  const env: Record<string, string> = {};
  for (const [k, v] of Object.entries(values.env)) if (v.trim()) env[k] = v;
  const params: Record<string, string> = {};
  for (const [k, v] of Object.entries(values.params)) if (v.trim()) params[k] = v.trim();
  return {
    catalog_id: entry.id,
    name: values.name?.trim() || undefined,
    params,
    env,
    enabled: true,
  };
}

export function missingRequired(entry: McpCatalogEntry, values: { params: Record<string, string>; env: Record<string, string> }) {
  const missing: string[] = [];
  for (const p of entry.params) if (p.required && !(values.params[p.name] || "").trim()) missing.push(p.name);
  for (const e of entry.env) if (e.required && !(values.env[e.name] || "").trim()) missing.push(e.name);
  return missing;
}

function CatalogCard({
  entry,
  onAdded,
}: {
  entry: McpCatalogEntry;
  onAdded: () => Promise<void>;
}) {
  const needsInput = entry.params.length > 0 || entry.env.length > 0;
  const [open, setOpen] = useState(false);
  const [name, setName] = useState(entry.id);
  const [params, setParams] = useState<Record<string, string>>({});
  const [env, setEnv] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const missing = missingRequired(entry, { params, env });

  const submit = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.mcpAdd(buildAddBody(entry, { name, params, env }));
      setOpen(false);
      await onAdded();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card className={cn("flex flex-col gap-3", entry.installed && "is-active")} data-testid={`catalog-${entry.id}`}>
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-sm font-semibold text-white truncate">{entry.name}</p>
          <p className="text-[11px] text-slate-500 truncate">{entry.vendor}</p>
        </div>
        <div className="flex items-center gap-1 shrink-0">
          <Pill tone={STATUS_TONE[entry.status] ?? "slate"}>{entry.status}</Pill>
          <Pill tone={RISK_TONE[entry.risk] ?? "slate"}>{entry.risk}</Pill>
        </div>
      </div>
      <p className="text-xs text-slate-300 leading-relaxed">{entry.description}</p>
      <p className="text-[11px] text-slate-500 leading-relaxed">
        <span className="text-emerald-400/80 font-mono uppercase tracking-wider text-[10px] mr-1">agent uses it to</span>
        {entry.agent_hint}
      </p>
      <div className="flex items-center gap-2 text-[11px] font-mono text-slate-500">
        <span className="truncate">
          {entry.transport === "http" ? entry.url : `${entry.command} ${(entry.args ?? []).join(" ")}`}
        </span>
      </div>
      <div className="mt-auto flex items-center justify-between gap-2 pt-1">
        <a
          className="inline-flex items-center gap-1 text-[11px] text-slate-400 hover:text-white"
          href={entry.docs_url}
          target="_blank"
          rel="noreferrer"
        >
          <BookOpen className="h-3 w-3" /> docs <ExternalLink className="h-3 w-3" />
        </a>
        {entry.installed ? (
          <Pill tone="neon"><Check className="h-3 w-3" /> installed</Pill>
        ) : (
          <Button
            variant="primary"
            className="text-[11px]"
            icon={<Plus className="h-3.5 w-3.5" />}
            onClick={() => (needsInput ? setOpen((v) => !v) : void submit())}
            disabled={busy}
          >
            {busy ? "Adding…" : needsInput ? "Add…" : "Add"}
          </Button>
        )}
      </div>
      {open && (
        <div className="rounded-lg border border-white/10 bg-slate-950/60 p-3 space-y-3">
          <Field label="Roster name" hint="How the agent refers to this server.">
            <TextInput value={name} onChange={(e) => setName(e.target.value)} />
          </Field>
          {entry.params.map((p) => (
            <Field key={p.name} label={`${p.name}${p.required ? " *" : ""}`} hint={p.description}>
              <TextInput
                placeholder={p.placeholder}
                value={params[p.name] ?? ""}
                onChange={(e) => setParams({ ...params, [p.name]: e.target.value })}
              />
            </Field>
          ))}
          {entry.env.map((v) => (
            <Field
              key={v.name}
              label={`${v.name}${v.required ? " *" : ""}`}
              hint={`${v.description}${v.secret ? " — stored in your OS keyring, never in mcp.json." : ""}`}
            >
              <TextInput
                type={v.secret ? "password" : "text"}
                autoComplete="off"
                value={env[v.name] ?? ""}
                onChange={(e) => setEnv({ ...env, [v.name]: e.target.value })}
              />
            </Field>
          ))}
          {error && <p className="text-xs text-red-400 font-mono">{error}</p>}
          <div className="flex items-center justify-end gap-2">
            <Button onClick={() => setOpen(false)}>Cancel</Button>
            <Button variant="primary" disabled={busy || missing.length > 0} onClick={() => void submit()}>
              {busy ? "Adding…" : missing.length ? `Fill ${missing.join(", ")}` : "Add server"}
            </Button>
          </div>
        </div>
      )}
    </Card>
  );
}

function ServerRow({
  server,
  onChanged,
}: {
  server: McpServer;
  onChanged: () => Promise<void>;
}) {
  const [busy, setBusy] = useState<string | null>(null);
  const [test, setTest] = useState<McpTestResult | null>(null);
  const [secretsOpen, setSecretsOpen] = useState(false);
  const [secrets, setSecrets] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);

  const run = async (label: string, fn: () => Promise<void>) => {
    setBusy(label);
    setError(null);
    try {
      await fn();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const endpoint = server.url || `${server.command ?? ""} ${(server.args ?? []).join(" ")}`.trim();
  const catalogEnv = server.catalog?.env ?? [];
  const vars = server.env_vars.length ? server.env_vars : catalogEnv.map((e) => e.name);

  return (
    <div className="rounded-xl border border-white/5 bg-slate-950/60 px-4 py-3 space-y-2" data-testid={`server-${server.name}`}>
      <div className="flex items-center gap-3">
        <div className="flex-1 min-w-0">
          <p className="text-sm text-slate-100 font-medium truncate flex items-center gap-2">
            {server.name}
            <span className="text-[10px] uppercase tracking-wider text-slate-500">{server.transport}</span>
            {server.catalog && <Pill tone={STATUS_TONE[server.catalog.status] ?? "slate"}>{server.catalog.status}</Pill>}
            {server.ready ? (
              <Pill tone="neon">ready</Pill>
            ) : (
              <Pill tone="amber"><KeyRound className="h-3 w-3" /> needs {server.missing_env.join(", ")}</Pill>
            )}
          </p>
          <p className="text-[11px] font-mono text-slate-500 truncate">{endpoint || "(no endpoint)"}</p>
          {(server.agent_hint || server.description) && (
            <p className="text-[11px] text-slate-400 mt-0.5">{server.agent_hint || server.description}</p>
          )}
        </div>
        <Toggle
          checked={server.enabled}
          label={busy === "toggle" ? "…" : server.enabled ? "enabled" : "disabled"}
          onChange={(v) => void run("toggle", async () => { await api.mcpToggle({ name: server.name, enabled: v }); await onChanged(); })}
        />
        <Button
          icon={<Wrench className="h-3.5 w-3.5" />}
          disabled={busy !== null}
          onClick={() => void run("test", async () => setTest(await api.mcpTest(server.name)))}
        >
          {busy === "test" ? "Testing…" : "Test"}
        </Button>
        {vars.length > 0 && (
          <Button icon={<KeyRound className="h-3.5 w-3.5" />} onClick={() => setSecretsOpen((v) => !v)}>
            Secrets
          </Button>
        )}
        <Button
          variant="danger"
          icon={<Trash2 className="h-3.5 w-3.5" />}
          disabled={busy !== null}
          onClick={() => {
            if (window.confirm(`Remove MCP server "${server.name}" and its stored secrets?`)) {
              void run("remove", async () => { await api.mcpRemove(server.name); await onChanged(); });
            }
          }}
        >
          Remove
        </Button>
      </div>
      {secretsOpen && (
        <div className="rounded-lg border border-white/10 bg-slate-950/80 p-3 grid gap-3 md:grid-cols-2">
          {vars.map((v) => {
            const meta = catalogEnv.find((e) => e.name === v);
            const missing = server.missing_env.includes(v);
            return (
              <Field key={v} label={v} hint={meta?.description ?? (missing ? "No value set." : "Value is set (hidden).")}>
                <TextInput
                  type="password"
                  autoComplete="off"
                  placeholder={missing ? "enter value" : "•••••••• (leave blank to keep)"}
                  value={secrets[v] ?? ""}
                  onChange={(e) => setSecrets({ ...secrets, [v]: e.target.value })}
                />
              </Field>
            );
          })}
          <div className="md:col-span-2 flex justify-end gap-2">
            <Button onClick={() => setSecretsOpen(false)}>Close</Button>
            <Button
              variant="primary"
              disabled={busy !== null || !Object.values(secrets).some((s) => s.trim())}
              onClick={() => void run("secrets", async () => {
                const env: Record<string, string> = {};
                for (const [k, val] of Object.entries(secrets)) if (val.trim()) env[k] = val;
                await api.mcpSetEnv(server.name, env);
                setSecrets({});
                setSecretsOpen(false);
                await onChanged();
              })}
            >
              Save secrets
            </Button>
          </div>
        </div>
      )}
      {test && (
        <div className={cn("rounded-lg border p-3 text-xs", test.ok ? "border-emerald-500/20 bg-emerald-500/5" : "border-amber-500/20 bg-amber-500/5")}>
          {test.ok ? (
            <>
              <p className="text-emerald-300 font-medium mb-1">Connected — {test.tools.length} tool{test.tools.length === 1 ? "" : "s"}</p>
              <ul className="grid gap-1 md:grid-cols-2">
                {test.tools.map((t) => (
                  <li key={t.name} className="text-slate-300">
                    <span className="font-mono text-emerald-200">{t.name}</span>
                    {t.description && <span className="text-slate-500"> — {t.description}</span>}
                  </li>
                ))}
              </ul>
            </>
          ) : (
            <p className="text-amber-300 flex items-start gap-2"><ShieldAlert className="h-4 w-4 shrink-0" /> {test.error}</p>
          )}
        </div>
      )}
      {error && <p className="text-xs text-red-400 font-mono">{error}</p>}
    </div>
  );
}

function CustomServerForm({ onAdded }: { onAdded: () => Promise<void> }) {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [transport, setTransport] = useState<"stdio" | "http">("stdio");
  const [command, setCommand] = useState("");
  const [url, setUrl] = useState("");
  const [tokenEnv, setTokenEnv] = useState("");
  const [hint, setHint] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  if (!open) {
    return (
      <Button icon={<Plus className="h-3.5 w-3.5" />} onClick={() => setOpen(true)}>
        Add custom server
      </Button>
    );
  }
  const submit = async () => {
    setBusy(true);
    setError(null);
    try {
      const parts = command.trim().split(/\s+/).filter(Boolean);
      await api.mcpAdd({
        name: name.trim(),
        transport,
        command: transport === "stdio" ? parts[0] : undefined,
        args: transport === "stdio" ? parts.slice(1) : [],
        url: transport === "http" ? url.trim() : undefined,
        auth: transport === "http" && tokenEnv.trim() ? { type: "bearer", token_env: tokenEnv.trim() } : {},
        agent_hint: hint.trim(),
        enabled: true,
      });
      setOpen(false);
      setName(""); setCommand(""); setUrl(""); setTokenEnv(""); setHint("");
      await onAdded();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="rounded-xl border border-white/10 bg-slate-950/60 p-4 grid gap-3 md:grid-cols-2">
      <Field label="Name"><TextInput value={name} onChange={(e) => setName(e.target.value)} placeholder="my-docs" /></Field>
      <Field label="Transport">
        <TabBar tabs={[{ key: "stdio", label: "stdio" }, { key: "http", label: "http" }]} active={transport} onChange={setTransport} />
      </Field>
      {transport === "stdio" ? (
        <Field label="Command" hint="Executable followed by its arguments." className="md:col-span-2">
          <TextInput value={command} onChange={(e) => setCommand(e.target.value)} placeholder="npx -y some-mcp-server --flag" />
        </Field>
      ) : (
        <>
          <Field label="URL"><TextInput value={url} onChange={(e) => setUrl(e.target.value)} placeholder="https://host/mcp" /></Field>
          <Field label="Bearer token env var" hint="Set its value with Secrets after adding.">
            <TextInput value={tokenEnv} onChange={(e) => setTokenEnv(e.target.value)} placeholder="MY_TOKEN" />
          </Field>
        </>
      )}
      <Field label="Agent hint" hint="One line telling the agent when to use it." className="md:col-span-2">
        <TextInput value={hint} onChange={(e) => setHint(e.target.value)} placeholder="look up our internal API docs" />
      </Field>
      {error && <p className="md:col-span-2 text-xs text-red-400 font-mono">{error}</p>}
      <div className="md:col-span-2 flex justify-end gap-2">
        <Button onClick={() => setOpen(false)}>Cancel</Button>
        <Button variant="primary" disabled={busy || !name.trim() || (transport === "stdio" ? !command.trim() : !url.trim())} onClick={() => void submit()}>
          {busy ? "Adding…" : "Add server"}
        </Button>
      </div>
    </div>
  );
}

export default function McpPage() {
  const [data, setData] = useState<McpServersResponse | null>(null);
  const [catalog, setCatalog] = useState<McpCatalogResponse | null>(null);
  const [category, setCategory] = useState<string>("all");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [servers, cat] = await Promise.all([api.mcpServers(), api.mcpCatalog()]);
      setData(servers);
      setCatalog(cat);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const servers = data?.servers ?? [];
  const tabs = useMemo(() => {
    const cats = catalog?.categories ?? {};
    return [{ key: "all", label: "All" }, ...Object.entries(cats).map(([k, v]) => ({ key: k, label: v }))];
  }, [catalog]);
  const entries = (catalog?.servers ?? []).filter((e) => category === "all" || e.category === category);

  return (
    <div className="p-6 space-y-6 overflow-y-auto h-full max-w-6xl">
      <CardHeader
        eyebrow="Control"
        title="MCP tool servers"
        description="External Model Context Protocol servers the agent can call: docs, browsers, databases, source control and more. The agent is told which servers exist and when to use each."
        right={
          <button className="av-btn-ghost" onClick={() => void load()}>
            <RefreshCw className={cn("h-3.5 w-3.5", loading && "animate-spin")} /> Refresh
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
          <span className="ml-auto font-mono text-[11px] text-slate-500">{data.config_path}</span>
        )}
      </div>

      {error && <p className="text-xs text-red-400 font-mono">{error}</p>}

      <Card padded>
        <CardHeader
          eyebrow="Roster"
          title="Your servers"
          description="Enabled servers are advertised to the Tech Lead and Developer. Secrets live in your OS keyring, never in mcp.json."
          right={<CustomServerForm onAdded={load} />}
        />
        <div className="mt-3 space-y-2">
          {servers.length === 0 && !loading && (
            <EmptyState
              icon={<Server className="h-5 w-5" />}
              title="No MCP servers yet"
              description="Add one from the catalog below, or add a custom server."
            />
          )}
          {servers.map((s) => (
            <ServerRow key={s.name} server={s} onChanged={load} />
          ))}
        </div>
      </Card>

      <Card padded>
        <CardHeader
          eyebrow="Catalog"
          title="Well-known developer servers"
          description="Curated, real packages: official reference servers, vendor-maintained and popular community servers. Add in one click; supply parameters or tokens when asked."
        />
        <div className="mt-2 mb-4">
          <TabBar tabs={tabs} active={category} onChange={setCategory} />
        </div>
        {entries.length === 0 ? (
          <EmptyState title="No catalog entries" description="The catalog could not be loaded." />
        ) : (
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {entries.map((e) => (
              <CatalogCard key={e.id} entry={e} onAdded={load} />
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}
