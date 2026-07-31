import { useEffect, useState } from "react";
import { Cog, Microchip, Plug, Radio, Plus, Search } from "lucide-react";
import { api, type PingResult, type ProfileSummary } from "../lib/api";
import { usePullState } from "../lib/pullManager";
import { useWorkspace } from "../store/workspace";
import { useTrace } from "../store/trace";
import { Card } from "../components/Card";
import { Pill } from "../components/Pill";
import { ActiveProviderCard } from "../components/settings/ActiveProviderCard";
import { ProfilesTable } from "../components/settings/ProfilesTable";
import { TracePanel } from "../components/settings/TracePanel";
import { HardwarePanel } from "../components/settings/HardwarePanel";
import { ProfileEditModal } from "../components/settings/ProfileEditModal";
import { emptyEdit, type ProfileEditState, type ProviderKind } from "../components/settings/providerKinds";
import { cn } from "../lib/utils";

type Category = "provider" | "profiles" | "observability" | "hardware";

const CATEGORIES: { key: Category; label: string; icon: typeof Plug }[] = [
  { key: "provider", label: "Active Provider", icon: Plug },
  { key: "profiles", label: "Saved Profiles", icon: Cog },
  { key: "observability", label: "Observability", icon: Radio },
  { key: "hardware", label: "Hardware", icon: Microchip },
];

export default function SettingsPage() {
  const { provider, setProvider, applyProfile } = useWorkspace();
  const [profiles, setProfiles] = useState<ProfileSummary[]>([]);
  const [edit, setEdit] = useState<ProfileEditState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [category, setCategory] = useState<Category>("provider");
  const [search, setSearch] = useState("");

  // Model lists for the active provider card
  const [models, setModels] = useState<string[]>([]);
  const [installedModels, setInstalledModels] = useState<string[]>([]);
  const [ollamaReachable, setOllamaReachable] = useState(false);

  const [pinging, setPinging] = useState(false);
  const [pingResult, setPingResult] = useState<PingResult | null>(null);
  const activePull = usePullState();

  const traceSettings = useTrace((s) => s.settings);
  const setTraceInStore = useTrace((s) => s.set);
  const refreshTraceInStore = useTrace((s) => s.refresh);
  const [traceBusy, setTraceBusy] = useState(false);
  const [traceError, setTraceError] = useState<string | null>(null);

  const loadProfiles = async () => {
    try {
      setProfiles(await api.listProfiles());
    } catch (e: any) {
      setError(String(e?.message || e));
    }
  };

  useEffect(() => {
    loadProfiles();
  }, []);

  useEffect(() => {
    void refreshTraceInStore();
  }, [refreshTraceInStore]);

  const refreshModels = () => {
    api
      .setupModels(provider.base_url)
      .then((r) => {
        setModels(r.choices);
        setInstalledModels(r.installed || []);
        setOllamaReachable(r.ollama_reachable ?? false);
      })
      .catch(() => {});
  };

  // Refresh model list when active provider changes
  useEffect(() => {
    const kind = provider.kind as ProviderKind;
    if (kind === "ollama") {
      refreshModels();
      return;
    }
    if (kind === "gemini" || kind === "openai-compat" || kind === "custom") {
      const inlineKey = provider.api_key || "";
      const profileName = provider.use_profile && provider.profile_name ? provider.profile_name : null;
      api
        .cloudModels({ kind, base_url: provider.base_url || null, api_key: inlineKey || null, profile_name: profileName })
        .then((r) => setModels(r.choices))
        .catch(() => setModels([]));
      return;
    }
    setModels([]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [provider.kind, provider.base_url, provider.use_profile, provider.profile_name, provider.api_key]);

  const toggleTrace = async (next: boolean) => {
    setTraceBusy(true);
    setTraceError(null);
    try {
      const updated = await api.setTraceSettings(next);
      setTraceInStore(updated);
    } catch (e: any) {
      setTraceError(e?.message || "failed to update trace setting");
    } finally {
      setTraceBusy(false);
    }
  };

  const startNew = () => {
    setEdit({ ...emptyEdit });
  };

  const startEdit = (p: ProfileSummary) => {
    setEdit({
      name: p.name,
      kind: (p.kind as ProviderKind) || "ollama",
      model: p.model,
      base_url: p.base_url,
      api_key: "",
      temperature: p.temperature,
      num_predict: p.num_predict,
      num_ctx: p.num_ctx ?? null,
      endpoint_path: p.endpoint_path || "/v1/chat/completions",
      allow_no_auth: p.allow_no_auth ?? false,
    });
  };

  const saveActiveAsProfile = () => {
    const suggestion = provider.use_profile && provider.profile_name ? `${provider.profile_name}-copy` : "";
    setEdit({
      name: suggestion,
      kind: (provider.kind as ProviderKind) || "ollama",
      model: provider.model,
      base_url: provider.base_url,
      api_key: provider.api_key || "",
      temperature: provider.temperature,
      num_predict: provider.num_predict,
      num_ctx: provider.num_ctx ?? null,
      endpoint_path: provider.endpoint_path || "/v1/chat/completions",
      allow_no_auth: provider.allow_no_auth ?? false,
    });
  };

  const runActivePing = async () => {
    setPinging(true);
    setPingResult(null);
    try {
      const result = await api.pingProvider({
        kind: provider.kind,
        base_url: provider.base_url,
        model: provider.model,
        api_key: provider.api_key || null,
        endpoint_path: provider.endpoint_path || "/v1/chat/completions",
        allow_no_auth: provider.allow_no_auth ?? false,
      });
      setPingResult(result);
    } catch (e: any) {
      setPingResult({ ok: false, error: String(e?.message || e) });
    } finally {
      setPinging(false);
    }
  };

  const remove = async (name: string) => {
    if (!confirm(`Delete profile "${name}"?`)) return;
    try {
      await api.deleteProfile(name);
      await loadProfiles();
    } catch (e: any) {
      setError(String(e?.message || e));
    }
  };

  const activeProfileName = provider.use_profile && provider.profile_name ? provider.profile_name : null;

  const visibleCategories = CATEGORIES.filter((c) =>
    c.label.toLowerCase().includes(search.toLowerCase()),
  );

  return (
    <div className="flex h-full overflow-hidden">
      {/* ── Left: category list ── */}
      <div className="w-60 border-r border-muted bg-header flex flex-col shrink-0">
        <div className="p-4 border-b border-muted">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-bold text-white">Settings</h2>
            <Pill tone="neon">valid</Pill>
          </div>
          <div className="relative">
            <Search className="h-3 w-3 text-slate-600 absolute left-2.5 top-1/2 -translate-y-1/2" />
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search settings…"
              className="av-input pl-7 text-xs"
            />
          </div>
        </div>
        <nav className="p-2 space-y-0.5 overflow-y-auto flex-1">
          {visibleCategories.map(({ key, label, icon: Icon }) => (
            <button
              key={key}
              onClick={() => setCategory(key)}
              className={cn("av-nav-item w-full", category === key && "is-active font-medium")}
            >
              <Icon className={cn("h-3.5 w-3.5 shrink-0", category === key ? "text-emerald-400" : "text-slate-500")} />
              <span className="truncate flex-1 text-left">{label}</span>
            </button>
          ))}
        </nav>
        <div className="p-3 border-t border-muted">
          <button onClick={startNew} className="av-btn-primary w-full justify-center">
            <Plus className="h-3 w-3" /> New profile
          </button>
        </div>
      </div>

      {/* ── Right: selected category ── */}
      <div className={cn("flex-1 overflow-y-auto p-6 space-y-6", category === "hardware" ? "max-w-6xl" : "max-w-4xl")}>
        {category === "provider" && (
          <ActiveProviderCard
            provider={provider}
            setProvider={setProvider}
            models={models}
            installedModels={installedModels}
            ollamaReachable={ollamaReachable}
            refreshModels={refreshModels}
            activePull={activePull}
            pinging={pinging}
            pingResult={pingResult}
            onPing={runActivePing}
            onPingReset={() => setPingResult(null)}
            onSaveAsProfile={saveActiveAsProfile}
          />
        )}

        {category === "profiles" && (
          <ProfilesTable
            profiles={profiles}
            activeProfileName={activeProfileName}
            onUse={applyProfile}
            onEdit={startEdit}
            onDelete={remove}
          />
        )}

        {category === "observability" && (
          <TracePanel
            traceSettings={traceSettings}
            traceBusy={traceBusy}
            traceError={traceError}
            onToggle={toggleTrace}
          />
        )}

        {category === "hardware" && <HardwarePanel />}

        {error && (
          <Card padded className="border-red-500/40">
            <p className="text-xs text-red-300 font-mono">{error}</p>
          </Card>
        )}
      </div>

      {edit && (
        <ProfileEditModal
          initial={edit}
          onClose={() => setEdit(null)}
          onSaved={() => {
            setEdit(null);
            loadProfiles();
          }}
        />
      )}
    </div>
  );
}
