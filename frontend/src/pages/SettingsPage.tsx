import { useEffect, useRef, useState } from "react";
import {
  Check, Download, Loader2, Plus, Save, ShieldCheck, Trash2,
  Wifi, WifiOff, X, BookmarkPlus,
} from "lucide-react";
import { api, type PingResult, type ProfileSummary } from "../lib/api";
import { cancelPull, startPull, usePullState } from "../lib/pullManager";
import { useWorkspace } from "../store/workspace";
import { Card, CardHeader } from "../components/Card";
import { Field, NumberInput, Select, TextInput } from "../components/Input";
import { Pill } from "../components/Pill";

type ProviderKind = "ollama" | "openai-compat" | "gemini" | "custom";

interface EditState {
  name: string;
  kind: ProviderKind;
  model: string;
  base_url: string;
  api_key: string;
  temperature: number;
  num_predict: number;
  endpoint_path: string;
  allow_no_auth: boolean;
}

interface PullState {
  model: string;
  status: string;
  total: number;
  completed: number;
  done: boolean;
  error: string | null;
}

const KIND_DEFAULTS: Record<ProviderKind, Partial<EditState>> = {
  ollama: {
    base_url: "http://localhost:11434",
    model: "qwen2.5-coder:3b",
    api_key: "",
    endpoint_path: "/v1/chat/completions",
    allow_no_auth: false,
  },
  "openai-compat": {
    base_url: "https://api.openai.com",
    model: "gpt-4o-mini",
    endpoint_path: "/v1/chat/completions",
    allow_no_auth: false,
  },
  gemini: {
    base_url: "https://generativelanguage.googleapis.com",
    model: "gemini-2.5-flash",
    endpoint_path: "/v1beta/models",
    allow_no_auth: false,
  },
  custom: {
    base_url: "",
    model: "",
    endpoint_path: "/v1/chat/completions",
    allow_no_auth: false,
  },
};

const emptyEdit: EditState = {
  name: "",
  kind: "ollama",
  model: "qwen2.5-coder:3b",
  base_url: "http://localhost:11434",
  api_key: "",
  temperature: 0.2,
  num_predict: 1024,
  endpoint_path: "/v1/chat/completions",
  allow_no_auth: false,
};

const KIND_LABELS: Record<ProviderKind, string> = {
  ollama: "Ollama (Local)",
  "openai-compat": "OpenAI (Cloud)",
  gemini: "Google Gemini (Cloud)",
  custom: "Custom Server (OpenAI-Compatible)",
};

export default function SettingsPage() {
  const { provider, setProvider, applyProfile } = useWorkspace();
  const [profiles, setProfiles] = useState<ProfileSummary[]>([]);
  const [edit, setEdit] = useState<EditState | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Model lists: choices = installed + full catalog
  const [models, setModels] = useState<string[]>([]);
  const [installedModels, setInstalledModels] = useState<string[]>([]);
  const [ollamaReachable, setOllamaReachable] = useState(false);
  const [editModels, setEditModels] = useState<string[]>([]);
  const [editInstalledModels, setEditInstalledModels] = useState<string[]>([]);
  const [editOllamaReachable, setEditOllamaReachable] = useState(false);

  // Separate ping state for active provider vs modal form
  const [activePingResult, setActivePingResult] = useState<PingResult | null>(null);
  const [activePinging, setActivePinging] = useState(false);
  const [editPingResult, setEditPingResult] = useState<PingResult | null>(null);
  const [editPinging, setEditPinging] = useState(false);

  // Active-provider pull: module-level singleton (survives tab switches)
  const activePull = usePullState();
  // Edit-modal pull: local state is fine — modal is always visible while pulling
  const [editPull, setEditPull] = useState<PullState | null>(null);
  const editPullRef = useRef<{ abort: () => void } | null>(null);

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

  // Refresh model list when active provider changes
  useEffect(() => {
    const kind = provider.kind as ProviderKind;
    if (kind === "ollama") {
      (async () => {
        try {
          const r = await api.setupModels(provider.base_url);
          setModels(r.choices);
          setInstalledModels(r.installed || []);
          setOllamaReachable(r.ollama_reachable ?? false);
        } catch {
          setModels([]);
          setInstalledModels([]);
          setOllamaReachable(false);
        }
      })();
      return;
    }
    if (kind === "gemini" || kind === "openai-compat" || kind === "custom") {
      const inlineKey = (provider as any).api_key || "";
      const profileName =
        provider.use_profile && provider.profile_name ? provider.profile_name : null;
      (async () => {
        try {
          const r = await api.cloudModels({
            kind,
            base_url: provider.base_url || null,
            api_key: inlineKey || null,
            profile_name: profileName,
          });
          setModels(r.choices);
        } catch {
          setModels([]);
        }
      })();
      return;
    }
    setModels([]);
  }, [
    provider.kind,
    provider.base_url,
    provider.use_profile,
    provider.profile_name,
    (provider as any).api_key,
  ]);

  // Refresh model list for edit form
  useEffect(() => {
    if (!edit) {
      setEditModels([]);
      setEditInstalledModels([]);
      return;
    }
    const kind = edit.kind;
    if (kind === "ollama") {
      (async () => {
        try {
          const r = await api.setupModels(edit.base_url);
          setEditModels(r.choices);
          setEditInstalledModels(r.installed || []);
          setEditOllamaReachable(r.ollama_reachable ?? false);
        } catch {
          setEditModels([]);
          setEditInstalledModels([]);
          setEditOllamaReachable(false);
        }
      })();
      return;
    }
    if (kind === "gemini" || kind === "openai-compat" || kind === "custom") {
      (async () => {
        try {
          const r = await api.cloudModels({
            kind,
            base_url: edit.base_url || null,
            api_key: edit.api_key || null,
            profile_name: edit.name || null,
          });
          setEditModels(r.choices);
        } catch {
          setEditModels([]);
        }
      })();
      return;
    }
    setEditModels([]);
  }, [edit?.kind, edit?.base_url, edit?.api_key, edit?.name]);

  const startNew = () => {
    setEdit({ ...emptyEdit });
    setEditPingResult(null);
    setEditPull(null);
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
      endpoint_path: p.endpoint_path || "/v1/chat/completions",
      allow_no_auth: p.allow_no_auth ?? false,
    });
    setEditPingResult(null);
    setEditPull(null);
  };

  const saveActiveAsProfile = () => {
    const suggestion =
      provider.use_profile && provider.profile_name
        ? `${provider.profile_name}-copy`
        : "";
    setEdit({
      name: suggestion,
      kind: (provider.kind as ProviderKind) || "ollama",
      model: provider.model,
      base_url: provider.base_url,
      api_key: (provider as any).api_key || "",
      temperature: provider.temperature,
      num_predict: provider.num_predict,
      endpoint_path: provider.endpoint_path || "/v1/chat/completions",
      allow_no_auth: provider.allow_no_auth ?? false,
    });
    setEditPingResult(null);
    setEditPull(null);
  };

  const handleKindChange = (kind: ProviderKind) => {
    const defaults = KIND_DEFAULTS[kind] || {};
    setEdit((prev) => (prev ? { ...prev, kind, ...defaults } : null));
    setEditPingResult(null);
    setEditPull(null);
  };

  const closeModal = () => {
    setEdit(null);
    setEditPingResult(null);
    editPullRef.current?.abort();
    editPullRef.current = null;
    setEditPull(null);
  };

  // Generic ping helper
  const runPing = async (
    src: EditState | typeof provider,
    setPinging: (v: boolean) => void,
    setPingResult: (r: PingResult | null) => void,
  ) => {
    setPinging(true);
    setPingResult(null);
    try {
      const result = await api.pingProvider({
        kind: (src as any).kind,
        base_url: (src as any).base_url,
        model: (src as any).model,
        api_key: (src as any).api_key || null,
        endpoint_path: (src as any).endpoint_path || "/v1/chat/completions",
        allow_no_auth: (src as any).allow_no_auth ?? false,
      });
      setPingResult(result);
    } catch (e: any) {
      setPingResult({ ok: false, error: String(e?.message || e) });
    } finally {
      setPinging(false);
    }
  };

  // Pull a model via Ollama — for the edit modal only (active-provider uses pullManager)
  const startEditPull = (model: string, baseUrl: string, afterDone?: () => void) => {
    editPullRef.current?.abort();
    setEditPull({ model, status: "Connecting…", total: 0, completed: 0, done: false, error: null });
    const conn = api.ollamaPull(
      model,
      baseUrl,
      (data) => {
        setEditPull((prev) =>
          prev
            ? {
                ...prev,
                status: data.status || prev.status,
                total: data.total ?? prev.total,
                completed: data.completed ?? prev.completed,
                done: data.status === "success",
              }
            : null,
        );
      },
      () => {
        setEditPull((prev) => (prev ? { ...prev, done: true, status: "Download complete" } : null));
        afterDone?.();
      },
      (err) => {
        setEditPull((prev) =>
          prev ? { ...prev, error: String(err), done: true } : null,
        );
      },
    );
    editPullRef.current = conn;
  };

  const save = async () => {
    if (!edit) return;
    if (!edit.name.trim()) {
      setError("Profile name is required.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await api.upsertProfile(edit.name, {
        name: edit.name,
        kind: edit.kind,
        model: edit.model,
        base_url: edit.base_url,
        api_key: edit.api_key || null,
        temperature: edit.temperature,
        num_predict: edit.num_predict,
        endpoint_path: edit.endpoint_path || "/v1/chat/completions",
        allow_no_auth: edit.allow_no_auth,
      });
      closeModal();
      await loadProfiles();
    } catch (e: any) {
      setError(String(e?.message || e));
    } finally {
      setBusy(false);
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

  const needsApiKey = (kind: ProviderKind) =>
    kind === "openai-compat" || kind === "gemini" || kind === "custom";
  const needsEndpointPath = (kind: ProviderKind) => kind === "custom";
  const needsBaseUrl = (kind: ProviderKind) => kind !== "gemini";

  // Show the pull button whenever Ollama is reachable and the model isn't installed yet.
  const showPullButton = (
    kind: ProviderKind,
    model: string,
    installed: string[],
    reachable: boolean,
    pulling: boolean,
  ) =>
    kind === "ollama" &&
    !!model &&
    reachable &&
    !installed.includes(model) &&
    !pulling;

  return (
    <div className="p-6 space-y-6 max-w-5xl overflow-y-auto h-full">
      <CardHeader
        title="Provider Settings & Profiles"
        description="Configure model profiles. Choose from Ollama (local), OpenAI, Google Gemini, or a custom OpenAI-compatible server."
        right={
          <button onClick={startNew} className="av-btn-primary">
            <Plus className="h-3 w-3" /> New profile
          </button>
        }
      />

      {/* ── Active provider (inline) ── */}
      <Card padded>
        <CardHeader
          eyebrow="Active provider"
          title={
            provider.use_profile && provider.profile_name
              ? `Using profile: ${provider.profile_name}`
              : "Inline configuration"
          }
          right={
            <div className="flex items-center gap-2">
              <button
                className="av-btn-ghost"
                onClick={() => runPing(provider, setActivePinging, setActivePingResult)}
                disabled={activePinging}
              >
                {activePinging ? <Loader2 className="h-3 w-3 animate-spin" /> : <Wifi className="h-3 w-3" />}
                {activePinging ? "Pinging…" : "Ping"}
              </button>
              <button
                className="av-btn-primary"
                onClick={saveActiveAsProfile}
                title="Save the current active configuration as a profile"
              >
                <BookmarkPlus className="h-3 w-3" /> Save as profile
              </button>
              <Pill tone={provider.use_profile ? "neon" : "slate"}>
                {provider.use_profile ? "PROFILE" : "INLINE"}
              </Pill>
            </div>
          }
        />
        <div className="grid grid-cols-2 gap-3">
          <Field label="Provider Type" className="col-span-2">
            <Select
              value={provider.kind}
              onChange={(e) => {
                const k = (e.target as any).value as ProviderKind;
                const defaults = KIND_DEFAULTS[k] || {};
                setProvider({ kind: k, use_profile: false, ...defaults });
                setActivePingResult(null);
                cancelPull();
              }}
            >
              {(Object.keys(KIND_LABELS) as ProviderKind[]).map((k) => (
                <option key={k} value={k}>
                  {KIND_LABELS[k]}
                </option>
              ))}
            </Select>
          </Field>

          <Field label="Model">
            <div className="flex gap-2">
              <TextInput
                value={provider.model}
                onChange={(e) =>
                  setProvider({ model: e.target.value, use_profile: false })
                }
                className="flex-1"
              />
              {models.length > 0 && (
                <Select
                  value=""
                  onChange={(e) => {
                    const v = (e.target as any).value;
                    if (v) setProvider({ model: v, use_profile: false });
                  }}
                  className="w-40"
                >
                  <option value="">presets…</option>
                  {models.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </Select>
              )}
              {activePull && activePull.model === provider.model && !activePull.done && !activePull.error ? (
                <button className="av-btn-ghost whitespace-nowrap" onClick={cancelPull}>
                  <X className="h-3 w-3" /> Cancel
                </button>
              ) : showPullButton(
                provider.kind as ProviderKind,
                provider.model,
                installedModels,
                ollamaReachable,
                false,
              ) ? (
                <button
                  className="av-btn-ghost whitespace-nowrap"
                  onClick={() =>
                    startPull(provider.model, provider.base_url, () => {
                      api.setupModels(provider.base_url).then((r) => {
                        setModels(r.choices);
                        setInstalledModels(r.installed || []);
                        setOllamaReachable(r.ollama_reachable ?? false);
                      }).catch(() => {});
                    })
                  }
                >
                  <Download className="h-3 w-3" /> Pull
                </button>
              ) : null}
            </div>
            <PullProgress pull={activePull} model={provider.model} />
          </Field>

          {needsApiKey(provider.kind as ProviderKind) && (
            <Field label="API Key">
              <TextInput
                type="password"
                placeholder={
                  provider.kind === "gemini"
                    ? "GEMINI_API_KEY"
                    : provider.kind === "openai-compat"
                    ? "OPENAI_API_KEY"
                    : "Bearer token (optional)"
                }
                value={(provider as any).api_key || ""}
                onChange={(e) =>
                  setProvider({ api_key: e.target.value, use_profile: false } as any)
                }
              />
            </Field>
          )}

          {needsBaseUrl(provider.kind as ProviderKind) && (
            <Field
              label="Base URL"
              className={needsApiKey(provider.kind as ProviderKind) ? "" : "col-span-2"}
            >
              <TextInput
                value={provider.base_url}
                onChange={(e) =>
                  setProvider({ base_url: e.target.value, use_profile: false })
                }
              />
            </Field>
          )}

          {needsEndpointPath(provider.kind as ProviderKind) && (
            <>
              <Field label="Endpoint Path">
                <TextInput
                  value={(provider as any).endpoint_path || "/v1/chat/completions"}
                  placeholder="/v1/chat/completions"
                  onChange={(e) =>
                    setProvider({ endpoint_path: e.target.value, use_profile: false } as any)
                  }
                />
              </Field>
              <Field label="Auth">
                <label className="flex items-center gap-2 text-xs text-slate-300 mt-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={(provider as any).allow_no_auth ?? false}
                    onChange={(e) =>
                      setProvider({ allow_no_auth: e.target.checked, use_profile: false } as any)
                    }
                    className="rounded"
                  />
                  Skip auth (private subnet)
                </label>
              </Field>
            </>
          )}

          <Field label="Temperature">
            <NumberInput
              step={0.05}
              value={provider.temperature}
              onChange={(e) =>
                setProvider({ temperature: Number(e.target.value), use_profile: false })
              }
            />
          </Field>
          <Field label="num_predict">
            <NumberInput
              step={64}
              value={provider.num_predict}
              onChange={(e) =>
                setProvider({ num_predict: Number(e.target.value), use_profile: false })
              }
            />
          </Field>
        </div>
        <PingBanner result={activePingResult} model={provider.model} />
      </Card>

      {/* ── Saved profiles ── */}
      <Card padded>
        <CardHeader
          eyebrow="Saved profiles"
          title="Profiles"
          description="Click a profile row to expand details, or use the action buttons to switch, edit, or delete."
        />
        {profiles.length === 0 ? (
          <p className="text-xs text-slate-500 text-center py-4">
            No saved profiles yet — click <strong>New profile</strong> above to save your current setup.
          </p>
        ) : (
          <div className="mt-3 rounded-lg border border-white/5 overflow-hidden">
            <table className="w-full text-xs font-mono">
              <thead className="bg-slate-950 text-slate-500 uppercase border-b border-white/5">
                <tr>
                  <th className="px-3 py-2 text-left text-[10px] tracking-wider">Name</th>
                  <th className="px-3 py-2 text-left text-[10px] tracking-wider">Provider</th>
                  <th className="px-3 py-2 text-left text-[10px] tracking-wider">Model</th>
                  <th className="px-3 py-2 text-left text-[10px] tracking-wider">Temp</th>
                  <th className="px-3 py-2 text-left text-[10px] tracking-wider">Key</th>
                  <th className="px-3 py-2 text-left text-[10px] tracking-wider">Status</th>
                  <th className="px-3 py-2 text-right text-[10px] tracking-wider">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {profiles.map((p) => {
                  const active = provider.use_profile && provider.profile_name === p.name;
                  return (
                    <tr
                      key={p.name}
                      className={`transition-colors ${active ? "bg-emerald-500/5" : "hover:bg-slate-950/60"}`}
                    >
                      <td className="px-3 py-2.5 font-semibold text-white">{p.name}</td>
                      <td className="px-3 py-2.5 text-slate-400">
                        {KIND_LABELS[p.kind as ProviderKind]?.split(" ")[0] || p.kind}
                      </td>
                      <td className="px-3 py-2.5 text-slate-300 max-w-[160px] truncate" title={p.model}>
                        {p.model}
                      </td>
                      <td className="px-3 py-2.5 text-slate-400">{p.temperature.toFixed(2)}</td>
                      <td className="px-3 py-2.5">
                        {p.has_api_key ? (
                          <span className="flex items-center gap-1 text-emerald-400">
                            <ShieldCheck className="h-3 w-3" /> stored
                          </span>
                        ) : (
                          <span className="text-slate-600">—</span>
                        )}
                      </td>
                      <td className="px-3 py-2.5">
                        <Pill tone={active ? "neon" : "slate"} className="text-[9px]">
                          {active ? "ACTIVE" : "STANDBY"}
                        </Pill>
                      </td>
                      <td className="px-3 py-2.5 text-right">
                        <div className="flex items-center justify-end gap-1.5">
                          <button
                            className="av-btn-primary py-1 px-2 text-[10px]"
                            onClick={() => applyProfile(p)}
                          >
                            <Check className="h-3 w-3" /> Use
                          </button>
                          <button
                            className="av-btn-ghost py-1 px-2 text-[10px]"
                            onClick={() => startEdit(p)}
                          >
                            Edit
                          </button>
                          <button
                            className="av-btn py-1 px-2 text-[10px] bg-red-500/10 text-red-300 border border-red-500/30 hover:bg-red-500/20"
                            onClick={() => remove(p.name)}
                          >
                            <Trash2 className="h-3 w-3" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {error && (
        <Card padded className="border-red-500/40">
          <p className="text-xs text-red-300 font-mono">{error}</p>
        </Card>
      )}

      {/* ── Profile edit / new — modal overlay ── */}
      {edit && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
          onMouseDown={(e) => {
            if (e.target === e.currentTarget) closeModal();
          }}
        >
          <div className="bg-[#0d1117] border border-white/10 rounded-xl w-full max-w-2xl mx-4 max-h-[90vh] overflow-y-auto shadow-2xl shadow-black/60">
            {/* Modal header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-white/5">
              <div>
                <p className="text-[10px] text-slate-500 uppercase tracking-widest font-mono mb-0.5">
                  {edit.name ? "Edit profile" : "New profile"}
                </p>
                <h2 className="text-sm font-semibold text-white font-mono">
                  {edit.name || "Untitled profile"}
                </h2>
              </div>
              <div className="flex items-center gap-2">
                <button
                  className="av-btn-ghost"
                  onClick={() => runPing(edit, setEditPinging, setEditPingResult)}
                  disabled={editPinging}
                >
                  {editPinging ? <Loader2 className="h-3 w-3 animate-spin" /> : <Wifi className="h-3 w-3" />}
                  {editPinging ? "Pinging…" : "Ping"}
                </button>
                <button className="av-btn-ghost" onClick={closeModal}>
                  <X className="h-3 w-3" /> Cancel
                </button>
                <button className="av-btn-primary" onClick={save} disabled={busy}>
                  <Save className="h-3 w-3" /> {busy ? "Saving…" : "Save"}
                </button>
              </div>
            </div>

            {/* Modal body */}
            <div className="p-6">
              <div className="grid grid-cols-2 gap-3">
                <Field label="Name">
                  <TextInput
                    value={edit.name}
                    onChange={(e) => setEdit({ ...edit, name: e.target.value })}
                  />
                </Field>

                <Field label="Provider Type">
                  <Select
                    value={edit.kind}
                    onChange={(e) =>
                      handleKindChange((e.target as any).value as ProviderKind)
                    }
                  >
                    {(Object.keys(KIND_LABELS) as ProviderKind[]).map((k) => (
                      <option key={k} value={k}>
                        {KIND_LABELS[k]}
                      </option>
                    ))}
                  </Select>
                </Field>

                <Field label="Model" className="col-span-2">
                  <div className="flex gap-2">
                    <TextInput
                      value={edit.model}
                      placeholder={
                        edit.kind === "gemini"
                          ? "gemini-2.5-flash"
                          : edit.kind === "openai-compat"
                          ? "gpt-4o-mini"
                          : "qwen2.5-coder:3b"
                      }
                      onChange={(e) => setEdit({ ...edit, model: e.target.value })}
                      className="flex-1"
                    />
                    {editModels.length > 0 && (
                      <Select
                        value=""
                        onChange={(e) => {
                          const v = (e.target as any).value;
                          if (v) setEdit({ ...edit, model: v });
                        }}
                        className="w-40"
                      >
                        <option value="">presets…</option>
                        {editModels.map((m) => (
                          <option key={m} value={m}>
                            {m}
                          </option>
                        ))}
                      </Select>
                    )}
                    {showPullButton(
                      edit.kind,
                      edit.model,
                      editInstalledModels,
                      editOllamaReachable,
                      !!(editPull && !editPull.done && !editPull.error),
                    ) && (
                      <button
                        className="av-btn-ghost whitespace-nowrap"
                        onClick={() =>
                          startEditPull(edit.model, edit.base_url, () => {
                            api.setupModels(edit.base_url).then((r) => {
                              setEditModels(r.choices);
                              setEditInstalledModels(r.installed || []);
                              setEditOllamaReachable(r.ollama_reachable ?? false);
                            }).catch(() => {});
                          })
                        }
                      >
                        <Download className="h-3 w-3" /> Pull
                      </button>
                    )}
                  </div>
                  <PullProgress pull={editPull} model={edit.model} />
                </Field>

                {needsBaseUrl(edit.kind) && (
                  <Field label="Base URL" className="col-span-2">
                    <TextInput
                      value={edit.base_url}
                      placeholder={KIND_DEFAULTS[edit.kind]?.base_url || ""}
                      onChange={(e) => setEdit({ ...edit, base_url: e.target.value })}
                    />
                  </Field>
                )}

                {needsApiKey(edit.kind) && (
                  <Field
                    label={
                      edit.kind === "gemini"
                        ? "Gemini API Key"
                        : edit.kind === "openai-compat"
                        ? "OpenAI API Key"
                        : "Bearer Token (optional)"
                    }
                    className="col-span-2"
                  >
                    <TextInput
                      type="password"
                      value={edit.api_key}
                      placeholder={
                        edit.allow_no_auth ? "skip — private subnet" : "sk-…"
                      }
                      onChange={(e) => setEdit({ ...edit, api_key: e.target.value })}
                    />
                  </Field>
                )}

                {needsEndpointPath(edit.kind) && (
                  <>
                    <Field label="Endpoint Path">
                      <TextInput
                        value={edit.endpoint_path}
                        placeholder="/v1/chat/completions"
                        onChange={(e) =>
                          setEdit({ ...edit, endpoint_path: e.target.value })
                        }
                      />
                    </Field>
                    <Field label="Auth">
                      <label className="flex items-center gap-2 text-xs text-slate-300 mt-2 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={edit.allow_no_auth}
                          onChange={(e) =>
                            setEdit({ ...edit, allow_no_auth: e.target.checked })
                          }
                          className="rounded"
                        />
                        Skip auth (private subnet / no-key server)
                      </label>
                    </Field>
                  </>
                )}

                <Field label="Temperature">
                  <NumberInput
                    step={0.05}
                    value={edit.temperature}
                    onChange={(e) =>
                      setEdit({ ...edit, temperature: Number(e.target.value) })
                    }
                  />
                </Field>
                <Field label="num_predict">
                  <NumberInput
                    step={64}
                    value={edit.num_predict}
                    onChange={(e) =>
                      setEdit({ ...edit, num_predict: Number(e.target.value) })
                    }
                  />
                </Field>
              </div>

              <PingBanner result={editPingResult} model={edit.model} className="mt-4" />
              {error && (
                <p className="mt-4 text-xs text-red-300 font-mono border border-red-500/30 bg-red-500/5 rounded px-3 py-2">
                  {error}
                </p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Sub-components ───────────────────────────────────────────────────────────

function PullProgress({ pull, model }: { pull: PullState | null; model: string }) {
  if (!pull || pull.model !== model) return null;
  const pct =
    pull.total > 0 ? Math.min(100, Math.round((pull.completed / pull.total) * 100)) : null;

  return (
    <div className="mt-1.5 space-y-1">
      <div className="h-1 bg-slate-800 rounded-full overflow-hidden">
        {pull.error ? (
          <div className="h-full w-full bg-red-500/60" />
        ) : pull.done ? (
          <div className="h-full w-full bg-emerald-500" />
        ) : pct !== null ? (
          <div
            className="h-full bg-emerald-500 transition-all duration-300"
            style={{ width: `${pct}%` }}
          />
        ) : (
          <div className="h-full w-1/3 bg-emerald-500/70 animate-pulse" />
        )}
      </div>
      <p className={`text-[10px] font-mono ${pull.error ? "text-red-400" : "text-slate-400"}`}>
        {pull.error
          ? pull.error.slice(0, 80)
          : pull.done
          ? "Download complete"
          : pct !== null
          ? `${pull.status} — ${pct}%`
          : pull.status}
      </p>
    </div>
  );
}

function PingBanner({
  result,
  model,
  className,
}: {
  result: PingResult | null;
  model: string;
  className?: string;
}) {
  if (!result) return null;

  if (result.ok) {
    return (
      <div
        className={`mt-3 flex items-start gap-3 rounded-lg border border-emerald-500/20 bg-emerald-500/5 px-4 py-3 text-xs font-mono ${className ?? ""}`}
      >
        <Wifi className="h-4 w-4 text-emerald-400 shrink-0 mt-0.5" />
        <div>
          <p className="text-emerald-400 font-semibold">Ping successful</p>
          <p className="text-slate-400 mt-0.5">
            Connected to <span className="text-white">{model}</span> in{" "}
            <span className="text-white">{result.latency_ms?.toFixed(0)}ms</span>. Provider is reachable and responding.
          </p>
        </div>
      </div>
    );
  }

  const errMsg = result.error || "Unknown error";
  const isNotInstalled =
    errMsg.toLowerCase().includes("not found") ||
    errMsg.toLowerCase().includes("model") ||
    errMsg.toLowerCase().includes("pull");

  return (
    <div
      className={`mt-3 flex items-start gap-3 rounded-lg border border-red-500/20 bg-red-500/5 px-4 py-3 text-xs font-mono ${className ?? ""}`}
    >
      <WifiOff className="h-4 w-4 text-red-400 shrink-0 mt-0.5" />
      <div>
        <p className="text-red-400 font-semibold">
          {isNotInstalled ? "Model not available" : "Ping failed"}
        </p>
        <p className="text-slate-400 mt-0.5 break-all">{errMsg}</p>
        {isNotInstalled && (
          <p className="text-slate-500 mt-1">
            Use the <strong className="text-slate-300">Pull</strong> button to download this model first.
          </p>
        )}
      </div>
    </div>
  );
}

function KvRead({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <label className="av-label">{label}</label>
      <div
        className="rounded-md border border-muted bg-slate-950/60 px-2.5 py-1.5
                   text-[11px] font-mono text-slate-300 select-text truncate"
        title={value}
      >
        {value || <span className="text-slate-600">—</span>}
      </div>
    </div>
  );
}
