import { useEffect, useState } from "react";
import {
  Check, Plus, Save, ShieldCheck, Trash2, Wifi, WifiOff, X, BookmarkPlus,
} from "lucide-react";
import { api, type PingResult, type ProfileSummary } from "../lib/api";
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
  const [models, setModels] = useState<string[]>([]);
  const [editModels, setEditModels] = useState<string[]>([]);
  const [pingResult, setPingResult] = useState<PingResult | null>(null);
  const [pinging, setPinging] = useState(false);

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
    const kind = provider.kind as ProviderKind;
    if (kind === "ollama") {
      (async () => {
        try {
          const r = await api.setupModels(provider.base_url);
          setModels(r.choices);
        } catch {
          setModels([]);
        }
      })();
      return;
    }
    if (kind === "gemini" || kind === "openai-compat" || kind === "custom") {
      // Use the active profile's stored key when no inline key was typed so the
      // dropdown can be populated without re-entering credentials.
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

  // Same idea as the active-provider effect, but driven by the editor draft so
  // the profile creation form also shows real current models for cloud kinds.
  useEffect(() => {
    if (!edit) {
      setEditModels([]);
      return;
    }
    const kind = edit.kind;
    if (kind === "ollama") {
      (async () => {
        try {
          const r = await api.setupModels(edit.base_url);
          setEditModels(r.choices);
        } catch {
          setEditModels([]);
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

  const startNew = () => setEdit({ ...emptyEdit });
  const startEdit = (p: ProfileSummary) =>
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

  // Snapshot the current Active Provider into a new-profile draft so the
  // user can persist the inline config without re-typing it.
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
    setPingResult(null);
  };

  const handleKindChange = (kind: ProviderKind) => {
    const defaults = KIND_DEFAULTS[kind] || {};
    setEdit((prev) =>
      prev ? { ...prev, kind, ...defaults } : null
    );
    setPingResult(null);
  };

  const ping = async (src: EditState | typeof provider) => {
    setPinging(true);
    setPingResult(null);
    try {
      const result = await api.pingProvider({
        kind: (src as any).kind,
        base_url: (src as any).base_url,
        model: (src as any).model,
        api_key: (src as any).api_key || (src as any).api_key || null,
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
      setEdit(null);
      setPingResult(null);
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

  return (
    <div className="p-6 space-y-6 max-w-5xl overflow-y-auto h-full">
      <CardHeader
        title="⚙️ Provider Settings & Profiles"
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
              {pingResult && (
                <span
                  className={`text-[10px] font-mono flex items-center gap-1 ${pingResult.ok ? "text-emerald-400" : "text-red-400"}`}
                >
                  {pingResult.ok ? (
                    <Wifi className="h-3 w-3" />
                  ) : (
                    <WifiOff className="h-3 w-3" />
                  )}
                  {pingResult.ok
                    ? `${pingResult.latency_ms?.toFixed(0)}ms`
                    : pingResult.error?.slice(0, 60)}
                </span>
              )}
              <button
                className="av-btn-ghost"
                onClick={() => ping(provider)}
                disabled={pinging}
              >
                <Wifi className="h-3 w-3" />
                {pinging ? "Pinging…" : "Ping"}
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
          {/* Provider type selector */}
          <Field label="Provider Type" className="col-span-2">
            <Select
              value={provider.kind}
              onChange={(e) => {
                const k = (e.target as any).value as ProviderKind;
                const defaults = KIND_DEFAULTS[k] || {};
                setProvider({ kind: k, use_profile: false, ...defaults });
                setPingResult(null);
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
            </div>
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
      </Card>

      {/* ── Saved profiles ── */}
      <div className="grid grid-cols-2 gap-4">
        {profiles.map((p) => {
          const active =
            provider.use_profile && provider.profile_name === p.name;
          return (
            <Card key={p.name} padded active={active}>
              <div className="flex justify-between items-center mb-3">
                <h3 className="text-xs font-semibold text-emerald-400 font-mono">
                  Profile: {p.name}
                </h3>
                <Pill tone={active ? "neon" : "slate"}>
                  {active ? "ACTIVE" : "STANDBY"}
                </Pill>
              </div>
              <div className="space-y-3 text-xs">
                <KvRead label="Provider" value={KIND_LABELS[p.kind as ProviderKind] || p.kind} />
                {p.kind !== "gemini" && (
                  <KvRead label="Base Endpoint URL" value={p.base_url} />
                )}
                {p.kind === "custom" && (
                  <KvRead label="Endpoint Path" value={p.endpoint_path || "/v1/chat/completions"} />
                )}
                <div className="grid grid-cols-2 gap-2">
                  <KvRead label="Model" value={p.model} />
                  <KvRead label="Temperature" value={p.temperature.toFixed(2)} />
                  <KvRead label="num_predict" value={String(p.num_predict)} />
                  {p.kind === "custom" && (
                    <KvRead
                      label="Auth bypass"
                      value={p.allow_no_auth ? "yes (no-auth)" : "no"}
                    />
                  )}
                </div>
                <p className="text-[10px] text-slate-500 font-mono flex items-center gap-1">
                  <ShieldCheck className="h-3 w-3" />
                  API key: {p.has_api_key ? "stored (keyring)" : "—"}
                </p>
                <p className="text-[10px] text-slate-500 font-mono">
                  Read-only snapshot — click <strong>Edit</strong> to change values, then <strong>Save</strong>.
                </p>
              </div>
              <div className="flex gap-2 mt-4">
                <button
                  className="av-btn-primary flex-1"
                  onClick={() => applyProfile(p)}
                >
                  <Check className="h-3 w-3" /> Use
                </button>
                <button className="av-btn-ghost" onClick={() => startEdit(p)}>
                  Edit
                </button>
                <button
                  className="av-btn bg-red-500/10 text-red-300 border border-red-500/30 hover:bg-red-500/20"
                  onClick={() => remove(p.name)}
                >
                  <Trash2 className="h-3 w-3" />
                </button>
              </div>
            </Card>
          );
        })}
        {profiles.length === 0 && (
          <Card padded className="col-span-2 text-center text-xs text-slate-400">
            No saved profiles yet. Click <strong>New profile</strong> to save your
            current setup for reuse.
          </Card>
        )}
      </div>

      {/* ── Edit / New profile form ── */}
      {edit && (
        <Card padded active>
          <CardHeader
            eyebrow={edit.name ? "Edit profile" : "New profile"}
            title={edit.name || "Untitled profile"}
            right={
              <>
                {/* Ping button */}
                <div className="flex items-center gap-2">
                  {pingResult && (
                    <span
                      className={`text-[10px] font-mono flex items-center gap-1 ${pingResult.ok ? "text-emerald-400" : "text-red-400"}`}
                    >
                      {pingResult.ok ? (
                        <Wifi className="h-3 w-3" />
                      ) : (
                        <WifiOff className="h-3 w-3" />
                      )}
                      {pingResult.ok
                        ? `OK · ${pingResult.latency_ms?.toFixed(0)}ms`
                        : pingResult.error?.slice(0, 80)}
                    </span>
                  )}
                  <button
                    className="av-btn-ghost"
                    onClick={() => ping(edit)}
                    disabled={pinging}
                  >
                    <Wifi className="h-3 w-3" />
                    {pinging ? "Pinging…" : "Ping"}
                  </button>
                </div>
                <button className="av-btn-ghost" onClick={() => { setEdit(null); setPingResult(null); }}>
                  <X className="h-3 w-3" /> Cancel
                </button>
                <button className="av-btn-primary" onClick={save} disabled={busy}>
                  <Save className="h-3 w-3" /> {busy ? "Saving…" : "Save"}
                </button>
              </>
            }
          />
          <div className="grid grid-cols-2 gap-3">
            <Field label="Name">
              <TextInput
                value={edit.name}
                onChange={(e) => setEdit({ ...edit, name: e.target.value })}
              />
            </Field>

            {/* Provider type selector */}
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
              </div>
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
        </Card>
      )}

      {error && (
        <Card padded className="border-red-500/40">
          <p className="text-xs text-red-300 font-mono">{error}</p>
        </Card>
      )}
    </div>
  );
}

function KvRead({ label, value }: { label: string; value: string }) {
  // Rendered as a static styled value (NOT an <input>) so users do not
  // mistake a saved-profile snapshot for an editable field.
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
