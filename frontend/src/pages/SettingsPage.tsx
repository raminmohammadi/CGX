import { useEffect, useState } from "react";
import { Check, Plus, Save, ShieldCheck, Trash2, X } from "lucide-react";
import { api, type ProfileSummary } from "../lib/api";
import { useWorkspace } from "../store/workspace";
import { Card, CardHeader } from "../components/Card";
import { Field, NumberInput, Select, TextInput } from "../components/Input";
import { Pill } from "../components/Pill";

interface EditState {
  name: string;
  kind: "ollama" | "openai-compat";
  model: string;
  base_url: string;
  api_key: string;
  temperature: number;
  num_predict: number;
}

const emptyEdit: EditState = {
  name: "",
  kind: "ollama",
  model: "qwen2.5-coder:3b",
  base_url: "http://localhost:11434",
  api_key: "",
  temperature: 0.2,
  num_predict: 1024,
};

export default function SettingsPage() {
  const { provider, setProvider, applyProfile } = useWorkspace();
  const [profiles, setProfiles] = useState<ProfileSummary[]>([]);
  const [edit, setEdit] = useState<EditState | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [models, setModels] = useState<string[]>([]);

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
    (async () => {
      try {
        const r = await api.setupModels(provider.base_url);
        setModels(r.choices);
      } catch {
        setModels([]);
      }
    })();
  }, [provider.base_url]);

  const startNew = () => setEdit({ ...emptyEdit });
  const startEdit = (p: ProfileSummary) =>
    setEdit({
      name: p.name,
      kind: (p.kind as EditState["kind"]) || "ollama",
      model: p.model,
      base_url: p.base_url,
      api_key: "",
      temperature: p.temperature,
      num_predict: p.num_predict,
    });

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
      });
      setEdit(null);
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

  return (
    <div className="p-6 space-y-6 max-w-5xl overflow-y-auto h-full">
      <CardHeader
        title="⚙️ Provider Settings & Profiles"
        description="Configure model profiles, token budgets, and client-side bucket parameters."
        right={
          <button onClick={startNew} className="av-btn-primary">
            <Plus className="h-3 w-3" /> New profile
          </button>
        }
      />

      <Card padded>
        <CardHeader
          eyebrow="Active provider"
          title={provider.use_profile && provider.profile_name ? `Using profile: ${provider.profile_name}` : "Inline configuration"}
          right={
            <Pill tone={provider.use_profile ? "neon" : "slate"}>
              {provider.use_profile ? "PROFILE" : "INLINE"}
            </Pill>
          }
        />
        <div className="grid grid-cols-2 gap-3">
          <Field label="Kind">
            <Select
              value={provider.kind}
              onChange={(e) =>
                setProvider({ kind: (e.target as any).value, use_profile: false })
              }
            >
              <option value="ollama">ollama</option>
              <option value="openai-compat">openai-compat</option>
            </Select>
          </Field>
          <Field label="Model">
            <div className="flex gap-2">
              <TextInput
                value={provider.model}
                onChange={(e) => setProvider({ model: e.target.value, use_profile: false })}
                className="flex-1"
              />
              {models.length > 0 && (
                <Select
                  value=""
                  onChange={(e) => {
                    const v = (e.target as any).value;
                    if (v) setProvider({ model: v, use_profile: false });
                  }}
                  className="w-32"
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
          <Field label="Base URL" className="col-span-2">
            <TextInput
              value={provider.base_url}
              onChange={(e) =>
                setProvider({ base_url: e.target.value, use_profile: false })
              }
            />
          </Field>
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

      <div className="grid grid-cols-2 gap-4">
        {profiles.map((p) => {
          const active = provider.use_profile && provider.profile_name === p.name;
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
                <KvRead label="Base Endpoint URL" value={p.base_url} />
                <div className="grid grid-cols-2 gap-2">
                  <KvRead label="Kind" value={p.kind} />
                  <KvRead label="Model" value={p.model} />
                  <KvRead label="Temperature" value={p.temperature.toFixed(2)} />
                  <KvRead label="num_predict" value={String(p.num_predict)} />
                </div>
                <p className="text-[10px] text-slate-500 font-mono flex items-center gap-1">
                  <ShieldCheck className="h-3 w-3" />
                  API key: {p.has_api_key ? "stored (keyring)" : "—"}
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

      {edit && (
        <Card padded active>
          <CardHeader
            eyebrow={edit.name ? "Edit profile" : "New profile"}
            title={edit.name || "Untitled profile"}
            right={
              <>
                <button className="av-btn-ghost" onClick={() => setEdit(null)}>
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
            <Field label="Kind">
              <Select
                value={edit.kind}
                onChange={(e) => setEdit({ ...edit, kind: (e.target as any).value })}
              >
                <option value="ollama">ollama</option>
                <option value="openai-compat">openai-compat</option>
              </Select>
            </Field>
            <Field label="Model" className="col-span-2">
              <TextInput
                value={edit.model}
                onChange={(e) => setEdit({ ...edit, model: e.target.value })}
              />
            </Field>
            <Field label="Base URL" className="col-span-2">
              <TextInput
                value={edit.base_url}
                onChange={(e) => setEdit({ ...edit, base_url: e.target.value })}
              />
            </Field>
            <Field label="API key (optional)">
              <TextInput
                type="password"
                value={edit.api_key}
                onChange={(e) => setEdit({ ...edit, api_key: e.target.value })}
              />
            </Field>
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
  return (
    <div>
      <label className="av-label">{label}</label>
      <input
        type="text"
        value={value}
        className="av-input text-slate-300"
        readOnly
      />
    </div>
  );
}
