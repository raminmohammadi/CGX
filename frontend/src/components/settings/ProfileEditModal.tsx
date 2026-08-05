import { useEffect, useRef, useState } from "react";
import { Download, Loader2, Save, Wifi, X } from "lucide-react";
import { api, type PingResult } from "../../lib/api";
import { Field, NumberInput, Select, TextInput } from "../Input";
import { PullProgress, type PullProgressState } from "./PullProgress";
import { PingBanner } from "./PingBanner";
import {
  KIND_DEFAULTS,
  KIND_LABELS,
  needsApiKey,
  needsBaseUrl,
  needsEndpointPath,
  showPullButton,
  type ProfileEditState,
  type ProviderKind,
} from "./providerKinds";

export function ProfileEditModal({
  initial,
  onClose,
  onSaved,
}: {
  initial: ProfileEditState;
  onClose: () => void;
  onSaved: () => void;
}) {
  const [edit, setEdit] = useState<ProfileEditState>(initial);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [editModels, setEditModels] = useState<string[]>([]);
  const [editInstalledModels, setEditInstalledModels] = useState<string[]>([]);
  const [editOllamaReachable, setEditOllamaReachable] = useState(false);

  const [pinging, setPinging] = useState(false);
  const [pingResult, setPingResult] = useState<PingResult | null>(null);

  const [pull, setPull] = useState<PullProgressState | null>(null);
  const pullRef = useRef<{ abort: () => void } | null>(null);

  useEffect(() => {
    return () => {
      pullRef.current?.abort();
    };
  }, []);

  // Refresh the model list whenever the edit form's kind/base_url/key/name changes.
  useEffect(() => {
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
    if (kind === "gemini" || kind === "openai-compat" || kind === "custom" || kind === "huggingface") {
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [edit.kind, edit.base_url, edit.api_key, edit.name]);

  const handleKindChange = (kind: ProviderKind) => {
    const defaults = KIND_DEFAULTS[kind] || {};
    setEdit((prev) => ({ ...prev, kind, ...defaults }) as ProfileEditState);
    setPingResult(null);
    pullRef.current?.abort();
    setPull(null);
  };

  const runPing = async () => {
    setPinging(true);
    setPingResult(null);
    try {
      const result = await api.pingProvider({
        kind: edit.kind,
        base_url: edit.base_url,
        model: edit.model,
        api_key: edit.api_key || null,
        endpoint_path: edit.endpoint_path || "/v1/chat/completions",
        allow_no_auth: edit.allow_no_auth ?? false,
      });
      setPingResult(result);
    } catch (e: any) {
      setPingResult({ ok: false, error: String(e?.message || e) });
    } finally {
      setPinging(false);
    }
  };

  const startEditPull = () => {
    pullRef.current?.abort();
    setPull({ model: edit.model, status: "Connecting…", total: 0, completed: 0, done: false, error: null });
    const conn = api.ollamaPull(
      edit.model,
      edit.base_url,
      (data) => {
        const errMsg =
          data.status === "error" || data.error
            ? String(data.error || data.status || "pull failed")
            : null;
        setPull((prev) =>
          prev
            ? {
                ...prev,
                status: data.status || prev.status,
                total: data.total ?? prev.total,
                completed: data.completed ?? prev.completed,
                done: data.status === "success" || errMsg != null,
                error: errMsg ?? prev.error,
              }
            : null,
        );
      },
      () => {
        setPull((prev) => {
          if (!prev) return null;
          if (prev.error) return { ...prev, done: true };
          if (prev.done) return { ...prev, status: "Download complete" };
          return { ...prev, done: true, error: "Pull ended without success; see Ollama logs." };
        });
        api
          .setupModels(edit.base_url)
          .then((r) => {
            setEditModels(r.choices);
            setEditInstalledModels(r.installed || []);
            setEditOllamaReachable(r.ollama_reachable ?? false);
          })
          .catch(() => {});
      },
      (err) => {
        setPull((prev) => (prev ? { ...prev, error: String(err), done: true } : null));
      },
    );
    pullRef.current = conn;
  };

  const save = async () => {
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
        num_ctx: edit.num_ctx,
        endpoint_path: edit.endpoint_path || "/v1/chat/completions",
        allow_no_auth: edit.allow_no_auth,
      });
      onSaved();
    } catch (e: any) {
      setError(String(e?.message || e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="bg-[#0d1117] border border-white/10 rounded-xl w-full max-w-2xl mx-4 max-h-[90vh] overflow-y-auto shadow-2xl shadow-black/60">
        <div className="flex items-center justify-between px-6 py-4 border-b border-white/5">
          <div>
            <p className="text-[10px] text-slate-500 uppercase tracking-widest font-mono mb-0.5">
              {edit.name ? "Edit profile" : "New profile"}
            </p>
            <h2 className="text-sm font-semibold text-white font-mono">{edit.name || "Untitled profile"}</h2>
          </div>
          <div className="flex items-center gap-2">
            <button className="av-btn-ghost" onClick={runPing} disabled={pinging}>
              {pinging ? <Loader2 className="h-3 w-3 animate-spin" /> : <Wifi className="h-3 w-3" />}
              {pinging ? "Pinging…" : "Ping"}
            </button>
            <button className="av-btn-ghost" onClick={onClose}>
              <X className="h-3 w-3" /> Cancel
            </button>
            <button className="av-btn-primary" onClick={save} disabled={busy}>
              <Save className="h-3 w-3" /> {busy ? "Saving…" : "Save"}
            </button>
          </div>
        </div>

        <div className="p-6">
          <div className="grid grid-cols-2 gap-3">
            <Field label="Name">
              <TextInput value={edit.name} onChange={(e) => setEdit({ ...edit, name: e.target.value })} />
            </Field>

            <Field label="Provider Type">
              <Select value={edit.kind} onChange={(e) => handleKindChange((e.target as any).value as ProviderKind)}>
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
                      : edit.kind === "huggingface"
                      ? "Qwen/Qwen2.5-Coder-32B-Instruct"
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
                {showPullButton(edit.kind, edit.model, editInstalledModels, editOllamaReachable, !!(pull && !pull.done && !pull.error)) && (
                  <button className="av-btn-ghost whitespace-nowrap" onClick={startEditPull}>
                    <Download className="h-3 w-3" /> Pull
                  </button>
                )}
              </div>
              <PullProgress pull={pull} model={edit.model} />
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
                    : edit.kind === "huggingface"
                    ? "Hugging Face Token"
                    : "Bearer Token (optional)"
                }
                className="col-span-2"
              >
                <TextInput
                  type="password"
                  value={edit.api_key}
                  placeholder={
                    edit.allow_no_auth
                      ? "skip -- private subnet"
                      : edit.kind === "huggingface"
                      ? "hf_…"
                      : "sk-…"
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
                    onChange={(e) => setEdit({ ...edit, endpoint_path: e.target.value })}
                  />
                </Field>
                <Field label="Auth">
                  <label className="flex items-center gap-2 text-xs text-slate-300 mt-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={edit.allow_no_auth}
                      onChange={(e) => setEdit({ ...edit, allow_no_auth: e.target.checked })}
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
                onChange={(e) => setEdit({ ...edit, temperature: Number(e.target.value) })}
              />
            </Field>
            <Field label="num_predict">
              <NumberInput
                step={64}
                value={edit.num_predict}
                onChange={(e) => setEdit({ ...edit, num_predict: Number(e.target.value) })}
              />
            </Field>
            {edit.kind === "ollama" && (
              <Field label="num_ctx (Ollama)" hint="Blank = auto (capped at 8192). Raise only with VRAM headroom.">
                <NumberInput
                  step={1024}
                  min={0}
                  placeholder="auto"
                  value={edit.num_ctx ?? ""}
                  onChange={(e) => {
                    const raw = e.target.value;
                    const n = raw === "" ? null : Number(raw);
                    setEdit({ ...edit, num_ctx: n !== null && n > 0 ? n : null });
                  }}
                />
              </Field>
            )}
          </div>

          <PingBanner result={pingResult} model={edit.model} className="mt-4" />
          {error && (
            <p className="mt-4 text-xs text-red-300 font-mono border border-red-500/30 bg-red-500/5 rounded px-3 py-2">{error}</p>
          )}
        </div>
      </div>
    </div>
  );
}
