import { BookmarkPlus, Download, Loader2, Wifi, X } from "lucide-react";
import { cancelPull, startPull, type PullState } from "../../lib/pullManager";
import type { PingResult, ProviderConfig } from "../../lib/api";
import { Card, CardHeader } from "../Card";
import { Field, NumberInput, Select, TextInput } from "../Input";
import { Pill } from "../Pill";
import { PullProgress } from "./PullProgress";
import { PingBanner } from "./PingBanner";
import {
  KIND_DEFAULTS,
  KIND_LABELS,
  needsApiKey,
  needsBaseUrl,
  needsEndpointPath,
  showPullButton,
  type ProviderKind,
} from "./providerKinds";

export function ActiveProviderCard({
  provider,
  setProvider,
  models,
  installedModels,
  ollamaReachable,
  refreshModels,
  activePull,
  pinging,
  pingResult,
  onPing,
  onPingReset,
  onSaveAsProfile,
}: {
  provider: ProviderConfig;
  setProvider: (patch: Partial<ProviderConfig>) => void;
  models: string[];
  installedModels: string[];
  ollamaReachable: boolean;
  refreshModels: () => void;
  activePull: PullState | null;
  pinging: boolean;
  pingResult: PingResult | null;
  onPing: () => void;
  onPingReset: () => void;
  onSaveAsProfile: () => void;
}) {
  const kind = provider.kind as ProviderKind;

  return (
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
            <button className="av-btn-ghost" onClick={onPing} disabled={pinging}>
              {pinging ? <Loader2 className="h-3 w-3 animate-spin" /> : <Wifi className="h-3 w-3" />}
              {pinging ? "Pinging…" : "Ping"}
            </button>
            <button
              className="av-btn-primary"
              onClick={onSaveAsProfile}
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
              setProvider({ kind: k, use_profile: false, ...defaults } as Partial<ProviderConfig>);
              onPingReset();
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
            ) : showPullButton(kind, provider.model, installedModels, ollamaReachable, false) ? (
              <button
                className="av-btn-ghost whitespace-nowrap"
                onClick={() => startPull(provider.model, provider.base_url, refreshModels)}
              >
                <Download className="h-3 w-3" /> Pull
              </button>
            ) : null}
          </div>
          <PullProgress pull={activePull} model={provider.model} />
        </Field>

        {needsApiKey(kind) && (
          <Field label="API Key">
            <TextInput
              type="password"
              placeholder={
                provider.kind === "gemini"
                  ? "GEMINI_API_KEY"
                  : provider.kind === "openai-compat"
                  ? "OPENAI_API_KEY"
                  : provider.kind === "huggingface"
                  ? "HF_TOKEN (hf_…)"
                  : "Bearer token (optional)"
              }
              value={provider.api_key || ""}
              onChange={(e) => setProvider({ api_key: e.target.value, use_profile: false })}
            />
          </Field>
        )}

        {needsBaseUrl(kind) && (
          <Field label="Base URL" className={needsApiKey(kind) ? "" : "col-span-2"}>
            <TextInput
              value={provider.base_url}
              onChange={(e) => setProvider({ base_url: e.target.value, use_profile: false })}
            />
          </Field>
        )}

        {needsEndpointPath(kind) && (
          <>
            <Field label="Endpoint Path">
              <TextInput
                value={provider.endpoint_path || "/v1/chat/completions"}
                placeholder="/v1/chat/completions"
                onChange={(e) => setProvider({ endpoint_path: e.target.value, use_profile: false })}
              />
            </Field>
            <Field label="Auth">
              <label className="flex items-center gap-2 text-xs text-slate-300 mt-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={provider.allow_no_auth ?? false}
                  onChange={(e) => setProvider({ allow_no_auth: e.target.checked, use_profile: false })}
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
            onChange={(e) => setProvider({ temperature: Number(e.target.value), use_profile: false })}
          />
        </Field>
        <Field label="num_predict">
          <NumberInput
            step={64}
            value={provider.num_predict}
            onChange={(e) => setProvider({ num_predict: Number(e.target.value), use_profile: false })}
          />
        </Field>
        {provider.kind === "ollama" && (
          <Field
            label="num_ctx (Ollama)"
            hint="Blank = auto (registry value, capped at 8192). Raise only if you have VRAM headroom."
          >
            <NumberInput
              step={1024}
              min={0}
              placeholder="auto"
              value={provider.num_ctx ?? ""}
              onChange={(e) => {
                const raw = e.target.value;
                const n = raw === "" ? null : Number(raw);
                setProvider({ num_ctx: n !== null && n > 0 ? n : null, use_profile: false });
              }}
            />
          </Field>
        )}
      </div>
      <PingBanner result={pingResult} model={provider.model} />
    </Card>
  );
}
