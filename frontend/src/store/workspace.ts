import { create } from "zustand";
import { persist } from "zustand/middleware";
import { api, type IndexLocation, type ProfileSummary, type ProviderConfig } from "../lib/api";

// Workspace store: per-browser persistence of the provider config, the
// active index location, and the selected session. The page components
// read/write these to keep Setup → Ask → Plan → Agent consistent.

export interface WorkspaceState {
  provider: ProviderConfig;
  index: IndexLocation;
  projectRoot: string;
  selectedSessionId: string | null;
  // One-shot guard so ``bootstrapProvider`` adopts a saved profile only on the
  // first run in a browser and never clobbers a deliberate later choice.
  providerInitialized: boolean;
  setProvider: (patch: Partial<ProviderConfig>) => void;
  setIndex: (patch: Partial<IndexLocation>) => void;
  setProjectRoot: (root: string) => void;
  setSelectedSession: (id: string | null) => void;
  applyProfile: (profile: { name: string; kind: string; model: string;
    base_url: string; temperature: number; num_predict: number;
    num_ctx?: number | null;
    endpoint_path?: string; allow_no_auth?: boolean }) => void;
}

const defaultProvider: ProviderConfig = {
  use_profile: false,
  profile_name: null,
  kind: "ollama",
  model: "qwen2.5-coder:3b",
  base_url: "http://localhost:11434",
  api_key: null,
  temperature: 0.2,
  num_predict: 1024,
  num_ctx: null,
  endpoint_path: "/v1/chat/completions",
  allow_no_auth: false,
  think: false,
  multi_agent_debate: false,
};

const defaultIndex: IndexLocation = {
  index_dir: "/tmp/cgx_index/indices",
  records: "/tmp/cgx_index/records.jsonl",
  embed_model: "jinaai/jina-embeddings-v2-base-code",
};

export const useWorkspace = create<WorkspaceState>()(
  persist(
    (set) => ({
      provider: defaultProvider,
      index: defaultIndex,
      projectRoot: "",
      selectedSessionId: null,
      providerInitialized: false,
      setProvider: (patch) =>
        set((s) => ({ provider: { ...s.provider, ...patch } })),
      setIndex: (patch) => set((s) => ({ index: { ...s.index, ...patch } })),
      setProjectRoot: (root) => set({ projectRoot: root }),
      setSelectedSession: (id) => set({ selectedSessionId: id }),
      applyProfile: (p) =>
        set((s) => ({
          provider: {
            ...s.provider,
            use_profile: true,
            profile_name: p.name,
            kind: p.kind as ProviderConfig["kind"],
            model: p.model,
            base_url: p.base_url,
            temperature: p.temperature,
            num_predict: p.num_predict,
            num_ctx: p.num_ctx ?? null,
            endpoint_path: p.endpoint_path ?? "/v1/chat/completions",
            allow_no_auth: p.allow_no_auth ?? false,
          },
        })),
    }),
    {
      name: "cgx-workspace",
      // Never persist the raw API key to localStorage: it's plaintext,
      // browser-extension-readable storage. Everything else about the
      // provider config is safe to keep across reloads; `api_key` is
      // re-entered or re-applied from a saved profile (server-side
      // keyring/secrets store) each session.
      partialize: (s) => ({
        ...s,
        provider: { ...s.provider, api_key: null },
      }),
    },
  ),
);

// Adopt a saved profile as the active provider on first run.
//
// The default provider is a placeholder (``qwen2.5-coder:3b``, ``use_profile:
// false``) that a user may never have installed. Without this, someone whose
// only saved profile is, say, a 14B GGUF would open the app pointed at a model
// that isn't on disk -- the "it's saved but won't connect" trap -- because
// ``applyProfile`` only ever ran on a manual click in Settings. This runs once
// per browser (guarded by ``providerInitialized``) so it fixes the cold start
// without ever overriding a deliberate later choice.
export async function bootstrapProvider(): Promise<void> {
  if (useWorkspace.getState().providerInitialized) return;

  let profiles: ProfileSummary[];
  try {
    profiles = await api.listProfiles();
  } catch {
    // Backend not up yet (the status poller is still retrying). Leave the flag
    // unset so a later shell mount / reload can try again.
    return;
  }

  // Mark handled regardless of the adoption outcome below: we only get one
  // shot at "first run", and re-running on every mount would clobber a user
  // who later picks an inline config on purpose.
  useWorkspace.setState({ providerInitialized: true });
  if (profiles.length === 0) return;

  const active = useWorkspace.getState().provider;
  const apply = useWorkspace.getState().applyProfile;

  // Respect (and refresh) an explicit, still-existing profile selection.
  if (active.use_profile && active.profile_name) {
    const current = profiles.find((p) => p.name === active.profile_name);
    if (current) {
      apply(current);
      return;
    }
  }

  // Otherwise adopt the best available profile: prefer an Ollama profile whose
  // model is actually installed locally so we land on something that connects
  // immediately; fall back to the first Ollama profile, then any profile.
  const ollama = profiles.filter((p) => p.kind === "ollama");
  for (const p of ollama) {
    try {
      const r = await api.setupModels(p.base_url);
      const has = (r.installed || []).some(
        (m) => m.toLowerCase() === p.model.toLowerCase());
      if (has) {
        apply(p);
        return;
      }
    } catch {
      // Daemon unreachable for this base_url — try the next profile.
    }
  }
  apply(ollama[0] ?? profiles[0]);
}
