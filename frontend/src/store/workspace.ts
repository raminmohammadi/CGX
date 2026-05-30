import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { IndexLocation, ProviderConfig } from "../lib/api";

// Workspace store: per-browser persistence of the provider config, the
// active index location, and the selected session. The page components
// read/write these to keep Setup → Ask → Plan → Agent consistent.

export interface WorkspaceState {
  provider: ProviderConfig;
  index: IndexLocation;
  projectRoot: string;
  selectedSessionId: string | null;
  setProvider: (patch: Partial<ProviderConfig>) => void;
  setIndex: (patch: Partial<IndexLocation>) => void;
  setProjectRoot: (root: string) => void;
  setSelectedSession: (id: string | null) => void;
  applyProfile: (profile: { name: string; kind: string; model: string;
    base_url: string; temperature: number; num_predict: number }) => void;
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
};

const defaultIndex: IndexLocation = {
  index_dir: "/tmp/averix_index/indices",
  records: "/tmp/averix_index/records.jsonl",
  embed_model: "jinaai/jina-embeddings-v2-base-code",
};

export const useWorkspace = create<WorkspaceState>()(
  persist(
    (set) => ({
      provider: defaultProvider,
      index: defaultIndex,
      projectRoot: "",
      selectedSessionId: null,
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
          },
        })),
    }),
    { name: "averix-workspace" },
  ),
);
