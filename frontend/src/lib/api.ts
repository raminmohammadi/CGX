// Tiny fetch wrapper. The dev server proxies /api → :8765 (see vite.config),
// so we always use the relative path and let same-origin / proxy take over.

import { streamSSE } from "./sse";

export type ProviderConfig = {
  use_profile: boolean;
  profile_name?: string | null;
  kind: "ollama" | "openai-compat" | "gemini" | "custom";
  model: string;
  base_url: string;
  api_key?: string | null;
  temperature: number;
  num_predict: number;
  // Ollama-only KV-cache window. ``null`` / undefined means "auto" (backend
  // picks a sensible default capped at 8K). Other provider kinds ignore this.
  num_ctx?: number | null;
  endpoint_path?: string;
  allow_no_auth?: boolean;
};

export type IndexLocation = {
  index_dir: string;
  records: string;
  embed_model: string;
};

export type ProfileSummary = {
  name: string;
  kind: string;
  model: string;
  base_url: string;
  has_api_key: boolean;
  temperature: number;
  num_predict: number;
  num_ctx?: number | null;
  endpoint_path?: string;
  allow_no_auth?: boolean;
};

export type RunningModel = {
  name: string;
  model?: string;
  size?: number | null;
  size_vram?: number | null;
  context_length?: number | null;
  expires_at?: string | null;
  digest?: string | null;
};

export type PingResult = {
  ok: boolean;
  latency_ms?: number | null;
  error?: string | null;
};

export type SessionSummary = {
  id: string;
  title: string;
  created_at: number;
  updated_at: number;
  message_count: number;
};

export type SessionMessage = {
  role: "user" | "assistant" | string;
  content: string;
  at?: number | null;
  meta?: Record<string, any> | null;
};

export type HardwareInfo = {
  ram_gb?: number | null;
  gpu_vram_gb?: number | null;
  // Torch CUDA probe surfaced by the backend so the Header can render an
  // Embed pill / warning. ``torch_installed`` is null on core-only installs.
  torch_installed?: boolean | null;
  torch_cuda_available?: boolean | null;
  torch_version?: string | null;
  torch_cuda_build?: string | null;
  torch_cuda_warning?: string | null;
};

export type StatusResponse = {
  app: string;
  version: string;
  ollama: {
    ok?: boolean;
    error?: string;
    running_models?: RunningModel[];
    [k: string]: any;
  };
  hardware: HardwareInfo;
  telemetry_enabled: boolean;
  profile_count: number;
  session_count: number;
  default_model: string;
};

export type HardwareMatrixRow = {
  model: string;
  params_b: number;
  min_ram_gb: number;
  rec_vram_gb: number;
  ctx_window: number;
  family: string;
  fit: "fits" | "tight" | "won't fit" | string;
  reason: string;
  notes: string;
};

export type TradeoffRow = {
  dimension: string;
  local: string;
  cloud: string;
  winner: string;
};

export type HardwareMatrixResponse = {
  hardware: HardwareInfo;
  rows: HardwareMatrixRow[];
  tradeoffs: TradeoffRow[];
};

export type EmbedModelInfo = {
  name: string;
  label: string;
  kind: string;
  dim: number;
  max_tokens: number;
  size_gb: number;
  description: string;
  cached: boolean;
};

export type EmbedModelsResponse = {
  choices: EmbedModelInfo[];
  recommended_default: string;
};

async function jsonReq<T>(
  path: string,
  method: "GET" | "POST" | "PUT" | "DELETE" = "GET",
  body?: unknown,
): Promise<T> {
  const res = await fetch(path, {
    method,
    headers: body ? { "content-type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${method} ${path} → ${res.status}: ${text.slice(0, 200)}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  status: () => jsonReq<StatusResponse>("/api/status"),
  ollamaHealth: (base_url: string) =>
    jsonReq<{ ok: boolean; error?: string }>(
      `/api/health/ollama?base_url=${encodeURIComponent(base_url)}`,
    ),
  setupModels: (base_url: string) =>
    jsonReq<{ choices: string[]; recommended_default: string; installed: string[]; ollama_reachable: boolean }>(
      `/api/setup/models?base_url=${encodeURIComponent(base_url)}`,
    ),

  ollamaPull: (
    model: string,
    base_url: string,
    onProgress: (data: { status?: string; total?: number; completed?: number; error?: string }) => void,
    onDone: () => void,
    onError?: (err: unknown) => void,
  ) => streamSSE(
    "/api/ollama/pull",
    { model, base_url },
    (event, data) => {
      if (event === "progress") onProgress(data);
      else if (event === "done") onDone();
    },
    onError,
  ),
  embedModels: () => jsonReq<EmbedModelsResponse>("/api/embed/models"),
  embedPull: (
    model: string,
    onProgress: (data: { status?: string; total?: number; completed?: number; error?: string }) => void,
    onDone: () => void,
    onError?: (err: unknown) => void,
  ) => streamSSE(
    "/api/embed/pull",
    { model },
    (event, data) => {
      if (event === "progress") onProgress(data);
      else if (event === "done") onDone();
    },
    onError,
  ),
  cloudModels: (body: {
    kind: string;
    base_url?: string | null;
    api_key?: string | null;
    profile_name?: string | null;
  }) =>
    jsonReq<{ choices: string[]; recommended_default: string }>(
      "/api/setup/cloud_models",
      "POST",
      body,
    ),
  hardwareMatrix: () => jsonReq<HardwareMatrixResponse>("/api/hardware/matrix"),
  detectHardware: () => jsonReq<HardwareInfo>("/api/setup/hardware"),

  listSessions: () => jsonReq<SessionSummary[]>("/api/sessions"),
  createSession: (title?: string) =>
    jsonReq<SessionSummary>("/api/sessions", "POST", { title: title || null }),
  sessionMessages: (sid: string) =>
    jsonReq<SessionMessage[]>(`/api/sessions/${encodeURIComponent(sid)}/messages`),
  deleteSession: (sid: string) =>
    jsonReq<{ deleted: string }>(
      `/api/sessions/${encodeURIComponent(sid)}`,
      "DELETE",
    ),

  listProfiles: () => jsonReq<ProfileSummary[]>("/api/profiles"),
  upsertProfile: (name: string, body: any) =>
    jsonReq<ProfileSummary>(
      `/api/profiles/${encodeURIComponent(name)}`,
      "PUT",
      body,
    ),
  deleteProfile: (name: string) =>
    jsonReq<{ deleted: string }>(
      `/api/profiles/${encodeURIComponent(name)}`,
      "DELETE",
    ),
  pingProvider: (body: {
    kind: string;
    base_url: string;
    model: string;
    api_key?: string | null;
    endpoint_path?: string;
    allow_no_auth?: boolean;
  }) => jsonReq<PingResult>("/api/provider/ping", "POST", body),

  uploadZip: async (file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch("/api/index/upload", { method: "POST", body: fd });
    if (!res.ok) throw new Error(`upload failed: ${res.status}`);
    return res.json() as Promise<{ path: string; original_name: string; size_bytes: number }>;
  },

  rollback: (project_root: string, backup_dir: string) =>
    jsonReq<RollbackResponse>("/api/rollback", "POST", { project_root, backup_dir }),

  agentPlan: (body: {
    goal: string;
    project_root?: string | null;
    stop_on_fail?: boolean;
    index: IndexLocation;
    provider: ProviderConfig;
  }) =>
    jsonReq<{
      plan?: { id: string; goal: string; tasks: any[]; rationale?: string };
      error?: string;
    }>(
      "/api/agent/plan",
      "POST",
      body,
    ),
};

export type RollbackResponse = {
  restored_files: string[];
  deleted_files: string[];
  failed_files: { file: string; error: string }[];
  error?: string | null;
};
