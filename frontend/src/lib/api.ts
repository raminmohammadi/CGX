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
  // Opt-in reasoning/"thinking" phase for ASK. The backend only honors it
  // when the selected model is reasoning-capable; otherwise it answers
  // directly. Undefined is treated as false.
  think?: boolean;
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

// --- session-shaped agent types (mirror cgx.session models) ---

export type SessionModeValue = "explore" | "greenfield";

export type TaskKind =
  | "explore"
  | "investigate"
  | "recommend"
  | "plan_change"
  | "apply"
  | "verify"
  | "ask_user"
  | "search"
  | "clarify_requirements"
  | "decompose"
  | "scaffold"
  | "bootstrap_env"
  | "repair"
  | "summarize";

export type TaskNodeStatus =
  | "pending"
  | "blocked"
  | "ready"
  | "in_progress"
  | "done"
  | "failed"
  | "abandoned";

export type ArtifactKind =
  | "directions_list"
  | "findings_bundle"
  | "recommendation_list"
  | "code_change_plan"
  | "applied_changes"
  | "verify_report"
  | "session_digest"
  | "requirements_sheet"
  | "work_plan"
  | "scaffold_patches"
  | "build_report"
  | "repair_plan"
  | "smoke_report"
  | "api_check_report";

export type FactKind =
  | "file" | "symbol" | "parameter" | "anchor" | "llm_call";

export type DecisionKind =
  | "choose_path"
  | "choose_recommendation"
  | "approve"
  | "freeform"
  | "clarify_answers"
  | "approve_plan";

export type SessionStatusValue =
  | "active" | "paused" | "completed" | "abandoned";

export interface AgentSessionSummary {
  session_id: string;
  title: string;
  original_objective: string;
  status: SessionStatusValue;
  mode: SessionModeValue;
  current_focus: string | null;
  root_task_id: string | null;
  project_root: string | null;
  created_at: number;
  updated_at: number;
}

export interface TaskNodeDTO {
  task_id: string;
  session_id: string;
  kind: TaskKind;
  name: string;
  description: string;
  parent_task_id: string | null;
  status: TaskNodeStatus;
  inputs: Record<string, any>;
  outputs: Record<string, any> | null;
  error: string | null;
  blockers: string[];
  children: string[];
  consumed_decision_ids: string[];
  produced_artifact_id: string | null;
  created_at: number;
  started_at: number | null;
  completed_at: number | null;
}

export interface ArtifactDTO {
  artifact_id: string;
  session_id: string;
  produced_by_task_id: string;
  kind: ArtifactKind;
  content: Record<string, any>;
  created_at: number;
}

export interface FactDTO {
  fact_id: string;
  session_id: string;
  kind: FactKind;
  content: Record<string, any>;
  surfaced_in_task_id: string | null;
  stale: boolean;
  created_at: number;
  updated_at: number;
}

export interface DecisionDTO {
  decision_id: string;
  session_id: string;
  resolved_task_id: string;
  kind: DecisionKind;
  question: string;
  chosen: Record<string, any>;
  rationale: string | null;
  made_at: number;
}

// Transient intra-task progress carried by ``task.output_partial`` SSE
// frames (not persisted). SCAFFOLD emits one per file so the UI can show
// a live "i / total", the active path, and a coarse ETA.
export interface TaskProgress {
  index: number;
  total: number;
  path: string;
  layer?: string;
  status: "start" | "done" | "failed" | string;
  bytes?: number;
  elapsed_ms?: number;
  eta_seconds: number | null;
}

export interface AgentSessionState {
  session: AgentSessionSummary;
  tasks: TaskNodeDTO[];
  artifacts: ArtifactDTO[];
  facts: FactDTO[];
  decisions: DecisionDTO[];
}

// Typed error thrown by ``jsonReq`` whenever the response is not 2xx.
// Callers can ``instanceof ApiError`` and branch on ``status`` to react
// to specific failures (e.g. 404 on a session id means the persisted
// active id is stale and should be cleared rather than retried).
export class ApiError extends Error {
  readonly status: number;
  readonly path: string;
  readonly body: string;
  constructor(method: string, path: string, status: number, body: string) {
    super(`${method} ${path} → ${status}: ${body.slice(0, 200)}`);
    this.name = "ApiError";
    this.status = status;
    this.path = path;
    this.body = body;
  }
}

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
    throw new ApiError(method, path, res.status, text);
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

  // --- session-shaped agent (Phase 4) ---
  agentSessionCreate: (body: {
    objective: string;
    project_root?: string | null;
    title?: string | null;
    mode?: SessionModeValue | null;
    index: IndexLocation;
    provider: ProviderConfig;
    run_initial_task?: boolean;
  }) => jsonReq<AgentSessionState>("/api/agent-session", "POST", body),
  agentSessionList: (projectRoot?: string | null) => {
    const q = projectRoot
      ? `?project_root=${encodeURIComponent(projectRoot)}`
      : "";
    return jsonReq<AgentSessionSummary[]>(`/api/agent-session${q}`);
  },
  agentSessionGet: (sid: string, projectRoot?: string | null) => {
    const q = projectRoot
      ? `?project_root=${encodeURIComponent(projectRoot)}`
      : "";
    return jsonReq<AgentSessionState>(
      `/api/agent-session/${encodeURIComponent(sid)}${q}`,
    );
  },
  agentSessionMessage: (sid: string, body: {
    message: string;
    index: IndexLocation;
    provider: ProviderConfig;
    run_initial_task?: boolean;
  }) =>
    jsonReq<AgentSessionState>(
      `/api/agent-session/${encodeURIComponent(sid)}/message`,
      "POST",
      body,
    ),
  agentSessionDecision: (sid: string, body: {
    task_id: string;
    chosen: Record<string, any>;
    rationale?: string | null;
    index: IndexLocation;
    provider: ProviderConfig;
    run_initial_task?: boolean;
  }) =>
    jsonReq<AgentSessionState>(
      `/api/agent-session/${encodeURIComponent(sid)}/decision`,
      "POST",
      body,
    ),
  // Same-origin SSE stream URL for a session's live events. Consumed by
  // the native ``EventSource`` (GET); the dev proxy forwards /api to :8765.
  agentSessionEventsUrl: (sid: string) =>
    `/api/agent-session/${encodeURIComponent(sid)}/events`,
  agentSessionCancel: (sid: string) =>
    jsonReq<AgentSessionState>(
      `/api/agent-session/${encodeURIComponent(sid)}/cancel`,
      "POST",
    ),
  agentSessionDelete: (sid: string, projectRoot?: string | null) => {
    const q = projectRoot
      ? `?project_root=${encodeURIComponent(projectRoot)}`
      : "";
    return jsonReq<{ deleted: string }>(
      `/api/agent-session/${encodeURIComponent(sid)}${q}`,
      "DELETE",
    );
  },

  getTraceSettings: () =>
    jsonReq<TraceSettings>("/api/settings/trace"),
  setTraceSettings: (enabled: boolean) =>
    jsonReq<TraceSettings>("/api/settings/trace", "POST", { enabled }),
};

export type TraceSettings = {
  enabled: boolean;
  source: "env" | "runtime";
};

export type RollbackResponse = {
  restored_files: string[];
  deleted_files: string[];
  failed_files: { file: string; error: string }[];
  error?: string | null;
};
