// Tiny fetch wrapper. The dev server proxies /api → :8765 (see vite.config),
// so we always use the relative path and let same-origin / proxy take over.

import { streamSSE } from "./sse";

export type ProviderConfig = {
  use_profile: boolean;
  profile_name?: string | null;
  kind: "ollama" | "openai-compat" | "gemini" | "huggingface" | "custom";
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

// Distinct from ProfileSummary (an LLM connection preset) -- an Agent
// Profile is a saved {task, skills} bundle a session can be launched from.
export type AgentProfileSummary = {
  name: string;
  objective: string;
  project_root: string;
  mode: string; // "" (auto) | "explore" | "greenfield"
  skills: string[];
};

export type SkillSummary = {
  name: string;
  role: string;
  aliases: string[];
  description: string;
  is_custom: boolean;
};

export type SkillValidationError = {
  error_kind: string;
  error_detail: string;
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
  status: "start" | "stream" | "done" | "failed" | string;
  bytes?: number;
  elapsed_ms?: number;
  eta_seconds: number | null;
  // Running count of files that failed so far. On failure ``index`` does not
  // advance, so this lets the UI surface the failure instead of showing what
  // looks like a counter reset.
  failed_count?: number;
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

// --- User Activity (Subsystem C): per-run observation store ---
export type RunRecord = {
  run_id: string;
  created_at: number;
  kind: string;
  model?: string | null;
  prompt_version?: string | null;
  owner?: string | null;
  project_root?: string | null;
  tokens_in?: number | null;
  tokens_out?: number | null;
  tokens_total?: number | null;
  cost_usd?: number | null;
  latency_ms?: number | null;
  n_sources: number;
  n_citations: number;
  confidence?: number | null;
  grounded?: boolean | null;
  status: string;
  question: string;
  labels: Record<string, any>;
};

export type ActivitySummary = {
  total: number;
  cost_usd: number;
  tokens_total: number;
  errors: number;
  by_kind: Record<string, { runs: number; cost_usd: number; tokens_total: number; errors: number }>;
};

export type RunDetail = {
  run: RunRecord;
  feedback: Array<Record<string, any>>;
  alerts: Array<Record<string, any>>;
};

// --- Admin (Subsystem D): logs / trace explorer, metrics, audit-lite ---
export type AdminLogEntry = Record<string, any> & { event?: string; ts?: number };

export type MetricSeries = { name: string; labels: Record<string, string>; value: number };
export type HistogramSeries = {
  name: string;
  labels: Record<string, string>;
  count: number;
  sum: number;
  buckets: Array<[number | string, number]>;
};
export type MetricsSnapshot = {
  counters: MetricSeries[];
  gauges: MetricSeries[];
  histograms: HistogramSeries[];
};

export type AdminOverview = {
  activity: Partial<ActivitySummary>;
  http: { requests: number; errors: number };
  feedback: Record<string, any>;
  alerts: {
    total: number;
    by_severity: Record<string, number>;
    recent: Array<Record<string, any>>;
  };
};

// --- AIOps monitoring (Subsystem G): persisted alerts ---
export type MonitorAlert = {
  alert_id: string;
  created_at: number;
  code: string;
  severity: "info" | "warning" | "critical" | string;
  run_id?: string | null;
  value?: number | null;
  threshold?: number | null;
  message: string;
  labels: Record<string, any>;
};

// --- Feedback loop (Subsystem H) ---
export type FeedbackStats = {
  total: number;
  up: number;
  down: number;
  satisfaction: number | null;
  by_kind: Record<string, { up: number; down: number }>;
};
export type FeedbackRow = {
  feedback_id: string;
  created_at: number;
  rating: "up" | "down" | string;
  run_id?: string | null;
  session_id?: string | null;
  kind: string;
  comment: string;
  question: string;
  answer_preview: string;
  model?: string | null;
  prompt_version?: string | null;
  labels: Record<string, any>;
};

// --- Cost & quota governance (Subsystem I) ---
export type OwnerUsage = {
  owner: string;
  state: "ok" | "warn" | "exceeded" | string;
  cost_used: number;
  cost_limit: number;
  tokens_used: number;
  tokens_limit: number;
  day?: string;
  tokens_in?: number;
  tokens_out?: number;
  tokens_total?: number;
  cost_usd?: number;
  calls?: number;
};
export type UsageRow = {
  owner: string;
  day: string;
  tokens_in: number;
  tokens_out: number;
  tokens_total: number;
  cost_usd: number;
  calls: number;
};

// --- Data governance (Subsystem M) ---
export type GovPolicy = {
  retention_days: number;
  store_full_text: boolean;
  scrub_pii: boolean;
  preview_cap: number;
};
export type GovScanResult = {
  findings: Array<{ type: string; count: number }>;
  total: number;
  scrubbed: string;
};

// --- Reliability & health (Subsystem J) ---
export type HealthCheck = {
  name: string;
  ok: boolean;
  critical: boolean;
  detail: Record<string, any>;
};
export type LivenessReport = { status: string; checks: HealthCheck[] };
export type ReadinessReport = {
  status: string;
  ready: boolean;
  ts?: number;
  checks: HealthCheck[];
};

// --- Hugging Face Hub browse (GGUF models pullable via Ollama) ---
export type HfHubModel = {
  id: string;
  downloads: number;
  likes: number;
  pipeline_tag?: string | null;
  gated: boolean;
  // Ready-to-use ``ollama pull`` target, e.g. ``hf.co/<repo>``.
  pull_tag: string;
  // Detected quantization labels (``Q4_K_M`` …) for ``pull_tag:<quant>``.
  quants: string[];
};

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
  hfModels: (params: { search?: string; sort?: string; limit?: number } = {}) => {
    const q = new URLSearchParams();
    if (params.search) q.set("search", params.search);
    if (params.sort) q.set("sort", params.sort);
    if (params.limit) q.set("limit", String(params.limit));
    const qs = q.toString();
    return jsonReq<{ models: HfHubModel[] }>(
      `/api/setup/hf_models${qs ? `?${qs}` : ""}`,
    );
  },
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

  listAgentProfiles: () => jsonReq<AgentProfileSummary[]>("/api/agent-profiles"),
  upsertAgentProfile: (name: string, body: Omit<AgentProfileSummary, "name">) =>
    jsonReq<AgentProfileSummary>(
      `/api/agent-profiles/${encodeURIComponent(name)}`,
      "PUT",
      { name, ...body },
    ),
  deleteAgentProfile: (name: string) =>
    jsonReq<{ deleted: string }>(
      `/api/agent-profiles/${encodeURIComponent(name)}`,
      "DELETE",
    ),

  listSkills: () => jsonReq<SkillSummary[]>("/api/skills"),
  getSkillSource: (name: string) =>
    jsonReq<{ name: string; source: string }>(
      `/api/skills/${encodeURIComponent(name)}/source`,
    ),
  createSkill: (source: string) =>
    jsonReq<SkillSummary>("/api/skills", "POST", { source }),
  updateSkill: (name: string, source: string) =>
    jsonReq<SkillSummary>(`/api/skills/${encodeURIComponent(name)}`, "PUT", { source }),
  deleteSkill: (name: string) =>
    jsonReq<{ deleted: string }>(`/api/skills/${encodeURIComponent(name)}`, "DELETE"),

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

  // --- session-shaped agent (Phase 4) ---
  agentSessionCreate: (body: {
    objective: string;
    project_root?: string | null;
    title?: string | null;
    mode?: SessionModeValue | null;
    index: IndexLocation;
    provider: ProviderConfig;
    run_initial_task?: boolean;
    skills?: string[];
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

  // --- User Activity (Subsystem C) ---
  activityRuns: (params?: { kind?: string; owner?: string; status?: string; limit?: number }) => {
    const q = new URLSearchParams();
    if (params?.kind) q.set("kind", params.kind);
    if (params?.owner) q.set("owner", params.owner);
    if (params?.status) q.set("status", params.status);
    if (params?.limit) q.set("limit", String(params.limit));
    const qs = q.toString();
    return jsonReq<{ runs: RunRecord[]; count: number }>(`/api/activity/runs${qs ? `?${qs}` : ""}`);
  },
  activitySummary: () => jsonReq<ActivitySummary>("/api/activity/summary"),
  activityRunDetail: (runId: string) =>
    jsonReq<RunDetail>(`/api/activity/runs/${encodeURIComponent(runId)}`),

  // --- Admin (Subsystem D) ---
  adminOverview: () => jsonReq<AdminOverview>("/api/admin/overview"),
  adminMetrics: () => jsonReq<MetricsSnapshot>("/api/admin/metrics"),
  adminLogs: (params?: { event?: string; limit?: number; project_root?: string }) => {
    const q = new URLSearchParams();
    if (params?.event) q.set("event", params.event);
    if (params?.limit) q.set("limit", String(params.limit));
    if (params?.project_root) q.set("project_root", params.project_root);
    const qs = q.toString();
    return jsonReq<{ source: string; logs: AdminLogEntry[]; count: number }>(
      `/api/admin/logs${qs ? `?${qs}` : ""}`,
    );
  },

  // --- AIOps monitoring (Subsystem G) ---
  monitorAlerts: (params?: { severity?: string; code?: string; limit?: number; since?: number }) => {
    const q = new URLSearchParams();
    if (params?.severity) q.set("severity", params.severity);
    if (params?.code) q.set("code", params.code);
    if (params?.limit) q.set("limit", String(params.limit));
    if (params?.since) q.set("since", String(params.since));
    const qs = q.toString();
    return jsonReq<{ alerts: MonitorAlert[]; count: number }>(
      `/api/monitor/alerts${qs ? `?${qs}` : ""}`,
    );
  },

  // --- Feedback loop (Subsystem H) ---
  feedbackStats: (since?: number) =>
    jsonReq<FeedbackStats>(`/api/feedback/stats${since ? `?since=${since}` : ""}`),
  feedbackList: (params?: { rating?: string; kind?: string; run_id?: string; limit?: number }) => {
    const q = new URLSearchParams();
    if (params?.rating) q.set("rating", params.rating);
    if (params?.kind) q.set("kind", params.kind);
    if (params?.run_id) q.set("run_id", params.run_id);
    if (params?.limit) q.set("limit", String(params.limit));
    const qs = q.toString();
    return jsonReq<{ feedback: FeedbackRow[]; count: number }>(
      `/api/feedback${qs ? `?${qs}` : ""}`,
    );
  },

  // --- Cost & quota governance (Subsystem I) ---
  usage: (owner?: string, day?: string) => {
    const q = new URLSearchParams();
    if (owner) q.set("owner", owner);
    if (day) q.set("day", day);
    const qs = q.toString();
    return jsonReq<OwnerUsage>(`/api/usage${qs ? `?${qs}` : ""}`);
  },
  usageSummary: (day?: string) =>
    jsonReq<{ usage: UsageRow[]; count: number }>(`/api/usage/summary${day ? `?day=${day}` : ""}`),

  // --- Data governance (Subsystem M) ---
  govPolicy: () => jsonReq<GovPolicy>("/api/govdata/policy"),
  govScan: (text: string) => jsonReq<GovScanResult>("/api/govdata/scan", "POST", { text }),
  govPurge: (retention_days?: number | null) =>
    jsonReq<{ ok: boolean; deleted: Record<string, number>; total: number }>(
      "/api/govdata/purge",
      "POST",
      { retention_days: retention_days ?? null },
    ),
  govErase: (body: { run_id?: string; owner?: string }) =>
    jsonReq<{ ok: boolean; deleted: Record<string, number>; total: number }>(
      "/api/govdata/erase",
      "POST",
      body,
    ),

  // --- Reliability & health (Subsystem J) ---
  // Probes live at the root (not under /api); the dev proxy forwards them too.
  liveness: () => jsonReq<LivenessReport>("/healthz"),
  // /readyz returns 503 when not ready, so parse the body regardless of status
  // rather than letting jsonReq throw on the non-2xx.
  readiness: async (): Promise<ReadinessReport> => {
    const res = await fetch("/readyz");
    return (await res.json()) as ReadinessReport;
  },
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
