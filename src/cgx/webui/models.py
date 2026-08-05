

"""Pydantic request/response models for the CGX web UI.

Every model here is the wire contract between the React frontend and
the FastAPI backend. Optional fields keep the surface forgiving for the
common case where the user has not yet picked a saved profile and is
configuring a provider inline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# --------------------- shared provider config ---------------------

class ProviderConfig(BaseModel):
    """Inline or saved-profile provider configuration."""

    use_profile: bool = False
    profile_name: Optional[str] = None
    kind: str = "ollama"  # "ollama" | "openai-compat" | "gemini" | "custom"
    model: str = "qwen2.5-coder:3b"
    base_url: str = "http://localhost:11434"
    api_key: Optional[str] = None
    temperature: float = 0.2
    num_predict: int = 1024
    # Ollama-only KV-cache context window override. ``None`` / ``0`` means
    # "auto": ``build_provider`` derives a sensible value from the model's
    # registry entry capped at ``DEFAULT_OLLAMA_NUM_CTX``. Other provider
    # kinds ignore this field.
    num_ctx: Optional[int] = None
    endpoint_path: str = "/v1/chat/completions"
    allow_no_auth: bool = False
    # Opt-in reasoning/"thinking" phase for the ASK stream. When True *and*
    # the selected model is reasoning-capable, ``stream_ask`` emits a thought
    # sketch before the grounded answer; otherwise it answers directly. Kept
    # False by default so ASK stays fast unless the user asks for thinking.
    think: bool = False


class IndexLocation(BaseModel):
    """Where the searchable index lives on disk."""

    index_dir: str = "/tmp/cgx_index/indices"
    records: str = "/tmp/cgx_index/records.jsonl"
    embed_model: str = "jinaai/jina-embeddings-v2-base-code"


# --------------------- requests ---------------------

class IndexBuildRequest(BaseModel):
    project_root: Optional[str] = None
    out_dir: str = "/tmp/cgx_index"
    embed_model: str = "jinaai/jina-embeddings-v2-base-code"
    metric: str = "cosine"
    index_type: str = "flat"
    # base64-encoded zip is uploaded via a separate multipart endpoint;
    # this field is the resulting on-disk path the caller wants indexed.
    zip_path: Optional[str] = None


class AskRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    index: IndexLocation = Field(default_factory=IndexLocation)
    provider: ProviderConfig = Field(default_factory=ProviderConfig)


class PlanRequest(BaseModel):
    task: str
    project_root: Optional[str] = None
    self_test: bool = False
    run_tests: bool = False
    index: IndexLocation = Field(default_factory=IndexLocation)
    provider: ProviderConfig = Field(default_factory=ProviderConfig)


class FeedbackRequest(BaseModel):
    """A thumbs up/down (+ optional comment) on an ask/plan result.

    ``run_id`` and the version fields are echoed back from the ``meta`` the
    ask/plan stream returned, so a rating joins to the exact execution.
    """

    rating: str  # "up" | "down"
    run_id: Optional[str] = None
    session_id: Optional[str] = None
    kind: str = "ask"  # "ask" | "plan"
    comment: Optional[str] = None
    question: Optional[str] = None
    answer_preview: Optional[str] = None
    model: Optional[str] = None
    prompt_version: Optional[str] = None
    labels: Dict[str, Any] = Field(default_factory=dict)


class ProfileUpsertRequest(BaseModel):
    name: str
    kind: str = "ollama"  # "ollama" | "openai-compat" | "gemini" | "custom"
    model: str = "qwen2.5-coder:3b"
    base_url: str = "http://localhost:11434"
    api_key: Optional[str] = None
    temperature: float = 0.2
    num_predict: int = 1024
    num_ctx: Optional[int] = None
    endpoint_path: str = "/v1/chat/completions"
    allow_no_auth: bool = False


class AgentProfileUpsertRequest(BaseModel):
    """Save a reusable {objective, project root, mode, skills} bundle.

    Distinct from :class:`ProfileUpsertRequest` (an LLM connection
    preset) -- this is a task template an agent session can be launched
    from.
    """
    name: str
    objective: str
    project_root: str = ""
    mode: str = ""  # "" (auto) | "explore" | "greenfield"
    skills: List[str] = Field(default_factory=list)


class SkillCreateRequest(BaseModel):
    """Full Python source for a new custom skill (one ``Skill`` subclass)."""
    source: str


class SkillUpdateRequest(BaseModel):
    source: str


class SessionCreateRequest(BaseModel):
    title: Optional[str] = None


class RollbackRequest(BaseModel):
    project_root: str
    backup_dir: str


# --------------------- agent-session (Phase 1) ---------------------

class AgentSessionCreateRequest(BaseModel):
    """Spin up a new session-backed agent run.

    ``objective`` is the user's long-form goal -- the session keeps it
    verbatim and the router uses it to seed the root task. Provider and
    index settings mirror the ask/plan requests so the frontend can
    reuse the same configuration UI. ``mode`` is optional: when omitted
    the server auto-detects ``explore`` vs ``greenfield`` from the
    project root and index location.
    """
    objective: str
    project_root: Optional[str] = None
    title: Optional[str] = None
    mode: Optional[str] = None
    index: IndexLocation = Field(default_factory=IndexLocation)
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    run_initial_task: bool = True
    # Explicit skill names to use instead of auto-detecting from the
    # objective text. Empty (the default) preserves today's behavior.
    skills: List[str] = Field(default_factory=list)


class AgentSessionMessageRequest(BaseModel):
    """Follow-up message to an existing session."""
    message: str
    index: IndexLocation = Field(default_factory=IndexLocation)
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    run_initial_task: bool = True


class AgentSessionDecisionRequest(BaseModel):
    """User response to a pending ASK_USER task."""
    task_id: str
    chosen: Dict[str, Any]
    rationale: Optional[str] = None
    index: IndexLocation = Field(default_factory=IndexLocation)
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    run_initial_task: bool = False


class AgentSessionState(BaseModel):
    """Full read of a session's persistent state.

    Returned by every mutating endpoint so the frontend can render the
    new tree / artifacts / facts in one round-trip.
    """
    session: Dict[str, Any]
    tasks: List[Dict[str, Any]] = Field(default_factory=list)
    artifacts: List[Dict[str, Any]] = Field(default_factory=list)
    facts: List[Dict[str, Any]] = Field(default_factory=list)
    decisions: List[Dict[str, Any]] = Field(default_factory=list)


# --------------------- responses ---------------------

class ProfileSummary(BaseModel):
    name: str
    kind: str
    model: str
    base_url: str
    has_api_key: bool
    temperature: float
    num_predict: int
    num_ctx: Optional[int] = None
    endpoint_path: str = "/v1/chat/completions"
    allow_no_auth: bool = False


class AgentProfileSummary(BaseModel):
    name: str
    objective: str
    project_root: str = ""
    mode: str = ""
    skills: List[str] = Field(default_factory=list)


class SkillSummary(BaseModel):
    name: str
    role: str
    aliases: List[str] = Field(default_factory=list)
    description: str = ""
    is_custom: bool = False


class SessionSummary(BaseModel):
    id: str
    title: str
    created_at: float
    updated_at: float
    message_count: int


class SessionMessage(BaseModel):
    role: str
    content: str
    at: Optional[float] = None
    meta: Optional[Dict[str, Any]] = None


class HardwareInfo(BaseModel):
    ram_gb: Optional[float] = None
    gpu_vram_gb: Optional[float] = None
    # Torch CUDA probe -- ``torch_installed`` is False on core-only installs;
    # ``torch_cuda_warning`` is populated when nvidia-smi reports a GPU but
    # ``torch.cuda.is_available()`` is False (usually a wheel/driver mismatch).
    torch_installed: Optional[bool] = None
    torch_cuda_available: Optional[bool] = None
    torch_version: Optional[str] = None
    torch_cuda_build: Optional[str] = None
    torch_cuda_warning: Optional[str] = None


class StatusResponse(BaseModel):
    app: str = "CGX"
    version: str = "0.2.0"
    ollama: Dict[str, Any] = Field(default_factory=dict)
    hardware: HardwareInfo = Field(default_factory=HardwareInfo)
    telemetry_enabled: bool = False
    profile_count: int = 0
    session_count: int = 0
    default_model: str = ""


class ModelChoicesResponse(BaseModel):
    choices: List[str] = Field(default_factory=list)
    recommended_default: str = ""
    installed: List[str] = Field(default_factory=list)
    ollama_reachable: bool = False


class HardwareMatrixRow(BaseModel):
    model: str
    params_b: float
    min_ram_gb: float
    rec_vram_gb: float
    ctx_window: int
    family: str
    fit: str
    reason: str
    notes: str


class TradeoffRow(BaseModel):
    dimension: str
    local: str
    cloud: str
    winner: str


class HardwareMatrixResponse(BaseModel):
    hardware: HardwareInfo
    rows: List[HardwareMatrixRow]
    tradeoffs: List[TradeoffRow]
