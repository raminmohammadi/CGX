

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
    endpoint_path: str = "/v1/chat/completions"
    allow_no_auth: bool = False


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


class AgentRequest(BaseModel):
    goal: str
    project_root: Optional[str] = None
    stop_on_fail: bool = True
    index: IndexLocation = Field(default_factory=IndexLocation)
    provider: ProviderConfig = Field(default_factory=ProviderConfig)


class ProfileUpsertRequest(BaseModel):
    name: str
    kind: str = "ollama"  # "ollama" | "openai-compat" | "gemini" | "custom"
    model: str = "qwen2.5-coder:3b"
    base_url: str = "http://localhost:11434"
    api_key: Optional[str] = None
    temperature: float = 0.2
    num_predict: int = 1024
    endpoint_path: str = "/v1/chat/completions"
    allow_no_auth: bool = False


class SessionCreateRequest(BaseModel):
    title: Optional[str] = None


class RollbackRequest(BaseModel):
    project_root: str
    backup_dir: str


# --------------------- responses ---------------------

class ProfileSummary(BaseModel):
    name: str
    kind: str
    model: str
    base_url: str
    has_api_key: bool
    temperature: float
    num_predict: int
    endpoint_path: str = "/v1/chat/completions"
    allow_no_auth: bool = False


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
