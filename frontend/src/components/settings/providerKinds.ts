export type ProviderKind = "ollama" | "openai-compat" | "gemini" | "custom";

export interface ProfileEditState {
  name: string;
  kind: ProviderKind;
  model: string;
  base_url: string;
  api_key: string;
  temperature: number;
  num_predict: number;
  num_ctx: number | null;
  endpoint_path: string;
  allow_no_auth: boolean;
}

export const KIND_DEFAULTS: Record<ProviderKind, Partial<ProfileEditState>> = {
  ollama: {
    base_url: "http://localhost:11434",
    model: "qwen2.5-coder:3b",
    api_key: "",
    endpoint_path: "/v1/chat/completions",
    allow_no_auth: false,
  },
  "openai-compat": {
    base_url: "https://api.openai.com",
    model: "gpt-4o-mini",
    endpoint_path: "/v1/chat/completions",
    allow_no_auth: false,
  },
  gemini: {
    base_url: "https://generativelanguage.googleapis.com",
    model: "gemini-2.5-flash",
    endpoint_path: "/v1beta/models",
    allow_no_auth: false,
  },
  custom: {
    base_url: "",
    model: "",
    endpoint_path: "/v1/chat/completions",
    allow_no_auth: false,
  },
};

export const emptyEdit: ProfileEditState = {
  name: "",
  kind: "ollama",
  model: "qwen2.5-coder:3b",
  base_url: "http://localhost:11434",
  api_key: "",
  temperature: 0.2,
  num_predict: 1024,
  num_ctx: null,
  endpoint_path: "/v1/chat/completions",
  allow_no_auth: false,
};

export const KIND_LABELS: Record<ProviderKind, string> = {
  ollama: "Ollama (Local)",
  "openai-compat": "OpenAI (Cloud)",
  gemini: "Google Gemini (Cloud)",
  custom: "Custom Server (OpenAI-Compatible)",
};

export function needsApiKey(kind: ProviderKind): boolean {
  return kind === "openai-compat" || kind === "gemini" || kind === "custom";
}

export function needsEndpointPath(kind: ProviderKind): boolean {
  return kind === "custom";
}

export function needsBaseUrl(kind: ProviderKind): boolean {
  return kind !== "gemini";
}

// Show the pull button whenever Ollama is reachable and the model isn't installed yet.
export function showPullButton(
  kind: ProviderKind,
  model: string,
  installed: string[],
  reachable: boolean,
  pulling: boolean,
): boolean {
  return kind === "ollama" && !!model && reachable && !installed.includes(model) && !pulling;
}
