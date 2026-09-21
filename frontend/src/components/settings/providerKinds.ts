export type ProviderKind = "ollama" | "openai-compat" | "gemini" | "huggingface" | "custom";

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
  huggingface: {
    // Fixed OpenAI-compatible router host; the backend hardcodes it too, so
    // the Base URL field is hidden for this kind.
    base_url: "https://router.huggingface.co",
    model: "Qwen/Qwen2.5-Coder-32B-Instruct",
    endpoint_path: "/v1/chat/completions",
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
  huggingface: "Hugging Face (Cloud)",
  custom: "Custom Server (OpenAI-Compatible)",
};

export function needsApiKey(kind: ProviderKind): boolean {
  return (
    kind === "openai-compat" || kind === "gemini" || kind === "huggingface" || kind === "custom"
  );
}

export function needsEndpointPath(kind: ProviderKind): boolean {
  return kind === "custom";
}

export function needsBaseUrl(kind: ProviderKind): boolean {
  // Gemini and Hugging Face both use a fixed, backend-hardcoded host.
  return kind !== "gemini" && kind !== "huggingface";
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

// Case-insensitive "is this model already pulled?" check. Ollama tags are
// matched exactly at generation time, but the installed list may differ only
// in case (a re-aliased ``hf.co`` pull lands lowercased), so compare loosely
// here to avoid showing "not installed" for a tag that is in fact present.
export function modelInstalled(installed: string[], model: string): boolean {
  if (!model) return false;
  const needle = model.toLowerCase();
  return installed.some((m) => m.toLowerCase() === needle);
}

// Map a model name to the correct ``ollama pull`` target plus the local tag it
// will resolve to. A plain Ollama tag (``qwen2.5-coder:3b``) pulls as-is. A
// Hugging Face repo id (``owner/Repo`` or ``hf.co/owner/repo``) must be pulled
// via the ``hf.co/<repo>`` form and lands under a lowercased local alias, which
// the caller should adopt as the active model so later exact-match pings pass.
export function ollamaPullPlan(model: string): { target: string; localName?: string } {
  const m = (model || "").trim();
  if (!m) return { target: m };
  if (m.startsWith("hf.co/")) {
    const repo = m.slice("hf.co/".length).split(":")[0];
    const leaf = repo.split("/").pop() || repo;
    return { target: m, localName: leaf.toLowerCase() };
  }
  // ``owner/repo`` Hugging Face id — has a slash but no scheme; route via hf.co.
  if (m.includes("/")) {
    const leaf = m.split("/").pop() || m;
    return { target: `hf.co/${m}`, localName: leaf.toLowerCase() };
  }
  // A bare tag or a slash-less GGUF display name: pull as-is. If it isn't a
  // real registry tag Ollama will report it; nothing we can rewrite safely.
  return { target: m };
}
