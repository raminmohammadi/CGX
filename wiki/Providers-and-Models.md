# Providers and Models

CGX is model-agnostic. Every generation path (Ask, Plan, Agent) talks to
an `LLMProvider` through a uniform `chat()` / `chat_stream()` interface,
so switching backends never changes the rest of the system. This page
covers the four provider types and the hardware-aware model picker.

---

## Supported providers

| Provider type          | Kind string(s)              | Where it runs |
|------------------------|-----------------------------|---------------|
| **Ollama (local)**     | `ollama`                    | `http://localhost:11434` (loopback) |
| **OpenAI (cloud)**     | `openai`                    | `https://api.openai.com` |
| **OpenAI-compatible**  | `openai-compat`             | Any `/v1/chat/completions` endpoint (Groq, Together, DeepSeek, vLLM, …) |
| **Google Gemini**      | `gemini`                    | `generativelanguage.googleapis.com` |
| **Custom server**      | `custom`                    | Your self-hosted endpoint (custom path, optional auth-bypass) |

Choose a provider from the **Setup** tab's *Provider Type* dropdown, or
pass `--provider` on the CLI. A **Ping** button (and
`POST /api/provider/ping`) runs a live connection test that reports
latency or the exact error.

---

## Ollama (default, local)

```bash
ollama serve
ollama pull qwen2.5-coder:3b
```

Set **Provider Type → Ollama (Local)**. Default base URL
`http://localhost:11434`; Ping exercises `GET /api/tags`. This path is
fully offline — nothing leaves your machine.

## OpenAI / OpenAI-compatible (cloud)

Set **Provider Type → OpenAI (Cloud)**, supply `OPENAI_API_KEY`, and pick
a model (`gpt-4o-mini`, `gpt-4o`, …). Any OpenAI-compatible endpoint works
here — set the base URL accordingly.

## Google Gemini (cloud)

Set **Provider Type → Google Gemini (Cloud)**, supply `GEMINI_API_KEY`,
and pick a model (`gemini-1.5-flash`, `gemini-1.5-pro`, …). Ping sends a
minimal `generateContent` request with `maxOutputTokens: 1`.

```python
from cgx.answer.providers import GeminiProvider
prov = GeminiProvider(model="gemini-1.5-flash", api_key="YOUR_KEY")
# or set GEMINI_API_KEY and omit api_key
```

## Custom server (OpenAI-compatible)

For a self-hosted model on a private subnet:

- **Host IP/URL** — e.g. `http://100.10.20.10:8080`
- **Endpoint Path** — the exact path suffix, e.g. `/completion` (default
  `/v1/chat/completions`)
- **Bearer Token** — optional; leave blank and tick **Skip auth** for
  servers that need no authentication.

```python
from cgx.answer.providers import OpenAICompatProvider
prov = OpenAICompatProvider(
    model="my-model",
    base_url="http://100.10.20.10:8080",
    endpoint_path="/completion",
    allow_no_auth=True,
)
```

---

## Profiles

Save any configuration as a named **Profile** (Profiles tab or
`cgx.answer.profiles.save_profile`). A profile persists `kind`, `model`,
`base_url`, sampling params, `endpoint_path`, `allow_no_auth`,
`enable_reranker`, and optional per-profile `rate_limit` / `max_retries`.

API keys are stored in the OS keyring when the `keyring` extra is
installed; otherwise in `~/.cgx/secrets.json` with `0600` permissions.
**Keys are never passed on the command line** — cloud providers read them
from the environment or the keyring-backed store. See
**[[Privacy and Security]]**.

---

## Schema-constrained decoding

Every `chat()` accepts an optional `json_schema`. When combined with
`force_json`, each provider requests structured output natively (Ollama
`format`, OpenAI `response_format`, Gemini `responseSchema`). Any backend
that rejects the schema degrades gracefully down a
`json_schema → json_object → plain` ladder, with a balanced-brace
extractor as the final safety net — so weaker models fall back cleanly
instead of failing.

---

## Rate limiting & retries

Every HTTP-backed provider goes through `cgx.answer.ratelimit`: a
thread-safe token-bucket limiter plus exponential-backoff retry
(honouring `Retry-After`) on HTTP **429** and **5xx**. Configure per
profile with `rate_limit` (req/sec) and `max_retries`. Setting
`rate_limit=None` (the default) makes the limiter a no-op. Details in
**[[Configuration and Tuning]]**.

---

## Hardware-aware model picker

The **Hardware** tab annotates a static catalogue of **21**
locally-runnable models (families: `coder`, `reasoning`, `general`)
against the RAM/VRAM detected by `detect_hardware()`. Each row reports:

| Column | Meaning |
|--------|---------|
| `name`                | Ollama tag (e.g. `qwen2.5-coder:3b`) |
| `params_b`            | Approx parameter count (billions) |
| `min_ram_gb`          | Lower bound for 4-bit quantised inference |
| `recommended_vram_gb` | VRAM for smooth throughput |
| `ctx_window`          | Max advertised prompt window |
| `family`              | `coder` / `reasoning` / `general` |
| `fit`                 | ✅ fits / ⚠️ tight / ❌ won't fit |
| `reason`              | The numeric comparison behind the verdict |

A second table shows the editorial **local-vs-cloud trade-off** across
privacy, cost, quality ceiling, latency, offline use, setup effort, and
operational risk. Everything is computed locally — opening this tab makes
**no network call**. The same data is exported to
[`docs/hardware_matrix.json`](https://github.com/raminmohammadi/Averix/blob/main/docs/hardware_matrix.json) and documented
in [`docs/hardware_matrix.md`](https://github.com/raminmohammadi/Averix/blob/main/docs/hardware_matrix.md).

---

## See also

- **[[Configuration and Tuning]]** — sampling, rate limits, embeddings.
- **[[Privacy and Security]]** — where credentials live and egress paths.
