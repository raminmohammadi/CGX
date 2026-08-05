# Providers and Models

CGX is model-agnostic. Every generation path (Ask, Plan, Agent) talks to
an `LLMProvider` through a uniform `chat()` / `chat_stream()` interface,
so switching backends never changes the rest of the system. This page
covers the provider types and the hardware-aware model picker.

---

## Supported providers

| Provider type          | Kind string(s)              | Where it runs |
|------------------------|-----------------------------|---------------|
| **Ollama (local)**     | `ollama`                    | `http://localhost:11434` (loopback) |
| **OpenAI (cloud)**     | `openai`                    | `https://api.openai.com` |
| **OpenAI-compatible**  | `openai-compat`             | Any `/v1/chat/completions` endpoint (Groq, Together, DeepSeek, vLLM, …) |
| **Google Gemini**      | `gemini`                    | `generativelanguage.googleapis.com` |
| **Hugging Face**       | `huggingface`               | `router.huggingface.co` (OpenAI-compatible Inference Providers) |
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

## Hugging Face Inference (cloud)

Set **Provider Type → Hugging Face (Cloud)**, paste a Hugging Face token
(`hf_…`), and pick a model. HF's **Inference Providers** expose an
OpenAI-compatible router at `https://router.huggingface.co/v1`, so CGX
reuses `OpenAICompatProvider` verbatim — the host and endpoint path are
hardcoded and only the token varies. The token is also read from the
`HF_TOKEN` / `HUGGINGFACEHUB_API_TOKEN` environment variables when not
supplied inline.

The model dropdown is populated from the router's **public**
`/v1/models` list (no token required to browse), so you can see what's
available before pasting a key; the token is only needed for the actual
inference call. Ping validates the token with a 1-token completion when a
model is set, or falls back to the public model list otherwise.

### Browse Hugging Face → pull GGUFs locally

The **Settings → Browse Hugging Face** panel lists GGUF repositories from
the Hub (`huggingface.co/api/models?filter=gguf`) with live search, sort
(trending / downloads / likes / recently updated), download and like
counts, and detected quantization labels. **Pull** hands the tag to your
local Ollama daemon via the existing `hf.co/<repo>[:<quant>]` mechanism
and streams progress through the shared `PullProgress` bar — no HF token
required, since the download goes through Ollama.

Once the download completes, the model is **re-aliased to a clean local
name** so it isn't stored under the full `hf.co/…` web address:
`hf.co/ornith-ai/Ornith-1.0-9B-GGUF` becomes `Ornith-1.0-9B-GGUF` (with
the chosen quant as the tag, e.g. `…:q4_k_m`). The panel derives this
name from the repo id and passes it as the optional `local_name` on
`POST /api/ollama/pull`; the backend then runs Ollama's `POST /api/copy`
followed by `DELETE /api/delete` (instant — no re-download) and reports
the final name back over the progress stream ("Download complete — saved
as `<name>`"). The re-alias is best-effort: if it fails, the original tag
is kept and the download is never reported as an error.

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
