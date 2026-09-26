# JEV decision provider (local logprobs)

CGX's JEV decision layer (`cgx.answer.jev.decide`) asks a small, fixed question
and gets back a **typed** answer — an enum choice, a score, or a bool — plus a
probability. The probability is only trustworthy if it comes from the model's
actual token distribution (logprobs). **Ollama does not expose logprobs**, so
by default the probability is self-reported and uncalibrated (CGX treats it as a
ranking hint only, and every probability *threshold* stays off until calibrated).

To get **true constrained-choice logprobs** locally, point the decision layer at
an OpenAI-compatible server that returns them — llama.cpp's `llama-server`, vLLM,
or LM Studio. Ollama stays the default for normal chat/coding turns; only
decisions called with `prefer_logprobs=True` route to this endpoint.

## Run llama-server (macOS / Metal, same GGUF models Ollama uses)

```sh
# from a llama.cpp build
llama-server -m ~/.cache/llama/qwen2.5-3b-instruct-q4_k_m.gguf \
  --port 8080 --ctx-size 8192 --n-gpu-layers 999
```

`llama-server` implements `/v1/chat/completions` with `logprobs` / `top_logprobs`
and GBNF/JSON-schema constrained decoding — exactly what a typed decision needs.

## Point CGX at it

```sh
export CGX_JEV_BASE_URL="http://127.0.0.1:8080"   # required to enable
export CGX_JEV_MODEL="local"                       # optional label
# export CGX_JEV_API_KEY=...                        # only if your server needs one
```

With `CGX_JEV_BASE_URL` set, `cgx.answer.jev.decision_provider()` returns an
`OpenAICompatProvider(supports_logprobs=True)`, and `decide(prefer_logprobs=True)`
sends `logprobs: true` + `top_logprobs`, then reads the chosen decision token's
probability from the response. Unset the variable to fall back to the passed
provider (Ollama) with a self-reported probability.

## Calibration (still required before gating)

A raw logprob is a real probability over the allowed choices but is **not
guaranteed calibrated** — a model can be confidently wrong. Before any threshold
is turned on, mine a reliability curve from `emit_trace` history vs actual
outcomes (temperature scaling / isotonic) so the threshold means what it says.
That calibration step is Phase 4 of the JEV plan.
