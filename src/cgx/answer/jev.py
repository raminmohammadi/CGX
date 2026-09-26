"""JEV -- the typed-decision layer that sits *beside* the model.

The idea (after the "JEV Engineering for Coding Agents" note): a coding agent's
leverage is not in its loop but in *what the loop feeds the model each turn*.
Many of those per-turn choices -- "is this repo relevant?", "what is the
minimal fix?", later "how visible should this chunk be?" -- are really the same
shape: hand the model a fixed QUESTION plus the current STATE and get back a
*typed* answer (an enum choice, a score, or a bool) that the harness can
VALIDATE and BRANCH on deterministically, instead of parsing prose.

:func:`decide` is that one entry point. It consolidates the
force-JSON -> extract -> validate -> (one bounded re-ask) -> clamp logic that
was copy-pasted across :mod:`cgx.session.tasks.diagnose` and
:mod:`cgx.session.tasks.swarm_assess`, and builds on the infrastructure CGX
already has:

* the provider ``chat(force_json=True, json_schema=...)`` contract on
  :class:`cgx.answer.providers.LLMProvider` (constrained decoding, with a
  graceful ``json_object`` -> plain-text fallback per backend), and
* :func:`cgx.answer.schemas.validate_json_schema`, which re-checks the parsed
  reply at this boundary so a backend that silently ignored the schema still
  yields an actionable violation list.

**Local-first honesty.** The ``probability`` a local model reports is
*uncalibrated*, and Ollama exposes no logprobs, so callers MUST treat it as a
ranking hint, never a calibrated gate, until a reliability curve is mined from
trace history. When a logprob-capable provider is wired (llama.cpp
``llama-server``), :func:`decide` prefers the true constrained-choice logprob
of the decision token over the self-reported field -- but the gating discipline
is unchanged.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

from cgx.answer.schemas import validate_json_schema

logger = logging.getLogger(__name__)

Messages = List[Dict[str, str]]


@dataclass
class TypedDecision:
    """The validated outcome of one JEV question.

    ``value`` is the thing the harness branches on -- the ``decision_key``
    field when one is given (e.g. the ``relevant`` bool), otherwise the whole
    parsed object (e.g. a DIAGNOSE turn, which is either a tool call or a
    verdict). ``probability`` is the model-reported confidence in ``[0, 1]`` or
    ``None`` -- UNCALIBRATED on local models; a ranking hint, not a gate.
    ``raw`` is the full parsed object; ``violations`` lists any schema breaches
    that survived the corrective re-ask; ``ok`` is ``False`` when no conforming
    object could be obtained (provider crash / unparseable), so callers can
    fall back to their safe default exactly as before.
    """

    value: Any
    probability: Optional[float]
    raw: Dict[str, Any]
    violations: List[str] = field(default_factory=list)
    ok: bool = True


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Balanced-brace extraction, reusing the engine's tolerant parser.

    Imported lazily: :mod:`cgx.answer.engine` is heavy and imports this package
    transitively, so a top-level import would risk a cycle. The parser itself
    is pure.
    """
    try:
        from cgx.answer.engine import _extract_json_object
    except Exception:  # pragma: no cover - defensive
        return None
    try:
        parsed = _extract_json_object(text or "")
    except Exception:  # pragma: no cover - defensive
        return None
    return parsed if isinstance(parsed, dict) else None


def _clamp01(value: Any) -> Optional[float]:
    """Coerce ``value`` to a float in ``[0, 1]``; ``None`` when not numeric."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(conf) or math.isinf(conf):
        return None
    return max(0.0, min(1.0, conf))


def _logprob_probability(
    resp: Dict[str, Any], value: Any,
) -> Optional[float]:
    """Recover a calibratable probability from provider-returned logprobs.

    Only meaningful for enum/bool decisions where ``value`` is emitted as a
    short token. When a logprob-capable provider (llama.cpp ``llama-server``)
    returns ``logprobs`` for the completion, we look for the top-token entry
    whose text matches ``value`` and convert its logprob to a probability. This
    is the true constrained-choice mass -- still to be calibrated, but far
    better than a self-reported field. Returns ``None`` when logprobs are
    absent or don't cover the decision token (the common case today, since
    Ollama exposes none).
    """
    lp = resp.get("logprobs") if isinstance(resp, dict) else None
    if not lp or value is None:
        return None
    target = str(value).strip().lower()
    if not target:
        return None
    # Accept both the OpenAI ``content: [{token, logprob, top_logprobs:[...]}]``
    # shape and a flat ``[{token, logprob}]`` list.
    entries: Sequence[Any]
    if isinstance(lp, dict):
        entries = lp.get("content") or []
    elif isinstance(lp, list):
        entries = lp
    else:
        return None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        candidates = [entry]
        top = entry.get("top_logprobs")
        if isinstance(top, list):
            candidates = top + [entry]
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            tok = str(cand.get("token") or "").strip().lower()
            if tok and (tok == target or target.startswith(tok) and len(tok) >= 3):
                try:
                    return max(0.0, min(1.0, math.exp(float(cand["logprob"]))))
                except (KeyError, TypeError, ValueError, OverflowError):
                    return None
    return None


def _as_messages(
    prompt: Union[str, Messages], system: Optional[str],
) -> Messages:
    if isinstance(prompt, str):
        msgs: Messages = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return msgs
    # Already a message list; prepend a system turn only when asked and absent.
    if system and not any(m.get("role") == "system" for m in prompt):
        return [{"role": "system", "content": system}, *prompt]
    return list(prompt)


def decide(
    provider: Any,
    prompt: Union[str, Messages],
    schema: Dict[str, Any],
    *,
    decision_key: Optional[str] = None,
    prob_key: str = "confidence",
    system: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = 800,
    prefer_logprobs: bool = False,
) -> TypedDecision:
    """Ask one typed question and return a validated :class:`TypedDecision`.

    ``prompt`` is either a ready message list or a user string (paired with the
    optional ``system``). ``schema`` is a canonical JSON-Schema (see
    :mod:`cgx.answer.schemas`); it is handed to the provider for constrained
    decoding AND re-checked here. On a violation, exactly one corrective re-ask
    is issued with the human-readable violation list appended, mirroring the
    executors' existing bounded-re-ask pattern.

    A provider crash, an empty reply, or an unparseable reply after the re-ask
    yields ``ok=False`` with ``raw={}`` -- the caller keeps its safe default.
    Never raises on a provider/parse failure.
    """
    messages = _as_messages(prompt, system)

    def _call(msgs: Messages) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = dict(
            messages=msgs, temperature=temperature,
            force_json=True, json_schema=schema,
        )
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if prefer_logprobs and getattr(provider, "supports_logprobs", False):
            # Opt-in, provider-honored hint; ignored by providers that don't
            # accept it (chat signatures take **kwargs downstream).
            kwargs["logprobs"] = True
        try:
            resp = provider.chat(**kwargs)
        except TypeError:
            # A provider whose chat() doesn't accept logprobs kwargs: retry
            # without the opt-in extras rather than fail the decision.
            kwargs.pop("logprobs", None)
            try:
                resp = provider.chat(**kwargs)
            except Exception:
                logger.exception("jev.decide: provider.chat crashed")
                return {}
        except Exception:
            logger.exception("jev.decide: provider.chat crashed")
            return {}
        return resp if isinstance(resp, dict) else {}

    resp = _call(messages)
    parsed = _extract_json(str(resp.get("content", "")))
    violations = (validate_json_schema(parsed, schema)
                  if isinstance(parsed, dict) else ["no JSON object in reply"])

    if violations and isinstance(parsed, dict):
        # One bounded corrective re-ask: show the model its own reply and the
        # exact violations, ask for a corrected object only.
        retry = [
            *messages,
            {"role": "assistant", "content": str(resp.get("content", ""))[:2000]},
            {"role": "user",
             "content": ("Your JSON did not conform:\n- "
                         + "\n- ".join(violations[:12])
                         + "\nReturn ONLY a corrected JSON object.")},
        ]
        resp2 = _call(retry)
        parsed2 = _extract_json(str(resp2.get("content", "")))
        if isinstance(parsed2, dict):
            v2 = validate_json_schema(parsed2, schema)
            if len(v2) <= len(violations):
                parsed, resp, violations = parsed2, resp2, v2

    if not isinstance(parsed, dict) or parsed == {}:
        return TypedDecision(value=None, probability=None, raw={},
                             violations=violations or ["empty reply"], ok=False)

    value = parsed.get(decision_key) if decision_key is not None else parsed
    probability = _clamp01(parsed.get(prob_key))
    lp = _logprob_probability(resp, value if decision_key is not None else None)
    if lp is not None:
        probability = lp
    return TypedDecision(value=value, probability=probability, raw=parsed,
                         violations=violations, ok=True)
