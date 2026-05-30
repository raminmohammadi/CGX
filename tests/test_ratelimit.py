"""Tests for the client-side rate limiter and 429/5xx retry helper."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from cgx.answer.ratelimit import (
    RateLimiter,
    backoff_seconds,
    request_with_retry,
    should_retry,
)


def test_disabled_limiter_is_noop():
    rl = RateLimiter(rate=0)
    t0 = time.monotonic()
    for _ in range(100):
        assert rl.acquire() is True
    assert time.monotonic() - t0 < 0.05  # no real sleeping


def test_limiter_serialises_requests_when_rate_low():
    rl = RateLimiter(rate=10, capacity=2)  # 2 token burst, 10/s refill
    # Drain the burst.
    assert rl.acquire() is True
    assert rl.acquire() is True
    t0 = time.monotonic()
    assert rl.acquire() is True  # must wait ~100ms for one more token
    elapsed = time.monotonic() - t0
    # Be generous to avoid flake on busy CI: at least 50ms.
    assert elapsed >= 0.05, f"expected >=50ms wait, got {elapsed*1000:.1f}ms"


def test_should_retry_classification():
    assert should_retry(429)
    assert should_retry(500)
    assert should_retry(503)
    assert not should_retry(200)
    assert not should_retry(400)
    assert not should_retry(404)


def test_backoff_honours_retry_after_seconds():
    s = backoff_seconds(1, retry_after="2")
    assert 1.99 <= s <= 2.01


def test_backoff_is_jittered_but_bounded():
    # 100 samples; all should be within [base*0.5, cap*1.5].
    for _ in range(100):
        s = backoff_seconds(attempt=5, base=0.5, cap=10.0)
        assert 0.0 <= s <= 15.0


def _fake_resp(status: int, retry_after: str | None = None):
    headers = {}
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return SimpleNamespace(status_code=status, headers=headers)


def test_request_with_retry_succeeds_first_attempt():
    calls = []

    def f():
        calls.append(1)
        return _fake_resp(200)

    resp = request_with_retry(f, max_retries=3, sleep=lambda _s: None)
    assert resp.status_code == 200
    assert len(calls) == 1


def test_request_with_retry_retries_on_429():
    seq = iter([_fake_resp(429), _fake_resp(429), _fake_resp(200)])
    slept: list[float] = []

    def f():
        return next(seq)

    resp = request_with_retry(f, max_retries=3, sleep=lambda s: slept.append(s))
    assert resp.status_code == 200
    assert len(slept) == 2


def test_request_with_retry_retries_on_5xx_then_gives_up():
    seq = iter([_fake_resp(503), _fake_resp(503), _fake_resp(503), _fake_resp(503)])
    slept: list[float] = []

    def f():
        return next(seq)

    resp = request_with_retry(f, max_retries=2, sleep=lambda s: slept.append(s))
    assert resp.status_code == 503
    # 2 retries → 2 sleep calls.
    assert len(slept) == 2


def test_request_with_retry_respects_retry_after_header():
    seq = iter([_fake_resp(429, retry_after="0.05"), _fake_resp(200)])
    slept: list[float] = []

    def f():
        return next(seq)

    request_with_retry(f, max_retries=3, sleep=lambda s: slept.append(s))
    assert slept and abs(slept[0] - 0.05) < 1e-6


def test_request_with_retry_raises_after_persistent_exception():
    calls = []

    def f():
        calls.append(1)
        raise RuntimeError("network down")

    with pytest.raises(RuntimeError):
        request_with_retry(f, max_retries=2, sleep=lambda _s: None)
    # 1 initial + 2 retries = 3 attempts.
    assert len(calls) == 3


def test_provider_kwargs_passthrough_does_not_break_old_callers():
    """Verify the new rate_limit / max_retries kwargs are accepted but optional."""
    from cgx.answer.providers import OllamaProvider, OpenAICompatProvider
    a = OllamaProvider()
    b = OllamaProvider(rate_limit=5.0, max_retries=2)
    c = OpenAICompatProvider(model="x", base_url="http://x")
    d = OpenAICompatProvider(model="x", base_url="http://x", rate_limit=1, max_retries=0)
    # Internal handles exist and are correctly typed.
    assert a._limiter is None and a._max_retries == 0
    assert b._limiter is not None and b._max_retries == 2
    # OpenAI compat default keeps 3 retries.
    assert c._max_retries == 3 and c._limiter is None
    assert d._limiter is not None and d._max_retries == 0
