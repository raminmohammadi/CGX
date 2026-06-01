"""Client-side rate limiting + retry helpers for LLM providers.

Design goals:

* **Pure-Python, no extra deps.** Uses ``threading.Lock`` and ``time``.
* **Per-provider isolation.** A token bucket lives on each provider
  instance so two profiles can have independent budgets.
* **Predictable backoff.** Exponential backoff with jitter on HTTP 429
  / 5xx, capped at ``max_retries`` attempts. Honours a ``Retry-After``
  header when the server provides one.
* **Side-effect-free when disabled.** A ``RateLimiter(rate=0)`` is a
  no-op; the default provider construction also opts out so existing
  behaviour is preserved unless callers explicitly enable it.

Public API:

* :class:`RateLimiter` — token bucket; call ``acquire()`` before a request.
* :func:`should_retry(response)` — returns True for 429 + 5xx.
* :func:`backoff_seconds(attempt, response=None)` — derives sleep.
* :func:`request_with_retry(func, *, limiter, max_retries)` — wraps a
  callable that returns a ``requests.Response``.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple thread-safe token-bucket limiter.

    Parameters
    ----------
    rate
        Tokens added per second. ``rate <= 0`` disables limiting (the
        limiter becomes a no-op).
    capacity
        Maximum tokens that can accumulate (i.e. burst size). Defaults
        to ``max(1, ceil(rate))``.
    """

    def __init__(self, rate: float, capacity: Optional[float] = None) -> None:
        self.rate = float(rate)
        if capacity is None:
            capacity = max(1.0, self.rate)
        self.capacity = float(capacity)
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0, timeout: Optional[float] = None) -> bool:
        """Block until ``tokens`` are available. Returns False on timeout.

        When ``rate <= 0`` this is a no-op that returns ``True`` immediately.
        """
        if self.rate <= 0:
            return True
        deadline = None if timeout is None else (time.monotonic() + timeout)
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last
                self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
                self._last = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True
                # Sleep just long enough to accumulate the deficit.
                deficit = tokens - self._tokens
                wait = deficit / self.rate
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                wait = min(wait, remaining)
            time.sleep(max(wait, 0.001))


def should_retry(status_code: int) -> bool:
    """Return True for HTTP statuses that warrant a retry."""
    return status_code == 429 or 500 <= status_code < 600


def backoff_seconds(
    attempt: int,
    *,
    base: float = 0.5,
    cap: float = 30.0,
    retry_after: Optional[str] = None,
) -> float:
    """Compute sleep duration for a given retry ``attempt`` (1-based).

    Honours the ``Retry-After`` header if it's a plain integer of seconds.
    Otherwise uses exponential backoff with a small uniform jitter:

        sleep = min(cap, base * 2**(attempt-1)) * uniform(0.5, 1.5)
    """
    if retry_after:
        try:
            return max(0.0, min(cap, float(retry_after)))
        except Exception:
            pass
    delay = min(cap, base * (2 ** max(0, attempt - 1)))
    return delay * random.uniform(0.5, 1.5)


def request_with_retry(
    func: Callable[[], Any],
    *,
    limiter: Optional[RateLimiter] = None,
    max_retries: int = 3,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Invoke ``func()`` with rate-limiting and 429/5xx retry.

    ``func`` must return an object with a ``status_code`` attribute and a
    ``headers`` mapping (i.e. a ``requests.Response``). Network exceptions
    propagate after the final attempt.
    """
    last_exc: Optional[BaseException] = None
    for attempt in range(1, max_retries + 2):  # +1 so max_retries=0 → 1 try
        if limiter is not None:
            limiter.acquire()
        try:
            resp = func()
        except Exception as e:
            last_exc = e
            if attempt > max_retries:
                logger.warning("ratelimit: giving up after %d attempt(s); last exception %s: %s",
                               attempt, type(e).__name__, e)
                raise
            delay = backoff_seconds(attempt)
            logger.info("ratelimit: attempt %d raised %s: %s — retrying in %.2fs",
                        attempt, type(e).__name__, e, delay)
            sleep(delay)
            continue
        sc = getattr(resp, "status_code", 200)
        if not should_retry(sc) or attempt > max_retries:
            return resp
        retry_after = None
        try:
            retry_after = resp.headers.get("Retry-After")  # type: ignore[attr-defined]
        except Exception:
            retry_after = None
        delay = backoff_seconds(attempt, retry_after=retry_after)
        logger.info("ratelimit: attempt %d returned status=%d — retrying in %.2fs "
                    "(Retry-After=%r)", attempt, sc, delay, retry_after)
        sleep(delay)
    if last_exc is not None:
        raise last_exc
    return resp  # type: ignore[possibly-unbound]
