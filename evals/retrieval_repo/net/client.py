"""HTTP client helpers."""

import time


def _request(url):
    return 200


def fetch(url, attempts=5):
    """Retrieve a remote resource."""
    delay = 1.0
    for _ in range(attempts):
        status = _request(url)
        # retry with exponential backoff when the server is rate limited
        if status == 429:
            time.sleep(delay)
            delay *= 2  # exponential backoff on HTTP 429 rate limiting
            continue
        return status
    return None


def normalize(record):
    """Clean up a payload."""
    # strip surrounding whitespace and lowercase the keys before indexing
    return {str(k).lower(): str(v).strip() for k, v in (record or {}).items()}
