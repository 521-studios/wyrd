"""Retry-with-backoff for paid-provider HTTP calls.

Anthropic and Gemini both apply per-org RPM caps; a parallel mining loop
(``mine-llm --concurrency 8``) can transiently exceed them and get HTTP
429 responses. Providers also shed load under their OWN pressure —
Anthropic returns 529 (``overloaded_error``) and Gemini 503 (model
overloaded). All three are transient blips, not genuine model decisions;
without retry they bubble up as transport errors and the operator sees a
chunk of declines that are really rate-limit / overload blips.

This helper wraps ``urllib.request.urlopen`` with exponential backoff +
jitter on those transient statuses (:data:`_RETRYABLE_STATUS`). Other HTTP
errors (500, 4xx auth/validation, …) and socket-level errors propagate
immediately so the existing error-handling paths in each provider's
``chat_json`` keep working unchanged.

Local Ollama (``extractors/llm.py``) doesn't rate-limit, so it doesn't
use this helper.
"""

from __future__ import annotations

import logging
import random
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_MAX_TOTAL_WAIT = 60.0

# Transient provider statuses worth retrying with backoff: 429 rate-limit
# (Anthropic + Gemini), plus the overload codes — Anthropic 529
# (``overloaded_error``) and Gemini 503 (model overloaded). All are blips that a
# retry recovers; a 500 or a 4xx auth/validation error is NOT transient and
# propagates immediately. (This helper is provider-scoped — only the Anthropic /
# Gemini extractors call it — so a 503 here is provider overload, not a generic
# proxy/gateway 503.)
_RETRYABLE_STATUS = frozenset({429, 503, 529})


def open_with_429_retry(
    req: urllib.request.Request,
    *,
    timeout: float,
    provider_label: str,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    max_total_wait: float = DEFAULT_MAX_TOTAL_WAIT,
) -> bytes:
    """Open ``req`` with exponential backoff on a transient provider status
    (:data:`_RETRYABLE_STATUS` — 429 rate-limit, 503/529 overload).

    Returns the raw response body (caller decodes / parses).

    Backoff schedule per attempt: 2^(attempt-1) seconds × jitter in
    [0.75, 1.25]. After ``max_attempts`` total tries OR cumulative wait
    exceeding ``max_total_wait``, the most recent error propagates so the
    provider's existing error path can convert it to a transport-error
    result.

    A non-retryable HTTPError (500, 4xx, …) and socket-level errors
    (TimeoutError, URLError) propagate immediately — retry doesn't help.
    """
    waited = 0.0
    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code not in _RETRYABLE_STATUS or attempt >= max_attempts:
                raise
            base = 2.0 ** (attempt - 1)
            sleep_s = base * random.uniform(0.75, 1.25)
            if waited + sleep_s > max_total_wait:
                raise
            # HTTPError doubles as a response object; close it before
            # sleeping so the underlying socket is freed promptly. Without
            # this, parallel mining at --concurrency 8+ can accumulate
            # half-open sockets across retries and hit the FD limit.
            e.close()
            logger.warning(
                "%s %d (transient); retry %d/%d in %.1fs",
                provider_label,
                e.code,
                attempt,
                max_attempts,
                sleep_s,
            )
            time.sleep(sleep_s)
            waited += sleep_s
    # Unreachable: the for-loop either returns from urlopen or raises.
    raise RuntimeError("provider_retry: exhausted attempts without raising")
