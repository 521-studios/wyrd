"""Retry-with-backoff for paid-provider HTTP calls.

Anthropic and Gemini both apply per-org RPM caps; a parallel mining loop
(``mine-llm --concurrency 8``) can transiently exceed them and get HTTP
429 responses. Without retry these bubble up as transport errors and the
operator sees a chunk of declines that are really rate-limit blips, not
genuine model decisions.

This helper wraps ``urllib.request.urlopen`` with exponential backoff +
jitter on 429 responses only. Non-429 HTTP errors and socket-level
errors propagate immediately so the existing error-handling paths in
each provider's ``chat_json`` keep working unchanged.

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


def open_with_429_retry(
    req: urllib.request.Request,
    *,
    timeout: float,
    provider_label: str,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    max_total_wait: float = DEFAULT_MAX_TOTAL_WAIT,
) -> bytes:
    """Open ``req`` with exponential backoff on HTTP 429.

    Returns the raw response body (caller decodes / parses).

    Backoff schedule per attempt: 2^(attempt-1) seconds × jitter in
    [0.75, 1.25]. After ``max_attempts`` total tries OR cumulative wait
    exceeding ``max_total_wait``, the most recent 429 propagates so the
    provider's existing error path can convert it to a transport-error
    result.

    Non-429 HTTPError and socket-level errors (TimeoutError, URLError)
    propagate immediately — retry doesn't help for those.
    """
    waited = 0.0
    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt >= max_attempts:
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
                "%s 429 rate-limited; retry %d/%d in %.1fs",
                provider_label,
                attempt,
                max_attempts,
                sleep_s,
            )
            time.sleep(sleep_s)
            waited += sleep_s
    # Unreachable: the for-loop either returns from urlopen or raises.
    raise RuntimeError("provider_retry: exhausted attempts without raising")
