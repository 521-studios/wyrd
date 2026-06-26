"""Shared LLM judge-with-retry loop for the cli/lexicon mining/audit commands.

Five commands (audit-toponym-reflexes, audit-cluster-reflexes, audit-descent,
audit-merges, mine-spurious-breakdowns) ran a byte-identical retry loop that
differed only in the injected ``build_prompt`` / ``parse_verdict`` and the
candidate's identifier for the skip message. Centralizing the loop kills the
drift the duplication had already produced: the ``last_err`` reset below (so the
skip diagnostic reflects the CURRENT attempt, not a stale prior exception) had
been applied to only two of the five copies (wyrd-l5n7).

``fold-variants`` keeps its own variant — it returns a ``(verdict, error)`` tuple
for the caller to handle rather than echoing + returning None — so it is
deliberately NOT routed through here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import click


def judge_with_retry(
    client: Any,
    c: Any,
    *,
    build_prompt: Callable[[Any], tuple[str, str]],
    parse_verdict: Callable[[Any], Any],
    ref: Any,
) -> Any:
    """Judge one candidate, one retry on transport/parse failure; ``None`` if
    unusable (caller skips WITHOUT recording, so a re-run retries). Echoes the
    last failure to stderr so a persistent Ollama outage is diagnosable rather
    than presenting as a silently-climbing skip count.

    ``build_prompt(c) -> (system, user)`` and ``parse_verdict(raw) -> verdict | None``
    are the per-command judge pieces; ``ref`` is the candidate identifier shown
    in the skip message.
    """
    system, user = build_prompt(c)
    last_err = "unparseable verdict"
    for _attempt in range(2):
        try:
            verdict = parse_verdict(client.chat_json(system, user, {}))
            if verdict is not None:
                return verdict
            # reflect THIS attempt, not a stale prior exception
            last_err = "unparseable verdict"
        except Exception as exc:  # transient LLM/transport noise: skip, don't crash the batch
            last_err = str(exc) or exc.__class__.__name__
    click.echo(f"  skip (judge failed for {ref}): {last_err}", err=True)
    return None
