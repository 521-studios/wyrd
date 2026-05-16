"""LLM-driven toponym-mention extraction — wyrd-x82p Phase 2.

Companion to ``toponym_reverse_search`` (Phase 1 pattern-based).
Walks a scholar source body via an LLM, extracts every place-name
mention with optional date_year + region_hint + surrounding context.
Captures the cases Phase 1 missed:

* **Homonym disambiguation**: Phase 1 excludes ambiguous forms
  (Newton across counties) from its lookup. Phase 2's LLM sees the
  surrounding prose ("Newton, near Durham") and can resolve to a
  specific toponym.
* **Undated mentions**: Phase 1 only extracts (form, YEAR) patterns.
  Phase 2 captures bare mentions like "compare Suttone" that lack
  a date but still anchor the form in this source's prose.
* **New-toponym discovery**: forms scholar prose mentions that the
  current toponym table doesn't carry (Suttone (Domesday) when
  Suttone isn't a row). Phase 2 surfaces these as candidates; the
  decision of whether to auto-create new toponym rows is gated
  behind operator review (Phase 2b).

Anti-hallucination: every extracted form is validated against the
chunk text. The check normalizes both form and chunk to a single-
space-collapsed view before matching, so a form the LLM emits as
"Newcastle upon Tyne" still matches a soft-wrapped "Newcastle
upon\\nTyne" in the source. The match also requires word-boundary
alignment on both sides — substring-only matching would silently
admit a partial like "on Ty" because "...up**on Ty**ne..." contains
those letters. Forms that fail either check are counted (not
silently dropped) so operators can spot a systemic prompt/encoding
mismatch.

Provider choice: the module accepts any client with the
``chat_json(system, user, schema=None) -> dict`` interface — same
contract as :class:`AnthropicClient` / Ollama / Gemini wrappers in
``llm_extractor.py``. The tiered-extraction pattern from the
project's wyrd-l0r infrastructure (Qwen bulk → Anthropic on the
residual) is implemented in the CLI orchestrator, not here.

Chunking limit: ``chunk_source_body`` does NOT overlap chunks; a
mention whose year/region hint falls on the OTHER side of a
paragraph boundary will be split, and the LLM will see only one
side of the pair. Paragraph boundaries are the strongest signal we
have for keeping each chunk's view coherent — sentence-overlap
chunking is a documented follow-up (DECISIONS.md if pursued).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field

# Year-citation range — same bounds as mine-attestations
# (_ATTESTED_YEAR_MIN_LOOKUP / _MAX_LOOKUP). 700-1700 covers
# post-Roman through pre-modern attestations. The LLM is
# instructed to clamp to this range; out-of-range years get
# treated as null by the validator.
_DATE_YEAR_MIN = 700
_DATE_YEAR_MAX = 1700


# Single-source-of-truth whitespace collapser for the
# anti-hallucination check. Keeping this at module scope avoids
# re-compiling the pattern on every mention.
_WHITESPACE_RUN = re.compile(r"\s+")


def _collapse_whitespace(text: str) -> str:
    """Collapse every whitespace run (spaces, tabs, newlines,
    soft-wrap hyphen + newline) to a single space. Used to make the
    form-in-chunk check tolerant of OCR/typesetter line breaks that
    the LLM correctly emits as bare-form mentions."""
    return _WHITESPACE_RUN.sub(" ", text).strip()


@dataclass(frozen=True)
class ToponymMention:
    """One place-name mention extracted from a scholar source body.

    * ``form`` — verbatim place name as it appears in the prose
      (preserves capitalization, diacritics, internal punctuation).
      Validated to appear in the source chunk before construction;
      the assembler drops mentions whose form doesn't substring-
      match the whitespace-normalized chunk (anti-hallucination
      guard).
    * ``date_year`` — a year in [700, 1700] from immediately
      surrounding context, if any. Out-of-range or unparseable
      values resolve to None and bump ``years_clamped`` on the
      run's report so operators can distinguish "LLM declined to
      cite a year" from "LLM cited but I rejected it."
    * ``region_hint`` — county / country name the LLM saw in the
      surrounding prose (Northumberland, Durham, England, etc.).
      Used downstream to disambiguate homonyms. None when the
      surrounding context doesn't mention a region.
    * ``context`` — ~100-char surrounding snippet (line breaks
      stripped) for operator verification. Always non-empty when
      the mention is admitted.
    """

    form: str
    date_year: int | None
    region_hint: str | None
    context: str


SYSTEM_PROMPT = """\
You are an extraction agent for British place-name etymology research.

Your task: given a chunk of scholarly prose from a published place-
name study (Mawer, Ekwall, Joyce, Skeat, Watson, etc.), extract
every PLACE NAME mentioned, with optional date and region context.

What counts as a place name:
* Cities, towns, villages, parishes, hundreds, manors
* Rivers, streams, lakes, hills, woods, fields with proper names
* Counties, regions, kingdoms (Northumberland, Wessex, etc.)
* Historical place forms (Cestretone, Eboracum, etc.)

What does NOT count (skip these):
* Personal names (Aelfric, William the Conqueror, Saint Cuthbert)
* Source / scholar names (Domesday Book, D.B., Inquisitio post mortem,
  Pipe Rolls, P., Ipm., LP, F.A.)
* Common-noun place words on their own (church, valley, hill) — only
  proper-noun place names

For each place name, identify:
* `form` — VERBATIM text as it appears in the prose. Preserve
  capitalization, diacritics, hyphens. Do NOT normalize.
* `date_year` — a year in [700, 1700] from text within ~50 chars
  before or after the form. Set null if no year is nearby OR the
  year is outside the range.
* `region_hint` — county/country/region name from surrounding
  prose (within ~200 chars), or null. Used to disambiguate
  homonyms.
* `context` — ~80-120 chars of immediate surrounding prose (line
  breaks stripped) — for operator verification.

Return STRICT JSON in the shape below. NO commentary. NO markdown
fences. NO explanation. JUST the JSON object.

{"mentions": [{"form": "...", "date_year": YYYY_or_null, "region_hint": "..." or null, "context": "..."}, ...]}

If the chunk has no place-name mentions, return {"mentions": []}.
"""

USER_TEMPLATE = """\
Extract all place name mentions from this scholar prose chunk:

```
{chunk}
```

Return only the JSON object. No other text.
"""

# JSON schema for providers that support schema-constrained output
# (Ollama, Gemini). Anthropic doesn't support this; for Anthropic the
# SYSTEM_PROMPT's strict-JSON instruction + post-hoc parse handles it.
RESPONSE_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "mentions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "form": {"type": "string"},
                    "date_year": {"type": ["integer", "null"]},
                    "region_hint": {"type": ["string", "null"]},
                    "context": {"type": "string"},
                },
                "required": ["form", "context"],
            },
        }
    },
    "required": ["mentions"],
}


@dataclass
class ValidationCounters:
    """Per-call counters for what ``_validated_mentions`` rejected.

    Used by the orchestrator to roll up across all chunks, so
    operators can see the failure-mode breakdown (vs. an opaque
    "0 mentions admitted" result is a load-bearing UX gap).
    """

    admitted: int = 0
    hallucinations_dropped: int = 0
    years_clamped: int = 0  # any year value the LLM emitted that we coerced to None


# Strict digit-only match for year strings — Python's int() accepts
# underscore separators ("1_086"), leading "+/-", and leading zeros
# ("01086"), all of which would silently admit non-canonical years
# from the LLM. Module-level for compile-once.
_YEAR_STRING_PATTERN = re.compile(r"\d{3,4}")


def _coerce_year(raw: object) -> tuple[int | None, bool]:
    """Return ``(year_or_None, was_clamped)``. ``was_clamped`` is True
    iff the LLM emitted a non-null value we rejected — so the caller
    can distinguish "LLM declined" (raw is None, was_clamped False)
    from "LLM emitted nonsense" (was_clamped True). Strings that
    parse cleanly (``"1086"``) are accepted; strings that don't
    (``"ten eighty-six"``, ``"1066 AD"``) become None+clamped."""
    if raw is None:
        return None, False
    if isinstance(raw, bool):
        # Python's bool is a subclass of int; treat True/False as nonsense
        # rather than year 0/1.
        return None, True
    if isinstance(raw, int):
        if _DATE_YEAR_MIN <= raw <= _DATE_YEAR_MAX:
            return raw, False
        return None, True
    if isinstance(raw, str):
        # Strict digit-only match — int() in Python 3 accepts "1_086",
        # "+1086", "01086" which are silent-semantic-drift admissions
        # (the LLM would be emitting non-canonical years). Restrict to
        # 3-4 ASCII digits.
        if not _YEAR_STRING_PATTERN.fullmatch(raw.strip()):
            return None, True
        n = int(raw.strip())
        if _DATE_YEAR_MIN <= n <= _DATE_YEAR_MAX:
            return n, False
        return None, True
    return None, True


def _form_in_chunk(form: str, chunk_collapsed: str) -> bool:
    """Whitespace-tolerant word-boundary match. ``chunk_collapsed``
    is the once-per-chunk pre-collapsed view; this function collapses
    the candidate form the same way and then searches with regex
    ``\\b...\\b`` so a partial like ``"on Ty"`` does NOT match
    ``"...upon Tyne..."``.

    Catches the soft-wrap case (chunk has ``"Newcastle upon\\nTyne"``
    but the LLM correctly emits ``"Newcastle upon Tyne"``) AND the
    substring-fragment case (LLM emits a partial that happens to be a
    letter run inside a longer real form). Both were silent-data-loss
    hazards: substring-only matching rejected soft-wraps; whitespace-
    normalized substring matching accepted fragments.
    """
    form_collapsed = _collapse_whitespace(form)
    if not form_collapsed:
        return False
    # Lookbehind/lookahead instead of \b — \b requires a word-class
    # transition, which fails for forms whose first/last character is
    # already a non-word char (an LLM-emitted "F." for an abbreviation,
    # "St." for "Saint", or "Stoke-" with a trailing hyphen). The
    # lookaround flavor only requires the IMMEDIATE neighbor to be
    # non-word, regardless of what the form's own edge character is.
    pattern = r"(?<!\w)" + re.escape(form_collapsed) + r"(?!\w)"
    return re.search(pattern, chunk_collapsed) is not None


def _validated_mentions(
    raw: list[dict],
    chunk: str,
    counters: ValidationCounters | None = None,
) -> list[ToponymMention]:
    """Convert LLM-emitted dicts to ``ToponymMention`` instances.

    Drops mentions whose form doesn't match the whitespace-normalized
    chunk (anti-hallucination). When ``counters`` is supplied, records
    the breakdown of rejections (hallucinations, year clamps) — used
    by the orchestrator to surface failure modes in the run report.

    Form-in-chunk is the bare-minimum guard. Stronger guards (form
    starts with capital, isn't a known blacklist word like Domesday,
    etc.) are downstream — the operator-review tool / the
    form→toponym resolver can apply those.
    """
    if counters is None:
        counters = ValidationCounters()
    chunk_collapsed = _collapse_whitespace(chunk)
    out: list[ToponymMention] = []
    for m in raw:
        if not isinstance(m, dict):
            continue
        form = (m.get("form") or "").strip()
        if not form:
            continue
        if not _form_in_chunk(form, chunk_collapsed):
            counters.hallucinations_dropped += 1
            continue
        date_year, was_clamped = _coerce_year(m.get("date_year"))
        if was_clamped:
            counters.years_clamped += 1
        region_hint = m.get("region_hint")
        if isinstance(region_hint, str):
            region_hint = region_hint.strip() or None
        else:
            region_hint = None
        context = _WHITESPACE_RUN.sub(" ", (m.get("context") or "")).strip()
        if not context:
            # Synthesize a context window from the chunk if the LLM
            # left it empty — gives operators something to inspect.
            # The substring check above guarantees the collapsed form
            # appears in the collapsed chunk, but the raw chunk may
            # not contain `form` verbatim (soft-wrap case). Search the
            # raw chunk first and fall back to the collapsed view.
            idx = chunk.find(form)
            if idx >= 0:
                start = max(0, idx - 40)
                end = min(len(chunk), idx + len(form) + 40)
                context = _WHITESPACE_RUN.sub(" ", chunk[start:end]).strip()
            else:
                # Fall back: locate the form in the collapsed chunk and
                # use that window (still useful for operator review).
                collapsed_idx = chunk_collapsed.find(_collapse_whitespace(form))
                if collapsed_idx >= 0:
                    start = max(0, collapsed_idx - 40)
                    end = min(
                        len(chunk_collapsed),
                        collapsed_idx + len(_collapse_whitespace(form)) + 40,
                    )
                    context = chunk_collapsed[start:end]
        out.append(
            ToponymMention(
                form=form,
                date_year=date_year,
                region_hint=region_hint,
                context=context,
            )
        )
        counters.admitted += 1
    return out


def extract_toponym_mentions_from_chunk(
    client,
    chunk: str,
    counters: ValidationCounters | None = None,
) -> list[ToponymMention]:
    """Run one LLM call against ``chunk`` and return validated
    mentions. ``client`` must have a ``chat_json(system, user,
    schema=None) -> dict`` method — same contract as
    :class:`AnthropicClient`, :class:`OllamaClient`,
    :class:`GeminiClient` in the existing extractor modules.

    Network / JSON-parse errors propagate as ``RuntimeError`` — the
    caller (per-source orchestrator) decides whether to skip the
    chunk and continue or abort the run. Structurally-malformed
    responses (non-dict envelope, non-list ``mentions``) also raise
    ``RuntimeError`` so the orchestrator can count them as failed
    chunks rather than silently producing zero mentions.
    """
    user = USER_TEMPLATE.format(chunk=chunk)
    # Positional schema arg — the three provider clients diverge on
    # the schema kwarg name (AnthropicClient: `schema`, GeminiClient:
    # `response_schema`, OllamaClient: `schema`). Using positional
    # avoids a TypeError under --provider gemini.
    response = client.chat_json(SYSTEM_PROMPT, user, RESPONSE_SCHEMA)
    if not isinstance(response, dict):
        raise RuntimeError(f"LLM returned non-dict envelope: {type(response).__name__}")
    raw = response.get("mentions")
    if raw is None:
        raise RuntimeError("LLM response missing 'mentions' key")
    if not isinstance(raw, list):
        raise RuntimeError(f"LLM response 'mentions' was {type(raw).__name__}, expected list")
    return _validated_mentions(raw, chunk, counters=counters)


# When a single paragraph exceeds this multiple of the target chunk
# size, ``mine_toponym_mentions`` warns to stderr — without the
# warning, a 30K-char single paragraph (kept whole by snap-to-
# paragraph) + Anthropic's 8192-token output cap silently truncates
# the JSON response, surfacing only as chunks_failed with no
# diagnostic hint that the cause is chunk size.
_OVERSIZED_PARAGRAPH_MULTIPLIER = 1.5


def chunk_source_body(body: str, *, target_chunk_size: int = 20000) -> list[str]:
    """Split a source body into chunks of roughly ``target_chunk_size``
    characters, snapping to paragraph boundaries (blank lines).
    Returns the list of non-empty chunks in order.

    Snap-to-paragraph keeps the LLM's view of each chunk coherent —
    splitting mid-paragraph would orphan year/region context that's
    part of the same scholar entry.

    For a 600KB source body and ``target_chunk_size=20000``, this
    yields ~30 chunks. Anthropic at ~3s per chunk = ~90s per source.

    A single paragraph larger than ``target_chunk_size`` is kept as
    one chunk (the snap-to-paragraph invariant). If that single
    paragraph is very large, the LLM's output cap may truncate the
    response — see ``mine_toponym_mentions``'s oversize warning.
    """
    if not body.strip():
        return []
    # Regex split tolerates blank lines with horizontal whitespace
    # (spaces/tabs) between the newlines — OCR-derived scholar prose
    # often has "\n  \n" where a tight \n\n would miss the boundary.
    # Restricted to [ \t] (not \s) so the middle of the blank line
    # can't itself contain another \n that a greedy \s* would consume,
    # falsely merging real content into a paragraph separator.
    paragraphs = re.split(r"\n[ \t]*\n", body)
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > target_chunk_size:
            chunks.append(current.strip())
            current = para
        else:
            if current:
                current += "\n\n" + para
            else:
                current = para
    if current.strip():
        chunks.append(current.strip())
    return chunks


@dataclass(frozen=True)
class FailedChunk:
    """Per-chunk failure record. Carried on the run report so operators
    see *which* chunk failed and *what kind* of error (rate-limit,
    JSON-truncation, schema-mismatch) — vs. an opaque count."""

    index: int
    error: str


# Memory ceiling for ``failed_chunks`` on a single source's report.
# A pathological rate-limit storm could otherwise accumulate thousands
# of records, blowing memory and flooding stderr.
#
# Two halves: keep the first N/2 failures (catches deterministic
# config errors that surface at run start — auth misconfig, schema
# mismatch, wrong endpoint) AND the last N/2 (catches transient mid-
# run failures the operator most wants to see — rate-limit storms,
# model deprecation kicking in, JSON-truncation on a deep oversized
# paragraph). Keeping only the first N would silently drop the
# diagnostic-rich tail; keeping only the last N would erase the
# auth/config errors at the head. Half-and-half is the right shape.
_FAILED_CHUNKS_CAP = 100
_FAILED_CHUNKS_HEAD = _FAILED_CHUNKS_CAP // 2
_FAILED_CHUNKS_TAIL = _FAILED_CHUNKS_CAP - _FAILED_CHUNKS_HEAD


@dataclass
class MineToponymMentionsReport:
    """Aggregate result of mining one source body for mentions.

    Beyond raw counts, this captures the rejection breakdown so an
    operator reading "0 mentions, 5 chunks processed" can tell
    whether the LLM was producing nothing (``mentions`` empty,
    rejections zero) or producing junk (``mentions`` empty,
    ``hallucinations_dropped`` non-zero)."""

    source_id: str
    chunks_processed: int = 0
    chunks_failed: int = 0
    hallucinations_dropped: int = 0
    years_clamped: int = 0
    mentions: list[ToponymMention] = field(default_factory=list)
    # First _FAILED_CHUNKS_CAP failures; the actual count is
    # ``chunks_failed`` (may exceed this list's length when many
    # chunks fail in a single run).
    failed_chunks: list[FailedChunk] = field(default_factory=list)


def mine_toponym_mentions(
    client,
    source_id: str,
    body: str,
    *,
    target_chunk_size: int = 20000,
    limit: int | None = None,
    on_chunk_done: Callable[[int, int, int], None] | None = None,
    log_warning: Callable[[str], None] | None = None,
) -> MineToponymMentionsReport:
    """Walk a source body chunk-by-chunk, accumulating extracted
    mentions. Returns the aggregate report.

    Parameters
    * ``limit`` — if set, process only the first N chunks. Slicing
      happens AFTER chunking, so the caller can't accidentally split
      paragraphs (caller previously did ``"\\n\\n".join(chunks[:N])``
      and re-chunked, which lied about the resulting chunk count).
    * ``on_chunk_done`` — invoked after each SUCCESSFUL chunk with
      ``(chunks_done, total_chunks, mentions_so_far)``. The total is
      the post-limit count, so progress lines display correctly.
    * ``log_warning`` — optional sink for diagnostic warnings
      (oversize paragraph, etc.). Defaults to silent; the CLI wires
      ``click.echo(..., err=True)``.
    """
    from collections import deque

    report = MineToponymMentionsReport(source_id=source_id)
    chunks = chunk_source_body(body, target_chunk_size=target_chunk_size)
    if limit is not None:
        chunks = chunks[:limit]
    total = len(chunks)
    if log_warning is not None:
        oversize_threshold = target_chunk_size * _OVERSIZED_PARAGRAPH_MULTIPLIER
        for i, c in enumerate(chunks):
            if len(c) > oversize_threshold:
                log_warning(
                    f"chunk {i}: oversized paragraph "
                    f"({len(c):,} chars > {int(oversize_threshold):,} threshold) — "
                    f"LLM output may truncate; consider lowering --chunk-size"
                )
    # head_failures: first N/2 captured directly. tail_failures: bounded
    # ring buffer of the most recent N/2. Merged at end before returning.
    head_failures: list[FailedChunk] = []
    tail_failures: deque[FailedChunk] = deque(maxlen=_FAILED_CHUNKS_TAIL)
    for i, chunk in enumerate(chunks):
        chunk_counters = ValidationCounters()
        try:
            mentions = extract_toponym_mentions_from_chunk(client, chunk, counters=chunk_counters)
        except (RuntimeError, json.JSONDecodeError) as e:
            # Per-chunk failure (transport, JSON-parse, schema mismatch)
            # is logged + skipped — don't poison the whole source's
            # output. Operator gets the chunk index + exception type
            # so they can re-run a targeted subset later or pivot to a
            # different provider.
            report.chunks_failed += 1
            failure = FailedChunk(index=i, error=f"{type(e).__name__}: {e}")
            if len(head_failures) < _FAILED_CHUNKS_HEAD:
                head_failures.append(failure)
            else:
                tail_failures.append(failure)
            if log_warning is not None:
                log_warning(f"chunk {i} failed: {type(e).__name__}: {e}")
            continue
        report.mentions.extend(mentions)
        report.chunks_processed += 1
        report.hallucinations_dropped += chunk_counters.hallucinations_dropped
        report.years_clamped += chunk_counters.years_clamped
        if on_chunk_done is not None:
            on_chunk_done(i + 1, total, len(report.mentions))
    # Merge head + tail. List shape preserves indices for the CLI's
    # display loop; the gap between head and tail (if any) is implicit
    # via the chunks_failed counter minus len(failed_chunks).
    report.failed_chunks.extend(head_failures)
    report.failed_chunks.extend(tail_failures)
    return report
