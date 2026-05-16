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
``llm_extractor.py``. The tiered-extraction pattern (Qwen bulk →
Anthropic on the residual) is implemented inline as
:func:`mine_toponym_mentions_tiered` below.

Chunking limit: ``chunk_source_body`` does NOT overlap chunks; a
mention whose year/region hint falls on the OTHER side of a
paragraph boundary will be split, and the LLM will see only one
side of the pair. Paragraph boundaries are the strongest signal we
have for keeping each chunk's view coherent — sentence-overlap
chunking is a documented follow-up (DECISIONS.md if pursued).
"""

from __future__ import annotations

import re
from collections import deque
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
# in standard JSON-Schema dialect (Ollama). Anthropic doesn't support
# this; for Anthropic the SYSTEM_PROMPT's strict-JSON instruction +
# post-hoc parse handles it.
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


# Gemini's responseSchema is OpenAPI-3.0-ish, NOT standard JSON
# Schema. Differences vs RESPONSE_SCHEMA above:
#   * uppercase type names ("OBJECT", "ARRAY", "STRING", "INTEGER")
#   * no `additionalProperties` keyword (Gemini's schema validator
#     rejects the keyword itself at schema-submit time with HTTP 400
#     `Unknown name "additionalProperties"`, NOT an enforcement of
#     "no unknown keys at response time")
#   * no `type: ["X", "null"]` unions — use `nullable: true` instead
#   * nullable fields must appear in `required` so the model emits
#     null explicitly rather than omitting the key (matches the
#     etymology extractor's GEMINI_RESPONSE_SCHEMA convention).
# Passing RESPONSE_SCHEMA (JSON-Schema dialect) directly to Gemini
# fails with HTTP 400 on every chunk — confirmed wyrd-pe4g.
# Closed set of dialect strings accepted by the dispatch in
# ``extract_toponym_mentions_from_chunk``. A client declaring any
# other value (typo, future provider that hasn't been added yet)
# raises rather than silently falling through to JSON Schema —
# silent-failure-hunter wyrd-pe4g round-2 finding.
_KNOWN_SCHEMA_DIALECTS = frozenset({"openapi", "json-schema"})


RESPONSE_SCHEMA_GEMINI: dict = {
    "type": "OBJECT",
    "properties": {
        "mentions": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "form": {"type": "STRING"},
                    "date_year": {"type": "INTEGER", "nullable": True},
                    "region_hint": {"type": "STRING", "nullable": True},
                    "context": {"type": "STRING"},
                },
                "required": ["form", "date_year", "region_hint", "context"],
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
    ``(?<!\\w)...(?!\\w)`` (lookaround) so a partial like ``"on Ty"``
    does NOT match ``"...upon Tyne..."``.

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
            # The form-in-chunk check above already passed (regex with
            # lookaround word-boundaries), so we know the form appears
            # at SOME boundary in the collapsed chunk. Search the raw
            # chunk first with word-boundary regex; fall back to the
            # collapsed view if soft-wrap means the form's whitespace
            # isn't literal in the raw chunk. Substring find() here
            # would risk locating a fragment INSIDE a longer word
            # (e.g. "on" inside "upon"/"Northampton") — the same
            # hazard the form-in-chunk guard prevents.
            form_collapsed = _collapse_whitespace(form)
            pattern = r"(?<!\w)" + re.escape(form_collapsed) + r"(?!\w)"
            raw_match = re.search(pattern, chunk)
            if raw_match is not None:
                start = max(0, raw_match.start() - 40)
                end = min(len(chunk), raw_match.end() + 40)
                context = _WHITESPACE_RUN.sub(" ", chunk[start:end]).strip()
            else:
                # Soft-wrap fallback: form spans a line break in raw
                # chunk so it's not found verbatim. The collapsed chunk
                # has it; locate it there with the same word-boundary
                # check to avoid in-word matches.
                collapsed_match = re.search(pattern, chunk_collapsed)
                if collapsed_match is not None:
                    start = max(0, collapsed_match.start() - 40)
                    end = min(len(chunk_collapsed), collapsed_match.end() + 40)
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
    #
    # Schema dialect dispatch: Gemini's responseSchema is OpenAPI-3.0,
    # not JSON Schema (see wyrd-pe4g + comment on RESPONSE_SCHEMA_GEMINI
    # above). Each client declares its dialect via a ``schema_dialect``
    # class attribute; absence defaults to "json-schema". The attribute-
    # based contract survives renames + subclasses (inherited via MRO)
    # and lets new providers declare their dialect without editing this
    # dispatch site.
    #
    # Whitelist guard: a typo like "Openapi"/"OpenAPI"/"openapi " on a
    # Gemini-talking client would otherwise silently fall through to
    # RESPONSE_SCHEMA (the default-on-mismatch path) and ship JSON Schema
    # to Gemini's API → HTTP 400 on every chunk with no dispatch-level
    # diagnostic. Raise loudly here so the cause surfaces at the call
    # site instead.
    dialect = getattr(client, "schema_dialect", "json-schema")
    if dialect not in _KNOWN_SCHEMA_DIALECTS:
        raise ValueError(
            f"unknown schema_dialect {dialect!r} on {type(client).__name__}; "
            f"expected one of {sorted(_KNOWN_SCHEMA_DIALECTS)}"
        )
    schema = RESPONSE_SCHEMA_GEMINI if dialect == "openapi" else RESPONSE_SCHEMA
    response = client.chat_json(SYSTEM_PROMPT, user, schema)
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


# Programmer-error exception types — re-raised before the generic
# per-chunk Exception catch so a typo or missing-method bug surfaces
# as a Python traceback rather than a silent 100%-failed run. These
# are not the kind of errors that should ever happen at runtime in a
# correct deployment; if they do, something is broken at the code
# level and the operator needs to see the stack.
#
# ValueError is in this tuple because the only source of ValueError
# on the chunk path is the dispatch-time whitelist guard for an
# unknown ``schema_dialect`` value (wyrd-pe4g round-2 + round-3:
# silent-failure-hunter). That's a misconfiguration on the client
# class, not a per-chunk transport hiccup; without re-raising, a
# typo'd dialect would silently produce 100% chunks_failed and
# (in tiered mode) the fallback would mask the misconfiguration
# entirely.
#
# For this carve-out to stay scoped to configuration errors, every
# transport-layer JSON parse path must funnel JSONDecodeError (a
# ValueError subclass) into RuntimeError. Both the inner content
# parses (anthropic_extractor.py / gemini_extractor.py /
# llm_extractor.py `json.loads(content)` blocks) AND the outer
# envelope parses (`json.loads(body)`) are wrapped to maintain
# this invariant — without wrapping the envelope parse, a CDN/proxy
# HTML error page on a 2xx response or a truncated body would
# leak ValueError into this re-raise and abort the run.
# wyrd-pe4g round-4: code-reviewer + silent-failure-hunter +
# comment-analyzer all converged on the missing envelope wrap.
_PROGRAMMER_ERROR_EXCEPTIONS: tuple[type[Exception], ...] = (
    AttributeError,
    TypeError,
    NameError,
    ImportError,
    ValueError,
)


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
    # Chunks the PRIMARY client failed AND the fallback client
    # subsequently rescued (tiered extraction only — stays 0 for
    # single-tier runs). Subtract from chunks_processed to learn
    # how many chunks the primary handled on its own.
    chunks_recovered_by_fallback: int = 0
    hallucinations_dropped: int = 0
    years_clamped: int = 0
    mentions: list[ToponymMention] = field(default_factory=list)
    # First _FAILED_CHUNKS_HEAD failures concatenated with the most-
    # recent _FAILED_CHUNKS_TAIL failures (half-and-half retention so
    # operators see both the auth/schema-mismatch errors that surface
    # at run start AND the rate-limit/mid-run transients at the tail).
    # When more than (HEAD + TAIL) chunks fail, the middle is elided
    # — ``chunks_failed`` counter remains the authoritative total;
    # the CLI surfaces the gap via ``chunks_failed - len(failed_chunks)``.
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
    report = MineToponymMentionsReport(source_id=source_id)
    chunks = chunk_source_body(body, target_chunk_size=target_chunk_size)
    if limit is not None:
        chunks = chunks[:limit]
    total = len(chunks)
    oversize_threshold = target_chunk_size * _OVERSIZED_PARAGRAPH_MULTIPLIER
    # head_failures: first N/2 captured directly. tail_failures: bounded
    # ring buffer of the most recent N/2. Merged at end before returning.
    head_failures: list[FailedChunk] = []
    tail_failures: deque[FailedChunk] = deque(maxlen=_FAILED_CHUNKS_TAIL)
    for i, chunk in enumerate(chunks):
        # Inline oversize warning — folded into the main loop to
        # avoid a separate pre-pass over the chunk list. Fires
        # BEFORE the LLM call so operators see the warning
        # interleaved with progress.
        if log_warning is not None and len(chunk) > oversize_threshold:
            log_warning(
                f"chunk {i}: oversized paragraph "
                f"({len(chunk):,} chars > {int(oversize_threshold):,} threshold) — "
                f"LLM output may truncate; consider lowering --chunk-size"
            )
        chunk_counters = ValidationCounters()
        try:
            mentions = extract_toponym_mentions_from_chunk(client, chunk, counters=chunk_counters)
        except _PROGRAMMER_ERROR_EXCEPTIONS:
            # Re-raise: AttributeError / TypeError / NameError /
            # ImportError are our bugs, not the LLM's. Letting them
            # propagate as tracebacks beats counting them as
            # per-chunk failures across every chunk in a 100%-failed
            # run with no diagnostic.
            raise
        except Exception as e:
            # Per-chunk failure (transport, JSON-parse, schema mismatch,
            # provider SDK exceptions like anthropic.APIError /
            # httpx.HTTPError / TimeoutError) is logged + skipped —
            # don't poison the whole source's output. Catching bare
            # Exception is intentional: a real run will see provider-
            # SDK error classes that don't derive from RuntimeError,
            # and we'd rather degrade gracefully per-chunk than abort
            # the whole multi-source run. Programmer-error types are
            # re-raised in the prior clause; KeyboardInterrupt and
            # SystemExit (BaseException) still propagate naturally.
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


def mine_toponym_mentions_tiered(
    primary_client,
    fallback_client,
    source_id: str,
    body: str,
    *,
    target_chunk_size: int = 20000,
    limit: int | None = None,
    on_chunk_done: Callable[[int, int, int], None] | None = None,
    log_warning: Callable[[str], None] | None = None,
) -> MineToponymMentionsReport:
    """Two-tier extraction: try ``primary_client`` on every chunk;
    when the primary fails (transport, JSON parse, schema mismatch),
    retry with ``fallback_client``. Built for the wyrd-l0r tiered
    pattern: Qwen-on-Hades for bulk first-pass (free, fast on
    GPU-hosted Ollama), Anthropic on the residual (better quality
    on the tricky chunks the small model couldn't parse).

    A chunk is counted as ``chunks_recovered_by_fallback`` when the
    primary raised AND the fallback succeeded. A chunk where BOTH
    tiers fail counts as a ``failed_chunk`` with BOTH errors recorded
    as ``primary=...; fallback=...`` so operators can see whether
    the chunk is genuinely hard (both quality tiers failed) vs. a
    transient issue (one tier transient-failed, the other timed out
    differently).

    Both client calls catch bare ``Exception`` so provider-SDK error
    classes (``anthropic.APIError``, ``httpx.HTTPError``,
    ``TimeoutError``) trigger the fallback rather than aborting the
    whole multi-source run. ``KeyboardInterrupt`` / ``SystemExit``
    still propagate.

    Resolved-mention counters (``hallucinations_dropped``,
    ``years_clamped``) reflect ONLY the tier that succeeded — the
    losing tier's per-chunk counters are discarded so a partial-
    populate from a primary that raised mid-validation can't double-
    count. ``extract_toponym_mentions_from_chunk`` does not currently
    populate counters partially before raising, so the discarded-
    state is defensive against future refactors.

    The report shape matches :class:`MineToponymMentionsReport` so
    the CLI's summary line and the Phase 2b.1 ingester both work
    unchanged.
    """
    report = MineToponymMentionsReport(source_id=source_id)
    chunks = chunk_source_body(body, target_chunk_size=target_chunk_size)
    if limit is not None:
        chunks = chunks[:limit]
    total = len(chunks)
    oversize_threshold = target_chunk_size * _OVERSIZED_PARAGRAPH_MULTIPLIER
    head_failures: list[FailedChunk] = []
    tail_failures: deque[FailedChunk] = deque(maxlen=_FAILED_CHUNKS_TAIL)
    for i, chunk in enumerate(chunks):
        # Inline oversize warning (matches the single-tier shape).
        # Fires before either LLM call so the operator sees the
        # warning interleaved with progress.
        if log_warning is not None and len(chunk) > oversize_threshold:
            log_warning(
                f"chunk {i}: oversized paragraph "
                f"({len(chunk):,} chars > {int(oversize_threshold):,} threshold) — "
                f"LLM output may truncate; consider lowering --chunk-size"
            )
        chunk_counters = ValidationCounters()
        primary_error: str | None = None
        mentions: list[ToponymMention] | None = None
        try:
            mentions = extract_toponym_mentions_from_chunk(
                primary_client, chunk, counters=chunk_counters
            )
        except _PROGRAMMER_ERROR_EXCEPTIONS:
            raise  # Our bug — let it surface as a traceback.
        except Exception as e:
            # Bare Exception (not RuntimeError) so provider-SDK error
            # classes (anthropic.APIError, httpx.HTTPError, TimeoutError)
            # trigger fallback rather than aborting the whole source.
            primary_error = f"{type(e).__name__}: {e}"
            if log_warning is not None:
                log_warning(f"chunk {i} primary failed: {primary_error} — retrying with fallback")
        if mentions is None:
            # Primary failed. Try fallback. Reset counters defensively
            # — extract_toponym_mentions_from_chunk doesn't populate
            # counters partially before raising today, but a future
            # refactor could. Keeping the discard explicit prevents
            # double-counting drift if that contract changes.
            chunk_counters = ValidationCounters()
            try:
                mentions = extract_toponym_mentions_from_chunk(
                    fallback_client, chunk, counters=chunk_counters
                )
                report.chunks_recovered_by_fallback += 1
            except _PROGRAMMER_ERROR_EXCEPTIONS:
                raise  # Our bug — let it surface as a traceback.
            except Exception as e:
                # BOTH tiers failed — record BOTH errors so operators
                # see the full failure chain (transient vs. genuinely-
                # hard chunk are distinguishable from the pair).
                fallback_error = f"{type(e).__name__}: {e}"
                report.chunks_failed += 1
                failure = FailedChunk(
                    index=i,
                    error=f"primary={primary_error}; fallback={fallback_error}",
                )
                if len(head_failures) < _FAILED_CHUNKS_HEAD:
                    head_failures.append(failure)
                else:
                    tail_failures.append(failure)
                if log_warning is not None:
                    log_warning(f"chunk {i} both tiers failed: {fallback_error}")
                continue
        report.mentions.extend(mentions)
        report.chunks_processed += 1
        report.hallucinations_dropped += chunk_counters.hallucinations_dropped
        report.years_clamped += chunk_counters.years_clamped
        if on_chunk_done is not None:
            on_chunk_done(i + 1, total, len(report.mentions))
    report.failed_chunks.extend(head_failures)
    report.failed_chunks.extend(tail_failures)
    return report
