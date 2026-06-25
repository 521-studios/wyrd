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
``extractors/llm.py``. The tiered-extraction pattern (Qwen bulk →
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


# Compiled at module scope to avoid per-mention recompilation. The
# fast-path identity-preservation of CPython's ``re.sub`` (no match →
# original object returned) is what lets _sanitize_for_utf8 cheaply
# handle the common no-surrogate case.
_LONE_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def _collapse_whitespace(text: str) -> str:
    """Collapse every whitespace run (spaces, tabs, newlines,
    soft-wrap hyphen + newline) to a single space. Used to make the
    form-in-chunk check tolerant of OCR/typesetter line breaks that
    the LLM correctly emits as bare-form mentions."""
    return _WHITESPACE_RUN.sub(" ", text).strip()


def _sanitize_for_utf8(text: str) -> str:
    """Replace unpaired surrogate codepoints (U+D800..U+DFFF) with
    U+FFFD so the string can round-trip through a utf-8 file write.

    json.loads accepts strings containing lone surrogates from a
    permissive LLM response, but a subsequent utf-8 write of the
    same string raises UnicodeEncodeError ('surrogates not allowed').
    Sanitizing at validation time keeps the crash from reaching the
    writer (wyrd-8wa4: concrete failure 2026-05-21 on
    longnon_1920_v2 mid-source crashed mining).

    Applied to every LLM-emitted str field before validation. A
    surrogate landing in ``form`` becomes U+FFFD and then almost
    always fails the form-in-chunk anti-hallucination check (the
    source body is decoded with errors='replace' on read, so
    legitimate U+FFFD appears only at corruption sites — the LLM-
    introduced U+FFFD is unlikely to align with one) — the mention
    drops via the existing ``hallucinations_dropped`` accounting. A
    surrogate in ``context`` / ``region_hint`` gets replaced and the
    mention is admitted with a U+FFFD marker visible to operators.
    Either way ``surrogates_sanitized`` bumps on the run's counters
    so operators can spot a model emitting bad codepoints at scale.
    """
    if not text:
        return text
    # CPython's re.sub returns the original string object identity-
    # equal when no substitution occurs, so the no-surrogate common
    # case allocates nothing.
    return _LONE_SURROGATE_RE.sub("�", text)


def _sanitize_and_count(text: str, counters: ValidationCounters) -> str:
    """Sanitize ``text`` and bump ``counters.surrogates_sanitized`` iff
    the sanitizer actually substituted anything.

    Defined here (not inline) so the three sanitize-and-count sites in
    ``_validated_mentions`` can't drift: extracting the pattern means a
    new free-text field added to ``ToponymMention`` only needs one
    extra call here, not a paired call+if. Relies on
    ``_sanitize_for_utf8``'s identity-preserving fast path — ``is not``
    on the input vs. output distinguishes "we did work" from "we
    returned the input unchanged."
    """
    sanitized = _sanitize_for_utf8(text)
    if sanitized is not text:
        counters.surrogates_sanitized += 1
    return sanitized


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


# Single source of truth for the closed set of dialects + which
# RESPONSE_SCHEMA each one selects. The dict's keys serve as the
# whitelist for ``extract_toponym_mentions_from_chunk``'s guard;
# its values are the dispatch targets. Keeping the two coupled
# in one structure means adding a new dialect can't silently fall
# through to JSON Schema by accident — the whitelist update and
# the dispatch update are the same edit. wyrd-pe4g round-2
# (silent-failure-hunter: typo guard) + round-6 (Gemini bot +
# type-design-analyzer: collapse to one structure).
_DIALECT_TO_SCHEMA: dict[str, dict] = {
    "openapi": RESPONSE_SCHEMA_GEMINI,
    "json-schema": RESPONSE_SCHEMA,
}


@dataclass
class ValidationCounters:
    """Per-call counters for what ``_validated_mentions`` rejected.

    Used by the orchestrator to roll up across all chunks, so
    operators can see the failure-mode breakdown (vs. an opaque
    "0 mentions admitted" result is a load-bearing UX gap).

    Helpers that produce a counters delta (e.g.
    :func:`extract_toponym_mentions_from_chunk`,
    :func:`_run_hallucination_rescue`) return a fresh instance for
    the caller to fold into a per-chunk accumulator via ``+=``.

    No ``__add__`` companion is provided — pure summation (``a + b``)
    isn't a supported operation, by design. Counters are
    fold-into-accumulator only; the absence of ``__add__`` keeps a
    standalone ``total = a + b`` from quietly producing a broken
    detached instance.

    The ``__iadd__`` implementation folds ALL three fields including
    ``admitted``. The pre-wyrd-oaq5 hand-rolled rescue merge in
    :func:`_run_hallucination_rescue` folded only
    ``hallucinations_dropped`` and ``years_clamped``, leaving
    ``admitted`` out. The orchestrator never reads
    ``chunk_counters.admitted`` into the report
    (:class:`MineToponymMentionsReport` lacks an ``admitted`` field
    today), so this is operationally inert. If a future PR adds
    ``report.admitted += chunk_counters.admitted``, be aware the
    hallucination-rescue path now double-counts admits across both
    tiers — that may want explicit accounting at the fold site
    rather than the field-wide ``+=`` shape.
    """

    admitted: int = 0
    hallucinations_dropped: int = 0
    years_clamped: int = 0  # any year value the LLM emitted that we coerced to None
    # any LLM-emitted str field that contained a lone surrogate the
    # validator had to replace with U+FFFD (wyrd-8wa4). Tracked so
    # operators can spot a model regression where surrogates start
    # appearing at scale — silent rewrites would otherwise look like
    # inflated hallucinations_dropped (when the surrogate is in form)
    # or invisible quality decay (when it's in context/region_hint).
    surrogates_sanitized: int = 0

    def __iadd__(self, other: ValidationCounters) -> ValidationCounters:
        self.admitted += other.admitted
        self.hallucinations_dropped += other.hallucinations_dropped
        self.years_clamped += other.years_clamped
        self.surrogates_sanitized += other.surrogates_sanitized
        return self


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


def _parse_mention_fields(
    m: dict,
    counters: ValidationCounters,
) -> tuple[str, str | None, str] | None:
    """Type-guard + surrogate-sanitize the LLM-emitted fields of one raw
    mention dict, returning ``(form, region_hint, context)`` or None to skip.

    Sanitize lone surrogates on each free-text LLM-emitted str field (form,
    region_hint, context) up front, before any validation that might
    short-circuit. The surrogates_sanitized counter must reflect what the
    upstream model actually emitted, not what survived later validation —
    otherwise a model regression that smears surrogates across every field
    looks identical to one that only damaged form (which would short-circuit
    at form-in-chunk before the validator ever inspects region_hint /
    context). date_year is also LLM-emitted, but ``_coerce_year`` enforces a
    digit-only regex that drops any surrogate-bearing value to None+clamped —
    no separate sanitization needed on that path. (wyrd-8wa4.)

    wyrd-2iiy: type-guard each LLM-emitted field before any string operation.
    With the wyrd-sj50 switch from format=<schema> to format='json', the
    Ollama server no longer constrains output shape — gemma4 occasionally
    returns ``form`` (or other fields) as a dict / list / int. Without these
    guards, the next ``.strip()`` AttributeErrors and the whole source aborts
    mid-mining. Returns None to drop the mention; counts a wrong-type form as
    a hallucination (a missing form is an uncounted protocol error — matches
    the pre-wyrd-2iiy contract the empty/None-form test pins).
    """
    raw_form = m.get("form")
    if raw_form is None:
        return None
    if not isinstance(raw_form, str):
        counters.hallucinations_dropped += 1
        return None
    form = _sanitize_and_count(raw_form.strip(), counters)
    raw_region = m.get("region_hint")
    if isinstance(raw_region, str):
        # Strip before sanitize so the sanitizer operates on a potentially
        # shorter string and the processing order matches the form/context
        # paths.
        region_hint: str | None = _sanitize_and_count(raw_region.strip(), counters) or None
    else:
        region_hint = None
    raw_context = m.get("context")
    if not isinstance(raw_context, str):
        # Non-str context isn't fatal: form is the load-bearing field. Treat
        # as missing context (empty string) and let the synthesize-from-chunk
        # path fill it.
        raw_context = ""
    context = _sanitize_and_count(_WHITESPACE_RUN.sub(" ", raw_context).strip(), counters)
    return (form, region_hint, context)


def _synthesize_context(form: str, chunk: str, chunk_collapsed: str) -> str:
    """A +/-40-char context window around ``form`` for when the LLM left
    context empty — gives operators something to inspect. Word-boundary regex
    (not substring) so a fragment INSIDE a longer word (e.g. "on" inside
    "upon"/"Northampton") can't match — the same hazard the form-in-chunk
    guard prevents. Searches the raw chunk first; falls back to the collapsed
    view when soft-wrap means the form's whitespace isn't literal in the raw
    chunk. Returns "" when neither locates it."""
    form_collapsed = _collapse_whitespace(form)
    pattern = r"(?<!\w)" + re.escape(form_collapsed) + r"(?!\w)"
    raw_match = re.search(pattern, chunk)
    if raw_match is not None:
        start = max(0, raw_match.start() - 40)
        end = min(len(chunk), raw_match.end() + 40)
        return _WHITESPACE_RUN.sub(" ", chunk[start:end]).strip()
    collapsed_match = re.search(pattern, chunk_collapsed)
    if collapsed_match is not None:
        start = max(0, collapsed_match.start() - 40)
        end = min(len(chunk_collapsed), collapsed_match.end() + 40)
        return chunk_collapsed[start:end]
    return ""


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
        fields = _parse_mention_fields(m, counters)
        if fields is None:
            # Missing form (uncounted protocol error) or wrong-type form
            # (counted hallucination) — the helper handled counting; skip.
            continue
        form, region_hint, context = fields
        # Validation gates fire AFTER full sanitization above.
        if not form:
            continue
        if not _form_in_chunk(form, chunk_collapsed):
            counters.hallucinations_dropped += 1
            continue
        date_year, was_clamped = _coerce_year(m.get("date_year"))
        if was_clamped:
            counters.years_clamped += 1
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
            context = _synthesize_context(form, chunk, chunk_collapsed)
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
) -> tuple[list[ToponymMention], ValidationCounters]:
    """Run one LLM call against ``chunk`` and return
    ``(validated_mentions, counters)``. ``client`` must have a
    ``chat_json(system, user, schema=None) -> dict`` method — same
    contract as :class:`AnthropicClient`, :class:`OllamaClient`,
    :class:`GeminiClient` in the existing extractor modules.

    The returned :class:`ValidationCounters` is a fresh delta — the
    caller folds it into its per-chunk accumulator via ``+=`` (see
    :func:`ValidationCounters.__iadd__`). This delta-return shape
    (wyrd-oaq5) keeps the helper free of caller-supplied mutable
    state: it produces a value, the caller decides how to combine it.

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
    # Gemini-talking client would otherwise produce a KeyError from the
    # dict lookup, which the chunk loop's bare-Exception catch would
    # silently bucket into chunks_failed (and in tiered mode, the
    # fallback would mask the misconfiguration). Raise ValueError with
    # the offending value + class name so the dispatch-level cause
    # surfaces immediately. _DIALECT_TO_SCHEMA is the single source of
    # truth for both the whitelist and the dispatch — adding a new
    # dialect is one edit, no two-place sync.
    dialect = getattr(client, "schema_dialect", "json-schema")
    if dialect not in _DIALECT_TO_SCHEMA:
        raise ValueError(
            f"unknown schema_dialect {dialect!r} on {type(client).__name__}; "
            f"expected one of {sorted(_DIALECT_TO_SCHEMA)}"
        )
    schema = _DIALECT_TO_SCHEMA[dialect]
    response = client.chat_json(SYSTEM_PROMPT, user, schema)
    if not isinstance(response, dict):
        raise RuntimeError(f"LLM returned non-dict envelope: {type(response).__name__}")
    raw = response.get("mentions")
    if raw is None:
        raise RuntimeError("LLM response missing 'mentions' key")
    if not isinstance(raw, list):
        raise RuntimeError(f"LLM response 'mentions' was {type(raw).__name__}, expected list")
    counters = ValidationCounters()
    mentions = _validated_mentions(raw, chunk, counters=counters)
    return mentions, counters


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
# transport-layer path that produces a ValueError-subclass must
# funnel it into RuntimeError. Three classes to keep in sync across
# all three clients (extractors/anthropic.py / extractors/gemini.py /
# extractors/llm.py):
#   1. UnicodeDecodeError from `.decode("utf-8")` on the success
#      response body (wyrd-pe4g round-5: silent-failure-hunter)
#   2. JSONDecodeError from outer `json.loads(body)` envelope parse
#      (wyrd-pe4g round-4: code-reviewer + silent-failure-hunter +
#      comment-analyzer)
#   3. JSONDecodeError from inner `json.loads(text/content)` parse
#      of the model-produced JSON (pre-existing)
# All nine call sites — for each of the 3 clients: outer
# envelope parse, inner content/text parse, and success-path decode
# — are wrapped to maintain this invariant. Without each wrap, a
# CDN/proxy HTML error page, a truncated 2xx body, or a binary blob
# on the success path would leak ValueError into this re-raise and
# abort the multi-source run.
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
    JSON-truncation, schema-mismatch) — vs. an opaque count.

    ``chunk_body`` carries the chunk text verbatim so the staged-cascade
    pipeline (wyrd-srd2) can re-run a failed chunk against a different
    model without re-chunking the original source. Both production
    constructors (``mine_toponym_mentions`` and
    ``mine_toponym_mentions_tiered``) always populate it; the default
    of ``""`` exists so test fixtures + diagnostic-only callers can
    still construct a ``FailedChunk`` without the body. The
    staged-cascade resume path validates non-empty at read time so an
    accidentally-empty record is rejected loudly."""

    index: int
    error: str
    chunk_body: str = ""


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
    # Chunks where the primary SUCCEEDED but emitted enough
    # hallucinated forms (per the chunk-level word-boundary guard)
    # that the operator's --hallucination-fallback-threshold fired
    # and we ran the fallback ALSO. Two counters, deliberately
    # separate (silent-failure-hunter wyrd-z8mq round 1 + round 2):
    # _attempted bumps on every rescue trigger regardless of
    # outcome; _succeeded bumps only when the fallback returned a
    # NON-EMPTY mentions list — i.e., actually delivered content,
    # not just "didn't raise". A gap (attempted > succeeded) is
    # the operationally critical signal that catches both
    # transport failures (bad API key, network outage) AND
    # silent-empty responses (content-policy refusal, mis-shaped
    # JSON producing {"mentions": []}). Without this distinction
    # the single-counter `halluc_rescued=N` would look healthy
    # while gemma4's hallucinations stayed unrescued. Distinct
    # from chunks_recovered_by_fallback (where the primary failed
    # outright). Tiered runs only; both stay 0 when the threshold
    # is disabled or unset.
    chunks_hallucination_rescue_attempted: int = 0
    chunks_hallucination_rescue_succeeded: int = 0
    hallucinations_dropped: int = 0
    years_clamped: int = 0
    # Lone-surrogate codepoints we replaced with U+FFFD across all
    # validated fields (wyrd-8wa4). A non-zero value here is operator
    # signal: the upstream model produced strings the utf-8 writer
    # would have rejected. Per-source; rolled up by the CLI.
    surrogates_sanitized: int = 0
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
    on_chunk_failed: Callable[[FailedChunk], None] | None = None,
    on_chunk_mentions: Callable[[int, list[ToponymMention]], None] | None = None,
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
    * ``on_chunk_failed`` — invoked for every failed chunk with the
      ``FailedChunk`` record (including chunk_body). Used by the
      staged-cascade CLI to stream failures to a JSONL file as they
      happen (wyrd-srd2) — bypassing the report's in-memory cap.
      Defaults to None (silent).
    * ``on_chunk_mentions`` — invoked for every SUCCESSFUL chunk with
      ``(chunk_index, mentions_in_this_chunk)`` BEFORE the report's
      counters are bumped and before ``on_chunk_done``. Used by the
      staged-cascade CLI (wyrd-w7i3) to stream per-chunk mentions to
      a durable in-progress log so SIGTERM/crash mid-source doesn't
      lose hours of LLM inference. The list is the chunk's mentions
      only (not the cumulative report). Exceptions raised by this
      callback are NOT caught — they propagate and abort the source.
      This is intentional: a persistence failure (e.g. fsync OSError)
      must not silently flip a chunk from "failed" to "processed".
      Defaults to None.
    * ``log_warning`` — optional sink for diagnostic warnings
      (oversize paragraph, etc.). Defaults to silent; the CLI wires
      ``click.echo(..., err=True)``.
    """
    chunks = chunk_source_body(body, target_chunk_size=target_chunk_size)
    if limit is not None:
        chunks = chunks[:limit]
    return _process_indexed_chunks(
        client,
        source_id,
        list(enumerate(chunks)),
        target_chunk_size=target_chunk_size,
        on_chunk_done=on_chunk_done,
        on_chunk_failed=on_chunk_failed,
        on_chunk_mentions=on_chunk_mentions,
        log_warning=log_warning,
    )


def mine_toponym_mentions_from_chunks(
    client,
    source_id: str,
    indexed_chunks: list[tuple[int, str]],
    *,
    target_chunk_size: int = 20000,
    on_chunk_done: Callable[[int, int, int], None] | None = None,
    on_chunk_failed: Callable[[FailedChunk], None] | None = None,
    on_chunk_mentions: Callable[[int, list[ToponymMention]], None] | None = None,
    log_warning: Callable[[str], None] | None = None,
) -> MineToponymMentionsReport:
    """Mine a list of pre-chunked ``(chunk_index, chunk_body)`` tuples.

    Used by the staged-cascade pipeline (wyrd-srd2) to retry chunks
    that failed at an earlier stage. The chunk indices are preserved
    from the original chunking pass so the failed-chunks file remains
    coherent across stages (a failure at chunk 47 stays chunk 47 even
    when stage 2 sees only the subset of chunks stage 1 couldn't
    handle).

    Same return shape and callback contract as :func:`mine_toponym_mentions`.
    ``target_chunk_size`` is retained for the oversize-warning
    threshold; it doesn't drive re-chunking here since chunks are
    already supplied.
    """
    return _process_indexed_chunks(
        client,
        source_id,
        indexed_chunks,
        target_chunk_size=target_chunk_size,
        on_chunk_done=on_chunk_done,
        on_chunk_failed=on_chunk_failed,
        on_chunk_mentions=on_chunk_mentions,
        log_warning=log_warning,
    )


def _record_chunk_failure(
    report: MineToponymMentionsReport,
    head_failures: list[FailedChunk],
    tail_failures: deque[FailedChunk],
    *,
    index: int,
    chunk: str,
    error: Exception,
    on_chunk_failed: Callable[[FailedChunk], None] | None,
    log_warning: Callable[[str], None] | None,
) -> None:
    """Record one per-chunk LLM failure: count it on ``report``, capture it into
    the head/tail ring buffer (first ``_FAILED_CHUNKS_HEAD`` directly, the rest in
    the bounded ``tail_failures`` deque), and fire the ``on_chunk_failed`` callback
    + the operator warning.

    The error string is UTF-8-sanitized (wyrd-8wa4): provider-SDK exceptions can
    carry LLM response excerpts (or stringified JSON parse offsets) containing
    lone surrogates, and without this the writer that persists failed-chunk
    records would hit that same crash class one layer downstream.
    """
    report.chunks_failed += 1
    failure = FailedChunk(
        index=index,
        error=_sanitize_for_utf8(f"{type(error).__name__}: {error}"),
        chunk_body=chunk,
    )
    if len(head_failures) < _FAILED_CHUNKS_HEAD:
        head_failures.append(failure)
    else:
        tail_failures.append(failure)
    if on_chunk_failed is not None:
        on_chunk_failed(failure)
    if log_warning is not None:
        log_warning(f"chunk {index} failed: {type(error).__name__}: {error}")


def _process_indexed_chunks(
    client,
    source_id: str,
    indexed_chunks: list[tuple[int, str]],
    *,
    target_chunk_size: int,
    on_chunk_done: Callable[[int, int, int], None] | None,
    on_chunk_failed: Callable[[FailedChunk], None] | None,
    on_chunk_mentions: Callable[[int, list[ToponymMention]], None] | None,
    log_warning: Callable[[str], None] | None,
) -> MineToponymMentionsReport:
    """Inner per-chunk loop shared by :func:`mine_toponym_mentions`
    (fresh chunking) and :func:`mine_toponym_mentions_from_chunks`
    (resume from a failures file)."""
    report = MineToponymMentionsReport(source_id=source_id)
    total = len(indexed_chunks)
    oversize_threshold = target_chunk_size * _OVERSIZED_PARAGRAPH_MULTIPLIER
    # head_failures: first N/2 captured directly. tail_failures: bounded
    # ring buffer of the most recent N/2. Merged at end before returning.
    head_failures: list[FailedChunk] = []
    tail_failures: deque[FailedChunk] = deque(maxlen=_FAILED_CHUNKS_TAIL)
    for done, (i, chunk) in enumerate(indexed_chunks, start=1):
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
        try:
            mentions, chunk_counters = extract_toponym_mentions_from_chunk(client, chunk)
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
            _record_chunk_failure(
                report,
                head_failures,
                tail_failures,
                index=i,
                chunk=chunk,
                error=e,
                on_chunk_failed=on_chunk_failed,
                log_warning=log_warning,
            )
            continue
        # wyrd-w7i3 crash-safety ordering: on_chunk_mentions fires
        # FIRST, before any mutation of `report` and before the
        # operator-visible progress echo. If the callback raises
        # (e.g. fsync OSError on a full disk), `chunks_processed`
        # stays accurate — the chunk is treated as not-yet-processed
        # and the exception propagates to abort the source. Without
        # this ordering, a fsync failure would leave the report
        # claiming a chunk was processed that has no durable record.
        # Any exception raised by on_chunk_mentions is intentionally
        # NOT caught here; persistence failures should abort the
        # source rather than silently skip a chunk.
        if on_chunk_mentions is not None:
            on_chunk_mentions(i, mentions)
        report.mentions.extend(mentions)
        report.chunks_processed += 1
        report.hallucinations_dropped += chunk_counters.hallucinations_dropped
        report.years_clamped += chunk_counters.years_clamped
        report.surrogates_sanitized += chunk_counters.surrogates_sanitized
        if on_chunk_done is not None:
            on_chunk_done(done, total, len(report.mentions))
    # Merge head + tail. List shape preserves indices for the CLI's
    # display loop; the gap between head and tail (if any) is implicit
    # via the chunks_failed counter minus len(failed_chunks).
    report.failed_chunks.extend(head_failures)
    report.failed_chunks.extend(tail_failures)
    return report


def _hallucination_rescue_dedup_key(
    mention: ToponymMention,
) -> tuple[str, int | None, str | None, str]:
    """Build the union-merge dedup key for the hallucination rescue
    path (wyrd-z8mq). Includes ``context`` deliberately: two
    extractions of the SAME chunk emitting the same form+year+region
    from different sentences are different attestations, not
    duplicates, and downstream JSONL serialization preserves
    ``context`` per row. Collapsing them on (form, year, region)
    alone would silently drop real attestation rows (silent-
    failure-hunter wyrd-z8mq round 1).

    Note: this is broader than the DB-side dedup in the Phase 2b.1
    ingester (``toponym_mention_ingest.py``), which dedupes on
    ``(toponym_id, form, date_year, source_doc)`` AFTER mentions are
    resolved to a toponym_id. The two layers' equivalence definitions
    are intentionally different: at extraction time we don't yet
    know the resolver verdict, so we keep ALL distinct attestations
    and let the ingester collapse on its narrower DB-side key.
    """
    return (mention.form, mention.date_year, mention.region_hint, mention.context)


def _run_hallucination_rescue(
    fallback_client,
    chunk: str,
    chunk_index: int,
    primary_mentions: list[ToponymMention],
    log_warning: Callable[[str], None] | None,
) -> tuple[list[ToponymMention], ValidationCounters, bool]:
    """Run the fallback on a chunk where the primary succeeded but
    emitted >= threshold hallucinated forms (wyrd-z8mq). Returns a
    3-tuple ``(mentions, delta_counters, ok)``: the possibly-merged
    mentions, the fallback's per-call counter delta for the caller
    to fold, and a bool indicating whether the fallback actually
    delivered content.

    Three outcomes the bool distinguishes:

    1. Fallback returned non-empty mentions → union-merges into
       primary's by :func:`_hallucination_rescue_dedup_key` (primary
       canonical on collision; fallback adds anything primary didn't
       see). Returns (merged_list, rescue_delta, True). Caller folds
       rescue_delta via ``+=`` into its chunk accumulator.
    2. Fallback returned an empty admit list — call succeeded
       mechanically but produced no usable content. Could be a
       content-policy refusal, a schema-validation glitch, a
       model declining to extract, OR an all-hallucinated response
       where every form failed the word-boundary guard. Operationally
       equivalent to a failed rescue from the operator's
       perspective: primary's hallucinations weren't caught.
       Returns (primary_mentions, rescue_delta, False). The
       fallback's per-call counters STILL flow back — an
       all-hallucinated fallback emits non-zero
       ``hallucinations_dropped`` that carries real signal about
       fallback model quality (an empty admit list with non-zero
       dropped count means "fallback emitted garbage we filtered
       out", not "fallback returned nothing"). Caller folds.
    3. Fallback raised (transport / API / parse error). primary's
       mentions returned unchanged; the returned counters delta is
       zero so caller's fold is a no-op. Returns
       (primary_mentions, zero_delta, False). The orchestrator logs
       and continues; primary's good mentions still flow through to
       the output (the rescue is opportunistic).

    ``_PROGRAMMER_ERROR_EXCEPTIONS`` propagates as a traceback
    (consistent with the rest of the file's contract).
    """
    try:
        rescue_mentions, rescue_counters = extract_toponym_mentions_from_chunk(
            fallback_client, chunk
        )
    except _PROGRAMMER_ERROR_EXCEPTIONS:
        raise  # Our bug — let it surface as a traceback.
    except Exception as e:
        # Fallback failed on the rescue. Primary's good mentions
        # still flow through to the output — don't poison the chunk
        # on a transport error during what's only an opportunistic
        # extra pass. Log so the operator can see which rescues
        # missed (the gap between attempted vs succeeded counters
        # is the budget-impact signal; the log line tells WHICH
        # chunks).
        if log_warning is not None:
            log_warning(
                f"chunk {chunk_index} rescue fallback failed: "
                f"{type(e).__name__}: {e} — primary mentions retained"
            )
        # Return a shallow copy to match the success path's contract
        # (always-fresh list). The caller doesn't currently mutate
        # the return value (just `.extend()`s it into the report),
        # but consistent ownership prevents a future refactor from
        # silently aliasing the failure-path return into shared state
        # — Gemini wyrd-z8mq round 5 MEDIUM. Zero counter delta on
        # this path: caller's ``+=`` is a no-op, primary's counters
        # untouched.
        return list(primary_mentions), ValidationCounters(), False

    # Dedup against primary's keys AND against fallback-internal
    # duplicates (Gemini wyrd-z8mq round 3 MEDIUM): if the fallback
    # itself emits the same mention twice, only one survives the
    # merge. The running ``seen_keys`` set carries forward as we
    # walk rescue_mentions so a second emission of an already-added
    # key doesn't slip through.
    seen_keys = {_hallucination_rescue_dedup_key(m) for m in primary_mentions}
    added: list[ToponymMention] = []
    for m in rescue_mentions:
        key = _hallucination_rescue_dedup_key(m)
        if key in seen_keys:
            continue
        added.append(m)
        seen_keys.add(key)
    # ``rescue_mentions`` empty = fallback ran cleanly but admitted
    # no content. Operationally equivalent to a failed rescue: the
    # primary's hallucinations weren't actually caught. Return False
    # so chunks_hallucination_rescue_succeeded reflects "fallback
    # delivered output", not just "fallback didn't raise"
    # (silent-failure-hunter wyrd-z8mq round 2 HIGH). The rescue
    # delta still flows back regardless of admit-list shape: an
    # all-hallucinated fallback's hallucinations_dropped/years_clamped
    # carry real signal about fallback model quality and stay in the
    # aggregate when the caller folds.
    return list(primary_mentions) + added, rescue_counters, bool(rescue_mentions)


def _run_primary_failure_fallback(
    fallback_client,
    chunk: str,
    chunk_index: int,
    primary_error: str | None,
    log_warning: Callable[[str], None] | None,
) -> tuple[list[ToponymMention] | None, ValidationCounters, FailedChunk | None]:
    """Run the fallback on a chunk where the primary raised. Returns a
    3-tuple ``(mentions, counters, failure)``:

    * ``(mentions, counters, None)`` — fallback recovered. Caller folds
      ``counters`` into the report and bumps
      ``chunks_recovered_by_fallback``.
    * ``(None, counters, failure)`` — BOTH tiers failed. ``failure``
      records the chained ``primary=...; fallback=...`` error so
      operators can distinguish a transient one-tier hiccup from a
      genuinely-hard chunk where both quality tiers gave up. Caller
      bumps ``chunks_failed``, buffers ``failure`` in head/tail, and
      ``continue``s to the next chunk.

    Counters delta is produced by
    :func:`extract_toponym_mentions_from_chunk` (wyrd-oaq5 delta-return
    shape) and flows back to the caller on success. On failure the
    helper returns a fresh-empty :class:`ValidationCounters` so the
    caller's ``+=`` fold is a no-op (the caller ``continue``s anyway,
    but the empty delta keeps the contract symmetric across both
    return shapes).

    ``_PROGRAMMER_ERROR_EXCEPTIONS`` propagates as a traceback
    (consistent with the rest of the file's contract).
    """
    try:
        mentions, chunk_counters = extract_toponym_mentions_from_chunk(fallback_client, chunk)
    except _PROGRAMMER_ERROR_EXCEPTIONS:
        raise  # Our bug — let it surface as a traceback.
    except Exception as e:
        # BOTH tiers failed — record BOTH errors so operators see the
        # full failure chain (transient vs. genuinely-hard chunk are
        # distinguishable from the pair).
        fallback_error = f"{type(e).__name__}: {e}"
        # Sanitize for the same reason as the single-tier path above:
        # both primary_error and fallback_error can carry surrogate-
        # bearing fragments from LLM responses (wyrd-8wa4).
        failure = FailedChunk(
            index=chunk_index,
            error=_sanitize_for_utf8(f"primary={primary_error}; fallback={fallback_error}"),
            chunk_body=chunk,
        )
        if log_warning is not None:
            log_warning(f"chunk {chunk_index} both tiers failed: {fallback_error}")
        return None, ValidationCounters(), failure
    return mentions, chunk_counters, None


def _maybe_warn_oversize(chunk, i, oversize_threshold, log_warning):
    """Inline oversize-paragraph warning (matches the single-tier shape).
    Fires before either LLM call so the operator sees it interleaved with
    progress."""
    if log_warning is not None and len(chunk) > oversize_threshold:
        log_warning(
            f"chunk {i}: oversized paragraph "
            f"({len(chunk):,} chars > {int(oversize_threshold):,} threshold) — "
            f"LLM output may truncate; consider lowering --chunk-size"
        )


def _should_rescue(threshold: int | None, counters: ValidationCounters) -> bool:
    """Hallucination-rescue predicate (wyrd-z8mq): the primary succeeded but
    emitted >= threshold dropped forms. None or <= 0 disables the rescue."""
    return threshold is not None and threshold > 0 and counters.hallucinations_dropped >= threshold


def _record_failed_chunk(report, failure, head_failures, tail_failures, on_chunk_failed):
    """Record a both-tiers-failed chunk: bump the counter, store into the head
    buffer (first N) or the bounded tail deque, and fire the callback."""
    report.chunks_failed += 1
    if len(head_failures) < _FAILED_CHUNKS_HEAD:
        head_failures.append(failure)
    else:
        tail_failures.append(failure)
    if on_chunk_failed is not None:
        on_chunk_failed(failure)


def _extract_chunk_tiered(
    primary_client,
    fallback_client,
    chunk,
    i,
    report,
    hallucination_fallback_threshold,
    log_warning,
):
    """Primary -> fallback -> hallucination-rescue for ONE chunk. Returns
    ``(mentions, chunk_counters, failure)``: ``failure`` is a FailedChunk when
    BOTH tiers failed (caller records it + skips), else None. Updates the
    report's ``chunks_recovered_by_fallback`` / hallucination-rescue counters
    in place. Bare ``except Exception`` (not RuntimeError) so provider-SDK
    error classes (anthropic.APIError, httpx.HTTPError, TimeoutError) trigger
    the fallback rather than aborting the whole source."""
    chunk_counters = ValidationCounters()
    primary_error: str | None = None
    mentions: list[ToponymMention] | None = None
    try:
        mentions, chunk_counters = extract_toponym_mentions_from_chunk(primary_client, chunk)
    except _PROGRAMMER_ERROR_EXCEPTIONS:
        raise  # Our bug — let it surface as a traceback.
    except Exception as e:
        primary_error = f"{type(e).__name__}: {e}"
        if log_warning is not None:
            log_warning(f"chunk {i} primary failed: {primary_error} — retrying with fallback")
    if mentions is None:
        # Phase 2: primary failed — try fallback. See
        # :func:`_run_primary_failure_fallback` for counter/failure semantics.
        # ``primary_error`` is non-None on this branch: the only path that
        # leaves ``mentions`` as None is the primary's except block above.
        mentions, chunk_counters, failure = _run_primary_failure_fallback(
            fallback_client, chunk, i, primary_error, log_warning
        )
        if failure is not None:
            return (mentions, chunk_counters, failure)
        report.chunks_recovered_by_fallback += 1
    elif _should_rescue(hallucination_fallback_threshold, chunk_counters):
        # Phase 3: primary succeeded but hallucinated — try rescue. See
        # :func:`_run_hallucination_rescue` for the detailed contract +
        # counter-folding semantics (wyrd-z8mq).
        report.chunks_hallucination_rescue_attempted += 1
        if log_warning is not None:
            log_warning(
                f"chunk {i} primary hallucinated "
                f"{chunk_counters.hallucinations_dropped} forms — "
                f"running fallback for rescue"
            )
        mentions, rescue_delta, rescue_ok = _run_hallucination_rescue(
            fallback_client, chunk, i, mentions, log_warning
        )
        chunk_counters += rescue_delta
        if rescue_ok:
            report.chunks_hallucination_rescue_succeeded += 1
    return (mentions, chunk_counters, None)


def mine_toponym_mentions_tiered(
    primary_client,
    fallback_client,
    source_id: str,
    body: str,
    *,
    target_chunk_size: int = 20000,
    limit: int | None = None,
    hallucination_fallback_threshold: int | None = None,
    on_chunk_done: Callable[[int, int, int], None] | None = None,
    on_chunk_failed: Callable[[FailedChunk], None] | None = None,
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
    ``years_clamped``) on the failure-rescue path (primary-failed-
    then-fallback) reflect ONLY the tier that succeeded — the
    losing tier's counters never enter the chunk's accumulator
    because :func:`extract_toponym_mentions_from_chunk` returns its
    counters by value (wyrd-oaq5 delta-return shape) and the
    orchestrator only unpacks them on success. (The hallucination-
    rescue path below has different counter-folding semantics —
    both tiers contribute to the aggregate. See the rescue section.)

    The report shape matches :class:`MineToponymMentionsReport` so
    the CLI's summary line and the Phase 2b.1 ingester both work
    unchanged.

    **Hallucination-triggered rescue (wyrd-z8mq).** When
    ``hallucination_fallback_threshold`` is set (``None`` or any
    value ``<= 0`` = disabled), a chunk where the PRIMARY succeeded
    but emitted ``hallucinations_dropped >= threshold`` ALSO runs
    the fallback on the same chunk; the fallback's mentions are
    union-merged into the primary's by
    :func:`_hallucination_rescue_dedup_key` (primary wins on
    collision, fallback adds anything it saw that primary didn't).
    This preserves everything the primary saw correctly while
    letting the fallback catch what the primary fabricated.

    Two counters track the rescue: ``chunks_hallucination_rescue_attempted``
    increments per chunk where the fallback fired on the threshold
    (regardless of fallback outcome), and
    ``chunks_hallucination_rescue_succeeded`` increments only when
    the fallback returned a NON-EMPTY mentions list (actually
    delivered content, not just "didn't raise"). A gap
    (attempted > succeeded) surfaces both broken-fallback
    configurations (bad API key, network outage) and silent-empty
    responses (content-policy refusal, mis-shaped JSON producing
    ``{"mentions": []}``) that would otherwise look healthy on a
    single counter.

    Counter folding: the fallback's per-chunk counters
    (``hallucinations_dropped``, ``years_clamped``) fold into the
    primary's whenever the fallback call returns — including the
    all-hallucinated case where the admit list is empty but the
    dropped-form count carries real signal about fallback model
    quality. An ERRORED fallback (Outcome 3 of
    :func:`_run_hallucination_rescue`) returns a zero
    :class:`ValidationCounters`, so the orchestrator's ``+=`` fold
    is a no-op on that path. The primary's good mentions always
    flow through to the output (the rescue is opportunistic, never
    destructive).

    Designed for the gemma4 / Anthropic split observed 2026-05-16:
    gemma4 primary's smoke had 1 hallucinated form across 79
    extracted mentions (1.3%) on 3 chunks of Mawer 1920. With
    threshold=1 the rescue fires on any chunk whose primary
    emitted >=1 fabrication; full-book frequency depends on the
    text density and the model's per-chunk fabrication rate
    (calibrate against a full-source run before setting the
    threshold in production).

    ``on_chunk_failed`` (wyrd-3gbx) fires per BOTH-TIERS-FAILED
    chunk with the chained ``FailedChunk`` record, mirroring the
    single-tier :func:`mine_toponym_mentions` callback. Used by
    failure-streaming CLIs that need to flush failures to disk in
    real time rather than waiting for the in-memory head/tail buffer
    to materialize at end of run. Defaults to None (silent). The
    callback fires only on the both-tiers-failed path — primary-only
    failures that the fallback recovers are reported via
    ``chunks_recovered_by_fallback`` and don't invoke this callback.

    Exceptions raised by ``on_chunk_failed`` (and the sibling
    ``on_chunk_done`` / ``log_warning`` callbacks) propagate
    UNWRAPPED out of this function — the current source's report is
    lost, and the orchestrator's caller sees the original exception.
    The contract matches the single-tier orchestrator. Callers that
    need transient-I/O resilience (e.g. CLI failure-streaming on a
    flaky disk) should wrap their callback body in try/except so an
    individual chunk's I/O hiccup doesn't trash a multi-source run.
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
        _maybe_warn_oversize(chunk, i, oversize_threshold, log_warning)
        mentions, chunk_counters, failure = _extract_chunk_tiered(
            primary_client,
            fallback_client,
            chunk,
            i,
            report,
            hallucination_fallback_threshold,
            log_warning,
        )
        if failure is not None:
            _record_failed_chunk(report, failure, head_failures, tail_failures, on_chunk_failed)
            continue
        report.mentions.extend(mentions)
        report.chunks_processed += 1
        report.hallucinations_dropped += chunk_counters.hallucinations_dropped
        report.years_clamped += chunk_counters.years_clamped
        report.surrogates_sanitized += chunk_counters.surrogates_sanitized
        if on_chunk_done is not None:
            on_chunk_done(i + 1, total, len(report.mentions))
    report.failed_chunks.extend(head_failures)
    report.failed_chunks.extend(tail_failures)
    return report
