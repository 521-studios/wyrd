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
chunk text (substring match). Forms the LLM emits that don't appear
in the source are dropped silently. Same guard the etymology
extractor uses (DECISIONS.md D3).

Provider choice: the module accepts any client with the
``chat_json(system, user, schema=None) -> dict`` interface — same
contract as :class:`AnthropicClient` / Ollama / Gemini wrappers in
``llm_extractor.py``. The tiered-extraction pattern from the
project's wyrd-l0r infrastructure (Qwen bulk → Anthropic on the
residual) is implemented in the CLI orchestrator, not here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

# Year-citation range — same bounds as mine-attestations
# (_ATTESTED_YEAR_MIN_LOOKUP / _MAX_LOOKUP). 700-1700 covers
# post-Roman through pre-modern attestations. The LLM is
# instructed to clamp to this range; out-of-range years get
# treated as null by the validator.
_DATE_YEAR_MIN = 700
_DATE_YEAR_MAX = 1700


@dataclass(frozen=True)
class ToponymMention:
    """One place-name mention extracted from a scholar source body.

    * ``form`` — verbatim place name as it appears in the prose
      (preserves capitalization, diacritics, internal punctuation).
      Validated to appear in the source chunk before construction;
      the assembler drops mentions whose form doesn't substring-
      match (anti-hallucination guard).
    * ``date_year`` — a year in [700, 1700] from immediately
      surrounding context, if any. Out-of-range or unparseable
      values resolve to None. None means "the LLM didn't find a
      nearby year"; downstream code can treat None as an undated
      attestation (vs. a dated-attestation Phase 1 would emit).
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


def _validated_mentions(raw: list[dict], chunk: str) -> list[ToponymMention]:
    """Convert LLM-emitted dicts to ``ToponymMention`` instances,
    dropping any whose form doesn't substring-match the chunk
    (anti-hallucination).

    Form-in-chunk is the bare-minimum guard. Stronger guards (form
    starts with capital, isn't a known blacklist word like Domesday,
    etc.) are downstream — the operator-review tool / the
    form→toponym resolver can apply those. Here we only filter the
    structural anti-hallucination case.
    """
    out: list[ToponymMention] = []
    for m in raw:
        if not isinstance(m, dict):
            continue
        form = (m.get("form") or "").strip()
        if not form:
            continue
        if form not in chunk:
            # Hallucinated form — drop silently.
            continue
        # Clamp date_year to the expected range; out-of-range → None.
        date_year = m.get("date_year")
        if isinstance(date_year, int) and not (_DATE_YEAR_MIN <= date_year <= _DATE_YEAR_MAX):
            date_year = None
        elif not isinstance(date_year, int):
            date_year = None
        # region_hint: keep as-is when non-empty string, else None.
        region_hint = m.get("region_hint")
        if isinstance(region_hint, str):
            region_hint = region_hint.strip() or None
        else:
            region_hint = None
        # context: trim whitespace; collapse newlines.
        context = (m.get("context") or "").strip().replace("\n", " ")
        if not context:
            # Synthesize a context window from the chunk if the LLM
            # left it empty — gives operators something to inspect.
            idx = chunk.find(form)
            if idx >= 0:
                start = max(0, idx - 40)
                end = min(len(chunk), idx + len(form) + 40)
                context = chunk[start:end].replace("\n", " ").strip()
        out.append(
            ToponymMention(
                form=form,
                date_year=date_year,
                region_hint=region_hint,
                context=context,
            )
        )
    return out


def extract_toponym_mentions_from_chunk(client, chunk: str) -> list[ToponymMention]:
    """Run one LLM call against ``chunk`` and return validated
    mentions. ``client`` must have a ``chat_json(system, user,
    schema=None) -> dict`` method — same contract as
    :class:`AnthropicClient`, :class:`OllamaClient`,
    :class:`GeminiClient` in the existing extractor modules.

    Network / JSON-parse errors propagate as ``RuntimeError`` — the
    caller (per-source orchestrator) decides whether to skip the
    chunk and continue or abort the run.
    """
    user = USER_TEMPLATE.format(chunk=chunk)
    response = client.chat_json(SYSTEM_PROMPT, user, schema=RESPONSE_SCHEMA)
    if not isinstance(response, dict):
        return []
    raw = response.get("mentions")
    if not isinstance(raw, list):
        return []
    return _validated_mentions(raw, chunk)


def chunk_source_body(body: str, *, target_chunk_size: int = 20000) -> list[str]:
    """Split a source body into chunks of roughly ``target_chunk_size``
    characters, snapping to paragraph boundaries (blank lines).
    Returns the list of non-empty chunks in order.

    Snap-to-paragraph keeps the LLM's view of each chunk coherent —
    splitting mid-paragraph would orphan year/region context that's
    part of the same scholar entry.

    For a 600KB source body and ``target_chunk_size=20000``, this
    yields ~30 chunks. Anthropic at ~3s per chunk = ~90s per source.
    """
    if not body.strip():
        return []
    # Split on blank-line boundaries (one or more blank lines).
    paragraphs = body.split("\n\n")
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        # If adding this paragraph would exceed the target AND we
        # already have content, flush the current chunk.
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


@dataclass
class MineToponymMentionsReport:
    """Aggregate result of mining one source body for mentions."""

    source_id: str
    chunks_processed: int = 0
    chunks_failed: int = 0
    mentions: list[ToponymMention] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.mentions is None:
            self.mentions = []


def mine_toponym_mentions(
    client,
    source_id: str,
    body: str,
    *,
    target_chunk_size: int = 20000,
    on_chunk_done: callable = None,  # type: ignore[type-arg]
) -> MineToponymMentionsReport:
    """Walk a source body chunk-by-chunk, accumulating extracted
    mentions. Returns the aggregate report.

    ``on_chunk_done`` is an optional progress callback called with
    ``(chunks_done, total_chunks, mentions_so_far)`` after each chunk
    succeeds — lets the CLI emit per-chunk progress to stderr without
    forcing this module to know about click.
    """
    report = MineToponymMentionsReport(source_id=source_id)
    chunks = chunk_source_body(body, target_chunk_size=target_chunk_size)
    total = len(chunks)
    for i, chunk in enumerate(chunks):
        try:
            mentions = extract_toponym_mentions_from_chunk(client, chunk)
        except (RuntimeError, json.JSONDecodeError):
            # Per-chunk failure (transport, JSON-parse, schema mismatch)
            # is logged + skipped — don't poison the whole source's
            # output. Operator can re-run the source later or pivot to
            # a different provider for the failed chunks.
            report.chunks_failed += 1
            continue
        report.mentions.extend(mentions)
        report.chunks_processed += 1
        if on_chunk_done is not None:
            on_chunk_done(i + 1, total, len(report.mentions))
    return report
