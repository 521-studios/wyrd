"""LLM-assisted etymology extraction via Ollama.

Wraps an Ollama endpoint with a tight JSON-schema-constrained prompt that
turns one Skeat-style entry body into a structured ParsedEntry. Every claim
the model makes is validated against the source text before being accepted —
forms and glosses must appear in the body, otherwise the entry is rejected.

The extractor produces the same ParsedEntry shape as the regex parser, so
the existing ingester (`ingest_parsed_entries`) can persist its output
directly. Successful extractions are flagged with a `notes` line so we can
distinguish LLM-derived rows from regex-derived ones in the lexicon.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Literal

from wyrd.generators.kenning.parsers.skeat import (
    SOURCE_QUOTE_BUDGET,
    ParsedElement,
    ParsedEntry,
)

# --- config ---------------------------------------------------------------

DEFAULT_OLLAMA_URL = os.environ.get("WYRD_OLLAMA_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.environ.get("WYRD_OLLAMA_MODEL", "qwen3.5:35b-a3b")

# Map the language strings the model is allowed to emit to the lexicon's
# canonical codes. Keep tight: the prompt enumerates these.
_ALLOWED_LANGUAGES = {
    "old-english",
    "old-norse",
    "norman-french",
    "celtic",
    "latin",
    "germanic",
    "greek",
    "biblical",
    "modern-english",
}

_ALLOWED_POSITIONS = {"pre", "inner", "post"}


# SOURCE_QUOTE_BUDGET (the source_quote cap for ParsedEntry rows the SPA
# citation view consumes) lives in parsers.skeat alongside ParsedEntry —
# the regex parser (parsers.skeat._shorten) and this LLM extractor share
# one constant. The reverse import direction would be a cycle.
# Imported above; kept aliased here so existing readers see the symbol
# without chasing the hop. wyrd-9v1.


# Response schema (Ollama's `format` field accepts a JSON Schema dict).
# The model must produce exactly this shape; Ollama enforces it via grammar.
# Year-citation range for attested_forms (D5-1, wyrd-3ux). 100-1700 covers
# Roman empire (Caesar's invasions ~55BC excluded; first-century AD
# Gallo-Roman onward) through Restoration. The original 800 floor (set
# in PR #27) was sized for British/Norman-period sources where Domesday
# 1086 is the typical earliest dated attestation, but it cut legitimate
# Roman-era charters and Gaulish gentilice formations on Romance/Celtic-
# substrate sources like d'Arbois 1890 — wyrd-z56 widened the floor to
# 100. Years outside this range are almost always (a) publication-year
# noise (1880-1928) or (b) folio / page numbers misread as years.
_ATTESTED_YEAR_MIN = 100
_ATTESTED_YEAR_MAX = 1700


RESPONSE_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "found": {"type": "boolean"},
        "historical_form": {"type": ["string", "null"]},
        "elements": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "form": {"type": "string", "minLength": 1},
                    "language": {
                        "type": "string",
                        "enum": sorted(_ALLOWED_LANGUAGES),
                    },
                    "position": {
                        "type": "string",
                        "enum": sorted(_ALLOWED_POSITIONS),
                    },
                    "gloss": {"type": ["string", "null"]},
                    "inflection": {"type": ["string", "null"]},
                },
                "required": ["form", "language", "position", "gloss", "inflection"],
            },
        },
        # Per-form attestation years from the body text (D5-1 / wyrd-3ux).
        # When the scholar cites dated historical forms (e.g.
        # "Aburwick, 1333 ; Abberwyke, 1346"), each (form, year) pair lands
        # here. Empty list when the body has no year citations — the model
        # should NOT invent dates, only extract ones it sees.
        "attested_forms": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "form": {"type": "string", "minLength": 1},
                    "year": {
                        "type": "integer",
                        "minimum": _ATTESTED_YEAR_MIN,
                        "maximum": _ATTESTED_YEAR_MAX,
                    },
                },
                "required": ["form", "year"],
            },
        },
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "notes": {"type": ["string", "null"]},
    },
    "required": [
        "found",
        "historical_form",
        "elements",
        "attested_forms",
        "confidence",
        "notes",
    ],
}


SYSTEM_PROMPT = """\
You extract structured place-name etymology from prose written by 19th- and
early-20th-century scholars (Skeat, Mawer, Ekwall, Johnston, etc.). The user
gives you a single entry: the modern place name and the prose paragraph
explaining its origin.

Output a JSON object with EXACTLY these fields and no others:

  {
    "found": true | false,
    "historical_form": "<reconstructed compound, hyphenated at boundaries, or null>",
    "elements": [
      {
        "form":       "<the etymon — e.g. 'bere', 'tun', 'Hædan'>",
        "language":   "<one of: old-english, old-norse, norman-french, celtic, latin, germanic, greek, biblical, modern-english>",
        "position":   "<one of: pre, inner, post>",
        "gloss":      "<English meaning, e.g. 'barley' — or null>",
        "inflection": "<e.g. 'genitive', 'dative', 'plural' — or null>"
      },
      ...
    ],
    "attested_forms": [
      { "form": "<historical spelling cited in the body>", "year": <integer 100-1700> },
      ...
    ],
    "confidence": "<one of: high, medium, low>",
    "notes": "<short hedge note, e.g. 'Skeat marks this as guesswork' — or null>"
  }

Field-name rules — these are STRICT:
  - The etymon string field is named "form". NOT "morpheme", NOT "etymon", NOT
    "word". Use "form".
  - The English-meaning field is named "gloss". NOT "meaning", NOT
    "translation". Use "gloss".

Content rules:

1. Extract the HISTORICAL morphemes the scholar identifies — Old English
   "bere", "tun", "Hædan"; Old Norse "býr"; Celtic "aber"; Latin "castrum".
   NOT the modern surface form.

2. Each "form" you emit MUST appear (case-insensitive, allowing OCR garble
   for diacritics) somewhere in the entry text. Do not invent etymons.

3. Position: "pre" = starts the compound (Hædan- in Hædan-ham); "post" =
   ends it (-ham); "inner" = middle of three-part compound.

4. Inflection only when the scholar names one explicitly. Otherwise null.

5. Distinguish two refusal cases CAREFULLY:

   (a) Set found=false, elements=[] ONLY when the entry has NO etymological
       claim at all — a pure attestation list with no analysis, a cross-
       reference ("see Stannington supra"), or the scholar explicitly says
       "meaning unknown" / "etymology obscure" / "requires no explanation".

   (b) If the scholar offers a HEDGED etymology — any of "perhaps",
       "possibly", "may be from", "I am inclined to think", "tentatively",
       "the prefix may represent", "uncertain but probably" — DO extract
       the proposed etymology. Set confidence="low" and put the scholar's
       hedge phrase in notes (e.g., "Skeat: 'I only make a guess'").

   Hedged data is still useful data. We want low-confidence rows; we don't
   want to lose proposed etymologies just because the scholar wasn't sure.

6. attested_forms — extract dated historical-form citations:

   When the body cites a place-name in a specific historical form WITH a year
   (e.g. "Aburwick, 1333 ; Abberwyke, 1346 ; Aburwick, 1428") emit one
   {form, year} entry per dated citation. The scholar pairs forms with
   years from charters, Domesday Book, Pipe Rolls, Subsidy Rolls,
   inquisitions, and similar. These pairs are GOLD — they tell us when
   the form was attested in the historical record.

   Rules for attested_forms:
     - Each "form" must appear in the body (same form-in-body rule as
       elements).
     - Each "year" must be a JSON INTEGER (NOT a string, NOT a float)
       in the range 100-1700 — Roman empire through Restoration.
       Years outside that range are almost always publication years,
       folio numbers, or page numbers. DO NOT include them.
       Right: {"form": "Aburwick", "year": 1333}
       Right: {"form": "Lugudunum", "year": 580}
       Wrong: {"form": "Aburwick", "year": "1333"}      ← string, not int
       Wrong: {"form": "Aburwick", "year": "1333 (?)"}  ← string with hedge
       Wrong: {"form": "Aburwick", "year": 1333.0}      ← float, not int
       If the body's year is uncertain ("ca. 1333", "1333?"), still emit
       the bare integer 1333 — the uncertainty stays in the entry's
       confidence/notes fields.
     - Skip non-attestation year mentions: publication-year citations
       ("Mawer 1920"), document-name dates referring to the document
       type rather than to a dated form of the place-name (see examples
       below), and abbreviation expansions ("DB" for Domesday Book).
     - Multiple forms-and-years in one entry — emit one object per
       {form, year} pair.
     - Empty list when no dated forms are cited. DO NOT INVENT dates from
       general phrases like "in the 12th century" — only extract explicit
       year citations.

   Examples:
     Body: "Aburwick, 1333 ; Abberwyke, 1346 ; Aburwick, 1428"
       → attested_forms: [
           {"form": "Aburwick", "year": 1333},
           {"form": "Abberwyke", "year": 1346},
           {"form": "Aburwick", "year": 1428}
         ]

     Body: "Aldenwic, 1226 ; Aldwick, 1268"
       → attested_forms: [
           {"form": "Aldenwic", "year": 1226},
           {"form": "Aldwick", "year": 1268}
         ]

     Body: "Allendale appears as Alanedale (1265) and Aldenedall (1428)."
       → attested_forms: [
           {"form": "Alanedale", "year": 1265},
           {"form": "Aldenedall", "year": 1428}
         ]

     Body: "From OE bere + tun. See Skeat (1901) for discussion."
       → attested_forms: []  (1901 is publication year, not attestation)

     Body: "Recorded in the Pipe Roll, 1199."
       → attested_forms: []
       (The 1199 here dates the document, not a specific form of the
       place-name. There's no spelling cited alongside the year.)

     Body: "Bedeham (Pipe Roll 1199) ; Bedham (Subsidy Roll, 1296)."
       → attested_forms: [
           {"form": "Bedeham", "year": 1199},
           {"form": "Bedham", "year": 1296}
         ]
       (Document names appear, but each is paired with a specific
       attested spelling — capture those.)

7. Output ONLY the JSON. No markdown, no prose, no thinking.
"""

USER_TEMPLATE = """\
Modern name: {toponym}
Section suffix (post-element hint, may be null): {suffix_hint}

Entry text:
\"\"\"
{body}
\"\"\"
"""


# --- transport helpers ----------------------------------------------------


def parse_transport_json(
    payload: str,
    *,
    provider: Literal["Anthropic", "Gemini", "Ollama"],
    kind: Literal["envelope", "content"],
) -> dict:
    """Parse JSON from a transport payload, normalizing JSONDecodeError to
    RuntimeError with a provider-specific diagnostic message.

    ``kind`` distinguishes the layer being parsed:
      * ``"envelope"`` — the outer HTTP body returned by the provider.
        Error verb: ``returned non-JSON envelope`` (transport-level concern:
        CDN/proxy HTML error page, truncated body on a dropped connection).
      * ``"content"`` — the inner JSON the model produced inside the
        envelope. Error verb: ``produced non-JSON content`` (model-level
        concern: model emitted malformed JSON despite the schema/prompt).

    Why route JSONDecodeError → RuntimeError (wyrd-pe4g cross-module
    invariant): the chunk loop in ``extractors/toponym_mentions.py`` lists
    ``ValueError`` in ``_PROGRAMMER_ERROR_EXCEPTIONS`` so a typo'd
    ``schema_dialect`` surfaces as a Python traceback rather than 100%
    chunks_failed. ``json.JSONDecodeError`` is a ``ValueError`` subclass —
    without this wrap, a 2xx response with a malformed body would propagate
    through the re-raise and abort the whole multi-source run instead of
    bucketing the chunk into chunks_failed where it belongs.

    All three transport clients (Anthropic, Gemini, Ollama) call this at
    both the outer envelope parse and the inner content parse — six sites,
    one invariant.
    """
    try:
        return json.loads(payload)
    except json.JSONDecodeError as e:
        if kind == "envelope":
            verb = "returned non-JSON envelope"
        else:
            verb = "produced non-JSON content"
        raise RuntimeError(f"{provider} {verb}: {payload[:500]}") from e


# --- HTTP client ----------------------------------------------------------


@dataclass
class OllamaClient:
    """Minimal Ollama chat client. No retries, single-shot per call.

    ``num_ctx`` and ``keep_alive`` are sent explicitly on every request
    because Ollama's per-model defaults are environment-dependent
    (Ollama version, env vars, prior requests) and silently drift —
    wyrd-dyf8 documents a concrete incident where a macbook reboot
    caused Ollama to reload gemma4:26b at its absolute-max 262K
    context, halving mining throughput. Sending these values per-
    request makes the wyrd client immune to that class of config drift.
    """

    base_url: str = DEFAULT_OLLAMA_URL
    model: str = DEFAULT_OLLAMA_MODEL
    timeout_s: float = 120.0
    temperature: float = 0.0
    # Right-sized for the production miner (staged cascade), whose
    # 3000-char chunks (~750 tokens) + ~500-token system prompt +
    # response stay well under 8K. The single-tier
    # ``mine-toponym-mentions`` CLI defaults to 20000-char chunks
    # (~5000 tokens) — dense responses could theoretically truncate
    # at 8192. That caller can override ``num_ctx`` at construction;
    # making it CLI-configurable per-caller is a deferred improvement
    # (see wyrd-qxmr for the empirical trade-off).
    #
    # wyrd-qxmr: PR #327 originally bumped this to 16384 to cover
    # the single-tier hypothetical. Empirical 2026-05-22 measurement
    # showed that increased gemma4:26b chunk-time from ~33s (8K) to
    # ~102s (16K) — a 2-3x throughput regression with no production
    # benefit (the single-tier path isn't used for corpus mining).
    # 8192 is the right default for the production hot path.
    num_ctx: int = 8192
    # Pin the model in memory across requests so an idle period
    # between chunks doesn't unload the model — a subsequent chunk
    # request would then have to reload it from disk (significant
    # latency for multi-GB models like gemma4:26b), often exceeding
    # ``timeout_s`` and being recorded as a chunk failure. -1 means
    # "keep loaded until Ollama restarts" — the right shape for
    # batch mining where we want the model warm for the duration
    # of the run.
    keep_alive: int | str = -1

    def chat_json(self, system: str, user: str, schema: dict) -> dict:
        """POST /api/chat with format='json', return parsed JSON content.

        Raises RuntimeError with a useful message on transport / parse error.

        wyrd-sj50: ``schema`` is accepted for API compatibility but NOT
        sent to the server. Ollama 0.24+ on the macbook hangs
        indefinitely when ``format`` is a JSON-Schema dict (probe results
        2026-05-22: schema-dict request never returns; equivalent
        ``'format': 'json'`` request returns in 500ms). The legacy 'json'
        mode tells the model to output JSON without grammar-constrained
        decoding. Caller-side validation (``_validated_mentions``)
        rejects malformed responses anyway, so losing server-side schema
        enforcement is safe. If Ollama fixes the schema-grammar
        regression upstream, this can revert to ``'format': schema``.
        """
        # ``schema`` deliberately unused — see docstring (wyrd-sj50).
        del schema
        url = f"{self.base_url.rstrip('/')}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "format": "json",
            # Qwen3.5 et al. are reasoning models with a separate `thinking`
            # output channel. With think enabled they consume the token
            # budget on internal chain-of-thought before producing any
            # `content`. We need the structured JSON content immediately.
            "think": False,
            # keep_alive is a top-level field on /api/chat (NOT inside
            # options). Misplacing it under options is silently
            # ignored by the Ollama server.
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = resp.read().decode("utf-8")
        except (TimeoutError, urllib.error.URLError) as e:
            # TimeoutError can be raised directly (Python 3.10+ aliased
            # socket.timeout to TimeoutError) and isn't a subclass of
            # URLError, so it needs explicit handling.
            raise RuntimeError(f"Ollama unreachable at {url}: {e}") from e
        except UnicodeDecodeError as e:
            # Non-UTF-8 body on a 2xx response — corrupted upstream,
            # proxy injecting binary content, etc. UnicodeDecodeError is
            # a ValueError subclass; without this wrap it would propagate
            # through the chunk loop's _PROGRAMMER_ERROR_EXCEPTIONS
            # re-raise (wyrd-pe4g) and abort the whole multi-source run.
            raise RuntimeError(f"Ollama returned non-UTF-8 body: {e}") from e

        envelope = parse_transport_json(body, provider="Ollama", kind="envelope")
        content = envelope.get("message", {}).get("content")
        if not content:
            raise RuntimeError(f"Ollama returned empty content. Raw: {body[:500]}")
        return parse_transport_json(content, provider="Ollama", kind="content")


# --- validation -----------------------------------------------------------


@dataclass
class ValidationFailure:
    reason: str
    detail: str = ""


def _normalize_for_match(s: str) -> str:
    """Lowercase + lossy-normalize for substring matching across OCR/orthography
    variants.

    Generous enough to match across:
      - OE characters (Hædan vs Haedan)
      - OCR mangles for those characters (Hcsdan, Hcedan)
      - Long-vowel diacritics (tūn vs tun, lēah vs leah)
      - Notational parens (Bacc(a) vs Bacca)
      - Plus collapsed whitespace
    """
    s = s.lower()
    # Ligature characters → ASCII pairs.
    s = s.replace("æ", "ae").replace("ð", "d").replace("þ", "th").replace("œ", "oe")
    # OCR mangles for ligature chars.
    s = s.replace("cs", "ae").replace("ce", "ae")
    # Long-vowel macrons → bare vowel. The model commonly emits canonical
    # OE forms with macrons; the OCR'd body has the bare letter (or vice
    # versa). Normalizing both directions makes the substring check work.
    for marked, bare in (
        ("ā", "a"),
        ("ē", "e"),
        ("ī", "i"),
        ("ō", "o"),
        ("ū", "u"),
        ("ȳ", "y"),
        ("ǣ", "ae"),
        ("ȳ", "y"),
        ("Á", "a"),
        ("É", "e"),
        ("Í", "i"),
        ("Ó", "o"),
        ("Ú", "u"),
        ("á", "a"),
        ("é", "e"),
        ("í", "i"),
        ("ó", "o"),
        ("ú", "u"),
    ):
        s = s.replace(marked, bare)
    # Strip notational parentheses ("Bacc(a)" → "bacca").
    s = s.replace("(", "").replace(")", "")
    # Strip leading/trailing dashes (position markers, not part of form).
    s = s.strip("-")
    s = re.sub(r"\s+", " ", s)
    return s


def _validate_attested_forms(response: dict, norm_body: str) -> list[ValidationFailure]:
    """Validate the D5-1 attested_forms field. Empty / missing → no failures.

    Run regardless of the ``found`` flag — a body can carry dated citations
    without an etymological claim (a pure attestation list with year stamps
    and no analysis), so the anti-hallucination guard on cited forms must
    apply even when the model declines the etymology. See the call site in
    ``validate_response``.
    """
    failures: list[ValidationFailure] = []

    # Tolerate absence: legacy stored fixtures and transport-error responses
    # synthesized by ``transport_error_result`` don't carry the field. The
    # live schema marks attested_forms required, so production responses
    # always include it (possibly empty) — this branch is for historical /
    # synthetic responses, not for live model output.
    if "attested_forms" not in response:
        return failures

    attested = response.get("attested_forms") or []
    if not isinstance(attested, list):
        failures.append(
            ValidationFailure("attested_forms_not_list", f"got {type(attested).__name__}")
        )
        return failures

    for j, entry in enumerate(attested):
        if not isinstance(entry, dict):
            failures.append(
                ValidationFailure(
                    "attested_form_not_object",
                    f"attested_forms[{j}] is {type(entry).__name__}",
                )
            )
            continue
        # Strip dashes the same way as element forms; otherwise the model
        # could bypass the length floor with "-A-" or similar. ``str()``
        # wrap defends against truthy non-strings — the schema requires
        # ``form`` to be a string but Gemini's schema enforcement has been
        # unreliable across versions and Anthropic uses prompt-only
        # enforcement, so it's worth not crashing on a stray bool / number.
        a_form = str(entry.get("form") or "").strip().strip("-")
        if len(a_form) < 2 or _normalize_for_match(a_form) not in norm_body:
            failures.append(
                ValidationFailure(
                    "attested_form_not_in_body",
                    f"attested_forms[{j}].form={entry.get('form')!r}",
                )
            )
        year = entry.get("year")
        # bool is a subclass of int in Python; reject explicitly.
        if not isinstance(year, int) or isinstance(year, bool):
            failures.append(
                ValidationFailure(
                    "attested_year_not_int",
                    f"attested_forms[{j}].year={year!r}",
                )
            )
        elif not (_ATTESTED_YEAR_MIN <= year <= _ATTESTED_YEAR_MAX):
            failures.append(
                ValidationFailure(
                    "attested_year_out_of_range",
                    f"attested_forms[{j}].year={year}",
                )
            )

    return failures


def validate_response(
    response: dict, body: str, allowed_languages: set[str] = _ALLOWED_LANGUAGES
) -> list[ValidationFailure]:
    """Run guardrails over a model response. Returns a list of failures
    (empty = clean).

    Checks:
    - Schema-shaped (already enforced by Ollama, but defensive).
    - Every element form appears in the body. Glosses are NOT validated
      because they're allowed to paraphrase the source.
    - Languages and positions are from the allowed sets.
    - Element count is sane (1–4).
    - attested_forms (D5-1) is well-shaped: each entry has a form that
      appears in body, with a year integer in the 100-1700 range.

    Element checks are skipped when ``found=false`` (a clean decline).
    The attested_forms check runs even on declines — see
    ``_validate_attested_forms`` for the rationale.
    """
    failures: list[ValidationFailure] = []

    if not isinstance(response, dict):
        failures.append(ValidationFailure("not_object"))
        return failures

    norm_body = _normalize_for_match(body)

    # attested_forms must be validated regardless of found state — a model
    # can emit dated citations on a "no etymology found" entry.
    failures.extend(_validate_attested_forms(response, norm_body))

    if not response.get("found", False):
        # Model declined the etymology; element-side checks don't apply.
        return failures

    elements = response.get("elements") or []
    if not (1 <= len(elements) <= 4):
        failures.append(
            ValidationFailure("element_count_out_of_range", f"got {len(elements)} elements")
        )

    for i, el in enumerate(elements):
        # ``str()`` coercion + ``.strip("-")`` matches the attested_forms
        # extraction (see ``_validate_attested_forms``) — same defensive
        # rationale (Gemini schema enforcement is unreliable across
        # versions; Anthropic uses prompt-only enforcement; better to
        # land as form-not-in-body than crash on a stray non-string).
        form = str(el.get("form") or "").strip().strip("-")
        if len(form) < 2 or _normalize_for_match(form) not in norm_body:
            failures.append(
                ValidationFailure("form_not_in_body", f"element[{i}].form={el.get('form')!r}")
            )
        # A digit in a linguistic form is an OCR misread, not a real form: no
        # OE/ME/Latin/Norse/reconstructed headword carries an Arabic numeral
        # (macrons / asterisks / æþð yes, 0-9 never). The anti-hallucination
        # check above PASSES such garbage when the OCR noise is in the body
        # ("6j^", "*h8ema", "0yrr"), so it would land as a corrupt etymon
        # canonical_form (wyrd-tru5). Reject the entry instead.
        if any(c.isdigit() for c in form):
            failures.append(
                ValidationFailure("form_has_digit", f"element[{i}].form={el.get('form')!r}")
            )
        lang = el.get("language")
        if lang not in allowed_languages:
            failures.append(ValidationFailure("bad_language", f"element[{i}].language={lang!r}"))
        pos = el.get("position")
        if pos not in _ALLOWED_POSITIONS:
            failures.append(ValidationFailure("bad_position", f"element[{i}].position={pos!r}"))
        # Gloss is allowed to be a paraphrase. The form-in-body check is
        # the real anti-hallucination guard; gloss is informational and
        # tracking it strictly against the source produces too many false
        # rejections (the model commonly compresses Skeat's "an enclosure"
        # to "enclosure", which is fine).

    return failures


# --- extraction orchestrator ---------------------------------------------


@dataclass
class LLMResult:
    """Result of running the extractor against one entry body."""

    entry: ParsedEntry | None
    raw_response: dict
    failures: list[ValidationFailure]
    accepted: bool


def transport_error_result(message: str) -> LLMResult:
    """Wrap a transport-level failure (HTTP timeout, JSON-parse error, etc.)
    in an LLMResult so the caller can treat all failures uniformly."""
    return LLMResult(
        entry=None,
        raw_response={"error": message},
        failures=[ValidationFailure("transport_error", message)],
        accepted=False,
    )


def assemble_extraction_result(
    response: dict,
    *,
    body: str,
    toponym: str,
    suffix_hint: str | None,
    notes_prefix: str,
) -> LLMResult:
    """Convert a validated LLM response into an LLMResult.

    Shared across Ollama / Gemini / Anthropic extractors — the only
    per-provider input is ``notes_prefix`` (e.g. ``extracted_by:gemini:<model>``)
    so D3 form-in-body validation and ParsedEntry assembly stay in one
    place. Three copies of this code path was a known drift risk —
    DECISIONS.md D3 calls validate_response load-bearing, so any divergence
    between providers' assemble paths could silently weaken it.
    """
    failures = validate_response(response, body)

    if not response.get("found", False):
        # Clean "no etymology in this entry" — record toponym as low-conf.
        # Per wyrd-6hd: prefix with the provider attribution so a later
        # query can distinguish 'Gemini declined this' from 'Qwen declined'
        # from 'baseline regex produced no body'. The body shrinks by the
        # prefix's width but the outer SOURCE_QUOTE_BUDGET cap is unchanged
        # — same shape as the accepted-path source_quote composition.
        decline_quote = f"{notes_prefix} | {body}"[:SOURCE_QUOTE_BUDGET]
        return LLMResult(
            entry=ParsedEntry(
                toponym=toponym,
                section_suffix=suffix_hint,
                historical_form=None,
                elements=[],
                confidence="low",
                source_quote=decline_quote,
            ),
            raw_response=response,
            failures=failures,
            accepted=True,
        )

    if failures:
        return LLMResult(entry=None, raw_response=response, failures=failures, accepted=False)

    elements = []
    for el in response.get("elements", []):
        elements.append(
            ParsedElement(
                form=el["form"].strip().lower(),
                language=el["language"],
                position=el["position"],
                gloss=(el.get("gloss") or "").strip() or None,
                inflection=(el.get("inflection") or "").strip() or None,
            )
        )

    confidence = response.get("confidence", "medium")
    notes = response.get("notes")
    combined_notes = f"{notes_prefix}; {notes}" if notes else notes_prefix

    return LLMResult(
        entry=ParsedEntry(
            toponym=toponym,
            section_suffix=suffix_hint,
            historical_form=response.get("historical_form"),
            elements=elements,
            confidence=confidence,
            source_quote=(combined_notes + " | " + body)[:SOURCE_QUOTE_BUDGET],
        ),
        raw_response=response,
        failures=[],
        accepted=True,
    )


def extract_one(
    client: OllamaClient,
    *,
    toponym: str,
    body: str,
    suffix_hint: str | None,
) -> LLMResult:
    """Run the Ollama LLM on one entry. Thin wrapper over
    ``assemble_extraction_result`` — Ollama is the only client whose
    chat_json takes the response schema explicitly (Gemini/Anthropic both
    rely on prompt-only schema enforcement)."""
    user = USER_TEMPLATE.format(toponym=toponym, suffix_hint=suffix_hint or "(none)", body=body)
    try:
        response = client.chat_json(SYSTEM_PROMPT, user, RESPONSE_SCHEMA)
    except RuntimeError as e:
        return transport_error_result(str(e))
    return assemble_extraction_result(
        response,
        body=body,
        toponym=toponym,
        suffix_hint=suffix_hint,
        notes_prefix=f"extracted_by:llm:{client.model}",
    )
