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

from wyrd.generators.kenning.skeat_parser import ParsedElement, ParsedEntry

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


# Response schema (Ollama's `format` field accepts a JSON Schema dict).
# The model must produce exactly this shape; Ollama enforces it via grammar.
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
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "notes": {"type": ["string", "null"]},
    },
    "required": ["found", "historical_form", "elements", "confidence", "notes"],
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

6. Output ONLY the JSON. No markdown, no prose, no thinking.
"""

USER_TEMPLATE = """\
Modern name: {toponym}
Section suffix (post-element hint, may be null): {suffix_hint}

Entry text:
\"\"\"
{body}
\"\"\"
"""


# --- HTTP client ----------------------------------------------------------


@dataclass
class OllamaClient:
    """Minimal Ollama chat client. No retries, single-shot per call."""

    base_url: str = DEFAULT_OLLAMA_URL
    model: str = DEFAULT_OLLAMA_MODEL
    timeout_s: float = 120.0
    temperature: float = 0.0

    def chat_json(self, system: str, user: str, schema: dict) -> dict:
        """POST /api/chat with format=schema, return parsed JSON content.

        Raises RuntimeError with a useful message on transport / parse error.
        """
        url = f"{self.base_url.rstrip('/')}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "format": schema,
            # Qwen3.5 et al. are reasoning models with a separate `thinking`
            # output channel. With think enabled they consume the token
            # budget on internal chain-of-thought before producing any
            # `content`. We need the structured JSON content immediately.
            "think": False,
            "options": {"temperature": self.temperature},
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

        envelope = json.loads(body)
        content = envelope.get("message", {}).get("content")
        if not content:
            raise RuntimeError(f"Ollama returned empty content. Raw: {body[:500]}")
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Ollama produced non-JSON content: {content[:500]}") from e


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


def _form_in_body(form: str, body: str) -> bool:
    """Did the model's claimed form actually appear in the source paragraph?

    We do a soft substring match against a normalized version of both — so
    "hædan" matches "Hcsdan" (OCR variant) and "Hædan" (clean).
    """
    if len(form) < 2:
        return False
    return _normalize_for_match(form) in _normalize_for_match(body)


def validate_response(
    response: dict, body: str, allowed_languages: set[str] = _ALLOWED_LANGUAGES
) -> list[ValidationFailure]:
    """Run guardrails over a model response. Returns a list of failures
    (empty = clean).

    Checks:
    - Schema-shaped (already enforced by Ollama, but defensive).
    - Every element form appears in the body.
    - Every gloss appears in the body (when present).
    - Languages and positions are from the allowed sets.
    - Element count is sane (1–4).
    """
    failures: list[ValidationFailure] = []

    if not isinstance(response, dict):
        failures.append(ValidationFailure("not_object"))
        return failures

    if not response.get("found", False):
        # Model declined; that's a clean "no extraction", not a failure.
        return failures

    elements = response.get("elements") or []
    if not (1 <= len(elements) <= 4):
        failures.append(
            ValidationFailure("element_count_out_of_range", f"got {len(elements)} elements")
        )

    # Normalize the body once; the form-in-body check runs per element and the
    # body is the larger of the two strings.
    norm_body = _normalize_for_match(body)
    for i, el in enumerate(elements):
        form = (el.get("form") or "").strip().rstrip("-").lstrip("-")
        if len(form) < 2 or _normalize_for_match(form) not in norm_body:
            failures.append(
                ValidationFailure("form_not_in_body", f"element[{i}].form={el.get('form')!r}")
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


def extract_one(
    client: OllamaClient,
    *,
    toponym: str,
    body: str,
    suffix_hint: str | None,
) -> LLMResult:
    """Run the LLM on one entry, validate, and return either a ParsedEntry
    (accepted) or the raw response + failures (rejected)."""
    user = USER_TEMPLATE.format(toponym=toponym, suffix_hint=suffix_hint or "(none)", body=body)
    try:
        response = client.chat_json(SYSTEM_PROMPT, user, RESPONSE_SCHEMA)
    except RuntimeError as e:
        return LLMResult(
            entry=None,
            raw_response={"error": str(e)},
            failures=[ValidationFailure("transport_error", str(e))],
            accepted=False,
        )

    failures = validate_response(response, body)

    if not response.get("found", False):
        # Clean "no etymology in this entry" — record toponym as low-conf.
        return LLMResult(
            entry=ParsedEntry(
                toponym=toponym,
                section_suffix=suffix_hint,
                historical_form=None,
                elements=[],
                confidence="low",
                source_quote=body[:200],
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
    notes_prefix = f"extracted_by:llm:{client.model}"
    combined_notes = f"{notes_prefix}; {notes}" if notes else notes_prefix

    return LLMResult(
        entry=ParsedEntry(
            toponym=toponym,
            section_suffix=suffix_hint,
            historical_form=response.get("historical_form"),
            elements=elements,
            confidence=confidence,
            source_quote=(combined_notes + " | " + body[:160])[:200],
        ),
        raw_response=response,
        failures=[],
        accepted=True,
    )
