"""Gemini-based etymology extraction.

Parallel implementation of `llm_extractor`'s `extract_one` flow but pointed
at Google's Gemini API instead of a local Ollama server. Same schema, same
validation guard, same `LLMResult`/`ParsedEntry` outputs — `mine-llm` can
swap providers via a `--provider` flag.

We use the Gemini REST API directly (urllib) rather than the
`google-generativeai` SDK to keep the project's runtime deps minimal. The
trade-off is no streaming, no batching — fine for our use case (one prompt
per entry, ~150 tokens out).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass

from wyrd.generators.kenning.llm_extractor import (
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    LLMResult,
    ValidationFailure,
    validate_response,
)
from wyrd.generators.kenning.skeat_parser import ParsedElement, ParsedEntry

# --- config ---------------------------------------------------------------

DEFAULT_GEMINI_MODEL = os.environ.get("WYRD_GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# Gemini's responseSchema is OpenAPI-3.0-like (not JSON Schema). It accepts
# fewer keywords than llm_extractor.RESPONSE_SCHEMA — we hand-translate to
# the supported subset. Notable absences vs JSON Schema:
#   - additionalProperties (Gemini disallows extras by default in tight mode)
#   - "type": ["string", "null"]  (Gemini uses nullable: true)
#   - minLength on strings (silently allowed)
GEMINI_RESPONSE_SCHEMA: dict = {
    "type": "OBJECT",
    "properties": {
        "found": {"type": "BOOLEAN"},
        "historical_form": {"type": "STRING", "nullable": True},
        "elements": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "form": {"type": "STRING"},
                    "language": {
                        "type": "STRING",
                        "enum": [
                            "old-english",
                            "old-norse",
                            "norman-french",
                            "celtic",
                            "latin",
                            "germanic",
                            "greek",
                            "biblical",
                            "modern-english",
                        ],
                    },
                    "position": {
                        "type": "STRING",
                        "enum": ["pre", "inner", "post"],
                    },
                    "gloss": {"type": "STRING", "nullable": True},
                    "inflection": {"type": "STRING", "nullable": True},
                },
                "required": ["form", "language", "position", "gloss", "inflection"],
            },
        },
        "confidence": {
            "type": "STRING",
            "enum": ["high", "medium", "low"],
        },
        "notes": {"type": "STRING", "nullable": True},
    },
    "required": ["found", "historical_form", "elements", "confidence", "notes"],
}


# --- HTTP client ----------------------------------------------------------


@dataclass
class GeminiClient:
    """Minimal Gemini generateContent client. Single-shot, no streaming."""

    api_key: str | None = None  # falls back to GEMINI_API_KEY at call time
    model: str = DEFAULT_GEMINI_MODEL
    timeout_s: float = 120.0
    temperature: float = 0.0

    @property
    def base_url(self) -> str:
        # Mimic OllamaClient.base_url so logging is interchangeable.
        return f"{GEMINI_API_BASE}/models/{self.model}"

    def _resolve_key(self) -> str:
        key = self.api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set. Configure it or pass api_key=.")
        return key

    def chat_json(self, system: str, user: str, response_schema: dict | None = None) -> dict:
        """Send a single prompt with JSON-schema-constrained output. Returns
        the parsed JSON content.

        `response_schema` defaults to the extraction schema
        (`GEMINI_RESPONSE_SCHEMA`). Callers building a different output
        shape (e.g. the disambiguator) pass their own Gemini-flavored
        OpenAPI schema. The schema must follow Gemini's responseSchema
        dialect (uppercase types, `nullable: true` instead of nullable
        unions, no `additionalProperties`).
        """
        url = f"{self.base_url}:generateContent"
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "responseMimeType": "application/json",
                "responseSchema": response_schema
                if response_schema is not None
                else GEMINI_RESPONSE_SCHEMA,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self._resolve_key(),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini HTTP {e.code}: {err_body[:500]}") from e
        except (TimeoutError, urllib.error.URLError) as e:
            # TimeoutError isn't a URLError subclass; needs separate catch
            # so a single slow Gemini call doesn't crash the review loop.
            raise RuntimeError(f"Gemini unreachable at {url}: {e}") from e

        envelope = json.loads(body)
        # Standard Gemini response shape: candidates[0].content.parts[0].text
        try:
            candidate = envelope["candidates"][0]
            text = candidate["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Gemini returned unexpected envelope: {body[:500]}") from e

        if not text:
            # Empty completion (rare, but possible if the model refused).
            raise RuntimeError(
                f"Gemini produced empty content. finish_reason={candidate.get('finishReason')}"
            )

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Gemini produced non-JSON content: {text[:500]}") from e


# --- extraction -----------------------------------------------------------


def extract_one(
    client: GeminiClient,
    *,
    toponym: str,
    body: str,
    suffix_hint: str | None,
) -> LLMResult:
    """Run Gemini against one entry body. Same return shape as the Ollama
    extractor."""
    user = USER_TEMPLATE.format(toponym=toponym, suffix_hint=suffix_hint or "(none)", body=body)
    try:
        response = client.chat_json(SYSTEM_PROMPT, user)
    except RuntimeError as e:
        return LLMResult(
            entry=None,
            raw_response={"error": str(e)},
            failures=[ValidationFailure("transport_error", str(e))],
            accepted=False,
        )

    failures = validate_response(response, body)

    if not response.get("found", False):
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
    notes_prefix = f"extracted_by:gemini:{client.model}"
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
