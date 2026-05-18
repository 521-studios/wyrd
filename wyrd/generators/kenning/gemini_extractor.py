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
from typing import ClassVar

from wyrd.generators.kenning.llm_extractor import (
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    LLMResult,
    assemble_extraction_result,
    parse_transport_json,
    transport_error_result,
)
from wyrd.generators.kenning.provider_retry import open_with_429_retry

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
        # D5-1 / wyrd-3ux. Year-range constraint lives in the prompt and
        # in validate_response, not in the schema — Gemini's responseSchema
        # dialect's enforcement of integer minimum / maximum has been
        # unreliable across model versions in our testing, so we don't
        # rely on it. The prompt asks for 100-1700 and validate_response
        # (in llm_extractor) is the load-bearing range gate.
        "attested_forms": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "form": {"type": "STRING"},
                    "year": {"type": "INTEGER"},
                },
                "required": ["form", "year"],
            },
        },
        "confidence": {
            "type": "STRING",
            "enum": ["high", "medium", "low"],
        },
        "notes": {"type": "STRING", "nullable": True},
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


# --- HTTP client ----------------------------------------------------------


@dataclass
class GeminiClient:
    """Minimal Gemini generateContent client. Single-shot, no streaming."""

    # Dispatch marker read by callers (e.g. toponym_mention_extractor) that
    # need to pick between JSON Schema and Gemini's OpenAPI-3.0-ish
    # responseSchema dialect. ClassVar so dataclass treats this as a
    # class-level attribute, not a field on instances. wyrd-pe4g.
    schema_dialect: ClassVar[str] = "openapi"

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
            body = open_with_429_retry(req, timeout=self.timeout_s, provider_label="Gemini").decode(
                "utf-8"
            )
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini HTTP {e.code}: {err_body[:500]}") from e
        except (TimeoutError, urllib.error.URLError) as e:
            # TimeoutError isn't a URLError subclass; needs separate catch
            # so a single slow Gemini call doesn't crash the review loop.
            raise RuntimeError(f"Gemini unreachable at {url}: {e}") from e
        except UnicodeDecodeError as e:
            # Non-UTF-8 body on a 2xx response — corrupted upstream,
            # CDN/proxy injecting binary content, etc. UnicodeDecodeError
            # is a ValueError subclass; without this wrap it would
            # propagate through the chunk loop's _PROGRAMMER_ERROR_EXCEPTIONS
            # re-raise (wyrd-pe4g) and abort the whole multi-source run.
            raise RuntimeError(f"Gemini returned non-UTF-8 body: {e}") from e

        envelope = parse_transport_json(body, provider="Gemini", kind="envelope")
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

        return parse_transport_json(text, provider="Gemini", kind="content")


# --- extraction -----------------------------------------------------------


def extract_one(
    client: GeminiClient,
    *,
    toponym: str,
    body: str,
    suffix_hint: str | None,
) -> LLMResult:
    """Run Gemini against one entry body. Thin wrapper over
    ``assemble_extraction_result`` — same return shape and validation
    path as the Ollama and Anthropic extractors. Gemini's chat_json takes
    only (system, user); the response schema enforcement happens via the
    GenerationConfig responseSchema attached at client init."""
    user = USER_TEMPLATE.format(toponym=toponym, suffix_hint=suffix_hint or "(none)", body=body)
    try:
        response = client.chat_json(SYSTEM_PROMPT, user)
    except RuntimeError as e:
        return transport_error_result(str(e))
    return assemble_extraction_result(
        response,
        body=body,
        toponym=toponym,
        suffix_hint=suffix_hint,
        notes_prefix=f"extracted_by:gemini:{client.model}",
    )
