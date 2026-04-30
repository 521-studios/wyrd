"""Anthropic Claude-based etymology extraction.

Third provider option (alongside Ollama and Gemini), positioned as the
"deluxe quality" tier in the layered review pipeline:

  Tier 1: Ollama / Qwen — bulk first-pass extraction (free, fast, ~70%
          quality on this task)
  Tier 2: Gemini Flash  — review pass on questionable Qwen rows
          (~$1.50/book, catches modern-form leakage and OCR confusion)
  Tier 3: Anthropic Claude (Sonnet 4.6 default) — final pass on cases
          where Qwen and Gemini disagree, or both flagged low-confidence
          (~$1.50/book, strongest at hedge-recognition and OE diacritic
          handling)

Same `extract_one(client, ...)` signature as the other extractors so the
review CLI can swap providers via `--provider anthropic`. Same shared
SYSTEM_PROMPT, USER_TEMPLATE, and validation guard from llm_extractor.

Uses the Anthropic Messages API directly via urllib (no SDK dep, like our
Gemini integration).
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

DEFAULT_ANTHROPIC_MODEL = os.environ.get("WYRD_ANTHROPIC_MODEL", "claude-sonnet-4-6")
ANTHROPIC_API_BASE = "https://api.anthropic.com/v1"
ANTHROPIC_API_VERSION = "2023-06-01"


# Anthropic Messages API doesn't have JSON-schema constrained output the way
# Ollama and Gemini do. We get reliable JSON via prompt engineering: tight
# SYSTEM_PROMPT (already in llm_extractor) plus a final user instruction
# making the format explicit. The validate_response guard catches anything
# that slips through.
#
# An alternative would be Anthropic's tool-use mode with a tool that takes
# the schema as its input — but that adds a tool-call/-result round trip
# for no quality gain on this constrained task.


@dataclass
class AnthropicClient:
    """Minimal Anthropic Messages-API client. Single-shot, no streaming."""

    api_key: str | None = None  # falls back to ANTHROPIC_API_KEY at call time
    model: str = DEFAULT_ANTHROPIC_MODEL
    timeout_s: float = 180.0
    temperature: float = 0.0
    max_tokens: int = 1024

    @property
    def base_url(self) -> str:
        # Mirror OllamaClient/GeminiClient's interface for status logging.
        return f"{ANTHROPIC_API_BASE}/messages (model={self.model})"

    def _resolve_key(self) -> str:
        key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Configure it or pass api_key=.")
        return key

    def chat_json(self, system: str, user: str, schema: dict | None = None) -> dict:
        """Send a single Anthropic message; return parsed JSON content.

        `schema` is accepted for interface parity with Ollama/Gemini but not
        used — Anthropic Messages doesn't have schema-constrained output.
        We rely on the prompt + validate_response.
        """
        url = f"{ANTHROPIC_API_BASE}/messages"
        payload = {
            "model": self.model,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "anthropic-version": ANTHROPIC_API_VERSION,
                "x-api-key": self._resolve_key(),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Anthropic HTTP {e.code}: {err_body[:500]}") from e
        except (TimeoutError, urllib.error.URLError) as e:
            raise RuntimeError(f"Anthropic unreachable at {url}: {e}") from e

        envelope = json.loads(body)
        # Standard Messages-API response shape: content[0].text
        try:
            blocks = envelope["content"]
            text_block = next((b for b in blocks if b.get("type") == "text"), None)
            if text_block is None:
                raise KeyError("no text block in response")
            text = text_block["text"]
        except (KeyError, IndexError, StopIteration) as e:
            raise RuntimeError(f"Anthropic returned unexpected envelope: {body[:500]}") from e

        # Strip a markdown fence if the model wrapped the JSON. Belt-and-
        # suspenders — the prompt forbids markdown but models occasionally
        # ignore that.
        text = text.strip()
        if text.startswith("```"):
            # Drop ```{lang}\n at the start and ``` at the end if present
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Anthropic produced non-JSON content: {text[:500]}") from e


def extract_one(
    client: AnthropicClient,
    *,
    toponym: str,
    body: str,
    suffix_hint: str | None,
) -> LLMResult:
    """Run Anthropic against one entry body. Same return shape as the Ollama
    and Gemini extractors."""
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
    notes_prefix = f"extracted_by:anthropic:{client.model}"
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
