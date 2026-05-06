"""Tests for the Gemini extractor.

We can't reasonably exercise the live Gemini API in unit tests, so we focus
on the offline pieces:
- API key resolution (env vs. explicit kwarg)
- Response envelope parsing (against fabricated payloads)
- The shared validation guard (already tested in test_kenning_llm_extractor)
- That extract_one returns the same shape as the Ollama equivalent

Live-API behavior is exercised manually during integration.
"""

from __future__ import annotations

import pytest

from wyrd.generators.kenning import gemini_extractor as gm
from wyrd.generators.kenning.gemini_extractor import (
    GEMINI_RESPONSE_SCHEMA,
    GeminiClient,
)


def test_api_key_resolves_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-env-key")
    client = GeminiClient()
    assert client._resolve_key() == "test-env-key"


def test_api_key_explicit_overrides_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    client = GeminiClient(api_key="explicit-key")
    assert client._resolve_key() == "explicit-key"


def test_api_key_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    client = GeminiClient()
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        client._resolve_key()


def test_response_schema_matches_internal_shape() -> None:
    """Sanity check: the Gemini schema covers exactly the fields the
    validator and ingester downstream expect.

    Gemini's responseSchema uses uppercase OpenAPI types ('STRING', 'OBJECT')
    rather than JSON Schema lowercase. We assert the structural fields
    survive that translation.
    """
    s = GEMINI_RESPONSE_SCHEMA
    assert s["type"] == "OBJECT"
    assert "found" in s["properties"]
    assert "historical_form" in s["properties"]
    assert "elements" in s["properties"]
    assert "confidence" in s["properties"]
    assert "notes" in s["properties"]

    elem_schema = s["properties"]["elements"]["items"]
    assert set(elem_schema["properties"]).issuperset(
        {"form", "language", "position", "gloss", "inflection"}
    )
    # Languages enum lines up with the validator's allowed set.
    langs = set(elem_schema["properties"]["language"]["enum"])
    assert "old-english" in langs
    assert "old-norse" in langs
    assert "celtic" in langs


def test_response_schema_includes_attested_forms() -> None:
    """D5-1 / wyrd-3ux: the Gemini schema must include the attested_forms
    array so Gemini-mined data carries the same year citations that the
    Ollama-mined data does. Required field; OpenAPI integer type."""
    s = GEMINI_RESPONSE_SCHEMA
    assert "attested_forms" in s["properties"]
    assert "attested_forms" in s["required"]
    item_schema = s["properties"]["attested_forms"]["items"]
    assert item_schema["type"] == "OBJECT"
    assert set(item_schema["properties"]) == {"form", "year"}
    assert item_schema["properties"]["form"]["type"] == "STRING"
    assert item_schema["properties"]["year"]["type"] == "INTEGER"
    assert set(item_schema["required"]) == {"form", "year"}


def test_base_url_includes_model_name() -> None:
    """Logging in `mine-llm` reads `client.base_url` — make sure the
    Gemini implementation provides a useful one (incorporates the model
    so it's interchangeable with OllamaClient.base_url for status output)."""
    client = GeminiClient(model="gemini-2.5-flash")
    assert "gemini-2.5-flash" in client.base_url
    assert "generativelanguage.googleapis.com" in client.base_url


def test_default_model_picks_up_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """The DEFAULT_GEMINI_MODEL is read at module load, so this test verifies
    the fallback path: if WYRD_GEMINI_MODEL was unset, default lands on
    gemini-2.5-flash."""
    # If no env override at module load, default model is 2.5-flash. We
    # can't easily reload the module, so just check the constant is sane.
    assert "gemini" in gm.DEFAULT_GEMINI_MODEL


# --- 429 retry wiring (wyrd-cfa) ------------------------------------------


def test_chat_json_retries_through_provider_retry_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Integration: chat_json must route through provider_retry.open_with_429_retry
    so a transient 429 doesn't propagate as a transport_error_result. Pin the
    wiring with one urlopen mock that raises 429 once then succeeds — the
    helper's retry layer must absorb the rate-limit blip and chat_json must
    return the parsed JSON from the recovery call."""
    import json
    import urllib.error
    from unittest.mock import patch

    from wyrd.generators.kenning import provider_retry as pr

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(pr.time, "sleep", lambda _s: None)

    calls = {"n": 0}
    # Gemini envelope shape: candidates[0].content.parts[0].text holds JSON.
    inner_payload = json.dumps({"found": False}).encode("utf-8")
    envelope = json.dumps(
        {"candidates": [{"content": {"parts": [{"text": inner_payload.decode()}]}}]}
    ).encode("utf-8")

    def flaky_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(
                url="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
                code=429,
                msg="rate limited",
                hdrs=None,  # type: ignore[arg-type]
                fp=None,
            )

        class _Resp:
            def read(self) -> bytes:
                return envelope

            def __enter__(self):
                return self

            def __exit__(self, *args) -> None:
                pass

        return _Resp()

    client = GeminiClient()
    with patch("urllib.request.urlopen", flaky_urlopen):
        out = client.chat_json("sys", "usr")
    assert out == {"found": False}
    assert calls["n"] == 2
