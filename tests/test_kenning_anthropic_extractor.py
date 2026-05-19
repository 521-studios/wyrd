"""Tests for the Anthropic extractor.

Mirrors the Gemini extractor test surface — same offline pieces:
- API key resolution (env vs. explicit kwarg)
- Markdown-fence stripping in chat_json's response handling
- That extract_one returns the same shape as the Ollama/Gemini equivalents,
  including the accepted-but-empty path for found=False
- That a transport error surfaces as a non-accepted result with a
  ValidationFailure of reason='transport_error'
- That successful extractions tag source_quote with the provider notes
  prefix so consumers can filter

Live-API behavior is exercised manually during integration.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from wyrd.generators.kenning import anthropic_extractor as ae
from wyrd.generators.kenning.extractors.anthropic import AnthropicClient, extract_one

# --- API key resolution ----------------------------------------------------


def test_api_key_resolves_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-env-key")
    client = AnthropicClient()
    assert client._resolve_key() == "test-env-key"


def test_api_key_explicit_overrides_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    client = AnthropicClient(api_key="explicit-key")
    assert client._resolve_key() == "explicit-key"


def test_api_key_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = AnthropicClient()
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        client._resolve_key()


# --- Base URL --------------------------------------------------------------


def test_base_url_includes_model_name() -> None:
    """`mine-llm` logs `client.base_url` for status output. The Anthropic
    implementation must surface the model name like Ollama/Gemini do."""
    client = AnthropicClient(model="claude-sonnet-4-6")
    assert "claude-sonnet-4-6" in client.base_url
    assert "anthropic.com" in client.base_url


def test_default_model_constant_is_sane() -> None:
    """If WYRD_ANTHROPIC_MODEL is unset at module load, default lands on a
    'claude-...-N-N' shaped string. Regression guard for the env-vs-default
    fallback path."""
    assert "claude" in ae.DEFAULT_ANTHROPIC_MODEL


# --- chat_json: markdown-fence stripping -----------------------------------


def _envelope_with(text: str) -> dict:
    """Build a fake Anthropic Messages-API response envelope wrapping `text`."""
    return {"content": [{"type": "text", "text": text}]}


def _patched_urlopen(envelope: dict):
    """Return a mock urlopen context manager that yields a body with `envelope`."""
    body = json.dumps(envelope).encode("utf-8")

    class _Resp:
        def read(self) -> bytes:
            return body

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            pass

    return lambda req, timeout=None: _Resp()


def test_chat_json_handles_unfenced_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    client = AnthropicClient()
    envelope = _envelope_with('{"found": false}')
    with patch("urllib.request.urlopen", _patched_urlopen(envelope)):
        out = client.chat_json("sys", "usr")
    assert out == {"found": False}


def test_chat_json_strips_triple_backtick_fence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Belt-and-suspenders fence stripping: models occasionally wrap JSON in
    ```json …``` even when told not to."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    client = AnthropicClient()
    envelope = _envelope_with('```json\n{"found": false}\n```')
    with patch("urllib.request.urlopen", _patched_urlopen(envelope)):
        out = client.chat_json("sys", "usr")
    assert out == {"found": False}


def test_chat_json_strips_bare_backtick_fence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same as above but without a language tag after the opening fence."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    client = AnthropicClient()
    envelope = _envelope_with('```\n{"found": false}\n```')
    with patch("urllib.request.urlopen", _patched_urlopen(envelope)):
        out = client.chat_json("sys", "usr")
    assert out == {"found": False}


def test_chat_json_non_utf8_body_raises_runtimeerror(monkeypatch: pytest.MonkeyPatch) -> None:
    """wyrd-pe4g round-5 silent-failure-hunter: a 2xx response whose
    body isn't valid UTF-8 (binary blob, corrupted upstream, proxy
    injection) must surface as RuntimeError. UnicodeDecodeError is a
    ValueError subclass — without the `.decode("utf-8")` wrap on the
    success path it leaks through _PROGRAMMER_ERROR_EXCEPTIONS and
    aborts the run identically to the round-4 envelope-parse leak."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    client = AnthropicClient()

    class _NonUtf8Resp:
        def read(self) -> bytes:
            return b"\xff\xfe\xfd binary garbage"  # 0xff is invalid UTF-8 leading byte

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            pass

    with (
        patch("urllib.request.urlopen", lambda req, timeout=None: _NonUtf8Resp()),
        pytest.raises(RuntimeError, match="non-UTF-8 body"),
    ):
        client.chat_json("sys", "usr")


def test_chat_json_malformed_envelope_raises_runtimeerror(monkeypatch: pytest.MonkeyPatch) -> None:
    """wyrd-pe4g round-4: a 2xx response with non-JSON body (CDN/proxy
    HTML error page, truncated body on a dropped connection) must
    surface as RuntimeError so the toponym mining chunk loop buckets
    it into chunks_failed. Without the outer envelope-parse wrap in
    anthropic_extractor.py, json.JSONDecodeError (a ValueError
    subclass) would propagate through _PROGRAMMER_ERROR_EXCEPTIONS
    (which lists ValueError to surface schema_dialect typos) and
    abort the whole multi-source run."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    client = AnthropicClient()

    class _MalformedResp:
        def read(self) -> bytes:
            return b"<html><body>503 Service Unavailable</body></html>"

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            pass

    with (
        patch("urllib.request.urlopen", lambda req, timeout=None: _MalformedResp()),
        pytest.raises(RuntimeError, match="non-JSON envelope"),
    ):
        client.chat_json("sys", "usr")


def test_chat_json_non_json_content_raises_runtimeerror(monkeypatch: pytest.MonkeyPatch) -> None:
    """wyrd-rmc6: the inner content parse (text block → JSON) must also
    raise RuntimeError on JSONDecodeError, not let it leak through
    _PROGRAMMER_ERROR_EXCEPTIONS as ValueError. This pins the helper
    extraction: a regression that drops the parse_transport_json wrap on
    the content path would abort the whole multi-source run on a single
    malformed model response."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    client = AnthropicClient()

    # Outer envelope is valid; content[0].text holds garbage that isn't JSON.
    envelope = json.dumps({"content": [{"type": "text", "text": "this is not json {{"}]}).encode(
        "utf-8"
    )

    class _Resp:
        def read(self) -> bytes:
            return envelope

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            pass

    with (
        patch("urllib.request.urlopen", lambda req, timeout=None: _Resp()),
        pytest.raises(RuntimeError, match="non-JSON content"),
    ):
        client.chat_json("sys", "usr")


def test_chat_json_unexpected_envelope_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """If Anthropic returns content blocks with no text block at all, surface
    a clean RuntimeError instead of letting a KeyError leak through."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    client = AnthropicClient()
    envelope = {"content": [{"type": "tool_use"}]}  # no text block
    with (
        patch("urllib.request.urlopen", _patched_urlopen(envelope)),
        pytest.raises(RuntimeError, match="unexpected envelope"),
    ):
        client.chat_json("sys", "usr")


# --- extract_one shape parity ---------------------------------------------


def test_extract_one_transport_error_returns_non_accepted_failure() -> None:
    """If the HTTP layer raises, extract_one must return accepted=False with
    a ValidationFailure of reason='transport_error'. Mirrors the Gemini and
    Ollama equivalents — the mining loop tracks failure reasons by this tag."""

    class FailingClient:
        def chat_json(self, system, user, schema=None):
            raise RuntimeError("boom")

    result = extract_one(
        FailingClient(),  # type: ignore[arg-type]
        toponym="Foobar",
        body="Foobar — said to be from Old Foo, a foo.",
        suffix_hint=None,
    )
    assert result.accepted is False
    assert any(f.reason == "transport_error" for f in result.failures)


def test_extract_one_not_found_returns_accepted_empty_entry() -> None:
    """When the model returns `found: false`, extract_one returns an
    accepted ParsedEntry with no elements (mirrors gemini/ollama behavior).
    The mining CLI counts these as 'declined' rather than 'rejected'."""

    class NotFoundClient:
        # `model` is referenced when assembling the notes_prefix even on the
        # not-found path (the prefix lives on the entry's source_quote). Real
        # clients always carry a model; minimal test stubs need to as well.
        model = "claude-sonnet-4-6"

        def chat_json(self, system, user, schema=None):
            return {"found": False, "notes": "no etymology in body"}

    result = extract_one(
        NotFoundClient(),  # type: ignore[arg-type]
        toponym="Zzqxk",
        body="Zzqxk — origin unknown.",
        suffix_hint=None,
    )
    assert result.accepted is True
    assert result.entry is not None
    assert result.entry.elements == []
    assert result.entry.confidence == "low"


def test_extract_one_tags_source_quote_with_provider_prefix() -> None:
    """Successful extractions must tag source_quote with `extracted_by:anthropic:`
    plus the model name. Consumers (D12 search-evidence vs. citation-evidence
    distinction; report views) filter on this prefix to attribute rows by
    provider."""

    class GoodClient:
        model = "claude-sonnet-4-6"

        def chat_json(self, system, user, schema=None):
            return {
                "found": True,
                "historical_form": "Foobar",
                "confidence": "medium",
                "elements": [
                    {
                        "form": "foo",
                        "language": "old-english",
                        "position": "pre",
                        "gloss": "a foo",
                        "inflection": None,
                    }
                ],
                "notes": "OE *foo attested",
            }

    result = extract_one(
        GoodClient(),  # type: ignore[arg-type]
        toponym="Foobar",
        body="Foobar — from OE foo, a foo, plus -bar suffix.",
        suffix_hint="-bar",
    )
    assert result.accepted is True
    assert result.entry is not None
    assert "extracted_by:anthropic:claude-sonnet-4-6" in (result.entry.source_quote or "")


# --- 429 retry wiring (wyrd-cfa) ------------------------------------------


def test_chat_json_retries_through_provider_retry_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Integration: chat_json must route through provider_retry.open_with_429_retry
    so a transient 429 doesn't propagate as a transport_error_result. Pin the
    wiring with one urlopen mock that raises 429 once then succeeds — the
    helper's retry layer must absorb the rate-limit blip and chat_json must
    return the parsed JSON from the recovery call."""
    import urllib.error

    from wyrd.generators.kenning import provider_retry as pr

    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr(pr.time, "sleep", lambda _s: None)

    calls = {"n": 0}
    envelope = _envelope_with('{"found": false}')
    body = json.dumps(envelope).encode("utf-8")

    def flaky_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(
                url="https://api.anthropic.com/v1/messages",
                code=429,
                msg="rate limited",
                hdrs=None,  # type: ignore[arg-type]
                fp=None,
            )

        class _Resp:
            def read(self) -> bytes:
                return body

            def __enter__(self):
                return self

            def __exit__(self, *args) -> None:
                pass

        return _Resp()

    client = AnthropicClient()
    with patch("urllib.request.urlopen", flaky_urlopen):
        out = client.chat_json("sys", "usr")
    assert out == {"found": False}
    assert calls["n"] == 2
