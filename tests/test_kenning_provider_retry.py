"""Tests for provider_retry.open_with_429_retry."""

from __future__ import annotations

import urllib.error
import urllib.request
from unittest.mock import patch

import pytest

from wyrd.generators.kenning import provider_retry as pr


def _ok_response(body: bytes = b'{"ok": true}'):
    """Mock urlopen context manager yielding ``body``."""

    class _Resp:
        def read(self) -> bytes:
            return body

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            pass

    return lambda req, timeout=None: _Resp()


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="https://example/",
        code=code,
        msg="rate limited" if code == 429 else "boom",
        hdrs=None,  # type: ignore[arg-type]
        fp=None,
    )


def _make_req() -> urllib.request.Request:
    return urllib.request.Request("https://example/")


def test_returns_body_on_success() -> None:
    with patch("urllib.request.urlopen", _ok_response(b"hi")):
        body = pr.open_with_429_retry(_make_req(), timeout=1.0, provider_label="X")
    assert body == b"hi"


def test_retries_then_succeeds_on_429(monkeypatch: pytest.MonkeyPatch) -> None:
    """First call raises 429; second call succeeds. Sleep is mocked to 0."""
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _http_error(429)

        class _Resp:
            def read(self) -> bytes:
                return b"recovered"

            def __enter__(self):
                return self

            def __exit__(self, *args) -> None:
                pass

        return _Resp()

    monkeypatch.setattr(pr.time, "sleep", lambda _s: None)
    with patch("urllib.request.urlopen", fake_urlopen):
        body = pr.open_with_429_retry(_make_req(), timeout=1.0, provider_label="X")
    assert body == b"recovered"
    assert calls["n"] == 2


@pytest.mark.parametrize("code", [429, 503, 529])
def test_retries_then_succeeds_on_transient_overload(
    code: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every transient status — 429 rate-limit AND the overload codes (Anthropic
    529, Gemini 503) — is retried, not just 429. The first call raises the code;
    the second succeeds."""
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _http_error(code)
        return _ok_response(b"recovered")(req, timeout=timeout)

    monkeypatch.setattr(pr.time, "sleep", lambda _s: None)
    with patch("urllib.request.urlopen", fake_urlopen):
        body = pr.open_with_429_retry(_make_req(), timeout=1.0, provider_label="X")
    assert body == b"recovered"
    assert calls["n"] == 2


def test_propagates_429_after_max_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    """After exhausting attempts the most recent 429 propagates so the
    provider's caller can convert it to a transport_error_result."""
    calls = {"n": 0}

    def always_429(req, timeout=None):
        calls["n"] += 1
        raise _http_error(429)

    monkeypatch.setattr(pr.time, "sleep", lambda _s: None)
    with (
        patch("urllib.request.urlopen", always_429),
        pytest.raises(urllib.error.HTTPError) as exc,
    ):
        pr.open_with_429_retry(_make_req(), timeout=1.0, provider_label="X", max_attempts=3)
    assert exc.value.code == 429
    assert calls["n"] == 3


def test_non_retryable_http_error_propagates_immediately() -> None:
    """A 500 is a genuine server error, NOT a transient overload (unlike 503/529),
    so it isn't in _RETRYABLE_STATUS — propagate without sleeping."""
    calls = {"n": 0}

    def always_500(req, timeout=None):
        calls["n"] += 1
        raise _http_error(500)

    with (
        patch("urllib.request.urlopen", always_500),
        pytest.raises(urllib.error.HTTPError) as exc,
    ):
        pr.open_with_429_retry(_make_req(), timeout=1.0, provider_label="X")
    assert exc.value.code == 500
    assert calls["n"] == 1


def test_socket_timeout_propagates_immediately() -> None:
    """TimeoutError is not retried — propagate so the provider can surface it."""

    def timing_out(req, timeout=None):
        raise TimeoutError("read timed out")

    with (
        patch("urllib.request.urlopen", timing_out),
        pytest.raises(TimeoutError),
    ):
        pr.open_with_429_retry(_make_req(), timeout=1.0, provider_label="X")


def test_429_response_is_closed_before_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the e.close() call between 429 and the next retry. urllib's
    HTTPError doubles as a response object; not closing it leaks the
    underlying socket, which compounds at high --concurrency. Regression
    guard: a future refactor that drops the close() (or moves it after the
    sleep) would let sockets pile up between attempts."""
    closed: list[urllib.error.HTTPError] = []

    class _ClosableHTTPError(urllib.error.HTTPError):
        def close(self) -> None:
            closed.append(self)

    calls = {"n": 0}

    def flaky(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _ClosableHTTPError(
                url="https://example/",
                code=429,
                msg="rate limited",
                hdrs=None,  # type: ignore[arg-type]
                fp=None,
            )
        return _ok_response(b"ok")(req, timeout=timeout)

    monkeypatch.setattr(pr.time, "sleep", lambda _s: None)
    with patch("urllib.request.urlopen", flaky):
        body = pr.open_with_429_retry(_make_req(), timeout=1.0, provider_label="X")
    assert body == b"ok"
    assert len(closed) == 1, "the 429 HTTPError should be closed before retry"


def test_max_total_wait_caps_cumulative_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the next backoff would push cumulative sleep past max_total_wait,
    propagate the 429 instead of sleeping further."""
    sleeps: list[float] = []

    def fake_sleep(s: float) -> None:
        sleeps.append(s)

    def always_429(req, timeout=None):
        raise _http_error(429)

    monkeypatch.setattr(pr.time, "sleep", fake_sleep)
    # Make jitter deterministic: always 1.0x.
    monkeypatch.setattr(pr.random, "uniform", lambda lo, hi: 1.0)
    with (
        patch("urllib.request.urlopen", always_429),
        pytest.raises(urllib.error.HTTPError),
    ):
        # max_total_wait=1.5 lets the first sleep (2^0=1.0) through but
        # blocks the second (cumulative would be 1.0+2.0=3.0 > 1.5).
        pr.open_with_429_retry(
            _make_req(),
            timeout=1.0,
            provider_label="X",
            max_attempts=5,
            max_total_wait=1.5,
        )
    assert sleeps == [1.0]
