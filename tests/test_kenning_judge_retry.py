"""Unit tests for the shared LLM judge-with-retry loop (cli/lexicon/_judge_retry).

The loop was deduped out of five mining/audit commands (wyrd-l5n7); these pin its
contract directly via injected fakes — no Ollama needed. The diagnostic-accuracy
test is the regression for the drift bug the dedup closed: three of the five
former copies lacked the ``last_err`` reset, so their skip message could
misattribute an unparseable-verdict failure to a stale prior transport exception.
"""

from __future__ import annotations

from wyrd.generators.kenning.cli.lexicon._judge_retry import judge_with_retry


class _FakeClient:
    """``chat_json`` replays a queued sequence; an Exception value is RAISED
    (simulating transport flakiness), anything else is returned."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def chat_json(self, system, user, schema):
        self.calls += 1
        r = self._responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _build_prompt(c):
    return ("system", f"user-{c}")


def _parse(raw):
    """Identity parser — the fake client already yields verdict-or-None."""
    return raw


def test_returns_verdict_on_first_success():
    client = _FakeClient(["VERDICT"])
    out = judge_with_retry(client, "c1", build_prompt=_build_prompt, parse_verdict=_parse, ref="r1")
    assert out == "VERDICT"
    assert client.calls == 1  # no retry needed


def test_retries_once_after_unparseable_then_succeeds():
    client = _FakeClient([None, "VERDICT"])
    out = judge_with_retry(client, "c1", build_prompt=_build_prompt, parse_verdict=_parse, ref="r1")
    assert out == "VERDICT"
    assert client.calls == 2


def test_none_and_unparseable_diagnostic_after_both_fail(capsys):
    client = _FakeClient([None, None])
    out = judge_with_retry(client, "c1", build_prompt=_build_prompt, parse_verdict=_parse, ref="r1")
    assert out is None
    assert "skip (judge failed for r1): unparseable verdict" in capsys.readouterr().err


def test_diagnostic_reflects_last_attempt_not_stale_exception(capsys):
    """Drift regression (wyrd-l5n7): attempt-1 raises, attempt-2 returns
    unparseable. The skip message must report THIS attempt ('unparseable
    verdict'), NOT the stale attempt-1 exception — the reset 3 of the 5 former
    copies lacked. Fail-before: without the reset, 'ollama down' leaks."""
    client = _FakeClient([ConnectionError("ollama down"), None])
    out = judge_with_retry(client, "c1", build_prompt=_build_prompt, parse_verdict=_parse, ref="r1")
    assert out is None
    err = capsys.readouterr().err
    assert "unparseable verdict" in err
    assert "ollama down" not in err  # the stale prior exception must not leak


def test_diagnostic_reports_exception_when_last_attempt_raises(capsys):
    client = _FakeClient([None, RuntimeError("boom")])
    out = judge_with_retry(client, "c1", build_prompt=_build_prompt, parse_verdict=_parse, ref="r1")
    assert out is None
    assert "boom" in capsys.readouterr().err
