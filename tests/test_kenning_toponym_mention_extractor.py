"""Tests for the LLM-driven toponym-mention extractor (wyrd-x82p Phase 2).

The extractor is provider-agnostic — it takes any object with a
``chat_json(system, user, schema=None) -> dict`` method. Tests use a
``FakeClient`` that replays canned responses, so the test suite never
hits a real LLM API. The real provider clients
(AnthropicClient/OllamaClient/GeminiClient) are already covered by
their own extractor's test files; here we focus on the toponym-
extraction-specific logic (validation, chunking, error handling,
range clamping, anti-hallucination, counter rollup).
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli as cli_root
from wyrd.generators.kenning.toponym_mention_extractor import (
    _DATE_YEAR_MAX,
    _DATE_YEAR_MIN,
    SYSTEM_PROMPT,
    MineToponymMentionsReport,
    ToponymMention,
    ValidationCounters,
    _coerce_year,
    _collapse_whitespace,
    _form_in_chunk,
    _validated_mentions,
    chunk_source_body,
    extract_toponym_mentions_from_chunk,
    mine_toponym_mentions,
)

# ---------- FakeClient ---------------------------------------------------


class FakeClient:
    """Minimal stand-in for AnthropicClient/OllamaClient/GeminiClient.

    ``responses`` is consumed in order — one response per ``chat_json``
    call. When the list is exhausted, the next call raises StopIteration
    so over-consumption tests cleanly. Pass an Exception instance in any
    slot to simulate a transport-error scenario (the FakeClient raises
    it on that call).
    """

    def __init__(self, responses: list[Any]):
        self.model = "fake-test-model"
        self._responses = list(responses)
        self.calls: list[tuple[str, str, dict | None]] = []

    def chat_json(self, system: str, user: str, schema: dict | None = None) -> dict:
        self.calls.append((system, user, schema))
        if not self._responses:
            raise StopIteration("FakeClient ran out of canned responses")
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


# ---------- _collapse_whitespace + _form_in_chunk ------------------------


def test_collapse_whitespace_handles_runs_and_newlines():
    """Runs of any whitespace (spaces, tabs, newlines, \\r) collapse to
    one space; leading/trailing trimmed. Anchors the contract the
    anti-hallucination check depends on — soft-wrapped scholar prose
    becomes uniformly searchable."""
    assert _collapse_whitespace("Newcastle  upon\nTyne") == "Newcastle upon Tyne"
    assert _collapse_whitespace("Newcastle\r\nupon\tTyne") == "Newcastle upon Tyne"
    assert _collapse_whitespace("   Edlingham   ") == "Edlingham"
    assert _collapse_whitespace("") == ""


def test_form_in_chunk_matches_across_soft_wrap():
    """Real failure mode: scholar prose has 'Newcastle upon\\nTyne' but
    the LLM emits 'Newcastle upon Tyne'. The naive ``form in chunk``
    rejects this; the whitespace-normalized check accepts. silent-
    failure-hunter round-1 finding."""
    chunk_collapsed = _collapse_whitespace("the city of Newcastle upon\nTyne is")
    assert _form_in_chunk("Newcastle upon Tyne", chunk_collapsed)


def test_form_in_chunk_still_rejects_real_hallucination():
    chunk_collapsed = _collapse_whitespace("Edlingham was first attested in 1086")
    # Near-neighbor confabulation (LLM might emit a plausible-looking
    # form not actually present).
    assert not _form_in_chunk("Eadulfham", chunk_collapsed)


def test_form_in_chunk_rejects_substring_fragment_inside_real_form():
    """Word-boundary check: a partial like "on Ty" must NOT match
    "Newcastle upon Tyne" even though those letters appear in
    sequence. Without word-boundary anchoring, the round-2 whitespace-
    normalized substring check would silently admit fragments of
    legitimate forms."""
    chunk_collapsed = _collapse_whitespace("the city of Newcastle upon Tyne is on the coast")
    assert not _form_in_chunk("on Ty", chunk_collapsed)
    assert not _form_in_chunk("upo", chunk_collapsed)
    # But the full form must still match.
    assert _form_in_chunk("Newcastle upon Tyne", chunk_collapsed)
    # And a form bounded by punctuation/whitespace also matches.
    assert _form_in_chunk("Tyne", chunk_collapsed)


def test_form_in_chunk_rejects_empty_form():
    """An empty form (after collapse) must not match — without this
    guard the regex would match every position in the chunk."""
    assert not _form_in_chunk("", "anything")
    assert not _form_in_chunk("   ", "anything")


# ---------- _coerce_year -------------------------------------------------


def test_coerce_year_accepts_in_range_int():
    assert _coerce_year(1086) == (1086, False)


@pytest.mark.parametrize("year", [_DATE_YEAR_MIN, _DATE_YEAR_MAX])
def test_coerce_year_accepts_exact_boundaries(year):
    """Both inclusive endpoints (700, 1700) must be accepted. A regression
    that flipped <= to < would slip past a test that only used midpoint
    values. fixture-data-reviewer round-1 finding."""
    assert _coerce_year(year) == (year, False)


@pytest.mark.parametrize("year", [_DATE_YEAR_MIN - 1, _DATE_YEAR_MAX + 1, 0, 9999])
def test_coerce_year_rejects_out_of_range_int(year):
    assert _coerce_year(year) == (None, True)


def test_coerce_year_parses_numeric_string():
    """LLMs occasionally emit a year as a string despite the schema —
    accept it if the string parses cleanly, track-as-clamped otherwise."""
    assert _coerce_year("1086") == (1086, False)
    assert _coerce_year("  1086  ") == (1086, False)
    assert _coerce_year("1701") == (None, True)  # out of range after parse


def test_coerce_year_rejects_unparseable_string():
    assert _coerce_year("ten eighty-six") == (None, True)
    assert _coerce_year("1066 AD") == (None, True)
    assert _coerce_year("c. 1086") == (None, True)


@pytest.mark.parametrize(
    "raw",
    [
        "+1086",  # Python int() accepts leading +
        "-1086",  # negatives parse but are out of range — still strict-reject the format
        "1_086",  # underscore separator (Python 3 numeric literals)
        "01086",  # leading zero
        "1086.0",  # decimal point
        "1086 ",  # trailing junk (after strip)
    ],
)
def test_coerce_year_strict_rejects_non_canonical_string(raw):
    """Python's int() is permissive: int('1_086') → 1086. The LLM
    wouldn't emit non-canonical year strings, so admitting them is
    silent semantic drift. Strict digit-only match guards against it."""
    # The stripped versions of "1086 " and "+1086" would parse to 1086,
    # which IS in range — but the FORMAT is wrong. The strict pattern
    # rejects them.
    if raw == "1086 ":
        # After strip(), this is plain "1086" — should accept.
        assert _coerce_year(raw) == (1086, False)
    else:
        assert _coerce_year(raw) == (None, True)


def test_coerce_year_rejects_float():
    """JSON allows numeric values without subtype info — a provider
    that returns 1086.0 (json.loads on `"1086.0"` gives float) must
    not silently admit a non-integer year. test-coverage round-2."""
    assert _coerce_year(1086.0) == (None, True)
    assert _coerce_year(float("nan")) == (None, True)
    assert _coerce_year(float("inf")) == (None, True)


def test_coerce_year_none_is_not_clamped():
    """``None`` input means 'LLM declined to cite a year' — must not be
    counted as a clamp, or operators will see a false 'LLM is producing
    bad years' signal."""
    assert _coerce_year(None) == (None, False)


def test_coerce_year_rejects_bool():
    """``isinstance(True, int) is True`` in Python — guarding against
    year 0/1 from a bool slip."""
    assert _coerce_year(True) == (None, True)
    assert _coerce_year(False) == (None, True)


# ---------- _validated_mentions ------------------------------------------


def test_validated_mentions_keeps_form_present_in_chunk():
    chunk = "Edlingham is the homestead of the sons of Eadwulf."
    raw = [
        {
            "form": "Edlingham",
            "date_year": None,
            "region_hint": None,
            "context": "Edlingham is the homestead",
        },
    ]
    out = _validated_mentions(raw, chunk)
    assert len(out) == 1
    assert out[0].form == "Edlingham"


def test_validated_mentions_drops_near_neighbor_hallucination():
    """A real failure mode the operator should care about: LLM emits a
    plausible-LOOKING form (similar charset, plausible OE pattern) that
    isn't actually in the chunk. fixture-data-reviewer round-1: the
    earlier test used 'Birmingham' which shares zero domain overlap and
    only proves the substring-check works on a trivial miss."""
    chunk = "Edlingham is the homestead of the sons of Eadwulf."
    raw = [
        {"form": "Edlingham", "context": "valid"},
        {"form": "Eadwulfham", "context": "near-neighbor confabulation"},
    ]
    counters = ValidationCounters()
    out = _validated_mentions(raw, chunk, counters=counters)
    assert [m.form for m in out] == ["Edlingham"]
    assert counters.hallucinations_dropped == 1
    assert counters.admitted == 1


def test_validated_mentions_admits_soft_wrapped_form():
    """LLM correctly emits a hyphen-free / single-line form for a
    multi-line chunk source. Without whitespace-normalized matching,
    this drops silently. silent-failure-hunter round-1 finding."""
    chunk = "the borough of Newcastle upon\nTyne is found in"
    raw = [{"form": "Newcastle upon Tyne", "context": "Newcastle upon Tyne is found"}]
    out = _validated_mentions(raw, chunk)
    assert [m.form for m in out] == ["Newcastle upon Tyne"]


def test_validated_mentions_in_range_year_survives():
    """Positive control — the year-clamping test must also confirm that
    an in-range year is preserved, otherwise a bug that nulls EVERY
    year passes the test. pr-test-analyzer round-1 finding."""
    chunk = "Edlingham in 1086"
    raw = [{"form": "Edlingham", "date_year": 1086, "context": "valid"}]
    counters = ValidationCounters()
    out = _validated_mentions(raw, chunk, counters=counters)
    assert out[0].date_year == 1086
    assert counters.years_clamped == 0


def test_validated_mentions_handles_non_int_year_with_positive_control():
    """Mix one good year with two bad ones — the good one must survive
    and the bad ones must clamp. Without a positive control, this test
    passes even if the implementation nulls every year. pr-test-
    analyzer round-1 finding."""
    chunk = "Edlingham mentioned"
    raw = [
        {"form": "Edlingham", "date_year": "ten eighty-six", "context": "x"},
        {"form": "Edlingham", "date_year": 1086, "context": "y"},  # positive control
        {"form": "Edlingham", "date_year": None, "context": "z"},
    ]
    counters = ValidationCounters()
    out = _validated_mentions(raw, chunk, counters=counters)
    years = [m.date_year for m in out]
    assert 1086 in years  # positive control survives
    assert years.count(None) == 2
    assert counters.years_clamped == 1  # only the "ten eighty-six", not None


def test_validated_mentions_empty_region_hint_becomes_none():
    chunk = "Edlingham mentioned"
    raw = [
        {"form": "Edlingham", "date_year": None, "region_hint": "", "context": "x"},
        {"form": "Edlingham", "date_year": None, "region_hint": "  ", "context": "x"},
        {
            "form": "Edlingham",
            "date_year": None,
            "region_hint": "Northumberland",
            "context": "x",
        },
    ]
    out = _validated_mentions(raw, chunk)
    assert out[0].region_hint is None
    assert out[1].region_hint is None
    assert out[2].region_hint == "Northumberland"


def test_validated_mentions_synthesizes_missing_context_with_neighbors():
    """Synthesized context must include surrounding prose, not just the
    form itself — the operator's job is to verify the mention in
    context. pr-test-analyzer round-1: the earlier assertion
    (just 'Edlingham' in context) passed trivially because the form
    was already in the chunk by the substring check."""
    chunk = "the village of Edlingham was first attested in 1086 as a small homestead"
    raw = [
        {"form": "Edlingham", "date_year": 1086, "region_hint": None, "context": ""},
    ]
    out = _validated_mentions(raw, chunk)
    ctx = out[0].context
    assert "Edlingham" in ctx
    assert ctx != "Edlingham"  # must include neighbors, not just the form
    # Must include at least one nearby word from either side.
    assert ("village" in ctx) or ("attested" in ctx) or ("1086" in ctx)


def test_validated_mentions_strips_form_whitespace():
    chunk = "the village of Edlingham"
    raw = [{"form": "  Edlingham  ", "context": "x"}]
    out = _validated_mentions(raw, chunk)
    assert out[0].form == "Edlingham"


def test_validated_mentions_drops_empty_form():
    chunk = "anything"
    raw = [
        {"form": "", "context": "x"},
        {"form": "   ", "context": "x"},
        {"form": None, "context": "x"},
    ]
    counters = ValidationCounters()
    assert _validated_mentions(raw, chunk, counters=counters) == []
    # Empty forms are NOT hallucinations (we never got far enough to check) —
    # they're protocol errors. Don't double-count.
    assert counters.hallucinations_dropped == 0


def test_validated_mentions_skips_non_dict_items():
    chunk = "the village of Edlingham"
    raw = ["a string", 42, None, {"form": "Edlingham", "context": "ok"}]
    out = _validated_mentions(raw, chunk)
    assert [m.form for m in out] == ["Edlingham"]


@pytest.mark.parametrize("sep", ["\n", "\r\n", "\t", "   "])
def test_validated_mentions_collapses_all_whitespace_in_context(sep):
    """All whitespace flavors (including \\r\\n from Windows-line-ended
    scholar transcripts) must collapse to a single space. test-coverage-
    reviewer round-1: earlier test only exercised plain \\n."""
    chunk = "Edlingham mentioned"
    raw = [{"form": "Edlingham", "context": f"line one{sep}line two{sep}line three"}]
    out = _validated_mentions(raw, chunk)
    assert "\n" not in out[0].context
    assert "\r" not in out[0].context
    assert "\t" not in out[0].context
    assert out[0].context == "line one line two line three"


# ---------- chunk_source_body --------------------------------------------


def test_chunk_source_body_empty_returns_empty_list():
    assert chunk_source_body("") == []
    assert chunk_source_body("   \n\n\t  ") == []


def test_chunk_source_body_single_short_body_one_chunk():
    body = "Single paragraph that fits in one chunk."
    chunks = chunk_source_body(body, target_chunk_size=20000)
    assert chunks == ["Single paragraph that fits in one chunk."]


def test_chunk_source_body_chunks_end_at_paragraph_boundaries():
    """Strong invariant: each emitted chunk must be a sequence of
    complete paragraphs (no chunk ends mid-paragraph). The earlier
    assertion ('rejoined contains substrings') passed even if the
    splitter returned the whole body as one chunk. pr-test-analyzer
    + test-coverage-reviewer round-1 finding."""
    body = "para one is short.\n\npara two is also short.\n\npara three completes."
    chunks = chunk_source_body(body, target_chunk_size=30)
    # No chunk should be a fragment of a paragraph — each chunk should
    # be one or more full paragraphs joined by \n\n.
    for c in chunks:
        # Every chunk's content must be reconstructible as concatenated
        # full paragraphs from the original.
        constituent = c.split("\n\n")
        for piece in constituent:
            assert piece in body, f"chunk fragment {piece!r} not a full paragraph"
    # Round-trip: every paragraph appears in exactly one chunk.
    rejoined = "\n\n".join(chunks)
    for para in ["para one is short.", "para two is also short.", "para three completes."]:
        assert rejoined.count(para) == 1


def test_chunk_source_body_single_long_paragraph_kept_as_one_chunk():
    """A single paragraph >target stays as one chunk (snap-to-paragraph
    invariant). The orchestrator surfaces this via log_warning so an
    operator can decide whether to risk LLM output truncation or split
    upstream."""
    body = "A" * 50000
    chunks = chunk_source_body(body, target_chunk_size=10000)
    assert chunks == [body]


def test_chunk_source_body_tolerates_whitespace_in_paragraph_boundaries():
    """OCR-derived scholar prose sometimes has \\n \\n or \\n\\t\\n
    between paragraphs (whitespace on the blank line). The regex
    split must still recognize these as paragraph boundaries.
    Gemini round-2 finding."""
    body = "para one.\n  \npara two.\n\t\npara three."
    chunks = chunk_source_body(body, target_chunk_size=15)
    # All three paragraphs must be present and chunked correctly.
    rejoined = " ".join(chunks)
    assert "para one" in rejoined
    assert "para two" in rejoined
    assert "para three" in rejoined
    # And the split must have produced multiple chunks (would be 1
    # if the regex failed and the splitter treated the whole body
    # as a single paragraph).
    assert len(chunks) >= 2


def test_chunk_source_body_packs_paragraphs_up_to_target():
    p_short = "short."
    body = "\n\n".join([p_short] * 100)
    chunks = chunk_source_body(body, target_chunk_size=100)
    # Per-chunk size: each chunk is one or more paragraphs joined by
    # \n\n; the longest packing should be <= target + one paragraph.
    max_allowed = 100 + len(p_short) + len("\n\n")
    assert all(len(c) <= max_allowed for c in chunks)
    # Total content preserved.
    rejoined = "\n\n".join(chunks)
    assert rejoined.count(p_short) == 100


# ---------- extract_toponym_mentions_from_chunk --------------------------


def test_extract_one_chunk_passes_schema_positionally():
    """Schema MUST be passed positionally, not as a kwarg. The three
    provider clients disagree on the kwarg name (AnthropicClient and
    OllamaClient use `schema`; GeminiClient uses `response_schema`) —
    passing as `schema=` would TypeError under --provider gemini.
    Gemini PR-#213 round-2 finding."""

    class StrictPositionalClient:
        """Verifies that the call uses 3 positional args (not kwarg)."""

        model = "strict-positional"

        def chat_json(self, *args, **kwargs):
            assert kwargs == {}, f"schema must be positional, got kwargs: {kwargs}"
            assert len(args) == 3, f"expected 3 positional args, got {len(args)}"
            system, user, schema = args
            assert isinstance(schema, dict), f"expected dict schema, got {type(schema)}"
            return {"mentions": [{"form": "Edlingham", "context": "x"}]}

    out = extract_toponym_mentions_from_chunk(StrictPositionalClient(), "Edlingham mentioned")
    assert [m.form for m in out] == ["Edlingham"]


def test_extract_one_chunk_happy_path_passes_chunk_and_schema_through():
    """The LLM call must receive (a) the chunk text embedded in the user
    prompt and (b) the RESPONSE_SCHEMA — a regression that swapped chunk
    for empty-string or dropped the schema arg would otherwise pass."""
    chunk = "Edlingham is in Northumberland."
    client = FakeClient(
        [
            {
                "mentions": [
                    {
                        "form": "Edlingham",
                        "date_year": None,
                        "region_hint": "Northumberland",
                        "context": "Edlingham is in Northumberland",
                    },
                ]
            }
        ]
    )
    out = extract_toponym_mentions_from_chunk(client, chunk)
    assert [m.form for m in out] == ["Edlingham"]
    # Verify the actual LLM call payload.
    assert len(client.calls) == 1
    system_arg, user_arg, schema_arg = client.calls[0]
    assert system_arg == SYSTEM_PROMPT
    assert chunk in user_arg
    assert schema_arg is not None  # RESPONSE_SCHEMA was passed
    assert schema_arg["type"] == "object"
    assert "mentions" in schema_arg["properties"]


@pytest.mark.parametrize(
    "response,match",
    [
        ("not a dict", "non-dict envelope"),
        ({}, "missing 'mentions' key"),
        ({"mentions": "should be a list"}, "mentions' was str"),
        ({"mentions": 42}, "mentions' was int"),
    ],
)
def test_extract_one_chunk_malformed_response_raises(response, match):
    """Structurally-invalid responses must raise so the orchestrator can
    count them as failed chunks. Returning [] silently identical to a
    chunk with zero mentions was a silent-failure-hunter round-1
    finding (operator sees '30/30 chunks OK, 0 mentions' even when the
    LLM was returning garbage)."""
    client = FakeClient([response])
    with pytest.raises(RuntimeError, match=match):
        extract_toponym_mentions_from_chunk(client, "Edlingham mentioned")


def test_extract_one_chunk_runtime_error_propagates():
    """Transport-level failures bubble up to the caller (the per-source
    orchestrator) which decides whether to skip the chunk or abort."""
    client = FakeClient([RuntimeError("Anthropic HTTP 429: rate limited")])
    with pytest.raises(RuntimeError, match="rate limited"):
        extract_toponym_mentions_from_chunk(client, "Edlingham")


def test_extract_one_chunk_counters_get_populated():
    """Counters must reflect hallucinations dropped and years clamped
    when supplied — otherwise the orchestrator can't roll them up."""
    chunk = "Edlingham was attested 1086"
    client = FakeClient(
        [
            {
                "mentions": [
                    {"form": "Edlingham", "date_year": 1086, "context": "x"},
                    # Hallucination + clamp.
                    {"form": "Eadwulfham", "date_year": "garbage", "context": "y"},
                    # Clamped year, valid form.
                    {"form": "Edlingham", "date_year": "garbage", "context": "z"},
                ]
            }
        ]
    )
    counters = ValidationCounters()
    out = extract_toponym_mentions_from_chunk(client, chunk, counters=counters)
    assert len(out) == 2
    assert counters.admitted == 2
    assert counters.hallucinations_dropped == 1
    assert counters.years_clamped == 1


# ---------- mine_toponym_mentions (orchestrator) -------------------------


def _three_chunk_body() -> str:
    """A deterministically 3-chunk body for orchestrator tests that need
    to pin response-count. pr-test-analyzer round-1 caught the earlier
    aggregates/continues-past-failure tests assuming exactly 3 chunks
    without verifying the splitter actually produces 3.

    Each paragraph is 8000 chars so target_chunk_size=10000 forces a
    flush after each (8000 + 2 + 8000 = 16002 > 10000). The trailing
    keyword (" Edlin"/" Wear"/" Tyne") is what the FakeClient's canned
    responses reference for the form-in-chunk substring check."""
    p1 = "A" * 7995 + " Edlin"  # 8001 chars
    p2 = "B" * 7996 + " Wear"  # 8001 chars
    p3 = "C" * 7996 + " Tyne"  # 8001 chars
    return f"{p1}\n\n{p2}\n\n{p3}"


def test_three_chunk_body_indeed_chunks_to_three():
    """Precondition for the orchestrator tests below — fail loudly here
    rather than letting the FakeClient's response-count mismatch leak."""
    assert len(chunk_source_body(_three_chunk_body(), target_chunk_size=10000)) == 3


def test_mine_toponym_mentions_aggregates_across_chunks():
    body = _three_chunk_body()
    # Replace the per-paragraph filler with a known form so the
    # form-in-chunk substring check passes.
    client = FakeClient(
        [
            {"mentions": [{"form": "Edlin", "context": "Edlin"}]},
            {"mentions": []},
            {"mentions": [{"form": "Tyne", "context": "Tyne"}]},
        ]
    )
    report = mine_toponym_mentions(client, "test_source", body, target_chunk_size=10000)
    assert isinstance(report, MineToponymMentionsReport)
    assert report.source_id == "test_source"
    assert report.chunks_processed == 3  # pinned, not "in"-based
    assert report.chunks_failed == 0
    assert [m.form for m in report.mentions] == ["Edlin", "Tyne"]


def test_mine_toponym_mentions_continues_past_chunk_failure():
    """One chunk failing (transport error) does NOT abort the run —
    the orchestrator skips the failed chunk and increments
    chunks_failed. Operator gets the chunk index + exception
    in the report."""
    body = _three_chunk_body()
    client = FakeClient(
        [
            {"mentions": [{"form": "Edlin", "context": "Edlin"}]},
            RuntimeError("Anthropic HTTP 503: service unavailable"),
            {"mentions": [{"form": "Tyne", "context": "Tyne"}]},
        ]
    )
    report = mine_toponym_mentions(client, "test_source", body, target_chunk_size=10000)
    assert report.chunks_failed == 1
    assert report.chunks_processed == 2
    forms = [m.form for m in report.mentions]
    assert "Edlin" in forms
    assert "Tyne" in forms
    # Failed-chunk detail surfaces the index + exception type.
    assert len(report.failed_chunks) == 1
    fc = report.failed_chunks[0]
    assert fc.index == 1
    assert "RuntimeError" in fc.error
    assert "503" in fc.error


def test_mine_toponym_mentions_empty_body_makes_no_llm_calls():
    """Cost-guard: empty body must not trigger a single LLM call.
    test-coverage-reviewer round-1 finding."""
    client = FakeClient([{"mentions": []}])  # would crash if called
    report = mine_toponym_mentions(client, "test_source", "")
    assert report.source_id == "test_source"
    assert report.chunks_processed == 0
    assert report.mentions == []
    assert client.calls == []  # no LLM invocations on empty input


def test_mine_toponym_mentions_progress_callback_fires_exact_count():
    """Callback fires exactly once per SUCCESSFUL chunk (not per failed
    chunk). pr-test-analyzer round-1: earlier assertion was len >= 1
    which a bug firing once outside the loop would pass."""
    body = _three_chunk_body()
    client = FakeClient(
        [
            {"mentions": [{"form": "Edlin", "context": "Edlin"}]},
            RuntimeError("transport"),  # chunk 1 fails → no callback
            {"mentions": [{"form": "Tyne", "context": "Tyne"}]},
        ]
    )
    calls: list[tuple[int, int, int]] = []
    report = mine_toponym_mentions(
        client,
        "test_source",
        body,
        target_chunk_size=10000,
        on_chunk_done=lambda d, t, m: calls.append((d, t, m)),
    )
    assert len(calls) == report.chunks_processed == 2
    # Total stays constant (post-limit chunk count, here 3).
    assert {t for (_, t, _) in calls} == {3}
    # Mentions count grows monotonically.
    mentions_seq = [m for (_, _, m) in calls]
    assert mentions_seq == sorted(mentions_seq)


def test_mine_toponym_mentions_limit_pass_through():
    """``limit`` must slice chunks AFTER chunking — without
    re-joining + re-chunking that could merge limited chunks back
    together. code-reviewer + silent-failure-hunter round-1 finding."""
    body = _three_chunk_body()
    # Only the first canned response will be consumed.
    client = FakeClient([{"mentions": [{"form": "Edlin", "context": "Edlin"}]}])
    report = mine_toponym_mentions(client, "test_source", body, target_chunk_size=10000, limit=1)
    assert report.chunks_processed == 1
    assert len(client.calls) == 1
    assert [m.form for m in report.mentions] == ["Edlin"]


def test_mine_toponym_mentions_oversized_paragraph_warns():
    """A single paragraph exceeding 1.5× target should emit a warning
    via ``log_warning``. silent-failure-hunter round-1: previously the
    8192-token cap silently truncated and the operator only saw
    'chunks_failed' with no diagnostic."""
    body = "Z" * 20000  # one huge paragraph, no \n\n
    client = FakeClient(
        [{"mentions": []}]  # in case the LLM is called
    )
    warnings: list[str] = []
    mine_toponym_mentions(
        client,
        "test_source",
        body,
        target_chunk_size=10000,  # 20000 > 10000 * 1.5
        log_warning=warnings.append,
    )
    assert any("oversized paragraph" in w for w in warnings)


def test_mine_toponym_mentions_report_has_independent_mentions_list():
    """field(default_factory=list) — two reports must NOT share the
    same list. The earlier None-default + __post_init__ hack avoided
    this by accident; the new dataclass shape locks it in. code-
    reviewer round-1 finding."""
    r1 = MineToponymMentionsReport(source_id="a")
    r2 = MineToponymMentionsReport(source_id="b")
    r1.mentions.append(ToponymMention(form="x", date_year=None, region_hint=None, context="y"))
    assert r2.mentions == []  # NOT contaminated by r1


def test_mine_toponym_mentions_rolls_up_validation_counters():
    """Per-chunk hallucinations + clamps must aggregate to the report
    so operators see the total failure mode breakdown across the
    source. silent-failure-hunter round-1 finding."""
    body = _three_chunk_body()
    client = FakeClient(
        [
            {
                "mentions": [
                    {"form": "Edlin", "date_year": 1086, "context": "x"},
                    {"form": "Hallucin", "context": "fake"},  # not in chunk
                    {"form": "Edlin", "date_year": "bad", "context": "y"},  # clamp
                ]
            },
            {"mentions": []},
            {"mentions": [{"form": "Tyne", "date_year": None, "context": "z"}]},
        ]
    )
    report = mine_toponym_mentions(client, "test_source", body, target_chunk_size=10000)
    assert report.hallucinations_dropped == 1
    assert report.years_clamped == 1


# ---------- CLI integration ----------------------------------------------


def test_cli_validates_missing_source_body(tmp_path, monkeypatch):
    """--source pointing at a non-existent .txt fails with a clear error."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions",
            "--source",
            "nonexistent_source",
            "--sources-dir",
            str(sources_dir),
        ],
    )
    assert result.exit_code != 0
    assert "source body not found" in result.output


def test_cli_path_traversal_is_neutralized_against_planted_file(tmp_path, monkeypatch):
    """Strong path-traversal test: plant a sensitive file OUTSIDE the
    sources_dir and confirm that ``--source ../planted`` doesn't read
    it. Path(...).name strips the traversal to bare 'planted', which
    doesn't exist inside sources_dir. code-reviewer round-1: earlier
    test couldn't distinguish neutralized-traversal from missing-file."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    # File OUTSIDE sources_dir — must not be reachable.
    planted_outside = tmp_path / "planted.txt"
    planted_outside.write_text("SECRET PLAINTEXT", encoding="utf-8")
    # Same-named file INSIDE sources_dir — Path(...).name traversal should
    # land on THIS one. If we read it back, traversal was neutralized.
    planted_inside = sources_dir / "planted.txt"
    planted_inside.write_text("INSIDE PROSE Edlingham mentioned.", encoding="utf-8")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")

    # Stub the AnthropicClient so we don't need a real API call. The
    # FakeClient just records what chunk was processed.
    fake = FakeClient([{"mentions": []}])
    import wyrd.generators.kenning.anthropic_extractor as ae_module

    monkeypatch.setattr(ae_module, "AnthropicClient", lambda **kw: fake)

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions",
            "--source",
            "../planted",
            "--sources-dir",
            str(sources_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    # The chunk that reached the LLM must be the INSIDE file's content,
    # not the outside one. Proves Path(...).name collapsed traversal.
    assert len(fake.calls) == 1
    user_arg = fake.calls[0][1]
    assert "INSIDE PROSE" in user_arg
    assert "SECRET PLAINTEXT" not in user_arg


def _stub_anthropic_for_cli(monkeypatch, fake_client: FakeClient) -> None:
    """Monkeypatch helper — the CLI imports AnthropicClient inside the
    function, so we patch the source module. If anyone moves that
    import to module-top, this stop mocking — leaving this guard as a
    visible coupling so the bug shows up loudly."""
    import wyrd.generators.kenning.anthropic_extractor as ae_module

    monkeypatch.setattr(ae_module, "AnthropicClient", lambda **kw: fake_client)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")


def test_cli_jsonl_output_multiple_mentions(tmp_path, monkeypatch):
    """End-to-end CLI test: 2 mentions across the JSONL output, with the
    expected key set on each line. fixture-data-reviewer round-1: the
    earlier single-mention test didn't verify line-per-mention shape."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    body_file = sources_dir / "stub_source.txt"
    body_file.write_text(
        "Edlingham is in Northumberland. Tynemouth is on the coast.\n",
        encoding="utf-8",
    )
    fake = FakeClient(
        [
            {
                "mentions": [
                    {
                        "form": "Edlingham",
                        "date_year": None,
                        "region_hint": "Northumberland",
                        "context": "Edlingham is in Northumberland",
                    },
                    {
                        "form": "Tynemouth",
                        "date_year": None,
                        "region_hint": None,
                        "context": "Tynemouth is on the coast",
                    },
                ]
            }
        ]
    )
    _stub_anthropic_for_cli(monkeypatch, fake)

    output_path = tmp_path / "out.jsonl"
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions",
            "--source",
            "stub_source",
            "--sources-dir",
            str(sources_dir),
            "--output",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, result.output
    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert {p["form"] for p in parsed} == {"Edlingham", "Tynemouth"}
    for p in parsed:
        assert p["source_id"] == "stub_source"
        assert set(p.keys()) == {"source_id", "form", "date_year", "region_hint", "context"}


def test_cli_preserves_diacritics_in_output(tmp_path, monkeypatch):
    """OE/ME place-name studies use æ, þ, ð — JSONL output must keep
    these as native UTF-8, not \\u-escape them. Gemini round-1
    finding (project convention is ensure_ascii=False)."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    body_file = sources_dir / "stub_source.txt"
    # Æþelstan-ham is a plausible OE form with both æ and þ.
    body_file.write_text("Æþelstanham mentioned in 1086.", encoding="utf-8")
    fake = FakeClient(
        [
            {
                "mentions": [
                    {
                        "form": "Æþelstanham",
                        "date_year": 1086,
                        "region_hint": None,
                        "context": "Æþelstanham mentioned",
                    }
                ]
            }
        ]
    )
    _stub_anthropic_for_cli(monkeypatch, fake)
    output_path = tmp_path / "out.jsonl"
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions",
            "--source",
            "stub_source",
            "--sources-dir",
            str(sources_dir),
            "--output",
            str(output_path),
        ],
    )
    assert result.exit_code == 0, result.output
    raw_text = output_path.read_text(encoding="utf-8")
    # ensure_ascii=False keeps the diacritics as bytes. With
    # ensure_ascii=True these would be escaped as \uXXXX. Check the
    # actual codepoints present in the form: Æ=U+00C6, þ=U+00FE.
    assert "Æþelstanham" in raw_text
    assert "\\u00c6" not in raw_text  # uppercase Æ escape
    assert "\\u00fe" not in raw_text  # thorn (þ) escape


def test_cli_refuses_to_overwrite_existing_output(tmp_path, monkeypatch):
    """Re-running with --output pointing at an existing file must error
    by default. --force opts in to overwriting. silent-failure-hunter
    round-1 finding (the pilot output is the only artifact — silently
    overwriting a full-source run is a foot-gun)."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "stub_source.txt").write_text("Edlingham mentioned.", encoding="utf-8")
    existing = tmp_path / "out.jsonl"
    existing.write_text("PREVIOUS RUN OUTPUT\n", encoding="utf-8")

    _stub_anthropic_for_cli(monkeypatch, FakeClient([{"mentions": []}]))

    # Without --force: should error.
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions",
            "--source",
            "stub_source",
            "--sources-dir",
            str(sources_dir),
            "--output",
            str(existing),
        ],
    )
    assert result.exit_code != 0
    assert "already exists" in result.output
    # Existing file unchanged.
    assert existing.read_text() == "PREVIOUS RUN OUTPUT\n"

    # With --force: succeeds and overwrites.
    _stub_anthropic_for_cli(monkeypatch, FakeClient([{"mentions": []}]))
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions",
            "--source",
            "stub_source",
            "--sources-dir",
            str(sources_dir),
            "--output",
            str(existing),
            "--force",
        ],
    )
    assert result.exit_code == 0, result.output
    # File overwritten with new (empty) content.
    assert "PREVIOUS RUN OUTPUT" not in existing.read_text()


def test_cli_summary_includes_validation_counters(tmp_path, monkeypatch):
    """TOTAL stderr line must include hallucinations_dropped and
    years_clamped so an operator running interactively sees the
    failure-mode breakdown without parsing the JSONL. silent-
    failure-hunter round-1 finding."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "stub.txt").write_text("Edlingham was attested.", encoding="utf-8")
    fake = FakeClient(
        [
            {
                "mentions": [
                    {"form": "Edlingham", "date_year": "bad", "context": "x"},
                    {"form": "Hallucination", "context": "fake"},
                ]
            }
        ]
    )
    _stub_anthropic_for_cli(monkeypatch, fake)
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions",
            "--source",
            "stub",
            "--sources-dir",
            str(sources_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "hallucinations_dropped=1" in result.output
    assert "years_clamped=1" in result.output


def test_cli_provider_gemini_routes_to_gemini_client(tmp_path, monkeypatch):
    """--provider gemini must instantiate GeminiClient. test-coverage-
    reviewer round-1 finding: only the anthropic branch had a test."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "stub.txt").write_text("Edlingham.", encoding="utf-8")
    fake = FakeClient([{"mentions": []}])

    import wyrd.generators.kenning.gemini_extractor as gemini_module

    monkeypatch.setattr(gemini_module, "GeminiClient", lambda **kw: fake)
    monkeypatch.setenv("GEMINI_API_KEY", "test-not-used")

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions",
            "--source",
            "stub",
            "--sources-dir",
            str(sources_dir),
            "--provider",
            "gemini",
        ],
    )
    assert result.exit_code == 0, result.output
    # The fake was instantiated and called.
    assert len(fake.calls) == 1


def test_cli_provider_ollama_routes_to_ollama_client(tmp_path, monkeypatch):
    """--provider ollama must instantiate OllamaClient. test-coverage-
    reviewer round-1 finding."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "stub.txt").write_text("Edlingham.", encoding="utf-8")
    fake = FakeClient([{"mentions": []}])

    import wyrd.generators.kenning.llm_extractor as llm_module

    monkeypatch.setattr(llm_module, "OllamaClient", lambda **kw: fake)

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions",
            "--source",
            "stub",
            "--sources-dir",
            str(sources_dir),
            "--provider",
            "ollama",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(fake.calls) == 1


def test_cli_invalid_utf8_warns_about_replacement_chars(tmp_path, monkeypatch):
    """If the source body contains invalid UTF-8 (replaced by U+FFFD),
    the CLI must warn — without the warning, mentions containing
    those characters get silently rejected by the form-in-chunk
    validator. silent-failure-hunter round-1 finding."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    bad_file = sources_dir / "stub.txt"
    # Write raw bytes with a Latin-1 sequence that's invalid UTF-8.
    bad_file.write_bytes(b"Edlingham \xe6\xfe mentioned")
    fake = FakeClient([{"mentions": []}])
    _stub_anthropic_for_cli(monkeypatch, fake)

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions",
            "--source",
            "stub",
            "--sources-dir",
            str(sources_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "invalid UTF-8" in result.output


# ---------- additional round-2/3 hardening -------------------------------


def test_mine_toponym_mentions_failed_chunks_cap(monkeypatch):
    """The failed_chunks list must cap at _FAILED_CHUNKS_CAP to avoid
    unbounded memory growth on a rate-limit storm — but ``chunks_failed``
    counter remains accurate. silent-failure-hunter round-2 finding."""
    from wyrd.generators.kenning import toponym_mention_extractor as tme

    # Lower the cap for test determinism.
    monkeypatch.setattr(tme, "_FAILED_CHUNKS_CAP", 3)

    # Construct a 5-chunk body so we exceed the cap.
    paragraphs = [("X" * 7995 + f" Park{i}") for i in range(5)]
    body = "\n\n".join(paragraphs)
    # Verify the splitter actually produces 5 chunks.
    assert len(chunk_source_body(body, target_chunk_size=10000)) == 5

    # Every chunk fails.
    client = FakeClient([RuntimeError(f"chunk-{i}") for i in range(5)])
    report = mine_toponym_mentions(client, "test_source", body, target_chunk_size=10000)
    # All 5 failures counted, but only 3 detail records kept.
    assert report.chunks_failed == 5
    assert len(report.failed_chunks) == 3
    # Indices of kept failures are the FIRST 3 (oldest = most actionable).
    assert [fc.index for fc in report.failed_chunks] == [0, 1, 2]


def test_mine_toponym_mentions_oversized_paragraph_no_warning_when_safe():
    """Negative control: a body whose paragraphs are well below the
    1.5× target should produce ZERO oversize warnings. Without this
    test, a bug emitting the warning for ALL chunks would still pass
    the existing positive-control test. pr-test-analyzer round-2."""
    safe_body = "Edlingham mentioned.\n\nTynemouth on the coast.\n\nWear river."
    client = FakeClient([{"mentions": []}])
    warnings: list[str] = []
    mine_toponym_mentions(
        client,
        "test_source",
        safe_body,
        target_chunk_size=10000,
        log_warning=warnings.append,
    )
    assert not any("oversized" in w for w in warnings)


def test_validated_mentions_synthesizes_context_for_softwrap_fallback():
    """When the LLM emits a soft-wrap-joined form (e.g. "Newcastle upon
    Tyne" for chunk "Newcastle upon\\nTyne") with empty context, the
    raw chunk.find(form) returns -1 — we must fall back to the
    collapsed-chunk window so operators get useful context. test-
    coverage round-2 finding."""
    chunk = "the borough of Newcastle upon\nTyne is found in early records"
    raw = [{"form": "Newcastle upon Tyne", "context": ""}]
    out = _validated_mentions(raw, chunk)
    assert len(out) == 1
    # Synthesized context comes from the COLLAPSED chunk view, so it
    # includes neighbors and is non-empty.
    assert out[0].context
    assert "Newcastle upon Tyne" in out[0].context
    assert out[0].context != "Newcastle upon Tyne"  # has neighbors


def test_cli_model_override_flows_to_client_factory(tmp_path, monkeypatch):
    """--model override must reach the AnthropicClient/GeminiClient/
    OllamaClient factory. test-coverage round-2: earlier provider-
    routing tests used `lambda **kw: fake` which silently discarded
    the model kwarg."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "stub.txt").write_text("Edlingham.", encoding="utf-8")

    captured_kwargs: list[dict] = []
    fake = FakeClient([{"mentions": []}])

    def recording_factory(**kw):
        captured_kwargs.append(kw)
        return fake

    import wyrd.generators.kenning.anthropic_extractor as ae_module

    monkeypatch.setattr(ae_module, "AnthropicClient", recording_factory)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-used")

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions",
            "--source",
            "stub",
            "--sources-dir",
            str(sources_dir),
            "--model",
            "claude-test-override-v9",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(captured_kwargs) == 1
    assert captured_kwargs[0]["model"] == "claude-test-override-v9"


def test_cli_force_overwrite_warns_to_stderr(tmp_path, monkeypatch):
    """With --force, the overwrite must produce a stderr warning so a
    typo'd re-run doesn't silently destroy a multi-hour pilot output.
    silent-failure-hunter round-2 finding."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "stub.txt").write_text("Edlingham.", encoding="utf-8")
    existing = tmp_path / "out.jsonl"
    existing.write_text("PREVIOUS\n", encoding="utf-8")

    _stub_anthropic_for_cli(monkeypatch, FakeClient([{"mentions": []}]))

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions",
            "--source",
            "stub",
            "--sources-dir",
            str(sources_dir),
            "--output",
            str(existing),
            "--force",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "overwriting" in result.output


def test_chunk_source_body_chunks_are_complete_paragraphs():
    """Stronger version of the paragraph-boundary test: each emitted
    chunk's constituent pieces (split by \\n\\n) must each appear as
    a complete paragraph in the body's split, not just as a substring.
    pr-test-analyzer round-2 finding: the earlier `piece in body`
    check would pass for any prefix of a paragraph."""
    body = "para one is short.\n\npara two is also short.\n\npara three completes."
    chunks = chunk_source_body(body, target_chunk_size=30)
    original_paragraphs = body.split("\n\n")
    for c in chunks:
        for piece in c.split("\n\n"):
            assert piece in original_paragraphs, (
                f"chunk fragment {piece!r} is not a complete paragraph"
            )
