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
from wyrd.generators.kenning.extractors.toponym_mentions import (
    _DATE_YEAR_MAX,
    _DATE_YEAR_MIN,
    RESPONSE_SCHEMA,
    RESPONSE_SCHEMA_GEMINI,
    SYSTEM_PROMPT,
    FailedChunk,
    MineToponymMentionsReport,
    ToponymMention,
    ValidationCounters,
    _coerce_year,
    _collapse_whitespace,
    _form_in_chunk,
    _sanitize_for_utf8,
    _validated_mentions,
    chunk_source_body,
    extract_toponym_mentions_from_chunk,
    mine_toponym_mentions,
    mine_toponym_mentions_from_chunks,
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


def test_form_in_chunk_matches_form_ending_in_non_word_char():
    """Lookbehind/lookahead instead of \\b allows forms whose first or
    last character is non-word (period, hyphen, apostrophe). The
    previous \\b-based check required word-class transition on both
    edges, silently rejecting abbreviated-form mentions like "F." or
    "St." emitted by the LLM verbatim."""
    chunk_collapsed = _collapse_whitespace("St. Bees is on the coast")
    assert _form_in_chunk("St.", chunk_collapsed)

    chunk_collapsed = _collapse_whitespace("notable forms: F. Eccleshall")
    assert _form_in_chunk("F.", chunk_collapsed)

    # And a form ending in a hyphen ("Stoke-") followed by a non-word
    # neighbor in the chunk.
    chunk_collapsed = _collapse_whitespace("the cluster Stoke- and -ton suffixes")
    assert _form_in_chunk("Stoke-", chunk_collapsed)


def test_form_in_chunk_lookbehind_still_rejects_word_boundary_violation():
    """The lookaround neighbor-is-non-word check must still reject a
    fragment landing INSIDE a longer word: "on" must not match in
    "upon", "Tyne" must not match in "Tyneside" if it would land
    mid-word — only at proper boundaries."""
    chunk_collapsed = _collapse_whitespace("upon the Tyneside")
    # "on" is at the END of "upon" — the next char is space (non-word),
    # but the previous char "p" is word — lookbehind (?<!\w) rejects.
    assert not _form_in_chunk("on", chunk_collapsed)
    # "Tyne" appears at START of "Tyneside" — previous char " " is
    # non-word, but the next char "s" is word — lookahead (?!\w) rejects.
    assert not _form_in_chunk("Tyne", chunk_collapsed)


# ---------- _coerce_year -------------------------------------------------


def test_coerce_year_accepts_in_range_int():
    assert _coerce_year(1086) == (1086, False)


@pytest.mark.parametrize("year", [_DATE_YEAR_MIN, _DATE_YEAR_MAX])
def test_coerce_year_accepts_exact_boundaries(year):
    """Both inclusive endpoints (700, 1700) must be accepted. A regression
    that flipped <= to < would slip past a test that only used midpoint
    values."""
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


@pytest.mark.parametrize(
    "raw",
    [
        "١٠٨٦",  # Arabic-Indic 1086
        "१०८६",  # Devanagari 1086
        "１０８６",  # fullwidth 1086
    ],
)
def test_coerce_year_rejects_unicode_digits(raw):
    """ASCII-only guard: ``\\d`` ALSO matches Unicode decimal digits, which
    Python ``int()`` then parses as a year (``int('١٠٨٦') == 1086``) — silently
    admitting the non-canonical/non-ASCII input this restriction exists to clamp.
    The pattern is ``[0-9]`` not ``\\d`` so these clamp like any other nonsense."""
    assert _coerce_year(raw) == (None, True)


def test_coerce_year_rejects_float():
    """JSON allows numeric values without subtype info — a provider
    that returns 1086.0 (json.loads on `"1086.0"` gives float) must
    not silently admit a non-integer year."""
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
    isn't actually in the chunk. earlier test used 'Birmingham' which shares zero domain overlap and
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
    this drops silently."""
    chunk = "the borough of Newcastle upon\nTyne is found in"
    raw = [{"form": "Newcastle upon Tyne", "context": "Newcastle upon Tyne is found"}]
    out = _validated_mentions(raw, chunk)
    assert [m.form for m in out] == ["Newcastle upon Tyne"]


def test_validated_mentions_in_range_year_survives():
    """Positive control — the year-clamping test must also confirm that
    an in-range year is preserved, otherwise a bug that nulls EVERY
    year passes the test."""
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
    context. earlier assertion
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


@pytest.mark.parametrize(
    "bad_form",
    [
        {"nested": "dict"},  # the exact 2026-05-23 production crash shape
        ["a", "list"],
        42,
        3.14,
        True,
    ],
)
def test_validated_mentions_drops_mention_with_non_str_form(bad_form):
    """wyrd-2iiy: under wyrd-sj50's format='json' (no server-side
    schema enforcement), the LLM occasionally emits non-string types
    for form. Before this guard, ``.strip()`` AttributeError'd and
    aborted the entire source mid-mining. Now the mention drops as a
    hallucination (the model invented a shape the schema would have
    forbidden) and counts toward the operator-visible total.

    None form is handled separately as a protocol error — see
    ``test_validated_mentions_drops_empty_form``.

    Parametrized over the types we've seen (dict, list, int, float,
    bool) plus the exact production crash shape (dict)."""
    chunk = "Edlingham is a place. Positive control mentioned here."
    raw = [
        {"form": bad_form, "context": "x"},
        {"form": "Edlingham", "context": "positive control"},
    ]
    counters = ValidationCounters()
    out = _validated_mentions(raw, chunk, counters=counters)
    # Positive control survives; bad-form mention drops as hallucination.
    assert [m.form for m in out] == ["Edlingham"]
    assert counters.hallucinations_dropped == 1
    assert counters.admitted == 1


def test_validated_mentions_accepts_mention_with_non_str_context():
    """wyrd-2iiy: non-str context isn't fatal — form is the load-
    bearing field. Treat non-str context as missing; let the
    synthesize-from-chunk fallback fill it. The mention is admitted
    with a derived context."""
    chunk = "Edlingham is a homestead in Northumberland."
    raw = [
        {"form": "Edlingham", "context": {"shape": "not a string"}},
    ]
    counters = ValidationCounters()
    out = _validated_mentions(raw, chunk, counters=counters)
    assert len(out) == 1
    assert out[0].form == "Edlingham"
    assert counters.admitted == 1
    # Context was synthesized from the chunk (not the dict).
    assert "Edlingham" in out[0].context
    assert out[0].context != "{'shape': 'not a string'}"


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


# ---------- _sanitize_for_utf8 (wyrd-8wa4) -------------------------------


@pytest.mark.parametrize("codepoint", [0xD800, 0xD8AA, 0xDAAA, 0xDBFF, 0xDC00, 0xDFFF])
def test_sanitize_for_utf8_replaces_lone_surrogates(codepoint):
    """The crash was specifically U+DAAA from gemma4:26b output. Cover the
    full surrogate range U+D800..U+DFFF (high + low halves) — Python str
    can hold any of these, but utf-8 encode rejects all of them."""
    s = f"Edling{chr(codepoint)}ham"
    cleaned = _sanitize_for_utf8(s)
    # Cleaned string must round-trip through utf-8 — that's the contract.
    cleaned.encode("utf-8")
    # The surrogate position becomes U+FFFD; surrounding chars survive.
    assert cleaned == "Edling�ham"


def test_sanitize_for_utf8_preserves_clean_strings():
    """Positive control: no-surrogate input must pass through unchanged
    (and identity-equal so a future caller can fast-path on `is`)."""
    s = "Newcastle upon Tyne"
    assert _sanitize_for_utf8(s) is s


def test_sanitize_for_utf8_preserves_legitimate_fffd():
    """A U+FFFD already in the input (e.g. from the source body being
    read with errors='replace') is itself a valid utf-8 codepoint and
    must pass through. We can't distinguish "LLM emitted bad data" from
    "OCR read bad bytes" at this point — both end up as U+FFFD downstream."""
    s = "Stoke �-on-Trent"
    assert _sanitize_for_utf8(s) == "Stoke �-on-Trent"


def test_sanitize_for_utf8_empty_string():
    assert _sanitize_for_utf8("") == ""


def test_validated_mentions_drops_form_with_surrogate_as_hallucination():
    """A lone-surrogate form, after sanitization to U+FFFD, almost
    never matches the source chunk (the chunk is read with
    errors='replace' so legitimate U+FFFD only appears at OCR-
    corruption sites; the LLM-introduced U+FFFD is unlikely to align).
    The existing hallucination-counter machinery drops it. AND the
    new surrogates_sanitized counter bumps so operators see the
    upstream model produced bad codepoints. This regression pins the
    wyrd-8wa4 crash: before the fix, this raw input crashed the
    writer; after, it's a counted hallucination plus a counted
    sanitization."""
    chunk = "Edlingham is the homestead of the sons of Eadwulf."
    raw = [
        {"form": "Edling\udaaaham", "context": "valid"},
        {"form": "Edlingham", "context": "valid positive control"},
    ]
    counters = ValidationCounters()
    out = _validated_mentions(raw, chunk, counters=counters)
    # Positive control survives; surrogate form is dropped.
    assert [m.form for m in out] == ["Edlingham"]
    assert counters.hallucinations_dropped == 1
    assert counters.admitted == 1
    # Surrogate-sanitize counter bumps even though the mention was
    # ultimately dropped — operator signal must reflect what the LLM
    # actually emitted, not what survived the validator.
    assert counters.surrogates_sanitized == 1
    # Round-trip the survivor through utf-8 to confirm no surrogates leak.
    for m in out:
        json.dumps(m.__dict__, ensure_ascii=False).encode("utf-8")


def test_validated_mentions_sanitizes_surrogate_in_context_admits_mention():
    """A surrogate in `context` (descriptive field) sanitizes to U+FFFD;
    the mention is still admitted with the cleaned context. Operators
    see the U+FFFD marker as a model-quality signal but don't lose the
    mention. The fix preserves utf-8-encodability end-to-end."""
    chunk = "Edlingham is the homestead"
    raw = [
        {"form": "Edlingham", "context": "context with \udaaa surrogate"},
    ]
    counters = ValidationCounters()
    out = _validated_mentions(raw, chunk, counters=counters)
    assert len(out) == 1
    assert counters.admitted == 1
    assert counters.surrogates_sanitized == 1
    assert "\udaaa" not in out[0].context
    assert "�" in out[0].context
    # Critical: the result must round-trip through utf-8 — that's the
    # write-path contract this fix restores.
    json.dumps(out[0].__dict__, ensure_ascii=False).encode("utf-8")


def test_validated_mentions_sanitizes_surrogate_in_region_hint():
    """A surrogate in `region_hint` (descriptive field) sanitizes to
    U+FFFD; the mention is admitted with the cleaned region. Same
    invariant as the context-field case but covers a separate field —
    we sanitize EACH str field, not just one."""
    chunk = "Edlingham mentioned"
    raw = [
        {
            "form": "Edlingham",
            "region_hint": "North\udaaaumberland",
            "context": "ok",
        },
    ]
    counters = ValidationCounters()
    out = _validated_mentions(raw, chunk, counters=counters)
    assert len(out) == 1
    assert counters.admitted == 1
    assert counters.surrogates_sanitized == 1
    assert out[0].region_hint == "North�umberland"
    json.dumps(out[0].__dict__, ensure_ascii=False).encode("utf-8")


def test_validated_mentions_surrogates_in_all_fields_bump_counter_three_times():
    """When the LLM emits surrogates in form AND region AND context of
    a single mention, the counter bumps once per affected field. The
    form's surrogate still drops the mention (form sanitized to U+FFFD
    almost never matches the chunk), but operator signal must still
    reflect that THREE fields needed sanitization. Otherwise a model
    regression that smears surrogates across every field looks
    identical to one that affects a single field — and the upstream
    diagnostic cost is much higher in the first case."""
    chunk = "Edlingham mentioned"
    raw = [
        {
            "form": "Edling\udaaaham",
            "region_hint": "North\ud800umberland",
            "context": "ctx\udfffstring",
        },
    ]
    counters = ValidationCounters()
    out = _validated_mentions(raw, chunk, counters=counters)
    assert out == []
    assert counters.hallucinations_dropped == 1
    assert counters.surrogates_sanitized == 3


def test_validated_mentions_admits_when_sanitized_form_matches_corrupted_chunk():
    """Edge case (pr-test-analyzer 5/10): the chunk was OCR-read with
    errors='replace', so a legitimate U+FFFD CAN appear at a corruption
    site. If the LLM faithfully reproduces the corruption (after sanitize,
    its form contains U+FFFD at the same position), the form-in-chunk
    check legitimately matches and the mention is ADMITTED — the LLM is
    correctly reflecting what's in the source, garbage character and all.
    Operators see U+FFFD in the form and know to treat the toponym as
    suspect. This pins admit (not drop) as the intended behavior for
    this case."""
    chunk = "Edling�ham is the homestead"  # OCR'd chunk has U+FFFD
    raw = [
        {"form": "Edling\udaaaham", "context": "Edling\udaaaham is the homestead"},
    ]
    counters = ValidationCounters()
    out = _validated_mentions(raw, chunk, counters=counters)
    assert len(out) == 1
    # Surrogate-sanitized form now matches the U+FFFD-bearing chunk.
    assert out[0].form == "Edling�ham"
    assert counters.admitted == 1
    assert counters.hallucinations_dropped == 0
    # Both form and context contained surrogates → 2 bumps.
    assert counters.surrogates_sanitized == 2
    json.dumps(out[0].__dict__, ensure_ascii=False).encode("utf-8")


def test_validated_mentions_clean_input_does_not_bump_sanitized_counter():
    """Negative control for the sanitization counter — without this, a
    bug that bumped the counter on EVERY mention would pass the other
    surrogate-counter assertions. The identity-fast-path is what
    distinguishes "we did real work" from "we sanitized nothing"."""
    chunk = "Edlingham is the homestead of the sons of Eadwulf."
    raw = [
        {"form": "Edlingham", "region_hint": "Durham", "context": "ok"},
    ]
    counters = ValidationCounters()
    out = _validated_mentions(raw, chunk, counters=counters)
    assert len(out) == 1
    assert counters.admitted == 1
    assert counters.surrogates_sanitized == 0


def test_mine_toponym_mentions_sanitizes_surrogate_in_failure_message():
    """The error string captured on FailedChunk MUST round-trip through
    a utf-8 file write — the staged miner persists failures to a JSONL
    sink, and a surrogate in the exception message would crash the
    same writer the in-band fix protects (wyrd-8wa4 second crash
    class). The fix sanitizes at FailedChunk construction; this test
    pins that contract. Without it, an LLM response excerpt embedded
    in a parse error (e.g. RuntimeError(f"bad JSON: {raw_body[:50]}"))
    re-introduces the crash one layer downstream."""
    body = _three_chunk_body()
    client = FakeClient(
        [
            {"mentions": [{"form": "Edlin", "context": "Edlin"}]},
            # Exception message itself contains a lone surrogate —
            # mimics a provider SDK that stringifies an LLM response
            # excerpt verbatim into its error message.
            RuntimeError("Anthropic parse failed near token: bad\udaaa"),
            {"mentions": [{"form": "Tyne", "context": "Tyne"}]},
        ]
    )
    report = mine_toponym_mentions(client, "test_source", body, target_chunk_size=10000)
    assert report.chunks_failed == 1
    fc = report.failed_chunks[0]
    # The persistence contract — the writer call:
    #   sink.write(json.dumps({"error": fc.error, ...}, ensure_ascii=False) + "\n")
    # — must not raise UnicodeEncodeError. Sanitizing at FailedChunk
    # construction is what makes that true.
    json.dumps({"error": fc.error}, ensure_ascii=False).encode("utf-8")
    # The surrogate becomes U+FFFD; the rest of the error message survives.
    assert "\udaaa" not in fc.error
    assert "RuntimeError" in fc.error
    assert "bad" in fc.error


def test_mine_toponym_mentions_rolls_up_surrogates_sanitized_across_chunks():
    """Pin the `report.surrogates_sanitized += chunk_counters.surrogates_sanitized`
    line — the chunk→report rollup is a separate code path from
    ValidationCounters.__iadd__ (it's a plain `int +=` on the report
    field). Without this test, a regression that drops the rollup
    line would pass every unit test (the iadd test exercises counter-
    to-counter folding, not the report's int field) but silently
    report 0 surrogates_sanitized at the source level even when the
    model emitted dozens. test-coverage-reviewer + pr-test-analyzer
    convergent finding."""
    body = _three_chunk_body()
    client = FakeClient(
        [
            {
                "mentions": [
                    {"form": "Edlin", "context": "ctx\udaaa chunk0"},
                ]
            },
            {"mentions": []},  # Clean middle chunk — positive control for
            # "rollup sums across, not last-chunk".
            {
                "mentions": [
                    {"form": "Tyne", "context": "ctx\ud800 chunk2"},
                ]
            },
        ]
    )
    report = mine_toponym_mentions(client, "test_source", body, target_chunk_size=10000)
    # Two distinct chunks each contributed one surrogate; rollup must SUM.
    assert report.surrogates_sanitized == 2
    # Positive control: mentions were still admitted (the surrogate was
    # in context, not form), so the rollup isn't being masked by
    # everything dropping as hallucination.
    assert len(report.mentions) == 2
    # Last-chunk-only regression would land here as 1 (whichever chunk
    # processed last). The assertion above (== 2) rules that out.


def test_cli_mine_toponym_mentions_emits_surrogates_in_summary(tmp_path, monkeypatch):
    """The CLI's TOTAL stderr line must surface the surrogates_sanitized
    counter — otherwise the operator running interactively has no way
    to spot a model emitting bad codepoints at scale. Pinning the
    literal format string is what catches typos (`surrogates_sanitised=`,
    `surrogates=`, missing the value) that pass unit tests because the
    field on the report is correct. Mirrors
    `test_cli_summary_includes_validation_counters` for the existing
    `hallucinations_dropped=` / `years_clamped=` surfaces."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "stub.txt").write_text("Edlingham was attested in the area.", encoding="utf-8")
    fake = FakeClient(
        [
            {
                "mentions": [
                    {"form": "Edlingham", "context": "ctx\udaaa with surrogate"},
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
    assert "surrogates_sanitized=1" in result.output


def test_validation_counters_iadd_folds_surrogates_sanitized():
    """Counters must fold via += across chunks — the rollup target is
    the per-source report. Without folding, the report's
    surrogates_sanitized would always be the last-chunk value."""
    acc = ValidationCounters(surrogates_sanitized=2, admitted=5)
    delta = ValidationCounters(surrogates_sanitized=3, admitted=1)
    acc += delta
    assert acc.surrogates_sanitized == 5
    assert acc.admitted == 6


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
    splitter returned the whole body as one chunk."""
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
    """OCR-derived scholar prose sometimes has "\\n  \\n" or "\\n\\t\\n"
    between paragraphs (horizontal whitespace on the blank line). The
    regex split must recognize these as paragraph boundaries. Exact
    count pinned — a regression that over-splits (e.g. on every
    whitespace character) would produce many tiny chunks, which a
    >=2 assertion wouldn't catch."""
    body = "para one.\n  \npara two.\n\t\npara three."
    chunks = chunk_source_body(body, target_chunk_size=15)
    assert len(chunks) == 3
    assert chunks[0] == "para one."
    assert chunks[1] == "para two."
    assert chunks[2] == "para three."


def test_chunk_source_body_does_not_split_on_single_newline():
    """A single newline (mid-paragraph soft-wrap) must NOT be a
    paragraph boundary — the regex requires \\n[ \\t]*\\n. The two
    halves of "para one" connected by a single \\n must end up in
    the SAME chunk (the soft-wrap didn't split them)."""
    body = "para one continues\nover two lines.\n\npara two."
    # target=15 forces flush after each paragraph; "para one continues..."
    # is one paragraph regardless of internal \n.
    chunks = chunk_source_body(body, target_chunk_size=15)
    assert len(chunks) == 2  # two real paragraphs
    # Both halves of para one in the same chunk: the soft-wrap was NOT
    # treated as a paragraph boundary.
    assert "continues" in chunks[0] and "two lines" in chunks[0]
    assert chunks[1] == "para two."


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
    passing as `schema=` would TypeError under --provider gemini."""

    class StrictPositionalClient:
        """Verifies that the call uses 3 positional args (not kwarg)."""

        model = "strict-positional"

        def chat_json(self, *args, **kwargs):
            assert kwargs == {}, f"schema must be positional, got kwargs: {kwargs}"
            assert len(args) == 3, f"expected 3 positional args, got {len(args)}"
            system, user, schema = args
            assert isinstance(schema, dict), f"expected dict schema, got {type(schema)}"
            return {"mentions": [{"form": "Edlingham", "context": "x"}]}

    out, _ = extract_toponym_mentions_from_chunk(StrictPositionalClient(), "Edlingham mentioned")
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
    out, _ = extract_toponym_mentions_from_chunk(client, chunk)
    assert [m.form for m in out] == ["Edlingham"]
    # Verify the actual LLM call payload.
    assert len(client.calls) == 1
    system_arg, user_arg, schema_arg = client.calls[0]
    assert system_arg == SYSTEM_PROMPT
    assert chunk in user_arg
    assert schema_arg is not None  # RESPONSE_SCHEMA was passed
    assert schema_arg["type"] == "object"
    assert "mentions" in schema_arg["properties"]


def test_extract_one_chunk_dispatches_openapi_dialect_to_openapi_clients():
    """wyrd-pe4g: a client whose ``schema_dialect`` is "openapi" (Gemini
    today; future OpenAPI-3.0-subset providers) must receive
    RESPONSE_SCHEMA_GEMINI. Direct-test against Gemini's API confirmed
    that the JSON Schema variant returns HTTP 400 on every chunk
    (rejected `additionalProperties` and `type: [..., "null"]` union).
    Real GeminiClient declares `schema_dialect = "openapi"` as a
    ClassVar; this test inlines the same contract on a stub."""

    class OpenApiClient:
        schema_dialect = "openapi"
        model = "fake-openapi-test"
        calls: list[dict | None] = []

        def chat_json(self, system, user, response_schema=None):
            # Param name matches real GeminiClient.chat_json (kwarg
            # `response_schema`, not `schema`) so the stub stays a
            # faithful fake — positional dispatch in production isn't
            # what motivates the parity; keyword-call compatibility
            # would be, if a future caller switched to kwargs.
            type(self).calls.append(response_schema)
            return {"mentions": [{"form": "Edlingham", "context": "Edlingham mentioned"}]}

    out, _ = extract_toponym_mentions_from_chunk(OpenApiClient(), "Edlingham mentioned")
    assert [m.form for m in out] == ["Edlingham"]
    assert len(OpenApiClient.calls) == 1
    assert OpenApiClient.calls[0] is RESPONSE_SCHEMA_GEMINI


def test_extract_one_chunk_dispatches_jsonschema_to_unmarked_clients():
    """Clients without a `schema_dialect` attribute (and clients that
    declare anything other than "openapi") get the standard JSON Schema
    variant. AnthropicClient + OllamaClient don't declare the attribute
    today and rely on this default."""
    client = FakeClient([{"mentions": [{"form": "Edlingham", "context": "Edlingham mentioned"}]}])
    out, _ = extract_toponym_mentions_from_chunk(client, "Edlingham mentioned")
    assert [m.form for m in out] == ["Edlingham"]
    assert len(client.calls) == 1
    _, _, schema_arg = client.calls[0]
    assert schema_arg is RESPONSE_SCHEMA


def test_extract_one_chunk_raises_on_unknown_schema_dialect():
    """wyrd-pe4g round-2: silent-failure-hunter finding. A typo'd
    ``schema_dialect`` value (e.g. "OpenAPI" instead of "openapi") on
    a Gemini-talking client would otherwise fall through to the
    default JSON Schema branch and ship the wrong dialect to Gemini's
    API, surfacing only as a generic HTTP 400. The whitelist guard
    raises ValueError at the dispatch site so the cause is obvious."""

    class TypoClient:
        schema_dialect = "OpenAPI"  # capitalization typo — Gemini wants lowercase

        def chat_json(self, system, user, response_schema=None):
            raise AssertionError("guard should fire before chat_json is reached")

    with pytest.raises(ValueError, match=r"unknown schema_dialect 'OpenAPI'"):
        extract_toponym_mentions_from_chunk(TypoClient(), "Edlingham mentioned")


def test_extract_one_chunk_dispatches_to_openapi_subclass():
    """Subclasses of an openapi-dialect client inherit the marker via
    MRO and must still receive RESPONSE_SCHEMA_GEMINI. silent-failure-
    hunter round-1 finding: the prior class-name dispatch silently
    routed subclasses to JSON Schema → HTTP 400 from Gemini."""

    class OpenApiBase:
        schema_dialect = "openapi"

    class InstrumentedGeminiClient(OpenApiBase):
        model = "wrapper-around-gemini"
        calls: list[dict | None] = []

        def chat_json(self, system, user, response_schema=None):
            type(self).calls.append(response_schema)
            return {"mentions": [{"form": "Edlingham", "context": "Edlingham mentioned"}]}

    extract_toponym_mentions_from_chunk(InstrumentedGeminiClient(), "Edlingham mentioned")
    assert InstrumentedGeminiClient.calls[0] is RESPONSE_SCHEMA_GEMINI


def test_response_schema_gemini_uses_openapi_dialect():
    """wyrd-pe4g: structural anchors so a future schema edit can't
    silently re-introduce JSON-Schema constructs that Gemini rejects.

    Gemini's responseSchema is OpenAPI-3.0-ish: uppercase type names,
    no `additionalProperties`, no `type: ["X", "null"]` unions (use
    `nullable: true` instead). The first two are documented HTTP-400
    modes from wyrd-pe4g's direct test against Gemini's API; the
    uppercase-type requirement is from Gemini's responseSchema spec
    (lowercase silently produces malformed structured output)."""
    schema = RESPONSE_SCHEMA_GEMINI

    def walk(node, *, path="$"):
        if isinstance(node, dict):
            assert "additionalProperties" not in node, (
                f"Gemini rejects additionalProperties (at {path})"
            )
            t = node.get("type")
            if isinstance(t, str):
                assert t.isupper(), f"Gemini wants uppercase types (got {t!r} at {path})"
            assert not isinstance(t, list), (
                f"Gemini rejects type unions; use nullable:true (at {path})"
            )
            for k, v in node.items():
                walk(v, path=f"{path}.{k}")
        elif isinstance(node, list):
            for i, item in enumerate(node):
                walk(item, path=f"{path}[{i}]")

    walk(schema)
    # Confirm the affirmative shape too — types are present and uppercase
    # where expected.
    assert schema["type"] == "OBJECT"
    assert schema["properties"]["mentions"]["type"] == "ARRAY"


def test_response_schema_gemini_nullable_fields_in_required():
    """Gemini convention (matches GEMINI_RESPONSE_SCHEMA in
    gemini_extractor.py for etymology): nullable fields must also
    appear in `required` so the model emits an explicit null rather
    than silently omitting the key. Without this, downstream
    consumers can't tell 'model declined to cite' from 'schema
    optional, model skipped.'"""
    item = RESPONSE_SCHEMA_GEMINI["properties"]["mentions"]["items"]
    nullable_fields = {
        name for name, spec in item["properties"].items() if spec.get("nullable") is True
    }
    assert {"date_year", "region_hint"}.issubset(nullable_fields)
    # The contract: every nullable field is in required.
    missing = nullable_fields - set(item["required"])
    assert not missing, f"nullable fields not in required: {missing}"


def test_response_schemas_have_matching_logical_shape():
    """The two schema variants describe the SAME logical output. A
    future field added to one but not the other (e.g. a new
    `confidence` slot landing only in the JSON Schema variant) would
    silently produce divergent outputs across providers. Anchor the
    fields + required set so a one-sided edit fails CI."""
    json_props = set(RESPONSE_SCHEMA["properties"]["mentions"]["items"]["properties"].keys())
    gem_props = set(RESPONSE_SCHEMA_GEMINI["properties"]["mentions"]["items"]["properties"].keys())
    assert json_props == gem_props, (
        f"schemas have different mention fields: json={json_props} gemini={gem_props}"
    )
    # Both must require the non-nullable fields (form + context).
    json_required = set(RESPONSE_SCHEMA["properties"]["mentions"]["items"]["required"])
    gem_required = set(RESPONSE_SCHEMA_GEMINI["properties"]["mentions"]["items"]["required"])
    assert {"form", "context"}.issubset(json_required)
    assert {"form", "context"}.issubset(gem_required)


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
    out, counters = extract_toponym_mentions_from_chunk(client, chunk)
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


def test_mine_toponym_mentions_propagates_unknown_dialect_value_error():
    """wyrd-pe4g round-3 silent-failure-hunter: the dispatch-time
    whitelist guard raises ValueError on a typo'd ``schema_dialect``,
    but the per-chunk bare-Exception catch in ``mine_toponym_mentions``
    would silently bucket every chunk into ``chunks_failed`` if
    ValueError weren't in ``_PROGRAMMER_ERROR_EXCEPTIONS``. The whole
    point of the round-2 guard is to make the misconfiguration loud;
    if a typo silently produced ``chunks_failed=N, chunks_processed=0,
    exit 0`` the guard would be a no-op in production (and the tiered
    fallback would mask it entirely). This test pins the propagation."""

    class TypoClient:
        schema_dialect = "Openapi"  # capitalization typo

        def chat_json(self, system, user, response_schema=None):
            raise AssertionError("guard should fire before chat_json is reached")

    with pytest.raises(ValueError, match=r"unknown schema_dialect 'Openapi'"):
        mine_toponym_mentions(TypoClient(), "test_source", _three_chunk_body())


def test_mine_toponym_mentions_empty_body_makes_no_llm_calls():
    """Cost-guard: empty body must not trigger a single LLM call."""
    client = FakeClient([{"mentions": []}])  # would crash if called
    report = mine_toponym_mentions(client, "test_source", "")
    assert report.source_id == "test_source"
    assert report.chunks_processed == 0
    assert report.mentions == []
    assert client.calls == []  # no LLM invocations on empty input


def test_mine_toponym_mentions_progress_callback_fires_exact_count():
    """Callback fires exactly once per SUCCESSFUL chunk (not per failed
    chunk). earlier assertion was len >= 1
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
    together."""
    body = _three_chunk_body()
    # Only the first canned response will be consumed.
    client = FakeClient([{"mentions": [{"form": "Edlin", "context": "Edlin"}]}])
    report = mine_toponym_mentions(client, "test_source", body, target_chunk_size=10000, limit=1)
    assert report.chunks_processed == 1
    assert len(client.calls) == 1
    assert [m.form for m in report.mentions] == ["Edlin"]


def test_mine_toponym_mentions_oversized_paragraph_warns():
    """A single paragraph exceeding 1.5× target should emit a warning
    via ``log_warning``. previously the
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
    source."""
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


# ---------- Staged-cascade hooks (wyrd-srd2) -----------------------------


def test_failed_chunk_carries_body_for_staged_resume():
    """wyrd-srd2 contract: the FailedChunk record returned in the
    report MUST include the chunk body verbatim — same text the LLM
    saw. Verify the captured body equals the actual middle chunk
    (not truncated, not the full source, not chunks 0/2). The
    staged-cascade CLI streams those bodies to a failures file so a
    downstream stage can re-process without re-chunking the source.
    R1 strengthening: previously asserted only non-empty/non-whitespace."""
    body = _three_chunk_body()
    expected_chunks = chunk_source_body(body, target_chunk_size=10000)
    assert len(expected_chunks) == 3, "fixture invariant — _three_chunk_body splits to 3"
    client = FakeClient(
        [
            {"mentions": [{"form": "Edlin", "context": "Edlin"}]},
            RuntimeError("Anthropic HTTP 503: service unavailable"),
            {"mentions": [{"form": "Tyne", "context": "Tyne"}]},
        ]
    )
    report = mine_toponym_mentions(client, "test_source", body, target_chunk_size=10000)
    assert len(report.failed_chunks) == 1
    fc = report.failed_chunks[0]
    # Verbatim equality, not "non-empty": pins truncation/placeholder regressions.
    assert fc.chunk_body == expected_chunks[1]


def test_on_chunk_failed_callback_fires_per_failure():
    """Verify the callback fires per failed chunk (the staged-cascade
    CLI relies on this to stream failures to disk as they happen, so
    failures aren't bounded by the in-memory FailedChunk cap)."""
    body = _three_chunk_body()
    expected_chunks = chunk_source_body(body, target_chunk_size=10000)
    client = FakeClient(
        [
            RuntimeError("err 1"),
            {"mentions": [{"form": "Edlin", "context": "Edlin"}]},
            RuntimeError("err 2"),
        ]
    )
    captured: list[FailedChunk] = []
    report = mine_toponym_mentions(
        client,
        "test_source",
        body,
        target_chunk_size=10000,
        on_chunk_failed=captured.append,
    )
    assert report.chunks_failed == 2
    assert len(captured) == 2
    assert captured[0].index == 0
    assert captured[1].index == 2
    # Verbatim equality: each captured chunk_body matches the chunk
    # at its index. Pins regressions where the callback might receive
    # an aliased reference to "current chunk" instead of the specific
    # failing chunk's body.
    assert captured[0].chunk_body == expected_chunks[0]
    assert captured[1].chunk_body == expected_chunks[2]


def test_on_chunk_failed_does_not_fire_on_success():
    """Negative control for the callback contract: when all chunks
    succeed, the failure callback never fires. Without this test, a
    regression that invoked on_chunk_failed for every chunk could pass
    test_on_chunk_failed_callback_fires_per_failure (length still
    matches the failure count if success paths happened to not append)."""
    body = _three_chunk_body()
    client = FakeClient(
        [
            {"mentions": [{"form": "Edlin", "context": "Edlin"}]},
            {"mentions": []},
            {"mentions": [{"form": "Tyne", "context": "Tyne"}]},
        ]
    )
    captured: list[FailedChunk] = []
    report = mine_toponym_mentions(
        client,
        "test_source",
        body,
        target_chunk_size=10000,
        on_chunk_failed=captured.append,
    )
    assert report.chunks_failed == 0
    assert captured == []


def test_on_chunk_failed_exception_propagates():
    """Pin the callback's failure semantics: if the operator's
    callback raises, the exception propagates out of mine_toponym_mentions
    rather than being silently swallowed mid-loop. The staged-cascade
    CLI relies on this — a write error on the failures-file sink should
    abort the run loudly, not silently lose failure records."""

    def boom(_fc: FailedChunk) -> None:
        raise OSError("disk full while writing failure record")

    body = _three_chunk_body()
    client = FakeClient(
        [
            RuntimeError("provider failure"),
            {"mentions": []},
            {"mentions": []},
        ]
    )
    with pytest.raises(OSError, match="disk full"):
        mine_toponym_mentions(
            client,
            "test_source",
            body,
            target_chunk_size=10000,
            on_chunk_failed=boom,
        )


def test_mine_toponym_mentions_from_chunks_all_fail_path():
    """Terminal-stage contract: when every retried chunk fails, the
    report shows chunks_processed=0, chunks_failed=N, mentions=[].
    Callback fires for each failure. The final cascade stage may hit
    this — operators need it to behave cleanly (no crash, no empty-
    mentions ambiguity)."""
    client = FakeClient(
        [
            RuntimeError("err 1"),
            RuntimeError("err 2"),
            RuntimeError("err 3"),
        ]
    )
    captured: list[FailedChunk] = []
    indexed = [(10, "chunk ten"), (20, "chunk twenty"), (30, "chunk thirty")]
    report = mine_toponym_mentions_from_chunks(
        client,
        "src",
        indexed,
        on_chunk_failed=captured.append,
    )
    assert report.chunks_processed == 0
    assert report.chunks_failed == 3
    assert report.mentions == []
    assert [fc.index for fc in captured] == [10, 20, 30]
    assert [fc.chunk_body for fc in captured] == ["chunk ten", "chunk twenty", "chunk thirty"]


def test_mine_toponym_mentions_from_chunks_preserves_indices():
    """Resume-from-failures (wyrd-srd2): a later stage re-runs only the
    failed chunks. The chunk indices come from the original chunking
    pass and must be preserved through the report — a chunk that was
    chunk 47 in stage 1 stays chunk 47 in the stage-2 failures file
    even when stage 2 sees only 6 chunks total."""
    client = FakeClient(
        [
            {"mentions": [{"form": "Bedlington", "context": "Bedlington"}]},
            {"mentions": [{"form": "Edlingham", "context": "Edlingham"}]},
        ]
    )
    # Original chunk indices 47 and 92 — non-contiguous, simulating
    # the cross-source failures-file shape after stage 1.
    indexed = [(47, "...Bedlington..."), (92, "...Edlingham...")]
    report = mine_toponym_mentions_from_chunks(client, "src", indexed)
    assert report.chunks_processed == 2
    assert {m.form for m in report.mentions} == {"Bedlington", "Edlingham"}
    # If a chunk fails here, its FailedChunk.index must match the
    # supplied index, not the iteration position.
    client2 = FakeClient([RuntimeError("upstream 500")])
    failed_chunks: list[FailedChunk] = []
    report2 = mine_toponym_mentions_from_chunks(
        client2,
        "src",
        [(102, "...Tyne...")],
        on_chunk_failed=failed_chunks.append,
    )
    assert report2.chunks_failed == 1
    assert failed_chunks[0].index == 102
    assert "...Tyne..." in failed_chunks[0].chunk_body


def test_mine_toponym_mentions_from_chunks_empty_input_is_noop():
    """Edge case: an empty failures file should produce an empty
    report with no LLM calls."""
    client = FakeClient([])
    report = mine_toponym_mentions_from_chunks(client, "src", [])
    assert report.chunks_processed == 0
    assert report.chunks_failed == 0
    assert report.mentions == []
    assert client.calls == []  # no LLM invocations


def test_mine_toponym_mentions_from_chunks_oversize_warning_fires():
    """The oversize-paragraph warning still fires in resume mode — a
    chunk that's bigger than target_chunk_size * multiplier should
    surface so operators see why a chunk might truncate on a deeper
    model. Same threshold logic as fresh-mining mode (shared helper)."""
    warnings: list[str] = []
    client = FakeClient([{"mentions": []}])
    # Oversize is 1.5× target_chunk_size by default (_OVERSIZED_PARAGRAPH_MULTIPLIER).
    big_chunk = "x" * 200
    mine_toponym_mentions_from_chunks(
        client,
        "src",
        [(0, big_chunk)],
        target_chunk_size=100,  # 1.5× = 150, so 200 > 150 triggers the warning
        log_warning=warnings.append,
    )
    assert any("oversized paragraph" in w for w in warnings)


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
    doesn't exist inside sources_dir. earlier
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
    import wyrd.generators.kenning.extractors.anthropic as ae_module

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
    import wyrd.generators.kenning.extractors.anthropic as ae_module

    monkeypatch.setattr(ae_module, "AnthropicClient", lambda **kw: fake_client)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")


def test_cli_jsonl_output_multiple_mentions(tmp_path, monkeypatch):
    """End-to-end CLI test: 2 mentions across the JSONL output, with the
    expected key set on each line. earlier single-mention test didn't verify line-per-mention shape."""
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
    # Mark the fake with the same dispatch marker the real GeminiClient
    # carries (wyrd-pe4g) so the CLI end-to-end exercises the openapi
    # branch of extract_toponym_mentions_from_chunk's dispatch.
    fake.schema_dialect = "openapi"

    import wyrd.generators.kenning.extractors.gemini as gemini_module

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
    # The fake was instantiated and called with the openapi-dialect schema.
    assert len(fake.calls) == 1
    _, _, schema_arg = fake.calls[0]
    assert schema_arg is RESPONSE_SCHEMA_GEMINI


def test_extract_toponym_mentions_from_chunk_sends_openapi_schema_to_gemini_wire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """wyrd-9air: end-to-end pin of the wire-level Gemini contract.

    test_cli_provider_gemini_routes_to_gemini_client (above) uses a
    FakeClient and asserts the dispatch picks RESPONSE_SCHEMA_GEMINI by
    object identity. That pins the choice at the dispatch site but
    bypasses the real GeminiClient.chat_json HTTP path — a bug that
    corrupted the schema after dispatch (inside chat_json itself, or
    in how the schema gets serialized into the request body) would
    not be caught by the FakeClient test. This test exercises the
    real GeminiClient through a mocked urlopen and asserts the wire
    payload's ``generationConfig.responseSchema`` carries the OpenAPI
    dialect (uppercase types, no additionalProperties, nullable:
    true), so a regression in either dispatch or wire-encoding trips
    CI here.
    """
    from unittest.mock import patch

    from wyrd.generators.kenning.extractors.gemini import GeminiClient

    monkeypatch.setenv("GEMINI_API_KEY", "test-not-used")
    calls: list[bytes] = []
    inner_payload = json.dumps({"mentions": []})
    envelope = json.dumps(
        {"candidates": [{"content": {"parts": [{"text": inner_payload}]}}]}
    ).encode("utf-8")

    class _Resp:
        def read(self) -> bytes:
            return envelope

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            pass

    def capturing_urlopen(req, timeout=None):
        calls.append(req.data)
        return _Resp()

    client = GeminiClient(model="gemini-2.5-flash")
    with patch("urllib.request.urlopen", capturing_urlopen):
        result, _ = extract_toponym_mentions_from_chunk(client, "Edlingham was held.")

    assert result == []
    # Pin call count so a regression that produces spurious retries
    # (or a double-dispatch in chat_json) doesn't silently pass with
    # only the last attempt's body captured — the wire contract is
    # exactly-one-request per chunk.
    assert len(calls) == 1, f"urlopen called {len(calls)} times, expected 1"
    sent = json.loads(calls[0].decode("utf-8"))
    schema = sent["generationConfig"]["responseSchema"]

    # Top-level shape is the Gemini OpenAPI dialect.
    assert schema["type"] == "OBJECT", "wire schema must use uppercase OpenAPI types"
    assert "additionalProperties" not in schema, (
        "Gemini rejects additionalProperties at schema-submit time; if this key "
        "is present, the dispatch is sending RESPONSE_SCHEMA (JSON Schema) "
        "instead of RESPONSE_SCHEMA_GEMINI"
    )

    # Item schema carries the dialect markers: uppercase types, no
    # additionalProperties, nullable: true (not type: [..., 'null']).
    item = schema["properties"]["mentions"]["items"]
    assert item["type"] == "OBJECT"
    assert "additionalProperties" not in item
    date_year_schema = item["properties"]["date_year"]
    assert date_year_schema.get("nullable") is True
    assert date_year_schema["type"] == "INTEGER", (
        "date_year must be OpenAPI nullable INTEGER, not JSON Schema type: [int, null]"
    )
    assert not isinstance(date_year_schema["type"], list)


def test_cli_provider_ollama_routes_to_ollama_client(tmp_path, monkeypatch):
    """--provider ollama must instantiate OllamaClient. test-coverage-
    reviewer round-1 finding."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "stub.txt").write_text("Edlingham.", encoding="utf-8")
    fake = FakeClient([{"mentions": []}])

    import wyrd.generators.kenning.extractors.llm as llm_module

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
    validator."""
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


def test_mine_toponym_mentions_failed_chunks_cap_keeps_head_and_tail(monkeypatch):
    """failed_chunks caps via head+tail retention. chunks_failed stays
    accurate; only the detail list is bounded. Half-and-half keeps the
    deterministic config errors at run start (auth, schema mismatch)
    AND the most recent transient failures (rate-limit storm, mid-run
    model deprecation, deep-paragraph JSON truncation). Pure keep-
    first would drop the diagnostic-rich tail; pure keep-last would
    erase the auth/config head."""
    from wyrd.generators.kenning import toponym_mention_extractor as tme

    # Lower head + tail for test determinism. head=1 + tail=2 = cap 3.
    monkeypatch.setattr(tme, "_FAILED_CHUNKS_HEAD", 1)
    monkeypatch.setattr(tme, "_FAILED_CHUNKS_TAIL", 2)

    # 5-chunk body — 2 more failures than head+tail capacity.
    paragraphs = [("X" * 7995 + f" Park{i}") for i in range(5)]
    body = "\n\n".join(paragraphs)
    assert len(chunk_source_body(body, target_chunk_size=10000)) == 5

    # Every chunk fails with a distinguishable error string.
    client = FakeClient([RuntimeError(f"chunk-{i}") for i in range(5)])
    report = mine_toponym_mentions(client, "test_source", body, target_chunk_size=10000)

    assert report.chunks_failed == 5  # counter accurate
    assert len(report.failed_chunks) == 3  # head + tail capacity
    # Head: first failure (index 0). Tail: last 2 (indices 3 and 4).
    # Middle failure (index 2) is elided.
    assert [fc.index for fc in report.failed_chunks] == [0, 3, 4]


def test_mine_toponym_mentions_failed_chunks_first_goes_to_head(monkeypatch):
    """Pins the head/tail dispatch invariant: the FIRST failure must
    land in the head bucket regardless of tail capacity. A `<=` for `<`
    flip at the dispatch site would silently push the first failure
    to the tail and re-shape the retention pattern."""
    from wyrd.generators.kenning import toponym_mention_extractor as tme

    # head=2, tail=2 — exactly 4 failures fits without elision; the
    # head must hold the first 2 (indices 0, 1) not the last 2.
    monkeypatch.setattr(tme, "_FAILED_CHUNKS_HEAD", 2)
    monkeypatch.setattr(tme, "_FAILED_CHUNKS_TAIL", 2)

    paragraphs = [("X" * 7995 + f" Park{i}") for i in range(4)]
    body = "\n\n".join(paragraphs)
    assert len(chunk_source_body(body, target_chunk_size=10000)) == 4

    client = FakeClient([RuntimeError(f"chunk-{i}") for i in range(4)])
    report = mine_toponym_mentions(client, "test_source", body, target_chunk_size=10000)

    assert report.chunks_failed == 4
    # All 4 fit (head 2 + tail 2 = exactly 4) — no elision.
    assert [fc.index for fc in report.failed_chunks] == [0, 1, 2, 3]


def test_mine_toponym_mentions_oversized_paragraph_no_warning_when_safe():
    """Negative control: a body whose paragraphs are well below the
    1.5× target should produce ZERO oversize warnings. Without this
    test, a bug emitting the warning for ALL chunks would still pass
    the existing positive-control test."""
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
    raw-chunk lookup fails — we must fall back to the collapsed-chunk
    window so operators get useful context."""
    chunk = "the borough of Newcastle upon\nTyne is found in early records"
    raw = [{"form": "Newcastle upon Tyne", "context": ""}]
    out = _validated_mentions(raw, chunk)
    assert len(out) == 1
    # Synthesized context comes from the COLLAPSED chunk view, so it
    # includes neighbors and is non-empty.
    assert out[0].context
    assert "Newcastle upon Tyne" in out[0].context
    assert out[0].context != "Newcastle upon Tyne"  # has neighbors


def test_validated_mentions_synthesized_context_respects_word_boundary():
    """Context synthesis must use word-boundary matching (not raw
    substring find) — otherwise a form like "on" emitted with empty
    context could be located INSIDE "upon" or "Northampton", giving
    the operator a misleading window centered on the wrong position.
    Same hallucination hazard the form-in-chunk guard prevents."""
    # Position "on" inside "upon" far from the standalone "on" word —
    # so the ±40-char window on each location is disjoint. If the
    # synthesizer used substring find(), the window would center on
    # the FIRST "on" (inside "upon") and contain "upon"+surrounding.
    # With word-boundary matching it lands on the standalone "on" word,
    # whose window is "river bend"-centric.
    prefix = "x" * 100  # pad so the two "on"s are far apart
    chunk = (
        prefix
        + "Hexham upon a hill, "
        + "y" * 100  # more padding
        + "north of camp, on the river bend, with shrines."
    )
    raw = [{"form": "on", "context": ""}]
    out = _validated_mentions(raw, chunk)
    assert len(out) == 1
    # Synthesized context must center on the STANDALONE "on" (near "river"),
    # not the in-word "on" inside "upon" (near "Hexham"/"hill").
    assert "river" in out[0].context
    # The "upon" portion of the chunk is now 100+ chars away — would
    # only appear in the context if the synthesizer matched there.
    assert "upon" not in out[0].context


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

    import wyrd.generators.kenning.extractors.anthropic as ae_module

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


def test_cli_ollama_url_flows_to_client_factory(tmp_path, monkeypatch):
    """--ollama-url reaches the OllamaClient base_url kwarg. Matches
    the tiered command's flag for operator consistency: same syntax
    works in both `mine-toponym-mentions` and
    `mine-toponym-mentions-tiered`."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "stub.txt").write_text("Edlingham.", encoding="utf-8")

    captured: list[dict] = []
    fake = FakeClient([{"mentions": []}])

    def recording_factory(**kw):
        captured.append(kw)
        return fake

    import wyrd.generators.kenning.extractors.llm as llm_module

    monkeypatch.setattr(llm_module, "OllamaClient", recording_factory)

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
            "--ollama-url",
            "http://hades:11434",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(captured) == 1
    assert captured[0]["base_url"] == "http://hades:11434"


def test_cli_force_overwrite_warns_to_stderr(tmp_path, monkeypatch):
    """With --force, the overwrite warning must appear on STDERR
    specifically (per project convention: progress lines go to stderr;
    JSONL data goes to stdout). A regression that wrote the warning to
    stdout would still satisfy the combined-output check, but it would
    pollute the operator's JSONL pipeline silently."""
    # mix_stderr=False keeps result.stdout / result.stderr separate.
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
    assert result.exit_code == 0, result.stderr
    # Warning specifically on stderr — would be on stdout if regression.
    assert "overwriting" in result.stderr
    assert "overwriting" not in result.stdout


def test_cli_force_without_existing_output_does_not_warn(tmp_path, monkeypatch):
    """Negative control: --force on a path that DOESN'T exist must
    NOT emit an "overwriting" warning. Without this, a regression
    that fires the warning unconditionally would pass the positive
    test."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "stub.txt").write_text("Edlingham.", encoding="utf-8")
    fresh_output = tmp_path / "fresh.jsonl"  # does NOT exist yet
    assert not fresh_output.exists()

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
            str(fresh_output),
            "--force",  # specified, but file doesn't exist
        ],
    )
    assert result.exit_code == 0, result.stderr
    assert "overwriting" not in result.stderr


# ---------- mine-toponym-mentions-staged (wyrd-srd2) ---------------------


def _read_jsonl(path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def test_cli_staged_fresh_mining_writes_per_source_with_extractor_tag(tmp_path, monkeypatch):
    """Fresh-mining mode (no --from-failures): walks sources_dir, writes
    one JSONL per source to --output-dir, each row tagged with the
    `extractor` so cascade-stage attribution survives in the data."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "alpha.txt").write_text("Edlingham is in Northumberland.", encoding="utf-8")
    (sources_dir / "beta.txt").write_text("Tyne is a river.", encoding="utf-8")
    output_dir = tmp_path / "out"

    _stub_anthropic_for_cli(
        monkeypatch,
        FakeClient(
            [
                {"mentions": [{"form": "Edlingham", "context": "Edlingham is in Northumberland"}]},
                {"mentions": [{"form": "Tyne", "context": "Tyne is a river"}]},
            ]
        ),
    )

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-staged",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(output_dir),
            "--provider",
            "anthropic",
            "--chunk-size",
            "5000",
        ],
    )
    assert result.exit_code == 0, result.output

    alpha_rows = _read_jsonl(output_dir / "alpha.jsonl")
    beta_rows = _read_jsonl(output_dir / "beta.jsonl")
    assert len(alpha_rows) == 1
    assert alpha_rows[0]["form"] == "Edlingham"
    assert alpha_rows[0]["extractor"].startswith("anthropic:")
    assert beta_rows[0]["form"] == "Tyne"


def test_cli_staged_fresh_mining_skips_atomic_write_when_all_chunks_fail(tmp_path, monkeypatch):
    """wyrd-st1r: when every chunk of a source fails (e.g. provider
    unreachable for the duration), the staged miner must NOT write a
    0-byte canonical file. Otherwise --skip-existing on the next run
    treats the source as 'done' and the 0 mentions become the
    permanent record. Concrete 2026-05-22 failure: gemma4 Ollama lost
    power mid-run; 12 sources promoted as 0-byte files and were
    silently skipped on restart until a manual file-size audit caught
    them.

    The guard fires only when (chunks_processed == 0 AND no pre-
    existing canonical file). A partial run (some chunks succeeded)
    still writes — partial captures are valuable. A pre-existing
    canonical file is preserved via the existing-rows copy path."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "alpha.txt").write_text("Edlingham is in Northumberland.", encoding="utf-8")
    output_dir = tmp_path / "out"

    # FakeClient raises on EVERY chunk → chunks_processed == 0,
    # report.mentions == [].
    _stub_anthropic_for_cli(
        monkeypatch,
        FakeClient([RuntimeError("provider unreachable")]),
    )

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-staged",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(output_dir),
            "--provider",
            "anthropic",
            "--chunk-size",
            "5000",
        ],
    )
    assert result.exit_code == 0, result.output

    # The canonical file must NOT exist — operator's next
    # --skip-existing run must re-try this source.
    canonical = output_dir / "alpha.jsonl"
    assert not canonical.exists(), (
        f"canonical file was written despite 0 successful chunks; "
        f"--skip-existing would silently skip the source on retry. "
        f"file: {canonical}"
    )

    # The CLI must surface the skip explicitly so operators see the
    # gap rather than discovering it on a later file-size audit.
    assert "no canonical file written" in result.output
    assert "alpha" in result.output


def test_cli_staged_fresh_mining_writes_partial_when_some_chunks_succeed(tmp_path, monkeypatch):
    """wyrd-st1r positive control: partial-success runs (some chunks
    landed mentions, some failed) STILL produce a canonical file
    with the partial captures. The guard from the preceding test
    must not regress this case — partial mining data is the load-
    bearing wyrd-w7i3 contract."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    body = (
        "Edlingham is in Northumberland.\n\n"
        "Tyne is a river.\n\n"
        "Durham is a city.\n\n"
        "Wear is also a river."
    )
    (sources_dir / "alpha.txt").write_text(body, encoding="utf-8")
    output_dir = tmp_path / "out"

    # Mix of success + failure across chunks.
    _stub_anthropic_for_cli(
        monkeypatch,
        FakeClient(
            [
                {"mentions": [{"form": "Edlingham", "context": "Edlingham is in"}]},
                RuntimeError("transient"),
                {"mentions": [{"form": "Durham", "context": "Durham is a city"}]},
                RuntimeError("transient"),
            ]
        ),
    )

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-staged",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(output_dir),
            "--provider",
            "anthropic",
            "--chunk-size",
            "30",
        ],
    )
    assert result.exit_code == 0, result.output

    canonical = output_dir / "alpha.jsonl"
    assert canonical.exists()
    rows = _read_jsonl(canonical)
    assert len(rows) == 2
    assert {r["form"] for r in rows} == {"Edlingham", "Durham"}


def test_cli_staged_fresh_mining_empty_source_does_not_trigger_outage_guard(tmp_path, monkeypatch):
    """wyrd-st1r predicate refinement: an empty source body produces
    chunks_processed=0 AND chunks_failed=0 AND no mentions — same
    surface shape as a 100%-outage run. Without the chunks_failed>0
    term in the guard predicate, the empty source would hit the
    "will retry on next run" path, creating an infinite retry loop
    on every subsequent --skip-existing invocation. This test pins
    the distinction: empty source writes an empty canonical (source
    is genuinely done), outage does NOT write (source needs retry)."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    # Whitespace-only body — chunk_source_body returns [] for this,
    # so no chunks are attempted (chunks_failed stays 0).
    (sources_dir / "empty.txt").write_text("   \n\n\t  \n", encoding="utf-8")
    output_dir = tmp_path / "out"

    _stub_anthropic_for_cli(monkeypatch, FakeClient([]))

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-staged",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(output_dir),
            "--provider",
            "anthropic",
            "--chunk-size",
            "5000",
        ],
    )
    assert result.exit_code == 0, result.output

    # Empty source produces an empty canonical file — the source IS
    # done; no useful work to retry. (Contrast with the outage case,
    # where chunks_failed>0 leaves no canonical.)
    canonical = output_dir / "empty.jsonl"
    assert canonical.exists(), (
        "empty source body must produce a canonical file (even if "
        "empty) so --skip-existing doesn't retry it forever"
    )
    assert canonical.read_text(encoding="utf-8") == ""
    # The "will retry" warning must NOT fire for empty sources.
    assert "will retry on the next run" not in result.output


def test_cli_staged_from_failures_skips_atomic_write_when_all_chunks_fail(tmp_path, monkeypatch):
    """wyrd-st1r resume-path coverage: the guard exists in BOTH
    _run_fresh_mining AND _run_resume_from_failures. The resume
    path is the operator's primary recovery surface — they run
    --from-failures after a provider outage. A future refactor
    that unifies the two paths could silently regress the resume
    guard without this test. (test-coverage-reviewer + pr-test-
    analyzer convergent finding on PR #327 round 1.)

    Scenario: failures.jsonl has 1 record; the retry client raises
    on every call; assert no canonical written, "no canonical file
    written" surfaces, inprogress cleaned up."""
    runner = CliRunner()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    failures_file = tmp_path / "failures.jsonl"
    failures_file.write_text(
        json.dumps(
            {
                "source_id": "alpha",
                "chunk_index": 0,
                "chunk_body": "Edlingham is in Northumberland.",
                "error": "RuntimeError: original 503",
                "extractor": "ollama:gemma4:26b",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    # The retry client raises on the single chunk being retried.
    _stub_anthropic_for_cli(
        monkeypatch,
        FakeClient([RuntimeError("retry also down")]),
    )

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-staged",
            "--from-failures",
            str(failures_file),
            "--output-dir",
            str(output_dir),
            "--provider",
            "anthropic",
        ],
    )
    assert result.exit_code == 0, result.output

    # No canonical written — operator's next run will retry alpha.
    canonical = output_dir / "alpha.jsonl"
    assert not canonical.exists(), (
        f"resume path produced a canonical despite 0 successful chunks; "
        f"--skip-existing on a follow-up run would silently skip alpha. "
        f"file: {canonical}"
    )
    assert "no canonical file written" in result.output
    assert "alpha" in result.output
    # Inprogress cleaned up — outage_with_no_progress path sets
    # canonical_completed=True which frees the unlink.
    inprogress_dir = output_dir / "_inprogress"
    if inprogress_dir.exists():
        leftover = list(inprogress_dir.iterdir())
        assert not leftover, f"inprogress leftover after outage skip: {leftover}"


def test_cli_staged_from_failures_preserves_existing_canonical_when_all_chunks_fail(
    tmp_path, monkeypatch
):
    """wyrd-st1r resume-path coverage: when a canonical already
    exists AND all retried chunks fail, the existing canonical
    MUST be preserved byte-identical. This is the common operator
    flow: resume after a partial-success run that left some
    chunks in failures.jsonl; the retry also fails for those
    chunks. The prior good data must not be destroyed.

    pr-test-analyzer + silent-failure-hunter both flagged that the
    no-write skip-write path leaves existing canonicals untouched —
    test pins that contract."""
    runner = CliRunner()
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    # Pre-existing canonical with 2 mentions from a prior run.
    canonical = output_dir / "alpha.jsonl"
    prior_rows = [
        {
            "source_id": "alpha",
            "form": "Edlingham",
            "date_year": None,
            "region_hint": None,
            "context": "Edlingham is in",
            "extractor": "ollama:gemma4:26b",
        },
        {
            "source_id": "alpha",
            "form": "Durham",
            "date_year": None,
            "region_hint": None,
            "context": "Durham is a city",
            "extractor": "ollama:gemma4:26b",
        },
    ]
    canonical.write_text(
        "\n".join(json.dumps(r) for r in prior_rows) + "\n",
        encoding="utf-8",
    )
    canonical_before = canonical.read_text(encoding="utf-8")
    # mtime check pins that the file isn't touched at all — a
    # rewrite-from-itself in the else branch would produce
    # byte-identical contents (the verbatim-copy is faithful) but
    # bump mtime. Without this assertion, a predicate inversion that
    # accidentally took the else branch would still pass the
    # byte-equality check. pr-test-analyzer R2 criticality-6 finding.
    mtime_before_ns = canonical.stat().st_mtime_ns

    failures_file = tmp_path / "failures.jsonl"
    failures_file.write_text(
        json.dumps(
            {
                "source_id": "alpha",
                "chunk_index": 5,
                "chunk_body": "Wear is a river.",
                "error": "RuntimeError: original 503",
                "extractor": "ollama:gemma4:26b",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    # Retry also fails.
    _stub_anthropic_for_cli(
        monkeypatch,
        FakeClient([RuntimeError("retry also down")]),
    )

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-staged",
            "--from-failures",
            str(failures_file),
            "--output-dir",
            str(output_dir),
            "--provider",
            "anthropic",
        ],
    )
    assert result.exit_code == 0, result.output

    # Existing canonical preserved byte-identical AND mtime-unchanged
    # — the skip-write path doesn't touch the file at all.
    assert canonical.read_text(encoding="utf-8") == canonical_before
    assert canonical.stat().st_mtime_ns == mtime_before_ns
    # CLI surfaces the no-progress skip so operators see the gap.
    assert "no canonical file written" in result.output


def test_cli_staged_fresh_mining_partial_outage_with_zero_mentions_skips_write(
    tmp_path, monkeypatch
):
    """wyrd-st1r R2 broadened predicate: when some chunks succeed
    (chunks_processed > 0) but emit no mentions AND some chunks fail
    (chunks_failed > 0), the source's net useful output is still
    zero. The prior (narrower) predicate (`chunks_processed == 0`)
    missed this case — the atomic-write fired, producing a 0-byte
    canonical file, and `--skip-existing` on the next run treated
    the source as done. Gemini R2 P2 + test-coverage-reviewer
    mutation test convergent finding.

    The broadened predicate is `not mentions AND chunks_failed > 0`,
    which fires for both total outages (this PR's original target)
    AND partial outages where mining made some progress but
    captured nothing usable."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    # Multi-paragraph body that chunks into 3.
    body = (
        "Edlingham is a place.\n\n"
        "Some prose with no place names at all.\n\n"
        "Tyne is a river.\n\n"
        "Wear is also a river."
    )
    (sources_dir / "alpha.txt").write_text(body, encoding="utf-8")
    output_dir = tmp_path / "out"

    # Chunk 0 returns no mentions (LLM saw the chunk but found none);
    # chunk 1 raises (failure); chunk 2 returns no mentions; chunk 3
    # raises. Net: chunks_processed=2, chunks_failed=2, mentions=[].
    _stub_anthropic_for_cli(
        monkeypatch,
        FakeClient(
            [
                {"mentions": []},
                RuntimeError("transient"),
                {"mentions": []},
                RuntimeError("transient"),
            ]
        ),
    )

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-staged",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(output_dir),
            "--provider",
            "anthropic",
            "--chunk-size",
            "25",
        ],
    )
    assert result.exit_code == 0, result.output

    # No canonical written despite chunks_processed > 0 — partial
    # outage with zero mentions captured is still suspect.
    canonical = output_dir / "alpha.jsonl"
    assert not canonical.exists(), (
        "partial-outage run with zero mentions produced a canonical "
        "despite some chunks failing; --skip-existing on retry would "
        "silently skip the source"
    )
    assert "no canonical file written" in result.output


def test_cli_staged_capture_failures_writes_chunk_body(tmp_path, monkeypatch):
    """--capture-failures captures every failed chunk including the
    chunk body verbatim — the downstream stage's --from-failures input
    relies on this."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    body = "Edlingham was first attested.\n\nTyne is a river.\n\nDurham is a city."
    (sources_dir / "alpha.txt").write_text(body, encoding="utf-8")
    output_dir = tmp_path / "out"
    failures_file = tmp_path / "failures.jsonl"

    _stub_anthropic_for_cli(
        monkeypatch,
        FakeClient(
            [
                {"mentions": [{"form": "Edlingham", "context": "Edlingham was first attested"}]},
                RuntimeError("HTTP 503"),
                {"mentions": [{"form": "Durham", "context": "Durham is a city"}]},
            ]
        ),
    )

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-staged",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(output_dir),
            "--capture-failures",
            str(failures_file),
            "--provider",
            "anthropic",
            "--chunk-size",
            "20",  # forces 3 chunks
        ],
    )
    assert result.exit_code == 0, result.output

    failures = _read_jsonl(failures_file)
    assert len(failures) == 1
    fc = failures[0]
    assert fc["source_id"] == "alpha"
    assert fc["chunk_index"] == 1  # middle chunk
    assert "Tyne is a river" in fc["chunk_body"]
    assert "RuntimeError" in fc["error"]
    assert fc["extractor"].startswith("anthropic:")


def test_cli_staged_from_failures_appends_with_dedup(tmp_path, monkeypatch):
    """--from-failures resumes pre-chunked failed records, appending
    new mentions to existing per-source files. Dedup on (form,
    date_year, region_hint, context) skips already-emitted rows so a
    re-run of the same stage is idempotent."""
    runner = CliRunner()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    # Pre-existing per-source file from a prior stage
    (output_dir / "alpha.jsonl").write_text(
        json.dumps(
            {
                "source_id": "alpha",
                "form": "Existing",
                "date_year": 1086,
                "region_hint": None,
                "context": "Existing was here",
                "extractor": "ollama:gemma4:26b",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    failures_file = tmp_path / "failures.jsonl"
    failures_file.write_text(
        json.dumps(
            {
                "source_id": "alpha",
                "chunk_index": 5,
                "chunk_body": "Bedlington is a town. Existing was here too.",
                "error": "RuntimeError: stage 1 timeout",
                "extractor": "ollama:gemma4:26b",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    _stub_anthropic_for_cli(
        monkeypatch,
        FakeClient(
            [
                {
                    "mentions": [
                        {"form": "Bedlington", "context": "Bedlington is a town"},
                        # Duplicate of the pre-existing row — must be deduped
                        {"form": "Existing", "date_year": 1086, "context": "Existing was here"},
                    ]
                }
            ]
        ),
    )

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-staged",
            "--from-failures",
            str(failures_file),
            "--output-dir",
            str(output_dir),
            "--provider",
            "anthropic",
        ],
    )
    assert result.exit_code == 0, result.output

    rows = _read_jsonl(output_dir / "alpha.jsonl")
    forms = [r["form"] for r in rows]
    # Existing stays (once), Bedlington added, no duplicate Existing.
    assert forms.count("Existing") == 1
    assert "Bedlington" in forms
    assert len(rows) == 2
    # Summary mentions deduped count
    assert "deduped=1" in result.output


def test_cli_staged_from_failures_captures_secondary_failures(tmp_path, monkeypatch):
    """Cascade contract: a stage 2 run that also fails on some chunks
    must capture those into ITS --capture-failures file so stage 3 can
    pick them up. The cross-stage cascade rides on this composability."""
    runner = CliRunner()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    failures_in = tmp_path / "stage1_failures.jsonl"
    failures_in.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "source_id": "alpha",
                        "chunk_index": 1,
                        "chunk_body": "Bedlington is a town",
                        "error": "stage1 err",
                        "extractor": "ollama:gemma4:26b",
                    }
                ),
                json.dumps(
                    {
                        "source_id": "alpha",
                        "chunk_index": 7,
                        "chunk_body": "Tyne is a river",
                        "error": "stage1 err",
                        "extractor": "ollama:gemma4:26b",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    failures_out = tmp_path / "stage2_failures.jsonl"

    _stub_anthropic_for_cli(
        monkeypatch,
        FakeClient(
            [
                {"mentions": [{"form": "Bedlington", "context": "Bedlington is a town"}]},
                RuntimeError("still failing in stage 2"),
            ]
        ),
    )

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-staged",
            "--from-failures",
            str(failures_in),
            "--output-dir",
            str(output_dir),
            "--capture-failures",
            str(failures_out),
            "--provider",
            "anthropic",
        ],
    )
    assert result.exit_code == 0, result.output

    # Stage 2 success: Bedlington recovered into alpha.jsonl
    alpha = _read_jsonl(output_dir / "alpha.jsonl")
    assert {r["form"] for r in alpha} == {"Bedlington"}

    # Stage 2 failure: Tyne chunk preserved in stage2_failures.jsonl
    # with its ORIGINAL chunk_index from stage 1 (7), not renumbered.
    out_failures = _read_jsonl(failures_out)
    assert len(out_failures) == 1
    assert out_failures[0]["chunk_index"] == 7
    assert "Tyne is a river" in out_failures[0]["chunk_body"]
    assert "still failing" in out_failures[0]["error"]


def test_cli_staged_from_failures_and_source_are_mutually_exclusive(tmp_path, monkeypatch):
    """Operator can't combine fresh-mining and resume modes — confusing
    semantics + would silently ignore one of the inputs. Hard-error
    instead."""
    runner = CliRunner()
    failures_file = tmp_path / "f.jsonl"
    failures_file.write_text("", encoding="utf-8")
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "alpha.txt").write_text("body", encoding="utf-8")

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-staged",
            "--from-failures",
            str(failures_file),
            "--source",
            "alpha",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


def test_cli_staged_from_failures_empty_input_is_noop(tmp_path, monkeypatch):
    """An empty failures file is operationally valid (nothing left to
    do) — exit cleanly with a friendly message rather than erroring."""
    runner = CliRunner()
    failures_file = tmp_path / "f.jsonl"
    failures_file.write_text("", encoding="utf-8")
    output_dir = tmp_path / "out"

    _stub_anthropic_for_cli(monkeypatch, FakeClient([]))

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-staged",
            "--from-failures",
            str(failures_file),
            "--output-dir",
            str(output_dir),
            "--provider",
            "anthropic",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "no records" in result.output


def test_cli_staged_skip_existing_and_force_are_mutually_exclusive(tmp_path):
    """Symmetric with mine-toponym-mentions-tiered: --skip-existing and
    --force can't both be set."""
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-staged",
            "--sources-dir",
            str(tmp_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--skip-existing",
            "--force",
        ],
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output


def test_cli_staged_from_failures_groups_records_by_source(tmp_path, monkeypatch):
    """A failures file may contain records from many sources (corpus-wide
    failure capture in stage 1). The resume command must group by
    source_id and write each source's mentions to its own output file."""
    runner = CliRunner()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    failures_file = tmp_path / "f.jsonl"
    failures_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "source_id": "alpha",
                        "chunk_index": 0,
                        "chunk_body": "Bedlington is a town",
                        "error": "x",
                        "extractor": "ollama:m",
                    }
                ),
                json.dumps(
                    {
                        "source_id": "beta",
                        "chunk_index": 0,
                        "chunk_body": "Tyne is a river",
                        "error": "x",
                        "extractor": "ollama:m",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    _stub_anthropic_for_cli(
        monkeypatch,
        FakeClient(
            [
                {"mentions": [{"form": "Bedlington", "context": "Bedlington is a town"}]},
                {"mentions": [{"form": "Tyne", "context": "Tyne is a river"}]},
            ]
        ),
    )

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-staged",
            "--from-failures",
            str(failures_file),
            "--output-dir",
            str(output_dir),
            "--provider",
            "anthropic",
        ],
    )
    assert result.exit_code == 0, result.output

    assert {r["form"] for r in _read_jsonl(output_dir / "alpha.jsonl")} == {"Bedlington"}
    assert {r["form"] for r in _read_jsonl(output_dir / "beta.jsonl")} == {"Tyne"}


def test_cli_staged_from_failures_rejects_malformed_jsonl(tmp_path, monkeypatch):
    """A failures file with a non-JSON line errors clearly, citing the
    line number — easier debugging than an opaque ValueError traceback."""
    runner = CliRunner()
    failures_file = tmp_path / "bad.jsonl"
    # Valid record on line 1 (with all required fields), garbage on line 2 —
    # without the line-by-line read, line 1 would succeed and the error
    # would surface much later. Pinning the line number is the bug.
    valid = json.dumps(
        {
            "source_id": "alpha",
            "chunk_index": 0,
            "chunk_body": "body",
            "error": "x",
        }
    )
    failures_file.write_text(f"{valid}\nnot json at all\n", encoding="utf-8")

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-staged",
            "--from-failures",
            str(failures_file),
            "--output-dir",
            str(tmp_path / "out"),
            "--provider",
            "anthropic",
        ],
    )
    assert result.exit_code != 0
    # Error should mention the file and the line number 2
    assert "bad.jsonl:2" in result.output


def test_cli_staged_from_failures_rejects_missing_required_field(tmp_path, monkeypatch):
    """A failures file row missing chunk_body errors clearly — without
    chunk_body the resume can't re-process the chunk, and silently
    skipping would lose data."""
    runner = CliRunner()
    failures_file = tmp_path / "incomplete.jsonl"
    failures_file.write_text(
        json.dumps({"source_id": "alpha", "chunk_index": 0, "error": "x"}) + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-staged",
            "--from-failures",
            str(failures_file),
            "--output-dir",
            str(tmp_path / "out"),
            "--provider",
            "anthropic",
        ],
    )
    assert result.exit_code != 0
    assert "chunk_body" in result.output


def test_cli_staged_capture_failures_appends_across_runs(tmp_path, monkeypatch):
    """--capture-failures opens the file in append mode — re-running
    a stage doesn't blow away earlier failure records (useful when
    iterating). Operator clears the file manually for a fresh slate."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "alpha.txt").write_text("Edlin.\n\nTyne.", encoding="utf-8")
    output_dir = tmp_path / "out"
    failures_file = tmp_path / "failures.jsonl"

    # First run: chunk 1 fails
    _stub_anthropic_for_cli(
        monkeypatch,
        FakeClient(
            [
                {"mentions": [{"form": "Edlin", "context": "Edlin"}]},
                RuntimeError("err1"),
            ]
        ),
    )
    result1 = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-staged",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(output_dir),
            "--capture-failures",
            str(failures_file),
            "--provider",
            "anthropic",
            "--chunk-size",
            "10",
            "--force",
        ],
    )
    assert result1.exit_code == 0, result1.output
    assert len(_read_jsonl(failures_file)) == 1

    # Second run on the same source (force-overwrite the per-source output)
    # — failure records APPEND, not overwrite.
    _stub_anthropic_for_cli(
        monkeypatch,
        FakeClient(
            [
                RuntimeError("err2_first"),
                {"mentions": [{"form": "Tyne", "context": "Tyne"}]},
            ]
        ),
    )
    result2 = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-staged",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(output_dir),
            "--capture-failures",
            str(failures_file),
            "--provider",
            "anthropic",
            "--chunk-size",
            "10",
            "--force",
        ],
    )
    assert result2.exit_code == 0, result2.output
    # 1 from run-1 + 1 from run-2 = 2 total
    records = _read_jsonl(failures_file)
    assert len(records) == 2
    # R1 strengthening: assert the two records carry DIFFERENT chunk_body
    # content + DIFFERENT error strings, so a regression that wrote the
    # same record twice would be caught (the prior "len==2" check would
    # pass even if both records were identical copies).
    bodies = {r["chunk_body"] for r in records}
    errors = {r["error"] for r in records}
    assert len(bodies) == 2, f"expected distinct chunk_bodies, got {bodies}"
    assert any("err1" in e for e in errors)
    assert any("err2_first" in e for e in errors)


def test_cli_staged_from_failures_rejects_empty_chunk_body(tmp_path, monkeypatch):
    """A failures record with an empty chunk_body would silently re-run
    on no text; reject it at read time so the operator sees the bug
    rather than a phantom "0 mentions extracted" result."""
    runner = CliRunner()
    failures_file = tmp_path / "empty_body.jsonl"
    failures_file.write_text(
        json.dumps(
            {
                "source_id": "alpha",
                "chunk_index": 0,
                "chunk_body": "",  # empty — invalid for resume
                "error": "x",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-staged",
            "--from-failures",
            str(failures_file),
            "--output-dir",
            str(tmp_path / "out"),
            "--provider",
            "anthropic",
        ],
    )
    assert result.exit_code != 0
    assert "chunk_body is empty" in result.output


def test_cli_staged_from_failures_rejects_non_dict_row(tmp_path, monkeypatch):
    """A failures file row that's valid JSON but a list/string at the
    top level isn't a record — reject explicitly. (Gemini R1 G1.)"""
    runner = CliRunner()
    failures_file = tmp_path / "non_dict.jsonl"
    failures_file.write_text("[1, 2, 3]\n", encoding="utf-8")

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-staged",
            "--from-failures",
            str(failures_file),
            "--output-dir",
            str(tmp_path / "out"),
            "--provider",
            "anthropic",
        ],
    )
    assert result.exit_code != 0
    assert "expected JSON object" in result.output


def test_cli_staged_from_failures_rejects_null_required_field(tmp_path, monkeypatch):
    """A failures record with a null required field (e.g.
    chunk_index: null) should error cleanly rather than crash when
    the int coercion runs. (Gemini R1 G1 paranoia.)"""
    runner = CliRunner()
    failures_file = tmp_path / "null_field.jsonl"
    failures_file.write_text(
        json.dumps(
            {
                "source_id": "alpha",
                "chunk_index": None,
                "chunk_body": "body",
                "error": "x",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-staged",
            "--from-failures",
            str(failures_file),
            "--output-dir",
            str(tmp_path / "out"),
            "--provider",
            "anthropic",
        ],
    )
    assert result.exit_code != 0
    assert "is null" in result.output


def test_cli_staged_from_failures_with_skip_existing_is_rejected(tmp_path, monkeypatch):
    """--skip-existing only makes sense in fresh-mining mode (it gates
    per-source output existence). Resume mode always APPENDS with
    dedup; silently ignoring --skip-existing would mislead operators."""
    runner = CliRunner()
    failures_file = tmp_path / "f.jsonl"
    failures_file.write_text(
        json.dumps(
            {
                "source_id": "a",
                "chunk_index": 0,
                "chunk_body": "body",
                "error": "x",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-staged",
            "--from-failures",
            str(failures_file),
            "--output-dir",
            str(tmp_path / "out"),
            "--skip-existing",
            "--provider",
            "anthropic",
        ],
    )
    assert result.exit_code != 0
    assert "only valid in fresh-mining mode" in result.output


def test_cli_staged_from_failures_with_force_is_rejected(tmp_path, monkeypatch):
    """Symmetric: --force is fresh-mode only, can't combine with --from-failures."""
    runner = CliRunner()
    failures_file = tmp_path / "f.jsonl"
    failures_file.write_text(
        json.dumps(
            {
                "source_id": "a",
                "chunk_index": 0,
                "chunk_body": "body",
                "error": "x",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-staged",
            "--from-failures",
            str(failures_file),
            "--output-dir",
            str(tmp_path / "out"),
            "--force",
            "--provider",
            "anthropic",
        ],
    )
    assert result.exit_code != 0
    assert "only valid in fresh-mining mode" in result.output


def test_cli_staged_capture_failures_warns_on_stale_records(tmp_path, monkeypatch):
    """Stale-records guard: pointing --capture-failures at a non-empty
    pre-existing file should warn the operator. Without this, records
    from a different cascade silently flow into the next stage's input
    (silent-failure-hunter R1 HIGH)."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "alpha.txt").write_text("Edlingham.", encoding="utf-8")
    output_dir = tmp_path / "out"
    # Pre-existing failures file from a "different cascade"
    failures_file = tmp_path / "stale.jsonl"
    failures_file.write_text(
        json.dumps(
            {
                "source_id": "older_source",
                "chunk_index": 99,
                "chunk_body": "stale chunk body",
                "error": "stale",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    _stub_anthropic_for_cli(
        monkeypatch, FakeClient([{"mentions": [{"form": "Edlingham", "context": "Edlingham"}]}])
    )

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-staged",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(output_dir),
            "--capture-failures",
            str(failures_file),
            "--provider",
            "anthropic",
            "--chunk-size",
            "5000",
        ],
    )
    assert result.exit_code == 0, result.output
    # The warning should mention the existing record count
    assert "already has 1 record" in result.output
    # And the new run should still succeed and append nothing (no
    # failures this run) — the stale record remains.
    records = _read_jsonl(failures_file)
    assert len(records) == 1
    assert records[0]["source_id"] == "older_source"
    # Status line confirms 0 new failures (so operator sees the file is unchanged).
    assert "0 new failures" in result.output


def test_cli_staged_from_failures_empty_input_falls_through_to_summary(tmp_path, monkeypatch):
    """Empty failures file is a no-op but must emit the TOTAL summary
    line for consistency with the CLAUDE.md mining-progress convention
    and so scripted cascades can parse a single line shape for every
    run (silent-failure-hunter R1 MEDIUM)."""
    runner = CliRunner()
    failures_file = tmp_path / "empty.jsonl"
    failures_file.write_text("", encoding="utf-8")
    output_dir = tmp_path / "out"

    _stub_anthropic_for_cli(monkeypatch, FakeClient([]))

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-staged",
            "--from-failures",
            str(failures_file),
            "--output-dir",
            str(output_dir),
            "--provider",
            "anthropic",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "no records" in result.output
    # Summary still prints — pin the convention
    assert "TOTAL sources=0" in result.output
    assert "chunks=0" in result.output
    assert "mentions=0" in result.output


def test_cli_staged_from_failures_idempotent_re_run(tmp_path, monkeypatch):
    """True idempotency: running the SAME --from-failures invocation
    twice produces the same per-source output (no duplicates, no
    re-emission). Pins the dedup contract end-to-end, not just for one
    duplicate row (test-coverage-reviewer R1 HIGH)."""
    runner = CliRunner()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    failures_file = tmp_path / "f.jsonl"
    failures_file.write_text(
        json.dumps(
            {
                "source_id": "alpha",
                "chunk_index": 5,
                "chunk_body": "Edlingham is mentioned. Bedlington also.",
                "error": "stage 1 timeout",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def _invoke():
        # Fresh FakeClient each invocation — same response, simulating
        # a deterministic LLM extraction.
        _stub_anthropic_for_cli(
            monkeypatch,
            FakeClient(
                [
                    {
                        "mentions": [
                            {"form": "Edlingham", "context": "Edlingham is mentioned"},
                            {"form": "Bedlington", "context": "Bedlington also"},
                        ]
                    }
                ]
            ),
        )
        return runner.invoke(
            cli_root,
            [
                "lexicon",
                "mine-toponym-mentions-staged",
                "--from-failures",
                str(failures_file),
                "--output-dir",
                str(output_dir),
                "--provider",
                "anthropic",
            ],
        )

    r1 = _invoke()
    assert r1.exit_code == 0, r1.output
    after_run1 = (output_dir / "alpha.jsonl").read_text(encoding="utf-8")

    r2 = _invoke()
    assert r2.exit_code == 0, r2.output
    after_run2 = (output_dir / "alpha.jsonl").read_text(encoding="utf-8")

    # Byte-identical: second run dedups everything, file unchanged.
    assert after_run1 == after_run2
    # And the summary on r2 shows deduped count equal to the row count.
    rows = _read_jsonl(output_dir / "alpha.jsonl")
    assert len(rows) == 2
    assert f"deduped={len(rows)}" in r2.output


def test_cli_staged_fresh_mining_failure_isolation_across_sources(tmp_path, monkeypatch):
    """Fresh-mining mode walks N sources. If source A's chunks all fail
    (provider returns errors), source B should still process — failures
    must not abort the whole multi-source run (test-coverage-reviewer
    R1 HIGH). The chunks_failed count aggregates, but the loop continues.

    wyrd-st1r update: alpha now leaves NO canonical file (was: 0-byte
    file) so the next --skip-existing run retries the failed source.
    Beta still processes normally — failure isolation is preserved."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "alpha.txt").write_text("Edlingham.", encoding="utf-8")
    (sources_dir / "beta.txt").write_text("Tyne.", encoding="utf-8")
    output_dir = tmp_path / "out"

    # alpha fails on its single chunk; beta succeeds.
    _stub_anthropic_for_cli(
        monkeypatch,
        FakeClient(
            [
                RuntimeError("alpha down"),
                {"mentions": [{"form": "Tyne", "context": "Tyne"}]},
            ]
        ),
    )

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-staged",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(output_dir),
            "--provider",
            "anthropic",
            "--chunk-size",
            "5000",
        ],
    )
    assert result.exit_code == 0, result.output
    # alpha left no canonical file — --skip-existing on rerun will retry.
    # The CLI surfaces the skip explicitly to operators.
    assert not (output_dir / "alpha.jsonl").exists()
    assert "no canonical file written" in result.output
    assert "alpha" in result.output
    # beta processed normally — failure isolation preserved.
    beta = _read_jsonl(output_dir / "beta.jsonl")
    assert {r["form"] for r in beta} == {"Tyne"}
    # The aggregate summary reflects failed=1 + mentions=1.
    assert "TOTAL sources=2" in result.output


def test_cli_staged_from_failures_resume_atomic_against_existing(tmp_path, monkeypatch):
    """Resume mode uses atomic rewrite (.tmp + replace), not direct
    append — a killed process mid-write must NOT leave a partially
    truncated line in the existing per-source JSONL (silent-failure
    R1 HIGH + Gemini R1 G3). Verify the existing rows are preserved
    byte-identical when the new run succeeds normally."""
    runner = CliRunner()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    # Pre-existing canonical row from a prior fresh-mining run
    existing_row = json.dumps(
        {
            "source_id": "alpha",
            "form": "Existing",
            "date_year": 1086,
            "region_hint": None,
            "context": "Existing was attested",
            "extractor": "ollama:gemma4:26b",
        }
    )
    (output_dir / "alpha.jsonl").write_text(existing_row + "\n", encoding="utf-8")

    failures_file = tmp_path / "f.jsonl"
    failures_file.write_text(
        json.dumps(
            {
                "source_id": "alpha",
                "chunk_index": 7,
                "chunk_body": "Bedlington is a town.",
                "error": "stage1 err",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    _stub_anthropic_for_cli(
        monkeypatch,
        FakeClient([{"mentions": [{"form": "Bedlington", "context": "Bedlington is a town"}]}]),
    )

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-staged",
            "--from-failures",
            str(failures_file),
            "--output-dir",
            str(output_dir),
            "--provider",
            "anthropic",
        ],
    )
    assert result.exit_code == 0, result.output

    final = (output_dir / "alpha.jsonl").read_text(encoding="utf-8")
    # The original row is preserved EXACTLY (including the stage-1
    # extractor attribution — resume must not re-serialize existing
    # rows, since that would lose the stage-1 model id).
    assert existing_row + "\n" in final
    # The new row was appended.
    rows = _read_jsonl(output_dir / "alpha.jsonl")
    assert {r["form"] for r in rows} == {"Existing", "Bedlington"}
    # And the .tmp file shouldn't linger after the atomic replace.
    assert not (output_dir / "alpha.jsonl.tmp").exists()


def test_cli_staged_from_failures_rejects_non_string_source_id(tmp_path, monkeypatch):
    """source_id as list / int / dict would explode at
    by_source.setdefault(rec["source_id"], ...) with an unhashable-type
    TypeError or silently group under the wrong key. Reject loudly at
    read-time. R2 silent-failure-hunter MEDIUM."""
    runner = CliRunner()
    failures_file = tmp_path / "bad_type.jsonl"
    failures_file.write_text(
        json.dumps(
            {
                "source_id": ["alpha"],  # list, not str
                "chunk_index": 0,
                "chunk_body": "body",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-staged",
            "--from-failures",
            str(failures_file),
            "--output-dir",
            str(tmp_path / "out"),
            "--provider",
            "anthropic",
        ],
    )
    assert result.exit_code != 0
    assert "source_id must be a string" in result.output


def test_cli_staged_from_failures_rejects_non_integer_chunk_index(tmp_path, monkeypatch):
    """chunk_index as a string would slip through the None-check and
    silently coerce via int() — but only for digit strings. A boolean
    True/False would coerce to 1/0 silently. Reject both."""
    runner = CliRunner()
    failures_file = tmp_path / "bad_idx.jsonl"
    failures_file.write_text(
        json.dumps(
            {
                "source_id": "alpha",
                "chunk_index": "five",  # string, not int
                "chunk_body": "body",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-staged",
            "--from-failures",
            str(failures_file),
            "--output-dir",
            str(tmp_path / "out"),
            "--provider",
            "anthropic",
        ],
    )
    assert result.exit_code != 0
    assert "chunk_index must be an integer" in result.output


def test_cli_staged_from_failures_rejects_bool_chunk_index(tmp_path, monkeypatch):
    """bool is a subclass of int in Python — without an explicit guard,
    True/False would silently coerce to 1/0 and process the wrong chunk.
    Reject bool explicitly. R2 silent-failure-hunter MEDIUM."""
    runner = CliRunner()
    failures_file = tmp_path / "bool_idx.jsonl"
    failures_file.write_text(
        json.dumps(
            {
                "source_id": "alpha",
                "chunk_index": True,  # bool — subclass of int!
                "chunk_body": "body",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-staged",
            "--from-failures",
            str(failures_file),
            "--output-dir",
            str(tmp_path / "out"),
            "--provider",
            "anthropic",
        ],
    )
    assert result.exit_code != 0
    assert "chunk_index must be an integer" in result.output


def test_cli_staged_from_failures_rejects_non_string_chunk_body(tmp_path, monkeypatch):
    """chunk_body as a list/dict would slip past the empty-strip check
    (str([]) == "[]" — non-empty), feeding garbage into the LLM call."""
    runner = CliRunner()
    failures_file = tmp_path / "bad_body.jsonl"
    failures_file.write_text(
        json.dumps(
            {
                "source_id": "alpha",
                "chunk_index": 0,
                "chunk_body": ["not", "a", "string"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-staged",
            "--from-failures",
            str(failures_file),
            "--output-dir",
            str(tmp_path / "out"),
            "--provider",
            "anthropic",
        ],
    )
    assert result.exit_code != 0
    assert "chunk_body must be a string" in result.output


def test_cli_staged_from_failures_resume_processes_non_monotonic_indices(tmp_path, monkeypatch):
    """A failures file may carry indices in any order (e.g., [47, 12, 92]
    for one source). The resume must sort by chunk_index per source so
    progress lines look chronological. R2 test-coverage HIGH."""
    runner = CliRunner()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    failures_file = tmp_path / "f.jsonl"
    # Intentionally out-of-order: 47, 12, 92
    failures_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "source_id": "alpha",
                        "chunk_index": 47,
                        "chunk_body": "chunk forty-seven",
                        "error": "x",
                    }
                ),
                json.dumps(
                    {
                        "source_id": "alpha",
                        "chunk_index": 12,
                        "chunk_body": "chunk twelve",
                        "error": "x",
                    }
                ),
                json.dumps(
                    {
                        "source_id": "alpha",
                        "chunk_index": 92,
                        "chunk_body": "chunk ninety-two",
                        "error": "x",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # Capture which chunks the LLM saw and in what order.
    captured_inputs: list[str] = []

    class _OrderingFakeClient:
        model = "fake-test-model"

        def chat_json(self, system, user, schema=None):
            captured_inputs.append(user)
            return {"mentions": []}

    fake = _OrderingFakeClient()
    import wyrd.generators.kenning.extractors.anthropic as ae_module

    monkeypatch.setattr(ae_module, "AnthropicClient", lambda **kw: fake)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-staged",
            "--from-failures",
            str(failures_file),
            "--output-dir",
            str(output_dir),
            "--provider",
            "anthropic",
        ],
    )
    assert result.exit_code == 0, result.output
    # Three calls in chunk_index order: 12, 47, 92.
    assert len(captured_inputs) == 3
    assert "chunk twelve" in captured_inputs[0]
    assert "chunk forty-seven" in captured_inputs[1]
    assert "chunk ninety-two" in captured_inputs[2]


def test_cli_staged_from_failures_limit_applies_per_source(tmp_path, monkeypatch):
    """--limit in resume mode slices each source's chunk list, NOT the
    global record count. Pins the semantics so a regression that
    interpreted --limit globally would be caught. R2 test-coverage HIGH."""
    runner = CliRunner()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    failures_file = tmp_path / "f.jsonl"
    # 3 chunks for alpha, 3 chunks for beta.
    failures_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "source_id": "alpha",
                        "chunk_index": i,
                        "chunk_body": f"a{i}",
                        "error": "x",
                    }
                )
                for i in (1, 2, 3)
            ]
            + [
                json.dumps(
                    {
                        "source_id": "beta",
                        "chunk_index": i,
                        "chunk_body": f"b{i}",
                        "error": "x",
                    }
                )
                for i in (1, 2, 3)
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    captured_inputs: list[str] = []

    class _CountingFakeClient:
        model = "fake-test-model"

        def chat_json(self, system, user, schema=None):
            captured_inputs.append(user)
            return {"mentions": []}

    import wyrd.generators.kenning.extractors.anthropic as ae_module

    monkeypatch.setattr(ae_module, "AnthropicClient", lambda **kw: _CountingFakeClient())
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-staged",
            "--from-failures",
            str(failures_file),
            "--output-dir",
            str(output_dir),
            "--provider",
            "anthropic",
            "--limit",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    # PER-SOURCE limit: 2 from alpha + 2 from beta = 4 calls (not 2 globally).
    assert len(captured_inputs) == 4
    # First two are alpha (sorted), next two are beta.
    assert all("a" in c for c in captured_inputs[:2])
    assert all("b" in c for c in captured_inputs[2:])


def test_cli_staged_resume_warns_on_malformed_existing_rows(tmp_path, monkeypatch):
    """If the existing per-source JSONL has malformed rows (from a
    SIGKILL'd prior write or a hand-edit), surface the count so the
    operator sees they can't be deduped against. R2 silent-failure-hunter
    MEDIUM: silent-skip was masking data corruption."""
    runner = CliRunner()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    # Existing per-source file with a mix of valid and corrupt rows.
    valid_row = json.dumps(
        {
            "source_id": "alpha",
            "form": "Existing",
            "date_year": None,
            "region_hint": None,
            "context": "ctx",
        }
    )
    (output_dir / "alpha.jsonl").write_text(
        valid_row + "\n" + "not json at all\n" + "[1, 2, 3]\n", encoding="utf-8"
    )

    failures_file = tmp_path / "f.jsonl"
    failures_file.write_text(
        json.dumps(
            {
                "source_id": "alpha",
                "chunk_index": 1,
                "chunk_body": "Bedlington is a town",
                "error": "x",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    _stub_anthropic_for_cli(
        monkeypatch,
        FakeClient([{"mentions": [{"form": "Bedlington", "context": "Bedlington is a town"}]}]),
    )

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-staged",
            "--from-failures",
            str(failures_file),
            "--output-dir",
            str(output_dir),
            "--provider",
            "anthropic",
        ],
    )
    assert result.exit_code == 0, result.output
    # Warning surfaces the malformed count
    assert "2 malformed row(s)" in result.output
    assert "can't be deduped" in result.output
    # Per-source header includes "2 malformed"
    assert "2 malformed" in result.output


def test_cli_staged_capture_failures_does_not_warn_on_empty_existing_file(tmp_path, monkeypatch):
    """The stale-records warning fires only when the file is NON-EMPTY.
    A pre-touched but empty file (e.g., from a prior run that captured
    no failures) should NOT trigger the warning. Pins both halves of
    the `exists() AND st_size > 0` gate."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "alpha.txt").write_text("Edlingham.", encoding="utf-8")
    output_dir = tmp_path / "out"
    failures_file = tmp_path / "empty.jsonl"
    failures_file.write_text("", encoding="utf-8")  # exists but empty

    _stub_anthropic_for_cli(
        monkeypatch,
        FakeClient([{"mentions": [{"form": "Edlingham", "context": "Edlingham"}]}]),
    )

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-staged",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(output_dir),
            "--capture-failures",
            str(failures_file),
            "--provider",
            "anthropic",
            "--chunk-size",
            "5000",
        ],
    )
    assert result.exit_code == 0, result.output
    # The stale-records warning must NOT have fired
    assert "already has" not in result.output


def test_cli_staged_resume_purges_malformed_existing_rows(tmp_path, monkeypatch):
    """The verbatim-copy preserves WELL-FORMED existing rows but DROPS
    malformed (non-JSON / non-dict) rows during the atomic rewrite.
    Without this, corrupt rows accumulate across every resume cycle
    forever. R3 silent-failure-hunter MEDIUM."""
    runner = CliRunner()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    valid_row = json.dumps(
        {
            "source_id": "alpha",
            "form": "Existing",
            "date_year": None,
            "region_hint": None,
            "context": "ctx",
        }
    )
    # File with 1 valid row + 2 malformed rows
    (output_dir / "alpha.jsonl").write_text(
        valid_row + "\n" + "not json at all\n" + "[1, 2, 3]\n", encoding="utf-8"
    )

    failures_file = tmp_path / "f.jsonl"
    failures_file.write_text(
        json.dumps(
            {
                "source_id": "alpha",
                "chunk_index": 1,
                "chunk_body": "Bedlington is a town",
                "error": "x",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    _stub_anthropic_for_cli(
        monkeypatch,
        FakeClient([{"mentions": [{"form": "Bedlington", "context": "Bedlington is a town"}]}]),
    )

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-staged",
            "--from-failures",
            str(failures_file),
            "--output-dir",
            str(output_dir),
            "--provider",
            "anthropic",
        ],
    )
    assert result.exit_code == 0, result.output
    # After atomic rewrite: the valid row + the new Bedlington row remain;
    # the two malformed rows are GONE.
    final_text = (output_dir / "alpha.jsonl").read_text(encoding="utf-8")
    assert "not json at all" not in final_text
    assert "[1, 2, 3]" not in final_text
    rows = _read_jsonl(output_dir / "alpha.jsonl")
    assert {r["form"] for r in rows} == {"Existing", "Bedlington"}
    # Operator-visible purge count appears in output
    assert "purged 2 malformed row(s)" in result.output
    # And aggregates into the TOTAL summary
    assert "malformed_purged=2" in result.output


def test_cli_staged_fresh_mining_force_overwrites_existing(tmp_path, monkeypatch):
    """--force in fresh-mining mode REPLACES an existing per-source
    output (atomic .tmp + replace). Pin the happy-path overwrite,
    complementing the existing mutex-error coverage. R3 test-coverage
    HIGH: only the error path was covered before."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "alpha.txt").write_text("Edlingham.", encoding="utf-8")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    # Pre-existing per-source output that --force should clobber
    stale = output_dir / "alpha.jsonl"
    stale.write_text(
        json.dumps(
            {
                "source_id": "alpha",
                "form": "StaleForm",
                "date_year": None,
                "region_hint": None,
                "context": "stale",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    _stub_anthropic_for_cli(
        monkeypatch,
        FakeClient([{"mentions": [{"form": "Edlingham", "context": "Edlingham"}]}]),
    )

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-staged",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(output_dir),
            "--provider",
            "anthropic",
            "--chunk-size",
            "5000",
            "--force",
        ],
    )
    assert result.exit_code == 0, result.output
    rows = _read_jsonl(stale)
    assert {r["form"] for r in rows} == {"Edlingham"}
    assert not any(r["form"] == "StaleForm" for r in rows)


def test_cli_staged_fresh_mining_warns_on_utf8_replacement_chars(tmp_path, monkeypatch):
    """When the source body contains invalid UTF-8 (read with
    errors='replace' produces U+FFFD), the staged command must warn
    the operator so silent mention rejection from the form-in-chunk
    guard doesn't go unnoticed. R3 test-coverage HIGH."""
    runner = CliRunner()
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    # Bytes 0xff 0xfe are illegal in UTF-8 and decode to U+FFFD via
    # errors='replace'.
    (sources_dir / "alpha.txt").write_bytes(b"Edlingham is here\xff\xfe and more.")
    output_dir = tmp_path / "out"

    _stub_anthropic_for_cli(
        monkeypatch,
        FakeClient([{"mentions": []}]),
    )

    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "mine-toponym-mentions-staged",
            "--sources-dir",
            str(sources_dir),
            "--output-dir",
            str(output_dir),
            "--provider",
            "anthropic",
            "--chunk-size",
            "5000",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "invalid UTF-8" in result.output


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
