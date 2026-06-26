"""Tests for the LLM extractor's validation guard + OllamaClient
transport safety wraps.

`validate_response` is the safety check that prevents hallucinated
etymons from being ingested — load-bearing piece of the LLM pipeline.
If it ever silently breaks, garbage flows into the lexicon.

OllamaClient transport: the dispatch-time and parse-time ValueError
re-raise path in `_PROGRAMMER_ERROR_EXCEPTIONS` (toponym_mention_
extractor.py) requires every transport-layer ValueError-subclass to
be wrapped into RuntimeError or chunks_failed gets bypassed in favour
of full-run abort. Tests below pin the OllamaClient envelope-parse
and decode wraps so a future refactor can't silently regress the
contract (Ollama is the default `--provider` — losing coverage here
is the worst direction). wyrd-pe4g rounds 4-5.
"""

from __future__ import annotations

import json
import re
from unittest.mock import patch

import pytest

from wyrd.generators.kenning.extractors.llm import (
    RESPONSE_SCHEMA,
    SOURCE_QUOTE_BUDGET,
    SYSTEM_PROMPT,
    OllamaClient,
    _normalize_for_match,
    validate_response,
)

# --- normalization ---------------------------------------------------------


def test_normalize_for_match_lowercases_and_collapses_diacritics() -> None:
    assert _normalize_for_match("Hædan") == _normalize_for_match("hædan")
    assert _normalize_for_match("Hædan") == _normalize_for_match("Haedan")
    # ð → d (the validation-side normalizer uses single-char mapping; the
    # lexicon's clustering normalizer uses dh — different functions, both
    # lossy enough for their own purpose).
    assert _normalize_for_match("hyð") == "hyd"
    # OCR mangle of æ → "cs" maps to same key as proper æ
    assert _normalize_for_match("Hcsdan") == _normalize_for_match("Hædan")


def test_normalize_for_match_preserves_distinct_words() -> None:
    """Different roots must NOT collapse together."""
    assert _normalize_for_match("ham") != _normalize_for_match("hamm")
    assert _normalize_for_match("tun") != _normalize_for_match("dun")


# --- validate_response -----------------------------------------------------


def _ok_response() -> dict:
    """A baseline well-formed response. Tests mutate copies of this."""
    return {
        "found": True,
        "historical_form": "bere-tun",
        "elements": [
            {
                "form": "bere",
                "language": "old-english",
                "position": "pre",
                "gloss": "barley",
                "inflection": None,
            },
            {
                "form": "tun",
                "language": "old-english",
                "position": "post",
                "gloss": "enclosure",
                "inflection": None,
            },
        ],
        "confidence": "high",
        "notes": None,
    }


_BARTON_BODY = (
    "Barton. This is the prov. E. barton, a farm-yard ; "
    "for which see the English Dialect Dictionary. "
    "It is the A.S. bere-tun, lit. corn-farm, or barley-enclosure ; "
    "from bere, barley, and tun. Thus the syllable Bar- is in this instance "
    "the same as the bar- in barley, see the New English Dictionary."
)


def test_validate_response_accepts_clean_response() -> None:
    failures = validate_response(_ok_response(), _BARTON_BODY)
    assert failures == []


def test_validate_response_accepts_decline() -> None:
    """A 'found: false' decline is a clean no-op, not a failure."""
    response = _ok_response()
    response["found"] = False
    response["elements"] = []
    failures = validate_response(response, "no etymology in this entry")
    assert failures == []


def test_validate_response_rejects_form_not_in_body() -> None:
    """The critical anti-hallucination check: every form must appear in body."""
    response = _ok_response()
    response["elements"][0]["form"] = "xyzfake"  # not in Barton's body
    failures = validate_response(response, _BARTON_BODY)
    reasons = [f.reason for f in failures]
    assert "form_not_in_body" in reasons


def test_validate_response_strips_dashes_before_form_check() -> None:
    """Model commonly emits 'karla-' (with trailing dash) as a pre-element.
    Validation must strip the dash before matching against the body."""
    response = _ok_response()
    response["elements"][0]["form"] = "bere-"  # trailing dash on "bere"
    failures = validate_response(response, _BARTON_BODY)
    # Should pass because "bere" (without dash) is in body.
    assert not any(f.reason == "form_not_in_body" for f in failures)


def test_validate_response_rejects_digit_bearing_form() -> None:
    """An OCR-misread element form with a digit ('h8ema', '6j^', '0yrr') is in
    the noisy source body, so it PASSES the anti-hallucination check — but a
    digit is never a real linguistic form (no OE/ME/Norse/reconstructed headword
    carries an Arabic numeral), so it must still be rejected. Without this,
    digit-garbage lands as a corrupt etymon canonical_form (wyrd-tru5)."""
    response = _ok_response()
    response["elements"][0]["form"] = "h8ema"
    body = _BARTON_BODY + " The OCR-mangled form h8ema also appears here."
    failures = validate_response(response, body)
    reasons = [f.reason for f in failures]
    assert "form_has_digit" in reasons
    # Isolate the new check: the garbage IS in the body, so the
    # anti-hallucination guard alone would NOT have caught it.
    assert "form_not_in_body" not in reasons


def test_validate_response_accepts_macron_and_asterisk_forms() -> None:
    """The digit guard must not over-reach onto legitimate non-ASCII form marks:
    macrons (ū), the reconstruction asterisk (*), and æ/þ/ð are NOT digits."""
    response = _ok_response()
    response["elements"][0]["form"] = "berġ"
    response["elements"][1]["form"] = "*tūn"
    body = _BARTON_BODY + " Compare berġ and the reconstructed *tūn."
    failures = validate_response(response, body)
    assert not any(f.reason == "form_has_digit" for f in failures)


def test_validate_response_rejects_bad_language() -> None:
    response = _ok_response()
    response["elements"][0]["language"] = "klingon"  # not in allowed set
    failures = validate_response(response, _BARTON_BODY)
    assert any(f.reason == "bad_language" for f in failures)


def test_validate_response_rejects_bad_position() -> None:
    response = _ok_response()
    response["elements"][0]["position"] = "sideways"
    failures = validate_response(response, _BARTON_BODY)
    assert any(f.reason == "bad_position" for f in failures)


def test_validate_response_rejects_oversized_element_count() -> None:
    """No place name has 5+ morphemes in a real breakdown."""
    response = _ok_response()
    elem = response["elements"][0].copy()
    response["elements"] = [elem] * 5
    failures = validate_response(response, _BARTON_BODY)
    assert any(f.reason == "element_count_out_of_range" for f in failures)


def test_validate_response_rejects_empty_form() -> None:
    """Empty-string form would slip past minLength schema if it ever arrived."""
    response = _ok_response()
    response["elements"][0]["form"] = ""
    failures = validate_response(response, _BARTON_BODY)
    assert any(f.reason == "form_not_in_body" for f in failures)


def test_validate_response_does_not_reject_paraphrased_glosses() -> None:
    """Glosses are allowed to paraphrase the source text — the model often
    compresses 'an enclosure' to 'enclosure', or paraphrases differently
    altogether. Only the form is required to be substring-quoted."""
    response = _ok_response()
    response["elements"][1]["gloss"] = "a fortified settlement"  # paraphrase
    failures = validate_response(response, _BARTON_BODY)
    # No gloss-related failures.
    assert not any("gloss" in f.reason for f in failures)


def test_system_prompt_distinguishes_hedge_from_no_claim() -> None:
    """The SYSTEM_PROMPT must instruct models to extract HEDGED etymologies
    with confidence='low', not decline them. This prevents data loss when
    19th-c scholars hedge ('perhaps', 'possibly', 'I only make a guess') —
    the proposed etymology is still useful data, just at low confidence.

    Regression test: if the prompt is shortened/reworded and loses this
    distinction, future mining runs will silently drop hedged entries.
    """
    prompt = SYSTEM_PROMPT.lower()

    # Must explicitly mention low-confidence hedge handling.
    assert "perhaps" in prompt
    assert "hedge" in prompt
    assert 'confidence="low"' in SYSTEM_PROMPT  # case-sensitive — must be exact
    # Must distinguish from the "no etymology at all" case (whitespace-tolerant
    # since the prompt may wrap the phrase across lines).
    assert re.search(r"no\s+etymological\s+claim", prompt) is not None
    # Should mention that hedged data is still wanted.
    assert "still useful" in prompt or "still wanted" in prompt


def test_validate_response_handles_non_object() -> None:
    """Defensive against malformed responses (Ollama schema *should* prevent
    this, but defense in depth is cheap)."""
    failures = validate_response("not a dict", _BARTON_BODY)
    assert any(f.reason == "not_object" for f in failures)


# --- attested_forms (D5-1 / wyrd-3ux) -------------------------------------


_DATED_BODY = (
    "Aburwick. Cited as Aburwick, 1333 ; Abberwyke, 1346 ; Aburwick, 1428. "
    "From OE Hædan + tun, with later spelling drift."
)


def test_validate_response_accepts_attested_forms() -> None:
    """Model emits dated historical-form citations from the body."""
    response = _ok_response()
    response["historical_form"] = "Hædan-tun"
    response["elements"][0]["form"] = "Hædan"
    response["attested_forms"] = [
        {"form": "Aburwick", "year": 1333},
        {"form": "Abberwyke", "year": 1346},
    ]
    failures = validate_response(response, _DATED_BODY)
    assert failures == []


def test_validate_response_accepts_empty_attested_forms() -> None:
    """When body has no year citations, attested_forms is the empty list."""
    response = _ok_response()
    response["attested_forms"] = []
    failures = validate_response(response, _BARTON_BODY)
    assert failures == []


def test_validate_response_back_compat_when_attested_forms_absent() -> None:
    """Legacy responses (and legacy test fixtures) won't have attested_forms.
    Validation should treat absence as 'no extraction' rather than a failure."""
    response = _ok_response()
    # _ok_response() does NOT include attested_forms — legacy shape.
    assert "attested_forms" not in response
    failures = validate_response(response, _BARTON_BODY)
    assert failures == []


def test_validate_response_rejects_attested_year_below_range() -> None:
    """Years before 100 are almost always folio numbers or page numbers
    misread as years (wyrd-z56 widened the floor from 800 to 100 to keep
    Roman-empire and Gallo-Roman attestations; 99 is still excluded)."""
    response = _ok_response()
    response["attested_forms"] = [{"form": "bere", "year": 79}]
    failures = validate_response(response, _BARTON_BODY)
    assert any(f.reason == "attested_year_out_of_range" for f in failures)


def test_validate_response_rejects_attested_year_above_range() -> None:
    """Years after 1700 are almost always publication-year noise (1880-1928)."""
    response = _ok_response()
    response["attested_forms"] = [{"form": "bere", "year": 1920}]
    failures = validate_response(response, _BARTON_BODY)
    assert any(f.reason == "attested_year_out_of_range" for f in failures)


def test_validate_response_rejects_attested_form_not_in_body() -> None:
    """Same form-in-body anti-hallucination guard applies to attested_forms.
    A model can't invent a historical spelling that isn't in the source."""
    response = _ok_response()
    response["attested_forms"] = [{"form": "Xyzfake", "year": 1333}]
    failures = validate_response(response, _DATED_BODY)
    assert any(f.reason == "attested_form_not_in_body" for f in failures)


def test_validate_response_rejects_attested_year_non_integer() -> None:
    """Year must be an integer. Strings or floats are reject candidates."""
    response = _ok_response()
    response["attested_forms"] = [{"form": "Aburwick", "year": "1333"}]
    failures = validate_response(response, _DATED_BODY)
    assert any(f.reason == "attested_year_not_int" for f in failures)


def test_validate_response_rejects_attested_year_bool() -> None:
    """Python bool is a subclass of int; reject it explicitly so True doesn't
    pass as year=1."""
    response = _ok_response()
    response["attested_forms"] = [{"form": "Aburwick", "year": True}]
    failures = validate_response(response, _DATED_BODY)
    assert any(f.reason == "attested_year_not_int" for f in failures)


def test_validate_response_rejects_attested_forms_not_list() -> None:
    """Schema enforces array type; defensive validation if it slips through."""
    response = _ok_response()
    response["attested_forms"] = "not a list"
    failures = validate_response(response, _BARTON_BODY)
    assert any(f.reason == "attested_forms_not_list" for f in failures)


def test_system_prompt_documents_attested_forms_extraction() -> None:
    """Regression guard: SYSTEM_PROMPT must explain how to extract dated
    historical-form citations. If the prompt is later shortened and loses
    this section, future mining will silently miss D5-1 data.
    """
    prompt = SYSTEM_PROMPT.lower()

    # Field name appears.
    assert "attested_forms" in prompt
    # Year-range constraint is documented (wyrd-z56 widened to 100-1700).
    assert "100" in prompt and "1700" in prompt
    # Examples include real Domesday/charter-style dated citations.
    assert "1333" in prompt or "1226" in prompt or "1086" in prompt
    # Skip-noise instruction is present (publication years vs attestations).
    assert "publication" in prompt
    # No-invention instruction is present.
    assert "do not invent" in prompt or "do not invent dates" in prompt
    # wyrd-z56: tightened integer-vs-string guidance with explicit examples.
    # The model was emitting strings like "1333 (?)" — prompt now flags the
    # right/wrong patterns explicitly.
    assert "json integer" in prompt or "(not a string" in prompt


def test_response_schema_includes_attested_forms() -> None:
    """The Ollama JSON schema must mark attested_forms as required, with
    year integer constrained to the 100-1700 range. If this constraint is
    removed the model can emit publication-year noise unchecked."""
    assert "attested_forms" in RESPONSE_SCHEMA["properties"]
    assert "attested_forms" in RESPONSE_SCHEMA["required"]
    item_schema = RESPONSE_SCHEMA["properties"]["attested_forms"]["items"]
    year_schema = item_schema["properties"]["year"]
    assert year_schema["type"] == "integer"
    assert year_schema["minimum"] == 100
    assert year_schema["maximum"] == 1700


def test_validate_response_validates_attested_forms_on_decline() -> None:
    """Even when the model declines (found=false), attested_forms must be
    validated. A body can carry dated citations without an etymological
    claim ('first attested 1086 but no etymology'), so we can't bypass
    the anti-hallucination guard just because elements are empty.

    Regression guard: if found=false short-circuits before
    attested_forms validation, an in-body-uncited form would slip
    through to raw_response unchecked."""
    response = {
        "found": False,
        "historical_form": None,
        "elements": [],
        "attested_forms": [{"form": "Xyzfake", "year": 1333}],
        "confidence": "low",
        "notes": None,
    }
    failures = validate_response(response, _DATED_BODY)
    assert any(f.reason == "attested_form_not_in_body" for f in failures)


def test_validate_response_rejects_attested_form_not_object() -> None:
    """An attested_forms entry that's a string / list / number — not a
    {form, year} dict — should be flagged."""
    response = _ok_response()
    response["attested_forms"] = ["Aburwick, 1333"]  # string instead of dict
    failures = validate_response(response, _DATED_BODY)
    assert any(f.reason == "attested_form_not_object" for f in failures)


def test_validate_response_strips_dashes_before_attested_form_check() -> None:
    """Trailing dashes on a real form are stripped before the body match
    so the legitimate case (e.g. ``Aburwick-`` from a citation context)
    passes the form-in-body check.

    Also pins the length-floor bypass guard: ``-A-`` strips to ``A``
    (length 1), which fails the ``len(a_form) < 2`` check and lands as
    attested_form_not_in_body. Without the dash strip, ``-A-`` would be
    length 3 and could slip past the floor.
    """
    response = _ok_response()
    response["attested_forms"] = [{"form": "Aburwick-", "year": 1333}]
    failures = validate_response(response, _DATED_BODY)
    assert not any(f.reason == "attested_form_not_in_body" for f in failures)

    # Bypass-attempt path: '-A-' strips to 'A' which fails the length floor.
    bypass = _ok_response()
    bypass["attested_forms"] = [{"form": "-A-", "year": 1333}]
    bypass_failures = validate_response(bypass, _DATED_BODY)
    assert any(f.reason == "attested_form_not_in_body" for f in bypass_failures)


def test_validate_response_handles_non_string_element_form() -> None:
    """Same defensive pattern as the attested_forms-side test:
    a non-string element form (bool, int, list) shouldn't crash the
    validator with AttributeError. Symmetric with
    ``test_validate_response_handles_non_string_attested_form``."""
    response = _ok_response()
    response["elements"][0]["form"] = True
    failures = validate_response(response, _BARTON_BODY)
    # Should NOT AttributeError. str(True) → "True" doesn't appear in body.
    assert any(f.reason == "form_not_in_body" for f in failures)


def test_validate_response_handles_non_string_attested_form() -> None:
    """Defensive against non-string form values (bool, int, list, etc.)
    slipping past schema enforcement. Gemini's responseSchema enforcement
    is unreliable across versions and Anthropic uses prompt-only schema
    enforcement, so a model emitting ``form: true`` shouldn't crash the
    validator with an AttributeError. The ``str()`` coercion converts
    truthy non-strings to their string representation, which then fails
    the form-in-body check (the safe outcome)."""
    response = _ok_response()
    response["attested_forms"] = [{"form": True, "year": 1333}]
    failures = validate_response(response, _DATED_BODY)
    # Should NOT AttributeError. Should land as form-not-in-body since
    # str(True) → "True" doesn't appear in _DATED_BODY.
    assert any(f.reason == "attested_form_not_in_body" for f in failures)


def test_validate_response_accepts_decline_with_valid_attested_forms() -> None:
    """D5-1 production case: body carries dated citations but the model
    declines to extract an etymology. Validation must pass cleanly — that's
    the positive-path counterpart to
    ``test_validate_response_validates_attested_forms_on_decline``.

    Without this test, a future refactor that re-adds an
    elements-required check on the decline path would still pass all the
    negative-path tests (which expect failures regardless), silently
    breaking the production scenario the bypass exists to enable.
    """
    response = {
        "found": False,
        "historical_form": None,
        "elements": [],
        "attested_forms": [{"form": "Aburwick", "year": 1333}],
        "confidence": "low",
        "notes": None,
    }
    failures = validate_response(response, _DATED_BODY)
    assert failures == []


def test_validate_response_attested_year_at_lower_boundary() -> None:
    """Inclusive lower bound: year=100 is the floor for the attested-year
    range (per _ATTESTED_YEAR_MIN, widened from 800 in wyrd-z56). Off-by-
    one regression guard."""
    body = _DATED_BODY + " Aburwick is also recorded in 100."
    response = _ok_response()
    response["attested_forms"] = [{"form": "Aburwick", "year": 100}]
    failures = validate_response(response, body)
    assert not any(f.reason.startswith("attested_year") for f in failures)


def test_validate_response_attested_year_below_lower_boundary() -> None:
    """Year=99 is below the 100 floor and should be rejected (folio /
    page-number territory)."""
    response = _ok_response()
    response["attested_forms"] = [{"form": "bere", "year": 99}]
    failures = validate_response(response, _BARTON_BODY)
    assert any(f.reason == "attested_year_out_of_range" for f in failures)


def test_validate_response_accepts_roman_era_attested_year() -> None:
    """wyrd-z56: Roman-era and pre-Carolingian dated attestations now
    pass. Concrete case d'Arbois 1890 cites: 'une charte de l'année
    856' / 'Actum Aria monasterio'-style charters, plus earlier
    Gallo-Roman material dating to the 5th-7th centuries (e.g. the
    Geographer of Ravenna, ~700 AD).

    Sample acceptable years across the widened range: 200 (Roman
    Gaul), 580 (Merovingian), 856 (Carolingian, was previously above
    floor anyway). The 800 floor lost legitimate Roman + early-
    Merovingian rows on Romance/Celtic-substrate sources.
    """
    body = (
        _DATED_BODY + " Earlier still: Lugudunum (200), Aria monasterio (580),"
        " and a Carolingian charter dated 856."
    )
    response = _ok_response()
    response["attested_forms"] = [
        {"form": "Lugudunum", "year": 200},
        {"form": "Aria", "year": 580},
        {"form": "monasterio", "year": 856},
    ]
    failures = validate_response(response, body)
    assert not any(f.reason.startswith("attested_year") for f in failures)
    assert not any(f.reason == "attested_form_not_in_body" for f in failures)


def test_validate_response_attested_year_at_upper_boundary() -> None:
    """Inclusive upper bound: year=1700 is the ceiling. Off-by-one regression
    guard."""
    body = _DATED_BODY + " Aburwick is also recorded in 1700."
    response = _ok_response()
    response["attested_forms"] = [{"form": "Aburwick", "year": 1700}]
    failures = validate_response(response, body)
    assert not any(f.reason.startswith("attested_year") for f in failures)


def test_validate_response_attested_year_above_upper_boundary() -> None:
    """Year=1701 is above the 1700 ceiling and should be rejected."""
    response = _ok_response()
    response["attested_forms"] = [{"form": "bere", "year": 1701}]
    failures = validate_response(response, _BARTON_BODY)
    assert any(f.reason == "attested_year_out_of_range" for f in failures)


def test_validate_response_rejects_attested_year_float() -> None:
    """Year 1333.0 (a float, possibly from a JSON decoder that promoted
    integers) is not an int and must be flagged. Realistic edge case
    when responses transit JSON layers that lose int/float distinction."""
    response = _ok_response()
    response["attested_forms"] = [{"form": "Aburwick", "year": 1333.0}]
    failures = validate_response(response, _DATED_BODY)
    assert any(f.reason == "attested_year_not_int" for f in failures)


def test_validate_response_attested_form_missing_year_key() -> None:
    """If the model omits the year key entirely, .get('year') yields None;
    None is not int, so flag it as attested_year_not_int."""
    response = _ok_response()
    response["attested_forms"] = [{"form": "Aburwick"}]  # year key missing
    failures = validate_response(response, _DATED_BODY)
    assert any(f.reason == "attested_year_not_int" for f in failures)


def test_validate_response_attested_form_missing_form_key() -> None:
    """If the model omits the form key entirely, .get('form') yields None;
    the .strip()/dash-strip chain coerces None→'', which fails the
    length-floor → form-not-in-body."""
    response = _ok_response()
    response["attested_forms"] = [{"year": 1333}]  # form key missing
    failures = validate_response(response, _DATED_BODY)
    assert any(f.reason == "attested_form_not_in_body" for f in failures)


def test_validate_response_reports_index_for_each_invalid_entry() -> None:
    """Mixed valid + invalid entries: validation should NOT short-circuit
    on the first failure. Every invalid entry must produce a failure
    tagged with its index (so callers can pinpoint which entries to
    drop)."""
    response = _ok_response()
    response["attested_forms"] = [
        {"form": "Aburwick", "year": 1333},  # valid
        {"form": "Xyzfake", "year": 1346},  # invalid: form not in body
        {"form": "Abberwyke", "year": 1346},  # valid
        {"form": "Aburwick", "year": 79},  # invalid: year out of range
    ]
    failures = validate_response(response, _DATED_BODY)
    detail_indices = [f.detail for f in failures]
    assert any("attested_forms[1]" in d for d in detail_indices)
    assert any("attested_forms[3]" in d for d in detail_indices)


# --- wyrd-bjs: shared assemble_extraction_result + transport_error_result --


def test_transport_error_result_packs_failure_into_llmresult():
    """transport_error_result returns a non-accepted LLMResult with a single
    transport_error failure carrying the message — uniform shape across
    Ollama/Gemini/Anthropic providers so the mining loop can count it as
    a 'rejected' regardless of which client raised."""
    from wyrd.generators.kenning.extractors.llm import transport_error_result

    result = transport_error_result("HTTP 503 Service Unavailable")
    assert result.accepted is False
    assert result.entry is None
    assert result.raw_response == {"error": "HTTP 503 Service Unavailable"}
    assert len(result.failures) == 1
    assert result.failures[0].reason == "transport_error"
    assert "HTTP 503" in result.failures[0].detail


def test_assemble_extraction_result_not_found_emits_low_confidence_entry():
    """Per D14, when the response says found=False the result is an
    accepted ParsedEntry with empty elements and confidence='low'. Per
    wyrd-6hd, the source_quote on the decline path now carries the
    provider notes_prefix so a later query can distinguish 'Gemini
    declined this' from 'Qwen declined' — matches the accepted-path
    composition."""
    from wyrd.generators.kenning.extractors.llm import assemble_extraction_result

    body = "Foobar — origin unknown."
    result = assemble_extraction_result(
        {"found": False},
        body=body,
        toponym="Foobar",
        suffix_hint=None,
        notes_prefix="extracted_by:test:1",
    )
    assert result.accepted is True
    assert result.entry is not None
    assert result.entry.elements == []
    assert result.entry.confidence == "low"
    # Provider attribution rides on the decline-path source_quote.
    assert result.entry.source_quote.startswith("extracted_by:test:1 | ")
    assert "Foobar — origin unknown." in result.entry.source_quote
    assert len(result.entry.source_quote) <= SOURCE_QUOTE_BUDGET


def test_assemble_extraction_result_decline_path_distinguishes_providers():
    """wyrd-6hd: two providers declining the same body land different
    source_quote prefixes — pins the audit-query shape that motivated
    the change ('show me Gemini's declines' vs 'show me Qwen's')."""
    from wyrd.generators.kenning.extractors.llm import assemble_extraction_result

    body = "Foobar — origin unknown."
    gemini = assemble_extraction_result(
        {"found": False},
        body=body,
        toponym="Foobar",
        suffix_hint=None,
        notes_prefix="extracted_by:gemini:gemini-2.5-flash",
    )
    qwen = assemble_extraction_result(
        {"found": False},
        body=body,
        toponym="Foobar",
        suffix_hint=None,
        notes_prefix="extracted_by:llm:qwen3.5:35b-a3b",
    )
    assert gemini.entry.source_quote.startswith("extracted_by:gemini:gemini-2.5-flash | ")
    assert qwen.entry.source_quote.startswith("extracted_by:llm:qwen3.5:35b-a3b | ")
    assert gemini.entry.source_quote != qwen.entry.source_quote


def test_assemble_extraction_result_validation_failure_marks_rejected():
    """Per D3, if validate_response returns failures (e.g. form_not_in_body),
    the result is rejected — not an accepted-with-warnings entry. This is
    the load-bearing safety guard; assemble_extraction_result must not
    swallow it."""
    from wyrd.generators.kenning.extractors.llm import assemble_extraction_result

    response = {
        "found": True,
        "historical_form": "Foobar",
        "confidence": "medium",
        "elements": [
            {
                "form": "fictitious-form-not-in-body",
                "language": "old-english",
                "position": "pre",
                "gloss": "fake",
                "inflection": None,
            }
        ],
    }
    result = assemble_extraction_result(
        response,
        body="Foobar — origin unknown.",
        toponym="Foobar",
        suffix_hint=None,
        notes_prefix="extracted_by:test:1",
    )
    assert result.accepted is False
    assert result.entry is None
    assert any(f.reason == "form_not_in_body" for f in result.failures)


def test_assemble_extraction_result_tags_source_quote_with_prefix():
    """Successful extractions stamp source_quote with the provider's
    notes_prefix — D12 search-evidence vs. citation-evidence separation
    relies on this for per-provider attribution. Truncated to 500 chars
    (wyrd-9kh.2 raised the budget from 200 to give the SPA citation view
    enough context)."""
    from wyrd.generators.kenning.extractors.llm import assemble_extraction_result

    body = "Foobar — said to be from Old English foo, a foo."
    response = {
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
    }
    result = assemble_extraction_result(
        response,
        body=body,
        toponym="Foobar",
        suffix_hint=None,
        notes_prefix="extracted_by:claude:opus-4-7",
    )
    assert result.accepted is True
    assert result.entry is not None
    assert result.entry.source_quote.startswith("extracted_by:claude:opus-4-7")
    assert len(result.entry.source_quote) <= SOURCE_QUOTE_BUDGET


def test_assemble_extraction_result_truncates_long_body_at_budget():
    """wyrd-9kh.2: source_quote budget is 500 chars total. A long body
    must truncate at the budget on both the decline path (per wyrd-6hd
    now also prefixed: 'notes_prefix | body') and the accepted path
    ('combined_notes | body'). Pins the budget so a future change that
    drops the truncation gets caught before the SPA citation view
    starts emitting megabyte-sized rows.
    """
    from wyrd.generators.kenning.extractors.llm import assemble_extraction_result

    # 1200-char body — well over the 500 budget. The form 'foo' must
    # appear in body to pass the form-in-body check.
    body = ("Foobar — from Old English foo, " + ("a " * 600))[:1200]
    assert len(body) == 1200

    # Decline path: prefix + ' | ' + body, capped at the budget (wyrd-6hd).
    declined = assemble_extraction_result(
        {"found": False},
        body=body,
        toponym="Foobar",
        suffix_hint=None,
        notes_prefix="extracted_by:test:1",
    )
    assert declined.accepted is True
    assert declined.entry is not None
    assert len(declined.entry.source_quote) == SOURCE_QUOTE_BUDGET
    expected_decline = f"extracted_by:test:1 | {body}"[:SOURCE_QUOTE_BUDGET]
    assert declined.entry.source_quote == expected_decline

    # Accepted path: combined_notes + ' | ' + body capped at the budget.
    response = {
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
    }
    accepted = assemble_extraction_result(
        response,
        body=body,
        toponym="Foobar",
        suffix_hint=None,
        notes_prefix="extracted_by:test:1",
    )
    assert accepted.accepted is True
    assert accepted.entry is not None
    sq = accepted.entry.source_quote
    assert len(sq) == SOURCE_QUOTE_BUDGET
    # Attribution prefix preserved at the front; body share fills the rest.
    assert sq.startswith("extracted_by:test:1")
    assert " | " in sq


def test_assemble_extraction_result_outer_cap_truncates_long_notes_prefix():
    """wyrd-9kh.2 (review-round-1): the outer ``[:SOURCE_QUOTE_BUDGET]``
    cap on the accepted path must clamp the result even when
    ``combined_notes + " | " + body`` overflows because the notes string
    is unusually long. Without an explicit test, a future refactor that
    drops the outer cap would silently let the source_quote grow
    unbounded for verbose notes_prefixes.
    """
    from wyrd.generators.kenning.extractors.llm import assemble_extraction_result

    body = "Foobar — from Old English foo, a foo. " + ("x " * 250)
    long_prefix = "extracted_by:test:" + ("a" * 300)  # ~318 chars
    assert len(long_prefix) + len(" | ") + len(body) > SOURCE_QUOTE_BUDGET
    response = {
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
    }
    result = assemble_extraction_result(
        response,
        body=body,
        toponym="Foobar",
        suffix_hint=None,
        notes_prefix=long_prefix,
    )
    assert result.accepted is True
    assert result.entry is not None
    assert len(result.entry.source_quote) == SOURCE_QUOTE_BUDGET
    assert result.entry.source_quote.startswith("extracted_by:test:")


def test_assemble_extraction_result_short_total_preserves_full_body():
    """wyrd-9kh.2 (review-round-2): when the assembled
    ``combined_notes + " | " + body`` is shorter than the budget, the
    whole body is preserved (no truncation). Pins the post-simplification
    behavior: body share is bounded only by the outer cap, not by an
    artificial inner slice — so a short notes_prefix gives the body the
    maximum available context.
    """
    from wyrd.generators.kenning.extractors.llm import assemble_extraction_result

    short_prefix = "p"
    body = "Foobar — from Old English foo, a foo." + ("y" * 100)
    assert len(short_prefix) + len(" | ") + len(body) < SOURCE_QUOTE_BUDGET
    response = {
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
    }
    result = assemble_extraction_result(
        response,
        body=body,
        toponym="Foobar",
        suffix_hint=None,
        notes_prefix=short_prefix,
    )
    assert result.accepted is True
    assert result.entry is not None
    # Full body is present; no truncation kicked in.
    assert result.entry.source_quote.endswith(body)
    assert result.entry.source_quote == short_prefix + " | " + body


# --- OllamaClient transport-layer wraps (wyrd-pe4g rounds 4-5) -----------
#
# Ollama is the default --provider; the JSONDecodeError + UnicodeDecodeError
# wraps on its chat_json path are load-bearing for the
# _PROGRAMMER_ERROR_EXCEPTIONS contract in toponym_mention_extractor.py.
# Tests below mirror the Anthropic/Gemini patterns so a future refactor
# can't silently regress the most-used client.


class _RawBytesResp:
    """Minimal context-manager-shaped urlopen response that returns raw bytes."""

    def __init__(self, raw: bytes) -> None:
        self._raw = raw

    def read(self) -> bytes:
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        pass


def test_ollama_chat_json_malformed_envelope_raises_runtimeerror() -> None:
    """wyrd-pe4g round-4: a 2xx response with non-JSON body (proxy HTML
    error page, truncated body on a dropped connection) must surface as
    RuntimeError so the toponym mining chunk loop buckets it into
    chunks_failed. Without the outer envelope-parse wrap in
    llm_extractor.py, json.JSONDecodeError (a ValueError subclass) would
    propagate through _PROGRAMMER_ERROR_EXCEPTIONS (which lists
    ValueError to surface schema_dialect typos) and abort the whole
    multi-source run. Ollama is the default --provider; this is the
    most-used client to keep covered."""
    client = OllamaClient()
    malformed = b"<html><body>503 Service Unavailable</body></html>"
    with (
        patch("urllib.request.urlopen", lambda req, timeout=None: _RawBytesResp(malformed)),
        pytest.raises(RuntimeError, match="non-JSON envelope"),
    ):
        client.chat_json("sys", "usr", {})


def test_ollama_chat_json_non_json_content_raises_runtimeerror() -> None:
    """wyrd-rmc6: the inner content parse (message.content → JSON) must also
    raise RuntimeError on JSONDecodeError, not let it leak through
    _PROGRAMMER_ERROR_EXCEPTIONS as ValueError. This pins the helper
    extraction: a regression that drops the parse_transport_json wrap on
    the content path would abort the whole multi-source run on a single
    malformed model response."""
    client = OllamaClient()
    # Outer envelope is valid; message.content is non-JSON garbage.
    envelope = json.dumps({"message": {"content": "this is not json {{"}}).encode("utf-8")
    with (
        patch("urllib.request.urlopen", lambda req, timeout=None: _RawBytesResp(envelope)),
        pytest.raises(RuntimeError, match="non-JSON content"),
    ):
        client.chat_json("sys", "usr", {})


def test_ollama_chat_json_non_utf8_body_raises_runtimeerror() -> None:
    """wyrd-pe4g round-5 silent-failure-hunter: a 2xx response whose
    body isn't valid UTF-8 (binary blob, corrupted upstream, proxy
    injection) must also surface as RuntimeError. UnicodeDecodeError
    is a ValueError subclass — without the `.decode("utf-8")` wrap on
    the success path it leaks through _PROGRAMMER_ERROR_EXCEPTIONS and
    aborts the run identically to the round-4 envelope-parse leak."""
    client = OllamaClient()
    # 0xff is invalid as a UTF-8 leading byte.
    non_utf8 = b"\xff\xfe\xfd binary garbage"
    with (
        patch("urllib.request.urlopen", lambda req, timeout=None: _RawBytesResp(non_utf8)),
        pytest.raises(RuntimeError, match="non-UTF-8 body"),
    ):
        client.chat_json("sys", "usr", {})


# --- OllamaClient config-drift resilience (wyrd-dyf8) --------------------


class _PayloadCapturingResp:
    """urlopen-shaped response shell. Returns a fixed body via the
    context-manager protocol so chat_json's success path completes.

    Request-payload capture happens in the standalone
    :func:`_capture_request_payload` (below) — that function decodes
    ``req.data`` and stashes it on this class's ``captured`` list
    before returning an instance of us. Tests then assert against
    ``_PayloadCapturingResp.captured``."""

    captured: list[dict] = []

    def __init__(self, raw: bytes) -> None:
        self._raw = raw

    def read(self) -> bytes:
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        pass


def _capture_request_payload(req, timeout=None):
    """urlopen replacement that decodes the POST body, stashes it on
    :attr:`_PayloadCapturingResp.captured` for assertion, and returns
    a minimal valid envelope so chat_json's success path completes."""
    _PayloadCapturingResp.captured.append(json.loads(req.data.decode("utf-8")))
    ok = json.dumps({"message": {"content": "{}"}}).encode("utf-8")
    return _PayloadCapturingResp(ok)


def test_ollama_chat_json_sends_explicit_num_ctx() -> None:
    """wyrd-dyf8: every chat_json request must include num_ctx in
    options. Without this, Ollama loads gemma4:26b at its absolute-
    max 262K context when the host's env defaults to that — concrete
    2026-05-22 incident where a macbook reboot doubled mining
    chunk-time. Pinning the per-request num_ctx makes the wyrd
    client immune to that class of config drift.

    Also asserts ``think: False`` is preserved at the payload top
    level — qwen3.5 and other reasoning models consume the token
    budget on internal chain-of-thought when think is unset/true,
    leaving no content in the response. A future refactor that
    consolidates the payload dict could silently drop this guard;
    the assertion turns that into a test failure."""
    _PayloadCapturingResp.captured = []
    client = OllamaClient()
    with patch("urllib.request.urlopen", _capture_request_payload):
        client.chat_json("sys", "usr", {})
    assert len(_PayloadCapturingResp.captured) == 1
    payload = _PayloadCapturingResp.captured[0]
    assert "num_ctx" in payload["options"], "num_ctx must be sent on every request"
    # wyrd-qxmr: 8192 is right-sized for the production staged miner
    # (3000-char chunks ~750 tokens). The single-tier CLI's 20000-
    # char chunks may need an override (see OllamaClient.num_ctx
    # docstring). PR #327 originally bumped this to 16384 but
    # empirical measurement showed 2-3x slower inference with no
    # production benefit.
    assert payload["options"]["num_ctx"] == 8192
    # qwen3.5 reasoning-channel guard preserved.
    assert payload.get("think") is False


def test_ollama_chat_json_sends_format_json_string_not_schema_dict() -> None:
    """wyrd-sj50: every chat_json request must send ``format='json'``
    (legacy mode) NOT the schema dict. Concrete failure 2026-05-22:
    Ollama 0.24+ on the macbook hangs indefinitely on chat requests
    with ``format=<schema dict>`` — probe with the equivalent
    ``format='json'`` returns in 500ms. The dict-grammar-constrained
    path is broken upstream; until Ollama fixes it, we must send the
    legacy string. Caller-side validation in ``_validated_mentions``
    catches malformed responses anyway, so losing server-side schema
    enforcement is safe.

    A regression that re-routed schema dicts through ``format`` would
    silently re-introduce the production hang."""
    _PayloadCapturingResp.captured = []
    client = OllamaClient()
    test_schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    with patch("urllib.request.urlopen", _capture_request_payload):
        client.chat_json("sys", "usr", test_schema)
    payload = _PayloadCapturingResp.captured[0]
    assert payload["format"] == "json", (
        f"format must be the literal string 'json' (legacy mode), "
        f"not a schema dict. Got: {payload['format']!r}. The dict-"
        f"grammar-constrained path is broken in Ollama 0.24+; see "
        f"wyrd-sj50."
    )
    # The schema parameter was accepted but deliberately discarded —
    # not in the payload anywhere.
    assert payload["format"] != test_schema


def test_ollama_chat_json_sends_keep_alive() -> None:
    """wyrd-dyf8: every chat_json request must send keep_alive at the
    payload top level (NOT inside options — Ollama silently ignores
    keep_alive nested under options). Default -1 pins the model
    resident across the run so a 5-minute lull between chunks
    doesn't trigger a 30-60s reload from disk."""
    _PayloadCapturingResp.captured = []
    client = OllamaClient()
    with patch("urllib.request.urlopen", _capture_request_payload):
        client.chat_json("sys", "usr", {})
    payload = _PayloadCapturingResp.captured[0]
    assert "keep_alive" in payload, "keep_alive must be at payload top level"
    assert "keep_alive" not in payload["options"], (
        "keep_alive nested under options is silently ignored by Ollama; "
        "must be at payload top level"
    )
    assert payload["keep_alive"] == -1


def test_ollama_chat_json_respects_custom_num_ctx_and_keep_alive() -> None:
    """Overrides at construction must flow through to the request
    payload — keeps the per-call defaults from becoming a tunnel
    that ignores caller intent. Also re-pins the keep_alive
    placement (top level, NOT under options) on the override path
    so a future refactor that moved keep_alive into options for
    overrides specifically would still fail this test."""
    _PayloadCapturingResp.captured = []
    client = OllamaClient(num_ctx=16384, keep_alive=3600)
    with patch("urllib.request.urlopen", _capture_request_payload):
        client.chat_json("sys", "usr", {})
    payload = _PayloadCapturingResp.captured[0]
    assert payload["options"]["num_ctx"] == 16384
    assert payload["keep_alive"] == 3600
    # Re-pin top-level placement on the override path (the defaults
    # test asserts this too; both paths must enforce the structural
    # contract because Ollama silently ignores keep_alive nested
    # under options).
    assert "keep_alive" not in payload["options"]
