"""Tests for the LLM extractor's validation guard.

The Ollama transport itself isn't tested here (would need a live endpoint or
a mock layer). We focus on `validate_response`, the safety check that
prevents hallucinated etymons from being ingested. That guard is the
load-bearing piece of the LLM pipeline — if it ever silently breaks,
garbage flows into the lexicon.
"""

from __future__ import annotations

import re

from wyrd.generators.kenning.llm_extractor import (
    RESPONSE_SCHEMA,
    SYSTEM_PROMPT,
    _form_in_body,
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


# --- form-in-body checks ---------------------------------------------------


def test_form_in_body_basic_substring() -> None:
    body = "This is the A.S. bere-tun, lit. 'corn-farm'."
    assert _form_in_body("bere", body)
    assert _form_in_body("tun", body)
    assert not _form_in_body("xyz", body)


def test_form_in_body_handles_ocr_variants() -> None:
    """Model emits 'hædan' but body has OCR-mangled 'Hcsdan' — should still match."""
    body = "O.E. Hcsdan-ham. The personal name."
    assert _form_in_body("hædan", body)
    assert _form_in_body("Hædan", body)
    assert _form_in_body("haedan", body)


def test_form_in_body_rejects_short_forms() -> None:
    """A single-character or empty form can't be a meaningful etymon —
    the substring match would too easily collide with random text."""
    body = "Some etymology text with ab embedded inside."
    assert not _form_in_body("", body)
    assert not _form_in_body("a", body)
    # 2-char forms are technically the minimum we accept; "ab" is in body.
    assert _form_in_body("ab", body)


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
    """Years before 800 are almost always folio numbers or page numbers
    misread as years."""
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
    # Year-range constraint is documented.
    assert "800" in prompt and "1700" in prompt
    # Examples include real Domesday/charter-style dated citations.
    assert "1333" in prompt or "1226" in prompt or "1086" in prompt
    # Skip-noise instruction is present (publication years vs attestations).
    assert "publication" in prompt
    # No-invention instruction is present.
    assert "do not invent" in prompt or "do not invent dates" in prompt


def test_response_schema_includes_attested_forms() -> None:
    """The Ollama JSON schema must mark attested_forms as required, with
    year integer constrained to 800-1700. If this constraint is removed
    the model can emit publication-year noise unchecked."""
    assert "attested_forms" in RESPONSE_SCHEMA["properties"]
    assert "attested_forms" in RESPONSE_SCHEMA["required"]
    item_schema = RESPONSE_SCHEMA["properties"]["attested_forms"]["items"]
    year_schema = item_schema["properties"]["year"]
    assert year_schema["type"] == "integer"
    assert year_schema["minimum"] == 800
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
    """Same dash-stripping the elements check applies — model can't bypass
    the length floor by wrapping a single character in dashes."""
    response = _ok_response()
    response["attested_forms"] = [{"form": "Aburwick-", "year": 1333}]
    failures = validate_response(response, _DATED_BODY)
    # Should pass — "Aburwick" (without dash) is in _DATED_BODY.
    assert not any(f.reason == "attested_form_not_in_body" for f in failures)


def test_validate_response_attested_year_at_lower_boundary() -> None:
    """Inclusive lower bound: year=800 is the floor for the attested-year
    range (per _ATTESTED_YEAR_MIN). Off-by-one regression guard."""
    body = _DATED_BODY + " Aburwick is also recorded in 800."
    response = _ok_response()
    response["attested_forms"] = [{"form": "Aburwick", "year": 800}]
    failures = validate_response(response, body)
    assert not any(f.reason.startswith("attested_year") for f in failures)


def test_validate_response_attested_year_below_lower_boundary() -> None:
    """Year=799 is below the 800 floor and should be rejected."""
    response = _ok_response()
    response["attested_forms"] = [{"form": "bere", "year": 799}]
    failures = validate_response(response, _BARTON_BODY)
    assert any(f.reason == "attested_year_out_of_range" for f in failures)


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
    from wyrd.generators.kenning.llm_extractor import transport_error_result

    result = transport_error_result("HTTP 503 Service Unavailable")
    assert result.accepted is False
    assert result.entry is None
    assert result.raw_response == {"error": "HTTP 503 Service Unavailable"}
    assert len(result.failures) == 1
    assert result.failures[0].reason == "transport_error"
    assert "HTTP 503" in result.failures[0].detail


def test_assemble_extraction_result_not_found_emits_low_confidence_entry():
    """Per D14, when the response says found=False the result is an
    accepted ParsedEntry with empty elements and confidence='low'. The
    notes_prefix is ignored on this path (no source_quote tagging) since
    the body itself is the source_quote."""
    from wyrd.generators.kenning.llm_extractor import assemble_extraction_result

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
    assert result.entry.source_quote == body[:200]


def test_assemble_extraction_result_validation_failure_marks_rejected():
    """Per D3, if validate_response returns failures (e.g. form_not_in_body),
    the result is rejected — not an accepted-with-warnings entry. This is
    the load-bearing safety guard; assemble_extraction_result must not
    swallow it."""
    from wyrd.generators.kenning.llm_extractor import assemble_extraction_result

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
    relies on this for per-provider attribution. Truncated to 200 chars."""
    from wyrd.generators.kenning.llm_extractor import assemble_extraction_result

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
    assert len(result.entry.source_quote) <= 200
