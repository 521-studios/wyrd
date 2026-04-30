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
