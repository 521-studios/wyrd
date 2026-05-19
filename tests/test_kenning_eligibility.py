"""Tests for wyrd-ecjp.3 Phase 3: eligibility-gate runtime.

Exercises the gate predicates against synthetic Meaning objects.
Production Meanings are loaded from the bundle; tests construct
Meanings directly to pin specific shape combinations.
"""

from __future__ import annotations

import pytest

from wyrd.generators.kenning.eligibility import (
    UnknownCultureError,
    admits,
    filter_meanings,
    passes_culture_gate,
    passes_era_gate,
    passes_pack_gate,
    passes_stratum_gate,
    passes_tag_excluded_gate,
    passes_tag_required_gate,
)
from wyrd.generators.kenning.runtime.meaning import Meaning
from wyrd.generators.kenning.vectors.schemas import (
    EligibilityGate,
    PackOverlay,
)

# ---------- fixtures -----------------------------------------------------


def _meaning(
    usage: str = "-tūn",
    tags: tuple[str, ...] = (),
    attested_years: dict[str, list[tuple[str, int]]] | None = None,
    stratum: dict[str, dict[str, str]] | None = None,
) -> Meaning:
    """Build a minimal Meaning for gate testing. Other fields default
    to empty (the bundle's optional-field model)."""
    return Meaning(
        usage=usage,
        tags=list(tags),
        meanings=["test"],
        sources=[],
        attested_years=attested_years,
        stratum=stratum,
    )


def _gate(
    culture: str = "english",
    era_min: int | None = None,
    era_max: int | None = None,
    stratum: str | None = None,
    allowed_pack_tags: frozenset[str] = frozenset(),
    excluded_pack_tags: frozenset[str] = frozenset(),
) -> EligibilityGate:
    return EligibilityGate(
        culture=culture,
        era_min=era_min,
        era_max=era_max,
        stratum=stratum,
        allowed_pack_tags=allowed_pack_tags,
        excluded_pack_tags=excluded_pack_tags,
    )


# ---------- passes_culture_gate -----------------------------------------


def test_culture_gate_accepts_english():
    assert passes_culture_gate("english") is True


def test_culture_gate_accepts_all_v1_cultures():
    """Every culture in the v1 CULTURES list passes."""
    for culture in ("english", "scottish", "welsh", "irish", "breton"):
        assert passes_culture_gate(culture) is True


def test_culture_gate_raises_on_unknown_culture():
    """Unknown culture string → loud UnknownCultureError. Fails
    requests where a misconfigured front-end passes 'klingon' rather
    than silently selecting from the union of all cultures."""
    with pytest.raises(UnknownCultureError, match="unknown culture 'klingon'"):
        passes_culture_gate("klingon")


def test_culture_gate_error_lists_available_cultures():
    """Operator-friendly diagnostic: the error message lists the
    valid cultures so a typo'd request is actionable."""
    with pytest.raises(UnknownCultureError) as exc_info:
        passes_culture_gate("englisch")  # typo
    msg = str(exc_info.value)
    for culture in ("english", "scottish", "welsh", "irish", "breton"):
        assert culture in msg


# ---------- passes_era_gate ---------------------------------------------


def test_era_gate_no_filter_passes_everything():
    """era_min=None AND era_max=None → no filter applied."""
    m = _meaning(attested_years={"old_english": [("tūn", 950)]})
    assert passes_era_gate(m, None, None) is True


def test_era_gate_meaning_in_window_passes():
    m = _meaning(attested_years={"old_english": [("tūn", 950)]})
    assert passes_era_gate(m, 800, 1100) is True


def test_era_gate_meaning_outside_window_fails():
    m = _meaning(attested_years={"old_english": [("tūn", 1500)]})
    assert passes_era_gate(m, 800, 1100) is False


def test_era_gate_meaning_with_no_attested_years_passes():
    """ "No data → pass" rule (D5 / D5-3 wyrd-lyp): a Meaning with no attested-year
    data passes any era filter. Most legacy bundle Meanings fall
    here today; tightening this rule waits on mining coverage rising.
    """
    m = _meaning(attested_years={})
    assert passes_era_gate(m, 800, 1100) is True


def test_era_gate_half_open_window():
    """`[start, end)` half-open semantics matching D5 / D5-3 (wyrd-lyp)."""
    m_at_start = _meaning(attested_years={"old_english": [("a", 800)]})
    m_at_end = _meaning(attested_years={"old_english": [("a", 1100)]})
    m_just_under_end = _meaning(attested_years={"old_english": [("a", 1099)]})
    assert passes_era_gate(m_at_start, 800, 1100) is True
    assert passes_era_gate(m_at_end, 800, 1100) is False
    assert passes_era_gate(m_just_under_end, 800, 1100) is True


def test_era_gate_only_lower_bound_set():
    """era_min set, era_max=None → 'attested after start' filter."""
    m_after = _meaning(attested_years={"old_english": [("a", 1500)]})
    m_before = _meaning(attested_years={"old_english": [("a", 500)]})
    assert passes_era_gate(m_after, 800, None) is True
    assert passes_era_gate(m_before, 800, None) is False


def test_era_gate_only_upper_bound_set():
    """era_min=None, era_max set → 'attested before end' filter."""
    m_after = _meaning(attested_years={"old_english": [("a", 1500)]})
    m_before = _meaning(attested_years={"old_english": [("a", 500)]})
    assert passes_era_gate(m_after, None, 1100) is False
    assert passes_era_gate(m_before, None, 1100) is True


# ---------- passes_stratum_gate ----------------------------------------


def test_stratum_gate_no_filter_passes_everything():
    m = _meaning(stratum={"welsh": {"tref": "native-welsh"}})
    assert passes_stratum_gate(m, None) is True


def test_stratum_gate_meaning_in_stratum_passes():
    m = _meaning(stratum={"welsh": {"tref": "native-welsh"}})
    assert passes_stratum_gate(m, "native-welsh") is True


def test_stratum_gate_meaning_in_different_stratum_fails():
    m = _meaning(stratum={"welsh": {"tref": "english-loan"}})
    assert passes_stratum_gate(m, "native-welsh") is False


def test_stratum_gate_meaning_with_no_stratum_data_passes():
    """wyrd-lr4 Phase 3 'no data → pass' rule mirrors the era rule.
    Only Welsh-family etymons are classified today; routing every
    culture through a strict stratum gate would gut bundles for
    unclassified families."""
    m = _meaning(stratum={})
    assert passes_stratum_gate(m, "native-welsh") is True


# ---------- passes_tag_required_gate -----------------------------------


def test_tag_required_empty_set_passes_everything():
    m = _meaning(tags=("plant",))
    assert passes_tag_required_gate(m, frozenset()) is True


def test_tag_required_meaning_with_empty_tags_fails_non_empty_required():
    """Meaning with no tags at all and a non-empty required-set →
    fails. Common production case (many bundle Meanings carry no
    semantic tags); pin the behavior so a regression silently
    flipping the rule is caught."""
    m = _meaning(tags=())
    assert passes_tag_required_gate(m, frozenset({"plant"})) is False


def test_tag_required_meaning_with_empty_tags_passes_empty_required():
    """Meaning with no tags + empty required-set → passes. Both
    empty-set fast-paths exercised together."""
    m = _meaning(tags=())
    assert passes_tag_required_gate(m, frozenset()) is True


def test_tag_required_single_tag_passes_when_present():
    m = _meaning(tags=("plant", "tree"))
    assert passes_tag_required_gate(m, frozenset({"plant"})) is True


def test_tag_required_single_tag_fails_when_absent():
    m = _meaning(tags=("water",))
    assert passes_tag_required_gate(m, frozenset({"plant"})) is False


def test_tag_required_multiple_tags_are_AND_semantics():
    """All required tags must appear. Pins AND semantics for
    `--tag plant --tag tree`."""
    m_both = _meaning(tags=("plant", "tree", "agriculture"))
    m_one = _meaning(tags=("plant",))
    m_neither = _meaning(tags=("water",))
    assert passes_tag_required_gate(m_both, frozenset({"plant", "tree"})) is True
    assert passes_tag_required_gate(m_one, frozenset({"plant", "tree"})) is False
    assert passes_tag_required_gate(m_neither, frozenset({"plant", "tree"})) is False


# ---------- passes_tag_excluded_gate -----------------------------------


def test_tag_excluded_empty_set_passes_everything():
    m = _meaning(tags=("fiction",))
    assert passes_tag_excluded_gate(m, frozenset()) is True


def test_tag_excluded_meaning_with_excluded_tag_fails():
    """Pairs with --exclude-tags (wyrd-yan)."""
    m = _meaning(tags=("fiction", "plant"))
    assert passes_tag_excluded_gate(m, frozenset({"fiction"})) is False


def test_tag_excluded_meaning_without_excluded_tag_passes():
    m = _meaning(tags=("plant",))
    assert passes_tag_excluded_gate(m, frozenset({"fiction"})) is True


def test_tag_excluded_multiple_tags_are_any_match():
    """ANY excluded tag matching disqualifies. Pins union-exclude
    semantics."""
    m_one_excluded = _meaning(tags=("fiction", "plant"))
    m_other_excluded = _meaning(tags=("manorial", "plant"))
    m_neither_excluded = _meaning(tags=("plant",))
    excluded = frozenset({"fiction", "manorial"})
    assert passes_tag_excluded_gate(m_one_excluded, excluded) is False
    assert passes_tag_excluded_gate(m_other_excluded, excluded) is False
    assert passes_tag_excluded_gate(m_neither_excluded, excluded) is True


# ---------- passes_pack_gate (stub behavior) ---------------------------


def test_pack_gate_returns_true_for_v1_meaning():
    """Pack gates are stubs today (no Meaning carries pack metadata).
    Pins the no-op behavior for the empty-pack-tags case so when
    scenario packs land this test will need to be updated."""
    m = _meaning()
    g = _gate()
    assert passes_pack_gate(m, g, ()) is True


def test_pack_gate_raises_when_allowed_pack_tags_non_empty():
    """A caller passing allowed_pack_tags is expressing intent to
    filter — but pack-tag filtering isn't implemented yet. Raise
    rather than silently returning True (which would silently
    accept the caller's intent and produce wrong results)."""
    m = _meaning(tags=("plant",))
    g = _gate(allowed_pack_tags=frozenset({"khuzdul"}))
    with pytest.raises(NotImplementedError, match="pack-tag filtering"):
        passes_pack_gate(m, g, ())


def test_pack_gate_raises_when_excluded_pack_tags_non_empty():
    """Same fail-loud behavior for excluded_pack_tags."""
    m = _meaning(tags=("plant",))
    g = _gate(excluded_pack_tags=frozenset({"khuzdul"}))
    with pytest.raises(NotImplementedError, match="pack-tag filtering"):
        passes_pack_gate(m, g, ())


def test_pack_gate_returns_true_with_packs_argument():
    """The ``packs`` argument is accepted today but not yet used
    by the predicate; pin the signature-stable contract."""
    m = _meaning()
    g = _gate()
    packs = (
        PackOverlay(
            pack_name="neo-khuzdul",
            template_donor="old-norse",
            template_recipient="old-english",
        ),
    )
    assert passes_pack_gate(m, g, packs) is True


# ---------- admits (top-level all-gates predicate) ---------------------


def test_admits_passes_when_all_gates_pass():
    m = _meaning(
        tags=("plant",),
        attested_years={"old_english": [("a", 950)]},
        stratum={"old_english": {"a": "native-old-english"}},
    )
    g = _gate(culture="english", era_min=800, era_max=1100, stratum="native-old-english")
    assert (
        admits(m, g, tag_required=frozenset({"plant"}), tag_excluded=frozenset({"fiction"})) is True
    )


def test_admits_fails_when_any_one_gate_fails():
    """Era passes, stratum passes, tag-required fails → admits False."""
    m = _meaning(
        tags=("water",),  # missing required 'plant'
        attested_years={"old_english": [("a", 950)]},
    )
    g = _gate(era_min=800, era_max=1100)
    assert admits(m, g, tag_required=frozenset({"plant"})) is False


def test_admits_does_NOT_validate_culture():
    """``admits`` is the per-Meaning hot path; culture validation
    happens ONCE in ``filter_meanings`` before the loop. Unknown
    culture passed straight to ``admits`` does not raise here —
    callers that bypass ``filter_meanings`` are responsible for
    validation."""
    m = _meaning()
    g = _gate(culture="klingon")  # unknown — would raise via filter_meanings
    # admits itself doesn't validate culture; it returns based on
    # the other gate predicates. With an empty gate, returns True.
    assert admits(m, g) is True


def test_admits_short_circuits_on_first_failing_gate():
    """A Meaning that fails the tag-required gate doesn't pay the
    cost of the era/stratum/pack predicates. Uses a Mock-like
    subclass that raises if ``attested_in_era_range`` is ever
    called — short-circuit semantics are pinned by the absence of
    the raise."""

    class _MeaningThatRaisesOnEra(Meaning):
        def attested_in_era_range(self, era_range):  # type: ignore[override]
            raise AssertionError("era gate should not have been reached — short-circuit failed")

    m = _MeaningThatRaisesOnEra(usage="-x", tags=[], meanings=["test"], sources=[])
    g = _gate(era_min=800, era_max=1100)
    # tag_required fails first → era predicate never called → no AssertionError.
    assert admits(m, g, tag_required=frozenset({"plant"})) is False


def test_admits_each_gate_fails_independently():
    """Per-gate failure isolation: hand each gate a Meaning that
    fails ONLY that gate (with all others passing). Pins that
    admits returns False for each early-return path, not just for
    the combined integration case."""
    # Tag-required failure
    m_no_required = _meaning(tags=("water",))
    assert admits(m_no_required, _gate(), tag_required=frozenset({"plant"})) is False
    # Tag-excluded failure
    m_excluded = _meaning(tags=("plant", "fiction"))
    assert admits(m_excluded, _gate(), tag_excluded=frozenset({"fiction"})) is False
    # Era failure (attested data outside window)
    m_late = _meaning(tags=("plant",), attested_years={"old_english": [("l", 1500)]})
    assert admits(m_late, _gate(era_min=800, era_max=1100)) is False
    # Stratum failure
    m_wrong_stratum = _meaning(tags=("plant",), stratum={"welsh": {"x": "english-loan"}})
    assert admits(m_wrong_stratum, _gate(stratum="native-welsh")) is False


# ---------- filter_meanings ---------------------------------------------


def test_filter_meanings_empty_input_returns_empty_list():
    g = _gate()
    assert filter_meanings([], g) == []


def test_filter_meanings_returns_only_passing_meanings():
    m1 = _meaning(usage="-a", tags=("plant",))
    m2 = _meaning(usage="-b", tags=("water",))
    m3 = _meaning(usage="-c", tags=("plant", "tree"))
    g = _gate()
    result = filter_meanings([m1, m2, m3], g, tag_required=frozenset({"plant"}))
    assert len(result) == 2
    assert m1 in result
    assert m3 in result
    assert m2 not in result


def test_filter_meanings_all_fail_returns_empty():
    """Pool of Meanings where EVERY one fails some gate → []. A
    buggy gate that leaks one extra Meaning would silently pass
    the existing 'integration with one winner' test; this test
    pins the all-fail case."""
    m1 = _meaning(usage="-a", tags=("water",))
    m2 = _meaning(usage="-b", tags=("fish",))
    m3 = _meaning(usage="-c", tags=("stone",))
    g = _gate()
    result = filter_meanings([m1, m2, m3], g, tag_required=frozenset({"plant"}))
    assert result == []


def test_filter_meanings_validates_culture_once_upfront():
    """Unknown culture → UnknownCultureError raised BEFORE the
    per-Meaning loop. Pins the validation-once-not-N-times contract."""
    m = _meaning(tags=("plant",))
    g = _gate(culture="klingon")
    with pytest.raises(UnknownCultureError, match="unknown culture"):
        filter_meanings([m], g, tag_required=frozenset({"plant"}))


def test_filter_meanings_culture_validation_runs_even_for_empty_pool():
    """Empty pool → validation still runs and still raises on
    unknown culture. Pins the order (validation before iteration)."""
    g = _gate(culture="klingon")
    with pytest.raises(UnknownCultureError):
        filter_meanings([], g)


def test_filter_meanings_preserves_input_order():
    """Order matters for downstream consumers that walk the filtered
    list for slot-filling. Pin the contract that the filter is a
    stable selection, not a re-ordering."""
    m_a = _meaning(usage="-a", tags=("plant",))
    m_b = _meaning(usage="-b", tags=("plant",))
    m_c = _meaning(usage="-c", tags=("plant",))
    g = _gate()
    result = filter_meanings([m_c, m_a, m_b], g, tag_required=frozenset({"plant"}))
    assert [m.usage for m in result] == ["-c", "-a", "-b"]


def test_filter_meanings_with_all_gates_simultaneously():
    """Integration test: pool of Meanings with mixed shapes; assert
    only the one that passes ALL gates makes it through."""
    # Wins: english culture, era 800-1100, native-old-english,
    # tag plant, no fiction tag.
    m_winner = _meaning(
        usage="-winner",
        tags=("plant", "tree"),
        attested_years={"old_english": [("w", 900)]},
        stratum={"old_english": {"w": "native-old-english"}},
    )
    # Loses on era (year 1500).
    m_late_era = _meaning(
        usage="-late",
        tags=("plant",),
        attested_years={"old_english": [("l", 1500)]},
        stratum={"old_english": {"l": "native-old-english"}},
    )
    # Loses on tag-required (no plant).
    m_no_plant = _meaning(
        usage="-noplant",
        tags=("water",),
        attested_years={"old_english": [("n", 900)]},
    )
    # Loses on tag-excluded (has fiction).
    m_fiction = _meaning(
        usage="-fiction",
        tags=("plant", "fiction"),
        attested_years={"old_english": [("f", 900)]},
    )
    # Loses on stratum.
    m_wrong_stratum = _meaning(
        usage="-loan",
        tags=("plant",),
        attested_years={"old_english": [("ld", 900)]},
        stratum={"old_english": {"ld": "norse-loan"}},
    )
    # Has no era data (passes era via 'no data → pass') BUT has
    # stratum data that fails the stratum gate. Confirms the two
    # 'no data → pass' rules don't mask each other: a Meaning that
    # passes one no-data rule still has to pass other gates that
    # DO have data.
    m_no_era_failing_stratum = _meaning(
        usage="-mixed",
        tags=("plant",),
        attested_years={},
        stratum={"old_english": {"mx": "norse-loan"}},
    )

    g = _gate(
        culture="english",
        era_min=800,
        era_max=1100,
        stratum="native-old-english",
    )
    result = filter_meanings(
        [
            m_winner,
            m_late_era,
            m_no_plant,
            m_fiction,
            m_wrong_stratum,
            m_no_era_failing_stratum,
        ],
        g,
        tag_required=frozenset({"plant"}),
        tag_excluded=frozenset({"fiction"}),
    )

    assert result == [m_winner]


def test_filter_meanings_no_filters_passes_all():
    """Bare gate with no constraints + empty tag filters → no
    filtering. Pins the default-behavior contract."""
    m1 = _meaning(usage="-a")
    m2 = _meaning(usage="-b")
    g = _gate()  # culture only; all other fields default to None / frozenset()
    result = filter_meanings([m1, m2], g)
    assert result == [m1, m2]
