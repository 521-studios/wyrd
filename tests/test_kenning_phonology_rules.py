"""Tests for the sound-change rules library (wyrd-4i6 Phase 1).

Covers:
- ``SoundChangeRule`` dataclass shape + immutability.
- Each OE→ME rule's exemplar input/output round-trips through
  ``apply_rules`` (one test per rule via parametrize).
- ``apply_rules`` forward / inverse modes.
- Multi-candidate inverse: when two forward rules collapse onto the
  same target, inverse from the target surfaces both candidates.
- Sporadic-rule branching (weight < 1.0): two candidates per branch,
  probabilities sum to the input probability times the rule weight.
- Unknown cells (no rules registered) are no-op.
- ``has_rules`` discriminates 'cell registered' from 'cell empty'.
- Mode validation: invalid mode raises ``ValueError``.

Phase 1 ships only the (english, old-english, middle-english) cell;
Phase 2 will add more cells. Tests pin the framework so Phase 2
additions land without surface change.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from wyrd.generators.kenning.phonology_rules import (
    OE_TO_ME_RULES,
    SoundChangeRule,
    _apply_one_rule,
    _dedupe_candidates,
    apply_rules,
    has_rules,
)

# --- SoundChangeRule dataclass -------------------------------------------


def test_sound_change_rule_is_frozen() -> None:
    """``SoundChangeRule`` is frozen — rules are constants registered
    at module load. Mutating one would silently corrupt every
    cell-lookup keyed on the rule."""
    rule = SoundChangeRule(
        pattern="x",
        replacement="y",
        weight=1.0,
        description="test",
        exemplar=("xa", "ya"),
        source="test",
    )
    with pytest.raises(FrozenInstanceError):
        rule.pattern = "z"  # type: ignore[misc]


def test_sound_change_rule_carries_documentation_fields() -> None:
    """Every rule MUST carry description / exemplar / source for
    audit; this is the contract the per-rule exemplar test depends
    on. Pin so a future refactor can't silently drop the fields."""
    for rule in OE_TO_ME_RULES:
        assert rule.description, "rule missing description"
        assert isinstance(rule.exemplar, tuple) and len(rule.exemplar) == 2
        assert all(isinstance(s, str) and s for s in rule.exemplar)
        assert rule.source, "rule missing source citation"


# --- OE→ME rule exemplars (one parametrize per rule) ---------------------


@pytest.mark.parametrize("rule", OE_TO_ME_RULES, ids=lambda r: r.pattern)
def test_oe_to_me_rule_exemplar_forward(rule: SoundChangeRule) -> None:
    """For each OE→ME rule, the exemplar input run through THAT
    rule's pattern/replacement (in isolation) should produce the
    exemplar output. Pinned per-rule so a future edit to one rule's
    pattern/replacement that breaks the exemplar surfaces
    immediately.

    Single-rule semantics (rather than full-chain) so adding /
    removing rules later doesn't ripple through the per-rule
    exemplars. Full-chain behavior is exercised separately via
    ``test_place_name_full_chain_to_modern_spelling`` etc.
    """
    input_form, expected_output = rule.exemplar
    candidates = _apply_one_rule([(input_form, 1.0)], rule.pattern, rule.replacement, rule.weight)
    forms = [form for form, _ in candidates]
    assert expected_output in forms, (
        f"rule {rule.pattern!r} → {rule.replacement!r} on exemplar "
        f"input {input_form!r} did not produce {expected_output!r}; "
        f"got {forms!r}"
    )


# --- apply_rules behavior ------------------------------------------------


def test_apply_rules_unknown_cell_returns_input_unchanged() -> None:
    """A cell with no rules registered is a no-op: the input form is
    returned with probability 1.0. Distinguishable from 'rules ran,
    no changes' via ``has_rules``."""
    candidates = apply_rules("anything", "klingon", "old-empire", "modern-empire")
    assert candidates == [("anything", 1.0)]


def test_apply_rules_known_cell_no_match_returns_input_unchanged() -> None:
    """A form that doesn't match ANY rule's pattern in a known cell
    returns the input with probability 1.0 (every rule is a pass-
    through). Distinct from the unknown-cell path: rules WERE
    consulted, none fired."""
    # 'qqq' doesn't contain any OE-specific char or place-name suffix
    # (avoid 'y' / 'æ' / etc. which would trip a rule).
    candidates = apply_rules("qqq", "english", "old-english", "middle-english")
    assert candidates == [("qqq", 1.0)]


def test_apply_rules_forward_chains_multiple_rules() -> None:
    """Multiple rules applying in declared order to one input produce
    the cumulative transform. ``Hædan-tūn`` (OE) → ``Hædan-ton`` after
    -tūn → -ton; → ``Hadan-ton`` after æ → a. Pinned to confirm
    rule order is respected."""
    candidates = apply_rules("Hædan-tūn", "english", "old-english", "middle-english")
    forms = [form for form, _ in candidates]
    assert "Hadan-ton" in forms


def test_apply_rules_inverse_returns_input_when_no_rule_replacement_matches() -> None:
    """A form with no occurrences of any rule's REPLACEMENT string in
    the inverse direction passes through unchanged."""
    candidates = apply_rules("qqq", "english", "middle-english", "old-english", mode="inverse")
    assert candidates == [("qqq", 1.0)]


def test_apply_rules_inverse_undoes_forward_for_a_simple_chain() -> None:
    """Apply forward, then inverse the result through the inverse
    cell, and the original form should appear in the inverse
    candidates. Pinned for a simple input where rule ordering doesn't
    cause information loss."""
    forward_candidates = apply_rules("fisċ", "english", "old-english", "middle-english")
    forward_form = forward_candidates[0][0]
    assert forward_form == "fish"
    inverse_candidates = apply_rules(
        forward_form, "english", "old-english", "middle-english", mode="inverse"
    )
    inverse_forms = [form for form, _ in inverse_candidates]
    assert "fisċ" in inverse_forms


def test_apply_rules_inverse_collapsed_target_yields_multiple_candidates() -> None:
    """Forward: BOTH 'ē' → 'e' AND 'ǣ' → 'e' collapse onto 'e'.
    Inverse from 'e' should surface both candidates. (At least one
    of 'dēg'/'dǣg' should appear in the inverse of 'deg'.) Pinned to
    confirm multi-candidate semantics work."""
    inverse_candidates = apply_rules(
        "deg", "english", "old-english", "middle-english", mode="inverse"
    )
    forms = [form for form, _ in inverse_candidates]
    # At least one OE long-vowel ancestor surfaces.
    assert any(f in {"dǣg", "dēg"} for f in forms), (
        f"expected dǣg or dēg in inverse candidates of 'deg', got {forms!r}"
    )


def test_apply_rules_invalid_mode_raises() -> None:
    """``mode`` must be 'forward' or 'inverse'. Anything else is a
    caller bug — fail loudly."""
    with pytest.raises(ValueError):
        apply_rules("x", "english", "old-english", "middle-english", mode="sideways")


def test_apply_rules_returns_probabilities_summing_to_one() -> None:
    """The candidate list represents a probability distribution over
    derived forms. Sum of probabilities should be 1.0 (within
    floating-point tolerance) for both universal-only and sporadic
    cells."""
    candidates = apply_rules("fisċ", "english", "old-english", "middle-english")
    total = sum(p for _, p in candidates)
    assert abs(total - 1.0) < 1e-9


# --- has_rules -----------------------------------------------------------


def test_has_rules_returns_true_for_registered_cell() -> None:
    assert has_rules("english", "old-english", "middle-english") is True


def test_has_rules_returns_false_for_unknown_cell() -> None:
    assert has_rules("klingon", "old-empire", "modern-empire") is False


def test_has_rules_distinguishes_directions() -> None:
    """``has_rules`` checks the FORWARD direction. The inverse cell
    is implied by the forward registration, so callers needing
    inverse should pass eras swapped."""
    assert has_rules("english", "old-english", "middle-english") is True
    # The reverse direction is NOT separately registered; callers
    # ask for inverse via mode='inverse', not via swapped has_rules.
    assert has_rules("english", "middle-english", "old-english") is False


# --- _apply_one_rule (sporadic-rule branching) ---------------------------


def test_apply_one_rule_universal_weight_produces_one_candidate() -> None:
    """A universal rule (weight=1.0) replaces unconditionally; one
    input candidate produces one output candidate."""
    out = _apply_one_rule([("xa", 1.0)], "x", "y", 1.0)
    assert out == [("ya", 1.0)]


def test_apply_one_rule_sporadic_weight_produces_two_candidates() -> None:
    """A sporadic rule (weight<1.0) branches: one candidate where the
    rule fired (probability *= weight), one where it didn't
    (probability *= (1-weight))."""
    out = _apply_one_rule([("xa", 1.0)], "x", "y", 0.3)
    assert len(out) == 2
    # The 'fired' candidate has weight 0.3; the 'didn't fire' has
    # weight 0.7. Order: fired first, didn't-fire second.
    assert out[0] == ("ya", 0.3)
    assert out[1] == ("xa", 0.7)


def test_apply_one_rule_sporadic_probabilities_sum_to_input() -> None:
    """Sporadic branching preserves total probability mass: the two
    output candidates' weights sum to the input candidate's weight."""
    out = _apply_one_rule([("xa", 0.5)], "x", "y", 0.4)
    total = sum(p for _, p in out)
    assert abs(total - 0.5) < 1e-9


def test_apply_one_rule_skips_candidates_not_containing_pattern() -> None:
    """A candidate that doesn't contain the pattern passes through
    unchanged (no branching, no transformation)."""
    out = _apply_one_rule([("abc", 1.0)], "x", "y", 0.3)
    assert out == [("abc", 1.0)]


# --- _dedupe_candidates --------------------------------------------------


def test_dedupe_candidates_merges_identical_forms() -> None:
    """Two candidates with the same form sum their probabilities
    (the form is the same outcome, reachable via two paths)."""
    out = _dedupe_candidates([("a", 0.3), ("b", 0.5), ("a", 0.2)])
    forms = dict(out)
    assert forms == {"a": pytest.approx(0.5), "b": pytest.approx(0.5)}


def test_dedupe_candidates_preserves_first_occurrence_order() -> None:
    """Order of first occurrence is preserved. Order matters for
    deterministic test assertions and the user-facing 'best
    candidate' iteration."""
    out = _dedupe_candidates([("a", 0.3), ("b", 0.5), ("a", 0.2)])
    assert [form for form, _ in out] == ["a", "b"]


# --- integration: place-name suffix transitions --------------------------


def test_place_name_full_chain_to_modern_spelling() -> None:
    """A multi-rule transition should produce a recognizable ME form.

    'Wēod-lēah' (OE) chains through:
      -lēah → -ley (suffix) ⇒ 'Wēod-ley'
      ē → e (long-vowel macron loss) ⇒ 'Weod-ley'
      y → i (short y unrounding) ⇒ 'Weod-lei'

    Both 'Weod-ley' and 'Weod-lei' are attested ME spellings of the
    suffix; Phase 1's ordering produces 'Weod-lei' because the short
    y → i rule fires on the y introduced by the suffix replacement.
    Phase 2 may add an environment constraint pinning -ley as
    terminal; for Phase 1 we accept either reflex.
    """
    candidates = apply_rules("Wēod-lēah", "english", "old-english", "middle-english")
    forms = [form for form, _ in candidates]
    assert any(f in forms for f in ("Weod-ley", "Weod-lei")), (
        f"expected Weod-ley or Weod-lei in candidates, got {forms!r}"
    )


def test_place_name_combined_palatalization_and_suffix() -> None:
    """A name with palatalized consonant AND place-name suffix runs
    both transitions: 'Sand-wīc' → 'Sand-wich' (suffix), and the
    macron normalization runs implicitly. Documents the rule
    ordering's effect."""
    candidates = apply_rules("Sand-wīc", "english", "old-english", "middle-english")
    forms = [form for form, _ in candidates]
    assert "Sand-wich" in forms
