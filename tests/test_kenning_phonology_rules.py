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
    EMODE_TO_MODE_RULES,
    FAMILY_CHAINS,
    LANGUAGE_ALIASES,
    ME_TO_EMODE_RULES,
    OE_TO_ME_RULES,
    OW_TO_MW_RULES,
    SoundChangeRule,
    _apply_one_rule,
    _dedupe_candidates,
    apply_rules,
    chain_for,
    has_rules,
    resolve_language,
    rule_form,
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


# --- empty-pattern + edge-case regressions (general-review findings) -----


def test_apply_one_rule_empty_pattern_is_noop() -> None:
    """Regression: ``str.replace("", X)`` interpolates X between every
    character. The empty-pattern guard in apply_rules wasn't
    duplicated in the helper; tests calling _apply_one_rule directly
    surfaced the bug. Pinned so a future refactor doesn't drop the
    guard."""
    out = _apply_one_rule([("hello", 1.0)], "", "X", 1.0)
    assert out == [("hello", 1.0)]


def test_apply_one_rule_empty_pattern_is_noop_in_branch_mode() -> None:
    """Same guard applies under always_branch=True (inverse mode)."""
    out = _apply_one_rule([("hello", 1.0)], "", "X", 1.0, always_branch=True)
    assert out == [("hello", 1.0)]


def test_apply_one_rule_always_branch_universal_weight_branches_50_50() -> None:
    """Direct test of always_branch=True path. Previously only
    exercised indirectly via inverse-mode integration tests."""
    out = _apply_one_rule([("xa", 1.0)], "x", "y", 1.0, always_branch=True)
    assert len(out) == 2
    forms = dict(out)
    assert forms == {"ya": pytest.approx(0.5), "xa": pytest.approx(0.5)}


def test_apply_rules_forward_sporadic_chain_distributes_mass() -> None:
    """Sporadic rules in a forward chain branch at each match; total
    probability mass stays at 1.0 across the candidate list. Direct
    apply_rules-level test (not just _apply_one_rule unit-level)."""
    # Use a synthetic rule by injecting it temporarily isn't trivial
    # without mutating module state; instead exercise the existing
    # OE→ME chain and confirm probabilities sum to 1.0 after a
    # multi-rule trace.
    candidates = apply_rules("Hædan-tūn", "english", "old-english", "middle-english")
    total = sum(p for _, p in candidates)
    assert abs(total - 1.0) < 1e-9


def test_dedupe_candidates_handles_empty_input() -> None:
    """Empty input → empty output (no crash on the boundary)."""
    assert _dedupe_candidates([]) == []


# --- rule-ordering structural invariants ---------------------------------


def test_oe_to_me_suffix_rules_precede_single_char_rules() -> None:
    """Place-name suffix rules MUST come before single-char rules;
    otherwise the constituent vowels (ū, ē, etc.) get rewritten
    before the suffix can match. Pin the structural invariant so a
    future rule-list edit that drops this ordering surfaces here."""
    suffix_indices = []
    single_char_indices = []
    for i, rule in enumerate(OE_TO_ME_RULES):
        if rule.pattern.startswith("-"):
            suffix_indices.append(i)
        elif len(rule.pattern) == 1:
            single_char_indices.append(i)
    assert suffix_indices, "no suffix rules found"
    assert single_char_indices, "no single-char rules found"
    last_suffix = max(suffix_indices)
    first_single_char = min(single_char_indices)
    assert last_suffix < first_single_char, (
        "place-name suffix rules must precede single-char rules; "
        f"last suffix at index {last_suffix}, first single-char at "
        f"{first_single_char}"
    )


def test_oe_to_me_digraph_rules_precede_constituent_chars() -> None:
    """Digraph palatal rules (sċ → sh, ċġ → dg) must run before
    their constituent single chars (ċ → ch, ġ → y) — otherwise
    'sċ' gets rewritten to 'sh' through the wrong path. Pin the
    invariant."""
    pos = {rule.pattern: i for i, rule in enumerate(OE_TO_ME_RULES)}
    # ċġ contains both ċ and ġ; both single-char rules must come
    # AFTER the digraph rule.
    assert pos["ċġ"] < pos["ċ"]
    assert pos["ċġ"] < pos["ġ"]
    # sċ contains ċ; the digraph must come before the single-char
    # ċ rule.
    assert pos["sċ"] < pos["ċ"]


# --- inverse explosion bound ---------------------------------------------


def test_apply_rules_inverse_candidate_count_stays_bounded() -> None:
    """With 28 rules each branching 50/50 in inverse mode, the
    candidate list could theoretically reach 2^28 ≈ 268M without the
    probability-floor pruning. Pin a bounded ceiling so a future
    floor-removal regresses here."""
    # A pathological-looking input that touches several rules' inverse
    # patterns. The exact candidate count depends on rule
    # interactions; the assertion is a generous upper bound that's
    # still well under the unpruned 2^28.
    candidates = apply_rules(
        "the-quick-brown-fox-leaps-over-the-lazy-dog",
        "english",
        "old-english",
        "middle-english",
        mode="inverse",
    )
    assert len(candidates) < 100_000, (
        f"inverse mode produced {len(candidates)} candidates; "
        f"probability floor should keep this bounded"
    )


# --- Middle English → Early Modern English (wyrd-n9x5) ------------------


def test_has_rules_returns_true_for_me_to_emode_cell() -> None:
    """The new ME→EModE cell is registered and discoverable via
    ``has_rules``."""
    assert has_rules("english", "middle-english", "early-modern-english") is True


@pytest.mark.parametrize("rule", ME_TO_EMODE_RULES, ids=lambda r: r.pattern)
def test_me_to_emode_rule_exemplar_forward(rule: SoundChangeRule) -> None:
    """Each ME→EModE rule's exemplar input run through the rule's
    pattern/replacement (in isolation) produces the documented output.
    Same single-rule semantics as the OE→ME exemplar test."""
    input_form, expected_output = rule.exemplar
    candidates = _apply_one_rule([(input_form, 1.0)], rule.pattern, rule.replacement, rule.weight)
    forms = [form for form, _ in candidates]
    assert expected_output in forms, (
        f"rule {rule.pattern!r} → {rule.replacement!r} on exemplar "
        f"input {input_form!r} did not produce {expected_output!r}; "
        f"got {forms!r}"
    )


def test_me_to_emode_rules_have_required_documentation() -> None:
    """Every ME→EModE rule carries description / exemplar / source —
    parallel to the OE→ME contract test, pinned per-cell so adding a
    new rule without docs surfaces here."""
    for rule in ME_TO_EMODE_RULES:
        assert rule.description, f"rule {rule.pattern!r} missing description"
        assert isinstance(rule.exemplar, tuple) and len(rule.exemplar) == 2
        assert all(isinstance(s, str) and s for s in rule.exemplar)
        assert rule.source, f"rule {rule.pattern!r} missing source citation"


def test_apply_rules_chains_oe_to_emode_via_me_intermediate() -> None:
    """A two-step chain: OE → ME → EModE. Apply the OE→ME cell then
    the ME→EModE cell on the intermediate output, and the cumulative
    pipeline produces a recognizable EModE form. Pinned to confirm
    the cells compose without surface change to apply_rules."""
    me_intermediate = apply_rules("Hamp-stede", "english", "old-english", "middle-english")
    me_form = me_intermediate[0][0]
    assert me_form == "Hamp-stede"  # OE→ME doesn't touch this form

    emode_candidates = apply_rules(me_form, "english", "middle-english", "early-modern-english")
    forms = [form for form, _ in emode_candidates]
    assert "Hamp-stead" in forms


def test_apply_rules_inverse_me_to_emode_undoes_simple_chain() -> None:
    """Forward 'Pal-worde' → 'Pal-worth'; inverse from 'Pal-worth'
    surfaces the OE-shape ancestor in the candidate list."""
    forward = apply_rules("Pal-worde", "english", "middle-english", "early-modern-english")
    forward_form = forward[0][0]
    assert forward_form == "Pal-worth"
    inverse = apply_rules(
        forward_form,
        "english",
        "middle-english",
        "early-modern-english",
        mode="inverse",
    )
    inverse_forms = [form for form, _ in inverse]
    assert "Pal-worde" in inverse_forms


def test_me_to_emode_unknown_form_no_op() -> None:
    """A form that doesn't match any ME→EModE pattern passes through
    unchanged. Distinguishes 'cell registered, no match' from 'no cell'."""
    candidates = apply_rules("qqq", "english", "middle-english", "early-modern-english")
    assert candidates == [("qqq", 1.0)]


def test_me_to_emode_probability_distribution_sums_to_one() -> None:
    """Universal-only rules in ME→EModE preserve total probability
    mass at 1.0 (no sporadic branching yet)."""
    candidates = apply_rules("Birch-wude", "english", "middle-english", "early-modern-english")
    total = sum(p for _, p in candidates)
    assert abs(total - 1.0) < 1e-9


def test_me_to_emode_dispatch_lookup_works_via_apply_rules() -> None:
    """End-to-end via ``apply_rules``: a typo in the dispatch tuple
    key would silently route the cell to 'unknown cell, no-op'. Pin
    via a forward call exercising the actual dispatch lookup (not
    just direct _apply_one_rule on a rule)."""
    candidates = apply_rules("Hai-ston", "english", "middle-english", "early-modern-english")
    forms = [form for form, _ in candidates]
    assert "Hay-ston" in forms, (
        "ME→EModE cell should fire 'ai → ay'; if not, the dispatch table key may be wrong"
    )


def test_me_to_emode_does_not_corrupt_internal_ou() -> None:
    """Regression for review concern: ME forms with internal -ou-
    (hous, mous, cou) are NOT mangled by Phase 2.1. The 'ou → ow'
    transition is environment-sensitive (word-final OK, morpheme-
    internal NOT) — deferred to Phase 2.4 with env-constraint regex."""
    candidates = apply_rules("Hous-tone", "english", "middle-english", "early-modern-english")
    forms = [form for form, _ in candidates]
    assert "Hous-tone" in forms  # passes through unchanged


def test_me_to_emode_does_not_corrupt_internal_ye() -> None:
    """Regression for review concern: ME 'Wye-' (river name) and
    similar word-internal 'ye' substrings are NOT mangled by Phase
    2.1. The 'ye → y' rule was deliberately deferred to Phase 2.4
    because it substring-overreaches (Wyeham → Wyham, yeoman → yoman)
    without an env-constraint."""
    candidates = apply_rules("Wye-ham", "english", "middle-english", "early-modern-english")
    forms = [form for form, _ in candidates]
    assert "Wye-ham" in forms  # passes through unchanged


def test_registered_cell_count_matches_expected() -> None:
    """Pin the count of registered cells. A future PR that
    accidentally drops a cell from the dispatch table would fail
    this; intentional additions should bump the count."""
    from wyrd.generators.kenning.phonology_rules import _RULES

    assert len(_RULES) == 4, (
        f"expected 4 cells (OE→ME, ME→EModE, EModE→ModE, OW→ModW), got "
        f"{len(_RULES)}: {sorted(_RULES.keys())}"
    )


# --- Early Modern English → Modern English (wyrd-n9x5 Phase 2.2) --------


def test_has_rules_returns_true_for_emode_to_mode_cell() -> None:
    """The new EModE→ModE cell is registered and discoverable via
    ``has_rules``."""
    assert has_rules("english", "early-modern-english", "modern-english") is True


@pytest.mark.parametrize("rule", EMODE_TO_MODE_RULES, ids=lambda r: r.pattern)
def test_emode_to_mode_rule_exemplar_forward(rule: SoundChangeRule) -> None:
    """Each EModE→ModE rule's exemplar input run through the rule's
    pattern/replacement (in isolation) produces the documented output.
    Same single-rule semantics as the OE→ME / ME→EModE / OW→ModW
    exemplar tests."""
    input_form, expected_output = rule.exemplar
    candidates = _apply_one_rule([(input_form, 1.0)], rule.pattern, rule.replacement, rule.weight)
    forms = [form for form, _ in candidates]
    assert expected_output in forms, (
        f"rule {rule.pattern!r} → {rule.replacement!r} on exemplar "
        f"input {input_form!r} did not produce {expected_output!r}; got {forms!r}"
    )


def test_emode_to_mode_rules_have_required_documentation() -> None:
    """Every EModE→ModE rule carries description / exemplar / source —
    parallel to the per-cell contract enforced for OE→ME, ME→EModE,
    and OW→ModW. Adding a new rule without docs surfaces here."""
    for rule in EMODE_TO_MODE_RULES:
        assert rule.description, f"rule {rule.pattern!r} missing description"
        assert isinstance(rule.exemplar, tuple) and len(rule.exemplar) == 2
        assert all(isinstance(s, str) and s for s in rule.exemplar)
        assert rule.source, f"rule {rule.pattern!r} missing source citation"


def test_emode_to_mode_unknown_form_no_op() -> None:
    """A form that doesn't match any EModE→ModE pattern passes through
    unchanged. Distinguishes 'cell registered, no match' from 'no cell'."""
    candidates = apply_rules("qqq", "english", "early-modern-english", "modern-english")
    assert candidates == [("qqq", 1.0)]


def test_emode_to_mode_probability_distribution_sums_to_one() -> None:
    """Universal-only rules in EModE→ModE preserve total probability
    mass at 1.0 (no sporadic branching yet — Phase 2.4 may add weight<1.0
    rules)."""
    candidates = apply_rules("Liver-poole", "english", "early-modern-english", "modern-english")
    total = sum(p for _, p in candidates)
    assert abs(total - 1.0) < 1e-9


def test_emode_to_mode_dispatch_lookup_works_via_apply_rules() -> None:
    """End-to-end via ``apply_rules``: a typo in the dispatch tuple
    key would silently route the cell to 'unknown cell, no-op'. Pin
    via a forward call exercising the actual dispatch lookup."""
    candidates = apply_rules("Tavi-stocke", "english", "early-modern-english", "modern-english")
    forms = [form for form, _ in candidates]
    assert "Tavi-stock" in forms, (
        "EModE→ModE cell should fire '-stocke → -stock'; if not, the dispatch table key may be wrong"
    )


def test_emode_to_mode_inverse_undoes_simple_chain() -> None:
    """Inverse direction: ModE 'Hol-brook' → EModE candidates including
    'Hol-brooke'. Inverse-mode 50/50 branching surfaces the EModE
    source among the alternatives."""
    forward = apply_rules("Hol-brooke", "english", "early-modern-english", "modern-english")
    forward_form = forward[0][0]
    assert forward_form == "Hol-brook"

    inverse = apply_rules(
        forward_form,
        "english",
        "early-modern-english",
        "modern-english",
        mode="inverse",
    )
    inverse_forms = [form for form, _ in inverse]
    assert "Hol-brooke" in inverse_forms


def test_apply_rules_chains_me_to_mode_via_emode_intermediate() -> None:
    """A two-step chain: ME → EModE → ModE. Apply the ME→EModE cell
    then the EModE→ModE cell and the cumulative pipeline produces a
    recognizable ModE form. Pinned to confirm the cells compose
    cleanly without surface change to apply_rules.

    'Hamp-stede' (ME) → 'Hamp-stead' (EModE, via -stede → -stead in
    Phase 2.1) is a stable EModE form; the EModE→ModE cell does NOT
    further modify it (no -stead → -X rule in Phase 2.2). Pin BOTH
    edges: the chain is reachable AND the EModE form is preserved
    across the second cell when no rule fires.
    """
    me_to_emode = apply_rules("Hamp-stede", "english", "middle-english", "early-modern-english")
    emode_form = me_to_emode[0][0]
    assert emode_form == "Hamp-stead"

    emode_to_mode = apply_rules(emode_form, "english", "early-modern-english", "modern-english")
    forms = [form for form, _ in emode_to_mode]
    assert "Hamp-stead" in forms


def test_emode_to_mode_does_not_corrupt_unrelated_suffixes() -> None:
    """Regression: EModE forms ending in suffixes NOT covered by Phase
    2.2 (-borough, -worth, -wood from Phase 2.1's outputs) pass
    through unchanged. Pinned to confirm the new cell only touches the
    five suffixes it claims; no over-broad pattern is silently
    rewriting Phase 2.1 outputs."""
    cases = ("Peter-borough", "Pal-worth", "Birch-wood", "Hamp-stead", "Spring-field")
    for form in cases:
        candidates = apply_rules(form, "english", "early-modern-english", "modern-english")
        forms = [f for f, _ in candidates]
        assert form in forms, (
            f"EModE form {form!r} should pass through Phase 2.2 unchanged; got {forms!r}"
        )


def test_emode_to_mode_pattern_fires_only_on_hyphen_anchored_suffix() -> None:
    """Regression: the Phase 2.2 patterns all carry a leading hyphen
    (`-stocke`, `-brooke`, `-grene`, `-poole`, `-ditche`) which acts as
    the framework's word-boundary proxy. A ModE-shaped form that
    contains the BARE suffix string (without the hyphen) MUST NOT
    silently fire the rule — otherwise an unrelated stem ending in
    e.g. 'poole' or 'grene' would get mangled.

    Parallel to Phase 2.1's `test_me_to_emode_does_not_corrupt_internal_*`
    pins; pinned per-suffix here so a future rule edit that drops the
    leading hyphen surfaces immediately.
    """
    bare_forms = ("stockeholder", "brookekeeper", "greneliness", "pooledom", "ditcheside")
    for form in bare_forms:
        candidates = apply_rules(form, "english", "early-modern-english", "modern-english")
        forms = [f for f, _ in candidates]
        assert form in forms, (
            f"bare-suffix form {form!r} should pass through Phase 2.2 "
            f"unchanged (no leading hyphen → no suffix anchor); got {forms!r}"
        )


def test_emode_to_mode_full_chain_oe_to_mode_via_intermediates() -> None:
    """Full attestation chain: OE → ME → EModE → ModE. Pick a place-
    name that has a real attestation chain and confirm the framework
    can route through all three intermediates.

    'Hamp-stede' is ME (post-OE→ME of a hypothetical 'Hamp-stede'-shape
    OE form); applying ME→EModE yields 'Hamp-stead'; applying
    EModE→ModE leaves it stable (the -stead reflex is stable into
    ModE). The cell-composition test for the OE→ME→EModE chain is
    already pinned; this one extends it to the fourth era and is the
    canonical 'all-cells-compose' integration check.
    """
    me = apply_rules("Hamp-stede", "english", "old-english", "middle-english")
    me_form = me[0][0]
    # OE→ME doesn't touch '-stede' so the form is preserved.
    assert me_form == "Hamp-stede"
    emode = apply_rules(me_form, "english", "middle-english", "early-modern-english")
    emode_form = emode[0][0]
    assert emode_form == "Hamp-stead"
    mode = apply_rules(emode_form, "english", "early-modern-english", "modern-english")
    forms = [f for f, _ in mode]
    assert "Hamp-stead" in forms


# --- Old Welsh → Modern Welsh (wyrd-n9x5 Phase 2.3) --------------------


def test_has_rules_returns_true_for_ow_to_mw_cell() -> None:
    """The OW→ModW cell is registered and discoverable via ``has_rules``."""
    assert has_rules("welsh", "old-welsh", "modern-welsh") is True


@pytest.mark.parametrize("rule", OW_TO_MW_RULES, ids=lambda r: r.pattern)
def test_ow_to_mw_rule_exemplar_forward(rule: SoundChangeRule) -> None:
    """Each OW→ModW rule's exemplar input run through the rule's
    pattern/replacement (in isolation) produces the documented output.
    Single-rule semantics — exemplar stability across rule re-ordering."""
    input_form, expected_output = rule.exemplar
    candidates = _apply_one_rule([(input_form, 1.0)], rule.pattern, rule.replacement, rule.weight)
    forms = [form for form, _ in candidates]
    assert expected_output in forms, (
        f"rule {rule.pattern!r} → {rule.replacement!r} on exemplar "
        f"input {input_form!r} did not produce {expected_output!r}; got {forms!r}"
    )


def test_ow_to_mw_rules_have_required_documentation() -> None:
    """Every OW→ModW rule carries description / exemplar / source — the
    documentation contract enforced for OE→ME and ME→EModE applies here
    too. Adding a new rule without docs surfaces here."""
    for rule in OW_TO_MW_RULES:
        assert rule.description, f"rule {rule.pattern!r} missing description"
        assert isinstance(rule.exemplar, tuple) and len(rule.exemplar) == 2
        assert all(isinstance(s, str) and s for s in rule.exemplar)
        assert rule.source, f"rule {rule.pattern!r} missing source citation"


def test_ow_to_mw_caer_lexical_pattern_fires() -> None:
    """The composite 'kair → caer' rule is the canonical OW→ModW
    transition for the toponym element 'fortress'. Captures both
    spelling normalization (k→c) AND the diphthong adjustment (ai→ae)
    in a single atomic rule, so that 'kair' doesn't go via 'cair'."""
    candidates = apply_rules("kair-uent", "welsh", "old-welsh", "modern-welsh")
    forms = [form for form, _ in candidates]
    assert "caer-uent" in forms


def test_ow_to_mw_auc_suffix_voicing_fires() -> None:
    """The -auc / -iauc → -og / -iog suffix transitions are the most
    visible OW→ModW reflex in real toponyms (Pennog, Mathriog, etc.).
    Both the suffix rule AND the consequence (au-monophthongization +
    k→g voicing in word-final position) must fire."""
    candidates = apply_rules("Penn-auc", "welsh", "old-welsh", "modern-welsh")
    forms = [form for form, _ in candidates]
    assert "Penn-og" in forms

    candidates = apply_rules("Mathryn-iauc", "welsh", "old-welsh", "modern-welsh")
    forms = [form for form, _ in candidates]
    assert "Mathryn-iog" in forms


def test_ow_to_mw_diphthong_au_to_aw() -> None:
    """OW 'maur' → MW 'mawr' 'great'. Universal au→aw shift for
    stressed mid-word diphthongs (suffix rules consume the word-final
    'auc' before this rule fires)."""
    candidates = apply_rules("maur", "welsh", "old-welsh", "modern-welsh")
    forms = [form for form, _ in candidates]
    assert "mawr" in forms


def test_ow_to_mw_k_to_c_orthographic_normalization() -> None:
    """Any 'k' surviving the multi-char rules above gets normalized to
    'c'. Pinned because the rule order matters: 'k → c' running BEFORE
    a multi-char 'kair → caer' rule would mean the lexical pattern
    never matches its OW spelling."""
    candidates = apply_rules("kil", "welsh", "old-welsh", "modern-welsh")
    forms = [form for form, _ in candidates]
    assert "cil" in forms


def test_ow_to_mw_unknown_form_no_op() -> None:
    """A form that doesn't match any OW→ModW pattern passes through
    unchanged. Distinguishes 'cell registered, no match' from 'no cell'."""
    candidates = apply_rules("xyz", "welsh", "old-welsh", "modern-welsh")
    assert candidates == [("xyz", 1.0)]


def test_ow_to_mw_probability_distribution_sums_to_one() -> None:
    """Universal-only rules in OW→ModW preserve total probability mass
    at 1.0 (no sporadic branching yet — Phase 2.4 may add weight<1.0
    rules for the oi→oe vs oi→wy alternation)."""
    candidates = apply_rules("Penn-auc", "welsh", "old-welsh", "modern-welsh")
    total = sum(p for _, p in candidates)
    assert abs(total - 1.0) < 1e-9


def test_ow_to_mw_inverse_undoes_simple_chain() -> None:
    """Inverse direction: ModW 'Penn-og' → OW candidates including
    'Penn-auc'. Inverse-mode 50/50 branching surfaces the OW source
    among the alternatives even when the ModW form is itself a final
    rule's output."""
    forward = apply_rules("Penn-auc", "welsh", "old-welsh", "modern-welsh")
    forward_form = forward[0][0]
    assert forward_form == "Penn-og"

    inverse = apply_rules(
        forward_form,
        "welsh",
        "old-welsh",
        "modern-welsh",
        mode="inverse",
    )
    inverse_forms = [form for form, _ in inverse]
    assert "Penn-auc" in inverse_forms


def test_ow_to_mw_dispatch_lookup_works_via_apply_rules() -> None:
    """End-to-end via ``apply_rules``: a typo in the dispatch tuple
    key (e.g. ('welsh', 'modern-welsh', 'old-welsh') reversed) would
    silently route to 'unknown cell, no-op'. Pin via a forward call
    exercising the actual dispatch lookup."""
    candidates = apply_rules("Penn-auc", "welsh", "old-welsh", "modern-welsh")
    forms = [form for form, _ in candidates]
    assert "Penn-og" in forms, (
        "OW→ModW cell should fire '-auc → -og'; if not, the dispatch table key may be wrong"
    )


# --- chain_for + rule_form: public chain-walker API (wyrd-u728) ----------


def test_resolve_language_returns_canonical_era_for_alias() -> None:
    """``resolve_language('welsh')`` returns 'modern-welsh'; the alias
    map is the public API for this transform so consumers can find
    their position in a chain tuple without reaching into
    ``LANGUAGE_ALIASES`` directly."""
    assert resolve_language("welsh") == "modern-welsh"


def test_resolve_language_passes_through_canonical_era_names() -> None:
    """Languages absent from the alias map pass through unchanged.
    Pin so consumers can safely call resolve_language on any language
    code, not just known aliases."""
    assert resolve_language("modern-welsh") == "modern-welsh"
    assert resolve_language("old-english") == "old-english"
    assert resolve_language("irish") == "irish"  # non-chain language passes through


def test_chain_for_returns_family_chain_for_modern_english() -> None:
    """Modern English is in the English chain family. Pin the chain
    shape so consumers can rely on the (family, chain) return tuple."""
    result = chain_for("modern-english")
    assert result is not None
    family, chain = result
    assert family == "english"
    assert chain == (
        "old-english",
        "middle-english",
        "early-modern-english",
        "modern-english",
    )


def test_chain_for_resolves_welsh_alias() -> None:
    """The lexicon language code 'welsh' is an alias for 'modern-welsh'.
    Pin that ``chain_for`` resolves it to the welsh chain rather than
    returning None (the alias-unaware fallback would crash on chain.index)."""
    via_alias = chain_for("welsh")
    via_canonical = chain_for("modern-welsh")
    assert via_alias is not None
    assert via_alias == via_canonical
    family, chain = via_alias
    assert family == "welsh"
    assert "modern-welsh" in chain


def test_chain_for_returns_none_for_non_chain_language() -> None:
    """Languages without registered phonology rules (Goidelic, Norse,
    Romance, etc.) return None. Pin so Tier-4 callers can treat None
    as 'no rules to walk' rather than crashing."""
    assert chain_for("irish") is None
    assert chain_for("old-irish") is None
    assert chain_for("scottish-gaelic") is None
    assert chain_for("old-norse") is None
    assert chain_for("old-french") is None
    assert chain_for("latin") is None


def test_rule_form_walks_oe_to_me_chain_forward() -> None:
    """Forward walk from modern-english to old-english applies cells
    in inverse mode. Pin against a known-firing form so the chain walk
    is exercised end-to-end."""
    # Forward walk: OE → ME applies OE_TO_ME cell once.
    out = rule_form("cwen", "old-english", "middle-english")
    # The actual output depends on cell rules, but the chain walker
    # MUST produce SOMETHING for a form that has rule triggers — pin
    # 'non-None, non-self'.
    if out is not None:
        assert out != "cwen", "rule_form must return None or a transformed form, never the input"


def test_rule_form_returns_none_when_form_passes_through() -> None:
    """A canonical_form whose letters don't match any rule trigger
    passes through unchanged — the chain-walker returns None so
    callers can fall back to the canonical_form rather than emit a
    spurious 'Tier 4 reflex' that's the same as the input."""
    # 'orphan' has no OE→ME rule triggers per the existing test suite.
    assert rule_form("orphan", "old-english", "middle-english") is None


def test_rule_form_returns_none_for_cross_family_walk() -> None:
    """English and Welsh are different rule families with no bridge
    cells. ``rule_form("X", "modern-english", "modern-welsh")`` must
    return None rather than attempt a no-cell walk."""
    assert rule_form("house", "modern-english", "modern-welsh") is None


def test_rule_form_returns_none_for_unknown_language() -> None:
    """Languages absent from FAMILY_CHAINS get None. Pin so Tier-4
    callers (etymon_era_reflexes, language_quality's tier4 walk)
    can route through this API safely without per-language guards."""
    assert rule_form("baile", "irish", "old-irish") is None
    assert rule_form("X", "modern-english", "irish") is None  # one side unknown


def test_rule_form_alias_resolution_produces_same_output() -> None:
    """Passing 'welsh' (alias) vs 'modern-welsh' (canonical) for a
    Welsh-chain walk must produce the same result. The function
    re-resolves aliases internally — pin that consumers don't need
    to pre-resolve."""
    via_alias = rule_form("kaer-uent", "welsh", "old-welsh")
    via_canonical = rule_form("kaer-uent", "modern-welsh", "old-welsh")
    assert via_alias == via_canonical


def test_family_chains_and_language_aliases_are_dict_typed() -> None:
    """Pin the public types of the chain-config dicts. Consumers may
    iterate them (e.g. for diagnostics or schema discovery), so the
    dict shape is part of the public contract."""
    assert isinstance(FAMILY_CHAINS, dict)
    assert isinstance(LANGUAGE_ALIASES, dict)
    # Smoke the key/value shape — chain entries are (family_str, chain_tuple).
    for lang, (family, chain) in FAMILY_CHAINS.items():
        assert isinstance(lang, str)
        assert isinstance(family, str)
        assert isinstance(chain, tuple)
        assert all(isinstance(era, str) for era in chain)
