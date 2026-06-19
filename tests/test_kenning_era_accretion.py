"""D46 contract gate: era pools ACCRETE — "in the record by then".

Supersedes the D44-era ``test_kenning_era_renders_not_gates.py``. The
refined contract (DECISIONS.md D46, wyrd-6ah2):

* A bounded historical era excludes ONLY morphemes whose earliest
  evidence (``Meaning.record_start`` — the most generous of its
  source-language layers' record-entry years and its earliest attested
  year) postdates the era's end. The Silicon rule: a modern coinage
  can't appear on a 1086 map.
* Pools accrete monotonically (pool(oe-early) ⊆ pool(me) ⊆
  pool(present)); a morpheme never expires — the wyrd-c6o1.3 ``-ton``
  starvation stays impossible by construction.
* Open-ended eras gate nothing: ``era=""`` (Mixed) and the present-day
  stage (the deployed default) draw from the full accreted pool, so
  they stay skeleton-identical per seed.
* Era's render effect (which reflex) is unchanged from D44.

Live-bundle tests run against whatever runtime DB the suite resolves
(the committed ``seed-runtime.db`` in CI); all seeded and
deterministic.
"""

from __future__ import annotations

import pytest

from wyrd.generators.kenning import _load_culture
from wyrd.generators.kenning.generators.kenning import Kenning
from wyrd.generators.kenning.runtime.meaning import Meaning
from wyrd.generators.kenning.runtime.proportions import _resolve_surface
from wyrd.generators.kenning.runtime.vector_name_select import build_non_position_eligible
from wyrd.generators.kenning.vectors.schemas import EligibilityGate
from wyrd.registry import GenerationResult

# Live-bundle smoke tests (runtime DB via the committed seed) — the
# authoring-lexicon isolation fixtures are irrelevant here; opt out like
# the other live-bundle files.
pytestmark = pytest.mark.no_lexicon_isolation


# ---- Meaning.record_start (the evidence rule) ------------------------------


def _m(sources, attested=None):
    return Meaning("-x", [], ["x"], sources, attested_years=attested or {})


def test_record_start_founding_stratum_passes_everything():
    """Any founding-stratum layer (OE / Celtic / Latin / unmapped) vouches
    the morpheme into every era — even when its dates are late."""
    assert _m({"old_english": ["tun"]}).record_start() is None
    assert _m({"celtic_mix": ["pen"]}).record_start() is None
    assert (
        _m(
            {"old_english": ["tun"], "modern_english": ["ton"]},
            attested={"old_english": [("tun", 1086)]},
        ).record_start()
        is None
    )
    # Unmapped language → no evidence of lateness → founding posture.
    assert _m({"klingon": ["qo"]}).record_start() is None


def test_record_start_contact_layers_carry_their_entry_year():
    assert _m({"old_scandinavian": ["thorpe"]}).record_start() == 800
    assert _m({"old_french": ["ville"]}).record_start() == 1066
    assert _m({"middle_english": ["mille"]}).record_start() == 1100
    assert _m({"modern_english": ["silicon"]}).record_start() == 1500


def test_record_start_attestation_vouches_older_than_stage():
    """A date can prove a morpheme OLDER than its stage suggests: a
    modern_english bucket with a 1086 Domesday date is in the record by
    oe-late."""
    m = _m({"modern_english": ["x"]}, attested={"modern_english": [("x", 1086)]})
    assert m.record_start() == 1086


def test_record_start_no_evidence_passes():
    assert _m({}).record_start() is None
    # Synthetic Meanings with a non-dict sources shape contribute no
    # evidence either (same posture as _meaning_langs).
    assert Meaning("-x", [], ["x"], []).record_start() is None


def test_record_start_empty_form_buckets_do_not_vouch():
    """A founding-stratum bucket with no usable forms ({'old_english': []}
    or all-empty entries) must NOT vouch a modern coinage into every era —
    same non-empty-form semantics as primary_language."""
    assert _m({"old_english": [], "modern_english": ["silicon"]}).record_start() == 1500
    assert _m({"old_english": ["", None], "modern_english": ["silicon"]}).record_start() == 1500


def test_record_start_dates_only_morpheme_is_gated_by_its_dates():
    """No source buckets but a dated attestation: the dates ARE the
    evidence (positively-evidenced age), so they gate."""
    assert _m({}, attested={"modern_english": [("x", 1817)]}).record_start() == 1817


def test_record_start_is_cached():
    m = _m({"modern_english": ["silicon"]})
    assert m.record_start() == 1500
    m.sources = {"old_english": ["si"]}  # mutate AFTER first compute
    assert m.record_start() == 1500  # cached — bundle data is immutable


# ---- the gate predicate ----------------------------------------------------


def test_record_gate_excludes_positively_late_morphemes_only():
    from wyrd.generators.kenning.eligibility import passes_record_gate

    silicon = _m({"modern_english": ["silicon"]})
    tun = _m({"old_english": ["tun"]}, attested={"old_english": [("tun", 1086)]})
    undated = _m({})
    # Bounded medieval era (me ends 1500): the modern coinage is out...
    assert passes_record_gate(silicon, 1500) is False
    # ...but everything as old or older — or unevidenced — is in.
    assert passes_record_gate(tun, 1500) is True
    assert passes_record_gate(undated, 1500) is True
    # Open cutoff (no era / present-day stage) gates nothing.
    assert passes_record_gate(silicon, None) is True
    # Half-open boundary: record_start == cutoff is NOT "by then" (the 1500
    # case above); one year past the entry (cutoff 1501) is.
    assert passes_record_gate(silicon, 1501) is True


def test_native_pool_applies_the_record_gate():
    silicon = _m({"modern_english": ["silicon"]})
    tun = _m({"old_english": ["tun"]})
    db = {"-x": [silicon, tun]}

    def pool(cutoff):
        return build_non_position_eligible(
            db,
            gate=EligibilityGate(culture="english", era_record_cutoff=cutoff),
            exclude_tags=frozenset(),
            pack_meaning_dbs=None,
            packs=(),
        )

    assert pool(None) == [silicon, tun]
    assert pool(1500) == [tun]


# ---- live-bundle properties -------------------------------------------------


def _english_pool(name_gen, cutoff):
    return build_non_position_eligible(
        name_gen.meaning_db,
        gate=EligibilityGate(culture="english", era_record_cutoff=cutoff),
        exclude_tags=frozenset(),
        pack_meaning_dbs=None,
        packs=(),
        culture_attested_usages=name_gen.culture_attested_usages,
        culture_attested_meanings=name_gen.culture_attested_meanings,
        include_unglossed=False,
    )


def test_pools_accrete_monotonically():
    """pool(oe-early) ⊆ pool(me) ⊆ pool(open) — morphemes only ever JOIN
    the record. A non-monotone pool would mean something expires, which
    is the bug class D44 fixed (the -ton starvation)."""
    name_gen, _ = _load_culture("english")
    open_pool = {id(m) for m in _english_pool(name_gen, None)}
    me_pool = {id(m) for m in _english_pool(name_gen, 1500)}
    oe_pool = {id(m) for m in _english_pool(name_gen, 800)}
    assert oe_pool <= me_pool <= open_pool
    # The gate must actually bite on the live bundle (modern-only
    # morphemes exist), else this whole file is vacuous.
    assert len(me_pool) < len(open_pool)


def test_bounded_era_picks_are_in_the_record_by_then():
    """The Silicon rule, end to end: every morpheme PICKED at era=me has
    at least one sibling reading whose record_start predates the era's
    end (1500). Sibling resolution is fold-aware (D40/D45: identity is
    the bare surface; the rendered usage string may be any stored
    dash-variant of the picked morpheme)."""
    gen = Kenning()
    name_gen, _ = _load_culture("english")
    db = name_gen.meaning_db
    checked = 0
    for seed in range(60):
        result = gen.generate({"culture": "english", "era": "me"}, seed)
        for word in result.morphemes_by_word:
            for morpheme in word:
                siblings = _resolve_surface(db, morpheme.get("usage"))
                starts = [s.record_start() for s in siblings]
                checked += 1
                assert not starts or any(s is None or s < 1500 for s in starts), (
                    f"seed {seed}: {morpheme.get('usage')!r} picked at era=me "
                    f"but every sibling's record_start is >= 1500: {starts} — "
                    "the D46 accretion gate has stopped gating"
                )
    assert checked > 50  # the sweep actually exercised picks


# ---- the ungated pair stays skeleton-identical -------------------------------


def _roll(gen: Kenning, era: str, seed: int) -> GenerationResult:
    params: dict = {"culture": "english"}
    if era:
        params["era"] = era
    return gen.generate(params, seed)


def test_mixed_and_present_day_stay_skeleton_identical():
    """era='' (Mixed) and the present-day stage both have an open record
    cutoff (no gate) — they MUST keep producing the same skeleton per
    seed (same picked morphemes, same modern_name()); only the render
    differs. This is the surviving seed-level half of the D44 invariant
    and the regression guard for the wyrd-c6o1.3 starvation: if the
    present-day stage ever gates again, this diverges.

    wyrd-yzul: 30 seeds (down from 60). This is a per-seed DETERMINISTIC
    equality, not a statistical rate — a re-gated present-day stage
    diverges on essentially every seed, so 30 catches it just as surely
    while halving the cost."""
    gen = Kenning()
    for seed in range(30):
        mixed = _roll(gen, "", seed)
        modern = _roll(gen, "present-day", seed)
        assert mixed.result_modern == modern.result_modern, (
            f"seed {seed}: {mixed.result_modern!r} != {modern.result_modern!r}"
        )
        usages = [
            [[m.get("usage") for m in word] for word in r.morphemes_by_word]
            for r in (mixed, modern)
        ]
        assert usages[0] == usages[1], f"seed {seed}: morpheme stacks differ: {usages}"


def test_bounded_era_render_actually_changes_the_surface():
    """The render half (D44, unchanged by D46): a bounded historical era
    must actually render period forms — its native result should differ
    from its own modern companion on a healthy share of names."""
    gen = Kenning()
    diverged = sum(
        1 for seed in range(60) if (r := _roll(gen, "oe-late", seed)).result != r.result_modern
    )
    assert diverged >= 15, (
        f"oe-late rendered identically to modern on {60 - diverged}/60 seeds — "
        "the era reflex rendering looks broken"
    )


def test_mixed_and_present_day_stay_identical_with_tags_combined():
    """The ungated-pair invariance must hold per-PARAMS, not just for the
    bare request — an era leak gated behind the --tag hard gate would slip
    the headline pair test (successor of the D44-era tags-combined sweep)."""
    gen = Kenning()
    for seed in range(30):
        results = []
        for era in ("", "present-day"):
            params: dict = {"culture": "english", "tags": ["water"]}
            if era:
                params["era"] = era
            results.append(gen.generate(params, seed))
        assert results[0].result_modern == results[1].result_modern, (
            f"seed {seed} (tags=water): {results[0].result_modern!r} != "
            f"{results[1].result_modern!r}"
        )
