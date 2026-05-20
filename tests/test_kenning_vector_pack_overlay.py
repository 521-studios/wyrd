"""Tests for scenario-pack overlay support in vector_name_select
(wyrd-ecjp.8 Phase 7).

The vector primitive admits pack lemmas alongside native culture
lemmas, filtered by the pack's allowed_pack_tags /
excluded_pack_tags + the existing gate predicates (culture / era /
stratum / exclude_tags). Pack lemmas score via the
baseline_score_pack path inside score(), which scales by pack.weight
per the D36.4 Option B template-donor baseline rule.

These tests pin the runtime pack-admission contract using synthetic
fixtures — no live bundle dependency.
"""

from __future__ import annotations

import random

from wyrd.generators.kenning.runtime.meaning import Meaning
from wyrd.generators.kenning.runtime.vector_name_select import (
    select_via_vector_scoring,
)
from wyrd.generators.kenning.vectors.schemas import (
    EligibilityGate,
    EmpiricalPriors,
    PackOverlay,
    PhonologicalVector,
    RegisterEffect,
    RequestVector,
    ScoringWeights,
)


def _m(usage: str, tags=(), phon=None) -> Meaning:
    return Meaning(
        usage=usage,
        tags=list(tags),
        meanings=[],
        sources=[],
        phonological_vector=phon,
    )


def _request_with_packs(packs: tuple[PackOverlay, ...] = ()) -> RequestVector:
    return RequestVector(
        gate=EligibilityGate(culture="english"),
        register=RegisterEffect(
            name="any",
            semantic_tags={
                "fortified": 1.0,
                "pastoral": 1.0,
                "war": 1.0,
                "trade": 1.0,
                "magic": 1.0,
            },
        ),
        weights=ScoringWeights(),
        packs=packs,
    )


# ---- single-pack admission ----------------------------------------------


def test_pack_lemmas_admitted_into_eligible_pool():
    """A pack declared in request.packs with a matching pack_meaning_db
    contributes its lemmas to the eligible pool alongside native."""
    native = _m("-shire", tags=["fortified"], phon=PhonologicalVector())
    pack_lemma = _m("-keep", tags=["fortified"], phon=PhonologicalVector())
    native_db = {"-shire": [native]}
    pack_db = {"-keep": [pack_lemma]}
    pack = PackOverlay(
        pack_name="test-pack",
        template_donor="old-norse",
        template_recipient="old-english",
        weight=1.0,
    )
    request = _request_with_packs((pack,))

    # Run many seeds — both should eventually appear
    picked = set()
    for seed in range(50):
        result = select_via_vector_scoring(
            random.Random(seed),
            native_db,
            structure=["post"],
            request=request,
            priors=EmpiricalPriors(),
            pack_meaning_dbs={"test-pack": pack_db},
        )
        if result:
            picked.add(result[0].usage)
    # Pack lemma DID appear in the pool (otherwise we'd only see -shire)
    assert "-keep" in picked, "pack lemma never admitted into eligible pool"


def test_pack_lemmas_skipped_when_pack_meaning_db_missing():
    """If request.packs declares a pack but pack_meaning_dbs has no
    entry for it, the pack contributes no lemmas. No raise — the
    score() function's pack-baseline path still runs against native
    lemmas (contributes 0 since no native lemma_ref will match)."""
    native = _m("-shire", tags=["fortified"], phon=PhonologicalVector())
    native_db = {"-shire": [native]}
    pack = PackOverlay(
        pack_name="missing-pack",
        template_donor="old-norse",
        template_recipient="old-english",
    )
    request = _request_with_packs((pack,))

    result = select_via_vector_scoring(
        random.Random(0),
        native_db,
        structure=["post"],
        request=request,
        priors=EmpiricalPriors(),
        pack_meaning_dbs={},  # missing-pack not present
    )
    # Native still picked; pack absence didn't crash
    assert len(result) == 1
    assert result[0].usage == "-shire"


# ---- pack-tag filter ----------------------------------------------------


def test_allowed_pack_tags_narrows_pack_subset():
    """PackOverlay.allowed_pack_tags admits only pack lemmas carrying
    at least one matching tag. Other pack lemmas are filtered out
    of the eligible pool pre-score."""
    war = _m("-fort", tags=["war"], phon=PhonologicalVector())
    trade = _m("-market", tags=["trade"], phon=PhonologicalVector())
    pack_db = {"-fort": [war], "-market": [trade]}
    pack = PackOverlay(
        pack_name="tatar-9c",
        template_donor="mongol",
        template_recipient="old-russian",
        allowed_pack_tags=frozenset({"war"}),
    )
    request = _request_with_packs((pack,))

    picked = set()
    for seed in range(50):
        result = select_via_vector_scoring(
            random.Random(seed),
            {},  # no native lemmas; only pack lemmas
            structure=["post"],
            request=request,
            priors=EmpiricalPriors(),
            pack_meaning_dbs={"tatar-9c": pack_db},
        )
        if result:
            picked.add(result[0].usage)
    # Only war lemma admitted
    assert picked == {"-fort"}


def test_excluded_pack_tags_drops_pack_subset():
    """PackOverlay.excluded_pack_tags drops pack lemmas carrying any
    matching tag."""
    war = _m("-fort", tags=["war"], phon=PhonologicalVector())
    religion = _m("-shrine", tags=["magic"], phon=PhonologicalVector())
    pack_db = {"-fort": [war], "-shrine": [religion]}
    pack = PackOverlay(
        pack_name="tatar-9c",
        template_donor="mongol",
        template_recipient="old-russian",
        excluded_pack_tags=frozenset({"magic"}),
    )
    request = _request_with_packs((pack,))

    picked = set()
    for seed in range(50):
        result = select_via_vector_scoring(
            random.Random(seed),
            {},
            structure=["post"],
            request=request,
            priors=EmpiricalPriors(),
            pack_meaning_dbs={"tatar-9c": pack_db},
        )
        if result:
            picked.add(result[0].usage)
    # War admitted; religion excluded
    assert "-fort" in picked
    assert "-shrine" not in picked


# ---- multi-pack composition ---------------------------------------------


def test_multi_pack_admits_lemmas_from_each_pack():
    """Two packs declared → lemmas from both contribute to the
    eligible pool."""
    pack_a = _m("-fort", tags=["war"], phon=PhonologicalVector())
    pack_b = _m("-tower", tags=["magic"], phon=PhonologicalVector())
    pack_a_db = {"-fort": [pack_a]}
    pack_b_db = {"-tower": [pack_b]}
    overlay_a = PackOverlay(
        pack_name="pack-a",
        template_donor="d1",
        template_recipient="r1",
    )
    overlay_b = PackOverlay(
        pack_name="pack-b",
        template_donor="d2",
        template_recipient="r2",
    )
    request = _request_with_packs((overlay_a, overlay_b))

    picked = set()
    for seed in range(50):
        result = select_via_vector_scoring(
            random.Random(seed),
            {},
            structure=["post"],
            request=request,
            priors=EmpiricalPriors(),
            pack_meaning_dbs={"pack-a": pack_a_db, "pack-b": pack_b_db},
        )
        if result:
            picked.add(result[0].usage)
    # Both packs contributed
    assert "-fort" in picked
    assert "-tower" in picked


def test_multi_pack_order_independent():
    """Pack-order doesn't affect the eligible pool — admitting pack A
    then B produces the same eligible set as B then A. Bit-stability
    aside (the rng walks deterministically), the SET of admitted
    lemmas should match between orderings."""
    pack_a = _m("-fort", tags=["war"], phon=PhonologicalVector())
    pack_b = _m("-tower", tags=["magic"], phon=PhonologicalVector())
    pack_a_db = {"-fort": [pack_a]}
    pack_b_db = {"-tower": [pack_b]}
    overlay_a = PackOverlay(pack_name="pack-a", template_donor="d1", template_recipient="r1")
    overlay_b = PackOverlay(pack_name="pack-b", template_donor="d2", template_recipient="r2")

    def collect_picks(packs):
        request = _request_with_packs(packs)
        picked = set()
        for seed in range(100):
            result = select_via_vector_scoring(
                random.Random(seed),
                {},
                structure=["post"],
                request=request,
                priors=EmpiricalPriors(),
                pack_meaning_dbs={"pack-a": pack_a_db, "pack-b": pack_b_db},
            )
            if result:
                picked.add(result[0].usage)
        return picked

    picks_ab = collect_picks((overlay_a, overlay_b))
    picks_ba = collect_picks((overlay_b, overlay_a))
    # Same SET of admitted lemmas regardless of declaration order.
    # (Bit-stable per-seed ordering may differ, but the eligible set
    # is order-independent.)
    assert picks_ab == picks_ba


# ---- back-compat: no pack_meaning_dbs param --------------------------


def test_no_pack_meaning_dbs_legacy_call_still_works():
    """Existing callers that don't pass pack_meaning_dbs continue to
    work (param defaults to None). Verifies wyrd-ecjp.8 didn't break
    wyrd-ecjp.5 PR B's primitive contract."""
    native = _m("-shire", tags=["fortified"], phon=PhonologicalVector())
    native_db = {"-shire": [native]}
    request = RequestVector(
        gate=EligibilityGate(culture="english"),
        register=RegisterEffect(name="any", semantic_tags={"fortified": 1.0}),
        weights=ScoringWeights(),
    )
    # No pack_meaning_dbs arg passed
    result = select_via_vector_scoring(
        random.Random(0),
        native_db,
        structure=["post"],
        request=request,
        priors=EmpiricalPriors(),
    )
    assert len(result) == 1
    assert result[0].usage == "-shire"


# ---- gate predicates apply to pack lemmas too ---------------------------


def test_pack_lemmas_respect_exclude_tags_gate():
    """exclude_tags filters pack lemmas the same way it filters native
    lemmas — pre-score, before they enter the eligible pool."""
    keep = _m("-keep", tags=["war"], phon=PhonologicalVector())
    fiction_keep = _m("-spire", tags=["fiction"], phon=PhonologicalVector())
    pack_db = {"-keep": [keep], "-spire": [fiction_keep]}
    pack = PackOverlay(
        pack_name="test-pack",
        template_donor="d",
        template_recipient="r",
    )
    request = _request_with_packs((pack,))

    picked = set()
    for seed in range(50):
        result = select_via_vector_scoring(
            random.Random(seed),
            {},
            structure=["post"],
            request=request,
            priors=EmpiricalPriors(),
            pack_meaning_dbs={"test-pack": pack_db},
            exclude_tags=frozenset({"fiction"}),
        )
        if result:
            picked.add(result[0].usage)
    assert "-keep" in picked
    assert "-spire" not in picked
