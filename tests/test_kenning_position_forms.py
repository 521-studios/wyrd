"""wyrd-eyjk/D40: the position-derived morpheme model.

Directly pins the NEW behavior of the refactor — behavior that was only
exercised indirectly (via end-to-end generation against the bundled seed)
before these tests:

* position-form RECORDING — a morpheme is re-dashed to its DERIVED position
  (index among the word's morphemes), not the variant the matcher fetched:
  ``Stokegiles`` records ``Stoke-`` + ``-giles``, a lone word records bare;
* ``_location_from_form`` — the inverse (form-dashes → position) used to bucket;
* ``Meaning.is_pure_proper_noun`` — the saint/companion-tag rule;
* the synthesized-saint-subject base-pool exclusion in BOTH scoring modes;
* bare-surface resolution of a position-form against a differently-stored
  dash-variant (``-giles`` → a ``Giles-``-keyed morpheme).
"""

from __future__ import annotations

import pytest

from wyrd.generators.kenning.runtime.meaning import Meaning
from wyrd.generators.kenning.runtime.proportions import (
    MeaningGenerator,
    _location_from_form,
    _resolve_surface,
)
from wyrd.generators.kenning.runtime.word import Word

# ---- position-form recording (Word.get_samples / get_lone_samples) ---------


def test_compound_records_morphemes_at_derived_positions():
    """A word-final ``Giles`` records as the post form ``-giles`` and the
    word-initial ``Stoke`` as the pre form ``Stoke-`` — regardless of which
    dash-variant the matcher fetched (here both are stored pre-shaped)."""
    w = Word([Meaning("Stoke-", [], [], {}), Meaning("Giles-", ["saint"], [], {})])
    assert w.get_samples() == {"Stoke-", "-giles"}
    # Gileston: Giles at word-initial → pre; -ton word-final → post.
    w2 = Word([Meaning("Giles-", ["saint"], [], {}), Meaning("-ton", ["settlement"], [], {})])
    assert w2.get_samples() == {"Giles-", "-ton"}


def test_inner_morpheme_records_at_inner_position():
    """An interior morpheme records as the inner form (dashes both sides),
    lowercased per D39."""
    w = Word(
        [
            Meaning("Pre-", [], [], {}),
            Meaning("Mid-", [], [], {}),
            Meaning("-post", [], [], {}),
        ]
    )
    assert w.get_samples() == {"Pre-", "-mid-", "-post"}


def test_lone_word_records_bare_surface():
    """A single-morpheme word's sole morpheme is structurally bare — recorded
    as its dash-stripped surface, even when stored as a dashed variant."""
    assert Word([Meaning("-andrew", ["male name"], [], {})]).get_lone_samples() == {"andrew"}
    # Bare keeps the stored surface case (only post/inner lowercase per D39),
    # so a lone ``Giles-`` records bare ``Giles``. The part pool is empty for a
    # lone word; the lone pool is empty for a compound.
    lone = Word([Meaning("Giles-", ["saint"], [], {})])
    assert lone.get_lone_samples() == {"Giles"}
    assert lone.get_samples() == set()


def test_partial_decomposition_positions_account_for_unmatched_fragments():
    """wyrd-eyjk round 2: a matched morpheme in a partially-decomposed word
    derives its position over ALL word elements, not just the Meanings — a
    word-final ``giles`` after an unmatched ``Stoke`` fragment is post (would
    record ``-giles``), not bare. (Recorded corpus words are fully decomposed,
    so this only bites a partial-decomposition caller; pinned for robustness +
    to keep get_samples / get_structure in lockstep.)"""
    w = Word(["Stoke", Meaning("Giles-", ["saint"], [], {})])
    assert w.get_samples() == {"-giles"}
    assert w.get_structure() == (("post",),)
    # And a leading matched morpheme before an unmatched tail → pre.
    w2 = Word([Meaning("Stoke-", [], [], {}), "giles"])
    assert w2.get_samples() == {"Stoke-"}
    assert w2.get_structure() == (("pre",),)


@pytest.mark.parametrize(
    "usage,expected",
    [("-x-", "inner"), ("x-", "pre"), ("-x", "post"), ("x", "bare"), ("", "bare")],
)
def test_location_from_form(usage, expected):
    """The form→position inverse used to bucket position-form usages —
    mirrors ``Meaning._set_location``."""
    assert _location_from_form(usage) == expected


# ---- Meaning.is_pure_proper_noun: saint + companion-tag rule ---------------


def test_is_pure_proper_noun_saint_branch():
    # A synthesized saint subject (saint + companion tags, no common-noun
    # sense) is pure.
    assert Meaning("Mary-", ["saint", "personal-name", "religious"], [], {}).is_pure_proper_noun()
    # A name with no common-noun tag is pure.
    assert Meaning("-andrew", ["male name"], [], {}).is_pure_proper_noun()
    # A place element merely CO-TAGGED with saint/name keeps a common-noun
    # tag → NOT pure (must still generate as a base element).
    assert not Meaning("-stan", ["saint", "geology"], [], {}).is_pure_proper_noun()
    assert not Meaning("stone", ["topography", "male name"], [], {}).is_pure_proper_noun()
    # Neither name nor saint → never pure.
    assert not Meaning("-ton", ["settlement"], [], {}).is_pure_proper_noun()


# ---- bare-surface resolution of a position-form ----------------------------


def test_resolve_surface_matches_position_form_to_stored_variant():
    """A recorded post-form ``-giles`` resolves to a morpheme the bundle stored
    only under the pre-variant key ``Giles-`` — match is by bare surface."""
    giles = Meaning("Giles-", ["saint"], ["Saint Giles"], {})
    meaning_db = {"Giles-": [giles], "-ton": [Meaning("-ton", ["settlement"], [], {})]}
    resolved = _resolve_surface(meaning_db, "-giles")
    assert giles in resolved


# ---- surface-index cache: bounded + identity-checked -----------------------


def test_surface_index_cache_is_bounded_and_identity_checked():
    """wyrd-eyjk round 3: the id()-keyed surface-index cache must stay bounded
    (ad-hoc meaning_dbs don't accumulate) and never cross-contaminate — a
    distinct meaning_db gets its OWN index, never a stale hit from a recycled
    id(). The stored reference + ``is`` check guarantee the latter."""
    from wyrd.generators.kenning.runtime import proportions as prop

    prop._SURFACE_INDEX_CACHE.clear()
    try:
        for i in range(prop._SURFACE_INDEX_CACHE_MAX + 10):
            prop._surface_index_for({f"-x{i}": [Meaning(f"-x{i}", [], [], {})]})
        assert len(prop._SURFACE_INDEX_CACHE) <= prop._SURFACE_INDEX_CACHE_MAX
        # Same object → cached (identity hit returns the same index object).
        a = {"-ay": [Meaning("-ay", ["t"], [], {})]}
        assert prop._surface_index_for(a) is prop._surface_index_for(a)
        # A different db never returns another db's index.
        b = {"-by": [Meaning("-by", ["t"], [], {})]}
        idx_b = prop._surface_index_for(b)
        assert "by" in idx_b and "ay" not in idx_b
    finally:
        prop._SURFACE_INDEX_CACHE.clear()


# ---- synthesized-saint-subject base-pool exclusion (both scoring modes) ----


def test_saint_subject_excluded_from_proportions_base_buckets():
    """A pure-proper-noun saint subject must not seed ANY generation bucket
    (it reaches names only via the param-gated St-dedication synthesis); a
    plain common morpheme at the same surface position still buckets."""
    meaning_db = {
        "Mary-": [Meaning("Mary-", ["saint", "personal-name", "religious"], ["Saint Mary"], {})],
        "-ham": [Meaning("-ham", ["settlement"], ["homestead"], {})],
    }
    # Record both as lone (bare) occurrences.
    mg = MeaningGenerator(meaning_db, {}, {})
    mg.load_parts({"mary": 1, "ham": 1}, "single")
    items_by_bucket = {key: set(g.elements) for key, g in mg.generators.items()}
    all_items = set().union(*items_by_bucket.values()) if items_by_bucket else set()
    assert "ham" in all_items, "common morpheme should bucket"
    assert "mary" not in all_items, "saint subject must not seed any base bucket"


def test_saint_subject_excluded_from_vector_base_pool():
    from wyrd.generators.kenning.runtime.vector_name_select import build_non_position_eligible
    from wyrd.generators.kenning.vectors.schemas import EligibilityGate

    db = {
        "Mary-": [Meaning("Mary-", ["saint", "personal-name", "religious"], ["Saint Mary"], {})],
        "-ton": [Meaning("-ton", ["settlement"], ["town"], {})],
    }
    pool = build_non_position_eligible(
        db,
        gate=EligibilityGate(culture="english"),
        exclude_tags=frozenset(),
        pack_meaning_dbs=None,
        packs=(),
    )
    usages = {m.usage for m in pool}
    assert "-ton" in usages
    assert "Mary-" not in usages, "saint subject must be excluded from the vector base pool"


# ---- wyrd-g1hj: single-morpheme structure exclusion + given-name base-pool --


def test_is_single_morpheme_structure():
    from wyrd.generators.kenning.runtime.proportions import _is_single_morpheme_structure

    assert _is_single_morpheme_structure(((("bare", "single"),),))  # 1 word, 1 morpheme
    assert _is_single_morpheme_structure(((("bare", "name", "single"),),))
    assert not _is_single_morpheme_structure(((("pre",), ("post",)),))  # Higham: 2 morphemes
    # Green Park: 2 words, 2 morphemes total → not single
    assert not _is_single_morpheme_structure(((("bare", "single"),), (("bare", "single"),)))


def test_is_given_name():
    assert Meaning("-andrew", ["male name"], [], {}).is_given_name()
    assert Meaning("Mary-", ["female name"], [], {}).is_given_name()
    # Family names are NOT given names (kept for manorial/place legitimacy).
    assert not Meaning("-smith", ["family name"], [], {}).is_given_name()
    assert not Meaning("-ton", ["settlement"], [], {}).is_given_name()


def test_load_proportions_excludes_single_morpheme_from_generation():
    """wyrd-g1hj: a single-morpheme structure (whole name = one morpheme) is
    excluded from the loaded generator (both ``structs`` and ``_all_structs``),
    so no generation path can produce it. Multi-morpheme structures survive.
    (The bundle/proportions still RECORD it — this is a load-time generation
    filter, not a mining change.)"""
    from wyrd.generators.kenning.runtime.proportions import load_proportions

    meaning_db = {
        "Stoke-": [Meaning("Stoke-", [], [], {})],
        "-ton": [Meaning("-ton", [], [], {})],
        "Bath": [Meaning("Bath", [], [], {})],
    }
    data = {
        "usages": {"Stoke-": 1, "-ton": 1},
        "single_usages": {"Bath": 1},
        "structures": [
            {
                "proportion": 5,
                "words": [[{"location": "pre"}, {"location": "post"}]],
            },  # 2 morphemes
            {"proportion": 5, "words": [[{"location": "bare"}]]},  # 1 morpheme → excluded
        ],
    }
    ng = load_proportions(data, meaning_db, {})
    assert ((("pre",), ("post",)),) in ng.structs, "multi-morpheme structure must survive"
    assert ((("bare", "single"),),) not in ng.structs, "single-morpheme excluded from generation"
    assert ((("bare", "single"),),) not in ng._all_structs


def test_given_name_excluded_from_base_pool_both_modes():
    """wyrd-g1hj: pure given-name etymons (male/female) are excluded from the
    base pool in BOTH scoring modes — like saints. Family names stay (manorial/
    toponymic + the synthetic fixtures use them); co-tagged place elements stay."""
    from wyrd.generators.kenning.runtime.vector_name_select import build_non_position_eligible
    from wyrd.generators.kenning.vectors.schemas import EligibilityGate

    db = {
        "-john": [Meaning("-john", ["male name"], ["John"], {})],
        "-smith": [Meaning("-smith", ["family name"], ["Smith"], {})],
        "-ham": [Meaning("-ham", ["settlement"], ["homestead"], {})],
    }
    # proportions mode (load_parts buckets)
    mg = MeaningGenerator(db, {}, {})
    mg.load_parts({"john": 1, "smith": 1, "ham": 1}, "single")
    items = (
        set().union(*[set(g.elements) for g in mg.generators.values()]) if mg.generators else set()
    )
    assert "ham" in items and "smith" in items
    assert "john" not in items, "given-name etymon excluded from proportions base pool"
    # vector mode (build_non_position_eligible)
    pool = {
        m.usage
        for m in build_non_position_eligible(
            db,
            gate=EligibilityGate(culture="english"),
            exclude_tags=frozenset(),
            pack_meaning_dbs=None,
            packs=(),
        )
    }
    assert "-ham" in pool and "-smith" in pool
    assert "-john" not in pool, "given-name etymon excluded from vector base pool"
