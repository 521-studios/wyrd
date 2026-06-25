"""Unit tests for kenning's Meaning model and meaning DB loader."""

from __future__ import annotations

import random
from collections import Counter

from wyrd.generators.kenning.runtime.meaning import Meaning, _mimic_case, load_meanings


def make_meaning(usage, tags=None, meanings=None, sources=None):
    return Meaning(usage, tags or [], meanings or [], sources or {})


def test_location_always_bare():
    # wyrd-aicu: identity is bare now (the producer de-dashes every stored usage
    # — D45), so a Meaning's ``.location`` is always "bare". The retired
    # dash-shape decode (-x- inner / x- pre / -x post) can no longer be
    # constructed in production; ``.location`` survives only as a render hint.
    assert make_meaning("ham").location == "bare"


def test_is_name_true_for_male_female_or_family():
    assert make_meaning("Alf-", tags=["male name"]).is_name()
    assert make_meaning("Alf-", tags=["female name"]).is_name()
    assert make_meaning("Alf-", tags=["family name"]).is_name()


def test_is_name_false_for_other_tags():
    assert not make_meaning("Bre-", tags=["water"]).is_name()


def test_is_saint_detection():
    assert make_meaning("saint", tags=["saint"]).is_saint()
    assert not make_meaning("ham", tags=["water"]).is_saint()


def test_key_simple_bare():
    # wyrd-aicu: identity is bare → ``.location`` (the first key element) is
    # always "bare" (the retired dash-shape decode is gone). Position rides on
    # the proportions axis now, not the morpheme key.
    assert make_meaning("ton").key() == ("bare",)


def test_key_with_name_tag():
    assert make_meaning("Alf", tags=["male name"]).key() == ("bare", "name")


def test_key_with_saint_usage():
    # Saint key path triggers when usage text matches "saint" exactly
    # (with dashes stripped, lowercase) — separate from is_name().
    assert make_meaning("saint", tags=[]).key() == ("bare", "saint")


# wyrd-57d8: base-pool class exclusions. is_manorial_subject identifies the
# synthesized Norman manorial subjects (tagged 'manorial'); is_base_pool_excluded
# is the single predicate every base pool consults so manorial / saint / given-
# name / connector morphemes never enter plain generation.


def test_is_manorial_subject_true_only_for_manorial_tag():
    assert make_meaning("Malet", tags=["manorial", "norman"]).is_manorial_subject()
    assert not make_meaning("ham", tags=["water"]).is_manorial_subject()
    assert not make_meaning("Smith-", tags=["family name"]).is_manorial_subject()


def test_is_base_pool_excluded_true_for_each_excluded_class():
    # Synthesized manorial subject (the wyrd-57d8 leak class).
    assert make_meaning("Malet", tags=["manorial", "norman"]).is_base_pool_excluded()
    # Pure-proper-noun saint subject (reaches output only via St-dedication).
    assert make_meaning(
        "Mary", tags=["saint", "personal-name", "religious"]
    ).is_base_pool_excluded()
    # Pure-proper-noun personal given name.
    assert make_meaning("Edmund", tags=["male name"]).is_base_pool_excluded()
    # Connector particle (needs a complement; would dangle as a lone word).
    assert make_meaning("cum", tags=[]).is_base_pool_excluded()


def test_is_base_pool_excluded_false_for_legitimate_base_morphemes():
    # Plain place element.
    assert not make_meaning("ham", tags=["water"]).is_base_pool_excluded()
    # Place element merely CO-TAGGED with a name (not a pure proper noun).
    assert not make_meaning("Stan-", tags=["male name", "stone"]).is_base_pool_excluded()
    # A real FAMILY-name etymon forms legitimate manorial toponyms — stays in
    # (it is a pure proper noun but neither saint nor given name nor a
    # synthesized 'manorial'-tagged subject).
    assert not make_meaning("Smith-", tags=["family name"]).is_base_pool_excluded()


def test_load_meanings_indexes_by_usage_and_tag():
    data = [
        {
            "modifier_tags": ["water"],
            "meaning": ["stream"],
            "words": [{"modern_usage": "-bourne", "old_english": "burna"}],
        }
    ]
    meaning_db, tag_db = load_meanings(data)
    assert "-bourne" in meaning_db
    assert meaning_db["-bourne"][0].sources == {"old_english": "burna"}
    assert "water" in tag_db
    assert "-bourne" in tag_db["water"]


def test_load_meanings_pluralizes_name_tagged_entries():
    data = [
        {
            "modifier_tags": ["male name"],
            "meaning": ["Alfred (Name)"],
            "words": [{"modern_usage": "Alf-"}],
        }
    ]
    meaning_db, tag_db = load_meanings(data)
    assert "Alf-" in meaning_db
    # Plural form added automatically for non-s-ending names.
    assert "Alf-s" in meaning_db
    assert "Alf-s" in tag_db["male name"]


def test_load_meanings_does_not_pluralize_already_plural_names():
    data = [
        {
            "modifier_tags": ["family name"],
            "meaning": ["Tribe (Name)"],
            "words": [{"modern_usage": "Saxons"}],
        }
    ]
    meaning_db, _ = load_meanings(data)
    assert "Saxons" in meaning_db
    assert "Saxonss" not in meaning_db


# --- D18 spelling variants -------------------------------------------------


def _meaning_with_variants(variants):
    return Meaning(
        usage="Bridg-",
        tags=["water"],
        meanings=["Bridge"],
        sources={"old_english": ["brycg"]},
        variants=variants,
    )


def test_pick_variant_returns_none_when_pool_empty():
    """No variants → no substitution, regardless of spelling_variety."""
    m = make_meaning("Bridg-", sources={"old_english": ["brycg"]})
    assert m.pick_variant(random.Random(0), 1.0) is None


def test_pick_variant_returns_none_when_variety_zero():
    """spelling_variety=0 always preserves the canonical surface form."""
    m = _meaning_with_variants({"old_english": [("brycg", 5), ("brigge", 3)]})
    assert m.pick_variant(random.Random(0), 0.0) is None


def test_pick_variant_returns_a_variant_when_variety_one():
    """At spelling_variety=1, a non-empty pool always yields a variant."""
    m = _meaning_with_variants({"old_english": [("brycg", 5), ("brigge", 3)]})
    picked = m.pick_variant(random.Random(0), 1.0)
    assert picked in {"brycg", "brigge"}


def test_pick_variant_weighted_distribution_favors_heavier_weight():
    """Over many draws, variants are picked roughly in proportion to weight."""
    m = _meaning_with_variants({"old_english": [("a", 90), ("b", 10)]})
    counts = Counter(m.pick_variant(random.Random(i), 1.0) for i in range(2000))
    # 'a' should dominate at roughly 9:1. Allow a wide band against RNG variance.
    assert counts["a"] > counts["b"] * 5


def test_load_meanings_extracts_variants_from_variant_field():
    """``<lang>_variants`` keys are split out of ``sources`` into the
    Meaning's variants dict; canonical language form lists stay in sources
    unchanged."""
    data = [
        {
            "modifier_tags": ["water"],
            "meaning": ["Bridge"],
            "words": [
                {
                    "modern_usage": "Bridg-",
                    "old_english": ["brycg"],
                    "old_english_variants": [
                        {"form": "brigge", "weight": 3},
                        {"form": "brige", "weight": 1},
                    ],
                }
            ],
        }
    ]
    meaning_db, _ = load_meanings(data)
    m = meaning_db["Bridg-"][0]
    assert m.variants == {"old_english": [("brigge", 3), ("brige", 1)]}
    # Canonical language array stays in sources, variant field stripped.
    assert m.sources == {"old_english": ["brycg"]}


def test_load_meanings_omits_variants_field_when_absent():
    """Legacy meanings.json data without the variant field still loads cleanly
    — every Meaning gets an empty variants dict."""
    data = [
        {
            "modifier_tags": [],
            "meaning": ["x"],
            "words": [{"modern_usage": "X-", "old_english": ["x"]}],
        }
    ]
    meaning_db, _ = load_meanings(data)
    assert meaning_db["X-"][0].variants == {}


def test_mimic_case_capitalizes_at_word_start():
    """A pre-position canonical 'Bridg-' should re-cap a variant."""
    assert _mimic_case("Bridg-", "brycg") == "Brycg"


def test_mimic_case_lowercases_at_word_end():
    """A post-position canonical '-water' should keep the variant lowercase
    even when the variant comes in upper-cased."""
    assert _mimic_case("-water", "WATYR") == "watyr"


def test_mimic_case_preserves_internal_capitals_under_title_template():
    """When the canonical template is title-case, the variant's first letter
    is capitalized but internal capitals are preserved — a future variant
    like 'McGregor' or 'O'Brien' shouldn't be flattened to lowercase."""
    assert _mimic_case("Mac-", "McGregor") == "McGregor"
    assert _mimic_case("Mc-", "MacKay") == "MacKay"


def test_mimic_case_handles_empty_template():
    """A template with no letters (just dashes) returns the variant verbatim."""
    assert _mimic_case("-", "x") == "x"


# --- D8 inflection picker --------------------------------------------------


def _meaning_with_inflections(inflections):
    return Meaning(
        usage="-cot",
        tags=["habitation"],
        meanings=["cottage"],
        sources={"old_english": ["cot"]},
        inflections=inflections,
    )


def test_pick_inflection_returns_none_when_pool_empty():
    """No inflections → no substitution, regardless of density."""
    m = make_meaning("-cot", sources={"old_english": ["cot"]})
    assert m.pick_inflection(random.Random(0), 1.0) is None


def test_pick_inflection_returns_none_when_density_zero():
    """inflection_density=0 always preserves the lemma."""
    m = _meaning_with_inflections({"old_english": [("cotum", "dative_or_pl")]})
    assert m.pick_inflection(random.Random(0), 0.0) is None


def test_pick_inflection_returns_form_and_label_when_density_one():
    """At density=1, a non-empty pool always yields (form, label)."""
    m = _meaning_with_inflections({"old_english": [("cotum", "dative_or_pl")]})
    picked = m.pick_inflection(random.Random(0), 1.0)
    assert picked == ("cotum", "dative_or_pl")


def test_pick_inflection_uniformly_samples_among_pool():
    """Without per-form weights (we don't track inflection frequency), the
    picker is uniform across pool entries."""
    m = _meaning_with_inflections(
        {
            "old_english": [
                ("cotum", "dative_or_pl"),
                ("cotan", "weak_oblique"),
                ("cotes", "genitive_strong"),
            ]
        }
    )
    counts = Counter(m.pick_inflection(random.Random(i), 1.0) for i in range(900))
    # Three entries, each should land roughly 300 times. Wide tolerance.
    for inflected in (
        ("cotum", "dative_or_pl"),
        ("cotan", "weak_oblique"),
        ("cotes", "genitive_strong"),
    ):
        assert 200 < counts[inflected] < 400


def test_pick_inflection_pools_across_languages():
    """If a meaning has inflections in multiple languages (rare but possible
    when rando-port words attach forms across language families), the
    picker draws from the union."""
    m = _meaning_with_inflections(
        {
            "old_english": [("cotum", "dative_or_pl")],
            "old_scandinavian": [("kotr", "nominative")],
        }
    )
    counts = Counter(m.pick_inflection(random.Random(i), 1.0) for i in range(600))
    assert counts[("cotum", "dative_or_pl")] > 0
    assert counts[("kotr", "nominative")] > 0


def test_load_meanings_extracts_inflections_from_inflection_field():
    """`<lang>_inflections` keys are split out of `sources` into the
    Meaning's inflections dict; canonical language form lists stay in
    sources unchanged."""
    data = [
        {
            "modifier_tags": ["habitation"],
            "meaning": ["cottage"],
            "words": [
                {
                    "modern_usage": "-cot",
                    "old_english": ["cot", "cotum"],
                    "old_english_inflections": [
                        {"form": "cotum", "inflection": "dative_or_pl"},
                    ],
                }
            ],
        }
    ]
    meaning_db, _ = load_meanings(data)
    m = meaning_db["-cot"][0]
    assert m.inflections == {"old_english": [("cotum", "dative_or_pl")]}
    # Canonical language array stays in sources, inflection field stripped.
    assert m.sources == {"old_english": ["cot", "cotum"]}


def test_load_meanings_omits_inflections_field_when_absent():
    """Legacy data without inflection metadata still loads cleanly."""
    data = [
        {
            "modifier_tags": [],
            "meaning": ["x"],
            "words": [{"modern_usage": "X-", "old_english": ["x"]}],
        }
    ]
    meaning_db, _ = load_meanings(data)
    assert meaning_db["X-"][0].inflections == {}


def test_load_meanings_extracts_citations_from_citation_field():
    """`<lang>_citations` keys are split out of `sources` into the
    Meaning's citations dict; canonical language form lists stay in
    sources unchanged. Mirrors the regression pattern for variants /
    inflections."""
    data = [
        {
            "modifier_tags": ["habitation"],
            "meaning": ["homestead"],
            "words": [
                {
                    "modern_usage": "-ham",
                    "old_english": ["ham"],
                    "old_english_citations": ["mawer_1920", "skeat_1901"],
                }
            ],
        }
    ]
    meaning_db, _ = load_meanings(data)
    m = meaning_db["-ham"][0]
    assert m.citations == {"old_english": ["mawer_1920", "skeat_1901"]}
    # Canonical language array stays in sources, citations field stripped.
    assert m.sources == {"old_english": ["ham"]}


def test_load_meanings_omits_citations_field_when_absent():
    """Legacy data without citation metadata still loads cleanly with an
    empty citations dict."""
    data = [
        {
            "modifier_tags": [],
            "meaning": ["x"],
            "words": [{"modern_usage": "X-", "old_english": ["x"]}],
        }
    ]
    meaning_db, _ = load_meanings(data)
    assert meaning_db["X-"][0].citations == {}


def test_load_meanings_extracts_attested_years_from_attested_year_field():
    """`<lang>_attested_years` keys (D5-1 / wyrd-bag) are split out of
    `sources` into the Meaning's attested_years dict; canonical language
    form lists stay in sources unchanged. Mirrors the regression pattern
    for variants / inflections / citations."""
    data = [
        {
            "modifier_tags": ["habitation"],
            "meaning": ["enclosure"],
            "words": [
                {
                    "modern_usage": "-tun",
                    "old_english": ["tun"],
                    "old_english_attested_years": [
                        {"form": "tun", "year": 1086},
                        {"form": "tunes", "year": 1242},
                    ],
                }
            ],
        }
    ]
    meaning_db, _ = load_meanings(data)
    m = meaning_db["-tun"][0]
    assert m.attested_years == {
        "old_english": [("tun", 1086), ("tunes", 1242)],
    }
    # Canonical language array stays in sources, attested-years field stripped.
    assert m.sources == {"old_english": ["tun"]}


def test_load_meanings_omits_attested_years_field_when_absent():
    """Legacy data without attested-year metadata still loads cleanly
    with an empty attested_years dict."""
    data = [
        {
            "modifier_tags": [],
            "meaning": ["x"],
            "words": [{"modern_usage": "X-", "old_english": ["x"]}],
        }
    ]
    meaning_db, _ = load_meanings(data)
    assert meaning_db["X-"][0].attested_years == {}


def test_load_meanings_propagates_inflections_to_plural_form():
    """A name-tagged meaning auto-pluralizes (e.g. 'Alf-' → 'Alf-s'). The
    pluralized Meaning must inherit EVERY constructor kwarg of the
    singular — a refactor that drops one on the plural branch should
    be caught here. This test extends as new kwargs are added to
    Meaning so the regression net stays complete."""
    data = [
        {
            "modifier_tags": ["male name"],
            "meaning": ["Alfred (Name)"],
            "words": [
                {
                    "modern_usage": "Alf-",
                    "old_english": ["alfred", "alfredan"],
                    "old_english_inflections": [
                        {"form": "alfredan", "inflection": "weak_oblique"},
                    ],
                    "old_english_variants": [
                        {"form": "ælfred", "weight": 5},
                    ],
                    "old_english_citations": ["mawer_1920"],
                    "old_english_attested_years": [
                        {"form": "alfred", "year": 893},
                    ],
                }
            ],
        }
    ]
    meaning_db, _ = load_meanings(data)
    plural = meaning_db["Alf-s"][0]
    assert plural.inflections == {"old_english": [("alfredan", "weak_oblique")]}
    assert plural.variants == {"old_english": [("ælfred", 5)]}
    assert plural.citations == {"old_english": ["mawer_1920"]}
    assert plural.attested_years == {"old_english": [("alfred", 893)]}


def test_pick_inflection_partial_density_gates_per_call():
    """At density=0.5 the substitution rate over many trials should be
    roughly half — the gate is a per-call random draw, not a global toggle."""
    m = _meaning_with_inflections({"old_english": [("cotum", "dative_or_pl")]})
    hits = sum(1 for i in range(2000) if m.pick_inflection(random.Random(i), 0.5) is not None)
    # Wide tolerance band against RNG variance — should land near 1000.
    assert 800 < hits < 1200


# --- attested_in_era_range: REMOVED (D44) ----------------------------------
# The attested-inside-WINDOW predicate this block pinned was retired with
# D44 (the attested-years DATA survives and now feeds D46's record gate).
# The era contract is pinned by tests/test_kenning_era_accretion.py.


# --- in_stratum (wyrd-lr4 Phase 3) ----------------------------------------


def _meaning_with_stratum_data(stratum: dict[str, dict[str, str]]) -> Meaning:
    return Meaning(
        usage="Test-",
        tags=[],
        meanings=[],
        sources={},
        stratum=stratum,
    )


def test_in_stratum_none_means_no_filter():
    """stratum=None is the runtime's 'no --stratum passed' signal —
    every Meaning passes regardless of its data."""
    m = _meaning_with_stratum_data({"celtic_mix": {"caer": "latin-loan"}})
    assert m.in_stratum(None) is True


def test_in_stratum_empty_data_passes_through():
    """Meaning with no stratum data passes any --stratum value — the
    documented Phase 3 rule that languages without a Phase 1
    classifier should not be silently dropped from filtered runs."""
    m = make_meaning("Bridg-")
    assert m.in_stratum("native-welsh") is True
    assert m.in_stratum("brittonic-substrate") is True


def test_in_stratum_form_in_target_stratum_passes():
    """A Meaning with at least one form classified into the target
    stratum is admitted."""
    m = _meaning_with_stratum_data({"celtic_mix": {"caer": "native-welsh"}})
    assert m.in_stratum("native-welsh") is True


def test_in_stratum_no_form_in_target_stratum_excluded():
    """A Meaning whose every form has stratum data BUT none of it
    matches the target tag is dropped — non-empty stratum dict turns
    off the no-data passthrough, so the filter actually applies."""
    m = _meaning_with_stratum_data({"celtic_mix": {"caer": "latin-loan"}})
    assert m.in_stratum("native-welsh") is False


def test_in_stratum_searches_across_languages():
    """The OR check spans every language entry — a Meaning with
    welsh-bucket forms in 'native-welsh' AND old-french-bucket forms
    in 'frankish' passes either filter (once Phase 4 lands; Welsh is
    the only classified family today, but the Meaning data shape
    supports cross-language stratum data)."""
    m = _meaning_with_stratum_data(
        {
            "celtic_mix": {"caer": "native-welsh"},
            "old_french": {"villa": "frankish"},
        }
    )
    assert m.in_stratum("native-welsh") is True
    assert m.in_stratum("frankish") is True
    assert m.in_stratum("medieval-welsh") is False


def test_in_stratum_returns_true_when_any_form_matches_in_mixed_data():
    """A Meaning carrying multiple forms with mixed strata is admitted
    as long as ANY form matches the target tag — the OR semantics
    apply within a single language bucket the same way they apply
    across languages."""
    m = _meaning_with_stratum_data(
        {
            "celtic_mix": {
                "afon": "native-welsh",
                "caer": "latin-loan",
                "din": "native-welsh",
            }
        }
    )
    assert m.in_stratum("native-welsh") is True
    assert m.in_stratum("medieval-welsh") is False


# ---------------------------------------------------------------------
# wyrd-ha9q Phase 2c: english_shaped accessor + load round-trip
# ---------------------------------------------------------------------


def test_english_shaped_defaults_to_empty_dict_for_legacy_meaning():
    """Meaning constructed without english_shaped kwarg (legacy
    bundle pre-Phase 2c) gets an empty dict."""
    m = Meaning(usage="Test-", tags=[], meanings=[], sources={})
    assert m.english_shaped == {}


def test_load_meanings_strips_english_shaped_suffix_into_dedicated_dict():
    """`<lang>_english_shaped` JSON arrays land in `Meaning.english_shaped`
    keyed by (lang_field, canonical_form). `sources` keeps the language
    form array but does NOT carry the _english_shaped sibling."""
    data = [
        {
            "modifier_tags": ["monster"],
            "meaning": ["golem"],
            "modifier_type": None,
            "words": [
                {
                    "modern_usage": "Golem",
                    "hebrew": ["גולם", "גלם"],
                    "hebrew_english_shaped": [
                        {"form": "גולם", "english_shaped": "golem"},
                        {"form": "גלם", "english_shaped": "gelem"},
                    ],
                }
            ],
        }
    ]
    meaning_db, _ = load_meanings(data)
    assert "Golem" in meaning_db
    m = meaning_db["Golem"][0]
    assert m.sources == {"hebrew": ["גולם", "גלם"]}
    assert m.english_shaped == {"hebrew": {"גולם": "golem", "גלם": "gelem"}}


def test_load_meanings_round_trips_with_no_english_shaped_field():
    """Legacy bundle entries (Latin-script langs / pre-Phase-2c) load
    cleanly with an empty english_shaped dict — no KeyError."""
    data = [
        {
            "modifier_tags": [],
            "meaning": ["bridge"],
            "modifier_type": "Topographical",
            "words": [
                {
                    "modern_usage": "Bridg-",
                    "old_english": ["brycg"],
                }
            ],
        }
    ]
    meaning_db, _ = load_meanings(data)
    m = meaning_db["Bridg-"][0]
    assert m.english_shaped == {}


def test_phase2d_renderings_default_to_empty_dict_for_legacy_meaning():
    """Meaning constructed without the Phase 2d kwargs (older bundles)
    gets empty dicts on all four."""
    m = Meaning(usage="Test-", tags=[], meanings=[], sources={})
    assert m.original_script == {}
    assert m.transliteration == {}
    assert m.pronunciation == {}


def test_load_meanings_strips_phase2d_suffixes_into_dedicated_dicts():
    """`<lang>_original_script` / `_transliteration` / `_pronunciation`
    JSON arrays land in dedicated Meaning attrs; `sources` does NOT
    pick them up. Mirrors the Phase 2c english_shaped suffix behavior."""
    data = [
        {
            "modifier_tags": [],
            "meaning": ["dog"],
            "modifier_type": None,
            "words": [
                {
                    "modern_usage": "Kelev",
                    "hebrew": ["כלב"],
                    "hebrew_original_script": [{"form": "כלב", "original_script": "כֶּלֶב"}],
                    "hebrew_transliteration": [{"form": "כלב", "transliteration": "kɛ́lɛḇ"}],
                    "hebrew_pronunciation": [
                        {"form": "כלב", "ipa": "/kalb/", "dialect": "Biblical-Hebrew"}
                    ],
                }
            ],
        }
    ]
    meaning_db, _ = load_meanings(data)
    m = meaning_db["Kelev"][0]
    assert m.sources == {"hebrew": ["כלב"]}
    assert m.original_script == {"hebrew": {"כלב": "כֶּלֶב"}}
    assert m.transliteration == {"hebrew": {"כלב": "kɛ́lɛḇ"}}
    assert m.pronunciation == {"hebrew": {"כלב": {"ipa": "/kalb/", "dialect": "Biblical-Hebrew"}}}


def test_load_meanings_handles_pronunciation_with_null_dialect():
    """Pronunciation entries with no dialect key get dialect=None on
    the loaded dict — preserves the wyrd-ha9q null-dialect contract
    that an untagged-canonical IPA leaves dialect at NULL."""
    data = [
        {
            "modifier_tags": [],
            "meaning": ["jinn"],
            "modifier_type": None,
            "words": [
                {
                    "modern_usage": "Jinn",
                    "arabic": ["جن"],
                    "arabic_pronunciation": [{"form": "جن", "ipa": "/d͡ʒɪnː/"}],
                }
            ],
        }
    ]
    meaning_db, _ = load_meanings(data)
    m = meaning_db["Jinn"][0]
    assert m.pronunciation["arabic"]["جن"] == {
        "ipa": "/d͡ʒɪnː/",
        "dialect": None,
    }


def test_db_to_export_to_load_round_trip_preserves_english_shaped(tmp_path):
    """End-to-end: seed a fresh DB with a wave-2 etymon carrying
    english_shaped, run `export_meanings`, then `load_meanings` on the
    output — the rendering survives all the way to `Meaning.english_shaped`.

    Pinned because the exporter and loader were tested independently;
    a field-name mismatch between `_emit_english_shaped_list`'s output
    keys and `_ENGLISH_SHAPED_SUFFIX`-stripping would pass both tests
    in isolation but break the round-trip. This locks the contract that
    the two ends agree on the wire format."""
    from wyrd.generators.kenning.lexicon import (
        LexiconDB,
        export_meanings,
        init_schema,
        seed_from_meanings,
    )

    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)

    # Seed a Hebrew family + populate english_shaped on the row.
    with LexiconDB(db_path) as db:
        db.upsert_source(id="rando-port", title="rando")
        seed_from_meanings(
            db,
            [
                {
                    "modifier_tags": ["monster"],
                    "meaning": ["golem"],
                    "modifier_type": None,
                    "words": [{"modern_usage": "Golem", "hebrew": ["גולם"]}],
                }
            ],
            "rando-port",
        )
        db.conn.execute(
            "UPDATE etymon SET english_shaped = ? WHERE canonical_form = ? AND language = ?",
            ("golem", "גולם", "he"),
        )
        db.commit()

        subjects = export_meanings(db, include_rando=True)

    # Round-trip through load_meanings.
    meaning_db, _ = load_meanings(subjects)
    assert "Golem" in meaning_db, list(meaning_db)
    m = meaning_db["Golem"][0]
    # Form arrived in the right bundle bucket AND english_shaped came
    # through to the runtime accessor.
    assert m.sources.get("hebrew") == ["גולם"]
    assert m.english_shaped.get("hebrew", {}).get("גולם") == "golem"


def test_db_to_export_to_load_round_trip_preserves_phase2d_renderings(tmp_path):
    """wyrd-qhs0 Phase 2d: end-to-end DB → export → load round-trip
    for ALL FOUR renderings. Pinned because a key-name mismatch
    between the bundle emitters (_emit_original_script_list /
    _emit_transliteration_list / _emit_pronunciation_list) and the
    runtime suffix-strip parsers would pass each side's unit tests
    independently but break the round-trip silently."""
    from wyrd.generators.kenning.lexicon import (
        LexiconDB,
        export_meanings,
        init_schema,
        seed_from_meanings,
    )

    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)

    with LexiconDB(db_path) as db:
        db.upsert_source(id="rando-port", title="rando")
        seed_from_meanings(
            db,
            [
                {
                    "modifier_tags": [],
                    "meaning": ["dog"],
                    "modifier_type": None,
                    "words": [{"modern_usage": "Kelev", "hebrew": ["כלב"]}],
                }
            ],
            "rando-port",
        )
        db.conn.execute(
            """UPDATE etymon SET
                 original_script = ?, transliteration = ?,
                 english_shaped = ?, pronunciation_ipa = ?,
                 pronunciation_dialect = ?
               WHERE canonical_form = ? AND language = ?""",
            (
                "כֶּלֶב",
                "kɛ́lɛḇ",
                "kelev",
                "/kalb/",
                "Biblical-Hebrew",
                "כלב",
                "he",
            ),
        )
        db.commit()

        subjects = export_meanings(db, include_rando=True)

    meaning_db, _ = load_meanings(subjects)
    m = meaning_db["Kelev"][0]
    assert m.original_script["hebrew"]["כלב"] == "כֶּלֶב"
    assert m.transliteration["hebrew"]["כלב"] == "kɛ́lɛḇ"
    assert m.english_shaped["hebrew"]["כלב"] == "kelev"
    assert m.pronunciation["hebrew"]["כלב"] == {
        "ipa": "/kalb/",
        "dialect": "Biblical-Hebrew",
    }


# ---------------------------------------------------------------------
# wyrd-lr4 Phase 2: stratum accessor + load round-trip
# ---------------------------------------------------------------------


def test_stratum_defaults_to_empty_dict_for_legacy_meaning():
    """Meaning constructed without stratum kwarg (legacy bundle pre-
    Phase 2) gets an empty dict."""
    m = Meaning(usage="Caer-", tags=[], meanings=[], sources={})
    assert m.stratum == {}


def test_load_meanings_strips_stratum_suffix_into_dedicated_dict():
    """``<lang>_stratum`` JSON arrays land in ``Meaning.stratum`` keyed
    by (lang_field, canonical_form). ``sources`` keeps the language
    form array but does NOT carry the _stratum sibling."""
    data = [
        {
            "modifier_tags": ["habitation"],
            "meaning": ["fort"],
            "modifier_type": "Habitative",
            "words": [
                {
                    "modern_usage": "Caer-",
                    "celtic_mix": ["caer", "din"],
                    "celtic_mix_stratum": [
                        {"form": "caer", "stratum": "latin-loan"},
                        {"form": "din", "stratum": "native-welsh"},
                    ],
                }
            ],
        }
    ]
    meaning_db, _ = load_meanings(data)
    assert "Caer-" in meaning_db
    m = meaning_db["Caer-"][0]
    assert m.sources == {"celtic_mix": ["caer", "din"]}
    assert m.stratum == {"celtic_mix": {"caer": "latin-loan", "din": "native-welsh"}}


def test_load_meanings_round_trips_with_no_stratum_field():
    """Legacy bundle entries (pre-Phase 2 / unclassified langs) load
    cleanly with an empty stratum dict — no KeyError."""
    data = [
        {
            "modifier_tags": [],
            "meaning": ["bridge"],
            "modifier_type": "Topographical",
            "words": [
                {
                    "modern_usage": "Bridg-",
                    "old_english": ["brycg"],
                }
            ],
        }
    ]
    meaning_db, _ = load_meanings(data)
    m = meaning_db["Bridg-"][0]
    assert m.stratum == {}


def test_load_meanings_handles_mixed_bundle_some_words_with_stratum():
    """A bundle where some words carry ``<lang>_stratum`` and others
    don't (the realistic upgrade scenario — one bundle re-emit lifts
    Welsh-classified families while leaving Latin / OE / etc. unchanged)
    loads cleanly. Each Meaning gets only its own stratum dict; no
    cross-contamination between word entries."""
    data = [
        {
            "modifier_tags": ["habitation"],
            "meaning": ["fort"],
            "modifier_type": "Habitative",
            "words": [
                {
                    "modern_usage": "Caer-",
                    "celtic_mix": ["caer"],
                    "celtic_mix_stratum": [{"form": "caer", "stratum": "latin-loan"}],
                },
                {
                    "modern_usage": "Bridg-",
                    "old_english": ["brycg"],
                },
            ],
        }
    ]
    meaning_db, _ = load_meanings(data)
    classified = meaning_db["Caer-"][0]
    assert classified.stratum == {"celtic_mix": {"caer": "latin-loan"}}
    unclassified = meaning_db["Bridg-"][0]
    assert unclassified.stratum == {}


def test_db_to_export_to_load_round_trip_preserves_stratum(tmp_path):
    """End-to-end: seed a fresh DB with a welsh family carrying
    stratum, run ``export_meanings``, then ``load_meanings`` on the
    output — the tag survives all the way to ``Meaning.stratum``.

    Pinned because the exporter and loader were tested independently;
    a field-name mismatch between ``_emit_stratum_list``'s output keys
    and ``_STRATUM_SUFFIX``-stripping would pass both tests in
    isolation but break the round-trip."""
    from wyrd.generators.kenning.lexicon import (
        LexiconDB,
        export_meanings,
        init_schema,
        seed_from_meanings,
    )

    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)

    with LexiconDB(db_path) as db:
        db.upsert_source(id="rando-port", title="rando")
        seed_from_meanings(
            db,
            [
                {
                    "modifier_tags": ["habitation"],
                    "meaning": ["fort"],
                    "modifier_type": "Habitative",
                    "words": [{"modern_usage": "Caer-", "celtic_mix": ["caer"]}],
                }
            ],
            "rando-port",
        )
        # Relabel to language='welsh' so the row mirrors a Phase 1
        # classifier output, then stamp the stratum.
        db.conn.execute(
            "UPDATE etymon SET language = 'welsh', stratum = ? WHERE canonical_form = 'caer'",
            ("latin-loan",),
        )
        db.commit()

        subjects = export_meanings(db, include_rando=True)

    meaning_db, _ = load_meanings(subjects)
    assert "Caer" in meaning_db  # wyrd-aicu.3: export reads the BARE reflex surface (was 'Caer-')
    m = meaning_db["Caer"][0]
    assert m.stratum["celtic_mix"]["caer"] == "latin-loan"


def test_first_nonempty_form_is_the_single_source_for_the_form_scan():
    """``Meaning._first_nonempty_form`` is the centralized "non-empty form" scan
    shared by ``_forms_nonempty`` / ``primary_language`` /
    ``vector_name_select._lemma_ref_for`` (unified so a forms[0]-only copy can't
    drift again, as it did in #788). Returns the FIRST usable form across ALL
    entries (string or ``{form|canonical_form}`` dict), or None when none."""
    f = Meaning._first_nonempty_form
    # No usable form → None.
    assert f(None) is None
    assert f([]) is None
    assert f([""]) is None
    assert f([None]) is None
    assert f([{"form": ""}]) is None
    assert f([{}]) is None
    # First non-empty across all entries (NOT just forms[0]).
    assert f(["wic"]) == "wic"
    assert f(["", "wic"]) == "wic"  # leading empty: scan continues
    assert f([{"form": "wic"}]) == "wic"
    assert f([{"canonical_form": "scir"}]) == "scir"  # form-less → canonical_form
    assert f([{"form": ""}, {"form": "wic"}]) == "wic"
    # form wins over canonical_form when both present.
    assert f([{"form": "a", "canonical_form": "b"}]) == "a"
    # _forms_nonempty is exactly "first non-empty is not None".
    assert Meaning._forms_nonempty(["", "wic"]) is True
    assert Meaning._forms_nonempty([""]) is False
