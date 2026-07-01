"""Tests for the language-quality dashboard (wyrd-wzwa Phase 1).

Covers:
- ``LanguageScorecard`` fields populated correctly from a fixture DB.
- Per-metric isolated tests (corpus depth, inflection, variants, era
  reflex, stratum, citation).
- Bundle-side metrics: word count, tag coverage, missing siblings,
  shared-sibling rendering.
- ``compute_report`` end-to-end on a fixture DB + bundle.
- JSON / markdown formatters round-trip the dataclass cleanly.
- Reference-tag loader: reads from proportions.json, falls back when
  the file is missing or shaped wrong.
- Era-chain endpoint handling: terminus-back / terminus-forward
  languages report n/a rather than 0% for the missing direction.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli
from wyrd.generators.kenning.language_quality import (
    _BUNDLE_LANG_KEY,
    DEFAULT_LANGUAGES,
    ERA_CHAINS,
    FALLBACK_REFERENCE_TAGS,
    LanguageQualityReport,
    LanguageScorecard,
    _bundle_attestation_breakdown,
    _bundle_era_reflex_coverage,
    _bundle_ipa_coverage,
    _bundle_language_word_count,
    _bundle_tag_coverage_for_sibling,
    _bundle_total_words,
    _lexicon_ipa_coverage,
    _tier4_phonology_coverage,
    compute_rando_port_grandfather_audit,
    compute_report,
    compute_scorecard,
    load_reference_tags,
    populate_eligible_etymon_table,
    report_to_json,
    report_to_markdown,
)

# TRANSITIONAL_NONCANONICAL_LANGUAGES is a drift-tripwire internal (emptied at
# wyrd-xdam.4), intentionally NOT re-exported from the package — import direct.
from wyrd.generators.kenning.language_quality.models import TRANSITIONAL_NONCANONICAL_LANGUAGES
from wyrd.generators.kenning.lexicon.controlled_vocab import LANGUAGE_CANONICAL


def _build_fixture_db() -> sqlite3.Connection:
    """In-memory lexicon shaped just enough to exercise every dashboard
    metric. Two languages: 'old-english' (rich) and 'welsh' (sparse)
    so the per-language asymmetry the dashboard surfaces is visible
    in tests."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE source (id TEXT PRIMARY KEY, year INTEGER, region TEXT);
        CREATE TABLE etymon (
            id INTEGER PRIMARY KEY,
            canonical_form TEXT NOT NULL,
            language TEXT NOT NULL,
            modifier_type TEXT,
            position_pref TEXT,
            notes TEXT,
            lemma_id INTEGER REFERENCES etymon(id),
            inflection TEXT,
            merged_into_id INTEGER REFERENCES etymon(id),
            lemma_method TEXT,
            cognate_id INTEGER REFERENCES etymon(id),
            cognate_method TEXT,
            stratum TEXT,
            pronunciation_ipa TEXT
        );
        CREATE TABLE etymon_citation (
            id INTEGER PRIMARY KEY,
            etymon_id INTEGER NOT NULL REFERENCES etymon(id),
            source_id TEXT NOT NULL REFERENCES source(id)
        );
        CREATE TABLE etymon_text_match (
            id INTEGER PRIMARY KEY,
            etymon_id INTEGER NOT NULL REFERENCES etymon(id),
            matched_form TEXT NOT NULL
        );
        CREATE TABLE etymon_descent (
            id INTEGER PRIMARY KEY,
            parent_id INTEGER NOT NULL REFERENCES etymon(id),
            child_id INTEGER NOT NULL REFERENCES etymon(id),
            edge_type TEXT
        );
        CREATE TABLE etymon_period_form (
            id INTEGER PRIMARY KEY,
            etymon_id INTEGER NOT NULL REFERENCES etymon(id),
            form TEXT NOT NULL,
            date_year INTEGER NOT NULL
        );
        CREATE TABLE etymon_tag (
            etymon_id INTEGER NOT NULL REFERENCES etymon(id),
            tag TEXT NOT NULL,
            PRIMARY KEY (etymon_id, tag)
        );

        CREATE VIEW etymon_consensus AS
          SELECT lemma_id, canonical_form, language, witnesses FROM (
            SELECT
              COALESCE(le.id, e.id) AS lemma_id,
              COALESCE(le.canonical_form, e.canonical_form) AS canonical_form,
              e.language,
              COUNT(DISTINCT c.source_id) AS witnesses
            FROM etymon e
            LEFT JOIN etymon le ON le.id = e.lemma_id
            LEFT JOIN etymon_citation c ON c.etymon_id = e.id
            GROUP BY COALESCE(e.lemma_id, e.id)
          );
        """
    )
    # Two sources for OE so witness counts can reach 2.
    conn.executescript(
        """
        INSERT INTO source(id, year) VALUES ('skeat_1901', 1901);
        INSERT INTO source(id, year) VALUES ('mawer_1920', 1920);
        INSERT INTO source(id, year) VALUES ('charles_1992', 1992);
        """
    )
    # OE: 3 lemmas (cot, ham, tun), 2 of which have inflected children.
    # cot has cotum (dative_pl) and cotan (genitive_strong) — 2 inflections,
    # 2 distinct labels. 'tun' has none. 'ham' has cotum-clone for variety
    # (single inflection child).
    conn.executescript(
        """
        INSERT INTO etymon(id, canonical_form, language, stratum, pronunciation_ipa)
          VALUES (1, 'cot', 'old-english', 'native-old-english', '/kot/');
        INSERT INTO etymon(id, canonical_form, language, stratum, pronunciation_ipa)
          VALUES (2, 'ham', 'old-english', 'native-old-english', '/ham/');
        INSERT INTO etymon(id, canonical_form, language, stratum)
          VALUES (3, 'tun', 'old-english', 'native-old-english');
        INSERT INTO etymon(id, canonical_form, language, lemma_id, inflection)
          VALUES (4, 'cotum', 'old-english', 1, 'dative_pl');
        INSERT INTO etymon(id, canonical_form, language, lemma_id, inflection)
          VALUES (5, 'cotan', 'old-english', 1, 'genitive_strong');
        INSERT INTO etymon(id, canonical_form, language, lemma_id, inflection)
          VALUES (6, 'hamum', 'old-english', 2, 'dative_pl');
        """
    )
    # ME children for tier-2-forward (cot → cote in middle-english).
    conn.executescript(
        """
        INSERT INTO etymon(id, canonical_form, language)
          VALUES (10, 'cote', 'middle-english');
        INSERT INTO etymon(id, canonical_form, language)
          VALUES (11, 'tonn', 'middle-english');
        INSERT INTO etymon_descent(parent_id, child_id, edge_type)
          VALUES (1, 10, 'inheritance');
        INSERT INTO etymon_descent(parent_id, child_id, edge_type)
          VALUES (3, 11, 'inheritance');
        """
    )
    # cognate cluster: cot ↔ welsh-side 'cwt' (different language,
    # exercises tier-1). Cited so the wyrd-bfv1 eligible-set restriction
    # keeps cwt visible in welsh metrics — cognates aren't part of the
    # eligibility-via-1-hop-descent rule (which is era progression
    # within a chain, not cross-language cognation), so welsh needs an
    # own-language citation to register on the dashboard.
    conn.executescript(
        """
        INSERT INTO etymon(id, canonical_form, language, cognate_id)
          VALUES (20, 'cwt', 'welsh', 1);
        UPDATE etymon SET cognate_id = 20 WHERE id = 1;
        INSERT INTO etymon_citation(etymon_id, source_id) VALUES (20, 'skeat_1901');
        """
    )
    # period-form for 'tun' (covers tier-3).
    conn.executescript(
        "INSERT INTO etymon_period_form(etymon_id, form, date_year) VALUES (3, 'ton', 1086);"
    )
    # Citations: cot cited by 3 sources (≥3 → promotion-eligible at OE
    # threshold 3); ham cited by 1 (not eligible). tun cited by 2.
    conn.executescript(
        """
        INSERT INTO etymon_citation(etymon_id, source_id) VALUES (1, 'skeat_1901');
        INSERT INTO etymon_citation(etymon_id, source_id) VALUES (1, 'mawer_1920');
        INSERT INTO etymon_citation(etymon_id, source_id) VALUES (1, 'charles_1992');
        INSERT INTO etymon_citation(etymon_id, source_id) VALUES (2, 'skeat_1901');
        INSERT INTO etymon_citation(etymon_id, source_id) VALUES (3, 'skeat_1901');
        INSERT INTO etymon_citation(etymon_id, source_id) VALUES (3, 'mawer_1920');
        """
    )
    # Variant: 'cotum' has matched_form 'cotvm' (≠ canonical) → triggers
    # variant coverage on ID 4 (which is OE).
    conn.executescript(
        "INSERT INTO etymon_text_match(etymon_id, matched_form) VALUES (4, 'cotvm');"
    )
    # Etymon tags: cot tagged architecture; tun tagged topography +
    # agriculture; cwt (welsh) tagged architecture. Exercises lexicon-
    # side semantic coverage (Metric C). 'ham' deliberately untagged so
    # the per-tag distinct-etymon count != total-tagged-etymon count.
    conn.executescript(
        """
        INSERT INTO etymon_tag(etymon_id, tag) VALUES (1, 'architecture');
        INSERT INTO etymon_tag(etymon_id, tag) VALUES (3, 'topography');
        INSERT INTO etymon_tag(etymon_id, tag) VALUES (3, 'agriculture');
        INSERT INTO etymon_tag(etymon_id, tag) VALUES (20, 'architecture');
        """
    )
    # Welsh has just the cognate mate (ID 20) — sparse on every other
    # axis. No citations, no inflections, no variants.
    conn.commit()
    return conn


def _fixture_bundle() -> list[dict]:
    """Tiny bundle exercising both the old_english sibling and the
    celtic_mix shared sibling. One subject tagged 'topography'
    (covers tag coverage), one tagged 'water', one untagged.

    Citation fields exercise the wyrd-lc94 bundle-attestation breakdown:
    cot has scholarly citation (mawer_1920), tun has only rando-port
    (grandfather class), afon has only wiktionary-empirical, the filler
    has no citations at all (uncited)."""
    return [
        {
            "meaning": ["cottage"],
            "modifier_tags": ["architecture"],
            "modifier_type": None,
            "words": [
                {
                    "modern_usage": "cot",
                    "old_english": ["cot"],
                    "old_english_citations": ["mawer_1920", "skeat_1901"],
                }
            ],
        },
        {
            "meaning": ["enclosure"],
            "modifier_tags": ["topography", "agriculture"],
            "modifier_type": None,
            "words": [
                {
                    "modern_usage": "tun",
                    "old_english": ["tun"],
                    "old_english_citations": ["rando-port"],
                    "celtic_mix": ["caer"],
                    "celtic_mix_citations": ["watson_1926"],
                }
            ],
        },
        {
            "meaning": ["river"],
            "modifier_tags": ["water"],
            "modifier_type": None,
            "words": [
                {
                    "modern_usage": "afon",
                    "celtic_mix": ["afon"],
                    "celtic_mix_citations": ["wiktionary-empirical"],
                }
            ],
        },
        {
            "meaning": [],
            "modifier_tags": [],
            "modifier_type": None,
            "words": [{"modern_usage": "filler"}],
        },
    ]


# --- bundle helpers ------------------------------------------------------


def test_bundle_total_words_counts_across_subjects() -> None:
    bundle = _fixture_bundle()
    assert _bundle_total_words(bundle) == 4


def test_bundle_total_words_handles_dict_shape() -> None:
    """Forward-compat: dict-shaped bundles (post-wyrd-c1vq) read the
    'subjects' key. Pinned so the migration doesn't silently fall over."""
    bundle = {"subjects": _fixture_bundle(), "joiners": {}}
    assert _bundle_total_words(bundle) == 4


def test_bundle_language_word_count_returns_zero_for_unknown_sibling() -> None:
    """When a language has no dedicated sibling key (sibling=None), the
    bundle word count is 0 by construction."""
    assert _bundle_language_word_count(_fixture_bundle(), None) == 0


def test_bundle_language_word_count_matches_sibling_presence() -> None:
    bundle = _fixture_bundle()
    # old_english appears in 2 of the 4 word entries.
    assert _bundle_language_word_count(bundle, "old_english") == 2
    # celtic_mix appears in 2 word entries.
    assert _bundle_language_word_count(bundle, "celtic_mix") == 2


def test_bundle_tag_coverage_filters_to_lang_subjects() -> None:
    """Tags only credited if the subject has a word with the sibling AND
    the subject carries the tag in modifier_tags."""
    bundle = _fixture_bundle()
    counts = _bundle_tag_coverage_for_sibling(
        bundle, "old_english", ["architecture", "topography", "water"]
    )
    # cot is tagged architecture; tun is tagged topography; afon is welsh-
    # only so 'water' shouldn't credit old_english.
    assert counts == {"architecture": 1, "topography": 1, "water": 0}


def test_bundle_tag_coverage_returns_zeros_when_sibling_none() -> None:
    """Languages without a bundle sibling get zeros for every tag — pinned
    because a None-sibling shouldn't accidentally credit anything."""
    counts = _bundle_tag_coverage_for_sibling(
        _fixture_bundle(), None, ["architecture", "topography"]
    )
    assert counts == {"architecture": 0, "topography": 0}


# --- wyrd-lc94: bundle attestation breakdown ----------------------------


def test_bundle_attestation_classifies_each_admit_path() -> None:
    """The four admission paths produce four distinct bins. Fixture:
    cot has scholar (mawer_1920), tun has only rando-port → grandfather,
    afon has only wiktionary-empirical → empirical-only, plus celtic_mix
    side which has watson_1926 (scholar) for tun and -empirical for afon."""
    bundle = _fixture_bundle()
    # OE sibling: cot (scholar), tun (rando-port-only) → 1+1=2 subjects.
    oe = _bundle_attestation_breakdown(bundle, "old_english")
    assert oe["total"] == 2
    assert oe["scholar_attested"] == 1  # cot via mawer_1920
    assert oe["rando_only"] == 1  # tun has only rando-port on OE side
    assert oe["empirical_only"] == 0
    assert oe["uncited"] == 0


def test_bundle_attestation_celtic_mix_bins_separately() -> None:
    """Per-language sibling attestation: same subjects can land in
    different bins for different languages. tun's celtic_mix side
    has scholar attestation (watson_1926); afon's celtic_mix side
    has wiktionary-empirical only — separate from how they classify
    on the OE side."""
    bundle = _fixture_bundle()
    cl = _bundle_attestation_breakdown(bundle, "celtic_mix")
    assert cl["total"] == 2
    assert cl["scholar_attested"] == 1  # tun via watson_1926
    assert cl["empirical_only"] == 1  # afon only has wiktionary-empirical
    assert cl["rando_only"] == 0
    assert cl["uncited"] == 0


def test_bundle_attestation_returns_empty_for_none_sibling() -> None:
    """Languages without a bundle sibling get all-zeros — same defensive
    behaviour as _bundle_tag_coverage_for_sibling."""
    counts = _bundle_attestation_breakdown(_fixture_bundle(), None)
    assert counts == {
        "total": 0,
        "scholar_attested": 0,
        "empirical_only": 0,
        "rando_only": 0,
        "uncited": 0,
    }


def test_bundle_attestation_handles_uncited_subjects() -> None:
    """A subject whose language sibling exists but has no _citations
    field bins as 'uncited' (older bundle entries or coverage paths
    that didn't carry attribution)."""
    bundle = [
        {
            "meaning": ["cottage"],
            "modifier_tags": ["architecture"],
            "modifier_type": None,
            "words": [{"modern_usage": "cot", "old_english": ["cot"]}],
        }
    ]
    counts = _bundle_attestation_breakdown(bundle, "old_english")
    assert counts["total"] == 1
    assert counts["uncited"] == 1
    assert counts["scholar_attested"] == 0


def test_bundle_attestation_rando_refs_recovers_rando_only() -> None:
    """wyrd-qkn0: the bundle scrubs rando-port from ``<lang>_citations``, so a
    live rando-only morpheme arrives with no citations and (pre-fix) misbins as
    'uncited'. Passing ``rando_refs`` — the rando-port ledger keyed by
    ``morpheme_id`` — recovers the grandfather flag and reclassifies it as
    rando_only."""
    bundle = [
        {
            "meaning": ["enclosure"],
            "modifier_tags": [],
            "modifier_type": None,
            "words": [
                {
                    "modern_usage": "-ton",
                    "old_english": ["tun"],
                    "morpheme_id": "old-english:tun",
                }
            ],
        }
    ]
    # Without the ledger: the structural blind spot — misbins as uncited.
    no_refs = _bundle_attestation_breakdown(bundle, "old_english")
    assert no_refs["uncited"] == 1
    assert no_refs["rando_only"] == 0
    # With the ledger: correctly rando_only, no longer uncited.
    with_refs = _bundle_attestation_breakdown(bundle, "old_english", frozenset({"old-english:tun"}))
    assert with_refs["rando_only"] == 1
    assert with_refs["uncited"] == 0


def test_compute_scorecard_threads_rando_refs_into_attestation() -> None:
    """wyrd-vwjz: compute_scorecard must forward ``rando_refs`` into the
    bundle-attestation breakdown so language-report's B2 ``bundle_rando_only``
    is populated (not structurally 0, the same blind spot the readiness gate
    fixed in wyrd-qkn0). Same rando-only OE morpheme as the breakdown test
    above, exercised end-to-end through the scorecard."""
    conn = _build_fixture_db()
    bundle = [
        {
            "meaning": ["enclosure"],
            "modifier_tags": [],
            "modifier_type": None,
            "words": [
                {
                    "modern_usage": "-ton",
                    "old_english": ["tun"],
                    "morpheme_id": "old-english:tun",
                }
            ],
        }
    ]
    ref = list(FALLBACK_REFERENCE_TAGS[:5])
    no_refs = compute_scorecard(conn, "old-english", bundle, ref, compute_tier4=False)
    with_refs = compute_scorecard(
        conn,
        "old-english",
        bundle,
        ref,
        compute_tier4=False,
        rando_refs=frozenset({"old-english:tun"}),
    )
    assert no_refs.bundle_rando_only == 0
    assert with_refs.bundle_rando_only == 1


def test_bundle_attestation_rando_refs_scholar_wins() -> None:
    """A rando-ledger morpheme that ALSO carries a scholar citation stays
    scholar_attested — corroborated morphemes are not rando-only. The
    cross-reference recovers the flag but never downgrades a real attestation
    (scholar wins all ties)."""
    bundle = [
        {
            "meaning": ["enclosure"],
            "modifier_tags": [],
            "modifier_type": None,
            "words": [
                {
                    "modern_usage": "-ton",
                    "old_english": ["tun"],
                    "morpheme_id": "old-english:tun",
                    "old_english_citations": ["mawer_1920_northumberland_durham"],
                }
            ],
        }
    ]
    counts = _bundle_attestation_breakdown(bundle, "old_english", frozenset({"old-english:tun"}))
    assert counts["scholar_attested"] == 1
    assert counts["rando_only"] == 0


def test_bundle_attestation_scholar_wins_mixed_citations() -> None:
    """When a subject has BOTH scholar and rando-port citations, scholar
    wins (the priority order in _bundle_attestation_breakdown). Pin so
    a future change doesn't accidentally bin scholar+grandfather as
    'rando_only' (which would understate real attestation)."""
    bundle = [
        {
            "meaning": ["test"],
            "modifier_tags": [],
            "modifier_type": None,
            "words": [
                {
                    "modern_usage": "x",
                    "old_english": ["x"],
                    "old_english_citations": ["rando-port", "skeat_1901"],
                }
            ],
        }
    ]
    counts = _bundle_attestation_breakdown(bundle, "old_english")
    assert counts["scholar_attested"] == 1
    assert counts["rando_only"] == 0


def test_bundle_attestation_scholar_wins_across_words() -> None:
    """Attestation is a SUBJECT-level (not per-word) classification: if
    ANY word in the subject has a scholar citation, the whole subject is
    scholar_attested — even when another word is rando-only. Pins the
    cross-word aggregation + the outer scholar short-circuit in
    _subject_citation_flags (wyrd-8uvi), which single-word fixtures don't
    exercise. Both word orders, to prove a later rando/empirical word
    can't override an earlier scholar (and vice-versa)."""
    for words in (
        # rando-only word first, scholar word second
        [
            {"modern_usage": "a", "old_english": ["a"], "old_english_citations": ["rando-port"]},
            {"modern_usage": "b", "old_english": ["b"], "old_english_citations": ["skeat_1901"]},
        ],
        # scholar word first, rando-only word second (outer break short-circuits)
        [
            {"modern_usage": "b", "old_english": ["b"], "old_english_citations": ["skeat_1901"]},
            {"modern_usage": "a", "old_english": ["a"], "old_english_citations": ["rando-port"]},
        ],
    ):
        bundle = [{"meaning": ["m"], "modifier_tags": [], "modifier_type": None, "words": words}]
        counts = _bundle_attestation_breakdown(bundle, "old_english")
        assert counts["total"] == 1
        assert counts["scholar_attested"] == 1
        assert counts["rando_only"] == 0


def test_bundle_attestation_handles_dict_shape() -> None:
    """Forward-compat: dict-shaped bundles (post-wyrd-c1vq) read the
    'subjects' key. Same coverage as the list-shaped fixture."""
    bundle = {"subjects": _fixture_bundle(), "joiners": {}}
    oe = _bundle_attestation_breakdown(bundle, "old_english")
    assert oe["total"] == 2
    assert oe["scholar_attested"] == 1
    assert oe["rando_only"] == 1


# --- wyrd-dxu2: IPA / pronunciation coverage -----------------------------


def _fixture_bundle_with_ipa() -> list[dict]:
    """Bundle exercising the bundle-side IPA coverage helper. Three
    subjects in old_english:
      - cot has IPA in old_english_pronunciation
      - tun has the language but NO pronunciation entry
      - ham has the language and a pronunciation entry but the entry
        carries no IPA string (defensive: we don't count it)
    """
    return [
        {
            "meaning": ["cottage"],
            "modifier_tags": ["architecture"],
            "modifier_type": None,
            "words": [
                {
                    "modern_usage": "cot",
                    "old_english": ["cot"],
                    "old_english_pronunciation": [{"form": "cot", "ipa": "/kot/"}],
                }
            ],
        },
        {
            "meaning": ["enclosure"],
            "modifier_tags": ["topography"],
            "modifier_type": None,
            "words": [{"modern_usage": "tun", "old_english": ["tun"]}],
        },
        {
            "meaning": ["home"],
            "modifier_tags": ["architecture"],
            "modifier_type": None,
            "words": [
                {
                    "modern_usage": "ham",
                    "old_english": ["ham"],
                    "old_english_pronunciation": [{"form": "ham"}],  # no ipa key
                }
            ],
        },
    ]


def test_lexicon_ipa_coverage_counts_only_non_null() -> None:
    """The DB-side helper counts etymons with non-null pronunciation_ipa.
    Fixture seed: cot + ham have IPA on the OE row, tun does not (3
    lemmas total, 2 with IPA)."""
    conn = _build_fixture_db()
    n = _lexicon_ipa_coverage(conn, "old-english")
    # cot ('/kot/') + ham ('/ham/') seeded; tun left NULL; the inflected
    # children (cotum/cotan/hamum) also have NULL since the seed didn't
    # set IPA on them.
    assert n == 2


def test_bundle_ipa_coverage_counts_subjects_with_non_empty_ipa() -> None:
    """The bundle-side helper counts subjects whose sibling is populated
    AND whose ``<sibling>_pronunciation`` has at least one entry with a
    truthy 'ipa' key. Defensive: entries without an 'ipa' key (or with
    an empty IPA string) don't count, even though the pronunciation
    array exists."""
    bundle = _fixture_bundle_with_ipa()
    # Only 'cot' has a pronunciation entry with a populated 'ipa' key.
    assert _bundle_ipa_coverage(bundle, "old_english") == 1


def test_bundle_ipa_coverage_returns_zero_for_none_sibling() -> None:
    """Languages without a bundle sibling get 0 — same defensive
    behaviour as the other bundle helpers."""
    assert _bundle_ipa_coverage(_fixture_bundle_with_ipa(), None) == 0


def test_bundle_ipa_coverage_handles_dict_shape() -> None:
    """Forward-compat: dict-shaped bundles (post-wyrd-c1vq)."""
    bundle = {"subjects": _fixture_bundle_with_ipa(), "joiners": {}}
    assert _bundle_ipa_coverage(bundle, "old_english") == 1


def _fixture_bundle_with_era_reflexes() -> list[dict]:
    """Bundle with era_reflexes data shaped per the post-wyrd-jbcu schema.
    Three subjects:
    - 'cot' (modern_english sibling) — has a modern-english reflex stop
      via cluster source: counts toward both old-english and modern-english F₂.
    - 'ham' (modern_english sibling) — has a phonology-rule:v1 modern-english
      reflex: counts (wyrd-98cs Tier-4 coverage MUST count just like cluster).
    - 'tun' (modern_english sibling) — has era_reflexes targeting only
      OLD-english, not modern-english: counts for old-english, NOT modern-english.
    """
    return [
        {
            "words": [
                {
                    "modern_english": "cot",
                    "old_english": "cot",
                    "era_reflexes": {
                        "modern-english": [{"form": "cot", "source": "cluster"}],
                        "old-english": [{"form": "cot", "source": "cluster"}],
                    },
                }
            ],
        },
        {
            "words": [
                {
                    "modern_english": "ham",
                    "era_reflexes": {
                        "modern-english": [{"form": "ham", "source": "phonology-rule:v1"}],
                    },
                }
            ],
        },
        {
            "words": [
                {
                    "modern_english": "tun",
                    "era_reflexes": {
                        "old-english": [{"form": "tūn", "source": "cluster"}],
                    },
                }
            ],
        },
    ]


def test_bundle_era_reflex_coverage_counts_target_language_hits() -> None:
    """F₂: count bundle subjects whose sibling is populated AND at least
    one word has a non-empty list under era_reflexes[language].

    Fixture: 3 subjects all have the modern_english sibling. Two have
    a modern-english era_reflexes entry (one cluster, one phonology-rule:v1);
    one targets only old-english. Modern-english F₂ should be 2, not 3."""
    bundle = _fixture_bundle_with_era_reflexes()
    assert _bundle_era_reflex_coverage(bundle, "modern-english", "modern_english") == 2


def test_bundle_era_reflex_coverage_includes_phonology_rule_source() -> None:
    """Pin wyrd-h5it's load-bearing claim: phonology-rule:v1 sourced
    reflexes count toward F₂, not just cluster / descent / period-form.
    Without this, the metric would silently re-create the lex-side
    blindspot (Tier 4 ignored) that motivated F₂."""
    bundle = [
        {
            "words": [
                {
                    "modern_english": "ham",
                    "era_reflexes": {
                        "modern-english": [{"form": "ham", "source": "phonology-rule:v1"}],
                    },
                }
            ],
        },
    ]
    assert _bundle_era_reflex_coverage(bundle, "modern-english", "modern_english") == 1


def test_bundle_era_reflex_coverage_requires_sibling_populated() -> None:
    """Denominator gate: a subject without the language's sibling
    populated doesn't count, even if it carries era_reflexes for the
    target. Matches H's denominator shape."""
    bundle = [
        {
            "words": [
                {
                    # No modern_english key — sibling unpopulated.
                    "era_reflexes": {
                        "modern-english": [{"form": "X", "source": "cluster"}],
                    },
                }
            ],
        },
    ]
    assert _bundle_era_reflex_coverage(bundle, "modern-english", "modern_english") == 0


def test_bundle_era_reflex_coverage_returns_zero_for_none_sibling() -> None:
    """Languages without a bundle sibling get 0 — same defensive
    behaviour as the other bundle-side helpers."""
    bundle = _fixture_bundle_with_era_reflexes()
    assert _bundle_era_reflex_coverage(bundle, "modern-english", None) == 0


def test_bundle_era_reflex_coverage_skips_empty_reflex_lists() -> None:
    """era_reflexes[language] == [] doesn't count — only non-empty lists
    are reflex evidence. Guards against a bundle export bug that wrote
    empty arrays as placeholders."""
    bundle = [
        {
            "words": [
                {
                    "modern_english": "cot",
                    "era_reflexes": {"modern-english": []},
                }
            ],
        },
    ]
    assert _bundle_era_reflex_coverage(bundle, "modern-english", "modern_english") == 0


def test_bundle_era_reflex_coverage_handles_dict_shape() -> None:
    """Forward-compat: dict-shaped bundles (post-wyrd-c1vq)."""
    bundle = {"subjects": _fixture_bundle_with_era_reflexes(), "joiners": {}}
    assert _bundle_era_reflex_coverage(bundle, "modern-english", "modern_english") == 2


def test_bundle_era_reflex_coverage_handles_missing_era_reflexes_key() -> None:
    """Words without the ``era_reflexes`` key entirely (not just empty)
    don't crash — the ``or {}`` defensive read handles missing keys
    the same way the bundle export does for older schema versions."""
    bundle = [
        {
            "words": [
                {"modern_english": "cot"},  # No era_reflexes key at all.
            ],
        },
    ]
    assert _bundle_era_reflex_coverage(bundle, "modern-english", "modern_english") == 0


def test_bundle_era_reflex_coverage_combines_sibling_and_reflex_across_words() -> None:
    """Per-subject (not per-word) semantic: a subject counts when its
    words collectively have BOTH the sibling populated AND a target
    reflex, even if those signals live in different words within the
    same subject.

    This is the intended bundle-export contract — a single subject is
    one Meaning whose words are cluster siblings, and 'this subject
    can render an era stop in language L' is a subject-level claim,
    not a per-word claim. Pin it so a future refactor that tightens
    the gate to 'same word' doesn't silently shift the metric."""
    bundle = [
        {
            "words": [
                # Word 1: has sibling, no reflex.
                {"modern_english": "cot"},
                # Word 2: no sibling, has target reflex.
                {"era_reflexes": {"modern-english": [{"form": "X", "source": "cluster"}]}},
            ],
        },
    ]
    assert _bundle_era_reflex_coverage(bundle, "modern-english", "modern_english") == 1


def test_populate_eligible_etymon_table_includes_cited(_built: None = None) -> None:
    """The eligible set always includes etymons with at least one
    ``etymon_citation`` row — that's the primary inclusion rule
    (rando-port grandfather + scholar place-name attestations).
    Pinned against the fixture DB which seeds citations for cot
    (3), ham (1), tun (2), cwt (1)."""
    conn = _build_fixture_db()
    n = populate_eligible_etymon_table(conn)
    assert n >= 4  # cot, ham, tun, cwt all cited
    ids = {r[0] for r in conn.execute("SELECT id FROM eligible_etymon")}
    assert 1 in ids and 2 in ids and 3 in ids and 20 in ids


def test_populate_eligible_etymon_table_includes_1hop_descent() -> None:
    """A 1-hop descent neighbor of a cited etymon is eligible. The
    fixture has cot (cited) → cote (ME) and tun (cited) → tonn (ME)
    via etymon_descent — both ME entries must be eligible despite
    having no own-language citation, since they're the 'forward form'
    of a cited word."""
    conn = _build_fixture_db()
    populate_eligible_etymon_table(conn)
    ids = {r[0] for r in conn.execute("SELECT id FROM eligible_etymon")}
    assert 10 in ids  # cote (ME, child of cot)
    assert 11 in ids  # tonn (ME, child of tun)


def test_populate_eligible_etymon_table_rolls_lemma_inflections() -> None:
    """Inflected children whose ``lemma_id`` points at an eligible
    etymon ride along. The fixture has cotum / cotan / hamum with
    lemma_id → cot or ham; all three must be eligible."""
    conn = _build_fixture_db()
    populate_eligible_etymon_table(conn)
    ids = {r[0] for r in conn.execute("SELECT id FROM eligible_etymon")}
    assert 4 in ids  # cotum (lemma_id=cot=1)
    assert 5 in ids  # cotan (lemma_id=cot=1)
    assert 6 in ids  # hamum (lemma_id=ham=2)


def test_populate_eligible_etymon_table_is_idempotent() -> None:
    """Re-running drops + repopulates rather than crashing on the
    pre-existing TEMP TABLE — the helper is idempotent so dashboard
    runs that share a connection across multiple compute_report
    invocations don't accumulate stale state."""
    conn = _build_fixture_db()
    first = populate_eligible_etymon_table(conn)
    second = populate_eligible_etymon_table(conn)
    assert first == second
    third = populate_eligible_etymon_table(conn, restrict_to_generator_eligible=False)
    raw_total = conn.execute("SELECT COUNT(*) FROM etymon").fetchone()[0]
    assert third == raw_total  # restrict=False populates with every etymon ID


def test_populate_eligible_etymon_table_source_filter_restricts_seed() -> None:
    """wyrd-5ecx: ``source_filter`` restricts the cited-step seed to
    etymons cited by ONLY that source. Descent + lemma rollup walk
    from the restricted seed but never re-broaden via other sources.

    Fixture has ham (id=2) cited by skeat_1901 ONLY; cot (id=1) cited
    by skeat_1901 + mawer_1920 + charles_1992. Filtering to
    ``source_filter='mawer_1920'`` seeds only cot (and tun, which is
    also cited by mawer_1920) — NOT ham."""
    conn = _build_fixture_db()
    populate_eligible_etymon_table(conn, source_filter="mawer_1920")
    ids = {r[0] for r in conn.execute("SELECT id FROM eligible_etymon")}
    assert 1 in ids  # cot cited by mawer_1920
    assert 3 in ids  # tun cited by mawer_1920
    assert 2 not in ids  # ham cited ONLY by skeat_1901 — excluded
    # Lemma rollup of cot brings its inflected children along.
    assert 4 in ids  # cotum (lemma_id=1)
    assert 5 in ids  # cotan (lemma_id=1)


def test_populate_eligible_etymon_table_source_filter_excludes_fantasy() -> None:
    """When source_filter is set, fantasy/monster/creature-tagged
    etymons are NOT auto-included. The slice view is the slice the
    user asked for, not the slice + the fiction exception."""
    conn = _build_fixture_db()
    # Add a fantasy-tagged etymon that isn't cited by anything.
    conn.executescript(
        """
        INSERT INTO etymon(id, canonical_form, language) VALUES (200, 'orc', 'klingon');
        INSERT INTO etymon_tag(etymon_id, tag) VALUES (200, 'fantasy');
        """
    )
    populate_eligible_etymon_table(conn, source_filter="skeat_1901")
    ids = {r[0] for r in conn.execute("SELECT id FROM eligible_etymon")}
    assert 200 not in ids


def test_populate_eligible_etymon_table_tag_filter_restricts_seed() -> None:
    """wyrd-5ecx: ``tag_filter`` restricts the seed to etymons with
    ONLY that tag (e.g. 'fantasy' alone, excluding 'monster' /
    'creature'). Cited step is skipped — the slice is fiction-only.

    Insert two tagged etymons: 'orc' (fantasy) and 'wyvern' (monster).
    Filter to tag_filter='fantasy' should seed only orc."""
    conn = _build_fixture_db()
    conn.executescript(
        """
        INSERT INTO etymon(id, canonical_form, language) VALUES (300, 'orc', 'klingon');
        INSERT INTO etymon(id, canonical_form, language) VALUES (301, 'wyvern', 'klingon');
        INSERT INTO etymon_tag(etymon_id, tag) VALUES (300, 'fantasy');
        INSERT INTO etymon_tag(etymon_id, tag) VALUES (301, 'monster');
        """
    )
    populate_eligible_etymon_table(conn, tag_filter="fantasy")
    ids = {r[0] for r in conn.execute("SELECT id FROM eligible_etymon")}
    assert 300 in ids  # tagged fantasy
    assert 301 not in ids  # tagged monster, NOT fantasy
    # cited step skipped — fixture's cited rows shouldn't appear
    # unless they end up in fantasy's lemma-rollup / descent context.
    assert 1 not in ids  # cot — cited but not fantasy-related


def test_populate_eligible_etymon_table_both_filters_produces_union() -> None:
    """wyrd-5ecx internal API contract: passing BOTH source_filter
    and tag_filter produces the union of the two slices (cited by
    source ∪ tagged with tag), then descent + lemma rollup walk from
    the combined seed. The CLI rejects this combination, but the
    function supports it for testing flexibility + future composite
    consumers.

    Fixture: cot (id=1) cited by skeat_1901; add an orc tagged
    fantasy. Passing both filters should include BOTH."""
    conn = _build_fixture_db()
    conn.executescript(
        """
        INSERT INTO etymon(id, canonical_form, language) VALUES (400, 'orc', 'klingon');
        INSERT INTO etymon_tag(etymon_id, tag) VALUES (400, 'fantasy');
        """
    )
    populate_eligible_etymon_table(
        conn,
        source_filter="skeat_1901",
        tag_filter="fantasy",
    )
    ids = {r[0] for r in conn.execute("SELECT id FROM eligible_etymon")}
    assert 1 in ids  # cot — from source_filter
    assert 400 in ids  # orc — from tag_filter


def test_populate_eligible_etymon_table_source_filter_walks_descent_from_seed() -> None:
    """wyrd-5ecx promise: descent + lemma rollup STILL apply from the
    restricted seed, so the slice view captures the era-progression
    context of the seeded words — not just the raw citations.

    Fixture: cot (OE, cited by skeat_1901 + mawer_1920 + charles_1992)
    is parent of cote (ME, id=10) via etymon_descent. Filtering to
    ``source_filter='skeat_1901'`` should include the OE seed AND
    its 1-hop ME descent child cote, even though cote isn't itself
    cited by skeat_1901."""
    conn = _build_fixture_db()
    populate_eligible_etymon_table(conn, source_filter="skeat_1901")
    ids = {r[0] for r in conn.execute("SELECT id FROM eligible_etymon")}
    assert 1 in ids  # cot — cited by skeat_1901
    assert 10 in ids  # cote — 1-hop descent child of cot
    assert 11 in ids  # tonn — 1-hop descent child of tun (also cited)


def test_compute_report_propagates_filter_to_dataclass_and_markdown() -> None:
    """wyrd-5ecx: ``compute_report`` stores the active filter in the
    returned ``LanguageQualityReport`` and the markdown formatter
    surfaces it in the header. Operator-visible contract — pin both
    the dataclass roundtrip AND the rendered string."""
    conn = _build_fixture_db()
    bundle = _fixture_bundle()
    report = compute_report(
        conn,
        bundle,
        languages=["old-english"],
        reference_tags=list(FALLBACK_REFERENCE_TAGS[:3]),
        source_filter="skeat_1901",
    )
    assert report.source_filter == "skeat_1901"
    assert report.tag_filter is None
    md = report_to_markdown(report)
    assert "Slice filter: source = `skeat_1901`" in md


def test_compute_report_no_filter_omits_header_line() -> None:
    """When no filter is active, the markdown header MUST NOT emit
    the slice-filter line. Pin the absence so a future refactor that
    always renders the line (with None / 'all') doesn't ship."""
    conn = _build_fixture_db()
    bundle = _fixture_bundle()
    report = compute_report(
        conn,
        bundle,
        languages=["old-english"],
        reference_tags=list(FALLBACK_REFERENCE_TAGS[:3]),
    )
    assert report.source_filter is None
    assert report.tag_filter is None
    md = report_to_markdown(report)
    assert "Slice filter:" not in md


def test_cli_language_report_source_tag_mutually_exclusive(tmp_path: Path) -> None:
    """wyrd-5ecx CLI contract: ``--source X --tag Y`` raises a
    UsageError. The two flags define different seed shapes; the CLI
    rejects the combination so operators don't have to think about
    union semantics. Pass an explicit ``--db`` (defaults to a path
    that doesn't exist on CI) so click's parameter validation
    passes BEFORE reaching the mutual-exclusion check."""
    # Create a minimal SQLite file so --db existence check passes.
    db_path = tmp_path / "lexicon.db"
    sqlite3.connect(db_path).close()

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "lexicon",
            "language-report",
            "--db",
            str(db_path),
            "--source",
            "x",
            "--tag",
            "y",
        ],
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output.lower()


def test_cli_language_report_bundle_without_proportions_does_not_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """wyrd-62cc regression: ``--bundle X`` WITHOUT ``--proportions`` must
    not raise UnboundLocalError. ``runtime_db`` and the
    ``proportions_dict_for_culture`` import were bound only inside the
    ``bundle_path is None`` branch, but the ``proportions_path is None``
    branch referenced them regardless — so passing ``--bundle`` alone
    crashed before reaching ``compute_report``.

    Stub the runtime-DB resolution + report pipeline (the fix is purely
    about variable binding before ``compute_report``, so trivial stubs
    isolate it without a real L4 bundle)."""
    import wyrd.generators.kenning.language_quality as lq
    import wyrd.generators.kenning.runtime.runtime_db as rdb
    import wyrd.generators.kenning.runtime.runtime_db_adapter as rda

    db_path = tmp_path / "lexicon.db"
    sqlite3.connect(db_path).close()
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps({"subjects": [], "joiners": {}}))

    sentinel_db = object()
    monkeypatch.setattr(rdb, "get_runtime_db", lambda: sentinel_db)
    monkeypatch.setattr(rda, "proportions_dict_for_culture", lambda db, culture: {})
    monkeypatch.setattr(lq, "load_reference_tags", lambda source, top_n: {})
    monkeypatch.setattr(lq, "compute_report", lambda *a, **k: [])
    monkeypatch.setattr(lq, "report_to_markdown", lambda *a, **k: "ok")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["lexicon", "language-report", "--db", str(db_path), "--bundle", str(bundle_path)],
    )
    assert not isinstance(result.exception, UnboundLocalError), (
        f"--bundle without --proportions regressed to UnboundLocalError: {result.exception!r}"
    )
    assert result.exit_code == 0, f"{result.output}\n{result.exception!r}"


def test_populate_eligible_etymon_table_handles_minimal_schema() -> None:
    """Defensive: minimal test fixtures may lack etymon_tag /
    etymon_descent / etymon_citation. The helper skips those branches
    gracefully rather than failing with 'no such table'."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE etymon (
            id INTEGER PRIMARY KEY,
            canonical_form TEXT NOT NULL,
            language TEXT NOT NULL,
            lemma_id INTEGER,
            merged_into_id INTEGER
        );
        INSERT INTO etymon (canonical_form, language) VALUES ('alpha', 'x');
        """
    )
    n = populate_eligible_etymon_table(conn)
    # No citation / descent / tag tables → eligible set is empty.
    assert n == 0
    # Restrict=False still works (populates with every etymon ID).
    n_unrestricted = populate_eligible_etymon_table(conn, restrict_to_generator_eligible=False)
    assert n_unrestricted == 1


def _seed_tier4_fixture_db() -> sqlite3.Connection:
    """In-memory etymon table seeded with three modern-english forms
    chosen so the wyrd-98cs phonology chain produces a transformed
    Tier-4 reflex (the rules walk back through early-modern-english /
    middle-english / old-english cells).

    The actual phonology_rules library is imported live so any cell
    re-tuning shows up as a test failure here — pinning the count
    would over-couple to internal rule probabilities.
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE etymon (
            id INTEGER PRIMARY KEY,
            canonical_form TEXT NOT NULL,
            language TEXT NOT NULL,
            lemma_id INTEGER REFERENCES etymon(id),
            merged_into_id INTEGER REFERENCES etymon(id)
        );
        CREATE TABLE etymon_citation (
            id INTEGER PRIMARY KEY,
            etymon_id INTEGER NOT NULL,
            source_id TEXT NOT NULL
        );
        """
    )
    rows = [
        ("house", "modern-english"),
        ("ham", "modern-english"),
        ("cot", "modern-english"),
        ("liorth", "modern-welsh"),  # phonology-rule-eligible welsh form
        # A non-chain-eligible language — Tier-4 must return -1.
        ("ríg", "old-irish"),
        # An empty canonical_form — defensive: must not crash.
        ("", "modern-english"),
    ]
    conn.executemany(
        "INSERT INTO etymon (canonical_form, language) VALUES (?, ?)",
        rows,
    )
    # wyrd-bfv1: cite every fixture row so the eligible-set filter
    # in _tier4_phonology_coverage (which restricts the walked
    # population to generator-eligible etymons) includes them all.
    # Without these citations the eligible set is empty and the walk
    # processes zero etymons.
    conn.executemany(
        "INSERT INTO etymon_citation (etymon_id, source_id) VALUES (?, 'fixture')",
        [(i,) for i in range(1, len(rows) + 1)],
    )
    conn.commit()
    return conn


def test_form_fires_tier4_firing_and_non_firing() -> None:
    """The per-etymon Tier-4 firing test extracted from
    _tier4_phonology_coverage, called directly. A form a phonology rule transforms
    for some target era fires; a form no rule touches doesn't; a falsy form never
    fires; an empty target list never fires (any() over nothing is False)."""
    from wyrd.generators.kenning.language_quality.audits import _form_fires_tier4
    from wyrd.generators.kenning.registers.phonology_rules import chain_for

    resolved, _family, chain = chain_for("modern-english")
    i_self = chain.index(resolved)
    targets = sorted(
        (s for s in chain if s != resolved), key=lambda s: abs(chain.index(s) - i_self)
    )

    assert _form_fires_tier4("house", resolved, targets) is True  # transformed across eras
    assert _form_fires_tier4("zzzz", resolved, targets) is False  # no rule touches it
    assert _form_fires_tier4("", resolved, targets) is False  # falsy form never fires
    assert _form_fires_tier4("house", resolved, []) is False  # no targets -> never fires


def test_tier4_phonology_coverage_returns_negative_one_for_no_chain() -> None:
    """Languages without a registered phonology chain (Goidelic, Norse,
    Romance, etc.) return -1 — sentinel for 'no rules to apply', which
    the markdown formatter renders as 'n/a' to distinguish it from
    '0 etymons fire'."""
    conn = _seed_tier4_fixture_db()
    assert _tier4_phonology_coverage(conn, "old-irish") == -1


def test_tier4_phonology_coverage_returns_zero_for_empty_language() -> None:
    """A chain-eligible language with no etymons in the DB returns 0
    (the chain exists, the population is just empty)."""
    conn = _seed_tier4_fixture_db()
    # middle-english is phonology-chain-eligible but unpopulated here.
    assert _tier4_phonology_coverage(conn, "middle-english") == 0


def test_tier4_phonology_coverage_counts_firing_etymons() -> None:
    """Etymons whose canonical_form produces a transformed Tier-4
    reflex (via phonology_rules.rule_form) for at least one chain stop
    count. Tests run against the live phonology_rules cells (wyrd-98cs
    et al.), so the count reflects real rule firing rather than a
    fixed expected value — empty canonical_form gets skipped.

    Bound the assertion to a range rather than a fixed number so cell-
    rule re-tuning doesn't break this test for the wrong reason; the
    invariant is 'at least one of {house, ham, cot} fires Tier-4 for
    at least one other English-chain era'."""
    conn = _seed_tier4_fixture_db()
    count = _tier4_phonology_coverage(conn, "modern-english")
    # Three real modern-english forms in the fixture; empirically the
    # rules fire for every form that has letters that the cells
    # transform (~99% in production). One empty-form row must not
    # crash and must not count.
    assert 0 <= count <= 3
    # Skip the strict lower-bound assertion if the rules library has
    # been disabled — fail loud if rules are wired but nothing fires.
    from wyrd.generators.kenning.registers.phonology_rules import rule_form

    if rule_form("house", "modern-english", "old-english") is not None:
        assert count >= 1, "phonology rules fire on 'house' but T4 count says 0"


def test_compute_scorecard_compute_tier4_false_skips_walk() -> None:
    """``compute_scorecard(..., compute_tier4=False)`` skips the Tier-4
    walk and returns ``-1`` — same rendering as a non-chain language
    but reached without paying the ~3 minute modern-english cost.
    Pin both: count is -1, AND the optional progress callback isn't
    invoked under the skip path."""
    conn = _build_fixture_db()
    bundle: list[dict] = []
    callbacks: list[tuple[int, int]] = []

    def cb(done: int, total: int) -> None:
        callbacks.append((done, total))

    # old-english is chain-eligible in phonology_rules.FAMILY_CHAINS, so
    # compute_tier4=True WOULD fire. compute_tier4=False must skip.
    card = compute_scorecard(
        conn,
        "old-english",
        bundle,
        list(FALLBACK_REFERENCE_TAGS[:3]),
        compute_tier4=False,
        tier4_progress_callback=cb,
    )
    assert card.tier4_phonology_count == -1
    assert callbacks == [], "callback must not fire under compute_tier4=False"


def test_compute_report_per_language_callback_threads_correct_lang() -> None:
    """The per-lang callback closure in compute_report wraps the
    operator's callback so each language's progress emissions carry
    the right language tag. Classic late-binding-closure-over-loop-var
    bug surface — pin that the closure uses the active language at
    each iteration, not the last value of ``lang`` after the loop.

    Pin: with a multi-language fixture, the callback records
    ``(language, done, total)`` tuples; verify each language emits
    at least one tick AND the language tag matches the active
    iteration (no cross-talk between languages)."""
    conn = _build_fixture_db()
    bundle: list[dict] = []
    received: list[tuple[str, int, int]] = []

    def cb(lang: str, done: int, total: int) -> None:
        received.append((lang, done, total))

    # Two chain-eligible languages in the fixture: old-english (rich,
    # 7 rows) + welsh (sparse, 2 rows).
    compute_report(
        conn,
        bundle,
        languages=["old-english", "welsh"],
        reference_tags=list(FALLBACK_REFERENCE_TAGS[:3]),
        tier4_progress_callback=cb,
    )
    languages_seen = {lang for lang, _, _ in received}
    assert "old-english" in languages_seen
    assert "welsh" in languages_seen
    # Specifically guard the late-binding bug: every callback for
    # 'old-english' carries 'old-english', not 'welsh'.
    oe_calls = [t for t in received if t[0] == "old-english"]
    welsh_calls = [t for t in received if t[0] == "welsh"]
    assert oe_calls, "old-english should have at least one callback"
    assert welsh_calls, "welsh should have at least one callback"


def test_tier4_phonology_coverage_progress_callback_fires() -> None:
    """The optional progress_callback receives (done, total) tuples.
    Pin the callback contract — tiny populations should still receive
    at least one tick (the final emission)."""
    conn = _seed_tier4_fixture_db()
    callbacks: list[tuple[int, int]] = []

    def cb(done: int, total: int) -> None:
        callbacks.append((done, total))

    _tier4_phonology_coverage(conn, "modern-english", progress_callback=cb)
    assert callbacks, "expected at least one progress callback for non-empty population"
    # Last callback reports completion.
    assert callbacks[-1][0] == callbacks[-1][1] == 4  # 4 modern-english rows in fixture


def test_summary_table_era_cell_renders_na_when_no_bundle_subjects() -> None:
    """Markdown formatter renders the bundle-side half of the F cell
    as 'n/a' when ``bundle_attestation_total == 0`` (the language has
    no bundle representation). Mirrors H's n/a branch — keeps
    'absent' visually distinct from '0% covered'."""
    scorecard = LanguageScorecard(
        language="latin",
        total_etymons=100,
        total_lemmas=100,
        # wyrd-bfv1: eligible denominator drives the percentage. Set
        # equal to total to mirror a corpus where every etymon is
        # generator-eligible (so 40 / 100 → 40.0%).
        eligible_etymons=100,
        eligible_lemmas=100,
        promotion_threshold=2,
        promotion_eligible=0,
        avg_witnesses=0.0,
        source_count=0,
        bundle_sibling=None,
        bundle_word_count=0,
        bundle_attestation_total=0,
        any_reflex_count=40,
    )
    report = LanguageQualityReport(
        schema_version="1.0",
        generated_at="2026-05-11T00:00:00Z",
        reference_tags=list(FALLBACK_REFERENCE_TAGS[:3]),
        bundle_total_words=100,
        languages=[scorecard],
    )
    md = report_to_markdown(report)
    # The summary table's era cell is 'lex_pct / bundle_pct'; with
    # zero bundle subjects, bundle side renders 'n/a'.
    assert "40.0% / n/a" in md
    # The per-language detail F₂ line must also render 'n/a' when
    # bundle_attestation_total==0 — consistent with the summary table,
    # keeps 'no bundle representation' visually distinct from
    # '0% covered'. Otherwise _format_pct returns '0.0%' for zero
    # denominator, which is misleading.
    assert "Bundle subjects with reflex stop targeting this language: 0/0 (n/a)" in md
    # The other zero-denominator detail sections must also render their 'n/a'
    # branch (this fixture has bundle_attestation_total=0, bundle_sibling=None,
    # and rando_total_cited_families=0). Pins the n/a wording each section owns.
    assert "B₂. Bundle attestation:** n/a" in md
    assert "C₂. Bundle tag visibility:** n/a" in md
    assert "bundle n/a — no bundle subjects" in md  # H. Pronunciation coverage
    assert "K. Rando-port grandfather audit:** n/a" in md


def test_scorecard_inheritance_warning_pin() -> None:
    """When lexicon-side IPA is 0 and bundle-side IPA is non-zero, the
    rendered markdown surfaces an explicit ⚠ warning so a reader can
    tell 'we have IPA' apart from 'we're showing cluster mates' IPA'.
    Pin: a fixture matching the modern-english post-wyrd-69s5 case
    (DB has 0 IPA, bundle inherits from cluster mates) renders the
    warning."""
    scorecard = LanguageScorecard(
        language="modern-english",
        total_etymons=21175,
        total_lemmas=21175,
        promotion_threshold=2,
        promotion_eligible=0,
        avg_witnesses=0.0,
        source_count=0,
        bundle_sibling="modern_english",
        bundle_word_count=5495,
        reference_tags=["a", "b"],
        bundle_attestation_total=5495,
        bundle_scholar_attested=2000,
        lexicon_etymons_with_ipa=0,
        lexicon_ipa_density=0.0,
        bundle_subjects_with_ipa=1083,
        bundle_ipa_rate=0.197,
    )
    report = LanguageQualityReport(
        schema_version="1.0",
        generated_at="2026-05-09T00:00:00Z",
        reference_tags=["a", "b"],
        bundle_total_words=18767,
        languages=[scorecard],
    )
    md = report_to_markdown(report)
    assert "Bundle IPA is INHERITED from cluster mates" in md


# --- wyrd-vn09: rando-port grandfather audit -----------------------------


def _seed_grandfather_fixture_db() -> sqlite3.Connection:
    """Tiny in-memory DB that exercises the three classification paths
    of compute_rando_port_grandfather_audit:
      - pure-grandfather family (etymon 100, rando-port only)
      - mixed-grandfather family (etymon 101, rando-port + scholar)
      - no-grandfather family (etymon 102, scholar only — should not
        count toward the audit's grandfather buckets but DOES count
        toward total_families since it's citation-bearing)
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE source (id TEXT PRIMARY KEY);
        INSERT INTO source(id) VALUES ('rando-port'), ('skeat_1901');
        CREATE TABLE etymon (
            id INTEGER PRIMARY KEY,
            canonical_form TEXT NOT NULL,
            language TEXT NOT NULL,
            lemma_id INTEGER,
            merged_into_id INTEGER
        );
        CREATE TABLE etymon_citation (
            id INTEGER PRIMARY KEY,
            etymon_id INTEGER NOT NULL,
            source_id TEXT NOT NULL
        );
        INSERT INTO etymon(id, canonical_form, language) VALUES (100, 'aecern', 'old-english');
        INSERT INTO etymon(id, canonical_form, language) VALUES (101, 'cot', 'old-english');
        INSERT INTO etymon(id, canonical_form, language) VALUES (102, 'ham', 'old-english');
        INSERT INTO etymon_citation(etymon_id, source_id) VALUES (100, 'rando-port');
        INSERT INTO etymon_citation(etymon_id, source_id) VALUES (101, 'rando-port');
        INSERT INTO etymon_citation(etymon_id, source_id) VALUES (101, 'skeat_1901');
        INSERT INTO etymon_citation(etymon_id, source_id) VALUES (102, 'skeat_1901');
        """
    )
    return conn


def test_grandfather_audit_classifies_three_paths() -> None:
    """Pure-grandfather, mixed-grandfather, and no-grandfather families
    each land in the right bucket. Pin: 1 pure, 1 mixed, total_families
    = 3 (all three are citation-bearing, including the scholar-only
    family which counts toward the denominator but neither grandfather
    bucket)."""
    conn = _seed_grandfather_fixture_db()
    audit = compute_rando_port_grandfather_audit(conn)
    oe = audit["old-english"]
    assert oe["pure_grandfather"] == 1
    assert oe["mixed_grandfather"] == 1
    assert oe["total_families"] == 3
    # wyrd-g7n6: the one mixed family (etymon 101: rando-port + skeat) has exactly
    # 1 corroborator, so it lands in corr_1; corr_2 / corr_3plus are empty.
    assert oe["corr_1"] == 1
    assert oe["corr_2"] == 0
    assert oe["corr_3plus"] == 0


def test_grandfather_audit_corroborator_buckets() -> None:
    """wyrd-g7n6: rando+corroborated families bin by corroborator count (distinct
    non-rando-port sources): 0 → pure (rando-only), 1 → corr_1, 2 → corr_2, ≥3 →
    corr_3plus. pure + corr_1 + corr_2 + corr_3plus == total cited grandfather-touching
    families; corr_1+corr_2+corr_3plus == mixed."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE source (id TEXT PRIMARY KEY);
        INSERT INTO source(id) VALUES
            ('rando-port'),('skeat_1901'),('mawer_1920'),('ekwall_1936'),('joyce_1875');
        CREATE TABLE etymon (
            id INTEGER PRIMARY KEY, canonical_form TEXT NOT NULL, language TEXT NOT NULL,
            lemma_id INTEGER, merged_into_id INTEGER
        );
        CREATE TABLE etymon_citation (
            id INTEGER PRIMARY KEY, etymon_id INTEGER NOT NULL, source_id TEXT NOT NULL
        );
        -- corr_0 (rando-only)
        INSERT INTO etymon(id, canonical_form, language) VALUES (200, 'a', 'old-english');
        INSERT INTO etymon_citation(etymon_id, source_id) VALUES (200, 'rando-port');
        -- corr_1 (rando + 1)
        INSERT INTO etymon(id, canonical_form, language) VALUES (201, 'b', 'old-english');
        INSERT INTO etymon_citation(etymon_id, source_id) VALUES (201, 'rando-port'),(201,'skeat_1901');
        -- corr_2 (rando + 2)
        INSERT INTO etymon(id, canonical_form, language) VALUES (202, 'c', 'old-english');
        INSERT INTO etymon_citation(etymon_id, source_id) VALUES
            (202, 'rando-port'),(202,'skeat_1901'),(202,'mawer_1920');
        -- corr_3plus (rando + 3)
        INSERT INTO etymon(id, canonical_form, language) VALUES (203, 'd', 'old-english');
        INSERT INTO etymon_citation(etymon_id, source_id) VALUES
            (203, 'rando-port'),(203,'skeat_1901'),(203,'mawer_1920'),(203,'ekwall_1936');
        """
    )
    oe = compute_rando_port_grandfather_audit(conn)["old-english"]
    assert oe["pure_grandfather"] == 1  # corr_0
    assert oe["corr_1"] == 1
    assert oe["corr_2"] == 1
    assert oe["corr_3plus"] == 1
    assert oe["mixed_grandfather"] == 3  # corr_1 + corr_2 + corr_3plus
    assert oe["total_families"] == 4


def test_grandfather_audit_excludes_uncited_families() -> None:
    """A family with NO citations anywhere isn't in the audit at all —
    uncited rows are reported by the citation-depth metric (I), not by
    the grandfather audit which is specifically about admission paths."""
    conn = _seed_grandfather_fixture_db()
    conn.execute(
        "INSERT INTO etymon(id, canonical_form, language) VALUES (200, 'uncited', 'old-english')"
    )
    conn.commit()
    audit = compute_rando_port_grandfather_audit(conn)
    # Same counts as before — the uncited row contributes nothing.
    oe = audit["old-english"]
    assert oe["total_families"] == 3


def test_grandfather_audit_follows_family_rollup() -> None:
    """Lemma-id and merged_into_id rollup matches lexicon.build_family_rollup:
    a family is rolled up to its root, and citations from any member count
    toward the family's classification. Pin: an inflected child of a pure-
    grandfather lemma stays in the pure-grandfather bucket, NOT a separate
    family."""
    conn = _seed_grandfather_fixture_db()
    # Add an inflected child of etymon 100 (pure-grandfather lemma)
    # plus a scholar citation on the CHILD (the family becomes mixed
    # because the rollup unions all members' sources).
    conn.execute(
        "INSERT INTO etymon(id, canonical_form, language, lemma_id) "
        "VALUES (110, 'aecernas', 'old-english', 100)"
    )
    conn.execute("INSERT INTO etymon_citation(etymon_id, source_id) VALUES (110, 'skeat_1901')")
    conn.commit()
    audit = compute_rando_port_grandfather_audit(conn)
    oe = audit["old-english"]
    # Family 100 was 'pure'; the new child's scholar citation rolls up,
    # so the family is now 'mixed'. Pure count drops by 1, mixed
    # increases by 1; total_families unchanged (110 rolls up to 100).
    assert oe["pure_grandfather"] == 0
    assert oe["mixed_grandfather"] == 2
    assert oe["total_families"] == 3


def test_grandfather_audit_skips_family_with_no_language_root() -> None:
    """A citation-bearing family whose root id has no etymon row (a
    dangling lemma_id — the schema doesn't FK-enforce it) is excluded
    from the audit: lang_by_id.get(root) is None so it counts toward no
    bucket and no total_families. Pins the no-language skip branch the
    C901 extraction owns in _classify_grandfather_families (wyrd-8uvi)."""
    conn = _seed_grandfather_fixture_db()
    # etymon 300 points its lemma_id at a non-existent root (9999); the
    # family rolls up to id 9999, which has no row → no language.
    conn.execute(
        "INSERT INTO etymon(id, canonical_form, language, lemma_id) "
        "VALUES (300, 'orphan', 'old-english', 9999)"
    )
    conn.execute("INSERT INTO etymon_citation(etymon_id, source_id) VALUES (300, 'rando-port')")
    conn.commit()
    audit = compute_rando_port_grandfather_audit(conn)
    # The dangling-root family is excluded — OE counts are unchanged from
    # the 3-family baseline, and no spurious bucket appears for it.
    oe = audit["old-english"]
    assert oe["total_families"] == 3
    assert oe["pure_grandfather"] == 1
    assert oe["mixed_grandfather"] == 1


# --- per-language scorecard ----------------------------------------------


def test_scorecard_old_english_full_metrics() -> None:
    """OE has the rich path through every metric:
    - 6 etymons (3 lemmas + 3 inflected children)
    - 2 lemmas with inflection children (cot, ham), 2 distinct labels
      (dative_pl, genitive_strong) — wait, 3 labels actually if hamum
      shares dative_pl with cotum: distinct count is 2 (dative_pl,
      genitive_strong).
    - 1 etymon with variant (cotum)
    - 1 cognate mate (cot ↔ cwt) — tier-1 = 1
    - 2 OE→ME descent edges (cot→cote, tun→tonn) — tier-2-fwd = 2
    - 0 backward edges — terminus-back per chain
    - 1 period-form (tun→ton) — tier-3 = 1
    - 3 citations on cot, 1 on ham, 2 on tun — promotion-eligible at
      threshold 3 = 1 (just cot's lemma family)."""
    conn = _build_fixture_db()
    bundle = _fixture_bundle()
    card = compute_scorecard(
        conn,
        "old-english",
        bundle,
        list(FALLBACK_REFERENCE_TAGS[:5]),
    )
    assert card.language == "old-english"
    assert card.total_etymons == 6  # cot, ham, tun + 3 inflected children
    assert card.total_lemmas == 3  # cot, ham, tun (inflected ones excluded)
    # wyrd-fssn: OE preset is ≥2 (was ≥3). cot=3 + tun=2 both promote
    # under the uniform-≥2 policy; ham=1 stays under-threshold.
    assert card.promotion_threshold == 2
    assert card.promotion_eligible == 2  # cot + tun cross ≥2 witnesses
    # wyrd-6n2x: eligible-pool avg over the 3 OE lemmas = (3+1+2)/3 = 2.0.
    # Pinning here catches future regressions on the canonical fixture.
    assert card.avg_witnesses == 2.0
    # bundle: old_english sibling, two words.
    assert card.bundle_sibling == "old_english"
    assert card.bundle_word_count == 2
    assert card.bundle_shared_with == []
    # inflection: cot + ham have linked children; tun doesn't.
    assert card.lemmas_with_inflections == 2
    assert card.distinct_inflection_labels == 2
    # variants: only cotum has a matched_form ≠ canonical.
    assert card.etymons_with_variants == 1
    # era reflex
    assert card.tier1_cognate_count == 1  # cot ↔ cwt
    assert card.tier2_forward_count == 2  # cot→cote, tun→tonn
    assert card.tier2_back_count == 0  # OE is terminus-back
    assert card.tier3_period_form_count == 1  # tun→ton
    # cot has cognate + descent + (no period form); tun has descent +
    # period-form; ham has no reflex of any kind. 'cotum'/'cotan'/'hamum'
    # are inflected children and don't carry any reflex on their own —
    # so the union count is 2 (cot, tun).
    assert card.any_reflex_count == 2
    assert card.chain_known is True
    assert card.chain_older == []  # terminus-back
    assert card.chain_younger == ["middle-english"]
    # stratum
    assert card.stratum_classified == 3  # all 3 lemmas classified
    # citations
    assert card.etymons_with_citations == 3  # cot, ham, tun all cited
    assert card.avg_citations > 1.5  # 3+1+2=6 across 3 etymons
    # lexicon-side tag coverage: cot tagged architecture, tun tagged
    # topography + agriculture. ham untagged. So 2 distinct tagged
    # etymons (cot, tun). Top-5 reference set is architecture,
    # topography, social, descriptive, plant — so architecture (cot)
    # and topography (tun) both hit; agriculture is in the lexicon
    # but NOT in top-5 → not credited.
    assert card.lexicon_tagged_etymons == 2  # cot, tun
    assert card.lexicon_tag_coverage["architecture"] == 1
    assert card.lexicon_tag_coverage["topography"] == 1
    assert card.lexicon_tag_coverage["plant"] == 0  # in top-5 but no matches
    assert "agriculture" not in card.lexicon_tag_coverage  # not in top-5
    # 2 of 5 top-5 tags hit (architecture, topography).
    assert abs(card.lexicon_tag_hit_rate - 0.4) < 1e-6
    # bundle-side tag visibility: OE has 'old_english' sibling; subjects
    # tagged architecture (cot subject) and topography (tun subject)
    # both have an old_english word. So 2 of 5 reference tags hit.
    assert abs(card.bundle_tag_hit_rate - 0.4) < 1e-6


def test_scorecard_avg_witnesses_filters_to_eligible_pool() -> None:
    """wyrd-6n2x regression for ``avg_witnesses``: the metric must
    divide by the eligible-lemma count, not the raw etymon_consensus
    row count. Pre-fix the metric collapsed to ~0.0 for languages
    with large ineligible tails (modern-english: 409 total witnesses
    / 1.26M raw rows → 0.0003 → rounded to 0.0). Surrounding metrics
    (eligible_etymons, promotion_eligible) all use the eligible-pool
    scope; this one was incoherent against them.

    Fixture: inject an ineligible OE lemma (`pollen`, 0 witnesses)
    and EXCLUDE it from ``eligible_etymon``. Pre-fix
    ``avg_witnesses`` = (3+1+2+0)/4 = 1.5; post-fix = (3+1+2)/3 = 2.0.
    The sibling
    :func:`test_scorecard_promotion_eligible_filters_to_eligible_pool`
    covers the matching tightening of ``promotion_eligible`` against
    a cited-but-ineligible row (which this fixture can't, because
    pollen has 0 witnesses)."""
    conn = _build_fixture_db()
    bundle = _fixture_bundle()
    conn.executescript(
        """
        INSERT INTO etymon(id, canonical_form, language)
          VALUES (300, 'pollen', 'old-english');
        """
    )
    populate_eligible_etymon_table(conn)
    conn.execute("DELETE FROM eligible_etymon WHERE id = 300")
    conn.commit()
    assert conn.execute("SELECT 1 FROM eligible_etymon WHERE id = 300").fetchone() is None

    card = compute_scorecard(
        conn,
        "old-english",
        bundle,
        list(FALLBACK_REFERENCE_TAGS[:5]),
    )
    assert card.avg_witnesses == 2.0, (
        f"avg_witnesses should average over the eligible-lemma pool "
        f"(2.0), not the raw etymon_consensus rows (1.5 with pollen "
        f"in pool); got {card.avg_witnesses}"
    )


def test_scorecard_promotion_eligible_filters_to_eligible_pool() -> None:
    """wyrd-6n2x regression for ``promotion_eligible``: the count
    must reject an ineligible row even if it has enough citations to
    cross the threshold. The fix's SQL JOIN tightens both
    ``avg_witnesses`` AND ``promotion_eligible`` — this test pins
    the latter half against a scenario the
    ``test_scorecard_avg_witnesses_*`` fixture can't exercise (its
    ineligible row has 0 witnesses, so it can't cross any threshold
    regardless of the JOIN).

    Fixture: inject an ineligible OE lemma `pollen` cited by all
    three sources (witnesses = 3, would cross OE's ≥3 threshold).
    Pre-fix ``promotion_eligible`` = 2 (cot + pollen); post-fix = 1
    (only cot — pollen is ineligible)."""
    conn = _build_fixture_db()
    bundle = _fixture_bundle()
    conn.executescript(
        """
        INSERT INTO etymon(id, canonical_form, language)
          VALUES (300, 'pollen', 'old-english');
        INSERT INTO etymon_citation(etymon_id, source_id) VALUES (300, 'skeat_1901');
        INSERT INTO etymon_citation(etymon_id, source_id) VALUES (300, 'mawer_1920');
        INSERT INTO etymon_citation(etymon_id, source_id) VALUES (300, 'charles_1992');
        """
    )
    populate_eligible_etymon_table(conn)
    conn.execute("DELETE FROM eligible_etymon WHERE id = 300")
    conn.commit()

    card = compute_scorecard(
        conn,
        "old-english",
        bundle,
        list(FALLBACK_REFERENCE_TAGS[:5]),
    )
    # wyrd-fssn: OE threshold relaxed to ≥2; cot=3 + tun=2 promote,
    # pollen still excluded (would have promoted at 3 pre-fix on the
    # ineligible path).
    assert card.promotion_eligible == 2, (
        f"promotion_eligible should reject ineligible pollen (3 "
        f"citations would cross ≥2 threshold pre-fix); got "
        f"{card.promotion_eligible}"
    )


def test_scorecard_witness_rollup_includes_lemma_descendants() -> None:
    """wyrd-6n2x: the per-lemma witness count must roll up citations
    from both edges the ``etymon_consensus`` view walks via
    ``COALESCE(e.merged_into_id, e.lemma_id)``: inflection children
    (``etymon.lemma_id = head.id``) AND OCR-merge / bridge losers
    (``etymon.merged_into_id = head.id``). Pre-fix the lemma-head
    query only counted the head's own citations, dropping both
    rollup branches — a regression hidden by the baseline fixture
    (no inflection or merge-loser has its own citations there).

    Fixture extension:
      - 2 citations on ``hamum`` (id=6, lemma_id=2 → ham) exercise
        the inflection-child branch.
      - A new merge-loser ``caut`` (id=400, merged_into_id=1 → cot)
        with 1 citation exercises the merge-loser branch. Loser is
        excluded from ``eligible_etymon`` (D22: merge losers aren't
        generator-eligible) so it doesn't appear as its own
        scorecard row.

    Expected rollup post-fix:
      - cot: 3 own (skeat/mawer/charles) + 1 loser (charles already
        in set) → still 3 distinct sources, promotion-eligible ≥3
      - ham: 1 own (skeat) + 2 inflection-child (mawer/charles) → 3
        distinct sources, promotion-eligible ≥3
      - tun: 2 own (skeat/mawer) → 2 distinct sources
      → avg = (3+3+2)/3 = 2.667, promotion_eligible = 2."""
    conn = _build_fixture_db()
    bundle = _fixture_bundle()
    conn.executescript(
        """
        INSERT INTO etymon_citation(etymon_id, source_id) VALUES (6, 'mawer_1920');
        INSERT INTO etymon_citation(etymon_id, source_id) VALUES (6, 'charles_1992');
        INSERT INTO etymon(id, canonical_form, language, merged_into_id)
          VALUES (400, 'caut', 'old-english', 1);
        INSERT INTO etymon_citation(etymon_id, source_id) VALUES (400, 'charles_1992');
        """
    )
    populate_eligible_etymon_table(conn)
    conn.execute("DELETE FROM eligible_etymon WHERE id = 400")
    conn.commit()

    card = compute_scorecard(
        conn,
        "old-english",
        bundle,
        list(FALLBACK_REFERENCE_TAGS[:5]),
    )
    # Both branches contribute distinct citations. Dropping either
    # OR-branch fails: without inflection-child rollup ham stays at
    # 1 witness (avg drops to (3+1+2)/3 = 2.0, promotion at 1);
    # without merge-loser rollup cot still has 3 own citations so
    # its count is unchanged here but the merge-loser path would
    # break for any lemma where the loser carries DISTINCT sources.
    assert abs(card.avg_witnesses - 2.667) < 1e-3, (
        f"witness rollup should sum cot=3, ham=3, tun=2 → 2.667; got {card.avg_witnesses}"
    )
    # wyrd-fssn: OE threshold ≥2; all three (cot=3, ham=3 via rollup,
    # tun=2) cross the bar.
    assert card.promotion_eligible == 3


def test_scorecard_welsh_sparse_terminus_forward() -> None:
    """Welsh in the fixture has just the cognate mate row — exercises
    the sparse path. Welsh chain marks it as terminus-forward (no
    younger language in corpus)."""
    conn = _build_fixture_db()
    bundle = _fixture_bundle()
    card = compute_scorecard(conn, "welsh", bundle, list(FALLBACK_REFERENCE_TAGS[:5]))
    assert card.total_etymons == 1
    assert card.total_lemmas == 1
    assert card.promotion_eligible == 0
    # welsh shares the celtic_mix sibling — count is the celtic_mix count.
    assert card.bundle_sibling == "celtic_mix"
    assert card.bundle_word_count == 2
    assert "irish" in card.bundle_shared_with
    # tier-1 fires from the cognate mate (welsh → OE).
    assert card.tier1_cognate_count == 1
    # terminus-forward → tier-2-fwd is 0, no descendants in corpus.
    assert card.tier2_forward_count == 0
    assert card.chain_younger == []
    # tier-2-back: welsh's 'older' chain is [middle-welsh, old-welsh] —
    # neither in fixture, so back edge count is 0 (different from n/a).
    assert card.tier2_back_count == 0


def test_scorecard_lexicon_tag_coverage_is_independent_of_bundle() -> None:
    """Regression for the 'middle-english reads 0/15 in metric C' bug:
    lexicon-side tag coverage MUST read etymon_tag directly so that
    languages with sparse / no bundle visibility still get a real
    reading from the corpus side.

    Use a language that's NOT in _BUNDLE_LANG_KEY at all (sibling
    resolves to None) so the bundle-side path returns zeros while the
    lexicon-side path reads the etymon_tag rows directly. Originally
    pinned with 'middle-english' which mapped to None; wyrd-i1s1
    fixed that map (ME now routes to the modern_english sibling), so
    the regression coverage moved to a synthetic language."""
    conn = _build_fixture_db()
    # Insert etymons in a synthetic language that has no _BUNDLE_LANG_KEY
    # entry. Tagged with reference-set entries to verify lex-side
    # reads them.
    conn.executescript(
        """
        INSERT INTO etymon(id, canonical_form, language)
          VALUES (100, 'ax', 'klingon');
        INSERT INTO etymon(id, canonical_form, language)
          VALUES (101, 'ay', 'klingon');
        INSERT INTO etymon_tag(etymon_id, tag) VALUES (100, 'architecture');
        INSERT INTO etymon_tag(etymon_id, tag) VALUES (101, 'topography');
        -- wyrd-bfv1: dashboard restricts to generator-eligible etymons
        -- (cited + 1-hop descent + lemma + fantasy). Tag-coverage
        -- counts only eligible etymons, so cite both klingon entries
        -- to make them eligible and verify the lex-side reading.
        INSERT INTO etymon_citation(etymon_id, source_id) VALUES (100, 'skeat_1901');
        INSERT INTO etymon_citation(etymon_id, source_id) VALUES (101, 'skeat_1901');
        """
    )
    bundle = _fixture_bundle()
    card = compute_scorecard(
        conn,
        "klingon",
        bundle,
        list(FALLBACK_REFERENCE_TAGS[:5]),
    )
    assert card.bundle_sibling is None  # no dedicated bundle sibling
    # lexicon side surfaces the tags
    assert card.lexicon_tagged_etymons == 2
    assert card.lexicon_tag_coverage["architecture"] == 1
    assert card.lexicon_tag_coverage["topography"] == 1
    assert abs(card.lexicon_tag_hit_rate - 0.4) < 1e-6
    # bundle side correctly reports 0 — but the user can read the
    # lexicon-side metric to know the language HAS tag data, just no
    # bundle visibility.
    assert all(v == 0 for v in card.bundle_tag_coverage.values())
    assert card.bundle_tag_hit_rate == 0.0


def test_scorecard_unknown_language_chain_is_marked_unknown() -> None:
    """A language not in ERA_CHAINS gets ``chain_known=False``; both
    direction lists empty. Pinned so the markdown renderer doesn't
    accidentally credit such a language with terminus-anything."""
    conn = _build_fixture_db()
    bundle = _fixture_bundle()
    # 'klingon' isn't in ERA_CHAINS or RECOMMENDED_LANG_THRESHOLDS;
    # it's also not in the fixture so total_etymons is 0.
    card = compute_scorecard(conn, "klingon", bundle, list(FALLBACK_REFERENCE_TAGS[:5]))
    assert card.chain_known is False
    assert card.chain_older == []
    assert card.chain_younger == []
    assert card.total_etymons == 0


# --- compute_report end-to-end -------------------------------------------


def test_compute_report_drops_empty_languages_by_default() -> None:
    """Languages with zero etymons don't get scorecards in the default
    output — every-zero rows are noise on the human report."""
    conn = _build_fixture_db()
    bundle = _fixture_bundle()
    report = compute_report(
        conn,
        bundle,
        languages=["old-english", "welsh", "klingon"],
        reference_tags=list(FALLBACK_REFERENCE_TAGS[:3]),
    )
    langs = {c.language for c in report.languages}
    assert langs == {"old-english", "welsh"}  # klingon dropped


def test_compute_report_keeps_empty_when_flag_set() -> None:
    """``drop_empty=False`` retains every requested language — for
    trend-tracking JSON snapshots where the absence is informative."""
    conn = _build_fixture_db()
    bundle = _fixture_bundle()
    report = compute_report(
        conn,
        bundle,
        languages=["old-english", "klingon"],
        reference_tags=list(FALLBACK_REFERENCE_TAGS[:3]),
        drop_empty=False,
    )
    langs = {c.language for c in report.languages}
    assert "klingon" in langs


def test_compute_report_carries_schema_metadata() -> None:
    conn = _build_fixture_db()
    bundle = _fixture_bundle()
    report = compute_report(
        conn,
        bundle,
        languages=["old-english"],
        reference_tags=list(FALLBACK_REFERENCE_TAGS[:3]),
    )
    assert report.schema_version == "1.0"
    assert report.generated_at.endswith("Z")
    assert report.bundle_total_words == 4
    assert report.reference_tags == list(FALLBACK_REFERENCE_TAGS[:3])


# --- formatters ----------------------------------------------------------


def test_report_to_json_round_trips_through_load() -> None:
    """JSON formatter emits valid JSON; reloading recovers every field
    of the dataclass at the schema-versioned shape."""
    conn = _build_fixture_db()
    bundle = _fixture_bundle()
    report = compute_report(
        conn,
        bundle,
        languages=["old-english"],
        reference_tags=list(FALLBACK_REFERENCE_TAGS[:3]),
    )
    text = report_to_json(report)
    data = json.loads(text)
    assert data["schema_version"] == "1.0"
    assert data["bundle_total_words"] == 4
    assert data["languages"][0]["language"] == "old-english"
    assert data["languages"][0]["promotion_threshold"] == 2
    # Both tag-coverage maps must round-trip as dicts.
    assert isinstance(data["languages"][0]["lexicon_tag_coverage"], dict)
    assert isinstance(data["languages"][0]["bundle_tag_coverage"], dict)


def test_report_to_markdown_includes_summary_and_detail() -> None:
    """Markdown formatter emits a summary table + per-language detail
    blocks. Pinned to avoid silent drift in the rendered shape."""
    conn = _build_fixture_db()
    bundle = _fixture_bundle()
    report = compute_report(
        conn,
        bundle,
        languages=["old-english", "welsh"],
        reference_tags=list(FALLBACK_REFERENCE_TAGS[:3]),
    )
    md = report_to_markdown(report)
    assert "## Summary" in md
    assert "## Per-language detail" in md
    assert "### old-english" in md
    assert "### welsh" in md
    # Welsh's bundle representation should annotate the shared sibling.
    assert "shared with" in md.lower()
    # OE's chain is terminus-back; the markdown should say so.
    assert "terminus-back" in md
    # Both lexicon-side and bundle-side tag-coverage sections appear.
    assert "Semantic-tag coverage (lexicon)" in md
    assert "Bundle tag visibility" in md
    # F₂ (wyrd-h5it): the summary table column is the lex/bundle split
    # and the per-language detail mentions the bundle-side reading.
    assert "Era reflex (lex / bundle)" in md
    assert "Bundle subjects with reflex stop targeting this language" in md
    # F Tier 4 (wyrd-pmne): the per-language detail breakdown lists a
    # Tier 4 phonology slot — 'n/a' for languages without a registered
    # phonology chain (Welsh in the integration fixture has no in-DB
    # canonical_forms whose chain is wired in lexicon._PHONOLOGY_*),
    # a count otherwise.
    assert "Tier 4 phonology" in md


def test_report_to_markdown_handles_no_bundle_sibling() -> None:
    """Languages without a dedicated bundle sibling get the explicit
    'no dedicated bundle sibling' line rather than a misleading 0/N.

    Uses a synthetic language not present in _BUNDLE_LANG_KEY so the
    sibling resolves to None — the test target moved off
    middle-english after wyrd-i1s1 mapped ME to the modern_english
    sibling (matching the actual export-side routing)."""
    conn = _build_fixture_db()
    # 'klingon' has no _BUNDLE_LANG_KEY entry — sibling resolves None.
    assert _BUNDLE_LANG_KEY.get("klingon") is None
    # Insert a klingon etymon so the language isn't dropped as empty.
    conn.execute("INSERT INTO etymon(id, canonical_form, language) VALUES (200, 'qet', 'klingon')")
    bundle = _fixture_bundle()
    report = compute_report(
        conn,
        bundle,
        languages=["klingon"],
        reference_tags=list(FALLBACK_REFERENCE_TAGS[:3]),
    )
    md = report_to_markdown(report)
    assert "no dedicated bundle sibling" in md


def test_report_to_markdown_renders_shared_with_for_modern_english_cluster() -> None:
    """wyrd-i1s1 added middle-english / early-modern-english / scots to
    the modern_english shared-sibling cluster. Pin the per-cluster
    'shared with' annotation so a future map edit that drops one of
    the languages surfaces here rather than silently in the rendered
    output."""
    conn = _build_fixture_db()
    # middle-english already in the fixture (cote, tonn at ids 10/11).
    bundle = _fixture_bundle()
    report = compute_report(
        conn,
        bundle,
        languages=["middle-english"],
        reference_tags=list(FALLBACK_REFERENCE_TAGS[:3]),
    )
    md = report_to_markdown(report)
    # The middle-english scorecard should annotate the shared sibling.
    assert "sibling=`modern_english`" in md
    assert "shared with" in md.lower()
    # The shared-with list should include the ME/EModE/ModE/Scots
    # cluster mates (other than middle-english itself).
    assert "early-modern-english" in md
    assert "modern-english" in md
    assert "scots" in md


# --- reference-tag loader ------------------------------------------------


def test_load_reference_tags_returns_top_n_from_proportions(tmp_path: Path) -> None:
    """Loader ranks tag_marginal by descending count, ties broken by
    tag name. Pin the deterministic ordering."""
    proportions = tmp_path / "english_proportions.json"
    proportions.write_text(json.dumps({"tag_marginal": {"foo": 100, "bar": 50, "baz": 50}}))
    tags = load_reference_tags(proportions, top_n=3)
    # foo > bar/baz; among ties bar < baz alphabetically.
    assert tags == ["foo", "bar", "baz"]


def test_load_reference_tags_falls_back_when_file_missing() -> None:
    """A non-existent path is the call site's job to handle defensively;
    fall through to FALLBACK_REFERENCE_TAGS rather than crashing."""
    tags = load_reference_tags(Path("/nonexistent/proportions.json"), top_n=3)
    assert tags == list(FALLBACK_REFERENCE_TAGS[:3])


def test_load_reference_tags_falls_back_when_marginal_absent(
    tmp_path: Path,
) -> None:
    """Legacy proportions snapshot (pre-wyrd-mj2) had no tag_marginal —
    the loader must not crash, just fall back."""
    proportions = tmp_path / "english_proportions.json"
    proportions.write_text(json.dumps({"single_usages": {}, "structures": {}}))
    tags = load_reference_tags(proportions, top_n=4)
    assert tags == list(FALLBACK_REFERENCE_TAGS[:4])


# --- structural pins -----------------------------------------------------


def test_default_languages_are_a_subset_of_chain_or_explicitly_unknown() -> None:
    """Every language in DEFAULT_LANGUAGES should either appear in
    ERA_CHAINS or be explicitly tolerated by the dashboard. Currently
    every default IS in ERA_CHAINS — pin this so dropping a language
    from ERA_CHAINS without updating the default list fails here."""
    for lang in DEFAULT_LANGUAGES:
        assert lang in ERA_CHAINS, f"DEFAULT_LANGUAGES has {lang!r} but no ERA_CHAINS entry"


def test_shared_bundle_siblings_reference_known_languages() -> None:
    """Each language listed under a shared sibling must have that
    sibling as its _BUNDLE_LANG_KEY value. Pin the bidirectional
    consistency so a future map edit can't silently de-sync."""
    from wyrd.generators.kenning.language_quality import _SHARED_BUNDLE_SIBLINGS

    for sibling, langs in _SHARED_BUNDLE_SIBLINGS.items():
        for lang in langs:
            assert _BUNDLE_LANG_KEY.get(lang) == sibling, (
                f"{lang!r} listed under shared sibling {sibling!r} but "
                f"_BUNDLE_LANG_KEY says {_BUNDLE_LANG_KEY.get(lang)!r}"
            )


def test_language_scorecard_dataclass_carries_required_fields() -> None:
    """Pin the dataclass surface — JSON consumers depend on every field
    being present even when zero. A field rename / removal here would
    break trend-tracking comparison across snapshots."""
    fields = {
        "language",
        "total_etymons",
        "total_lemmas",
        # A. eligible-set restriction (wyrd-bfv1)
        "eligible_etymons",
        "eligible_lemmas",
        "promotion_threshold",
        "promotion_eligible",
        "avg_witnesses",
        "source_count",
        "bundle_sibling",
        "bundle_word_count",
        "bundle_shared_with",
        "attribution_mode",
        "reference_tags",
        "lexicon_tagged_etymons",
        "lexicon_tag_coverage",
        "lexicon_tag_hit_rate",
        "bundle_tag_coverage",
        "bundle_tag_hit_rate",
        "lemmas_with_inflections",
        "inflection_density",
        "distinct_inflection_labels",
        "etymons_with_variants",
        "variant_density",
        "chain_older",
        "chain_younger",
        "chain_known",
        "tier1_cognate_count",
        "tier2_forward_count",
        "tier2_back_count",
        "tier3_period_form_count",
        # F (Tier 4). Phonology-rule fallback (wyrd-pmne)
        "tier4_phonology_count",
        "any_reflex_count",
        "any_reflex_rate",
        # F₂. Bundle-side era-reflex coverage (wyrd-h5it)
        "bundle_subjects_with_reflex",
        "bundle_reflex_rate",
        "stratum_classified",
        "stratum_density",
        "etymons_with_citations",
        "avg_citations",
        # J. Bundle attestation breakdown (wyrd-lc94)
        "bundle_attestation_total",
        "bundle_scholar_attested",
        "bundle_empirical_only",
        "bundle_rando_only",
        "bundle_uncited",
        "bundle_scholar_attestation_rate",
        # H. Pronunciation / IPA coverage (wyrd-dxu2)
        "lexicon_etymons_with_ipa",
        "lexicon_ipa_density",
        "bundle_subjects_with_ipa",
        "bundle_ipa_rate",
        # K. Rando-port grandfather audit (wyrd-vn09)
        "rando_pure_grandfather_families",
        "rando_mixed_grandfather_families",
        "rando_corr1_families",
        "rando_corr2_families",
        "rando_corr3plus_families",
        "rando_total_cited_families",
        "rando_pure_grandfather_rate",
    }
    actual = set(LanguageScorecard.__dataclass_fields__.keys())
    assert actual == fields, f"missing: {fields - actual}, extra: {actual - fields}"


def test_default_languages_within_canonical_or_transitional():
    """wyrd-xdam.1 drift tripwire: every DEFAULT_LANGUAGES entry is either a
    canonical language (controlled_vocab) or one of the documented transitional
    fine-grained Celtic names. Pins the exact gap so a NEW non-canonical language
    can't drift into the dashboard list unnoticed. As of wyrd-xdam.4 the
    transitional set is EMPTY — the fine-grained Celtic names are now first-class
    canonical — so every DEFAULT_LANGUAGES entry must be canonical.

    Two directional checks so the failure message points the right way: a new
    non-canonical entry trips the first assertion; a stale transitional entry (no
    longer non-canonical) trips the second."""
    noncanonical = set(DEFAULT_LANGUAGES) - set(LANGUAGE_CANONICAL)
    transitional = set(TRANSITIONAL_NONCANONICAL_LANGUAGES)

    sneaked_in = noncanonical - transitional
    assert not sneaked_in, (
        f"New non-canonical language(s) in DEFAULT_LANGUAGES: {sorted(sneaked_in)}. "
        "Add each to LANGUAGE_CANONICAL, or (if intentionally transitional) to "
        "TRANSITIONAL_NONCANONICAL_LANGUAGES."
    )
    stale_transitional = transitional - noncanonical
    assert not stale_transitional, (
        f"TRANSITIONAL_NONCANONICAL_LANGUAGES is stale: {sorted(stale_transitional)} "
        "is no longer a non-canonical DEFAULT_LANGUAGES entry — remove it from the "
        "transitional set (it was either made canonical or dropped from the dashboard)."
    )


def test_celtic_branches_wired_into_report_and_era() -> None:
    """wyrd-xdam.3: the branch-level Celtic languages are wired into every report
    + era structure additively (the fine-grained welsh/irish stay until xdam.4)."""
    from wyrd.generators.kenning.era.cells import LANGUAGE_TO_FAMILY

    for branch in ("brittonic", "goidelic"):
        assert branch in DEFAULT_LANGUAGES
        assert branch in ERA_CHAINS  # satisfies the DEFAULT⊆ERA_CHAINS tripwire
        assert _BUNDLE_LANG_KEY[branch] == "celtic_mix"
    # era family resolution (the deferred wyrd-xdam.1 wiring)
    assert LANGUAGE_TO_FAMILY["brittonic"] == "brythonic"
    assert LANGUAGE_TO_FAMILY["goidelic"] == "goidelic"
    # additive: the fine-grained Celtic + umbrella are NOT removed (wiktextract L3
    # data lives until xdam.4)
    for kept in ("welsh", "irish", "celtic"):
        assert kept in DEFAULT_LANGUAGES
    # each branch's era-chain younger list holds only that branch's languages:
    # every member present in LANGUAGE_TO_FAMILY must map to the branch's family
    # (modern-welsh is era-chain-only, not in the family map → tolerated)
    for branch, family in (("brittonic", "brythonic"), ("goidelic", "goidelic")):
        for member in ERA_CHAINS[branch]["younger"]:
            if member in LANGUAGE_TO_FAMILY:
                assert LANGUAGE_TO_FAMILY[member] == family, (branch, member)
    assert "goidelic" not in ERA_CHAINS["brittonic"]["younger"]  # no cross-branch leak


# ---- wyrd-v7wg: bundle-attribution mode (donor-only vs data-gap vs corpus-thin) ----


def _minimal_card(**overrides) -> LanguageScorecard:
    """A LanguageScorecard with just the fields the bundle-representation render
    needs; overrides set the attribution-relevant ones."""
    base = {"language": "latin", "total_etymons": 10, "total_lemmas": 10}
    base.update(overrides)
    return LanguageScorecard(**base)


def _md_for(card: LanguageScorecard, bundle_total_words: int = 100) -> str:
    report = LanguageQualityReport(
        schema_version="1.0",
        generated_at="2026-06-21T00:00:00Z",
        reference_tags=list(FALLBACK_REFERENCE_TAGS[:3]),
        bundle_total_words=bundle_total_words,
        languages=[card],
    )
    return report_to_markdown(report)


def test_attribution_mode_for_classifies_donor_receiver_both() -> None:
    from wyrd.generators.kenning.language_quality import ATTRIBUTION_MODE, attribution_mode_for

    assert attribution_mode_for("latin") == "donor-only"
    assert attribution_mode_for("greek") == "donor-only"
    assert attribution_mode_for("old-norse") == "both"
    assert attribution_mode_for("old-french") == "both"
    # receiver is the default for anything absent (incl. the english family + unknowns)
    assert attribution_mode_for("old-english") == "receiver"
    assert attribution_mode_for("modern-english") == "receiver"
    assert attribution_mode_for("totally-made-up-lang") == "receiver"
    # the constant only carries the non-default entries (mirrors the curated-map pattern)
    assert "old-english" not in ATTRIBUTION_MODE
    assert ATTRIBUTION_MODE["latin"] == "donor-only"


def test_compute_scorecard_threads_attribution_mode_via_map() -> None:
    """The actual wiring: compute_scorecard sets attribution_mode from the map for
    the real language (not the field default). Deleting the audits.py wiring line
    would fall back to 'receiver' and fail this. attribution_mode is map-derived,
    not DB-derived, so it's correct even for a language with no fixture etymons."""
    conn = _build_fixture_db()
    ref = list(FALLBACK_REFERENCE_TAGS[:3])
    latin = compute_scorecard(conn, "latin", [], ref, compute_tier4=False)
    oe = compute_scorecard(conn, "old-english", [], ref, compute_tier4=False)
    norse = compute_scorecard(conn, "old-norse", [], ref, compute_tier4=False)
    assert latin.attribution_mode == "donor-only"
    assert oe.attribution_mode == "receiver"
    assert norse.attribution_mode == "both"


def test_donor_only_renders_na_not_zero() -> None:
    """A donor-only language must read 'donor-only (n/a)', not '0 (0.0%)', in both
    the summary table and the detail section — the zero is by-design (loans are
    receiver-attributed), not a should-fix gap."""
    md = _md_for(
        _minimal_card(
            language="latin",
            attribution_mode="donor-only",
            bundle_sibling="latin",
            bundle_word_count=0,
        )
    )
    assert "donor-only (n/a)" in md
    assert "loans are attributed to the receiving language" in md
    # the bundle cell/detail must NOT present the misleading should-fix zero
    # (other unrelated zero metrics for latin may still render 0.0% — scope to bundle)
    assert "Bundle representation:** 0 words" not in md


def test_donor_only_with_words_shows_count_not_na() -> None:
    """wyrd-v7wg (Gemini review): the 'donor-only (n/a)' placeholder is only for the
    EXPECTED zero. A donor-only language that unexpectedly HAS bundle words must show
    the real count so the anomaly is visible, not hidden behind n/a."""
    md = _md_for(
        _minimal_card(
            language="latin",
            attribution_mode="donor-only",
            bundle_sibling="latin",
            bundle_word_count=5,
        )
    )
    assert "donor-only (n/a)" not in md
    assert "Bundle representation:** 5 words" in md


def test_receiver_zero_with_promotables_reads_data_gap() -> None:
    md = _md_for(
        _minimal_card(
            language="welsh",
            attribution_mode="receiver",
            bundle_sibling="celtic_mix",
            bundle_word_count=0,
            promotion_threshold=3,
            promotion_eligible=7,
        )
    )
    assert "data-gap" in md
    assert "7 promotable lemmas" in md
    assert "corpus-thin" not in md


def test_receiver_zero_without_promotables_reads_corpus_thin() -> None:
    md = _md_for(
        _minimal_card(
            language="breton",
            attribution_mode="receiver",
            bundle_sibling="celtic_mix",
            bundle_word_count=0,
            promotion_eligible=0,
        )
    )
    assert "corpus-thin" in md
    assert "data-gap" not in md


def test_receiver_with_coverage_gets_no_gap_annotation() -> None:
    md = _md_for(
        _minimal_card(
            language="old-english",
            attribution_mode="receiver",
            bundle_sibling="old_english",
            bundle_word_count=40,
        )
    )
    assert "data-gap" not in md and "corpus-thin" not in md
    assert "donor-only" not in md


def test_both_mode_renders_as_receiver_not_donor_only() -> None:
    """'both' (old-norse/old-french/norman-french) donate AND receive, so coverage
    is expected from the receiving direction — render the receiver-style signal
    (data-gap / corpus-thin / numeric cell), never 'donor-only (n/a)'."""
    md = _md_for(
        _minimal_card(
            language="old-norse",
            attribution_mode="both",
            bundle_sibling="old_scandinavian",
            bundle_word_count=0,
            promotion_threshold=2,
            promotion_eligible=4,
        )
    )
    assert "donor-only" not in md
    assert "data-gap" in md and "4 promotable lemmas" in md


def test_grandfather_k_row_renders_corroborator_breakdown() -> None:
    """wyrd-g7n6: the K-row detail expands the mixed bucket into the corroborator
    distribution (rando-only / +1 / +2 / +≥3)."""
    card = LanguageScorecard(
        language="old-english",
        total_etymons=10,
        total_lemmas=10,
        rando_pure_grandfather_families=5,
        rando_mixed_grandfather_families=6,
        rando_corr1_families=3,
        rando_corr2_families=2,
        rando_corr3plus_families=1,
        rando_total_cited_families=11,
    )
    report = LanguageQualityReport(
        schema_version="1.0",
        generated_at="2026-06-21T00:00:00Z",
        reference_tags=list(FALLBACK_REFERENCE_TAGS[:3]),
        bundle_total_words=100,
        languages=[card],
    )
    md = report_to_markdown(report)
    assert "5 rando-only" in md and "retirement candidates" in md
    assert "6 corroborated" in md
    assert "+1: 3" in md and "+2: 2" in md and "+≥3: 1" in md
    assert "of 11 cited families" in md
