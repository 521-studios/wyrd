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
    _tier4_phonology_coverage,
    _bundle_tag_coverage_for_sibling,
    _bundle_total_words,
    _lexicon_ipa_coverage,
    compute_rando_port_grandfather_audit,
    compute_report,
    compute_scorecard,
    load_reference_tags,
    report_to_json,
    report_to_markdown,
)


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
    # exercises tier-1).
    conn.executescript(
        """
        INSERT INTO etymon(id, canonical_form, language, cognate_id)
          VALUES (20, 'cwt', 'welsh', 1);
        UPDATE etymon SET cognate_id = 20 WHERE id = 1;
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
            merged_into_id INTEGER REFERENCES etymon(id)
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
    conn.commit()
    return conn


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
    reflex (via _phonology_rule_form) for at least one chain stop
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
    from wyrd.generators.kenning.lexicon import _phonology_rule_form

    if _phonology_rule_form("house", "modern-english", "old-english") is not None:
        assert count >= 1, "phonology rules fire on 'house' but T4 count says 0"


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
    """Lemma-id and merged_into_id rollup matches lexicon._build_family_rollup:
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
    assert card.promotion_threshold == 3  # OE entry in RECOMMENDED_LANG_THRESHOLDS
    assert card.promotion_eligible == 1  # only cot family hits ≥3 witnesses
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
    assert data["languages"][0]["promotion_threshold"] == 3
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
        "promotion_threshold",
        "promotion_eligible",
        "avg_witnesses",
        "source_count",
        "bundle_sibling",
        "bundle_word_count",
        "bundle_shared_with",
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
        "rando_total_cited_families",
        "rando_pure_grandfather_rate",
    }
    actual = set(LanguageScorecard.__dataclass_fields__.keys())
    assert actual == fields, f"missing: {fields - actual}, extra: {actual - fields}"
