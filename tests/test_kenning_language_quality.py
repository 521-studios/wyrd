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
    LanguageScorecard,
    _bundle_language_word_count,
    _bundle_tag_coverage_for_sibling,
    _bundle_total_words,
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
            stratum TEXT
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
        INSERT INTO etymon(id, canonical_form, language, stratum)
          VALUES (1, 'cot', 'old-english', 'native-old-english');
        INSERT INTO etymon(id, canonical_form, language, stratum)
          VALUES (2, 'ham', 'old-english', 'native-old-english');
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
    (covers tag coverage), one tagged 'water', one untagged."""
    return [
        {
            "meaning": ["cottage"],
            "modifier_tags": ["architecture"],
            "modifier_type": None,
            "words": [{"modern_usage": "cot", "old_english": ["cot"]}],
        },
        {
            "meaning": ["enclosure"],
            "modifier_tags": ["topography", "agriculture"],
            "modifier_type": None,
            "words": [
                {
                    "modern_usage": "tun",
                    "old_english": ["tun"],
                    "celtic_mix": ["caer"],
                }
            ],
        },
        {
            "meaning": ["river"],
            "modifier_tags": ["water"],
            "modifier_type": None,
            "words": [{"modern_usage": "afon", "celtic_mix": ["afon"]}],
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
    languages without a bundle sibling still get a real reading.

    Build a fresh fixture where 'middle-english' has 2 tagged etymons
    but NO bundle sibling. Pin: lexicon_tagged_etymons reflects the
    etymon_tag rows; bundle_tag_coverage is all-zero (no sibling)."""
    conn = _build_fixture_db()
    # ME etymons cote (id 10) and tonn (id 11) — tag them.
    conn.executescript(
        """
        INSERT INTO etymon_tag(etymon_id, tag) VALUES (10, 'architecture');
        INSERT INTO etymon_tag(etymon_id, tag) VALUES (11, 'topography');
        """
    )
    bundle = _fixture_bundle()
    card = compute_scorecard(
        conn,
        "middle-english",
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


def test_report_to_markdown_handles_no_bundle_sibling() -> None:
    """Languages without a dedicated bundle sibling get the explicit
    'no dedicated bundle sibling' line rather than a misleading 0/N."""
    conn = _build_fixture_db()
    # middle-english is _BUNDLE_LANG_KEY=None per the map.
    assert _BUNDLE_LANG_KEY["middle-english"] is None
    bundle = _fixture_bundle()
    # middle-english has 2 etymons in the fixture (cote, tonn) so the
    # report keeps it.
    report = compute_report(
        conn,
        bundle,
        languages=["middle-english"],
        reference_tags=list(FALLBACK_REFERENCE_TAGS[:3]),
    )
    md = report_to_markdown(report)
    assert "no dedicated bundle sibling" in md


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
        "any_reflex_count",
        "any_reflex_rate",
        "stratum_classified",
        "stratum_density",
        "etymons_with_citations",
        "avg_citations",
    }
    actual = set(LanguageScorecard.__dataclass_fields__.keys())
    assert actual == fields, f"missing: {fields - actual}, extra: {actual - fields}"
