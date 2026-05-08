"""Language-quality dashboard — per-language scorecard from the lexicon
DB + bundled meanings.json (wyrd-wzwa Phase 1, parent epic wyrd-eni4).

Per-language metrics:

- A. Corpus depth — total etymons / lemmas, promotion threshold (per
  language), promotion-eligible count at that threshold, avg witnesses,
  source count.
- B. Bundle representation — count of bundle words carrying this
  language as a sibling field.
- C. Semantic-tag coverage — split into two views, both keyed off the
  top-N tags of the English reference distribution (top-15 of
  english_proportions.json ``tag_marginal``):
  * **Lexicon side** (``etymon_tag`` join): how many etymons in this
    language carry ≥1 tag, plus per-reference-tag distinct etymon
    count. Surfaces the corpus-side gap independent of bundle export
    so middle-english / modern-english (no bundle sibling) still get a
    real reading.
  * **Bundle side** (``modifier_tags`` × bundle sibling): how many
    bundle subjects expose this language with each reference tag.
    Captures the runtime-visibility view; bundle-less languages read 0
    across the board.
- D. Inflection coverage (D8) — % of lemmas with at least one linked
  inflected child + distinct case labels.
- E. Spelling-variant coverage (D18) — % of etymons with a
  ``etymon_text_match.matched_form != canonical_form`` row.
- F. Era-reflex coverage (D33) — Tier 1 (cognate cluster), Tier 2
  forward (descent edge to a "younger" language per the chain), Tier 2
  back (descent edge from an "older" language), Tier 3 (period-form
  projection), and the union "any reflex" rate. Languages can be
  marked terminus-back / terminus-forward in ``ERA_CHAINS`` so missing
  edges don't penalize languages with no parent / child cell in corpus.
- G. Stratum coverage (D32) — % of etymons with ``stratum`` set.
- I. Citation depth — % of etymons with ≥1 ``etymon_citation`` row +
  average citations per cited etymon (denominator is cited etymons,
  not total etymons; reports depth among the cited population rather
  than diluting with uncited zeros).

Phase-1 metric C uses the simpler tag-marginal hit rate; per-meaning
catalog coverage (cataloging meanings INSIDE each subject so we can
estimate meaning-space coverage) is a deliberate follow-on.

Output: markdown report (human review) + JSON snapshot (machine /
trend tracking) from one CLI command (``wyrd kenning lexicon
language-report``).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from wyrd.generators.kenning.lexicon import RECOMMENDED_LANG_THRESHOLDS

# --- era chain config -----------------------------------------------------
#
# Per-language historical chain. ``older=[]`` means terminus-back (no
# parent cell in the corpus by construction — proto-languages); ``younger=[]``
# means terminus-forward (modern-language leaves). Used by tier 2
# forward/back metrics so a language with no parent cell doesn't get
# penalized for 0% backward linkage.
#
# Languages absent from this map fall back to "no chain known" — both
# directions reported as N/A in the scorecard.

ERA_CHAINS: dict[str, dict[str, list[str]]] = {
    # English chain
    "old-english": {"older": [], "younger": ["middle-english"]},
    "middle-english": {
        "older": ["old-english"],
        "younger": ["early-modern-english", "modern-english"],
    },
    "early-modern-english": {
        "older": ["middle-english"],
        "younger": ["modern-english"],
    },
    "modern-english": {
        "older": ["early-modern-english", "middle-english"],
        "younger": [],
    },
    # Welsh chain
    "old-welsh": {
        "older": [],
        "younger": ["middle-welsh", "modern-welsh", "welsh"],
    },
    "middle-welsh": {"older": ["old-welsh"], "younger": ["modern-welsh", "welsh"]},
    "modern-welsh": {"older": ["middle-welsh", "old-welsh"], "younger": []},
    "welsh": {"older": ["middle-welsh", "old-welsh"], "younger": []},
    # Goidelic chain
    "old-irish": {
        "older": [],
        "younger": ["middle-irish", "irish", "scottish-gaelic"],
    },
    "middle-irish": {
        "older": ["old-irish"],
        "younger": ["irish", "scottish-gaelic"],
    },
    "irish": {"older": ["middle-irish", "old-irish"], "younger": []},
    "scottish-gaelic": {"older": ["middle-irish", "old-irish"], "younger": []},
    # Brythonic Breton
    "breton": {"older": ["middle-breton", "old-breton"], "younger": []},
    # Norse / Germanic
    "old-norse": {
        "older": ["proto-germanic", "proto-norse"],
        "younger": ["icelandic", "faroese", "norwegian", "swedish", "danish"],
    },
    "icelandic": {"older": ["old-norse"], "younger": []},
    # Romance
    "latin": {"older": [], "younger": ["old-french", "norman-french", "french"]},
    "old-french": {
        "older": ["latin"],
        "younger": ["norman-french", "french", "modern-french"],
    },
    "norman-french": {"older": ["old-french", "latin"], "younger": []},
    # rando-port meta-buckets (carried in the legacy seed)
    "celtic": {
        "older": [],
        "younger": ["welsh", "irish", "scottish-gaelic", "breton"],
    },
}


# Default languages the dashboard reports on. Each is verified at query
# time — languages with zero etymons in the DB are dropped from the
# output rather than reported as zeros across every axis.
DEFAULT_LANGUAGES: tuple[str, ...] = (
    "modern-english",
    "middle-english",
    "old-english",
    "modern-welsh",
    "welsh",
    "old-welsh",
    "irish",
    "old-irish",
    "scottish-gaelic",
    "breton",
    "old-norse",
    "old-french",
    "norman-french",
    "latin",
    "celtic",
)


# Number of top English tags to use as the reference semantic profile.
# Read from english_proportions.json's tag_marginal at runtime; this is
# a fallback if the proportions file isn't reachable.
REFERENCE_TAG_COUNT: int = 15

FALLBACK_REFERENCE_TAGS: tuple[str, ...] = (
    "architecture",
    "topography",
    "social",
    "descriptive",
    "plant",
    "agriculture",
    "water",
    "animal",
    "measurement",
    "geology",
    "religious",
    "tree",
    "color",
    "size",
    "domestic",
)


# --- dataclasses ----------------------------------------------------------


@dataclass
class LanguageScorecard:
    """Metrics for one language. Values are absolute counts plus rate
    fields where a rate is meaningful. Era-chain fields surface the
    structural endpoint markers so consumers can render N/A for
    terminal-at-this-end axes.

    Schema field-order matches the rendered markdown sections (A → I)
    so JSON snapshots stay self-explanatory."""

    language: str
    # A. Corpus depth
    total_etymons: int
    total_lemmas: int
    promotion_threshold: int
    promotion_eligible: int
    avg_witnesses: float
    source_count: int
    # B. Bundle representation
    bundle_sibling: str | None = None
    bundle_word_count: int = 0
    bundle_shared_with: list[str] = field(default_factory=list)
    # C. Semantic-profile coverage (vs English reference)
    reference_tags: list[str] = field(default_factory=list)
    # Lexicon-side: counts of etymons in this language tagged with each
    # reference tag. Reads etymon_tag directly so middle-english /
    # modern-english (no bundle sibling) still get a meaningful reading.
    lexicon_tagged_etymons: int = 0
    lexicon_tag_coverage: dict[str, int] = field(default_factory=dict)
    lexicon_tag_hit_rate: float = 0.0
    # Bundle-side: counts of bundle subjects whose modifier_tags include
    # the reference tag AND whose words contain a sibling for this
    # language. Captures the runtime-visibility view; bundle-less
    # languages (middle-english, norman-french) read 0 across the board.
    bundle_tag_coverage: dict[str, int] = field(default_factory=dict)
    bundle_tag_hit_rate: float = 0.0
    # D. Inflection coverage
    lemmas_with_inflections: int = 0
    inflection_density: float = 0.0
    distinct_inflection_labels: int = 0
    # E. Spelling-variant coverage
    etymons_with_variants: int = 0
    variant_density: float = 0.0
    # F. Era-reflex coverage
    chain_older: list[str] = field(default_factory=list)
    chain_younger: list[str] = field(default_factory=list)
    chain_known: bool = False
    tier1_cognate_count: int = 0
    tier2_forward_count: int = 0
    tier2_back_count: int = 0
    tier3_period_form_count: int = 0
    any_reflex_count: int = 0
    any_reflex_rate: float = 0.0
    # G. Stratum coverage
    stratum_classified: int = 0
    stratum_density: float = 0.0
    # I. Citation depth
    etymons_with_citations: int = 0
    avg_citations: float = 0.0


@dataclass
class LanguageQualityReport:
    """Top-level report — one scorecard per language plus the reference
    profile used to compute metric C, plus a generation timestamp.
    JSON-serializable via ``asdict``."""

    schema_version: str
    generated_at: str
    reference_tags: list[str]
    bundle_total_words: int
    languages: list[LanguageScorecard] = field(default_factory=list)


# --- reference profile ----------------------------------------------------


def load_reference_tags(
    proportions_path: Path | None = None,
    *,
    top_n: int = REFERENCE_TAG_COUNT,
) -> list[str]:
    """Return the top-``top_n`` tags from English place-name proportions.

    Reads ``english_proportions.json`` and ranks the ``tag_marginal``
    map. Falls back to ``FALLBACK_REFERENCE_TAGS`` if the file is
    unreadable or the ``tag_marginal`` field is absent (e.g. legacy
    proportions snapshot pre-wyrd-mj2)."""
    if proportions_path is None or not proportions_path.exists():
        return list(FALLBACK_REFERENCE_TAGS[:top_n])
    try:
        data = json.loads(proportions_path.read_text())
    except (OSError, json.JSONDecodeError):
        return list(FALLBACK_REFERENCE_TAGS[:top_n])
    marginal = data.get("tag_marginal") or {}
    if not isinstance(marginal, dict) or not marginal:
        return list(FALLBACK_REFERENCE_TAGS[:top_n])
    ranked = sorted(marginal.items(), key=lambda kv: (-kv[1], kv[0]))
    return [tag for tag, _ in ranked[:top_n]]


# --- bundle helpers -------------------------------------------------------

# Lexicon-language → bundle-sibling-key map. The bundle uses meta-bucket
# fields (snake_case) like ``old_english`` and ``celtic_mix``; the
# lexicon uses specific kebab-case language codes. For families where
# the bundle rolls multiple languages into one sibling (Celtic →
# ``celtic_mix``, all Norse-family → ``old_scandinavian``), the same
# sibling key is reused — consumers must surface the sharing in the
# report so per-language bundle counts aren't read as exclusive.
# Languages with no bundle sibling at all (middle-english, norman-
# french, old-irish, etc.) map to ``None`` and report 0 bundle words
# with an explicit "no dedicated sibling" annotation.

_BUNDLE_LANG_KEY: dict[str, str | None] = {
    "old-english": "old_english",
    # ME / EModE / Scots route to the modern_english sibling at export
    # per _LANG_CODE_TO_JSON_FIELD in lexicon.py; the dashboard map
    # mirrors that so bundle-side tag visibility reads correctly.
    # Originally mapped to None in PR #144, which mis-reported
    # middle-english at 0/15 bundle tag hits despite 13/15 lex tag
    # hits — fixed alongside wyrd-i1s1's cluster-mate tag rollup.
    "middle-english": "modern_english",
    "early-modern-english": "modern_english",
    "modern-english": "modern_english",
    "scots": "modern_english",
    "old-welsh": "celtic_mix",
    "welsh": "celtic_mix",
    "modern-welsh": "celtic_mix",
    "middle-welsh": "celtic_mix",
    "irish": "celtic_mix",
    "old-irish": "celtic_mix",
    "middle-irish": "celtic_mix",
    "scottish-gaelic": "celtic_mix",
    "breton": "celtic_mix",
    "old-norse": "old_scandinavian",
    "icelandic": "old_scandinavian",
    # OF / Anglo-Norman / NF / Middle French route to old_french per
    # export-side map.
    "old-french": "old_french",
    "norman-french": "old_french",
    "anglo-norman": "old_french",
    "middle-french": "old_french",
    "latin": "latin",
    "biblical": "biblical",
    "celtic": "celtic_mix",
    "germanic": "germanic",
    "greek": "greek",
}


# Languages that share a bundle sibling with siblings of the same family.
# Used to render a "[shared with X, Y, Z]" annotation in the markdown
# so a per-language bundle count of 1200 isn't misread as exclusive
# (Welsh AND Irish AND Breton all see the same celtic_mix count).
_SHARED_BUNDLE_SIBLINGS: dict[str, list[str]] = {
    "celtic_mix": [
        "welsh",
        "old-welsh",
        "modern-welsh",
        "middle-welsh",
        "irish",
        "old-irish",
        "middle-irish",
        "scottish-gaelic",
        "breton",
        "celtic",
    ],
    "old_scandinavian": ["old-norse", "icelandic"],
    # ME / EModE / ModE / Scots share the modern_english sibling; the
    # dashboard surfaces the sharing so per-language bundle counts
    # aren't read as exclusive.
    "modern_english": [
        "middle-english",
        "early-modern-english",
        "modern-english",
        "scots",
    ],
    # OF / NF / Anglo-Norman / Middle French share old_french.
    "old_french": ["old-french", "norman-french", "anglo-norman", "middle-french"],
}


def _bundle_language_word_count(
    bundle: list[dict[str, Any]] | dict[str, Any], sibling: str | None
) -> int:
    """Count bundle word entries that carry a non-empty value for the
    given sibling key. Returns 0 when ``sibling`` is None (language has
    no dedicated bundle sibling). Handles both list-shaped (legacy) and
    dict-shaped bundles."""
    if sibling is None:
        return 0
    subjects = bundle if isinstance(bundle, list) else bundle.get("subjects") or []
    n = 0
    for subj in subjects:
        for word in subj.get("words") or []:
            value = word.get(sibling)
            if value:
                n += 1
    return n


def _bundle_total_words(bundle: list[dict[str, Any]] | dict[str, Any]) -> int:
    """Total word entries across all subjects (denominator for bundle %)."""
    subjects = bundle if isinstance(bundle, list) else bundle.get("subjects") or []
    return sum(len(subj.get("words") or []) for subj in subjects)


def _bundle_tag_coverage_for_sibling(
    bundle: list[dict[str, Any]] | dict[str, Any],
    sibling: str | None,
    reference_tags: list[str],
) -> dict[str, int]:
    """For each reference tag, count bundle subjects that (a) carry the
    tag in ``modifier_tags`` AND (b) contain at least one word with a
    non-empty value for ``sibling``. Returns an all-zero map when
    ``sibling`` is None (no bundle sibling for this language)."""
    counts: dict[str, int] = dict.fromkeys(reference_tags, 0)
    if sibling is None:
        return counts
    subjects = bundle if isinstance(bundle, list) else bundle.get("subjects") or []
    ref_set = set(reference_tags)
    for subj in subjects:
        tags = subj.get("modifier_tags") or []
        if not any(t in ref_set for t in tags):
            continue
        has_lang_word = any(word.get(sibling) for word in subj.get("words") or [])
        if not has_lang_word:
            continue
        for t in tags:
            if t in counts:
                counts[t] += 1
    return counts


# --- per-language metric computation -------------------------------------


def _lexicon_tag_coverage(
    conn: sqlite3.Connection,
    language: str,
    reference_tags: list[str],
) -> dict[str, Any]:
    """Metric C (lexicon side). Counts off the ``etymon_tag`` table —
    independent of bundle export, so middle-english + modern-english
    (which have no bundle sibling) still get a real reading.

    Returns:
    - ``tagged_etymons``: distinct etymons in this language with ≥1
      etymon_tag row.
    - ``tag_coverage``: ``{reference_tag: distinct_etymon_count}`` —
      hits per reference tag.
    - ``hits``: count of reference tags with ≥1 etymon (the M of M/N).
    """
    tagged_total = conn.execute(
        """
        SELECT COUNT(DISTINCT et.etymon_id)
          FROM etymon_tag et
          JOIN etymon e ON e.id = et.etymon_id
         WHERE e.language = ?
           AND e.merged_into_id IS NULL
        """,
        (language,),
    ).fetchone()[0]
    coverage: dict[str, int] = dict.fromkeys(reference_tags, 0)
    if reference_tags:
        placeholders = ",".join("?" * len(reference_tags))
        rows = conn.execute(
            f"""
            SELECT et.tag, COUNT(DISTINCT et.etymon_id)
              FROM etymon_tag et
              JOIN etymon e ON e.id = et.etymon_id
             WHERE e.language = ?
               AND e.merged_into_id IS NULL
               AND et.tag IN ({placeholders})
             GROUP BY et.tag
            """,
            (language, *reference_tags),
        ).fetchall()
        for tag, count in rows:
            coverage[tag] = count
    hits = sum(1 for v in coverage.values() if v > 0)
    return {
        "lexicon_tagged_etymons": tagged_total,
        "lexicon_tag_coverage": coverage,
        "lexicon_tag_hits": hits,
    }


def _corpus_depth(conn: sqlite3.Connection, language: str, threshold: int) -> dict[str, Any]:
    """Metric A. Counts off the etymon table + etymon_consensus view."""
    total_etymons = conn.execute(
        "SELECT COUNT(*) FROM etymon WHERE language = ? AND merged_into_id IS NULL",
        (language,),
    ).fetchone()[0]
    total_lemmas = conn.execute(
        """
        SELECT COUNT(*) FROM etymon
         WHERE language = ?
           AND merged_into_id IS NULL
           AND lemma_id IS NULL
        """,
        (language,),
    ).fetchone()[0]
    consensus_rows = conn.execute(
        "SELECT witnesses FROM etymon_consensus WHERE language = ?",
        (language,),
    ).fetchall()
    if consensus_rows:
        avg_witnesses = sum(r[0] for r in consensus_rows) / len(consensus_rows)
        promotion_eligible = sum(1 for r in consensus_rows if r[0] >= threshold)
    else:
        avg_witnesses = 0.0
        promotion_eligible = 0
    source_count = conn.execute(
        """
        SELECT COUNT(DISTINCT c.source_id)
          FROM etymon_citation c
          JOIN etymon e ON e.id = c.etymon_id
         WHERE e.language = ?
        """,
        (language,),
    ).fetchone()[0]
    return {
        "total_etymons": total_etymons,
        "total_lemmas": total_lemmas,
        "promotion_threshold": threshold,
        "promotion_eligible": promotion_eligible,
        "avg_witnesses": round(avg_witnesses, 3),
        "source_count": source_count,
    }


def _inflection_coverage(conn: sqlite3.Connection, language: str) -> dict[str, Any]:
    """Metric D. Lemmas with linked inflected children + distinct labels."""
    # Lemmas (etymon.lemma_id IS NULL & inflection IS NULL) where another
    # etymon links lemma_id back to this one.
    lemmas_with_children = conn.execute(
        """
        SELECT COUNT(*) FROM etymon parent
         WHERE parent.language = ?
           AND parent.merged_into_id IS NULL
           AND parent.lemma_id IS NULL
           AND EXISTS (
             SELECT 1 FROM etymon child
              WHERE child.lemma_id = parent.id
                AND child.merged_into_id IS NULL
                AND child.inflection IS NOT NULL
           )
        """,
        (language,),
    ).fetchone()[0]
    distinct_labels = conn.execute(
        """
        SELECT COUNT(DISTINCT inflection)
          FROM etymon
         WHERE language = ?
           AND merged_into_id IS NULL
           AND inflection IS NOT NULL
        """,
        (language,),
    ).fetchone()[0]
    return {
        "lemmas_with_inflections": lemmas_with_children,
        "distinct_inflection_labels": distinct_labels,
    }


def _variant_coverage(conn: sqlite3.Connection, language: str) -> int:
    """Metric E. Etymons with at least one matched_form != canonical_form."""
    return conn.execute(
        """
        SELECT COUNT(DISTINCT etm.etymon_id)
          FROM etymon_text_match etm
          JOIN etymon e ON e.id = etm.etymon_id
         WHERE e.language = ?
           AND e.merged_into_id IS NULL
           AND etm.matched_form != e.canonical_form
        """,
        (language,),
    ).fetchone()[0]


def _era_reflex_coverage(conn: sqlite3.Connection, language: str) -> dict[str, Any]:
    """Metric F. Tier 1 / 2-fwd / 2-back / 3 counts per the chain."""
    chain = ERA_CHAINS.get(language, {})
    older = chain.get("older", [])
    younger = chain.get("younger", [])
    chain_known = bool(chain)

    # Tier 1: etymon has cognate_id pointing to an etymon in a different
    # language. (Cognate cluster mate in another era / family.)
    tier1 = conn.execute(
        """
        SELECT COUNT(*)
          FROM etymon e
          JOIN etymon mate ON mate.id = e.cognate_id
         WHERE e.language = ?
           AND e.merged_into_id IS NULL
           AND mate.language != e.language
        """,
        (language,),
    ).fetchone()[0]

    # Tier 2 forward: this etymon is parent of an edge whose child is in
    # a "younger" language per chain config. Empty chain.younger means
    # terminus-forward — count is structurally 0.
    if younger:
        placeholders = ",".join("?" * len(younger))
        tier2_forward = conn.execute(
            f"""
            SELECT COUNT(DISTINCT ed.parent_id)
              FROM etymon_descent ed
              JOIN etymon parent ON parent.id = ed.parent_id
              JOIN etymon child ON child.id = ed.child_id
             WHERE parent.language = ?
               AND parent.merged_into_id IS NULL
               AND child.language IN ({placeholders})
            """,
            (language, *younger),
        ).fetchone()[0]
    else:
        tier2_forward = 0

    # Tier 2 back: this etymon is child of an edge whose parent is in an
    # "older" language. Empty chain.older = terminus-back.
    if older:
        placeholders = ",".join("?" * len(older))
        tier2_back = conn.execute(
            f"""
            SELECT COUNT(DISTINCT ed.child_id)
              FROM etymon_descent ed
              JOIN etymon parent ON parent.id = ed.parent_id
              JOIN etymon child ON child.id = ed.child_id
             WHERE child.language = ?
               AND child.merged_into_id IS NULL
               AND parent.language IN ({placeholders})
            """,
            (language, *older),
        ).fetchone()[0]
    else:
        tier2_back = 0

    # Tier 3: period-form projection rows.
    tier3 = conn.execute(
        """
        SELECT COUNT(DISTINCT pf.etymon_id)
          FROM etymon_period_form pf
          JOIN etymon e ON e.id = pf.etymon_id
         WHERE e.language = ?
           AND e.merged_into_id IS NULL
        """,
        (language,),
    ).fetchone()[0]

    # ANY reflex: union over the four tiers, distinct etymon ids.
    any_reflex = conn.execute(
        f"""
        SELECT COUNT(DISTINCT e.id)
          FROM etymon e
         WHERE e.language = ?
           AND e.merged_into_id IS NULL
           AND (
             EXISTS (
               SELECT 1 FROM etymon mate
                WHERE mate.id = e.cognate_id AND mate.language != e.language
             )
             {
            "OR EXISTS (SELECT 1 FROM etymon_descent ed JOIN etymon c ON c.id = ed.child_id "
            f"WHERE ed.parent_id = e.id AND c.language IN ({','.join('?' * len(younger))}))"
            if younger
            else ""
        }
             {
            "OR EXISTS (SELECT 1 FROM etymon_descent ed JOIN etymon p ON p.id = ed.parent_id "
            f"WHERE ed.child_id = e.id AND p.language IN ({','.join('?' * len(older))}))"
            if older
            else ""
        }
             OR EXISTS (
               SELECT 1 FROM etymon_period_form pf
                WHERE pf.etymon_id = e.id
             )
           )
        """,
        (language, *younger, *older),
    ).fetchone()[0]

    return {
        "chain_older": list(older),
        "chain_younger": list(younger),
        "chain_known": chain_known,
        "tier1_cognate_count": tier1,
        "tier2_forward_count": tier2_forward,
        "tier2_back_count": tier2_back,
        "tier3_period_form_count": tier3,
        "any_reflex_count": any_reflex,
    }


def _stratum_coverage(conn: sqlite3.Connection, language: str) -> int:
    """Metric G. Etymons with non-NULL stratum (D32 classifier output)."""
    return conn.execute(
        """
        SELECT COUNT(*) FROM etymon
         WHERE language = ?
           AND merged_into_id IS NULL
           AND stratum IS NOT NULL
        """,
        (language,),
    ).fetchone()[0]


def _citation_depth(conn: sqlite3.Connection, language: str) -> dict[str, Any]:
    """Metric I. Etymons with ≥1 citation + average citations per etymon."""
    cited_etymons = conn.execute(
        """
        SELECT COUNT(DISTINCT c.etymon_id)
          FROM etymon_citation c
          JOIN etymon e ON e.id = c.etymon_id
         WHERE e.language = ?
           AND e.merged_into_id IS NULL
        """,
        (language,),
    ).fetchone()[0]
    total_citations = conn.execute(
        """
        SELECT COUNT(*)
          FROM etymon_citation c
          JOIN etymon e ON e.id = c.etymon_id
         WHERE e.language = ?
           AND e.merged_into_id IS NULL
        """,
        (language,),
    ).fetchone()[0]
    return {
        "etymons_with_citations": cited_etymons,
        "avg_citations": round(total_citations / cited_etymons, 3) if cited_etymons else 0.0,
    }


# --- top-level scorecard --------------------------------------------------


def compute_scorecard(
    conn: sqlite3.Connection,
    language: str,
    bundle: list[dict[str, Any]] | dict[str, Any],
    reference_tags: list[str],
    *,
    default_threshold: int = 3,
    lang_thresholds: dict[str, int] | None = None,
) -> LanguageScorecard:
    """Compute the full scorecard for one language."""
    if lang_thresholds is None:
        lang_thresholds = RECOMMENDED_LANG_THRESHOLDS
    threshold = lang_thresholds.get(language, default_threshold)

    depth = _corpus_depth(conn, language, threshold)
    sibling = _BUNDLE_LANG_KEY.get(language)
    bundle_words = _bundle_language_word_count(bundle, sibling)
    bundle_tag_counts = _bundle_tag_coverage_for_sibling(bundle, sibling, reference_tags)
    shared_with: list[str] = []
    if sibling is not None and sibling in _SHARED_BUNDLE_SIBLINGS:
        shared_with = [other for other in _SHARED_BUNDLE_SIBLINGS[sibling] if other != language]
    bundle_tags_hit = sum(1 for v in bundle_tag_counts.values() if v > 0)
    bundle_tag_hit_rate = bundle_tags_hit / len(reference_tags) if reference_tags else 0.0
    lex_tag = _lexicon_tag_coverage(conn, language, reference_tags)
    lex_tag_hit_rate = lex_tag["lexicon_tag_hits"] / len(reference_tags) if reference_tags else 0.0
    inflect = _inflection_coverage(conn, language)
    variants = _variant_coverage(conn, language)
    era = _era_reflex_coverage(conn, language)
    stratum = _stratum_coverage(conn, language)
    citations = _citation_depth(conn, language)

    total_lemmas = depth["total_lemmas"]
    total_etymons = depth["total_etymons"]
    return LanguageScorecard(
        language=language,
        total_etymons=total_etymons,
        total_lemmas=total_lemmas,
        promotion_threshold=threshold,
        promotion_eligible=depth["promotion_eligible"],
        avg_witnesses=depth["avg_witnesses"],
        source_count=depth["source_count"],
        bundle_sibling=sibling,
        bundle_word_count=bundle_words,
        bundle_shared_with=shared_with,
        reference_tags=list(reference_tags),
        lexicon_tagged_etymons=lex_tag["lexicon_tagged_etymons"],
        lexicon_tag_coverage=lex_tag["lexicon_tag_coverage"],
        lexicon_tag_hit_rate=round(lex_tag_hit_rate, 3),
        bundle_tag_coverage=bundle_tag_counts,
        bundle_tag_hit_rate=round(bundle_tag_hit_rate, 3),
        lemmas_with_inflections=inflect["lemmas_with_inflections"],
        inflection_density=(
            round(inflect["lemmas_with_inflections"] / total_lemmas, 4) if total_lemmas else 0.0
        ),
        distinct_inflection_labels=inflect["distinct_inflection_labels"],
        etymons_with_variants=variants,
        variant_density=(round(variants / total_etymons, 4) if total_etymons else 0.0),
        chain_older=era["chain_older"],
        chain_younger=era["chain_younger"],
        chain_known=era["chain_known"],
        tier1_cognate_count=era["tier1_cognate_count"],
        tier2_forward_count=era["tier2_forward_count"],
        tier2_back_count=era["tier2_back_count"],
        tier3_period_form_count=era["tier3_period_form_count"],
        any_reflex_count=era["any_reflex_count"],
        any_reflex_rate=(
            round(era["any_reflex_count"] / total_etymons, 4) if total_etymons else 0.0
        ),
        stratum_classified=stratum,
        stratum_density=(round(stratum / total_etymons, 4) if total_etymons else 0.0),
        etymons_with_citations=citations["etymons_with_citations"],
        avg_citations=citations["avg_citations"],
    )


def compute_report(
    conn: sqlite3.Connection,
    bundle: list[dict[str, Any]] | dict[str, Any],
    *,
    languages: list[str] | None = None,
    reference_tags: list[str] | None = None,
    default_threshold: int = 3,
    lang_thresholds: dict[str, int] | None = None,
    drop_empty: bool = True,
) -> LanguageQualityReport:
    """Compute the full multi-language report.

    Languages with zero etymons are dropped when ``drop_empty=True``
    (Phase-1 default — every-zero scorecards are noise on the human
    output and don't help mining decisions). Set
    ``drop_empty=False`` for trend tracking where the absence is
    informative."""
    languages_to_check = list(languages or DEFAULT_LANGUAGES)
    if reference_tags is None:
        reference_tags = list(FALLBACK_REFERENCE_TAGS[:REFERENCE_TAG_COUNT])
    cards: list[LanguageScorecard] = []
    for lang in languages_to_check:
        card = compute_scorecard(
            conn,
            lang,
            bundle,
            reference_tags,
            default_threshold=default_threshold,
            lang_thresholds=lang_thresholds,
        )
        if drop_empty and card.total_etymons == 0:
            continue
        cards.append(card)
    return LanguageQualityReport(
        schema_version="1.0",
        generated_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        reference_tags=list(reference_tags),
        bundle_total_words=_bundle_total_words(bundle),
        languages=cards,
    )


# --- formatters -----------------------------------------------------------


def report_to_json(report: LanguageQualityReport) -> str:
    """JSON snapshot — schema-versioned, ISO-timestamped, full per-language
    metric set. Persistable for trend tracking."""
    return json.dumps(asdict(report), indent=2, sort_keys=False)


def _format_pct(num: int, denom: int) -> str:
    if not denom:
        return "0.0%"
    return f"{100 * num / denom:.1f}%"


def _format_chain(scorecard: LanguageScorecard) -> str:
    if not scorecard.chain_known:
        return "(no chain config)"
    older = ", ".join(scorecard.chain_older) if scorecard.chain_older else "terminus-back"
    younger = ", ".join(scorecard.chain_younger) if scorecard.chain_younger else "terminus-forward"
    return f"older=[{older}] younger=[{younger}]"


def report_to_markdown(report: LanguageQualityReport) -> str:
    """Human-readable per-language scorecard. Summary table at top,
    then one detail block per language. Section letters (A → I) match
    the dataclass docstring + ticket scope."""
    lines: list[str] = []
    lines.append(f"# Language-quality scorecard — {report.generated_at}")
    lines.append("")
    lines.append(f"Schema version: {report.schema_version}")
    lines.append(f"Reference profile (top {len(report.reference_tags)} English tags):")
    lines.append("  " + ", ".join(report.reference_tags))
    lines.append(f"Bundle total words: {report.bundle_total_words}")
    lines.append("")

    # --- summary table ---
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| Language | Etymons | Lemmas | Promo (≥thr) | Bundle | Lex tag hit | Bundle tag hit | Inflect | Variants | Era reflex |"
    )
    lines.append("|---|---:|---:|---|---:|---|---|---|---|---|")
    for c in report.languages:
        promo = f"{c.promotion_eligible} (≥{c.promotion_threshold})"
        bundle_pct = _format_pct(c.bundle_word_count, report.bundle_total_words)
        bundle_cell = f"{c.bundle_word_count} ({bundle_pct})"
        ref_n = len(c.reference_tags)
        lex_hit = f"{int(round(c.lexicon_tag_hit_rate * ref_n))}/{ref_n}"
        bundle_hit = f"{int(round(c.bundle_tag_hit_rate * ref_n))}/{ref_n}"
        inflect = f"{c.lemmas_with_inflections} ({_format_pct(c.lemmas_with_inflections, c.total_lemmas)})"
        variants = (
            f"{c.etymons_with_variants} ({_format_pct(c.etymons_with_variants, c.total_etymons)})"
        )
        era = f"{c.any_reflex_count} ({_format_pct(c.any_reflex_count, c.total_etymons)})"
        lines.append(
            f"| {c.language} | {c.total_etymons} | {c.total_lemmas} | {promo} | "
            f"{bundle_cell} | {lex_hit} | {bundle_hit} | {inflect} | {variants} | {era} |"
        )
    lines.append("")

    # --- per-language detail ---
    lines.append("## Per-language detail")
    lines.append("")
    for c in report.languages:
        lines.append(f"### {c.language}")
        lines.append(f"_chain: {_format_chain(c)}_")
        lines.append("")
        lines.append(
            f"- **A. Corpus depth:** {c.total_etymons} etymons "
            f"({c.total_lemmas} lemmas), "
            f"{c.promotion_eligible} promotion-eligible at threshold ≥{c.promotion_threshold}, "
            f"avg {c.avg_witnesses} witnesses across {c.source_count} sources."
        )
        if c.bundle_sibling is None:
            bundle_line = (
                "- **B. Bundle representation:** no dedicated bundle sibling for this language."
            )
        else:
            bundle_line = (
                f"- **B. Bundle representation:** {c.bundle_word_count} words "
                f"({_format_pct(c.bundle_word_count, report.bundle_total_words)} of "
                f"{report.bundle_total_words} bundle words; sibling=`{c.bundle_sibling}`"
            )
            if c.bundle_shared_with:
                bundle_line += f"; **shared with** {', '.join(c.bundle_shared_with)}"
            bundle_line += ")."
        lines.append(bundle_line)
        lex_hit = sum(1 for v in c.lexicon_tag_coverage.values() if v > 0)
        lex_missing = [t for t, v in c.lexicon_tag_coverage.items() if v == 0]
        lex_miss_str = ", ".join(lex_missing) if lex_missing else "none"
        lines.append(
            f"- **C. Semantic-tag coverage (lexicon):** {c.lexicon_tagged_etymons}/"
            f"{c.total_etymons} etymons "
            f"({_format_pct(c.lexicon_tagged_etymons, c.total_etymons)}) "
            f"carry ≥1 tag. Reference-tag hits: {lex_hit}/{len(c.reference_tags)} "
            f"({_format_pct(lex_hit, len(c.reference_tags))}). Missing: {lex_miss_str}."
        )
        bundle_hit = sum(1 for v in c.bundle_tag_coverage.values() if v > 0)
        if c.bundle_sibling is None:
            bundle_tag_line = (
                "- **C₂. Bundle tag visibility:** n/a — language has no dedicated bundle sibling."
            )
        else:
            bundle_missing = [t for t, v in c.bundle_tag_coverage.items() if v == 0]
            bundle_miss_str = ", ".join(bundle_missing) if bundle_missing else "none"
            bundle_tag_line = (
                f"- **C₂. Bundle tag visibility:** {bundle_hit}/{len(c.reference_tags)} "
                f"({_format_pct(bundle_hit, len(c.reference_tags))}). Missing: {bundle_miss_str}."
            )
        lines.append(bundle_tag_line)
        lines.append(
            f"- **D. Inflection coverage:** {c.lemmas_with_inflections}/{c.total_lemmas} "
            f"lemmas ({_format_pct(c.lemmas_with_inflections, c.total_lemmas)}) "
            f"with linked inflections; {c.distinct_inflection_labels} distinct case labels."
        )
        lines.append(
            f"- **E. Spelling-variant coverage:** {c.etymons_with_variants}/{c.total_etymons} "
            f"etymons ({_format_pct(c.etymons_with_variants, c.total_etymons)})."
        )
        # F. era-reflex
        forward_label = f"{c.tier2_forward_count}" if c.chain_younger else "n/a (terminus-forward)"
        back_label = f"{c.tier2_back_count}" if c.chain_older else "n/a (terminus-back)"
        lines.append(
            f"- **F. Era-reflex coverage:** Tier 1 cognate {c.tier1_cognate_count}, "
            f"Tier 2 forward {forward_label}, Tier 2 back {back_label}, "
            f"Tier 3 period-form {c.tier3_period_form_count}; "
            f"ANY reflex {c.any_reflex_count} "
            f"({_format_pct(c.any_reflex_count, c.total_etymons)})."
        )
        lines.append(
            f"- **G. Stratum coverage:** {c.stratum_classified}/{c.total_etymons} "
            f"({_format_pct(c.stratum_classified, c.total_etymons)})."
        )
        lines.append(
            f"- **I. Citation depth:** {c.etymons_with_citations}/{c.total_etymons} "
            f"({_format_pct(c.etymons_with_citations, c.total_etymons)}) cited; "
            f"avg {c.avg_citations} citations per cited etymon."
        )
        lines.append("")
    return "\n".join(lines)
