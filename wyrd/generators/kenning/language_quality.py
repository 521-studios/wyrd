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


# wyrd-lc94: source-IDs that admit a family root via paths OTHER than
# scholarly witness consensus. The bundle attestation breakdown bins
# each shipped subject by which path it relied on so the dashboard can
# surface the actionable runtime-quality signal alongside the existing
# DB-level metrics. See ``_bundle_attestation_breakdown`` for the
# classification rules.
_GRANDFATHER_CITATION_SOURCES: frozenset[str] = frozenset({"rando-port"})
_EMPIRICAL_CITATION_SOURCES: frozenset[str] = frozenset({"wiktionary-empirical"})


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
    # H. Pronunciation / IPA coverage (wyrd-dxu2)
    # Two parallel views, mirroring the DB-vs-bundle split used for tag
    # coverage (C / C₂) and attestation (B₂):
    #   * Lexicon side: how many etymons in this language have a non-null
    #     ``pronunciation_ipa`` column. Surfaces the DB-curation signal
    #     ('how much of the lexicon has IPA at all').
    #   * Bundle side: how many shipped subjects in this language have a
    #     non-empty ``*_pronunciation`` slot. Surfaces the runtime-quality
    #     signal ('what % of what we ship can be rendered with IPA in
    #     the SPA's etymological-provenance panel').
    # The two ARE NOT THE SAME — the bundle-side number can include IPA
    # inherited from cluster-mate languages (e.g. modern_english forms
    # that share a surface with a middle-english member with IPA), so
    # the bundle figure overstates true per-language coverage. Both are
    # useful: the DB number is the cleanest 'do we have IPA for this
    # language?' signal, the bundle number is what users actually see.
    lexicon_etymons_with_ipa: int = 0
    lexicon_ipa_density: float = 0.0
    bundle_subjects_with_ipa: int = 0
    bundle_ipa_rate: float = 0.0
    # I. Citation depth
    etymons_with_citations: int = 0
    avg_citations: float = 0.0
    # J. Bundle attestation breakdown (wyrd-lc94)
    # Per-subject classification of how each shipped subject containing
    # this language reached the bundle. Surfaces the runtime-quality
    # signal that DB-level corpus-depth metrics don't. See
    # ``_bundle_attestation_breakdown`` for classification rules.
    bundle_attestation_total: int = 0
    bundle_scholar_attested: int = 0
    bundle_empirical_only: int = 0
    bundle_rando_only: int = 0
    bundle_uncited: int = 0
    bundle_scholar_attestation_rate: float = 0.0
    # K. Rando-port grandfather audit (wyrd-vn09)
    # Family-level (etymon-rollup) classification of which morpheme
    # families ride solely on the rando-port legacy seed for admission
    # vs which are corroborated through the cluster. Distinct from J's
    # bundle-attestation reading because that one operated on the
    # bundle's per-word citation list, which propagates scholar
    # attestation across cluster siblings. THIS metric operates on the
    # raw etymon_citation table at the family level — the cleanest
    # answer to 'how many morphemes would lose their bundle slot if we
    # turned off include_rando=True'.
    #
    # ``rando_pure_grandfather_families``: families where every cited
    # etymon's only citation is 'rando-port'. Curation candidates.
    # ``rando_mixed_grandfather_families``: families with rando-port +
    # scholar/empirical on some sibling. Already corroborated through
    # the cluster — safe to keep, useful as a trend indicator (count
    # should drift down as mining propagates).
    # ``rando_total_cited_families``: denominator (families whose root
    # is in this language, with at least one citation anywhere).
    rando_pure_grandfather_families: int = 0
    rando_mixed_grandfather_families: int = 0
    rando_total_cited_families: int = 0
    rando_pure_grandfather_rate: float = 0.0


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
    subjects = _bundle_subjects(bundle)
    n = 0
    for subj in subjects:
        for word in subj.get("words") or []:
            value = word.get(sibling)
            if value:
                n += 1
    return n


def _bundle_subjects(bundle: Any) -> list[dict[str, Any]]:
    """Defensive subjects-extraction shared by every bundle helper.

    Returns the list-shaped subjects array from either:

    * a list-shaped bundle (legacy pre-wyrd-c1vq) — returns it directly,
    * a dict-shaped bundle with a ``subjects`` list — returns the list,
    * any other shape (None, malformed, non-list 'subjects' value) —
      returns an empty list.

    The defensive return path catches malformed bundle inputs that
    would otherwise raise ``AttributeError`` (None) or ``TypeError``
    (truthy non-list ``subjects``). Cheaper to handle once here than
    to repeat the isinstance dance in every helper.
    """
    if isinstance(bundle, list):
        return bundle
    if isinstance(bundle, dict):
        subjects = bundle.get("subjects")
        if isinstance(subjects, list):
            return subjects
    return []


def _bundle_total_words(bundle: list[dict[str, Any]] | dict[str, Any]) -> int:
    """Total word entries across all subjects (denominator for bundle %)."""
    return sum(len(subj.get("words") or []) for subj in _bundle_subjects(bundle))


def _bundle_attestation_breakdown(
    bundle: list[dict[str, Any]] | dict[str, Any],
    sibling: str | None,
) -> dict[str, int]:
    """wyrd-lc94: classify every bundle subject containing the language
    sibling by which admission path landed it in the bundle.

    Subjects are binned (priority order, scholar wins ties):

    * ``scholar_attested`` — at least one citation that's NOT
      ``rando-port`` and NOT ``wiktionary-empirical``. The actionable
      'real-attestation' class — these are subjects backed by named
      scholarly sources.
    * ``empirical_only`` — only ``wiktionary-empirical`` citations.
      Admitted via the wyrd-4hx7 corpus-mining path (Wiktionary
      headwords matched against unaccounted-fragment misses); not
      scholar-attested per se but at least confirmed by an empirical
      Wiktionary lookup.
    * ``rando_only`` — only ``rando-port`` citations. The legacy
      hand-curated grandfather class (export_meanings'
      ``include_rando=True`` path). These haven't accumulated
      independent attestation through mining yet.
    * ``uncited`` — language sibling present but no citation field
      (older bundles or coverage paths that didn't carry attribution).

    Returns a dict with ``total`` (subjects in this language) plus the
    four bin counts. Empty (all zeros) when ``sibling is None``.

    Used by the dashboard to surface 'what % of what we ship has
    independent scholarly backing vs grandfathered' — the runtime-
    quality signal that DB-level depth metrics don't capture.
    """
    counts = {
        "total": 0,
        "scholar_attested": 0,
        "empirical_only": 0,
        "rando_only": 0,
        "uncited": 0,
    }
    if sibling is None:
        return counts
    citation_key = f"{sibling}_citations"
    subjects = _bundle_subjects(bundle)
    for subj in subjects:
        # Classify per-subject (not per-word): if any word in the
        # subject has a scholar citation, the subject counts as
        # scholar-attested. Treats subject-level admission as the
        # natural unit since one subject = one (meaning, tags) family.
        has_lang = False
        scholar = False
        empirical = False
        rando = False
        for word in subj.get("words") or []:
            if not word.get(sibling):
                continue
            has_lang = True
            for c in word.get(citation_key) or []:
                if c in _GRANDFATHER_CITATION_SOURCES:
                    rando = True
                elif c in _EMPIRICAL_CITATION_SOURCES:
                    empirical = True
                else:
                    scholar = True
                    break  # inner break — scholar locks classification
            if scholar:
                # Outer break too — has_lang is already True (set above
                # for the current word) and scholar wins all ties, so
                # examining additional words can't change the bin.
                break
        if not has_lang:
            continue
        counts["total"] += 1
        if scholar:
            counts["scholar_attested"] += 1
        elif empirical:
            counts["empirical_only"] += 1
        elif rando:
            counts["rando_only"] += 1
        else:
            counts["uncited"] += 1
    return counts


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
    subjects = _bundle_subjects(bundle)
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


def _lexicon_ipa_coverage(conn: sqlite3.Connection, language: str) -> int:
    """Metric H (lexicon side). Count of etymons in this language with
    non-null ``pronunciation_ipa``. Excludes merged-away rows (consensus
    view convention). The DB column is populated by
    ``wiktextract_ingester._process_entry`` and (post-wyrd-69s5)
    ``wiktextract_corpus_miner._write_one``; absent on languages
    whose wiktextract slice doesn't carry ``sounds`` arrays."""
    return conn.execute(
        """
        SELECT COUNT(*) FROM etymon
         WHERE language = ?
           AND merged_into_id IS NULL
           AND pronunciation_ipa IS NOT NULL
           AND pronunciation_ipa != ''
        """,
        (language,),
    ).fetchone()[0]


def _bundle_ipa_coverage(
    bundle: list[dict[str, Any]] | dict[str, Any],
    sibling: str | None,
) -> int:
    """Metric H (bundle side). Count of bundle subjects whose ``sibling``
    is populated AND carry a ``<sibling>_pronunciation`` entry with at
    least one non-empty IPA string. Caveat: this OVER-counts true
    per-language IPA coverage because the bundle export's family rollup
    can attach a cluster-mate's IPA to a sibling slot when the canonical
    forms match (e.g. modern-english 'brede' inherits middle-english
    'brede's IPA). Compare with ``_lexicon_ipa_coverage`` to detect
    the cross-language inheritance — a 0% lexicon-side reading paired
    with a non-zero bundle-side reading flags a language whose bundle
    IPA is entirely inherited from cluster mates."""
    if sibling is None:
        return 0
    pronunciation_key = f"{sibling}_pronunciation"
    n = 0
    for subj in _bundle_subjects(bundle):
        for word in subj.get("words") or []:
            if not word.get(sibling):
                continue
            pron_list = word.get(pronunciation_key) or []
            if any(p.get("ipa") for p in pron_list if isinstance(p, dict)):
                n += 1
                break  # Per-subject count, not per-word.
    return n


def compute_rando_port_grandfather_audit(
    conn: sqlite3.Connection,
) -> dict[str, dict[str, int]]:
    """wyrd-vn09: classify every citation-bearing family in the lexicon
    by whether it depends solely on the rando-port legacy seed for
    admission, and bucket the results by the family root's language.

    The lc94 dashboard surfaced 0% rando-port-only at the bundle level,
    because the bundle's per-word citation list is the union across the
    family — any scholar citation on any cluster sibling propagates to
    every member. The actionable curation question is at the etymon-
    family level: which morpheme families would lose their bundle slot
    if ``include_rando=True`` were flipped off?

    Three classes per family (citation-bearing only — uncited families
    are reported separately):

    * ``pure_grandfather`` — the entire family's only citations are
      ``rando-port``. Admission rides solely on the legacy hand-curated
      seed. Curation candidates: spot-check, look for typos /
      transcription errors / spurious entries, target for empirical
      mining corroboration.
    * ``mixed_grandfather`` — the family has rando-port citations AND
      at least one scholar / wiktionary-empirical citation on some
      sibling. The legacy seed is corroborated through cluster — these
      are SAFE to keep but interesting to track for trend purposes
      (over time the count should drift down as mining propagates).
    * ``no_grandfather`` (implicit, not returned per-language) — the
      family has citations but none from rando-port; all-modern
      attestation, not part of the audit.

    Returns ``{language: {pure_grandfather: int, mixed_grandfather:
    int, total_families: int}}``. ``total_families`` is the count of
    citation-bearing families whose root is in this language —
    denominator for grandfather-density rates.

    Family rollup matches ``lexicon._build_family_rollup``: target =
    merged_into_id OR lemma_id OR self; root = lemma_id of target OR
    target itself. So OCR-cluster losers and inflected children both
    surface their lemma as root. Computed once per report (caller
    threads the result into ``compute_scorecard``) — the rollup walks
    every etymon row, so per-language recomputation would be wasteful.
    """
    # Single-pass scan of the ~860K-row etymon table to populate both
    # the rollup maps AND the language lookup. Integer indices instead
    # of row_factory-dependent column names so the function is robust
    # to callers whose connection doesn't set sqlite3.Row.
    target_by_id: dict[int, int] = {}
    lemma_by_id: dict[int, int | None] = {}
    lang_by_id: dict[int, str] = {}
    for r in conn.execute("SELECT id, merged_into_id, lemma_id, language FROM etymon"):
        eid, merged, lemma, lang = r[0], r[1], r[2], r[3]
        target_by_id[eid] = merged or lemma or eid
        lemma_by_id[eid] = lemma
        lang_by_id[eid] = lang

    def root_of(eid: int) -> int:
        target = target_by_id.get(eid, eid)
        return lemma_by_id.get(target) or target

    family_members: dict[int, list[int]] = {}
    for eid in target_by_id:
        family_members.setdefault(root_of(eid), []).append(eid)

    # Citation map: etymon_id → set of source_ids. Same integer-index
    # robustness as above.
    cites_by_etymon: dict[int, set[str]] = {}
    for r in conn.execute("SELECT etymon_id, source_id FROM etymon_citation"):
        cites_by_etymon.setdefault(r[0], set()).add(r[1])

    audit: dict[str, dict[str, int]] = {}
    for root_id, members in family_members.items():
        family_sources: set[str] = set()
        for mid in members:
            # ``update`` with the get-default-tuple avoids the per-miss
            # empty-set allocation that ``|= cites_by_etymon.get(mid,
            # set())`` triggered — ~860K families × member loop adds
            # up under GC pressure.
            family_sources.update(cites_by_etymon.get(mid, ()))
        if not family_sources:
            continue  # uncited family — reported elsewhere (citation depth metric I)
        lang = lang_by_id.get(root_id)
        if lang is None:
            continue
        bucket = audit.setdefault(
            lang, {"pure_grandfather": 0, "mixed_grandfather": 0, "total_families": 0}
        )
        bucket["total_families"] += 1
        # Use the same ``_GRANDFATHER_CITATION_SOURCES`` constant as
        # ``_bundle_attestation_breakdown`` so changes to the
        # grandfather source list propagate to both audits.
        if family_sources == _GRANDFATHER_CITATION_SOURCES:
            bucket["pure_grandfather"] += 1
        elif family_sources & _GRANDFATHER_CITATION_SOURCES:
            bucket["mixed_grandfather"] += 1
    return audit


# --- top-level scorecard --------------------------------------------------


def compute_scorecard(
    conn: sqlite3.Connection,
    language: str,
    bundle: list[dict[str, Any]] | dict[str, Any],
    reference_tags: list[str],
    *,
    default_threshold: int = 3,
    lang_thresholds: dict[str, int] | None = None,
    rando_port_audit: dict[str, dict[str, int]] | None = None,
) -> LanguageScorecard:
    """Compute the full scorecard for one language.

    ``rando_port_audit`` (wyrd-vn09, optional) is the precomputed
    family-level grandfather classification from
    ``compute_rando_port_grandfather_audit``. Passed in rather than
    recomputed because the rollup walks all ~900K etymons. ``None``
    skips the K. fields (they stay at default 0).
    """
    if lang_thresholds is None:
        lang_thresholds = RECOMMENDED_LANG_THRESHOLDS
    threshold = lang_thresholds.get(language, default_threshold)

    depth = _corpus_depth(conn, language, threshold)
    sibling = _BUNDLE_LANG_KEY.get(language)
    bundle_words = _bundle_language_word_count(bundle, sibling)
    bundle_tag_counts = _bundle_tag_coverage_for_sibling(bundle, sibling, reference_tags)
    attestation = _bundle_attestation_breakdown(bundle, sibling)
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
    lex_ipa = _lexicon_ipa_coverage(conn, language)
    bundle_ipa = _bundle_ipa_coverage(bundle, sibling)
    rando_bucket = (rando_port_audit or {}).get(
        language, {"pure_grandfather": 0, "mixed_grandfather": 0, "total_families": 0}
    )

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
        bundle_attestation_total=attestation["total"],
        bundle_scholar_attested=attestation["scholar_attested"],
        bundle_empirical_only=attestation["empirical_only"],
        bundle_rando_only=attestation["rando_only"],
        bundle_uncited=attestation["uncited"],
        bundle_scholar_attestation_rate=(
            round(attestation["scholar_attested"] / attestation["total"], 4)
            if attestation["total"]
            else 0.0
        ),
        lexicon_etymons_with_ipa=lex_ipa,
        lexicon_ipa_density=(round(lex_ipa / total_etymons, 4) if total_etymons else 0.0),
        bundle_subjects_with_ipa=bundle_ipa,
        bundle_ipa_rate=(
            round(bundle_ipa / attestation["total"], 4) if attestation["total"] else 0.0
        ),
        rando_pure_grandfather_families=rando_bucket["pure_grandfather"],
        rando_mixed_grandfather_families=rando_bucket["mixed_grandfather"],
        rando_total_cited_families=rando_bucket["total_families"],
        rando_pure_grandfather_rate=(
            round(rando_bucket["pure_grandfather"] / rando_bucket["total_families"], 4)
            if rando_bucket["total_families"]
            else 0.0
        ),
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
    # wyrd-vn09: precompute the rando-port grandfather audit ONCE
    # (rollup walks every etymon row, so per-language recomputation
    # would be wasteful). Each scorecard reads its bucket from the
    # precomputed map.
    rando_port_audit = compute_rando_port_grandfather_audit(conn)
    cards: list[LanguageScorecard] = []
    for lang in languages_to_check:
        card = compute_scorecard(
            conn,
            lang,
            bundle,
            reference_tags,
            default_threshold=default_threshold,
            lang_thresholds=lang_thresholds,
            rando_port_audit=rando_port_audit,
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
        "| Language | Etymons | Lemmas | Promo (≥thr) | Bundle | "
        "Scholar atst | IPA (db / bundle) | Grandfather | Lex tag hit | "
        "Bundle tag hit | Inflect | Variants | Era reflex |"
    )
    lines.append("|---|---:|---:|---|---:|---:|---|---:|---|---|---|---|---|")
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
        # Scholar-attestation column: scholar / total bundle subjects
        # for this language. Renders 'n/a' when language has no bundle
        # representation (denominator zero) so a dashboard reader can
        # tell 'absent' apart from '0% attested'.
        if c.bundle_attestation_total == 0:
            scholar_atst = "n/a"
        else:
            scholar_atst = (
                f"{c.bundle_scholar_attested}/"
                f"{c.bundle_attestation_total} "
                f"({_format_pct(c.bundle_scholar_attested, c.bundle_attestation_total)})"
            )
        # IPA column: 'db_pct / bundle_pct' so the gap between the
        # two is visible at a glance. A high bundle / 0 db reading
        # flags pure cluster-mate inheritance (modern_english is the
        # canonical example post-wyrd-69s5).
        lex_ipa_pct = _format_pct(c.lexicon_etymons_with_ipa, c.total_etymons)
        if c.bundle_attestation_total == 0:
            bundle_ipa_pct = "n/a"
        else:
            bundle_ipa_pct = _format_pct(c.bundle_subjects_with_ipa, c.bundle_attestation_total)
        ipa_cell = f"{lex_ipa_pct} / {bundle_ipa_pct}"
        # Grandfather column: pure-grandfather families / cited families.
        # 'n/a' when this language has no citation-bearing families at
        # all (denominator zero).
        if c.rando_total_cited_families == 0:
            grandfather_cell = "n/a"
        else:
            grandfather_cell = (
                f"{c.rando_pure_grandfather_families}/{c.rando_total_cited_families} "
                f"({_format_pct(c.rando_pure_grandfather_families, c.rando_total_cited_families)})"
            )
        lines.append(
            f"| {c.language} | {c.total_etymons} | {c.total_lemmas} | {promo} | "
            f"{bundle_cell} | {scholar_atst} | {ipa_cell} | {grandfather_cell} | "
            f"{lex_hit} | {bundle_hit} | {inflect} | {variants} | {era} |"
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
        # B₂. Bundle attestation breakdown — wyrd-lc94. Surfaces the
        # actionable runtime-quality signal: of what's actually shipping,
        # what fraction is scholar-attested vs grandfathered. Distinct
        # from A (DB-level corpus depth) which counts mining residue
        # that never reaches the bundle.
        if c.bundle_attestation_total == 0:
            atst_line = "- **B₂. Bundle attestation:** n/a — no bundle subjects for this language."
        else:
            atst_line = (
                f"- **B₂. Bundle attestation:** "
                f"{c.bundle_scholar_attested} scholar-attested "
                f"({_format_pct(c.bundle_scholar_attested, c.bundle_attestation_total)}), "
                f"{c.bundle_empirical_only} wiktionary-empirical-only "
                f"({_format_pct(c.bundle_empirical_only, c.bundle_attestation_total)}), "
                f"{c.bundle_rando_only} rando-port-grandfather-only "
                f"({_format_pct(c.bundle_rando_only, c.bundle_attestation_total)}), "
                f"{c.bundle_uncited} uncited "
                f"({_format_pct(c.bundle_uncited, c.bundle_attestation_total)}); "
                f"of {c.bundle_attestation_total} subjects shipped."
            )
        lines.append(atst_line)
        # K. Rando-port grandfather audit (wyrd-vn09). Family-level
        # classification of which morpheme families ride solely on the
        # legacy seed. Distinct from B₂ above because that operated on
        # the bundle's per-word citation list (cluster-propagated). The
        # 'pure_grandfather' count is the curation candidate set —
        # families that would lose their bundle slot if include_rando
        # were turned off.
        if c.rando_total_cited_families == 0:
            grandfather_line = (
                "- **K. Rando-port grandfather audit:** n/a — no "
                "citation-bearing families rooted in this language."
            )
        else:
            grandfather_line = (
                f"- **K. Rando-port grandfather audit:** "
                f"{c.rando_pure_grandfather_families} pure-grandfather "
                f"({_format_pct(c.rando_pure_grandfather_families, c.rando_total_cited_families)}), "
                f"{c.rando_mixed_grandfather_families} mixed (rando-port + scholar/empirical), "
                f"of {c.rando_total_cited_families} cited families "
                f"rooted in this language."
            )
        lines.append(grandfather_line)
        # H. Pronunciation / IPA coverage (wyrd-dxu2). Two-axis line:
        # lexicon-side (true per-language IPA we have in the DB) and
        # bundle-side (what users see, which can include cluster-mate
        # inheritance). The gap between the two is the diagnostic
        # signal — a 0% / 20% reading means the bundle's IPA on this
        # language is entirely inherited from cluster mates.
        ipa_db_pct = _format_pct(c.lexicon_etymons_with_ipa, c.total_etymons)
        if c.bundle_attestation_total == 0:
            ipa_line = (
                f"- **H. Pronunciation coverage:** lexicon "
                f"{c.lexicon_etymons_with_ipa}/{c.total_etymons} ({ipa_db_pct}); "
                f"bundle n/a — no bundle subjects for this language."
            )
        else:
            ipa_bundle_pct = _format_pct(c.bundle_subjects_with_ipa, c.bundle_attestation_total)
            ipa_line = (
                f"- **H. Pronunciation coverage:** lexicon "
                f"{c.lexicon_etymons_with_ipa}/{c.total_etymons} ({ipa_db_pct}); "
                f"bundle {c.bundle_subjects_with_ipa}/{c.bundle_attestation_total} "
                f"({ipa_bundle_pct})."
            )
            # Diagnostic note when bundle reading is entirely inherited
            # from cluster mates (lexicon side is 0 but bundle side is
            # non-zero). Helps the reader distinguish 'we have IPA for
            # this language' from 'we're showing cluster mates' IPA'.
            if c.lexicon_etymons_with_ipa == 0 and c.bundle_subjects_with_ipa > 0:
                ipa_line += (
                    " ⚠ Bundle IPA is INHERITED from cluster mates "
                    "(no native IPA in the DB for this language)."
                )
        lines.append(ipa_line)
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
