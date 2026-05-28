"""Per-language audit + scorecard computation for the language-quality
dashboard (wyrd-wzwa Phase 1, parent epic wyrd-eni4).

Companion to ``models.py`` (dataclasses + module constants) and
``reporting.py`` (markdown + JSON output). This module owns the
expensive DB queries and bundle traversals that produce the
``LanguageScorecard`` rows + ``LanguageQualityReport`` orchestration.

Per-language metrics (each surfaced by a ``_corpus_depth`` /
``_inflection_coverage`` / ``_era_reflex_coverage`` / etc. helper):

- A. Corpus depth — total etymons / lemmas, promotion threshold (per
  language), promotion-eligible count at that threshold, avg witnesses,
  source count.
- B. Bundle representation — count of bundle words carrying this
  language as a sibling field.
- C. Semantic-tag coverage — lexicon-side (etymon_tag join) +
  bundle-side (modifier_tags x bundle sibling), both keyed off the
  top-N English reference tags.
- D. Inflection coverage (D8) — % of lemmas with at least one linked
  inflected child.
- E. Spelling-variant coverage (D18).
- F. Era-reflex coverage (D33) — Tier 1 + Tier 2 forward/back + Tier 3.
- G. Stratum coverage (D32).
- I. Citation depth — % of etymons with >=1 etymon_citation row.

See the full metric description in ``models.py``'s docstring and the
overarching wyrd-wzwa ticket.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from wyrd.generators.kenning.lexicon import RECOMMENDED_LANG_THRESHOLDS
from wyrd.generators.kenning.registers.phonology_rules import chain_for, rule_form

from .models import (
    _BUNDLE_LANG_KEY,
    _EMPIRICAL_CITATION_SOURCES,
    _GRANDFATHER_CITATION_SOURCES,
    _SHARED_BUNDLE_SIBLINGS,
    DEFAULT_LANGUAGES,
    ERA_CHAINS,
    FALLBACK_REFERENCE_TAGS,
    REFERENCE_TAG_COUNT,
    LanguageQualityReport,
    LanguageScorecard,
)

# --- reference profile ----------------------------------------------------


def load_reference_tags(
    proportions: Path | dict | None = None,
    *,
    top_n: int = REFERENCE_TAG_COUNT,
) -> list[str]:
    """Return the top-``top_n`` tags from English place-name proportions.

    Accepts either a ``Path`` (read JSON from disk) or a ``dict`` (use
    in-memory; the d90t cutover paths get proportions straight from the
    L4 runtime DB and avoid a JSON round-trip). Ranks ``tag_marginal``;
    falls back to ``FALLBACK_REFERENCE_TAGS`` if the source is missing,
    unreadable, or lacks ``tag_marginal`` (e.g. legacy proportions
    snapshot pre-wyrd-mj2)."""
    data: dict | None
    if proportions is None:
        return list(FALLBACK_REFERENCE_TAGS[:top_n])
    if isinstance(proportions, dict):
        data = proportions
    else:
        if not proportions.exists():
            return list(FALLBACK_REFERENCE_TAGS[:top_n])
        try:
            data = json.loads(proportions.read_text())
        except (OSError, json.JSONDecodeError):
            return list(FALLBACK_REFERENCE_TAGS[:top_n])
    marginal = (data or {}).get("tag_marginal") or {}
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


def populate_eligible_etymon_table(
    conn: sqlite3.Connection,
    *,
    restrict_to_generator_eligible: bool = True,
    source_filter: str | None = None,
    tag_filter: str | None = None,
) -> int:
    """Create (or recreate) a TEMP TABLE ``eligible_etymon`` that the
    per-language metric helpers JOIN against to filter their numerators
    and denominators (wyrd-bfv1).

    Generator-eligible etymon = any of:

    1. Has at least one row in ``etymon_citation`` (any source — rando-
       port grandfather seeds + scholar place-name attestations).
    2. Is a 1-hop descent-graph neighbor of a cited etymon (the
       ``back form`` or ``forward form`` of a cited word — one era
       step more eldritch or one era step more modern). Only the
       ``etymon_descent`` parent/child edges are traversed; cognate
       cluster mates (``etymon.cognate_id``) are NOT included since
       they're cross-language sibling forms, not era progressions.
    3. Tagged ``fantasy`` / ``monster`` / ``creature`` in
       ``etymon_tag`` — the explicit fiction-vocabulary exception.
    4. ``lemma_id`` rollup: both directions across the above seed —
       the lemma of any eligible inflection is itself eligible
       (upward), and every inflection of an eligible lemma rides
       along (downward).

    Pre-wyrd-bfv1 the dashboard divided every coverage metric by the
    raw etymon count, which was denominator-collapsed by the wyrd-dxu2
    Kaikki wiktextract ingest (1.4M modern-english entries, mostly
    common vocabulary the kenning generator never uses). Restricting
    to the eligible set yields the operator-actionable signal:
    'of the words the generator could actually use, how many have
    inflection / IPA / variants / etc.'.

    With ``restrict_to_generator_eligible=False`` the table is
    populated with every etymon ID — useful for diagnostic runs or
    tests that need to bypass the eligibility gate without modifying
    the helpers.

    With ``source_filter`` set (wyrd-5ecx) step (1) restricts to
    etymons cited by that specific source_id (e.g. ``'rando-port'``
    or ``'mawer_1920_northumberland_durham'``) and step (3)
    fantasy is skipped. Operator answers questions like 'how are
    the rando-port grandfather words doing on inflection?' or 'how
    is the Mawer-1920 attestation slice doing on IPA?'.

    With ``tag_filter`` set step (3) restricts to etymons with that
    tag (e.g. ``'fantasy'`` or ``'monster'``) and step (1) cited is
    skipped. Operator answers 'how is the fantasy slice doing on
    variant coverage?'.

    ``source_filter`` and ``tag_filter`` are mutually exclusive in
    the CLI surface — composing them muddies the per-row "what slice
    am I looking at?" semantic for an operator-facing dashboard.
    At the function level, passing both is supported as a union
    (cited-by-source ∪ tagged-with-tag, with descent + lemma rollup
    walking from the combined seed) — useful for testing flexibility
    and for any future internal caller that wants a composite slice.
    The CLI raises ``UsageError`` on the combination so the operator
    never has to think about union semantics.

    Returns the row count for caller logging.
    """
    conn.executescript("DROP TABLE IF EXISTS eligible_etymon;")
    if not restrict_to_generator_eligible:
        conn.executescript(
            """
            CREATE TEMP TABLE eligible_etymon (id INTEGER PRIMARY KEY);
            INSERT INTO eligible_etymon (id) SELECT id FROM etymon;
            """
        )
        return conn.execute("SELECT COUNT(*) FROM eligible_etymon").fetchone()[0]

    # Detect optional tables — minimal test fixtures may not include
    # etymon_tag / etymon_descent / etymon_citation. Skip those branches
    # gracefully when they're absent rather than crashing.
    existing = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name IN ('etymon_citation', 'etymon_descent', 'etymon_tag')"
        )
    }
    has_citation = "etymon_citation" in existing
    has_descent = "etymon_descent" in existing
    has_tag = "etymon_tag" in existing

    conn.executescript("CREATE TEMP TABLE eligible_etymon (id INTEGER PRIMARY KEY);")

    # Filter semantics:
    #   * Both unset: cited (any source) + fantasy (any of the
    #     fiction tag set) — default generator-eligible behavior.
    #   * Only source_filter: cited-by-this-source seed; fantasy
    #     step skipped (slice is the named source only).
    #   * Only tag_filter: tagged-with-this-tag seed; cited step
    #     skipped (slice is the named tag only).
    #   * Both set (CLI rejects this; internal callers can still
    #     pass both): union — cited-by-source PLUS tagged-with-tag.
    include_cited = has_citation and (tag_filter is None or source_filter is not None)
    include_fantasy = has_tag and (source_filter is None or tag_filter is not None)

    if include_cited:
        # (1) cited etymons. When source_filter is set, restrict to
        # that single source (e.g. 'rando-port'); otherwise any source.
        if source_filter is not None:
            conn.execute(
                "INSERT OR IGNORE INTO eligible_etymon (id) "
                "SELECT DISTINCT etymon_id FROM etymon_citation "
                "WHERE source_id = ? AND etymon_id IS NOT NULL",
                (source_filter,),
            )
        else:
            conn.execute(
                "INSERT OR IGNORE INTO eligible_etymon (id) "
                "SELECT DISTINCT etymon_id FROM etymon_citation "
                "WHERE etymon_id IS NOT NULL"
            )
    if include_fantasy:
        # Insert the tag seed BEFORE descent expansion so descent and
        # lemma rollup walk from BOTH the cited and fantasy seeds.
        # When tag_filter is set, restrict to that single tag (e.g.
        # 'fantasy' alone, excluding 'monster' / 'creature');
        # otherwise include the full fiction-tag set.
        if tag_filter is not None:
            conn.execute(
                "INSERT OR IGNORE INTO eligible_etymon (id) "
                "SELECT etymon_id FROM etymon_tag WHERE tag = ?",
                (tag_filter,),
            )
        else:
            conn.execute(
                "INSERT OR IGNORE INTO eligible_etymon (id) "
                "SELECT etymon_id FROM etymon_tag "
                "WHERE tag IN ('fantasy', 'monster', 'creature')"
            )
    if has_descent and (include_cited or include_fantasy):
        # 1-hop descent parents + children of the current seed (cited
        # + fantasy, whichever are active). Walks from EVERY seed
        # type uniformly — so e.g. fantasy-tagged 'Orc' picks up its
        # OE ancestor 'orcneas' via descent.
        conn.execute(
            """
            INSERT OR IGNORE INTO eligible_etymon (id)
            SELECT DISTINCT id FROM (
                SELECT ed.parent_id AS id FROM etymon_descent ed
                 WHERE ed.child_id IN (SELECT id FROM eligible_etymon)
                   AND ed.parent_id IS NOT NULL
                UNION
                SELECT ed.child_id AS id FROM etymon_descent ed
                 WHERE ed.parent_id IN (SELECT id FROM eligible_etymon)
                   AND ed.child_id IS NOT NULL
            )
            """
        )
    # (4) Lemma rollup: BOTH directions.
    #   (4a) Upward — promote the lemma of any eligible inflected
    #        form. Citations on the inflected form (e.g. ``cotum``)
    #        should make their base lemma (``cot``) eligible too;
    #        otherwise lemma-based metrics like D undercount
    #        because the base form isn't in the eligible set.
    #   (4b) Downward — inflected children of an eligible lemma
    #        ride along (the within-language inflection
    #        inheritance the inflection-coverage metric counts).
    # Order matters: 4a populates more lemmas first so 4b picks
    # up their inflected children in a single pass.
    conn.execute(
        """
        INSERT OR IGNORE INTO eligible_etymon (id)
        SELECT e.lemma_id FROM etymon e
         WHERE e.id IN (SELECT id FROM eligible_etymon)
           AND e.lemma_id IS NOT NULL
        """
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO eligible_etymon (id)
        SELECT e.id FROM etymon e
         WHERE e.lemma_id IN (SELECT id FROM eligible_etymon)
        """
    )
    return conn.execute("SELECT COUNT(*) FROM eligible_etymon").fetchone()[0]


def _ensure_eligible_etymon_table(conn: sqlite3.Connection) -> None:
    """Best-effort eligibility-table guard. If the caller hasn't
    populated ``eligible_etymon`` already, populate it now with the
    default generator-eligible set. Idempotent — safe to call from
    every helper, but compute_scorecard fronts it for efficiency."""
    try:
        conn.execute("SELECT 1 FROM eligible_etymon LIMIT 1").fetchone()
    except sqlite3.OperationalError:
        populate_eligible_etymon_table(conn)


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
    _ensure_eligible_etymon_table(conn)
    # wyrd-bfv1: INNER JOIN against eligible_etymon (much smaller than
    # the etymon table for languages with bulk wiktextract ingest) lets
    # SQLite drive from the small side. Replaces an EXISTS subquery
    # that was scanning the etymon table for each row.
    tagged_total = conn.execute(
        """
        SELECT COUNT(DISTINCT et.etymon_id)
          FROM etymon_tag et
          JOIN etymon e ON e.id = et.etymon_id
          JOIN eligible_etymon ee ON ee.id = e.id
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
              JOIN eligible_etymon ee ON ee.id = e.id
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
    """Metric A. Raw + generator-eligible corpus counts.

    Returns both:
    * ``total_etymons`` / ``total_lemmas`` — raw counts off the etymon
      table (DB-depth signal, dominated by the wyrd-dxu2 Kaikki
      wiktextract ingest on modern-english).
    * ``eligible_etymons`` / ``eligible_lemmas`` — restricted to the
      generator-eligible set (wyrd-bfv1). The eligible figures drive
      every coverage percentage so the operator sees 'of the words
      the generator could use, what % have IPA / inflection / etc.'.
    """
    _ensure_eligible_etymon_table(conn)
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
    eligible_etymons = conn.execute(
        """
        SELECT COUNT(*) FROM etymon e
          JOIN eligible_etymon ee ON ee.id = e.id
         WHERE e.language = ?
           AND e.merged_into_id IS NULL
        """,
        (language,),
    ).fetchone()[0]
    eligible_lemmas = conn.execute(
        """
        SELECT COUNT(*) FROM etymon e
          JOIN eligible_etymon ee ON ee.id = e.id
         WHERE e.language = ?
           AND e.merged_into_id IS NULL
           AND e.lemma_id IS NULL
        """,
        (language,),
    ).fetchone()[0]
    # wyrd-6n2x: filter consensus rows to the generator-eligible
    # lemma pool. The unfiltered query counts every row in the
    # ``etymon_consensus`` view — which for modern-english is the
    # full 1.26M-row wiktextract corpus — and drives ``avg_witnesses``
    # to ~0.0 (e.g. modern-english: 409 total witnesses / 1.26M rows
    # → 0.0003 → round to 0.0). The surrounding ``eligible_etymons``
    # and ``eligible_lemmas`` counts already use the eligibility scope;
    # the corpus-depth metric is incoherent if avg_witnesses and
    # promotion_eligible use a different denominator. ``lemma_id``
    # is the etymon ID for the lemma row (etymon_consensus is keyed
    # per-lemma); joining ``eligible_etymon`` on that ID restricts
    # the rollup to lemmas the generator can actually use.
    consensus_rows = conn.execute(
        """
        SELECT ec.witnesses
          FROM etymon_consensus ec
          JOIN eligible_etymon ee ON ee.id = ec.lemma_id
         WHERE ec.language = ?
        """,
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
        "eligible_etymons": eligible_etymons,
        "eligible_lemmas": eligible_lemmas,
        "promotion_threshold": threshold,
        "promotion_eligible": promotion_eligible,
        "avg_witnesses": round(avg_witnesses, 3),
        "source_count": source_count,
    }


def _inflection_coverage(conn: sqlite3.Connection, language: str) -> dict[str, Any]:
    """Metric D. Lemmas with linked inflected children + distinct labels.

    Both the numerator (lemmas with children) and denominator (eligible
    lemmas, computed in ``_corpus_depth``) restrict to the generator-
    eligible set per wyrd-bfv1.
    """
    _ensure_eligible_etymon_table(conn)
    # Lemmas (etymon.lemma_id IS NULL & inflection IS NULL) where another
    # etymon links lemma_id back to this one.
    lemmas_with_children = conn.execute(
        """
        SELECT COUNT(*) FROM etymon parent
          JOIN eligible_etymon ee ON ee.id = parent.id
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
        SELECT COUNT(DISTINCT e.inflection)
          FROM etymon e
          JOIN eligible_etymon ee ON ee.id = e.id
         WHERE e.language = ?
           AND e.merged_into_id IS NULL
           AND e.inflection IS NOT NULL
        """,
        (language,),
    ).fetchone()[0]
    return {
        "lemmas_with_inflections": lemmas_with_children,
        "distinct_inflection_labels": distinct_labels,
    }


def _variant_coverage(conn: sqlite3.Connection, language: str) -> int:
    """Metric E. Etymons with at least one matched_form != canonical_form.
    Restricted to the generator-eligible set per wyrd-bfv1."""
    _ensure_eligible_etymon_table(conn)
    return conn.execute(
        """
        SELECT COUNT(DISTINCT etm.etymon_id)
          FROM etymon_text_match etm
          JOIN etymon e ON e.id = etm.etymon_id
          JOIN eligible_etymon ee ON ee.id = e.id
         WHERE e.language = ?
           AND e.merged_into_id IS NULL
           AND etm.matched_form != e.canonical_form
        """,
        (language,),
    ).fetchone()[0]


def _era_reflex_coverage(conn: sqlite3.Connection, language: str) -> dict[str, Any]:
    """Metric F. Tier 1 / 2-fwd / 2-back / 3 counts per the chain.
    All counts restricted to the generator-eligible set per wyrd-bfv1."""
    _ensure_eligible_etymon_table(conn)
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
          JOIN eligible_etymon ee ON ee.id = e.id
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
              JOIN eligible_etymon ee ON ee.id = parent.id
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
              JOIN eligible_etymon ee ON ee.id = child.id
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
          JOIN eligible_etymon ee ON ee.id = e.id
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
          JOIN eligible_etymon ee ON ee.id = e.id
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
    """Metric G. Etymons with non-NULL stratum (D32 classifier output).
    Restricted to the generator-eligible set per wyrd-bfv1."""
    _ensure_eligible_etymon_table(conn)
    return conn.execute(
        """
        SELECT COUNT(*) FROM etymon e
          JOIN eligible_etymon ee ON ee.id = e.id
         WHERE e.language = ?
           AND e.merged_into_id IS NULL
           AND e.stratum IS NOT NULL
        """,
        (language,),
    ).fetchone()[0]


def _citation_depth(conn: sqlite3.Connection, language: str) -> dict[str, Any]:
    """Metric I. Etymons with ≥1 citation + average citations per etymon.
    Restricted to the generator-eligible set per wyrd-bfv1 — note that
    by construction every cited etymon IS eligible (cited ⊆ eligible),
    so the restriction primarily affects the denominator at the
    scorecard level rather than the numerator here."""
    _ensure_eligible_etymon_table(conn)
    cited_etymons = conn.execute(
        """
        SELECT COUNT(DISTINCT c.etymon_id)
          FROM etymon_citation c
          JOIN etymon e ON e.id = c.etymon_id
          JOIN eligible_etymon ee ON ee.id = e.id
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
          JOIN eligible_etymon ee ON ee.id = e.id
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
    """Metric H (lexicon side). Count of eligible etymons in this language
    with non-null ``pronunciation_ipa``. Excludes merged-away rows
    (consensus view convention) AND non-eligible rows (per wyrd-bfv1).
    The DB column is populated by ``wiktextract_ingester._process_entry``
    and (post-wyrd-69s5) ``wiktextract_corpus_miner._write_one``;
    absent on languages whose wiktextract slice doesn't carry
    ``sounds`` arrays."""
    _ensure_eligible_etymon_table(conn)
    return conn.execute(
        """
        SELECT COUNT(*) FROM etymon e
          JOIN eligible_etymon ee ON ee.id = e.id
         WHERE e.language = ?
           AND e.merged_into_id IS NULL
           AND e.pronunciation_ipa IS NOT NULL
           AND e.pronunciation_ipa != ''
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


def _tier4_phonology_coverage(
    conn: sqlite3.Connection,
    language: str,
    *,
    progress_callback: Any = None,
) -> int:
    """Metric F Tier-4 (wyrd-pmne). Count of etymons in ``language``
    whose ``canonical_form`` produces a transformed reflex for at
    least one other era in the same phonology chain when routed
    through ``phonology_rules.rule_form`` (wyrd-98cs Tier-4 fallback).

    Returns ``-1`` for languages without a registered phonology chain
    (the LanguageScorecard renders this as 'n/a' to distinguish 'no
    rules registered' from '0 etymons fire'). Returns ``0`` for
    chain-eligible languages with no etymons or no etymon whose form
    fires a rule.

    Full-population walk: the rule library doesn't persist outputs to
    DB tables, so the count can only be computed by applying the rules
    over every etymon. Early-breaks on the first target language that
    produces a non-None result, since 'this etymon fires Tier-4 for at
    least one era' is the metric semantic. Optional
    ``progress_callback(done, total)`` lets the CLI surface progress
    on the long modern-english walk (~3 minutes for 1.4M etymons).
    """
    _ensure_eligible_etymon_table(conn)
    chain_info = chain_for(language)
    if chain_info is None:
        return -1
    resolved, _family, chain = chain_info
    # Order targets by chain-distance from self ascending so the inner
    # loop's early-break tries the cheapest walk first. For ModE the
    # walk to EModE is one cell, to ME two cells, to OE three cells —
    # at ~99% fire rate the closest target almost always wins, so the
    # average per-etymon cost drops materially vs. declaration order.
    i_self = chain.index(resolved)
    other_stops = sorted(
        (stop for stop in chain if stop != resolved),
        key=lambda stop: abs(chain.index(stop) - i_self),
    )
    if not other_stops:
        return -1

    # wyrd-bfv1: walk only the eligible-etymon slice — the 1.4M
    # ModE raw etymon table is dominated by general-vocabulary Kaikki
    # wiktextract ingest the generator doesn't use. Restricting to
    # eligible drops modern-english from 1.4M to ~22K (~60x speedup,
    # ~3 min → ~3 s on the modern-english walk).
    total_in_lang = conn.execute(
        """
        SELECT COUNT(*) FROM etymon e
          JOIN eligible_etymon ee ON ee.id = e.id
         WHERE e.language=? AND e.merged_into_id IS NULL
        """,
        (language,),
    ).fetchone()[0]
    if total_in_lang == 0:
        return 0

    # Throttle to ~20 callback fires per language; clamp floor at 1 so
    # tiny populations (old-welsh 80 eligible) still see a single tick
    # rather than only the final emission. The CLI callback in turn
    # throttles to ~10 stderr lines per language for human-paced output.
    progress_step = max(1, total_in_lang // 20)
    fires = 0
    seen = 0
    for (canonical_form,) in conn.execute(
        """
        SELECT e.canonical_form FROM etymon e
          JOIN eligible_etymon ee ON ee.id = e.id
         WHERE e.language=? AND e.merged_into_id IS NULL
        """,
        (language,),
    ):
        seen += 1
        if canonical_form:
            for target in other_stops:
                # Pass the pre-resolved language name to skip the alias
                # lookup that rule_form would otherwise repeat per call.
                if rule_form(canonical_form, resolved, target) is not None:
                    fires += 1
                    break
        if progress_callback is not None and seen % progress_step == 0:
            progress_callback(seen, total_in_lang)
    if progress_callback is not None and seen % progress_step != 0:
        progress_callback(seen, total_in_lang)
    return fires


def _bundle_era_reflex_coverage(
    bundle: list[dict[str, Any]] | dict[str, Any],
    language: str,
    sibling: str | None,
) -> int:
    """Metric F₂ (bundle side, wyrd-h5it). Count of bundle subjects whose
    ``sibling`` is populated AND at least one word carries a non-empty
    ``era_reflexes[language]`` entry. Includes wyrd-98cs Tier-4
    phonology-rule:v1 reflexes that aren't in DB tables, so this is the
    deploy-ready 'can the SPA render an era stop in this language'
    signal rather than the DB-persistence reading the lex-side metric
    produces.

    A subject counts when it has the language's sibling populated AND
    any of its words has a non-empty list under ``era_reflexes[language]``.
    The sibling gate matches H's denominator shape: 'of the N bundle
    subjects in this language's cluster, how many can time-travel to
    this language as an era stop'. Subjects without the sibling are
    excluded from the denominator (see ``bundle_attestation_total``).
    """
    if sibling is None:
        return 0
    n = 0
    for subj in _bundle_subjects(bundle):
        has_sibling = False
        has_target_reflex = False
        for word in subj.get("words") or []:
            if word.get(sibling):
                has_sibling = True
            reflexes = word.get("era_reflexes") or {}
            if reflexes.get(language):
                has_target_reflex = True
            if has_sibling and has_target_reflex:
                break
        if has_sibling and has_target_reflex:
            n += 1
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
    compute_tier4: bool = True,
    tier4_progress_callback: Any = None,
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
    bundle_reflex = _bundle_era_reflex_coverage(bundle, language, sibling)
    if compute_tier4:
        tier4_count = _tier4_phonology_coverage(
            conn, language, progress_callback=tier4_progress_callback
        )
    else:
        tier4_count = -1
    rando_bucket = (rando_port_audit or {}).get(
        language, {"pure_grandfather": 0, "mixed_grandfather": 0, "total_families": 0}
    )

    total_lemmas = depth["total_lemmas"]
    total_etymons = depth["total_etymons"]
    eligible_etymons = depth["eligible_etymons"]
    eligible_lemmas = depth["eligible_lemmas"]
    return LanguageScorecard(
        language=language,
        total_etymons=total_etymons,
        total_lemmas=total_lemmas,
        eligible_etymons=eligible_etymons,
        eligible_lemmas=eligible_lemmas,
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
        # wyrd-bfv1: every lex-side rate uses the generator-eligible
        # denominator instead of the raw count so the dashboard's
        # coverage numbers reflect 'how useful is the useful slice'
        # rather than being denominator-collapsed by Kaikki ingest.
        inflection_density=(
            round(inflect["lemmas_with_inflections"] / eligible_lemmas, 4)
            if eligible_lemmas
            else 0.0
        ),
        distinct_inflection_labels=inflect["distinct_inflection_labels"],
        etymons_with_variants=variants,
        variant_density=(round(variants / eligible_etymons, 4) if eligible_etymons else 0.0),
        chain_older=era["chain_older"],
        chain_younger=era["chain_younger"],
        chain_known=era["chain_known"],
        tier1_cognate_count=era["tier1_cognate_count"],
        tier2_forward_count=era["tier2_forward_count"],
        tier2_back_count=era["tier2_back_count"],
        tier3_period_form_count=era["tier3_period_form_count"],
        tier4_phonology_count=tier4_count,
        any_reflex_count=era["any_reflex_count"],
        any_reflex_rate=(
            round(era["any_reflex_count"] / eligible_etymons, 4) if eligible_etymons else 0.0
        ),
        bundle_subjects_with_reflex=bundle_reflex,
        bundle_reflex_rate=(
            round(bundle_reflex / attestation["total"], 4) if attestation["total"] else 0.0
        ),
        stratum_classified=stratum,
        stratum_density=(round(stratum / eligible_etymons, 4) if eligible_etymons else 0.0),
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
        lexicon_ipa_density=(round(lex_ipa / eligible_etymons, 4) if eligible_etymons else 0.0),
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
    compute_tier4: bool = True,
    tier4_progress_callback: Any = None,
    source_filter: str | None = None,
    tag_filter: str | None = None,
) -> LanguageQualityReport:
    """Compute the full multi-language report.

    Languages with zero etymons are dropped when ``drop_empty=True``
    (Phase-1 default — every-zero scorecards are noise on the human
    output and don't help mining decisions). Set
    ``drop_empty=False`` for trend tracking where the absence is
    informative.

    ``compute_tier4=False`` skips the wyrd-pmne Tier-4 phonology walk
    (which adds ~3 minutes for the modern-english 1.4M etymon
    population). Tier-4 counts render as 'n/a' in the dashboard when
    skipped — same visual rendering as 'no phonology rules registered'.
    ``tier4_progress_callback(language, done, total)`` lets the CLI
    surface a CLAUDE.md-shape progress line on the long English walk.

    ``source_filter`` / ``tag_filter`` (wyrd-5ecx) restrict the
    eligible-set seed to a specific source_id (e.g. ``'rando-port'``)
    or tag (e.g. ``'fantasy'``) — operator-facing 'slice views' of
    the dashboard. See ``populate_eligible_etymon_table`` for the
    exact seed semantics.
    """
    languages_to_check = list(languages or DEFAULT_LANGUAGES)
    if reference_tags is None:
        reference_tags = list(FALLBACK_REFERENCE_TAGS[:REFERENCE_TAG_COUNT])
    # wyrd-5ecx: populate the eligible-etymon temp table ONCE up
    # front with the active filter (if any). Per-language helpers
    # then JOIN against it. Without this the filter wouldn't be
    # applied until the first auto-populate inside _ensure_*,
    # which would use default (unfiltered) semantics.
    populate_eligible_etymon_table(
        conn,
        source_filter=source_filter,
        tag_filter=tag_filter,
    )
    # wyrd-vn09: precompute the rando-port grandfather audit ONCE
    # (rollup walks every etymon row, so per-language recomputation
    # would be wasteful). Each scorecard reads its bucket from the
    # precomputed map.
    rando_port_audit = compute_rando_port_grandfather_audit(conn)
    cards: list[LanguageScorecard] = []
    for lang in languages_to_check:
        per_lang_callback = None
        if compute_tier4 and tier4_progress_callback is not None:

            def per_lang_callback(done: int, total: int, lang: str = lang) -> None:
                tier4_progress_callback(lang, done, total)

        card = compute_scorecard(
            conn,
            lang,
            bundle,
            reference_tags,
            default_threshold=default_threshold,
            lang_thresholds=lang_thresholds,
            rando_port_audit=rando_port_audit,
            compute_tier4=compute_tier4,
            tier4_progress_callback=per_lang_callback,
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
        source_filter=source_filter,
        tag_filter=tag_filter,
    )
