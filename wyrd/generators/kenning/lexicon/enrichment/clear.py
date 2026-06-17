"""Selective revert of any enrichment stage (D22).

Each enrichment pass writes to a clearly-bounded column or table set:
``lemma_id`` / ``inflection`` / ``lemma_method`` for link_lemmas,
``merged_into_id`` for OCR clustering, ``etymon_text_match`` rows for
reverse + fuzzy search, etc. ``clear_enrichment`` walks one stage at
a time and resets just its outputs, so an operator iterating on a
heuristic can:

    clear_enrichment(db, stage="ocr", apply=True)
    cluster_ocr_variants(db, apply=True)   # new heuristic

without re-mining. Stage names + their reset surface are documented
inline at each branch.

The ``--stage=all-derived`` path is the fully-reset escape hatch:
clears every enrichment-derived column and table in one shot, so the
post-mining chain can re-run from a clean state.
"""

from __future__ import annotations

from dataclasses import dataclass

from wyrd.generators.kenning.lexicon.canonicalization_projection import clear_canonical_graph
from wyrd.generators.kenning.lexicon.db import LexiconDB


@dataclass(frozen=True)
class _ClearStage:
    """One single-query enrichment-clear stage: its name, the counts-dict
    key it populates, the COUNT(*) query for the dry-run preview, and the
    single UPDATE/DELETE that resets it."""

    name: str
    count_key: str
    count_sql: str
    apply_sql: str


# The single-query stages (one COUNT + one reset statement each). The
# multi-query ``attested-years`` stage (two row sources + a text-match
# guard) is handled explicitly in the helpers below.
_SIMPLE_CLEAR_STAGES: tuple[_ClearStage, ...] = (
    _ClearStage(
        "ocr",
        "ocr_merges_to_clear",
        "SELECT COUNT(*) FROM etymon WHERE merged_into_id IS NOT NULL",
        "UPDATE etymon SET merged_into_id = NULL",
    ),
    _ClearStage(
        "lemmas",
        "lemma_links_to_clear",
        "SELECT COUNT(*) FROM etymon WHERE lemma_id IS NOT NULL",
        "UPDATE etymon SET lemma_id = NULL, inflection = NULL, lemma_method = NULL",
    ),
    _ClearStage(
        "text-match",
        "text_match_rows_to_clear",
        "SELECT COUNT(*) FROM etymon_text_match",
        "DELETE FROM etymon_text_match",
    ),
    _ClearStage(
        "cognates",
        "cognate_assignments_to_clear",
        "SELECT COUNT(*) FROM etymon WHERE cognate_id IS NOT NULL",
        "UPDATE etymon SET cognate_id = NULL, cognate_method = NULL",
    ),
    _ClearStage(
        "attestations",
        "attestation_rows_to_clear",
        "SELECT COUNT(*) FROM toponym_attestation",
        "DELETE FROM toponym_attestation",
    ),
    _ClearStage(
        "period-forms",
        "period_form_rows_to_clear",
        "SELECT COUNT(*) FROM etymon_period_form",
        "DELETE FROM etymon_period_form",
    ),
)

# Every enrichment-derived stage (what ``all-derived`` expands to).
_ALL_DERIVED_STAGES = frozenset(
    {
        "ocr",
        "lemmas",
        "text-match",
        "cognates",
        "attested-years",
        "attestations",
        "period-forms",
        "canonical",
    }
)
_VALID_STAGES = _ALL_DERIVED_STAGES | {"all-derived"}


def _populate_clear_counts(db: LexiconDB, stages: set[str], counts: dict) -> None:
    """Fill the dry-run preview counts for the requested stages: a COUNT(*)
    per single-query stage, plus the two-row-source sum for
    ``attested-years`` (etymon_text_match + toponym_etymology)."""
    for spec in _SIMPLE_CLEAR_STAGES:
        if spec.name in stages:
            counts[spec.count_key] = db.conn.execute(spec.count_sql).fetchone()[0]
    if "attested-years" in stages:
        counts["attested_years_to_clear"] = (
            db.conn.execute(
                "SELECT COUNT(*) FROM etymon_text_match WHERE attested_year IS NOT NULL"
            ).fetchone()[0]
            + db.conn.execute(
                "SELECT COUNT(*) FROM toponym_etymology WHERE attested_year IS NOT NULL"
            ).fetchone()[0]
        )
    if "canonical" in stages:
        counts["canonical_nodes_to_clear"] = sum(
            db.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]  # noqa: S608 — fixed tuple
            for t in ("canonical_morpheme", "canonical_place", "canonical_sense")
        )


def _apply_stage_clears(db: LexiconDB, stages: set[str]) -> None:
    """Execute the reset statement for each requested stage, then commit.

    ``attested-years`` clears both row sources, but guards the
    etymon_text_match UPDATE: when ``text-match`` is also in ``stages``
    (notably under ``all-derived``) that table was already DELETEd, so only
    the toponym_etymology UPDATE need run."""
    for spec in _SIMPLE_CLEAR_STAGES:
        if spec.name in stages:
            db.conn.execute(spec.apply_sql)
    if "attested-years" in stages:
        if "text-match" not in stages:
            db.conn.execute("UPDATE etymon_text_match SET attested_year = NULL")
        db.conn.execute("UPDATE toponym_etymology SET attested_year = NULL")
    if "canonical" in stages:
        clear_canonical_graph(db)
    db.commit()


def clear_enrichment(db: LexiconDB, *, stage: str, apply: bool = False) -> dict:
    """Reset one or more enrichment stages so they can be re-run.

    Stages:
      ocr             - UPDATE etymon SET merged_into_id = NULL
                        (un-marks all OCR-cluster merges)
      lemmas          - UPDATE etymon SET lemma_id = NULL,
                                           inflection = NULL,
                                           lemma_method = NULL
                        (un-links every inflected variant from its lemma)
      text-match      - DELETE FROM etymon_text_match
                        (drops every reverse-search / fuzzy-search row)
      cognates        - UPDATE etymon SET cognate_id = NULL,
                                           cognate_method = NULL
                        (un-clusters every cognate-set assignment from
                        cluster-cognates; D27 / wyrd-81n)
      attested-years  - UPDATE etymon_text_match SET attested_year = NULL
                        AND UPDATE toponym_etymology SET
                                                       attested_year = NULL
                        (drops every lookup-attested-years assignment
                        on both row sources; D5-1 / wyrd-3ux + wyrd-bag)
      attestations    - DELETE FROM toponym_attestation
                        (drops every mine-attestations ingest row;
                        wyrd-skm Phase 3.0a)
      period-forms    - DELETE FROM etymon_period_form
                        (drops every project-period-forms output;
                        wyrd-unuo Phase 3.3)
      canonical       - NULL the canonical_*_id binding columns + DELETE
                        the canonical_morpheme/place/sense + canonical_label
                        tables (drops the project-canonical L2->L3
                        projection; wyrd-u6fn.3 / D50.6)
      all-derived     - all eight of the above

    Mining evidence (etymon, etymon_citation, etymon_gloss, etymon_tag,
    etymon_descent, toponym, toponym_etymology, toponym_etymology_element)
    is never touched — that's the load-bearing data and rebuilding it
    costs hours of LLM time. Per D21, only enrichment inferences are
    reversible; the extraction layer is sacred.

    Note: stage='ocr' is *partially* reversible. It un-marks
    merged_into_id but does NOT restore the lemma_id re-parenting that
    cluster_ocr_variants applies to inflected children at merge time.
    For a fully clean revert, use stage='all-derived' (which also
    clears lemma_id) and re-run link-lemmas.

    With apply=False the dry-run reports what would change. Pass
    apply=True to actually write.
    """
    if stage not in _VALID_STAGES:
        raise ValueError(f"unknown stage {stage!r}; must be one of {sorted(_VALID_STAGES)}")

    stages = set(_ALL_DERIVED_STAGES) if stage == "all-derived" else {stage}

    counts = {
        "stage": stage,
        "ocr_merges_to_clear": 0,
        "lemma_links_to_clear": 0,
        "text_match_rows_to_clear": 0,
        "cognate_assignments_to_clear": 0,
        "attested_years_to_clear": 0,
        "attestation_rows_to_clear": 0,
        "period_form_rows_to_clear": 0,
        "canonical_nodes_to_clear": 0,
    }
    _populate_clear_counts(db, stages, counts)

    if not apply:
        return counts

    _apply_stage_clears(db, stages)
    return counts
