"""Selective revert of any enrichment stage (D22).

Each enrichment pass writes to a clearly-bounded column or table set:
``lemma_id`` / ``inflection`` / ``lemma_method`` for link_lemmas,
``merged_into_id`` for OCR clustering, ``etymon_text_match`` rows for
reverse + fuzzy search, etc. ``clear_enrichment`` walks one stage at
a time and resets just its outputs, so an operator iterating on a
heuristic can:

    clear_enrichment(stage="ocr", apply=True)
    cluster_ocr_variants(db, apply=True)   # new heuristic

without re-mining. Stage names + their reset surface are documented
inline at each branch.

The ``--stage=all-derived`` path is the fully-reset escape hatch:
clears every enrichment-derived column and table in one shot, so the
post-mining chain can re-run from a clean state.
"""

from __future__ import annotations

from wyrd.generators.kenning.lexicon.db import LexiconDB


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
      all-derived     - all seven of the above

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
    valid = {
        "ocr",
        "lemmas",
        "text-match",
        "cognates",
        "attested-years",
        "attestations",
        "period-forms",
        "all-derived",
    }
    if stage not in valid:
        raise ValueError(f"unknown stage {stage!r}; must be one of {sorted(valid)}")

    stages = (
        {
            "ocr",
            "lemmas",
            "text-match",
            "cognates",
            "attested-years",
            "attestations",
            "period-forms",
        }
        if stage == "all-derived"
        else {stage}
    )

    counts = {
        "stage": stage,
        "ocr_merges_to_clear": 0,
        "lemma_links_to_clear": 0,
        "text_match_rows_to_clear": 0,
        "cognate_assignments_to_clear": 0,
        "attested_years_to_clear": 0,
        "attestation_rows_to_clear": 0,
        "period_form_rows_to_clear": 0,
    }
    if "ocr" in stages:
        counts["ocr_merges_to_clear"] = db.conn.execute(
            "SELECT COUNT(*) FROM etymon WHERE merged_into_id IS NOT NULL"
        ).fetchone()[0]
    if "lemmas" in stages:
        counts["lemma_links_to_clear"] = db.conn.execute(
            "SELECT COUNT(*) FROM etymon WHERE lemma_id IS NOT NULL"
        ).fetchone()[0]
    if "text-match" in stages:
        counts["text_match_rows_to_clear"] = db.conn.execute(
            "SELECT COUNT(*) FROM etymon_text_match"
        ).fetchone()[0]
    if "cognates" in stages:
        counts["cognate_assignments_to_clear"] = db.conn.execute(
            "SELECT COUNT(*) FROM etymon WHERE cognate_id IS NOT NULL"
        ).fetchone()[0]
    if "attested-years" in stages:
        counts["attested_years_to_clear"] = (
            db.conn.execute(
                "SELECT COUNT(*) FROM etymon_text_match WHERE attested_year IS NOT NULL"
            ).fetchone()[0]
            + db.conn.execute(
                "SELECT COUNT(*) FROM toponym_etymology WHERE attested_year IS NOT NULL"
            ).fetchone()[0]
        )
    if "attestations" in stages:
        counts["attestation_rows_to_clear"] = db.conn.execute(
            "SELECT COUNT(*) FROM toponym_attestation"
        ).fetchone()[0]
    if "period-forms" in stages:
        counts["period_form_rows_to_clear"] = db.conn.execute(
            "SELECT COUNT(*) FROM etymon_period_form"
        ).fetchone()[0]

    if not apply:
        return counts

    if "ocr" in stages:
        db.conn.execute("UPDATE etymon SET merged_into_id = NULL")
    if "lemmas" in stages:
        db.conn.execute("UPDATE etymon SET lemma_id = NULL, inflection = NULL, lemma_method = NULL")
    if "text-match" in stages:
        db.conn.execute("DELETE FROM etymon_text_match")
    if "cognates" in stages:
        db.conn.execute("UPDATE etymon SET cognate_id = NULL, cognate_method = NULL")
    if "attested-years" in stages:
        # 'text-match' DELETEs etymon_text_match if it's also in stages
        # (notably under 'all-derived'), so guard that UPDATE — the
        # toponym_etymology UPDATE always runs since text-match doesn't
        # touch that table.
        if "text-match" not in stages:
            db.conn.execute("UPDATE etymon_text_match SET attested_year = NULL")
        db.conn.execute("UPDATE toponym_etymology SET attested_year = NULL")
    if "attestations" in stages:
        db.conn.execute("DELETE FROM toponym_attestation")
    if "period-forms" in stages:
        db.conn.execute("DELETE FROM etymon_period_form")
    db.commit()
    return counts


# --- D27 / wyrd-81n: cognate clustering ------------------------------------


_COGNATE_BRIDGING_EDGES = ("inheritance", "borrowing")
_CLUSTER_COGNATES_METHOD = "cluster-cognates-v1"
