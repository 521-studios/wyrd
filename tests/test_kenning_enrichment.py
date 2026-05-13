"""Tests for the L3 enrichment orchestrator — wyrd-ilam (Phase 2a).

Uses the public ``LexiconDB`` / ``init_schema`` flow rather than an
in-memory fixture: the orchestrator wraps real ``cluster_ocr_variants``
+ ``link_lemmas`` functions and needs their full schema. Tests target
the orchestrator's coordination behavior + the status-reporting layer,
not the individual passes' logic (those have their own coverage in
test_kenning_lexicon.py).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from wyrd.generators.kenning.enrichment import (
    LEMMA_METHOD_VERSION,
    OCR_METHOD_VERSION,
    enrichment_status,
    format_enrichment_run,
    format_enrichment_status,
    run_full_enrichment,
)
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema


def _build_db(tmp_path: Path) -> Path:
    """Real on-disk DB with the production schema. The orchestrator
    needs full schema + real LexiconDB methods, so an in-memory
    fixture with a slimmed schema won't work."""
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    return db_path


def _seed_for_enrichment(db_path: Path) -> dict[str, int]:
    """Insert a minimal OE scenario the orchestrator can act on:

    - ``cæt`` + ``caet``: real OCR-variant pair — both normalize to
      ``caet`` via the æ→ae mapping in normalize_ocr_form. cæt wins
      (lower id); caet becomes a tombstone.
    - ``cætan``: weak-oblique inflection. The link-lemmas suffix rule
      ``an`` → weak_oblique strips to ``cæt``, which matches the OCR-
      cluster canonical.

    Returns ``{name: etymon_id}`` for assertions."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES ('cæt', 'old-english')"
    )
    caet_winner_id = cur.lastrowid
    cur = conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES ('caet', 'old-english')"
    )
    caet_loser_id = cur.lastrowid
    cur = conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES ('cætan', 'old-english')"
    )
    caetan_id = cur.lastrowid
    conn.commit()
    conn.close()
    return {"winner": caet_winner_id, "loser": caet_loser_id, "inflected": caetan_id}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def test_orchestrator_runs_passes_in_canonical_order(tmp_path: Path):
    db_path = _build_db(tmp_path)
    _seed_for_enrichment(db_path)
    with LexiconDB(db_path) as db:
        result = run_full_enrichment(db, apply=False)
    # Order pins normalize-ocr first so lemma linkage targets canonicals
    # not tombstones.
    assert result["order"] == ["normalize-ocr", "link-lemmas"]
    assert result["applied"] is False
    assert result["ocr"]["method_version"] == OCR_METHOD_VERSION
    assert result["lemmas"]["method_version"] == LEMMA_METHOD_VERSION


def test_orchestrator_dry_run_does_not_write(tmp_path: Path):
    db_path = _build_db(tmp_path)
    ids = _seed_for_enrichment(db_path)
    with LexiconDB(db_path) as db:
        run_full_enrichment(db, apply=False)
    # No writes happened — merged_into_id + lemma_id still NULL.
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    for eid in ids.values():
        row = conn.execute(
            "SELECT merged_into_id, lemma_id FROM etymon WHERE id=?", (eid,)
        ).fetchone()
        assert row["merged_into_id"] is None
        assert row["lemma_id"] is None
    conn.close()


def test_orchestrator_apply_writes_merge_and_lemma(tmp_path: Path):
    """End-to-end: apply mode merges the OCR variant + links the
    inflected form to the canonical lemma."""
    db_path = _build_db(tmp_path)
    ids = _seed_for_enrichment(db_path)
    with LexiconDB(db_path) as db:
        result = run_full_enrichment(db, apply=True)
    assert result["applied"] is True
    # OCR-merge tombstoned one of the (cæt, caet) pair.
    assert result["ocr"]["etymons_merged"] >= 1
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    # Exactly one of cæt / caet should still be canonical
    # (merged_into_id is NULL); the other is a tombstone.
    winner_row = conn.execute(
        "SELECT merged_into_id FROM etymon WHERE id=?", (ids["winner"],)
    ).fetchone()
    loser_row = conn.execute(
        "SELECT merged_into_id FROM etymon WHERE id=?", (ids["loser"],)
    ).fetchone()
    tombstones = sum(1 for r in (winner_row, loser_row) if r["merged_into_id"] is not None)
    assert tombstones == 1
    conn.close()


def test_orchestrator_lemma_link_resolves_against_canonical_after_ocr_merge(tmp_path: Path):
    """Critical ordering check: after OCR-merge tombstones one of (cæt,
    caet), the lemma link from cætan must still find the surviving
    canonical (not the tombstone). The order normalize-ocr → link-lemmas
    is what makes this work."""
    db_path = _build_db(tmp_path)
    ids = _seed_for_enrichment(db_path)
    with LexiconDB(db_path) as db:
        run_full_enrichment(db, apply=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    inflected_row = conn.execute(
        "SELECT lemma_id, inflection, lemma_method FROM etymon WHERE id=?",
        (ids["inflected"],),
    ).fetchone()
    # cætan should link to the canonical lemma (cæt, the OCR-cluster
    # winner) — never to the tombstone caet. link_lemmas's lemmas_by_key
    # only includes rows with merged_into_id IS NULL, so the tombstone
    # is unreachable as a link target.
    assert inflected_row["lemma_id"] is not None
    target = conn.execute(
        "SELECT merged_into_id FROM etymon WHERE id=?", (inflected_row["lemma_id"],)
    ).fetchone()
    assert target["merged_into_id"] is None  # not a tombstone
    assert inflected_row["lemma_method"] == LEMMA_METHOD_VERSION
    conn.close()


# ---------------------------------------------------------------------------
# enrichment_status
# ---------------------------------------------------------------------------


def test_status_reports_zero_coverage_on_fresh_db(tmp_path: Path):
    db_path = _build_db(tmp_path)
    _seed_for_enrichment(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    status = enrichment_status(conn)
    conn.close()
    assert status["total_etymons"] == 3
    # Nothing enriched yet
    for column in ("lemma_id", "merged_into_id", "cognate_id", "stratum", "english_shaped"):
        assert status["columns"][column]["populated"] == 0


def test_status_restores_caller_row_factory(tmp_path: Path):
    """enrichment_status swaps row_factory to sqlite3.Row internally
    but must restore the original on return — the caller's connection
    state shouldn't leak. Verifies the try/finally guard."""
    db_path = _build_db(tmp_path)
    _seed_for_enrichment(db_path)
    conn = sqlite3.connect(db_path)
    # Default factory returns tuples — NOT sqlite3.Row.
    assert conn.row_factory is None
    enrichment_status(conn)
    # After call, factory is back to the original.
    assert conn.row_factory is None
    conn.close()


def test_status_works_without_row_factory_set(tmp_path: Path):
    """A caller with the default tuple factory should still get the
    structured dict back — the function doesn't depend on the caller's
    factory."""
    db_path = _build_db(tmp_path)
    _seed_for_enrichment(db_path)
    conn = sqlite3.connect(db_path)
    # No row_factory set — default tuples.
    status = enrichment_status(conn)
    conn.close()
    assert status["total_etymons"] == 3
    assert "lemma_id" in status["columns"]


def test_status_reports_coverage_after_enrichment(tmp_path: Path):
    db_path = _build_db(tmp_path)
    _seed_for_enrichment(db_path)
    with LexiconDB(db_path) as db:
        run_full_enrichment(db, apply=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    status = enrichment_status(conn)
    conn.close()
    # ≥1 merged (caet tombstoned) + ≥1 lemma-linked (cætan → cæt).
    assert status["columns"]["merged_into_id"]["populated"] >= 1
    assert status["columns"]["lemma_id"]["populated"] >= 1
    # Method version shows up in the lemma column distribution.
    lemma_methods = [m["method"] for m in status["columns"]["lemma_id"]["methods"]]
    assert LEMMA_METHOD_VERSION in lemma_methods


# ---------------------------------------------------------------------------
# format_enrichment_run / format_enrichment_status
# ---------------------------------------------------------------------------


def test_format_run_dry_run_uses_mergeable_verb():
    result = {
        "order": ["normalize-ocr", "link-lemmas"],
        "applied": False,
        "ocr": {
            "method_version": OCR_METHOD_VERSION,
            "groups": 2,
            "etymons_merged": 3,
            "sample_groups": [],
        },
        "lemmas": {
            "method_version": LEMMA_METHOD_VERSION,
            "candidates": 5,
            "sample": [],
        },
    }
    md = format_enrichment_run(result)
    assert "Order: normalize-ocr → link-lemmas" in md
    assert "Applied: False" in md
    assert "Etymons mergeable: 3" in md
    assert "Inflected etymons linkable: 5" in md
    assert OCR_METHOD_VERSION in md
    assert LEMMA_METHOD_VERSION in md


def test_format_run_apply_mode_uses_past_tense_verbs():
    result = {
        "order": ["normalize-ocr", "link-lemmas"],
        "applied": True,
        "ocr": {"method_version": OCR_METHOD_VERSION, "groups": 1, "etymons_merged": 1},
        "lemmas": {"method_version": LEMMA_METHOD_VERSION, "candidates": 2},
    }
    md = format_enrichment_run(result)
    assert "merged: 1" in md
    assert "linked: 2" in md


def test_format_status_shows_per_column_coverage():
    status = {
        "total_etymons": 100,
        "columns": {
            "lemma_id": {
                "populated": 25,
                "methods": [{"method": "link-lemmas-v1", "count": 25}],
            },
            "merged_into_id": {"populated": 3, "methods": []},
            "cognate_id": {"populated": 0, "methods": []},
            "stratum": {"populated": 50, "methods": []},
            "english_shaped": {"populated": 10, "methods": []},
        },
    }
    md = format_enrichment_status(status)
    assert "Total etymons: 100" in md
    assert "`lemma_id`" in md and "25 / 100" in md and "25.0%" in md
    assert "`merged_into_id`" in md and "3 / 100" in md
    assert "method=`link-lemmas-v1`: 25" in md
    assert "`cognate_id`" in md and "0 / 100" in md


def test_format_status_handles_empty_db():
    """Zero etymons in DB shouldn't divide-by-zero."""
    status = {
        "total_etymons": 0,
        "columns": {
            "lemma_id": {"populated": 0, "methods": []},
            "merged_into_id": {"populated": 0, "methods": []},
            "cognate_id": {"populated": 0, "methods": []},
            "stratum": {"populated": 0, "methods": []},
            "english_shaped": {"populated": 0, "methods": []},
        },
    }
    md = format_enrichment_status(status)
    assert "Total etymons: 0" in md
    assert "0.0%" in md
