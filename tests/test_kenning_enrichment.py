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
        result = run_full_enrichment(db, apply=False, skip_l3_derivations=True)
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
        run_full_enrichment(db, apply=False, skip_l3_derivations=True)
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
        result = run_full_enrichment(db, apply=True, skip_l3_derivations=True)
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
        run_full_enrichment(db, apply=True, skip_l3_derivations=True)
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
        run_full_enrichment(db, apply=True, skip_l3_derivations=True)
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


def test_format_run_renders_by_stratum_breakdown():
    """The stratum section renders the per-language by_stratum distribution
    when present, and omits the breakdown line when absent."""
    result = {
        "order": ["normalize-ocr", "link-lemmas", "classify-stratum"],
        "applied": True,
        "ocr": {"method_version": OCR_METHOD_VERSION, "groups": 0, "etymons_merged": 0},
        "lemmas": {"method_version": LEMMA_METHOD_VERSION, "candidates": 0},
        "stratum": {
            "applied": 1,
            "languages": {
                "welsh": {
                    "proposed": 7,
                    "written": 5,
                    "skipped": 2,
                    "by_stratum": {"native-welsh": 3, "latin-loan": 2},
                },
                "french": {
                    "proposed": 0,
                    "written": 0,
                    "skipped": 0,
                },
            },
        },
    }
    md = format_enrichment_run(result)
    # Per-language line still rendered.
    assert "welsh: proposed=7" in md and "written=5" in md and "skipped=2" in md
    # New: by_stratum breakdown is sorted alphabetically, deterministic output.
    assert "by stratum: latin-loan=2, native-welsh=3" in md
    # When by_stratum is absent (french line above), no breakdown line follows.
    welsh_idx = md.index("welsh:")
    french_idx = md.index("french:")
    welsh_section = md[welsh_idx:french_idx]
    assert "by stratum:" in welsh_section
    french_section = md[french_idx:]
    assert "by stratum:" not in french_section


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


# ---------------------------------------------------------------------
# format_enrichment_run: per-section coverage (wyrd-s9z3)
# ---------------------------------------------------------------------
#
# The existing tests above cover ocr+lemmas + stratum's by_stratum
# section. Each optional section (decompose, cognates, english_shaped,
# period_forms) had zero direct render coverage; the absent-section
# branch was also untested. Without these pins, a future field addition
# to the result dict could break the renderer for one chain (CLI vs
# bundled enrichment) without the test suite noticing.


def _base_result(**overrides):
    """Minimal valid result with ocr + lemmas — required by the
    renderer's unconditional prefix."""
    base = {
        "order": ["normalize-ocr", "link-lemmas"],
        "applied": True,
        "ocr": {"method_version": OCR_METHOD_VERSION, "groups": 0, "etymons_merged": 0},
        "lemmas": {"method_version": LEMMA_METHOD_VERSION, "candidates": 0},
    }
    base.update(overrides)
    return base


def test_format_run_renders_decompose_section_when_present():
    """The decompose section prints toponyms_scanned + decompositions
    + a single-line rule-firings breakdown matching the dict keys
    decompose_all returns."""
    result = _base_result(
        decompose={
            "applied": 1,
            "toponyms_scanned": 12,
            "decompositions": 30,
            "scholar": 4,
            "scholar-disagreement": 1,
            "unique-zero-unaccounted": 5,
            "tiebreaker": 2,
            "no-canonical": 0,
        }
    )
    md = format_enrichment_run(result)
    assert "### Decomposition" in md
    assert "Toponyms scanned: 12" in md
    assert "Decompositions written: 30" in md
    assert "scholar=4" in md
    assert "scholar-disagreement=1" in md
    assert "unique-zero=5" in md
    assert "tiebreaker=2" in md
    assert "no-canonical=0" in md


def test_format_run_omits_decompose_section_when_absent():
    """When the result dict has no 'decompose' key, the section is
    absent from the rendered markdown. Pins the optional-section
    branch."""
    md = format_enrichment_run(_base_result())
    assert "### Decomposition" not in md


def test_format_run_renders_cognates_section_when_present():
    result = _base_result(cognates={"roots": 17, "candidates": 42})
    md = format_enrichment_run(result)
    assert "### Cognate clustering" in md
    assert "Roots walked: 17" in md
    assert "Cognate_id assignments: 42" in md


def test_format_run_omits_cognates_section_when_absent():
    md = format_enrichment_run(_base_result())
    assert "### Cognate clustering" not in md


def test_format_run_renders_english_shaped_section_when_present():
    """The english_shaped section prints candidates / written /
    skipped — the three counters derive_english_shaped_all returns."""
    result = _base_result(english_shaped={"candidates": 50, "written": 47, "skipped_no_input": 3})
    md = format_enrichment_run(result)
    assert "### English-shaped derivation" in md
    assert "Candidates: 50" in md
    assert "Written: 47" in md
    assert "Skipped (no input): 3" in md


def test_format_run_omits_english_shaped_section_when_absent():
    md = format_enrichment_run(_base_result())
    assert "### English-shaped derivation" not in md


def test_format_run_renders_period_forms_section_when_present():
    result = _base_result(
        period_forms={
            "rows_scanned": 100,
            "rows_projected": 80,
            "candidates": 200,
            "rows_written": 75,
        }
    )
    md = format_enrichment_run(result)
    assert "### Period-form projection" in md
    assert "Rows scanned: 100" in md
    assert "Projected: 80" in md
    assert "Candidates: 200" in md
    assert "Rows written: 75" in md


def test_format_run_omits_period_forms_section_when_absent():
    md = format_enrichment_run(_base_result())
    assert "### Period-form projection" not in md


def test_format_run_renders_all_optional_sections_together():
    """A full-chain run populates every optional section. Pin the
    section ordering so the rendered report stays grep-friendly."""
    result = _base_result(
        decompose={
            "applied": 1,
            "toponyms_scanned": 1,
            "decompositions": 1,
            "scholar": 1,
            "scholar-disagreement": 0,
            "unique-zero-unaccounted": 0,
            "tiebreaker": 0,
            "no-canonical": 0,
        },
        cognates={"roots": 1, "candidates": 1},
        stratum={
            "applied": 1,
            "languages": {
                "welsh": {"proposed": 1, "written": 1, "skipped": 0},
            },
        },
        english_shaped={"candidates": 1, "written": 1, "skipped_no_input": 0},
        period_forms={
            "rows_scanned": 1,
            "rows_projected": 1,
            "candidates": 1,
            "rows_written": 1,
        },
    )
    md = format_enrichment_run(result)
    # Each headline appears exactly once.
    for heading in (
        "### Decomposition",
        "### Cognate clustering",
        "### Stratum classification",
        "### English-shaped derivation",
        "### Period-form projection",
    ):
        assert md.count(heading) == 1, f"section heading {heading!r} count mismatch"
