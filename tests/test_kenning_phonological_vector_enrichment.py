"""Tests for the phonological-vector enrichment pass (wyrd-kq7w.1).

Companion to ``test_kenning_phonological_vector_compute.py``. That
file tests the IPA → vector computation; this file tests the
``tag_phonological_vectors_all`` orchestrator that walks the etymon
table and persists vectors as JSON blobs.

Fixture pattern: in-memory schema-only LexiconDB (via init_schema),
hand-insert a handful of etymon rows covering IPA-present and
IPA-absent cases, then run the pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wyrd.generators.kenning.lexicon import LexiconDB, init_schema
from wyrd.generators.kenning.lexicon.phonological_vector_enrichment import (
    PHON_VECTOR_METHOD_VERSION,
    phonological_vector_coverage,
    tag_phonological_vectors_all,
)


def _seed_etymons(db: LexiconDB, rows: list[tuple[str, str, str | None]]) -> None:
    """Insert a small set of etymon rows for enrichment-pass testing.

    rows: list of (language, canonical_form, ipa) tuples. id auto-assigned.
    """
    for lang, canonical, ipa in rows:
        db.conn.execute(
            "INSERT INTO etymon (language, canonical_form, pronunciation_ipa) VALUES (?, ?, ?)",
            (lang, canonical, ipa),
        )
    db.conn.commit()


@pytest.fixture
def db_with_seed(tmp_path: Path) -> LexiconDB:
    """Synthetic lexicon DB seeded with 4 etymons across language families.

    Covers: IPA-present English, IPA-present Italian (vowel-final),
    IPA-present Hawaiian (all-soft), IPA-absent fallback path.
    """
    db_path = tmp_path / "test.db"
    init_schema(db_path)
    db = LexiconDB(db_path)
    _seed_etymons(
        db,
        [
            ("modern-english", "strict", "strɪkt"),
            ("italian", "casa", "kaza"),
            ("hawaiian", "aloha", "aloha"),
            ("welsh", "cwm", None),  # IPA-absent → orthographic fallback
        ],
    )
    return db


# ---- dry-run behavior ----------------------------------------------------


def test_dry_run_counts_candidates_without_writing(db_with_seed: LexiconDB):
    """Dry-run reports the full candidate count + a 5-row sample but
    does NOT write the phonological_vector column."""
    result = tag_phonological_vectors_all(db_with_seed, apply=False)
    assert result["candidates"] == 4
    assert result["written"] == 0
    # Sample is populated for operator spot-check (capped at 5)
    assert len(result["sample"]) == 4
    # Column remains NULL
    row = db_with_seed.conn.execute(
        "SELECT COUNT(*) AS n FROM etymon WHERE phonological_vector IS NOT NULL"
    ).fetchone()
    assert row["n"] == 0


def test_dry_run_sample_includes_computed_blobs(db_with_seed: LexiconDB):
    """Sample tuples are (etymon_id, canonical_form, json_blob) — the
    json_blob is the actual computed vector so an operator can sanity-
    check shape without committing."""
    result = tag_phonological_vectors_all(db_with_seed, apply=False)
    for _etymon_id, _canonical_form, blob in result["sample"]:
        parsed = json.loads(blob)
        # JSON shape: dict with the 14 named dimensions
        assert "cluster_density" in parsed
        assert "vowel_final_bias" in parsed
        # All dimension values are floats
        for k, v in parsed.items():
            if k == "extras":
                continue
            assert isinstance(v, (int, float)), f"non-float dim {k}: {v!r}"


# ---- apply behavior ------------------------------------------------------


def test_apply_writes_phonological_vector_blob(db_with_seed: LexiconDB):
    """--apply persists computed vectors into the etymon column."""
    result = tag_phonological_vectors_all(db_with_seed, apply=True)
    assert result["candidates"] == 4
    assert result["written"] == 4
    row = db_with_seed.conn.execute(
        "SELECT COUNT(*) AS n FROM etymon WHERE phonological_vector IS NOT NULL"
    ).fetchone()
    assert row["n"] == 4


def test_apply_writes_correct_blob_for_italian_casa(db_with_seed: LexiconDB):
    """Per-row spot check: Italian 'casa' should have vowel_final_bias=1.0."""
    tag_phonological_vectors_all(db_with_seed, apply=True)
    row = db_with_seed.conn.execute(
        "SELECT phonological_vector FROM etymon WHERE canonical_form = 'casa'"
    ).fetchone()
    parsed = json.loads(row["phonological_vector"])
    assert parsed["vowel_final_bias"] == 1.0
    assert parsed["final_fortition"] == 0.0


def test_apply_uses_orthographic_fallback_for_ipa_absent_rows(
    db_with_seed: LexiconDB,
):
    """Welsh 'cwm' has IPA=None → orthographic fallback path. The
    vector should still be populated (not NULL), even though the
    dimensions that require IPA precision read 0."""
    tag_phonological_vectors_all(db_with_seed, apply=True)
    row = db_with_seed.conn.execute(
        "SELECT phonological_vector FROM etymon WHERE canonical_form = 'cwm'"
    ).fetchone()
    assert row["phonological_vector"] is not None
    parsed = json.loads(row["phonological_vector"])
    # Fallback path can't classify retroflexion / palatalization /
    # pharyngeal / aspiration / vowel_height / vowel_backness from
    # orthography alone → those dimensions read 0.0
    assert parsed["palatalization"] == 0.0
    assert parsed["retroflexion"] == 0.0
    assert parsed["pharyngeal"] == 0.0
    assert parsed["aspirated_voiceless"] == 0.0
    assert parsed["vowel_height"] == 0.0
    assert parsed["vowel_backness"] == 0.0
    # But the orthographic dimensions DO get populated
    assert parsed["final_fortition"] == 0.0  # cwm ends in 'm', not a stop
    assert parsed["soft_consonants"] > 0.0  # m is soft (nasal)


# ---- incremental vs force ------------------------------------------------


def test_incremental_skips_already_tagged_rows(db_with_seed: LexiconDB):
    """Default mode is NULL-only — re-running the pass on a fully
    tagged DB does nothing."""
    tag_phonological_vectors_all(db_with_seed, apply=True)
    # Second run with the same DB
    result = tag_phonological_vectors_all(db_with_seed, apply=True)
    assert result["candidates"] == 0
    assert result["written"] == 0


def test_force_recomputes_all_rows(db_with_seed: LexiconDB):
    """--force walks every row regardless of existing column value."""
    tag_phonological_vectors_all(db_with_seed, apply=True)
    result = tag_phonological_vectors_all(db_with_seed, apply=True, force=True)
    assert result["candidates"] == 4
    assert result["written"] == 4
    assert result["force"] is True


# ---- idempotence + determinism -------------------------------------------


def test_idempotent_re_runs_produce_byte_identical_blobs(
    db_with_seed: LexiconDB,
):
    """The bit-stability contract: re-running the pass on an unchanged
    DB writes byte-identical phonological_vector blobs. Critical for
    the L2/L3 round-trip property."""
    tag_phonological_vectors_all(db_with_seed, apply=True)
    blobs_first = {
        row["canonical_form"]: row["phonological_vector"]
        for row in db_with_seed.conn.execute(
            "SELECT canonical_form, phonological_vector FROM etymon"
        )
    }
    # Force-recompute and verify byte-identity
    tag_phonological_vectors_all(db_with_seed, apply=True, force=True)
    blobs_second = {
        row["canonical_form"]: row["phonological_vector"]
        for row in db_with_seed.conn.execute(
            "SELECT canonical_form, phonological_vector FROM etymon"
        )
    }
    assert blobs_first == blobs_second, "re-run produced different blobs"


def test_method_version_stamp_is_stable():
    """The method-version constant is part of the audit-trail contract;
    re-naming it requires intentional revision."""
    assert PHON_VECTOR_METHOD_VERSION == "compute-phon-vector-v1"


# ---- per-row exception isolation -----------------------------------------


def test_per_row_failure_does_not_abort_batch(tmp_path: Path):
    """If a single etymon raises during compute, the rest of the batch
    still runs. The failure surfaces in the result dict's `failed`
    count + `sample_failures` list rather than crashing the whole
    enrichment pass mid-batch. Caught by round-1 silent-failure-hunter."""
    db_path = tmp_path / "test.db"
    init_schema(db_path)
    db = LexiconDB(db_path)
    # Seed a valid row + an invalid one + another valid row.
    # Use the IPA column to inject a value of the wrong type — sqlite
    # silently stores bytes when given bytes, which trips the NFC
    # normalize step on read.
    db.conn.execute(
        "INSERT INTO etymon (language, canonical_form, pronunciation_ipa) VALUES (?, ?, ?)",
        ("modern-english", "good_pre", "kæt"),
    )
    db.conn.execute(
        "INSERT INTO etymon (language, canonical_form, pronunciation_ipa) VALUES (?, ?, ?)",
        ("modern-english", "bad_row", b"\xff\xfe"),  # bytes; raises in NFC
    )
    db.conn.execute(
        "INSERT INTO etymon (language, canonical_form, pronunciation_ipa) VALUES (?, ?, ?)",
        ("modern-english", "good_post", "dɒg"),
    )
    db.conn.commit()
    result = tag_phonological_vectors_all(db, apply=True)
    # The 2 good rows still got written even though one row failed
    assert result["written"] == 2
    assert result["failed"] == 1
    # Failure sample surfaces the offending etymon's canonical_form for
    # operator audit
    assert any(canonical == "bad_row" for _id, canonical, _msg in result["sample_failures"])
    # Coverage reflects partial state
    row = db.conn.execute(
        "SELECT canonical_form FROM etymon WHERE phonological_vector IS NOT NULL"
    ).fetchall()
    canonical_forms = {r["canonical_form"] for r in row}
    assert canonical_forms == {"good_pre", "good_post"}


# ---- coverage report -----------------------------------------------------


def test_coverage_report_zero_before_apply(db_with_seed: LexiconDB):
    cov = phonological_vector_coverage(db_with_seed.conn)
    assert cov["total"] == 4
    assert cov["tagged"] == 0
    assert cov["ratio"] == 0.0
    assert cov["by_language"] == []


def test_coverage_report_full_after_apply(db_with_seed: LexiconDB):
    tag_phonological_vectors_all(db_with_seed, apply=True)
    cov = phonological_vector_coverage(db_with_seed.conn)
    assert cov["total"] == 4
    assert cov["tagged"] == 4
    assert cov["ratio"] == 1.0
    # by_language sorts alphabetically; each row is (lang, total, tagged)
    langs = {entry[0] for entry in cov["by_language"]}
    assert langs == {"modern-english", "italian", "hawaiian", "welsh"}


def test_coverage_report_partial_after_partial_apply(db_with_seed: LexiconDB):
    """When only some rows are tagged (e.g. mid-batch failure), the
    coverage report surfaces the partial state per-language. Simulate
    by manually writing to one row only."""
    db_with_seed.conn.execute(
        "UPDATE etymon SET phonological_vector = '{}' WHERE canonical_form = 'aloha'"
    )
    db_with_seed.conn.commit()
    cov = phonological_vector_coverage(db_with_seed.conn)
    assert cov["total"] == 4
    assert cov["tagged"] == 1
    assert cov["ratio"] == 0.25
    # Only the language with a tagged row appears
    langs_tagged = [entry for entry in cov["by_language"] if entry[2] > 0]
    assert len(langs_tagged) == 1
    assert langs_tagged[0][0] == "hawaiian"
    assert langs_tagged[0][2] == 1
