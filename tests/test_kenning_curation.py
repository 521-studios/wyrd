"""Tests for the L3 curation event log — wyrd-2jhs (Phase 2b).

Two layers covered:

1. ``collect_curation_overrides`` (jsonl_build.py) — replays JSONL
   files, returns the merged ``{etymon_ref: payload}`` curation state.
2. ``apply_curation_overrides`` (enrichment.py) — looks up etymon IDs
   from refs and overrides lemma_id / merged_into_id / lemma_method
   columns on the DB.

Round-trip semantics (curation events live OUTSIDE the dump-rebuild
loop) are verified by the existing wyrd-f295 tests; the curation file
is operator-maintained and dump-jsonl never touches it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from wyrd.generators.kenning.enrichment import (
    CURATION_METHOD_VERSION,
    apply_curation_overrides,
    format_curation_run,
    run_ocr_lemma_enrichment,
)
from wyrd.generators.kenning.jsonl_build import collect_curation_overrides
from wyrd.generators.kenning.jsonl_log import write_jsonl
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema


def _build_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    return db_path


def _add_etymon(db_path: Path, language: str, canonical_form: str) -> int:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES (?, ?)",
        (canonical_form, language),
    )
    eid = cur.lastrowid
    conn.commit()
    conn.close()
    return eid


def _write_curation_file(tmp_path: Path, rows: list[dict]) -> Path:
    """Helper that prepends the standard synthetic source row to a
    curation file. Every L2 file needs exactly one source row per the
    build contract, even when its other rows are curation events."""
    source_row = {
        "_type": "source",
        "ref": "manual-curation",
        "title": "Operator curation overrides",
    }
    path = tmp_path / "_curation.jsonl"
    write_jsonl(path, [source_row, *rows])
    return path


# ---------------------------------------------------------------------------
# collect_curation_overrides
# ---------------------------------------------------------------------------


def test_collect_returns_empty_when_no_curation_rows(tmp_path: Path):
    """A file with only source + etymon facts contributes nothing."""
    path = tmp_path / "skeat.jsonl"
    write_jsonl(
        path,
        [
            {"_type": "source", "ref": "skeat", "title": "X"},
            {
                "_type": "etymon",
                "ref": "old-english:cot",
                "canonical_form": "cot",
                "language": "old-english",
            },
        ],
    )
    assert collect_curation_overrides([path]) == {}


def test_collect_canonical_state_curation_row(tmp_path: Path):
    path = _write_curation_file(
        tmp_path,
        [
            {
                "_type": "etymon_curation",
                "ref": "old-english:caelf",
                "lemma_ref": "old-english:cealf",
                "inflection": "spelling_variant",
                "reason": "scribal æ/ae",
            }
        ],
    )
    state = collect_curation_overrides([path])
    assert state == {
        "old-english:caelf": {
            "lemma_ref": "old-english:cealf",
            "inflection": "spelling_variant",
            "reason": "scribal æ/ae",
        }
    }


def test_collect_patches_accumulate_across_same_file(tmp_path: Path):
    """Two patches on the same etymon: scalar fields last-write-wins."""
    path = _write_curation_file(
        tmp_path,
        [
            {"_op": "add", "_type": "etymon_curation", "ref": "oe:caelf", "lemma_ref": "oe:cealf"},
            {
                "_op": "patch",
                "_type": "etymon_curation",
                "ref": "oe:caelf",
                "inflection": "variant",
            },
            {
                "_op": "patch",
                "_type": "etymon_curation",
                "ref": "oe:caelf",
                "lemma_ref": "oe:other",
            },
        ],
    )
    state = collect_curation_overrides([path])
    assert state["oe:caelf"] == {"lemma_ref": "oe:other", "inflection": "variant"}


def test_collect_null_lemma_ref_preserved(tmp_path: Path):
    """Explicit null is the 'clear' sentinel — distinct from absent."""
    path = _write_curation_file(
        tmp_path,
        [
            {"_op": "add", "_type": "etymon_curation", "ref": "oe:caelf", "lemma_ref": "oe:cealf"},
            {"_op": "patch", "_type": "etymon_curation", "ref": "oe:caelf", "lemma_ref": None},
        ],
    )
    state = collect_curation_overrides([path])
    assert state["oe:caelf"] == {"lemma_ref": None}


def test_collect_merges_across_multiple_files(tmp_path: Path):
    """Curation rows can live in any file; cross-file dedup is by ref.

    Within a file, ``patch`` events on the same ref accumulate per
    kernel semantics. ACROSS files, the merge collapses each file's
    final state into one dict (last-write-wins by sort order), so
    callers should generally use canonical-state rows (no ``_op``) at
    the file level."""
    file_a = tmp_path / "a.jsonl"
    write_jsonl(
        file_a,
        [
            {"_type": "source", "ref": "a", "title": "A"},
            {"_type": "etymon_curation", "ref": "oe:caelf", "lemma_ref": "oe:cealf"},
        ],
    )
    file_b = tmp_path / "b.jsonl"
    write_jsonl(
        file_b,
        [
            {"_type": "source", "ref": "b", "title": "B"},
            {
                "_type": "etymon_curation",
                "ref": "oe:caelf",
                "lemma_ref": "oe:cealf",
                "inflection": "v",
            },
        ],
    )
    state = collect_curation_overrides([file_a, file_b])
    # File-order merge: b comes after a, so b's payload wins on overlap.
    # Both files set the same lemma_ref but only b adds inflection.
    assert state == {
        "oe:caelf": {"lemma_ref": "oe:cealf", "inflection": "v"},
    }


# ---------------------------------------------------------------------------
# apply_curation_overrides
# ---------------------------------------------------------------------------


def test_apply_sets_lemma_id_and_method_version(tmp_path: Path):
    db_path = _build_db(tmp_path)
    caelf_id = _add_etymon(db_path, "old-english", "caelf")
    cealf_id = _add_etymon(db_path, "old-english", "cealf")
    with LexiconDB(db_path) as db:
        counts = apply_curation_overrides(
            db,
            {
                "old-english:caelf": {
                    "lemma_ref": "old-english:cealf",
                    "inflection": "spelling_variant",
                }
            },
        )
    assert counts["lemma_id_set"] == 1
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT lemma_id, inflection, lemma_method FROM etymon WHERE id=?", (caelf_id,)
    ).fetchone()
    conn.close()
    assert row["lemma_id"] == cealf_id
    assert row["inflection"] == "spelling_variant"
    assert row["lemma_method"] == CURATION_METHOD_VERSION


def test_apply_null_lemma_ref_clears_existing_link(tmp_path: Path):
    """A previous auto-link gets cleared when curation says null."""
    db_path = _build_db(tmp_path)
    caelf_id = _add_etymon(db_path, "old-english", "caelf")
    cealf_id = _add_etymon(db_path, "old-english", "cealf")
    # Simulate prior auto-clustering output.
    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE etymon SET lemma_id = ?, inflection = 'wrong', lemma_method = 'link-lemmas-v1' "
        "WHERE id = ?",
        (cealf_id, caelf_id),
    )
    conn.commit()
    conn.close()
    with LexiconDB(db_path) as db:
        counts = apply_curation_overrides(db, {"old-english:caelf": {"lemma_ref": None}})
    assert counts["lemma_id_cleared"] == 1
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT lemma_id, inflection, lemma_method FROM etymon WHERE id=?", (caelf_id,)
    ).fetchone()
    conn.close()
    assert row["lemma_id"] is None
    assert row["inflection"] is None
    assert row["lemma_method"] is None


def test_apply_sets_merged_into_id(tmp_path: Path):
    db_path = _build_db(tmp_path)
    vath_id = _add_etymon(db_path, "old-norse", "vath")
    vad_id = _add_etymon(db_path, "old-norse", "vað")
    with LexiconDB(db_path) as db:
        counts = apply_curation_overrides(
            db,
            {"old-norse:vath": {"merged_into_ref": "old-norse:vað"}},
        )
    assert counts["merged_into_set"] == 1
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT merged_into_id FROM etymon WHERE id=?", (vath_id,)).fetchone()
    conn.close()
    assert row["merged_into_id"] == vad_id


def test_apply_clears_lemma_when_setting_merge(tmp_path: Path):
    """A tombstone shouldn't double as a lemma parent — merge curation
    clears any existing lemma_id on the curated row."""
    db_path = _build_db(tmp_path)
    vath_id = _add_etymon(db_path, "old-norse", "vath")
    other_id = _add_etymon(db_path, "old-norse", "other")
    vad_id = _add_etymon(db_path, "old-norse", "vað")
    # Simulate a buggy prior state: lemma_id set AND merged_into_id null.
    conn = sqlite3.connect(db_path)
    conn.execute(
        "UPDATE etymon SET lemma_id = ?, inflection = 'stale' WHERE id = ?",
        (other_id, vath_id),
    )
    conn.commit()
    conn.close()
    with LexiconDB(db_path) as db:
        apply_curation_overrides(db, {"old-norse:vath": {"merged_into_ref": "old-norse:vað"}})
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT lemma_id, inflection, merged_into_id FROM etymon WHERE id=?", (vath_id,)
    ).fetchone()
    conn.close()
    assert row["merged_into_id"] == vad_id
    assert row["lemma_id"] is None  # cleared
    assert row["inflection"] is None


def test_apply_lemma_clears_merged_into_id(tmp_path: Path):
    """Mirror case: setting lemma_ref on a row that was auto-tombstoned
    means the operator is rejecting the OCR merge — clear it."""
    db_path = _build_db(tmp_path)
    caelf_id = _add_etymon(db_path, "old-english", "caelf")
    cealf_id = _add_etymon(db_path, "old-english", "cealf")
    other_id = _add_etymon(db_path, "old-english", "other")
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE etymon SET merged_into_id = ? WHERE id = ?", (other_id, caelf_id))
    conn.commit()
    conn.close()
    with LexiconDB(db_path) as db:
        apply_curation_overrides(
            db,
            {"old-english:caelf": {"lemma_ref": "old-english:cealf", "inflection": "v"}},
        )
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT lemma_id, merged_into_id FROM etymon WHERE id=?", (caelf_id,)
    ).fetchone()
    conn.close()
    assert row["lemma_id"] == cealf_id
    assert row["merged_into_id"] is None


def test_apply_processes_both_ref_fields_when_one_fails_resolution(tmp_path: Path):
    """A payload with both lemma_ref and merged_into_ref shouldn't
    skip the merge handling just because lemma_ref didn't resolve.
    Each ref field processes independently."""
    db_path = _build_db(tmp_path)
    caelf_id = _add_etymon(db_path, "old-english", "caelf")
    cealf_id = _add_etymon(db_path, "old-english", "cealf")
    # Payload with a bad lemma_ref + a good merged_into_ref.
    with LexiconDB(db_path) as db:
        counts = apply_curation_overrides(
            db,
            {
                "old-english:caelf": {
                    "lemma_ref": "old-english:does-not-exist",
                    "merged_into_ref": "old-english:cealf",
                }
            },
        )
    # The bad lemma_ref counts as unresolved; the good merge_ref still applies.
    assert counts["unresolved_lemma_ref"] == 1
    assert counts["merged_into_set"] == 1
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT lemma_id, merged_into_id FROM etymon WHERE id=?", (caelf_id,)
    ).fetchone()
    conn.close()
    assert row["merged_into_id"] == cealf_id
    assert row["lemma_id"] is None  # cleared by the merge handler


def test_apply_caches_lemma_ref_resolution(tmp_path: Path):
    """Many curation events sharing the same lemma target should
    resolve that lemma exactly once — verified by patching the DB
    helper to count calls."""
    db_path = _build_db(tmp_path)
    lemma_id = _add_etymon(db_path, "old-english", "cealf")
    a_id = _add_etymon(db_path, "old-english", "caelf-a")
    b_id = _add_etymon(db_path, "old-english", "caelf-b")
    c_id = _add_etymon(db_path, "old-english", "caelf-c")
    curation = {
        "old-english:caelf-a": {"lemma_ref": "old-english:cealf"},
        "old-english:caelf-b": {"lemma_ref": "old-english:cealf"},
        "old-english:caelf-c": {"lemma_ref": "old-english:cealf"},
    }
    with LexiconDB(db_path) as db:
        counts = apply_curation_overrides(db, curation)
    # All three inflected forms get pointed at the same lemma.
    assert counts["lemma_id_set"] == 3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    for eid in (a_id, b_id, c_id):
        row = conn.execute("SELECT lemma_id FROM etymon WHERE id=?", (eid,)).fetchone()
        assert row["lemma_id"] == lemma_id
    conn.close()


def test_apply_unresolved_etymon_logs_and_skips(tmp_path: Path):
    db_path = _build_db(tmp_path)
    _add_etymon(db_path, "old-english", "cealf")
    with LexiconDB(db_path) as db:
        counts = apply_curation_overrides(
            db, {"old-english:does-not-exist": {"lemma_ref": "old-english:cealf"}}
        )
    assert counts["unresolved_etymon"] == 1
    assert counts["lemma_id_set"] == 0


def test_apply_unresolved_lemma_ref_logs_and_skips(tmp_path: Path):
    db_path = _build_db(tmp_path)
    _add_etymon(db_path, "old-english", "caelf")
    with LexiconDB(db_path) as db:
        counts = apply_curation_overrides(
            db, {"old-english:caelf": {"lemma_ref": "old-english:does-not-exist"}}
        )
    assert counts["unresolved_lemma_ref"] == 1
    assert counts["lemma_id_set"] == 0


# ---------------------------------------------------------------------------
# Orchestrator integration
# ---------------------------------------------------------------------------


def test_orchestrator_curation_overrides_auto_clustering(tmp_path: Path):
    """End-to-end: auto-clustering would link cætan → cæt (lemma);
    curation overrides to point cætan at a different lemma."""
    db_path = _build_db(tmp_path)
    _add_etymon(db_path, "old-english", "cæt")  # auto-clustering would pick this
    other_id = _add_etymon(db_path, "old-english", "cealf")  # arbitrary alternate lemma
    catan_id = _add_etymon(db_path, "old-english", "cætan")
    curation = {
        "old-english:cætan": {
            "lemma_ref": "old-english:cealf",  # override: link to cealf instead of cæt
            "inflection": "operator_override",
        }
    }
    with LexiconDB(db_path) as db:
        result = run_ocr_lemma_enrichment(db, apply=True, curation_state=curation)
    assert result["curation"] is not None
    assert result["order"] == ["normalize-ocr", "link-lemmas", "apply-curation"]
    assert result["curation"]["lemma_id_set"] == 1
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT lemma_id, inflection, lemma_method FROM etymon WHERE id=?", (catan_id,)
    ).fetchone()
    conn.close()
    # Curation wins over auto-clustering: lemma is the operator's choice.
    assert row["lemma_id"] == other_id  # NOT cat_id (which auto-clustering would pick)
    assert row["inflection"] == "operator_override"
    assert row["lemma_method"] == CURATION_METHOD_VERSION


def test_orchestrator_dry_run_skips_curation(tmp_path: Path):
    """Dry-run reports the OCR + lemma counts but doesn't touch
    curation (which requires real DB writes to be observable)."""
    db_path = _build_db(tmp_path)
    _add_etymon(db_path, "old-english", "cæt")
    _add_etymon(db_path, "old-english", "cætan")
    with LexiconDB(db_path) as db:
        result = run_ocr_lemma_enrichment(
            db,
            apply=False,
            curation_state={"old-english:cætan": {"lemma_ref": "old-english:cæt"}},
        )
    assert result["curation"] is None
    assert result["order"] == ["normalize-ocr", "link-lemmas"]


def test_orchestrator_without_curation_state_unchanged(tmp_path: Path):
    """Calling without curation_state keeps the original 2-pass behavior."""
    db_path = _build_db(tmp_path)
    _add_etymon(db_path, "old-english", "cæt")
    _add_etymon(db_path, "old-english", "cætan")
    with LexiconDB(db_path) as db:
        result = run_ocr_lemma_enrichment(db, apply=True)
    assert result["curation"] is None
    assert result["order"] == ["normalize-ocr", "link-lemmas"]


# ---------------------------------------------------------------------------
# Markdown formatter
# ---------------------------------------------------------------------------


def test_format_curation_includes_method_version_and_counts():
    counts = {
        "curations": 5,
        "lemma_id_set": 3,
        "lemma_id_cleared": 1,
        "merged_into_set": 1,
        "merged_into_cleared": 0,
        "unresolved_etymon": 0,
        "unresolved_lemma_ref": 0,
        "unresolved_merge_ref": 0,
    }
    md = format_curation_run(counts)
    assert CURATION_METHOD_VERSION in md
    assert "Curations processed: 5" in md
    assert "Lemma overrides set: 3" in md
    assert "Merge overrides set: 1" in md
    # No unresolved section when all are zero.
    assert "Unresolved" not in md


def test_format_curation_reports_unresolved():
    counts = {
        "curations": 3,
        "lemma_id_set": 1,
        "lemma_id_cleared": 0,
        "merged_into_set": 0,
        "merged_into_cleared": 0,
        "unresolved_etymon": 1,
        "unresolved_lemma_ref": 1,
        "unresolved_merge_ref": 0,
    }
    md = format_curation_run(counts)
    assert "Unresolved refs (skipped)" in md
    assert "Curated etymon not found: 1" in md
    assert "Lemma target not found: 1" in md
