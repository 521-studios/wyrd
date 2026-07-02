"""Tests for the wyrd-vx09 forms-mining path in
``wiktextract_corpus_miner`` — ingesting Wiktionary's ``forms``
arrays as etymon_text_match rows so the bundle's per-language
_variants pool gets non-OE/ON variant data.

Covers:
- ``_meaningful_form_label`` filters noise tags + builds labels
- ``_iter_meaningful_forms`` skips self-matches + applies the label
  filter end-to-end
- ``mine_corpus_forms`` enriches existing etymons (doesn't create
  new ones) and is idempotent on re-runs
- CLI smoke (``wyrd kenning lexicon mine-wiktextract-forms``) writes
  rows in --apply mode and reports counts in dry-run.
"""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema, migrate_schema
from wyrd.generators.kenning.wiktextract_corpus_miner import (
    WIKTIONARY_FORMS_SOURCE_ID,
    _iter_meaningful_forms,
    _meaningful_form_label,
    mine_corpus_forms,
)

# --- _meaningful_form_label ----------------------------------------------


def test_meaningful_form_label_returns_none_for_no_tags() -> None:
    """Forms with no tags are noise (Wiktionary scaffolding rows)."""
    assert _meaningful_form_label({"form": "x"}) is None


def test_meaningful_form_label_skips_table_tags() -> None:
    """Wiktionary's table-rendering scaffolding rows are noise."""
    assert _meaningful_form_label({"form": "no-table-tags", "tags": ["table-tags"]}) is None
    assert (
        _meaningful_form_label(
            {"form": "cy-mut", "tags": ["inflection-template"], "source": "mutation"}
        )
        is None
    )


def test_meaningful_form_label_skips_unrecognized_forms() -> None:
    """``error-unrecognized-form`` entries are wiktextract parse
    failures — never write them as variants."""
    assert (
        _meaningful_form_label(
            {"form": "afal", "tags": ["error-unrecognized-form"], "source": "mutation"}
        )
        is None
    )


def test_meaningful_form_label_skips_verbal_conjugation() -> None:
    """Verbal conjugation isn't useful for place-name variant pools.
    Skip first/second/third-person, indicative/subjunctive,
    present/preterite/future, etc."""
    assert (
        _meaningful_form_label(
            {"form": "byddai", "tags": ["third-person", "conditional", "singular"]}
        )
        is None
    )
    assert _meaningful_form_label({"form": "ydw", "tags": ["first-person", "present"]}) is None


def test_meaningful_form_label_keeps_inflection_tags() -> None:
    """Plural / comparative / superlative / equative are meaningful
    inflection tags — pin so a future regression doesn't filter
    them as 'noise'."""
    assert _meaningful_form_label({"form": "cwn", "tags": ["plural"]}) == "plural"
    assert _meaningful_form_label({"form": "brownach", "tags": ["comparative"]}) == "comparative"
    assert _meaningful_form_label({"form": "brownaf", "tags": ["superlative"]}) == "superlative"


def test_meaningful_form_label_prefixes_mutation_source() -> None:
    """Mutation entries (source='mutation') get a ``mutation:`` prefix
    in the label so consumers can distinguish initial-mutation
    variants from plain inflection."""
    assert (
        _meaningful_form_label({"form": "hafal", "tags": ["prothesis-h"], "source": "mutation"})
        == "mutation:prothesis-h"
    )


def test_meaningful_form_label_joins_multiple_meaningful_tags() -> None:
    """When multiple non-noise tags coexist, they join with hyphens."""
    assert (
        _meaningful_form_label({"form": "X", "tags": ["feminine", "singular"]})
        == "feminine-singular"
    )


# --- _iter_meaningful_forms ----------------------------------------------


def test_iter_meaningful_forms_skips_self_matches() -> None:
    """A form identical to the canonical word is not a variant."""
    entry = {
        "word": "Brown",
        "forms": [
            {"form": "brown", "tags": ["lowercase"]},  # self-match (case-insensitive)
            {"form": "brownach", "tags": ["comparative"]},  # genuine variant
        ],
    }
    out = list(_iter_meaningful_forms(entry))
    assert ("brown", "lowercase") not in out
    assert ("brownach", "comparative") in out


def test_iter_meaningful_forms_filters_noise_and_keeps_inflection() -> None:
    """End-to-end iter: walks a real-shape entry, filters noise,
    yields only meaningful (form, label) pairs."""
    entry = {
        "word": "march",
        "forms": [
            {"form": "meirch", "tags": ["plural"]},  # keep
            {"form": "caseg", "tags": ["feminine"]},  # keep
            {"form": "no-table-tags", "source": "mutation", "tags": ["table-tags"]},
            {"form": "cy-mut", "source": "mutation", "tags": ["inflection-template"]},
            {"form": "march", "tags": ["error-unrecognized-form"], "source": "mutation"},
        ],
    }
    out = list(_iter_meaningful_forms(entry))
    assert sorted(out) == sorted([("meirch", "plural"), ("caseg", "feminine")])


def test_iter_meaningful_forms_skips_empty_forms_array() -> None:
    """Entries without a forms array yield nothing."""
    assert list(_iter_meaningful_forms({"word": "X"})) == []
    assert list(_iter_meaningful_forms({"word": "X", "forms": []})) == []


def test_iter_meaningful_forms_skips_non_dict_entries() -> None:
    """Defensive: a malformed forms array entry (string, null, etc.)
    is skipped silently rather than crashing the walk."""
    entry = {
        "word": "X",
        "forms": [
            None,
            "not-a-dict",
            {"form": "good", "tags": ["plural"]},
        ],
    }
    out = list(_iter_meaningful_forms(entry))
    assert out == [("good", "plural")]


# --- mine_corpus_forms ---------------------------------------------------


def _seed_minimal_db(db_path: Path) -> int:
    """Build a tiny lexicon with one Welsh etymon ('march') so the
    forms-mining pass has something to enrich."""
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        migrate_schema(db)
        eid = db.upsert_etymon("march", "welsh")
        db.commit()
    return eid


def _write_slice(slice_path: Path, entries: list[dict]) -> None:
    """Write a wiktextract-shaped JSONL file."""
    with slice_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def test_mine_corpus_forms_enriches_existing_etymon(tmp_path: Path) -> None:
    """wyrd-vx09: forms-mining writes etymon_text_match rows for the
    canonical etymon's variants. Pin the source_id and matched_form
    shape so consumers reading the bundle's _variants pool get the
    expected attribution."""
    db_path = tmp_path / "lexicon.db"
    eid = _seed_minimal_db(db_path)
    slice_path = tmp_path / "wiktextract_welsh.jsonl"
    _write_slice(
        slice_path,
        [
            {
                "word": "march",
                "lang_code": "cy",
                "forms": [
                    {"form": "meirch", "tags": ["plural"]},
                    {"form": "caseg", "tags": ["feminine"]},
                ],
            }
        ],
    )

    with LexiconDB(db_path) as db:
        counts = mine_corpus_forms(db, slice_path, apply=True)

    assert counts["entries_walked"] == 1
    assert counts["forms_processed"] == 2
    assert counts["forms_written"] == 2
    assert counts["etymons_missing"] == 0

    # Verify the rows were actually written.
    with LexiconDB(db_path) as db:
        rows = list(
            db.conn.execute(
                "SELECT matched_form, source_id, snippet "
                "FROM etymon_text_match WHERE etymon_id = ? ORDER BY matched_form",
                (eid,),
            )
        )
    assert [(r["matched_form"], r["source_id"], r["snippet"]) for r in rows] == [
        ("caseg", WIKTIONARY_FORMS_SOURCE_ID, "feminine"),
        ("meirch", WIKTIONARY_FORMS_SOURCE_ID, "plural"),
    ]


def test_mine_corpus_forms_skips_when_etymon_absent(tmp_path: Path) -> None:
    """Entries whose canonical headword isn't already in the lexicon
    are SKIPPED (counted into ``etymons_missing``), not created. The
    forms-mining path enriches existing rows; ``mine_corpus`` is the
    headword-creation path."""
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        migrate_schema(db)
    slice_path = tmp_path / "wiktextract_welsh.jsonl"
    _write_slice(
        slice_path,
        [
            {
                "word": "march",  # NOT in DB
                "lang_code": "cy",
                "forms": [{"form": "meirch", "tags": ["plural"]}],
            }
        ],
    )
    with LexiconDB(db_path) as db:
        counts = mine_corpus_forms(db, slice_path, apply=True)
    assert counts["etymons_missing"] == 1
    assert counts["forms_written"] == 0


def test_mine_corpus_forms_skips_entry_without_word(tmp_path: Path) -> None:
    """An entry with a missing/blank ``word`` is a silent skip — it bumps
    ``entries_walked`` but is neither a missing-etymon nor a write. Pins the
    no-word guard in _process_corpus_entry (wyrd-8uvi)."""
    db_path = tmp_path / "lexicon.db"
    _seed_minimal_db(db_path)  # seeds welsh 'march'
    slice_path = tmp_path / "wiktextract_welsh.jsonl"
    _write_slice(
        slice_path,
        [
            {"lang_code": "cy", "forms": [{"form": "x", "tags": ["plural"]}]},  # no 'word'
            {"word": "march", "lang_code": "cy", "forms": [{"form": "meirch", "tags": ["plural"]}]},
        ],
    )
    with LexiconDB(db_path) as db:
        counts = mine_corpus_forms(db, slice_path, apply=True)
    assert counts["entries_walked"] == 2  # both walked
    assert counts["etymons_missing"] == 0  # the no-word entry is NOT a missing-etymon
    assert counts["forms_written"] == 1  # only the valid 'march' entry wrote


def test_mine_corpus_forms_skips_entry_without_language(tmp_path: Path) -> None:
    """An entry with a ``word`` but no resolvable language is a silent skip
    BEFORE the etymon lookup — distinct from the etymon-absent path (it does
    NOT increment ``etymons_missing`` even though 'march' exists). Pins the
    no-language guard in _process_corpus_entry (wyrd-8uvi)."""
    db_path = tmp_path / "lexicon.db"
    _seed_minimal_db(db_path)  # seeds welsh 'march'
    slice_path = tmp_path / "wiktextract_welsh.jsonl"
    _write_slice(
        slice_path,
        # 'march' IS in the DB, but no lang_code -> _canonical_language() == ""
        # -> short-circuit skip before the etymon SELECT.
        [{"word": "march", "forms": [{"form": "meirch", "tags": ["plural"]}]}],
    )
    with LexiconDB(db_path) as db:
        counts = mine_corpus_forms(db, slice_path, apply=True)
    assert counts["entries_walked"] == 1
    assert counts["etymons_missing"] == 0  # NOT counted as missing — skipped earlier
    assert counts["forms_written"] == 0


def test_mine_corpus_forms_dry_run_does_not_write(tmp_path: Path) -> None:
    """Without ``--apply``, the run only counts; no rows written."""
    db_path = tmp_path / "lexicon.db"
    eid = _seed_minimal_db(db_path)
    slice_path = tmp_path / "wiktextract_welsh.jsonl"
    _write_slice(
        slice_path,
        [
            {
                "word": "march",
                "lang_code": "cy",
                "forms": [{"form": "meirch", "tags": ["plural"]}],
            }
        ],
    )
    with LexiconDB(db_path) as db:
        counts = mine_corpus_forms(db, slice_path, apply=False)
        rows_after = db.conn.execute(
            "SELECT COUNT(*) FROM etymon_text_match WHERE etymon_id = ?",
            (eid,),
        ).fetchone()[0]
    assert counts["forms_processed"] == 1
    assert counts["forms_written"] == 0  # dry-run reports 0 writes
    assert rows_after == 0  # no rows actually written


def test_mine_corpus_forms_idempotent_on_rerun(tmp_path: Path) -> None:
    """Re-running with --apply increments match_count rather than
    creating duplicate rows. Pinned via the (etymon_id, source_id,
    matched_form) UNIQUE on etymon_text_match."""
    db_path = tmp_path / "lexicon.db"
    eid = _seed_minimal_db(db_path)
    slice_path = tmp_path / "wiktextract_welsh.jsonl"
    _write_slice(
        slice_path,
        [
            {
                "word": "march",
                "lang_code": "cy",
                "forms": [{"form": "meirch", "tags": ["plural"]}],
            }
        ],
    )
    # Apply twice.
    with LexiconDB(db_path) as db:
        mine_corpus_forms(db, slice_path, apply=True)
    with LexiconDB(db_path) as db:
        mine_corpus_forms(db, slice_path, apply=True)
        rows = list(
            db.conn.execute(
                "SELECT match_count FROM etymon_text_match WHERE etymon_id = ?",
                (eid,),
            )
        )
    # One row, match_count incremented to 2.
    assert len(rows) == 1
    assert rows[0]["match_count"] == 2


def test_mine_corpus_forms_skips_self_matched_forms(tmp_path: Path) -> None:
    """A form identical to the canonical word doesn't write a row —
    no value in storing 'march' as a variant of 'march'."""
    db_path = tmp_path / "lexicon.db"
    eid = _seed_minimal_db(db_path)
    slice_path = tmp_path / "wiktextract_welsh.jsonl"
    _write_slice(
        slice_path,
        [
            {
                "word": "march",
                "lang_code": "cy",
                "forms": [
                    {"form": "march", "tags": ["singular"]},  # self-match: skip
                    {"form": "meirch", "tags": ["plural"]},  # write
                ],
            }
        ],
    )
    with LexiconDB(db_path) as db:
        counts = mine_corpus_forms(db, slice_path, apply=True)
        rows = list(
            db.conn.execute(
                "SELECT matched_form FROM etymon_text_match WHERE etymon_id = ?",
                (eid,),
            )
        )
    assert counts["forms_written"] == 1
    assert {r["matched_form"] for r in rows} == {"meirch"}


def test_mine_corpus_forms_counts_noise_skipped(tmp_path: Path) -> None:
    """The ``forms_skipped_noise`` counter tracks how many entries
    in each entry's forms array were filtered out (table-tags /
    inflection-template / verbal conjugation / etc.). Surfaces in
    the dashboard so operators can tell how much of a slice's forms
    array is structural noise vs meaningful inflection."""
    db_path = tmp_path / "lexicon.db"
    _seed_minimal_db(db_path)
    slice_path = tmp_path / "wiktextract_welsh.jsonl"
    _write_slice(
        slice_path,
        [
            {
                "word": "march",
                "lang_code": "cy",
                "forms": [
                    # 1 meaningful + 3 noise
                    {"form": "meirch", "tags": ["plural"]},
                    {"form": "no-table-tags", "tags": ["table-tags"]},
                    {"form": "cy-mut", "tags": ["inflection-template"]},
                    {"form": "ydw", "tags": ["first-person", "present"]},
                ],
            }
        ],
    )
    with LexiconDB(db_path) as db:
        counts = mine_corpus_forms(db, slice_path, apply=True)
    assert counts["forms_processed"] == 1
    assert counts["forms_skipped_noise"] == 3


def test_mine_corpus_forms_surface_in_fetch_member_variants(tmp_path: Path) -> None:
    """End-to-end audit-gap closure (wyrd-f6v4): rows written by
    mine_corpus_forms actually surface in the bundle's _variants
    pool via ``_fetch_member_variants`` (the function _gather_family
    calls at bundle-export time). Pin the full chain — without this,
    a future regression that drops 'wiktionary-forms' from the
    variant lookup wouldn't be caught by the unit tests above."""
    from wyrd.generators.kenning.lexicon import _fetch_member_variants

    db_path = tmp_path / "lexicon.db"
    eid = _seed_minimal_db(db_path)
    slice_path = tmp_path / "wiktextract_welsh.jsonl"
    _write_slice(
        slice_path,
        [
            {
                "word": "march",
                "lang_code": "cy",
                "forms": [{"form": "meirch", "tags": ["plural"]}],
            }
        ],
    )
    with LexiconDB(db_path) as db:
        mine_corpus_forms(db, slice_path, apply=True)
        # _fetch_member_variants returns {member_id: list[(form,
        # weight)]} keyed by etymon row id (it's per-member; the
        # bundle export later remaps to lang_field via the family
        # rollup). Pin the per-etymon variant pool here.
        variants = _fetch_member_variants(db, [eid], {"march"})
    assert eid in variants
    forms = {form for form, _weight in variants[eid]}
    assert "meirch" in forms


def test_mine_corpus_forms_modern_english_plurals_and_verbs(tmp_path: Path) -> None:
    """wyrd-m032: modern-english slice mining keeps plurals /
    comparatives / superlatives and filters English verb conjugations.

    Pins the noise-filter behavior on the modern-english corpus
    specifically. The wyrd-r1ks expansion added ``past`` /
    ``infinitive`` / ``participle`` / etc. to filter OE verb forms;
    those tags appear on English verb-form entries too, so this
    test guards against a future filter regression that would
    re-leak ``walked`` / ``walking`` into the noun variant pool.
    """
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        migrate_schema(db)
        ancient_id = db.upsert_etymon("ancient", "modern-english")
        walk_id = db.upsert_etymon("walk", "modern-english")
        db.commit()
    slice_path = tmp_path / "wiktextract_english.jsonl"
    _write_slice(
        slice_path,
        [
            {
                "word": "ancient",
                "lang_code": "en",
                "forms": [
                    {"form": "ancients", "tags": ["plural"]},
                    {"form": "more ancient", "tags": ["comparative"]},
                    {"form": "most ancient", "tags": ["superlative"]},
                ],
            },
            {
                "word": "walk",
                "lang_code": "en",
                "forms": [
                    {"form": "walks", "tags": ["plural"]},  # keep (noun plural)
                    {"form": "walked", "tags": ["past"]},  # filter (verb)
                    {"form": "walking", "tags": ["participle"]},  # filter (verb)
                    {"form": "walks", "tags": ["third-person", "present"]},  # filter
                ],
            },
        ],
    )
    with LexiconDB(db_path) as db:
        counts = mine_corpus_forms(db, slice_path, apply=True)
        rows = list(
            db.conn.execute(
                "SELECT etymon_id, matched_form, snippet "
                "FROM etymon_text_match "
                "WHERE source_id = ? ORDER BY matched_form",
                (WIKTIONARY_FORMS_SOURCE_ID,),
            )
        )

    assert counts["entries_walked"] == 2
    # 4 kept (3 ancient + 1 noun-plural walk), 3 verb-form-noise skipped.
    assert counts["forms_written"] == 4
    assert counts["forms_skipped_noise"] == 3

    written = {(r["matched_form"], r["snippet"]) for r in rows}
    assert written == {
        ("ancients", "plural"),
        ("more ancient", "comparative"),
        ("most ancient", "superlative"),
        ("walks", "plural"),
    }
    # Verify cross-etymon attribution (each form attached to the
    # correct headword's row, not collapsed onto one).
    walk_forms = {r["matched_form"] for r in rows if r["etymon_id"] == walk_id}
    assert walk_forms == {"walks"}
    ancient_forms = {r["matched_form"] for r in rows if r["etymon_id"] == ancient_id}
    assert ancient_forms == {"ancients", "more ancient", "most ancient"}


def test_mine_corpus_forms_respects_limit(tmp_path: Path) -> None:
    """``limit=N`` stops after N entries — useful for smoke-testing
    a slice before committing to a full ingest."""
    db_path = tmp_path / "lexicon.db"
    _seed_minimal_db(db_path)
    with LexiconDB(db_path) as db:
        db.upsert_etymon("ci", "welsh")
        db.upsert_etymon("ty", "welsh")
        db.commit()
    slice_path = tmp_path / "wiktextract_welsh.jsonl"
    _write_slice(
        slice_path,
        [
            {"word": "march", "lang_code": "cy", "forms": [{"form": "meirch", "tags": ["plural"]}]},
            {"word": "ci", "lang_code": "cy", "forms": [{"form": "cwn", "tags": ["plural"]}]},
            {"word": "ty", "lang_code": "cy", "forms": [{"form": "tai", "tags": ["plural"]}]},
        ],
    )
    with LexiconDB(db_path) as db:
        counts = mine_corpus_forms(db, slice_path, apply=True, limit=2)
    assert counts["entries_walked"] == 2


# --- CLI -----------------------------------------------------------------


def test_cli_mine_wiktextract_forms_apply_writes_rows(tmp_path: Path) -> None:
    """End-to-end CLI: ``wyrd kenning lexicon mine-wiktextract-forms
    SLICE_PATH --apply --db DB_PATH`` writes rows + reports counts."""
    db_path = tmp_path / "lexicon.db"
    _seed_minimal_db(db_path)
    slice_path = tmp_path / "wiktextract_welsh.jsonl"
    _write_slice(
        slice_path,
        [
            {
                "word": "march",
                "lang_code": "cy",
                "forms": [{"form": "meirch", "tags": ["plural"]}],
            }
        ],
    )
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "lexicon",
            "mine-wiktextract-forms",
            str(slice_path),
            "--db",
            str(db_path),
            "--apply",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    # stderr (where Click writes the count summary) is captured in
    # result.output for click.testing.CliRunner.
    assert "forms_written=1" in result.output
    assert "--apply" in result.output


def test_cli_mine_wiktextract_forms_dry_run_reports_zero_written(
    tmp_path: Path,
) -> None:
    """Without ``--apply``, the CLI reports forms_written=0 even
    when forms are processed — the count tracks actual writes."""
    db_path = tmp_path / "lexicon.db"
    _seed_minimal_db(db_path)
    slice_path = tmp_path / "wiktextract_welsh.jsonl"
    _write_slice(
        slice_path,
        [
            {
                "word": "march",
                "lang_code": "cy",
                "forms": [{"form": "meirch", "tags": ["plural"]}],
            }
        ],
    )
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "lexicon",
            "mine-wiktextract-forms",
            str(slice_path),
            "--db",
            str(db_path),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0
    assert "forms_processed=1" in result.output
    assert "forms_written=0" in result.output
    assert "(dry-run)" in result.output
