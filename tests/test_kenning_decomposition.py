"""Tests for matcher-derived decomposition multiplicity (wyrd-08m Phase 1).

Covers:
- Schema: ``toponym_decomposition`` table + indexes land via fresh-install
  + migration paths.
- ``compute_decompositions``: single-word and multi-word toponyms emit
  correct rows; idempotent on re-run; signature stability across runs.
- ``pick_canonical_decomposition``:
  - rule (a) scholar match: a single scholar's breakdown wins canonical
    with source='scholar'; multiple scholars proposing different
    breakdowns (each matching a stored decomposition) get
    source='scholar-disagreement' on every matching row.
  - rule (b) unique-zero-unaccounted: the only zero-unaccounted
    decomposition wins canonical with source='unique-zero-unaccounted'.
  - multi-zero / no-zero / all-unaccounted fall through with no
    canonical pick (Phase 1 deliberately leaves these for Phase 2's
    tiebreaker).
- CLI ``lexicon decompose``: dry-run / --apply / --toponym-id / --limit
  paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli as cli_root
from wyrd.generators.kenning.decomposition import (
    _candidates_for_usage,
    _cross_product_decompositions,
    _decomposition_payload,
    _dedupe_scholar_seqs,
    _scholar_form_sequences,
    _scholar_seq_matches_decomposition,
    _signature_for_payload,
    compute_decompositions,
    pick_canonical_decomposition,
    populate_and_pick,
)
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema, migrate_schema
from wyrd.generators.kenning.meaning import Meaning, load_meanings
from wyrd.generators.kenning.name import Name


@pytest.fixture
def fresh_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    return db_path


def _word_db_for(subjects: list[dict]) -> dict:
    """Build a runtime word_db from a list of meanings.json subjects."""
    word_db, _ = load_meanings(subjects)
    return word_db


def _make_subject(
    *, meaning: list[str], words: list[dict], modifier_tags: list[str] | None = None
) -> dict:
    return {
        "meaning": meaning,
        "modifier_tags": modifier_tags or ["topography"],
        "modifier_type": "Topographical",
        "words": words,
    }


def _seed_source(db: LexiconDB) -> None:
    db.upsert_source(id="skeat-1901", title="Skeat 1901")
    db.commit()


def _insert_toponym(db: LexiconDB, modern_name: str) -> int:
    cur = db.conn.execute(
        "INSERT INTO toponym (modern_name, region) VALUES (?, NULL)",
        (modern_name,),
    )
    db.commit()
    return cur.lastrowid


def _insert_scholar_etymology(
    db: LexiconDB,
    *,
    toponym_id: int,
    elements: list[tuple[str, str]],
    source_id: str = "skeat-1901",
) -> int:
    """Insert a toponym_etymology + element rows. Each element is a
    (canonical_form, language) pair; etymons are upserted as needed.
    Returns the toponym_etymology id."""
    cur = db.conn.execute(
        """
        INSERT INTO toponym_etymology (toponym_id, source_id, confidence)
        VALUES (?, ?, 'high')
        """,
        (toponym_id, source_id),
    )
    etymology_id = cur.lastrowid
    for ordinal, (form, language) in enumerate(elements):
        etymon_id = db.upsert_etymon(form, language)
        db.conn.execute(
            """
            INSERT INTO toponym_etymology_element
              (toponym_etymology_id, ordinal, etymon_id, inflection, surface_in_modern)
            VALUES (?, ?, ?, NULL, NULL)
            """,
            (etymology_id, ordinal, etymon_id),
        )
    db.commit()
    return etymology_id


# --- schema ---------------------------------------------------------------


def test_init_schema_creates_toponym_decomposition_table(fresh_db: Path) -> None:
    """Fresh installs pick up the ``toponym_decomposition`` table from
    ``data/lexicon.sql``."""
    with LexiconDB(fresh_db) as db:
        tables = {
            row["name"]
            for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "toponym_decomposition" in tables


def test_init_schema_creates_toponym_decomposition_indexes(fresh_db: Path) -> None:
    with LexiconDB(fresh_db) as db:
        indexes = {
            row["name"]
            for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
        }
    assert "idx_toponym_decomposition_unique" in indexes
    assert "idx_toponym_decomposition_topo" in indexes
    assert "idx_toponym_decomposition_canonical" in indexes


def test_migrate_schema_adds_toponym_decomposition_to_legacy_db(tmp_path: Path) -> None:
    """A legacy DB created before this table existed gets it added by
    ``migrate_schema``. Idempotent — second run is a no-op."""
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        db.conn.execute("DROP TABLE toponym_decomposition")
        db.commit()

    with LexiconDB(db_path) as db:
        applied = migrate_schema(db)
    assert applied["toponym_decomposition_table"] is True

    with LexiconDB(db_path) as db:
        tables = {
            row["name"]
            for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "toponym_decomposition" in tables

    # Second migrate is a no-op.
    with LexiconDB(db_path) as db:
        applied2 = migrate_schema(db)
    assert applied2["toponym_decomposition_table"] is False


# --- signature ------------------------------------------------------------


def test_signature_stable_across_runs() -> None:
    """The same payload shape always hashes to the same signature.
    Without this, idempotent re-runs of the populator would write
    duplicate rows."""
    payload = [["morpheme", "Strat-"], ["morpheme", "-ford"]]
    assert _signature_for_payload(payload) == _signature_for_payload(payload)


def test_signature_distinguishes_morpheme_kinds() -> None:
    """Two decompositions with the same surface but different kinds
    (matched vs unaccounted) hash to different signatures."""
    matched = [["morpheme", "Strat-"], ["morpheme", "-ford"]]
    one_unaccounted = [["morpheme", "Strat-"], ["unaccounted", "-ford"]]
    assert _signature_for_payload(matched) != _signature_for_payload(one_unaccounted)


def test_signature_distinguishes_slot_order() -> None:
    """Slot ordering is part of the decomposition's identity. 'Strat-'
    + '-ford' is a different shape from '-ford' + 'Strat-'."""
    forward = [["morpheme", "Strat-"], ["morpheme", "-ford"]]
    backward = [["morpheme", "-ford"], ["morpheme", "Strat-"]]
    assert _signature_for_payload(forward) != _signature_for_payload(backward)


def test_decomposition_payload_for_meaning() -> None:
    """A Meaning slot becomes ``["morpheme", usage]`` preserving dash
    markers (location)."""
    m = Meaning("Bridg-", tags=[], meanings=["Bridge"], sources={"old_english": ["brycg"]})
    payload = _decomposition_payload([m])
    assert payload == [["morpheme", "Bridg-"]]


def test_decomposition_payload_for_unaccounted_string() -> None:
    """A bare-string slot becomes ``["unaccounted", chars]``."""
    payload = _decomposition_payload(["xyz"])
    assert payload == [["unaccounted", "xyz"]]


# --- compute_decompositions ----------------------------------------------


def _two_morpheme_subjects() -> list[dict]:
    """Subjects mimicking the canonical 'Stratford' setup: two real
    morphemes ('strat-' from OE *strǣt*, '-ford' from OE *ford*)."""
    return [
        _make_subject(
            meaning=["Street"],
            words=[
                {"modern_usage": "Strat-", "old_english": ["stræt"]},
            ],
        ),
        _make_subject(
            meaning=["Ford"],
            words=[
                {"modern_usage": "-ford", "old_english": ["ford"]},
            ],
        ),
    ]


def test_compute_decompositions_writes_at_least_one_row(fresh_db: Path) -> None:
    """A toponym name that the matcher decomposes produces at least one
    ``toponym_decomposition`` row."""
    word_db = _word_db_for(_two_morpheme_subjects())
    with LexiconDB(fresh_db) as db:
        topo_id = _insert_toponym(db, "Stratford")
        compute_decompositions(db, topo_id, "Stratford", word_db)
        db.commit()
        count = db.conn.execute(
            "SELECT COUNT(*) AS c FROM toponym_decomposition WHERE toponym_id = ?",
            (topo_id,),
        ).fetchone()["c"]
    assert count >= 1


def test_compute_decompositions_idempotent_on_rerun(fresh_db: Path) -> None:
    """Re-running with the same word_db produces no additional rows
    (UNIQUE on signature)."""
    word_db = _word_db_for(_two_morpheme_subjects())
    with LexiconDB(fresh_db) as db:
        topo_id = _insert_toponym(db, "Stratford")
        compute_decompositions(db, topo_id, "Stratford", word_db)
        db.commit()
        first = db.conn.execute(
            "SELECT COUNT(*) AS c FROM toponym_decomposition WHERE toponym_id = ?",
            (topo_id,),
        ).fetchone()["c"]
        compute_decompositions(db, topo_id, "Stratford", word_db)
        db.commit()
        second = db.conn.execute(
            "SELECT COUNT(*) AS c FROM toponym_decomposition WHERE toponym_id = ?",
            (topo_id,),
        ).fetchone()["c"]
    assert first == second
    assert first >= 1


def test_compute_decompositions_records_unaccounted_fragments(fresh_db: Path) -> None:
    """When the matcher can't account for everything, the leftover
    chars land in ``unaccounted_fragments`` and ``unaccounted_count``
    reflects them."""
    word_db = _word_db_for(_two_morpheme_subjects())
    with LexiconDB(fresh_db) as db:
        topo_id = _insert_toponym(db, "Stratfordxyz")  # 'xyz' won't match
        compute_decompositions(db, topo_id, "Stratfordxyz", word_db)
        db.commit()
        rows = db.conn.execute(
            "SELECT unaccounted_fragments, unaccounted_count "
            "FROM toponym_decomposition WHERE toponym_id = ?",
            (topo_id,),
        ).fetchall()
    # At least one decomposition has nonzero unaccounted_count.
    assert any(row["unaccounted_count"] > 0 for row in rows)
    # And the fragments JSON is parseable.
    for row in rows:
        parsed = json.loads(row["unaccounted_fragments"])
        assert isinstance(parsed, list)
        assert len(parsed) == row["unaccounted_count"]


def test_compute_decompositions_multi_word_toponym_cross_product(fresh_db: Path) -> None:
    """A multi-word toponym ('Saint Mary Bourne') produces decompositions
    that are the cross-product of per-word options. With one option per
    word this is one row; tested here is just that the multi-word path
    runs without crashing and produces valid rows."""
    subjects = [
        _make_subject(
            meaning=["Saint"],
            words=[{"modern_usage": "Saint", "old_english": ["sanct"]}],
        ),
        _make_subject(
            meaning=["Mary"],
            words=[{"modern_usage": "Mary", "old_english": ["maria"]}],
        ),
        _make_subject(
            meaning=["Bourne"],
            words=[{"modern_usage": "Bourne", "old_english": ["burna"]}],
        ),
    ]
    word_db = _word_db_for(subjects)
    with LexiconDB(fresh_db) as db:
        topo_id = _insert_toponym(db, "Saint Mary Bourne")
        compute_decompositions(db, topo_id, "Saint Mary Bourne", word_db)
        db.commit()
        rows = db.conn.execute(
            "SELECT morpheme_count FROM toponym_decomposition WHERE toponym_id = ?",
            (topo_id,),
        ).fetchall()
    assert len(rows) >= 1
    # Each row should have 3 morpheme slots — one per word.
    assert any(row["morpheme_count"] == 3 for row in rows)


# --- pick_canonical_decomposition ----------------------------------------


def test_pick_canonical_rule_a_scholar_match(fresh_db: Path) -> None:
    """Rule (a): when a scholar's etymon list lines up with one stored
    decomposition's slot list, that row is canonical with
    source='scholar'."""
    word_db = _word_db_for(_two_morpheme_subjects())
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        topo_id = _insert_toponym(db, "Stratford")
        compute_decompositions(db, topo_id, "Stratford", word_db)
        # Scholar says: stræt + ford (both old-english).
        _insert_scholar_etymology(
            db,
            toponym_id=topo_id,
            elements=[("stræt", "old-english"), ("ford", "old-english")],
        )
        result = pick_canonical_decomposition(db, topo_id, word_db)
        db.commit()
        rows = db.conn.execute(
            "SELECT is_canonical, canonical_source FROM toponym_decomposition "
            "WHERE toponym_id = ? AND is_canonical = 1",
            (topo_id,),
        ).fetchall()
    assert result["rule"] == "scholar"
    assert result["canonical_count"] == 1
    assert len(rows) == 1
    assert rows[0]["canonical_source"] == "scholar"


def test_pick_canonical_rule_a_scholar_disagreement(fresh_db: Path) -> None:
    """Rule (a) disagreement: when two scholars propose different
    zero-unaccounted breakdowns AND each matches a stored
    decomposition, all matching rows are tagged 'scholar-disagreement'.

    Setup: 'Bridgewater' admits BOTH a one-morpheme parse
    ('Bridgewater' as a single etymon) AND a two-morpheme parse
    ('Bridge-' + '-water'). Both are zero-unaccounted; both can be
    valid scholarly breakdowns. Real scholarship genuinely disagrees on
    cases like this — schema preserves the disagreement.
    """
    subjects = [
        _make_subject(
            meaning=["Bridge"],
            words=[{"modern_usage": "Bridge-", "old_english": ["brycg"]}],
        ),
        _make_subject(
            meaning=["Water"],
            words=[{"modern_usage": "-water", "old_english": ["wæter"]}],
        ),
        _make_subject(
            meaning=["Bridgewater"],
            words=[{"modern_usage": "Bridgewater", "old_english": ["brycgwæter"]}],
        ),
    ]
    word_db = _word_db_for(subjects)
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        topo_id = _insert_toponym(db, "Bridgewater")
        compute_decompositions(db, topo_id, "Bridgewater", word_db)
        # Scholar A: brycg + wæter (two morphemes).
        _insert_scholar_etymology(
            db,
            toponym_id=topo_id,
            elements=[("brycg", "old-english"), ("wæter", "old-english")],
        )
        # Scholar B: brycgwæter (one morpheme).
        _insert_scholar_etymology(
            db,
            toponym_id=topo_id,
            elements=[("brycgwæter", "old-english")],
        )
        result = pick_canonical_decomposition(db, topo_id, word_db)
        db.commit()
        rows = db.conn.execute(
            "SELECT canonical_source, morpheme_count FROM toponym_decomposition "
            "WHERE toponym_id = ? AND is_canonical = 1",
            (topo_id,),
        ).fetchall()
    assert result["rule"] == "scholar-disagreement"
    # Both decompositions are canonical (each scholar's breakdown
    # matched a different stored row).
    assert len(rows) == 2
    assert all(row["canonical_source"] == "scholar-disagreement" for row in rows)
    morpheme_counts = sorted(row["morpheme_count"] for row in rows)
    assert morpheme_counts == [1, 2]


def test_pick_canonical_rule_b_unique_zero_unaccounted(fresh_db: Path) -> None:
    """Rule (b): when no scholar match fires AND exactly one stored
    decomposition has zero unaccounted, that one is canonical with
    source='unique-zero-unaccounted'."""
    word_db = _word_db_for(_two_morpheme_subjects())
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        topo_id = _insert_toponym(db, "Stratford")
        compute_decompositions(db, topo_id, "Stratford", word_db)
        # No scholar etymology at all — rule (a) skipped.
        result = pick_canonical_decomposition(db, topo_id, word_db)
        db.commit()
        canonical = db.conn.execute(
            "SELECT canonical_source, unaccounted_count FROM toponym_decomposition "
            "WHERE toponym_id = ? AND is_canonical = 1",
            (topo_id,),
        ).fetchone()
    assert result["rule"] == "unique-zero-unaccounted"
    assert result["canonical_count"] == 1
    assert canonical["canonical_source"] == "unique-zero-unaccounted"
    assert canonical["unaccounted_count"] == 0


def test_pick_canonical_no_zero_falls_through(fresh_db: Path) -> None:
    """When every stored decomposition has unaccounted fragments AND no
    scholar match fires, no row is canonical (Phase 2 will tiebreak)."""
    # word_db with NO morpheme matching 'Stratford' — the matcher
    # produces only a single decomposition with the entire surface as
    # one unaccounted fragment.
    word_db = _word_db_for(
        [
            _make_subject(
                meaning=["Other"],
                words=[{"modern_usage": "Otherword", "old_english": ["other"]}],
            )
        ]
    )
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        topo_id = _insert_toponym(db, "Stratford")
        compute_decompositions(db, topo_id, "Stratford", word_db)
        result = pick_canonical_decomposition(db, topo_id, word_db)
        db.commit()
        canonical = db.conn.execute(
            "SELECT 1 FROM toponym_decomposition WHERE toponym_id = ? AND is_canonical = 1",
            (topo_id,),
        ).fetchone()
    assert result["rule"] is None
    assert result["canonical_count"] == 0
    assert canonical is None


def test_pick_canonical_no_decompositions_returns_zero(fresh_db: Path) -> None:
    """A toponym that has never been decomposed surfaces a zero-count
    summary, doesn't crash, and writes nothing."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        topo_id = _insert_toponym(db, "Stratford")
        result = pick_canonical_decomposition(db, topo_id, {})
    assert result == {"rule": None, "canonical_count": 0, "decomposition_count": 0}


def test_pick_canonical_idempotent_rerun(fresh_db: Path) -> None:
    """Running the picker twice on the same state yields the same
    canonical assignment (no double-canonical, no flicker)."""
    word_db = _word_db_for(_two_morpheme_subjects())
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        topo_id = _insert_toponym(db, "Stratford")
        compute_decompositions(db, topo_id, "Stratford", word_db)
        first = pick_canonical_decomposition(db, topo_id, word_db)
        second = pick_canonical_decomposition(db, topo_id, word_db)
        db.commit()
        canonical_count = db.conn.execute(
            "SELECT COUNT(*) AS c FROM toponym_decomposition "
            "WHERE toponym_id = ? AND is_canonical = 1",
            (topo_id,),
        ).fetchone()["c"]
    assert first == second
    assert canonical_count == 1


def test_pick_canonical_clears_prior_state(fresh_db: Path) -> None:
    """Running the picker resets prior canonical state — if a row was
    canonical via rule (b) and a new scholar etymology lands, the
    re-run should reassign canonical to the scholar match alone."""
    word_db = _word_db_for(_two_morpheme_subjects())
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        topo_id = _insert_toponym(db, "Stratford")
        compute_decompositions(db, topo_id, "Stratford", word_db)
        # First pass: rule (b) fires.
        first = pick_canonical_decomposition(db, topo_id, word_db)
        assert first["rule"] == "unique-zero-unaccounted"
        # Now add a scholar claim and re-run.
        _insert_scholar_etymology(
            db,
            toponym_id=topo_id,
            elements=[("stræt", "old-english"), ("ford", "old-english")],
        )
        second = pick_canonical_decomposition(db, topo_id, word_db)
        db.commit()
        canonical_rows = db.conn.execute(
            "SELECT canonical_source FROM toponym_decomposition "
            "WHERE toponym_id = ? AND is_canonical = 1",
            (topo_id,),
        ).fetchall()
    assert second["rule"] == "scholar"
    assert len(canonical_rows) == 1
    assert canonical_rows[0]["canonical_source"] == "scholar"


def test_scholar_seq_match_rejects_unaccounted_decomposition() -> None:
    """A scholar match requires zero unaccounted fragments — a
    decomposition with even one unaccounted slot can't be scholarly
    (scholars publish complete breakdowns)."""
    scholar_seq = [("old_english", "stræt"), ("old_english", "ford")]
    # Slots: one morpheme + one unaccounted string. Should fail.
    slots = [{("old_english", "stræt")}, "ford"]
    assert _scholar_seq_matches_decomposition(scholar_seq, slots) is False


def test_scholar_seq_match_rejects_length_mismatch() -> None:
    """The morpheme slot count must equal the scholar element count.
    A 3-element scholar breakdown can't match a 2-slot decomposition."""
    scholar_seq = [
        ("old_english", "stræt"),
        ("old_english", "for"),
        ("old_english", "d"),
    ]
    slots = [{("old_english", "stræt")}, {("old_english", "ford")}]
    assert _scholar_seq_matches_decomposition(scholar_seq, slots) is False


def test_scholar_seq_match_requires_in_order() -> None:
    """Slot ordering matters — scholar reads left-to-right, matcher
    preserves slot order. A reversed scholar list shouldn't match."""
    scholar_seq = [("old_english", "ford"), ("old_english", "stræt")]
    slots = [{("old_english", "stræt")}, {("old_english", "ford")}]
    assert _scholar_seq_matches_decomposition(scholar_seq, slots) is False


# --- populate_and_pick (orchestration) -----------------------------------


def test_populate_and_pick_runs_both_phases(fresh_db: Path) -> None:
    """The convenience entry runs compute then pick. Verifies the rule
    summary surfaces both the decomposition count AND the picker's
    rule firing."""
    word_db = _word_db_for(_two_morpheme_subjects())
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        topo_id = _insert_toponym(db, "Stratford")
        summary = populate_and_pick(db, topo_id, "Stratford", word_db)
    assert summary["decomposition_count"] >= 1
    assert summary["rule"] in {"scholar", "unique-zero-unaccounted"}


# --- CLI ------------------------------------------------------------------


def test_cli_decompose_dry_run_does_not_write(fresh_db: Path) -> None:
    """``decompose`` (no --apply) prints the rule-firing counts but
    leaves the DB unchanged."""
    word_db_subjects = _two_morpheme_subjects()
    bundle_path = Path(fresh_db).parent / "meanings.json"
    bundle_path.write_text(json.dumps(word_db_subjects))
    with LexiconDB(fresh_db) as db:
        _insert_toponym(db, "Stratford")
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "decompose",
            "--db",
            str(fresh_db),
            "--meanings",
            str(bundle_path),
        ],
    )
    assert result.exit_code == 0, result.output
    with LexiconDB(fresh_db) as db:
        count = db.conn.execute("SELECT COUNT(*) AS c FROM toponym_decomposition").fetchone()["c"]
    assert count == 0


def test_cli_decompose_apply_writes_rows(fresh_db: Path) -> None:
    """``decompose --apply`` persists rows AND a canonical pick under
    rule (b) when no scholar matches exist."""
    bundle_path = Path(fresh_db).parent / "meanings.json"
    bundle_path.write_text(json.dumps(_two_morpheme_subjects()))
    with LexiconDB(fresh_db) as db:
        _insert_toponym(db, "Stratford")
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "decompose",
            "--db",
            str(fresh_db),
            "--meanings",
            str(bundle_path),
            "--apply",
        ],
    )
    assert result.exit_code == 0, result.output
    with LexiconDB(fresh_db) as db:
        canonical_count = db.conn.execute(
            "SELECT COUNT(*) AS c FROM toponym_decomposition WHERE is_canonical = 1"
        ).fetchone()["c"]
    assert canonical_count == 1


def test_cli_decompose_targets_single_toponym(fresh_db: Path) -> None:
    """``--toponym-id`` scopes processing to one row; other toponyms
    are untouched."""
    bundle_path = Path(fresh_db).parent / "meanings.json"
    bundle_path.write_text(json.dumps(_two_morpheme_subjects()))
    with LexiconDB(fresh_db) as db:
        target_id = _insert_toponym(db, "Stratford")
        _insert_toponym(db, "Bridgeton")  # second toponym — should be untouched
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "decompose",
            "--db",
            str(fresh_db),
            "--meanings",
            str(bundle_path),
            "--toponym-id",
            str(target_id),
            "--apply",
        ],
    )
    assert result.exit_code == 0, result.output
    with LexiconDB(fresh_db) as db:
        target_count = db.conn.execute(
            "SELECT COUNT(*) AS c FROM toponym_decomposition WHERE toponym_id = ?",
            (target_id,),
        ).fetchone()["c"]
        other_count = db.conn.execute(
            "SELECT COUNT(*) AS c FROM toponym_decomposition WHERE toponym_id != ?",
            (target_id,),
        ).fetchone()["c"]
    assert target_count >= 1
    assert other_count == 0


def test_cli_decompose_limit_caps_processing(fresh_db: Path) -> None:
    """``--limit N`` processes at most N toponyms in id order."""
    bundle_path = Path(fresh_db).parent / "meanings.json"
    bundle_path.write_text(json.dumps(_two_morpheme_subjects()))
    with LexiconDB(fresh_db) as db:
        first_id = _insert_toponym(db, "Stratford")
        _insert_toponym(db, "Bridgeton")
        _insert_toponym(db, "Marketown")
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "decompose",
            "--db",
            str(fresh_db),
            "--meanings",
            str(bundle_path),
            "--limit",
            "1",
            "--apply",
        ],
    )
    assert result.exit_code == 0, result.output
    with LexiconDB(fresh_db) as db:
        touched_topos = {
            row["toponym_id"]
            for row in db.conn.execute("SELECT DISTINCT toponym_id FROM toponym_decomposition")
        }
    # Only the lowest-id toponym should have been processed.
    assert touched_topos == {first_id}


def test_cli_decompose_handles_empty_corpus(fresh_db: Path) -> None:
    """An empty toponym table reports clearly and exits cleanly."""
    bundle_path = Path(fresh_db).parent / "meanings.json"
    bundle_path.write_text(json.dumps(_two_morpheme_subjects()))
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "decompose",
            "--db",
            str(fresh_db),
            "--meanings",
            str(bundle_path),
            "--apply",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "No toponym rows" in result.output or "No toponym rows" in result.stderr_bytes.decode()


# --- regression + missing-coverage cases ---------------------------------


def test_pick_canonical_corroborating_scholars_not_disagreement(fresh_db: Path) -> None:
    """Regression: two scholars publishing IDENTICAL etymologies are not
    in disagreement — they corroborate. Picker tags 'scholar', not
    'scholar-disagreement'.

    Earlier draft tracked matched scholar INDICES (s_idx) instead of
    DISTINCT scholar content; two duplicate rows would inflate the
    distinct-scholar count and mis-tag the strongest-canonical case.
    """
    word_db = _word_db_for(_two_morpheme_subjects())
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        topo_id = _insert_toponym(db, "Stratford")
        compute_decompositions(db, topo_id, "Stratford", word_db)
        # Two scholars, same breakdown.
        _insert_scholar_etymology(
            db,
            toponym_id=topo_id,
            elements=[("stræt", "old-english"), ("ford", "old-english")],
        )
        _insert_scholar_etymology(
            db,
            toponym_id=topo_id,
            elements=[("stræt", "old-english"), ("ford", "old-english")],
        )
        result = pick_canonical_decomposition(db, topo_id, word_db)
        db.commit()
        rows = db.conn.execute(
            "SELECT canonical_source FROM toponym_decomposition "
            "WHERE toponym_id = ? AND is_canonical = 1",
            (topo_id,),
        ).fetchall()
    assert result["rule"] == "scholar"
    assert len(rows) == 1
    assert rows[0]["canonical_source"] == "scholar"


def test_dedupe_scholar_seqs_preserves_first_occurrence_order() -> None:
    """``_dedupe_scholar_seqs`` keeps the first-seen sequence and drops
    later content-duplicates. Order matters — downstream consumers may
    care about scholar-source priority encoded in row order."""
    seqs = [
        [("old_english", "stræt"), ("old_english", "ford")],
        [("old_english", "brycgwæter")],
        [("old_english", "stræt"), ("old_english", "ford")],  # duplicate
    ]
    distinct = _dedupe_scholar_seqs(seqs)
    assert distinct == [
        [("old_english", "stræt"), ("old_english", "ford")],
        [("old_english", "brycgwæter")],
    ]


def test_scholar_form_sequences_drops_unmapped_language(fresh_db: Path) -> None:
    """An etymon whose ``language`` doesn't appear in
    ``_LANG_CODE_TO_JSON_FIELD`` can't possibly match a Meaning's
    sources; ``_scholar_form_sequences`` filters it out so the empty
    placeholder doesn't shorten the sequence and silently fail
    matching."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        topo_id = _insert_toponym(db, "Stratford")
        cur = db.conn.execute(
            "INSERT INTO toponym_etymology (toponym_id, source_id, confidence) "
            "VALUES (?, 'skeat-1901', 'high')",
            (topo_id,),
        )
        etymology_id = cur.lastrowid
        unmapped_etymon = db.upsert_etymon("ar-tawa", "no-such-language")
        db.conn.execute(
            "INSERT INTO toponym_etymology_element "
            "(toponym_etymology_id, ordinal, etymon_id, inflection, surface_in_modern) "
            "VALUES (?, 0, ?, NULL, NULL)",
            (etymology_id, unmapped_etymon),
        )
        db.commit()
        seqs = _scholar_form_sequences(db, topo_id)
    # Etymology row had only one element with an unmapped language;
    # after filtering it's empty, and empty sequences are dropped.
    assert seqs == []


def test_scholar_form_sequences_drops_etymon_with_no_canonical_form(fresh_db: Path) -> None:
    """An etymon row with NULL canonical_form (corrupt / partial mining
    output) is dropped from the sequence — it can't possibly match a
    Meaning's sources entry."""
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        topo_id = _insert_toponym(db, "Stratford")
        cur = db.conn.execute(
            "INSERT INTO toponym_etymology (toponym_id, source_id, confidence) "
            "VALUES (?, 'skeat-1901', 'high')",
            (topo_id,),
        )
        etymology_id = cur.lastrowid
        # Direct INSERT to bypass upsert_etymon's NOT NULL guard.
        cur = db.conn.execute(
            "INSERT INTO etymon (canonical_form, language) VALUES ('', 'old-english')"
        )
        empty_etymon_id = cur.lastrowid
        db.conn.execute(
            "INSERT INTO toponym_etymology_element "
            "(toponym_etymology_id, ordinal, etymon_id, inflection, surface_in_modern) "
            "VALUES (?, 0, ?, NULL, NULL)",
            (etymology_id, empty_etymon_id),
        )
        db.commit()
        seqs = _scholar_form_sequences(db, topo_id)
    assert seqs == []


def test_candidates_for_usage_unions_co_located_meanings() -> None:
    """When a single ``modern_usage`` carries multiple Meanings (different
    senses sharing one surface), ``_candidates_for_usage`` returns the
    UNION of their (lang_field, form) pairs. Without the union, a
    scholar match against the second sense would fail because the slot
    only carried the first sense's forms."""
    # Build a word_db where '-y' has TWO Meanings: 'island' and 'district'.
    subjects = [
        _make_subject(
            meaning=["Island"],
            words=[{"modern_usage": "-y", "old_english": ["īg"]}],
        ),
        _make_subject(
            meaning=["District"],
            words=[{"modern_usage": "-y", "old_english": ["gē"]}],
        ),
    ]
    word_db = _word_db_for(subjects)
    candidates = _candidates_for_usage(word_db, "-y")
    assert ("old_english", "īg") in candidates
    assert ("old_english", "gē") in candidates


def test_cross_product_decompositions_with_multi_option_per_word() -> None:
    """Multi-word toponyms with multiple matcher options per word should
    produce the full N×M cross-product. Pin this so the ``-y``-as-
    'island'-or-'district' multi-sense case (and similar) round-trips
    correctly through the cross-product path.
    """
    # 'salem' with TWO options: full-word match, and 'sal-' + '-em'.
    subjects = [
        _make_subject(
            meaning=["Sale"],
            words=[{"modern_usage": "Sal-", "old_english": ["sal"]}],
        ),
        _make_subject(
            meaning=["End"],
            words=[{"modern_usage": "-em", "old_english": ["em"]}],
        ),
        _make_subject(
            meaning=["Salem"],
            words=[{"modern_usage": "Salem", "old_english": ["salem"]}],
        ),
    ]
    word_db = _word_db_for(subjects)
    name = Name("Salem Salem")  # multi-word, each word multi-option
    name.find_meaning(word_db, reduce=False)
    combos = _cross_product_decompositions(name)
    # Both words admit at least 2 zero-unaccounted parses, so the
    # cross-product yields at least 2*2 = 4 combinations. The exact
    # count depends on partial-match alternatives the trie emits; the
    # invariant we pin is "more than just the per-word sum, i.e.
    # cross-product was actually computed."
    n_per_word = len(name.words["Salem"])
    assert n_per_word >= 2
    assert len(combos) == n_per_word * n_per_word
    # And every combo is double the morpheme/unaccounted-slot count of
    # a single-word combo (each word contributes its slots).
    single_word_lens = {len(w.word) for w in name.words["Salem"]}
    expected_total_lens = {a + b for a in single_word_lens for b in single_word_lens}
    assert {len(combo) for combo in combos} == expected_total_lens
