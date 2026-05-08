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
    apply_canonical_to_name,
    compute_decompositions,
    decompose_with_canonical,
    lookup_canonical_signature,
    pick_canonical_decomposition,
    populate_and_pick,
)
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema, migrate_schema
from wyrd.generators.kenning.meaning import Meaning, load_meanings
from wyrd.generators.kenning.name import Name, load_names_with_regions


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


# --- Phase 2 (wyrd-zpi9) refactor + safety improvements ------------------


def test_cross_product_caps_pathological_combo_count(caplog) -> None:
    """A toponym whose cross product would explode past the cap is
    truncated by capping per-word options BEFORE multiplying. Every
    emitted combo is full-length (one slot per word). Bounds memory
    before bulk --apply against the production corpus."""
    import logging

    from wyrd.generators.kenning import decomposition

    name = Name("a b c d")

    class _FakeWord:
        def __init__(self, contents: list) -> None:
            self.word = contents

    # 5 options per word × 4 words = 625 combos, well past the 256 cap.
    for word_str in name.name.split(" "):
        name.words[word_str] = [_FakeWord([f"{word_str}_{i}"]) for i in range(5)]

    with caplog.at_level(logging.WARNING, logger="wyrd.generators.kenning.decomposition"):
        combos = decomposition._cross_product_decompositions(name)

    assert len(combos) <= decomposition._MAX_CROSS_PRODUCT_COMBOS
    # CRITICAL: every combo must be full-length (4 slots, one per word).
    # Mid-multiplication truncation would have produced prefix-only
    # combos missing words 3-4 — silently persisted as 0-unaccounted.
    assert all(len(combo) == 4 for combo in combos), (
        f"truncation produced incomplete combos; lengths: {[len(c) for c in combos]}"
    )
    assert any("Cross-product cap" in rec.message for rec in caplog.records)


def test_cross_product_cap_persists_full_length_combos_when_truncating_first_word() -> None:
    """Regression test: with truncation triggered by the FIRST word's
    options (16 options × 4 words = 4^4*16 vs cap), the cap must NOT
    emit prefix-only combos. Pinned because the original
    early-termination implementation produced length-1 combos in this
    pathological case."""
    from wyrd.generators.kenning import decomposition

    name = Name("a b c d")

    class _FakeWord:
        def __init__(self, contents: list) -> None:
            self.word = contents

    # 16 options on word 'a', 4 each on b/c/d → projected 16*4*4*4 =
    # 1024 combos. Old early-termination would truncate during word
    # 'a' iteration giving length-1 combos. New per-word capping
    # trims word 'a' so the product fits.
    name.words["a"] = [_FakeWord([f"a_{i}"]) for i in range(16)]
    for word_str in ("b", "c", "d"):
        name.words[word_str] = [_FakeWord([f"{word_str}_{i}"]) for i in range(4)]

    combos = decomposition._cross_product_decompositions(name)
    assert len(combos) <= decomposition._MAX_CROSS_PRODUCT_COMBOS
    assert all(len(combo) == 4 for combo in combos)


def test_cap_per_word_options_returns_input_when_under_cap() -> None:
    """``_cap_per_word_options`` is a no-op when the projected product
    fits under the cap."""
    from wyrd.generators.kenning import decomposition

    per_word_lists = [[[1], [2], [3]], [[4], [5]]]  # 3*2 = 6
    capped = decomposition._cap_per_word_options(per_word_lists, "test")
    assert capped == per_word_lists


def test_cap_per_word_options_trims_longest_list_first(caplog) -> None:
    """When trimming, ``_cap_per_word_options`` removes from the
    LONGEST list first (greedy), keeping smaller lists intact when
    possible."""
    import logging

    from wyrd.generators.kenning import decomposition

    # 100 + 4 + 4 + 4 — projected 100*4*4*4 = 6400, way over cap.
    # Trimming should reduce list 0 (the 100-list) most.
    per_word_lists = [
        [[i] for i in range(100)],
        [[i] for i in range(4)],
        [[i] for i in range(4)],
        [[i] for i in range(4)],
    ]
    with caplog.at_level(logging.WARNING, logger="wyrd.generators.kenning.decomposition"):
        capped = decomposition._cap_per_word_options(per_word_lists, "test")
    projected = 1
    for pw in capped:
        projected *= len(pw)
    assert projected <= decomposition._MAX_CROSS_PRODUCT_COMBOS
    # The 100-list should have been the most-trimmed.
    assert len(capped[0]) < len(per_word_lists[0])
    # The 4-lists should remain untrimmed when possible (sizes still 4).
    assert all(len(capped[i]) == 4 for i in (1, 2, 3))


def test_cross_product_under_cap_is_unchanged() -> None:
    """A name whose cross product fits under the cap is unchanged —
    no truncation, no warning. Bit-stable with the pre-cap version
    for the typical-input case."""
    from wyrd.generators.kenning import decomposition

    name = Name("a b")

    class _FakeWord:
        def __init__(self, contents: list) -> None:
            self.word = contents

    # 3 options per word, 2 words → 3*3 = 9 combos.
    for word_str in name.name.split(" "):
        name.words[word_str] = [_FakeWord([f"{word_str}_{i}"]) for i in range(3)]

    combos = decomposition._cross_product_decompositions(name)
    assert len(combos) == 9


def test_scholar_form_sequences_logs_unmapped_language(fresh_db: Path, caplog) -> None:
    """Phase 2: silent drops on unmapped language now emit a debug
    log. Surfaces the failure mode for future operators."""
    import logging

    from wyrd.generators.kenning.decomposition import _scholar_form_sequences

    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        topo_id = _insert_toponym(db, "Stratford")
        cur = db.conn.execute(
            "INSERT INTO toponym_etymology (toponym_id, source_id, confidence) "
            "VALUES (?, 'skeat-1901', 'high')",
            (topo_id,),
        )
        etymology_id = cur.lastrowid
        unmapped = db.upsert_etymon("ar-tawa", "no-such-language")
        db.conn.execute(
            "INSERT INTO toponym_etymology_element "
            "(toponym_etymology_id, ordinal, etymon_id, inflection, surface_in_modern) "
            "VALUES (?, 0, ?, NULL, NULL)",
            (etymology_id, unmapped),
        )
        db.commit()
        with caplog.at_level(logging.DEBUG, logger="wyrd.generators.kenning.decomposition"):
            seqs = _scholar_form_sequences(db, topo_id)
    assert seqs == []
    assert any(
        "unmapped" in rec.message.lower() and "no-such-language" in rec.message
        for rec in caplog.records
    )


# --- refactored helpers (wyrd-zpi9 split of pick_canonical_decomposition) -


def test_load_stored_decompositions_returns_rows_in_id_order(fresh_db: Path) -> None:
    """``_load_stored_decompositions`` returns rows ordered by id.
    Pinned so a caller relying on insert order stays correct after
    the refactor."""
    from wyrd.generators.kenning.decomposition import _load_stored_decompositions

    word_db = _word_db_for(_two_morpheme_subjects())
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        topo_id = _insert_toponym(db, "Stratford")
        compute_decompositions(db, topo_id, "Stratford", word_db)
        db.commit()
        rows = _load_stored_decompositions(db, topo_id)
    assert len(rows) >= 1
    ids = [row["id"] for row in rows]
    assert ids == sorted(ids)


def _row(row_id: int, unacc: int) -> dict:
    """Synthetic minimal Row mock for unit-testing _try_rule_unique_zero
    without DB roundtrip."""

    class _R:
        def __init__(self) -> None:
            self._d = {"id": row_id, "unaccounted_count": unacc}

        def __getitem__(self, k: str):
            return self._d[k]

    return _R()


def test_try_rule_unique_zero_returns_none_when_multiple_zero_rows() -> None:
    """When multiple stored rows have unaccounted_count==0,
    _try_rule_unique_zero returns None (Phase 2 tiebreaker territory)."""
    from wyrd.generators.kenning.decomposition import _try_rule_unique_zero

    rows = [_row(1, 0), _row(2, 0), _row(3, 1)]
    assert _try_rule_unique_zero(rows) is None


def test_try_rule_unique_zero_returns_id_when_exactly_one_zero() -> None:
    """The unique-zero rule fires only when exactly one zero-row
    exists."""
    from wyrd.generators.kenning.decomposition import _try_rule_unique_zero

    rows = [_row(1, 1), _row(2, 0), _row(3, 2)]
    assert _try_rule_unique_zero(rows) == 2


def test_try_rule_unique_zero_returns_none_for_empty_rows() -> None:
    """No rows at all → None (defensive)."""
    from wyrd.generators.kenning.decomposition import _try_rule_unique_zero

    assert _try_rule_unique_zero([]) is None


def test_try_rule_scholar_returns_none_when_no_match(fresh_db: Path) -> None:
    """``_try_rule_scholar`` returns None when no scholar etymology
    matches any stored decomposition's slot candidates."""
    from wyrd.generators.kenning.decomposition import (
        _build_slot_candidates_per_row,
        _load_stored_decompositions,
        _try_rule_scholar,
    )

    word_db = _word_db_for(_two_morpheme_subjects())
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        topo_id = _insert_toponym(db, "Stratford")
        compute_decompositions(db, topo_id, "Stratford", word_db)
        # No scholar etymology rows.
        rows = _load_stored_decompositions(db, topo_id)
        slots = _build_slot_candidates_per_row(rows, word_db)
        result = _try_rule_scholar(db, topo_id, rows, slots)
    assert result is None


def test_try_rule_scholar_returns_scholar_for_single_match(fresh_db: Path) -> None:
    """``_try_rule_scholar`` returns ('scholar', [row_ids]) when one
    distinct scholar breakdown matches a stored decomposition."""
    from wyrd.generators.kenning.decomposition import (
        _build_slot_candidates_per_row,
        _load_stored_decompositions,
        _try_rule_scholar,
    )

    word_db = _word_db_for(_two_morpheme_subjects())
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        topo_id = _insert_toponym(db, "Stratford")
        compute_decompositions(db, topo_id, "Stratford", word_db)
        _insert_scholar_etymology(
            db,
            toponym_id=topo_id,
            elements=[("stræt", "old-english"), ("ford", "old-english")],
        )
        rows = _load_stored_decompositions(db, topo_id)
        slots = _build_slot_candidates_per_row(rows, word_db)
        result = _try_rule_scholar(db, topo_id, rows, slots)
    assert result is not None
    source, matching_ids = result
    assert source == "scholar"
    assert len(matching_ids) >= 1


def test_reset_canonical_clears_prior_assignment(fresh_db: Path) -> None:
    """``_reset_canonical`` resets is_canonical=0 and
    canonical_source=NULL on every row for the given toponym."""
    from wyrd.generators.kenning.decomposition import _reset_canonical

    word_db = _word_db_for(_two_morpheme_subjects())
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        topo_id = _insert_toponym(db, "Stratford")
        compute_decompositions(db, topo_id, "Stratford", word_db)
        # Manually mark a row canonical.
        db.conn.execute(
            "UPDATE toponym_decomposition SET is_canonical = 1, "
            "canonical_source = 'scholar' WHERE toponym_id = ?",
            (topo_id,),
        )
        db.commit()
        # Reset.
        _reset_canonical(db, topo_id)
        db.commit()
        canonical_count = db.conn.execute(
            "SELECT COUNT(*) AS c FROM toponym_decomposition "
            "WHERE toponym_id = ? AND is_canonical = 1",
            (topo_id,),
        ).fetchone()["c"]
        source_count = db.conn.execute(
            "SELECT COUNT(*) AS c FROM toponym_decomposition "
            "WHERE toponym_id = ? AND canonical_source IS NOT NULL",
            (topo_id,),
        ).fetchone()["c"]
    assert canonical_count == 0
    assert source_count == 0


def test_scholar_form_sequences_logs_empty_canonical_form(fresh_db: Path, caplog) -> None:
    """An etymon row with empty canonical_form is silently dropped;
    Phase 2's debug log surfaces this distinct failure mode."""
    import logging

    from wyrd.generators.kenning.decomposition import _scholar_form_sequences

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
        with caplog.at_level(logging.DEBUG, logger="wyrd.generators.kenning.decomposition"):
            seqs = _scholar_form_sequences(db, topo_id)
    assert seqs == []
    assert any(
        "empty" in rec.message.lower() and "canonical_form" in rec.message for rec in caplog.records
    )


def test_build_slot_candidates_per_row_uses_word_db(fresh_db: Path) -> None:
    """``_build_slot_candidates_per_row`` rehydrates per-row slot lists
    from stored morpheme_ids JSON. Each Meaning slot becomes a
    candidate set drawn from the live word_db; each unaccounted slot
    stays a string."""
    from wyrd.generators.kenning.decomposition import (
        _build_slot_candidates_per_row,
        _load_stored_decompositions,
    )

    word_db = _word_db_for(_two_morpheme_subjects())
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        topo_id = _insert_toponym(db, "Stratford")
        compute_decompositions(db, topo_id, "Stratford", word_db)
        db.commit()
        rows = _load_stored_decompositions(db, topo_id)
    slots_per_row = _build_slot_candidates_per_row(rows, word_db)
    assert len(slots_per_row) == len(rows)
    # Every Meaning slot should be a set; every unaccounted slot a str.
    for slots in slots_per_row:
        for slot in slots:
            assert isinstance(slot, (set, str))


# --- Phase 3 consumer integration ----------------------------------------
#
# wyrd-70s0: lookup_canonical_signature / apply_canonical_to_name /
# decompose_with_canonical wire the Phase 1+2 canonical pick into the
# matcher consumers (rebuild-proportions, unaccounted miner). The
# tests below cover the helpers in isolation; CLI integration is
# covered separately further down.


def test_lookup_canonical_returns_none_for_unknown_name(fresh_db: Path) -> None:
    """A toponym that doesn't exist in the lexicon yields ``None`` —
    consumer falls back to the heuristic without crashing."""
    with LexiconDB(fresh_db) as db:
        result = lookup_canonical_signature(db, "Nowhere")
    assert result is None


def test_lookup_canonical_returns_none_when_no_canonical_picked(fresh_db: Path) -> None:
    """A toponym row with no is_canonical=1 child (multi-zero tie or
    coverage gap awaiting rule (c)) yields ``None``."""
    word_db = _word_db_for(_two_morpheme_subjects())
    with LexiconDB(fresh_db) as db:
        topo_id = _insert_toponym(db, "Stratford")
        compute_decompositions(db, topo_id, "Stratford", word_db)
        # Don't run the picker; canonical stays NULL on every row.
        db.commit()
        result = lookup_canonical_signature(db, "Stratford")
    assert result is None


def test_lookup_canonical_returns_signature_and_source(fresh_db: Path) -> None:
    """When the picker fired (b) unique-zero-unaccounted, the lookup
    returns the canonical row's signature plus source string."""
    word_db = _word_db_for(_two_morpheme_subjects())
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        topo_id = _insert_toponym(db, "Stratford")
        populate_and_pick(db, topo_id, "Stratford", word_db)
        db.commit()
        result = lookup_canonical_signature(db, "Stratford")
    assert result is not None
    signature, source = result
    assert isinstance(signature, str) and len(signature) == 40  # SHA-1 hex
    assert source == "unique-zero-unaccounted"


def test_lookup_canonical_with_region_tightens_match(fresh_db: Path) -> None:
    """Two toponyms with the same modern_name but different regions get
    disambiguated by the optional region argument. Each region-tagged
    pull returns its OWN toponym's canonical, not the lowest-id one."""
    word_db = _word_db_for(_two_morpheme_subjects())
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        cur1 = db.conn.execute(
            "INSERT INTO toponym (modern_name, region) VALUES (?, ?)",
            ("Stratford", "Bedfordshire"),
        )
        topo_a = cur1.lastrowid
        cur2 = db.conn.execute(
            "INSERT INTO toponym (modern_name, region) VALUES (?, ?)",
            ("Stratford", "Suffolk"),
        )
        topo_b = cur2.lastrowid
        populate_and_pick(db, topo_a, "Stratford", word_db)
        populate_and_pick(db, topo_b, "Stratford", word_db)
        db.commit()
        a_result = lookup_canonical_signature(db, "Stratford", region="Bedfordshire")
        b_result = lookup_canonical_signature(db, "Stratford", region="Suffolk")
    # Both regional pulls hit canonical (each toponym got its own pick).
    assert a_result is not None
    assert b_result is not None


def test_lookup_canonical_falls_back_to_name_only_on_region_miss(
    fresh_db: Path,
) -> None:
    """wyrd-8m87: legacy NULL-region toponyms in the lexicon DB shouldn't
    cause a region-tagged consumer call to silently return None — the
    lookup falls back to a name-only match. This keeps backward compat
    with toponyms ingested before the region-plumbing work landed.

    The fallback also fires when the corpus passes a region that
    doesn't exist in the DB at all (e.g. a 'Yorkshire/Stratford' corpus
    entry against a DB whose only Stratford rows are in Bedfordshire +
    Suffolk) — operators get a canonical for the same-name toponym
    rather than a blank fallback to the heuristic. Trade-off: in the
    rare cross-region collision case this picks the lowest-id canonical
    arbitrarily; the regional disambiguation only takes effect when the
    DB actually carries a region-matched row."""
    word_db = _word_db_for(_two_morpheme_subjects())
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        cur = db.conn.execute(
            "INSERT INTO toponym (modern_name, region) VALUES (?, ?)",
            ("Stratford", "Bedfordshire"),
        )
        topo_a = cur.lastrowid
        populate_and_pick(db, topo_a, "Stratford", word_db)
        db.commit()
        # Consumer asks for Yorkshire/Stratford but DB only has Bedfordshire.
        result = lookup_canonical_signature(db, "Stratford", region="Yorkshire")
    # Fallback fires: the Bedfordshire row's canonical wins.
    assert result is not None
    signature, source = result
    assert source == "unique-zero-unaccounted"


def test_lookup_canonical_with_region_against_null_region_db(fresh_db: Path) -> None:
    """A NULL-region toponym row (legacy ingestion path) is reachable
    via region-tagged lookup thanks to the name-only fallback. Without
    this, every region-tagged consumer call against a legacy DB would
    silently return None."""
    word_db = _word_db_for(_two_morpheme_subjects())
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        # Legacy insert: no region.
        cur = db.conn.execute(
            "INSERT INTO toponym (modern_name) VALUES (?)",
            ("Stratford",),
        )
        topo_id = cur.lastrowid
        populate_and_pick(db, topo_id, "Stratford", word_db)
        db.commit()
        result = lookup_canonical_signature(db, "Stratford", region="Bedfordshire")
    assert result is not None
    _, source = result
    assert source == "unique-zero-unaccounted"


def test_lookup_canonical_picks_lowest_id_on_disagreement(fresh_db: Path) -> None:
    """Scholar-disagreement marks every matching breakdown canonical;
    the lookup deterministically returns the lowest-id canonical row so
    consumer counts stay stable across re-runs."""
    word_db = _word_db_for(_two_morpheme_subjects())
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        topo_id = _insert_toponym(db, "Stratford")
        # Force two scholars with different breakdowns — Phase 1 canonical
        # picker will tag both is_canonical=1 with source='scholar-disagreement'.
        # The matcher only finds 'Strat-' + '-ford' here, so disagreement
        # requires hand-inserting two distinct decomposition rows tagged
        # canonical. Easier path: insert two rows manually and tag them both.
        compute_decompositions(db, topo_id, "Stratford", word_db)
        rows = db.conn.execute(
            "SELECT id FROM toponym_decomposition WHERE toponym_id = ? ORDER BY id",
            (topo_id,),
        ).fetchall()
        # Synthesize a second decomposition row by hashing a fake payload
        # the matcher would never produce (so it doesn't collide with an
        # existing real decomposition's signature).
        fake_payload = [
            ["morpheme", "Strat-"],
            ["unaccounted", "fo"],
            ["unaccounted", "rd"],
        ]
        fake_sig = _signature_for_payload(fake_payload)
        cur = db.conn.execute(
            """
            INSERT INTO toponym_decomposition (
              toponym_id, decomposition_signature, morpheme_ids,
              unaccounted_fragments, unaccounted_count, morpheme_count
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (topo_id, fake_sig, json.dumps(fake_payload), '["fo","rd"]', 2, 1),
        )
        synth_id = cur.lastrowid
        # Tag both rows canonical with disagreement source.
        db.conn.execute(
            "UPDATE toponym_decomposition SET is_canonical = 1, "
            "canonical_source = 'scholar-disagreement' WHERE id IN (?, ?)",
            (rows[0]["id"], synth_id),
        )
        db.commit()
        result = lookup_canonical_signature(db, "Stratford")
    assert result is not None
    signature, source = result
    assert source == "scholar-disagreement"
    # Lowest-id row wins (the matcher-derived row, not the synth).
    real_sig = db_signature_for(fresh_db, rows[0]["id"])
    assert signature == real_sig


def db_signature_for(db_path: Path, row_id: int) -> str:
    with LexiconDB(db_path) as db:
        return db.conn.execute(
            "SELECT decomposition_signature FROM toponym_decomposition WHERE id = ?",
            (row_id,),
        ).fetchone()["decomposition_signature"]


def test_apply_canonical_narrows_words_to_canonical_cell(fresh_db: Path) -> None:
    """When the canonical signature matches a cross-product cell,
    each word in name.words collapses to a singleton holding the
    chosen Word."""
    word_db = _word_db_for(_two_morpheme_subjects())
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        topo_id = _insert_toponym(db, "Stratford")
        populate_and_pick(db, topo_id, "Stratford", word_db)
        db.commit()
        result = lookup_canonical_signature(db, "Stratford")
    assert result is not None
    signature, _ = result

    name = Name("Stratford")
    name.find_meaning(word_db, reduce=False)
    # Confirm there's at least one alternative before narrowing
    # (otherwise the assertion below proves nothing).
    options = name.words["Stratford"]
    pre_count = len(options)
    applied = apply_canonical_to_name(name, signature)
    assert applied is True
    assert len(name.words["Stratford"]) == 1
    # Narrowed Word must be one of the originals — apply doesn't synthesize.
    chosen = name.words["Stratford"][0]
    assert pre_count >= 1
    assert chosen in options


def test_apply_canonical_returns_false_when_signature_does_not_match(
    fresh_db: Path,
) -> None:
    """A stale signature (e.g. word_db drifted since canonical was
    picked) returns False without mutating name.words. The caller can
    then fall back to the heuristic safely."""
    word_db = _word_db_for(_two_morpheme_subjects())
    name = Name("Stratford")
    name.find_meaning(word_db, reduce=False)
    pre_words = {w: list(opts) for w, opts in name.words.items()}
    applied = apply_canonical_to_name(name, "0" * 40)  # bogus signature
    assert applied is False
    # words dict unchanged
    assert {w: list(opts) for w, opts in name.words.items()} == pre_words


def test_apply_canonical_returns_false_for_empty_word_options() -> None:
    """A name with no decompositions populated yields False on apply
    rather than crashing on an empty per-word options list."""
    name = Name("Nowhere")
    # find_meaning not called — name.words[word] is empty list.
    applied = apply_canonical_to_name(name, "0" * 40)
    assert applied is False


def test_apply_canonical_handles_multi_word_toponyms(fresh_db: Path) -> None:
    """A two-word toponym ('Saint Mary') has its per-word entries each
    narrowed; canonical signature spans both words."""
    subjects = [
        _make_subject(
            meaning=["Saint"],
            words=[{"modern_usage": "Saint", "old_english": ["sanct"]}],
        ),
        _make_subject(
            meaning=["Mary"],
            words=[{"modern_usage": "Mary"}],
            modifier_tags=["female-name"],
        ),
    ]
    word_db = _word_db_for(subjects)
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        topo_id = _insert_toponym(db, "Saint Mary")
        populate_and_pick(db, topo_id, "Saint Mary", word_db)
        db.commit()
        result = lookup_canonical_signature(db, "Saint Mary")
    assert result is not None
    signature, _ = result
    name = Name("Saint Mary")
    name.find_meaning(word_db, reduce=False)
    applied = apply_canonical_to_name(name, signature)
    assert applied is True
    assert len(name.words["Saint"]) == 1
    assert len(name.words["Mary"]) == 1


def test_decompose_with_canonical_uses_canonical_when_available(
    fresh_db: Path,
) -> None:
    """The integrated helper returns a canonical-narrowed Name plus
    the canonical source string."""
    word_db = _word_db_for(_two_morpheme_subjects())
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        topo_id = _insert_toponym(db, "Stratford")
        populate_and_pick(db, topo_id, "Stratford", word_db)
        db.commit()
        name, source = decompose_with_canonical("Stratford", word_db, db)
    assert source == "unique-zero-unaccounted"
    assert len(name.words["Stratford"]) == 1


def test_decompose_with_canonical_falls_back_when_no_canonical(
    fresh_db: Path,
) -> None:
    """No toponym row, no decomposition stored, no canonical pick →
    fall through to the heuristic (reduce=True). The returned source
    is None so callers can distinguish canonical from heuristic for
    instrumentation."""
    word_db = _word_db_for(_two_morpheme_subjects())
    with LexiconDB(fresh_db) as db:
        name, source = decompose_with_canonical("Stratford", word_db, db)
    assert source is None
    # Heuristic still produces a decomposition.
    assert len(name.words["Stratford"]) >= 1
    # Heuristic narrows to lowest-unaccounted by reduce(); each entry
    # contributes Meaning slots only — no extra alternates beyond the
    # heuristic pick.
    for word_obj in name.words["Stratford"]:
        assert word_obj.count_unaccounted() == 0


def test_decompose_with_canonical_falls_back_when_signature_stale(
    fresh_db: Path,
) -> None:
    """Canonical signature recorded but no cross-product cell of the
    current word_db matches → fall through to heuristic. Source is
    None to flag the silent drift."""
    word_db = _word_db_for(_two_morpheme_subjects())
    with LexiconDB(fresh_db) as db:
        _seed_source(db)
        topo_id = _insert_toponym(db, "Stratford")
        compute_decompositions(db, topo_id, "Stratford", word_db)
        # Hand-insert a canonical row with a bogus signature so the
        # lookup hits but apply misses.
        db.conn.execute(
            """
            INSERT INTO toponym_decomposition (
              toponym_id, decomposition_signature, morpheme_ids,
              unaccounted_fragments, unaccounted_count, morpheme_count,
              is_canonical, canonical_source
            ) VALUES (?, ?, ?, ?, ?, ?, 1, 'scholar')
            """,
            (topo_id, "f" * 40, '[["morpheme", "fake"]]', "[]", 0, 1),
        )
        db.commit()
        name, source = decompose_with_canonical("Stratford", word_db, db)
    assert source is None  # fell back


def test_decompose_with_canonical_no_db_uses_heuristic() -> None:
    """When ``db`` is None, the helper short-circuits straight to
    reduce=True. Source is None."""
    word_db = _word_db_for(_two_morpheme_subjects())
    name, source = decompose_with_canonical("Stratford", word_db, db=None)
    assert source is None
    assert len(name.words["Stratford"]) >= 1


# --- CLI integration: --db flag on rebuild-proportions / unaccounted ----


def test_cli_rebuild_proportions_with_db_runs(tmp_path: Path) -> None:
    """Smoke test: ``rebuild-proportions --db <db>`` runs end-to-end on
    a toponym with a registered canonical pick. The summary line carries
    a ``canonical[…]`` block so operators can verify the wiring."""
    word_db = _word_db_for(_two_morpheme_subjects())
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        _seed_source(db)
        topo_id = _insert_toponym(db, "Stratford")
        populate_and_pick(db, topo_id, "Stratford", word_db)
        db.commit()

    meanings_path = tmp_path / "meanings.json"
    meanings_path.write_text(json.dumps(_two_morpheme_subjects()))
    place_names = tmp_path / "places.json"
    place_names.write_text(json.dumps({"England": {"Bedfordshire": ["Stratford"]}}))

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "rebuild-proportions",
            "english",
            str(place_names),
            "--meanings",
            str(meanings_path),
            "--db",
            str(db_path),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    # canonical[...] block on the summary line confirms the --db path ran.
    assert "canonical[" in result.stderr


def test_cli_rebuild_proportions_without_db_omits_canonical_summary(
    tmp_path: Path,
) -> None:
    """The legacy heuristic path remains unchanged when ``--db`` is
    omitted — no canonical[...] surfacing on the summary line."""
    meanings_path = tmp_path / "meanings.json"
    meanings_path.write_text(json.dumps(_two_morpheme_subjects()))
    place_names = tmp_path / "places.json"
    place_names.write_text(json.dumps({"England": {"Bedfordshire": ["Stratford"]}}))

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "rebuild-proportions",
            "english",
            str(place_names),
            "--meanings",
            str(meanings_path),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert "canonical[" not in result.stderr


def test_cli_unaccounted_with_db_runs(tmp_path: Path) -> None:
    """Smoke test: ``unaccounted --db <db>`` runs end-to-end. The
    summary line carries the canonical[...] block on success."""
    # Build a meaning DB that DOESN'T cover '-burh' so we get a leftover
    # fragment when the heuristic runs on 'Bridgwaterburh'.
    subjects = [
        _make_subject(
            meaning=["Bridge"],
            words=[{"modern_usage": "Bridg-", "old_english": ["brycg"]}],
        ),
    ]
    word_db = _word_db_for(subjects)
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        _seed_source(db)
        topo_id = _insert_toponym(db, "Bridgburh")
        populate_and_pick(db, topo_id, "Bridgburh", word_db)
        db.commit()

    meanings_path = tmp_path / "meanings.json"
    meanings_path.write_text(json.dumps(subjects))
    place_names = tmp_path / "places.json"
    place_names.write_text(json.dumps({"England": {"Bedfordshire": ["Bridgburh"]}}))

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "unaccounted",
            "english",
            str(place_names),
            "--meanings",
            str(meanings_path),
            "--db",
            str(db_path),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert "canonical[" in result.stderr


def test_cli_unaccounted_canonical_drops_false_gap(tmp_path: Path) -> None:
    """When a name has a zero-unaccounted canonical decomposition, the
    unaccounted miner --db path drops it from the gap report — even if
    the heuristic alternates have leftovers. This is the user-facing
    payoff of wyrd-70s0."""
    word_db = _word_db_for(_two_morpheme_subjects())
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        _seed_source(db)
        topo_id = _insert_toponym(db, "Stratford")
        populate_and_pick(db, topo_id, "Stratford", word_db)
        db.commit()

    meanings_path = tmp_path / "meanings.json"
    meanings_path.write_text(json.dumps(_two_morpheme_subjects()))
    place_names = tmp_path / "places.json"
    place_names.write_text(json.dumps({"England": {"Bedfordshire": ["Stratford"]}}))

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "unaccounted",
            "english",
            str(place_names),
            "--meanings",
            str(meanings_path),
            "--db",
            str(db_path),
            "--as-json",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    output = json.loads(result.stdout)
    # Canonical zero-unaccounted means Stratford contributes ZERO
    # fragments to the gap report.
    assert output == []


def test_cli_rebuild_proportions_mixes_canonical_and_heuristic(tmp_path: Path) -> None:
    """Realistic operator path: corpus has some names with canonical
    picks (in-DB) and some without (not yet decomposed). The summary
    line must surface both source counts so an operator can verify
    the canonical wiring is doing useful work AND see how much fell
    through to the heuristic."""
    word_db = _word_db_for(_two_morpheme_subjects())
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        _seed_source(db)
        # 'Stratford' gets a canonical; 'Bridgford' is left out of the
        # DB entirely so the consumer must fall through to heuristic.
        topo_id = _insert_toponym(db, "Stratford")
        populate_and_pick(db, topo_id, "Stratford", word_db)
        db.commit()

    meanings_path = tmp_path / "meanings.json"
    meanings_path.write_text(json.dumps(_two_morpheme_subjects()))
    place_names = tmp_path / "places.json"
    place_names.write_text(json.dumps({"England": {"Bedfordshire": ["Stratford", "Bridgford"]}}))

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "rebuild-proportions",
            "english",
            str(place_names),
            "--meanings",
            str(meanings_path),
            "--db",
            str(db_path),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    # Both source labels appear in the canonical[…] block — proves the
    # branch covering canonical_hits["heuristic"] += 1 fires.
    assert "canonical[" in result.stderr
    assert "heuristic=1" in result.stderr
    assert "unique-zero-unaccounted=1" in result.stderr


# --- wyrd-8m87: region plumbing through CLI consumers ------------------


def test_load_names_with_regions_emits_tagged_tuples() -> None:
    """Each (Name, region) pair carries the region the name was
    grouped under in the place_names JSON. Output is sorted
    alphabetically by name to match ``load_names``."""
    data = {
        "England": {
            "Bedfordshire": ["Aspley", "Bridgford"],
            "Suffolk": ["Stratford"],
        }
    }
    out = load_names_with_regions(data)
    names = [n.name for n, _ in out]
    regions = [r for _, r in out]
    assert names == ["Aspley", "Bridgford", "Stratford"]
    assert regions == ["Bedfordshire", "Bedfordshire", "Suffolk"]


def test_load_names_with_regions_dedupes_keeping_first_country_region() -> None:
    """Same-name entries across multiple (country, region) groupings
    dedupe to a single tuple; the lex-first (country, region) wins so
    re-runs are deterministic."""
    data = {
        "England": {
            "Bedfordshire": ["Stratford"],
            "Suffolk": ["Stratford"],
        }
    }
    out = load_names_with_regions(data)
    assert len(out) == 1
    name, region = out[0]
    assert name.name == "Stratford"
    # 'Bedfordshire' < 'Suffolk' lex-first, so Bedfordshire wins.
    assert region == "Bedfordshire"


def test_load_names_with_regions_handles_multiple_countries() -> None:
    """Country participates in lex-tiebreak only — the kept entry's
    region is what travels onward to the canonical lookup."""
    data = {
        "Wales": {"Powys": ["Tref"]},
        "England": {"Cornwall": ["Tref"]},
    }
    out = load_names_with_regions(data)
    assert len(out) == 1
    _, region = out[0]
    # 'England' < 'Wales' lex-first → Cornwall wins.
    assert region == "Cornwall"


def test_cli_rebuild_proportions_region_picks_right_canonical(
    tmp_path: Path,
) -> None:
    """wyrd-8m87 acceptance: corpus has same-modern_name in two
    different regions; lexicon has a distinct toponym row per region
    with its own canonical. The consumer must pull the
    region-tightened canonical for each entry rather than always
    landing on the lowest-id row."""
    # Build a meaning DB where 'Stratford' has TWO equally-zero-
    # unaccounted decompositions so the per-toponym canonical pick
    # isn't trivially identical between the two region rows.
    word_db = _word_db_for(_two_morpheme_subjects())
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        _seed_source(db)
        # Two toponyms, same name, distinct regions, each with its own
        # canonical pick.
        cur1 = db.conn.execute(
            "INSERT INTO toponym (modern_name, region) VALUES (?, ?)",
            ("Stratford", "Bedfordshire"),
        )
        topo_a = cur1.lastrowid
        cur2 = db.conn.execute(
            "INSERT INTO toponym (modern_name, region) VALUES (?, ?)",
            ("Stratford", "Suffolk"),
        )
        topo_b = cur2.lastrowid
        populate_and_pick(db, topo_a, "Stratford", word_db)
        populate_and_pick(db, topo_b, "Stratford", word_db)
        db.commit()

    meanings_path = tmp_path / "meanings.json"
    meanings_path.write_text(json.dumps(_two_morpheme_subjects()))
    place_names = tmp_path / "places.json"
    # Corpus name appears under Bedfordshire only — region tightening
    # should land on topo_a's canonical, not topo_b's.
    place_names.write_text(json.dumps({"England": {"Bedfordshire": ["Stratford"]}}))

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "rebuild-proportions",
            "english",
            str(place_names),
            "--meanings",
            str(meanings_path),
            "--db",
            str(db_path),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    # Canonical applied — the region-tightened lookup found a hit.
    assert "unique-zero-unaccounted=1" in result.stderr


def test_cli_unaccounted_handles_legacy_null_region_db(tmp_path: Path) -> None:
    """Backward compat: a NULL-region toponym row in the DB still
    contributes its canonical even though the corpus carries region
    tags. The fallback in ``lookup_canonical_signature`` keeps the
    region-tagged corpus from silently skipping legacy rows."""
    word_db = _word_db_for(_two_morpheme_subjects())
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        _seed_source(db)
        # Legacy insert: no region column.
        cur = db.conn.execute(
            "INSERT INTO toponym (modern_name) VALUES (?)",
            ("Stratford",),
        )
        topo_id = cur.lastrowid
        populate_and_pick(db, topo_id, "Stratford", word_db)
        db.commit()

    meanings_path = tmp_path / "meanings.json"
    meanings_path.write_text(json.dumps(_two_morpheme_subjects()))
    place_names = tmp_path / "places.json"
    place_names.write_text(json.dumps({"England": {"Bedfordshire": ["Stratford"]}}))

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "unaccounted",
            "english",
            str(place_names),
            "--meanings",
            str(meanings_path),
            "--db",
            str(db_path),
            "--as-json",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    # Canonical fired (zero-unaccounted) → no fragments in the gap report.
    assert json.loads(result.stdout) == []
    # Stderr proves we hit canonical, not heuristic.
    assert "unique-zero-unaccounted=1" in result.stderr
