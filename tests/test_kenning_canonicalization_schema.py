"""Schema test for the canonicalization graph migration (wyrd-u6fn.2, 0019)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from wyrd.generators.kenning.lexicon.sql import upgrade_head
from wyrd.generators.kenning.lexicon.sql._config import alembic_config

_CANONICAL_TABLES = (
    "canonical_morpheme",
    "canonical_place",
    "canonical_sense",
    "canonical_label",
)
_BIND_COLUMNS = {
    "etymon": "canonical_morpheme_id",
    "etymon_gloss": "canonical_sense_id",
    "toponym": "canonical_place_id",
    "toponym_etymology": "canonical_place_id",
}


def _tables(conn) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _columns(conn, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def test_fresh_head_creates_canonical_graph(tmp_path: Path) -> None:
    db = tmp_path / "lexicon.db"
    upgrade_head(db)  # runs all migrations incl. 0019
    conn = sqlite3.connect(db)
    try:
        tabs = _tables(conn)
        for t in _CANONICAL_TABLES:
            assert t in tabs, f"missing canonical table {t}"
        for table, col in _BIND_COLUMNS.items():
            assert col in _columns(conn, table), f"{table} missing {col}"
    finally:
        conn.close()


def test_merged_into_self_ref_and_binding_fk(tmp_path: Path) -> None:
    db = tmp_path / "lexicon.db"
    upgrade_head(db)
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        conn.execute("INSERT INTO canonical_morpheme(id) VALUES ('CM-a')")
        # merge-canonical reconciliation: merged_into is a self-FK
        conn.execute("INSERT INTO canonical_morpheme(id, merged_into) VALUES ('CM-b', 'CM-a')")
        # an observation binds to a canonical node
        conn.execute(
            "INSERT INTO etymon(canonical_form, language, canonical_morpheme_id) "
            "VALUES ('niwe', 'old-english', 'CM-a')"
        )
        conn.commit()
        # a dangling bind violates the FK
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO etymon(canonical_form, language, canonical_morpheme_id) "
                "VALUES ('x', 'old-english', 'CM-nope')"
            )
    finally:
        conn.close()


def test_canonical_label_stratum_defaults_empty_and_pk(tmp_path: Path) -> None:
    db = tmp_path / "lexicon.db"
    upgrade_head(db)
    conn = sqlite3.connect(db)
    try:
        # stratum omitted → '' default (era-invariant); composite PK holds
        conn.execute(
            "INSERT INTO canonical_label(canonical_type, canonical_id, value) "
            "VALUES ('canonical_place', 'CP-newton', 'Newton')"
        )
        conn.execute(
            "INSERT INTO canonical_label(canonical_type, canonical_id, stratum, value) "
            "VALUES ('canonical_place', 'CP-newton', 'domesday', 'Neutune')"
        )
        conn.commit()
        rows = conn.execute(
            "SELECT stratum, value FROM canonical_label "
            "WHERE canonical_type='canonical_place' AND canonical_id='CP-newton' "
            "ORDER BY stratum"
        ).fetchall()
        assert rows == [("", "Newton"), ("domesday", "Neutune")]
        # duplicate (type, id, stratum) violates the PK
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO canonical_label(canonical_type, canonical_id, value) "
                "VALUES ('canonical_place', 'CP-newton', 'Newtown')"
            )
    finally:
        conn.close()


def test_downgrade_reverses_the_migration(tmp_path: Path) -> None:
    from alembic.command import downgrade

    db = tmp_path / "lexicon.db"
    upgrade_head(db)
    cfg = alembic_config(db)
    downgrade(cfg, "0018_genitive_split_prior")
    conn = sqlite3.connect(db)
    try:
        tabs = _tables(conn)
        for t in _CANONICAL_TABLES:
            assert t not in tabs, f"{t} survived downgrade"
        for table, col in _BIND_COLUMNS.items():
            assert col not in _columns(conn, table), f"{table}.{col} survived downgrade"
    finally:
        conn.close()


def test_on_delete_set_null_nulls_the_bind(tmp_path: Path) -> None:
    """Deleting a bound canonical node nulls the bind (ON DELETE SET NULL),
    never errors or cascades — the merge/un-bind reconciliation semantic."""
    db = tmp_path / "lexicon.db"
    upgrade_head(db)
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        conn.execute("INSERT INTO canonical_morpheme(id) VALUES ('CM-a')")
        conn.execute(
            "INSERT INTO etymon(canonical_form, language, canonical_morpheme_id) "
            "VALUES ('niwe', 'old-english', 'CM-a')"
        )
        conn.commit()
        conn.execute("DELETE FROM canonical_morpheme WHERE id='CM-a'")
        conn.commit()
        bind = conn.execute(
            "SELECT canonical_morpheme_id FROM etymon WHERE canonical_form='niwe'"
        ).fetchone()[0]
        assert bind is None, "bind should be SET NULL, not cascaded/orphaned"
    finally:
        conn.close()


def test_merged_into_set_null_on_target_delete(tmp_path: Path) -> None:
    """Deleting a merge target nulls the loser's merged_into (SET NULL),
    un-merging it rather than cascading the row away."""
    db = tmp_path / "lexicon.db"
    upgrade_head(db)
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        conn.execute("INSERT INTO canonical_place(id) VALUES ('CP-a')")
        conn.execute("INSERT INTO canonical_place(id, merged_into) VALUES ('CP-b', 'CP-a')")
        conn.commit()
        conn.execute("DELETE FROM canonical_place WHERE id='CP-a'")
        conn.commit()
        row = conn.execute("SELECT id, merged_into FROM canonical_place WHERE id='CP-b'").fetchone()
        assert row == ("CP-b", None), "loser should survive with merged_into NULLed"
    finally:
        conn.close()


def test_override_bind_and_gloss_bind_fks(tmp_path: Path) -> None:
    """The toponym_etymology override bind (D50.6) and the etymon_gloss bind
    both enforce their canonical FK — exercising the columns the presence
    checks don't insert into."""
    db = tmp_path / "lexicon.db"
    upgrade_head(db)
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        conn.execute("INSERT INTO canonical_place(id) VALUES ('CP-x')")
        conn.execute("INSERT INTO canonical_sense(id) VALUES ('CS-x')")
        conn.execute("INSERT INTO source(id, title) VALUES ('s1', 'T')")
        tid = conn.execute(
            "INSERT INTO toponym(modern_name) VALUES ('Newton') RETURNING id"
        ).fetchone()[0]
        # valid override bind on the etymology row
        conn.execute(
            "INSERT INTO toponym_etymology(toponym_id, source_id, canonical_place_id) "
            "VALUES (?, 's1', 'CP-x')",
            (tid,),
        )
        eid = conn.execute(
            "INSERT INTO etymon(canonical_form, language) VALUES ('tun', 'old-english') "
            "RETURNING id"
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO etymon_gloss(etymon_id, gloss, canonical_sense_id) "
            "VALUES (?, 'settlement', 'CS-x')",
            (eid,),
        )
        conn.commit()
        # dangling refs rejected on both
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO toponym_etymology(toponym_id, source_id, canonical_place_id) "
                "VALUES (?, 's1', 'CP-nope')",
                (tid,),
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO etymon_gloss(etymon_id, gloss, canonical_sense_id) "
                "VALUES (?, 'farm', 'CS-nope')",
                (eid,),
            )
    finally:
        conn.close()


def test_canonical_type_check_rejects_unknown(tmp_path: Path) -> None:
    """canonical_label.canonical_type is CHECK-constrained to the three node
    types — a typo'd discriminator is rejected, not silently orphaned."""
    db = tmp_path / "lexicon.db"
    upgrade_head(db)
    conn = sqlite3.connect(db)
    try:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO canonical_label(canonical_type, canonical_id, value) "
                "VALUES ('canonical_morphmeme', 'CM-x', 'oops')"  # typo
            )
    finally:
        conn.close()


_NEW_INDEXES = (
    "idx_canonical_morpheme_merged",
    "idx_canonical_place_merged",
    "idx_canonical_sense_merged",
    "idx_etymon_canonical_morpheme",
    "idx_etymon_gloss_canonical_sense",
    "idx_toponym_canonical_place",
    "idx_toponym_etymology_canonical_place",
)


def _indexes(conn) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}


def test_indexes_created_then_dropped(tmp_path: Path) -> None:
    from alembic.command import downgrade

    db = tmp_path / "lexicon.db"
    upgrade_head(db)
    conn = sqlite3.connect(db)
    try:
        idx = _indexes(conn)
        for name in _NEW_INDEXES:
            assert name in idx, f"index {name} missing after upgrade"
    finally:
        conn.close()
    downgrade(alembic_config(db), "0018_genitive_split_prior")
    conn = sqlite3.connect(db)
    try:
        idx = _indexes(conn)
        for name in _NEW_INDEXES:
            assert name not in idx, f"index {name} survived downgrade"
    finally:
        conn.close()


def test_upgrade_downgrade_upgrade_is_rerunnable(tmp_path: Path) -> None:
    """A surviving artifact from the hand-ordered downgrade would make a second
    upgrade fail with 'already exists'. Prove the round-trip is clean."""
    from alembic.command import downgrade

    db = tmp_path / "lexicon.db"
    upgrade_head(db)
    cfg = alembic_config(db)
    downgrade(cfg, "0018_genitive_split_prior")
    upgrade_head(db)  # must not raise
    conn = sqlite3.connect(db)
    try:
        for t in _CANONICAL_TABLES:
            assert t in _tables(conn), f"{t} missing after re-upgrade"
    finally:
        conn.close()
