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
            "WHERE canonical_id='CP-newton' ORDER BY stratum"
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
