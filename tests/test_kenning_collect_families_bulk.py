"""wyrd-4zyb: the bulk attested-years prefetch (one indexed-temp-table join)
must return results equal to the per-family query it replaces — that per-root
query was ~96% of collect_families' wall time."""

from __future__ import annotations

from pathlib import Path

from wyrd.generators.kenning.lexicon import LexiconDB, init_schema
from wyrd.generators.kenning.lexicon.bundle._family import (
    _bulk_attested_years,
    _fetch_member_attested_years,
)


def _seed(db: LexiconDB) -> tuple[int, int, int]:
    db.upsert_source(id="s", title="S")
    e1 = db.upsert_etymon("denu", "old-english")  # year from BOTH sources → cross-source MIN
    e2 = db.upsert_etymon("ham", "old-english")  # year via etymon_text_match only
    e3 = db.upsert_etymon("noyear", "old-english")  # no attested year on either source
    # e1: a text-match year (1200) AND an earlier toponym year (1086) → MIN must cross sources.
    db.conn.execute(
        "INSERT INTO etymon_text_match (etymon_id, source_id, matched_form, "
        "match_count, edit_distance, attested_year) VALUES (?, 's', 'denu', 1, 0, 1200)",
        (e1,),
    )
    tid = db.conn.execute("INSERT INTO toponym (modern_name) VALUES ('T')").lastrowid
    te = db.conn.execute(
        "INSERT INTO toponym_etymology (toponym_id, source_id, attested_year) VALUES (?, 's', 1086)",
        (tid,),
    ).lastrowid
    db.conn.execute(
        "INSERT INTO toponym_etymology_element (toponym_etymology_id, ordinal, etymon_id) "
        "VALUES (?, 0, ?)",
        (te, e1),
    )
    # e2: text-match year only.
    db.conn.execute(
        "INSERT INTO etymon_text_match (etymon_id, source_id, matched_form, "
        "match_count, edit_distance, attested_year) VALUES (?, 's', 'ham', 1, 0, 950)",
        (e2,),
    )
    # e3 carries a text-match row but NULL attested_year → must be absent from the result.
    db.conn.execute(
        "INSERT INTO etymon_text_match (etymon_id, source_id, matched_form, "
        "match_count, edit_distance, attested_year) VALUES (?, 's', 'noyear', 1, 0, NULL)",
        (e3,),
    )
    db.commit()
    return e1, e2, e3


def test_bulk_matches_per_root(tmp_path: Path) -> None:
    p = tmp_path / "lex.db"
    init_schema(p)
    with LexiconDB(p) as db:
        e1, e2, e3 = _seed(db)
        ids = [e1, e2, e3]
        bulk = _bulk_attested_years(db, ids)
        # equivalence with the query it replaces (the whole point of the change)
        assert bulk == _fetch_member_attested_years(db, ids)
        # MIN across BOTH sources; e3 (NULL year) absent
        assert bulk == {e1: 1086, e2: 950}


def test_temp_table_is_dropped(tmp_path: Path) -> None:
    p = tmp_path / "lex.db"
    init_schema(p)
    with LexiconDB(p) as db:
        e1, e2, e3 = _seed(db)
        _bulk_attested_years(db, [e1, e2, e3])
        temp = {
            r[0] for r in db.conn.execute("SELECT name FROM sqlite_temp_master WHERE type='table'")
        }
        assert "_bulk_attested_members" not in temp  # connection-local, dropped on the way out


def test_empty_input(tmp_path: Path) -> None:
    p = tmp_path / "lex.db"
    init_schema(p)
    with LexiconDB(p) as db:
        assert _bulk_attested_years(db, []) == {}
        assert _bulk_attested_years(db, set()) == {}
