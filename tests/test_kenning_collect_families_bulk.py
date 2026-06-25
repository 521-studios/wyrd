"""wyrd-4zyb: the bulk attested-years prefetch (one indexed-temp-table join)
must return results equal to the per-family query it replaces — that per-root
query was ~96% of collect_families' wall time."""

from __future__ import annotations

from pathlib import Path

import pytest

from wyrd.generators.kenning.lexicon import LexiconDB, init_schema
from wyrd.generators.kenning.lexicon.bundle._export import _iterate_families_with_progress
from wyrd.generators.kenning.lexicon.bundle._family import (
    _bulk_attested_years,
    _fetch_member_attested_years,
    _fetch_member_variants,
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


# --- wyrd-nfg2: fault-injection for the cursor-close-before-DROP guard --------


class _ExplodingCursor:
    """A SELECT cursor that fails after yielding one row, deliberately leaving
    the underlying real cursor OPEN. This reproduces the condition the
    ``finally`` guard in ``_bulk_attested_years`` defends against: a still-open
    cursor holds a read lock, so a subsequent ``DROP TABLE`` raises "database
    table is locked" and masks the real exception."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        # Delegate any other cursor method/attr (fetchall, description, …) so the
        # double survives a refactor that touches the cursor in a new way.
        return getattr(self._inner, name)

    def __iter__(self):
        for row in self._inner:
            yield row  # inner cursor is now mid-scan, holding a read lock
            raise RuntimeError("boom mid-iteration")

    def close(self):
        self._inner.close()


class _SelectExplodesConn:
    """Delegates everything to a real sqlite3 connection except the result query
    that joins ``_bulk_attested_members``, which it wraps in an
    :class:`_ExplodingCursor`."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def __enter__(self):
        # Return self (not the raw connection) so a future `with conn as c:` in
        # the code under test still routes through this wrapper.
        self._real.__enter__()
        return self

    def __exit__(self, *exc):
        return self._real.__exit__(*exc)

    def execute(self, sql, *args):
        cur = self._real.execute(sql, *args)
        # Pin the fault to the bulk RESULT query (a SELECT/WITH that joins the
        # temp table) — not the DROP/CREATE that also name it, and robust if the
        # query is later rewritten as a CTE.
        head = sql.lstrip().upper()
        if head.startswith(("SELECT", "WITH")) and "_bulk_attested_members" in sql:
            return _ExplodingCursor(cur)
        return cur


class _FakeDB:
    """Minimal stand-in exposing only the ``conn`` attribute the function uses."""

    def __init__(self, conn):
        self.conn = conn


def test_cleanup_when_iteration_raises(tmp_path: Path) -> None:
    """wyrd-nfg2: if iterating the result cursor raises, ``_bulk_attested_years``
    must close that cursor in ``finally`` BEFORE dropping the temp table.
    Otherwise the open cursor's read lock makes ``DROP`` raise "database table
    is locked", masking the original error and leaving the table behind. Assert
    the ORIGINAL exception propagates and the connection is still reusable."""
    p = tmp_path / "lex.db"
    init_schema(p)
    with LexiconDB(p) as db:
        e1, e2, _ = _seed(db)
        rigged = _FakeDB(_SelectExplodesConn(db.conn))
        with pytest.raises(RuntimeError, match="boom mid-iteration"):
            _bulk_attested_years(rigged, [e1, e2])
        # The temp table was dropped despite the failure, and the real
        # connection still works — a fresh call against it succeeds.
        assert "_bulk_attested_members" not in {
            r[0] for r in db.conn.execute("SELECT name FROM sqlite_temp_master WHERE type='table'")
        }
        assert _bulk_attested_years(db, [e1, e2]) == {e1: 1086, e2: 950}


def test_cross_family_isolation(tmp_path: Path) -> None:
    """wyrd-nfg2: the prefetch builds ONE map over all members up front, then
    each family slices only the ids it owns. Two single-member families with
    different attested years must each carry only their own year — the bulk map
    must not bleed one family's year into another."""
    p = tmp_path / "lex.db"
    init_schema(p)
    with LexiconDB(p) as db:
        db.upsert_source(id="s", title="S")
        a = db.upsert_etymon("alpha", "old-english")
        b = db.upsert_etymon("beta", "old-english")
        db.conn.execute(
            "INSERT INTO etymon_text_match (etymon_id, source_id, matched_form, "
            "match_count, edit_distance, attested_year) VALUES (?, 's', 'alpha', 1, 0, 1000)",
            (a,),
        )
        db.conn.execute(
            "INSERT INTO etymon_text_match (etymon_id, source_id, matched_form, "
            "match_count, edit_distance, attested_year) VALUES (?, 's', 'beta', 1, 0, 2000)",
            (b,),
        )
        db.commit()
        families = _iterate_families_with_progress(db, [a, b], {a: [a], b: [b]})
    # _iterate_families_with_progress preserves root_ids order, so the exact
    # list pins per-family slicing AND isolation (no extra or bled-in keys).
    years = [f["member_attested_years"] for f in families]
    assert years == [{a: 1000}, {b: 2000}]


def test_fetch_member_variants_deterministic_casing_on_collision(tmp_path: Path) -> None:
    """Case-variant collisions ("Chimaera"/"chimaera") fold into ONE
    ``GROUP BY etymon_id, LOWER(matched_form)`` row, so the emitted casing must be
    deterministic. A bare non-aggregated ``matched_form`` would surface an
    engine-defined arbitrary row's casing — a byte-reproducibility hole, since the
    bundle must rebuild identically across SQLite versions / query plans. The query
    pins it with ``MAX(matched_form)`` → the lexicographically-greatest casing
    (lowercase-leading wins in ASCII)."""
    p = tmp_path / "lex.db"
    init_schema(p)
    with LexiconDB(p) as db:
        db.upsert_source(id="s", title="S")
        eid = db.upsert_etymon("chim", "old-english")  # canonical differs from the variant
        for form in ("Chimaera", "chimaera"):
            db.conn.execute(
                "INSERT INTO etymon_text_match (etymon_id, source_id, matched_form, "
                "match_count, edit_distance, attested_year, method) "
                "VALUES (?, 's', ?, 2, 0, NULL, 'reverse-search')",
                (eid, form),
            )
        db.commit()
        variants = _fetch_member_variants(db, [eid], canonical_forms_lower=set())
        forms = [f for f, _count in variants[eid]]
        # The two casings fold to ONE variant, deterministically the MAX casing.
        assert forms == ["chimaera"]  # MAX("Chimaera", "chimaera") — not engine-arbitrary
