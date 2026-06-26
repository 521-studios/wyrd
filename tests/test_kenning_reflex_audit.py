"""wyrd-862b — reflex-link audit: deterministic pre-screen + prune.

The LLM judgment isn't tested (it's the arbiter, mocked nowhere — D10 real-DB
style covers the deterministic halves). These pin the two code paths the apply
relies on: which links the pre-screen treats as suspicious (LLM candidates) vs
known-forms (kept outright), and that prune deletes only the verdicted links
while leaving reflex ROWS intact.
"""

from __future__ import annotations

import sqlite3

import pytest

from wyrd.generators.kenning.lexicon.reflex_audit import (
    _known_forms,
    find_candidates,
    prune_links,
    strip_surface,
)


def _fixture() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    db.executescript(
        """
        CREATE TABLE etymon (id INTEGER PRIMARY KEY, canonical_form TEXT, language TEXT,
                             cognate_id INTEGER, merged_into_id INTEGER);
        CREATE TABLE etymon_gloss (etymon_id INTEGER, gloss TEXT);
        CREATE TABLE etymon_descent (parent_id INTEGER, child_id INTEGER, edge_type TEXT);
        CREATE TABLE etymon_variant (etymon_id INTEGER, form TEXT);
        CREATE TABLE reflex (id INTEGER PRIMARY KEY, surface_form TEXT, position TEXT, productivity INTEGER);
        CREATE TABLE reflex_etymon (reflex_id INTEGER, etymon_id INTEGER);
        """
    )
    # hām (id1), cognate cluster 100 with a modern member 'home' (id2).
    db.execute("INSERT INTO etymon VALUES (1,'hām','old-english',100,NULL)")
    db.execute("INSERT INTO etymon VALUES (2,'home','modern-english',100,NULL)")
    db.execute("INSERT INTO etymon_gloss VALUES (1,'dwelling')")
    # reflexes: '-ham' (known form), '-burnham' (compound), 'Belch-' (neighbor).
    db.execute("INSERT INTO reflex VALUES (10,'-ham','post',0)")
    db.execute("INSERT INTO reflex VALUES (11,'-burnham','post',0)")
    db.execute("INSERT INTO reflex VALUES (12,'Belch-','pre',0)")
    db.executemany(
        "INSERT INTO reflex_etymon VALUES (?,?)",
        [(10, 1), (11, 1), (12, 1)],
    )
    db.commit()
    return db


def test_pre_screen_keeps_known_forms_flags_the_rest():
    db = _fixture()
    cands = find_candidates(db)
    surfaces = {c.surface for c in cands}
    # '-ham' is a known form (== canonical 'ham') -> NOT a candidate.
    assert "-ham" not in surfaces
    # the compound + neighbor are not known forms -> candidates for the LLM.
    assert surfaces == {"-burnham", "Belch-"}


def test_known_form_via_cluster_mate_is_kept():
    """A surface matching a cognate-cluster mate (not the canonical) is a known
    form — 'home' is hām's modern cluster reflex, so a '-home' reflex is kept."""
    db = _fixture()
    db.execute("INSERT INTO reflex VALUES (13,'-home','post',0)")
    db.execute("INSERT INTO reflex_etymon VALUES (13,1)")
    db.commit()
    assert "-home" not in {c.surface for c in find_candidates(db)}


def test_dedup_key_collapses_ocr_case_variants_and_prunes_all_links():
    """The verdict key folds accent + case, so one judgment on 'tūn' covers its
    OCR/case-variant etymon rows (Tun/tun); a single PRUNE key must delete the
    bad reflex link to EVERY variant."""
    db = _fixture()
    # two variant etymons of the same morpheme, both linked from one bad surface.
    db.execute("INSERT INTO etymon VALUES (3,'tūn','old-english',NULL,NULL)")
    db.execute("INSERT INTO etymon VALUES (4,'Tun','old-english',NULL,NULL)")
    db.execute("INSERT INTO etymon_gloss VALUES (3,'enclosure')")
    db.execute("INSERT INTO etymon_gloss VALUES (4,'enclosure')")
    db.execute("INSERT INTO reflex VALUES (20,'-bolton','post',0)")
    db.executemany("INSERT INTO reflex_etymon VALUES (?,?)", [(20, 3), (20, 4)])
    db.commit()
    cands = {c.etymon_id: c for c in find_candidates(db) if c.surface == "-bolton"}
    # both variant links are candidates AND share ONE key (accent/case folded).
    assert cands[3].key == cands[4].key == (strip_surface("-bolton"), "tun", "enclosure")
    n = prune_links(db, {cands[3].key})
    assert n == 2  # one verdict deleted BOTH variant links
    assert not [c for c in find_candidates(db) if c.surface == "-bolton"]


def test_pre_screen_keeps_variant_and_inheritance_forms_not_cognate():
    """A surface matching an etymon_variant form or an inheritance/borrowing
    descent child is a known form (kept); a 'cognate' descent edge is too loose
    and does NOT count, so a surface only reachable via cognate stays suspicious."""
    db = _fixture()
    db.execute("INSERT INTO etymon_variant VALUES (1,'hamm')")  # spelling variant of hām
    db.execute(
        "INSERT INTO etymon VALUES (5,'heem','modern-english',NULL,NULL)"
    )  # inheritance child
    db.execute("INSERT INTO etymon VALUES (6,'hejm','old-norse',NULL,NULL)")  # cognate peer
    db.execute("INSERT INTO etymon_descent VALUES (1,5,'inheritance')")
    db.execute("INSERT INTO etymon_descent VALUES (1,6,'cognate')")
    db.executemany(
        "INSERT INTO reflex VALUES (?,?,?,0)",
        [(30, "-hamm", "post"), (31, "-heem", "post"), (32, "-hejm", "post")],
    )
    db.executemany("INSERT INTO reflex_etymon VALUES (?,?)", [(30, 1), (31, 1), (32, 1)])
    db.commit()
    suspicious = {c.surface for c in find_candidates(db)}
    assert "-hamm" not in suspicious  # etymon_variant form -> known
    assert "-heem" not in suspicious  # inheritance descent child -> known
    assert "-hejm" in suspicious  # cognate edge does NOT count -> still suspicious


def test_prune_deletes_verdicted_links_keeps_reflex_rows():
    db = _fixture()
    # PRUNE the two suspicious surfaces; the verdict key is (strip, canon, gloss).
    prune_keys = {
        (strip_surface("-burnham"), strip_surface("hām"), "dwelling"),
        (strip_surface("Belch-"), strip_surface("hām"), "dwelling"),
    }
    n = prune_links(db, prune_keys)
    assert n == 2
    # the bad links are gone; the good '-ham' link survives.
    remaining = {
        r[0]
        for r in db.execute(
            "SELECT rf.surface_form FROM reflex_etymon re JOIN reflex rf ON rf.id=re.reflex_id"
        )
    }
    assert remaining == {"-ham"}
    # reflex ROWS are untouched — '-burnham' survives as a standalone (orphan)
    # subject, never re-polluting hām's grid.
    all_rows = {r[0] for r in db.execute("SELECT surface_form FROM reflex")}
    assert all_rows == {"-ham", "-burnham", "Belch-"}


def test_known_forms_tolerates_absent_etymon_variant():
    """etymon_variant is an additive table: on an older/partial DB that lacks it,
    _known_forms must degrade to an empty variant set, not crash. Pins the
    intended fail-soft (preserved by the narrowed except)."""
    db = _fixture()
    db.row_factory = sqlite3.Row
    db.execute("DROP TABLE etymon_variant")
    _et, _cluster, var, _desc, _gloss = _known_forms(db)
    assert dict(var) == {}  # absent table tolerated, no exception


def test_known_forms_surfaces_etymon_variant_schema_error():
    """Regression: a REAL operational error on etymon_variant — here a schema
    mismatch (table present but missing the 'form' column, e.g. a partial
    migration) — must SURFACE, not be silently swallowed as if the table were
    absent. The former bare ``except OperationalError: pass`` masked it and
    degraded the audit to an empty variant set on a corrupt DB."""
    db = _fixture()
    db.row_factory = sqlite3.Row
    db.execute("DROP TABLE etymon_variant")
    db.execute("CREATE TABLE etymon_variant (etymon_id INTEGER)")  # no 'form' column
    with pytest.raises(sqlite3.OperationalError, match="form"):
        _known_forms(db)
