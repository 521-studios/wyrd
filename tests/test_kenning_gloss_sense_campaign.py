"""Tests for the compressed-gloss rail (wyrd-u6fn.10.1): impact ordering, the
committed-ledger remainder (wyrd-1rw4), the unglossed/full-inventory cohorts,
parking, and the coverage scoreboard.
"""

from __future__ import annotations

from wyrd.generators.kenning.canonicalization import append_assertion
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema
from wyrd.generators.kenning.lexicon.gloss_sense import GlossSenseCandidate, Sense, sense_assertions
from wyrd.generators.kenning.lexicon.gloss_sense_campaign import (
    next_slice,
    sense_progress,
)


def _db(tmp_path):
    init_schema(tmp_path / "lexicon.db")
    return LexiconDB(tmp_path / "lexicon.db")


def _etymon(db, form, glosses, *, language="old-english"):
    eid = db.conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES (?, ?)", (form, language)
    ).lastrowid
    for g in glosses:
        db.conn.execute("INSERT INTO etymon_gloss (etymon_id, gloss) VALUES (?, ?)", (eid, g))
    return eid


_toponym_seq = [0]


def _attribute(db, etymon_id, n_toponyms):
    """Attribute a morpheme to n distinct toponyms (so its impact == n)."""
    db.conn.execute("INSERT OR IGNORE INTO source (id, title) VALUES ('s', 's')")
    for _ in range(n_toponyms):
        _toponym_seq[0] += 1
        t = db.conn.execute(
            "INSERT INTO toponym (modern_name, country) VALUES (?, 'England')",
            (f"T{_toponym_seq[0]}",),
        ).lastrowid
        te = db.conn.execute(
            "INSERT INTO toponym_etymology (toponym_id, source_id, confidence) VALUES (?, 's', 'high')",
            (t,),
        ).lastrowid
        db.conn.execute(
            "INSERT INTO toponym_etymology_element (toponym_etymology_id, etymon_id, ordinal) VALUES (?, ?, 0)",
            (te, etymon_id),
        )


def _author_sense(tmp_path, ref, language, form, label, glosses):
    c = GlossSenseCandidate(
        ref=ref,
        language=language,
        canonical_form=form,
        senses=(Sense(label, tuple(glosses)),),
        confidence="high",
        method="opus-gloss-sense-v1",
        model="opus",
    )
    for a in sense_assertions(c):
        append_assertion(tmp_path, a)


def test_next_slice_is_impact_ordered_and_glossed(tmp_path):
    db = _db(tmp_path)
    tun = _etymon(db, "tūn", ["enclosure", "town"])
    leah = _etymon(db, "lēah", ["wood", "clearing"])
    _attribute(db, tun, 3)
    _attribute(db, leah, 2)
    db.commit()
    sl = next_slice(db.conn, tmp_path, n=10, parked_path=tmp_path / "_sense_parked.jsonl")
    assert [e.canonical_form for e in sl] == ["tūn", "lēah"]  # impact 3 before 2
    assert sl[0].glosses == ("enclosure", "town")
    db.close()


def test_next_slice_excludes_committed_and_below_impact(tmp_path):
    db = _db(tmp_path)
    tun = _etymon(db, "tūn", ["enclosure", "town"])
    leah = _etymon(db, "lēah", ["wood", "clearing"])
    rare = _etymon(db, "rāru", ["obscure"])
    _attribute(db, tun, 3)
    _attribute(db, leah, 2)
    _attribute(db, rare, 1)  # below the impact>=2 floor
    db.commit()
    # tūn already sense-bound in the committed ledger -> excluded (wyrd-1rw4).
    _author_sense(tmp_path, "old-english:tūn", "old-english", "tūn", "town", ["enclosure", "town"])
    refs = [
        e.ref
        for e in next_slice(db.conn, tmp_path, n=10, parked_path=tmp_path / "_sense_parked.jsonl")
    ]
    assert refs == ["old-english:lēah"]  # tūn done, rāru below floor
    db.close()


def test_next_slice_parked_excluded(tmp_path):
    db = _db(tmp_path)
    leah = _etymon(db, "lēah", ["wood"])
    _attribute(db, leah, 2)
    db.commit()
    parked = tmp_path / "_sense_parked.jsonl"
    parked.write_text('{"ref": "old-english:lēah", "reason": "fragment"}\n', encoding="utf-8")
    assert next_slice(db.conn, tmp_path, n=10, parked_path=parked) == []
    db.close()


def test_unglossed_cohort_for_phase_2(tmp_path):
    db = _db(tmp_path)
    glossed = _etymon(db, "tūn", ["town"])
    bare = _etymon(db, "barf", [])  # no gloss wall
    _attribute(db, glossed, 2)
    _attribute(db, bare, 2)
    db.commit()
    parked = tmp_path / "_sense_parked.jsonl"
    glossed_slice = [
        e.canonical_form for e in next_slice(db.conn, tmp_path, n=10, parked_path=parked)
    ]
    unglossed_slice = [
        e.canonical_form
        for e in next_slice(db.conn, tmp_path, n=10, require_gloss=False, parked_path=parked)
    ]
    assert glossed_slice == ["tūn"]
    assert unglossed_slice == ["barf"]  # phase 2 sees only the UNGLOSSED ones
    db.close()


def test_full_inventory_reaches_non_breakdown_morphemes(tmp_path):
    db = _db(tmp_path)
    # A glossed morpheme NOT attributed to any toponym (impact 0) is invisible to
    # phase 1 but surfaced by phase 3 (full inventory).
    _etymon(db, "orphan", ["lonely", "stray"])
    db.commit()
    parked = tmp_path / "_sense_parked.jsonl"
    assert next_slice(db.conn, tmp_path, n=10, parked_path=parked) == []
    inv = next_slice(db.conn, tmp_path, n=10, full_inventory=True, parked_path=parked)
    assert [e.canonical_form for e in inv] == ["orphan"]
    db.close()


def test_sense_progress_counts(tmp_path):
    db = _db(tmp_path)
    a = _etymon(db, "tūn", ["town"])
    b = _etymon(db, "lēah", ["wood"])
    c = _etymon(db, "ford", ["ford"])
    for e in (a, b, c):
        _attribute(db, e, 2)
    db.commit()
    parked = tmp_path / "_sense_parked.jsonl"
    parked.write_text('{"ref": "old-english:ford", "reason": "x"}\n', encoding="utf-8")
    _author_sense(tmp_path, "old-english:tūn", "old-english", "tūn", "town", ["town"])
    done, parked_n, total = sense_progress(db.conn, tmp_path, parked, min_impact=2)
    assert (done, parked_n, total) == (1, 1, 3)
    db.close()
