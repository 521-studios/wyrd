"""Tests for gloss-sense clustering + authoring (wyrd-u6fn.10).

Covers candidate parsing, the grounding/partition validation gate, the
canonical_sense identity authoring, the committed-ledger remainder, and an
end-to-end round-trip: candidate -> assertions -> projection sets
etymon_gloss.canonical_sense_id + the per-sense compressed label.
"""

from __future__ import annotations

from wyrd.generators.kenning.canonicalization import (
    append_assertion,
    effective_assertions,
    load_assertions,
)
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema
from wyrd.generators.kenning.lexicon.canonicalization_projection import project_canonical
from wyrd.generators.kenning.lexicon.gloss_sense import (
    GlossSenseCandidate,
    Sense,
    committed_sense_refs,
    parse_candidate,
    sense_assertions,
    validate_gloss_sense_candidates,
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
    db.commit()
    return eid


def _cand(ref, language, form, senses, *, confidence="high"):
    return GlossSenseCandidate(
        ref=ref,
        language=language,
        canonical_form=form,
        senses=tuple(Sense(label, tuple(gl)) for label, gl in senses),
        confidence=confidence,
        method="opus-gloss-sense-v1",
        model="opus",
    )


def test_parse_candidate_roundtrips_a_well_formed_row():
    row = {
        "_type": "gloss_sense",
        "ref": "old-english:tūn",
        "language": "old-english",
        "canonical_form": "tūn",
        "senses": [{"label": "town", "glosses": ["enclosure", "farmstead", "town"]}],
        "confidence": "high",
        "method": "opus-gloss-sense-v1",
        "model": "opus",
    }
    c = parse_candidate(row)
    assert c is not None
    assert c.ref == "old-english:tūn"
    assert len(c.senses) == 1
    assert c.senses[0].label == "town"
    assert c.all_glosses == ["enclosure", "farmstead", "town"]


def test_parse_candidate_rejects_malformed():
    assert parse_candidate({"_type": "other"}) is None
    assert parse_candidate({"_type": "gloss_sense", "ref": "x", "senses": []}) is None
    assert (
        parse_candidate(
            {"_type": "gloss_sense", "ref": "x", "senses": [{"label": "", "glosses": ["a"]}]}
        )
        is None
    )


def test_validate_accepts_grounded_complete_partition(tmp_path):
    db = _db(tmp_path)
    _etymon(db, "tūn", ["enclosure", "farmstead", "town"])
    c = _cand(
        "old-english:tūn", "old-english", "tūn", [("town", ["enclosure", "farmstead", "town"])]
    )
    assert validate_gloss_sense_candidates(db.conn, set(), [c]) == []
    db.close()


def test_validate_rejects_invented_gloss(tmp_path):
    db = _db(tmp_path)
    _etymon(db, "tūn", ["enclosure", "town"])
    c = _cand(
        "old-english:tūn", "old-english", "tūn", [("town", ["enclosure", "town", "metropolis"])]
    )
    errs = validate_gloss_sense_candidates(db.conn, set(), [c])
    assert any("invented" in e for e in errs)
    db.close()


def test_validate_rejects_incomplete_partition(tmp_path):
    db = _db(tmp_path)
    _etymon(db, "tūn", ["enclosure", "farmstead", "town"])
    c = _cand("old-english:tūn", "old-english", "tūn", [("town", ["enclosure", "town"])])
    errs = validate_gloss_sense_candidates(db.conn, set(), [c])
    assert any("unassigned" in e for e in errs)
    db.close()


def test_validate_rejects_long_label_paraphrase(tmp_path):
    db = _db(tmp_path)
    _etymon(db, "tūn", ["town"])
    c = _cand(
        "old-english:tūn", "old-english", "tūn", [("a farmstead or enclosed settlement", ["town"])]
    )
    errs = validate_gloss_sense_candidates(db.conn, set(), [c])
    assert any("short compressions" in e for e in errs)
    db.close()


def test_validate_rejects_already_committed_and_dupes(tmp_path):
    db = _db(tmp_path)
    _etymon(db, "tūn", ["town"])
    c = _cand("old-english:tūn", "old-english", "tūn", [("town", ["town"])])
    assert any(
        "already sense-bound" in e
        for e in validate_gloss_sense_candidates(db.conn, {"old-english:tūn"}, [c])
    )
    assert any("duplicated" in e for e in validate_gloss_sense_candidates(db.conn, set(), [c, c]))
    db.close()


def test_roundtrip_monosemous_wall_projects_one_sense(tmp_path):
    db = _db(tmp_path)
    _etymon(db, "tūn", ["enclosure", "farmstead", "town"])
    c = _cand(
        "old-english:tūn", "old-english", "tūn", [("town", ["enclosure", "farmstead", "town"])]
    )
    assert validate_gloss_sense_candidates(db.conn, set(), [c]) == []
    for a in sense_assertions(c):
        append_assertion(tmp_path, a)
    res = project_canonical(db, mining_dir=tmp_path, apply=True)
    assert res.minted == {"canonical_sense": 1}
    assert res.bound == {"etymon_gloss": 3}
    # Every gloss row points at the one sense hub, which carries the short label.
    hubs = {
        r["canonical_sense_id"]
        for r in db.conn.execute("SELECT canonical_sense_id FROM etymon_gloss")
    }
    assert len(hubs) == 1 and None not in hubs
    label = db.conn.execute(
        "SELECT value FROM canonical_label WHERE canonical_type='canonical_sense'"
    ).fetchone()["value"]
    assert label == "town"
    db.close()


def test_roundtrip_polysemous_wall_projects_distinct_senses(tmp_path):
    db = _db(tmp_path)
    _etymon(db, "weall", ["wall, rampart", "spring", "well, source"])
    c = _cand(
        "old-english:weall",
        "old-english",
        "weall",
        [("wall", ["wall, rampart"]), ("well", ["spring", "well, source"])],
    )
    assert validate_gloss_sense_candidates(db.conn, set(), [c]) == []
    for a in sense_assertions(c):
        append_assertion(tmp_path, a)
    res = project_canonical(db, mining_dir=tmp_path, apply=True)
    assert res.minted == {"canonical_sense": 2}
    labels = {
        r["value"]
        for r in db.conn.execute(
            "SELECT value FROM canonical_label WHERE canonical_type='canonical_sense'"
        )
    }
    assert labels == {"wall", "well"}
    db.close()


def test_committed_sense_refs_reads_the_ledger(tmp_path):
    db = _db(tmp_path)
    _etymon(db, "tūn", ["town"])
    c = _cand("old-english:tūn", "old-english", "tūn", [("town", ["town"])])
    for a in sense_assertions(c):
        append_assertion(tmp_path, a)
    done = committed_sense_refs(effective_assertions(list(load_assertions(tmp_path))))
    assert done == {"old-english:tūn"}
    db.close()
