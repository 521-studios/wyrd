"""Tests for the scholar-morpheme enrichment campaign rail (wyrd-eni4.1).

Builds a tiny lexicon — two real scholar morphemes (``tūn``→-ton, ``by``→-by),
one ``unknown``-language placeholder, and one multi-word phrase etymon — plus a
committed ``_reflexes.jsonl`` ledger, then exercises the two loop primitives:

* ``next_slice`` is impact-ordered, excludes the junk/phrase etymons, and uses
  the LEDGER (not the DB) as the "already has a reflex" remainder source;
* ``validate_candidates`` accepts a grounded reflex and rejects every failure
  mode: invented surface (grounding guard), unresolved ref, bad position,
  out-of-campaign-scope ref, and a duplicate of a committed reflex.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wyrd.generators.kenning.lexicon import LexiconDB, init_schema, tag_mining
from wyrd.generators.kenning.lexicon.enrichment_campaign import (
    committed_reflex_refs,
    next_slice,
    reflex_progress,
    tag_next_slice,
    tag_progress,
    validate_candidates,
    validate_tag_candidates,
)


def _etymon(db, form, *, language="old-english", gloss=None):
    eid = db.conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES (?, ?)", (form, language)
    ).lastrowid
    if gloss is not None:
        db.conn.execute("INSERT INTO etymon_gloss (etymon_id, gloss) VALUES (?, ?)", (eid, gloss))
    return eid


def _toponym(db, name, elements):
    tid = db.conn.execute(
        "INSERT INTO toponym (modern_name, country) VALUES (?, 'England')", (name,)
    ).lastrowid
    teid = db.conn.execute(
        "INSERT INTO toponym_etymology (toponym_id, source_id, confidence) "
        "VALUES (?, 'test_src', 'high')",
        (tid,),
    ).lastrowid
    for ordinal, eid in enumerate(elements):
        db.conn.execute(
            "INSERT INTO toponym_etymology_element "
            "(toponym_etymology_id, ordinal, etymon_id) VALUES (?, ?, ?)",
            (teid, ordinal, eid),
        )
    return tid


@pytest.fixture
def world(tmp_path):
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    db = LexiconDB(db_path)
    db.conn.execute("INSERT INTO source (id, title) VALUES ('test_src', 'Test')")

    tun = _etymon(db, "tūn", gloss="An enclosure; a farmstead, a village.")
    by = _etymon(db, "by", language="old-norse", gloss="Estate, farm, village.")
    placeholder = _etymon(db, "Place-name", language="unknown")  # junk → excluded
    phrase = _etymon(db, "Bartholomew County", language="modern-english")  # phrase → excluded

    # tūn appears in 3 toponyms (impact 3); by in 2 (impact 2).
    _toponym(db, "Newton", [tun])
    _toponym(db, "Thornton", [tun])
    _toponym(db, "Bishopston", [tun])
    _toponym(db, "Westby", [by])
    _toponym(db, "Irby", [by])
    _toponym(db, "Keysoe", [placeholder])
    _toponym(db, "Greatworth", [phrase])
    db.commit()
    return db, tmp_path


def _ledger(tmp_path: Path, *rows: dict) -> Path:
    path = tmp_path / "_reflexes.jsonl"
    path.write_text(
        "".join(json.dumps({"_type": "reflex", **r}) + "\n" for r in rows), encoding="utf-8"
    )
    return path


def test_next_slice_impact_ordered_and_scope_filtered(world):
    db, tmp_path = world
    empty = tmp_path / "empty.jsonl"
    slice_ = next_slice(db.conn, empty, n=10)
    refs = [e.ref for e in slice_]
    # Only the two real morphemes, tūn (impact 3) before by (impact 2).
    assert refs == ["old-english:tūn", "old-norse:by"]
    assert slice_[0].impact == 3
    # Junk + phrase etymons never surface.
    assert all("Place-name" not in r and "Bartholomew" not in r for r in refs)
    # Evidence is attached for grounding.
    assert {t["modern_name"] for t in slice_[0].toponyms} == {"Newton", "Thornton", "Bishopston"}


def test_next_slice_remainder_from_ledger_not_db(world):
    db, tmp_path = world
    # tūn already has a committed reflex → it drops out; by remains.
    ledger = _ledger(
        tmp_path, {"surface_form": "ton", "position": "post", "etymon_refs": ["old-english:tūn"]}
    )
    assert committed_reflex_refs(ledger) == {"old-english:tūn"}
    slice_ = next_slice(db.conn, ledger, n=10)
    assert [e.ref for e in slice_] == ["old-norse:by"]


def test_progress_counts_committed(world):
    db, tmp_path = world
    ledger = _ledger(
        tmp_path, {"surface_form": "ton", "position": "post", "etymon_refs": ["old-english:tūn"]}
    )
    done, parked, total = reflex_progress(db.conn, ledger)
    assert (done, parked, total) == (1, 0, 2)


def test_parked_refs_excluded_from_slice(world):
    db, tmp_path = world
    empty = tmp_path / "empty.jsonl"
    parked = tmp_path / "_reflex_parked.jsonl"
    parked.write_text(
        json.dumps({"ref": "old-english:tūn", "reason": "no confident grounded surface"}) + "\n",
        encoding="utf-8",
    )
    # tūn (top impact) is parked → only by remains in the slice.
    slice_ = next_slice(db.conn, empty, n=10, parked_path=parked)
    assert [e.ref for e in slice_] == ["old-norse:by"]
    done, parked_count, total = reflex_progress(db.conn, empty, parked_path=parked)
    assert (done, parked_count, total) == (0, 1, 2)


def test_validate_accepts_grounded_reflex(world):
    db, tmp_path = world
    empty = tmp_path / "empty.jsonl"
    errors = validate_candidates(
        db.conn,
        empty,
        [
            {
                "_type": "reflex",
                "surface_form": "ton",
                "position": "post",
                "etymon_refs": ["old-english:tūn"],
            }
        ],
    )
    assert errors == []


def test_validate_rejects_failure_modes(world):
    db, tmp_path = world
    empty = tmp_path / "empty.jsonl"
    cases = {
        "grounding": {
            "surface_form": "zzqx",
            "position": "post",
            "etymon_refs": ["old-english:tūn"],
        },
        "resolve": {"surface_form": "ton", "position": "post", "etymon_refs": ["old-english:nope"]},
        "position": {
            "surface_form": "ton",
            "position": "sideways",
            "etymon_refs": ["old-english:tūn"],
        },
        "scope": {"surface_form": "key", "position": "pre", "etymon_refs": ["unknown:Place-name"]},
    }
    for label, rec in cases.items():
        errors = validate_candidates(db.conn, empty, [{"_type": "reflex", **rec}])
        assert errors, f"expected {label!r} to fail"


def test_validate_rejects_committed_duplicate(world):
    db, tmp_path = world
    ledger = _ledger(
        tmp_path, {"surface_form": "ton", "position": "post", "etymon_refs": ["old-english:tūn"]}
    )
    errors = validate_candidates(
        db.conn,
        ledger,
        [
            {
                "_type": "reflex",
                "surface_form": "ton",
                "position": "post",
                "etymon_refs": ["old-english:tūn"],
            }
        ],
    )
    assert any("duplicate" in e for e in errors)


# --- tag uplift (phase 2) ---------------------------------------------------

_VALID_TAG = tag_mining.TAG_VOCAB[0]


def test_tag_next_slice_surfaces_glossed_untagged_admit(world):
    db, tmp_path = world
    empty = tmp_path / "_tags.jsonl"
    slice_ = tag_next_slice(db.conn, tmp_path / "lexicon.db", empty, n=10)
    refs = {t["ref"] for t in slice_}
    # tūn (3 toponyms, glossed, untagged) and by (2, glossed, untagged) qualify.
    assert refs == {"old-english:tūn", "old-norse:by"}
    assert all(t["glosses"] for t in slice_)
    assert slice_[0]["vocab"] == list(tag_mining.TAG_VOCAB)
    # Impact-ordered: tūn (3) before by (2).
    assert slice_[0]["ref"] == "old-english:tūn"


def test_tag_validate_vocab_and_none_outcome(world):
    db, tmp_path = world
    empty = tmp_path / "_tags.jsonl"
    ok = [
        {"_type": "tags", "ref": "old-english:tūn", "tags": [_VALID_TAG]},
        {"_type": "tags", "ref": "old-norse:by", "tags": []},  # 'none' outcome is valid
    ]
    assert validate_tag_candidates(db.conn, empty, ok) == []
    bad_vocab = [{"_type": "tags", "ref": "old-english:tūn", "tags": ["not-a-real-tag"]}]
    assert any("vocabulary" in e for e in validate_tag_candidates(db.conn, empty, bad_vocab))
    bad_ref = [{"_type": "tags", "ref": "old-english:nope", "tags": [_VALID_TAG]}]
    assert any(
        "resolves to no etymon" in e for e in validate_tag_candidates(db.conn, empty, bad_ref)
    )


def test_tag_validate_rejects_committed_duplicate(world):
    db, tmp_path = world
    tags = tmp_path / "_tags.jsonl"
    tags.write_text(
        json.dumps({"_type": "tags", "ref": "old-english:tūn", "tags": [_VALID_TAG]}) + "\n",
        encoding="utf-8",
    )
    errors = validate_tag_candidates(
        db.conn, tags, [{"_type": "tags", "ref": "old-english:tūn", "tags": [_VALID_TAG]}]
    )
    assert any("duplicate" in e for e in errors)
    # And the committed ref drops out of the next slice.
    slice_ = tag_next_slice(db.conn, tmp_path / "lexicon.db", tags, n=10)
    assert "old-english:tūn" not in {t["ref"] for t in slice_}
    assert tag_progress(db.conn, tmp_path / "lexicon.db", tags) == 1  # only 'by' remains
