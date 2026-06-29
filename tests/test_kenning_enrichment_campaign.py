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
    bands_status,
    committed_reflex_refs,
    gloss_next_slice,
    identity_reflex_rows,
    ipa_next_slice,
    next_slice,
    reflex_progress,
    scholar_impact_ranking,
    tag_next_slice,
    tag_progress,
    validate_candidates,
    validate_collapse_candidates,
    validate_gloss_candidates,
    validate_ipa_candidates,
    validate_tag_candidates,
    variant_gap_census,
    variant_gap_next_slice,
)


def _etymon(db, form, *, language="old-english", gloss=None, merged_into=None):
    eid = db.conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES (?, ?)", (form, language)
    ).lastrowid
    if gloss is not None:
        db.conn.execute("INSERT INTO etymon_gloss (etymon_id, gloss) VALUES (?, ?)", (eid, gloss))
    if merged_into is not None:
        db.conn.execute("UPDATE etymon SET merged_into_id = ? WHERE id = ?", (merged_into, eid))
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


def test_variant_gap_census_classifies_pairs(tmp_path):
    """A scholar pair is matched (a reflex substring-matches this toponym),
    variant-gap (has a reflex, none match this spelling), or reflex-less."""
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    db = LexiconDB(db_path)
    db.conn.execute("INSERT INTO source (id, title) VALUES ('test_src', 'Test')")
    tun = _etymon(db, "tūn")
    bare = _etymon(db, "zzz")  # no reflex → reflex-less
    rid = db.conn.execute(
        "INSERT INTO reflex (surface_form, position, productivity) VALUES ('ton', 'post', 1)"
    ).lastrowid
    db.conn.execute("INSERT INTO reflex_etymon (reflex_id, etymon_id) VALUES (?, ?)", (rid, tun))
    _toponym(db, "Newton", [tun])  # 'ton' ⊂ 'newton' → matched
    _toponym(db, "Tunbridge", [tun])  # reflex 'ton' ⊄ 'tunbridge' → variant-gap
    _toponym(db, "Zzzham", [bare])  # no reflex → reflex-less
    db.commit()

    c = variant_gap_census(db.conn)
    assert c["total"] == 3
    assert (c["matched"], c["variant_gap"], c["reflex_less"]) == (1, 1, 1)


def test_variant_gap_next_slice_selects_and_ledger_excludes(tmp_path):
    """Selects variant-gap morphemes (have a reflex, none matching this toponym),
    flags flip-CAN-IT (sole unmatched morpheme), and drains once the committed
    ledger gives the morpheme a surface that matches."""
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    db = LexiconDB(db_path)
    db.conn.execute("INSERT INTO source (id, title) VALUES ('test_src', 'Test')")
    tun = _etymon(db, "tūn", gloss="enclosure")
    by = _etymon(db, "by", language="old-norse", gloss="farm")
    for eid, surf in ((tun, "ton"), (by, "by")):
        rid = db.conn.execute(
            "INSERT INTO reflex (surface_form, position, productivity) VALUES (?, 'post', 1)",
            (surf,),
        ).lastrowid
        db.conn.execute(
            "INSERT INTO reflex_etymon (reflex_id, etymon_id) VALUES (?, ?)", (rid, eid)
        )
    _toponym(db, "Tunby", [tun, by])  # 'ton'⊄'tunby' (tun gap), 'by'⊂'tunby' (by matched)
    _toponym(db, "Newton", [tun])  # 'ton'⊂'newton' → tun matched here, no gap
    db.commit()

    empty = tmp_path / "_reflexes.jsonl"
    tasks = variant_gap_next_slice(db.conn, empty, n=10)
    assert {t.ref for t in tasks} == {"old-english:tūn"}  # only tun is variant-gap
    t = tasks[0]
    assert t.flips_can_it is True  # tun is the sole unmatched morpheme in Tunby
    assert t.gap_toponyms == 1
    assert t.evidence[0]["modern_name"] == "Tunby"
    assert "tun" in t.evidence[0]["residual_span"].lower()

    # A parked ref is excluded so an un-authorable morpheme can't re-surface.
    parked = tmp_path / "_reflex_parked.jsonl"
    parked.write_text(
        json.dumps({"ref": "old-english:tūn", "reason": "x"}) + "\n", encoding="utf-8"
    )
    assert variant_gap_next_slice(db.conn, empty, n=10, parked_path=parked) == []

    # Commit a 'tun' surface to the ledger → tun now matches 'tunby' → slice drains.
    empty.write_text(
        json.dumps(
            {
                "_type": "reflex",
                "surface_form": "tun",
                "position": "pre",
                "etymon_refs": ["old-english:tūn"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert variant_gap_next_slice(db.conn, empty, n=10) == []


def test_variant_gap_authored_reflex_validates(tmp_path):
    """End-to-end Rail A: a variant-gap morpheme's worn span, authored as a reflex,
    passes validate_candidates (the span is in the toponym's modern name); an
    off-name span is rejected by the same grounding gate. No new authoring code —
    the path reuses validate_candidates / `enrich-campaign validate`."""
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    db = LexiconDB(db_path)
    db.conn.execute("INSERT INTO source (id, title) VALUES ('test_src', 'Test')")
    dun = _etymon(db, "dūn", gloss="hill")
    rid = db.conn.execute(
        "INSERT INTO reflex (surface_form, position, productivity) VALUES ('don', 'post', 1)"
    ).lastrowid
    db.conn.execute("INSERT INTO reflex_etymon (reflex_id, etymon_id) VALUES (?, ?)", (rid, dun))
    _toponym(db, "Battlesden", [dun])  # 'don' ⊄ 'battlesden' → dūn variant-gap, worn span 'den'
    db.commit()

    empty = tmp_path / "_reflexes.jsonl"
    assert "old-english:dūn" in {t.ref for t in variant_gap_next_slice(db.conn, empty, n=10)}

    # Author the worn span 'den' → grounded in 'battlesden' → accepted.
    ok = [
        {
            "_type": "reflex",
            "surface_form": "den",
            "position": "post",
            "etymon_refs": ["old-english:dūn"],
        }
    ]
    assert validate_candidates(db.conn, empty, ok) == []
    # An off-name span is refused by the same grounding gate.
    bad = [
        {
            "_type": "reflex",
            "surface_form": "xyz",
            "position": "post",
            "etymon_refs": ["old-english:dūn"],
        }
    ]
    assert any("grounding guard" in e for e in validate_candidates(db.conn, empty, bad))


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


def test_parked_ref_excluded_across_unicode_normalization(tmp_path):
    """A parked ref written NFC (precomposed ō) must still exclude a DB etymon
    stored NFD (o + combining macron) — same morpheme, different bytes."""
    import unicodedata

    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    db = LexiconDB(db_path)
    db.conn.execute("INSERT INTO source (id, title) VALUES ('test_src', 'Test')")
    nfd = unicodedata.normalize("NFD", "Bōfa")  # stored as o + U+0304
    eid = _etymon(db, nfd, language="old-english")
    _toponym(db, "Bowcombe", [eid])
    _toponym(db, "Coveney", [eid])  # impact 2 → in scope
    db.commit()

    empty = tmp_path / "empty.jsonl"
    assert [e.canonical_form for e in next_slice(db.conn, empty, n=10)] == [nfd]
    parked = tmp_path / "_reflex_parked.jsonl"
    nfc = unicodedata.normalize("NFC", "Bōfa")  # parked as precomposed ō
    assert nfc != nfd
    parked.write_text(
        json.dumps({"ref": f"old-english:{nfc}", "reason": "x"}) + "\n", encoding="utf-8"
    )
    assert next_slice(db.conn, empty, n=10, parked_path=parked) == []


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


def test_tag_cohort_excludes_merged_tombstone(world):
    """wyrd-tze2: a bare-surface stub folded into its canonical (merged_into_id
    set) is a tombstone and must NOT surface in the tag cohort — else a later
    slice re-tags the tombstone, polluting downstream consumers that read
    etymon_tag as live (the #804 / wyrd-ckro harm). Without the
    ``merged_into_id IS NULL`` filter this stub qualifies (glossed, untagged,
    impact 2) and the assertion fails — i.e. the filter line is load-bearing."""
    db, tmp_path = world
    tun = db.conn.execute("SELECT id FROM etymon WHERE canonical_form = 'tūn'").fetchone()[0]
    stub = _etymon(db, "ton", gloss="A worn form of tūn.", merged_into=tun)
    _toponym(db, "Acton", [stub])
    _toponym(db, "Bolton", [stub])  # impact 2 — would otherwise admit
    db.conn.commit()
    empty = tmp_path / "_tags.jsonl"
    refs = {t["ref"] for t in tag_next_slice(db.conn, tmp_path / "lexicon.db", empty, n=10)}
    assert "old-english:ton" not in refs
    # the live etymons still surface — the filter is narrow, not a blanket drop.
    assert refs == {"old-english:tūn", "old-norse:by"}
    # The same merged_into_id exclusion guards scholar_impact_ranking
    # (_IMPACT_SQL), which drives the reflex / variant-gap / band cohorts too
    # (wyrd-tze2, Gemini round 2): the tombstoned stub is absent there as well.
    impact_forms = {r["canonical_form"] for r in scholar_impact_ranking(db.conn)}
    assert "ton" not in impact_forms
    assert "tūn" in impact_forms


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


def test_tag_done_set_matches_across_unicode_normalization(world):
    """A tag decision committed in one Unicode form must mark the etymon done even
    when the DB ref folds to the other (NFC/NFD): the tag ``done``/``tag_led`` sets
    are NFC-normalized like every other dimension. Without it, diacritic-heavy
    etymons (tūn, bōfa, …) re-surface forever (Gemini, wyrd-eni4.3)."""
    import unicodedata

    db, tmp_path = world
    tags = tmp_path / "_tags.jsonl"
    nfd_ref = "old-english:" + unicodedata.normalize("NFD", "tūn")
    assert nfd_ref != "old-english:tūn"  # genuinely byte-different forms
    tags.write_text(
        json.dumps({"_type": "tags", "ref": nfd_ref, "tags": [_VALID_TAG]}) + "\n",
        encoding="utf-8",
    )
    # tūn is committed (in NFD) → drops from the slice + progress, and an NFC
    # re-decision reads as a duplicate.
    slice_refs = {t["ref"] for t in tag_next_slice(db.conn, tmp_path / "lexicon.db", tags, n=10)}
    assert "old-english:tūn" not in slice_refs
    assert tag_progress(db.conn, tmp_path / "lexicon.db", tags) == 1
    dup = validate_tag_candidates(
        db.conn, tags, [{"_type": "tags", "ref": "old-english:tūn", "tags": [_VALID_TAG]}]
    )
    assert any("duplicate" in e for e in dup)


# --- band-by-band holistic loop (wyrd-eni4.3) -------------------------------


def test_next_slice_impact_band_filter(world):
    db, tmp_path = world
    empty = tmp_path / "empty.jsonl"
    # tūn has impact 3, by impact 2 — band filter isolates each.
    assert [e.ref for e in next_slice(db.conn, empty, n=10, impact=3)] == ["old-english:tūn"]
    assert [e.ref for e in next_slice(db.conn, empty, n=10, impact=2)] == ["old-norse:by"]


def test_ipa_next_slice_and_validate(world):
    db, tmp_path = world
    empty = tmp_path / "_pronunciation.jsonl"
    # neither tūn nor by has pronunciation_ipa → both surface, glosses attached.
    refs = {t["ref"] for t in ipa_next_slice(db.conn, empty, n=10)}
    assert refs == {"old-english:tūn", "old-norse:by"}
    ok = [{"_type": "pronunciation", "ref": "old-english:tūn", "ipa": "/tuːn/"}]
    assert validate_ipa_candidates(db.conn, empty, ok) == []
    bad = [
        {"_type": "pronunciation", "ref": "old-english:tūn", "ipa": "tuun"},  # not delimited
        {"_type": "pronunciation", "ref": "old-english:nope", "ipa": "/x/"},  # bad ref
    ]
    assert len(validate_ipa_candidates(db.conn, empty, bad)) == 2


def test_gloss_next_slice_and_validate(tmp_path):
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    db = LexiconDB(db_path)
    db.conn.execute("INSERT INTO source (id, title) VALUES ('test_src', 'Test')")
    glossed = _etymon(db, "tūn", gloss="enclosure")
    unglossed = _etymon(db, "strēt", language="old-english")  # impact 2, no gloss
    _toponym(db, "Newton", [glossed])
    _toponym(db, "Thornton", [glossed])
    _toponym(db, "Stratford", [unglossed])
    _toponym(db, "Stretton", [unglossed])
    db.commit()
    empty = tmp_path / "_curation.jsonl"
    refs = {t["ref"] for t in gloss_next_slice(db.conn, empty, n=10)}
    assert refs == {"old-english:strēt"}  # only the unglossed one
    ok = [
        {
            "_type": "etymon_gloss_add",
            "ref": "old-english:strēt",
            "additions": [{"gloss": "a Roman road"}],
        }
    ]
    assert validate_gloss_candidates(db.conn, empty, ok) == []
    bad = [{"_type": "etymon_gloss_add", "ref": "old-english:strēt", "additions": []}]
    assert validate_gloss_candidates(db.conn, empty, bad)


def test_bands_status_isolates_dimensions(world):
    db, tmp_path = world
    e = tmp_path / "e.jsonl"
    # nothing committed in any ledger → both etymons (impact 3 and 2) need
    # reflex+ipa (no tag: untagged+unglossed-in-vocab still counts as tag-missing).
    bands = bands_status(
        db.conn, reflexes_path=e, parked_path=e, tags_path=e, ipa_path=e, gloss_path=e
    )
    by_impact = {b["impact"]: b for b in bands}
    assert set(by_impact) == {3, 2}
    assert by_impact[3]["reflex"] == 1 and by_impact[3]["ipa"] == 1
    assert by_impact[3]["gloss"] == 0  # tūn is glossed in the fixture
    # committing tūn's reflex clears the reflex gap for band 3.
    e.write_text(
        json.dumps(
            {
                "_type": "reflex",
                "surface_form": "ton",
                "position": "post",
                "etymon_refs": ["old-english:tūn"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    bands2 = {
        b["impact"]: b
        for b in bands_status(
            db.conn,
            reflexes_path=e,
            parked_path=tmp_path / "x",
            tags_path=tmp_path / "x",
            ipa_path=tmp_path / "x",
            gloss_path=tmp_path / "x",
        )
    }
    assert bands2[3]["reflex"] == 0


def test_validate_collapse_gate(world):
    """Garbled OCR-dupe → correct-twin merge passes; '#'-sense suffixes, self-
    merges, and unresolvable twins are rejected (wyrd-eni4.3.2)."""
    db, tmp_path = world
    # add a garbled OCR-dupe of tūn ("tvn") so a real twin exists
    db.conn.execute("INSERT INTO etymon (canonical_form, language) VALUES ('tvn', 'old-english')")
    db.commit()
    empty = tmp_path / "_curation.jsonl"
    ok = [{"_type": "collapse", "ref": "old-english:tvn", "into": "old-english:tūn"}]
    assert validate_collapse_candidates(db.conn, empty, ok) == []
    bad = [
        {
            "_type": "collapse",
            "ref": "old-english:bol#hill",
            "into": "old-english:tūn",
        },  # unresolvable ref
        {"_type": "collapse", "ref": "old-english:tūn", "into": "old-english:tūn"},  # self
        {"_type": "collapse", "ref": "old-english:tvn", "into": "old-english:nope"},  # no twin
    ]
    assert len(validate_collapse_candidates(db.conn, empty, bad)) == 3


def test_validate_collapse_rejects_sense_suffix_in_into(world):
    """The '#' sense-suffix guard fires when it's the resolvable side's ``into``
    that carries the suffix. (The ref-side guard is unreachable on its own: an
    unresolvable '#' ref short-circuits at the resolve check first — pr-test.)"""
    db, tmp_path = world
    empty = tmp_path / "_curation.jsonl"
    cand = [{"_type": "collapse", "ref": "old-english:tūn", "into": "old-english:bol#hill"}]
    errors = validate_collapse_candidates(db.conn, empty, cand)
    assert any("'#' present" in e for e in errors), errors


def test_cmd_park_validates_ref_and_dedups(world):
    """``enrich-campaign park`` rejects a ref that resolves to no etymon (exit 1,
    nothing written) and skips a no-op re-park of an already-listed ref."""
    from click.testing import CliRunner

    from wyrd.generators.kenning.cli.lexicon.enrich_campaign import enrich_campaign

    db, tmp_path = world
    parked = tmp_path / "_reflex_parked.jsonl"
    runner = CliRunner()
    base = ["park", "--db", str(db.path), "--parked-path", str(parked)]

    bogus = runner.invoke(enrich_campaign, [*base, "--ref", "old-english:nope", "--reason", "x"])
    assert bogus.exit_code == 1
    assert not parked.exists()

    first = runner.invoke(
        enrich_campaign, [*base, "--ref", "old-english:tūn", "--reason", "no groundable span"]
    )
    assert first.exit_code == 0, first.output
    assert len(parked.read_text().splitlines()) == 1

    again = runner.invoke(enrich_campaign, [*base, "--ref", "old-english:tūn", "--reason", "again"])
    assert again.exit_code == 0, again.output
    assert len(parked.read_text().splitlines()) == 1  # deduped, not appended


def test_identity_reflexes_grounded_only(world):
    """An identity reflex is emitted only when the morpheme's folded canonical
    actually appears in one of its toponyms (wyrd-eni4.3). 'by' appears in
    Westby/Irby (→ grounded, post); 'tun' (folded tūn) does NOT appear in
    Newton/Thornton/Bishopston (anglicized to -ton) → skipped."""
    db, tmp_path = world
    empty = tmp_path / "empty.jsonl"
    rows = identity_reflex_rows(db.conn, empty)
    by_rows = [r for r in rows if r["etymon_refs"] == ["old-norse:by"]]
    assert len(by_rows) == 1 and by_rows[0]["surface_form"] == "by"
    assert by_rows[0]["position"] == "post"  # Westby/Irby end with 'by'
    # tūn's folded canonical 'tun' isn't in any of its (anglicized) toponyms.
    assert not any(r["etymon_refs"] == ["old-english:tūn"] for r in rows)
    # Dedup: once 'by/post' is committed, it's not re-emitted.
    committed = tmp_path / "committed.jsonl"
    committed.write_text(
        json.dumps(
            {
                "_type": "reflex",
                "surface_form": "by",
                "position": "post",
                "etymon_refs": ["old-norse:by"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert not any(
        r["etymon_refs"] == ["old-norse:by"] for r in identity_reflex_rows(db.conn, committed)
    )


def test_identity_reflexes_require_gloss_gate(tmp_path):
    """The gloss gate (default on) excludes UNGLOSSED morphemes — a grounded
    identity reflex is only blanket-asserted for morphemes we've confirmed are
    real (gloss = that confirmation). --no-require-gloss includes them."""
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    db = LexiconDB(db_path)
    db.conn.execute("INSERT INTO source (id, title) VALUES ('test_src', 'Test')")
    glossed = _etymon(db, "ham", gloss="a homestead")  # canonical 'ham' grounds + glossed
    unglossed = _etymon(db, "wīc")  # canonical 'wic' grounds but NO gloss
    _toponym(db, "Hampton", [glossed])  # 'ham' in 'hampton'
    _toponym(db, "Wicford", [unglossed])  # 'wic' in 'wicford'
    db.commit()
    empty = tmp_path / "empty.jsonl"
    # default require_gloss=True → only the glossed morpheme
    refs_gated = {r["etymon_refs"][0] for r in identity_reflex_rows(db.conn, empty)}
    assert refs_gated == {"old-english:ham"}
    # opt out → the unglossed grounded morpheme is included too
    refs_all = {
        r["etymon_refs"][0] for r in identity_reflex_rows(db.conn, empty, require_gloss=False)
    }
    assert refs_all == {"old-english:ham", "old-english:wīc"}
