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


def test_validate_dedups_affix_dash_ref_variant(tmp_path):
    # Regression (D45): `old-english:tun` and `old-english:-tun` resolve to the SAME
    # bare etymon (an affix-position dash is never identity). The dedup must treat
    # them as ONE morpheme; otherwise both author a canonical_sense hub for one
    # morpheme and the projection (which de-dashes) conflicts → the gloss is left
    # unbound + two orphan hubs minted.
    db = _db(tmp_path)
    _etymon(db, "tun", ["enclosure", "farmstead"])  # stored BARE
    bare = _cand(
        "old-english:tun", "old-english", "tun", [("homestead", ["enclosure", "farmstead"])]
    )
    dashed = _cand(
        "old-english:-tun", "old-english", "-tun", [("homestead", ["enclosure", "farmstead"])]
    )
    errs = validate_gloss_sense_candidates(db.conn, set(), [bare, dashed])
    # the dashed variant is the same morpheme as the bare one → in-batch duplicate
    assert any("duplicated in this batch" in e for e in errs), errs
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


def test_validate_rejects_whitespace_only_label(tmp_path):
    """A whitespace-only label ('   ') is truthy and splits to zero words, so it
    must be rejected by _label_ok's strip guard — else author would mint an empty
    canonical-label. validate is the independent gate (a hand-built candidate can
    carry a label that never passed parse_candidate's strip)."""
    db = _db(tmp_path)
    _etymon(db, "tūn", ["town"])
    c = _cand("old-english:tūn", "old-english", "tūn", [("   ", ["town"])])
    errs = validate_gloss_sense_candidates(db.conn, set(), [c])
    assert any("short compressions" in e for e in errs)
    db.close()


def test_validate_rejects_ref_inconsistent_with_form(tmp_path):
    """validate grounds on the etymon from c.ref, but sense_assertions authors the
    same-sense binds from c.language/c.canonical_form. A hand-built candidate whose
    ref disagrees with those fields must be rejected — else the binds resolve
    elsewhere at projection and the hub is minted bound to nothing."""
    db = _db(tmp_path)
    _etymon(db, "tūn", ["town"])
    # ref resolves to tūn, but canonical_form points elsewhere (lēah)
    c = _cand("old-english:tūn", "old-english", "lēah", [("town", ["town"])])
    errs = validate_gloss_sense_candidates(db.conn, set(), [c])
    assert any("disagrees with language/canonical_form" in e for e in errs)
    db.close()


def test_validate_rejects_sense_with_no_glosses(tmp_path):
    """A sense covering zero gloss rows mints a canonical_sense hub + label bound to
    nothing; reject it even when the OTHER senses cover the whole wall."""
    db = _db(tmp_path)
    _etymon(db, "tūn", ["town", "farm"])
    c = _cand(
        "old-english:tūn",
        "old-english",
        "tūn",
        [("town", ["town", "farm"]), ("empty", [])],  # 2nd sense covers nothing
    )
    errs = validate_gloss_sense_candidates(db.conn, set(), [c])
    assert any("cover no glosses" in e for e in errs)
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


def test_parse_candidate_normalizes_confidence_and_derives_fields():
    """_norm_conf coerces case + fails safe to 'low'; language/canonical_form fall
    back to the ref halves when their keys are absent."""
    up = parse_candidate(
        {
            "_type": "gloss_sense",
            "ref": "a:b",
            "senses": [{"label": "x", "glosses": ["g"]}],
            "confidence": "HIGH",
        }
    )
    assert up is not None
    assert up.confidence == "high"  # upper-cased -> lower
    assert up.language == "a" and up.canonical_form == "b"  # derived from ref
    bogus = parse_candidate(
        {
            "_type": "gloss_sense",
            "ref": "a:b",
            "senses": [{"label": "x", "glosses": ["g"]}],
            "confidence": "bogus",
        }
    )
    assert bogus is not None and bogus.confidence == "low"  # unknown -> fail-safe low


def test_parse_candidate_rejects_non_dict():
    assert parse_candidate("not a dict") is None  # type: ignore[arg-type]


def test_validate_rejects_unresolvable_ref(tmp_path):
    db = _db(tmp_path)
    c = _cand("old-english:nope", "old-english", "nope", [("x", ["g"])])
    errs = validate_gloss_sense_candidates(db.conn, set(), [c])
    assert any("resolves to no etymon" in e for e in errs)
    db.close()


def test_validate_rejects_morpheme_with_no_gloss_wall(tmp_path):
    db = _db(tmp_path)
    _etymon(db, "barf", [])  # resolves, but no gloss wall to compress
    c = _cand("old-english:barf", "old-english", "barf", [("x", ["g"])])
    errs = validate_gloss_sense_candidates(db.conn, set(), [c])
    assert any("no gloss wall" in e for e in errs)
    db.close()


def test_validate_rejects_gloss_in_two_senses(tmp_path):
    """The partition-cleanliness invariant (distinct from the completeness check):
    a gloss assigned to two senses is not a partition."""
    db = _db(tmp_path)
    _etymon(db, "tūn", ["town", "city"])
    c = _cand(
        "old-english:tūn",
        "old-english",
        "tūn",
        [("town", ["town"]), ("city", ["town", "city"])],  # 'town' in both senses
    )
    errs = validate_gloss_sense_candidates(db.conn, set(), [c])
    assert any("more than one sense" in e for e in errs)
    db.close()


def test_sense_assertions_is_idempotent():
    """Re-authoring the same candidate mints the SAME hub id and SAME gloss bind
    subjects — the projection dedups by node id, so a drifting hub id would create
    duplicate canonical_sense hubs / double-bind glosses on a re-run."""
    c = _cand("old-english:tūn", "old-english", "tūn", [("town", ["enclosure", "town"])])

    def _subjects(asserts, predicate):
        return [a.subject.ref for a in asserts if a.predicate == predicate]

    first, second = sense_assertions(c), sense_assertions(c)
    assert _subjects(first, "mint-canonical") == _subjects(second, "mint-canonical")
    assert _subjects(first, "bind") == _subjects(second, "bind")
    assert _subjects(first, "mint-canonical")  # guard: non-empty


def test_sense_assertions_hub_id_is_label_and_idx_keyed():
    """Pin the hub-identity formula (node, language, form, label, idx) the
    idempotency + reuse docstring relies on — in particular that ``idx`` is part of
    it (a polysemous re-run emitting labels in a different order lands a different
    hub). Dropping str(idx) from the mint would break this exact assertion."""
    from wyrd.generators.kenning.canonicalization.assertions import mint_canonical_id

    c = _cand(
        "old-english:weall",
        "old-english",
        "weall",
        [("wall", ["wall, rampart"]), ("well", ["spring"])],
    )
    mints = [a.subject.ref for a in sense_assertions(c) if a.predicate == "mint-canonical"]
    expected = [
        mint_canonical_id("canonical_sense", "old-english", "weall", "wall", "0"),
        mint_canonical_id("canonical_sense", "old-english", "weall", "well", "1"),
    ]
    assert mints == expected  # keyed on label AND position (idx)
    assert len(set(mints)) == 2  # distinct hubs per sense


def test_validate_rejects_already_committed_non_nfc_ref(tmp_path):
    """validate must NFC-fold the candidate ref before the committed-dedup check,
    matching committed_done (NFC). A decomposed candidate ref whose NFC form is
    already committed must be rejected — else author re-mints a second
    canonical_sense hub for the same morpheme (the diacritic-heavy OE/Welsh case)."""
    import unicodedata

    decomposed = unicodedata.normalize("NFD", "tūn")
    assert decomposed != "tūn"  # guard: genuinely non-NFC
    db = _db(tmp_path)
    _etymon(db, decomposed, ["town"])  # DB stores the decomposed form
    committed = {"old-english:tūn"}  # NFC form, as committed_done would yield
    c = _cand(f"old-english:{decomposed}", "old-english", decomposed, [("town", ["town"])])
    errs = validate_gloss_sense_candidates(db.conn, committed, [c])
    assert any("already sense-bound" in e for e in errs)
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
