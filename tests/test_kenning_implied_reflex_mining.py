"""Tests for implied-reflex residual attribution (wyrd-65jh).

A multi-element toponym breakdown maps a modern name to its etymon elements; when
every element BUT ONE is covered by a KNOWN reflex surface, the anchored elements
tile a prefix + suffix of the modern name and leave one contiguous residual span
that is an IMPLIED reflex of the remaining element (Houghton − ton(tūn) ⇒ hough is
an implied reflex of hōh).

Guards: position-order only (a head-first/ordinal-reversed breakdown still aligns
by surface position); single residual (ambiguous → skip, D46); noise/short residual
→ skip; already-known residual → skip; support tiers; idempotency; the
dump→rebuild round-trip; determinism.
"""

from __future__ import annotations

from wyrd.generators.kenning.canonicalization import mint_canonical_id
from wyrd.generators.kenning.jsonl.build import build_from_jsonl
from wyrd.generators.kenning.jsonl.dump import dump_reflexes_to_file, dump_source_to_file
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema
from wyrd.generators.kenning.lexicon.implied_reflex_mining import (
    apply_implied_reflexes,
    extract_implied_reflexes,
    implied_reflex_assertions,
)


def _db(tmp_path, name="lexicon.db"):
    init_schema(tmp_path / name)
    return LexiconDB(tmp_path / name)


def _etymon(db, form, *, language="old-english", cognate=None):
    eid = db.conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES (?, ?)", (form, language)
    ).lastrowid
    db.conn.execute("UPDATE etymon SET cognate_id = ? WHERE id = ?", (cognate or eid, eid))
    return eid


def _reflex(db, surface, position, etymon_id):
    """Make ``surface`` a KNOWN reflex of ``etymon_id`` (an anchor surface)."""
    rid = db.upsert_reflex(surface, position)
    db.link_reflex_etymon(rid, etymon_id)


def _breakdown(db, modern_name, elements, *, region=None):
    """Insert a toponym + one breakdown with ``elements`` = [(etymon_id, ordinal)]."""
    db.conn.execute("INSERT OR IGNORE INTO source (id, title) VALUES ('topo', 'topo')")
    tid = db.conn.execute(
        "INSERT INTO toponym (modern_name, country, region) VALUES (?, 'England', ?)",
        (modern_name, region),
    ).lastrowid
    teid = db.conn.execute(
        "INSERT INTO toponym_etymology (toponym_id, source_id, confidence) "
        "VALUES (?, 'topo', 'high')",
        (tid,),
    ).lastrowid
    for etymon_id, ordinal in elements:
        db.conn.execute(
            "INSERT INTO toponym_etymology_element "
            "(toponym_etymology_id, ordinal, etymon_id) VALUES (?, ?, ?)",
            (teid, ordinal, etymon_id),
        )
    return tid, teid


# --- core alignment --------------------------------------------------------


def test_houghton_residual_attribution(tmp_path):
    db = _db(tmp_path)
    hoh = _etymon(db, "hōh")  # the residual element — NO known reflex
    tun = _etymon(db, "tūn")  # anchor — known reflex "ton"
    _reflex(db, "ton", "post", tun)
    _breakdown(db, "Houghton", [(hoh, 0), (tun, 1)])
    db.commit()

    out = extract_implied_reflexes(db)
    assert len(out) == 1
    r = out[0]
    assert (r.language, r.canonical_form) == ("old-english", "hōh")
    assert r.residual == "hough"
    assert r.position == "pre"  # head-anchored
    assert r.support == 1


def test_winterborne_residual_attribution(tmp_path):
    db = _db(tmp_path)
    winter = _etymon(db, "winter")  # anchor — known reflex "winter"
    burna = _etymon(db, "burna")  # residual element
    _reflex(db, "winter", "pre", winter)
    _breakdown(db, "Winterborne", [(winter, 0), (burna, 1)])
    db.commit()

    out = extract_implied_reflexes(db)
    assert len(out) == 1
    r = out[0]
    assert (r.language, r.canonical_form) == ("old-english", "burna")
    assert r.residual == "borne"
    assert r.position == "post"  # tail residual (winter anchors the head)


def test_position_aware_ordinal_reversed_still_aligns(tmp_path):
    """The anchor is stored HEAD-FIRST in ordinal but surfaces at the TAIL — the
    tiler must use surface position, NOT the db ordinal (which mismatches ~40%)."""
    db = _db(tmp_path)
    tun = _etymon(db, "tūn")  # anchor "ton" surfaces at the TAIL...
    hoh = _etymon(db, "hōh")  # residual
    _reflex(db, "ton", "post", tun)
    # ...but is stored as ordinal 0 (head-first), residual stored ordinal 1.
    _breakdown(db, "Houghton", [(tun, 0), (hoh, 1)])
    db.commit()

    out = extract_implied_reflexes(db)
    assert len(out) == 1
    assert out[0].residual == "hough"
    assert out[0].canonical_form == "hōh"


# --- guards ----------------------------------------------------------------


def test_ambiguous_two_valid_tilings_skipped(tmp_path):
    """If BOTH residual choices yield a valid tiling, the breakdown is ambiguous
    (D46 leave-separate) — emit nothing."""
    db = _db(tmp_path)
    # Two anchors that each have a known reflex matching opposite ends, plus a
    # name where either could be the residual.
    aa = _etymon(db, "aaa")
    bb = _etymon(db, "bbb")
    _reflex(db, "ab", "pre", aa)  # "ab" can tile the head
    _reflex(db, "cd", "post", aa)  # "cd" can tile the tail (same cluster aa)
    _reflex(db, "ab", "pre", bb)  # ...and also bb (homograph) — ambiguous anchors
    _reflex(db, "cd", "post", bb)
    _breakdown(db, "Abxxcd", [(aa, 0), (bb, 1)])
    db.commit()

    out = extract_implied_reflexes(db)
    assert out == []


def test_noise_residual_skipped(tmp_path):
    """A multi-token modern name (whitespace) is skipped at load (v1 single-token)."""
    db = _db(tmp_path)
    tun = _etymon(db, "tūn")
    hoh = _etymon(db, "hōh")
    _reflex(db, "ton", "post", tun)
    _breakdown(db, "Hough ton", [(hoh, 0), (tun, 1)])  # space → not single-token
    db.commit()
    assert extract_implied_reflexes(db) == []


def test_one_char_residual_skipped(tmp_path):
    """A 1-char residual is too noisy to attribute."""
    db = _db(tmp_path)
    tun = _etymon(db, "tūn")
    x = _etymon(db, "x")
    _reflex(db, "ton", "post", tun)
    _breakdown(db, "Xton", [(x, 0), (tun, 1)])  # residual "x" len 1
    db.commit()
    assert extract_implied_reflexes(db) == []


def test_already_known_reflex_residual_skipped(tmp_path):
    """If the residual span already equals a KNOWN reflex surface of a breakdown
    cluster, it's a homograph / already-handled surface — don't re-attribute."""
    db = _db(tmp_path)
    tun = _etymon(db, "tūn")
    hoh = _etymon(db, "hōh")
    _reflex(db, "ton", "post", tun)
    _reflex(db, "hough", "pre", hoh)  # "hough" is ALREADY a known reflex of hōh
    _breakdown(db, "Houghton", [(hoh, 0), (tun, 1)])
    db.commit()
    assert extract_implied_reflexes(db) == []


# --- boundary-leak (false-positive) regression ----------------------------


def test_boundary_leak_maximal_anchor_no_nton(tmp_path):
    """Downton = dūn + tūn. The dūn anchor can match the SHORT 'dow'/'do' OR the
    FULL 'down'; a short match would leak 'n' into the residual → 'nton'←tūn (a
    known false positive). Maximal-anchor tiling must pick 'down', yielding the
    correct 'ton'... but 'ton' is a known reflex of tūn (the residual element), so
    the already-known guard skips it — and crucially we NEVER mint 'nton'←tūn."""
    db = _db(tmp_path)
    dun = _etymon(db, "dūn")  # anchor — surfaces 'do', 'dow', 'down'
    tun = _etymon(db, "tūn")  # residual element — known reflex 'ton'
    _reflex(db, "do", "pre", dun)
    _reflex(db, "dow", "pre", dun)
    _reflex(db, "down", "pre", dun)
    _reflex(db, "ton", "post", tun)
    _breakdown(db, "Downton", [(dun, 0), (tun, 1)])
    db.commit()

    out = extract_implied_reflexes(db)
    # The maximal anchor 'down' leaves 'ton', which equals a known reflex of tūn →
    # skipped (homograph guard). The buggy short-anchor 'nton' is never produced.
    assert all(r.residual != "nton" for r in out)
    assert out == []  # nothing minted for this breakdown


def test_boundary_leak_residual_recovers_true_reflex(tmp_path):
    """When the residual element has NO known reflex, maximal-anchor tiling still
    consumes the boundary char so the residual is the TRUE span, not <leak>+span.
    Here 'down' (anchor) + residual element 'hōh' over 'Downhough' → 'hough', not
    'nhough' (hōh folds to 'hoh' ≠ 'hough', so the homograph guard doesn't fire)."""
    db = _db(tmp_path)
    dun = _etymon(db, "dūn")
    hoh = _etymon(db, "hōh")  # residual element — NO known reflex; folds to 'hoh'
    _reflex(db, "do", "pre", dun)
    _reflex(db, "dow", "pre", dun)
    _reflex(db, "down", "pre", dun)
    _breakdown(db, "Downhough", [(dun, 0), (hoh, 1)])
    db.commit()

    out = extract_implied_reflexes(db)
    assert len(out) == 1
    assert out[0].canonical_form == "hōh"
    assert out[0].residual == "hough"  # NOT 'nhough' — anchor consumed the boundary 'n'


def test_known_reflex_leftover_guard_rejects_leak(tmp_path):
    """Belt-and-suspenders: even if maximal tiling can't consume the boundary
    (anchor lacks the long surface), the known-reflex-leftover guard rejects a
    residual that is <extra-char> + a known reflex of the residual element. dūn here
    only has 'dow' (no 'down'), so 'nton' would be the residual — but 'ton' is a
    known reflex of tūn and a suffix of 'nton' with leftover 'n' → REJECT, never
    'nton'←tūn."""
    db = _db(tmp_path)
    dun = _etymon(db, "dūn")  # anchor — only short 'dow' (cannot consume the 'n')
    tun = _etymon(db, "tūn")  # residual element — known reflex 'ton'
    _reflex(db, "dow", "pre", dun)
    _reflex(db, "ton", "post", tun)
    _breakdown(db, "Downton", [(dun, 0), (tun, 1)])
    db.commit()

    out = extract_implied_reflexes(db)
    assert all(r.residual != "nton" for r in out)
    assert out == []


def test_over_consume_known_reflex_superstring_rejected(tmp_path):
    """Mirror of the under-consume guard. Acton = āc + tūn. The āc anchor can match
    the SHORT 'ac' OR the OVER-long 'act' (stealing the 't' that belongs to tūn's
    true reflex 'ton'). If 'act' wins, the residual is 'on' — a PROPER SUFFIX of the
    known reflex 'ton' of tūn (the residual element), i.e. 'ton' over-consumed by the
    anchor. That's the over-consume leak → REJECT, never 'on'←tūn. The clean outcome
    is 'ton'←tūn (already-known guard) or nothing — never the fragment 'on'."""
    db = _db(tmp_path)
    ac = _etymon(db, "āc")  # anchor — surfaces 'ac' AND the over-long 'act'
    tun = _etymon(db, "tūn")  # residual element — known reflex 'ton'
    _reflex(db, "ac", "pre", ac)
    _reflex(db, "act", "pre", ac)
    _reflex(db, "ton", "post", tun)
    _breakdown(db, "Acton", [(ac, 0), (tun, 1)])
    db.commit()

    out = extract_implied_reflexes(db)
    # 'act' anchor → residual 'on' is a proper suffix of known 'ton' → rejected.
    # 'ac' anchor → residual 'ton' equals a known reflex of tūn → already-known skip.
    assert all(r.residual != "on" for r in out)
    assert out == []  # nothing minted — never the over-consumed fragment 'on'


def test_over_consume_pre_residual_superstring_rejected(tmp_path):
    """Over-consume on the PRE side: a tail anchor over-consumes the boundary char
    that belonged to the head residual element's true reflex. The residual is then a
    proper PREFIX of a longer known reflex of that residual element → REJECT."""
    db = _db(tmp_path)
    head = _etymon(db, "headword")  # residual element — known reflex 'down'
    tun = _etymon(db, "tūn")  # tail anchor — over-long 'nton' steals the 'n'
    _reflex(db, "down", "pre", head)  # the TRUE longer reflex of the head element
    _reflex(db, "nton", "post", tun)  # over-long tail anchor
    _reflex(db, "ton", "post", tun)
    _breakdown(db, "Downton", [(head, 0), (tun, 1)])
    db.commit()

    out = extract_implied_reflexes(db)
    # Maximal tiling: 'nton' (4) beats 'ton' (3) → residual 'dow', a proper prefix of
    # the known 'down' of the head element → over-consume leak → rejected.
    assert all(r.residual != "dow" for r in out)
    assert out == []


def test_over_consume_rejected_via_cross_cluster_cognate(tmp_path):
    """wyrd-65jh: the over-consume guard must consult the COGNATE-EQUIVALENT cluster
    union, not just the residual element's own (impoverished singleton) cluster.

    The residual element is an impoverished old-english ``Tun`` whose ONLY known
    surface is its own folded canonical form ``tun`` — it does NOT know the longer
    reflex ``ton``. That ``ton`` lives in a SEPARATE old-english cognate ``tūn``
    cluster (the macron/ASCII split). Acton = āc + Tun; the āc anchor over-consumes
    to ``act`` (stealing the ``t``), leaving residual ``on``. With only the element's
    own cluster (``{tun}``), ``on`` is not a proper affix of ``tun`` → the leak
    survives as ``on``←Tun. The cross-cluster expansion lets ``Tun`` inherit ``ton``
    from its cognate ``tūn`` → ``on`` is a proper SUFFIX of ``ton`` → REJECT."""
    db = _db(tmp_path)
    ac = _etymon(db, "āc")  # anchor — surfaces 'ac' AND the over-long 'act'
    # Residual element: impoverished singleton — distinct cluster, knows only 'tun'
    # (its own folded canonical form; no reflex 'ton' linked to IT).
    tun_ascii = _etymon(db, "Tun", cognate=None)
    # Its cognate twin in a SEPARATE cluster: same (language, fold) key, knows 'ton'.
    tun_macron = _etymon(db, "tūn")
    _reflex(db, "ac", "pre", ac)
    _reflex(db, "act", "pre", ac)
    _reflex(db, "ton", "post", tun_macron)  # 'ton' is known ONLY on the cognate twin
    _breakdown(db, "Acton", [(ac, 0), (tun_ascii, 1)])
    db.commit()

    # Sanity: the residual element's OWN cluster does NOT know 'ton' — only the
    # cross-cluster cognate expansion supplies it.
    own = db.conn.execute(
        "SELECT 1 FROM reflex r JOIN reflex_etymon re ON re.reflex_id=r.id "
        "WHERE re.etymon_id=? AND r.surface_form='ton'",
        (tun_ascii,),
    ).fetchone()
    assert own is None

    out = extract_implied_reflexes(db)
    # 'act' anchor → residual 'on'; without the cross-cluster guard this leaks as
    # 'on'←Tun (support 1). With it, 'on' is a proper suffix of the inherited 'ton'
    # → rejected. 'ac' anchor → residual 'ton' (== inherited known surface) is also
    # filtered. Nothing minted.
    assert all(r.residual != "on" for r in out)
    assert out == []


def test_singleton_cluster_cognate_uses_cognate_id_grouping(tmp_path):
    """The cross-cluster expansion must NOT over-reject: a residual element whose
    cognate twin is a genuinely DIFFERENT morpheme (different fold key) does not
    inherit that twin's surfaces. Here the residual element 'hōh' (folds 'hoh') has
    a cognate-id-distinct neighbor 'tūn'; 'hough' stays minted."""
    db = _db(tmp_path)
    hoh = _etymon(db, "hōh")  # residual element — folds 'hoh', no shared key with tūn
    tun = _etymon(db, "tūn")  # anchor — known reflex 'ton'
    _reflex(db, "ton", "post", tun)
    _breakdown(db, "Houghton", [(hoh, 0), (tun, 1)])
    db.commit()

    out = extract_implied_reflexes(db)
    # 'hough' (folds 'hough' ≠ 'hoh') is NOT a known reflex of hōh on ANY cognate
    # cluster, so the cross-cluster guard does not fire — the reflex is minted.
    assert [(r.canonical_form, r.residual) for r in out] == [("hōh", "hough")]


# --- apply min_support default ---------------------------------------------


def test_apply_defaults_to_min_support_2(tmp_path):
    """The EFFECTFUL apply path defaults to min_support=2: a support-1 reflex is NOT
    written (extract still returns it), but a support-2 one is."""
    db = _db(tmp_path)
    hoh = _etymon(db, "hōh")
    tun = _etymon(db, "tūn")
    burna = _etymon(db, "burna")
    winter = _etymon(db, "winter")
    _reflex(db, "ton", "post", tun)
    _reflex(db, "winter", "pre", winter)
    # hough←hōh: TWO distinct toponyms → support 2 → WRITTEN.
    _breakdown(db, "Houghton", [(hoh, 0), (tun, 1)], region="A")
    _breakdown(db, "Houghton", [(hoh, 0), (tun, 1)], region="B")
    # borne←burna: ONE toponym → support 1 → NOT written by default.
    _breakdown(db, "Winterborne", [(winter, 0), (burna, 1)])
    db.commit()

    # extract returns BOTH (default floor 1).
    assert {r.canonical_form for r in extract_implied_reflexes(db)} == {"hōh", "burna"}

    counts = apply_implied_reflexes(db)  # default min_support=2
    assert counts["reflexes"] == 1  # only hough←hōh
    assert (
        db.conn.execute("SELECT COUNT(*) FROM reflex WHERE surface_form='hough'").fetchone()[0] == 1
    )
    assert (
        db.conn.execute("SELECT COUNT(*) FROM reflex WHERE surface_form='borne'").fetchone()[0] == 0
    )


# --- confidence tiers ------------------------------------------------------


def test_support_tiers(tmp_path):
    db = _db(tmp_path)
    hoh = _etymon(db, "hōh")
    tun = _etymon(db, "tūn")
    burna = _etymon(db, "burna")
    _reflex(db, "ton", "post", tun)
    # hough←hōh attested by TWO distinct toponyms → high.
    _breakdown(db, "Houghton", [(hoh, 0), (tun, 1)], region="A")
    _breakdown(db, "Houghton", [(hoh, 0), (tun, 1)], region="B")
    # borne←burna attested by ONE → medium.
    _reflex(db, "winter", "pre", _etymon(db, "winter"))
    winter_id = db.conn.execute("SELECT id FROM etymon WHERE canonical_form='winter'").fetchone()[
        "id"
    ]
    _breakdown(db, "Winterborne", [(winter_id, 0), (burna, 1)])
    db.commit()

    out = {r.canonical_form: r for r in extract_implied_reflexes(db)}
    assert out["hōh"].support == 2
    assert out["burna"].support == 1
    assertions = {
        a.subject.ref: a for a in implied_reflex_assertions(list(out.values()), source="t")
    }
    hoh_node = mint_canonical_id("canonical_morpheme", "old-english", "hōh")
    burna_node = mint_canonical_id("canonical_morpheme", "old-english", "burna")
    assert assertions[hoh_node].confidence == "high"
    assert assertions[burna_node].confidence == "medium"


# --- assertion shape -------------------------------------------------------


def test_assertion_shape(tmp_path):
    db = _db(tmp_path)
    hoh = _etymon(db, "hōh")
    tun = _etymon(db, "tūn")
    _reflex(db, "ton", "post", tun)
    _breakdown(db, "Houghton", [(hoh, 0), (tun, 1)])
    db.commit()

    out = extract_implied_reflexes(db)
    assertions = implied_reflex_assertions(out, source="implied-reflex-residual")
    assert len(assertions) == 1
    a = assertions[0]
    assert a.predicate == "canonical-label"
    assert a.subject.type == "canonical_morpheme"
    assert a.subject.ref == mint_canonical_id("canonical_morpheme", "old-english", "hōh")
    assert a.qualifiers == {"stratum": "modern-english", "value": "hough"}
    assert a.id  # minted, deterministic


# --- apply: reflex + surface_in_modern -------------------------------------


def test_apply_writes_reflex_and_surface_in_modern(tmp_path):
    db = _db(tmp_path)
    hoh = _etymon(db, "hōh")
    tun = _etymon(db, "tūn")
    _reflex(db, "ton", "post", tun)
    _, teid = _breakdown(db, "Houghton", [(hoh, 0), (tun, 1)])
    db.commit()

    counts = apply_implied_reflexes(db, min_support=1)  # single-toponym fixture
    assert counts["reflexes"] == 1
    assert counts["reflex_links"] == 1
    # The implied reflex row + link exist.
    row = db.conn.execute(
        "SELECT r.id, r.position FROM reflex r WHERE r.surface_form='hough'"
    ).fetchone()
    assert row is not None and row["position"] == "pre"
    linked = db.conn.execute(
        "SELECT 1 FROM reflex_etymon re JOIN reflex r ON r.id=re.reflex_id "
        "WHERE r.surface_form='hough' AND re.etymon_id=?",
        (hoh,),
    ).fetchone()
    assert linked is not None
    # surface_in_modern filled on BOTH elements (anchor + residual).
    spans = dict(
        db.conn.execute(
            "SELECT etymon_id, surface_in_modern FROM toponym_etymology_element "
            "WHERE toponym_etymology_id=?",
            (teid,),
        ).fetchall()
    )
    assert spans[hoh] == "hough"
    assert spans[tun] == "ton"


def test_apply_idempotent(tmp_path):
    db = _db(tmp_path)
    hoh = _etymon(db, "hōh")
    tun = _etymon(db, "tūn")
    _reflex(db, "ton", "post", tun)
    _breakdown(db, "Houghton", [(hoh, 0), (tun, 1)])
    db.commit()

    apply_implied_reflexes(db, min_support=1)
    before = db.conn.execute("SELECT COUNT(*) FROM reflex").fetchone()[0]
    before_links = db.conn.execute("SELECT COUNT(*) FROM reflex_etymon").fetchone()[0]
    apply_implied_reflexes(db, min_support=1)  # re-run
    after = db.conn.execute("SELECT COUNT(*) FROM reflex").fetchone()[0]
    after_links = db.conn.execute("SELECT COUNT(*) FROM reflex_etymon").fetchone()[0]
    assert (before, before_links) == (after, after_links)


# --- round-trip + determinism ----------------------------------------------


def test_reflex_round_trips_through_rebuild(tmp_path):
    """apply → dump_reflexes_to_file → build_from_jsonl into a fresh DB reproduces
    the implied reflex (the rebuild-durability gate, ned5/br5o)."""
    db = _db(tmp_path)
    hoh = _etymon(db, "hōh")
    tun = _etymon(db, "tūn")
    _reflex(db, "ton", "post", tun)
    _breakdown(db, "Houghton", [(hoh, 0), (tun, 1)])
    db.commit()
    apply_implied_reflexes(db, min_support=1)

    out_dir = tmp_path / "mining"
    out_dir.mkdir()
    reflex_path, n = dump_reflexes_to_file(db.conn, out_dir)
    assert n > 0
    # The residual etymon (hōh) round-trips via its per-source dump (it's a
    # breakdown element of the 'topo' source) — that's where build resolves the
    # natural-key ref the reflex row points at (production: _reflexes.jsonl rides
    # alongside the full per-source L2).
    topo_path, _ = dump_source_to_file(db.conn, "topo", out_dir)
    db.close()

    fresh = _db(tmp_path, name="fresh.db")
    build_from_jsonl(fresh.conn, [reflex_path, topo_path])
    fresh.commit()
    reproduced = fresh.conn.execute(
        "SELECT r.surface_form, r.position FROM reflex r "
        "JOIN reflex_etymon re ON re.reflex_id=r.id "
        "JOIN etymon e ON e.id=re.etymon_id "
        "WHERE e.canonical_form='hōh' AND r.surface_form='hough'"
    ).fetchone()
    assert reproduced is not None
    assert reproduced["position"] == "pre"
    # surface_in_modern also round-trips through the toponym_etymology L2 dump.
    sim = fresh.conn.execute(
        "SELECT tee.surface_in_modern FROM toponym_etymology_element tee "
        "JOIN etymon e ON e.id=tee.etymon_id WHERE e.canonical_form='hōh'"
    ).fetchone()
    assert sim is not None and sim["surface_in_modern"] == "hough"


def test_determinism(tmp_path):
    db = _db(tmp_path)
    hoh = _etymon(db, "hōh")
    tun = _etymon(db, "tūn")
    burna = _etymon(db, "burna")
    winter = _etymon(db, "winter")
    _reflex(db, "ton", "post", tun)
    _reflex(db, "winter", "pre", winter)
    _breakdown(db, "Houghton", [(hoh, 0), (tun, 1)])
    _breakdown(db, "Winterborne", [(winter, 0), (burna, 1)])
    db.commit()

    a1 = [a.id for a in implied_reflex_assertions(extract_implied_reflexes(db), source="t")]
    a2 = [a.id for a in implied_reflex_assertions(extract_implied_reflexes(db), source="t")]
    assert a1 == a2
    # And the mined order is content-determined (not SQL row order).
    forms = [r.canonical_form for r in extract_implied_reflexes(db)]
    assert forms == sorted(set(forms), key=forms.index)  # stable across calls
