"""Tests for wyrd-aicu.9: genitive-``s`` split prior extraction.

Builds small in-memory lexicon DBs with a synthetic ``ston`` / ``ton``
homograph (town cluster vs stone cluster) and exercises the classifier,
attestation detector, extraction, persistence, dump, and lookup smoothing.
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli as cli_root
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema
from wyrd.generators.kenning.lexicon.genitive_priors import (
    PairCounts,
    _fold,
    attestation_verdict,
    discover_candidate_pairs,
    dump_genitive_priors_to_json,
    extract_genitive_priors,
    global_split_rate,
    load_genitive_priors_from_json,
    mine_genitive_priors,
    split_probability,
)


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    db = LexiconDB(db_path)
    db.conn.execute("INSERT INTO source (id, title) VALUES ('test_src', 'Test')")
    db.commit()
    yield db


def _etymon(db, form, language="old-english", cluster=False):
    """Insert an etymon. ``cluster=True`` makes it its own cognate-cluster head
    (cognate_id = its own id) — distinct etymons with cluster=True are distinct
    clusters; ``cognate_id`` is a self-FK so a fabricated id would fail."""
    cur = db.conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES (?, ?)",
        (form, language),
    )
    eid = cur.lastrowid
    if cluster:
        db.conn.execute("UPDATE etymon SET cognate_id = ? WHERE id = ?", (eid, eid))
    return eid


def _reflex(db, surface, position, etymon_id):
    cur = db.conn.execute(
        "INSERT INTO reflex (surface_form, position) VALUES (?, ?)",
        (surface, position),
    )
    db.conn.execute(
        "INSERT INTO reflex_etymon (reflex_id, etymon_id) VALUES (?, ?)",
        (cur.lastrowid, etymon_id),
    )


def _toponym(db, name, country="England"):
    cur = db.conn.execute(
        "INSERT INTO toponym (modern_name, country) VALUES (?, ?)", (name, country)
    )
    return cur.lastrowid


def _etymology(db, toponym_id, elements):
    cur = db.conn.execute(
        "INSERT INTO toponym_etymology (toponym_id, source_id, confidence) "
        "VALUES (?, 'test_src', 'high')",
        (toponym_id,),
    )
    for ordinal, eid in enumerate(elements):
        db.conn.execute(
            "INSERT INTO toponym_etymology_element "
            "(toponym_etymology_id, ordinal, etymon_id) VALUES (?, ?, ?)",
            (cur.lastrowid, ordinal, eid),
        )


def _attest(db, toponym_id, form, year=1086):
    db.conn.execute(
        "INSERT INTO toponym_attestation (toponym_id, form, date_year) VALUES (?, ?, ?)",
        (toponym_id, form, year),
    )


@pytest.fixture
def ston_world(db):
    """A ston/ton homograph: tūn (town, cluster 100) vs stān (stone, 200)."""
    tun = _etymon(db, "tūn", cluster=True)
    stan = _etymon(db, "stān", cluster=True)
    bishop = _etymon(db, "Bishop")
    rud = _etymon(db, "Rud")
    aelfric = _etymon(db, "Ælfrīc")
    # reflexes: 'ton'/'tone' -> town cluster; 'ston'/'stone' -> stone cluster.
    _reflex(db, "-ton", "post", tun)
    _reflex(db, "-tone", "post", tun)
    _reflex(db, "-ston", "post", stan)
    _reflex(db, "-stone", "post", stan)
    db.commit()
    return {"tun": tun, "stan": stan, "bishop": bishop, "rud": rud, "aelfric": aelfric}


# ---------- _fold --------------------------------------------------------


def test_fold_deaccents_lowercases_strips_nonletters():
    assert _fold("Ælfrīce-s-tūn") == "aelfricestun"
    assert _fold("-STON-") == "ston"


# ---------- discover_candidate_pairs -------------------------------------


def test_discover_pairs_finds_s_overlap():
    pairs = discover_candidate_pairs({"ston", "ton", "sley", "ley", "lone"})
    assert pairs["ston"] == "ton"
    assert pairs["sley"] == "ley"
    assert "lone" not in pairs  # 'one' not present, and doesn't start with the s-rule


def test_discover_pairs_requires_short_form_present():
    # 'ston' present but 'ton' absent -> not a pair.
    assert discover_candidate_pairs({"ston", "xyz"}) == {}


# ---------- attestation_verdict ------------------------------------------


def test_attestation_split_on_unambiguous_es_town_stem():
    # Kingston <- Chinge·s·tun : 'es' + town stem 'tun', no stone-stem ending.
    assert (
        attestation_verdict(["Chingestun"], split_stems={"tun", "ton"}, literal_stems={"stone"})
        == "split"
    )


def test_attestation_literal_on_stone_stem():
    # Rudston <- Rode·stan : a bound stone stem, no genitive-marked town stem.
    assert (
        attestation_verdict(
            ["Rodestan"],
            split_stems={"ton", "tone"},
            literal_stems={"ston", "stone", "stan"},
        )
        == "literal"
    )


def test_attestation_defers_when_ambiguous():
    # 'alvricestone' = 'es'+'tone' (town) OR 'e'+'stone' (stone) — genuinely
    # ambiguous at the string level, so defer to the breakdown cluster (None).
    v = attestation_verdict(["Alvricestone"], split_stems={"tone"}, literal_stems={"stone"})
    assert v is None


def test_attestation_none_when_undecisive():
    assert attestation_verdict(["Bryntir"], split_stems={"ton"}, literal_stems={"ston"}) is None


# ---------- extraction ---------------------------------------------------


def test_extract_end_to_end(db, ston_world):
    w = ston_world
    # three towns (genitive split), one genuine stone (literal).
    t1 = _toponym(db, "Bishopston")
    _etymology(db, t1, [w["bishop"], w["tun"]])
    t2 = _toponym(db, "Kingston")
    _etymology(db, t2, [w["bishop"], w["tun"]])
    t3 = _toponym(db, "Garston")
    _etymology(db, t3, [w["bishop"], w["tun"]])
    s1 = _toponym(db, "Rudston")
    _etymology(db, s1, [w["rud"], w["stan"]])
    # attestation override: breakdown points at stone, but the dated form shows
    # an UNAMBIGUOUS genitive 'es' + town stem ('eston', no stone-stem ending)
    # -> split wins over the breakdown, and is counted as attested.
    a1 = _toponym(db, "Alfriston")
    _etymology(db, a1, [w["stan"]])
    _attest(db, a1, "Alvriceston")
    db.commit()

    counts, summary = extract_genitive_priors(db)
    pair = counts[("ston", "ton")]
    assert pair.split == 4  # 3 town breakdowns + 1 attestation override
    assert pair.literal == 1  # Rudston
    assert pair.attested_split == 1  # Alfriston via attestation
    assert summary.from_attestation == 1
    assert summary.from_breakdown == 4


def test_mine_apply_writes_and_is_idempotent(db, ston_world):
    w = ston_world
    t1 = _toponym(db, "Bishopston")
    _etymology(db, t1, [w["bishop"], w["tun"]])
    s1 = _toponym(db, "Rudston")
    _etymology(db, s1, [w["rud"], w["stan"]])
    db.commit()

    mine_genitive_priors(db, apply=True)
    rows = list(
        db.conn.execute(
            "SELECT long_form, short_form, split_count, literal_count " "FROM genitive_split_prior"
        )
    )
    assert (rows[0]["long_form"], rows[0]["short_form"]) == ("ston", "ton")
    assert rows[0]["split_count"] == 1
    assert rows[0]["literal_count"] == 1

    # idempotent: a second apply produces identical contents.
    mine_genitive_priors(db, apply=True)
    rows2 = list(db.conn.execute("SELECT * FROM genitive_split_prior"))
    assert len(rows2) == 1


def test_dry_run_does_not_write(db, ston_world):
    w = ston_world
    t1 = _toponym(db, "Bishopston")
    _etymology(db, t1, [w["bishop"], w["tun"]])
    db.commit()
    mine_genitive_priors(db, apply=False)
    n = db.conn.execute("SELECT COUNT(*) AS n FROM genitive_split_prior").fetchone()["n"]
    assert n == 0


# ---------- dump + round-trip --------------------------------------------


def test_dump_json_byte_stable_and_round_trips(db, ston_world, tmp_path):
    w = ston_world
    t1 = _toponym(db, "Bishopston")
    _etymology(db, t1, [w["bishop"], w["tun"]])
    s1 = _toponym(db, "Rudston")
    _etymology(db, s1, [w["rud"], w["stan"]])
    db.commit()
    mine_genitive_priors(db, apply=True)

    out1 = tmp_path / "g1.json"
    out2 = tmp_path / "g2.json"
    dump_genitive_priors_to_json(db, out1, version="v1")
    dump_genitive_priors_to_json(db, out2, version="v1")
    assert out1.read_bytes() == out2.read_bytes()  # byte-stable

    loaded = load_genitive_priors_from_json(out1)
    assert loaded[("ston", "ton")] == PairCounts(split=1, literal=1)
    artifact = json.loads(out1.read_text())
    assert artifact["version"] == "v1"


# ---------- lookup smoothing ---------------------------------------------


def test_global_split_rate():
    counts = {
        ("ston", "ton"): PairCounts(split=9, literal=1),
        ("sley", "ley"): PairCounts(split=10, literal=0),
    }
    assert global_split_rate(counts) == pytest.approx(19 / 20)


def test_split_probability_smooths_toward_backoff():
    counts = {("ston", "ton"): PairCounts(split=3, literal=1)}
    # explicit backoff 0.5, pseudocount 5: (3 + 2.5) / (4 + 5)
    p = split_probability(counts, "ston", "ton", pseudocount=5.0, backoff=0.5)
    assert p == pytest.approx(5.5 / 9)


def test_split_probability_absent_pair_is_backoff():
    counts = {("ston", "ton"): PairCounts(split=3, literal=1)}
    p = split_probability(counts, "sham", "ham", pseudocount=5.0, backoff=0.4)
    assert p == pytest.approx(0.4)


def test_split_probability_large_n_approaches_empirical():
    counts = {("ston", "ton"): PairCounts(split=940, literal=60)}
    p = split_probability(counts, "ston", "ton", pseudocount=5.0, backoff=0.5)
    assert p == pytest.approx(0.94, abs=0.01)


# ---------- CLI smoke ----------------------------------------------------


def test_cli_mine_and_dump(db, ston_world, tmp_path):
    w = ston_world
    t1 = _toponym(db, "Bishopston")
    _etymology(db, t1, [w["bishop"], w["tun"]])
    db.commit()
    db_path = db.conn.execute("PRAGMA database_list").fetchone()["file"]
    db.close()

    runner = CliRunner()
    r = runner.invoke(
        cli_root,
        ["lexicon", "mine-genitive-priors", "--db", db_path, "--apply"],
    )
    assert r.exit_code == 0, r.output
    out = tmp_path / "g.json"
    r2 = runner.invoke(
        cli_root,
        ["lexicon", "dump-genitive-priors", "--db", db_path, "--output", str(out)],
    )
    assert r2.exit_code == 0, r2.output
    assert json.loads(out.read_text())["pairs"][0]["long_form"] == "ston"
