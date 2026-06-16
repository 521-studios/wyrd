"""Tests for wyrd-aicu.9: genitive-``s`` split prior extraction.

Builds small in-memory lexicon DBs with a synthetic ``ston`` / ``ton``
homograph (town cognate cluster vs stone cognate cluster) and exercises the
classifier, extraction (split / literal / both-skip / unclassified-skip),
cross-region pooling, replace-not-merge persistence, byte-stable dump, and
lookup smoothing.
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
    _genitive_ambiguous,
    detect_historical_genitive,
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
    # Production stores the BARE de-dashed surface (D45 / migration 0017); the
    # explicit position lives in the position column.
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


def _etymology(db, toponym_id, elements, historical_form=None):
    cur = db.conn.execute(
        "INSERT INTO toponym_etymology (toponym_id, source_id, confidence, historical_form) "
        "VALUES (?, 'test_src', 'high', ?)",
        (toponym_id, historical_form),
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
    """A ston/ton homograph: tūn (town, its own cluster) vs stān (stone, its own
    cluster). Reflexes stored bare (production de-dashed shape). ``stan`` is the
    unambiguous bound stone reflex; ``ston``/``stone`` are genitive-ambiguous."""
    tun = _etymon(db, "tūn", cluster=True)
    stan = _etymon(db, "stān", cluster=True)
    bishop = _etymon(db, "Bishop")
    rud = _etymon(db, "Rud")
    _reflex(db, "ton", "post", tun)
    _reflex(db, "tone", "post", tun)
    _reflex(db, "ston", "post", stan)
    _reflex(db, "stone", "post", stan)
    _reflex(db, "stan", "post", stan)
    db.commit()
    return {"tun": tun, "stan": stan, "bishop": bishop, "rud": rud}


# ---------- _fold --------------------------------------------------------


def test_fold_deaccents_lowercases_strips_nonletters():
    assert _fold("Ælfrīce-s-tūn") == "aelfricestun"
    assert _fold("-STON-") == "ston"


# ---------- discover_candidate_pairs -------------------------------------


def test_discover_pairs_finds_s_overlap():
    pairs = discover_candidate_pairs({"ston", "ton", "sley", "ley", "lone"})
    assert pairs["ston"] == "ton"
    assert pairs["sley"] == "ley"
    assert "lone" not in pairs


def test_discover_pairs_requires_short_form_present():
    # 'ston' present but 'ton' absent -> not a pair.
    assert discover_candidate_pairs({"ston", "xyz"}) == {}


# ---------- extraction ---------------------------------------------------


def test_extract_classifies_town_split_and_stone_literal(db, ston_world):
    w = ston_world
    for nm in ("Bishopston", "Kingston", "Garston"):  # 3 towns (tūn)
        t = _toponym(db, nm)
        _etymology(db, t, [w["bishop"], w["tun"]])
    s = _toponym(db, "Rudston")  # genuine stone (stān) — must stay literal
    _etymology(db, s, [w["rud"], w["stan"]])
    db.commit()

    counts, summary = extract_genitive_priors(db)
    assert counts[("ston", "ton")] == PairCounts(split=3, literal=1)
    assert summary.classified_split == 3
    assert summary.classified_literal == 1
    assert summary.skipped_both == 0
    assert summary.skipped_unclassified == 0


def test_extract_pools_same_name_across_regions(db, ston_world):
    w = ston_world
    # 'Kingston' in two different regions, both broken down as town -> 2 split.
    for region in ("Cheshire", "Dorset"):
        cur = db.conn.execute(
            "INSERT INTO toponym (modern_name, country, region) VALUES ('Kingston','England',?)",
            (region,),
        )
        _etymology(db, cur.lastrowid, [w["bishop"], w["tun"]])
    db.commit()
    counts, _ = extract_genitive_priors(db)
    assert counts[("ston", "ton")].split == 2  # every place counts


def test_extract_skips_both_and_unclassified(db, ston_world):
    w = ston_world
    # pooled breakdown touches BOTH town and stone clusters -> skipped_both.
    t = _toponym(db, "Bothston")
    _etymology(db, t, [w["tun"], w["stan"]])
    # breakdown touches NEITHER class (only the specifier) -> unclassified.
    u = _toponym(db, "Plainston")
    _etymology(db, u, [w["bishop"]])
    db.commit()
    counts, summary = extract_genitive_priors(db)
    assert ("ston", "ton") not in counts  # nothing classified
    assert summary.skipped_both == 1
    assert summary.skipped_unclassified == 1


def test_overlap_cluster_disambiguates_toward_literal(db, ston_world):
    # An etymon that is a reflex for BOTH 'ton' and 'ston' (a shared cluster):
    # `_pair_clusters` subtracts it from the SPLIT side, so a breakdown using
    # only it classifies LITERAL — not skipped-both. Without the subtraction the
    # shared cluster would be in both classes and the toponym would skip. Guards
    # the overlap-disambiguation that replaced the removed attestation gate.
    shared = _etymon(db, "shared", cluster=True)
    # link `shared` to the EXISTING 'ton' + 'ston' reflex rows (reflex is
    # UNIQUE(surface_form, position); the many-to-many lives in reflex_etymon).
    for surf in ("ton", "ston"):
        rid = db.conn.execute(
            "SELECT id FROM reflex WHERE surface_form=? AND position='post'", (surf,)
        ).fetchone()["id"]
        db.conn.execute(
            "INSERT INTO reflex_etymon (reflex_id, etymon_id) VALUES (?, ?)", (rid, shared)
        )
    t = _toponym(db, "Shareston")
    _etymology(db, t, [shared])
    db.commit()
    counts, summary = extract_genitive_priors(db)
    assert counts[("ston", "ton")] == PairCounts(split=0, literal=1)
    assert summary.skipped_both == 0


def test_extract_empty_db_yields_no_counts(db, ston_world):
    counts, summary = extract_genitive_priors(db)
    assert counts == {}
    assert summary.active_pairs == 0
    assert global_split_rate(counts) == 0.5  # no-evidence fallback


def test_extract_progress_path_runs(db, ston_world, capsys):
    w = ston_world
    for nm in ("Bishopston", "Kingston"):
        t = _toponym(db, nm)
        _etymology(db, t, [w["bishop"], w["tun"]])
    db.commit()
    counts, _ = extract_genitive_priors(db, progress_every=1)
    assert counts[("ston", "ton")].split == 2
    assert "/2]" in capsys.readouterr().err  # [scanned/total] progress shape


# ---------- persistence --------------------------------------------------


def test_mine_apply_writes_and_is_idempotent(db, ston_world):
    w = ston_world
    t = _toponym(db, "Bishopston")
    _etymology(db, t, [w["bishop"], w["tun"]])
    s = _toponym(db, "Rudston")
    _etymology(db, s, [w["rud"], w["stan"]])
    db.commit()

    mine_genitive_priors(db, apply=True)
    rows = list(
        db.conn.execute(
            "SELECT long_form, short_form, split_count, literal_count " "FROM genitive_split_prior"
        )
    )
    assert (rows[0]["long_form"], rows[0]["short_form"]) == ("ston", "ton")
    assert (rows[0]["split_count"], rows[0]["literal_count"]) == (1, 1)

    mine_genitive_priors(db, apply=True)  # idempotent
    assert len(list(db.conn.execute("SELECT * FROM genitive_split_prior"))) == 1


def test_mine_apply_is_replace_not_merge(db, ston_world):
    w = ston_world
    s = _toponym(db, "Rudston")
    _etymology(db, s, [w["rud"], w["stan"]])
    db.commit()
    mine_genitive_priors(db, apply=True)
    assert (
        db.conn.execute(
            "SELECT literal_count FROM genitive_split_prior WHERE long_form='ston'"
        ).fetchone()["literal_count"]
        == 1
    )

    # Remove the stone observation; re-apply must EVICT the stale pair, not keep it.
    db.conn.execute("DELETE FROM toponym_etymology")
    db.commit()
    mine_genitive_priors(db, apply=True)
    assert db.conn.execute("SELECT COUNT(*) AS n FROM genitive_split_prior").fetchone()["n"] == 0


def test_dry_run_does_not_write(db, ston_world):
    w = ston_world
    t = _toponym(db, "Bishopston")
    _etymology(db, t, [w["bishop"], w["tun"]])
    db.commit()
    mine_genitive_priors(db, apply=False)
    assert db.conn.execute("SELECT COUNT(*) AS n FROM genitive_split_prior").fetchone()["n"] == 0


# ---------- dump + round-trip --------------------------------------------


def test_dump_json_byte_stable_and_round_trips(db, ston_world, tmp_path):
    w = ston_world
    t = _toponym(db, "Bishopston")
    _etymology(db, t, [w["bishop"], w["tun"]])
    s = _toponym(db, "Rudston")
    _etymology(db, s, [w["rud"], w["stan"]])
    db.commit()
    mine_genitive_priors(db, apply=True)

    out1, out2 = tmp_path / "g1.json", tmp_path / "g2.json"
    dump_genitive_priors_to_json(db, out1, version="v1")
    dump_genitive_priors_to_json(db, out2, version="v1")
    assert out1.read_bytes() == out2.read_bytes()  # byte-stable

    loaded = load_genitive_priors_from_json(out1)
    assert loaded[("ston", "ton")] == PairCounts(split=1, literal=1)
    assert json.loads(out1.read_text())["version"] == "v1"


# ---------- lookup smoothing ---------------------------------------------


def test_global_split_rate():
    counts = {
        ("ston", "ton"): PairCounts(split=9, literal=1),
        ("sley", "ley"): PairCounts(split=10, literal=0),
    }
    assert global_split_rate(counts) == pytest.approx(19 / 20)


def test_split_probability_smooths_toward_backoff():
    counts = {("ston", "ton"): PairCounts(split=3, literal=1)}
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


# ---------- attestation refinement (wyrd-aicu.9.1) -----------------------


def test_genitive_ambiguous_drops_s_plus_townstem():
    # 'ston' = s+ton and 'stone' = s+tone collide with the genitive town
    # reading; 'stan' (bound stone stem) does not.
    town = {"ton", "tone"}
    assert _genitive_ambiguous("ston", town) is True
    assert _genitive_ambiguous("stone", town) is True
    assert _genitive_ambiguous("stan", town) is False


def test_detect_historical_genitive_town_on_es_marker():
    # Kingston <- cyninge·s·tun : genuine historical 'es' + town stem.
    assert (
        detect_historical_genitive(
            ["Cyningeston"], "Kingston", town_stems={"ton", "tone"}, stone_stems={"stan"}
        )
        == "split"
    )


def test_detect_historical_genitive_stone_on_bound_stem():
    assert (
        detect_historical_genitive(
            ["Rodestan"], "Rudston", town_stems={"ton", "tone"}, stone_stems={"stan"}
        )
        == "literal"
    )


def test_detect_historical_genitive_ignores_modern_echo():
    # The form merely echoes the modern name -> no genuine historical signal.
    assert (
        detect_historical_genitive(
            ["Beeston"], "Beeston", town_stems={"ton", "tone"}, stone_stems={"stan"}
        )
        is None
    )


def test_detect_historical_genitive_none_without_forms():
    assert (
        detect_historical_genitive([], "Rudston", town_stems={"ton"}, stone_stems={"stan"}) is None
    )


def test_detect_historical_genitive_defers_on_conflict():
    # One historical form shows a genitive town marker ('es'+'ton'), another ends
    # in a bound stone stem ('stan') -> the forms disagree, defer to the cluster.
    assert (
        detect_historical_genitive(
            ["Cyningeston", "Rodestan"], "Whatever", town_stems={"ton"}, stone_stems={"stan"}
        )
        is None
    )


def test_attestation_resolves_unclassified_to_town(db, ston_world):
    w = ston_world
    # Cluster-UNCLASSIFIED (breakdown is only the specifier), but a genuine
    # historical form shows the genitive 'es' + town stem -> resolved to split.
    t = _toponym(db, "Kingston")
    _etymology(db, t, [w["bishop"]], historical_form="Cyningeston")
    db.commit()
    counts, summary = extract_genitive_priors(db)
    assert counts[("ston", "ton")] == PairCounts(split=1, literal=0)
    assert summary.resolved_by_attestation == 1
    assert summary.skipped_unclassified == 0


def test_attestation_never_overrides_decisive_cluster(db, ston_world):
    w = ston_world
    # Rudston: breakdown is decisively STONE (stān cluster). Even with a
    # genitive-looking historical form, the decisive cluster verdict wins.
    s = _toponym(db, "Rudston")
    _etymology(db, s, [w["rud"], w["stan"]], historical_form="Rodeston")
    db.commit()
    counts, summary = extract_genitive_priors(db)
    assert counts[("ston", "ton")] == PairCounts(split=0, literal=1)
    assert summary.resolved_by_attestation == 0  # attestation not consulted


def test_attestation_modern_echo_stays_skipped(db, ston_world):
    w = ston_world
    # Unclassified + the only form echoes the modern name -> stays skipped (the
    # exact regression the genuine-historical filter prevents).
    t = _toponym(db, "Beeston")
    _etymology(db, t, [w["bishop"]])
    _attest(db, t, "Beeston")
    db.commit()
    counts, summary = extract_genitive_priors(db)
    assert ("ston", "ton") not in counts
    assert summary.resolved_by_attestation == 0
    assert summary.skipped_unclassified == 1


def test_detect_historical_genitive_consonant_s_arm():
    # 'wulfricston' = ...c + s + ton : the consonant-`s` arm (no 'e' vowel),
    # the reason that arm exists. Bare modern 'wolston' would not match it.
    assert (
        detect_historical_genitive(
            ["Wulfricston"], "Wolston", town_stems={"ton"}, stone_stems={"stan"}
        )
        == "split"
    )


def test_attestation_resolves_from_attestation_form_source(db, ston_world):
    w = ston_world
    # Unclassified breakdown; the genitive signal comes from toponym_attestation
    # (not historical_form) — exercises that source path end-to-end.
    t = _toponym(db, "Kingston")
    _etymology(db, t, [w["bishop"]])
    _attest(db, t, "Cyningeston")
    db.commit()
    counts, summary = extract_genitive_priors(db)
    assert counts[("ston", "ton")] == PairCounts(split=1, literal=0)
    assert summary.resolved_by_attestation == 1


def test_attestation_resolves_both_cluster_residue(db, ston_world):
    w = ston_world
    # Breakdown touches BOTH clusters (normally skipped_both), but a genuine
    # historical form resolves it — the `both`→attestation branch of _resolve.
    t = _toponym(db, "Kingston")
    _etymology(db, t, [w["tun"], w["stan"]], historical_form="Cyningeston")
    db.commit()
    counts, summary = extract_genitive_priors(db)
    assert counts[("ston", "ton")].split == 1
    assert summary.resolved_by_attestation == 1
    assert summary.skipped_both == 0


# ---------- CLI smoke ----------------------------------------------------


def test_cli_mine_and_dump(db, ston_world, tmp_path):
    w = ston_world
    t = _toponym(db, "Bishopston")
    _etymology(db, t, [w["bishop"], w["tun"]])
    db.commit()
    db_path = db.conn.execute("PRAGMA database_list").fetchone()["file"]
    db.close()

    runner = CliRunner()
    r = runner.invoke(cli_root, ["lexicon", "mine-genitive-priors", "--db", db_path, "--apply"])
    assert r.exit_code == 0, r.output
    out = tmp_path / "g.json"
    r2 = runner.invoke(
        cli_root, ["lexicon", "dump-genitive-priors", "--db", db_path, "--output", str(out)]
    )
    assert r2.exit_code == 0, r2.output
    assert json.loads(out.read_text())["pairs"][0]["long_form"] == "ston"
