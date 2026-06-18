"""Tests for the wyrd-aicu.9 decomposition grader (Phase 3).

Builds a small in-memory lexicon (the ``ston``/``ton`` homograph + a
non-homograph ``sworth`` genitive) and a matching matcher trie, then grades the
matcher against the synthetic scholarly breakdowns connective OFF vs ON. Proves:

* the cognate-cluster bridge + scholar-corpus pooling;
* per-toponym recall / precision / exact / coverage / head metrics;
* the connective ON gives ``Grimsworth`` a coverage gain (the ``s`` stops being
  unaccounted) with NO regression — a non-homograph clean win;
* under a town-dominant prior the connective splits ``Bishopston`` to the town
  (an improvement) while the genuine-stone ``Rudston`` mis-splits — and the
  grader correctly surfaces that as a REGRESSION (the suffix-level prior can't
  tell the 12 genuine-stone names apart from the 181 towns; the tripwire firing
  on the known, defensible minority is exactly its job).
"""

from __future__ import annotations

import pytest

from wyrd.generators.kenning.lexicon import LexiconDB, init_schema
from wyrd.generators.kenning.lexicon.decomposition_grader import (
    grade_configuration,
    grade_corpus_diff,
    grade_passthrough_diff,
    load_cluster_index,
    load_scholar_corpus,
)
from wyrd.generators.kenning.runtime.connective import DEFAULT_CONNECTIVE_INVENTORY
from wyrd.generators.kenning.runtime.meaning import Meaning
from wyrd.generators.kenning.runtime.trie_matcher import build_morpheme_trie

# Town-dominant prior for the shared ('ston', 'ton') suffix pair (~the live
# ston P≈0.94). One pair, so it drives EVERY -ston name the same way.
TOWN_PRIOR = {("ston", "ton"): 0.9}


def _etymon(db, form, *, cluster=True, language="old-english"):
    cur = db.conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES (?, ?)", (form, language)
    )
    eid = cur.lastrowid
    if cluster:
        db.conn.execute("UPDATE etymon SET cognate_id = ? WHERE id = ?", (eid, eid))
    return eid


def _reflex(db, surface, etymon_id, position="post"):
    cur = db.conn.execute(
        "INSERT INTO reflex (surface_form, position) VALUES (?, ?)", (surface, position)
    )
    db.conn.execute(
        "INSERT INTO reflex_etymon (reflex_id, etymon_id) VALUES (?, ?)",
        (cur.lastrowid, etymon_id),
    )


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
    """Lexicon + matcher trie sharing the ston/ton/worth surfaces."""
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    db = LexiconDB(db_path)
    db.conn.execute("INSERT INTO source (id, title) VALUES ('test_src', 'Test')")

    tun = _etymon(db, "tūn")
    stan = _etymon(db, "stān")
    bishop = _etymon(db, "Bishop")
    rud = _etymon(db, "Rud")
    grim = _etymon(db, "Grim")
    worth = _etymon(db, "worð")

    _reflex(db, "ton", tun)
    _reflex(db, "ston", stan)
    _reflex(db, "stone", stan)
    _reflex(db, "bishop", bishop, position="pre")
    _reflex(db, "rud", rud, position="pre")
    _reflex(db, "grim", grim, position="pre")
    _reflex(db, "worth", worth)

    # Scholarly breakdowns: Bishopston = bishop + tūn (TOWN); Rudston = rud + stān
    # (genuine STONE); Grimsworth = grim + worð (non-homograph genitive).
    _toponym(db, "Bishopston", [bishop, tun])
    _toponym(db, "Rudston", [rud, stan])
    _toponym(db, "Grimsworth", [grim, worth])
    db.commit()

    # Matcher inventory: ston (stone) + ton (town) are the homograph; 'sworth'
    # is NOT a morpheme, so the genitive s of Grimsworth needs the connective.
    meaning_db = {
        "bishop": [Meaning("bishop", [], [], {})],
        "ston": [Meaning("ston", [], [], {})],
        "ton": [Meaning("ton", [], [], {})],
        "rud": [Meaning("rud", [], [], {})],
        "grim": [Meaning("grim", [], [], {})],
        "worth": [Meaning("worth", [], [], {})],
    }
    trie = build_morpheme_trie(meaning_db)
    yield db, trie
    db.close()


def _by_name(grades):
    return {g.name: g for g in grades}


def test_cluster_index_and_corpus(world):
    db, _ = world
    index = load_cluster_index(db)
    # ston resolves to the stone cluster; ton to the town cluster; they differ.
    assert index.clusters_for("ston") == index.clusters_for("stone")
    assert index.clusters_for("ston").isdisjoint(index.clusters_for("ton"))
    corpus = load_scholar_corpus(db, index)
    assert [sc.name for sc in corpus] == ["Bishopston", "Grimsworth", "Rudston"]
    # Bishopston's scholar clusters = {bishop, tūn} and DON'T include stone.
    bishopston = _by_name(corpus)["Bishopston"]
    assert bishopston.clusters.isdisjoint(index.clusters_for("ston"))
    assert not bishopston.clusters.isdisjoint(index.clusters_for("ton"))


def test_off_baseline_reproduces_ston_bug(world):
    db, trie = world
    index = load_cluster_index(db)
    corpus = load_scholar_corpus(db, index)
    off = _by_name(
        grade_configuration(
            corpus,
            trie,
            index,
            culture_languages=None,
            connective_inventory=None,
            genitive_prior=None,
        )
    )
    # Bishopston mis-parses to bishop+ston(stone): the town cluster is missed
    # and stone is spurious, so the head is NOT a scholar cluster.
    assert off["Bishopston"].morphemes == ("bishop", "ston")
    assert off["Bishopston"].recall == 0.5
    assert not off["Bishopston"].exact
    assert not off["Bishopston"].head_attested
    # Rudston is correct OFF (genuine stone): exact + head attested.
    assert off["Rudston"].exact
    assert off["Rudston"].head_attested
    # Grimsworth recovers both morphemes but leaks the genitive s → no coverage.
    assert off["Grimsworth"].morphemes == ("grim", "worth")
    assert off["Grimsworth"].exact
    assert not off["Grimsworth"].coverage


def test_connective_coverage_gain_no_prior(world):
    db, trie = world
    index = load_cluster_index(db)
    corpus = load_scholar_corpus(db, index)
    on = _by_name(
        grade_configuration(
            corpus,
            trie,
            index,
            culture_languages=None,
            connective_inventory=DEFAULT_CONNECTIVE_INVENTORY,
            genitive_prior={},  # no homograph tiebreak
        )
    )
    # Grimsworth: the connective absorbs the s → a clean 0-unaccounted parse,
    # same morphemes. Pure coverage gain.
    assert on["Grimsworth"].morphemes == ("grim", "worth")
    assert on["Grimsworth"].coverage
    assert on["Grimsworth"].exact


def test_passthrough_expansion_recovers_constituents(tmp_path):
    # wyrd-h5u1/D51.3: the matcher parses Aldington as ald+ington (composite);
    # without passthrough, ington attributes to its own cluster and misses the
    # scholar's ing+tūn. The passthrough map expands ington -> (ing, tūn) for
    # attribution -> recall/head recovered, coverage unchanged.
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    db = LexiconDB(db_path)
    db.conn.execute("INSERT INTO source (id, title) VALUES ('test_src', 'Test')")
    ald = _etymon(db, "Ald")
    ing = _etymon(db, "ing")
    tun = _etymon(db, "tūn")
    ington = _etymon(db, "ington")  # the composite
    _reflex(db, "ald", ald, position="pre")
    _reflex(db, "ington", ington)
    _toponym(db, "Aldington", [ald, ing, tun])  # scholar: the finer breakdown
    db.commit()

    index = load_cluster_index(db)
    meaning_db = {  # matcher inventory has the composite, not ing/tūn separately
        "ald": [Meaning("ald", [], [], {})],
        "ington": [Meaning("ington", [], [], {})],
    }
    trie = build_morpheme_trie(meaning_db)
    corpus = load_scholar_corpus(db, index)
    pmap = {index.cluster_of[ington]: (index.cluster_of[ing], index.cluster_of[tun])}

    def grade(passthrough_map):
        return grade_configuration(
            corpus,
            trie,
            index,
            culture_languages=None,
            connective_inventory=None,
            genitive_prior=None,
            passthrough_map=passthrough_map,
        )[0]

    off = grade(None)
    on = grade(pmap)
    db.close()

    assert off.morphemes == ("ald", "ington")  # parse identical either way
    assert off.recall < 1.0 and not off.head_attested  # ington opaque -> misses ing+tūn
    assert on.recall == 1.0 and on.exact and on.head_attested  # expanded -> recovered
    assert off.coverage and on.coverage  # passthrough doesn't change the parse/coverage


def test_diff_improves_town_regresses_genuine_stone(world):
    db, trie = world
    diff = grade_corpus_diff(
        db,
        trie,
        culture_languages=None,
        genitive_prior=TOWN_PRIOR,
    )
    # Aggregate: coverage strictly up (Grimsworth gained a clean parse).
    assert diff.on.coverage_rate > diff.off.coverage_rate

    regressed = {c.name for c in diff.regressions}
    improved = {c.name for c in diff.improvements}

    # Bishopston (majority town) improves; Grimsworth coverage-gains; neither
    # regresses.
    assert "Bishopston" in improved
    assert "Grimsworth" in improved
    assert "Bishopston" not in regressed
    assert "Grimsworth" not in regressed

    # Rudston (genuine stone) shares the ('ston','ton') suffix with Bishopston,
    # so the town-dominant prior mis-splits it — the grader surfaces it as the
    # known, defensible genuine-stone regression (the tripwire working).
    assert "Rudston" in regressed
    rud = next(c for c in diff.regressions if c.name == "Rudston")
    assert "recall" in rud.dimensions
    assert tuple(rud.off_parse) == ("rud", "ston")
    assert tuple(rud.on_parse) == ("rud", "ton")


def test_grade_passthrough_diff_recovers_constituents(tmp_path):
    # wyrd-h5u1 net-win check: grade_passthrough_diff holds the matcher constant and
    # diffs attribution OFF vs ON. Aldington parses to ald+ington; crediting ington's
    # constituents (ing+tūn) recovers recall/head with NO regression + flat coverage.
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    db = LexiconDB(db_path)
    db.conn.execute("INSERT INTO source (id, title) VALUES ('test_src', 'Test')")
    ald = _etymon(db, "Ald")
    ing = _etymon(db, "ing")
    tun = _etymon(db, "tūn")
    ington = _etymon(db, "ington")
    _reflex(db, "ald", ald, position="pre")
    _reflex(db, "ington", ington)
    _toponym(db, "Aldington", [ald, ing, tun])
    db.commit()

    index = load_cluster_index(db)
    trie = build_morpheme_trie(
        {"ald": [Meaning("ald", [], [], {})], "ington": [Meaning("ington", [], [], {})]}
    )
    pmap = {index.cluster_of[ington]: (index.cluster_of[ing], index.cluster_of[tun])}

    diff = grade_passthrough_diff(
        db,
        trie,
        passthrough_map=pmap,
        culture_languages=None,
        genitive_prior=None,
        connective_inventory=None,
    )
    db.close()

    assert diff.on.mean_recall > diff.off.mean_recall  # ing+tūn recovered
    assert diff.on.head_attested_rate > diff.off.head_attested_rate
    assert diff.on.coverage_rate == diff.off.coverage_rate  # parse unchanged
    assert not diff.regressions  # attribution only credits, never worsens
    assert any(c.name == "Aldington" for c in diff.improvements)


def test_grade_passthrough_diff_irrelevant_map_is_noop(world):
    # A passthrough_map whose composite cluster never matches in the corpus must
    # leave the grade untouched: OFF == ON, zero improvements, zero regressions.
    db, trie = world
    diff = grade_passthrough_diff(
        db,
        trie,
        passthrough_map={"nonexistent-cluster": ("a", "b")},
        culture_languages=None,
        genitive_prior=None,
        connective_inventory=None,
    )
    assert not diff.improvements and not diff.regressions
    assert diff.on.mean_recall == diff.off.mean_recall
    assert diff.on.coverage_rate == diff.off.coverage_rate
