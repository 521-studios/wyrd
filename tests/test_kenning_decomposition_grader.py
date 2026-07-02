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
    MatcherConfig,
    ToponymGrade,
    diff_grades,
    grade_can_it,
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
            config=MatcherConfig(),
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
            config=MatcherConfig(
                connective_inventory=DEFAULT_CONNECTIVE_INVENTORY,
                genitive_prior={},  # no homograph tiebreak
            ),
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
            config=MatcherConfig(),
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
    # No connective_inventory in the config: grade_corpus_diff defaults the ON
    # side to DEFAULT_CONNECTIVE_INVENTORY (the pre-wyrd-buye signature default),
    # so this still diffs the connective rather than producing a no-op OFF==ON.
    diff = grade_corpus_diff(
        db,
        trie,
        config=MatcherConfig(genitive_prior=TOWN_PRIOR),
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


def test_diff_grades_tradeoff_lands_on_both_lists():
    """wyrd-gzjo: diff_grades classifies a config change PER DIMENSION, so a
    single toponym that improves one metric (recall up) while regressing another
    (precision down) lands on BOTH the regressions and improvements lists — the
    trade-off case the test corpus's separate-toponym cases don't exercise."""
    common = {"unaccounted": 0, "exact": False, "coverage": True, "head_attested": True}
    off = [ToponymGrade(name="Tradeoff", morphemes=("a",), recall=0.5, precision=1.0, **common)]
    # on: recall rises (0.5 -> 1.0), precision falls (1.0 -> 0.5) — a genuine trade-off.
    on = [ToponymGrade(name="Tradeoff", morphemes=("a", "b"), recall=1.0, precision=0.5, **common)]

    diff = diff_grades(off, on)
    regressed = {c.name: c for c in diff.regressions}
    improved = {c.name: c for c in diff.improvements}

    assert "Tradeoff" in regressed
    assert "Tradeoff" in improved
    # Exact: ONLY precision worsened and ONLY recall improved — the other three
    # dimensions held flat, so neither list over-reports a moved metric.
    assert regressed["Tradeoff"].dimensions == ("precision",)
    assert improved["Tradeoff"].dimensions == ("recall",)
    # The record carries the same off/on parse on both lists.
    assert regressed["Tradeoff"].on_parse == ("a", "b")
    assert improved["Tradeoff"].off_parse == ("a",)


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
        config=MatcherConfig(),
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
        config=MatcherConfig(),
    )
    assert not diff.improvements and not diff.regressions
    assert diff.on.mean_recall == diff.off.mean_recall
    assert diff.on.coverage_rate == diff.off.coverage_rate


def test_grade_can_it_separates_generated_from_selected(world):
    """CAN-IT (the decomposer GENERATES a scholar-recovering parse) is a ceiling
    above DID-WE (it SELECTS it). Bishopston is the canonical gap: bishop+ston
    (stone) outscores the bishop+[s]+ton parse that recovers the scholar's tūn,
    so the right parse is generated but not selected."""
    db, trie = world
    index = load_cluster_index(db)
    corpus = load_scholar_corpus(db, index)  # Bishopston, Grimsworth, Rudston
    # No connective inventory → the selected parse is always a plain candidate, so
    # DID-WE ⊆ CAN-IT and the gap identity holds exactly.
    s = grade_can_it(corpus, trie, index, config=MatcherConfig(), progress_every=0)

    assert s.graded == 3
    # All three scholar breakdowns are generable from the inventory (ceiling).
    assert s.can_it_recall1 == 3
    # Selection leaves at least Bishopston on the floor.
    assert s.did_we_recall1 < s.can_it_recall1
    assert s.gap_recall1 >= 1
    # GAP is exactly the generated-but-not-selected difference (DID-WE ⊆ CAN-IT).
    assert s.gap_recall1 == s.can_it_recall1 - s.did_we_recall1
    # Exact-equality is the stricter sibling: never exceeds recall=1.
    assert s.can_it_exact <= s.can_it_recall1
    assert s.did_we_exact <= s.did_we_recall1


def test_report_can_it_handles_zero_graded(capsys):
    """``_report_can_it``'s ``graded or 1`` guard prevents a ZeroDivisionError
    when the corpus produced no gradable toponyms."""
    from wyrd.generators.kenning.cli.lexicon.grade_decomposition import _report_can_it
    from wyrd.generators.kenning.lexicon.decomposition_grader import CanItSummary

    _report_can_it(
        CanItSummary(
            graded=0,
            product_capped=0,
            can_it_exact=0,
            did_we_exact=0,
            gap_exact=0,
            can_it_recall1=0,
            did_we_recall1=0,
            gap_recall1=0,
        )
    )
    err = capsys.readouterr().err
    assert "CAN-IT vs DID-WE over 0 toponyms" in err


def _merge(db, loser_id, winner_id):
    """Tombstone ``loser_id`` into ``winner_id`` (the D22 merged_into_id redirect)
    WITHOUT repointing its reflex / breakdown rows — the exact live-DB shape
    wyrd-qdau guards against (a merge sets merged_into_id on the loser but leaves
    reflex_etymon / toponym_etymology_element pointing at the dead id)."""
    db.conn.execute(
        "UPDATE etymon SET merged_into_id = ?, cognate_id = NULL WHERE id = ?",
        (winner_id, loser_id),
    )


def test_cluster_index_excludes_merged_losers(tmp_path):
    """wyrd-qdau: load_cluster_index / load_scholar_corpus must filter
    merged_into_id IS NULL like every sibling index-builder. A merged loser
    (cognate_id NULL) whose reflex + breakdown rows still point at it must NOT
    yield a stale singleton ``e<loser>`` key that fractures the surviving
    cluster, and its breakdown must not pool the dead identity (nor KeyError)."""
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    db = LexiconDB(db_path)
    db.conn.execute("INSERT INTO source (id, title) VALUES ('test_src', 'Test')")

    tun = _etymon(db, "tūn")  # merge WINNER (own cognate cluster)
    tun_dup = _etymon(db, "tun", cluster=False)  # duplicate, cognate_id NULL

    # ONE shared 'ton' reflex with TWO etymon links (winner + loser) — the
    # tūn/ton cognate bridge whose evidence should pool onto a single cluster.
    rid = db.conn.execute(
        "INSERT INTO reflex (surface_form, position) VALUES ('ton', 'post')"
    ).lastrowid
    db.conn.execute("INSERT INTO reflex_etymon (reflex_id, etymon_id) VALUES (?, ?)", (rid, tun))
    db.conn.execute(
        "INSERT INTO reflex_etymon (reflex_id, etymon_id) VALUES (?, ?)", (rid, tun_dup)
    )

    _toponym(db, "Merton", [tun_dup])  # breakdown anchored solely on the loser
    _merge(db, tun_dup, tun)
    db.commit()

    index = load_cluster_index(db)
    # The loser's dead identity is absent — no stale e<loser> singleton, so the
    # shared 'ton' surface resolves to the winner cluster ALONE (not fractured
    # into {c<tun>, e<tun_dup>}).
    assert tun_dup not in index.cluster_of
    assert index.clusters_for("ton") == frozenset({f"c{tun}"})
    assert f"e{tun_dup}" not in index.clusters_for("ton")

    # A breakdown anchored only on the loser yields no cluster → dropped
    # (not a KeyError, not the loser's dead cluster).
    corpus = load_scholar_corpus(db, index)
    assert "Merton" not in {sc.name for sc in corpus}
    db.close()
