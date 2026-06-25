"""Cross-cluster same-meaning MERGE detection (v3ow P3).

Sibling of test_variant_fold_detect: that covers the BARREN fold; this covers
the minor↔major cross-cluster merge that unifies a sparse parallel lineage
(OE ``field``) into its rich canonical cluster (OE ``feld``) so a generated
surface stops dead-ending short of the modern era.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wyrd.generators.kenning.lexicon import LexiconDB, init_schema
from wyrd.generators.kenning.lexicon.meaning_merge_detect import (
    detect_meaning_merge_candidates,
)


@pytest.fixture
def fresh_db(tmp_path: Path) -> Path:
    p = tmp_path / "lexicon.db"
    init_schema(p)
    return p


def _e(db: LexiconDB, form: str, lang: str, *glosses: str) -> int:
    eid = db.upsert_etymon(form, lang, modifier_type="Topographical")
    for g in glosses:
        db.add_gloss(eid, g)
    return eid


def _cluster(db: LexiconDB, form: str, lang: str, *glosses: str, size: int) -> int:
    """A cluster of `size` etymons sharing one cognate_id (self-FK to the head).
    Only the HEAD carries the meaningful gloss(es); the cognate fillers get a
    distinct filler gloss so they count toward cluster richness but don't land in
    the head's gloss-token group (real cognate mates are distinct forms, not
    near-duplicates of the head — modelling that keeps the test honest)."""
    head = _e(db, form, lang, *glosses)
    db.conn.execute("UPDATE etymon SET cognate_id = ? WHERE id = ?", (head, head))
    # Filler gloss + form carry a cluster-unique token ("<form>only") so the
    # fillers never share a gloss-token group with the other cluster — they add
    # richness only, leaving the HEAD forms as the sole cross-cluster pair.
    for i in range(size - 1):
        mate = _e(db, f"{form}only{i}", lang, f"{form}only")
        db.conn.execute("UPDATE etymon SET cognate_id = ? WHERE id = ?", (head, mate))
    return head


def test_minor_cluster_merges_into_dominant_major(fresh_db: Path) -> None:
    # field (cluster 6) and feld (cluster 60) — same lang, same gloss token,
    # form-similar, distinct cognates, major ≥ 5× minor → a merge candidate.
    with LexiconDB(fresh_db) as db:
        _cluster(db, "field", "old-english", "open land, field", size=6)
        _cluster(db, "feld", "old-english", "open country, field", size=60)
        db.commit()
        cands = detect_meaning_merge_candidates(db.conn, min_similarity=0.6)
        field = [c for c in cands if c.minor_ref == "old-english:field"]
        assert len(field) == 1
        assert field[0].major_ref == "old-english:feld"
        assert field[0].minor_cluster_size == 6 and field[0].major_cluster_size == 60


def test_barren_minor_is_not_a_merge_candidate(fresh_db: Path) -> None:
    # cluster size 1 is the BARREN fold's job (variant_fold_detect), not here —
    # the merge floor is 2.
    with LexiconDB(fresh_db) as db:
        _e(db, "field", "old-english", "open land, field")  # cluster 1 (barren)
        _cluster(db, "feld", "old-english", "open country, field", size=60)
        db.commit()
        assert detect_meaning_merge_candidates(db.conn, min_similarity=0.6) == []


def test_ratio_gate_blocks_comparable_clusters(fresh_db: Path) -> None:
    # major only ~2× the minor → not a clear dominant; no merge (avoid merging
    # two comparably-attested, possibly-distinct morphemes).
    with LexiconDB(fresh_db) as db:
        _cluster(db, "field", "old-english", "open land, field", size=6)
        _cluster(db, "feld", "old-english", "open country, field", size=12)  # < 5×6
        db.commit()
        assert detect_meaning_merge_candidates(db.conn, min_similarity=0.6, major_min=10) == []


def test_equal_size_clusters_not_bidirectional_at_low_ratio(fresh_db: Path) -> None:
    # Regression: two same-language EQUAL-size clusters sharing a gloss token + form.
    # With overlapping bands (major_min ≤ minor_max) and ratio=1 the old dominance
    # guard `richness < ratio*rmi` reduced to `richness < rmi`, so an equal-size pair
    # passed in BOTH directions (mere→meer AND meer→mere) — two conflicting merges
    # each tombstoning the other. Strict dominance (major must be STRICTLY richer)
    # rejects the equal-size pair entirely, in either direction.
    with LexiconDB(fresh_db) as db:
        _cluster(db, "mere", "old-english", "pool, lake", size=5)
        _cluster(db, "meer", "old-english", "pool, lake", size=5)  # equal size, shared token
        db.commit()
        cands = detect_meaning_merge_candidates(
            db.conn, min_similarity=0.6, minor_max=10, major_min=5, ratio=1
        )
        assert cands == []  # no clear major → no candidate either way


def test_same_cluster_is_not_a_candidate(fresh_db: Path) -> None:
    # two forms already in ONE cognate cluster are not a cross-cluster merge.
    with LexiconDB(fresh_db) as db:
        head = _cluster(db, "feld", "old-english", "field", size=60)
        field = _e(db, "field", "old-english", "field")
        db.conn.execute("UPDATE etymon SET cognate_id = ? WHERE id = ?", (head, field))
        db.commit()
        assert detect_meaning_merge_candidates(db.conn, min_similarity=0.6) == []


def test_already_edged_pair_is_excluded(fresh_db: Path) -> None:
    with LexiconDB(fresh_db) as db:
        minor = _cluster(db, "field", "old-english", "field", size=6)
        major = _cluster(db, "feld", "old-english", "field", size=60)
        db.upsert_source(id="s", title="s")
        db.conn.execute(
            "INSERT INTO etymon_descent (parent_id, child_id, edge_type, source_id) "
            "VALUES (?, ?, 'inheritance', 's')",
            (major, minor),
        )
        db.commit()
        assert detect_meaning_merge_candidates(db.conn, min_similarity=0.6) == []


def test_cross_language_is_not_a_same_language_merge(fresh_db: Path) -> None:
    with LexiconDB(fresh_db) as db:
        _cluster(db, "field", "middle-english", "field", size=6)
        _cluster(db, "feld", "old-english", "field", size=60)
        db.commit()
        assert detect_meaning_merge_candidates(db.conn, min_similarity=0.6) == []


def test_skeleton_admits_sub_similarity_vowel_shift(fresh_db: Path) -> None:
    # wode↔wudu: difflib ~0.5 (below the floor) but a shared 'wd' skeleton
    # rescues the pair — exercises the skeleton-only admission branch.
    with LexiconDB(fresh_db) as db:
        _cluster(db, "wode", "old-english", "wood, forest", size=6)
        _cluster(db, "wudu", "old-english", "wood, a forest", size=60)
        db.commit()
        cands = detect_meaning_merge_candidates(db.conn, min_similarity=0.7)
        wode = [c for c in cands if c.minor_ref == "old-english:wode"]
        assert len(wode) == 1 and wode[0].major_ref == "old-english:wudu"
        assert wode[0].similarity < 0.7  # admitted purely on the skeleton path


def test_neither_similar_nor_skeleton_is_excluded(fresh_db: Path) -> None:
    # same gloss token + ratio, but form-dissimilar AND different skeleton → out.
    with LexiconDB(fresh_db) as db:
        _cluster(db, "leah", "old-english", "clearing, meadow", size=6)
        _cluster(db, "feld", "old-english", "clearing, meadow", size=60)
        db.commit()
        assert detect_meaning_merge_candidates(db.conn, min_similarity=0.7) == []


def test_tie_break_picks_lowest_id_major(fresh_db: Path) -> None:
    # two equally-rich, equally-similar majors for one minor → the total
    # (richness, sim, -id) tie-break picks the lowest-id one (feld, inserted
    # first), reproducibly across runs.
    with LexiconDB(fresh_db) as db:
        _cluster(db, "feld", "old-english", "field", size=60)  # lower id
        _cluster(db, "fild", "old-english", "field", size=60)
        _cluster(db, "field", "old-english", "field", size=6)
        db.commit()
        picks = set()
        field_major = None
        for _ in range(5):
            cands = detect_meaning_merge_candidates(db.conn, min_similarity=0.6)
            field = [c for c in cands if c.minor_ref == "old-english:field"]
            assert len(field) == 1  # one best major per minor
            field_major = field[0].major_ref
            picks.add(field_major)
        assert picks == {"old-english:feld"}  # lowest-id major, stable across runs


# ---- CLI: dry-run, failure-skip, terminal-merge ----------------------------


class _Stub:
    """An OllamaClient stand-in with a scripted ``chat_json``."""

    def __init__(self, fn):
        self._fn = fn

    def __call__(self, *a, **k):  # OllamaClient(...) constructor
        return self

    def chat_json(self, system, user, schema):
        return self._fn()


def _seed_field(db_path: Path) -> None:
    with LexiconDB(db_path) as db:
        _cluster(db, "field", "old-english", "open land, field", size=6)
        _cluster(db, "feld", "old-english", "open country, field", size=60)
        db.commit()


def _run(monkeypatch, stub, db_path, ledger, *extra):
    from click.testing import CliRunner

    from wyrd.generators.kenning.cli.lexicon.merge_lemmas import lexicon_merge_lemmas

    monkeypatch.setattr("wyrd.generators.kenning.extractors.llm.OllamaClient", stub)
    return CliRunner().invoke(
        lexicon_merge_lemmas,
        ["--db", str(db_path), "--collapse-file", str(ledger), "--min-similarity", "0.6", *extra],
    )


def test_cli_merge_writes_into_row(fresh_db: Path, tmp_path: Path, monkeypatch) -> None:
    import json as _json

    _seed_field(fresh_db)
    ledger = tmp_path / "_collapses.jsonl"
    stub = _Stub(lambda: {"same_morpheme": True, "confidence": "high", "reason": "stub"})
    res = _run(monkeypatch, stub, fresh_db, ledger)
    assert res.exit_code == 0, res.output
    rows = [_json.loads(x) for x in ledger.read_text().splitlines() if x.strip()]
    merge = next(r for r in rows if r.get("into"))
    assert merge["ref"] == "old-english:field" and merge["into"] == "old-english:feld"
    assert merge["method"] == "llm-meaning-merge-v1"


def test_cli_dry_run_writes_nothing(fresh_db: Path, tmp_path: Path, monkeypatch) -> None:
    _seed_field(fresh_db)
    ledger = tmp_path / "_collapses.jsonl"
    stub = _Stub(lambda: {"same_morpheme": True, "confidence": "high", "reason": "stub"})
    res = _run(monkeypatch, stub, fresh_db, ledger, "--dry-run")
    assert res.exit_code == 0, res.output
    assert "(dry-run)" in res.output
    assert not ledger.exists()


def test_cli_skips_on_llm_failure_without_crashing(
    fresh_db: Path, tmp_path: Path, monkeypatch
) -> None:
    _seed_field(fresh_db)
    ledger = tmp_path / "_collapses.jsonl"

    def _boom():
        raise RuntimeError("transport down")

    res = _run(monkeypatch, _Stub(_boom), fresh_db, ledger)
    assert res.exit_code == 0, res.output
    assert "1 skipped" in res.output and "[skip]" in res.output
    assert not ledger.exists() or "into" not in ledger.read_text()


def test_cli_merge_is_terminal(fresh_db: Path, tmp_path: Path, monkeypatch) -> None:
    # A minor already merged must NOT be re-judged — no later reject can cancel
    # it at replay. Pre-seed a merge row; a reject stub must not be consulted.
    import json as _json

    _seed_field(fresh_db)
    ledger = tmp_path / "_collapses.jsonl"
    ledger.write_text(
        _json.dumps(
            {
                "_type": "collapse",
                "ref": "old-english:field",
                "into": "old-english:feld",
                "candidate_lemma": "old-english:feld",
                "method": "llm-meaning-merge-v1",
            }
        )
        + "\n"
    )
    before = ledger.read_text()
    stub = _Stub(lambda: {"same_morpheme": False, "confidence": "high", "reason": "no"})
    res = _run(monkeypatch, stub, fresh_db, ledger)
    assert res.exit_code == 0, res.output
    assert ledger.read_text() == before  # untouched — terminal merge held


# ---- verdict round-trip ----------------------------------------------------


def _cand():
    from wyrd.generators.kenning.lexicon.meaning_merge_detect import MeaningMergeCandidate

    return MeaningMergeCandidate(
        minor_ref="old-english:field",
        major_ref="old-english:feld",
        minor_form="field",
        major_form="feld",
        minor_glosses=("open land, field", "fold, crease"),
        major_glosses=("open country, field",),
        similarity=0.889,
        minor_cluster_size=6,
        major_cluster_size=224,
    )


def test_build_merge_prompt_frames_minor_major_and_homograph() -> None:
    from wyrd.generators.kenning.lexicon.meaning_merge_detect import build_merge_judge_prompt

    system, user = build_merge_judge_prompt(_cand())
    assert "HOMOGRAPH" in system or "homograph" in system.lower()
    assert "field" in user and "feld" in user
    assert "fold, crease" in user  # the minor's homograph sense is shown to the judge


def test_parse_merge_verdict() -> None:
    from wyrd.generators.kenning.lexicon.meaning_merge_detect import parse_merge_verdict

    v = parse_merge_verdict({"same_morpheme": True, "confidence": "high", "reason": "variant"})
    assert v.same_morpheme and v.confidence == "high"
    assert parse_merge_verdict({"same_morpheme": "no"}).same_morpheme is False
    assert parse_merge_verdict({"confidence": "high"}) is None
    assert parse_merge_verdict("nope") is None


def test_merge_verdict_to_row_records_candidate_lemma() -> None:
    from wyrd.generators.kenning.lexicon.meaning_merge_detect import (
        LLM_MEANING_MERGE_METHOD,
        LLM_MEANING_REJECT_METHOD,
        MeaningMergeVerdict,
        merge_verdict_to_row,
    )

    c = _cand()
    merge = merge_verdict_to_row(c, MeaningMergeVerdict(True, "high", "v"), "medium")
    assert merge["ref"] == "old-english:field" and merge["into"] == "old-english:feld"
    assert merge["method"] == LLM_MEANING_MERGE_METHOD
    assert merge["candidate_lemma"] == "old-english:feld"
    rej = merge_verdict_to_row(c, MeaningMergeVerdict(False, "high", "distinct"), "medium")
    assert rej["into"] == "" and rej["method"] == LLM_MEANING_REJECT_METHOD
    assert rej["candidate_lemma"] == "old-english:feld"  # pair-dedup even on reject


def test_best_major_prefers_richest_cluster_over_more_similar(fresh_db: Path) -> None:
    """When a minor has two eligible majors, the RICHER cluster wins even if a
    poorer major is more form-similar — richness is the first sort key (pins
    _update_best_major's primary discriminator)."""
    with LexiconDB(fresh_db) as db:
        _cluster(db, "field", "old-english", "open land, field", size=6)
        # richer (60) but LESS similar to 'field'; poorer (40) but MORE similar.
        _cluster(db, "feldx", "old-english", "country, field", size=60)
        _cluster(db, "feld", "old-english", "meadow, field", size=40)
        db.commit()
        field = [
            c
            for c in detect_meaning_merge_candidates(db.conn, min_similarity=0.6)
            if c.minor_ref == "old-english:field"
        ]
        assert len(field) == 1
        assert field[0].major_ref == "old-english:feldx"  # richest wins over more-similar
        assert field[0].major_cluster_size == 60
