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
