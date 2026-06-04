"""Same-language variant-FOLD candidate detection (empty-grid root fix).

Sibling of test_rogd15_reflex_detect: that covers the cross-era reflex LINK;
this covers the same-language barren↔rich FOLD that collapses path-dependent
duplicates so a generated morpheme stops resolving to an empty-grid duplicate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from wyrd.generators.kenning.lexicon import LexiconDB, init_schema
from wyrd.generators.kenning.lexicon.variant_fold_detect import (
    _consonant_skeleton,
    _gloss_tokens,
    detect_variant_fold_candidates,
)


@pytest.fixture
def fresh_db(tmp_path: Path) -> Path:
    p = tmp_path / "lexicon.db"
    init_schema(p)
    return p


def _e(db: LexiconDB, form: str, lang: str, *glosses: str) -> int:
    eid = db.upsert_etymon(form, lang, modifier_type="Habitative")
    for g in glosses:
        db.add_gloss(eid, g)
    return eid


def _rich(db: LexiconDB, form: str, lang: str, *glosses: str, cluster: int = 6) -> int:
    """A well-attested lemma: real etymon + (cluster-1) cognate fillers all
    sharing one cognate_id (etymon.cognate_id is a self-FK to an etymon id), so
    the lemma's cluster_size clears the lemma_min richness floor."""
    eid = _e(db, form, lang, *glosses)
    db.conn.execute("UPDATE etymon SET cognate_id = ? WHERE id = ?", (eid, eid))
    for i in range(cluster - 1):
        mate = _e(db, f"{form}~mate{i}", lang, glosses[0] if glosses else "x")
        db.conn.execute("UPDATE etymon SET cognate_id = ? WHERE id = ?", (eid, mate))
    return eid


# ---- detection -------------------------------------------------------------


def test_bare_gloss_folds_into_descriptive_lemma(fresh_db: Path) -> None:
    # The exact-gloss miss that token grouping fixes: barren 'stone' is glossed
    # just "stone"; the rich lemma 'stān' is glossed descriptively. They share
    # the *token* 'stone', so the barren reaches the lemma.
    with LexiconDB(fresh_db) as db:
        _e(db, "stone", "old-english", "stone")
        _rich(db, "stān", "old-english", "A stone, stone, rock.")
        db.commit()
        cands = detect_variant_fold_candidates(db.conn, min_similarity=0.6)
        stone = [c for c in cands if c.barren_ref == "old-english:stone"]
        assert len(stone) == 1
        assert stone[0].lemma_ref == "old-english:stān"
        assert stone[0].lemma_cluster_size >= 5


def test_vowel_shift_folds_via_consonant_skeleton(fresh_db: Path) -> None:
    # wood↔wudu: difflib ratio ~0.5 (below the floor) but a shared consonant
    # skeleton 'wd' rescues the pair.
    with LexiconDB(fresh_db) as db:
        _e(db, "wood", "old-english", "wood")
        _rich(db, "wudu", "old-english", "wood, a forest")
        db.commit()
        cands = detect_variant_fold_candidates(db.conn, min_similarity=0.7)
        wood = [c for c in cands if c.barren_ref == "old-english:wood"]
        assert len(wood) == 1 and wood[0].lemma_ref == "old-english:wudu"
        assert wood[0].similarity < 0.7  # admitted purely on the skeleton path


def test_needs_one_barren_and_one_rich(fresh_db: Path) -> None:
    # two rich same-gloss lemmas (no barren) → nothing to fold away.
    with LexiconDB(fresh_db) as db:
        _rich(db, "burg", "old-english", "fortified place")
        _rich(db, "burh", "old-english", "fortified place")
        db.commit()
        assert detect_variant_fold_candidates(db.conn, min_similarity=0.6) == []


def test_cross_language_is_not_folded(fresh_db: Path) -> None:
    # OE barren + ME rich sharing a gloss = a cross-era reflex LINK (rogd.15),
    # never a same-language FOLD.
    with LexiconDB(fresh_db) as db:
        _e(db, "hill", "old-english", "hill")
        _rich(db, "hill", "middle-english", "hill")
        db.commit()
        assert detect_variant_fold_candidates(db.conn, min_similarity=0.6) == []


def test_already_edged_pair_is_excluded(fresh_db: Path) -> None:
    with LexiconDB(fresh_db) as db:
        barren = _e(db, "hill", "old-english", "hill")
        lemma = _rich(db, "hyll", "old-english", "hill")
        db.upsert_source(id="s", title="s")
        db.conn.execute(
            "INSERT INTO etymon_descent (parent_id, child_id, edge_type, source_id) "
            "VALUES (?, ?, 'inheritance', 's')",
            (lemma, barren),
        )
        db.commit()
        assert detect_variant_fold_candidates(db.conn, min_similarity=0.6) == []


def test_one_best_lemma_per_barren_richest_wins(fresh_db: Path) -> None:
    with LexiconDB(fresh_db) as db:
        _e(db, "hill", "old-english", "hill")
        _rich(db, "hyll", "old-english", "hill", cluster=6)
        _rich(db, "hil", "old-english", "hill", cluster=20)  # richer
        db.commit()
        cands = detect_variant_fold_candidates(db.conn, min_similarity=0.6)
        hill = [c for c in cands if c.barren_ref == "old-english:hill"]
        assert len(hill) == 1
        assert hill[0].lemma_ref == "old-english:hil"  # the richer cluster wins


# ---- helpers ---------------------------------------------------------------


def test_gloss_tokens_splits_and_filters() -> None:
    assert "stone" in _gloss_tokens("A stone, stone, rock.")
    assert "rock" in _gloss_tokens("A stone, stone, rock.")
    assert _gloss_tokens("the and for") == set()  # stopwords
    assert _gloss_tokens("of a an") == set()  # length < 3
    assert _gloss_tokens("hill") == {"hill"}


def test_consonant_skeleton_survives_vowel_shift() -> None:
    assert _consonant_skeleton("wood") == _consonant_skeleton("wudu") == "wd"
    assert _consonant_skeleton("hūs") == _consonant_skeleton("house") == "hs"
    assert _consonant_skeleton("dūn") == _consonant_skeleton("doun") == "dn"
    # diacritics stripped, dashes/stars stripped, vowels (incl. y) removed
    assert _consonant_skeleton("-stān*") == "stn"


# ---- verdict round-trip ----------------------------------------------------


def _cand():
    from wyrd.generators.kenning.lexicon.variant_fold_detect import VariantFoldCandidate

    return VariantFoldCandidate(
        barren_ref="old-english:stone",
        lemma_ref="old-english:stān",
        barren_form="stone",
        lemma_form="stān",
        shared_glosses=("stone",),
        similarity=0.67,
        lemma_cluster_size=81,
        lemma_glosses=("A stone, stone, rock.",),
    )


def test_build_fold_prompt_shows_both_glosses() -> None:
    from wyrd.generators.kenning.lexicon.variant_fold_detect import build_fold_judge_prompt

    system, user = build_fold_judge_prompt(_cand())
    assert "spelling/accent variant" in system.lower() or "SPELLING/ACCENT" in system
    assert "stone" in user and "stān" in user
    assert "A stone, stone, rock." in user  # the lemma's descriptive gloss


def test_parse_fold_verdict() -> None:
    from wyrd.generators.kenning.lexicon.variant_fold_detect import parse_fold_verdict

    v = parse_fold_verdict({"same_morpheme": True, "confidence": "high", "reason": "variant"})
    assert v.same_morpheme and v.confidence == "high"
    assert parse_fold_verdict({"same_morpheme": "yes", "confidence": "??"}).confidence == "low"
    assert parse_fold_verdict({"same_morpheme": "no"}).same_morpheme is False
    assert parse_fold_verdict({"confidence": "high"}) is None
    assert parse_fold_verdict("nope") is None


def test_fold_verdict_to_row_records_candidate_lemma() -> None:
    from wyrd.generators.kenning.lexicon.variant_fold_detect import (
        LLM_VARIANT_FOLD_METHOD,
        LLM_VARIANT_REJECT_METHOD,
        VariantFoldVerdict,
        fold_verdict_to_row,
    )

    c = _cand()
    fold = fold_verdict_to_row(c, VariantFoldVerdict(True, "high", "v"), "medium")
    assert fold["ref"] == "old-english:stone" and fold["into"] == "old-english:stān"
    assert fold["method"] == LLM_VARIANT_FOLD_METHOD
    assert fold["candidate_lemma"] == "old-english:stān"
    # reject still records candidate_lemma so re-runs dedup at the PAIR level
    rej = fold_verdict_to_row(c, VariantFoldVerdict(False, "high", "homograph"), "medium")
    assert rej["into"] == "" and rej["method"] == LLM_VARIANT_REJECT_METHOD
    assert rej["candidate_lemma"] == "old-english:stān"
    # below min confidence → no-op
    low = fold_verdict_to_row(c, VariantFoldVerdict(True, "low", "maybe"), "high")
    assert low["into"] == ""


# ---- CLI end-to-end --------------------------------------------------------


def test_cli_fold_variants_writes_into_rows(fresh_db: Path, tmp_path: Path, monkeypatch) -> None:
    import json as _json

    from click.testing import CliRunner

    from wyrd.generators.kenning.cli.lexicon.fold_variants import lexicon_fold_variants

    with LexiconDB(fresh_db) as db:
        _e(db, "stone", "old-english", "stone")
        _rich(db, "stān", "old-english", "A stone, stone, rock.")
        db.commit()

    class _StubClient:
        def __init__(self, *a, **k):
            pass

        def chat_json(self, system, user, schema):
            return {"same_morpheme": True, "confidence": "high", "reason": "stub"}

    monkeypatch.setattr("wyrd.generators.kenning.extractors.llm.OllamaClient", _StubClient)
    ledger = tmp_path / "_collapses.jsonl"
    res = CliRunner().invoke(
        lexicon_fold_variants,
        ["--db", str(fresh_db), "--collapse-file", str(ledger), "--min-similarity", "0.6"],
    )
    assert res.exit_code == 0, res.output
    rows = [
        _json.loads(line)
        for line in ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    fold = next(r for r in rows if r.get("into"))
    assert fold["ref"] == "old-english:stone" and fold["into"] == "old-english:stān"

    # second invocation: the (barren, lemma) pair is already judged → no new rows
    res2 = CliRunner().invoke(
        lexicon_fold_variants,
        ["--db", str(fresh_db), "--collapse-file", str(ledger), "--min-similarity", "0.6"],
    )
    assert res2.exit_code == 0
    rows2 = [line for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows2) == len(rows)  # pair-level dedup held
