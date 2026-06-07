"""wyrd-vewk — import curated modern-English reflexes, end to end.

Real-DB (D10): a clustered morpheme with NO modern reflex gets one via the
importer, and ``etymon_era_reflexes`` then surfaces it in the modern stage —
the whole point (era-grid modern cell fills). Plus: 'none' rows are skipped,
the run is idempotent, and the descent edge is recorded.
"""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from wyrd.generators.kenning.cli.lexicon.import_modern_reflexes import (
    lexicon_import_modern_reflexes,
)
from wyrd.generators.kenning.lexicon.db import LexiconDB
from wyrd.generators.kenning.lexicon.era_reflex import etymon_era_reflexes
from wyrd.generators.kenning.lexicon.modern_reflex_import import (
    MODERN_REFLEX_SOURCE,
    import_modern_reflexes,
)
from wyrd.generators.kenning.lexicon.schema import init_schema


def _clustered_morpheme(db: LexiconDB, canonical: str, language: str) -> int:
    """Insert a morpheme that sits in a cognate cluster (cognate_id = own id,
    a valid 1-member cluster) so the importer's Tier-1 cluster-join path fires."""
    eid = db.upsert_etymon(canonical, language)
    db.conn.execute("UPDATE etymon SET cognate_id=? WHERE id=?", (eid, eid))
    db.commit()
    return eid


def _write(path: Path, *rows: dict) -> Path:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    return path


def test_import_fills_modern_era_reflex_for_clustered_morpheme(tmp_path):
    init_schema(tmp_path / "lexicon.db")
    db = LexiconDB(tmp_path / "lexicon.db")
    ham = _clustered_morpheme(db, "hām", "old-english")
    # pre: no curated modern reflex in the cluster (a Tier-4 phonology-rule
    # form may be synthesized, but 'home'/'ham' are absent).
    pre = {r.form for r in etymon_era_reflexes(db, ham, target_language="modern-english")}
    assert "home" not in pre and "ham" not in pre

    f = _write(
        tmp_path / "_modern_reflexes.jsonl",
        {"_type": "source", "ref": "modern-reflex-curation", "title": "t"},
        {
            "_type": "modern_reflex",
            "etymon_ref": "old-english:hām",
            "gloss": "homestead",
            "modern_forms": ["home", "ham"],
            "confidence": "high",
            "reference": "OED",
        },
    )
    counts = import_modern_reflexes(db, f, apply=True)
    assert counts["reflex_created"] == 2
    assert counts["clustered"] == 2

    # post: the era-grid modern stage now surfaces the curated reflexes.
    forms = {r.form for r in etymon_era_reflexes(db, ham, target_language="modern-english")}
    assert forms == {"home", "ham"}
    # the new etymon carries the gloss + a descent edge from the morpheme.
    assert (
        db.conn.execute(
            "SELECT gloss FROM etymon_gloss g JOIN etymon e ON e.id=g.etymon_id "
            "WHERE e.canonical_form='home' AND e.language='modern-english'"
        ).fetchone()[0]
        == "homestead"
    )
    assert (
        db.conn.execute(
            "SELECT COUNT(*) FROM etymon_descent WHERE source_id=? AND edge_type='inheritance'",
            (MODERN_REFLEX_SOURCE,),
        ).fetchone()[0]
        == 2
    )


def test_none_rows_skipped_and_idempotent(tmp_path):
    init_schema(tmp_path / "lexicon.db")
    db = LexiconDB(tmp_path / "lexicon.db")
    _clustered_morpheme(db, "skógr", "old-norse")
    ham = _clustered_morpheme(db, "hām", "old-english")
    f = _write(
        tmp_path / "_modern_reflexes.jsonl",
        {
            "_type": "modern_reflex",
            "etymon_ref": "old-norse:skógr",
            "modern_forms": [],
            "confidence": "none",
            "reference": "no reflex",
        },
        {
            "_type": "modern_reflex",
            "etymon_ref": "old-english:hām",
            "modern_forms": ["home"],
            "confidence": "high",
            "reference": "OED",
        },
    )
    c1 = import_modern_reflexes(db, f, apply=True)
    assert c1["skipped_no_reflex"] == 1 and c1["reflex_created"] == 1
    # idempotent: a second run creates nothing new, no duplicate descent edges.
    c2 = import_modern_reflexes(db, f, apply=True)
    assert c2.get("reflex_created", 0) == 0
    assert (
        db.conn.execute(
            "SELECT COUNT(*) FROM etymon_descent WHERE source_id=?", (MODERN_REFLEX_SOURCE,)
        ).fetchone()[0]
        == 1
    )
    assert {r.form for r in etymon_era_reflexes(db, ham, target_language="modern-english")} == {
        "home"
    }


def test_dry_run_writes_nothing(tmp_path):
    init_schema(tmp_path / "lexicon.db")
    db = LexiconDB(tmp_path / "lexicon.db")
    ham = _clustered_morpheme(db, "hām", "old-english")
    f = _write(
        tmp_path / "_modern_reflexes.jsonl",
        {
            "_type": "modern_reflex",
            "etymon_ref": "old-english:hām",
            "modern_forms": ["home"],
            "confidence": "high",
            "reference": "OED",
        },
    )
    counts = import_modern_reflexes(db, f, apply=False)
    assert counts.get("reflex_would_create") == 1  # dry-run reports, writes nothing
    assert counts.get("would_link") == 1  # total links that would be made
    assert "home" not in {
        r.form for r in etymon_era_reflexes(db, ham, target_language="modern-english")
    }


def test_unclustered_morpheme_fills_via_tier2_descent(tmp_path):
    """A morpheme with NO cognate cluster relies on the descent edge: the
    importer records descent_only and etymon_era_reflexes Tier-2 surfaces it."""
    init_schema(tmp_path / "lexicon.db")
    db = LexiconDB(tmp_path / "lexicon.db")
    # NOT clustered — leave cognate_id NULL.
    eid = db.upsert_etymon("creca", "old-english")
    db.commit()
    f = _write(
        tmp_path / "_modern_reflexes.jsonl",
        {
            "_type": "modern_reflex",
            "etymon_ref": "old-english:creca",
            "modern_forms": ["creek"],
            "confidence": "high",
            "reference": "OED",
        },
    )
    counts = import_modern_reflexes(db, f, apply=True)
    assert counts["descent_only"] == 1 and counts.get("clustered", 0) == 0
    assert "creek" in {
        r.form for r in etymon_era_reflexes(db, eid, target_language="modern-english")
    }


def test_morpheme_missing_is_counted_and_skipped(tmp_path):
    init_schema(tmp_path / "lexicon.db")
    db = LexiconDB(tmp_path / "lexicon.db")
    f = _write(
        tmp_path / "_modern_reflexes.jsonl",
        {
            "_type": "modern_reflex",
            "etymon_ref": "old-english:nonesuch",
            "modern_forms": ["nope"],
            "confidence": "high",
            "reference": "OED",
        },
    )
    counts = import_modern_reflexes(db, f, apply=True)
    assert counts == {"morpheme_missing": 1}
    assert (
        db.conn.execute("SELECT COUNT(*) FROM etymon WHERE language='modern-english'").fetchone()[0]
        == 0
    )


def test_existing_reflex_in_own_cluster_is_left_alone(tmp_path):
    """An existing modern reflex that already sits in its own cluster gets the
    descent edge but its cognate_id is NOT overwritten (reflex_pre_clustered)."""
    init_schema(tmp_path / "lexicon.db")
    db = LexiconDB(tmp_path / "lexicon.db")
    mersc = _clustered_morpheme(db, "mersc", "old-english")
    # pre-existing modern 'marsh' in its OWN distinct cluster.
    marsh = db.upsert_etymon("marsh", "modern-english")
    db.conn.execute("UPDATE etymon SET cognate_id=? WHERE id=?", (marsh, marsh))
    db.commit()
    f = _write(
        tmp_path / "_modern_reflexes.jsonl",
        {
            "_type": "modern_reflex",
            "etymon_ref": "old-english:mersc",
            "modern_forms": ["marsh"],
            "confidence": "high",
            "reference": "OED",
        },
    )
    counts = import_modern_reflexes(db, f, apply=True)
    assert counts["reflex_existing"] == 1
    assert counts["reflex_pre_clustered"] == 1
    assert counts.get("clustered", 0) == 0
    # cognate_id untouched (not hijacked into mersc's cluster)
    assert (
        db.conn.execute("SELECT cognate_id FROM etymon WHERE id=?", (marsh,)).fetchone()[0] == marsh
    )
    # the provenance descent edge is still recorded.
    assert (
        db.conn.execute(
            "SELECT COUNT(*) FROM etymon_descent WHERE parent_id=? AND child_id=?", (mersc, marsh)
        ).fetchone()[0]
        == 1
    )


def test_missing_file_is_a_noop(tmp_path):
    init_schema(tmp_path / "lexicon.db")
    db = LexiconDB(tmp_path / "lexicon.db")
    assert import_modern_reflexes(db, tmp_path / "nope.jsonl", apply=True) == {"file_missing": 1}


def test_malformed_jsonl_raises_with_line_number(tmp_path):
    init_schema(tmp_path / "lexicon.db")
    db = LexiconDB(tmp_path / "lexicon.db")
    bad = tmp_path / "_modern_reflexes.jsonl"
    bad.write_text('{"_type": "modern_reflex"}\nnot json\n', encoding="utf-8")
    import pytest

    with pytest.raises(ValueError, match=r":2: malformed JSONL"):
        import_modern_reflexes(db, bad, apply=True)


def test_cli_missing_db_errors(tmp_path):
    result = CliRunner().invoke(
        lexicon_import_modern_reflexes, ["--db", str(tmp_path / "absent.db"), "--apply"]
    )
    assert result.exit_code != 0
    assert "not found" in result.output


def test_cli_apply_happy_path(tmp_path):
    init_schema(tmp_path / "lexicon.db")
    db = LexiconDB(tmp_path / "lexicon.db")
    _clustered_morpheme(db, "hām", "old-english")
    db.close()
    f = _write(
        tmp_path / "_modern_reflexes.jsonl",
        {
            "_type": "modern_reflex",
            "etymon_ref": "old-english:hām",
            "modern_forms": ["home"],
            "confidence": "high",
            "reference": "OED",
        },
    )
    result = CliRunner().invoke(
        lexicon_import_modern_reflexes,
        ["--db", str(tmp_path / "lexicon.db"), "--jsonl", str(f), "--apply"],
    )
    assert result.exit_code == 0, result.output
    assert "APPLIED" in result.output
    db2 = LexiconDB(tmp_path / "lexicon.db")
    assert (
        db2.conn.execute(
            "SELECT COUNT(*) FROM etymon WHERE canonical_form='home' AND language='modern-english'"
        ).fetchone()[0]
        == 1
    )
