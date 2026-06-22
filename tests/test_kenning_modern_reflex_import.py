"""wyrd-vewk — import curated modern-English reflexes, end to end.

Real-DB (D10): a clustered morpheme with NO modern reflex gets one via the
importer, and ``etymon_era_reflexes`` then surfaces it in the modern stage —
the whole point (era-grid modern cell fills). Plus: 'none' rows are skipped,
the run is idempotent, the descent edge is recorded, the un-clustered Tier-2
path works, and the CLI validates its inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
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


@pytest.fixture
def lex(tmp_path):
    """An open LexiconDB on a fresh tmp-path schema, closed on teardown so the
    SQLite connection / file lock is released before tmp cleanup."""
    init_schema(tmp_path / "lexicon.db")
    db = LexiconDB(tmp_path / "lexicon.db")
    try:
        yield db
    finally:
        db.close()


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


def test_import_fills_modern_era_reflex_for_clustered_morpheme(lex, tmp_path):
    ham = _clustered_morpheme(lex, "hām", "old-english")
    # pre: no curated modern reflex in the cluster (a Tier-4 phonology-rule
    # form may be synthesized, but 'home'/'ham' are absent).
    pre = {r.form for r in etymon_era_reflexes(lex, ham, target_language="modern-english")}
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
            "reference": "OED s.v. home",
        },
    )
    counts = import_modern_reflexes(lex, f, apply=True)
    assert counts["reflex_created"] == 2
    assert counts["clustered"] == 2

    # post: the era-grid modern stage now surfaces the curated reflexes.
    forms = {r.form for r in etymon_era_reflexes(lex, ham, target_language="modern-english")}
    assert forms == {"home", "ham"}
    # the new etymon carries the gloss + a descent edge + the curated reference
    # preserved as the citation short_quote.
    assert (
        lex.conn.execute(
            "SELECT gloss FROM etymon_gloss g JOIN etymon e ON e.id=g.etymon_id "
            "WHERE e.canonical_form='home' AND e.language='modern-english'"
        ).fetchone()[0]
        == "homestead"
    )
    assert (
        lex.conn.execute(
            "SELECT short_quote FROM etymon_citation c JOIN etymon e ON e.id=c.etymon_id "
            "WHERE e.canonical_form='home' AND e.language='modern-english'"
        ).fetchone()[0]
        == "OED s.v. home"
    )
    assert (
        lex.conn.execute(
            "SELECT COUNT(*) FROM etymon_descent WHERE source_id=? AND edge_type='inheritance'",
            (MODERN_REFLEX_SOURCE,),
        ).fetchone()[0]
        == 2
    )


def test_none_rows_skipped_and_idempotent(lex, tmp_path):
    _clustered_morpheme(lex, "skógr", "old-norse")
    ham = _clustered_morpheme(lex, "hām", "old-english")
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
    c1 = import_modern_reflexes(lex, f, apply=True)
    assert c1["skipped_no_reflex"] == 1 and c1["reflex_created"] == 1
    # idempotent: a second run creates nothing new, no duplicate descent edges.
    c2 = import_modern_reflexes(lex, f, apply=True)
    assert c2.get("reflex_created", 0) == 0
    assert (
        lex.conn.execute(
            "SELECT COUNT(*) FROM etymon_descent WHERE source_id=?", (MODERN_REFLEX_SOURCE,)
        ).fetchone()[0]
        == 1
    )
    assert {r.form for r in etymon_era_reflexes(lex, ham, target_language="modern-english")} == {
        "home"
    }


def test_junk_modern_reflex_form_skipped_and_counted(lex, tmp_path):
    """wyrd-aicu.8 (D45): a modern reflex form that de-dashes to empty junk
    ('-') is dropped before the upsert choke would raise — counted as
    reflex_skipped_junk — while a valid form alongside it is still created."""
    _clustered_morpheme(lex, "hām", "old-english")
    f = _write(
        tmp_path / "_modern_reflexes.jsonl",
        {"_type": "source", "ref": "modern-reflex-curation", "title": "t"},
        {
            "_type": "modern_reflex",
            "etymon_ref": "old-english:hām",
            "modern_forms": ["home", "-"],  # one real form, one strip-to-empty junk
            "confidence": "high",
            "reference": "OED",
        },
    )
    counts = import_modern_reflexes(lex, f, apply=True)
    assert counts["reflex_skipped_junk"] == 1
    assert counts["reflex_created"] == 1
    forms = {
        r[0]
        for r in lex.conn.execute(
            "SELECT canonical_form FROM etymon WHERE language='modern-english'"
        )
    }
    assert "home" in forms
    assert "-" not in forms and "" not in forms


def test_dry_run_writes_nothing(lex, tmp_path):
    ham = _clustered_morpheme(lex, "hām", "old-english")
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
    counts = import_modern_reflexes(lex, f, apply=False)
    assert counts.get("reflex_would_create") == 1  # dry-run reports, writes nothing
    assert counts.get("would_link") == 1  # total links that would be made
    assert "home" not in {
        r.form for r in etymon_era_reflexes(lex, ham, target_language="modern-english")
    }


def test_unclustered_morpheme_fills_via_tier2_descent(lex, tmp_path):
    """A morpheme with NO cognate cluster relies on the descent edge: the
    importer records descent_only and etymon_era_reflexes Tier-2 surfaces it."""
    eid = lex.upsert_etymon("creca", "old-english")  # NOT clustered (cognate_id NULL)
    lex.commit()
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
    counts = import_modern_reflexes(lex, f, apply=True)
    assert counts["descent_only"] == 1 and counts.get("clustered", 0) == 0
    assert "creek" in {
        r.form for r in etymon_era_reflexes(lex, eid, target_language="modern-english")
    }


def test_morpheme_missing_is_counted_and_skipped(lex, tmp_path):
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
    counts = import_modern_reflexes(lex, f, apply=True)
    assert counts == {"morpheme_missing": 1}
    assert (
        lex.conn.execute("SELECT COUNT(*) FROM etymon WHERE language='modern-english'").fetchone()[
            0
        ]
        == 0
    )


def test_existing_reflex_in_own_cluster_is_left_alone(lex, tmp_path):
    """An existing modern reflex that already sits in its own cluster gets the
    descent edge but its cognate_id is NOT overwritten (reflex_pre_clustered)."""
    mersc = _clustered_morpheme(lex, "mersc", "old-english")
    marsh = lex.upsert_etymon("marsh", "modern-english")  # pre-existing, own cluster
    lex.conn.execute("UPDATE etymon SET cognate_id=? WHERE id=?", (marsh, marsh))
    lex.commit()
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
    counts = import_modern_reflexes(lex, f, apply=True)
    assert counts["reflex_existing"] == 1
    assert counts["reflex_pre_clustered"] == 1
    assert counts.get("clustered", 0) == 0
    # cognate_id untouched (not hijacked into mersc's cluster)
    assert (
        lex.conn.execute("SELECT cognate_id FROM etymon WHERE id=?", (marsh,)).fetchone()[0]
        == marsh
    )
    # the provenance descent edge is still recorded.
    assert (
        lex.conn.execute(
            "SELECT COUNT(*) FROM etymon_descent WHERE parent_id=? AND child_id=?", (mersc, marsh)
        ).fetchone()[0]
        == 1
    )


def test_missing_file_is_a_noop(lex, tmp_path):
    assert import_modern_reflexes(lex, tmp_path / "nope.jsonl", apply=True) == {"file_missing": 1}


def test_malformed_jsonl_raises_with_line_number(lex, tmp_path):
    bad = tmp_path / "_modern_reflexes.jsonl"
    bad.write_text('{"_type": "modern_reflex"}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r":2: malformed JSONL"):
        import_modern_reflexes(lex, bad, apply=True)


def test_cli_missing_db_errors(tmp_path):
    result = CliRunner().invoke(
        lexicon_import_modern_reflexes, ["--db", str(tmp_path / "absent.db"), "--apply"]
    )
    assert result.exit_code != 0
    assert "not found" in result.output


def test_cli_missing_jsonl_errors(tmp_path):
    init_schema(tmp_path / "lexicon.db")
    LexiconDB(tmp_path / "lexicon.db").close()
    result = CliRunner().invoke(
        lexicon_import_modern_reflexes,
        [
            "--db",
            str(tmp_path / "lexicon.db"),
            "--jsonl",
            str(tmp_path / "absent.jsonl"),
            "--apply",
        ],
    )
    assert result.exit_code != 0
    assert "not found" in result.output


def test_cli_apply_happy_path(tmp_path):
    init_schema(tmp_path / "lexicon.db")
    with LexiconDB(tmp_path / "lexicon.db") as db:
        _clustered_morpheme(db, "hām", "old-english")
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
    with LexiconDB(tmp_path / "lexicon.db") as db2:
        assert (
            db2.conn.execute(
                "SELECT COUNT(*) FROM etymon WHERE canonical_form='home' AND language='modern-english'"
            ).fetchone()[0]
            == 1
        )
