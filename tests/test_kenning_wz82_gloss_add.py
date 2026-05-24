"""Tests for wyrd-wz82 etymon_gloss_add event type.

Inverse of etymon_gloss_suppression. Operator surfaces a sense the
auto-mining missed — e.g. OE finn carries "clear, transparent, bright"
+ "fin"; Roberts 1914 cites it as a personal name (a third sense the
bundle has no gloss for) → curate-add-gloss adds it without requiring
a full lexicon re-mine.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli as cli_root
from wyrd.generators.kenning.enrichment import (
    GLOSS_ADD_METHOD_VERSION,
    apply_gloss_additions,
    format_gloss_addition_run,
    run_full_enrichment,
)
from wyrd.generators.kenning.jsonl.build import collect_gloss_additions
from wyrd.generators.kenning.jsonl.log import write_jsonl
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema


def _build_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    return db_path


def _add_etymon_with_glosses(
    db_path: Path, language: str, canonical_form: str, glosses: list[str]
) -> int:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES (?, ?)",
        (canonical_form, language),
    )
    eid = cur.lastrowid
    for gloss in glosses:
        conn.execute(
            "INSERT INTO etymon_gloss (etymon_id, gloss) VALUES (?, ?)",
            (eid, gloss),
        )
    conn.commit()
    conn.close()
    return eid


def _write_curation_file(tmp_path: Path, rows: list[dict]) -> Path:
    source_row = {
        "_type": "source",
        "ref": "manual-curation",
        "title": "Operator curation overrides",
    }
    path = tmp_path / "_curation.jsonl"
    write_jsonl(path, [source_row, *rows])
    return path


# ---------------------------------------------------------------------------
# collect_gloss_additions
# ---------------------------------------------------------------------------


def test_collect_addition_returns_empty_when_no_rows(tmp_path: Path):
    path = tmp_path / "skeat.jsonl"
    write_jsonl(path, [{"_type": "source", "ref": "skeat", "title": "X"}])
    assert collect_gloss_additions([path]) == {}


def test_collect_addition_canonical_state_row(tmp_path: Path):
    path = _write_curation_file(
        tmp_path,
        [
            {
                "_type": "etymon_gloss_add",
                "ref": "old-english:finn",
                "additions": [{"gloss": "personal name", "reason": "Roberts 1914"}],
            }
        ],
    )
    state = collect_gloss_additions([path])
    assert state == {
        "old-english:finn": {"additions": [{"gloss": "personal name", "reason": "Roberts 1914"}]}
    }


def test_collect_addition_does_not_pick_up_suppression_rows(tmp_path: Path):
    """known kernel contract: _type filter keeps suppression / split /
    addition event streams independent in the same file."""
    path = _write_curation_file(
        tmp_path,
        [
            {
                "_type": "etymon_gloss_suppression",
                "ref": "old-english:dry",
                "suppressions": [{"gloss": "wizard, sorcerer"}],
            },
            {
                "_type": "etymon_gloss_add",
                "ref": "old-english:finn",
                "additions": [{"gloss": "personal name"}],
            },
        ],
    )
    state = collect_gloss_additions([path])
    assert "old-english:finn" in state
    assert "old-english:dry" not in state


# ---------------------------------------------------------------------------
# apply_gloss_additions
# ---------------------------------------------------------------------------


def test_apply_adds_named_glosses(tmp_path: Path):
    db_path = _build_db(tmp_path)
    eid = _add_etymon_with_glosses(db_path, "old-english", "finn", ["bright"])

    db = LexiconDB(db_path)
    state = {
        "old-english:finn": {"additions": [{"gloss": "personal name", "reason": "Roberts 1914"}]}
    }
    counts = apply_gloss_additions(db, state, apply=True)
    assert counts["glosses_added"] == 1
    assert counts["additions_already_present"] == 0
    remaining = sorted(
        row["gloss"]
        for row in db.conn.execute("SELECT gloss FROM etymon_gloss WHERE etymon_id = ?", (eid,))
    )
    assert remaining == ["bright", "personal name"]
    db.close()


def test_apply_idempotent_on_existing_gloss(tmp_path: Path):
    """Re-applying with the same gloss → additions_already_present
    counter increments; no error, no duplicate row."""
    db_path = _build_db(tmp_path)
    eid = _add_etymon_with_glosses(db_path, "old-english", "finn", ["personal name"])
    db = LexiconDB(db_path)
    state = {"old-english:finn": {"additions": [{"gloss": "personal name"}]}}
    counts = apply_gloss_additions(db, state, apply=True)
    assert counts["additions_already_present"] == 1
    assert counts["glosses_added"] == 0
    # No duplicate row.
    rowcount = db.conn.execute(
        "SELECT COUNT(*) AS n FROM etymon_gloss WHERE etymon_id = ?", (eid,)
    ).fetchone()["n"]
    assert rowcount == 1
    db.close()


def test_apply_dry_run_makes_no_changes(tmp_path: Path):
    db_path = _build_db(tmp_path)
    eid = _add_etymon_with_glosses(db_path, "old-english", "finn", ["bright"])
    db = LexiconDB(db_path)
    state = {"old-english:finn": {"additions": [{"gloss": "personal name"}]}}
    counts = apply_gloss_additions(db, state, apply=False)
    assert counts["glosses_added"] == 1
    assert counts["applied"] is False
    rowcount = db.conn.execute(
        "SELECT COUNT(*) AS n FROM etymon_gloss WHERE etymon_id = ?", (eid,)
    ).fetchone()["n"]
    assert rowcount == 1  # only the original
    db.close()


def test_apply_unresolved_etymon_counted(tmp_path: Path):
    db_path = _build_db(tmp_path)
    db = LexiconDB(db_path)
    state = {"old-english:nonexistent": {"additions": [{"gloss": "x"}]}}
    counts = apply_gloss_additions(db, state, apply=True)
    assert counts["unresolved_etymon"] == 1
    assert counts["glosses_added"] == 0
    db.close()


def test_format_addition_run_markdown():
    counts = {
        "etymons_touched": 2,
        "glosses_added": 3,
        "additions_already_present": 1,
        "unresolved_etymon": 0,
        "applied": True,
        "method_version": GLOSS_ADD_METHOD_VERSION,
    }
    rendered = format_gloss_addition_run(counts)
    assert "Etymons touched: 2" in rendered
    assert "Glosses added: 3" in rendered
    assert "already on etymon (idempotent): 1" in rendered


# ---------------------------------------------------------------------------
# CLI: lexicon curate-add-gloss
# ---------------------------------------------------------------------------


def test_cli_curate_add_gloss_appends_event(tmp_path: Path):
    curation_file = tmp_path / "_curation.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "curate-add-gloss",
            "old-english:finn",
            "personal name",
            "--reason",
            "Roberts 1914 cites for -dene compound",
            "--curation-file",
            str(curation_file),
        ],
    )
    assert result.exit_code == 0, result.output
    lines = curation_file.read_text().splitlines()
    assert len(lines) == 2  # source + event
    event = json.loads(lines[1])
    assert event["_type"] == "etymon_gloss_add"
    assert event["ref"] == "old-english:finn"
    assert event["additions"] == [
        {"gloss": "personal name", "reason": "Roberts 1914 cites for -dene compound"}
    ]


def test_cli_curate_add_gloss_multiple_glosses_one_event(tmp_path: Path):
    """N glosses in one CLI invocation → ONE event with N entries
    (mirrors the corn-regression test from wyrd-kutx; same KEYED_TYPES
    last-write-wins contract applies)."""
    curation_file = tmp_path / "_curation.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "curate-add-gloss",
            "old-english:corn",
            "personal name",
            "stake",
            "--curation-file",
            str(curation_file),
        ],
    )
    assert result.exit_code == 0, result.output
    event = json.loads(curation_file.read_text().splitlines()[1])
    assert len(event["additions"]) == 2
    assert [a["gloss"] for a in event["additions"]] == ["personal name", "stake"]


# ---------------------------------------------------------------------------
# run_full_enrichment plumbing for the new addition_state kwarg
# ---------------------------------------------------------------------------


def test_run_full_enrichment_threads_addition_state(tmp_path: Path):
    """The new kwarg flows through and the applier output appears in
    the result dict. Also asserts the documented run order:
    apply-gloss-suppressions → apply-gloss-additions → apply-etymon-splits
    so a single enrich call can both drop a stale gloss and add a new one."""
    db_path = _build_db(tmp_path)
    _add_etymon_with_glosses(db_path, "old-english", "finn", ["bright"])

    addition_state = {"old-english:finn": {"additions": [{"gloss": "personal name"}]}}
    with LexiconDB(db_path) as db:
        result = run_full_enrichment(
            db,
            apply=True,
            addition_state=addition_state,
            skip_l3_derivations=True,
        )
    order = result["order"]
    assert "apply-gloss-additions" in order
    assert result["gloss_additions"]["glosses_added"] == 1
