"""Tests for wyrd-kutx gloss-suppression + etymon-split event types.

Same shape as the wyrd-2jhs curation tests: small synthetic DB,
write JSONL events, replay through the collectors, run the appliers,
assert the SQL state.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from wyrd.generators.kenning.enrichment import (
    ETYMON_SPLIT_METHOD_VERSION,
    GLOSS_SUPPRESSION_METHOD_VERSION,
    apply_etymon_splits,
    apply_gloss_suppressions,
    format_etymon_split_run,
    format_gloss_suppression_run,
)
from wyrd.generators.kenning.jsonl.build import (
    collect_etymon_splits,
    collect_gloss_suppressions,
)
from wyrd.generators.kenning.jsonl.log import write_jsonl
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema


def _build_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    return db_path


def _add_etymon_with_glosses(
    db_path: Path,
    language: str,
    canonical_form: str,
    glosses: list[str],
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
# collect_gloss_suppressions
# ---------------------------------------------------------------------------


def test_collect_suppression_returns_empty_when_no_rows(tmp_path: Path):
    path = tmp_path / "skeat.jsonl"
    write_jsonl(path, [{"_type": "source", "ref": "skeat", "title": "X"}])
    assert collect_gloss_suppressions([path]) == {}


def test_collect_suppression_canonical_state_row(tmp_path: Path):
    path = _write_curation_file(
        tmp_path,
        [
            {
                "_type": "etymon_gloss_suppression",
                "ref": "old-english:dry",
                "suppressions": [
                    {"gloss": "wizard, sorcerer", "reason": "no toponymy use"}
                ],
            }
        ],
    )
    state = collect_gloss_suppressions([path])
    assert state == {
        "old-english:dry": {
            "suppressions": [
                {"gloss": "wizard, sorcerer", "reason": "no toponymy use"}
            ]
        }
    }


# ---------------------------------------------------------------------------
# apply_gloss_suppressions
# ---------------------------------------------------------------------------


def test_apply_drops_named_gloss_only(tmp_path: Path):
    db_path = _build_db(tmp_path)
    eid = _add_etymon_with_glosses(
        db_path, "old-english", "dry", ["dry", "wizard, sorcerer"]
    )

    db = LexiconDB(db_path)
    state = {
        "old-english:dry": {
            "suppressions": [{"gloss": "wizard, sorcerer", "reason": "test"}]
        }
    }
    counts = apply_gloss_suppressions(db, state, apply=True)

    assert counts["glosses_dropped"] == 1
    assert counts["etymons_touched"] == 1
    assert counts["unresolved_etymon"] == 0
    assert counts["unresolved_gloss"] == 0

    remaining = [
        row["gloss"]
        for row in db.conn.execute(
            "SELECT gloss FROM etymon_gloss WHERE etymon_id = ?", (eid,)
        )
    ]
    assert remaining == ["dry"]
    db.close()


def test_apply_dry_run_makes_no_changes(tmp_path: Path):
    db_path = _build_db(tmp_path)
    eid = _add_etymon_with_glosses(
        db_path, "old-english", "dry", ["dry", "wizard, sorcerer"]
    )

    db = LexiconDB(db_path)
    state = {
        "old-english:dry": {
            "suppressions": [{"gloss": "wizard, sorcerer"}]
        }
    }
    counts = apply_gloss_suppressions(db, state, apply=False)
    assert counts["glosses_dropped"] == 1
    assert counts["applied"] is False

    remaining = sorted(
        row["gloss"]
        for row in db.conn.execute(
            "SELECT gloss FROM etymon_gloss WHERE etymon_id = ?", (eid,)
        )
    )
    assert remaining == ["dry", "wizard, sorcerer"]
    db.close()


def test_apply_unresolved_etymon_counted(tmp_path: Path):
    db_path = _build_db(tmp_path)
    db = LexiconDB(db_path)
    state = {
        "old-english:nonexistent": {
            "suppressions": [{"gloss": "x"}]
        }
    }
    counts = apply_gloss_suppressions(db, state, apply=True)
    assert counts["unresolved_etymon"] == 1
    assert counts["glosses_dropped"] == 0
    db.close()


def test_apply_unresolved_gloss_counted(tmp_path: Path):
    db_path = _build_db(tmp_path)
    _add_etymon_with_glosses(db_path, "old-english", "dry", ["dry"])

    db = LexiconDB(db_path)
    state = {
        "old-english:dry": {
            "suppressions": [{"gloss": "wizard, sorcerer"}]
        }
    }
    counts = apply_gloss_suppressions(db, state, apply=True)
    assert counts["unresolved_gloss"] == 1
    assert counts["glosses_dropped"] == 0
    db.close()


def _add_etymon_with_glosses_and_tags(
    db_path: Path,
    language: str,
    canonical_form: str,
    glosses: list[str],
    tags: list[str] | None = None,
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
    for tag in tags or []:
        conn.execute(
            "INSERT INTO etymon_tag (etymon_id, tag) VALUES (?, ?)",
            (eid, tag),
        )
    conn.commit()
    conn.close()
    return eid


# ---------------------------------------------------------------------------
# collect_etymon_splits
# ---------------------------------------------------------------------------


def test_collect_split_canonical_state(tmp_path: Path):
    path = _write_curation_file(
        tmp_path,
        [
            {
                "_type": "etymon_split",
                "ref": "old-english:gear",
                "into": [
                    {
                        "suffix": "weir",
                        "glosses": ["Weir (fishing enclosure)"],
                        "primary": True,
                    },
                    {"suffix": "year", "glosses": ["year"]},
                ],
                "reason": "real homograph",
            }
        ],
    )
    state = collect_etymon_splits([path])
    assert state["old-english:gear"]["into"][0]["suffix"] == "weir"
    assert state["old-english:gear"]["into"][1]["suffix"] == "year"
    assert state["old-english:gear"]["reason"] == "real homograph"


# ---------------------------------------------------------------------------
# apply_etymon_splits
# ---------------------------------------------------------------------------


def test_apply_split_creates_children_and_moves_glosses(tmp_path: Path):
    db_path = _build_db(tmp_path)
    parent_id = _add_etymon_with_glosses_and_tags(
        db_path,
        "old-english",
        "gear",
        ["Weir (fishing enclosure)", "year"],
        tags=["topography", "time"],
    )

    db = LexiconDB(db_path)
    state = {
        "old-english:gear": {
            "into": [
                {
                    "suffix": "weir",
                    "glosses": ["Weir (fishing enclosure)"],
                    "tags": ["topography"],
                    "primary": True,
                },
                {"suffix": "year", "glosses": ["year"], "tags": ["time"]},
            ]
        }
    }
    counts = apply_etymon_splits(db, state, apply=True)

    assert counts["splits_processed"] == 1
    assert counts["children_created"] == 2
    assert counts["glosses_moved"] == 2
    assert counts["tags_moved"] == 2

    # Parent has no glosses or tags left.
    parent_glosses = list(
        db.conn.execute(
            "SELECT gloss FROM etymon_gloss WHERE etymon_id = ?", (parent_id,)
        )
    )
    assert parent_glosses == []
    parent_tags = list(
        db.conn.execute(
            "SELECT tag FROM etymon_tag WHERE etymon_id = ?", (parent_id,)
        )
    )
    assert parent_tags == []

    # Parent's notes column carries the split marker.
    parent_notes = db.conn.execute(
        "SELECT notes FROM etymon WHERE id = ?", (parent_id,)
    ).fetchone()["notes"]
    assert "[wyrd-kutx-split:2]" in parent_notes

    # Children exist with disambiguated canonical_forms.
    children = sorted(
        row["canonical_form"]
        for row in db.conn.execute(
            "SELECT canonical_form FROM etymon WHERE language = ? AND canonical_form LIKE 'gear#%'",
            ("old-english",),
        )
    )
    assert children == ["gear#weir", "gear#year"]

    # Each child has its own gloss.
    weir = db.conn.execute(
        "SELECT id FROM etymon WHERE language = ? AND canonical_form = ?",
        ("old-english", "gear#weir"),
    ).fetchone()["id"]
    weir_glosses = [
        row["gloss"]
        for row in db.conn.execute(
            "SELECT gloss FROM etymon_gloss WHERE etymon_id = ?", (weir,)
        )
    ]
    assert weir_glosses == ["Weir (fishing enclosure)"]
    db.close()


def test_apply_split_dry_run_makes_no_changes(tmp_path: Path):
    db_path = _build_db(tmp_path)
    parent_id = _add_etymon_with_glosses_and_tags(
        db_path, "old-english", "gear", ["a", "b"]
    )

    db = LexiconDB(db_path)
    state = {
        "old-english:gear": {
            "into": [
                {"suffix": "x", "glosses": ["a"]},
                {"suffix": "y", "glosses": ["b"]},
            ]
        }
    }
    counts = apply_etymon_splits(db, state, apply=False)
    assert counts["splits_processed"] == 1
    assert counts["children_created"] == 2
    assert counts["glosses_moved"] == 2

    # No new rows in etymon.
    n_etymons = db.conn.execute("SELECT COUNT(*) AS n FROM etymon").fetchone()["n"]
    assert n_etymons == 1
    # Parent still has its glosses.
    parent_glosses = sorted(
        row["gloss"]
        for row in db.conn.execute(
            "SELECT gloss FROM etymon_gloss WHERE etymon_id = ?", (parent_id,)
        )
    )
    assert parent_glosses == ["a", "b"]
    db.close()


def test_apply_split_unresolved_parent_counted(tmp_path: Path):
    db_path = _build_db(tmp_path)
    db = LexiconDB(db_path)
    state = {
        "old-english:nonexistent": {
            "into": [{"suffix": "x", "glosses": ["x"]}]
        }
    }
    counts = apply_etymon_splits(db, state, apply=True)
    assert counts["unresolved_etymon"] == 1
    assert counts["children_created"] == 0
    db.close()


def test_apply_split_empty_into_is_noop(tmp_path: Path):
    db_path = _build_db(tmp_path)
    _add_etymon_with_glosses_and_tags(db_path, "old-english", "gear", ["a"])

    db = LexiconDB(db_path)
    state = {"old-english:gear": {"into": []}}
    counts = apply_etymon_splits(db, state, apply=True)
    assert counts["empty_into_skipped"] == 1
    assert counts["splits_processed"] == 0
    db.close()


def test_apply_split_gloss_not_on_parent_counted(tmp_path: Path):
    db_path = _build_db(tmp_path)
    _add_etymon_with_glosses_and_tags(db_path, "old-english", "gear", ["a"])

    db = LexiconDB(db_path)
    state = {
        "old-english:gear": {
            "into": [{"suffix": "x", "glosses": ["a", "ghost"]}]
        }
    }
    counts = apply_etymon_splits(db, state, apply=True)
    assert counts["glosses_moved"] == 1
    assert counts["glosses_missing"] == 1
    db.close()


def test_apply_split_reusing_existing_child_is_idempotent(tmp_path: Path):
    """Re-running the same split event after a successful apply doesn't
    duplicate children or re-move glosses (which have already been moved
    off the parent)."""
    db_path = _build_db(tmp_path)
    _add_etymon_with_glosses_and_tags(db_path, "old-english", "gear", ["a", "b"])

    db = LexiconDB(db_path)
    state = {
        "old-english:gear": {
            "into": [
                {"suffix": "x", "glosses": ["a"]},
                {"suffix": "y", "glosses": ["b"]},
            ]
        }
    }
    apply_etymon_splits(db, state, apply=True)
    counts = apply_etymon_splits(db, state, apply=True)
    # Second run: children already exist; glosses already moved (so
    # they're "missing" from the parent now).
    assert counts["children_already_existed"] == 2
    assert counts["children_created"] == 0
    assert counts["glosses_missing"] == 2
    assert counts["glosses_moved"] == 0
    db.close()


def _add_citation(db_path: Path, etymon_id: int, source_id: str) -> None:
    conn = sqlite3.connect(db_path)
    # Source row first (etymon_citation has FK).
    conn.execute(
        "INSERT OR IGNORE INTO source (id, title) VALUES (?, ?)",
        (source_id, source_id),
    )
    conn.execute(
        "INSERT INTO etymon_citation (etymon_id, source_id) VALUES (?, ?)",
        (etymon_id, source_id),
    )
    conn.commit()
    conn.close()


def test_apply_split_moves_citations_to_primary(tmp_path: Path):
    """Citations on the parent migrate to the primary=true child so the
    bundle's witness count survives the split."""
    db_path = _build_db(tmp_path)
    parent_id = _add_etymon_with_glosses_and_tags(
        db_path, "old-english", "gear", ["weir", "year"]
    )
    _add_citation(db_path, parent_id, "mawer-1920")
    _add_citation(db_path, parent_id, "ekwall-1928")

    db = LexiconDB(db_path)
    state = {
        "old-english:gear": {
            "into": [
                {"suffix": "weir", "glosses": ["weir"], "primary": True},
                {"suffix": "year", "glosses": ["year"]},
            ]
        }
    }
    counts = apply_etymon_splits(db, state, apply=True)
    assert counts["citations_moved"] == 2

    # Parent has no citations.
    parent_cits = db.conn.execute(
        "SELECT COUNT(*) AS n FROM etymon_citation WHERE etymon_id = ?",
        (parent_id,),
    ).fetchone()["n"]
    assert parent_cits == 0

    # Primary child (suffix=weir) inherits both.
    weir_id = db.conn.execute(
        "SELECT id FROM etymon WHERE language = ? AND canonical_form = ?",
        ("old-english", "gear#weir"),
    ).fetchone()["id"]
    weir_cits = db.conn.execute(
        "SELECT COUNT(*) AS n FROM etymon_citation WHERE etymon_id = ?",
        (weir_id,),
    ).fetchone()["n"]
    assert weir_cits == 2

    # Non-primary child has zero citations.
    year_id = db.conn.execute(
        "SELECT id FROM etymon WHERE language = ? AND canonical_form = ?",
        ("old-english", "gear#year"),
    ).fetchone()["id"]
    year_cits = db.conn.execute(
        "SELECT COUNT(*) AS n FROM etymon_citation WHERE etymon_id = ?",
        (year_id,),
    ).fetchone()["n"]
    assert year_cits == 0
    db.close()


def test_apply_split_defaults_primary_to_first_child(tmp_path: Path):
    """No `primary` flag on any child → first child gets the evidence."""
    db_path = _build_db(tmp_path)
    parent_id = _add_etymon_with_glosses_and_tags(
        db_path, "old-english", "gear", ["a", "b"]
    )
    _add_citation(db_path, parent_id, "mawer-1920")

    db = LexiconDB(db_path)
    state = {
        "old-english:gear": {
            "into": [
                {"suffix": "first", "glosses": ["a"]},
                {"suffix": "second", "glosses": ["b"]},
            ]
        }
    }
    counts = apply_etymon_splits(db, state, apply=True)
    assert counts["no_primary_defaulted"] == 1
    assert counts["citations_moved"] == 1

    first_id = db.conn.execute(
        "SELECT id FROM etymon WHERE language = ? AND canonical_form = ?",
        ("old-english", "gear#first"),
    ).fetchone()["id"]
    first_cits = db.conn.execute(
        "SELECT COUNT(*) AS n FROM etymon_citation WHERE etymon_id = ?",
        (first_id,),
    ).fetchone()["n"]
    assert first_cits == 1
    db.close()


def test_format_etymon_split_run_markdown(tmp_path: Path):
    counts = {
        "splits_processed": 2,
        "children_created": 4,
        "children_already_existed": 0,
        "glosses_moved": 6,
        "tags_moved": 3,
        "glosses_missing": 0,
        "tags_missing": 0,
        "unresolved_etymon": 0,
        "empty_into_skipped": 0,
        "applied": True,
        "method_version": ETYMON_SPLIT_METHOD_VERSION,
    }
    rendered = format_etymon_split_run(counts)
    assert "Splits processed: 2" in rendered
    assert "Children created: 4" in rendered
    assert "Glosses moved: 6" in rendered


def test_format_gloss_suppression_run_markdown(tmp_path: Path):
    counts = {
        "etymons_touched": 3,
        "glosses_dropped": 5,
        "unresolved_etymon": 1,
        "unresolved_gloss": 0,
        "applied": True,
        "method_version": GLOSS_SUPPRESSION_METHOD_VERSION,
    }
    rendered = format_gloss_suppression_run(counts)
    assert "Etymons touched: 3" in rendered
    assert "Glosses dropped: 5" in rendered
    assert "Unresolved etymon refs: 1" in rendered
