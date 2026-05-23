"""Tests for wyrd-kutx gloss-suppression + etymon-split event types.

Same shape as the wyrd-2jhs curation tests: small synthetic DB,
write JSONL events, replay through the collectors, run the appliers,
assert the SQL state.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli as cli_root
from wyrd.generators.kenning.enrichment import (
    ETYMON_SPLIT_METHOD_VERSION,
    GLOSS_SUPPRESSION_METHOD_VERSION,
    apply_etymon_splits,
    apply_gloss_suppressions,
    format_etymon_split_run,
    format_gloss_suppression_run,
    run_full_enrichment,
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
                "suppressions": [{"gloss": "wizard, sorcerer", "reason": "no toponymy use"}],
            }
        ],
    )
    state = collect_gloss_suppressions([path])
    assert state == {
        "old-english:dry": {
            "suppressions": [{"gloss": "wizard, sorcerer", "reason": "no toponymy use"}]
        }
    }


# ---------------------------------------------------------------------------
# apply_gloss_suppressions
# ---------------------------------------------------------------------------


def test_apply_drops_named_gloss_only(tmp_path: Path):
    db_path = _build_db(tmp_path)
    eid = _add_etymon_with_glosses(db_path, "old-english", "dry", ["dry", "wizard, sorcerer"])

    db = LexiconDB(db_path)
    state = {"old-english:dry": {"suppressions": [{"gloss": "wizard, sorcerer", "reason": "test"}]}}
    counts = apply_gloss_suppressions(db, state, apply=True)

    assert counts["glosses_dropped"] == 1
    assert counts["etymons_touched"] == 1
    assert counts["unresolved_etymon"] == 0
    assert counts["unresolved_gloss"] == 0

    remaining = [
        row["gloss"]
        for row in db.conn.execute("SELECT gloss FROM etymon_gloss WHERE etymon_id = ?", (eid,))
    ]
    assert remaining == ["dry"]
    db.close()


def test_apply_dry_run_makes_no_changes(tmp_path: Path):
    db_path = _build_db(tmp_path)
    eid = _add_etymon_with_glosses(db_path, "old-english", "dry", ["dry", "wizard, sorcerer"])

    db = LexiconDB(db_path)
    state = {"old-english:dry": {"suppressions": [{"gloss": "wizard, sorcerer"}]}}
    counts = apply_gloss_suppressions(db, state, apply=False)
    assert counts["glosses_dropped"] == 1
    assert counts["applied"] is False

    remaining = sorted(
        row["gloss"]
        for row in db.conn.execute("SELECT gloss FROM etymon_gloss WHERE etymon_id = ?", (eid,))
    )
    assert remaining == ["dry", "wizard, sorcerer"]
    db.close()


def test_apply_unresolved_etymon_counted(tmp_path: Path):
    db_path = _build_db(tmp_path)
    db = LexiconDB(db_path)
    state = {"old-english:nonexistent": {"suppressions": [{"gloss": "x"}]}}
    counts = apply_gloss_suppressions(db, state, apply=True)
    assert counts["unresolved_etymon"] == 1
    assert counts["glosses_dropped"] == 0
    db.close()


def test_apply_unresolved_gloss_counted(tmp_path: Path):
    db_path = _build_db(tmp_path)
    _add_etymon_with_glosses(db_path, "old-english", "dry", ["dry"])

    db = LexiconDB(db_path)
    state = {"old-english:dry": {"suppressions": [{"gloss": "wizard, sorcerer"}]}}
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
        db.conn.execute("SELECT gloss FROM etymon_gloss WHERE etymon_id = ?", (parent_id,))
    )
    assert parent_glosses == []
    parent_tags = list(
        db.conn.execute("SELECT tag FROM etymon_tag WHERE etymon_id = ?", (parent_id,))
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
        for row in db.conn.execute("SELECT gloss FROM etymon_gloss WHERE etymon_id = ?", (weir,))
    ]
    assert weir_glosses == ["Weir (fishing enclosure)"]
    db.close()


def test_apply_split_dry_run_makes_no_changes(tmp_path: Path):
    db_path = _build_db(tmp_path)
    parent_id = _add_etymon_with_glosses_and_tags(db_path, "old-english", "gear", ["a", "b"])

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
    state = {"old-english:nonexistent": {"into": [{"suffix": "x", "glosses": ["x"]}]}}
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
    state = {"old-english:gear": {"into": [{"suffix": "x", "glosses": ["a", "ghost"]}]}}
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
    parent_id = _add_etymon_with_glosses_and_tags(db_path, "old-english", "gear", ["weir", "year"])
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
    parent_id = _add_etymon_with_glosses_and_tags(db_path, "old-english", "gear", ["a", "b"])
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


def test_apply_split_jsonl_hand_edit_empty_suffix_counted(tmp_path: Path):
    """Defense-in-depth: the CLI rejects empty suffix, but a hand-edited
    JSONL with `suffix: ''` lands here. Counter surfaces the bypass
    rather than silently skipping."""
    db_path = _build_db(tmp_path)
    _add_etymon_with_glosses_and_tags(db_path, "old-english", "gear", ["weir"])
    db = LexiconDB(db_path)
    state = {
        "old-english:gear": {
            "into": [
                {"suffix": "", "glosses": ["weir"], "primary": True},
                {"suffix": "year", "glosses": ["year"]},
            ]
        }
    }
    counts = apply_etymon_splits(db, state, apply=True)
    assert counts["children_skipped_no_suffix"] == 1
    assert counts["children_created"] == 1  # only the 'year' child
    db.close()


def test_apply_split_jsonl_multi_primary_counted(tmp_path: Path):
    """Defense-in-depth: CLI rejects two-primary specs, JSONL might
    not. Applier picks first and surfaces the collapse via counter."""
    db_path = _build_db(tmp_path)
    _add_etymon_with_glosses_and_tags(db_path, "old-english", "gear", ["a", "b"])
    db = LexiconDB(db_path)
    state = {
        "old-english:gear": {
            "into": [
                {"suffix": "x", "glosses": ["a"], "primary": True},
                {"suffix": "y", "glosses": ["b"], "primary": True},
            ]
        }
    }
    counts = apply_etymon_splits(db, state, apply=True)
    assert counts["multiple_primary_collapsed"] == 1
    # First primary still wins.
    first_id = db.conn.execute(
        "SELECT id FROM etymon WHERE language = ? AND canonical_form = ?",
        ("old-english", "gear#x"),
    ).fetchone()["id"]
    second_id = db.conn.execute(
        "SELECT id FROM etymon WHERE language = ? AND canonical_form = ?",
        ("old-english", "gear#y"),
    ).fetchone()["id"]
    assert first_id != second_id
    db.close()


def test_apply_split_parent_not_stamped_when_zero_children_succeed(tmp_path: Path):
    """If a JSONL hand-edit has all-invalid children (every spec has
    empty suffix), the parent must NOT be stamped as split — otherwise
    audit queries find a "split" parent that still has all its glosses
    and citations attached."""
    db_path = _build_db(tmp_path)
    parent_id = _add_etymon_with_glosses_and_tags(db_path, "old-english", "gear", ["weir", "year"])
    db = LexiconDB(db_path)
    state = {
        "old-english:gear": {
            "into": [
                {"suffix": "", "glosses": ["weir"], "primary": True},
                {"suffix": "", "glosses": ["year"]},
            ]
        }
    }
    counts = apply_etymon_splits(db, state, apply=True)
    assert counts["children_skipped_no_suffix"] == 2
    assert counts["children_created"] == 0

    parent_notes = db.conn.execute(
        "SELECT notes FROM etymon WHERE id = ?", (parent_id,)
    ).fetchone()["notes"]
    assert parent_notes is None or "wyrd-kutx-split" not in (parent_notes or "")

    # Parent still has its glosses (nothing moved).
    parent_glosses = sorted(
        row["gloss"]
        for row in db.conn.execute(
            "SELECT gloss FROM etymon_gloss WHERE etymon_id = ?", (parent_id,)
        )
    )
    assert parent_glosses == ["weir", "year"]
    db.close()


def test_apply_split_citations_skipped_conflict_counted(tmp_path: Path):
    """When the primary child already carries a citation matching the
    parent's UNIQUE (etymon_id, source_id, COALESCE(page,'')) tuple,
    UPDATE OR IGNORE leaves that row on the parent and the counter
    surfaces the skip. Avoids the "citations_moved overstates"
    telemetry bug."""
    db_path = _build_db(tmp_path)
    parent_id = _add_etymon_with_glosses_and_tags(db_path, "old-english", "gear", ["weir"])
    # Pre-create the primary child carrying a citation that the
    # parent's same source_id would collide with.
    conn = sqlite3.connect(db_path)
    cur = conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES (?, ?)",
        ("gear#weir", "old-english"),
    )
    child_id = cur.lastrowid
    for source_id in ("colliding-source", "moved-source"):
        conn.execute(
            "INSERT OR IGNORE INTO source (id, title) VALUES (?, ?)",
            (source_id, source_id),
        )
    # Parent carries 2 citations; child already carries the
    # colliding-source citation that would conflict on UNIQUE.
    conn.execute(
        "INSERT INTO etymon_citation (etymon_id, source_id) VALUES (?, ?)",
        (parent_id, "colliding-source"),
    )
    conn.execute(
        "INSERT INTO etymon_citation (etymon_id, source_id) VALUES (?, ?)",
        (parent_id, "moved-source"),
    )
    conn.execute(
        "INSERT INTO etymon_citation (etymon_id, source_id) VALUES (?, ?)",
        (child_id, "colliding-source"),
    )
    conn.commit()
    conn.close()

    db = LexiconDB(db_path)
    state = {
        "old-english:gear": {
            "into": [{"suffix": "weir", "glosses": ["weir"], "primary": True}],
        }
    }
    counts = apply_etymon_splits(db, state, apply=True)

    # 1 citation moved (moved-source), 1 skipped on the conflict.
    assert counts["citations_moved"] == 1
    assert counts["citations_skipped_conflict"] == 1

    # Stranded row is still on the parent.
    parent_cits = sorted(
        row["source_id"]
        for row in db.conn.execute(
            "SELECT source_id FROM etymon_citation WHERE etymon_id = ?",
            (parent_id,),
        )
    )
    assert parent_cits == ["colliding-source"]
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


# ---------------------------------------------------------------------------
# CLI: lexicon curate-suppress-gloss
# ---------------------------------------------------------------------------


def test_cli_curate_suppress_gloss_appends_event(tmp_path: Path):
    curation_file = tmp_path / "_curation.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "curate-suppress-gloss",
            "old-english:dry",
            "wizard, sorcerer",
            "--reason",
            "no toponymy use",
            "--curation-file",
            str(curation_file),
        ],
    )
    assert result.exit_code == 0, result.output
    lines = curation_file.read_text().splitlines()
    assert len(lines) == 2  # source + event
    event = json.loads(lines[1])
    assert event["_type"] == "etymon_gloss_suppression"
    assert event["ref"] == "old-english:dry"
    assert event["suppressions"] == [{"gloss": "wizard, sorcerer", "reason": "no toponymy use"}]


def test_cli_curate_suppress_gloss_creates_source_row_on_first_invocation(
    tmp_path: Path,
):
    curation_file = tmp_path / "_curation.jsonl"
    CliRunner().invoke(
        cli_root,
        [
            "lexicon",
            "curate-suppress-gloss",
            "old-english:dry",
            "wizard, sorcerer",
            "--curation-file",
            str(curation_file),
        ],
    )
    source_row = json.loads(curation_file.read_text().splitlines()[0])
    assert source_row["_type"] == "source"
    assert source_row["ref"] == "manual-curation"


def test_cli_curate_suppress_gloss_multiple_glosses_one_event(tmp_path: Path):
    """Passing N glosses in one invocation writes ONE event with N
    entries — so KEYED_TYPES last-write-wins replay preserves all of
    them. Catches the regression that motivated collapsing the audit's
    4 corn events into one."""
    curation_file = tmp_path / "_curation.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "curate-suppress-gloss",
            "old-english:corn",
            "Crane",
            "Heron",
            "corn",
            "grain; corn",
            "--reason",
            "audit cleanup",
            "--curation-file",
            str(curation_file),
        ],
    )
    assert result.exit_code == 0, result.output
    event = json.loads(curation_file.read_text().splitlines()[1])
    assert len(event["suppressions"]) == 4
    assert [s["gloss"] for s in event["suppressions"]] == [
        "Crane",
        "Heron",
        "corn",
        "grain; corn",
    ]


def test_cli_curate_suppress_gloss_appends_without_dupe_source(tmp_path: Path):
    curation_file = tmp_path / "_curation.jsonl"
    runner = CliRunner()
    args = [
        "lexicon",
        "curate-suppress-gloss",
        "old-english:dry",
        "wizard, sorcerer",
        "--curation-file",
        str(curation_file),
    ]
    runner.invoke(cli_root, args)
    runner.invoke(cli_root, args)
    lines = curation_file.read_text().splitlines()
    # Source + 2 events; no second source row.
    assert sum(1 for line in lines if json.loads(line)["_type"] == "source") == 1
    assert sum(1 for line in lines if json.loads(line)["_type"] == "etymon_gloss_suppression") == 2


# ---------------------------------------------------------------------------
# CLI: lexicon curate-split-etymon
# ---------------------------------------------------------------------------


def test_cli_curate_split_etymon_appends_event(tmp_path: Path):
    curation_file = tmp_path / "_curation.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "curate-split-etymon",
            "old-english:gear",
            "--into",
            "suffix=weir|glosses=Weir (fishing enclosure);weir|primary",
            "--into",
            "suffix=year|glosses=year",
            "--reason",
            "real homograph",
            "--curation-file",
            str(curation_file),
        ],
    )
    assert result.exit_code == 0, result.output
    event = json.loads(curation_file.read_text().splitlines()[1])
    assert event["_type"] == "etymon_split"
    assert event["ref"] == "old-english:gear"
    assert event["reason"] == "real homograph"
    assert event["into"] == [
        {
            "suffix": "weir",
            "glosses": ["Weir (fishing enclosure)", "weir"],
            "primary": True,
        },
        {"suffix": "year", "glosses": ["year"]},
    ]


def test_cli_curate_split_etymon_rejects_unknown_key(tmp_path: Path):
    """Typo'd --into key (e.g. 'glosess' or 'color') should error
    rather than silently swallow the operator's intent."""
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "curate-split-etymon",
            "old-english:gear",
            "--into",
            "suffix=weir|glosess=oops|primary",
            "--curation-file",
            str(tmp_path / "_curation.jsonl"),
        ],
    )
    assert result.exit_code != 0
    assert "unknown key" in result.output.lower()


def test_cli_curate_split_etymon_rejects_empty_suffix(tmp_path: Path):
    """Empty `suffix=` value used to silently produce a no-op child
    in the applier. Now caught at the CLI."""
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "curate-split-etymon",
            "old-english:gear",
            "--into",
            "suffix=|glosses=foo",
            "--curation-file",
            str(tmp_path / "_curation.jsonl"),
        ],
    )
    assert result.exit_code != 0
    assert "suffix" in result.output.lower()


def test_cli_curate_split_etymon_rejects_missing_suffix(tmp_path: Path):
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "curate-split-etymon",
            "old-english:gear",
            "--into",
            "glosses=year",
            "--curation-file",
            str(tmp_path / "_curation.jsonl"),
        ],
    )
    assert result.exit_code != 0
    assert "suffix" in result.output.lower()


def test_cli_curate_split_etymon_rejects_multiple_primary(tmp_path: Path):
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "curate-split-etymon",
            "old-english:gear",
            "--into",
            "suffix=weir|glosses=Weir|primary",
            "--into",
            "suffix=year|glosses=year|primary",
            "--curation-file",
            str(tmp_path / "_curation.jsonl"),
        ],
    )
    assert result.exit_code != 0
    assert "primary" in result.output.lower()


# ---------------------------------------------------------------------------
# run_full_enrichment plumbing
# ---------------------------------------------------------------------------


def test_run_full_enrichment_threads_suppression_and_split_state(tmp_path: Path):
    """The new kwargs flow through and the applier outputs appear in
    the result dict. Also asserts the documented run order:
    apply-curation → apply-gloss-suppressions → apply-etymon-splits
    runs before the L3 derivations.

    skip_l3_derivations=True keeps this test fast (the L3 derivations
    are tested elsewhere)."""
    db_path = _build_db(tmp_path)
    _add_etymon_with_glosses_and_tags(
        db_path, "old-english", "dim", ["a down or hill", "alternative form of dimm"]
    )
    _add_etymon_with_glosses_and_tags(
        db_path, "old-norse", "lum", ["a pool", "loom or ember-goose"]
    )

    sup_state = {"old-english:dim": {"suppressions": [{"gloss": "alternative form of dimm"}]}}
    split_state = {
        "old-norse:lum": {
            "into": [
                {"suffix": "pool", "glosses": ["a pool"], "primary": True},
                {"suffix": "diver", "glosses": ["loom or ember-goose"]},
            ]
        }
    }

    with LexiconDB(db_path) as db:
        result = run_full_enrichment(
            db,
            apply=True,
            suppression_state=sup_state,
            split_state=split_state,
            skip_l3_derivations=True,
        )

    # Both applier names appear in the canonical order.
    order = result["order"]
    assert "apply-gloss-suppressions" in order
    assert "apply-etymon-splits" in order
    assert order.index("apply-gloss-suppressions") < order.index("apply-etymon-splits"), (
        "suppressions must run before splits (drop dead glosses before moving live ones)"
    )

    # Counts surfaced on the top-level result.
    assert result["gloss_suppressions"]["glosses_dropped"] == 1
    assert result["etymon_splits"]["splits_processed"] == 1
    assert result["etymon_splits"]["children_created"] == 2
