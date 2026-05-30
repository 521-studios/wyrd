"""wyrd-bo01.1: ``_insert_toponym`` merges on conflict instead of crashing.

The ``idx_toponym_unique`` index makes (modern_name, country, region) unique.
An additive ``build_from_jsonl`` replay whose toponym already exists in the DB
(e.g. KEPN ingesting Bedford/Bedfordshire after a scholarly source mined it)
must REUSE the existing row's id — merging the new decomposition onto it —
rather than abort the whole build with ``sqlite3.IntegrityError``.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from wyrd.generators.kenning.jsonl.build import build_from_jsonl, jsonl_paths_in
from wyrd.generators.kenning.lexicon import init_schema


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )


def _fresh_db(tmp_path: Path) -> Path:
    db = tmp_path / "lexicon.db"
    init_schema(db)
    return db


def test_additive_replay_reuses_existing_toponym_on_conflict(tmp_path: Path):
    """Pre-existing Cotton/England/Huntingdonshire + a JSONL replay that
    re-declares the same toponym → one row (reused), build does not crash,
    and the replay's etymology element attaches to the existing toponym."""
    db = _fresh_db(tmp_path)
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    # Pre-seed the toponym (as a prior scholarly source would have).
    cur = conn.execute(
        "INSERT INTO toponym (modern_name, country, region) VALUES (?, ?, ?)",
        ("Cotton", "England", "Huntingdonshire"),
    )
    existing_id = cur.lastrowid
    conn.commit()

    _write_jsonl(
        tmp_path / "kepn.jsonl",
        [
            {"_type": "source", "ref": "kepn", "title": "KEPN"},
            {
                "_type": "etymon",
                "ref": "old-english:cot",
                "language": "old-english",
                "canonical_form": "cot",
            },
            {
                "_type": "toponym",
                "ref": "Cotton@Hunts",
                "modern_name": "Cotton",
                "country": "England",
                "region": "Huntingdonshire",
            },
            {
                "_type": "etymology_element",
                "toponym_ref": "Cotton@Hunts",
                "elements": [{"ordinal": 1, "etymon_ref": "old-english:cot"}],
            },
        ],
    )

    # Must not raise sqlite3.IntegrityError on the colliding toponym.
    build_from_jsonl(conn, jsonl_paths_in(tmp_path))
    conn.commit()

    # Reused, not duplicated.
    assert conn.execute("SELECT COUNT(*) FROM toponym").fetchone()[0] == 1
    # The replay's decomposition attached to the pre-existing toponym id
    # (element → toponym_etymology → toponym).
    rows = conn.execute(
        "SELECT DISTINCT te.toponym_id FROM toponym_etymology_element el "
        "JOIN toponym_etymology te ON te.id = el.toponym_etymology_id"
    ).fetchall()
    assert [r["toponym_id"] for r in rows] == [existing_id]
    conn.close()


def test_additive_replay_inserts_new_toponym_when_no_conflict(tmp_path: Path):
    """A toponym not already present is inserted normally (the non-conflict
    branch still returns a fresh id)."""
    db = _fresh_db(tmp_path)
    conn = sqlite3.connect(db)
    _write_jsonl(
        tmp_path / "src.jsonl",
        [
            {"_type": "source", "ref": "src", "title": "S"},
            {
                "_type": "toponym",
                "ref": "Newtown@Devon",
                "modern_name": "Newtown",
                "country": "England",
                "region": "Devon",
            },
        ],
    )
    build_from_jsonl(conn, jsonl_paths_in(tmp_path))
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM toponym").fetchone()[0] == 1
    assert conn.execute("SELECT modern_name FROM toponym").fetchone()[0] == "Newtown"
    conn.close()
