"""Tests for the new-toponym operator-review tool (wyrd-x82p Phase 2b.3).

Two CLI commands + module-level helpers:
* ``prepare-toponym-candidates`` — fuzzy-match suggestions
* ``commit-toponym-candidates`` — apply triage decisions

Tests use in-memory SQLite so the suite stays fast.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli as cli_root
from wyrd.generators.kenning.toponym_candidate_review import (
    PreparedCandidate,
    _toponym_rows,
    commit_triage_decisions,
    fuzzy_match_toponyms,
    prepare_candidate,
)

# Minimal schema fragments — toponym + toponym_attestation with the
# same UNIQUE indexes the production DB carries.
_FIXTURE_DDL = """
CREATE TABLE toponym (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    modern_name TEXT NOT NULL,
    country     TEXT,
    region      TEXT
);
CREATE UNIQUE INDEX idx_toponym_unique
    ON toponym(modern_name, COALESCE(country, ''), COALESCE(region, ''));
CREATE TABLE toponym_attestation (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    toponym_id  INTEGER NOT NULL REFERENCES toponym(id),
    form        TEXT NOT NULL,
    date_year   INTEGER,
    source_doc  TEXT NOT NULL
);
CREATE UNIQUE INDEX idx_attestation_unique
    ON toponym_attestation(toponym_id, form, date_year, source_doc);
"""


def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(_FIXTURE_DDL)
    return conn


def _seed_toponyms(conn: sqlite3.Connection, rows: list[dict]) -> dict[str, int]:
    name_to_id: dict[str, int] = {}
    for r in rows:
        cursor = conn.execute(
            "INSERT INTO toponym(modern_name, country, region) VALUES (?, ?, ?)",
            (r["modern_name"], r.get("country"), r.get("region")),
        )
        name_to_id[r["modern_name"]] = cursor.lastrowid
    return name_to_id


# ---------- fuzzy_match_toponyms ----------------------------------------


def test_fuzzy_match_returns_close_match():
    """A near-spelling-variant should surface as the top match."""
    conn = _make_conn()
    _seed_toponyms(conn, [{"modern_name": "Edlingham", "region": "Northumberland"}])
    toponyms = _toponym_rows(conn)
    suggestions = fuzzy_match_toponyms("Eadlingham", toponyms)
    assert len(suggestions) == 1
    assert suggestions[0].modern_name == "Edlingham"
    # Score reflects high similarity — exact-string ratio is 1.0;
    # one-letter difference here is ~0.94. Round trip preserves the
    # similarity floor.
    assert suggestions[0].fuzzy_score >= 0.8


def test_fuzzy_match_returns_empty_below_threshold():
    """A wildly different form (no shared substring) drops below the
    min_ratio cutoff and yields no suggestions — operator's review
    list stays focused."""
    conn = _make_conn()
    _seed_toponyms(conn, [{"modern_name": "Edlingham", "region": "Northumberland"}])
    toponyms = _toponym_rows(conn)
    suggestions = fuzzy_match_toponyms("ZyxwvutsrqXY", toponyms)
    assert suggestions == []


def test_fuzzy_match_caps_at_three_suggestions():
    """Default top-N is 3 — even with many candidates above
    threshold, the operator's eye doesn't get drowned."""
    conn = _make_conn()
    _seed_toponyms(
        conn,
        [
            {"modern_name": "Edlingham"},
            {"modern_name": "Edlinghame"},
            {"modern_name": "Edlinghan"},
            {"modern_name": "Edlinghum"},
            {"modern_name": "Edlinghom"},
        ],
    )
    toponyms = _toponym_rows(conn)
    suggestions = fuzzy_match_toponyms("Edlingham", toponyms)
    assert len(suggestions) == 3


def test_fuzzy_match_empty_form_returns_empty():
    """Empty/whitespace form must not match anything — without this
    guard the normalize step would produce an empty key that
    SequenceMatcher might score weirdly against short toponyms."""
    conn = _make_conn()
    _seed_toponyms(conn, [{"modern_name": "Edlingham"}])
    toponyms = _toponym_rows(conn)
    assert fuzzy_match_toponyms("", toponyms) == []
    assert fuzzy_match_toponyms("   ", toponyms) == []


def test_fuzzy_match_region_hint_boost_promotes_regional_match():
    """When two toponyms have the same fuzzy ratio against the form,
    the one whose region matches the hint comes first. Helps the
    operator pick the regionally-correct duplicate at a glance."""
    conn = _make_conn()
    _seed_toponyms(
        conn,
        [
            {"modern_name": "Newton", "region": "Berkshire"},
            {"modern_name": "Newton", "region": "Northumberland"},
        ],
    )
    toponyms = _toponym_rows(conn)
    suggestions = fuzzy_match_toponyms("Newton", toponyms, region_hint="Northumberland")
    assert len(suggestions) == 2
    # Hint-matched region comes FIRST.
    assert suggestions[0].region == "Northumberland"
    assert suggestions[1].region == "Berkshire"


def test_fuzzy_match_region_hint_does_not_promote_below_threshold():
    """A toponym whose name is unrelated to the form doesn't get
    rescued by the region-hint boost — the boost is a tiebreaker,
    not a free pass past the ratio threshold."""
    conn = _make_conn()
    _seed_toponyms(
        conn,
        [
            {"modern_name": "Newton", "region": "Northumberland"},
            {"modern_name": "Wessex Hill", "region": "Northumberland"},
        ],
    )
    toponyms = _toponym_rows(conn)
    # "Newton" is close to "Newton"; "Wessex Hill" shares almost no
    # letters. The hint shouldn't drag the Wessex Hill into the list.
    suggestions = fuzzy_match_toponyms("Newton", toponyms, region_hint="Northumberland")
    assert len(suggestions) == 1
    assert suggestions[0].modern_name == "Newton"


# ---------- prepare_candidate -------------------------------------------


def test_prepare_candidate_attaches_suggestions():
    conn = _make_conn()
    _seed_toponyms(conn, [{"modern_name": "Edlingham", "region": "Northumberland"}])
    toponyms = _toponym_rows(conn)
    raw = {
        "source_id": "mawer_1920",
        "form": "Eadlingham",
        "date_year": 1086,
        "region_hint": "Northumberland",
        "context": "the manor of Eadlingham",
    }
    prepared = prepare_candidate(raw, toponyms)
    assert isinstance(prepared, PreparedCandidate)
    assert prepared.action == "defer"
    assert prepared.form == "Eadlingham"
    assert prepared.date_year == 1086
    assert prepared.region_hint == "Northumberland"
    assert prepared.source_id == "mawer_1920"
    assert len(prepared.suggestions) == 1
    assert prepared.suggestions[0].modern_name == "Edlingham"


def test_prepare_candidate_handles_missing_optional_fields():
    """Raw candidates sometimes lack date_year/region_hint/context.
    Prepare must tolerate that without crashing."""
    conn = _make_conn()
    _seed_toponyms(conn, [{"modern_name": "Edlingham"}])
    toponyms = _toponym_rows(conn)
    raw = {"source_id": "mawer_1920", "form": "Edlingham"}
    prepared = prepare_candidate(raw, toponyms)
    assert prepared.date_year is None
    assert prepared.region_hint is None
    assert prepared.context == ""


def test_prepare_candidate_non_int_date_year_treated_as_null():
    conn = _make_conn()
    toponyms = _toponym_rows(conn)
    raw = {"source_id": "s", "form": "X", "date_year": "1086"}
    prepared = prepare_candidate(raw, toponyms)
    # string date_year — coerced to None at the prepare layer; the
    # commit layer will re-coerce numeric strings to int if the
    # operator ends up using map/create.
    assert prepared.date_year is None


# ---------- commit_triage_decisions: map --------------------------------


def test_commit_map_writes_attestation():
    conn = _make_conn()
    name_to_id = _seed_toponyms(conn, [{"modern_name": "Edlingham"}])
    rows = [
        {
            "source_id": "mawer_1920",
            "form": "Eadlingham",
            "date_year": 1086,
            "action": "map",
            "toponym_id": name_to_id["Edlingham"],
        }
    ]
    report = commit_triage_decisions(conn, rows, apply=True)
    assert report.mapped == 1
    assert report.errors == 0
    # Attestation row landed.
    row = conn.execute(
        "SELECT toponym_id, form, date_year, source_doc FROM toponym_attestation"
    ).fetchone()
    assert row["toponym_id"] == name_to_id["Edlingham"]
    assert row["form"] == "Eadlingham"
    assert row["date_year"] == 1086
    assert row["source_doc"] == "mawer_1920"


def test_commit_map_dry_run_does_not_write():
    conn = _make_conn()
    name_to_id = _seed_toponyms(conn, [{"modern_name": "Edlingham"}])
    rows = [
        {
            "source_id": "mawer_1920",
            "form": "Eadlingham",
            "action": "map",
            "toponym_id": name_to_id["Edlingham"],
        }
    ]
    report = commit_triage_decisions(conn, rows, apply=False)
    # Counter accurate, but no row written.
    assert report.mapped == 1
    count = conn.execute("SELECT COUNT(*) FROM toponym_attestation").fetchone()[0]
    assert count == 0


def test_commit_map_idempotent_on_reapply():
    conn = _make_conn()
    name_to_id = _seed_toponyms(conn, [{"modern_name": "Edlingham"}])
    rows = [
        {
            "source_id": "mawer_1920",
            "form": "Eadlingham",
            "date_year": 1086,
            "action": "map",
            "toponym_id": name_to_id["Edlingham"],
        }
    ]
    commit_triage_decisions(conn, rows, apply=True)
    # Re-apply: UNIQUE-index INSERT OR IGNORE silently no-ops.
    commit_triage_decisions(conn, rows, apply=True)
    count = conn.execute("SELECT COUNT(*) FROM toponym_attestation").fetchone()[0]
    assert count == 1


def test_commit_map_missing_toponym_id_is_error():
    conn = _make_conn()
    _seed_toponyms(conn, [{"modern_name": "Edlingham"}])
    rows = [
        {
            "source_id": "mawer_1920",
            "form": "Eadlingham",
            "action": "map",
            # toponym_id missing
        }
    ]
    report = commit_triage_decisions(conn, rows, apply=True)
    assert report.mapped == 0
    assert report.errors == 1
    idx, msg = report.error_records[0]
    assert idx == 0
    assert "toponym_id missing" in msg


def test_commit_map_unknown_toponym_id_is_error():
    """Operator typoed the toponym_id — must error, not silently
    create a foreign-key-violating attestation row."""
    conn = _make_conn()
    _seed_toponyms(conn, [{"modern_name": "Edlingham"}])
    rows = [
        {
            "source_id": "mawer_1920",
            "form": "Eadlingham",
            "action": "map",
            "toponym_id": 99999,  # doesn't exist
        }
    ]
    report = commit_triage_decisions(conn, rows, apply=True)
    assert report.mapped == 0
    assert report.errors == 1
    assert "doesn't exist" in report.error_records[0][1]


# ---------- commit_triage_decisions: create -----------------------------


def test_commit_create_inserts_toponym_and_attestation():
    conn = _make_conn()
    rows = [
        {
            "source_id": "mawer_1920",
            "form": "Suttone",
            "date_year": 1086,
            "action": "create",
            "create_modern_name": "Sutton-on-Tees",
            "create_country": "England",
            "create_region": "Durham",
        }
    ]
    report = commit_triage_decisions(conn, rows, apply=True)
    assert report.created == 1
    assert report.errors == 0
    tres = conn.execute("SELECT id, modern_name, country, region FROM toponym").fetchall()
    assert len(tres) == 1
    assert tres[0]["modern_name"] == "Sutton-on-Tees"
    assert tres[0]["country"] == "England"
    assert tres[0]["region"] == "Durham"
    ares = conn.execute(
        "SELECT toponym_id, form, date_year, source_doc FROM toponym_attestation"
    ).fetchall()
    assert len(ares) == 1
    assert ares[0]["toponym_id"] == tres[0]["id"]
    assert ares[0]["form"] == "Suttone"


def test_commit_create_unique_collision_demotes_to_map():
    """If an existing toponym already has (modern_name, country,
    region), CREATE silently converts to MAP — prevents accidental
    duplicate toponym rows when the operator didn't notice the
    fuzzy suggestion."""
    conn = _make_conn()
    name_to_id = _seed_toponyms(
        conn,
        [{"modern_name": "Edlingham", "country": "England", "region": "Northumberland"}],
    )
    rows = [
        {
            "source_id": "mawer_1920",
            "form": "Eadlingham",
            "action": "create",
            "create_modern_name": "Edlingham",
            "create_country": "England",
            "create_region": "Northumberland",
        }
    ]
    report = commit_triage_decisions(conn, rows, apply=True)
    # Counted as mapped, not created.
    assert report.created == 0
    assert report.mapped == 1
    # Still only one toponym row (the pre-seeded one).
    count = conn.execute("SELECT COUNT(*) FROM toponym").fetchone()[0]
    assert count == 1
    # Attestation written, pointing at the existing toponym.
    ares = conn.execute("SELECT toponym_id, form FROM toponym_attestation").fetchone()
    assert ares["toponym_id"] == name_to_id["Edlingham"]


def test_commit_create_missing_modern_name_is_error():
    conn = _make_conn()
    rows = [
        {
            "source_id": "mawer_1920",
            "form": "Suttone",
            "action": "create",
            # create_modern_name missing
        }
    ]
    report = commit_triage_decisions(conn, rows, apply=True)
    assert report.created == 0
    assert report.errors == 1
    assert "create_modern_name missing" in report.error_records[0][1]


def test_commit_create_null_region_and_country_persists_as_null():
    """Operator may leave region/country blank for an as-yet-
    unlocalized place. The new row's region/country columns must
    be NULL, not empty-string (preserves UNIQUE-index semantics)."""
    conn = _make_conn()
    rows = [
        {
            "source_id": "x",
            "form": "Foo",
            "action": "create",
            "create_modern_name": "Foo",
            "create_country": "",  # operator typed empty
            "create_region": None,  # operator left null
        }
    ]
    commit_triage_decisions(conn, rows, apply=True)
    row = conn.execute("SELECT country, region FROM toponym WHERE modern_name='Foo'").fetchone()
    assert row["country"] is None
    assert row["region"] is None


# ---------- commit_triage_decisions: skip / defer / errors ---------------


def test_commit_skip_no_db_writes():
    conn = _make_conn()
    _seed_toponyms(conn, [{"modern_name": "Edlingham"}])
    rows = [
        {"source_id": "x", "form": "Random", "action": "skip"},
    ]
    report = commit_triage_decisions(conn, rows, apply=True)
    assert report.skipped == 1
    assert report.errors == 0
    count = conn.execute("SELECT COUNT(*) FROM toponym_attestation").fetchone()[0]
    assert count == 0


def test_commit_defer_no_db_writes():
    conn = _make_conn()
    rows = [
        {"source_id": "x", "form": "Random", "action": "defer"},
    ]
    report = commit_triage_decisions(conn, rows, apply=True)
    assert report.deferred == 1
    assert report.errors == 0


def test_commit_unknown_action_is_error():
    conn = _make_conn()
    rows = [
        {"source_id": "x", "form": "Random", "action": "frobnicate"},
    ]
    report = commit_triage_decisions(conn, rows, apply=True)
    assert report.errors == 1
    assert "unknown action: 'frobnicate'" in report.error_records[0][1]


def test_commit_action_case_insensitive():
    """`action: MAP` works like `action: map` — operator's caps don't
    matter."""
    conn = _make_conn()
    name_to_id = _seed_toponyms(conn, [{"modern_name": "Edlingham"}])
    rows = [
        {
            "source_id": "x",
            "form": "Edlingham",
            "action": "MAP",
            "toponym_id": name_to_id["Edlingham"],
        }
    ]
    report = commit_triage_decisions(conn, rows, apply=True)
    assert report.mapped == 1
    assert report.errors == 0


def test_commit_non_dict_row_is_error():
    """A list/int slipped into the rows must not crash; treated as
    an error row with index recorded."""
    conn = _make_conn()
    rows = ["not a dict", {"source_id": "x", "form": "y", "action": "defer"}]  # type: ignore[list-item]
    report = commit_triage_decisions(conn, rows, apply=True)
    assert report.errors == 1
    assert report.deferred == 1
    assert report.error_records[0][0] == 0


def test_commit_does_not_auto_commit():
    """Caller-owned transaction control — mirrors Phase 1 + 2b.1
    convention. If the caller rolls back, no rows persist."""
    conn = _make_conn()
    rows = [
        {
            "source_id": "x",
            "form": "Foo",
            "action": "create",
            "create_modern_name": "Foo",
        }
    ]
    commit_triage_decisions(conn, rows, apply=True)
    conn.rollback()
    count = conn.execute("SELECT COUNT(*) FROM toponym").fetchone()[0]
    assert count == 0


# ---------- CLI: prepare-toponym-candidates ------------------------------


def _make_db(path: Path, toponyms: list[dict]) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_FIXTURE_DDL)
        for t in toponyms:
            conn.execute(
                "INSERT INTO toponym(modern_name, country, region) VALUES (?, ?, ?)",
                (t["modern_name"], t.get("country"), t.get("region")),
            )
        conn.commit()
    finally:
        conn.close()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_cli_prepare_attaches_suggestions(tmp_path):
    db = tmp_path / "lex.db"
    _make_db(
        db,
        [
            {"modern_name": "Edlingham", "region": "Northumberland"},
            {"modern_name": "Berwick", "region": "Northumberland"},
        ],
    )
    candidates = tmp_path / "candidates.jsonl"
    _write_jsonl(
        candidates,
        [
            {
                "source_id": "mawer_1920",
                "form": "Eadlingham",
                "date_year": 1086,
                "region_hint": "Northumberland",
                "context": "the manor of Eadlingham",
            }
        ],
    )
    triage = tmp_path / "triage.jsonl"

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "prepare-toponym-candidates",
            "--jsonl",
            str(candidates),
            "--db",
            str(db),
            "--out",
            str(triage),
        ],
    )
    assert result.exit_code == 0, result.output
    assert triage.exists()
    row = json.loads(triage.read_text().strip())
    assert row["action"] == "defer"
    assert row["form"] == "Eadlingham"
    assert len(row["suggestions"]) >= 1
    assert any(s["modern_name"] == "Edlingham" for s in row["suggestions"])


def test_cli_prepare_refuses_existing_out_without_force(tmp_path):
    db = tmp_path / "lex.db"
    _make_db(db, [])
    candidates = tmp_path / "candidates.jsonl"
    _write_jsonl(candidates, [])
    triage = tmp_path / "triage.jsonl"
    triage.write_text("PREVIOUS\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "prepare-toponym-candidates",
            "--jsonl",
            str(candidates),
            "--db",
            str(db),
            "--out",
            str(triage),
        ],
    )
    assert result.exit_code != 0
    assert "already exists" in result.output
    assert triage.read_text() == "PREVIOUS\n"


def test_cli_prepare_force_overwrites(tmp_path):
    db = tmp_path / "lex.db"
    _make_db(db, [{"modern_name": "Edlingham"}])
    candidates = tmp_path / "candidates.jsonl"
    _write_jsonl(candidates, [{"source_id": "x", "form": "Edlingham"}])
    triage = tmp_path / "triage.jsonl"
    triage.write_text("PREVIOUS\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "prepare-toponym-candidates",
            "--jsonl",
            str(candidates),
            "--db",
            str(db),
            "--out",
            str(triage),
            "--force",
        ],
    )
    assert result.exit_code == 0, result.output
    new = triage.read_text()
    assert "PREVIOUS" not in new
    assert "Edlingham" in new


def test_cli_prepare_invalid_json_errors_with_line(tmp_path):
    db = tmp_path / "lex.db"
    _make_db(db, [])
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text('{"source_id":"x","form":"y"}\nNOT VALID\n', encoding="utf-8")
    triage = tmp_path / "triage.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "prepare-toponym-candidates",
            "--jsonl",
            str(candidates),
            "--db",
            str(db),
            "--out",
            str(triage),
        ],
    )
    assert result.exit_code != 0
    assert "invalid JSON" in result.output
    assert ":2:" in result.output


def test_cli_prepare_non_dict_jsonl_row_errors(tmp_path):
    db = tmp_path / "lex.db"
    _make_db(db, [])
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text('{"source_id":"x","form":"y"}\n[1,2,3]\n', encoding="utf-8")
    triage = tmp_path / "triage.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "prepare-toponym-candidates",
            "--jsonl",
            str(candidates),
            "--db",
            str(db),
            "--out",
            str(triage),
        ],
    )
    assert result.exit_code != 0
    assert "not a JSON object" in result.output


def test_cli_prepare_atomic_write_no_tmp_leftover(tmp_path):
    db = tmp_path / "lex.db"
    _make_db(db, [])
    candidates = tmp_path / "candidates.jsonl"
    _write_jsonl(candidates, [{"source_id": "x", "form": "y"}])
    triage = tmp_path / "triage.jsonl"

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "prepare-toponym-candidates",
            "--jsonl",
            str(candidates),
            "--db",
            str(db),
            "--out",
            str(triage),
        ],
    )
    assert result.exit_code == 0, result.output
    assert triage.exists()
    assert not triage.with_suffix(".jsonl.tmp").exists()


# ---------- CLI: commit-toponym-candidates -------------------------------


def test_cli_commit_dry_run_predicts_counts(tmp_path):
    db = tmp_path / "lex.db"
    _make_db(db, [{"modern_name": "Edlingham"}])
    triage = tmp_path / "triage.jsonl"
    _write_jsonl(
        triage,
        [
            {
                "source_id": "mawer",
                "form": "Eadlingham",
                "date_year": 1086,
                "action": "map",
                "toponym_id": 1,
            },
            {"source_id": "x", "form": "Junk", "action": "skip"},
            {"source_id": "x", "form": "Later", "action": "defer"},
        ],
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "commit-toponym-candidates",
            "--jsonl",
            str(triage),
            "--db",
            str(db),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "mapped=1" in result.output
    assert "skipped=1" in result.output
    assert "deferred=1" in result.output
    assert "dry-run" in result.output
    # No rows in DB.
    conn = sqlite3.connect(db)
    count = conn.execute("SELECT COUNT(*) FROM toponym_attestation").fetchone()[0]
    conn.close()
    assert count == 0


def test_cli_commit_apply_writes_decisions(tmp_path):
    db = tmp_path / "lex.db"
    _make_db(db, [{"modern_name": "Edlingham"}])
    triage = tmp_path / "triage.jsonl"
    _write_jsonl(
        triage,
        [
            {
                "source_id": "mawer",
                "form": "Eadlingham",
                "date_year": 1086,
                "action": "map",
                "toponym_id": 1,
            },
            {
                "source_id": "mawer",
                "form": "Suttone",
                "date_year": 1086,
                "action": "create",
                "create_modern_name": "Sutton",
                "create_region": "Durham",
            },
        ],
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "commit-toponym-candidates",
            "--jsonl",
            str(triage),
            "--db",
            str(db),
            "--apply",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "APPLIED" in result.output
    conn = sqlite3.connect(db)
    # 2 toponym rows (Edlingham + new Sutton); 2 attestations.
    tcount = conn.execute("SELECT COUNT(*) FROM toponym").fetchone()[0]
    acount = conn.execute("SELECT COUNT(*) FROM toponym_attestation").fetchone()[0]
    conn.close()
    assert tcount == 2
    assert acount == 2


def test_cli_commit_verbose_surfaces_per_row_errors(tmp_path):
    db = tmp_path / "lex.db"
    _make_db(db, [])
    triage = tmp_path / "triage.jsonl"
    _write_jsonl(
        triage,
        [
            {"source_id": "x", "form": "Y", "action": "frobnicate"},
        ],
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "commit-toponym-candidates",
            "--jsonl",
            str(triage),
            "--db",
            str(db),
            "--verbose",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "errors=1" in result.output
    assert "frobnicate" in result.output


def test_cli_commit_errors_hint_appears_without_verbose(tmp_path):
    """When errors are present but --verbose isn't passed, the
    summary line should hint at how to see per-row error messages."""
    db = tmp_path / "lex.db"
    _make_db(db, [])
    triage = tmp_path / "triage.jsonl"
    _write_jsonl(
        triage,
        [{"source_id": "x", "form": "Y", "action": "garbage"}],
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "commit-toponym-candidates",
            "--jsonl",
            str(triage),
            "--db",
            str(db),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "errors=1" in result.output
    assert "--verbose" in result.output


def test_cli_commit_apply_idempotent_reapply(tmp_path):
    """Re-running commit with the same triage JSONL is a no-op
    for already-applied rows. New rows wouldn't be in this test
    but the existing ones must not multiply attestations."""
    db = tmp_path / "lex.db"
    _make_db(db, [{"modern_name": "Edlingham"}])
    triage = tmp_path / "triage.jsonl"
    _write_jsonl(
        triage,
        [
            {
                "source_id": "mawer",
                "form": "Eadlingham",
                "date_year": 1086,
                "action": "map",
                "toponym_id": 1,
            }
        ],
    )
    runner = CliRunner()
    # First apply.
    runner.invoke(
        cli_root,
        [
            "lexicon",
            "commit-toponym-candidates",
            "--jsonl",
            str(triage),
            "--db",
            str(db),
            "--apply",
        ],
    )
    # Re-apply.
    runner.invoke(
        cli_root,
        [
            "lexicon",
            "commit-toponym-candidates",
            "--jsonl",
            str(triage),
            "--db",
            str(db),
            "--apply",
        ],
    )
    # Still only one attestation row.
    conn = sqlite3.connect(db)
    acount = conn.execute("SELECT COUNT(*) FROM toponym_attestation").fetchone()[0]
    conn.close()
    assert acount == 1


def test_cli_commit_invalid_json_errors_with_line(tmp_path):
    db = tmp_path / "lex.db"
    _make_db(db, [])
    triage = tmp_path / "triage.jsonl"
    triage.write_text(
        '{"source_id":"x","form":"y","action":"defer"}\nNOT VALID\n',
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "commit-toponym-candidates",
            "--jsonl",
            str(triage),
            "--db",
            str(db),
        ],
    )
    assert result.exit_code != 0
    assert "invalid JSON" in result.output
    assert ":2:" in result.output
