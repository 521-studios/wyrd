"""Tests for the LLM-mention DB ingester (wyrd-x82p Phase 2b.1).

The ingester reads pre-extracted mentions (JSONL emitted by Phase 2
``mine-toponym-mentions``) and writes ``toponym_attestation`` rows
for resolvable mentions. Tests use in-memory SQLite databases so the
suite stays fast and doesn't touch ~/.wyrd/lexicon.db.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli as cli_root
from wyrd.generators.kenning.toponym_mention_ingest import (
    ResolverIndexes,
    build_resolver_indexes,
    ingest_mentions,
    resolve_mention,
)

# Schema mirrors the production DDL fragments needed for these tests:
# toponym (id, modern_name, region), toponym_attestation (toponym_id,
# form, date_year, source_doc) + UNIQUE index.
_FIXTURE_DDL = """
CREATE TABLE toponym (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    modern_name TEXT NOT NULL,
    country     TEXT,
    region      TEXT
);
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
    """Insert toponym rows and return name → id map for assertions."""
    name_to_id: dict[str, int] = {}
    for row in rows:
        cursor = conn.execute(
            "INSERT INTO toponym(modern_name, country, region) VALUES (?, ?, ?)",
            (row["modern_name"], row.get("country"), row.get("region")),
        )
        name_to_id[row["modern_name"]] = cursor.lastrowid
    return name_to_id


# ---------- build_resolver_indexes ----------------------------------------


def test_build_resolver_indexes_includes_unambiguous_form():
    conn = _make_conn()
    _seed_toponyms(
        conn,
        [
            {"modern_name": "Edlingham", "region": "Northumberland"},
        ],
    )
    indexes = build_resolver_indexes(conn)
    assert "edlingham" in indexes.form_to_ids
    assert len(indexes.form_to_ids["edlingham"]) == 1


def test_build_resolver_indexes_keeps_ambiguous_form():
    """Phase 1's lookup excluded ambiguous forms; Phase 2b.1 KEEPS them
    so the resolver can use region_hint to pick. Both toponyms must
    appear in the candidate set."""
    conn = _make_conn()
    _seed_toponyms(
        conn,
        [
            {"modern_name": "Newton", "region": "Northumberland"},
            {"modern_name": "Newton", "region": "Berkshire"},
        ],
    )
    indexes = build_resolver_indexes(conn)
    assert len(indexes.form_to_ids["newton"]) == 2


def test_build_resolver_indexes_captures_region_from_toponym_row():
    conn = _make_conn()
    name_to_id = _seed_toponyms(
        conn,
        [
            {"modern_name": "Edlingham", "region": "Northumberland"},
            {"modern_name": "Anonymous", "region": None},
        ],
    )
    indexes = build_resolver_indexes(conn)
    assert indexes.toponym_regions[name_to_id["Edlingham"]] == "northumberland"
    assert indexes.toponym_regions[name_to_id["Anonymous"]] is None


def test_build_resolver_indexes_includes_attestation_forms():
    """Historical forms attested elsewhere (Edlingaham, Eadwulfingaham)
    should also reach the form_to_ids lookup so a re-mention of one of
    those historical spellings still resolves."""
    conn = _make_conn()
    name_to_id = _seed_toponyms(
        conn,
        [{"modern_name": "Edlingham", "region": "Northumberland"}],
    )
    conn.execute(
        "INSERT INTO toponym_attestation(toponym_id, form, date_year, source_doc) "
        "VALUES (?, ?, ?, ?)",
        (name_to_id["Edlingham"], "Eadwulfingaham", 1086, "domesday"),
    )
    indexes = build_resolver_indexes(conn)
    assert "eadwulfingaham" in indexes.form_to_ids
    assert next(iter(indexes.form_to_ids["eadwulfingaham"])) == name_to_id["Edlingham"]


# ---------- resolve_mention -----------------------------------------------


def test_resolve_mention_unambiguous_returns_id():
    indexes = ResolverIndexes(
        form_to_ids={"edlingham": {7}},
        toponym_regions={7: "northumberland"},
    )
    assert resolve_mention("Edlingham", None, indexes) == 7
    assert resolve_mention("Edlingham", "Northumberland", indexes) == 7


def test_resolve_mention_no_candidates_returns_none():
    indexes = ResolverIndexes(form_to_ids={}, toponym_regions={})
    assert resolve_mention("UnknownPlace", None, indexes) is None
    assert resolve_mention("UnknownPlace", "Northumberland", indexes) is None


def test_resolve_mention_ambiguous_without_hint_returns_none():
    """Phase 1's "lowest id wins" was the round-1 review HIGH finding —
    pattern-based resolution can't disambiguate, so it defers. Phase
    2b.1 inherits that same defer-without-hint policy because picking
    one of two homonyms blindly produces wrong attestations (Hereford
    in a Herefordshire source mapping to Hereford@Cheshire was the
    historical Phase-1 failure mode)."""
    indexes = ResolverIndexes(
        form_to_ids={"newton": {1, 2}},
        toponym_regions={1: "northumberland", 2: "berkshire"},
    )
    assert resolve_mention("Newton", None, indexes) is None


def test_resolve_mention_ambiguous_with_matching_region_hint_resolves():
    indexes = ResolverIndexes(
        form_to_ids={"newton": {1, 2}},
        toponym_regions={1: "northumberland", 2: "berkshire"},
    )
    assert resolve_mention("Newton", "Northumberland", indexes) == 1
    assert resolve_mention("Newton", "Berkshire", indexes) == 2


def test_resolve_mention_region_hint_matches_substring_either_direction():
    """The LLM may emit "Northumberland" (matches) or "co. Northumberland"
    (toponym region is shorter — must still match via the longer-hint
    substring direction). Both directions of substring containment
    succeed."""
    indexes = ResolverIndexes(
        form_to_ids={"newton": {1, 2}},
        toponym_regions={1: "northumberland", 2: "berkshire"},
    )
    assert resolve_mention("Newton", "co. Northumberland, England", indexes) == 1


def test_resolve_mention_ambiguous_with_matching_multiple_regions_returns_none():
    """If region_hint somehow matches MULTIPLE candidates' regions
    (degenerate: two toponyms both in 'northumberland'), we can't
    pick — defer to operator review."""
    indexes = ResolverIndexes(
        form_to_ids={"newton": {1, 2}},
        toponym_regions={1: "northumberland", 2: "northumberland"},
    )
    assert resolve_mention("Newton", "Northumberland", indexes) is None


def test_resolve_mention_ambiguous_with_unmatched_region_hint_returns_none():
    indexes = ResolverIndexes(
        form_to_ids={"newton": {1, 2}},
        toponym_regions={1: "northumberland", 2: "berkshire"},
    )
    assert resolve_mention("Newton", "Wessex", indexes) is None


def test_resolve_mention_normalizes_form_case_and_punctuation():
    indexes = ResolverIndexes(
        form_to_ids={"edlingham": {7}},
        toponym_regions={7: "northumberland"},
    )
    # Phase 1's normalize_for_match handles surrounding punctuation +
    # case-fold. Re-verify here so the integration is locked in.
    assert resolve_mention("EDLINGHAM", None, indexes) == 7
    assert resolve_mention("'Edlingham'", None, indexes) == 7
    assert resolve_mention("Edlingham,", None, indexes) == 7


def test_resolve_mention_empty_form_returns_none():
    indexes = ResolverIndexes(
        form_to_ids={"edlingham": {7}},
        toponym_regions={7: "northumberland"},
    )
    assert resolve_mention("", None, indexes) is None
    assert resolve_mention("   ", None, indexes) is None


def test_resolve_mention_skips_toponym_with_null_region():
    """A toponym whose region is None can't be selected by region_hint
    — we don't know what region it's in, so we can't confirm a match.
    This prevents picking a region-less duplicate when the LLM has
    region context."""
    indexes = ResolverIndexes(
        form_to_ids={"newton": {1, 2}},
        toponym_regions={1: "northumberland", 2: None},
    )
    # hint matches toponym 1's region; toponym 2 has no region info →
    # disambiguates to toponym 1.
    assert resolve_mention("Newton", "Northumberland", indexes) == 1


# ---------- ingest_mentions (orchestrator) -------------------------------


def test_ingest_mentions_resolved_mention_predicts_insert_in_dry_run():
    """Preflight uniqueness check makes dry-run accurate — same
    invariant Phase 1 enforced (PR #212)."""
    conn = _make_conn()
    name_to_id = _seed_toponyms(
        conn,
        [{"modern_name": "Edlingham", "region": "Northumberland"}],
    )
    indexes = build_resolver_indexes(conn)
    mentions = [
        {"form": "Edlingham", "date_year": 1086, "region_hint": None, "context": "x"},
    ]
    report = ingest_mentions(conn, "source-1", mentions, indexes, apply=False)
    assert report.resolved == 1
    assert report.inserted == 1
    assert report.already_present == 0
    # No row actually written in dry-run.
    row_count = conn.execute("SELECT COUNT(*) FROM toponym_attestation").fetchone()[0]
    assert row_count == 0
    _ = name_to_id  # silence unused warning


def test_ingest_mentions_apply_writes_row():
    conn = _make_conn()
    _seed_toponyms(
        conn,
        [{"modern_name": "Edlingham", "region": "Northumberland"}],
    )
    indexes = build_resolver_indexes(conn)
    mentions = [
        {"form": "Edlingham", "date_year": 1086, "region_hint": None, "context": "x"},
    ]
    report = ingest_mentions(conn, "source-1", mentions, indexes, apply=True)
    assert report.inserted == 1
    row = conn.execute(
        "SELECT toponym_id, form, date_year, source_doc FROM toponym_attestation"
    ).fetchone()
    assert row["form"] == "Edlingham"
    assert row["date_year"] == 1086
    assert row["source_doc"] == "source-1"


def test_ingest_mentions_idempotent_re_apply_inserts_zero():
    conn = _make_conn()
    _seed_toponyms(
        conn,
        [{"modern_name": "Edlingham", "region": "Northumberland"}],
    )
    indexes = build_resolver_indexes(conn)
    mentions = [
        {"form": "Edlingham", "date_year": 1086, "region_hint": None, "context": "x"},
    ]
    ingest_mentions(conn, "source-1", mentions, indexes, apply=True)
    # Re-apply: zero inserts, all go to already_present.
    report2 = ingest_mentions(conn, "source-1", mentions, indexes, apply=True)
    assert report2.inserted == 0
    assert report2.already_present == 1


def test_ingest_mentions_dedups_null_year_with_is_null_semantics():
    """SQLite's UNIQUE index treats NULL as distinct; without explicit
    IS-NULL preflight, two undated mentions of the same form for the
    same toponym in the same source would both insert. The ingester
    uses IS NULL preflight to enforce the operator's intent ("same
    form + same source + no date = same attestation")."""
    conn = _make_conn()
    _seed_toponyms(
        conn,
        [{"modern_name": "Edlingham", "region": "Northumberland"}],
    )
    indexes = build_resolver_indexes(conn)
    mentions = [
        {"form": "Edlingham", "date_year": None, "region_hint": None, "context": "first"},
        {"form": "Edlingham", "date_year": None, "region_hint": None, "context": "second"},
    ]
    report = ingest_mentions(conn, "source-1", mentions, indexes, apply=True)
    # First inserts; second sees existing null-year row → already_present.
    assert report.inserted == 1
    assert report.already_present == 1
    # Only one row in DB.
    count = conn.execute("SELECT COUNT(*) FROM toponym_attestation").fetchone()[0]
    assert count == 1


def test_ingest_mentions_unresolved_records_captured():
    conn = _make_conn()
    _seed_toponyms(
        conn,
        [{"modern_name": "Edlingham", "region": "Northumberland"}],
    )
    indexes = build_resolver_indexes(conn)
    mentions = [
        {"form": "UnknownPlace", "date_year": 1086, "region_hint": None, "context": "ctx"},
    ]
    report = ingest_mentions(conn, "source-1", mentions, indexes, apply=True)
    assert report.resolved == 0
    assert report.unresolved == 1
    assert report.inserted == 0
    assert len(report.unresolved_records) == 1
    rec = report.unresolved_records[0]
    assert rec["form"] == "UnknownPlace"
    assert rec["source_id"] == "source-1"
    assert rec["context"] == "ctx"


def test_ingest_mentions_ambiguous_form_without_hint_unresolved():
    conn = _make_conn()
    _seed_toponyms(
        conn,
        [
            {"modern_name": "Newton", "region": "Northumberland"},
            {"modern_name": "Newton", "region": "Berkshire"},
        ],
    )
    indexes = build_resolver_indexes(conn)
    mentions = [
        {"form": "Newton", "date_year": 1086, "region_hint": None, "context": "x"},
    ]
    report = ingest_mentions(conn, "source-1", mentions, indexes, apply=True)
    assert report.unresolved == 1
    assert report.inserted == 0


def test_ingest_mentions_ambiguous_form_with_region_hint_resolves():
    conn = _make_conn()
    # Two Newtons — _seed_toponyms returns name→id, but the dict
    # overwrites on duplicate name. Capture by region to compare.
    _seed_toponyms(
        conn,
        [
            {"modern_name": "Newton", "region": "Northumberland"},
            {"modern_name": "Newton", "region": "Berkshire"},
        ],
    )
    nb_id = conn.execute(
        "SELECT id FROM toponym WHERE modern_name='Newton' AND region='Northumberland'"
    ).fetchone()["id"]
    indexes = build_resolver_indexes(conn)
    mentions = [
        {
            "form": "Newton",
            "date_year": 1086,
            "region_hint": "Northumberland",
            "context": "x",
        },
    ]
    report = ingest_mentions(conn, "source-1", mentions, indexes, apply=True)
    assert report.resolved == 1
    assert report.inserted == 1
    row = conn.execute("SELECT toponym_id FROM toponym_attestation WHERE form='Newton'").fetchone()
    # Region hint disambiguated to the Northumberland Newton specifically.
    assert row["toponym_id"] == nb_id


def test_ingest_mentions_does_not_commit():
    """Caller-owned transaction control: mirrors Phase 1's PR #212
    finding. The function may not auto-commit; if the caller rolls
    back, no rows persist."""
    conn = _make_conn()
    _seed_toponyms(
        conn,
        [{"modern_name": "Edlingham", "region": "Northumberland"}],
    )
    indexes = build_resolver_indexes(conn)
    mentions = [
        {"form": "Edlingham", "date_year": 1086, "region_hint": None, "context": "x"},
    ]
    ingest_mentions(conn, "source-1", mentions, indexes, apply=True)
    conn.rollback()
    # Row was inserted into the transaction but rolled back; no rows now.
    count = conn.execute("SELECT COUNT(*) FROM toponym_attestation").fetchone()[0]
    assert count == 0


def test_ingest_mentions_empty_form_unresolved_not_crashed():
    conn = _make_conn()
    _seed_toponyms(
        conn,
        [{"modern_name": "Edlingham", "region": "Northumberland"}],
    )
    indexes = build_resolver_indexes(conn)
    mentions = [
        {"form": "", "date_year": 1086, "region_hint": None, "context": "x"},
        {"form": "   ", "date_year": 1086, "region_hint": None, "context": "x"},
    ]
    report = ingest_mentions(conn, "source-1", mentions, indexes, apply=True)
    assert report.unresolved == 2
    assert report.inserted == 0
    # Empty-form rows should NOT pollute the unresolved-records sink
    # (they have no form to write).
    assert report.unresolved_records == []


def test_ingest_mentions_non_int_year_treated_as_null():
    """A mention with a bogus date_year (string, bool, float) should
    not crash the ingester — treat as None and continue. Phase 2's
    extractor already clamps these, but the ingester accepts the
    raw JSONL so we re-validate at the boundary."""
    conn = _make_conn()
    _seed_toponyms(
        conn,
        [{"modern_name": "Edlingham", "region": "Northumberland"}],
    )
    indexes = build_resolver_indexes(conn)
    mentions = [
        {"form": "Edlingham", "date_year": "not a year", "region_hint": None},
        {"form": "Edlingham", "date_year": True, "region_hint": None},
        {"form": "Edlingham", "date_year": 1086.5, "region_hint": None},
    ]
    report = ingest_mentions(conn, "source-1", mentions, indexes, apply=True)
    assert report.resolved == 3
    # All three resolve to the same toponym + null date_year + same
    # source — IS-NULL dedup means only one row inserts.
    assert report.inserted == 1
    assert report.already_present == 2


# ---------- CLI integration ----------------------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _make_db(path: Path, toponyms: list[dict]) -> None:
    """Create an on-disk DB file with the production schema fragments
    the CLI needs (toponym + toponym_attestation + UNIQUE index)."""
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


def test_cli_dry_run_reports_inserts_without_writing(tmp_path):
    db = tmp_path / "lex.db"
    _make_db(db, [{"modern_name": "Edlingham", "region": "Northumberland"}])
    jsonl = tmp_path / "mentions.jsonl"
    _write_jsonl(
        jsonl,
        [
            {
                "source_id": "src1",
                "form": "Edlingham",
                "date_year": 1086,
                "region_hint": None,
                "context": "x",
            }
        ],
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "ingest-toponym-mentions",
            "--jsonl",
            str(jsonl),
            "--db",
            str(db),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "inserted=1" in result.output
    assert "dry-run" in result.output
    # No actual rows in DB.
    conn = sqlite3.connect(db)
    count = conn.execute("SELECT COUNT(*) FROM toponym_attestation").fetchone()[0]
    conn.close()
    assert count == 0


def test_cli_apply_writes_and_idempotent_reapply(tmp_path):
    db = tmp_path / "lex.db"
    _make_db(db, [{"modern_name": "Edlingham", "region": "Northumberland"}])
    jsonl = tmp_path / "mentions.jsonl"
    _write_jsonl(
        jsonl,
        [
            {
                "source_id": "src1",
                "form": "Edlingham",
                "date_year": 1086,
                "region_hint": None,
                "context": "x",
            }
        ],
    )
    runner = CliRunner()
    # First apply.
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "ingest-toponym-mentions",
            "--jsonl",
            str(jsonl),
            "--db",
            str(db),
            "--apply",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "APPLIED" in result.output

    # Row landed.
    conn = sqlite3.connect(db)
    count = conn.execute("SELECT COUNT(*) FROM toponym_attestation").fetchone()[0]
    conn.close()
    assert count == 1

    # Re-apply: idempotent.
    result2 = runner.invoke(
        cli_root,
        [
            "lexicon",
            "ingest-toponym-mentions",
            "--jsonl",
            str(jsonl),
            "--db",
            str(db),
            "--apply",
        ],
    )
    assert result2.exit_code == 0, result2.output
    assert "inserted=0" in result2.output
    assert "dup=1" in result2.output


def test_cli_writes_candidates_jsonl_for_unresolved(tmp_path):
    db = tmp_path / "lex.db"
    _make_db(db, [])  # empty DB — everything is unresolved
    jsonl = tmp_path / "mentions.jsonl"
    _write_jsonl(
        jsonl,
        [
            {
                "source_id": "src1",
                "form": "UnknownPlace",
                "date_year": 1086,
                "region_hint": None,
                "context": "ctx",
            }
        ],
    )
    candidates = tmp_path / "candidates.jsonl"
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "ingest-toponym-mentions",
            "--jsonl",
            str(jsonl),
            "--db",
            str(db),
            "--candidates-out",
            str(candidates),
        ],
    )
    assert result.exit_code == 0, result.output
    assert candidates.exists()
    rec = json.loads(candidates.read_text().strip())
    assert rec["form"] == "UnknownPlace"
    assert rec["context"] == "ctx"


def test_cli_refuses_to_overwrite_candidates_without_force(tmp_path):
    db = tmp_path / "lex.db"
    _make_db(db, [])
    jsonl = tmp_path / "mentions.jsonl"
    _write_jsonl(jsonl, [{"source_id": "src1", "form": "X"}])
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text("PREVIOUS\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "ingest-toponym-mentions",
            "--jsonl",
            str(jsonl),
            "--db",
            str(db),
            "--candidates-out",
            str(candidates),
        ],
    )
    assert result.exit_code != 0
    assert "already exists" in result.output
    # File unchanged.
    assert candidates.read_text() == "PREVIOUS\n"


def test_cli_source_override_supersedes_jsonl_source_id(tmp_path):
    db = tmp_path / "lex.db"
    _make_db(db, [{"modern_name": "Edlingham", "region": "Northumberland"}])
    jsonl = tmp_path / "mentions.jsonl"
    _write_jsonl(
        jsonl,
        [
            {
                "source_id": "WRONG-source-in-jsonl",
                "form": "Edlingham",
                "date_year": 1086,
                "region_hint": None,
                "context": "x",
            }
        ],
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "ingest-toponym-mentions",
            "--jsonl",
            str(jsonl),
            "--db",
            str(db),
            "--source",
            "OVERRIDE-source",
            "--apply",
        ],
    )
    assert result.exit_code == 0, result.output
    # Verify the source_doc on the attestation row is the override.
    conn = sqlite3.connect(db)
    row = conn.execute(
        "SELECT source_doc FROM toponym_attestation WHERE form='Edlingham'"
    ).fetchone()
    conn.close()
    assert row[0] == "OVERRIDE-source"


def test_cli_invalid_json_line_errors_clearly(tmp_path):
    db = tmp_path / "lex.db"
    _make_db(db, [])
    jsonl = tmp_path / "mentions.jsonl"
    jsonl.write_text(
        '{"source_id":"src1","form":"X"}\nNOT VALID JSON HERE\n',
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "ingest-toponym-mentions",
            "--jsonl",
            str(jsonl),
            "--db",
            str(db),
        ],
    )
    assert result.exit_code != 0
    assert "invalid JSON" in result.output


def test_cli_missing_source_id_errors_with_line_number(tmp_path):
    db = tmp_path / "lex.db"
    _make_db(db, [])
    jsonl = tmp_path / "mentions.jsonl"
    # First row has source_id; second is missing it AND there's no
    # --source override.
    jsonl.write_text(
        '{"source_id":"src1","form":"X"}\n{"form":"Y"}\n',
        encoding="utf-8",
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        [
            "lexicon",
            "ingest-toponym-mentions",
            "--jsonl",
            str(jsonl),
            "--db",
            str(db),
        ],
    )
    assert result.exit_code != 0
    assert "missing source_id" in result.output
    assert ":2:" in result.output  # line number cited
