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


def test_resolve_mention_region_hint_longer_than_region_matches():
    """LLM emits long hint with surrounding context: ``"co. Northumberland,
    England"``. Toponym region is the short canonical ``"northumberland"``.
    Tokenization handles the punctuation; the padded substring check
    finds the region within the hint."""
    indexes = ResolverIndexes(
        form_to_ids={"newton": {1, 2}},
        toponym_regions={1: "northumberland", 2: "berkshire"},
    )
    assert resolve_mention("Newton", "co. Northumberland, England", indexes) == 1


def test_resolve_mention_region_hint_shorter_than_region_matches():
    """The OTHER direction of substring containment: LLM emits the short
    canonical ``"Yorkshire"``, the toponym is in ``"West Yorkshire"``.
    The region tokens contain the hint tokens. (The previous test only
    exercised one direction.)"""
    indexes = ResolverIndexes(
        form_to_ids={"newton": {1, 2}},
        toponym_regions={1: "west yorkshire", 2: "berkshire"},
    )
    # Hint "Yorkshire" appears as a token inside the region "west yorkshire".
    assert resolve_mention("Newton", "Yorkshire", indexes) == 1


def test_resolve_mention_region_substring_does_not_overmatch_distinct_counties():
    """Word-boundary protection: ``"Sussex"`` must NOT match ``"Essex"``
    even though "essex" is a substring of "sussex". Same for
    ``"Ham"`` not matching ``"Hampshire"``. silent-failure-hunter +
    Gemini both flagged this as the load-bearing fix."""
    indexes = ResolverIndexes(
        form_to_ids={"newton": {1, 2}},
        toponym_regions={1: "essex", 2: "berkshire"},
    )
    # Naive substring matching would have wrongly picked id 1 here.
    assert resolve_mention("Newton", "Sussex", indexes) is None

    indexes2 = ResolverIndexes(
        form_to_ids={"newton": {1, 2}},
        toponym_regions={1: "hampshire", 2: "berkshire"},
    )
    # "Ham" must not match "Hampshire".
    assert resolve_mention("Newton", "Ham", indexes2) is None


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


def test_cli_non_dict_jsonl_row_errors(tmp_path):
    """A JSONL row that's valid JSON but not an object (a number, a
    list, etc.) must error with a clear message — without this guard
    `row.get(...)` would AttributeError mid-stream and the operator
    couldn't tell whether the file was corrupt or the code crashed."""
    db = tmp_path / "lex.db"
    _make_db(db, [])
    jsonl = tmp_path / "mentions.jsonl"
    jsonl.write_text('{"source_id":"src1","form":"X"}\n[1, 2, 3]\n', encoding="utf-8")
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
    assert "not a JSON object" in result.output
    assert ":2:" in result.output


def test_cli_verbose_shows_unresolved_samples(tmp_path):
    """--verbose surfaces a few unresolved-mention samples per source
    so the operator can sense-check the failure mode without opening
    the candidates file. Without this flag, the per-source line just
    shows counters."""
    db = tmp_path / "lex.db"
    _make_db(db, [])  # empty toponyms → everything unresolved
    jsonl = tmp_path / "mentions.jsonl"
    _write_jsonl(
        jsonl,
        [
            {"source_id": "src1", "form": "Yorkville", "region_hint": "Toronto"},
        ],
    )
    runner = CliRunner()
    # Without --verbose: counters but no sample line.
    result_quiet = runner.invoke(
        cli_root,
        ["lexicon", "ingest-toponym-mentions", "--jsonl", str(jsonl), "--db", str(db)],
    )
    assert "unresolved:" not in result_quiet.output

    # With --verbose: sample line surfaces.
    result_verbose = runner.invoke(
        cli_root,
        [
            "lexicon",
            "ingest-toponym-mentions",
            "--jsonl",
            str(jsonl),
            "--db",
            str(db),
            "--verbose",
        ],
    )
    assert "unresolved:" in result_verbose.output
    assert "Yorkville" in result_verbose.output


def test_cli_walks_multiple_sources_in_one_jsonl(tmp_path):
    """A single JSONL with rows from two different source_ids must
    produce two separate per-source progress lines and write attestations
    with the correct source_doc for each row."""
    db = tmp_path / "lex.db"
    _make_db(db, [{"modern_name": "Edlingham", "region": "Northumberland"}])
    jsonl = tmp_path / "mixed.jsonl"
    _write_jsonl(
        jsonl,
        [
            {"source_id": "src_a", "form": "Edlingham", "date_year": 1086},
            {"source_id": "src_b", "form": "Edlingham", "date_year": 1086},
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
            "--apply",
        ],
    )
    assert result.exit_code == 0, result.output
    # Two separate source-progress lines.
    assert "src_a" in result.output
    assert "src_b" in result.output
    # Two rows in DB, distinguished by source_doc.
    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT source_doc FROM toponym_attestation ORDER BY source_doc").fetchall()
    conn.close()
    assert [r[0] for r in rows] == ["src_a", "src_b"]


def test_cli_summary_includes_unresolved_count_without_candidates_out(tmp_path):
    """Even without --candidates-out, the TOTAL summary line must
    report unresolved=N so the operator can see they have outstanding
    work to triage. Negative control for the --candidates-out
    optional sink."""
    db = tmp_path / "lex.db"
    _make_db(db, [])
    jsonl = tmp_path / "mentions.jsonl"
    _write_jsonl(jsonl, [{"source_id": "src1", "form": "UnknownPlace"}])
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
    assert "unresolved=1" in result.output


def test_cli_candidates_atomic_write_leaves_no_tmp_file(tmp_path):
    """The atomic-rename write pattern must remove the .tmp file on
    successful completion (rename consumes it) — verify no .tmp
    artifact pollutes the operator's directory after a clean run."""
    db = tmp_path / "lex.db"
    _make_db(db, [])
    jsonl = tmp_path / "mentions.jsonl"
    _write_jsonl(jsonl, [{"source_id": "src1", "form": "UnknownPlace"}])
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
    # Tempfile must have been renamed away.
    tmp_path_candidates = candidates.with_suffix(candidates.suffix + ".tmp")
    assert not tmp_path_candidates.exists()


def test_ingest_mentions_first_run_anchors_inserted_count():
    """Idempotent-reapply test should bracket the first run's
    inserted count too — otherwise a regression that silently inserts
    zero on the first run would also pass the 'second run = dup'
    check. pr-test-analyzer round-1 finding."""
    conn = _make_conn()
    _seed_toponyms(conn, [{"modern_name": "Edlingham", "region": "Northumberland"}])
    indexes = build_resolver_indexes(conn)
    mentions = [{"form": "Edlingham", "date_year": 1086, "context": "x"}]
    report1 = ingest_mentions(conn, "source-1", mentions, indexes, apply=True)
    # First run anchored.
    assert report1.inserted == 1
    assert report1.already_present == 0
    # Row count anchored before re-apply.
    count = conn.execute("SELECT COUNT(*) FROM toponym_attestation").fetchone()[0]
    assert count == 1
    report2 = ingest_mentions(conn, "source-1", mentions, indexes, apply=True)
    assert report2.inserted == 0
    assert report2.already_present == 1


def test_ingest_mentions_non_int_year_lands_with_null_date():
    """Verify the single landed row when the LLM emits non-int years —
    the row's date_year column must literally be NULL, not 0 or a
    coerced int. Earlier test only checked counts."""
    conn = _make_conn()
    _seed_toponyms(conn, [{"modern_name": "Edlingham", "region": "Northumberland"}])
    indexes = build_resolver_indexes(conn)
    mentions = [
        {"form": "Edlingham", "date_year": "not a year"},
        {"form": "Edlingham", "date_year": True},
    ]
    ingest_mentions(conn, "source-1", mentions, indexes, apply=True)
    row = conn.execute(
        "SELECT date_year FROM toponym_attestation WHERE form='Edlingham'"
    ).fetchone()
    assert row["date_year"] is None


def test_ingest_mentions_integer_valued_float_year_kept():
    """`1086.0` from a sloppy JSON encoder should land as 1086, not be
    silently dropped to NULL. Fractional floats (1086.5) and NaN/inf
    still go to NULL."""
    conn = _make_conn()
    _seed_toponyms(conn, [{"modern_name": "Edlingham", "region": "Northumberland"}])
    indexes = build_resolver_indexes(conn)
    mentions = [
        {"form": "Edlingham", "date_year": 1086.0, "context": "integer-float"},
    ]
    ingest_mentions(conn, "source-1", mentions, indexes, apply=True)
    row = conn.execute(
        "SELECT date_year FROM toponym_attestation WHERE form='Edlingham'"
    ).fetchone()
    assert row["date_year"] == 1086


def test_ingest_mentions_skips_non_dict_mention():
    """A non-dict mention (a stray int/list slipped into the list) is
    counted in mentions_processed + unresolved but does NOT crash on
    `.get()`. silent-failure-hunter round-1 finding."""
    conn = _make_conn()
    _seed_toponyms(conn, [{"modern_name": "Edlingham", "region": "Northumberland"}])
    indexes = build_resolver_indexes(conn)
    mentions = [
        {"form": "Edlingham", "date_year": 1086},
        "not a dict",  # type: ignore[list-item]
        42,  # type: ignore[list-item]
        [1, 2],  # type: ignore[list-item]
    ]
    report = ingest_mentions(conn, "source-1", mentions, indexes, apply=True)
    assert report.mentions_processed == 4
    assert report.resolved == 1
    assert report.unresolved == 3


def test_ingest_mentions_non_str_form_treated_as_empty():
    """Mention with form=int/list/None — coerce to empty, treat as
    unresolved, don't crash. silent-failure-hunter round-1 finding."""
    conn = _make_conn()
    _seed_toponyms(conn, [{"modern_name": "Edlingham", "region": "Northumberland"}])
    indexes = build_resolver_indexes(conn)
    mentions = [
        {"form": 123, "date_year": 1086},
        {"form": ["Edlingham"], "date_year": 1086},
        {"form": None, "date_year": 1086},
    ]
    report = ingest_mentions(conn, "source-1", mentions, indexes, apply=True)
    assert report.resolved == 0
    assert report.unresolved == 3
    assert report.inserted == 0


def test_ingest_mentions_uses_bulk_existing_load_not_n_plus_one():
    """Performance contract: each ``ingest_mentions`` call should
    execute ONE preflight SELECT regardless of how many mentions it
    processes. A regression to per-mention SELECT would balloon
    queries at full scope. Gemini round-1 N+1 finding.

    Uses sqlite3's set_trace_callback to count SQL statements
    (Connection.execute is read-only, can't monkeypatch directly)."""
    conn = _make_conn()
    _seed_toponyms(conn, [{"modern_name": "Edlingham", "region": "Northumberland"}])
    indexes = build_resolver_indexes(conn)
    # 100 identical mentions — under N+1, would issue 100 SELECTs.
    mentions = [{"form": "Edlingham", "date_year": 1086} for _ in range(100)]

    statements: list[str] = []
    conn.set_trace_callback(statements.append)
    try:
        ingest_mentions(conn, "source-1", mentions, indexes, apply=True)
    finally:
        conn.set_trace_callback(None)

    select_count = sum(1 for s in statements if s.lstrip().upper().startswith("SELECT"))
    # Exactly one preflight SELECT (the bulk load) for this function.
    # build_resolver_indexes ran BEFORE the trace callback was set,
    # so its SELECTs don't count here.
    assert select_count == 1, (
        f"expected 1 SELECT, got {select_count}; statements: {statements}"
    )
