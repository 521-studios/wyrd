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
    """A near-spelling-variant surfaces as the top match. Asserts a
    realistic ratio range (NOT exact match — would slip a regression
    that returned 1.0 for everything) AND that a distractor toponym
    is ranked lower than the target."""
    conn = _make_conn()
    _seed_toponyms(
        conn,
        [
            {"modern_name": "Edlingham", "region": "Northumberland"},
            # Distractor: shares a couple letters but is clearly not
            # the target. Must rank LOWER than Edlingham.
            {"modern_name": "Berwick", "region": "Northumberland"},
        ],
    )
    toponyms = _toponym_rows(conn)
    suggestions = fuzzy_match_toponyms("Eadlingham", toponyms)
    assert len(suggestions) >= 1
    assert suggestions[0].modern_name == "Edlingham"
    # Score reflects high but NOT-exact similarity — exact-match
    # ratio would be 1.0, one-letter difference here is ~0.94. The
    # range pins both: a regression returning 1.0 for everything
    # would fail the upper bound; a regression returning 0 would
    # fail the lower bound.
    assert 0.8 <= suggestions[0].fuzzy_score < 1.0
    # Distractor (if it made it past the threshold) ranks LOWER.
    for sugg in suggestions[1:]:
        assert sugg.modern_name != "Edlingham"
        assert sugg.fuzzy_score < suggestions[0].fuzzy_score


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
    threshold, the operator's eye doesn't get drowned. Verifies
    (a) length is 3, (b) scores are sorted descending (so the
    operator sees the strongest match first), and (c) the exact
    match is included (a regression returning the first 3 in DB-
    insert order would fail the descending-score assertion)."""
    conn = _make_conn()
    _seed_toponyms(
        conn,
        [
            # Exact match — must appear and rank first.
            {"modern_name": "Edlingham"},
            # Decreasing similarity to "Edlingham".
            {"modern_name": "Edlinghame"},
            {"modern_name": "Edlinghan"},
            {"modern_name": "Edlinghum"},
            {"modern_name": "Edlinghom"},
        ],
    )
    toponyms = _toponym_rows(conn)
    suggestions = fuzzy_match_toponyms("Edlingham", toponyms)
    assert len(suggestions) == 3
    # Descending fuzzy score — top-3 BY ratio, not by DB-insert order.
    scores = [s.fuzzy_score for s in suggestions]
    assert scores == sorted(scores, reverse=True)
    # Exact match is the top result.
    assert suggestions[0].modern_name == "Edlingham"
    assert suggestions[0].fuzzy_score == 1.0


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


def test_prepare_candidate_coerces_numeric_string_date_year():
    """Prepare uses the same _coerce_date_year as commit, so a
    stringified-int upstream gets preserved as an int in the triage
    output. Previously prepare dropped non-int years silently —
    asymmetric with commit-layer coercion — and lost data."""
    conn = _make_conn()
    toponyms = _toponym_rows(conn)
    raw = {"source_id": "s", "form": "X", "date_year": "1086"}
    prepared = prepare_candidate(raw, toponyms)
    assert prepared.date_year == 1086


def test_prepare_candidate_coerces_integer_valued_float_date_year():
    """``1086.0`` from a sloppy JSON encoder lands as int 1086 in
    the triage output (matches commit-layer semantics)."""
    conn = _make_conn()
    toponyms = _toponym_rows(conn)
    raw = {"source_id": "s", "form": "X", "date_year": 1086.0}
    prepared = prepare_candidate(raw, toponyms)
    assert prepared.date_year == 1086


def test_prepare_candidate_rejects_unparseable_date_year():
    """Values that can't be coerced to a clean int land as None —
    same as commit layer. Includes Unicode digit-but-not-decimal strings
    (superscripts) that ``isdigit()`` admits but ``int()`` rejects: these must
    drop to None, NOT raise ValueError (the isdigit→isdecimal fix)."""
    conn = _make_conn()
    toponyms = _toponym_rows(conn)
    for bad in ["ten eighty-six", "1066 AD", 1086.5, True, False, "²", "1086²", "⁵"]:
        raw = {"source_id": "s", "form": "X", "date_year": bad}
        assert prepare_candidate(raw, toponyms).date_year is None


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
    """Anchor BOTH runs' counts to lock the idempotency contract.
    A regression where the FIRST run did nothing and the SECOND
    counted a fresh insert would still leave count==1 — the
    explicit first-run.mapped==1 + final count==1 chain catches it."""
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
    report1 = commit_triage_decisions(conn, rows, apply=True)
    # First run: counted, row landed.
    assert report1.mapped == 1
    assert report1.errors == 0
    assert conn.execute("SELECT COUNT(*) FROM toponym_attestation").fetchone()[0] == 1
    # Re-apply: counter still bumps (it's `mapped` regardless of
    # whether INSERT OR IGNORE writes), but the DB row count stays
    # at 1 because the UNIQUE index dedups.
    report2 = commit_triage_decisions(conn, rows, apply=True)
    assert report2.mapped == 1
    assert conn.execute("SELECT COUNT(*) FROM toponym_attestation").fetchone()[0] == 1


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


def test_commit_map_bool_toponym_id_is_error():
    """A boolean toponym_id is rejected as invalid, NOT silently treated
    as id 1. `isinstance(True, int)` is True and `True == 1` in Python, so
    without the explicit bool guard `toponym_id: true` would map onto
    whatever toponym has id 1. Pins that guard (wyrd-8uvi: it lives in the
    extracted _commit_map_row)."""
    conn = _make_conn()
    # Seed so id 1 exists — a regressed guard would map onto it.
    _seed_toponyms(conn, [{"modern_name": "Edlingham"}])
    rows = [
        {
            "source_id": "mawer_1920",
            "form": "Eadlingham",
            "action": "map",
            "toponym_id": True,
        }
    ]
    report = commit_triage_decisions(conn, rows, apply=True)
    assert report.mapped == 0
    assert report.errors == 1
    idx, msg = report.error_records[0]
    assert idx == 0
    assert "missing/invalid" in msg
    # No attestation written (would have landed on toponym 1 if the guard regressed).
    count = conn.execute("SELECT COUNT(*) FROM toponym_attestation").fetchone()[0]
    assert count == 0


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


def test_commit_create_folds_country_alias():
    """wyrd-fxxf: an operator-supplied country alias is canonicalized on create."""
    conn = _make_conn()
    rows = [
        {
            "source_id": "s",
            "form": "Baile",
            "date_year": 1200,
            "action": "create",
            "create_modern_name": "Somewhere",
            "create_country": "Republic of Ireland",
            "create_region": None,
        }
    ]
    report = commit_triage_decisions(conn, rows, apply=True)
    assert report.created == 1 and report.errors == 0
    assert conn.execute("SELECT country FROM toponym").fetchone()["country"] == "Ireland"


def test_commit_create_rejects_unknown_region_as_row_error():
    """wyrd-fxxf: a bad operator-supplied England region rejects THAT row
    (record_error) without aborting the commit or inserting a toponym."""
    conn = _make_conn()
    rows = [
        {
            "source_id": "s",
            "form": "X",
            "date_year": 1200,
            "action": "create",
            "create_modern_name": "Nowhere",
            "create_country": "England",
            "create_region": "Borsetshire",  # not a canonical England region
        }
    ]
    report = commit_triage_decisions(conn, rows, apply=True)
    assert report.created == 0 and report.errors == 1
    assert conn.execute("SELECT count(*) c FROM toponym").fetchone()["c"] == 0


def _create_row(name, region, country="England"):
    return {
        "source_id": "s",
        "form": name,
        "date_year": 1200,
        "action": "create",
        "create_modern_name": name,
        "create_country": country,
        "create_region": region,
    }


def test_commit_bad_region_row_does_not_block_good_rows():
    """wyrd-fxxf: a bad-region row is rejected per-row; the good rows around it in
    the same batch still commit (record_error, not a batch abort)."""
    conn = _make_conn()
    rows = [
        _create_row("Goodone", "Kent"),
        _create_row("Badone", "Borsetshire"),  # unknown region → row error
        _create_row("Goodtwo", "Surrey"),
    ]
    report = commit_triage_decisions(conn, rows, apply=True)
    assert report.created == 2 and report.errors == 1
    names = {r["modern_name"] for r in conn.execute("SELECT modern_name FROM toponym")}
    assert names == {"Goodone", "Goodtwo"}


def test_commit_create_country_fold_enables_collision_demote():
    """wyrd-fxxf: folding create_country BEFORE collision-detect means a
    'Republic of Ireland' CREATE collides with the existing canonical 'Ireland'
    toponym and demotes to MAP — instead of inserting a duplicate under the
    raw alias."""
    conn = _make_conn()
    _seed_toponyms(conn, [{"modern_name": "Dublin", "country": "Ireland", "region": None}])
    rows = [_create_row("Dublin", None, country="Republic of Ireland")]
    report = commit_triage_decisions(conn, rows, apply=True)
    assert report.created == 0 and report.mapped == 1  # demoted, no duplicate
    assert conn.execute("SELECT count(*) c FROM toponym").fetchone()["c"] == 1


def test_commit_create_derives_country_from_region():
    """wyrd-61p9: a CREATE with a region but no country derives the canonical
    country (England for Suffolk) rather than inserting a NULL country — uniform
    with the parser ingest's _upsert_toponym."""
    conn = _make_conn()
    rows = [_create_row("Ixworth", "Suffolk", country=None)]
    report = commit_triage_decisions(conn, rows, apply=True)
    assert report.created == 1
    row = conn.execute("SELECT country, region FROM toponym WHERE modern_name='Ixworth'").fetchone()
    assert (row["country"], row["region"]) == ("England", "Suffolk")


def test_commit_create_canonicalizes_country_before_region(monkeypatch):
    """wyrd-61p9: country-first ordering in the review guard too — a (hypothetical)
    alias that folds to England makes a bad England region scoped + rejected.
    Pins the order for this guard (region-first would have passed it through)."""
    import wyrd.generators.kenning.lexicon.controlled_vocab as cv

    monkeypatch.setitem(cv.COUNTRY_ALIASES, "Albion", "England")
    conn = _make_conn()
    rows = [_create_row("Nowhere", "Borsetshire", country="Albion")]
    report = commit_triage_decisions(conn, rows, apply=True)
    assert report.created == 0 and report.errors == 1
    assert conn.execute("SELECT count(*) c FROM toponym").fetchone()["c"] == 0


def test_commit_create_unique_collision_demotes_to_map():
    """If an existing toponym already has (modern_name, country,
    region), CREATE converts to MAP — prevents accidental duplicate
    toponym rows when the operator didn't notice the fuzzy suggestion.

    The demotion is RECORDED in ``demoted_records`` so the operator
    sees which CREATE rows landed as MAPs — silent demotion was a
    UX gap otherwise (operator with 30 CREATEs can't tell which 7
    collided). A distractor toponym is seeded so "mapped to the right
    one" is non-trivially true (would have been ambiguous with only
    one row in the table)."""
    conn = _make_conn()
    name_to_id = _seed_toponyms(
        conn,
        [
            # Target — what the CREATE should collide against.
            {"modern_name": "Edlingham", "country": "England", "region": "Northumberland"},
            # Distractor — different name/region; collision must NOT
            # find this one. Without it, mapped_to could be either id
            # and the test would still pass.
            {"modern_name": "Berwick", "country": "England", "region": "Northumberland"},
        ],
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
    # Demoted record captured for operator visibility.
    assert report.demoted_count == 1
    assert len(report.demoted_records) == 1
    row_idx, demoted_tid, name = report.demoted_records[0]
    assert row_idx == 0
    assert demoted_tid == name_to_id["Edlingham"]
    assert name == "Edlingham"
    # Still only the two seeded toponyms; no new row inserted.
    count = conn.execute("SELECT COUNT(*) FROM toponym").fetchone()[0]
    assert count == 2
    # Attestation written specifically pointing at the EDLINGHAM
    # (target), not Berwick (distractor).
    ares = conn.execute("SELECT toponym_id, form FROM toponym_attestation").fetchone()
    assert ares["toponym_id"] == name_to_id["Edlingham"]


def test_commit_create_collision_dry_run_records_demotion_without_writing():
    """Dry-run on a CREATE-collision row must STILL record the
    demotion (counter accuracy + demoted_records list), but write
    no attestation. Operator can preview demotions before --apply."""
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
    report = commit_triage_decisions(conn, rows, apply=False)
    assert report.created == 0
    assert report.mapped == 1
    assert len(report.demoted_records) == 1
    assert report.demoted_records[0][1] == name_to_id["Edlingham"]
    # No DB writes in dry-run.
    count = conn.execute("SELECT COUNT(*) FROM toponym_attestation").fetchone()[0]
    assert count == 0


def test_commit_create_dry_run_pure_create_does_not_write():
    """Dry-run on a pure-CREATE (no collision) row counts the
    create but writes neither toponym nor attestation."""
    conn = _make_conn()
    rows = [
        {
            "source_id": "x",
            "form": "Suttone",
            "action": "create",
            "create_modern_name": "Sutton",
        }
    ]
    report = commit_triage_decisions(conn, rows, apply=False)
    assert report.created == 1
    assert report.mapped == 0
    # No DB writes.
    assert conn.execute("SELECT COUNT(*) FROM toponym").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM toponym_attestation").fetchone()[0] == 0


def test_commit_action_map_with_create_fields_ignores_create_fields():
    """When the operator sets BOTH action=map AND create_modern_name
    (e.g. they typed in suggested fields but later changed action),
    the action field wins — create_modern_name is silently ignored
    and the MAP path executes. Locks the precedence contract."""
    conn = _make_conn()
    name_to_id = _seed_toponyms(conn, [{"modern_name": "Edlingham"}])
    rows = [
        {
            "source_id": "x",
            "form": "Eadlingham",
            "action": "map",
            "toponym_id": name_to_id["Edlingham"],
            # Lingering CREATE fields — must be ignored.
            "create_modern_name": "SomeNewName",
            "create_country": "England",
        }
    ]
    report = commit_triage_decisions(conn, rows, apply=True)
    assert report.mapped == 1
    assert report.created == 0
    # No new toponym row from the lingering CREATE fields.
    count = conn.execute("SELECT COUNT(*) FROM toponym").fetchone()[0]
    assert count == 1


def test_commit_error_records_cap_truncates_but_counter_accurate():
    """error_records list caps at _ERROR_RECORDS_CAP. The counter
    keeps counting beyond the cap so the operator sees the true
    total of bad rows."""
    from wyrd.generators.kenning import toponym_candidate_review as tcr

    conn = _make_conn()
    # 5 errors, cap=2 → 5 counted, only first 2 captured.
    cap = tcr._ERROR_RECORDS_CAP
    try:
        tcr._ERROR_RECORDS_CAP = 2
        rows = [{"action": "frobnicate", "source_id": "x", "form": "Y"} for _ in range(5)]
        report = commit_triage_decisions(conn, rows, apply=True)
        assert report.errors == 5
        assert len(report.error_records) == 2
    finally:
        tcr._ERROR_RECORDS_CAP = cap


def test_commit_empty_rows_list_zero_processed():
    """Empty input is a clean no-op: zero counters across the board."""
    conn = _make_conn()
    report = commit_triage_decisions(conn, [], apply=True)
    assert report.processed == 0
    assert report.mapped == 0
    assert report.created == 0
    assert report.skipped == 0
    assert report.deferred == 0
    assert report.errors == 0
    assert report.demoted_count == 0
    assert report.error_records == []
    assert report.demoted_records == []


def test_commit_demoted_records_cap_truncates_but_counter_accurate():
    """demoted_records caps at _DEMOTED_RECORDS_CAP; demoted_count
    counter stays accurate past the cap. Mirrors the error_records
    cap pattern (Gemini round-2 + combined-agent finding)."""
    from wyrd.generators.kenning import toponym_candidate_review as tcr

    conn = _make_conn()
    _seed_toponyms(
        conn,
        [{"modern_name": "Edlingham", "country": "England", "region": "Northumberland"}],
    )
    cap = tcr._DEMOTED_RECORDS_CAP
    try:
        tcr._DEMOTED_RECORDS_CAP = 2
        # 4 CREATEs that all collide → 4 demoted, cap=2.
        rows = [
            {
                "source_id": f"s{i}",
                "form": f"Eadlingham{i}",
                "action": "create",
                "create_modern_name": "Edlingham",
                "create_country": "England",
                "create_region": "Northumberland",
            }
            for i in range(4)
        ]
        report = commit_triage_decisions(conn, rows, apply=True)
        assert report.demoted_count == 4
        assert len(report.demoted_records) == 2
        # All 4 still counted as mapped (the operator-visible total).
        assert report.mapped == 4
    finally:
        tcr._DEMOTED_RECORDS_CAP = cap


def test_toponym_rows_pre_normalizes_name_and_region():
    """``_toponym_rows`` returns 6-tuples with pre-normalized name +
    region so the fuzzy-match hot loop doesn't re-normalize per
    candidate. Gemini round-2 perf finding."""
    conn = _make_conn()
    _seed_toponyms(
        conn,
        [{"modern_name": "Edlingham", "country": "England", "region": "Northumberland"}],
    )
    rows = _toponym_rows(conn)
    assert len(rows) == 1
    tid, modern_name, region, country, name_norm, region_norm = rows[0]
    # Raw values preserved.
    assert modern_name == "Edlingham"
    assert region == "Northumberland"
    assert country == "England"
    # Pre-normalized fields are present and lowercase.
    assert name_norm == "edlingham"
    assert region_norm == "northumberland"


def test_fuzzy_match_sort_does_not_TypeError_on_tied_scores():
    """When two suggestions have the same effective score (e.g. two
    same-name toponyms with no region hint to break the tie), the
    sort must NOT raise TypeError. FuzzyToponymSuggestion is a
    plain dataclass with no ordering, so a sort key of just the
    score would fall back to comparing the suggestion objects and
    crash. Tiebreaker on toponym_id keeps the sort deterministic.
    Gemini round-3 HIGH finding."""
    conn = _make_conn()
    # Two identical-name toponyms — same fuzzy ratio against the form.
    _seed_toponyms(
        conn,
        [
            {"modern_name": "Newton", "region": "Berkshire"},
            {"modern_name": "Newton", "region": "Northumberland"},
        ],
    )
    toponyms = _toponym_rows(conn)
    # No region hint — both will tie on effective score.
    suggestions = fuzzy_match_toponyms("Newton", toponyms)
    # Must not crash; deterministic order.
    assert len(suggestions) == 2


def test_fuzzy_match_length_bound_does_not_drop_marginal_matches():
    """The length-bound gate uses the theoretical max ratio:
    2 * min(L1, L2) / (L1 + L2). The naive (1 - min_ratio) length-
    delta would drop pairs that could legitimately ratio above the
    threshold. Gemini round-3 MEDIUM finding: with min_ratio=0.6,
    L1=10, L2=5, the theoretical max is 2*5/15 ≈ 0.667 (above
    threshold), but the naive delta gate would reject. Use a
    string pair that demonstrably can ratio above the threshold."""
    conn = _make_conn()
    _seed_toponyms(
        conn,
        [
            # "Edlinghamington" is 15 chars; "Edlingham" is 9 chars.
            # 2 * 9 / 24 = 0.75, max ratio for the pair. Real ratio
            # of "Edlingham" vs the prefix portion is well above 0.6.
            {"modern_name": "Edlinghamington", "region": "X"},
        ],
    )
    toponyms = _toponym_rows(conn)
    # form is "Edlingham" — the matching 9 chars are a prefix
    # of the toponym name; real ratio ≈ 0.75.
    suggestions = fuzzy_match_toponyms("Edlingham", toponyms)
    assert len(suggestions) == 1
    assert suggestions[0].fuzzy_score >= 0.6


def test_commit_non_str_field_does_not_crash():
    """If the operator's hand-edited JSONL has a non-string value
    where a string is expected (number, list, None), the code must
    NOT crash on `.strip()` — coerce to empty and let the
    downstream "missing field" error path handle it. Gemini round-3
    MEDIUM finding."""
    conn = _make_conn()
    _seed_toponyms(conn, [{"modern_name": "Edlingham"}])
    rows = [
        # Non-string action — coerces to "" and falls through to
        # "unknown action" error (NOT AttributeError on .strip()).
        {"source_id": "x", "form": "y", "action": 42},
        # Non-string form on map — coerces to "" → "missing form" error.
        {"source_id": "x", "form": [1, 2], "action": "map", "toponym_id": 1},
        # Non-string create_modern_name on create.
        {
            "source_id": "x",
            "form": "y",
            "action": "create",
            "create_modern_name": 99,
        },
    ]
    # Must not raise.
    report = commit_triage_decisions(conn, rows, apply=True)
    # All three rows recorded as errors (process didn't crash).
    assert report.errors == 3


def test_prepare_non_str_field_does_not_crash():
    """Same coercion applies to prepare_candidate — non-string
    form/source_id/context must not crash on .strip()."""
    conn = _make_conn()
    toponyms = _toponym_rows(conn)
    raw = {
        "source_id": 99,  # int, not str
        "form": [1, 2],  # list
        "region_hint": 42,  # int
        "context": None,  # None
    }
    # Must not raise.
    prepared = prepare_candidate(raw, toponyms)
    # All non-string inputs coerced to empty/None.
    assert prepared.source_id == ""
    assert prepared.form == ""
    assert prepared.region_hint is None
    assert prepared.context == ""


def test_toponym_rows_pre_normalizes_null_region_as_none():
    """A toponym with no region must have name_norm AND
    region_norm=None (not empty string) so the fuzzy-match region
    boost correctly skips it."""
    conn = _make_conn()
    _seed_toponyms(conn, [{"modern_name": "Atlantis", "region": None}])
    rows = _toponym_rows(conn)
    _, _, region, _, name_norm, region_norm = rows[0]
    assert region is None
    assert region_norm is None
    assert name_norm == "atlantis"


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


def test_commit_action_case_insensitive_across_all_paths():
    """All four actions (map/create/skip/defer) must be case-
    insensitive — operator caps shouldn't matter. Earlier test only
    covered MAP."""
    # MAP with mixed case.
    conn = _make_conn()
    name_to_id = _seed_toponyms(conn, [{"modern_name": "Edlingham"}])
    report = commit_triage_decisions(
        conn,
        [
            {
                "source_id": "x",
                "form": "Edlingham",
                "action": "MaP",
                "toponym_id": name_to_id["Edlingham"],
            }
        ],
        apply=True,
    )
    assert report.mapped == 1 and report.errors == 0

    # CREATE with caps.
    conn = _make_conn()
    report = commit_triage_decisions(
        conn,
        [
            {
                "source_id": "x",
                "form": "X",
                "action": "CREATE",
                "create_modern_name": "Foo",
            }
        ],
        apply=True,
    )
    assert report.created == 1 and report.errors == 0

    # SKIP with caps.
    conn = _make_conn()
    report = commit_triage_decisions(
        conn, [{"source_id": "x", "form": "Y", "action": "Skip"}], apply=True
    )
    assert report.skipped == 1 and report.errors == 0

    # DEFER with caps.
    conn = _make_conn()
    report = commit_triage_decisions(
        conn, [{"source_id": "x", "form": "Y", "action": "DEFER"}], apply=True
    )
    assert report.deferred == 1 and report.errors == 0


def test_commit_action_trailing_whitespace_stripped():
    """`"action": "map "` (paste from a spreadsheet, trailing space)
    must work like "map". The strip is part of the operator-input
    tolerance."""
    conn = _make_conn()
    name_to_id = _seed_toponyms(conn, [{"modern_name": "Edlingham"}])
    rows = [
        {
            "source_id": "x",
            "form": "Edlingham",
            "action": "  map  ",  # leading + trailing whitespace
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


def test_cli_commit_warns_when_mostly_deferred(tmp_path):
    """A triage file that's still mostly action=defer (>=80% with
    >=5 processed) emits a warning — catches running commit on an
    unedited file. This branch lives in the ``_echo_commit_report``
    helper (C901 extraction, wyrd-8uvi)."""
    db = tmp_path / "lex.db"
    _make_db(db, [])
    triage = tmp_path / "triage.jsonl"
    _write_jsonl(
        triage,
        [{"source_id": "x", "form": f"F{i}", "action": "defer"} for i in range(5)],
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
    assert "warning:" in result.output
    assert "action=defer" in result.output
    assert "deferred=5" in result.output


def test_cli_commit_verbose_surfaces_create_to_map_demotion(tmp_path):
    """With --verbose, a CREATE row that collides with an existing
    toponym surfaces a CREATE→MAP demotion line per row. This branch
    lives in the ``_echo_commit_report`` helper (C901 extraction,
    wyrd-8uvi)."""
    db = tmp_path / "lex.db"
    _make_db(
        db,
        [{"modern_name": "Edlingham", "country": "England", "region": "Northumberland"}],
    )
    triage = tmp_path / "triage.jsonl"
    _write_jsonl(
        triage,
        [
            {
                "source_id": "mawer_1920",
                "form": "Eadlingham",
                "action": "create",
                "create_modern_name": "Edlingham",
                "create_country": "England",
                "create_region": "Northumberland",
            }
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
            "--verbose",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "CREATE→MAP" in result.output
    assert "Edlingham" in result.output
    assert "demoted=1" in result.output


def test_cli_commit_no_defer_warning_below_count_threshold(tmp_path):
    """The defer-warning is gated on processed >= 5 so a tiny
    intentional-defer invocation isn't nagged. With 3 defers
    (< 5 processed), no warning fires even at 100% defer ratio.
    Pins the count guard the C901 extraction now owns (wyrd-8uvi)."""
    db = tmp_path / "lex.db"
    _make_db(db, [])
    triage = tmp_path / "triage.jsonl"
    _write_jsonl(
        triage,
        [{"source_id": "x", "form": f"F{i}", "action": "defer"} for i in range(3)],
    )
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        ["lexicon", "commit-toponym-candidates", "--jsonl", str(triage), "--db", str(db)],
    )
    assert result.exit_code == 0, result.output
    assert "warning:" not in result.output
    assert "deferred=3" in result.output


def test_cli_commit_no_defer_warning_below_ratio_threshold(tmp_path):
    """The defer-warning is gated on a >= 0.8 defer ratio. With 5
    processed rows but only 3 deferred (60%), no warning fires.
    Pins the ratio guard the C901 extraction now owns (wyrd-8uvi)."""
    db = tmp_path / "lex.db"
    _make_db(db, [])
    triage = tmp_path / "triage.jsonl"
    rows = [{"source_id": "x", "form": f"D{i}", "action": "defer"} for i in range(3)]
    rows += [{"source_id": "x", "form": f"S{i}", "action": "skip"} for i in range(2)]
    _write_jsonl(triage, rows)
    runner = CliRunner()
    result = runner.invoke(
        cli_root,
        ["lexicon", "commit-toponym-candidates", "--jsonl", str(triage), "--db", str(db)],
    )
    assert result.exit_code == 0, result.output
    assert "warning:" not in result.output
    assert "deferred=3" in result.output
    assert "skipped=2" in result.output


def test_cli_commit_demotion_summary_without_verbose_omits_per_row_line(tmp_path):
    """Without --verbose, a CREATE→MAP demotion still counts in the
    summary (demoted=1) but emits no per-row CREATE→MAP line — the
    verbose-gating symmetric to the errors-hint-without-verbose case.
    Pins the verbose gate the C901 extraction now owns (wyrd-8uvi)."""
    db = tmp_path / "lex.db"
    _make_db(
        db,
        [{"modern_name": "Edlingham", "country": "England", "region": "Northumberland"}],
    )
    triage = tmp_path / "triage.jsonl"
    _write_jsonl(
        triage,
        [
            {
                "source_id": "mawer_1920",
                "form": "Eadlingham",
                "action": "create",
                "create_modern_name": "Edlingham",
                "create_country": "England",
                "create_region": "Northumberland",
            }
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
    assert "demoted=1" in result.output
    assert "CREATE→MAP" not in result.output
