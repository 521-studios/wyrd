"""Tests for the wyrd-ami fantasy-name etymology pipeline."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from wyrd.generators.kenning import fantasy_pipeline as fp
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema, migrate_schema


@pytest.fixture
def fresh_db(tmp_path: Path) -> Path:
    """Lexicon DB with the schema created and migrated."""
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        migrate_schema(db)
        db.commit()
    return db_path


def _seed_etymon(
    db: LexiconDB,
    *,
    canonical_form: str,
    language: str,
    gloss: str | None = None,
) -> int:
    cur = db.conn.execute(
        "INSERT INTO etymon (canonical_form, language) VALUES (?, ?)",
        (canonical_form, language),
    )
    eid = cur.lastrowid
    if gloss:
        db.conn.execute(
            "INSERT INTO etymon_gloss (etymon_id, gloss) VALUES (?, ?)",
            (eid, gloss),
        )
    return eid


def _seed_descent(
    db: LexiconDB,
    *,
    parent_id: int,
    child_id: int,
    edge_type: str = "inheritance",
    source_id: str = "test",
) -> None:
    db.conn.execute(
        "INSERT OR IGNORE INTO source (id, title, author) VALUES (?, 'Test', 'Test')",
        (source_id,),
    )
    db.conn.execute(
        """INSERT INTO etymon_descent (parent_id, child_id, edge_type, source_id, confidence)
           VALUES (?, ?, ?, ?, 'high')""",
        (parent_id, child_id, edge_type, source_id),
    )


# ---------------------------------------------------------------------
# descent_walking_lookup
# ---------------------------------------------------------------------


def test_descent_walking_lookup_finds_greek_ancestor_for_modern_form(fresh_db: Path) -> None:
    """The harpy case: input matches modern-english harpy; lookup walks
    one descent edge back to ancient-greek ἅρπυια and returns it."""
    with LexiconDB(fresh_db) as db:
        greek_id = _seed_etymon(
            db, canonical_form="ἅρπυια", language="ancient-greek", gloss="snatcher"
        )
        modern_id = _seed_etymon(db, canonical_form="harpy", language="modern-english")
        _seed_descent(db, parent_id=greek_id, child_id=modern_id)
        db.commit()

    matches = fp.descent_walking_lookup(fresh_db, "harpy")
    assert len(matches) == 1
    assert matches[0].canonical_form == "ἅρπυια"
    assert matches[0].language == "ancient-greek"
    assert matches[0].edge_type == "inheritance"


def test_descent_walking_lookup_returns_direct_hit_with_priority(fresh_db: Path) -> None:
    """When the input form itself is an approved-family etymon (e.g.
    'wyrm' as OE), it shows up as a direct hit BEFORE any ancestor
    surfaced via descent."""
    with LexiconDB(fresh_db) as db:
        oe_id = _seed_etymon(db, canonical_form="wyrm", language="old-english", gloss="serpent")
        modern_id = _seed_etymon(db, canonical_form="wyrm", language="modern-english")
        _seed_descent(db, parent_id=oe_id, child_id=modern_id)
        db.commit()

    matches = fp.descent_walking_lookup(fresh_db, "wyrm")
    # Direct OE hit comes first.
    assert matches[0].language == "old-english"
    assert matches[0].edge_type == "(direct)"


def test_descent_walking_lookup_no_match_returns_empty(fresh_db: Path) -> None:
    """No etymon matches the input form → empty list, no exception."""
    with LexiconDB(fresh_db) as db:
        # Seed something unrelated so the table isn't empty.
        _seed_etymon(db, canonical_form="bera", language="old-english")
        db.commit()

    assert fp.descent_walking_lookup(fresh_db, "gnoll") == []


def test_descent_walking_lookup_skips_unapproved_languages(fresh_db: Path) -> None:
    """Ancestors whose language isn't in the approved set are walked
    THROUGH but not returned — Proto-Finnic ancestors should not
    appear, but should still let us reach an OE ancestor beyond them."""
    with LexiconDB(fresh_db) as db:
        oe_id = _seed_etymon(db, canonical_form="root-form", language="old-english")
        finnic_id = _seed_etymon(db, canonical_form="*intermediate", language="urj-fin-pro")
        modern_id = _seed_etymon(db, canonical_form="testword", language="modern-english")
        _seed_descent(db, parent_id=finnic_id, child_id=modern_id)
        _seed_descent(db, parent_id=oe_id, child_id=finnic_id)
        db.commit()

    matches = fp.descent_walking_lookup(fresh_db, "testword")
    # Should reach OE through the Finnic intermediate.
    langs = {m.language for m in matches}
    assert "old-english" in langs
    assert "urj-fin-pro" not in langs


# ---------------------------------------------------------------------
# resolve
# ---------------------------------------------------------------------


def test_resolve_via_pre_filter_short_circuits_llm(fresh_db: Path) -> None:
    """When the pre-filter resolves cleanly, the LLM is NOT called."""
    with LexiconDB(fresh_db) as db:
        greek_id = _seed_etymon(db, canonical_form="ἅρπυια", language="ancient-greek")
        modern_id = _seed_etymon(db, canonical_form="harpy", language="modern-english")
        _seed_descent(db, parent_id=greek_id, child_id=modern_id)
        db.commit()

    def _llm_must_not_be_called(name, description):
        raise AssertionError("LLM should not be called when pre-filter resolves")

    res = fp.resolve(fresh_db, "harpy", "winged Greek monster", llm_caller=_llm_must_not_be_called)
    assert res.usable is True
    assert res.etymon_id == greek_id
    assert res.resolution_method == "descent_lookup"
    assert res.confidence == "high"
    assert res.bar_reason is None


def test_resolve_skip_llm_bars_pre_filter_misses(fresh_db: Path) -> None:
    """With skip_llm=True, pre-filter misses are barred without an LLM call."""
    res = fp.resolve(fresh_db, "gnoll", "hyena humanoid", skip_llm=True)
    assert res.usable is False
    assert res.bar_reason == fp.BAR_REASON_NO_ETYMOLOGY
    assert res.resolution_method == "descent_lookup"


def test_resolve_via_llm_when_pre_filter_misses(fresh_db: Path) -> None:
    """Pre-filter misses → LLM is called. A clean attested response
    produces a usable Resolution."""

    def _stub_llm(name, description):
        assert name == "Harpy"
        return {
            "attested_in": "ancient-greek",
            "historical_form": "ἅρπυια",
            "gloss": "snatcher",
            "citation": "LSJ",
            "confidence": "high",
            "bar_reason": None,
            "reasoning": "stub reasoning",
        }

    res = fp.resolve(fresh_db, "Harpy", "Greek monster", llm_caller=_stub_llm)
    assert res.usable is True
    assert res.resolution_method == "llm_full_research"
    assert res.citation == "LSJ"


def test_resolve_via_llm_low_confidence_is_barred(fresh_db: Path) -> None:
    """Per the user's conservative-default directive: low confidence
    LLM answers get barred as uncertain_attestation, even when the LLM
    names a real language."""

    def _stub_llm(name, description):
        return {
            "attested_in": "old-english",
            "historical_form": "*maybe-something",
            "gloss": None,
            "citation": None,
            "confidence": "low",
            "bar_reason": None,
            "reasoning": "I'm not sure",
        }

    res = fp.resolve(fresh_db, "Mystery", "unknown creature", llm_caller=_stub_llm)
    assert res.usable is False
    assert res.bar_reason == fp.BAR_REASON_UNCERTAIN


def test_resolve_via_llm_modern_coinage_is_barred(fresh_db: Path) -> None:
    """Dunsany's Gnoll: LLM correctly identifies as outside the family."""

    def _stub_llm(name, description):
        return {
            "attested_in": "outside_family",
            "historical_form": None,
            "gloss": None,
            "citation": None,
            "confidence": "high",
            "bar_reason": "modern_coinage",
            "reasoning": "Dunsany 1912",
        }

    res = fp.resolve(fresh_db, "Gnoll", "hyena humanoid", llm_caller=_stub_llm)
    assert res.usable is False
    assert res.bar_reason == fp.BAR_REASON_MODERN_COINAGE


def test_resolve_via_llm_failure_bars_as_uncertain(fresh_db: Path) -> None:
    """A network/parse failure during LLM call doesn't crash; rows get
    barred as uncertain so the input can be re-processed later."""

    def _stub_llm(name, description):
        raise RuntimeError("simulated network failure")

    res = fp.resolve(fresh_db, "Mystery", "unknown", llm_caller=_stub_llm)
    assert res.usable is False
    assert res.bar_reason == fp.BAR_REASON_UNCERTAIN


# ---------------------------------------------------------------------
# write_resolution
# ---------------------------------------------------------------------


def test_write_resolution_inserts_one_row(fresh_db: Path) -> None:
    res = fp.Resolution(
        usable=True,
        etymon_id=42,
        bar_reason=None,
        resolution_method="descent_lookup",
        confidence="high",
        citation="wiktionary-descent-chain",
        reasoning="test",
    )
    with LexiconDB(fresh_db) as db:
        # need an etymon row to satisfy FK
        db.conn.execute(
            "INSERT INTO etymon (id, canonical_form, language) VALUES (42, 'x', 'old-english')"
        )
        rid = fp.write_resolution(db.conn, input_name="Test", input_description="t", resolution=res)
        db.commit()
    assert rid > 0

    conn = sqlite3.connect(fresh_db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM fantasy_morpheme").fetchall()
    assert len(rows) == 1
    r = rows[0]
    assert r["input_name"] == "Test"
    assert r["usable"] == 1
    assert r["etymon_id"] == 42
    assert r["approach_version"] == fp.APPROACH_VERSION


def test_write_resolution_upserts_on_same_name_and_version(fresh_db: Path) -> None:
    """Re-running with the same approach_version updates in place; a
    bumped version creates a NEW row so we keep the audit trail."""
    with LexiconDB(fresh_db) as db:
        db.conn.execute(
            "INSERT INTO etymon (id, canonical_form, language) VALUES (1, 'x', 'old-english')"
        )
        # First write: barred
        fp.write_resolution(
            db.conn,
            input_name="Test",
            input_description="d1",
            resolution=fp.Resolution(
                usable=False,
                etymon_id=None,
                bar_reason=fp.BAR_REASON_NO_ETYMOLOGY,
                resolution_method="descent_lookup",
                confidence="low",
                citation=None,
                reasoning="r1",
            ),
        )
        # Second write same name + version: upserts to usable
        fp.write_resolution(
            db.conn,
            input_name="Test",
            input_description="d2",
            resolution=fp.Resolution(
                usable=True,
                etymon_id=1,
                bar_reason=None,
                resolution_method="llm_full_research",
                confidence="high",
                citation="LSJ",
                reasoning="r2",
            ),
        )
        # Third write with a different version: separate row
        fp.write_resolution(
            db.conn,
            input_name="Test",
            input_description="d3",
            resolution=fp.Resolution(
                usable=True,
                etymon_id=1,
                bar_reason=None,
                resolution_method="manual",
                confidence="high",
                citation=None,
                reasoning="r3",
            ),
            approach_version="fantasy-v2",
        )
        db.commit()

    conn = sqlite3.connect(fresh_db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM fantasy_morpheme ORDER BY id").fetchall()
    assert len(rows) == 2
    # First row got upserted to the second write's content
    assert rows[0]["usable"] == 1
    assert rows[0]["resolution_method"] == "llm_full_research"
    assert rows[0]["citation"] == "LSJ"
    assert rows[0]["approach_version"] == fp.APPROACH_VERSION
    # Second row is the v2 audit
    assert rows[1]["approach_version"] == "fantasy-v2"
    assert rows[1]["resolution_method"] == "manual"


def test_tag_etymon_as_fantasy_is_idempotent(fresh_db: Path) -> None:
    with LexiconDB(fresh_db) as db:
        db.conn.execute(
            "INSERT INTO etymon (id, canonical_form, language) VALUES (1, 'x', 'old-english')"
        )
        fp.tag_etymon_as_fantasy(db.conn, 1)
        fp.tag_etymon_as_fantasy(db.conn, 1)  # second call should not fail
        db.commit()

    conn = sqlite3.connect(fresh_db)
    rows = conn.execute("SELECT tag FROM etymon_tag WHERE etymon_id = 1").fetchall()
    assert rows == [("fantasy",)]


# ---------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------


def test_cli_mine_fantasy_name_single_with_apply(fresh_db: Path, monkeypatch) -> None:
    """End-to-end CLI: pre-filter resolves, --apply writes the row + tags."""
    from click.testing import CliRunner

    from wyrd.generators.kenning import cli as cli_mod

    with LexiconDB(fresh_db) as db:
        greek_id = _seed_etymon(db, canonical_form="ἅρπυια", language="ancient-greek")
        modern_id = _seed_etymon(db, canonical_form="harpy", language="modern-english")
        _seed_descent(db, parent_id=greek_id, child_id=modern_id)
        db.commit()

    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli,
        [
            "lexicon",
            "mine-fantasy-name",
            "--db",
            str(fresh_db),
            "--name",
            "Harpy",
            "--description",
            "winged creature from Greek mythology",
            "--apply",
        ],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")

    conn = sqlite3.connect(fresh_db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM fantasy_morpheme").fetchall()
    assert len(rows) == 1
    assert rows[0]["usable"] == 1
    assert rows[0]["etymon_id"] == greek_id
    # 'fantasy' tag got applied to the resolved etymon
    tags = {
        r[0]
        for r in conn.execute(
            "SELECT tag FROM etymon_tag WHERE etymon_id = ?", (greek_id,)
        ).fetchall()
    }
    assert "fantasy" in tags


def test_cli_mine_fantasy_name_dry_run_writes_nothing(fresh_db: Path) -> None:
    from click.testing import CliRunner

    from wyrd.generators.kenning import cli as cli_mod

    with LexiconDB(fresh_db) as db:
        greek_id = _seed_etymon(db, canonical_form="ἅρπυια", language="ancient-greek")
        modern_id = _seed_etymon(db, canonical_form="harpy", language="modern-english")
        _seed_descent(db, parent_id=greek_id, child_id=modern_id)
        db.commit()

    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli,
        [
            "lexicon",
            "mine-fantasy-name",
            "--db",
            str(fresh_db),
            "--name",
            "Harpy",
            "--description",
            "winged creature",
        ],
    )
    assert result.exit_code == 0
    conn = sqlite3.connect(fresh_db)
    rows = conn.execute("SELECT COUNT(*) FROM fantasy_morpheme").fetchone()
    assert rows[0] == 0


def test_cli_mine_fantasy_name_batch_mode(fresh_db: Path, tmp_path: Path) -> None:
    """JSONL batch input: two names, one resolves, one is barred via skip-llm."""
    from click.testing import CliRunner

    from wyrd.generators.kenning import cli as cli_mod

    with LexiconDB(fresh_db) as db:
        greek_id = _seed_etymon(db, canonical_form="ἅρπυια", language="ancient-greek")
        modern_id = _seed_etymon(db, canonical_form="harpy", language="modern-english")
        _seed_descent(db, parent_id=greek_id, child_id=modern_id)
        db.commit()

    batch = tmp_path / "names.jsonl"
    batch.write_text(
        json.dumps({"name": "Harpy", "description": "Greek winged creature"})
        + "\n"
        + json.dumps({"name": "Gnoll", "description": "Dunsany hyena humanoid"})
        + "\n"
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli,
        [
            "lexicon",
            "mine-fantasy-name",
            "--db",
            str(fresh_db),
            "--batch",
            str(batch),
            "--skip-llm",
            "--apply",
        ],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")

    conn = sqlite3.connect(fresh_db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM fantasy_morpheme ORDER BY input_name").fetchall()
    assert len(rows) == 2
    by_name = {r["input_name"]: r for r in rows}
    assert by_name["Harpy"]["usable"] == 1
    assert by_name["Gnoll"]["usable"] == 0
    assert by_name["Gnoll"]["bar_reason"] == fp.BAR_REASON_NO_ETYMOLOGY


def test_cli_mine_fantasy_name_rejects_conflicting_input_modes(
    fresh_db: Path, tmp_path: Path
) -> None:
    from click.testing import CliRunner

    from wyrd.generators.kenning import cli as cli_mod

    batch = tmp_path / "names.jsonl"
    batch.write_text("")  # exists but empty so Click's exists=True passes
    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli,
        [
            "lexicon",
            "mine-fantasy-name",
            "--db",
            str(fresh_db),
            "--name",
            "Harpy",
            "--description",
            "x",
            "--batch",
            str(batch),
        ],
    )
    assert result.exit_code != 0
    assert "mutually exclusive" in (result.output + (result.stderr or ""))


def test_cli_mine_fantasy_name_requires_input(fresh_db: Path) -> None:
    from click.testing import CliRunner

    from wyrd.generators.kenning import cli as cli_mod

    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli,
        ["lexicon", "mine-fantasy-name", "--db", str(fresh_db)],
    )
    assert result.exit_code != 0
