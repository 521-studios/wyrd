"""Tests for the wyrd-vz7f fantasy creature runtime surface.

Covers:
- ``collect_fantasy_morphemes`` lexicon → bundle projection
- ``load_fantasy_morphemes`` bundle → runtime parse (with legacy
  list-shape and dict-shape inputs)
- ``KenningCreature`` Generator end-to-end
- ``wyrd kenning creature`` CLI command
- Defensive paths: no fantasy_morpheme table, unknown creature,
  OCR-merged linked etymon (follows the merge chain).
"""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema
from wyrd.generators.kenning.meaning import load_fantasy_morphemes


def _seed_fantasy_db(db_path: Path) -> None:
    """Build a tiny lexicon with two usable fantasy morphemes — one
    direct (Harpy → ancient-greek ἅρπυια) and one OCR-merged (Angel,
    where the linked etymon was merged into a winner). Plus an
    'usable=0' row that should be excluded."""
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        # Migrate to ensure fantasy_morpheme table exists (the
        # init_schema baseline may predate it).
        from wyrd.generators.kenning.lexicon import migrate_schema

        migrate_schema(db)
        # Direct case: Harpy → 'ἅρπυια' (ancient-greek), no merge.
        harpy_etymon = db.upsert_etymon("ἅρπυια", "ancient-greek")
        # Add a gloss for richer explanation rendering.
        db.add_gloss(harpy_etymon, "snatcher")
        # OCR-merge case: 'Angel' linked to an etymon that's been
        # merged into a winner.
        winner = db.upsert_etymon("Engel", "old-english")
        loser = db.upsert_etymon("Eangel", "old-english")
        db.conn.execute("UPDATE etymon SET merged_into_id = ? WHERE id = ?", (winner, loser))
        # Unusable row (bar_reason=modern_coinage) — should NOT surface.
        modern_etymon = db.upsert_etymon("transformer", "modern-english")
        db.conn.execute(
            """
            INSERT INTO fantasy_morpheme
              (input_name, usable, etymon_id, resolution_method,
               approach_version, citation, processed_at)
            VALUES
              ('Harpy', 1, ?, 'descent', 'v1', 'wiktionary-descent-chain',
                '2026-05-08T00:00:00Z'),
              ('Angel', 1, ?, 'descent', 'v1', 'wiktionary-descent-chain',
                '2026-05-08T00:00:00Z'),
              ('Transformer', 0, ?, 'llm', 'v1', '', '2026-05-08T00:00:00Z')
            """,
            (harpy_etymon, loser, modern_etymon),
        )
        db.commit()


# --- collector -----------------------------------------------------------


def test_collect_fantasy_morphemes_returns_usable_rows(tmp_path: Path) -> None:
    """wyrd-vz7f: only usable=1 rows surface in the bundle projection."""
    from wyrd.generators.kenning.lexicon import collect_fantasy_morphemes

    db_path = tmp_path / "lexicon.db"
    _seed_fantasy_db(db_path)
    with LexiconDB(db_path) as db:
        out = collect_fantasy_morphemes(db)
    # Harpy + Angel surface; Transformer (usable=0) excluded.
    assert set(out.keys()) == {"Harpy", "Angel"}
    harpy = out["Harpy"]
    assert harpy["language"] == "ancient-greek"
    assert harpy["canonical_form"] == "ἅρπυια"
    assert harpy["citation"] == "wiktionary-descent-chain"
    assert "snatcher" in harpy["glosses"]


def test_collect_fantasy_morphemes_follows_merge_chain(tmp_path: Path) -> None:
    """Angel's linked etymon was OCR-merged into 'Engel'; the
    collector follows the merged_into chain so the surfaced
    canonical_form is the winner's, not the loser's. Without this
    follow-through, ~10 of the 244 usable rows would route to a
    tombstone form. Glosses and era reflexes also need to be
    queried against the WINNER (since etymon_id surfaces as the
    winner's id, not the loser's) — pinned here so a future query
    edit that splits these from canonical_form's resolution would
    surface as a test failure."""
    from wyrd.generators.kenning.lexicon import collect_fantasy_morphemes

    db_path = tmp_path / "lexicon.db"
    _seed_fantasy_db(db_path)
    # Add a gloss to the WINNER and the LOSER — the collector should
    # surface the winner's, not the loser's.
    with LexiconDB(db_path) as db:
        winner_id = db.conn.execute(
            "SELECT id FROM etymon WHERE canonical_form = 'Engel'"
        ).fetchone()["id"]
        loser_id = db.conn.execute(
            "SELECT id FROM etymon WHERE canonical_form = 'Eangel'"
        ).fetchone()["id"]
        db.add_gloss(winner_id, "messenger")
        db.add_gloss(loser_id, "loser-gloss-should-not-surface")
        db.commit()
        out = collect_fantasy_morphemes(db)
    angel = out["Angel"]
    assert angel["canonical_form"] == "Engel"  # winner, not loser
    assert angel["language"] == "old-english"
    # etymon_id surfaces the WINNER's id (so era reflexes / glosses
    # are queried against the right row).
    assert angel["etymon_id"] == winner_id
    # Winner's gloss surfaces; loser's stays on the loser row.
    assert "messenger" in angel["glosses"]
    assert "loser-gloss-should-not-surface" not in angel["glosses"]


def test_collect_fantasy_morphemes_returns_empty_when_table_missing(
    tmp_path: Path,
) -> None:
    """Older DBs predate the wyrd-ami pipeline. Collector returns
    empty dict rather than crashing the export — same defensive
    pattern as wyrd-c1vq's collect_canonical_decompositions."""
    from wyrd.generators.kenning.lexicon import collect_fantasy_morphemes

    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    # Drop the table to simulate a pre-wyrd-ami DB.
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP TABLE IF EXISTS fantasy_morpheme")
        conn.commit()
    with LexiconDB(db_path) as db:
        out = collect_fantasy_morphemes(db)
    assert out == {}


# --- loader --------------------------------------------------------------


def test_load_fantasy_morphemes_parses_bundle_field() -> None:
    """Bundle dict-shape with fantasy_morphemes field → runtime
    {lowercase_input_name: {...}} map."""
    bundle = {
        "subjects": [],
        "fantasy_morphemes": {
            "Harpy": {
                "input_name": "Harpy",
                "etymon_id": 42,
                "language": "ancient-greek",
                "canonical_form": "ἅρπυια",
                "english_shaped": "harpyia",
                "glosses": ["snatcher", "harpy"],
                "citation": "wiktionary-descent-chain",
                "era_reflexes": {
                    "modern-english": [{"form": "harpy", "source": "cluster"}],
                },
            }
        },
    }
    out = load_fantasy_morphemes(bundle)
    # Lowercased key for case-insensitive lookup.
    assert "harpy" in out
    assert "Harpy" not in out  # case-folded
    entry = out["harpy"]
    assert entry["input_name"] == "Harpy"  # display case preserved
    assert entry["language"] == "ancient-greek"
    assert entry["glosses"] == ["snatcher", "harpy"]
    # era_reflexes round-tripped to internal (form, source) tuple shape.
    assert entry["era_reflexes"]["modern-english"] == [("harpy", "cluster")]


def test_load_fantasy_morphemes_empty_for_legacy_list_shape() -> None:
    """Legacy list-shape bundles (pre-wyrd-c1vq) have no top-level
    fantasy_morphemes field — return empty rather than crash."""
    assert load_fantasy_morphemes([]) == {}
    assert load_fantasy_morphemes([{"meaning": [], "modifier_tags": [], "words": []}]) == {}


def test_load_fantasy_morphemes_empty_when_field_absent() -> None:
    """Dict-shape bundles without the field return empty (the field
    is only emitted when the lexicon has fantasy_morpheme data)."""
    assert load_fantasy_morphemes({"subjects": []}) == {}
    assert load_fantasy_morphemes({"subjects": [], "joiners": {}}) == {}


def test_load_fantasy_morphemes_skips_malformed_entries() -> None:
    """Defensive parse: non-dict entries skipped. A stray null or
    integer in the bundle shouldn't crash the load."""
    bundle = {
        "subjects": [],
        "fantasy_morphemes": {
            "GoodEntry": {
                "input_name": "GoodEntry",
                "etymon_id": 1,
                "language": "latin",
                "canonical_form": "bonum",
            },
            "BadEntry": "not-a-dict",
            "AlsoBad": None,
        },
    }
    out = load_fantasy_morphemes(bundle)
    assert "goodentry" in out
    assert "badentry" not in out
    assert "alsobad" not in out


# --- KenningCreature Generator -------------------------------------------


def test_kenning_creature_generator_finds_known_creature(bundle_swapper) -> None:
    """End-to-end Generator: populate the bundle, swap it into the
    package data, invoke KenningCreature.generate, assert the
    explanation surfaces the linked etymon. wyrd-lqlw: migrated from
    the source-tree file-swap antipattern to ``bundle_swapper`` so a
    test interrupt can't leave a corrupted meanings.json behind."""
    from wyrd.generators.kenning import KenningCreature

    bundle = {
        "subjects": [],
        "fantasy_morphemes": {
            "Harpy": {
                "input_name": "Harpy",
                "etymon_id": 1,
                "language": "ancient-greek",
                "canonical_form": "ἅρπυια",
                "english_shaped": "harpyia",
                "glosses": ["snatcher"],
                "citation": "test-citation",
                "era_reflexes": {},
            }
        },
    }
    with bundle_swapper(bundle):
        gen = KenningCreature()
        result = gen.generate({"name": "Harpy"}, seed=0)
        assert result.result == "Harpy"
        assert "ancient-greek" in result.explanation
        assert "ἅρπυια" in result.explanation
        assert "harpyia" in result.explanation
        assert "snatcher" in result.explanation
        assert "test-citation" in result.explanation


def test_kenning_creature_generator_case_insensitive(bundle_swapper) -> None:
    """Lowercase / uppercase / mixed-case input all resolve to the
    same row, matching the lexicon's COLLATE NOCASE semantics."""
    from wyrd.generators.kenning import KenningCreature

    bundle = {
        "subjects": [],
        "fantasy_morphemes": {
            "Drake": {
                "input_name": "Drake",
                "etymon_id": 2,
                "language": "sco",
                "canonical_form": "drake",
                "english_shaped": "",
                "glosses": [],
                "citation": "test",
                "era_reflexes": {},
            }
        },
    }
    with bundle_swapper(bundle):
        gen = KenningCreature()
        for cased in ("drake", "Drake", "DRAKE", "DrAkE"):
            result = gen.generate({"name": cased}, seed=0)
            assert result.result == "Drake"
            assert "sco" in result.explanation


def test_kenning_creature_generator_unknown_returns_polite_message(
    bundle_swapper,
) -> None:
    """Unknown creature names return a 'no etymology found' result,
    not an exception. Pinned so chaining the CLI against a list of
    arbitrary inputs is safe."""
    from wyrd.generators.kenning import KenningCreature

    with bundle_swapper({"subjects": [], "fantasy_morphemes": {}}):
        gen = KenningCreature()
        result = gen.generate({"name": "NotARealCreature"}, seed=0)
        assert result.result == "NotARealCreature"
        assert "No etymology found" in result.explanation


def test_kenning_creature_generator_renders_era_reflexes(bundle_swapper) -> None:
    """When era reflexes are present, they appear in the explanation
    line. Verifies the wyrd-jbcu source-aware shape round-trips
    through to the human-readable output."""
    from wyrd.generators.kenning import KenningCreature

    bundle = {
        "subjects": [],
        "fantasy_morphemes": {
            "Wyrm": {
                "input_name": "Wyrm",
                "etymon_id": 3,
                "language": "old-english",
                "canonical_form": "wyrm",
                "english_shaped": "",
                "glosses": ["serpent"],
                "citation": "scholarly",
                "era_reflexes": {
                    "middle-english": [
                        {"form": "worm", "source": "cluster"},
                        {"form": "wyrm", "source": "phonology-rule:v1"},
                    ],
                    "modern-english": [{"form": "worm", "source": "cluster"}],
                },
            }
        },
    }
    with bundle_swapper(bundle):
        gen = KenningCreature()
        result = gen.generate({"name": "Wyrm"}, seed=0)
        assert "Era reflexes" in result.explanation
        assert "middle-english" in result.explanation
        assert "worm" in result.explanation
        assert "modern-english" in result.explanation


def test_kenning_creature_generator_blank_input_raises() -> None:
    """Blank / empty input is a caller bug — fail loudly rather than
    return a misleading 'not found' for an empty string."""
    import pytest

    from wyrd.generators.kenning import KenningCreature

    gen = KenningCreature()
    with pytest.raises(ValueError, match="name is required"):
        gen.generate({"name": ""}, seed=0)
    with pytest.raises(ValueError, match="name is required"):
        gen.generate({}, seed=0)


# --- CLI -----------------------------------------------------------------


def test_kenning_creature_cli_prints_result_and_explanation(bundle_swapper) -> None:
    """`wyrd kenning creature Harpy` prints the result line + the
    indented explanation line. End-to-end pin so the CLI stays in
    sync with the Generator's output shape."""
    bundle = {
        "subjects": [],
        "fantasy_morphemes": {
            "Harpy": {
                "input_name": "Harpy",
                "etymon_id": 1,
                "language": "ancient-greek",
                "canonical_form": "ἅρπυια",
                "english_shaped": "",
                "glosses": [],
                "citation": "test",
                "era_reflexes": {},
            }
        },
    }
    with bundle_swapper(bundle):
        runner = CliRunner()
        result = runner.invoke(cli, ["creature", "Harpy"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "Harpy" in result.output
        assert "ancient-greek" in result.output
        assert "ἅρπυια" in result.output


def test_kenning_creature_cli_unknown_creature_polite(bundle_swapper) -> None:
    """CLI on unknown creature exits cleanly with the 'no etymology
    found' message rather than a stack trace."""
    with bundle_swapper({"subjects": [], "fantasy_morphemes": {}}):
        runner = CliRunner()
        result = runner.invoke(cli, ["creature", "Frobnicator"], catch_exceptions=False)
        assert result.exit_code == 0
        assert "No etymology found" in result.output
        assert "Frobnicator" in result.output


# --- generator registration ----------------------------------------------


def test_kenning_creature_registered_in_generator_registry() -> None:
    """The Generator class is registered with name='kenning-creature'
    so the SPA's manifest API surfaces it. Pin the registration so a
    future refactor that drops the register() call surfaces here."""
    import wyrd.generators.kenning  # noqa: F401  ensure registration ran
    from wyrd.registry import get

    gen = get("kenning-creature")
    assert gen is not None
    assert gen.display_name == "Kenning — Creature etymology"
