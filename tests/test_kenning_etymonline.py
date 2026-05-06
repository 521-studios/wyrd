"""Tests for the Etymonline parser + ingester (wyrd-zqkp).

Synthetic prose fixtures + 4 captured real-page texts in
tests/fixtures/etymonline/. Network-free.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from wyrd.generators.kenning.etymonline_ingester import (
    ETYMONLINE_SOURCE_ID,
    ensure_source,
    ingest_sense,
    ingest_text,
)
from wyrd.generators.kenning.etymonline_parser import (
    ChainLink,
    parse_chain,
    parse_headword,
    parse_text,
    split_senses,
)
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema, migrate_schema


@pytest.fixture
def fresh_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        migrate_schema(db)
        db.commit()
    return db_path


_FIXTURES = Path(__file__).parent / "fixtures" / "etymonline"


# --- parse_headword ---------------------------------------------------


def test_parse_headword_simple_pos():
    assert parse_headword("harpy(n.)") == ("harpy", "n.")


def test_parse_headword_numbered_sense():
    assert parse_headword("troll(n.1)") == ("troll", "n.1")


def test_parse_headword_no_pos_returns_none():
    """Defensive — a line without a parenthesized POS marker returns
    (line, None) rather than crashing."""
    assert parse_headword("harpy") == ("harpy", None)


# --- split_senses -----------------------------------------------------


def test_split_senses_single_sense():
    text = (
        "harpy(n.)\n\n"
        "winged monster of ancient mythology, late 14c., from Old French harpie...\n\n"
        "also from late 14c."
    )
    senses = split_senses(text)
    assert len(senses) == 1
    assert senses[0][0] == "harpy(n.)"
    assert "from Old French harpie" in senses[0][1]


def test_split_senses_multiple_senses():
    """troll page has v., n.1, n.2 — three sense blocks."""
    text = _FIXTURES.joinpath("troll.txt").read_text()
    senses = split_senses(text)
    head_lines = [head for head, _ in senses]
    assert "troll(v.)" in head_lines
    assert "troll(n.1)" in head_lines
    assert "troll(n.2)" in head_lines


def test_split_senses_empty_input():
    assert split_senses("") == []
    assert split_senses("\n\n\n") == []


# --- parse_chain (synthetic prose) ------------------------------------


def test_parse_chain_simple_three_link():
    """A clean chain produces three ChainLinks in descent order."""
    summary = (
        "winged monster of ancient mythology, late 14c., "
        "from Old French harpie (14c.), "
        "from Latin harpyia, "
        "from Greek Harpyia."
    )
    chain = parse_chain(summary)
    assert [(c.language, c.word) for c in chain] == [
        ("old-french", "harpie"),
        ("latin", "harpyia"),
        ("ancient-greek", "Harpyia"),
    ]
    assert chain[0].attested_year == "14c."


def test_parse_chain_probably_from_sets_medium_confidence():
    summary = "probably from Old French troller, a hunting term."
    chain = parse_chain(summary)
    assert len(chain) == 1
    assert chain[0].confidence == "medium"
    assert chain[0].language == "old-french"
    assert chain[0].word == "troller"


def test_parse_chain_perhaps_from_sets_low_confidence():
    summary = "perhaps from Old Norse troll."
    chain = parse_chain(summary)
    assert len(chain) == 1
    assert chain[0].confidence == "low"


def test_parse_chain_borrowed_from_emits_borrowing_edge_type():
    summary = "borrowed from Old French gobelin."
    chain = parse_chain(summary)
    assert len(chain) == 1
    assert chain[0].edge_type == "borrowing"


def test_parse_chain_unknown_language_is_skipped():
    """A hedge followed by a language we don't have in the map
    advances past without emitting (no crash, no false match)."""
    summary = "from Klingon ghuy."
    chain = parse_chain(summary)
    assert chain == []


def test_parse_chain_cognate_list_paren_is_ignored():
    """A `(source also of ...)` paren after a chain link doesn't
    pollute subsequent chain detection."""
    summary = (
        "from PIE *derk- (source also of Sanskrit darsata-, Old Irish adcondarc), from Latin extra."
    )
    chain = parse_chain(summary)
    langs = [c.language for c in chain]
    # Must include both the PIE root and the Latin link AFTER the cognate paren.
    assert "proto-indo-european" in langs
    assert "latin" in langs


def test_parse_chain_compare_paren_is_ignored():
    """`(compare Old High German trollen ...)` is a parallel cognate
    note, not a chain link."""
    summary = (
        "from a Germanic source "
        '(compare Old High German trollen "to walk with short steps"), '
        "from Proto-Germanic *truzlanan."
    )
    chain = parse_chain(summary)
    # The Old High German inside `(compare ...)` must NOT be picked up;
    # only Proto-Germanic should appear (the Germanic-source link itself
    # has no specific word so isn't pickable).
    langs = [c.language for c in chain]
    assert "old-high-german" not in langs
    assert "proto-germanic" in langs


def test_parse_chain_gloss_is_captured():
    summary = 'from Greek drakon "serpent, giant seafish".'
    chain = parse_chain(summary)
    assert chain[0].gloss == "serpent, giant seafish"


def test_parse_chain_gloss_trailing_punctuation_is_stripped():
    """Glosses end with `,` or `.` punctuation in real prose; we strip
    those for cleaner storage."""
    summary = 'from Greek kobalos "impudent rogue, knave," from earlier.'
    chain = parse_chain(summary)
    # Glosses end with `,` or `.` punctuation in real prose; strip.
    assert chain[0].gloss == "impudent rogue, knave"


def test_parse_chain_attested_year_forms():
    summary = (
        "from Old French harpie (14c.), "
        "from Latin foo (c. 1400), "
        "from Old French bar (early 15c.), "
        "from Old French baz (1610s)."
    )
    chain = parse_chain(summary)
    years = [c.attested_year for c in chain]
    assert years == ["14c.", "c. 1400", "early 15c.", "1610s"]


def test_parse_chain_pie_alias_resolves():
    """'PIE *derk-' should resolve to proto-indo-european."""
    summary = "from PIE *derk-."
    chain = parse_chain(summary)
    assert chain[0].language == "proto-indo-european"
    assert chain[0].word == "*derk-"


# --- parse_text (real fixtures) ---------------------------------------


def test_parse_text_harpy_fixture():
    text = (_FIXTURES / "harpy.txt").read_text()
    senses = parse_text(text)
    assert len(senses) == 1
    sense = senses[0]
    assert sense.headword == "harpy"
    assert sense.pos == "n."
    chain_pairs = [(c.language, c.word) for c in sense.chain]
    assert chain_pairs == [
        ("old-french", "harpie"),
        ("latin", "harpyia"),
        ("ancient-greek", "Harpyia"),
    ]


def test_parse_text_dragon_fixture():
    """dragon has the deepest chain in the fixture set: Old French
    → Latin → Greek → PIE *derk-."""
    text = (_FIXTURES / "dragon.txt").read_text()
    senses = parse_text(text)
    assert len(senses) == 1
    chain = senses[0].chain
    langs = [c.language for c in chain]
    assert langs == ["old-french", "latin", "ancient-greek", "proto-indo-european"]


def test_parse_text_troll_v_fixture_mixes_confidence_levels():
    """troll(v.) has 'probably from Old French troller' (medium) and
    'from Proto-Germanic *truzlanan' (high)."""
    text = (_FIXTURES / "troll.txt").read_text()
    senses = parse_text(text)
    troll_v = next(s for s in senses if s.pos == "v.")
    by_lang = {c.language: c for c in troll_v.chain}
    assert by_lang["old-french"].confidence == "medium"
    assert by_lang["proto-germanic"].confidence == "high"


def test_parse_text_goblin_fixture_three_links():
    text = (_FIXTURES / "goblin.txt").read_text()
    senses = parse_text(text)
    assert len(senses) == 1
    chain = senses[0].chain
    langs = [c.language for c in chain]
    assert "old-french" in langs
    assert "medieval-latin" in langs
    assert "ancient-greek" in langs


# --- ingest_sense (DB writes) -----------------------------------------


def _seed_etymon(db: LexiconDB, *, canonical_form: str, language: str) -> int:
    return db.upsert_etymon(canonical_form, language)


def _all_descent_edges(db: LexiconDB) -> list[tuple[str, str, str, str]]:
    rows = db.conn.execute(
        """SELECT p.canonical_form AS parent, c.canonical_form AS child,
                  d.edge_type, d.source_id
           FROM etymon_descent d
           JOIN etymon p ON p.id = d.parent_id
           JOIN etymon c ON c.id = d.child_id
           ORDER BY parent, child, edge_type"""
    ).fetchall()
    return [(r["parent"], r["child"], r["edge_type"], r["source_id"]) for r in rows]


def test_ingest_sense_writes_chain_etymons_and_edges(fresh_db: Path) -> None:
    """Adjacent chain links produce descent edges, parent → child in
    the leaf-ward direction."""
    text = (_FIXTURES / "harpy.txt").read_text()
    sense = parse_text(text)[0]
    with LexiconDB(fresh_db) as db:
        # Seed the headword so the leaf edge has somewhere to go.
        _seed_etymon(db, canonical_form="harpy", language="modern-english")
        ensure_source(db)
        counts = ingest_sense(db, sense, apply=True)
        db.commit()
        edges = _all_descent_edges(db)

    # Three chain links → 2 adjacent edges + 1 leaf edge = 3 edges total.
    assert counts["edges_added"] == 3
    # Source attribution is etymonline on every edge.
    sources = {source for *_, source in edges}
    assert sources == {ETYMONLINE_SOURCE_ID}
    # The full chain is wired: Greek → Latin → OFr → modern-english.
    edge_pairs = {(p, c) for p, c, _, _ in edges}
    assert ("Harpyia", "harpyia") in edge_pairs
    assert ("harpyia", "harpie") in edge_pairs
    assert ("harpie", "harpy") in edge_pairs


def test_ingest_sense_skips_leaf_edge_when_headword_missing(fresh_db: Path) -> None:
    """If the headword isn't in the etymon table (not in our wiktextract
    corpus), still write the chain and adjacent edges — but skip the
    leaf edge and report it via the counter."""
    text = (_FIXTURES / "harpy.txt").read_text()
    sense = parse_text(text)[0]
    with LexiconDB(fresh_db) as db:
        # NO headword seeded.
        ensure_source(db)
        counts = ingest_sense(db, sense, apply=True)
        db.commit()
    assert counts["leaf_edge_skipped_no_headword"] == 1
    # Still added 2 adjacent chain edges.
    assert counts["edges_added"] == 2


def test_ingest_sense_is_idempotent(fresh_db: Path) -> None:
    """Re-running on the same Sense produces zero new edges via the
    etymon_descent UNIQUE on (parent, child, edge_type, source)."""
    text = (_FIXTURES / "harpy.txt").read_text()
    sense = parse_text(text)[0]
    with LexiconDB(fresh_db) as db:
        _seed_etymon(db, canonical_form="harpy", language="modern-english")
        ensure_source(db)
        counts1 = ingest_sense(db, sense, apply=True)
        db.commit()
        ensure_source(db)
        counts2 = ingest_sense(db, sense, apply=True)
        db.commit()
    assert counts1["edges_added"] == 3
    assert counts2["edges_added"] == 0
    assert counts2["edges_skipped_dupe"] == 3


def test_ingest_sense_dry_run_writes_nothing(fresh_db: Path) -> None:
    sense_obj = parse_text((_FIXTURES / "harpy.txt").read_text())[0]
    with LexiconDB(fresh_db) as db:
        _seed_etymon(db, canonical_form="harpy", language="modern-english")
        ensure_source(db)
        counts = ingest_sense(db, sense_obj, apply=False)
        n_etymons = db.conn.execute("SELECT COUNT(*) FROM etymon").fetchone()[0]
        n_edges = db.conn.execute("SELECT COUNT(*) FROM etymon_descent").fetchone()[0]
    assert counts["chain_links"] == 3
    # Only the seeded headword exists; no chain links written.
    assert n_etymons == 1
    assert n_edges == 0


def test_ingest_sense_carries_confidence_through_to_edge(fresh_db: Path) -> None:
    """`probably from` chain links land as confidence='medium' on the
    descent edge."""
    sense_obj = parse_text(
        "troll(v.)\n\nlate 14c., probably from Old French troller, a hunting term.\n\n"
    )[0]
    with LexiconDB(fresh_db) as db:
        _seed_etymon(db, canonical_form="troll", language="modern-english")
        ensure_source(db)
        ingest_sense(db, sense_obj, apply=True)
        db.commit()
        rows = db.conn.execute(
            "SELECT confidence FROM etymon_descent WHERE source_id='etymonline'"
        ).fetchall()
    assert rows[0]["confidence"] == "medium"


def test_ingest_sense_no_chain_returns_early(fresh_db: Path) -> None:
    """A sense without a parseable chain (e.g. all hedges led to
    unknown languages) is a no-op on the DB."""
    sense_obj = parse_text(
        "fakeword(n.)\n\nof unknown origin, perhaps related to other unknown words.\n\n"
    )[0]
    with LexiconDB(fresh_db) as db:
        ensure_source(db)
        counts = ingest_sense(db, sense_obj, apply=True)
        n = db.conn.execute("SELECT COUNT(*) FROM etymon_descent").fetchone()[0]
    assert counts["edges_added"] == 0
    assert n == 0


def test_ingest_sense_records_glosses(fresh_db: Path) -> None:
    """Glosses captured by the parser are persisted as etymon_gloss rows."""
    sense_obj = parse_text(
        'dragon(n.)\n\nmid-13c., from Greek drakon "serpent, giant seafish".\n\n'
    )[0]
    with LexiconDB(fresh_db) as db:
        _seed_etymon(db, canonical_form="dragon", language="modern-english")
        ensure_source(db)
        counts = ingest_sense(db, sense_obj, apply=True)
        db.commit()
        rows = db.conn.execute(
            """SELECT eg.gloss FROM etymon_gloss eg
               JOIN etymon e ON e.id = eg.etymon_id
               WHERE e.canonical_form='drakon'"""
        ).fetchall()
    assert counts["glosses_added"] == 1
    assert rows[0]["gloss"] == "serpent, giant seafish"


# --- ingest_text (multi-sense pages + source row) ---------------------


def test_ingest_text_creates_etymonline_source_row(fresh_db: Path) -> None:
    """First call creates the source row; subsequent calls upsert."""
    text = "harpy(n.)\n\nlate 14c., from Old French harpie."
    with LexiconDB(fresh_db) as db:
        ingest_text(db, text, apply=True)
        rows = db.conn.execute(
            "SELECT id, title FROM source WHERE id=?", (ETYMONLINE_SOURCE_ID,)
        ).fetchall()
    assert len(rows) == 1
    assert "Etymonline" in rows[0]["title"]


def test_ingest_text_handles_multi_sense_page(fresh_db: Path) -> None:
    """A page like troll.txt has v., n.1, n.2 — all three senses are
    parsed and ingested."""
    text = (_FIXTURES / "troll.txt").read_text()
    with LexiconDB(fresh_db) as db:
        _seed_etymon(db, canonical_form="troll", language="modern-english")
        totals = ingest_text(db, text, apply=True)
    assert totals["senses_parsed"] >= 3
    # At least the v. and n.1 senses have chains we can extract.
    assert totals["senses_with_chain"] >= 2


def test_ingest_text_dry_run_returns_counts(fresh_db: Path) -> None:
    text = (_FIXTURES / "harpy.txt").read_text()
    with LexiconDB(fresh_db) as db:
        totals = ingest_text(db, text, apply=False)
        n_edges = db.conn.execute("SELECT COUNT(*) FROM etymon_descent").fetchone()[0]
    assert totals["senses_parsed"] == 1
    assert totals["chain_links"] == 3
    assert n_edges == 0  # dry-run doesn't write


# --- ChainLink dataclass ----------------------------------------------


def test_chain_link_dataclass_is_frozen():
    """ChainLink is frozen so callers can use it as a dict key /
    set element if useful."""
    link = ChainLink(
        language="latin",
        word="draconem",
        edge_type="inheritance",
        confidence="high",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        link.word = "different"  # type: ignore[misc]
