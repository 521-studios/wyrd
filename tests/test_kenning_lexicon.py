"""Tests for the kenning lexicon authoring DB."""

from __future__ import annotations

import json
import sqlite3
from importlib import resources
from pathlib import Path

import pytest

from wyrd.generators.kenning.lexicon import (
    LANGUAGE_FIELDS,
    NON_LANGUAGE_FIELDS,
    LexiconDB,
    _position_from_usage,
    cluster_ocr_variants,
    derive_lemma_candidate,
    ingest_parsed_entries,
    init_schema,
    link_lemmas,
    migrate_schema,
    normalize_ocr_form,
    seed_from_meanings,
)
from wyrd.generators.kenning.skeat_parser import ParsedElement, ParsedEntry


@pytest.fixture
def fresh_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    return db_path


def test_position_from_usage() -> None:
    assert _position_from_usage("Ac-") == "pre"
    assert _position_from_usage("-ham") == "post"
    assert _position_from_usage("-ar-") == "inner"


def test_init_schema_creates_tables(fresh_db: Path) -> None:
    conn = sqlite3.connect(fresh_db)
    try:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
        tables = {row[0] for row in cur.fetchall()}
    finally:
        conn.close()

    expected = {
        "source",
        "etymon",
        "etymon_gloss",
        "etymon_tag",
        "etymon_citation",
        "reflex",
        "reflex_etymon",
        "toponym",
        "toponym_attestation",
        "toponym_etymology",
        "toponym_etymology_element",
    }
    assert expected.issubset(tables)


def test_upsert_etymon_returns_same_id_on_duplicate(fresh_db: Path) -> None:
    with LexiconDB(fresh_db) as db:
        first = db.upsert_etymon("ham", "old-english", modifier_type="Habitative")
        second = db.upsert_etymon("ham", "old-english")
        assert first == second
        # Different language → different etymon row.
        third = db.upsert_etymon("ham", "old-norse")
        assert third != first


def test_upsert_reflex_returns_same_id_on_duplicate(fresh_db: Path) -> None:
    with LexiconDB(fresh_db) as db:
        first = db.upsert_reflex("-ham", "post")
        second = db.upsert_reflex("-ham", "post")
        assert first == second
        # Different position → distinct reflex.
        third = db.upsert_reflex("-ham", "inner")
        assert third != first


def test_seed_from_minimal_meanings(fresh_db: Path) -> None:
    """A small synthetic meanings.json fragment seeds correctly."""
    data = [
        {
            "meaning": ["Acorn"],
            "modifier_tags": ["plant", "food"],
            "modifier_type": "Topographical",
            "words": [
                {"modern_usage": "Ac-", "old_english": ["aecern"]},
                {"modern_usage": "-ock", "old_english": ["aecern"]},
            ],
        },
        {
            "meaning": ["Home"],
            "modifier_tags": [],
            "modifier_type": "Habitative",
            "words": [
                {"modern_usage": "-ham", "old_english": ["ham"]},
            ],
        },
    ]

    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="test-src", title="Test")
        db.commit()
        seed_from_meanings(db, data, "test-src")
        stats = db.stats()
        # Two distinct etymons (aecern, ham), three reflexes (Ac-, -ock, -ham).
        assert stats["etymon"] == 2
        assert stats["reflex"] == 3
        # aecern reflexes both link to the aecern etymon: 2 links + 1 for ham.
        assert stats["reflex_etymon"] == 3
        # Citations: one per etymon-source pair.
        assert stats["etymon_citation"] == 2

        # Glosses: aecern gets "Acorn", ham gets "Home".
        gloss_rows = db.conn.execute(
            "SELECT etymon_id, gloss FROM etymon_gloss ORDER BY gloss"
        ).fetchall()
        assert {row["gloss"] for row in gloss_rows} == {"Acorn", "Home"}

        # Tags from the first subject only — second has none.
        tag_rows = db.conn.execute("SELECT tag FROM etymon_tag").fetchall()
        assert {row["tag"] for row in tag_rows} == {"plant", "food"}


def test_seed_against_bundled_meanings(fresh_db: Path) -> None:
    """End-to-end: ingest the real bundled meanings.json and sanity-check counts."""
    text = resources.files("wyrd.generators.kenning.data").joinpath("meanings.json").read_text()
    data = json.loads(text)

    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="Rando port seed")
        db.commit()
        seed_from_meanings(db, data, "rando-port")
        stats = db.stats()

    # 747 subjects, 2,751 words in the bundled file. Reflex count should match
    # roughly the unique (modern_usage, position) pairs — a couple hundred
    # collisions are expected, but we should be in the low thousands.
    assert 1500 < stats["reflex"] < 3500
    # Etymons are grouped by (form, language); should be fewer than reflexes.
    assert 500 < stats["etymon"] < stats["reflex"]
    # Every etymon has at least one citation back to rando-port.
    assert stats["etymon_citation"] == stats["etymon"]


def test_ingest_parsed_entries_writes_etymology_rows(fresh_db: Path) -> None:
    """High/medium-confidence entries get full etymology rows; low-confidence
    entries record the toponym only."""
    entries = [
        ParsedEntry(
            toponym="Alpha",
            section_suffix="-ton",
            historical_form="fake-tun",
            elements=[
                ParsedElement(form="fake", language="old-english", position="pre"),
                ParsedElement(
                    form="tun", language="old-english", gloss="enclosure", position="post"
                ),
            ],
            confidence="high",
            source_quote="Alpha. This is the A.S. fake-tun.",
        ),
        ParsedEntry(
            toponym="Beta",
            section_suffix="-ton",
            historical_form=None,
            elements=[],
            confidence="low",
            source_quote="Beta. (no etymology recovered)",
        ),
    ]

    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="test-src", title="Test")
        db.commit()
        counts = ingest_parsed_entries(db, entries, "test-src", region="Testshire")
        stats = db.stats()

    # Both toponyms recorded; only Alpha has an etymology row.
    assert counts["toponyms"] == 2
    assert counts["etymologies"] == 1
    assert counts["elements"] == 2
    assert stats["toponym"] == 2
    assert stats["toponym_etymology"] == 1
    assert stats["toponym_etymology_element"] == 2
    # The "tun" gloss was recorded.
    assert stats["etymon_gloss"] == 1


def test_ingest_then_consensus_view(fresh_db: Path) -> None:
    """Citing the same etymon from two sources promotes its witness count."""
    entries_a = [
        ParsedEntry(
            toponym="Foo",
            section_suffix="-ton",
            historical_form="bar-tun",
            elements=[
                ParsedElement(form="bar", language="old-english", position="pre"),
                ParsedElement(form="tun", language="old-english", position="post"),
            ],
            confidence="high",
            source_quote="Foo. From A.S. bar-tun.",
        ),
    ]
    entries_b = [
        ParsedEntry(
            toponym="Baz",
            section_suffix="-ton",
            historical_form="qux-tun",
            elements=[
                ParsedElement(form="qux", language="old-english", position="pre"),
                ParsedElement(form="tun", language="old-english", position="post"),
            ],
            confidence="high",
            source_quote="Baz. From A.S. qux-tun.",
        ),
    ]

    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src-a", title="Source A")
        db.upsert_source(id="src-b", title="Source B")
        db.commit()
        ingest_parsed_entries(db, entries_a, "src-a")
        ingest_parsed_entries(db, entries_b, "src-b")

        consensus = db.conn.execute(
            "SELECT canonical_form, witnesses FROM etymon_consensus ORDER BY witnesses DESC"
        ).fetchall()

    by_form = {r["canonical_form"]: r["witnesses"] for r in consensus}
    assert by_form["tun"] == 2  # cited by both sources
    assert by_form["bar"] == 1
    assert by_form["qux"] == 1


def test_normalize_ocr_form_strips_dashes_and_lowers() -> None:
    assert normalize_ocr_form("Hædan") == "haedan"
    assert normalize_ocr_form("-tun") == "tun"
    assert normalize_ocr_form("HAM") == "ham"
    assert normalize_ocr_form("Bre-") == "bre"


def test_normalize_ocr_form_collapses_ocr_variants() -> None:
    """The same OE word in different OCR renderings normalizes to one key."""
    canonical = normalize_ocr_form("Hædan")
    assert normalize_ocr_form("Hcsdan") == canonical  # OCR mangle of æ→cs
    assert normalize_ocr_form("Haedan") == canonical  # ASCII-equivalent
    assert normalize_ocr_form("HÆDAN") == canonical  # uppercase
    # Eth (ð) variants
    assert normalize_ocr_form("hyð") == normalize_ocr_form("hydh")


def test_normalize_ocr_form_preserves_distinct_etymons() -> None:
    """Genuinely different forms must NOT collapse to the same key."""
    assert normalize_ocr_form("ham") != normalize_ocr_form("hamm")
    assert normalize_ocr_form("tun") != normalize_ocr_form("dun")
    assert normalize_ocr_form("bere") != normalize_ocr_form("bera")


def test_cluster_ocr_variants_dry_run_reports_without_writing(fresh_db: Path) -> None:
    """Dry-run must not touch the DB."""
    with LexiconDB(fresh_db) as db:
        # Create three variants of "Hædan" + a distinct etymon
        haedan = db.upsert_etymon("Hædan", "old-english")
        hcsdan = db.upsert_etymon("Hcsdan", "old-english")
        haedan_ascii = db.upsert_etymon("Haedan", "old-english")
        ham = db.upsert_etymon("ham", "old-english")
        db.commit()

        before = db.stats()
        result = cluster_ocr_variants(db, apply=False)
        after = db.stats()

    # Dry-run reports 1 mergeable group, 2 etymons removable.
    assert result["groups"] == 1
    assert result["etymons_merged"] == 2
    # No mutation.
    assert before["etymon"] == after["etymon"]
    assert haedan != hcsdan != haedan_ascii != ham


def test_cluster_ocr_variants_apply_merges_and_repoints(fresh_db: Path) -> None:
    """With apply=True, redundant etymons are merged into the lowest-id one
    and citations/glosses/links repoint correctly."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src-a", title="A")
        db.upsert_source(id="src-b", title="B")
        db.commit()

        # Three OCR variants. Lowest id wins.
        winner = db.upsert_etymon("Hædan", "old-english")
        loser_a = db.upsert_etymon("Hcsdan", "old-english")
        loser_b = db.upsert_etymon("Haedan", "old-english")
        # Distinct etymon should be left alone.
        keeper = db.upsert_etymon("ham", "old-english")

        # Add some citations and glosses to losers — must transfer.
        db.add_gloss(loser_a, "personal name")
        db.add_gloss(loser_b, "from heathland (per Mawer)")
        db.add_tag(loser_a, "male name")
        db.add_citation(loser_a, "src-a", page="47")
        db.add_citation(loser_b, "src-b", page="23")
        db.add_citation(winner, "src-a", page="10")
        db.commit()

        result = cluster_ocr_variants(db, apply=True)

        # Verify the merge.
        rows = db.conn.execute("SELECT id, canonical_form FROM etymon ORDER BY id").fetchall()
        ids = {r["canonical_form"]: r["id"] for r in rows}

    assert result["groups"] == 1
    assert result["etymons_merged"] == 2
    # The losers are gone.
    assert "Hcsdan" not in ids
    assert "Haedan" not in ids
    # Winner and unrelated etymon survive.
    assert ids["Hædan"] == winner
    assert ids["ham"] == keeper

    # Citations from losers are now on winner (winner's existing citation
    # plus the two from losers; same source_id+page is dedup'd via INSERT
    # OR IGNORE).
    with LexiconDB(fresh_db) as db2:
        cit_count = db2.conn.execute(
            "SELECT COUNT(*) FROM etymon_citation WHERE etymon_id = ?",
            (winner,),
        ).fetchone()[0]
        gloss_count = db2.conn.execute(
            "SELECT COUNT(*) FROM etymon_gloss WHERE etymon_id = ?",
            (winner,),
        ).fetchone()[0]
        tag_count = db2.conn.execute(
            "SELECT COUNT(*) FROM etymon_tag WHERE etymon_id = ?",
            (winner,),
        ).fetchone()[0]
    # 3 distinct (source, page) citations: src-a/47, src-b/23, src-a/10
    assert cit_count == 3
    # 2 distinct glosses migrated from losers
    assert gloss_count == 2
    # 1 tag migrated
    assert tag_count == 1


def test_cluster_ocr_variants_repoints_lemma_children_and_text_matches(
    fresh_db: Path,
) -> None:
    """When the loser is itself a lemma with inflected children, those
    children's lemma_id must be re-parented to the winner before the loser
    is deleted; otherwise the DELETE fails on the etymon.lemma_id self-FK.

    Same for etymon_text_match: ON DELETE CASCADE would silently drop
    rows that should follow the merge to the winner.
    """
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src-a", title="A")
        db.commit()

        # Two OCR variants of the same lemma. Lowest id wins.
        winner = db.upsert_etymon("Hædan", "old-english")
        loser = db.upsert_etymon("Hcsdan", "old-english")

        # An inflected child points at the loser via lemma_id (this is the
        # state link-lemmas leaves behind when it picks the loser as the
        # canonical lemma for an inflected variant).
        child = db.upsert_etymon("Hædanes", "old-english")
        db.conn.execute(
            "UPDATE etymon SET lemma_id = ?, inflection = ? WHERE id = ?",
            (loser, "genitive_strong", child),
        )

        # Text-match rows on both winner and loser, with one (matched_form,
        # source) pair colliding so we exercise the UNIQUE constraint.
        db.conn.execute(
            "INSERT INTO etymon_text_match "
            "(etymon_id, source_id, matched_form, match_count, edit_distance, snippet) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (winner, "src-a", "haedan", 3, 0, "winner snippet"),
        )
        db.conn.execute(
            "INSERT INTO etymon_text_match "
            "(etymon_id, source_id, matched_form, match_count, edit_distance, snippet) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (loser, "src-a", "haedan", 5, 0, "loser snippet"),
        )
        db.conn.execute(
            "INSERT INTO etymon_text_match "
            "(etymon_id, source_id, matched_form, match_count, edit_distance, snippet) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (loser, "src-a", "hcsdan", 7, 0, "loser-only form"),
        )
        db.commit()

        # Without the FK repoints this raises sqlite3.IntegrityError.
        cluster_ocr_variants(db, apply=True)

    with LexiconDB(fresh_db) as db2:
        # (a) Loser is gone, child now points at winner.
        loser_row = db2.conn.execute(
            "SELECT id FROM etymon WHERE canonical_form = ?", ("Hcsdan",)
        ).fetchone()
        assert loser_row is None
        child_lemma = db2.conn.execute(
            "SELECT lemma_id FROM etymon WHERE canonical_form = ?", ("Hædanes",)
        ).fetchone()[0]
        assert child_lemma == winner

        # (b) Text-match rows survive the merge attached to the winner.
        rows = db2.conn.execute(
            "SELECT matched_form, source_id FROM etymon_text_match "
            "WHERE etymon_id = ? ORDER BY matched_form",
            (winner,),
        ).fetchall()
        assert [(r["matched_form"], r["source_id"]) for r in rows] == [
            ("haedan", "src-a"),
            ("hcsdan", "src-a"),
        ]

        # (c) UNIQUE (etymon_id, source_id, matched_form) is intact: only
        # one ('haedan', 'src-a') row, with the winner's original count
        # preserved (INSERT OR IGNORE drops the loser's collision).
        haedan = db2.conn.execute(
            "SELECT match_count FROM etymon_text_match "
            "WHERE etymon_id = ? AND matched_form = 'haedan'",
            (winner,),
        ).fetchall()
        assert len(haedan) == 1
        assert haedan[0]["match_count"] == 3


def test_derive_lemma_candidate_strips_oe_inflections() -> None:
    """Standard OE genitive/dative endings should suggest the stem."""
    # Weak oblique (-an): cotan → cot
    assert derive_lemma_candidate("cotan", "old-english") == ("cot", "weak_oblique")
    # Strong genitive (-es): ceorles → ceorl
    assert derive_lemma_candidate("ceorles", "old-english") == ("ceorl", "genitive_strong")
    # Dative plural (-um): cotum → cot
    assert derive_lemma_candidate("cotum", "old-english") == ("cot", "dative_pl")
    # Strong plural (-as): cotas → cot? Actually with the rule order, "as"
    # matches -as before -an or -e.
    assert derive_lemma_candidate("cotas", "old-english") == ("cot", "plural_strong")


def test_derive_lemma_candidate_strips_on_inflections() -> None:
    """Old Norse: nominative -r, genitive/plural -ar."""
    assert derive_lemma_candidate("klakkr", "old-norse") == ("klakk", "nominative")
    assert derive_lemma_candidate("dalir", "old-norse") == ("dal", "plural")


def test_derive_lemma_candidate_returns_none_for_unknown_lang() -> None:
    """Languages without rules don't produce candidates."""
    assert derive_lemma_candidate("foobar", "klingon") is None


def test_derive_lemma_candidate_respects_min_stem_length() -> None:
    """Refuse to produce a 2-char-or-less stem."""
    # "an" stripped from "ban" would give a 1-char stem; reject.
    assert derive_lemma_candidate("ban", "old-english") is None


def test_migrate_schema_idempotent_on_fresh_db(fresh_db: Path) -> None:
    """A fresh DB created by init_schema already has lemma_id; migration
    should be a no-op (rebuilding view aside)."""
    with LexiconDB(fresh_db) as db:
        first = migrate_schema(db)
        second = migrate_schema(db)
    # Schema bumps should not re-apply.
    assert not first["etymon.lemma_id"]
    assert not first["etymon.inflection"]
    assert not second["etymon.lemma_id"]


def test_link_lemmas_links_inflected_to_existing_lemma(fresh_db: Path) -> None:
    """An inflected etymon with a matching lemma in the DB gets linked."""
    with LexiconDB(fresh_db) as db:
        cot_id = db.upsert_etymon("cot", "old-english")
        cotan_id = db.upsert_etymon("cotan", "old-english")
        cotes_id = db.upsert_etymon("cotes", "old-english")
        # Distinct etymon: should NOT link to anything
        ham_id = db.upsert_etymon("ham", "old-english")
        db.commit()

        result = link_lemmas(db, apply=True)
        rows = db.conn.execute(
            "SELECT id, canonical_form, lemma_id, inflection FROM etymon"
        ).fetchall()

    by_id = {r["id"]: r for r in rows}
    assert by_id[cotan_id]["lemma_id"] == cot_id
    assert by_id[cotan_id]["inflection"] == "weak_oblique"
    assert by_id[cotes_id]["lemma_id"] == cot_id
    assert by_id[cotes_id]["inflection"] == "genitive_strong"
    # Lemma stays its own
    assert by_id[cot_id]["lemma_id"] is None
    # Unrelated etymon stays unlinked
    assert by_id[ham_id]["lemma_id"] is None
    assert result["candidates"] >= 2


def test_link_lemmas_dry_run_does_not_write(fresh_db: Path) -> None:
    with LexiconDB(fresh_db) as db:
        db.upsert_etymon("cot", "old-english")
        cotan_id = db.upsert_etymon("cotan", "old-english")
        db.commit()

        result = link_lemmas(db, apply=False)
        row = db.conn.execute("SELECT lemma_id FROM etymon WHERE id = ?", (cotan_id,)).fetchone()
    assert result["candidates"] == 1
    assert row["lemma_id"] is None


def test_link_lemmas_no_match_when_lemma_absent(fresh_db: Path) -> None:
    """If the proposed lemma isn't in the DB, no linkage proposal is made
    (we never fabricate lemmas in this pass)."""
    with LexiconDB(fresh_db) as db:
        # Only the inflected form exists; no "cot" lemma.
        db.upsert_etymon("cotan", "old-english")
        db.commit()
        result = link_lemmas(db, apply=False)
    assert result["candidates"] == 0


def test_consensus_view_rolls_up_to_lemma(fresh_db: Path) -> None:
    """Two sources, each citing a different inflected form of the same
    lemma, should produce a single consensus row with witnesses=2."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src-a", title="A")
        db.upsert_source(id="src-b", title="B")
        db.upsert_etymon("cot", "old-english")  # the lemma
        cotan_id = db.upsert_etymon("cotan", "old-english")
        cotes_id = db.upsert_etymon("cotes", "old-english")
        # Each inflected form has a different source citation.
        db.add_citation(cotan_id, "src-a")
        db.add_citation(cotes_id, "src-b")
        # Lemma itself has no direct citation.
        db.commit()

        # Before linking: each etymon is its own group; cotan has 1 witness,
        # cotes has 1 witness, cot has 0. Three rows.
        before = db.conn.execute(
            "SELECT canonical_form, witnesses FROM etymon_consensus ORDER BY canonical_form"
        ).fetchall()
        before_map = {r["canonical_form"]: r["witnesses"] for r in before}
        assert before_map.get("cotan") == 1
        assert before_map.get("cotes") == 1

        link_lemmas(db, apply=True)

        # After linking: one consensus row for the lemma 'cot' with
        # witnesses=2 (src-a + src-b unioned).
        after = db.conn.execute(
            "SELECT canonical_form, witnesses FROM etymon_consensus ORDER BY canonical_form"
        ).fetchall()
        after_map = {r["canonical_form"]: r["witnesses"] for r in after}

    assert after_map["cot"] == 2
    # cotan/cotes rows are gone from the consensus view (rolled into cot)
    assert "cotan" not in after_map
    assert "cotes" not in after_map


def test_language_field_mapping_covers_known_codes() -> None:
    """LANGUAGE_FIELDS must cover every source-language field used in
    meanings.json. Catches regressions if a new root is added in JSON but the
    seeder is forgotten."""
    text = resources.files("wyrd.generators.kenning.data").joinpath("meanings.json").read_text()
    data = json.loads(text)

    seen_fields: set[str] = set()
    for subject in data:
        for word in subject.get("words", []):
            seen_fields.update(word.keys())

    handled = LANGUAGE_FIELDS.keys() | NON_LANGUAGE_FIELDS
    missing = seen_fields - handled
    assert not missing, f"Unhandled fields in meanings.json: {missing}"
