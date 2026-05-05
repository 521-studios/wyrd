"""Tests for the kenning lexicon authoring DB."""

from __future__ import annotations

import json
import sqlite3
from importlib import resources
from pathlib import Path

import pytest
from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli as kenning_cli
from wyrd.generators.kenning.lexicon import (
    LANGUAGE_FIELDS,
    NON_LANGUAGE_FIELDS,
    RECOMMENDED_LANG_THRESHOLDS,
    LexiconDB,
    _normalize_for_quote_match,
    _position_from_usage,
    _quote_body_excerpt,
    assign_etymon_to_meaning_synset,
    backfill_citation_pages,
    clear_enrichment,
    cluster_ocr_variants,
    derive_lemma_candidate,
    detect_running_headers,
    export_meanings,
    fuzzy_search_attestations,
    get_meaning_preserving_candidates,
    get_meaning_synsets_for_etymon,
    ingest_parsed_entries,
    init_schema,
    link_lemmas,
    list_meaning_synsets,
    lookup_attested_years,
    migrate_schema,
    normalize_ocr_form,
    parse_skeat_section_header_pages,
    record_mining_run,
    seed_from_meanings,
    seed_meaning_synsets,
)
from wyrd.generators.kenning.meaning import load_meanings
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


def test_lexicon_db_opens_in_wal_mode(fresh_db: Path) -> None:
    """LexiconDB.__init__ sets journal_mode=WAL so readers and a single
    writer don't block each other. journal_mode is persistent on the
    file — pin via PRAGMA query so a regression that drops the setting
    surfaces immediately.

    Multi-session workflow context: one Claude can be mining (writer)
    while another queries corpus state (reader). Without WAL, the
    reader either blocks or sees a stale snapshot. With WAL, the reader
    sees the last committed snapshot and proceeds without contending."""
    with LexiconDB(fresh_db) as db:
        mode = db.conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_lexicon_db_wal_mode_persists_across_reopen(fresh_db: Path) -> None:
    """SQLite stores journal_mode in the file header, so once WAL is
    set, every subsequent open inherits it — even one that doesn't
    re-run the PRAGMA. Pin this so the docstring's persistence claim
    is load-bearing in tests, not implied empirically.

    Open #1 sets WAL via LexiconDB.__init__. Open #2 uses raw
    sqlite3.connect (no PRAGMAs) and confirms the mode survived."""
    with LexiconDB(fresh_db) as db:
        assert db.conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"

    raw = sqlite3.connect(fresh_db)
    try:
        mode = raw.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        raw.close()
    assert mode.lower() == "wal"


def test_lexicon_db_uses_synchronous_normal(fresh_db: Path) -> None:
    """Pairs with WAL: synchronous=NORMAL is the recommended setting
    under WAL — preserves crash safety (WAL replays on next open) while
    skipping the per-transaction fsync that synchronous=FULL imposes.
    Pin so a regression flipping it back to FULL (the SQLite default)
    surfaces in the slowdown rather than going silent."""
    with LexiconDB(fresh_db) as db:
        # PRAGMA returns 0=OFF, 1=NORMAL, 2=FULL, 3=EXTRA.
        sync = db.conn.execute("PRAGMA synchronous").fetchone()[0]
    assert sync == 1, f"expected synchronous=NORMAL (1), got {sync}"


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


def test_cluster_ocr_variants_apply_marks_losers_merged_non_destructively(
    fresh_db: Path,
) -> None:
    """With apply=True, redundant etymons are MARKED merged into the
    lowest-id row via merged_into_id. Losers stay in the etymon table
    (per D22, mining evidence is non-destructive); their citations,
    glosses, and tags remain attached to them. The etymon_canonical view
    hides them; etymon_consensus rolls them up via merged_into_id."""
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

        db.add_gloss(loser_a, "personal name")
        db.add_gloss(loser_b, "from heathland (per Mawer)")
        db.add_tag(loser_a, "male name")
        db.add_citation(loser_a, "src-a", page="47")
        db.add_citation(loser_b, "src-b", page="23")
        db.add_citation(winner, "src-a", page="10")
        db.commit()

        result = cluster_ocr_variants(db, apply=True)

    assert result["groups"] == 1
    assert result["etymons_merged"] == 2

    with LexiconDB(fresh_db) as db2:
        # Losers still exist in the raw table.
        rows = db2.conn.execute(
            "SELECT id, canonical_form, merged_into_id FROM etymon ORDER BY id"
        ).fetchall()
        by_form = {r["canonical_form"]: r for r in rows}
        assert "Hcsdan" in by_form
        assert "Haedan" in by_form
        assert "Hædan" in by_form
        assert "ham" in by_form

        # Losers point at the winner; winner and keeper are not merged.
        assert by_form["Hcsdan"]["merged_into_id"] == winner
        assert by_form["Haedan"]["merged_into_id"] == winner
        assert by_form["Hædan"]["merged_into_id"] is None
        assert by_form["ham"]["merged_into_id"] is None

        # etymon_canonical view excludes the merged losers.
        canonical_forms = {
            r["canonical_form"]
            for r in db2.conn.execute("SELECT canonical_form FROM etymon_canonical")
        }
        assert canonical_forms == {"Hædan", "ham"}

        # Citations / glosses / tags stay on losers — not moved (D22).
        loser_a_cit_count = db2.conn.execute(
            "SELECT COUNT(*) FROM etymon_citation WHERE etymon_id = ?", (loser_a,)
        ).fetchone()[0]
        winner_cit_count = db2.conn.execute(
            "SELECT COUNT(*) FROM etymon_citation WHERE etymon_id = ?", (winner,)
        ).fetchone()[0]
        assert loser_a_cit_count == 1
        assert winner_cit_count == 1

        loser_a_gloss_count = db2.conn.execute(
            "SELECT COUNT(*) FROM etymon_gloss WHERE etymon_id = ?", (loser_a,)
        ).fetchone()[0]
        assert loser_a_gloss_count == 1

        # etymon_consensus rolls them up correctly: the Hædan canonical
        # group sees all three sources (src-a from winner, src-a from
        # loser_a's page=47, src-b from loser_b).
        consensus = db2.conn.execute(
            "SELECT witnesses FROM etymon_consensus WHERE canonical_form = ? AND language = ?",
            ("Hædan", "old-english"),
        ).fetchone()
        # src-a (winner+loser_a counted once across rollup) and src-b → 2 distinct
        assert consensus["witnesses"] == 2

    # Unused name reference — keeper's id used for differentiation.
    assert keeper != winner


def test_cluster_ocr_variants_keeps_lemma_children_consistent_via_redirect(
    fresh_db: Path,
) -> None:
    """When the loser is itself a lemma with inflected children, those
    children's lemma_id must be re-parented to the canonical so the
    single-level consensus rollup still works. Text-match rows stay
    attached to the loser etymon (D22 — non-destructive) and are
    rolled up via merged_into_id."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src-a", title="A")
        db.commit()

        winner = db.upsert_etymon("Hædan", "old-english")
        loser = db.upsert_etymon("Hcsdan", "old-english")

        # Inflected child of the loser.
        child = db.upsert_etymon("Hædanes", "old-english")
        db.conn.execute(
            "UPDATE etymon SET lemma_id = ?, inflection = ? WHERE id = ?",
            (loser, "genitive_strong", child),
        )

        # Text-match rows on both winner and loser. Stay where they are.
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
            (loser, "src-a", "hcsdan", 7, 0, "loser-only form"),
        )
        db.commit()

        cluster_ocr_variants(db, apply=True)

    with LexiconDB(fresh_db) as db2:
        # Loser still exists, marked merged.
        loser_row = db2.conn.execute(
            "SELECT id, merged_into_id FROM etymon WHERE canonical_form = ?",
            ("Hcsdan",),
        ).fetchone()
        assert loser_row is not None
        assert loser_row["merged_into_id"] == winner

        # Child re-parented to winner so consensus rolls up cleanly.
        child_lemma = db2.conn.execute(
            "SELECT lemma_id FROM etymon WHERE canonical_form = ?", ("Hædanes",)
        ).fetchone()[0]
        assert child_lemma == winner

        # Text-match rows stay on their original etymons. Loser's row is
        # NOT moved to winner — the rollup happens via merged_into_id at
        # query time, not by data movement.
        rows_on_loser = db2.conn.execute(
            "SELECT matched_form FROM etymon_text_match WHERE etymon_id = ?",
            (loser,),
        ).fetchall()
        rows_on_winner = db2.conn.execute(
            "SELECT matched_form FROM etymon_text_match WHERE etymon_id = ?",
            (winner,),
        ).fetchall()
        assert [r["matched_form"] for r in rows_on_loser] == ["hcsdan"]
        assert [r["matched_form"] for r in rows_on_winner] == ["haedan"]


def test_cluster_ocr_variants_flattens_lemma_chain_when_winner_is_child(
    fresh_db: Path,
) -> None:
    """If the winner is itself an inflected variant of some canonical lemma,
    the loser's merged_into_id and the loser's children's lemma_id must
    both point directly at the canonical lemma — not at the winner.
    Otherwise the single-level consensus rollup would split witnesses.
    """
    with LexiconDB(fresh_db) as db:
        canonical = db.upsert_etymon("had", "old-english")
        winner = db.upsert_etymon("Hædan", "old-english")
        loser = db.upsert_etymon("Hcsdan", "old-english")
        db.conn.execute("UPDATE etymon SET lemma_id = ? WHERE id = ?", (canonical, winner))
        child = db.upsert_etymon("Hcsdanes", "old-english")
        db.conn.execute("UPDATE etymon SET lemma_id = ? WHERE id = ?", (loser, child))
        db.commit()

        cluster_ocr_variants(db, apply=True)

    with LexiconDB(fresh_db) as db2:
        # The loser's redirect goes to canonical, not to winner.
        loser_merged = db2.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE canonical_form = ?", ("Hcsdan",)
        ).fetchone()[0]
        assert loser_merged == canonical

        # The child also points at canonical directly.
        child_lemma = db2.conn.execute(
            "SELECT lemma_id FROM etymon WHERE canonical_form = ?", ("Hcsdanes",)
        ).fetchone()[0]
        assert child_lemma == canonical

        # Winner's own pointer is unchanged.
        winner_lemma = db2.conn.execute(
            "SELECT lemma_id FROM etymon WHERE canonical_form = ?", ("Hædan",)
        ).fetchone()[0]
        assert winner_lemma == canonical


def test_cluster_ocr_variants_skips_already_merged_on_rerun(fresh_db: Path) -> None:
    """Re-running cluster_ocr_variants on a DB that already has merges
    must NOT re-cluster the tombstoned losers. The candidate query
    filters merged_into_id IS NULL."""
    with LexiconDB(fresh_db) as db:
        db.upsert_etymon("Hædan", "old-english")
        db.upsert_etymon("Hcsdan", "old-english")
        db.upsert_etymon("Haedan", "old-english")
        db.commit()

        first = cluster_ocr_variants(db, apply=True)
        assert first["groups"] == 1
        assert first["etymons_merged"] == 2

        # Second run: nothing left to cluster.
        second = cluster_ocr_variants(db, apply=True)
        assert second["groups"] == 0
        assert second["etymons_merged"] == 0


def test_cluster_ocr_variants_picks_lemma_as_winner_when_present(
    fresh_db: Path,
) -> None:
    """If a cluster contains both a lemma and an inflected variant of it,
    the lemma must win regardless of id ordering. Otherwise canonical_id
    resolves to the lemma but the lemma is in loser_ids, creating a
    self-referencing merged_into_id loop."""
    with LexiconDB(fresh_db) as db:
        # The inflected variant gets the LOWER id (would normally win
        # under lowest-id-wins). The lemma comes second.
        inflected = db.upsert_etymon("Hædan", "old-english")
        lemma = db.upsert_etymon("Hædan_lemma_synthetic", "old-english")
        # Manually fix canonical_form so they share an OCR fingerprint
        # (the synthetic suffix above kept upsert_etymon's UNIQUE happy).
        db.conn.execute("UPDATE etymon SET canonical_form = 'Haedan' WHERE id = ?", (lemma,))
        db.conn.execute("UPDATE etymon SET lemma_id = ? WHERE id = ?", (lemma, inflected))
        db.commit()

        cluster_ocr_variants(db, apply=True)

    with LexiconDB(fresh_db) as db2:
        # Lemma wins. Inflected variant is the loser, redirected to the lemma.
        assert (
            db2.conn.execute("SELECT merged_into_id FROM etymon WHERE id = ?", (lemma,)).fetchone()[
                0
            ]
            is None
        )
        assert (
            db2.conn.execute(
                "SELECT merged_into_id FROM etymon WHERE id = ?", (inflected,)
            ).fetchone()[0]
            == lemma
        )


def test_cluster_ocr_variants_follows_merged_chain_to_canonical(
    fresh_db: Path,
) -> None:
    """If winner.lemma_id points at a row that was already merged in a
    prior pass (e.g., from a stale link-lemmas pointer), cluster must
    follow the merged_into_id chain to the ultimate canonical — not
    plant losers' merged_into_id on a tombstone (which would create a
    depth-2 chain that the rollup splits)."""
    with LexiconDB(fresh_db) as db:
        # L is the canonical lemma. L2 is its OCR-merge tombstone.
        canonical = db.upsert_etymon("had", "old-english")
        tombstone = db.upsert_etymon("hadx_ocr", "old-english")
        db.conn.execute(
            "UPDATE etymon SET merged_into_id = ? WHERE id = ?",
            (canonical, tombstone),
        )

        # Winner W has a stale pointer at the tombstone (simulates a
        # link-lemmas pass that ran before the tombstone-filter fix).
        winner = db.upsert_etymon("Hædan", "old-english")
        db.conn.execute("UPDATE etymon SET lemma_id = ? WHERE id = ?", (tombstone, winner))

        # Loser X joins the cluster with W on OCR fingerprint.
        loser = db.upsert_etymon("Hcsdan", "old-english")
        db.commit()

        cluster_ocr_variants(db, apply=True)

    with LexiconDB(fresh_db) as db2:
        # Loser must redirect to the ultimate canonical, not the tombstone.
        loser_redirect = db2.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (loser,)
        ).fetchone()[0]
        assert loser_redirect == canonical, (
            f"loser should chain-follow through tombstone to canonical; got "
            f"merged_into_id={loser_redirect}"
        )


def test_link_lemmas_skips_merged_tombstone_as_lemma_candidate(
    fresh_db: Path,
) -> None:
    """An OCR-merge tombstone (merged_into_id IS NOT NULL) must not be
    selected as a lemma by link_lemmas — pointing an inflected variant at
    a tombstone would create a child → tombstone → canonical chain that
    the rollup splits."""
    with LexiconDB(fresh_db) as db:
        # `cot` (canonical lemma) and `cotx` (its OCR-merged tombstone
        # — synthetic but the principle generalizes). Set up the merge
        # state directly so we don't depend on cluster_ocr_variants.
        cot = db.upsert_etymon("cot", "old-english")
        cotx = db.upsert_etymon("cotx", "old-english")
        db.conn.execute("UPDATE etymon SET merged_into_id = ? WHERE id = ?", (cot, cotx))
        # An inflected variant that the stripper might match against
        # 'cotx' (e.g. 'cotxes' → 'cotx'). link_lemmas must skip cotx
        # because it's merged, even though it has lemma_id IS NULL.
        cotxes = db.upsert_etymon("cotxes", "old-english")
        db.commit()

        link_lemmas(db, apply=True)

        cotxes_lemma = db.conn.execute(
            "SELECT lemma_id FROM etymon WHERE id = ?", (cotxes,)
        ).fetchone()[0]
        # Either matched directly to canonical 'cot' (if the stripper
        # reaches it) OR stayed unlinked (if no stripped form matches a
        # non-merged candidate). Critically: NOT linked to cotx.
        assert cotxes_lemma != cotx


def test_cluster_ocr_variants_flattens_existing_redirects_to_loser(
    fresh_db: Path,
) -> None:
    """If any row already has merged_into_id pointing at the current
    loser (from a prior cluster pass, manual fixup, etc.), marking the
    loser merged into a new canonical must re-route those rows too —
    otherwise X → loser → canonical chains form and the rollup splits."""
    with LexiconDB(fresh_db) as db:
        a = db.upsert_etymon("Hædan", "old-english")
        b = db.upsert_etymon("Hcsdan", "old-english")
        # X has a separate OCR signature so the candidate query would
        # group {a, b} alone. We set X.merged_into_id = b manually to
        # simulate a residual pointer (e.g. from a manual fixup or a
        # prior clustering pass that touched only some rows).
        x = db.upsert_etymon("Hbcdan", "old-english")
        db.conn.execute("UPDATE etymon SET merged_into_id = ? WHERE id = ?", (b, x))
        db.commit()

        cluster_ocr_variants(db, apply=True)

    with LexiconDB(fresh_db) as db2:
        # b's merge target is a (direct).
        assert (
            db2.conn.execute("SELECT merged_into_id FROM etymon WHERE id = ?", (b,)).fetchone()[0]
            == a
        )
        # x's redirect was flattened from b to a so no chain forms.
        assert (
            db2.conn.execute("SELECT merged_into_id FROM etymon WHERE id = ?", (x,)).fetchone()[0]
            == a
        )


def test_etymon_consensus_handles_depth_2_merged_into_lemma_chain(
    fresh_db: Path,
) -> None:
    """The consensus view must handle merged_into_id → lemma_id chains
    correctly: an OCR variant of an inflected form must roll up to the
    lemma group, not stop at the inflected form. This can arise if
    clustering runs before link-lemmas (against doc'd order) or if
    link-lemmas runs after clustering."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src-a", title="A")
        db.upsert_source(id="src-b", title="B")
        db.commit()

        # Lemma + inflected variant + OCR variant of the inflected form.
        lemma = db.upsert_etymon("had", "old-english")
        inflected = db.upsert_etymon("hædan", "old-english")
        ocr_variant = db.upsert_etymon("hcsdan", "old-english")

        # Set up the chain manually: ocr_variant.merged_into_id =
        # inflected; inflected.lemma_id = lemma. (cluster_ocr_variants
        # would normally flatten through the lemma at merge time, but
        # this test exercises the case where link-lemmas runs AFTER
        # clustering — which is reverse of the documented order — to
        # prove the view is robust to it.)
        db.conn.execute(
            "UPDATE etymon SET merged_into_id = ? WHERE id = ?",
            (inflected, ocr_variant),
        )
        db.conn.execute("UPDATE etymon SET lemma_id = ? WHERE id = ?", (lemma, inflected))

        db.add_citation(ocr_variant, "src-a")
        db.add_citation(inflected, "src-b")
        db.commit()

        # The lemma group must aggregate citations from all three rows
        # (ocr_variant via merged_into_id → inflected → lemma; inflected
        # via lemma_id → lemma; lemma itself).
        consensus = db.conn.execute(
            "SELECT lemma_id, witnesses FROM etymon_consensus WHERE language = 'old-english'"
        ).fetchall()
    assert len(consensus) == 1, (
        f"Expected exactly one consensus group (the lemma); got {len(consensus)}"
    )
    assert consensus[0]["lemma_id"] == lemma
    assert consensus[0]["witnesses"] == 2  # src-a + src-b


def test_clear_enrichment_dry_run_reports_without_writing(fresh_db: Path) -> None:
    """Dry-run reports counts but does not modify the DB."""
    with LexiconDB(fresh_db) as db:
        winner = db.upsert_etymon("Hædan", "old-english")
        db.upsert_etymon("Hcsdan", "old-english")
        db.commit()
        cluster_ocr_variants(db, apply=True)

        before = db.conn.execute(
            "SELECT COUNT(*) FROM etymon WHERE merged_into_id IS NOT NULL"
        ).fetchone()[0]
        assert before == 1

        result = clear_enrichment(db, stage="ocr", apply=False)
        assert result["ocr_merges_to_clear"] == 1

        after = db.conn.execute(
            "SELECT COUNT(*) FROM etymon WHERE merged_into_id IS NOT NULL"
        ).fetchone()[0]
        assert after == 1
    # Use winner reference for differentiation.
    assert winner is not None


def test_clear_enrichment_ocr_reverts_merges(fresh_db: Path) -> None:
    """clear_enrichment(stage='ocr', apply=True) un-marks all merges,
    restoring the etymon_canonical view to include the previously-tombstoned
    rows. Citations/glosses/tags are intact since cluster_ocr_variants
    didn't move them."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src-a", title="A")
        db.commit()
        winner = db.upsert_etymon("Hædan", "old-english")
        loser = db.upsert_etymon("Hcsdan", "old-english")
        db.add_citation(loser, "src-a", page="47")
        db.commit()
        cluster_ocr_variants(db, apply=True)

        # Before clear: etymon_canonical excludes loser.
        before = {
            r["canonical_form"]
            for r in db.conn.execute("SELECT canonical_form FROM etymon_canonical")
        }
        assert before == {"Hædan"}

        result = clear_enrichment(db, stage="ocr", apply=True)
        assert result["ocr_merges_to_clear"] == 1

        # After clear: loser is canonical again, citation untouched.
        after = {
            r["canonical_form"]
            for r in db.conn.execute("SELECT canonical_form FROM etymon_canonical")
        }
        assert after == {"Hædan", "Hcsdan"}
        loser_cit = db.conn.execute(
            "SELECT page FROM etymon_citation WHERE etymon_id = ?", (loser,)
        ).fetchone()
        assert loser_cit["page"] == "47"
    assert winner is not None


def test_clear_enrichment_lemmas_resets_lemma_id_inflection_method(
    fresh_db: Path,
) -> None:
    """clear_enrichment(stage='lemmas') nulls out lemma_id, inflection,
    and lemma_method on every row."""
    with LexiconDB(fresh_db) as db:
        lemma = db.upsert_etymon("cot", "old-english")
        inflected = db.upsert_etymon("cotan", "old-english")
        db.conn.execute(
            "UPDATE etymon SET lemma_id = ?, inflection = 'weak_oblique', "
            "lemma_method = 'link-lemmas-v1' WHERE id = ?",
            (lemma, inflected),
        )
        db.commit()

        result = clear_enrichment(db, stage="lemmas", apply=True)
        assert result["lemma_links_to_clear"] == 1

        row = db.conn.execute(
            "SELECT lemma_id, inflection, lemma_method FROM etymon WHERE id = ?",
            (inflected,),
        ).fetchone()
        assert row["lemma_id"] is None
        assert row["inflection"] is None
        assert row["lemma_method"] is None


def test_clear_enrichment_text_match_drops_all_rows(fresh_db: Path) -> None:
    """clear_enrichment(stage='text-match') deletes every text-match row
    so a re-run of reverse-search/fuzzy-search starts fresh."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src-a", title="A")
        eid = db.upsert_etymon("denu", "old-english")
        db.commit()
        db.conn.execute(
            "INSERT INTO etymon_text_match (etymon_id, source_id, matched_form, "
            "match_count, edit_distance, snippet) VALUES (?, ?, ?, ?, ?, ?)",
            (eid, "src-a", "denu", 12, 0, "snippet"),
        )
        db.commit()

        result = clear_enrichment(db, stage="text-match", apply=True)
        assert result["text_match_rows_to_clear"] == 1

        remaining = db.conn.execute("SELECT COUNT(*) FROM etymon_text_match").fetchone()[0]
        assert remaining == 0


def test_clear_enrichment_all_derived_resets_three_stages(fresh_db: Path) -> None:
    """stage='all-derived' clears OCR merges, lemma links, AND text-match
    rows in a single invocation. (Cognate + attested-year stages are
    exercised in their own tests; the three-stage shape is the legacy
    one and stays pinned here.)"""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src-a", title="A")
        winner = db.upsert_etymon("Hædan", "old-english")
        db.upsert_etymon("Hcsdan", "old-english")
        lemma = db.upsert_etymon("cot", "old-english")
        inflected = db.upsert_etymon("cotan", "old-english")
        db.commit()
        cluster_ocr_variants(db, apply=True)
        db.conn.execute("UPDATE etymon SET lemma_id = ? WHERE id = ?", (lemma, inflected))
        db.conn.execute(
            "INSERT INTO etymon_text_match (etymon_id, source_id, matched_form, "
            "match_count, edit_distance, snippet) VALUES (?, ?, ?, ?, ?, ?)",
            (winner, "src-a", "haedan", 5, 0, "s"),
        )
        db.commit()

        result = clear_enrichment(db, stage="all-derived", apply=True)
        assert result["ocr_merges_to_clear"] == 1
        assert result["lemma_links_to_clear"] == 1
        assert result["text_match_rows_to_clear"] == 1

        leftover = db.conn.execute(
            "SELECT "
            "  (SELECT COUNT(*) FROM etymon WHERE merged_into_id IS NOT NULL),"
            "  (SELECT COUNT(*) FROM etymon WHERE lemma_id IS NOT NULL),"
            "  (SELECT COUNT(*) FROM etymon_text_match)"
        ).fetchone()
        assert tuple(leftover) == (0, 0, 0)


def test_clear_enrichment_rejects_unknown_stage(fresh_db: Path) -> None:
    """Defensive: unknown stage raises ValueError before touching the DB."""
    with LexiconDB(fresh_db) as db, pytest.raises(ValueError, match="unknown stage"):
        clear_enrichment(db, stage="bogus", apply=True)


# --- D5-1 / wyrd-3ux: lookup_attested_years --------------------------------


def _seed_text_match(
    db: LexiconDB,
    *,
    source_id: str,
    canonical_form: str,
    matched_form: str | None = None,
    language: str = "old-english",
) -> int:
    """Insert one etymon + one text-match row pointing at it; return the
    text-match row id. Used as a fixture builder for the lookup-attested-
    years tests."""
    eid = db.upsert_etymon(canonical_form, language)
    matched = matched_form or canonical_form
    cur = db.conn.execute(
        "INSERT INTO etymon_text_match (etymon_id, source_id, matched_form, "
        "match_count, edit_distance, snippet) VALUES (?, ?, ?, ?, ?, ?)",
        (eid, source_id, matched, 1, 0, ""),
    )
    db.commit()
    return cur.lastrowid


def test_lookup_attested_years_finds_year_after_form_with_comma(
    fresh_db: Path, tmp_path: Path
) -> None:
    """The dominant scholarly toponym citation pattern: 'Tune, 1086' —
    matched form, comma, year. Year falls within the [100, 1700] range
    and the preceding context contains ', ' (a recognised marker), so
    it qualifies."""
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "mawer_1920.txt").write_text(
        "BACKWORTH. Backworth, 1240; Bakworth, 1242. From OE bæc and worð.\n"
        "TUNE. Tune, 1086 (DB); Tunes, 1242."
    )
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="mawer_1920", title="Mawer")
        rid = _seed_text_match(db, source_id="mawer_1920", canonical_form="tune")
        result = lookup_attested_years(db, sources, apply=True)

        year = db.conn.execute(
            "SELECT attested_year FROM etymon_text_match WHERE id = ?", (rid,)
        ).fetchone()["attested_year"]

    assert result["candidates"] == 1
    assert result["rows_written"] == 1
    assert year == 1086


def test_lookup_attested_years_picks_earliest_when_multiple_qualify(
    fresh_db: Path, tmp_path: Path
) -> None:
    """When a form is cited at multiple years (typical for well-attested
    toponyms), the earliest qualifying year wins. Matches the v1 design
    documented on lookup_attested_years."""
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "ekwall_1922.txt").write_text("TUNE. Tune, 1086 (DB); Tunes, 1242; Tunna, 1340.")
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="ekwall_1922", title="Ekwall")
        rid = _seed_text_match(db, source_id="ekwall_1922", canonical_form="tune")
        lookup_attested_years(db, sources, apply=True)
        year = db.conn.execute(
            "SELECT attested_year FROM etymon_text_match WHERE id = ?", (rid,)
        ).fetchone()["attested_year"]
    assert year == 1086


def test_lookup_attested_years_rejects_publication_year(fresh_db: Path, tmp_path: Path) -> None:
    """1920 is a publication year, not an attestation. _ATTESTED_YEAR_MAX_LOOKUP
    = 1700 cuts off well before publication years for the corpus, so
    nothing should be written."""
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "src.txt").write_text("denu (Mawer, 1920) appears throughout.")
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        rid = _seed_text_match(db, source_id="src", canonical_form="denu")
        lookup_attested_years(db, sources, apply=True)
        year = db.conn.execute(
            "SELECT attested_year FROM etymon_text_match WHERE id = ?", (rid,)
        ).fetchone()["attested_year"]
    assert year is None


def test_lookup_attested_years_handles_complex_citation_list(
    fresh_db: Path, tmp_path: Path
) -> None:
    """Real EPNS pattern: 'speldredge (c. 1290 sac 6, 134, 1320 ch)'
    has THREE digit runs, only one of which is a real year-citation.
    The form-attached pattern with the 'c.' marker correctly extracts
    1290 (year) and rejects 134 (section reference — preceded by ', '
    after a digit, not after alpha). 1320 is also a real year, but
    1290 wins on 'earliest'.

    Pin so a future loosening of the form-attached pattern doesn't
    reintroduce the digit-list FP class."""
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "src.txt").write_text("speldredge (c. 1290 sac 6, 134, 1320 ch).")
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        rid = _seed_text_match(db, source_id="src", canonical_form="speldredge")
        lookup_attested_years(db, sources, apply=True)
        year = db.conn.execute(
            "SELECT attested_year FROM etymon_text_match WHERE id = ?", (rid,)
        ).fetchone()["attested_year"]
    assert year == 1290


def test_lookup_attested_years_rejects_year_without_date_marker(
    fresh_db: Path, tmp_path: Path
) -> None:
    """A digit run not preceded by a recognised date-context marker is
    not a date citation. Filters page numbers, statute counts, etc."""
    sources = tmp_path / "sources"
    sources.mkdir()
    # 'See note 1240 below' has '1240' but no preceding marker — it's a
    # cross-reference number, not a date citation.
    (sources / "src.txt").write_text("hamtun see note 1240 below for the discussion.")
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        rid = _seed_text_match(db, source_id="src", canonical_form="hamtun")
        lookup_attested_years(db, sources, apply=True)
        year = db.conn.execute(
            "SELECT attested_year FROM etymon_text_match WHERE id = ?", (rid,)
        ).fetchone()["attested_year"]
    assert year is None


def test_lookup_attested_years_dry_run_does_not_write(fresh_db: Path, tmp_path: Path) -> None:
    """apply=False reports candidates but writes nothing — same shape as
    every other enrichment stage."""
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "src.txt").write_text("Tune, 1086 (DB).")
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        rid = _seed_text_match(db, source_id="src", canonical_form="tune")
        result = lookup_attested_years(db, sources, apply=False)

        year = db.conn.execute(
            "SELECT attested_year FROM etymon_text_match WHERE id = ?", (rid,)
        ).fetchone()["attested_year"]

    assert result["candidates"] == 1
    assert result["rows_written"] == 0
    assert result["applied"] is False
    assert year is None


def test_lookup_attested_years_idempotent_on_rerun(fresh_db: Path, tmp_path: Path) -> None:
    """Re-running on a DB that already has attested_year set is a no-op:
    only rows where attested_year IS NULL are scanned. Lets a future run
    pick up newly-added text-match rows without re-clobbering existing
    values."""
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "src.txt").write_text("Tune, 1086 (DB).")
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        _seed_text_match(db, source_id="src", canonical_form="tune")

        first = lookup_attested_years(db, sources, apply=True)
        assert first["rows_written"] == 1
        # Second run sees zero unscored rows.
        second = lookup_attested_years(db, sources, apply=True)

    assert second["rows_scanned"] == 0
    assert second["candidates"] == 0
    assert second["rows_written"] == 0


def test_lookup_attested_years_groups_rows_by_source_id(fresh_db: Path, tmp_path: Path) -> None:
    """The streaming restructure (7681838) groups text-match rows by
    source_id so each source body is loaded only once. Pin that the
    grouping is correct: rows from source A get scanned against A's
    body, NOT B's body, even when both have year-citations against
    the same matched_form value. A regression that flattens the
    grouping back to a single shared body would silently swap years
    between sources."""
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "src_a.txt").write_text("Tune, 1086 (DB).")
    (sources / "src_b.txt").write_text("Tune, 1242 (DB).")
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src_a", title="A")
        db.upsert_source(id="src_b", title="B")
        rid_a = _seed_text_match(db, source_id="src_a", canonical_form="tune")
        rid_b = _seed_text_match(db, source_id="src_b", canonical_form="tune")
        lookup_attested_years(db, sources, apply=True)
        years = dict(db.conn.execute("SELECT id, attested_year FROM etymon_text_match").fetchall())
    assert years[rid_a] == 1086
    assert years[rid_b] == 1242


def test_lookup_attested_years_warns_on_missing_source_file(fresh_db: Path, tmp_path: Path) -> None:
    """A text-match row pointing at a source_id whose .txt isn't in
    sources_dir is counted in sources_missing and skipped without
    raising. Common in real corpora where some books have been mined
    but the OCR text is gitignored / absent locally."""
    sources = tmp_path / "sources"
    sources.mkdir()  # Empty — no .txt files.
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="missing_src", title="M")
        _seed_text_match(db, source_id="missing_src", canonical_form="tune")
        result = lookup_attested_years(db, sources, apply=True)

    assert result["sources_missing"] == 1
    assert result["candidates"] == 0
    assert result["rows_written"] == 0


# --- D5-1 / wyrd-bag: toponym_etymology.notes scan -------------------------


def _seed_toponym_etymology(
    db: LexiconDB,
    *,
    notes: str,
    modern_name: str = "Tune",
    source_id: str = "src",
    historical_form: str = "Tūn",
) -> int:
    """Insert a toponym + toponym_etymology row carrying ``notes``.
    Returns the toponym_etymology row id. Used as a fixture builder
    for the wyrd-bag tests."""
    cur = db.conn.execute("INSERT INTO toponym (modern_name) VALUES (?)", (modern_name,))
    toponym_id = cur.lastrowid
    cur = db.conn.execute(
        "INSERT INTO toponym_etymology "
        "(toponym_id, source_id, historical_form, confidence, notes) "
        "VALUES (?, ?, ?, ?, ?)",
        (toponym_id, source_id, historical_form, "high", notes),
    )
    db.commit()
    return cur.lastrowid


def test_lookup_attested_years_populates_toponym_etymology_from_notes(
    fresh_db: Path, tmp_path: Path
) -> None:
    """wyrd-bag: notes column carries dense scholarly date strings
    ('1086 DB; 1242 IPM'). The post-mining stage picks the earliest
    plausibly-attested year (>=700) and writes it to attested_year.
    Pin the dominant case — multiple year citations, earliest wins."""
    sources = tmp_path / "sources"
    sources.mkdir()  # required by lookup_attested_years even when empty
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        rid = _seed_toponym_etymology(
            db,
            notes="extracted_by:llm:qwen | Tune (DB). 1086 DB; 1242 IPM; 1340 Cl",
        )
        result = lookup_attested_years(db, sources, apply=True)

        year = db.conn.execute(
            "SELECT attested_year FROM toponym_etymology WHERE id = ?", (rid,)
        ).fetchone()["attested_year"]

    assert result["toponym_etymology"]["candidates"] == 1
    assert result["toponym_etymology"]["rows_written"] == 1
    assert year == 1086


def test_lookup_attested_years_rejects_pre_700_years_in_notes(
    fresh_db: Path, tmp_path: Path
) -> None:
    """The simple year-range filter (>=700) rules out the dominant
    false-positive class: page numbers, volume numbers, footnote refs.
    Pin that '233' (volume) and '47' (page) don't get picked even when
    they precede the form. Earliest qualifying year is 1086."""
    sources = tmp_path / "sources"
    sources.mkdir()
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        rid = _seed_toponym_etymology(
            db,
            notes=("VI. 233 (close) — Tune. p. 47. Buketon c. 1086 DB; Tunes 1242 ; 1340 Cl"),
        )
        lookup_attested_years(db, sources, apply=True)

        year = db.conn.execute(
            "SELECT attested_year FROM toponym_etymology WHERE id = ?", (rid,)
        ).fetchone()["attested_year"]

    assert year == 1086


def test_earliest_year_in_notes_does_not_false_skip_p_inside_word(
    fresh_db: Path,
) -> None:
    """Regression for PR #53 Gemini-medium: the page-marker filter
    must require a word boundary before 'p.' / 'ch.' / etc. so a
    real word ending in 'p.' (or any short letter sequence + dot)
    doesn't false-skip a real year-citation. The substring shape
    'p.' appears inside 'chap.', 'pp.', etc. — those are themselves
    markers, so the FP risk is words like 'Bp.' (Bishop abbreviation)
    or stray letter+dot sequences.

    Pin: 'Bp 1086' must yield 1086, not None."""
    from wyrd.generators.kenning.lexicon import _earliest_year_in_notes

    # 'Bp.' (Bishop) — not a page marker. Year 1086 should be picked.
    assert _earliest_year_in_notes("Bp. 1086 DB.") == 1086
    # 'pp.' IS a page marker. Year 1086 should be skipped (no other
    # year present → None).
    assert _earliest_year_in_notes("pp. 1086 (book index)") is None
    # 'p.' alone IS a page marker. Year 755 should be skipped.
    assert _earliest_year_in_notes("p. 755") is None
    # 'p' without a trailing dot is NOT a page marker — it's just the
    # letter p ending some prior word. Year should be picked.
    assert _earliest_year_in_notes("group 1086") == 1086
    # Multiple spaces between marker and year: PR #53 round-3 Gemini
    # flagged that an 8-char window would miss "p.   755". Full-prefix
    # regex with $ anchor must still catch it.
    assert _earliest_year_in_notes("p.   755") is None
    assert _earliest_year_in_notes("vol.        1244") is None
    # Marker-preceded page-ref ≥700 BEFORE a real year in the same
    # note: the $ anchor confines marker matching to the immediate
    # predecessor of each year candidate, so a distant page ref
    # doesn't suppress the real date that follows. Symmetry pin
    # (the test_lookup_attested_years_skips_page_marker_false_positive
    # test covers the reverse ordering — real year, then page ref).
    assert _earliest_year_in_notes("p. 755 ; Tune 1086") == 1086


def test_lookup_attested_years_skips_page_marker_false_positive(
    fresh_db: Path, tmp_path: Path
) -> None:
    """Real EPNS-style note: 'Plac. de quo War. p. 755.' has 755
    preceded by 'p. ', a PAGE reference, not a year. Pin that the
    page-marker filter rejects it. Earliest qualifying year is the
    actual citation '1086'."""
    sources = tmp_path / "sources"
    sources.mkdir()
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        rid = _seed_toponym_etymology(
            db,
            notes="Westhamconett 1086 DB ; Plac. de quo War. p. 755.",
        )
        lookup_attested_years(db, sources, apply=True)
        year = db.conn.execute(
            "SELECT attested_year FROM toponym_etymology WHERE id = ?", (rid,)
        ).fetchone()["attested_year"]
    assert year == 1086


def test_lookup_attested_years_skips_toponym_etymology_with_null_notes(
    fresh_db: Path, tmp_path: Path
) -> None:
    """Defensive: a toponym_etymology row with notes=NULL (legacy or
    very-low-confidence row) doesn't crash the scan."""
    sources = tmp_path / "sources"
    sources.mkdir()
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        cur = db.conn.execute("INSERT INTO toponym (modern_name) VALUES (?)", ("X",))
        toponym_id = cur.lastrowid
        db.conn.execute(
            "INSERT INTO toponym_etymology "
            "(toponym_id, source_id, confidence, notes) "
            "VALUES (?, ?, ?, ?)",
            (toponym_id, "src", "low", None),
        )
        db.commit()
        result = lookup_attested_years(db, sources, apply=True)

    # null-notes rows are excluded by the WHERE clause, so they don't
    # even surface as scanned.
    assert result["toponym_etymology"]["rows_scanned"] == 0
    assert result["toponym_etymology"]["candidates"] == 0


def test_lookup_attested_years_idempotent_on_toponym_etymology(
    fresh_db: Path, tmp_path: Path
) -> None:
    """Re-running lookup-attested-years against the same DB doesn't
    re-process toponym_etymology rows whose attested_year is already
    populated. Mirror of the etymon_text_match idempotency guarantee
    so cron-style scheduled runs are cheap no-ops."""
    sources = tmp_path / "sources"
    sources.mkdir()
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        _seed_toponym_etymology(db, notes="X 1086 DB")

        first = lookup_attested_years(db, sources, apply=True)
        assert first["toponym_etymology"]["rows_written"] == 1

        second = lookup_attested_years(db, sources, apply=True)
    assert second["toponym_etymology"]["rows_scanned"] == 0
    assert second["toponym_etymology"]["candidates"] == 0
    assert second["toponym_etymology"]["rows_written"] == 0


def test_lookup_attested_years_aggregates_both_row_sources(fresh_db: Path, tmp_path: Path) -> None:
    """The aggregate keys (rows_scanned/candidates/rows_written) sum
    across etymon_text_match AND toponym_etymology. Pin so a future
    refactor that drops one source from the aggregate surfaces."""
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "src.txt").write_text("Tune, 1086 (DB).")
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        _seed_text_match(db, source_id="src", canonical_form="tune")
        _seed_toponym_etymology(db, notes="Tune 1086 DB")
        result = lookup_attested_years(db, sources, apply=True)

    # Each source contributes one row; aggregate is 2.
    assert result["etymon_text_match"]["rows_written"] == 1
    assert result["toponym_etymology"]["rows_written"] == 1
    assert result["rows_written"] == 2
    assert result["candidates"] == 2


def test_clear_enrichment_attested_years_clears_both_tables(
    fresh_db: Path,
) -> None:
    """wyrd-bag: clear_enrichment(stage='attested-years') must NULL
    the column on BOTH etymon_text_match AND toponym_etymology.
    Otherwise a re-run would only re-process half the corpus and
    leave stale years on the other half."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        # Seed a populated etymon_text_match row.
        rid_etm = _seed_text_match(db, source_id="src", canonical_form="tune")
        db.conn.execute(
            "UPDATE etymon_text_match SET attested_year = ? WHERE id = ?",
            (1086, rid_etm),
        )
        # Seed a populated toponym_etymology row.
        rid_te = _seed_toponym_etymology(db, notes="Tune 1086")
        db.conn.execute(
            "UPDATE toponym_etymology SET attested_year = ? WHERE id = ?",
            (1086, rid_te),
        )
        db.commit()

        result = clear_enrichment(db, stage="attested-years", apply=True)
        # The combined count covers both tables.
        assert result["attested_years_to_clear"] == 2

        etm_year = db.conn.execute(
            "SELECT attested_year FROM etymon_text_match WHERE id = ?", (rid_etm,)
        ).fetchone()["attested_year"]
        te_year = db.conn.execute(
            "SELECT attested_year FROM toponym_etymology WHERE id = ?", (rid_te,)
        ).fetchone()["attested_year"]

    assert etm_year is None
    assert te_year is None


def test_lookup_attested_years_accepts_year_at_upper_bound_1700(
    fresh_db: Path, tmp_path: Path
) -> None:
    """Regression for PR #47 Gemini-medium: the bare-year regex
    `1[0-6]\\d{2}` cuts off at 1699 but `_ATTESTED_YEAR_MAX_LOOKUP` is
    1700. Pin that 1700 itself qualifies as a bare year via the
    explicit |1700 alternation."""
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "src.txt").write_text("Tune, 1700.")
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        rid = _seed_text_match(db, source_id="src", canonical_form="tune")
        lookup_attested_years(db, sources, apply=True)
        year = db.conn.execute(
            "SELECT attested_year FROM etymon_text_match WHERE id = ?", (rid,)
        ).fetchone()["attested_year"]
    assert year == 1700


def test_lookup_attested_years_lowercases_form_before_matching(
    fresh_db: Path, tmp_path: Path
) -> None:
    """Regression for PR #47 Gemini-medium: source body text is
    lowercased before scanning, so a matched_form like 'TUNE' or 'Tune'
    must lowercase before being interpolated into the regex pattern.
    Without the form.lower() guard this row would silently miss its
    citation."""
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "src.txt").write_text("TUNE, 1086 (DB).")
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        # Stored matched_form is mixed-case, body is upper. Lowercasing
        # both at scan time makes them line up.
        rid = _seed_text_match(db, source_id="src", canonical_form="Tune")
        lookup_attested_years(db, sources, apply=True)
        year = db.conn.execute(
            "SELECT attested_year FROM etymon_text_match WHERE id = ?", (rid,)
        ).fetchone()["attested_year"]
    assert year == 1086


def test_clear_enrichment_attested_years_resets_only_year_column(
    fresh_db: Path,
) -> None:
    """clear_enrichment(stage='attested-years') nulls out attested_year
    on every etymon_text_match row but leaves the row itself intact —
    matched_form, snippet, method, and the etymon link all survive."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        rid = _seed_text_match(db, source_id="src", canonical_form="tune")
        db.conn.execute(
            "UPDATE etymon_text_match SET attested_year = ? WHERE id = ?",
            (1086, rid),
        )
        db.commit()

        result = clear_enrichment(db, stage="attested-years", apply=True)
        row = db.conn.execute(
            "SELECT matched_form, attested_year FROM etymon_text_match WHERE id = ?",
            (rid,),
        ).fetchone()

    assert result["attested_years_to_clear"] == 1
    assert row["matched_form"] == "tune"  # row not deleted
    assert row["attested_year"] is None


def test_clear_enrichment_all_derived_includes_attested_years(
    fresh_db: Path,
) -> None:
    """stage='all-derived' covers attested-years too — verifies the
    rollup includes the new stage so a fully clean reset really is."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        rid = _seed_text_match(db, source_id="src", canonical_form="tune")
        db.conn.execute(
            "UPDATE etymon_text_match SET attested_year = ? WHERE id = ?",
            (1086, rid),
        )
        db.commit()

        result = clear_enrichment(db, stage="all-derived", apply=True)
        # text-match also clears the row entirely, so attested_year
        # naturally goes to None — but the COUNT before the clear should
        # have been 1, recorded in the dry-run-style counts.
        assert result["attested_years_to_clear"] == 1
        # text-match clear runs alongside, so the row is gone.
        remaining = db.conn.execute("SELECT COUNT(*) FROM etymon_text_match").fetchone()[0]
    assert remaining == 0


def test_record_mining_run_inserts_row_with_full_fields(fresh_db: Path) -> None:
    """record_mining_run persists every field the writer cares about and
    serializes by_failure as JSON."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="test-src", title="Test")
        db.commit()

        run_id = record_mining_run(
            db,
            source_id="test-src",
            provider="ollama",
            model="qwen3.5:9b",
            mode="mine",
            parsed_count=60,
            accepted=39,
            declined=16,
            rejected=5,
            by_failure={"form_not_in_body": 5},
            started_at="2026-04-30T14:30:00Z",
            completed_at="2026-04-30T14:47:00Z",
            notes="round 1",
        )
        assert run_id > 0

        row = db.conn.execute("SELECT * FROM mining_run WHERE id = ?", (run_id,)).fetchone()
    assert row["source_id"] == "test-src"
    assert row["provider"] == "ollama"
    assert row["model"] == "qwen3.5:9b"
    assert row["mode"] == "mine"
    assert row["parsed_count"] == 60
    assert row["accepted"] == 39
    assert row["declined"] == 16
    assert row["rejected"] == 5
    assert row["by_failure"] == '{"form_not_in_body": 5}'
    assert row["started_at"] == "2026-04-30T14:30:00Z"
    assert row["completed_at"] == "2026-04-30T14:47:00Z"
    assert row["notes"] == "round 1"


def test_record_mining_run_idempotent_on_dedupe_key(fresh_db: Path) -> None:
    """Same (source, provider, model, mode, completed_at) inserts once.

    Lets back-fill scripts re-run safely without duplicating runs.
    """
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="test-src", title="Test")
        db.commit()

        kwargs = {
            "source_id": "test-src",
            "provider": "ollama",
            "model": "qwen3.5:9b",
            "mode": "mine",
            "parsed_count": 10,
            "accepted": 8,
            "declined": 1,
            "rejected": 1,
            "completed_at": "2026-04-30T14:47:00Z",
        }
        first = record_mining_run(db, **kwargs)
        second = record_mining_run(db, **kwargs)
        assert first > 0
        # INSERT OR IGNORE returns lastrowid=0 on UNIQUE conflict
        assert second == 0

        n = db.conn.execute("SELECT COUNT(*) FROM mining_run").fetchone()[0]
        assert n == 1


def test_record_mining_run_rejects_invalid_mode(fresh_db: Path) -> None:
    """The mode CHECK constraint rejects anything outside 'mine' / 'review'."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="test-src", title="Test")
        db.commit()

        with pytest.raises(sqlite3.IntegrityError):
            record_mining_run(
                db,
                source_id="test-src",
                provider="ollama",
                model="qwen3.5:9b",
                mode="bogus",
                parsed_count=1,
                accepted=1,
                declined=0,
                rejected=0,
            )


def test_import_mining_log_inserts_jsonl_records(fresh_db: Path, tmp_path: Path) -> None:
    """The back-fill CLI parses JSONL mining-run records and inserts them
    via record_mining_run, idempotently."""
    from click.testing import CliRunner

    from wyrd.generators.kenning.cli import cli

    # Seed two source rows so FK on mining_run.source_id passes.
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="skeat_1901_cambridgeshire", title="Skeat Cambridgeshire")
        db.upsert_source(id="mawer_1920_northumberland_durham", title="Mawer Northumberland")
        db.commit()

    log_path = tmp_path / "runs.jsonl"
    log_path.write_text(
        "\n".join(
            [
                '{"source_id":"skeat_1901_cambridgeshire","provider":"ollama",'
                '"model":"qwen3.5:9b","mode":"mine","accepted":39,"declined":16,'
                '"rejected":5,"by_failure":{"form_not_in_body":5},'
                '"completed_at":"2026-04-30T14:47:00Z","notes":"round 1"}',
                '{"source_id":"mawer_1920_northumberland_durham","provider":"anthropic",'
                '"model":"claude-haiku-4-5-20251001","mode":"mine","accepted":482,'
                '"declined":210,"rejected":112,"completed_at":"2026-05-01T03:00:00Z"}',
                "",
                "# comment lines and blanks are skipped",
            ]
        )
    )

    runner = CliRunner()

    # Dry-run reports counts without inserting.
    result = runner.invoke(
        cli, ["lexicon", "import-mining-log", str(log_path), "--db", str(fresh_db)]
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    with LexiconDB(fresh_db) as db:
        assert db.conn.execute("SELECT COUNT(*) FROM mining_run").fetchone()[0] == 0

    # Apply: rows land.
    result = runner.invoke(
        cli,
        ["lexicon", "import-mining-log", str(log_path), "--db", str(fresh_db), "--apply"],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    with LexiconDB(fresh_db) as db:
        assert db.conn.execute("SELECT COUNT(*) FROM mining_run").fetchone()[0] == 2

    # Re-apply: dedupe — count stays at 2.
    result = runner.invoke(
        cli,
        ["lexicon", "import-mining-log", str(log_path), "--db", str(fresh_db), "--apply"],
    )
    assert result.exit_code == 0
    with LexiconDB(fresh_db) as db:
        assert db.conn.execute("SELECT COUNT(*) FROM mining_run").fetchone()[0] == 2


def test_import_mining_log_tolerates_malformed_records(fresh_db: Path, tmp_path: Path) -> None:
    """A single bad row must not crash the whole import — every failure
    mode (non-object JSON, non-numeric counts, by_failure not a dict,
    invalid mode hitting the CHECK constraint) surfaces as an error and
    the rest of the file proceeds."""
    from click.testing import CliRunner

    from wyrd.generators.kenning.cli import cli

    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="real-src", title="Real")
        db.commit()

    log_path = tmp_path / "mixed.jsonl"
    log_path.write_text(
        "\n".join(
            [
                # Good: should land
                '{"source_id":"real-src","provider":"ollama","model":"qwen3.5:9b",'
                '"mode":"mine","accepted":10,"declined":2,"rejected":1,'
                '"completed_at":"2026-04-30T14:00:00Z"}',
                # Bad: non-object JSON
                "[1, 2, 3]",
                # Bad: non-numeric count
                '{"source_id":"real-src","provider":"ollama","model":"x",'
                '"mode":"mine","accepted":"not-a-number","completed_at":"2026-04-30T15:00:00Z"}',
                # Bad: by_failure is a string, not an object
                '{"source_id":"real-src","provider":"ollama","model":"x",'
                '"mode":"mine","accepted":1,"by_failure":"oops",'
                '"completed_at":"2026-04-30T16:00:00Z"}',
                # Bad: invalid mode (CHECK constraint)
                '{"source_id":"real-src","provider":"ollama","model":"x",'
                '"mode":"bogus","accepted":1,"completed_at":"2026-04-30T17:00:00Z"}',
                # Good: another valid row
                '{"source_id":"real-src","provider":"gemini","model":"gemini-2.5-flash",'
                '"mode":"review","accepted":5,"completed_at":"2026-04-30T18:00:00Z"}',
            ]
        )
    )

    result = CliRunner().invoke(
        cli,
        [
            "lexicon",
            "import-mining-log",
            str(log_path),
            "--db",
            str(fresh_db),
            "--apply",
        ],
    )
    assert result.exit_code == 0, (result.output or "") + (result.stderr or "")

    # The two good rows landed; the four bad rows didn't crash the import.
    with LexiconDB(fresh_db) as db:
        n = db.conn.execute("SELECT COUNT(*) FROM mining_run").fetchone()[0]
    assert n == 2


def test_import_mining_log_skips_unknown_sources(fresh_db: Path, tmp_path: Path) -> None:
    """Records pointing at sources not in the source table get skipped with
    an error count, not a hard FK failure."""
    from click.testing import CliRunner

    from wyrd.generators.kenning.cli import cli

    log_path = tmp_path / "runs.jsonl"
    log_path.write_text(
        '{"source_id":"never-existed","provider":"ollama","model":"x",'
        '"mode":"mine","accepted":1,"declined":0,"rejected":0,'
        '"completed_at":"2026-05-01T00:00:00Z"}\n'
    )
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["lexicon", "import-mining-log", str(log_path), "--db", str(fresh_db), "--apply"],
    )
    assert result.exit_code == 0
    with LexiconDB(fresh_db) as db:
        assert db.conn.execute("SELECT COUNT(*) FROM mining_run").fetchone()[0] == 0


def test_lexicon_mine_llm_records_mining_run_at_end_of_run(
    fresh_db: Path, tmp_path: Path, monkeypatch
) -> None:
    """The CLI integration: running `lexicon mine-llm` end-to-end against
    a stubbed extractor must persist a mining_run row. Without this test,
    a regression that drops the record_mining_run call in lexicon_mine_llm
    passes CI silently — only the helper's unit tests would notice."""
    from click.testing import CliRunner

    from wyrd.generators.kenning import cli as cli_mod
    from wyrd.generators.kenning import llm_extractor
    from wyrd.generators.kenning.llm_extractor import LLMResult
    from wyrd.generators.kenning.skeat_parser import ParsedElement, ParsedEntry

    # Fake parsed entries — bypasses the regex parser.
    fake_parsed = [
        ParsedEntry(
            toponym="Faketon",
            section_suffix="-ton",
            historical_form="fake-tun",
            elements=[],
            confidence="low",
            source_quote="Faketon. fake body.",
        ),
        ParsedEntry(
            toponym="Otherton",
            section_suffix="-ton",
            historical_form="other-tun",
            elements=[],
            confidence="low",
            source_quote="Otherton. other body.",
        ),
    ]
    monkeypatch.setattr(cli_mod, "_select_parser_and_run", lambda text, parser, **_: fake_parsed)

    # Fake OllamaClient — only needs .model and .base_url for echoed output.
    class FakeClient:
        model = "fake-model:0b"
        base_url = "http://localhost:0"

        def __init__(self, **_kwargs):
            pass

    monkeypatch.setattr(llm_extractor, "OllamaClient", FakeClient)

    # Fake extract_one: first entry accepts with one element, second declines.
    accept_result = LLMResult(
        entry=ParsedEntry(
            toponym="Faketon",
            section_suffix="-ton",
            historical_form="fake-tun",
            elements=[
                ParsedElement(
                    form="fake", language="old-english", position="pre", gloss="fake gloss"
                ),
            ],
            confidence="high",
            source_quote="Faketon. fake body.",
        ),
        raw_response={},
        failures=[],
        accepted=True,
    )
    decline_result = LLMResult(
        entry=ParsedEntry(
            toponym="Otherton",
            section_suffix="-ton",
            historical_form=None,
            elements=[],
            confidence="low",
            source_quote="Otherton. other body.",
        ),
        raw_response={},
        failures=[],
        accepted=True,
    )
    seq = iter([accept_result, decline_result])
    monkeypatch.setattr(llm_extractor, "extract_one", lambda client, **kw: next(seq))

    # Need a non-empty file for the click PathExists check.
    src_path = tmp_path / "test_book.txt"
    src_path.write_text("Faketon. fake body.\n\nOtherton. other body.\n")

    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli,
        ["lexicon", "mine-llm", str(src_path), "--db", str(fresh_db), "--provider", "ollama"],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")

    # The writer call landed exactly one mining_run row with the expected counts.
    with LexiconDB(fresh_db) as db:
        rows = db.conn.execute(
            "SELECT source_id, provider, model, mode, parsed_count, accepted, declined, "
            "rejected FROM mining_run"
        ).fetchall()
    assert len(rows) == 1
    row = rows[0]
    assert row["source_id"] == "test_book"
    assert row["provider"] == "ollama"
    assert row["model"] == "fake-model:0b"
    assert row["mode"] == "mine"
    assert row["parsed_count"] == 2
    assert row["accepted"] == 1
    assert row["declined"] == 1
    assert row["rejected"] == 0


# --- wyrd-9kh.5: running-header page parser ----------------------------


def test_parse_running_header_pages_finds_mawer_style_headers():
    """Mawer-style running headers like 'BACKWORTH 9' get parsed into
    (offset, page) pairs sorted by offset."""
    from wyrd.generators.kenning.lexicon import parse_running_header_pages

    body = (
        "front matter\n"
        "BACKWORTH 9\n"
        "  body of page 9...\n"
        "BEDLINGTON 15\n"
        "  body of page 15...\n"
        "BISHOPWEARMOUTH 23\n"
    )
    headers = parse_running_header_pages(body)
    pages = [page for _offset, page in headers]
    assert pages == [9, 15, 23]
    offsets = [offset for offset, _page in headers]
    assert offsets == sorted(offsets)


def test_parse_running_header_pages_handles_multi_word_headwords():
    """Multi-word ALL CAPS headwords (e.g. 'ABBEY DORE 1') are detected."""
    from wyrd.generators.kenning.lexicon import parse_running_header_pages

    body = "ABBEY DORE 1\n  body...\nST PETER'S 47\n"
    headers = parse_running_header_pages(body)
    pages = [page for _offset, page in headers]
    assert 1 in pages
    assert 47 in pages


def test_parse_running_header_pages_returns_empty_when_no_match():
    """Skeat-style §-section headers don't match the Mawer pattern;
    return an empty list rather than raising."""
    from wyrd.generators.kenning.lexicon import parse_running_header_pages

    skeat_style = "§   2.      NAMES   ENDING   IN   -TON.  9\n  body of section\n"
    assert parse_running_header_pages(skeat_style) == []


def test_page_for_offset_returns_closest_preceding_header():
    """A character offset between two headers gets the page of the earlier
    header — that's the page the body text physically sits on."""
    from wyrd.generators.kenning.lexicon import page_for_offset

    headers = [(100, 1), (200, 2), (300, 3)]
    assert page_for_offset(headers, 50) is None  # before any header
    assert page_for_offset(headers, 100) == 1  # at the first header
    assert page_for_offset(headers, 150) == 1  # between 1st and 2nd
    assert page_for_offset(headers, 200) == 2  # at the second header
    assert page_for_offset(headers, 999) == 3  # after the last header


def test_page_for_offset_empty_headers_returns_none():
    """A book with no detectable running headers returns None for any
    offset — caller can fall back to NULL page."""
    from wyrd.generators.kenning.lexicon import page_for_offset

    assert page_for_offset([], 0) is None
    assert page_for_offset([], 12345) is None


# --- wyrd-8st: parse_skeat_section_header_pages ----------------------


def test_parse_skeat_section_header_pages_finds_canonical_header():
    """Canonical Skeat-§ header `§ N. NAMES ENDING IN -X. <page>` parses
    as (offset, page)."""
    body = (
        "front matter\n"
        "§ 2. NAMES ENDING IN -TON. 5\n"
        "body of page 5...\n"
        "§ 4. NAMES ENDING IN -HAM. 21\n"
        "body of page 21...\n"
    )
    headers = parse_skeat_section_header_pages(body)
    pages = [page for _offset, page in headers]
    assert pages == [5, 21]


def test_parse_skeat_section_header_pages_handles_ocr_variants():
    """OCR-tolerated variants: '§8.' (no space after §), 'IX' OCR misread
    of 'IN', multi-suffix `-BRIDGE, -HITHE` titles."""
    body = (
        "§8.      NAMES    ENDING    IX    -BRIDGE,   -HITHE.  33\n"
        "§   8.      NAMES    ENDING    IN    -WELL.  37\n"
    )
    headers = parse_skeat_section_header_pages(body)
    pages = [page for _offset, page in headers]
    assert pages == [33, 37]


def test_parse_skeat_section_header_pages_rejects_title_case_section_openers():
    """Section openers in the source (`§ 2. The suffix -ton.`) are
    title-case with no trailing page — they must NOT be parsed as
    running headers, even though they share the leading `§ N.`."""
    body = "§ 2. The suffix -ton.\ndiscussion follows\n§ 4. The suffix -ham.\n"
    assert parse_skeat_section_header_pages(body) == []


def test_parse_skeat_section_header_pages_rejects_toc_entries():
    """TOC entries (`§ 1. Prefatory Remarks 1`, `§ 5. The suffix -stead :
    — Olmstead 25`) are title-case and don't share the body running-
    header shape — they should not match."""
    toc = (
        "§ 1.     Prefatory Remarks 1\n"
        "§ 5.     The suffix -stead : — Olmstead 25\n"
        "§ 12.    List of Ancient Manors 74\n"
    )
    # Either no matches, or the page numbers should NOT overlap the
    # canonical body header pages (which would corrupt offset lookups).
    headers = parse_skeat_section_header_pages(toc)
    assert headers == [], (
        f"TOC entries leaked through Skeat parser as {headers}; the regex "
        "should require an ALL-CAPS title block, not title-case."
    )


def test_parse_skeat_section_header_pages_returns_empty_when_no_match():
    """A Mawer-style `BACKWORTH 9` body produces no Skeat-§ matches —
    callers can fall back to the Mawer parser."""
    mawer_body = "BACKWORTH 9\nbody\nBEDLINGTON 15\nbody\n"
    assert parse_skeat_section_header_pages(mawer_body) == []


def test_detect_running_headers_picks_skeat_when_higher_yield():
    """Per-source dispatch: if Skeat-§ matches more than Mawer (e.g. a
    Skeat county dictionary), the Skeat parser wins."""
    body = "§ 2. NAMES ENDING IN -TON. 5\nbody\n§ 4. NAMES ENDING IN -HAM. 21\nbody\n"
    headers, parser = detect_running_headers(body)
    assert parser == "skeat-§"
    assert [p for _o, p in headers] == [5, 21]


def test_detect_running_headers_picks_mawer_when_higher_yield():
    """A Mawer-style book wins dispatch when the Mawer parser produces
    more matches than the Skeat parser."""
    body = "BACKWORTH 9\nbody\nBEDLINGTON 15\nbody\nBISHOPWEARMOUTH 23\n"
    headers, parser = detect_running_headers(body)
    assert parser == "mawer"
    assert [p for _o, p in headers] == [9, 15, 23]


def test_detect_running_headers_returns_none_when_neither_matches():
    """A source with no recognizable header convention returns ([], 'none')
    so backfill_citation_pages can record no_headers and skip cleanly."""
    body = "this is just prose with no running headers anywhere\n"
    headers, parser = detect_running_headers(body)
    assert headers == []
    assert parser == "none"


def test_detect_running_headers_breaks_tie_to_mawer():
    """Tie-break: when both parsers return the same nonzero count,
    Mawer wins (documented in detect_running_headers). Pinned here so
    a refactor can't silently flip it — Mawer is the older / more
    permissive pattern; defaulting to it on ties keeps page-resolution
    behavior bit-stable for sources already routed through it."""
    body = "BACKWORTH 9\nbody content\n§ 2. NAMES ENDING IN -TON. 5\nmore body content\n"
    headers, parser = detect_running_headers(body)
    assert parser == "mawer"
    assert [p for _o, p in headers] == [9]


def test_backfill_citation_pages_resolves_skeat_section_headers(fresh_db: Path) -> None:
    """End-to-end: a Skeat-style book whose body has §-section running
    headers gets pages resolved via the new parser dispatch — would
    have failed under the Mawer-only path with no_headers=1.
    Both etymon_citation and toponym_etymology pages get written in
    the same pass."""
    quote = "discussion of -ton suffix in cot"
    body = (
        "§ 2. NAMES ENDING IN -TON. 5\n"
        f"earlier text\n"
        "§ 4. NAMES ENDING IN -HAM. 21\n"
        f"{quote} continuing\n"
    )
    with LexiconDB(fresh_db) as db:
        _, cit_id, ety_id = _seed_citation_for_backfill(db, "skeat_test", "cot", quote)
        counts = backfill_citation_pages(db, "skeat_test", body, apply=True)
        cit_page = db.conn.execute(
            "SELECT page FROM etymon_citation WHERE id = ?", (cit_id,)
        ).fetchone()["page"]
        ety_page = db.conn.execute(
            "SELECT page FROM toponym_etymology WHERE id = ?", (ety_id,)
        ).fetchone()["page"]

    assert counts["citations_updated"] == 1
    assert counts["etymologies_updated"] == 1
    assert cit_page == "21"
    assert ety_page == "21"
    assert counts["no_headers"] == 0


def test_parse_pages_cli_reports_skeat_parser_for_skeat_source(tmp_path: Path) -> None:
    """The parse-pages CLI audit must report which parser matched so a
    user inspecting a source knows whether it's Mawer-style or
    Skeat-§. Pin the 'parser: skeat-§' line in the output."""
    body = "§ 2. NAMES ENDING IN -TON. 5\nbody content\n§ 4. NAMES ENDING IN -HAM. 21\n"
    source = tmp_path / "skeat_test.txt"
    source.write_text(body)

    runner = CliRunner()
    result = runner.invoke(kenning_cli, ["lexicon", "parse-pages", str(source)])

    assert result.exit_code == 0, result.output
    assert "2 running header(s) detected" in result.output
    assert "parser: skeat-§" in result.output
    assert "Page range: 5 → 21" in result.output


def test_parse_pages_cli_reports_no_match_when_neither_convention_fires(
    tmp_path: Path,
) -> None:
    """A header-less source body produces 'parser: none' and a stderr
    note explaining both conventions were tried — pin so the wording
    can't silently revert to the old Mawer-only language."""
    body = "this is just prose with no headers anywhere\n"
    source = tmp_path / "treatise.txt"
    source.write_text(body)

    runner = CliRunner()
    result = runner.invoke(kenning_cli, ["lexicon", "parse-pages", str(source)])

    assert result.exit_code == 0, result.output
    assert "0 running header(s) detected" in result.output
    assert "parser: none" in result.output
    assert "either Mawer-style or Skeat-§ patterns" in result.output


# --- wyrd-azv: backfill_citation_pages ---------------------------------


def test_quote_body_excerpt_strips_provider_attribution_prefix() -> None:
    """`assemble_extraction_result` writes quotes shaped as
    `extracted_by:<provider>:<model>; <commentary> | <body excerpt>`.
    `_quote_body_excerpt` must return only the body part, since that's
    the only substring that exists in source_text."""
    quote = "extracted_by:llm:qwen3.5:9b; Scholar identifies foo | Abberwick (Eglingham). 1169 P"
    assert _quote_body_excerpt(quote) == "Abberwick (Eglingham). 1169 P"


def test_quote_body_excerpt_uses_first_separator_not_last() -> None:
    """Pin the leftmost-find behavior: if an OCR body picks up a
    spurious ` | ` (table rule, page artefact), the prefix splitter
    must still terminate at the FIRST ` | ` because the
    extracted_by:... prefix never contains the separator. Using rfind
    instead would return only the trailing artefact."""
    quote = "extracted_by:gemini:flash; commentary | body | with | rules"
    assert _quote_body_excerpt(quote) == "body | with | rules"


def test_quote_body_excerpt_returns_whole_quote_when_no_separator() -> None:
    """Legacy citations and commentary-only rows have no ` | ` —
    fall back to the literal quote so caller's str.find still has
    something to try."""
    assert _quote_body_excerpt("plain body excerpt") == "plain body excerpt"


def test_backfill_normalizes_ocr_multispace_against_single_space_quote(
    fresh_db: Path,
) -> None:
    """The parser collapses OCR multi-space runs before sending text to
    the LLM, so the stored quote ends up single-spaced while the source
    has the original `Abberwick  (Eglingham)` double-space runs.
    `_normalize_for_quote_match` exists to bridge that — pin it here so
    a regression in the normalization or the norm_to_orig offset
    translation surfaces immediately."""
    quote = "the body discussion of cot here"
    body = (
        "BACKWORTH 9\n"
        "earlier text\n"
        "BEDLINGTON 15\n"
        "the  body   discussion  of cot here  continuing\n"  # multi-space OCR
    )
    with LexiconDB(fresh_db) as db:
        _, cit_id, _ = _seed_citation_for_backfill(db, "ocr_book", "cot", quote)
        counts = backfill_citation_pages(db, "ocr_book", body, apply=True)
        cit_page = db.conn.execute(
            "SELECT page FROM etymon_citation WHERE id = ?", (cit_id,)
        ).fetchone()["page"]

    assert counts["citations_updated"] == 1
    assert cit_page == "15"


def test_backfill_resolves_page_with_provider_attribution_prefix(fresh_db: Path) -> None:
    """End-to-end check that the prefix-stripping + body matching
    works on the real `extracted_by:<provider>:<model>; <commentary> |
    <body>` shape produced by `assemble_extraction_result`. Pins the
    contract so a regression in either `_quote_body_excerpt` or the
    matching path surfaces here."""
    body_excerpt = "the body discussion of cot here"
    full_quote = (
        "extracted_by:llm:qwen3.5:9b; Scholar identifies cot as the suffix; "
        "notes contextual usage | " + body_excerpt
    )
    body = f"BACKWORTH 9\n{body_excerpt} continuing\n"
    with LexiconDB(fresh_db) as db:
        _, cit_id, _ = _seed_citation_for_backfill(db, "real_book", "cot", full_quote)
        counts = backfill_citation_pages(db, "real_book", body, apply=True)
        cit_page = db.conn.execute(
            "SELECT page FROM etymon_citation WHERE id = ?", (cit_id,)
        ).fetchone()["page"]

    assert counts["citations_updated"] == 1
    assert cit_page == "9"


def test_normalize_for_quote_match_collapses_multispace_and_tracks_offsets() -> None:
    """Direct unit test on the helper: collapsed text is single-spaced;
    norm_to_orig maps each normalized character back to its original
    offset so the page lookup uses the source coordinate system."""
    text = "AB  CD\n  EF"
    norm, mapping = _normalize_for_quote_match(text)
    assert norm == "AB CD EF"
    # 'A' is at original 0; 'B' at original 1; ' ' (collapsed) at original 2;
    # 'C' at original 4; 'D' at original 5; ' ' (collapsed from \n+spaces) at 6;
    # 'E' at 9; 'F' at 10
    assert mapping == [0, 1, 2, 4, 5, 6, 9, 10]
    # Round-trip: substring search returns a normalized offset that
    # translates back to the correct original-text position.
    norm_offset = norm.find("EF")
    assert text[mapping[norm_offset] : mapping[norm_offset] + 2] == "EF"


def _seed_citation_for_backfill(
    db: LexiconDB, source_id: str, etymon_form: str, quote: str
) -> tuple[int, int, int]:
    """Build the minimum DB shape backfill_citation_pages needs:
    a source, an etymon, an etymon_citation row, plus a toponym +
    toponym_etymology row carrying the same source_quote in `notes`.
    Returns (etymon_id, citation_id, etymology_id)."""
    db.upsert_source(id=source_id, title=source_id)
    etymon_id = db.upsert_etymon(etymon_form, "old-english")
    db.add_citation(etymon_id, source_id, short_quote=quote)
    cit = db.conn.execute(
        "SELECT id FROM etymon_citation WHERE etymon_id = ? AND source_id = ?",
        (etymon_id, source_id),
    ).fetchone()
    cur = db.conn.execute("INSERT INTO toponym (modern_name) VALUES (?)", (etymon_form.title(),))
    toponym_id = cur.lastrowid
    cur = db.conn.execute(
        "INSERT INTO toponym_etymology (toponym_id, source_id, confidence, notes) "
        "VALUES (?, ?, 'high', ?)",
        (toponym_id, source_id, quote),
    )
    db.commit()
    return etymon_id, cit["id"], cur.lastrowid


def test_backfill_citation_pages_writes_pages_when_quote_in_text(fresh_db: Path) -> None:
    """Happy path: an etymon_citation whose short_quote appears in the
    source body after a running header gets that header's page written."""

    quote = "the body discussion of cot here"
    body = (
        "front matter\n"
        "BACKWORTH 9\n"
        "earlier text on page 9\n"
        "BEDLINGTON 15\n"
        f"{quote} continuing the entry\n"
    )
    with LexiconDB(fresh_db) as db:
        _, cit_id, ety_id = _seed_citation_for_backfill(db, "mawer_1920_test", "cot", quote)
        counts = backfill_citation_pages(db, "mawer_1920_test", body, apply=True)
        cit_page = db.conn.execute(
            "SELECT page FROM etymon_citation WHERE id = ?", (cit_id,)
        ).fetchone()["page"]
        ety_page = db.conn.execute(
            "SELECT page FROM toponym_etymology WHERE id = ?", (ety_id,)
        ).fetchone()["page"]

    assert counts["citations_updated"] == 1
    assert counts["etymologies_updated"] == 1
    assert cit_page == "15"
    assert ety_page == "15"


def test_backfill_citation_pages_dry_run_does_not_write(fresh_db: Path) -> None:
    """apply=False reports counts but leaves page columns NULL."""

    quote = "discussion of ham morpheme"
    body = f"HAMSTEAD 22\n{quote}\n"
    with LexiconDB(fresh_db) as db:
        _, cit_id, ety_id = _seed_citation_for_backfill(db, "test_book", "ham", quote)
        counts = backfill_citation_pages(db, "test_book", body, apply=False)
        cit_page = db.conn.execute(
            "SELECT page FROM etymon_citation WHERE id = ?", (cit_id,)
        ).fetchone()["page"]
        ety_page = db.conn.execute(
            "SELECT page FROM toponym_etymology WHERE id = ?", (ety_id,)
        ).fetchone()["page"]

    assert counts["citations_updated"] == 1
    assert counts["etymologies_updated"] == 1
    assert cit_page is None
    assert ety_page is None


def test_backfill_citation_pages_no_headers_returns_zero(fresh_db: Path) -> None:
    """A source whose body matches NEITHER Mawer nor Skeat-§ convention
    (e.g. a treatise / chapter prose book like Joyce's Irish Names)
    reports no_headers=1 and zero updates without crashing."""

    treatise_body = (
        "This chapter discusses the topographical naming conventions "
        "of the western counties without using running headers anywhere "
        "in the body discussion here.\n"
    )
    with LexiconDB(fresh_db) as db:
        _, cit_id, _ = _seed_citation_for_backfill(
            db, "treatise_book", "ton", "body discussion here"
        )
        counts = backfill_citation_pages(db, "treatise_book", treatise_body, apply=True)
        cit_page = db.conn.execute(
            "SELECT page FROM etymon_citation WHERE id = ?", (cit_id,)
        ).fetchone()["page"]

    assert counts["no_headers"] == 1
    assert counts["citations_updated"] == 0
    assert counts["etymologies_updated"] == 0
    assert cit_page is None


def test_backfill_citation_pages_quote_not_in_text_increments_miss(fresh_db: Path) -> None:
    """A citation whose short_quote isn't present in the source body
    (parser drift, OCR mangle) is skipped and counted under
    quote_not_in_text rather than failing the run."""

    body = "BACKWORTH 9\nactual body content here\n"
    quote_not_in_body = "this exact phrase is nowhere"
    with LexiconDB(fresh_db) as db:
        _, cit_id, _ = _seed_citation_for_backfill(db, "mawer_1920_test", "cot", quote_not_in_body)
        counts = backfill_citation_pages(db, "mawer_1920_test", body, apply=True)
        cit_page = db.conn.execute(
            "SELECT page FROM etymon_citation WHERE id = ?", (cit_id,)
        ).fetchone()["page"]

    # citation + etymology rows both miss → counter at 2.
    assert counts["quote_not_in_text"] == 2
    assert counts["citations_updated"] == 0
    assert cit_page is None


def test_backfill_citation_pages_idempotent(fresh_db: Path) -> None:
    """A second --apply run is a no-op: rows where page IS NOT NULL are
    excluded from the candidate query."""

    quote = "body of the entry"
    body = f"ABBEY DORE 7\n{quote}\n"
    with LexiconDB(fresh_db) as db:
        _, cit_id, _ = _seed_citation_for_backfill(db, "test_book", "abbey", quote)
        first = backfill_citation_pages(db, "test_book", body, apply=True)
        second = backfill_citation_pages(db, "test_book", body, apply=True)
        cit_page = db.conn.execute(
            "SELECT page FROM etymon_citation WHERE id = ?", (cit_id,)
        ).fetchone()["page"]

    assert first["citations_updated"] == 1
    assert second["citations_updated"] == 0
    assert cit_page == "7"


def test_backfill_citation_pages_skips_offset_before_first_header(fresh_db: Path) -> None:
    """An entry that sits before the first running header (e.g. front-
    matter) gets counted under before_first_page, not silently dropped."""

    quote = "preface discussing the methodology"
    body = f"{quote}\nBACKWORTH 9\nbody content\n"
    with LexiconDB(fresh_db) as db:
        _, cit_id, _ = _seed_citation_for_backfill(db, "test_book", "cot", quote)
        counts = backfill_citation_pages(db, "test_book", body, apply=True)
        cit_page = db.conn.execute(
            "SELECT page FROM etymon_citation WHERE id = ?", (cit_id,)
        ).fetchone()["page"]

    # citation + etymology, both before page 1 → counter at 2.
    assert counts["before_first_page"] == 2
    assert counts["citations_updated"] == 0
    assert cit_page is None


def test_backfill_citation_pages_skips_no_quote_rows(fresh_db: Path) -> None:
    """Rows with NULL short_quote (legacy citations or rando-port seed
    rows) are counted under no_quote rather than crashing on str.find."""

    body = "BACKWORTH 9\nbody content\n"
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="legacy_book", title="legacy")
        etymon_id = db.upsert_etymon("cot", "old-english")
        db.add_citation(etymon_id, "legacy_book", short_quote=None)
        db.commit()

        counts = backfill_citation_pages(db, "legacy_book", body, apply=True)
        cit_page = db.conn.execute(
            "SELECT page FROM etymon_citation WHERE etymon_id = ?", (etymon_id,)
        ).fetchone()["page"]

    assert counts["no_quote"] == 1
    assert counts["citations_updated"] == 0
    assert cit_page is None


def test_backfill_pages_cli_dry_run_reports_without_writing(fresh_db: Path, tmp_path: Path) -> None:
    """End-to-end CLI test: dry-run reports counts to stderr and leaves
    the DB untouched. --apply is required to write."""

    quote = "discussion of cot in the entry"
    body = f"BACKWORTH 9\n{quote}\n"
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "mawer_1920_test.txt").write_text(body)

    with LexiconDB(fresh_db) as db:
        _, cit_id, _ = _seed_citation_for_backfill(db, "mawer_1920_test", "cot", quote)

    runner = CliRunner()
    result = runner.invoke(
        kenning_cli,
        [
            "lexicon",
            "backfill-pages",
            str(sources_dir),
            "--db",
            str(fresh_db),
        ],
    )

    assert result.exit_code == 0, result.output
    # Dry-run leaves page NULL
    with LexiconDB(fresh_db) as db:
        cit_page = db.conn.execute(
            "SELECT page FROM etymon_citation WHERE id = ?", (cit_id,)
        ).fetchone()["page"]
    assert cit_page is None
    assert "dry-run" in result.output
    assert "citations_updated     = 1" in result.output


def test_backfill_pages_cli_apply_writes_to_db(fresh_db: Path, tmp_path: Path) -> None:
    """--apply persists the page lookup; CLI summary mirrors lib counts."""
    quote = "discussion of cot in the entry"
    body = f"BACKWORTH 9\n{quote}\n"
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "mawer_1920_test.txt").write_text(body)

    with LexiconDB(fresh_db) as db:
        _, cit_id, _ = _seed_citation_for_backfill(db, "mawer_1920_test", "cot", quote)

    runner = CliRunner()
    result = runner.invoke(
        kenning_cli,
        [
            "lexicon",
            "backfill-pages",
            str(sources_dir),
            "--db",
            str(fresh_db),
            "--apply",
        ],
    )

    assert result.exit_code == 0, result.output
    with LexiconDB(fresh_db) as db:
        cit_page = db.conn.execute(
            "SELECT page FROM etymon_citation WHERE id = ?", (cit_id,)
        ).fetchone()["page"]
    assert cit_page == "9"
    assert "dry-run" not in result.output


def test_backfill_pages_cli_source_filter_skips_other_books(fresh_db: Path, tmp_path: Path) -> None:
    """--source X processes only sources/X.txt; other books in the dir
    don't get touched."""
    quote_a = "entry-A body content"
    quote_b = "entry-B body content"
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "book_a.txt").write_text(f"AAA 1\n{quote_a}\n")
    (sources_dir / "book_b.txt").write_text(f"BBB 1\n{quote_b}\n")

    with LexiconDB(fresh_db) as db:
        _, cit_a, _ = _seed_citation_for_backfill(db, "book_a", "aa", quote_a)
        _, cit_b, _ = _seed_citation_for_backfill(db, "book_b", "bb", quote_b)

    runner = CliRunner()
    result = runner.invoke(
        kenning_cli,
        [
            "lexicon",
            "backfill-pages",
            str(sources_dir),
            "--db",
            str(fresh_db),
            "--source",
            "book_a",
            "--apply",
        ],
    )

    assert result.exit_code == 0, result.output
    with LexiconDB(fresh_db) as db:
        page_a = db.conn.execute(
            "SELECT page FROM etymon_citation WHERE id = ?", (cit_a,)
        ).fetchone()["page"]
        page_b = db.conn.execute(
            "SELECT page FROM etymon_citation WHERE id = ?", (cit_b,)
        ).fetchone()["page"]
    assert page_a == "1"
    assert page_b is None


def test_reverse_search_captures_wider_snippet_window(fresh_db: Path, tmp_path: Path) -> None:
    """wyrd-9kh.4: reverse_search_attestations writes ±100 char snippets
    (up from the prior ±60). Pins the wider window so the SPA citation
    panel has enough surrounding scholarly prose to display."""
    from wyrd.generators.kenning.lexicon import reverse_search_attestations

    # Build a long body with the matched form near the middle so ±100
    # chars on either side comfortably stay within the body.
    pre = "a " * 80  # 160 chars
    post = " b" * 80  # 160 chars
    body = f"{pre}MARKER ham TRAIL{post}"
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "test_book.txt").write_text(body)

    with LexiconDB(fresh_db) as db:
        ham_id = db.upsert_etymon("ham", "old-english")
        db.upsert_source(id="rando-port", title="Rando port (seed)")
        db.upsert_source(id="test_book", title="Test Book")
        db.add_citation(ham_id, "rando-port")
        db.commit()

        reverse_search_attestations(db, sources_dir, apply=True, min_form_length=3)

        snippet = db.conn.execute(
            "SELECT snippet FROM etymon_text_match WHERE etymon_id = ?",
            (ham_id,),
        ).fetchone()["snippet"]

    # Both the leading MARKER and trailing TRAIL words should be
    # captured: they sit ~6 chars from the «ham» match. A ±60 window
    # would still catch them; the wider window also pulls in some of
    # the surrounding 'a / b' filler. Verify by character count: the
    # snippet should be at least 150 chars (well over the old ±60
    # window's 126-char ceiling around the 5-char match).
    assert "marker" in snippet.lower()
    assert "trail" in snippet.lower()
    assert "«ham»" in snippet
    assert len(snippet) > 150, f"snippet only {len(snippet)} chars: {snippet!r}"


def test_fuzzy_search_skips_match_when_token_is_canonical_etymon(
    fresh_db: Path, tmp_path: Path
) -> None:
    """Per wyrd-c3x: fuzzy-search must not claim 'X is a fuzzy variant of
    Y' when X is itself a canonical etymon in the lexicon. The herath ↔
    heath case (ON herað 'district' vs OE hǣþ 'untilled land') passed
    the gloss anchor because 'land' is a generic gloss, but the result
    is structurally wrong: 'heath' has its own etymon row.

    The cross-check fix: if the body token equals any canonical_form,
    skip the fuzzy claim. OCR variants between two etymons are
    normalize-ocr's job, not fuzzy-search's.
    """
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "test_book.txt").write_text(
        "The heath is bare untilled land where shepherds once grazed.\n"
    )

    with LexiconDB(fresh_db) as db:
        # rando-port-only candidate with a generic gloss — the kind that
        # would historically false-positive against 'heath'.
        herath_id = db.upsert_etymon("herath", "old-norse")
        db.add_gloss(herath_id, "land")  # generic — easy to anchor
        db.upsert_source(id="rando-port", title="Rando port (seed)")
        db.upsert_source(id="test_book", title="Test Book")
        db.add_citation(herath_id, "rando-port", short_quote="seed")
        # The body word 'heath' is itself a canonical etymon. Without
        # the cross-check, fuzzy-search would claim herath ↔ heath.
        db.upsert_etymon("heath", "old-english")
        db.commit()

        result = fuzzy_search_attestations(db, sources_dir, apply=True)

        rows = db.conn.execute(
            "SELECT etymon_id, matched_form FROM etymon_text_match "
            "WHERE etymon_id = ? AND edit_distance > 0",
            (herath_id,),
        ).fetchall()

    assert rows == [], (
        "fuzzy-search must not claim herath as a variant of 'heath' when "
        "'heath' is itself a canonical etymon"
    )
    assert result["written"] == 0


def test_lexicon_review_low_conf_counts_as_declined_not_written(
    fresh_db: Path, tmp_path: Path, monkeypatch
) -> None:
    """When the Tier-2 provider returns confidence='low', the writer
    skips the etymology row (lexicon.py:1394). The CLI must count this
    as declined, not written, so the summary and mining_run audit
    reflect what's actually persisted in the DB.

    Regression test for wyrd-dt9: previously, every accepted result
    incremented `written` even when the writer dropped the row,
    overstating real persistence by ~60% on books where Gemini was
    uncertain."""
    from click.testing import CliRunner

    from wyrd.generators.kenning import cli as cli_mod
    from wyrd.generators.kenning import gemini_extractor
    from wyrd.generators.kenning.llm_extractor import LLMResult
    from wyrd.generators.kenning.skeat_parser import ParsedElement, ParsedEntry

    # Pre-seed: one source, one toponym, one Qwen-tagged low-conf
    # etymology row that's eligible for Tier-2 review.
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "test_book.txt").write_text("Faketon. fake body text.\n")

    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="test_book", title="Test Book")
        toponym_id = db.conn.execute(
            "INSERT INTO toponym (modern_name, region) VALUES (?, ?)",
            ("Faketon", "Testshire"),
        ).lastrowid
        db.conn.execute(
            "INSERT INTO toponym_etymology (toponym_id, source_id, confidence, notes) "
            "VALUES (?, ?, ?, ?)",
            (toponym_id, "test_book", "medium", "extracted_by:llm:qwen3.5:fake"),
        )
        db.commit()

    # Stub the parser so the review path finds Faketon's body.
    fake_parsed = [
        ParsedEntry(
            toponym="Faketon",
            section_suffix=None,
            historical_form="fake-tun",
            elements=[],
            confidence="low",
            source_quote="Faketon. fake body text.",
        ),
    ]
    monkeypatch.setattr(cli_mod, "_select_parser_and_run", lambda text, parser, **_: fake_parsed)

    # Stub Gemini: return a "low"-confidence result with elements
    # present. The writer will silently drop this; the CLI must catch
    # that and count it as declined.
    class FakeClient:
        model = "fake-gemini-model"

        def __init__(self, **_kwargs):
            pass

    monkeypatch.setattr(gemini_extractor, "GeminiClient", FakeClient)
    monkeypatch.setattr(
        gemini_extractor,
        "extract_one",
        lambda client, **kw: LLMResult(
            entry=ParsedEntry(
                toponym="Faketon",
                section_suffix=None,
                historical_form="fake-tun",
                elements=[
                    ParsedElement(form="fake", language="old-english", position="pre"),
                ],
                confidence="low",
                source_quote="extracted_by:gemini:fake-gemini-model | Faketon.",
            ),
            raw_response={},
            failures=[],
            accepted=True,
        ),
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli,
        [
            "lexicon",
            "review",
            str(sources_dir),
            "--db",
            str(fresh_db),
            "--provider",
            "gemini",
            "--apply",
        ],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")

    # The mining_run audit row must say declined=1, accepted=0 — the
    # row was not actually written to toponym_etymology.
    with LexiconDB(fresh_db) as db:
        runs = db.conn.execute(
            "SELECT accepted, declined, rejected FROM mining_run WHERE mode = 'review'"
        ).fetchall()
        gemini_rows = db.conn.execute(
            "SELECT COUNT(*) AS n FROM toponym_etymology WHERE notes LIKE '%gemini%'"
        ).fetchone()
    assert len(runs) == 1
    assert runs[0]["accepted"] == 0, "low-conf should not count as accepted"
    assert runs[0]["declined"] == 1, "low-conf must count as declined"
    assert runs[0]["rejected"] == 0
    assert gemini_rows["n"] == 0, "writer must not have persisted a gemini row"


def test_lexicon_review_dry_run_does_not_persist_or_count_writes(
    fresh_db: Path, tmp_path: Path, monkeypatch
) -> None:
    """Dry-run (no --apply) must not write rows to the DB and must not count
    'would-have-written' results as written. The summary log explicitly says
    '(dry-run; pass --apply to write)' — counting writes anyway would lie.

    Regression for an extraction bug caught on PR #12: an early version of
    the refactor moved `counts['written'] += 1` outside the apply guard, so
    dry-run reported nonzero writes that didn't exist in the DB."""
    from click.testing import CliRunner

    from wyrd.generators.kenning import cli as cli_mod
    from wyrd.generators.kenning import gemini_extractor
    from wyrd.generators.kenning.llm_extractor import LLMResult
    from wyrd.generators.kenning.skeat_parser import ParsedElement, ParsedEntry

    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "test_book.txt").write_text("Faketon. fake body.\n")

    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="test_book", title="Test")
        toponym_id = db.conn.execute(
            "INSERT INTO toponym (modern_name, region) VALUES (?, ?)",
            ("Faketon", "Testshire"),
        ).lastrowid
        db.conn.execute(
            "INSERT INTO toponym_etymology (toponym_id, source_id, confidence, notes) "
            "VALUES (?, ?, ?, ?)",
            (toponym_id, "test_book", "medium", "extracted_by:llm:qwen3.5:fake"),
        )
        db.commit()

    fake_parsed = [
        ParsedEntry(
            toponym="Faketon",
            section_suffix=None,
            historical_form="fake-tun",
            elements=[],
            confidence="low",
            source_quote="Faketon. fake body.",
        ),
    ]
    monkeypatch.setattr(cli_mod, "_select_parser_and_run", lambda text, parser, **_: fake_parsed)

    class FakeClient:
        model = "fake-gemini-model"

        def __init__(self, **_kwargs):
            pass

    monkeypatch.setattr(gemini_extractor, "GeminiClient", FakeClient)
    monkeypatch.setattr(
        gemini_extractor,
        "extract_one",
        lambda client, **kw: LLMResult(
            entry=ParsedEntry(
                toponym="Faketon",
                section_suffix=None,
                historical_form="fake-tun",
                elements=[
                    ParsedElement(form="fake", language="old-english", position="pre"),
                ],
                confidence="high",
                source_quote="extracted_by:gemini:fake | Faketon",
            ),
            raw_response={},
            failures=[],
            accepted=True,
        ),
    )

    runner = CliRunner()
    result = runner.invoke(
        cli_mod.cli,
        [
            "lexicon",
            "review",
            str(sources_dir),
            "--db",
            str(fresh_db),
            "--provider",
            "gemini",
            # NOTE: no --apply — this is the dry-run case under test.
        ],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")

    combined = result.output + (result.stderr or "")
    assert "'written': 0" in combined, "dry-run summary must report written: 0 — got:\n" + combined

    with LexiconDB(fresh_db) as db:
        n_runs = db.conn.execute("SELECT COUNT(*) FROM mining_run").fetchone()[0]
        n_gemini_rows = db.conn.execute(
            "SELECT COUNT(*) FROM toponym_etymology WHERE notes LIKE '%gemini%'"
        ).fetchone()[0]
    assert n_runs == 0
    assert n_gemini_rows == 0


def test_mining_run_table_present_after_init(fresh_db: Path) -> None:
    """init_schema creates mining_run; migrate_schema is a no-op on a fresh DB."""
    with LexiconDB(fresh_db) as db:
        applied = migrate_schema(db)
    assert applied["mining_run_table"] is False
    with LexiconDB(fresh_db) as db:
        tables = {
            r["name"] for r in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert "mining_run" in tables


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


def test_migrate_schema_adds_citation_context_snippet_to_legacy_db(
    fresh_db: Path,
) -> None:
    """A legacy DB whose etymon_citation predates wyrd-9kh.3 picks up the
    new context_snippet column on migrate_schema. Existing rows survive
    with NULL in the new column. Idempotent on a re-run."""
    with LexiconDB(fresh_db) as db:
        # Simulate a pre-9kh.3 schema by dropping the etymon_consensus view
        # (it references etymon_citation, blocking the table swap), recreating
        # etymon_citation without context_snippet, and seeding one legacy row.
        # The migration's _rebuild_etymon_views step will reinstate the view.
        db.conn.execute("DROP VIEW IF EXISTS etymon_consensus")
        db.conn.execute("DROP VIEW IF EXISTS etymon_canonical")
        db.conn.execute("DROP TABLE etymon_citation")
        db.conn.execute(
            """
            CREATE TABLE etymon_citation (
              id          INTEGER PRIMARY KEY AUTOINCREMENT,
              etymon_id   INTEGER NOT NULL REFERENCES etymon(id) ON DELETE CASCADE,
              source_id   TEXT NOT NULL REFERENCES source(id) ON DELETE CASCADE,
              page        TEXT,
              short_quote TEXT
            )
            """
        )
        db.upsert_source(id="src-a", title="A")
        legacy_etymon_id = db.upsert_etymon("ham", "old-english")
        db.conn.execute(
            "INSERT INTO etymon_citation (etymon_id, source_id) VALUES (?, ?)",
            (legacy_etymon_id, "src-a"),
        )
        db.commit()

        # Pre-migration: column is missing.
        cols = {row["name"] for row in db.conn.execute("PRAGMA table_info(etymon_citation)")}
        assert "context_snippet" not in cols

        applied = migrate_schema(db)
        assert applied["etymon_citation.context_snippet"] is True

        # Post-migration: column present, existing row preserved with NULL.
        cols = {row["name"] for row in db.conn.execute("PRAGMA table_info(etymon_citation)")}
        assert "context_snippet" in cols
        row = db.conn.execute(
            "SELECT etymon_id, source_id, context_snippet FROM etymon_citation"
        ).fetchone()
        assert row["etymon_id"] == legacy_etymon_id
        assert row["context_snippet"] is None

        # Idempotent re-run.
        applied2 = migrate_schema(db)
        assert applied2["etymon_citation.context_snippet"] is False


def test_migrate_schema_adds_toponym_etymology_attested_year_to_legacy_db(
    fresh_db: Path,
) -> None:
    """A legacy DB whose toponym_etymology predates wyrd-bag picks up
    the new attested_year column on migrate_schema. Existing rows
    survive with NULL in the new column. Idempotent on a re-run.
    Pins the migration path that fresh_db tests don't exercise (since
    init_schema already includes the column from lexicon.sql)."""
    with LexiconDB(fresh_db) as db:
        # Simulate a pre-wyrd-bag toponym_etymology by re-creating it
        # without attested_year. Drop dependent rows / FKs first to
        # avoid constraint failures, then re-seed one legacy row.
        db.conn.execute("DELETE FROM toponym_etymology_element")
        db.conn.execute("DROP TABLE toponym_etymology")
        db.conn.execute(
            """
            CREATE TABLE toponym_etymology (
              id              INTEGER PRIMARY KEY AUTOINCREMENT,
              toponym_id      INTEGER NOT NULL REFERENCES toponym(id) ON DELETE CASCADE,
              source_id       TEXT NOT NULL REFERENCES source(id) ON DELETE CASCADE,
              page            TEXT,
              historical_form TEXT,
              confidence      TEXT CHECK (confidence IN ('high', 'medium', 'low')),
              notes           TEXT
            )
            """
        )
        db.upsert_source(id="legacy-src", title="Legacy")
        cur = db.conn.execute("INSERT INTO toponym (modern_name) VALUES (?)", ("Tune",))
        toponym_id = cur.lastrowid
        cur = db.conn.execute(
            "INSERT INTO toponym_etymology (toponym_id, source_id, notes) VALUES (?, ?, ?)",
            (toponym_id, "legacy-src", "1086 DB"),
        )
        legacy_te_id = cur.lastrowid
        db.commit()

        # Pre-migration: column is missing.
        cols = {row["name"] for row in db.conn.execute("PRAGMA table_info(toponym_etymology)")}
        assert "attested_year" not in cols

        applied = migrate_schema(db)
        assert applied["toponym_etymology.attested_year"] is True

        # Post-migration: column present, existing row preserved with NULL.
        cols = {row["name"] for row in db.conn.execute("PRAGMA table_info(toponym_etymology)")}
        assert "attested_year" in cols
        row = db.conn.execute(
            "SELECT id, notes, attested_year FROM toponym_etymology WHERE id = ?",
            (legacy_te_id,),
        ).fetchone()
        assert row["notes"] == "1086 DB"
        assert row["attested_year"] is None

        # Idempotent re-run: migration helper reports False.
        applied2 = migrate_schema(db)
        assert applied2["toponym_etymology.attested_year"] is False


def test_add_citation_persists_context_snippet(fresh_db: Path) -> None:
    """LexiconDB.add_citation accepts context_snippet and writes it; a
    NULL-snippet caller stays compatible (existing callers haven't been
    updated yet)."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="mawer_1920", title="Mawer")
        etymon_id = db.upsert_etymon("ham", "old-english")
        db.add_citation(
            etymon_id,
            "mawer_1920",
            context_snippet="Acklington... O.E. Æcceling(a)tun = farm of Æccel.",
        )
        db.add_citation(etymon_id, "mawer_1920", page="42")  # legacy shape, no snippet
        db.commit()

        rows = db.conn.execute(
            "SELECT page, context_snippet FROM etymon_citation WHERE etymon_id = ? ORDER BY id",
            (etymon_id,),
        ).fetchall()

    assert len(rows) == 2
    # First row: snippet captured, no page.
    assert rows[0]["page"] is None
    assert "Acklington" in rows[0]["context_snippet"]
    # Second row: legacy shape with page only — context_snippet is NULL.
    assert rows[1]["page"] == "42"
    assert rows[1]["context_snippet"] is None


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
            "SELECT id, canonical_form, lemma_id, inflection, lemma_method FROM etymon"
        ).fetchall()

    by_id = {r["id"]: r for r in rows}
    assert by_id[cotan_id]["lemma_id"] == cot_id
    assert by_id[cotan_id]["inflection"] == "weak_oblique"
    assert by_id[cotes_id]["lemma_id"] == cot_id
    assert by_id[cotes_id]["inflection"] == "genitive_strong"
    # Method/version stamp is written on every linked row so a future
    # link-lemmas-v2 can selectively rebuild only its predecessor's work.
    assert by_id[cotan_id]["lemma_method"] == "link-lemmas-v1"
    assert by_id[cotes_id]["lemma_method"] == "link-lemmas-v1"
    # Lemma stays its own — no stamp written when not linked.
    assert by_id[cot_id]["lemma_id"] is None
    assert by_id[cot_id]["lemma_method"] is None
    # Unrelated etymon stays unlinked, no stamp.
    assert by_id[ham_id]["lemma_id"] is None
    assert by_id[ham_id]["lemma_method"] is None
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
    seeder is forgotten.

    D18 spelling variants ride on per-language ``<lang>_variants`` keys; those
    are handled by the runtime's load_meanings (split out into Meaning.variants)
    rather than by LANGUAGE_FIELDS, so they're treated as known-handled here."""
    text = resources.files("wyrd.generators.kenning.data").joinpath("meanings.json").read_text()
    data = json.loads(text)

    seen_fields: set[str] = set()
    for subject in data:
        for word in subject.get("words", []):
            seen_fields.update(word.keys())

    handled = LANGUAGE_FIELDS.keys() | NON_LANGUAGE_FIELDS
    # Also tolerate per-language metadata fields emitted by the export step:
    # *_variants (D18 spelling variants), *_inflections (D8 inflection
    # labels), *_citations (scholarly attribution), *_attested_years
    # (D5-2 era cells). They legitimately don't map to a LANGUAGE_FIELDS
    # code; the runtime's load_meanings handles them via separate Meaning
    # attributes.
    metadata_suffixes = ("_variants", "_inflections", "_citations", "_attested_years")
    missing = {
        f for f in seen_fields - handled if not any(f.endswith(s) for s in metadata_suffixes)
    }
    assert not missing, f"Unhandled fields in meanings.json: {missing}"


# --- export_meanings tests --------------------------------------------------


def _seed_subject(
    db: LexiconDB,
    *,
    source_id: str,
    glosses: list[str],
    tags: list[str],
    modifier_type: str | None,
    words: list[dict],
) -> None:
    """Helper: ingest one meanings.json subject via seed_from_meanings."""
    seed_from_meanings(
        db,
        [
            {
                "meaning": glosses,
                "modifier_tags": tags,
                "modifier_type": modifier_type,
                "words": words,
            }
        ],
        source_id,
    )


def test_export_meanings_includes_rando_etymons_with_no_scholar_witnesses(
    fresh_db: Path,
) -> None:
    """A rando-port-only family is exported when include_rando=True (D4 legacy
    seed kept until corroborated)."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["Acorn"],
            tags=["plant", "food"],
            modifier_type="Topographical",
            words=[{"modern_usage": "-ock", "old_english": ["aecern"]}],
        )

        with_rando = export_meanings(db, include_rando=True)
        without_rando = export_meanings(db, include_rando=False)

    assert len(with_rando) == 1
    subj = with_rando[0]
    assert subj["meaning"] == ["Acorn"]
    assert subj["modifier_tags"] == ["food", "plant"]
    assert subj["modifier_type"] == "Topographical"
    assert subj["words"] == [{"modern_usage": "-ock", "old_english": ["aecern"]}]
    assert without_rando == []


def test_export_meanings_promotes_at_three_witnesses(fresh_db: Path) -> None:
    """A family with no rando-port citation needs ≥ min_witnesses to export."""
    with LexiconDB(fresh_db) as db:
        for src in ("a", "b", "c"):
            db.upsert_source(id=src, title=src)
        ham_id = db.upsert_etymon("ham", "old-english", modifier_type="Habitative")
        db.add_gloss(ham_id, "homestead")
        db.add_tag(ham_id, "habitation")
        db.add_citation(ham_id, "a")
        db.add_citation(ham_id, "b")
        db.commit()

        two_witnesses = export_meanings(db, include_rando=False, min_witnesses=3)

        db.add_citation(ham_id, "c")
        db.commit()
        three_witnesses = export_meanings(db, include_rando=False, min_witnesses=3)

    assert two_witnesses == []
    assert len(three_witnesses) == 1
    subj = three_witnesses[0]
    assert subj["meaning"] == ["homestead"]
    # No reflex links in this minimal scenario, so the word is synthesized
    # from the canonical_form. position_pref defaults to no-dash → bare form.
    # All three scholarly sources (a, b, c) ride along as the wyrd-9kh.1
    # `<lang>_citations` sibling field — a GM holding this name in chat
    # sees who attests it.
    assert subj["words"] == [
        {
            "modern_usage": "ham",
            "old_english": ["ham"],
            "old_english_citations": ["a", "b", "c"],
        }
    ]


def test_export_meanings_per_language_thresholds_apply(fresh_db: Path) -> None:
    """Old-Norse promotes at 2 witnesses while Old-English needs 3 — the
    per-language map is honored and the global ``min_witnesses`` is the
    fallback for languages not in the map."""
    with LexiconDB(fresh_db) as db:
        for src in ("a", "b"):
            db.upsert_source(id=src, title=src)
        # OE etymon at 2 witnesses — should NOT promote at OE threshold 3.
        oe = db.upsert_etymon("ham", "old-english")
        db.add_gloss(oe, "homestead")
        db.add_tag(oe, "habitation")
        db.add_citation(oe, "a")
        db.add_citation(oe, "b")
        # ON etymon at 2 witnesses — should promote at ON threshold 2.
        on = db.upsert_etymon("fell", "old-norse")
        db.add_gloss(on, "mountain")
        db.add_tag(on, "topography")
        db.add_citation(on, "a")
        db.add_citation(on, "b")
        db.commit()

        subjects = export_meanings(
            db,
            include_rando=False,
            min_witnesses=3,
            lang_thresholds={"old-english": 3, "old-norse": 2},
        )

    assert len(subjects) == 1
    subj = subjects[0]
    assert subj["meaning"] == ["mountain"]
    # The OE 'ham' must NOT appear; only 'fell' (ON) made the cut.
    forms_seen = {
        form for w in subj["words"] for k, v in w.items() if isinstance(v, list) for form in v
    }
    assert "fell" in forms_seen
    assert "ham" not in forms_seen


def test_export_meanings_default_uses_recommended_preset(fresh_db: Path) -> None:
    """Calling export_meanings() without ``lang_thresholds`` applies the
    RECOMMENDED_LANG_THRESHOLDS preset, so OE gates at 3 and ON gates at 2."""
    assert RECOMMENDED_LANG_THRESHOLDS["old-english"] == 3
    assert RECOMMENDED_LANG_THRESHOLDS["old-norse"] == 2

    with LexiconDB(fresh_db) as db:
        for src in ("a", "b"):
            db.upsert_source(id=src, title=src)
        on = db.upsert_etymon("fell", "old-norse")
        db.add_gloss(on, "mountain")
        db.add_citation(on, "a")
        db.add_citation(on, "b")
        db.commit()
        # No lang_thresholds passed → preset applies → ON at 2 witnesses promotes.
        subjects = export_meanings(db, include_rando=False)

    assert len(subjects) == 1
    assert subjects[0]["meaning"] == ["mountain"]


def test_export_meanings_empty_lang_thresholds_falls_back_to_uniform(
    fresh_db: Path,
) -> None:
    """Passing ``lang_thresholds={}`` disables the preset — every language
    uses the global ``min_witnesses`` uniformly."""
    with LexiconDB(fresh_db) as db:
        for src in ("a", "b"):
            db.upsert_source(id=src, title=src)
        on = db.upsert_etymon("fell", "old-norse")
        db.add_gloss(on, "mountain")
        db.add_citation(on, "a")
        db.add_citation(on, "b")
        db.commit()
        # ON has 2 witnesses; preset would promote it but {} disables that
        # and applies min_witnesses=3 uniformly → not promoted.
        subjects = export_meanings(db, include_rando=False, min_witnesses=3, lang_thresholds={})
    assert subjects == []


def test_export_meanings_rolls_inflected_variants_into_lemma(fresh_db: Path) -> None:
    """`cot` + linked inflections export as one entry; the language form list
    contains the lemma followed by its variants (D8)."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["cottage"],
            tags=["habitation"],
            modifier_type="Habitative",
            words=[{"modern_usage": "-cot", "old_english": ["cot"]}],
        )
        # Add inflected children — these point at the lemma via lemma_id.
        cot_id = db.conn.execute("SELECT id FROM etymon WHERE canonical_form = 'cot'").fetchone()[
            "id"
        ]
        cotan_id = db.upsert_etymon("cotan", "old-english")
        cotes_id = db.upsert_etymon("cotes", "old-english")
        db.conn.execute(
            "UPDATE etymon SET lemma_id = ? WHERE id IN (?, ?)", (cot_id, cotan_id, cotes_id)
        )
        db.commit()

        subjects = export_meanings(db, include_rando=True)

    assert len(subjects) == 1
    word = subjects[0]["words"][0]
    assert word["modern_usage"] == "-cot"
    # Lemma form leads; variants follow in the same language array.
    assert word["old_english"][0] == "cot"
    assert set(word["old_english"]) == {"cot", "cotan", "cotes"}


def _add_text_match(db, etymon_id, source_id, matched_form, count, method="fuzzy-search-v1"):
    """Helper: insert a row directly into etymon_text_match for D18 fixture
    setup. The schema doesn't ship a write helper because the post-mining
    pipeline owns this table."""
    db.conn.execute(
        "INSERT INTO etymon_text_match "
        "(etymon_id, source_id, matched_form, match_count, method, edit_distance) "
        "VALUES (?, ?, ?, ?, ?, 1)",
        (etymon_id, source_id, matched_form, count, method),
    )


def test_export_meanings_emits_variant_pool_per_language(fresh_db: Path) -> None:
    """An etymon with text-match rows whose matched_form differs from the
    canonical_form gets a `<lang>_variants` field populated in its word
    entry, ordered by descending weight (most-attested first)."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        db.upsert_source(id="src-a", title="A")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["bridge"],
            tags=["water"],
            modifier_type="Topographical",
            words=[{"modern_usage": "Bridg-", "old_english": ["brycg"]}],
        )
        brycg_id = db.conn.execute(
            "SELECT id FROM etymon WHERE canonical_form = 'brycg'"
        ).fetchone()["id"]
        _add_text_match(db, brycg_id, "src-a", "brigge", 5)
        # Use reverse-search method on the singleton so it survives the
        # low-confidence-singleton filter (which drops count=1 fuzzy /
        # llm-disambiguated rows as the dominant OCR-noise class).
        _add_text_match(db, brycg_id, "src-a", "brige", 1, method="reverse-search-v1")
        # Same matched_form from a different source: weights sum.
        _add_text_match(db, brycg_id, "rando-port", "brigge", 3)
        db.commit()

        subjects = export_meanings(db, include_rando=True)

    word = subjects[0]["words"][0]
    assert "old_english_variants" in word
    # Sorted by descending weight; brigge (5+3=8) leads brige (1).
    assert word["old_english_variants"] == [
        {"form": "brigge", "weight": 8},
        {"form": "brige", "weight": 1},
    ]


def test_export_meanings_dedupes_variant_against_canonical_forms(
    fresh_db: Path,
) -> None:
    """A matched_form that's already a canonical form of any family member
    gets dropped from the variant pool — emitting it again would just
    duplicate what's already in the language array."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        db.upsert_source(id="src-a", title="A")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["bridge"],
            tags=["water"],
            modifier_type="Topographical",
            words=[{"modern_usage": "Bridg-", "old_english": ["brycg"]}],
        )
        brycg_id = db.conn.execute(
            "SELECT id FROM etymon WHERE canonical_form = 'brycg'"
        ).fetchone()["id"]
        # 'BRYCG' (case-insensitive match against canonical) should be dropped.
        _add_text_match(db, brycg_id, "src-a", "BRYCG", 100)
        # 'brigge' is genuinely different and should survive.
        _add_text_match(db, brycg_id, "src-a", "brigge", 5)
        db.commit()

        subjects = export_meanings(db, include_rando=True)

    word = subjects[0]["words"][0]
    forms = [v["form"] for v in word.get("old_english_variants", [])]
    assert "brigge" in forms
    assert "BRYCG" not in forms
    assert "brycg" not in forms


def test_export_meanings_aggregates_variants_across_descendants(
    fresh_db: Path,
) -> None:
    """A reflex linked to a lemma surfaces variants from the lemma AND its
    inflected descendants (D8 rollup composes with D18 variant pools).
    Weights for the same matched_form across descendants sum together."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        db.upsert_source(id="src-a", title="A")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["cottage"],
            tags=["habitation"],
            modifier_type="Habitative",
            words=[{"modern_usage": "-cot", "old_english": ["cot"]}],
        )
        cot_id = db.conn.execute("SELECT id FROM etymon WHERE canonical_form = 'cot'").fetchone()[
            "id"
        ]
        cotan_id = db.upsert_etymon("cotan", "old-english")
        cotes_id = db.upsert_etymon("cotes", "old-english")
        db.conn.execute(
            "UPDATE etymon SET lemma_id = ? WHERE id IN (?, ?)",
            (cot_id, cotan_id, cotes_id),
        )
        # The lemma and its inflections each produce 'cotte' as a variant —
        # weights should sum.
        _add_text_match(db, cot_id, "src-a", "cotte", 4)
        _add_text_match(db, cotan_id, "src-a", "cotte", 3)
        _add_text_match(db, cotes_id, "rando-port", "cotte", 2)
        db.commit()

        subjects = export_meanings(db, include_rando=True)

    word = subjects[0]["words"][0]
    variants = word["old_english_variants"]
    assert variants == [{"form": "cotte", "weight": 9}]


def test_emit_variant_list_sorts_by_descending_weight_then_form(
    fresh_db: Path,
) -> None:
    """`_emit_variant_list` sorts (form, weight) pairs by descending weight,
    breaking ties on form alphabetically — both axes matter for diff
    stability across rebuilds."""
    from wyrd.generators.kenning.lexicon import _emit_variant_list

    out = _emit_variant_list({"alpha": 5, "beta": 5, "gamma": 7, "delta": 1})
    assert out == [
        {"form": "gamma", "weight": 7},
        {"form": "alpha", "weight": 5},
        {"form": "beta", "weight": 5},
        {"form": "delta", "weight": 1},
    ]


def test_emit_variant_list_empty_input_returns_empty(fresh_db: Path) -> None:
    from wyrd.generators.kenning.lexicon import _emit_variant_list

    assert _emit_variant_list({}) == []


def test_export_meanings_emits_inflection_pool_per_language(fresh_db: Path) -> None:
    """An etymon family with inflected children gets a `<lang>_inflections`
    field listing the (form, label) pairs for each inflected member.
    The lemma form does NOT appear in the inflections list — it's the
    unmarked headword."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["cottage"],
            tags=["habitation"],
            modifier_type="Habitative",
            words=[{"modern_usage": "-cot", "old_english": ["cot"]}],
        )
        cot_id = db.conn.execute("SELECT id FROM etymon WHERE canonical_form = 'cot'").fetchone()[
            "id"
        ]
        cotum_id = db.upsert_etymon("cotum", "old-english")
        cotan_id = db.upsert_etymon("cotan", "old-english")
        db.conn.execute(
            "UPDATE etymon SET lemma_id = ?, inflection = ? WHERE id = ?",
            (cot_id, "dative_or_pl", cotum_id),
        )
        db.conn.execute(
            "UPDATE etymon SET lemma_id = ?, inflection = ? WHERE id = ?",
            (cot_id, "weak_oblique", cotan_id),
        )
        db.commit()

        subjects = export_meanings(db, include_rando=True)

    word = subjects[0]["words"][0]
    assert "old_english_inflections" in word
    inflections = word["old_english_inflections"]
    forms = {entry["form"]: entry["inflection"] for entry in inflections}
    assert forms == {"cotum": "dative_or_pl", "cotan": "weak_oblique"}
    # cot (the lemma) doesn't appear in the inflection pool.
    assert "cot" not in {entry["form"] for entry in inflections}


def test_export_meanings_synthesized_family_emits_inflections(fresh_db: Path) -> None:
    """A family without any linked reflex (synthesized-word path) emits
    inflection metadata too. _build_words_for_group has separate aggregation
    blocks for the reflex-linked and synthesized branches; this test pins
    the synthesized side so a regression there can't slip past the
    reflex-only test above."""
    with LexiconDB(fresh_db) as db:
        for src in ("a", "b", "c"):
            db.upsert_source(id=src, title=src)
        # No rando-port reflex — hits the families_without_reflex branch.
        cot_id = db.upsert_etymon("cot", "old-english", modifier_type="Habitative")
        db.add_gloss(cot_id, "cottage")
        db.add_tag(cot_id, "habitation")
        for src in ("a", "b", "c"):
            db.add_citation(cot_id, src)
        cotum_id = db.upsert_etymon("cotum", "old-english")
        db.conn.execute(
            "UPDATE etymon SET lemma_id = ?, inflection = ? WHERE id = ?",
            (cot_id, "dative_or_pl", cotum_id),
        )
        db.commit()

        subjects = export_meanings(db, include_rando=False, min_witnesses=3)

    word = subjects[0]["words"][0]
    assert word["modern_usage"] == "cot"  # synthesized from canonical_form
    assert word["old_english_inflections"] == [{"form": "cotum", "inflection": "dative_or_pl"}]


def test_emit_inflection_list_sorts_by_form_alphabetically(fresh_db: Path) -> None:
    """`_emit_inflection_list` sorts entries by form so output is byte-stable
    regardless of dict insertion order."""
    from wyrd.generators.kenning.lexicon import _emit_inflection_list

    out = _emit_inflection_list(
        {"cotum": "dative_or_pl", "cotan": "weak_oblique", "cotes": "genitive_strong"}
    )
    assert out == [
        {"form": "cotan", "inflection": "weak_oblique"},
        {"form": "cotes", "inflection": "genitive_strong"},
        {"form": "cotum", "inflection": "dative_or_pl"},
    ]


def test_emit_inflection_list_empty_input_returns_empty(fresh_db: Path) -> None:
    from wyrd.generators.kenning.lexicon import _emit_inflection_list

    assert _emit_inflection_list({}) == []


# --- wyrd-8tp: gloss + variant data-quality filters -----------------------


def test_filter_concatenation_glosses_drops_comma_concat():
    """Glosses that are entirely a concatenation of already-present
    singleton glosses should be filtered."""
    from wyrd.generators.kenning.lexicon import _filter_concatenation_glosses

    out = _filter_concatenation_glosses(["lake", "pond", "lake, pond"])
    assert out == ["lake", "pond"]


def test_filter_concatenation_glosses_keeps_partial_overlap():
    """A multi-token gloss that contains a singleton plus extra info should
    be kept — only the strict-subset case is considered redundant."""
    from wyrd.generators.kenning.lexicon import _filter_concatenation_glosses

    # 'shore' is not in the singleton set, so 'bank, shore' is informative.
    out = _filter_concatenation_glosses(["bank", "bank, shore"])
    assert out == ["bank", "bank, shore"]


def test_filter_concatenation_glosses_recognizes_semicolon_and_or():
    """Splitter accepts ',', ';' and ' or ' as connectors."""
    from wyrd.generators.kenning.lexicon import _filter_concatenation_glosses

    out = _filter_concatenation_glosses(["book", "charter", "charter; book", "book or charter"])
    assert out == ["book", "charter"]


def test_filter_concatenation_glosses_keeps_singletons_unconditionally():
    """A single-token gloss is never dropped, even if its tokens overlap
    other singletons (since by definition it has only one token)."""
    from wyrd.generators.kenning.lexicon import _filter_concatenation_glosses

    out = _filter_concatenation_glosses(["bank", "edge", "shore"])
    assert out == ["bank", "edge", "shore"]


def test_export_meanings_filters_low_confidence_singleton_variants(
    fresh_db: Path,
) -> None:
    """A variant with match_count=1 from fuzzy-search-v1 alone should be
    dropped (the dominant OCR-noise class). One with match_count>=2 OR
    any high-confidence method survives."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        db.upsert_source(id="src-a", title="A")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["sharp"],
            tags=[],
            modifier_type=None,
            words=[{"modern_usage": "-scearp", "old_english": ["scearu"]}],
        )
        scearu_id = db.conn.execute(
            "SELECT id FROM etymon WHERE canonical_form = 'scearu'"
        ).fetchone()["id"]
        # OCR-noise: count=1, fuzzy-search-v1. Should be dropped.
        _add_text_match(db, scearu_id, "src-a", "saearp", 1, method="fuzzy-search-v1")
        # OCR-noise: count=1, llm-disambiguated-v1. Same low-confidence class;
        # locks in that the second member of _LOW_CONFIDENCE_METHODS is honored.
        _add_text_match(db, scearu_id, "src-a", "scrarp", 1, method="llm-disambiguated-v1")
        # Legitimate variant: count=2 from fuzzy-search-v1 — survives because
        # ≥2 attestations gives more confidence.
        _add_text_match(db, scearu_id, "src-a", "scerp", 2, method="fuzzy-search-v1")
        # Legitimate variant: count=1 but reverse-search-v1 (canonical-form
        # body scan) is high-confidence. Survives.
        _add_text_match(db, scearu_id, "src-a", "scearp-form", 1, method="reverse-search-v1")
        # Mixed-methods row: count=1 split between fuzzy-search-v1 AND
        # reverse-search-v1 from different sources. methods set is NOT a
        # subset of low-confidence (reverse-search is high-confidence), so
        # the form survives. This pins the issubset check vs a simpler
        # "all-fuzzy" equality test.
        db.upsert_source(id="src-b", title="B")
        _add_text_match(db, scearu_id, "src-a", "scearp-mix", 1, method="fuzzy-search-v1")
        _add_text_match(db, scearu_id, "src-b", "scearp-mix", 1, method="reverse-search-v1")
        db.commit()

        subjects = export_meanings(db, include_rando=True)

    word = subjects[0]["words"][0]
    forms = [v["form"] for v in word.get("old_english_variants", [])]
    # Low-confidence singletons dropped.
    assert "saearp" not in forms
    assert "scrarp" not in forms
    # ≥2-count fuzzy survives.
    assert "scerp" in forms
    # Pure high-confidence singleton survives.
    assert "scearp-form" in forms
    # Mixed-methods singleton survives because not all methods are low-confidence.
    assert "scearp-mix" in forms


def test_export_meanings_drops_concatenation_glosses_in_subjects(
    fresh_db: Path,
) -> None:
    """A subject whose family carries 'lake', 'pond', AND the redundant
    'lake, pond' should emit only the singletons in its `meaning` list."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        # Manually inject etymon + glosses for the test, since _seed_subject
        # runs through the seed_from_meanings path which deduplicates.
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["lake", "pond"],
            tags=[],
            modifier_type=None,
            words=[{"modern_usage": "-mere", "old_english": ["mere"]}],
        )
        mere_id = db.conn.execute("SELECT id FROM etymon WHERE canonical_form = 'mere'").fetchone()[
            "id"
        ]
        # Add a redundant concatenation gloss directly.
        db.conn.execute(
            "INSERT OR IGNORE INTO etymon_gloss (etymon_id, gloss) VALUES (?, ?)",
            (mere_id, "lake, pond"),
        )
        db.commit()

        subjects = export_meanings(db, include_rando=True)

    # The mere subject's meaning list should have 'lake' and 'pond' but not
    # 'lake, pond'.
    found = next(s for s in subjects if any(w["modern_usage"] == "-mere" for w in s["words"]))
    assert "lake" in found["meaning"]
    assert "pond" in found["meaning"]
    assert "lake, pond" not in found["meaning"]


def test_export_meanings_omits_merge_loser_as_separate_subject(
    fresh_db: Path,
) -> None:
    """An OCR-cluster loser doesn't become its own subject — its form rides
    along in the canonical's language list (D22)."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["valley"],
            tags=["topography"],
            modifier_type="Topographical",
            words=[{"modern_usage": "-den", "old_english": ["denu"]}],
        )
        denu_id = db.conn.execute("SELECT id FROM etymon WHERE canonical_form = 'denu'").fetchone()[
            "id"
        ]
        # Loser: an OCR variant pointing into denu.
        dene_id = db.upsert_etymon("dene", "old-english")
        db.conn.execute("UPDATE etymon SET merged_into_id = ? WHERE id = ?", (denu_id, dene_id))
        db.commit()

        subjects = export_meanings(db, include_rando=True)

    assert len(subjects) == 1
    forms = subjects[0]["words"][0]["old_english"]
    assert forms[0] == "denu"  # canonical leads
    assert "dene" in forms  # loser rides along


def test_export_meanings_synthesizes_word_for_mined_only_family(
    fresh_db: Path,
) -> None:
    """A family with ≥3 witnesses but no linked reflex gets a synthesized
    `words` entry from its canonical_form."""
    with LexiconDB(fresh_db) as db:
        for src in ("a", "b", "c"):
            db.upsert_source(id=src, title=src)
        tune_id = db.upsert_etymon("tune", "old-english", modifier_type="Habitative")
        db.add_gloss(tune_id, "settlement")
        db.add_tag(tune_id, "habitation")
        for src in ("a", "b", "c"):
            db.add_citation(tune_id, src)
        db.commit()

        subjects = export_meanings(db, include_rando=False, min_witnesses=3)

    assert len(subjects) == 1
    assert subjects[0]["words"] == [
        {
            "modern_usage": "tune",
            "old_english": ["tune"],
            "old_english_citations": ["a", "b", "c"],
        }
    ]


def test_export_meanings_synthesize_path_emits_attested_years(
    fresh_db: Path,
) -> None:
    """The synth-without-linked-reflex code path (mined-only families,
    no rando-port reflex) needs its own attested-year coverage. The
    structure differs from the linked-reflex path (flat
    member_form_by_id walk vs descendant walk), so a regression here
    wouldn't be caught by the reflex-path tests in the wyrd-bag set.

    Pin: a tune-id mined etymon with three citations + an ETM row
    carrying attested_year=1086 surfaces the year in the synthesized
    word's `<lang>_attested_years` field."""
    with LexiconDB(fresh_db) as db:
        for src in ("a", "b", "c"):
            db.upsert_source(id=src, title=src)
        tune_id = db.upsert_etymon("tune", "old-english", modifier_type="Habitative")
        db.add_gloss(tune_id, "settlement")
        db.add_tag(tune_id, "habitation")
        for src in ("a", "b", "c"):
            db.add_citation(tune_id, src)
        db.conn.execute(
            "INSERT INTO etymon_text_match (etymon_id, source_id, matched_form, "
            "match_count, edit_distance, attested_year) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (tune_id, "a", "tune", 1, 0, 1086),
        )
        db.commit()

        subjects = export_meanings(db, include_rando=False, min_witnesses=3)

    word = subjects[0]["words"][0]
    assert word["old_english_attested_years"] == [{"form": "tune", "year": 1086}]


# --- wyrd-9kh.1: per-language citation attribution -----------------------


def test_export_meanings_filters_rando_port_from_citations(fresh_db: Path) -> None:
    """The rando-port seed isn't a real scholarly attribution — it's the
    Wikipedia-derived legacy bootstrap. Filter it from `<lang>_citations`
    so a rando-only word emits no citations field at all (rather than a
    confusing 'cited by rando-port')."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["acorn"],
            tags=["plant"],
            modifier_type=None,
            words=[{"modern_usage": "-ock", "old_english": ["aecern"]}],
        )
        subjects = export_meanings(db, include_rando=True)

    word = subjects[0]["words"][0]
    assert "old_english" in word
    assert "old_english_citations" not in word


def test_export_meanings_emits_citations_for_scholarly_witnesses(fresh_db: Path) -> None:
    """Mixed scholarly + rando-port citations: only the scholars surface in
    `<lang>_citations`, sorted alphabetically. Set semantics dedupe across
    the descendant walk."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        for src in ("mawer_1920", "skeat_1901"):
            db.upsert_source(id=src, title=src)
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["bridge"],
            tags=["water"],
            modifier_type="Topographical",
            words=[{"modern_usage": "Bridg-", "old_english": ["brycg"]}],
        )
        brycg_id = db.conn.execute(
            "SELECT id FROM etymon WHERE canonical_form = 'brycg'"
        ).fetchone()["id"]
        # Two scholars + a duplicate of mawer (ON CONFLICT DO NOTHING dedupes).
        db.add_citation(brycg_id, "mawer_1920")
        db.add_citation(brycg_id, "skeat_1901")
        db.add_citation(brycg_id, "mawer_1920")
        db.commit()
        subjects = export_meanings(db, include_rando=True)

    word = subjects[0]["words"][0]
    assert word["old_english_citations"] == ["mawer_1920", "skeat_1901"]


def test_export_meanings_emits_attested_years_from_etymon_text_match(
    fresh_db: Path,
) -> None:
    """wyrd-bag: ``<lang>_attested_years`` is sourced from the rollup
    of ``etymon_text_match.attested_year`` (PR #47) and
    ``toponym_etymology.attested_year`` (PR #53). This test exercises
    the ETM path: a scholar-cited etymon with one text-match row
    carrying year 1086 → bundle entry ``[{form: brycg, year: 1086}]``."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        db.upsert_source(id="mawer_1920", title="Mawer")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["bridge"],
            tags=["water"],
            modifier_type="Topographical",
            words=[{"modern_usage": "Bridg-", "old_english": ["brycg"]}],
        )
        brycg_id = db.conn.execute(
            "SELECT id FROM etymon WHERE canonical_form = 'brycg'"
        ).fetchone()["id"]
        db.add_citation(brycg_id, "mawer_1920")
        db.conn.execute(
            "INSERT INTO etymon_text_match (etymon_id, source_id, matched_form, "
            "match_count, edit_distance, attested_year) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (brycg_id, "mawer_1920", "brycg", 1, 0, 1086),
        )
        db.commit()
        subjects = export_meanings(db, include_rando=True)

    word = subjects[0]["words"][0]
    assert word["old_english_attested_years"] == [{"form": "brycg", "year": 1086}]


def test_export_meanings_emits_attested_years_from_toponym_etymology(
    fresh_db: Path,
) -> None:
    """wyrd-bag: the toponym_etymology path. A toponym whose etymology
    cites this etymon as one element, and whose row has attested_year
    populated, surfaces the year in the bundle."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        db.upsert_source(id="mawer_1920", title="Mawer")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["enclosure"],
            tags=["habitation"],
            modifier_type="Habitative",
            words=[{"modern_usage": "-tun", "old_english": ["tun"]}],
        )
        tun_id = db.conn.execute("SELECT id FROM etymon WHERE canonical_form = 'tun'").fetchone()[
            "id"
        ]
        db.add_citation(tun_id, "mawer_1920")
        # Set up a toponym + etymology + element pointing at tun, with
        # attested_year on the etymology row.
        cur = db.conn.execute("INSERT INTO toponym (modern_name) VALUES (?)", ("Town",))
        toponym_id = cur.lastrowid
        cur = db.conn.execute(
            "INSERT INTO toponym_etymology "
            "(toponym_id, source_id, historical_form, confidence, notes, attested_year) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (toponym_id, "mawer_1920", "tūn", "high", "Tune 1086 DB", 1086),
        )
        te_id = cur.lastrowid
        db.conn.execute(
            "INSERT INTO toponym_etymology_element "
            "(toponym_etymology_id, ordinal, etymon_id) VALUES (?, ?, ?)",
            (te_id, 0, tun_id),
        )
        db.commit()
        subjects = export_meanings(db, include_rando=True)

    word = subjects[0]["words"][0]
    assert word["old_english_attested_years"] == [{"form": "tun", "year": 1086}]


def test_export_meanings_attested_years_picks_earliest_across_sources(
    fresh_db: Path,
) -> None:
    """An etymon attested via BOTH etymon_text_match and toponym_etymology
    surfaces the EARLIEST year — pin via SQL MIN(year) so a future
    refactor that drops the rollup surfaces immediately."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        db.upsert_source(id="mawer_1920", title="Mawer")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["enclosure"],
            tags=["habitation"],
            modifier_type="Habitative",
            words=[{"modern_usage": "-tun", "old_english": ["tun"]}],
        )
        tun_id = db.conn.execute("SELECT id FROM etymon WHERE canonical_form = 'tun'").fetchone()[
            "id"
        ]
        db.add_citation(tun_id, "mawer_1920")
        # ETM row attests year 1242. Toponym_etymology attests 1086.
        # MIN should win.
        db.conn.execute(
            "INSERT INTO etymon_text_match (etymon_id, source_id, matched_form, "
            "match_count, edit_distance, attested_year) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (tun_id, "mawer_1920", "tun", 1, 0, 1242),
        )
        cur = db.conn.execute("INSERT INTO toponym (modern_name) VALUES (?)", ("Town",))
        toponym_id = cur.lastrowid
        cur = db.conn.execute(
            "INSERT INTO toponym_etymology "
            "(toponym_id, source_id, confidence, notes, attested_year) "
            "VALUES (?, ?, ?, ?, ?)",
            (toponym_id, "mawer_1920", "high", "1086 DB", 1086),
        )
        te_id = cur.lastrowid
        db.conn.execute(
            "INSERT INTO toponym_etymology_element "
            "(toponym_etymology_id, ordinal, etymon_id) VALUES (?, ?, ?)",
            (te_id, 0, tun_id),
        )
        db.commit()
        subjects = export_meanings(db, include_rando=True)

    word = subjects[0]["words"][0]
    # 1086 wins over 1242.
    assert word["old_english_attested_years"] == [{"form": "tun", "year": 1086}]


def test_export_meanings_omits_attested_years_field_when_no_year_evidence(
    fresh_db: Path,
) -> None:
    """A morpheme with no attested year on either row source has no
    ``<lang>_attested_years`` field in its emitted word — keeps the
    bundle clean and lets the runtime --era filter pass it through."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["wood"],
            tags=["topography"],
            modifier_type="Topographical",
            words=[{"modern_usage": "-wood", "old_english": ["wudu"]}],
        )
        subjects = export_meanings(db, include_rando=True)

    word = subjects[0]["words"][0]
    assert "old_english_attested_years" not in word


def test_export_meanings_attested_years_sorted_by_year_ascending(
    fresh_db: Path,
) -> None:
    """The emit list is sorted by year ascending so the runtime --era
    filter (D5-2) can short-circuit on the first match. Ties broken
    alphabetically by form."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        db.upsert_source(id="mawer_1920", title="Mawer")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["enclosure"],
            tags=["habitation"],
            modifier_type="Habitative",
            words=[{"modern_usage": "-tun", "old_english": ["tun"]}],
        )
        tun_id = db.conn.execute("SELECT id FROM etymon WHERE canonical_form = 'tun'").fetchone()[
            "id"
        ]
        db.add_citation(tun_id, "mawer_1920")
        # Three inflected variants, intentionally inserted in non-
        # chronological order. Each has its own etymon row + ETM row
        # with attested_year. The lemma_id chains them under tun.
        for form, year in [("tunna", 1340), ("tun", 1086), ("tunes", 1242)]:
            child_id = db.upsert_etymon(form, "old-english")
            db.conn.execute(
                "UPDATE etymon SET lemma_id = ? WHERE id = ?",
                (tun_id, child_id),
            )
            db.conn.execute(
                "INSERT INTO etymon_text_match (etymon_id, source_id, matched_form, "
                "match_count, edit_distance, attested_year) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (child_id, "mawer_1920", form, 1, 0, year),
            )
        db.commit()
        subjects = export_meanings(db, include_rando=True)

    word = subjects[0]["words"][0]
    # Sorted by year ascending; tun(1086) beats tunes(1242) beats tunna(1340).
    assert word["old_english_attested_years"] == [
        {"form": "tun", "year": 1086},
        {"form": "tunes", "year": 1242},
        {"form": "tunna", "year": 1340},
    ]


def test_export_meanings_citations_dedupe_across_descendants(fresh_db: Path) -> None:
    """A reflex linked to a lemma surfaces citations from the lemma AND its
    OCR-cluster losers / inflected children — the same scholar cited on a
    descendant shouldn't double in the bundle output."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        for src in ("mawer_1920", "skeat_1901"):
            db.upsert_source(id=src, title=src)
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["valley"],
            tags=["topography"],
            modifier_type="Topographical",
            words=[{"modern_usage": "-den", "old_english": ["denu"]}],
        )
        denu_id = db.conn.execute("SELECT id FROM etymon WHERE canonical_form = 'denu'").fetchone()[
            "id"
        ]
        # OCR-cluster loser: another spelling of the same etymon merged into denu.
        loser_id = db.upsert_etymon("denū", "old-english")
        db.conn.execute("UPDATE etymon SET merged_into_id = ? WHERE id = ?", (denu_id, loser_id))
        # mawer cites both the canonical and the OCR-loser (because it
        # extracted from a body where the OCR garbled the form). skeat
        # cites only the canonical.
        db.add_citation(denu_id, "mawer_1920")
        db.add_citation(denu_id, "skeat_1901")
        db.add_citation(loser_id, "mawer_1920")
        db.commit()
        subjects = export_meanings(db, include_rando=True)

    word = subjects[0]["words"][0]
    # mawer appears once despite citing two etymons in the family.
    assert word["old_english_citations"] == ["mawer_1920", "skeat_1901"]


def test_export_meanings_synthesizes_with_position_pref(fresh_db: Path) -> None:
    """When position_pref is set, the synthesized usage carries the matching
    dash markers (`pre`/`post`/`inner`)."""
    with LexiconDB(fresh_db) as db:
        for src in ("a", "b", "c"):
            db.upsert_source(id=src, title=src)
        ham_id = db.upsert_etymon(
            "ham", "old-english", modifier_type="Habitative", position_pref="post"
        )
        db.add_gloss(ham_id, "homestead")
        for src in ("a", "b", "c"):
            db.add_citation(ham_id, src)
        db.commit()

        subjects = export_meanings(db, include_rando=False, min_witnesses=3)

    assert subjects[0]["words"][0]["modern_usage"] == "-ham"


def test_export_meanings_synthesizes_pre_position_with_trailing_dash(
    fresh_db: Path,
) -> None:
    """position_pref='pre' produces 'foo-' usage."""
    with LexiconDB(fresh_db) as db:
        for src in ("a", "b", "c"):
            db.upsert_source(id=src, title=src)
        e_id = db.upsert_etymon("ada", "old-english", position_pref="pre")
        db.add_gloss(e_id, "Ada (Name)")
        for src in ("a", "b", "c"):
            db.add_citation(e_id, src)
        db.commit()
        subjects = export_meanings(db, include_rando=False, min_witnesses=3)
    assert subjects[0]["words"][0]["modern_usage"] == "ada-"


def test_export_meanings_synthesizes_inner_position_with_dashes_both_sides(
    fresh_db: Path,
) -> None:
    """position_pref='inner' produces '-foo-' usage."""
    with LexiconDB(fresh_db) as db:
        for src in ("a", "b", "c"):
            db.upsert_source(id=src, title=src)
        e_id = db.upsert_etymon("ar", "old-english", position_pref="inner")
        db.add_gloss(e_id, "of the")
        for src in ("a", "b", "c"):
            db.add_citation(e_id, src)
        db.commit()
        subjects = export_meanings(db, include_rando=False, min_witnesses=3)
    assert subjects[0]["words"][0]["modern_usage"] == "-ar-"


def test_export_meanings_includes_orphan_reflex_subjects(fresh_db: Path) -> None:
    """A reflex with no linked etymon (rando seed of {'modern_usage': 'Adam-'}
    or {'celtic_mix': [], 'modern_usage': 'Bre-'}) still gets exported as its
    own subject so per-culture proportions can resolve the usage."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        # Subject with empty language slots — produces an orphan reflex.
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["Adam (Name)"],
            tags=["male name"],
            modifier_type="Habitative",
            words=[{"modern_usage": "Adam-"}],
        )
        subjects = export_meanings(db, include_rando=True)

    orphan_words = [
        w["modern_usage"]
        for s in subjects
        for w in s["words"]
        if not any(k != "modern_usage" for k in w)
    ]
    assert "Adam-" in orphan_words


def test_export_meanings_orphan_reflexes_skipped_when_include_rando_false(
    fresh_db: Path,
) -> None:
    """Orphan reflexes are rando-port-only data; --no-include-rando drops them."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=[],
            tags=[],
            modifier_type=None,
            words=[{"modern_usage": "Adam-"}],
        )
        subjects = export_meanings(db, include_rando=False)
    assert subjects == []


def test_export_meanings_groups_multiple_families_into_one_subject(
    fresh_db: Path,
) -> None:
    """Two distinct families with the same (modifier_type, glosses, tags)
    signature collapse into one subject with multiple words. This exercises
    the dict-based grouping in _group_families_into_subjects."""
    with LexiconDB(fresh_db) as db:
        for src in ("a", "b", "c", "d", "e", "f"):
            db.upsert_source(id=src, title=src)
        # Two distinct lemmas, same gloss/tag/modifier_type signature.
        # Each gets a separate reflex linked to it.
        cot_id = db.upsert_etymon(
            "cot", "old-english", modifier_type="Habitative", position_pref="post"
        )
        ham_id = db.upsert_etymon(
            "ham", "old-english", modifier_type="Habitative", position_pref="post"
        )
        for e_id in (cot_id, ham_id):
            db.add_gloss(e_id, "homestead")
            db.add_tag(e_id, "habitation")
        for src in ("a", "b", "c"):
            db.add_citation(cot_id, src)
        for src in ("d", "e", "f"):
            db.add_citation(ham_id, src)
        db.commit()

        subjects = export_meanings(db, include_rando=False, min_witnesses=3)

    # Single subject because both families share the same signature.
    assert len(subjects) == 1
    subj = subjects[0]
    assert subj["meaning"] == ["homestead"]
    assert subj["modifier_tags"] == ["habitation"]
    usages = sorted(w["modern_usage"] for w in subj["words"])
    assert usages == ["-cot", "-ham"]


def test_export_meanings_reflex_linked_to_inflection_only_emits_that_inflection(
    fresh_db: Path,
) -> None:
    """A reflex linked directly to an inflection (not the lemma) emits only
    that inflection's form plus the inflection's own descendants — not the
    lemma's form or sibling inflections. The descendant walk is downward,
    not upward; if the seed wanted the reflex to cover the whole morpheme,
    it should have linked to the lemma."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        # Seed with reflex linked to inflected form 'cotan' rather than lemma 'cot'.
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["cottage"],
            tags=["habitation"],
            modifier_type="Habitative",
            words=[{"modern_usage": "-cotan", "old_english": ["cotan"]}],
        )
        # Add the lemma + sibling inflection as separate entities later.
        cotan_id = db.conn.execute(
            "SELECT id FROM etymon WHERE canonical_form = 'cotan'"
        ).fetchone()["id"]
        cot_id = db.upsert_etymon("cot", "old-english", modifier_type="Habitative")
        cotes_id = db.upsert_etymon("cotes", "old-english")
        # Wire D8 lemma linkage: cotan and cotes both inflect from cot.
        db.conn.execute(
            "UPDATE etymon SET lemma_id = ? WHERE id IN (?, ?)",
            (cot_id, cotan_id, cotes_id),
        )
        db.commit()

        subjects = export_meanings(db, include_rando=True)

    # All three etymons roll up to cot (the lemma). The subject reconstruction
    # groups by (modifier_type, glosses, tags); here the seed gave glosses
    # only to cotan, so cot/cotes have no glosses → they form a separate
    # subject (no overlap). Find the subject with the -cotan reflex.
    word = next(w for s in subjects for w in s["words"] if w["modern_usage"] == "-cotan")
    # Reflex linked only to cotan; descendants of cotan = {cotan} (no children
    # below). Lemma 'cot' and sibling 'cotes' must NOT appear here.
    assert word["old_english"] == ["cotan"]


def test_export_meanings_narrows_word_language_to_linked_etymons(
    fresh_db: Path,
) -> None:
    """A subject grouping a Celtic family and an Old English family must NOT
    bleed forms across reflexes: '-don' (linked to OE) carries old_english
    only; 'Bre-' (linked to Celtic) carries celtic_mix only. Without per-
    reflex narrowing the explainer would falsely attribute 'don' to Celtic
    and vice versa (wyrd-mzl regression guard)."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["Hill"],
            tags=["topography"],
            modifier_type="Topographical",
            words=[
                {"modern_usage": "Bre-", "celtic_mix": ["bre"]},
                {"modern_usage": "-don", "old_english": ["dun"]},
            ],
        )
        subjects = export_meanings(db, include_rando=True)

    assert len(subjects) == 1
    by_usage = {w["modern_usage"]: w for w in subjects[0]["words"]}
    assert "celtic_mix" in by_usage["Bre-"]
    assert "old_english" not in by_usage["Bre-"]
    assert "old_english" in by_usage["-don"]
    assert "celtic_mix" not in by_usage["-don"]


def test_export_meanings_round_trips_through_load_meanings(fresh_db: Path) -> None:
    """A seeded subject exports to a structure the runtime can re-load via
    load_meanings — closes the authoring↔runtime loop end-to-end."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["Hill"],
            tags=["topography"],
            modifier_type="Topographical",
            words=[
                {"modern_usage": "Bre-", "celtic_mix": ["bre", "bryn"]},
                {"modern_usage": "-don", "old_english": ["dun"]},
            ],
        )
        subjects = export_meanings(db, include_rando=True)

    meaning_db, tag_db = load_meanings(subjects)
    assert "Bre-" in meaning_db
    assert "-don" in meaning_db
    assert any("topography" in m.tags for m in meaning_db["Bre-"])
    assert "topography" in tag_db


def test_export_meanings_cli_writes_file(fresh_db: Path, tmp_path: Path) -> None:
    """CLI command writes JSON to --output path and reports the subject count."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["Hill"],
            tags=["topography"],
            modifier_type="Topographical",
            words=[{"modern_usage": "-hill", "old_english": ["hyll"]}],
        )

    out_path = tmp_path / "meanings.json"
    result = CliRunner().invoke(
        kenning_cli,
        [
            "lexicon",
            "export-meanings",
            "--db",
            str(fresh_db),
            "--output",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Wrote 1 subjects" in result.stderr
    payload = json.loads(out_path.read_text())
    assert len(payload) == 1
    assert payload[0]["meaning"] == ["Hill"]
    assert payload[0]["words"][0]["modern_usage"] == "-hill"


def test_export_meanings_cli_emits_to_stdout(fresh_db: Path) -> None:
    """Without --output, the CLI emits the JSON payload to stdout."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="rando-port", title="rando")
        _seed_subject(
            db,
            source_id="rando-port",
            glosses=["Hill"],
            tags=["topography"],
            modifier_type="Topographical",
            words=[{"modern_usage": "-hill", "old_english": ["hyll"]}],
        )

    result = CliRunner().invoke(
        kenning_cli,
        ["lexicon", "export-meanings", "--db", str(fresh_db)],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert any(s["meaning"] == ["Hill"] for s in payload)


def test_export_meanings_cli_lang_threshold_flag(fresh_db: Path) -> None:
    """--lang-threshold LANG=N overrides the preset for a specific language."""
    with LexiconDB(fresh_db) as db:
        for src in ("a", "b"):
            db.upsert_source(id=src, title=src)
        # Two ON sources for 'fell'; with --lang-threshold old-norse=3 it
        # falls under the gate even though the preset would have promoted it.
        on = db.upsert_etymon("fell", "old-norse")
        db.add_gloss(on, "mountain")
        db.add_citation(on, "a")
        db.add_citation(on, "b")
        db.commit()

    # Without override (preset has ON at 2): promotes.
    r1 = CliRunner().invoke(
        kenning_cli,
        ["lexicon", "export-meanings", "--db", str(fresh_db), "--no-include-rando"],
    )
    assert r1.exit_code == 0, r1.output
    promoted = any(s["meaning"] == ["mountain"] for s in json.loads(r1.stdout))
    assert promoted

    # With --lang-threshold old-norse=3: doesn't promote.
    r2 = CliRunner().invoke(
        kenning_cli,
        [
            "lexicon",
            "export-meanings",
            "--db",
            str(fresh_db),
            "--no-include-rando",
            "--lang-threshold",
            "old-norse=3",
        ],
    )
    assert r2.exit_code == 0, r2.output
    promoted = any(s["meaning"] == ["mountain"] for s in json.loads(r2.stdout))
    assert not promoted


def test_export_meanings_cli_lang_threshold_accepts_json_field_alias(
    fresh_db: Path,
) -> None:
    """--lang-threshold accepts the JSON-field alias ('old_scandinavian') and
    normalizes to the lexicon's canonical code ('old-norse'). Without
    normalization, --lang-threshold old_scandinavian=3 would silently miss
    every ON consensus row (since the column stores 'old-norse')."""
    with LexiconDB(fresh_db) as db:
        for src in ("a", "b"):
            db.upsert_source(id=src, title=src)
        on = db.upsert_etymon("fell", "old-norse")
        db.add_gloss(on, "mountain")
        db.add_citation(on, "a")
        db.add_citation(on, "b")
        db.commit()

    result = CliRunner().invoke(
        kenning_cli,
        [
            "lexicon",
            "export-meanings",
            "--db",
            str(fresh_db),
            "--no-include-rando",
            "--lang-threshold",
            "old_scandinavian=3",
        ],
    )
    assert result.exit_code == 0, result.output
    # Threshold 3 with only 2 witnesses → not promoted.
    promoted = any(s["meaning"] == ["mountain"] for s in json.loads(result.stdout))
    assert not promoted


def test_export_meanings_cli_no_preset_uses_uniform_min_witnesses(fresh_db: Path) -> None:
    """--no-preset clears the preset; the global --min-witnesses applies uniformly."""
    with LexiconDB(fresh_db) as db:
        for src in ("a", "b"):
            db.upsert_source(id=src, title=src)
        on = db.upsert_etymon("fell", "old-norse")
        db.add_gloss(on, "mountain")
        db.add_citation(on, "a")
        db.add_citation(on, "b")
        db.commit()

    result = CliRunner().invoke(
        kenning_cli,
        [
            "lexicon",
            "export-meanings",
            "--db",
            str(fresh_db),
            "--no-include-rando",
            "--no-preset",
            "--min-witnesses",
            "3",
        ],
    )
    assert result.exit_code == 0, result.output
    promoted = any(s["meaning"] == ["mountain"] for s in json.loads(result.stdout))
    assert not promoted


def test_export_meanings_cli_rejects_malformed_lang_threshold(fresh_db: Path) -> None:
    """--lang-threshold must be LANG=N; bad input fails fast with a clear message."""
    result = CliRunner().invoke(
        kenning_cli,
        [
            "lexicon",
            "export-meanings",
            "--db",
            str(fresh_db),
            "--lang-threshold",
            "old-norse-2",  # missing '='
        ],
    )
    assert result.exit_code != 0
    assert "LANG=N" in result.output


def test_export_meanings_cli_rejects_non_integer_threshold(fresh_db: Path) -> None:
    """The N in --lang-threshold LANG=N must be an integer; non-int fails clearly."""
    result = CliRunner().invoke(
        kenning_cli,
        [
            "lexicon",
            "export-meanings",
            "--db",
            str(fresh_db),
            "--lang-threshold",
            "old-norse=foo",
        ],
    )
    assert result.exit_code != 0
    assert "must be an integer" in result.output


def test_export_meanings_cli_rejects_empty_lang_or_threshold(fresh_db: Path) -> None:
    """--lang-threshold =3 or old-norse= give a clearer 'non-empty' error
    rather than the misleading 'must be an integer' message you'd get from
    int('') alone."""
    for spec in ("=3", "old-norse=", " = "):
        result = CliRunner().invoke(
            kenning_cli,
            [
                "lexicon",
                "export-meanings",
                "--db",
                str(fresh_db),
                "--lang-threshold",
                spec,
            ],
        )
        assert result.exit_code != 0, f"spec={spec!r} should have failed"
        assert "non-empty" in result.output, f"spec={spec!r}: {result.output}"


def test_export_meanings_cli_no_preset_with_lang_threshold_overrides(
    fresh_db: Path,
) -> None:
    """--no-preset --lang-threshold old-norse=2 → only ON gates at 2; everything
    else uses --min-witnesses uniformly. Pins the dict-merge ordering so a
    bug where the empty {} got the preset re-applied wouldn't slip through."""
    with LexiconDB(fresh_db) as db:
        for src in ("a", "b"):
            db.upsert_source(id=src, title=src)
        # ON at 2 witnesses — should promote via the explicit override.
        on = db.upsert_etymon("fell", "old-norse")
        db.add_gloss(on, "mountain")
        db.add_citation(on, "a")
        db.add_citation(on, "b")
        # Celtic at 2 witnesses — should NOT promote (preset would have said
        # celtic=2, but --no-preset cleared that, so --min-witnesses=3 applies).
        cel = db.upsert_etymon("bryn", "celtic")
        db.add_gloss(cel, "hill")
        db.add_citation(cel, "a")
        db.add_citation(cel, "b")
        db.commit()

    result = CliRunner().invoke(
        kenning_cli,
        [
            "lexicon",
            "export-meanings",
            "--db",
            str(fresh_db),
            "--no-include-rando",
            "--no-preset",
            "--min-witnesses",
            "3",
            "--lang-threshold",
            "old-norse=2",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    glosses = {tuple(s["meaning"]) for s in payload}
    assert ("mountain",) in glosses
    assert ("hill",) not in glosses


def test_build_witness_filter_empty_uses_uniform_threshold() -> None:
    """Direct unit test: empty lang_thresholds returns the simple fragment.
    Parenthesized to match the per-language path's contract so the helper
    composes safely when dropped into a larger expression."""
    from wyrd.generators.kenning.lexicon import _build_witness_filter

    sql, params = _build_witness_filter({}, 3)
    assert sql == "(witnesses >= ?)"
    assert params == [3]


def test_build_witness_filter_emits_clauses_in_sorted_order() -> None:
    """Direct unit test: with multiple lang_thresholds, params land in
    sorted-by-lang order (so output is byte-stable regardless of dict
    insertion order), and the trailing fallback NOT IN (...) clause carries
    the global min_witnesses."""
    from wyrd.generators.kenning.lexicon import _build_witness_filter

    # Insertion order intentionally not sorted, to verify the helper sorts.
    sql, params = _build_witness_filter({"old-norse": 2, "celtic": 2, "old-english": 3}, 4)
    # SQL fragment uses ? placeholders; param order is what's deterministic.
    # Per-language clauses interleave (lang, threshold) in sorted order,
    # then the NOT IN list (sorted langs), then the fallback threshold.
    assert params == [
        "celtic",
        2,
        "old-english",
        3,
        "old-norse",
        2,
        "celtic",
        "old-english",
        "old-norse",
        4,
    ]
    # Three per-language clauses + one NOT IN fallback clause.
    assert sql.count("language = ?") == 3
    assert "language NOT IN" in sql
    assert sql.count("witnesses >= ?") == 4


def test_export_meanings_output_is_byte_stable_across_runs(fresh_db: Path) -> None:
    """Same DB → same JSON bytes. AUTOINCREMENT root_ids and SQLite's
    iteration order can shuffle results between fresh seeds; the export must
    sort comprehensively enough to defeat that."""
    with LexiconDB(fresh_db) as db:
        for src in ("a", "b", "c"):
            db.upsert_source(id=src, title=src)
        # Two families with the same signature — exercises tie-breaking in
        # _group_families_into_subjects' sort key.
        for form in ("cot", "ham"):
            e_id = db.upsert_etymon(
                form, "old-english", modifier_type="Habitative", position_pref="post"
            )
            db.add_gloss(e_id, "homestead")
            db.add_tag(e_id, "habitation")
            for src in ("a", "b", "c"):
                db.add_citation(e_id, src)
        db.commit()
        first = json.dumps(export_meanings(db, include_rando=False, min_witnesses=3))
        second = json.dumps(export_meanings(db, include_rando=False, min_witnesses=3))
    assert first == second


# --- D27 / wyrd-0ug: etymon_descent + cognate-cluster schema --------------------


def test_init_schema_creates_etymon_descent_table(fresh_db: Path) -> None:
    """Fresh install carries the etymon_descent table from data/lexicon.sql
    so the migration and fresh-install paths stay in lockstep."""
    with LexiconDB(fresh_db) as db:
        tables = {
            row["name"]
            for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert "etymon_descent" in tables


def test_init_schema_etymon_has_cognate_columns(fresh_db: Path) -> None:
    """Fresh install carries cognate_id + cognate_method on etymon."""
    with LexiconDB(fresh_db) as db:
        cols = {row["name"] for row in db.conn.execute("PRAGMA table_info(etymon)")}
    assert "cognate_id" in cols
    assert "cognate_method" in cols


def test_init_schema_creates_etymon_descent_indexes(fresh_db: Path) -> None:
    """Both directional indexes on etymon_descent must exist so
    'descendants of X' and 'ancestors of Y' queries are O(degree) not
    O(rows)."""
    with LexiconDB(fresh_db) as db:
        indexes = {
            row["name"]
            for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
        }
    assert "idx_etymon_descent_parent" in indexes
    assert "idx_etymon_descent_child" in indexes
    assert "idx_etymon_cognate" in indexes


def test_etymon_descent_round_trip_insert_and_query(fresh_db: Path) -> None:
    """Round-trip: insert a Proto-Germanic → Old English inheritance edge
    and retrieve it cleanly. Pins the column shape (parent_id, child_id,
    edge_type, source_id, confidence, notes) so a future schema tweak
    surfaces here."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="wiktionary", title="Wiktionary")
        proto_id = db.upsert_etymon("*tūnaz", "proto-germanic")
        oe_id = db.upsert_etymon("tūn", "old-english")
        db.conn.execute(
            "INSERT INTO etymon_descent "
            "(parent_id, child_id, edge_type, source_id, confidence, notes) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (proto_id, oe_id, "inheritance", "wiktionary", "high", "via {{inh}}"),
        )
        db.commit()
        row = db.conn.execute(
            "SELECT parent_id, child_id, edge_type, source_id, confidence, notes "
            "FROM etymon_descent WHERE parent_id = ?",
            (proto_id,),
        ).fetchone()

    assert row["parent_id"] == proto_id
    assert row["child_id"] == oe_id
    assert row["edge_type"] == "inheritance"
    assert row["source_id"] == "wiktionary"
    assert row["confidence"] == "high"
    assert row["notes"] == "via {{inh}}"


def test_etymon_descent_unique_per_parent_child_edge_source(fresh_db: Path) -> None:
    """The same (parent, child, edge_type, source) tuple deduplicates,
    so re-mining Wiktionary doesn't double-insert. Different edge_types
    or different sources for the same (parent, child) are SEPARATE rows
    — distinct evidence."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="wiktionary", title="Wiktionary")
        db.upsert_source(id="other", title="Other source")
        proto_id = db.upsert_etymon("*tūnaz", "proto-germanic")
        oe_id = db.upsert_etymon("tūn", "old-english")

        def insert(edge_type: str, source: str) -> None:
            db.conn.execute(
                "INSERT OR IGNORE INTO etymon_descent "
                "(parent_id, child_id, edge_type, source_id) VALUES (?, ?, ?, ?)",
                (proto_id, oe_id, edge_type, source),
            )

        insert("inheritance", "wiktionary")
        insert("inheritance", "wiktionary")  # dedupes
        insert("inheritance", "other")  # different source → separate row
        insert("borrowing", "wiktionary")  # different edge_type → separate row
        db.commit()

        n = db.conn.execute(
            "SELECT COUNT(*) AS n FROM etymon_descent WHERE parent_id = ?",
            (proto_id,),
        ).fetchone()["n"]
    assert n == 3


def test_etymon_descent_check_constraint_rejects_unknown_edge_type(
    fresh_db: Path,
) -> None:
    """The CHECK constraint on edge_type must reject typos / made-up
    values so a misspelled wiktextract template kind doesn't silently
    poison the graph."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="wiktionary", title="Wiktionary")
        a_id = db.upsert_etymon("a", "old-english")
        b_id = db.upsert_etymon("b", "old-english")
        with pytest.raises(sqlite3.IntegrityError):
            db.conn.execute(
                "INSERT INTO etymon_descent "
                "(parent_id, child_id, edge_type, source_id) "
                "VALUES (?, ?, ?, ?)",
                (a_id, b_id, "made-up-kind", "wiktionary"),
            )


def test_etymon_consensus_does_not_count_descent_as_witness(
    fresh_db: Path,
) -> None:
    """Critical contract pin: descent edges are a different axis of
    evidence than extraction citations (D4). A morpheme with one
    extraction citation but ten descent edges must still report
    witnesses=1 — otherwise the descent graph would inflate the
    promotion threshold and let unverified morphemes 'promote' just
    because Wiktionary attests their etymology chain."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="wiktionary", title="Wiktionary")
        db.upsert_source(id="mawer_1920", title="Mawer")
        oe_id = db.upsert_etymon("tūn", "old-english")
        # One extraction witness from a real scholar.
        db.add_citation(oe_id, "mawer_1920")
        # Many descent edges — these MUST NOT count toward witnesses.
        for parent_form in ("*tūnaz", "*deuh₂-", "*dūnom"):
            parent_id = db.upsert_etymon(parent_form, "proto-germanic")
            db.conn.execute(
                "INSERT INTO etymon_descent "
                "(parent_id, child_id, edge_type, source_id) "
                "VALUES (?, ?, ?, ?)",
                (parent_id, oe_id, "inheritance", "wiktionary"),
            )
        db.commit()
        consensus_row = db.conn.execute(
            "SELECT witnesses FROM etymon_consensus WHERE canonical_form = 'tūn'"
        ).fetchone()
    assert consensus_row["witnesses"] == 1


def test_etymon_descent_descendants_query_walks_inheritance_chain(
    fresh_db: Path,
) -> None:
    """The reference query 'modern descendants of <root>' walks
    inheritance + borrowing edges via a recursive CTE. Pins the SQL so
    a schema tweak (column renames, edge type changes) surfaces here.

    Graph: *tūnaz → tūn (OE) → town (modern English)
                  → tún (ON) → tún (modern Icelandic)"""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="wiktionary", title="Wiktionary")
        proto_id = db.upsert_etymon("*tūnaz", "proto-germanic")
        oe_id = db.upsert_etymon("tūn", "old-english")
        on_id = db.upsert_etymon("tún", "old-norse")
        me_id = db.upsert_etymon("town", "modern-english")
        is_id = db.upsert_etymon("tún", "icelandic")

        for parent, child in (
            (proto_id, oe_id),
            (proto_id, on_id),
            (oe_id, me_id),
            (on_id, is_id),
        ):
            db.conn.execute(
                "INSERT INTO etymon_descent "
                "(parent_id, child_id, edge_type, source_id) "
                "VALUES (?, ?, 'inheritance', 'wiktionary')",
                (parent, child),
            )
        db.commit()

        # Recursive CTE — proposed reference query for INGESTION.md.
        rows = db.conn.execute(
            """
            WITH RECURSIVE descendants(id) AS (
              SELECT child_id FROM etymon_descent
                WHERE parent_id = ? AND edge_type IN ('inheritance', 'borrowing')
              UNION
              SELECT d.child_id FROM etymon_descent d
                JOIN descendants ON d.parent_id = descendants.id
                WHERE d.edge_type IN ('inheritance', 'borrowing')
            )
            SELECT e.canonical_form, e.language
            FROM descendants
            JOIN etymon e ON e.id = descendants.id
            ORDER BY e.language, e.canonical_form
            """,
            (proto_id,),
        ).fetchall()

    forms = [(row["canonical_form"], row["language"]) for row in rows]
    assert ("tún", "icelandic") in forms
    assert ("town", "modern-english") in forms
    assert ("tūn", "old-english") in forms
    assert ("tún", "old-norse") in forms
    # Root itself should NOT appear in descendants (the seed row is
    # parent_id = proto_id, but we select child_id).
    assert ("*tūnaz", "proto-germanic") not in forms


def test_etymon_descent_ancestors_query_walks_inheritance_chain_upward(
    fresh_db: Path,
) -> None:
    """Symmetric pin for the 'ancestors of X' recursive CTE (documented
    in DECISIONS.md D27 and INGESTION.md). Without this, a regression
    that drops idx_etymon_descent_child or breaks the upward walk would
    only surface in production. Same graph as the descendants test:

        *tūnaz → tūn (OE) → town (modern English)
               → tún (ON) → tún (modern Icelandic)

    Walking UP from modern Icelandic 'tún' should reach ON 'tún' and
    Proto-Germanic '*tūnaz' (but NOT modern English 'town' — that's
    a sibling-via-different-branch, not an ancestor)."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="wiktionary", title="Wiktionary")
        proto_id = db.upsert_etymon("*tūnaz", "proto-germanic")
        oe_id = db.upsert_etymon("tūn", "old-english")
        on_id = db.upsert_etymon("tún", "old-norse")
        me_id = db.upsert_etymon("town", "modern-english")
        is_id = db.upsert_etymon("tún", "icelandic")
        for parent, child in (
            (proto_id, oe_id),
            (proto_id, on_id),
            (oe_id, me_id),
            (on_id, is_id),
        ):
            db.conn.execute(
                "INSERT INTO etymon_descent "
                "(parent_id, child_id, edge_type, source_id) "
                "VALUES (?, ?, 'inheritance', 'wiktionary')",
                (parent, child),
            )
        db.commit()

        # Recursive CTE — reference query for INGESTION.md / DECISIONS.md.
        rows = db.conn.execute(
            """
            WITH RECURSIVE ancestors(id) AS (
              SELECT parent_id FROM etymon_descent
                WHERE child_id = ? AND edge_type IN ('inheritance', 'borrowing')
              UNION
              SELECT d.parent_id FROM etymon_descent d
                JOIN ancestors ON d.child_id = ancestors.id
                WHERE d.edge_type IN ('inheritance', 'borrowing')
            )
            SELECT e.canonical_form, e.language
            FROM ancestors
            JOIN etymon e ON e.id = ancestors.id
            ORDER BY e.language, e.canonical_form
            """,
            (is_id,),
        ).fetchall()

    forms = [(row["canonical_form"], row["language"]) for row in rows]
    assert ("tún", "old-norse") in forms
    assert ("*tūnaz", "proto-germanic") in forms
    # Sibling branch — must NOT appear in ancestors of Icelandic tún.
    assert ("town", "modern-english") not in forms
    assert ("tūn", "old-english") not in forms
    # Self should not appear either.
    assert ("tún", "icelandic") not in forms


def test_etymon_descent_cascades_when_parent_deleted(fresh_db: Path) -> None:
    """ON DELETE CASCADE on etymon_descent.parent_id — when an etymon row
    is deleted, every descent edge pointing at it as parent goes away
    too. Without this guard, deleting a Proto-* root would leave dangling
    edges with broken parent_id references. Pin so a regression to
    SET NULL or constraint removal surfaces here."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="wiktionary", title="Wiktionary")
        proto_id = db.upsert_etymon("*tūnaz", "proto-germanic")
        oe_id = db.upsert_etymon("tūn", "old-english")
        on_id = db.upsert_etymon("tún", "old-norse")
        for child in (oe_id, on_id):
            db.conn.execute(
                "INSERT INTO etymon_descent "
                "(parent_id, child_id, edge_type, source_id) "
                "VALUES (?, ?, 'inheritance', 'wiktionary')",
                (proto_id, child),
            )
        db.commit()
        before = db.conn.execute("SELECT COUNT(*) AS n FROM etymon_descent").fetchone()["n"]
        assert before == 2

        # Delete the parent etymon — both descent edges should vanish.
        db.conn.execute("DELETE FROM etymon WHERE id = ?", (proto_id,))
        db.commit()

        after = db.conn.execute("SELECT COUNT(*) AS n FROM etymon_descent").fetchone()["n"]
    assert after == 0, (
        "ON DELETE CASCADE on parent_id failed; got dangling descent edges. "
        "Check the FK declaration in data/lexicon.sql + the migration helper."
    )


def test_etymon_descent_cascades_when_child_deleted(fresh_db: Path) -> None:
    """ON DELETE CASCADE on etymon_descent.child_id — symmetric pin to
    the parent variant. Required so deleting a misclassified etymon
    doesn't leave orphan edges pointing at a non-existent child_id."""
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="wiktionary", title="Wiktionary")
        proto_id = db.upsert_etymon("*tūnaz", "proto-germanic")
        oe_id = db.upsert_etymon("tūn", "old-english")
        on_id = db.upsert_etymon("tún", "old-norse")
        for child in (oe_id, on_id):
            db.conn.execute(
                "INSERT INTO etymon_descent "
                "(parent_id, child_id, edge_type, source_id) "
                "VALUES (?, ?, 'inheritance', 'wiktionary')",
                (proto_id, child),
            )
        db.commit()

        # Delete the OE child — only the proto→oe edge should vanish;
        # proto→on must survive.
        db.conn.execute("DELETE FROM etymon WHERE id = ?", (oe_id,))
        db.commit()

        rows = db.conn.execute(
            "SELECT child_id FROM etymon_descent WHERE parent_id = ?",
            (proto_id,),
        ).fetchall()
    remaining = [row["child_id"] for row in rows]
    assert remaining == [on_id]


def test_migrate_schema_adds_etymon_descent_to_legacy_db(fresh_db: Path) -> None:
    """A pre-D27 DB without etymon_descent picks it up on migrate_schema.
    Verifies (1) the table is created, (2) ON DELETE CASCADE survives
    the migration DDL — not just the data/lexicon.sql DDL the fresh-
    install cascade tests cover. Catches a drift between the two
    parallel CREATE TABLE statements in lexicon.py vs lexicon.sql.
    Idempotent on re-run."""
    with LexiconDB(fresh_db) as db:
        # Simulate a pre-D27 schema by dropping the descent table.
        db.conn.execute("DROP TABLE IF EXISTS etymon_descent")
        # Pre-migration: table is missing.
        tables = {
            row["name"]
            for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "etymon_descent" not in tables

        applied = migrate_schema(db)
        assert applied["etymon_descent_table"] is True

        tables = {
            row["name"]
            for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "etymon_descent" in tables

        # Cascade behavior must survive the migration-path DDL. PRAGMA
        # foreign_key_list reports each FK's on_delete action, so we can
        # assert both parent_id and child_id report 'CASCADE' without
        # having to seed-and-delete (cheaper, more direct).
        fks = list(db.conn.execute("PRAGMA foreign_key_list(etymon_descent)"))
        on_delete = {fk["from"]: fk["on_delete"] for fk in fks}
        assert on_delete.get("parent_id") == "CASCADE", (
            f"migration-path DDL dropped ON DELETE CASCADE on parent_id; got {on_delete!r}"
        )
        assert on_delete.get("child_id") == "CASCADE", (
            f"migration-path DDL dropped ON DELETE CASCADE on child_id; got {on_delete!r}"
        )

        # Idempotent re-run.
        applied2 = migrate_schema(db)
        assert applied2["etymon_descent_table"] is False


def test_migrate_schema_adds_etymon_cognate_columns_to_legacy_db(
    fresh_db: Path,
) -> None:
    """A pre-D27 etymon table without cognate_id / cognate_method picks
    them up on migrate_schema. Existing rows survive with NULL in the
    new columns."""
    with LexiconDB(fresh_db) as db:
        # Drop the columns by recreating the table without them. Simpler
        # to drop and recreate the etymon table from a minimal pre-D27
        # shape; FK targets are restored by the consensus view rebuild.
        db.upsert_source(id="src-a", title="A")
        legacy_id = db.upsert_etymon("ham", "old-english")
        db.commit()

        # Pre-migration check: post-init the columns ARE there (fresh
        # install). Drop them to simulate a legacy DB.
        # Note: `CREATE TABLE etymon_legacy AS SELECT ...` silently drops
        # the original PK / UNIQUE / FK / NOT NULL constraints — the
        # resulting table is a content snapshot only. That's enough for
        # this test (which only checks that the synset_* columns are
        # ADDED on migration), but it's not a faithful simulation of a
        # production pre-D27 schema. A FK-cascade test or constraint
        # check on this fixture would silently pass.
        db.conn.execute("DROP VIEW IF EXISTS etymon_consensus")
        db.conn.execute("DROP VIEW IF EXISTS etymon_canonical")
        db.conn.execute(
            "CREATE TABLE etymon_legacy AS "
            "SELECT id, canonical_form, language, modifier_type, position_pref, "
            "       notes, lemma_id, inflection, lemma_method, merged_into_id "
            "FROM etymon"
        )
        db.conn.execute("DROP TABLE etymon")
        db.conn.execute("ALTER TABLE etymon_legacy RENAME TO etymon")

        cols = {row["name"] for row in db.conn.execute("PRAGMA table_info(etymon)")}
        assert "cognate_id" not in cols
        assert "cognate_method" not in cols

        applied = migrate_schema(db)
        assert applied["etymon.cognate_id"] is True
        assert applied["etymon.cognate_method"] is True
        assert applied["idx_etymon_cognate"] is True

        cols = {row["name"] for row in db.conn.execute("PRAGMA table_info(etymon)")}
        assert "cognate_id" in cols
        assert "cognate_method" in cols

        # Existing row survives with NULL in the new columns.
        row = db.conn.execute(
            "SELECT id, cognate_id, cognate_method FROM etymon WHERE id = ?",
            (legacy_id,),
        ).fetchone()
        assert row["cognate_id"] is None
        assert row["cognate_method"] is None

        # Idempotent re-run.
        applied2 = migrate_schema(db)
        assert applied2["etymon.cognate_id"] is False
        assert applied2["etymon.cognate_method"] is False


# --- D27 / wyrd-81n: cluster-cognates enrichment pass ------------------


def _seed_descent_chain(db: LexiconDB, *edges: tuple[str, str, str]) -> dict[str, int]:
    """Helper: insert a list of (parent_form, child_form, edge_type)
    descent edges, upserting the etymon rows as needed. Returns a
    {form: id} map for the test to reference. All etymons are written
    with language='proto-germanic' for fixture simplicity — the cluster
    pass walks edges, not languages."""
    db.upsert_source(id="wiktionary", title="Wiktionary")
    forms: dict[str, int] = {}
    for parent_form, child_form, _edge in edges:
        for form in (parent_form, child_form):
            if form not in forms:
                forms[form] = db.upsert_etymon(form, "proto-germanic")
    for parent_form, child_form, edge_type in edges:
        db.conn.execute(
            "INSERT INTO etymon_descent "
            "(parent_id, child_id, edge_type, source_id) VALUES (?, ?, ?, ?)",
            (forms[parent_form], forms[child_form], edge_type, "wiktionary"),
        )
    db.commit()
    return forms


def test_cluster_cognates_assigns_root_id_to_inheritance_chain(fresh_db: Path) -> None:
    """Happy path: a → b → c inheritance chain. All three rows get
    cognate_id = a.id, cognate_method = 'cluster-cognates-v1'."""
    from wyrd.generators.kenning.lexicon import cluster_cognates

    with LexiconDB(fresh_db) as db:
        forms = _seed_descent_chain(
            db,
            ("a", "b", "inheritance"),
            ("b", "c", "inheritance"),
        )
        result = cluster_cognates(db, apply=True)
        rows = {
            row["canonical_form"]: row
            for row in db.conn.execute(
                "SELECT canonical_form, cognate_id, cognate_method FROM etymon"
            )
        }

    assert result["roots"] == 1
    assert result["candidates"] == 3
    assert result["applied"] is True
    for form in ("a", "b", "c"):
        assert rows[form]["cognate_id"] == forms["a"]
        assert rows[form]["cognate_method"] == "cluster-cognates-v1"


def test_cluster_cognates_dry_run_does_not_write(fresh_db: Path) -> None:
    """apply=False reports counts but leaves cognate_id NULL on every
    candidate row."""
    from wyrd.generators.kenning.lexicon import cluster_cognates

    with LexiconDB(fresh_db) as db:
        _seed_descent_chain(
            db,
            ("a", "b", "inheritance"),
            ("b", "c", "inheritance"),
        )
        result = cluster_cognates(db, apply=False)
        cognate_ids = [
            row["cognate_id"] for row in db.conn.execute("SELECT cognate_id FROM etymon")
        ]

    assert result["candidates"] == 3
    assert result["applied"] is False
    assert all(sid is None for sid in cognate_ids)


def test_cluster_cognates_separate_roots_get_separate_cognate_clusters(fresh_db: Path) -> None:
    """Two unrelated roots produce two distinct cognate_id values; rows
    in each cluster point at their own root."""
    from wyrd.generators.kenning.lexicon import cluster_cognates

    with LexiconDB(fresh_db) as db:
        forms = _seed_descent_chain(
            db,
            ("a", "b", "inheritance"),
            ("x", "y", "inheritance"),
        )
        result = cluster_cognates(db, apply=True)
        rows = {
            row["canonical_form"]: row["cognate_id"]
            for row in db.conn.execute("SELECT canonical_form, cognate_id FROM etymon")
        }

    assert result["roots"] == 2
    assert rows["a"] == forms["a"]
    assert rows["b"] == forms["a"]
    assert rows["x"] == forms["x"]
    assert rows["y"] == forms["x"]
    assert rows["a"] != rows["x"]


def test_cluster_cognates_cognate_edge_does_not_bridge_cognate_clusters(fresh_db: Path) -> None:
    """The 'cognate' edge_type is a peer relation, NOT a chain — it must
    not bridge cognate-cluster assignments. Pin the contract: a {{cog}}-only
    relation between two etymons leaves both with NULL cognate_id (no
    inheritance/borrowing chain to walk)."""
    from wyrd.generators.kenning.lexicon import cluster_cognates

    with LexiconDB(fresh_db) as db:
        # Only a cognate edge — no inheritance.
        forms = _seed_descent_chain(
            db,
            ("a", "b", "cognate"),
        )
        cluster_cognates(db, apply=True)
        rows = {
            row["canonical_form"]: row["cognate_id"]
            for row in db.conn.execute("SELECT canonical_form, cognate_id FROM etymon")
        }

    assert rows["a"] is None
    assert rows["b"] is None
    # The forms map should still have both etymons, just with no
    # cognate_id assignment because no bridging edge exists.
    assert "a" in forms and "b" in forms


def test_cluster_cognates_borrowing_edges_bridge_cognate_clusters(fresh_db: Path) -> None:
    """Borrowing edges DO bridge — a borrowed word is part of the
    borrowing language's cognate set in practice (D27 / wyrd-81n
    documents this choice)."""
    from wyrd.generators.kenning.lexicon import cluster_cognates

    with LexiconDB(fresh_db) as db:
        forms = _seed_descent_chain(
            db,
            ("source-form", "borrowed-form", "borrowing"),
        )
        cluster_cognates(db, apply=True)
        rows = {
            row["canonical_form"]: row["cognate_id"]
            for row in db.conn.execute("SELECT canonical_form, cognate_id FROM etymon")
        }

    assert rows["source-form"] == forms["source-form"]
    assert rows["borrowed-form"] == forms["source-form"]


def test_cluster_cognates_diamond_graph_picks_smallest_root(
    fresh_db: Path,
) -> None:
    """When a descendant is reachable from multiple roots (rare; happens
    when scholars disagree on the chain), the smallest root id wins.
    Pin the determinism so re-runs are bit-stable."""
    from wyrd.generators.kenning.lexicon import cluster_cognates

    with LexiconDB(fresh_db) as db:
        # Diamond: root1 → mid; root2 → mid. mid is reachable from both.
        forms = _seed_descent_chain(
            db,
            ("root1", "mid", "inheritance"),
            ("root2", "mid", "inheritance"),
        )
        cluster_cognates(db, apply=True)
        mid_synset = db.conn.execute(
            "SELECT cognate_id FROM etymon WHERE canonical_form = 'mid'"
        ).fetchone()["cognate_id"]

    # root1 was inserted first (lower id), so it wins.
    assert mid_synset == forms["root1"]


def test_cluster_cognates_idempotent_on_rerun(fresh_db: Path) -> None:
    """A second --apply run with no schema changes is a content no-op:
    the same rows get the same assignments. 'candidates' count stays
    constant because the implementation doesn't track 'already-assigned'
    vs 'newly-assigned' — it overwrites with the same values."""
    from wyrd.generators.kenning.lexicon import cluster_cognates

    with LexiconDB(fresh_db) as db:
        _seed_descent_chain(
            db,
            ("a", "b", "inheritance"),
            ("b", "c", "inheritance"),
        )
        first = cluster_cognates(db, apply=True)
        first_ids = {
            row["canonical_form"]: row["cognate_id"]
            for row in db.conn.execute("SELECT canonical_form, cognate_id FROM etymon")
        }
        second = cluster_cognates(db, apply=True)
        second_ids = {
            row["canonical_form"]: row["cognate_id"]
            for row in db.conn.execute("SELECT canonical_form, cognate_id FROM etymon")
        }

    assert first["candidates"] == second["candidates"]
    assert first_ids == second_ids


def test_cluster_cognates_skips_etymons_without_descent_edges(fresh_db: Path) -> None:
    """Singleton etymons (rando-port seed entries with no Wiktionary
    chain) get NULL cognate_id. They're not in any cognate set."""
    from wyrd.generators.kenning.lexicon import cluster_cognates

    with LexiconDB(fresh_db) as db:
        # One chain, plus a singleton that has no descent edges.
        _seed_descent_chain(
            db,
            ("a", "b", "inheritance"),
        )
        singleton_id = db.upsert_etymon("singleton", "old-english")
        db.commit()

        cluster_cognates(db, apply=True)

        singleton_synset = db.conn.execute(
            "SELECT cognate_id FROM etymon WHERE id = ?",
            (singleton_id,),
        ).fetchone()["cognate_id"]

    assert singleton_synset is None


def test_clear_enrichment_cognates_stage_resets_cognate_assignments(
    fresh_db: Path,
) -> None:
    """clear-enrichment --stage=cognates undoes a cluster-cognates run:
    cognate_id and cognate_method go back to NULL on every row. Mining
    evidence (etymon_descent rows themselves) is untouched."""
    from wyrd.generators.kenning.lexicon import cluster_cognates

    with LexiconDB(fresh_db) as db:
        _seed_descent_chain(
            db,
            ("a", "b", "inheritance"),
        )
        cluster_cognates(db, apply=True)
        # Pre-clear: assignments exist.
        before = db.conn.execute(
            "SELECT COUNT(*) AS n FROM etymon WHERE cognate_id IS NOT NULL"
        ).fetchone()["n"]
        assert before == 2

        result = clear_enrichment(db, stage="cognates", apply=True)
        after = db.conn.execute(
            "SELECT COUNT(*) AS n FROM etymon WHERE cognate_id IS NOT NULL"
        ).fetchone()["n"]
        # etymon_descent rows must survive — they're mining evidence (D21).
        descent_after = db.conn.execute("SELECT COUNT(*) AS n FROM etymon_descent").fetchone()["n"]

    assert result["cognate_assignments_to_clear"] == 2
    assert after == 0
    assert descent_after == 1


def test_clear_enrichment_all_derived_includes_cognates(fresh_db: Path) -> None:
    """all-derived sweeps the cognate stage too — verifies the new stage
    was wired into the all-derived alias. Without this, callers using
    the --stage=all-derived shortcut would silently leave synset
    assignments behind."""
    from wyrd.generators.kenning.lexicon import cluster_cognates

    with LexiconDB(fresh_db) as db:
        _seed_descent_chain(
            db,
            ("a", "b", "inheritance"),
        )
        cluster_cognates(db, apply=True)
        result = clear_enrichment(db, stage="all-derived", apply=True)
        synset_count = db.conn.execute(
            "SELECT COUNT(*) AS n FROM etymon WHERE cognate_id IS NOT NULL"
        ).fetchone()["n"]

    assert result["cognate_assignments_to_clear"] == 2
    assert synset_count == 0


def test_lexicon_path_cli_prints_resolved_db_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`wyrd kenning lexicon path` prints whatever default_lexicon_path
    resolves to. Pin via env override (the simplest way to control it
    from a test). Confirms the new wyrd-366 CLI surface is wired."""
    target = tmp_path / "scratch" / "lex.db"
    monkeypatch.setenv("WYRD_LEXICON_DB", str(target))
    runner = CliRunner()
    result = runner.invoke(kenning_cli, ["lexicon", "path"])
    assert result.exit_code == 0, result.output
    assert str(target) in result.output


def test_lexicon_era_cell_cli_resolves_known_pair() -> None:
    """`wyrd kenning lexicon era-cell <language> <year>` prints the
    family/label pair for a valid (language, year) input. Confirms
    the wyrd-38d CLI surface is wired through the era module."""
    runner = CliRunner()
    result = runner.invoke(kenning_cli, ["lexicon", "era-cell", "old-english", "950"])
    assert result.exit_code == 0, result.output
    assert "english/oe-late" in result.output


def test_lexicon_era_cell_cli_exits_nonzero_for_proto_language() -> None:
    """Proto-Germanic doesn't have an era family. The CLI surfaces
    that as a non-zero exit so a script piping to xargs etc. fails
    loudly rather than silently producing no output."""
    runner = CliRunner()
    result = runner.invoke(kenning_cli, ["lexicon", "era-cell", "proto-germanic", "500"])
    assert result.exit_code == 1
    assert "no era family" in result.output


def test_lexicon_era_cell_cli_exits_nonzero_for_year_out_of_range() -> None:
    """Latin's renaissance ends at 1800; year 2025 has no defined
    cell. CLI exits non-zero with a descriptive stderr message."""
    runner = CliRunner()
    result = runner.invoke(kenning_cli, ["lexicon", "era-cell", "latin", "2025"])
    assert result.exit_code == 1
    assert "outside the defined cells" in result.output


def test_lexicon_help_does_not_trigger_default_path_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Critical regression guard: `wyrd kenning lexicon migrate --help`
    must NOT call default_lexicon_path() (and therefore must NOT
    trigger the legacy-DB migration as a side effect of viewing help).
    Click invokes a `default=` callable when `show_default=True` in
    order to display the resolved value; this PR uses the static
    LEXICON_DB_DEFAULT_DISPLAY string instead so the callable stays
    quiet on --help.

    A future regression that flips one option back to
    `show_default=True` would re-introduce the surprise side-effect.
    """
    from wyrd.generators.kenning import paths

    call_count = [0]

    def counting_resolver() -> Path:
        call_count[0] += 1
        return Path("/sentinel/should-never-be-called")

    monkeypatch.setattr(paths, "default_lexicon_path", counting_resolver)
    # Also patch the alias re-bound at cli import time.
    from wyrd.generators.kenning import cli as cli_mod

    monkeypatch.setattr(cli_mod, "_DEFAULT_LEXICON_PATH", counting_resolver)

    runner = CliRunner()
    # Hit --help on a command whose --db option uses the default
    # callable; if any of them silently resolve, the count jumps.
    for cmd_path in (
        ["lexicon", "migrate", "--help"],
        ["lexicon", "cluster-cognates", "--help"],
        ["lexicon", "lookup-attested-years", "--help"],
        ["lexicon", "clear-enrichment", "--help"],
    ):
        result = runner.invoke(kenning_cli, cmd_path)
        assert result.exit_code == 0, f"{cmd_path}: {result.output}"
    assert call_count[0] == 0, (
        f"--help should not invoke default_lexicon_path "
        f"(got {call_count[0]} calls). Some --db option likely uses "
        f"show_default=True instead of LEXICON_DB_DEFAULT_DISPLAY."
    )


def test_cluster_cognates_cli_dry_run_reports_without_writing(
    fresh_db: Path,
) -> None:
    """End-to-end CLI smoke: dry-run reports root + candidate counts
    and leaves the DB untouched."""
    with LexiconDB(fresh_db) as db:
        _seed_descent_chain(
            db,
            ("a", "b", "inheritance"),
            ("b", "c", "inheritance"),
        )

    runner = CliRunner()
    result = runner.invoke(
        kenning_cli,
        ["lexicon", "cluster-cognates", "--db", str(fresh_db)],
    )

    assert result.exit_code == 0, result.output
    assert "1 root(s) walked" in result.output
    assert "would assign cognate_id on 3 etymon(s)" in result.output
    assert "dry-run" in result.output

    with LexiconDB(fresh_db) as db:
        cognate_ids = [
            row["cognate_id"] for row in db.conn.execute("SELECT cognate_id FROM etymon")
        ]
    assert all(sid is None for sid in cognate_ids)


def test_cluster_cognates_cli_apply_writes_assignments(fresh_db: Path) -> None:
    """--apply persists the cluster pass; subsequent SELECT shows the
    canonical cognate_id pointer."""
    with LexiconDB(fresh_db) as db:
        forms = _seed_descent_chain(
            db,
            ("a", "b", "inheritance"),
        )

    runner = CliRunner()
    result = runner.invoke(
        kenning_cli,
        ["lexicon", "cluster-cognates", "--db", str(fresh_db), "--apply"],
    )

    assert result.exit_code == 0, result.output
    assert "assigned cognate_id on 2 etymon(s)" in result.output
    # Pin the rows_written echo branch (added round-2 fix) — without
    # this assertion a typo in the format string would slip through
    # despite the lib-level dict field having coverage.
    assert "rows_written = 2" in result.output
    assert "dry-run" not in result.output

    with LexiconDB(fresh_db) as db:
        cognate_ids = {
            row["canonical_form"]: row["cognate_id"]
            for row in db.conn.execute("SELECT canonical_form, cognate_id FROM etymon")
        }
    assert cognate_ids["a"] == forms["a"]
    assert cognate_ids["b"] == forms["a"]


# --- wyrd-81n round-1 review follow-ups -------------------------------


@pytest.mark.parametrize("non_bridging_edge", ["derivation", "calque", "compound", "unknown"])
def test_cluster_cognates_non_bridging_edge_types_do_not_cluster(
    fresh_db: Path, non_bridging_edge: str
) -> None:
    """D27 lists derivation, calque, compound, unknown as non-bridging
    (alongside cognate). Pin each so a refactor that widens
    _COGNATE_BRIDGING_EDGES to include any of them surfaces here.
    Without these tests, only 'cognate' was pinned and the others
    could silently start bridging."""
    from wyrd.generators.kenning.lexicon import cluster_cognates

    with LexiconDB(fresh_db) as db:
        _seed_descent_chain(db, ("a", "b", non_bridging_edge))
        cluster_cognates(db, apply=True)
        rows = {
            row["canonical_form"]: row["cognate_id"]
            for row in db.conn.execute("SELECT canonical_form, cognate_id FROM etymon")
        }
    assert rows["a"] is None, f"{non_bridging_edge} edge unexpectedly bridged"
    assert rows["b"] is None, f"{non_bridging_edge} edge unexpectedly bridged"


def test_cluster_cognates_self_loop_silently_filtered(fresh_db: Path) -> None:
    """Self-loops (X→X) are meaningless data — they assert an etymon is
    its own ancestor. Per wyrd-223, the canonical_edges builder filters
    self-loops upstream of the BFS, so they don't generate cycle_orphans
    or otherwise pollute the cluster pass. BFS must terminate (no
    infinite loop) and the etymon stays NULL — silently."""
    from wyrd.generators.kenning.lexicon import cluster_cognates

    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="wiktionary", title="Wiktionary")
        x_id = db.upsert_etymon("x", "proto-germanic")
        db.conn.execute(
            "INSERT INTO etymon_descent "
            "(parent_id, child_id, edge_type, source_id) VALUES (?, ?, ?, ?)",
            (x_id, x_id, "inheritance", "wiktionary"),
        )
        db.commit()

        result = cluster_cognates(db, apply=True)

        cognate = db.conn.execute("SELECT cognate_id FROM etymon WHERE id = ?", (x_id,)).fetchone()[
            "cognate_id"
        ]

    assert result["roots"] == 0
    # cycle_orphans=0 because the self-loop edge was filtered upstream;
    # the etymon doesn't participate in any canonical edge.
    assert result["cycle_orphans"] == 0
    assert cognate is None


def test_cluster_cognates_mutual_cycle_terminates_and_orphans(fresh_db: Path) -> None:
    """A two-node cycle (A→B, B→A) with no external root: BFS must
    terminate and both etymons flagged as cycle orphans."""
    from wyrd.generators.kenning.lexicon import cluster_cognates

    with LexiconDB(fresh_db) as db:
        forms = _seed_descent_chain(
            db,
            ("a", "b", "inheritance"),
            ("b", "a", "inheritance"),
        )
        result = cluster_cognates(db, apply=True)
        rows = {
            row["canonical_form"]: row["cognate_id"]
            for row in db.conn.execute("SELECT canonical_form, cognate_id FROM etymon")
        }

    assert result["roots"] == 0
    assert result["cycle_orphans"] == 2
    assert rows["a"] is None
    assert rows["b"] is None
    assert "a" in forms and "b" in forms


def test_cluster_cognates_anchored_cycle_assigns_via_root(fresh_db: Path) -> None:
    """root → a → b → a (cycle reachable from a real root): the cycle
    members get root's cognate_id, and BFS terminates because the
    not-in-assignments check doubles as a visited-set."""
    from wyrd.generators.kenning.lexicon import cluster_cognates

    with LexiconDB(fresh_db) as db:
        forms = _seed_descent_chain(
            db,
            ("root", "a", "inheritance"),
            ("a", "b", "inheritance"),
            ("b", "a", "inheritance"),
        )
        result = cluster_cognates(db, apply=True)
        rows = {
            row["canonical_form"]: row["cognate_id"]
            for row in db.conn.execute("SELECT canonical_form, cognate_id FROM etymon")
        }

    assert result["cycle_orphans"] == 0
    assert rows["root"] == forms["root"]
    assert rows["a"] == forms["root"]
    assert rows["b"] == forms["root"]


def test_cluster_cognates_idempotent_apply_skips_unchanged_rows(fresh_db: Path) -> None:
    """The second apply against unchanged data must produce
    rows_written = 0 — the WHERE-clause guard short-circuits no-op
    UPDATEs. Without the guard, every re-run would touch every row
    and inflate write counters."""
    from wyrd.generators.kenning.lexicon import cluster_cognates

    with LexiconDB(fresh_db) as db:
        _seed_descent_chain(
            db,
            ("a", "b", "inheritance"),
            ("b", "c", "inheritance"),
        )
        first = cluster_cognates(db, apply=True)
        second = cluster_cognates(db, apply=True)

    assert first["rows_written"] == 3
    assert second["rows_written"] == 0
    assert first["candidates"] == second["candidates"]


def test_clear_enrichment_cli_cognates_stage_round_trips(fresh_db: Path) -> None:
    """End-to-end CLI smoke for clear-enrichment --stage=cognates: the
    new Click choice + new echo branch had no test (only the lib
    function did). Pin via CliRunner so a typo in the choice list
    surfaces immediately."""
    from wyrd.generators.kenning.lexicon import cluster_cognates

    with LexiconDB(fresh_db) as db:
        _seed_descent_chain(db, ("a", "b", "inheritance"))
        cluster_cognates(db, apply=True)

    runner = CliRunner()
    dry = runner.invoke(
        kenning_cli,
        ["lexicon", "clear-enrichment", "--db", str(fresh_db), "--stage", "cognates"],
    )
    assert dry.exit_code == 0, dry.output
    assert "Stage: cognates" in dry.output
    assert "would clear 2 cognate_id assignments" in dry.output
    assert "dry-run" in dry.output

    apply = runner.invoke(
        kenning_cli,
        [
            "lexicon",
            "clear-enrichment",
            "--db",
            str(fresh_db),
            "--stage",
            "cognates",
            "--apply",
        ],
    )
    assert apply.exit_code == 0, apply.output
    assert "cleared 2 cognate_id assignments" in apply.output

    with LexiconDB(fresh_db) as db:
        remaining = db.conn.execute(
            "SELECT COUNT(*) AS n FROM etymon WHERE cognate_id IS NOT NULL"
        ).fetchone()["n"]
    assert remaining == 0


def test_clear_enrichment_cli_attested_years_stage_round_trips(
    fresh_db: Path, tmp_path: Path
) -> None:
    """End-to-end CLI smoke for clear-enrichment --stage=attested-years
    (D5-1 / wyrd-3ux). Pin via CliRunner so a typo in the new Click
    Choice or the new echo branch surfaces immediately — the lib-level
    test (test_clear_enrichment_attested_years_resets_only_year_column)
    can't catch CLI wiring bugs."""
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "src.txt").write_text("Tune, 1086 (DB).")
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        rid = _seed_text_match(db, source_id="src", canonical_form="tune")
        db.conn.execute(
            "UPDATE etymon_text_match SET attested_year = ? WHERE id = ?",
            (1086, rid),
        )
        db.commit()

    runner = CliRunner()
    dry = runner.invoke(
        kenning_cli,
        ["lexicon", "clear-enrichment", "--db", str(fresh_db), "--stage", "attested-years"],
    )
    assert dry.exit_code == 0, dry.output
    assert "Stage: attested-years" in dry.output
    assert "would clear 1 attested_year values" in dry.output
    assert "dry-run" in dry.output

    apply = runner.invoke(
        kenning_cli,
        [
            "lexicon",
            "clear-enrichment",
            "--db",
            str(fresh_db),
            "--stage",
            "attested-years",
            "--apply",
        ],
    )
    assert apply.exit_code == 0, apply.output
    assert "cleared 1 attested_year values" in apply.output

    with LexiconDB(fresh_db) as db:
        remaining = db.conn.execute(
            "SELECT COUNT(*) AS n FROM etymon_text_match WHERE attested_year IS NOT NULL"
        ).fetchone()["n"]
    assert remaining == 0


def test_lookup_attested_years_cli_dry_run_reports_without_writing(
    fresh_db: Path, tmp_path: Path
) -> None:
    """End-to-end CLI smoke: dry-run reports row + candidate counts and
    leaves attested_year NULL. Pins the Click wiring (--db, --apply
    flag, sources_dir argument) so a typo surfaces immediately."""
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "src.txt").write_text("Tune, 1086 (DB).")
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        _seed_text_match(db, source_id="src", canonical_form="tune")

    runner = CliRunner()
    result = runner.invoke(
        kenning_cli,
        ["lexicon", "lookup-attested-years", str(sources), "--db", str(fresh_db)],
    )
    assert result.exit_code == 0, result.output
    assert "etymon_text_match" in result.output
    assert "scanned=    1" in result.output
    assert "candidates=    1" in result.output
    assert "rows_written=    0" in result.output
    assert "dry-run" in result.output

    with LexiconDB(fresh_db) as db:
        years = [
            row["attested_year"]
            for row in db.conn.execute("SELECT attested_year FROM etymon_text_match")
        ]
    assert all(y is None for y in years)


def test_lookup_attested_years_cli_apply_writes_years(fresh_db: Path, tmp_path: Path) -> None:
    """--apply persists attested_year on each qualifying row. Pins the
    rows_written echo branch — without this assertion a typo in the
    format string would slip through despite the lib-level dict field
    having coverage."""
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "src.txt").write_text("Tune, 1086 (DB).")
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="src", title="S")
        rid = _seed_text_match(db, source_id="src", canonical_form="tune")

    runner = CliRunner()
    result = runner.invoke(
        kenning_cli,
        [
            "lexicon",
            "lookup-attested-years",
            str(sources),
            "--db",
            str(fresh_db),
            "--apply",
        ],
    )
    assert result.exit_code == 0, result.output
    # etymon_text_match line shows the actual write; toponym_etymology
    # line is zero (no rows seeded for that table).
    assert "etymon_text_match" in result.output
    assert "rows_written=    1" in result.output
    assert "dry-run" not in result.output

    with LexiconDB(fresh_db) as db:
        year = db.conn.execute(
            "SELECT attested_year FROM etymon_text_match WHERE id = ?", (rid,)
        ).fetchone()["attested_year"]
    assert year == 1086


def test_cluster_cognates_cli_warns_on_cycle_orphans(fresh_db: Path) -> None:
    """A descent graph with a closed cycle and no external root produces
    a warn line on stderr so operators notice the missed cluster."""
    with LexiconDB(fresh_db) as db:
        _seed_descent_chain(
            db,
            ("a", "b", "inheritance"),
            ("b", "a", "inheritance"),
        )

    runner = CliRunner()
    result = runner.invoke(
        kenning_cli,
        ["lexicon", "cluster-cognates", "--db", str(fresh_db), "--apply"],
    )
    assert result.exit_code == 0, result.output
    assert "warn:" in result.output
    assert "2 etymon(s) sit in a bridging-edge cycle" in result.output
    # rows_written = 0 here: there are 0 candidates because no roots,
    # so the apply branch echoes 'rows_written = 0'.
    assert "rows_written = 0" in result.output


# --- wyrd-223: cluster-cognates resolves through merged_into_id -------


def test_cluster_cognates_resolves_descent_edges_through_merged_into_id(
    fresh_db: Path,
) -> None:
    """Bug discovered during PR #41 live ingest 2026-05-03: a descent
    edge pointing at an OCR-merge tombstone was being walked literally,
    so cognate_id landed on the tombstone (loser) rather than the
    canonical winner. Cluster-cognates now resolves both endpoints
    through merged_into_id before BFS walking.

    Setup mirrors the real wiktextract+place-name case:
      - canonical 'tūn' (with macron) — the winner from a prior OCR merge
      - tombstone 'tun' (no macron) — merged_into_id → tūn
      - descent edge: gmw-pro:*tūn → ang:tun (points at TOMBSTONE)
    Expected: cognate_id lands on the canonical 'tūn', not on 'tun'.
    """
    from wyrd.generators.kenning.lexicon import cluster_cognates

    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="wiktionary", title="Wiktionary")
        proto_id = db.upsert_etymon("*tūn", "gmw-pro")
        canonical_id = db.upsert_etymon("tūn", "old-english")
        tombstone_id = db.upsert_etymon("tun", "old-english")
        # Tombstone the ASCII variant.
        db.conn.execute(
            "UPDATE etymon SET merged_into_id = ? WHERE id = ?",
            (canonical_id, tombstone_id),
        )
        # Edge points at the TOMBSTONE id (the wiktextract upsert path).
        db.conn.execute(
            "INSERT INTO etymon_descent (parent_id, child_id, edge_type, source_id) "
            "VALUES (?, ?, 'inheritance', 'wiktionary')",
            (proto_id, tombstone_id),
        )
        db.commit()

        result = cluster_cognates(db, apply=True)
        canonical_synset = db.conn.execute(
            "SELECT cognate_id FROM etymon WHERE id = ?", (canonical_id,)
        ).fetchone()["cognate_id"]
        tombstone_synset = db.conn.execute(
            "SELECT cognate_id FROM etymon WHERE id = ?", (tombstone_id,)
        ).fetchone()["cognate_id"]

    # Critical: the canonical 'tūn' got the synset assignment, NOT the tombstone.
    assert canonical_synset == proto_id
    assert tombstone_synset is None
    # Sanity: the cluster has 2 canonical members (proto root + canonical tūn).
    assert result["candidates"] == 2


def test_cluster_cognates_canonical_root_wins_when_root_was_merged(
    fresh_db: Path,
) -> None:
    """If a tombstone is itself a parent in some descent edge, the
    canonical winner becomes the parent in the canonical edge set.
    Pin so a refactor that drops the parent-side resolution doesn't
    silently regress."""
    from wyrd.generators.kenning.lexicon import cluster_cognates

    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="wiktionary", title="Wiktionary")
        canonical_root_id = db.upsert_etymon("*canonical_root", "gmw-pro")
        tombstone_root_id = db.upsert_etymon("*tombstone_root", "gmw-pro")
        child_id = db.upsert_etymon("child", "old-english")
        # Tombstone one root variant.
        db.conn.execute(
            "UPDATE etymon SET merged_into_id = ? WHERE id = ?",
            (canonical_root_id, tombstone_root_id),
        )
        # Edge points FROM the tombstone root.
        db.conn.execute(
            "INSERT INTO etymon_descent (parent_id, child_id, edge_type, source_id) "
            "VALUES (?, ?, 'inheritance', 'wiktionary')",
            (tombstone_root_id, child_id),
        )
        db.commit()

        cluster_cognates(db, apply=True)
        rows = {
            row["canonical_form"]: row["cognate_id"]
            for row in db.conn.execute("SELECT canonical_form, cognate_id FROM etymon")
        }

    assert rows["*canonical_root"] == canonical_root_id
    assert rows["child"] == canonical_root_id
    # Tombstone stays NULL.
    assert rows["*tombstone_root"] is None


def test_cluster_cognates_self_loop_via_merge_does_not_crash(fresh_db: Path) -> None:
    """If both endpoints of an edge merge into the same canonical (the
    merge itself created a self-loop in canonical space), the edge is
    filtered out — no infinite loop, no spurious cluster of a single
    etymon under itself."""
    from wyrd.generators.kenning.lexicon import cluster_cognates

    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="wiktionary", title="Wiktionary")
        canonical_id = db.upsert_etymon("canonical", "old-english")
        tomb_a_id = db.upsert_etymon("tomb_a", "old-english")
        tomb_b_id = db.upsert_etymon("tomb_b", "old-english")
        # Both tombstones merge into the same canonical.
        for tomb in (tomb_a_id, tomb_b_id):
            db.conn.execute(
                "UPDATE etymon SET merged_into_id = ? WHERE id = ?",
                (canonical_id, tomb),
            )
        # Edge between two tombstones that resolve to the same canonical.
        db.conn.execute(
            "INSERT INTO etymon_descent (parent_id, child_id, edge_type, source_id) "
            "VALUES (?, ?, 'inheritance', 'wiktionary')",
            (tomb_a_id, tomb_b_id),
        )
        db.commit()

        result = cluster_cognates(db, apply=True)
    # No roots (no canonical edges left after self-loop filter);
    # no cluster either. Just a clean no-op, no infinite walk.
    assert result["roots"] == 0
    assert result["candidates"] == 0
    assert result["cycle_orphans"] == 0


# --- wyrd-083: bridge_generic_language ---------------------------------


def test_bridge_generic_language_picks_priority_match(fresh_db: Path) -> None:
    """A generic 'celtic' etymon with a matching canonical_form in
    'irish' AND 'old-irish' bridges to old-irish (higher priority per
    the candidate order). Pin so a refactor of the priority semantics
    surfaces here."""
    from wyrd.generators.kenning.lexicon import bridge_generic_language

    with LexiconDB(fresh_db) as db:
        # Older entries get smaller ids; pre-create the generic FIRST
        # to verify priority order — not insertion order — wins.
        celtic_id = db.upsert_etymon("bun", "celtic")
        irish_id = db.upsert_etymon("bun", "irish")
        oi_id = db.upsert_etymon("bun", "old-irish")
        db.commit()
        result = bridge_generic_language(
            db,
            generic_lang="celtic",
            candidate_langs=("old-irish", "irish"),
            apply=True,
        )
        merged_target = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (celtic_id,)
        ).fetchone()["merged_into_id"]

    assert merged_target == oi_id  # old-irish is higher priority than irish
    assert merged_target != irish_id
    assert result["bridged"] == 1
    assert result["unmatched"] == 0


def test_bridge_generic_language_no_match_leaves_unmerged(fresh_db: Path) -> None:
    """A generic etymon with no specific-language match keeps
    merged_into_id NULL; counter increments unmatched."""
    from wyrd.generators.kenning.lexicon import bridge_generic_language

    with LexiconDB(fresh_db) as db:
        no_match_id = db.upsert_etymon("rma", "celtic")  # not in any specific-celtic
        db.upsert_etymon("bun", "irish")  # exists but different form
        db.commit()
        result = bridge_generic_language(
            db,
            generic_lang="celtic",
            candidate_langs=("irish", "welsh"),
            apply=True,
        )
        merged_target = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (no_match_id,)
        ).fetchone()["merged_into_id"]

    assert merged_target is None
    assert result["bridged"] == 0
    assert result["unmatched"] == 1


def test_bridge_generic_language_case_insensitive_match(fresh_db: Path) -> None:
    """Canonical form match is case-insensitive — Wiktextract sometimes
    capitalizes proper nouns (`Nás`) where place-name dicts lowercase
    them (`nás`). Pin so a refactor doesn't drop the case-fold."""
    from wyrd.generators.kenning.lexicon import bridge_generic_language

    with LexiconDB(fresh_db) as db:
        celtic_id = db.upsert_etymon("nás", "celtic")
        ir_id = db.upsert_etymon("Nás", "irish")
        db.commit()
        bridge_generic_language(
            db,
            generic_lang="celtic",
            candidate_langs=("irish",),
            apply=True,
        )
        merged_target = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (celtic_id,)
        ).fetchone()["merged_into_id"]

    assert merged_target == ir_id


def test_bridge_generic_language_dry_run_does_not_write(fresh_db: Path) -> None:
    """apply=False reports counts but leaves merged_into_id NULL."""
    from wyrd.generators.kenning.lexicon import bridge_generic_language

    with LexiconDB(fresh_db) as db:
        celtic_id = db.upsert_etymon("bun", "celtic")
        db.upsert_etymon("bun", "irish")
        db.commit()
        result = bridge_generic_language(
            db,
            generic_lang="celtic",
            candidate_langs=("irish",),
            apply=False,
        )
        merged_target = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (celtic_id,)
        ).fetchone()["merged_into_id"]

    assert merged_target is None
    assert result["bridged"] == 1
    assert result["rows_written"] == 0


def test_bridge_generic_language_idempotent_apply_skips_unchanged(
    fresh_db: Path,
) -> None:
    """A second apply against unchanged data writes zero rows — the
    UPDATE has a WHERE-clause guard."""
    from wyrd.generators.kenning.lexicon import bridge_generic_language

    with LexiconDB(fresh_db) as db:
        db.upsert_etymon("bun", "celtic")
        db.upsert_etymon("bun", "irish")
        db.commit()
        first = bridge_generic_language(
            db, generic_lang="celtic", candidate_langs=("irish",), apply=True
        )
        second = bridge_generic_language(
            db, generic_lang="celtic", candidate_langs=("irish",), apply=True
        )

    assert first["rows_written"] == 1
    assert second["rows_written"] == 0


def test_bridge_generic_language_already_merged_tombstones_get_chain_flattened(
    fresh_db: Path,
) -> None:
    """Generic etymons that are already OCR-merged (merged_into_id IS
    NOT NULL) aren't directly examined by the bridge candidate scan
    (the SELECT filters on merged_into_id IS NULL). BUT when the
    canonical they point at gets bridged, the chain-flatten clause
    re-routes them onto the new target so the consensus rollup chain
    stays at depth 1 (round-1 fix for wyrd-083 / wyrd-ft3).

    Pre: tomb_celtic → target_celtic
    Bridge: target_celtic → irish
    Post: BOTH tomb_celtic AND target_celtic point at irish
    """
    from wyrd.generators.kenning.lexicon import bridge_generic_language

    with LexiconDB(fresh_db) as db:
        target_celtic_id = db.upsert_etymon("bun", "celtic")
        tomb_celtic_id = db.upsert_etymon("BUN", "celtic")
        db.conn.execute(
            "UPDATE etymon SET merged_into_id = ? WHERE id = ?",
            (target_celtic_id, tomb_celtic_id),
        )
        irish_id = db.upsert_etymon("bun", "irish")
        db.commit()
        result = bridge_generic_language(
            db, generic_lang="celtic", candidate_langs=("irish",), apply=True
        )
        rows = {
            r["id"]: r["merged_into_id"]
            for r in db.conn.execute("SELECT id, merged_into_id FROM etymon")
        }

    # Both rows now point at irish — chain stays at depth 1.
    assert rows[target_celtic_id] == irish_id
    assert rows[tomb_celtic_id] == irish_id
    # Only the canonical celtic etymon was examined as a bridge source.
    assert result["generic_etymons"] == 1


def test_bridge_generic_language_empty_candidates_raises(fresh_db: Path) -> None:
    """An empty candidate_langs is a programmer error; raise rather
    than silently process zero candidates."""
    from wyrd.generators.kenning.lexicon import bridge_generic_language

    with LexiconDB(fresh_db) as db, pytest.raises(ValueError, match="candidate_langs"):
        bridge_generic_language(db, generic_lang="celtic", candidate_langs=(), apply=False)


def test_bridge_celtic_then_cluster_cognates_includes_generic_in_cluster(
    fresh_db: Path,
) -> None:
    """Load-bearing integration test: bridge celtic → irish, then run
    cluster-cognates (which is redirect-aware per wyrd-223), and
    confirm the generic celtic etymon is rolled up via the merged_into_id
    chain into the irish entry's cluster.

    Tree: proto-celtic *bunV → irish bun → (celtic bun is now a
    tombstone pointing at irish bun)
    """
    from wyrd.generators.kenning.lexicon import (
        bridge_generic_language,
        cluster_cognates,
    )

    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="wiktionary", title="Wiktionary")
        proto_id = db.upsert_etymon("*bunV", "proto-celtic")
        irish_id = db.upsert_etymon("bun", "irish")
        celtic_id = db.upsert_etymon("bun", "celtic")
        db.conn.execute(
            "INSERT INTO etymon_descent (parent_id, child_id, edge_type, source_id) "
            "VALUES (?, ?, 'inheritance', 'wiktionary')",
            (proto_id, irish_id),
        )
        db.commit()

        # Pre-bridge: celtic bun is its own canonical, no synset link.
        cluster_cognates(db, apply=True)
        celtic_synset = db.conn.execute(
            "SELECT cognate_id FROM etymon WHERE id = ?", (celtic_id,)
        ).fetchone()["cognate_id"]
        assert celtic_synset is None  # no edges anchor it

        # Bridge celtic → irish.
        bridge_generic_language(db, generic_lang="celtic", candidate_langs=("irish",), apply=True)
        # Re-cluster after the bridge (redirect-aware so irish's synset
        # rollup applies to celtic via the merged_into_id chain).
        from wyrd.generators.kenning.lexicon import clear_enrichment

        clear_enrichment(db, stage="cognates", apply=True)
        cluster_cognates(db, apply=True)

        irish_synset = db.conn.execute(
            "SELECT cognate_id FROM etymon WHERE id = ?", (irish_id,)
        ).fetchone()["cognate_id"]
        celtic_target = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (celtic_id,)
        ).fetchone()["merged_into_id"]

    # Critical: the celtic etymon now points at irish via merged_into_id;
    # irish has the synset assignment from the proto-celtic root.
    assert celtic_target == irish_id
    assert irish_synset == proto_id


# --- wyrd-ft3: bridge_phonological_oe ----------------------------------


def test_bridge_phonological_oe_merges_known_pair(fresh_db: Path) -> None:
    """A known OE place-name form (`ton`) with a canonical Wiktionary
    target (`tūn`) gets merged_into_id set to the canonical's id.
    Pin so a refactor of _OE_PHONOLOGICAL_BRIDGES surfaces here."""
    from wyrd.generators.kenning.lexicon import bridge_phonological_oe

    with LexiconDB(fresh_db) as db:
        place_id = db.upsert_etymon("ton", "old-english")
        wiktionary_id = db.upsert_etymon("tūn", "old-english")
        db.commit()
        result = bridge_phonological_oe(db, apply=True)
        merged_target = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (place_id,)
        ).fetchone()["merged_into_id"]

    assert merged_target == wiktionary_id
    assert result["bridged"] == 1
    assert result["unmatched"] == 1  # the wiktionary 'tūn' itself isn't a place-name form


@pytest.mark.parametrize(
    "place_form,wiktionary_form",
    [
        ("ton", "tūn"),
        ("lea", "lēah"),
        ("cote", "cot"),
        ("burgh", "burh"),
        ("dale", "dæl"),
        ("hall", "heall"),
        ("ey", "ieg"),
        ("burn", "burna"),
        ("hale", "healh"),
        ("new", "nīwe"),
        ("wood", "wudu"),
        ("bridge", "brycg"),
        ("stone", "stān"),
        ("hill", "hyll"),
        # wyrd-9ig additions:
        ("wella", "welle"),
        ("inga", "ing"),
        ("cippa", "cipp"),
        ("hāran", "hār"),
        ("cloppa", "clopp"),
        ("ticce", "ticcen"),
        ("kirk", "cirice"),
        ("hala", "healh"),
        ("pool", "pol"),
        ("scelf", "scylfe"),
        ("ēg", "ieg"),
        ("hare", "hara"),
        ("hlyp", "hlīep"),
        ("lech", "lacu"),
    ],
)
def test_bridge_phonological_oe_table_entries_resolve(
    fresh_db: Path, place_form: str, wiktionary_form: str
) -> None:
    """Each high-witness entry in the hand-curated bridge table is
    pinned individually. A typo or accidental drop in
    _OE_PHONOLOGICAL_BRIDGES would surface against the specific
    place-name → wiktionary pair it broke."""
    from wyrd.generators.kenning.lexicon import bridge_phonological_oe

    with LexiconDB(fresh_db) as db:
        place_id = db.upsert_etymon(place_form, "old-english")
        target_id = db.upsert_etymon(wiktionary_form, "old-english")
        db.commit()
        bridge_phonological_oe(db, apply=True)
        merged_target = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (place_id,)
        ).fetchone()["merged_into_id"]
    assert merged_target == target_id, f"{place_form!r} did not bridge to {wiktionary_form!r}"


def test_bridge_phonological_oe_no_target_increments_missing_target(
    fresh_db: Path,
) -> None:
    """If the table names a target form that doesn't exist as a
    canonical OE etymon (because it hasn't been ingested), the
    missing_target counter increments and the place-name etymon
    stays unmerged. Pin so operators see the signal in CLI output."""
    from wyrd.generators.kenning.lexicon import bridge_phonological_oe

    with LexiconDB(fresh_db) as db:
        # 'ton' is in the table → 'tūn', but no 'tūn' etymon exists.
        place_id = db.upsert_etymon("ton", "old-english")
        db.commit()
        result = bridge_phonological_oe(db, apply=True)
        merged_target = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (place_id,)
        ).fetchone()["merged_into_id"]

    assert merged_target is None
    assert result["bridged"] == 0
    assert result["missing_target"] == 1


def test_bridge_phonological_oe_unmatched_form_left_alone(fresh_db: Path) -> None:
    """An OE etymon whose canonical_form isn't in the bridge table is
    counted as unmatched (silent — no missing_target signal because
    the table doesn't claim it should bridge)."""
    from wyrd.generators.kenning.lexicon import bridge_phonological_oe

    with LexiconDB(fresh_db) as db:
        unknown_id = db.upsert_etymon("xyzzy_unmapped", "old-english")
        db.commit()
        result = bridge_phonological_oe(db, apply=True)
        merged_target = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (unknown_id,)
        ).fetchone()["merged_into_id"]

    assert merged_target is None
    assert result["bridged"] == 0
    assert result["unmatched"] == 1
    assert result["missing_target"] == 0


def test_bridge_phonological_oe_ignores_other_languages(fresh_db: Path) -> None:
    """The pass only walks language='old-english' rows. A 'modern-english'
    or 'celtic' etymon with the same form (`ton`) is not touched."""
    from wyrd.generators.kenning.lexicon import bridge_phonological_oe

    with LexiconDB(fresh_db) as db:
        modern_id = db.upsert_etymon("ton", "modern-english")
        db.upsert_etymon("tūn", "old-english")
        db.commit()
        bridge_phonological_oe(db, apply=True)
        modern_target = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (modern_id,)
        ).fetchone()["merged_into_id"]
    assert modern_target is None  # Modern English 'ton' untouched


def test_bridge_phonological_oe_dry_run_does_not_write(fresh_db: Path) -> None:
    """apply=False reports counts but doesn't persist merges."""
    from wyrd.generators.kenning.lexicon import bridge_phonological_oe

    with LexiconDB(fresh_db) as db:
        place_id = db.upsert_etymon("ton", "old-english")
        db.upsert_etymon("tūn", "old-english")
        db.commit()
        result = bridge_phonological_oe(db, apply=False)
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (place_id,)
        ).fetchone()["merged_into_id"]

    assert merged is None
    assert result["bridged"] == 1
    assert result["rows_written"] == 0


def test_bridge_phonological_oe_idempotent_apply_skips_unchanged(
    fresh_db: Path,
) -> None:
    """Re-running --apply against unchanged data writes zero rows."""
    from wyrd.generators.kenning.lexicon import bridge_phonological_oe

    with LexiconDB(fresh_db) as db:
        db.upsert_etymon("ton", "old-english")
        db.upsert_etymon("tūn", "old-english")
        db.commit()
        first = bridge_phonological_oe(db, apply=True)
        second = bridge_phonological_oe(db, apply=True)
    assert first["rows_written"] == 1
    assert second["rows_written"] == 0


def test_bridge_phonological_oe_then_cluster_cognates_includes_place_form(
    fresh_db: Path,
) -> None:
    """End-to-end: bridge OE place-name 'ton' onto Wiktionary canonical
    'tūn', then run redirect-aware cluster_cognates, and confirm the
    place-name etymon is rolled up via merged_into_id into the cluster
    that contains the Proto-Germanic ancestor."""
    from wyrd.generators.kenning.lexicon import (
        bridge_phonological_oe,
        cluster_cognates,
    )

    with LexiconDB(fresh_db) as db:
        db.upsert_source(id="wiktionary", title="Wiktionary")
        proto_id = db.upsert_etymon("*tūnaz", "proto-germanic")
        oe_canonical_id = db.upsert_etymon("tūn", "old-english")
        oe_place_id = db.upsert_etymon("ton", "old-english")
        db.conn.execute(
            "INSERT INTO etymon_descent (parent_id, child_id, edge_type, source_id) "
            "VALUES (?, ?, 'inheritance', 'wiktionary')",
            (proto_id, oe_canonical_id),
        )
        db.commit()

        bridge_phonological_oe(db, apply=True)
        cluster_cognates(db, apply=True)

        canonical_synset = db.conn.execute(
            "SELECT cognate_id FROM etymon WHERE id = ?", (oe_canonical_id,)
        ).fetchone()["cognate_id"]
        place_target = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (oe_place_id,)
        ).fetchone()["merged_into_id"]

    assert place_target == oe_canonical_id
    assert canonical_synset == proto_id


# --- wyrd-083 + wyrd-ft3 round-1 fix: chain-flatten -------------------


def test_bridge_generic_language_flattens_existing_redirect_chain(
    fresh_db: Path,
) -> None:
    """Regression for the round-1 critical finding: when bridging
    generic→target, any pre-existing OCR-tombstone that points AT the
    generic etymon must be re-routed to the target too. Without this
    flatten, a 2-deep chain X → generic → target forms, and the
    single-level COALESCE rollup in etymon_consensus / etymon_canonical
    splits witnesses."""
    from wyrd.generators.kenning.lexicon import bridge_generic_language

    with LexiconDB(fresh_db) as db:
        # Order matters: target_id < generic_id, so the bridge picks target.
        target_id = db.upsert_etymon("bun", "irish")
        generic_id = db.upsert_etymon("bun", "celtic")
        # Pre-existing OCR tombstone pointing at the generic celtic row
        # (e.g. an earlier normalize-ocr pass that merged 'BUN' → 'bun').
        old_ocr_loser_id = db.upsert_etymon("BUN", "celtic")
        db.conn.execute(
            "UPDATE etymon SET merged_into_id = ? WHERE id = ?",
            (generic_id, old_ocr_loser_id),
        )
        db.commit()

        bridge_generic_language(db, generic_lang="celtic", candidate_langs=("irish",), apply=True)
        # After bridge: BOTH the generic celtic AND the old OCR loser
        # must point at the target_id (the irish entry), so the rollup
        # chain stays at depth 1.
        rows = {
            r["id"]: r["merged_into_id"]
            for r in db.conn.execute("SELECT id, merged_into_id FROM etymon")
        }
    assert rows[generic_id] == target_id
    assert rows[old_ocr_loser_id] == target_id, (
        "OCR-tombstone wasn't re-routed; 2-deep chain remains."
    )


def test_bridge_phonological_oe_flattens_existing_redirect_chain(
    fresh_db: Path,
) -> None:
    """Same regression as bridge_generic_language: a pre-existing OCR
    tombstone pointing AT a place-name OE etymon must re-route to the
    Wiktionary canonical when the bridge fires."""
    from wyrd.generators.kenning.lexicon import bridge_phonological_oe

    with LexiconDB(fresh_db) as db:
        wiktionary_id = db.upsert_etymon("tūn", "old-english")
        place_id = db.upsert_etymon("ton", "old-english")
        # Pre-existing OCR tombstone (e.g. 'TON' → 'ton' from a prior pass).
        old_ocr_loser_id = db.upsert_etymon("TON", "old-english")
        db.conn.execute(
            "UPDATE etymon SET merged_into_id = ? WHERE id = ?",
            (place_id, old_ocr_loser_id),
        )
        db.commit()

        bridge_phonological_oe(db, apply=True)
        rows = {
            r["id"]: r["merged_into_id"]
            for r in db.conn.execute("SELECT id, merged_into_id FROM etymon")
        }
    assert rows[place_id] == wiktionary_id
    assert rows[old_ocr_loser_id] == wiktionary_id, (
        "OCR-tombstone wasn't re-routed; 2-deep chain remains."
    )


def test_bridge_generic_language_reparents_lemma_children(fresh_db: Path) -> None:
    """If an inflected etymon has lemma_id pointing at the generic
    etymon being bridged, the lemma_id is re-routed to the canonical
    target so etymon_consensus's single-level lemma_id rollup keeps
    inflected variants in the same group as the canonical."""
    from wyrd.generators.kenning.lexicon import bridge_generic_language

    with LexiconDB(fresh_db) as db:
        target_id = db.upsert_etymon("bun", "irish")
        generic_id = db.upsert_etymon("bun", "celtic")
        inflected_id = db.upsert_etymon("buin", "celtic")  # hypothetical inflection
        db.conn.execute(
            "UPDATE etymon SET lemma_id = ? WHERE id = ?",
            (generic_id, inflected_id),
        )
        db.commit()

        bridge_generic_language(db, generic_lang="celtic", candidate_langs=("irish",), apply=True)
        new_lemma = db.conn.execute(
            "SELECT lemma_id FROM etymon WHERE id = ?", (inflected_id,)
        ).fetchone()["lemma_id"]
    assert new_lemma == target_id


def test_bridge_phonological_oe_reparents_lemma_children(fresh_db: Path) -> None:
    """Same lemma_id re-parent for the OE phonological bridge."""
    from wyrd.generators.kenning.lexicon import bridge_phonological_oe

    with LexiconDB(fresh_db) as db:
        wiktionary_id = db.upsert_etymon("tūn", "old-english")
        place_id = db.upsert_etymon("ton", "old-english")
        inflected_id = db.upsert_etymon("tones", "old-english")
        db.conn.execute(
            "UPDATE etymon SET lemma_id = ? WHERE id = ?",
            (place_id, inflected_id),
        )
        db.commit()

        bridge_phonological_oe(db, apply=True)
        new_lemma = db.conn.execute(
            "SELECT lemma_id FROM etymon WHERE id = ?", (inflected_id,)
        ).fetchone()["lemma_id"]
    assert new_lemma == wiktionary_id


# --- CLI smoke tests for bridge-language / bridge-phonological-oe -----


def test_cli_bridge_language_dry_run(fresh_db: Path) -> None:
    """`lexicon bridge-language --generic celtic` (no --apply) reports
    counts without writing merged_into_id. Pinned so the dry-run path
    stays the default for operator safety."""
    with LexiconDB(fresh_db) as db:
        generic_id = db.upsert_etymon("bun", "celtic")
        db.upsert_etymon("bun", "irish")
        db.commit()

    result = CliRunner().invoke(
        kenning_cli,
        [
            "lexicon",
            "bridge-language",
            "--db",
            str(fresh_db),
            "--generic",
            "celtic",
            "--candidates",
            "irish",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "would bridge 1" in result.stderr
    assert "(dry-run; pass --apply to commit)" in result.stderr

    with LexiconDB(fresh_db) as db:
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (generic_id,)
        ).fetchone()["merged_into_id"]
    assert merged is None


def test_cli_bridge_language_apply_uses_celtic_default_candidates(
    fresh_db: Path,
) -> None:
    """`--generic celtic` with no --candidates resolves through the
    built-in _CELTIC_CANDIDATES_DEFAULT priority list and writes the
    merge when --apply is set."""
    with LexiconDB(fresh_db) as db:
        generic_id = db.upsert_etymon("bun", "celtic")
        # Proto-Celtic has higher priority than Irish in the default list.
        proto_id = db.upsert_etymon("bun", "proto-celtic")
        db.upsert_etymon("bun", "irish")
        db.commit()

    result = CliRunner().invoke(
        kenning_cli,
        [
            "lexicon",
            "bridge-language",
            "--db",
            str(fresh_db),
            "--generic",
            "celtic",
            "--apply",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "bridged 1" in result.stderr
    assert "rows_written" in result.stderr

    with LexiconDB(fresh_db) as db:
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (generic_id,)
        ).fetchone()["merged_into_id"]
    assert merged == proto_id


def test_cli_bridge_language_requires_candidates_for_unknown_generic(
    fresh_db: Path,
) -> None:
    """No built-in candidate list exists for arbitrary generic tags;
    the CLI must fail loudly with --candidates missing rather than
    silently no-op."""
    result = CliRunner().invoke(
        kenning_cli,
        [
            "lexicon",
            "bridge-language",
            "--db",
            str(fresh_db),
            "--generic",
            "germanic",
        ],
    )
    assert result.exit_code == 1
    assert "no built-in default for 'germanic'" in result.stderr


def test_cli_bridge_language_empty_candidates_string_exits_cleanly(
    fresh_db: Path,
) -> None:
    """`--candidates ""` resolves to an empty tuple and would raise
    ValueError from bridge_generic_language. The CLI must catch this
    case before the call and exit 1 with a readable message rather
    than dumping a stack trace."""
    result = CliRunner().invoke(
        kenning_cli,
        [
            "lexicon",
            "bridge-language",
            "--db",
            str(fresh_db),
            "--generic",
            "celtic",
            "--candidates",
            "",
        ],
    )
    assert result.exit_code == 1
    assert "empty list" in result.stderr


def test_cli_bridge_phonological_oe_dry_run(fresh_db: Path) -> None:
    """`lexicon bridge-phonological-oe` (no --apply) reports counts
    from the hand-curated table without writing merged_into_id."""
    with LexiconDB(fresh_db) as db:
        place_id = db.upsert_etymon("ton", "old-english")
        db.upsert_etymon("tūn", "old-english")
        db.commit()

    result = CliRunner().invoke(
        kenning_cli,
        ["lexicon", "bridge-phonological-oe", "--db", str(fresh_db)],
    )
    assert result.exit_code == 0, result.output
    assert "would bridge 1" in result.stderr
    assert "(dry-run; pass --apply to commit)" in result.stderr

    with LexiconDB(fresh_db) as db:
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (place_id,)
        ).fetchone()["merged_into_id"]
    assert merged is None


def test_cli_bridge_phonological_oe_apply_writes_merge(fresh_db: Path) -> None:
    """`--apply` commits the bridge for known place-name → wiktionary
    pairs and reports rows_written."""
    with LexiconDB(fresh_db) as db:
        place_id = db.upsert_etymon("ton", "old-english")
        target_id = db.upsert_etymon("tūn", "old-english")
        db.commit()

    result = CliRunner().invoke(
        kenning_cli,
        ["lexicon", "bridge-phonological-oe", "--db", str(fresh_db), "--apply"],
    )
    assert result.exit_code == 0, result.output
    assert "bridged 1" in result.stderr
    assert "rows_written" in result.stderr

    with LexiconDB(fresh_db) as db:
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (place_id,)
        ).fetchone()["merged_into_id"]
    assert merged == target_id


def test_cli_bridge_phonological_oe_warns_on_missing_target(fresh_db: Path) -> None:
    """When a table entry names a wiktionary target that hasn't been
    ingested, the CLI emits the missing_target warning so operators
    know to ingest more or extend the table."""
    with LexiconDB(fresh_db) as db:
        # Place-name form present, wiktionary target absent.
        db.upsert_etymon("ton", "old-english")
        db.commit()

    result = CliRunner().invoke(
        kenning_cli,
        ["lexicon", "bridge-phonological-oe", "--db", str(fresh_db), "--apply"],
    )
    assert result.exit_code == 0, result.output
    assert "warn:" in result.stderr
    assert "target that doesn't exist" in result.stderr


# --- wyrd-9ig: bridge_phonological_oe target_index redirect-following ---


def test_bridge_phonological_oe_resolves_through_tombstone_target(
    fresh_db: Path,
) -> None:
    """When the bridge-table value names a form that exists only as a
    tombstone (merged_into another canonical), the target_index must
    follow the merged_into_id chain and resolve to the live canonical.
    Without this fix, _OE_PHONOLOGICAL_BRIDGES entries like 'dale → dæl'
    silently miss when a prior OCR-cluster pass has merged dæl into a
    macron-stripped 'dael' canonical.

    Reproduces the production bug where 5+ source citations of 'dale'
    stayed unbridged because dæl was merged into dael by normalize-ocr."""
    from wyrd.generators.kenning.lexicon import bridge_phonological_oe

    with LexiconDB(fresh_db) as db:
        # Live canonical that the macroned form has been OCR-merged into.
        live_id = db.upsert_etymon("dael", "old-english")
        # The macroned form named in the bridge table, but it's a tombstone.
        tombstone_id = db.upsert_etymon("dæl", "old-english")
        db.conn.execute(
            "UPDATE etymon SET merged_into_id = ? WHERE id = ?",
            (live_id, tombstone_id),
        )
        # The place-name form the bridge wants to route.
        place_id = db.upsert_etymon("dale", "old-english")
        db.commit()
        result = bridge_phonological_oe(db, apply=True)
        merged_target = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (place_id,)
        ).fetchone()["merged_into_id"]

    # Place-name 'dale' must end up pointing at the LIVE canonical, not
    # the tombstone (which would create a 2-deep chain) and not None
    # (which is the pre-fix bug behavior).
    assert merged_target == live_id
    assert result["bridged"] == 1
    assert result["missing_target"] == 0


def test_bridge_phonological_oe_target_index_handles_multi_step_chain(
    fresh_db: Path,
) -> None:
    """Multi-step merged_into_id chains (A → B → C) must resolve all the
    way to the live canonical C, not stop at intermediate B. Pin so the
    target_index walk doesn't regress to single-step lookup."""
    from wyrd.generators.kenning.lexicon import bridge_phonological_oe

    with LexiconDB(fresh_db) as db:
        c_live = db.upsert_etymon("dael", "old-english")
        b_mid = db.upsert_etymon("daell", "old-english")  # arbitrary intermediate
        a_top = db.upsert_etymon("dæl", "old-english")
        db.conn.execute("UPDATE etymon SET merged_into_id = ? WHERE id = ?", (c_live, b_mid))
        db.conn.execute("UPDATE etymon SET merged_into_id = ? WHERE id = ?", (b_mid, a_top))
        place_id = db.upsert_etymon("dale", "old-english")
        db.commit()
        bridge_phonological_oe(db, apply=True)
        merged_target = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (place_id,)
        ).fetchone()["merged_into_id"]

    assert merged_target == c_live


def test_bridge_phonological_oe_target_index_skips_self_loop(fresh_db: Path) -> None:
    """A merged_into_id self-loop (data corruption) must not infinite-loop
    the target_index resolution. The walk should bail and treat the row
    as its own canonical."""
    from wyrd.generators.kenning.lexicon import bridge_phonological_oe

    with LexiconDB(fresh_db) as db:
        weird_id = db.upsert_etymon("ton", "old-english")  # bridge-table key
        target_id = db.upsert_etymon("tūn", "old-english")
        # Synthetic self-loop on the target.
        db.conn.execute("UPDATE etymon SET merged_into_id = ? WHERE id = ?", (target_id, target_id))
        db.commit()
        # Should not hang.
        bridge_phonological_oe(db, apply=True)
        # And ton should still successfully bridge to tūn.
        merged_target = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (weird_id,)
        ).fetchone()["merged_into_id"]

    assert merged_target == target_id


def test_bridge_phonological_oe_ey_bridges_to_macron_stripped_ieg(
    fresh_db: Path,
) -> None:
    """The bridge value for 'ey' was changed from 'īeg' (which never
    exists in the corpus — Wiktionary's macroned headword wasn't
    ingested in that form) to 'ieg' (the OCR-stripped form that IS
    canonical in the DB). Pin the corrected mapping so a regression
    can't accidentally re-introduce 'īeg' as the bridge value."""
    from wyrd.generators.kenning.lexicon import bridge_phonological_oe

    with LexiconDB(fresh_db) as db:
        ieg_id = db.upsert_etymon("ieg", "old-english")
        ey_id = db.upsert_etymon("ey", "old-english")
        db.commit()
        bridge_phonological_oe(db, apply=True)
        merged_target = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (ey_id,)
        ).fetchone()["merged_into_id"]

    assert merged_target == ieg_id


# --- wyrd-fkg: bridge_phonological_on ----------------------------------


def test_bridge_phonological_on_merges_known_pair(fresh_db: Path) -> None:
    """Smoke test: a known place-name → wiktionary ON pair (`by` → `býr`)
    gets merged when --apply is on."""
    from wyrd.generators.kenning.lexicon import bridge_phonological_on

    with LexiconDB(fresh_db) as db:
        target_id = db.upsert_etymon("býr", "old-norse")
        place_id = db.upsert_etymon("by", "old-norse")
        db.commit()
        result = bridge_phonological_on(db, apply=True)
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (place_id,)
        ).fetchone()["merged_into_id"]

    assert merged == target_id
    assert result["bridged"] == 1
    assert result["missing_target"] == 0
    assert result["rows_written"] == 1


@pytest.mark.parametrize(
    "place_form,wiktionary_form",
    [
        ("by", "býr"),
        ("byr", "býr"),
        ("holm", "holmr"),
        ("kirk", "kirkja"),
        ("dal", "dalr"),
        ("dale", "dalr"),
        ("gardr", "garðr"),
        ("garth", "garðr"),
        ("griss", "gríss"),
        ("skogr", "skógr"),
        ("tun", "tún"),
        ("vik", "vík"),
        ("dyr", "dýr"),
        ("hest", "hestr"),
        ("krokr", "krókr"),
        ("nord", "norðr"),
        ("rauthr", "rauðr"),
        ("vollr", "vǫllr"),
        ("hogg", "hǫgg"),
        ("buski", "buskr"),
        ("flad", "flatr"),
        ("hain", "hagi"),
    ],
)
def test_bridge_phonological_on_table_entries_resolve(
    fresh_db: Path, place_form: str, wiktionary_form: str
) -> None:
    """Each high-witness ON bridge-table entry pinned individually so a
    typo or table edit surfaces against the specific pair it broke."""
    from wyrd.generators.kenning.lexicon import bridge_phonological_on

    with LexiconDB(fresh_db) as db:
        place_id = db.upsert_etymon(place_form, "old-norse")
        target_id = db.upsert_etymon(wiktionary_form, "old-norse")
        db.commit()
        bridge_phonological_on(db, apply=True)
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (place_id,)
        ).fetchone()["merged_into_id"]
    assert merged == target_id, f"{place_form!r} did not bridge to {wiktionary_form!r}"


def test_bridge_phonological_on_no_target_increments_missing_target(
    fresh_db: Path,
) -> None:
    """If the table names a target form that doesn't exist as a canonical
    ON etymon (because it hasn't been ingested), missing_target ticks."""
    from wyrd.generators.kenning.lexicon import bridge_phonological_on

    with LexiconDB(fresh_db) as db:
        place_id = db.upsert_etymon("by", "old-norse")  # 'by' → 'býr'
        db.commit()  # No 'býr' etymon present.
        result = bridge_phonological_on(db, apply=True)
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (place_id,)
        ).fetchone()["merged_into_id"]

    assert merged is None
    assert result["bridged"] == 0
    assert result["missing_target"] == 1


def test_bridge_phonological_on_unmatched_form_left_alone(fresh_db: Path) -> None:
    """An ON etymon whose canonical_form isn't in the bridge table is
    counted as unmatched (silent — no missing_target signal)."""
    from wyrd.generators.kenning.lexicon import bridge_phonological_on

    with LexiconDB(fresh_db) as db:
        unknown_id = db.upsert_etymon("xyzzy_unmapped_on", "old-norse")
        db.commit()
        result = bridge_phonological_on(db, apply=True)
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (unknown_id,)
        ).fetchone()["merged_into_id"]

    assert merged is None
    assert result["bridged"] == 0
    assert result["unmatched"] >= 1
    assert result["missing_target"] == 0


def test_bridge_phonological_on_ignores_other_languages(fresh_db: Path) -> None:
    """Bridge only touches old-norse rows; OE / Modern English / etc.
    are untouched even if their canonical_form happens to match a key."""
    from wyrd.generators.kenning.lexicon import bridge_phonological_on

    with LexiconDB(fresh_db) as db:
        oe_by = db.upsert_etymon("by", "old-english")  # accidental form match
        on_target = db.upsert_etymon("býr", "old-norse")
        on_by = db.upsert_etymon("by", "old-norse")
        db.commit()
        bridge_phonological_on(db, apply=True)
        oe_state = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (oe_by,)
        ).fetchone()["merged_into_id"]
        on_state = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (on_by,)
        ).fetchone()["merged_into_id"]

    assert oe_state is None  # OE 'by' untouched
    assert on_state == on_target  # ON 'by' bridged


def test_bridge_phonological_on_dry_run_does_not_write(fresh_db: Path) -> None:
    """apply=False reports counts without writing merged_into_id."""
    from wyrd.generators.kenning.lexicon import bridge_phonological_on

    with LexiconDB(fresh_db) as db:
        db.upsert_etymon("býr", "old-norse")
        place_id = db.upsert_etymon("by", "old-norse")
        db.commit()
        result = bridge_phonological_on(db, apply=False)
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (place_id,)
        ).fetchone()["merged_into_id"]

    assert merged is None
    assert result["bridged"] == 1  # would-bridge count
    assert result["rows_written"] == 0
    assert result["applied"] is False


def test_bridge_phonological_on_idempotent_apply_skips_unchanged(
    fresh_db: Path,
) -> None:
    """Re-running apply on an already-bridged corpus writes 0 new rows."""
    from wyrd.generators.kenning.lexicon import bridge_phonological_on

    with LexiconDB(fresh_db) as db:
        db.upsert_etymon("býr", "old-norse")
        db.upsert_etymon("by", "old-norse")
        db.commit()
        first = bridge_phonological_on(db, apply=True)
        second = bridge_phonological_on(db, apply=True)
    assert first["rows_written"] >= 1
    assert second["rows_written"] == 0


def test_bridge_phonological_on_resolves_through_tombstone_target(
    fresh_db: Path,
) -> None:
    """Same redirect-follow guarantee as the OE bridge: when the table
    value names a tombstone (e.g. þorp merged into thorp), the lookup
    must walk the chain and route to the live canonical."""
    from wyrd.generators.kenning.lexicon import bridge_phonological_on

    with LexiconDB(fresh_db) as db:
        live_id = db.upsert_etymon("thorp", "old-norse")  # the live canonical
        tombstone_id = db.upsert_etymon("þorp", "old-norse")
        db.conn.execute(
            "UPDATE etymon SET merged_into_id = ? WHERE id = ?",
            (live_id, tombstone_id),
        )
        place_id = db.upsert_etymon("thorpe", "old-norse")
        db.commit()
        result = bridge_phonological_on(db, apply=True)
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (place_id,)
        ).fetchone()["merged_into_id"]

    assert merged == live_id
    assert result["bridged"] == 1


def test_bridge_phonological_on_flattens_existing_redirect_chain(
    fresh_db: Path,
) -> None:
    """Chain-flatten: if a row already redirects AT the place-name etymon
    being bridged, the bridge re-routes that redirect onto the canonical
    target so no 2-deep chain forms (which would split witnesses in the
    single-level COALESCE rollup)."""
    from wyrd.generators.kenning.lexicon import bridge_phonological_on

    with LexiconDB(fresh_db) as db:
        target_id = db.upsert_etymon("býr", "old-norse")
        place_id = db.upsert_etymon("by", "old-norse")
        upstream_id = db.upsert_etymon("by-variant", "old-norse")
        db.conn.execute(
            "UPDATE etymon SET merged_into_id = ? WHERE id = ?",
            (place_id, upstream_id),
        )
        db.commit()
        bridge_phonological_on(db, apply=True)
        upstream_after = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (upstream_id,)
        ).fetchone()["merged_into_id"]
    assert upstream_after == target_id


def test_bridge_phonological_on_reparents_lemma_children(fresh_db: Path) -> None:
    """Same lemma_id re-parent for the ON phonological bridge."""
    from wyrd.generators.kenning.lexicon import bridge_phonological_on

    with LexiconDB(fresh_db) as db:
        wiktionary_id = db.upsert_etymon("býr", "old-norse")
        place_id = db.upsert_etymon("by", "old-norse")
        inflected_id = db.upsert_etymon("byjar", "old-norse")  # arbitrary inflection
        db.conn.execute(
            "UPDATE etymon SET lemma_id = ? WHERE id = ?",
            (place_id, inflected_id),
        )
        db.commit()
        bridge_phonological_on(db, apply=True)
        new_lemma = db.conn.execute(
            "SELECT lemma_id FROM etymon WHERE id = ?", (inflected_id,)
        ).fetchone()["lemma_id"]
    assert new_lemma == wiktionary_id


# --- CLI smoke tests for bridge-phonological-on -----------------------


def test_cli_bridge_phonological_on_dry_run(fresh_db: Path) -> None:
    """`lexicon bridge-phonological-on` (no --apply) reports counts
    without writing merged_into_id."""
    with LexiconDB(fresh_db) as db:
        place_id = db.upsert_etymon("by", "old-norse")
        db.upsert_etymon("býr", "old-norse")
        db.commit()

    result = CliRunner().invoke(
        kenning_cli,
        ["lexicon", "bridge-phonological-on", "--db", str(fresh_db)],
    )
    assert result.exit_code == 0, result.output
    assert "would bridge 1" in result.stderr
    assert "(dry-run; pass --apply to commit)" in result.stderr

    with LexiconDB(fresh_db) as db:
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (place_id,)
        ).fetchone()["merged_into_id"]
    assert merged is None


def test_cli_bridge_phonological_on_apply_writes_merge(fresh_db: Path) -> None:
    """`--apply` commits the bridge for known place-name → wiktionary
    pairs and reports rows_written."""
    with LexiconDB(fresh_db) as db:
        place_id = db.upsert_etymon("by", "old-norse")
        target_id = db.upsert_etymon("býr", "old-norse")
        db.commit()

    result = CliRunner().invoke(
        kenning_cli,
        ["lexicon", "bridge-phonological-on", "--db", str(fresh_db), "--apply"],
    )
    assert result.exit_code == 0, result.output
    assert "bridged 1" in result.stderr
    assert "rows_written" in result.stderr

    with LexiconDB(fresh_db) as db:
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (place_id,)
        ).fetchone()["merged_into_id"]
    assert merged == target_id


def test_cli_bridge_phonological_on_warns_on_missing_target(fresh_db: Path) -> None:
    """When the table names a wiktionary target that hasn't been ingested,
    the CLI emits the missing_target warning."""
    with LexiconDB(fresh_db) as db:
        # Place-name form present, wiktionary target absent.
        db.upsert_etymon("by", "old-norse")
        db.commit()

    result = CliRunner().invoke(
        kenning_cli,
        ["lexicon", "bridge-phonological-on", "--db", str(fresh_db), "--apply"],
    )
    assert result.exit_code == 0, result.output
    assert "warn:" in result.stderr
    assert "target that doesn't exist" in result.stderr


# --- wyrd-7k4: bridge_celtic_forms --------------------------------


def test_bridge_celtic_forms_genitive_to_lemma(fresh_db: Path) -> None:
    """Smoke test: a known Goidelic genitive (`choill` = gen of coill
    'wood') bridges to the Irish lemma."""
    from wyrd.generators.kenning.lexicon import bridge_celtic_forms

    with LexiconDB(fresh_db) as db:
        irish_id = db.upsert_etymon("coill", "irish")
        # Pretend the Irish entry is clustered (has a synset).
        db.conn.execute("UPDATE etymon SET cognate_id = id WHERE id = ?", (irish_id,))
        choill_id = db.upsert_etymon("choill", "celtic")
        db.commit()
        result = bridge_celtic_forms(db, apply=True)
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (choill_id,)
        ).fetchone()["merged_into_id"]

    assert merged == irish_id
    assert result["bridged"] == 1
    assert result["missing_target"] == 0


def test_bridge_celtic_forms_prefers_clustered_target(fresh_db: Path) -> None:
    """When the lemma exists in multiple candidate languages, the bridge
    must pick the one with a non-NULL cognate_id (clustered) over an
    unclustered alternative — even if priority order would name the
    unclustered one first.

    This is the wyrd-083 stub-target bug: the original celtic bridge
    picked old-irish/mac (no synset, isolated stub) over irish/mac
    (synset 87349, clustered) because old-irish came first in the
    priority list. The new bridge prefers clustered."""
    from wyrd.generators.kenning.lexicon import bridge_celtic_forms

    with LexiconDB(fresh_db) as db:
        # Higher-priority candidate (per default order: irish first), but
        # we'll un-cluster it and cluster the lower-priority one to
        # exercise the prefer-clustered branch.
        db.upsert_etymon("mac", "irish")  # higher-priority but unclustered
        old_irish_clustered = db.upsert_etymon("mac", "old-irish")
        # Cluster only the old-irish entry.
        db.conn.execute(
            "UPDATE etymon SET cognate_id = id WHERE id = ?",
            (old_irish_clustered,),
        )
        # Add a celtic form whose lemma is 'mac' via the Anglicized
        # 'mhic' (lenited gen of mac) entry in _CELTIC_FORM_BRIDGES.
        celtic_mhic = db.upsert_etymon("mhic", "celtic")
        db.commit()
        bridge_celtic_forms(db, apply=True)
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (celtic_mhic,)
        ).fetchone()["merged_into_id"]

    # Despite irish/mac being higher-priority, the clustered old-irish/mac
    # is chosen because the prefer-clustered logic kicks in.
    assert merged == old_irish_clustered


def test_bridge_celtic_forms_falls_back_to_unclustered(
    fresh_db: Path,
) -> None:
    """If no candidate has a synset, the bridge falls back to the
    first-found by priority order (unclustered target). Prevents
    silently dropping bridges when the entire Wiktionary chain is stub."""
    from wyrd.generators.kenning.lexicon import bridge_celtic_forms

    with LexiconDB(fresh_db) as db:
        # Both candidates unclustered.
        irish_id = db.upsert_etymon("coill", "irish")
        db.upsert_etymon("coill", "scottish-gaelic")
        choill_id = db.upsert_etymon("choill", "celtic")
        db.commit()
        bridge_celtic_forms(db, apply=True)
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (choill_id,)
        ).fetchone()["merged_into_id"]

    # Falls back to irish (priority-order first).
    assert merged == irish_id


def test_bridge_celtic_forms_reroutes_existing_stub_bridge(
    fresh_db: Path,
) -> None:
    """The killer feature: if a celtic etymon was previously bridged to
    an unclustered stub (e.g. by an earlier wyrd-083 run), this pass
    re-routes it via chain-flatten to the clustered alternative.

    The chain-flatten OR-clause `WHERE id = ? OR merged_into_id = ?`
    catches the celtic etymon AND any rows that were already redirected
    onto it (which would now form a 2-deep chain otherwise)."""
    from wyrd.generators.kenning.lexicon import bridge_celtic_forms

    with LexiconDB(fresh_db) as db:
        # The pre-existing wyrd-083 stub-bridge target (unclustered).
        stub_id = db.upsert_etymon("mac", "old-irish")
        # The clustered alternative we want to land on.
        clustered_id = db.upsert_etymon("mac", "irish")
        db.conn.execute("UPDATE etymon SET cognate_id = id WHERE id = ?", (clustered_id,))
        # Celtic etymon already pointed at the stub (simulating wyrd-083).
        celtic_mhic = db.upsert_etymon("mhic", "celtic")
        db.conn.execute(
            "UPDATE etymon SET merged_into_id = ? WHERE id = ?",
            (stub_id, celtic_mhic),
        )
        db.commit()
        bridge_celtic_forms(db, apply=True)
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (celtic_mhic,)
        ).fetchone()["merged_into_id"]

    assert merged == clustered_id


def test_bridge_celtic_forms_dry_run_does_not_write(fresh_db: Path) -> None:
    """apply=False reports counts without writing merged_into_id."""
    from wyrd.generators.kenning.lexicon import bridge_celtic_forms

    with LexiconDB(fresh_db) as db:
        db.upsert_etymon("coill", "irish")
        choill_id = db.upsert_etymon("choill", "celtic")
        db.commit()
        result = bridge_celtic_forms(db, apply=False)
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (choill_id,)
        ).fetchone()["merged_into_id"]

    assert merged is None
    assert result["bridged"] == 1
    assert result["rows_written"] == 0
    assert result["applied"] is False


def test_bridge_celtic_forms_idempotent_apply_skips_unchanged(
    fresh_db: Path,
) -> None:
    """Re-running apply on an already-bridged corpus writes 0 new rows."""
    from wyrd.generators.kenning.lexicon import bridge_celtic_forms

    with LexiconDB(fresh_db) as db:
        db.upsert_etymon("coill", "irish")
        db.upsert_etymon("choill", "celtic")
        db.commit()
        first = bridge_celtic_forms(db, apply=True)
        second = bridge_celtic_forms(db, apply=True)
    assert first["rows_written"] >= 1
    assert second["rows_written"] == 0


def test_bridge_celtic_forms_unmatched_celtic_form_left_alone(
    fresh_db: Path,
) -> None:
    """A celtic etymon whose canonical_form isn't in the inflection table
    is counted as `unmatched` (silent — table makes no claim)."""
    from wyrd.generators.kenning.lexicon import bridge_celtic_forms

    with LexiconDB(fresh_db) as db:
        unknown_id = db.upsert_etymon("xyzzy_unknown_celtic", "celtic")
        db.commit()
        result = bridge_celtic_forms(db, apply=True)
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (unknown_id,)
        ).fetchone()["merged_into_id"]
    assert merged is None
    assert result["unmatched"] >= 1
    assert result["missing_target"] == 0


def test_bridge_celtic_forms_missing_target_increments_counter(
    fresh_db: Path,
) -> None:
    """When the table names a lemma but no candidate-language etymon
    exists with that form, missing_target ticks (operator signal)."""
    from wyrd.generators.kenning.lexicon import bridge_celtic_forms

    with LexiconDB(fresh_db) as db:
        # 'choill' is in the table → 'coill', but no Irish/SG/Welsh
        # entry for 'coill' exists.
        db.upsert_etymon("choill", "celtic")
        db.commit()
        result = bridge_celtic_forms(db, apply=True)

    assert result["bridged"] == 0
    assert result["missing_target"] == 1


def test_bridge_celtic_forms_resolves_through_tombstone_lemma(
    fresh_db: Path,
) -> None:
    """Candidate-index regression: when the lemma named in the bridge
    table exists in a candidate language as a TOMBSTONE (already
    OCR-merged into a canonical), the resolve step must walk the
    merged_into_id chain so the bridge routes to the LIVE canonical,
    not to the tombstone (which would create a 2-deep chain that the
    single-level COALESCE rollup would split).

    This pins the redirect-resolve loop in the candidate_index build."""
    from wyrd.generators.kenning.lexicon import bridge_celtic_forms

    with LexiconDB(fresh_db) as db:
        # The live canonical Irish lemma (clustered).
        live_id = db.upsert_etymon("coill-canonical", "irish")
        db.conn.execute("UPDATE etymon SET cognate_id = id WHERE id = ?", (live_id,))
        # The tombstone Irish entry that the bridge table value 'coill'
        # actually names — already OCR-merged onto the live canonical.
        tombstone_id = db.upsert_etymon("coill", "irish")
        db.conn.execute(
            "UPDATE etymon SET merged_into_id = ? WHERE id = ?",
            (live_id, tombstone_id),
        )
        celtic_choill = db.upsert_etymon("choill", "celtic")
        db.commit()
        bridge_celtic_forms(db, apply=True)
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?",
            (celtic_choill,),
        ).fetchone()["merged_into_id"]

    # The bridge resolved through tombstone → live canonical.
    assert merged == live_id


def test_bridge_celtic_forms_self_bridge_is_noop(fresh_db: Path) -> None:
    """If the celtic etymon and the resolved target happen to be the
    SAME etymon row (same id), the bridge is a no-op — bridges count
    excludes it. Pin the `target_id == row['id']` guard."""
    with LexiconDB(fresh_db) as db:
        # One celtic row whose canonical_form matches a table key, AND
        # the lemma the table names ALSO equals that same canonical_form,
        # AND there's no separate candidate-language entry — but we make
        # a celtic entry for the lemma itself. In practice the function
        # iterates ALL celtic rows; we exercise the self-bridge guard by
        # using a candidate-language match that happens to be the same id.
        # Concretely: 'cu' (celtic) → 'cú' lemma. If we add 'cú' as the
        # lemma in irish AND the celtic 'cu' is identified, no self-bridge
        # because they're different rows. The self-bridge guard only fires
        # if `target_id == row['id']`, which requires both to be the SAME
        # etymon — only possible if a celtic row's lemma resolves back to
        # itself. We construct that: 'cu' as both celtic and the only
        # candidate by overriding the candidate_langs to include 'celtic'.
        celtic_cu = db.upsert_etymon("cu", "celtic")
        db.upsert_etymon("cú", "irish")
        db.commit()
        # Override candidate_langs to include 'celtic' so the celtic
        # entry itself becomes a candidate. With only 'cu' (celtic)
        # mapping to 'cú', and the 'cú' (irish) being a different row,
        # the bridge would fire. To exercise the self-guard we need the
        # candidate to BE the celtic row. Use a custom table where the
        # form maps to its own canonical_form.
        from wyrd.generators.kenning.lexicon import bridge_celtic_forms as bic

        result = bic(
            db,
            apply=True,
            table={"cu": "cu"},  # self-mapping
            candidate_langs=("celtic",),
        )
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (celtic_cu,)
        ).fetchone()["merged_into_id"]

    # Self-bridge guard kept the row untouched.
    assert merged is None
    assert result["bridged"] == 0


def test_bridge_celtic_forms_iterates_tombstones(fresh_db: Path) -> None:
    """Unlike same-language phonological bridges, this bridge iterates
    ALL celtic rows (canonical + tombstones). A tombstone celtic row
    whose stub target is unclustered must still be considered for re-route
    via the chain-flatten OR-clause.

    This pins that the bridge does NOT filter out tombstones up-front."""
    from wyrd.generators.kenning.lexicon import bridge_celtic_forms

    with LexiconDB(fresh_db) as db:
        clustered = db.upsert_etymon("mac", "irish")
        db.conn.execute("UPDATE etymon SET cognate_id = id WHERE id = ?", (clustered,))
        # 'mhic' celtic → 'mac' lemma. Tombstone it onto a totally
        # unrelated etymon to simulate corrupt prior bridge.
        bystander = db.upsert_etymon("zzz_bystander", "celtic")
        celtic_mhic = db.upsert_etymon("mhic", "celtic")
        db.conn.execute(
            "UPDATE etymon SET merged_into_id = ? WHERE id = ?",
            (bystander, celtic_mhic),
        )
        db.commit()
        result = bridge_celtic_forms(db, apply=True)

    # Examined count includes BOTH the canonical bystander and the
    # tombstone celtic_mhic (irish/mac is non-celtic, doesn't count).
    assert result["examined"] == 2
    # Tombstone re-routed to clustered target (chain-flatten OR-clause).
    with LexiconDB(fresh_db) as db2:
        actual = db2.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (celtic_mhic,)
        ).fetchone()["merged_into_id"]
    assert actual == clustered


# --- CLI smoke for bridge-celtic-forms ----------------------------


def test_cli_bridge_celtic_forms_dry_run(fresh_db: Path) -> None:
    """`lexicon bridge-celtic-forms` (no --apply) reports counts
    without writing merged_into_id."""
    with LexiconDB(fresh_db) as db:
        db.upsert_etymon("coill", "irish")
        place_id = db.upsert_etymon("choill", "celtic")
        db.commit()

    result = CliRunner().invoke(
        kenning_cli,
        ["lexicon", "bridge-celtic-forms", "--db", str(fresh_db)],
    )
    assert result.exit_code == 0, result.output
    assert "would bridge 1" in result.stderr
    assert "(dry-run; pass --apply to commit)" in result.stderr

    with LexiconDB(fresh_db) as db:
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (place_id,)
        ).fetchone()["merged_into_id"]
    assert merged is None


def test_cli_bridge_celtic_forms_apply_writes_merge(fresh_db: Path) -> None:
    """`--apply` commits the bridge and reports rows_written."""
    with LexiconDB(fresh_db) as db:
        target_id = db.upsert_etymon("coill", "irish")
        place_id = db.upsert_etymon("choill", "celtic")
        db.commit()

    result = CliRunner().invoke(
        kenning_cli,
        ["lexicon", "bridge-celtic-forms", "--db", str(fresh_db), "--apply"],
    )
    assert result.exit_code == 0, result.output
    assert "bridged 1" in result.stderr
    assert "rows_written" in result.stderr

    with LexiconDB(fresh_db) as db:
        merged = db.conn.execute(
            "SELECT merged_into_id FROM etymon WHERE id = ?", (place_id,)
        ).fetchone()["merged_into_id"]
    assert merged == target_id


def test_cli_bridge_celtic_forms_warns_on_missing_target(
    fresh_db: Path,
) -> None:
    """When a table entry names a lemma that doesn't exist in any
    candidate language, the CLI emits the missing_target warning."""
    with LexiconDB(fresh_db) as db:
        # 'choill' → 'coill', but no 'coill' lemma anywhere.
        db.upsert_etymon("choill", "celtic")
        db.commit()

    result = CliRunner().invoke(
        kenning_cli,
        ["lexicon", "bridge-celtic-forms", "--db", str(fresh_db), "--apply"],
    )
    assert result.exit_code == 0, result.output
    assert "warn:" in result.stderr
    assert "doesn't exist" in result.stderr


# --- meaning_synset (wyrd-7tz Phase 1) -------------------------------------


def test_init_schema_creates_meaning_synset_tables(fresh_db: Path) -> None:
    """Fresh DB must have both meaning_synset + etymon_meaning_synset tables.
    Pin so a regression that drops either table from lexicon.sql surfaces
    immediately."""
    conn = sqlite3.connect(fresh_db)
    try:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
        tables = {row[0] for row in cur.fetchall()}
    finally:
        conn.close()
    assert "meaning_synset" in tables
    assert "etymon_meaning_synset" in tables


def test_init_schema_meaning_synset_has_unique_canonical_label(fresh_db: Path) -> None:
    """canonical_label is the natural key — duplicate inserts must fail."""
    with LexiconDB(fresh_db) as db:
        db.conn.execute(
            "INSERT INTO meaning_synset (canonical_label) VALUES (?)", ("water/flowing",)
        )
        with pytest.raises(sqlite3.IntegrityError):
            db.conn.execute(
                "INSERT INTO meaning_synset (canonical_label) VALUES (?)", ("water/flowing",)
            )


def test_init_schema_etymon_meaning_synset_fit_check_constraint(fresh_db: Path) -> None:
    """fit must be 'core' or 'peripheral' — bad values must fail at the
    DB layer so a buggy caller can't silently insert garbage."""
    with LexiconDB(fresh_db) as db:
        db.conn.execute(
            "INSERT INTO meaning_synset (canonical_label) VALUES (?)", ("water/flowing",)
        )
        cognate_id = db.conn.execute(
            "SELECT id FROM meaning_synset WHERE canonical_label = ?", ("water/flowing",)
        ).fetchone()["id"]
        etymon_id = db.upsert_etymon("wæter", "old-english")
        with pytest.raises(sqlite3.IntegrityError):
            db.conn.execute(
                """
                INSERT INTO etymon_meaning_synset (etymon_id, meaning_synset_id, fit)
                VALUES (?, ?, ?)
                """,
                (etymon_id, cognate_id, "fuzzy"),
            )


def test_seed_meaning_synsets_inserts_full_catalog(fresh_db: Path) -> None:
    """Seed loader must populate the meaning_synset table from the bundled
    JSON catalog. The bundled catalog has 50+ entries — check for a known
    one to confirm the read+insert path works end-to-end."""
    with LexiconDB(fresh_db) as db:
        result = seed_meaning_synsets(db)
    assert result["inserted"] >= 50
    assert result["updated"] == 0
    assert result["unchanged"] == 0
    with LexiconDB(fresh_db) as db:
        labels = {
            row["canonical_label"]
            for row in db.conn.execute("SELECT canonical_label FROM meaning_synset")
        }
    # Spot-check spec'd labels.
    assert "water/flowing" in labels
    assert "water/body" in labels
    assert "hill/rounded" in labels
    assert "enclosure/farmstead" in labels


def test_seed_meaning_synsets_resolves_hypernyms(fresh_db: Path) -> None:
    """The two-pass seed walks the catalog so 'water/flowing' (declared
    with hypernym='water') gets its hypernym_id resolved against the
    'water' row inserted in pass 1. Pin via FK lookup."""
    with LexiconDB(fresh_db) as db:
        seed_meaning_synsets(db)
        water_id = db.conn.execute(
            "SELECT id FROM meaning_synset WHERE canonical_label = ?", ("water",)
        ).fetchone()["id"]
        flowing_hypernym = db.conn.execute(
            "SELECT hypernym_id FROM meaning_synset WHERE canonical_label = ?",
            ("water/flowing",),
        ).fetchone()["hypernym_id"]
    assert flowing_hypernym == water_id


def test_seed_meaning_synsets_idempotent(fresh_db: Path) -> None:
    """Running the seed twice on the same DB is a no-op — the second run
    reports 0 inserted, 0 updated, all unchanged."""
    with LexiconDB(fresh_db) as db:
        seed_meaning_synsets(db)
        result_second = seed_meaning_synsets(db)
    assert result_second["inserted"] == 0
    assert result_second["updated"] == 0
    assert result_second["unchanged"] >= 50


def test_seed_meaning_synsets_updates_drifted_notes(fresh_db: Path) -> None:
    """If a row's notes have drifted from the catalog (e.g. someone
    edited the JSON to clarify a definition), the next seed run brings
    them into sync — counted as 'updated', not 'unchanged'."""
    with LexiconDB(fresh_db) as db:
        seed_meaning_synsets(db)
        db.conn.execute(
            "UPDATE meaning_synset SET notes = ? WHERE canonical_label = ?",
            ("STALE", "water/flowing"),
        )
        db.commit()
        result = seed_meaning_synsets(db)
    assert result["updated"] == 1
    with LexiconDB(fresh_db) as db:
        notes = db.conn.execute(
            "SELECT notes FROM meaning_synset WHERE canonical_label = ?", ("water/flowing",)
        ).fetchone()["notes"]
    assert notes != "STALE"
    assert "OE wæter" in notes


def test_assign_etymon_to_meaning_synset_inserts_first_time(fresh_db: Path) -> None:
    """First-time assignment returns True (inserted)."""
    with LexiconDB(fresh_db) as db:
        seed_meaning_synsets(db)
        etymon_id = db.upsert_etymon("wæter", "old-english")
        db.commit()
        inserted = assign_etymon_to_meaning_synset(db, etymon_id, "water/flowing")
    assert inserted is True


def test_assign_etymon_to_meaning_synset_idempotent(fresh_db: Path) -> None:
    """Re-assigning the same (etymon, synset, fit) triple is a no-op
    returning False (not inserted, not changed)."""
    with LexiconDB(fresh_db) as db:
        seed_meaning_synsets(db)
        etymon_id = db.upsert_etymon("wæter", "old-english")
        db.commit()
        assign_etymon_to_meaning_synset(db, etymon_id, "water/flowing")
        again = assign_etymon_to_meaning_synset(db, etymon_id, "water/flowing")
    assert again is False


def test_assign_etymon_to_meaning_synset_updates_fit(fresh_db: Path) -> None:
    """Re-assigning with a different fit updates the existing row in
    place, returns False (not a fresh insert)."""
    with LexiconDB(fresh_db) as db:
        seed_meaning_synsets(db)
        etymon_id = db.upsert_etymon("wæter", "old-english")
        db.commit()
        assign_etymon_to_meaning_synset(db, etymon_id, "water/flowing", fit="peripheral")
        result = assign_etymon_to_meaning_synset(db, etymon_id, "water/flowing", fit="core")
    assert result is False
    with LexiconDB(fresh_db) as db:
        synsets = get_meaning_synsets_for_etymon(db, etymon_id)
    assert synsets[0]["fit"] == "core"


def test_assign_etymon_to_meaning_synset_raises_on_unknown_etymon(fresh_db: Path) -> None:
    """Bad etymon_id must surface as a ValueError so a typo at the CLI
    surface doesn't silently insert nothing."""
    with LexiconDB(fresh_db) as db:
        seed_meaning_synsets(db)
        with pytest.raises(ValueError, match="unknown etymon_id"):
            assign_etymon_to_meaning_synset(db, 9999, "water/flowing")


def test_assign_etymon_to_meaning_synset_raises_on_unknown_synset(fresh_db: Path) -> None:
    """Bad synset_label must surface as a ValueError naming the bad
    label so the user sees what they typo'd."""
    with LexiconDB(fresh_db) as db:
        seed_meaning_synsets(db)
        etymon_id = db.upsert_etymon("wæter", "old-english")
        db.commit()
        with pytest.raises(ValueError, match="unknown meaning_synset label"):
            assign_etymon_to_meaning_synset(db, etymon_id, "water/imaginary")


def test_assign_etymon_to_meaning_synset_raises_on_bad_fit(fresh_db: Path) -> None:
    """Validation runs in Python before SQLite is touched — bad fit
    raises ValueError, not IntegrityError."""
    with LexiconDB(fresh_db) as db:
        seed_meaning_synsets(db)
        etymon_id = db.upsert_etymon("wæter", "old-english")
        db.commit()
        with pytest.raises(ValueError, match="fit must be"):
            assign_etymon_to_meaning_synset(db, etymon_id, "water/flowing", fit="fuzzy")


def test_get_meaning_synsets_for_etymon_orders_core_first(fresh_db: Path) -> None:
    """Multi-synset membership must surface 'core' before 'peripheral'
    so callers can pick the primary sense without sorting again."""
    with LexiconDB(fresh_db) as db:
        seed_meaning_synsets(db)
        etymon_id = db.upsert_etymon("wæter", "old-english")
        db.commit()
        # water/body is alphabetically before water/flowing, but assigned
        # peripheral; water/flowing is core.
        assign_etymon_to_meaning_synset(db, etymon_id, "water/body", fit="peripheral")
        assign_etymon_to_meaning_synset(db, etymon_id, "water/flowing", fit="core")
        synsets = get_meaning_synsets_for_etymon(db, etymon_id)
    assert [s["canonical_label"] for s in synsets] == ["water/flowing", "water/body"]


def test_get_meaning_preserving_candidates_returns_cross_language(fresh_db: Path) -> None:
    """Spec'd regression test: OE wæter + ON bekkr both in water/flowing
    AND OE mere in water/body. Candidates for wæter include bekkr but
    NOT mere."""
    with LexiconDB(fresh_db) as db:
        seed_meaning_synsets(db)
        waeter = db.upsert_etymon("wæter", "old-english")
        stream = db.upsert_etymon("strēam", "old-english")
        mere = db.upsert_etymon("mere", "old-english")
        bekkr = db.upsert_etymon("bekkr", "old-norse")
        db.commit()
        assign_etymon_to_meaning_synset(db, waeter, "water/flowing")
        assign_etymon_to_meaning_synset(db, stream, "water/flowing")
        assign_etymon_to_meaning_synset(db, mere, "water/body")
        assign_etymon_to_meaning_synset(db, bekkr, "water/flowing")
        rows = get_meaning_preserving_candidates(db, waeter)
    forms = {r["canonical_form"] for r in rows}
    assert "strēam" in forms
    assert "bekkr" in forms
    assert "mere" not in forms
    assert "wæter" not in forms  # default excludes self


def test_get_meaning_preserving_candidates_target_language_filter(fresh_db: Path) -> None:
    """target_language filter restricts candidates to one language —
    the anglicize/foreignize transform's primary use case."""
    with LexiconDB(fresh_db) as db:
        seed_meaning_synsets(db)
        waeter = db.upsert_etymon("wæter", "old-english")
        stream = db.upsert_etymon("strēam", "old-english")
        bekkr = db.upsert_etymon("bekkr", "old-norse")
        db.commit()
        for eid in (waeter, stream, bekkr):
            assign_etymon_to_meaning_synset(db, eid, "water/flowing")
        rows = get_meaning_preserving_candidates(db, waeter, target_language="old-norse")
    assert {r["canonical_form"] for r in rows} == {"bekkr"}


def test_get_meaning_preserving_candidates_fit_filter(fresh_db: Path) -> None:
    """fit='core' restricts BOTH ends of the join — only candidates
    where the source AND the candidate are in the synset as 'core'."""
    with LexiconDB(fresh_db) as db:
        seed_meaning_synsets(db)
        waeter = db.upsert_etymon("wæter", "old-english")
        stream = db.upsert_etymon("strēam", "old-english")
        burna = db.upsert_etymon("burna", "old-english")
        db.commit()
        assign_etymon_to_meaning_synset(db, waeter, "water/flowing", fit="core")
        assign_etymon_to_meaning_synset(db, stream, "water/flowing", fit="core")
        assign_etymon_to_meaning_synset(db, burna, "water/flowing", fit="peripheral")
        rows = get_meaning_preserving_candidates(db, waeter, fit="core")
    assert {r["canonical_form"] for r in rows} == {"strēam"}


def test_get_meaning_preserving_candidates_include_self(fresh_db: Path) -> None:
    """include_self=True keeps the source etymon in the result — useful
    for dedup-aware callers."""
    with LexiconDB(fresh_db) as db:
        seed_meaning_synsets(db)
        waeter = db.upsert_etymon("wæter", "old-english")
        stream = db.upsert_etymon("strēam", "old-english")
        db.commit()
        assign_etymon_to_meaning_synset(db, waeter, "water/flowing")
        assign_etymon_to_meaning_synset(db, stream, "water/flowing")
        rows = get_meaning_preserving_candidates(db, waeter, include_self=True)
    assert {r["canonical_form"] for r in rows} == {"wæter", "strēam"}


def test_get_meaning_preserving_candidates_empty_when_no_membership(fresh_db: Path) -> None:
    """An etymon with no synset memberships has no meaning-preserving
    candidates — empty list, no crash."""
    with LexiconDB(fresh_db) as db:
        seed_meaning_synsets(db)
        etymon_id = db.upsert_etymon("wæter", "old-english")
        db.commit()
        rows = get_meaning_preserving_candidates(db, etymon_id)
    assert rows == []


def test_list_meaning_synsets_with_member_counts(fresh_db: Path) -> None:
    """member_count column reports per-synset membership tally so a CLI
    user can spot empty/under-populated synsets."""
    with LexiconDB(fresh_db) as db:
        seed_meaning_synsets(db)
        waeter = db.upsert_etymon("wæter", "old-english")
        stream = db.upsert_etymon("strēam", "old-english")
        db.commit()
        assign_etymon_to_meaning_synset(db, waeter, "water/flowing")
        assign_etymon_to_meaning_synset(db, stream, "water/flowing")
        synsets = list_meaning_synsets(db, with_member_counts=True)
    flowing = [s for s in synsets if s["canonical_label"] == "water/flowing"][0]
    body = [s for s in synsets if s["canonical_label"] == "water/body"][0]
    assert flowing["member_count"] == 2
    assert body["member_count"] == 0


def test_migrate_schema_creates_meaning_synset_tables_on_legacy_db(tmp_path: Path) -> None:
    """A pre-wyrd-7tz DB without the meaning_synset tables should get
    them via migrate_schema — both tables created, both reported as
    applied=True."""
    # Build a DB with the WAL-pragma schema but DROP the new tables to
    # simulate a legacy state.
    db_path = tmp_path / "legacy.db"
    init_schema(db_path)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DROP TABLE etymon_meaning_synset")
        conn.execute("DROP TABLE meaning_synset")
        conn.commit()
    finally:
        conn.close()
    with LexiconDB(db_path) as db:
        applied = migrate_schema(db)
    assert applied["meaning_synset_table"] is True
    assert applied["etymon_meaning_synset_table"] is True
    # And re-running is a no-op.
    with LexiconDB(db_path) as db:
        applied_second = migrate_schema(db)
    assert applied_second["meaning_synset_table"] is False
    assert applied_second["etymon_meaning_synset_table"] is False


# --- CLI: lexicon synsets ---------------------------------------------------


def test_cli_lexicon_synsets_seed(fresh_db: Path) -> None:
    """`lexicon synsets seed` populates the catalog from bundled JSON."""
    result = CliRunner().invoke(kenning_cli, ["lexicon", "synsets", "seed", "--db", str(fresh_db)])
    assert result.exit_code == 0, result.output
    assert "inserted=" in result.output
    with LexiconDB(fresh_db) as db:
        n = db.conn.execute("SELECT COUNT(*) AS c FROM meaning_synset").fetchone()["c"]
    assert n >= 50


def test_cli_lexicon_synsets_list_with_counts(fresh_db: Path) -> None:
    """`synsets list --with-counts` shows per-synset membership tallies."""
    runner = CliRunner()
    runner.invoke(kenning_cli, ["lexicon", "synsets", "seed", "--db", str(fresh_db)])
    with LexiconDB(fresh_db) as db:
        eid = db.upsert_etymon("wæter", "old-english")
        db.commit()
        assign_etymon_to_meaning_synset(db, eid, "water/flowing")
    result = runner.invoke(
        kenning_cli,
        ["lexicon", "synsets", "list", "--db", str(fresh_db), "--with-counts"],
    )
    assert result.exit_code == 0, result.output
    assert "water/flowing" in result.output
    # The water/flowing line should show 1 member.
    flowing_line = next(ln for ln in result.output.splitlines() if "water/flowing" in ln)
    assert "1 members" in flowing_line


def test_cli_lexicon_synsets_assign_and_show(fresh_db: Path) -> None:
    """`synsets assign` writes a membership; `synsets show` reads it back."""
    runner = CliRunner()
    runner.invoke(kenning_cli, ["lexicon", "synsets", "seed", "--db", str(fresh_db)])
    with LexiconDB(fresh_db) as db:
        eid = db.upsert_etymon("wæter", "old-english")
        db.commit()
    assign_result = runner.invoke(
        kenning_cli,
        ["lexicon", "synsets", "assign", str(eid), "water/flowing", "--db", str(fresh_db)],
    )
    assert assign_result.exit_code == 0, assign_result.output
    assert "Added" in assign_result.output

    show_result = runner.invoke(
        kenning_cli,
        ["lexicon", "synsets", "show", str(eid), "--db", str(fresh_db)],
    )
    assert show_result.exit_code == 0, show_result.output
    assert "water/flowing" in show_result.output
    assert "core" in show_result.output


def test_cli_lexicon_synsets_assign_unknown_synset_exits_nonzero(fresh_db: Path) -> None:
    """A bad synset label at the CLI surface exits non-zero with a
    friendly error rather than spilling a Python traceback."""
    runner = CliRunner()
    runner.invoke(kenning_cli, ["lexicon", "synsets", "seed", "--db", str(fresh_db)])
    with LexiconDB(fresh_db) as db:
        eid = db.upsert_etymon("wæter", "old-english")
        db.commit()
    result = runner.invoke(
        kenning_cli,
        [
            "lexicon",
            "synsets",
            "assign",
            str(eid),
            "water/imaginary",
            "--db",
            str(fresh_db),
        ],
    )
    assert result.exit_code != 0
    assert "unknown meaning_synset label" in result.output


def test_cli_lexicon_synsets_candidates_cross_language(fresh_db: Path) -> None:
    """End-to-end: assign two etymons to the same synset, then ask for
    candidates of one — the other should appear."""
    runner = CliRunner()
    runner.invoke(kenning_cli, ["lexicon", "synsets", "seed", "--db", str(fresh_db)])
    with LexiconDB(fresh_db) as db:
        waeter = db.upsert_etymon("wæter", "old-english")
        bekkr = db.upsert_etymon("bekkr", "old-norse")
        db.commit()
        assign_etymon_to_meaning_synset(db, waeter, "water/flowing")
        assign_etymon_to_meaning_synset(db, bekkr, "water/flowing")
    result = runner.invoke(
        kenning_cli,
        ["lexicon", "synsets", "candidates", str(waeter), "--db", str(fresh_db)],
    )
    assert result.exit_code == 0, result.output
    assert "bekkr" in result.output
    assert "water/flowing" in result.output


def test_cli_lexicon_synsets_candidates_target_language(fresh_db: Path) -> None:
    """--target-language restricts the candidate pool to one language."""
    runner = CliRunner()
    runner.invoke(kenning_cli, ["lexicon", "synsets", "seed", "--db", str(fresh_db)])
    with LexiconDB(fresh_db) as db:
        waeter = db.upsert_etymon("wæter", "old-english")
        stream = db.upsert_etymon("strēam", "old-english")
        bekkr = db.upsert_etymon("bekkr", "old-norse")
        db.commit()
        for eid in (waeter, stream, bekkr):
            assign_etymon_to_meaning_synset(db, eid, "water/flowing")
    result = runner.invoke(
        kenning_cli,
        [
            "lexicon",
            "synsets",
            "candidates",
            str(waeter),
            "--target-language",
            "old-norse",
            "--db",
            str(fresh_db),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "bekkr" in result.output
    assert "strēam" not in result.output


# --- wyrd-7tz Phase 1 round-2 follow-ups -----------------------------------


def test_get_meaning_preserving_candidates_dedupe_collapses_multi_synset_pairs(
    fresh_db: Path,
) -> None:
    """A (target, candidate) pair sharing N synsets produces N rows
    raw, but ONE row with dedupe=True (default). Phase 3 transforms
    want one row per candidate; the raw cartesian is for audit only."""
    with LexiconDB(fresh_db) as db:
        seed_meaning_synsets(db)
        waeter = db.upsert_etymon("wæter", "old-english")
        bekkr = db.upsert_etymon("bekkr", "old-norse")
        db.commit()
        # Both etymons in TWO shared synsets — the cartesian would
        # produce 2 rows per candidate without dedupe.
        assign_etymon_to_meaning_synset(db, waeter, "water/flowing", fit="core")
        assign_etymon_to_meaning_synset(db, waeter, "water/body", fit="peripheral")
        assign_etymon_to_meaning_synset(db, bekkr, "water/flowing", fit="core")
        assign_etymon_to_meaning_synset(db, bekkr, "water/body", fit="peripheral")
        raw = get_meaning_preserving_candidates(db, waeter, dedupe=False)
        deduped = get_meaning_preserving_candidates(db, waeter)  # default dedupe=True
    assert len(raw) == 2
    assert len(deduped) == 1
    # Dedupe picks the strongest fit pair — core/core beats peripheral/peripheral.
    assert deduped[0]["target_fit"] == "core"
    assert deduped[0]["candidate_fit"] == "core"


def test_get_meaning_preserving_candidates_dedupe_picks_strongest_fit(
    fresh_db: Path,
) -> None:
    """When a pair shares two synsets, one core/core and one
    core/peripheral, dedupe picks the core/core row."""
    with LexiconDB(fresh_db) as db:
        seed_meaning_synsets(db)
        waeter = db.upsert_etymon("wæter", "old-english")
        bekkr = db.upsert_etymon("bekkr", "old-norse")
        db.commit()
        assign_etymon_to_meaning_synset(db, waeter, "water/flowing", fit="core")
        assign_etymon_to_meaning_synset(db, waeter, "water/body", fit="core")
        assign_etymon_to_meaning_synset(db, bekkr, "water/flowing", fit="core")
        assign_etymon_to_meaning_synset(db, bekkr, "water/body", fit="peripheral")
        rows = get_meaning_preserving_candidates(db, waeter)
    assert len(rows) == 1
    # core/core wins over core/peripheral.
    assert rows[0]["synset_label"] == "water/flowing"
    assert rows[0]["candidate_fit"] == "core"


def test_get_meaning_preserving_candidates_raises_on_bad_fit(fresh_db: Path) -> None:
    """A bad --fit value raises ValueError before the SQL is executed."""
    with LexiconDB(fresh_db) as db:
        seed_meaning_synsets(db)
        eid = db.upsert_etymon("wæter", "old-english")
        db.commit()
        with pytest.raises(ValueError, match="fit filter must be"):
            get_meaning_preserving_candidates(db, eid, fit="garbage")


def test_get_meaning_preserving_candidates_target_language_and_fit_combined(
    fresh_db: Path,
) -> None:
    """Both filters applied together — only candidates in the target
    language AND with matching fit on both ends survive."""
    with LexiconDB(fresh_db) as db:
        seed_meaning_synsets(db)
        waeter = db.upsert_etymon("wæter", "old-english")
        stream = db.upsert_etymon("strēam", "old-english")  # OE, core
        burna = db.upsert_etymon("burna", "old-english")  # OE, peripheral
        bekkr = db.upsert_etymon("bekkr", "old-norse")  # ON, core
        db.commit()
        assign_etymon_to_meaning_synset(db, waeter, "water/flowing", fit="core")
        assign_etymon_to_meaning_synset(db, stream, "water/flowing", fit="core")
        assign_etymon_to_meaning_synset(db, burna, "water/flowing", fit="peripheral")
        assign_etymon_to_meaning_synset(db, bekkr, "water/flowing", fit="core")
        # OE + core: only strēam (burna excluded by fit, bekkr by language).
        rows = get_meaning_preserving_candidates(
            db, waeter, target_language="old-english", fit="core"
        )
    assert {r["canonical_form"] for r in rows} == {"strēam"}


def test_seed_meaning_synsets_warns_on_unknown_hypernym(tmp_path: Path) -> None:
    """A seed entry pointing at a hypernym the catalog doesn't define
    (typo or out-of-catalog) emits a warning rather than silently
    dropping the link."""
    db_path = tmp_path / "hypernym_warn.db"
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        # Inject a synthetic catalog with a bad hypernym pointer by
        # patching _meaning_synsets_seed at module level — easier than
        # writing a temp JSON file.
        import warnings

        from wyrd.generators.kenning import lexicon as lex

        original = lex._meaning_synsets_seed
        try:
            lex._meaning_synsets_seed = lambda: {
                "synsets": [
                    {"label": "water"},
                    {"label": "water/flowing", "hypernym": "ocean"},  # 'ocean' undefined
                ]
            }
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                lex.seed_meaning_synsets(db)
        finally:
            lex._meaning_synsets_seed = original
    assert any("unknown hypernym 'ocean'" in str(w.message) for w in caught)


def test_seed_meaning_synsets_clears_removed_hypernym_link(tmp_path: Path) -> None:
    """If a hypernym is removed from the JSON between seed runs, the FK
    on the existing row gets NULL'd out — drift-aware like notes."""
    db_path = tmp_path / "hypernym_clear.db"
    init_schema(db_path)
    with LexiconDB(db_path) as db:
        from wyrd.generators.kenning import lexicon as lex

        original = lex._meaning_synsets_seed
        try:
            # First run: water/flowing → water hypernym
            lex._meaning_synsets_seed = lambda: {
                "synsets": [
                    {"label": "water"},
                    {"label": "water/flowing", "hypernym": "water"},
                ]
            }
            lex.seed_meaning_synsets(db)
            row = db.conn.execute(
                "SELECT hypernym_id FROM meaning_synset WHERE canonical_label = ?",
                ("water/flowing",),
            ).fetchone()
            assert row["hypernym_id"] is not None
            # Second run: hypernym removed from the catalog
            lex._meaning_synsets_seed = lambda: {
                "synsets": [
                    {"label": "water"},
                    {"label": "water/flowing"},  # no hypernym key
                ]
            }
            lex.seed_meaning_synsets(db)
        finally:
            lex._meaning_synsets_seed = original
        row = db.conn.execute(
            "SELECT hypernym_id FROM meaning_synset WHERE canonical_label = ?",
            ("water/flowing",),
        ).fetchone()
    assert row["hypernym_id"] is None


def test_cli_lexicon_synsets_assign_by_form_and_language(fresh_db: Path) -> None:
    """The --language flag lets a curator assign by canonical_form
    rather than looking up the integer id first."""
    runner = CliRunner()
    runner.invoke(kenning_cli, ["lexicon", "synsets", "seed", "--db", str(fresh_db)])
    with LexiconDB(fresh_db) as db:
        eid = db.upsert_etymon("wæter", "old-english")
        db.commit()
    result = runner.invoke(
        kenning_cli,
        [
            "lexicon",
            "synsets",
            "assign",
            "wæter",
            "water/flowing",
            "--language",
            "old-english",
            "--db",
            str(fresh_db),
        ],
    )
    assert result.exit_code == 0, result.output
    # Resolved to the right id and assigned.
    with LexiconDB(fresh_db) as db:
        synsets = get_meaning_synsets_for_etymon(db, eid)
    assert {s["canonical_label"] for s in synsets} == {"water/flowing"}


def test_cli_lexicon_synsets_assign_by_form_unknown_exits_nonzero(fresh_db: Path) -> None:
    """A canonical_form that doesn't exist in the named language fails
    fast with a friendly message naming both."""
    runner = CliRunner()
    runner.invoke(kenning_cli, ["lexicon", "synsets", "seed", "--db", str(fresh_db)])
    result = runner.invoke(
        kenning_cli,
        [
            "lexicon",
            "synsets",
            "assign",
            "ghostform",
            "water/flowing",
            "--language",
            "old-english",
            "--db",
            str(fresh_db),
        ],
    )
    assert result.exit_code != 0
    assert "no etymon with canonical_form='ghostform'" in result.output


def test_cli_lexicon_synsets_assign_non_numeric_without_language_exits_nonzero(
    fresh_db: Path,
) -> None:
    """Without --language, the positional must be numeric — passing a
    word fails with a friendly message rather than a Python traceback."""
    runner = CliRunner()
    runner.invoke(kenning_cli, ["lexicon", "synsets", "seed", "--db", str(fresh_db)])
    result = runner.invoke(
        kenning_cli,
        [
            "lexicon",
            "synsets",
            "assign",
            "wæter",
            "water/flowing",
            "--db",
            str(fresh_db),
        ],
    )
    assert result.exit_code != 0
    assert "not numeric" in result.output


def test_cli_lexicon_synsets_show_empty_membership(fresh_db: Path) -> None:
    """An etymon with no synset memberships gets a friendly message
    (not an empty stdout)."""
    runner = CliRunner()
    runner.invoke(kenning_cli, ["lexicon", "synsets", "seed", "--db", str(fresh_db)])
    with LexiconDB(fresh_db) as db:
        eid = db.upsert_etymon("wæter", "old-english")
        db.commit()
    result = runner.invoke(
        kenning_cli,
        ["lexicon", "synsets", "show", str(eid), "--db", str(fresh_db)],
    )
    assert result.exit_code == 0, result.output
    assert "no meaning_synset memberships" in result.output


def test_cli_lexicon_synsets_candidates_empty(fresh_db: Path) -> None:
    """An etymon with no candidates gets a friendly message."""
    runner = CliRunner()
    runner.invoke(kenning_cli, ["lexicon", "synsets", "seed", "--db", str(fresh_db)])
    with LexiconDB(fresh_db) as db:
        eid = db.upsert_etymon("wæter", "old-english")
        db.commit()
        assign_etymon_to_meaning_synset(db, eid, "water/flowing")
    result = runner.invoke(
        kenning_cli,
        ["lexicon", "synsets", "candidates", str(eid), "--db", str(fresh_db)],
    )
    assert result.exit_code == 0, result.output
    assert "no meaning-preserving candidates" in result.output


# --- wyrd-44a: rename etymon.synset_id → etymon.cognate_id ----------------


def test_migrate_schema_renames_legacy_synset_id_to_cognate_id(tmp_path: Path) -> None:
    """A legacy DB carrying the old `synset_id` / `synset_method` columns
    + `idx_etymon_synset` index must be migrated to the new naming —
    column data preserved, index re-created under the new name."""
    db_path = tmp_path / "legacy_synset.db"
    init_schema(db_path)
    # Simulate a pre-rename DB: drop the new names and add the legacy
    # ones so the migration helper has work to do.
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("ALTER TABLE etymon RENAME COLUMN cognate_id TO synset_id")
        conn.execute("ALTER TABLE etymon RENAME COLUMN cognate_method TO synset_method")
        conn.execute("DROP INDEX IF EXISTS idx_etymon_cognate")
        conn.execute("CREATE INDEX idx_etymon_synset ON etymon(synset_id)")
        # Insert a row carrying real data on the legacy column so the
        # migration's data-preservation guarantee can be checked.
        conn.execute(
            "INSERT INTO etymon (canonical_form, language) VALUES (?, ?)",
            ("tūn", "old-english"),
        )
        eid = conn.execute("SELECT id FROM etymon WHERE canonical_form = ?", ("tūn",)).fetchone()[0]
        conn.execute(
            "UPDATE etymon SET synset_id = ?, synset_method = ? WHERE id = ?",
            (eid, "cluster-cognates-v1", eid),
        )
        conn.commit()
    finally:
        conn.close()
    with LexiconDB(db_path) as db:
        applied = migrate_schema(db)
    assert applied["etymon.synset_id_renamed_to_cognate_id"] is True
    assert applied["etymon.synset_method_renamed_to_cognate_method"] is True
    assert applied["idx_etymon_synset_renamed_to_idx_etymon_cognate"] is True
    # Post-migration: new names exist, old names are gone, data preserved.
    with LexiconDB(db_path) as db:
        cols = {row["name"] for row in db.conn.execute("PRAGMA table_info(etymon)")}
        indexes = {
            row["name"]
            for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
        }
        row = db.conn.execute(
            "SELECT cognate_id, cognate_method FROM etymon WHERE canonical_form = ?", ("tūn",)
        ).fetchone()
    assert "cognate_id" in cols
    assert "cognate_method" in cols
    assert "synset_id" not in cols
    assert "synset_method" not in cols
    assert "idx_etymon_cognate" in indexes
    assert "idx_etymon_synset" not in indexes
    assert row["cognate_id"] == eid
    assert row["cognate_method"] == "cluster-cognates-v1"


def test_migrate_schema_rename_is_idempotent_on_already_migrated_db(fresh_db: Path) -> None:
    """Running migrate_schema on a fresh DB (already on the new naming)
    is a no-op for the rename helper — applied=False on all three rename
    keys."""
    with LexiconDB(fresh_db) as db:
        applied = migrate_schema(db)
    assert applied["etymon.synset_id_renamed_to_cognate_id"] is False
    assert applied["etymon.synset_method_renamed_to_cognate_method"] is False
    assert applied["idx_etymon_synset_renamed_to_idx_etymon_cognate"] is False


def test_init_schema_uses_cognate_id_not_synset_id(fresh_db: Path) -> None:
    """Fresh installs go straight to the new naming — no legacy
    synset_id column ever appears."""
    with LexiconDB(fresh_db) as db:
        cols = {row["name"] for row in db.conn.execute("PRAGMA table_info(etymon)")}
        indexes = {
            row["name"]
            for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
        }
    assert "cognate_id" in cols
    assert "cognate_method" in cols
    assert "synset_id" not in cols
    assert "synset_method" not in cols
    assert "idx_etymon_cognate" in indexes
    assert "idx_etymon_synset" not in indexes
