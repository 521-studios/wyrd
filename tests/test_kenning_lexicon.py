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
    _position_from_usage,
    clear_enrichment,
    cluster_ocr_variants,
    derive_lemma_candidate,
    export_meanings,
    fuzzy_search_attestations,
    ingest_parsed_entries,
    init_schema,
    link_lemmas,
    migrate_schema,
    normalize_ocr_form,
    record_mining_run,
    seed_from_meanings,
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
    rows in a single invocation."""
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
    monkeypatch.setattr(cli_mod, "_select_parser_and_run", lambda text, parser: fake_parsed)

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
    monkeypatch.setattr(cli_mod, "_select_parser_and_run", lambda text, parser: fake_parsed)

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
    monkeypatch.setattr(cli_mod, "_select_parser_and_run", lambda text, parser: fake_parsed)

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
    # Also tolerate the per-language variant pool fields emitted by the
    # D18 export step, which legitimately don't map to a LANGUAGE_FIELDS code.
    missing = {f for f in seen_fields - handled if not f.endswith("_variants")}
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
    assert subj["words"] == [{"modern_usage": "ham", "old_english": ["ham"]}]


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
        _add_text_match(db, brycg_id, "src-a", "brige", 1)
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
    assert subjects[0]["words"] == [{"modern_usage": "tune", "old_english": ["tune"]}]


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
