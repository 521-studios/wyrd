"""Tests for the empirical Wiktionary corpus miner (wyrd-4hx7).

Validates the per-culture language scope filter, the redirect-skip
gloss selection, the dry-run vs apply semantics, and the citation
write that lets the entry pass the export-meanings empirical-class
admission gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wyrd.generators.kenning.lexicon import LexiconDB, init_schema
from wyrd.generators.kenning.wiktextract_corpus_miner import (
    CULTURE_LANG_SCOPE,
    WIKTIONARY_EMPIRICAL_SOURCE_ID,
    _classify_positions,
    _select_canonical_sense,
    _surface_form_for_position,
    build_index,
    derive_positions,
    mine_corpus,
)


@pytest.fixture
def fresh_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "lexicon.db"
    init_schema(db_path)
    return db_path


def _write_slice(path: Path, entries: list[dict]) -> None:
    """Write a list of dicts as a wiktextract jsonl file."""
    with path.open("w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")


def test_select_canonical_sense_skips_alt_of_redirect() -> None:
    entry = {
        "word": "demesne",
        "senses": [
            {"glosses": ["alternative form of demaine"], "tags": ["alt-of"]},
            {"glosses": ["lord's domain"]},
        ],
    }
    result = _select_canonical_sense(entry)
    assert result is not None
    gloss, tags = result
    assert gloss == "lord's domain"


def test_select_canonical_sense_falls_back_to_redirect_when_only_option() -> None:
    """If every sense is a redirect, take the first one rather than skipping
    the entry entirely. A placeholder gloss beats no entry."""
    entry = {
        "word": "demesne",
        "senses": [
            {"glosses": ["alternative form of demaine"], "tags": ["alt-of"]},
        ],
    }
    result = _select_canonical_sense(entry)
    assert result is not None
    gloss, _ = result
    assert "demaine" in gloss


def test_select_canonical_sense_returns_none_for_glossless_entry() -> None:
    entry = {"word": "x", "senses": [{"glosses": []}, {}]}
    assert _select_canonical_sense(entry) is None


def test_select_canonical_sense_maps_categories_to_tags() -> None:
    entry = {
        "word": "eglwys",
        "senses": [
            {
                "glosses": ["church"],
                "categories": [{"name": "Christianity"}, {"name": "Places of worship"}],
            }
        ],
    }
    result = _select_canonical_sense(entry)
    assert result is not None
    gloss, tags = result
    assert gloss == "church"
    assert "religious" in tags


def test_build_index_filters_by_target_form_and_language(tmp_path: Path) -> None:
    """Headwords outside the target set or in unrelated languages are dropped."""
    welsh_slice = tmp_path / "wiktextract_welsh.jsonl"
    _write_slice(
        welsh_slice,
        [
            {"word": "eglwys", "lang_code": "cy", "senses": [{"glosses": ["church"]}]},
            {"word": "betws", "lang_code": "cy", "senses": [{"glosses": ["chapel"]}]},
            {"word": "noise", "lang_code": "cy", "senses": [{"glosses": ["x"]}]},
            # Cross-language stub the slice sometimes contains:
            {"word": "eglwys", "lang_code": "la", "senses": [{"glosses": ["church"]}]},
        ],
    )
    index = build_index(tmp_path, {"eglwys", "betws"}, {"welsh"})
    # Only the welsh-language headwords whose form is in the target set
    # appear; the lang-mismatch row is dropped, the unrelated 'noise' row
    # is dropped.
    assert ("eglwys", "welsh") in index
    assert ("betws", "welsh") in index
    assert len(index) == 2


def test_culture_scope_keeps_welsh_morphemes_out_of_english_bundle(
    tmp_path: Path, fresh_db: Path
) -> None:
    """A welsh headword should not be admitted for the english culture even
    if the form happens to match an english fragment."""
    welsh_slice = tmp_path / "wiktextract_welsh.jsonl"
    _write_slice(
        welsh_slice,
        [{"word": "coch", "lang_code": "cy", "senses": [{"glosses": ["red"]}]}],
    )
    from collections import Counter

    fragments_by_culture = {"english": Counter({"coch": 1})}
    with LexiconDB(fresh_db) as db:
        counts = mine_corpus(
            db,
            fragments_by_culture=fragments_by_culture,
            sources_dir=tmp_path,
            apply=True,
        )
    # English's allowed scope doesn't include welsh, so the welsh 'coch'
    # entry never gets attached to an etymon for the english fragment.
    assert counts["wiktionary_hits"] == 0
    assert counts["etymons_upserted"] == 0
    # Sanity: welsh IS in the culture scope for the welsh culture.
    assert "welsh" in CULTURE_LANG_SCOPE["welsh"]
    assert "welsh" not in CULTURE_LANG_SCOPE["english"]


def test_apply_writes_etymon_gloss_and_empirical_citation(tmp_path: Path, fresh_db: Path) -> None:
    welsh_slice = tmp_path / "wiktextract_welsh.jsonl"
    _write_slice(
        welsh_slice,
        [{"word": "betws", "lang_code": "cy", "senses": [{"glosses": ["chapel"]}]}],
    )
    from collections import Counter

    fragments = {"welsh": Counter({"betws": 4})}
    with LexiconDB(fresh_db) as db:
        counts = mine_corpus(
            db,
            fragments_by_culture=fragments,
            sources_dir=tmp_path,
            apply=True,
        )
    assert counts["etymons_upserted"] == 1
    assert counts["citations_added"] == 1

    # Verify the row is queryable: etymon + gloss + citation pointing at
    # the synthetic empirical source.
    with LexiconDB(fresh_db) as db:
        row = db.conn.execute(
            "SELECT id FROM etymon WHERE canonical_form='betws' AND language='welsh'"
        ).fetchone()
        assert row is not None
        etymon_id = row["id"]
        glosses = [
            r["gloss"]
            for r in db.conn.execute(
                "SELECT gloss FROM etymon_gloss WHERE etymon_id=?", (etymon_id,)
            )
        ]
        assert glosses == ["chapel"]
        cit_sources = [
            r["source_id"]
            for r in db.conn.execute(
                "SELECT source_id FROM etymon_citation WHERE etymon_id=?", (etymon_id,)
            )
        ]
        assert cit_sources == [WIKTIONARY_EMPIRICAL_SOURCE_ID]


def test_dry_run_writes_no_rows(tmp_path: Path, fresh_db: Path) -> None:
    welsh_slice = tmp_path / "wiktextract_welsh.jsonl"
    _write_slice(
        welsh_slice,
        [{"word": "betws", "lang_code": "cy", "senses": [{"glosses": ["chapel"]}]}],
    )
    from collections import Counter

    fragments = {"welsh": Counter({"betws": 4})}
    with LexiconDB(fresh_db) as db:
        counts = mine_corpus(
            db,
            fragments_by_culture=fragments,
            sources_dir=tmp_path,
            apply=False,
        )
    # Hits are reported (so the dry-run summary is meaningful) but no
    # rows are written.
    assert counts["wiktionary_hits"] == 1
    assert counts["etymons_upserted"] == 0
    with LexiconDB(fresh_db) as db:
        row = db.conn.execute("SELECT 1 FROM etymon WHERE canonical_form='betws'").fetchone()
        assert row is None


def test_idempotent_reapply_does_not_duplicate_glosses_or_citations(
    tmp_path: Path, fresh_db: Path
) -> None:
    welsh_slice = tmp_path / "wiktextract_welsh.jsonl"
    _write_slice(
        welsh_slice,
        [{"word": "betws", "lang_code": "cy", "senses": [{"glosses": ["chapel"]}]}],
    )
    from collections import Counter

    fragments = {"welsh": Counter({"betws": 4})}
    for _ in range(3):
        with LexiconDB(fresh_db) as db:
            mine_corpus(
                db,
                fragments_by_culture=fragments,
                sources_dir=tmp_path,
                apply=True,
            )
    with LexiconDB(fresh_db) as db:
        gloss_count = db.conn.execute("SELECT COUNT(*) AS n FROM etymon_gloss").fetchone()["n"]
        cit_count = db.conn.execute("SELECT COUNT(*) AS n FROM etymon_citation").fetchone()["n"]
        etymon_count = db.conn.execute("SELECT COUNT(*) AS n FROM etymon").fetchone()["n"]
    # All three were INSERT-OR-IGNORE'd into idempotency.
    assert gloss_count == 1
    assert cit_count == 1
    assert etymon_count == 1


def test_classify_positions_detects_pre_post_inner() -> None:
    """A morpheme appearing at the start of a name is pre, end is post,
    interior is inner. Full-name match counts as pre+post."""
    flat_names = [
        "greatabington",  # 'great' at start → pre
        "abbeydemesne",  # 'demesne' at end → post
        "stowmarket",  # 'mark' is interior → inner
        "great",  # full match → pre+post
    ]
    counts = _classify_positions("great", flat_names)
    assert counts["pre"] == 2  # greatabington + great-as-full
    assert counts["post"] == 1  # great-as-full
    assert counts["inner"] == 0

    counts = _classify_positions("demesne", flat_names)
    assert counts["pre"] == 0
    assert counts["post"] == 1
    assert counts["inner"] == 0

    counts = _classify_positions("mark", flat_names)
    assert counts["pre"] == 0
    assert counts["post"] == 0
    assert counts["inner"] == 1


def test_classify_positions_skips_too_short() -> None:
    """Single-char canonical forms are too noisy to classify."""
    counts = _classify_positions("a", ["abington", "great"])
    assert counts == {"pre": 0, "post": 0, "inner": 0}


def test_surface_form_for_position_emits_dash_conventions() -> None:
    """Pre = title-case + trailing dash; inner = lowercase wrapped;
    post = lowercase + leading dash."""
    assert _surface_form_for_position("great", "pre") == "Great-"
    assert _surface_form_for_position("great", "inner") == "-great-"
    assert _surface_form_for_position("great", "post") == "-great"


def test_derive_positions_writes_pre_and_post_reflexes_for_north(
    tmp_path: Path, fresh_db: Path
) -> None:
    """Concrete worked example: 'north' appears at both ends in the
    English corpus → derive_positions writes both Pre + Post reflex
    rows + links them to the etymon. The bundle export then emits two
    bundle words (one per dashed surface_form)."""
    place_names_path = tmp_path / "english_place_names.json"
    place_names_path.write_text(
        json.dumps(
            {
                "England": {
                    "ceremonial_counties": [
                        "North Braitham",  # pre
                        "Braitham North",  # post
                        "North Braitham North",  # one of each
                        "Hithernorthville",  # 'north' interior
                    ]
                }
            }
        )
    )
    with LexiconDB(fresh_db) as db:
        # Seed an empirical 'north' etymon.
        db.upsert_source(id=WIKTIONARY_EMPIRICAL_SOURCE_ID, title="test")
        eid = db.upsert_etymon("north", "modern-english")
        db.add_citation(eid, WIKTIONARY_EMPIRICAL_SOURCE_ID)
        db.commit()

        counts = derive_positions(
            db,
            {"english": place_names_path},
            apply=True,
        )

    assert counts["etymons_examined"] == 1
    # All three positions observed across the test corpus.
    assert counts["by_position"]["pre"] == 1
    assert counts["by_position"]["post"] == 1
    assert counts["by_position"]["inner"] == 1

    # Verify the reflex rows landed.
    with LexiconDB(fresh_db) as db:
        north_id = db.conn.execute("SELECT id FROM etymon WHERE canonical_form='north'").fetchone()[
            "id"
        ]
        north_reflexes = [
            (r["surface_form"], r["position"])
            for r in db.conn.execute(
                """
                SELECT r.surface_form, r.position
                FROM reflex_etymon re
                JOIN reflex r ON r.id = re.reflex_id
                WHERE re.etymon_id = ?
                ORDER BY r.position
                """,
                (north_id,),
            )
        ]
        assert ("North-", "pre") in north_reflexes
        assert ("-north", "post") in north_reflexes
        assert ("-north-", "inner") in north_reflexes


def test_derive_positions_great_is_pre_only(tmp_path: Path, fresh_db: Path) -> None:
    """User's other example: 'Great Braitham' parses (great is pre);
    'Braitham Great' shouldn't (great is not post in any English place
    name corpus). derive_positions must reflect this asymmetry — no
    post reflex written if the corpus has no post evidence."""
    place_names_path = tmp_path / "english_place_names.json"
    place_names_path.write_text(
        json.dumps(
            {
                "England": [
                    "Great Abington",
                    "Great Addington",
                    "Greater Braitham",
                    "Great Alne",
                ]
            }
        )
    )
    with LexiconDB(fresh_db) as db:
        db.upsert_source(id=WIKTIONARY_EMPIRICAL_SOURCE_ID, title="test")
        gid = db.upsert_etymon("great", "modern-english")
        db.add_citation(gid, WIKTIONARY_EMPIRICAL_SOURCE_ID)
        db.commit()

        derive_positions(db, {"english": place_names_path}, apply=True)

    with LexiconDB(fresh_db) as db:
        gid = db.conn.execute("SELECT id FROM etymon WHERE canonical_form='great'").fetchone()["id"]
        positions = sorted(
            r["position"]
            for r in db.conn.execute(
                """
                SELECT r.position
                FROM reflex_etymon re
                JOIN reflex r ON r.id = re.reflex_id
                WHERE re.etymon_id = ?
                """,
                (gid,),
            )
        )
    # 'great' is pre-only in this corpus — no post reflex written.
    assert positions == ["pre"]


def test_derive_positions_culture_scope_filters_languages(tmp_path: Path, fresh_db: Path) -> None:
    """A welsh empirical etymon is examined only against the welsh corpus,
    not the english one — keeps Welsh morphemes from sucking up English
    place-name occurrences they don't belong to."""
    welsh_corpus = tmp_path / "welsh_place_names.json"
    welsh_corpus.write_text(json.dumps({"Wales": ["Bryneglwys"]}))
    english_corpus = tmp_path / "english_place_names.json"
    english_corpus.write_text(json.dumps({"England": ["Glwysworth"]}))

    with LexiconDB(fresh_db) as db:
        db.upsert_source(id=WIKTIONARY_EMPIRICAL_SOURCE_ID, title="test")
        eglwys_id = db.upsert_etymon("eglwys", "welsh")
        db.add_citation(eglwys_id, WIKTIONARY_EMPIRICAL_SOURCE_ID)
        db.commit()

        counts = derive_positions(
            db,
            {"welsh": welsh_corpus, "english": english_corpus},
            apply=True,
        )

    # 'eglwys' is welsh; only the welsh corpus contributes. 'eglwys'
    # appears at the END of 'bryneglwys' (post). It does NOT pick up
    # 'glwysworth' from the english corpus (welsh is not in english scope).
    with LexiconDB(fresh_db) as db:
        reflexes = list(
            db.conn.execute(
                """
                SELECT r.surface_form, r.position
                FROM reflex_etymon re
                JOIN reflex r ON r.id = re.reflex_id
                WHERE re.etymon_id = ?
                """,
                (eglwys_id,),
            )
        )
    positions = {r["position"] for r in reflexes}
    assert positions == {"post"}, f"expected {{post}} from welsh-only corpus, got {positions}"


def test_missing_slice_file_is_silently_skipped(tmp_path: Path, fresh_db: Path) -> None:
    """If the on-disk slice for a language doesn't exist, the miner skips
    that language without raising. Real-world: we don't have a Latin slice
    on disk; latin lookups should produce zero hits, not an error."""
    # No slice files in tmp_path at all.
    from collections import Counter

    fragments = {"english": Counter({"thing": 1})}
    with LexiconDB(fresh_db) as db:
        counts = mine_corpus(
            db,
            fragments_by_culture=fragments,
            sources_dir=tmp_path,
            apply=True,
        )
    assert counts["wiktionary_hits"] == 0
    assert counts["etymons_upserted"] == 0
