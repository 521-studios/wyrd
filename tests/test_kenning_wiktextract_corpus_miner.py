"""Tests for the empirical Wiktionary corpus miner (wyrd-4hx7).

Validates the per-culture language scope filter, the redirect-skip
gloss selection, the dry-run vs apply semantics, and the citation
write that lets the entry pass the export-meanings empirical-class
admission gate.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from click.testing import CliRunner

from wyrd.generators.kenning.cli import cli
from wyrd.generators.kenning.lexicon import LexiconDB, init_schema
from wyrd.generators.kenning.wiktextract_corpus_miner import (
    CULTURE_LANG_SCOPE,
    WIKTIONARY_EMPIRICAL_SOURCE_ID,
    _accept_candidate,
    _classify_positions,
    _select_canonical_sense,
    _surface_form_for_position,
    build_index,
    compute_unaccounted_fragments,
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


def test_select_canonical_sense_drops_redirect_only_entries() -> None:
    """wyrd-a106 (2026-05-23): when every sense is a redirect, skip the
    etymon entirely. The pre-fix fallback ('better than skipping') was
    THE root cause of ~7500 garbage entries in the bundle — 'alternative
    form of X' entries with no resolvable target. Returning None is now
    the correct behavior; the caller drops the etymon."""
    entry = {
        "word": "demesne",
        "senses": [
            {"glosses": ["alternative form of demaine"], "tags": ["alt-of"]},
        ],
    }
    assert _select_canonical_sense(entry) is None


def test_select_canonical_sense_drops_prose_derivative_entries() -> None:
    """wyrd-a106: belt-and-suspenders for redirects that Wiktionary
    didn't tag with alt-of/form-of but whose gloss text matches the
    derivative classifier ('inflection of X' / 'plural of Y' / 'soft
    mutation of Z')."""
    for gloss in (
        "inflection of dōn",
        "plural of bord",
        "soft mutation of glin",
        "singular imperative of sendan",
    ):
        entry = {"word": "x", "senses": [{"glosses": [gloss], "tags": []}]}
        assert _select_canonical_sense(entry) is None, f"{gloss!r} should be skipped"


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


def test_apply_threads_pronunciation_ipa_into_etymon(tmp_path: Path, fresh_db: Path) -> None:
    """wyrd-69s5: the corpus-miner path used to drop the wiktextract
    ``sounds`` field on the floor, so European-language etymons admitted
    via empirical mining shipped with NULL ``pronunciation_ipa`` even
    when the source slice carried IPA. After threading
    ``_extract_pronunciation`` through ``_write_one`` the IPA + dialect
    columns get filled on upsert, and the rebuild-bundle path can render
    pronunciation in the SPA's etymological-provenance panel."""
    oe_slice = tmp_path / "wiktextract_old_english.jsonl"
    _write_slice(
        oe_slice,
        [
            {
                "word": "word",
                "lang_code": "ang",
                "senses": [{"glosses": ["a word"]}],
                "sounds": [{"ipa": "/word/"}, {"ipa": "[worˠd]"}],
            }
        ],
    )
    fragments = {"english": Counter({"word": 4})}
    with LexiconDB(fresh_db) as db:
        counts = mine_corpus(
            db,
            fragments_by_culture=fragments,
            sources_dir=tmp_path,
            apply=True,
        )
    assert counts.get("pronunciation_captured") == 1

    with LexiconDB(fresh_db) as db:
        row = db.conn.execute(
            "SELECT pronunciation_ipa, pronunciation_dialect FROM etymon "
            "WHERE canonical_form='word' AND language='old-english'"
        ).fetchone()
        assert row is not None
        # Untagged-IPA preference: '/word/' (no dialect) wins over the
        # tagged fallback. _extract_pronunciation's strategy is shared
        # between corpus miner + full ingester, so the same priority
        # rule applies.
        assert row["pronunciation_ipa"] == "/word/"
        assert row["pronunciation_dialect"] is None


def test_apply_threads_original_script_and_transliteration_into_etymon(
    tmp_path: Path, fresh_db: Path
) -> None:
    """wyrd-69s5 (parity with full ingester): non-Latin entries' head
    templates carry both an original-script form and a Latin-script
    transliteration. The corpus miner must thread both through
    ``upsert_etymon`` so the bundle's runtime rendering layer can show
    original-script alongside transliteration even on entries admitted
    via empirical mining. Pre-fix the corpus miner dropped these on
    the floor; post-fix the kwargs flow through and the per-run summary
    counters mirror the full ingester's shape."""
    # OE slice with a head_templates entry (OE rarely has them in real
    # data, but the wiring under test is the corpus miner's flow into
    # upsert_etymon, not the wiktextract parser's coverage). Counters
    # are the behavioural pin — exact key extraction depends on
    # _extract_head_template_renderings' heuristics, which are tested
    # separately in the ingester's test suite.
    oe_slice = tmp_path / "wiktextract_old_english.jsonl"
    _write_slice(
        oe_slice,
        [
            {
                "word": "wyrd",
                "lang_code": "ang",
                "senses": [{"glosses": ["fate, destiny"]}],
                "head_templates": [
                    {
                        "name": "head",
                        "args": {"1": "ang", "head": "wyrd"},
                        "expansion": "wyrd",
                    }
                ],
            }
        ],
    )
    fragments = {"english": Counter({"wyrd": 3})}
    with LexiconDB(fresh_db) as db:
        counts = mine_corpus(
            db,
            fragments_by_culture=fragments,
            sources_dir=tmp_path,
            apply=True,
        )
    # Behavioural pin: the row gets written, AND the per-run counter
    # keys mirror the full ingester's shape so dashboard/dry-run output
    # is consistent across both paths.
    assert "pronunciation_captured" in counts
    assert "original_script_captured" in counts
    assert "transliteration_captured" in counts
    with LexiconDB(fresh_db) as db:
        row = db.conn.execute(
            "SELECT pronunciation_ipa, original_script, transliteration "
            "FROM etymon WHERE canonical_form='wyrd' AND language='old-english'"
        ).fetchone()
        assert row is not None
        # Columns exist on the row (wiring works); whether the helper
        # extracts content from a vanilla head_templates entry is a
        # separate concern — the ingester-side helper tests cover that.


def test_apply_skips_pronunciation_for_entries_without_sounds(
    tmp_path: Path, fresh_db: Path
) -> None:
    """When the wiktextract entry has no ``sounds`` field, the IPA
    columns stay NULL and the per-run counter doesn't increment."""
    welsh_slice = tmp_path / "wiktextract_welsh.jsonl"
    _write_slice(
        welsh_slice,
        [{"word": "betws", "lang_code": "cy", "senses": [{"glosses": ["chapel"]}]}],
    )
    fragments = {"welsh": Counter({"betws": 4})}
    with LexiconDB(fresh_db) as db:
        counts = mine_corpus(
            db,
            fragments_by_culture=fragments,
            sources_dir=tmp_path,
            apply=True,
        )
    # Counter stays 0 (or absent) when no entry had sounds.
    assert counts.get("pronunciation_captured", 0) == 0
    with LexiconDB(fresh_db) as db:
        row = db.conn.execute(
            "SELECT pronunciation_ipa FROM etymon WHERE canonical_form='betws'"
        ).fetchone()
        assert row is not None
        assert row["pronunciation_ipa"] is None


def test_dry_run_writes_no_rows(tmp_path: Path, fresh_db: Path) -> None:
    welsh_slice = tmp_path / "wiktextract_welsh.jsonl"
    _write_slice(
        welsh_slice,
        [{"word": "betws", "lang_code": "cy", "senses": [{"glosses": ["chapel"]}]}],
    )
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

        derive_positions(
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


# wyrd-7bu1: closing test-coverage gaps flagged by the test-coverage-reviewer
# agent on PR #68. The functions below have direct unit coverage rather
# than just transitive coverage via mine_corpus / derive_positions.


def _make_meanings_with_one_subject() -> list[dict]:
    """Tiny meanings-data fixture: one rando-port-style subject so the
    matcher has a non-empty word_db. compute_unaccounted_fragments
    needs SOMETHING in the bundle for ``find_meaning`` to run; absent
    matches surface as candidates."""
    return [
        {
            "meaning": ["River"],
            "modifier_tags": ["water"],
            "modifier_type": "Topographical",
            "words": [{"modern_usage": "Av-", "old_english": ["av"]}],
        }
    ]


def test_compute_unaccounted_fragments_collects_matcher_leftovers(tmp_path: Path) -> None:
    """Class 1 of the candidate generator: matcher-leftover fragments
    from imperfect place names. With a bundle that only knows ``Av-``,
    ``Avlonwich`` decomposes into ``<Av-> + ['lonwich']`` — ``lonwich``
    is the unaccounted leftover and should land in the candidate set."""
    place_names_path = tmp_path / "english_place_names.json"
    place_names_path.write_text(json.dumps({"England": {"all": ["Avlonwich"]}}))
    meanings = _make_meanings_with_one_subject()
    candidates = compute_unaccounted_fragments("english", place_names_path, meanings, min_length=3)
    assert "lonwich" in candidates
    # Per-name dedupe: one occurrence in the corpus → count=1.
    assert candidates["lonwich"] == 1


def test_compute_unaccounted_fragments_emits_prefix_and_suffix_substrings(
    tmp_path: Path,
) -> None:
    """Class 3 of the candidate generator: per-word prefixes and
    suffixes (length min..max) are emitted. For a single-word imperfect
    place name like ``Cluainacarrick`` the candidate set should contain
    the full word, every prefix length 3..12, and every suffix length
    3..12 — the high-impact morphemes (``cluain`` as a prefix) live in
    this class."""
    place_names_path = tmp_path / "irish_place_names.json"
    place_names_path.write_text(json.dumps({"Republic of Ireland": {"all": ["Cluainacarrick"]}}))
    meanings = _make_meanings_with_one_subject()
    candidates = compute_unaccounted_fragments(
        "irish", place_names_path, meanings, min_length=3, max_length=12
    )
    # Prefixes (the 'cluain' is the wyrd-1cjg target):
    assert "clu" in candidates
    assert "cluain" in candidates
    # Suffixes:
    assert "ick" in candidates
    assert "carrick" in candidates
    # Whitespace-word: the whole place name (length ≤ max_length).
    # Cluainacarrick is 14 chars, so it doesn't make the ≤12 cap —
    # but a shorter case would. Confirm the cap holds:
    assert all(len(c) <= 12 for c in candidates)


def test_compute_unaccounted_fragments_dedupe_per_name(tmp_path: Path) -> None:
    """Per-name dedup means a candidate that appears multiple times in
    the SAME place name only contributes once to its counter. A
    candidate that appears in N different imperfect names has count=N."""
    place_names_path = tmp_path / "english_place_names.json"
    place_names_path.write_text(
        json.dumps(
            {
                "England": {
                    "all": [
                        "Smithford",  # 'smith' and 'ford' substrings
                        "Smithfield",  # 'smith' again — should still count once across names
                    ]
                }
            }
        )
    )
    meanings = _make_meanings_with_one_subject()
    candidates = compute_unaccounted_fragments("english", place_names_path, meanings, min_length=3)
    # 'smith' appears in both names → count = 2.
    assert candidates["smith"] == 2


def test_accept_candidate_filters_too_short_and_too_long() -> None:
    """The length filter on the helper. Below min_length: dropped.
    Above max_length: dropped. In bounds: counted."""
    candidates: Counter = Counter()
    seen: set[str] = set()
    _accept_candidate("ab", candidates, seen, min_length=3, max_length=12)  # too short
    _accept_candidate(
        "abcdefghijklmn", candidates, seen, min_length=3, max_length=12
    )  # too long (14)
    _accept_candidate("foo", candidates, seen, min_length=3, max_length=12)  # in bounds
    assert "foo" in candidates
    assert "ab" not in candidates
    assert "abcdefghijklmn" not in candidates
    # Counter incremented to 1 for the in-bounds candidate.
    assert candidates["foo"] == 1


def test_accept_candidate_dedupe_via_seen_set() -> None:
    """The per-name seen set ensures a candidate isn't double-counted
    when the matcher emits the same fragment via different paths
    (matcher-leftover AND prefix substring AND suffix substring)."""
    candidates: Counter = Counter()
    seen: set[str] = set()
    _accept_candidate("foo", candidates, seen, min_length=3, max_length=12)
    _accept_candidate("foo", candidates, seen, min_length=3, max_length=12)
    _accept_candidate("foo", candidates, seen, min_length=3, max_length=12)
    assert candidates["foo"] == 1
    assert "foo" in seen


def test_accept_candidate_skips_empty_string() -> None:
    """Defensive: empty fragment strings are dropped without raising."""
    candidates: Counter = Counter()
    seen: set[str] = set()
    _accept_candidate("", candidates, seen, min_length=3, max_length=12)
    assert candidates == {}


def test_mine_wiktextract_corpus_cli_apply_and_dry_run(fresh_db: Path) -> None:
    """End-to-end: ``wyrd kenning lexicon mine-wiktextract-corpus``
    via CliRunner. Runs the dry-run path then the apply path against
    a tmp lexicon DB and a synthetic wiktextract slice. ``fresh_db``
    already creates ``tmp_path / lexicon.db``; we derive the sources
    dir from its parent so a single fixture covers both."""
    from click.testing import CliRunner

    from wyrd.generators.kenning.cli import cli

    db_path = fresh_db
    sources_dir = fresh_db.parent / "sources"
    sources_dir.mkdir()
    welsh_slice = sources_dir / "wiktextract_welsh.jsonl"
    _write_slice(
        welsh_slice,
        [{"word": "betws", "lang_code": "cy", "senses": [{"glosses": ["chapel"]}]}],
    )

    runner = CliRunner()

    # The CLI builds candidates from the bundled per-culture place_names
    # JSONs via importlib.resources, so we have to feed it a real culture
    # (welsh) whose corpus contains 'betws' as a substring of some name.
    # Bryneglwys and Betws Coch are both in the bundled corpus.

    # Dry-run: should hit but write nothing.
    result = runner.invoke(
        cli,
        [
            "lexicon",
            "mine-wiktextract-corpus",
            "--db",
            str(db_path),
            "--sources-dir",
            str(sources_dir),
            "--culture",
            "welsh",
        ],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    # Stderr summary should mention dry-run.
    output = result.stderr or result.output
    assert "dry-run" in output
    with LexiconDB(db_path) as db:
        cit_count = db.conn.execute(
            "SELECT COUNT(*) FROM etymon_citation WHERE source_id=?",
            (WIKTIONARY_EMPIRICAL_SOURCE_ID,),
        ).fetchone()[0]
    assert cit_count == 0

    # Apply: should write rows.
    result = runner.invoke(
        cli,
        [
            "lexicon",
            "mine-wiktextract-corpus",
            "--db",
            str(db_path),
            "--sources-dir",
            str(sources_dir),
            "--culture",
            "welsh",
            "--apply",
        ],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")
    with LexiconDB(db_path) as db:
        cit_count = db.conn.execute(
            "SELECT COUNT(*) FROM etymon_citation WHERE source_id=?",
            (WIKTIONARY_EMPIRICAL_SOURCE_ID,),
        ).fetchone()[0]
    # 'betws' should have landed as an empirical etymon in welsh.
    assert cit_count >= 1


def test_mine_wiktextract_corpus_writes_mining_run_audit_row(fresh_db: Path) -> None:
    """wyrd-qepf / D24: a successful --apply run must persist a
    mining_run row so the operation is queryable from the DB. The rich
    structural counters (etymons_upserted, by_culture, by_language,
    position_reflexes_written) land in by_failure (the audit-shaped
    JSON column); accept/decline/reject-style fields use the closest
    analogues since empirical mining doesn't have those semantics."""
    db_path = fresh_db
    sources_dir = fresh_db.parent / "sources"
    sources_dir.mkdir()
    welsh_slice = sources_dir / "wiktextract_welsh.jsonl"
    _write_slice(
        welsh_slice,
        [{"word": "betws", "lang_code": "cy", "senses": [{"glosses": ["chapel"]}]}],
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "lexicon",
            "mine-wiktextract-corpus",
            "--db",
            str(db_path),
            "--sources-dir",
            str(sources_dir),
            "--culture",
            "welsh",
            "--apply",
        ],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")

    with LexiconDB(db_path) as db:
        rows = db.conn.execute(
            "SELECT provider, model, mode, parsed_count, accepted, declined, "
            "rejected, by_failure FROM mining_run WHERE source_id = ?",
            (WIKTIONARY_EMPIRICAL_SOURCE_ID,),
        ).fetchall()

    assert len(rows) == 1, "exactly one audit row per --apply run"
    row = rows[0]
    assert row["provider"] == "wiktionary-empirical"
    assert row["model"] == "corpus-substring-v1"
    assert row["mode"] == "mine"
    assert row["accepted"] >= 1  # at least 'betws' landed
    assert row["declined"] == 0
    assert row["rejected"] == 0

    audit = json.loads(row["by_failure"])
    metrics = audit["empirical_metrics"]
    assert metrics["etymons_upserted"] >= 1
    assert metrics["citations_added"] >= 1
    # Per-culture and per-language breakdowns must land in the audit so
    # operators can filter mining_run for "what did the welsh slice
    # contribute" without re-running the mining pass.
    assert "welsh" in metrics["by_culture"]
    assert metrics["by_culture"]["welsh"]["hits"] >= 1
    # by_language is keyed by the canonical language name (build_index
    # normalizes 'cy' → 'welsh' via _canonical_language), not the raw
    # wiktextract lang_code.
    assert metrics["by_language"].get("welsh", 0) >= 1
    assert metrics["target_cultures"] == ["welsh"]


def test_mine_wiktextract_corpus_dry_run_skips_mining_run_audit_row(
    fresh_db: Path,
) -> None:
    """The audit row is conditional on --apply (mirrors the rest of the
    write side). A dry-run must not pollute mining_run with rows that
    don't correspond to actual writes."""
    db_path = fresh_db
    sources_dir = fresh_db.parent / "sources"
    sources_dir.mkdir()
    welsh_slice = sources_dir / "wiktextract_welsh.jsonl"
    _write_slice(
        welsh_slice,
        [{"word": "betws", "lang_code": "cy", "senses": [{"glosses": ["chapel"]}]}],
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "lexicon",
            "mine-wiktextract-corpus",
            "--db",
            str(db_path),
            "--sources-dir",
            str(sources_dir),
            "--culture",
            "welsh",
        ],
    )
    assert result.exit_code == 0, result.output + (result.stderr or "")

    with LexiconDB(db_path) as db:
        rows = db.conn.execute(
            "SELECT COUNT(*) FROM mining_run WHERE source_id = ?",
            (WIKTIONARY_EMPIRICAL_SOURCE_ID,),
        ).fetchone()[0]
    assert rows == 0
