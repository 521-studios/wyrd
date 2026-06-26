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
    _DASH_TO_SPACE,
    CULTURE_LANG_SCOPE,
    WIKTIONARY_EMPIRICAL_SOURCE_ID,
    _accept_candidate,
    _classify_positions,
    _iter_index_entries,
    _select_canonical_sense,
    _surface_form_for_position,
    _walk_names,
    build_index,
    compute_unaccounted_fragments,
    derive_positions,
    mine_corpus,
    missing_slice_languages,
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


@pytest.mark.parametrize("pos", ["prep", "postp", "conj", "pron", "particle", "det", "article"])
def test_select_canonical_sense_skips_function_word_pos(pos: str) -> None:
    """wyrd-33cv: function-word entries (preposition / conjunction /
    pronoun / particle / determiner / article / postposition) are
    skipped at ingest — they never carry toponymic meaning, so a
    surface match against an unaccounted place-name fragment is noise.
    The POS gate fires BEFORE sense inspection, so even a perfectly
    good content gloss on a function-word entry is dropped."""
    entry = {
        "word": "by",
        "pos": pos,
        "senses": [{"glosses": ["a clean content-looking gloss"], "tags": []}],
    }
    assert _select_canonical_sense(entry) is None, f"pos={pos!r} should be skipped"


@pytest.mark.parametrize("pos", ["noun", "verb", "adj", "adv", "name", "num", "suffix"])
def test_select_canonical_sense_keeps_content_word_pos(pos: str) -> None:
    """wyrd-33cv: content POS pass through unchanged — the POS gate
    only denies the function-word set, it doesn't allowlist."""
    entry = {"word": "tun", "pos": pos, "senses": [{"glosses": ["enclosure, farmstead"]}]}
    result = _select_canonical_sense(entry)
    assert result is not None, f"pos={pos!r} should be kept"
    gloss, _tags = result
    assert gloss == "enclosure, farmstead"


@pytest.mark.parametrize(
    "entry",
    [
        {"word": "tun", "senses": [{"glosses": ["enclosure, farmstead"]}]},  # absent
        {"word": "tun", "pos": "", "senses": [{"glosses": ["enclosure, farmstead"]}]},  # empty
        {
            "word": "tun",
            "pos": "  ",
            "senses": [{"glosses": ["enclosure, farmstead"]}],
        },  # whitespace
    ],
    ids=["absent", "empty", "whitespace"],
)
def test_select_canonical_sense_keeps_entry_with_missing_pos(entry: dict) -> None:
    """wyrd-33cv: a missing/empty/whitespace pos is NOT treated as a
    function word — the gate is ``(entry.get("pos") or "").strip() in
    _FUNCTION_WORD_POS``, so absent → '', empty → '', whitespace →
    '' after strip, none of which are in the denylist. Denylist, not
    allowlist: entries lacking a usable pos still get their gloss
    selected."""
    result = _select_canonical_sense(entry)
    assert result is not None
    assert result[0] == "enclosure, farmstead"


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


@pytest.mark.parametrize("redirect_first", [True, False])
def test_build_index_prefers_entry_with_usable_sense(tmp_path: Path, redirect_first: bool) -> None:
    """When a headword has multiple etymology_number entries — a redirect-only
    one (no usable sense) and a real-sense one — build_index must keep the
    real-sense entry regardless of slice order. A blind first-wins kept the
    redirect when it came first, so _select_canonical_sense returned None and
    the etymon was silently dropped despite a usable sense existing for the same
    (form, language). Real Wiktionary dumps routinely list the redirect first."""
    redirect = {
        "word": "bryn",
        "lang_code": "cy",
        "etymology_number": 1,
        "senses": [{"glosses": ["alternative form of brynn"], "tags": ["alt-of"]}],
    }
    real = {
        "word": "bryn",
        "lang_code": "cy",
        "etymology_number": 2,
        "senses": [{"glosses": ["hill, mound"]}],
    }
    entries = [redirect, real] if redirect_first else [real, redirect]
    _write_slice(tmp_path / "wiktextract_welsh.jsonl", entries)
    index = build_index(tmp_path, {"bryn"}, {"welsh"})
    kept = index[("bryn", "welsh")]
    selected = _select_canonical_sense(kept)
    assert selected is not None, "the real-sense entry was dropped"
    assert selected[0] == "hill, mound"


def test_iter_index_entries_filters_and_keys(tmp_path: Path) -> None:
    """``_iter_index_entries`` (the per-slice matcher extracted from build_index)
    yields ``((form_lower, canonical_language), entry)`` for matching headwords
    only: empty words, off-target forms, and lang-mismatched stubs are skipped,
    and the key form is lower-cased. It does NOT dedup — the caller does."""
    slice_path = tmp_path / "wiktextract_welsh.jsonl"
    _write_slice(
        slice_path,
        [
            {"word": "  EGLWYS  ", "lang_code": "cy", "senses": [{"glosses": ["church"]}]},
            {"word": "", "lang_code": "cy", "senses": []},  # empty word -> skipped
            {"word": "noise", "lang_code": "cy", "senses": []},  # off-target -> skipped
            {"word": "eglwys", "lang_code": "la", "senses": []},  # lang mismatch -> skipped
            {"word": "eglwys", "lang_code": "cy", "senses": [{"glosses": ["dup"]}]},  # 2nd match
        ],
    )
    pairs = list(_iter_index_entries(slice_path, "welsh", {"eglwys"}))
    # Both welsh 'eglwys' rows yielded (no dedup here), key lower-cased; the
    # empty / off-target / lang-mismatch rows are filtered out.
    assert [k for k, _ in pairs] == [("eglwys", "welsh"), ("eglwys", "welsh")]
    assert [e["senses"][0]["glosses"][0] for _, e in pairs] == ["church", "dup"]


def test_missing_slice_languages_reports_only_published_absent_slices(
    tmp_path: Path,
) -> None:
    """wyrd-34kt: a published slice absent from sources_dir is reported;
    a language with no published slice (None mapping) is NOT — its
    absence is expected, not an operator error."""
    # welsh IS present; proto-celtic + scottish-gaelic are NOT.
    _write_slice(
        tmp_path / "wiktextract_welsh.jsonl",
        [{"word": "eglwys", "lang_code": "cy", "senses": [{"glosses": ["church"]}]}],
    )
    missing = missing_slice_languages(
        tmp_path,
        # 'latin' maps to None (no published slice) → must not be reported.
        {"welsh", "proto-celtic", "scottish-gaelic", "latin"},
    )
    assert missing == [
        ("proto-celtic", "wiktextract_proto_celtic.jsonl"),
        ("scottish-gaelic", "wiktextract_scottish_gaelic.jsonl"),
    ]


def test_missing_slice_languages_empty_when_all_present(tmp_path: Path) -> None:
    _write_slice(
        tmp_path / "wiktextract_welsh.jsonl",
        [{"word": "eglwys", "lang_code": "cy", "senses": [{"glosses": ["church"]}]}],
    )
    assert missing_slice_languages(tmp_path, {"welsh", "latin"}) == []


def test_mine_corpus_surfaces_missing_slices_in_counts(tmp_path: Path, fresh_db: Path) -> None:
    """The wrong-sources-dir failure mode (wyrd-34kt) lands as an
    observable counter, not a silent 0-hits run: welsh's slice is on
    disk but proto-celtic's is not, so it's reported."""
    _write_slice(
        tmp_path / "wiktextract_welsh.jsonl",
        [{"word": "eglwys", "lang_code": "cy", "senses": [{"glosses": ["church"]}]}],
    )
    fragments_by_culture = {"welsh": Counter({"eglwys": 1})}
    with LexiconDB(fresh_db) as db:
        counts = mine_corpus(
            db,
            fragments_by_culture=fragments_by_culture,
            sources_dir=tmp_path,
            apply=False,
        )
    # welsh resolved (the eglwys headword is a hit); proto-celtic's slice
    # is absent and surfaces in missing_slices.
    assert counts["wiktionary_hits"] == 1
    assert ("proto-celtic", "wiktextract_proto_celtic.jsonl") in counts["missing_slices"]
    assert ("welsh", "wiktextract_welsh.jsonl") not in counts["missing_slices"]


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


def test_mine_corpus_de_dashes_headword_and_drops_junk(tmp_path: Path, fresh_db: Path) -> None:
    """wyrd-aicu.8 (D45): a dashed empirical headword is stored BARE, and a
    headword that de-dashes to empty junk (a bare '-') is dropped at the ingest
    boundary — never handed to the upsert choke — and counted (words_skipped_empty)
    so the drop isn't silent."""
    welsh_slice = tmp_path / "wiktextract_welsh.jsonl"
    _write_slice(
        welsh_slice,
        [
            {"word": "-ach", "lang_code": "cy", "senses": [{"glosses": ["suffix"]}]},
            {"word": "-", "lang_code": "cy", "senses": [{"glosses": ["junk"]}]},
        ],
    )
    fragments = {"welsh": Counter({"-ach": 3, "-": 2})}
    with LexiconDB(fresh_db) as db:
        counts = mine_corpus(db, fragments_by_culture=fragments, sources_dir=tmp_path, apply=True)
        forms = {r[0] for r in db.conn.execute("SELECT canonical_form FROM etymon")}
    # The dashed headword is stored de-dashed; the junk '-' is dropped + counted.
    assert "ach" in forms
    assert "-ach" not in forms and "-" not in forms
    assert counts["words_skipped_empty"] == 1
    assert counts["etymons_upserted"] == 1


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
    """wyrd-7nab word-aware: a morpheme at the start of a WORD is pre, end is
    post, interior is inner; a morpheme that IS the whole word is bare."""
    words = [
        "greatabington",  # 'great' at start → pre
        "abbeydemesne",  # 'demesne' at end → post
        "stowmarket",  # 'mark' is interior → inner
        "great",  # whole word → bare (was pre+post pre-fix)
    ]
    counts = _classify_positions("great", words)
    assert counts["pre"] == 1  # greatabington
    assert counts["bare"] == 1  # 'great' standalone word
    assert counts["post"] == 0 and counts["inner"] == 0

    counts = _classify_positions("demesne", words)
    assert counts == {"bare": 0, "pre": 0, "post": 1, "inner": 0}

    counts = _classify_positions("mark", words)
    assert counts == {"bare": 0, "pre": 0, "post": 0, "inner": 1}


def test_classify_positions_skips_too_short() -> None:
    """Single-char canonical forms are too noisy to classify."""
    counts = _classify_positions("a", ["abington", "great"])
    assert counts == {"bare": 0, "pre": 0, "post": 0, "inner": 0}


def test_walk_names_splits_on_unicode_dashes_too():
    """Place-name tokenization splits on ANY dash, not just ASCII hyphen: a
    re-sourced / typographic corpus emitting an en-dash (``Stratford–upon–Avon``)
    must tokenize the same as the ASCII-hyphen form, or a standalone morpheme
    keys as a prefix of one un-split mega-token (mis-classified position)."""
    ascii_out: list[str] = []
    _walk_names("Stratford-upon-Avon", ascii_out)
    uni_out: list[str] = []
    _walk_names("Stratford–upon–Avon", uni_out)  # en-dashes
    assert uni_out == ascii_out == ["stratford", "upon", "avon"]
    # U+2010 HYPHEN variant too.
    out: list[str] = []
    _walk_names("Stoke‐on‐Trent", out)
    assert out == ["stoke", "on", "trent"]
    # Data-driven: every dash in _DASH_TO_SPACE is a word boundary.
    for ch in _DASH_TO_SPACE:
        tok: list[str] = []
        _walk_names(f"great{chr(ch)}tun", tok)
        assert tok == ["great", "tun"], f"U+{ch:04X} not split"


def test_surface_form_for_position_emits_bare_surface() -> None:
    """D45/wyrd-aicu.3: the stored surface is BARE — no position-encoding dash
    (position lives in reflex.position). Case still mirrors the slot: pre keeps
    the front-cap, inner/post lowercase."""
    assert _surface_form_for_position("great", "pre") == "Great"
    assert _surface_form_for_position("great", "inner") == "great"
    assert _surface_form_for_position("great", "post") == "great"


def test_derive_positions_classifies_standalone_word_as_bare(
    tmp_path: Path, fresh_db: Path
) -> None:
    """wyrd-7nab: 'North' as a standalone word ('North Braitham') is classified
    bare — a grammatical standalone word — not a prefix/suffix. Before the
    word-aware fix, stripped whitespace made it look like a prefix/suffix, which
    is what turned two-word toponyms into ungrammatical (filtered) structures."""
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
    # wyrd-7nab word-aware: 'North' is a STANDALONE word in 'North Braitham' /
    # 'Braitham North' → bare (NOT a prefix/suffix — that was the
    # whitespace-stripping artifact). Only 'Hithernorthville' is interior.
    # by_position counts positions OBSERVED (1 per position), not occurrences.
    assert counts["by_position"]["bare"] == 1
    assert counts["by_position"]["inner"] == 1
    assert counts["by_position"]["pre"] == 0
    assert counts["by_position"]["post"] == 0

    with LexiconDB(fresh_db) as db:
        north_id = db.conn.execute("SELECT id FROM etymon WHERE canonical_form='north'").fetchone()[
            "id"
        ]
        north_reflexes = [
            (r["surface_form"], r["position"])
            for r in db.conn.execute(
                "SELECT r.surface_form, r.position FROM reflex_etymon re "
                "JOIN reflex r ON r.id = re.reflex_id WHERE re.etymon_id = ?",
                (north_id,),
            )
        ]
        # bare standalone-word reflex stored under 'post'; interior under 'inner'
        # (D45/aicu.3: surfaces are bare — position is the reflex.position column).
        assert ("north", "post") in north_reflexes
        assert ("north", "inner") in north_reflexes
        assert not any(p == "pre" for _, p in north_reflexes)


def test_derive_positions_picks_up_name_tagged_etymons(tmp_path: Path, fresh_db: Path) -> None:
    """wyrd-5z5j Defect A: a proper-name etymon (Briggs-style — name-tagged but
    NOT cited by wiktionary-empirical) is now scanned + given its ATTESTED
    positioned reflexes (Buna-/-buna-/-buna), instead of collapsing to a single
    bare capitalized form. post/inner are lowercased; pre is word-initial cap."""
    place_names_path = tmp_path / "english_place_names.json"
    place_names_path.write_text(
        json.dumps(
            {
                "England": {
                    "ceremonial_counties": [
                        "Bunaford",  # bunaford   -> pre
                        "Oldbuna",  # oldbuna    -> post
                        "Webunary",  # webunary   -> inner
                        "Buna Hall",  # standalone -> bare
                    ]
                }
            }
        )
    )
    with LexiconDB(fresh_db) as db:
        eid = db.upsert_etymon("Buna", "modern-english")
        # name-tagged, but NO wiktionary-empirical citation — qualifies only via
        # the new name-tag arm of derive_positions.
        db.conn.execute("INSERT INTO etymon_tag (etymon_id, tag) VALUES (?, 'male name')", (eid,))
        db.commit()
        counts = derive_positions(db, {"english": place_names_path}, apply=True)

    assert counts["etymons_examined"] == 1  # picked up via the name tag alone
    with LexiconDB(fresh_db) as db:
        bid = db.conn.execute("SELECT id FROM etymon WHERE canonical_form='Buna'").fetchone()["id"]
        reflexes = {
            (r["surface_form"], r["position"])
            for r in db.conn.execute(
                "SELECT r.surface_form, r.position FROM reflex_etymon re "
                "JOIN reflex r ON r.id = re.reflex_id WHERE re.etymon_id = ?",
                (bid,),
            )
        }
    assert ("Buna", "pre") in reflexes  # word-initial cap (was 'Buna-')
    assert ("buna", "inner") in reflexes  # interior (was '-buna-')
    # D45/aicu.3: the post-suffix ('-buna') and the bare standalone ('buna') both
    # de-dash to ('buna', 'post') and MERGE — the dash that distinguished them is
    # no longer part of the stored identity (position is the reflex.position axis).
    assert ("buna", "post") in reflexes


def test_derive_positions_great_is_bare_and_pre_never_a_suffix(
    tmp_path: Path, fresh_db: Path
) -> None:
    """'Great' is a standalone qualifier word (bare) and a prefix (in 'Greater'),
    but NEVER a real suffix — derive_positions must not write a '-great' post
    reflex. wyrd-7nab: the bare standalone reflex is the dash-less 'great'
    (stored under 'post'); a real suffix would be the dashed '-great'."""
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
        reflexes = {
            (r["surface_form"], r["position"])
            for r in db.conn.execute(
                "SELECT r.surface_form, r.position FROM reflex_etymon re "
                "JOIN reflex r ON r.id = re.reflex_id WHERE re.etymon_id = ?",
                (gid,),
            )
        }
    # prefix (in 'Greater') + standalone qualifier word (bare→post). D45/aicu.3:
    # a real post-suffix would NOW also store as ('great','post') — the dash no
    # longer distinguishes a suffix from a bare standalone. 'Great' has no suffix
    # OCCURRENCE in this corpus, so the only post entry is the standalone; pin
    # the set is EXACTLY the prefix + standalone (no spurious suffix reflex).
    assert reflexes == {("Great", "pre"), ("great", "post")}


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
