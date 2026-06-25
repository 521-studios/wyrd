"""Empirical Wiktionary corpus mining for kenning bundle gaps.

Mines the unaccounted-fragment misses surfaced by ``wyrd kenning unaccounted``
against the wiktextract slices already on disk. For each fragment that hits
a Wiktionary headword in a culture-allowed language, writes an etymon row
plus gloss + citation pointing at a synthetic ``wiktionary-empirical``
source. Treated as an "empirical" class (alongside ``rando-port``) — admitted
to the bundle without the D4 ≥3-scholar-witness gate.

Distinct from ``wiktextract_ingester.py``:
  - The ingester walks every Wiktionary entry, follows etymology/descendants
    templates, and writes ``etymon_descent`` edges. Output: relational graph,
    no glosses, no bundle relevance.
  - This miner walks ONLY headwords that match a corpus-derived miss
    fragment, captures the gloss, and writes a citation that lifts the
    morpheme into the bundle. Output: bundle-eligible empirical entries.

Idempotent: re-runs are no-ops via ``etymon`` UNIQUE(canonical_form, language)
+ INSERT OR IGNORE on gloss/tag/citation.
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import click

from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.lexicon.morpheme_surface import (
    _BOUNDARY_DASHES,
    normalize_morpheme_surface,
)
from wyrd.generators.kenning.lexicon.wiktextract_ingester import (
    _canonical_language,
    _extract_head_template_renderings,
    _extract_pronunciation,
    _map_categories_to_tags,
)
from wyrd.generators.kenning.runtime.meaning import load_meanings
from wyrd.generators.kenning.runtime.name import load_names

# Translate ANY dash (ASCII hyphen + the Unicode variants in _BOUNDARY_DASHES) to
# a space when tokenizing a place name into words. The bundled corpora hyphenate
# with ASCII (``Saint-Pierre``), but a re-sourced / typographic corpus can carry
# an en-dash (``Saint–Pierre``); ASCII-only ``replace("-", " ")`` left that as one
# un-split token, mis-tokenizing the name (D45 — a dash is a boundary, never
# identity; same class as the resolver #777 / cluster-key #778 normalizers).
_DASH_TO_SPACE = str.maketrans(dict.fromkeys(_BOUNDARY_DASHES, " "))

# Synthetic source row written on first --apply run.
WIKTIONARY_EMPIRICAL_SOURCE_ID = "wiktionary-empirical"
_SOURCE_TITLE = "Wiktionary (empirical corpus mining)"
_SOURCE_NOTES = (
    "Empirical class. Headwords matched to unaccounted-fragment misses in "
    "the place-name corpora; admitted to the bundle without the D4 "
    "scholar-witness gate. wiktextract data via kaikki.org (CC-BY-SA 3.0)."
)

# wyrd-vx09: distinct source row for the forms-mining path. Forms are
# variants of existing etymons (e.g. plural / comparative / mutated
# spellings) ingested into etymon_text_match, NOT new etymons. The
# distinct source_id lets dashboard / clear-enrichment paths
# distinguish 'forms' rows from the 'empirical' headword ingest.
WIKTIONARY_FORMS_SOURCE_ID = "wiktionary-forms"
_FORMS_SOURCE_TITLE = "Wiktionary (forms variant mining)"
_FORMS_SOURCE_NOTES = (
    "Variant-mining class. Wiktionary 'forms' arrays ingested as "
    "etymon_text_match rows so the bundle's per-language _variants pool "
    "carries plural / comparative / mutated spellings for non-OE/ON "
    "languages (audit wyrd-f6v4). wiktextract data via kaikki.org "
    "(CC-BY-SA 3.0)."
)

# Tags that mark a forms entry as noise / not a meaningful variant.
# Wiktionary's table-rendering scaffolding leaks structural-template
# entries into the forms array; filtering at ingest keeps them out of
# etymon_text_match. Excludes verbal-conjugation tags too — verbal
# forms aren't useful for place-name variant pools, and including them
# would inflate the variant pool with conjugation noise.
_NOISE_FORM_TAGS: frozenset[str] = frozenset(
    {
        # Structural / rendering scaffolding
        "table-tags",
        "no-table-tags",
        "inflection-template",
        "error-unrecognized-form",
        "class",  # OE strong/weak verb class markers — not a surface variant
        "no-equivalent",  # parity with full ingester's metadata set
        # Verbal conjugation (not relevant to place-name variants).
        # Wiktextract uses both 'preterite' (most modern languages) and
        # 'past' (OE) for past-tense forms; both need to be filtered.
        # Similarly 'infinitive' for the verb head form. wyrd-r1ks
        # review: round-1 OE forms-mining surfaced verb forms of ingān
        # ('inēode', 'ingānne', 'ingān') in the -ing- suffix variant
        # pool because wiktextract emits OE conjugations with 'past' /
        # 'infinitive' tags, neither of which the pre-r1ks set caught.
        "first-person",
        "second-person",
        "third-person",
        # Moods
        "indicative",
        "subjunctive",
        "imperative",
        "conditional",
        "optative",
        # Tenses (preterite vs past for cross-language coverage)
        "present",
        "preterite",
        "past",
        "future",
        "imperfect",
        "pluperfect",
        "perfect",
        "aorist",
        # Voices
        "active",
        "middle",
        "passive",
        # Non-finite + generic verbal
        "impersonal",
        "participle",
        "infinitive",
        "gerund",
        "gerundive",
        "supine",
        "verbal",
    }
)

# Per-culture allowed canonical languages. The miner only accepts a
# Wiktionary entry as evidence for a fragment if the entry's language is
# in the culture's scope — keeps Welsh morphemes from leaking into the
# English bundle and vice versa.
CULTURE_LANG_SCOPE: dict[str, frozenset[str]] = {
    "english": frozenset(
        {
            "old-english",
            "middle-english",
            "old-norse",
            "old-french",
            "anglo-norman",
            "latin",
            "modern-english",
        }
    ),
    "scottish": frozenset(
        {
            "scottish-gaelic",
            "old-irish",
            "middle-irish",
            "old-norse",
            "old-english",
            "scots",
            "latin",
        }
    ),
    "welsh": frozenset(
        {
            "welsh",
            "middle-welsh",
            "old-welsh",
            "proto-celtic",
            "latin",
        }
    ),
    "irish": frozenset(
        {
            "irish",
            "old-irish",
            "middle-irish",
            "latin",
        }
    ),
    "breton": frozenset(
        {
            "breton",
            "middle-breton",
            "old-breton",
            "old-french",
            "latin",
            "proto-celtic",
        }
    ),
}

# Per-canonical-language wiktextract slice basename (None when no slice
# is on disk; lookups in those languages are silently skipped).
_LANG_TO_SLICE_BASENAME: dict[str, str | None] = {
    "old-english": "wiktextract_old_english.jsonl",
    "middle-english": "wiktextract_middle_english.jsonl",
    # wyrd-dxu2: Kaikki English dump downloaded as
    # wiktextract_english.jsonl (~3GB, ~1.47M entries). Mapping was
    # None pre-wyrd-dxu2 because no slice was on disk.
    "modern-english": "wiktextract_english.jsonl",
    "old-norse": "wiktextract_old_norse.jsonl",
    "old-french": "wiktextract_old_french.jsonl",
    "anglo-norman": None,
    "welsh": "wiktextract_welsh.jsonl",
    "middle-welsh": None,
    "old-welsh": None,
    "proto-celtic": "wiktextract_proto_celtic.jsonl",
    "irish": "wiktextract_irish.jsonl",
    "old-irish": "wiktextract_old_irish.jsonl",
    "middle-irish": "wiktextract_middle_irish.jsonl",
    "scottish-gaelic": "wiktextract_scottish_gaelic.jsonl",
    "breton": "wiktextract_breton.jsonl",
    "middle-breton": None,
    "old-breton": None,
    "latin": None,
    "scots": None,
}

# (start_year, end_year) per canonical language for D5 attested_year cell.
# end_year=None means the language is still living. Years are Common Era
# (negative for BCE). These are deliberately coarse — the runtime --era
# filter is itself coarse, and the alternative (per-entry attestation
# year mining from Wiktionary's quotations) is a separate project.
_LANGUAGE_ERA_RANGE: dict[str, tuple[int, int | None]] = {
    "old-english": (700, 1100),
    "middle-english": (1100, 1500),
    "modern-english": (1500, None),
    "old-norse": (700, 1300),
    "old-french": (800, 1300),
    "anglo-norman": (1066, 1300),
    "welsh": (1500, None),
    "middle-welsh": (1100, 1500),
    "old-welsh": (700, 1100),
    "proto-celtic": (-1000, 0),
    "irish": (1500, None),
    "old-irish": (600, 900),
    "middle-irish": (900, 1200),
    "scottish-gaelic": (1500, None),
    "scots": (1100, 1500),
    "breton": (1500, None),
    "middle-breton": (1100, 1500),
    "old-breton": (700, 1100),
    "latin": (-100, 600),
}


def compute_unaccounted_fragments(
    culture: str,
    place_names_path: Path,
    meanings_data: list[dict[str, Any]],
    *,
    min_length: int = 3,
    max_length: int = 12,
) -> Counter:
    """Return ``Counter[candidate_lower]`` for words/morphemes worth looking
    up in Wiktionary as gap-fill candidates for this culture.

    Generates three classes of candidate per imperfect place name:

    1. **Matcher leftover fragments** — the bits the bundle's existing
       matcher couldn't account for. Same set surfaced by the
       ``unaccounted`` CLI.
    2. **Whitespace-separated words** — every space-bounded word in the
       imperfect name (e.g. ``Abbey Demesne`` → ``abbey``, ``demesne``).
       Catches multi-word place names where the matcher partial-matched
       a bogus middle joiner and dropped the real morpheme as adjacent
       short fragments.
    3. **Per-word prefix and suffix substrings** — for each space-word,
       all prefixes and suffixes of length ``[min_length, max_length]``.
       Catches single-word compound names (``Cluainacarrick`` →
       prefix ``cluain``; ``Plouaret`` → prefix ``plou``) where the
       morpheme isn't a whitespace-bounded word but is a clean
       boundary-adjacent substring. Interior substrings are NOT
       generated — too noisy and the trie matcher will eventually
       handle them correctly via segmentation.

    The boundary-only restriction is intentional: morpheme positions
    are usually pre-/post- in place names, not deeply interior. Limiting
    to those positions keeps the noise floor manageable while still
    surfacing the highest-leverage gap-fills.
    """
    word_db, _ = load_meanings(meanings_data)
    names = load_names(json.loads(Path(place_names_path).read_text()))
    candidates: Counter = Counter()
    for name in names:
        name.find_meaning(word_db)
        if name.count_unaccounted() == 0:
            continue
        seen_in_name: set[str] = set()
        _collect_matcher_fragments(name, candidates, seen_in_name, min_length, max_length)
        _collect_word_substrings(name.name, candidates, seen_in_name, min_length, max_length)
    return candidates


def _collect_matcher_fragments(
    name: Any,
    candidates: Counter,
    seen_in_name: set[str],
    min_length: int,
    max_length: int,
) -> None:
    """Class 1 of the candidate generator: pull the unaccounted string
    chunks the matcher left behind in ``name.words[*][*].word`` (the
    ``str`` elements that didn't resolve to a Meaning)."""
    for word_list in name.words.values():
        for word in word_list:
            for chunk in word.word:
                if not isinstance(chunk, str):
                    continue
                _accept_candidate(chunk.lower(), candidates, seen_in_name, min_length, max_length)


def _collect_word_substrings(
    raw_name: str,
    candidates: Counter,
    seen_in_name: set[str],
    min_length: int,
    max_length: int,
) -> None:
    """Classes 2 + 3 of the candidate generator: whitespace-bounded
    words from the original name PLUS each word's boundary-adjacent
    prefixes and suffixes within ``[min_length, max_length]``.

    Splitting on space, dash (any variant — :data:`_DASH_TO_SPACE`), and
    apostrophe matches the conventions of the bundled place-name corpora
    (``Saint-Pierre`` / ``O'Connor`` / ``Bryneglwys``).
    """
    flat_lower = raw_name.lower()
    for word in flat_lower.translate(_DASH_TO_SPACE).replace("'", " ").split():
        if len(word) < min_length:
            continue
        _accept_candidate(word, candidates, seen_in_name, min_length, max_length)
        wlen = len(word)
        # Per-word prefixes (likely pre-position morphemes)
        for end in range(min_length, min(wlen, max_length) + 1):
            _accept_candidate(word[:end], candidates, seen_in_name, min_length, max_length)
        # Per-word suffixes (likely post-position morphemes)
        for start in range(max(0, wlen - max_length), wlen - min_length + 1):
            _accept_candidate(word[start:], candidates, seen_in_name, min_length, max_length)


def _accept_candidate(
    candidate: str,
    candidates: Counter,
    seen_in_name: set[str],
    min_length: int,
    max_length: int,
) -> None:
    """Helper: add a candidate to the counter if it passes the
    length filter and isn't already seen for this place name. The
    per-name dedupe keeps a 12-char place name from inflating
    occurrence counts via overlapping prefix/suffix coverage."""
    if not candidate:
        return
    if len(candidate) < min_length or len(candidate) > max_length:
        return
    if candidate in seen_in_name:
        return
    seen_in_name.add(candidate)
    candidates[candidate] += 1


def _iter_wiktextract_entries(slice_path: Path) -> Iterator[dict[str, Any]]:
    """Stream a wiktextract jsonl, yielding parsed entries one at a time.

    Tolerant of malformed lines (skipped silently) since wiktextract
    output occasionally has stray non-JSON debug lines. The bulk
    streaming model means we never load a 277MB slice into memory; we
    walk it once per --apply call.
    """
    with slice_path.open("rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# wyrd-33cv: wiktextract part-of-speech codes for the function-word
# class. A place-name lexicon wants content words (noun / verb / adj /
# name / …); function words (prepositions, conjunctions, pronouns,
# particles, determiners, articles) that happen to surface-match an
# unaccounted fragment are noise — they never carry toponymic meaning.
# Skipping them at ingest stops the ~function-word slice of the
# wiktextract pollution the wyrd-smtc epic identified. Content POS pass
# through unchanged.
_FUNCTION_WORD_POS: frozenset[str] = frozenset(
    {"prep", "postp", "conj", "pron", "particle", "det", "article"}
)


def _select_canonical_sense(entry: dict[str, Any]) -> tuple[str, list[str]] | None:
    """Pick a representative gloss for an entry, plus mapped tags.

    Returns ``(gloss, tag_list)`` or ``None`` if the entry has no usable
    sense — and the etymon is then skipped at the caller.

    Skips, in order:

    * function-word entries by part-of-speech (wyrd-33cv) — the entry's
      top-level ``pos`` is in ``_FUNCTION_WORD_POS``;
    * Wiktionary-tagged redirect senses (``alt-of`` / ``form-of``);
    * any sense whose gloss TEXT matches our derivative classifier
      (catches the ~50% of Wiktionary derivative entries that aren't
      tag-marked — "inflection of X" / "plural of X" / "soft mutation
      of X" in prose).

    wyrd-a106 root-cause fix (2026-05-23): pre-fix, when every sense
    was a redirect, a fallback returned the first redirect anyway
    ("better than skipping the morpheme entirely"). That fallback was
    THE root cause of ~7500 garbage entries in the bundle — pure
    'alternative form of X' / 'plural of X' / Welsh mutations / etc.
    The fallback is GONE: returning None drops the etymon, which is
    the correct call. Audit (wyrd-36ez) confirmed every wiktionary-
    empirical noise entry was a redirect-fallback ingest.
    """
    # Lazy import to avoid a module-load cycle — wyrd.generators.kenning
    # imports from this package elsewhere.
    from wyrd.generators.kenning import _is_derivative_gloss

    # wyrd-33cv: POS gate — drop function words before inspecting senses.
    if (entry.get("pos") or "").strip() in _FUNCTION_WORD_POS:
        return None

    senses = entry.get("senses") or []
    for sense in senses:
        glosses = sense.get("glosses") or []
        if not glosses:
            continue
        sense_tags = sense.get("tags") or []
        # Wiktionary-tagged redirect.
        if any(t in {"alt-of", "form-of"} for t in sense_tags):
            continue
        gloss = glosses[0].strip()
        if not gloss:
            continue
        # Gloss-text-level derivative check — catches prose
        # 'inflection of X' / 'plural of X' / 'soft mutation of X'
        # even when Wiktionary didn't tag them as alt-of / form-of.
        if _is_derivative_gloss(gloss):
            continue
        cats = [c.get("name", "") for c in (sense.get("categories") or [])]
        return gloss, _map_categories_to_tags(cats)
    return None


# wyrd-vsvi: _CATEGORY_PATTERNS + _map_categories_to_tags moved to
# wiktextract_ingester.py so both the corpus miner and the full
# ingester can use them without an import cycle. Re-exported here
# (top-level import above) so existing call sites in this module
# keep working unchanged.


def missing_slice_languages(
    sources_dir: Path,
    languages: Iterable[str],
) -> list[tuple[str, str]]:
    """Return ``[(language, slice_basename), ...]`` for languages that
    HAVE a published wiktextract slice (``_LANG_TO_SLICE_BASENAME[lang]``
    is non-None) but whose file is absent from ``sources_dir``.

    This is the observable counterpart to ``build_index``'s silent skip.
    A language mapped to ``None`` (Latin, Anglo-Norman, the unpublished
    Welsh/Breton splits) is NOT reported — there's no slice to find, so
    its absence is expected, not an operator error. A language mapped to
    a real basename whose file isn't on disk almost always means
    ``--sources-dir`` points at the wrong place (the repo's OCR-text
    ``sources/`` instead of the bulk cache at ``~/.wyrd/sources``) — the
    failure mode behind wyrd-34kt, where every slice silently skipped and
    the run reported 0 hits with no error.
    """
    seen: set[str] = set()
    missing: list[tuple[str, str]] = []
    for lang in languages:
        if lang in seen:
            continue
        seen.add(lang)
        slice_basename = _LANG_TO_SLICE_BASENAME.get(lang)
        if not slice_basename:
            continue
        if not (sources_dir / slice_basename).exists():
            missing.append((lang, slice_basename))
    return sorted(missing)


def build_index(
    sources_dir: Path,
    target_forms_lower: set[str],
    languages: Iterable[str],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Stream the on-disk wiktextract slices and build
    ``{(form_lower, canonical_language): entry}`` for every headword
    matching the target set.

    Languages with no slice file on disk are silently skipped (covered
    by ``_LANG_TO_SLICE_BASENAME[lang] is None``). That includes Latin,
    Anglo-Norman, Scots, and the old/middle Welsh+Breton splits that
    aren't separately published — those targets simply don't surface
    from this miner. (Modern English is NOT in this set — it maps to
    ``wiktextract_english.jsonl`` since wyrd-dxu2.)

    NOTE: a language that DOES have a published slice but whose file is
    absent from ``sources_dir`` is also skipped silently here — by design,
    so the index builder stays a pure pass over whatever's present. The
    observable signal for that case is ``missing_slice_languages``, which
    ``mine_corpus`` surfaces into ``counts['missing_slices']`` so the CLI
    can warn the operator (wyrd-34kt).
    """
    index: dict[tuple[str, str], dict[str, Any]] = {}
    seen_languages: set[str] = set()
    for lang in languages:
        if lang in seen_languages:
            continue
        seen_languages.add(lang)
        slice_basename = _LANG_TO_SLICE_BASENAME.get(lang)
        if not slice_basename:
            continue
        slice_path = sources_dir / slice_basename
        if not slice_path.exists():
            continue
        for key, entry in _iter_index_entries(slice_path, lang, target_forms_lower):
            # Same headword can have multiple entries with
            # ``etymology_number`` distinguishing senses (e.g. coch¹/coch²);
            # keep the first non-empty one we see.
            if key not in index:
                index[key] = entry
    return index


def _iter_index_entries(
    slice_path: Path, lang: str, target_forms_lower: set[str]
) -> Iterator[tuple[tuple[str, str], dict[str, Any]]]:
    """Yield ``((form_lower, canonical_language), entry)`` for each headword in
    ``slice_path`` matching the target set.

    Skips entries with an empty word, words outside ``target_forms_lower``, and —
    defensively — entries whose declared language doesn't match the slice's
    intended ``lang``: Wiktionary occasionally has cross-language entries in a
    slice (Latin terms in the OE dump as etymon stubs, etc.) that aren't real
    headwords. Pure pass over the slice; the caller applies first-wins dedup.
    """
    for entry in _iter_wiktextract_entries(slice_path):
        word = (entry.get("word") or "").strip()
        if not word:
            continue
        wl = word.lower()
        if wl not in target_forms_lower:
            continue
        entry_lang = _canonical_language(entry.get("lang_code", ""))
        if entry_lang != lang:
            continue
        yield (wl, entry_lang), entry


def _ensure_source_row(db: LexiconDB) -> None:
    db.upsert_source(
        id=WIKTIONARY_EMPIRICAL_SOURCE_ID,
        title=_SOURCE_TITLE,
        notes=_SOURCE_NOTES,
    )


def mine_corpus(
    db: LexiconDB,
    *,
    fragments_by_culture: dict[str, Counter],
    sources_dir: Path,
    apply: bool = False,
) -> dict[str, Any]:
    """Mine empirical Wiktionary entries for the given per-culture
    fragment sets.

    For each fragment that hits a Wiktionary headword in a
    culture-allowed language, upsert the etymon, attach the gloss
    (and any mapped tags) and the empirical citation. Returns counters
    suitable for CLI summary printing.
    """
    if apply:
        _ensure_source_row(db)

    target_forms, union_languages = _candidate_targets(fragments_by_culture)
    index = build_index(sources_dir, target_forms, union_languages)
    counts = _init_mine_counts(target_forms, index, fragments_by_culture)
    # wyrd-34kt: surface languages whose published slice file is absent
    # from sources_dir so a wrong --sources-dir (or an un-fetched bulk
    # cache) reads as a loud warning instead of a silent 0-hits run.
    counts["missing_slices"] = missing_slice_languages(sources_dir, union_languages)

    if not apply:
        _dry_run_count(fragments_by_culture, index, counts)
        return counts

    _apply_write(db, fragments_by_culture, index, counts)
    db.commit()
    return counts


def _candidate_targets(
    fragments_by_culture: dict[str, Counter],
) -> tuple[set[str], set[str]]:
    """Union the per-culture fragment sets and the per-culture language
    scopes into the two flat sets ``build_index`` consumes (target
    forms + allowed languages)."""
    target_forms: set[str] = set()
    for c in fragments_by_culture.values():
        target_forms.update(c.keys())
    union_languages: set[str] = set()
    for culture in fragments_by_culture:
        union_languages.update(CULTURE_LANG_SCOPE.get(culture, frozenset()))
    return target_forms, union_languages


def _init_mine_counts(
    target_forms: set[str],
    index: dict[tuple[str, str], dict[str, Any]],
    fragments_by_culture: dict[str, Counter],
) -> dict[str, Any]:
    """Build the counters dict mine_corpus returns. Per-culture
    fragment counts are seeded up front so a dry-run still reports
    each culture even when zero hits land in its scope."""
    counts: dict[str, Any] = {
        "fragments_input": len(target_forms),
        "wiktionary_hits": len(index),
        "etymons_upserted": 0,
        "glosses_added": 0,
        "tags_added": 0,
        "citations_added": 0,
        # wyrd-aicu.8: headwords dropped because they de-dash to empty junk —
        # counted so the de-dash's drop count is visible, not silent.
        "words_skipped_empty": 0,
        # wyrd-69s5: capture-stat counters mirroring the full ingester
        # so per-run dry-run / apply summaries surface multi-script
        # rendering coverage in the same shape across both paths.
        "pronunciation_captured": 0,
        "original_script_captured": 0,
        "transliteration_captured": 0,
        "by_culture": defaultdict(lambda: {"fragments": 0, "hits": 0}),
        "by_language": Counter(),
        # wyrd-34kt: [(language, slice_basename), ...] for published
        # slices missing from sources_dir; populated by mine_corpus.
        "missing_slices": [],
    }
    for culture, frags in fragments_by_culture.items():
        counts["by_culture"][culture]["fragments"] = len(frags)
    return counts


def _dry_run_count(
    fragments_by_culture: dict[str, Counter],
    index: dict[tuple[str, str], dict[str, Any]],
    counts: dict[str, Any],
) -> None:
    """Walk per-culture (fragment × language) pairs and increment the
    by-culture / by-language counters for any (frag, lang) hit in the
    Wiktextract index. No DB writes."""
    for culture, frags in fragments_by_culture.items():
        scope = CULTURE_LANG_SCOPE.get(culture, frozenset())
        for frag in frags:
            for lang in scope:
                if (frag, lang) in index:
                    counts["by_culture"][culture]["hits"] += 1
                    counts["by_language"][lang] += 1


def _apply_write(
    db: LexiconDB,
    fragments_by_culture: dict[str, Counter],
    index: dict[tuple[str, str], dict[str, Any]],
    counts: dict[str, Any],
) -> None:
    """Apply path: walk per-culture (fragment × language) pairs and
    write each unique (frag, lang) hit as an etymon + gloss + tag +
    citation. Dedupe across cultures via the ``written`` set so a
    morpheme that's in scope for multiple cultures (e.g. old-french
    appears in both english and breton scopes) is only persisted once."""
    written: set[tuple[str, str]] = set()
    for culture, frags in fragments_by_culture.items():
        scope = CULTURE_LANG_SCOPE.get(culture, frozenset())
        for frag in frags:
            for lang in scope:
                key = (frag, lang)
                entry = index.get(key)
                if entry is None:
                    continue
                counts["by_culture"][culture]["hits"] += 1
                counts["by_language"][lang] += 1
                if key in written:
                    continue
                written.add(key)
                _write_one(db, entry, lang, counts)


def _write_one(
    db: LexiconDB,
    entry: dict[str, Any],
    canonical_lang: str,
    counts: dict[str, Any],
) -> None:
    """Persist one wiktextract entry as an empirical etymon.

    wyrd-69s5: threads ``pronunciation_ipa`` + ``pronunciation_dialect``
    AND the multi-script renderings (``original_script`` +
    ``transliteration``) from the entry's ``sounds`` and
    ``head_templates`` arrays into the upsert. Full parity with the
    ingester path (``wiktextract_ingester._process_entry``) which has
    been doing this since wyrd-ha9q. Pre-fix, the corpus-miner path
    dropped all four fields, so European-language etymons admitted via
    empirical mining shipped with empty pronunciation + rendering
    columns even though the source data carried them. Reusing the
    ingester's helpers keeps both paths consistent on extraction
    strategy.

    The English culture scope includes non-Latin-script languages
    (Latin → Greek borrowing chains, etc.), so ``original_script`` /
    ``transliteration`` matters here too — not just for the explicitly-
    non-Latin slices the full ingester also handles.
    """
    # De-dash to the bare morpheme surface (wyrd-aicu.8, D45); a strip-to-empty
    # junk word (a bare '-' headword) normalizes to None and is dropped here at
    # the ingest boundary, before the upsert choke would reject it — counted so
    # the de-dash's drop is visible.
    canonical_form = normalize_morpheme_surface(entry.get("word"))
    if not canonical_form:
        counts["words_skipped_empty"] += 1
        return
    sense = _select_canonical_sense(entry)
    if sense is None:
        return
    gloss, tags = sense

    pron_ipa, pron_dialect = _extract_pronunciation(entry.get("sounds") or [])
    orig_script, translit = _extract_head_template_renderings(
        entry.get("head_templates") or [], canonical_form
    )

    etymon_id = db.upsert_etymon(
        canonical_form,
        canonical_lang,
        pronunciation_ipa=pron_ipa,
        pronunciation_dialect=pron_dialect,
        original_script=orig_script,
        transliteration=translit,
    )
    counts["etymons_upserted"] += 1
    if pron_ipa:
        counts["pronunciation_captured"] = counts.get("pronunciation_captured", 0) + 1
    if orig_script:
        counts["original_script_captured"] = counts.get("original_script_captured", 0) + 1
    if translit:
        counts["transliteration_captured"] = counts.get("transliteration_captured", 0) + 1

    db.add_gloss(etymon_id, gloss)
    counts["glosses_added"] += 1

    for tag in tags:
        db.add_tag(etymon_id, tag)
        counts["tags_added"] += 1

    db.add_citation(
        etymon_id,
        WIKTIONARY_EMPIRICAL_SOURCE_ID,
        short_quote=gloss[:200],
    )
    counts["citations_added"] += 1


# --- wyrd-vx09: forms-mining path -----------------------------------------


def _meaningful_form_label(form_entry: dict[str, Any]) -> str | None:
    """For a wiktextract forms-array entry, return a single-string
    label describing the form (e.g. 'plural', 'comparative',
    'soft-mutation'), or None if the entry is noise.

    Wiktionary's forms array carries both meaningful inflection /
    mutation entries AND rendering scaffolding (table-tags,
    inflection-template). Tags from ``_NOISE_FORM_TAGS`` exclude the
    entry entirely. The label joins remaining tags with hyphens —
    a 'soft' + 'mutation' form becomes 'soft-mutation'. Mutation-
    specific entries (source='mutation') prefix with 'mutation:'
    so consumers can distinguish initial-mutation variants from
    plain inflection.
    """
    tags = form_entry.get("tags") or []
    if not tags:
        return None
    if any(t in _NOISE_FORM_TAGS for t in tags):
        return None
    # Join remaining (meaningful) tags into a label.
    label = "-".join(tags)
    if not label:
        return None
    if form_entry.get("source") == "mutation":
        label = f"mutation:{label}"
    return label


def _iter_meaningful_forms(
    entry: dict[str, Any],
) -> Iterator[tuple[str, str]]:
    """For a wiktextract entry, yield ``(form, label)`` pairs for each
    meaningful (non-noise) form. Skips:

    * Forms identical to the canonical word (no self-match).
    * Forms with only noise tags (table-tags / inflection-template /
      verbal-conjugation).
    * Forms whose stripped surface is empty.

    Comparison is case-insensitive on the self-match check — Welsh
    'A' and 'a' are the same surface for variant-pool purposes.
    """
    canonical = (entry.get("word") or "").strip()
    canonical_lower = canonical.lower()
    for form_entry in entry.get("forms") or []:
        if not isinstance(form_entry, dict):
            continue
        form = (form_entry.get("form") or "").strip()
        if not form:
            continue
        if form.lower() == canonical_lower:
            continue
        label = _meaningful_form_label(form_entry)
        if label is None:
            continue
        yield form, label


def _ensure_forms_source_row(db: LexiconDB) -> None:
    """Idempotently insert the wiktionary-forms synthetic source row.
    Mirror of ``_ensure_source_row`` for the empirical path."""
    db.upsert_source(
        id=WIKTIONARY_FORMS_SOURCE_ID,
        title=_FORMS_SOURCE_TITLE,
        notes=_FORMS_SOURCE_NOTES,
    )


def _process_corpus_entry(
    db: LexiconDB, entry: dict[str, Any], counts: dict[str, Any], *, apply: bool
) -> bool:
    """Enrich one wiktextract entry's meaningful ``forms`` into
    etymon_text_match. Returns True when the entry was processed (its canonical
    etymon exists), False for a skip (no word, no resolvable language, or the
    canonical headword isn't in the lexicon — counted under ``etymons_missing``;
    forms-mining enriches existing rows, never creates them). Mutates ``counts``
    in place; DB writes (idempotent ON CONFLICT upsert) are gated on ``apply``."""
    # De-dash the lookup form (wyrd-aicu.8, D45) so a dashed wiktextract headword
    # resolves to the bare etymon row the de-dashed write path now stores.
    canonical_form = normalize_morpheme_surface(entry.get("word"))
    if not canonical_form:
        return False
    # ``_canonical_language`` returns the empty string for missing lang_code
    # (not None); guard on truthiness so we short-circuit before a wasted SELECT.
    canonical_lang = _canonical_language(entry.get("lang_code") or "")
    if not canonical_lang:
        return False
    row = db.conn.execute(
        "SELECT id FROM etymon WHERE canonical_form = ? AND language = ? AND merged_into_id IS NULL",
        (canonical_form, canonical_lang),
    ).fetchone()
    if row is None:
        counts["etymons_missing"] += 1
        return False
    etymon_id = row["id"]
    # Count noise BEFORE filtering so the dashboard can surface how much of the
    # slice's forms array is structural noise vs meaningful inflection.
    forms_in_entry = entry.get("forms") or []
    meaningful = list(_iter_meaningful_forms(entry))
    counts["forms_skipped_noise"] += len(forms_in_entry) - len(meaningful)
    for form, label in meaningful:
        counts["forms_processed"] += 1
        if not apply:
            continue
        db.conn.execute(
            """
            INSERT INTO etymon_text_match
              (etymon_id, source_id, matched_form, match_count,
               edit_distance, snippet, method)
            VALUES (?, ?, ?, 1, 0, ?, 'wiktionary-forms-v1')
            ON CONFLICT(etymon_id, source_id, matched_form) DO UPDATE
              SET match_count = match_count + 1
            """,
            (etymon_id, WIKTIONARY_FORMS_SOURCE_ID, form, label),
        )
        counts["forms_written"] += 1
    return True


def _emit_corpus_progress(counts: dict[str, Any], started_at: float, *, final: bool) -> None:
    """Emit the CLAUDE.md-convention progress line (stderr, with s/entry rate)
    — periodic (every 1000 entries) or the final summary (which also reports
    ``forms_skipped_noise`` so the last partial chunk shows up)."""
    walked = counts["entries_walked"]
    rate = (time.time() - started_at) / walked if walked else 0.0
    if final:
        click.echo(
            f"  [{walked}] (final) "
            f"forms_processed={counts['forms_processed']} "
            f"forms_skipped_noise={counts['forms_skipped_noise']} "
            f"forms_written={counts['forms_written']} "
            f"etymons_missing={counts['etymons_missing']} "
            f"({rate:.4f}s/entry)",
            err=True,
        )
    else:
        click.echo(
            f"  [{walked}] "
            f"forms_processed={counts['forms_processed']} "
            f"forms_written={counts['forms_written']} "
            f"etymons_missing={counts['etymons_missing']} "
            f"({rate:.4f}s/entry)",
            err=True,
        )


def mine_corpus_forms(
    db: LexiconDB,
    slice_path: Path,
    *,
    apply: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """wyrd-vx09: walk a wiktextract slice and persist each entry's
    ``forms`` array into ``etymon_text_match`` so the bundle's per-
    language _variants pool gets non-OE/ON variant data.

    Audit context (wyrd-f6v4): scholarly etymology dictionaries (the
    OE/ON corpora) have prose bodies the existing reverse-search +
    fuzzy-search infrastructure can scan; Welsh / Irish / Celtic /
    French wiktextract corpora are headword-only with no prose. The
    forms array IS the structured-data path for those languages.

    For each entry:
    * Resolve the canonical etymon (must already exist — typically
      ingested via ``mine_corpus`` first; this pass enriches existing
      rows, not creates new ones).
    * Filter forms via ``_iter_meaningful_forms`` (skip noise tags,
      verbal conjugation, self-matches).
    * Upsert each meaningful form as an ``etymon_text_match`` row
      with ``source_id='wiktionary-forms'`` and the form-label as
      snippet for human-readable provenance.

    Returns dict of counts: ``entries_walked``, ``forms_processed``,
    ``forms_skipped_noise`` (forms filtered out by
    ``_iter_meaningful_forms``), ``forms_written``,
    ``etymons_missing`` (canonical headword wasn't in the lexicon —
    likely the corpus was never ingested via mine_corpus).

    Idempotent on (etymon_id, source_id, matched_form) UNIQUE: re-
    runs silently update match_count without duplicating rows.

    Progress reporting per CLAUDE.md convention: stderr line every
    1000 entries with running counts + s/entry rate so operators
    can extrapolate ETA on multi-minute slice walks.
    """
    if apply:
        _ensure_forms_source_row(db)

    counts: dict[str, Any] = {
        "entries_walked": 0,
        "forms_processed": 0,
        "forms_skipped_noise": 0,
        "forms_written": 0,
        "etymons_missing": 0,
    }
    started_at = time.time()
    for entry in _iter_wiktextract_entries(slice_path):
        if limit is not None and counts["entries_walked"] >= limit:
            break
        counts["entries_walked"] += 1
        if not _process_corpus_entry(db, entry, counts, apply=apply):
            continue  # skipped entry (no word / lang / etymon) — no progress line
        # Periodic commit + progress line every 1000 PROCESSED entries
        # (CLAUDE.md mining-progress convention: ~every-N records, stderr,
        # s/entry rate). 1000 is the right cadence for 50k+-entry slices.
        if counts["entries_walked"] % 1000 == 0:
            _emit_corpus_progress(counts, started_at, final=False)
            if apply:
                db.commit()

    # Final progress line so the last partial chunk shows up.
    _emit_corpus_progress(counts, started_at, final=True)

    if apply:
        db.commit()
    return counts


def derive_positions(
    db: LexiconDB,
    place_names_paths_by_culture: dict[str, Path],
    *,
    min_count: int = 1,
    apply: bool = False,
) -> dict[str, Any]:
    """For every etymon cited by ``wiktionary-empirical`` OR tagged as a
    proper name (male/female/family name), scan the relevant cultures'
    place-name corpora and write reflex rows for every position the
    canonical form occurs in.

    Necessary because the empirical-mine writes etymons with no
    ``position_pref`` set; without per-position reflex rows the bundle
    export emits a single undecorated ``modern_usage`` which the
    runtime treats as post-only. ``Great Braitham`` would then never
    parse via the empirical entry — ``Great`` is a pre morpheme.

    Algorithm: for each empirical etymon, find the cultures whose
    language-scope includes that etymon's language; flatten each
    relevant place name to lowercase + strip whitespace/hyphens/
    apostrophes; locate every occurrence of the canonical form;
    classify each occurrence by its position within the flattened
    name (start = pre, end = post, interior = inner, full = pre+post).
    For every position with count ≥ ``min_count``, upsert a reflex row
    (``Great-``, ``-great-``, ``-great``) and link it to the etymon.

    Idempotent: ``upsert_reflex`` and ``link_reflex_etymon`` are both
    safe on re-run.
    """
    flat_names_by_culture: dict[str, list[str]] = {
        culture: _load_flat_names(p) for culture, p in place_names_paths_by_culture.items()
    }
    # Candidates: etymons cited by wiktionary-empirical, OR tagged as a proper
    # name (wyrd-5z5j / Defect A). Personal-name etymons (Briggs) are cited by
    # the names index, not wiktionary-empirical, so without the name-tag arm
    # they never get positioned reflexes — they collapse to a single bare
    # capitalized `modern_usage` and the runtime drops them mid-word as
    # `...Buna...`. Scanning them against the place-name corpora here gives each
    # name its ATTESTED positioned reflexes (`Buna-`, `-buna-`, `-buna`); names
    # that never occur in a toponym observe no position and stay bare (correct).
    rows = list(
        db.conn.execute(
            """
            SELECT DISTINCT e.id, e.canonical_form, e.language
            FROM etymon e
            WHERE e.merged_into_id IS NULL
              AND (
                EXISTS (
                    SELECT 1 FROM etymon_citation c
                    WHERE c.etymon_id = e.id AND c.source_id = ?
                )
                OR EXISTS (
                    SELECT 1 FROM etymon_tag t
                    WHERE t.etymon_id = e.id
                      AND t.tag IN ('male name', 'female name', 'family name')
                )
              )
            ORDER BY e.id
            """,
            (WIKTIONARY_EMPIRICAL_SOURCE_ID,),
        )
    )
    counts: dict[str, Any] = {
        "etymons_examined": len(rows),
        "reflex_rows_upserted": 0,
        "links_added": 0,
        "by_position": Counter(),
        "etymons_with_no_observed_position": 0,
    }
    for row in rows:
        position_counts = _classify_one_etymon(
            row["canonical_form"], row["language"], flat_names_by_culture
        )
        # ``None`` means the etymon's language has no culture in scope —
        # the etymon was never EVALUATED, so don't bucket it as
        # "evaluated but no position observed". Mirrors the original
        # ``continue`` before the counter increment.
        if position_counts is None:
            continue
        if not any(v >= min_count for v in position_counts.values()):
            counts["etymons_with_no_observed_position"] += 1
            continue
        _write_reflexes_for(
            db, row["id"], row["canonical_form"], position_counts, min_count, apply, counts
        )
    if apply:
        db.commit()
    return counts


def _classify_one_etymon(
    canonical_form: str,
    lang: str,
    flat_names_by_culture: dict[str, list[str]],
) -> dict[str, int] | None:
    """Determine which cultures' corpora are in scope for ``lang`` and
    classify positions of ``canonical_form`` across the union of their
    flattened names.

    Returns the per-position count dict (pre/post/inner) when at least
    one culture's scope includes ``lang``. Returns ``None`` when no
    culture is in scope — the caller distinguishes 'not evaluated'
    from 'evaluated but no observed position' so the
    ``etymons_with_no_observed_position`` counter only counts the
    latter."""
    relevant_cultures = [c for c, scope in CULTURE_LANG_SCOPE.items() if lang in scope]
    if not relevant_cultures:
        return None
    names_for_culture: list[str] = []
    for c in relevant_cultures:
        names_for_culture.extend(flat_names_by_culture.get(c, []))
    return _classify_positions(canonical_form, names_for_culture)


def _write_reflexes_for(
    db: LexiconDB,
    etymon_id: int,
    canonical_form: str,
    position_counts: dict[str, int],
    min_count: int,
    apply: bool,
    counts: dict[str, Any],
) -> None:
    """For each position with ``count >= min_count``, write a reflex
    row (``Great-`` / ``-great-`` / ``-great``) and link it to the
    etymon. Mutates the global counter dict in place."""
    for pos, n in position_counts.items():
        if n < min_count:
            continue
        counts["by_position"][pos] += 1
        if not apply:
            continue
        surface_form = _surface_form_for_position(canonical_form, pos)
        # wyrd-7nab: a 'bare' (standalone-word) reflex has a DASH-LESS surface;
        # the runtime derives 'bare' from that (no dash markers), and
        # reflex.position is only a sort key whose CHECK predates 'bare'. Store
        # it under 'post' — matching the existing rando-port convention
        # (position_from_usage maps a dash-less modern_usage to 'post').
        stored_pos = "post" if pos == "bare" else pos
        reflex_id = db.upsert_reflex(surface_form, stored_pos)
        counts["reflex_rows_upserted"] += 1
        db.link_reflex_etymon(reflex_id, etymon_id)
        counts["links_added"] += 1


def _load_flat_names(place_names_path: Path) -> list[str]:
    """Read a place-names JSON corpus and return a flat list of lowercased
    WORDS — each space-/hyphen-separated word of every name, apostrophes
    stripped.

    wyrd-7nab: words are kept SEPARATE (not flattened into one string) so a
    standalone word (`Combe` in `Castle Combe`) is classifiable as ``bare``,
    not mistaken for a suffix. Splitting on space AND hyphen treats
    ``Great-Abington`` like ``Great Abington`` — both are two words. This is
    what restores grammatical two-word structures (and the ~30% two-word rate).
    """
    # encoding="utf-8" is required: place-name corpora carry non-ASCII
    # diacritics (Welsh ŵ/ŷ, French é/è, OE æ/ð/þ) and the platform default
    # encoding would raise UnicodeDecodeError on non-UTF-8 hosts (Gemini review).
    data = json.loads(place_names_path.read_text(encoding="utf-8"))
    out: list[str] = []
    _walk_names(data, out)
    return out


def _walk_names(node: Any, out: list[str]) -> None:
    """Recursive walk of the nested place_names JSON, emitting per-word tokens
    (split on space + dash of any variant, lowercased, apostrophes stripped)."""
    if isinstance(node, str):
        for word in node.lower().translate(_DASH_TO_SPACE).replace("'", "").split():
            if word:
                out.append(word)
    elif isinstance(node, list):
        for child in node:
            _walk_names(child, out)
    elif isinstance(node, dict):
        for v in node.values():
            _walk_names(v, out)


def _classify_positions(canonical_form: str, words: list[str]) -> dict[str, int]:
    """Count bare/pre/post/inner occurrences of ``canonical_form`` across the
    word list (wyrd-7nab — word-aware). A word that IS the form → ``bare`` (a
    standalone word); otherwise prefix → ``pre``, suffix → ``post``, interior →
    ``inner``. The ``bare`` count is what was missing before (whitespace was
    stripped, so standalone words always looked like a prefix/suffix)."""
    form = canonical_form.lower()
    pos_counts = {"bare": 0, "pre": 0, "post": 0, "inner": 0}
    if len(form) < 2:
        return pos_counts
    flen = len(form)
    for word in words:
        if word == form:
            pos_counts["bare"] += 1
            continue
        i = 0
        wlen = len(word)
        while True:
            idx = word.find(form, i)
            if idx == -1:
                break
            if idx == 0:
                pos_counts["pre"] += 1
            elif idx + flen == wlen:
                pos_counts["post"] += 1
            else:
                pos_counts["inner"] += 1
            i = idx + 1  # allow overlapping matches
    return pos_counts


def _surface_form_for_position(canonical_form: str, position: str) -> str:
    """The BARE surface to store in ``reflex.surface_form`` for a (form,
    position). D45 / wyrd-aicu.3: the stored identity is the bare surface;
    position lives in the ``reflex.position`` column, never dash-encoded.

    Case mirrors the slot (unchanged from the old dash convention, minus the
    dashes): ``pre`` keeps the front-cap (it leads a name); ``inner`` / ``post``
    lowercase. So ``pre`` → ``Great``, ``inner`` / ``post`` → ``great`` — the
    same surfaces the 0017 migration's ``TRIM(surface_form, '-')`` produces for
    the pre-existing dashed rows.
    """
    form = canonical_form.lower()
    if position == "pre":
        return f"{form[:1].upper()}{form[1:]}"
    return form


# derive_attested_year was previously defined here but never called from
# CLI or mine_corpus. Removed in wyrd-7bu1 per the test-coverage-reviewer
# note that it was untested dead code. Reintroduce here (or write to
# etymon_text_match.attested_year directly during mine_corpus._write_one)
# when the empirical-mining pipeline starts populating per-form era data
# — at which point _LANGUAGE_ERA_RANGE below is the natural source of
# truth.
