"""CLI commands for Kenning. Mounted under `wyrd kenning <subcommand>`."""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from importlib import resources
from pathlib import Path
from typing import Any

import click

from wyrd.generators.kenning import (
    _INTERNAL_TAGS,
    CULTURES,
    MOODS,
    Kenning,
    KenningExplain,
    _load_meanings,
    anthropic_extractor,
    available_tags,
    etymonline_ingester,
    fantasy_pipeline,
    gemini_extractor,
    llm_extractor,
    pfsrd2_monster_extractor,
)
from wyrd.generators.kenning.dictionary_parser import (
    parse_alphabetical_text,
    parse_numbered_list_text,
)
from wyrd.generators.kenning.english_shaping import (
    PHASE2A_NON_LATIN_LANGS,
    derive_english_shaped,
)
from wyrd.generators.kenning.era import (
    era_cell,
    era_cell_for_input,
    language_family,
)
from wyrd.generators.kenning.lexicon import (
    LANGUAGE_FIELDS,
    RECOMMENDED_LANG_THRESHOLDS,
    LexiconDB,
    annotate_fragments_with_corpus_evidence,
    assign_etymon_to_meaning_synset,
    backfill_citation_pages,
    bridge_celtic_forms,
    bridge_generic_language,
    bridge_phonological_oe,
    bridge_phonological_on,
    clear_enrichment,
    cluster_cognates,
    cluster_ocr_variants,
    collect_canonical_decompositions,
    collect_fantasy_morphemes,
    detect_running_headers,
    etymon_era_reflexes,
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
    mine_toponym_attestations,
    project_period_forms,
    record_mining_run,
    reverse_search_attestations,
    seed_from_meanings,
    seed_meaning_synsets,
)
from wyrd.generators.kenning.meaning import Meaning, load_meanings
from wyrd.generators.kenning.name import load_names_with_regions
from wyrd.generators.kenning.paths import (
    LEXICON_DB_DEFAULT_DISPLAY,
    default_lexicon_path,
)
from wyrd.generators.kenning.rewind import (
    DEFAULT_ENGLISH_ERAS,
    rewind_name,
)
from wyrd.generators.kenning.skeat_parser import parse_skeat_text
from wyrd.generators.kenning.wiktextract_corpus_miner import (
    WIKTIONARY_EMPIRICAL_SOURCE_ID,
    compute_unaccounted_fragments,
    derive_positions,
    mine_corpus,
    mine_corpus_forms,
)
from wyrd.generators.kenning.wiktextract_ingester import ingest_wiktextract_path
from wyrd.seed import resolve_seed, rng_for

# wyrd-hidb Phase 2: PHASE2A_NON_LATIN_LANGS lives in
# english_shaping.py (the wrapper that consumes it lives there too).
# The CLI imports it for the SELECT-filter on `lexicon derive-english-shaped`.
_PHASE2A_NON_LATIN_LANGS = PHASE2A_NON_LATIN_LANGS


def _select_parser_and_run(
    text: str,
    parser: str,
    *,
    min_entry_number: int = 1,
    max_entry_number: int | None = None,
) -> list:
    """Pick the right parser for a book.

    Modes:
      skeat         - force Skeat-format parser
      alphabetical  - force alphabetical-dictionary parser
      numbered-list - force numbered-list parser (wyrd-5af). Right tool
                      for chrestomathy / treatise sources whose etymology
                      bodies are organized as ordinal-prefixed lists
                      (Longnon vol 2 saint-names: '1659. Caradocus :
                      Saint-Caradu (...)'). Skip on dictionary-shaped
                      sources — alphabetical is the right tool there.
      auto          - try Skeat first; if it returns ≥5 entries, use it.
                      Otherwise use alphabetical. ``auto`` does NOT pick
                      numbered-list — opt in explicitly because the
                      numbered-list shape co-exists with body prose in
                      treatise sources, and we don't want a few stray
                      ordinals to flip a Mawer-shaped book onto the
                      wrong parser.

    ``min_entry_number`` / ``max_entry_number`` apply ONLY to the
    numbered-list parser (the other modes don't have an ordinal axis).
    Inclusive bounds; default to "no filter" so the historic
    skeat/alphabetical paths stay bit-stable.
    """
    if parser == "skeat":
        return parse_skeat_text(text)
    if parser == "alphabetical":
        return parse_alphabetical_text(text)
    if parser == "numbered-list":
        return parse_numbered_list_text(
            text,
            min_entry_number=min_entry_number,
            max_entry_number=max_entry_number,
        )
    skeat_result = parse_skeat_text(text)
    if len(skeat_result) >= 5:
        return skeat_result
    return parse_alphabetical_text(text)


def _load_meanings_data(meanings: Path | None) -> dict:
    """Load meanings.json from an optional override path or the bundled default."""
    if meanings is None:
        text = resources.files("wyrd.generators.kenning.data").joinpath("meanings.json").read_text()
    else:
        text = meanings.read_text()
    return json.loads(text)


# Default location for the authoring lexicon DB. The callable is
# resolved per CLI invocation so the WYRD_LEXICON_DB env override and
# the one-time legacy → ~/.wyrd migration both fire at call time, not
# at module-import time. See wyrd/generators/kenning/paths.py.
_DEFAULT_LEXICON_PATH = default_lexicon_path

# Synthetic source row representing the meanings.json data inherited from the
# Rando port. Authority unverified per-entry; entries flagged for review as
# scholarly sources are mined.
_RANDO_SOURCE = {
    "id": "rando-port",
    "title": "Inherited morpheme database from Rando port",
    "author": "Rando project (unknown contributors)",
    "region": "British Isles",
    "language_focus": "mixed",
    "notes": (
        "Original meanings.json ported from Rando. Per-entry authority "
        "unverified; treat as low confidence pending corroboration from "
        "scholarly sources."
    ),
}


@click.group()
def cli() -> None:
    """Kenning — town name generation and authoring tools."""


@cli.command()
@click.argument("culture", type=click.Choice(CULTURES))
@click.option("--tag", "tags", multiple=True, help="Filter by tag (repeatable).")
@click.option(
    "--count", "-n", type=click.IntRange(1, 10), default=5, help="Generate N names (1–10)."
)
@click.option("--seed", type=int, default=None, help="Reproducible 64-bit seed.")
@click.option(
    "--describe/--no-describe",
    default=True,
    help="Print the morpheme breakdown after each name.",
)
@click.option(
    "--spelling-variety",
    type=click.FloatRange(0.0, 1.0),
    default=0.0,
    show_default=True,
    help=(
        "D18 spelling-variant substitution probability (0..1). At >0, each morpheme "
        "is rolled for replacement with an attested archaic spelling drawn from "
        "the etymon's variant pool."
    ),
)
@click.option(
    "--novelty",
    type=click.FloatRange(0.0, 1.0),
    default=0.0,
    show_default=True,
    help=(
        "D17 mixture knob (0..1). 0 = pure empirical-frequency sampling (today's "
        "behavior); 1 = uniform marginal over in-bucket morphemes; intermediate "
        "values blend, allowing plausible-but-unattested combinations."
    ),
)
@click.option(
    "--inflection-density",
    type=click.FloatRange(0.0, 1.0),
    default=0.0,
    show_default=True,
    help=(
        "D8 inflection knob (0..1). Per-morpheme probability of substituting an "
        "inflected form (cotum/cotan/cotes) for the lemma (cot)."
    ),
)
@click.option(
    "--mood",
    "moods",
    multiple=True,
    help=(
        "D6 stylistic-mood preset (repeatable). 'grim' applies a menacing "
        "semantic-tag union; 'harsh' biases sampling toward stop-final / "
        "cluster-heavy morphemes; 'harsh:VALUE' graduates the phonological "
        "skew (e.g. 'harsh:0.5'). Multiple --mood flags compose."
    ),
)
@click.option(
    "--include-fiction",
    "include_fiction",
    is_flag=True,
    default=False,
    help=(
        "wyrd-yan: allow morphemes tagged 'fiction' (constructed etymologies "
        "for bestiary / NPC / homebrew content) to appear. Off by default — "
        "realistic mode draws only from scholarly-attested morphemes."
    ),
)
@click.option(
    "--era",
    type=str,
    default=None,
    help=(
        "D5-2 era filter (wyrd-lyp). Pass a year (e.g. 1086), a cell label "
        "('oe-late', 'me', 'middle-irish'), or 'family/label' to "
        "disambiguate. Morphemes with no attested-year data pass through."
    ),
)
@click.option(
    "--stratum",
    type=str,
    default=None,
    help=(
        "wyrd-lr4 Phase 3 within-language stratum filter. Pass a register "
        "tag — for Welsh: 'native-welsh', 'brittonic-substrate', "
        "'medieval-welsh', 'latin-loan', 'english-loan'; for French: "
        "'native-french', 'medieval-french', 'gallo-roman', "
        "'gaulish-substrate', 'frankish-substrate'; for Old English: "
        "'native-old-english', 'celtic-substrate', 'norse-loan', "
        "'latin-loan'; for Old Norse: 'native-old-norse', 'east-norse', "
        "'gaelic-substrate', 'english-loan', 'low-german-loan', "
        "'latin-loan'. Morphemes with no stratum data pass through. "
        "Composes with --era via intersection."
    ),
)
@click.option(
    "--cohesion",
    type=click.FloatRange(0.0, 1.0),
    default=0.0,
    show_default=True,
    help=(
        "wyrd-mj2 tag co-occurrence bias (0..1). 0 keeps independent slot "
        "sampling (today's behavior); higher values bias each slot toward "
        "usages whose tags co-occur with previously-picked slots in the "
        "empirical corpus. Composes orthogonally with --novelty."
    ),
)
def generate(
    culture: str,
    tags: tuple[str, ...],
    count: int,
    seed: int | None,
    describe: bool,
    spelling_variety: float,
    novelty: float,
    inflection_density: float,
    moods: tuple[str, ...],
    include_fiction: bool,
    era: str | None,
    stratum: str | None,
    cohesion: float,
) -> None:
    """Generate town names. Replaces Rando's `bin/generator`."""
    # `available_tags()` already strips _INTERNAL_TAGS for the SPA dropdown,
    # but the CLI needs to ACCEPT internal tags (some scripts pass them
    # explicitly) — so widen the validation set with _INTERNAL_TAGS rather
    # than the literal subset, which would silently desync as new internal
    # markers (e.g. wyrd-yan's 'fiction') get added.
    known_tags = set(available_tags()) | _INTERNAL_TAGS
    bad = [t for t in tags if t not in known_tags]
    if bad:
        click.echo(f"Unknown tag(s): {', '.join(bad)}", err=True)
        click.echo("Available tags:", err=True)
        for t in sorted(known_tags - _INTERNAL_TAGS):
            click.echo(f"  {t}", err=True)
        sys.exit(1)
    bad_moods = [m for m in moods if m.split(":", 1)[0] not in MOODS]
    if bad_moods:
        click.echo(f"Unknown mood(s): {', '.join(bad_moods)}", err=True)
        click.echo(f"Available moods: {', '.join(sorted(MOODS))}", err=True)
        sys.exit(1)

    resolved = resolve_seed(seed)
    seed_rng = rng_for(resolved)
    kenning = Kenning()
    params = {
        "culture": culture,
        "tags": list(tags),
        "spelling_variety": spelling_variety,
        "novelty": novelty,
        "inflection_density": inflection_density,
        "mood": list(moods),
        "include_fiction": include_fiction,
        "era": era,
        "stratum": stratum,
        "cohesion": cohesion,
    }
    for _ in range(count):
        try:
            result = kenning.generate(params, seed_rng.randrange(2**63))
        except ValueError as exc:
            # Surface user-input errors (bad --era, etc.) as friendly
            # CLI messages on stderr + exit non-zero, matching the
            # tags/moods pre-validation pattern above. Other unexpected
            # exceptions still propagate so a real bug isn't silenced.
            click.echo(f"Error: {exc}", err=True)
            sys.exit(1)
        click.echo(result.result)
        if describe:
            click.echo(result.explanation)
    click.echo(f"(seed: {resolved})", err=True)


def _aggregate_unaccounted_fragments(
    resolved_names: list,
    *,
    min_length: int,
    max_examples: int,
) -> tuple[Counter, dict[str, list[str]], int]:
    """Walk decomposed Names and tally their leftover string slots.

    Per name, dedups fragments within that name (so 'ham' appearing
    twice in one toponym counts once toward frequency). Examples are
    capped at ``max_examples`` per fragment. Fragments shorter than
    ``min_length`` are skipped.

    Returns ``(fragments, examples_by_frag, imperfect_count)`` where
    ``imperfect_count`` is the number of names contributing at least
    one fragment (i.e. didn't reach zero unaccounted).
    """
    fragments: Counter = Counter()
    examples_by_frag: dict[str, list[str]] = {}
    imperfect_count = 0
    for resolved in resolved_names:
        if resolved.count_unaccounted() == 0:
            continue
        imperfect_count += 1
        seen_in_name: set[str] = set()
        for word_list in resolved.words.values():
            for word in word_list:
                for chunk in word.word:
                    if not isinstance(chunk, str):
                        continue
                    frag = chunk.lower()
                    if len(frag) < min_length or frag in seen_in_name:
                        continue
                    seen_in_name.add(frag)
                    fragments[frag] += 1
                    bucket = examples_by_frag.setdefault(frag, [])
                    if resolved.name not in bucket and len(bucket) < max_examples:
                        bucket.append(resolved.name)
    return fragments, examples_by_frag, imperfect_count


def _decompose_corpus(
    name_entries: list,
    word_db: dict,
    db_path: Path | None,
) -> tuple[list, Counter]:
    """Decompose a corpus of Names, honouring canonical picks when a
    lexicon DB path is supplied.

    ``name_entries`` is a list of ``(Name, region)`` pairs from
    ``load_names_with_regions`` (wyrd-8m87). Region is forwarded into
    the canonical lookup so same-modern_name-different-region
    collisions in the lexicon resolve to the right toponym row.
    Bare ``Name`` entries (legacy ``load_names`` callers) are accepted
    too — region defaults to ``None``, and the lookup fallback in
    ``decompose_with_canonical`` handles NULL-region DBs gracefully.

    Centralises (a) DB context-manager hygiene via ``nullcontext``,
    (b) the per-name canonical-vs-heuristic branch, (c) the
    ``canonical_hits`` Counter accumulation.

    Returns ``(resolved_names, canonical_hits)``. ``canonical_hits`` is
    empty when ``db_path`` is ``None`` so callers can use truthiness to
    decide whether to surface the ``canonical[...]`` summary block.
    """
    from wyrd.generators.kenning.decomposition import decompose_with_canonical
    from wyrd.generators.kenning.name import Name

    canonical_hits: Counter = Counter()
    resolved_names: list = []
    db_context = LexiconDB(db_path) if db_path is not None else nullcontext()
    with db_context as db:
        for entry in name_entries:
            if isinstance(entry, Name):
                name, region = entry, None
            else:
                name, region = entry[0], entry[1]
            if db is not None:
                resolved, source = decompose_with_canonical(name.name, word_db, db, region=region)
                canonical_hits[source or "heuristic"] += 1
            else:
                resolved = name
                resolved.find_meaning(word_db)
            resolved_names.append(resolved)
    return resolved_names, canonical_hits


@cli.command("rebuild-proportions")
@click.argument("culture", type=click.Choice(CULTURES))
@click.argument("place_names", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--meanings",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to meanings.json (defaults to bundled).",
)
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "wyrd-70s0: lexicon DB to read canonical decompositions from. "
        "When set, names with a canonical pick (from `lexicon decompose "
        "--apply`) use that breakdown; names without one fall back to "
        "the matcher's reduce=True heuristic. Without this flag the "
        "command runs heuristic-only (legacy behaviour)."
    ),
)
def rebuild_proportions(
    culture: str,
    place_names: Path,
    meanings: Path | None,
    db_path: Path | None,
) -> None:
    """Recompute a culture's proportions from a corpus + meanings DB.

    Replaces Rando's `bin/load_names`. Reads place_names JSON (the same shape as
    the bundled `<culture>_place_names.json` files), deconstructs each name
    against the meaning DB, and emits a fresh proportions JSON to stdout.

    With ``--db``, canonical decompositions (wyrd-08m / wyrd-70s0) drive
    the proportion counts for toponyms with a registered canonical pick;
    others fall back to the matcher heuristic. This makes counts stable
    across re-runs and reflects scholarly attribution where available.
    """
    meanings_data = _load_meanings_data(meanings)

    names_data = json.loads(place_names.read_text())
    name_entries = load_names_with_regions(names_data)
    word_db, _ = load_meanings(meanings_data)

    resolved_names, canonical_hits = _decompose_corpus(name_entries, word_db, db_path)
    good_names = [n for n in resolved_names if n.count_unaccounted() == 0]
    word_names = sum(1 for n in resolved_names if n.has_name())
    word_saints = sum(1 for n in resolved_names if n.has_saint())

    summary = (
        f"culture={culture} perfect={len(good_names)} names={word_names} "
        f"saints={word_saints} total={len(name_entries)}"
    )
    if canonical_hits:
        breakdown = " ".join(f"{src}={n}" for src, n in sorted(canonical_hits.items()))
        summary += f" canonical[{breakdown}]"
    click.echo(summary, err=True)
    proportions = _proportions_from(good_names)
    # wyrd-xmk3 review: sort_keys so the on-disk proportions JSON is
    # deterministic across re-runs. Insertion-order dicts produced
    # non-determinism in usages / single_usages / tag_cooccurrence
    # maps, polluting the diff in every rebuild even when counts
    # were identical. The structures array is a list of dicts; each
    # 'words' position is iterated by index so the array order is
    # already determined by the corpus walk — sort_keys handles the
    # dict slots inside.
    click.echo(json.dumps(proportions, sort_keys=True))


def _proportions_from(names) -> dict:
    part_proportions: Counter = Counter()
    lone_proportions: Counter = Counter()
    struct_proportions: Counter = Counter()
    tag_cooccurrence: Counter = Counter()
    tag_marginal: Counter = Counter()
    for name in names:
        for u in name.get_samples():
            part_proportions[u] += 1
        for u in name.get_lone_samples():
            lone_proportions[u] += 1
        for structure in name.get_structure():
            struct_proportions[structure] += 1
        # Tag co-occurrence: ordered consecutive Meaning pairs across the
        # full name (within-word AND between-word). Each pair contributes
        # the cartesian product of (left.tags × right.tags) — an etymon
        # often has several tags (plant, food, tree), and any of them can
        # legitimately drive co-occurrence.
        for left_tags, right_tags in _ordered_tag_pairs(name):
            for tl in left_tags:
                tag_marginal[tl] += 1
                for tr in right_tags:
                    tag_cooccurrence[(tl, tr)] += 1
            for tr in right_tags:
                tag_marginal[tr] += 1
    return {
        "usages": dict(part_proportions),
        "single_usages": dict(lone_proportions),
        "structures": _encode_structs(struct_proportions),
        # Serialize tag-pair keys as "left|right" so they survive a JSON
        # round-trip. Generator side splits on the pipe.
        "tag_cooccurrence": {f"{a}|{b}": v for (a, b), v in tag_cooccurrence.items()},
        "tag_marginal": dict(tag_marginal),
    }


def _ordered_tag_pairs(name) -> list[tuple[list[str], list[str]]]:
    """Walk a decomposed Name in order, yielding consecutive (left.tags,
    right.tags) pairs across the entire name (within and between words).

    For 'Bridgwater Gate' this yields:
        (Bridg-.tags, -water.tags)   # within word 1
        (-water.tags, Gate.tags)     # between words

    When a word has multiple equivalent decompositions (e.g. mining surfaced
    a bare-form synthesized usage that competes with the dashed reflex), prefer
    the decomp with the most tag-bearing Meanings — that's the one that
    actually feeds co-occurrence statistics rather than emitting empty
    () pairs.
    """

    def _tag_density(word_obj) -> int:
        if not hasattr(word_obj, "word"):
            return -1
        return sum(1 for c in word_obj.word if isinstance(c, Meaning) and c.tags)

    flat_meanings: list[Meaning] = []
    for word_text in name.name.split(" "):
        word_options = name.words.get(word_text, [])
        if not word_options:
            continue
        chosen = max(word_options, key=_tag_density)
        if not hasattr(chosen, "word"):
            continue
        for chunk in chosen.word:
            if isinstance(chunk, Meaning):
                flat_meanings.append(chunk)

    out: list[tuple[list[str], list[str]]] = []
    for i in range(len(flat_meanings) - 1):
        out.append((list(flat_meanings[i].tags), list(flat_meanings[i + 1].tags)))
    return out


def _encode_meaning(qualities) -> dict:
    out: dict = {}
    for quality in qualities:
        if quality in ("pre", "post", "inner"):
            out["location"] = quality
        else:
            out[quality] = True
    return out


def _encode_structs(struct: Counter) -> list:
    """Serialize the structure → count map into the proportions-JSON
    shape, sorted for deterministic output (wyrd-dxu2 review). Counter
    iteration order is insertion-order, so the underlying corpus walk
    determines initial order — but two re-runs against the same corpus
    can iterate in slightly different orders if the matcher's
    decomposition outputs change. Sorting by (-proportion, structure
    key) gives a stable canonical order: most-frequent first, with the
    structure tuple as deterministic tiebreaker."""
    # Sort Counter.items() once before serializing so we don't need
    # a temp _sort_key on each dict. Sort by (-proportion, key) for
    # most-frequent-first with the structure tuple as a deterministic
    # tiebreaker.
    sorted_items = sorted(struct.items(), key=lambda item: (-item[1], item[0]))
    return [
        {
            "proportion": value,
            "words": [[_encode_meaning(meaning) for meaning in word] for word in key],
        }
        for key, value in sorted_items
    ]


@cli.command("add-meaning")
@click.option("--tag", "tags", multiple=True, help="Modifier tag (repeatable).")
def add_meaning(tags: tuple[str, ...]) -> None:
    """Read names from stdin, emit JSON entries suitable for meanings.json.

    Replaces Rando's `bin/generate_names`. One name per line; '/'-separated
    forms become alternate `modern_usage` entries.
    """
    if not tags:
        click.echo("At least one --tag is required.", err=True)
        sys.exit(1)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        forms = line.split("/")
        entry = {
            "meaning": [f"{n} (Name)" for n in forms],
            "modifier_tags": list(tags),
            "modifier_type": "Habitative",
            "words": [{"modern_usage": f"{n}-"} for n in forms],
        }
        click.echo(json.dumps(entry) + ",")


@cli.command("unaccounted")
@click.argument("culture", type=click.Choice(CULTURES))
@click.argument("place_names", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--meanings",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to meanings.json (defaults to bundled).",
)
@click.option("--top", type=int, default=50, help="Show top N most-frequent fragments.")
@click.option("--examples", type=int, default=3, help="Up to N example names per fragment.")
@click.option("--min-length", type=int, default=2, help="Skip fragments shorter than this.")
@click.option("--as-json", is_flag=True, default=False, help="Emit machine-readable JSON.")
@click.option(
    "--sources-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help=(
        "wyrd-bvp: when set, annotate each top-N fragment with "
        "scholarly-corpus presence — count of source files where "
        "the fragment appears at a word boundary, plus 1-2 sample "
        "snippets with surrounding context. Strong-hit indicator "
        "flags snippets that look like an etymology body."
    ),
)
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "wyrd-70s0: lexicon DB to read canonical decompositions from. "
        "When set, gap counts honour the canonical pick — a fragment "
        "appears in the gap report only if it's unaccounted under the "
        "toponym's canonical decomposition. False-gap fragments that "
        "the heuristic surfaced but a zero-unaccounted alternate "
        "covered will silently disappear from the report."
    ),
)
def unaccounted(
    culture: str,
    place_names: Path,
    meanings: Path | None,
    top: int,
    examples: int,
    min_length: int,
    as_json: bool,
    sources_dir: Path | None,
    db_path: Path | None,
) -> None:
    """Surface morpheme fragments the corpus uses but the meaning DB doesn't.

    Decomposes every name in <place_names>; for names that don't fully decompose,
    aggregates the leftover string fragments by frequency. The top fragments are
    candidates for new entries in meanings.json — this is the mining tool that
    surfaces what the previous deconstruction silently dropped.

    With ``--db``, canonical decompositions (wyrd-08m / wyrd-70s0) drive
    the gap counting: a fragment is reported as unaccounted only if it
    appears in the toponym's canonical breakdown's leftovers. Names
    without a canonical fall back to the heuristic. Operators see fewer
    false-gap fragments at the cost of needing a recent
    ``lexicon decompose --apply`` run.
    """
    meanings_data = _load_meanings_data(meanings)

    names_data = json.loads(place_names.read_text())
    name_entries = load_names_with_regions(names_data)
    word_db, _ = load_meanings(meanings_data)

    resolved_names, canonical_hits = _decompose_corpus(name_entries, word_db, db_path)
    fragments, examples_by_frag, imperfect_count = _aggregate_unaccounted_fragments(
        resolved_names, min_length=min_length, max_examples=examples
    )

    top_fragments = fragments.most_common(top)
    # wyrd-bvp: when --sources-dir is set, scan all source-text
    # files for each top-N fragment and annotate with corpus
    # presence. The annotation is computed once and threaded
    # through both JSON and text output paths so consumers see the
    # same evidence shape regardless of format.
    evidence_by_frag: dict[str, dict[str, Any]] = {}
    if sources_dir is not None:
        evidence_by_frag = annotate_fragments_with_corpus_evidence(
            [frag for frag, _ in top_fragments],
            sources_dir,
        )

    summary = (
        f"culture={culture} imperfect_names={imperfect_count} unique_fragments={len(fragments)}"
    )
    if canonical_hits:
        breakdown = " ".join(f"{src}={n}" for src, n in sorted(canonical_hits.items()))
        summary += f" canonical[{breakdown}]"
    click.echo(summary, err=True)

    if as_json:
        out = []
        for frag, count in top_fragments:
            entry = {"fragment": frag, "count": count, "examples": examples_by_frag[frag]}
            if evidence_by_frag:
                entry.update(evidence_by_frag.get(frag, {}))
            out.append(entry)
        click.echo(json.dumps(out, indent=2))
        return

    if evidence_by_frag:
        click.echo(f"{'fragment':<20} {'count':>6}  {'corpus':>6}  {'strong':>6}  examples")
    else:
        click.echo(f"{'fragment':<20} {'count':>6}  examples")
    click.echo("-" * 70)
    for frag, count in top_fragments:
        ex = ", ".join(examples_by_frag[frag])
        if evidence_by_frag:
            ev = evidence_by_frag.get(frag, {})
            corpus = ev.get("corpus_hits", 0)
            strong = ev.get("strong_hits", 0)
            click.echo(f"{frag:<20} {count:>6}  {corpus:>6}  {strong:>6}  {ex}")
        else:
            click.echo(f"{frag:<20} {count:>6}  {ex}")


@cli.command("explain")
@click.argument("name")
def explain(name: str) -> None:
    """Decompose a name into morphemes; print every matching reading."""
    explainer = KenningExplain()
    results = explainer.generate_all({"name": name}, 0)
    for r in results:
        click.echo(r.explanation)


@cli.command("rewind")
@click.argument("name")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--era",
    "era_input",
    multiple=True,
    help=(
        "Era stop to render. Pass multiple times for a custom ladder; "
        "default is 'oe-late me modern' for the english family. "
        "Each value is a cell label (oe-late) or family/label "
        "(english/me). Year input not supported here — use cell labels "
        "for predictable rewind output."
    ),
)
def rewind(name: str, db_path: Path, era_input: tuple[str, ...]) -> None:
    """wyrd-rni Phase 3.1: render NAME at multiple historical eras.

    Decomposes the input via the kenning explainer, anchors each
    morpheme to its source-language etymon (typically Old English),
    and renders the compound at each requested era stop using the
    cognate-cluster era-reflex picker (wyrd-skm Phase 3.0b).

    Default ladder is three English-family stops: oe-late (800-1100),
    me (1100-1500), modern (1700+). Pass ``--era`` repeatedly to
    customise — e.g. ``--era oe-early --era oe-late --era me`` for a
    five-cell pre-modern ladder.

    Morphemes that don't anchor (no matching etymon row in the
    lexicon DB, or anchor lacks both cognate_id and descent edges)
    fall back to their modern usage and are flagged so the user can
    see which morphemes 'don't have era data' for the requested
    cell.
    """
    eras: tuple[tuple[str, str], ...]
    if era_input:
        try:
            eras = tuple(era_cell_for_input(value, default_family="english") for value in era_input)
        except ValueError as exc:
            click.echo(str(exc), err=True)
            raise click.exceptions.Exit(1) from exc
    else:
        eras = DEFAULT_ENGLISH_ERAS

    meaning_db, _ = _load_meanings()

    try:
        with LexiconDB(db_path) as db:
            result = rewind_name(name, db, meaning_db, eras=eras)
    except ValueError as exc:
        click.echo(f"Error: {exc}", err=True)
        raise click.exceptions.Exit(1) from exc

    if not result.eras:
        click.echo("(no era stops requested)", err=True)
        return

    click.echo(f"{result.name} — decomposed: {result.decomposition}")
    if result.unaccounted:
        click.echo(f"  unaccounted: {' '.join(result.unaccounted)}", err=True)
    # Bail early when nothing decomposed — printing empty era rows
    # ('english/oe-late  ') with no rendered content clutters the
    # output and confuses readers about what failed.
    if not any(stop.morphemes for stop in result.eras):
        return

    width = max(len(f"{stop.family}/{stop.cell}") for stop in result.eras)
    for stop in result.eras:
        label = f"{stop.family}/{stop.cell}".ljust(width)
        flag = " *" if any(m.fallback for m in stop.morphemes) else ""
        click.echo(f"  {label}  {stop.rendered}{flag}")
    if any(any(m.fallback for m in stop.morphemes) for stop in result.eras):
        click.echo(
            "  * one or more morphemes lacked era data; rendered with "
            "the modern usage as fallback.",
            err=True,
        )


@cli.command("era-map")
@click.argument("culture", type=click.Choice(CULTURES))
@click.option("--count", type=int, default=6, help="How many names to roll.")
@click.option("--seed", type=int, default=0, help="RNG seed; successive names use seed+i.")
@click.option("--as-json", is_flag=True, default=False, help="Emit JSON.")
def era_map(culture: str, count: int, seed: int, as_json: bool) -> None:
    """wyrd-381: bulk-generate N names AND render each at era stops.

    The Domesday-vs-modern map demo. Composes the existing Kenning
    name-generator with the wyrd-skm era-reflex rewinder: same roll,
    same morpheme stack, multiple eras of paper.

    Default ladder is the three English stops (oe-late / me / modern).
    Names whose morphemes haven't been mined for era_reflexes render
    uniformly across all eras (the bundle-only renderer falls back
    to the modern usage); that's a coverage limit, not a bug.
    """
    from wyrd.generators.kenning import KenningEraMap

    gen = KenningEraMap()
    results = gen.generate_all({"culture": culture, "count": count}, seed)

    if as_json:
        out = []
        for r in results:
            comp = r.components[0] if r.components else {}
            out.append(
                {
                    "name": comp.get("name"),
                    "era_cells": comp.get("era_cells", []),
                    "rendered_modern": r.result,
                }
            )
        click.echo(json.dumps(out, indent=2))
        return

    click.echo(f"culture={culture} count={len(results)} (requested {count})", err=True)
    for r in results:
        comp = r.components[0] if r.components else {}
        name = comp.get("name", "?")
        cells = comp.get("era_cells", [])
        click.echo(f"\n{name}")
        for cell in cells:
            label = f"{cell.get('family', '?')}/{cell.get('era', '?')}"
            click.echo(f"  {label:<20} {cell.get('rendered', '?')}")


@cli.command("creature")
@click.argument("name")
def creature(name: str) -> None:
    """wyrd-vz7f: surface fantasy-creature etymology from the
    wyrd-ami pipeline.

    Looks up the creature name in the bundled ``fantasy_morphemes``
    map (populated by ``lexicon export-meanings`` from the
    ``fantasy_morpheme`` table). Prints the linked attested ancestor
    (language, canonical form, glosses, citation) and any era
    reflexes mined for that etymon's cluster.

    Unknown names return a polite 'not found' rather than erroring,
    so chaining ``wyrd kenning creature <random>`` against a list
    is safe.
    """
    from wyrd.generators.kenning import KenningCreature

    gen = KenningCreature()
    result = gen.generate({"name": name}, seed=0)
    click.echo(result.result)
    if result.explanation:
        click.echo(f"  {result.explanation}")


@cli.group("lexicon")
def lexicon() -> None:
    """Manage the authoring lexicon DB (etymology data store)."""


@lexicon.command("build")
@click.option(
    "--db",
    "db_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
    help="Where to write the SQLite DB.",
)
@click.option(
    "--meanings",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to meanings.json (defaults to bundled).",
)
def lexicon_build(db_path: Path, meanings: Path | None) -> None:
    """Initialize the lexicon DB and seed it from meanings.json.

    Wipes any existing DB at the path. The seed data is attributed to the
    synthetic 'rando-port' source.
    """
    meanings_data = _load_meanings_data(meanings)

    init_schema(db_path)
    with LexiconDB(db_path) as db:
        db.upsert_source(**_RANDO_SOURCE)
        db.commit()
        counts = seed_from_meanings(db, meanings_data, _RANDO_SOURCE["id"])
        stats = db.stats()

    click.echo(f"Built {db_path}", err=True)
    click.echo(f"Seeded as source={_RANDO_SOURCE['id']}: {counts}", err=True)
    click.echo("Tables:", err=True)
    for table, n in stats.items():
        click.echo(f"  {table:<30} {n:>8}", err=True)


# Known public-domain volumes. Used to register a `source` row automatically
# when mining one of these files. Add more as they're targeted.
_KNOWN_SKEAT_BOOKS = {
    "skeat_1901_cambridgeshire": {
        "title": "The Place-Names of Cambridgeshire",
        "author": "Walter W. Skeat",
        "year": 1901,
        "region": "Cambridgeshire",
        "language_focus": "old-english",
    },
    "skeat_1906_bedfordshire": {
        "title": "The Place-Names of Bedfordshire",
        "author": "Walter W. Skeat",
        "year": 1906,
        "region": "Bedfordshire",
        "language_focus": "old-english",
    },
    "skeat_1911_berkshire": {
        "title": "The Place-Names of Berkshire",
        "author": "Walter W. Skeat",
        "year": 1911,
        "region": "Berkshire",
        "language_focus": "old-english",
    },
    "skeat_1913_suffolk": {
        "title": "The Place-Names of Suffolk",
        "author": "Walter W. Skeat",
        "year": 1913,
        "region": "Suffolk",
        "language_focus": "old-english",
    },
    "mawer_1920_northumberland_durham": {
        "title": "The Place-Names of Northumberland and Durham",
        "author": "Allen Mawer",
        "year": 1920,
        "region": "Northumberland and Durham",
        "language_focus": "old-english",
    },
    "mawer_1922_place_names_and_history": {
        "title": "Place-Names and History",
        "author": "Allen Mawer",
        "year": 1922,
        "region": "England",
        "language_focus": "old-english",
    },
    "ekwall_1922_lancashire": {
        "title": "The Place-Names of Lancashire",
        "author": "Eilert Ekwall",
        "year": 1922,
        "region": "Lancashire",
        "language_focus": "old-english",
    },
    "wyld_hirst_1911_lancashire": {
        "title": "The Place-Names of Lancashire: Their Origin and History",
        "author": "H.C. Wyld and T.O. Hirst",
        "year": 1911,
        "region": "Lancashire",
        "language_focus": "old-english",
    },
    "moorman_1910_west_riding_yorkshire": {
        "title": "Place-Names of the West Riding of Yorkshire",
        "author": "F.W. Moorman",
        "year": 1910,
        "region": "West Riding of Yorkshire",
        "language_focus": "old-english",
    },
    "goodall_1914_sw_yorkshire": {
        "title": "Place-Names of South-West Yorkshire",
        "author": "Armitage Goodall",
        "year": 1914,
        "region": "South-West Yorkshire",
        "language_focus": "old-english",
    },
    "duignan_1902_staffordshire": {
        "title": "Notes on Staffordshire Place-Names",
        "author": "W.H. Duignan",
        "year": 1902,
        "region": "Staffordshire",
        "language_focus": "old-english",
    },
    "harrison_1898_liverpool": {
        "title": "The Place-Names of the Liverpool District",
        "author": "Henry Harrison",
        "year": 1898,
        "region": "Lancashire",
        "language_focus": "old-english",
    },
    "johnston_1892_place_names_of_scotland": {
        "title": "Place-Names of Scotland",
        "author": "James B. Johnston",
        "year": 1892,
        "region": "Scotland",
        "language_focus": "celtic",
    },
    "johnston_1904_stirlingshire": {
        "title": "The Place-Names of Stirlingshire",
        "author": "James B. Johnston",
        "year": 1904,
        "region": "Stirlingshire",
        "language_focus": "celtic",
    },
    "johnston_1915_place_names_of_england_and_wales": {
        "title": "The Place-Names of England and Wales",
        "author": "James B. Johnston",
        "year": 1915,
        "region": "England and Wales",
        "language_focus": "old-english",
    },
    "watson_1904_ross_and_cromarty": {
        "title": "Place-Names of Ross and Cromarty",
        "author": "William J. Watson",
        "year": 1904,
        "region": "Ross and Cromarty",
        "language_focus": "celtic",
    },
    "watson_1926_celtic_place_names_of_scotland": {
        "title": "The History of the Celtic Place-Names of Scotland",
        "author": "William J. Watson",
        "year": 1926,
        "region": "Scotland",
        "language_focus": "celtic",
    },
    "macbain_1922_highlands_and_islands": {
        "title": "Place Names, Highlands and Islands of Scotland",
        "author": "Alexander Macbain",
        "year": 1922,
        "region": "Scottish Highlands and Islands",
        "language_focus": "celtic",
    },
    "gillies_1906_argyll": {
        "title": "The Place-Names of Argyll",
        "author": "H.C. Gillies",
        "year": 1906,
        "region": "Argyll",
        "language_focus": "celtic",
    },
    "morgan_1887_wales_monmouthshire": {
        "title": "Handbook of the Origin of Place-Names in Wales and Monmouthshire",
        "author": "Thomas Morgan",
        "year": 1887,
        "region": "Wales and Monmouthshire",
        "language_focus": "celtic",
    },
    "morgan_1912_wales": {
        "title": "The Place-Names of Wales",
        "author": "Thomas Morgan",
        "year": 1912,
        "region": "Wales",
        "language_focus": "celtic",
    },
    "joyce_1875_irish_names_vol1": {
        "title": "The Origin and History of Irish Names of Places, vol. 1",
        "author": "P.W. Joyce",
        "year": 1875,
        "region": "Ireland",
        "language_focus": "celtic",
    },
    "joyce_1898_irish_names_vol3": {
        "title": "The Origin and History of Irish Names of Places, vol. 3",
        "author": "P.W. Joyce",
        "year": 1898,
        "region": "Ireland",
        "language_focus": "celtic",
    },
    "joyce_1913_irish_names_of_places": {
        "title": "Irish Names of Places",
        "author": "P.W. Joyce",
        "year": 1913,
        "region": "Ireland",
        "language_focus": "celtic",
    },
    "moore_1890_isle_of_man": {
        "title": "The Surnames and Place-Names of the Isle of Man",
        "author": "A.W. Moore",
        "year": 1890,
        "region": "Isle of Man",
        "language_focus": "celtic",
    },
    "mcclure_1910_british_place_names": {
        "title": "British Place-Names in their Historical Setting",
        "author": "Edmund McClure",
        "year": 1910,
        "region": "British Isles",
        "language_focus": "old-english",
    },
    "taylor_1893_words_and_places": {
        "title": "Words and Places",
        "author": "Isaac Taylor",
        "year": 1893,
        "region": "Europe",
        "language_focus": "mixed",
    },
    "lindkvist_1912_scandinavian_place_names": {
        "title": "Middle-English Place-Names of Scandinavian Origin",
        "author": "Harald Lindkvist",
        "year": 1912,
        "region": "England (Danelaw)",
        "language_focus": "old-norse",
    },
    "zachrisson_1909_anglo_norman_influence": {
        "title": "A Contribution to the Study of Anglo-Norman Influence on English Place-Names",
        "author": "R.E. Zachrisson",
        "year": 1909,
        "region": "England",
        "language_focus": "norman-french",
    },
    "mawer_stenton_1924_introduction_to_survey": {
        "title": "Introduction to the Survey of English Place-Names",
        "author": "Allen Mawer and F.M. Stenton",
        "year": 1924,
        "region": "England",
        "language_focus": "old-english",
    },
    "mawer_1924_chief_elements": {
        "title": "The Chief Elements Used in English Place-Names",
        "author": "Allen Mawer",
        "year": 1924,
        "region": "England",
        "language_focus": "old-english",
    },
    "bannister_1916_herefordshire": {
        "title": "The Place-Names of Herefordshire",
        "author": "A.T. Bannister",
        "year": 1916,
        "region": "Herefordshire",
        "language_focus": "norman-french",
    },
    "quilgars_1906_loire_inferieure": {
        "title": "Dictionnaire topographique du département de la Loire-Inférieure",
        "author": "Henri Quilgars",
        "year": 1906,
        "region": "Loire-Inférieure (Brittany)",
        "language_focus": "celtic",
    },
}


@lexicon.command("export-meanings")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write to this path instead of stdout.",
)
@click.option(
    "--min-witnesses",
    type=int,
    default=3,
    show_default=True,
    help="Promote etymons with this many distinct scholar witnesses (D4). "
    "Used as the fallback for languages not covered by --lang-threshold or the preset.",
)
@click.option(
    "--lang-threshold",
    "lang_threshold_specs",
    multiple=True,
    metavar="LANG=N",
    help="Per-language witness threshold override (repeatable). Example: "
    "--lang-threshold old-norse=2. Merges with the recommended preset; pass "
    "--no-preset to start from an empty map (then --min-witnesses applies uniformly).",
)
@click.option(
    "--preset/--no-preset",
    "use_preset",
    default=True,
    show_default=True,
    help="Start from RECOMMENDED_LANG_THRESHOLDS (calibrated against corpus availability). "
    "--no-preset uses --min-witnesses uniformly across all languages.",
)
@click.option(
    "--include-rando/--no-include-rando",
    default=True,
    show_default=True,
    help="Include rando-port-cited families regardless of witness count (D4 legacy).",
)
@click.option(
    "--include-wiktionary-empirical/--no-include-wiktionary-empirical",
    default=True,
    show_default=True,
    help="Include 'wiktionary-empirical'-cited families regardless of witness "
    "count (wyrd-4hx7 corpus-mined gap-fills, treated like rando-port).",
)
@click.option(
    "--joiners-from",
    "joiners_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Optional path to a joiners-pool sidecar JSON (wyrd-q0g6 Phase 2). "
        'Shape: {lang_field: [{"form": str, "weight": int}, ...], ...}. '
        "When provided + non-empty, the bundle's ``joiners`` key is "
        "populated; the runtime joiner_density knob then has data to act on."
    ),
)
def lexicon_export_meanings(
    db_path: Path,
    output_path: Path | None,
    min_witnesses: int,
    lang_threshold_specs: tuple[str, ...],
    use_preset: bool,
    include_rando: bool,
    include_wiktionary_empirical: bool,
    joiners_path: Path | None,
) -> None:
    """Export the lexicon DB as a meanings.json document for the runtime.

    Inverse of ``lexicon build``: walks the authoring DB, applies the D4
    promotion rule, and emits the JSON shape the kenning generator expects.
    Inflected variants (D8) and OCR-cluster losers (D22) ride along in their
    lemma's language form list rather than appearing as separate subjects.

    Per-language witness thresholds are calibrated against corpus availability:
    well-mined languages (old-english) gate at 3 witnesses to keep Tier-1
    extraction noise out, while corpus-thin languages (old-norse, latin) gate
    at 2. See RECOMMENDED_LANG_THRESHOLDS in lexicon.py for rationale.
    """
    lang_thresholds: dict[str, int] = dict(RECOMMENDED_LANG_THRESHOLDS) if use_preset else {}
    for spec in lang_threshold_specs:
        if "=" not in spec:
            raise click.BadParameter(f"--lang-threshold expects LANG=N, got {spec!r}")
        lang, _, n_str = spec.partition("=")
        lang = lang.strip()
        n_str = n_str.strip()
        if not lang or not n_str:
            raise click.BadParameter(
                f"--lang-threshold {spec!r}: both LANG and N must be non-empty"
            )
        try:
            n = int(n_str)
        except ValueError as exc:
            raise click.BadParameter(f"--lang-threshold {spec!r}: N must be an integer") from exc
        # Accept JSON-field aliases ('old_english', 'celtic_mix') and map them to
        # the lexicon's canonical codes ('old-english', 'celtic'). Without this,
        # an override like --lang-threshold old_english=2 silently fails to match
        # any consensus row and the preset fallback applies instead.
        lang = LANGUAGE_FIELDS.get(lang, lang)
        lang_thresholds[lang] = n
    with LexiconDB(db_path) as db:
        subjects = export_meanings(
            db,
            min_witnesses=min_witnesses,
            lang_thresholds=lang_thresholds,
            include_rando=include_rando,
            include_wiktionary_empirical=include_wiktionary_empirical,
        )
        canonical_decompositions = collect_canonical_decompositions(db)
        fantasy_morphemes = collect_fantasy_morphemes(db)

    joiners: dict[str, list[dict[str, Any]]] = {}
    if joiners_path is not None:
        joiners = _load_joiners_sidecar(joiners_path)

    # wyrd-c1vq: bundle is always dict-shape going forward. The
    # runtime loaders (load_meanings, load_canonical_decompositions,
    # load_joiners, load_fantasy_morphemes) tolerate both list-shape
    # (legacy) and dict-shape input, so flipping the default doesn't
    # break consumers; future field additions just become new keys
    # instead of re-deciding the shape each time. Optional fields
    # only appear when populated so empty-dict noise stays out of the
    # rendered JSON.
    bundle: dict[str, Any] = {"subjects": subjects}
    if canonical_decompositions:
        bundle["canonical_decompositions"] = canonical_decompositions
    if joiners:
        bundle["joiners"] = joiners
    if fantasy_morphemes:
        bundle["fantasy_morphemes"] = fantasy_morphemes

    payload = json.dumps(bundle, ensure_ascii=False, indent=2)
    if output_path is None:
        click.echo(payload)
    else:
        output_path.write_text(payload + "\n")
        notes = []
        if canonical_decompositions:
            notes.append(f"{len(canonical_decompositions)} canonical decompositions")
        if joiners:
            joiner_total = sum(len(entries) for entries in joiners.values())
            notes.append(f"{joiner_total} joiners across {len(joiners)} fields")
        if fantasy_morphemes:
            notes.append(f"{len(fantasy_morphemes)} fantasy morphemes")
        suffix = " + " + " + ".join(notes) if notes else ""
        click.echo(
            f"Wrote {len(subjects)} subjects{suffix} to {output_path}",
            err=True,
        )


def _load_joiners_sidecar(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Read + validate a joiners-pool sidecar JSON.

    Expected shape: ``{lang_field: [{"form": str, "weight": int}, ...], ...}``.
    Empty dict returned if the file is empty or shaped wrong; specific
    issues raised as ``click.ClickException`` so the operator gets a
    pointed error rather than a JSON parse traceback."""
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"--joiners-from {path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise click.ClickException(
            f"--joiners-from {path}: expected top-level dict keyed by lang_field"
        )
    out: dict[str, list[dict[str, Any]]] = {}
    for lang_field, entries in data.items():
        if not isinstance(entries, list):
            raise click.ClickException(
                f"--joiners-from {path}: entries for {lang_field!r} must be a list"
            )
        validated: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict) or "form" not in entry or "weight" not in entry:
                raise click.ClickException(
                    f"--joiners-from {path}: each entry under {lang_field!r} must "
                    "be a dict with 'form' and 'weight' keys"
                )
            validated.append({"form": entry["form"], "weight": entry["weight"]})
        if validated:
            out[lang_field] = validated
    return out


@lexicon.command("mine-skeat")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--source-id",
    default=None,
    help="Override the source row id (defaults to filename stem).",
)
def lexicon_mine_skeat(path: Path, db_path: Path, source_id: str | None) -> None:
    """Mine a Skeat-format etymology text and ingest into the lexicon.

    Resolves a `source` row from the filename stem (or --source-id), parses
    the text with the Skeat parser, and writes toponym + etymology rows
    attributed to that source. Existing data is left in place; this command
    is additive — re-runs will accumulate citations.
    """
    derived_id = source_id or path.stem
    metadata = _KNOWN_SKEAT_BOOKS.get(derived_id)
    if metadata is None:
        # Allow ingesting unknown files but warn — caller should add to the
        # known-books table or pass --source-id pointing at an existing row.
        metadata = {
            "title": derived_id.replace("_", " ").title(),
            "author": "unknown",
            "year": None,
            "region": None,
            "language_focus": None,
        }
        click.echo(f"warn: no metadata for source_id={derived_id!r}; using stub", err=True)

    text = path.read_text()
    entries = parse_skeat_text(text)

    by_conf: dict[str, int] = {}
    for e in entries:
        by_conf[e.confidence] = by_conf.get(e.confidence, 0) + 1

    with LexiconDB(db_path) as db:
        db.upsert_source(id=derived_id, **metadata)
        db.commit()
        counts = ingest_parsed_entries(db, entries, derived_id, region=metadata.get("region"))

    click.echo(f"Source: {derived_id}", err=True)
    click.echo(f"Parsed entries: {len(entries)} (by confidence: {by_conf})", err=True)
    click.echo(f"Ingested: {counts}", err=True)


def _mine_entries(
    *,
    parsed: list,
    client: Any,
    extract_one: Any,
    concurrency: int,
    progress_start: float,
    log_every: int = 10,
) -> tuple[list, int, int, dict[str, int]]:
    """Run ``extract_one`` over every parsed entry, optionally in parallel.

    wyrd-l0r: serial loop (concurrency=1) is bit-stable with the pre-PR
    flow. concurrency>1 fans out via ThreadPoolExecutor so paid-provider
    network latency stops dominating mining wall-clock time. Per-entry
    validation (D3) runs inside extract_one and is preserved unchanged
    on both paths — there's no batched validation here.

    Accepted entries come back in INPUT (parsed) order on both paths so
    downstream ingestion order is deterministic with respect to the
    source document, regardless of completion order.
    """
    accepted_slot: list = [None] * len(parsed)
    declined = 0
    rejected = 0
    by_failure: dict[str, int] = {}

    def _process(i: int, result) -> None:
        nonlocal declined, rejected
        if result.accepted and result.entry and result.entry.elements:
            accepted_slot[i] = result.entry
        elif result.accepted:
            declined += 1
        else:
            rejected += 1
            for f in result.failures:
                by_failure[f.reason] = by_failure.get(f.reason, 0) + 1

    def _emit_progress(completed: int) -> None:
        # Caller must increment `completed` BEFORE calling so it's >= 1
        # — both branches below uphold this. The elapsed/completed
        # division would otherwise ZeroDivisionError on the first call.
        if completed % log_every:
            return
        elapsed = time.time() - progress_start
        accepted_count = sum(1 for x in accepted_slot if x is not None)
        click.echo(
            f"  [{completed}/{len(parsed)}]  "
            f"accepted={accepted_count} declined={declined} rejected={rejected} "
            f"({elapsed / completed:.1f}s/entry)",
            err=True,
        )

    if concurrency <= 1:
        # Serial path — bit-stable with the pre-wyrd-l0r mining loop.
        for i, p in enumerate(parsed):
            result = extract_one(
                client,
                toponym=p.toponym,
                body=p.body_text,
                suffix_hint=p.section_suffix,
            )
            _process(i, result)
            _emit_progress(i + 1)
    else:
        # Parallel path — fan out via ThreadPoolExecutor. The state
        # mutation in _process / _emit_progress runs only on the main
        # thread because as_completed yields results back here serially,
        # so no per-counter lock is needed. Worker threads only call
        # extract_one (which is HTTP I/O, GIL-friendly).

        def _task(i: int):
            p = parsed[i]
            return i, extract_one(
                client,
                toponym=p.toponym,
                body=p.body_text,
                suffix_hint=p.section_suffix,
            )

        completed_count = 0
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(_task, i) for i in range(len(parsed))]
            try:
                for fut in as_completed(futures):
                    i, result = fut.result()
                    _process(i, result)
                    completed_count += 1
                    _emit_progress(completed_count)
            except BaseException:
                # ThreadPoolExecutor.__exit__ would otherwise drain every
                # remaining submitted future before the exception surfaces
                # (shutdown(wait=True, cancel_futures=False) is the default).
                # On a real 5xx burst that means ~N-1 extra paid-API calls
                # fire after the first failure — meaningful cost asymmetry on
                # Anthropic / Gemini. Cancel pending futures eagerly so we
                # abort at the first failure rather than after the full batch.
                executor.shutdown(wait=False, cancel_futures=True)
                raise

    accepted_entries = [e for e in accepted_slot if e is not None]
    return accepted_entries, declined, rejected, by_failure


@lexicon.command("mine-llm")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--source-id",
    default=None,
    help="Override the source row id (defaults to filename stem).",
)
@click.option(
    "--provider",
    type=click.Choice(["ollama", "gemini", "anthropic"]),
    default="ollama",
    show_default=True,
    help="LLM backend to use.",
)
@click.option(
    "--ollama-url",
    default=None,
    help="Override Ollama base URL (default: $WYRD_OLLAMA_URL or localhost).",
)
@click.option(
    "--model",
    default=None,
    help=(
        "Override the model. "
        "Ollama default: $WYRD_OLLAMA_MODEL or qwen3.5:35b-a3b. "
        "Gemini default: $WYRD_GEMINI_MODEL or gemini-2.5-flash. "
        "Anthropic default: $WYRD_ANTHROPIC_MODEL or claude-sonnet-4-6."
    ),
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Process only the first N entries (smoke testing).",
)
@click.option(
    "--timeout",
    type=float,
    default=180.0,
    show_default=True,
    help="Per-entry HTTP timeout in seconds.",
)
@click.option(
    "--parser",
    type=click.Choice(["auto", "skeat", "alphabetical", "numbered-list"]),
    default="auto",
    show_default=True,
    help=(
        "Which parser to use. 'auto' picks based on first-pass yield. "
        "'numbered-list' (wyrd-5af) is opt-in only — for ordinal-prefixed "
        "treatise content (Longnon vol 2 saint-names section, etc.)."
    ),
)
@click.option(
    "--min-entry",
    type=int,
    default=1,
    show_default=True,
    help=(
        "Numbered-list only: skip ordinals below this value. Use to drop "
        "front-matter / TOC noise when the actual numbered section opens "
        "mid-document (e.g. Longnon vol 2 saint-names start at 1659)."
    ),
)
@click.option(
    "--max-entry",
    type=int,
    default=None,
    help=(
        "Numbered-list only: skip ordinals above this value. Use to cap "
        "at the section boundary so index-appendix cross-references don't "
        "cost API spend on guaranteed declines (e.g. Longnon vol 2 "
        "saint-names end at 2138)."
    ),
)
@click.option(
    "--concurrency",
    type=click.IntRange(1, 64),
    default=1,
    show_default=True,
    help=(
        "Number of parallel extract_one calls (wyrd-l0r). 1 = serial "
        "(bit-stable with the pre-PR mining loop). For paid providers "
        "(Anthropic, Gemini), bump to 4-8 to amortize network latency; "
        "watch your provider's RPM cap. Ollama on a single-GPU host "
        "serializes at the model layer, so concurrency >1 buys nothing "
        "there."
    ),
)
@click.option(
    "--declines-only",
    is_flag=True,
    default=False,
    help=(
        "wyrd-bgr: skip parsed entries whose toponym already has a "
        "toponym_etymology row from any earlier Tier-1 mining run on "
        "this source. Lets you cheaply re-mine just the gaps with a "
        "stronger model — e.g. after Qwen Tier-1 leaves declines, run "
        "`mine-llm --declines-only --provider anthropic` to fill them "
        "in with Haiku without re-paying for already-extracted "
        "toponyms. Filter applies before --limit."
    ),
)
def lexicon_mine_llm(
    path: Path,
    db_path: Path,
    source_id: str | None,
    provider: str,
    ollama_url: str | None,
    model: str | None,
    limit: int | None,
    timeout: float,
    parser: str,
    concurrency: int,
    min_entry: int,
    max_entry: int | None,
    declines_only: bool,
) -> None:
    """Mine an etymology text via LLM (Ollama or Gemini).

    Uses the regex parser to segment entries, then feeds each entry body to
    the chosen LLM with a strict JSON schema. Every extracted form is
    validated against the source text — hallucinated etymons are rejected,
    not ingested. Successful extractions are tagged in `notes` with the
    provider+model so we can supersede them later with a stronger model.
    """
    _warn_if_paid_concurrency_too_high(provider, concurrency)
    derived_id = source_id or path.stem
    metadata = _KNOWN_SKEAT_BOOKS.get(
        derived_id,
        {
            "title": derived_id.replace("_", " ").title(),
            "author": "unknown",
            "year": None,
            "region": None,
            "language_focus": None,
        },
    )

    text = path.read_text()
    parsed = _select_parser_and_run(
        text, parser, min_entry_number=min_entry, max_entry_number=max_entry
    )
    if declines_only:
        already = _select_already_extracted_toponyms(db_path, derived_id)
        before = len(parsed)
        parsed = [p for p in parsed if p.toponym not in already]
        click.echo(
            f"--declines-only: filtered out {before - len(parsed)} already-extracted "
            f"toponyms (source already has {len(already)} rows). "
            f"{len(parsed)} entries remain.",
            err=True,
        )
        if not parsed:
            click.echo("Nothing to mine after declines filter.", err=True)
            return
    if limit is not None:
        parsed = parsed[:limit]

    client, extract_one = _build_llm_client(
        provider, model=model, ollama_url=ollama_url, timeout=timeout
    )

    click.echo(
        f"Mining {len(parsed)} entries from {path.name} via {client.base_url} model={client.model}",
        err=True,
    )

    started_at = datetime.now(UTC).isoformat()
    start = time.time()
    accepted_entries, declined, rejected, by_failure = _mine_entries(
        parsed=parsed,
        client=client,
        extract_one=extract_one,
        concurrency=concurrency,
        progress_start=start,
    )
    elapsed = time.time() - start
    completed_at = datetime.now(UTC).isoformat()
    click.echo(
        f"Done in {elapsed:.0f}s ({elapsed / max(len(parsed), 1):.1f}s/entry). "
        f"accepted={len(accepted_entries)} declined={declined} rejected={rejected}",
        err=True,
    )
    if by_failure:
        click.echo(f"Rejection reasons: {by_failure}", err=True)

    with LexiconDB(db_path) as db:
        db.upsert_source(id=derived_id, **metadata)
        db.commit()
        counts = ingest_parsed_entries(
            db, accepted_entries, derived_id, region=metadata.get("region")
        )
        # Persist the run summary (D23: stop losing accept/decline/reject
        # counts to stdout). Idempotent on (book, provider, model, mode,
        # completed_at) so re-running with --apply won't dupe.
        record_mining_run(
            db,
            source_id=derived_id,
            provider=provider,
            model=client.model,
            mode="mine",
            parsed_count=len(parsed),
            accepted=len(accepted_entries),
            declined=declined,
            rejected=rejected,
            by_failure=by_failure or None,
            started_at=started_at,
            completed_at=completed_at,
        )

    click.echo(f"Ingested: {counts}", err=True)


@lexicon.command("mine-fantasy-name")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option("--name", default=None, help="Single creature/morpheme name to research.")
@click.option(
    "--description",
    default=None,
    help="Paragraph describing the creature (passed to the LLM when pre-filter misses).",
)
@click.option(
    "--batch",
    "batch_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        'JSONL file with one {"name": ..., "description": ...} per line. '
        "Mutually exclusive with --name/--description. Resumes correctly: "
        "re-running with the same approach_version updates rows in place."
    ),
)
@click.option(
    "--skip-llm",
    is_flag=True,
    default=False,
    help=(
        "Run only the descent-walking pre-filter; bar pre-filter misses "
        "instead of calling the LLM. Useful for offline tests, smokes, and "
        "estimating how many inputs the pre-filter alone covers."
    ),
)
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    default=False,
    help="Write fantasy_morpheme rows + add 'fantasy' tag on usable etymons. Without this, dry-run.",
)
@click.option(
    "--skip-resolved",
    is_flag=True,
    default=False,
    help=(
        "Skip inputs whose name already has a fantasy_morpheme row for "
        "the current approach_version. Lets you grow the input corpus "
        "(e.g. extractor v2) and re-run without paying for already-"
        "resolved entries. Looks up the existing names once at startup. "
        "Comparison is case-insensitive (matches the table's "
        "COLLATE NOCASE UNIQUE constraint)."
    ),
)
@click.option(
    "--model",
    default=None,
    help=(
        "Override the LLM model used for both the semantic-check pass on "
        "pre-filter resolutions and the full-research fallback. Gemini "
        "default: gemini-2.5-flash. Pass a different Gemini model name "
        "(gemini-2.5-pro, gemini-2.5-flash-lite, etc.) to substitute it."
    ),
)
def lexicon_mine_fantasy_name(
    db_path: Path,
    name: str | None,
    description: str | None,
    batch_path: Path | None,
    skip_llm: bool,
    apply_changes: bool,
    skip_resolved: bool,
    model: str | None,
) -> None:
    """Research the etymology of a fantasy/gaming creature name (wyrd-ami).

    Each input (name + paragraph description) is routed through:
    1. A descent-walking lookup against the wiktionary-derived etymon
       corpus. Resolves common inputs without an LLM call (harpy →
       ancient-greek ἅρπυια, troll → ON trǫll, etc.).
    2. A Gemini-Flash full-research call when the pre-filter misses.

    Every input produces ONE row in `fantasy_morpheme`: usable=1 with
    `etymon_id` set when we resolved to an attested etymon; usable=0
    with `bar_reason` (e.g. modern_coinage for Dunsany's "gnoll") when
    we couldn't. Conservative-by-default: low-confidence LLM answers
    get barred as `uncertain_attestation`.

    Single mode: `--name X --description Y`
    Batch mode:  `--batch path/to/names.jsonl`
    """
    fp = fantasy_pipeline  # local alias matches the helper-style usage below

    if batch_path and (name or description):
        raise click.ClickException("--batch is mutually exclusive with --name/--description")
    if not batch_path and not (name and description):
        raise click.ClickException("provide either --batch or both --name and --description")

    inputs: list[tuple[str, str]] = []
    if batch_path:
        with batch_path.open() as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    raise click.ClickException(f"line {i}: invalid JSON: {e}") from e
                if "name" not in rec or "description" not in rec:
                    raise click.ClickException(f"line {i}: missing 'name' or 'description'")
                inputs.append((rec["name"], rec["description"]))
    else:
        inputs.append((name, description))  # type: ignore[arg-type]

    skipped_count = 0
    if skip_resolved:
        # Lowercase-fold both sides — fantasy_morpheme.input_name uses
        # COLLATE NOCASE, so 'Harpy' and 'harpy' are the same row.
        # Without folding, varying input casing would re-trigger LLM
        # calls even though the upsert would later collapse them.
        with LexiconDB(db_path) as resolved_db:
            already = {n.lower() for n in fp.fetch_resolved_input_names(resolved_db.conn)}
        before = len(inputs)
        inputs = [(n, d) for (n, d) in inputs if n.lower() not in already]
        skipped_count = before - len(inputs)

    click.echo(
        f"Routing {len(inputs)} fantasy-name input(s)"
        f"{f' (skipped {skipped_count} already-resolved)' if skip_resolved else ''}. "
        f"approach_version={fp.APPROACH_VERSION}. "
        f"{'Applying' if apply_changes else 'Dry-run'}.",
        err=True,
    )

    counts = {"usable": 0, "barred": 0, "by_method": {}, "by_bar": {}}
    # Bind the LLM callers to the user-supplied model (if any). The
    # pipeline calls semantic_check_caller(name, desc, ancestor, glosses)
    # and llm_caller(name, desc); functools.partial pre-binds the model
    # kwarg without changing those signatures.
    if model:
        llm_caller = partial(fp._llm_full_research, model=model)
        semantic_check_caller = partial(fp._llm_semantic_check, model=model)
    else:
        llm_caller = fp._llm_full_research
        semantic_check_caller = fp._llm_semantic_check

    # Mining-progress convention: same shape as `lexicon mine-llm` so
    # operators can read both batches the same way. See repo CLAUDE.md
    # under "Mining progress reporting".
    progress_start = time.time()
    progress_every = 10
    total_inputs = len(inputs)

    write_db: LexiconDB | None = LexiconDB(db_path) if apply_changes else None
    try:
        for completed, (in_name, in_desc) in enumerate(inputs, start=1):
            res = fp.resolve(
                db_path,
                in_name,
                in_desc,
                skip_llm=skip_llm,
                llm_caller=llm_caller,
                semantic_check_caller=semantic_check_caller,
            )
            if res.usable:
                counts["usable"] += 1
                tag = f"USABLE  {in_name:<18}"
            else:
                counts["barred"] += 1
                tag = f"BARRED  {in_name:<18}"
                counts["by_bar"][res.bar_reason or "?"] = (
                    counts["by_bar"].get(res.bar_reason or "?", 0) + 1
                )
            counts["by_method"][res.resolution_method] = (
                counts["by_method"].get(res.resolution_method, 0) + 1
            )
            click.echo(
                f"  {tag} via={res.resolution_method:<22} conf={res.confidence or '-':<7}"
                f"{(' bar=' + res.bar_reason) if res.bar_reason else ''}"
                f"{(' citation=' + res.citation) if res.citation else ''}",
                err=True,
            )
            if write_db is not None:
                fp.write_resolution(
                    write_db.conn,
                    input_name=in_name,
                    input_description=in_desc,
                    resolution=res,
                )
                if res.usable and res.etymon_id is not None:
                    fp.tag_etymon_as_fantasy(write_db.conn, res.etymon_id)
                # Commit per resolution: each input cost an LLM call, so
                # crashing partway through a long batch shouldn't lose
                # already-paid-for results.
                write_db.commit()
            # Periodic progress line (matches mine-llm shape so operators
            # can scan both batches the same way).
            if completed % progress_every == 0 or completed == total_inputs:
                elapsed = time.time() - progress_start
                rate = elapsed / completed
                click.echo(
                    f"  [{completed}/{total_inputs}]  "
                    f"usable={counts['usable']} barred={counts['barred']} "
                    f"({rate:.1f}s/entry)",
                    err=True,
                )
    finally:
        if write_db is not None:
            write_db.close()

    click.echo("", err=True)
    click.echo(f"Summary: {counts}", err=True)
    if not apply_changes:
        click.echo("(dry-run; pass --apply to write)", err=True)


@lexicon.command("ingest-domesday")
@click.argument(
    "mdb_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    default=False,
    help=(
        "Actually insert toponym + attestation rows. Without this the "
        "command parses the MDB and reports counts without writing."
    ),
)
@click.option(
    "--include-uncertain",
    is_flag=True,
    default=False,
    help=(
        "Include PlaceForm rows tagged OScodes='speculative' / "
        "'unknown' (lost vills, uncertain modern identifications). "
        "Default skips these (~600 rows)."
    ),
)
def lexicon_ingest_domesday(
    mdb_path: Path,
    db_path: Path,
    apply_changes: bool,
    include_uncertain: bool,
) -> None:
    """Ingest the Open Domesday (Hull) place-name corpus (wyrd-el93).

    Downloads not handled here — point ``MDB_PATH`` at a local copy
    of the Hull Places database (.mdb file from
    digitalcollections.hull.ac.uk/downloads/th83kz829). Walks the
    PlaceForm table (~20K rows across ~14K places) and writes one
    toponym row per (modern_name, county) + one toponym_attestation
    row per Domesday entry with date_year=1086.

    Idempotent: re-runs detect existing toponyms by (modern_name,
    region) and existing attestations by (toponym_id, form, year,
    source_doc).

    License: Hull's data is CC-BY-NC-SA 4.0 (J.J.N. Palmer team).
    See ``domesday_ingester.DOMESDAY_SOURCE_NOTES`` for the full
    attribution recorded in the ``source`` table.
    """
    from urllib.parse import quote

    from wyrd.generators.kenning.domesday_ingester import ingest_domesday

    # URL-quote the path so spaces / reserved chars (?, #) in
    # operator-supplied paths don't get parsed as URI components.
    db_uri = f"file:{quote(str(db_path.absolute()))}?mode={'rw' if apply_changes else 'ro'}"
    conn = sqlite3.connect(db_uri, uri=True)
    conn.row_factory = sqlite3.Row

    last_pct = -1

    def _progress(done: int, total: int) -> None:
        nonlocal last_pct
        pct = int(done / total * 100.0) if total else 100
        # Throttle stderr to once per percent-tick.
        if pct != last_pct:
            last_pct = pct
            click.echo(
                f"  domesday-ingest: {done}/{total} ({pct}%)",
                err=True,
            )

    try:
        counts = ingest_domesday(
            conn,
            mdb_path,
            apply=apply_changes,
            include_uncertain=include_uncertain,
            progress_callback=_progress,
        )
    finally:
        conn.close()

    click.echo("", err=True)
    click.echo(f"Summary: {counts}", err=True)
    if not apply_changes:
        click.echo("(dry-run; pass --apply to write)", err=True)


@lexicon.group("browse")
def lexicon_browse() -> None:
    """Read-only navigation of the lexicon DB (wyrd-7oma).

    Grep-friendly markdown views of toponyms, etymons, decompositions,
    and sources. Use BEFORE running a curation pass (lemma normalize,
    dead-rando prune, language retag) to verify which rows you're
    about to edit.
    """


@contextmanager
def _readonly_lexicon(db_path: Path):
    """Yield a read-only ``sqlite3.Connection`` for browse subcommands,
    closing it on exit. Used as a context manager so the connection
    can't leak when a command early-exits via ``raise SystemExit``."""
    from urllib.parse import quote

    db_uri = f"file:{quote(str(db_path.absolute()))}?mode=ro"
    conn = sqlite3.connect(db_uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


@lexicon_browse.command("etymon")
@click.argument("ref")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
def lexicon_browse_etymon(ref: str, db_path: Path) -> None:
    """Show one etymon: glosses, tags, citations, descent, lemma family.

    REF is the cross-file etymon ref, e.g. 'old-english:cot'.
    """
    from wyrd.generators.kenning.browse import fetch_etymon, format_etymon

    with _readonly_lexicon(db_path) as conn:
        data = fetch_etymon(conn, ref)

    if data is None:
        click.echo(f"No etymon found for ref={ref!r}", err=True)
        raise SystemExit(1)
    click.echo(format_etymon(data))


@lexicon_browse.command("toponym")
@click.argument("query")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
def lexicon_browse_toponym(query: str, db_path: Path) -> None:
    """Show one toponym: attestations + etymologies + decompositions.

    QUERY is either a bare modern name ('Cotton') or 'name@region'
    ('Cotton@Norfolk'). A bare name that matches multiple toponyms
    prints the disambiguation list instead.
    """
    from wyrd.generators.kenning.browse import (
        fetch_toponym_detail,
        fetch_toponyms_matching,
        format_toponym,
        format_toponym_list,
        parse_toponym_ref,
    )

    name, region = parse_toponym_ref(query)
    with _readonly_lexicon(db_path) as conn:
        matches = fetch_toponyms_matching(conn, name, region)
        if not matches:
            click.echo(f"No toponym found for query={query!r}", err=True)
            raise SystemExit(1)
        if len(matches) > 1:
            click.echo(format_toponym_list(matches))
            return
        data = fetch_toponym_detail(conn, matches[0]["id"])
    click.echo(format_toponym(data))


@lexicon_browse.command("decomposition")
@click.argument("query")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
def lexicon_browse_decomposition(query: str, db_path: Path) -> None:
    """Show matcher decompositions for one toponym.

    QUERY is 'name' or 'name@region'. Surfaces the canonical pick +
    every alternative the matcher emitted.
    """
    from wyrd.generators.kenning.browse import (
        fetch_decompositions,
        fetch_toponyms_matching,
        format_decompositions,
        format_toponym_list,
        parse_toponym_ref,
    )

    name, region = parse_toponym_ref(query)
    with _readonly_lexicon(db_path) as conn:
        matches = fetch_toponyms_matching(conn, name, region)
        if not matches:
            click.echo(f"No toponym found for query={query!r}", err=True)
            raise SystemExit(1)
        if len(matches) > 1:
            click.echo(format_toponym_list(matches))
            return
        decompositions = fetch_decompositions(conn, matches[0]["id"])
        match = matches[0]
    click.echo(format_decompositions(match["modern_name"], match["region"], decompositions))


@lexicon_browse.command("source")
@click.argument("source_id")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--list-toponyms",
    is_flag=True,
    default=False,
    help="Also print the full list of toponyms covered by this source.",
)
def lexicon_browse_source(source_id: str, db_path: Path, list_toponyms: bool) -> None:
    """Show one source: bibliographic record + contribution counts.

    SOURCE_ID is the source row's id (e.g. 'skeat_1901_cambridgeshire').
    """
    from wyrd.generators.kenning.browse import fetch_source, format_source

    with _readonly_lexicon(db_path) as conn:
        data = fetch_source(conn, source_id)

    if data is None:
        click.echo(f"No source found for id={source_id!r}", err=True)
        raise SystemExit(1)
    click.echo(format_source(data, list_toponyms=list_toponyms))


@lexicon.command("era-timeline")
@click.argument("query")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
def lexicon_era_timeline(query: str, db_path: Path) -> None:
    """Show one toponym's attestations bucketed by era (wyrd-ub76).

    QUERY is either a bare modern name ('Acton') or 'name@region'
    ('Acton@Cheshire'). For ambiguous bare names the first region in
    ASCII order is shown — disambiguate with @region.

    Era buckets: Anglo-Saxon (<1100), Middle English (1100-1500),
    Early Modern (1500-1800), Modern (>=1800), Undated.
    """
    from wyrd.generators.kenning.era_timeline import (
        fetch_era_timeline,
        format_era_timeline,
    )

    with _readonly_lexicon(db_path) as conn:
        timeline = fetch_era_timeline(conn, query)

    if timeline is None:
        click.echo(f"No toponym found for query={query!r}", err=True)
        raise SystemExit(1)
    click.echo(format_era_timeline(timeline))


@lexicon.command("era-coverage")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
def lexicon_era_coverage(db_path: Path) -> None:
    """Aggregate cross-era coverage across the toponym corpus (wyrd-ub76).

    Reports per-era toponym counts, the histogram of how many eras
    each toponym is attested in, and the percent with >=3 dated eras
    — the headline 'is the cross-era story landing' number.
    """
    from wyrd.generators.kenning.era_timeline import (
        fetch_era_coverage,
        format_era_coverage,
    )

    with _readonly_lexicon(db_path) as conn:
        report = fetch_era_coverage(conn)
    click.echo(format_era_coverage(report))


@lexicon.command("audit-short-quotes")
@click.option(
    "--dir",
    "directory",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data/mining"),
    show_default=True,
    help="Directory of <source>.jsonl files to scan.",
)
@click.option(
    "--source",
    "sources",
    multiple=True,
    help="Restrict to named source_ids (repeat for multiple).",
)
@click.option(
    "--samples",
    "sample_limit",
    type=int,
    default=3,
    show_default=True,
    help="Max sample short_quotes to print per source.",
)
@click.option(
    "--top",
    "top_n",
    type=int,
    default=20,
    show_default=True,
    help="Max sources to include in the report.",
)
def lexicon_audit_short_quotes(
    directory: Path, sources: tuple[str, ...], sample_limit: int, top_n: int
) -> None:
    """Audit citation short_quotes for LLM truncation (wyrd-bd68).

    Walks every <source>.jsonl in DIRECTORY, flags short_quotes that
    look cut off mid-sentence, and prints a markdown report. Operator
    decides whether to re-mine the flagged sources or context-snippet
    refill via the page-number infrastructure.
    """
    from wyrd.generators.kenning.short_quote_audit import (
        audit_jsonl_dir,
        format_audit_report,
    )

    report = audit_jsonl_dir(
        directory,
        sample_limit=sample_limit,
        sources=sources or None,
    )
    click.echo(format_audit_report(report, top_n=top_n))


@lexicon.command("refill-short-quotes")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
    help="Lexicon SQLite DB to read + (with --apply) write.",
)
@click.option(
    "--sources-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("sources"),
    show_default=True,
    help="Directory of <source_id>.txt source bodies for refill lookup.",
)
@click.option(
    "--source",
    "sources",
    multiple=True,
    help="Restrict to named source_ids (repeat for multiple).",
)
@click.option(
    "--apply",
    is_flag=True,
    default=False,
    help="Actually UPDATE etymon_citation.short_quote rows. Without this, dry-run.",
)
@click.option(
    "--window",
    type=int,
    default=200,
    show_default=True,
    help="Forward chars of recovered context past the matched tail.",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Print before/after samples per source so operators can spot-check the refill quality.",
)
def lexicon_refill_short_quotes(
    db_path: Path,
    sources_dir: Path,
    sources: tuple[str, ...],
    apply: bool,
    window: int,
    verbose: bool,
) -> None:
    """Refill truncated short_quotes from source body text (wyrd-1hpc).

    For each citation whose short_quote is flagged by
    `audit-short-quotes`, locates the tail in `sources/<source_id>.txt`
    and extends forward by --window chars to recover the
    LLM-truncated continuation. Non-locatable cases (LLM-hallucinated
    commentary that was never in the source body) are reported as
    `hallucinated`; those need an LLM re-mine, not refill.

    After --apply, run `lexicon dump-jsonl` to propagate the refilled
    short_quotes into the committed `data/mining/*.jsonl` files.
    """
    # Deferred for CLI help-snappiness — the refill module imports
    # short_quote_audit (re + regex compilation), and we only want
    # to pay that cost when the command actually runs. Pattern
    # documented in wyrd-xz6l.
    from wyrd.generators.kenning.short_quote_refill import refill_source

    total_truncated = 0
    total_refilled = 0
    total_hallucinated = 0

    # Use LexiconDB context manager for consistency with other lexicon
    # commands (Gemini PR #211 round-2): the wrapper sets PRAGMA
    # foreign_keys=ON + synchronous=NORMAL on every open, which the
    # bare sqlite3.connect skipped.
    with LexiconDB(db_path) as db:
        # Validate --source values up front (silent-failure-hunter
        # round-1 finding): a misspelled source id used to silently
        # produce zero output, masking operator intent. Surface as a
        # ClickException with the list of unknown ids.
        if sources:
            known = {r["id"] for r in db.conn.execute("SELECT id FROM source")}
            unknown = sorted(set(sources) - known)
            if unknown:
                raise click.ClickException(
                    f"unknown source_id(s): {', '.join(unknown)}. "
                    f"Check spelling against `lexicon report --sources` output."
                )
            target_sources = list(sources)
        else:
            # Only iterate sources that actually have citations
            # (Gemini PR #211 round-7 efficiency). Many sources in
            # the source table have zero citations (placeholder rows
            # for not-yet-mined books, retired sources, etc.) —
            # skipping them avoids per-source refill_source calls
            # that would return empty reports anyway.
            target_sources = [
                r[0]
                for r in db.conn.execute(
                    "SELECT DISTINCT source_id FROM etymon_citation ORDER BY source_id"
                )
            ]

        for source_id in target_sources:
            report = refill_source(db.conn, source_id, sources_dir, apply=apply, window=window)
            if report.total_truncated == 0:
                continue
            total_truncated += report.total_truncated
            total_refilled += report.refilled
            total_hallucinated += report.hallucinated
            # Report content goes to stdout so operators can redirect
            # cleanly (`refill-short-quotes > report.txt`) while genuine
            # errors stay on stderr. Matches sibling report-generating
            # CLIs (audit-short-quotes, audit-etymology-alignment).
            # Gemini PR #211 round-4.
            click.echo(
                f"{source_id:<55} truncated={report.total_truncated:>4} "
                f"refilled={report.refilled:>4} "
                f"hallucinated={report.hallucinated:>4}"
            )
            # Gemini PR #211 round-2 visibility fix (kept narrow per
            # round-5 reconsideration on hallucinated false-positives,
            # broadened scope per Gemini round-5b on default-mode
            # visibility): warning fires when refill_source returned
            # without ever consulting the source body — which only
            # happens when _load_source_body returned None (missing
            # file, UnicodeDecodeError, OSError, or empty/whitespace-
            # only body). The narrow condition (refilled==0 AND
            # hallucinated==0) correctly distinguishes 'source body
            # unavailable' from 'source body fine but LLM
            # hallucinated everything'.
            #
            # No longer gated on `sources` (explicit --source) per
            # Gemini round-5b: default-mode runs across all sources
            # should ALSO surface missing-body gaps. The narrow
            # condition keeps the noise floor low (only fires for
            # sources that have truncated rows but no consulted
            # source body), so default-mode runs only warn for
            # sources where the operator actually has a refill
            # surface to fix.
            if report.refilled == 0 and report.hallucinated == 0:
                # Use the centralized helper so this warning's reported
                # path is guaranteed to match what _load_source_body
                # looked at (Gemini PR #211 round-9 DRY fix; the
                # round-7 path-traversal hardening + round-8
                # consistency fix now live in one place).
                from wyrd.generators.kenning.short_quote_refill import source_body_path

                txt = source_body_path(sources_dir, source_id)
                if not txt.exists():
                    click.echo(f"    warning: {txt} not found — refill needs the source body")
                else:
                    click.echo(
                        f"    warning: {txt} exists but read returned empty/unreadable "
                        f"content (non-UTF8 bytes? whitespace-only?) — refill needs a "
                        f"readable source body"
                    )
            if verbose:
                # Surface RefillReport.samples so operators can
                # spot-check before/after quality without manual DB
                # inspection .
                for sample in report.samples:
                    click.echo(f"    [{sample.status}] {sample.etymon_ref}")
                    if sample.old_short_quote:
                        click.echo(f"      old (...{sample.old_short_quote[-60:]!r})")
                    if sample.new_short_quote:
                        click.echo(
                            f"      new (+{sample.recovered_chars}c: "
                            f"...{sample.new_short_quote[-100:]!r})"
                        )
    click.echo("")
    click.echo(
        f"TOTAL: truncated={total_truncated} refilled={total_refilled} "
        f"hallucinated={total_hallucinated} "
        f"({'APPLIED' if apply else 'dry-run'})"
    )


@lexicon.command("reverse-search-toponyms")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
    help="Lexicon SQLite DB.",
)
@click.option(
    "--sources-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("sources"),
    show_default=True,
    help="Directory of <source_id>.txt source bodies to scan.",
)
@click.option(
    "--source",
    "sources",
    multiple=True,
    help="Restrict to named source_ids (repeat for multiple).",
)
@click.option(
    "--apply",
    is_flag=True,
    default=False,
    help="INSERT OR IGNORE matched attestations. Without this, dry-run.",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Print samples of unmatched forms per source (Phase 2 LLM-mining candidates).",
)
def lexicon_reverse_search_toponyms(
    db_path: Path,
    sources_dir: Path,
    sources: tuple[str, ...],
    apply: bool,
    verbose: bool,
) -> None:
    """Toponym reverse-search: extract (form, year) attestations from scholar
    source bodies (wyrd-x82p Phase 1).

    Walks each scholar source's ``.txt`` body, runs the
    ``mine-attestations`` regex set against it (rather than only against
    ``toponym_etymology.notes``), maps each extracted form back to a known
    toponym, and emits new ``toponym_attestation`` rows via the existing
    UNIQUE-index-respecting INSERT OR IGNORE pattern.

    Forms that don't map to a known toponym are tallied with sample
    output (--verbose) — those are Phase 2 LLM-mining candidates for
    new-toponym discovery.

    After --apply, run ``lexicon dump-jsonl`` to propagate to the
    committed ``data/mining/<source>.jsonl`` files.
    """
    # Deferred for CLI help-snappiness (per wyrd-xz6l pattern).
    from wyrd.generators.kenning.toponym_reverse_search import (
        _build_form_to_toponym_lookup,
        reverse_search_source,
    )

    totals = {
        "pairs": 0,
        "matched": 0,
        "unmatched": 0,
        "inserted": 0,
        "already_present": 0,
    }

    with LexiconDB(db_path) as db:
        if sources:
            known = {r["id"] for r in db.conn.execute("SELECT id FROM source")}
            unknown = sorted(set(sources) - known)
            if unknown:
                raise click.ClickException(
                    f"unknown source_id(s): {', '.join(unknown)}. "
                    f"Check spelling against `lexicon report --sources` output."
                )
            target_sources = list(sources)
        else:
            # Iterate the source table — every registered scholar
            # source is a candidate for body-text reverse-search
            # regardless of whether it has citations yet. Earlier
            # version used `DISTINCT source_id FROM etymon_citation`
            # (mirroring wyrd-1hpc); Gemini PR #212 round-2 finding
            # corrected: a source can have toponym attestations
            # without etymons (operator-uploaded source body before
            # the etymon mining pass runs, or a source that's
            # toponym-only by design). Sources without a committed
            # .txt body skip via _load_source_body returning None;
            # no harm in iterating them.
            target_sources = [r["id"] for r in db.conn.execute("SELECT id FROM source ORDER BY id")]

        # Build the form→toponym lookup ONCE up front. 21,969+ toponyms
        # in the live DB; rebuilding per-source would re-execute two
        # full-table scans per source.
        click.echo("Building form→toponym lookup...", err=True)
        form_to_toponym = _build_form_to_toponym_lookup(db.conn)
        click.echo(f"  {len(form_to_toponym):,} normalized forms indexed", err=True)

        for source_id in target_sources:
            report = reverse_search_source(
                db.conn, source_id, sources_dir, form_to_toponym, apply=apply
            )
            if report.pairs_extracted == 0:
                continue
            totals["pairs"] += report.pairs_extracted
            totals["matched"] += report.matched
            totals["unmatched"] += report.unmatched
            totals["inserted"] += report.inserted
            totals["already_present"] += report.already_present
            # Per-source progress goes to stderr — diagnostic output;
            # operators redirecting to a report file want the TOTAL
            # tally on stdout (set after the loop).
            click.echo(
                f"{source_id:<55} pairs={report.pairs_extracted:>5} "
                f"matched={report.matched:>5} unmatched={report.unmatched:>5} "
                f"inserted={report.inserted:>5} dup={report.already_present:>5}",
                err=True,
            )
            if verbose and report.unmatched_samples:
                # Unmatched forms are the prime candidates for Phase 2's
                # LLM-driven new-toponym discovery. Display ALL samples
                # the module collected (default cap is 10); operators
                # shouldn't see a smaller slice than the data the report
                # exposes.
                for form, year in report.unmatched_samples:
                    click.echo(f"    unmatched: {form!r} ({year})", err=True)
        # Commit once after walking all sources — the function-level
        # commit was removed for caller transaction control; this is
        # the natural commit point with partial-failure rollback
        # semantics if a later source raises.
        if apply:
            db.conn.commit()
    click.echo("")
    click.echo(
        f"TOTAL: pairs={totals['pairs']} matched={totals['matched']} "
        f"unmatched={totals['unmatched']} inserted={totals['inserted']} "
        f"dup={totals['already_present']} "
        f"({'APPLIED' if apply else 'dry-run'})"
    )


@lexicon.command("mine-toponym-mentions")
@click.option(
    "--source",
    "source_id",
    required=True,
    help="Source id (e.g. mawer_1920_northumberland_durham) to mine.",
)
@click.option(
    "--sources-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("sources"),
    show_default=True,
    help="Directory of <source_id>.txt source bodies.",
)
@click.option(
    "--provider",
    type=click.Choice(["anthropic", "gemini", "ollama"]),
    default="anthropic",
    show_default=True,
    help="LLM backend to use.",
)
@click.option(
    "--model",
    default=None,
    help="Override model id (provider-specific default if omitted).",
)
@click.option(
    "--chunk-size",
    type=int,
    default=20000,
    show_default=True,
    help="Target chunk size in characters (snaps to paragraph boundaries).",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Process only the first N chunks (smoke testing).",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write extracted mentions to this JSONL file. Default: stdout.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite --output if it exists. Without this flag, an existing path errors.",
)
@click.option(
    "--ollama-url",
    default=None,
    help="Override Ollama base URL when --provider ollama (default: "
    "$WYRD_OLLAMA_URL or localhost). Same shape as the mine-llm flag.",
)
def lexicon_mine_toponym_mentions(
    source_id: str,
    sources_dir: Path,
    provider: str,
    model: str | None,
    chunk_size: int,
    limit: int | None,
    output: Path | None,
    force: bool,
    ollama_url: str | None,
) -> None:
    """LLM-mine a scholar source body for place-name mentions (wyrd-x82p Phase 2).

    Pilot tool: walks one source's .txt body in chunks, calls an LLM to
    extract every place-name mention with optional date_year + region_hint
    + surrounding context, emits results as JSONL (one mention per line)
    to stdout or ``--output``.

    Does NOT touch the DB — the pilot output is for operator review.
    Phase 2b will add DB persistence + a resolver that maps mentions
    to existing toponyms (via the form→toponym lookup from Phase 1)
    OR creates new toponym rows for unresolved mentions.
    """
    from wyrd.generators.kenning.toponym_mention_extractor import (
        ToponymMention,
        mine_toponym_mentions,
    )

    txt_path = sources_dir / f"{Path(source_id).name}.txt"
    if not txt_path.exists():
        raise click.ClickException(f"source body not found: {txt_path}")
    body = txt_path.read_text(encoding="utf-8", errors="replace")
    if "�" in body:
        # errors="replace" silently mapped invalid bytes to U+FFFD.
        # The form-in-chunk validator will then drop any mention
        # containing those characters, with no visible signal.
        # OE/ME place-name texts genuinely use æ/þ/ð — surface a
        # warning so the operator can fix the encoding upstream.
        click.echo(
            f"warning: {txt_path} contained invalid UTF-8 sequences "
            f"(now U+FFFD); some mentions may be silently rejected",
            err=True,
        )
    click.echo(f"Loaded {len(body):,} chars from {txt_path}", err=True)

    if output is not None and output.exists():
        if not force:
            raise click.ClickException(
                f"--output {output} already exists; pass --force to overwrite"
            )
        # --force was passed; warn the operator so a multi-hour pilot
        # run doesn't get silently obliterated by a typo'd re-run.
        click.echo(
            f"warning: --force overwriting existing {output}",
            err=True,
        )

    # Route through the shared _build_extractor_client helper (same
    # dispatch as mine-toponym-mentions-tiered) so both commands use
    # the same provider/model/ollama-url contract. The helper is
    # defined later in this module — Python looks up names at call
    # time so the forward reference works.
    client = _build_extractor_client(provider, model, ollama_url=ollama_url)

    click.echo(f"Using {provider} model={client.model}", err=True)

    start_ts = time.monotonic()

    def progress(done: int, total: int, mentions: int) -> None:
        # Project convention (CLAUDE.md): "[done/total] mentions=N (Xs/chunk)"
        elapsed = time.monotonic() - start_ts
        rate = elapsed / done if done else 0.0
        click.echo(
            f"  [{done}/{total}] mentions={mentions} ({rate:.1f}s/chunk)",
            err=True,
        )

    def warn(msg: str) -> None:
        click.echo(f"  warning: {msg}", err=True)

    report = mine_toponym_mentions(
        client,
        source_id,
        body,
        target_chunk_size=chunk_size,
        limit=limit,
        on_chunk_done=progress,
        log_warning=warn,
    )

    click.echo("", err=True)
    click.echo(
        f"TOTAL chunks_processed={report.chunks_processed} "
        f"chunks_failed={report.chunks_failed} "
        f"mentions={len(report.mentions)} "
        f"hallucinations_dropped={report.hallucinations_dropped} "
        f"years_clamped={report.years_clamped}",
        err=True,
    )
    if report.failed_chunks:
        click.echo("Failed chunks:", err=True)
        for fc in report.failed_chunks:
            click.echo(f"  chunk {fc.index}: {fc.error}", err=True)
        # When the cap truncates the list, mention how many were elided.
        elided = report.chunks_failed - len(report.failed_chunks)
        if elided > 0:
            click.echo(f"  ... and {elided} more failures (cap reached)", err=True)

    # ensure_ascii=False preserves Old/Middle English diacritics
    # (æ, þ, ð) in the JSONL output — project convention used by the
    # existing diff/export commands. For stdout we use click.echo
    # : it handles terminal-encoding edge cases
    # (Latin-1 PYTHONIOENCODING, stdout=non-UTF-8 device) by re-
    # encoding via UTF-8 or replacement, where a bare sys.stdout.write
    # would raise UnicodeEncodeError. For --output we write the file
    # directly with explicit UTF-8 encoding.
    def _line(m: ToponymMention) -> str:
        return json.dumps(
            {
                "source_id": source_id,
                "form": m.form,
                "date_year": m.date_year,
                "region_hint": m.region_hint,
                "context": m.context,
            },
            ensure_ascii=False,
        )

    if output is not None:
        with output.open("w", encoding="utf-8") as sink:
            for m in report.mentions:
                sink.write(_line(m) + "\n")
        click.echo(f"Wrote {len(report.mentions)} mentions → {output}", err=True)
    else:
        for m in report.mentions:
            click.echo(_line(m))


# 8192-token max-tokens for Anthropic — fits ~100-150 mentions per
# chunk comfortably; the SDK default of 1024 truncates mid-JSON on
# dense chunks. Single source-of-truth used by both single-tier
# (mine-toponym-mentions) and tiered (mine-toponym-mentions-tiered)
# Anthropic client construction.
_ANTHROPIC_MAX_TOKENS_FOR_MENTION_EXTRACTION = 8192


def _build_extractor_client(provider: str, model: str | None, ollama_url: str | None = None):
    """Construct an extractor client for the named provider. Shared
    by both `mine-toponym-mentions` (single-tier) and
    `mine-toponym-mentions-tiered` (primary + fallback) so the
    provider/model/ollama-url contract is identical across the two
    CLI commands.

    ``ollama_url`` is forwarded to OllamaClient when provider=="ollama";
    None falls back to DEFAULT_OLLAMA_URL ($WYRD_OLLAMA_URL or
    localhost). Other providers ignore the kwarg."""
    if provider == "anthropic":
        from wyrd.generators.kenning.anthropic_extractor import (
            DEFAULT_ANTHROPIC_MODEL,
            AnthropicClient,
        )

        return AnthropicClient(
            model=model or DEFAULT_ANTHROPIC_MODEL,
            max_tokens=_ANTHROPIC_MAX_TOKENS_FOR_MENTION_EXTRACTION,
        )
    if provider == "gemini":
        from wyrd.generators.kenning.gemini_extractor import (
            DEFAULT_GEMINI_MODEL,
            GeminiClient,
        )

        return GeminiClient(model=model or DEFAULT_GEMINI_MODEL)
    if provider == "ollama":
        from wyrd.generators.kenning.llm_extractor import (
            DEFAULT_OLLAMA_MODEL,
            DEFAULT_OLLAMA_URL,
            OllamaClient,
        )

        return OllamaClient(
            base_url=ollama_url or DEFAULT_OLLAMA_URL,
            model=model or DEFAULT_OLLAMA_MODEL,
        )
    # Unreachable in practice — click.Choice gates the option. The
    # explicit raise keeps the type-checker happy and surfaces a
    # clear error if the Choice list and this dispatch drift apart.
    raise click.ClickException(f"unknown provider: {provider}")


@lexicon.command("mine-toponym-mentions-tiered")
@click.option(
    "--sources-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("sources"),
    show_default=True,
    help="Directory of <source_id>.txt scholar bodies.",
)
@click.option(
    "--source",
    "sources",
    multiple=True,
    help="Restrict to named source_ids (repeat for multiple). Default: every "
    "*.txt under --sources-dir.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/mining/phase2"),
    show_default=True,
    help="Directory to write <source_id>.jsonl extraction output. Created if missing.",
)
@click.option(
    "--primary-provider",
    type=click.Choice(["anthropic", "gemini", "ollama"]),
    default="ollama",
    show_default=True,
    help="Primary client — runs first on every chunk. Ollama (Qwen on Hades) "
    "is the cost-efficient default; the fallback client only fires on chunks "
    "the primary failed.",
)
@click.option(
    "--primary-model",
    default=None,
    help="Override primary model id (defaults to the provider's default).",
)
@click.option(
    "--fallback-provider",
    type=click.Choice(["anthropic", "gemini", "ollama"]),
    default="anthropic",
    show_default=True,
    help="Fallback client — retries chunks the primary failed. Anthropic is the "
    "default (highest quality on hard chunks).",
)
@click.option(
    "--fallback-model",
    default=None,
    help="Override fallback model id (defaults to the provider's default).",
)
@click.option(
    "--ollama-url",
    default=None,
    help="Override Ollama base URL when either tier is `ollama` (default: "
    "$WYRD_OLLAMA_URL or localhost). Used for both primary and fallback if "
    "either is set to ollama.",
)
@click.option(
    "--chunk-size",
    type=int,
    default=20000,
    show_default=True,
    help="Target chunk size in characters (snaps to paragraph boundaries).",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Process only the first N chunks per source (smoke testing).",
)
@click.option(
    "--skip-existing",
    is_flag=True,
    default=False,
    help="Skip sources whose output JSONL already exists in --output-dir. "
    "Idempotent multi-source resume — sources you've already run stay untouched.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite existing per-source output files (default: error). "
    "Mutually-exclusive with --skip-existing.",
)
@click.option(
    "--hallucination-fallback-threshold",
    type=click.IntRange(min=0),
    default=None,
    help="When set to N >= 1, chunks where the primary SUCCEEDED but emitted "
    "at least N hallucinated forms (per the word-boundary guard) ALSO get "
    "re-extracted by the fallback; mentions are union-merged by "
    "(form, date_year, region_hint, context). Primary wins on byte-identical "
    "collision; fallback adds what primary didn't see. Omitting the flag "
    "or passing 0 disables the rescue (default). Negative values are "
    "rejected by IntRange. Use with `--primary-provider ollama` + small "
    "local model + `--fallback-provider anthropic` to keep gemma4's good "
    "mentions while letting Anthropic catch its fabrications. wyrd-z8mq.",
)
def lexicon_mine_toponym_mentions_tiered(
    sources_dir: Path,
    sources: tuple[str, ...],
    output_dir: Path,
    primary_provider: str,
    primary_model: str | None,
    fallback_provider: str,
    fallback_model: str | None,
    ollama_url: str | None,
    chunk_size: int,
    limit: int | None,
    skip_existing: bool,
    force: bool,
    hallucination_fallback_threshold: int | None,
) -> None:
    """Two-tier LLM mention extraction across one or many sources
    (wyrd-x82p Phase 2b.2).

    Runs the primary client first on every chunk; retries chunks that
    failed (transport, JSON parse, schema mismatch) with the fallback
    client. Designed for the wyrd-l0r tiered pattern: Qwen on Hades
    for bulk first-pass (free, fast), Anthropic on the residual (best
    quality on hard chunks).

    Walks every source_id given via --source (or every *.txt under
    --sources-dir when --source is omitted). Writes one JSONL per
    source to --output-dir/<source_id>.jsonl, ready to feed into
    ``lexicon ingest-toponym-mentions``.

    Idempotent resume: --skip-existing lets you re-run after an
    interrupt without re-processing already-extracted sources.
    """
    import time

    from wyrd.generators.kenning.toponym_mention_extractor import (
        ToponymMention,
        mine_toponym_mentions_tiered,
    )

    if skip_existing and force:
        raise click.ClickException("--skip-existing and --force are mutually exclusive")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve source list. If --source isn't given, walk *.txt under
    # the sources dir; sort for deterministic ordering across runs.
    if sources:
        # Validate each requested source exists as a .txt under sources_dir.
        source_ids = []
        for sid in sources:
            txt = sources_dir / f"{Path(sid).name}.txt"
            if not txt.exists():
                raise click.ClickException(f"source body not found: {txt}")
            source_ids.append(Path(sid).name)
    else:
        source_ids = sorted(p.stem for p in sources_dir.glob("*.txt") if p.name != "MANIFEST.md")

    if not source_ids:
        raise click.ClickException(f"no sources under {sources_dir}")

    click.echo(f"Sources to process: {len(source_ids)}", err=True)
    click.echo(
        f"Primary: {primary_provider} (model={primary_model or 'default'})",
        err=True,
    )
    click.echo(
        f"Fallback: {fallback_provider} (model={fallback_model or 'default'})",
        err=True,
    )

    # Lazy client construction — only instantiate when we have an
    # actual source to extract. For a 200-source resume where every
    # source is --skip-existing'd, this avoids unnecessary ANTHROPIC_
    # API_KEY validation and Ollama-URL probing. The pair is built
    # once on the first real extraction and cached for the remainder.
    primary_client = None
    fallback_client = None

    def _ensure_clients():
        nonlocal primary_client, fallback_client
        if primary_client is None:
            primary_client = _build_extractor_client(
                primary_provider, primary_model, ollama_url=ollama_url
            )
            fallback_client = _build_extractor_client(
                fallback_provider, fallback_model, ollama_url=ollama_url
            )

    def _line(source_id: str, m: ToponymMention) -> str:
        return json.dumps(
            {
                "source_id": source_id,
                "form": m.form,
                "date_year": m.date_year,
                "region_hint": m.region_hint,
                "context": m.context,
            },
            ensure_ascii=False,
        )

    totals = {
        "sources_processed": 0,
        "sources_skipped": 0,
        "chunks_processed": 0,
        "chunks_recovered": 0,
        "chunks_hallucination_rescue_attempted": 0,
        "chunks_hallucination_rescue_succeeded": 0,
        "chunks_failed": 0,
        "hallucinations_dropped": 0,
        "years_clamped": 0,
        "mentions": 0,
    }

    for src_i, source_id in enumerate(source_ids, start=1):
        out_path = output_dir / f"{source_id}.jsonl"
        if out_path.exists():
            if skip_existing:
                click.echo(
                    f"[{src_i}/{len(source_ids)}] {source_id}: SKIP (output exists)",
                    err=True,
                )
                totals["sources_skipped"] += 1
                continue
            if not force:
                raise click.ClickException(
                    f"output {out_path} exists; pass --skip-existing to resume or "
                    f"--force to overwrite"
                )

        txt_path = sources_dir / f"{source_id}.txt"
        body = txt_path.read_text(encoding="utf-8", errors="replace")
        if "�" in body:
            click.echo(
                f"  warning: {source_id} contained invalid UTF-8 sequences; some "
                f"mentions may be silently rejected",
                err=True,
            )

        click.echo(
            f"[{src_i}/{len(source_ids)}] {source_id}: {len(body):,} chars",
            err=True,
        )

        # Defer client construction to first real extraction —
        # --skip-existing resume runs where every source is skipped
        # don't need either client.
        _ensure_clients()

        start_ts = time.monotonic()

        def progress(done: int, total: int, mentions: int, _start_ts: float = start_ts) -> None:
            # Bind start_ts via default arg — B023: a closure reference
            # to the loop variable would all share the LAST iteration's
            # start_ts when this function is later invoked.
            elapsed = time.monotonic() - _start_ts
            rate = elapsed / done if done else 0.0
            click.echo(
                f"    chunk [{done}/{total}] mentions={mentions} ({rate:.1f}s/chunk)",
                err=True,
            )

        def warn(msg: str) -> None:
            click.echo(f"    warning: {msg}", err=True)

        report = mine_toponym_mentions_tiered(
            primary_client,
            fallback_client,
            source_id,
            body,
            target_chunk_size=chunk_size,
            limit=limit,
            hallucination_fallback_threshold=hallucination_fallback_threshold,
            on_chunk_done=progress,
            log_warning=warn,
        )

        # Atomic write: temp file + rename so an interrupt mid-write
        # leaves the previous output (if any) intact rather than
        # truncated. Same pattern as Phase 2b.1's candidates-out.
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as sink:
            for m in report.mentions:
                sink.write(_line(source_id, m) + "\n")
        tmp_path.replace(out_path)

        # halluc_rescued shape: succeeded/attempted. Gap (succeeded <
        # attempted) is the broken-fallback signal — could be every
        # fallback call erroring (bad API key, network outage) OR
        # every fallback returning empty (content-policy refusal,
        # all-hallucinated response). Either way, the primary's
        # hallucinations weren't actually caught. wyrd-z8mq R1
        # silent-failure-hunter HIGH + R2 follow-on.
        click.echo(
            f"  → {out_path} | chunks={report.chunks_processed} "
            f"(recovered={report.chunks_recovered_by_fallback} "
            f"halluc_rescued={report.chunks_hallucination_rescue_succeeded}/"
            f"{report.chunks_hallucination_rescue_attempted} "
            f"failed={report.chunks_failed}) mentions={len(report.mentions)} "
            f"halluc={report.hallucinations_dropped} "
            f"clamped={report.years_clamped}",
            err=True,
        )

        totals["sources_processed"] += 1
        totals["chunks_processed"] += report.chunks_processed
        totals["chunks_recovered"] += report.chunks_recovered_by_fallback
        totals["chunks_hallucination_rescue_attempted"] += (
            report.chunks_hallucination_rescue_attempted
        )
        totals["chunks_hallucination_rescue_succeeded"] += (
            report.chunks_hallucination_rescue_succeeded
        )
        totals["chunks_failed"] += report.chunks_failed
        totals["hallucinations_dropped"] += report.hallucinations_dropped
        totals["years_clamped"] += report.years_clamped
        totals["mentions"] += len(report.mentions)

    click.echo("", err=True)
    click.echo(
        f"TOTAL sources={totals['sources_processed']} "
        f"(skipped={totals['sources_skipped']}) "
        f"chunks={totals['chunks_processed']} "
        f"recovered_by_fallback={totals['chunks_recovered']} "
        f"halluc_rescued={totals['chunks_hallucination_rescue_succeeded']}/"
        f"{totals['chunks_hallucination_rescue_attempted']} "
        f"failed={totals['chunks_failed']} "
        f"halluc={totals['hallucinations_dropped']} "
        f"clamped={totals['years_clamped']} "
        f"mentions={totals['mentions']}",
        err=True,
    )


def _read_failures_jsonl(path: Path) -> list[dict]:
    """Read a captured-failures JSONL into a list of dict records.

    Required fields: source_id (str), chunk_index (int), chunk_body
    (non-empty str). Optional fields: error, extractor. Rejects
    malformed JSON, non-dict rows, missing required fields, null or
    wrong-type required values, and empty ``chunk_body`` (the resume
    path would silently process an empty string as a chunk otherwise —
    FailedChunk.chunk_body defaults to "" for back-compat but
    staged-cascade input must always carry the real chunk text)."""
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_num, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as e:
                raise click.ClickException(f"{path}:{line_num}: not valid JSON: {e}") from e
            if not isinstance(rec, dict):
                raise click.ClickException(
                    f"{path}:{line_num}: expected JSON object, got {type(rec).__name__}"
                )
            for key in ("source_id", "chunk_index", "chunk_body"):
                if key not in rec:
                    raise click.ClickException(f"{path}:{line_num}: missing required field {key!r}")
                if rec[key] is None:
                    raise click.ClickException(f"{path}:{line_num}: required field {key!r} is null")
            # Type-check required fields. ``source_id`` as a list/dict
            # would blow up at ``by_source.setdefault(rec["source_id"],
            # …)`` with an unhashable-type TypeError far downstream;
            # ``chunk_index`` as a string ("5") would silently coerce
            # through ``int()`` while True/False (subclasses of int)
            # would silently extract chunk 0 or chunk 1; ``chunk_body``
            # as a list would AttributeError on ``.strip()`` mid-loop.
            # Reject all three loudly at read time. R2 silent-failure-
            # hunter MEDIUM.
            if not isinstance(rec["source_id"], str):
                raise click.ClickException(
                    f"{path}:{line_num}: source_id must be a string, got "
                    f"{type(rec['source_id']).__name__}"
                )
            # bool is a subclass of int in Python — reject explicitly
            # so True/False values aren't silently coerced to 1/0.
            if not isinstance(rec["chunk_index"], int) or isinstance(rec["chunk_index"], bool):
                raise click.ClickException(
                    f"{path}:{line_num}: chunk_index must be an integer, got "
                    f"{type(rec['chunk_index']).__name__}"
                )
            if not isinstance(rec["chunk_body"], str):
                raise click.ClickException(
                    f"{path}:{line_num}: chunk_body must be a string, got "
                    f"{type(rec['chunk_body']).__name__}"
                )
            if not rec["chunk_body"].strip():
                raise click.ClickException(
                    f"{path}:{line_num}: chunk_body is empty — staged-cascade resume "
                    f"can't re-extract from no text (was this captured before wyrd-srd2?)"
                )
            records.append(rec)
    return records


def _load_existing_mention_keys(
    out_path: Path,
    log_warning: Callable[[str], None] | None = None,
) -> tuple[set[tuple], int]:
    """Build the dedup-key set from an existing per-source output JSONL.
    Returns ``(keys, malformed_count)``. Empty set + 0 if the file
    doesn't exist. The key shape mirrors what the writer emits —
    staged cascade re-runs use this to skip already-emitted
    attestations when appending new mentions.

    Non-dict rows and malformed-JSON rows are skipped (defensive: a
    partially-written file after a crash shouldn't bomb the next
    resume), but the ``malformed_count`` return + optional
    log_warning surface them to the operator so a corrupted file
    doesn't quietly grow duplicate rows. R2 silent-failure-hunter
    MEDIUM."""
    if not out_path.exists():
        return set(), 0
    keys: set[tuple] = set()
    malformed = 0
    with out_path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if not isinstance(row, dict):
                malformed += 1
                continue
            keys.add(
                (
                    row.get("form"),
                    row.get("date_year"),
                    row.get("region_hint"),
                    row.get("context"),
                )
            )
    if malformed and log_warning is not None:
        log_warning(
            f"{malformed} malformed row(s) in {out_path} can't be deduped "
            f"against — new mentions matching them will appear as new"
        )
    return keys, malformed


def _make_chunk_callbacks(source_id: str, start_ts: float, emit_failure):
    """Build the (progress, warn, on_chunk_failed) callback trio shared
    by both fresh-mining and resume-from-failures source loops. Hoisted
    out of the loop body so the late-binding default-arg trick that
    captures loop variables can be replaced with real parameter
    bindings — clearer, and forces a single source of truth for the
    log line shape (the project's mining-progress convention from
    CLAUDE.md)."""

    def progress(done: int, total: int, mentions: int) -> None:
        elapsed = time.monotonic() - start_ts
        rate = elapsed / done if done else 0.0
        click.echo(
            f"    chunk [{done}/{total}] mentions={mentions} ({rate:.1f}s/chunk)",
            err=True,
        )

    def warn(msg: str) -> None:
        click.echo(f"    warning: {msg}", err=True)

    def on_fail(fc) -> None:
        emit_failure(source_id, fc)

    return progress, warn, on_fail


@lexicon.command("mine-toponym-mentions-staged")
@click.option(
    "--sources-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("sources"),
    show_default=True,
    help="Directory of <source_id>.txt scholar bodies. Used in fresh-mining "
    "mode (when --from-failures is not given).",
)
@click.option(
    "--source",
    "sources",
    multiple=True,
    help="Restrict to named source_ids (repeat for multiple). Default in "
    "fresh-mining mode: every *.txt under --sources-dir. Ignored in "
    "--from-failures mode.",
)
@click.option(
    "--from-failures",
    "from_failures",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Read pre-chunked (source_id, chunk_index, chunk_body) records from "
    "this JSONL (typically the --capture-failures output of an earlier stage) "
    "and re-process those chunks with the configured provider. Mutually "
    "exclusive with --sources-dir-based fresh mining.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/mining/phase2"),
    show_default=True,
    help="Directory for <source_id>.jsonl extraction output. Created if "
    "missing. In --from-failures mode, mentions are APPENDED to existing "
    "per-source files with deduplication on (form, date_year, region_hint, "
    "context); idempotent re-runs.",
)
@click.option(
    "--capture-failures",
    "capture_failures",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write a JSONL record for every chunk that failed at this stage "
    "(transport, JSON parse, schema mismatch). Each record has source_id, "
    "chunk_index, chunk_body, error, extractor — ready to feed a downstream "
    "stage via --from-failures. Required only to enable a downstream cascade "
    "stage; omit if this is the terminal stage.",
)
@click.option(
    "--provider",
    type=click.Choice(["anthropic", "gemini", "ollama"]),
    default="ollama",
    show_default=True,
    help="LLM backend for this stage.",
)
@click.option(
    "--model",
    default=None,
    help="Override model id (provider-specific default if omitted).",
)
@click.option(
    "--ollama-url",
    default=None,
    help="Override Ollama base URL when --provider ollama (default: "
    "$WYRD_OLLAMA_URL or localhost).",
)
@click.option(
    "--chunk-size",
    type=int,
    default=3000,
    show_default=True,
    help="Target chunk size in characters (snaps to paragraph boundaries). "
    "In --from-failures mode chunks are pre-sliced; the value is then used "
    "only for the oversize-paragraph warning threshold.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Process only the first N chunks per source (smoke testing).",
)
@click.option(
    "--skip-existing",
    is_flag=True,
    default=False,
    help="In fresh-mining mode, skip sources whose output JSONL already exists in --output-dir.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="In fresh-mining mode, overwrite existing per-source output files "
    "(default: error). Mutually exclusive with --skip-existing.",
)
def lexicon_mine_toponym_mentions_staged(
    sources_dir: Path,
    sources: tuple[str, ...],
    from_failures: Path | None,
    output_dir: Path,
    capture_failures: Path | None,
    provider: str,
    model: str | None,
    ollama_url: str | None,
    chunk_size: int,
    limit: int | None,
    skip_existing: bool,
    force: bool,
) -> None:
    """Stage-at-a-time LLM mention extraction with failure capture (wyrd-srd2).

    Two operating modes share this command:

    1. **Fresh-mining mode** (default): walks every source_id given via
       --source (or every *.txt under --sources-dir when --source is
       omitted). For each source, mines chunks with the configured
       provider+model, writes successful mentions to
       --output-dir/<source_id>.jsonl, captures failed chunks to
       --capture-failures (if given).

    2. **Resume-from-failures mode** (--from-failures FILE): reads
       pre-chunked records from FILE and re-processes those chunks with
       the configured provider+model. Mentions are APPENDED to existing
       per-source output files with deduplication; chunks that fail at
       this stage flow into --capture-failures for the next stage.

    Operator-paced cascade:

        # Stage 1: free local model on everything
        wyrd kenning lexicon mine-toponym-mentions-staged \\
            --provider ollama --model gemma4:26b --chunk-size 3000 \\
            --output-dir data/mining/phase2 \\
            --capture-failures data/mining/phase2_failures/stage1.jsonl

        # Inspect: wc -l data/mining/phase2_failures/stage1.jsonl

        # Stage 2: cheap cloud model on the residual
        wyrd kenning lexicon mine-toponym-mentions-staged \\
            --from-failures data/mining/phase2_failures/stage1.jsonl \\
            --provider gemini --model gemini-2.5-flash \\
            --output-dir data/mining/phase2 \\
            --capture-failures data/mining/phase2_failures/stage2.jsonl

        # Stage 3: more capable model on the still-failed
        wyrd kenning lexicon mine-toponym-mentions-staged \\
            --from-failures data/mining/phase2_failures/stage2.jsonl \\
            --provider anthropic --model claude-haiku-4-5-20251001 \\
            --output-dir data/mining/phase2 \\
            --capture-failures data/mining/phase2_failures/stage3.jsonl

    Output JSONL rows include ``extractor:"provider:model"`` so post-hoc
    analysis can attribute each mention to the stage that captured it.
    """
    from wyrd.generators.kenning.toponym_mention_extractor import (
        ToponymMention,
        mine_toponym_mentions,
        mine_toponym_mentions_from_chunks,
    )

    if from_failures is not None and sources:
        raise click.ClickException("--from-failures and --source are mutually exclusive")
    if from_failures is not None and (skip_existing or force):
        # --skip-existing and --force only make sense in fresh-mining
        # mode (they gate per-source output existence). Resume-mode
        # always APPENDS with dedup against existing rows. Silently
        # ignoring the flags would mislead an operator who passed
        # `--from-failures … --force` expecting overwrite.
        bad = "--skip-existing" if skip_existing else "--force"
        raise click.ClickException(
            f"{bad} is only valid in fresh-mining mode (cannot combine with --from-failures)"
        )
    if skip_existing and force:
        raise click.ClickException("--skip-existing and --force are mutually exclusive")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Lazy client construction (matches the tiered command's pattern):
    # a --skip-existing resume where every source is skipped shouldn't
    # require ANTHROPIC_API_KEY validation or an Ollama probe.
    client_box: dict = {"client": None, "extractor": f"{provider}:{model or 'default'}"}

    def _ensure_client():
        if client_box["client"] is None:
            built = _build_extractor_client(provider, model, ollama_url=ollama_url)
            # Refresh extractor name now that we know the resolved
            # model id — JSONL rows pin the actual model that ran,
            # not the placeholder "default".
            client_box["client"] = built
            client_box["extractor"] = f"{provider}:{built.model}"

    failure_sink = None
    if capture_failures is not None:
        capture_failures.parent.mkdir(parents=True, exist_ok=True)
        # Stale-records warning: appending to a non-empty failures file
        # is legitimate (an operator may be incrementally building one),
        # but if the file came from a DIFFERENT cascade run the stale
        # records would silently flow into the next stage's input. Warn
        # so the operator can confirm/clear before continuing. wyrd-srd2
        # R1 silent-failure-hunter HIGH.
        if capture_failures.exists() and capture_failures.stat().st_size > 0:
            # Stream-count rather than read_text() so a large stale
            # failures file doesn't load into memory just to compute
            # the warning count.
            with capture_failures.open("r", encoding="utf-8") as _fh:
                stale = sum(1 for ln in _fh if ln.strip())
            click.echo(
                f"  warning: --capture-failures {capture_failures} already has "
                f"{stale} record(s); appending (`> {capture_failures}` to clear)",
                err=True,
            )
        # Append-mode: re-running the same stage doesn't blow away the
        # tail of the prior run. The operator can clear the file
        # manually if a fresh slate is wanted (see warning above).
        failure_sink = capture_failures.open("a", encoding="utf-8")

    def _emit_failure(source_id, fc):
        if failure_sink is None:
            return
        rec = {
            "source_id": source_id,
            "chunk_index": fc.index,
            "chunk_body": fc.chunk_body,
            "error": fc.error,
            "extractor": client_box["extractor"],
        }
        failure_sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
        failure_sink.flush()

    def _line(source_id: str, m: ToponymMention) -> str:
        return json.dumps(
            {
                "source_id": source_id,
                "form": m.form,
                "date_year": m.date_year,
                "region_hint": m.region_hint,
                "context": m.context,
                "extractor": client_box["extractor"],
            },
            ensure_ascii=False,
        )

    totals = {
        "sources_processed": 0,
        "sources_skipped": 0,
        "chunks_processed": 0,
        "chunks_failed": 0,
        "hallucinations_dropped": 0,
        "years_clamped": 0,
        "mentions": 0,
        "mentions_deduped": 0,
        "malformed_purged": 0,
    }

    try:
        if from_failures is not None:
            _run_resume_from_failures(
                from_failures=from_failures,
                output_dir=output_dir,
                provider=provider,
                model=model,
                chunk_size=chunk_size,
                limit=limit,
                client_box=client_box,
                ensure_client=_ensure_client,
                emit_failure=_emit_failure,
                line_fn=_line,
                totals=totals,
                mine_fn=mine_toponym_mentions_from_chunks,
            )
        else:
            _run_fresh_mining(
                sources=sources,
                sources_dir=sources_dir,
                output_dir=output_dir,
                provider=provider,
                model=model,
                chunk_size=chunk_size,
                limit=limit,
                skip_existing=skip_existing,
                force=force,
                client_box=client_box,
                ensure_client=_ensure_client,
                emit_failure=_emit_failure,
                line_fn=_line,
                totals=totals,
                mine_fn=mine_toponym_mentions,
            )
    finally:
        if failure_sink is not None:
            failure_sink.close()

    click.echo("", err=True)
    summary = (
        f"TOTAL sources={totals['sources_processed']} "
        f"(skipped={totals['sources_skipped']}) "
        f"chunks={totals['chunks_processed']} "
        f"failed={totals['chunks_failed']} "
        f"halluc={totals['hallucinations_dropped']} "
        f"clamped={totals['years_clamped']} "
        f"mentions={totals['mentions']}"
    )
    if from_failures is not None:
        summary += f" deduped={totals['mentions_deduped']}"
        if totals["malformed_purged"]:
            summary += f" malformed_purged={totals['malformed_purged']}"
    click.echo(summary, err=True)
    # Always surface the capture-failures status — operators want to
    # know whether the next stage has work to do (or whether the file
    # is now stale and should be cleared). wyrd-srd2 R1 silent-failure
    # MEDIUM: silent "0 failures, file unchanged" path was ambiguous.
    if capture_failures is not None:
        if totals["chunks_failed"] > 0:
            click.echo(
                f"  {totals['chunks_failed']} failed chunk(s) appended → {capture_failures}",
                err=True,
            )
        else:
            click.echo(
                f"  0 new failures → {capture_failures} unchanged this run",
                err=True,
            )


def _run_resume_from_failures(
    *,
    from_failures: Path,
    output_dir: Path,
    provider: str,
    model: str | None,
    chunk_size: int,
    limit: int | None,
    client_box: dict,
    ensure_client,
    emit_failure,
    line_fn,
    totals: dict,
    mine_fn,
) -> None:
    """Resume-from-failures mode body, extracted out of the click command
    so the dispatcher stays narrow. mine_fn is injected for testability
    (matches mine_toponym_mentions_from_chunks at the only call site)."""
    records = _read_failures_jsonl(from_failures)
    if not records:
        # Fall through to summary (don't early-return): operator wants
        # the unambiguous "sources=0 mentions=0" line even on a no-op.
        click.echo(
            f"--from-failures {from_failures} contained no records; nothing to do",
            err=True,
        )
        return
    by_source: dict[str, list[tuple[int, str]]] = {}
    for rec in records:
        by_source.setdefault(rec["source_id"], []).append(
            (int(rec["chunk_index"]), rec["chunk_body"])
        )
    # Sort each source's chunks by original index so progress lines
    # look chronological. Sort sources alphabetically for determinism
    # across resume invocations.
    for src in by_source:
        by_source[src].sort(key=lambda t: t[0])
    click.echo(
        f"Resume from {from_failures}: {len(records)} chunks across {len(by_source)} source(s)",
        err=True,
    )
    click.echo(f"Provider: {provider} (model={model or 'default'})", err=True)

    for src_i, source_id in enumerate(sorted(by_source), start=1):
        indexed_chunks = by_source[source_id]
        if limit is not None:
            indexed_chunks = indexed_chunks[:limit]
        ensure_client()
        out_path = output_dir / f"{source_id}.jsonl"
        start_ts = time.monotonic()
        progress, warn, on_fail = _make_chunk_callbacks(source_id, start_ts, emit_failure)
        existing_keys, malformed = _load_existing_mention_keys(out_path, log_warning=warn)
        click.echo(
            f"[{src_i}/{len(by_source)}] {source_id}: "
            f"{len(indexed_chunks)} chunk(s) to retry "
            f"({len(existing_keys)} existing mentions"
            + (f", {malformed} malformed" if malformed else "")
            + ")",
            err=True,
        )

        report = mine_fn(
            client_box["client"],
            source_id,
            indexed_chunks,
            target_chunk_size=chunk_size,
            on_chunk_done=progress,
            on_chunk_failed=on_fail,
            log_warning=warn,
        )

        # Atomic-write append: build the union (existing + new-deduped)
        # in a .tmp file, then replace the original. A killed process
        # mid-write leaves the original intact, vs. a direct
        # ``open("a")`` which could leave a half-written final line
        # that downstream commands would silently drop or stumble on.
        # wyrd-srd2 R1 silent-failure-hunter HIGH.
        new_count = 0
        dup_count = 0
        purged_count = 0
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as sink:
            if out_path.exists():
                # Preserve well-formed existing rows verbatim (don't
                # re-serialize: stage-specific fields like ``extractor``
                # must keep their original value through the cascade).
                # DROP malformed rows (non-JSON, non-dict) — without
                # this the corruption persists every resume and the
                # file's bad-row count grows forever. R3 silent-failure
                # MEDIUM. ``_load_existing_mention_keys`` already
                # surfaced the malformed_count to the operator; here
                # we also remove them.
                with out_path.open("r", encoding="utf-8") as src:
                    for line in src:
                        stripped = line.rstrip("\n")
                        if not stripped:
                            continue
                        try:
                            row = json.loads(stripped)
                        except json.JSONDecodeError:
                            purged_count += 1
                            continue
                        if not isinstance(row, dict):
                            purged_count += 1
                            continue
                        sink.write(stripped + "\n")
            for m in report.mentions:
                key = (m.form, m.date_year, m.region_hint, m.context)
                if key in existing_keys:
                    dup_count += 1
                    continue
                sink.write(line_fn(source_id, m) + "\n")
                existing_keys.add(key)
                new_count += 1
        tmp_path.replace(out_path)
        if purged_count:
            click.echo(
                f"    purged {purged_count} malformed row(s) from {out_path}",
                err=True,
            )

        click.echo(
            f"  → {out_path} | chunks={report.chunks_processed} "
            f"failed={report.chunks_failed} new_mentions={new_count} "
            f"deduped={dup_count} halluc={report.hallucinations_dropped} "
            f"clamped={report.years_clamped}",
            err=True,
        )

        totals["sources_processed"] += 1
        totals["chunks_processed"] += report.chunks_processed
        totals["chunks_failed"] += report.chunks_failed
        totals["hallucinations_dropped"] += report.hallucinations_dropped
        totals["years_clamped"] += report.years_clamped
        totals["mentions"] += new_count
        totals["mentions_deduped"] += dup_count
        totals["malformed_purged"] += purged_count


def _run_fresh_mining(
    *,
    sources: tuple[str, ...],
    sources_dir: Path,
    output_dir: Path,
    provider: str,
    model: str | None,
    chunk_size: int,
    limit: int | None,
    skip_existing: bool,
    force: bool,
    client_box: dict,
    ensure_client,
    emit_failure,
    line_fn,
    totals: dict,
    mine_fn,
) -> None:
    """Fresh-mining mode body, extracted out of the click command. mine_fn
    is injected (matches mine_toponym_mentions at the only call site)."""
    if sources:
        source_ids = []
        for sid in sources:
            txt = sources_dir / f"{Path(sid).name}.txt"
            if not txt.exists():
                raise click.ClickException(f"source body not found: {txt}")
            source_ids.append(Path(sid).name)
    else:
        source_ids = sorted(p.stem for p in sources_dir.glob("*.txt") if p.name != "MANIFEST.md")
    if not source_ids:
        raise click.ClickException(f"no sources under {sources_dir}")

    click.echo(f"Sources to process: {len(source_ids)}", err=True)
    click.echo(f"Provider: {provider} (model={model or 'default'})", err=True)

    for src_i, source_id in enumerate(source_ids, start=1):
        out_path = output_dir / f"{source_id}.jsonl"
        if out_path.exists():
            if skip_existing:
                click.echo(
                    f"[{src_i}/{len(source_ids)}] {source_id}: SKIP (output exists)",
                    err=True,
                )
                totals["sources_skipped"] += 1
                continue
            if not force:
                raise click.ClickException(
                    f"output {out_path} exists; pass --skip-existing to leave it "
                    f"or --force to overwrite"
                )

        txt_path = sources_dir / f"{source_id}.txt"
        body = txt_path.read_text(encoding="utf-8", errors="replace")
        if "�" in body:
            click.echo(
                f"  warning: {source_id} contained invalid UTF-8 sequences (now U+FFFD)",
                err=True,
            )

        click.echo(
            f"[{src_i}/{len(source_ids)}] {source_id}: {len(body):,} chars",
            err=True,
        )
        ensure_client()
        start_ts = time.monotonic()
        progress, warn, on_fail = _make_chunk_callbacks(source_id, start_ts, emit_failure)

        report = mine_fn(
            client_box["client"],
            source_id,
            body,
            target_chunk_size=chunk_size,
            limit=limit,
            on_chunk_done=progress,
            on_chunk_failed=on_fail,
            log_warning=warn,
        )

        # Atomic write: fresh per-source files use full rewrite.
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as sink:
            for m in report.mentions:
                sink.write(line_fn(source_id, m) + "\n")
        tmp_path.replace(out_path)

        click.echo(
            f"  → {out_path} | chunks={report.chunks_processed} "
            f"failed={report.chunks_failed} mentions={len(report.mentions)} "
            f"halluc={report.hallucinations_dropped} "
            f"clamped={report.years_clamped}",
            err=True,
        )

        totals["sources_processed"] += 1
        totals["chunks_processed"] += report.chunks_processed
        totals["chunks_failed"] += report.chunks_failed
        totals["hallucinations_dropped"] += report.hallucinations_dropped
        totals["years_clamped"] += report.years_clamped
        totals["mentions"] += len(report.mentions)


@lexicon.command("ingest-toponym-mentions")
@click.option(
    "--jsonl",
    "jsonl_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="JSONL file of mentions (output of mine-toponym-mentions).",
)
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
    help="Lexicon SQLite DB.",
)
@click.option(
    "--source",
    "source_id_override",
    default=None,
    help="Override the source_id used for attestation rows. Default: read "
    "the source_id field from each JSONL row.",
)
@click.option(
    "--apply",
    is_flag=True,
    default=False,
    help="Write to DB. Without this flag, runs in dry-run mode and prints "
    "the predicted insert/dup counts.",
)
@click.option(
    "--candidates-out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write unresolved mentions to this JSONL for Phase 2b.3 operator "
    "review. Default: omit (counted but not persisted).",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite --candidates-out if it exists.",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Print per-source breakdown to stderr.",
)
def lexicon_ingest_toponym_mentions(
    jsonl_path: Path,
    db_path: Path,
    source_id_override: str | None,
    apply: bool,
    candidates_out: Path | None,
    force: bool,
    verbose: bool,
) -> None:
    """Ingest LLM-extracted mentions (wyrd-x82p Phase 2b.1).

    Consumes the JSONL emitted by ``mine-toponym-mentions`` and writes
    ``toponym_attestation`` rows for mentions that resolve to a known
    toponym. Unresolved mentions are tallied (and optionally written
    to ``--candidates-out``) for the Phase 2b.3 review tool to
    triage.

    Resolver: matches form to toponym via Phase 1's form-to-toponym
    lookup, using ``region_hint`` to disambiguate homonyms when the
    bare form maps to multiple toponyms. Mentions that don't resolve
    are NOT inserted — Phase 2b.1 only emits attestations for
    toponyms that already exist; new-toponym creation is Phase 2b.3.

    Idempotent: re-running against the same JSONL is a no-op.
    """
    from wyrd.generators.kenning.toponym_mention_ingest import (
        IngestReport,
        build_resolver_indexes,
        ingest_mentions,
    )

    click.echo(f"Using DB {db_path}", err=True)

    if candidates_out is not None and candidates_out.exists() and not force:
        raise click.ClickException(
            f"--candidates-out {candidates_out} already exists; pass --force to overwrite"
        )
    if candidates_out is not None and candidates_out.exists() and force:
        click.echo(
            f"warning: --force overwriting existing {candidates_out}",
            err=True,
        )

    # Parse JSONL once and group by source. Each row carries its own
    # source_id (the JSONL format permits mixed sources), so we have
    # to read the whole file before dispatching per-source. The
    # groupings stay in memory; for the pilot scope (~thousands of
    # mentions) that's bounded, and the per-source SELECT bulk-load
    # downstream amortizes the parse cost. A future streaming
    # rewrite (Phase 2b.2 full-scope) would emit per-source
    # sub-batches and avoid the upfront groupby.
    mentions_by_source: dict[str, list[dict]] = {}
    total_lines = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise click.ClickException(f"{jsonl_path}:{line_no}: invalid JSON: {e}") from e
            if not isinstance(row, dict):
                raise click.ClickException(
                    f"{jsonl_path}:{line_no}: row is not a JSON object ({type(row).__name__})"
                )
            sid = source_id_override or row.get("source_id")
            if not sid:
                raise click.ClickException(
                    f"{jsonl_path}:{line_no}: mention missing source_id "
                    f"and no --source override given"
                )
            mentions_by_source.setdefault(sid, []).append(row)
            total_lines += 1
            if total_lines % 10000 == 0:
                # Project mining-progress convention from CLAUDE.md —
                # surface load progress for large files.
                click.echo(
                    f"  loaded {total_lines:,} mentions...",
                    err=True,
                )

    click.echo(
        f"Loaded {total_lines:,} mentions across {len(mentions_by_source)} source(s)",
        err=True,
    )

    totals = IngestReport(source_id="TOTAL")
    all_unresolved: list[dict] = []

    with LexiconDB(db_path) as db:
        indexes = build_resolver_indexes(db.conn)
        click.echo(
            f"Built resolver indexes: {len(indexes.form_to_ids):,} forms, "
            f"{sum(1 for r in indexes.toponym_regions.values() if r):,} "
            f"toponyms with region",
            err=True,
        )
        source_count = len(mentions_by_source)
        for src_i, (sid, mentions) in enumerate(sorted(mentions_by_source.items()), start=1):
            report = ingest_mentions(db.conn, sid, mentions, indexes, apply=apply)
            totals.mentions_processed += report.mentions_processed
            totals.resolved += report.resolved
            totals.unresolved += report.unresolved
            totals.inserted += report.inserted
            totals.already_present += report.already_present
            all_unresolved.extend(report.unresolved_records)
            # Project mining-progress convention: per-source line to
            # stderr (always, not only --verbose) so a multi-source
            # run shows progress.
            click.echo(
                f"  [{src_i}/{source_count}] {sid}: "
                f"processed={report.mentions_processed} "
                f"resolved={report.resolved} unresolved={report.unresolved} "
                f"inserted={report.inserted} dup={report.already_present}",
                err=True,
            )
            if verbose and report.unresolved_records:
                # In verbose mode, show a few unresolved samples per
                # source so the operator can sense-check the failure
                # mode without opening the candidates file.
                for rec in report.unresolved_records[:5]:
                    click.echo(
                        f"      unresolved: {rec['form']!r} region_hint={rec['region_hint']!r}",
                        err=True,
                    )
        if apply:
            db.conn.commit()

    click.echo("", err=True)
    click.echo(
        f"TOTAL processed={totals.mentions_processed} "
        f"resolved={totals.resolved} "
        f"unresolved={totals.unresolved} "
        f"inserted={totals.inserted} "
        f"dup={totals.already_present} "
        f"({'APPLIED' if apply else 'dry-run'})",
        err=True,
    )

    if candidates_out is not None:
        # Atomic write: write to a sibling tempfile then rename, so a
        # mid-write crash leaves the prior file intact rather than
        # truncated. silent-failure round-1 finding (the JSONL is the
        # only artifact of the unresolved set; an interrupted write
        # under --force was a real foot-gun).
        tmp_path = candidates_out.with_suffix(candidates_out.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as sink:
            for rec in all_unresolved:
                sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp_path.replace(candidates_out)
        click.echo(
            f"Wrote {len(all_unresolved)} unresolved → {candidates_out}",
            err=True,
        )


@lexicon.command("prepare-toponym-candidates")
@click.option(
    "--jsonl",
    "candidates_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="JSONL of unresolved candidates (output of ingest-toponym-mentions --candidates-out).",
)
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
    help="Lexicon SQLite DB.",
)
@click.option(
    "--out",
    "triage_path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Write triage JSONL to this path. Operator hand-edits this file, "
    "then feeds it to commit-toponym-candidates.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite --out if it exists.",
)
def lexicon_prepare_toponym_candidates(
    candidates_path: Path,
    db_path: Path,
    triage_path: Path,
    force: bool,
) -> None:
    """Prepare a triage JSONL from Phase 2b.1's unresolved-candidates
    output (wyrd-x82p Phase 2b.3).

    For each candidate, attaches the top-3 fuzzy-matched existing
    toponyms (by SequenceMatcher.ratio over normalized forms), with
    a region-hint boost when the candidate carries one. Emits a
    JSONL where each row pre-fills ``action: "defer"`` —
    operator edits the action to ``"map"`` (with ``toponym_id``),
    ``"create"`` (with ``create_modern_name`` / ``create_country``
    / ``create_region``), or ``"skip"``, then feeds the file to
    ``commit-toponym-candidates``.
    """
    from wyrd.generators.kenning.toponym_candidate_review import (
        _toponym_rows,
        prepare_candidate,
    )

    if triage_path.exists() and not force:
        raise click.ClickException(f"--out {triage_path} already exists; pass --force to overwrite")
    if triage_path.exists() and force:
        click.echo(
            f"warning: --force overwriting existing {triage_path}",
            err=True,
        )

    click.echo(f"Using DB {db_path}", err=True)
    with LexiconDB(db_path) as db:
        toponyms = _toponym_rows(db.conn)
    click.echo(f"Loaded {len(toponyms):,} toponyms for fuzzy match", err=True)

    # Stream the candidates: read input one line at a time, write
    # triage output as we go. Atomic-rename pattern (temp + replace)
    # so a mid-write crash leaves any prior triage file intact.
    tmp_path = triage_path.with_suffix(triage_path.suffix + ".tmp")
    n_prepared = 0
    n_with_suggestions = 0
    with (
        candidates_path.open("r", encoding="utf-8") as src,
        tmp_path.open("w", encoding="utf-8") as sink,
    ):
        for line_no, line in enumerate(src, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                raise click.ClickException(f"{candidates_path}:{line_no}: invalid JSON: {e}") from e
            if not isinstance(raw, dict):
                raise click.ClickException(f"{candidates_path}:{line_no}: row is not a JSON object")
            prepared = prepare_candidate(raw, toponyms)
            row_out = {
                "source_id": prepared.source_id,
                "form": prepared.form,
                "date_year": prepared.date_year,
                "region_hint": prepared.region_hint,
                "context": prepared.context,
                "suggestions": [
                    {
                        "toponym_id": s.toponym_id,
                        "modern_name": s.modern_name,
                        "region": s.region,
                        "country": s.country,
                        "fuzzy_score": s.fuzzy_score,
                    }
                    for s in prepared.suggestions
                ],
                "action": prepared.action,
                "toponym_id": prepared.toponym_id,
                "create_modern_name": prepared.create_modern_name,
                "create_country": prepared.create_country,
                "create_region": prepared.create_region,
            }
            sink.write(json.dumps(row_out, ensure_ascii=False) + "\n")
            n_prepared += 1
            if prepared.suggestions:
                n_with_suggestions += 1
            if n_prepared % 1000 == 0:
                click.echo(f"  prepared {n_prepared:,} candidates...", err=True)
    tmp_path.replace(triage_path)
    click.echo(f"Wrote {n_prepared:,} triage rows → {triage_path}", err=True)
    click.echo(
        f"  with fuzzy suggestions: {n_with_suggestions:,} "
        f"({n_prepared - n_with_suggestions:,} likely new-toponym candidates)",
        err=True,
    )


@lexicon.command("commit-toponym-candidates")
@click.option(
    "--jsonl",
    "triage_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Operator-edited triage JSONL (from prepare-toponym-candidates).",
)
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
    help="Lexicon SQLite DB.",
)
@click.option(
    "--apply",
    is_flag=True,
    default=False,
    help="Write to DB. Default: dry-run reports predicted counts only.",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Print per-row decisions to stderr (one line per non-defer row).",
)
def lexicon_commit_toponym_candidates(
    triage_path: Path,
    db_path: Path,
    apply: bool,
    verbose: bool,
) -> None:
    """Apply the operator's triage decisions (wyrd-x82p Phase 2b.3).

    Reads the edited triage JSONL. For each row:

    * ``action: map`` — writes a ``toponym_attestation`` row pointing
      at ``toponym_id``.
    * ``action: create`` — creates a new ``toponym`` row from
      ``create_modern_name`` / ``create_country`` / ``create_region``
      and a matching attestation. If the (name, country, region)
      tuple already exists (UNIQUE-index collision), the decision is
      treated as a MAP to the existing id — prevents accidental
      duplicate toponym rows.
    * ``action: skip`` — no DB write.
    * ``action: defer`` — no DB write; left for a future pass.

    Idempotent for the ``map`` path via the UNIQUE index on
    ``toponym_attestation``. Re-running with the same triage JSONL
    is a no-op for already-applied rows.
    """
    from wyrd.generators.kenning.toponym_candidate_review import (
        commit_triage_decisions,
    )

    click.echo(f"Using DB {db_path}", err=True)
    # Read the whole JSONL into memory — pilot scale is hundreds to
    # thousands of rows; streaming isn't worth the complexity here.
    rows: list[dict] = []
    with triage_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise click.ClickException(f"{triage_path}:{line_no}: invalid JSON: {e}") from e
    click.echo(f"Loaded {len(rows):,} triage rows", err=True)

    with LexiconDB(db_path) as db:
        report = commit_triage_decisions(db.conn, rows, apply=apply)
        if apply:
            db.conn.commit()

    if verbose:
        for idx, msg in report.error_records:
            click.echo(f"  row {idx}: ERROR — {msg}", err=True)
        # Surface CREATE → MAP demotions per row so the operator
        # sees which of their CREATEs collided with an existing
        # toponym (silently demoting was a load-bearing UX gap
        # otherwise).
        for idx, tid, name in report.demoted_records:
            click.echo(
                f"  row {idx}: CREATE→MAP — {name!r} collides with existing toponym {tid}",
                err=True,
            )

    # Warn when most rows are still at the default-defer placeholder:
    # operator may have run commit on an unedited triage file. Gated
    # on processed >= 5 so a single-row "I deferred this on purpose"
    # invocation doesn't get a noisy 100% warning.
    if report.deferred > 0 and report.processed >= 5:
        defer_ratio = report.deferred / report.processed
        if defer_ratio >= 0.8:
            click.echo(
                f"warning: {report.deferred}/{report.processed} "
                f"({defer_ratio:.0%}) rows are still at action=defer — "
                f"if this is unintended, the triage file may not have been edited yet",
                err=True,
            )

    click.echo("", err=True)
    click.echo(
        f"TOTAL processed={report.processed} "
        f"mapped={report.mapped} "
        f"created={report.created} "
        f"demoted={report.demoted_count} "
        f"skipped={report.skipped} "
        f"deferred={report.deferred} "
        f"errors={report.errors} "
        f"({'APPLIED' if apply else 'dry-run'})",
        err=True,
    )
    if report.errors and not verbose:
        click.echo(
            "  (re-run with --verbose to see per-row error messages)",
            err=True,
        )


@lexicon.command("audit-etymology-alignment")
@click.option(
    "--dir",
    "directory",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data/mining"),
    show_default=True,
    help="Directory of <source>.jsonl files to scan.",
)
@click.option(
    "--source",
    "sources",
    multiple=True,
    help="Restrict to named source_ids (repeat for multiple).",
)
@click.option(
    "--samples",
    "sample_limit",
    type=int,
    default=3,
    show_default=True,
    help="Max sample findings to print per source.",
)
@click.option(
    "--top",
    "top_n",
    type=int,
    default=20,
    show_default=True,
    help="Max sources to include in the report.",
)
def lexicon_audit_etymology_alignment(
    directory: Path, sources: tuple[str, ...], sample_limit: int, top_n: int
) -> None:
    """Audit toponym/etymology cross-misalignment (wyrd-8upf).

    Walks every <source>.jsonl in DIRECTORY, flags etymology_element
    rows whose element list doesn't appear supported by the
    historical_form reconstruction. Markdown report on stdout.
    """
    from wyrd.generators.kenning.etymology_alignment_audit import (
        audit_jsonl_dir,
        format_audit_report,
    )

    report = audit_jsonl_dir(
        directory,
        sample_limit=sample_limit,
        sources=sources or None,
    )
    click.echo(format_audit_report(report, top_n=top_n))


@lexicon.command("rando-port-readiness")
@click.option(
    "--bundle",
    "bundle_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("wyrd/generators/kenning/data/meanings.json"),
    show_default=True,
    help="Path to meanings.json (the runtime bundle).",
)
@click.option(
    "--coverage-threshold",
    type=float,
    default=0.80,
    show_default=True,
    help="Per-language scholar+empirical attestation threshold (criterion 1).",
)
@click.option(
    "--language",
    "languages",
    multiple=True,
    help="Restrict to named target bundle siblings (repeat for multiple). "
    "Default: old_english, old_french, old_scandinavian, celtic_mix, latin. "
    "Names use the bundle's underscore form (Welsh/Irish are conflated under "
    "celtic_mix in the current bundle).",
)
def lexicon_rando_port_readiness(
    bundle_path: Path,
    coverage_threshold: float,
    languages: tuple[str, ...],
) -> None:
    """Report whether the rando-port retirement gate is open (wyrd-j2bv).

    Three criteria per target language:
      1. Scholar + empirical coverage ≥ threshold (default 80%)
      2. Zero rando-only bundle subjects
      3. Rando-only count ≤ scholar-attested count

    Exits 0 if all pass (gate OPEN, retirement can proceed); exit 1
    if any fail (gate CLOSED).
    """
    from wyrd.generators.kenning.rando_port_readiness import (
        DEFAULT_TARGET_LANGUAGES,
        compute_readiness,
        format_readiness,
        load_bundle,
    )

    target = languages or DEFAULT_TARGET_LANGUAGES
    bundle = load_bundle(bundle_path)
    report = compute_readiness(
        bundle, target_languages=target, coverage_threshold=coverage_threshold
    )
    click.echo(format_readiness(report))
    if not report.overall_passes:
        raise SystemExit(1)


@lexicon.command("fetch-bulk-sources")
@click.option(
    "--slice",
    "slice_names",
    multiple=True,
    help="Restrict to named slices (repeatable). Default: all slices in manifest.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Re-download even when local cache sha256 matches the manifest.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Report what would happen; don't touch S3 or the local cache.",
)
def lexicon_fetch_bulk_sources(slice_names: tuple[str, ...], force: bool, dry_run: bool) -> None:
    """Populate ~/.wyrd/sources/ from S3 (wyrd-0vj3).

    Reads data/mining/_bulk_manifest.json, downloads any slice whose
    local cache file is missing or whose sha256 doesn't match.
    """
    from wyrd.generators.kenning.bulk_sources import (
        fetch_missing_slices,
        load_config,
        load_manifest,
    )

    manifest = load_manifest()
    config = load_config(manifest)
    result = fetch_missing_slices(
        manifest,
        config,
        slice_names=list(slice_names) if slice_names else None,
        force=force,
        dry_run=dry_run,
    )
    if dry_run:
        click.echo("(dry-run; nothing written)", err=True)
    click.echo(f"Fetched: {len(result.fetched)}", err=True)
    for name in result.fetched:
        click.echo(f"  + {name}", err=True)
    click.echo(f"Skipped: {len(result.skipped)}  (already current)", err=True)
    if result.failed:
        click.echo(f"FAILED:  {len(result.failed)}", err=True)
        for name, reason in result.failed:
            click.echo(f"  ! {name}: {reason}", err=True)
        raise SystemExit(1)


@lexicon.command("push-bulk-sources")
@click.option(
    "--slice",
    "slice_names",
    multiple=True,
    help="Restrict to named slices (repeatable). Default: every slice in manifest.",
)
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/mining/_bulk_manifest.json"),
    show_default=True,
)
def lexicon_push_bulk_sources(slice_names: tuple[str, ...], manifest_path: Path) -> None:
    """Upload ~/.wyrd/sources/ slices to S3 and rewrite the manifest
    (wyrd-0vj3).

    For each manifest slice: if a <slice>.jsonl.zst exists locally,
    upload it; else if <slice>.jsonl exists, compress + upload;
    else skip (slice not mined locally). The manifest file is
    overwritten with new sha256s + sizes; operator commits the
    manifest change to git.
    """
    from wyrd.generators.kenning.bulk_sources import (
        load_config,
        load_manifest,
        manifest_to_json,
        upload_slices,
    )

    manifest = load_manifest(manifest_path)
    config = load_config(manifest)
    result = upload_slices(
        manifest,
        config,
        slice_names=list(slice_names) if slice_names else None,
    )

    click.echo(f"Uploaded: {len(result.uploaded)}", err=True)
    for name in result.uploaded:
        click.echo(f"  + {name}", err=True)
    click.echo(f"Skipped:  {len(result.skipped)}  (no local file)", err=True)
    if result.failed:
        click.echo(f"FAILED:   {len(result.failed)}", err=True)
        for name, reason in result.failed:
            click.echo(f"  ! {name}: {reason}", err=True)
        raise SystemExit(1)

    manifest_path.write_text(manifest_to_json(result.new_manifest), encoding="utf-8")
    click.echo(f"\nWrote {manifest_path}", err=True)


@lexicon.command("verify-bulk-sources")
def lexicon_verify_bulk_sources() -> None:
    """Verify ~/.wyrd/sources/ matches the manifest (wyrd-0vj3).

    Walks each manifest slice, checks the local cache for presence
    + sha256 match, and exits 1 if any are missing or mismatched.
    Operator-on-demand only — not gated in CI.
    """
    from wyrd.generators.kenning.bulk_sources import (
        load_config,
        load_manifest,
        verify_local_cache,
    )

    manifest = load_manifest()
    config = load_config(manifest)
    statuses = verify_local_cache(manifest, config)

    ok = 0
    missing = 0
    mismatch = 0
    for status in statuses:
        if not status.present:
            click.echo(f"  ! {status.slice_name}: MISSING ({status.cache_path})", err=True)
            missing += 1
        elif not status.sha256_matches:
            click.echo(
                f"  ! {status.slice_name}: SHA256 MISMATCH ({status.cache_path})",
                err=True,
            )
            mismatch += 1
        else:
            ok += 1

    click.echo(f"\nOK: {ok}   Missing: {missing}   Mismatch: {mismatch}", err=True)
    if missing or mismatch:
        click.echo(
            "\nRun `lexicon fetch-bulk-sources` to repair the local cache.",
            err=True,
        )
        raise SystemExit(1)


@lexicon.command("dump-jsonl")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
    help="Source SQLite DB to read from.",
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/mining"),
    show_default=True,
    help="Directory to write <source_id>.jsonl files into.",
)
@click.option(
    "--source-id",
    default=None,
    help=(
        "Dump just one source's file. Without this, every source row is "
        "dumped to its own JSONL file."
    ),
)
@click.option(
    "--include-bulk",
    is_flag=True,
    default=False,
    help=(
        "Also dump bulk wiktextract-derived sources (wiktionary, "
        "wiktionary-empirical, wiktionary-forms). Default skips them — "
        "their rows are re-derivable from L1 raw inputs and the "
        "wiktionary dump file alone is ~200MB+."
    ),
)
def lexicon_dump_jsonl(
    db_path: Path,
    out_dir: Path,
    source_id: str | None,
    include_bulk: bool,
) -> None:
    """Dump per-source L2 facts from the lexicon DB to JSONL (wyrd-f295).

    Reads the current SQLite lexicon and emits one
    ``<source_id>.jsonl`` per ``source`` row, containing canonical-state
    rows for the source itself + every etymon it cites, plus list rows
    for citations, descent edges, mining-run audits, and the source's
    toponym etymologies (with element lists inline).

    Output is the "first compaction" — pure canonical-state rows that
    replay back to the same DB state via ``lexicon rebuild`` (when that
    lands). Once committed to git, ``data/mining/<source>.jsonl`` becomes
    the source of truth and the SQLite DB becomes a rebuildable build
    artifact.
    """
    from urllib.parse import quote

    from wyrd.generators.kenning.jsonl_dump import (
        DEFAULT_BULK_EXCLUDED_SOURCES,
        dump_all_sources,
        dump_fantasy_morphemes_to_file,
        dump_source_to_file,
    )

    # Read-only DB access — dump never writes.
    db_uri = f"file:{quote(str(db_path.absolute()))}?mode=ro"
    conn = sqlite3.connect(db_uri, uri=True)
    conn.row_factory = sqlite3.Row

    exclude = () if include_bulk else DEFAULT_BULK_EXCLUDED_SOURCES

    # Pre-initialize so the post-try summary doesn't UnboundLocalError
    # if dump_all_sources raises (the finally still runs; control then
    # jumps past the summary lines, but a future maintainer reorganizing
    # this block won't trip on a phantom name binding).
    counts: dict[str, int] = {}
    fm_count = 0
    try:
        if source_id is not None:
            path, count = dump_source_to_file(conn, source_id, out_dir)
            click.echo(f"Wrote {count} rows → {path}", err=True)
            return
        counts = dump_all_sources(conn, out_dir, exclude=exclude)
        # wyrd-2thc: fantasy_morpheme has no source attribution — emit
        # to the synthetic ``_fantasy_morphemes.jsonl`` file.
        _, fm_count = dump_fantasy_morphemes_to_file(conn, out_dir)
    finally:
        conn.close()

    total_rows = sum(counts.values()) + fm_count
    sources_dumped = len(counts) + (1 if fm_count else 0)
    click.echo(f"Dumped {sources_dumped} sources, {total_rows} rows → {out_dir}", err=True)
    for sid, n in sorted(counts.items()):
        click.echo(f"  {sid:<40} {n:>6}", err=True)
    if fm_count:
        click.echo(f"  {'fantasy-mining':<40} {fm_count:>6}", err=True)


@lexicon.command("rebuild-from-jsonl")
@click.option(
    "--db",
    "db_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
    help="Target SQLite path. WIPED if it exists.",
)
@click.option(
    "--jsonl-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data/mining"),
    show_default=True,
    help="Directory of <source_id>.jsonl files to replay into the DB.",
)
@click.option(
    "--with-enrichment",
    is_flag=True,
    default=False,
    help=(
        "After replay, run the L3 enrichment passes (normalize-ocr + "
        "link-lemmas) in canonical order. Without this flag the rebuild "
        "stops at L2 and operators run the enrichment commands manually."
    ),
)
@click.option(
    "--skip-bulk",
    is_flag=True,
    default=False,
    help=(
        "Skip L1 wiktextract bulk ingest. By default rebuild-from-jsonl "
        "ingests every slice listed in data/mining/_bulk_manifest.json "
        "from ~/.wyrd/sources/. --skip-bulk produces an L2-only DB — "
        "useful for fast iteration on a single curated source file or "
        "for debugging without paying the multi-minute wiktextract cost."
    ),
)
@click.option(
    "--fetch-bulk",
    is_flag=True,
    default=False,
    help=(
        "Download missing/mismatched bulk slices from S3 to "
        "~/.wyrd/sources/ before ingesting. Without this flag, missing "
        "slices fail loud with a hint to run `lexicon fetch-bulk-sources` "
        "separately. Opt-in to keep CI / offline operators from silently "
        "fetching."
    ),
)
def lexicon_rebuild_from_jsonl(
    db_path: Path,
    jsonl_dir: Path,
    with_enrichment: bool,
    skip_bulk: bool,
    fetch_bulk: bool,
) -> None:
    """Rebuild the lexicon SQLite from per-source JSONL + L1 bulk
    wiktextract slices — wyrd-f295 + wyrd-hidb.

    Initializes a fresh schema at ``--db`` (wiping any existing DB at
    that path), then:

    1. (default) Ingests each L1 wiktextract slice listed in
       ``data/mining/_bulk_manifest.json`` from ``~/.wyrd/sources/``.
       Pass ``--skip-bulk`` for an L2-only debug rebuild; pass
       ``--fetch-bulk`` to auto-download missing slices from S3.
    2. Replays every ``<source_id>.jsonl`` under ``--jsonl-dir`` into
       the SQLite. Each L2 file must contain exactly one ``source``
       canonical-state row — its ``ref`` is the source_id for every
       list-type row in the file. Etymons appearing in multiple files
       are merged: glosses + tags are unioned; scalar fields follow
       last-write-wins by file order.
    3. (with ``--with-enrichment``) Runs the full L3 enrichment chain
       via ``run_full_enrichment`` (wyrd-hidb Phase 2): normalize-ocr →
       link-lemmas → apply-curation → decompose → cluster-cognates →
       classify-stratum → derive-english-shaped → project-period-forms.
       All eight passes run in one invocation; standalone CLI commands
       remain for targeted reruns (e.g. with ``--force``).
    """
    from wyrd.generators.kenning.jsonl_build import (
        build_from_jsonl,
        jsonl_paths_in,
    )
    from wyrd.generators.kenning.lexicon import init_schema

    init_schema(db_path)

    # L1 bulk ingest first — its rows are inputs for the L2 replay
    # (citations + descent edges referencing wiktionary-empirical
    # etymons would orphan-skip otherwise).
    if not skip_bulk:
        from wyrd.generators.kenning.bulk_sources import ingest_all_slices

        click.echo("Bulk ingest (L1):", err=True)
        with LexiconDB(db_path) as db:
            bulk_result = ingest_all_slices(db, apply=True, fetch=fetch_bulk)
        if bulk_result.fetched:
            click.echo(f"  fetched from S3:   {len(bulk_result.fetched)}", err=True)
        click.echo(f"  slices ingested:   {len(bulk_result.per_slice)}", err=True)
        for k, v in bulk_result.totals.items():
            click.echo(f"  {k:<35} {v:>10,}", err=True)
        if bulk_result.failed:
            click.echo(f"  FAILED: {len(bulk_result.failed)}", err=True)
            for name, reason in bulk_result.failed:
                click.echo(f"    ! {name}: {reason}", err=True)
            raise SystemExit(1)
        click.echo("", err=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        paths = jsonl_paths_in(jsonl_dir)
        counts = build_from_jsonl(conn, paths)
    finally:
        conn.close()

    click.echo(f"Rebuilt {db_path} from {len(paths)} JSONL files", err=True)
    click.echo("Inserted:", err=True)
    for table, n in counts.items():
        # Some count fields are sample-lists (orphan_refs from wyrd-lene)
        # rather than ints. Format the scalar counts as right-aligned
        # numbers; render list fields as just their length so the line
        # stays readable.
        if isinstance(n, list):
            click.echo(f"  {table:<35} ({len(n)} samples)", err=True)
        else:
            click.echo(f"  {table:<35} {n:>8}", err=True)

    if with_enrichment:
        from wyrd.generators.kenning.enrichment import (
            format_enrichment_run,
            run_full_enrichment,
        )
        from wyrd.generators.kenning.jsonl_build import collect_curation_overrides

        curation_state = collect_curation_overrides(paths)
        click.echo("", err=True)
        with LexiconDB(db_path) as db:
            result = run_full_enrichment(db, apply=True, curation_state=curation_state or None)
        click.echo(format_enrichment_run(result), err=True)


@lexicon.command("enrich")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    default=False,
    help=(
        "Actually write merged_into_id + lemma_id. Without this flag the "
        "command runs both passes as a dry-run and reports counts."
    ),
)
@click.option(
    "--jsonl-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data/mining"),
    show_default=True,
    help=(
        "Directory holding the per-source JSONL files. Scanned for "
        "etymon_curation events (wyrd-2jhs) that apply after the auto-"
        "clustering passes. Skip via --no-curation."
    ),
)
@click.option(
    "--with-curation/--no-curation",
    default=True,
    show_default=True,
    help=(
        "Apply operator curation overrides from the JSONL dir as the "
        "final enrichment step. Default on; disable for a pure auto-"
        "clustering run."
    ),
)
def lexicon_enrich(
    db_path: Path,
    apply_changes: bool,
    jsonl_dir: Path,
    with_curation: bool,
) -> None:
    """Run L3 enrichment passes in canonical order (wyrd-ilam).

    Today this is normalize-ocr → link-lemmas — the FIRST L3 enrichment
    migration per the wyrd-eni4 plan. Follow-on PRs extend the
    orchestrator to cluster-cognates, classify-stratum,
    derive-english-shaped, project-period-forms, etc.

    Order matters: normalize-ocr writes merged_into_id (OCR tombstones)
    before link-lemmas's lemma_id targets are picked, so inflected
    forms can only link to canonical etymons, never to tombstones.

    Standalone-pass commands (`lexicon normalize-ocr`,
    `lexicon link-lemmas`) stay available for fine-grained operator
    control; this orchestrator is the canonical one-shot.
    """
    from wyrd.generators.kenning.enrichment import (
        format_enrichment_run,
        run_full_enrichment,
    )
    from wyrd.generators.kenning.jsonl_build import (
        collect_curation_overrides,
        jsonl_paths_in,
    )

    curation_state = None
    if with_curation:
        curation_state = collect_curation_overrides(jsonl_paths_in(jsonl_dir)) or None
    with LexiconDB(db_path) as db:
        result = run_full_enrichment(db, apply=apply_changes, curation_state=curation_state)
    click.echo(format_enrichment_run(result), err=True)
    if not apply_changes:
        click.echo("\n(dry-run; pass --apply to commit)", err=True)


@lexicon.command("curate-etymon")
@click.argument("etymon_ref")
@click.option(
    "--lemma-ref",
    default=None,
    help=(
        "Set the etymon's lemma_id to this ref ('<language>:<canonical_form>'). "
        "Pass an empty string '' to explicitly CLEAR the lemma assignment."
    ),
)
@click.option(
    "--inflection",
    default=None,
    help=(
        "Inflection label written alongside --lemma-ref (e.g. 'spelling_variant', 'weak_oblique')."
    ),
)
@click.option(
    "--merged-into-ref",
    default=None,
    help=(
        "Set the etymon's merged_into_id (OCR-cluster tombstone target) "
        "to this ref. Pass '' to CLEAR. Mutually informative with "
        "--lemma-ref but applied independently."
    ),
)
@click.option(
    "--reason",
    default=None,
    help="Operator note. Doesn't affect the DB; recorded for audit.",
)
@click.option(
    "--curation-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/mining/_curation.jsonl"),
    show_default=True,
    help="JSONL file to append the curation event to. Created if it doesn't exist.",
)
def lexicon_curate_etymon(
    etymon_ref: str,
    lemma_ref: str | None,
    inflection: str | None,
    merged_into_ref: str | None,
    reason: str | None,
    curation_file: Path,
) -> None:
    """Append a curation event to the L2 curation file (wyrd-2jhs).

    Records an operator override that's applied AFTER the auto-clustering
    enrichment passes when `lexicon enrich --apply` (or
    `lexicon rebuild-from-jsonl --with-enrichment`) runs.

    ETYMON_REF is the etymon to curate (e.g. 'old-english:caelf').
    Pass --lemma-ref / --merged-into-ref / --inflection / --reason for
    the override fields; absent flags mean "no opinion on that column",
    and the literal empty string '' explicitly CLEARS the column.
    """
    if lemma_ref is None and merged_into_ref is None:
        raise click.UsageError(
            "Need at least one of --lemma-ref / --merged-into-ref to record a curation."
        )
    # SET both → the applier would produce a contradictory net state
    # (lemma_ref handler runs first then merged_into_ref clears it, or
    # vice versa). Surface the conflict at the CLI.
    # CLEAR both (both passed as '') → legitimate revert-to-standalone
    # operation; the applier correctly nulls out both columns.
    if lemma_ref and merged_into_ref:
        raise click.UsageError(
            "--lemma-ref and --merged-into-ref are mutually exclusive when "
            "SETTING values — an etymon is either a canonical inflection "
            "(lemma) or an OCR-cluster tombstone, not both. "
            "Clearing both with '' is allowed."
        )

    payload: dict[str, str | None] = {"_type": "etymon_curation", "ref": etymon_ref}
    # CLI '' → explicit null (clear column). Absent flag → key omitted.
    if lemma_ref is not None:
        payload["lemma_ref"] = None if lemma_ref == "" else lemma_ref
    if inflection is not None:
        payload["inflection"] = inflection
    if merged_into_ref is not None:
        payload["merged_into_ref"] = None if merged_into_ref == "" else merged_into_ref
    if reason is not None:
        payload["reason"] = reason

    # Create the file with its synthetic source row if it doesn't exist
    # yet — every L2 JSONL file needs exactly one source row.
    if not curation_file.exists():
        curation_file.parent.mkdir(parents=True, exist_ok=True)
        with curation_file.open("w", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "_type": "source",
                        "ref": "manual-curation",
                        "title": "Operator curation overrides (wyrd-2jhs)",
                        "notes": (
                            "L3 enrichment overrides applied after the auto-"
                            "clustering passes. Each event records a "
                            "deliberate operator decision; reverting is "
                            "appending another event with the cleared "
                            "value (lemma_ref: '')."
                        ),
                    },
                    ensure_ascii=False,
                )
            )
            fh.write("\n")

    with curation_file.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False))
        fh.write("\n")

    click.echo(f"Appended curation event for `{etymon_ref}` → {curation_file}", err=True)


@lexicon.command("diff-rebuild")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
    help="Current SQLite DB to compare against.",
)
@click.option(
    "--jsonl-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data/mining"),
    show_default=True,
    help="Source JSONL directory to rebuild from.",
)
@click.option(
    "--with-enrichment",
    is_flag=True,
    default=False,
    help=(
        "Run the L3 enrichment orchestrator after rebuild. Without "
        "this, L3 columns (lemma_id, merged_into_id, etc.) on the "
        "rebuilt DB are NULL — only L2 row counts are compared."
    ),
)
def lexicon_diff_rebuild(db_path: Path, jsonl_dir: Path, with_enrichment: bool) -> None:
    """Rebuild from JSONL into a temp DB; compare row counts to live (wyrd-f4nl).

    Snapshots the current DB's L2 table counts, rebuilds a fresh DB from
    ``--jsonl-dir`` into a temp file, and reports per-table deltas.
    Useful for catching regressions in the JSONL pipeline as ingesters
    get converted — CI can run this and gate on the exit code.

    Exit status: 0 when no table changed row count; 1 when any did.
    """
    import tempfile
    from urllib.parse import quote

    from wyrd.generators.kenning.jsonl_build import (
        build_from_jsonl,
        diff_table_counts,
        format_diff_rebuild,
        has_any_delta,
        jsonl_paths_in,
        table_counts,
    )
    from wyrd.generators.kenning.lexicon import init_schema

    # Read current DB counts before touching anything else.
    db_uri = f"file:{quote(str(db_path.absolute()))}?mode=ro"
    current_conn = sqlite3.connect(db_uri, uri=True)
    current_conn.row_factory = sqlite3.Row
    try:
        before = table_counts(current_conn)
    finally:
        current_conn.close()

    # Rebuild into a temp directory so the WAL sidecars (-wal, -shm
    # that init_schema creates because of journal_mode=WAL) get
    # cleaned up alongside the main DB file. NamedTemporaryFile
    # would only track the main file and leak the sidecars in /tmp.
    # Directory also dodges the init_schema-unlinks-the-tempfile
    # ownership confusion.
    with tempfile.TemporaryDirectory(prefix="wyrd-diff-rebuild-") as tmpdir:
        rebuilt_path = Path(tmpdir) / "rebuilt.db"
        init_schema(rebuilt_path)
        rebuilt_conn = sqlite3.connect(rebuilt_path)
        rebuilt_conn.row_factory = sqlite3.Row
        try:
            paths = jsonl_paths_in(jsonl_dir)
            build_from_jsonl(rebuilt_conn, paths)
        finally:
            rebuilt_conn.close()

        if with_enrichment:
            from wyrd.generators.kenning.enrichment import run_full_enrichment
            from wyrd.generators.kenning.jsonl_build import collect_curation_overrides

            curation = collect_curation_overrides(paths) or None
            with LexiconDB(rebuilt_path) as db:
                run_full_enrichment(db, apply=True, curation_state=curation)

        rebuilt_conn = sqlite3.connect(rebuilt_path)
        rebuilt_conn.row_factory = sqlite3.Row
        try:
            after = table_counts(rebuilt_conn)
        finally:
            rebuilt_conn.close()

    rows = diff_table_counts(before, after)
    click.echo(f"## diff-rebuild: {db_path} ↔ rebuild({jsonl_dir})")
    click.echo("")
    click.echo(format_diff_rebuild(rows))

    if has_any_delta(rows):
        click.echo("\n⚠️  Row counts differ between current and rebuild.", err=True)
        raise SystemExit(1)
    click.echo("\n✓ No row-count deltas.", err=True)


@lexicon.command("diff-bundle")
@click.option(
    "--jsonl-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data/mining"),
    show_default=True,
    help="Directory of <source_id>.jsonl files to rebuild from.",
)
@click.option(
    "--bundle",
    "bundle_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("wyrd/generators/kenning/data/meanings.json"),
    show_default=True,
    help="Committed meanings.json to compare the rebuild against.",
)
@click.option(
    "--skip-bulk",
    is_flag=True,
    default=False,
    help=(
        "Skip L1 wiktextract bulk ingest. Useful when ~/.wyrd/sources/ is "
        "unavailable; produces an L2-only rebuild which WILL show large "
        "drift (wiktionary-empirical subjects missing) — use --skip-bulk "
        "for fast pipeline-shape checks, not for release decisions."
    ),
)
@click.option(
    "--fetch-bulk",
    is_flag=True,
    default=False,
    help=(
        "Download missing/mismatched bulk slices from S3 to "
        "~/.wyrd/sources/ before ingesting. Without this flag, missing "
        "slices fail loud with a hint to run `lexicon fetch-bulk-sources`."
    ),
)
@click.option(
    "--joiners-from",
    "joiners_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Optional joiners-pool sidecar JSON (matches `export-meanings`'s "
        "flag). Required when the committed bundle includes a `joiners` "
        "field — otherwise the diff will trivially flag the missing key."
    ),
)
@click.option(
    "--min-witnesses",
    type=int,
    default=3,
    show_default=True,
    help="Witness threshold fallback for the export step (matches export-meanings).",
)
@click.option(
    "--lang-threshold",
    "lang_threshold_specs",
    multiple=True,
    metavar="LANG=N",
    help="Per-language witness threshold override (repeatable).",
)
@click.option(
    "--preset/--no-preset",
    "use_preset",
    default=True,
    show_default=True,
    help="Start from RECOMMENDED_LANG_THRESHOLDS (matches export-meanings).",
)
@click.option(
    "--include-rando/--no-include-rando",
    default=True,
    show_default=True,
)
@click.option(
    "--include-wiktionary-empirical/--no-include-wiktionary-empirical",
    default=True,
    show_default=True,
)
def lexicon_diff_bundle(
    jsonl_dir: Path,
    bundle_path: Path,
    skip_bulk: bool,
    fetch_bulk: bool,
    joiners_path: Path | None,
    min_witnesses: int,
    lang_threshold_specs: tuple[str, ...],
    use_preset: bool,
    include_rando: bool,
    include_wiktionary_empirical: bool,
) -> None:
    """Operator-runnable real-data round-trip drift check (wyrd-w3x0).

    Rebuilds the lexicon from ``--jsonl-dir`` (plus L1 bulk slices unless
    ``--skip-bulk``), runs the full enrichment chain, exports a fresh
    bundle, and diffs it against ``--bundle`` (the committed
    meanings.json). The CI-runnable round-trip test (wyrd-xrnw) catches
    pipeline-level non-determinism; this command catches *data-level*
    drift — JSONL changes the committed bundle doesn't yet reflect, or
    export-side config drift between rebuild times.

    Exit status: 0 when bytes match; 1 when drift exists. The textual
    report is always written to stdout; structural-diff highlights
    (added/removed/changed subjects + first byte divergence) are
    included whenever both bundles parse as JSON. Use this before a
    bundle re-export PR to know what's about to change.
    """
    # Deferred imports: bulk_sources, enrichment, jsonl_build pull in
    # heavy dependencies (LLM clients, sqlite extension setup) that
    # other lexicon commands don't need; deferring them keeps `wyrd
    # kenning --help` snappy. ``init_schema`` is already imported at
    # module level (line 72) so we don't re-import it here.
    import tempfile

    from wyrd.generators.kenning.bulk_sources import ingest_all_slices
    from wyrd.generators.kenning.bundle_diff import compute_bundle_diff, format_bundle_diff
    from wyrd.generators.kenning.enrichment import run_full_enrichment
    from wyrd.generators.kenning.jsonl_build import (
        build_from_jsonl,
        collect_curation_overrides,
        jsonl_paths_in,
    )

    # Mirror the lang-threshold parsing from `export-meanings` — keeping
    # the two CLIs accepting identical flag shapes means operators can
    # validate the actual command that produced the committed bundle.
    lang_thresholds: dict[str, int] = dict(RECOMMENDED_LANG_THRESHOLDS) if use_preset else {}
    for spec in lang_threshold_specs:
        if "=" not in spec:
            raise click.BadParameter(f"--lang-threshold expects LANG=N, got {spec!r}")
        lang, _, n_str = spec.partition("=")
        lang = lang.strip()
        n_str = n_str.strip()
        if not lang or not n_str:
            raise click.BadParameter(
                f"--lang-threshold {spec!r}: both LANG and N must be non-empty"
            )
        try:
            n = int(n_str)
        except ValueError as exc:
            raise click.BadParameter(f"--lang-threshold {spec!r}: N must be an integer") from exc
        lang = LANGUAGE_FIELDS.get(lang, lang)
        lang_thresholds[lang] = n

    joiners = _load_joiners_sidecar(joiners_path) if joiners_path is not None else {}

    with tempfile.TemporaryDirectory(prefix="wyrd-diff-bundle-") as tmpdir:
        rebuilt_path = Path(tmpdir) / "rebuilt.db"
        init_schema(rebuilt_path)

        if not skip_bulk:
            click.echo("Bulk ingest (L1)...", err=True)
            with LexiconDB(rebuilt_path) as db:
                bulk_result = ingest_all_slices(db, apply=True, fetch=fetch_bulk)
            if bulk_result.failed:
                click.echo(f"Bulk ingest failed: {len(bulk_result.failed)} slice(s)", err=True)
                for name, reason in bulk_result.failed:
                    click.echo(f"  ! {name}: {reason}", err=True)
                raise SystemExit(1)

        click.echo("L2 replay...", err=True)
        paths = jsonl_paths_in(jsonl_dir)
        with LexiconDB(rebuilt_path) as db:
            build_from_jsonl(db.conn, paths)

        click.echo("L3 enrichment chain...", err=True)
        curation_state = collect_curation_overrides(paths)
        with LexiconDB(rebuilt_path) as db:
            run_full_enrichment(db, apply=True, curation_state=curation_state or None)

        click.echo("Building bundle from rebuild...", err=True)
        with LexiconDB(rebuilt_path) as db:
            subjects = export_meanings(
                db,
                min_witnesses=min_witnesses,
                lang_thresholds=lang_thresholds,
                include_rando=include_rando,
                include_wiktionary_empirical=include_wiktionary_empirical,
            )
            canonical_decompositions = collect_canonical_decompositions(db)
            fantasy_morphemes = collect_fantasy_morphemes(db)

    # Build the bundle dict the same way lexicon_export_meanings does
    # (around cli.py:1354). If these two builders ever drift, the
    # round-trip test in tests/test_kenning_hidb_phase3_round_trip.py +
    # the integration test for this command will catch it.
    bundle: dict[str, Any] = {"subjects": subjects}
    if canonical_decompositions:
        bundle["canonical_decompositions"] = canonical_decompositions
    if joiners:
        bundle["joiners"] = joiners
    if fantasy_morphemes:
        bundle["fantasy_morphemes"] = fantasy_morphemes

    rebuilt_text = json.dumps(bundle, ensure_ascii=False, indent=2) + "\n"
    committed_text = bundle_path.read_text(encoding="utf-8")

    diff = compute_bundle_diff(committed_text, rebuilt_text)
    click.echo(format_bundle_diff(diff))

    if not diff.bytes_match:
        raise SystemExit(1)


def _append_remove_event(
    jsonl_dir: Path,
    source_id: str,
    row_type: str,
    ref: str,
    reason: str | None,
    ref_format_hint: str,
) -> Path:
    """Shared implementation for `lexicon prune-toponym` and
    `lexicon prune-etymon`. Validates that the source file exists and
    that the named ref actually resolves inside it before appending
    a remove event. Returns the target file path.

    The validation step matters because the kernel's
    `bucket.pop(ref, None)` silently swallows missing refs; combined
    with orphan-skip a typo would produce a successful-looking rebuild
    that doesn't actually prune anything.

    ``ref_format_hint`` is per-row-type guidance for the error message
    (etymon: ``lang:form``; toponym: ``name@region``).
    """
    target_file = jsonl_dir / f"{source_id}.jsonl"
    if not target_file.exists():
        raise click.UsageError(
            f"Source file not found: {target_file}. "
            f"Use --jsonl-dir to point at the right directory."
        )

    from wyrd.generators.kenning.jsonl_log import replay_file

    state = replay_file(target_file)
    if ref not in state.keyed[row_type]:
        raise click.UsageError(
            f"{row_type.capitalize()} ref {ref!r} not found in {target_file}. "
            f"Check spelling ({ref_format_hint}) and confirm the source "
            f"owns this row."
        )

    payload: dict[str, str] = {"_op": "remove", "_type": row_type, "ref": ref}
    if reason is not None:
        # Carried as a JSON key but ignored by the kernel (remove ops
        # discard payload beyond _type + ref); useful for git blame.
        payload["_reason"] = reason

    with target_file.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False))
        fh.write("\n")

    return target_file


@lexicon.command("ingest-os-open-names")
@click.argument(
    "csv_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/mining/os_open_names.jsonl"),
    show_default=True,
    help="Where to write the JSONL events.",
)
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    default=False,
    help="Actually write the JSONL. Without this the parse runs but writes nothing.",
)
def lexicon_ingest_os_open_names(csv_path: Path, out_path: Path, apply_changes: bool) -> None:
    """Ingest OS Open Names CSV → JSONL events (wyrd-3ypp).

    Reads the OS Open Names data product, filters to populated places
    (City / Town / Village / Hamlet / Suburban Area / Other Settlement),
    and writes toponym + attestation events to the named JSONL.

    Operator workflow:
        1. Download OS Open Names CSV to sources/os_open_names.csv
        2. lexicon ingest-os-open-names sources/os_open_names.csv --apply
        3. lexicon rebuild-from-jsonl

    The JSONL file is gitignored (30-50MB) — it's regenerated from
    the L1 CSV on demand. Source attribution: synthetic 'os_open_names'
    source, recorded with OGL v3 license notes.
    """
    from wyrd.generators.kenning.os_open_names_ingester import ingest

    counts = ingest(csv_path, out_path, apply=apply_changes)
    click.echo(f"CSV rows scanned:       {counts['csv_rows_scanned']:>10}", err=True)
    click.echo(f"Populated places kept:  {counts['populated_places_kept']:>10}", err=True)
    click.echo(f"Toponym events:         {counts['toponym_events']:>10}", err=True)
    click.echo(f"Attestation events:     {counts['attestation_events']:>10}", err=True)
    if not apply_changes:
        click.echo("\n(dry-run; pass --apply to write)", err=True)
    else:
        click.echo(f"\nWrote {out_path}", err=True)


@lexicon.command("ingest-hundred-rolls")
@click.argument(
    "csv_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/mining/rotuli_hundredorum.jsonl"),
    show_default=True,
    help="Where to write the JSONL events.",
)
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    default=False,
    help="Actually write the JSONL. Without this the parse runs but writes nothing.",
)
def lexicon_ingest_hundred_rolls(csv_path: Path, out_path: Path, apply_changes: bool) -> None:
    """Ingest Hundred Rolls (1279-80) CSV → JSONL events (wyrd-3atv).

    Reads a transcribed Rotuli Hundredorum CSV (operator-supplied) and
    writes toponym + attestation events with date_year=1279.

    Expected CSV columns (named in header row):
      vill (required), hundred, county, country, modern_name

    Cross-era cross-linking is automatic — when a Hundred Rolls
    modern_name matches a Domesday or OS Open Names toponym ref, all
    attestations attach to the same toponym in the rebuilt DB.
    """
    from wyrd.generators.kenning.hundred_rolls_ingester import ingest

    counts = ingest(csv_path, out_path, apply=apply_changes)
    click.echo(f"CSV rows scanned:    {counts['csv_rows_scanned']:>10}", err=True)
    click.echo(f"Toponym events:      {counts['toponym_events']:>10}", err=True)
    click.echo(f"Attestation events:  {counts['attestation_events']:>10}", err=True)
    if not apply_changes:
        click.echo("\n(dry-run; pass --apply to write)", err=True)
    else:
        click.echo(f"\nWrote {out_path}", err=True)


@lexicon.command("ingest-speed-1611")
@click.argument(
    "csv_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/mining/speed_1611.jsonl"),
    show_default=True,
    help="Where to write the JSONL events.",
)
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    default=False,
    help="Actually write the JSONL. Without this the parse runs but writes nothing.",
)
def lexicon_ingest_speed_1611(csv_path: Path, out_path: Path, apply_changes: bool) -> None:
    """Ingest a Speed-1611 transcription CSV → JSONL events (wyrd-myh1).

    Expected CSV columns (named header row):
      place_name (required), county, parish, country, modern_name
    """
    from wyrd.generators.kenning.speed_1611_ingester import ingest

    counts = ingest(csv_path, out_path, apply=apply_changes)
    click.echo(f"CSV rows scanned:    {counts['csv_rows_scanned']:>10}", err=True)
    click.echo(f"Toponym events:      {counts['toponym_events']:>10}", err=True)
    click.echo(f"Attestation events:  {counts['attestation_events']:>10}", err=True)
    if not apply_changes:
        click.echo("\n(dry-run; pass --apply to write)", err=True)
    else:
        click.echo(f"\nWrote {out_path}", err=True)


@lexicon.command("ingest-hearth-tax")
@click.argument(
    "csv_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/mining/hearth_tax_1660s.jsonl"),
    show_default=True,
    help="Where to write the JSONL events.",
)
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    default=False,
    help="Actually write the JSONL. Without this the parse runs but writes nothing.",
)
def lexicon_ingest_hearth_tax(csv_path: Path, out_path: Path, apply_changes: bool) -> None:
    """Ingest a Hearth-Tax-returns CSV → JSONL events (wyrd-myh1).

    Expected CSV columns (named header row):
      place_name (required), parish, county, year_specific, country, modern_name

    year_specific captures the actual collection year (1662-1674);
    blank/unparseable falls back to 1665. Out-of-range integers warn.
    """
    from wyrd.generators.kenning.hearth_tax_ingester import ingest

    counts = ingest(csv_path, out_path, apply=apply_changes)
    click.echo(f"CSV rows scanned:    {counts['csv_rows_scanned']:>10}", err=True)
    click.echo(f"Toponym events:      {counts['toponym_events']:>10}", err=True)
    click.echo(f"Attestation events:  {counts['attestation_events']:>10}", err=True)
    if not apply_changes:
        click.echo("\n(dry-run; pass --apply to write)", err=True)
    else:
        click.echo(f"\nWrote {out_path}", err=True)


@lexicon.command("prune-toponym")
@click.argument("toponym_ref")
@click.argument("source_id")
@click.option(
    "--reason",
    default=None,
    help=("Operator note. Doesn't affect the DB; recorded in the JSONL event for git-blame audit."),
)
@click.option(
    "--jsonl-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data/mining"),
    show_default=True,
)
def lexicon_prune_toponym(
    toponym_ref: str, source_id: str, reason: str | None, jsonl_dir: Path
) -> None:
    """Append a 'remove' event for a toponym to its source's L2 file (wyrd-lene).

    TOPONYM_REF is the toponym to prune ('Cart@Herefordshire'). SOURCE_ID
    is the source whose JSONL file owns the toponym row (must already
    exist as <jsonl-dir>/<source_id>.jsonl).

    On the next `lexicon rebuild-from-jsonl`, the toponym row is dropped
    from the DB and any etymology_element rows referencing it are
    counted as orphans + skipped. Operator commits the JSONL change in
    git; reverting is appending an 'add' event with the original data.
    """
    target_file = _append_remove_event(
        jsonl_dir,
        source_id,
        "toponym",
        toponym_ref,
        reason,
        ref_format_hint="`name@region` format; `@-` is null region",
    )
    click.echo(f"Appended remove event for toponym `{toponym_ref}` → {target_file}", err=True)


@lexicon.command("prune-etymon")
@click.argument("etymon_ref")
@click.argument("source_id")
@click.option(
    "--reason",
    default=None,
    help=("Operator note. Doesn't affect the DB; recorded in the JSONL event for git-blame audit."),
)
@click.option(
    "--jsonl-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("data/mining"),
    show_default=True,
)
def lexicon_prune_etymon(
    etymon_ref: str, source_id: str, reason: str | None, jsonl_dir: Path
) -> None:
    """Append a 'remove' event for an etymon to its source's L2 file (wyrd-8wgr).

    ETYMON_REF is the etymon to prune ('old-english:pīe'). SOURCE_ID
    is the source whose JSONL file owns the etymon row (must already
    exist as <jsonl-dir>/<source_id>.jsonl).

    On the next `lexicon rebuild-from-jsonl`, the etymon row is dropped
    from the DB and any referencing fact-rows (citations, descent edges
    in this source, etymology elements that name it) are counted as
    orphans and skipped. Operator commits the JSONL change in git;
    reverting is appending an 'add' event with the original data.

    Typical use: dead-rando audit prunes (spurious morphemes the LLM
    hallucinated or scribal variants the operator wants removed).
    """
    target_file = _append_remove_event(
        jsonl_dir,
        source_id,
        "etymon",
        etymon_ref,
        reason,
        ref_format_hint="`<language>:<canonical_form>` format",
    )
    click.echo(f"Appended remove event for etymon `{etymon_ref}` → {target_file}", err=True)


@lexicon.command("enrichment-status")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
def lexicon_enrichment_status(db_path: Path) -> None:
    """Report L3 enrichment coverage on the current DB (wyrd-ilam).

    Read-only: shows per-column populated/total counts and method-
    version distributions where available. Useful BEFORE running
    `lexicon enrich --apply` to see what's already done, and AFTER to
    verify the passes populated what was expected.
    """
    from urllib.parse import quote

    from wyrd.generators.kenning.enrichment import (
        enrichment_status,
        format_enrichment_status,
    )

    db_uri = f"file:{quote(str(db_path.absolute()))}?mode=ro"
    conn = sqlite3.connect(db_uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        status = enrichment_status(conn)
    finally:
        conn.close()
    click.echo(format_enrichment_status(status))


@lexicon.command("compact-jsonl")
@click.argument(
    "jsonl_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
def lexicon_compact_jsonl(jsonl_path: Path) -> None:
    """Compact a JSONL event-log into canonical-state rows in place (wyrd-f295).

    Reads ``JSONL_PATH``, replays every row (canonical seed + add/set/
    patch/remove events) into final state, and writes back as canonical-
    state rows only. Idempotent — running on an already-compacted file
    is a no-op.

    Useful after a curation pass appends a batch of events (e.g.
    dead-rando prune, lemma normalization corrections) — once review
    is complete, compaction folds the events back into a clean canonical
    state for diff readability.
    """
    from wyrd.generators.kenning.jsonl_log import compact_file

    n = compact_file(jsonl_path)
    click.echo(f"Compacted {jsonl_path} → {n} canonical rows", err=True)


@lexicon.command("ingest-etymonline")
@click.argument(
    "source_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    default=False,
    help=(
        "Actually upsert etymons + descent edges. Without this, the "
        "command parses each file and reports counts without writing."
    ),
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Stop after processing N files (smoke testing).",
)
def lexicon_ingest_etymonline(
    source_dir: Path,
    db_path: Path,
    apply_changes: bool,
    limit: int | None,
) -> None:
    """Ingest pre-scraped Etymonline prose into the etymology graph (wyrd-zqkp).

    Each file in `source_dir` should be one Etymonline word page's
    prose, captured separately by the operator (e.g. via `rodney text
    'section.prose-lg' > harpy.txt`). The CLI doesn't fetch — it
    consumes saved text — so the scrape rate, identity, and policy
    decisions stay with the operator.

    Per-file processing:
    1. Parse the prose into one or more Senses (multi-sense pages
       like 'troll' produce v./n.1/n.2 blocks).
    2. For each Sense's etymological chain, upsert (language, word)
       pairs and connect them with `etymon_descent` edges
       source_id='etymonline'.
    3. Wire chain[0] to the headword's existing modern-english
       etymon row when present.

    Idempotent: re-running over the same dir is a no-op via the
    etymon_descent UNIQUE on (parent, child, edge_type, source).
    """
    files = sorted(source_dir.glob("*.txt"))
    if limit is not None:
        files = files[:limit]
    click.echo(
        f"Ingesting {len(files)} Etymonline file(s) from {source_dir}. "
        f"{'Applying' if apply_changes else 'Dry-run'}.",
        err=True,
    )
    totals = {
        "files": 0,
        "senses_parsed": 0,
        "senses_with_chain": 0,
        "senses_without_chain": 0,
        "chain_links": 0,
        "etymons_added_or_existing": 0,
        "edges_added": 0,
        "edges_skipped_dupe": 0,
        "leaf_edge_skipped_no_headword": 0,
        "glosses_added": 0,
    }
    progress_start = time.time()
    total_files = len(files)
    with LexiconDB(db_path) as db:
        if apply_changes:
            etymonline_ingester.ensure_source(db)
        for completed, f in enumerate(files, start=1):
            text = f.read_text()
            counts = etymonline_ingester.ingest_text(db, text, apply=apply_changes)
            totals["files"] += 1
            for k, v in counts.items():
                if k in totals:
                    totals[k] += v
            edges_field = f" edges_added={counts['edges_added']}" if apply_changes else ""
            elapsed = time.time() - progress_start
            rate = elapsed / completed
            click.echo(
                f"  [{completed}/{total_files}] {f.name:<32} "
                f"senses={counts['senses_parsed']:<2} "
                f"links={counts['chain_links']:<3}{edges_field} "
                f"({rate:.2f}s/file)",
                err=True,
            )
        if apply_changes:
            db.commit()

    click.echo("", err=True)
    click.echo(f"Summary: {totals}", err=True)
    if not apply_changes:
        click.echo("(dry-run; pass --apply to write)", err=True)


@lexicon.command("extract-pfsrd2-monsters")
@click.argument(
    "corpus_root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Write JSONL to this file. Default: stdout. The output is "
        "ready to feed into `wyrd kenning lexicon mine-fantasy-name "
        "--batch <output>` once you've reviewed it."
    ),
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Stop after emitting N records (smoke testing).",
)
def lexicon_extract_pfsrd2_monsters(
    corpus_root: Path,
    output_path: Path | None,
    limit: int | None,
) -> None:
    """Extract pfsrd2-data monsters into wyrd-ami JSONL input (wyrd-ld5x).

    Walks `<corpus_root>/monsters/**/*.json` and emits one
    `{"name": ..., "description": ...}` per distinct creature
    morpheme. Family roots (Bugbear) win over per-variant names
    (Bugbear Thug); compound family names (Dragon, Red) are split
    on `,` to get the base. Multi-word creatures with no family are
    skipped — they don't represent a single morpheme cleanly.

    Dedupes by lowercased input_name across the whole walk so each
    morpheme is emitted once even if it appears across many variants.

    Typical pipeline:

        wyrd kenning lexicon extract-pfsrd2-monsters \\
            ~/521Studios/pfsrd2-data \\
            --output /tmp/pfsrd2-monsters.jsonl

        wyrd kenning lexicon mine-fantasy-name \\
            --batch /tmp/pfsrd2-monsters.jsonl --apply
    """
    record_count = 0
    out_handle = output_path.open("w") if output_path is not None else None
    try:
        for record in pfsrd2_monster_extractor.extract_corpus(corpus_root, limit=limit):
            line = json.dumps(record, ensure_ascii=False)
            if out_handle is not None:
                out_handle.write(line + "\n")
            else:
                # click.echo (not sys.stdout.write) so CliRunner captures
                # stdout output reliably; nl=True adds the newline.
                click.echo(line)
            record_count += 1
    finally:
        if out_handle is not None:
            out_handle.close()
    dest = str(output_path) if output_path is not None else "stdout"
    click.echo(
        f"Extracted {record_count} record(s) from {corpus_root} → {dest}",
        err=True,
    )


@lexicon.command("backfill-fantasy-tags")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    default=False,
    help="Actually write the new etymon_tag rows. Without this, dry-run.",
)
def lexicon_backfill_fantasy_tags(db_path: Path, apply_changes: bool) -> None:
    """Tag existing seed monster etymons as fantasy (wyrd-ami).

    The seed entries from meanings.json that already carry `monster`
    (wyrm, elf, thyrs, grīma, sceocca, phúca, ...) don't know about
    the `fantasy` register tag. This adds it. Idempotent — safe to
    re-run.
    """
    fp = fantasy_pipeline

    if apply_changes:
        with LexiconDB(db_path) as db:
            n_etymons, n_tags = fp.backfill_fantasy_tag_from_monster_tag(db.conn)
            db.commit()
        click.echo(
            f"Backfilled `fantasy` tag onto {n_etymons} monster-tagged etymon(s).",
            err=True,
        )
    else:
        # Dry-run: count without writing.
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            n = conn.execute("""SELECT COUNT(DISTINCT et.etymon_id)
                   FROM etymon_tag et
                   WHERE et.tag = 'monster'
                     AND NOT EXISTS (
                       SELECT 1 FROM etymon_tag et2
                       WHERE et2.etymon_id = et.etymon_id AND et2.tag = 'fantasy'
                     )""").fetchone()[0]
        finally:
            conn.close()
        click.echo(
            f"Would backfill `fantasy` tag onto {n} monster-tagged etymon(s). "
            f"(dry-run; pass --apply to write)",
            err=True,
        )


@lexicon.command("review")
@click.argument("source_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--book",
    default=None,
    help="Limit review to one source_id (e.g. mawer_1920_northumberland_durham).",
)
@click.option(
    "--confidence",
    multiple=True,
    default=("low", "medium"),
    show_default=True,
    help="Re-review rows whose confidence is in this set (repeatable).",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Re-review at most N rows (smoke testing).",
)
@click.option(
    "--provider",
    type=click.Choice(["gemini", "anthropic"]),
    default="gemini",
    show_default=True,
    help=(
        "Review provider. The pipeline is layered: run with gemini first, "
        "then with anthropic on whatever remains. Each tier writes its own "
        "row tagged with its provider name, so consensus stays auditable."
    ),
)
@click.option(
    "--model",
    default=None,
    help=(
        "Override the model. Gemini default: $WYRD_GEMINI_MODEL or "
        "gemini-2.5-flash. Anthropic default: $WYRD_ANTHROPIC_MODEL or "
        "claude-sonnet-4-6."
    ),
)
@click.option(
    "--timeout",
    type=float,
    default=180.0,
    show_default=True,
)
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    default=False,
    help="Actually insert the new rows. Without this, the command runs as "
    "a dry-run reporting what would happen.",
)
@click.option(
    "--include-haiku",
    is_flag=True,
    default=False,
    help=(
        "wyrd-eca: also pick up Haiku-Tier-1-mined rows as Tier-2 review "
        "candidates. Off by default to keep the original Qwen→Gemini "
        "pipeline untouched. Pass when re-reviewing a non-Celtic book "
        "that was mined Tier-1 with Haiku (Bannister 1916 Welsh-Marches, "
        "Harrison 1898 Liverpool, etc.) — those rows otherwise slip past "
        "the candidate query because their notes don't match the "
        "Ollama/Qwen patterns."
    ),
)
@click.option(
    "--include-sonnet",
    is_flag=True,
    default=False,
    help=(
        "wyrd-0rsk: companion to --include-haiku. Also pick up "
        "Sonnet-Tier-1-mined rows as Tier-2 review candidates. Off by "
        "default. Pass when re-reviewing a book whose Tier-1 mining used "
        "Anthropic Sonnet — Joyce 1875 v1+v2, Moore 1890, Johnston 1892, "
        "Morgan 1887 are the current set. Same wiring as --include-haiku, "
        "different LIKE clause: matches `extracted_by:anthropic:%sonnet%`."
    ),
)
def lexicon_review(
    source_dir: Path,
    db_path: Path,
    book: str | None,
    confidence: tuple[str, ...],
    limit: int | None,
    provider: str,
    model: str | None,
    timeout: float,
    apply_changes: bool,
    include_haiku: bool,
    include_sonnet: bool,
) -> None:
    """Re-extract questionable Ollama-mined entries via a stronger LLM.

    Picks toponym_etymology rows whose `notes` indicate Ollama (Qwen)
    origin and whose confidence is "low" or "medium". For each, re-parses
    the source text from <source_dir>, runs through the chosen provider,
    and (with --apply) inserts a NEW toponym_etymology row tagged with
    the provider name. The original Qwen row stays in place; the consensus
    view shows where the two providers agree or disagree.

    Layered usage:
        # Tier 2: Gemini Flash on Qwen low/medium rows
        wyrd kenning lexicon review sources/ --provider gemini --apply

        # Tier 3: Anthropic Sonnet on whatever remains uncertain
        wyrd kenning lexicon review sources/ --provider anthropic --apply

    Each tier is idempotent: skips toponyms that already have a row from
    the chosen provider.
    """
    # click.Choice on the --provider option restricts to {gemini, anthropic};
    # 'ollama' never reaches this body (Click rejects it at parse time).
    provider_tag = f"extracted_by:{provider}:"

    candidates = _select_review_candidates(
        db_path=db_path,
        provider_tag=provider_tag,
        confidence=confidence,
        book=book,
        limit=limit,
        include_haiku=include_haiku,
        include_sonnet=include_sonnet,
    )

    if not candidates:
        click.echo("No candidates to review.", err=True)
        return

    # Group by source_id so we only re-parse each book once.
    by_book: dict[str, list[sqlite3.Row]] = {}
    for c in candidates:
        by_book.setdefault(c["source_id"], []).append(c)

    click.echo(
        f"{len(candidates)} candidates across {len(by_book)} source(s). "
        f"{'Applying' if apply_changes else 'Dry-run'}.",
        err=True,
    )

    client, provider_extract_one = _build_llm_client(provider, model=model, timeout=timeout)

    write_db: LexiconDB | None = None
    if apply_changes:
        write_db = LexiconDB(db_path)

    counts = {
        "reviewed": 0,  # entries we actually called the provider on
        "written": 0,  # new toponym_etymology rows inserted
        "declined": 0,  # provider declined to extract
        "rejected": 0,  # validator rejected (form not in body, etc.)
        "missing_source": 0,  # source file not found in source_dir
        "missing_entry": 0,  # toponym not located in re-parsed source
    }

    try:
        for source_id, rows in by_book.items():
            source_path = source_dir / f"{source_id}.txt"
            if not source_path.exists():
                click.echo(
                    f"  source file missing: {source_path} — skipping {len(rows)} rows",
                    err=True,
                )
                counts["missing_source"] += len(rows)
                continue

            book_counts = _review_one_book(
                source_id=source_id,
                rows=rows,
                source_path=source_path,
                client=client,
                provider_extract_one=provider_extract_one,
                provider=provider,
                apply_changes=apply_changes,
                write_db=write_db,
            )
            for k in ("reviewed", "written", "declined", "rejected", "missing_entry"):
                counts[k] += book_counts[k]
    finally:
        if write_db is not None:
            write_db.close()

    click.echo("", err=True)
    click.echo(f"Review summary: {counts}", err=True)
    if not apply_changes:
        click.echo("(dry-run; pass --apply to write)", err=True)


PAID_CONCURRENCY_WARN_THRESHOLD = 16
PAID_PROVIDERS = ("anthropic", "gemini")


def _warn_if_paid_concurrency_too_high(provider: str, concurrency: int) -> None:
    """Warn when ``--concurrency`` exceeds ``PAID_CONCURRENCY_WARN_THRESHOLD``
    against a paid provider (Anthropic, Gemini).

    The IntRange on ``--concurrency`` is permissive (1-64) because Ollama on
    a multi-GPU host can usefully run many parallel requests. Paid providers
    have per-org RPM caps, so a copy-pasted ``--concurrency 64`` against
    Anthropic blows past the cap and ~64 entries fail simultaneously with
    transport_error_result. ``provider_retry`` softens this with 429
    backoff but won't save the operator from a 4-5x oversaturation.

    Surfaced by Claude review on PR #62 (wyrd-mk7).
    """
    if provider in PAID_PROVIDERS and concurrency > PAID_CONCURRENCY_WARN_THRESHOLD:
        click.echo(
            f"warning: --concurrency {concurrency} is high for paid provider "
            f"'{provider}' (RPM-capped). 4-8 is the documented sweet spot; "
            f"4-{PAID_CONCURRENCY_WARN_THRESHOLD} is reasonable. The 429 "
            f"retry-with-backoff layer will absorb brief overruns but a "
            f"sustained {concurrency}-way fan-out will saturate the cap.",
            err=True,
        )


def _build_llm_client(
    provider: str,
    *,
    model: str | None = None,
    ollama_url: str | None = None,
    timeout: float,
) -> tuple[Any, Any]:
    """Construct (client, extract_one) for one of the three supported
    LLM providers. Centralises the 3-branch dispatch that used to
    live inline in lexicon_mine_llm and lexicon_review (wyrd-lqw).

    ``ollama_url`` only applies to the ``ollama`` provider; ignored
    for gemini/anthropic. ``model`` overrides the provider's default
    when set.

    Raises ``click.ClickException`` for unknown provider names so the
    CLI exits cleanly without a stack trace.
    """
    if provider == "ollama":
        kwargs: dict[str, Any] = {"timeout_s": timeout}
        if ollama_url:
            kwargs["base_url"] = ollama_url
        if model:
            kwargs["model"] = model
        return llm_extractor.OllamaClient(**kwargs), llm_extractor.extract_one
    if provider == "gemini":
        kwargs = {"timeout_s": timeout}
        if model:
            kwargs["model"] = model
        return gemini_extractor.GeminiClient(**kwargs), gemini_extractor.extract_one
    if provider == "anthropic":
        kwargs = {"timeout_s": timeout}
        if model:
            kwargs["model"] = model
        return anthropic_extractor.AnthropicClient(**kwargs), anthropic_extractor.extract_one
    raise click.ClickException(f"unknown provider: {provider}")


def _select_already_extracted_toponyms(db_path: Path, source_id: str) -> set[str]:
    """Return modern_name set for toponyms already extracted from `source_id`.

    Used by `mine-llm --declines-only` (wyrd-bgr) to skip re-mining toponyms
    that already have any toponym_etymology row for this source — regardless
    of which Tier-1 provider produced it. The decline-recovery workflow is:
    after Qwen leaves declines on a book, point a stronger model at the gaps
    without paying to re-extract the entries Qwen already got.

    Returns an empty set if the source row doesn't exist yet (fresh book) or
    if no etymology rows have been written for it.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            """
            SELECT DISTINCT t.modern_name
            FROM toponym t
            JOIN toponym_etymology te ON te.toponym_id = t.id
            WHERE te.source_id = ?
            """,
            (source_id,),
        ).fetchall()
        return {r[0] for r in rows}
    finally:
        conn.close()


def _select_review_candidates(
    *,
    db_path: Path,
    provider_tag: str,
    confidence: tuple[str, ...],
    book: str | None,
    limit: int | None,
    include_haiku: bool = False,
    include_sonnet: bool = False,
) -> list[sqlite3.Row]:
    """Pick `toponym_etymology` rows eligible for Tier-2 re-extraction.

    Eligibility: row was extracted by an LLM provider (notes contain
    `extracted_by:llm:` or a tier-1 marker like `qwen` / `ollama`),
    confidence is in the `confidence` set, and the same (toponym, source)
    doesn't already have a row tagged with the chosen Tier-2 provider.

    ``include_haiku`` (wyrd-eca) widens the candidate set to also pick up
    rows tagged ``extracted_by:anthropic:...haiku...``. Off by default so
    the original Qwen→Gemini pipeline stays untouched. Use this when a
    non-Celtic book was mined Tier-1 with Haiku (Bannister 1916,
    Harrison 1898, etc.) and needs a Gemini second-pass — Haiku rows
    otherwise slip past the candidate query because their notes don't
    match the Ollama/Qwen patterns.

    ``include_sonnet`` (wyrd-0rsk) is the parallel companion: also pick
    up rows tagged ``extracted_by:anthropic:...sonnet...``. Off by
    default. Pass when re-reviewing a book whose Tier-1 mining used
    Sonnet (Joyce 1875 v1+v2, Moore 1890, Johnston 1892, Morgan 1887
    are the current set). Composes orthogonally with include_haiku —
    setting both picks up Anthropic-tagged rows from either model.

    The provider_tag NOT EXISTS guard still ensures we don't re-review
    a row that already has a Tier-2 pass from the chosen provider, so
    flipping the flags is idempotent.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        placeholders = ",".join("?" * len(confidence))
        provider_clauses = [
            "te.notes LIKE '%extracted_by:llm:%'",
            "te.notes LIKE '%qwen%'",
            "te.notes LIKE '%llama%'",
            "te.notes LIKE '%ollama%'",
        ]
        if include_haiku:
            # Match any anthropic-haiku tagged row regardless of model
            # version suffix (claude-haiku-4-5-20251001, future
            # claude-haiku-X-X-20260101, etc.).
            provider_clauses.append("te.notes LIKE '%anthropic:%haiku%'")
        if include_sonnet:
            # Mirror of the haiku clause for sonnet-tagged Tier-1 rows.
            # Same wildcard shape: matches any model version suffix.
            provider_clauses.append("te.notes LIKE '%anthropic:%sonnet%'")
        provider_or = " OR ".join(provider_clauses)
        sql = f"""
            SELECT te.id, te.toponym_id, te.source_id, te.confidence,
                   t.modern_name, t.region
            FROM toponym_etymology te
            JOIN toponym t ON t.id = te.toponym_id
            WHERE te.confidence IN ({placeholders})
              AND te.source_id IN (SELECT id FROM source)
              AND ({provider_or})
              AND NOT EXISTS (
                  SELECT 1 FROM toponym_etymology te2
                  WHERE te2.toponym_id = te.toponym_id
                    AND te2.source_id  = te.source_id
                    AND te2.notes LIKE ?
                    AND te2.id != te.id
              )
            """
        args: list[object] = list(confidence) + [f"%{provider_tag}%"]
        if book:
            sql += " AND te.source_id = ?"
            args.append(book)
        sql += " ORDER BY te.source_id, t.modern_name"
        if limit:
            sql += " LIMIT ?"
            args.append(limit)
        return conn.execute(sql, args).fetchall()
    finally:
        conn.close()


@dataclass(frozen=True)
class _ReviewOutcome:
    """One row's classified review result, ready for the loop to act on."""

    counter: str  # 'rejected' | 'declined' | 'written'
    log_line: str
    failures: tuple[str, ...] = ()  # rejection reasons (rejected only)
    persist_entry: Any = None  # ParsedEntry to write (written only)


def _classify_review_result(name: str, result: Any, dt: float) -> _ReviewOutcome:
    """Take a provider's LLMResult and produce a structured outcome.

    Mirrors the writer's drop-low gate (`lexicon.py:1394`) so dry-run and
    apply-mode summaries both report what's actually persisted.
    """
    if not result.accepted:
        failures = tuple(f.reason for f in result.failures)
        return _ReviewOutcome(
            counter="rejected",
            log_line=f"  REJECT  {name:24} ({dt:.1f}s)  " + ", ".join(failures),
            failures=failures,
        )
    if result.entry is None or not result.entry.elements:
        conf = result.entry.confidence if result.entry else "?"
        return _ReviewOutcome(
            counter="declined",
            log_line=f"  DECLINE {name:24} ({dt:.1f}s)  conf={conf}",
        )
    breakdown = "+".join(el.form for el in result.entry.elements)
    if result.entry.confidence == "low":
        return _ReviewOutcome(
            counter="declined",
            log_line=f"  DROPPED {name:24} ({dt:.1f}s)  {breakdown:30}  conf=low (writer skips)",
        )
    return _ReviewOutcome(
        counter="written",
        log_line=(
            f"  REVIEW  {name:24} ({dt:.1f}s)  {breakdown:30}  conf={result.entry.confidence}"
        ),
        persist_entry=result.entry,
    )


def _record_review_run(
    write_db: LexiconDB,
    *,
    source_id: str,
    provider: str,
    model: str,
    counts: dict[str, int],
    by_failure: dict[str, int],
    started_at: str,
) -> None:
    """Persist a `mining_run` audit row for one book's review pass (D23)."""
    record_mining_run(
        write_db,
        source_id=source_id,
        provider=provider,
        model=model,
        mode="review",
        parsed_count=counts["written"] + counts["declined"] + counts["rejected"],
        accepted=counts["written"],
        declined=counts["declined"],
        rejected=counts["rejected"],
        by_failure=by_failure or None,
        started_at=started_at,
        completed_at=datetime.now(UTC).isoformat(),
    )


def _review_one_book(
    *,
    source_id: str,
    rows: list[sqlite3.Row],
    source_path: Path,
    client: Any,
    provider_extract_one: Any,
    provider: str,
    apply_changes: bool,
    write_db: LexiconDB | None,
) -> dict[str, int]:
    """Run the per-book Tier-2 review loop and persist its audit row."""
    counts: dict[str, int] = {
        "reviewed": 0,
        "written": 0,
        "declined": 0,
        "rejected": 0,
        "missing_entry": 0,
    }
    by_failure: dict[str, int] = {}

    text = source_path.read_text()
    parsed = _select_parser_and_run(text, "auto")
    entries_by_name: dict[str, object] = {p.toponym: p for p in parsed}

    click.echo(f"=== {source_id} ({len(rows)} candidates) ===", err=True)
    started_at = datetime.now(UTC).isoformat()

    for row in rows:
        name = row["modern_name"]
        parsed_entry = entries_by_name.get(name)
        if parsed_entry is None:
            counts["missing_entry"] += 1
            continue

        t0 = time.time()
        result = provider_extract_one(
            client,
            toponym=name,
            body=parsed_entry.body_text,
            suffix_hint=parsed_entry.section_suffix,
        )
        outcome = _classify_review_result(name, result, time.time() - t0)
        click.echo(outcome.log_line, err=True)
        counts["reviewed"] += 1
        for reason in outcome.failures:
            by_failure[reason] = by_failure.get(reason, 0) + 1

        if outcome.counter == "written" and apply_changes and write_db is not None:
            ingest_parsed_entries(
                write_db, [outcome.persist_entry], source_id, region=row["region"]
            )
            counts["written"] += 1
        elif outcome.counter != "written":
            counts[outcome.counter] += 1
        # else: dry-run "would-write" — log line shown above but don't count

    if (
        apply_changes
        and write_db is not None
        and (counts["written"] or counts["declined"] or counts["rejected"])
    ):
        _record_review_run(
            write_db,
            source_id=source_id,
            provider=provider,
            model=client.model,
            counts=counts,
            by_failure=by_failure,
            started_at=started_at,
        )

    return counts


def _import_mining_log_record(
    db: LexiconDB,
    raw_line: str,
    line_no: int,
    known_sources: set[str],
    apply_changes: bool,
) -> str | None:
    """Parse and ingest one JSONL mining-log line.

    Returns:
      "inserted"  — record was inserted (or would be, in dry-run)
      "skipped"   — UNIQUE conflict, already in DB
      <error msg> — parse / validation failure (string for caller to log)
      None        — blank line or comment, no action needed

    Tolerates malformed lines: invalid JSON, non-object JSON, missing
    or unknown source_id, non-numeric counts, by_failure not an object,
    invalid mode hitting the CHECK constraint — every failure mode
    surfaces as an error string and the caller continues. The back-fill
    runs once over noisy transcript-extracted data; one bad row should
    not kill the whole import.
    """
    raw = raw_line.strip()
    if not raw or raw.startswith("#"):
        return None
    try:
        rec = json.loads(raw)
        if not isinstance(rec, dict):
            raise ValueError(f"expected JSON object, got {type(rec).__name__}")
        src = rec.get("source_id")
        if not src:
            raise ValueError("missing source_id")
        if src not in known_sources:
            raise ValueError(f"unknown source_id {src!r}")
        by_failure = rec.get("by_failure")
        if by_failure is not None and not isinstance(by_failure, dict):
            raise ValueError("by_failure must be a JSON object")
        accepted = int(rec.get("accepted") or 0)
        declined = int(rec.get("declined") or 0)
        rejected = int(rec.get("rejected") or 0)
        parsed_val = rec.get("parsed_count")
        parsed = int(parsed_val) if parsed_val is not None else accepted + declined + rejected
        if not apply_changes:
            return "inserted"
        row_id = record_mining_run(
            db,
            source_id=src,
            provider=rec.get("provider") or "unknown",
            model=rec.get("model") or "unknown",
            mode=rec.get("mode") or "mine",
            parsed_count=parsed,
            accepted=accepted,
            declined=declined,
            rejected=rejected,
            by_failure=by_failure,
            started_at=rec.get("started_at"),
            completed_at=rec.get("completed_at"),
            notes=rec.get("notes"),
        )
        return "inserted" if row_id else "skipped"
    except (
        json.JSONDecodeError,
        ValueError,
        TypeError,
        AttributeError,
        sqlite3.Error,
    ) as e:
        return f"line {line_no}: {e}"


@lexicon.command("import-mining-log")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    default=False,
    help="Actually insert mining_run rows. Without this flag the command runs as a dry-run.",
)
def lexicon_import_mining_log(path: Path, db_path: Path, apply_changes: bool) -> None:
    """Back-fill the mining_run table from a JSONL log of historical runs.

    The input file is line-delimited JSON; each line describes one
    mining or review run with the fields written by record_mining_run.
    Recovered from session transcripts, hand-curated logs, etc. Use this
    once after wyrd-ej4 lands to seed the audit table with everything
    that already happened pre-table.

    Required fields per JSON line:
      source_id, provider, model, mode, accepted, declined, rejected
    Optional:
      parsed_count (defaults to accepted+declined+rejected),
      by_failure (object), started_at, completed_at, notes

    Idempotent on (source_id, provider, model, mode, completed_at) so
    re-running with the same JSONL is safe.
    """
    inserted = 0
    skipped = 0
    errors: list[str] = []

    with LexiconDB(db_path) as db:
        # Check that mining_run exists; if not, run migrate first.
        tables = {
            r["name"] for r in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "mining_run" not in tables:
            raise click.ClickException(
                "mining_run table missing — run `wyrd kenning lexicon migrate` first."
            )

        # Pre-load known source ids so we can flag rows that reference a
        # source that doesn't exist yet (rather than FK-failing on insert).
        known = {r["id"] for r in db.conn.execute("SELECT id FROM source")}

        with path.open() as f:
            for ln, raw in enumerate(f, 1):
                outcome = _import_mining_log_record(db, raw, ln, known, apply_changes)
                if outcome == "inserted":
                    inserted += 1
                elif outcome == "skipped":
                    skipped += 1
                elif outcome is not None:
                    errors.append(outcome)

    verb = "inserted" if apply_changes else "would insert"
    click.echo(f"{verb}: {inserted}", err=True)
    if apply_changes:
        click.echo(f"skipped (duplicate): {skipped}", err=True)
    if errors:
        click.echo(f"errors: {len(errors)}", err=True)
        for e in errors[:20]:
            click.echo(f"  {e}", err=True)
        if len(errors) > 20:
            click.echo(f"  ... +{len(errors) - 20} more", err=True)
    if not apply_changes:
        click.echo("(dry-run; pass --apply to write)", err=True)


@lexicon.command("mine-wiktextract-corpus")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--sources-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("sources"),
    show_default=True,
    help="Directory containing wiktextract_*.jsonl slice files.",
)
@click.option(
    "--culture",
    "cultures",
    type=click.Choice([*CULTURES, "all"]),
    default=("all",),
    multiple=True,
    show_default=True,
    help="Cultures to mine for (repeatable, or 'all'). Each culture's "
    "place-name corpus is decomposed against the current bundle; "
    "unaccounted fragments seed the wiktextract lookup.",
)
@click.option(
    "--min-length",
    type=int,
    default=3,
    show_default=True,
    help="Skip fragments shorter than this. Bigrams are too noisy + overlap "
    "with the joiner question.",
)
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    default=False,
    help="Write to the lexicon DB. Without --apply this is a dry run that "
    "reports per-culture / per-language hit counts only.",
)
def lexicon_mine_wiktextract_corpus(
    db_path: Path,
    sources_dir: Path,
    cultures: tuple[str, ...],
    min_length: int,
    apply_changes: bool,
) -> None:
    """Empirically mine Wiktionary headwords against unaccounted-fragment misses.

    For each requested culture: decompose every name in the corresponding
    bundled place-name corpus against the current ``meanings.json``;
    collect unaccounted fragments of length ≥ ``--min-length``; look each
    up against the per-culture allowed wiktextract slices; for every hit,
    upsert an etymon + gloss + tag + citation in the lexicon DB tagged at
    the synthetic ``wiktionary-empirical`` source.

    Treated like ``rando-port`` (empirical class) — admitted to the
    bundle without the D4 ≥3-scholar-witness gate. The export-meanings
    promotion CTE has a third UNION branch for this source.

    The bundled ``meanings.json`` and per-culture place-name corpora are
    read via ``importlib.resources``, so this command runs from any cwd
    as long as ``--sources-dir`` points at the wiktextract dumps.
    """
    target_cultures: list[str] = sorted(set(CULTURES if "all" in cultures else cultures))
    meanings_data = _load_meanings_data(None)
    tmpdir, place_names_paths_by_culture = _stage_place_names_paths(target_cultures)
    try:
        fragments_by_culture = _collect_fragments_per_culture(
            target_cultures, place_names_paths_by_culture, meanings_data, min_length
        )
        started_at = datetime.now(UTC).isoformat()
        with LexiconDB(db_path) as db:
            counts = mine_corpus(
                db,
                fragments_by_culture=fragments_by_culture,
                sources_dir=sources_dir,
                apply=apply_changes,
            )
        _print_mine_summary(counts, apply_changes)
        if not apply_changes:
            return
        # Empirical etymons land with no position_pref — without
        # per-position reflex rows the bundle export emits a single
        # undecorated modern_usage that the runtime treats as post-only.
        # derive_positions scans the relevant cultures' corpora and
        # writes one reflex per observed position so the export emits
        # one bundle word per position with proper dash markers.
        with LexiconDB(db_path) as db:
            pos_counts = derive_positions(db, place_names_paths_by_culture, apply=True)
            # wyrd-qepf / D24: persist a mining_run audit row so the
            # operation is queryable from the DB alongside LLM mining
            # runs. Empirical mining doesn't have accept/decline/reject
            # semantics in the LLM sense, so the rich structural
            # counters land in by_failure (the audit-shaped JSON
            # column). parsed_count/accepted use the closest analogues:
            # candidates considered (fragments_input) and (frag, lang)
            # entries that landed (wiktionary_hits).
            _record_empirical_mining_audit(db, counts, pos_counts, started_at, target_cultures)
        _print_position_summary(pos_counts)
    finally:
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)


def _record_empirical_mining_audit(
    db: LexiconDB,
    counts: dict[str, Any],
    pos_counts: dict[str, Any],
    started_at: str,
    target_cultures: list[str],
) -> None:
    """Write the wyrd-qepf audit row for an empirical wiktextract mining run.

    Stuffs the empirical-specific structural counters (etymons_upserted,
    glosses/tags/citations added, by_culture, by_language, position
    classification breakdown) into the by_failure JSON field; the
    LLM-style accept/decline/reject columns are zeroed because the
    empirical pipeline doesn't have those semantics.

    The provider/model/mode strings follow the LLM mining_run convention
    (see record_mining_run); 'wiktionary-empirical' / 'corpus-substring-v1'
    / 'mine' make this run distinguishable from LLM mining without
    needing a separate audit table.
    """
    by_culture_serializable = {
        culture: dict(stats) for culture, stats in counts.get("by_culture", {}).items()
    }
    by_language_serializable = dict(counts.get("by_language", {}))
    # Nest the empirical structural counters under an "empirical_metrics"
    # key so that consumers reading the by_failure JSON column don't
    # mistake them for failure tags. The mining_run schema uses
    # by_failure for audit-shaped JSON regardless of column name; the
    # nesting tags this row's payload as success metrics, not failures.
    audit = {
        "empirical_metrics": {
            "etymons_upserted": counts.get("etymons_upserted", 0),
            "glosses_added": counts.get("glosses_added", 0),
            "tags_added": counts.get("tags_added", 0),
            "citations_added": counts.get("citations_added", 0),
            "wiktionary_hits": counts.get("wiktionary_hits", 0),
            "by_culture": by_culture_serializable,
            "by_language": by_language_serializable,
            "position_reflexes_written": pos_counts.get("reflexes_written", 0),
            "target_cultures": target_cultures,
        },
    }
    record_mining_run(
        db,
        source_id=WIKTIONARY_EMPIRICAL_SOURCE_ID,
        provider="wiktionary-empirical",
        model="corpus-substring-v1",
        mode="mine",
        parsed_count=counts.get("fragments_input", 0),
        accepted=counts.get("wiktionary_hits", 0),
        declined=0,
        rejected=0,
        by_failure=audit,
        started_at=started_at,
        completed_at=datetime.now(UTC).isoformat(),
    )


def _stage_place_names_paths(
    target_cultures: list[str],
) -> tuple[Path, dict[str, Path]]:
    """Stage bundled place_names corpora to a tmpdir as files since
    compute_unaccounted_fragments / derive_positions want Path-shaped
    inputs and importlib.resources only hands back text. Returns the
    tmpdir + the per-culture path map; caller is responsible for
    cleanup via the returned tmpdir.

    On any failure during staging (missing resource, write_text raising),
    the partially-populated tmpdir is cleaned up before re-raising so
    the caller's try/finally never sees a leaked dir.
    """
    import shutil
    import tempfile

    tmpdir = Path(tempfile.mkdtemp(prefix="wyrd-empirical-mine-"))
    paths: dict[str, Path] = {}
    try:
        for culture in target_cultures:
            text = (
                resources.files("wyrd.generators.kenning.data")
                .joinpath(f"{culture}_place_names.json")
                .read_text()
            )
            p = tmpdir / f"{culture}_place_names.json"
            p.write_text(text)
            paths[culture] = p
    except Exception:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise
    return tmpdir, paths


def _collect_fragments_per_culture(
    target_cultures: list[str],
    place_names_paths_by_culture: dict[str, Path],
    meanings_data: list[dict[str, Any]],
    min_length: int,
) -> dict[str, Counter]:
    """Run compute_unaccounted_fragments per culture, echo a one-line
    summary per culture, and return the {culture: Counter} map."""
    fragments_by_culture: dict[str, Counter] = {}
    for culture in target_cultures:
        frags = compute_unaccounted_fragments(
            culture,
            place_names_paths_by_culture[culture],
            meanings_data,
            min_length=min_length,
        )
        fragments_by_culture[culture] = frags
        click.echo(
            f"culture={culture} unaccounted_fragments={len(frags)} "
            f"total_occurrences={sum(frags.values())}",
            err=True,
        )
    return fragments_by_culture


def _print_mine_summary(counts: dict[str, Any], apply_changes: bool) -> None:
    """Echo per-culture + per-language hit summaries to stderr."""
    click.echo(
        f"fragments_input={counts['fragments_input']} wiktionary_hits={counts['wiktionary_hits']}",
        err=True,
    )
    if apply_changes:
        click.echo(
            f"etymons_upserted={counts['etymons_upserted']} "
            f"glosses_added={counts['glosses_added']} "
            f"tags_added={counts['tags_added']} "
            f"citations_added={counts['citations_added']}",
            err=True,
        )
    click.echo("per-culture summary:", err=True)
    for culture, sub in sorted(counts["by_culture"].items()):
        click.echo(
            f"  {culture:10} fragments={sub['fragments']:6d} hits={sub['hits']:6d}",
            err=True,
        )
    click.echo("per-language hits:", err=True)
    for lang, n in counts["by_language"].most_common():
        click.echo(f"  {lang:25s} {n:6d}", err=True)
    if not apply_changes:
        click.echo("(dry-run; pass --apply to write)", err=True)


def _print_position_summary(pos_counts: dict[str, Any]) -> None:
    """Echo derive_positions counters to stderr."""
    click.echo("position derivation:", err=True)
    click.echo(
        f"  etymons_examined={pos_counts['etymons_examined']} "
        f"reflex_rows_upserted={pos_counts['reflex_rows_upserted']} "
        f"links_added={pos_counts['links_added']} "
        f"no_observed_position={pos_counts['etymons_with_no_observed_position']}",
        err=True,
    )
    for pos, n in sorted(pos_counts["by_position"].items()):
        click.echo(f"  {pos:10s} reflexes_for={n}", err=True)


@lexicon.command("mine-wiktextract-forms")
@click.argument("slice_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Stop after N entries (for smoke testing). Default: walk the full slice.",
)
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    default=False,
    help="Write etymon_text_match rows. Without --apply this is a dry run that "
    "reports counts only.",
)
def lexicon_mine_wiktextract_forms(
    slice_path: Path,
    db_path: Path,
    limit: int | None,
    apply_changes: bool,
) -> None:
    """wyrd-vx09: ingest a wiktextract slice's ``forms`` arrays as
    etymon_text_match rows so the bundle's per-language _variants
    pool gets non-OE/ON variant data (audit follow-up wyrd-f6v4).

    Walks the slice file and, for each entry whose canonical
    headword is already an etymon in the lexicon, writes each
    meaningful form (plural, comparative, soft mutation, etc.) as a
    text-match row tagged ``source='wiktionary-forms'``. Verbal
    conjugation forms and Wiktionary table-rendering scaffolding
    (``table-tags`` / ``inflection-template`` / ``error-unrecognized-
    form``) are filtered out — they're not useful for place-name
    variant pools.

    Pass ``--apply`` to write; without it the run is a dry-count
    pass for size-checking before committing to a full slice ingest.

    Idempotent: re-runs increment match_count on existing
    (etymon_id, source_id, matched_form) rows rather than duplicating.
    """
    with LexiconDB(db_path) as db:
        counts = mine_corpus_forms(
            db,
            slice_path,
            apply=apply_changes,
            limit=limit,
        )
    suffix = "--apply" if apply_changes else "(dry-run)"
    click.echo(
        f"slice={slice_path.name} {suffix} "
        f"entries_walked={counts['entries_walked']} "
        f"forms_processed={counts['forms_processed']} "
        f"forms_written={counts['forms_written']} "
        f"etymons_missing={counts['etymons_missing']}",
        err=True,
    )


@lexicon.command("path")
def lexicon_path() -> None:
    """Print the resolved lexicon DB path and exit.

    Resolution order:

    \b
    1. WYRD_LEXICON_DB env var
    2. ~/.wyrd/lexicon.db (default)

    Triggers the one-time legacy → ~/.wyrd migration if a pre-wyrd-366
    DB still lives in the repo's data/ dir. Useful for confirming what
    DB a CLI run would hit.
    """
    click.echo(default_lexicon_path())


@lexicon.command("era-reflex")
@click.argument("etymon_id", type=int)
@click.option(
    "--era",
    "era_input",
    required=True,
    help=(
        "Target era: a year (1086), a cell label (oe-late), or a "
        "family/label pair (english/me). Resolved against the "
        "etymon's source-language family."
    ),
)
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
def lexicon_era_reflex(etymon_id: int, era_input: str, db_path: Path) -> None:
    """wyrd-skm Phase 3.0b: print cognate-cluster mates of ETYMON_ID
    that match the target era. Used by wyrd-rni / wyrd-381 demos to
    render etymons across historical strata.

    One row per matching cluster mate on stdout, tab-separated:
    ``<reflex_id>\\t<canonical_form>\\t<language>``. The metadata
    header (etymon canonical / language / target cell) goes to
    stderr.

    Empty output when the etymon has no cognate_id (~76% of OE
    toponym etymons today — cluster_cognates coverage gap) or when
    the target cell has no canonical language tag (most Norse +
    Latin late cells; see ``CANONICAL_LANGUAGE_FOR_CELL`` in era.py).
    Extending coverage means adding entries to that dict; the picker
    surfaces any new entry the next call.
    """
    with LexiconDB(db_path) as db:
        row = db.conn.execute(
            "SELECT canonical_form, language FROM etymon WHERE id = ?",
            (etymon_id,),
        ).fetchone()
        if row is None:
            click.echo(f"no etymon with id={etymon_id}", err=True)
            raise click.exceptions.Exit(1)

        family = language_family(row["language"])
        if family is None:
            click.echo(
                f"etymon {etymon_id} ({row['canonical_form']!r}, "
                f"language={row['language']!r}) has no era family — "
                "proto-languages and untracked classical languages "
                "don't define era cells",
                err=True,
            )
            raise click.exceptions.Exit(1)

        # Resolve --era directly to (family, cell) — the era-reflex
        # picker is keyed on cell labels, so round-tripping through
        # year ranges (resolve_era_input → era_cell) loses information
        # for open-low cells like 'oe-early' where start=None.
        try:
            family, cell = era_cell_for_input(era_input, default_family=family)
        except ValueError as exc:
            click.echo(str(exc), err=True)
            raise click.exceptions.Exit(1) from exc

        reflexes = etymon_era_reflexes(db, etymon_id, target_family_cell=(family, cell))

    click.echo(
        f"# era-reflex etymon={etymon_id} "
        f"({row['canonical_form']!r}, {row['language']}) "
        f"target={family}/{cell}",
        err=True,
    )
    if not reflexes:
        click.echo("(no cluster mates at this era)", err=True)
        return
    for reflex in reflexes:
        click.echo(f"{reflex.etymon_id}\t{reflex.form}\t{reflex.language}")


@lexicon.command("era-cell")
@click.argument("language")
@click.argument("year", type=int)
def lexicon_era_cell(language: str, year: int) -> None:
    """Resolve a (language, year) pair to its D5-2 era-cell label.

    Prints ``family/label`` (e.g. ``english/oe-late``) on stdout. Exits
    non-zero with a stderr message if the language has no era family
    (proto-languages, untracked classical languages) or if the year
    falls outside any defined cell. Useful for ad-hoc cell lookup
    when sanity-checking attestation data.
    """
    family = language_family(language)
    if family is None:
        click.echo(
            f"language {language!r} has no era family defined "
            "(proto-languages and untracked classical languages "
            "intentionally don't get cell labels)",
            err=True,
        )
        raise click.exceptions.Exit(1)
    label = era_cell(language, year)
    if label is None:
        click.echo(
            f"year {year} is outside the defined cells for family {family!r}",
            err=True,
        )
        raise click.exceptions.Exit(1)
    click.echo(f"{family}/{label}")


@lexicon.command("migrate")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
def lexicon_migrate(db_path: Path) -> None:
    """Apply pending schema migrations to an existing lexicon DB.

    Idempotent. Currently brings older DBs up to the lemma_id /
    inflection schema and rebuilds the etymon_consensus view.
    """
    with LexiconDB(db_path) as db:
        applied = migrate_schema(db)
    click.echo("Migrations:", err=True)
    for name, did in applied.items():
        click.echo(f"  {name:32} {'applied' if did else 'no-op'}", err=True)


@lexicon.command("parse-pages")
@click.argument("source_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--limit",
    type=int,
    default=10,
    show_default=True,
    help="Number of detected (page, headword-fragment) pairs to print.",
)
def lexicon_parse_pages(source_path: Path, limit: int) -> None:
    """Audit running-header coverage on a source book (wyrd-9kh.5, wyrd-8st).

    Tries both header conventions — Mawer-style `<HEADWORD> <number>`
    and Skeat-§ `§ N. NAMES IN -X. <page>` — and reports which matched
    along with a sample. Use this to decide whether a book is amenable
    to page-anchored citations.
    """
    text = source_path.read_text(errors="replace")
    headers, parser = detect_running_headers(text)
    click.echo(f"{source_path.name}: {len(headers)} running header(s) detected (parser: {parser})")
    if not headers:
        click.echo(
            "  No headers matched either Mawer-style or Skeat-§ patterns. The "
            "book may use a third convention entirely.",
            err=True,
        )
        return
    click.echo(f"  Page range: {headers[0][1]} → {headers[-1][1]}")
    click.echo(f"  Sample (first {min(limit, len(headers))}):")
    for offset, page in headers[:limit]:
        # Snippet of the matched line for visual inspection.
        line_end = text.find("\n", offset)
        line = text[offset : line_end if line_end > 0 else offset + 80].strip()
        click.echo(f"    p.{page:>4}  @offset={offset:>7}  {line}")


@lexicon.command("backfill-pages")
@click.argument(
    "sources_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    default=False,
    help="Actually write page numbers. Without this, dry-run reporting only.",
)
@click.option(
    "--source",
    "source_filter",
    type=str,
    default=None,
    help="Process only the source matching this id (filename stem). Default: every .txt in sources_dir.",
)
def lexicon_backfill_pages(
    sources_dir: Path,
    db_path: Path,
    apply_changes: bool,
    source_filter: str | None,
) -> None:
    """Backfill etymon_citation.page + toponym_etymology.page (wyrd-azv).

    Walks each .txt in <sources_dir>, tries both Mawer-style and
    Skeat-§ running header conventions, and for rows where page IS
    NULL locates the row's quoted excerpt in body text to resolve a
    page number. Books matching neither convention are skipped
    (reported as no_headers).

    Idempotent: re-runs only touch rows where page is still NULL.
    """
    totals = dict.fromkeys(_BACKFILL_TOTALS_KEYS, 0)

    with LexiconDB(db_path) as db:
        for path in sorted(sources_dir.glob("*.txt")):
            source_id = path.stem
            if source_filter and source_id != source_filter:
                continue
            text = path.read_text(errors="replace")
            counts = backfill_citation_pages(db, source_id, text, apply=apply_changes)
            totals["sources_processed"] += 1
            if counts["no_headers"]:
                totals["sources_no_headers"] += 1
                click.echo(f"  [{source_id:50}]  no recognized headers; skipped", err=True)
                continue
            click.echo(
                f"  [{source_id:50}]  cit={counts['citations_updated']:>4}  "
                f"ety={counts['etymologies_updated']:>4}  "
                f"miss={counts['quote_not_in_text']:>3}",
                err=True,
            )
            for key in _BACKFILL_PER_SOURCE_KEYS:
                totals[key] += counts[key]

    if source_filter and totals["sources_processed"] == 0:
        click.echo(
            f"warn: --source {source_filter!r} matched no .txt file in {sources_dir}",
            err=True,
        )
    _print_backfill_totals(totals, apply_changes)


_BACKFILL_PER_SOURCE_KEYS = (
    "citations_updated",
    "etymologies_updated",
    "quote_not_in_text",
    "no_quote",
    "before_first_page",
)
_BACKFILL_TOTALS_KEYS = (*_BACKFILL_PER_SOURCE_KEYS, "sources_processed", "sources_no_headers")


def _print_backfill_totals(totals: dict[str, int], apply_changes: bool) -> None:
    """Render the backfill summary. Header counts come from sources_*
    keys; per-source counters listed below come from _BACKFILL_PER_SOURCE_KEYS
    in display order."""
    click.echo(
        f"\nTotals: {totals['sources_processed']} sources processed, "
        f"{totals['sources_no_headers']} skipped (no headers).",
        err=True,
    )
    for key in _BACKFILL_PER_SOURCE_KEYS:
        click.echo(f"  {key:<22}= {totals[key]}", err=True)
    if not apply_changes:
        click.echo("\n(dry-run; pass --apply to write)", err=True)


@lexicon.command("link-lemmas")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    default=False,
    help="Actually link lemmas. Without this, dry-run reporting only.",
)
def lexicon_link_lemmas(db_path: Path, apply_changes: bool) -> None:
    """Link inflected etymons to their lemma rows.

    For each etymon with lemma_id IS NULL, tries common inflection-strip
    rules per language. If the stripped stem matches an EXISTING etymon
    row in the same language, sets lemma_id + inflection on the inflected
    etymon. Conservative — never fabricates a lemma.

    Run after `migrate` (and after any major mining/OCR-cluster work).
    """
    with LexiconDB(db_path) as db:
        result = link_lemmas(db, apply=apply_changes)

    click.echo(
        f"Lemma linkage candidates: {result['candidates']}  "
        f"({'applied' if result['applied'] else 'dry-run'})",
        err=True,
    )
    if result["sample"]:
        click.echo("First 25 proposals:", err=True)
        for p in result["sample"]:
            click.echo(
                f"  [{p['language']:14}] {p['inflected_form']:18} → lemma={p['lemma_form']:14} ({p['inflection']})",
                err=True,
            )
    if not apply_changes:
        click.echo("(dry-run; pass --apply to write)", err=True)


@lexicon.command("reverse-search")
@click.argument(
    "sources_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    default=False,
    help="Insert search-attested citation rows. Without this, dry-run only.",
)
@click.option(
    "--min-form-length",
    type=int,
    default=4,
    show_default=True,
    help="Skip rando etymons with canonical_form shorter than this (avoids "
    "false-positive substring matches).",
)
def lexicon_reverse_search(
    sources_dir: Path,
    db_path: Path,
    apply_changes: bool,
    min_form_length: int,
) -> None:
    """Reverse-direction verification + parser-bug diagnostic.

    For each rando-port etymon with no scholarly citation, search the
    bundled source-book texts (sources_dir/*.txt) for the form as a
    word-boundary match. Two outputs:

    1. PROMOTION: forms that appear in scholarly texts but were never
       formally extracted get a 'search-attested:<book>' citation,
       moving them from 'unverified rando' to 'mentioned in published
       philology'.

    2. PARSER-BUG DIAGNOSTIC: forms that appear MANY times in source
       text but were never produced as an etymon by any extractor are
       likely systemic pipeline misses. Worth investigating the prompt,
       the parser segmentation, or both.

    Conservative: minimum form length defaults to 4 to avoid false-
    positive substring matches.
    """
    with LexiconDB(db_path) as db:
        result = reverse_search_attestations(
            db, sources_dir, apply=apply_changes, min_form_length=min_form_length
        )

    click.echo(
        f"Rando-only candidates: {result['rando_only_candidates']}",
        err=True,
    )
    click.echo(
        f"Etymons with text-match: {result['etymons_with_match']} "
        f"({100 * result['etymons_with_match'] / max(result['rando_only_candidates'], 1):.1f}%)",
        err=True,
    )
    click.echo(
        f"Total citation records: {result['total_match_records']}",
        err=True,
    )

    click.echo("", err=True)
    click.echo("=== Sample promotions (first 25) ===", err=True)
    for s in result["sample"]:
        books = ", ".join(f"{b}({c})" for b, c, _ in s["matches"])
        click.echo(f"  {s['form']:24} {books}", err=True)
        # Show one sample snippet per etymon so the user can see context
        if s["matches"]:
            _, _, snippet = s["matches"][0]
            click.echo(f"      → {snippet[:140]}", err=True)

    if result["parser_bug_suspects"]:
        click.echo("", err=True)
        click.echo(
            "=== Parser-bug suspects: high text-count, zero extraction-count ===",
            err=True,
        )
        click.echo(
            "(These forms appear often in scholarly texts but our LLM "
            "extractor never emitted them as etymons. Likely pipeline gaps.)",
            err=True,
        )
        for s in result["parser_bug_suspects"][:20]:
            click.echo(
                f"  {s['form']:20} text-count={s['text_count']:>5}  "
                f"extracted=0  in: {', '.join(s['books'])}",
                err=True,
            )

    if not apply_changes:
        click.echo("", err=True)
        click.echo("(dry-run; pass --apply to write search-attested citations)", err=True)


@lexicon.command("lookup-attested-years")
@click.argument(
    "sources_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    default=False,
    help="Populate etymon_text_match.attested_year. Without this, dry-run only.",
)
def lexicon_lookup_attested_years(
    sources_dir: Path,
    db_path: Path,
    apply_changes: bool,
) -> None:
    """Post-mining stage: scan source bodies for date citations near
    matched forms and populate etymon_text_match.attested_year (D5-1 /
    wyrd-3ux). Foundation for D5-2 era-cell sampling.

    LLM-free, idempotent, reversible. Per D21/D22 enrichment-only — no
    mining evidence is touched. Re-runs against unchanged data are
    no-ops; only rows where attested_year IS NULL are scanned.

    Reverse via:

        wyrd kenning lexicon clear-enrichment --stage=attested-years --apply
    """
    with LexiconDB(db_path) as db:
        result = lookup_attested_years(db, sources_dir, apply=apply_changes)

    etm = result["etymon_text_match"]
    te = result["toponym_etymology"]
    click.echo("lookup-attested-years:", err=True)
    click.echo(
        f"  etymon_text_match  scanned={etm['rows_scanned']:>5}  "
        f"candidates={etm['candidates']:>5}  rows_written={etm['rows_written']:>5}",
        err=True,
    )
    click.echo(
        f"  toponym_etymology  scanned={te['rows_scanned']:>5}  "
        f"candidates={te['candidates']:>5}  rows_written={te['rows_written']:>5}",
        err=True,
    )
    if result["sources_missing"]:
        click.echo(
            f"  warn: {result['sources_missing']} source_id(s) referenced in text-match "
            "rows have no .txt file under sources_dir; their rows were skipped.",
            err=True,
        )
    if not apply_changes:
        click.echo("(dry-run; pass --apply to write attested_year values)", err=True)


@lexicon.command("mine-attestations")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    default=False,
    help="Insert into toponym_attestation. Without this, dry-run only.",
)
@click.option(
    "--progress-every",
    type=int,
    default=500,
    show_default=True,
    help="Emit a progress line every N rows scanned.",
)
def lexicon_mine_attestations(
    db_path: Path,
    apply_changes: bool,
    progress_every: int,
) -> None:
    """wyrd-skm Phase 3.0a: extract (form, year) attestation pairs from
    toponym_etymology.notes and populate toponym_attestation.

    Pattern set: ``FORM in YEAR`` / ``FORM, YEAR`` / ``FORM in Domesday
    Book`` / ``Domesday has FORM`` / ``FORM, D.B.``. Year range is
    700-1700 (post-Roman through pre-modern), matching the
    lookup-attested-years filter. Page-reference false positives
    (``"Bedinga feld, p. 59"``) are rejected via the same page-marker
    guard that ``_earliest_year_in_notes`` uses.

    LLM-free, idempotent (unique-index on
    ``toponym_id, form, date_year, source_doc``), reversible:

        wyrd kenning lexicon clear-enrichment --stage=attestations --apply

    The output rows are the raw input that wyrd-skm Phase 3.0b will
    derive per-etymon period-keyed surface forms from.
    """
    with LexiconDB(db_path) as db:
        click.echo("mine-attestations:", err=True)
        result = mine_toponym_attestations(db, apply=apply_changes, progress_every=progress_every)

    click.echo(
        f"  scanned={result['rows_scanned']:>5}  "
        f"rows_with_pairs={result['rows_with_pairs']:>5}  "
        f"candidates={result['candidates']:>5}  "
        f"rows_written={result['rows_written']:>5}",
        err=True,
    )
    if not apply_changes:
        click.echo("(dry-run; pass --apply to write toponym_attestation rows)", err=True)


@lexicon.command("project-period-forms")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    default=False,
    help="Insert into etymon_period_form. Without this, dry-run only.",
)
@click.option(
    "--progress-every",
    type=int,
    default=200,
    show_default=True,
    help="Emit a progress line every N rows scanned.",
)
def lexicon_project_period_forms(
    db_path: Path,
    apply_changes: bool,
    progress_every: int,
) -> None:
    """wyrd-unuo Phase 3.3: project per-etymon period forms from
    toponym_attestation rows.

    For each binary toponym breakdown, segment the attested historical
    form's suffix against the last morpheme's known reflexes (canonical
    form + cognate-cluster mates + etymon variants); the remaining
    prefix is the first morpheme's projected period form. Bradford
    (1377) "Bradeford" projects to "Brade" (OE 'brad') + "ford" (OE
    'ford'); Chesterton (1210) "Cestretone" projects to "Cestre" (OE
    'ceaster') + "tone" (OE 'tūn').

    Output is the Tier 3 fallback for ``etymon_era_reflexes`` —
    closes the coverage gap on isolated OE etymons (no cognate_id, no
    descent edges) when the toponym had a binary attested form.

    LLM-free, idempotent (unique-index on
    ``etymon_id, form, date_year, source_doc``), reversible:

        wyrd kenning lexicon clear-enrichment --stage=period-forms --apply

    Run after ``mine-attestations`` populates the toponym_attestation
    table; re-run any time the cluster mates expand to recover more
    suffix matches.
    """
    with LexiconDB(db_path) as db:
        click.echo("project-period-forms:", err=True)
        result = project_period_forms(db, apply=apply_changes, progress_every=progress_every)

    click.echo(
        f"  scanned={result['rows_scanned']:>5}  "
        f"rows_projected={result['rows_projected']:>5}  "
        f"candidates={result['candidates']:>5}  "
        f"rows_written={result['rows_written']:>5}",
        err=True,
    )
    if not apply_changes:
        click.echo("(dry-run; pass --apply to write etymon_period_form rows)", err=True)


@lexicon.command("fuzzy-search")
@click.argument(
    "sources_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    default=False,
    help="Write fuzzy matches to etymon_text_match. Without this, dry-run.",
)
@click.option(
    "--max-distance",
    type=int,
    default=1,
    show_default=True,
    help="Max Levenshtein distance for fuzzy match. 1 catches spelling "
    "variants; 2 produces many more false positives.",
)
@click.option(
    "--min-form-length",
    type=int,
    default=5,
    show_default=True,
    help="Skip etymons shorter than this (short forms have too many fuzzy neighbors).",
)
def lexicon_fuzzy_search(
    sources_dir: Path,
    db_path: Path,
    apply_changes: bool,
    max_distance: int,
    min_form_length: int,
) -> None:
    """Fuzzy-match rando-only etymons against scholarly source text.

    For each rando-only etymon NOT already exact-matched, find tokens in
    source text within Levenshtein distance N. Filter by gloss-anchor:
    only count a fuzzy candidate if one of the etymon's glosses appears
    within ±100 chars of the candidate's first occurrence.

    The gloss anchor is the safety mechanism. Without it, edit-distance-1
    would match 'bere' (barley) to 'bera' (bear) — completely different
    morphemes. With it, we only count matches where the meaning aligns.

    19th-c scholarly OE spelling wasn't standardized — denu/dene/denū are
    all the same word. This command finds those variants automatically.

    Run AFTER `reverse-search` (which handles exact matches).
    """
    with LexiconDB(db_path) as db:
        result = fuzzy_search_attestations(
            db,
            sources_dir,
            apply=apply_changes,
            max_distance=max_distance,
            min_form_length=min_form_length,
        )

    click.echo(
        f"Candidates with at least one gloss: {result['candidates_with_gloss']}",
        err=True,
    )
    click.echo(
        f"Etymons with fuzzy match (gloss-confirmed): {result['etymons_with_fuzzy_match']}",
        err=True,
    )
    click.echo(f"Total fuzzy match records: {result['total_match_records']}", err=True)
    click.echo("", err=True)
    click.echo("=== Sample fuzzy matches (first 25) ===", err=True)
    for s in result["sample"]:
        ms = ", ".join(f"{src}:{mf}(d={d},×{c})" for src, mf, d, c in s["matches"])
        click.echo(f"  {s['form']:20} → {ms}", err=True)
    if not apply_changes:
        click.echo("", err=True)
        click.echo("(dry-run; pass --apply to write to etymon_text_match)", err=True)


def _classify_dry_run_row_counts(
    result: Any,
    case: Any,
) -> dict[str, int]:
    """Mirror ``apply_disambiguator_result``'s per-row classification
    without writing. Used by the CLI's dry-run path so the summary
    tells the operator what WOULD have happened on --apply."""
    n_rows = len(case.text_match_ids)
    if result.chosen_etymon_id is None:
        return {"kept": 0, "reassigned": 0, "deleted": n_rows}
    counts = {"kept": 0, "reassigned": 0, "deleted": 0}
    for current in case.current_etymon_ids:
        key = "kept" if result.chosen_etymon_id == current else "reassigned"
        counts[key] += 1
    return counts


def _disambiguate_dispatch(
    client: Any,
    case: Any,
    expander: Any,
    *,
    agentic: bool,
    max_expansions: int,
    total_char_cap: int,
) -> tuple[Any, Any]:
    """Single-shot vs agentic dispatch. Returns ``(result, stats)``;
    ``stats`` is None on the single-shot path. Raises RuntimeError on
    transport failure — caller increments the error counter and
    continues."""
    from wyrd.generators.kenning.disambiguator import (
        disambiguate_one,
        disambiguate_one_agentic,
    )

    if agentic:
        result, stats = disambiguate_one_agentic(
            client,
            case,
            expander,
            max_expansions=max_expansions,
            total_char_cap=total_char_cap,
        )
        return result, stats
    return disambiguate_one(client, case), None


@lexicon.command("disambiguate-fuzzy")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    default=False,
    help="Apply the disambiguator's verdicts to etymon_text_match. Without this, dry-run.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Process at most N ambiguous rows (smoke testing).",
)
@click.option(
    "--model",
    type=str,
    default=None,
    help="Override the Gemini model. Default: $WYRD_GEMINI_MODEL or gemini-2.5-flash.",
)
@click.option(
    "--timeout",
    type=float,
    default=120.0,
    show_default=True,
)
@click.option(
    "--agentic",
    is_flag=True,
    default=False,
    help=(
        "wyrd-uct: enable the agentic expand-context loop. The model can "
        "request a wider snippet when 200 chars isn't enough. Requires "
        "--sources-dir so wider snippets can be drawn from the source body."
    ),
)
@click.option(
    "--sources-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help=(
        "Directory of normalized source *.txt files. Required when "
        "--agentic is set. Same shape as the directory used by "
        "reverse-search and fuzzy-search."
    ),
)
@click.option(
    "--max-expansions",
    # >=0 — zero means "no expansions, force commit on round 0",
    # which is a valid degenerate config equivalent to single-shot
    # but with the agentic schema's defensive parsing.
    type=click.IntRange(min=0),
    default=3,
    show_default=True,
    help="Max expand-context rounds per case (only used with --agentic).",
)
@click.option(
    "--total-char-cap",
    type=click.IntRange(min=1),
    default=2000,
    show_default=True,
    help="Hard cap on total snippet chars per case (only used with --agentic).",
)
def lexicon_disambiguate_fuzzy(
    db_path: Path,
    apply_changes: bool,
    limit: int | None,
    model: str | None,
    timeout: float,
    agentic: bool,
    sources_dir: Path | None,
    max_expansions: int,
    total_char_cap: int,
) -> None:
    """Re-resolve fuzzy text-matches that have multiple plausible etymons.

    Per wyrd-6z7: when a body word X is within edit distance ≤ 1 of more
    than one canonical etymon, gloss anchoring alone isn't reliable. This
    command finds those ambiguous rows and asks Gemini to pick the right
    etymon based on the surrounding passage. The chosen row's `method`
    becomes 'llm-disambiguated-v1' and the model's reasoning is stored in
    `disambiguator_reason`. Rows where the model says "none" are deleted.

    Cost gate: rows with only one candidate etymon (no ambiguity) are
    skipped entirely — no LLM call.

    With --agentic (wyrd-uct), the disambiguator can request a wider
    snippet when the initial 200-char window isn't enough. Caps at
    --max-expansions rounds or --total-char-cap chars.
    """
    # Local imports keep the disambiguator deps off the cold-start path
    # for unrelated CLI commands.
    from wyrd.generators.kenning.disambiguator import (
        SnippetExpander,
        apply_disambiguator_result,
        find_ambiguous_rows,
    )
    from wyrd.generators.kenning.gemini_extractor import GeminiClient

    if agentic and sources_dir is None:
        raise click.UsageError("--agentic requires --sources-dir")

    with LexiconDB(db_path) as db:
        cases = find_ambiguous_rows(db, max_distance=1, limit=limit)

    if not cases:
        click.echo("No ambiguous fuzzy rows found.", err=True)
        return

    mode_label = "Agentic" if agentic else "Single-shot"
    click.echo(
        f"{len(cases)} ambiguous rows found. {mode_label} "
        f"{'apply' if apply_changes else 'dry-run'}.",
        err=True,
    )

    client = GeminiClient(timeout_s=timeout, **({"model": model} if model else {}))
    expander = SnippetExpander(sources_dir=sources_dir) if agentic else None

    counts = {"kept": 0, "reassigned": 0, "deleted": 0, "errors": 0}
    # Aggregate agentic stats so the operator can see how often the loop
    # had to widen and how often the cap forced a commit. Not surfaced
    # in the non-agentic path.
    agentic_totals = {"expansions": 0, "forced_commits": 0}
    write_db = LexiconDB(db_path) if apply_changes else None
    try:
        for case in cases:
            try:
                result, stats = _disambiguate_dispatch(
                    client,
                    case,
                    expander,
                    agentic=agentic,
                    max_expansions=max_expansions,
                    total_char_cap=total_char_cap,
                )
            except RuntimeError as e:
                counts["errors"] += 1
                click.echo(f"  ERROR  {case.matched_form:20} {e}", err=True)
                continue
            if stats is not None:
                agentic_totals["expansions"] += stats.expansions
                if stats.forced_commit:
                    agentic_totals["forced_commits"] += 1
            choice_str = (
                f"id={result.chosen_etymon_id}" if result.chosen_etymon_id is not None else "none"
            )
            n_rows = len(case.text_match_ids)
            row_label = f" ({n_rows} rows)" if n_rows > 1 else ""
            click.echo(
                f"  {case.matched_form:20}{row_label} → {choice_str} "
                f"(conf={result.confidence}) — {result.reason[:80]}",
                err=True,
            )
            if apply_changes and write_db is not None:
                row_counts = apply_disambiguator_result(write_db, case, result)
                # Commit per-case so a crash mid-run doesn't lose previously
                # disambiguated rows (the LLM calls have already been paid for).
                write_db.commit()
            else:
                row_counts = _classify_dry_run_row_counts(result, case)
            for k, v in row_counts.items():
                counts[k] += v
    finally:
        if write_db is not None:
            write_db.close()

    click.echo("", err=True)
    click.echo(f"Disambiguator summary: {counts}", err=True)
    if agentic:
        click.echo(
            f"Agentic stats: expansions={agentic_totals['expansions']} "
            f"forced_commits={agentic_totals['forced_commits']}",
            err=True,
        )
    if not apply_changes:
        click.echo("(dry-run; pass --apply to update etymon_text_match)", err=True)


@lexicon.command("normalize-ocr")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    default=False,
    help="Actually perform the merge. Without this flag the command runs as a dry-run.",
)
def lexicon_normalize_ocr(db_path: Path, apply_changes: bool) -> None:
    """Cluster etymons by OCR-normalized form and merge variants.

    Without --apply this is a dry-run that just reports what would happen.
    Pass --apply to actually merge redundant etymon rows. Each merge group
    keeps the lowest-id row and repoints citations, glosses, tags, reflex
    links, and breakdown elements onto it.
    """
    with LexiconDB(db_path) as db:
        result = cluster_ocr_variants(db, apply=apply_changes)

    click.echo(
        f"OCR-variant groups: {result['groups']}, "
        f"etymons {'merged' if apply_changes else 'mergeable'}: "
        f"{result['etymons_merged']}",
        err=True,
    )
    if result["sample_groups"]:
        click.echo("Sample merges (first 20):", err=True)
        for g in result["sample_groups"]:
            members = ", ".join(g["members"])
            click.echo(
                f"  [{g['language']:12}] {g['normalized']:20} <- {members}",
                err=True,
            )
    if not apply_changes:
        click.echo("(dry-run; pass --apply to commit)", err=True)


@lexicon.command("clear-enrichment")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--stage",
    type=click.Choice(
        [
            "ocr",
            "lemmas",
            "text-match",
            "cognates",
            "attested-years",
            "attestations",
            "period-forms",
            "all-derived",
        ]
    ),
    required=True,
    help=(
        "Which enrichment stage to clear. 'ocr' un-marks merged_into_id, "
        "'lemmas' resets lemma_id/inflection/lemma_method, 'text-match' "
        "drops the etymon_text_match table, 'cognates' resets "
        "cognate_id/cognate_method (D27/wyrd-81n), 'attested-years' resets "
        "etymon_text_match.attested_year (D5-1/wyrd-3ux), 'attestations' "
        "drops every toponym_attestation row (wyrd-skm Phase 3.0a), "
        "'period-forms' drops every etymon_period_form row "
        "(wyrd-unuo Phase 3.3), 'all-derived' does all seven."
    ),
)
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    default=False,
    help="Actually perform the clear. Without this flag the command runs as a dry-run.",
)
def lexicon_clear_enrichment(db_path: Path, stage: str, apply_changes: bool) -> None:
    """Reset one or more enrichment stages so they can be re-run.

    Mining evidence (etymons, citations, glosses, tags, etymon_descent,
    toponyms, toponym_etymology rows) is never touched. Per D21, only
    enrichment inferences are reversible.

    Run before re-running the relevant stage with new heuristics:

        wyrd kenning lexicon clear-enrichment --stage=ocr --apply
        wyrd kenning lexicon normalize-ocr --apply
    """
    with LexiconDB(db_path) as db:
        result = clear_enrichment(db, stage=stage, apply=apply_changes)

    verb = "cleared" if apply_changes else "would clear"
    click.echo(f"Stage: {result['stage']}", err=True)
    if result["ocr_merges_to_clear"]:
        click.echo(f"  {verb} {result['ocr_merges_to_clear']} merged_into_id rows", err=True)
    if result["lemma_links_to_clear"]:
        click.echo(f"  {verb} {result['lemma_links_to_clear']} lemma links", err=True)
    if result["text_match_rows_to_clear"]:
        click.echo(f"  {verb} {result['text_match_rows_to_clear']} text-match rows", err=True)
    if result["cognate_assignments_to_clear"]:
        click.echo(
            f"  {verb} {result['cognate_assignments_to_clear']} cognate_id assignments", err=True
        )
    if result.get("attested_years_to_clear"):
        click.echo(f"  {verb} {result['attested_years_to_clear']} attested_year values", err=True)
    if result.get("attestation_rows_to_clear"):
        click.echo(
            f"  {verb} {result['attestation_rows_to_clear']} toponym_attestation rows",
            err=True,
        )
    if result.get("period_form_rows_to_clear"):
        click.echo(
            f"  {verb} {result['period_form_rows_to_clear']} etymon_period_form rows",
            err=True,
        )
    if not apply_changes:
        click.echo("(dry-run; pass --apply to commit)", err=True)


@lexicon.command("cluster-cognates")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    default=False,
    help="Actually write cognate_id assignments. Without this flag the command runs as a dry-run.",
)
def lexicon_cluster_cognates(db_path: Path, apply_changes: bool) -> None:
    """Walk the etymon_descent graph from each root and assign cognate_id
    to every reachable descendant (D27 / wyrd-81n).

    Roots are etymons that participate in inheritance/borrowing edges as
    parent but never as child — typically Proto-* forms. Every descendant
    reachable via inheritance + borrowing edges gets cognate_id =
    root.id, so cross-language cognates cluster behind a single canonical
    pointer. The 'cognate' edge type is a peer relation, NOT a chain,
    and does NOT bridge synsets.

    Run after Wiktionary mining (wyrd-4rt) populates etymon_descent.
    Reversible via `clear-enrichment --stage=cognates --apply`.
    """
    with LexiconDB(db_path) as db:
        result = cluster_cognates(db, apply=apply_changes)

    verb = "assigned" if apply_changes else "would assign"
    click.echo(
        f"cluster-cognates: {result['roots']} root(s) walked, "
        f"{verb} cognate_id on {result['candidates']} etymon(s)",
        err=True,
    )
    if apply_changes:
        click.echo(f"  rows_written = {result['rows_written']}", err=True)
    if result["cycle_orphans"]:
        click.echo(
            f"  warn: {result['cycle_orphans']} etymon(s) sit in a bridging-edge "
            f"cycle with no external root and were left unassigned",
            err=True,
        )
    if not apply_changes:
        click.echo("(dry-run; pass --apply to commit)", err=True)


# Place-name dictionaries write 'celtic' as a generic catch-all for
# morphemes whose specific Celtic-family origin isn't pinned. Wiktextract
# uses specific codes. The default --candidates resolves 'celtic' onto
# Proto-Celtic (most ancestral) first, then Old-* and Middle-* forms,
# then modern reflexes.
_CELTIC_CANDIDATES_DEFAULT = (
    "proto-celtic",
    "old-irish",
    "old-welsh",
    "old-breton",
    "middle-irish",
    "middle-welsh",
    "middle-breton",
    "irish",
    "welsh",
    "scottish-gaelic",
    "manx",
    "breton",
    "cornish",
)


@lexicon.command("bridge-language")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--generic",
    "generic_lang",
    type=str,
    required=True,
    help=(
        "Generic language tag to bridge (e.g. 'celtic'). Each etymon with "
        "this language gets merged_into_id set to the highest-priority "
        "specific-language match with the same canonical_form."
    ),
)
@click.option(
    "--candidates",
    "candidates_csv",
    type=str,
    default=None,
    help=(
        "Comma-separated specific languages to consider, in priority order. "
        "If omitted and --generic is 'celtic', uses the built-in Celtic "
        "candidate list (Proto-Celtic > Old-* > Middle-* > modern reflexes)."
    ),
)
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    default=False,
    help="Actually write merged_into_id. Without this, dry-run reporting only.",
)
def lexicon_bridge_language(
    db_path: Path,
    generic_lang: str,
    candidates_csv: str | None,
    apply_changes: bool,
) -> None:
    """Bridge a generic-language etymon family onto specific-language
    canonicals via merged_into_id.

    Place-name dictionaries write generic tags like 'celtic' for
    morphemes whose specific Celtic-family origin isn't pinned.
    Wiktextract entries are language-specific. This pass canonicalizes
    each generic-language etymon onto a specific-language match with
    the same canonical_form, so cluster-cognates' redirect-aware walk
    will fold the generic etymons into the cross-source cognate clusters.

    Run AFTER wiktextract ingest + AFTER normalize-ocr. Re-running
    cluster-cognates after this pass is recommended to refresh
    cognate_id assignments.

    Reverse via `clear-enrichment --stage=ocr --apply` (the bridge uses
    the same merged_into_id mechanism).
    """
    if candidates_csv is not None:
        candidates = tuple(s.strip() for s in candidates_csv.split(",") if s.strip())
        if not candidates:
            click.echo(
                "error: --candidates resolved to an empty list (need at least "
                "one specific-language code)",
                err=True,
            )
            sys.exit(1)
    elif generic_lang == "celtic":
        candidates = _CELTIC_CANDIDATES_DEFAULT
    else:
        click.echo(
            f"error: --candidates required when --generic is not 'celtic' "
            f"(no built-in default for {generic_lang!r})",
            err=True,
        )
        sys.exit(1)

    with LexiconDB(db_path) as db:
        result = bridge_generic_language(
            db,
            generic_lang=generic_lang,
            candidate_langs=candidates,
            apply=apply_changes,
        )

    verb = "bridged" if apply_changes else "would bridge"
    click.echo(
        f"bridge-language: {result['generic_etymons']} generic '{generic_lang}' etymon(s) examined; "
        f"{verb} {result['bridged']}, {result['unmatched']} unmatched.",
        err=True,
    )
    if apply_changes:
        click.echo(f"  rows_written = {result['rows_written']}", err=True)
    if not apply_changes:
        click.echo("(dry-run; pass --apply to commit)", err=True)


@lexicon.command("bridge-celtic-forms")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    default=False,
    help="Actually write merged_into_id. Without this, dry-run reporting only.",
)
def lexicon_bridge_celtic_forms(db_path: Path, apply_changes: bool) -> None:
    """Bridge celtic place-name etymons to Wiktionary lemmas via a
    hand-curated form→lemma table.

    Where bridge-language only handles same-form lookups (celtic/dun →
    old-irish/dun), this pass uses a curated table to map inflected /
    Anglicized celtic forms (choill, drum, lough, kin, gcorr) to their
    lemma equivalents (coill, druim, loch, ceann, corr) and then searches
    across the celtic candidate languages, preferring clustered targets.

    Iterates ALL celtic rows (including pre-existing tombstones from
    bridge-language) so a stub-bridge to an unclustered old-irish entry
    can be re-routed to a clustered modern Irish / Welsh / Scottish-Gaelic
    counterpart in the same pass.

    Run AFTER bridge-language and AFTER wiktextract ingest.
    Re-run cluster-cognates afterward to refresh cognate_id assignments
    via the merged_into_id rollup.

    Reverse via `clear-enrichment --stage=ocr --apply` (the bridge uses
    the same merged_into_id mechanism).
    """
    with LexiconDB(db_path) as db:
        result = bridge_celtic_forms(db, apply=apply_changes)

    verb = "bridged" if apply_changes else "would bridge"
    click.echo(
        f"bridge-celtic-forms: {result['examined']} celtic etymon(s) examined; "
        f"{verb} {result['bridged']}, {result['unmatched']} unmatched, "
        f"{result['missing_target']} missing-target.",
        err=True,
    )
    if result["missing_target"]:
        click.echo(
            f"  warn: {result['missing_target']} table entry/entries name a "
            f"lemma that doesn't exist as a canonical etymon in any candidate "
            f"language (operator may need to ingest more or fix the table)",
            err=True,
        )
    if apply_changes:
        click.echo(f"  rows_written = {result['rows_written']}", err=True)
    if not apply_changes:
        click.echo("(dry-run; pass --apply to commit)", err=True)


@lexicon.command("bridge-phonological-oe")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    default=False,
    help="Actually write merged_into_id. Without this, dry-run reporting only.",
)
def lexicon_bridge_phonological_oe(db_path: Path, apply_changes: bool) -> None:
    """Bridge OE place-name forms to Wiktionary canonicals.

    Place-name dictionaries write modernized OE forms (`ton`, `lea`,
    `burgh`, `dale`); Wiktionary uses scholarly orthography (`tūn`,
    `lēah`, `burh`, `dæl`). normalize-ocr handles macron-strip OCR
    variants but NOT vowel-weakening / silent-e / gh-spelling shifts.
    This pass uses a hand-curated mapping table to merge known pairs
    via merged_into_id (D22 non-destructive).

    Run AFTER wiktextract ingest. Re-run cluster-cognates afterward
    to refresh cognate_id assignments via the merged_into_id rollup.

    Reverse via `clear-enrichment --stage=ocr --apply` (the bridge
    uses the same merged_into_id mechanism).
    """
    with LexiconDB(db_path) as db:
        result = bridge_phonological_oe(db, apply=apply_changes)

    verb = "bridged" if apply_changes else "would bridge"
    click.echo(
        f"bridge-phonological-oe: {result['examined']} canonical OE etymon(s) examined; "
        f"{verb} {result['bridged']}, {result['unmatched']} unmatched.",
        err=True,
    )
    if result["missing_target"]:
        click.echo(
            f"  warn: {result['missing_target']} table entry/entries name a "
            f"target that doesn't exist as a canonical OE etymon "
            f"(operator may need to ingest more or extend the bridge table)",
            err=True,
        )
    if apply_changes:
        click.echo(f"  rows_written = {result['rows_written']}", err=True)
    if not apply_changes:
        click.echo("(dry-run; pass --apply to commit)", err=True)


@lexicon.command("bridge-phonological-on")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    default=False,
    help="Actually write merged_into_id. Without this, dry-run reporting only.",
)
def lexicon_bridge_phonological_on(db_path: Path, apply_changes: bool) -> None:
    """Bridge ON place-name forms to Wiktionary canonicals.

    Place-name dictionaries write Anglicized / modernized ON forms
    (`by`, `holm`, `dale`, `thwaite`, `kirk`, `gardr`); Wiktionary
    uses scholarly orthography with acutes, þ/ð, -r endings, ǫ
    (`býr`, `hólmr`, `dalr`, `þveit`, `kirkja`, `garðr`). This pass
    uses a hand-curated mapping table to merge known pairs via
    merged_into_id (D22 non-destructive).

    Run AFTER wiktextract ingest. Re-run cluster-cognates afterward
    to refresh cognate_id assignments via the merged_into_id rollup.

    Reverse via `clear-enrichment --stage=ocr --apply` (the bridge
    uses the same merged_into_id mechanism).
    """
    with LexiconDB(db_path) as db:
        result = bridge_phonological_on(db, apply=apply_changes)

    verb = "bridged" if apply_changes else "would bridge"
    click.echo(
        f"bridge-phonological-on: {result['examined']} canonical ON etymon(s) examined; "
        f"{verb} {result['bridged']}, {result['unmatched']} unmatched.",
        err=True,
    )
    if result["missing_target"]:
        click.echo(
            f"  warn: {result['missing_target']} table entry/entries name a "
            f"target that doesn't exist as a canonical ON etymon "
            f"(operator may need to ingest more or extend the bridge table)",
            err=True,
        )
    if apply_changes:
        click.echo(f"  rows_written = {result['rows_written']}", err=True)
    if not apply_changes:
        click.echo("(dry-run; pass --apply to commit)", err=True)


@lexicon.command("ingest-wiktionary")
@click.argument(
    "source_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    default=False,
    help="Actually write etymons + descent edges. Without this, dry-run reporting only.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Stop after processing N entries (smoke / partial run).",
)
@click.option(
    "--since-line",
    type=int,
    default=0,
    help="Skip the first N lines of the JSONL (resume across multi-hour ingest sessions).",
)
def lexicon_ingest_wiktionary(
    source_path: Path,
    db_path: Path,
    apply_changes: bool,
    limit: int | None,
    since_line: int,
) -> None:
    """Ingest a wiktextract JSONL dump into etymon + etymon_descent
    (wyrd-4rt / wyrd-hun).

    Walks each entry's Etymology + Descendants sections, upserts the
    referenced etymons (across languages), and inserts descent edges
    with edge_type per the D27 taxonomy: inh→inheritance, bor→borrowing,
    der→derivation, cal→calque, desc→inheritance (downward).

    Pure structured-data ingest — no LLM, no form-in-body validation.
    Source attribution is the synthetic 'wiktionary' source row.

    Run `wyrd kenning lexicon cluster-cognates --apply` afterward to
    populate cognate_id from the new descent edges.

    Both .jsonl and .jsonl.gz are accepted; the suffix selects the
    open mode.
    """
    with LexiconDB(db_path) as db:
        result = ingest_wiktextract_path(
            db, source_path, apply=apply_changes, limit=limit, since_line=since_line
        )

    click.echo(
        f"ingest-wiktionary: {result['lines_read']} line(s) read, "
        f"{result['entries_parsed']} entry/entries parsed",
        err=True,
    )
    if result["entries_skipped_malformed"]:
        click.echo(
            f"  skipped {result['entries_skipped_malformed']} malformed line(s)",
            err=True,
        )
    click.echo(f"  upward_edges    = {result['upward_edges']}", err=True)
    click.echo(f"  downward_edges  = {result['downward_edges']}", err=True)
    if result["unsupported_templates"]:
        click.echo(
            f"  unsupported_templates = {result['unsupported_templates']} "
            f"(operator may want to extend the maps in wiktextract_ingester.py)",
            err=True,
        )
    # wyrd-ha9q Phase 2a: pronunciation + multi-script capture stats.
    # Only show non-zero counters so the slim Latin-script slices
    # (Old English, Old Norse, etc. that wiktextract usually leaves
    # without `sounds` arrays) don't print three zero lines.
    pron = result.get("pronunciation_captured", 0)
    orig = result.get("original_script_captured", 0)
    trans = result.get("transliteration_captured", 0)
    if pron or orig or trans:
        click.echo(
            f"  pronunciation captured: ipa={pron} original_script={orig} transliteration={trans}",
            err=True,
        )
    # wyrd-vsvi: tag-extraction stats from sense categories.
    tags_added = result.get("tags_added", 0)
    entries_with_tags = result.get("entries_with_tags", 0)
    if tags_added:
        click.echo(
            f"  tags captured: {tags_added} tag-writes across {entries_with_tags} entries",
            err=True,
        )
    if not apply_changes:
        click.echo("(dry-run; pass --apply to commit)", err=True)


@lexicon.command("derive-english-shaped")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    default=False,
    help=(
        "Write derived english_shaped values to the etymon table. Without "
        "this flag, the command runs as a dry-run that prints what WOULD "
        "be written and the per-language coverage delta."
    ),
)
@click.option(
    "--language",
    "language_filter",
    type=str,
    default=None,
    help=(
        "Restrict derivation to a single language code (e.g. 'he', 'ar', "
        "'sa'). Useful for incremental smoke runs. When omitted, every "
        "non-Latin-script language with NULL english_shaped is processed."
    ),
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help=(
        "Process at most N rows. For smoke / sample runs before a full "
        "pass. Combined with --language to narrow further."
    ),
)
@click.option(
    "--reshape",
    is_flag=True,
    default=False,
    help=(
        "Re-derive english_shaped on rows that ALREADY have a value (not "
        "just NULL ones). Use after editing KNOWN_FORM_OVERRIDES or the "
        "diacritic-strip rules to push the new derivation through. Default "
        "behavior leaves existing values alone."
    ),
)
def lexicon_derive_english_shaped(
    db_path: Path,
    apply_changes: bool,
    language_filter: str | None,
    limit: int | None,
    reshape: bool,
) -> None:
    """Derive english_shaped values for non-Latin-script etymon rows
    (wyrd-ha9q Phase 2b).

    Walks every etymon row whose `english_shaped` is NULL (or all rows
    when ``--reshape`` is set), runs each through
    ``english_shaping.derive_english_shaped``, and UPDATEs the row when
    the derivation produced a non-None result. Latin-script source
    languages are skipped at the derivation level (the function returns
    None for them); the SQL filter narrows further to the wave-2
    non-Latin set so we don't sweep millions of OE / latin / etc. rows
    that wouldn't yield anything anyway.

    Mining-progress shape: one [N/total] line every 1,000 rows + a
    final summary, matching the convention in CLAUDE.md "Mining
    progress reporting".
    """
    # Build the WHERE predicate via parameterized SQL — never f-string
    # interpolate user-supplied values (--language) into the query.
    if language_filter:
        lang_pred = "language = ?"
        lang_params: tuple[str, ...] = (language_filter,)
    else:
        placeholders = ",".join("?" * len(_PHASE2A_NON_LATIN_LANGS))
        lang_pred = f"language IN ({placeholders})"
        lang_params = _PHASE2A_NON_LATIN_LANGS
    where_pred = lang_pred + ("" if reshape else " AND english_shaped IS NULL")
    # --limit: int(limit) is safe (Click already cast) and an int is
    # not interpolatable to anything dangerous.
    limit_clause = f" LIMIT {int(limit)}" if limit else ""

    with LexiconDB(db_path) as db:
        rows = db.conn.execute(
            f"""SELECT id, canonical_form, language,
                       transliteration, pronunciation_ipa
                FROM etymon
                WHERE {where_pred}
                ORDER BY language, id
                {limit_clause}""",
            lang_params,
        ).fetchall()

    total = len(rows)
    click.echo(
        f"derive-english-shaped: {total} candidate row(s) "
        f"({'applying' if apply_changes else 'dry-run'}; "
        f"reshape={'yes' if reshape else 'no'})",
        err=True,
    )

    written = 0
    skipped_no_input = 0
    by_language: dict[str, int] = {}
    progress_every = 1000

    db_writer = LexiconDB(db_path) if apply_changes else None
    try:
        for completed, row in enumerate(rows, start=1):
            shaped = derive_english_shaped(
                canonical_form=row["canonical_form"],
                language=row["language"],
                transliteration=row["transliteration"],
                pronunciation_ipa=row["pronunciation_ipa"],
            )
            if shaped is None:
                skipped_no_input += 1
            else:
                written += 1
                by_language[row["language"]] = by_language.get(row["language"], 0) + 1
                if db_writer is not None:
                    db_writer.conn.execute(
                        "UPDATE etymon SET english_shaped = ? WHERE id = ?",
                        (shaped, row["id"]),
                    )
                    if completed % progress_every == 0:
                        db_writer.commit()
            if completed % progress_every == 0 or completed == total:
                click.echo(
                    f"  [{completed}/{total}]  written={written} "
                    f"skipped_no_input={skipped_no_input}",
                    err=True,
                )
        if db_writer is not None:
            db_writer.commit()
    finally:
        if db_writer is not None:
            db_writer.close()

    click.echo(
        f"derive-english-shaped: {written} written, {skipped_no_input} skipped (no usable input).",
        err=True,
    )
    if by_language:
        click.echo("  per-language:", err=True)
        for lang, n in sorted(by_language.items(), key=lambda kv: -kv[1]):
            click.echo(f"    {lang}: {n}", err=True)
    if not apply_changes:
        click.echo("(dry-run; pass --apply to write english_shaped)", err=True)


@lexicon.command("classify-stratum")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--language",
    type=click.Choice(["welsh", "french", "old-english", "old-norse"]),
    required=True,
    help=(
        "Language family to classify. Welsh covers etymons with "
        "language IN ('welsh', 'middle-welsh', 'old-welsh', "
        "'cel-bry-pro'); French covers ('french', 'old-french', "
        "'middle-french', 'norman-french', 'vulgar-latin', 'frk', "
        "'cel-gau'); Old English covers language='old-english' "
        "(loan-source ancestors only — dialect axis deferred); Old "
        "Norse covers ('old-norse', 'gmq-osw' Old Swedish, 'gmq-oda' "
        "Old Danish)."
    ),
)
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    default=False,
    help="Write the proposed stratum values. Without this, dry-run.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help=(
        "Overwrite existing stratum values. Default: skip rows that "
        "already have a non-NULL stratum (idempotent re-runs)."
    ),
)
def lexicon_classify_stratum(
    db_path: Path,
    language: str,
    apply_changes: bool,
    force: bool,
) -> None:
    """Within-language stratum tagging (wyrd-lr4).

    Partitions a language's etymons into named buckets (e.g. for
    'welsh': latin-loan, english-loan, brittonic-substrate,
    medieval-welsh, native-welsh) so generation can filter to a
    specific register. Strata are language-specific — see
    ``wyrd/generators/kenning/strata.py`` for the per-language
    vocabularies and the heuristic rules that drive classification.
    """
    classifier, strata_order = _resolve_stratum_classifier(language)

    with LexiconDB(db_path) as db:
        proposals = classifier(db)

    if not proposals:
        click.echo(f"No etymons found for --language={language}.", err=True)
        return

    _summarize_stratum_proposals(language, proposals, strata_order)

    if not apply_changes:
        click.echo("", err=True)
        click.echo("(dry-run; pass --apply to write stratum)", err=True)
        return

    _write_stratum_proposals(db_path, proposals, force=force)


def _resolve_stratum_classifier(language: str) -> tuple[Any, tuple[str, ...]]:
    """Dispatch table for ``classify-stratum --language``. Each entry
    maps the CLI choice to a (classifier_callable, STRATA_tuple) pair.

    Local import keeps the cold-start path lean for unrelated CLI
    commands; same pattern as other lexicon subcommands.
    """
    from wyrd.generators.kenning.strata import (
        FRENCH_STRATA,
        OLD_ENGLISH_STRATA,
        OLD_NORSE_STRATA,
        WELSH_STRATA,
        classify_french,
        classify_old_english,
        classify_old_norse,
        classify_welsh,
    )

    classifiers = {
        "welsh": (classify_welsh, WELSH_STRATA),
        "french": (classify_french, FRENCH_STRATA),
        "old-english": (classify_old_english, OLD_ENGLISH_STRATA),
        "old-norse": (classify_old_norse, OLD_NORSE_STRATA),
    }
    return classifiers[language]


def _summarize_stratum_proposals(
    language: str,
    proposals: dict[int, str],
    strata_order: tuple[str, ...],
) -> None:
    """Emit the per-stratum count table to stderr. The summary lists
    every bucket in priority order (matching ``strata_order``) so an
    operator reading the dry-run output gets a stable shape across
    runs and across languages."""
    counts: dict[str, int] = dict.fromkeys(strata_order, 0)
    for stratum in proposals.values():
        counts[stratum] = counts.get(stratum, 0) + 1
    click.echo(f"{language} stratum classification: {len(proposals)} etymons", err=True)
    for stratum in strata_order:
        click.echo(f"  {stratum:25} {counts.get(stratum, 0)}", err=True)


# SQLite's 999-variable ceiling means a CASE-WHEN batch using 3 placeholders
# per row (id × 2 in the CASE clause + id in the IN clause) caps at ~333
# rows. Pick a round number under that to leave headroom for any future
# additional WHERE-clause params.
_STRATUM_WRITE_BATCH_SIZE = 300


def _write_stratum_proposals(
    db_path: Path,
    proposals: dict[int, str],
    *,
    force: bool,
) -> None:
    """Apply stratum proposals to the etymon table via CASE-batched
    UPDATEs (wyrd-b43r). Replaces the prior per-row loop, which scaled
    poorly at 100k+ rows once Phase 4 classifiers landed; the batch
    UPDATE chunks at SQLite's 999-variable ceiling (3 placeholders
    per row → ~333 rows max per batch).

    The WHERE clause carries the load-bearing ``AND stratum IS NULL``
    gate when ``--force`` is off — see the comment block below for
    the wyrd-lr4 Phase 4d idempotency contract.

    Emits a periodic progress line per CLAUDE.md's mining-progress
    convention so operators can extrapolate ETA on multi-minute runs;
    line shape: ``[N/total]  written=X skipped=Y (R s/entry)``.
    """
    items = list(proposals.items())
    total = len(items)
    written = 0
    skipped = 0
    started = time.monotonic()

    with LexiconDB(db_path) as db:
        for batch_start in range(0, total, _STRATUM_WRITE_BATCH_SIZE):
            chunk = items[batch_start : batch_start + _STRATUM_WRITE_BATCH_SIZE]
            cur = db.conn.execute(*_build_case_update(chunk, force=force))
            written += cur.rowcount
            skipped += len(chunk) - cur.rowcount
            processed = min(batch_start + _STRATUM_WRITE_BATCH_SIZE, total)
            elapsed = time.monotonic() - started
            rate = elapsed / processed if processed > 0 else 0
            click.echo(
                f"  [{processed}/{total}]  written={written} skipped={skipped} ({rate:.4f}s/entry)",
                err=True,
            )
        db.commit()

    click.echo("", err=True)
    if force:
        # Under --force the WHERE clause has no stratum-NULL gate, so
        # a zero-rowcount delta means the row vanished between the
        # classifier read and the write (deleted, or its id changed).
        # Don't claim a "skipped because pre-existing" reason that
        # didn't drive the skip.
        click.echo(
            f"Wrote {written} rows (forced overwrite). "
            f"{skipped} rows missed (row no longer present at write time).",
            err=True,
        )
    else:
        click.echo(
            f"Wrote {written} rows. "
            f"Skipped {skipped} that already had a stratum"
            f"{' (use --force to overwrite)' if skipped else ''}.",
            err=True,
        )


def _build_case_update(
    chunk: list[tuple[int, str]],
    *,
    force: bool,
) -> tuple[str, list[Any]]:
    """Build a CASE-batched UPDATE statement + parameter list for one
    chunk of stratum proposals.

    Shape: ``UPDATE etymon SET stratum = CASE id WHEN ? THEN ? ...
    END WHERE id IN (?, ?, ...)`` — plus the load-bearing ``AND
    stratum IS NULL`` clause when ``force=False`` (wyrd-lr4 Phase 4d
    idempotency contract: hand-corrections via ``set-stratum`` survive
    subsequent ``classify-stratum --apply`` runs because they have
    non-NULL stratum and this WHERE excludes them; removing the
    clause would silently blow away every operator hand-correction.
    Pinned end-to-end by
    ``test_cli_set_stratum_survives_classify_stratum_apply_without_force``).

    Returns ``(sql, params)`` ready for ``db.conn.execute(*pair)``.
    """
    # Defensive precondition. The caller (_write_stratum_proposals)
    # iterates batches via range(0, total, BATCH_SIZE) which never
    # produces an empty chunk, but an empty chunk would generate a
    # syntactically-invalid ``CASE id END`` (no WHEN clauses) — assert
    # so the failure mode is loud rather than a confusing SQL error.
    assert chunk, "_build_case_update called with empty chunk"
    case_clauses = " ".join("WHEN ? THEN ?" for _ in chunk)
    id_placeholders = ",".join("?" * len(chunk))
    params: list[Any] = []
    for eid, stratum in chunk:
        params.extend([eid, stratum])
    params.extend(eid for eid, _ in chunk)
    where_extra = "" if force else " AND stratum IS NULL"
    sql = (
        f"UPDATE etymon SET stratum = CASE id {case_clauses} END "  # noqa: S608 — placeholders
        f"WHERE id IN ({id_placeholders}){where_extra}"
    )
    return sql, params


@lexicon.command("set-stratum")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--etymon-id",
    type=int,
    default=None,
    help=("Direct etymon row identifier. Mutually exclusive with --canonical-form / --language."),
)
@click.option(
    "--canonical-form",
    type=str,
    default=None,
    help=(
        "Canonical form to look up. Must be paired with --language. "
        "The (canonical_form, language) pair is unique in the etymon "
        "table, so the lookup yields 0 or 1 rows; missing rows error, "
        "found rows update."
    ),
)
@click.option(
    "--language",
    type=str,
    default=None,
    help=(
        "Language code paired with --canonical-form for the lookup. "
        "The lexicon language code (e.g. 'welsh', 'old-french', "
        "'gmq-osw'), not the human-readable name."
    ),
)
@click.option(
    "--stratum",
    type=str,
    default=None,
    help=(
        "New stratum value. Must be one of the known strata for the "
        "target row's language family — e.g. a French row accepts "
        "values from FRENCH_STRATA only (wyrd-j3gy family-aware "
        "validation). Languages without a classifier yet fall back "
        "to the cross-family ALL_STRATA typo-check. Mutually exclusive "
        "with --clear."
    ),
)
@click.option(
    "--clear",
    is_flag=True,
    default=False,
    help=("Set the row's stratum to NULL. Mutually exclusive with --stratum."),
)
def lexicon_set_stratum(
    db_path: Path,
    etymon_id: int | None,
    canonical_form: str | None,
    language: str | None,
    stratum: str | None,
    clear: bool,
) -> None:
    """Manually set the ``stratum`` value on a single etymon row.

    Hand-correction CLI (wyrd-lr4) for high-stakes etymons that
    ``classify-stratum`` got wrong. The classifier's heuristic
    (ancestor walk + self-language pass) misses some signals — most
    commonly when the ``etymon_descent`` graph lacks a parent edge
    that scholarly sources actually attest. Use this command to
    write the correct stratum for an individual row.

    Hand-corrections survive subsequent ``classify-stratum --apply``
    runs (without ``--force``); pass ``classify-stratum --force`` to
    deliberately overwrite them, or ``set-stratum --clear`` to opt
    one row back into bulk classification.

    The stratum value is checked against the row's language family
    (wyrd-j3gy family-aware validation): a French-language row
    accepts FRENCH_STRATA values, an Old English row accepts
    OLD_ENGLISH_STRATA values, etc. Languages without a classifier
    yet fall back to a cross-family ALL_STRATA typo-check so the
    operator can still hand-correct on unclassified languages.

    Note: this REVERSES Phase 4d's intentional cross-family
    tolerance (writing a Welsh stratum onto a French row used to
    succeed silently on the operator's say-so). The runtime
    --stratum filter validates per-culture, so the SPA needed the
    write-side validator to match — making a Welsh stratum on a
    French row produce an unfilterable result was the worse failure
    mode. Pass through ``classify-stratum --apply`` if a true
    cross-family override is needed (it bypasses set-stratum).
    """
    _validate_set_stratum_identification(etymon_id, canonical_form, language)
    _validate_set_stratum_value(stratum, clear)

    with LexiconDB(db_path) as db:
        row = _resolve_set_stratum_target(db, etymon_id, canonical_form, language)
        _validate_set_stratum_family_match(row, stratum)
        old_stratum = row["stratum"]
        new_stratum = None if clear else stratum
        db.conn.execute(
            "UPDATE etymon SET stratum = ? WHERE id = ?",
            (new_stratum, row["id"]),
        )
        db.commit()

    click.echo(
        f"etymon {row['id']} ({row['canonical_form']}, {row['language']}): "
        f"{old_stratum!r} → {new_stratum!r}",
        err=True,
    )


def _validate_set_stratum_identification(
    etymon_id: int | None,
    canonical_form: str | None,
    language: str | None,
) -> None:
    """Validate the mutually-exclusive --etymon-id vs --canonical-form
    + --language identification flags. Exits non-zero on misuse so
    Click's command-level handler doesn't have to know which row-
    lookup mode is selected. Three failure cases:

      1. Both modes together — ambiguous; pick one.
      2. Neither mode — no target.
      3. Pair mode partially supplied — canonical-form alone is
         ambiguous across languages.
    """
    by_id = etymon_id is not None
    by_pair = canonical_form is not None or language is not None
    if by_id and by_pair:
        click.echo(
            "Error: pass --etymon-id OR --canonical-form/--language, not both.",
            err=True,
        )
        sys.exit(1)
    if not by_id and not by_pair:
        click.echo(
            "Error: must specify either --etymon-id or --canonical-form + --language.",
            err=True,
        )
        sys.exit(1)
    if by_pair and (canonical_form is None or language is None):
        click.echo(
            "Error: --canonical-form and --language must both be passed "
            "(canonical_form alone can be ambiguous across languages).",
            err=True,
        )
        sys.exit(1)


def _validate_set_stratum_family_match(row: Any, stratum: str | None) -> None:
    """wyrd-j3gy family-aware validation. Runs after the target row
    is resolved: looks up the row's language family and verifies the
    stratum is in that family's STRATA tuple.

    Languages without a classifier (no entry in ``LANGUAGE_TO_FAMILY``)
    skip this check — the typo-check against ``ALL_STRATA`` already
    ran in ``_validate_set_stratum_value``, and the operator may
    legitimately want to hand-correct on an unclassified language.

    --clear callers pass ``stratum=None`` and skip this check
    entirely (clearing has no family-validity question).
    """
    from wyrd.generators.kenning.strata import STRATA_BY_FAMILY, family_for_language

    if stratum is None:
        return  # --clear path — no family validation needed
    family = family_for_language(row["language"])
    if family is None:
        return  # unclassified language — fall back to ALL_STRATA typo-check
    valid = STRATA_BY_FAMILY[family]
    if stratum not in valid:
        click.echo(
            f"Error: stratum {stratum!r} is not valid for language "
            f"{row['language']!r} (family {family!r}). Valid: "
            f"{', '.join(valid)}.",
            err=True,
        )
        sys.exit(1)


def _validate_set_stratum_value(stratum: str | None, clear: bool) -> None:
    """Validate --stratum vs --clear. Mutually exclusive, one
    required, and the stratum value (when given) must be in the
    cross-family ALL_STRATA registry for typo protection. The
    failed-validation path exits non-zero with a friendly error."""
    from wyrd.generators.kenning.strata import ALL_STRATA

    if stratum is not None and clear:
        click.echo(
            "Error: --stratum and --clear are mutually exclusive.",
            err=True,
        )
        sys.exit(1)
    if stratum is None and not clear:
        click.echo(
            "Error: must specify --stratum VALUE or --clear.",
            err=True,
        )
        sys.exit(1)
    if stratum is not None and stratum not in ALL_STRATA:
        # Sorted list of known strata for the error message — operator
        # gets a friendly menu rather than a typo silently writing.
        known = ", ".join(sorted(ALL_STRATA))
        click.echo(
            f"Error: unknown stratum {stratum!r}. Known: {known}.",
            err=True,
        )
        sys.exit(1)


def _resolve_set_stratum_target(
    db: LexiconDB,
    etymon_id: int | None,
    canonical_form: str | None,
    language: str | None,
) -> Any:
    """Look up the target etymon row by id or by (canonical_form,
    language) pair. The pair lookup leans on the etymon table's
    UNIQUE (canonical_form, language) constraint to guarantee 0 or
    1 matches. Exits non-zero with a friendly error when the row
    doesn't exist; warns and proceeds when the row is an OCR-cluster
    loser (merged_into_id IS NOT NULL) since the classifier filters
    those out and a hand-correction on a merged row is invisible to
    the bundle export."""
    if etymon_id is not None:
        row = db.conn.execute(
            "SELECT id, canonical_form, language, stratum, merged_into_id FROM etymon WHERE id = ?",
            (etymon_id,),
        ).fetchone()
        if row is None:
            click.echo(f"Error: no etymon with id={etymon_id}.", err=True)
            sys.exit(1)
    else:
        row = db.conn.execute(
            "SELECT id, canonical_form, language, stratum, merged_into_id FROM etymon "
            "WHERE canonical_form = ? AND language = ?",
            (canonical_form, language),
        ).fetchone()
        if row is None:
            click.echo(
                f"Error: no etymon with canonical_form={canonical_form!r} "
                f"and language={language!r}.",
                err=True,
            )
            sys.exit(1)

    # Warn-and-proceed for merged-cluster losers. classify_family
    # skips these (they're folded into a canonical winner); a
    # hand-correction on one survives in the column but doesn't
    # surface in the bundle export. Operators occasionally do this
    # intentionally (planning to undo the merge later), so don't
    # block — just flag.
    if row["merged_into_id"] is not None:
        click.echo(
            f"Warning: etymon {row['id']} is a merged-cluster loser "
            f"(merged_into_id={row['merged_into_id']}); the classifier "
            f"and bundle export both filter these rows out, so the "
            f"hand-correction may not surface as expected.",
            err=True,
        )
    return row


@lexicon.command("decompose")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--meanings",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to meanings.json (defaults to bundled).",
)
@click.option(
    "--toponym-id",
    type=int,
    default=None,
    help=(
        "Run only this toponym (by id). Useful for spot-checking the "
        "picker on a single name. Without this, processes every "
        "toponym row up to --limit."
    ),
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Process at most N toponyms. Without this, processes all.",
)
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    default=False,
    help="Write the decomposition rows + canonical picks. Without this, dry-run.",
)
def lexicon_decompose(
    db_path: Path,
    meanings: Path | None,
    toponym_id: int | None,
    limit: int | None,
    apply_changes: bool,
) -> None:
    """Compute and persist matcher-derived decompositions per toponym
    (wyrd-08m Phase 1).

    For each toponym:
      1. Run ``Name.find_meaning(reduce=False)`` to enumerate every
         plausible breakdown (morphemes + unaccounted fragments).
      2. Persist each unique decomposition into
         ``toponym_decomposition`` (idempotent on signature).
      3. Apply the canonical picker (rules a + b — Phase 1).

    Rules (a) [scholar match] and (b) [unique-zero-unaccounted] are
    implemented. Multi-zero ties fall through with no canonical pick
    (rule (c) tiebreaker is a future enhancement). Consumer integration
    for ``rebuild-proportions`` and ``unaccounted`` lives in Phase 3a
    (wyrd-70s0); ``KenningExplain`` (Lambda / SPA path) has no DB access
    and remains heuristic-only until a future bundle-side projection
    lands.
    """
    meanings_data = _load_meanings_data(meanings)
    word_db, _ = load_meanings(meanings_data)

    from wyrd.generators.kenning.decomposition import populate_and_pick

    with LexiconDB(db_path) as db:
        if toponym_id is not None:
            rows = db.conn.execute(
                "SELECT id, modern_name FROM toponym WHERE id = ?",
                (toponym_id,),
            ).fetchall()
        else:
            sql = "SELECT id, modern_name FROM toponym ORDER BY id"
            params: tuple = ()
            if limit is not None:
                sql += " LIMIT ?"
                params = (limit,)
            rows = db.conn.execute(sql, params).fetchall()

        if not rows:
            click.echo("No toponym rows to process.", err=True)
            return

        rule_counts: dict[str, int] = {
            "scholar": 0,
            "scholar-disagreement": 0,
            "unique-zero-unaccounted": 0,
            "tiebreaker": 0,
            "no-canonical": 0,
        }
        decomposition_total = 0
        for idx, row in enumerate(rows, start=1):
            summary = populate_and_pick(db, row["id"], row["modern_name"], word_db)
            decomposition_total += int(summary["decomposition_count"] or 0)
            rule_key = summary["rule"] or "no-canonical"
            rule_counts[rule_key] = rule_counts.get(rule_key, 0) + 1
            if idx % 10 == 0 or idx == len(rows):
                click.echo(
                    f"  [{idx}/{len(rows)}] decompositions={decomposition_total} "
                    f"scholar={rule_counts['scholar']} "
                    f"scholar-disagreement={rule_counts['scholar-disagreement']} "
                    f"unique-zero={rule_counts['unique-zero-unaccounted']} "
                    f"tiebreaker={rule_counts['tiebreaker']} "
                    f"no-canonical={rule_counts['no-canonical']}",
                    err=True,
                )

        if not apply_changes:
            db.conn.rollback()
            click.echo("(dry-run; pass --apply to persist)", err=True)
            return
        db.commit()
        click.echo(
            f"applied={apply_changes} toponyms={len(rows)} "
            f"decompositions={decomposition_total} "
            f"scholar={rule_counts['scholar']} "
            f"scholar-disagreement={rule_counts['scholar-disagreement']} "
            f"unique-zero={rule_counts['unique-zero-unaccounted']} "
            f"tiebreaker={rule_counts['tiebreaker']} "
            f"no-canonical={rule_counts['no-canonical']}",
            err=True,
        )


@lexicon.command("report")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--top",
    type=int,
    default=15,
    show_default=True,
    help="Show top-N items in each ranking section.",
)
def lexicon_report(db_path: Path, top: int) -> None:
    """Read-only summary of the lexicon: who said what, where it agrees."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    def q(sql: str, *args) -> list:
        return conn.execute(sql, args).fetchall()

    def count(sql: str, *args) -> int:
        return conn.execute(sql, args).fetchone()[0]

    click.echo("=== sources ===")
    for r in q("SELECT id, year, region FROM source ORDER BY year, id"):
        year = r["year"] or "?"
        region = r["region"] or "?"
        click.echo(f"  {r['id']:50}  {year}  {region}")

    click.echo("")
    click.echo("=== totals ===")
    for tbl in (
        "etymon",
        "etymon_citation",
        "etymon_gloss",
        "etymon_tag",
        "reflex",
        "reflex_etymon",
        "toponym",
        "toponym_etymology",
        "toponym_etymology_element",
    ):
        click.echo(f"  {tbl:32} {count(f'SELECT COUNT(*) FROM {tbl}'):>8}")

    click.echo("")
    click.echo("=== consensus ===")
    consensus_dist = q(
        "SELECT witnesses, COUNT(*) AS n FROM etymon_consensus "
        "GROUP BY witnesses ORDER BY witnesses DESC"
    )
    for r in consensus_dist:
        click.echo(f"  etymons cited by {r['witnesses']:>2} sources: {r['n']:>5}")

    click.echo("")
    click.echo(f"=== top {top} etymons by witnesses ===")
    rows = q(
        "SELECT canonical_form, language, witnesses FROM etymon_consensus "
        "WHERE witnesses > 1 ORDER BY witnesses DESC, canonical_form LIMIT ?",
        top,
    )
    for r in rows:
        click.echo(f"  {r['canonical_form']:24} ({r['language']:14})  {r['witnesses']:>2} sources")

    click.echo("")
    click.echo("=== languages of mined etymons ===")
    rows = q("SELECT language, COUNT(*) AS n FROM etymon GROUP BY language ORDER BY n DESC")
    for r in rows:
        click.echo(f"  {r['language']:20} {r['n']:>6}")

    click.echo("")
    click.echo("=== confidence distribution (toponym_etymology) ===")
    rows = q(
        "SELECT confidence, COUNT(*) AS n FROM toponym_etymology "
        "GROUP BY confidence ORDER BY n DESC"
    )
    for r in rows:
        click.echo(f"  {r['confidence'] or '(null)':12} {r['n']:>6}")

    click.echo("")
    click.echo("=== toponyms per source ===")
    rows = q(
        "SELECT source_id, COUNT(*) AS n FROM toponym_etymology GROUP BY source_id ORDER BY n DESC"
    )
    for r in rows:
        click.echo(f"  {r['source_id']:50} {r['n']:>5}")

    click.echo("")
    click.echo("=== disagreements ===")
    # A toponym with >1 distinct breakdown signature has scholars disagreeing.
    disagreement_count = count("""
        SELECT COUNT(*) FROM (
            SELECT toponym_id
            FROM toponym_breakdown_signature
            GROUP BY toponym_id
            HAVING COUNT(DISTINCT signature) > 1
        )
        """)
    click.echo(f"  toponyms with conflicting breakdowns: {disagreement_count}")

    if disagreement_count > 0 and top > 0:
        click.echo(f"  examples (first {min(top, disagreement_count)}):")
        rows = q(
            """
            SELECT t.modern_name,
                   GROUP_CONCAT(DISTINCT tbs.signature) AS sigs,
                   GROUP_CONCAT(DISTINCT tbs.source_id) AS srcs
            FROM toponym_breakdown_signature tbs
            JOIN toponym t ON t.id = tbs.toponym_id
            GROUP BY tbs.toponym_id
            HAVING COUNT(DISTINCT tbs.signature) > 1
            LIMIT ?
            """,
            top,
        )
        for r in rows:
            click.echo(f"    {r['modern_name']:24} sources={r['srcs']}")

    conn.close()


@lexicon.command("stats")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
def lexicon_stats(db_path: Path) -> None:
    """Show row counts per table in the lexicon DB."""
    with LexiconDB(db_path) as db:
        stats = db.stats()
    for table, n in stats.items():
        click.echo(f"{table:<30} {n:>8}")


@lexicon.command("language-report")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--bundle",
    "bundle_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Bundled meanings.json (default: package-bundled "
        "wyrd/generators/kenning/data/meanings.json)."
    ),
)
@click.option(
    "--proportions",
    "proportions_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "english_proportions.json — used to derive the top-N reference "
        "tags (default: package-bundled english_proportions.json)."
    ),
)
@click.option(
    "--language",
    "language_filter",
    multiple=True,
    help=(
        "Restrict report to one or more languages (repeatable). "
        "Default: a curated list of place-name-relevant languages."
    ),
)
@click.option(
    "--top-tags",
    type=int,
    default=15,
    show_default=True,
    help="Number of top English tags to use as the reference profile.",
)
@click.option(
    "--keep-empty/--drop-empty",
    "keep_empty",
    default=False,
    show_default=True,
    help=(
        "Keep languages that have zero etymons in the DB. Default drops "
        "them — useful for trend-tracking snapshots when re-enabled."
    ),
)
@click.option(
    "--json-out",
    "json_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help=("Persist the JSON snapshot to this path. Markdown still goes to stdout."),
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["markdown", "json"]),
    default="markdown",
    show_default=True,
    help="Output format on stdout.",
)
@click.option(
    "--tier4/--no-tier4",
    "compute_tier4",
    default=True,
    show_default=True,
    help=(
        "Compute the wyrd-pmne Tier-4 phonology-rule reflex count "
        "(full-population walk over each chain-eligible language; ~3 "
        "minutes for modern-english's 1.4M etymons). --no-tier4 skips "
        "the walk and renders the cell as 'n/a' — fast for iteration."
    ),
)
@click.option(
    "--source",
    "source_filter",
    type=str,
    default=None,
    help=(
        "wyrd-5ecx: restrict the eligible-set seed to etymons cited "
        "by this specific source_id (e.g. 'rando-port', "
        "'mawer_1920_northumberland_durham'). Descent + lemma rollup "
        "still apply from the restricted seed, so the dashboard shows "
        "the slice + its era-progression context. Mutually exclusive "
        "with --tag."
    ),
)
@click.option(
    "--tag",
    "tag_filter",
    type=str,
    default=None,
    help=(
        "wyrd-5ecx: restrict the eligible-set seed to etymons tagged "
        "with this specific tag (e.g. 'fantasy', 'monster'). Descent "
        "+ lemma rollup still apply from the restricted seed. "
        "Mutually exclusive with --source."
    ),
)
def lexicon_language_report(
    db_path: Path,
    bundle_path: Path | None,
    proportions_path: Path | None,
    language_filter: tuple[str, ...],
    top_tags: int,
    keep_empty: bool,
    json_path: Path | None,
    output_format: str,
    compute_tier4: bool,
    source_filter: str | None,
    tag_filter: str | None,
) -> None:
    """Per-language quality scorecard (wyrd-wzwa Phase 1).

    Computes per-language metrics from the lexicon DB + bundled
    meanings.json: corpus depth, bundle representation, semantic-
    profile coverage vs the English reference, inflection / variant /
    era-reflex / stratum / citation coverage. Emits a markdown table
    by default; ``--format json`` swaps stdout to the schema-versioned
    JSON snapshot. ``--json-out PATH`` persists the JSON snapshot
    independently of the stdout format so trend tracking can keep both
    perspectives.
    """
    from wyrd.generators.kenning.language_quality import (
        DEFAULT_LANGUAGES,
        compute_report,
        load_reference_tags,
        report_to_json,
        report_to_markdown,
    )

    if bundle_path is None:
        bundle_path = Path(
            resources.files("wyrd.generators.kenning.data").joinpath("meanings.json")
        )
    if proportions_path is None:
        proportions_path = Path(
            resources.files("wyrd.generators.kenning.data").joinpath("english_proportions.json")
        )

    bundle = json.loads(bundle_path.read_text())
    reference_tags = load_reference_tags(proportions_path, top_n=top_tags)
    languages = list(language_filter) if language_filter else list(DEFAULT_LANGUAGES)

    # CLAUDE.md-shape stderr progress for the long modern-english
    # Tier-4 walk. The inner helper throttles to ~20 callbacks per
    # language; we emit them all (so 20 lines per language, finer
    # cadence than 10). Only fires when --tier4 is enabled.
    tier4_progress_state: dict[str, str] = {"last_lang": ""}

    def _tier4_progress(lang: str, done: int, total: int) -> None:
        if lang != tier4_progress_state["last_lang"]:
            click.echo(f"  tier4-phonology {lang}: walking {total} etymons…", err=True)
            tier4_progress_state["last_lang"] = lang
        pct = (done / total * 100.0) if total else 0.0
        click.echo(
            f"  tier4-phonology {lang}: {done}/{total} ({pct:.1f}%)",
            err=True,
        )

    if source_filter is not None and tag_filter is not None:
        raise click.UsageError("--source and --tag are mutually exclusive")

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        report = compute_report(
            conn,
            bundle,
            languages=languages,
            reference_tags=reference_tags,
            drop_empty=not keep_empty,
            compute_tier4=compute_tier4,
            tier4_progress_callback=_tier4_progress if compute_tier4 else None,
            source_filter=source_filter,
            tag_filter=tag_filter,
        )
    finally:
        conn.close()

    if json_path is not None:
        json_path.write_text(report_to_json(report))

    if output_format == "json":
        click.echo(report_to_json(report))
    else:
        click.echo(report_to_markdown(report))


@lexicon.group("synsets")
def lexicon_synsets() -> None:
    """Meaning-synset commands (wyrd-7tz Phase 1).

    Manage the meaning_synset / etymon_meaning_synset tables — fine-
    grained semantic equivalence classes used by upcoming generator
    transforms (calque, anglicize, drift-toward-X). Distinct from the
    cognate-cluster etymon.cognate_id populated by cluster-cognates.
    """


@lexicon_synsets.command("seed")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
def lexicon_synsets_seed(db_path: Path) -> None:
    """Idempotently populate meaning_synset from the bundled catalog
    (data/meaning_synsets.json). Reports inserts / updates / unchanged."""
    with LexiconDB(db_path) as db:
        result = seed_meaning_synsets(db)
    click.echo(
        f"meaning_synset seed: inserted={result['inserted']} "
        f"updated={result['updated']} unchanged={result['unchanged']}"
    )


@lexicon_synsets.command("list")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--with-counts",
    "with_counts",
    is_flag=True,
    default=False,
    help="Include member-count column (etymons assigned per synset).",
)
def lexicon_synsets_list(db_path: Path, with_counts: bool) -> None:
    """List all meaning_synset rows, sorted by canonical_label."""
    with LexiconDB(db_path) as db:
        synsets = list_meaning_synsets(db, with_member_counts=with_counts)
    if not synsets:
        click.echo("(no meaning_synset rows — run `wyrd kenning lexicon synsets seed`)")
        return
    for s in synsets:
        if with_counts:
            click.echo(f"  {s['canonical_label']:<32} ({s['member_count']:>4} members)")
        else:
            click.echo(f"  {s['canonical_label']}")


@lexicon_synsets.command("assign")
@click.argument("etymon_ident", type=str)
@click.argument("synset_label", type=str)
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--language",
    type=str,
    default=None,
    help=(
        "When set, ETYMON_IDENT is treated as a canonical_form looked up "
        "via (form, language) — more ergonomic for human curators than "
        "looking up the integer id first. Without --language, "
        "ETYMON_IDENT must be a numeric etymon.id."
    ),
)
@click.option(
    "--fit",
    type=click.Choice(["core", "peripheral"]),
    default="core",
    show_default=True,
    help="'core' = primary sense; 'peripheral' = secondary sense.",
)
def lexicon_synsets_assign(
    etymon_ident: str,
    synset_label: str,
    db_path: Path,
    language: str | None,
    fit: str,
) -> None:
    """Add an etymon as a member of a meaning_synset. Idempotent: re-running
    with the same args is a no-op; re-running with a different --fit
    updates the existing row.

    ETYMON_IDENT is either a numeric etymon.id (default) or a
    canonical_form when --language is given (e.g. ``assign wæter
    water/flowing --language old-english``).
    """
    with LexiconDB(db_path) as db:
        try:
            etymon_id = _resolve_etymon_ident(db, etymon_ident, language)
            inserted = assign_etymon_to_meaning_synset(db, etymon_id, synset_label, fit=fit)
        except ValueError as exc:
            click.echo(f"Error: {exc}", err=True)
            sys.exit(1)
    click.echo(f"{'Added' if inserted else 'Updated'} etymon {etymon_id} → {synset_label} ({fit})")


def _resolve_etymon_ident(db: LexiconDB, ident: str, language: str | None) -> int:
    """Resolve an etymon-identifying CLI arg to an etymon.id.

    Without ``--language``, ident must be a numeric id. With
    ``--language``, ident is treated as a canonical_form and looked up
    via (form, language). Raises ValueError with a friendly message on
    every failure mode (non-numeric without --language; missing form;
    ambiguous form). The caller surfaces the message on stderr +
    exits non-zero, matching the rest of this CLI's error pattern."""
    if language is None:
        try:
            return int(ident)
        except ValueError:
            raise ValueError(
                f"etymon ident {ident!r} is not numeric; pass --language to "
                f"look up by canonical_form, or use an integer etymon.id"
            ) from None
    row = db.conn.execute(
        "SELECT id FROM etymon WHERE canonical_form = ? AND language = ?",
        (ident, language),
    ).fetchone()
    if row is None:
        raise ValueError(f"no etymon with canonical_form={ident!r} language={language!r}")
    return row["id"]


@lexicon_synsets.command("show")
@click.argument("etymon_id", type=int)
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
def lexicon_synsets_show(etymon_id: int, db_path: Path) -> None:
    """Show the meaning_synsets an etymon belongs to."""
    with LexiconDB(db_path) as db:
        synsets = get_meaning_synsets_for_etymon(db, etymon_id)
    if not synsets:
        click.echo(f"(etymon {etymon_id} has no meaning_synset memberships)")
        return
    for s in synsets:
        click.echo(f"  {s['canonical_label']:<32} ({s['fit']})")


@lexicon_synsets.command("candidates")
@click.argument("etymon_id", type=int)
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--target-language",
    type=str,
    default=None,
    help="Restrict candidates to one language (e.g. 'old-english').",
)
@click.option(
    "--fit",
    type=click.Choice(["core", "peripheral"]),
    default=None,
    help="Restrict to high-confidence ('core') or peripheral memberships only.",
)
def lexicon_synsets_candidates(
    etymon_id: int,
    db_path: Path,
    target_language: str | None,
    fit: str | None,
) -> None:
    """Show meaning-preserving substitution candidates for an etymon —
    other etymons that share at least one meaning_synset.

    Used by upcoming Lab transforms (replace-root, calque, anglicize)
    to look up same-meaning morphemes across languages and registers.
    """
    with LexiconDB(db_path) as db:
        rows = get_meaning_preserving_candidates(
            db, etymon_id, target_language=target_language, fit=fit
        )
    if not rows:
        click.echo(f"(no meaning-preserving candidates for etymon {etymon_id})")
        return
    for r in rows:
        click.echo(
            f"  {r['canonical_form']:<24} {r['language']:<14} "
            f"{r['synset_label']:<28} (fit: {r['candidate_fit']})"
        )


@cli.command("validate-meanings")
@click.argument(
    "meanings",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=False,
)
def validate_meanings(meanings: Path | None) -> None:
    """Check meanings.json: every modern_usage must contain a hyphen.

    Replaces Rando's `bin/validate_names`. Reads the path argument or stdin.
    Exits non-zero if any entries are malformed.
    """
    raw = meanings.read_text() if meanings else sys.stdin.read()
    data = json.loads(raw)

    bad = []
    for subject in data:
        for word in subject["words"]:
            if "-" not in word["modern_usage"]:
                bad.append(word)

    for word in bad:
        click.echo(json.dumps(word), err=True)
    if bad:
        click.echo(f"{len(bad)} malformed entries.", err=True)
        sys.exit(1)
    click.echo(f"OK: {sum(len(s['words']) for s in data)} entries valid.", err=True)
