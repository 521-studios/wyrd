"""CLI commands for Kenning. Mounted under `wyrd kenning <subcommand>`."""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
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
    anthropic_extractor,
    available_tags,
    gemini_extractor,
    llm_extractor,
)
from wyrd.generators.kenning.dictionary_parser import parse_alphabetical_text
from wyrd.generators.kenning.era import era_cell, language_family
from wyrd.generators.kenning.lexicon import (
    LANGUAGE_FIELDS,
    RECOMMENDED_LANG_THRESHOLDS,
    LexiconDB,
    assign_etymon_to_meaning_synset,
    backfill_citation_pages,
    bridge_celtic_forms,
    bridge_generic_language,
    bridge_phonological_oe,
    bridge_phonological_on,
    clear_enrichment,
    cluster_cognates,
    cluster_ocr_variants,
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
    record_mining_run,
    reverse_search_attestations,
    seed_from_meanings,
    seed_meaning_synsets,
)
from wyrd.generators.kenning.meaning import Meaning, load_meanings
from wyrd.generators.kenning.name import load_names
from wyrd.generators.kenning.paths import (
    LEXICON_DB_DEFAULT_DISPLAY,
    default_lexicon_path,
)
from wyrd.generators.kenning.skeat_parser import parse_skeat_text
from wyrd.generators.kenning.wiktextract_ingester import ingest_wiktextract_path
from wyrd.seed import resolve_seed, rng_for


def _select_parser_and_run(text: str, parser: str) -> list:
    """Pick the right parser for a book.

    Modes:
      skeat        - force Skeat-format parser
      alphabetical - force alphabetical-dictionary parser
      auto         - try Skeat first; if it returns ≥5 entries, use it.
                     Otherwise use alphabetical.
    """
    if parser == "skeat":
        return parse_skeat_text(text)
    if parser == "alphabetical":
        return parse_alphabetical_text(text)
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


@cli.command("rebuild-proportions")
@click.argument("culture", type=click.Choice(CULTURES))
@click.argument("place_names", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--meanings",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to meanings.json (defaults to bundled).",
)
def rebuild_proportions(culture: str, place_names: Path, meanings: Path | None) -> None:
    """Recompute a culture's proportions from a corpus + meanings DB.

    Replaces Rando's `bin/load_names`. Reads place_names JSON (the same shape as
    the bundled `<culture>_place_names.json` files), deconstructs each name
    against the meaning DB, and emits a fresh proportions JSON to stdout.
    """
    meanings_data = _load_meanings_data(meanings)

    names_data = json.loads(place_names.read_text())
    names = load_names(names_data)
    word_db, _ = load_meanings(meanings_data)

    perfect = 0
    word_names = 0
    word_saints = 0
    good_names = []
    for name in names:
        name.find_meaning(word_db)
        if name.has_name():
            word_names += 1
        if name.has_saint():
            word_saints += 1
        if name.count_unaccounted() == 0:
            perfect += 1
            good_names.append(name)

    click.echo(
        f"culture={culture} perfect={perfect} names={word_names} saints={word_saints} "
        f"total={len(names)}",
        err=True,
    )
    proportions = _proportions_from(good_names)
    click.echo(json.dumps(proportions))


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
    structs = []
    for key, value in struct.items():
        words = [[_encode_meaning(meaning) for meaning in word] for word in key]
        structs.append({"proportion": value, "words": words})
    return structs


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
def unaccounted(
    culture: str,
    place_names: Path,
    meanings: Path | None,
    top: int,
    examples: int,
    min_length: int,
    as_json: bool,
) -> None:
    """Surface morpheme fragments the corpus uses but the meaning DB doesn't.

    Decomposes every name in <place_names>; for names that don't fully decompose,
    aggregates the leftover string fragments by frequency. The top fragments are
    candidates for new entries in meanings.json — this is the mining tool that
    surfaces what the previous deconstruction silently dropped.
    """
    meanings_data = _load_meanings_data(meanings)

    names_data = json.loads(place_names.read_text())
    names = load_names(names_data)
    word_db, _ = load_meanings(meanings_data)

    fragments: Counter = Counter()
    examples_by_frag: dict[str, list[str]] = {}
    imperfect_count = 0
    for name in names:
        name.find_meaning(word_db)
        if name.count_unaccounted() == 0:
            continue
        imperfect_count += 1
        seen_in_name: set[str] = set()
        for word_list in name.words.values():
            for word in word_list:
                for chunk in word.word:
                    if not isinstance(chunk, str):
                        continue
                    frag = chunk.lower()
                    if len(frag) < min_length:
                        continue
                    if frag in seen_in_name:
                        continue
                    seen_in_name.add(frag)
                    fragments[frag] += 1
                    bucket = examples_by_frag.setdefault(frag, [])
                    if name.name not in bucket and len(bucket) < examples:
                        bucket.append(name.name)

    if as_json:
        out = [
            {"fragment": frag, "count": count, "examples": examples_by_frag[frag]}
            for frag, count in fragments.most_common(top)
        ]
        click.echo(json.dumps(out, indent=2))
        return

    click.echo(
        f"culture={culture} imperfect_names={imperfect_count} unique_fragments={len(fragments)}",
        err=True,
    )
    click.echo(f"{'fragment':<20} {'count':>6}  examples")
    click.echo("-" * 70)
    for frag, count in fragments.most_common(top):
        ex = ", ".join(examples_by_frag[frag])
        click.echo(f"{frag:<20} {count:>6}  {ex}")


@cli.command("explain")
@click.argument("name")
def explain(name: str) -> None:
    """Decompose a name into morphemes; print every matching reading."""
    explainer = KenningExplain()
    results = explainer.generate_all({"name": name}, 0)
    for r in results:
        click.echo(r.explanation)


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
def lexicon_export_meanings(
    db_path: Path,
    output_path: Path | None,
    min_witnesses: int,
    lang_threshold_specs: tuple[str, ...],
    use_preset: bool,
    include_rando: bool,
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
        )
    payload = json.dumps(subjects, ensure_ascii=False, indent=2)
    if output_path is None:
        click.echo(payload)
    else:
        output_path.write_text(payload + "\n")
        click.echo(f"Wrote {len(subjects)} subjects to {output_path}", err=True)


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
    type=click.Choice(["auto", "skeat", "alphabetical"]),
    default="auto",
    show_default=True,
    help="Which parser to use; 'auto' picks based on first-pass yield.",
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
) -> None:
    """Mine an etymology text via LLM (Ollama or Gemini).

    Uses the regex parser to segment entries, then feeds each entry body to
    the chosen LLM with a strict JSON schema. Every extracted form is
    validated against the source text — hallucinated etymons are rejected,
    not ingested. Successful extractions are tagged in `notes` with the
    provider+model so we can supersede them later with a stronger model.
    """
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
    parsed = _select_parser_and_run(text, parser)
    if limit is not None:
        parsed = parsed[:limit]

    if provider == "ollama":
        client_kwargs: dict[str, Any] = {"timeout_s": timeout}
        if ollama_url:
            client_kwargs["base_url"] = ollama_url
        if model:
            client_kwargs["model"] = model
        client = llm_extractor.OllamaClient(**client_kwargs)
        extract_one = llm_extractor.extract_one
    elif provider == "gemini":
        gemini_kwargs: dict[str, Any] = {"timeout_s": timeout}
        if model:
            gemini_kwargs["model"] = model
        client = gemini_extractor.GeminiClient(**gemini_kwargs)
        extract_one = gemini_extractor.extract_one
    elif provider == "anthropic":
        anthropic_kwargs: dict[str, Any] = {"timeout_s": timeout}
        if model:
            anthropic_kwargs["model"] = model
        client = anthropic_extractor.AnthropicClient(**anthropic_kwargs)
        extract_one = anthropic_extractor.extract_one
    else:
        raise click.ClickException(f"unknown provider: {provider}")

    click.echo(
        f"Mining {len(parsed)} entries from {path.name} via {client.base_url} model={client.model}",
        err=True,
    )

    accepted_entries = []
    declined = 0
    rejected = 0
    by_failure: dict[str, int] = {}

    started_at = datetime.now(UTC).isoformat()
    start = time.time()
    for i, p in enumerate(parsed, 1):
        result = extract_one(
            client, toponym=p.toponym, body=p.body_text, suffix_hint=p.section_suffix
        )
        if result.accepted and result.entry and result.entry.elements:
            accepted_entries.append(result.entry)
        elif result.accepted:
            declined += 1
        else:
            rejected += 1
            for f in result.failures:
                by_failure[f.reason] = by_failure.get(f.reason, 0) + 1
        if i % 10 == 0:
            elapsed = time.time() - start
            click.echo(
                f"  [{i}/{len(parsed)}]  "
                f"accepted={len(accepted_entries)} declined={declined} rejected={rejected} "
                f"({elapsed / i:.1f}s/entry)",
                err=True,
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
    if provider == "gemini":
        Client = gemini_extractor.GeminiClient
        provider_extract_one = gemini_extractor.extract_one
        provider_tag = "extracted_by:gemini:"
    elif provider == "anthropic":
        Client = anthropic_extractor.AnthropicClient
        provider_extract_one = anthropic_extractor.extract_one
        provider_tag = "extracted_by:anthropic:"
    else:
        raise click.ClickException(f"unknown provider: {provider}")

    candidates = _select_review_candidates(
        db_path=db_path,
        provider_tag=provider_tag,
        confidence=confidence,
        book=book,
        limit=limit,
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

    client = Client(timeout_s=timeout, **({"model": model} if model else {}))

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


def _select_review_candidates(
    *,
    db_path: Path,
    provider_tag: str,
    confidence: tuple[str, ...],
    book: str | None,
    limit: int | None,
) -> list[sqlite3.Row]:
    """Pick `toponym_etymology` rows eligible for Tier-2 re-extraction.

    Eligibility: row was extracted by an LLM provider (notes contain
    `extracted_by:llm:` or a tier-1 marker like `qwen` / `ollama`),
    confidence is in the `confidence` set, and the same (toponym, source)
    doesn't already have a row tagged with the chosen Tier-2 provider.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        placeholders = ",".join("?" * len(confidence))
        sql = f"""
            SELECT te.id, te.toponym_id, te.source_id, te.confidence,
                   t.modern_name, t.region
            FROM toponym_etymology te
            JOIN toponym t ON t.id = te.toponym_id
            WHERE te.confidence IN ({placeholders})
              AND te.source_id IN (SELECT id FROM source)
              AND (te.notes LIKE '%extracted_by:llm:%'
                   OR te.notes LIKE '%qwen%'
                   OR te.notes LIKE '%llama%'
                   OR te.notes LIKE '%ollama%')
              AND NOT EXISTS (
                  SELECT 1 FROM toponym_etymology te2
                  WHERE te2.toponym_id = te.toponym_id
                    AND te2.source_id  = te.source_id
                    AND te2.notes LIKE ?
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
def lexicon_disambiguate_fuzzy(
    db_path: Path,
    apply_changes: bool,
    limit: int | None,
    model: str | None,
    timeout: float,
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
    """
    # Local imports keep the disambiguator deps off the cold-start path
    # for unrelated CLI commands.
    from wyrd.generators.kenning.disambiguator import (
        apply_disambiguator_result,
        disambiguate_one,
        find_ambiguous_rows,
    )
    from wyrd.generators.kenning.gemini_extractor import GeminiClient

    with LexiconDB(db_path) as db:
        cases = find_ambiguous_rows(db, max_distance=1, limit=limit)

    if not cases:
        click.echo("No ambiguous fuzzy rows found.", err=True)
        return

    click.echo(
        f"{len(cases)} ambiguous rows found. {'Applying' if apply_changes else 'Dry-run'}.",
        err=True,
    )

    client = GeminiClient(timeout_s=timeout, **({"model": model} if model else {}))

    counts = {"kept": 0, "reassigned": 0, "deleted": 0, "errors": 0}
    write_db = LexiconDB(db_path) if apply_changes else None
    try:
        for case in cases:
            try:
                result = disambiguate_one(client, case)
            except RuntimeError as e:
                counts["errors"] += 1
                click.echo(f"  ERROR  {case.matched_form:20} {e}", err=True)
                continue
            choice_str = (
                f"id={result.chosen_etymon_id}" if result.chosen_etymon_id is not None else "none"
            )
            click.echo(
                f"  {case.matched_form:20} → {choice_str} (conf={result.confidence}) "
                f"— {result.reason[:80]}",
                err=True,
            )
            if apply_changes and write_db is not None:
                action = apply_disambiguator_result(write_db, case, result)
                # Commit per-row so a crash mid-run doesn't lose previously
                # disambiguated rows (the LLM calls have already been paid for).
                write_db.commit()
            else:
                # Dry-run: mirror the same action-classification so the
                # summary tells the operator what would have happened.
                if result.chosen_etymon_id is None:
                    action = "deleted"
                elif result.chosen_etymon_id == case.current_etymon_id:
                    action = "kept"
                else:
                    action = "reassigned"
            counts[action] += 1
    finally:
        if write_db is not None:
            write_db.close()

    click.echo("", err=True)
    click.echo(f"Disambiguator summary: {counts}", err=True)
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
    type=click.Choice(["ocr", "lemmas", "text-match", "cognates", "attested-years", "all-derived"]),
    required=True,
    help=(
        "Which enrichment stage to clear. 'ocr' un-marks merged_into_id, "
        "'lemmas' resets lemma_id/inflection/lemma_method, 'text-match' "
        "drops the etymon_text_match table, 'cognates' resets "
        "synset_id/synset_method (D27/wyrd-81n), 'attested-years' resets "
        "etymon_text_match.attested_year (D5-1/wyrd-3ux), 'all-derived' "
        "does all five."
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
    if result["synset_assignments_to_clear"]:
        click.echo(
            f"  {verb} {result['synset_assignments_to_clear']} synset_id assignments", err=True
        )
    if result.get("attested_years_to_clear"):
        click.echo(f"  {verb} {result['attested_years_to_clear']} attested_year values", err=True)
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
    help="Actually write synset_id assignments. Without this flag the command runs as a dry-run.",
)
def lexicon_cluster_cognates(db_path: Path, apply_changes: bool) -> None:
    """Walk the etymon_descent graph from each root and assign synset_id
    to every reachable descendant (D27 / wyrd-81n).

    Roots are etymons that participate in inheritance/borrowing edges as
    parent but never as child — typically Proto-* forms. Every descendant
    reachable via inheritance + borrowing edges gets synset_id =
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
        f"{verb} synset_id on {result['candidates']} etymon(s)",
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
    synset_id assignments.

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
    Re-run cluster-cognates afterward to refresh synset_id assignments
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
    to refresh synset_id assignments via the merged_into_id rollup.

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
    to refresh synset_id assignments via the merged_into_id rollup.

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
    populate synset_id from the new descent edges.

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
    if not apply_changes:
        click.echo("(dry-run; pass --apply to commit)", err=True)


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
    disagreement_count = count(
        """
        SELECT COUNT(*) FROM (
            SELECT toponym_id
            FROM toponym_breakdown_signature
            GROUP BY toponym_id
            HAVING COUNT(DISTINCT signature) > 1
        )
        """
    )
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


@lexicon.group("synsets")
def lexicon_synsets() -> None:
    """Meaning-synset commands (wyrd-7tz Phase 1).

    Manage the meaning_synset / etymon_meaning_synset tables — fine-
    grained semantic equivalence classes used by upcoming generator
    transforms (calque, anglicize, drift-toward-X). Distinct from the
    cognate-cluster etymon.synset_id populated by cluster-cognates.
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
