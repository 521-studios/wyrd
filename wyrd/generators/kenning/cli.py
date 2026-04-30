"""CLI commands for Kenning. Mounted under `wyrd kenning <subcommand>`."""

from __future__ import annotations

import json
import sys
from collections import Counter
from importlib import resources
from pathlib import Path
from typing import Any

import click

from wyrd.generators.kenning import (
    CULTURES,
    Kenning,
    available_tags,
)
from wyrd.generators.kenning.dictionary_parser import parse_alphabetical_text
from wyrd.generators.kenning.lexicon import (
    LexiconDB,
    cluster_ocr_variants,
    fuzzy_search_attestations,
    ingest_parsed_entries,
    init_schema,
    link_lemmas,
    migrate_schema,
    reverse_search_attestations,
    seed_from_meanings,
)
from wyrd.generators.kenning.meaning import load_meanings
from wyrd.generators.kenning.name import load_names
from wyrd.generators.kenning.skeat_parser import parse_skeat_text
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
        text = (
            resources.files("wyrd.generators.kenning.data").joinpath("meanings.json").read_text()
        )
    else:
        text = meanings.read_text()
    return json.loads(text)


# Default location for the authoring lexicon DB. Build artifact, gitignored.
_DEFAULT_LEXICON_PATH = Path("wyrd/generators/kenning/data/lexicon.db")

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
def generate(
    culture: str, tags: tuple[str, ...], count: int, seed: int | None, describe: bool
) -> None:
    """Generate town names. Replaces Rando's `bin/generator`."""
    known_tags = set(available_tags()) | {"male name", "female name", "saint"}
    bad = [t for t in tags if t not in known_tags]
    if bad:
        click.echo(f"Unknown tag(s): {', '.join(bad)}", err=True)
        click.echo("Available tags:", err=True)
        for t in sorted(known_tags - {"male name", "female name", "saint"}):
            click.echo(f"  {t}", err=True)
        sys.exit(1)

    resolved = resolve_seed(seed)
    seed_rng = rng_for(resolved)
    kenning = Kenning()
    params = {"culture": culture, "tags": list(tags)}
    for _ in range(count):
        result = kenning.generate(params, seed_rng.randrange(2**63))
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
    """
    from wyrd.generators.kenning.meaning import Meaning

    flat_meanings: list[Meaning] = []
    for word_text in name.name.split(" "):
        word_options = name.words.get(word_text, [])
        if not word_options:
            continue
        # Pick the FIRST (best, post-reduction) decomposition for the word.
        chosen = word_options[0]
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
    # Local import keeps the cli module light and avoids a circular import,
    # since wyrd.generators.kenning registers generators at import time.
    from wyrd.generators.kenning import KenningExplain

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
    show_default=True,
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
}


@lexicon.command("mine-skeat")
@click.argument("path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=True,
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
    show_default=True,
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
        from wyrd.generators.kenning.llm_extractor import OllamaClient, extract_one

        client_kwargs: dict[str, Any] = {"timeout_s": timeout}
        if ollama_url:
            client_kwargs["base_url"] = ollama_url
        if model:
            client_kwargs["model"] = model
        client = OllamaClient(**client_kwargs)
    elif provider == "gemini":
        from wyrd.generators.kenning.gemini_extractor import GeminiClient
        from wyrd.generators.kenning.gemini_extractor import extract_one as gemini_extract_one

        gemini_kwargs: dict[str, Any] = {"timeout_s": timeout}
        if model:
            gemini_kwargs["model"] = model
        client = GeminiClient(**gemini_kwargs)
        extract_one = gemini_extract_one
    elif provider == "anthropic":
        from wyrd.generators.kenning.anthropic_extractor import AnthropicClient
        from wyrd.generators.kenning.anthropic_extractor import extract_one as anthropic_extract_one

        anthropic_kwargs: dict[str, Any] = {"timeout_s": timeout}
        if model:
            anthropic_kwargs["model"] = model
        client = AnthropicClient(**anthropic_kwargs)
        extract_one = anthropic_extract_one
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

    import time

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

    click.echo(f"Ingested: {counts}", err=True)


@lexicon.command("review")
@click.argument("source_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=True,
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
    import sqlite3
    import time

    if provider == "gemini":
        from wyrd.generators.kenning.gemini_extractor import GeminiClient as Client
        from wyrd.generators.kenning.gemini_extractor import (
            extract_one as provider_extract_one,
        )

        provider_tag = "extracted_by:gemini:"
    elif provider == "anthropic":
        from wyrd.generators.kenning.anthropic_extractor import AnthropicClient as Client
        from wyrd.generators.kenning.anthropic_extractor import (
            extract_one as provider_extract_one,
        )

        provider_tag = "extracted_by:anthropic:"
    else:
        raise click.ClickException(f"unknown provider: {provider}")

    # Read-only candidate query first; only open the writable connection if
    # we'll actually write.
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    # Candidates: rows extracted by an LLM provider tagged "llm:" (currently
    # only Ollama writes that prefix; Gemini writes "gemini:") whose
    # confidence is below the bar. Skip toponyms that already have a row
    # whose source_quote indicates a Gemini origin.
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
    # Build the LIKE pattern for the chosen provider's tag
    provider_skip_pattern = f"%{provider_tag}%"
    args: list[object] = list(confidence) + [provider_skip_pattern]
    if book:
        sql += " AND te.source_id = ?"
        args.append(book)
    sql += " ORDER BY te.source_id, t.modern_name"
    if limit:
        sql += " LIMIT ?"
        args.append(limit)

    candidates = conn.execute(sql, args).fetchall()
    conn.close()

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

            text = source_path.read_text()
            parsed = _select_parser_and_run(text, "auto")
            entries_by_name: dict[str, object] = {p.toponym: p for p in parsed}

            click.echo(f"=== {source_id} ({len(rows)} candidates) ===", err=True)

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
                dt = time.time() - t0

                counts["reviewed"] += 1
                if not result.accepted:
                    counts["rejected"] += 1
                    failures = ", ".join(f.reason for f in result.failures)
                    click.echo(f"  REJECT  {name:24} ({dt:.1f}s)  {failures}", err=True)
                    continue

                if result.entry is None or not result.entry.elements:
                    counts["declined"] += 1
                    click.echo(
                        f"  DECLINE {name:24} ({dt:.1f}s)  conf={result.entry.confidence if result.entry else '?'}",
                        err=True,
                    )
                    continue

                breakdown = "+".join(el.form for el in result.entry.elements)
                click.echo(
                    f"  REVIEW  {name:24} ({dt:.1f}s)  {breakdown:30}  "
                    f"conf={result.entry.confidence}",
                    err=True,
                )

                if apply_changes and write_db is not None:
                    ingest_parsed_entries(write_db, [result.entry], source_id, region=row["region"])
                    counts["written"] += 1
    finally:
        if write_db is not None:
            write_db.close()

    click.echo("", err=True)
    click.echo(f"Review summary: {counts}", err=True)
    if not apply_changes:
        click.echo("(dry-run; pass --apply to write)", err=True)


@lexicon.command("migrate")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=True,
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


@lexicon.command("link-lemmas")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=True,
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
    show_default=True,
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
    show_default=True,
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


@lexicon.command("normalize-ocr")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=True,
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


@lexicon.command("report")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=True,
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
    import sqlite3

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
    show_default=True,
)
def lexicon_stats(db_path: Path) -> None:
    """Show row counts per table in the lexicon DB."""
    with LexiconDB(db_path) as db:
        stats = db.stats()
    for table, n in stats.items():
        click.echo(f"{table:<30} {n:>8}")


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
