"""CLI commands for Kenning. Mounted under `wyrd kenning <subcommand>`.

This module is the back-compat shim for the wyrd-g143 split. The
top-level ``@cli`` group lives here; top-level subcommands and the
``@lexicon`` sub-group live in their own modules under
``wyrd/generators/kenning/cli/``. The remaining
``@lexicon.command`` bodies in this file are pending extraction
(slices 2-5 of wyrd-g143) — each will eventually move to its own
module under ``cli/lexicon/<command>.py``.

Tests that need package-internal helpers (currently
``_select_parser_and_run``) import them from this module's
back-compat re-exports below.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from importlib import resources
from pathlib import Path
from typing import Any

import click

from wyrd.generators.kenning import (
    CULTURES,
    anthropic_extractor,
    etymonline_ingester,
    fantasy_pipeline,
    gemini_extractor,
    llm_extractor,
    pfsrd2_monster_extractor,
)
from wyrd.generators.kenning.cli import (
    add_meaning as _add_meaning_module,
)
from wyrd.generators.kenning.cli import (
    creature as _creature_module,
)
from wyrd.generators.kenning.cli import (
    era_map as _era_map_module,
)
from wyrd.generators.kenning.cli import (
    explain as _explain_module,
)
from wyrd.generators.kenning.cli import (
    generate as _generate_module,
)
from wyrd.generators.kenning.cli import (
    rebuild_proportions as _rebuild_proportions_module,
)
from wyrd.generators.kenning.cli import (
    rewind as _rewind_module,
)
from wyrd.generators.kenning.cli import (
    unaccounted as _unaccounted_module,
)
from wyrd.generators.kenning.cli import (
    validate_meanings as _validate_meanings_module,
)
from wyrd.generators.kenning.cli.lexicon import (
    add_to as _register_lexicon_group,
)
from wyrd.generators.kenning.cli.lexicon import (
    lexicon,
)

# Back-compat re-exports of private helpers tests grab directly.
# Keep this list narrow — anything here is part of the implicit test
# surface and re-locating these helpers means coordinating with the
# tests that reach in. ``tests/test_kenning_cooccurrence.py`` drives
# the rebuild-proportions tag-cooccurrence math via these two helpers
# without spinning up the full click invocation.
from wyrd.generators.kenning.cli.lexicon.classify_stratum import (  # noqa: F401
    _build_case_update,
)
from wyrd.generators.kenning.cli.rebuild_proportions import (  # noqa: F401
    _ordered_tag_pairs,
    _proportions_from,
)

# Re-exports from cli.utils. Three of the four names below are
# actually used by the @lexicon.command bodies still in this file
# (_DEFAULT_LEXICON_PATH on most lexicon commands' --db default,
# _load_meanings_data on `lexicon build` and `lexicon export-meanings`,
# _select_parser_and_run on the mine-* commands). _decompose_corpus
# is re-exported purely for back-compat with
# `tests/test_kenning_decomposition` which imports it via
# `from wyrd.generators.kenning.cli import _decompose_corpus`; the
# noqa: F401 below documents that the import is intentional even
# though this module doesn't itself call it.
from wyrd.generators.kenning.cli.utils import (  # noqa: F401  (_decompose_corpus)
    _DEFAULT_LEXICON_PATH,
    _decompose_corpus,
    _load_meanings_data,
    _select_parser_and_run,
)
from wyrd.generators.kenning.lexicon import (
    LexiconDB,
    assign_etymon_to_meaning_synset,
    get_meaning_preserving_candidates,
    get_meaning_synsets_for_etymon,
    ingest_parsed_entries,
    init_schema,
    list_meaning_synsets,
    record_mining_run,
    seed_from_meanings,
    seed_meaning_synsets,
)
from wyrd.generators.kenning.lexicon.empirical_priors import (
    dump_empirical_priors_to_json,
    mine_empirical_baselines,
)
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY
from wyrd.generators.kenning.skeat_parser import parse_skeat_text
from wyrd.generators.kenning.wiktextract_corpus_miner import (
    WIKTIONARY_EMPIRICAL_SOURCE_ID,
    compute_unaccounted_fragments,
    derive_positions,
    mine_corpus,
    mine_corpus_forms,
)

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


# Register the extracted top-level subcommands on the @cli group. Each
# module exposes an ``add_to(parent)`` that wires its click command(s)
# onto the parent group. Adding a new top-level subcommand means
# creating a new ``cli/<name>.py`` module and adding an
# ``<name>_module.add_to(cli)`` line below.
_add_meaning_module.add_to(cli)
_creature_module.add_to(cli)
_era_map_module.add_to(cli)
_explain_module.add_to(cli)
_generate_module.add_to(cli)
_rebuild_proportions_module.add_to(cli)
_rewind_module.add_to(cli)
_unaccounted_module.add_to(cli)
_validate_meanings_module.add_to(cli)

# Register the @lexicon sub-group. The group object is imported above so
# the remaining ``@lexicon.command`` decorators in this file (pending
# extraction in slices 2-5 of wyrd-g143) decorate against the same
# shared group.
_register_lexicon_group(cli)


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
@click.option(
    "--capture-failures",
    "capture_failures",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write a JSONL record for every chunk where BOTH tiers failed "
    "(primary AND fallback). Each record has source_id, chunk_index, "
    "chunk_body, error (chained primary=...; fallback=...). Streamed as "
    "failures happen rather than buffered in-memory — bypasses the "
    "head/tail cap on report.failed_chunks. wyrd-3gbx parity with the "
    "single-tier mine-toponym-mentions failure-streaming.",
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
    capture_failures: Path | None,
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
        FailedChunk,
        ToponymMention,
        mine_toponym_mentions_tiered,
    )

    if skip_existing and force:
        raise click.ClickException("--skip-existing and --force are mutually exclusive")

    output_dir.mkdir(parents=True, exist_ok=True)

    # --capture-failures sink setup. Append mode so a resume across
    # multiple invocations accumulates failures end-to-end (mirrors
    # the staged-cascade behavior). Warn on existing non-empty file
    # so an operator-blind append doesn't silently bury old records.
    failure_sink = None
    failures_appended = 0
    if capture_failures is not None:
        capture_failures.parent.mkdir(parents=True, exist_ok=True)
        if capture_failures.exists() and capture_failures.stat().st_size > 0:
            with capture_failures.open("r", encoding="utf-8") as _fh:
                stale = sum(1 for ln in _fh if ln.strip())
            click.echo(
                f"  warning: --capture-failures {capture_failures} already has "
                f"{stale} record(s); appending (`> {capture_failures}` to clear)",
                err=True,
            )
        failure_sink = capture_failures.open("a", encoding="utf-8")

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

    # try/finally wraps the source loop so the --capture-failures sink
    # is closed on every exit path — ClickException for output-exists
    # conflicts (raised inside the loop), KeyboardInterrupt mid-iteration,
    # unexpected raises from mine_toponym_mentions_tiered, atomic-write
    # failures. Without this, the sink handle leaks on every non-clean
    # exit; matches the staged CLI's pattern.
    try:
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

            # Per-source on_chunk_failed closure — captures source_id and
            # the extractor model pair by default-arg (B023 closure-over-
            # loop-var, same shape as ``progress`` above) so the values
            # baked into each emitted record are the CURRENT iteration's,
            # not the last one's. ``extractor`` field mirrors the staged
            # CLI's shape so downstream tooling (mine-toponym-mentions-
            # staged --from-failures, etc.) can identify which model pair
            # produced the failure.
            extractor_tag = f"tiered:{primary_client.model}+{fallback_client.model}"

            def on_fail(
                fc: FailedChunk,
                _sid: str = source_id,
                _sink=failure_sink,
                _extractor: str = extractor_tag,
            ) -> None:
                nonlocal failures_appended
                _sink.write(
                    json.dumps(
                        {
                            "source_id": _sid,
                            "chunk_index": fc.index,
                            "chunk_body": fc.chunk_body,
                            "error": fc.error,
                            "extractor": _extractor,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                _sink.flush()
                failures_appended += 1

            report = mine_toponym_mentions_tiered(
                primary_client,
                fallback_client,
                source_id,
                body,
                target_chunk_size=chunk_size,
                limit=limit,
                hallucination_fallback_threshold=hallucination_fallback_threshold,
                on_chunk_done=progress,
                on_chunk_failed=on_fail if failure_sink is not None else None,
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
    finally:
        if failure_sink is not None:
            failure_sink.close()

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
    # Always surface the capture-failures status — operators want to
    # know whether the streamed file got new records this run or stayed
    # unchanged (mirrors the staged-cascade CLI's behavior).
    if capture_failures is not None:
        if failures_appended:
            click.echo(
                f"  {failures_appended} failed chunk(s) appended → {capture_failures}",
                err=True,
            )
        else:
            click.echo(
                f"  0 new failures → {capture_failures} unchanged this run",
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


@lexicon.command("canonicalize-toponym-etymology")
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
    "source_id",
    default=None,
    help="Restrict canonicalization to toponyms with at least one "
    "toponym_etymology row from this source_id. Default: every toponym.",
)
@click.option(
    "--apply",
    is_flag=True,
    default=False,
    help="Write to DB. Without this flag, runs in dry-run mode and prints "
    "the predicted promote/no-consensus counts.",
)
@click.option(
    "--audit-out",
    "audit_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write one JSONL row per decision (toponym_id, promoted_etymology_id, "
    "consensus_size, cluster_key, runner_up_witness_count, total_clusters). "
    "Use the audit trail to diff cross-run promotions or feed downstream "
    "consumers a list of newly-canonical toponyms.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite --audit-out if it exists.",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Print per-toponym decisions to stderr (suppressed by default — "
    "the corpus has ~thousands of toponyms).",
)
def lexicon_canonicalize_toponym_etymology(
    db_path: Path,
    source_id: str | None,
    apply: bool,
    audit_path: Path | None,
    force: bool,
    verbose: bool,
) -> None:
    """Canonicalize toponym_etymology rows via multi-extractor consensus (wyrd-08qv).

    Three LLM extractor passes (qwen, haiku, gemini) over the same
    source body produce three ``toponym_etymology`` rows per toponym.
    When ≥2 of them agree on the element-tuple (normalized form +
    language per ordinal), the answer is settled. This command:

    1. Clusters rows by (normalized element-tuple, languages) per toponym.
    2. Counts distinct extractor witnesses per cluster (parsing the
       ``extracted_by:provider:model`` prefix from the notes field).
    3. Promotes the largest agreeing cluster (≥2 witnesses) to canonical
       — marks one row ``is_canonical=1`` + stamps ``consensus_size`` +
       ``cluster_key`` on every row for audit visibility.
    4. Re-runs are idempotent: previously-canonical rows that lose
       consensus get demoted; single-witness rows stay
       ``is_canonical=0`` (but still get ``cluster_key`` /
       ``consensus_size`` stamped each pass for audit completeness).

    Downstream consumers read ``toponym_etymology_canonical`` (a VIEW
    over the canonical-only rows). Future LLM re-extraction passes can
    skip toponyms with ``is_canonical = 1`` rows to save cost on
    settled answers.
    """
    from wyrd.generators.kenning.toponym_etymology_canonical import (
        CanonicalDecision,
        apply_canonical_decisions,
        compute_canonical_plans,
    )

    if audit_path is not None and audit_path.exists() and not force:
        raise click.ClickException(f"--audit-out {audit_path} exists; pass --force to overwrite")
    db = LexiconDB(db_path)
    click.echo(f"Using DB {db_path}", err=True)

    # Use compute_canonical_plans (not compute_canonical_decisions) so
    # the per-toponym row_updates are built once. apply_canonical_decisions
    # consumes the plans directly; this avoids the previous N+1 reload
    # on the apply side (wyrd-08qv R1 — Gemini HIGH).
    plans = compute_canonical_plans(db, source_id=source_id)
    if not plans:
        click.echo("No toponym_etymology rows to canonicalize.", err=True)
        return

    # Dry-run preview: count outcomes BEFORE writing.
    promoted = sum(1 for p in plans if p.decision.promoted_etymology_id is not None)
    no_consensus = sum(
        1 for p in plans if p.decision.promoted_etymology_id is None and not p.all_empty_elements
    )
    no_elements = sum(1 for p in plans if p.all_empty_elements)
    click.echo(
        f"Examining {len(plans)} toponym(s)"
        + (f" (filtered to source_id={source_id!r})" if source_id else "")
        + f": promote={promoted} no_consensus={no_consensus} no_elements={no_elements}",
        err=True,
    )

    if verbose:
        for plan in plans:
            d = plan.decision
            if d.promoted_etymology_id is not None:
                click.echo(
                    f"  toponym#{d.toponym_id}: promote ety#{d.promoted_etymology_id} "
                    f"(consensus={d.consensus_size}, clusters={d.total_clusters}, "
                    f"runner_up={d.runner_up_witness_count}) "
                    f"key={d.cluster_key!r}",
                    err=True,
                )
            else:
                click.echo(
                    f"  toponym#{d.toponym_id}: no consensus "
                    f"(clusters={d.total_clusters}, "
                    f"max_witnesses={d.runner_up_witness_count})",
                    err=True,
                )

    if not apply:
        click.echo("(dry-run — pass --apply to write)", err=True)
        return

    audit_sink = None
    if audit_path is not None:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_sink = audit_path.open("w", encoding="utf-8")

    def _audit(decision: CanonicalDecision) -> None:
        if audit_sink is None:
            return
        audit_sink.write(
            json.dumps(
                {
                    "toponym_id": decision.toponym_id,
                    "promoted_etymology_id": decision.promoted_etymology_id,
                    "consensus_size": decision.consensus_size,
                    "cluster_key": decision.cluster_key,
                    "runner_up_witness_count": decision.runner_up_witness_count,
                    "total_clusters": decision.total_clusters,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    try:
        summary = apply_canonical_decisions(db, plans, on_decision=_audit)
    finally:
        if audit_sink is not None:
            audit_sink.close()

    click.echo(
        f"TOTAL toponyms={summary.toponyms_examined} "
        f"promoted={summary.toponyms_promoted} "
        f"no_consensus={summary.toponyms_no_consensus} "
        f"no_elements={summary.toponyms_no_elements} "
        f"rows_updated={summary.rows_updated} (APPLIED)",
        err=True,
    )
    if audit_path is not None:
        click.echo(f"Wrote audit JSONL → {audit_path}", err=True)


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


@lexicon.command("mine-empirical-baselines")
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
        "Replace empirical_priors_native + empirical_priors_loan with "
        "freshly-extracted counts. Without this, dry-run only."
    ),
)
@click.option(
    "--progress-every",
    type=int,
    default=5000,
    show_default=True,
    help="Emit a stderr progress line every N rows scanned. 0 = silent.",
)
def lexicon_mine_empirical_baselines(
    db_path: Path, apply_changes: bool, progress_every: int
) -> None:
    """wyrd-ecjp.2: extract empirical-baseline priors per (culture x
    position x tag x era) from toponym_etymology + elements + tags.

    Two parallel views (D36.4 Option B):
      * empirical_priors_native — all elements in <culture>-recipient
        toponyms.
      * empirical_priors_loan — same observations partitioned by donor
        language; scenario packs read this for their template's
        loan-relationship baseline.

    LLM-free, deterministic, idempotent. Replace-not-merge each apply
    run — stale cells from removed observations don't persist.

    Pair with ``lexicon dump-empirical-priors --output ...`` to emit a
    review-friendly JSON sidecar after writing the tables.
    """
    with LexiconDB(db_path) as db:
        result = mine_empirical_baselines(db, apply=apply_changes, progress_every=progress_every)
    click.echo("mine-empirical-baselines:", err=True)
    click.echo(
        "  scanned={scanned:>6}  emitted={emitted:>6}  "
        "native_cells={nc:>5}  loan_cells={lc:>5}".format(
            scanned=result["rows_scanned"],
            emitted=result["rows_emitted"],
            nc=result["native_cells_written"],
            lc=result["loan_cells_written"],
        ),
        err=True,
    )
    click.echo(
        "  skipped: country_unknown={cu:>5}  country_unmapped={cm:>5}  "
        "year_unknown={yu:>5}  year_out_of_range={yo:>5}  "
        "no_tag={nt:>5}".format(
            cu=result["skipped_country_unknown"],
            cm=result["skipped_country_unmapped"],
            yu=result["skipped_year_unknown"],
            yo=result["skipped_year_out_of_range"],
            nt=result["skipped_no_tag"],
        ),
        err=True,
    )
    if not apply_changes:
        click.echo(
            "(dry-run; pass --apply to write empirical_priors_* rows)",
            err=True,
        )


@lexicon.command("dump-empirical-priors")
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
    required=True,
    help="Path to write the JSON priors artifact.",
)
@click.option(
    "--version",
    type=str,
    default="unversioned",
    show_default=True,
    help=(
        "Version string written verbatim into the JSON. Use a content "
        "hash, sequential int, or date per your versioning policy."
    ),
)
def lexicon_dump_empirical_priors(db_path: Path, output_path: Path, version: str) -> None:
    """wyrd-ecjp.2: dump empirical_priors_* tables to a sorted, byte-
    stable JSON artifact. Run after ``mine-empirical-baselines --apply``.

    Output shape: a top-level dict with ``version`` (verbatim from
    --version) and two lists of cell records (``native``, ``loan``).
    Each cell record carries its key fields + a sorted ``lemmas``
    dict mapping lemma_ref to count. Sort order is deterministic so
    re-running on the same DB state produces a byte-identical file.

    The committed JSON is the operator-visible diff surface for
    priors evolution.
    """
    with LexiconDB(db_path) as db:
        result = dump_empirical_priors_to_json(db, output_path, version=version)
    click.echo("dump-empirical-priors:", err=True)
    click.echo(
        f"  wrote {result['native_cells']} native cells + "
        f"{result['loan_cells']} loan cells to {output_path}",
        err=True,
    )


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
