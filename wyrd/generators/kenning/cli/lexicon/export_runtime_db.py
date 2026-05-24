"""``wyrd kenning lexicon export-runtime-db`` — emit the L4 runtime SQLite DB.

Thin click wrapper around
:func:`wyrd.generators.kenning.lexicon.runtime_db_export.write_runtime_db`
— see that module for the architecture rationale (D38).
"""

from __future__ import annotations

from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import (
    LANGUAGE_FIELDS,
    RECOMMENDED_LANG_THRESHOLDS,
    LexiconDB,
    collect_canonical_decompositions,
    collect_fantasy_morphemes,
    export_meanings,
)
from wyrd.generators.kenning.lexicon.runtime_db_export import (
    EMITTER_VERSION,
    SCHEMA_VERSION,
    write_runtime_db,
)
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY

# Re-exports kept for tests that import the schema-version constants
# through this module path.
__all__ = ["EMITTER_VERSION", "SCHEMA_VERSION", "lexicon_export_runtime_db"]


def _bundled_proportions_dir():
    """Return the bundled proportions directory as a Traversable.

    Uses importlib.resources so the default keeps working when the
    package is shipped via a zip-loader. write_runtime_db accepts
    either Path (operator override) or Traversable (this default).
    """
    return resources.files("wyrd.generators.kenning.data")


@click.command("export-runtime-db")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
    help="L3 lexicon DB to read meanings + fantasy morphemes + canonicals from.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Path to write the L4 SQLite DB to. Overwritten if it exists.",
)
@click.option(
    "--proportions-dir",
    "proportions_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help=(
        "Directory containing <culture>_proportions.json files. "
        "Defaults to the bundled wyrd/generators/kenning/data directory "
        "(the same files the runtime reads today)."
    ),
)
@click.option(
    "--min-witnesses",
    type=int,
    default=3,
    show_default=True,
    help="D4 promotion threshold; passed through to export_meanings.",
)
@click.option(
    "--lang-threshold",
    "lang_threshold_specs",
    multiple=True,
    metavar="LANG=N",
    help="Per-language witness override; matches export-meanings semantics.",
)
@click.option(
    "--preset/--no-preset",
    "use_preset",
    default=True,
    show_default=True,
    help="Start from RECOMMENDED_LANG_THRESHOLDS (calibrated per language).",
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
@click.option(
    "--include-wave2-enriched/--no-include-wave2-enriched",
    default=True,
    show_default=True,
)
def lexicon_export_runtime_db(
    db_path: Path,
    output_path: Path,
    proportions_dir: Path | None,
    min_witnesses: int,
    lang_threshold_specs: tuple[str, ...],
    use_preset: bool,
    include_rando: bool,
    include_wiktionary_empirical: bool,
    include_wave2_enriched: bool,
) -> None:
    """Emit the L4 runtime SQLite DB.

    Self-contained snapshot of everything the kenning runtime needs:
    morpheme meanings (blob), fantasy morphemes (blob),
    canonical-decomposition front-loading (blob), and per-culture
    proportions distributions (normalized with precomputed cumulative
    columns so weighted-random sampling is an O(log n) index seek).

    Replaces ``meanings.json`` + 5 ``<culture>_proportions.json`` files
    in a single artifact downloaded from S3 on Lambda cold start (D38).
    """
    lang_thresholds = _parse_lang_thresholds(lang_threshold_specs, use_preset=use_preset)

    proportions_source: Path | Traversable = (
        proportions_dir if proportions_dir is not None else _bundled_proportions_dir()
    )

    with LexiconDB(db_path) as db:
        subjects = export_meanings(
            db,
            min_witnesses=min_witnesses,
            lang_thresholds=lang_thresholds,
            include_rando=include_rando,
            include_wiktionary_empirical=include_wiktionary_empirical,
            include_wave2_enriched=include_wave2_enriched,
        )
        canonical_decompositions = collect_canonical_decompositions(db)
        fantasy_morphemes = collect_fantasy_morphemes(db)

    counts = write_runtime_db(
        output_path=output_path,
        subjects=subjects,
        fantasy_morphemes=fantasy_morphemes,
        canonical_decompositions=canonical_decompositions,
        proportions_dir=proportions_source,
        source_lexicon_db=db_path,
    )

    proportion_total = (
        counts.get("proportions_usage", 0)
        + counts.get("proportions_single_usage", 0)
        + counts.get("proportions_structure", 0)
        + counts.get("proportions_tag_marginal", 0)
        + counts.get("proportions_tag_cooccurrence", 0)
    )
    click.echo(
        f"Wrote L4 DB to {output_path}: "
        f"{counts['meanings']} meaning rows, "
        f"{counts['fantasy_morphemes']} fantasy morphemes, "
        f"{counts['canonical_decompositions']} canonical decompositions, "
        f"{proportion_total} proportion rows across {counts['cultures']} cultures.",
        err=True,
    )


def _parse_lang_thresholds(specs: tuple[str, ...], *, use_preset: bool) -> dict[str, int]:
    """Parse ``LANG=N`` overrides. Mirrors the export-meanings shape."""
    lang_thresholds: dict[str, int] = dict(RECOMMENDED_LANG_THRESHOLDS) if use_preset else {}
    for spec in specs:
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
    return lang_thresholds


def add_to(parent: click.Group) -> None:
    """Register ``export-runtime-db`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_export_runtime_db)
