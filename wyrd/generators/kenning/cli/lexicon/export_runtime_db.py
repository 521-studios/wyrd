"""``wyrd kenning lexicon export-runtime-db`` — emit the L4 runtime SQLite DB.

Thin click wrapper around
:func:`wyrd.generators.kenning.lexicon.runtime_db_export.write_runtime_db`
— see that module for the architecture rationale (D38).
"""

from __future__ import annotations

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
    DEV_TOP_N_PER_CULTURE,
    EMITTER_VERSION,
    SCHEMA_VERSION,
    write_runtime_db,
)
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY

# Re-exports kept for tests that import the schema-version constants
# through this module path.
__all__ = ["EMITTER_VERSION", "SCHEMA_VERSION", "lexicon_export_runtime_db"]

# Canonical defaults for the upstream filter flags. Used in BOTH the
# @click.option ``default=`` declarations AND _reject_non_default_filters_under_dev
# so the two can't drift — bumping a default here propagates to both
# sites at once.
_DEFAULT_MIN_WITNESSES = 3
_DEFAULT_USE_PRESET = True
_DEFAULT_INCLUDE_RANDO = True
_DEFAULT_INCLUDE_WIKTIONARY_EMPIRICAL = True
_DEFAULT_INCLUDE_WAVE2_ENRICHED = True


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
        "Directory containing <culture>_proportions.json files (operator "
        "override). Default behavior rebuilds the proportions inline from "
        "the L3 lexicon + bundled <culture>_place_names.json corpora, so "
        "no intermediate JSON artifact is required."
    ),
)
@click.option(
    "--min-witnesses",
    type=int,
    default=_DEFAULT_MIN_WITNESSES,
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
    default=_DEFAULT_USE_PRESET,
    show_default=True,
    help="Start from RECOMMENDED_LANG_THRESHOLDS (calibrated per language).",
)
@click.option(
    "--include-rando/--no-include-rando",
    default=_DEFAULT_INCLUDE_RANDO,
    show_default=True,
)
@click.option(
    "--include-wiktionary-empirical/--no-include-wiktionary-empirical",
    default=_DEFAULT_INCLUDE_WIKTIONARY_EMPIRICAL,
    show_default=True,
)
@click.option(
    "--include-wave2-enriched/--no-include-wave2-enriched",
    default=_DEFAULT_INCLUDE_WAVE2_ENRICHED,
    show_default=True,
)
@click.option(
    "--dev",
    "dev_subset",
    is_flag=True,
    default=False,
    help=(
        "Emit a committed-seed-shaped subset of the L4 (top N usages per "
        "culture + referenced meanings + all fantasy + all canonical, with "
        "fixed built_at + source_lexicon_db sentinels for byte-stability). "
        "Used to regenerate wyrd/generators/kenning/data/seed-runtime.db. "
        "All upstream filtering knobs (--min-witnesses, --lang-threshold, "
        "--include-rando, --include-wiktionary-empirical, "
        "--include-wave2-enriched) are FORCED to their canonical defaults "
        "in this mode so the seed is reproducible across operators — the "
        "CLI errors if you pass non-default values alongside --dev."
    ),
)
@click.option(
    "--dev-top-n",
    "dev_top_n",
    type=int,
    default=DEV_TOP_N_PER_CULTURE,
    show_default=True,
    help="Per-culture cap on usages / single_usages in --dev mode.",
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
    dev_subset: bool,
    dev_top_n: int,
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
    if dev_subset:
        _reject_non_default_filters_under_dev(
            min_witnesses=min_witnesses,
            lang_threshold_specs=lang_threshold_specs,
            use_preset=use_preset,
            include_rando=include_rando,
            include_wiktionary_empirical=include_wiktionary_empirical,
            include_wave2_enriched=include_wave2_enriched,
        )

    lang_thresholds = _parse_lang_thresholds(lang_threshold_specs, use_preset=use_preset)

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
        proportions_dir=proportions_dir,
        source_lexicon_db=db_path,
        dev_subset=dev_subset,
        dev_top_n_per_culture=dev_top_n,
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


def _reject_non_default_filters_under_dev(
    *,
    min_witnesses: int,
    lang_threshold_specs: tuple[str, ...],
    use_preset: bool,
    include_rando: bool,
    include_wiktionary_empirical: bool,
    include_wave2_enriched: bool,
) -> None:
    """Hard-fail when --dev is combined with any non-default upstream
    filter. The committed seed-runtime.db is byte-stable only if every
    operator runs the same canonical invocation; silently honoring a
    custom --min-witnesses would produce a seed that differs from the
    one in the repo without the operator noticing.

    The canonical defaults come from the module-level ``_DEFAULT_*``
    constants — same source the @click.option ``default=`` declarations
    use, so a future bump to one of those propagates to both sites at
    once and can't drift.
    """
    offenders: list[str] = []
    if min_witnesses != _DEFAULT_MIN_WITNESSES:
        offenders.append(f"--min-witnesses={min_witnesses} (canonical: {_DEFAULT_MIN_WITNESSES})")
    if lang_threshold_specs:
        offenders.append(f"--lang-threshold {lang_threshold_specs!r} (canonical: none)")
    if use_preset != _DEFAULT_USE_PRESET:
        offenders.append("--no-preset (canonical: --preset)")
    if include_rando != _DEFAULT_INCLUDE_RANDO:
        offenders.append("--no-include-rando (canonical: --include-rando)")
    if include_wiktionary_empirical != _DEFAULT_INCLUDE_WIKTIONARY_EMPIRICAL:
        offenders.append(
            "--no-include-wiktionary-empirical (canonical: --include-wiktionary-empirical)"
        )
    if include_wave2_enriched != _DEFAULT_INCLUDE_WAVE2_ENRICHED:
        offenders.append("--no-include-wave2-enriched (canonical: --include-wave2-enriched)")
    if offenders:
        raise click.UsageError(
            "--dev requires canonical defaults on every upstream filter "
            "(otherwise the committed seed-runtime.db is unreproducible "
            "across operators). Conflicts:\n  - " + "\n  - ".join(offenders)
        )


def add_to(parent: click.Group) -> None:
    """Register ``export-runtime-db`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_export_runtime_db)
