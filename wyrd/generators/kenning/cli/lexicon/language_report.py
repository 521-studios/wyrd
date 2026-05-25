"""``wyrd kenning lexicon language-report`` — per-language quality / dashboard report."""

from __future__ import annotations

import json
from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH, _readonly_lexicon
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("language-report")
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
        # d90t cutover: the bundle ships as L4 SQLite, not JSON.
        # Rehydrate to the bundle-dict shape the rest of the report
        # consumes (subjects[], joiners, canonical_decompositions).
        from wyrd.generators.kenning.runtime.runtime_db import get_runtime_db
        from wyrd.generators.kenning.runtime.runtime_db_adapter import (
            bundle_dict_from_runtime_db,
        )

        bundle = bundle_dict_from_runtime_db(get_runtime_db())
    else:
        bundle = json.loads(bundle_path.read_text())

    if proportions_path is None:
        # English proportions come from the same L4 (proportions_*
        # tables). Build the dict in-memory so load_reference_tags's
        # JSON-path expectation stays one branch wide; we write a
        # temp file rather than divergent in-memory/path codepaths.
        import tempfile

        from wyrd.generators.kenning.runtime.runtime_db import get_runtime_db
        from wyrd.generators.kenning.runtime.runtime_db_adapter import (
            proportions_dict_for_culture,
        )

        proportions_dict = proportions_dict_for_culture(get_runtime_db(), "english")
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tmp:
            json.dump(proportions_dict, tmp)
            proportions_path = Path(tmp.name)

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

    with _readonly_lexicon(db_path) as conn:
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

    if json_path is not None:
        json_path.write_text(report_to_json(report))

    if output_format == "json":
        click.echo(report_to_json(report))
    else:
        click.echo(report_to_markdown(report))


def add_to(parent: click.Group) -> None:
    """Register ``language-report`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_language_report)
