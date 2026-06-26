"""``wyrd kenning lexicon diff-bundle`` — diff the committed meanings.json against a fresh export (wyrd-w3x0)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import click

from wyrd.generators.kenning.cli.lexicon._threshold_args import parse_lang_thresholds
from wyrd.generators.kenning.cli.lexicon.export_meanings import _load_joiners_sidecar
from wyrd.generators.kenning.lexicon import (
    LexiconDB,
    collect_canonical_decompositions,
    collect_fantasy_morphemes,
    export_meanings,
    init_schema,
)


@click.command("diff-bundle")
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
@click.option(
    "--include-wave2-enriched/--no-include-wave2-enriched",
    default=True,
    show_default=True,
    help="wyrd-z3cp: matches export-meanings's flag of the same name.",
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
    include_wave2_enriched: bool,
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
    # Deferred imports: bulk_sources, enrichment, jsonl.build pull in
    # heavy dependencies (LLM clients, sqlite extension setup) that
    # other lexicon commands don't need; deferring them keeps `wyrd
    # kenning --help` snappy. ``init_schema`` is already imported at
    # module level so we don't re-import it here.

    from wyrd.generators.kenning.bundle_diff import compute_bundle_diff, format_bundle_diff

    # Shared with `export-meanings` so the two CLIs accept identical
    # flag shapes — operators validating the committed bundle can pass
    # the same `--lang-threshold` flags the export used.
    lang_thresholds = parse_lang_thresholds(lang_threshold_specs, use_preset=use_preset)

    joiners = _load_joiners_sidecar(joiners_path) if joiners_path is not None else {}

    with tempfile.TemporaryDirectory(prefix="wyrd-diff-bundle-") as tmpdir:
        rebuilt_path = Path(tmpdir) / "rebuilt.db"
        init_schema(rebuilt_path)
        _rebuild_lexicon_db(
            rebuilt_path,
            jsonl_dir=jsonl_dir,
            fetch_bulk=fetch_bulk,
            skip_bulk=skip_bulk,
        )
        bundle = _assemble_bundle_dict(
            rebuilt_path,
            min_witnesses=min_witnesses,
            lang_thresholds=lang_thresholds,
            include_rando=include_rando,
            include_wiktionary_empirical=include_wiktionary_empirical,
            include_wave2_enriched=include_wave2_enriched,
            joiners=joiners,
        )

    rebuilt_text = json.dumps(bundle, ensure_ascii=False, indent=2) + "\n"
    committed_text = bundle_path.read_text(encoding="utf-8")

    diff = compute_bundle_diff(committed_text, rebuilt_text)
    click.echo(format_bundle_diff(diff))

    if not diff.bytes_match:
        raise SystemExit(1)


def _rebuild_lexicon_db(
    rebuilt_path: Path,
    *,
    jsonl_dir: Path,
    fetch_bulk: bool,
    skip_bulk: bool,
) -> None:
    """Run the L1 bulk ingest (unless --skip-bulk), L2 JSONL replay, and
    L3 enrichment chain against an already-init_schema'd DB at
    ``rebuilt_path``. The caller owns the tmpdir lifetime — this helper
    only writes; it does not delete."""
    # Deferred imports: bulk_sources, enrichment, jsonl.build pull in
    # heavy dependencies (LLM clients, sqlite extension setup) that
    # other lexicon commands don't need. Mirrors the deferral in the
    # parent lexicon_diff_bundle docstring — kept here too because
    # this helper is also reachable from future internal callers that
    # may want the same `wyrd kenning --help` snappiness guarantee.
    from wyrd.generators.kenning.bulk_sources import ingest_all_slices
    from wyrd.generators.kenning.enrichment import run_full_enrichment
    from wyrd.generators.kenning.jsonl.build import (
        build_from_jsonl,
        collect_collapses,
        collect_curation_overrides,
        collect_etymon_splits,
        collect_gloss_additions,
        collect_gloss_suppressions,
        jsonl_paths_in,
    )

    if not skip_bulk:
        click.echo("Bulk ingest (L1)...", err=True)
        with LexiconDB(rebuilt_path) as db:
            bulk_result = ingest_all_slices(db, apply=True, fetch=fetch_bulk)
        if bulk_result.failed:
            details = [
                f"Bulk ingest failed: {len(bulk_result.failed)} slice(s)",
                *(f"  ! {name}: {reason}" for name, reason in bulk_result.failed),
            ]
            # ClickException renders to stderr + exits with status 1 at
            # the click command boundary, so the helper can stay pure
            # (no SystemExit inside a library function) while preserving
            # the same operator-visible failure behavior. Gemini round-2
            # review on PR #285.
            raise click.ClickException("\n".join(details))

    click.echo("L2 replay...", err=True)
    paths = jsonl_paths_in(jsonl_dir)
    with LexiconDB(rebuilt_path) as db:
        build_from_jsonl(db.conn, paths)

    click.echo("L3 enrichment chain...", err=True)
    curation_state = collect_curation_overrides(paths)
    suppression_state = collect_gloss_suppressions(paths)
    addition_state = collect_gloss_additions(paths)
    split_state = collect_etymon_splits(paths)
    collapse_state = collect_collapses(paths)
    from wyrd.generators.kenning.lexicon.element_gloss_backfill import (
        collect_element_glosses,
    )

    element_gloss_state = collect_element_glosses(
        str(jsonl_dir / "_element_glosses.jsonl"),
        str(jsonl_dir / "_element_gloss_adjudications.jsonl"),
    )
    from wyrd.generators.kenning.lexicon.pronunciation_backfill import collect_pronunciation

    pronunciation_state = collect_pronunciation(str(jsonl_dir / "_pronunciation.jsonl"))
    with LexiconDB(rebuilt_path) as db:
        run_full_enrichment(
            db,
            apply=True,
            curation_state=curation_state or None,
            suppression_state=suppression_state or None,
            addition_state=addition_state or None,
            split_state=split_state or None,
            collapse_state=collapse_state or None,
            element_gloss_state=element_gloss_state or None,
            pronunciation_state=pronunciation_state or None,
        )


def _assemble_bundle_dict(
    rebuilt_path: Path,
    *,
    min_witnesses: int,
    lang_thresholds: dict[str, int],
    include_rando: bool,
    include_wiktionary_empirical: bool,
    include_wave2_enriched: bool,
    joiners: dict[str, Any],
) -> dict[str, Any]:
    """Build the bundle dict the same way lexicon_export_meanings does.
    If these two builders ever drift, the round-trip test in
    tests/test_kenning_hidb_phase3_round_trip.py + the integration
    test for this command will catch it."""
    click.echo("Building bundle from rebuild...", err=True)
    with LexiconDB(rebuilt_path) as db:
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
    bundle: dict[str, Any] = {"subjects": subjects}
    if canonical_decompositions:
        bundle["canonical_decompositions"] = canonical_decompositions
    if joiners:
        bundle["joiners"] = joiners
    if fantasy_morphemes:
        bundle["fantasy_morphemes"] = fantasy_morphemes
    return bundle


def add_to(parent: click.Group) -> None:
    """Register ``diff-bundle`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_diff_bundle)
