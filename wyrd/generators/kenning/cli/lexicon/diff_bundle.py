"""``wyrd kenning lexicon diff-bundle`` — diff the committed meanings.json against a fresh export (wyrd-w3x0)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import click

from wyrd.generators.kenning.cli.lexicon.export_meanings import _load_joiners_sidecar
from wyrd.generators.kenning.lexicon import (
    LANGUAGE_FIELDS,
    RECOMMENDED_LANG_THRESHOLDS,
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
    # Deferred imports: bulk_sources, enrichment, jsonl.build pull in
    # heavy dependencies (LLM clients, sqlite extension setup) that
    # other lexicon commands don't need; deferring them keeps `wyrd
    # kenning --help` snappy. ``init_schema`` is already imported at
    # module level so we don't re-import it here.

    from wyrd.generators.kenning.bulk_sources import ingest_all_slices
    from wyrd.generators.kenning.bundle_diff import compute_bundle_diff, format_bundle_diff
    from wyrd.generators.kenning.enrichment import run_full_enrichment
    from wyrd.generators.kenning.jsonl.build import (
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


def add_to(parent: click.Group) -> None:
    """Register ``diff-bundle`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_diff_bundle)
