"""``wyrd kenning lexicon mine-wiktextract-corpus`` — ingest a wiktextract slice into the empirical etymon table."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click

from wyrd.generators.kenning import CULTURES
from wyrd.generators.kenning.bulk_sources import DEFAULT_LOCAL_CACHE_DIR
from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH, _load_meanings_data
from wyrd.generators.kenning.lexicon import LexiconDB, record_mining_run
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY, seed_data_path
from wyrd.generators.kenning.wiktextract_corpus_miner import (
    WIKTIONARY_EMPIRICAL_SOURCE_ID,
    compute_unaccounted_fragments,
    derive_positions,
    mine_corpus,
)


@click.command("mine-wiktextract-corpus")
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
    default=DEFAULT_LOCAL_CACHE_DIR,
    show_default="~/.wyrd/sources",
    help="Directory containing wiktextract_*.jsonl slice files. Defaults "
    "to the bulk-source cache (~/.wyrd/sources), NOT the repo's "
    "OCR-text sources/ dir — this command consumes wiktextract slices, "
    "not the scholarly OCR books. Run 'fetch-bulk-sources' to populate "
    "the cache on a fresh checkout.",
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
            text = seed_data_path(f"{culture}_place_names.json").read_text(encoding="utf-8")
            p = tmpdir / f"{culture}_place_names.json"
            p.write_text(text, encoding="utf-8")
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
    _warn_missing_slices(counts)
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


def _warn_missing_slices(counts: dict[str, Any]) -> None:
    """Loudly warn when languages in scope have a published wiktextract
    slice that isn't present in --sources-dir.

    This is the wyrd-34kt guardrail: such a run produces 0 hits for the
    affected languages with no other signal. Pointing --sources-dir at
    the repo's OCR-text ``sources/`` (instead of ~/.wyrd/sources) skips
    EVERY slice this way, which is exactly how the f1ss rebuild's
    empirical layer came back empty.
    """
    missing = counts.get("missing_slices") or []
    if not missing:
        return
    click.echo(
        f"WARNING: {len(missing)} in-scope language slice(s) NOT found in "
        "--sources-dir — those languages contribute 0 hits:",
        err=True,
    )
    for lang, basename in missing:
        click.echo(f"  missing: {lang:20s} ({basename})", err=True)
    click.echo(
        "  → Check --sources-dir points at the bulk cache (~/.wyrd/sources), "
        "not the OCR-text sources/ dir; run 'fetch-bulk-sources' if the "
        "cache is empty.",
        err=True,
    )


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


def add_to(parent: click.Group) -> None:
    """Register ``mine-wiktextract-corpus`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_mine_wiktextract_corpus)
