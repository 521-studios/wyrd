"""``wyrd kenning lexicon classify-stratum`` — classify etymons into within-language stratum buckets (D32 / wyrd-lr4)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("classify-stratum")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--language",
    type=click.Choice(
        [
            "welsh",
            "french",
            "old-english",
            "old-norse",
            "brittonic",
            "goidelic",
            "celtic",
        ]
    ),
    required=True,
    help=(
        "Language family to classify. Welsh covers etymons with "
        "language IN ('welsh', 'middle-welsh', 'old-welsh', "
        "'cel-bry-pro'); French covers ('french', 'old-french', "
        "'middle-french', 'norman-french', 'vulgar-latin', 'frk', "
        "'cel-gau'); Old English covers language='old-english' "
        "(loan-source ancestors only — dialect axis deferred); Old "
        "Norse covers ('old-norse', 'gmq-osw' Old Swedish, 'gmq-oda' "
        "Old Danish). The Celtic-family classifiers (wyrd-de77) each "
        "cover one coarse leaf tag — brittonic covers "
        "language='brittonic', goidelic covers language='goidelic', "
        "celtic covers language='celtic' — classifying by proto-celtic "
        "/ Gaelic-stage ancestry; 87% of these are orphans landing in "
        "the native-* default (no loan buckets — the data carries no "
        "loan ancestries)."
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
    ``wyrd/generators/kenning/lexicon/strata.py`` for the per-language
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
    from wyrd.generators.kenning.lexicon.strata import (
        BRITTONIC_STRATA,
        CELTIC_STRATA,
        FRENCH_STRATA,
        GOIDELIC_STRATA,
        OLD_ENGLISH_STRATA,
        OLD_NORSE_STRATA,
        WELSH_STRATA,
        classify_brittonic,
        classify_celtic,
        classify_french,
        classify_goidelic,
        classify_old_english,
        classify_old_norse,
        classify_welsh,
    )

    classifiers = {
        "welsh": (classify_welsh, WELSH_STRATA),
        "french": (classify_french, FRENCH_STRATA),
        "old-english": (classify_old_english, OLD_ENGLISH_STRATA),
        "old-norse": (classify_old_norse, OLD_NORSE_STRATA),
        "brittonic": (classify_brittonic, BRITTONIC_STRATA),
        "goidelic": (classify_goidelic, GOIDELIC_STRATA),
        "celtic": (classify_celtic, CELTIC_STRATA),
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


def add_to(parent: click.Group) -> None:
    """Register ``classify-stratum`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_classify_stratum)
