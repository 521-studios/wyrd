"""``wyrd kenning lexicon mine-skeat`` — mine a Skeat-format etymology book via the LLM extractor."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import click

from wyrd.generators.kenning.cli.lexicon.build import _KNOWN_SKEAT_BOOKS
from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB, ingest_parsed_entries
from wyrd.generators.kenning.parsers.skeat import parse_skeat_text
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("mine-skeat")
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


def add_to(parent: click.Group) -> None:
    """Register ``mine-skeat`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_mine_skeat)
