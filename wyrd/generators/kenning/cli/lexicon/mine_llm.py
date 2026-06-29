"""``wyrd kenning lexicon mine-llm`` — mine an alphabetical / numbered-list etymology book via the LLM extractor."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import click

from wyrd.generators.kenning.cli.lexicon.build import _KNOWN_SKEAT_BOOKS
from wyrd.generators.kenning.cli.lexicon.mine_skeat import _mine_entries
from wyrd.generators.kenning.cli.lexicon.review import (
    _build_llm_client,
    _select_already_extracted_toponyms,
    _warn_if_paid_concurrency_too_high,
)
from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH, _select_parser_and_run
from wyrd.generators.kenning.lexicon import LexiconDB, ingest_parsed_entries, record_mining_run
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("mine-llm")
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

    text = path.read_text(encoding="utf-8")
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


def add_to(parent: click.Group) -> None:
    """Register ``mine-llm`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_mine_llm)
