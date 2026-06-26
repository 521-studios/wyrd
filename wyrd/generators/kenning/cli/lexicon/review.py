"""``wyrd kenning lexicon review`` — Tier-2 LLM review pass over Tier-1 mining declines / lows (D2/D19)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click

from wyrd.generators.kenning import anthropic_extractor, gemini_extractor, llm_extractor
from wyrd.generators.kenning.cli.utils import (
    _DEFAULT_LEXICON_PATH,
    _readonly_lexicon,
    _select_parser_and_run,
)
from wyrd.generators.kenning.lexicon import (
    LexiconDB,
    LexiconError,
    ingest_parsed_entries,
    record_mining_run,
)
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("review")
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
    by_book: dict[str, list[Any]] = {}
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
    with _readonly_lexicon(db_path) as conn:
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


def _select_review_candidates(
    *,
    db_path: Path,
    provider_tag: str,
    confidence: tuple[str, ...],
    book: str | None,
    limit: int | None,
    include_haiku: bool = False,
    include_sonnet: bool = False,
) -> list[Any]:
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
    with _readonly_lexicon(db_path) as conn:
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


@dataclass(frozen=True)
class _ReviewOutcome:
    """One row's classified review result, ready for the loop to act on."""

    counter: str  # 'rejected' | 'declined' | 'written'
    log_line: str
    failures: tuple[str, ...] = ()  # rejection reasons (rejected only)
    persist_entry: Any = None  # ParsedEntry to write (written only)


def _classify_review_result(name: str, result: Any, dt: float) -> _ReviewOutcome:
    """Take a provider's LLMResult and produce a structured outcome.

    Mirrors the writer's drop-low gate (`lexicon/ingest.py`
    ``ingest_parsed_entries``: ``has_breakdown = confidence != "low" and
    bool(entry.elements)``) so dry-run and apply-mode summaries both report
    what's actually persisted.
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
    rows: list[Any],
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
        LexiconError,
    ) as e:
        return f"line {line_no}: {e}"


def add_to(parent: click.Group) -> None:
    """Register ``review`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_review)
