"""``wyrd kenning lexicon mine-toponym-mentions-tiered`` — tiered toponym-mention mining (cheap → expensive)."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import click

from wyrd.generators.kenning.cli.lexicon._dedup import dedup_key
from wyrd.generators.kenning.cli.lexicon._mention_mining_io import (
    open_failure_sink,
    resolve_source_ids,
)
from wyrd.generators.kenning.cli.lexicon.mine_toponym_mentions import _build_extractor_client


def _tiered_line(source_id: str, m) -> str:
    return json.dumps(
        {
            "source_id": source_id,
            "form": m.form,
            "date_year": m.date_year,
            "region_hint": m.region_hint,
            "context": m.context,
        },
        ensure_ascii=False,
    )


def _emit_source_summary(out_path: Path, report) -> None:
    """Per-source summary line. halluc_rescued shape succeeded/attempted: a gap
    is the broken-fallback signal (every fallback erroring OR returning empty —
    either way the primary's hallucinations weren't caught). wyrd-z8mq."""
    click.echo(
        f"  → {out_path} | chunks={report.chunks_processed} "
        f"(recovered={report.chunks_recovered_by_fallback} "
        f"halluc_rescued={report.chunks_hallucination_rescue_succeeded}/"
        f"{report.chunks_hallucination_rescue_attempted} "
        f"failed={report.chunks_failed}) mentions={len(report.mentions)} "
        f"halluc={report.hallucinations_dropped} "
        f"clamped={report.years_clamped} "
        f"surrogates={report.surrogates_sanitized}",
        err=True,
    )


def _accumulate_totals(totals: dict, report) -> None:
    """Fold one source's report into the run totals."""
    totals["sources_processed"] += 1
    totals["chunks_processed"] += report.chunks_processed
    totals["chunks_recovered"] += report.chunks_recovered_by_fallback
    totals["chunks_hallucination_rescue_attempted"] += report.chunks_hallucination_rescue_attempted
    totals["chunks_hallucination_rescue_succeeded"] += report.chunks_hallucination_rescue_succeeded
    totals["chunks_failed"] += report.chunks_failed
    totals["hallucinations_dropped"] += report.hallucinations_dropped
    totals["years_clamped"] += report.years_clamped
    totals["surrogates_sanitized"] += report.surrogates_sanitized
    totals["mentions"] += len(report.mentions)


def _emit_tiered_total(totals: dict) -> None:
    click.echo("", err=True)
    click.echo(
        f"TOTAL sources={totals['sources_processed']} "
        f"(skipped={totals['sources_skipped']}) "
        f"chunks={totals['chunks_processed']} "
        f"recovered_by_fallback={totals['chunks_recovered']} "
        f"halluc_rescued={totals['chunks_hallucination_rescue_succeeded']}/"
        f"{totals['chunks_hallucination_rescue_attempted']} "
        f"failed={totals['chunks_failed']} "
        f"halluc={totals['hallucinations_dropped']} "
        f"clamped={totals['years_clamped']} "
        f"surrogates={totals['surrogates_sanitized']} "
        f"mentions={totals['mentions']}",
        err=True,
    )


def _report_capture_failures_status(capture_failures: Path | None, failures_appended: int) -> None:
    """Always surface whether the streamed --capture-failures file got new
    records this run or stayed unchanged (mirrors the staged-cascade CLI)."""
    if capture_failures is None:
        return
    if failures_appended:
        click.echo(
            f"  {failures_appended} failed chunk(s) appended → {capture_failures}",
            err=True,
        )
    else:
        click.echo(
            f"  0 new failures → {capture_failures} unchanged this run",
            err=True,
        )


@dataclass
class _TieredRun:
    """Shared config + mutable run state for one mine-toponym-mentions-tiered
    invocation, so _process_one_source takes just (ctx, src_i, source_id)."""

    sources_dir: Path
    output_dir: Path
    primary_provider: str
    primary_model: str | None
    fallback_provider: str
    fallback_model: str | None
    ollama_url: str | None
    chunk_size: int
    limit: int | None
    hallucination_fallback_threshold: int | None
    skip_existing: bool
    force: bool
    failure_sink: object
    n_sources: int
    totals: dict
    primary_client: object = None
    fallback_client: object = None
    failures_appended: int = 0

    def ensure_clients(self) -> None:
        """Lazy client construction — only instantiate on the first real
        extraction (a --skip-existing resume where every source is skipped
        needs neither client + avoids ANTHROPIC_API_KEY validation / Ollama-URL
        probing). Built once and cached for the remainder."""
        if self.primary_client is None:
            self.primary_client = _build_extractor_client(
                self.primary_provider, self.primary_model, ollama_url=self.ollama_url
            )
            self.fallback_client = _build_extractor_client(
                self.fallback_provider, self.fallback_model, ollama_url=self.ollama_url
            )


def _process_one_source(ctx: _TieredRun, src_i: int, source_id: str) -> None:
    """Extract one source: skip/overwrite handling, body read, tiered mining,
    atomic JSONL write, per-source summary, totals fold."""
    from wyrd.generators.kenning.extractors.toponym_mentions import (
        FailedChunk,
        mine_toponym_mentions_tiered,
    )

    out_path = ctx.output_dir / f"{source_id}.jsonl"
    if out_path.exists():
        if ctx.skip_existing:
            click.echo(
                f"[{src_i}/{ctx.n_sources}] {source_id}: SKIP (output exists)",
                err=True,
            )
            ctx.totals["sources_skipped"] += 1
            return
        if not ctx.force:
            raise click.ClickException(
                f"output {out_path} exists; pass --skip-existing to resume or --force to overwrite"
            )

    txt_path = ctx.sources_dir / f"{source_id}.txt"
    body = txt_path.read_text(encoding="utf-8", errors="replace")
    if "�" in body:
        click.echo(
            f"  warning: {source_id} contained invalid UTF-8 sequences; some "
            f"mentions may be silently rejected",
            err=True,
        )

    click.echo(
        f"[{src_i}/{ctx.n_sources}] {source_id}: {len(body):,} chars",
        err=True,
    )

    ctx.ensure_clients()

    start_ts = time.monotonic()

    def progress(done: int, total: int, mentions: int, _start_ts: float = start_ts) -> None:
        # Bind start_ts via default arg — B023: a closure reference to the loop
        # variable would all share the LAST iteration's start_ts.
        elapsed = time.monotonic() - _start_ts
        rate = elapsed / done if done else 0.0
        click.echo(
            f"    chunk [{done}/{total}] mentions={mentions} ({rate:.1f}s/chunk)",
            err=True,
        )

    def warn(msg: str) -> None:
        click.echo(f"    warning: {msg}", err=True)

    # extractor tag mirrors the staged CLI's shape so downstream tooling
    # (mine-toponym-mentions-staged --from-failures) can identify the model pair.
    extractor_tag = f"tiered:{ctx.primary_client.model}+{ctx.fallback_client.model}"

    def on_fail(
        fc: FailedChunk,
        _sid: str = source_id,
        _sink=ctx.failure_sink,
        _extractor: str = extractor_tag,
    ) -> None:
        _sink.write(
            json.dumps(
                {
                    "source_id": _sid,
                    "chunk_index": fc.index,
                    "chunk_body": fc.chunk_body,
                    "error": fc.error,
                    "extractor": _extractor,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        _sink.flush()
        ctx.failures_appended += 1

    report = mine_toponym_mentions_tiered(
        ctx.primary_client,
        ctx.fallback_client,
        source_id,
        body,
        target_chunk_size=ctx.chunk_size,
        limit=ctx.limit,
        hallucination_fallback_threshold=ctx.hallucination_fallback_threshold,
        on_chunk_done=progress,
        on_chunk_failed=on_fail if ctx.failure_sink is not None else None,
        log_warning=warn,
    )

    # Atomic write: temp file + rename so an interrupt mid-write leaves the
    # previous output (if any) intact rather than truncated.
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open(
        "w", encoding="utf-8", errors="replace"
    ) as sink:  # wyrd-klod surrogate backstop
        for m in report.mentions:
            sink.write(_tiered_line(source_id, m) + "\n")
    tmp_path.replace(out_path)

    _emit_source_summary(out_path, report)
    _accumulate_totals(ctx.totals, report)


@click.command("mine-toponym-mentions-tiered")
@click.option(
    "--sources-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("sources"),
    show_default=True,
    help="Directory of <source_id>.txt scholar bodies.",
)
@click.option(
    "--source",
    "sources",
    multiple=True,
    help="Restrict to named source_ids (repeat for multiple). Default: every "
    "*.txt under --sources-dir.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/mining/phase2"),
    show_default=True,
    help="Directory to write <source_id>.jsonl extraction output. Created if missing.",
)
@click.option(
    "--primary-provider",
    type=click.Choice(["anthropic", "gemini", "ollama"]),
    default="ollama",
    show_default=True,
    help="Primary client — runs first on every chunk. Ollama (Qwen on Hades) "
    "is the cost-efficient default; the fallback client only fires on chunks "
    "the primary failed.",
)
@click.option(
    "--primary-model",
    default=None,
    help="Override primary model id (defaults to the provider's default).",
)
@click.option(
    "--fallback-provider",
    type=click.Choice(["anthropic", "gemini", "ollama"]),
    default="anthropic",
    show_default=True,
    help="Fallback client — retries chunks the primary failed. Anthropic is the "
    "default (highest quality on hard chunks).",
)
@click.option(
    "--fallback-model",
    default=None,
    help="Override fallback model id (defaults to the provider's default).",
)
@click.option(
    "--ollama-url",
    default=None,
    help="Override Ollama base URL when either tier is `ollama` (default: "
    "$WYRD_OLLAMA_URL or localhost). Used for both primary and fallback if "
    "either is set to ollama.",
)
@click.option(
    "--chunk-size",
    type=int,
    default=20000,
    show_default=True,
    help="Target chunk size in characters (snaps to paragraph boundaries).",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Process only the first N chunks per source (smoke testing).",
)
@click.option(
    "--skip-existing",
    is_flag=True,
    default=False,
    help="Skip sources whose output JSONL already exists in --output-dir. "
    "Idempotent multi-source resume — sources you've already run stay untouched.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite existing per-source output files (default: error). "
    "Mutually-exclusive with --skip-existing.",
)
@click.option(
    "--hallucination-fallback-threshold",
    type=click.IntRange(min=0),
    default=None,
    help="When set to N >= 1, chunks where the primary SUCCEEDED but emitted "
    "at least N hallucinated forms (per the word-boundary guard) ALSO get "
    "re-extracted by the fallback; mentions are union-merged by "
    "(form, date_year, region_hint, context). Primary wins on byte-identical "
    "collision; fallback adds what primary didn't see. Omitting the flag "
    "or passing 0 disables the rescue (default). Negative values are "
    "rejected by IntRange. Use with `--primary-provider ollama` + small "
    "local model + `--fallback-provider anthropic` to keep gemma4's good "
    "mentions while letting Anthropic catch its fabrications. wyrd-z8mq.",
)
@click.option(
    "--capture-failures",
    "capture_failures",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write a JSONL record for every chunk where BOTH tiers failed "
    "(primary AND fallback). Each record has source_id, chunk_index, "
    "chunk_body, error (chained primary=...; fallback=...). Streamed as "
    "failures happen rather than buffered in-memory — bypasses the "
    "head/tail cap on report.failed_chunks. wyrd-3gbx parity with the "
    "single-tier mine-toponym-mentions failure-streaming.",
)
def lexicon_mine_toponym_mentions_tiered(
    sources_dir: Path,
    sources: tuple[str, ...],
    output_dir: Path,
    primary_provider: str,
    primary_model: str | None,
    fallback_provider: str,
    fallback_model: str | None,
    ollama_url: str | None,
    chunk_size: int,
    limit: int | None,
    skip_existing: bool,
    force: bool,
    hallucination_fallback_threshold: int | None,
    capture_failures: Path | None,
) -> None:
    """Two-tier LLM mention extraction across one or many sources
    (wyrd-x82p Phase 2b.2).

    Runs the primary client first on every chunk; retries chunks that
    failed (transport, JSON parse, schema mismatch) with the fallback
    client. Designed for the wyrd-l0r tiered pattern: Qwen on Hades
    for bulk first-pass (free, fast), Anthropic on the residual (best
    quality on hard chunks).

    Walks every source_id given via --source (or every *.txt under
    --sources-dir when --source is omitted). Writes one JSONL per
    source to --output-dir/<source_id>.jsonl, ready to feed into
    ``lexicon ingest-toponym-mentions``.

    Idempotent resume: --skip-existing lets you re-run after an
    interrupt without re-processing already-extracted sources.
    """
    if skip_existing and force:
        raise click.ClickException("--skip-existing and --force are mutually exclusive")

    output_dir.mkdir(parents=True, exist_ok=True)
    failure_sink = open_failure_sink(capture_failures)
    source_ids = resolve_source_ids(sources, sources_dir)

    click.echo(f"Sources to process: {len(source_ids)}", err=True)
    click.echo(
        f"Primary: {primary_provider} (model={primary_model or 'default'})",
        err=True,
    )
    click.echo(
        f"Fallback: {fallback_provider} (model={fallback_model or 'default'})",
        err=True,
    )

    totals = {
        "sources_processed": 0,
        "sources_skipped": 0,
        "chunks_processed": 0,
        "chunks_recovered": 0,
        "chunks_hallucination_rescue_attempted": 0,
        "chunks_hallucination_rescue_succeeded": 0,
        "chunks_failed": 0,
        "hallucinations_dropped": 0,
        "years_clamped": 0,
        "surrogates_sanitized": 0,
        "mentions": 0,
    }
    ctx = _TieredRun(
        sources_dir=sources_dir,
        output_dir=output_dir,
        primary_provider=primary_provider,
        primary_model=primary_model,
        fallback_provider=fallback_provider,
        fallback_model=fallback_model,
        ollama_url=ollama_url,
        chunk_size=chunk_size,
        limit=limit,
        hallucination_fallback_threshold=hallucination_fallback_threshold,
        skip_existing=skip_existing,
        force=force,
        failure_sink=failure_sink,
        n_sources=len(source_ids),
        totals=totals,
    )

    # try/finally so the --capture-failures sink is closed on every exit
    # path (ClickException for output-exists conflicts raised inside the loop,
    # KeyboardInterrupt mid-iteration, unexpected raises, atomic-write
    # failures). Without it the sink handle leaks on every non-clean exit.
    try:
        for src_i, source_id in enumerate(source_ids, start=1):
            _process_one_source(ctx, src_i, source_id)
    finally:
        if failure_sink is not None:
            failure_sink.close()

    _emit_tiered_total(totals)
    _report_capture_failures_status(capture_failures, ctx.failures_appended)


def _validate_failure_record(rec: dict, path: Path, line_num: int) -> None:
    """Validate one captured-failures record's required fields. Raises
    ClickException on any violation.

    Type-check rationale: ``source_id`` as a list/dict would blow up at
    ``by_source.setdefault(rec["source_id"], …)`` with an unhashable-type
    TypeError far downstream; ``chunk_index`` as a string ("5") would silently
    coerce through ``int()`` while True/False (subclasses of int) would
    silently extract chunk 0 or chunk 1; ``chunk_body`` as a list would
    AttributeError on ``.strip()`` mid-loop. Reject all three loudly at read
    time. R2 silent-failure-hunter MEDIUM.
    """
    for key in ("source_id", "chunk_index", "chunk_body"):
        if key not in rec:
            raise click.ClickException(f"{path}:{line_num}: missing required field {key!r}")
        if rec[key] is None:
            raise click.ClickException(f"{path}:{line_num}: required field {key!r} is null")
    if not isinstance(rec["source_id"], str):
        raise click.ClickException(
            f"{path}:{line_num}: source_id must be a string, got {type(rec['source_id']).__name__}"
        )
    # bool is a subclass of int in Python — reject explicitly so True/False
    # values aren't silently coerced to 1/0.
    if not isinstance(rec["chunk_index"], int) or isinstance(rec["chunk_index"], bool):
        raise click.ClickException(
            f"{path}:{line_num}: chunk_index must be an integer, got "
            f"{type(rec['chunk_index']).__name__}"
        )
    if not isinstance(rec["chunk_body"], str):
        raise click.ClickException(
            f"{path}:{line_num}: chunk_body must be a string, got "
            f"{type(rec['chunk_body']).__name__}"
        )
    if not rec["chunk_body"].strip():
        raise click.ClickException(
            f"{path}:{line_num}: chunk_body is empty — staged-cascade resume "
            f"can't re-extract from no text (was this captured before wyrd-srd2?)"
        )


def _read_failures_jsonl(path: Path) -> list[dict]:
    """Read a captured-failures JSONL into a list of dict records.

    Required fields: source_id (str), chunk_index (int), chunk_body
    (non-empty str). Optional fields: error, extractor. Rejects
    malformed JSON, non-dict rows, missing required fields, null or
    wrong-type required values, and empty ``chunk_body`` (the resume
    path would silently process an empty string as a chunk otherwise —
    FailedChunk.chunk_body defaults to "" for back-compat but
    staged-cascade input must always carry the real chunk text)."""
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_num, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as e:
                raise click.ClickException(f"{path}:{line_num}: not valid JSON: {e}") from e
            if not isinstance(rec, dict):
                raise click.ClickException(
                    f"{path}:{line_num}: expected JSON object, got {type(rec).__name__}"
                )
            _validate_failure_record(rec, path, line_num)
            records.append(rec)
    return records


def _load_existing_mention_keys(
    out_path: Path,
    log_warning: Callable[[str], None] | None = None,
) -> tuple[set[tuple], int]:
    """Build the dedup-key set from an existing per-source output JSONL.
    Returns ``(keys, malformed_count)``. Empty set + 0 if the file
    doesn't exist. The key shape mirrors what the writer emits —
    staged cascade re-runs use this to skip already-emitted
    attestations when appending new mentions.

    Non-dict rows and malformed-JSON rows are skipped (defensive: a
    partially-written file after a crash shouldn't bomb the next
    resume), but the ``malformed_count`` return + optional
    log_warning surface them to the operator so a corrupted file
    doesn't quietly grow duplicate rows. R2 silent-failure-hunter
    MEDIUM."""
    if not out_path.exists():
        return set(), 0
    keys: set[tuple] = set()
    malformed = 0
    with out_path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if not isinstance(row, dict):
                malformed += 1
                continue
            keys.add(dedup_key(row))
    if malformed and log_warning is not None:
        log_warning(
            f"{malformed} malformed row(s) in {out_path} can't be deduped "
            f"against — new mentions matching them will appear as new"
        )
    return keys, malformed


def _make_chunk_callbacks(source_id: str, start_ts: float, emit_failure):
    """Build the (progress, warn, on_chunk_failed) callback trio shared
    by both fresh-mining and resume-from-failures source loops. Hoisted
    out of the loop body so the late-binding default-arg trick that
    captures loop variables can be replaced with real parameter
    bindings — clearer, and forces a single source of truth for the
    log line shape (the project's mining-progress convention from
    CLAUDE.md)."""

    def progress(done: int, total: int, mentions: int) -> None:
        elapsed = time.monotonic() - start_ts
        rate = elapsed / done if done else 0.0
        click.echo(
            f"    chunk [{done}/{total}] mentions={mentions} ({rate:.1f}s/chunk)",
            err=True,
        )

    def warn(msg: str) -> None:
        click.echo(f"    warning: {msg}", err=True)

    def on_fail(fc) -> None:
        emit_failure(source_id, fc)

    return progress, warn, on_fail


def add_to(parent: click.Group) -> None:
    """Register ``mine-toponym-mentions-tiered`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_mine_toponym_mentions_tiered)
