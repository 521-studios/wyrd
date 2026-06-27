"""``wyrd kenning lexicon mine-toponym-mentions-staged`` — staged toponym-mention mining with resumption (wyrd-stg)."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path

import click

from wyrd.generators.kenning.cli.lexicon._dedup import dedup_key_from_mention
from wyrd.generators.kenning.cli.lexicon._mention_mining_io import (
    open_failure_sink,
    resolve_source_ids,
)
from wyrd.generators.kenning.cli.lexicon.mine_toponym_mentions import _build_extractor_client
from wyrd.generators.kenning.cli.lexicon.mine_toponym_mentions_tiered import (
    _load_existing_mention_keys,
    _make_chunk_callbacks,
    _read_failures_jsonl,
)
from wyrd.generators.kenning.extractors.toponym_mentions import ToponymMention

# wyrd-w7i3: in-progress chunk-log directory under each --output-dir.
# A SIGTERM/crash mid-source leaves <_INPROGRESS_SUBDIR>/<source>.chunks.jsonl
# behind, which the ``recover-inprogress-chunks`` CLI promotes/merges
# into the canonical <source>.jsonl on next operator pass.
_INPROGRESS_SUBDIR = "_inprogress"
_INPROGRESS_SUFFIX = ".chunks.jsonl"


def _inprogress_path(output_dir: Path, source_id: str) -> Path:
    """Per-source in-progress chunk-log path (wyrd-w7i3)."""
    return output_dir / _INPROGRESS_SUBDIR / f"{source_id}{_INPROGRESS_SUFFIX}"


def _check_inprogress_clear(inprogress: Path, output_dir: Path) -> None:
    """Refuse to mine a source whose in-progress log already exists —
    silently overwriting would lose the prior crash's recovery candidate.

    The operator must either run ``recover-inprogress-chunks`` or
    delete the file explicitly. wyrd-w7i3.

    Exception: a 0-byte log carries no recovery value (the crash hit
    before chunk 0 succeeded). Auto-clear it so the operator isn't
    blocked by a stale marker with nothing to recover.
    """
    if not inprogress.exists():
        return
    if inprogress.stat().st_size == 0:
        inprogress.unlink()
        return
    raise click.ClickException(
        f"in-progress chunk log already exists: {inprogress}\n"
        f"This indicates a prior mining run was interrupted mid-source.\n"
        f"Recover the data first:\n"
        f"  wyrd kenning lexicon recover-inprogress-chunks --output-dir {output_dir}\n"
        f"Or, to discard, remove the file manually."
    )


def _make_chunk_mentions_writer(
    sink,
    source_id: str,
    line_fn: Callable[[str, ToponymMention], str],
) -> Callable[[int, list[ToponymMention]], None]:
    """Build the ``on_chunk_mentions`` callback that streams per-chunk
    mentions to the in-progress log and fsyncs them so SIGTERM cannot
    lose the chunk's work (wyrd-w7i3).

    Each chunk's mentions are written in canonical JSONL form (same
    ``line_fn`` used by the final atomic-write path), so the in-progress
    log IS a valid canonical file in the fresh-mining case. The
    chunk_index parameter is currently unused but is part of the
    callback contract — kept so a future resume-mid-source feature
    can persist chunk boundaries without a format break.
    """

    def _write(_chunk_index: int, mentions: list[ToponymMention]) -> None:
        # _chunk_index is reserved for a future resume-mid-source feature
        # that needs chunk-boundary metadata in the log; underscore-prefix
        # signals "intentionally unused" without runtime ceremony.
        for m in mentions:
            sink.write(line_fn(source_id, m) + "\n")
        sink.flush()
        # fsync once per chunk pushes the chunk's mentions out of the
        # kernel page cache. After this returns, a SIGTERM/process-kill
        # cannot lose the chunk's work (file data is durable). Note this
        # does NOT fsync the parent directory, so power-loss durability
        # of a brand-new in-progress file would additionally require a
        # parent-dir fsync — out of scope for the wyrd-w7i3 threat model
        # (process kill, not host crash). Per-mention fsync would be too
        # expensive for the volumes involved (thousands of mentions per
        # source).
        os.fsync(sink.fileno())

    return _write


def _validate_staged_flags(from_failures, sources, skip_existing, force) -> None:
    """Reject incompatible flag combinations. --from-failures (resume mode)
    can't combine with --source or the fresh-mode-only --skip-existing/--force
    (they gate per-source output existence; resume always APPENDS with dedup,
    so silently ignoring them would mislead an operator); --skip-existing and
    --force are themselves mutually exclusive."""
    if from_failures is not None and sources:
        raise click.ClickException("--from-failures and --source are mutually exclusive")
    if from_failures is not None and (skip_existing or force):
        bad = "--skip-existing" if skip_existing else "--force"
        raise click.ClickException(
            f"{bad} is only valid in fresh-mining mode (cannot combine with --from-failures)"
        )
    if skip_existing and force:
        raise click.ClickException("--skip-existing and --force are mutually exclusive")


def _emit_staged_summary(totals: dict, from_failures, capture_failures) -> None:
    """Final TOTAL line (+ resume-mode dedup / malformed-purge counts) and the
    capture-failures status — always surfaced so operators know whether the
    next stage has work or the file is now stale (wyrd-srd2 R1 MEDIUM)."""
    click.echo("", err=True)
    summary = (
        f"TOTAL sources={totals['sources_processed']} "
        f"(skipped={totals['sources_skipped']}) "
        f"chunks={totals['chunks_processed']} "
        f"failed={totals['chunks_failed']} "
        f"halluc={totals['hallucinations_dropped']} "
        f"clamped={totals['years_clamped']} "
        f"surrogates={totals['surrogates_sanitized']} "
        f"mentions={totals['mentions']}"
    )
    if from_failures is not None:
        summary += f" deduped={totals['mentions_deduped']}"
        if totals["malformed_purged"]:
            summary += f" malformed_purged={totals['malformed_purged']}"
    click.echo(summary, err=True)
    if capture_failures is not None:
        if totals["chunks_failed"] > 0:
            click.echo(
                f"  {totals['chunks_failed']} failed chunk(s) appended → {capture_failures}",
                err=True,
            )
        else:
            click.echo(
                f"  0 new failures → {capture_failures} unchanged this run",
                err=True,
            )


@click.command("mine-toponym-mentions-staged")
@click.option(
    "--sources-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("sources"),
    show_default=True,
    help="Directory of <source_id>.txt scholar bodies. Used in fresh-mining "
    "mode (when --from-failures is not given).",
)
@click.option(
    "--source",
    "sources",
    multiple=True,
    help="Restrict to named source_ids (repeat for multiple). Default in "
    "fresh-mining mode: every *.txt under --sources-dir. Ignored in "
    "--from-failures mode.",
)
@click.option(
    "--from-failures",
    "from_failures",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Read pre-chunked (source_id, chunk_index, chunk_body) records from "
    "this JSONL (typically the --capture-failures output of an earlier stage) "
    "and re-process those chunks with the configured provider. Mutually "
    "exclusive with --sources-dir-based fresh mining.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/mining/phase2"),
    show_default=True,
    help="Directory for <source_id>.jsonl extraction output. Created if "
    "missing. In --from-failures mode, mentions are APPENDED to existing "
    "per-source files with deduplication on (form, date_year, region_hint, "
    "context); idempotent re-runs.",
)
@click.option(
    "--capture-failures",
    "capture_failures",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write a JSONL record for every chunk that failed at this stage "
    "(transport, JSON parse, schema mismatch). Each record has source_id, "
    "chunk_index, chunk_body, error, extractor — ready to feed a downstream "
    "stage via --from-failures. Required only to enable a downstream cascade "
    "stage; omit if this is the terminal stage.",
)
@click.option(
    "--provider",
    type=click.Choice(["anthropic", "gemini", "ollama"]),
    default="ollama",
    show_default=True,
    help="LLM backend for this stage.",
)
@click.option(
    "--model",
    default=None,
    help="Override model id (provider-specific default if omitted).",
)
@click.option(
    "--ollama-url",
    default=None,
    help="Override Ollama base URL when --provider ollama (default: "
    "$WYRD_OLLAMA_URL or localhost).",
)
@click.option(
    "--chunk-size",
    type=int,
    default=3000,
    show_default=True,
    help="Target chunk size in characters (snaps to paragraph boundaries). "
    "In --from-failures mode chunks are pre-sliced; the value is then used "
    "only for the oversize-paragraph warning threshold.",
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
    help="In fresh-mining mode, skip sources whose output JSONL already exists in --output-dir.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="In fresh-mining mode, overwrite existing per-source output files "
    "(default: error). Mutually exclusive with --skip-existing.",
)
def lexicon_mine_toponym_mentions_staged(
    sources_dir: Path,
    sources: tuple[str, ...],
    from_failures: Path | None,
    output_dir: Path,
    capture_failures: Path | None,
    provider: str,
    model: str | None,
    ollama_url: str | None,
    chunk_size: int,
    limit: int | None,
    skip_existing: bool,
    force: bool,
) -> None:
    """Stage-at-a-time LLM mention extraction with failure capture (wyrd-srd2).

    Two operating modes share this command:

    1. **Fresh-mining mode** (default): walks every source_id given via
       --source (or every *.txt under --sources-dir when --source is
       omitted). For each source, mines chunks with the configured
       provider+model, writes successful mentions to
       --output-dir/<source_id>.jsonl, captures failed chunks to
       --capture-failures (if given).

    2. **Resume-from-failures mode** (--from-failures FILE): reads
       pre-chunked records from FILE and re-processes those chunks with
       the configured provider+model. Mentions are APPENDED to existing
       per-source output files with deduplication; chunks that fail at
       this stage flow into --capture-failures for the next stage.

    Operator-paced cascade:

        # Stage 1: free local model on everything
        wyrd kenning lexicon mine-toponym-mentions-staged \\
            --provider ollama --model gemma4:26b --chunk-size 3000 \\
            --output-dir data/mining/phase2 \\
            --capture-failures data/mining/phase2_failures/stage1.jsonl

        # Inspect: wc -l data/mining/phase2_failures/stage1.jsonl

        # Stage 2: cheap cloud model on the residual
        wyrd kenning lexicon mine-toponym-mentions-staged \\
            --from-failures data/mining/phase2_failures/stage1.jsonl \\
            --provider gemini --model gemini-2.5-flash \\
            --output-dir data/mining/phase2 \\
            --capture-failures data/mining/phase2_failures/stage2.jsonl

        # Stage 3: more capable model on the still-failed
        wyrd kenning lexicon mine-toponym-mentions-staged \\
            --from-failures data/mining/phase2_failures/stage2.jsonl \\
            --provider anthropic --model claude-haiku-4-5-20251001 \\
            --output-dir data/mining/phase2 \\
            --capture-failures data/mining/phase2_failures/stage3.jsonl

    Output JSONL rows include ``extractor:"provider:model"`` so post-hoc
    analysis can attribute each mention to the stage that captured it.
    """
    # Lazy import: the extractor module pulls in heavy LLM-prompt
    # plumbing; defer until this CLI is actually invoked so the
    # `wyrd --help` discovery path stays cheap. ToponymMention itself
    # is imported at module scope because _make_chunk_mentions_writer
    # needs the type annotation.
    from wyrd.generators.kenning.extractors.toponym_mentions import (
        mine_toponym_mentions,
        mine_toponym_mentions_from_chunks,
    )

    _validate_staged_flags(from_failures, sources, skip_existing, force)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Lazy client construction (matches the tiered command's pattern):
    # a --skip-existing resume where every source is skipped shouldn't
    # require ANTHROPIC_API_KEY validation or an Ollama probe.
    client_box: dict = {"client": None, "extractor": f"{provider}:{model or 'default'}"}

    def _ensure_client():
        if client_box["client"] is None:
            built = _build_extractor_client(provider, model, ollama_url=ollama_url)
            # Refresh extractor name now that we know the resolved
            # model id — JSONL rows pin the actual model that ran,
            # not the placeholder "default".
            client_box["client"] = built
            client_box["extractor"] = f"{provider}:{built.model}"

    # Validate sources before opening the failure sink: in fresh-mining mode a
    # bad --source raises ClickException (inside _run_fresh_mining below), and
    # opening the sink first would leave a stray empty failure file (+ mkdir'd
    # parent) for a run that processed nothing. Mirrors the tiered ordering;
    # _run_fresh_mining re-resolves cheaply for the actual list.
    if from_failures is None:
        resolve_source_ids(sources, sources_dir)
    failure_sink = open_failure_sink(capture_failures)

    def _emit_failure(source_id, fc):
        if failure_sink is None:
            return
        rec = {
            "source_id": source_id,
            "chunk_index": fc.index,
            "chunk_body": fc.chunk_body,
            "error": fc.error,
            "extractor": client_box["extractor"],
        }
        failure_sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
        failure_sink.flush()

    def _line(source_id: str, m: ToponymMention) -> str:
        return json.dumps(
            {
                "source_id": source_id,
                "form": m.form,
                "date_year": m.date_year,
                "region_hint": m.region_hint,
                "context": m.context,
                "extractor": client_box["extractor"],
            },
            ensure_ascii=False,
        )

    totals = {
        "sources_processed": 0,
        "sources_skipped": 0,
        "chunks_processed": 0,
        "chunks_failed": 0,
        "hallucinations_dropped": 0,
        "years_clamped": 0,
        "surrogates_sanitized": 0,
        "mentions": 0,
        "mentions_deduped": 0,
        "malformed_purged": 0,
    }

    try:
        if from_failures is not None:
            _run_resume_from_failures(
                from_failures=from_failures,
                output_dir=output_dir,
                provider=provider,
                model=model,
                chunk_size=chunk_size,
                limit=limit,
                client_box=client_box,
                ensure_client=_ensure_client,
                emit_failure=_emit_failure,
                line_fn=_line,
                totals=totals,
                mine_fn=mine_toponym_mentions_from_chunks,
            )
        else:
            _run_fresh_mining(
                sources=sources,
                sources_dir=sources_dir,
                output_dir=output_dir,
                provider=provider,
                model=model,
                chunk_size=chunk_size,
                limit=limit,
                skip_existing=skip_existing,
                force=force,
                client_box=client_box,
                ensure_client=_ensure_client,
                emit_failure=_emit_failure,
                line_fn=_line,
                totals=totals,
                mine_fn=mine_toponym_mentions,
            )
    finally:
        if failure_sink is not None:
            failure_sink.close()

    _emit_staged_summary(totals, from_failures, capture_failures)


def _merge_canonical_output(out_path, report, existing_keys, line_fn, source_id):
    """Atomic-write append: build the union (existing + new-deduped) in a .tmp
    file, then replace the original — a killed process mid-write leaves the
    original intact, vs an ``open("a")`` that could leave a half-written final
    line downstream commands would stumble on (wyrd-srd2 R1). Preserves well-
    formed existing rows VERBATIM (don't re-serialize: stage-specific fields
    like ``extractor`` must keep their value through the cascade) and DROPS
    malformed rows (non-JSON / non-dict) so corruption doesn't persist every
    resume (R3). Returns ``(new_count, dup_count, purged_count)``."""
    new_count = 0
    dup_count = 0
    purged_count = 0
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open(
        "w", encoding="utf-8", errors="replace"
    ) as sink:  # wyrd-klod surrogate backstop
        if out_path.exists():
            with out_path.open("r", encoding="utf-8") as src:
                for line in src:
                    stripped = line.rstrip("\n")
                    if not stripped:
                        continue
                    try:
                        row = json.loads(stripped)
                    except json.JSONDecodeError:
                        purged_count += 1
                        continue
                    if not isinstance(row, dict):
                        purged_count += 1
                        continue
                    sink.write(stripped + "\n")
        for m in report.mentions:
            key = dedup_key_from_mention(m)
            if key in existing_keys:
                dup_count += 1
                continue
            sink.write(line_fn(source_id, m) + "\n")
            existing_keys.add(key)
            new_count += 1
    tmp_path.replace(out_path)
    return new_count, dup_count, purged_count


def _resume_one_source(
    src_i,
    source_id,
    indexed_chunks,
    n_sources,
    limit,
    output_dir,
    chunk_size,
    client_box,
    ensure_client,
    emit_failure,
    line_fn,
    mine_fn,
    totals,
):
    """Re-mine one source's failed chunks with crash-safe per-chunk
    persistence (wyrd-w7i3): the in-progress log holds this invocation's NEW
    mentions; on crash the canonical file (intact via atomic-write) + the
    in-progress log are merged by recover-inprogress-chunks. Updates totals."""
    if limit is not None:
        indexed_chunks = indexed_chunks[:limit]
    ensure_client()
    out_path = output_dir / f"{source_id}.jsonl"
    start_ts = time.monotonic()
    progress, warn, on_fail = _make_chunk_callbacks(source_id, start_ts, emit_failure)
    existing_keys, malformed = _load_existing_mention_keys(out_path, log_warning=warn)
    click.echo(
        f"[{src_i}/{n_sources}] {source_id}: "
        f"{len(indexed_chunks)} chunk(s) to retry "
        f"({len(existing_keys)} existing mentions"
        + (f", {malformed} malformed" if malformed else "")
        + ")",
        err=True,
    )

    # The try block wraps mine_fn AND the atomic-merge — a crash in the merge
    # (e.g. ENOSPC during the verbatim-copy of existing rows) leaves the
    # canonical file untouched and the in-progress log behind. wyrd-w7i3
    # CRITICAL-1.
    inprogress = _inprogress_path(output_dir, source_id)
    _check_inprogress_clear(inprogress, output_dir)
    inprogress.parent.mkdir(parents=True, exist_ok=True)
    inprogress_sink = inprogress.open(
        "w", encoding="utf-8", errors="replace"
    )  # wyrd-klod surrogate backstop
    new_count = 0
    dup_count = 0
    purged_count = 0
    canonical_completed = False
    try:
        chunk_writer = _make_chunk_mentions_writer(inprogress_sink, source_id, line_fn)
        report = mine_fn(
            client_box["client"],
            source_id,
            indexed_chunks,
            target_chunk_size=chunk_size,
            on_chunk_done=progress,
            on_chunk_failed=on_fail,
            on_chunk_mentions=chunk_writer,
            log_warning=warn,
        )

        # wyrd-st1r: skip the atomic-write when no useful output was produced
        # AND at least one chunk FAILED — otherwise a provider that times out
        # every chunk leaves a 0-byte canonical that --skip-existing treats as
        # "done", silently losing the source. ``not report.mentions`` covers
        # total- AND partial-outage (Gemini R2 P2); ``chunks_failed > 0``
        # distinguishes a real outage from an empty source body (which would
        # otherwise loop forever on "will retry").
        outage_with_no_progress = not report.mentions and report.chunks_failed > 0
        if outage_with_no_progress:
            click.echo(
                f"  → no canonical file written for {source_id}: "
                f"0 mentions captured ({report.chunks_failed} chunks failed); "
                f"--skip-existing will retry on the next run",
                err=True,
            )
            canonical_completed = True  # frees the inprogress unlink
        else:
            new_count, dup_count, purged_count = _merge_canonical_output(
                out_path, report, existing_keys, line_fn, source_id
            )
            canonical_completed = True
    except BaseException:
        # Tell the operator the work is recoverable. The canonical file is
        # still intact (atomic-write contract), and the in-progress log holds
        # the NEW mentions from this resume.
        click.echo(
            f"  → in-progress log preserved at {inprogress}\n"
            f"    canonical file {out_path} is unchanged\n"
            f"    recover with: wyrd kenning lexicon recover-inprogress-chunks "
            f"--output-dir {output_dir}",
            err=True,
        )
        raise
    finally:
        try:
            inprogress_sink.close()
        except Exception as close_err:
            click.echo(
                f"  warning: in-progress log close failed: {close_err}",
                err=True,
            )
    # Only unlink once the canonical merge is durably in place.
    if canonical_completed:
        inprogress.unlink(missing_ok=True)
    if purged_count:
        click.echo(
            f"    purged {purged_count} malformed row(s) from {out_path}",
            err=True,
        )

    click.echo(
        f"  → {out_path} | chunks={report.chunks_processed} "
        f"failed={report.chunks_failed} new_mentions={new_count} "
        f"deduped={dup_count} halluc={report.hallucinations_dropped} "
        f"clamped={report.years_clamped} "
        f"surrogates={report.surrogates_sanitized}",
        err=True,
    )

    totals["sources_processed"] += 1
    totals["chunks_processed"] += report.chunks_processed
    totals["chunks_failed"] += report.chunks_failed
    totals["hallucinations_dropped"] += report.hallucinations_dropped
    totals["years_clamped"] += report.years_clamped
    totals["surrogates_sanitized"] += report.surrogates_sanitized
    totals["mentions"] += new_count
    totals["mentions_deduped"] += dup_count
    totals["malformed_purged"] += purged_count


def _run_resume_from_failures(
    *,
    from_failures: Path,
    output_dir: Path,
    provider: str,
    model: str | None,
    chunk_size: int,
    limit: int | None,
    client_box: dict,
    ensure_client,
    emit_failure,
    line_fn,
    totals: dict,
    mine_fn,
) -> None:
    """Resume-from-failures mode body, extracted out of the click command
    so the dispatcher stays narrow. mine_fn is injected for testability
    (matches mine_toponym_mentions_from_chunks at the only call site)."""
    records = _read_failures_jsonl(from_failures)
    if not records:
        # Fall through to summary (don't early-return): operator wants
        # the unambiguous "sources=0 mentions=0" line even on a no-op.
        click.echo(
            f"--from-failures {from_failures} contained no records; nothing to do",
            err=True,
        )
        return
    by_source: dict[str, list[tuple[int, str]]] = {}
    for rec in records:
        by_source.setdefault(rec["source_id"], []).append(
            (int(rec["chunk_index"]), rec["chunk_body"])
        )
    # Sort each source's chunks by original index so progress lines
    # look chronological. Sort sources alphabetically for determinism
    # across resume invocations.
    for src in by_source:
        by_source[src].sort(key=lambda t: t[0])
    click.echo(
        f"Resume from {from_failures}: {len(records)} chunks across {len(by_source)} source(s)",
        err=True,
    )
    click.echo(f"Provider: {provider} (model={model or 'default'})", err=True)

    for src_i, source_id in enumerate(sorted(by_source), start=1):
        _resume_one_source(
            src_i,
            source_id,
            by_source[source_id],
            len(by_source),
            limit,
            output_dir,
            chunk_size,
            client_box,
            ensure_client,
            emit_failure,
            line_fn,
            mine_fn,
            totals,
        )


def _run_fresh_mining(
    *,
    sources: tuple[str, ...],
    sources_dir: Path,
    output_dir: Path,
    provider: str,
    model: str | None,
    chunk_size: int,
    limit: int | None,
    skip_existing: bool,
    force: bool,
    client_box: dict,
    ensure_client,
    emit_failure,
    line_fn,
    totals: dict,
    mine_fn,
) -> None:
    """Fresh-mining mode body, extracted out of the click command. mine_fn
    is injected (matches mine_toponym_mentions at the only call site)."""
    source_ids = resolve_source_ids(sources, sources_dir)

    click.echo(f"Sources to process: {len(source_ids)}", err=True)
    click.echo(f"Provider: {provider} (model={model or 'default'})", err=True)

    for src_i, source_id in enumerate(source_ids, start=1):
        out_path = output_dir / f"{source_id}.jsonl"
        if _should_skip_existing_output(
            out_path,
            source_id,
            src_i,
            len(source_ids),
            skip_existing=skip_existing,
            force=force,
            totals=totals,
        ):
            continue

        txt_path = sources_dir / f"{source_id}.txt"
        body = txt_path.read_text(encoding="utf-8", errors="replace")
        if "�" in body:
            click.echo(
                f"  warning: {source_id} contained invalid UTF-8 sequences (now U+FFFD)",
                err=True,
            )

        click.echo(
            f"[{src_i}/{len(source_ids)}] {source_id}: {len(body):,} chars",
            err=True,
        )
        ensure_client()
        start_ts = time.monotonic()
        progress, warn, on_fail = _make_chunk_callbacks(source_id, start_ts, emit_failure)

        # wyrd-w7i3: crash-safe per-chunk persistence. Open the
        # in-progress log BEFORE invoking mine_fn so each successful
        # chunk's mentions are durably on disk before the next chunk
        # starts. SIGTERM kills the process between chunks → the log
        # survives → recover-inprogress-chunks promotes it.
        inprogress = _inprogress_path(output_dir, source_id)
        _check_inprogress_clear(inprogress, output_dir)
        inprogress.parent.mkdir(parents=True, exist_ok=True)
        inprogress_sink = inprogress.open(
            "w", encoding="utf-8", errors="replace"
        )  # wyrd-klod surrogate backstop
        canonical_completed = False
        try:
            chunk_writer = _make_chunk_mentions_writer(inprogress_sink, source_id, line_fn)
            report = mine_fn(
                client_box["client"],
                source_id,
                body,
                target_chunk_size=chunk_size,
                limit=limit,
                on_chunk_done=progress,
                on_chunk_failed=on_fail,
                on_chunk_mentions=chunk_writer,
                log_warning=warn,
            )

            # wyrd-st1r: skip the atomic-write entirely when the
            # source produced ZERO usable output BECAUSE OF A REAL
            # OUTAGE. Mirrors the resume-path guard above; see that
            # location for the full rationale and predicate
            # breakdown. The fresh path has the same shape: ``--force``
            # can leave a pre-existing canonical at out_path, and
            # skipping the atomic-write preserves it intact rather
            # than destroying good data with a 0-byte write.
            outage_with_no_progress = not report.mentions and report.chunks_failed > 0
            if outage_with_no_progress:
                click.echo(
                    f"  → no canonical file written for {source_id}: "
                    f"0 mentions captured ({report.chunks_failed} chunks failed); "
                    f"--skip-existing will retry on the next run",
                    err=True,
                )
                canonical_completed = True  # frees the inprogress unlink
            else:
                # Atomic write: fresh per-source files use full rewrite.
                tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
                with tmp_path.open(
                    "w", encoding="utf-8", errors="replace"
                ) as sink:  # wyrd-klod surrogate backstop
                    for m in report.mentions:
                        sink.write(line_fn(source_id, m) + "\n")
                tmp_path.replace(out_path)
                canonical_completed = True
        except BaseException:
            # Tell the operator the work is recoverable BEFORE the
            # exception propagates — otherwise a SIGINT'd run looks
            # like data loss to anyone who doesn't know the recovery
            # CLI exists. (wyrd-w7i3 HIGH-3.)
            click.echo(
                f"  → in-progress log preserved at {inprogress}\n"
                f"    recover with: wyrd kenning lexicon recover-inprogress-chunks "
                f"--output-dir {output_dir}",
                err=True,
            )
            raise
        finally:
            # Suppress close-time errors so they can't mask the original
            # cause (e.g. flush hitting ENOSPC would otherwise replace an
            # LLM-side failure). wyrd-w7i3 HIGH-4.
            try:
                inprogress_sink.close()
            except Exception as close_err:
                click.echo(
                    f"  warning: in-progress log close failed: {close_err}",
                    err=True,
                )
        # Only unlink once the canonical file is durably in place. On
        # any exception above, canonical_completed stays False and the
        # in-progress log survives as the recovery candidate.
        if canonical_completed:
            inprogress.unlink(missing_ok=True)

        click.echo(
            f"  → {out_path} | chunks={report.chunks_processed} "
            f"failed={report.chunks_failed} mentions={len(report.mentions)} "
            f"halluc={report.hallucinations_dropped} "
            f"clamped={report.years_clamped} "
            f"surrogates={report.surrogates_sanitized}",
            err=True,
        )

        totals["sources_processed"] += 1
        totals["chunks_processed"] += report.chunks_processed
        totals["chunks_failed"] += report.chunks_failed
        totals["hallucinations_dropped"] += report.hallucinations_dropped
        totals["years_clamped"] += report.years_clamped
        totals["surrogates_sanitized"] += report.surrogates_sanitized
        totals["mentions"] += len(report.mentions)


def _should_skip_existing_output(
    out_path: Path,
    source_id: str,
    src_i: int,
    total: int,
    *,
    skip_existing: bool,
    force: bool,
    totals: dict,
) -> bool:
    """Decide whether the per-source mining loop should skip this
    source's already-existing canonical output.

    Three outcomes encoded as bool-return + side-effect-or-raise:
      - ``False`` (proceed): no canonical exists OR ``force`` is set →
        caller mines and overwrites.
      - ``True`` (skip): canonical exists AND ``skip_existing`` is set →
        emits the SKIP progress line and bumps
        ``totals['sources_skipped']`` for the run summary.
      - ``raise ClickException``: canonical exists AND neither flag is
        set → operator must pick one explicitly.
    """
    if not out_path.exists():
        return False
    if skip_existing:
        click.echo(
            f"[{src_i}/{total}] {source_id}: SKIP (output exists)",
            err=True,
        )
        totals["sources_skipped"] += 1
        return True
    if not force:
        raise click.ClickException(
            f"output {out_path} exists; pass --skip-existing to leave it or --force to overwrite"
        )
    return False


def add_to(parent: click.Group) -> None:
    """Register ``mine-toponym-mentions-staged`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_mine_toponym_mentions_staged)
