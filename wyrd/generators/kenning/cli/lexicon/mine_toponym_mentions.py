"""``wyrd kenning lexicon mine-toponym-mentions`` — extract toponym mentions from a source via LLM."""

from __future__ import annotations

import json
import time
from pathlib import Path

import click


@click.command("mine-toponym-mentions")
@click.option(
    "--source",
    "source_id",
    required=True,
    help="Source id (e.g. mawer_1920_northumberland_durham) to mine.",
)
@click.option(
    "--sources-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("sources"),
    show_default=True,
    help="Directory of <source_id>.txt source bodies.",
)
@click.option(
    "--provider",
    type=click.Choice(["anthropic", "gemini", "ollama"]),
    default="anthropic",
    show_default=True,
    help="LLM backend to use.",
)
@click.option(
    "--model",
    default=None,
    help="Override model id (provider-specific default if omitted).",
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
    help="Process only the first N chunks (smoke testing).",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write extracted mentions to this JSONL file. Default: stdout.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite --output if it exists. Without this flag, an existing path errors.",
)
@click.option(
    "--ollama-url",
    default=None,
    help="Override Ollama base URL when --provider ollama (default: "
    "$WYRD_OLLAMA_URL or localhost). Same shape as the mine-llm flag.",
)
def lexicon_mine_toponym_mentions(
    source_id: str,
    sources_dir: Path,
    provider: str,
    model: str | None,
    chunk_size: int,
    limit: int | None,
    output: Path | None,
    force: bool,
    ollama_url: str | None,
) -> None:
    """LLM-mine a scholar source body for place-name mentions (wyrd-x82p Phase 2).

    Pilot tool: walks one source's .txt body in chunks, calls an LLM to
    extract every place-name mention with optional date_year + region_hint
    + surrounding context, emits results as JSONL (one mention per line)
    to stdout or ``--output``.

    Does NOT touch the DB — the pilot output is for operator review.
    Phase 2b will add DB persistence + a resolver that maps mentions
    to existing toponyms (via the form→toponym lookup from Phase 1)
    OR creates new toponym rows for unresolved mentions.
    """
    from wyrd.generators.kenning.extractors.toponym_mentions import (
        ToponymMention,
        mine_toponym_mentions,
    )

    txt_path = sources_dir / f"{Path(source_id).name}.txt"
    if not txt_path.exists():
        raise click.ClickException(f"source body not found: {txt_path}")
    body = txt_path.read_text(encoding="utf-8", errors="replace")
    if "�" in body:
        # errors="replace" silently mapped invalid bytes to U+FFFD.
        # The form-in-chunk validator will then drop any mention
        # containing those characters, with no visible signal.
        # OE/ME place-name texts genuinely use æ/þ/ð — surface a
        # warning so the operator can fix the encoding upstream.
        click.echo(
            f"warning: {txt_path} contained invalid UTF-8 sequences "
            f"(now U+FFFD); some mentions may be silently rejected",
            err=True,
        )
    click.echo(f"Loaded {len(body):,} chars from {txt_path}", err=True)

    if output is not None and output.exists():
        if not force:
            raise click.ClickException(
                f"--output {output} already exists; pass --force to overwrite"
            )
        # --force was passed; warn the operator so a multi-hour pilot
        # run doesn't get silently obliterated by a typo'd re-run.
        click.echo(
            f"warning: --force overwriting existing {output}",
            err=True,
        )

    # Route through the shared _build_extractor_client helper (same
    # dispatch as mine-toponym-mentions-tiered) so both commands use
    # the same provider/model/ollama-url contract. The helper is
    # defined later in this module — Python looks up names at call
    # time so the forward reference works.
    client = _build_extractor_client(provider, model, ollama_url=ollama_url)

    click.echo(f"Using {provider} model={client.model}", err=True)

    start_ts = time.monotonic()

    def progress(done: int, total: int, mentions: int) -> None:
        # Project convention (CLAUDE.md): "[done/total] mentions=N (Xs/chunk)"
        elapsed = time.monotonic() - start_ts
        rate = elapsed / done if done else 0.0
        click.echo(
            f"  [{done}/{total}] mentions={mentions} ({rate:.1f}s/chunk)",
            err=True,
        )

    def warn(msg: str) -> None:
        click.echo(f"  warning: {msg}", err=True)

    report = mine_toponym_mentions(
        client,
        source_id,
        body,
        target_chunk_size=chunk_size,
        limit=limit,
        on_chunk_done=progress,
        log_warning=warn,
    )

    click.echo("", err=True)
    click.echo(
        f"TOTAL chunks_processed={report.chunks_processed} "
        f"chunks_failed={report.chunks_failed} "
        f"mentions={len(report.mentions)} "
        f"hallucinations_dropped={report.hallucinations_dropped} "
        f"years_clamped={report.years_clamped} "
        f"surrogates_sanitized={report.surrogates_sanitized}",
        err=True,
    )
    if report.failed_chunks:
        click.echo("Failed chunks:", err=True)
        for fc in report.failed_chunks:
            click.echo(f"  chunk {fc.index}: {fc.error}", err=True)
        # When the cap truncates the list, mention how many were elided.
        elided = report.chunks_failed - len(report.failed_chunks)
        if elided > 0:
            click.echo(f"  ... and {elided} more failures (cap reached)", err=True)

    # ensure_ascii=False preserves Old/Middle English diacritics
    # (æ, þ, ð) in the JSONL output — project convention used by the
    # existing diff/export commands. For stdout we use click.echo
    # : it handles terminal-encoding edge cases
    # (Latin-1 PYTHONIOENCODING, stdout=non-UTF-8 device) by re-
    # encoding via UTF-8 or replacement, where a bare sys.stdout.write
    # would raise UnicodeEncodeError. For --output we write the file
    # directly with explicit UTF-8 encoding.
    def _line(m: ToponymMention) -> str:
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

    if output is not None:
        # errors="replace" is a defense-in-depth backstop (wyrd-klod): the
        # in-band sanitizer (wyrd-8wa4 / PR #309) already maps surrogate
        # code points to U+FFFD before they reach here, so this only fires
        # if a future path bypasses that guard. It substitutes ASCII "?"
        # rather than crashing — so a "?" in the output is the operator-
        # visible signal that the in-band guard was bypassed somewhere.
        with output.open("w", encoding="utf-8", errors="replace") as sink:
            for m in report.mentions:
                sink.write(_line(m) + "\n")
        click.echo(f"Wrote {len(report.mentions)} mentions → {output}", err=True)
    else:
        for m in report.mentions:
            click.echo(_line(m))


# 8192-token max-tokens for Anthropic — fits ~100-150 mentions per
# chunk comfortably; the SDK default of 1024 truncates mid-JSON on
# dense chunks. Single source-of-truth used by both single-tier
# (mine-toponym-mentions) and tiered (mine-toponym-mentions-tiered)
# Anthropic client construction.
_ANTHROPIC_MAX_TOKENS_FOR_MENTION_EXTRACTION = 8192


def _build_extractor_client(provider: str, model: str | None, ollama_url: str | None = None):
    """Construct an extractor client for the named provider. Shared
    by both `mine-toponym-mentions` (single-tier) and
    `mine-toponym-mentions-tiered` (primary + fallback) so the
    provider/model/ollama-url contract is identical across the two
    CLI commands.

    ``ollama_url`` is forwarded to OllamaClient when provider=="ollama";
    None falls back to DEFAULT_OLLAMA_URL ($WYRD_OLLAMA_URL or
    localhost). Other providers ignore the kwarg."""
    if provider == "anthropic":
        from wyrd.generators.kenning.extractors.anthropic import (
            DEFAULT_ANTHROPIC_MODEL,
            AnthropicClient,
        )

        return AnthropicClient(
            model=model or DEFAULT_ANTHROPIC_MODEL,
            max_tokens=_ANTHROPIC_MAX_TOKENS_FOR_MENTION_EXTRACTION,
        )
    if provider == "gemini":
        from wyrd.generators.kenning.extractors.gemini import (
            DEFAULT_GEMINI_MODEL,
            GeminiClient,
        )

        return GeminiClient(model=model or DEFAULT_GEMINI_MODEL)
    if provider == "ollama":
        from wyrd.generators.kenning.extractors.llm import (
            DEFAULT_OLLAMA_MODEL,
            DEFAULT_OLLAMA_URL,
            OllamaClient,
        )

        return OllamaClient(
            base_url=ollama_url or DEFAULT_OLLAMA_URL,
            model=model or DEFAULT_OLLAMA_MODEL,
        )
    # Unreachable in practice — click.Choice gates the option. The
    # explicit raise keeps the type-checker happy and surfaces a
    # clear error if the Choice list and this dispatch drift apart.
    raise click.ClickException(f"unknown provider: {provider}")


def add_to(parent: click.Group) -> None:
    """Register ``mine-toponym-mentions`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_mine_toponym_mentions)
