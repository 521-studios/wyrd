"""``wyrd kenning lexicon mine-tags-llm`` — LLM tag backfill (wyrd-xz3g).

Classifies the gloss of every generation-reachable etymon that has a gloss but
NO semantic tag into the controlled tag vocabulary (44 existing + ``kinship``),
using a local LLM (gemma4:26b on the macbook Ollama). A mandatory "none"
outcome leaves genuinely-abstract glosses blank — operator constraint: NO bad
matches, NO tag sprawl.

PAID mining → writes the L2 record ``data/mining/_tags.jsonl`` (one decision
per line, including the 'none' outcomes). It does NOT touch the lexicon DB; the
``apply-tag-additions`` enrichment pass (run by ``lexicon enrich`` /
``rebuild-from-jsonl --with-enrichment``) replays the jsonl into ``etymon_tag``
for free, so a from-scratch rebuild never re-pays for the LLM (D21 / REBUILD.md).

Resumable: ``--skip-resolved`` (default) skips refs already in the output, so a
crash mid-run loses nothing — each decision is flushed as it is made.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.extractors.llm import OllamaClient
from wyrd.generators.kenning.lexicon import tag_mining
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY

_DEFAULT_OUTPUT = Path("data/mining/_tags.jsonl")
_DEFAULT_MODEL = "gemma4:26b"
# Abort the run after this many back-to-back API failures — a real outage
# (Ollama down / wrong URL), not a single flaky call. A successful call resets
# the counter.
_MAX_CONSECUTIVE_FAILS = 10


def _select_todo(
    db_path: Path, output: Path, *, skip_resolved: bool, limit: int | None
) -> tuple[list[tuple[str, str, str]], int, int]:
    """The untagged-with-gloss etymons still needing classification, plus the raw
    target / already-resolved counts for the run's plan line.

    ``skip_resolved`` drops ``lang:form`` refs already recorded in ``output``
    (resume without re-paying); ``limit`` caps the list (smoke runs). Pure read —
    no LLM, no writes. Returns ``(todo, n_targets, n_resolved)``.
    """
    targets = tag_mining.select_targets(str(db_path))
    resolved = tag_mining.existing_refs(str(output)) if skip_resolved else set()
    todo = [
        (lang, form, gloss) for (lang, form, gloss) in targets if f"{lang}:{form}" not in resolved
    ]
    if limit is not None:
        todo = todo[:limit]
    return todo, len(targets), len(resolved)


@click.command("mine-tags-llm")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=_DEFAULT_OUTPUT,
    show_default=True,
    help="L2 jsonl to append tag decisions to.",
)
@click.option("--model", default=_DEFAULT_MODEL, show_default=True, help="Ollama model id.")
@click.option(
    "--ollama-url",
    default=None,
    help="Override Ollama base URL (default: $WYRD_OLLAMA_URL or localhost).",
)
@click.option("--limit", type=int, default=None, help="Process only the first N targets (smoke).")
@click.option(
    "--skip-resolved/--no-skip-resolved",
    default=True,
    show_default=True,
    help="Skip etymons already recorded in --output (resume without re-paying).",
)
def lexicon_mine_tags_llm(
    db_path: Path,
    output: Path,
    model: str,
    ollama_url: str | None,
    limit: int | None,
    skip_resolved: bool,
) -> None:
    """Classify untagged-but-glossed etymons into the controlled tag vocab."""
    todo, n_targets, n_resolved = _select_todo(
        db_path, output, skip_resolved=skip_resolved, limit=limit
    )
    click.echo(
        f"mine-tags-llm: {n_targets} untagged-with-gloss targets, "
        f"{n_resolved} already resolved, {len(todo)} to classify "
        f"→ {output} ({model})",
        err=True,
    )
    if not todo:
        click.echo("Nothing to do.", err=True)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists() or output.stat().st_size == 0:
        # First line of every _tags.jsonl is the canonical L2 `source` row so
        # rebuild-from-jsonl's per-file source contract is met (wyrd-xz3g).
        with open(output, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(tag_mining.source_row(), ensure_ascii=False, sort_keys=True) + "\n")
    client = (
        OllamaClient(model=model)
        if ollama_url is None
        else OllamaClient(base_url=ollama_url, model=model)
    )

    tagged = none = err = api_fail = 0
    consecutive = 0
    aborted = False
    t0 = time.time()
    with open(output, "a", encoding="utf-8") as fh:
        for i, (lang, form, gloss) in enumerate(todo, 1):
            system, user = tag_mining.build_messages(gloss or "")
            try:
                raw = client.chat_json(system, user, {})
            except Exception as e:  # noqa: BLE001 — API/network failure
                # Do NOT write a record: an error row carries a `ref` and would
                # be treated as resolved by --skip-resolved next run, so a
                # transient/systematic outage (Ollama down, wrong --ollama-url,
                # model not loaded) would PERMANENTLY skip every remaining
                # etymon. Leaving it unwritten keeps it retryable. Abort after
                # _MAX_CONSECUTIVE_FAILS in a row (a real outage, not one bad row).
                click.echo(f"  [{i}] {lang}:{form} API call failed: {e}", err=True)
                api_fail += 1
                consecutive += 1
                if consecutive >= _MAX_CONSECUTIVE_FAILS:
                    click.echo(
                        f"Aborting after {consecutive} consecutive API failures "
                        "— is Ollama up at the configured URL?",
                        err=True,
                    )
                    aborted = True
                    break
                continue
            consecutive = 0
            # parse_response returns None only when the call SUCCEEDED but the
            # model emitted unparseable JSON — a genuine per-item failure, so we
            # DO record it (an error row, counted, resolved) and move on.
            rec = tag_mining.record(lang, form, tag_mining.parse_response(raw), model)
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
            fh.flush()
            if "error" in rec:
                err += 1
            elif rec.get("tags"):
                tagged += 1
            else:
                none += 1
            if i % 10 == 0 or i == len(todo):
                rate = (time.time() - t0) / i
                click.echo(
                    f"  [{i}/{len(todo)}]  tagged={tagged} none={none} "
                    f"err={err} api_fail={api_fail} ({rate:.2f}s/entry)",
                    err=True,
                )

    click.echo(
        f"{'Aborted' if aborted else 'Done'}. tagged={tagged} none={none} "
        f"err={err} api_fail={api_fail} (not recorded — retryable) → {output}",
        err=True,
    )


def add_to(parent: click.Group) -> None:
    """Register ``mine-tags-llm`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_mine_tags_llm)
