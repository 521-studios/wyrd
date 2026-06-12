"""``wyrd kenning lexicon mine-tags-llm`` — LLM tag backfill (wyrd-xz3g).

Classifies the gloss of every generation-reachable etymon that has a gloss but
NO semantic tag into the controlled tag vocabulary (45 existing + ``kinship``),
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
    targets = tag_mining.select_targets(str(db_path))
    resolved = tag_mining.existing_refs(str(output)) if skip_resolved else set()
    todo = [
        (lang, form, gloss)
        for (lang, form, gloss) in targets
        if f"{lang}:{form}" not in resolved
    ]
    if limit is not None:
        todo = todo[:limit]

    click.echo(
        f"mine-tags-llm: {len(targets)} untagged-with-gloss targets, "
        f"{len(resolved)} already resolved, {len(todo)} to classify "
        f"→ {output} ({model})",
        err=True,
    )
    if not todo:
        click.echo("Nothing to do.", err=True)
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    client = OllamaClient(model=model) if ollama_url is None else OllamaClient(
        base_url=ollama_url, model=model
    )

    tagged = none = err = 0
    t0 = time.time()
    with open(output, "a", encoding="utf-8") as fh:
        for i, (lang, form, gloss) in enumerate(todo, 1):
            system, user = tag_mining.build_messages(gloss or "")
            try:
                raw = client.chat_json(system, user, {})
                parsed = tag_mining.parse_response(raw)
            except Exception:  # noqa: BLE001 — one bad call shouldn't kill the run
                parsed = None
            rec = tag_mining.record(lang, form, parsed, model)
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
                    f"err={err} ({rate:.2f}s/entry)",
                    err=True,
                )

    click.echo(
        f"Done. tagged={tagged} none={none} err={err} → {output}", err=True
    )


def add_to(parent: click.Group) -> None:
    """Register ``mine-tags-llm`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_mine_tags_llm)
