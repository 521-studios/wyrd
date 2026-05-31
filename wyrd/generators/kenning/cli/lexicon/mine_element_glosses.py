"""``wyrd kenning lexicon mine-element-glosses`` — grounded gloss backfill (wyrd-u9k6).

Derive glosses for unglossed generation surfaces by consensus-voting the real
toponym etymologies, and write ``data/mining/_element_glosses.jsonl`` — the
rebuildable L2 record that ``run_full_enrichment`` replays into ``reflex_etymon``.
Deterministic + free (no LLM). The weak/ambiguous tail is handled separately by
``adjudicate-element-glosses``.
"""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("mine-element-glosses")
@click.option(
    "--census",
    "census_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Runtime bundle (export-runtime-db output) holding proportions_usage + "
    "meaning — the source of the unglossed generation surfaces.",
)
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
    help="Lexicon DB supplying the toponym etymologies + authoritative glosses.",
)
@click.option("--culture", default="english", show_default=True)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/mining/_element_glosses.jsonl"),
    show_default=True,
)
def lexicon_mine_element_glosses(
    census_path: Path, db_path: Path, culture: str, output_path: Path
) -> None:
    """Grounded-consensus gloss backfill → ``_element_glosses.jsonl`` (wyrd-u9k6)."""
    from wyrd.generators.kenning.lexicon.element_gloss_backfill import mine_to_jsonl

    rows = mine_to_jsonl(str(census_path), str(db_path), str(output_path), culture)

    confident = sum(1 for r in rows if r.get("is_element"))
    weak = sum(1 for r in rows if not r.get("is_element") and r.get("candidates"))
    nomatch = sum(1 for r in rows if not r.get("candidates"))
    click.echo(
        f"[{len(rows)}/{len(rows)}]  confident={confident} weak={weak} no-match={nomatch} "
        f"-> {output_path}",
        err=True,
    )
    click.echo(
        "Replayed into reflex_etymon by run_full_enrichment's curation slot "
        "(rebuildable). Run `adjudicate-element-glosses` for the weak tail.",
        err=True,
    )


def add_to(parent: click.Group) -> None:
    """Register ``mine-element-glosses`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_mine_element_glosses)
