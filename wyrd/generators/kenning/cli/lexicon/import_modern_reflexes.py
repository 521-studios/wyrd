"""``wyrd kenning lexicon import-modern-reflexes`` — land the curated
modern-English reflexes (wyrd-vewk).

Reads ``data/mining/_modern_reflexes.jsonl`` (curated morpheme -> modern-English
reflex links) and creates/links the modern reflex etymons so the era-grid's
modern stage populates for morphemes that previously had no modern reflex
(``-ham`` -> home, ``holmr`` -> holm/holme, ...). Free + deterministic + a
documented rebuild step: run AFTER ``enrich``/cluster-cognates so each new
reflex inherits its morpheme's cognate cluster. Dry-run by default; ``--apply``
to write. See ``lexicon/modern_reflex_import.py`` for the mechanism.
"""

from __future__ import annotations

import os
from pathlib import Path

import click

from wyrd.generators.kenning.lexicon.db import LexiconDB
from wyrd.generators.kenning.lexicon.modern_reflex_import import import_modern_reflexes


@click.command("import-modern-reflexes")
@click.option(
    "--db",
    "db_path",
    type=click.Path(),
    default=None,
    help="Lexicon DB (default: $WYRD_LEXICON_DB or ~/.wyrd/lexicon.db).",
)
@click.option(
    "--jsonl",
    type=click.Path(),
    default="data/mining/_modern_reflexes.jsonl",
    help="Curated modern-reflex L2 source file.",
)
@click.option("--apply", "do_apply", is_flag=True, help="Write changes (default: dry-run).")
def lexicon_import_modern_reflexes(db_path, jsonl, do_apply):
    if db_path:
        db_file = Path(db_path)
    else:
        env_path = os.environ.get("WYRD_LEXICON_DB")
        db_file = Path(env_path) if env_path else Path.home() / ".wyrd" / "lexicon.db"
    if not db_file.exists():
        raise click.ClickException(f"Lexicon DB not found: {db_file}")
    jsonl_file = Path(jsonl)
    if not jsonl_file.exists():
        raise click.ClickException(f"Curated reflex file not found: {jsonl_file}")

    with LexiconDB(db_file) as db:
        counts = import_modern_reflexes(db, jsonl_file, apply=do_apply)
    mode = "APPLIED" if do_apply else "DRY-RUN (pass --apply to write)"
    click.echo(f"import-modern-reflexes [{mode}] {db_file}", err=True)
    for k in sorted(counts):
        click.echo(f"  {k}: {counts[k]}", err=True)


def add_to(parent: click.Group) -> None:
    """Register ``import-modern-reflexes`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_import_modern_reflexes)
