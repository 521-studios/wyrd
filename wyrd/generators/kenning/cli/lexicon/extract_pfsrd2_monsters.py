"""``wyrd kenning lexicon extract-pfsrd2-monsters`` — extract creature names from pfsrd2-data into a JSONL."""

from __future__ import annotations

import json
from pathlib import Path

import click

from wyrd.generators.kenning import pfsrd2_monster_extractor


@click.command("extract-pfsrd2-monsters")
@click.argument(
    "corpus_root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Write JSONL to this file. Default: stdout. The output is "
        "ready to feed into `wyrd kenning lexicon mine-fantasy-name "
        "--batch <output>` once you've reviewed it."
    ),
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Stop after emitting N records (smoke testing).",
)
def lexicon_extract_pfsrd2_monsters(
    corpus_root: Path,
    output_path: Path | None,
    limit: int | None,
) -> None:
    """Extract pfsrd2-data monsters into wyrd-ami JSONL input (wyrd-ld5x).

    Walks `<corpus_root>/monsters/**/*.json` and emits one
    `{"name": ..., "description": ...}` per distinct creature
    morpheme. Family roots (Bugbear) win over per-variant names
    (Bugbear Thug); compound family names (Dragon, Red) are split
    on `,` to get the base. Multi-word creatures with no family are
    skipped — they don't represent a single morpheme cleanly.

    Dedupes by lowercased input_name across the whole walk so each
    morpheme is emitted once even if it appears across many variants.

    Typical pipeline:

        wyrd kenning lexicon extract-pfsrd2-monsters \\
            ~/521Studios/pfsrd2-data \\
            --output /tmp/pfsrd2-monsters.jsonl

        wyrd kenning lexicon mine-fantasy-name \\
            --batch /tmp/pfsrd2-monsters.jsonl --apply
    """
    record_count = 0
    out_handle = output_path.open("w", encoding="utf-8") if output_path is not None else None
    try:
        for record in pfsrd2_monster_extractor.extract_corpus(corpus_root, limit=limit):
            line = json.dumps(record, ensure_ascii=False)
            if out_handle is not None:
                out_handle.write(line + "\n")
            else:
                # click.echo (not sys.stdout.write) so CliRunner captures
                # stdout output reliably; nl=True adds the newline.
                click.echo(line)
            record_count += 1
    finally:
        if out_handle is not None:
            out_handle.close()
    dest = str(output_path) if output_path is not None else "stdout"
    click.echo(
        f"Extracted {record_count} record(s) from {corpus_root} → {dest}",
        err=True,
    )


def add_to(parent: click.Group) -> None:
    """Register ``extract-pfsrd2-monsters`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_extract_pfsrd2_monsters)
