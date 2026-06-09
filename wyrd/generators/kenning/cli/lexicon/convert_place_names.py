"""``wyrd kenning lexicon convert-place-names`` (wyrd-bo01.2)."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.paths import seed_data_path
from wyrd.generators.kenning.place_names_converter import CULTURES, convert_file


@click.command("convert-place-names")
@click.option(
    "--culture",
    type=click.Choice(CULTURES),
    default=None,
    help="Convert only this culture's gazetteer (default: all five).",
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/mining"),
    show_default=True,
    help="Directory to write {culture}_place_names.jsonl into.",
)
def lexicon_convert_place_names(culture: str | None, out_dir: Path) -> None:
    """Convert the bundled ``{culture}_place_names.json`` gazetteer(s) into L2
    JSONL toponym sources (wyrd-bo01.2).

    Each gazetteer (a ``{country: {region: [names]}}`` dict) becomes one
    ``source`` row (``{culture}_place_names`` — a low-trust seed, the
    place-name analogue of ``rando-port``) plus one ``toponym`` row per place.
    The JSONL is the canonical artifact; ``rebuild-from-jsonl`` restores the
    toponyms from it. Run once and commit the output; Phase 2 then switches
    the proportions builder to read the DB and deletes the static JSON.
    """
    cultures = [culture] if culture else list(CULTURES)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for c in cultures:
        json_path = seed_data_path(f"{c}_place_names.json")
        jsonl_path = out_dir / f"{c}_place_names.jsonl"
        n = convert_file(c, json_path, jsonl_path)
        total += n
        click.echo(f"  {c}: {n} toponyms -> {jsonl_path}", err=True)
    click.echo(f"Done. {total} toponyms across {len(cultures)} culture(s).", err=True)


def add_to(parent: click.Group) -> None:
    parent.add_command(lexicon_convert_place_names)
