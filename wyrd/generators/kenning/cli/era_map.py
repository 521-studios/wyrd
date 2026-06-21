"""``wyrd kenning era-map`` — bulk-roll N names + render each at era stops."""

from __future__ import annotations

import json
from pathlib import Path

import click

from wyrd.generators.kenning import CULTURES, KenningEraMap


@click.command("era-map")
@click.argument("culture", type=click.Choice(CULTURES), required=False)
@click.option("--count", type=int, default=6, help="How many names to roll.")
@click.option("--seed", type=int, default=0, help="RNG seed; successive names use seed+i.")
@click.option("--as-json", is_flag=True, default=False, help="Emit JSON.")
@click.option(
    "--era",
    "era_input",
    multiple=True,
    help=(
        "Era stop to render (cell label like 'oe-late' or family/label like "
        "'english/me'); repeat for a custom ladder. Default: oe-late me modern. "
        "Only applies to the --from-json path (the generated path uses "
        "KenningEraMap's own ladder)."
    ),
)
@click.option(
    "--from-json",
    "from_json",
    type=click.Path(dir_okay=False, allow_dash=True, path_type=Path),
    default=None,
    help=(
        "wyrd-hyeb: render pre-generated names instead of rolling fresh ones. "
        "Reads 'kenning generate --json' object(s) (one per line) from FILE (or "
        "'-' for stdin), and era-renders each via the generator's actual morpheme "
        "picks (mirrors 'kenning rewind --from-json' but in era-map's table "
        "shape). CULTURE is optional when this is set."
    ),
)
@click.option(
    "--db",
    "db_path",
    # NOT exists=True: the default-generated path must not break the (DB-free)
    # generate path in CI where no live lexicon DB exists. The --from-json path
    # opens it explicitly and fails loud if it's missing.
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Lexicon DB for the --from-json era-reflex lookups (defaults to the standard path).",
)
def era_map(
    culture: str | None,
    count: int,
    seed: int,
    as_json: bool,
    era_input: tuple[str, ...],
    from_json: Path | None,
    db_path: Path | None,
) -> None:
    """wyrd-381: bulk-generate N names AND render each at era stops.

    The Domesday-vs-modern map demo. Composes the existing Kenning
    name-generator with the wyrd-skm era-reflex rewinder: same roll,
    same morpheme stack, multiple eras of paper.

    Default ladder is the three English stops (oe-late / me / modern).
    Names whose morphemes haven't been mined for era_reflexes render
    uniformly across all eras (the bundle-only renderer falls back
    to the modern usage); that's a coverage limit, not a bug.

    With ``--from-json`` (wyrd-hyeb), feed pre-generated ``kenning generate
    --json`` output (one object per line, or '-' for stdin) instead of rolling
    fresh names — useful for era-rendering a fixed set for comparison.
    """
    if from_json is not None:
        _emit_era_map_from_json(from_json, era_input, db_path, as_json)
        return

    if culture is None:
        click.echo("Error: CULTURE argument is required when --from-json is not set", err=True)
        raise click.exceptions.Exit(1)

    gen = KenningEraMap()
    results = gen.generate_all({"culture": culture, "count": count}, seed)

    if as_json:
        out = []
        for r in results:
            comp = r.components[0] if r.components else {}
            out.append(
                {
                    "name": comp.get("name"),
                    "era_cells": comp.get("era_cells", []),
                    "rendered_modern": r.result,
                }
            )
        click.echo(json.dumps(out, indent=2))
        return

    click.echo(f"culture={culture} count={len(results)} (requested {count})", err=True)
    for r in results:
        comp = r.components[0] if r.components else {}
        name = comp.get("name", "?")
        cells = comp.get("era_cells", [])
        click.echo(f"\n{name}")
        for cell in cells:
            label = f"{cell.get('family', '?')}/{cell.get('era', '?')}"
            click.echo(f"  {label:<20} {cell.get('rendered', '?')}")


def _emit_era_map_from_json(
    from_json: Path, era_input: tuple[str, ...], db_path: Path | None, as_json: bool
) -> None:
    """wyrd-hyeb: era-render pre-generated ``generate --json`` object(s) in
    era-map's table shape, reusing rewind's JSON parsing + the era-reflex
    rewinder so the generator's committed morpheme picks (not a trie
    re-decomposition) drive the rendering."""
    # Lazy imports: keep the DB/era stack off the (DB-free) generate path's
    # import cost and avoid a top-level cycle through the kenning package.
    from wyrd.generators.kenning import _load_meanings
    from wyrd.generators.kenning.cli.rewind import _load_json_objects
    from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
    from wyrd.generators.kenning.era.cells import era_cell_for_input
    from wyrd.generators.kenning.era.rewind import (
        DEFAULT_ENGLISH_ERAS,
        rewind_from_morphemes,
    )
    from wyrd.generators.kenning.lexicon import LexiconDB

    if era_input:
        try:
            eras = tuple(era_cell_for_input(value, default_family="english") for value in era_input)
        except ValueError as exc:
            click.echo(str(exc), err=True)
            raise click.exceptions.Exit(1) from exc
    else:
        eras = DEFAULT_ENGLISH_ERAS

    objects = _load_json_objects(from_json)
    meaning_db, _ = _load_meanings()
    resolved_db = db_path if db_path is not None else _DEFAULT_LEXICON_PATH

    try:
        with LexiconDB(resolved_db) as db:
            results = [
                rewind_from_morphemes(
                    obj.get("name", ""), obj.get("words", []), db, meaning_db, eras=eras
                )
                for obj in objects
            ]
    except ValueError as exc:
        click.echo(f"Error: {exc}", err=True)
        raise click.exceptions.Exit(1) from exc

    if as_json:
        out = [
            {
                "name": r.name,
                "era_cells": [
                    {"family": s.family, "era": s.cell, "rendered": s.rendered} for s in r.eras
                ],
            }
            for r in results
        ]
        click.echo(json.dumps(out, indent=2))
        return

    click.echo(f"count={len(results)} (from {from_json})", err=True)
    for r in results:
        click.echo(f"\n{r.name}")
        for stop in r.eras:
            label = f"{stop.family}/{stop.cell}"
            flag = " *" if any(m.fallback for m in stop.morphemes) else ""
            click.echo(f"  {label:<20} {stop.rendered}{flag}")


def add_to(parent: click.Group) -> None:
    """Register ``era-map`` on the top-level ``@cli`` group."""
    parent.add_command(era_map)
