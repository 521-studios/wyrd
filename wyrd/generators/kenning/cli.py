"""CLI commands for Kenning. Mounted under `wyrd kenning <subcommand>`."""

from __future__ import annotations

import json
import sys
from collections import Counter
from importlib import resources
from pathlib import Path

import click

from wyrd.generators.kenning import (
    CULTURES,
    Kenning,
    available_tags,
)
from wyrd.generators.kenning.meaning import load_meanings
from wyrd.generators.kenning.name import load_names
from wyrd.seed import resolve_seed, rng_for


@click.group()
def cli() -> None:
    """Kenning — town name generation and authoring tools."""


@cli.command()
@click.argument("culture", type=click.Choice(CULTURES))
@click.option("--tag", "tags", multiple=True, help="Filter by tag (repeatable).")
@click.option(
    "--count", "-n", type=click.IntRange(1, 10), default=5, help="Generate N names (1–10)."
)
@click.option("--seed", type=int, default=None, help="Reproducible 64-bit seed.")
@click.option(
    "--describe/--no-describe",
    default=True,
    help="Print the morpheme breakdown after each name.",
)
def generate(
    culture: str, tags: tuple[str, ...], count: int, seed: int | None, describe: bool
) -> None:
    """Generate town names. Replaces Rando's `bin/generator`."""
    known_tags = set(available_tags()) | {"male name", "female name", "saint"}
    bad = [t for t in tags if t not in known_tags]
    if bad:
        click.echo(f"Unknown tag(s): {', '.join(bad)}", err=True)
        click.echo("Available tags:", err=True)
        for t in sorted(known_tags - {"male name", "female name", "saint"}):
            click.echo(f"  {t}", err=True)
        sys.exit(1)

    resolved = resolve_seed(seed)
    seed_rng = rng_for(resolved)
    kenning = Kenning()
    params = {"culture": culture, "tags": list(tags)}
    for _ in range(count):
        result = kenning.generate(params, seed_rng.randrange(2**63))
        click.echo(result.result)
        if describe:
            click.echo(result.explanation)
    click.echo(f"(seed: {resolved})", err=True)


@cli.command("rebuild-proportions")
@click.argument("culture", type=click.Choice(CULTURES))
@click.argument("place_names", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--meanings",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to meanings.json (defaults to bundled).",
)
def rebuild_proportions(culture: str, place_names: Path, meanings: Path | None) -> None:
    """Recompute a culture's proportions from a corpus + meanings DB.

    Replaces Rando's `bin/load_names`. Reads place_names JSON (the same shape as
    the bundled `<culture>_place_names.json` files), deconstructs each name
    against the meaning DB, and emits a fresh proportions JSON to stdout.
    """
    if meanings is None:
        meanings_text = (
            resources.files("wyrd.generators.kenning.data").joinpath("meanings.json").read_text()
        )
        meanings_data = json.loads(meanings_text)
    else:
        meanings_data = json.loads(meanings.read_text())

    names_data = json.loads(place_names.read_text())
    names = load_names(names_data)
    word_db, _ = load_meanings(meanings_data)

    perfect = 0
    word_names = 0
    word_saints = 0
    good_names = []
    for name in names:
        name.find_meaning(word_db)
        if name.has_name():
            word_names += 1
        if name.has_saint():
            word_saints += 1
        if name.count_unaccounted() == 0:
            perfect += 1
            good_names.append(name)

    click.echo(
        f"culture={culture} perfect={perfect} names={word_names} saints={word_saints} "
        f"total={len(names)}",
        err=True,
    )
    proportions = _proportions_from(good_names)
    click.echo(json.dumps(proportions))


def _proportions_from(names) -> dict:
    part_proportions: Counter = Counter()
    lone_proportions: Counter = Counter()
    struct_proportions: Counter = Counter()
    for name in names:
        for u in name.get_samples():
            part_proportions[u] += 1
        for u in name.get_lone_samples():
            lone_proportions[u] += 1
        for structure in name.get_structure():
            struct_proportions[structure] += 1
    return {
        "usages": dict(part_proportions),
        "single_usages": dict(lone_proportions),
        "structures": _encode_structs(struct_proportions),
    }


def _encode_meaning(qualities) -> dict:
    out: dict = {}
    for quality in qualities:
        if quality in ("pre", "post", "inner"):
            out["location"] = quality
        else:
            out[quality] = True
    return out


def _encode_structs(struct: Counter) -> list:
    structs = []
    for key, value in struct.items():
        words = [[_encode_meaning(meaning) for meaning in word] for word in key]
        structs.append({"proportion": value, "words": words})
    return structs


@cli.command("add-meaning")
@click.option("--tag", "tags", multiple=True, help="Modifier tag (repeatable).")
def add_meaning(tags: tuple[str, ...]) -> None:
    """Read names from stdin, emit JSON entries suitable for meanings.json.

    Replaces Rando's `bin/generate_names`. One name per line; '/'-separated
    forms become alternate `modern_usage` entries.
    """
    if not tags:
        click.echo("At least one --tag is required.", err=True)
        sys.exit(1)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        forms = line.split("/")
        entry = {
            "meaning": [f"{n} (Name)" for n in forms],
            "modifier_tags": list(tags),
            "modifier_type": "Habitative",
            "words": [{"modern_usage": f"{n}-"} for n in forms],
        }
        click.echo(json.dumps(entry) + ",")


@cli.command("validate-meanings")
@click.argument(
    "meanings",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=False,
)
def validate_meanings(meanings: Path | None) -> None:
    """Check meanings.json: every modern_usage must contain a hyphen.

    Replaces Rando's `bin/validate_names`. Reads the path argument or stdin.
    Exits non-zero if any entries are malformed.
    """
    raw = meanings.read_text() if meanings else sys.stdin.read()
    data = json.loads(raw)

    bad = []
    for subject in data:
        for word in subject["words"]:
            if "-" not in word["modern_usage"]:
                bad.append(word)

    for word in bad:
        click.echo(json.dumps(word), err=True)
    if bad:
        click.echo(f"{len(bad)} malformed entries.", err=True)
        sys.exit(1)
    click.echo(f"OK: {sum(len(s['words']) for s in data)} entries valid.", err=True)
