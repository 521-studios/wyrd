"""``wyrd kenning rewind`` — render a name at multiple historical era stops."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from wyrd.generators.kenning import _load_meanings
from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.era.cells import era_cell_for_input
from wyrd.generators.kenning.era.rewind import (
    DEFAULT_ENGLISH_ERAS,
    rewind_from_morphemes,
    rewind_name,
)
from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


def _load_json_objects(from_json: Path) -> list[dict]:
    """wyrd-cp2d: parse newline-delimited JSON from FILE (or stdin
    via '-'). Returns one dict per non-blank line.

    Extracted from ``rewind`` to keep the CLI body under the C901
    cyclomatic-complexity floor. Loud-failure on malformed JSON or
    empty input (matching the existing pre-validation pattern for
    bad --era / --culture values)."""
    if str(from_json) == "-":
        raw = sys.stdin.read()
    else:
        raw = from_json.read_text(encoding="utf-8")
    objects: list[dict] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            objects.append(json.loads(line))
        except json.JSONDecodeError as exc:
            click.echo(f"Error: malformed JSON line: {exc}", err=True)
            raise click.exceptions.Exit(1) from exc
    if not objects:
        click.echo("Error: --from-json input had no JSON objects", err=True)
        raise click.exceptions.Exit(1)
    return objects


@click.command("rewind")
@click.argument("name", required=False)
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--era",
    "era_input",
    multiple=True,
    help=(
        "Era stop to render. Pass multiple times for a custom ladder; "
        "default is 'oe-late me modern' for the english family. "
        "Each value is a cell label (oe-late) or family/label "
        "(english/me). Year input not supported here — use cell labels "
        "for predictable rewind output."
    ),
)
@click.option(
    "--from-json",
    "from_json",
    type=click.Path(dir_okay=False, allow_dash=True, path_type=Path),
    default=None,
    help=(
        "wyrd-cp2d: read a 'kenning generate --json' object (one per "
        "line) from FILE (or '-' for stdin) instead of re-decomposing "
        "the NAME argument. Bypasses the trie matcher and uses the "
        "generator's actual morpheme picks — eliminates the 'Cranwith → "
        "Cornlandian' situation where re-decomposition lands on a "
        "different etymological cluster. NAME argument is optional "
        "when --from-json is set; if both are present, the JSON wins."
    ),
)
def rewind(
    name: str | None,
    db_path: Path,
    era_input: tuple[str, ...],
    from_json: Path | None,
) -> None:
    """wyrd-rni Phase 3.1: render NAME at multiple historical eras.

    Decomposes the input via the kenning explainer, anchors each
    morpheme to its source-language etymon (typically Old English),
    and renders the compound at each requested era stop using the
    cognate-cluster era-reflex picker (wyrd-skm Phase 3.0b).

    Default ladder is three English-family stops: oe-late (800-1100),
    me (1100-1500), modern (1700+). Pass ``--era`` repeatedly to
    customise — e.g. ``--era oe-early --era oe-late --era me`` for a
    five-cell pre-modern ladder.

    Morphemes that don't anchor (no matching etymon row in the
    lexicon DB, or anchor lacks both cognate_id and descent edges)
    fall back to their modern usage and are flagged so the user can
    see which morphemes 'don't have era data' for the requested
    cell.
    """
    eras: tuple[tuple[str, str], ...]
    if era_input:
        try:
            eras = tuple(era_cell_for_input(value, default_family="english") for value in era_input)
        except ValueError as exc:
            click.echo(str(exc), err=True)
            raise click.exceptions.Exit(1) from exc
    else:
        eras = DEFAULT_ENGLISH_ERAS

    meaning_db, _ = _load_meanings()

    # wyrd-cp2d: --from-json path. Parse JSON object(s); each line
    # is one object so operators can pipe `kenning generate --json
    # | kenning rewind --from-json -` for the continuity flow.
    json_objects = _load_json_objects(from_json) if from_json is not None else []
    if not json_objects and not name:
        click.echo("Error: NAME argument is required when --from-json is not set", err=True)
        raise click.exceptions.Exit(1)

    try:
        with LexiconDB(db_path) as db:
            if json_objects:
                results = [
                    rewind_from_morphemes(
                        obj.get("name", ""), obj.get("words", []), db, meaning_db, eras=eras
                    )
                    for obj in json_objects
                ]
            else:
                results = [rewind_name(name or "", db, meaning_db, eras=eras)]
    except ValueError as exc:
        click.echo(f"Error: {exc}", err=True)
        raise click.exceptions.Exit(1) from exc

    for i, result in enumerate(results):
        if i > 0:
            click.echo("")  # blank line between multiple rewinds
        _emit_rewind(result)


def _emit_rewind(result) -> None:
    """Print one Rewind result in the CLI's standard table shape."""
    if not result.eras:
        click.echo("(no era stops requested)", err=True)
        return

    click.echo(f"{result.name} — decomposed: {result.decomposition}")
    if result.unaccounted:
        click.echo(f"  unaccounted: {' '.join(result.unaccounted)}", err=True)
    # Bail early when nothing decomposed — printing empty era rows
    # ('english/oe-late  ') with no rendered content clutters the
    # output and confuses readers about what failed.
    if not any(stop.morphemes for stop in result.eras):
        return

    width = max(len(f"{stop.family}/{stop.cell}") for stop in result.eras)
    for stop in result.eras:
        label = f"{stop.family}/{stop.cell}".ljust(width)
        flag = " *" if any(m.fallback for m in stop.morphemes) else ""
        click.echo(f"  {label}  {stop.rendered}{flag}")
    if any(any(m.fallback for m in stop.morphemes) for stop in result.eras):
        click.echo(
            "  * one or more morphemes lacked era data; rendered with "
            "the modern usage as fallback.",
            err=True,
        )


def add_to(parent: click.Group) -> None:
    """Register ``rewind`` on the top-level ``@cli`` group."""
    parent.add_command(rewind)
