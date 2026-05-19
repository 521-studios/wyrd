"""``wyrd kenning generate`` — produce town names from a culture's morpheme proportions."""

from __future__ import annotations

import sys

import click

from wyrd.generators.kenning import (
    _INTERNAL_TAGS,
    CULTURES,
    MOODS,
    Kenning,
    available_tags,
)
from wyrd.seed import resolve_seed, rng_for


@click.command()
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
@click.option(
    "--spelling-variety",
    type=click.FloatRange(0.0, 1.0),
    default=0.0,
    show_default=True,
    help=(
        "D18 spelling-variant substitution probability (0..1). At >0, each morpheme "
        "is rolled for replacement with an attested archaic spelling drawn from "
        "the etymon's variant pool."
    ),
)
@click.option(
    "--novelty",
    type=click.FloatRange(0.0, 1.0),
    default=0.0,
    show_default=True,
    help=(
        "D17 mixture knob (0..1). 0 = pure empirical-frequency sampling (today's "
        "behavior); 1 = uniform marginal over in-bucket morphemes; intermediate "
        "values blend, allowing plausible-but-unattested combinations."
    ),
)
@click.option(
    "--inflection-density",
    type=click.FloatRange(0.0, 1.0),
    default=0.0,
    show_default=True,
    help=(
        "D8 inflection knob (0..1). Per-morpheme probability of substituting an "
        "inflected form (cotum/cotan/cotes) for the lemma (cot)."
    ),
)
@click.option(
    "--mood",
    "moods",
    multiple=True,
    help=(
        "D6 stylistic-mood preset (repeatable). 'grim' applies a menacing "
        "semantic-tag union; 'harsh' biases sampling toward stop-final / "
        "cluster-heavy morphemes; 'harsh:VALUE' graduates the phonological "
        "skew (e.g. 'harsh:0.5'). Multiple --mood flags compose."
    ),
)
@click.option(
    "--include-fiction",
    "include_fiction",
    is_flag=True,
    default=False,
    help=(
        "wyrd-yan: allow morphemes tagged 'fiction' (constructed etymologies "
        "for bestiary / NPC / homebrew content) to appear. Off by default — "
        "realistic mode draws only from scholarly-attested morphemes."
    ),
)
@click.option(
    "--era",
    type=str,
    default=None,
    help=(
        "D5-2 era filter (wyrd-lyp). Pass a year (e.g. 1086), a cell label "
        "('oe-late', 'me', 'middle-irish'), or 'family/label' to "
        "disambiguate. Morphemes with no attested-year data pass through."
    ),
)
@click.option(
    "--stratum",
    type=str,
    default=None,
    help=(
        "wyrd-lr4 Phase 3 within-language stratum filter. Pass a register "
        "tag — for Welsh: 'native-welsh', 'brittonic-substrate', "
        "'medieval-welsh', 'latin-loan', 'english-loan'; for French: "
        "'native-french', 'medieval-french', 'gallo-roman', "
        "'gaulish-substrate', 'frankish-substrate'; for Old English: "
        "'native-old-english', 'celtic-substrate', 'norse-loan', "
        "'latin-loan'; for Old Norse: 'native-old-norse', 'east-norse', "
        "'gaelic-substrate', 'english-loan', 'low-german-loan', "
        "'latin-loan'. Morphemes with no stratum data pass through. "
        "Composes with --era via intersection."
    ),
)
@click.option(
    "--cohesion",
    type=click.FloatRange(0.0, 1.0),
    default=0.0,
    show_default=True,
    help=(
        "wyrd-mj2 tag co-occurrence bias (0..1). 0 keeps independent slot "
        "sampling (today's behavior); higher values bias each slot toward "
        "usages whose tags co-occur with previously-picked slots in the "
        "empirical corpus. Composes orthogonally with --novelty."
    ),
)
def generate(
    culture: str,
    tags: tuple[str, ...],
    count: int,
    seed: int | None,
    describe: bool,
    spelling_variety: float,
    novelty: float,
    inflection_density: float,
    moods: tuple[str, ...],
    include_fiction: bool,
    era: str | None,
    stratum: str | None,
    cohesion: float,
) -> None:
    """Generate town names. Replaces Rando's `bin/generator`."""
    # `available_tags()` already strips _INTERNAL_TAGS for the SPA dropdown,
    # but the CLI needs to ACCEPT internal tags (some scripts pass them
    # explicitly) — so widen the validation set with _INTERNAL_TAGS rather
    # than the literal subset, which would silently desync as new internal
    # markers (e.g. wyrd-yan's 'fiction') get added.
    known_tags = set(available_tags()) | _INTERNAL_TAGS
    bad = [t for t in tags if t not in known_tags]
    if bad:
        click.echo(f"Unknown tag(s): {', '.join(bad)}", err=True)
        click.echo("Available tags:", err=True)
        for t in sorted(known_tags - _INTERNAL_TAGS):
            click.echo(f"  {t}", err=True)
        sys.exit(1)
    bad_moods = [m for m in moods if m.split(":", 1)[0] not in MOODS]
    if bad_moods:
        click.echo(f"Unknown mood(s): {', '.join(bad_moods)}", err=True)
        click.echo(f"Available moods: {', '.join(sorted(MOODS))}", err=True)
        sys.exit(1)

    resolved = resolve_seed(seed)
    seed_rng = rng_for(resolved)
    kenning = Kenning()
    params = {
        "culture": culture,
        "tags": list(tags),
        "spelling_variety": spelling_variety,
        "novelty": novelty,
        "inflection_density": inflection_density,
        "mood": list(moods),
        "include_fiction": include_fiction,
        "era": era,
        "stratum": stratum,
        "cohesion": cohesion,
    }
    for _ in range(count):
        try:
            result = kenning.generate(params, seed_rng.randrange(2**63))
        except ValueError as exc:
            # Surface user-input errors (bad --era, etc.) as friendly
            # CLI messages on stderr + exit non-zero, matching the
            # tags/moods pre-validation pattern above. Other unexpected
            # exceptions still propagate so a real bug isn't silenced.
            click.echo(f"Error: {exc}", err=True)
            sys.exit(1)
        click.echo(result.result)
        if describe:
            click.echo(result.explanation)
    click.echo(f"(seed: {resolved})", err=True)


def add_to(parent: click.Group) -> None:
    """Register ``generate`` on the top-level ``@cli`` group."""
    parent.add_command(generate)
