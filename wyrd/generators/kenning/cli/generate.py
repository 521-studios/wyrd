"""``wyrd kenning generate`` — produce town names from a culture's morpheme proportions."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from wyrd.generators.kenning import (
    _INTERNAL_TAGS,
    CULTURES,
    Kenning,
    available_tags,
)
from wyrd.generators.kenning.registers.effects import available_register_effects
from wyrd.seed import resolve_seed, rng_for

_VALID_SCORING_MODES = ("proportions", "vector")


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
    "--json",
    "json_output",
    is_flag=True,
    default=False,
    help=(
        "wyrd-cp2d: emit JSON instead of plain text. One JSON object per "
        "line (newline-delimited) carrying the name + per-word morpheme "
        "breakdown with per-language source lemmas. Consumed by 'kenning "
        "rewind --from-json' / 'kenning era-map --from-json' to skip "
        "re-decomposition and use the generator's actual morpheme "
        "picks. Suppresses --describe (the morpheme structure is in "
        "the JSON itself)."
    ),
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
@click.option(
    "--scoring-mode",
    type=click.Choice(_VALID_SCORING_MODES),
    default="proportions",
    show_default=True,
    help=(
        "wyrd-ecjp.5: per-slot sampling pipeline. 'proportions' (default) "
        "uses the pre-baked per-(culture × tag × position) tables — "
        "bit-stable with the legacy path. 'vector' uses the D36.2 "
        "gate→score→sample primitive: each slot's per-lemma weight is "
        "computed at request time from phon + sem + pos + empirical-"
        "baseline axes. 'vector' requires per-lemma phonological_vector "
        "data in the bundle (kq7w.1) and benefits from --priors-path; "
        "without either, the baseline axis contributes 0 and the score "
        "falls back to phon + sem + pos."
    ),
)
@click.option(
    "--priors-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "wyrd-ecjp.5 PR C: filesystem path to a JSON empirical-priors "
        "sidecar (emitted by 'wyrd kenning lexicon dump-empirical-"
        "priors'). Only consulted when --scoring-mode=vector. When "
        "absent, the vector path's baseline axis contributes 0."
    ),
)
@click.option(
    "--baseline-weight",
    type=click.FloatRange(0.0, 10.0),
    default=1.0,
    show_default=True,
    help=(
        "wyrd-ecjp.9: weight scalar on the empirical-baseline axis "
        "(D36.2). 0 disables the baseline contribution entirely (like "
        "running without --priors-path). 1.0 is the canonical "
        "weighting; >1 over-weights baseline at the expense of phon / "
        "sem / pos. Only meaningful with --scoring-mode=vector."
    ),
)
@click.option(
    "--phonological-weight",
    type=click.FloatRange(0.0, 10.0),
    default=1.0,
    show_default=True,
    help=(
        "wyrd-ecjp.9: weight scalar on the phonological axis (D36.2). "
        "1.0 is canonical; >1 over-weights phonological character "
        "(harshness / softness etc.) relative to other axes. Only "
        "meaningful with --scoring-mode=vector."
    ),
)
@click.option(
    "--semantic-weight",
    type=click.FloatRange(0.0, 10.0),
    default=1.0,
    show_default=True,
    help=(
        "wyrd-ecjp.9: weight scalar on the semantic-tag axis (D36.2). "
        "1.0 is canonical; >1 over-weights tag matching relative to "
        "other axes. Only meaningful with --scoring-mode=vector."
    ),
)
@click.option(
    "--position-weight",
    type=click.FloatRange(0.0, 10.0),
    default=1.0,
    show_default=True,
    help=(
        "wyrd-ecjp.9: weight scalar on the position axis (D36.2). "
        "1.0 is canonical. Only meaningful with --scoring-mode=vector."
    ),
)
@click.option(
    "--pack",
    "pack_specs",
    multiple=True,
    metavar="NAME[:WEIGHT]",
    help=(
        "wyrd-ecjp.11: declare a scenario-pack overlay (D36.4). NAME "
        "must match a bundled pack (see 'wyrd kenning generate --help' "
        "for the available list once packs are bundled). Optional "
        ":WEIGHT (float in [0, 10]) overrides the pack manifest's "
        "default_weight. Repeatable for multi-pack composition. Only "
        "meaningful with --scoring-mode=vector. Pack lemmas enter the "
        "eligible pool alongside native lemmas and inherit their "
        "manifest's template (donor → recipient) for baseline scoring."
    ),
)
@click.option(
    "--pack-tag-filter",
    "pack_tag_filter_specs",
    multiple=True,
    metavar="PACK:TAG1,TAG2,...",
    help=(
        "wyrd-ecjp.11: narrow a declared pack's lemmas to those "
        "carrying at least one matching tag (PackOverlay."
        "allowed_pack_tags). PACK must match a --pack declaration. "
        "Repeatable for filtering multiple packs (one entry per pack "
        "max). Used for narrative subset selection (wyrd-sreb): a "
        "war scene can pull only the 'war'-tagged subset of a "
        "Tatar-9c pack while leaving the religious / trade subsets "
        "behind."
    ),
)
def generate(
    culture: str,
    tags: tuple[str, ...],
    count: int,
    seed: int | None,
    describe: bool,
    json_output: bool,
    spelling_variety: float,
    novelty: float,
    inflection_density: float,
    moods: tuple[str, ...],
    include_fiction: bool,
    era: str | None,
    stratum: str | None,
    cohesion: float,
    scoring_mode: str,
    priors_path: Path | None,
    baseline_weight: float,
    phonological_weight: float,
    semantic_weight: float,
    position_weight: float,
    pack_specs: tuple[str, ...],
    pack_tag_filter_specs: tuple[str, ...],
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
    known_moods = available_register_effects()
    bad_moods = [m for m in moods if m.split(":", 1)[0] not in known_moods]
    if bad_moods:
        click.echo(f"Unknown mood(s): {', '.join(bad_moods)}", err=True)
        click.echo(f"Available moods: {', '.join(known_moods)}", err=True)
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
        "scoring_mode": scoring_mode,
    }
    if priors_path is not None:
        params["priors_path"] = str(priors_path)
    # ScoringWeights only fire when scoring_mode=vector. In proportions
    # mode the legacy path doesn't read scoring_weights — adding it
    # would be inert but cluttering, so we omit. This is the operator-
    # friendly contract: weight flags in proportions mode become a
    # silent no-op (also covered by
    # test_proportions_mode_ignores_vector_only_flags).
    if scoring_mode == "vector":
        params["scoring_weights"] = {
            "phon_w": phonological_weight,
            "sem_w": semantic_weight,
            "pos_w": position_weight,
            "base_w": baseline_weight,
        }
    # wyrd-ecjp.11: pack overlays. Parse + validate at the CLI
    # boundary so invalid input surfaces as a friendly Click error
    # before reaching the generator. Pack-related flags in
    # proportions mode are deliberate silent no-ops (same contract
    # as the weight flags); the parsed packs land in params anyway
    # so the schema stays uniform, but Kenning.generate only
    # consumes them in vector mode.
    if pack_specs or pack_tag_filter_specs:
        from wyrd.generators.kenning import _load_packs
        from wyrd.generators.kenning.cli._pack_args import (
            parse_pack_specs,
            parse_pack_tag_filter_specs,
            validate_packs_against_catalog,
        )

        try:
            parsed_packs = parse_pack_specs(pack_specs)
            declared = {s.pack_name for s in parsed_packs}
            tag_filters = parse_pack_tag_filter_specs(pack_tag_filter_specs, declared)
            validate_packs_against_catalog(parsed_packs, set(_load_packs().keys()))
        except click.BadParameter as exc:
            click.echo(f"Error: {exc.format_message()}", err=True)
            sys.exit(1)
        params["packs"] = [
            {
                "pack_name": spec.pack_name,
                "weight_override": spec.weight_override,
                "allowed_pack_tags": sorted(tag_filters.get(spec.pack_name, ())),
            }
            for spec in parsed_packs
        ]
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
        if json_output:
            # wyrd-cp2d: newline-delimited JSON, one object per name.
            # Includes the rendered string + per-word morpheme breakdown
            # (with per-language source lemmas) so downstream commands
            # can skip re-decomposition and use the generator's actual
            # morpheme picks.
            import json

            click.echo(
                json.dumps(
                    {
                        "name": result.result,
                        "culture": culture,
                        "words": result.morphemes_by_word or [],
                    },
                    sort_keys=True,
                )
            )
        else:
            click.echo(result.result)
            if describe:
                click.echo(result.explanation)
    click.echo(f"(seed: {resolved})", err=True)


def add_to(parent: click.Group) -> None:
    """Register ``generate`` on the top-level ``@cli`` group."""
    parent.add_command(generate)
