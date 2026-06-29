"""``wyrd kenning rebuild-proportions`` — recompute a culture's proportions JSON."""

from __future__ import annotations

import json
from pathlib import Path

import click

from wyrd.generators.kenning import CULTURES
from wyrd.generators.kenning.cli.utils import _decompose_corpus, _load_meanings_data

# The core proportions-building helpers (proportions_from / ordered_tag_pairs /
# encode_structs / encode_meaning) moved to ``lexicon/proportions_builder.py``
# during the d90t cutover so both this CLI and the L4 emit can call into them
# without lexicon/ depending on cli/. Re-exported under the legacy underscored
# names so external callers (and the test suite) keep working unchanged.
from wyrd.generators.kenning.lexicon.proportions_builder import (
    encode_meaning as _encode_meaning,
)
from wyrd.generators.kenning.lexicon.proportions_builder import (
    encode_structs as _encode_structs,
)
from wyrd.generators.kenning.lexicon.proportions_builder import (
    ordered_tag_pairs as _ordered_tag_pairs,
)
from wyrd.generators.kenning.lexicon.proportions_builder import (
    proportions_from as _proportions_from,
)
from wyrd.generators.kenning.runtime.meaning import load_meanings
from wyrd.generators.kenning.runtime.name import load_names_with_regions

__all__ = [
    "_encode_meaning",
    "_encode_structs",
    "_ordered_tag_pairs",
    "_proportions_from",
    "add_to",
    "rebuild_proportions",
]


@click.command("rebuild-proportions")
@click.argument("culture", type=click.Choice(CULTURES))
@click.argument("place_names", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--meanings",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to meanings.json (defaults to bundled).",
)
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "wyrd-70s0: lexicon DB to read canonical decompositions from. "
        "When set, names with a canonical pick (from `lexicon decompose "
        "--apply`) use that breakdown; names without one fall back to "
        "the matcher's reduce=True heuristic. Without this flag the "
        "command runs heuristic-only (legacy behaviour)."
    ),
)
def rebuild_proportions(
    culture: str,
    place_names: Path,
    meanings: Path | None,
    db_path: Path | None,
) -> None:
    """Recompute a culture's proportions from a corpus + meanings DB.

    Replaces Rando's `bin/load_names`. Reads place_names JSON (the same shape as
    the bundled `<culture>_place_names.json` files), deconstructs each name
    against the meaning DB, and emits a fresh proportions JSON to stdout.

    With ``--db``, canonical decompositions (wyrd-08m / wyrd-70s0) drive
    the proportion counts for toponyms with a registered canonical pick;
    others fall back to the matcher heuristic. This makes counts stable
    across re-runs and reflects scholarly attribution where available.
    """
    meanings_data = _load_meanings_data(meanings)

    names_data = json.loads(place_names.read_text(encoding="utf-8"))
    name_entries = load_names_with_regions(names_data)
    word_db, _ = load_meanings(meanings_data)

    # wyrd-pfoo: thread the culture's expected-language set through to
    # the matcher's tiebreaker so per-culture splits prefer culture-
    # aligned tagged Meanings on ambiguous decompositions. Cultures
    # missing from CULTURE_LANGUAGES (custom corpus / future culture)
    # get None → matcher falls back to pre-PR bit-stable behaviour.
    from wyrd.generators.kenning.lexicon.proportions_builder import CULTURE_LANGUAGES

    # wyrd-aicu.9: when a --db is supplied, build the genitive-split prob-map
    # from its mined counts (smoothing applied here) and engage the default
    # connective inventory so the rebuilt proportions train on the corrected
    # X·s·head decompositions. Without --db (or an unmined DB) the map is empty
    # → connective stays off → legacy bit-stable heuristic path.
    connective_inventory = None
    genitive_prior = None
    if db_path is not None:
        from wyrd.generators.kenning.lexicon import LexiconDB
        from wyrd.generators.kenning.lexicon.genitive_priors import (
            build_split_probability_map,
            load_genitive_prior_counts,
        )
        from wyrd.generators.kenning.runtime.connective import DEFAULT_CONNECTIVE_INVENTORY

        with LexiconDB(db_path) as prior_db:
            genitive_prior = build_split_probability_map(load_genitive_prior_counts(prior_db))
        if genitive_prior:
            connective_inventory = DEFAULT_CONNECTIVE_INVENTORY

    resolved_names, canonical_hits = _decompose_corpus(
        name_entries,
        word_db,
        db_path,
        culture_languages=CULTURE_LANGUAGES.get(culture),
        connective_inventory=connective_inventory,
        genitive_prior=genitive_prior,
    )
    good_names = [n for n in resolved_names if n.count_unaccounted() == 0]
    word_names = sum(1 for n in resolved_names if n.has_name())
    word_saints = sum(1 for n in resolved_names if n.has_saint())

    summary = (
        f"culture={culture} perfect={len(good_names)} names={word_names} "
        f"saints={word_saints} total={len(name_entries)}"
    )
    if canonical_hits:
        breakdown = " ".join(f"{src}={n}" for src, n in sorted(canonical_hits.items()))
        summary += f" canonical[{breakdown}]"
    click.echo(summary, err=True)
    proportions = _proportions_from(good_names)
    # wyrd-xmk3 review: sort_keys so the on-disk proportions JSON is
    # deterministic across re-runs. Insertion-order dicts produced
    # non-determinism in usages / single_usages / tag_cooccurrence
    # maps, polluting the diff in every rebuild even when counts
    # were identical. The structures array is a list of dicts; each
    # 'words' position is iterated by index so the array order is
    # already determined by the corpus walk — sort_keys handles the
    # dict slots inside.
    click.echo(json.dumps(proportions, sort_keys=True))


def add_to(parent: click.Group) -> None:
    """Register ``rebuild-proportions`` on the top-level ``@cli`` group."""
    parent.add_command(rebuild_proportions)
