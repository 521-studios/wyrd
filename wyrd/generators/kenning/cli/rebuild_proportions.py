"""``wyrd kenning rebuild-proportions`` — recompute a culture's proportions JSON."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import click

from wyrd.generators.kenning import CULTURES
from wyrd.generators.kenning.cli.utils import _decompose_corpus, _load_meanings_data
from wyrd.generators.kenning.runtime.meaning import Meaning, load_meanings
from wyrd.generators.kenning.runtime.name import load_names_with_regions


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

    names_data = json.loads(place_names.read_text())
    name_entries = load_names_with_regions(names_data)
    word_db, _ = load_meanings(meanings_data)

    resolved_names, canonical_hits = _decompose_corpus(name_entries, word_db, db_path)
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


def _proportions_from(names) -> dict:
    part_proportions: Counter = Counter()
    lone_proportions: Counter = Counter()
    struct_proportions: Counter = Counter()
    tag_cooccurrence: Counter = Counter()
    tag_marginal: Counter = Counter()
    for name in names:
        for u in name.get_samples():
            part_proportions[u] += 1
        for u in name.get_lone_samples():
            lone_proportions[u] += 1
        for structure in name.get_structure():
            struct_proportions[structure] += 1
        # Tag co-occurrence: ordered consecutive Meaning pairs across the
        # full name (within-word AND between-word). Each pair contributes
        # the cartesian product of (left.tags × right.tags) — an etymon
        # often has several tags (plant, food, tree), and any of them can
        # legitimately drive co-occurrence.
        for left_tags, right_tags in _ordered_tag_pairs(name):
            for tl in left_tags:
                tag_marginal[tl] += 1
                for tr in right_tags:
                    tag_cooccurrence[(tl, tr)] += 1
            for tr in right_tags:
                tag_marginal[tr] += 1
    return {
        "usages": dict(part_proportions),
        "single_usages": dict(lone_proportions),
        "structures": _encode_structs(struct_proportions),
        # Serialize tag-pair keys as "left|right" so they survive a JSON
        # round-trip. Generator side splits on the pipe.
        "tag_cooccurrence": {f"{a}|{b}": v for (a, b), v in tag_cooccurrence.items()},
        "tag_marginal": dict(tag_marginal),
    }


def _ordered_tag_pairs(name) -> list[tuple[list[str], list[str]]]:
    """Walk a decomposed Name in order, yielding consecutive (left.tags,
    right.tags) pairs across the entire name (within and between words).

    For 'Bridgwater Gate' this yields:
        (Bridg-.tags, -water.tags)   # within word 1
        (-water.tags, Gate.tags)     # between words

    When a word has multiple equivalent decompositions (e.g. mining surfaced
    a bare-form synthesized usage that competes with the dashed reflex), prefer
    the decomp with the most tag-bearing Meanings — that's the one that
    actually feeds co-occurrence statistics rather than emitting empty
    () pairs.
    """

    def _tag_density(word_obj) -> int:
        if not hasattr(word_obj, "word"):
            return -1
        return sum(1 for c in word_obj.word if isinstance(c, Meaning) and c.tags)

    flat_meanings: list[Meaning] = []
    for word_text in name.name.split(" "):
        word_options = name.words.get(word_text, [])
        if not word_options:
            continue
        chosen = max(word_options, key=_tag_density)
        if not hasattr(chosen, "word"):
            continue
        for chunk in chosen.word:
            if isinstance(chunk, Meaning):
                flat_meanings.append(chunk)

    out: list[tuple[list[str], list[str]]] = []
    for i in range(len(flat_meanings) - 1):
        out.append((list(flat_meanings[i].tags), list(flat_meanings[i + 1].tags)))
    return out


def _encode_meaning(qualities) -> dict:
    out: dict = {}
    for quality in qualities:
        if quality in ("pre", "post", "inner"):
            out["location"] = quality
        else:
            out[quality] = True
    return out


def _encode_structs(struct: Counter) -> list:
    """Serialize the structure → count map into the proportions-JSON
    shape, sorted for deterministic output (wyrd-dxu2 review). Counter
    iteration order is insertion-order, so the underlying corpus walk
    determines initial order — but two re-runs against the same corpus
    can iterate in slightly different orders if the matcher's
    decomposition outputs change. Sorting by (-proportion, structure
    key) gives a stable canonical order: most-frequent first, with the
    structure tuple as deterministic tiebreaker.

    wyrd-zzli: drops multi-word structures with any single-word slot
    that's a bare 'pre' or 'post' morpheme. These templates were
    causing 'By Green'-style ungrammatical output where the runtime
    rendered an attachment-only morpheme (``-by``, ``green-``) as a
    standalone word. 46.7% of English structure weight was in this
    shape before the filter; corresponding runtime defense in
    ``proportions.is_structurally_grammatical``. Real qualifier-word
    two-word names (Bishop's Stortford, Great Yarmouth) survive
    because their lead morpheme carries the ``name`` flag, which is
    treated as a distinct position key by ``word_to_key`` and the
    runtime."""
    from wyrd.generators.kenning.runtime.proportions import is_structurally_grammatical

    sorted_items = sorted(struct.items(), key=lambda item: (-item[1], item[0]))
    return [
        {
            "proportion": value,
            "words": [[_encode_meaning(meaning) for meaning in word] for word in key],
        }
        for key, value in sorted_items
        if is_structurally_grammatical(key)
    ]


def add_to(parent: click.Group) -> None:
    """Register ``rebuild-proportions`` on the top-level ``@cli`` group."""
    parent.add_command(rebuild_proportions)
