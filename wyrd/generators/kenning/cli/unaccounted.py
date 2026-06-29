"""``wyrd kenning unaccounted`` — surface morpheme gaps in the meaning DB."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import click

from wyrd.generators.kenning import CULTURES
from wyrd.generators.kenning.cli.utils import _decompose_corpus, _load_meanings_data
from wyrd.generators.kenning.lexicon import annotate_fragments_with_corpus_evidence
from wyrd.generators.kenning.runtime.meaning import load_meanings
from wyrd.generators.kenning.runtime.name import load_names_with_regions


@click.command("unaccounted")
@click.argument("culture", type=click.Choice(CULTURES))
@click.argument("place_names", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--meanings",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to meanings.json (defaults to bundled).",
)
@click.option("--top", type=int, default=50, help="Show top N most-frequent fragments.")
@click.option("--examples", type=int, default=3, help="Up to N example names per fragment.")
@click.option("--min-length", type=int, default=2, help="Skip fragments shorter than this.")
@click.option("--as-json", is_flag=True, default=False, help="Emit machine-readable JSON.")
@click.option(
    "--sources-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help=(
        "wyrd-bvp: when set, annotate each top-N fragment with "
        "scholarly-corpus presence — count of source files where "
        "the fragment appears at a word boundary, plus 1-2 sample "
        "snippets with surrounding context. Strong-hit indicator "
        "flags snippets that look like an etymology body."
    ),
)
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "wyrd-70s0: lexicon DB to read canonical decompositions from. "
        "When set, gap counts honour the canonical pick — a fragment "
        "appears in the gap report only if it's unaccounted under the "
        "toponym's canonical decomposition. False-gap fragments that "
        "the heuristic surfaced but a zero-unaccounted alternate "
        "covered will silently disappear from the report."
    ),
)
def unaccounted(
    culture: str,
    place_names: Path,
    meanings: Path | None,
    top: int,
    examples: int,
    min_length: int,
    as_json: bool,
    sources_dir: Path | None,
    db_path: Path | None,
) -> None:
    """Surface morpheme fragments the corpus uses but the meaning DB doesn't.

    Decomposes every name in <place_names>; for names that don't fully decompose,
    aggregates the leftover string fragments by frequency. The top fragments are
    candidates for new entries in meanings.json — this is the mining tool that
    surfaces what the previous deconstruction silently dropped.

    With ``--db``, canonical decompositions (wyrd-08m / wyrd-70s0) drive
    the gap counting: a fragment is reported as unaccounted only if it
    appears in the toponym's canonical breakdown's leftovers. Names
    without a canonical fall back to the heuristic. Operators see fewer
    false-gap fragments at the cost of needing a recent
    ``lexicon decompose --apply`` run.
    """
    meanings_data = _load_meanings_data(meanings)

    names_data = json.loads(place_names.read_text(encoding="utf-8"))
    name_entries = load_names_with_regions(names_data)
    word_db, _ = load_meanings(meanings_data)

    resolved_names, canonical_hits = _decompose_corpus(name_entries, word_db, db_path)
    fragments, examples_by_frag, imperfect_count = _aggregate_unaccounted_fragments(
        resolved_names, min_length=min_length, max_examples=examples
    )

    top_fragments = fragments.most_common(top)
    # wyrd-bvp: when --sources-dir is set, scan all source-text
    # files for each top-N fragment and annotate with corpus
    # presence. The annotation is computed once and threaded
    # through both JSON and text output paths so consumers see the
    # same evidence shape regardless of format.
    evidence_by_frag: dict[str, dict[str, Any]] = {}
    if sources_dir is not None:
        evidence_by_frag = annotate_fragments_with_corpus_evidence(
            [frag for frag, _ in top_fragments],
            sources_dir,
        )

    summary = (
        f"culture={culture} imperfect_names={imperfect_count} unique_fragments={len(fragments)}"
    )
    if canonical_hits:
        breakdown = " ".join(f"{src}={n}" for src, n in sorted(canonical_hits.items()))
        summary += f" canonical[{breakdown}]"
    click.echo(summary, err=True)

    if as_json:
        out = []
        for frag, count in top_fragments:
            entry = {"fragment": frag, "count": count, "examples": examples_by_frag[frag]}
            if evidence_by_frag:
                entry.update(evidence_by_frag.get(frag, {}))
            out.append(entry)
        click.echo(json.dumps(out, indent=2))
        return

    if evidence_by_frag:
        click.echo(f"{'fragment':<20} {'count':>6}  {'corpus':>6}  {'strong':>6}  examples")
    else:
        click.echo(f"{'fragment':<20} {'count':>6}  examples")
    click.echo("-" * 70)
    for frag, count in top_fragments:
        ex = ", ".join(examples_by_frag[frag])
        if evidence_by_frag:
            ev = evidence_by_frag.get(frag, {})
            corpus = ev.get("corpus_hits", 0)
            strong = ev.get("strong_hits", 0)
            click.echo(f"{frag:<20} {count:>6}  {corpus:>6}  {strong:>6}  {ex}")
        else:
            click.echo(f"{frag:<20} {count:>6}  {ex}")


def _aggregate_unaccounted_fragments(
    resolved_names: list,
    *,
    min_length: int,
    max_examples: int,
) -> tuple[Counter, dict[str, list[str]], int]:
    """Walk decomposed Names and tally their leftover string slots.

    Per name, dedups fragments within that name (so 'ham' appearing
    twice in one toponym counts once toward frequency). Examples are
    capped at ``max_examples`` per fragment. Fragments shorter than
    ``min_length`` are skipped.

    Returns ``(fragments, examples_by_frag, imperfect_count)`` where
    ``imperfect_count`` is the number of names contributing at least
    one fragment (i.e. didn't reach zero unaccounted).
    """
    fragments: Counter = Counter()
    examples_by_frag: dict[str, list[str]] = {}
    imperfect_count = 0
    for resolved in resolved_names:
        if resolved.count_unaccounted() == 0:
            continue
        imperfect_count += 1
        seen_in_name: set[str] = set()
        for word_list in resolved.words.values():
            for word in word_list:
                for chunk in word.word:
                    if not isinstance(chunk, str):
                        continue
                    frag = chunk.lower()
                    if len(frag) < min_length or frag in seen_in_name:
                        continue
                    seen_in_name.add(frag)
                    fragments[frag] += 1
                    bucket = examples_by_frag.setdefault(frag, [])
                    if resolved.name not in bucket and len(bucket) < max_examples:
                        bucket.append(resolved.name)
    return fragments, examples_by_frag, imperfect_count


def add_to(parent: click.Group) -> None:
    """Register ``unaccounted`` on the top-level ``@cli`` group."""
    parent.add_command(unaccounted)
