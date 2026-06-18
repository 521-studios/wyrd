"""``wyrd kenning lexicon grade-decomposition`` — grade the matcher against the
scholarly corpus and diff connective OFF vs ON (wyrd-aicu.9 Phase 3)."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning import CULTURES
from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH, _load_meanings_data
from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.lexicon.decomposition_grader import (
    CorpusDiff,
    GradeSummary,
    grade_corpus_diff,
    grade_passthrough_diff,
)
from wyrd.generators.kenning.lexicon.genitive_priors import (
    build_split_probability_map,
    load_genitive_prior_counts,
)
from wyrd.generators.kenning.lexicon.passthrough_mining import (
    build_passthrough_map,
    extract_cross_scholar_passthroughs,
)
from wyrd.generators.kenning.lexicon.proportions_builder import CULTURE_LANGUAGES
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY
from wyrd.generators.kenning.runtime.meaning import load_meanings
from wyrd.generators.kenning.runtime.trie_matcher import build_morpheme_trie

_PROGRESS_EVERY = 1000


def _fmt_summary(s: GradeSummary) -> str:
    return (
        f"recall={s.mean_recall:.3f} precision={s.mean_precision:.3f} "
        f"exact={s.exact_rate:.3f} coverage={s.coverage_rate:.3f} "
        f"head={s.head_attested_rate:.3f}"
    )


def _delta(off: float, on: float) -> str:
    d = on - off
    return f"{d:+.3f}"


def _report(diff: CorpusDiff, *, max_list: int) -> None:
    click.echo("", err=True)
    click.echo(f"graded {diff.on.graded} toponyms (connective off vs on)", err=True)
    click.echo(f"  off : {_fmt_summary(diff.off)}", err=True)
    click.echo(f"  on  : {_fmt_summary(diff.on)}", err=True)
    click.echo(
        "  Δ   : "
        f"recall={_delta(diff.off.mean_recall, diff.on.mean_recall)} "
        f"precision={_delta(diff.off.mean_precision, diff.on.mean_precision)} "
        f"exact={_delta(diff.off.exact_rate, diff.on.exact_rate)} "
        f"coverage={_delta(diff.off.coverage_rate, diff.on.coverage_rate)} "
        f"head={_delta(diff.off.head_attested_rate, diff.on.head_attested_rate)}",
        err=True,
    )
    click.echo(
        f"  improvements={len(diff.improvements)}  regressions={len(diff.regressions)}",
        err=True,
    )

    if diff.regressions:
        click.echo("", err=True)
        click.echo(
            "REGRESSIONS (on worse than off — the overfitting tripwire; "
            "must be empty or each individually defensible):",
            err=True,
        )
        for chg in diff.regressions[:max_list]:
            click.echo(
                f"  {chg.name}: [{', '.join(chg.dimensions)}]  "
                f"off={list(chg.off_parse)} -> on={list(chg.on_parse)}",
                err=True,
            )
        if len(diff.regressions) > max_list:
            click.echo(f"  ... +{len(diff.regressions) - max_list} more", err=True)

    if diff.improvements:
        click.echo("", err=True)
        click.echo(f"sample improvements (first {max_list}):", err=True)
        for chg in diff.improvements[:max_list]:
            click.echo(
                f"  {chg.name}: [{', '.join(chg.dimensions)}]  "
                f"off={list(chg.off_parse)} -> on={list(chg.on_parse)}",
                err=True,
            )


@click.command("grade-decomposition")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--meanings",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to a meanings.json bundle (default: the resolved runtime DB bundle).",
)
@click.option(
    "--culture",
    type=click.Choice(CULTURES),
    default="english",
    show_default=True,
    help="Culture whose language set is threaded into the matcher tiebreaker (both configs).",
)
@click.option(
    "--suffix",
    default=None,
    help="Grade only toponyms whose folded name ends with this (e.g. 'ston') — fast focused probe.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Grade at most this many toponyms (after sorting + suffix filter).",
)
@click.option(
    "--max-list",
    type=int,
    default=40,
    show_default=True,
    help="Max regression / improvement rows to print.",
)
@click.option(
    "--passthroughs",
    is_flag=True,
    default=False,
    help="wyrd-h5u1: diff composed-of attribution OFF vs ON (a matched composite "
    "credited to its constituents) instead of connective OFF vs ON — the graded "
    "net-win check that gates wyrd-oth3. Mines passthroughs live from the corpus.",
)
def lexicon_grade_decomposition(
    db_path: Path,
    meanings: Path | None,
    culture: str,
    suffix: str | None,
    limit: int | None,
    max_list: int,
    passthroughs: bool,
) -> None:
    """wyrd-aicu.9 Phase 3: grade the decomposition matcher against the scholarly
    breakdown corpus, diffing connective OFF (today's matcher) vs ON (genitive
    connective + the smoothed split prior).

    Builds the matcher trie from the runtime bundle (the same morpheme inventory
    the generation proportions train on), grades every labeled toponym by
    cognate-cluster recall / precision / exact-set agreement / coverage, and
    prints the off→on deltas plus the regression list — the overfitting tripwire.

    The genitive prior is read from ``genitive_split_prior`` (run
    ``mine-genitive-priors --apply`` first); if the table is empty the ON config
    still measures the connective's coverage gains, only the homograph tiebreak
    is silent (reported).
    """
    meanings_data = _load_meanings_data(meanings)
    meaning_db, _ = load_meanings(meanings_data)
    trie = build_morpheme_trie(meaning_db)
    culture_languages = CULTURE_LANGUAGES.get(culture)

    with LexiconDB(db_path) as db:
        counts = load_genitive_prior_counts(db)
        genitive_prior = build_split_probability_map(counts)
        click.echo("grade-decomposition:", err=True)
        click.echo(
            f"  trie morphemes={trie.morpheme_count} culture={culture} "
            f"genitive_prior_pairs={len(genitive_prior)}",
            err=True,
        )
        if not genitive_prior:
            click.echo(
                "  WARNING: genitive_split_prior is empty (table missing or "
                "un-mined) — ON measures coverage only, no homograph tiebreak.",
                err=True,
            )
        if passthroughs:
            pmap = build_passthrough_map(extract_cross_scholar_passthroughs(db))
            click.echo(f"  passthrough composites mined={len(pmap)}", err=True)
            diff = grade_passthrough_diff(
                db,
                trie,
                passthrough_map=pmap,
                culture_languages=culture_languages,
                genitive_prior=genitive_prior,
                suffix=suffix,
                limit=limit,
                progress_every=_PROGRESS_EVERY,
            )
        else:
            diff = grade_corpus_diff(
                db,
                trie,
                culture_languages=culture_languages,
                genitive_prior=genitive_prior,
                suffix=suffix,
                limit=limit,
                progress_every=_PROGRESS_EVERY,
            )
    _report(diff, max_list=max_list)


def add_to(parent: click.Group) -> None:
    """Register ``grade-decomposition`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_grade_decomposition)
