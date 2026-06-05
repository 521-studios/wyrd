"""``wyrd kenning lexicon curate-collapse-etymon`` — append a manual
collapse (fold ± descent-edge detach) to the ``_collapses.jsonl`` ledger
(wyrd-qp9c).

The collapse ledger is normally written by the deterministic
``detect-collapses`` pass and the LLM ``merge-collapses`` pass. This
command is the hand-curated escape hatch for a one-off consolidation an
operator has reasoned through directly — in particular the
**homograph-conflation** case the detectors can't safely judge.

The motivating case: OE ``don`` is one ``(canonical_form, language)`` row
serving two morphemes — the toponym "hill" (a reduced form of ``dūn``)
AND the verb "to do". The verb sense dragged in a chain of descent edges
that put the toponym into the 91-member verb cluster (do/doff/done), so
the ``-don`` era-grid rendered verb reflexes. The fix detaches the verb
edges, then folds the clean toponym into ``dūn``::

    wyrd kenning lexicon curate-collapse-etymon old-english:don \\
        --into old-english:dūn \\
        --detach-parent gmw-pro:*dōn \\
        --detach-parent proto-indo-european:*dʰeh₁- \\
        --detach-child modern-english:do \\
        --detach-child middle-english:don \\
        --reason "wyrd-qp9c: OE toponym don (hill, =dūn) conflated with verb dōn"

(``--detach-child`` abbreviated above — the live wyrd-qp9c row in
``_collapses.jsonl`` detaches all six ME/ModE verb reflexes.)

Replayed by ``enrichment.apply_collapses`` in ``run_full_enrichment``'s
curation slot, BEFORE cluster-cognates — so a from-scratch rebuild
reproduces the detach + fold with no re-mining.
"""

from __future__ import annotations

import json
from pathlib import Path

import click

_COLLAPSE_SOURCE_ROW = {
    "_type": "source",
    "ref": "collapse",
    "title": "Etymon collapse ledger (wyrd-y651)",
    "notes": (
        "Form-of / variant etymons folded into their lemma. Replayed by "
        "run_full_enrichment's curation slot. Deterministic rows from "
        "detect-collapses; LLM rows from the merge pass. Revert with into: ''."
    ),
}


def _ensure_collapse_file(collapse_file: Path) -> None:
    """Create the collapse JSONL with its synthetic source row if absent —
    every L2 JSONL file needs exactly one source row."""
    if collapse_file.exists():
        return
    collapse_file.parent.mkdir(parents=True, exist_ok=True)
    with collapse_file.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(_COLLAPSE_SOURCE_ROW, ensure_ascii=False))
        fh.write("\n")


def _append_event(collapse_file: Path, payload: dict) -> None:
    _ensure_collapse_file(collapse_file)
    with collapse_file.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False))
        fh.write("\n")


@click.command("curate-collapse-etymon")
@click.argument("etymon_ref")
@click.option(
    "--into",
    "into_ref",
    default=None,
    help=(
        "Surviving lemma ref ``<language>:<canonical_form>`` to fold ETYMON_REF "
        "into. Omit for a detach-only event (no fold)."
    ),
)
@click.option(
    "--detach-parent",
    "detach_parents",
    multiple=True,
    help=(
        "Parent ref whose ``<parent> -> ETYMON_REF`` descent edge is WRONG and "
        "should be deleted before the fold. Repeatable."
    ),
)
@click.option(
    "--detach-child",
    "detach_children",
    multiple=True,
    help=(
        "Child ref whose ``ETYMON_REF -> <child>`` descent edge is WRONG and "
        "should be deleted before the fold. Repeatable."
    ),
)
@click.option(
    "--variant-class",
    default="alternative",
    show_default=True,
    help="etymon_variant class recorded for the folded form.",
)
@click.option(
    "--reason",
    default=None,
    help="Operator note recorded for git-blame audit; doesn't affect the DB.",
)
@click.option(
    "--collapse-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/mining/_collapses.jsonl"),
    show_default=True,
    help="JSONL ledger to append the collapse event to.",
)
def lexicon_curate_collapse_etymon(
    etymon_ref: str,
    into_ref: str | None,
    detach_parents: tuple[str, ...],
    detach_children: tuple[str, ...],
    variant_class: str,
    reason: str | None,
    collapse_file: Path,
) -> None:
    """Hand-curate a collapse: detach wrong-sense descent edges and/or fold
    ETYMON_REF into a surviving lemma (wyrd-qp9c).

    ETYMON_REF is ``<language>:<canonical_form>`` (e.g. ``old-english:don``).

    At least one of ``--into`` or ``--detach-parent`` / ``--detach-child``
    must be given (omitting all three is a usage error). ``--into ''``
    (explicit empty) is allowed and emits a REVERT of a prior fold of
    ETYMON_REF (last-write-wins per ref), the documented un-collapse path.

    The detach endpoints are deleted from ``etymon_descent`` BEFORE the
    fold, so cluster-cognates (which resolves a tombstone's edges through
    ``merged_into_id``) never redirects the wrong-sense lineage onto the
    surviving lemma. Validation of the refs happens at apply time —
    ``lexicon enrich`` (dry-run) reports any unresolved endpoints in its
    collapse telemetry; re-emit the row to correct a typo (last-write-wins
    per ref).
    """
    # `into_ref is None` (option omitted), NOT `not into_ref`: an explicit
    # `--into ''` is a legitimate revert event and must pass the guard.
    if into_ref is None and not detach_parents and not detach_children:
        raise click.UsageError(
            "Provide at least one of --into, --detach-parent, or --detach-child."
        )

    payload: dict = {
        "_type": "collapse",
        "ref": etymon_ref,
        "into": into_ref or "",
        "method": "manual-homograph-detach",
    }
    if into_ref:
        payload["variant_class"] = variant_class
    if detach_parents:
        payload["detach_parents"] = list(detach_parents)
    if detach_children:
        payload["detach_children"] = list(detach_children)
    if reason:
        payload["reason"] = reason

    _append_event(collapse_file, payload)
    detached = len(detach_parents) + len(detach_children)
    if into_ref:
        fold_desc = f" → `{into_ref}`"
    elif into_ref == "":
        fold_desc = " (revert)"
    else:
        fold_desc = " (detach-only)"
    click.echo(
        f"Appended collapse event for `{etymon_ref}`"
        + fold_desc
        + (f" detaching {detached} edge(s)" if detached else "")
        + f" → {collapse_file}",
        err=True,
    )


def add_to(parent: click.Group) -> None:
    """Register ``curate-collapse-etymon`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_curate_collapse_etymon)
