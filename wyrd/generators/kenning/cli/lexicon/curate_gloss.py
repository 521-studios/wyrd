"""``wyrd kenning lexicon curate-suppress-gloss`` / ``curate-split-etymon`` —
append etymon_gloss_suppression / etymon_split events (wyrd-kutx)."""

from __future__ import annotations

import json
from pathlib import Path

import click

# wyrd-van9: single source of truth for the suffix charset constraint —
# imported from the applier so a future change can't drift one side
# without the other. The applier owns the constant because it's the
# authoritative defense layer (hand-edited JSONL bypasses the CLI but
# still hits the applier).
from wyrd.generators.kenning.enrichment import SUFFIX_PATTERN

_CURATION_SOURCE_ROW = {
    "_type": "source",
    "ref": "manual-curation",
    "title": "Operator curation overrides (wyrd-2jhs)",
    "notes": (
        "L3 enrichment overrides applied after the auto-clustering passes. "
        "Each event records a deliberate operator decision; reverting is "
        "appending another event with the cleared value (lemma_ref: '')."
    ),
}


def _ensure_curation_file(curation_file: Path) -> None:
    """Create the curation JSONL file with its synthetic source row if
    it doesn't exist yet — every L2 JSONL file needs exactly one
    source row."""
    if curation_file.exists():
        return
    curation_file.parent.mkdir(parents=True, exist_ok=True)
    with curation_file.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(_CURATION_SOURCE_ROW, ensure_ascii=False))
        fh.write("\n")


def _append_event(curation_file: Path, payload: dict) -> None:
    _ensure_curation_file(curation_file)
    with curation_file.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False))
        fh.write("\n")


@click.command("curate-suppress-gloss")
@click.argument("etymon_ref")
@click.argument("glosses", nargs=-1, required=True)
@click.option(
    "--reason",
    default=None,
    help=(
        "Operator note shared across all suppressed glosses in this event. "
        "Recorded for audit; doesn't affect the DB beyond visibility."
    ),
)
@click.option(
    "--curation-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/mining/_curation.jsonl"),
    show_default=True,
    help="JSONL file to append the suppression event to.",
)
def lexicon_curate_suppress_gloss(
    etymon_ref: str,
    glosses: tuple[str, ...],
    reason: str | None,
    curation_file: Path,
) -> None:
    """Drop named glosses from an etymon (wyrd-kutx).

    ETYMON_REF is ``<language>:<canonical_form>`` (e.g.
    ``old-english:dry``). One or more GLOSSES follow.

    Use case: Wiktionary-mined etymons whose entry conflates two
    senses where only one is used in place-names. OE ``dry`` carries
    "dry" + "wizard, sorcerer"; the wizard sense has no toponymy use,
    so drop it::

        wyrd kenning lexicon curate-suppress-gloss old-english:dry \\
            "wizard, sorcerer" --reason "no toponymy use"

    Mining evidence (citations, descent edges, toponym elements) stays
    attached to the etymon — only the gloss layer is touched.
    """
    suppressions = [{"gloss": g, **({"reason": reason} if reason else {})} for g in glosses]
    payload = {
        "_type": "etymon_gloss_suppression",
        "ref": etymon_ref,
        "suppressions": suppressions,
    }
    _append_event(curation_file, payload)
    click.echo(
        f"Appended gloss-suppression event for `{etymon_ref}` "
        f"({len(glosses)} gloss(es)) → {curation_file}",
        err=True,
    )


@click.command("curate-split-etymon")
@click.argument("etymon_ref")
@click.option(
    "--into",
    "into_specs",
    multiple=True,
    required=True,
    help=(
        "One per child: ``suffix=<s>|glosses=<g1>;<g2>|tags=<t1>;<t2>[|primary]``. "
        "Pass --into multiple times to create multiple children. "
        "Example: ``--into 'suffix=weir|glosses=Weir (fishing enclosure)|primary' "
        "--into 'suffix=year|glosses=year'``"
    ),
)
@click.option(
    "--reason",
    default=None,
    help="Operator note. Recorded for audit; doesn't affect the DB.",
)
@click.option(
    "--curation-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/mining/_curation.jsonl"),
    show_default=True,
    help="JSONL file to append the split event to.",
)
def lexicon_curate_split_etymon(
    etymon_ref: str,
    into_specs: tuple[str, ...],
    reason: str | None,
    curation_file: Path,
) -> None:
    """Split a conflated etymon into distinct child etymons (wyrd-kutx).

    ETYMON_REF is the original etymon. Each --into spec names one
    child via a pipe-delimited string with ``key=value`` fields. The
    children get canonical form ``<original-form>#<suffix>`` and ref
    ``<language>:<original-form>#<suffix>``::

        wyrd kenning lexicon curate-split-etymon old-english:gear \\
            --into 'suffix=weir|glosses=Weir (fishing enclosure)|primary' \\
            --into 'suffix=year|glosses=year' \\
            --reason "real homograph"

    Named glosses + tags are repointed from the parent to each child.
    Mining evidence (citations, descent edges, toponym_etymology_element
    refs) moves WHOLESALE to the ``primary=true`` child so it inherits
    the parent's witness count for bundle promotion (per D21 evidence is
    preserved, not destroyed). At most one child may be flagged
    ``primary``; if none is, the first child in the ``--into`` order
    becomes the default primary.
    """
    _ALLOWED_KEYS = {"suffix", "glosses", "tags"}
    into: list[dict] = []
    seen_primary = False
    for spec in into_specs:
        child: dict = {}
        for part in spec.split("|"):
            part = part.strip()
            if not part:
                continue
            if "=" in part:
                key, _, value = part.partition("=")
                key = key.strip()
                value = value.strip()
                if key not in _ALLOWED_KEYS:
                    raise click.UsageError(
                        f"--into spec {spec!r}: unknown key {key!r}; "
                        f"allowed keys: {sorted(_ALLOWED_KEYS)} "
                        "(plus the bare token 'primary')."
                    )
                if key in ("glosses", "tags"):
                    child[key] = [g.strip() for g in value.split(";") if g.strip()]
                else:
                    child[key] = value
            elif part == "primary":
                if seen_primary:
                    raise click.UsageError(
                        "More than one --into spec flagged 'primary'; at "
                        "most one child may be primary."
                    )
                child["primary"] = True
                seen_primary = True
            else:
                raise click.UsageError(
                    f"--into spec {spec!r}: unrecognized token {part!r}; "
                    "use key=value form or the bare token 'primary'."
                )
        suffix = child.get("suffix")
        if not suffix:
            raise click.UsageError(
                f"--into spec {spec!r}: missing or empty required ``suffix`` field."
            )
        # wyrd-van9: suffix charset validation. Children land at
        # canonical form ``<original-form>#<suffix>`` and ref
        # ``<language>:<original-form>#<suffix>``. A suffix carrying
        # ``:`` (the ref-grammar separator), ``#`` (the disambiguator),
        # or whitespace would corrupt downstream ref parsers that
        # naively split on those. Lock to ``[a-z0-9_-]+`` for human
        # readability + ref-safety.
        if not SUFFIX_PATTERN.fullmatch(suffix):
            raise click.UsageError(
                f"--into spec {spec!r}: suffix {suffix!r} contains "
                "disallowed characters; allowed charset is [a-z0-9_-]+ "
                "(lowercase alphanumeric, underscore, hyphen). Disambiguators "
                "like #/: would corrupt downstream ref parsing."
            )
        into.append(child)

    payload: dict = {
        "_type": "etymon_split",
        "ref": etymon_ref,
        "into": into,
    }
    if reason:
        payload["reason"] = reason
    _append_event(curation_file, payload)
    click.echo(
        f"Appended split event for `{etymon_ref}` ({len(into)} children) → {curation_file}",
        err=True,
    )


def add_to(parent: click.Group) -> None:
    """Register both subcommands on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_curate_suppress_gloss)
    parent.add_command(lexicon_curate_split_etymon)
