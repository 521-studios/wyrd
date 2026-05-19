"""``wyrd kenning lexicon set-stratum`` — manually set an etymon stratum override (D32 idempotency contract)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("set-stratum")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--etymon-id",
    type=int,
    default=None,
    help=("Direct etymon row identifier. Mutually exclusive with --canonical-form / --language."),
)
@click.option(
    "--canonical-form",
    type=str,
    default=None,
    help=(
        "Canonical form to look up. Must be paired with --language. "
        "The (canonical_form, language) pair is unique in the etymon "
        "table, so the lookup yields 0 or 1 rows; missing rows error, "
        "found rows update."
    ),
)
@click.option(
    "--language",
    type=str,
    default=None,
    help=(
        "Language code paired with --canonical-form for the lookup. "
        "The lexicon language code (e.g. 'welsh', 'old-french', "
        "'gmq-osw'), not the human-readable name."
    ),
)
@click.option(
    "--stratum",
    type=str,
    default=None,
    help=(
        "New stratum value. Must be one of the known strata for the "
        "target row's language family — e.g. a French row accepts "
        "values from FRENCH_STRATA only (wyrd-j3gy family-aware "
        "validation). Languages without a classifier yet fall back "
        "to the cross-family ALL_STRATA typo-check. Mutually exclusive "
        "with --clear."
    ),
)
@click.option(
    "--clear",
    is_flag=True,
    default=False,
    help=("Set the row's stratum to NULL. Mutually exclusive with --stratum."),
)
def lexicon_set_stratum(
    db_path: Path,
    etymon_id: int | None,
    canonical_form: str | None,
    language: str | None,
    stratum: str | None,
    clear: bool,
) -> None:
    """Manually set the ``stratum`` value on a single etymon row.

    Hand-correction CLI (wyrd-lr4) for high-stakes etymons that
    ``classify-stratum`` got wrong. The classifier's heuristic
    (ancestor walk + self-language pass) misses some signals — most
    commonly when the ``etymon_descent`` graph lacks a parent edge
    that scholarly sources actually attest. Use this command to
    write the correct stratum for an individual row.

    Hand-corrections survive subsequent ``classify-stratum --apply``
    runs (without ``--force``); pass ``classify-stratum --force`` to
    deliberately overwrite them, or ``set-stratum --clear`` to opt
    one row back into bulk classification.

    The stratum value is checked against the row's language family
    (wyrd-j3gy family-aware validation): a French-language row
    accepts FRENCH_STRATA values, an Old English row accepts
    OLD_ENGLISH_STRATA values, etc. Languages without a classifier
    yet fall back to a cross-family ALL_STRATA typo-check so the
    operator can still hand-correct on unclassified languages.

    Note: this REVERSES Phase 4d's intentional cross-family
    tolerance (writing a Welsh stratum onto a French row used to
    succeed silently on the operator's say-so). The runtime
    --stratum filter validates per-culture, so the SPA needed the
    write-side validator to match — making a Welsh stratum on a
    French row produce an unfilterable result was the worse failure
    mode. Pass through ``classify-stratum --apply`` if a true
    cross-family override is needed (it bypasses set-stratum).
    """
    _validate_set_stratum_identification(etymon_id, canonical_form, language)
    _validate_set_stratum_value(stratum, clear)

    with LexiconDB(db_path) as db:
        row = _resolve_set_stratum_target(db, etymon_id, canonical_form, language)
        _validate_set_stratum_family_match(row, stratum)
        old_stratum = row["stratum"]
        new_stratum = None if clear else stratum
        db.conn.execute(
            "UPDATE etymon SET stratum = ? WHERE id = ?",
            (new_stratum, row["id"]),
        )
        db.commit()

    click.echo(
        f"etymon {row['id']} ({row['canonical_form']}, {row['language']}): "
        f"{old_stratum!r} → {new_stratum!r}",
        err=True,
    )


def _validate_set_stratum_identification(
    etymon_id: int | None,
    canonical_form: str | None,
    language: str | None,
) -> None:
    """Validate the mutually-exclusive --etymon-id vs --canonical-form
    + --language identification flags. Exits non-zero on misuse so
    Click's command-level handler doesn't have to know which row-
    lookup mode is selected. Three failure cases:

      1. Both modes together — ambiguous; pick one.
      2. Neither mode — no target.
      3. Pair mode partially supplied — canonical-form alone is
         ambiguous across languages.
    """
    by_id = etymon_id is not None
    by_pair = canonical_form is not None or language is not None
    if by_id and by_pair:
        click.echo(
            "Error: pass --etymon-id OR --canonical-form/--language, not both.",
            err=True,
        )
        sys.exit(1)
    if not by_id and not by_pair:
        click.echo(
            "Error: must specify either --etymon-id or --canonical-form + --language.",
            err=True,
        )
        sys.exit(1)
    if by_pair and (canonical_form is None or language is None):
        click.echo(
            "Error: --canonical-form and --language must both be passed "
            "(canonical_form alone can be ambiguous across languages).",
            err=True,
        )
        sys.exit(1)


def _validate_set_stratum_family_match(row: Any, stratum: str | None) -> None:
    """wyrd-j3gy family-aware validation. Runs after the target row
    is resolved: looks up the row's language family and verifies the
    stratum is in that family's STRATA tuple.

    Languages without a classifier (no entry in ``LANGUAGE_TO_FAMILY``)
    skip this check — the typo-check against ``ALL_STRATA`` already
    ran in ``_validate_set_stratum_value``, and the operator may
    legitimately want to hand-correct on an unclassified language.

    --clear callers pass ``stratum=None`` and skip this check
    entirely (clearing has no family-validity question).
    """
    from wyrd.generators.kenning.lexicon.strata import STRATA_BY_FAMILY, family_for_language

    if stratum is None:
        return  # --clear path — no family validation needed
    family = family_for_language(row["language"])
    if family is None:
        return  # unclassified language — fall back to ALL_STRATA typo-check
    valid = STRATA_BY_FAMILY[family]
    if stratum not in valid:
        click.echo(
            f"Error: stratum {stratum!r} is not valid for language "
            f"{row['language']!r} (family {family!r}). Valid: "
            f"{', '.join(valid)}.",
            err=True,
        )
        sys.exit(1)


def _validate_set_stratum_value(stratum: str | None, clear: bool) -> None:
    """Validate --stratum vs --clear. Mutually exclusive, one
    required, and the stratum value (when given) must be in the
    cross-family ALL_STRATA registry for typo protection. The
    failed-validation path exits non-zero with a friendly error."""
    from wyrd.generators.kenning.lexicon.strata import ALL_STRATA

    if stratum is not None and clear:
        click.echo(
            "Error: --stratum and --clear are mutually exclusive.",
            err=True,
        )
        sys.exit(1)
    if stratum is None and not clear:
        click.echo(
            "Error: must specify --stratum VALUE or --clear.",
            err=True,
        )
        sys.exit(1)
    if stratum is not None and stratum not in ALL_STRATA:
        # Sorted list of known strata for the error message — operator
        # gets a friendly menu rather than a typo silently writing.
        known = ", ".join(sorted(ALL_STRATA))
        click.echo(
            f"Error: unknown stratum {stratum!r}. Known: {known}.",
            err=True,
        )
        sys.exit(1)


def _resolve_set_stratum_target(
    db: LexiconDB,
    etymon_id: int | None,
    canonical_form: str | None,
    language: str | None,
) -> Any:
    """Look up the target etymon row by id or by (canonical_form,
    language) pair. The pair lookup leans on the etymon table's
    UNIQUE (canonical_form, language) constraint to guarantee 0 or
    1 matches. Exits non-zero with a friendly error when the row
    doesn't exist; warns and proceeds when the row is an OCR-cluster
    loser (merged_into_id IS NOT NULL) since the classifier filters
    those out and a hand-correction on a merged row is invisible to
    the bundle export."""
    if etymon_id is not None:
        row = db.conn.execute(
            "SELECT id, canonical_form, language, stratum, merged_into_id FROM etymon WHERE id = ?",
            (etymon_id,),
        ).fetchone()
        if row is None:
            click.echo(f"Error: no etymon with id={etymon_id}.", err=True)
            sys.exit(1)
    else:
        row = db.conn.execute(
            "SELECT id, canonical_form, language, stratum, merged_into_id FROM etymon "
            "WHERE canonical_form = ? AND language = ?",
            (canonical_form, language),
        ).fetchone()
        if row is None:
            click.echo(
                f"Error: no etymon with canonical_form={canonical_form!r} "
                f"and language={language!r}.",
                err=True,
            )
            sys.exit(1)

    # Warn-and-proceed for merged-cluster losers. classify_family
    # skips these (they're folded into a canonical winner); a
    # hand-correction on one survives in the column but doesn't
    # surface in the bundle export. Operators occasionally do this
    # intentionally (planning to undo the merge later), so don't
    # block — just flag.
    if row["merged_into_id"] is not None:
        click.echo(
            f"Warning: etymon {row['id']} is a merged-cluster loser "
            f"(merged_into_id={row['merged_into_id']}); the classifier "
            f"and bundle export both filter these rows out, so the "
            f"hand-correction may not surface as expected.",
            err=True,
        )
    return row


def add_to(parent: click.Group) -> None:
    """Register ``set-stratum`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_set_stratum)
