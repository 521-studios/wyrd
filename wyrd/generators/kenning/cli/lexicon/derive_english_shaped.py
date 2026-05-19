"""``wyrd kenning lexicon derive-english-shaped`` — derive english-readable transliteration for non-Latin source forms (D31 Phase 2b)."""

from __future__ import annotations

from pathlib import Path

import click

from wyrd.generators.kenning.cli.utils import _DEFAULT_LEXICON_PATH
from wyrd.generators.kenning.english_shaping import (
    PHASE2A_NON_LATIN_LANGS as _PHASE2A_NON_LATIN_LANGS,
)
from wyrd.generators.kenning.english_shaping import derive_english_shaped
from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.paths import LEXICON_DB_DEFAULT_DISPLAY


@click.command("derive-english-shaped")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=_DEFAULT_LEXICON_PATH,
    show_default=LEXICON_DB_DEFAULT_DISPLAY,
)
@click.option(
    "--apply",
    "apply_changes",
    is_flag=True,
    default=False,
    help=(
        "Write derived english_shaped values to the etymon table. Without "
        "this flag, the command runs as a dry-run that prints what WOULD "
        "be written and the per-language coverage delta."
    ),
)
@click.option(
    "--language",
    "language_filter",
    type=str,
    default=None,
    help=(
        "Restrict derivation to a single language code (e.g. 'he', 'ar', "
        "'sa'). Useful for incremental smoke runs. When omitted, every "
        "non-Latin-script language with NULL english_shaped is processed."
    ),
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help=(
        "Process at most N rows. For smoke / sample runs before a full "
        "pass. Combined with --language to narrow further."
    ),
)
@click.option(
    "--reshape",
    is_flag=True,
    default=False,
    help=(
        "Re-derive english_shaped on rows that ALREADY have a value (not "
        "just NULL ones). Use after editing KNOWN_FORM_OVERRIDES or the "
        "diacritic-strip rules to push the new derivation through. Default "
        "behavior leaves existing values alone."
    ),
)
def lexicon_derive_english_shaped(
    db_path: Path,
    apply_changes: bool,
    language_filter: str | None,
    limit: int | None,
    reshape: bool,
) -> None:
    """Derive english_shaped values for non-Latin-script etymon rows
    (wyrd-ha9q Phase 2b).

    Walks every etymon row whose `english_shaped` is NULL (or all rows
    when ``--reshape`` is set), runs each through
    ``english_shaping.derive_english_shaped``, and UPDATEs the row when
    the derivation produced a non-None result. Latin-script source
    languages are skipped at the derivation level (the function returns
    None for them); the SQL filter narrows further to the wave-2
    non-Latin set so we don't sweep millions of OE / latin / etc. rows
    that wouldn't yield anything anyway.

    Mining-progress shape: one [N/total] line every 1,000 rows + a
    final summary, matching the convention in CLAUDE.md "Mining
    progress reporting".
    """
    # Build the WHERE predicate via parameterized SQL — never f-string
    # interpolate user-supplied values (--language) into the query.
    if language_filter:
        lang_pred = "language = ?"
        lang_params: tuple[str, ...] = (language_filter,)
    else:
        placeholders = ",".join("?" * len(_PHASE2A_NON_LATIN_LANGS))
        lang_pred = f"language IN ({placeholders})"
        lang_params = _PHASE2A_NON_LATIN_LANGS
    where_pred = lang_pred + ("" if reshape else " AND english_shaped IS NULL")
    # --limit: int(limit) is safe (Click already cast) and an int is
    # not interpolatable to anything dangerous.
    limit_clause = f" LIMIT {int(limit)}" if limit else ""

    with LexiconDB(db_path) as db:
        rows = db.conn.execute(
            f"""SELECT id, canonical_form, language,
                       transliteration, pronunciation_ipa
                FROM etymon
                WHERE {where_pred}
                ORDER BY language, id
                {limit_clause}""",
            lang_params,
        ).fetchall()

    total = len(rows)
    click.echo(
        f"derive-english-shaped: {total} candidate row(s) "
        f"({'applying' if apply_changes else 'dry-run'}; "
        f"reshape={'yes' if reshape else 'no'})",
        err=True,
    )

    written = 0
    skipped_no_input = 0
    by_language: dict[str, int] = {}
    progress_every = 1000

    db_writer = LexiconDB(db_path) if apply_changes else None
    try:
        for completed, row in enumerate(rows, start=1):
            shaped = derive_english_shaped(
                canonical_form=row["canonical_form"],
                language=row["language"],
                transliteration=row["transliteration"],
                pronunciation_ipa=row["pronunciation_ipa"],
            )
            if shaped is None:
                skipped_no_input += 1
            else:
                written += 1
                by_language[row["language"]] = by_language.get(row["language"], 0) + 1
                if db_writer is not None:
                    db_writer.conn.execute(
                        "UPDATE etymon SET english_shaped = ? WHERE id = ?",
                        (shaped, row["id"]),
                    )
                    if completed % progress_every == 0:
                        db_writer.commit()
            if completed % progress_every == 0 or completed == total:
                click.echo(
                    f"  [{completed}/{total}]  written={written} "
                    f"skipped_no_input={skipped_no_input}",
                    err=True,
                )
        if db_writer is not None:
            db_writer.commit()
    finally:
        if db_writer is not None:
            db_writer.close()

    click.echo(
        f"derive-english-shaped: {written} written, {skipped_no_input} skipped (no usable input).",
        err=True,
    )
    if by_language:
        click.echo("  per-language:", err=True)
        for lang, n in sorted(by_language.items(), key=lambda kv: -kv[1]):
            click.echo(f"    {lang}: {n}", err=True)
    if not apply_changes:
        click.echo("(dry-run; pass --apply to write english_shaped)", err=True)


def add_to(parent: click.Group) -> None:
    """Register ``derive-english-shaped`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_derive_english_shaped)
