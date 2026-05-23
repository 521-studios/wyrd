"""``wyrd kenning lexicon cleanup-wiktionary-empirical`` — purge polluted
wiktionary-empirical etymons from the lexicon DB.

User-driven root-cause fix (wyrd-a106, 2026-05-23). The
``mine-wiktextract-corpus`` pipeline ingested every Wiktionary entry
that surface-matched an unaccounted place-name fragment, bypassing
the D4 ≥3-scholar-witness gate. Its ``_select_canonical_sense``
fallback kept first-redirect glosses ("better than skipping the
morpheme entirely" — that was the wrong call); the result was
~7500 garbage entries plus ~5500 modern_english homographs polluting
the bundle.

This cleanup is a ONE-SHOT operator action against the lexicon DB:
identify wiktionary-empirical-only etymons matching either of:

  * **derivative-gloss only** — every gloss attached to the etymon
    matches our ``_is_derivative_gloss`` classifier ("alternative
    form of X" / "plural of X" / "soft mutation of X" / ...).
  * **modern-english / middle-english only** — the etymon's
    language is modern English (any historical-stratum entry stays;
    these are modern homographs of place-name fragments and have
    no place in a HISTORICAL etymological lexicon).

Etymons cited by ANY OTHER source (real scholarly works, rando-port,
wiktionary-empirical PLUS another source) are LEFT ALONE. The
cleanup only touches entries where wiktionary-empirical is the
sole authority.

After cleanup, re-export the bundle with ``lexicon export-meanings``.
"""

from __future__ import annotations

from pathlib import Path

import click


# Modern languages that should NEVER be authoritative etymons for
# British place names. Historical languages (old-english,
# old-scandinavian, middle-english, etc.) are kept regardless of
# citation source. Middle-english is debatable but dry-run review
# 2026-05-23 showed legit place-name etymons in the ME wiktionary-
# empirical set (Kent, launde, helle, lough, bran, hund, port) mixed
# with noise (six, sex, and, now, open); pulling them wholesale
# would lose signal. Modern-english by definition is modern
# homographs of place-name fragments, not etymons themselves.
_MODERN_LANGUAGES = frozenset({"modern-english"})


@click.command("cleanup-wiktionary-empirical")
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Lexicon DB (default: $WYRD_LEXICON_DB or ~/.wyrd/lexicon.db).",
)
@click.option(
    "--apply",
    is_flag=True,
    default=False,
    help="Commit deletions. Without this flag, runs as a dry-run that "
    "reports counts + sample entries but does NOT modify the DB.",
)
@click.option(
    "--samples",
    type=int,
    default=20,
    show_default=True,
    help="How many sample dropped entries to print per category.",
)
def lexicon_cleanup_wiktionary_empirical(
    db_path: Path | None,
    apply: bool,
    samples: int,
) -> None:
    """Purge polluted wiktionary-empirical-only etymons.

    See module docstring for the why + heuristics. Always dry-run by
    default — pass ``--apply`` to commit.
    """
    import os
    from wyrd.generators.kenning import _is_derivative_gloss
    from wyrd.generators.kenning.lexicon.db import LexiconDB

    if db_path is None:
        env_path = os.environ.get("WYRD_LEXICON_DB")
        if env_path:
            db_path = Path(env_path)
        else:
            db_path = Path.home() / ".wyrd" / "lexicon.db"
        if not db_path.exists():
            raise click.ClickException(f"Lexicon DB not found at {db_path}. Pass --db to override.")

    with LexiconDB(db_path) as db:
        # Find every etymon whose ONLY citation source is
        # wiktionary-empirical. Etymons with additional sources
        # (scholarly works) stay regardless of language/gloss shape.
        wikt_only_ids = [
            row[0]
            for row in db.conn.execute(
                """
                SELECT etymon_id
                FROM etymon_citation
                GROUP BY etymon_id
                HAVING MIN(source_id) = 'wiktionary-empirical'
                   AND MAX(source_id) = 'wiktionary-empirical'
                """
            ).fetchall()
        ]
        click.echo(
            f"Found {len(wikt_only_ids)} etymons cited ONLY by wiktionary-empirical",
            err=True,
        )

        # Classify each into one of three buckets:
        #   - derivative_only (all glosses match _is_derivative_gloss)
        #   - modern_only (language is modern/middle English)
        #   - keep (has at least one semantic gloss + historical language)
        derivative_only: list[tuple[int, str, str, list[str]]] = []
        modern_only: list[tuple[int, str, str, list[str]]] = []
        # (id, canonical_form, language, glosses)
        for eid in wikt_only_ids:
            row = db.conn.execute(
                "SELECT canonical_form, language FROM etymon WHERE id = ?",
                (eid,),
            ).fetchone()
            if not row:
                continue
            canonical_form, language = row
            glosses = [
                g[0]
                for g in db.conn.execute(
                    "SELECT gloss FROM etymon_gloss WHERE etymon_id = ?",
                    (eid,),
                ).fetchall()
            ]
            if not glosses:
                # Etymon with no glosses at all is also garbage; treat
                # as derivative-only for cleanup purposes.
                derivative_only.append((eid, canonical_form, language, glosses))
                continue
            if all(_is_derivative_gloss(g) for g in glosses):
                derivative_only.append((eid, canonical_form, language, glosses))
            elif language in _MODERN_LANGUAGES:
                modern_only.append((eid, canonical_form, language, glosses))

        to_delete_ids = {row[0] for row in derivative_only} | {row[0] for row in modern_only}
        click.echo(
            f"  → {len(derivative_only)} derivative-only "
            f"+ {len(modern_only)} modern-english-only "
            f"= {len(to_delete_ids)} unique to delete",
            err=True,
        )
        click.echo(
            f"  ({len(wikt_only_ids) - len(to_delete_ids)} stay — semantic+historical)",
            err=True,
        )

        click.echo(f"\nSample DERIVATIVE-ONLY drops (first {samples}):", err=True)
        for eid, form, lang, glosses in derivative_only[:samples]:
            preview = " | ".join(glosses[:2])[:70]
            click.echo(f"  [{eid:>7}] {form:>14} ({lang:18}): {preview}", err=True)

        click.echo(f"\nSample MODERN-ENGLISH drops (first {samples}):", err=True)
        for eid, form, lang, glosses in modern_only[:samples]:
            preview = " | ".join(glosses[:2])[:70]
            click.echo(f"  [{eid:>7}] {form:>14} ({lang:18}): {preview}", err=True)

        if not apply:
            click.echo(
                "\nDRY RUN — no changes made. Pass --apply to commit.",
                err=True,
            )
            return

        # Most FKs to etymon are ON DELETE CASCADE (etymon_gloss,
        # etymon_tag, etymon_citation, etymon_text_match,
        # etymon_descent, etymon_variant, etymon_period_form,
        # etymon_meaning_synset, reflex_etymon). Two are NO ACTION
        # and would block the parent delete:
        #
        #   - etymon.lemma_id (self-reference) — set NULL on
        #     children whose lemma is being deleted; the children
        #     stay alive but become 'unattached' until a future
        #     clustering pass.
        #   - toponym_etymology_element.etymon_id — delete the
        #     element rows; they're bad toponym decompositions
        #     citing a garbage etymon, no value in keeping.
        click.echo(f"\nDeleting {len(to_delete_ids)} etymons...", err=True)
        ids_list = list(to_delete_ids)
        CHUNK = 500
        for i in range(0, len(ids_list), CHUNK):
            chunk = ids_list[i : i + CHUNK]
            placeholders = ",".join("?" for _ in chunk)
            # Pre-handle the two NO ACTION FKs.
            db.conn.execute(
                f"UPDATE etymon SET lemma_id = NULL WHERE lemma_id IN ({placeholders})",
                chunk,
            )
            db.conn.execute(
                f"DELETE FROM toponym_etymology_element WHERE etymon_id IN ({placeholders})",
                chunk,
            )
            db.conn.execute(f"DELETE FROM etymon WHERE id IN ({placeholders})", chunk)
        db.commit()
        click.echo(
            "Done. Re-export the bundle with: wyrd kenning lexicon export-meanings", err=True
        )


def add_to(parent: click.Group) -> None:
    """Register ``cleanup-wiktionary-empirical`` on the parent ``@lexicon`` group."""
    parent.add_command(lexicon_cleanup_wiktionary_empirical)
