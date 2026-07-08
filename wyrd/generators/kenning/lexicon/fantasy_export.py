"""Fantasy-morpheme inventory export (consumed by the bundle build).

``collect_fantasy_morphemes`` walks ``fantasy_morpheme`` rows where
``usable = 1`` (etymon successfully resolved during curation).
Tombstone-routing is non-destructive: a ``LEFT JOIN`` through
``etymon.merged_into_id`` follows the OCR-cluster chain so a
post-link merge transparently routes the row to its winner's
canonical_form via ``COALESCE`` instead of dropping the row. For
each row it pulls glosses from ``etymon_gloss`` and pre-computes
per-target-language era reflexes via ``etymon_era_reflexes`` so
the runtime KenningWeaver can render creature names at user-
chosen eras without a DB.

Output shape (wyrd-vz7f): ``{input_name: {input_name, etymon_id,
language, canonical_form, english_shaped, glosses, citation,
era_reflexes}}``. ``era_reflexes`` follows the wyrd-jbcu source-
aware schema ``{lang: [{form, source}, ...]}``.
"""

from __future__ import annotations

from typing import Any

from wyrd.generators.kenning.lexicon.bundle._family import (
    _better_era_reflex_source,
    _case_fold_reflex_entries,
    _is_derived_name_pollution,
    _modern_stage_languages,
)
from wyrd.generators.kenning.lexicon.db import LexiconDB
from wyrd.generators.kenning.lexicon.era_reflex import etymon_era_reflexes


def collect_fantasy_morphemes(db: LexiconDB) -> dict[str, dict[str, Any]]:
    """wyrd-vz7f: project usable ``fantasy_morpheme`` rows into a
    bundle-shaped lookup keyed by ``input_name`` so the runtime
    generator can surface creature etymology without a DB.

    Walks ``fantasy_morpheme`` rows where ``usable=1`` (etymon
    successfully resolved). A ``LEFT JOIN`` through
    ``etymon.merged_into_id`` follows the OCR-cluster chain so
    tombstone rows route to their winner's canonical_form via
    ``COALESCE`` instead of being dropped — the user's "Harpy"
    → ancient-greek ἅρπυια lookup stays semantically valid even
    after a post-link OCR merge moves the canonical voice. Joins
    canonical_form / language / english_shaped from ``etymon`` and
    pulls glosses from ``etymon_gloss``. Era reflexes are computed
    via ``etymon_era_reflexes`` for every target language in the
    creature's family chain — same precision as the toponym path,
    so a creature whose linked etymon is in old-english (Angel →
    OE 'Engel') surfaces ME / EModE / ModE forms when they exist.

    Returns ``{}`` when the ``fantasy_morpheme`` table is missing
    (older DBs predate wyrd-ami) — defensive against the same shape
    of failure that wyrd-c1vq fixed for ``toponym_decomposition``.

    Output shape:
    ``{input_name: {input_name, etymon_id, language, canonical_form,
                    english_shaped, glosses, citation, era_reflexes}}``.
    ``era_reflexes`` follows the wyrd-jbcu source-aware schema
    (``{lang: [{form, source}, ...]}``).
    """
    table_exists = db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='fantasy_morpheme'"
    ).fetchone()
    if not table_exists:
        return {}

    from wyrd.generators.kenning.era.cells import (
        CANONICAL_LANGUAGE_FOR_CELL,
        language_family,
    )

    # Follow the merged_into chain so OCR-cluster losers route to their
    # winner's canonical_form. wyrd-ami linked at a moment in time;
    # post-link OCR clustering may have moved the canonical voice to
    # the merge winner, but the user's "Harpy" → ancient-greek ἅρπυια
    # lookup is still semantically valid. The COALESCE pattern matches
    # the etymon_consensus view's two-step rollup.
    #
    # ``english_shaped`` semantics: when the winner's column is NULL
    # but the loser's was populated, the COALESCE returns NULL —
    # winner-authoritative per D22's sacred-evidence rule. The loser's
    # english_shaped was a property of the loser's specific form, not
    # the cluster's; it stays attached to the loser row but isn't
    # promoted to the winner's voice.
    rows = db.conn.execute(
        """
        SELECT fm.input_name,
               COALESCE(target.id, e.id) AS etymon_id,
               COALESCE(target.canonical_form, e.canonical_form) AS canonical_form,
               COALESCE(target.language, e.language) AS language,
               COALESCE(target.english_shaped, e.english_shaped) AS english_shaped,
               fm.citation
          FROM fantasy_morpheme fm
          JOIN etymon e ON e.id = fm.etymon_id
          LEFT JOIN etymon target ON target.id = e.merged_into_id
         WHERE fm.usable = 1
         ORDER BY fm.input_name
        """
    ).fetchall()
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        etymon_id = row["etymon_id"]
        glosses = [
            r["gloss"]
            for r in db.conn.execute(
                "SELECT DISTINCT gloss FROM etymon_gloss WHERE etymon_id = ? ORDER BY gloss",
                (etymon_id,),
            )
        ]
        # Per-target-language era reflexes: same precision as the
        # toponym path's _fetch_family_era_reflexes (cluster + descent +
        # period-form + phonology-rule via etymon_era_reflexes).
        family = language_family(row["language"])
        era_reflexes: dict[str, list[dict[str, str]]] = {}
        if family is not None:
            target_languages = sorted(
                {
                    lang
                    for (fam, _cell), lang in CANONICAL_LANGUAGE_FOR_CELL.items()
                    if fam == family
                }
            )
            modern_languages = _modern_stage_languages()
            for target_language in target_languages:
                refs = etymon_era_reflexes(db, etymon_id, target_language=target_language)
                if target_language in modern_languages:
                    # wyrd-rogd.11: same derived-proper-noun filter as the
                    # toponym path so fantasy/creature bundles don't carry
                    # West Ham / Wesley / Düne in their modern stage either.
                    refs = [r for r in refs if not _is_derived_name_pollution(r.form)]
                if not refs:
                    continue
                # Same dedupe-by-form-keep-best-source pattern as
                # _fetch_family_era_reflexes (wyrd-jbcu).
                best: dict[str, str] = {}
                for r in refs:
                    existing = best.get(r.form)
                    if existing is None or _better_era_reflex_source(r.source, existing):
                        best[r.form] = r.source
                # D55: case is render-only — fold the fantasy path's reflex
                # surfaces too (same choke point as the toponym emit). etymon
                # canonical_forms carry case at rest (wyrd-n2j6), so without this
                # the creature/fantasy bundle leaks capitalized reflexes.
                era_reflexes[target_language] = _case_fold_reflex_entries(
                    [{"form": form, "source": best[form]} for form in sorted(best)]
                )
        out[row["input_name"]] = {
            "input_name": row["input_name"],
            "etymon_id": etymon_id,
            "language": row["language"],
            "canonical_form": row["canonical_form"],
            "english_shaped": row["english_shaped"] or "",
            "glosses": glosses,
            "citation": row["citation"] or "",
            "era_reflexes": era_reflexes,
        }
    return out
