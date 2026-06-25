"""Seed a lexicon DB from a bundled meanings.json document.

``seed_from_meanings`` is the one entry point. The Rando-port seed
(``data/meanings.json``) is ingested under ``source_id='rando-port'``;
subsequent mining passes layer scholarly citations on top of the same
etymon rows. Called from ``wyrd kenning lexicon build`` and from
tests that need a populated DB.
"""

from __future__ import annotations

from typing import Any

from wyrd.generators.kenning.lexicon.constants import LANGUAGE_FIELDS, position_from_usage
from wyrd.generators.kenning.lexicon.db import LexiconDB
from wyrd.generators.kenning.lexicon.morpheme_surface import normalize_morpheme_surface


def _seed_subject_etymons(
    db: LexiconDB,
    subject: dict[str, Any],
    source_id: str,
    counts: dict[str, int],
) -> dict[tuple[str, str], int]:
    """Collect every (form, language) tuple this subject mentions across its
    ``words[]`` and upsert one shared etymon per tuple, attaching the
    subject's glosses + tags + a ``source_id`` citation to each new etymon.

    Returns the ``(form, lang) -> etymon_id`` map and mutates ``counts``.
    """
    glosses: list[str] = subject.get("meaning", []) or []
    tags: list[str] = subject.get("modifier_tags", []) or []
    modifier_type: str | None = subject.get("modifier_type")
    words: list[dict[str, Any]] = subject.get("words", []) or []
    etymons_in_subject: dict[tuple[str, str], int] = {}
    for word in words:
        for json_field, lang_code in LANGUAGE_FIELDS.items():
            forms = word.get(json_field) or []
            for form in forms:
                # De-dash the seed form (wyrd-aicu.8, D45): a junk form (strips
                # to empty) is dropped before the upsert choke would raise; a
                # dashed-but-real form is normalized so the per-subject dedup key
                # (and the stored row) is the bare surface — a dashed + bare form
                # of the same morpheme in one subject must not fork into two keys.
                normalized_form = normalize_morpheme_surface(form)
                if normalized_form is None:
                    continue
                form = normalized_form
                key = (form, lang_code)
                if key in etymons_in_subject:
                    continue
                etymon_id = db.upsert_etymon(form, lang_code, modifier_type=modifier_type)
                etymons_in_subject[key] = etymon_id
                counts["etymons"] += 1
                for gloss in glosses:
                    db.add_gloss(etymon_id, gloss)
                    counts["glosses"] += 1
                for tag in tags:
                    db.add_tag(etymon_id, tag)
                    counts["tags"] += 1
                db.add_citation(etymon_id, source_id)
                counts["citations"] += 1
    return etymons_in_subject


def _link_subject_reflexes(
    db: LexiconDB,
    subject: dict[str, Any],
    etymons_in_subject: dict[tuple[str, str], int],
    counts: dict[str, int],
) -> None:
    """Each word gives one reflex (from ``modern_usage``); link it to every
    etymon implied by the language fields on the same word. Mutates ``counts``."""
    words: list[dict[str, Any]] = subject.get("words", []) or []
    for word in words:
        modern_usage = word.get("modern_usage")
        if not modern_usage:
            continue
        # Position is derived from the (dash-decorated) input modern_usage, but
        # the surface is stored BARE — D45/wyrd-aicu.3: position lives in the
        # reflex.position column, never dash-encoded in the identity. Use the
        # shared ``normalize_morpheme_surface`` choke (as the etymon path above
        # already does) rather than a bespoke ``strip("-")``: it trims Unicode
        # boundary dashes + whitespace too, so reflex identity de-dashes
        # IDENTICALLY to etymon identity and to modern_reflex_import — a marker
        # drawn with an en-dash / U+2010 must not fork one morpheme into two
        # reflex rows. A form that strips to junk is dropped (no empty reflex).
        bare_surface = normalize_morpheme_surface(modern_usage)
        if bare_surface is None:
            continue
        position = position_from_usage(modern_usage)
        reflex_id = db.upsert_reflex(bare_surface, position)
        counts["reflexes"] += 1
        for json_field, lang_code in LANGUAGE_FIELDS.items():
            forms = word.get(json_field) or []
            for form in forms:
                etymon_id = etymons_in_subject.get((form, lang_code))
                if etymon_id is None:
                    continue
                db.link_reflex_etymon(reflex_id, etymon_id)
                counts["links"] += 1


def seed_from_meanings(
    db: LexiconDB,
    meanings_data: list[dict[str, Any]] | dict[str, Any],
    source_id: str,
) -> dict[str, int]:
    """Ingest a meanings.json document into the lexicon as <source_id>.

    Accepts either the legacy list-shape bundle OR the wyrd-q0g6 /
    wyrd-h8k1 dict-shape bundle ``{"subjects": [...], ...}`` — sibling
    fields (``joiners``, ``canonical_decompositions``) are ignored
    here since they don't seed lexicon rows.

    Each subject in the meanings.json contributes:
      - one etymon per (canonical_form, language) tuple that appears across
        its `words[]` entries
      - the subject's `meaning[]` glosses, attached to every etymon
      - the subject's `modifier_tags[]` tags, attached to every etymon
      - the subject's `modifier_type`, set on every etymon
      - one reflex per `modern_usage`, position-tagged from its dash markers
      - reflex→etymon links: a reflex points to every etymon implied by the
        language fields on the originating word
      - one citation row per etymon, pointing at <source_id>

    Returns a counts dict for sanity-checking.
    """
    # Deferred import: runtime/meaning.py is a runtime-layer module that the
    # lexicon package shouldn't pull at import time (Lambda cold-start
    # path). The seeder is operator-only, paid only when called.
    from wyrd.generators.kenning.runtime.meaning import _bundle_subjects

    counts = {
        "etymons": 0,
        "reflexes": 0,
        "links": 0,
        "glosses": 0,
        "tags": 0,
        "citations": 0,
    }

    for subject in _bundle_subjects(meanings_data):
        # First pass: collect every (form, lang) tuple this subject mentions,
        # so each one becomes a single etymon shared across its reflexes.
        etymons_in_subject = _seed_subject_etymons(db, subject, source_id, counts)
        # Second pass: each word gives one reflex; link it to every etymon
        # implied by the language fields on the same word.
        _link_subject_reflexes(db, subject, etymons_in_subject, counts)

    db.commit()
    return counts
