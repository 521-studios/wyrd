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
    # Deferred import: meaning.py is a runtime-layer module that the
    # lexicon package shouldn't pull at import time (Lambda cold-start
    # path). The seeder is operator-only, paid only when called.
    from wyrd.generators.kenning.meaning import _bundle_subjects

    counts = {
        "etymons": 0,
        "reflexes": 0,
        "links": 0,
        "glosses": 0,
        "tags": 0,
        "citations": 0,
    }

    for subject in _bundle_subjects(meanings_data):
        glosses: list[str] = subject.get("meaning", []) or []
        tags: list[str] = subject.get("modifier_tags", []) or []
        modifier_type: str | None = subject.get("modifier_type")
        words: list[dict[str, Any]] = subject.get("words", []) or []

        # First pass: collect every (form, lang) tuple this subject mentions,
        # so each one becomes a single etymon shared across its reflexes.
        etymons_in_subject: dict[tuple[str, str], int] = {}
        for word in words:
            for json_field, lang_code in LANGUAGE_FIELDS.items():
                forms = word.get(json_field) or []
                for form in forms:
                    key = (form, lang_code)
                    if key in etymons_in_subject:
                        continue
                    etymon_id = db.upsert_etymon(
                        form,
                        lang_code,
                        modifier_type=modifier_type,
                    )
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

        # Second pass: each word gives one reflex; link it to every etymon
        # implied by the language fields on the same word.
        for word in words:
            modern_usage = word.get("modern_usage")
            if not modern_usage:
                continue
            position = position_from_usage(modern_usage)
            reflex_id = db.upsert_reflex(modern_usage, position)
            counts["reflexes"] += 1
            for json_field, lang_code in LANGUAGE_FIELDS.items():
                forms = word.get(json_field) or []
                for form in forms:
                    etymon_id = etymons_in_subject.get((form, lang_code))
                    if etymon_id is None:
                        continue
                    db.link_reflex_etymon(reflex_id, etymon_id)
                    counts["links"] += 1

    db.commit()
    return counts
