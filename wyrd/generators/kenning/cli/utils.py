"""Shared helpers used across the kenning CLI subcommand modules.

The kenning CLI is split into one module per click subcommand under
``wyrd/generators/kenning/cli/`` (wyrd-g143). Anything that more than
one subcommand needs lives here so each per-subcommand module can
stay focused on its own click decorators + command body.

Names starting with ``_`` are kept under that prefix even though they
are imported across modules in the cli package — the leading
underscore signals "package-internal helper, not part of the public
CLI surface." Tests that need to drive a helper directly
(``_select_parser_and_run`` via
``tests/test_kenning_dictionary_parser_numbered_list``;
``_decompose_corpus`` via ``tests/test_kenning_decomposition``)
import from the ``wyrd.generators.kenning.cli`` back-compat shim,
which re-exports the names from this module.
"""

from __future__ import annotations

import json
from collections import Counter
from contextlib import nullcontext
from importlib import resources
from pathlib import Path

from wyrd.generators.kenning.decomposition import decompose_with_canonical
from wyrd.generators.kenning.dictionary_parser import (
    parse_alphabetical_text,
    parse_numbered_list_text,
)
from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.name import Name
from wyrd.generators.kenning.paths import default_lexicon_path
from wyrd.generators.kenning.skeat_parser import parse_skeat_text


def _select_parser_and_run(
    text: str,
    parser: str,
    *,
    min_entry_number: int = 1,
    max_entry_number: int | None = None,
) -> list:
    """Pick the right parser for a book.

    Modes:
      skeat         - force Skeat-format parser
      alphabetical  - force alphabetical-dictionary parser
      numbered-list - force numbered-list parser (wyrd-5af). Right tool
                      for chrestomathy / treatise sources whose etymology
                      bodies are organized as ordinal-prefixed lists
                      (Longnon vol 2 saint-names: '1659. Caradocus :
                      Saint-Caradu (...)'). Skip on dictionary-shaped
                      sources — alphabetical is the right tool there.
      auto          - try Skeat first; if it returns ≥5 entries, use it.
                      Otherwise use alphabetical. ``auto`` does NOT pick
                      numbered-list — opt in explicitly because the
                      numbered-list shape co-exists with body prose in
                      treatise sources, and we don't want a few stray
                      ordinals to flip a Mawer-shaped book onto the
                      wrong parser.

    ``min_entry_number`` / ``max_entry_number`` apply ONLY to the
    numbered-list parser (the other modes don't have an ordinal axis).
    Inclusive bounds; default to "no filter" so the historic
    skeat/alphabetical paths stay bit-stable.
    """
    if parser == "skeat":
        return parse_skeat_text(text)
    if parser == "alphabetical":
        return parse_alphabetical_text(text)
    if parser == "numbered-list":
        return parse_numbered_list_text(
            text,
            min_entry_number=min_entry_number,
            max_entry_number=max_entry_number,
        )
    skeat_result = parse_skeat_text(text)
    if len(skeat_result) >= 5:
        return skeat_result
    return parse_alphabetical_text(text)


def _load_meanings_data(meanings: Path | None) -> dict:
    """Load meanings.json from an optional override path or the bundled default."""
    if meanings is None:
        text = resources.files("wyrd.generators.kenning.data").joinpath("meanings.json").read_text()
    else:
        text = meanings.read_text()
    return json.loads(text)


def _decompose_corpus(
    name_entries: list,
    word_db: dict,
    db_path: Path | None,
) -> tuple[list, Counter]:
    """Decompose a corpus of Names, honouring canonical picks when a
    lexicon DB path is supplied.

    ``name_entries`` is a list of ``(Name, region)`` pairs from
    ``load_names_with_regions`` (wyrd-8m87). Region is forwarded into
    the canonical lookup so same-modern_name-different-region
    collisions in the lexicon resolve to the right toponym row.
    Bare ``Name`` entries (legacy ``load_names`` callers) are accepted
    too — region defaults to ``None``, and the lookup fallback in
    ``decompose_with_canonical`` handles NULL-region DBs gracefully.

    Centralises (a) DB context-manager hygiene via ``nullcontext``,
    (b) the per-name canonical-vs-heuristic branch, (c) the
    ``canonical_hits`` Counter accumulation.

    Returns ``(resolved_names, canonical_hits)``. ``canonical_hits`` is
    empty when ``db_path`` is ``None`` so callers can use truthiness to
    decide whether to surface the ``canonical[...]`` summary block.
    """
    canonical_hits: Counter = Counter()
    resolved_names: list = []
    db_context = LexiconDB(db_path) if db_path is not None else nullcontext()
    with db_context as db:
        for entry in name_entries:
            if isinstance(entry, Name):
                name, region = entry, None
            else:
                name, region = entry[0], entry[1]
            if db is not None:
                resolved, source = decompose_with_canonical(name.name, word_db, db, region=region)
                canonical_hits[source or "heuristic"] += 1
            else:
                resolved = name
                resolved.find_meaning(word_db)
            resolved_names.append(resolved)
    return resolved_names, canonical_hits


# Default location for the authoring lexicon DB. The callable is
# resolved per CLI invocation so the WYRD_LEXICON_DB env override and
# the one-time legacy → ~/.wyrd migration both fire at call time, not
# at module-import time. See wyrd/generators/kenning/paths.py.
_DEFAULT_LEXICON_PATH = default_lexicon_path
