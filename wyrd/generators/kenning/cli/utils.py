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
import sqlite3
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path

import click

from wyrd.generators.kenning.lexicon import LexiconDB
from wyrd.generators.kenning.parsers.dictionary import (
    parse_alphabetical_text,
    parse_numbered_list_text,
)
from wyrd.generators.kenning.parsers.skeat import parse_skeat_text
from wyrd.generators.kenning.paths import default_lexicon_path
from wyrd.generators.kenning.runtime.decomposition import decompose_with_canonical
from wyrd.generators.kenning.runtime.name import Name

# The read-only connection type yielded by _readonly_lexicon, re-exported so
# cli/lexicon/*.py consumers can annotate against it WITHOUT importing sqlite3
# directly (the package-boundary rule — test_kenning_package_boundary).
Connection = sqlite3.Connection


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
    """Build the bundle dict the decompose pipeline consumes.

    With ``meanings`` supplied, reads it as a JSON-encoded bundle (for
    operators who want to decompose against a frozen / hand-edited
    bundle). Default ``None`` rehydrates the bundle from the L4
    runtime DB — d90t cutover moved the canonical bundle storage from
    on-disk JSON to SQLite, so the default source is now the runtime
    DB (bundled seed-runtime.db, or whatever ``WYRD_RUNTIME_DB`` /
    ``WYRD_RUNTIME_DB_BUCKET`` resolves to). To re-emit from L3 the
    operator pipes through ``wyrd kenning lexicon export-runtime-db``
    + ``WYRD_RUNTIME_DB`` rather than touching this helper.
    """
    if meanings is None:
        from wyrd.generators.kenning.runtime.runtime_db import get_runtime_db
        from wyrd.generators.kenning.runtime.runtime_db_adapter import (
            bundle_dict_from_runtime_db,
        )

        return bundle_dict_from_runtime_db(get_runtime_db())
    return json.loads(meanings.read_text(encoding="utf-8"))


def _decompose_corpus(
    name_entries: list,
    word_db: dict,
    db_path: Path | None,
    *,
    culture_languages: frozenset[str] | None = None,
    connective_inventory=None,
    genitive_prior=None,
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

    ``culture_languages`` (wyrd-pfoo) is forwarded to the heuristic
    matcher path so per-culture splits prefer culture-aligned tagged
    Meanings on ambiguous decompositions. The canonical-pick path is
    unaffected — scholar-attested decompositions carry their own
    correct sense and don't pass through the tiebreaker.

    ``connective_inventory`` / ``genitive_prior`` (wyrd-aicu.9) are
    forwarded to both the db (``decompose_with_canonical``) and no-db
    (``find_meaning``) heuristic paths so the genitive connective +
    homograph tiebreak fire on the live generation / proportions
    decompose. Both ``None`` (default) keeps the legacy bit-stable path.

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
                resolved, source = decompose_with_canonical(
                    name.name,
                    word_db,
                    db,
                    region=region,
                    culture_languages=culture_languages,
                    connective_inventory=connective_inventory,
                    genitive_prior=genitive_prior,
                )
                canonical_hits[source or "heuristic"] += 1
            else:
                resolved = name
                resolved.find_meaning(
                    word_db,
                    culture_languages=culture_languages,
                    connective_inventory=connective_inventory,
                    genitive_prior=genitive_prior,
                )
            resolved_names.append(resolved)
    return resolved_names, canonical_hits


# Default location for the authoring lexicon DB. The callable is
# resolved per CLI invocation so the WYRD_LEXICON_DB env override and
# the one-time legacy → ~/.wyrd migration both fire at call time, not
# at module-import time. See wyrd/generators/kenning/paths.py.
_DEFAULT_LEXICON_PATH = default_lexicon_path


@contextmanager
def _readonly_lexicon(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Yield a read-only ``sqlite3.Connection`` for browse / report
    subcommands, closing it on exit.

    Used as a context manager so the connection can't leak when a
    command early-exits via ``raise SystemExit``. The connection
    opens via the ``file:...?mode=ro`` URI form so any accidental
    writes fail loudly at the driver level rather than silently
    mutating the lexicon DB.
    """
    from urllib.parse import quote

    db_uri = f"file:{quote(str(db_path.absolute()))}?mode=ro"
    conn = sqlite3.connect(db_uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _append_remove_event(
    jsonl_dir: Path,
    source_id: str,
    row_type: str,
    ref: str,
    reason: str | None,
    ref_format_hint: str,
) -> Path:
    """Shared implementation for `lexicon prune-toponym` and
    `lexicon prune-etymon`. Validates that the source file exists and
    that the named ref actually resolves inside it before appending
    a remove event. Returns the target file path.

    The validation step matters because the kernel's
    `bucket.pop(ref, None)` silently swallows missing refs; combined
    with orphan-skip a typo would produce a successful-looking rebuild
    that doesn't actually prune anything.

    ``ref_format_hint`` is per-row-type guidance for the error message
    (etymon: ``lang:form``; toponym: ``name@region``).
    """
    target_file = jsonl_dir / f"{source_id}.jsonl"
    if not target_file.exists():
        raise click.UsageError(
            f"Source file not found: {target_file}. "
            f"Use --jsonl-dir to point at the right directory."
        )

    from wyrd.generators.kenning.jsonl.log import replay_file

    state = replay_file(target_file)
    if ref not in state.keyed[row_type]:
        raise click.UsageError(
            f"{row_type.capitalize()} ref {ref!r} not found in {target_file}. "
            f"Check spelling ({ref_format_hint}) and confirm the source "
            f"owns this row."
        )

    payload: dict[str, str] = {"_op": "remove", "_type": row_type, "ref": ref}
    if reason is not None:
        # Carried as a JSON key but ignored by the kernel (remove ops
        # discard payload beyond _type + ref); useful for git blame.
        payload["_reason"] = reason

    with target_file.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False))
        fh.write("\n")

    return target_file
