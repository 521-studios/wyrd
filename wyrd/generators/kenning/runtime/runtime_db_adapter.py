"""Adapter: L4 runtime DB → bundle-dict shape consumed by ``load_meanings``.

The L4 runtime DB (wyrd-d90t PR 1 + 4) stores per-usage rows whose
``data`` blob is the JSON shape ``{entries: [{meaning, modifier_tags,
modifier_type, word}]}`` that ``runtime_db_export._write_meanings``
emits. This module rehydrates those rows into the bundle-dict shape
``{subjects: [...], canonical_decompositions: ..., fantasy_morphemes:
..., joiners: ...}`` that the existing JSON-bundle loaders consume —
so the runtime can flip from the JSON bundle to the runtime DB
without touching ``load_meanings`` / ``load_canonical_decompositions``
/ ``load_fantasy_morphemes`` / ``load_joiners`` themselves.

The adapter synthesizes one pseudo-subject per L4 entry rather than
re-grouping entries back into multi-word subjects: ``load_meanings``
iterates per-word and the subject-level grouping doesn't affect the
final ``Meaning`` objects, so the simpler shape produces an
identical ``meaning_db``.

(Note: irish_anglicizations + Norman manorial families are still
loaded from JSON bundle sidecars by the caller — they're not in the
L4 runtime DB today. Folding them into the L4 emit is a follow-up;
filed under the d90t epic.)
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any


def bundle_dict_from_runtime_db(conn: sqlite3.Connection) -> dict[str, Any]:
    """Read every L4 table the bundle loaders care about and return a
    dict shaped like ``meanings.json``.

    Output keys (matching ``lexicon export-meanings`` output):
      * ``subjects`` — list of pseudo-subjects (one per L4 entry).
      * ``canonical_decompositions`` — dict[toponym_name → payload].
      * ``fantasy_morphemes`` — dict[input_name → payload].

    The ``joiners`` key is omitted (the L4 schema doesn't carry it
    today — it lived on a separate sidecar in the JSON-bundle world
    and never made it into the L4 emit).
    """
    return {
        "subjects": _read_subjects(conn),
        "canonical_decompositions": _read_canonical_decompositions(conn),
        "fantasy_morphemes": _read_fantasy_morphemes(conn),
    }


def _read_subjects(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Walk every meaning row + each row's ``data.entries[*]`` and
    synthesize a one-word pseudo-subject per entry.

    ``load_meanings`` iterates per-word and assigns subject-level
    metadata (meaning / modifier_tags / modifier_type) to each
    resulting Meaning — collapsing multi-entry rows into one-word
    subjects produces identical Meaning objects since the per-word
    iteration is what matters.

    Cursor-iterated rather than ``fetchall()`` so the meaning table's
    blobs (largest column in the L4) don't all land in memory at once
    on production-scale bundles.
    """
    subjects: list[dict[str, Any]] = []
    cursor = conn.execute("SELECT data FROM meaning ORDER BY usage_key")
    for (blob,) in cursor:
        payload = json.loads(blob)
        for entry in payload.get("entries") or []:
            word = entry.get("word")
            if not word:
                continue
            subjects.append(
                {
                    "meaning": entry.get("meaning") or [],
                    "modifier_tags": entry.get("modifier_tags") or [],
                    "modifier_type": entry.get("modifier_type"),
                    "words": [word],
                }
            )
    return subjects


def _read_canonical_decompositions(conn: sqlite3.Connection) -> dict[str, dict[str, str]]:
    """Project ``canonical_decomposition`` rows back into the
    ``{toponym_name: {signature, source}}`` map ``load_canonical_decompositions``
    expects. Cursor-iterated for the same memory reason as
    ``_read_subjects``."""
    cursor = conn.execute(
        "SELECT toponym_name, data FROM canonical_decomposition ORDER BY toponym_name"
    )
    return {name: json.loads(blob) for name, blob in cursor}


def _read_fantasy_morphemes(conn: sqlite3.Connection) -> dict[str, dict]:
    """Project ``fantasy_morpheme`` rows back into the
    ``{lowercase_input_name: payload}`` map ``load_fantasy_morphemes``
    expects. Keys come back lowercased per the L4 COLLATE NOCASE PK
    contract (the emitter already lowercases on write)."""
    cursor = conn.execute("SELECT usage_key, data FROM fantasy_morpheme ORDER BY usage_key")
    return {key: json.loads(blob) for key, blob in cursor}


def empirical_priors_payload_from_runtime_db(
    conn: sqlite3.Connection,
) -> dict[str, Any] | None:
    """Read the singleton ``empirical_priors`` row + return the parsed
    payload (same shape ``dump_empirical_priors_to_json`` writes /
    ``load_empirical_priors_from_payload`` consumes).

    Returns ``None`` when the row is absent — happens when the L3
    hadn't run ``mine-empirical-baselines`` at emit time. The runtime
    loader treats ``None`` as "use the empty ``EmpiricalPriors``".

    Schema mismatch (missing table from a pre-v2 DB) doesn't reach
    this code — ``runtime_db.get_runtime_db`` rejects pre-v2 DBs at
    connection-open time via
    :class:`runtime_db.RuntimeDBVersionMismatch`, so any DB this
    function sees is guaranteed to have the table.
    """
    row = conn.execute("SELECT data FROM empirical_priors WHERE id = 1").fetchone()
    if row is None:
        return None
    return json.loads(row[0])


def genitive_split_map_from_runtime_db(
    conn: sqlite3.Connection,
) -> dict[tuple[str, str], float] | None:
    """Read the singleton ``genitive_split_prior`` row + return the precomputed
    ``{(long_form, short_form): split_probability}`` float map the decomposition
    matcher consumes (wyrd-aicu.9, D-3). Mirrors
    :func:`empirical_priors_payload_from_runtime_db`.

    Returns ``None`` when the prior is unavailable — the row is absent (a
    wipe-without-remine L4, or one predating ``mine-genitive-priors --apply`` at
    emit time) OR the whole table is absent (a pre-aicu.9 seed: the table is
    ADDITIVE/OPTIONAL with no schema bump, so an older bundle simply lacks it).
    The runtime loader treats ``None`` as ``{}``: the connective still wins
    non-homograph coverage on score, only the homograph tiebreak goes silent.
    """
    try:
        row = conn.execute("SELECT data FROM genitive_split_prior WHERE id = 1").fetchone()
    except sqlite3.OperationalError:
        return None  # additive table absent on a pre-aicu.9 bundle (no schema bump)
    if row is None:
        return None
    payload = json.loads(row[0])
    return {
        (p["long_form"], p["short_form"]): p["split_probability"] for p in payload.get("pairs", [])
    }


def proportions_dict_for_culture(conn: sqlite3.Connection, culture: str) -> dict[str, Any]:
    """Build the per-culture proportions dict that ``load_proportions``
    consumes from the L4 ``proportions_*`` tables (wyrd-d90t PR 6).

    Output shape matches the JSON sidecar ``<culture>_proportions.json``:

    ::

        {
            "usages": {usage_key: weight, ...},
            "single_usages": {usage_key: weight, ...},
            "structures": [{"proportion": int, "words": [...]}, ...],
            "tag_marginal": {tag: weight, ...},
            "tag_cooccurrence": {"tagA|tagB": weight, ...},
            "attested_languages": {usage_key: [primary_language, ...], ...},
            "bare_word_positions": {position: {usage_key: weight, ...}, ...},
        }

    Why dict-shaped, not SQL-sampled: PR 6's contract is bit-identical
    generator output for a given seed across both backends. The
    generator's sampling logic runs on these dicts in Python; pushing
    the sampling into SQL would diverge bit-equal output for the same
    seed. A future PR could introduce a fast-path
    SQL sampler for the no-filter case (D38.3 sampling pattern) when
    the bit-equivalence contract relaxes.

    Iteration order matches the cumulative column (ascending) so the
    rehydrated dict's insertion order is the same order the emitter
    wrote rows in.
    """
    return {
        "usages": _read_proportions_usage_map(conn, "proportions_usage", culture),
        # D45: single_usages is bare-only (no within-word position axis); flatten
        # the nested read-back to ``{surface: weight}`` so the bare-only
        # consumers (``load_bare_word_positions``, which iterates flat
        # ``{usage: weight}``) get the shape they expect. ``load_parts`` tolerates
        # both via ``_iter_part_proportions`` either way.
        "single_usages": _flatten_single_usages(
            _read_proportions_usage_map(conn, "proportions_single_usage", culture)
        ),
        "structures": _read_proportions_structures(conn, culture),
        "tag_marginal": _read_proportions_tag_marginal(conn, culture),
        "tag_cooccurrence": _read_proportions_tag_cooccurrence(conn, culture),
        "attested_languages": _read_proportions_attested_languages(conn, culture),
        "bare_word_positions": _read_proportions_bare_word_positions(conn, culture),
    }


def _read_proportions_bare_word_positions(
    conn: sqlite3.Connection, culture: str
) -> dict[str, dict[str, int]]:
    """wyrd-rogd.13: read the per-word-position counts for bare words.
    Returns ``{position: {usage_key: weight}}`` (positions among
    pre/inner/post — the word's slot within the NAME, not the D39
    within-word morpheme position). Counts are raw; the load-time
    threshold (WYRD_BARE_POSITION_THRESHOLD) decides naked↔structured.
    Empty dict for L4 DBs built before the table existed — the runtime
    then skips the per-word-position buckets entirely and bare slots
    sample from the general single_usage distribution, pre-rogd.13
    behavior."""
    try:
        cursor = conn.execute(
            "SELECT position, usage_key, weight FROM proportions_bare_word_position "
            "WHERE culture = ? ORDER BY position, usage_key",
            (culture,),
        )
    except sqlite3.OperationalError as exc:
        # Only the missing-table shape is the legacy fallback. A locked DB,
        # disk I/O error, or corrupt page must surface loudly, not be
        # silently absorbed as "{}" (error-handling review, round 1).
        if "no such table" not in str(exc):
            raise
        return {}
    out: dict[str, dict[str, int]] = {}
    for position, usage_key, weight in cursor:
        out.setdefault(position, {})[usage_key] = weight
    return out


def _read_proportions_attested_languages(
    conn: sqlite3.Connection, culture: str
) -> dict[str, list[str]]:
    """wyrd-pfoo: read the per-culture per-Meaning attestation map.
    Returns ``{usage_key: [primary_language, ...]}`` so the runtime
    vector filter can narrow eligibility from the per-usage set to
    the per-(usage, primary_language) set. Empty dict for legacy L4
    DBs built before the table existed (defensive against re-import
    of old runtime DBs)."""
    try:
        cursor = conn.execute(
            "SELECT usage_key, primary_language FROM proportions_attested_language "
            "WHERE culture = ? ORDER BY usage_key, primary_language",
            (culture,),
        )
    except sqlite3.OperationalError as exc:
        # Missing-table only — see _read_proportions_bare_word_positions.
        if "no such table" not in str(exc):
            raise
        return {}
    out: dict[str, list[str]] = {}
    for usage_key, lang in cursor:
        out.setdefault(usage_key, []).append(lang)
    return out


# Tables _read_proportions_usage_map is allowed to query. Mirrors
# runtime_db_export._CUMULATIVE_USAGE_TABLES — the table name is
# f-string-interpolated into the SELECT, so the whitelist is the
# load-bearing safety check.
_USAGE_TABLES = frozenset({"proportions_usage", "proportions_single_usage"})


def _read_proportions_usage_map(
    conn: sqlite3.Connection, table: str, culture: str
) -> dict[str, Any]:
    """Read the part/single usage pool from a cumulative-shaped table.

    D45 (wyrd-aicu, schema v3+): the table carries an explicit ``position``
    column and bare-surface keys, so this returns the nested
    ``{surface: {position: weight}}`` shape ``load_parts`` consumes (position
    is an explicit axis, not a dash-encoded form).

    Defensive legacy read (schema v2 DBs, the ``proportions_attested_language``
    precedent): a ``no such column: position`` error means a pre-D45 bundle —
    fall back to the flat ``{dashed_usage: weight}`` shape, which ``load_parts``
    still tolerates via ``_iter_part_proportions``. Iteration order = cumulative
    ascending = the order the emitter wrote in.
    """
    if table not in _USAGE_TABLES:
        raise ValueError(
            f"_read_proportions_usage_map target {table!r} not in whitelist {sorted(_USAGE_TABLES)}"
        )
    try:
        cursor = conn.execute(
            f"SELECT usage_key, position, weight FROM {table} "
            f"WHERE culture = ? ORDER BY cumulative",
            (culture,),
        )
        nested: dict[str, dict[str, int]] = {}
        for usage_key, position, weight in cursor:
            nested.setdefault(usage_key, {})[position] = weight
        return nested
    except sqlite3.OperationalError as exc:
        if "no such column" not in str(exc):
            raise
        cursor = conn.execute(
            f"SELECT usage_key, weight FROM {table} WHERE culture = ? ORDER BY cumulative",
            (culture,),
        )
        return dict(cursor)


def _flatten_single_usages(single: dict[str, Any]) -> dict[str, int]:
    """Collapse the single-usage pool to flat ``{surface: weight}``. v3 DBs read
    back nested ``{surface: {'bare': weight}}`` (single usages are always bare —
    D45); sum across the (single) position. v2 DBs already return flat ints —
    pass through. The bare-only consumers (``load_bare_word_positions``) need
    flat; ``load_parts`` tolerates either."""
    out: dict[str, int] = {}
    for surface, value in single.items():
        out[surface] = sum(value.values()) if isinstance(value, dict) else value
    return out


def _read_proportions_structures(conn: sqlite3.Connection, culture: str) -> list[dict[str, Any]]:
    """Read the per-culture ``structures`` list. The emitter stored
    each row's ``words`` template as a JSON blob; decode here so the
    consumer sees the same nested-list shape as the source JSON."""
    cursor = conn.execute(
        "SELECT template, weight FROM proportions_structure WHERE culture = ? ORDER BY cumulative",
        (culture,),
    )
    return [{"proportion": weight, "words": json.loads(template)} for template, weight in cursor]


def _read_proportions_tag_marginal(conn: sqlite3.Connection, culture: str) -> dict[str, int]:
    cursor = conn.execute(
        "SELECT tag, weight FROM proportions_tag_marginal WHERE culture = ? ORDER BY tag",
        (culture,),
    )
    return dict(cursor)


def _read_proportions_tag_cooccurrence(conn: sqlite3.Connection, culture: str) -> dict[str, int]:
    """Rebuild the ``tag1|tag2`` pipe-delimited key the JSON sidecar
    used. The L4 emitter split on the pipe and stored tag1 / tag2 as
    separate columns; rejoin here so the consumer sees the same key
    shape as the JSON path."""
    cursor = conn.execute(
        "SELECT tag1, tag2, weight FROM proportions_tag_cooccurrence "
        "WHERE culture = ? ORDER BY tag1, tag2",
        (culture,),
    )
    return {f"{tag1}|{tag2}": weight for tag1, tag2, weight in cursor}
