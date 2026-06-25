"""L4 runtime SQLite DB emitter (wyrd-d90t).

Writes the runtime DB the Lambda loader pulls from S3 on cold start.
Source inputs:

  * the L3 lexicon DB (meanings, fantasy morphemes, canonical
    decompositions), passed in via the already-opened ``LexiconDB`` or
    its query helpers
  * the 5 per-culture proportions JSONs (read from the bundled data
    directory or any override path)

The schema lives in ``data/seed/runtime.sql``. The
runtime treats the emitted DB as read-only (``file:...?mode=ro`` URI);
re-emit and ship a new versioned key on S3 to update. See D38 for the
L4 architecture rationale.

This module lives in the ``lexicon/`` package per the
``test_kenning_package_boundary`` rule: anything opening raw sqlite3
connections (other than via ``_readonly_lexicon``) belongs in the
lexicon layer, not the CLI shim. The CLI side
(``cli/lexicon/export_runtime_db.py``) is a click-options wrapper that
delegates here.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import sqlite3
from collections import Counter
from collections.abc import Iterable
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any
from urllib.parse import quote

from wyrd.generators.kenning.lexicon.morpheme_surface import _BOUNDARY_DASHES
from wyrd.generators.kenning.paths import seed_data_path

_logger = logging.getLogger(__name__)

# Bumped when the L4 table shape changes incompatibly. The loader rejects
# DBs whose schema_version row doesn't match the runtime's expected
# version, forcing a re-download.
#
# v2 (2026-05-25): added empirical_priors singleton blob row so vector-
# scoring works out of the box without a separate priors.json sidecar.
# v3: D45 (wyrd-aicu) — bare-surface keys + explicit position column.
# wyrd-aicu.9: added the genitive_split_prior singleton blob row as an ADDITIVE,
# OPTIONAL table — the loader tolerates its absence (→ {}), so a pre-aicu.9 v3
# seed still loads and the homograph tiebreak just stays silent. Backward-
# compatible, so NO schema bump (the seed refresh that would populate it is a
# separate concern — the seed is stale since #629).
SCHEMA_VERSION = "3"  # genitive_split_prior is additive/optional — no bump (wyrd-aicu.9)

# Stamped into bundle_metadata so operators can correlate a deployed DB
# back to its emitter source.
EMITTER_VERSION = "wyrd-d90t"

# Per-culture proportions filenames. Same five files _load_culture reads
# today; the JSON cutover removes them from the bundle.
CULTURE_PROPORTIONS = (
    "english",
    "scottish",
    "welsh",
    "irish",
    "breton",
)

# wyrd-bo01.3: culture → the set of ``toponym.country`` values that belong
# to it. Unions the gazetteer's country names (England / The Isle of Man /
# Scotland / Wales / Northern Ireland / Republic of Ireland / Brittany) with
# the scholarly + KEPN rows' country variants (Ireland and France, plus the
# region-derived ``Isle of Man`` form distinct from the gazetteer's ``The
# Isle of Man``) — otherwise irish and breton silently lose the half of
# their corpus that came from the other source. Replaces the static
# {culture}_place_names.json corpus: the proportions corpus is now the DB's
# toponyms, filtered by culture.
CULTURE_COUNTRIES: dict[str, frozenset[str]] = {
    "english": frozenset({"England", "Isle of Man", "The Isle of Man"}),
    "scottish": frozenset({"Scotland"}),
    "welsh": frozenset({"Wales"}),
    "irish": frozenset({"Ireland", "Northern Ireland", "Republic of Ireland"}),
    "breton": frozenset({"Brittany", "France"}),
}


def _load_culture_toponyms(conn: sqlite3.Connection, culture: str) -> list[tuple[str, str | None]]:
    """The culture's toponyms from the L3 DB as ``(modern_name, region)``,
    deduped by surface name (first in ``(name, country, region)`` sort
    order wins) — matching ``load_names_with_regions``'s dedup contract so
    the matcher doesn't double-count a name shared across counties. Replaces
    the static ``{culture}_place_names.json`` load."""
    countries = CULTURE_COUNTRIES.get(culture)
    if not countries:
        return []
    placeholders = ",".join("?" * len(countries))
    rows = conn.execute(
        f"SELECT modern_name, country, region FROM toponym WHERE country IN ({placeholders})",
        tuple(sorted(countries)),
    ).fetchall()
    tagged = sorted((str(n), c or "", r or "") for n, c, r in rows)
    out: list[tuple[str, str | None]] = []
    seen: set[str] = set()
    for name, _country, region in tagged:
        if name in seen:
            continue
        seen.add(name)
        out.append((name, region or None))
    return out


# Sibling-suffix metadata keys on word entries that are NOT language axes.
# Mirrors the suffix set ``runtime/meaning.py:load_meanings`` strips off
# when building Meaning objects (per D26). Used by _pick_primary_language
# to exclude metadata from the per-usage language counts; ``_citations``
# is particularly load-bearing because it's a ``list[str]`` (source IDs)
# and would otherwise be misread as a language form array.
_NON_LANGUAGE_SUFFIXES = (
    "_variants",
    "_inflections",
    "_citations",
    "_attested_years",
    "_english_shaped",
    "_original_script",
    "_transliteration",
    "_pronunciation",
    "_stratum",
    "_phonological_vector",
)

# Tables whose ``_insert_cumulative`` writes are valid. Validation keeps
# the f-string-formatted INSERT honest if a future caller refactors and
# passes an external string by mistake.
_CUMULATIVE_USAGE_TABLES = frozenset({"proportions_usage", "proportions_single_usage"})

# Fixed timestamp stamped into ``bundle_metadata.built_at`` when --dev is
# active so the committed seed-runtime.db stays byte-stable across emits.
# Outside --dev the wall-clock timestamp is used (the production path
# wants the operational correlation; the dev seed wants byte-equality
# for the drift-detection test).
DEV_BUILT_AT = "1970-01-01T00:00:00Z"

# Sentinel stamped into ``bundle_metadata.source_lexicon_db`` under --dev.
# The wall-clock path resolves to the operator's local lexicon (~/.wyrd/
# lexicon.db on one machine, /var/folders/... on another) which would
# defeat cross-environment byte-stability; the sentinel keeps the dev
# emit reproducible across operators + CI.
DEV_SOURCE_LEXICON_SENTINEL = "DEV_SOURCE_LEXICON"

# Default per-culture cap on usages / single_usages in --dev mode. Sized
# to keep the committed seed-runtime.db under ~5MB while still covering
# enough of the morpheme inventory for the test suite + local dev to
# exercise the generator surface end-to-end.
DEV_TOP_N_PER_CULTURE = 200


def _bare_modern_usage(value: str) -> str:
    """The stored bare surface for a subject word's ``modern_usage``: dashes are
    render-time decoration (D45/D39) and surrounding whitespace is never part of
    a morpheme's identity (wyrd-an8u), so both are stripped. Keeps the meaning /
    dormant-morpheme blob keys bare + whitespace-clean, so a dirty ``'Oak- '``
    merges into the ``'Oak'`` meaning row rather than duplicating it and
    splitting its weight. Mirrors ``word._bare_surface`` (the proportions key
    path)."""
    return value.replace("-", "").strip()


def write_runtime_db(
    *,
    output_path: Path,
    subjects: list[dict[str, Any]],
    fantasy_morphemes: dict[str, Any],
    canonical_decompositions: dict[str, dict[str, str]],
    proportions_dir: Path | Traversable | None,
    empirical_priors_payload: dict[str, Any] | None = None,
    genitive_split_map: dict[tuple[str, str], float] | None = None,
    source_lexicon_db: Path,
    dev_subset: bool = False,
    generation_subset: bool = False,
    dev_top_n_per_culture: int = DEV_TOP_N_PER_CULTURE,
) -> dict[str, int]:
    """Emit the L4 SQLite DB at ``output_path``.

    Overwrites any existing file at the path. Returns a per-section
    rowcount dict suitable for an operator-facing summary line.

    ``proportions_dir`` controls where per-culture proportions come from:

      * ``None`` (default) — rebuild proportions inline by decomposing the
        bundled ``<culture>_place_names.json`` corpus against the just-
        exported ``subjects``. Self-contained: L3 in, L4 out, no
        intermediate JSON artifact required (d90t cutover state).
      * ``Path`` — a filesystem directory of ``<culture>_proportions.json``
        files (operator-supplied override, e.g. for diffing a frozen set
        against a fresh L3 emit).
      * ``Traversable`` — same as Path, but resolved from an
        importlib.resources backend (e.g. a zip-loaded package).

    ``dev_subset`` (PR 2 of wyrd-d90t): when True, run every input
    through :func:`select_dev_subset` first — keeps the top N usages /
    single_usages per culture (``dev_top_n_per_culture``) plus only the
    meanings actually referenced by the kept usages. ``built_at`` is
    pinned to a fixed sentinel so re-emits with identical inputs
    produce byte-equal output (drift detection on the committed
    ``seed-runtime.db``).
    ``generation_subset`` (wyrd-ukq0): the production cold-start trim. Runs
    the same referenced-subset filter as ``dev_subset`` but UNCAPPED (keeps
    every proportion-referenced usage, not just the top N), dropping only the
    decomposition-only corpus, and stamps REAL ``built_at`` / ``source_lexicon_db``
    (not the dev sentinels). Mutually exclusive with ``dev_subset``.
    """
    # Validate flag combination before any expensive proportions work.
    if dev_subset and generation_subset:
        raise ValueError(
            "dev_subset (tiny top-N seed) and generation_subset (full generatable "
            "set; decomposition-only corpus dropped) are mutually exclusive"
        )
    if proportions_dir is None:
        proportions_by_culture = _compute_proportions_inline(
            subjects, canonical_decompositions, source_lexicon_db, genitive_split_map
        )
    else:
        proportions_by_culture = _load_proportions(proportions_dir)

    if dev_subset or generation_subset:
        # Same referenced-subset filter for both; generation_subset passes
        # top_n=None to keep ALL proportion-referenced usages (no cap), dev
        # keeps the top-N per culture. Either way the decomposition-only
        # corpus (subjects no proportion references) is dropped. Metadata
        # sentinels below stay keyed on dev_subset only, so a generation_subset
        # DB carries the real built_at / source_lexicon_db.
        subjects, fantasy_morphemes, canonical_decompositions, proportions_by_culture = (
            select_dev_subset(
                subjects,
                fantasy_morphemes,
                canonical_decompositions,
                proportions_by_culture,
                top_n_per_culture=(dev_top_n_per_culture if dev_subset else None),
            )
        )

    output_path.unlink(missing_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        conn = sqlite3.connect(str(output_path))
        try:
            # Build-time emit perf: the DB is committed once at the end so
            # journaling and fsync are pure overhead during the write. The
            # outer try/except below removes any half-written file on
            # failure, so the runtime never sees a partial commit. Cache
            # size is in pages (~10MB at the default 1KB page size) —
            # plenty for the build's hot set.
            conn.execute("PRAGMA journal_mode = OFF")
            conn.execute("PRAGMA synchronous = OFF")
            conn.execute("PRAGMA cache_size = 10000")
            _init_runtime_schema(conn)
            # D45 (wyrd-aicu.1): de-dash every subject word's modern_usage ONCE
            # so BOTH blob tables (meaning + the dormant morpheme) store the bare
            # surface — the runtime keys meaning_db off word['modern_usage'], and
            # the render (D39) owns positional dashes + case. Stored case is kept
            # (initial-slot front-cap preserves internal name caps like McLeod).
            # Proportions are already computed above (they tally bare via
            # get_samples regardless), so this only affects the blob tables; the
            # mutation is export-local (subjects is not reused after this).
            for subject in subjects:
                for word in subject.get("words") or []:
                    mu = word.get("modern_usage")
                    if mu:
                        word["modern_usage"] = _bare_modern_usage(mu)
            n_meanings = _write_meanings(conn, subjects)
            n_morphemes = _write_morphemes(conn, subjects)  # wyrd-rogd.10 (dormant)
            n_fantasy = _write_fantasy_morphemes(conn, fantasy_morphemes)
            n_canonical = _write_canonical_decompositions(conn, canonical_decompositions)
            n_priors = _write_empirical_priors(conn, empirical_priors_payload)
            n_genitive_pairs = _write_genitive_split_prior(conn, genitive_split_map)
            proportion_counts = _write_proportions(conn, proportions_by_culture)
            _write_metadata(
                conn,
                source_lexicon_db=(
                    DEV_SOURCE_LEXICON_SENTINEL if dev_subset else str(source_lexicon_db.resolve())
                ),
                built_at=DEV_BUILT_AT if dev_subset else None,
            )
            # D45 (wyrd-aicu.5): refuse to ship a dash-marked stored identity.
            # Runs on every export so a future regression fails the BUILD loudly
            # instead of silently re-introducing the bug class D45 forbids.
            _verify_no_dashed_identity(conn)
            # Populate sqlite_stat tables so the runtime query planner has
            # accurate index statistics for the sampling queries (the L4
            # DB is read-only at runtime; ANALYZE pays off on every cold
            # start).
            conn.execute("ANALYZE")
            conn.commit()
        finally:
            conn.close()
    except BaseException:
        # Don't leave a half-written L4 on disk — a subsequent build
        # step (PR 3 uploader, PR 4 loader) would happily ship a
        # corrupt DB. missing_ok=True covers the case where the file
        # was never created (e.g. sqlite3.connect itself raised).
        output_path.unlink(missing_ok=True)
        raise

    return {
        "meanings": n_meanings,
        "morphemes": n_morphemes,
        "fantasy_morphemes": n_fantasy,
        "canonical_decompositions": n_canonical,
        "empirical_priors_native_cells": n_priors.get("native_cells", 0),
        "empirical_priors_loan_cells": n_priors.get("loan_cells", 0),
        "genitive_split_pairs": n_genitive_pairs,
        "cultures": len(proportions_by_culture),
        **proportion_counts,
    }


def _load_proportions(
    proportions_dir: Path | Traversable,
) -> dict[str, dict[str, Any]]:
    """Read all 5 per-culture proportions JSON files into memory
    (operator-supplied override path).

    Cultures whose file is absent are silently skipped — a missing
    file means that culture isn't part of this override (covers dev
    fixtures + partial emits).

    Accepts either a ``Path`` (operator override) or a Traversable.
    Iterates via ``joinpath()`` + ``is_file()`` — both shapes implement
    the Traversable protocol — so the source works under any
    importlib.resources backend including zip loaders.
    """
    out: dict[str, dict[str, Any]] = {}
    for culture in CULTURE_PROPORTIONS:
        entry = proportions_dir.joinpath(f"{culture}_proportions.json")
        if not entry.is_file():
            continue
        with entry.open(encoding="utf-8") as f:
            out[culture] = json.load(f)
    return out


def _compute_proportions_inline(
    subjects: list[dict[str, Any]],
    canonical_decompositions: dict[str, dict[str, str]],
    source_lexicon_db: Path,
    genitive_split_map: dict[tuple[str, str], float] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build per-culture proportions on the fly from the subjects we just
    exported + the culture's toponyms read from the L3 DB.

    wyrd-bo01.3: the proportions corpus is now the DB's ``toponym`` rows
    (filtered by :data:`CULTURE_COUNTRIES`), not the static
    ``{culture}_place_names.json`` gazetteer — so the gazetteer's own
    names PLUS every attested KEPN/scholarly toponym contribute, and the
    static JSON is retired from the runtime path. Operators still supply
    ``--proportions-dir`` to bypass this with a hand-edited or frozen set.

    ``canonical_decompositions`` lets the inline rebuild honor
    scholar-attributed picks the L3 carries (the ``is_canonical=1`` rows
    on ``toponym_decomposition``, already projected into the L4 emit by
    ``collect_canonical_decompositions``). Without it the rebuild would
    drop to heuristic-only matching, losing the canonical-pick lift
    that's the load-bearing reason ``decompose_with_canonical`` exists.
    Names without a canonical entry fall through to the
    ``find_meaning(reduce=True)`` heuristic path — same shape, just no
    scholar override.

    ``genitive_split_map`` (wyrd-aicu.9): the precomputed genitive-split
    prob-map. When supplied, the default connective inventory + this prior are
    threaded into the heuristic ``find_meaning(reduce=True)`` calls so the
    proportions train on the CORRECTED decompositions (``Bishop·s·tūn`` parses to
    the town head, not ``Bishop + ston`` stone). The canonical-pick reduce=False
    branch is untouched — scholar-attested decompositions carry their own correct
    attribution. ``None``/empty → bit-stable with the pre-connective inline build.
    """
    from wyrd.generators.kenning.lexicon.proportions_builder import (
        CULTURE_LANGUAGES,
        proportions_from,
    )
    from wyrd.generators.kenning.runtime.connective import DEFAULT_CONNECTIVE_INVENTORY
    from wyrd.generators.kenning.runtime.decomposition import apply_canonical_to_name
    from wyrd.generators.kenning.runtime.meaning import load_meanings
    from wyrd.generators.kenning.runtime.name import Name

    # wyrd-aicu.9: only engage the connective when a prior is present, so a
    # wipe-without-remine emit (empty map) stays bit-stable with the legacy path.
    connective_inventory = DEFAULT_CONNECTIVE_INVENTORY if genitive_split_map else None

    word_db, _ = load_meanings({"subjects": subjects})
    # wyrd-bo01.3: read the per-culture corpus from the L3 toponym table
    # (read-only) instead of the static {culture}_place_names.json gazetteer.
    # quote() the path so a directory containing ?/#/space doesn't corrupt
    # the file: URI (matches cli/utils._readonly_lexicon's convention).
    conn = sqlite3.connect(f"file:{quote(str(source_lexicon_db))}?mode=ro", uri=True)
    try:
        culture_toponyms = {c: _load_culture_toponyms(conn, c) for c in CULTURE_PROPORTIONS}
    finally:
        conn.close()
    out: dict[str, dict[str, Any]] = {}
    for culture in CULTURE_PROPORTIONS:
        toponyms = culture_toponyms[culture]
        if not toponyms:
            continue
        # wyrd-pfoo: thread the culture's expected-language set into
        # heuristic matcher fallbacks (canonical-pick path bypasses
        # the tiebreaker; only the reduce=True heuristic branch
        # consults it).
        culture_languages = CULTURE_LANGUAGES.get(culture)
        resolved = []
        for name_str, _region in toponyms:
            name = Name(name_str)
            canonical = canonical_decompositions.get(name.name)
            if canonical:
                # Populate every alternate (reduce=False) so
                # apply_canonical_to_name can pick the canonical cell
                # from the full cross-product. Fall back to the
                # heuristic on miss — happens when the bundle's
                # word_db has shifted since the canonical was picked.
                name.find_meaning(word_db, reduce=False)
                if not apply_canonical_to_name(name, canonical["signature"]):
                    # wyrd-pfoo: the reduce=False call above already
                    # populated ``name.words`` with every parse.
                    # ``find_meaning(reduce=True, ...)`` deduplicates
                    # against the existing entries via seen_keys, so
                    # the canonical-tiebreaker output (a subset of
                    # all_decompositions) gets filtered out as
                    # "already there" before the tiebreaker effect
                    # can land. Rebuilding from a fresh Name re-runs
                    # the matcher cleanly — same pattern as
                    # decomposition.decompose_with_canonical's
                    # canonical-miss fallback.
                    name = Name(name.name)
                    name.find_meaning(
                        word_db,
                        reduce=True,
                        culture_languages=culture_languages,
                        connective_inventory=connective_inventory,
                        genitive_prior=genitive_split_map,
                    )
            else:
                name.find_meaning(
                    word_db,
                    reduce=True,
                    culture_languages=culture_languages,
                    connective_inventory=connective_inventory,
                    genitive_prior=genitive_split_map,
                )
            resolved.append(name)
        good_names = [n for n in resolved if n.count_unaccounted() == 0]
        out[culture] = proportions_from(good_names)
    return out


def _init_runtime_schema(conn: sqlite3.Connection) -> None:
    """Run runtime.sql against the freshly-created DB."""
    sql = seed_data_path("runtime.sql").read_text()
    conn.executescript(sql)


# D45 (wyrd-aicu.5): the stored-identity surfaces the exporter guards. KEY
# columns are short bare surfaces, so an any-dash ``GLOB`` (``_ANY_DASH_GLOB``)
# is exact; the BLOB tables need a JSON parse (a greedy whole-blob match
# false-positives on dashes in OTHER fields).
#
# ``morpheme_id`` (the L3-derived content key ``language:form``) is guarded
# SEPARATELY (below) at the FORM level (wyrd-aicu.8) — NOT in this list —
# because its language PREFIX legitimately carries a dash (``old-english:``) and
# an interior in-word hyphen is legitimate (``al-Quadim``); only a BOUNDARY dash
# on the form is the affix-position decoration D45 forbids, so a blanket
# any-dash match here would false-flag every dash-language morpheme forever.
#
# One column is still deliberately ABSENT:
#   * ``fantasy_morpheme.usage_key`` — a whole input-NAME lookup key, never a
#     positionally dash-decorated morpheme surface, so it's outside D45.
_DASH_GUARD_KEY_COLUMNS = (
    ("meaning", "usage_key"),
    ("proportions_usage", "usage_key"),
    ("proportions_single_usage", "usage_key"),
    ("proportions_attested_language", "usage_key"),
    ("proportions_bare_word_position", "usage_key"),
)
_DASH_GUARD_BLOB_TABLES = ("meaning", "morpheme")

# A position-marker dash is any dash-like codepoint, not just the ASCII hyphen —
# the de-dash this guard backstops is Unicode-aware (``normalize_morpheme_surface``,
# #769), so a regression that left an en-dash / U+2010 boundary marker on a stored
# identity must trip the guard too, or it ships unseen. SQLite ``GLOB`` matches a
# ``[...]`` char-class by Unicode codepoint (unlike ``LIKE``, whose only dash is the
# ASCII one); the ASCII hyphen sits LAST in the class so it isn't read as a range.
# These are ASCII byte-identical to the old ``LIKE '%-%'`` / ``LIKE '-%'`` for
# ASCII-only data — they only ADD the Unicode variants.
_DASH_GLOB_CLASS = "".join(d for d in _BOUNDARY_DASHES if d != "-") + "-"
_ANY_DASH_GLOB = f"*[{_DASH_GLOB_CLASS}]*"  # a dash anywhere
_LEAD_DASH_GLOB = f"[{_DASH_GLOB_CLASS}]*"  # leading dash
_TRAIL_DASH_GLOB = f"*[{_DASH_GLOB_CLASS}]"  # trailing dash
_DASH_CHARS = frozenset(_BOUNDARY_DASHES)  # Python membership (str values)
_DASH_BYTES = tuple(d.encode("utf-8") for d in _BOUNDARY_DASHES)  # blob (bytes) prefilter

# morpheme_id is ``language:form``; flag a LEADING/TRAILING dash (ANY variant) on
# the FORM part (after the first ':') — the affix-position decoration
# (D45/wyrd-aicu.8). The language prefix and interior in-word hyphens are left
# alone (the boundary GLOBs match only the ends, never an interior ``al-Quadim``).
_MORPHEME_ID_FORM_DASH_SQL = (
    "SELECT COUNT(*) FROM morpheme "
    f"WHERE substr(morpheme_id, instr(morpheme_id, ':') + 1) GLOB '{_LEAD_DASH_GLOB}' "
    f"   OR substr(morpheme_id, instr(morpheme_id, ':') + 1) GLOB '{_TRAIL_DASH_GLOB}'"
)


def _count_dashed_blob_modern_usage(conn: sqlite3.Connection, table: str) -> int:
    """Count blob rows in ``table`` whose ``word.modern_usage`` (the identity)
    carries a dash of ANY variant. Full scan, but skip the JSON parse on any row
    whose RAW payload has no dash at all — if there's no dash anywhere, modern_usage
    can't have one. The prefilter is a Python substring check on the actual fetched
    value (bytes for this BLOB column — match the UTF-8 encoding of each dash
    codepoint), NOT a SQLite ``LIKE`` — a LIKE with a TEXT pattern silently matches
    ZERO blob rows and would neuter the guard (wyrd-aicu.5 round-3 regression). This
    check sees the real value regardless of storage class, so it can't under-match."""
    dashed = 0
    for (data,) in conn.execute(f"SELECT data FROM {table}"):
        if isinstance(data, bytes):
            if not any(db in data for db in _DASH_BYTES):
                continue
        elif not _DASH_CHARS.intersection(data):
            continue
        for entry in json.loads(data).get("entries", []):
            if _DASH_CHARS.intersection((entry.get("word") or {}).get("modern_usage") or ""):
                dashed += 1
    return dashed


def _verify_no_dashed_identity(conn: sqlite3.Connection) -> None:
    """D45 (wyrd-aicu.5) exporter guard: raise if any stored morpheme IDENTITY
    carries a dash — a morpheme's identity is its bare surface; position is an
    explicit column and dashes are render-time decoration (D39). Checks the
    bare-keyed usage columns (SQL) + the meaning/morpheme blob ``modern_usage``
    values (JSON parse) + the ``morpheme_id`` content key's FORM part (de-dashed
    by wyrd-aicu.8; boundary dash only — the language prefix / interior hyphens
    are legitimate). Runs on every emit so a regression fails the build, not the
    operator."""
    violations: list[str] = []
    for table, col in _DASH_GUARD_KEY_COLUMNS:
        n = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {col} GLOB ?", (_ANY_DASH_GLOB,)
        ).fetchone()[0]
        if n:
            violations.append(f"{table}.{col}: {n} dashed")
    # morpheme_id (language:form) — boundary dash on the FORM part only (D45/wyrd-aicu.8).
    n = conn.execute(_MORPHEME_ID_FORM_DASH_SQL).fetchone()[0]
    if n:
        violations.append(f"morpheme.morpheme_id form: {n} boundary-dashed")
    for table in _DASH_GUARD_BLOB_TABLES:
        dashed = _count_dashed_blob_modern_usage(conn, table)
        if dashed:
            violations.append(f"{table} blob modern_usage: {dashed} dashed")
    if violations:
        raise RuntimeError(
            "D45 (wyrd-aicu.5): refusing to emit dash-marked stored identity — "
            + "; ".join(violations)
            + ". A morpheme's stored identity must be its bare surface (position "
            "is an explicit column; dashes are render-time decoration, D39). "
            "morpheme_id is guarded at the form level (de-dashed by wyrd-aicu.8)."
        )


def _write_meanings(conn: sqlite3.Connection, subjects: list[dict[str, Any]]) -> int:
    """Insert per-surface meaning rows. D45 (wyrd-aicu): a morpheme's stored
    identity is its BARE surface — the up-to-4 dash-variant rows per surface
    (``-ton`` / ``Ton-`` / ``-ton-`` / ``ton``) merge into ONE row keyed by the
    bare surface, with their ``{meaning, modifier_tags, modifier_type, word}``
    entries UNIONED. ``primary_language`` / ``stratum`` are re-picked over the
    unioned entries (regroup-then-repick — divergent values across variants
    resolve consistently).

    Each word's ``modern_usage`` is already bare here — the caller
    (``write_runtime_db``) de-dashes every subject word once before this so the
    meaning + dormant morpheme blob tables agree. The grouping key is that bare
    surface; the runtime keys ``meaning_db`` off it and the render (D39) owns
    positional dashes + case.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for subject in subjects:
        subject_meta = {
            "meaning": subject.get("meaning") or [],
            "modifier_tags": subject.get("modifier_tags") or [],
            "modifier_type": subject.get("modifier_type"),
        }
        for word in subject.get("words") or []:
            usage = word.get("modern_usage")
            if not usage:
                continue
            entry = dict(subject_meta)
            entry["word"] = word
            grouped.setdefault(usage, []).append(entry)

    rows = []
    for usage_key, entries in grouped.items():
        primary_language = _pick_primary_language(entries)
        stratum = _pick_unanimous_stratum(entries)
        payload = json.dumps({"entries": entries}, ensure_ascii=False)
        rows.append((usage_key, primary_language, stratum, payload.encode("utf-8")))

    conn.executemany(
        "INSERT INTO meaning (usage_key, primary_language, stratum, data) VALUES (?, ?, ?, ?)",
        rows,
    )
    return len(rows)


def _write_morphemes(conn: sqlite3.Connection, subjects: list[dict[str, Any]]) -> int:
    """wyrd-rogd.10: emit the morpheme entity — the SAME per-word
    ``{meaning, modifier_*, word}`` entries as :func:`_write_meanings`, but
    regrouped by the family ``morpheme_id`` (root etymon id) instead of the
    connective ``modern_usage`` string. One morpheme thus owns all its surface
    variants (``-ing-``/``-ing``/``Ing-`` → one record), which is what lets the
    Phase-2 runtime follow a stable id instead of dash-stripping a surface.

    Words without a ``morpheme_id`` (un-attributable) are skipped — they stay in
    the usage-keyed ``meaning`` table. Dormant in Phase 1 (no runtime reader).
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for subject in subjects:
        subject_meta = {
            "meaning": subject.get("meaning") or [],
            "modifier_tags": subject.get("modifier_tags") or [],
            "modifier_type": subject.get("modifier_type"),
        }
        for word in subject.get("words") or []:
            morpheme_id = word.get("morpheme_id")
            if morpheme_id is None:
                continue
            entry = dict(subject_meta)
            entry["word"] = word
            grouped.setdefault(morpheme_id, []).append(entry)

    rows = []
    # Sort by morpheme_id so the L4 row order is byte-stable independent of the
    # subject/word iteration order — belt-and-suspenders for the discard-and-
    # rebuild reproducibility guarantee (morpheme_id is a content key).
    for morpheme_id in sorted(grouped):
        entries = grouped[morpheme_id]
        primary_language = _pick_primary_language(entries)
        stratum = _pick_unanimous_stratum(entries)
        payload = json.dumps({"entries": entries}, ensure_ascii=False)
        rows.append((morpheme_id, primary_language, stratum, payload.encode("utf-8")))

    conn.executemany(
        "INSERT INTO morpheme (morpheme_id, primary_language, stratum, data) VALUES (?, ?, ?, ?)",
        rows,
    )
    return len(rows)


def _pick_primary_language(entries: list[dict[str, Any]]) -> str | None:
    """Dominant language across all entries grouped under one usage.

    Counts language-form-array keys across every word; ties broken
    alphabetically so re-emits with identical inputs produce identical
    rows. Returns ``None`` when no language axis is present (orphan-
    reflex entries from ``_orphan_reflex_subjects`` carry only
    ``modern_usage`` — they're matcher-only and have no etymon-backed
    language; the schema column is nullable to express this category
    rather than masking it with an empty string).

    Excludes the metadata sibling-suffix set (``_citations``,
    ``_attested_years``, ``_variants``, etc. — see
    ``_NON_LANGUAGE_SUFFIXES``) so list-of-strings siblings like
    ``celtic_mix_citations`` aren't mis-counted as a language axis.
    """
    counts: Counter[str] = Counter()
    for entry in entries:
        word = entry.get("word") or {}
        for key, value in word.items():
            if key in ("modern_usage", "era_reflexes"):
                continue
            if any(key.endswith(suffix) for suffix in _NON_LANGUAGE_SUFFIXES):
                continue
            if not isinstance(value, list):
                continue
            counts[key] += 1
    if not counts:
        return None
    most = counts.most_common()
    top_count = most[0][1]
    tied = sorted(k for k, v in most if v == top_count)
    return tied[0]


def _strata_for_word(word: dict[str, Any]) -> Iterable[str]:
    """Yield every per-form stratum value declared on a word.

    Walks the ``<lang>_stratum`` sibling-suffix fields (D32) and emits
    the ``"stratum"`` value of each entry. Empty / missing strata are
    skipped. Used by ``_pick_unanimous_stratum`` to keep the unanimity
    check flat (depth 2) instead of nesting 5 deep.
    """
    for key, value in word.items():
        if not key.endswith("_stratum"):
            continue
        if not isinstance(value, list):
            continue
        for sub in value:
            if isinstance(sub, dict) and sub.get("stratum"):
                yield sub["stratum"]


def _pick_unanimous_stratum(entries: list[dict[str, Any]]) -> str | None:
    """Populate the stratum column only when every entry's per-form
    stratum dict surfaces the same single value. Mixed-stratum usages
    (common for cross-cultural morphemes like Latin loans) get NULL and
    rely on the runtime's per-form ``Meaning.stratum`` dict for actual
    filtering. The column is purely an index hint for bulk queries.
    """
    seen: set[str] = set()
    for entry in entries:
        for stratum in _strata_for_word(entry.get("word") or {}):
            seen.add(stratum)
            if len(seen) > 1:
                return None
    if len(seen) == 1:
        return next(iter(seen))
    return None


def _write_fantasy_morphemes(conn: sqlite3.Connection, fantasy_morphemes: dict[str, Any]) -> int:
    """Lowercase the key to match the existing bundle's ``COLLATE NOCASE``
    semantics."""
    rows = [
        (key.lower(), json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        for key, payload in fantasy_morphemes.items()
    ]
    conn.executemany(
        "INSERT INTO fantasy_morpheme (usage_key, data) VALUES (?, ?)",
        rows,
    )
    return len(rows)


def _write_canonical_decompositions(
    conn: sqlite3.Connection, canonical_decompositions: dict[str, dict[str, str]]
) -> int:
    """Culture-agnostic — see D38.4 for the rationale + how a future
    culture column would land."""
    rows = [
        (name, json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        for name, payload in canonical_decompositions.items()
    ]
    conn.executemany(
        "INSERT INTO canonical_decomposition (toponym_name, data) VALUES (?, ?)",
        rows,
    )
    return len(rows)


def _write_empirical_priors(
    conn: sqlite3.Connection, payload: dict[str, Any] | None
) -> dict[str, int]:
    """Write the empirical-priors payload as a singleton blob row.

    Accepts ``None`` for emits from an L3 that hasn't run
    ``mine-empirical-baselines`` — the row is omitted, and the runtime
    loader falls back to an empty ``EmpiricalPriors`` (vector mode
    needs a non-trivial register to produce output in that case).

    Returns the per-cell counts the operator-summary line reads."""
    if payload is None:
        return {"native_cells": 0, "loan_cells": 0}
    conn.execute(
        "INSERT INTO empirical_priors (id, data) VALUES (1, ?)",
        (json.dumps(payload, ensure_ascii=False).encode("utf-8"),),
    )
    return {
        "native_cells": len(payload.get("native") or []),
        "loan_cells": len(payload.get("loan") or []),
    }


def _write_genitive_split_prior(
    conn: sqlite3.Connection,
    genitive_split_map: dict[tuple[str, str], float] | None,
) -> int:
    """Write the genitive-split prob-map as a singleton blob row (wyrd-aicu.9,
    D-3). Mirrors :func:`_write_empirical_priors`.

    ``genitive_split_map`` is the precomputed ``{(long_form, short_form):
    P_split}`` float map (smoothing + backoff already applied at emit time via
    ``build_split_probability_map``), so the runtime stays DB-free and looks up a
    plain float per genitive decision.

    ``None`` / empty → no row is written (a wipe-without-remine L3, or one
    predating ``mine-genitive-priors --apply``). The runtime loader treats the
    absent row as ``{}``: the connective still wins non-homograph coverage on
    score, only the homograph tiebreak goes silent. Pairs are sorted
    deterministically (``-split_probability``, then the pair) so the blob is
    byte-stable across re-emits. Returns the pair count for the operator summary.
    """
    if not genitive_split_map:
        return 0
    pairs = [
        {"long_form": long, "short_form": short, "split_probability": prob}
        for (long, short), prob in sorted(
            genitive_split_map.items(), key=lambda kv: (-kv[1], kv[0])
        )
    ]
    payload = {"version": EMITTER_VERSION, "pairs": pairs}
    conn.execute(
        "INSERT INTO genitive_split_prior (id, data) VALUES (1, ?)",
        (json.dumps(payload, ensure_ascii=False).encode("utf-8"),),
    )
    return len(pairs)


def _write_proportions(
    conn: sqlite3.Connection, proportions_by_culture: dict[str, dict[str, Any]]
) -> dict[str, int]:
    """Insert the per-culture proportions tables.

    For sampled tables (``usages``, ``single_usages``, ``structures``)
    cumulative is precomputed at emit time: iteration order matches the
    JSON file's key order (Python 3.7+ preserves dict insertion order),
    and ``cumulative_i = cumulative_{i-1} + weight_i``. Runtime
    weighted-random sampling becomes ``WHERE cumulative >= ? ORDER BY
    cumulative LIMIT 1`` — see D38.3.

    For statistics tables (``tag_marginal``, ``tag_cooccurrence``)
    cumulative is unnecessary — they're point lookups, not sampling
    sources.
    """
    counts: Counter[str] = Counter()
    for culture, data in proportions_by_culture.items():
        counts["proportions_usage"] += _insert_cumulative(
            conn,
            "proportions_usage",
            culture,
            _iter_usage_rows(data.get("usages") or {}),
        )
        counts["proportions_single_usage"] += _insert_cumulative(
            conn,
            "proportions_single_usage",
            culture,
            _iter_single_usage_rows(data.get("single_usages") or {}),
        )
        counts["proportions_structure"] += _insert_structures(
            conn, culture, data.get("structures") or []
        )
        counts["proportions_tag_marginal"] += _insert_tag_marginal(
            conn, culture, data.get("tag_marginal") or {}
        )
        counts["proportions_tag_cooccurrence"] += _insert_tag_cooccurrence(
            conn, culture, data.get("tag_cooccurrence") or {}
        )
        counts["proportions_attested_language"] += _insert_attested_languages(
            conn, culture, data.get("attested_languages") or {}
        )
        counts["proportions_bare_word_position"] += _insert_bare_word_positions(
            conn, culture, data.get("bare_word_positions") or {}
        )
    return dict(counts)


def _insert_bare_word_positions(
    conn: sqlite3.Connection,
    culture: str,
    bare_word_positions: dict[str, dict[str, int]],
) -> int:
    """wyrd-rogd.13: write per-word-position counts for bare words of
    multi-word toponyms. One row per (culture, position, usage_key);
    counts are RAW — the naked↔structured threshold is the runtime's
    job (WYRD_BARE_POSITION_THRESHOLD at load time), so re-tuning never
    requires a re-export. Point-lookup table (no cumulative column);
    the vector path builds its per-word-position pools in Python."""
    rows = []
    for position in sorted(bare_word_positions):
        for usage_key, weight in sorted(bare_word_positions[position].items()):
            # int() coercion mirrors the int(v) at the _insert_cumulative
            # call sites: defends operator-supplied --proportions-dir JSON
            # (floats / numeric strings); annotations aren't enforced there.
            count = int(weight)
            if count > 0:
                rows.append((culture, position, usage_key, count))
    if not rows:
        return 0
    conn.executemany(
        "INSERT INTO proportions_bare_word_position "
        "(culture, position, usage_key, weight) VALUES (?, ?, ?, ?)",
        rows,
    )
    return len(rows)


def _insert_attested_languages(
    conn: sqlite3.Connection,
    culture: str,
    attested: dict[str, list[str]],
) -> int:
    """wyrd-pfoo: write per-culture per-Meaning attestation rows. One row per
    (culture, usage_key, primary_language) triple. The runtime vector path's
    eligibility filter joins against these to admit only Meanings whose primary
    language was actually attested in this culture's corpus via that surface.

    D45 (wyrd-aicu): keys are bare SURFACES. ``proportions_from`` already folds
    to bare-lower; this also folds + unions defensively so operator-JSON dashed
    keys collapse. This fixes the wyrd-c6o1.3 homograph leak AT THE SOURCE (the
    exact-key narrowing leaked welsh ``ton`` into english through the un-folded
    dash-variants). The c6o1.3 load-time fold-union in ``load_proportions`` is
    now redundant on v3 bundles (the keys arrive bare) but is kept there as the
    legacy v2-bundle compat path; the aicu fold-site sweep (wyrd-aicu.4) removes
    it once v2 is gone."""
    folded: dict[str, set[str]] = {}
    for usage_key, langs in attested.items():
        # Fold dash + surrounding whitespace (wyrd-an8u) so a dirty operator-JSON
        # key keys the same surface as the clean one and matches the stripped
        # runtime lookup.
        surface = usage_key.replace("-", "").strip().lower()
        folded.setdefault(surface, set()).update(langs)
    # Sort the outer surface loop too so byte-stability is local to this write
    # site rather than leaning on the caller's dict insertion order (seed-repro).
    rows = [
        (culture, surface, lang) for surface in sorted(folded) for lang in sorted(folded[surface])
    ]
    if not rows:
        return 0
    conn.executemany(
        "INSERT INTO proportions_attested_language "
        "(culture, usage_key, primary_language) VALUES (?, ?, ?)",
        rows,
    )
    return len(rows)


def _iter_single_usage_rows(single_usages: dict[str, int]) -> Iterable[tuple[str, str, int]]:
    """Flatten ``single_usages`` (lone-word pool) into ``(surface, 'bare', weight)``
    rows for the L4 writer. D45: lone words are bare by construction (position is
    always ``'bare'``) and the key is the bare surface — fold any dash +
    surrounding whitespace defensively (wyrd-an8u) so an operator-JSON / legacy
    key can't smuggle a dash or trailing space into the table. Mirrors the
    ``_iter_usage_rows`` fold for the single-usage path."""
    for key, weight in single_usages.items():
        yield key.replace("-", "").strip(), "bare", int(weight)


def _iter_usage_rows(usages: dict[str, dict[str, int]]) -> Iterable[tuple[str, str, int]]:
    """D45 (wyrd-aicu): flatten the nested ``{surface: {position: weight}}``
    part pool into ``(surface, position, weight)`` rows for the L4 writer. The
    nested Counter from ``proportions_from`` already SUMS any (surface, position)
    collisions, so no weight-merge / cumulative-rebuild is needed on a fresh
    export — each row appears once."""
    for key, value in usages.items():
        # Fold the key defensively so operator-JSON / dirty keys can't smuggle a
        # dash or surrounding whitespace into the table (D45 / wyrd-an8u).
        bare = key.replace("-", "").strip()
        for position, weight in value.items():
            yield bare, position, int(weight)


def _insert_cumulative(
    conn: sqlite3.Connection,
    table: str,
    culture: str,
    items: Iterable[tuple[str, str, int]],
) -> int:
    """Insert weighted rows with monotonically-increasing cumulative column.

    Each item is ``(usage_key, position, weight)`` (D45: position is an explicit
    column). Skips zero-or-negative weights — they don't contribute to the
    sample distribution, and including them would create duplicate cumulative
    values that the ``PRIMARY KEY (culture, cumulative)`` constraint rejects.

    The table name is f-string-interpolated into the INSERT and the
    whitelist is the load-bearing safety check (SQLite parameter
    binding doesn't cover table identifiers).
    """
    if table not in _CUMULATIVE_USAGE_TABLES:
        raise ValueError(
            f"_insert_cumulative target table {table!r} not in whitelist "
            f"{sorted(_CUMULATIVE_USAGE_TABLES)}"
        )
    rows = []
    cumulative = 0
    for key, position, weight in items:
        if weight <= 0:
            continue
        cumulative += weight
        rows.append((culture, key, position, weight, cumulative))
    try:
        conn.executemany(
            f"INSERT INTO {table} (culture, usage_key, position, weight, cumulative) "
            f"VALUES (?, ?, ?, ?, ?)",
            rows,
        )
    except sqlite3.IntegrityError as exc:
        # The PRIMARY KEY is (culture, cumulative); cumulative is strictly
        # monotonic within a single call and zero/negative weights are
        # filtered above, so the realistic causes are: (a) _insert_cumulative
        # being called twice for the same (table, culture) pair in a single
        # emit (future-refactor regression), or (b) the L4 DB already
        # holding rows for this (table, culture) at insert time (re-emit
        # without first deleting the output file).
        raise RuntimeError(
            f"cumulative insert into {table} for culture {culture!r} hit a "
            f"constraint violation — duplicate (culture, cumulative) row. "
            f"Check whether {table} was already populated for this culture "
            f"earlier in the emit, or whether the output DB wasn't truncated "
            f"before writing: {exc}"
        ) from exc
    return len(rows)


def _insert_structures(
    conn: sqlite3.Connection,
    culture: str,
    structures: list[dict[str, Any]],
) -> int:
    """Structure rows store the template (the ``words`` field of each
    structure) as a JSON blob; the runtime consumes the whole template
    after sampling on cumulative."""
    rows = []
    cumulative = 0
    for struct in structures:
        weight = int(struct.get("proportion") or 0)
        if weight <= 0:
            continue
        template = struct.get("words") or []
        cumulative += weight
        rows.append(
            (
                culture,
                json.dumps(template, ensure_ascii=False).encode("utf-8"),
                weight,
                cumulative,
            )
        )
    conn.executemany(
        "INSERT INTO proportions_structure (culture, template, weight, cumulative) "
        "VALUES (?, ?, ?, ?)",
        rows,
    )
    return len(rows)


def _insert_tag_marginal(
    conn: sqlite3.Connection, culture: str, tag_marginal: dict[str, Any]
) -> int:
    rows = [(culture, tag, int(weight)) for tag, weight in tag_marginal.items()]
    conn.executemany(
        "INSERT INTO proportions_tag_marginal (culture, tag, weight) VALUES (?, ?, ?)",
        rows,
    )
    return len(rows)


def _insert_tag_cooccurrence(
    conn: sqlite3.Connection, culture: str, tag_cooccurrence: dict[str, Any]
) -> int:
    """tag_cooccurrence keys are ``tag1|tag2``; split on the pipe.

    Raises ``ValueError`` on a malformed key. Build-time tooling, loud-
    fail is the right shape per D24 — silently dropping rows would lose
    cohesion data without an observability path.
    """
    rows = []
    for key, weight in tag_cooccurrence.items():
        if "|" not in key:
            raise ValueError(
                f"tag_cooccurrence key {key!r} (culture={culture!r}) is "
                "missing the 'tag1|tag2' separator; refusing to silently "
                "drop the row"
            )
        tag1, _, tag2 = key.partition("|")
        rows.append((culture, tag1, tag2, int(weight)))
    conn.executemany(
        "INSERT INTO proportions_tag_cooccurrence (culture, tag1, tag2, weight) "
        "VALUES (?, ?, ?, ?)",
        rows,
    )
    return len(rows)


def select_dev_subset(
    subjects: list[dict[str, Any]],
    fantasy_morphemes: dict[str, Any],
    canonical_decompositions: dict[str, dict[str, str]],
    proportions_by_culture: dict[str, dict[str, Any]],
    *,
    top_n_per_culture: int | None,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, dict[str, str]],
    dict[str, dict[str, Any]],
]:
    """Trim the L4 input set to the subjects actually reachable by generation.

    ``top_n_per_culture`` controls how aggressive the trim is:

    * ``int`` (``--dev``): keep only the top-N highest-weighted usages per
      culture — a tiny deterministic seed slice.
    * ``None`` (``--generation-subset``, wyrd-ukq0): keep ALL proportion-
      referenced usages (``_top_n_by_weight(d, None)`` slices ``[:None]`` =
      everything), dropping only the decomposition-only corpus — subjects that
      no culture's proportions reference. This is the production cold-start trim:
      generation only ever samples proportion-referenced morphemes, so the
      dropped ~70% is dead weight at generation load time (it served the
      arbitrary-name decomposition/explain path, deferred per wyrd-ukq0).

    Selection rules — designed so re-running on the same inputs yields
    byte-equal output (the committed ``seed-runtime.db`` is a fixed
    artifact of these rules):

    * **Proportions usages / single_usages**: keep the top
      ``top_n_per_culture`` entries by weight, ties broken alphabetically
      on the usage_key. Sorted-by-weight order — NOT the source JSON's
      insertion order — so future re-mining that reshuffles JSON keys
      doesn't perturb the seed.
    * **Proportions structures**: kept in full (always ≤50 entries).
    * **Proportions tag_marginal / tag_cooccurrence**: kept in full
      (a few hundred rows max; the runtime cohesion path relies on full
      tag coverage even at low usage counts).
    * **Meanings (subjects)**: drop any word whose bare surface
      (``modern_usage`` dash-stripped + lowercased — D45) is not in the
      kept-surface set across all 5 cultures. Drop any subject that has
      no surviving words after the filter.
    * **Fantasy morphemes**: kept in full (only ~240 total in the live
      corpus).
    * **Canonical decompositions**: kept in full (modest count; the
      lookup is per-toponym-name and the seed shouldn't lose coverage
      of common names).

    Returns ``(subjects, fantasy_morphemes, canonical_decompositions,
    proportions_by_culture)`` in the same shape as the inputs, with
    each container trimmed per the rules above.
    """
    # D45 (wyrd-aicu): identity/comparison sets are bare-LOWER surfaces (the
    # build folds attested_languages to bare-lower; usage/single keys keep
    # stored case for the render). Fold everything to bare-lower when MATCHING
    # so case-twins (`Ghyll`/`ghyll`) and the attested fold align.
    keep_surfaces: set[str] = set()
    trimmed_proportions: dict[str, dict[str, Any]] = {}
    # Sort by culture key so the SQL INSERT order is the same across
    # platforms regardless of the source dict's iteration order.
    for culture in sorted(proportions_by_culture):
        data = proportions_by_culture[culture]
        # D45: ``usages`` is nested {surface: {position: weight}} — rank surfaces
        # by TOTAL weight across positions, keep the full nested entry.
        usages = data.get("usages") or {}
        single_usages = data.get("single_usages") or {}
        usage_totals = {s: sum(by_pos.values()) for s, by_pos in usages.items()}
        kept_usage_surfaces = [s for s, _w in _top_n_by_weight(usage_totals, top_n_per_culture)]
        kept_usages = {s: usages[s] for s in kept_usage_surfaces}
        kept_single = dict(_top_n_by_weight(single_usages, top_n_per_culture))
        keep_surfaces.update(s.lower() for s in kept_usages)
        keep_surfaces.update(s.lower() for s in kept_single)
        # Sort tag dicts so re-runs against the same data produce the
        # same insertion order — defensive against future ingesters that
        # might emit a different dict order than today's.
        # wyrd-pfoo: narrow attested_languages to THIS culture's kept
        # usage set so the per-Meaning attestation table stays aligned
        # with this culture's per-usage proportions tables in --dev
        # mode. Using ``keep_surfaces`` (the cumulative cross-culture
        # set) would leak rows: a usage_key kept in culture A's top-N
        # but absent from culture B's would survive in B's attestation
        # output. Build a per-culture set instead.
        culture_keep_surfaces = {s.lower() for s in kept_usages} | {s.lower() for s in kept_single}
        raw_attested = data.get("attested_languages") or {}
        trimmed_attested = {
            k: sorted(raw_attested[k])
            for k in sorted(raw_attested)
            if k.lower().replace("-", "") in culture_keep_surfaces
        }
        # wyrd-rogd.13: narrow the per-word-position bare stats to THIS
        # culture's kept single_usages (the bare pool the runtime samples
        # from) so the seed's positional rows never reference a usage the
        # trimmed bare pool can't supply. Same per-culture-set rationale
        # as attested_languages above. Compared by bare SURFACE: the
        # positional rows are surface-keyed (D40 identity) while
        # kept_single carries stored-form keys (case twins like 'Ghyll').
        kept_single_surfaces = {u.lower().replace("-", "") for u in kept_single}
        raw_bare_positions = data.get("bare_word_positions") or {}
        trimmed_bare_positions = {
            position: {
                usage: count
                for usage, count in sorted(raw_bare_positions[position].items())
                if usage.lower().replace("-", "") in kept_single_surfaces
            }
            for position in sorted(raw_bare_positions)
        }
        trimmed_bare_positions = {p: m for p, m in trimmed_bare_positions.items() if m}
        trimmed_proportions[culture] = {
            "usages": kept_usages,
            "single_usages": kept_single,
            "structures": data.get("structures") or [],
            "tag_marginal": dict(sorted((data.get("tag_marginal") or {}).items())),
            "tag_cooccurrence": dict(sorted((data.get("tag_cooccurrence") or {}).items())),
            "attested_languages": trimmed_attested,
            "bare_word_positions": trimmed_bare_positions,
        }

    trimmed_subjects: list[dict[str, Any]] = []
    for subject in subjects:
        # D45: subject modern_usage is still dash-marked here (de-dashed later
        # before _write_meanings); fold to bare-lower to match keep_surfaces.
        # Must use the SAME fold as the keep_surfaces source (_bare_surface →
        # dash + surrounding-whitespace strip, wyrd-an8u): keep_surfaces is
        # whitespace-clean, so a raw ``'Oak- '`` folded with replace-only would
        # be ``'oak '`` and miss the stripped ``'oak'`` — silently dropping the
        # word from the --dev / generation_subset (prod cold-start) bundle.
        kept_words = [
            word
            for word in (subject.get("words") or [])
            if _bare_modern_usage(word.get("modern_usage") or "").lower() in keep_surfaces
        ]
        if not kept_words:
            continue
        trimmed = dict(subject)
        trimmed["words"] = kept_words
        trimmed_subjects.append(trimmed)

    # Sort subjects by a canonical key so two subjects that share a
    # modern_usage land in the same grouping order in _write_meanings's
    # per-usage blob. Defensive against future upstream code that might
    # emit subjects in a non-deterministic order (today's export_meanings
    # sorts; this guards the contract regardless).
    trimmed_subjects.sort(key=_subject_sort_key)

    # wyrd-9eqk: prune structures that the top-N usage trim left
    # UNRENDERABLE. select_dev_subset keeps every structure but trims
    # usages to top-N-by-weight, so a structure whose only filler usage(s)
    # ranked below the cut is orphaned — at runtime none of its slots can
    # be filled and it renders an empty name. (welsh hit exactly this until
    # wyrd-5z5j's bare-bucket fix; the *class* recurs whenever the corpus,
    # mining, or top_n shifts, so this is a STANDING guard that runs on
    # every --dev export rather than a one-off fix.) Build the real
    # per-culture bucket set from the trimmed data and drop structures with
    # no fillable slot; partially-fillable structures stay (they render a
    # shorter, non-empty name). Dev-export-only — the full production export
    # trims nothing, so it never orphans and this is a no-op there. The
    # runtime's empty-pool degradation (an unfillable slot yields a None slot
    # the NewName drops, wyrd-cj6f) is the complementary crash guard; this
    # keeps the dev seed itself empty-name-free.
    _prune_unrenderable_dev_structures(trimmed_proportions, trimmed_subjects)

    # Fantasy + canonical: sort by key so re-runs against the same
    # source pick the same INSERT order across platforms / Python
    # versions, defensive against any future ingester whose dict
    # iteration order isn't already stable.
    return (
        trimmed_subjects,
        dict(sorted(fantasy_morphemes.items())),
        dict(sorted(canonical_decompositions.items())),
        trimmed_proportions,
    )


def _prune_unrenderable_dev_structures(
    proportions_by_culture: dict[str, dict[str, Any]],
    subjects: list[dict[str, Any]],
) -> None:
    """wyrd-9eqk: drop fully-unrenderable structures from each culture's
    ``--dev`` proportions, in place. A structure is unrenderable when NONE
    of its slots reference a bucket the (already top-N-trimmed) meaning
    generator can fill — such a structure can only ever render an empty
    name. Partially-fillable structures are kept (they render a shorter,
    non-empty name).

    Reuses the real runtime bucket logic — a ``MeaningGenerator`` built
    from the trimmed usages exposes ``.generators`` (the authoritative
    bucket set, including wyrd-5z5j's bare-bucket registration), and
    ``word_to_key`` decodes each encoded structure's slots the same way the
    runtime does — so no slot-key construction is duplicated here. It filters
    the encoded structures list directly (no re-encode), and deliberately
    avoids ``load_proportions`` / ``NameGenerator`` so it doesn't inherit the
    grammar-filter loud-raise on minimal/test inputs.

    Emits one warning per culture when it drops anything; on a clean build
    (the expected case) it's silent and a no-op.
    """
    from wyrd.generators.kenning.runtime.meaning import load_meanings
    from wyrd.generators.kenning.runtime.proportions import MeaningGenerator, word_to_key

    meaning_db, tag_db = load_meanings({"subjects": subjects})
    for culture in sorted(proportions_by_culture):
        cp = proportions_by_culture[culture]
        # Build the runtime bucket set exactly as load_proportions does:
        # multi usages register under their dash-shape key; single usages
        # also register a bare variant ("single" addkey, wyrd-5z5j).
        meaning_gen = MeaningGenerator(meaning_db, tag_db, cp.get("usages") or {})
        meaning_gen.load_parts(cp.get("single_usages") or {}, "single")
        gens = meaning_gen.generators
        kept_structs: list[dict[str, Any]] = []
        dropped = 0
        for encoded in cp.get("structures") or []:
            decoded = tuple(word_to_key(w) for w in encoded["words"])
            if any(slot_key in gens for word in decoded for slot_key in word):
                kept_structs.append(encoded)
            else:
                dropped += 1
        if dropped:
            _logger.warning(
                "wyrd-9eqk: --dev seed dropped %d unrenderable structure(s) for "
                "culture %r — their filler usage(s) fell below the top-N usage cut, "
                "so they would render empty names. Seed stays renderable; revisit "
                "dev_top_n_per_culture if this is unexpected.",
                dropped,
                culture,
            )
            cp["structures"] = kept_structs


def _subject_sort_key(subject: dict[str, Any]) -> tuple:
    """Canonical sort key for a subject. Stable across platforms and
    upstream-ordering changes — combines sorted modern_usages across
    words with the subject-level metadata so two distinct subjects
    that happen to share a modern_usage still order deterministically.
    """
    words = subject.get("words") or []
    usage_tuple = tuple(sorted(w.get("modern_usage", "") for w in words))
    return (
        usage_tuple,
        tuple(subject.get("meaning") or []),
        subject.get("modifier_type") or "",
        tuple(sorted(subject.get("modifier_tags") or [])),
    )


def _top_n_by_weight(items: dict[str, int], n: int | None) -> list[tuple[str, int]]:
    """Return the top-N highest-weighted (key, weight) pairs in a stable
    order: weight descending first, then key ascending for ties. The
    key-ascending tiebreaker keeps the output bit-stable across re-runs
    even if the input dict's insertion order changes.

    ``n=None`` keeps ALL items (``[:None]`` slices the whole list) — the
    uncapped ``--generation-subset`` path (wyrd-ukq0).
    """
    return sorted(items.items(), key=lambda kv: (-kv[1], kv[0]))[:n]


def _write_metadata(
    conn: sqlite3.Connection,
    *,
    source_lexicon_db: str,
    built_at: str | None = None,
) -> None:
    """Stamp bundle_metadata operational keys.

    ``built_at=None`` (the default) stamps wall-clock UTC. Callers in
    --dev mode pass the ``DEV_BUILT_AT`` sentinel so the seed-runtime.db
    stays byte-stable across emits — the drift-detection test on the
    committed seed depends on a deterministic timestamp.

    ``source_lexicon_db`` is the resolved absolute path of the L3 source
    in production emit, or ``DEV_SOURCE_LEXICON_SENTINEL`` in --dev mode
    (operator-path differences would break cross-environment byte-
    stability of the committed seed).
    """
    stamp = (
        built_at
        if built_at is not None
        else _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    conn.executemany(
        "INSERT INTO bundle_metadata (key, value) VALUES (?, ?)",
        [
            ("schema_version", SCHEMA_VERSION),
            ("built_at", stamp),
            ("emitter_version", EMITTER_VERSION),
            ("source_lexicon_db", source_lexicon_db),
        ],
    )
