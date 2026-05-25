"""L4 runtime SQLite DB emitter (wyrd-d90t).

Writes the runtime DB the Lambda loader pulls from S3 on cold start.
Source inputs:

  * the L3 lexicon DB (meanings, fantasy morphemes, canonical
    decompositions), passed in via the already-opened ``LexiconDB`` or
    its query helpers
  * the 5 per-culture proportions JSONs (read from the bundled data
    directory or any override path)

The schema lives in ``wyrd/generators/kenning/data/runtime.sql``. The
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
import sqlite3
from collections import Counter
from collections.abc import Iterable
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any

# Bumped when the L4 table shape changes incompatibly. The loader rejects
# DBs whose schema_version row doesn't match the runtime's expected
# version, forcing a re-download.
SCHEMA_VERSION = "1"

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


def write_runtime_db(
    *,
    output_path: Path,
    subjects: list[dict[str, Any]],
    fantasy_morphemes: dict[str, Any],
    canonical_decompositions: dict[str, dict[str, str]],
    proportions_dir: Path | Traversable | None,
    source_lexicon_db: Path,
    dev_subset: bool = False,
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
    """
    if proportions_dir is None:
        proportions_by_culture = _compute_proportions_inline(subjects)
    else:
        proportions_by_culture = _load_proportions(proportions_dir)

    if dev_subset:
        subjects, fantasy_morphemes, canonical_decompositions, proportions_by_culture = (
            select_dev_subset(
                subjects,
                fantasy_morphemes,
                canonical_decompositions,
                proportions_by_culture,
                top_n_per_culture=dev_top_n_per_culture,
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
            n_meanings = _write_meanings(conn, subjects)
            n_fantasy = _write_fantasy_morphemes(conn, fantasy_morphemes)
            n_canonical = _write_canonical_decompositions(conn, canonical_decompositions)
            proportion_counts = _write_proportions(conn, proportions_by_culture)
            _write_metadata(
                conn,
                source_lexicon_db=(
                    DEV_SOURCE_LEXICON_SENTINEL if dev_subset else str(source_lexicon_db.resolve())
                ),
                built_at=DEV_BUILT_AT if dev_subset else None,
            )
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
        "fantasy_morphemes": n_fantasy,
        "canonical_decompositions": n_canonical,
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
) -> dict[str, dict[str, Any]]:
    """Build per-culture proportions on the fly from the subjects we
    just exported + the bundled ``<culture>_place_names.json`` corpus.

    The d90t cutover removed the bundled ``<culture>_proportions.json``
    files; this rebuilds the same data deterministically as part of
    the emit so an operator only needs to point ``export-runtime-db``
    at the L3 to produce a fully-populated L4. Operator can still
    supply ``--proportions-dir`` to bypass this with a hand-edited or
    frozen proportions set.
    """
    # Local imports avoid pulling the rebuild_proportions / runtime
    # name-decomposition deps into module init for callers that pass
    # ``--proportions-dir`` and never need the inline path.
    from wyrd.generators.kenning.cli.rebuild_proportions import _proportions_from
    from wyrd.generators.kenning.runtime.meaning import load_meanings
    from wyrd.generators.kenning.runtime.name import load_names_with_regions

    word_db, _ = load_meanings({"subjects": subjects})
    out: dict[str, dict[str, Any]] = {}
    for culture in CULTURE_PROPORTIONS:
        place_names_entry = resources.files("wyrd.generators.kenning.data").joinpath(
            f"{culture}_place_names.json"
        )
        if not place_names_entry.is_file():
            continue
        names_data = json.loads(place_names_entry.read_text())
        name_entries = load_names_with_regions(names_data)
        # decompose each name against the just-built meaning_db; the
        # caller already gated subjects via export_meanings's witness
        # promotion, so this rolls in whatever the operator's emit
        # decided to ship.
        resolved = []
        for name, _region in name_entries:
            name.find_meaning(word_db, reduce=True)
            resolved.append(name)
        good_names = [n for n in resolved if n.count_unaccounted() == 0]
        out[culture] = _proportions_from(good_names)
    return out


def _init_runtime_schema(conn: sqlite3.Connection) -> None:
    """Run runtime.sql against the freshly-created DB."""
    sql = resources.files("wyrd.generators.kenning.data").joinpath("runtime.sql").read_text()
    conn.executescript(sql)


def _write_meanings(conn: sqlite3.Connection, subjects: list[dict[str, Any]]) -> int:
    """Insert per-usage meaning rows. Groups multiple subjects sharing a
    ``modern_usage`` into one row whose ``data`` blob is a JSON list of
    ``{meaning, modifier_tags, modifier_type, word}`` entries — matches
    the per-word iteration ``load_meanings`` does today (each entry
    becomes one Meaning at load time).
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
    rely on the runtime's per-form ``Meaning.stratum_for`` for actual
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
            ((k, int(v)) for k, v in (data.get("usages") or {}).items()),
        )
        counts["proportions_single_usage"] += _insert_cumulative(
            conn,
            "proportions_single_usage",
            culture,
            ((k, int(v)) for k, v in (data.get("single_usages") or {}).items()),
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
    return dict(counts)


def _insert_cumulative(
    conn: sqlite3.Connection,
    table: str,
    culture: str,
    items: Iterable[tuple[str, int]],
) -> int:
    """Insert weighted rows with monotonically-increasing cumulative column.

    Skips zero-or-negative weights — they don't contribute to the sample
    distribution, and including them would create duplicate cumulative
    values that the ``PRIMARY KEY (culture, cumulative)`` constraint
    rejects.

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
    for key, weight in items:
        if weight <= 0:
            continue
        cumulative += weight
        rows.append((culture, key, weight, cumulative))
    try:
        conn.executemany(
            f"INSERT INTO {table} (culture, usage_key, weight, cumulative) VALUES (?, ?, ?, ?)",
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
    top_n_per_culture: int,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, dict[str, str]],
    dict[str, dict[str, Any]],
]:
    """Trim the L4 input set down to a deterministic ``--dev`` slice.

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
    * **Meanings (subjects)**: drop any word whose ``modern_usage`` is
      not in the kept-usage set across all 5 cultures. Drop any subject
      that has no surviving words after the filter.
    * **Fantasy morphemes**: kept in full (only ~240 total in the live
      corpus).
    * **Canonical decompositions**: kept in full (modest count; the
      lookup is per-toponym-name and the seed shouldn't lose coverage
      of common names).

    Returns ``(subjects, fantasy_morphemes, canonical_decompositions,
    proportions_by_culture)`` in the same shape as the inputs, with
    each container trimmed per the rules above.
    """
    keep_usage_keys: set[str] = set()
    trimmed_proportions: dict[str, dict[str, Any]] = {}
    # Sort by culture key so the SQL INSERT order is the same across
    # platforms regardless of the source dict's iteration order.
    for culture in sorted(proportions_by_culture):
        data = proportions_by_culture[culture]
        usages = data.get("usages") or {}
        single_usages = data.get("single_usages") or {}
        kept_usages = dict(_top_n_by_weight(usages, top_n_per_culture))
        kept_single = dict(_top_n_by_weight(single_usages, top_n_per_culture))
        keep_usage_keys.update(kept_usages.keys())
        keep_usage_keys.update(kept_single.keys())
        # Sort tag dicts so re-runs against the same data produce the
        # same insertion order — defensive against future ingesters that
        # might emit a different dict order than today's.
        trimmed_proportions[culture] = {
            "usages": kept_usages,
            "single_usages": kept_single,
            "structures": data.get("structures") or [],
            "tag_marginal": dict(sorted((data.get("tag_marginal") or {}).items())),
            "tag_cooccurrence": dict(sorted((data.get("tag_cooccurrence") or {}).items())),
        }

    trimmed_subjects: list[dict[str, Any]] = []
    for subject in subjects:
        kept_words = [
            word
            for word in (subject.get("words") or [])
            if word.get("modern_usage") in keep_usage_keys
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


def _top_n_by_weight(items: dict[str, int], n: int) -> list[tuple[str, int]]:
    """Return the top-N highest-weighted (key, weight) pairs in a stable
    order: weight descending first, then key ascending for ties. The
    key-ascending tiebreaker keeps the output bit-stable across re-runs
    even if the input dict's insertion order changes.
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
