"""L4 runtime SQLite DB emitter (wyrd-d90t PR 1).

Writes the runtime DB that PR 4's loader will pull from S3 on Lambda
cold start. Source inputs:

  * the L3 lexicon DB (meanings, fantasy morphemes, canonical
    decompositions), passed in via the already-opened ``LexiconDB`` or
    its query helpers
  * the 5 per-culture proportions JSONs (read from the bundled data
    directory or any override path)

The schema lives in ``wyrd/generators/kenning/data/runtime.sql``. The
runtime treats the emitted DB as read-only (``file:...?mode=ro`` URI);
re-emit and ship a new versioned key on S3 to update.

See D38 for the L4 architecture rationale.

This module lives in the ``lexicon/`` package per the
``test_kenning_package_boundary`` rule: anything that opens raw sqlite3
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
from importlib import resources
from pathlib import Path
from typing import Any, Iterable

# Bumped when the L4 table shape changes incompatibly. The loader will
# reject DBs whose schema_version row doesn't match the runtime's
# expected version, forcing a re-download.
SCHEMA_VERSION = "1"

# Stamped into bundle_metadata so operators can correlate a deployed DB
# back to its emitter source.
EMITTER_VERSION = "wyrd-d90t-pr1"

# Per-culture proportions filenames. Same five files _load_culture reads
# today; the cutover (PR 8) removes them from the bundle.
CULTURE_PROPORTIONS = (
    "english",
    "scottish",
    "welsh",
    "irish",
    "breton",
)


def write_runtime_db(
    *,
    output_path: Path,
    subjects: list[dict[str, Any]],
    fantasy_morphemes: dict[str, Any],
    canonical_decompositions: dict[str, dict[str, str]],
    proportions_dir: Path,
    source_lexicon_db: Path,
) -> dict[str, int]:
    """Emit the L4 SQLite DB at ``output_path``.

    Overwrites any existing file at the path. Returns a per-section
    rowcount dict suitable for an operator-facing summary line.
    """
    proportions_by_culture = _load_proportions(proportions_dir)

    if output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(output_path))
    try:
        _init_runtime_schema(conn)
        n_meanings = _write_meanings(conn, subjects)
        n_fantasy = _write_fantasy_morphemes(conn, fantasy_morphemes)
        n_canonical = _write_canonical_decompositions(conn, canonical_decompositions)
        proportion_counts = _write_proportions(conn, proportions_by_culture)
        _write_metadata(conn, source_lexicon_db=source_lexicon_db)
        conn.commit()
    finally:
        conn.close()

    return {
        "meanings": n_meanings,
        "fantasy_morphemes": n_fantasy,
        "canonical_decompositions": n_canonical,
        "cultures": len(proportions_by_culture),
        **proportion_counts,
    }


def _load_proportions(proportions_dir: Path) -> dict[str, dict[str, Any]]:
    """Read all 5 per-culture proportions JSON files into memory.

    Cultures whose file is absent are silently skipped — the bundled set
    is the truth-source; a missing file means that culture isn't shipped
    from this emit run (covers PR-1-era dev fixtures + future cultures-
    not-yet-built scenarios alike).
    """
    out: dict[str, dict[str, Any]] = {}
    for culture in CULTURE_PROPORTIONS:
        path = proportions_dir / f"{culture}_proportions.json"
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as f:
            out[culture] = json.load(f)
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
        "INSERT INTO meaning (usage_key, primary_language, stratum, data) "
        "VALUES (?, ?, ?, ?)",
        rows,
    )
    return len(rows)


def _pick_primary_language(entries: list[dict[str, Any]]) -> str:
    """Dominant language across all entries grouped under one usage.

    Counts language-form-array keys across every word; ties broken
    alphabetically so re-emits with identical inputs produce identical
    rows. Returns ``''`` when no language is present (shouldn't happen
    — every word has at least one language form array — but the schema
    requires NOT NULL).
    """
    counts: Counter[str] = Counter()
    for entry in entries:
        word = entry.get("word") or {}
        for key, value in word.items():
            # Per the D26 sibling-suffix pattern, language form arrays are
            # the non-suffix, non-"modern_usage", non-"era_reflexes" keys
            # whose values are lists of bare form strings.
            if key in ("modern_usage", "era_reflexes"):
                continue
            if not isinstance(value, list):
                continue
            if value and isinstance(value[0], dict):
                # Sidecar fields (variants/inflections/etc.) — list-of-dicts.
                # Skip; only bare-string form arrays are language axes.
                continue
            counts[key] += 1
    if not counts:
        return ""
    most = counts.most_common()
    top_count = most[0][1]
    tied = sorted(k for k, v in most if v == top_count)
    return tied[0]


def _pick_unanimous_stratum(entries: list[dict[str, Any]]) -> str | None:
    """Populate the stratum column only when every entry's per-form
    stratum dict surfaces the same single value. Mixed-stratum usages
    (common for cross-cultural morphemes like Latin loans) get NULL and
    rely on the runtime's per-form ``Meaning.stratum_for`` for actual
    filtering. The column is purely an index hint for bulk queries.
    """
    seen: set[str] = set()
    for entry in entries:
        word = entry.get("word") or {}
        for key, value in word.items():
            if not key.endswith("_stratum"):
                continue
            if not isinstance(value, list):
                continue
            for sub in value:
                if isinstance(sub, dict) and sub.get("stratum"):
                    seen.add(sub["stratum"])
                    if len(seen) > 1:
                        return None
    if len(seen) == 1:
        return next(iter(seen))
    return None


def _write_fantasy_morphemes(
    conn: sqlite3.Connection, fantasy_morphemes: dict[str, Any]
) -> int:
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
    """Culture-agnostic in PR 1 — see D38.4 for the rationale + how a
    future culture column would land."""
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
    counts: dict[str, int] = {}
    for culture, data in proportions_by_culture.items():
        counts["proportions_usage"] = counts.get("proportions_usage", 0) + _insert_cumulative(
            conn,
            "proportions_usage",
            culture,
            ((k, int(v)) for k, v in (data.get("usages") or {}).items()),
        )
        counts["proportions_single_usage"] = counts.get(
            "proportions_single_usage", 0
        ) + _insert_cumulative(
            conn,
            "proportions_single_usage",
            culture,
            ((k, int(v)) for k, v in (data.get("single_usages") or {}).items()),
        )
        counts["proportions_structure"] = counts.get(
            "proportions_structure", 0
        ) + _insert_structures(conn, culture, data.get("structures") or [])
        counts["proportions_tag_marginal"] = counts.get(
            "proportions_tag_marginal", 0
        ) + _insert_tag_marginal(conn, culture, data.get("tag_marginal") or {})
        counts["proportions_tag_cooccurrence"] = counts.get(
            "proportions_tag_cooccurrence", 0
        ) + _insert_tag_cooccurrence(conn, culture, data.get("tag_cooccurrence") or {})
    return counts


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
    """
    rows = []
    cumulative = 0
    for key, weight in items:
        if weight <= 0:
            continue
        cumulative += weight
        rows.append((culture, key, weight, cumulative))
    conn.executemany(
        f"INSERT INTO {table} (culture, usage_key, weight, cumulative) VALUES (?, ?, ?, ?)",
        rows,
    )
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
    """tag_cooccurrence keys are ``tag1|tag2``; split on the pipe."""
    rows = []
    for key, weight in tag_cooccurrence.items():
        if "|" not in key:
            continue
        tag1, _, tag2 = key.partition("|")
        rows.append((culture, tag1, tag2, int(weight)))
    conn.executemany(
        "INSERT INTO proportions_tag_cooccurrence (culture, tag1, tag2, weight) "
        "VALUES (?, ?, ?, ?)",
        rows,
    )
    return len(rows)


def _write_metadata(conn: sqlite3.Connection, *, source_lexicon_db: Path) -> None:
    """Stamp bundle_metadata operational keys."""
    built_at = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn.executemany(
        "INSERT INTO bundle_metadata (key, value) VALUES (?, ?)",
        [
            ("schema_version", SCHEMA_VERSION),
            ("built_at", built_at),
            ("emitter_version", EMITTER_VERSION),
            ("source_lexicon_db", str(source_lexicon_db)),
        ],
    )
