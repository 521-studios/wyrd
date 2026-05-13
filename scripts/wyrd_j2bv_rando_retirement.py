"""Retire pure-grandfather rando-port etymons — wyrd-j2bv Phase 4b.

Splits ``data/mining/rando-port.jsonl`` into two files:

* ``data/mining/rando-port.jsonl`` (rewritten) — keeps the source row
  plus the SURVIVING etymons and their citations. This is the active
  source-of-truth for rando-port going forward.
* ``data/mining/retired/rando-retired.jsonl`` (new) — the RETIRED
  etymons + their citations + a synthetic source row. Preserved as
  a tombstone for future audit; NOT loaded by the build pipeline
  because ``data/mining/*.jsonl`` glob doesn't recurse into the
  ``retired/`` subdirectory.

Why split rather than append remove-events
==========================================

An earlier version appended ``_op=remove`` events to rando-port.jsonl.
That works but mixes the active source-of-truth with retirement
history in one file — every future read has to replay the events
to get the active set. The split keeps the active file clean
(canonical rows only, no event-log noise) and preserves the retired
data verbatim so future passes can re-examine the etymons that got
dropped (un-retire some, mine them as candidates for re-attestation,
etc.).

Selection criteria (verified against the live DB)
=================================================

1. Etymon's only citation source is ``rando-port`` (no scholar /
   wiktionary-empirical / other source cites it).
2. Etymon isn't referenced by any ``toponym_etymology_element``.
3. Etymon isn't referenced by any ``etymon_descent`` parent/child.

Run once. Re-running compares the current DB query against the
already-retired set in the tombstone file and only retires deltas."""

from __future__ import annotations

import json
import os
import sqlite3
from collections import Counter
from pathlib import Path

ACTIVE_PATH = Path("data/mining/rando-port.jsonl")
RETIRED_DIR = Path("data/mining/retired")
RETIRED_PATH = RETIRED_DIR / "rando-retired.jsonl"
MARKER = "wyrd-j2bv-retirement"
RETIRED_SOURCE_ID = "rando-retired"


def _candidate_refs() -> set[str]:
    """Return the set of etymon refs (``language:canonical_form``)
    safe to retire per the three criteria."""
    db_path = os.environ.get("WYRD_LEXICON_DB") or os.path.expanduser("~/.wyrd/lexicon.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    q = """
    WITH etymon_sources AS (
      SELECT e.id, e.language, e.canonical_form,
             GROUP_CONCAT(DISTINCT ec.source_id) AS sources
      FROM etymon e
      LEFT JOIN etymon_citation ec ON ec.etymon_id = e.id
      GROUP BY e.id
    )
    SELECT language, canonical_form
    FROM etymon_sources
    WHERE sources = 'rando-port'
      AND NOT EXISTS (
        SELECT 1 FROM toponym_etymology_element tee
         WHERE tee.etymon_id = etymon_sources.id
      )
      AND NOT EXISTS (
        SELECT 1 FROM etymon_descent ed
         WHERE ed.parent_id = etymon_sources.id
            OR ed.child_id = etymon_sources.id
      )
    """
    refs = {f"{r['language']}:{r['canonical_form']}" for r in conn.execute(q)}
    conn.close()
    return refs


_RETIRED_SOURCE_ROW = {
    "_type": "source",
    "ref": RETIRED_SOURCE_ID,
    "title": "Retired rando-port etymons (wyrd-j2bv)",
    "notes": (
        "Tombstone file for etymons retired from rando-port during the "
        "wyrd-j2bv Phase 4b cleanup. These rows are NOT loaded by the "
        "build pipeline (lives in data/mining/retired/, outside the "
        "non-recursive *.jsonl glob). Preserved verbatim so future "
        "passes can re-examine candidates for un-retirement or "
        "scholar-mining catch-up. See "
        "scripts/wyrd_j2bv_rando_retirement.py for selection criteria."
    ),
}


def _already_retired_refs(path: Path) -> set[str]:
    """Etymon refs already moved to the tombstone — skip on re-run."""
    refs: set[str] = set()
    if not path.exists():
        return refs
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("_type") == "etymon" and row.get("ref"):
                refs.add(row["ref"])
    return refs


def main() -> None:
    candidates = _candidate_refs()
    print(f"Retirement candidates from DB: {len(candidates)}")

    already_retired = _already_retired_refs(RETIRED_PATH)
    print(f"Already in tombstone:          {len(already_retired)}")

    to_retire = candidates - already_retired
    print(f"New retirements this run:      {len(to_retire)}")

    if not to_retire:
        return

    # Read the active rando-port file.
    rows: list[dict] = []
    with ACTIVE_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    active_rows: list[dict] = []
    retired_rows: list[dict] = []
    for row in rows:
        rtype = row.get("_type")
        if rtype == "source":
            active_rows.append(row)
            continue
        if rtype == "etymon":
            ref = row.get("ref")
            if ref in to_retire:
                retired_rows.append(row)
            else:
                active_rows.append(row)
            continue
        if rtype == "citation":
            etymon_ref = row.get("etymon_ref")
            if etymon_ref in to_retire:
                # Tag with the retirement marker for audit clarity.
                row = {**row, "retirement_marker": MARKER}
                retired_rows.append(row)
            else:
                active_rows.append(row)
            continue
        # Anything else (mining_run, etc.) — keep in active by default.
        active_rows.append(row)

    # Rewrite the active file with surviving rows only.
    with ACTIVE_PATH.open("w", encoding="utf-8") as fh:
        for row in active_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Write the tombstone file (creating retired/ if needed).
    RETIRED_DIR.mkdir(parents=True, exist_ok=True)
    write_header = not RETIRED_PATH.exists()
    with RETIRED_PATH.open("a", encoding="utf-8") as fh:
        if write_header:
            fh.write(json.dumps(_RETIRED_SOURCE_ROW, ensure_ascii=False) + "\n")
        for row in retired_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Reporting.
    retired_etymons = sum(1 for r in retired_rows if r.get("_type") == "etymon")
    retired_citations = sum(1 for r in retired_rows if r.get("_type") == "citation")
    print()
    print(f"Active file:   {ACTIVE_PATH} ({len(active_rows)} rows)")
    print(
        f"Tombstone:     {RETIRED_PATH} ({retired_etymons} etymons + {retired_citations} citations)"
    )

    print()
    print("Per-language breakdown:")
    by_lang: Counter[str] = Counter()
    for row in retired_rows:
        if row.get("_type") != "etymon":
            continue
        by_lang[row["ref"].split(":", 1)[0]] += 1
    for lang, n in sorted(by_lang.items(), key=lambda kv: -kv[1]):
        print(f"  {lang:25s} {n:>5d}")


if __name__ == "__main__":
    main()
