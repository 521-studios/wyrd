"""Manual meaning-synset clustering (wyrd-7tz Phase 1).

Group etymons that share a meaning into ``meaning_synset`` clusters
so the bundle can roll up multi-language witness counts across the
group. Seed data is hand-curated (the seed-from-JSON pass below);
Phase 2 will add an LLM-assisted candidate proposer that hangs off
``get_meaning_preserving_candidates``.

* ``seed_meaning_synsets`` loads the static seed JSON into the
  ``meaning_synset`` + ``meaning_synset_member`` tables.
* ``list_meaning_synsets`` enumerates synsets for the CLI.
* ``assign_etymon_to_meaning_synset`` is the per-etymon write path.
* ``get_meaning_synsets_for_etymon`` is the per-etymon read path.
* ``get_meaning_preserving_candidates`` finds etymons whose glosses
  overlap a seed synset's centroid gloss — feed for the Phase 2
  proposer.
"""

from __future__ import annotations

import json

from wyrd.generators.kenning.lexicon.db import LexiconDB
from wyrd.generators.kenning.paths import seed_data_path


def _meaning_synsets_seed() -> dict:
    """Read the bundled seed catalog from package data."""
    raw = seed_data_path("meaning_synsets.json").read_text(encoding="utf-8")
    return json.loads(raw)


def seed_meaning_synsets(db: LexiconDB) -> dict[str, int]:
    """Idempotently populate meaning_synset from the bundled catalog.

    Inserts canonical_label rows that don't yet exist; for existing
    rows, updates hypernym + notes if they drifted. Returns
    {'inserted': N, 'updated': M, 'unchanged': K} so callers can
    report.

    Hypernyms resolve in a two-pass walk so a child can be declared
    before its parent in the JSON without breaking the FK. Pass one
    inserts/updates rows with hypernym=NULL; pass two writes the
    hypernym links once every label has a known id.
    """
    catalog = _meaning_synsets_seed()
    entries: list[dict] = catalog["synsets"]
    inserted = updated = unchanged = 0
    # Pass 1: rows with no hypernym dependency. Use INSERT OR IGNORE
    # for the canonical_label uniqueness; update notes separately so
    # we can count drift.
    label_to_id: dict[str, int] = {}
    for entry in entries:
        label = entry["label"]
        notes = entry.get("notes")
        cur = db.conn.execute(
            "SELECT id, notes FROM meaning_synset WHERE canonical_label = ?",
            (label,),
        )
        row = cur.fetchone()
        if row is None:
            cur = db.conn.execute(
                "INSERT INTO meaning_synset (canonical_label, notes) VALUES (?, ?)",
                (label, notes),
            )
            label_to_id[label] = cur.lastrowid
            inserted += 1
            continue
        label_to_id[label] = row["id"]
        if row["notes"] != notes:
            db.conn.execute(
                "UPDATE meaning_synset SET notes = ? WHERE id = ?",
                (notes, row["id"]),
            )
            updated += 1
        else:
            unchanged += 1
    # Pass 2: hypernym links. Always run on every entry so a hypernym
    # removed from the JSON file gets its FK NULL'd out (drift-aware
    # like the notes update in pass 1). Missing hypernyms (typo or
    # out-of-catalog) emit a warning so seed-file typos surface, but
    # don't fail the whole seed run — the row keeps its current
    # hypernym_id.
    import warnings as _warnings

    for entry in entries:
        hypernym_label = entry.get("hypernym")
        if hypernym_label is None:
            db.conn.execute(
                "UPDATE meaning_synset SET hypernym_id = NULL WHERE canonical_label = ?",
                (entry["label"],),
            )
            continue
        if hypernym_label not in label_to_id:
            _warnings.warn(
                f"meaning_synsets.json: synset {entry['label']!r} references "
                f"unknown hypernym {hypernym_label!r}; leaving FK unchanged",
                stacklevel=2,
            )
            continue
        db.conn.execute(
            "UPDATE meaning_synset SET hypernym_id = ? WHERE canonical_label = ?",
            (label_to_id[hypernym_label], entry["label"]),
        )
    db.commit()
    return {"inserted": inserted, "updated": updated, "unchanged": unchanged}


def list_meaning_synsets(db: LexiconDB, *, with_member_counts: bool = False) -> list[dict]:
    """Return all meaning_synset rows ordered by canonical_label.

    With ``with_member_counts=True`` includes a 'member_count' integer
    per row — useful for the CLI to surface which synsets are still
    empty after the LLM pass.
    """
    if with_member_counts:
        sql = """
            SELECT s.id, s.canonical_label, s.hypernym_id, s.notes,
                   COUNT(ems.etymon_id) AS member_count
            FROM meaning_synset s
            LEFT JOIN etymon_meaning_synset ems ON ems.meaning_synset_id = s.id
            GROUP BY s.id
            ORDER BY s.canonical_label
        """
    else:
        sql = """
            SELECT id, canonical_label, hypernym_id, notes
            FROM meaning_synset
            ORDER BY canonical_label
        """
    return [dict(row) for row in db.conn.execute(sql)]


def assign_etymon_to_meaning_synset(
    db: LexiconDB,
    etymon_id: int,
    synset_label: str,
    *,
    fit: str = "core",
) -> bool:
    """Add a (etymon, meaning_synset) membership row, or update its fit
    if the row already exists. Returns True on first-time insert,
    False on update / unchanged.

    Raises ValueError on an unknown etymon_id, unknown synset_label, or
    bad fit value — fail loudly so a typo at the CLI surface doesn't
    silently insert nothing.
    """
    if fit not in ("core", "peripheral"):
        raise ValueError(f"fit must be 'core' or 'peripheral', got {fit!r}")
    etymon_row = db.conn.execute("SELECT id FROM etymon WHERE id = ?", (etymon_id,)).fetchone()
    if etymon_row is None:
        raise ValueError(f"unknown etymon_id: {etymon_id}")
    synset_row = db.conn.execute(
        "SELECT id FROM meaning_synset WHERE canonical_label = ?",
        (synset_label,),
    ).fetchone()
    if synset_row is None:
        raise ValueError(f"unknown meaning_synset label: {synset_label!r}")
    existing = db.conn.execute(
        """
        SELECT fit FROM etymon_meaning_synset
        WHERE etymon_id = ? AND meaning_synset_id = ?
        """,
        (etymon_id, synset_row["id"]),
    ).fetchone()
    if existing is None:
        db.conn.execute(
            """
            INSERT INTO etymon_meaning_synset (etymon_id, meaning_synset_id, fit)
            VALUES (?, ?, ?)
            """,
            (etymon_id, synset_row["id"], fit),
        )
        db.commit()
        return True
    if existing["fit"] != fit:
        db.conn.execute(
            """
            UPDATE etymon_meaning_synset SET fit = ?
            WHERE etymon_id = ? AND meaning_synset_id = ?
            """,
            (fit, etymon_id, synset_row["id"]),
        )
        db.commit()
    return False


def get_meaning_synsets_for_etymon(db: LexiconDB, etymon_id: int) -> list[dict]:
    """Return the synset rows an etymon belongs to, ordered by fit
    ('core' first) then canonical_label."""
    return [
        dict(row)
        for row in db.conn.execute(
            """
            SELECT s.id, s.canonical_label, s.hypernym_id, ems.fit
            FROM etymon_meaning_synset ems
            JOIN meaning_synset s ON s.id = ems.meaning_synset_id
            WHERE ems.etymon_id = ?
            ORDER BY CASE ems.fit WHEN 'core' THEN 0 ELSE 1 END, s.canonical_label
            """,
            (etymon_id,),
        )
    ]


def get_meaning_preserving_candidates(
    db: LexiconDB,
    etymon_id: int,
    *,
    target_language: str | None = None,
    fit: str | None = None,
    include_self: bool = False,
    dedupe: bool = True,
) -> list[dict]:
    """Return etymons that share at least one meaning_synset with the
    target etymon — i.e. candidates for a meaning-preserving
    substitution by the upcoming Lab transforms.

    Each result row carries:
      - etymon_id, canonical_form, language
      - synset_label, meaning_synset_id (the shared meaning_synset)
      - target_fit (the source etymon's fit in this synset)
      - candidate_fit (the candidate etymon's fit)

    ``dedupe`` (default True) collapses (target, candidate) pairs that
    share multiple synsets into one row per candidate, picking the
    'best' shared synset by fit ('core' beats 'peripheral'); ties break
    on canonical_label. This is what the Phase 3 transforms want
    (replace-root etc. need ONE candidate per etymon, not one per
    shared synset). Pass ``dedupe=False`` to get the raw cartesian for
    introspection — the audit/debugging case.

    ``target_language`` restricts candidates to one language (e.g. for
    anglicize/foreignize transforms). ``fit`` restricts BOTH the target
    side and the candidate side (e.g. 'core' for high-confidence
    substitutions only). ``include_self`` keeps the target etymon in
    the result set; default is to exclude it.
    """
    if fit is not None and fit not in ("core", "peripheral"):
        raise ValueError(f"fit filter must be 'core' or 'peripheral', got {fit!r}")
    sql = """
        SELECT
          ems_other.etymon_id              AS etymon_id,
          e.canonical_form                 AS canonical_form,
          e.language                       AS language,
          s.canonical_label                AS synset_label,
          s.id                             AS meaning_synset_id,
          ems_self.fit                     AS target_fit,
          ems_other.fit                    AS candidate_fit
        FROM etymon_meaning_synset ems_self
        JOIN meaning_synset s         ON s.id = ems_self.meaning_synset_id
        JOIN etymon_meaning_synset ems_other ON ems_other.meaning_synset_id = s.id
        JOIN etymon e                 ON e.id = ems_other.etymon_id
        WHERE ems_self.etymon_id = ?
    """
    params: list = [etymon_id]
    if not include_self:
        sql += " AND ems_other.etymon_id != ?"
        params.append(etymon_id)
    if target_language is not None:
        sql += " AND e.language = ?"
        params.append(target_language)
    if fit is not None:
        sql += " AND ems_self.fit = ? AND ems_other.fit = ?"
        params.extend([fit, fit])
    sql += " ORDER BY s.canonical_label, e.language, e.canonical_form"
    rows = [dict(row) for row in db.conn.execute(sql, params)]
    if not dedupe:
        return rows
    # Collapse to one row per candidate etymon. Sort the per-candidate
    # rows by (target_fit_priority, candidate_fit_priority,
    # synset_label) so 'core/core' beats 'core/peripheral' beats
    # 'peripheral/peripheral'; ties break on label for determinism.
    fit_rank = {"core": 0, "peripheral": 1}
    by_candidate: dict[int, dict] = {}
    for row in rows:
        eid = row["etymon_id"]
        existing = by_candidate.get(eid)
        if existing is None:
            by_candidate[eid] = row
            continue
        new_key = (
            fit_rank[row["target_fit"]],
            fit_rank[row["candidate_fit"]],
            row["synset_label"],
        )
        old_key = (
            fit_rank[existing["target_fit"]],
            fit_rank[existing["candidate_fit"]],
            existing["synset_label"],
        )
        if new_key < old_key:
            by_candidate[eid] = row
    return sorted(
        by_candidate.values(),
        key=lambda r: (r["language"], r["canonical_form"]),
    )
