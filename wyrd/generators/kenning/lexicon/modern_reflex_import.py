"""Import curated modern-English reflexes (wyrd-vewk).

Many English-family morphemes (OE/ON/OF/ME) have no modern-English reflex in
their cognate cluster/descent, so their era-grid modern cell renders empty
(``-ham`` had no ``home``; ``holmr`` no ``holm``). ``data/mining/
_modern_reflexes.jsonl`` is a hand-curated, referenced L2 source mapping each
such morpheme to its modern-English reflex form(s). This importer lands it.

For every entry with ``modern_forms``, per form it:
  * upserts a modern-english etymon (``canonical_form=form``), attaching the
    curated gloss + a citation to the synthetic ``modern-reflex-curation``
    source;
  * adds an ``inheritance`` ``etymon_descent`` edge morpheme -> reflex. This
    edge is the durable provenance: ``cluster_cognates`` re-derives the cluster
    from it on a from-scratch rebuild, and the ``etymon_era_reflexes`` Tier-2
    path uses it to surface the reflex for un-clustered morphemes;
  * when the morpheme is in a cognate cluster and the (new) reflex isn't,
    sets ``reflex.cognate_id = morpheme.cognate_id`` so the era-grid Tier-1
    (cluster) lookup surfaces it immediately — this is exactly what
    ``cluster_cognates`` would compute from the descent edge, so running this
    pass AFTER cluster-cognates is correct and needs no global re-cluster.

``modern_forms: []`` rows ("none" — a morpheme with no genuine modern reflex)
are skipped. Fully idempotent: upserts + ``INSERT OR IGNORE`` descent edges +
a cognate_id set guarded on the reflex being unclustered.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

MODERN_REFLEX_SOURCE = "modern-reflex-curation"
_SOURCE_TITLE = "Curated modern-English reflexes (wyrd-vewk)"
MODERN_REFLEX_COGNATE_METHOD = "modern-reflex-curation-v1"


def import_modern_reflexes(db, jsonl_path: str | Path, *, apply: bool = True) -> dict:
    """Apply ``_modern_reflexes.jsonl`` to ``db`` (a ``LexiconDB``). Returns a
    counts dict. ``apply=False`` reports what WOULD change without writing."""
    path = Path(jsonl_path)
    counts: Counter = Counter()
    if not path.is_file():
        return {"file_missing": 1}

    conn = db.conn
    if apply:
        db.upsert_source(
            id=MODERN_REFLEX_SOURCE,
            title=_SOURCE_TITLE,
            notes="Curated OE/ON/OF/ME morpheme -> modern-English reflex links (wyrd-vewk).",
        )

    for lineno, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: malformed JSONL: {exc}") from exc
        if row.get("_type") != "modern_reflex":
            continue
        forms = row.get("modern_forms") or []
        if not forms:
            counts["skipped_no_reflex"] += 1
            continue
        lang, _sep, canon = (row.get("etymon_ref") or "").partition(":")
        morpheme = conn.execute(
            "SELECT id, cognate_id FROM etymon "
            "WHERE canonical_form=? AND language=? AND merged_into_id IS NULL",
            (canon, lang),
        ).fetchone()
        if morpheme is None:
            counts["morpheme_missing"] += 1
            continue
        m_id, m_cog = morpheme[0], morpheme[1]
        for form in forms:
            existing = conn.execute(
                "SELECT id, cognate_id FROM etymon "
                "WHERE canonical_form=? AND language='modern-english' AND merged_into_id IS NULL",
                (form,),
            ).fetchone()
            if existing is not None:
                reflex_id, reflex_cog = existing[0], existing[1]
                counts["reflex_existing"] += 1
            else:
                reflex_cog = None
                if apply:
                    reflex_id = db.upsert_etymon(form, "modern-english")
                    counts["reflex_created"] += 1
                else:
                    counts["reflex_would_create"] += 1
            if not apply:
                # total links this run WOULD make (both existing + to-create
                # reflexes); reflex_existing/reflex_would_create give the split.
                counts["would_link"] += 1
                continue
            if row.get("gloss"):
                db.add_gloss(reflex_id, row["gloss"])
            # preserve the curated scholarly reference as the citation provenance.
            db.add_citation(reflex_id, MODERN_REFLEX_SOURCE, short_quote=row.get("reference"))
            conn.execute(
                "INSERT OR IGNORE INTO etymon_descent"
                "(parent_id, child_id, edge_type, source_id, confidence, notes) "
                "VALUES (?, ?, 'inheritance', ?, ?, ?)",
                (
                    m_id,
                    reflex_id,
                    MODERN_REFLEX_SOURCE,
                    row.get("confidence"),
                    "curated modern reflex (wyrd-vewk)",
                ),
            )
            counts["descent_edges"] += 1
            if m_cog is not None and reflex_cog is None:
                conn.execute(
                    "UPDATE etymon SET cognate_id=?, cognate_method=? WHERE id=?",
                    (m_cog, MODERN_REFLEX_COGNATE_METHOD, reflex_id),
                )
                counts["clustered"] += 1
            elif m_cog is None:
                # morpheme has no cluster — the descent edge drives Tier-2.
                counts["descent_only"] += 1
            else:
                # reflex already in its own cluster — leave it; the descent edge
                # records the link (re-cluster would merge if ever wanted).
                counts["reflex_pre_clustered"] += 1
    if apply:
        db.commit()
    return dict(counts)
