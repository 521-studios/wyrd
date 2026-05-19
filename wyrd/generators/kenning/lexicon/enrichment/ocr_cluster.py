"""Non-destructive OCR variant clustering (D22 / wyrd-et0).

Merge OCR-mangled spellings (``Hædan`` / ``Hcsdan`` / ``Haedan``) into
a single canonical etymon without deleting the losers. Per D22, the
losing rows get their ``merged_into_id`` set to point at the canonical
winner; citations / glosses / tags / text-match rows stay attached to
the loser exactly where they were originally written, and the
``etymon_consensus`` / ``etymon_*_canonical`` views roll them up via
the chain.

The non-destructive shape lets ``clear_enrichment(stage="ocr")``
revert the ``merged_into_id`` tombstones and re-run clustering
with a new heuristic — no re-mining needed. Note that ``stage="ocr"``
alone leaves behind the ``lemma_id`` re-parenting this pass does
to inflected children at merge time; for a fully clean revert
use ``stage="all-derived"``, which clears both columns.

Flatten-at-write-time rules keep the consensus rollup correct without
recursive CTEs:

* Lemma children of a loser are re-parented to the canonical
  destination at merge time.
* Existing redirects pointing at the loser are also re-routed to
  canonical, so no ``X → loser → canonical`` chain forms.

Both rules are mining-evidence-preserving (no citation/gloss data
moves) but they are NOT cosmetic — a 2-deep chain through
``etymon_consensus`` would split witnesses across two GROUP BY buckets,
undercounting the canonical morpheme's witness total.
"""

from __future__ import annotations

from wyrd.generators.kenning.lexicon.constants import normalize_ocr_form
from wyrd.generators.kenning.lexicon.db import LexiconDB


def cluster_ocr_variants(db: LexiconDB, *, apply: bool = False) -> dict:
    """Find etymons that differ only in OCR ligature variation, mark them
    merged into a canonical winner.

    Groups every NOT-yet-merged etymon by (language,
    normalize_ocr_form(canonical_form)). Within each group of >1 row, the
    row with the LOWEST id is taken as canonical, and the others get
    `merged_into_id` pointing at the canonical row's lemma_id (if the
    winner is itself an inflected variant) or the winner itself.

    Per D22, the merge is non-destructive: citations, glosses, tags,
    text-match evidence, and reflex links stay attached to the loser
    etymon. The `etymon_consensus` view rolls citation witness counts
    up to the canonical row via the `merged_into_id` column. Gloss,
    tag, and text-match data stay attached to the original etymon ids;
    consumers wanting "all evidence for this canonical group" should
    JOIN through merged_into_id (or wait for wyrd-7lo's per-table
    canonical-rollup views).

    To revert, run `clear_enrichment(stage="ocr")` (or `UPDATE etymon
    SET merged_into_id = NULL`); the underlying mining-evidence rows
    are intact so a subsequent `cluster_ocr_variants` run with new
    heuristics will see fresh state.

    With apply=False (default) we only report what would happen — no
    writes. Pass apply=True to actually mark the merges.

    Returns a dict with keys:
      groups          - count of (lang, normalized) groups with >1 etymon
      etymons_merged  - count of redundant etymons that would be marked merged
      sample_groups   - first 20 example groupings, for human review
    """
    # Skip already-merged rows on re-run — they're tombstones, not merge
    # candidates. New heuristics should clear merged_into_id first
    # (clear_enrichment stage='ocr') before re-clustering.
    cur = db.conn.execute(
        "SELECT id, canonical_form, language, lemma_id FROM etymon "
        "WHERE merged_into_id IS NULL ORDER BY id"
    )
    groups: dict[tuple[str, str], list[tuple[int, str, int | None]]] = {}
    for row in cur:
        key = (row["language"], normalize_ocr_form(row["canonical_form"]))
        groups.setdefault(key, []).append((row["id"], row["canonical_form"], row["lemma_id"]))

    duplicate_groups = {k: v for k, v in groups.items() if len(v) > 1}
    sample = []
    for (lang, norm), members in list(duplicate_groups.items())[:20]:
        sample.append(
            {
                "language": lang,
                "normalized": norm,
                "members": [m[1] for m in members],
                "winner": members[0][1],
            }
        )
    counts = {
        "groups": len(duplicate_groups),
        "etymons_merged": sum(len(v) - 1 for v in duplicate_groups.values()),
        "sample_groups": sample,
    }
    if not apply:
        return counts

    # Apply the merges. Lowest id wins; the rest are tombstoned via
    # merged_into_id.
    for members in duplicate_groups.values():
        winner_id, _, winner_lemma_id = members[0]
        loser_ids = [m[0] for m in members[1:]]
        # If the winner is itself an inflected variant of a lemma, point
        # losers directly at the lemma so the rollup chain stays at depth
        # 1 on the lemma_id axis.
        canonical_id = winner_lemma_id if winner_lemma_id is not None else winner_id

        # Self-reference safety: if the canonical destination is another
        # member of this same cluster (i.e., the lemma row sits in the
        # cluster too), that row must be the winner rather than a loser.
        # Otherwise we'd write loser.merged_into_id = self_id. Promote
        # the lemma; demote the previous winner.
        if canonical_id in loser_ids:
            promoted = canonical_id
            loser_ids = [lid for lid in loser_ids if lid != promoted]
            loser_ids.append(winner_id)
            winner_id = promoted
            canonical_id = winner_id

        # Chain-follow: if the canonical destination itself was merged in
        # an earlier pass (e.g., link-lemmas linked the winner to a row
        # that's since been merged), follow merged_into_id to the
        # ultimate canonical. Defends against stale lemma_id pointers
        # that predate the link_lemmas merged-tombstone filter. Tracks
        # visited ids so a malformed cycle (canonical-of-canonical points
        # back) breaks cleanly instead of looping forever.
        visited = {canonical_id}
        while True:
            next_id = db.conn.execute(
                "SELECT merged_into_id FROM etymon WHERE id = ?",
                (canonical_id,),
            ).fetchone()[0]
            if next_id is None or next_id in visited:
                break
            canonical_id = next_id
            visited.add(canonical_id)

        # Batch the two per-loser UPDATEs into one statement each per
        # cluster (wyrd-v3h). OCR clusters are tiny in practice (2-5
        # members) so SQLite's 999-parameter limit is never close — no
        # chunking needed. Citations / glosses / tags / text-match /
        # reflex links stay attached to their original etymons; the
        # consensus view rolls them up via merged_into_id. Mining
        # evidence (D21) is preserved exactly as written.
        #
        # The surrounding loop's invariant (duplicate_groups filtered to
        # len > 1) guarantees ≥1 loser, but the empty-list guard keeps
        # the IN (...) form from raising a SQLite syntax error if a
        # future refactor changes that invariant.
        if not loser_ids:
            continue
        placeholders = ",".join("?" for _ in loser_ids)
        # Re-parent any inflected children the losers were acting as a
        # lemma for, so consensus rolls them into the canonical group.
        # The redirect via merged_into_id alone wouldn't suffice
        # because the rollup is single-level on lemma_id.
        db.conn.execute(
            f"UPDATE etymon SET lemma_id = ? WHERE lemma_id IN ({placeholders})",
            (canonical_id, *loser_ids),
        )
        # Mark merged AND flatten any pre-existing redirects so
        # merged_into_id chains can't form. Without the second clause,
        # rows from a prior cluster pass that point at any of these
        # losers would create a 2-deep chain X → loser → canonical, and
        # the single-level COALESCE rollup would split witnesses.
        db.conn.execute(
            f"UPDATE etymon SET merged_into_id = ? "
            f"WHERE id IN ({placeholders}) OR merged_into_id IN ({placeholders})",
            (canonical_id, *loser_ids, *loser_ids),
        )

    db.commit()
    return counts
