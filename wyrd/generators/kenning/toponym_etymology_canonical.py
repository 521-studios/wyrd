"""Multi-extractor consensus canonicalization for toponym etymologies (wyrd-08qv).

Three different LLM extractors (qwen / haiku / gemini) running over the
same source body produce three ``toponym_etymology`` rows per toponym.
When they agree on a decomposition the answer is settled; future
re-extraction passes can skip the toponym and downstream consumers
(proportions, generation, validation) can read one canonical row
instead of disambiguating.

This module clusters the rows by normalized element-tuple and language,
counts the number of distinct extractor witnesses per cluster, and
promotes the largest agreeing cluster to canonical when it has ≥2
witnesses. Witnesses are identified by the ``extracted_by:provider:model``
prefix the existing ingest pipelines write into the ``notes`` field —
two rows with the same extractor (e.g. two qwen passes) count as ONE
witness, not two.

The clustering normalization is deliberately conservative: lowercase,
strip diacritics + macrons + Mawer's parenthetical "(a)" notation, strip
whitespace + hyphens. Two extractors emitting ``tūn`` and ``tun`` count
as agreement (the macron is a transliteration convention, not semantic
difference); two extractors emitting ``tun`` and ``thun`` do NOT (the
``th-`` is a real spelling variant Mawer's notes call out as
phonologically meaningful).

The companion CLI is ``wyrd kenning lexicon canonicalize-toponym-etymology``.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field

from wyrd.generators.kenning.lexicon import LexiconDB

# Captures the "extracted_by:provider:model" prefix the ingest pipelines
# write into toponym_etymology.notes. Tolerates legacy variants:
#   - "extracted_by:llm:qwen3.5:9b; …" (semicolon-delimited)
#   - "extracted_by:anthropic:claude-haiku-4-5-20251001; …"
#   - "extracted_by:gemini:gemini-2.5-flash | …" (pipe-delimited)
# The captured group is the FULL provider:model identifier so two
# different ingest passes by the same extractor count as ONE witness.
_EXTRACTOR_RE = re.compile(r"^extracted_by:([^;|]+?)(?:\s*[;|]|$)")


# Mawer's "(a)" notation marks an optional inflectional vowel in the
# Old English form (e.g. Bedeling(a) = Bedeling or Bedelinga). Both
# variants normalize to the same morphological root, so we strip the
# parenthetical for clustering. Same logic strips bracketed [...] notes.
_PAREN_OPTIONAL_RE = re.compile(r"\([^)]*\)|\[[^\]]*\]")


def _normalize_form(canonical_form: str) -> str:
    """Normalize an etymon canonical_form for cluster-key comparison.

    Lowercase, strip diacritics (NFKD → ASCII), strip Mawer's
    parenthetical "(a)" notation, drop whitespace + hyphens. Two
    forms compare equal iff their normalized forms match.

    Examples:
      "tūn" → "tun"
      "Bedeling(a)" → "bedeling"
      "tun (Old English)" → "tunoldenglish"  (parenthetical content
        stripped; in practice the ``language`` field carries that
        info separately, so the canonical_form should not include it)
    """
    if not canonical_form:
        return ""
    # NFKD decomposes diacritics into base + combining marks; drop the
    # combining marks. tūn (U+016B) → tun (U+0074 U+0075 U+006E).
    decomposed = unicodedata.normalize("NFKD", canonical_form)
    no_diacritics = "".join(c for c in decomposed if not unicodedata.combining(c))
    lowered = no_diacritics.lower()
    no_paren = _PAREN_OPTIONAL_RE.sub("", lowered)
    # Strip whitespace, hyphens, asterisks (Indo-Europeanist
    # reconstruction marker). Underscores left in — they show up in
    # some forms as morpheme boundaries that ARE semantically real.
    no_punct = re.sub(r"[\s\-*]+", "", no_paren)
    return no_punct


def _extractor_from_notes(notes: str | None) -> str | None:
    """Parse the ``extracted_by:provider:model`` prefix from a notes
    field. Returns the provider:model identifier or None when the
    notes don't carry the prefix (legacy rows, hand-curated rows).

    A row with no extractor identifier counts as its own witness — it's
    distinct from any LLM-emitted row that has a known extractor."""
    if not notes:
        return None
    m = _EXTRACTOR_RE.match(notes.strip())
    return m.group(1).strip() if m else None


@dataclass
class _EtymologyRow:
    """One toponym_etymology row + its element tuple + extractor tag.

    Carried in memory for the duration of a single canonicalize pass.
    """

    id: int
    toponym_id: int
    notes: str | None
    extractor: str | None  # parsed from notes, may be None
    # Ordered tuple of (normalized_form, language) per element.
    elements: tuple[tuple[str, str], ...]


@dataclass
class _Cluster:
    """A group of rows agreeing on element-tuple. Multiple distinct
    extractors witnessing the same cluster bumps its witness count;
    two rows from the SAME extractor count as one witness."""

    cluster_key: str
    rows: list[_EtymologyRow] = field(default_factory=list)
    witnesses: set[str] = field(default_factory=set)

    def add(self, row: _EtymologyRow) -> None:
        self.rows.append(row)
        if row.extractor is not None:
            self.witnesses.add(row.extractor)
        else:
            # Anonymous-witness rows still count: use the row id as a
            # synthetic witness key so a hand-curated row IS a witness,
            # but two different rows with no extractor stay distinct.
            self.witnesses.add(f"_anon:{row.id}")

    @property
    def witness_count(self) -> int:
        return len(self.witnesses)


def _build_cluster_key(elements: tuple[tuple[str, str], ...]) -> str:
    """Serialize an element-tuple into a stable string key.

    Pipe-delimited ``norm-form:language`` pairs in ordinal order. The
    cluster_key is persisted on the row (toponym_etymology.cluster_key)
    so operators can grep the canonicalize-run output and see what
    cluster a row belongs to without re-running the clustering logic.
    """
    return "|".join(f"{form}:{lang}" for form, lang in elements)


@dataclass
class CanonicalDecision:
    """One toponym's canonicalization outcome. Emitted for the audit
    log so an operator can see which clusters won, which were
    rejected as single-witness, and which toponyms had no clusters
    eligible for canonicalization.
    """

    toponym_id: int
    promoted_etymology_id: int | None  # None when no cluster won
    consensus_size: int  # witness count of the winning cluster (or 0)
    cluster_key: str | None
    runner_up_witness_count: int  # for tie-detection visibility
    total_clusters: int  # how many distinct hypotheses for this toponym


def _select_winning_cluster(clusters: list[_Cluster]) -> _Cluster | None:
    """Pick the canonical cluster from candidate clusters for one toponym.

    Criteria, in order:

    1. Witness count >= 2 (single-witness clusters are not canonical —
       even if the toponym has no competing hypotheses, one extractor
       alone isn't consensus).
    2. Among eligible clusters, the largest witness count wins.
    3. Ties broken by the lowest-id row in the cluster (deterministic
       so re-running over the same data picks the same winner).

    Returns None when no cluster has >=2 witnesses.
    """
    eligible = [c for c in clusters if c.witness_count >= 2]
    if not eligible:
        return None
    eligible.sort(key=lambda c: (-c.witness_count, min(r.id for r in c.rows)))
    return eligible[0]


def _select_promoted_row(cluster: _Cluster) -> int:
    """Within the winning cluster, pick which row to mark is_canonical=1.

    Deterministic: lowest row id wins. The other rows in the cluster
    stay is_canonical=0 but get their consensus_size set so the audit
    trail still shows they participated in the agreement.
    """
    return min(r.id for r in cluster.rows)


def compute_canonical_decisions(
    db: LexiconDB,
    *,
    source_id: str | None = None,
) -> list[CanonicalDecision]:
    """Compute the canonical-promotion plan for every toponym (or just
    those with rows from ``source_id`` if given).

    Pure read — does NOT mutate the DB. The companion ``apply_*``
    function writes the decisions in one transaction.
    """
    # Load rows + elements for the in-scope toponyms.
    if source_id is None:
        toponym_ids = [
            r["toponym_id"]
            for r in db.conn.execute(
                "SELECT DISTINCT toponym_id FROM toponym_etymology ORDER BY toponym_id"
            )
        ]
    else:
        toponym_ids = [
            r["toponym_id"]
            for r in db.conn.execute(
                "SELECT DISTINCT toponym_id FROM toponym_etymology WHERE source_id = ? "
                "ORDER BY toponym_id",
                (source_id,),
            )
        ]

    decisions: list[CanonicalDecision] = []
    for tid in toponym_ids:
        rows = _load_rows_for_toponym(db, tid)
        if not rows:
            continue
        clusters_by_key: dict[str, _Cluster] = {}
        for row in rows:
            key = _build_cluster_key(row.elements)
            if key not in clusters_by_key:
                clusters_by_key[key] = _Cluster(cluster_key=key)
            clusters_by_key[key].add(row)
        clusters = list(clusters_by_key.values())
        winner = _select_winning_cluster(clusters)
        if winner is None:
            runner_up = max((c.witness_count for c in clusters), default=0)
            decisions.append(
                CanonicalDecision(
                    toponym_id=tid,
                    promoted_etymology_id=None,
                    consensus_size=0,
                    cluster_key=None,
                    runner_up_witness_count=runner_up,
                    total_clusters=len(clusters),
                )
            )
            continue
        # Find the SECOND-place cluster's witness count so the audit
        # trail can surface "winner had N, runner-up had M" — useful
        # for spotting borderline canonicalizations (e.g. 2 vs 2 tie
        # broken on id, where an operator might want to look closer).
        ranked = sorted(
            (c.witness_count for c in clusters if c is not winner),
            reverse=True,
        )
        runner_up = ranked[0] if ranked else 0
        decisions.append(
            CanonicalDecision(
                toponym_id=tid,
                promoted_etymology_id=_select_promoted_row(winner),
                consensus_size=winner.witness_count,
                cluster_key=winner.cluster_key,
                runner_up_witness_count=runner_up,
                total_clusters=len(clusters),
            )
        )
    return decisions


def _load_rows_for_toponym(db: LexiconDB, toponym_id: int) -> list[_EtymologyRow]:
    """Load all toponym_etymology rows + their element tuples for one
    toponym. The element tuple is built in ordinal order — that order
    is part of the cluster key, since "tun + bedeling" and
    "bedeling + tun" are semantically different decompositions.
    """
    ety_rows = list(
        db.conn.execute(
            "SELECT id, toponym_id, notes FROM toponym_etymology WHERE toponym_id = ? ORDER BY id",
            (toponym_id,),
        )
    )
    rows: list[_EtymologyRow] = []
    for ety in ety_rows:
        element_rows = list(
            db.conn.execute(
                "SELECT e.canonical_form, e.language "
                "FROM toponym_etymology_element tee "
                "JOIN etymon e ON e.id = tee.etymon_id "
                "WHERE tee.toponym_etymology_id = ? "
                "ORDER BY tee.ordinal",
                (ety["id"],),
            )
        )
        elements = tuple(
            (_normalize_form(r["canonical_form"]), (r["language"] or "").lower())
            for r in element_rows
        )
        rows.append(
            _EtymologyRow(
                id=ety["id"],
                toponym_id=ety["toponym_id"],
                notes=ety["notes"],
                extractor=_extractor_from_notes(ety["notes"]),
                elements=elements,
            )
        )
    return rows


@dataclass
class CanonicalizeSummary:
    """Aggregate counters for a canonicalize run. Surfaces to the CLI
    so operators see how many toponyms were promoted, how many were
    untouched (single-witness or all-disagreement), and how many
    rows were stamped with cluster_key + consensus_size."""

    toponyms_examined: int = 0
    toponyms_promoted: int = 0
    toponyms_no_consensus: int = 0  # had clusters but none reached 2 witnesses
    toponyms_no_elements: int = 0  # rows existed but none had elements
    rows_updated: int = 0  # is_canonical OR cluster_key/consensus_size changed


def apply_canonical_decisions(
    db: LexiconDB,
    decisions: list[CanonicalDecision],
    *,
    on_decision: Callable[[CanonicalDecision], None] | None = None,
) -> CanonicalizeSummary:
    """Write the canonical-promotion plan to the DB in one transaction.

    For each decision:
      * Promoted row gets is_canonical=1, consensus_size=N, cluster_key=K
      * Other rows in the SAME cluster get is_canonical=0,
        consensus_size=N, cluster_key=K
      * Rows in OTHER clusters get is_canonical=0,
        consensus_size=<their cluster's witness count>,
        cluster_key=<their cluster's key>
      * Toponyms with no winning cluster get all rows' is_canonical
        reset to 0 (a re-canonicalize pass over previously-canonical
        data must demote stale canonicals when consensus is lost).

    ``on_decision`` fires per decision for audit logging — the CLI
    wires it to a JSONL writer.
    """
    summary = CanonicalizeSummary()

    # Pre-load rows-per-cluster for every toponym we have a decision
    # for; we need to update consensus_size + cluster_key on EVERY row,
    # not just the canonical one, so the audit trail is complete.
    for decision in decisions:
        summary.toponyms_examined += 1
        rows = _load_rows_for_toponym(db, decision.toponym_id)
        if not rows:
            continue
        # Bucket rows by their cluster_key.
        clusters: dict[str, list[_EtymologyRow]] = {}
        for row in rows:
            key = _build_cluster_key(row.elements)
            clusters.setdefault(key, []).append(row)
        # Need the per-cluster witness count too — recompute (matches
        # what compute_canonical_decisions saw).
        witness_count_by_key: dict[str, int] = {}
        for key, cluster_rows in clusters.items():
            witnesses: set[str] = set()
            for row in cluster_rows:
                if row.extractor is not None:
                    witnesses.add(row.extractor)
                else:
                    witnesses.add(f"_anon:{row.id}")
            witness_count_by_key[key] = len(witnesses)
        # Detect "no elements anywhere" so we can count it separately.
        if all(not row.elements for row in rows):
            summary.toponyms_no_elements += 1
            # Reset any stale canonicalization on this toponym.
            for row in rows:
                _write_row_canonical(db, row.id, 0, 1, None, summary)
            if on_decision is not None:
                on_decision(decision)
            continue
        # Write each row.
        for row in rows:
            key = _build_cluster_key(row.elements)
            is_canon = 1 if row.id == decision.promoted_etymology_id else 0
            cs = witness_count_by_key[key]
            _write_row_canonical(db, row.id, is_canon, cs, key, summary)
        if decision.promoted_etymology_id is not None:
            summary.toponyms_promoted += 1
        else:
            summary.toponyms_no_consensus += 1
        if on_decision is not None:
            on_decision(decision)
    db.commit()
    return summary


def _write_row_canonical(
    db: LexiconDB,
    row_id: int,
    is_canonical: int,
    consensus_size: int,
    cluster_key: str | None,
    summary: CanonicalizeSummary,
) -> None:
    """Update one toponym_etymology row's canonical fields.

    Only counts toward summary.rows_updated when at least one field
    actually changed — re-canonicalizing a no-op DB shouldn't inflate
    the count.
    """
    cur = db.conn.execute(
        "UPDATE toponym_etymology "
        "SET is_canonical = ?, consensus_size = ?, cluster_key = ? "
        "WHERE id = ? AND ("
        "  is_canonical IS NOT ? OR consensus_size IS NOT ? OR cluster_key IS NOT ?"
        ")",
        (
            is_canonical,
            consensus_size,
            cluster_key,
            row_id,
            is_canonical,
            consensus_size,
            cluster_key,
        ),
    )
    if cur.rowcount > 0:
        summary.rows_updated += 1
