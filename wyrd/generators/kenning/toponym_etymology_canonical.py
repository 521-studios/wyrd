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

A row WITHOUT an ``extracted_by:`` prefix (legacy / hand-curated)
contributes ZERO witnesses to its cluster. Consensus requires
multi-extractor agreement; an anonymous row alone (or paired with
other anonymous rows) cannot promote a cluster to canonical. Operators
can still mark hand-curated entries canonical via direct DB update if
needed — this auto-promotion path is for LLM-extractor consensus only.
R1 silent-failure-hunter HIGH: the previous ``_anon:{row.id}`` scheme
treated each anonymous row as its own witness, opening a false-consensus
path (two curated rows on the same toponym would have promoted as if
two distinct extractors agreed).

The clustering normalization is deliberately conservative: lowercase,
strip diacritics + macrons + Mawer's parenthetical "(a)" notation, strip
whitespace + hyphens. Two extractors emitting ``tūn`` and ``tun`` count
as agreement (the macron is a transliteration convention, not semantic
difference); two extractors emitting ``tun`` and ``thun`` do NOT (the
``th-`` is a real spelling variant Mawer's notes call out as
phonologically meaningful).

Cluster keys are JSON-encoded for the persisted ``cluster_key`` column
so a normalized form containing ``|`` or ``:`` can't collide with a
different element-tuple. R1 silent-failure-hunter MEDIUM + type-design
MEDIUM: the previous pipe-delimited "form:lang|form:lang" format could
ambiguously split, e.g. ``("a:b","lang") == ("a","b:lang")``.

The companion CLI is ``wyrd kenning lexicon canonicalize-toponym-etymology``.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NamedTuple

from wyrd.generators.kenning.lexicon import LexiconDB


class _RowUpdate(NamedTuple):
    """One row's target canonical state. Named-tuple so the four
    fields are visible at construction + unpack sites instead of
    being positional in a raw tuple. R2 type-design + clarity MEDIUM
    flagged the previous opaque ``tuple[int, int, int, str | None]``.
    """

    row_id: int
    is_canonical: int
    consensus_size: int
    cluster_key: str | None


# Captures the "extracted_by:provider:model" prefix the ingest pipelines
# write into toponym_etymology.notes. Tolerates legacy variants:
#   - "extracted_by:llm:qwen3.5:9b; …" (semicolon-delimited)
#   - "extracted_by:anthropic:claude-haiku-4-5-20251001; …"
#   - "extracted_by:gemini:gemini-2.5-flash | …" (pipe-delimited)
# Inner colons inside the provider:model identifier are fine — the
# character class `[^;|]+?` excludes only the actual delimiters.
# The captured group is the FULL provider:model identifier so two
# different ingest passes by the same extractor count as ONE witness.
_EXTRACTOR_RE = re.compile(r"^extracted_by:([^;|]+?)(?:\s*[;|]|$)")


# Mawer's "(a)" notation marks an optional inflectional vowel in the
# Old English form (e.g. Bedeling(a) = Bedeling or Bedelinga). Both
# variants normalize to the same morphological root, so we strip the
# parenthetical for clustering. Same logic strips bracketed [...] notes.
_PAREN_OPTIONAL_RE = re.compile(r"\([^)]*\)|\[[^\]]*\]")
# Non-semantic punctuation dropped from a cluster-key form: whitespace, the
# asterisk reconstruction marker, and ANY dash — ASCII hyphen plus the Unicode
# dash variants (U+2010..U+2015 hyphen / NB-hyphen / figure / en / em /
# horizontal-bar, U+2212 minus, U+FE58/FE63/FF0D small+fullwidth). Stripping only
# the ASCII hyphen left ``al–Quadim`` (en-dash) keyed differently from
# ``al-Quadim`` → the same morpheme forks into separate clusters. Mirrors the
# morpheme de-dash dash set (D45); underscores are KEPT (real morpheme boundaries).
_CLUSTER_KEY_DROP_RE = re.compile(r"[\s*\-‐-―−﹘﹣－]+")


def _normalize_form(canonical_form: str) -> str:
    """Normalize an etymon canonical_form for cluster-key comparison.

    Lowercase, strip diacritics (NFKD → ASCII), strip Mawer's
    parenthetical "(a)" notation and bracketed editorial notes, drop
    whitespace + hyphens + asterisks. Two forms compare equal iff
    their normalized forms match.

    Examples:
      "tūn" → "tun"
      "Bedeling(a)" → "bedeling"
      "tun (Old English)" → "tun"  (parenthetical content stripped
        entirely; the ``language`` field carries that info separately,
        so a canonical_form ideally shouldn't include it)
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
    no_punct = _CLUSTER_KEY_DROP_RE.sub("", no_paren)
    return no_punct


def _extractor_from_notes(notes: str | None) -> str | None:
    """Parse the ``extracted_by:provider:model`` prefix from a notes
    field. Returns the provider:model identifier or None when the
    notes don't carry the prefix (legacy rows, hand-curated rows).

    Anonymous rows (return value None here) count as ZERO witnesses
    for consensus purposes — see the ``_Cluster.add`` semantics.

    A malformed prefix that strips to empty (``"extracted_by:   ;..."``)
    is treated as anonymous. R3 silent-failure-hunter LOW: returning
    an empty-string identifier would collapse multiple malformed-notes
    rows into a single ``""`` witness, falsely producing consensus."""
    if not notes:
        return None
    m = _EXTRACTOR_RE.match(notes.strip())
    if not m:
        return None
    extractor = m.group(1).strip()
    return extractor or None


@dataclass(frozen=True)
class _EtymologyRow:
    """One toponym_etymology row + its element tuple + extractor tag.

    Frozen so apply-path can't accidentally mutate a row mid-pass."""

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
    two rows from the SAME extractor count as one witness. Rows
    without an ``extracted_by:`` prefix contribute ZERO witnesses
    (consensus requires multi-extractor agreement)."""

    cluster_key: str
    rows: list[_EtymologyRow] = field(default_factory=list)
    witnesses: set[str] = field(default_factory=set)

    def add(self, row: _EtymologyRow) -> None:
        self.rows.append(row)
        if row.extractor is not None:
            self.witnesses.add(row.extractor)
        # Anonymous rows (extractor is None) are cluster members but
        # NOT witnesses. They still get cluster_key + consensus_size
        # stamped on apply for audit visibility, but they can't push
        # a cluster over the >=2-witness consensus threshold.
        # R1 silent-failure-hunter HIGH: the previous synthetic-id
        # scheme treated each anonymous row as its own witness,
        # opening a false-consensus path.

    @property
    def witness_count(self) -> int:
        return len(self.witnesses)


def _build_cluster_key(elements: tuple[tuple[str, str], ...]) -> str:
    """Serialize an element-tuple into a stable string key.

    JSON-encoded for unambiguous round-trip + collision resistance.
    A normalized form containing ``|`` or ``:`` (rare, but possible
    for reconstructed forms or unusual transliterations) won't
    accidentally collide with a different element-tuple's key.
    Operators grepping the persisted cluster_key column will see e.g.
    ``[["bedelinga","old-english"],["tun","old-english"]]`` —
    greppable enough; correctness is the load-bearing property.
    """
    return json.dumps(list(elements), ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class CanonicalDecision:
    """One toponym's canonicalization outcome. Emitted for the audit
    log so an operator can see which clusters won, which were
    rejected as single-witness, and which toponyms had no clusters
    eligible for canonicalization.

    Frozen so audit-log callbacks can't mutate it before JSONL
    serialization. R1 type-design MEDIUM: the dataclass was billed
    as frozen in the design but the decorator was missing.

    Invariants (enforced by ``compute_canonical_decisions``):

    * ``promoted_etymology_id is None`` ⇔ ``consensus_size == 0`` ⇔
      ``cluster_key is None`` (no-winner case)
    * ``consensus_size >= 2`` when ``promoted_etymology_id is not None``
      (otherwise the cluster wouldn't have been eligible)
    * ``runner_up_witness_count <= consensus_size``
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


@dataclass(frozen=True)
class _ToponymPlan:
    """Per-toponym apply plan. Carries everything ``apply_*`` needs
    without re-querying or re-clustering — eliminates the N+1 reload
    and duplicated witness-counting that R1 (Gemini HIGH + simplifier
    MEDIUM + clarity MEDIUM) flagged."""

    decision: CanonicalDecision
    row_updates: tuple[_RowUpdate, ...]
    # True when this toponym has rows but none have elements at all —
    # apply demotes any stale canonical state but records under the
    # no_elements bucket rather than no_consensus.
    all_empty_elements: bool


def compute_canonical_decisions(  # noqa: V103 — audit-shape API; tests pin canonicalization (PR #498 triage)
    db: LexiconDB,
    *,
    source_id: str | None = None,
) -> list[CanonicalDecision]:
    """Compute the canonical-promotion plan for every toponym (or just
    those with rows from ``source_id`` if given).

    Pure read — does NOT mutate the DB. The companion ``apply_*``
    function writes the decisions in one transaction.

    Use ``compute_canonical_plans`` (the internal builder) when you
    need the apply-side row updates too; this function returns only
    the audit-shape ``CanonicalDecision`` list.
    """
    return [plan.decision for plan in compute_canonical_plans(db, source_id=source_id)]


def compute_canonical_plans(
    db: LexiconDB,
    *,
    source_id: str | None = None,
) -> list[_ToponymPlan]:
    """Compute per-toponym plans. Internal — used by both
    ``compute_canonical_decisions`` (audit shape) and
    ``apply_canonical_decisions`` (write shape) so clustering happens
    exactly ONCE per toponym. R1 Gemini HIGH + clarity MEDIUM:
    previously apply re-loaded rows + re-counted witnesses.
    """
    rows_by_toponym = _load_rows_bulk(db, source_id=source_id)
    plans: list[_ToponymPlan] = []
    for tid in sorted(rows_by_toponym):
        rows = rows_by_toponym[tid]
        if not rows:
            continue
        # All-empty-elements detection FIRST. If every row's element
        # tuple is empty, no cluster is meaningfully promotable even
        # if anonymous-witness collapse would have permitted it. Emit
        # a no-winner decision so the audit log doesn't claim a
        # promotion that apply will override. R1 silent-failure-
        # hunter HIGH.
        all_empty = all(not row.elements for row in rows)
        # Cache per-row cluster key + cluster to avoid redundant
        # JSON serialization in row_updates construction below
        # (R2 simplifier LOW + silent-failure-hunter LOW: was
        # called twice per row).
        keys_by_row_id: dict[int, str] = {}
        clusters_by_key: dict[str, _Cluster] = {}
        for row in rows:
            key = _build_cluster_key(row.elements)
            keys_by_row_id[row.id] = key
            if key not in clusters_by_key:
                clusters_by_key[key] = _Cluster(cluster_key=key)
            clusters_by_key[key].add(row)
        clusters = list(clusters_by_key.values())
        winner = None if all_empty else _select_winning_cluster(clusters)
        if winner is None:
            runner_up = max((c.witness_count for c in clusters), default=0)
            decision = CanonicalDecision(
                toponym_id=tid,
                promoted_etymology_id=None,
                consensus_size=0,
                cluster_key=None,
                runner_up_witness_count=runner_up,
                total_clusters=len(clusters),
            )
        else:
            # SECOND-place cluster's witness count for audit trail —
            # useful for spotting borderline canonicalizations (2 vs
            # 2 tie broken on id, where an operator might want to
            # look closer).
            ranked = sorted(
                (c.witness_count for c in clusters if c is not winner),
                reverse=True,
            )
            runner_up = ranked[0] if ranked else 0
            decision = CanonicalDecision(
                toponym_id=tid,
                promoted_etymology_id=min(r.id for r in winner.rows),
                consensus_size=winner.witness_count,
                cluster_key=winner.cluster_key,
                runner_up_witness_count=runner_up,
                total_clusters=len(clusters),
            )
        # Build the row_updates tuple: which fields apply will write
        # on each row. Stamp cluster_key + consensus_size on every
        # row regardless of canonical status so the audit trail is
        # complete; is_canonical is 1 only for the chosen row.
        # All-empty-elements rows get consensus_size=0 (matches the
        # actual witness count — no elements means no cluster shape
        # to count witnesses against). R2 silent-failure MEDIUM:
        # previously stamped consensus_size=1 which conflated
        # 'no-elements' with 'one-witness'.
        if all_empty:
            row_updates = tuple(_RowUpdate(row.id, 0, 0, None) for row in rows)
        else:
            promoted_id = decision.promoted_etymology_id
            row_updates = tuple(
                _RowUpdate(
                    row.id,
                    1 if (winner is not None and row.id == promoted_id) else 0,
                    clusters_by_key[keys_by_row_id[row.id]].witness_count,
                    keys_by_row_id[row.id],
                )
                for row in rows
            )
        plans.append(
            _ToponymPlan(
                decision=decision,
                row_updates=row_updates,
                all_empty_elements=all_empty,
            )
        )
    return plans


def _load_rows_bulk(
    db: LexiconDB,
    *,
    source_id: str | None = None,
) -> dict[int, list[_EtymologyRow]]:
    """Load all toponym_etymology rows + their elements in TWO BULK
    queries (one for rows, one for elements with IN-clause chunking
    at 500 to stay within SQLite's parameter limit). Grouped by
    toponym_id on return. R2 comment LOW: previously claimed "TWO
    queries total" — strictly accurate for <=500 etymology rows;
    chunking means 1 + ceil(N/500) queries at corpus scale, still
    constant per-row.
    """
    # Restrict toponym set if --source filter is in play. The filter
    # picks which TOPONYMS get canonicalized, NOT which rows
    # participate in their consensus — all rows for an in-scope
    # toponym contribute, regardless of source.
    if source_id is None:
        ety_rows = list(
            db.conn.execute(
                "SELECT id, toponym_id, notes FROM toponym_etymology ORDER BY toponym_id, id"
            )
        )
    else:
        ety_rows = list(
            db.conn.execute(
                "SELECT id, toponym_id, notes FROM toponym_etymology "
                "WHERE toponym_id IN ("
                "  SELECT DISTINCT toponym_id FROM toponym_etymology WHERE source_id = ?"
                ") "
                "ORDER BY toponym_id, id",
                (source_id,),
            )
        )
    if not ety_rows:
        return {}
    ety_ids = [r["id"] for r in ety_rows]
    # Load all elements in one query via IN clause. SQLite has a
    # parameter limit (default ~999 in older builds, 32766 in
    # 3.32+); for safety, chunk to 500.
    elements_by_ety: dict[int, list[tuple[str, str]]] = {eid: [] for eid in ety_ids}
    for chunk_start in range(0, len(ety_ids), 500):
        chunk = ety_ids[chunk_start : chunk_start + 500]
        placeholders = ",".join("?" * len(chunk))
        for r in db.conn.execute(
            f"SELECT tee.toponym_etymology_id AS ety_id, e.canonical_form, e.language "
            f"FROM toponym_etymology_element tee "
            f"JOIN etymon e ON e.id = tee.etymon_id "
            f"WHERE tee.toponym_etymology_id IN ({placeholders}) "
            f"ORDER BY tee.toponym_etymology_id, tee.ordinal",
            chunk,
        ):
            elements_by_ety[r["ety_id"]].append(
                (_normalize_form(r["canonical_form"]), (r["language"] or "").lower())
            )

    rows_by_toponym: dict[int, list[_EtymologyRow]] = {}
    for ety in ety_rows:
        row = _EtymologyRow(
            id=ety["id"],
            toponym_id=ety["toponym_id"],
            notes=ety["notes"],
            extractor=_extractor_from_notes(ety["notes"]),
            elements=tuple(elements_by_ety.get(ety["id"], [])),
        )
        rows_by_toponym.setdefault(ety["toponym_id"], []).append(row)
    return rows_by_toponym


@dataclass
class CanonicalizeSummary:
    """Aggregate counters for a canonicalize run. Surfaces to the CLI
    so operators see how many toponyms were promoted, how many
    remained ``is_canonical=0`` (single-witness or all-disagreement),
    and how many rows were stamped with cluster_key + consensus_size.

    Invariant: ``toponyms_examined == toponyms_promoted +
    toponyms_no_consensus + toponyms_no_elements`` (the three result
    buckets are disjoint)."""

    toponyms_examined: int = 0
    # Promoted a cluster to canonical (>=2 distinct extractors agreed).
    toponyms_promoted: int = 0
    # Had clusters but none reached 2 witnesses. Single-witness rows
    # get cluster_key + consensus_size stamped on each pass even when
    # no canonical promotion fires.
    toponyms_no_consensus: int = 0
    # All rows had empty element tuples. Canonical demotion still
    # applied (defensive against stale state) but no promotion.
    toponyms_no_elements: int = 0
    # Number of rows whose canonical fields actually changed value
    # this pass — useful as an idempotency signal (0 on a no-op
    # re-run, even when toponyms_examined is large).
    rows_updated: int = 0


def apply_canonical_decisions(
    db: LexiconDB,
    plans: list[_ToponymPlan] | list[CanonicalDecision],
    *,
    on_decision: Callable[[CanonicalDecision], None] | None = None,
) -> CanonicalizeSummary:
    """Write the canonical-promotion plan to the DB in one transaction.

    For each plan:
      * Promoted row gets is_canonical=1, consensus_size=N, cluster_key=K
      * Other rows in the SAME cluster get is_canonical=0,
        consensus_size=N, cluster_key=K
      * Rows in OTHER clusters get is_canonical=0,
        consensus_size=<their cluster's witness count>,
        cluster_key=<their cluster's key>
      * Toponyms with no winning cluster get all rows demoted to
        is_canonical=0 + consensus_size=0 + cluster_key=None
        (a re-canonicalize pass over previously-canonical data must
        demote stale canonicals when consensus is lost).

    Accepts either ``_ToponymPlan`` (the internal builder's output,
    no DB re-query needed) or ``CanonicalDecision`` (back-compat
    where the caller only has the audit shape — the decision's
    promoted_etymology_id is IGNORED and re-derived from the CURRENT
    DB state, so a stale decision can't write the wrong canonical).

    ``on_decision`` fires per plan during the param-collection loop —
    BEFORE the executemany UPDATE writes the rows, and obviously
    before ``db.commit``. The callback observes the decision the
    plan WILL write, not the post-write DB state. Audit-log writers
    that need post-commit timing should not use this hook.
    """
    summary = CanonicalizeSummary()

    if plans and not isinstance(plans[0], _ToponymPlan):
        # Back-compat path: caller passed CanonicalDecisions. Recompute
        # plans from current DB state — the input decisions are used
        # ONLY as a toponym_id filter, NOT as a source of cluster
        # state. R2 silent-failure-hunter HIGH: a stale decision (row
        # deleted, rows added, etc.) would have produced wrong DB
        # writes via the previous implementation.
        plans = _plans_from_decisions(db, plans)  # type: ignore[arg-type]

    # Collect updates for executemany (R1 Gemini MEDIUM: per-row UPDATE
    # in a loop is slow on corpus-scale runs). The WHERE clause filters
    # out no-op writes so rows_updated counts ACTUAL changes.
    update_params: list[tuple] = []
    for plan in plans:
        plan_typed: _ToponymPlan = plan  # type: ignore[assignment]
        summary.toponyms_examined += 1
        if plan_typed.all_empty_elements:
            summary.toponyms_no_elements += 1
        elif plan_typed.decision.promoted_etymology_id is not None:
            summary.toponyms_promoted += 1
        else:
            summary.toponyms_no_consensus += 1
        for update in plan_typed.row_updates:
            update_params.append(
                (
                    update.is_canonical,
                    update.consensus_size,
                    update.cluster_key,
                    update.row_id,
                    update.is_canonical,
                    update.consensus_size,
                    update.cluster_key,
                )
            )
        if on_decision is not None:
            on_decision(plan_typed.decision)

    if update_params:
        # Use total_changes (a CONNECTION-level cumulative counter)
        # for the rows_updated diff. R2 code-reviewer HIGH: the
        # previous code called ``changes()`` (last-statement count)
        # around a SELECT, which always returned the SELECT's own
        # change count (0) and made the fallback unconditionally
        # report 0 updates. ``total_changes`` covers the cross-
        # statement count executemany accumulates.
        before = db.conn.total_changes
        db.conn.executemany(
            "UPDATE toponym_etymology "
            "SET is_canonical = ?, consensus_size = ?, cluster_key = ? "
            "WHERE id = ? AND ("
            "  is_canonical IS NOT ? OR consensus_size IS NOT ? OR cluster_key IS NOT ?"
            ")",
            update_params,
        )
        summary.rows_updated = db.conn.total_changes - before

    db.commit()
    return summary


def _plans_from_decisions(db: LexiconDB, decisions: list[CanonicalDecision]) -> list[_ToponymPlan]:
    """Back-compat shim: rebuild plans by re-running compute against
    current DB state, scoped to the toponym_ids in the input decisions.

    The input decisions' ``promoted_etymology_id`` + ``cluster_key``
    + ``consensus_size`` are IGNORED — only ``toponym_id`` is used to
    scope which toponyms to re-canonicalize. This makes the path
    correct under any DB drift between the original compute and the
    apply call (rows added, rows deleted, etymons changed). R2
    silent-failure-hunter HIGH flagged the previous behavior:
    inheriting the stale ``promoted_etymology_id`` could write the
    wrong canonical when the original winner row had been deleted,
    or write nothing at all (silent demotion).

    Most callers should use ``compute_canonical_plans`` directly —
    this shim exists so tests and audit-log replay can still pass
    ``CanonicalDecision`` objects through ``apply_canonical_decisions``
    without juggling the lower-level type.

    Two behavior notes after the R2 rewrite, neither affecting
    correctness in the CLI path (which uses plans directly):

    1. Toponyms whose rows have been DELETED between the original
       compute and this apply are silently omitted from the returned
       plan list — ``summary.toponyms_examined`` will reflect surviving
       toponyms, not the input decision count. R3 silent-failure LOW.
    2. The recompute does NOT propagate any original ``source_id``
       scope (CanonicalDecision has no source_id field). For a
       previously source-scoped run, the back-compat shim does a
       full-corpus re-cluster + filter — correct but O(corpus)
       work instead of O(decisions). R3 code-reviewer MEDIUM.
    """
    toponym_ids = {d.toponym_id for d in decisions}
    if not toponym_ids:
        return []
    # Re-run the canonical compute on current state, then keep only
    # the plans for toponyms the caller cared about.
    return [p for p in compute_canonical_plans(db) if p.decision.toponym_id in toponym_ids]
