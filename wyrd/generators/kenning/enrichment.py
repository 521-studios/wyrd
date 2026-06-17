"""L3 enrichment orchestrator — wyrd-ilam → wyrd-hidb.

After ``lexicon rebuild-from-jsonl`` produces a fresh SQLite from the
L2 JSONL source-of-truth, the derived columns on ``etymon``
(``lemma_id``, ``inflection``, ``lemma_method``, ``merged_into_id``,
``cognate_id``, ``stratum``, ``english_shaped``) plus the derived
tables (``toponym_decomposition``, ``etymon_period_form``) are empty
— the dump explicitly excludes them per the L2/L3 boundary contract.
Operators re-derive them by running the enrichment passes documented
in ``L2_L3_BOUNDARY.md``.

:func:`run_full_enrichment` is the canonical orchestrator. It walks
the L3 chain in dependency order:

    cluster-ocr → link-lemmas → apply-curation → decompose →
    cluster-cognates → classify-stratum → derive-english-shaped →
    tag-phonological-vectors → project-period-forms

Each pass keeps its existing standalone CLI command for operators
who want fine-grained control; the wrappers
(:func:`decomposition.decompose_all`,
:func:`english_shaping.derive_english_shaped_all`,
:func:`strata.classify_stratum_all`) are the uniform
``(db, *, apply)`` shape this module's orchestrator can call directly.

``skip_l3_derivations=True`` stops after the OCR+lemma+curation
prefix — useful for fast curation iteration + for the focused
synthetic-DB enrichment / curation tests that don't seed the
toponym / descent rows the downstream passes scan.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .lexicon import (
    LexiconDB,
    cluster_cognates,
    cluster_ocr_variants,
    link_lemmas,
    project_period_forms,
)
from .lexicon.canonicalization_projection import project_canonical
from .lexicon.english_shaping import derive_english_shaped_all
from .lexicon.phonological_vector_enrichment import (
    tag_phonological_vectors_all,
)
from .lexicon.pronunciation_backfill import derive_pronunciation_ipa
from .lexicon.relational_projection import project_descent_assertions
from .lexicon.strata import classify_stratum_all
from .runtime.decomposition import decompose_all

# Method-version provenance currently lives in code (the writer-side
# UPDATE statements stamp these strings). Surfacing them here in one
# place makes :func:`enrichment_status` reports + L2_L3_BOUNDARY.md
# stay in sync.
OCR_METHOD_VERSION = "cluster-ocr-v1"
LEMMA_METHOD_VERSION = "link-lemmas-v1"

# Stamped on etymon rows whose lemma_id was set by a curation override
# (wyrd-2jhs) rather than the auto-clustering link_lemmas pass. Keeps
# the L3 provenance distinguishable: a future audit can find every
# operator-curated row by querying ``lemma_method='manual-curation-v1'``.
CURATION_METHOD_VERSION = "manual-curation-v1"

# wyrd-kutx: operator-driven gloss suppression + etymon-split events
# carry their own provenance stamps so audit queries can find them.
GLOSS_SUPPRESSION_METHOD_VERSION = "gloss-suppression-v1"
GLOSS_ADD_METHOD_VERSION = "gloss-add-v1"
TAG_ADD_METHOD_VERSION = "llm-tags-v1"
ETYMON_SPLIT_METHOD_VERSION = "etymon-split-v1"
COLLAPSE_METHOD_VERSION = "collapse-v1"
# Allowed etymon_variant.variant_class values (mirrors the table's CHECK
# constraint). A collapse row with anything else is coerced to "other"
# so a hand-edited / LLM-emitted payload can't raise a CHECK violation.
_VARIANT_CLASSES = frozenset({"alternative", "inflection", "romanization", "canonical", "other"})
# Synthetic source for collapse-registered variant rows (the assertion
# "this form is a variant of that lemma" comes from the collapse decision
# itself, not a corpus). Derived + idempotently re-created on rebuild.
COLLAPSE_VARIANT_SOURCE_ID = "collapse"

# wyrd-van9: single source of truth for the split-suffix charset
# constraint. ``curate-split-etymon --into 'suffix=X'`` (CLI) and
# ``apply_etymon_splits`` (applier) both consult this pattern; the
# CLI imports it from this module so a charset change can't drift
# between layers. Locked to ``[a-z0-9_-]+`` so the resulting ref
# ``<lang>:<form>#<suffix>`` can't carry ``:`` / ``#`` / whitespace
# that would corrupt downstream ref parsers. Children failing the
# check are counted under ``children_skipped_invalid_suffix``.
SUFFIX_PATTERN = re.compile(r"[a-z0-9_-]+")


def _resolve_etymon_id(conn: sqlite3.Connection, ref: str) -> int | None:
    """Resolve ``"<language>:<canonical_form>"`` → etymon.id. Returns
    None when the ref doesn't match any row."""
    if ":" not in ref:
        return None
    lang, form = ref.split(":", 1)
    row = conn.execute(
        "SELECT id FROM etymon WHERE language = ? AND canonical_form = ?",
        (lang, form),
    ).fetchone()
    return row["id"] if row else None


def _build_ref_resolver(conn: sqlite3.Connection):
    """Memoized ref → etymon_id resolver. Multiple curation events
    often target the same lemma (e.g. dozens of inflected forms
    pointing at a single canonical lemma), so caching saves the
    redundant SELECT round-trips."""
    cache: dict[str, int | None] = {}

    def resolve(ref: str) -> int | None:
        if ref not in cache:
            cache[ref] = _resolve_etymon_id(conn, ref)
        return cache[ref]

    return resolve


@dataclass
class _CurationContext:
    """Shared state threaded through the per-ref curation appliers: the
    DB handle, the memoized ref resolver, the running telemetry counts
    dict (mutated in place), and the apply/dry-run flag."""

    db: LexiconDB
    resolve: Callable[[str], int | None]
    counts: dict[str, Any]
    apply: bool


def _apply_curated_lemma(ctx: _CurationContext, payload: dict[str, Any], etymon_id: int) -> None:
    """Apply a curation event's ``lemma_ref`` decision to one etymon.

    ``None`` clears lemma_id + inflection (stamping the method version so
    the operator decision stays auditable via ``WHERE lemma_method =
    'manual-curation-v1'``). A resolvable ref points lemma_id at it and
    clears merged_into_id (the etymon stays canonical); ``inflection`` is
    written only when the key is present, so an auto-detected inflection
    from link-lemmas survives when the operator gave no opinion.
    Unresolved and self-reference refs are counted and skipped. DB writes
    are gated on ``ctx.apply``.
    """
    lemma_ref = payload["lemma_ref"]
    if lemma_ref is None:
        ctx.counts["lemma_id_cleared"] += 1
        if ctx.apply:
            ctx.db.conn.execute(
                "UPDATE etymon SET lemma_id = NULL, inflection = NULL, "
                "lemma_method = ? WHERE id = ?",
                (CURATION_METHOD_VERSION, etymon_id),
            )
        return
    lemma_id = ctx.resolve(lemma_ref)
    if lemma_id is None:
        ctx.counts["unresolved_lemma_ref"] += 1
    elif lemma_id == etymon_id:
        ctx.counts["self_reference_lemma"] += 1
    else:
        ctx.counts["lemma_id_set"] += 1
        if ctx.apply:
            # Dynamic SET list: only touch ``inflection`` when the
            # operator explicitly provided it (absent key = no opinion).
            set_clauses = [
                "lemma_id = ?",
                "lemma_method = ?",
                "merged_into_id = NULL",
            ]
            values: list[Any] = [lemma_id, CURATION_METHOD_VERSION]
            if "inflection" in payload:
                set_clauses.append("inflection = ?")
                values.append(payload["inflection"])
            values.append(etymon_id)
            ctx.db.conn.execute(
                f"UPDATE etymon SET {', '.join(set_clauses)} WHERE id = ?",
                tuple(values),
            )


def _apply_curated_merge(ctx: _CurationContext, payload: dict[str, Any], etymon_id: int) -> None:
    """Apply a curation event's ``merged_into_ref`` decision to one etymon.

    ``None`` clears merged_into_id (method-stamped for audit). A
    resolvable ref points merged_into_id at it as an OCR-cluster
    tombstone and clears lemma_id + inflection so the tombstone doesn't
    double as a lemma parent. Unresolved and self-reference refs are
    counted and skipped. DB writes are gated on ``ctx.apply``.
    """
    merge_ref = payload["merged_into_ref"]
    if merge_ref is None:
        ctx.counts["merged_into_cleared"] += 1
        if ctx.apply:
            ctx.db.conn.execute(
                "UPDATE etymon SET merged_into_id = NULL, lemma_method = ? WHERE id = ?",
                (CURATION_METHOD_VERSION, etymon_id),
            )
        return
    merge_id = ctx.resolve(merge_ref)
    if merge_id is None:
        ctx.counts["unresolved_merge_ref"] += 1
    elif merge_id == etymon_id:
        ctx.counts["self_reference_merge"] += 1
    else:
        ctx.counts["merged_into_set"] += 1
        if ctx.apply:
            ctx.db.conn.execute(
                "UPDATE etymon SET merged_into_id = ?, "
                "lemma_id = NULL, inflection = NULL, "
                "lemma_method = ? WHERE id = ?",
                (merge_id, CURATION_METHOD_VERSION, etymon_id),
            )


def apply_curation_overrides(
    db: LexiconDB,
    curation_state: dict[str, dict[str, Any]],
    *,
    apply: bool = True,
) -> dict[str, Any]:
    """Apply operator curation overrides to L3 enrichment columns (wyrd-2jhs).

    ``curation_state`` is the merged ``{etymon_ref: payload}`` dict
    from :func:`jsonl.build.collect_curation_overrides`. For each
    curated etymon:

    - ``lemma_ref`` set → ``lemma_id`` pointed at that etymon, plus
      ``lemma_method='manual-curation-v1'``. Also clears
      ``merged_into_id`` (the etymon stays canonical; its inflection
      target is the curated lemma, not a tombstone).
    - ``inflection`` set → written alongside ``lemma_id``. Required
      when ``lemma_ref`` is set; otherwise the inflection column would
      lie about what's there.
    - ``merged_into_ref`` set → ``merged_into_id`` pointed at that
      etymon (OCR-cluster tombstone). Clears ``lemma_id`` so the
      tombstone doesn't claim a lemma role.
    - Explicit ``null`` on any of those refs clears the corresponding
      column.

    Run AFTER the auto-clustering passes (normalize-ocr + link-lemmas)
    so curation overrides their output. Bad curation events
    (unresolved refs, self-references) are logged in the counts dict
    and skipped — never raise, so a stale ref doesn't block the
    rebuild.

    Validation always runs (resolves refs, counts unresolved /
    self-references); DB writes only happen when ``apply=True``.
    Dry-run mode is useful for operators previewing a curation batch
    before committing.

    Self-reference guard: an etymon can't be its own lemma_id or
    merged_into_id target. Such curation events would create cycles
    (e.g. ``find_canonical(x)`` resolving infinitely). Counted under
    ``self_reference_lemma`` / ``self_reference_merge`` and skipped.

    Returns a counts dict for telemetry.
    """
    counts = {
        "curations": len(curation_state),
        "lemma_id_set": 0,
        "lemma_id_cleared": 0,
        "merged_into_set": 0,
        "merged_into_cleared": 0,
        "unresolved_etymon": 0,
        "unresolved_lemma_ref": 0,
        "unresolved_merge_ref": 0,
        "self_reference_lemma": 0,
        "self_reference_merge": 0,
        "applied": apply,
    }
    resolve = _build_ref_resolver(db.conn)
    ctx = _CurationContext(db=db, resolve=resolve, counts=counts, apply=apply)

    for etymon_ref, payload in curation_state.items():
        etymon_id = resolve(etymon_ref)
        if etymon_id is None:
            counts["unresolved_etymon"] += 1
            continue

        # A present ref key (even with None value) means "operator made a
        # decision"; an absent key means "no opinion, leave auto-
        # clustering's output alone". The two ref fields process
        # independently within a payload — a failed lemma_ref resolution
        # shouldn't skip merged_into_ref in the same event.
        if "lemma_ref" in payload:
            _apply_curated_lemma(ctx, payload, etymon_id)
        if "merged_into_ref" in payload:
            _apply_curated_merge(ctx, payload, etymon_id)

    if apply:
        db.commit()
    return counts


def format_curation_run(counts: dict[str, Any]) -> str:
    """Render :func:`apply_curation_overrides` output as markdown."""
    applied = counts.get("applied", True)
    verb_set = "set" if applied else "would set"
    verb_clear = "cleared" if applied else "would clear"
    lines: list[str] = [
        "## L3 enrichment: curation overrides",
        "",
        f"- Method version: `{CURATION_METHOD_VERSION}`",
        f"- Curations processed: {counts['curations']}",
        f"- Lemma overrides {verb_set}: {counts['lemma_id_set']}",
        f"- Lemma overrides {verb_clear}: {counts['lemma_id_cleared']}",
        f"- Merge overrides {verb_set}: {counts['merged_into_set']}",
        f"- Merge overrides {verb_clear}: {counts['merged_into_cleared']}",
    ]
    unresolved = (
        counts["unresolved_etymon"]
        + counts["unresolved_lemma_ref"]
        + counts["unresolved_merge_ref"]
    )
    if unresolved:
        lines.append("")
        lines.append("### Unresolved refs (skipped)")
        if counts["unresolved_etymon"]:
            lines.append(f"- Curated etymon not found: {counts['unresolved_etymon']}")
        if counts["unresolved_lemma_ref"]:
            lines.append(f"- Lemma target not found: {counts['unresolved_lemma_ref']}")
        if counts["unresolved_merge_ref"]:
            lines.append(f"- Merge target not found: {counts['unresolved_merge_ref']}")
    self_refs = counts.get("self_reference_lemma", 0) + counts.get("self_reference_merge", 0)
    if self_refs:
        lines.append("")
        lines.append("### Self-references (skipped)")
        if counts.get("self_reference_lemma"):
            lines.append(f"- Etymon would be its own lemma: {counts['self_reference_lemma']}")
        if counts.get("self_reference_merge"):
            lines.append(f"- Etymon would tombstone to itself: {counts['self_reference_merge']}")
    return "\n".join(lines)


def apply_gloss_suppressions(
    db: LexiconDB,
    suppression_state: dict[str, dict[str, Any]],
    *,
    apply: bool = True,
) -> dict[str, Any]:
    """Drop operator-listed glosses from etymons (wyrd-kutx).

    ``suppression_state`` is the merged ``{etymon_ref: payload}`` dict
    from :func:`jsonl.build.collect_gloss_suppressions`. Each payload
    carries a ``suppressions`` array of ``{gloss, reason?}`` entries;
    each entry drops the matching ``etymon_gloss`` row.

    Use case: Wiktionary-mined etymons whose entry conflates two senses
    where only one is used in place-names. OE ``dry`` carries "dry"
    plus "wizard, sorcerer"; the wizard sense has no toponymy use, so
    the operator drops that gloss to keep the place-name sense clean.

    Mining evidence is preserved end-to-end — citations, descent edges,
    and toponym_etymology_element refs all stay attached to the etymon
    (D21). Only the gloss layer is touched.

    Etymon tags are LEFT ALONE — tags don't track per-gloss provenance
    in the current schema, so we can't tell which gloss contributed a
    tag. Operators who need to drop a tag should follow up with a
    separate event type (deferred).

    Validation always runs; DB writes happen only when ``apply=True``.
    Returns a counts dict for telemetry.
    """
    counts: dict[str, Any] = {
        "etymons_touched": len(suppression_state),
        "glosses_dropped": 0,
        "unresolved_etymon": 0,
        "unresolved_gloss": 0,
        "applied": apply,
        "method_version": GLOSS_SUPPRESSION_METHOD_VERSION,
    }
    resolve = _build_ref_resolver(db.conn)

    for etymon_ref, payload in suppression_state.items():
        etymon_id = resolve(etymon_ref)
        if etymon_id is None:
            counts["unresolved_etymon"] += 1
            continue
        suppressions = payload.get("suppressions") or []
        for entry in suppressions:
            gloss = entry.get("gloss") if isinstance(entry, dict) else None
            if not gloss:
                continue
            row = db.conn.execute(
                "SELECT 1 FROM etymon_gloss WHERE etymon_id = ? AND gloss = ?",
                (etymon_id, gloss),
            ).fetchone()
            if row is None:
                counts["unresolved_gloss"] += 1
                continue
            counts["glosses_dropped"] += 1
            if apply:
                db.conn.execute(
                    "DELETE FROM etymon_gloss WHERE etymon_id = ? AND gloss = ?",
                    (etymon_id, gloss),
                )

    if apply:
        db.commit()
    return counts


def apply_gloss_additions(
    db: LexiconDB,
    addition_state: dict[str, dict[str, Any]],
    *,
    apply: bool = True,
) -> dict[str, Any]:
    """Add operator-curated glosses to etymons (wyrd-wz82).

    Inverse of :func:`apply_gloss_suppressions`. ``addition_state`` is
    the merged ``{etymon_ref: payload}`` dict from
    :func:`jsonl.build.collect_gloss_additions`. Each payload carries
    an ``additions`` array of ``{gloss, reason?}`` entries; each entry
    INSERTs a row into ``etymon_gloss`` (with ``INSERT OR IGNORE`` so
    re-runs are idempotent — the (etymon_id, gloss) PK absorbs the
    duplicate).

    Use case: surfacing a sense the auto-mining missed. OE ``finn``
    carries "clear, transparent, bright" + "fin" today; Roberts 1914
    Sussex cites it as a personal name, a third sense the bundle has
    no gloss for. Operator adds "personal name" via this event so the
    sense is queryable without a full lexicon re-mine.

    Counts that the gloss was a duplicate (already on the etymon)
    under ``additions_already_present`` rather than as an error — re-
    running an apply is a normal idempotency case.

    Validation always runs; DB writes happen only when ``apply=True``.
    Returns a counts dict for telemetry.
    """
    counts: dict[str, Any] = {
        "etymons_touched": len(addition_state),
        "glosses_added": 0,
        "additions_already_present": 0,
        "unresolved_etymon": 0,
        "applied": apply,
        "method_version": GLOSS_ADD_METHOD_VERSION,
    }
    resolve = _build_ref_resolver(db.conn)

    for etymon_ref, payload in addition_state.items():
        etymon_id = resolve(etymon_ref)
        if etymon_id is None:
            counts["unresolved_etymon"] += 1
            continue
        additions = payload.get("additions") or []
        for entry in additions:
            gloss = entry.get("gloss") if isinstance(entry, dict) else None
            if not gloss:
                continue
            existing = db.conn.execute(
                "SELECT 1 FROM etymon_gloss WHERE etymon_id = ? AND gloss = ?",
                (etymon_id, gloss),
            ).fetchone()
            if existing is not None:
                counts["additions_already_present"] += 1
                continue
            counts["glosses_added"] += 1
            if apply:
                db.conn.execute(
                    "INSERT OR IGNORE INTO etymon_gloss (etymon_id, gloss) VALUES (?, ?)",
                    (etymon_id, gloss),
                )

    if apply:
        db.commit()
    return counts


def apply_tag_additions(
    db: LexiconDB,
    tag_state: dict[str, dict[str, Any]],
    *,
    apply: bool = True,
) -> dict[str, Any]:
    """Add LLM-classified semantic tags to untagged etymons (wyrd-xz3g).

    ``tag_state`` is the ``{etymon_ref: {"tags": [...]}}`` dict from
    :func:`lexicon.tag_mining.collect_tags` — the decisions persisted to L2
    ``data/mining/_tags.jsonl`` by ``mine-tags-llm`` (gemma4:26b classifying a
    gloss into the controlled tag vocabulary). Each tag INSERTs a row into
    ``etymon_tag`` (``INSERT OR IGNORE`` via :meth:`LexiconDB.add_tag` — the
    (etymon_id, tag) PK absorbs duplicates so re-runs are idempotent). The
    'none' outcome (empty tag list) is filtered upstream in ``collect_tags``,
    so every ref here adds ≥1 tag.

    Mirrors :func:`apply_gloss_additions`: paid LLM work persisted to L2 and
    replayed FREE by :func:`run_full_enrichment` on a from-scratch rebuild
    (D21 / REBUILD.md), so the LLM pass never re-runs. ``etymon_tag`` is an L2
    column, so these additions land in the same layer the dump/rebuild
    round-trips. Validation always runs; DB writes only when ``apply=True``.
    """
    counts: dict[str, Any] = {
        "etymons_touched": len(tag_state),
        "tags_added": 0,
        "tags_already_present": 0,
        "unresolved_etymon": 0,
        "applied": apply,
        "method_version": TAG_ADD_METHOD_VERSION,
    }
    resolve = _build_ref_resolver(db.conn)

    for etymon_ref, payload in tag_state.items():
        etymon_id = resolve(etymon_ref)
        if etymon_id is None:
            counts["unresolved_etymon"] += 1
            continue
        # One SELECT per etymon (not per tag): fetch the existing tag set once
        # and check membership in memory — O(N) queries, not O(N*tags). The
        # SELECT is only for the added/already-present telemetry split;
        # db.add_tag is INSERT OR IGNORE, so the write is idempotent regardless.
        present = {
            row[0]
            for row in db.conn.execute(
                "SELECT tag FROM etymon_tag WHERE etymon_id = ?", (etymon_id,)
            )
        }
        for tag in payload.get("tags") or []:
            if tag in present:
                counts["tags_already_present"] += 1
                continue
            counts["tags_added"] += 1
            present.add(tag)  # a tag repeated within one payload counts once
            if apply:
                db.add_tag(etymon_id, tag)

    if apply:
        db.commit()
    return counts


def format_gloss_addition_run(counts: dict[str, Any]) -> str:
    """Render :func:`apply_gloss_additions` output as markdown."""
    applied = counts.get("applied", True)
    verb = "added" if applied else "would add"
    lines = [
        "## L3 enrichment: gloss additions",
        "",
        f"- Method version: `{counts.get('method_version', GLOSS_ADD_METHOD_VERSION)}`",
        f"- Etymons touched: {counts['etymons_touched']}",
        f"- Glosses {verb}: {counts['glosses_added']}",
    ]
    if counts.get("additions_already_present"):
        lines.append(
            f"- Additions already on etymon (idempotent): {counts['additions_already_present']}"
        )
    if counts.get("unresolved_etymon"):
        lines.append(f"- Unresolved etymon refs: {counts['unresolved_etymon']}")
    return "\n".join(lines)


def format_gloss_suppression_run(counts: dict[str, Any]) -> str:
    """Render :func:`apply_gloss_suppressions` output as markdown."""
    applied = counts.get("applied", True)
    verb = "dropped" if applied else "would drop"
    lines = [
        "## L3 enrichment: gloss suppressions",
        "",
        f"- Method version: `{counts.get('method_version', GLOSS_SUPPRESSION_METHOD_VERSION)}`",
        f"- Etymons touched: {counts['etymons_touched']}",
        f"- Glosses {verb}: {counts['glosses_dropped']}",
    ]
    if counts.get("unresolved_etymon"):
        lines.append(f"- Unresolved etymon refs: {counts['unresolved_etymon']}")
    if counts.get("unresolved_gloss"):
        lines.append(f"- Glosses not on the named etymon: {counts['unresolved_gloss']}")
    return "\n".join(lines)


@dataclass
class _SplitContext:
    """Shared state threaded through the etymon-split phase helpers: the
    DB handle, the running telemetry counts dict (mutated in place), and
    the apply/dry-run flag."""

    db: LexiconDB
    counts: dict[str, Any]
    apply: bool


@dataclass
class _SplitParent:
    """The resolved parent etymon being split — its row id, canonical
    form, language, and the ``<lang>:<form>`` ref that named it."""

    id: int
    form: str
    lang: str
    ref: str


@dataclass(frozen=True)
class _AttrMoveSpec:
    """Identifies a parent→child named-attribute repoint (gloss or tag):
    the table, its value column, and the moved/missing counter keys. The
    gloss and tag moves are structurally identical bar these four."""

    table: str
    col: str
    moved_key: str
    missing_key: str


_GLOSS_MOVE = _AttrMoveSpec("etymon_gloss", "gloss", "glosses_moved", "glosses_missing")
_TAG_MOVE = _AttrMoveSpec("etymon_tag", "tag", "tags_moved", "tags_missing")


def _pick_primary_index(into: list[Any], counts: dict[str, Any]) -> int:
    """Choose which child in ``into`` inherits the parent's mining
    evidence. The first child flagged ``primary`` wins; absent any flag
    the first child defaults (counted as ``no_primary_defaulted``). The
    CLI rejects multi-primary specs, but a hand-edited JSONL could slip
    through — count the collapse (``multiple_primary_collapsed``) so it's
    visible in the summary."""
    primary_flags = sum(1 for child in into if isinstance(child, dict) and child.get("primary"))
    if primary_flags > 1:
        counts["multiple_primary_collapsed"] += 1
    for idx, child in enumerate(into):
        if isinstance(child, dict) and child.get("primary"):
            return idx
    counts["no_primary_defaulted"] += 1
    return 0


def _repoint_child_attr(
    ctx: _SplitContext,
    orig_id: int,
    child_id: int | None,
    names: list[str] | None,
    spec: _AttrMoveSpec,
) -> None:
    """Repoint named gloss/tag rows from the parent to the child via
    DELETE-then-INSERT-OR-IGNORE (the PRIMARY KEY ``(etymon_id, value)``
    means a plain UPDATE could hit a uniqueness conflict when the child
    already carries the value from a prior run). Names not present on the
    parent are counted under ``spec.missing_key`` and skipped. Counting
    always runs; writes only when ``apply`` and a real child id exists."""
    for name in names or []:
        row = ctx.db.conn.execute(
            f"SELECT 1 FROM {spec.table} WHERE etymon_id = ? AND {spec.col} = ?",
            (orig_id, name),
        ).fetchone()
        if row is None:
            ctx.counts[spec.missing_key] += 1
            continue
        ctx.counts[spec.moved_key] += 1
        if ctx.apply and child_id is not None:
            ctx.db.conn.execute(
                f"DELETE FROM {spec.table} WHERE etymon_id = ? AND {spec.col} = ?",
                (orig_id, name),
            )
            ctx.db.conn.execute(
                f"INSERT OR IGNORE INTO {spec.table} (etymon_id, {spec.col}) VALUES (?, ?)",
                (child_id, name),
            )


def _apply_split_child(ctx: _SplitContext, parent: _SplitParent, child: Any) -> int | None:
    """Create (or reuse) one split child etymon and repoint its named
    glosses + tags off the parent. Returns the child's etymon id, or
    ``None`` when the child spec is rejected (missing/invalid suffix) or
    the row doesn't exist yet under dry-run.

    ``canonical_form`` is ``<orig_form>#<suffix>``; the suffix charset
    gate mirrors the CLI's ``_SUFFIX_PATTERN`` to defend against
    hand-edited JSONL events whose suffix would corrupt the child ref.
    """
    suffix = child.get("suffix") if isinstance(child, dict) else None
    if not suffix:
        # CLI rejects this, but a hand-edited JSONL could land here.
        ctx.counts["children_skipped_no_suffix"] += 1
        return None
    if not SUFFIX_PATTERN.fullmatch(suffix):
        ctx.counts["children_skipped_invalid_suffix"] += 1
        return None
    child_form = f"{parent.form}#{suffix}"

    existing = ctx.db.conn.execute(
        "SELECT id FROM etymon WHERE language = ? AND canonical_form = ?",
        (parent.lang, child_form),
    ).fetchone()
    if existing is not None:
        ctx.counts["children_already_existed"] += 1
        child_id: int | None = existing["id"]
    else:
        ctx.counts["children_created"] += 1
        if ctx.apply:
            cur = ctx.db.conn.execute(
                "INSERT INTO etymon (canonical_form, language, notes) VALUES (?, ?, ?)",
                (child_form, parent.lang, f"[wyrd-kutx-split-child of {parent.ref}]"),
            )
            child_id = cur.lastrowid
        else:
            child_id = None  # dry-run: no row exists yet

    _repoint_child_attr(ctx, parent.id, child_id, child.get("glosses"), _GLOSS_MOVE)
    _repoint_child_attr(ctx, parent.id, child_id, child.get("tags"), _TAG_MOVE)
    return child_id


def _move_parent_evidence(ctx: _SplitContext, orig_id: int, primary_child_id: int | None) -> None:
    """Move the parent's mining evidence (citations, descent edges,
    etymology elements) to the primary child so it inherits the witness
    count for bundle promotion (D21: evidence is moved, not destroyed).

    Apply path uses ``UPDATE OR IGNORE`` to survive UNIQUE conflicts on a
    re-apply where the primary already carries matching rows; the
    conflict rows stay on the parent and surface under ``*_skipped_
    conflict`` (counted via ``rowcount`` so the summary reflects what
    actually moved). Dry-run (or no primary) moves nothing but counts the
    eligible rows so an operator preview shows the impact size."""
    cit_count = ctx.db.conn.execute(
        "SELECT COUNT(*) AS n FROM etymon_citation WHERE etymon_id = ?",
        (orig_id,),
    ).fetchone()["n"]
    descent_count = ctx.db.conn.execute(
        "SELECT COUNT(*) AS n FROM etymon_descent WHERE parent_id = ? OR child_id = ?",
        (orig_id, orig_id),
    ).fetchone()["n"]
    element_count = ctx.db.conn.execute(
        "SELECT COUNT(*) AS n FROM toponym_etymology_element WHERE etymon_id = ?",
        (orig_id,),
    ).fetchone()["n"]
    if ctx.apply and primary_child_id is not None:
        cur = ctx.db.conn.execute(
            "UPDATE OR IGNORE etymon_citation SET etymon_id = ? WHERE etymon_id = ?",
            (primary_child_id, orig_id),
        )
        cit_moved = cur.rowcount or 0
        cur = ctx.db.conn.execute(
            "UPDATE OR IGNORE etymon_descent SET parent_id = ? WHERE parent_id = ?",
            (primary_child_id, orig_id),
        )
        descent_parent_moved = cur.rowcount or 0
        cur = ctx.db.conn.execute(
            "UPDATE OR IGNORE etymon_descent SET child_id = ? WHERE child_id = ?",
            (primary_child_id, orig_id),
        )
        descent_child_moved = cur.rowcount or 0
        descent_moved = descent_parent_moved + descent_child_moved
        cur = ctx.db.conn.execute(
            "UPDATE OR IGNORE toponym_etymology_element SET etymon_id = ? WHERE etymon_id = ?",
            (primary_child_id, orig_id),
        )
        element_moved = cur.rowcount or 0
    else:
        cit_moved = cit_count
        descent_moved = descent_count
        element_moved = element_count

    ctx.counts["citations_moved"] += cit_moved
    ctx.counts["descent_edges_moved"] += descent_moved
    ctx.counts["etymology_elements_moved"] += element_moved
    ctx.counts["citations_skipped_conflict"] += max(0, cit_count - cit_moved)
    ctx.counts["descent_edges_skipped_conflict"] += max(0, descent_count - descent_moved)
    ctx.counts["etymology_elements_skipped_conflict"] += max(0, element_count - element_moved)


def _stamp_split_parent(
    ctx: _SplitContext, parent: _SplitParent, into: list[Any], primary_child_id: int | None
) -> None:
    """Mark the parent's ``notes`` as split — but only when at least one
    child produced a usable row (``primary_child_id`` set). Without this
    guard a JSONL hand-edit with all-invalid child specs would stamp the
    parent 'split' while leaving it full of glosses/citations: an
    audit-query trap."""
    if ctx.apply and primary_child_id is not None:
        ctx.db.conn.execute(
            "UPDATE etymon SET notes = ? WHERE id = ?",
            (f"[wyrd-kutx-split:{len(into)}] (was: {parent.form})", parent.id),
        )


def apply_etymon_splits(
    db: LexiconDB,
    split_state: dict[str, dict[str, Any]],
    *,
    apply: bool = True,
) -> dict[str, Any]:
    """Split conflated etymons into distinct child etymons (wyrd-kutx).

    ``split_state`` is the merged ``{etymon_ref: payload}`` dict from
    :func:`jsonl.build.collect_etymon_splits`. Each payload's ``into``
    array names the children::

        {
          "into": [
            {"suffix": "weir", "glosses": [...], "tags": [...], "primary": true},
            {"suffix": "year", "glosses": [...], "tags": [...]}
          ],
          "reason": "..."
        }

    For each child, a new etymon row is created with
    ``canonical_form='<orig>#<suffix>'`` and the same language. The
    named glosses + tags are REPOINTED from the original to the child
    (DELETE-then-INSERT).

    Mining evidence (etymon_citation, etymon_descent,
    toponym_etymology_element) on the parent is MOVED to the
    ``primary=true`` child wholesale. If no child is flagged primary,
    the first child in the ``into`` array is used as the default.
    This lets the primary child inherit the parent's witness count for
    bundle promotion without operator pre-curation; per-citation
    reassignment to non-primary children is a follow-on op (deferred).

    Per D21 mining evidence is preserved (moved, not destroyed). The
    parent's ``notes`` column gets a marker so the parent's emptied
    state is identifiable post-hoc.

    Empty ``into`` array is a no-op (operator-cleared event).

    Validation always runs; DB writes happen only when ``apply=True``.
    Returns a counts dict for telemetry.
    """
    counts: dict[str, Any] = {
        "splits_processed": 0,
        "children_created": 0,
        "children_already_existed": 0,
        "glosses_moved": 0,
        "tags_moved": 0,
        "glosses_missing": 0,
        "tags_missing": 0,
        "citations_moved": 0,
        "descent_edges_moved": 0,
        "etymology_elements_moved": 0,
        # Rows that OR-IGNORE skipped due to UNIQUE conflicts on the
        # primary child (re-apply / manual-move scenarios); stay on the
        # parent until a citation-reassign event handles them.
        "citations_skipped_conflict": 0,
        "descent_edges_skipped_conflict": 0,
        "etymology_elements_skipped_conflict": 0,
        # Defense-in-depth counters for JSONL events that bypass the
        # CLI guards (hand-edited files, future emitters). See
        # ``curate_gloss.py`` for the parallel CLI-side rejections.
        "children_skipped_no_suffix": 0,
        "children_skipped_invalid_suffix": 0,
        "multiple_primary_collapsed": 0,
        "unresolved_etymon": 0,
        "no_primary_defaulted": 0,
        "empty_into_skipped": 0,
        "applied": apply,
        "method_version": ETYMON_SPLIT_METHOD_VERSION,
    }

    ctx = _SplitContext(db=db, counts=counts, apply=apply)

    for etymon_ref, payload in split_state.items():
        into = payload.get("into") or []
        if not into:
            counts["empty_into_skipped"] += 1
            continue

        # Re-resolve per outer ref so freshly-inserted child etymons
        # are visible to subsequent collisions / lookups within the same
        # apply call. Don't cache across the loop.
        orig_row = db.conn.execute(
            "SELECT id, canonical_form, language FROM etymon "
            "WHERE (language || ':' || canonical_form) = ?",
            (etymon_ref,),
        ).fetchone()
        if orig_row is None:
            counts["unresolved_etymon"] += 1
            continue

        parent = _SplitParent(
            id=orig_row["id"],
            form=orig_row["canonical_form"],
            lang=orig_row["language"],
            ref=etymon_ref,
        )
        counts["splits_processed"] += 1

        # Pick the primary child up-front so the child loop can stash the
        # chosen primary's id for the post-loop evidence move.
        primary_index = _pick_primary_index(into, counts)
        primary_child_id: int | None = None
        for idx, child in enumerate(into):
            child_id = _apply_split_child(ctx, parent, child)
            if idx == primary_index:
                primary_child_id = child_id

        _move_parent_evidence(ctx, parent.id, primary_child_id)
        _stamp_split_parent(ctx, parent, into, primary_child_id)

    if apply:
        db.commit()
    return counts


# Optional summary lines for format_etymon_split_run, in render order.
# Each (key, label) emits "- {label}: {counts[key]}" only when the count
# is truthy — every line follows that uniform shape, so a data table +
# loop replaces a dozen near-identical guards.
_SPLIT_OPTIONAL_LINES: list[tuple[str, str]] = [
    ("no_primary_defaulted", "Splits with no `primary` flag (defaulted to first child)"),
    ("children_already_existed", "Children that already existed (reused)"),
    ("unresolved_etymon", "Unresolved parent refs"),
    ("glosses_missing", "Glosses not on parent"),
    ("tags_missing", "Tags not on parent"),
    ("empty_into_skipped", "Empty splits skipped"),
    ("children_skipped_invalid_suffix", "Children skipped (suffix outside [a-z0-9_-]+ charset)"),
    ("children_skipped_no_suffix", "Children skipped (missing/empty `suffix` in JSONL event)"),
    (
        "multiple_primary_collapsed",
        "Splits with >1 primary flag (collapsed to first; CLI normally rejects this)",
    ),
    ("citations_skipped_conflict", "Citations left on parent due to UNIQUE conflict on primary"),
    (
        "descent_edges_skipped_conflict",
        "Descent edges left on parent due to UNIQUE conflict on primary",
    ),
    (
        "etymology_elements_skipped_conflict",
        "Etymology elements left on parent due to UNIQUE conflict on primary",
    ),
]


def format_etymon_split_run(counts: dict[str, Any]) -> str:
    """Render :func:`apply_etymon_splits` output as markdown."""
    applied = counts.get("applied", True)
    verb_create = "created" if applied else "would create"
    verb_move = "moved" if applied else "would move"
    lines = [
        "## L3 enrichment: etymon splits",
        "",
        f"- Method version: `{counts.get('method_version', ETYMON_SPLIT_METHOD_VERSION)}`",
        f"- Splits processed: {counts['splits_processed']}",
        f"- Children {verb_create}: {counts['children_created']}",
        f"- Glosses {verb_move}: {counts['glosses_moved']}",
        f"- Tags {verb_move}: {counts['tags_moved']}",
        f"- Citations {verb_move} to primary: {counts.get('citations_moved', 0)}",
        f"- Descent edges {verb_move} to primary: {counts.get('descent_edges_moved', 0)}",
        f"- Etymology elements {verb_move} to primary: {counts.get('etymology_elements_moved', 0)}",
    ]
    # Optional lines: emit only when the count is truthy (the table fixes
    # render order). Each follows the uniform "- {label}: {value}" shape.
    for key, label in _SPLIT_OPTIONAL_LINES:
        if counts.get(key):
            lines.append(f"- {label}: {counts[key]}")
    return "\n".join(lines)


def _resolve_live_etymon(conn: sqlite3.Connection, ref: str | None):
    """Resolve ``"<language>:<canonical_form>"`` → an etymon row,
    EXCLUDING already-tombstoned rows (``merged_into_id IS NULL``). This
    is what makes collapse idempotent: a second apply no longer finds a
    ``from`` that an earlier row already folded. Returns ``None`` for an
    empty/absent or unmatched ref."""
    if not ref:
        return None
    return conn.execute(
        "SELECT id, canonical_form, language FROM etymon "
        "WHERE (language || ':' || canonical_form) = ? AND merged_into_id IS NULL",
        (ref,),
    ).fetchone()


@dataclass
class _CollapseContext:
    """Shared state threaded through the collapse row-type handlers: the
    DB handle, the running telemetry counts dict (mutated in place), and
    the apply/dry-run flag."""

    db: LexiconDB
    counts: dict[str, Any]
    apply: bool


def _apply_reflex_link(ctx: _CollapseContext, from_ref: str, payload: dict[str, Any]) -> None:
    """wyrd-rogd.15 reflex-LINK row (carries ``inherits``): assert ``from_ref``
    is a later-era reflex of the ``inherits`` ancestor — the same morpheme
    across eras — by adding an ``inheritance`` descent edge (parent = ancestor,
    child = ref). Both etymons stay DISTINCT (vs the ``into`` fold, which
    tombstones). ``inherits: ""`` (a recorded LLM rejection / revert) is a
    no-op counted as ``link_rejections`` — it must NOT fall through to the fold.
    Idempotent via OR IGNORE."""
    inherits_ref = payload.get("inherits")
    if not inherits_ref:
        ctx.counts["link_rejections"] += 1
        return
    child_row = _resolve_live_etymon(ctx.db.conn, from_ref)
    if child_row is None:
        ctx.counts["unresolved_from"] += 1
        return
    parent_row = _resolve_live_etymon(ctx.db.conn, inherits_ref)
    if parent_row is None:
        ctx.counts["unresolved_inherits"] += 1
        return
    if child_row["id"] == parent_row["id"]:
        ctx.counts["self_link_skipped"] += 1
        return
    ctx.counts["links_processed"] += 1
    if ctx.apply:
        ctx.db.conn.execute(
            "INSERT OR IGNORE INTO etymon_descent "
            "(parent_id, child_id, edge_type, source_id, confidence, notes) "
            "VALUES (?, ?, 'inheritance', ?, ?, ?)",
            (
                parent_row["id"],
                child_row["id"],
                COLLAPSE_VARIANT_SOURCE_ID,
                payload.get("confidence"),
                payload.get("notes") or payload.get("reason") or "wyrd-rogd.15 reflex-link",
            ),
        )


def _apply_edge_detach(ctx: _CollapseContext, from_ref: str, payload: dict[str, Any]) -> bool:
    """wyrd-qp9c descent-edge DETACH. ``detach_parents`` / ``detach_children``
    name wrong-sense edge endpoints to DELETE BEFORE the fold, so
    cluster_cognates — which resolves a tombstone's edges through
    ``merged_into_id`` — never redirects the wrong lineage onto ``into``.
    ``detach_parents`` removes ``<parent> -> from`` edges; ``detach_children``
    removes ``from -> <child>`` edges (any edge_type for the named pair: the
    operator is asserting the two etymons are unrelated). Returns whether the
    row carried ANY detach refs, so the caller can tell a legitimate
    detach-only row from a truly empty one.

    Detach refs resolve TOMBSTONE-AGNOSTICALLY (``_resolve_etymon_id``, which
    has no ``merged_into_id IS NULL`` filter): edges are id-keyed regardless of
    tombstone state, so an endpoint (or ``from``) already folded by an earlier
    row must still resolve, else its wrong-sense edge would silently survive.
    ``detach_unresolved_*`` thus counts only genuine typos, and a re-run is a
    clean no-op (DELETE of an absent row → rowcount 0). ``dict.fromkeys`` dedups
    while preserving order so an accidental duplicate ref can't make dry-run
    (apply=False, COUNT returns 1 per occurrence) telemetry diverge from the
    applied result (the second DELETE hits 0 rows)."""
    detach_parents = list(dict.fromkeys(payload.get("detach_parents") or []))
    detach_children = list(dict.fromkeys(payload.get("detach_children") or []))
    if not (detach_parents or detach_children):
        return False
    detach_from_id = _resolve_etymon_id(ctx.db.conn, from_ref)
    if detach_from_id is None:
        ctx.counts["detach_unresolved_from"] += 1
        return True
    for endpoint_ref, is_parent in (
        *((p, True) for p in detach_parents),
        *((c, False) for c in detach_children),
    ):
        endpoint_id = _resolve_etymon_id(ctx.db.conn, endpoint_ref)
        if endpoint_id is None:
            ctx.counts["detach_unresolved_endpoint"] += 1
            continue
        # detach_parents: <endpoint> is the PARENT; detach_children:
        # <endpoint> is the CHILD. detach_from is the other end.
        if is_parent:
            parent_id, child_id = endpoint_id, detach_from_id
        else:
            parent_id, child_id = detach_from_id, endpoint_id
        if ctx.apply:
            cur = ctx.db.conn.execute(
                "DELETE FROM etymon_descent WHERE parent_id = ? AND child_id = ?",
                (parent_id, child_id),
            )
            ctx.counts["detach_edges_removed"] += cur.rowcount or 0
        else:
            ctx.counts["detach_edges_removed"] += ctx.db.conn.execute(
                "SELECT COUNT(*) AS n FROM etymon_descent WHERE parent_id = ? AND child_id = ?",
                (parent_id, child_id),
            ).fetchone()["n"]
    return True


def _migrate_collapse_evidence(
    ctx: _CollapseContext, from_row: Any, into_row: Any, payload: dict[str, Any]
) -> None:
    """The fold's DB writes: record the from-form as an ``etymon_variant`` of
    the surviving lemma, migrate reflexes + citations to ``into`` (OR IGNORE;
    UNIQUE-conflict survivors stay on ``from``, counted under
    ``*_skipped_conflict``), stamp migrated citations' ``attested_form`` with
    the from-form (wyrd-jhdw hybrid; the ``attested_form IS NULL`` guard keeps
    an earlier collapse's form from being re-stamped), then tombstone ``from``
    into ``into`` via ``merged_into_id``. Caller gates this on ``apply``."""
    from_id, into_id = from_row["id"], into_row["id"]
    from_form = from_row["canonical_form"]
    variant_class = payload.get("variant_class") or "alternative"
    if variant_class not in _VARIANT_CLASSES:
        # OOV variant_class from a hand-edited / LLM row would trip the
        # table's CHECK constraint; coerce to the catch-all.
        variant_class = "other"

    # 1. Record the form as a variant of the surviving lemma. UNIQUE
    #    (etymon_id, form, variant_class) → OR IGNORE no-ops when the
    #    bulk ingest already registered it (the common case).
    cur = ctx.db.conn.execute(
        "INSERT OR IGNORE INTO etymon_variant "
        "(etymon_id, form, variant_class, source_id) VALUES (?, ?, ?, ?)",
        (into_id, from_form, variant_class, COLLAPSE_VARIANT_SOURCE_ID),
    )
    ctx.counts["variants_created"] += cur.rowcount or 0

    # 2. Migrate reflexes. PRIMARY KEY (reflex_id, etymon_id) → OR
    #    IGNORE survives the case where `into` already carries the
    #    same reflex; those rows stay on `from` (counted).
    reflex_before = ctx.db.conn.execute(
        "SELECT COUNT(*) AS n FROM reflex_etymon WHERE etymon_id = ?", (from_id,)
    ).fetchone()["n"]
    cur = ctx.db.conn.execute(
        "UPDATE OR IGNORE reflex_etymon SET etymon_id = ? WHERE etymon_id = ?",
        (into_id, from_id),
    )
    reflex_moved = cur.rowcount or 0
    ctx.counts["reflexes_moved"] += reflex_moved
    ctx.counts["reflexes_skipped_conflict"] += reflex_before - reflex_moved

    # 3. Migrate citations + stamp the attested form (wyrd-jhdw hybrid).
    #    UNIQUE (etymon_id, source_id, COALESCE(page,'')) → OR IGNORE.
    cit_before = ctx.db.conn.execute(
        "SELECT COUNT(*) AS n FROM etymon_citation WHERE etymon_id = ? AND attested_form IS NULL",
        (from_id,),
    ).fetchone()["n"]
    cur = ctx.db.conn.execute(
        "UPDATE OR IGNORE etymon_citation "
        "SET etymon_id = ?, attested_form = ? "
        "WHERE etymon_id = ? AND attested_form IS NULL",
        (into_id, from_form, from_id),
    )
    cit_moved = cur.rowcount or 0
    ctx.counts["citations_moved"] += cit_moved
    ctx.counts["citations_skipped_conflict"] += cit_before - cit_moved

    # 4. Tombstone `from` into `into` — it is no longer an independent
    #    lemma. cluster_cognates + the export resolution follow this.
    ctx.db.conn.execute(
        "UPDATE etymon SET merged_into_id = ? WHERE id = ?",
        (into_id, from_id),
    )


def _apply_collapse_fold(
    ctx: _CollapseContext, from_ref: str, payload: dict[str, Any], *, had_detach: bool
) -> None:
    """The ``into`` fold path. A detach-only row (no ``into``) is a legitimate
    no-fold event; only a row with neither detach nor into is an
    ``empty_into_skipped``. Resolves from/into live (non-tombstoned), skips a
    self-collapse, counts ``collapses_processed``, and — only when applying —
    performs the variant/reflex/citation migration + tombstone via
    :func:`_migrate_collapse_evidence`."""
    into_ref = payload.get("into")
    if not into_ref:
        if not had_detach:
            ctx.counts["empty_into_skipped"] += 1
        return
    from_row = _resolve_live_etymon(ctx.db.conn, from_ref)
    if from_row is None:
        ctx.counts["unresolved_from"] += 1
        return
    into_row = _resolve_live_etymon(ctx.db.conn, into_ref)
    if into_row is None:
        ctx.counts["unresolved_into"] += 1
        return
    if from_row["id"] == into_row["id"]:
        ctx.counts["self_collapse_skipped"] += 1
        return
    ctx.counts["collapses_processed"] += 1
    if not ctx.apply:
        return
    _migrate_collapse_evidence(ctx, from_row, into_row, payload)


def apply_collapses(
    db: LexiconDB,
    collapse_state: dict[str, dict[str, Any]],
    *,
    apply: bool = True,
) -> dict[str, Any]:
    """Fold form-of / variant etymons into their lemma (wyrd-y651, the
    consolidation epic's shared collapse applier — used by both the
    deterministic detector and the LLM merge pass, which differ only in
    how they emit ``_collapses.jsonl`` rows).

    ``collapse_state`` is the merged ``{from_ref: payload}`` dict from
    :func:`jsonl.build.collect_collapses`. ``from_ref`` is the etymon
    being absorbed (``"<language>:<canonical_form>"``); the payload names
    the surviving lemma + variant class::

        {
          "into": "old-english:burg",
          "variant_class": "alternative",
          "method": "deterministic-gloss-overlap",
          "reason": "..."
        }

    For each collapse the ``from`` etymon is TOMBSTONED into ``into`` via
    ``merged_into_id`` (the existing pointer cluster_cognates / the export
    resolution respect), its surface form recorded as an
    ``etymon_variant`` of ``into``, and its **reflexes + citations
    MIGRATED** to ``into`` (D21: evidence is moved, not destroyed).
    Migrating citations stamp ``attested_form`` with the from-form
    (wyrd-jhdw hybrid) so scholar-witness counts read off the surviving
    lemma while the original attested form ("Ekwall attests the burh form
    of burg") is preserved.

    UNIQUE-conflict survivors — a reflex/cite ``into`` already has — are
    ``OR IGNORE`` skipped and counted; they stay on the tombstoned
    ``from``, which ``merged_into_id`` routes through at read time.

    No-ops: empty/absent ``into``, ``into == from``, or an unresolved
    ref. Idempotent: ``_resolve_live_etymon`` excludes already-tombstoned
    rows (``merged_into_id IS NULL``), so a second apply no longer finds
    the ``from`` etymon — it is counted ``unresolved_from`` and changes
    nothing further.

    An out-of-vocabulary ``variant_class`` is coerced to ``"other"`` so
    a hand-edited / LLM-emitted row can't trip the table's CHECK
    constraint.

    Validation always runs; DB writes happen only when ``apply=True``.
    Each row's type is dispatched to a handler — reflex-link
    (:func:`_apply_reflex_link`), edge-detach (:func:`_apply_edge_detach`),
    or the ``into`` fold (:func:`_apply_collapse_fold`). Returns a counts
    dict for telemetry.
    """
    counts: dict[str, Any] = {
        "collapses_processed": 0,
        "variants_created": 0,
        "reflexes_moved": 0,
        "reflexes_skipped_conflict": 0,
        "citations_moved": 0,
        "citations_skipped_conflict": 0,
        "unresolved_from": 0,
        "unresolved_into": 0,
        "self_collapse_skipped": 0,
        "empty_into_skipped": 0,
        # wyrd-rogd.15: reflex-LINK rows (inherits) — add an inheritance descent
        # edge instead of folding, so cross-era reflexes stay distinct era forms
        # of one morpheme (the rollup follows the edge).
        "links_processed": 0,
        "link_rejections": 0,
        "unresolved_inherits": 0,
        "self_link_skipped": 0,
        # wyrd-qp9c: descent-edge DETACH — wrong-sense edges removed from a
        # homograph-conflated row before the fold (see the detach block below).
        "detach_edges_removed": 0,
        "detach_unresolved_from": 0,
        "detach_unresolved_endpoint": 0,
        "applied": apply,
        "method_version": COLLAPSE_METHOD_VERSION,
    }

    if apply and collapse_state:
        # Ensure the synthetic variant source exists (derived; re-created
        # idempotently on every rebuild). source.title is NOT NULL.
        db.conn.execute(
            "INSERT OR IGNORE INTO source (id, title) VALUES (?, ?)",
            (COLLAPSE_VARIANT_SOURCE_ID, "Etymon collapse (derived)"),
        )

    ctx = _CollapseContext(db=db, counts=counts, apply=apply)
    for from_ref, payload in collapse_state.items():
        # A row carrying ``inherits`` is a reflex-LINK, regardless of value
        # (mutually exclusive with ``into``); handle it and move on so an
        # ``inherits: ""`` no-op never falls through to the fold.
        if "inherits" in payload:
            _apply_reflex_link(ctx, from_ref, payload)
            continue
        # Detach wrong-sense edges (if any) BEFORE the fold; the return tells
        # the fold whether a no-``into`` row was a legit detach-only event or
        # a truly empty skip.
        had_detach = _apply_edge_detach(ctx, from_ref, payload)
        _apply_collapse_fold(ctx, from_ref, payload, had_detach=had_detach)

    return counts


def _run_curation_slot_passes(
    db: LexiconDB,
    states: dict[str, Any],
    order: list[str],
    *,
    apply: bool,
) -> dict[str, Any]:
    """Run the optional operator/derived curation-slot passes whose state
    is provided, in dependency order, appending each run pass's pipeline
    name to ``order``. Returns ``{result_key: counts}`` (None when the
    pass was skipped).

    The slot runs AFTER link-lemmas and BEFORE the L3 derivations so the
    derivations see the post-curation / post-split / post-collapse graph
    and the cleaned-up gloss + tag inventory. Validation + counting always
    run for a provided state; only the DB writes are gated by ``apply``.

    Table order is the dependency order: curation overrides first, then
    gloss suppression then addition (suppress-then-re-add in one call),
    then etymon-splits + their inverse collapses, then element-gloss
    backfill over the post-collapse graph, then the LLM tag backfill
    (apply-tag-additions, wyrd-xz3g) — independent of the others; it just
    INSERT-OR-IGNOREs etymon_tag rows, so it runs last in the slot.
    """
    # Lazy import (cold-start cost; no import cycle) — kept function-local.
    from .lexicon.element_gloss_backfill import apply_element_glosses

    # (result_key, pass function, pipeline-order name), in execution order.
    passes = [
        ("curation", apply_curation_overrides, "apply-curation"),
        ("gloss_suppressions", apply_gloss_suppressions, "apply-gloss-suppressions"),
        ("gloss_additions", apply_gloss_additions, "apply-gloss-additions"),
        ("etymon_splits", apply_etymon_splits, "apply-etymon-splits"),
        ("collapses", apply_collapses, "apply-collapses"),
        ("element_glosses", apply_element_glosses, "apply-element-glosses"),
        ("tag_additions", apply_tag_additions, "apply-tag-additions"),
    ]
    counts: dict[str, Any] = {}
    for key, pass_fn, order_name in passes:
        state = states.get(key)
        if state is None:
            counts[key] = None
            continue
        counts[key] = pass_fn(db, state, apply=apply)
        order.append(order_name)
    return counts


def run_full_enrichment(
    db: LexiconDB,
    *,
    apply: bool = False,
    curation_state: dict[str, dict[str, Any]] | None = None,
    suppression_state: dict[str, dict[str, Any]] | None = None,
    addition_state: dict[str, dict[str, Any]] | None = None,
    split_state: dict[str, dict[str, Any]] | None = None,
    collapse_state: dict[str, dict[str, Any]] | None = None,
    element_gloss_state: list[dict[str, str]] | None = None,
    tag_state: dict[str, dict[str, Any]] | None = None,
    pronunciation_state: dict[tuple[str, str], str] | None = None,
    canonicalization_dir: Path | None = None,
    skip_l3_derivations: bool = False,
) -> dict[str, Any]:
    """Run the canonical L3 enrichment chain (wyrd-hidb Phase 2).

    Order (with dependency rationale per the wyrd-hidb plan):

    1. ``cluster_ocr_variants`` — tombstone OCR-variant duplicates so
       later passes don't waste work on dead rows.
    2. ``link_lemmas`` — inflected → canonical-lemma linkage. Must
       follow OCR (lemma candidates can't be tombstoned variants).
    3. ``apply_curation_overrides`` / ``apply_etymon_splits`` /
       ``apply_collapses`` (when the respective state is passed) — overlay
       operator + derived decisions on lemma / merge / split / collapse
       assignments. Collapses (wyrd-y651) fold form-of/variant etymons
       into their lemma; they run here, before the L3 derivations, so
       decompose / cluster / proportions see the post-collapse graph.
    4. ``decompose_all`` — matcher-derived toponym decompositions.
       Matches the toponym's modern_name against the live
       ``meanings.json`` bundle, so it doesn't strictly need OCR /
       lemma settled — but running it after the auto-clustering +
       curation passes means the etymon ids the canonical breakdowns
       reference correspond to the post-cluster canonical rows, not
       tombstoned ones.
    5. ``cluster_cognates`` — walks the ``etymon_descent`` graph
       from each Proto-* root and assigns ``cognate_id`` to every
       reachable descendant. The OCR pass populates
       ``merged_into_id`` (tombstone pointer), which cluster_cognates
       respects when picking the canonical etymon for each cluster;
       so OCR must precede this step.
    6. ``classify_stratum_all`` — per-language stratum buckets. Reads
       ``etymon.language`` + the ``etymon_descent`` parent edges
       populated by L2 replay; idempotent under the
       ``stratum IS NULL`` gate.
    7. ``derive_english_shaped_all`` — non-Latin-script
       transliteration. Reads ``etymon.canonical_form`` +
       ``transliteration`` + ``pronunciation_ipa``; independent of the
       earlier passes' column writes, but placed after them so a
       round-trip rebuild has stable canonical_form values by the
       time english_shaped derivation runs.
    8. ``project_period_forms`` — wyrd-unuo Phase 3.3: Tier-3
       period-form fallback. Reads ``toponym_attestation`` rows + the
       cognate_id assignments from step 5 (cluster mates contribute
       suffix candidates), so cluster_cognates must precede this
       step.

    Dry-run (``apply=False``) walks every pass and reports
    candidates without writing. Curation overrides ARE validated on
    dry-run; the per-pass dry-run behavior is delegated to each
    helper.

    ``skip_l3_derivations=True`` stops the chain after the
    OCR+lemma+curation prefix — useful for tests that don't care
    about the derived columns + for fast iteration on curation
    work. Defaults to False (Phase 2 default is "run everything").
    """
    ocr_result = cluster_ocr_variants(db, apply=apply)
    lemma_result = link_lemmas(db, apply=apply)

    # Optional curation-slot passes (operator + derived overlays): each
    # runs only when its state was provided, in the dependency order fixed
    # by _run_curation_slot_passes' table. They execute AFTER link-lemmas
    # and BEFORE the L3 derivations so decompose / cluster-cognates /
    # stratum / english_shaped see the post-curation, post-split,
    # post-collapse graph + cleaned-up gloss/tag inventory.
    order: list[str] = ["normalize-ocr", "link-lemmas"]
    slot_states = {
        "curation": curation_state,
        "gloss_suppressions": suppression_state,
        "gloss_additions": addition_state,
        "etymon_splits": split_state,
        "collapses": collapse_state,
        "element_glosses": element_gloss_state,
        "tag_additions": tag_state,
    }
    slot_counts = _run_curation_slot_passes(db, slot_states, order, apply=apply)

    decompose_result: dict[str, Any] | None = None
    descent_result: dict[str, Any] | None = None
    cognate_result: dict[str, Any] | None = None
    stratum_result: dict[str, Any] | None = None
    english_shaped_result: dict[str, Any] | None = None
    pronunciation_result: dict[str, Any] | None = None
    phonological_vector_result: dict[str, Any] | None = None
    period_form_result: dict[str, Any] | None = None
    canonical_result: dict[str, Any] | None = None

    if not skip_l3_derivations:
        decompose_result = decompose_all(db, apply=apply)
        order.append("decompose")
        # wyrd-zrce.1: project Family-B descends-from assertions into etymon_descent
        # BEFORE cluster_cognates, so mined cognate-descent edges (uplift 1b) feed the
        # cognate_id rollup in this same run. Relational + EARLY (unlike the terminal
        # identity project_canonical). Only when a mining dir is supplied (rebuild
        # passes it); medium-gated by default (D50.4 — relational, never-collapse).
        if canonicalization_dir is not None:
            descent_projection = project_descent_assertions(
                db, mining_dir=canonicalization_dir, apply=apply
            )
            descent_result = {
                "inserted": descent_projection.inserted,
                "cleared": descent_projection.cleared,
                "gated_out": descent_projection.gated_out,
                "refuted": descent_projection.refuted,
            }
            order.append("project-descent")
        cognate_result = cluster_cognates(db, apply=apply)
        order.append("cluster-cognates")
        stratum_result = classify_stratum_all(db, apply=apply)
        order.append("classify-stratum")
        english_shaped_result = derive_english_shaped_all(db, apply=apply)
        order.append("derive-english-shaped")
        # wyrd-vm8t: backfill pronunciation_ipa from the G2P (fill the
        # ON/OE/welsh gaps + fix OE initial-h /x/→/h/). Runs before
        # phonological-vectors so the vectors read the now-fuller IPA.
        pronunciation_result = derive_pronunciation_ipa(
            db, apply=apply, llm_state=pronunciation_state
        )
        order.append("derive-pronunciation-ipa")
        # wyrd-kq7w.1: tag phonological vectors after english_shaped so
        # both passes have settled IPA + canonical_form values to read
        # from. Incremental (NULL-only) by default; the standalone
        # ``lexicon tag-phonological-vectors --force`` recomputes the
        # whole corpus.
        phonological_vector_result = tag_phonological_vectors_all(db, apply=apply)
        order.append("tag-phonological-vectors")
        period_form_result = project_period_forms(db, apply=apply)
        order.append("project-period-forms")
        # Terminal pass: project the L2 canonicalization assertions into the L3
        # collapse graph (wyrd-u6fn.3, D50.6), folding today's deterministic
        # merged_into_id/lemma_id identity clustering by default (wyrd-u6fn.4) so
        # the canonical graph reproduces it. Additive — it populates canonical_*
        # but does NOT yet change the legacy merged_into_id/cognate_id/lemma_id
        # READERS (that cutover is wyrd-b2mf). Only runs when a mining dir is
        # supplied (rebuild-from-jsonl passes it).
        if canonicalization_dir is not None:
            canonical_projection = project_canonical(
                db, mining_dir=canonicalization_dir, apply=apply
            )
            canonical_result = {
                "minted": canonical_projection.minted,
                "bound": canonical_projection.bound,
                "merged": canonical_projection.merged,
                "labels": canonical_projection.labels,
                "conflicts": len(canonical_projection.conflicts),
                "skipped_unsupported": canonical_projection.skipped_unsupported,
            }
            order.append("project-canonical")

    return {
        "order": order,
        "applied": apply,
        "ocr": {
            "method_version": OCR_METHOD_VERSION,
            "groups": ocr_result["groups"],
            "etymons_merged": ocr_result["etymons_merged"],
            "sample_groups": ocr_result.get("sample_groups", []),
        },
        "lemmas": {
            "method_version": LEMMA_METHOD_VERSION,
            "candidates": lemma_result["candidates"],
            "sample": lemma_result.get("sample", []),
        },
        "curation": slot_counts["curation"],
        "gloss_suppressions": slot_counts["gloss_suppressions"],
        "gloss_additions": slot_counts["gloss_additions"],
        "etymon_splits": slot_counts["etymon_splits"],
        "collapses": slot_counts["collapses"],
        "element_glosses": slot_counts["element_glosses"],
        "tag_additions": slot_counts["tag_additions"],
        "pronunciation_ipa": pronunciation_result,
        "decompose": decompose_result,
        "descent": descent_result,
        "cognates": cognate_result,
        "stratum": stratum_result,
        "english_shaped": english_shaped_result,
        "phonological_vectors": phonological_vector_result,
        "period_forms": period_form_result,
        "canonical": canonical_result,
    }


def enrichment_status(conn: sqlite3.Connection) -> dict[str, Any]:
    """Report which L3 enrichment columns are populated on ``etymon``.

    Per-column coverage: how many rows have the column set vs total
    etymon rows. Pulls method-version distributions for the two
    columns that carry one (``lemma_method``, ``cognate_method``).

    All counts come from a single aggregated SELECT — one table scan
    on a 2M+ row table instead of six. ``COUNT(col)`` in SQLite
    counts only non-NULL values, so each ``populated`` figure is one
    column in the same aggregate. The GROUP BY method-distribution
    queries are separate (one per method-bearing column) but use
    hardcoded column names, never operator input — no SQL injection
    surface.

    Read-only against the DB. Temporarily swaps ``conn.row_factory``
    to ``sqlite3.Row`` so column-name access works regardless of the
    caller's factory, then restores the original via try/finally —
    the connection's state is preserved across the call.
    """
    original_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        counts = conn.execute(
            """SELECT COUNT(*)                   AS total,
                      COUNT(lemma_id)            AS lemma_id_pop,
                      COUNT(merged_into_id)      AS merged_into_id_pop,
                      COUNT(cognate_id)          AS cognate_id_pop,
                      COUNT(stratum)             AS stratum_pop,
                      COUNT(english_shaped)      AS english_shaped_pop,
                      COUNT(phonological_vector) AS phonological_vector_pop
                 FROM etymon"""
        ).fetchone()

        lemma_methods = [
            {"method": r["lemma_method"], "count": r["count"]}
            for r in conn.execute(
                "SELECT lemma_method, COUNT(*) AS count FROM etymon "
                "WHERE lemma_method IS NOT NULL "
                "GROUP BY lemma_method ORDER BY count DESC"
            )
        ]
        cognate_methods = [
            {"method": r["cognate_method"], "count": r["count"]}
            for r in conn.execute(
                "SELECT cognate_method, COUNT(*) AS count FROM etymon "
                "WHERE cognate_method IS NOT NULL "
                "GROUP BY cognate_method ORDER BY count DESC"
            )
        ]
    finally:
        conn.row_factory = original_factory

    return {
        "total_etymons": counts["total"],
        "columns": {
            "lemma_id": {
                "populated": counts["lemma_id_pop"],
                "methods": lemma_methods,
            },
            "merged_into_id": {
                "populated": counts["merged_into_id_pop"],
                # cluster_ocr_variants doesn't write a per-row method
                # version (the 'cluster-ocr-v1' string is implicit in
                # the writer). When that changes the methods list grows.
                "methods": [],
            },
            "cognate_id": {
                "populated": counts["cognate_id_pop"],
                "methods": cognate_methods,
            },
            "stratum": {
                "populated": counts["stratum_pop"],
                "methods": [],
            },
            "english_shaped": {
                "populated": counts["english_shaped_pop"],
                "methods": [],
            },
            "phonological_vector": {
                "populated": counts["phonological_vector_pop"],
                # tag_phonological_vectors_all stamps the version
                # implicitly via PHON_VECTOR_METHOD_VERSION; the per-row
                # column doesn't carry it. When a future version-bump
                # adds a per-row method column the methods list grows.
                "methods": [],
            },
        },
    }


def format_enrichment_status(status: dict[str, Any]) -> str:
    """Render :func:`enrichment_status` as a grep-friendly markdown
    report."""
    total = status["total_etymons"]
    lines: list[str] = [
        "## L3 enrichment status",
        "",
        f"- Total etymons: {total}",
        "",
        "### Column coverage",
    ]
    for column, info in status["columns"].items():
        pct = (info["populated"] / total * 100.0) if total else 0.0
        lines.append(f"- `{column}`: {info['populated']:>8} / {total} ({pct:5.1f}%)")
        for m in info.get("methods", []):
            lines.append(f"  - method=`{m['method']}`: {m['count']}")
    return "\n".join(lines)


def _format_decompose_section(d: dict[str, Any]) -> list[str]:
    return [
        "### Decomposition",
        f"- Toponyms scanned: {d['toponyms_scanned']}",
        f"- Decompositions written: {d['decompositions']}",
        f"- Rule firings: scholar={d.get('scholar', 0)}, "
        f"scholar-disagreement={d.get('scholar-disagreement', 0)}, "
        f"unique-zero={d.get('unique-zero-unaccounted', 0)}, "
        f"tiebreaker={d.get('tiebreaker', 0)}, "
        f"no-canonical={d.get('no-canonical', 0)}",
    ]


def _format_cognates_section(c: dict[str, Any]) -> list[str]:
    return [
        "### Cognate clustering",
        f"- Roots walked: {c.get('roots', 0)}",
        f"- Cognate_id assignments: {c.get('candidates', 0)}",
    ]


def _format_stratum_section(s: dict[str, Any]) -> list[str]:
    out = ["### Stratum classification"]
    for lang, counts in s["languages"].items():
        out.append(
            f"- {lang}: proposed={counts['proposed']}, "
            f"written={counts.get('written', 0)}, "
            f"skipped={counts.get('skipped', 0)}"
        )
        # Sorted for deterministic report output (the underlying dict is
        # built by classify_stratum_all in proposal-iteration order).
        by_stratum = counts.get("by_stratum") or {}
        if by_stratum:
            breakdown = ", ".join(f"{stratum}={n}" for stratum, n in sorted(by_stratum.items()))
            out.append(f"  - by stratum: {breakdown}")
    return out


def _format_english_shaped_section(e: dict[str, Any]) -> list[str]:
    return [
        "### English-shaped derivation",
        f"- Candidates: {e['candidates']}",
        f"- Written: {e['written']}",
        f"- Skipped (no input): {e['skipped_no_input']}",
    ]


def _format_period_forms_section(p: dict[str, Any]) -> list[str]:
    return [
        "### Period-form projection",
        f"- Rows scanned: {p.get('rows_scanned', 0)}",
        f"- Projected: {p.get('rows_projected', 0)}",
        f"- Candidates: {p.get('candidates', 0)}",
        f"- Rows written: {p.get('rows_written', 0)}",
    ]


# Table of optional sections: (result-dict key, render function). Order
# here drives section ordering in the rendered markdown. To add a new
# pass: append a (key, render_fn) tuple and write a _format_<x>_section
# helper. No edits to format_enrichment_run needed.
_OPTIONAL_SECTIONS: tuple[tuple[str, Any], ...] = (
    ("decompose", _format_decompose_section),
    ("cognates", _format_cognates_section),
    ("stratum", _format_stratum_section),
    ("english_shaped", _format_english_shaped_section),
    ("period_forms", _format_period_forms_section),
)


def format_enrichment_run(result: dict[str, Any]) -> str:
    """Render :func:`run_full_enrichment` output as markdown."""
    verb_ocr = "merged" if result["applied"] else "mergeable"
    verb_lemmas = "linked" if result["applied"] else "linkable"
    lines: list[str] = [
        "## L3 enrichment chain",
        "",
        f"- Order: {' → '.join(result['order'])}",
        f"- Applied: {result['applied']}",
        "",
        "### OCR clustering (`" + result["ocr"]["method_version"] + "`)",
        f"- Variant groups: {result['ocr']['groups']}",
        f"- Etymons {verb_ocr}: {result['ocr']['etymons_merged']}",
        "",
        "### Lemma linkage (`" + result["lemmas"]["method_version"] + "`)",
        f"- Inflected etymons {verb_lemmas}: {result['lemmas']['candidates']}",
    ]
    if result.get("curation"):
        lines.append("")
        lines.append(format_curation_run(result["curation"]))
    if result.get("gloss_suppressions"):
        lines.append("")
        lines.append(format_gloss_suppression_run(result["gloss_suppressions"]))
    if result.get("gloss_additions"):
        lines.append("")
        lines.append(format_gloss_addition_run(result["gloss_additions"]))
    if result.get("etymon_splits"):
        lines.append("")
        lines.append(format_etymon_split_run(result["etymon_splits"]))
    if result.get("element_glosses"):
        eg = result["element_glosses"]
        lines.append("")
        lines.append("### Element-gloss backfill (wyrd-u9k6)")
        lines.append(
            f"- Orphan-reflex surfaces linked to a grounded etymon: "
            f"{eg.get('linked_surfaces', 0)} (links {eg.get('links', 0)}; "
            f"no-reflex {eg.get('no_reflex', 0)}, no-etymon {eg.get('no_etymon', 0)})"
        )
    if result.get("tag_additions"):
        ta = result["tag_additions"]
        lines.append("")
        lines.append("### Tag backfill (`" + ta["method_version"] + "`, wyrd-xz3g)")
        lines.append(
            f"- Etymons tagged: {ta['etymons_touched']} "
            f"(tags added {ta['tags_added']}, already-present "
            f"{ta['tags_already_present']}, unresolved {ta['unresolved_etymon']})"
        )
    if result.get("pronunciation_ipa"):
        from .lexicon.pronunciation_backfill import format_pronunciation_run

        lines.append("")
        lines.append(format_pronunciation_run(result["pronunciation_ipa"]))
    for key, render in _OPTIONAL_SECTIONS:
        payload = result.get(key)
        if payload is None:
            continue
        lines.append("")
        lines.extend(render(payload))
    return "\n".join(lines)


def format_unresolved_warnings(result: dict[str, Any]) -> str:
    """wyrd-van9: pull the unresolved-ref / typo counters out of the
    enrichment result + render a stderr warning block when any are
    non-zero. Empty string when everything resolved cleanly.

    The CLI's ``--strict`` flag uses the non-empty return as a fail
    signal; without ``--strict`` the warnings still surface on stderr
    so an interactive operator sees them, but the exit code stays 0.
    """
    sections: list[tuple[str, dict[str, int]]] = []
    curation = result.get("curation")
    if curation:
        sections.append(
            (
                "curation",
                {
                    "unresolved_etymon": curation.get("unresolved_etymon", 0),
                    "unresolved_lemma_ref": curation.get("unresolved_lemma_ref", 0),
                    "unresolved_merge_ref": curation.get("unresolved_merge_ref", 0),
                    "self_reference_lemma": curation.get("self_reference_lemma", 0),
                    "self_reference_merge": curation.get("self_reference_merge", 0),
                },
            )
        )
    sup = result.get("gloss_suppressions")
    if sup:
        sections.append(
            (
                "gloss_suppressions",
                {
                    "unresolved_etymon": sup.get("unresolved_etymon", 0),
                    "unresolved_gloss": sup.get("unresolved_gloss", 0),
                },
            )
        )
    addition = result.get("gloss_additions")
    if addition:
        sections.append(
            (
                "gloss_additions",
                {
                    "unresolved_etymon": addition.get("unresolved_etymon", 0),
                },
            )
        )
    splits = result.get("etymon_splits")
    if splits:
        sections.append(
            (
                "etymon_splits",
                {
                    "unresolved_etymon": splits.get("unresolved_etymon", 0),
                    "glosses_missing": splits.get("glosses_missing", 0),
                    "tags_missing": splits.get("tags_missing", 0),
                    "children_skipped_no_suffix": splits.get("children_skipped_no_suffix", 0),
                    "children_skipped_invalid_suffix": splits.get(
                        "children_skipped_invalid_suffix", 0
                    ),
                    "multiple_primary_collapsed": splits.get("multiple_primary_collapsed", 0),
                },
            )
        )
    flagged: list[tuple[str, str, int]] = []
    for name, counters in sections:
        for counter, value in counters.items():
            if value:
                flagged.append((name, counter, value))
    if not flagged:
        return ""
    lines = ["## ⚠ Unresolved-ref / typo signals (wyrd-van9)"]
    lines.append("")
    lines.append("These signals indicate operator-typo or stale-event")
    lines.append("conditions that produced silent no-ops during apply.")
    lines.append("Each line is `<event-type> <counter> <count>`:")
    lines.append("")
    for name, counter, value in flagged:
        lines.append(f"- {name}.{counter}: {value}")
    return "\n".join(lines)
