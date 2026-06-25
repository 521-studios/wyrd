"""Same-morpheme identity uplift for admitted breakdown morphemes (wyrd-c7iz, D50).

Uplift 1a: an admitted breakdown morpheme (wyrd-oth3) that is the SAME morpheme as
one already shipped should be bound to a shared ``canonical_morpheme`` node (D50
Family A **identity**), not enter the bundle as a duplicate family. This module
authors the D50-native assertions: a ``mint-canonical`` node + a same-morpheme
``bind`` per member observation.

Identity, NOT mere cognate (D50.2): a ``bind`` asserts SAME morpheme — collapsible
— so it has a high-confidence bar and defaults to leave-separate. Cognate-but-
distinct pairs (``silly``/``selig``) must NOT bind; that is the relational
``descends-from`` axis, handled elsewhere. The signal here is therefore stronger
than a shared cognate cluster:

    a breakdown-only etymon X and a shipped etymon Y are bound iff they share a
    **folded surface** AND (**same cognate cluster** OR **overlapping gloss**),
    and Y is the UNIQUE such shipped match (ambiguous → left separate, D46).

Same surface + same cluster → ``high`` confidence (a near-certain duplicate row);
same surface + gloss-only (unclustered/cross-cluster) → ``medium``. The folded
surface guards against binding a genuine homograph (``ton``=town vs ``ton``=wave:
same surface, disjoint gloss + cluster → no bind).

DORMANT until u6fn.3: the binds live in the L2 canonicalization streams
(``_canonical_nodes.jsonl`` + ``_assert_bind.jsonl``); the projection that folds
them into the L3 collapse graph (generalizing the ``cognate_id`` / ``merged_into_id``
rollup) is wyrd-u6fn.3. Until then the binds are authored + validated, not applied.
Canonical-node ids are content-keyed off the shipped morpheme's rebuild-stable
``(language, canonical_form)`` (never an autoincrement id, D50.1). LLM-free,
deterministic. Mirrors ``passthrough_mining``.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from wyrd.generators.kenning.canonicalization import Assertion, NodeRef, mint_canonical_id
from wyrd.generators.kenning.lexicon.bundle._export import (
    RECOMMENDED_LANG_THRESHOLDS,
    build_family_rollup,
    select_promoted_root_ids,
)
from wyrd.generators.kenning.lexicon.db import LexiconDB
from wyrd.generators.kenning.lexicon.etymon_refs import etymon_refs_for
from wyrd.generators.kenning.lexicon.genitive_priors import fold_surface

METHOD = "same-morpheme-uplift-v1"

# Confidence tiers a bind can carry: high = shared cognate cluster (a near-certain
# duplicate); medium = same surface + gloss overlap only.
Confidence = Literal["high", "medium"]

# Placeholder glosses that carry NO morpheme identity — "a personal name" is on 8507
# etymons, "personal name" on 137. They mean "this is a person's name used in a
# place-name", not a morpheme meaning, so two DISTINCT morphemes sharing only one
# (plus a folded surface) are NOT the same morpheme. The medium tier matches glosses
# by string equality, which collapse_merge already documents can't decide a generic
# gloss; here we simply refuse to let a placeholder-only overlap be identity evidence
# (a wrong bind is identity-collapse corruption, D46/D50.2 — leave separate instead).
_GENERIC_GLOSSES: frozenset[str] = frozenset({"a personal name", "personal name"})


@dataclass(frozen=True)
class EtymonRow:
    """The per-etymon facts the matcher needs (one row of the candidate scan)."""

    etymon_id: int
    cluster: str  # cognate cluster key (c<cognate_id> or singleton e<id>)
    language: str
    canonical_form: str
    folded: str
    glosses: frozenset[str]
    is_shipped: bool  # promoted via a non-breakdown path (already in the bundle)
    is_breakdown: bool  # appears as a scholarly toponym_etymology element


def _load_rows(db: LexiconDB) -> list[EtymonRow]:
    """One ``EtymonRow`` per etymon, with cluster, folded surface, glosses, and
    the shipped / breakdown flags."""
    members_by_root, root_of = build_family_rollup(db)
    shipped_roots = set(
        select_promoted_root_ids(
            db,
            lang_thresholds=RECOMMENDED_LANG_THRESHOLDS,
            min_witnesses=2,
            include_rando=True,
            include_wiktionary_empirical=True,
            include_wave2_enriched=True,
            include_toponym_breakdown=False,
            root_of=root_of,
            members_by_root=members_by_root,
        )
    )
    # Flatten root_of into an O(1) lookup (members_by_root already covers every
    # etymon) — root_of() is a cycle-walking loop, and calling it per etymon over
    # the ~700k-row table is the dominant cost (Gemini perf finding).
    member_to_root = {
        member: root for root, members in members_by_root.items() for member in members
    }
    breakdown = {
        r["etymon_id"]
        for r in db.conn.execute("SELECT DISTINCT etymon_id FROM toponym_etymology_element")
    }

    # Only shipped or breakdown etymons can match: a "neither" etymon is never in a
    # group's `shipped` list and is always skipped as a candidate `x`, so it can't
    # affect any bind. Drop them in a first pass — over the ~700k-row table that
    # skips the fold + row allocation (and gloss-set building) for the ~600k that
    # are neither (Gemini perf finding).
    relevant: dict[int, tuple[int | None, str, str, str]] = {}
    for r in db.conn.execute("SELECT id, cognate_id, language, canonical_form FROM etymon"):
        eid = r["id"]
        if not (member_to_root.get(eid, eid) in shipped_roots or eid in breakdown):
            continue
        form = r["canonical_form"]
        folded = fold_surface(form) if form else ""
        if not folded:
            continue
        relevant[eid] = (r["cognate_id"], r["language"] or "", form, folded)

    glosses: dict[int, set[str]] = defaultdict(set)
    for r in db.conn.execute("SELECT etymon_id, gloss FROM etymon_gloss"):
        if r["etymon_id"] in relevant and r["gloss"]:
            glosses[r["etymon_id"]].add(r["gloss"].strip().lower())

    return [
        EtymonRow(
            etymon_id=eid,
            cluster=f"c{cid}" if cid is not None else f"e{eid}",
            language=language,
            canonical_form=form,
            folded=folded,
            glosses=frozenset(glosses.get(eid, ())),
            is_shipped=member_to_root.get(eid, eid) in shipped_roots,
            is_breakdown=eid in breakdown,
        )
        for eid, (cid, language, form, folded) in relevant.items()
    ]


@dataclass(frozen=True)
class BindGroup:
    """A same-morpheme identity: one shipped etymon + the breakdown-only variants
    bound to it. ``canonical_parts`` are the rebuild-stable content key for the
    minted node. ``breakdown_confidences`` is each breakdown variant's OWN match
    confidence (``high`` = shared cluster, ``medium`` = gloss-only); ``confidence``
    is the group summary (``high`` if any variant is high). Per-bind authoring uses
    the per-variant confidence so a gloss-only variant isn't overstated as ``high``
    just because a cluster-matched sibling shares the node. ``member_etymons`` are
    all observations bound to the canonical node (shipped first)."""

    canonical_parts: tuple[str, str]
    shipped_etymon: int
    breakdown_confidences: dict[int, Confidence]
    confidence: Confidence
    rep_form: str

    @property
    def breakdown_etymons(self) -> tuple[int, ...]:
        """The bound breakdown variants, sorted — derived from
        ``breakdown_confidences`` (the single source of truth) so the two
        can never disagree."""
        return tuple(sorted(self.breakdown_confidences))

    @property
    def member_etymons(self) -> tuple[int, ...]:
        return (self.shipped_etymon, *self.breakdown_etymons)

    def confidence_for(self, etymon_id: int) -> Confidence:
        """This member's own bind confidence — ``high`` for the shipped etymon
        (it IS the morpheme), else the variant's recorded match confidence."""
        return self.breakdown_confidences.get(etymon_id, "high")


def _match_shipped(x: EtymonRow, shipped: list[EtymonRow]) -> tuple[EtymonRow, Confidence] | None:
    """The UNIQUE shipped same-morpheme match for breakdown-only ``x`` within its
    folded-surface group, plus the confidence. Same cluster → high; else gloss
    overlap → medium. More than one candidate (ambiguous) → ``None`` (D46
    leave-separate)."""
    same_cluster = [y for y in shipped if y.cluster == x.cluster]
    if len(same_cluster) == 1:
        return same_cluster[0], "high"
    if same_cluster:
        return None  # ambiguous within-cluster
    # The unique gloss match (len == 1) is identity evidence only if the OVERLAP
    # carries a NON-placeholder gloss: a shared "a personal name" alone binds two
    # distinct morphemes (old-norse ami / old-english Ami) as one — identity-collapse.
    # The uniqueness gate (== 1) is unchanged, so this never resolves an ambiguous
    # >1-match into a bind; it only suppresses a placeholder-only overlap.
    gloss_match = [y for y in shipped if x.glosses & y.glosses]
    if len(gloss_match) == 1 and (x.glosses & gloss_match[0].glosses) - _GENERIC_GLOSSES:
        return gloss_match[0], "medium"
    return None


def mine_same_morpheme_binds(db: LexiconDB) -> list[BindGroup]:
    """Find breakdown-only etymons that are the same morpheme as a unique shipped
    etymon (shared folded surface + cluster-or-gloss), grouped by the shipped
    target. Deterministic; sorted by the canonical content key."""
    rows = _load_rows(db)
    by_surface: dict[str, list[EtymonRow]] = defaultdict(list)
    for row in rows:
        by_surface[row.folded].append(row)

    # shipped_etymon_id -> {breakdown_etymon_id: its own match confidence}. Per
    # variant, NOT a single group confidence — else a gloss-only (medium) variant
    # would be overstated as high just because a cluster-matched sibling is high.
    matched: dict[int, dict[int, Confidence]] = defaultdict(dict)
    rep: dict[int, EtymonRow] = {}
    for group in by_surface.values():
        shipped = [r for r in group if r.is_shipped]
        if not shipped:
            continue
        for x in group:
            if not x.is_breakdown or x.is_shipped:
                continue
            hit = _match_shipped(x, shipped)
            if hit is None:
                continue
            y, confidence = hit
            matched[y.etymon_id][x.etymon_id] = confidence
            rep[y.etymon_id] = y

    out: list[BindGroup] = []
    for shipped_id, breakdown_confidences in matched.items():
        y = rep[shipped_id]
        out.append(
            BindGroup(
                # Key the canonical node by the shipped morpheme's UNIQUE, rebuild-
                # stable (language, canonical_form) — folded surface is not unique
                # (two same-language etymons can fold alike), which would collide
                # two distinct shipped morphemes onto one node.
                canonical_parts=(y.language, y.canonical_form),
                shipped_etymon=shipped_id,
                breakdown_confidences=dict(breakdown_confidences),
                confidence="high" if "high" in breakdown_confidences.values() else "medium",
                rep_form=y.canonical_form,
            )
        )
    # shipped_etymon in the key gives a total order (canonical_parts alone can tie).
    out.sort(key=lambda g: (g.canonical_parts, g.shipped_etymon))
    return out


def bind_assertions(
    groups: list[BindGroup], *, conn: sqlite3.Connection, source: str, actor: str = ""
) -> list[Assertion]:
    """Author the D50 assertions for each bind group: one ``mint-canonical``
    declaration of the ``canonical_morpheme`` node, then a same-morpheme ``bind``
    for every member observation (shipped + breakdown variants). Deterministic ids
    (no timestamp), so re-mining the same corpus is idempotent (D36.9).

    wyrd-c6wu: a bind's etymon ref is the stable natural key
    ``"language:canonical_form"`` (via ``etymon_refs_for``), NEVER the etymon
    row-id — row-ids are reassigned on every ``rebuild-from-jsonl``, so an
    id-keyed bind would orphan after any corpus change. ``conn`` is the live
    lexicon connection the member ids were mined from in THIS transaction, so a
    member id must be a live row; ``refs[etymon_id]`` therefore indexes directly
    and raises loudly on a miss — a missing freshly-mined id is a programming
    error, not a data condition (mirrors ``descent_assertions`` / its
    raise-on-incomplete-refs contract). The rebuild-survival safety net lives on
    the READ side (the projection's resolve-by-natural-key, skip-on-miss)."""
    refs = etymon_refs_for(conn, {eid for g in groups for eid in g.member_etymons})
    out: list[Assertion] = []
    for g in groups:
        node_id = mint_canonical_id("canonical_morpheme", *g.canonical_parts)
        node = NodeRef("canonical_morpheme", node_id)
        rationale = (
            f"same-morpheme uplift: breakdown variant(s) of shipped '{g.rep_form}' "
            f"({len(g.breakdown_etymons)} bound)"
        )
        out.append(
            Assertion(
                predicate="mint-canonical",
                subject=node,
                confidence=g.confidence,
                method=METHOD,
                source=source,
                actor=actor,
                rationale=rationale,
            ).with_id()
        )
        for etymon_id in g.member_etymons:
            out.append(
                Assertion(
                    predicate="bind",
                    subject=NodeRef("etymon", refs[etymon_id]),
                    object=node,
                    qualifiers={"kind": "same-morpheme"},
                    # Per-variant confidence: a gloss-only variant stays medium even
                    # if a cluster-matched sibling shares the node (shipped = high).
                    confidence=g.confidence_for(etymon_id),
                    method=METHOD,
                    source=source,
                    actor=actor,
                    rationale=rationale,
                ).with_id()
            )
    return out
