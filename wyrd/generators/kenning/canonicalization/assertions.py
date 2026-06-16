"""Canonicalization assertion record model + predicate catalog (D50).

A canonicalization assertion is one reified, append-only claim about the
corpus. Every assertion has the same uniform shape; the ``predicate`` decides
its arity, which node types its endpoints may carry, and which qualifiers are
valid. The predicate catalog below IS the D50.3 edge taxonomy, enforced:

- **Family A — identity** (collapsible; generalizes ``merged_into_id``):
  ``mint-canonical``, ``bind``, ``merge-canonical``, ``canonical-label``.
  Identity is realized as mint + bind, so one endpoint is ALWAYS a synthetic
  canonical node (D50.1) — there are no observation<->observation identity
  edges.
- **Family B — relational** (never collapse; orthogonal axes, D28):
  ``descends-from``, ``means-same-as``, ``glosses-as``, ``decomposes-into``,
  ``composed-of``, ``contains``, ``succeeds``, ``located-in``.
- **Family C — canonicalization choice** (select / flag among Family-B
  alternatives): ``canonical-decomposition``, ``decomposition-spurious``.

Two invariants the model bakes in:

- **Identity is bare and timeless** (D40/D45/D44/D46): position / era /
  inflection ride in ``qualifiers`` (``ordinal`` / ``stratum`` / ``edge_type``
  / ``kind``), never folded into a node ref. A morpheme's ref is its bare
  surface; a dash is never part of identity.
- **Append-only with polarity** (D21/D22): ``refute`` (a negative claim) and
  ``retract`` (withdraw a prior assertion by id) are new records, never edits.

This module is pure-Python (no DB, no I/O). Streams live in ``streams``; the
L3 projection that consumes these records is the separate ``u6fn.3`` work.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

# --- vocabularies ---------------------------------------------------------

FAMILIES = frozenset({"identity", "relational", "choice"})
POLARITIES = frozenset({"affirm", "refute", "retract"})

# Synthetic identity hubs (D50.1). One endpoint of every Family-A edge.
CANONICAL_NODE_TYPES = frozenset({"canonical_morpheme", "canonical_place", "canonical_sense"})

# ``bind.kind`` values — which identity axis a bind collapses on.
BIND_KINDS = frozenset({"same-morpheme", "same-place", "same-sense", "inflection-of"})

# ``descends-from.edge_type`` values — mirrors the etymon_descent CHECK (D27).
EDGE_TYPES = frozenset(
    {
        "inheritance",
        "borrowing",
        "calque",
        "compound",
        "derivation",
        "cognate",
        "unknown",
    }
)

# Confidence labels accepted alongside a bare 0..1 float.
CONFIDENCE_LABELS = frozenset({"high", "medium", "low"})


class AssertionValidationError(ValueError):
    """Raised when an assertion violates its predicate's spec."""


# --- predicate catalog (the D50.3 taxonomy, as code) ----------------------


@dataclass(frozen=True)
class PredicateSpec:
    """The shape constraints for one predicate.

    ``subject_types`` / ``object_types`` of ``None`` mean "any type". A
    ``None`` ``object_types`` marks a unary predicate (``object`` must be
    absent). ``required`` / ``optional`` are qualifier-key sets; any qualifier
    key outside their union is rejected.
    """

    name: str
    family: str
    subject_types: frozenset[str] | None
    object_types: frozenset[str] | None  # None => unary (no object)
    required: frozenset[str] = frozenset()
    optional: frozenset[str] = frozenset()

    @property
    def is_unary(self) -> bool:
        return self.object_types is None


def _spec(
    name: str,
    family: str,
    subject_types: frozenset[str] | None,
    object_types: frozenset[str] | None,
    required: frozenset[str] = frozenset(),
    optional: frozenset[str] = frozenset(),
) -> PredicateSpec:
    return PredicateSpec(name, family, subject_types, object_types, required, optional)


_OBS_PLACE = frozenset({"toponym", "toponym_etymology"})

PREDICATES: dict[str, PredicateSpec] = {
    # --- Family A: identity ---
    "mint-canonical": _spec("mint-canonical", "identity", CANONICAL_NODE_TYPES, None),
    "bind": _spec(
        "bind",
        "identity",
        frozenset({"etymon", "etymon_gloss", "toponym", "toponym_etymology"}),
        CANONICAL_NODE_TYPES,
        required=frozenset({"kind"}),
    ),
    "merge-canonical": _spec(
        "merge-canonical", "identity", CANONICAL_NODE_TYPES, CANONICAL_NODE_TYPES
    ),
    "canonical-label": _spec(
        "canonical-label",
        "identity",
        CANONICAL_NODE_TYPES,
        None,
        required=frozenset({"value"}),
        optional=frozenset({"stratum"}),
    ),
    # --- Family B: relational ---
    "descends-from": _spec(
        "descends-from",
        "relational",
        frozenset({"etymon"}),
        frozenset({"etymon"}),
        required=frozenset({"edge_type"}),
    ),
    "means-same-as": _spec(
        "means-same-as", "relational", frozenset({"etymon"}), frozenset({"etymon"})
    ),
    "glosses-as": _spec(
        "glosses-as",
        "relational",
        frozenset({"etymon"}),
        frozenset({"sense", "etymon_gloss"}),
    ),
    "decomposes-into": _spec(
        "decomposes-into",
        "relational",
        _OBS_PLACE,
        frozenset({"etymon"}),
        required=frozenset({"ordinal"}),
    ),
    # wyrd-h5u1: a COMPOSITE morpheme is composed of finer constituent morphemes
    # (ington composed-of ing + tūn). The morpheme-level analog of the
    # place-level decomposes-into: part-whole at the etymon axis, orthogonal to
    # descent (a compound is not a descendant of its parts) and to identity (the
    # composite is never the same node as a part — the self-loop guard enforces
    # it). Ordered by ``ordinal``. Mined gloss-correctly from scholarly evidence
    # (cross-scholar coarse-vs-fine breakdowns), never surface segmentation.
    "composed-of": _spec(
        "composed-of",
        "relational",
        frozenset({"etymon"}),
        frozenset({"etymon"}),
        required=frozenset({"ordinal"}),
    ),
    "contains": _spec(
        "contains",
        "relational",
        frozenset({"region_node"}),
        frozenset({"region_node"}),
        optional=frozenset({"stratum"}),
    ),
    "succeeds": _spec(
        "succeeds",
        "relational",
        frozenset({"region_node", "jurisdiction"}),
        frozenset({"region_node", "jurisdiction"}),
        optional=frozenset({"stratum"}),
    ),
    "located-in": _spec(
        "located-in",
        "relational",
        _OBS_PLACE,
        frozenset({"region_node"}),
        optional=frozenset({"stratum"}),
    ),
    # --- Family C: canonicalization choice ---
    "canonical-decomposition": _spec(
        "canonical-decomposition",
        "choice",
        frozenset({"toponym"}),
        frozenset({"decomposition"}),
        optional=frozenset({"culture"}),
    ),
    "decomposition-spurious": _spec(
        "decomposition-spurious",
        "choice",
        frozenset({"toponym_etymology", "decomposition"}),
        None,
        optional=frozenset({"reason"}),
    ),
}


def _validate_predicate_families(
    predicates: dict[str, PredicateSpec], families: frozenset[str]
) -> None:
    """Guard the catalog invariant: every predicate's family is a known family.

    Extracted (rather than inlined at module scope) so the rejection path is
    testable with a synthetic bad-family catalog — the real ``PREDICATES``
    always passes, so an in-place loop could never exercise the raise.
    """
    for spec in predicates.values():
        if spec.family not in families:
            raise AssertionValidationError(
                f"predicate {spec.name!r} has unknown family {spec.family!r}; "
                f"expected one of {sorted(families)}"
            )


# Guards future edits to PREDICATES from introducing a typo'd family string.
_validate_predicate_families(PREDICATES, FAMILIES)


# --- records --------------------------------------------------------------


@dataclass(frozen=True)
class NodeRef:
    """A typed reference to a node (observation row or canonical node)."""

    type: str
    ref: str

    def to_row(self) -> dict[str, str]:
        return {"type": self.type, "ref": self.ref}

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> NodeRef:
        return cls(type=str(row["type"]), ref=str(row["ref"]))


@dataclass(frozen=True)
class Assertion:
    """One canonicalization claim — the uniform record (D50.4).

    ``id`` is content-derived (see :func:`mint_assertion_id`) so re-authoring
    the identical claim is idempotent. ``qualifiers`` carries the
    predicate-specific axes (ordinal / stratum / edge_type / kind / value /
    culture / reason) — never identity.

    ``frozen=True`` blocks field rebinding but not mutation of the
    ``qualifiers`` dict; treat it as read-only after construction (codebase
    convention, cf. ``PhonologicalVector.extras``). The id is derived from
    ``qualifiers``, so mutating it post-``with_id()`` would desync the cached
    id and break the D36.9 determinism guarantee — author a fresh record
    instead.
    """

    predicate: str
    subject: NodeRef
    object: NodeRef | None = None
    polarity: str = "affirm"
    retracts: str | None = None
    qualifiers: dict[str, Any] = field(default_factory=dict)
    confidence: str | float = "medium"
    method: str = ""
    source: str = ""
    actor: str = ""
    rationale: str = ""
    timestamp: str = ""
    id: str = ""

    def with_id(self) -> Assertion:
        """Return a copy whose ``id`` is the minted content hash (if absent)."""
        if self.id:
            return self
        from dataclasses import replace

        return replace(self, id=mint_assertion_id(self))

    def to_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "_type": "canonical_assertion",
            "id": self.id or mint_assertion_id(self),
            "predicate": self.predicate,
            "subject": self.subject.to_row(),
            "polarity": self.polarity,
        }
        if self.object is not None:
            row["object"] = self.object.to_row()
        if self.retracts:
            row["retracts"] = self.retracts
        if self.qualifiers:
            row["qualifiers"] = dict(self.qualifiers)
        row["confidence"] = self.confidence
        for k in ("method", "source", "actor", "rationale", "timestamp"):
            v = getattr(self, k)
            if v:
                row[k] = v
        return row

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Assertion:
        obj = row.get("object")
        return cls(
            predicate=str(row["predicate"]),
            subject=NodeRef.from_row(row["subject"]),
            object=NodeRef.from_row(obj) if obj is not None else None,
            polarity=str(row.get("polarity", "affirm")),
            retracts=row.get("retracts"),
            qualifiers=dict(row.get("qualifiers", {})),
            confidence=row.get("confidence", "medium"),
            method=str(row.get("method", "")),
            source=str(row.get("source", "")),
            actor=str(row.get("actor", "")),
            rationale=str(row.get("rationale", "")),
            timestamp=str(row.get("timestamp", "")),
            id=str(row.get("id", "")),
        )


# --- id minting (deterministic, D36.9) ------------------------------------


def mint_assertion_id(a: Assertion) -> str:
    """A stable content hash of the WHOLE record (claim + provenance).

    Every identifying and provenance field is hashed via canonical JSON
    (``sort_keys`` so qualifier / field order is irrelevant), EXCEPT ``id``
    itself. Consequences:

    - Re-authoring a byte-identical record yields the same id, so an authoring
      pass is idempotent and a rebuild is deterministic (D36.9).
    - Any difference — confidence, method, source, actor, timestamp, rationale,
      qualifiers — yields a distinct id, so every authored witness is preserved
      on the append-only log (no silent collision drops a row).
    - Re-affirming a *retracted* claim is therefore a NEW record (distinct
      timestamp / actor / rationale → distinct id → live again); the original
      withdrawn id stays withdrawn.

    The JSON encoding is unambiguous (no separator-collision risk from a space
    appearing inside a rationale or ref).
    """
    payload = {
        "predicate": a.predicate,
        "subject": a.subject.to_row(),
        "object": a.object.to_row() if a.object is not None else None,
        "polarity": a.polarity,
        "retracts": a.retracts,
        "qualifiers": a.qualifiers,
        "confidence": a.confidence,
        "method": a.method,
        "source": a.source,
        "actor": a.actor,
        "rationale": a.rationale,
        "timestamp": a.timestamp,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    digest = hashlib.md5(blob.encode("utf-8"), usedforsecurity=False).hexdigest()
    return f"a-{digest[:16]}"


_CANONICAL_PREFIX = {
    "canonical_morpheme": "CM",
    "canonical_place": "CP",
    "canonical_sense": "CS",
}


def mint_canonical_id(node_type: str, *parts: str) -> str:
    """Mint a stable synthetic canonical-node id from content ``parts``.

    Ids are L2-declared (carried by a ``mint-canonical`` record), so the node's
    identity survives canonical-label drift across strata (D50.1). The prefix
    (CM / CP / CS) keeps the node type legible in the log.
    """
    if node_type not in CANONICAL_NODE_TYPES:
        raise AssertionValidationError(
            f"unknown canonical node type {node_type!r}; "
            f"expected one of {sorted(CANONICAL_NODE_TYPES)}"
        )
    prefix = _CANONICAL_PREFIX[node_type]
    content = "\x1f".join(parts)  # unit separator — unambiguous across parts
    digest = hashlib.md5(content.encode("utf-8"), usedforsecurity=False).hexdigest()
    return f"{prefix}-{digest[:12]}"


# --- validation -----------------------------------------------------------


def _check_types(label: str, node: NodeRef, allowed: frozenset[str] | None) -> None:
    if allowed is not None and node.type not in allowed:
        raise AssertionValidationError(
            f"{label} type {node.type!r} not allowed; expected one of {sorted(allowed)}"
        )


def _check_arity(a: Assertion, spec: PredicateSpec) -> None:
    _check_types("subject", a.subject, spec.subject_types)
    if spec.is_unary:
        if a.object is not None:
            raise AssertionValidationError(f"{a.predicate!r} is unary; got an object {a.object!r}")
        return
    if a.object is None:
        raise AssertionValidationError(f"{a.predicate!r} requires an object")
    _check_types("object", a.object, spec.object_types)
    # No binary predicate has a meaningful self-loop (merge-canonical of a node
    # with itself, an etymon descending from / same-as itself, etc.), so reject
    # subject == object outright — a degenerate edge the L3 projection would
    # otherwise have to special-case.
    if a.subject == a.object:
        raise AssertionValidationError(
            f"{a.predicate!r} subject and object must differ (no self-loop): {a.subject!r}"
        )


def _check_qualifiers(a: Assertion, spec: PredicateSpec) -> None:
    keys = set(a.qualifiers)
    missing = spec.required - keys
    if missing:
        raise AssertionValidationError(
            f"{a.predicate!r} missing required qualifier(s) {sorted(missing)}"
        )
    unknown = keys - (spec.required | spec.optional)
    if unknown:
        raise AssertionValidationError(
            f"{a.predicate!r} has unknown qualifier(s) {sorted(unknown)}; "
            f"allowed: {sorted(spec.required | spec.optional)}"
        )
    if "kind" in keys and a.qualifiers["kind"] not in BIND_KINDS:
        raise AssertionValidationError(
            f"bind kind {a.qualifiers['kind']!r} not in {sorted(BIND_KINDS)}"
        )
    if "edge_type" in keys and a.qualifiers["edge_type"] not in EDGE_TYPES:
        raise AssertionValidationError(
            f"edge_type {a.qualifiers['edge_type']!r} not in {sorted(EDGE_TYPES)}"
        )
    if "ordinal" in keys:
        ordinal = a.qualifiers["ordinal"]
        if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 0:
            raise AssertionValidationError(f"ordinal must be a non-negative int, got {ordinal!r}")


def _check_polarity_and_confidence(a: Assertion) -> None:
    if a.polarity not in POLARITIES:
        raise AssertionValidationError(f"polarity {a.polarity!r} not in {sorted(POLARITIES)}")
    if a.polarity == "retract" and not a.retracts:
        raise AssertionValidationError("retract polarity requires a 'retracts' id")
    if a.polarity != "retract" and a.retracts:
        raise AssertionValidationError("'retracts' is only valid when polarity == 'retract'")
    conf = a.confidence
    if isinstance(conf, bool) or not (
        (isinstance(conf, str) and conf in CONFIDENCE_LABELS)
        or (isinstance(conf, (int, float)) and 0.0 <= float(conf) <= 1.0)
    ):
        raise AssertionValidationError(
            f"confidence must be one of {sorted(CONFIDENCE_LABELS)} or a float in "
            f"[0, 1]; got {conf!r}"
        )


def validate(a: Assertion) -> None:
    """Raise :class:`AssertionValidationError` if ``a`` violates its spec."""
    spec = PREDICATES.get(a.predicate)
    if spec is None:
        raise AssertionValidationError(
            f"unknown predicate {a.predicate!r}; expected one of {sorted(PREDICATES)}"
        )
    _check_arity(a, spec)
    _check_qualifiers(a, spec)
    _check_polarity_and_confidence(a)
