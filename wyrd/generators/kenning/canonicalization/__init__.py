"""Canonicalization assertion layer (Class-B entity resolution, D49 + D50).

The append-only L2 assertion log that drives identity resolution, relational
edges, and canonicalization choices over the kenning corpus. This package owns
the *L2* half (the record model + per-predicate JSONL streams); the *L3*
projection that folds assertions into a collapse graph is the separate
``u6fn.3`` work.

- ``assertions`` — the uniform record model, the predicate catalog (D50.3
  taxonomy, enforced), validation, and deterministic id minting.
- ``streams`` — the per-predicate JSONL stream layout under
  ``data/mining/canonicalization/`` plus append / load / retraction-resolution.

See ``wyrd/generators/kenning/DECISIONS.md`` D49 (the two-class partition) and
D50 (the assertion-layer mechanism: synthetic canonical nodes, the edge
taxonomy, the record schema).
"""

from __future__ import annotations

from .assertions import (
    BIND_KINDS,
    CANONICAL_NODE_TYPES,
    EDGE_TYPES,
    FAMILIES,
    POLARITIES,
    PREDICATES,
    Assertion,
    AssertionValidationError,
    NodeRef,
    PredicateSpec,
    mint_canonical_id,
    validate,
)
from .streams import (
    append_assertion,
    effective_assertions,
    load_assertions,
    stream_filename,
)

__all__ = [
    "Assertion",
    "AssertionValidationError",
    "BIND_KINDS",
    "CANONICAL_NODE_TYPES",
    "EDGE_TYPES",
    "FAMILIES",
    "NodeRef",
    "POLARITIES",
    "PREDICATES",
    "PredicateSpec",
    "append_assertion",
    "effective_assertions",
    "load_assertions",
    "mint_canonical_id",
    "stream_filename",
    "validate",
]
