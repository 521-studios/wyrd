"""Per-predicate JSONL streams for canonicalization assertions (D50.5).

Assertions live under ``data/mining/canonicalization/`` as per-predicate
append-only JSONL, mirroring the existing L2 sidecars (``_reflexes.jsonl``,
``_collapses.jsonl``, …) so diffs stay scoped and the
``rebuild-runbook-currency-reviewer`` can track each as a discrete rebuild
input:

- ``_canonical_nodes.jsonl`` — the ``mint-canonical`` declarations (the stable
  synthetic-node ids).
- ``_assert_<predicate>.jsonl`` — one file per other predicate
  (``_assert_bind.jsonl``, ``_assert_decomposes_into.jsonl``, …).

This module reuses the low-level line I/O from :mod:`wyrd.generators.kenning.jsonl.log`
(``read_jsonl``) and appends with the same ``json.dumps`` idiom. The
``_op``/``ref`` replay kernel in that module is for source-mining entity
updates; assertions use their own polarity resolution (:func:`effective_assertions`)
instead. The L3 projection that folds the effective set into a collapse graph
is the separate ``u6fn.3`` work.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from ..jsonl.log import read_jsonl
from .assertions import Assertion, AssertionValidationError, validate

CANONICALIZATION_DIRNAME = "canonicalization"
_NODES_FILE = "_canonical_nodes.jsonl"


def stream_filename(predicate: str) -> str:
    """The JSONL filename a given predicate's assertions live in."""
    if predicate == "mint-canonical":
        return _NODES_FILE
    return f"_assert_{predicate.replace('-', '_')}.jsonl"


def stream_path(base_dir: str | Path, predicate: str) -> Path:
    """Resolve the stream path for ``predicate`` under ``base_dir``.

    ``base_dir`` is the mining root (e.g. ``data/mining``); the
    ``canonicalization/`` subdirectory is created on first append.
    """
    return Path(base_dir) / CANONICALIZATION_DIRNAME / stream_filename(predicate)


def _append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, sort_keys=False))
        fh.write("\n")


def append_assertion(base_dir: str | Path, assertion: Assertion) -> Assertion:
    """Validate and append one assertion to its predicate's stream.

    Returns the assertion with its minted ``id`` populated. Append-only: a
    ``refute`` / ``retract`` is a new line, never an edit (D21/D22).
    """
    validate(assertion)
    stamped = assertion.with_id()
    _append_row(stream_path(base_dir, stamped.predicate), stamped.to_row())
    return stamped


def load_assertions(base_dir: str | Path) -> Iterator[Assertion]:
    """Yield every validated assertion across all canonicalization streams.

    Files are read in sorted filename order; lines within a file in file order.
    Each row is parsed, validated, and yielded — an invalid row raises (the
    stream is the source of truth and must stay clean).
    """
    cdir = Path(base_dir) / CANONICALIZATION_DIRNAME
    if not cdir.is_dir():
        return
    for path in sorted(cdir.glob("*.jsonl")):
        for line_no, row in read_jsonl(path):
            if row.get("_type") not in (None, "canonical_assertion"):
                continue
            try:
                assertion = Assertion.from_row(row)
                validate(assertion)
            except (AssertionValidationError, KeyError, TypeError) as exc:
                # A malformed parse (missing key → KeyError in from_row) or a
                # spec violation surfaces as ONE located error — the stream is
                # the source of truth and the offending line must be findable.
                raise AssertionValidationError(
                    f"{path.name}:{line_no}: invalid assertion row: {exc}"
                ) from exc
            yield assertion


def effective_assertions(assertions: Iterable[Assertion]) -> list[Assertion]:
    """Resolve the live set: drop assertions a later ``retract`` withdraws.

    The L2-level replay for the assertion log (analogous to the kernel's
    ``ReplayState`` for ``_op`` streams, but for polarity): an affirm/refute is
    live until a ``retract`` naming its id appears. ``retract`` rows are
    themselves dropped from the result — they exist only to withdraw. Order is
    preserved for the survivors. Projecting the live set into the L3 collapse
    graph (applying binds, the leave-separate confidence gate, conflict
    handling) is ``u6fn.3``; this function only honours retraction.
    """
    retracted: set[str] = {a.retracts for a in assertions if a.polarity == "retract" and a.retracts}
    return [a for a in assertions if a.polarity != "retract" and (a.id or "") not in retracted]
