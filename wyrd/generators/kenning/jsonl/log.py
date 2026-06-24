"""JSONL event-log + canonical-state replay kernel — wyrd-f295.

Per-source ``data/mining/<source>.jsonl`` files are the source of truth
for what each external source contributed to the lexicon. Two row shapes
coexist in the same file::

  # Canonical-state row (no _op) — "compacted" form.
  {"_type": "etymon", "ref": "old-english:cot", ...}

  # Event row (has _op) — append during active curation / parallel mining.
  {"_op": "add", "_type": "etymon", "ref": "old-english:cot", ...}
  {"_op": "patch", "_type": "etymon", "ref": "old-english:cot", "glosses_add": ["hut"]}
  {"_op": "remove", "_type": "etymon", "ref": "old-english:cot"}

This module provides the pure-function kernel that folds a sequence of
rows into per-``(_type, ref)`` final state. No DB, no I/O — callers
read JSONL lines, hand them to :func:`replay`, and walk the result.

``_op`` vocabulary
==================

``add``
    Strict insert. ``ReplayError`` if ``ref`` already exists. Used by
    mining ingesters writing new rows for the first time.

``set``
    Idempotent overwrite. Replaces whatever's there. Used by compaction
    + ingesters that "own" a source.

``patch``
    Partial update. Scalar fields = last-write-wins. Array fields use
    ``<field>_add`` / ``<field>_remove`` suffixes for set-union / set-diff
    semantics. Patching a missing ``ref`` raises ``ReplayError``.

``remove``
    Delete by ref. Removing a missing ref is a no-op (idempotent).

Identity
========

A row is identified by ``(_type, ref)``. File-order is canonical — no
timestamps, no clocks. The reader processes rows top-to-bottom, applying
each event to the running state for that ``(_type, ref)`` pair.

Some row types are immutable facts that legitimately repeat per
(source, document, page) without a stable string identity (citations,
attestations, etymology elements, descent edges, mining_run audit rows).
Those rows live in the file without a ``ref`` field and are accumulated
as a list under their ``_type`` — no event semantics, just append. See
:data:`LIST_TYPES`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

# Row types that own an immutable ``ref`` identity and support
# add/patch/set/remove semantics. The kernel keeps a ``dict[ref, state]``
# for each of these.
#
# ``etymon_curation`` rows (wyrd-2jhs) live in ``data/mining/_curation.jsonl``
# and carry operator-applied overrides to L3 enrichment output. The ``ref``
# field is the etymon being curated (e.g. ``"old-english:caelf"``); payload
# fields ``lemma_ref`` / ``merged_into_ref`` / ``inflection`` override the
# auto-clustering passes' lemma_id / merged_into_id / inflection columns.
# See L2_L3_BOUNDARY.md for the apply-after-enrichment workflow.
#
# ``etymon_gloss_suppression`` rows (wyrd-kutx) drop named glosses
# from an etymon. ``ref`` is the etymon ref; payload carries a
# ``suppressions`` array of ``{gloss, reason?}`` entries. Per-event
# semantics are last-write-wins (KEYED_TYPES contract): re-emit the
# FULL suppressions list to revise, don't append a sibling event for
# the same ref. Use case: Wiktionary-mined etymons whose entry
# conflates two senses, only one of which is used in place-names
# (e.g. OE ``dry`` has "dry" + "wizard, sorcerer"; the latter has
# zero toponymy use). Tags are NOT touched by this event — they
# don't carry per-gloss provenance in the current schema.
#
# ``etymon_split`` rows (wyrd-kutx) split a conflated etymon into two
# or more distinct etymons. ``ref`` is the original etymon; payload's
# ``into`` array names the children (each ``{suffix, glosses, tags?,
# primary?, reason?}``). Children land as new etymons whose canonical
# form is ``"<original-form>#<suffix>"``; the resulting ref is
# ``"<language>:<original-form>#<suffix>"`` (e.g. ``old-english:gear``
# splits into ``old-english:gear#weir`` and ``old-english:gear#year``).
# The named glosses + tags move from the parent to each child; mining
# evidence (citations, descent edges, toponym_etymology_element refs)
# moves WHOLESALE to the ``primary=true`` child so the primary inherits
# the parent's witness count for bundle promotion. Per D21 evidence is
# preserved (moved, not destroyed). Per-citation reassignment to non-
# primary children is a deferred follow-on op.
#
# ``etymon_gloss_add`` rows (wyrd-wz82) add operator-curated glosses to
# an etymon — the inverse of ``etymon_gloss_suppression``. ``ref`` is
# the etymon ref; payload carries an ``additions`` array of
# ``{gloss, reason?}`` entries. Use case: surfacing a sense the
# auto-mining missed — e.g. OE ``finn`` carries "clear, transparent,
# bright" + "fin" (with the latter suppressed in wyrd-kutx); but
# Roberts 1914 Sussex cites it as a personal name ("Finn"), a third
# sense the bundle has no gloss for. Add-gloss surfaces it without
# requiring a full lexicon re-mine. Last-write-wins per ref (same
# kernel contract as the other curation event types).
#
# ``collapse`` rows (wyrd-y651, the consolidation epic) fold a form-of /
# variant etymon into its lemma — the inverse of ``etymon_split``.
# ``ref`` is the etymon being absorbed; payload's ``into`` names the
# surviving lemma (``"<language>:<canonical_form>"``) plus ``variant_class``
# / ``method`` / ``reason``. Replayed by ``enrichment.apply_collapses`` in
# the curation slot: tombstones ``ref`` into ``into`` (merged_into_id),
# records the form as a variant of the lemma, and migrates reflexes +
# citations (stamping ``attested_form``) to the lemma. Single ledger for
# all consolidations — the deterministic detector AND the LLM merge pass
# append here, differing only in ``method``. Last-write-wins per ref;
# ``into: ""`` reverts a recorded collapse.
KEYED_TYPES: frozenset[str] = frozenset(
    {
        "etymon",
        "etymon_curation",
        "etymon_gloss_suppression",
        "etymon_gloss_add",
        "etymon_split",
        "collapse",
        "toponym",
        "source",
        # wyrd-2b50: secondary sources a file cites ON BEHALF OF. ``ref``
        # is the source_id (e.g. ``"pase"``, ``"dlv"``,
        # ``"anglo_saxon_charters"``); payload carries title/notes like a
        # ``source`` row. Distinct from ``source`` (of which each file has
        # exactly one — its PRIMARY) so a single L2 file can register the
        # other scholarly references it attests without fabricating a
        # separate JSONL per source. Build pass 2 upserts each into the
        # ``source`` table; ``citation`` rows then target them via an
        # explicit ``source_id``. This is the general "a source adds
        # citations for sources it has consumed" path — Briggs is the
        # first consumer (it indexes PASE / DLV / Anglo-Saxon-charter
        # attestations we have never ingested directly).
        "cited_source",
    }
)

# Row types accumulated as an ordered list under their ``_type`` —
# repeated facts (multiple citations of the same etymon, multiple
# attestations of the same toponym, repeat descent edges). No event
# semantics; rows are appended in file order.
LIST_TYPES: frozenset[str] = frozenset(
    {
        "citation",
        "attestation",
        "etymology_element",
        "etymon_descent",
        "mining_run",
        "fantasy_morpheme",
        # wyrd-ned5: seed reflex layer (modern_usage → etymon links).
        # Carried in the synthetic ``_reflexes.jsonl`` file so the
        # canonical morpheme→etymon mappings (e.g. -ton → OE tūn) survive
        # a full rebuild-from-jsonl — they're created only by
        # seed_from_meanings and are NOT otherwise re-derivable from L2.
        "reflex",
        # wyrd-6hbv: merge-audit verdict log (``data/mining/_merge_audit.jsonl``).
        # Idempotency + audit trail only — the applied reverts live in the
        # collapse / curation ledgers; no build helper consumes this list, so it
        # replays inert. Registered here so rebuild-from-jsonl's glob accepts the
        # file instead of raising ReplayError — _merge_audit conforms (every row is
        # a registered _type) and is NOT in build.REPLAY_EXCLUDED_LEDGERS. The other
        # audit logs each contain at least one row with an unregistered/absent _type
        # and ARE excluded from the glob there (wyrd-5qg7).
        "merge_audit",
        # wyrd-xz3g: LLM tag-backfill decisions (``data/mining/_tags.jsonl``).
        # Same shape as merge_audit — no build helper consumes the replay
        # accumulation (``collect_tags`` reads the file directly for
        # ``apply_tag_additions``); registered so rebuild-from-jsonl's glob
        # accepts the file instead of raising ReplayError. The file also
        # carries the mandatory canonical ``source`` row (ref ``tag-backfill``)
        # like every other synthetic L2 file.
        "tags",
    }
)

ALL_TYPES: frozenset[str] = KEYED_TYPES | LIST_TYPES

VALID_OPS: frozenset[str] = frozenset({"add", "set", "patch", "remove"})


class ReplayError(ValueError):
    """Raised when a row fails replay (unknown _type, duplicate add,
    patch-missing-ref, unknown _op, malformed row). Carries the
    offending row in :attr:`row` and the 1-indexed file line number in
    :attr:`line_no` when known."""

    def __init__(
        self, message: str, *, row: dict[str, Any] | None = None, line_no: int | None = None
    ):
        prefix = f"line {line_no}: " if line_no is not None else ""
        super().__init__(prefix + message)
        self.row = row
        self.line_no = line_no


class ReplayState:
    """In-memory result of folding a JSONL stream.

    - ``keyed[_type][ref] = dict`` for :data:`KEYED_TYPES`. Each value
      is the current canonical state for that ``(_type, ref)``.
    - ``lists[_type] = list[dict]`` for :data:`LIST_TYPES`. Each entry
      is one fact-row in file order.
    """

    __slots__ = ("keyed", "lists")

    def __init__(self) -> None:
        self.keyed: dict[str, dict[str, dict[str, Any]]] = {t: {} for t in KEYED_TYPES}
        self.lists: dict[str, list[dict[str, Any]]] = {t: [] for t in LIST_TYPES}

    def get(self, _type: str, ref: str) -> dict[str, Any] | None:
        if _type not in KEYED_TYPES:
            raise ValueError(f"{_type!r} is not a keyed type; cannot get(ref=...)")
        return self.keyed[_type].get(ref)

    def to_canonical_rows(self) -> list[dict[str, Any]]:
        """Emit current state as canonical-state rows (no ``_op``).

        Order: every keyed type in ``KEYED_TYPES`` declaration order,
        sorted by ref; then every list type in ``LIST_TYPES`` order,
        preserving insertion order. This is the shape :func:`compact_file`
        writes back to disk.
        """
        out: list[dict[str, Any]] = []
        for _type in (
            "etymon",
            "etymon_curation",
            "etymon_gloss_suppression",
            "etymon_gloss_add",
            "etymon_split",
            "toponym",
            "source",
            "cited_source",
        ):
            entries = self.keyed.get(_type, {})
            for ref in sorted(entries):
                row = {"_type": _type, "ref": ref}
                row.update(entries[ref])
                out.append(row)
        for _type in (
            "citation",
            "attestation",
            "etymology_element",
            "etymon_descent",
            "mining_run",
            "fantasy_morpheme",
        ):
            for fact in self.lists.get(_type, []):
                row = {"_type": _type}
                row.update(fact)
                out.append(row)
        return out


def _patch_field(target: dict[str, Any], key: str, value: Any) -> None:
    """Apply one field of a patch event to ``target``.

    Scalar fields (no ``_add`` / ``_remove`` suffix) are last-write-wins.
    Array fields use suffixed forms::

        glosses_add = ["hut"]        → union into target["glosses"]
        glosses_remove = ["wrong"]   → set-difference from target["glosses"]
        tags_set = ["arch", "ME"]    → replace target["tags"] outright

    ``_add`` / ``_remove`` on a missing field implicitly initialize it
    to an empty list.
    """
    if key.endswith("_add"):
        field = key[:-4]
        existing = list(target.get(field, []))
        seen = set(existing)
        for item in value:
            if item not in seen:
                existing.append(item)
                seen.add(item)
        target[field] = existing
    elif key.endswith("_remove"):
        field = key[:-7]
        existing = list(target.get(field, []))
        drop = set(value)
        target[field] = [x for x in existing if x not in drop]
    elif key.endswith("_set"):
        field = key[:-4]
        target[field] = list(value)
    else:
        target[key] = value


def _validate_row(row: dict[str, Any], *, line_no: int | None = None) -> None:
    """Common-shape check. Raises :class:`ReplayError` on violation."""
    if not isinstance(row, dict):
        raise ReplayError("row is not a JSON object", row=row, line_no=line_no)
    _type = row.get("_type")
    if _type is None:
        raise ReplayError("row missing '_type' field", row=row, line_no=line_no)
    if _type not in ALL_TYPES:
        raise ReplayError(
            f"unknown _type {_type!r}; expected one of {sorted(ALL_TYPES)}",
            row=row,
            line_no=line_no,
        )
    op = row.get("_op")
    if op is not None and op not in VALID_OPS:
        raise ReplayError(
            f"unknown _op {op!r}; expected one of {sorted(VALID_OPS)}",
            row=row,
            line_no=line_no,
        )
    if _type in KEYED_TYPES:
        ref = row.get("ref")
        if ref is None or not isinstance(ref, str) or not ref:
            raise ReplayError(
                f"{_type!r} row missing non-empty 'ref' field",
                row=row,
                line_no=line_no,
            )


def _apply_row(state: ReplayState, row: dict[str, Any], *, line_no: int | None = None) -> None:
    _validate_row(row, line_no=line_no)
    _type = row["_type"]
    op = row.get("_op")
    if _type in LIST_TYPES:
        _apply_list_row(state, _type, op, row, line_no)
    else:
        _apply_keyed_row(state, _type, op, row, line_no)


def _apply_list_row(
    state: ReplayState, _type: str, op: str | None, row: dict[str, Any], line_no: int | None
) -> None:
    """Append-only list fact. List types take no ``_op`` (or ``_op=add``)."""
    if op is not None and op != "add":
        raise ReplayError(
            f"list-type {_type!r} does not support _op={op!r}; only append (no _op or _op=add)",
            row=row,
            line_no=line_no,
        )
    fact = {k: v for k, v in row.items() if k not in ("_type", "_op")}
    state.lists[_type].append(fact)


def _apply_keyed_row(
    state: ReplayState, _type: str, op: str | None, row: dict[str, Any], line_no: int | None
) -> None:
    """Keyed fact dispatched by ``_op``: set / add (no-clobber) / patch (per-field
    merge) / remove (idempotent) — all keyed on ``row["ref"]``."""
    ref = row["ref"]
    bucket = state.keyed[_type]
    payload = {k: v for k, v in row.items() if k not in ("_type", "_op", "ref")}

    if op is None or op == "set":
        bucket[ref] = payload
        return
    if op == "add":
        if ref in bucket:
            raise ReplayError(
                f"duplicate add: {_type}/{ref} already exists; use _op=set or _op=patch",
                row=row,
                line_no=line_no,
            )
        bucket[ref] = payload
        return
    if op == "patch":
        if ref not in bucket:
            raise ReplayError(
                f"patch on missing {_type}/{ref}; use _op=add or _op=set first",
                row=row,
                line_no=line_no,
            )
        target = bucket[ref]
        for k, v in payload.items():
            _patch_field(target, k, v)
        return
    if op == "remove":
        bucket.pop(ref, None)
        return
    raise AssertionError(f"unreachable _op={op!r}")  # _validate_row guards this


def replay(rows: Iterable[dict[str, Any]]) -> ReplayState:  # noqa: V103 — public kernel primitive; tests pin replay semantics (PR #498 triage)
    """Fold ``rows`` (in order) into a :class:`ReplayState`.

    Pure function. Raises :class:`ReplayError` on the first bad row,
    annotated with a 1-indexed line number if ``rows`` is an
    ``enumerate``-able sequence — but we don't auto-enumerate here;
    callers reading from disk should pass enumerated rows via
    :func:`read_jsonl` which sets ``line_no`` on errors.
    """
    state = ReplayState()
    for row in rows:
        _apply_row(state, row)
    return state


def read_jsonl(path: str | Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield ``(line_no, row)`` pairs from a JSONL file.

    Blank lines + ``#``-prefixed comment lines are skipped. ``line_no``
    is 1-indexed against the raw file (useful for error reporting).
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ReplayError(f"invalid JSON: {exc}", line_no=line_no) from exc
            yield line_no, row


def replay_file(path: str | Path) -> ReplayState:
    """Stream a JSONL file through :func:`replay` with line-number error annotations."""
    state = ReplayState()
    for line_no, row in read_jsonl(path):
        _apply_row(state, row, line_no=line_no)
    return state


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    """Write ``rows`` to ``path`` as JSONL. Returns the number of rows
    written. One JSON object per line, trailing newline, ``ensure_ascii=False``
    so non-ASCII glosses (þ, æ, ī) stay readable in diffs.

    The parent directory is created if it doesn't exist.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with p.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=False))
            fh.write("\n")
            count += 1
    return count


def compact_file(path: str | Path) -> int:
    """Compact a JSONL file in place. Returns the number of rows written."""
    state = replay_file(path)
    canonical = state.to_canonical_rows()
    return write_jsonl(path, canonical)
