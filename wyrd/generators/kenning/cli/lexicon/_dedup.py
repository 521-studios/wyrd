"""Shared dedup-key helpers for the staged-miner + recovery CLI pair.

The staged miner (``mine_toponym_mentions_staged._run_resume_from_failures``)
and the recovery command (``recover_inprogress_chunks``) both build a
union of existing-canonical + new-mention rows, deduped on the same
key tuple. Originally the key was inlined at both sites with a comment
asking the reader to keep them in sync; this module is the load-bearing
seam that makes the invariant a code-level guarantee instead of a
documentation guarantee."""

from __future__ import annotations


def dedup_key(row: dict) -> tuple[str, int | None, str | None, str]:
    """Identity tuple for a mention row.

    Two rows with the same return value are considered duplicates and
    the second is dropped. Both stage callers (resume mining, recover)
    MUST use this function — divergence here would let a recover pass
    re-insert mentions a prior resume already wrote (or vice versa)."""
    return (
        row.get("form", ""),
        row.get("date_year"),
        row.get("region_hint"),
        row.get("context", ""),
    )
