"""Shared dedup-key helpers for the toponym-mention mining + recovery
CLI pair.

The dict-form callers (recover_inprogress_chunks, mine_toponym_mentions_
tiered._load_existing_mention_keys) and the dataclass-form caller
(mine_toponym_mentions_staged._run_resume_from_failures, working on
ToponymMention instances) both build a union of existing-canonical +
new-mention rows, deduped on the same logical key tuple. This module
is the seam that prevents the field set from drifting across the three
sites — adding a new identity field (say, ``source_doc``) now means
updating both helpers below, not three inlined tuples."""

from __future__ import annotations

from typing import Any


def dedup_key(row: dict) -> tuple[str, int | None, str | None, str]:
    """Identity tuple for a mention dict.

    Two rows whose dicts hash to the same return value are considered
    duplicates and the second is dropped. The dict-form callers
    (recover_inprogress_chunks, mine_toponym_mentions_tiered's
    _load_existing_mention_keys) use this directly; the dataclass-form
    caller in mine_toponym_mentions_staged uses :func:`dedup_key_from_mention`
    which produces the same shape."""
    return (
        row.get("form", ""),
        row.get("date_year"),
        row.get("region_hint"),
        row.get("context", ""),
    )


def dedup_key_from_mention(m: Any) -> tuple[str, int | None, str | None, str]:
    """Identity tuple for a ToponymMention dataclass instance — same
    shape as :func:`dedup_key`. Kept here (not on the dataclass) so the
    seam is co-located: changing the identity field set means updating
    both functions in one file.

    ``m`` is typed ``Any`` rather than ``ToponymMention`` to keep this
    module a leaf-level helper — importing the dataclass would couple
    the dedup seam to the extractors package."""
    return (m.form, m.date_year, m.region_hint, m.context)
