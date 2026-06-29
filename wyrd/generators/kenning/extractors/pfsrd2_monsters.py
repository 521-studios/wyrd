"""pfsrd2-data → wyrd-ami JSONL extractor (wyrd-ld5x).

The wyrd-ami fantasy-name pipeline takes JSONL input where each line
is `{"name": ..., "description": ...}`. This module walks the
pfsrd2-data monster JSON corpus (`~/521Studios/pfsrd2-data/monsters/`)
and produces that JSONL — one entry per distinct creature morpheme.

Extraction strategy (verified against creature.schema.1.4.json and
spot-checked across bugbear_thug / harpy / dragons / trolls / genie):

- **Both family root AND single-word variant names are emitted.**
  A "Djinni" creature with `family = "Genie"` produces two records:
  `Genie` (the family root, deduped across all 34 genie-family files)
  and `Djinni` (an etymologically distinct Arabic morpheme that's
  worth researching independently). Earlier versions of this extractor
  collapsed all variants to the family root, which silently lost
  morphemes like Djinni / Efreeti / Marid / Ifrit / Balor / Cornugon /
  Bythos that share a family but are distinct words. Per user
  2026-05-06.

- **Family root preference**: A "Bugbear Thug" entry has
  `stat_block.creature_type.family.name = "Bugbear"`; we emit
  "Bugbear" (the morpheme) rather than the multi-word variant name.
  Compound family names like "Dragon, Red" are split on `,` to get
  the base ("Dragon").

- **Variant name emission rule**: emit the monster's own `name` as a
  separate record IFF (a) it is single-word AND (b) it differs from
  the family root (case-insensitive). "Bugbear Thug" stays dropped
  (multi-word). "Bugbear" under family "Bugbear" doesn't double-emit.

- **Single-word top-level fallback**: a creature like Harpy has no
  family (`family is None`). Its `name = "Harpy"` is itself the
  morpheme.

- **Multi-word no-family creatures are skipped**: per user 2026-05-06,
  "Ancient Black Dragon" with no family field is problematic — there's
  no clean morpheme to extract. We bail rather than guess.

- **Description differs by record kind**:
  - Family-root record: `family.text` + recursive `family.sections`
    only. (Family-level lore — no per-variant content.)
  - Variant record: same as v1 — `family.text` + family.sections +
    monster.sections. The family.text gives the LLM context ("Djinni
    is a kind of Genie"); monster.sections gives variant-specific
    flavor.
  - No-family record: `monster.sections` only.

- **Dedupe by lowercased name across the walk**. First occurrence
  wins, so the family-root record's description comes from whichever
  variant file is encountered first (sorted order makes this
  deterministic).

The extractor doesn't HTTP-fetch — it's a pure JSON walker. Run it
out-of-band, pipe the output JSONL into `wyrd kenning lexicon
mine-fantasy-name --batch`.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def _collect_section_texts(sections: list[dict[str, Any]] | None) -> list[str]:
    """Walk the recursive sections tree, collect every node's `text`
    field in document order. Each node may have its own children
    under `sections`."""
    out: list[str] = []
    for s in sections or []:
        text = s.get("text")
        if text:
            out.append(text)
        out.extend(_collect_section_texts(s.get("sections")))
    return out


def _family_root(family: dict[str, Any] | None) -> str | None:
    """Return the single-word family root for the given `family` dict
    (a `stat_block.creature_type.family` node), or None if the dict is
    missing/empty or its base (after comma-split) is multi-word.

    Takes the already-fetched family dict rather than re-navigating
    from the monster — callers usually need both `family` and
    `family_root` together (e.g. to pass `family` to a description
    helper), so we save a redundant lookup at every call site.
    """
    if not (family and family.get("name")):
        return None
    candidate = family["name"].split(",")[0].strip()
    if not candidate or " " in candidate:
        return None
    return candidate


def _get_family(monster: dict[str, Any]) -> dict[str, Any] | None:
    """Safely read `monster.stat_block.creature_type.family`, defaulting
    to None if any segment is missing."""
    return monster.get("stat_block", {}).get("creature_type", {}).get("family")


def _monster_single_word_name(monster: dict[str, Any]) -> str | None:
    """Return the monster's own `name` if it's single-word, else None."""
    name = (monster.get("name") or "").strip()
    if not name or " " in name:
        return None
    return name


def _family_only_description(family: dict[str, Any]) -> str:
    """Description for a family-root record: family.text + recursive
    family.sections. No monster-specific content (would be arbitrary
    since the family root represents the whole family)."""
    parts: list[str] = []
    family_text = family.get("text")
    if family_text:
        parts.append(family_text)
    parts.extend(_collect_section_texts(family.get("sections")))
    return "\n\n".join(p for p in parts if p)


def _variant_description(family: dict[str, Any] | None, monster: dict[str, Any]) -> str:
    """Description for a variant or no-family record: family.text +
    family.sections (as context, when family is truthy) plus
    monster.sections (variant-specific flavor). When `family` is None
    or empty, degrades cleanly to monster.sections only — so this
    helper covers both the "real family, single-word variant" case
    and the "no family at all" case."""
    parts: list[str] = []
    if family:
        family_text = family.get("text")
        if family_text:
            parts.append(family_text)
        parts.extend(_collect_section_texts(family.get("sections")))
    parts.extend(_collect_section_texts(monster.get("sections")))
    return "\n\n".join(p for p in parts if p)


def extract_records(monster: dict[str, Any]) -> Iterator[dict[str, str]]:
    """Yield up to two records for one monster file:

    1. The family root, with a family-only description (when the
       family root is single-word).
    2. The monster's own name, with a variant-specific description
       (when the name is single-word and != family root).

    When the monster has no family and a single-word name, yield one
    no-family record. When neither axis produces a clean morpheme,
    yields nothing.
    """
    family = _get_family(monster)
    family_root = _family_root(family)
    monster_name = _monster_single_word_name(monster)

    if family_root is not None:
        yield {
            "name": family_root,
            "description": _family_only_description(family),
        }
        if monster_name is not None and monster_name.lower() != family_root.lower():
            yield {
                "name": monster_name,
                "description": _variant_description(family, monster),
            }
    elif monster_name is not None:
        # No usable family root, but monster name is single-word. This
        # also catches the "family exists but is multi-word (e.g.
        # 'Blightburn Genies') AND monster name is single-word" edge
        # case — emit the variant on its own. _variant_description
        # handles both branches (family present → include family.text
        # as context; family None → just monster.sections).
        yield {"name": monster_name, "description": _variant_description(family, monster)}


def iter_monster_files(corpus_root: Path) -> Iterator[Path]:
    """Yield each monster JSON file under `corpus_root/monsters/` in
    sorted order. Sorted so the dedupe (first-occurrence-wins) is
    deterministic across runs."""
    monsters_dir = corpus_root / "monsters"
    if not monsters_dir.is_dir():
        return
    yield from sorted(monsters_dir.rglob("*.json"))


def extract_corpus(
    corpus_root: Path,
    *,
    limit: int | None = None,
) -> Iterator[dict[str, str]]:
    """Walk pfsrd2-data, dedupe by lowercased name across all records
    yielded by `extract_records`, yield {name, description} dicts ready
    for JSONL output. First occurrence wins — sibling variants of the
    same family contribute the family root once, then their own name
    once, never duplicating either."""
    seen: set[str] = set()
    emitted = 0
    for path in iter_monster_files(corpus_root):
        try:
            monster = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        for record in extract_records(monster):
            key = record["name"].lower()
            if key in seen:
                continue
            seen.add(key)
            yield record
            emitted += 1
            if limit is not None and emitted >= limit:
                return
